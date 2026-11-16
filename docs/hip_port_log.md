# 工作记录：把光栅化算子移植到海光 DCU（HIP C++）

这份记录按时间顺序写下每一步：为什么做、怎么做、遇到什么问题、怎么解决、测到什么数。
目的是能照着复现，也能当学习材料。

## 0. 背景：为什么要移植

- 分层分解器的瓶颈是光栅化（把一串高斯图元画成图）。在 NVIDIA 5070 上我们用 **Triton** 写了 tile
  光栅化器，比 PyTorch 稠密实现快 64 倍（见 `bench/REPORT_raster.md`）。
- 租到的海光 DCU（型号 BW，64 GB）上 **没有 Triton**：DTK 26.04 的 PyTorch 2.9 不带 Triton，机器又不能
  上外网，pip 装不上。于是 DCU 上只能退回稠密实现，比本机还慢。
- DTK 自带 **HIP** 编译器 `hipcc`。HIP 是 AMD/海光的 GPU 编程接口，语法几乎和 CUDA 一样。
  所以思路是：**写一份 CUDA 风格的 C++ 源码，NVIDIA 上用 nvcc 编译，DCU 上由 PyTorch 自动
  hipify 后用 hipcc 编译**。同一份算法，两个平台，可以直接对比。

## 1. 环境摸底（先查清楚再动手）

| 项目 | DCU（SCNet notebook） | 本机 |
|---|---|---|
| 卡 | 海光 BW，64 GB，80 个计算单元（CU），`gfx936` | RTX 5070 Laptop，8 GB，36 SM，sm_120 |
| 编译器 | `hipcc`（底层是海光 dcc，基于 clang 17） | `nvcc`（/opt/cuda/bin） |
| PyTorch | 2.9.0 + DTK 26.04（`torch.version.hip = 6.3`） | 2.15 + cu132 |
| ninja | 有（/usr/local/bin/ninja） | 有 |
| **warp 宽度** | **64**（AMD 系叫 wavefront） | **32** |
| CPU 配额 | 15 核（`nproc` 显示宿主机 128 核，要看 cgroup） | 32 核 |

查法（都是一行命令）：

```sh
hipcc --version                                         # 编译器版本
python -c "import torch.utils.cpp_extension as c; print(c.ROCM_HOME)"   # PyTorch 找到的 DTK 路径
grep -n "__shfl_down" /opt/dtk/include/hip/amd_detail/amd_warp_functions.h   # wave 内 shuffle 函数
cat /sys/fs/cgroup/cpu.max                              # 容器真实 CPU 配额
```

**最关键的一条差异是 warp 宽度：NVIDIA 32，海光/AMD 64。** 凡是写死 32 的代码（比如
`__shfl_down_sync(0xffffffff, ...)` 的 32 位掩码、按 32 线程分组的归约）在 DCU 上都会算错。
所以源码里一律用内置变量 `warpSize`，不写死数字。

## 2. Triton → HIP/CUDA 的概念对应

Triton 是"以块为单位"写代码，一个 program 处理一整个向量；CUDA/HIP 是"以线程为单位"写代码。

| Triton 写法 | CUDA / HIP 写法 | 本算子里的含义 |
|---|---|---|
| `tl.program_id(0)` | `blockIdx.x` | 第几个 (图, tile) |
| `tl.arange(0, 256)` 的一个元素 | `threadIdx.x`（一个线程） | tile 里的一个像素 |
| `tl.sum(x, axis=0)` | wave 内 shuffle 归约 + 共享内存跨 wave 汇总 | 一个 tile 内所有像素对某图元梯度的求和 |
| `tl.load(ptr + j)`（标量） | 普通全局内存读（同一 wave 读同一地址会广播） | 读图元参数 |
| `constexpr` 参数 | 模板参数 / `#define` | tile 大小等 |
| Triton 自动决定寄存器/共享内存 | 自己管理 `__shared__` 和 `__syncthreads()` | 归约的中间结果 |

## 3. 先抽出公共部分，再写 kernel（2026-11-15）

`python/lpp/render/tiles.py` 放两个后端共用的 torch 操作：

1. `pack`：把位置、逆协方差、硬度、不透明度、颜色以及截断常数打包成 `[B, N, 12]`。
   这样 kernel 读到的是连续内存，不必反复计算每个图元的常数。
2. `bin_tiles`：为每个 16×16 tile 列出可能碰到它的图元。先做轴对齐包围盒剔除，再做有向包围盒
   的分离轴测试，进一步去掉细长旋转图元碰不到的 tile。这里必须保守，不能漏掉有效像素。
3. 用前缀和给保留下来的图元安排位置，保持原来的绘制顺序。alpha 合成有前后关系，不能任意重排。

Triton 只需要导入这两个函数；原生 CUDA/HIP 包装器也导入同一份。因此移植时只需对齐两个 GPU kernel，
不必重新调试分箱逻辑。

## 4. 前向、反向是怎么写的

源码：`csrc/raster/raster_ext.cu`。Python 的 autograd 包装在 `python/lpp/render/native_raster.py`。

**前向**：一个线程块处理一张图的一个 tile，256 个线程各自处理一个像素。按从后到前的顺序遍历
本 tile 的图元，计算像素不透明度 `a`，更新预乘颜色 `P = P*(1-a) + rgb*a` 和透射率 `T *= 1-a`。
输出 `(P, T)`，与参考实现的接口一致，后面仍能逐层合成。

**反向**：保存最终的 `(P, T)`，从前往后逆序遍历图元，逐步累积前景颜色与透射率，求出每个图元对
该像素的梯度。链式法则继续传到位置、逆协方差、硬度、颜色和不透明度；Python autograd 再把逆协方差
的梯度传回 `sigma` 和 `theta`。不必保存 `[B,N,H,W]` 的所有像素不透明度。

同一个图元的梯度需要把整个 tile 的贡献相加：

- 先用 shuffle 在一个 wave 内归约，步长从 `warpSize/2` 开始减半。
- 各 wave 的 lane 0 写共享内存；同步后，固定顺序把各 wave 的部分和相加。
- 用两份共享内存交替写入，避免下一次循环覆盖还没读完的结果，每个图元只需一次块同步。
- 每个 tile 的结果单独写出，最后用 torch 沿 tile 维求和；没有跨块浮点 `atomicAdd`，重复执行的
  梯度可逐位相同。代价是一个 `[B, tile数, N, 10]` 临时张量，规模大时仍要留意显存。

注意两处数学边界：超出截断半径的贡献为零；不透明度达到上限时，被 clamp 的像素不继续传不透明度梯度。
这些都要和参考实现对齐，不能只看渲染出来是否像同一张图。

## 5. 编译时遇到的问题

通过 `torch.utils.cpp_extension.load` 即时编译，无需额外安装一套 torch：NVIDIA 使用 nvcc，
DCU 上 PyTorch 自动把 CUDA 源码 hipify，再调用 hipcc。

- 本机首次编译约 **30 秒**；DCU 首次编译约 **84 秒**，两边均成功。
- 本机的 venv 路径 `~/AI infra/.venv` 含空格，torch 生成的 ninja 链接命令中 `-L` 没有正确引用路径，
  导致找不到 `c10`。包装器为 torch 的库目录建立不含空格的缓存软链接，再把链接路径交给编译器。
  没有复制 CUDA 包，也没有改动 Python 环境。
- DCU 编译缓存放在持久化目录 `/root/private_data/ext_build`，调用时设置 `LPP_EXT_BUILD`；
  代码、数据、权重也都放 `/root/private_data`，避免实例关闭后只留下系统盘缓存。

## 6. 正确性与测速结果

以下是原会话在 2026-11-15 标注全量启动**之前**测出的数据，续接时从远程
`/root/private_data/bench_hip_v1.txt` 核对。测试代码和数据未更改，不能把这张表当作与标注并行时的吞吐。

| DCU，256px | PyTorch 稠密 | HIP tile v1 | 加速 |
|---|---:|---:|---:|
| 分解单步，B=4，N=256，前向+反向 | 29.656 ms | 3.180 ms | 9.33× |
| 分解单步，B=16，N=256，前向+反向 | 117.173 ms | 4.563 ms | 25.68× |
| 整图渲染，B=4，N=960，只前向 | 38.676 ms | 2.213 ms | 17.48× |
| 整图渲染，B=16，N=960，只前向 | 157.910 ms | 4.168 ms | 37.88× |

用 GPU event 计时，预热后测 30 次，表中为中位数，完整 p10/p90 在原始记录里。
分箱、参数打包与梯度归约都包含在被测渲染调用内。这是算子测试，不代表整条分解流水线也加速同样倍数。

旧会话验证：本机 Triton + CUDA 各 15 项，共 30 项通过；DCU HIP 的 15 项通过（18.31 秒）。
覆盖空图元、非 16 整数倍画布、细小图元、padding、硬边、所有参数的梯度，以及反向重复执行的确定性。

Profiler 的一个分解单步样本：墙钟 3.67 ms，GPU kernel 时间合计 3.50 ms；反向 kernel 占 GPU 时间
49.5%，前向占 10.0%。说明后续优化应先看反向归约、临时梯度张量和分箱开销。此处只记录方向，尚未做 v2 优化。

## 7. 续接后补上的接入与验证

原会话在启动中文标注后因额度中断，`dispatch.py` 当时还只认识 `torch` 和 `triton`。
所以 HIP 虽然已经编译并验证，实际分解器仍不会自动使用它。

续接改动：

- GPU float32：优先 Triton；没有 Triton 时，尝试编译并使用原生 CUDA/HIP；编译不可用则回到参考实现。
- CPU 或其他 dtype 自动走参考实现。
- 用 `LPP_RASTER=native` 可以明确要求 HIP/CUDA；明确选择的后端出错会直接报错，便于发现编译问题。
- 新增 8 个选择逻辑测试，覆盖 DCU 无 Triton、编译不可用、CPU/双精度回退、环境变量和显式参数的优先级。
- 本机重新运行完整测试：**84 passed in 45.19s**，同时覆盖 Triton 和原生 CUDA。
- 同步接入代码后，DCU 重新运行完整测试：**69 passed in 32.84s**，原生 HIP 的 15 项均实际执行。
  额外 3 条提示来自远程已有依赖的弃用提示和未编译 memory-efficient attention，与测试失败无关。
  远程旧 `dispatch.py` 备份在 `/root/private_data/lpp-project-backups/2026-11-15/dispatch.before-codex.py`。

复现命令（在项目目录运行）：

```sh
# 本机：复用现有 uv 管理的 ML 环境
OMP_NUM_THREADS=8 "$HOME/AI infra/.venv/bin/python" -m pytest

# DCU：在远程终端运行，缓存保留在个人数据目录
cd /root/private_data/lpp
OMP_NUM_THREADS=4 LPP_EXT_BUILD=/root/private_data/ext_build python -m pytest -ra

# 只看 HIP/CUDA 算子对齐
LPP_EXT_BUILD=/root/private_data/ext_build python -m pytest tests/test_fast_raster.py -v

# GPU 空闲时测速；不要把与 VLM 标注抢卡时的数据和上表直接比较
LPP_EXT_BUILD=/root/private_data/ext_build python bench/bench_raster.py

# 后续真正运行分解时明确选用 HIP；现有输出分片会跳过
LPP_RASTER=native LPP_EXT_BUILD=/root/private_data/ext_build \
  python tools/decompose_dataset.py --data <已准备的图像分片目录> --out <新的序列目录>
```

读代码的建议顺序：`reference.py` 的 alpha 与 over 公式 → `tiles.py` 的分箱 → `raster_ext.cu`
前向 → 反向 → `native_raster.py` 的 autograd 包装 → `dispatch.py` → `test_fast_raster.py`。
先理解单像素如何合成，再看线程块和 wave 如何把很多像素的梯度汇总。

## 8. 续接时其他任务的真实状态

- 远程 small 试训已经跑完 **3000 步**，验证 loss **20.0401**。产物在
  `/root/private_data/runs/small_v0a/`：`last.pt`、`log.jsonl`、每 500 步的样图。
  这是约 6000 张分解序列的无条件试训，尚不是全量或中文条件模型。
- 全量中文标注在运行：`/root/private_data/captions.jsonl`，日志
  `/root/private_data/caption_full.log`。续接初查 1280/30000，约 3 张/秒。
  不另启第二个标注进程，也不在同一张卡上启动全量训练。
- 本机全量分解仍在运行，续接初查已进入第 012 个分片（编号 011），完整分片落在 `data/seq/v0/`。
- 后续顺序：等分解和标注完成 → 检查缺失 ID、空描述和输出截断 → 接入文本条件与训练数据 →
  上传完成的序列、启动全量训练。目前不把这些待办标为完成。

在远程终端检查进度：

```sh
tail -3 /root/private_data/caption_full.log
wc -l /root/private_data/captions.jsonl
pgrep -af '[c]aption_vlm.py'
tail -2 /root/private_data/runs/small_v0a/log.jsonl
```

## 2026-11-16：反向 kernel v2（跳过空闲 wave）在 DCU 上的实测

脚本：`bench/bench_native_versions.py`，同一份编译好的扩展、同一组张量，v1 与 v2 交替测量（AB/BA），
各 5 次预热、40 次计时。测量的是完整的一步：前向 + 反向 + MSE。batch 16，每张 256 个图元，256px。

| 设备 | v1 中位数 | v2 中位数 | 加速 |
|---|---:|---:|---:|
| 海光 BW（DTK 26.04，HIP 6.3） | 4.548 ms | 3.968 ms | **1.15×**（延迟降低 12.7%） |
| RTX 5070 Laptop（之前测的） | — | — | 反向 1.41× |

p10 到 p90 的波动小于 1%，所以这个差距是真实的，不是噪声。
DCU 上的收益明显比 5070 小。v2 省掉的是“wave 里没有任何被覆盖像素”时的归约；
猜测 DCU 的 wave 是 64 线程，空闲 wave 的比例和调度开销都和 NVIDIA 的 32 线程 warp 不同，
加上这里测的是整步时间，前向和其他小 kernel 冲淡了占比。这只是猜测，还要用 rocprof 按 kernel 拆时间来验证。

原始结果：`bench/results/native_versions_dcu_2026-11-16.json`。
