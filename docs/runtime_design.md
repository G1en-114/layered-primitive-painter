# C++ 推理运行时设计（部署主线第一步，2026-11-16）

目标：一个独立的 C++ 程序。输入中文提示词的文字特征，输出分层图元序列和渲染好的图片。
和 Python 参照实现 `tools/ort_generate.py` 用同样的均匀随机数时，结果逐 token 一致；
个别因浮点误差产生的分叉单独统计（见下面“一致性”一节）。
后面的量化、TensorRT、CUDA 渲染和批量推理，都在这个程序上逐步替换组件。

## 已有的输入：ONNX 图

`tools/export_onnx.py` 导出 5 个图，外加 `manifest.json`（模型配置、协议、输入输出名、动态维度）。
ORT 和 PyTorch 的最大误差在 1e-5 以内。

| 图 | 运行时什么时候调用 |
|---|---|
| `embed_text` | 每个请求一次（文字前缀） |
| `embed_record` | 每生成一条 record 一次 |
| `chunk` | 前缀一次，之后每条 record 一次；KV cache 由运行时管理，图只返回本步新的 K/V |
| `decode_heads` | 每条 record 1 次（事件）；如果是图元，再加 8 次（属性） |
| `canvas_enc` | 每到一个检查点（控制记录，或每 32 个图元）一次 |

## 模块

```
runtime/
  CMakeLists.txt          链接 wheel 自带的 libonnxruntime.so.1.32.0，头文件在 third_party/
  src/json.hpp            nlohmann/json 单头文件，用来读 manifest
  src/protocol.{h,cpp}    协议：属性的分档 / 码本、事件表、图层上限
  src/grammar.{h,cpp}     lpp.sequence.Grammar 的移植：允许的事件、step、EOS 约束
  src/sampler.{h,cpp}     温度 + top-p，按外部给的均匀数做逆 CDF；CFG 合并
  src/codec.{h,cpp}       token → 图元参数（分档反量化、颜色码本 Oklab→sRGB）
  src/raster_cpu.{h,cpp}  广义高斯的参考光栅化：premultiplied over，分层状态
  src/session.{h,cpp}     ORT 会话，以及按名字调用各个图
  src/painter.{h,cpp}     解码循环：KV cache、检查点画布、记录输出
  src/main.cpp            命令行：读文字特征 / 均匀数（npy），输出序列（npy）和 PNG
```

## 先用 CPU，再逐步替换

1. **v0：ORT CPU EP + CPU 光栅化。** 和 Python 逐 token 对齐，作为正确性的基准。
2. **v1：ORT CUDA EP。** KV cache 和中间结果用 IOBinding 留在显存里，不再每步来回拷贝；
   光栅化改用 `csrc/raster` 的 CUDA kernel。
3. **v2：TensorRT EP，FP16 / INT8。** 量化校准用真实的解码激活。报告 token 一致率、PSNR 和延迟。
4. **v3：批量推理和连续批处理。** 多个请求共享一个 batch，序列结束的槽位立刻接新请求。

## 一致性怎么测

`tools/ort_generate.py` 的 `--compare` 已经给出了 Python ORT 和 PyTorch 的结果：
150 条 record 内，3 条序列里有 2 条逐 token 一致，第 3 条在第 117 条分叉。原因是约 1e-6 的数值差，
恰好让某个均匀数落在 CDF 边界的另一侧。C++ 和 Python ORT 用的是同一组图和同一种算子，
所以同样的输入下应该做到完全一致。报告的指标是：完全一致的序列比例，以及第一次分叉的位置。

## 先不做的

- **中文分词和文字塔：** v0 直接读 Python 算好的文字特征（npy）。之后把 Chinese-CLIP 文字塔
  导出成 ONNX，再用 C++ 实现 BERT 的 WordPiece 分词。
- **REDRAW / REFINE 编辑模式和图像条件：** 等 v1 稳定后再接。

## v0 实测结果（2026-11-16）

`runtime/` 编译出 `lpp_paint`：CMake 构建，链接 ORT 1.32 的 CPU EP，光栅化也在 CPU 上做。
一致性测试的流程：`tools/ort_generate.py --dump DIR` 写出文字特征和一整张均匀数表
（`[max_records, 1+A, B]`），Python 和 C++ 用的随机数完全相同。

| 对比（6 条提示词 × 300 条 record，CFG 2.0，top-p 0.9） | 完全一致的序列 | 分叉位置 |
|---|---:|---|
| Python ORT vs PyTorch | 6/6 | — |
| C++ vs Python ORT | 4/6 | 第 262、103 条（都是颜色 token，分别在检查点之后第 17、6 条） |

速度（RTX 5070 笔记本，CPU EP，4 线程，B=6，都包括画布渲染）：Python ORT 循环 40 records/s，**C++ 288 records/s**。

### 关键改动：逆 CDF 按词表顺序走

最初的采样器按概率从大到小的顺序做逆 CDF。位置这类属性有 512 档，相邻几档概率几乎一样，
约 1e-7 的数值差就能让它们在排序里换位，同一个均匀数于是落到隔好几档的 token 上
（实测 y 档 375 和 380）。

现在的做法：top-p 支撑集仍按概率排序来选，但累计 CDF 时按原来的词表顺序走支撑集里的 token。
**采样分布完全不变**，但映射对微小数值差不再敏感。改动之后，Python ORT 和 PyTorch 从“3 条里 2 条一致”
变成 6/6 全部一致。

### 剩下的分叉

C++ 逐个图元地绘制画布，参考实现则按层在对数空间里累加。两者的画布相差约 1e-7，
经过画布编码器后会让 logits 有很小的差别。现在出现的两次分叉都是颜色码本 token（4096 类），
而且都发生在检查点之后几条，和这个解释吻合，但还没有单独验证。
后面可以把参考实现的对数空间累加照搬过来，或者把画布 token 固定住再比一次，确认原因。

### CUDA EP：TF32 是分叉的主要来源（v0，CUDA EP 实测）

`--provider cuda`：通过 `OrtCUDAProviderOptionsV2` 设置 `use_tf32=0`。`--provider cuda-tf32` 则保留 ORT 的默认值，打开 TF32。
同一组 6 × 300 条 record，与 C++ CPU EP 对比：

| 提供器 | 完全一致的序列 | 速度 |
|---|---:|---:|
| CUDA，FP32（关掉 TF32） | **6/6** | 148 records/s |
| CUDA，开 TF32（ORT 默认） | 0/6，在第 2–8 条就分叉 | 150 records/s |

**结论：**

1. ORT 的 CUDA EP 默认用 TF32 做 FP32 矩阵乘，尾数只有 10 位，几条 record 之内采样就会分叉。
   做一致性对比时必须关掉。后面做 FP16 / INT8 也是同样的道理：逐 token 一致没有意义，
   要比的是分布层面的指标（图像质量、CLIP 分数、属性一致率），不能拿序列是否相同来判断。
2. 在 v0 的设计下，CUDA EP 反而比 CPU 还慢（148 对 288 records/s）。每一步都要把完整的 KV cache
   从主机内存拷到显存，约 30MB，计算本身占比很小。v1 的目标就是用 IOBinding 让 KV cache 一直留在显存里。
   另外，这次测速时本机 GPU 同时在跑动漫数据的分解。

## v1：KV cache 常驻显存（2026-11-16）

新加的 `chunk_cached` 图：输入整块 cache（`[L,B,kv,Tmax,hd]`）和写入位置 `write_pos`，在图内拼好新的 K/V，
输出更新后的整块 cache。C++ 端用 IOBinding 在显存里开两组缓冲（K、V、mask），每一步一组读、一组写，交替使用；
第一次调用（前缀）从主机端的全零张量读。cache 从此不再经过 PCIe，每步只是一次约 30MB 的显存内拷贝。
与 PyTorch 的误差 1.0e-6；`tests/test_export.py` 检查了它等于 `Chunk` 加手动写入。

**一致性：** v1 的 CPU 和 CUDA（关 TF32）都与 v0 CPU 完全一致，6/6 条序列、全部 300 条 record 相同。

**单图耗时**（`LPP_PROFILE=1`，B=6，CFG 下实际是 12 行；本机 GPU 同时在跑动漫分解，CUDA 这一列被争用拖慢）：

| 图 | CPU EP | CUDA EP（GPU 被占用） |
|---|---:|---:|
| chunk / chunk_cached | 11.1 ms/次 | 15.2 ms/次 |
| decode_heads（每步 9 次） | 0.76 ms/次 | 1.60 ms/次 |
| canvas_enc | 0.51 ms/次 | 10.2 ms/次 |
| 合计 | 302 records/s | 173 records/s |

**解读：** 20M 的模型在 batch 12、每次只算一个位置时，耗时主要来自 kernel 启动延迟：每次调用都是几百个很小的 kernel。
GPU 被另一个进程占满的时候，这些 kernel 还要排队等。所以 v1 解决的是数据搬运的问题，
CUDA 这一列的速度要等 GPU 空闲后重新测。之后的优化方向：

1. CUDA Graph：ORT 的 `enable_cuda_graph`，要求输入形状固定、绑定的缓冲区地址固定。
   v1 的静态 cache 加 ping-pong 缓冲已经很接近这个要求了。
2. TensorRT EP：把小算子融合起来，再试 FP16。
3. 把 9 次属性解码合并：每一步一次性算出全部属性的 logits，只能用于不需要按属性逐个采样的近似；
   或者把采样本身也放进图里。

## 2026-11-17：按图放置，以及 CUDA Graph 的前提条件

- **按图放置**（`--provider cuda-auto`）：B≤2 时只把主干放到 GPU（`cuda-hybrid`），否则主干和输出头放 GPU，嵌入和画布编码器放 CPU。
  B=1 时从 163 提高到 265 records/s，B=32 时 1117，和全部放 GPU 持平。详见 `docs/quant_report.md`。
- **canvas_enc 在 CUDA 上很慢的原因**：这个 ORT 1.32 的 CUDA EP 加载不了 pip 装的 cuDNN（它找的是无版本号的 `libcudnn.so`，
  补软链接后又报找不到 `cudnnCreate` 符号）。含 Conv 的图现在放在 CPU 上。
- **为什么现在还不能用 CUDA Graph**：主干 `chunk_cached` 的 857 个节点里有 241 个被 ORT 放到了 CPU 上
  （Gather 95、Div 57、Add 50、Concat 37、Slice 32、Mul 24……）。这些都是动态维度（batch、chunk 长度、cache 位置）带来的形状计算。
  ORT 的 CUDA Graph 要求所有节点都在 CUDA EP 上；这些 CPU 节点也是每次调用都要同步和拷贝内存的原因，一次主干调用要 1.3 ms，它们是主要成本之一。
- **计划**：
  1. 导出“定形”的单步主干：batch 固定为 R、T=1、cache 长度固定为 Tmax，让形状在导出时成为常量，由常量折叠消掉形状计算。
     每个常用的 R 导一个图。
  2. 检查导出的图里 CPU 节点数为 0。
  3. 运行时用 IOBinding 把所有输入放到固定的显存地址（位置、mask 这类小张量每步用 cudaMemcpyAsync 写进去）。
     KV cache 的 ping-pong 两组缓冲各捕获一张图（RunOptions 的 `gpu_graph_id` 设为 0 或 1）。
  4. 验收：生成的 token 与当前实现一致（同样的随机数），再测 B=1/8/32 的吞吐。

### 定形单步主干：实测（2026-11-17）

`lpp.export.ChunkStep`：T=1、R 和 Tmax 都是导出时的常量，新位置用 one-hot 选择写入 cache，不再用切片。
布尔型的 `Where` 在 ORT 里没有 kernel，所以 mask 用 `(hot & valid) | (~hot & mask)` 来写。
输出与 `ChunkCached` 逐位相同（最大差 0）。

| 主干单步（R=2，Tmax=1024，IOBinding，输入都在显存里） | 每次调用 |
|---|---:|
| chunk_cached（动态形状，241 个节点在 CPU 上） | 1.347 ms |
| chunk_step（定形，**全部节点都在 CUDA 上**） | 1.081 ms |
| chunk_step + CUDA Graph | 1.033 ms |

CUDA Graph 只再省了 5%。剖析一下：一步要跑 **489 个算子**，没有哪个占大头
（Mul 14%、MatMul 12%、Transpose / Split / Reshape / Concat / Unsqueeze 各约 8%）。
它们来自手写的 RoPE、GQA 的重复 KV 头、手写的注意力和 RMSNorm。
每个小 kernel 在 GPU 上也要约 2 µs，所以**瓶颈是 kernel 的个数**。CUDA Graph 只省掉了 CPU 端的启动开销。

**下一步：算子融合。** 导出时把 RoPE、注意力、归一化写成 ORT 的融合算子
（`com.microsoft` 的 `RotaryEmbedding`、`GroupQueryAttention`、`SkipSimplifiedLayerNormalization`），
目标是把一步的 kernel 数降到几十个，然后再接 CUDA Graph。验收方式不变：token 与现在一致，再测吞吐。

### 融合 RoPE（`com.microsoft::RotaryEmbedding`，2026-11-17）

`Chunk(fused_rope=True)` 把每次旋转编码导出成一个 `RotaryEmbedding` 节点（半分布局，`interleaved=0`，
cos/sin 表的形状是 `[max_pos, hd/2]`），代替原来约 8 个节点组成的链。这一步共 16 次旋转，原始节点数从 1373 降到 1045。

| 主干单步（R=2） | 每次调用 | 与动态版的最大差 |
|---|---:|---:|
| chunk_cached（动态形状） | 1.339 ms | — |
| chunk_step（定形） | 1.079 ms | 0 |
| + 融合 RoPE | 0.969 ms | 9.5e-7 |
| + 融合 RoPE + CUDA Graph | **0.935 ms** | 9.5e-7 |

最大差 9.5e-7 是 FP32 运算顺序不同带来的舍入误差。下一步是注意力：
ORT 的 GroupQueryAttention 只支持“每行连续的有效前缀”，表达不了 cache mask 里由文字前缀 padding 造成的空洞，
所以准备改用 ONNX 标准的 `Attention` 算子（opset 23）。它原生支持 GQA 和任意的布尔 mask，
能把每层的 Tile（重复 KV 头）、matmul、mask、softmax 合并成一个节点。
