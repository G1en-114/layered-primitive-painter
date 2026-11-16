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
