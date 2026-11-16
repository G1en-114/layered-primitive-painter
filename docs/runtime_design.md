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
