# 量化报告：FP16 / INT8 / 仅权重 INT4·INT8（2026-11-17）

模型：GRPO-attr2 step 100（small，20.4M 参数），由 `tools/export_onnx.py` 导出为 6 个 ONNX 图，各图与 PyTorch 的误差都在 1e-5 以内。
量化工具：`tools/quantize_onnx.py`；质量评估：`tools/quant_eval.py`；速度测试：`bench/runtime_quant_repeat.sh`（C++ 运行时，每个配置跑 3 次取中位数）。
硬件：AMD Ryzen 9 9955HX（16 核，支持 AVX-512 VNNI），RTX 5070 Laptop；ONNX Runtime 1.32。

## 变体

| 变体 | 做法 | 主干 chunk 的大小 | 输出头大小 |
|---|---|---:|---:|
| FP32 | 原始导出 | 53.0 MB | 16.2 MB |
| FP16 | 权重和计算都用 float16，输入输出保持 float32 | 26.7 MB | 8.1 MB |
| INT8（动态） | MatMul 权重按通道量化为 int8，激活每次调用时动态量化；注意力里“激活×激活”的 MatMul 不量化 | 13.8 MB | 7.3 MB |
| INT8（仅权重） | MatMulNBits，每 32 个权重一块，对称 int8；激活保持浮点 | 15.2 MB | 11.8 MB |
| INT4（仅权重） | 同上，int4 | 8.7 MB | 11.0 MB |

`embed_text` 和 `canvas_enc` 这两个图很小，对精度也敏感，所以保持 FP32。

## 质量怎么衡量

量化后，采样在第 2–3 步就会和 FP32 走上不同的分支，之后整张图都不一样了（两者之间的 PSNR 只有 12–14 dB）。
所以**逐 token 对比、逐张图对比都没有意义**。这里分两层来验收：

1. **单步数值保真度**：在 FP32 生成的过程中，记录每个图实际收到的输入（每个图 40 组），再把同样的输入喂给量化后的图，比较输出。
   输出头看两个 softmax 分布之间的 KL 散度和 top-1 是否一致；其它张量看相对 L2 误差。每一步单独比较，误差不会沿序列累积。
2. **生成分布**：48 条验证集提示词、CFG 2.0、完整长度（约 735 个记录）；比较 CLIP 分数、平均长度、提前结束的次数。

| 变体 | 输出头 KL | top-1 一致率 | 主干相对误差 | CLIP 分数 | 平均长度 |
|---|---:|---:|---:|---:|---:|
| FP32（参照） | — | — | — | 43.14 | 734 |
| FP16 | 2.1e-5 | 99.1% | 0.10% | 43.34 | 732 |
| INT8 仅权重 | 3.8e-5 | 98.2% | 0.45% | 43.04 | 729 |
| INT8 动态 | 1.8e-3 | 87.9% | 1.73% | 43.07 | 735 |
| INT4 仅权重 | 7.6e-3 | 79.3% | 7.42% | 43.18 | 732 |

- 所有变体的 CLIP 分数差别都在 ±0.3 以内。48 个样本下这在采样噪声范围内，这一层看不出质量下降。
- 单步保真度把各变体清楚地拉开了：**FP16 和 INT8 仅权重几乎无损；INT8 动态误差大约是它们的 50 倍；INT4 每 5 步就有 1 步选出的 top-1 不一样**。
  分布指标暂时没变差，但用到更长的序列或更大的模型时，INT4 的风险最大。
- 另外看到一点：FP32 在 CUDA 上和在 CPU 上的单步输出完全一致（TF32 已关闭），但整条序列在第 218 个记录左右（中位数）开始分叉，来源是浮点误差的累积。

## 速度（C++ 运行时，records/s，3 次取中位数，每个样本最多 400 个记录）

| 设备 | 精度 | B=1 | B=8 | B=32 |
|---|---|---:|---:|---:|
| CPU | FP32 | 162 | 262 | 192 |
| CPU | **INT8 动态** | **264（+63%）** | **323（+23%）** | **226（+18%）** |
| CPU | INT8 仅权重 | 77（−52%） | 211 | 196 |
| CPU | INT4 仅权重 | 227（+40%） | 267 | 205 |
| CUDA | FP32 | 162 | 731 | 1110 |
| CUDA | **FP16** | 150（−7%） | 769（+5%） | **1275（+15%）** |
| CUDA | INT8 仅权重 | 158 | 698 | 1090 |
| CUDA | INT4 仅权重 | 156 | 680 | 1069 |

3 次运行之间的差异都在 5% 以内。

## 结论

1. **GPU 上部署选 FP16**：质量几乎无损，batch 32 时快 15%。
   batch 为 1 时 GPU 只有 162 records/s，和 CPU 一样；量化也帮不上忙，因为这时每一步都在等 kernel 启动，而不是在等计算或显存。
   下一步应该用 CUDA Graph 消除启动开销，或者做算子融合。
2. **CPU 上部署选动态 INT8**：单张图生成快 63%。这台 CPU 有 AVX-512 VNNI，int8 点积指令直接生效。
   代价是单步保真度比 FP16 差约 50 倍，但在分布层面看不出差别。
3. **仅权重量化在这个模型上不划算**：
   - 在 GPU 上，所有 batch 都比 FP32 慢 2–6%。模型很小（主干 53 MB），一步解码受启动延迟限制，不受显存带宽限制，在线反量化反而多做了工作。
   - 在 CPU 上，INT4 单张图快 40%，但它的质量在所有变体里最差；INT8 仅权重在 B=1 时只有 77（推测是 ORT 1.32 的 8 bit MatMulNBits 在 CPU 上没有 4 bit 那样优化过的内核，**待用 `LPP_PROFILE` 剖析确认**）。
   - 等模型放大到 base（126M）、以后到 1B，解码会变成受显存带宽限制，仅权重量化才会有收益。到时候用同一套工具重新测一遍。
4. **方法上的教训**：
   - Python 解码循环会扭曲结论：在 Python 里测，INT8 反而慢 10%；用 C++ 测，B=1 时快 63%。
   - 量化之后不能要求逐 token 一致，要用“单步保真度 + 分布指标”两层来验收。
   - 每个配置跑 3 次取中位数，单次测量的噪声可能超过量化带来的差别。

## 后续：按图放置到 CPU 或 GPU（比量化更有效）

剖析 B=1 时 CUDA 上的耗时（`LPP_PROFILE=1`，400 个记录）：`decode_heads` 占 60%（每个记录 9 次调用，每次 0.40 ms），
主干 `chunk_cached` 占 22%（每次 1.30 ms），`canvas_enc` 每次 21.5 ms。同一组图放在 CPU 上：`decode_heads` 每次 0.22 ms，
`embed_record` 0.08 ms，`canvas_enc` 0.42 ms，只有主干更慢（3.3 ms）。
**小图都是 CPU 更快，只有大的主干图 GPU 更快。** 主干输出的 `h` 本来就要拷回 CPU，所以把小图挪到 CPU 不会增加额外的数据传输。

C++ 运行时新增了几种放置方式（`--provider`）：

| 放置方式 | 哪些图在 GPU 上 | B=1 | B=8 | B=32 |
|---|---|---:|---:|---:|
| cpu | 无 | 185 | 287 | 205 |
| cuda-all（原来的做法） | 全部 | 163 | 735 | 1114 |
| cuda | 主干、输出头 | 174 | 762 | 1114 |
| cuda-hybrid | 只有主干 | 266 | 593 | 663 |
| **cuda-auto** | B≤2 用 hybrid，否则用 cuda | **265** | **765** | **1117** |

单位 records/s，3 次的中位数。cuda-auto 在 B=1 时比原来全放 GPU 快 63%，比纯 CPU 快 43%。
一致性：B=8、400 个记录内，cuda-hybrid 和 cuda-all 与纯 CPU 的 8 条序列完全相同；cuda 有 1 条在第 254 个记录处分叉（浮点误差的累积）。
`canvas_enc` 和 `embed_text` 在 CUDA EP 上每次都要约 19 ms，把 cuDNN 卷积算法搜索改成 HEURISTIC 后也没有变化，原因还不清楚。现在这两个图放到 CPU 上，问题就绕过去了。

## 复现

```bash
python tools/export_onnx.py --ckpt out/ckpt/grpo_attr2_step100.pt --out out/onnx/grpo_attr2_s100
for m in fp16 int8 int8w int4w; do python tools/quantize_onnx.py --src out/onnx/grpo_attr2_s100 --out out/onnx/grpo_attr2_s100_$m --mode $m; done
python tools/quant_eval.py --text-encoder data/models/chinese-clip-vit-base-patch16 --captions data/captions/captions_clean_v1.jsonl \
  --split-manifest data/celebamask256_split_v1.jsonl --n 48 --capture 40 --out out/quant_eval/x \
  --variant fp32-cpu=out/onnx/grpo_attr2_s100:CPU --variant int8-cpu=out/onnx/grpo_attr2_s100_int8:CPU \
  --variant fp32-cuda=out/onnx/grpo_attr2_s100:CUDA --variant fp16-cuda=out/onnx/grpo_attr2_s100_fp16:CUDA
bash bench/runtime_quant_repeat.sh out/onnx/grpo_attr2_s100 out/runtime_bench
```

## 下一步

- CUDA Graph：针对 B=1 时 GPU 受启动开销限制的问题。
- 安装 TensorRT，和 ORT CUDA EP 对比。
- CPU 上的静态 INT8（QDQ 加校准），省掉动态量化每一步的开销。
- 剖析 INT8 仅权重在 B=1 时变慢的原因。
- base_v3 训练完成后，用同一套工具重测，验证“模型变大后仅权重量化开始有收益”这个判断。
