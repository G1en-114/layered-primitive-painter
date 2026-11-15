# Layered Primitive Painter

一个**用可编辑图元一笔一笔画画、并且天然分图层**的自回归图像模型，以及配套的**部署、量化与算子工具链**。

> 思路受 [Primitive Operation Painter 2](https://github.com/wonderfulearth/primitive-operation-painter)（MIT）启发：
> 图像不是像素，而是一串可以查看、修改、继续生成的绘画操作。本项目从头实现，
> 图元换成广义 2D 高斯，加入语义图层，模型结构、数据流水线和推理运行时均为独立设计。

## 与 POP 2 的主要区别

| | POP 2 | 本项目 |
|---|---|---|
| 图元 | 矩形 / 椭圆 | 广义 2D 高斯：边缘硬度 β 统一软高斯和实心椭圆 |
| 组织方式 | 全局 → 4×4 → 8×8 区域 | **语义图层**（背景 / 身体 / 皮肤 / 五官 / 头发 / 饰品），每层内由粗到细 |
| 属性编码 | 9 个独立属性 | 6 步：位置 x, y + 形状码本（log 协方差）+ 颜色码本（Oklab）+ 不透明度 |
| 图元内属性依赖 | 多分支 FlexAttention | 两级 Transformer：主干每图元一位置 + 小型属性解码器 |
| 图层编辑 | — | `REDRAW_k` 单独重画一层、`REFINE_k` 给一层补细节，分层导出 ORA / SVG |
| 数据分解 | GPU 随机搜索 | 可微渲染 + 梯度优化，量化感知拟合 |
| 推理 | libtorch 原生引擎 | ONNX → ONNX Runtime / TensorRT / CPU，自研 C++ 运行时、量化工具与算子库 |

## 当前状态

- [x] 序列协议 `protocol.json`（Python 与 C++ 共用的唯一定义）
- [x] 颜色（Oklab）、形状（log 协方差）参数化与 k-means 码本
- [x] PyTorch 参考光栅化器：可微、分图层渲染、精确分层导出
- [x] 编解码与量化感知 snap（直通估计器）
- [x] 分层分解器（图片 + 分割 → 分层图元序列）：量化感知拟合、amodal 补全、图层越界约束
- [x] 协议支持文本前缀与 `REDRAW_k` / `REFINE_k`
- [x] Triton tile 光栅化算子（前向 + 反向，确定性梯度）：分解器单步快 64×，端到端 19.8 s → 0.8 s/图，见 [bench/REPORT_raster.md](bench/REPORT_raster.md)
- [x] 原生 CUDA/HIP C++ tile 光栅化（前向/反向）：同一源码适配 NVIDIA 与海光 DCU，DCU 算子基准快 9–38×，见 [移植学习记录](docs/hip_port_log.md)
- [ ] CPU SIMD 光栅化（部署用）
- [x] CelebAMask-HQ 数据处理：18 类合并为 6 个按深度排序的图层，3 万张 256px 分片
- [x] 语法状态机、full / redraw / refine 序列、批量分解流水线（3 万张，本机 5070 约 4 小时）
- [x] 两级模型（tiny 1.7M / small 22M / base 123M）、带 KV cache 的参考采样器、训练循环
- [x] 海光 DCU（BW，DTK 26.04）上训练：测试 46 通过（无 Triton 的 14 个跳过），small 模型 3.6 分钟/轮
- [ ] 中文标注（Qwen2.5-VL-7B，离线在 DCU 上运行）与文本条件
- [ ] small / base 在全量数据上训练，规模对比
- [ ] ONNX 导出、C++ 运行时、量化工具、基准测试
- [ ] 二次元风格后训练（见下）

二次元数据已完成首批 1000 张头像裁剪、原图关联与角色/背景伪标签，953 张通过自动检查；
8 张已做图元分解试验，尚未启动风格训练。见 [数据处理记录](docs/anime_data_log.md)。

## 计划：二次元风格后训练

主模型在 CelebAMask-HQ（真人、精确分层标注）上预训练，再用动漫人脸微调出二次元风格，
对比同一模型微调前后的变化。

- 图像与描述：[huanngzh/anime_face_control_60k](https://huggingface.co/datasets/huanngzh/anime_face_control_60k)
  （6 万张动漫人脸，含文字描述与面部关键点，5.4 GB）；分辨率与关键点质量待确认。
- 分层伪标签（无现成的动漫人脸逐像素分层标注）：角色 / 背景用
  [skytnt/anime-segmentation](https://github.com/SkyTNT/anime-segmentation) 的分割模型；
  脸部轮廓关键点 → 皮肤层，眼睛 / 嘴巴关键点 → 五官层，角色其余部分按位置分为头发与身体。
- 精确的角色 / 背景两层数据：[skytnt/anime-segmentation 数据集](https://huggingface.co/datasets/skytnt/anime-segmentation)
  的透明背景角色 PNG 合成到背景图上（被遮挡的背景也是真实的）。
- 版权：上述图片多抓取自 getchu / danbooru，仅作非商业研究用途；仓库不分发图片，
  发布权重时注明数据来源与限制。

## 目录

```
protocol.json          序列协议：画布、图元公式、属性词表、事件、图层
python/lpp/
  protocol.py          协议加载与校验
  color.py  shape.py   Oklab 颜色空间；形状的 log 协方差参数化
  codebook.py codec.py k-means 码本；连续参数 <-> token
  render/reference.py  稠密参考光栅化器（正确性基准）
  render/triton_raster.py  tile 光栅化：前向/反向 Triton kernel
  render/native_raster.py  原生 CUDA/HIP 包装器（首次调用即时编译）
  render/tiles.py      Triton 与 CUDA/HIP 共用的参数打包、tile 分箱
  render/dispatch.py   后端选择（GPU float32 优先 Triton，再尝试 CUDA/HIP；LPP_RASTER 可强制）
  decompose/           分层分解器：图片 + 图层标签 -> 有序、量化后的图元
  synthetic.py         已知真值的合成分层场景（测试与 demo 用）
assets/codebooks/      先验码本（数据驱动码本会覆盖它们）
tests/                 性质测试：结合律、唯一性、幂等性、梯度
tools/                 码本构建、渲染与分解 demo
bench/                 算子基准测试与优化报告
```

## 运行

```sh
python -m pytest
python tools/build_prior_codebooks.py
python tools/render_demo.py --out out/demo
python tools/decompose_demo.py --out out/decompose   # 合成场景分解报告
```

需要 PyTorch（CUDA 可选）、numpy、pillow、pytest。

## 许可

代码采用 MIT 许可。训练数据（CelebAMask-HQ）仅限非商业研究用途，不随仓库分发。
