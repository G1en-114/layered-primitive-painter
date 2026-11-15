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
- [ ] 分层分解器（图片 + 分割 → 分层图元序列）
- [ ] Triton / CUDA 光栅化算子
- [ ] 模型与训练
- [ ] ONNX 导出、C++ 运行时、量化工具、基准测试

## 目录

```
protocol.json          序列协议：画布、图元公式、属性词表、事件、图层
python/lpp/
  protocol.py          协议加载与校验
  color.py  shape.py   Oklab 颜色空间；形状的 log 协方差参数化
  codebook.py codec.py k-means 码本；连续参数 <-> token
  render/reference.py  稠密参考光栅化器（正确性基准）
assets/codebooks/      先验码本（数据驱动码本会覆盖它们）
tests/                 性质测试：结合律、唯一性、幂等性、梯度
tools/                 码本构建、渲染 demo
```

## 运行

```sh
python -m pytest
python tools/build_prior_codebooks.py
python tools/render_demo.py --out out/demo
```

需要 PyTorch（CUDA 可选）、numpy、pillow、pytest。

## 许可

代码采用 MIT 许可。训练数据（CelebAMask-HQ）仅限非商业研究用途，不随仓库分发。
