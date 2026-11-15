# 动漫数据处理记录（2026-11-15）

本次完成首批数据处理与接口验证；未启动动漫模型训练，也未把整个 6 万条数据集标为已处理。

## 来源与下载

- 数据：[huanngzh/anime_face_control_60k](https://huggingface.co/datasets/huanngzh/anime_face_control_60k)。
- 固定版本：`04e4f951a987a668cd90e43c5db1bcbf74b986c2`。
- 下载第 1/11 个 Parquet 分片：468040927 字节、5455 条记录。
- 分片 SHA256：`0bf262b05b09149559878c7fff91d490e7708f380d4051b210139cf02c8044b8`。
- 原始元数据、上游说明和分片：`data/raw/anime_face_control_60k/`。
- 实际图片字段是 **target**；`source` 是黑底关键点控制图，`visual` 是叠加关键点的预览。
  已通过三个字段的实际图片确认，训练图片没有取错为控制图。
- 上游数据卡未明确给出图片许可；保存原始来源与说明，未上传/分发任何图片。

## 处理结果

输出：`data/anime_portraits256_v1/`，约 459 MiB（包括掩码缓存与未修改的原始图片）。

| 项目 | 数量 |
|---|---:|
| 实际扫描记录 | 1139 |
| 通过头像筛选 | 1000 |
| 标签筛选排除 | 124 |
| 非单人构图排除 | 13 |
| 关键点越界排除 | 2 |
| 该批检测到的重复图 | 0 |
| 生成两层伪标签 | 1000 |
| 掩码自动检查通过 | 953 |
| 通过样本中的训练 / 验证 | 902 / 51 |
| 隔离待复核 | 47 |

1000 张初始划分为 946/54。隔离掩码可疑项后变为 902/51，保留原有划分，不重新随机分配。
47 张中均有前景面积极端提示，1 张还触发面部关键点前景分数偏低；
角色占满裁剪画面也可能触发提示，因此“待复核”不等于已经证明图片有错。

主要筛选方式：

1. 基于标签选单人构图，排除不适合当前头像试验的内容与多视图。
2. 检查 28 个关键点的形状、有限性、图内位置及脸部尺寸；不冒充已获得关键点的逐点语义定义。
3. 根据关键点包围框生成略向上偏移的方形头像裁剪，缩放至 256px，同时变换关键点。
4. 检查精确像素重复及 pHash 近重复，包含镜像；近重复还要求缩略 RGB 误差足够小。
   这不是身份识别，也不能保证发现所有变形/重绘副本。
5. 提供保守的中文标签模板描述，只含头像、发色、眼睛、表情等。全身描述留作原始元数据，
   不直接给裁剪后的头像当监督。模板没有经过 VLM 逐图复核。

## 原图与训练/验证关联

`manifest.jsonl` 记录来源、原始 ID、全局整数 ID、版本、split、原始文件校验值、裁剪框、
变换后的关键点、分片行号、原始描述与裁剪用描述。

全局 ID 使用 `200000000 + 原始数字 ID`，与现有 CelebAMask 的 0..29999 分开。
`originals/` 保存数据集提供的图片原字节（通常 512px），没有重编码；
这不代表获取了数据集加工前更高分辨率的原作。

`train_ids.json` / `val_ids.json` / `eligible_ids.json` 只含通过掩码检查的样本。
`review_ids.json` 单独列出待复核项；`mask_quality.jsonl` 保存每张的前景面积、不确定区域比例与脸部前景分数。

分解脚本新增 `--ids-file`，按原始行位置读取筛选后的 ID，不把压缩后的 ID 顺序错配给未压缩图片数组。
训练脚本新增 `--split-manifest`：显式采用数据清单中的固定划分，覆盖默认的 ID 哈希划分。

## 两层伪标签

- 模型：[skytnt/anime-seg](https://huggingface.co/skytnt/anime-seg)。
- 固定版本：`493cb60893f47441b26ec4fb9a306bce9e342982`，`isnetis.onnx`，176069933 字节。
- ONNX 输入固定为 `[1,3,1024,1024]`，RGB/255，双线性缩放；输出前景概率。
- 输出 `alpha_XXX.npy` float16 和 `labels_XXX.npy` uint8，0=背景、1=角色，阈值 0.5。
- 实际使用已有 ONNX Runtime CUDA provider，没有安装包或新建环境。
- 最初限制 1 GiB 工作区不足；调整本进程工作区上限为 2.5 GiB 后成功。
  首轮 1000 张约 78 秒，逐图缓存可断点续跑。该数字不是独占 GPU 的标准基准。

这些是**可见角色/背景伪标签**，不是人工六层标注，不包含真实的遮挡补全。
独立 `decomposition_protocol.json` 定义两个图层，背景最多 96 个图元、角色最多 768 个。
最坏上下文 901/1024，仍使用原有形状/颜色码本。此 profile 用于数据实验，不能直接替代六层模型 checkpoint 的协议。

## 图元与模型接口验证

从 953 个合格 ID 中按顺序均匀抽取 8 个，实际完成 80 步/轮的图元分解：

- 序列：`data/seq/anime_pilot_v1/`，保持原图、图层、token 与全局 ID 的关联。
- 平均重建 PSNR：**21.78 dB**，高对比细线和眼睛细节仍损失明显。
- 8 条序列通过两层 grammar 验证。
- tiny 模型使用该 profile 完成一次有限 loss 的前向和反向，验证数据接口；未进行模型训练。

因此本次保留完整原图监督和 8 张试验目标，**不把其余 945 张自动批量转成同样质量的离散教师**。
数据清洗与伪标签阶段完成，图元教师需要继续改善后再批量制作。

首个试验分片的 4 张进一步做了有原图的有界参数优化（各分支同样 80 步）：

| 方式 | 平均 PSNR |
|---|---:|
| 初始离散目标 | 21.55 dB |
| 量化感知逐图修正 | 22.66 dB |
| 连续逐图修正 | 27.37 dB |
| 连续修正后直接压回旧码本 | 22.88 dB |

指标与参数教师：`out/anime_raw_probe_000/metrics.json`、`continuous_parameters.npz`。
这是逐图优化，不是微调后模型的画质，也没有证明风格生成效果。

## 复现命令

复用现有 `~/AI infra/.venv`，没有增删包。

```sh
# 已下载分片的筛选、裁剪和清单生成；输出目录必须是新版本目录
"$HOME/AI infra/.venv/bin/python" tools/prepare_anime.py --limit 1000

# 断点续跑角色/背景伪标签；已缓存图片不会重新推理
OMP_NUM_THREADS=2 "$HOME/AI infra/.venv/bin/python" tools/segment_anime.py

# 8 张试验分解（pilot_ids.json 已记录固定选择）
OMP_NUM_THREADS=4 LPP_RASTER=native LPP_EXT_BUILD=/tmp/lpp-native-verify \
  "$HOME/AI infra/.venv/bin/python" tools/decompose_dataset.py \
  --data data/anime_portraits256_v1 --out data/seq/anime_pilot_v1 --batch 8 --steps 80 \
  --protocol data/anime_portraits256_v1/decomposition_protocol.json \
  --ids-file data/anime_portraits256_v1/pilot_ids.json

# 数据与划分性质检查：17 passed（包含原有协议测试）
OMP_NUM_THREADS=4 "$HOME/AI infra/.venv/bin/python" -m pytest tests/test_anime_data.py tests/test_protocol.py
```

另外已逐一校验 1000 个原始文件的 SHA256、全部 RGB/label/alpha 数组的形状与阈值对应关系，
以及 902/51 划分互斥、953 合格项与 47 待复核项互斥且覆盖全部 1000 张。

## 下一步

保持现有真人数据任务继续运行；先改善动漫细线的拟合、量化或连续修正表示。
质量门槛通过后，再扩大图片处理、批量图元化，并在统一图层协议下做风格微调。
