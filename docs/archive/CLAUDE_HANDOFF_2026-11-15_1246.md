# Claude 接续计划：Layered Primitive Painter

> 历史交接快照：本文记录的是12:46状态，之后已经实现v2、中文条件、画布反馈等。
> 当前接续请先阅读 [最新续接记录](../continuation_2026-11-15.md)，不要照旧状态重复启动任务。

交接时间：2026-11-15，纽约时间约 12:46。工作目录：`/home/martis/picture_train`。
这是一份执行计划；状态数字是交接快照，接手后先查日志更新，不要当作实时进度。

## 直接给 Claude 的任务

请先阅读本文件、用户提供的 AGENTS.md 规则和现有 git diff，然后按 P0 → P1 的顺序执行。
先把现有数据任务收好，并用固定样本改善图元教师质量，产出对照图和实验表；不要再次停留在方案讨论。
后续按 P2–P5 推进，依据实验结果决定是否改变表示和协议。过程中持续记录原因、命令、结果和下一步。

用户目标：约一个月内做成求职项目，面向 AI 部署工具、高性能算子岗位。
用户已将产品范围扩展为真人、风景、动物、物体的通用可编辑分层绘画；动漫作为风格后训练方向。
现阶段优先完成扎实基线，不同时重写所有模块。

## 一、真实状态

### 正在运行的任务

| 任务 | 交接快照 | 路径/说明 |
|---|---|---|
| 本机 CelebAMask-HQ 分解 | 19096/30000；19 个完整分片 | `data/decompose_v0.log`；`data/seq/v0/` |
| 远程中文标注 | 12544/30000，约 3 张/秒，日志 unparsed=0 | `/root/private_data/caption_full.log`、`captions.jsonl` |
| 远程 small 试训 | 已结束 3000 步；val loss=20.0401 | `/root/private_data/runs/small_v0a/` |
| 新模型训练 | 没有运行 | 不要把数据处理或逐图优化说成训练了新模型 |

本机正在运行的配置：`data/seq/v0/config.json`，**steps=80、seed=0、flip=false**。
脚本默认 steps=120；恢复旧任务必须显式保持 80，否则新增的配置保护会拒绝混写。
重启会跳过完整分片，当前未完成分片会重新计算。先确认任务是否仍在运行，避免启动第二份。

远程现有标注进程 PID 在交接时为 12317；不应依赖 PID 永久不变，也不要启动第二个同输出进程。
使用个人数据目录保留产物，未获新指示不要关闭正在运行的 notebook。

### SSH 与环境

已有 SSH ControlMaster socket：`/run/user/1000/scnet.sock`。只读核对已确认目标是用户 SCNet 个人数据目录。

```sh
# 避开本环境 /etc/ssh 配置权限问题，复用已有连接
ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o ConnectTimeout=12 x \
  'tail -2 /root/private_data/caption_full.log; wc -l /root/private_data/captions.jsonl'
```

`x` 在这里是已有主连接的占位主机名；socket 失效后不要假定它能直接解析或凭空重建连接。
文档不含密码；不要从旧聊天里复制密码进代码、日志或计划。

- 本机唯一 ML 环境：`/home/martis/AI infra/.venv/bin/python`，uv 管理，torch 2.15.0+cu132，RTX5070 Laptop 8GB。
- 不新建 CUDA 环境，不重复安装 torch/CUDA 包；本轮未增删包。
- 本机临时原生扩展缓存：`/tmp/lpp-native-verify`；原先持久缓存也在 `~/.cache/lpp/ext`。
- 远程是海光 BW 64GB、DTK26.04、torch2.9/HIP；无 Triton，使用原生 HIP。
- 远程持久扩展缓存：`LPP_EXT_BUILD=/root/private_data/ext_build`。
- 远程代码：`/root/private_data/lpp`。此前仅同步了 HIP 接入、选择测试与文档；**本轮新加的动漫工具和数据尚未同步**。
- 远程容器 CPU 配额约 15 核，不能按宿主机 128 核开线程；使用已有 runtime_env 配额逻辑。
- 有 50 小时卡时是用户最初提供的额度，不代表现在还剩 50 小时；不要据此承诺剩余预算。

## 二、已有实现与未实现边界

已经有：协议、码本/量化、PyTorch 参考与 Triton/CUDA/HIP 光栅化、分层分解、两级 Transformer、
训练循环、BOS 开始的参考采样器、REDRAW/REFINE 协议与训练样本、数据预处理。

尚未完整接通：

- **中文条件训练**：模型支持文本特征接口，数据集有 text_fn，但 train.py 仍未传入真实文字特征。
  caption 生成完成不等于文本训练已实现；当前 small 是无条件模型。
- **交互局部编辑**：generate() 从 BOS 开始；尚无完整的已有图元前缀编辑/恢复推理入口。
- **连续修正预测头**：没有训练；probe 只是拿着原图逐张优化。
- 通用实例图层、全局粗绘+局部续画、原图图像损失、LoRA、正式动漫后训练均未完成。
- ONNX 导出、C++ 推理、量化工具链仍属项目后半程。

## 三、git 与测试状态

HEAD：`88d507e`。工作树有跨 Claude/Codex 会话的 **已暂存、未暂存、新文件混合修改，尚未提交**。
不要 reset、clean 或覆盖这些改动，也不要把它们误认为无关用户修改。
提交仅使用用户当前 git identity，不设 author、不改身份、不加 AI co-author/Generated with 文案。
本轮只写交接文档，不代替用户启动新的 Claude 会话。

主要修改组：

1. HIP/CUDA：`csrc/raster/raster_ext.cu`、`python/lpp/render/{native_raster,tiles,dispatch,triton_raster}.py`，
   fast-raster/dispatch 测试与 benchmark；旧 test_triton_raster.py 已重命名为 test_fast_raster.py。
2. caption_vlm.py：HIP 强制 math SDPA，改成两行中文输出与单行回退，batch128、max-new-tokens110。
3. 动漫：`python/lpp/data/anime.py`、`splits.py`，`tools/prepare_anime.py`、`segment_anime.py`。
4. `tools/decompose_dataset.py` 新增 --protocol、--ids-file、输出配置一致性保护。
5. `tools/train.py` 新增 --protocol、--split-manifest，保存 protocol 元数据；**尚无 --init-weights**。
6. `tools/probe_raw_supervision.py`：有界逐图参数优化、量化对照、连续参数保存；不是训练器。

已做验证：

- HIP 接入时完整测试：本机 84 passed，远程 69 passed；这些数字早于动漫新增代码。
- 动漫/显式划分/原有协议测试：17 passed。
- 8 条动漫序列 grammar 验证，tiny 模型用两层 profile 做过一次前向/反向。
- 全部 1000 张动漫原始字节 SHA256、RGB/label/alpha 对齐及划分互斥检查通过。
- 改动文件的基础 Ruff E4/E7/E9/F 检查、git diff HEAD --check 通过。
- 没有宣称全部新增代码已重新通过本机和远程完整套件；正式提交前补适当检查。

## 四、数据与结果在哪里

### 真人

- 原始：`data/raw/CelebAMask-HQ/`。
- 256px 图片与六层标签：`data/celebamask256/`。
- 正在生成的量化图元：`data/seq/v0/`。
- 本机 JPEG 标注输入：`data/caption_images/`；VLM：`data/models/Qwen2.5-VL-7B-Instruct/`。
- 远程 VLM、图片、描述：`/root/private_data/models/`、`caption_images/`、`captions.jsonl`。
- 远程试训用约 6000 张旧子集，不是全量 30000；权重/日志/样图位于 `/root/private_data/runs/small_v0a/`。
- 本机已取回样图：`data/remote_samples/small_v0a_step3000.png`。

### 动漫：首批已完成清洗与伪标签

- 来源 `huanngzh/anime_face_control_60k`；只下载 11 个 Parquet 中的第 1 个，5455 条。
- 固定 revision：`04e4f951a987a668cd90e43c5db1bcbf74b986c2`。
- 原始：`data/raw/anime_face_control_60k/`。**target 才是插画；source 是黑底关键点图。**
- 处理后：`data/anime_portraits256_v1/`。
- 扫描1139条，接受1000；排除124条标签项、13条非单人、2条关键点越界；这批检测到重复0条。
- 生成1000张角色/背景掩码；953张自动检查通过：**902 train / 51 val**。
- 47张隔离待复核，多数因为角色几乎占满画面，不等于47张都标错。
- `manifest.jsonl`、`originals/`、images/labels/alpha_XXX.npy、eligible/train/val/review_ids.json 均已生成。
- 原图是数据集提供的通常512px插画原字节，保留校验值；不是未经上游加工的更高分辨率原作。
- 当前中文描述是头像标签模板，原有全身 prompt/BLIP 描述只作为元数据保存。
- 原始ID映射为 `200000000 + source_id`，避免和真人 ID 冲突；后续来源也需独立命名空间。
- pHash + RGB 校验含镜像近重复检查，不能声称已穷尽所有重复；扩充时需要跨分片统一去重。
- 分割模型：`data/models/anime-seg/isnetis.onnx`，固定 revision
  `493cb60893f47441b26ec4fb9a306bce9e342982`，输入1024px，CUDA工作区上限2.5GiB；全批约78秒。
- `decomposition_protocol.json` 是独立的**两层实验 profile**，不是人工六层、不含真实 amodal 标注。
  它与现有六层 checkpoint 的事件/embedding 尺寸不兼容，不能直接混合或 strict-load。
- `data/seq/anime_pilot_v1/` 只有8张分解试验，平均21.78dB；其余945张尚未批量图元化。

### 诊断实验：不能当成生成模型的结果

| 样本 | 初始目标 | 80步量化感知修正 | 80步有界连续修正 |
|---|---:|---:|---:|
| 真人8张 | 26.18dB | 27.73dB | 31.48dB |
| 动漫4张 | 21.55dB | 22.66dB | 27.37dB |

两分支从相同初始参数独立开始；保持图元数、图层、顺序和 beta 不变。连续分支拿到了目标原图。
修正范围：xy±2像素、log_sigma±0.25、theta±0.25rad、RGB/alpha±0.10。
这些结果提示拟合/量化值得改善，样本太小，不能说全数据或自由生成将获得同样增益。

- 真人：`out/raw_supervision_probe/metrics.json`、`comparison_compact.png`。
- 动漫：`out/anime_raw_probe_000/metrics.json`、`continuous_parameters.npz`、`comparison.png`。
- 8张动漫原图/标签/重建：`data/anime_portraits256_v1/decomposition_contact_sheet.jpg`。

## 五、执行计划与验收

### P0：先收好正在跑的基线

1. 更新日志进度，确认进程状态；不重复跑、不无故终止。
2. 分解结束后核对30个完整分片、30000个唯一ID、offset/token/layer/round一致性和数值有限性。
3. 标注结束后核对30000个唯一ID、缺失/重复/空描述、输出截断和单行回退的比例。
   日志 unparsed=0 只代表解析成功，不代表内容完整或正确；抽查不同构图与长描述。
4. 回收远程 captions 至本机，保留原件与校验值。传输中断使用临时文件续传/原子完成，不覆盖唯一副本。
5. 保留small_v0a权重、日志和固定样图。准备全量无条件small基线；文本条件另接真实文字特征，
   不用随机特征假装已经接通。数据与配置版本必须写进产物。

验收：一份清楚的数据审计表、可恢复路径、完整基线产物；不重新下载/重做已有正确数据。

### P1：立即做固定样本的图元教师改进（当前最优先开发）

先从合格动漫样本中按既有split固定64张train/16张val，另从真人固定32张对照。
保存ID清单，不按好看的结果挑图；参数在train上选，val作独立判断。

按一次只改一个因素的顺序比较：

1. 当前80步基线及增加优化步数的控制组，避免把更多计算误认为新算法收益。
2. 分层拟合结束后，全图联合的**量化感知**微调，约束越界与图层外观。
3. 固定几何/alpha时的颜色拟合，再与几何优化交替；先推导适用于有序高斯合成的公式，
   不能照搬POP2固定alpha的颜色求解。比较RGB误差与Oklab码本选择标准。
4. 边缘/关键区域损失与预算分配：动漫细线、眼睛、轮廓优先；增加图元数时单独记为预算对照。
5. 连续小残差与低比特残差实验，评估能保留多少连续优化收益，以及额外字节数和延迟。

输出每图指标与同图对照：PSNR、结构/边缘指标、图层越界、图元数、时间、峰值显存、记录大小。
不要求为了一个数字牺牲层的可编辑性。建议先以独立val平均至少约+1dB、没有明显视觉/越界退化为
纯离散方案的采用起点；这是实验选择门槛，不是已保证的结果。未达标时说明失败原因并换路线。

验收：实验脚本、固定ID、配置、完整指标、代表性及失败样图、采用/不采用的明确结论。
先完成这些，再决定是否批量图元化剩余动漫样本，避免大量生产低质量教师。

### P2：若残差方案胜出，接入原图监督的小修正头

1. 离散事件和属性仍做交叉熵；新头从模型隐状态和当前图元属性预测有界连续修正。
2. 使用配对原图的teacher-forced重建/局部块监督，先验证冻结主干只训修正头，再逐步解冻。
3. 图像损失可回传到修正头及共享表示；argmax/采样仍不直接可微。
   不以“可微渲染器已经有了”代替完整梯度路径验证。
4. 推理只给文字时，修正头不能依赖训练时才有的目标图特征。
5. 原图重建任务与自由生成任务分开评估；不要拿任意生成图与任意原图计算逐像素监督。
6. 形状残差优先研究唯一的log协方差表示，检查近圆形处梯度；不默认把多峰token分布取均值。
7. 若输出格式变化，做新协议版本，明确旧权重/数据兼容与迁移，不覆盖旧码本或重解释旧token。

验收：未见图重建、固定种子自由生成、延迟/大小对照，确认真实学习收益而非逐图优化收益。

### P3：通用图像与局部编辑的最小闭环

1. 层槽位/实例ID、语义类别、合成顺序分开表示；先背景+少量实例，不一次加入复杂嵌套层级。
2. 真人六层和动漫两层先独立保存，制定统一映射及checkpoint迁移后才混合训练。
3. 风景/动物/物体各约500张工程样本：优先有现成分割的COCO Panoptic/ADE20K；
   保留来源、许可、去重分组、固定划分。此批尚未下载，不要报告已在训练。
4. 先实现“指定图层/区域的图元前缀续画”：加载已存在前缀、恢复grammar、重建KV缓存，输出合法序列。
5. 区域限制作用于图元中心时，图元边缘仍可越界；训练和推理要使用一致规则并检测区域外变化。
6. 扰动少量图元/移除细节后训练修复，配合REDRAW/REFINE；固定样例检查保留区域是否稳定。
7. 引入全局粗绘+局部细化前，先验证短段续画。完整历史外部保存，缓存仅复用完全相同的有序前缀。

验收：照片、简单风景、动物、物体各有可编辑样例；能改一个区域，并说明区域外变化和未知遮挡部分。
单图无法提供被遮挡内容的真实监督；先做可见层编辑，补全内容需明确是推测。

### P4：规模训练与动漫风格后训练

1. 在稳定数据/协议上完成small全量基线，接入中文条件；再与base约123M比较，不直接跳1B。
2. 训练数据质量通过后，逐批扩充动漫与其他类别，不先把数量堆到最大。
3. 同图先试2–4条轨迹、少量描述版本；所有轨迹/翻转/裁剪保持同一split。
4. 正式后训练从基础权重初始化，**新建优化器和学习率调度**。补 --init-weights 或等效入口，
   不把 --resume 当风格微调：现有 --resume 会恢复旧优化器及step。
5. small先比较全参微调，多个可切换风格包再做LoRA；需要动漫数据，不能指望只换提示词就学会。
6. 若保留通用能力，加入原数据回放和风格条件，分别评估风格、内容、坏图比例、多样性和遗忘。

### P5：回到求职目标的部署与算子交付

正式模型接口稳定后，推进ONNX子图、ORT/C++推理、量化与精度回归、性能表。
保留PyTorch oracle与跨后端一致性，分别报告CUDA/HIP算子、端到端延迟、显存、质量和硬件条件。
避免项目全部时间耗在扩大训练，最后没有部署工具与C++成果。

## 六、常用验证与参考

```sh
cd /home/martis/picture_train
tail -3 data/decompose_v0.log

# 小检查；完整测试在适当GPU资源下运行，不要与大任务挤到显存溢出
OMP_NUM_THREADS=4 "/home/martis/AI infra/.venv/bin/python" -m pytest tests/test_anime_data.py tests/test_protocol.py
git diff HEAD --check

# 原图监督诊断：会逐图优化，不是训练生成模型
OMP_NUM_THREADS=4 LPP_EXT_BUILD=/tmp/lpp-native-verify \
  "/home/martis/AI infra/.venv/bin/python" tools/probe_raw_supervision.py \
  --prepared data/anime_portraits256_v1 --sequences data/seq/anime_pilot_v1 \
  --protocol data/anime_portraits256_v1/decomposition_protocol.json \
  --shard 0 --count 4 --steps 80 --quantized-control --out out/anime_raw_probe_recheck
```

已有学习记录：

- `docs/hip_port_log.md`：原生光栅化接入与性能。
- `docs/anime_data_log.md`：本轮数据来源、清洗、掩码、验证与限制。
- `docs/general_painter_design.md`：原图监督、残差、通用图层设计。
- `docs/data_expansion_plan.md`：候选来源与分阶段扩充。
- `docs/pop2_comparison.md`：参考POP2，重点是多轨迹、空间组织、编辑前缀与颜色拟合。

参考仓库：https://github.com/wonderfulearth/primitive-operation-painter 。
POP2模型看 `model/`，数据工具看 `data_tools/image_to_sequence/`，原生编辑看 `app/src/`；
根目录部分Python文件为旧版保留，别把POP1当POP2。此仓库并非本项目的直接基础分支。

## 七、工作纪律

- 继续遵守用户AGENTS.md：复用uv环境；系统配置/包服务变更需按用户规则备份、记系统日志和快照。
  纯项目代码、数据产物不是系统配置改动，不因此引入多余的系统操作。
- `data/`、`out/` 已被忽略；不要把原始图片、大权重或凭据推到公开仓库。
- 不把自动检查通过写成人工标注；不把逐图优化写成生成效果；不把候选计划写成已运行任务。
- 每个阶段提交可审阅的结果并更新实验记录。仅有新的数据/代码变化、失败或明确疑点时扩展测试，
  不反复跑已通过且未受影响的重测试。
- 首先自主执行P0/P1，普通可逆实现选择无需反复询问用户；碰到真正缺少的信息或不可替代的权限再说明。
