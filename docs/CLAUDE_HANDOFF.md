# Claude 最新交接：v2 已正式开训，继续数据生成与训练验证

状态核查：2026-11-15 **纽约时间17:27左右**（UTC22:27），工作目录 `/home/martis/picture_train`。
接续 Claude 会话 `4612f68c-db51-4649-9c22-cc88ed5b4fc5`。

**先查实际进程和日志，保持已运行的训练与数据任务继续执行。当前已经正式开训，不要重新启动另一份。**
本文数字及PID都是核查快照；不要把静态数字当实时状态。

## 1. 接手后立即做什么

1. 阅读用户的AGENTS规则、本文与 `git status`。代码截至 `00af76f` 已push到origin/main；
   本交接会另行提交，实际HEAD以git log为准。详细过程见 `docs/continuation_2026-11-15.md`。
2. 核实远程训练仍在推进，优先关注第500步检查点与第1000步验证/样图。
3. 保持本地轨迹队列、远程三项分解和跨机器同步继续运行，检查错误和产物完整性。
4. 四轨迹完成后检查清理结果，为下一阶段多轨迹训练准备数据；与当前t0阶段分开记录。

用户之前询问本地GPU服务，只要求查看，**没有要求停掉本地任务**。

## 2. 正式训练已经运行

- 启动时间：2026-11-15纽约时间 **17:24:23**。
- 远程训练PID快照：**29620**，已核实存活且实际输出训练指标。
- 运行目录：`/root/private_data/runs/small_v2_t0_text_canvas/`。
- 最新已核实：**step200 / 8000，loss36.2179，gnorm4.27**。
- step50/100/150的loss分别39.8792 / 38.2527 / 37.0078，梯度有限。
- 当前约26080 records/s；这是图元序列记录吞吐，不是图片/秒。
- 第500步才首次保存 `last.pt`；本次step200检查时尚无检查点，属于预期。
- 第1000步才进行首轮验证、采样并生成 `samples_001000.png`。目前不能评价生成质量或收敛。

实际配置：

| 项目 | 配置 |
|---|---|
| 模型 | v2 small，约20.41M参数，从头初始化；未加载v1权重 |
| 数据 | t0完整30000张清理序列，29400 train / 600 val |
| 序列协议 | v2，9步：事件+8项属性 |
| 条件 | 冻结Chinese-CLIP；中文长/短描述、随机属性模板；text dropout0.1 |
| 画布与扰动 | canvas K32、64px；历史扰动概率0.3 |
| 训练 | batch32，8000步，seed0，workers2，OMP/MKL各4 |
| 优化 | AdamW，lr3e-4、min_lr3e-5、warmup500、wd0.1、梯度裁剪1.0 |
| 采样/保存 | CFG2.0；每500步保存，每1000步验证与采样 |
| 后端 | LPP_RASTER=native，LPP_EXT_BUILD=/root/private_data/ext_build_v2，SDPA math |

重要文件：

- 真实训练指标：`/root/private_data/runs/small_v2_t0_text_canvas/log.jsonl`。
- stdout和错误：同目录 `train.stdout.log`。
- 启动时间、真实argv、PID、清理统计：同目录 `launch.json`。
- 本机启动计划：`/home/martis/picture_train/data/launch_small_v2_t0.json`。
- 本机启动结果：同名 `.state.json`，当前阶段 `training_launched`；同名 `.log`记录全过程。
- 启动器已经完成并退出，不需要重启它。它对重复启动和已有训练目录有保护。

远程读取命令（已有SSH主连接；x是占位名，不能假定DNS可解析）：

```sh
ssh -F /dev/null -S /run/user/1000/scnet.sock -o BatchMode=yes -o ConnectTimeout=12 -o LogLevel=ERROR x \
  'tail -n 4 /root/private_data/runs/small_v2_t0_text_canvas/log.jsonl; tail -n 8 /root/private_data/runs/small_v2_t0_text_canvas/train.stdout.log; ps -p 29620 -o pid,etime,stat,args'
```

连接失效时先确认socket/主连接，不要从旧聊天复制凭据到文件，也不要凭空重建未知端点。

## 3. 本地GPU现在在做什么

17:27核查：RTX5070 Laptop 8GB，GPU利用率100%，总显存1902MiB。

| 本机进程快照 | 工作 | GPU占用 |
|---|---|---|
| 444466 | t1偶数分片分解，seed1，batch32，steps80，part0/2 | 约1840MiB |
| 444424 | 本机有限轨迹队列，等待t1后运行t2、t3 | 未列入GPU计算进程 |
| 457459 | 远程分片汇集、SHA校验及后续清理控制器 | 当前未列入GPU计算进程 |
| kwin_wayland，PID1484 | 桌面合成器 | 约7MiB |

没有发现额外的本地模型推理服务或本地正式训练进程。
先前t0清理PID464967、启动器PID467334已经完成，不应把它们当作仍运行的GPU任务。

```sh
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
cat /home/martis/picture_train/data/trajectory_plan_local_v2.state.json
tail -n 3 /home/martis/picture_train/data/decompose_v2_t1_local.log
```

## 4. 四轨迹进度与控制器

同一批30000张原图的四种轨迹，共120000条序列，**不是120000张独立图片**。
每条本机负责15000张偶数分片，远程负责15000张奇数分片。
以下图片数包含未落盘分片中的已完成批次：

| 轨迹 | seed / flip | 本机 | 远程 | 原始目录后缀 |
|---|---|---:|---:|---|
| t0 | 0 / false | 15000，完成 | 15000，完成 | seq/v2 |
| t1 | 1 / false | 12256 | 9032 | seq/v2_t1 |
| t2 | 2 / true | 等待 | 3032 | seq/v2_t2 |
| t3 | 3 / true | 等待 | 2352 | seq/v2_t3 |

远程t1/t2/t3实际并行运行，PID快照22694 / 25039 / 25954。
远程t0的18481已正常结束。控制器25955依次加入既有进程等待和验证，不再派生相同输出任务。
不要看到控制器只写“waiting t1”就误认为t2/t3没运行。

本机文件：

- 计划 `/home/martis/picture_train/data/trajectory_plan_local_v2.json`。
- 状态同名 `.state.json`、队列日志同名 `.log`。
- 各轨迹日志：`/home/martis/picture_train/data/decompose_v2_local.log`及`decompose_v2_t1_local.log`、
  `decompose_v2_t2_local.log`、`decompose_v2_t3_local.log`。
- 同步状态 `/home/martis/picture_train/data/trajectory_plan_local_v2.sync.json`，日志同名 `.sync.log`。

远程文件：

- 计划 `/root/private_data/trajectory_plan_dcu_v2.json`，状态同名 `.state.json`，日志同名 `.log`。
- 分解日志 `/root/private_data/decompose_v2_dcu.log`及`t1/t2/t3`对应的`decompose_v2_tN_dcu.log`。
- 新代码目录 `/root/private_data/lpp_next`，后续任务使用该目录。
- 老t0记录的协议位于 `/root/private_data/lpp/protocol.json`；与lpp_next协议内容哈希相同。

本机同步器每60秒收集已原子落盘的远程分片并核对SHA。
当前阶段collecting，无报错；t0已验证30000张，其他原始完整分片在持续汇集。
四条原始轨迹全部完成后自动调用清理，输出 `/home/martis/picture_train/data/seq/v2_clean/t0..t3/`。
已有t0清理结果会核对源SHA后跳过，不重复生成。

**自动流程的边界：** 其余清理轨迹不会自动发布到远程，也不会热加入正在运行的t0训练。
当前DataLoader在启动时固定；后续多轨迹阶段需要明确同步、检查并另外启动。

## 5. 已就绪的数据与关键修复

- 训练清理目录：本机 `/home/martis/picture_train/data/seq/v2_clean/t0/`；
  远程 `/root/private_data/seq/v2_clean/t0/`，30片均已同步并逐文件SHA校验。
- 图元数21536248→20653078，删除约4.10%；平均PSNR损失0.0307468dB，最大0.0999928dB。
- 清理按真实RGB重新渲染计分；下降超过0.1dB保留原序列；原始元数据和源SHA保留。
- 远程训练配置中保留了本机原始来源路径作为溯源，训练使用显式--data/--protocol路径。
  不要为消除源路径外观差异重写已做SHA校验的文件。
- 清理版描述：本机 `/home/martis/picture_train/data/captions/captions_clean_v1.jsonl`；
  远程 `/root/private_data/captions_clean_v1.jsonl`，30000条、无缺失/重复/空描述。
- 描述SHA：`4caa472301e3f3ab89df66d540f4d7bb50cd25706e143d80e98dc74071c45265`。
- 17条截断/重复描述已单独重写，原始版本保留。未逐图人工审核全部描述。
- 属性：本机 `/home/martis/picture_train/data/celebamask256/attributes.json`；远程同结构数据目录。
- 固定划分：本机 `/home/martis/picture_train/data/celebamask256_split_v1.jsonl`；
  远程 `/root/private_data/celebamask256_split_v1.jsonl`，29400 train / 600 val。
  同一图片四条轨迹必须保持相同划分；四轨迹对应117600 train / 2400 val序列。
- 文字编码器已修复96token隐式截断：先完整编码（容量512），再池化到32个特征位置。
- 水平翻转序列自动交换描述里的显式左右方向，保留“左右”等成对表述。
- 训练前31项运行时代码、协议和v2码本已核对本机/远程一致。
- 当前协议SHA256：`bb54b53748557229f1135f26ad5ed9a750ff7410170930c396e47e5787a9d58c`。

## 6. 海光注意力兼容问题已解决

实际开训检查发现无mask画布注意力尝试加载缺失的 `flash_attn_2_cuda*.so`。
已给train.py增加显式 `--sdpa-backend math`，本次正式运行已带该参数。
默认auto不变；没有安装FlashAttention、改系统环境或移除画布模块。

当前stdout可能仍显示“not compiled with memory efficient attention”的一次性警告。
正式step200已实际运行成功，不要因这条警告擅自重装torch或安装CUDA包。
恢复本次训练时保留math选项。单独做推理/评估时也需确保SDPA后端与该环境兼容。

真实batch32、完整中文/画布/扰动的两步检查已通过，峰值分配显存15.93GiB。
本机结果 `/home/martis/picture_train/out/smoke_small_v2_t0_text_canvas.json`，远程同名JSON在个人数据根目录。

## 7. 后续执行顺序

1. **先确认训练持续推进。** 到500步检查last.pt及其step；1000步取回验证指标与六提示词样图，
   检查文字条件、轮廓、图层、孤儿率、是否早停。早期训练loss下降只证明优化在运行。
2. 继续收好t1/t2/t3。若控制器失败，核对实际argv、输出目录、锁、协议哈希、seed和flip后恢复；
   不按旧PID盲目杀任务或启动重复生产者。
3. 多轨迹清理完成后，验证每条覆盖30000张、源SHA、真实PSNR、固定划分及翻转描述一致性，
   同步到远程并校验，再计划第二阶段。
4. 第二阶段单独命名run，记录数据变化、优化器和学习率策略。
   可以用--init-weights从本轮兼容权重开启新优化器；若用--resume延长总步数，注意余弦LR计划会改变。
   不覆盖首阶段目录，不加载不兼容的v1形状词表。
5. 当前用户已同意真人优先；动漫并行准备用于后训练，不把旧两层动漫序列混入六层v2。
   风景/动物/物体仍是通用化后续方向，本轮尚未加入正式训练。

v1对照模型已完成8000步，权重 `/home/martis/picture_train/out/baselines/small_v1data_step8000.pt`，
采样评估在 `/home/martis/picture_train/out/sample_eval_v1data_final/`。
v1与v2属性头/词表不同，不直接比较原始token loss；图像指标和采样条件需对齐。

## 8. 环境、验证和提交规则

- 本机唯一CUDA/ML环境：`/home/martis/AI infra/.venv/bin/python`，uv管理；不要新装重复CUDA栈。
- 远程使用已有python/torch2.9 HIP与DTK，海光BW 64GB；项目代码在lpp_next。
- 最新完整回归 **148 passed in 29.98s**；Ruff E4/E7/E9/F及diff检查通过。
- 远程文字/清理相关15项测试通过；正式前向反向和正式step200均已实测。
- 不声称ONNX、C++推理、量化部署或动漫后训练已经完成。
- 提交使用用户现有git身份，不改身份、不加AI co-author或Generated with标记。
- 包/系统配置改动遵守用户系统变更日志、备份、Timeshift规则；本轮未做系统配置或包变更。
- 本交接之前最近三次提交：00af76f（正式开训记录）、8ae0fc2（启动门槛与SDPA修复）、
  ef76238（数据/文字/清理流水线），均已push。

中午旧交接已归档到 `docs/archive/CLAUDE_HANDOFF_2026-11-15_1246.md`，只作历史参考。
不要按旧交接重启v0分解、VLM标注或另一份训练。
