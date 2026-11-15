# 续接记录：标注收尾、长文本修复与多轨迹队列

2026-11-15，纽约时间约 15:30–15:47。续接 Claude 会话
`4612f68c-db51-4649-9c22-cc88ed5b4fc5` 的最新状态，而非中午的旧交接快照。

用户在最新会话中已同意：这一轮先用真人人脸验证 v2、多轨迹、文字条件与画布反馈；
动漫并行准备，留给后训练，不在当前训练中临时混入。

## 1. 接手时已经完成的工作

- HEAD `951d92d`：原生反向 v2 跳过没有贡献的 wave。
- 协议 v2 已从形状码本改为独立尺度、角度、beta 分档，保留 v1 兼容路径。
- 中文条件、CFG、画布反馈、扰动训练与 --init-weights 已实现并有测试/冒烟结果。
- v1 small 对照组已完成 8000 步，val loss=19.4814，无条件生成，约22.1M参数。
- 本机已有 v1 正式采样评估，32张/设置：baseline孤儿率5.94%，Rényi α=4为5.48%，
  温度0.8为9.69%；各组提前停止率0。32张只能作为当前样本对照，不能据此证明普遍改进。
- 远程中文描述已完成30000条，断在取回本机时的校验失败。

对照组权重：`out/baselines/small_v1data_step8000.pt`。
采样图和指标：`out/sample_eval_v1data_final/`。

## 2. 修复标注回传，并保留原始版本

远程完整文件12564682字节、30000行；接手时本机 `captions.jsonl.tmp` 只有6422528字节。
只确认传输不完整，没有证据把原因归结为数据损坏或某种工具输出限制。

重新按1MiB分块取回，每块检查退出状态和长度，最后验证完整SHA256及ID覆盖，再原子改名。

- 本机原标注：`data/captions/captions.jsonl`。
- 远程原标注：`/root/private_data/captions.jsonl`。
- 两者SHA256：`e723d700d0df93ac12cae8a15941966b265d91702fac44bb662c65ed6e1b7472`。
- 原来的不完整 `.tmp` 保留用于检查，不作为训练输入。

原标注审计：30000条、30000唯一ID、无缺失、无重复、无空长短描述。
7289条（24.3%）使用单行输出回退；这说明格式回退，不自动等同于语义错误。

进一步用本机Qwen tokenizer核对：18条原始输出达到110个token，其中17条未正常结束，
并能看到重复语句或截断。这与之前字符长度启发式的17条相符。

只对这17张重新生成：更简短的两行提示、max_new_tokens160、repetition_penalty1.1、
no_repeat_ngram_size6，并确认每条生成到EOS。原始30000条不覆盖。

- 修复记录本机：`data/captions/repairs_v1.jsonl`。
- 修复记录远程：`/root/private_data/captions_repairs_v1.jsonl`。
- 清理版本机：`data/captions/captions_clean_v1.jsonl`。
- 清理版远程：`/root/private_data/captions_clean_v1.jsonl`。
- 清理版SHA256：`4caa472301e3f3ab89df66d540f4d7bb50cd25706e143d80e98dc74071c45265`。
- 清理版仍为30000唯一ID、无空描述。
- 原始审计、候选ID、修复来源与清理审计保存在 `data/captions/` 的JSON/JSONL文件中。

后续中文训练应使用清理版路径，而不是不完整tmp。完整性和EOS检查不等于逐图人工语义审核。

## 3. 修复中文编码器的隐式截断

`TextEncoder` 原实现虽然把编码后的特征池化到32个位置，但tokenizer先截断到了96个token。
原标注中有42条长描述超过96；17条修复之后，仍有25条完整描述超过96，最长133。

现在默认使用文字模型实际容量（当前Chinese-CLIP为512），tokenizer不截断：
先完整编码，再池化到最多32个画图前缀位置。超出容量或显式设定上限时明确报错。
图元模型的前缀长度与总上下文预算不增加。

验证：

- 新测试覆盖长描述尾部、None/空条件、超过模型容量、显式96上限。
- 真实Chinese-CLIP用一条143-token原始长描述检查，输出与“完整编码143 → 池化32”逐元素一致。
  结果：`out/text_encoder_long_caption_check.json`。
- 更新的 `python/lpp/text.py` 已备份后同步到远程 `/root/private_data/lpp_next/`。
  备份：`/root/private_data/lpp-project-backups/2026-11-15/codex-continue-1540/text.py`。

## 4. 完成 HIP v2 的独立对照

脚本：`bench/bench_native_versions.py`。
相同输入，在同一已编译扩展内切换反向模式，预热各5次、AB/BA交替各40次。
形状B=16、N=256、256px，统计整个前向+反向MSE步骤，GPU events计时。

为了消除争用，临时暂停了经argv核验的两项远程分解，结束后finally恢复；另设60秒独立恢复保护。
已确认两个进程恢复并继续输出分解日志。

| 海光BW | median | p10–p90 |
|---|---:|---:|
| v1反向模式，完整单步 | 4.541ms | 4.525–4.593ms |
| v2跳过空闲wave，完整单步 | 3.965ms | 3.952–3.996ms |

吞吐加速1.145×，单步延迟降低12.7%。这不是端到端数据集转换的加速倍数，也不是NVIDIA上
单个反向kernel的41%吞吐改善；硬件、统计范围不同。
原始结果：本机 `out/bench_hip_v2_comparison.json`、远程 `/root/private_data/bench_hip_v2_comparison.json`。

## 5. 四条轨迹的有限队列

`tools/run_decomposition_plan.py` 只完成指定四条轨迹，不自动启动训练。
它会检查数据/协议哈希、种子、翻转、分片ID、offset、属性宽度、数组长度与有限PSNR。
同一plan使用flock防止重复控制器。既有生产进程按PID和实际argv中的输出目录加入等待；
进程结束但数据不完整时停止报错，不假装完成。

| 轨迹 | 种子 | 翻转 | 目录 |
|---|---:|---|---|
| t0 | 0 | 否 | `seq/v2` |
| t1 | 1 | 否 | `seq/v2_t1` |
| t2 | 2 | 是 | `seq/v2_t2` |
| t3 | 3 | 是 | `seq/v2_t3` |

每条目标仍是30000张：本机负责0/2偶数分片，远程负责1/2奇数分片。
这些是同一批30000张图片的4种轨迹，不是120000张独立图片。
翻转描述和RGB目标的处理已补齐，见下文；完整轨迹仍在生成中。

本机：

- t0的15个偶数分片已完成并验证，共15000张。
- t1已实际启动，输出 `data/seq/v2_t1/`，日志 `data/decompose_v2_t1_local.log`。
- 之后依次t2、t3。
- 计划：`data/trajectory_plan_local_v2.json`。
- 状态：`data/trajectory_plan_local_v2.state.json`；控制器日志：同名 `.log`。

远程：

- t0、t1由已有进程18481/22694分别运行；这些PID是快照，不能长期当作固定常量。
- 用户随后要求利用远程余量增加并行度，t2/t3已提前启动，快照PID25039/25954。
- 原控制器24757在确认尚未派生子进程时停止；计划备份后加入t2/t3的真实PID和join_only，
  再启动控制器25955。现在四项均为加入等待、验证，不会再次派生同输出任务。
- 计划/状态/控制器日志：`/root/private_data/trajectory_plan_dcu_v2.{json,state.json,log}`。
- t2/t3使用 `/root/private_data/lpp_next` 与 `/root/private_data/ext_build_v2`。
- 老t0记录的协议路径位于 `lpp/protocol.json`，新任务位于 `lpp_next/protocol.json`，
  内容SHA256相同，队列按各自真实配置验证。

若机器或控制器重启，先读state/log与实际进程；不要盲目重复启动旧plan。
计划锁能防重复控制器，不能阻止别人完全绕过控制器手动起同输出任务。

## 6. 下一步与验证范围

1. 持续完成本机/远程轨迹；跨机器汇集和全量完成后的清理已接入，尚未处理完全部轨迹。
2. 使用清理版描述、属性模板和固定验证划分准备下一轮small。
3. 按原定计划训练中文条件+画布反馈+扰动模型，与已有v1对照组分开报告。
4. 动漫仍并行准备供后训练；不把两层旧试验序列混进六层v2。

本轮最新代码本机完整GPU测试：**143 passed in 21.87s**。
远程文字方向/清理/长文本测试：**15 passed**（两条远程已有依赖弃用提示）。
另已进行真实GPU清理试跑，结果见下文。基础Ruff与git diff检查通过。

## 7. 汇集、翻转一致性和清理（约16:35之后更新）

`tools/sync_decomposition_plan.py` 使用rsync取回远程已经原子落盘的奇数分片，逐文件核对SHA256；
只规范配置中的机器路径，种子、翻转、步数、协议版本和哈希必须一致。
同一计划有独立flock，不会重复启动汇集器；传输中断最多重试三次，完整性冲突会停止。

- 当前汇集器快照PID457459；日志 `data/trajectory_plan_local_v2.sync.log`。
- 状态 `data/trajectory_plan_local_v2.sync.json` 记录分片、校验和与阶段。
- 首次已取回19片：t0为13片，t1为6片；与本机合计t0为28/30、t1为12/30。
  这是首次汇集快照，查看状态文件获取后续进度。
- 有限watch每60秒检查；四条轨迹各30片全量验证通过后，调用清理输出 `data/seq/v2_clean/t0..t3/`。
- 不启动模型训练。SSH master失效后需要恢复连接，再核对现有进程和日志恢复。

`load_shards` 从父目录config读取flip标记。`SequenceDataset` 在挑选描述后，交换显式中英文左右方向，
保留“左右”“左、右”等成对表述。模板属性不受影响。
对30000条清理版长短描述检查：469条长描述、17条短描述发生方向替换；60000次正反变换均恢复原文。
结果 `data/captions/mirror_audit_v1.json`。这属于确定性方向校正，不代表重新人工审核全部描述。

修复 `tools/prune_sequences.py` 原来的指标问题：删除图元后不再沿用删前PSNR。

- 按原始RGB目标重新渲染，翻转轨迹匹配翻转目标，记录真实删前/删后PSNR。
- 保留 `source_psnr` 原始元数据、`unpruned_psnr` 重新计算值、源NPZ的SHA256。
- 可见质量用64个图元一块计算精确后缀透射率，限制显存峰值。
- 默认删除可见质量不足2像素的图元；若实际PSNR下降超过0.1dB，保留该图原序列。
- 不允许源/输出嵌套，不覆盖配置不一致的目录；续跑验证源哈希，至少保留一个图元。

真实8张样本试跑：5745→5429图元，删除5.50%；平均29.60528→29.57371dB，
平均损失0.03157dB、最大0.06038dB；重新计算的删前分数与原始元数据最大差约1.9e-6dB。
用镜像RGB及flip配置重复检查，输出指标逐项一致；源文件SHA256不变。
结果 `out/pruning_pipeline_check.json`。小样本用于验证流程，全量质量分布尚未统计。

已固化原训练器默认的ID哈希划分：`data/celebamask256_split_v1.jsonl`，
29400张train、600张val，同一图片四种轨迹保持同一划分；不会把翻转/不同种子分到两侧。
完整四轨迹将对应117600条train、2400条val序列，不等同于独立图片数。

远程并行计划原件备份于 `/root/private_data/lpp-project-backups/2026-11-15/parallel-remote/`。
文字/数据加载/清理代码同步前的备份位于同日期 `dataset-pipeline/`。
本轮只修改项目代码、数据与项目进程，没有安装包或修改系统配置。
