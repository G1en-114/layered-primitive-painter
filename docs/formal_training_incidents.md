# 正式 v3 训练上云：问题与排查记录（2026-11-19）

目标：把 protocol v3（高斯 + Bézier 线条）的正式训练放到租用的 2×RTX 5090（32 GB/卡）容器上跑：
LP-FT 两阶段（A：只训新增部分 1000 步；B：全参 12000 步），DDP，每 2000 步双领域评测。
脚本在 `ops/v3_train/`（train.sh / eval_watch.sh / launch.sh / up.sh / go.sh / connect.sh / fetch.sh / dry_run.sh）。

本文按"现象 → 排查过程 → 根因 → 修复 → 面试可能追问"记录每一次卡住的地方，供面试复盘。
时间为 UTC（北京时间 +8）。花费为估算：5090 租价约 ¥3–4/卡·时（具体以平台账单为准，本会话没有核对过）。

## 总览

| 时间 (UTC) | 事件 | 影响 | 花费 |
|---|---|---|---|
| 12:54–13:00 | 笔记本 dry run：stage A 双 rank 通过，stage B OOM；改为单卡验证 B | 无（本地） | 0 |
| 12:56 | expandable_segments 触发 NVML INTERNAL ASSERT（驱动升级未重启） | 无（本地） | 0 |
| 13:05–13:15 | 第一次开双卡机器：装 transformers 版本冲突（5.x 不认 torch 2.6.0a0） | 双卡空转 | 约 ¥1–2 |
| 13:10–13:20 | 上传只有 ~0.3 MB/s，4 GB 预计近 3 小时 | 双卡空转计费 → 用户要求关机 | 约 ¥1–2 |
| 13:18 起 | 切到**无卡模式**开机做上传和模型下载 | 只付无卡费用 | 低 |
| 13:24 | 用户运行 up.sh，"没有进度条"，Ctrl-C 打断，模型下载连带失败 | 重跑一次 | 低 |
| 13:2x | configuration.json 在 hf-mirror 404 | 加 ModelScope 回退 | 低 |
| 14:08 | 双卡模式开机；裁判模型 fp16 哈希不一致 → 定位到 [PAD] 行 | 几分钟 | 小 |
| 14:09 | "training already running" 误报（pgrep 自匹配）；eval_watch 启动目录错 | 几分钟 | 小 |
| 14:09–14:11 | stage A 双卡启动即 NCCL 报错 `Cuda failure 1 'invalid argument'` | 训练未开始 | 排查中 |

---

## 1. 笔记本 8 GB 显存上双 rank DDP：stage A 通过、stage B OOM

**现象**
- dry run（`ops/v3_train/dry_run.sh`）用两个 DDP rank 共享一张 RTX 5070 Laptop（7.62 GiB），后端 gloo。
- stage A（只训 2.94M/129.3M 参数）跑通，两 rank 参数校验和 drift = 0。
- stage B 启动即报 `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 MiB. GPU 0 has a total capacity of 7.62 GiB of which 27.56 MiB is free. Of the allocated memory 3.44 GiB is allocated by PyTorch`。

**排查过程**
1. 先怀疑 batch 太大：把 batch 从 2 降到 1（accum 2），仍 OOM（3.24 GiB/rank）。
2. 算账：129M 参数 × (fp32 权重 4 + 梯度 4 + Adam 两个状态 8) ≈ 2 GB/rank，加文本编码器、CUDA context、canvas 渲染，约 3.2 GB/rank；两个 rank 在同一张 8 GB 卡上 ≈ 6.5 GB + 激活，必然不够。
3. stage A 能过，是因为 `--train-only` 冻结了 99.8% 的参数：没有梯度、没有 Adam 状态，每 rank 只多几十 MB。

**根因**：笔记本单卡模拟双卡时，两份全参优化器状态放不下；正式机器一卡一 rank、32 GB，不受影响。

**修复**
- dry run 拆开验证：A 用 2 rank gloo（验证 DDP + 冻结 + 数据混合），B 用单卡 base 模型（验证从 A 的快照初始化、全参训练、快照、评测）。`dry_run.sh` 相应改成两次调用；`train.sh` 支持 `STEPS_B=0` 只跑 A。
- B 的双 rank 全参 DDP 在笔记本上没有验证（当时设想过用 tiny 配置单独验证 DDP 全参路径，但 init 权重是 base 尺寸，没有做）。**这是上云前唯一未验证的路径。**
- 提前排查 DDP 卡死风险：`attr_logits` 里 `torch.where(event == LINE, line_heads(h), heads(h))` 对每一行都计算 line_heads，所以即使某个 rank 的 batch 全是人脸（无 LINE 笔画），line_heads 也有（为零的）梯度，DDP 不会因 unused parameters 永久等待 all-reduce。永远拿不到梯度的参数（最后一个属性的 decoder embedding、无文本时的 text_proj）在 DDP 包装前冻结。

**面试可能追问**
- Q：一个 1.3 亿参数的模型训练时显存大概多少？A：fp32 + AdamW 约 16 字节/参数 ≈ 2 GB，再加激活；混合精度下激活用 bf16，但主权重和优化器状态仍是 fp32。
- Q：为什么冻结参数后显存骤降？A：冻结参数没有 `.grad`，优化器也不为它们建状态；只剩权重本身。
- Q：DDP 什么时候会卡死？A：某个参数在某个 rank 上没参与前向，梯度 bucket 永远等不齐。要么 `find_unused_parameters=True`（每步遍历 autograd 图，慢），要么保证所有可训练参数每步都在图里，要么提前冻结。
- Q：怎么确认 DDP 副本一致？A：训练结束时每个 rank 算参数校验和，broadcast rank 0 的值比较（train.py 末尾，drift = 0）。

## 2. expandable_segments 触发 NVML INTERNAL ASSERT

**现象**：为缓解 OOM 设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后，启动即崩：
`RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlInit_v2_() INTERNAL ASSERT FAILED at ".../c10/cuda/PeerToPeerAccess.cpp":83`。

**排查过程**
1. 早先 `nvidia-smi` 就报 `Failed to initialize NVML: Driver/library version mismatch`（用户态 620.70，已加载内核模块 615.78）：驱动包被升级但没重启。
2. 普通 CUDA 运算仍然可用（`torch.cuda.is_available()` 为 True），说明只是 NVML 坏了。
3. expandable_segments 的实现要走 NVML 查询，所以单独这一项触发断言。

**根因**：系统驱动升级未重启，NVML 用户库与内核模块版本不一致。

**修复**：去掉 expandable_segments（它本来也救不了第 1 条的 OOM），提醒用户重启（之后内核已变为 7.3.4，问题消失）。

**面试可能追问**
- Q：CUDA 能用但 nvidia-smi 不能用，可能是什么？A：NVML（libnvidia-ml）与内核驱动版本不匹配，典型是升级驱动后未重启；CUDA runtime 走的是另一条路径。
- Q：expandable_segments 解决什么问题？A：缓存分配器碎片化：允许段按需扩展，减少"reserved 但不可用"的显存；解决不了总量不够。

## 3. 上传只有 ~0.3 MB/s

**现象**：第一次把 4 GB（token 分片 1.8 GB、init 权重 0.5 GB、两个文本编码器 1.5 GB、验证集等）逐文件 ssh 上传，测得本机 Wi-Fi 发送速率约 405 KB/s，另一次 62 秒只涨 12 MB（约 0.2 MB/s）。预计近 3 小时，比训练本身还长，而这期间双卡在计费。

**排查过程**
1. 先算体积，找能不传的：两个 Chinese-CLIP 是公开模型，本地有 `aria2.list`（来源 ModelScope）和 `SHA256SUMS`。在远端测下载：ModelScope 约 2.4 MB/s，hf-mirror 约 8.5 MB/s，比笔记本上传快约 20 倍。
2. 测压缩率：token 分片 zstd -9 后约减半（12.9 MB→6.9 MB，6.8 MB→3.3 MB）；init 权重（fp32）几乎不可压。
3. 查路由：`ip route get <box>` 显示走 `dev Meta`，源地址 198.18.0.1——Clash/mihomo 的 TUN 模式（fake-IP 段 198.18.0.0/15），去国内机器的 SSH 也被代理接管，很可能绕了代理节点。
4. 之前同一台机器曾上传过 13 GB 图像，速度明显更快，侧面印证是本机网络路径变了，而不是机器入口限速（这一点没有严格验证）。

**根因**：瓶颈在笔记本上行链路，并且流量走了本地 TUN 代理。

**修复**（commit f5fbb52、679dd1c、9637030）
- 公开模型让远端自己下载并按 SHA256SUMS 校验（`launch.sh models`）；judge 用 fp32 large 在远端转 fp16。
- 其余数据 zstd 压缩，改为**单个 tar 流**（`tar -ch | pv | zstd | ssh "zstd -d | tar -x"`），代替每个文件一次 ssh 往返。
- 按远端已有文件大小跳过，断了重跑即续传。
- **计费策略**：用户要求后改成"无卡模式开机 → 上传/下载/装环境 → 关机 → 双卡模式开机只做训练"。
- 进度条：装 `pv`（pacman，已记入 `~/system-changes/README.md` 并做 Timeshift 快照）。
- 建议用户关 TUN 或加规则 `IP-CIDR,<box>/32,DIRECT`（用户的代理配置，没有替用户改）。

**面试可能追问**
- Q：数据怎么上云最省钱？A：先算"体积/带宽 = 时间"，再算"时间 × 计费单价"。公共资源让云端自己拉，私有数据压缩后单流传，在不计 GPU 费的模式下做准备。
- Q：为什么单 tar 流比逐文件快？A：省掉每个文件一次 ssh 会话/命令的往返延迟，压缩器也能看到连续数据。
- Q：怎么确认远端下载的模型和本地一样？A：用发布方或本地的 SHA256SUMS 逐文件校验，不一致就重下。
- Q：198.18.x.x 是什么？A：RFC 2544 基准测试保留段，Clash/mihomo fake-IP 和 TUN 常用它，看到它说明流量进了本地代理。

## 4. "怎么没有进度条"：等待被误认为卡住，Ctrl-C 连带杀掉后台下载

**现象**：用户运行 `up.sh`，输出停在 `== 4/4 code + data`，看起来不动，按了两次 Ctrl-C。之后 `out/v3_models.log` 显示 `upload first` / `step models failed`。

**排查过程**
1. `git archive HEAD | wc -c` = 5,089,280 字节，按 0.3 MB/s 约 17 秒；这一步没有任何输出，进度条只在后面的数据步骤才出现。
2. 远端检查：base 模型已下了一半（缺 configuration.json，见第 5 条），large 目录只有 SHA256SUMS。
3. 后台的 `launch.sh models` 是 `&` 启动的子进程，和前台在同一个进程组；Ctrl-C 的 SIGINT 发给整个前台进程组，里面的 ssh 自己处理 SIGINT 后退出，`test -f ...SHA256SUMS.ok` 失败，于是打印 "upload first"（这是最可能的解释，没有逐步复现）。

**根因**：脚本有长时间静默的步骤，用户体验像卡死；后台任务没有和前台终端隔离。

**修复**（commit 9637030）：每一步开始前打印在做什么；代码上传也经过 `pv`；后台模型下载用 `setsid ... < /dev/null &` 放进独立会话，Ctrl-C 不再波及。

**面试可能追问**
- Q：为什么 Ctrl-C 会杀掉后台任务？A：SIGINT 发给终端的前台进程组；脚本里 `&` 启动的子进程默认仍在同一进程组/会话，只有用 setsid 或 nohup 和信号处理隔离才安全。
- Q：长任务脚本的可观测性怎么做？A：每步打印开始/结束和预计耗时，传输用进度条，结果落日志文件，可重入。

## 5. configuration.json 只在 ModelScope 上有

**现象**：远端按 SHA256SUMS 下载 base 模型时，只缺 `configuration.json`；`set -e` 下 curl `-f` 失败，整个下载步骤中止。

**排查过程**：逐文件对两个镜像发 HEAD：hf-mirror（OFA-Sys 仓库）上 `configuration.json` 返回 404，其余文件 200；ModelScope 上全部 200。configuration.json 是 ModelScope 特有的元数据文件，本地副本本来就来自 ModelScope。

**修复**：`launch.sh models` 先试 hf-mirror（快），失败回退 ModelScope；最终仍以 SHA256SUMS 为准校验。

**面试可能追问**
- Q：多源下载如何保证一致？A：以内容哈希为准，不以来源为准；任何源下的文件都要过同一份校验和。

## 6. 远端 Python 环境：transformers 与 NGC torch 2.6.0a0 的版本冲突

**现象**
- 容器自带 NVIDIA NGC 的 torch `2.6.0a0+ecf3bae40a.nv25.01`、numpy 1.26.4，缺 transformers。
- 直接 `pip install` 被 PEP 668 拦截（`externally-managed-environment`）。
- 装 transformers 5.23.0 后：`Disabling PyTorch because PyTorch >= 2.6 is required but found 2.6.0a0`。

**排查过程**
1. 2.6.0a0 是预发布版本，按 PEP 440 排序**小于** 2.6.0，所以 5.x 的版本检查判定 torch 不可用。
2. 想到 transformers 4.50 之后，在 torch < 2.6 时拒绝 `torch.load` 加载 `.bin`（CVE-2025-32434 的防护），而文本编码器正是 `pytorch_model.bin`，所以不能随便选新的 4.x。
3. 代码只用 `BertTokenizer`、`ChineseCLIPModel`、`ChineseCLIPImageProcessor`，这几个在 4.x 里早已稳定。
4. 吸取前一天的教训：在这台 5090 上 pip 曾把 numpy 升到 2，导致 torch 损坏（当时装 `numpy<2` 修复）。

**修复**（commit fd75a42）：`pip install --break-system-packages -i <清华镜像> transformers==4.48.3 numpy==1.26.4`，把 numpy 钉在现有版本。这是一次性租用的容器，所以用 `--break-system-packages` 可以接受。`launch.sh setup` 固化了这个逻辑。

**面试可能追问**
- Q：为什么 2.6.0a0 < 2.6？A：PEP 440 中 pre-release（a/b/rc）排在对应正式版之前。
- Q：怎么防止装一个包把 torch 的依赖搞坏？A：显式钉住关键依赖版本（numpy==当前版本），或者用约束文件 `-c`，或者用独立 venv。
- Q：为什么新版本 transformers 不让旧 torch 加载 .bin？A：.bin 是 pickle，`torch.load` 可能执行任意代码；torch 2.6 默认 `weights_only=True` 修了这个问题，transformers 因此要求 2.6+；safetensors 没有这个问题。

## 7. 裁判模型（fp16 judge）在远端复现后哈希不一致

**现象**：远端从 fp32 large 转出的 fp16 safetensors，文件大小（812,538,338 字节）和 key 数（591）都与本地一致，但按 key 排序后的总哈希不同（55095148… 和 8d61a450…）。

**排查过程**
1. 逐 tensor 比哈希：591 个里只有 `text_model.embeddings.word_embeddings.weight` 不同。
2. 逐行比这个 tensor（21128 行）：只有第 0 行不同。本地第 0 行全零，远端绝对值和为 26.45。第 0 行是 `[PAD]`。
3. 判断影响：[PAD] 位置被 attention mask 屏蔽，文本特征取 [CLS]，所以对评测结果没有影响。本地那份是在 SCNet 上做的，转换细节没有记录，猜测是 transformers 版本差异导致 padding_idx 行被置零（未验证）。

**根因**：两边转换环境不同，[PAD] 的嵌入行处理不一致。

**修复**：为了与 base_v3 基线**严格可比**，在远端把该行置零并保留 metadata 重新保存，再次逐 tensor 比对，"judge identical"。

**面试可能追问**
- Q：两个模型文件不一致，怎么最快定位？A：分层哈希：文件 → tensor → 行或块，二分式缩小范围，不必传输整个文件。
- Q：PAD embedding 不同会影响输出吗？A：有 attention mask 并且 pooling 不取 PAD 位置时不影响；如果用 mean pooling 而且没有处理 mask，就会影响。
- Q：评测为什么要求裁判逐比特一致？A：所有快照都和 base_v3 的历史分数比较，裁判变了，差异就无法归因于模型本身。

## 8. pgrep -f 自匹配："training already running" 误报

**现象**：`launch.sh start` 输出 `training already running`，但远端 `ps` 里没有任何 train.sh / train.py。

**排查过程**：检查逻辑是 `$G "pgrep -f 'v3_train/train.sh' > /dev/null"`。远端 sshd 以 `bash -c "pgrep -f 'v3_train/train.sh' ..."` 执行，这个 bash 的命令行本身就包含该字符串，所以 pgrep 匹配到了自己的父 shell。之前在本地用 pkill/pgrep 也误杀过自己的 shell（exit 143/144），是同一类问题。

**修复**：改成 `pgrep -f '[v]3_train/train.sh'`。正则 `[v]` 匹配字母 v，但命令行里写的是 `[v]`，字面上不再包含 `v3_train`，所以不会匹配自身（本次修复写在 launch.sh，提交情况以主会话为准）。

**面试可能追问**
- Q：`ps | grep foo` 为什么总能看到 grep 自己？怎么避免？A：grep 的命令行里有 foo；可以用 `grep [f]oo`、`pgrep -x` 精确匹配进程名，或者用 pid 文件 / systemd unit 管理进程，而不是靠模式匹配。
- Q：远程执行时多了哪一层？A：ssh 远端用登录 shell 的 `-c` 执行命令串，所以命令串本身会出现在进程表里。

## 9. `cd $R && cmd &` 只作用于后台子 shell

**现象**：start 之后只看到 train.sh 在跑，eval_watch.sh 不在；`/root/lpp/v3train/eval_watch.nohup` 不存在。

**排查过程**：命令串是 `cd $R && setsid nohup bash code/.../train.sh > ... &`，换行后是 `setsid nohup bash code/.../eval_watch.sh ... &`。`&` 作用于整个 `cd && cmd` 列表，它在后台子 shell 里执行，所以 cd 只改变那个子 shell 的目录。下一行在 ssh 登录目录（~）下执行，相对路径 `code/...` 不存在，nohup 输出也落在 ~。另外 stage A 很快失败（第 10 条），train.sh 写了 `train_failed`，所以 watcher 就算启动了也会立即退出。

**修复**：改为 `cd $R; setsid nohup ... &`，先在当前 shell 里 cd（修复状态以主会话为准）。

**面试可能追问**
- Q：`a && b &` 的语义？A：整个 AND 列表放到后台子 shell 执行，里面对环境（cwd、变量）的修改不影响当前 shell。

## 10. NCCL 2.25.1 在 RTX 5090 上通信失败（进行中）

**现象**：stage A 双卡（torchrun，2 GPU × batch 16）启动约 1.5 分钟后（数据已加载：218 shards，135,071 train / 4,250 val），在 DDP 构造时的 `_verify_param_shape_across_processes`，也就是第一次集合通信处崩溃：
`torch.distributed.DistBackendError: NCCL error ... unhandled cuda error ... NCCL version 2.25.1`
`ncclUnhandledCudaError: Call to CUDA function failed. ... Cuda failure 1 'invalid argument'`

**排查过程**
1. 写最小复现 `nt.py`（init nccl → 1M 元素 all_reduce），排除我们自己的代码：同样失败。
2. 猜测一：消费级卡不支持 P2P，所以设置 `NCCL_P2P_DISABLE=1`，仍失败。
3. 猜测二：/dev/shm 太小（实际 2 GB），所以再加 `NCCL_SHM_DISABLE=1`，仍失败。
4. `NCCL_DEBUG=INFO` 看到关键一行：`ncclMaxSharedMem 82240 exceeds device/fn maxSharedMem 79856`。也就是说，NCCL 2.25.1（NGC 25.01 自带，基于 cuda12.8 构建）的 kernel 需要的每块共享内存超过了 RTX 5090（sm_120，消费级 Blackwell）的上限。后续 kernel 启动参数非法，于是报 'invalid argument'。驱动是 610.43.02（CUDA UMD 13.3），torch 的 arch 列表包含 sm_120，所以计算 kernel 本身没问题。

**根因**：容器自带的 NCCL 版本没有适配消费级 Blackwell 的共享内存上限（P2P/SHM 只是传输层，换掉也绕不开 kernel 本身）。

**修复方案（进行中）**：用 `pip install --no-deps --target /root/nccl nvidia-nccl-cu12==2.27.7` 把新版 NCCL 装到独立目录，启动时 `LD_PRELOAD=/root/nccl/nvidia/nccl/lib/libnccl.so.2` 替换系统库，不动镜像原有的包。备选：gloo 后端（all-reduce 走 CPU，每步约 0.5 GB 梯度，会明显变慢）。

> TODO（主会话补）：2.27.7 + LD_PRELOAD 的最小复现结果；train.sh 里如何固化（环境变量或 launch 脚本）；最终训练速度。

**面试可能追问**
- Q：NCCL 报 'invalid argument'，你怎么排查？A：先用最小 all_reduce 脚本排除业务代码；再开 `NCCL_DEBUG=INFO` 看拓扑、传输和 kernel 配置；逐个关掉 P2P/SHM/IB 缩小范围；最后对照 NCCL 版本、驱动和 GPU 架构的兼容性。
- Q：为什么关掉 P2P 也不行？A：P2P/SHM/NET 只决定数据怎么搬，归约 kernel 总要在 GPU 上跑；kernel 的共享内存需求超过硬件上限，换什么传输都会失败。
- Q：不重装镜像怎么换 NCCL？A：NCCL 是动态库，ABI 稳定，可以用 LD_PRELOAD 或 LD_LIBRARY_PATH 指向 pip 装的 nvidia-nccl-cu12；用 `NCCL_DEBUG=VERSION` 确认实际加载的版本。
- Q：消费级卡做多卡训练有什么坑？A：没有 NVLink，P2P 常被禁用（走 PCIe/主机内存）；还有共享内存上限与数据中心卡（如 B200）不同、新架构需要较新的 NCCL/CUDA 等兼容性问题。

## 11. 更早的几个问题（简版）

- **线条属性没有 loss，线条宽了 18 倍**：attribute loss 只对 PRIMITIVE 事件计算，LINE 的属性头从未被训练，用户发现"用椭圆在画线"。修复为 `torch.isin(ev, stroke_events)` 覆盖所有笔画事件，加独立 LINE heads，并加测试 `test_line_strokes_get_attribute_losses`。修复后 anime FD 从 0.367（对照）降到 0.332。
- **逐步量化吃掉线条增益**：分解器每一步都把线条吸附到 token 网格（straight-through），增益为 +0.00 dB。改为最后 20% 步数冻结位置和角度、其余属性继续适配（freeze 模式），+0.48 dB。
- **DCU（HIP）上 flash attention 失败**：在 HIP 上 SDPA 默认改为 math 后端（train / eval_text / train_grpo 一致）。
- **Python 3.14 默认 forkserver 撑爆 /dev/shm 配额**：forkserver 会把整个数据集 pickle 到 /dev/shm。在 Linux 上 DataLoader 显式改用 `multiprocessing_context="fork"`（copy-on-write 共享）。
- **SCNet 网关截断长 SSH 输出流**：大文件下载会断。改为分段 `dd`（每段后 `sleep 3`），校验大小，不对就重试（`ops/plan_v3/local_grpo_eval.sh` 的 `fetch()`）。

## 可迁移的经验

- **先算带宽再决定传什么**：体积 / 带宽 = 时间，时间 × 单价 = 钱。公开资源让云端自己拉（哈希校验），私有数据压缩后单流传。
- **计费 GPU 上只做训练**：上传、装环境、下模型都在无卡模式或本地完成；必要时开机后先跑最小复现（如 nt.py），确认能跑再起正式任务。
- **上云前在本地做最便宜的端到端彩排**（dry run），并明确列出彩排没覆盖的路径（本次是双 rank 全参 DDP 和 NCCL 版本），这些正是上云后最容易出问题的地方。
- **差异用分层哈希定位**：文件 → tensor → 行，不传大文件也能精确找到不同点，再判断是否影响结论。
- **进程管理不要靠模式匹配**：pgrep/pkill 会匹配到自身或父 shell；用 `[x]` 技巧、pid 文件或 systemd unit。
- **后台任务要和终端隔离**：`setsid ... < /dev/null &` 或 systemd-run，避免 Ctrl-C/关终端波及；`a && b &` 里的 cd 不影响后续命令。
- **镜像自带的通信库也要验证**：新 GPU 架构（消费级 Blackwell）上，先用 `NCCL_DEBUG=INFO` 的最小 all_reduce 验证，再起正式训练。
- **长脚本要可观测、可重入**：每步打印、进度条、日志落盘、按大小或标记跳过已完成部分。
