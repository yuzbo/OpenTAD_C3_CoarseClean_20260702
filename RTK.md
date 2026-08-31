# RTK Project Rules

## Research priority

The first obligation is to improve the scientific model: a stronger, more
novel, and empirically defensible offline-TAD algorithm. Engineering rigor is
required only to the extent that it makes the model, comparison, and claim
auditable. Do not substitute a more elaborate framework, provenance layer, or
visualization for a model improvement; do not spend material research time on
infrastructure that cannot change a scientific decision. Conversely, do not make
an efficiency, novelty, or paper claim without the minimum matched evidence
that can falsify it.

## 论文实验推进优先

ZoomToken 由 Pro 冻结唯一科学任务与裁决门，Codex 只做最小忠实实现和授权实验，Critic/Evaluator 独立且有限范围。论文实验推进优先，基础设施失败只做最小修复。每个正式终态或客观 blocker 后必须进入新的 Project Pro 复盘；失败路线先做有界、证据化根因分析，复盘同轮给出唯一下一任务，不得停在纯 STOP 或无限审计。具体角色边界见 `docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`。

当证据形成真实科学选择、缺失信息或来源冲突时，Codex 可主动提交充分权威上下文并请求 Pro 独立建议、裁决或下一任务；不得预置偏好路线、默认方案、期望裁决或穷尽候选，必须允许 Pro 拒绝当前 framing、提出未列替代并独立决定方向。

ZoomToken 冻结任务连续执行规则：用户已授权的冻结科学任务由 Codex 连续推进至正式终态证据摄取和 fresh post-result Pro 裁决，不因普通 Git、远端、Slurm 或证据摄取步骤重复请示。长时任务提交并确认后，等待责任转交计算系统，Codex 结束主动推理阶段；这不构成任务放弃。Pro 请求无实际 submission 时恢复同一 request；已有 submission/conversation 时只等待或回取，禁止 duplicate/follow-up。工程或协议失败没有科学方向，正式终态默认不得自动重跑，除非冻结任务明确授权。

一次科学实验限制的是一次真正进入模型、canonical 数据或 evaluator 的科学尝试，不简单等同于一次 `sbatch` 调用。确定性、结果盲、零科学执行的外部 launcher 或正式 execution-harness import/registry/config/构造/checkpoint-load 缺陷，在候选、模型、配置、数据、checkpoint、seed、资源语义、evaluator/NMS、测量和主张边界均不漂移时，可按角色合同的有界规则处理；旧作业永久计入 scheduler submission ordinal。显式 scheduler `1/1` 上限禁止 Codex 自动补提，只有 fresh Pro 可建立独立任务并授权一次 scheduler-2/scientific-1 replacement；replacement 失败后无第三次提交。

`PRE_RUN_READY` 必须覆盖正式入口在首个科学操作前的真实 import/registry、对象构造和 checkpoint strict-load 图；静态检查、配置字段、checkpoint 容器、Git identity 或替代 fixture 不能代替可执行 construction witness。若冻结任务依赖真实形状，可在同一正式准备函数中执行一次无计时、无显存、无 prediction/metric 的 dry ledger；witness 不通过即返回 blocker。

同一候选连续两个独立 construction blocker 后退出普通 repair 模式：fresh Pro 只能 STOP，或授权一次无 replacement 的 production-faithful atomic witness+measurement；该动作任一失败即终止精确定义候选。

### 长任务的事件驱动等待

- 已提交的下载、传输、Slurm 作业、训练或 Pro 生成不是主动推理工作。提交方先记录不可变 job/session 身份、配置身份、预期终态产物、最长时限和恢复规则，并确认后端已经接受任务。
- 墙钟等待必须由真实终端计时完成，不得由模型推理、估算运行时长、无间隔快速轮询或提前结束 Goal 来代替。Bash/WSL/Linux/macOS 使用 `sleep <秒数>`，Windows PowerShell 使用 `Start-Sleep -Seconds <秒数>`；分钟级外部任务在没有另行冻结频率时默认每 `600` 秒执行一次“检查 -> 真实等待 -> 再检查”。
- 能由计算系统记录终态时，优先使用作业自身的 terminal receipt 或依赖收尾；既有任务不允许追加远端收尾作业时，可用恰好一个机器侧后台等待进程，在该终端进程中执行有限间隔的 `sleep` 与只读状态判断。已有等待器时不得再创建第二套前台或后台轮询；Codex 只在它给出终态或预定恢复条件成立后消费结果。
- 若没有可用的后台等待器且当前 Goal 的唯一正确动作确实是等待，Codex 必须在当前终端执行有界的“状态检查 -> `sleep`/`Start-Sleep` -> 状态复查”循环，直至明确完成、失败、预注册停止条件或真正需要人类凭据/资源/科学决策的阻塞。等待属于 Goal 的执行步骤，不能以“尚无新输出”为由把未完成 Goal 结束或交还给用户。
- 没有另行冻结频率时，长实验等待默认使用 10 分钟真实终端计时。计时窗口必须完全静默，且计时器是该窗口唯一活跃命令：不并行调用任何工具，不发送 Codex commentary/final，不打印倒计时、进度字符或等待文字，不做状态/日志/partial 查询，不改文件、不操作浏览器、不提交任务，也不执行除该计时器之外的命令；计时结束后只做一次权威终态检查，非终态则进入下一轮同样的静默等待。
- “倒计时”只表示阻塞式 `sleep`/`Start-Sleep` 墙钟计时，不表示可视 countdown；严禁逐秒、逐分钟或以任何频率打印剩余时间、心跳字符、进度条或占位消息。整个计时窗口的可见输出必须为空。
- 同一等待对象只允许一个 terminal-only waiter 或计时循环。不得因普通 heartbeat、goal continuation 或界面刷新并发建立第二个 waiter；已有 waiter 时只在既定恢复节点读取一次其终态输出。
- Codex 不主动消费无关 heartbeat，也不得为了确认“仍在运行”而连续调用 `squeue`、`sacct`、`tail`、`ps`、`stat` 或等价工具。普通 goal continuation 不构成额外状态查询理由，也不输出“仍在等待”一类前台文字；恢复后先做一次终态核验，再收集结果或保存 blocker。
- 当前闭环只需使用普通的 `SUBMITTED`、`TERMINAL`、`NEEDS_SCIENTIFIC_DECISION` 三种含义：`SUBMITTED` 后 Goal 保持活动并转入终端真实计时，不再进行额外模型推理或重复查询；`TERMINAL` 后摄取冻结证据；只有完整结果或必须改变科学条件时才进入 Pro 科学裁决。

进度不是科学结果，不需要持续占用模型注意力。`/goal` 保存未完成目标；终端真实计时或唯一后台等待器负责墙钟等待，二者都不把“等待中”误写成完成或科研结果。

本文件是当前仓库的简短上下文锚点。详细研究记忆以 `research-wiki/` 为单一事实源。

## 当前目标

最终研究目标是任务感知动态时序采集：根据视频、窗口、动作区域和难度动态分配帧/片段/Token，减少持续时序冗余，把更有用的信息送入 TAD 检测器，并保护高 IoU 定位性能。

当前科学状态：

- C3/PAction/GAS-VT/lattice 保留为固定预算、no-leak、归因与失败诊断基线；
- BPNS-R1 只用当前观测，在 VideoMAE 前选择连续 `8×8/K64` 原生支持，并让 K64 完整执行全部主干和 Adapter；它不使用历史 hidden/KV、carry 或深度跳过。其单种子准确率可行，但 v004 的真实 full-stack p50 只改善约 `1.51%`，已停止独立效率 headline；
- 四臂只读 full-stack job `1262120` 已在完整 THUMOS14 validation 211 videos/792 ordered windows 上完成 16-pass Williams replay。相对 A/R1-FULL64，B/DSR6-KV、C/MOD32-KV、D/DROP32 的 p50 比值为 `1.112325/1.110211/1.102807`，gross-energy 比值为 `1.102719/1.099873/1.064409`；无 Pareto survivor。fresh Pro 将其裁决为窄范围 claim-grade 负系统结果，永久停止这三个固定点及同底座局部 K/depth/refresh sweep，但不外推整个动态计算家族；
- 当前唯一任务是 `ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`。它只在评测侧为 K100/R1 对称实现 bounded per-video rolling decode reuse，保持模型、checkpoint、数据语义和 decode-to-Soft-NMS 计时边界不变，检验去除重叠窗口重复解码后 R1 是否同时达到 wall-time/energy `<=0.95` 与 memory `<=1.05`；
- DUCA、RC32 carry、当前 APM 载体和若干直接缓存/深度路线保留为历史候选与负证据，不得复活为当前主方法；
- ChronoTransport 与 PhysTime 是独立并行假设，不得与 BPNS-R1 或历史 DUCA 证据混用。

固定 384/768 或 50% 输入只是归因、安全门和失败诊断锚点，不是最终动态采集目标。

## 仓库范围

本仓库只应包含：

- `opentad/` OpenTAD 主库与当前 C3 接入代码；
- `configs/` 中当前 C3 路线所需配置与最小 base config；
- `tools/bata/` 中当前 coarse probe、model matrix、ledger conversion、validator；
- `scripts/` 中当前 N16R4 Slurm GPU 启动器/ watcher；
- focused `tests/`；
- `research-wiki/` 当前研究记忆；
- `README.md`、`AGENTS.md`、本 `RTK.md`。

不要加入旧 tracker、旧 server logs、生成图、检查点、数据集、压缩包、bundle、临时 worktree 或旧路线报告。

## 科学证据与运行边界

- 开始工作前必须读 `research-wiki/query_pack.md` 与 `research-wiki/anti_repetition.md`。
- 科学决策、重要负结果、有效实验结果和论文主张变化写入相应 wiki 节点；原始历史只追加，不以文档润色改变既有证据。长期记忆保持精简，不保存浏览器、队列或一般协调流量。
- 每轮出现实质科学讨论、冻结任务、代码实现、独立审查、正式实验、结果解释或路线裁决时，必须在同轮写入其既有单一事实源，并明确区分讨论、实现、运行与论文证据。无状态变化的实验轮询及浏览器、队列、锁等一般协调信息不形成科研记录。
- 如使用 Project Sources，只保留当前目标/边界、当前研究状态和重要实验历史的最新聚合材料；当轮代码差异、专用报告与临时审查作为本轮输入，不制造版本堆积。
- 不允许 validation/test GT 参与测试时选择。
- 不允许 validation/test teacher leakage。
- 不允许 hidden raw-prediction cache shortcut。
- ledger 若用于 deployable selection，必须记录 no-GT/no-teacher/no-oracle/no-raw-prediction/no-checkpoint flags。
- 可比较的正式训练必须覆盖冻结协议指定的完整官方训练 population、完整 epoch/update 计划和全部样本；子集、截断 epoch、缩短 loader 或 smoke/precheck 只能作为工程证据，不能进入模型准确率、泛化或论文主张。
- 可比较的正式评测必须让所有 arm 覆盖同一个完整官方评测 population，并匹配 annotation、媒体 inventory、loader population/order、evaluator、postprocess 与 NMS。若论文结论要求 official test，则必须在独立冻结且无泄漏的 test-opening 协议下完整运行，不能用 validation 子集或 development replay 冒充。
- 训练或评测的完整 population 不能由终态 receipt 证明时，分类为协议 blocker，不从子集外推科学结论。当前四臂只读成本回放不是训练实验：它固定使用完整 THUMOS14 validation population（211 videos / 792 ordered windows）做 matched full-stack measurement，official test 保持未打开，因此只能形成该 population 上的成本证据，不能升级为 official-test 准确率证据。
- GPU 任务必须使用 Slurm 正常分配的设备；不得固定物理索引或覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`。单卡任务在进程内使用 `cuda:0`。
- 历史文件名中残留的 `gpu0`/`gpu1` 只代表旧协议，不得直接复用；再次运行前必须改成正常 Slurm 映射并重新门禁。
- 不在 N16R4 登录节点直接训练；正式训练使用 Slurm 或已授权保护分配。

### 浏览器配置串行与长时 Pro 存活检查（2026-08-28）

- 同一个 iXBrowser profile 在任一时刻只允许一个浏览器操作。Source 上传、Pro 提交、生成等待和结果回取必须共用一个按稳定 profile ID 命名的独占锁；不同标签页不构成隔离。
- computer-use 只负责创建或进入正确 Project、上传 Source 并确认远端文件名。完成后关闭自己创建的标签页并释放锁，不创建科研对话、不选择模型、不提交科研提示词。
- Oracle 只负责在精确 Project URL 中创建全新对话、选择最高可验证的 Pro 模型、提交一次提示词并保存完整回复。每轮使用独立 `ORACLE_HOME_DIR`、唯一 nonce、Oracle session 与 conversation URL；不得依赖当前聚焦标签页、标签页序号、旧对话或 follow-up。
- Oracle 从提交前路由核验开始一直持有 profile 锁，直到完整回复、session 元数据、conversation URL 和终态报告均已保存。其他进程发现锁存在时只能等待，不得打开标签页、进行浏览器预检查或触碰同一 profile。
- Pro 深度思考超过一小时本身不是失败。若 Oracle 日志连续 30 分钟没有新增输出，只允许当前锁持有者使用 computer-use 对该次精确 conversation 做一次只读屏幕检查，判断页面仍在思考、已经完成或明确报错；不得切换 Project、点击提前回答、发送 follow-up 或重新提交。
- 若页面仍显示正在思考，继续等待原 Oracle invocation；若已经完成，使用原 session 和 conversation 回取；若出现错误或 Project、nonce、conversation 任一绑定无法确认，返回客观阻塞并停止。未知状态下宁可不采信，也不得按内容相似性猜测归属。
- 只有使用不同 iXBrowser profile、独立运行端点和独立 `ORACLE_HOME_DIR` 时才允许并行 Pro 讨论；共享同一 profile 的多个项目必须严格串行。

## 共享 AdaTAD 官方基线（跨项目唯一）

- ZoomToken 是原始 AdaTAD 官方基线的**唯一执行负责人**。所有相关 TAD 项目只能只读引用
  `docs/aris/ADATAD_SHARED_OFFICIAL_BASELINE_PACKET-2026-08-17.md` 及其最终 durable receipt，
  不得各自重复 released-checkpoint evaluation 或从头训练。
- 共享运行必须固定 clean official revision、未改原始 config、canonical THUMOS14 411、预训练或
  released checkpoint、seed、evaluator/NMS、EMA/final 选择、运行时身份和唯一结果根；receipt
  缺任一绑定即不构成共享 baseline。
- 先且只先评测可验证的 released checkpoint；仅在 checkpoint 确实不可得且同一负责人确认需要
  reproduction 时，才执行一次 clean untouched official training。`66.42/67.14/65.99` 是
  matched-source dense，永远不得冒充官方复现。
- 等待共享 dense 数字不能让各项目全面停工：ZoomToken 可继续已接受方法的最小实现、独立审查、
  launcher、checkpoint 恢复与 PRE_RUN 准备；共享数值在 receipt 到位前仅是待绑定输入，不得
  触发方法质量或论文 claim。

## N16R4 环境

远端写入边界：

```bash
~/run/yuzibo
/data/run01/sczc063/yuzibo
```

默认环境：

```bash
BASE=/data/run01/sczc063/yuzibo
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
```

需要下载外部资源时使用登录节点代理：

```bash
export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

远端 GitHub 同步的固定规则：

- 每次在 N16R4 执行 `git clone`、`git fetch`、`git pull`、`git
  ls-remote` 或下载 GitHub release，必须先设置上面的学术加速代理四个
  环境变量；不得先尝试公网直连，也不得在直连失败后才临时切换代理。
- 同步必须先用代理解析远端 branch/commit，再通过同一代理 clone/fetch；
  完成后验证 `git rev-parse HEAD` 等于预期完整 SHA、对应 remote-tracking ref
  指向同一 SHA，且 `git status --porcelain` 为空。
- 学术加速节点不可用时同步 fail closed，保留诊断并重试代理；不得用未提交的
  rsync/scp 源码覆盖来伪造 GitHub 同步。Git bundle 只能传递已经推送且可由
  commit/ref 验证的对象，并在最终代理同步成功后删除。

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/raw/Validation Data/validation
$BASE/raw/Test Data/TH14_test_set_mp4
```

## 常用检查

本地轻量检查：

```bash
python -m py_compile tools/train.py tools/test.py tools/bata/train_lowres_action_probe.py tools/bata/c3_coarse_classifier_model_matrix.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

远端启动前：

```bash
PRECHECK_ONLY=1 bash scripts/run_c3_asformer_delta_ledger_adatad_full_train_gpu1.sh
```

当前本机 Windows Python 的 user-site `torch` 可能加载 `c10.dll` 失败；完整 Torch 相关测试优先在 N16R4 OpenTAD 环境中验证。

## ChronoTransport 动态特征刷新并行路线

ChronoTransport 与 BPNS-R1 相互独立，不做 pre-backbone 删帧。v1 在 48 个 16-frame clip × layer-group 上调度 VideoMAE heavy attention/MLP，保持 patch embedding、AdaTAD temporal adapter、384→768 后处理和 detector head dense。TRANSPORT 必须从 latest cache 递推；正式 learned scheduler 必须使用按硬件、精度、batch、schedule 形状与 selected rows 实测的 p50/p95 cost lookup。

Stage A/B 的 dense reference 与 counterfactual branch 必须同 batch、同增广、同 RNG；ledger 只能保存 compact signal、schedule、cost 与 regret label，不能在推理时查询。所有 deploy、metric、latency 与 paper claim 默认关闭，直到三种子 kill gate 通过。
