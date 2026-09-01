# RTK Project Rules

本文件是当前仓库的简短上下文锚点。详细研究记忆以 `research-wiki/` 为单一事实源。

## 科研语言与上下文忠实原则

- 使用时序动作检测和机器学习论文中广泛接受的术语。首次出现的非公认缩写应给出中文含义和英文全称；除原始代码标识、配置名与历史实验臂名称外，不创造内部缩写、私人状态语言或自造术语。
- 面向外部研究者的文档应让熟悉机器学习论文、但未参与本项目的人第一次阅读即可理解。任务编号、角色队列、浏览器调度、内部状态码和普通工程日志不进入论文叙述。
- 实现、实验、审查和文档必须忠实于当前权威科学状态，不得在转述中改变科学问题、机制、预测、实验目的、公平性、数据划分、指标或论文主张。缺失信息保持未知，冲突信息明确暴露并交由科学判断解决。
- 新文档只是已有科学状态和证据的表达，不产生新事实。代码存在、局部测试通过、作业成功运行、点估计和统计结论必须分别陈述，不得相互替代。

## 最高身份与研究取向（2026-07-27）

- **当前任务执行者首先是论文级模型研究的实现者，而不是以工程整理或版本迭代为目标的维护者。**
- 所有工作首先服务于可发表的科学问题、可证伪的理论假设、具有解释力的模型机制，以及公平且能够裁决假设的实验。官方检测性能能否提高必须由真实实验回答，不能在实验前写成既定结果。
- 代码、配置、版本控制、启动器、日志和审计只是承载模型与证据的最低限度工具，不得成为研究主线或独立目标。除非工程问题会改变模型行为、破坏实验可比性、引入泄漏或阻止真实训练评测，否则不得让它挤占模型分析、优化和实验时间。
- 每轮工作开始前都要先回答：当前动作检验了什么模型假设，会产生什么新的性能或机理信息，如何推进一篇可在顶级计算机视觉会议发表的论文。无法回答时，应停止无信息增益的工程工作，回到模型问题本身。
- DUCA 的优先事项依次是：恢复可信的官方基线；形成统一而纯粹的前置智能选帧理论模型；在公平训练预算下优化官方平均精度和高重叠阈值定位；通过关键消融解释增益来源；验证跨检测器可插拔性与免训练运行方式。版本整理和工程完备度不得替代这些目标。

## DUCA 总计 60 轮论文合同（2026-07-27）

- 公平论文主臂最多使用 6,000 次成功 detector update。K=384 的 `65.385724%` 与
  K=192 的 `57.967272%` 均是 30+60、共 90 轮的历史诊断，不是最终主结果。
- 任何 learned-selector 长训练前，必须先建立 released-weight dense、clean native
  K=384/K=192 uniform 与 clean/wrapper parity，并冻结唯一的
  `e -> p -> F -> y -> S` exact-K 有界时间重参数化合同。
- 纯 pre-backbone 路径在插件接口把 GT 从物理时间映到 warped time；推理时把 detector
  raw proposals 在 NMS 前逆映射回物理时间，再运行算法和参数不变的官方 NMS。不得把
  detector 内部真实时间改造混入纯插件主结果。
- 贡献排序教师和直接检测梯度分别受 `G_rank` 与 `G_direct` 约束。前者失败时删除当前
  连续梯度贡献教师，后者失败时仅删除 direct-gradient 臂；不得通过降低门槛或更换
  test 集补救。
- 第一枚种子只用于 development screening，不进入最终统计。结构冻结后使用未参与开发的
  预登记种子。没有严格训练侧留出集合时统一使用第 6,000 次更新的 terminal EMA；
  有留出集合时所有臂必须共享同一选择规则，official test 不得选择 epoch。
- reviewer 给出的密度界、DP 常数、RDD 公式、性能增益和成本阈值是待验证设计提案，不是
  自动冻结的项目事实。inverse-CDF、DP、课程、日志和版本治理本身都不是论文创新。
- 真正 frozen-detector 的免训练模式与 task-adapted 主模型共用解码器和坐标合同，但分开
  声明；共享合同通过后可并行建立最小基线，不得被遗忘或混入主结果。

## 当前目标

最终研究目标是面向离线时序动作检测（Temporal Action Detection, TAD）的任务感知动态时序采集：根据视频、窗口、动作区域和难度动态分配帧、片段或时序表示的计算量，减少持续时间冗余，并保护高时间交并比（temporal Intersection over Union, tIoU）下的边界定位性能。

当前研究执行状态：

- C3、PAction、GAS-VT 和 lattice 保留为固定预算、无数据泄漏的归因与失败诊断基线；
- DUCA 是离线全窗口、在模型前向过程中即时生成选择的待裁决候选，不是流式在线时序动作检测；
- `70aa069` 和 `a5e1774` 是历史固定 K=384 训练与成本审计锚点，不再代表当前代码身份；当前实现与实验身份必须从对应实验页和原始回执读取；
- DUCA 的长期问题是依据低成本动作性与边界证据分配逐视频或逐窗口动态预算；当前固定 K=384 实验只隔离间接非均匀选帧与物理时间表示，不是最终论文主张；
- 在匹配基线、选择效用、物理时间坐标和端到端成本形成闭环前，不得把 DUCA 称为论文最终方法；
- ChronoTransport 与 PhysTime 是独立并行假设，不得混用 DUCA 结果。

固定 384/768 或 50% 输入只是归因、安全门和失败诊断锚点，不是最终动态采集目标。

## 共享官方 AdaTAD 基线与 DUCA 并行合同（2026-08-17）

- 原始、未修改的 AdaTAD THUMOS14 复现是所有相关时序动作检测项目共享的一次性基线，DUCA 不得重复评测或训练同一官方模型。
- 共享基线作业 `1245842` 已使用官方 revision `01c58b9f...`、seed 42、60 个训练轮次和官方评估器完成，最终平均检测精度（Avg-mAP）为 `68.73`，比论文公开锚点 `69.03` 低 `0.30` 个百分点。DUCA 只读引用该结果；历史 65.xx 或 66.xx 结果不得冒充官方 dense 基线。
- 正式比较还必须绑定该共享结果的原始配置、THUMOS14 规范 411 视频入口、预训练权重、非极大值抑制（Non-Maximum Suppression, NMS）、最终模型或指数移动平均模型的选择规则、运行环境和结果目录。缺少其中任何身份时，只能把 `68.73` 作为共享参考，不能据此计算方法增益。
- DUCA 当前科学合同是：低成本侦察模型学习逐时刻二元动作性与边界重要性，再由确定性采集规则导出帧重要性和物理位置；长期论文问题进一步要求依据聚合语义证据确定逐视频或逐窗口动态预算。小模型直接预测帧索引只能作为消融，固定 K 只能作为归因、对照或回退。
- 所有未来 DUCA 完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；若未修改的官方配置保存更频繁，则保留官方间隔。恢复点不改变预先登记的最终模型或最终指数移动平均模型选择规则，并应保存模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态；至少保留最近三个有效恢复点和预定义里程碑与最终检查点。

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

## 协议规则

- 开始工作前必须读 `research-wiki/query_pack.md` 与 `research-wiki/anti_repetition.md`。
- 新决策、否定路线、实验结果和 claim 变化必须同步更新 wiki 与 append-only `research-wiki/log.md`。
- 不允许 validation/test GT 参与测试时选择。
- 不允许 validation/test teacher leakage。
- 不允许 hidden raw-prediction cache shortcut。
- ledger 若用于 deployable selection，必须记录 no-GT/no-teacher/no-oracle/no-raw-prediction/no-checkpoint flags。
- GPU 任务必须使用 Slurm 正常分配的设备；不得固定物理索引或覆盖 Slurm 的 `CUDA_VISIBLE_DEVICES`。单卡任务在进程内使用 `cuda:0`。
- 历史文件名中残留的 `gpu0`/`gpu1` 只代表旧协议，不得直接复用；再次运行前必须改成正常 Slurm 映射并重新门禁。
- 不在 N16R4 登录节点直接训练；正式训练使用 Slurm 或已授权保护分配。
- 使用网页端外部科学评审时，必须通过当前分配的浏览器配置和运行端点进入指定项目，并在提交前后核对项目、会话和问题文本。不得依赖历史固定端口、当前聚焦标签页或其他项目的会话。
- 浏览器端口、租约、队列、会话标识和路由故障只记录在运行回执中，不写入论文进展或科学结论；外部评审意见也必须经过本地代码与原始证据核验后才能改变项目判断。

## 执行优先级与防延误合同（2026-07-22）

本节用于防止 DUCA/R 系列再次因重复审计、重复记录和过度门禁而延误。根因不是算力，也不是
`monitor-experiment`、`run-experiment` 等执行型 skill；根因是曾把几乎每个小补丁都升级为
“关键版本”，反复执行全量测试、独立 MAX 审核和部署暂停。

以下 R0--R5、U/G、MAX、PASS 和 KILL 是 2026-07-22 历史执行合同中的原始标识，用于追溯当时的任务与证据，不是领域通用术语，也不自动代表当前路线。当前对外文档应改用完整的自然语言说明实验、通过条件和停止条件。

- **性能证据优先。** 用户要求用 mAP 回答核心问题时，关键路径依次是：检查现有作业与可恢复
  产物、提交所有依赖安全的实验、监控、解析原始 mAP、再做非阻塞文档整理。不得让开放式讨论、
  重复可见性认证或无变化状态汇报排在可运行实验之前。
- **实现可并行，运行才受门禁约束。** R0/R1/R2/R3/R4/R5 中彼此不写同一代码面的实现、配置、
  成本统计和测试应提前并行完成；上游证据只通过 `afterok`、fail-closed consumer 或结果校验约束
  下游实际运行，不得用“尚未解锁运行”作为不准备代码的理由。
- **真正关键版本只有三类。** 独立 MAX 审核只在以下版本完整形成后各执行一次：
  1. R0 Oracle/冻结检测器 mAP 实现；
  2. R2/R3 最终可训练模型与 matched U/G0 主实验实现；
  3. R4 合法 hard-swap 对齐与 G1/G2 实现。
  split 字段、哈希路径、journal、aggregate、文档、启动器等小修默认只做相关 focused test，
  除非它们改变模型行为、数据协议或证据真实性，否则不得单独升级为新一轮完整审计门禁。
- **一次集中修复，一次集中复审。** 独立审核返回多个有界 blocker 时，应一次性登记并并行关闭，
  合并成一个候选版本后只复审一次；禁止形成“修一项 -> 新提交 -> 新 MAX -> 再发现一项”的串行循环。
- **Skill 是工具，不是自动工作流。** 性能关键路径默认只按需使用 `run-experiment`、
  `monitor-experiment`、`analyze-results`、`experiment-queue`。`auto-review-loop`、
  `research-review`、`kill-argument`、开放式 Pro 讨论和重复独立 MAX 不得自动成为每个提交的前置步骤。
- **Wiki 只在证据变化时更新。** 仅在出现新设计决策、新代码提交、新 Job、真实失败、新 mAP、
  新成本结果或 claim 裁决时，同轮更新 Wiki 与 `research-wiki/log.md`。状态无变化时不得反复写长记录；
  Wiki 维护不得阻塞提交、监控或结果解析。
- **禁止重复造轮子。** 修改前先查 `research-wiki/duca_model_version_registry.md`、当前 tree 和历史实现；
  已有 selector、decoder、梯度桥、成本统计或启动器能够复用时不得另建同义版本、worktree 或实验套件。
- **模型实现优先于启动工程。** 启动器只保留直接运行、依赖与结果路径所需的最小逻辑；禁止扩张
  通用编排、复杂 schema/journal/router 或形式化框架。时间首先投入会改变 mAP 的粗分类证据、
  边界微簇选择、真实硬选帧、检测梯度回传与真实检测后端。
- **显式截止时间采用时间盒。** 当用户给出四小时等硬截止时间时，前 15 分钟内必须完成作业/产物盘点，
  前 30 分钟内提交所有依赖安全的关键实验或明确唯一物理阻塞；审计与文档并行进行。截止时先提交
  可核验的原始 mAP 表，并明确区分 terminal、diagnostic 和仍在运行的结果，不得用代理损失冒充 mAP，
  也不得因等待非关键审计而空耗 GPU 时间。
- **相同授权与可见性不重复认证。** 用户已授权且仓库、远端身份、分支和路径未变化时，不得再次发起
  同类许可确认或仓库可见性认证；只有访问事实发生变化或真实权限失败时才重新核验。

### R0-R5 四小时生产交付验收

当前四小时时间盒的明确任务是：**R0-R5 生产代码全部完成、真实后端门禁通过、所有正式实验形成
可运行配置并实际提交 Slurm，Job、依赖、精确提交和产物路径完整记录。** 以下条件缺一项都不得称
“完整实现”或“完整部署”：

- R0 必须产生四族真实冻结 detector mAP、逐视频 paired bootstrap、封存 summary 与唯一 family 裁决；
- R1 必须完成生产合同、真实数据/模型 identity、无泄漏、原子提交与 fail-closed 消费链；
- R2 必须以 R0 选中 family 运行真实 coarse/transition/burst P0 训练与 holdout 质量裁决；
- R3 必须使用同一精确协议部署 matched exact-uniform U 与选中 family G0 的 official TAD 正式训练；
- R4 必须实现真实合法 hard-swap、冻结 official detector signed utility/alignment 门禁，并部署真实 G1/G2；
- R5 必须覆盖三种子、K384/K256、主 official AdaTAD、仓库中真实可运行的第二 TAD backend 和完整
  端到端成本测量；所有矩阵项都必须有正式配置、manifest、依赖和产物目录。
- “代码完成”要求生产入口、正式配置、真实 backend、focused/contract tests 和可合并提交全部存在；
  mock、占位 backend、sentinel、仅 `PRECHECK_ONLY`、TODO 或“后续再接入”均属于半成品。
- “门禁通过”要求真实 loader/model/CUDA/backend 路径运行成功；纯静态测试不能替代真实后端门禁。
- “已部署”只在 `sbatch` 已被 Slurm 接受、返回有效 Job ID，并记录 dependency、exact commit、run root、
  manifest/hash 和预期终端产物后成立；仅生成脚本或配置不算部署。
- 并行智能体的文字报告不算交付；其代码必须被父任务检查、集成、测试并进入唯一精确提交。完成的
  关键合并版本只做一次完全独立 MAX 审核，通过后立即部署，禁止再用小修复循环拖延。
- 全部 R0-R5 生产实现与真实后端门禁完成后、任何新的正式 Slurm 实验 DAG 部署前，必须启动一个
  全新、无实现上下文的独立 MAX 审阅智能体。只有审阅通过才允许部署正式实验；不得把审阅拖到结果
  收割之后。审阅主轴是模型设计与机理：是否忠于“粗分类证据间接定位状态转变与边界、Oracle 式
  边界微簇、剩余上下文覆盖、下游 TAD 反馈”的最初目标，实际梯度归属、硬选帧行为和训练推理合同
  是否正确。工程问题只在会改变模型行为、实验真实性或可复现性时作为 blocker，不得纠缠无关细节。

上述规则不取消无数据泄漏、精确代码身份、真实后端和证据可追溯要求；它们只禁止把工程审计本身变成研究关键路径。当重复流程已不再产生新的模型实现、实验结果、机制信息或明确的停止证据时，应基于现有事实结束重复工作并返回最短的性能证据路径；不得为了满足固定次数而继续，也不得因为次数未满而接受错误实现。

## 模型算法优先原则（2026-07-22）

- **第一要务始终是模型性能与算法创新。** 本仓库是研究模型的实验载体，不是需要不断扩张的通用工程平台。时间优先投入到更有解释力的模型假设、能提高官方指标的结构与监督、关键消融及真实成本收益。
- **先做最短的模型闭环。** 模型假设明确后，优先完成“最小端到端实现 -> matched baseline -> 官方 mAP/高 tIoU/真实成本 -> 机理诊断”；不得用长期协议雕琢、框架抽象或辅助工具建设代替模型训练和性能裁决。
- **工程只做到实验正确可运行。** 启动器、manifest、日志、验证器、统计和可视化只保留支撑正确训练、公平测评、基本复现与论文声明所必需的最小功能；优先复用官方组件和已有实现，禁止为形式完整另建同义框架。
- **仅四类工程问题可以阻塞模型实验。** 会改变模型行为、造成数据/GT 泄漏、破坏指标或成本真实性、或使训练评测无法运行的问题必须修复；代码风格、通用抽象、额外 schema、日志美化、重复审计和非必要完备性不得阻塞模型优化与 Slurm 投递。
- **评价工作价值以模型信息增益为准。** 优先回答“性能是否提高、为什么提高、创新点是否成立、成本是否真实下降”；仅增加工程完整度而不产生新模型假设、有效实验、官方指标、机制结论或明确 KILL 决策的工作应停止。

## 论文实验优先与 Pro 科研闭环（2026-08-27）

- **任务必须面向论文问题和决定性实验。** 路线探索、模型代码、完整训练、官方评测与结果归因是主线；不得把复杂合同、通用框架、防御性工具、版图整理或重复审计扩张为研究目标。
- **工程只走最短必要路径。** 只有会改变模型行为、破坏数据合法性/公平比较/指标真实性，或确实阻止训练评测的问题才可阻塞实验；其余工程问题记录后并行处理，不得终止科学路线。
- **实现失败不等于科学失败。** 启动器、环境、收据、封存或实现包缺陷只能触发有界的最小修复；只有真实科学证据和明确的 Pro 科学裁决可以停止或切换路线。路线转换前必须完成失败根因、混杂因素和可证伪结论的闭环分析。
- **高能力外部科学评审模型（下称 Pro）负责前后两端的科研裁决。** 实现前，Pro 明确当前科学问题、最小改动、决定性实验与截止时间；正式代码实现和完整训练或评测后，Pro 再审阅实现忠实度、成功或失败根因、论文可发表性、是否达到当前问题的结论及下一路线。每次讨论都必须同步原始证据与未知项，并在同轮把科学裁决和证据边界写回研究记忆。
- 分角色职责和最小同步格式见 `research-wiki/PRO_RESEARCH_ROLE_RULES.md`；具体角色任务与验收顺序可由 Pro 在每轮规划中直接指定，但不得演化为新的工程门禁系统。

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

### Slurm shell bootstrap

- A standalone Slurm script must execute `source /etc/profile` **before**
  `set -u` and before every `module load`. Slurm batch shells are
  non-interactive and do not otherwise define `module`.
- A failure at this point is a zero-update launch-environment error. Repair
  the one wrapper line and resubmit the exact same model commit; do not create
  a new model revision, new gate, or a performance conclusion.

需要下载已获准的外部资源时，从个人环境加载代理配置。代理地址和凭据不得写入仓库：

```bash
export http_proxy='<authorized-proxy-url>'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/thumos14/raw_data/video
$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
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

ChronoTransport 与 C3/DUCA 并行存在，不做 pre-backbone 删帧。v1 在 48 个 16-frame clip × layer-group 上调度 VideoMAE heavy attention/MLP，保持 patch embedding、AdaTAD temporal adapter、384→768 后处理和 detector head dense。TRANSPORT 必须从 latest cache 递推；正式 learned scheduler 必须使用按硬件、精度、batch、schedule 形状与 selected rows 实测的 p50/p95 cost lookup。

Stage A/B 的 dense reference 与 counterfactual branch 必须同 batch、同增广、同 RNG；ledger 只能保存 compact signal、schedule、cost 与 regret label，不能在推理时查询。所有 deploy、metric、latency 与 paper claim 默认关闭，直到三种子 kill gate 通过。
