# GeoSparse-TAD：完整科研论证与 Agents 任务命令 v3

本文件含完整研究计划、实验/统计/绘图协议、总控与12角色指令。机器可读核心请求在同名ZIP的manifests目录。

<!-- source: README.zh.md -->

# GeoSparse-TAD 三路线科研与创新性证据包 v3

## 这是什么

一套供实现agents使用的**科研设计、最小新方法规格、强竞争对照、实验工单、统计/绘图协议和可执行请求编译工具**。不是三个新网络的已完成实现，不含训练结果，不含任何40/67伪数据。

首先交给总控：`MASTER_AGENT_COMMAND.zh.md`。
完整单文件说明：上级交付文件 `GeoSparse_TAD_Research_and_Agent_Commands_v3.md`，包内也有同内容 `COMPLETE_COMMANDS.zh.md`。

## 文件结构

- protocol/RESEARCH_SPEC.zh.md：A/B/C问题、最小新增方法、反事实、强基线和失败条件。
- protocol/STATISTICS_AND_EXECUTION.zh.md：split、60epoch/3seed、成本/因果/统计/并行协议。
- protocol/FIGURES_AND_PAPER.zh.md：22项图件和6张表，全部原始数据与caption规范。
- protocol/IMPLEMENTATION_API.zh.md：agent要实现的backend、plan、counterfactual、receipt契约。
- agents/：12个可分别派发的角色指令。
- configs/research_design.json：37个命名方案与30个科研工单，主比较及预算显式声明。
- manifests/run_requests.jsonl：345个训练请求、345个评估请求、115个硬件请求；这是核心805项，不是已经启动。
- manifests/research_tasks.jsonl：30个科研工单；包含明确的子控制要求，由agent绑定真实配置后形成amendment。
- manifests/source_run_index.json：实际读取旧manifest中的609个训练任务，只用于候选核对，不自动认定可复用。
- tools/compile_plan.py：确定性请求编译器，**不是训练器，也不是远程/多agent调度器**。
- tools/research_checks.py、validate_receipt.py：少量MAC/support/收据结构检查，不验证真实模型准确率。
- tests/：13项工具级测试，已通过；不代表TAD前向/梯度测试通过。
- sources/：经本轮检索的主要一手文献。

## 如何使用

```bash
# 编译只登记，不运行模型
python tools/compile_plan.py \
  --previous-manifest /ABS/PATH/old/experiments.jsonl \
  --out manifests_local

# 工具单元测试
python -m unittest discover -s tests -v

# 真实结果到达后验证收据字段与文件hash
python tools/validate_receipt.py \
  --receipt /ABS/PATH/run/receipt.json \
  --evidence-root /ABS/PATH/run
```

实际模型入口须由agents在真实仓库实现，见IMPLEMENTATION_API。所有train_request初始为NEEDS_RESOLUTION：还要绑定source config、数据/权重hash、正确性和实际实现。不要把科学配置中的REQUIRED占位解释成可忽略字段。

## 与旧执行包关系

本包追加SCI3命名空间，不更写旧1538项。旧JSONL SHA256已保存summary，原任务ID来自真实文件读取，不根据训练表猜造。source_run_index里“候选”不表示配置、split或源码相同；复用率当前是0（未审计运行成果）。

## 工作量与并行

30个工单围绕证明链组织，不是盲目把所有变量做全笛卡尔积。核心345个train requests含discovery/confirmatory/transfer、固定/动态预算和control；每个都需要实际资源。首次profiling后估算总GPU-hours并使用用户已授权资源，不擅自租赁。物理排队、资产和自己的checkpoint可以阻塞；mAP/Oracle/seed0是否成功不能阻塞另一条路线。

所有slots/resolution/额外训练时长等子控制先按工单形成精确amendment再启动；不能误称805项覆盖了任何可能的新参数组合。若资源需要调整，登记预算修订而不按有利结果筛选。

## 状态

当前完成：研究规格、agents命令、核心工作清单、编译器、工具级测试。
当前未完成：接入用户源码；A/B/C及近邻新增实现；任何模型训练；任何实测科研图；论文最终结果。

用户静态核对版本340a3541为锚点，不代表本包作者已访问该Windows仓库。预设40% Heavy MAC/67.0 mAP禁止作为实测或成功解锁条件。


---

<!-- source: MASTER_AGENT_COMMAND.zh.md -->

# 可以直接转交实现 agents 的总命令

你是 GeoSparse-TAD 三路线科研项目的总控研究/工程 agent。任务不是润色创新点，而是通过强基线、实际代码、成对干预和真实硬件实验，判定 A/B/C 各自是否具有独立、可迁移的学术贡献。

## 1. 必须读的材料

完整读取本包的 `protocol/RESEARCH_SPEC.zh.md`、`protocol/STATISTICS_AND_EXECUTION.zh.md`、`protocol/FIGURES_AND_PAPER.zh.md`、`configs/research_design.json`、`manifests/research_tasks.jsonl` 和 `manifests/run_requests.jsonl`，以及相应角色的 `agents/*.md`。这些材料是统一契约，不得只看摘要后自行补全。

同时只读核验真实仓库。用户提供的基线锚点是 `GeoSparse_TAD_20260907@340a3541`，C3为另一分支。你必须核对实际commit、配置、source AdaTAD算子、数据和权重，不以网络图或旧PPT作为实现证明。

## 2. 并行执行，不以科研结果解锁

平台有真实并行agent工具时，同时委派01–11；平台没有时不得声称已spawn。按隔离worktree和明确文件所有权开发。接口快速冻结后，A、B、C、反事实引擎、评价、绘图、硬件和统计同时推进。

禁止“Oracle好才写模型”“A有效才做B/C”“seed0高才补seed1/2”。全部预注册任务立即登记。代码/资产/正确性/自身checkpoint/实际资源是唯一合法依赖。最终统计需要所有预注册seed齐全不构成性能门槛。单个分支阻塞不冻结其他分支。

Discovery、Confirmatory和Transfer使用预先锁定的数据划分与配置，默认任务可并行训练。探索结果触发的修改必须新增amendment和ID，不能覆盖确认性任务。禁止test标签调Router阈值、loss权重、样本挑选或超参数。

## 3. 三路线的科研责任

A：承认与MoD/CoDA相近。实现强MoD+同一TIA、CoDA-TAD、LITE-TAD及完整Router×Risk×CF因子。检查定位价值监督是否有增益、能否迁移给MoD；不是靠删除对手TIA/位置编码获胜。

B：承认与Coarse-Fine/mTAN/Perceiver/LookupViT相近。固定实际evidence，比较带完整metadata的strong time-attention、mTAN和provenance gate；检查gap、同类粘连与召回。不要把support之外的预测自动算错，不要把坐标槽说成ROI获取。

C：承认与MSViT/ToMeSD/TokenFuser/ALGM相近。实现成熟merge-process-unmerge对照，研究相似性和真实refinement收益的关系；随机干预计量模式切换，保留matched error和miss。不是把共享residual性质说成新的无损理论。

任何新方法先保留最小实现：A的等成本CF监督；B的provenance-conditioned残差gate；C的refinement价值与coarse-affine校准。所有消融已登记并行运行。最终无独立贡献的模块从主模型删去，但其原始实验不删除。

## 4. 实际代码要求

新增独立科研扩展命名空间，保留旧版current模型。建议实现下列接口（命名是本任务要求，不代表仓库已经存在）：

- `NativePlan`：native IDs、parent/layer分组、mode、actual supports、member mapping、valid mask、序列化hash。
- `forward_with_plan(video, plan)`：真实压缩的Heavy执行；不得dense后乘mask。
- `collect_query_risk(outputs, gt, frozen_assignment)`：R0/R1、cls/loc/bg分解。
- `CounterfactualRunner.evaluate_pair(plan_a, plan_b)`：同checkpoint/输入/随机状态/normalizer；检查成本；重跑受影响路径；返回原始对照记录。
- `EvidenceBank`：明确固定编码计划及feature/query hashes；变更获取集合则失效。
- `CostTrace`：每layer-parent的实际QKV/MLP长度、padding、MAC、主机与GPU时间。
- `evaluate_official`、`evaluate_slices`、`paired_video_bootstrap`：原始预测到官方指标及风险分析。

每个新增模块的参数必须在实际forward中被使用，并通过变更参数改变输出/梯度的测试；inspect.signature存在不等于参数生效。未知字段fail-fast，注册/数据/head/optimizer命名空间明确映射，不准**kwargs静默吞掉。

数值测试：K0/Kall、A/C同权重dense极限、B空evidence、native pairing/PE、parent边界、support union、coarse成员回写、frozen权重下上游梯度、真实检测反传。测试存在不等于通过。c10.dll只在隔离环境诊断，不私自修改用户全局环境。

## 5. 反事实和统计要求

Addition、retention、swap、refinement分开。固定预算优先同parent等cost swap；不能把末端feature ablation冒充原始计算变化。Attention/TIA受依赖改变时必须fresh forward。与test无关的训练probe可以用当前模型，不引入独立Dense Teacher，但其成本必须记账。

预设40/67不得进入任何实测表。finite-menu loss-best不是mAP理论上界。attention热力不是utility。计算切换和错误边界接近不是因果证明。按video聚类而非把tokens当独立样本；mAP bootstrap必须重建整份数据，不能平均per-video AP。漏检必须始终保留。

## 6. 任务编译与旧矩阵

运行：
`python tools/compile_plan.py --previous-manifest <真实旧experiments.jsonl> --out manifests_local`

该脚本只生成科研请求，不是现有模型训练器。将request绑定真实源配置、源码snapshot、数据和权重hash后生成runnable manifest；所有NEEDS_RESOLUTION请求禁止直接训练。

旧1538项manifest保持原样。新request有SCI3命名空间。旧source_run_index只提供真实候选ID，完整配置/数据/seed/commit/成本口径一致才可复用。没有旧JSONL时不猜原任务ID。

清单中805项是核心train/eval/benchmark请求，不代表已执行。30份科研工单还要求按明示的slots/resolution/FFN/paired-intervention等参数网格解析子控制，先形成精确amendment再启动；不能把未编译的子控制默默算成已完成核心实验。所有子控制也不得设性能解锁门槛。

## 7. 图表与论文

按22个figure_id及6张表生成真实raw data、SVG/PDF/PNG、脚本、caption和receipt。绘图agent可提前完成schema/空轴测试，但正式图只能消费MEASURED；解析关系和合成诊断单独标记。不得生成虚构视频帧、fake heatmap或合理数值。

至少给出每条路线的一张强近邻主性能图、一张成对机制图、一张失败/风险图；总论文给出跨数据与真实latency。每条“更好”“更稳健”“定位更准确”句子必须绑定claim ledger、原始数据和有效统计。

## 8. 交付状态而非空口完成

每次报告：request/experiment ID、代码commit、resolved config、数据/权重/checkpoint hash、真实状态、完成产物、复现命令、失败原因、下一项实际动作。只建类/文件/图框不算完成。所有任务最终状态为SUCCEEDED、FAILED、BLOCKED_ASSET、BLOCKED_IMPLEMENTATION或INFEASIBLE；不因负结果而消失。

你不得自动租算力、删除数据、覆盖用户原实现、杀其他用户进程。资源优先级可按覆盖面设置，但不能按有利结果筛选。科研成功的标准是排除替代解释并给出可信结论，不是凑齐预期数值。


---

<!-- source: protocol/RESEARCH_SPEC.zh.md -->

# GeoSparse-TAD：三路线科研、实现与证据论证规范 v3

**交付性质：预注册研究方案与 agents 实施命令，不是已实现新算法或已获得实验结果。**
**来源边界：**现有模型描述来自用户对 `GeoSparse_TAD_20260907@340a3541` 的静态审查；本包未访问该本机仓库，也未运行真实 TAD 模型。旧版 40% Heavy MACs / 67.0 mAP 是演示预设，禁止进入任何实测结果、回归、显著性检验或结论。

## 0. 研究目标与可失败的主张

统一问题：在固定真实成本下，哪些重型计算对**动作区间的分类、双端边界和实例完整性**是必要的？

- A：计算位置与受益位置不必相同。研究显式、等成本的区间价值监督是否比强 MoD+TIA / CoDA / LITE-TAD 路由更可靠。
- B：稀疏高层语义可帮助上下文，但不应自动被解释为当地持续动作已被观察。研究观测来源、支持集合与廉价当地状态如何共同减少错误外推。
- C：外观相似不保证共享更新对定位无害。研究 coarse→fine 的任务收益，以及计算尺度切换造成的时间近似误差。

这些是**待检验假设**，不是已经发现的事实。保留完整位置、gather/scatter、cross-attention、merge/unmerge、轻重双分支本身均有已有方法覆盖，不能单独声称首次提出。

### 0.1 论文贡献的证据层级

L0：代码正确，所有路径真正执行；L1：失败现象在真实数据存在；L2：控制干预支持机制解释；L3：最小方法改善该机制；L4：击败强近邻和简单替代；L5：跨数据/预算/硬件可迁移。

只有 L0 不是科研贡献；只有 L1 的相关性不是机制证明；只有 Avg-mAP 提升不能支持边界故事；只有 FLOPs 下降不能支持端到端加速。允许论文最终成为可迁移的训练原则，而不是强行包装三种全新骨架。

## 1. 实现锚点：先只读核验，不以图片替代源码

### 1.1 版本与范围

基线锚点是用户报告的 `340a3541`。不得混用 C3 分支；agent 必须核验本地 HEAD、祖先关系、dirty diff、配置及运行环境，输出 source_audit.json。文件路径、行号由真实仓库读取，不凭本文补造。

输入默认是 768 个采样位置、224×224。原生 tubelet 为 2 输入位置，得到 384×14×14 token 网格；16 输入位置为 parent，包含 8×14×14 tokens。2×2 空间选择组对应 7×7 原生组；不要再次把生成图中的 384×28×28 当作真实原子网格。Scout 自身特征尺寸须 trace，不根据图推断。

### 1.2 A 不变量

Scout 直接读原始低分辨率帧；空间 DW/PW Conv2d、时间 Conv1d、atom pooling、gain 输出。每层实际流程按源 residual/LN 运算顺序实现 selected Heavy→scatter→source dense TIA；未选位置仅绕过 Heavy，后续仍接受 TIA。Heavy attention 不跨 parent。Dense PatchEmbed 和 dense state 仍付费。

### 1.3 B 不变量

所有 parent 仍 dense PatchEmbed；selected tokens 执行 Heavy；Heavy 内无 TIA。稠密 query 来自 cheap stream。当前 support_attention 为 query norm+cross-attention+residual，无 query FFN。receiver 后 regular TIA。空间槽坐标不是已实现 ROI，当前 roi_mode=none。M=K_time×slots_per_time，可以大于 Q；例 192×4=768 而 Q=384。

### 1.4 C 不变量

固定 2×2 原生组；每层从完整 state 重建 1 coarse 或 4 fine；coarse residual 加回四成员，fine 写回成员输出；使用源 block.adapter。不是一次构造的持久 mixed sequence，不存在新的独立 geometry-TIA。全 coarse 仍 Heavy；空间 token 数下限 49/196，不代表总 MACs/延迟下限 1/4。

### 1.5 共用检测接口

A/C 最终空间平均、插值至 Q=384/768，传入 [B,C,Q]；实际多尺度来自 Conv1DTransformerProj，之后 FPNIdentity 和 ActionFormerHead。source adapter 位置与 residual 次序须真实 trace。冻结权重不等于 no_grad；对上游可训练状态的梯度仍可能经过 frozen Heavy。

## 2. 三类证据必须分开获取

### 2.1 同一模型内的成对干预（paired intervention）

固定视频、GT、checkpoint、weights、receiver、随机增强、dropout状态、loss normalizer和合法预算，只更改明确的计算计划/支持元数据。目标是估计**当前模型**对此操作的条件响应，不是普遍因果定律。

### 2.2 端到端训练比较

不同方法用相同源权重、数据、epochs、effective batch、head、冻结范围、超参数搜索额度训练。不能用只在推理时强行删 token 的未适配模型证明对方原则错误。

### 2.3 实际部署成本比较

计入解码、Scout、Dense PatchEmbed、选择/排序、pack、Heavy、write-back、TIA、receiver、head和postprocessing。训练 probe 的额外时间单列。论文主性能与机制诊断都须匹配真实预算，不把 token 比例等同于成本比例。

## 3. 统一数学对象与反事实引擎

### 3.1 计算操作而非抽象“重要帧”

Plan 包含 native atom IDs、parent IDs、层使用范围、coarse/fine 模式、实际时间支持、空间成员、valid mask和执行布局。A 的一次操作可能作用于多个层；B 的 acquisition 改变 Heavy 输入集合；C 的 refinement 改变组内表示粒度。分别实现，不用一个末端 feature mask 冒充全部操作。

### 3.2 三种真实目标

addition: u_add(a|S)=R(S)-R(S∪{a})。
retention: u_keep(a|S)=R(S\{a})-R(S)。
swap: u_swap(a←b|S)=R(S)-R((S\{b})∪{a})。
C 的 refinement: u_ref(g|S)=R(S,g=coarse)-R(S,g=fine)。

保留符号；不要取绝对值后仍称任务收益。固定预算主要验证等成本 swap，动态预算使用 addition 与实际边际成本。以上值都依赖 checkpoint、当前集合和风险函数。

### 3.3 风险 R0 与 R1

R0=源检测损失，须暴露每个 query、各分类/回归项及正常化常量。
R1=实例平衡的诊断/Router目标：每个 GT 实例对其源 assigner 的有效正样本先平均，再对实例平均；背景项独立平均。主 detector 先保持 R0 不变，避免一开始就同时改 detector loss 和 Router目标。

R1 = mean_j mean_{q∈P_j}[l_cls(q,j)+β l_loc(q,j)] + γ mean_{q∈N}l_bg(q)。

β/γ继承并记录源权重；若调整，使用同额搜索并同步提供给强基线。固定 GT-based assignment；重叠动作沿用源 assigner，不用不同预测集合重新匹配来制造收益。无正样本实例必须记录，不可静默丢弃；若额外 nearest-positive assignment 需作为明确新变体。纯背景窗口定义 R1 为背景风险。反事实 pair 中 loss normalizer不随 plan 改变。

归一化 start/end 误差用于诊断；不要把训练 loss-best 称 mAP-best。mAP是数据集排序指标，不能对每视频 AP 平均后称官方 mAP。

### 3.4 成本匹配

对 parent p、层 l，MLP ratio=4 的 Heavy 近似 MACs:
C_H = Σ_l Σ_p [12 k_lp d_l² + 2 k_lp² d_l]。
若增加 s 个 token，ΔC=12sd²+2d(2k_ps+s²)。相同 Σk 不保证相同 Σk²。

同 parent、同原子大小、同层范围的 swap 首先用于精确控制；跨 parent 时按真实成本匹配。标称目标 Heavy ratio 为 0.30/0.40/0.60，实际配额由 geometry-only cost resolver 确定，不用 test accuracy 调配。容差默认±2%目标成本，相对误差超出则报告实际点，不假装“等成本”。执行时间不能用解析MACs代替。

### 3.5 反事实路径和缓存

改变了 attention 输入的 query/key/value集合，就需重跑受影响 parent；改变早层状态会影响后续 TIA 和未来所有依赖位置。A/C默认完整受影响后缀重算。B可以缓存**同一获取集合**的 evidence用于 receiver-only 对照；改变 acquisition 集合后不能盲用缓存。

训练 probe 在同一次迭代的固定参数快照上计算，不更新 BN/EMA，不将评估标签反向用于 test-time selection。可以在当前训练模型内做 stop-grad 反事实，这不是独立 Dense Teacher；但确实产生了额外训练成本。

### 3.6 随机性与有效样本

成对干预默认 eval/dropout off，完全相同 tensor/data hash。需要训练态估计时使用 common random numbers 并记录随机状态；KL/utility 自身改变输出匹配不应与 dropout 噪声混淆。有限差分重复同计划作为 noise floor。

## 4. A：定位价值学习，而非重命名 MoD

### 4.1 H-A

在相同 sparse executor、parent、dense TIA、预训练和成本下，显式等成本的区间收益监督能否优于强端到端 MoD/CoDA/LITE-TAD？收益是否来自监督，而非前置 Scout 信息或额外训练成本？

### 4.2 强基线

同时实现原始风格 MoD linear router+TIA、contextual/grouped MoD+TIA、CoDA-TAD、LITE-inspired TAD selector、uniform/random/motion/actionness。MoD必须保留合法残差完整位置、正确PE/parent和同样TIA，不能人为删掉对方时间建模。原论文 gated residual和交错dense层属于原机制，独立复现；统一executor比较则明确标为 controlled adaptation。

LITE若官方仓库只有README，按论文重实现并标 paper-based adaptation；不要声称官方完整复现。TAD训练目标也给LITE，不拿分类目标弱迁移当唯一对手。

### 4.3 最小新方法 A-ECV（Equal-Cost Value；内部实验名，非已确认新颖名称）

只在已有 A 上新增/替换 Router 监督：硬策略优化R0或R1；每32个minibatch抽一个等成本swap，在同模型计算 signed outcome，Huber拟合预测差 u_hat(a)-u_hat(b)。正负目标都保留。默认一次预先计划跨层使用，逐层refresh作为独立消融。

在线CF权重默认1.0，target scale仅用训练池固定/慢EMA全局尺度并stop-grad，不逐视频z-score（避免破坏动态预算）。0–5 epochs random预算训练；6–20探索由0.5降至0.1；21–59保留0.1。严格作为预注册起点，真实源实现若不同，旧版current保留不覆盖。

### 4.4 必做因子

Router: Scout / contextual MoD。
Risk: R0 / R1。
CF: off / on。
构成2×2×2完整实验。额外原始MoD、CoDA、LITE为邻近对照。Gate scaling、router容量和训练额外步数作为匹配控制；等epochs与等训练GPU-hour分别报告，额外probe成本不隐藏。

### 4.5 A关键诊断

A-E1：候选 ΔL_cls 与 ΔL_loc 散点，区分核心、边界、上下文；不要预设边界永远最高价值。
A-E2：真实 u_swap 对预测值、Spearman、sign AUROC/precision、成本匹配regret；按video聚合，再估不确定性。
A-E3：计算位置 a→受益query q 的风险变化图；这不是attention heatmap。
A-E4：基线MoD加入相同R1/CF后是否也改善；若成立，故事转为可迁移训练原则。
A-E5：固定gate任务图 vs 每层refresh，比较收益与额外延迟，不预设一次规划更优。
A-E6：Scout分辨率/信息可预测性；同候选菜单比较cheap、深层信息router和finite-menu loss-best，信息Oracle单列。

### 4.6 失败判据

若R1-only即可解释所有提升，不宣称CF新机制。若A与MoD在同监督后相当，不宣称骨架优越。若只有低tIoU改善，不以高精度定位为主贡献。若收益不足抵偿训练开销，报告训练成本局限。

## 5. B：不完整证据的可用范围，而非通用 cross-attention

### 5.1 H-B

即便给标准receiver完整时间编码、valid mask和null，它是否仍会将稀疏上下文错误外推为连续动作？实际支持/来源校准能否在固定evidence下减少同类动作粘连、gap附近过延伸，同时不损伤未Heavy区域的真实召回？

### 5.2 明確三个支持

anchor_support：slot锚定的native tubelet/空间位置。
source_support：实际输入观测并集，保留不连续区间。
context_support：按编码计算图可达的观测范围；是潜在依赖，不是因果解释或attention权重等价物。

parent内全局attention后slot可能共享context_support。不得给每个slot伪造独占的局部观测；若support全部饱和且相同，要报告没有额外可辨信息。

禁止以“预测区间落在heavy support之外”直接定义错误，因为cheap branch仍观察当地。GT和标准匹配才定义检测错误。

### 5.3 强receiver控制

同一 evidence bank、相同cheap query容量与选择分布：physical interpolation+projection、时间attention（允许FFN/gate）、mTAN-style content/time、同样support元数据的普通attention、provenance-gated版本。允许基线利用同样metadata/参数数量，避免“给我们更多输入”的伪创新。

另外注册Coarse-Fine/LookupViT/TokenFuser的合理TAD adaptation，分别标注native公开协议和统一源VideoMAE协议；权重不能无根据移植时记录结构和初始化差异。

### 5.4 最小新方法 B-PGR（Provenance-Gated Residual；实验名）

在已有support receiver外仅加入一个小型residual gate。b(·)输入anchor距离、到actual support的最小距离、固定物理窗口内coverage fraction、有效槽和重复ID；gate输入本地cheap query以及同一组统计，允许保留不更新。

z_q = q + sigmoid(g([q, phi_q])) × Attn(q,E; b(anchor,source,context))。

phi不包含GT boundary/duration，物理窗口从Q的spacing和预注册尺度2/4/8构造。null/empty正确旁路。普通content-gate和参数匹配time-gate作为控制；各方都可收到完整metadata。

默认不加入分类/边界双head；作为已注册可选variant，在实验矩阵中与simple gate并行，不依据base表现好坏决定是否运行。最终没有独立收益就不纳入论文主方法。

### 5.5 B关键干预

B-E1：anchor-only / hull / exact-union；metadata corruption仅诊断，不作为强性能基线。
B-E2：同一raw video、固定budget/parent-count，gap位置随机变化；按实际gap/GT duration分箱。重新编码获取集合，不从旧evidence直接删除冒充减少计算。
B-E3：同类相邻GT实例之间gap bridge。配对指标：预测与两实例各重叠至少50%并覆盖中间gap至少80%；阈值在研究协议冻结，另做敏感性。不称官方指标。
B-E4：video内容不变，重排packed storage同时保留metadata应近数值不变；同时错改metadata是negative control，两者分开。
B-E5：metadata rich baseline vs proposed；重新训练适配与同权重metadata干预分开报告。
B-E6：receiver×selector交叉：uniform/random/LITE-TAD/ECV至少在固定bank上评估；row/column效应区分“选得好”“用得好”。
B-E7：M/Q、slots1/4/8、receiver层数1/2、无FFN/FFN成本；不能声称M<<Q。
B-E8：标定cheap strength，防止提升只来自更强coarse分支。

### 5.6 失败判据

如果time-attention+gate+同metadata已经等效，就没有必要宣称新receiver。若只胜过hull/rank错误版本，只能说明修复几何bug。若少粘连但漏掉更多短动作，需要同时报告recall，不能选择性讲优势。

## 6. C：任务允许的共享更新，与计算模式引入的时间误差

### 6.1 H-C

当前token相似度不能充分预测共享Heavy更新对区间风险的影响；粗细切换可能造成随时间突变的近似误差。研究真实refinement价值与最小尺度校准是否能保留短动作/边界，且优于成熟merge-unmerge基线。

### 6.2 不能冒充新颖的部分

ToMeSD已有merge→heavy→unmerge→skip；MSViT已有coarse/fine；TokenFuser支持返回dense；ALGM用于dense segmentation。当前C的共享delta保持同组原差值不是独有定理，更不保证信息无损。StructSAM/CubistMerge在近邻审查与扩展实验中登记。

### 6.3 可检验的数学关系

令完整参考表示z_t、混合表示z_t+e_t，则时间差分误差是(e_{t+1}-e_t)/Δt。该恒等式只提示应测时间误差；不是检测边界误差的上界，也不证明head是差分算子。

同模型all-fine可作**诊断参考**，但不是GT或通用上界。不能把该参考偷偷用作独立Dense Teacher训练。

### 6.4 最小新方法 C-TR（Task-conditioned Refinement；实验名）

固定2×2和原生回写，不重写整个backbone。Router预测真实coarse→fine的R0/R1差；固定budget使用同parent一组升fine、另一组降coarse的等成本swap。

额外最小校准：仅对coarse delta使用逐通道 a_c⊙δ_c+b_c，初始a=1,b=0，fine路径保持源算子。这样all-fine结构极限可保留。单独generic affine-only、utility-only、utility+affine作完整因子，参数匹配校准也提供给ToMeSD-style对照。

可选实验C-PC：同一模型、同一视频两种合法等成本计划，各自接受GT检测监督；在双方都具有有效正样本的query上加入输出一致性。系数默认0.1、mask和loss在实现前锁定；不得把正常视频沿时间平滑，也不得只保留容易样本评估。它是同模型在线配对训练，不是独立dense teacher；双forward成本记账。没有收益就从最终简化模型删除，但任务仍执行。

### 6.5 C关键干预

C-E1：cosine相似性/variance/motion对u_ref的相关性、负值、regret；与LITE-type梯度代理比较。
C-E2：固定视频与每parent token数；随机生成低切换/高切换粗细计划，尽量固定每时间fine数量，并平衡每空间组的边际被选概率。用多次随机耦合估policy干预效应；单一高低mask也改变了选中位置，不能把所有差异都归因为switch本身。
C-E3：记录层级e_t、Δe_t、预测start/end漂移、miss和伪边界，分开相关性与随机干预结果。
C-E4：静态重复帧仅用于无内容变化的诊断，必须标合成；不能沿用原动作GT报告mAP。
C-E5：coarse residual广播 vs直接覆盖、persistent merge vs每层重建、calibrator有无；所有差异明示。
C-E6：全coarse/all-fine/固定random/learned，比较真实MACs和移动成本。
C-E7：ToMeSD attention-only和attention+MLP都测；ALGM/CubistMerge按parent与native时间关系适配，不让对方非法跨parent。

### 6.6 失败判据

若affine-only解释全部提升，故事是尺度校准，不是utility。若高低switch干预在平衡控制后无差异，不继续强调伪边界机制。若相似性已接近finite-menu最佳，任务价值预测没有足够增量。若仅MACs改善、merge/unmerge导致更慢，报告系统局限。

## 7. 共同强控制

1. Dense AdaTAD、A-all、B-all、C-all-fine各自训练并比较；不能拿B的结构损失混入采样损失。A/C结构全量极限做同权重输出/梯度测试。
2. 固定浅层4/6/8/10或原已注册静态depth基线，teacher-free PBD-inspired不得冒充原论文完整复现。
3. uniform/random保相同geometry、context、cost；不要给baseline错误rank位置或少训练。
4. 动态budget vs最佳固定点 vs预算直方图shuffle vs长度分层shuffle；shuffle只打乱预算，再用该视频自己的选择分数选位置。
5. mask-only、actual-evidence、zero-content、random-same-statistics等净内容贡献诊断；mask本身由输入生成不是自动标签泄漏。
6. 3D/low-res/ROI信息能力的扩展要区分observability，当前roi_mode=none维持，真实ROI新variant和资产接口单列。

## 8. 数据、统计、资源与实验纪律

详见 STATISTICS_AND_EXECUTION.zh.md。研究探索和最终验证分开；每条路线所有预注册variant同时登记，不设置性能gate。数据/代码/正确性/自己的checkpoint/资源才构成依赖。

## 9. 完成的定义

每条主张至少有：最近竞争方法、可复现实验、控制与干预、实际代码、真实图表、负结果解释、适用范围。缺数据标PENDING；不允许用预设40/67或合成漂亮曲线填主图。

若三路线最终只有一种机制有独立收益，论文主方法收敛到它；其余作为强对照或补充。若相同监督提升多个骨架，优先叙述可迁移原则。任务的科学价值不是跑出预期结论，而是区分替代解释。


---

<!-- source: protocol/STATISTICS_AND_EXECUTION.zh.md -->

# 统计、可比性、并行调度与证据纪律

## 1. 两条并行研究轨，而不是性能解锁链

- **Discovery**：只使用源协议的官方training pool，按video ID分组得到fit80%、probe10%、select10%，分组seed=1701；稀有类冲突需记录，不可强求不可能的类别平衡。源THUMOS14命名可能以validation为training，必须依官方协议解析，不按文件夹名字推断。
- **Confirmatory**：使用冻结的默认新方案和强基线，在完整官方training pool训练，官方evaluation/test只评价已注册配置。所有默认配置与全部seed可同时启动，不等待Discovery结果“通过”。Discovery触发的修改进入新amendment、增加新ID；不能偷换已注册确认性任务。
- 研究效果只能在未用于该模型训练的video上称held-out。已见训练视频的诊断仍有用途，但标IN_SAMPLE，不当作可预测性泛化证据。
- FineAction作为独立数据迁移，源标注/协议可用则启动，不等待THUMOS成功。无资产标BLOCKED_ASSET，其他路线继续。
- 默认60 epochs、3 seeds(0/1/2)，最后完成的第60 epoch为主评价checkpoint；idx通常59，具体源日志索引需保存。20、40、60完成epoch用于诊断，不能把不同索引含义的旧checkpoint混用。
- baseline与新方法相同预训练、冻结范围、input、augmentation、optimizer、effective batch、train updates、head与postprocessing；不可用更大的Scout或更多训练次数而不作控制。
- equal epochs与equal train GPU-hours是两种报告。额外probe/多前向的增量必须记账；等训练资源基线通过额外监督训练或相同probe但不使用gain的版本比较，不能只比较epochs。

## 2. 配置与成本

科学工作清单是本包的run_requests.jsonl；其source_run_index包含旧JSONL真实ID，仅作候选引用，不是自动复用许可。

每个run还需绑定 resolved_source_config、commit snapshot、dataset split、weights hash和显式新variant。未解析source参数状态是NEEDS_RESOLUTION，不是可执行的完整模型配置。agent必须生成resolved_config.json及哈希，然后才可训练。不要把本包编译器当成已经实现训练入口。

目标Heavy MAC ratio =0.30/0.40/0.60；default 0.40是用于研究预算，不是承诺可以达到67mAP。A/B/C各自的token容量根据真实执行器解析，记录k_lp、Σk_lp²、padding。部分不可行预算写INFEASIBLE_BUDGET，不用错误零Heavy兜底。

同成本对照优先固定每层每parent容量向量；复杂路由允许全局计划变化时分别做等token、等HeavyMACs、等总latency对照。阈值/控制器校准仅用training/select pool，不读test标签。

## 3. 样本与反事实预算

默认发现集固定抽96个有效窗口，每video最多3个，至少32个video；若不足则全用并报告实际n。按background/short/long/overlap做分层，层标准从training标注计算。

每窗口抽8个legal candidate及8个paired equal-cost replacement，不仅取Router已经选的位置；均匀探索和学得分采样各一半，并保存采样概率。至少在一份固定、与策略无关的uniform bank报告价值相关性，以避免selection bias。

finite menu仅在24个窗口：在2个parent中分别给4个同成本候选各选2个，枚举6×6=36种局部组合，其余计划保持相同。这是该菜单和当前checkpoint下loss-best；不是理论上界、不是全局mAP Oracle，也不保证其他表示/训练达到同样上限。

记录oracle_recovery=(R_random-R_learned)/(R_random-R_menu_best)，分母≤数值容差或选择不属相同菜单时记NA。基于loss的regret与实际official mAP分开。

## 4. 主要终点与诊断终点

THUMOS14：按源官方评估的tIoU 0.3/0.4/0.5/0.6/0.7及平均值；primary localization endpoint=tIoU0.7 mAP；other datasets沿官方阈值定义，不强套0.7平均。Avg-mAP是共同主表，不将mean over per-video AP冒充。

时长分组用training GT duration的25/75分位，另附固定秒数敏感性；阈值与count写入split_contract。分组**召回**是primary slice指标，每video固定proposal预算，报告所有GT分母。若计算duration-slice AP，必须实现ignore其他时长GT及对应预测的明确协议；不能删除其余GT让原本正确检测全变FP。

边界误差：保存class-aware与class-agnostic匹配版本；匹配规则预注册（IoU>=0.1为宽松诊断门槛、同一GT和prediction至多匹配一次）。报告abs/signed start、end、duration-normalized误差，同时报告unmatchedGT与FP。只在matched子集MAE变好而recall降低，不能说定位总体更好。

相邻同类动作粘连：当预测覆盖两个GT各≥50%且间隙≥80%，标bridge；threshold sensitivity单列。单独处理gap内还有同类标注、GT overlap、标注完整性。over-extension量是预测区间在匹配GT外的额外持续时间；不能以Heavy support外预测直接定义错误。

Gap:计算的是Heavy支持并集的补集；cheap coverage一直存在，名称须为heavy gap，不能称“完全未观察”。gap/duration、boundary-to-nearest-heavy-distance/duration，分别报告。真实VFR从PTS获得；没有原始PTS就只做规则采样，不伪造VFR实测。

小物体、低运动等没有可靠GT时：motion可作明示代理特征；小物体需盲于方法结果的人工双标与抽样协议，否则NA。不要按结果挑选“困难小物体”实例。

## 5. 统计

- 主方法差值以同split/seed成对报告；均值±std及全部seed点。3 seeds只能粗略观察训练波动，不承诺强统计功效。
- Test uncertainty：以video为cluster进行2000次paired bootstrap。每次重建整份GT/预测并重新计算official mAP；重复抽到video须赋独立临时ID。所有方法用同一重采样索引。不要把AP按视频均值近似替代。
- seed不确定性与test bootstrap分别报，禁止把3个seed×1000个video当3000个独立训练重复。复杂hierarchical bootstrap可补充但注明估计对象。
- Utility候选相关性先每video汇总或cluster bootstrap，不能把同一视频数千token当独立n。
- 每路线锁定一个primary contrast、一个primary localization endpoint和cost口径；多对照用Holm校正，探索图可用BH但必须标exploratory。不只报告p值，报告效果、CI、绝对性能和资源。
- 小差异不显著不等于等效；需要等效/非劣时在训练前定义可接受容差并使用对应区间检验。本包不擅自给用户定义业务容忍度。
- 多预算曲线只连接实际点；不将非单调曲线强行单调化。Pareto点选择在固定规则中完成，不在test上挑最优budget再当预注册配置。

## 6. 因果解释边界

固定输入、checkpoint和预算的计算计划干预，是当前模型内的机制实验，不证明现实世界动作的因果关系。

B 的rank/hull错误是negative control，不是最强竞争方法；新方法须胜过metadata完整且训练匹配的time-attention。

C 高/低switch计划会同时改变选中位置；通过固定每parent/时间细化数量、多种随机耦合、平衡每空间组被选概率降低混杂。优先使用固定行列边际的binary mask sampler；记录种子和所有计划。若不能固定所需边际，明确报告不平衡并限制结论。

方法分数的attention heatmap只是机制可视化；真正增益图必须来自fresh counterfactual forward。plan改变后的特征不满足缓存依赖时重算。

人工时间retiming必须同步变换视频、PTS、GT；只改metadata是corruption测试。静态重复帧或合成事件只作为诊断，不能伪造benchmark实例。

## 7. 真实硬件测量

同一GPU型号、驱动、精度、内核配置、batch、输入window。训练和benchmark不共卡；不自动租赁算力、不杀无关进程。

至少三次独立测量session；每session预热50 iterations、正式200 iterations，若运行太短延长使正式段>=30s；记录所有逐iter值与运行顺序。Batch=1给p50/p95，Batch=8/32给吞吐，OOM是结果，不临时降低分辨率。

CUDA event测device部分；host wall clock加同步测端到端；区分encoded-video decode-to-detections与predecoded frames-to-detections。磁盘warm/cold cache分开；不能把模块时间简单相加替代存在异步重叠的实测总时间。

峰值allocated/reserved显存、k distribution、padded token数、kernel版本、有效batch、context长度都保存。理论MACs和profiler事件交叉校验；activation mask不是实际省QKV/MLP。

## 8. 并行执行

所有discovery和已锁定confirmatory默认variant一次性登记，code owners并行；无性能门槛。图模板/分析器先写，但只在对应真实数据到达后画结果，不等待其他路线或所有seed完成；最终推断等待预注册全部seed属于统计完整性，不是成功解锁。

允许依赖：资产、接口、编译/数值测试、自己的checkpoint、资源。禁止依赖：上一方法mAP超过某阈值、Oracle很好、seed0成功。

安全改动：OOM可调整microbatch并相应accumulation，但BN统计、优化步数、dropout、effective batch可能变化；先验证或登记amendment。不能静默改变训练协议。c10.dll问题在隔离环境处理，不覆盖用户全局Torch。

## 9. 数据产物

每run：resolved_config.json、source_audit.json、code_snapshot_hash、weights/split hashes、train log、checkpoint、raw predictions、raw metrics、cost_trace、status receipt。

每paired干预：video/window、seed/checkpoint、plan_A/B、cost_A/B、risk_A/B、cls/loc/bg分量、assignment_hash、normalizer_hash、effective support、dropout/eval状态、cache证明与fresh_forward标记。

每图：raw CSV/JSON、source receipts、过滤脚本、统计脚本、plot脚本、SVG/PDF/300dpi PNG、caption、n/seed/CI单位、所有排除理由。

数据状态分开：MEASURED / ANALYTIC / SYNTHETIC_DIAGNOSTIC / PRESET / PENDING。主结果仅MEASURED；分析公式标ANALYTIC；禁止将PRESET填入主图。测试fixture仅unit test不进入论文。

## 10. 自然分布与压力测试不可混用

人工极端gap、高频尺度切换、重复帧、错置metadata的结果必须单独标STRESS_TEST。它们说明机制在干预下的行为，不单独证明常规测试视频中该问题常见。自然学习计划的发生频率、现实子集结果和受控压力响应三者都报告。超出训练计划分布的性能变化可能是一般distribution shift，不自动归因于所假设的新机制。

严格成本allocator分成两类并标注：geometry-only固定parent容量profile，用于隔离位置策略；global-marginal-cost分配，用于部署比较。后者对每次新增合法原子重新计算ΔMAC或bucket真实成本，以相同allocator服务不同score基线；它是可行启发式，不宣称全局最优。原MoD交错dense层的计算下限可能超过0.4，因此paper-style A10单列0.6工作点，不能要求不可行的低预算后再宣称其失败；controlled MoD提供低预算公平比较。


---

<!-- source: protocol/IMPLEMENTATION_API.zh.md -->

# 共享科研实现接口与数据表

以下是给agents实现的接口契约，不宣称当前仓库已经提供这些类。

## NativePlan

必须记录plan_id/hash、video/window ID、input采样index与PTS、native tubelet pair、source resolution、parent partition、selected atom IDs、token IDs、layer scope、coarse/fine mode、成员mapping、validity。A/B/C共享元数据格式但不同执行语义，不强行以单个dense mask模拟全部路线。

## CounterfactualRecord（每条成对干预一行）

`experiment_id, run_id, checkpoint_sha256, split_role, video_id, window_id, seed, candidate_sampling_probability, operation_type, plan_A_sha256, plan_B_sha256, realized_parent_counts_A, realized_parent_counts_B, heavy_macs_A, heavy_macs_B, wall_ms_A, wall_ms_B, risk_name, risk_A, risk_B, delta_cls, delta_loc, delta_bg, gt_assignment_sha256, normalizer_sha256, eval_mode, fresh_forward, cache_scope_proof, input_tensor_sha256, data_status`。

缺少fresh/cache依赖证明的记录不能作为exact counterfactual。Counterfactual只有当前模型条件意义。负值合法。

## EvidenceRecord

`video/window/checkpoint/plan hashes, anchor_native_ids, anchor_support, raw_source_support_union, graph_context_support_union, spatial_slot_support, source_resolution, valid_mask, parent_ids, compression_method, slots_per_time, total_M, Q, feature_tensor_sha256`。

支撑集合可保存为紧凑native IDs映射真实区间，不能把大包络替代不连续集合。context是计算可达范围，不作为attention因果解释。slot数量可以大于query。

## DetectionRecord

保存全部预NMS和后NMS预测：`video_id, proposal_id, class_id, score, start_sec, end_sec, source_window, source_query_index, predicted_local_offsets, pts_mapping_hash`。GT单独保存原注释hash。窗口拼接、clip/padding、NMS参数统一。切片结果不能只保存matched预测。

## CErrorRecord

`video/window/layer/parent/group/time, plan_A/B, fine_mask_A/B, time_fine_counts, space_marginals, switch_count, reference_kind, e_t, delta_e_t, task_risk, prediction_matching_id, start_drift, end_drift, miss, false_positive`。

feature summary只用于诊断，完整必要tensor按固定样本集合保存。不要根据最后的错误筛掉异常tensor；所有nan/overflow作为失败记录。

## CostRecord

`run_id, session_id, GPU/driver/CUDA/PyTorch/precision/kernel versions, batch, video/window IDs, warmup_or_measure, timing_scope, start_stop_sync, iter_ms, module_times, native_tokens, selected_tokens_per_parent_layer, padded_tokens, total_MACs, heavy_MACs, receiver_MACs, peak_allocated, peak_reserved, decode_source, cache_state`。

总体耗时必须实测；module异步段不能相加冒充end-to-end。

## RunReceipt

`run_id,data_status,commit,config_sha256,split_sha256,checkpoint_sha256,seed,dataset,artifacts[{path,sha256}],claim_ids`是本包结构校验器的最小字段。正式agent需补充执行日志、exit code、代码snapshot及训练update数。结构校验通过不等于科研结论正确。

## 启动入口

建议真实仓库实现 `python -m geosparse_research.entry --request <json> --bindings <json>`，支持 resolve/train/evaluate/intervene/profile/plot。此入口由agents创建，本包未提供实际训练器。

resolve必须输出参数消费映射、真正模型类、模块树、trainable parameter list、native schedule、预算可行性、source config引用及code hash。任何NEEDS_RESOLUTION或未通过相应正确性测试的任务不得进入train。


---

<!-- source: protocol/FIGURES_AND_PAPER.zh.md -->

# 论文图表、呈现方式和结论生成规则

## 0. 图表是证据，不是插画

模型结构可用矢量绘图，必须来自forward/shape trace；定量图和选帧/粗细热图只能使用真实模型日志。禁止用生成式图像代替实测heatmap、样本序列、模型预测或实验曲线。允许单独画解析示意，但明确标ANALYTIC。需要真实视频画面时记录video_id/PTS/frame index及授权/数据来源。

图上不得默认写“ours wins”。先画全部预注册方法和seed，经过冻结统计协议再写结论。每图保留source hash与caption，缺数据给PENDING schema，不画装饰性趋势线。

## 1. 主文建议组织（不要求三个路线都成为独立贡献）

Figure 1：TAD失败现象与本文问题。真实例子说明类别正确但边界错误，再显示输入/原生/检测网格；静态例子不能单独证明普遍性。
Figure 2：统一执行接口及A/B/C。图区分已有conditional compute骨架与本次新增监督/校准；每层loop、parent边界、密集固定成本和真实形状都标明。
Figure 3：强近邻下的accuracy–cost与accuracy–latency。
Figure 4：A的价值分歧、成对gain和可迁移训练因子。
Figure 5：B的fixed-evidence receiver效果、gap错误和真实support。
Figure 6：C的相似性不足、尺度干预和时间误差。
Figure 7：短动作、高tIoU、失败案例与端到端成本。

可将每个Figure的panel分别输出独立SVG/PDF，再由排版脚本组合。正文优先保留真正支撑最终主张的结果，其余进入附录，但不删除负结果数据。

## 2. 22项明确图件任务

| figure_id | 对应实验族 | 原始数据/横纵轴 | 呈现/判读 |
|---|---|---|---|
| V01_problem | X01 | GT、raw predictions、实际帧；可辅以解析tIoU(d,epsilon) | 真实分类正确/边界错误案例+解析图，二者标签分开，不能以例子替代统计 |
| V02_pareto_macs | X02 | x=actual Heavy MAC ratio；y=Avg-mAP和AP@0.7 | 分别作图，点含seed；Dense、强基线、current与new完整显示，MAC/totalMAC分开 |
| V03_pareto_latency | X07 | x=end-to-end p50或batch throughput；y=相应精度 | 相同GPU/precision，p95另图，标记仅device计时的点，不能混轴 |
| V04_A_value_disagreement | A01 | x=true delta_cls；y=true delta_loc；位置类型 | 零线、四象限、样本数；边界/核心/外部由GT诊断分层，不用于test路由 |
| V05_A_gain_calibration | A02 | x=predicted signed gain；y=fresh paired gain | 校准bin固定于training，保留负值；按video cluster CI，附rank和regret，不只Pearson |
| V06_A_influence | A03 | 行=被改计算操作原时间；列=受益检测query；值=delta risk | 正负双向热图，带GT上下文；不是attention图；如果归一化需提供原始量 |
| V07_A_factorial | A04 | Router×R0/R1×CF on/off的配对效果 | 完整2×2×2交互图，指出监督迁移到MoD后是否仍有效 |
| V08_A_observability | A05 | scout resolution/info set vs heldout menu regret | cheap/deep-information/menu-best分开；oracle不是理论上界 |
| V09_B_gap | B02 | x=heavy gap/duration或boundary distance/duration；y=recall/extension | 固定evidence receiver对照，各bin的GT和video数，保留miss，不强设曲线单调 |
| V10_B_receiver_selector | B03 | 行=receiver；列=selector；值=AP和差值 | fixed evidence bank与end-to-end两种protocol分开；显示相同metadata强基线 |
| V11_B_bridge | B04 | real GT intervals、各receiver预测、source union、context union | 上方视频序列，下方同轴timeline；至少含成功、失败和无优势例子 |
| V12_B_provenance | B01 | anchor/hull/union/provenance对prediction的成对差 | metadata mutation仅negative-control；trained rich baseline另列，避免稻草人 |
| V13_B_slots_cost | B05 | slots、M/Q、receiver depth vs精度/延迟/显存 | 不预设M<Q；同时记录Heavy预算，不把receiver成本省略 |
| V14_C_similarity_gain | C01 | x=cosine/variance proxy；y=true refinement/swap gain | 比较真实gain与相似性排序的分歧，负值和区间CI |
| V15_C_error_timeline | C02 | 同视频z_dense,z_mixed的e_t,delta e_t；GT/prediction | 特征参考与任务GT分开，不把all-fine当真实最优；曲线不人为平滑 |
| V16_C_switch_intervention | C03 | x=随机化switch density；y=boundary shift/miss/FP | 固定parent成本、边际平衡，显示paired effects；未平衡的相关性另图 |
| V17_C_coarsefine | C04 | 每时间7×7 coarse/fine mask、frames、预测 | 标真实2×2成员与时间，所有窗口同阈值；不生成虚构显著性图 |
| V18_C_shared_update | C05 | merge方法/affine/utility/consistency vs AP和开销 | factor图，ToMeSD-attn和attn+MLP都列，实测train/infer cost |
| V19_risk_slices | X03 | short/medium/long、overlap、gap分组 | Primary recall及miss，matched boundary单列；避免subsetAP假FP |
| V20_system_cost | X07 | per-component时长、critical total、padding、peak memory | 堆叠仅用于无重叠测量；异步trace另展示，不将相加值当总延迟 |
| V21_budget_train_transfer | X04/X05/X06 | learned/shuffle预算、训练GPUh、第二数据集 | 三个独立panel，shuffle同直方图并保留每视频自己的score |
| V22_failures | X08 | 预注册错误类型的真实GT/预测/plan/frames | 至少漏短动作、同类粘连、尺度切换、lowdetail不辨；附总体频率不只精选 |

## 3. 表格

T1 Main Pareto：方法/骨架、预训练、可训练参数、预算kind、实际Heavy与total MAC、AP thresholds、Avg、AP@0.7、short recall、p50/p95、throughput/VRAM。paper-native不同预训练协议分成不同panel，不制造不公平排名。
T2 Architecture tax：Dense、A-all、B-all、C-all-fine，同训练条件精度/成本；A/C同权重dense limit另报数值/gradient误差，不混为训练性能。
T3 Closest prior controls：来源、官方/论文重实现/controlled adaptation、是否同metadata/TIA/loss/trainingcost、无法匹配项。
T4 Value estimators：PG、gradient retention、zero-probe、actual add/swap，heldout regret及额外训练成本。
T5 Robustness：short、高tIoU、start/end+miss、overlap、background、VFR/tail可用性。
T6 Claim ledger：主张、证据任务、效应/CI、强替代、status=SUPPORTED/INCONCLUSIVE/REFUTED/UNTESTED。

## 4. 图件格式

每图输出 figures/{figure_id}/data.csv|json + plot.py + figure.svg + figure.pdf + figure.png + caption.md + receipt.json。至少300dpi，矢量文字，统一术语与单位；缩为论文单栏/双栏后可读。图例同时用marker/line pattern，避免仅颜色区分；同方法跨图一致。所有CI注明单位(video/seed/session)。精度轴不得裁切掩盖负结果；无法比较的成本不要强插共同曲线。

同一个figure目录必须包含生成命令和输入hash，所有过滤条件可追溯；caption明确discovery/confirmatory、是否同权重干预、采样窗口规则、成本容差、是否包含decode。

## 5. 论文写作规则

标题暂用 GeoSparse-TAD: Task-Value, Evidence Support, and Update Sharing for Efficient Temporal Localization。最终根据真实证据收敛，不为了三个路线都像贡献而堆模块。

Abstract先用研究目标与方法描述，空结果使用[MEASURED_RESULT_PENDING]；正式投稿不得包含预设40/67。
Introduction：TAD的区间约束→现有稀疏骨架已经可用→仍未知的任务价值/不完整证据/共享更新误差→可测试方案→真实结果。
Related Work：清楚承认MoD/CoDA/LITE、Coarse-Fine/mTAN/LookupViT、MSViT/ToMeSD/ALGM，不攻击缺乏合理适配的弱基线。
Method：先给共同execution contract和cost，再给各路线最小新增项；理论只陈述真正推导出的关系，不把一阶代理写成有限差分真值。
Experiments：主结果+最接近对照+paired干预+资源+失败范围；区分相关性和干预。
Discussion：允许最终结果是同一训练原则增强多个骨架；允许某route新模块不成立而被精简。预注册负结果不删除，附录保留所有run终态。


---

<!-- source: agents/00_ORCHESTRATOR.zh.md -->

# Agent 00_ORCHESTRATOR｜总控PI与并行编排

你负责的科研工单：**全部30工单**。

## 文件与模块所有权
task graph、统一契约、amendment、资源授权和总状态；不直接重写各路线实现。

## 必须执行的任务
读取MASTER_AGENT_COMMAND并立即分配01–11。冻结共享API、目录ownership与分支集成规则。所有基线和最小新方案同时登记。分别维护discovery与confirmatory设计锁，禁止根据测试集重选主对照。将805项核心请求转为真实配置前检查NEEDS_RESOLUTION。新子控制必须有明确ID/参数/来源，不从Markdown臆造旧ID。审批只针对代码、资产和预算，不针对结果是否有利。

## 验收产物
全量状态表、已授权GPU-hour估算、任务DAG、所有例外及重新注册记录。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/01_SOURCE_DATA_AUDIT.zh.md -->

# Agent 01_SOURCE_DATA_AUDIT｜源码、数据和数值审计

你负责的科研工单：**X00/X01**。

## 文件与模块所有权
audit/、data_contracts/、tests/contracts/；只读原始基座。

## 必须执行的任务
核验GeoSparse_TAD_20260907@340a3541，分清C3，输出commit/dirty/source路径。追踪routing.py/model.py/sparse.py/receivers.py/geometry.py的实际连接。核验A/C每层源TIA，B noFFN/noROI/M可以>Q，shape不能信生成图。确定源THUMOS14训练/评估split和PTS。建立fit/probe/select video-level划分。真实执行K0/Kall、梯度、mask、tail、empty-GT和DDP不同K测试；测试文件存在不算通过。c10.dll使用独立环境诊断，不能改用户全局系统。

## 验收产物
source_audit.json、split_contract.json、numerical_test_report.json、architecture_tax数据及不可用资产清单。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/02_A_BASELINES.zh.md -->

# Agent 02_A_BASELINES｜A执行器与强条件计算基线

你负责的科研工单：**A04/A06/A07**。

## 文件与模块所有权
backends/route_a/、baselines/mod/、baselines/coda/、对应测试。

## 必须执行的任务
从原论文实现MoD gated residual和合理dense/routed layer配置；另实现与A共享executor、同parent分组、同TIA的contextual MoD。保留原始协议与controlled adaptation两种命名。实现CoDA-TAD，审计LITE论文及官方仓库真实可用性，不能把README叫完整代码。与03对接R0/R1/CF，使相同新监督也能作用于MoD。严格实际压缩QKV/MLP，不跨parent，unselected仅bypass Heavy。执行完整Router×Risk×CF因子和等残差缩放/路由容量控制。

## 验收产物
A全部recipe真实代码、source adaptation ledger、执行张量trace、dense-limit梯度测试、强基线训练及raw predictions。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/03_VALUE_LEARNING.zh.md -->

# Agent 03_VALUE_LEARNING｜任务风险与价值学习

你负责的科研工单：**A01/A02/A05；与C01衔接**。

## 文件与模块所有权
risk/、routing/value_heads/、routing/policy_estimators/、价值校准脚本。

## 必须执行的任务
实现R0与实例平衡R1，保持源GT assignment/normalizer在成对前向中固定。新增ECV signed equal-cost swap supervision，不以绝对梯度替代真实价值。PG/STE/gradient代理作为明确不同估计器；训练额外probe计费。raw Scout与逐层router的信息可比性单列。用uniform heldout candidate bank评价regret/sign/rank，包含未选候选，记录采样概率。不得用test标签生成部署路由。实现代价约束下addition与swap不同target、全局utility尺度和动态K。

## 验收产物
风险逐query分解、真实target与预测表、有限菜单regret、全局gain标定、训练开销明细和A因子结果。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/04_B_RECEIVERS.zh.md -->

# Agent 04_B_RECEIVERS｜B来源建模与强receiver研究

你负责的科研工单：**B01–B07**。

## 文件与模块所有权
backends/route_b/、receivers/、evidence/、B基线和测试。

## 必须执行的任务
保留current B不变另命名新版本。实现physical interpolation、metadata-rich standard cross-attention、mTAN-content、provenance residual gate；基线可有FFN/门控，不能只给我们metadata。区分anchor/raw-source/context support，计算图中attention使support饱和要真实记录。先做共享evidence bank接收器对照，同时运行各方法端到端适配，不等待结果。实现Coarse-Fine/Lookup/TokenFuser合理adaptation并标初始化差异。roi_mode保持none，M/Q和receiver成本实测。gap、bridge的错误以GT定义，不以Heavy support外推自动判错。

## 验收产物
全部B模块、evidence bank及缓存契约、receiver×selector数据、gap/bridge风险、M/Q开销和B失败机制。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/05_C_REFINEMENT.zh.md -->

# Agent 05_C_REFINEMENT｜C混合尺度与共享更新研究

你负责的科研工单：**C01–C07**。

## 文件与模块所有权
backends/route_c/、merging/、calibration/、C基线和测试。

## 必须执行的任务
固定2×2/current逐层回写作为锚点。实现MSViT-style、ToMeSD attention-only与attention+MLP、ALGM/CubistMerge和StructSAM-inspired近邻；保留合法时间与parent，不因原论文任务不同而制造弱适配。最小新方案=真实refinement risk gate+coarse-only affine；fine路径不能破坏dense limit。做utility-only、affine-only、两者、paired-plan consistency及ToMeSD匹配calibrator控制。记录同模型all-fine诊断e_t和delta e_t，不拿allfine当GT/理论上界，不引入独立dense teacher。

## 验收产物
C全部recipe、成员map/回写gradient报告、真实refinement gains、尺度计划、时序近似误差和强merge基线结果。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/06_INTERVENTIONS.zh.md -->

# Agent 06_INTERVENTIONS｜共用反事实引擎与随机干预

你负责的科研工单：**A02/A03/B01/B02/B04/C02/C03**。

## 文件与模块所有权
interventions/、counterfactual/、sample_banks/；通过backend API调用，不改各路线算子。

## 必须执行的任务
实现CounterfactualRunner，以相同checkpoint、输入tensor hash、assignment、normalizer、随机状态比较实际计划。A/C改选中集合后重跑依赖后缀；B receiver-only可在同获取集合cache，改变acquisition时失效。输出每操作到每query的风险差而非attention。C固定parent/time成本、尽量固定行列边际、用多随机coupling形成高低switch；无法平衡要写限制。B gap干预重新编码；metadata重排不变性与corruption明确分开。24窗口36choice有限菜单按协议穷举，不把启发式当oracle。

## 验收产物
可重放plan bank、paired records、noise floor、cache等价证据、机制图原始数据和所有排除记录。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/07_HARDWARE.zh.md -->

# Agent 07_HARDWARE｜真实算子成本与部署测量

你负责的科研工单：**X05/X07/B05/C06**。

## 文件与模块所有权
profiling/、kernels/可选优化、hardware_receipts/；优化必须等价。

## 必须执行的任务
审核所有方法是否真的缩小QKV/MLP。计入dense PatchEmbed/TIA、Scout、slot压缩、receiver、pack/unpack/sort/crop。执行B1/8/32、三session、50warmup/200measure且>=30s，记录CUDA同步和host wall。分别报告device/frame-to-detection/encoded-video路径。每层parent实际k和padding全记录。不同框架的MAC定义统一，不把flash mask当稀疏计算。优化kernel先验证数值/梯度等价；不与训练同卡计时，不租外部算力。

## 验收产物
raw timing、profiler traces、peak memory、cost breakdown、真实Pareto数据、训练CF开销和优化前后等价结果。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/08_FIGURES.zh.md -->

# Agent 08_FIGURES｜科研绘图与可编辑图件

你负责的科研工单：**V01–V22/T1–T6**。

## 文件与模块所有权
plots/、figures/、captions/；不修改任何模型结果文件。

## 必须执行的任务
现在就实现图schema与脚本，但无MEASURED数据时不画假曲线。每图消费receipt+raw table，按同方法统一图例、带CI/n/seed/成本scope。V06必须来自真实counterfactual不是attention，C图不伪造coarse masks。真实例子按数据/错误类型冻结抽样，并有成功/失败/无收益；展示分布防止cherry-pick。交付独立vector panels再compose论文多panel；文字缩为论文尺寸仍可读。公式解析图独立标ANALYTIC，重复帧标SYNTHETIC_DIAGNOSTIC。

## 验收产物
22套图件各含CSV/JSON、SVG/PDF/PNG、脚本、caption和hash；没有数据的图明确PENDING。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/09_STATISTICS.zh.md -->

# Agent 09_STATISTICS｜盲化评价和统计检验

你负责的科研工单：**X02/X03/X04/X06；跨路线主对比**。

## 文件与模块所有权
evaluation/、statistics/、claim_tables/；不得调整模型超参。

## 必须执行的任务
实现官方mAP和duration recall，边界matched/unmatched分开。video-cluster paired bootstrap每次重建完整数据、重复video换独立ID，不能平均per-videoAP。主对比A23-A11/B21-B13/C22-C18按0.4 AP0.7冻结，多重比较处理；探索与确认性分开。根据训练GT定义durationbins，不能测试后移动短动作界限。所有3seed都列，不因某seed不好排除。预算shuffle保直方图和每视频自身scores，长度分层控制。若统计功效不足写inconclusive，不强行宣称等效。

## 验收产物
official predictions与metrics、paired CI、效应量、预注册检验表、slice分母与NA理由、shuffle对照。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/10_EXECUTION.zh.md -->

# Agent 10_EXECUTION｜配置解析、调度与不可变复现

你负责的科研工单：**核心请求+全部amendment**。

## 文件与模块所有权
integration/、launch/、run_registry/、receipts/；不修改旧manifest。

## 必须执行的任务
读取旧JSONL的真实ID和hash，不猜原任务。把SCI3 scientific requests绑定实际source configs、weights、splits与代码能力；未解析禁止启动。实现param-consumption映射、注册表import、strict unknown-key失败。每run独立不可变snapshot。仅相同全部关键条件可复用旧结果，否则新amendment；current快照与modified methods分名。所有slots/resolution等子控制从工单明确参数展开，先登记后执行。新增训练入口由你真实实现，不把本包编译器当训练器。OOM只在不改变有效协议时调整，否则登记变更；进程/断点/重试收据完整。

## 验收产物
runnable resolved manifests、运行入口、source reuse ledger、状态全集、重试/阻塞原因、复现命令。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: agents/11_REDTEAM_PAPER.zh.md -->

# Agent 11_REDTEAM_PAPER｜查新、替代解释审查与论文写作

你负责的科研工单：**全部claim ledger**。

## 文件与模块所有权
literature/、paper/、redteam_reviews/；不按结果修改主终点。

## 必须执行的任务
核对sources原论文和公开代码版本，尤其MoD/CoDA/LITE、CF/mTAN/Lookup、MSViT/ToMeSD/ALGM；最新preprint与已验证机制分开。对每条主张提出最强替代解释：额外参数、更多训练、元数据不平等、预算不相等、未适配基线、幸存者偏差、cache错用、测试集泄漏。结果回来即写Claim→Experiment→Figure→Scope，缺失写UNTESTED。先写无虚构结果的论文骨架，不以40/67达标判科研完成。根据真实证据决定统一训练原则、B证据接收或C时序误差哪条成为主贡献，不将所有模块都宣布创新。

## 验收产物
literature audit、baseline adaptation diff、reviewer-style反驳清单、论文草稿、8分钟oral故事和逐图科学caption。

## 全角色共同约束

完整读MASTER_AGENT_COMMAND和三份protocol，尤其实现边界、数据划分、预算与反事实定义。用户静态审查不是运行验证；本包13项测试仅针对编译/参考工具，不是TAD测试通过。

不等待其他路线性能；只等待自身代码/资产/正确性/产物。既有default训练60epochs、3seeds，不用seed0成功解锁。不得自动租算力、删用户数据、杀无关进程。所有新模块先并行注册，最终可因无收益从论文简化但实验保留。

不使用预设40/67作为实测；不静默吞参数、不伪造baseline源实现、不把freeze当no_grad、不把选中mask当减少计算、不把loss-best当mAP理论上界。主张需以真实数据支持，可得到否定结果。

每次回传必须含任务ID、commit/config/split/weights/checkpoint hashes、状态、产物路径、复现命令、失败原因和对应claim。仅文件存在不算完成。


---

<!-- source: sources/PRIMARY_SOURCES.md -->

# 已核验的主要研究来源（检索日期：2026-09-07）

这些文献说明已有范式和必须面对的竞争者，不证明本包提出的新机制有效。公开论文的任务/预训练不同，需区分paper-native复现、paper-based重实现与controlled TAD adaptation。Agent 11须在实施时重新核对公开代码/版本/许可证，尤其新近预印本。

| ID | 文献与链接 | 本项目使用边界 |
|---|---|---|
| S01 | [Mixture-of-Depths](https://arxiv.org/html/2404.02258v1) | token可跳过重层但完整残差存在；与TIA兼容，是A强基线，不声称MoD天然丢失几何 |
| S02 | [Conditional Adapters](https://arxiv.org/html/2304.04947v2) | dense-light/sparse-heavy与冻结预训练的直接近邻 |
| S03 | [Principles of Visual Tokens / LITE](https://arxiv.org/html/2411.13626v2) | token价值代理、小预测器和预算分配已有先例；不能把这些术语当独创 |
| S04 | [LITE作者仓库](https://github.com/maggieHao/Efficient-LITE) | 本次检索页面显示README/Coming Soon；实施时查实际commit，不把论文重实现写成官方代码复现 |
| S05 | [AdaTAD](https://arxiv.org/html/2311.17241v2) | TIA与参数高效适配；冻结不等于省去前向，也不保证无激活梯度开销 |
| S06 | [Coarse-Fine Networks](https://openaccess.thecvf.com/content/CVPR2021/html/Kahatapitiya_Coarse-Fine_Networks_for_Temporal_Activity_Detection_in_Videos_CVPR_2021_paper.html) | 动态时间采样和多阶段粗细融合；原设置不等同所有interval TAD协议 |
| S07 | [mTAN](https://arxiv.org/abs/2101.10318) | 连续时间不规则观测到固定表示，B需给它合理内容/时间适配 |
| S08 | [LookupViT](https://arxiv.org/html/2407.12753v1) | 少量重型压缩tokens与大量轻量状态的交互 |
| S09 | [TokenLearner / TokenFuser](https://arxiv.org/abs/2106.11297) | 学习压缩和返回dense表示，含视频，B/C共同近邻 |
| S10 | [Token Merging for Fast Stable Diffusion](https://ar5iv.labs.arxiv.org/html/2303.17604) | merge/process/unmerge/residual已有，需比较attention-only与其他压缩配置 |
| S11 | [MSViT](https://arxiv.org/html/2307.02321v1) | mixed coarse/fine token gating已有；覆盖完整不等于信息无损 |
| S12 | [ALGM](https://arxiv.org/html/2406.09936v1) | dense segmentation上的token merging，不可假装所有近邻仅做分类 |
| S13 | [CubistMerge](https://arxiv.org/abs/2509.21764) | structured spatial-preserving merging近邻，实施核对版本与可用代码 |
| S14 | [StructSAM](https://arxiv.org/abs/2603.07307) | 2026预印本，边界保护与merge/unmerge近邻；不直接转移其性能结论到TAD |
| S15 | [Action Sensitivity Learning](https://arxiv.org/abs/2305.15701) | 分类/定位重要性差异已有研究；本项目需证明实际计算分配的增量 |
| S16 | [ActionFormer](https://arxiv.org/abs/2202.07925) | 局部时间建模、多尺度与区间回归；沿用官方评价而非自造AP平均 |
| S17 | [AdaSpot](https://arxiv.org/abs/2602.22073) | 低分辨率全局+ROI用于事件点定位，空间扩展的强对照；当前B未实现ROI |

## 查新交付要求

每个近邻保存：论文版本、官方repo和commit、任务与数据、预训练、是否需要教师、执行计算发生位置、能否共享backbone权重、可比性差异、公开结果与自己重现实测的边界。不要引用综述/社交媒体替代算法定义。没有源码仍可按论文实现，但明确复现程度。
