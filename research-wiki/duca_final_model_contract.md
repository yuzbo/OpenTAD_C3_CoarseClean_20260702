---
type: research_contract
title: "DUCA 最终模型合同与论文实验闭环"
status: historical_contract_superseded_for_mainline
canonical: false
updated: 2026-07-27
---

# DUCA 最终模型合同与论文实验闭环

> **2026-07-27 继承说明**
>
> 本页保留 `9f97f2c` 边界微簇/R0--R5 路线的历史合同与负证据，但已不再是当前论文
> 主线的唯一权威说明。当前主合同是
> `research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`：
> DUCA 主方法恢复为不修改下游检测器结构的纯 pre-backbone 选帧插件；正式 detector
> 训练总预算恢复为 60 轮/6,000 次成功更新；官方 dense `69.03%`、干净原生
> K=384/K=192 均匀下采样必须先闭环。物理时间检测头和 assignment/regression 改造
> 只作为诊断或增强集成，不再定义纯插件主结果。若下文与该恢复合同冲突，以恢复合同、
> 最新 `query_pack.md`、`anti_repetition.md` 和实验节点为准。

下文是 2026-07-23 冻结的历史目标、模型结构、训练方式和实验路线，用于保留讨论过程
与负证据，不再自动约束当前论文主线。修改本页时仍必须同轮更新 `query_pack.md`、
`anti_repetition.md`、`duca_model_version_registry.md` 和 `log.md`。

## 当前唯一执行实例（2026-07-23）

当前唯一 R0--R5 模型身份是
`codex/duca-boundary-burst-20260722@9f97f2c7f081b10fbf1f63d0602a621c6b43a780`，GitHub 为
`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9f97f2c7f081b10fbf1f63d0602a621c6b43a780`。
远端干净快照为
`/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_9f97f2c_20260722`，正式根为
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`。
Jobs `1180490--1180496` 覆盖 R1、四个并行 P0/R2/R3 候选、唯一共享 R0/P0/U/G0/alignment、
R4、五预算两后端三种子 R5 与聚合。旧 `cd68d89/a00498e` Jobs 已因重复 bootstrap 与
GPU step 资源继承错误取消，只保留执行故障历史。

`4f81299f826a4d33b18f21af8436ec1bd8cc4f51` 只新增四种官方粗分类后端 P0 诊断入口，
Jobs 为 `1180502--1180505`；它不改变 `9f97f2c` DUCA 模型，也不产生 TAD mAP。

当前状态仅为 `experiment_running`。selected-axis true-time 风险、mandatory group
completeability 和 detector surrogate alignment 均已登记为待 terminal mAP 裁决的问题；
TTDI 尚未实现，也不是已经冻结的最终模型。

## 1. 最终到底交付什么

最终论文交付一个面向 **离线时序动作检测（offline TAD）** 的固定预算
**pre-backbone temporal acquisition plugin**：它先用低成本、低分辨率的完整时序观察
判断“哪里可能发生动作状态变化”，再从原视频时间轴选择 exact-K 个高价值观测交给
昂贵 VideoMAE/TAD detector。

首个正式 backend 是现有 official-derived AdaTAD/ActionFormer。插件不重新发明
VideoMAE、projection、neck、ActionFormerHead、assigner、detector loss 或 NMS。论文的
主要贡献是检测器之前的任务感知时序计算分配，而不是一个新 TAD detector。

最终模型是一个统一部署图和一个两阶段训练课程，不是三套独立部署模型：

```text
低分辨率完整窗口
  -> 粗动作状态模型
  -> 间接状态转变与边界微簇 selector
  -> exact-K / 最大空洞结构化求解
  -> 原时间轴 hard selected observations
  -> official-derived AdaTAD/ActionFormer
  -> TAD proposals 与 mAP
```

### 主论文固定范围

- primary budget：`K=384`；
- first efficiency extension：`K=256`；
- primary dataset/backend：THUMOS14 + official-derived AdaTAD/ActionFormer；
- primary method：fixed-budget DUCA；
- dynamic MUST、X3D/SlowFast frozen prior、post-VideoMAE EU-CRR fusion 均不属于主方法。

## 2. 明确不是什么

- 不是 Online TAD 或 streaming TAD；模型可观察完整离线窗口。
- 不是 actionness top-k；动作内部高置信度不是最终选帧目标。
- 不是 class-specific start/end detector；selector 只从粗动作状态变化间接推断边界中心。
- 不是 one-frame-per-cell；允许预算跨区域转移并在边界附近形成局部聚集。
- 不是三阶段独立训练的 ASFormer -> selector -> detector pipeline。
- 不是通过 LongTensor hard index 直接反向传播；离散决策只能使用经过实证对齐的代理。
- 不是默认的 coarse/VideoMAE feature fusion。启用该路径后方法身份必须另行改为
  acquisition-and-fusion adapter。
- 不是只报告 backbone FLOPs 的“高效”方法；完整成本必须包含数据通路和 cheap probe。

## 3. 最终模型结构

### 3.1 输入与低成本粗动作状态模型

对每个候选窗口构造按原时间顺序排列的低分辨率 dense proxy `X_c`，长度记为 `T`
（当前 THUMOS 合同为 `T=768`）。粗模型由项目的轻量 spatial stem 与固定来源的
official ASFormer temporal core 组成，输出：

- `p_action[t]`：动作/背景二分类概率；
- `H[t]`：ASFormer temporal hidden；
- valid mask 与完整来源/成本记录。

粗 action head 的训练目标只有二分类动作语义。GT start/end 不直接训练一个新的粗粒度
start/end head，避免违背“把难边界定位转成粗分类后的间接状态转变定位”这一初心。

### 3.2 deploy-visible 状态转变证据

selector 观察以下推理期可获得的证据：

```text
p_action
abs(delta p_action)
signed delta p_action
uncertainty / entropy change
local hidden change ||H[t]-H[t-1]||
ASFormer hidden H[t]
valid/time-position encoding
```

这些证据送入现有 V8 transition scorer 的有界扩展，产生 class-agnostic transition
center score `s[t]`。训练期 GT action starts/ends 只用于把 `s[t]` 校准到真实状态转变
中心；validation/test/inference 不得读取 GT、teacher、ledger、prediction cache 或预提取
actionness JSONL。

### 3.3 Oracle-calibrated bilateral boundary burst

每个候选 center 同时产生一个有限偏移轮廓 `b[t,delta]`，其中
`delta in [-R,R]`。该轮廓必须表达：

- center 附近有高价值观测；
- 边界左侧和右侧都获得观测；
- 每个边界允许形成 Oracle 式 3--5 帧微簇，而不是只选一帧；
- 达到 train-split Oracle 标定的局部配额后奖励饱和；
- 相邻 start/end 或短动作的重叠微簇进行确定性去重；
- 一个强峰不能吞掉所有边界预算。

多个 center profile 通过局部 saturating union 合并，避免相邻高分位置重复累加无限收益。
剩余预算由一个低权重全局 context utility 分配。`R`、局部配额和最大空洞 `G` 必须先
由 train-split Oracle reachability 冻结；当前 V8 的 `G=2` 只是诊断设置，不是科学真理。

### 3.4 结构化 exact-K hard selection

最终 utility 仍交给现有 global structured DP，在同一可行集合中求解：

```text
maximize    sum_t z[t] * utility[t]
subject to  z[t] in {0,1}
            sum_t z[t] = K
            every valid unselected hole <= G
```

输出必须是按时间升序、无重复、严格 exact-K 的原时间轴 indices。不得在 decode 后进行
不可审计的补点，也不得回退到 local-cell。hard forward 与训练 soft distribution 必须来自
同一 exact-K/G 动态规划族。

### 3.5 official TAD backend

plugin hard gather K 个 RGB observations，按时间顺序送入现有 VideoMAE adapter 和
AdaTAD/ActionFormer。原始 ActionFormerHead、检测损失、NMS 保持不变；adapter 只负责：

- selected-axis training target mapping；
- original-time timestamp metadata；
- detector 输出到原时间轴的确定性 inverse mapping。

必须诚实承认：VideoMAE 当前仍把 selected-rank 当作规则序列。G3 的 physical-time 风险
由 same-selected-frames 几何诊断裁决；physical-grid head 不是默认主方法，也不能静默
混入 matched U/G0 对比。

### 3.6 True-time 候选修正边界

当前 `9f97f2c` 的 detector 内部仍按 selected rank 处理不规则采样，最终 inverse map 只能修复
输出坐标，不能恢复 VideoMAE tubelet、projection 和 head 已使用的时间度量。该风险已由
`a00498e` Pro 审查提出并在当前远端代码中复核确认。

但 TTDI 目前只是**结果驱动的候选修正**，不是最终合同的一部分。只有当 learned hard selection
的边界/上下文质量明显优于 uniform、而 official mAP 特别是高 tIoU 仍不优时，才先加入零初始化
true-time feature residual。physical-coordinate head/GT assignment 是第二个独立变量，不能与
feature residual 同时引入而失去归因。

## 4. 数据与真实成本合同

最终实现区分两种执行模式：

1. **训练/严格复现模式**：dense tensor 已进入模型，plugin 在 VideoMAE 前 hard gather。
   该模式可证明 heavy-backbone frame reduction，但不能单独证明完整视频处理降本。
2. **论文部署模式**：先生成低分辨率 dense proxy，selector 输出 indices 后，只对 K 个
   source observations 执行高分辨率 materialization/H2D/heavy backbone。若视频解码器
   无法实现可靠随机访问或二次读取，必须把实际额外 decode 成本计入账本。

完整推理成本至少包括：decode、CPU transform、host materialization、H2D、coarse
spatial stem、ASFormer、transition/burst scorer、DP、gather、VideoMAE、projection、
neck、head、inverse mapping 和 NMS。论文只有在部署模式实测总成本低于 dense route
时，才能声称端到端高效；否则只能声称减少 heavy-backbone processed frames。

## 5. 训练与梯度所有权

### 5.1 损失定义

```text
L_P0 = L_action + L_anchor + L_bilateral + L_quota + L_fair/context

L_official = L_TAD + lambda_aux(step) * L_selector
```

- `L_action`：动作/背景二分类，只更新 spatial stem、ASFormer trunk、action head。
- `L_anchor`：把 transition center 拉到 train-only GT endpoint 附近，只更新 scorer/burst。
- `L_bilateral`：要求每个 endpoint 左右两侧都有预期 selected mass，只更新 scorer/burst。
- `L_quota`：局部有用 mass 达到 Oracle 标定配额后停止奖励，并惩罚无界堆积。
- `L_fair/context`：保护弱 endpoint 与剩余全局上下文，只更新 scorer/burst。
- `L_TAD`：官方 detector cls/reg loss；始终更新 VideoMAE adapters 与 detector，只有在
  real legal hard-swap alignment 通过后才允许通过 protected surrogate 更新 scorer/burst。

selector 读取的 coarse hidden 默认 detach。这样 coarse branch 学的是稳定二分类状态，
selector 学的是如何把状态变化变成边界采样，检测器不会把 coarse action semantics 改写
成一个难以解释的隐式 detector。

各 selector loss 必须按 endpoint/有效长度归一化到可比尺度，首版采用等语义权重；动作
loss 与 selector losses 因参数所有权分离，不用靠夸张 loss weight 争夺主导权。不得在
official test mAP 上搜索 loss 权重。

### 5.2 两阶段课程

#### 阶段 A：frontend P0

- 使用 sealed train-only split，最多 20 epochs；
- heavy VideoMAE/AdaTAD 完全跳过；
- 训练 coarse action semantics 与 transition/burst selector；
- checkpoint 只按预注册 coarse calibration、center error、bilateral count、quota/fairness、
  short-action endpoint 和 exact-K/G 机制门选择；
- 选最早通过门禁的 checkpoint，不看 validation/test TAD mAP。

前期“粗分类、状态转变和边界覆盖占主导”通过**跳过 detector**实现，而不是人为把几个
loss 放大几个数量级。

#### 阶段 B：official-60 task adaptation

保持 6000 successful updates，不增加 detector 训练预算：

| updates | positions | coarse | selector | detector feedback |
|---:|---|---|---|---|
| 0--999 | exact-uniform K | frozen/eval | frozen | off；只预热 official detector |
| 1000--2499 | uniform -> learned policy ramp | frozen/eval | auxiliary training | off |
| 2500--3999 | learned hard policy | frozen/eval | auxiliary training | G0 off；G1 仅在 alignment GO 后 0 -> inherited rho=0.25 |
| 4000--5999 | learned hard policy | frozen/eval | auxiliary floor | G0 off；G1 fixed rho=0.25 |

`lambda_aux(step)` 随 policy ramp 下降但保持一个预注册非零 floor，防止 selector 只追逐
噪声 detector loss。主结果固定 terminal EMA，不用中间 mAP 挑 checkpoint。

### 5.4 当前实现锚点（2026-07-22 04:00）

当前精确候选为
`codex/duca-boundary-burst-20260722@aa3352ecf803c81d007a62ed5398667d9551684b`。
它已经实现 R0--R3 所需的 Gaussian/R2Q3/R4Q5、U/G0 四臂和同一全局 DP，远端 Linux
回归通过。上一提交 `899630a` 的第二轮独立 MAX 给出的 runtime binder、pooled
crop-validity 和证据来源/hash blocker 已在 `aa3352e` 就地修复；干净远端快照通过 DUCA
`139 passed, 3 skipped`、强制 C3 `23 passed`、compile/bash/HEAD/clean。全新独立 MAX
`019f8647-ad93-70f3-a763-218f7552ac95` 正审查该精确提交。真实 CUDA 门禁、R0 headroom
和 terminal mAP 均未完成；不得称 V9、empirically supported 或 paper ready。

### 5.3 三个归因臂，不是三个模型

- `U`：exact-uniform K；证明 matched detector baseline。
- `G0`：corrected boundary-burst policy，无 detector feedback；证明选择目标本身。
- `G1`：G0 + 通过 alignment 的 protected detector feedback；证明 task adaptation。
- `G2`：G1 + training-only uniform companion；仅作为稳定性消融，推理时不存在。

## 6. 完整实验路线

当前 V8 `63e25eb` 只提供旧 Gaussian-mass objective 的冻结诊断证据。它的终局必须先
封存，但无论结果如何都不能冒充下述 boundary-burst successor 已实现。

### R0：无训练 Oracle/K/G 可达性

- 在 train split 比较 exact-uniform、unrestricted GT Oracle、projected Oracle；
- 扫描有限候选 `K`、`G`、burst radius/quota；
- 用 compatible frozen detector 计算 matched mAP，并统计 center distance、双侧数量、
  missed endpoint、重叠率、短动作和 max hole；
- 选择最弱但保留显著 Oracle headroom 的 coverage cap；若 bootstrap 下 projected Oracle
  不优于 uniform，先改变可行集合，禁止调 loss。

### R1：数学、代码和真实模型门禁

- 小 T/K/G brute force 与 DP oracle 完全一致；
- 每个非强制 event 有非零有效梯度；
- hard/soft 同一可行族、strict K、unique chronological positions；
- bilateral/quota saturation、endpoint overlap deduplication 可构造验证；
- full-model optimizer exact coverage、EMA、AMP/DDP、no-leak、original-time mapping；
- official detector head/loss/NMS 与 baseline 配置 diff 白名单通过。

### R2：P0 单变量机制裁决

同一 coarse architecture/split/预算比较：

1. simple `abs(delta p_action)`；
2. 旧 V8 Gaussian-mass objective；
3. corrected anchor + bilateral + quota + fairness burst objective。

必须分别报告 coarse AUROC/AUPRC/ECE 与 selector center/bilateral/burst 指标，定位“粗模型
没学好”还是“selector 没用好”。corrected selector 若仍不优于 simple delta，停止
scorer 训练并审计证据可辨识性。

### R3：主结果锚点 U vs G0

- 同一 exact commit、P0 hash、seed、数据、pretrain、6000 updates、terminal EMA；
- 先只运行 U 与 G0；
- G0 不超过 U 时，该 K/G learned allocation KILL，不允许用 G1、更多 epoch 或 fusion
  掩盖失败。

### R4：检测任务反馈 G1/G2

- 在真实 detector 上枚举 legal hard one-swap，比较 detector-loss utility 与 surrogate
  gradient 的 sign/rank；
- 只有 alignment GO 才运行 G1；
- `G1-G0` 单独证明 detector feedback，`G2-G1` 只证明 companion 稳定性；
- 最终有效性仍由 terminal mAP 裁决，detector loss 本身不是论文结论。

### R5：论文闭环

- U 与最佳 learned arm 三 seeds；
- fixed-budget curve：至少 K=384/256，低预算只在稳定后扩展；
- dense、exact-uniform、random、actionness top-k、simple delta、old V8、final DUCA；
- IoU-wise mAP、短/中/长动作、start/end error、双侧数量、burst concentration 与可视化；
- 一个第二 official TAD backend，证明插件接口和趋势，而不是重新调一套 selector；
- 训练成本与两种推理模式的完整 latency/FLOPs/memory/energy 账本。

### 6.1 从当前实验到论文主表的总计划

| 阶段 | 当前要回答的唯一问题 | 进入条件 | 退出证据 | 论文角色 |
|---|---|---|---|---|
| V8 终局封存 | 旧 Gaussian-mass scorer 为什么没有把已学到的粗动作证据转成更好的边界分配？ | `63e25eb` 精确提交与训练侧 holdout | 三个 P0 候选、winner/失败原因及 U/G0/G1/G2 终局（若 DAG 继续） | 负结果与设计动机，不是最终主方法 |
| R0 Oracle 可达性 | 在固定 K 下，允许边界微簇且保留全局覆盖的可行域本身能否在 detector mAP 上超过均匀采样？ | V8 证据封存 | exact-uniform、unrestricted Oracle、projected Oracle 的同协议 mAP 与 K/G/radius/quota 冻结表 | 上限与可行域依据 |
| R1 实现门禁 | 最终边界微簇目标、DP、梯度所有权和 official backend 是否数学与代码一致？ | R0 有正 headroom | focused tests、真实 batch/CUDA 一步、配置白名单和成本字段全部通过 | 可复现性证据 |
| R2 P0 机制实验 | 粗模型没学好，还是旧 selector 没用好？新 burst objective 是否优于 simple delta 和旧 V8？ | R1 通过 | coarse calibration 与 center/bilateral/quota/fairness 指标；最早过门禁 checkpoint | 前端机制消融 |
| R3 U vs G0 | 不依赖 detector feedback，正确的选帧目标能否直接提高最终 TAD mAP？ | R2 通过 | same-commit seed-0 terminal EMA；先只跑 U/G0 | 第一主结果锚点 |
| R4 G1/G2 | 真实检测任务反馈是否进一步改善 G0，而不是只让 surrogate loss 变小？ | legal hard-swap alignment 通过且 G0>U | G1-G0 与 G2-G1 terminal EMA | 任务适配与稳定性消融 |
| R5 论文闭环 | 结果能否跨种子、预算和 detector 保持，并在完整成本上成立？ | seed-0 GO | 三种子、K=384/256、第二 backend、完整成本与诊断 | 论文主表、泛化表和成本表 |

当前执行状态：V8 已完成并作为负证据封存；R0 的第一次运行因诊断 Oracle 合同错误而
作废，corrected R0 正处于精确提交独立复审；R1--R5 尚未解锁。当前实验想证明的不是
“模型已经超过 65”，而是先证明 **同一 K/G 下允许 Oracle 式边界微簇的合法可行空间，
在最终 detector mAP 上确实优于 exact-uniform**。只有这一步成立，学习这个空间才有科学意义。

V8/P0 已完成负向封存：Job `1178989` 的 12 个候选全部结束，`winner=null`；粗动作
AUROC 最高约 `0.620`，但 learned transition 始终弱于 simple delta，endpoint distance
始终差于 uniform，因此旧 scorer 被机制门禁主动停止。第一次 R0 Job `1179392` 又暴露了
privileged Oracle 的 fixed-nearest-Q 假不可行，并在 detector evaluation 前终止；它没有
mAP。当前实际位置是 **corrected R0 exact-commit 复审**：候选 `22555a4` 已通过远端
focused tests 与真实失败样本 replay，正在等待独立 MAX 对 corrected R0 给出明确裁决。
在 corrected R0 形成 paired-bootstrap mAP headroom 前，P0、full-model gate 与
official-60 均不得提交。

### 5.5 当前独立审计裁决（2026-07-22 04:26）

独立 MAX 对精确提交 `aa3352ecf803c81d007a62ed5398667d9551684b` 的裁决为
`HOLD_FIX_REQUIRED`。审计没有否定边界微簇目标、全局 exact-K/max-hole DP、selected-axis
映射、no-leak 或 official detector。正式部署前只允许修复两个证据合同：

1. 在提交时冻结 AdaTAD 预训练权重路径与 SHA，并由 P0/frontend/gate 每个消费者在执行前复核；
2. 在真实 legal hard-swap alignment 通过前，由 production runtime binder fail-closed 拒绝 G1/G2。

远端 `139+23` 测试不是部署许可。旧 V8 Job `1178989` 已因机制 HOLD 终止；当前没有
boundary-burst CUDA、R0、P0 或 official-60 作业，因此论文位置仍是 R0 前的 R1 合同修复。

### 5.6 精确修复候选（2026-07-22 04:45）

上述两个 blocker 已在不改变模型结构的前提下最小修复为
`f629ad79461941f405bc2028f087034abd17a840`：提交时冻结的 AdaTAD pretrain path/SHA
由 P0、每个 frontend arm 和 gate 独立复核；G1/G2 在 production runtime binder 入口无条件
拒绝。远端精确快照通过受影响 DUCA `63 passed`、必需 C3 `23 passed` 以及
pycompile/bash/HEAD/clean。全新无上下文 MAX 正在审查；在其明确给出
`GO_TO_REAL_CUDA_GATE` 前，仍不得运行 CUDA/R0/P0/official-60。

### 6.2 论文最终结果表

1. **主性能/效率表**：THUMOS14 + official-derived AdaTAD，dense、random、
   actionness top-k、simple delta、exact-uniform、old V8、final DUCA；报告 K=384/256、
   IoU-wise mAP、Avg-mAP 与完整部署成本。
2. **方法消融表**：U、G0、G1、G2，以及 anchor/bilateral/quota/fairness、coarse hidden、
   protected feedback 的单变量消融。
3. **泛化表**：冻结同一插件合同，在预注册的第二 official backend 上比较 uniform 与
   final DUCA，不按结果挑 detector。
4. **上限与选帧质量表**：unrestricted/projected Oracle headroom、端点距离、边界双侧
   帧数、3--5 帧微簇分布、短动作覆盖、最大空洞和重复率。
5. **成本表**：dense、研究复现模式和真实部署模式的 decode/CPU/H2D/probe/DP/
   VideoMAE/head/NMS 分项 p50/p95、吞吐、FLOPs/MACs、峰值显存与能耗。

### 6.3 Pro 讨论边界

当前**不再需要一次开放式、重新发散方向的完整 Pro 讨论**。精确提交审查、目标审查和
Uni-AdaFocus/EU-CRR 审查已经完成，继续开放式讨论只会重复改名和重造 selector。后续只保留：

1. R0 结束后，对冻结的 `K/G/radius/quota` 和 projected-Oracle headroom 做一次有界统计审查；
2. R1 完成后、提交 R3 长训前，对 exact commit 做一次逐行 diff/数学/official-backend 门禁审查；
3. R4 前只审查 legal hard-swap alignment，不重新讨论主架构。

只有触发第 7 节停止条件且现有证据无法裁决时，才允许重新发起方向级 Pro 讨论。

EU-CRR U0/U1 只在 R0 后作为 G24 的正交表示诊断；它不阻塞 R1--R3，也不能替代
boundary-burst 选帧证据。

第二 backend 必须在查看其 DUCA mAP 前按工程兼容性预注册；默认优先使用仓库已有
official TriDet 配置。只有缺少同等 raw-frame/VideoMAE 接口时才改用 TemporalMaxer，
并在 manifest 中记录非性能原因，禁止在多个 head 之间按结果挑最好者。

## 7. GO / HOLD / KILL

### 主方法 GO

当前 THUMOS/AdaTAD 锚点必须同时满足：

- terminal-EMA Avg-mAP `>=65.00`；
- 相对 same-commit matched U `>=+0.20`；
- mAP@0.6 和 mAP@0.7 相对 U 的下降均不超过 `0.20`；
- 三种子均值仍为正，并报告方差；
- 论文部署模式的完整实测推理成本低于 dense route。

历史 Oracle 约 78 和历史 uniform 约 65 只能作为背景；只有同提交、同协议重测结果能
进入主表。报告 final DUCA 关闭了多少 matched feasible-Oracle headroom，但不把 Oracle
当作可部署方法。

### 有界停止条件

- projected Oracle 无 mAP headroom：修改 K/G 可行域，不修改 scorer loss。
- corrected P0 不优于 simple delta：停止 learned scorer，定位 coarse evidence。
- G0 不优于 U：终止该 fixed-K learned allocation 主张。
- hard-swap alignment 不通过：保留 G0，删除 detector-feedback 主张。
- G1 不优于 G0：删除 task-adapted feedback，只保留 boundary-supervised selector。
- 完整成本不低于 dense：删除端到端效率主张，只能报告 heavy-backbone frame reduction。
- 第二 detector 不复现趋势：删除 plug-and-play/generalization 主张。

## 8. 实现边界与当前阶段

未来实现不得建立另一套 selector/decoder。R0 GO 后只允许在现有 V8 主干做以下有界
修改：

- `opentad/models/duca/acquisition.py`：扩展 transition scorer，输出有限 burst offset
  profile、saturating union utility、归一化 selector losses 与诊断；
- 现有 structured exact-K/max-hole solver：复用原求解器和 hard/soft family，只接收新
  utility，不创建第二 decoder；
- `opentad/models/selectors/duca_online_frame_selector.py`：接入 train-only endpoint
  targets、loss ownership、original-time metadata 与 two-mode cost audit；
- 现有 U/G0/G1/G2 config family：只替换 objective/必要字段，不复制四套模型类；
- focused tests 与 Oracle/cost tools：增加 bilateral/quota/dedup、hard-swap alignment、
  deployment materialization 和 same-commit result aggregation。

ActionFormer head/loss/NMS、ChronoTransport、Spatial-Zoom、SparseHead 和历史 local-cell
代码不在本次修改范围。

当前 Wiki 最后封存状态中，V8 是 `mechanism_hold_negative_diagnostic`；boundary-burst
successor 是 `implemented_exact_candidate_under_independent_reaudit`。候选模型主体和
runtime/evidence 修复已经存在，但尚未通过全新独立 MAX 与真实 CUDA 门禁。R0 尚未形成
hash-bound headroom，R2 没有
P0 winner，R3 没有 terminal mAP，因此还没有 V9、没有经验支持、没有可写入论文主表的
final DUCA 结果。

## 9. 最终论文可声称什么

只有 R0--R5 闭环后，论文才可声称：

> DUCA 利用低成本动作状态证据间接定位状态转变，以 Oracle 标定的双侧边界微簇和
> 全局 exact-K 约束分配重计算，在固定预算下优于均匀采样，并在完整实测成本下降时
> 保护离线 TAD 的高 tIoU 定位。

在此之前只能称为 `canonical_successor_designed_not_implemented`。实现必须扩展现有
V8 scorer/DP，不得新建同义 selector、decoder、detector wrapper、worktree 或启动器。

## 10. 2026-07-22 05:45 当前精确执行位置

当前唯一候选为
`codex/duca-boundary-burst-20260722@86f7663a94d628eace316d17e31db7043f731f75`。
`7b9ad0b` 已经封闭 terminal checkpoint、官方评测、prediction 与 aggregate 的哈希链，
但真实历史 selected-axis sidecar 证明 production audit builder 没有写出 validator 强制要求的
`formal_protocol` 和 `training_profile`；手工测试夹具曾掩盖这一差异。`86f7663` 只修复该
producer/consumer 合同，并未改变模型、损失、K/G、全局 DP 或 official detector。

本地回归为受影响 DUCA `60 passed, 1 skipped`、C3/update evidence `29 passed`；远端干净
精确快照为 DUCA `64 passed`、C3/update evidence `29 passed`，compile、bash、HEAD 与 clean
检查通过。全新无上下文 MAX `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac` 正在完整审计模型、
训练、评测、聚合和 Slurm DAG。当前仍是 **R0 前的 R1 最终精确提交复审**，不是主实验
running；在其给出 `GO_TO_REAL_CUDA_GATE` 前不提交 CUDA/R0/P0/official-60。

本次审计通过后无需开放式 Pro 讨论。后续只保留 R0 后 K/G/radius/quota 统计审查、R3
长训前 exact-commit 门禁确认和 R4 legal hard-swap alignment 审查。

## 11. 2026-07-22 06:21 独立复审裁决与当前阶段

无上下文 MAX `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac` 已完成对精确提交
`86f7663a94d628eace316d17e31db7043f731f75` 的审查，裁决为
`HOLD_FIX_REQUIRED`，不是方向 KILL。审查确认模型主体、双侧边界微簇、全局
exact-K/max-hole DP、selected-axis 映射、no-leak 与 official-derived AdaTAD 主体没有新的
P0 结构阻断，但部署前仍有三项 P1 证据合同必须关闭：

1. R0 必须真实包含 exact-uniform、projected Oracle 与 unrestricted GT Oracle，并通过
   逐视频 bootstrap 的置信区间冻结唯一的最弱可行 `K/G/radius/quota`，不能只看单点 mAP；
2. R0 消费端必须独立复核并重算 official mAP，绑定 prediction、checkpoint、config、
   annotation、class map、evaluation config、train-only subset 与 blocked-video hashes；
3. simple `abs(delta p_action)` 必须进入与 learned policy 相同的 global exact-K/max-hole DP，
   并把“corrected selector 不优于 simple delta 则停止”写入真实候选门禁。

同时关闭三个不改变方法身份的 P2 缺口：R0 burst 统计消费 crop endpoint validity、Slurm
提交日志逐作业原子落盘以防重复提交、增加一次不 mock official evaluator 的生产链集成测试。
在这些修复形成新 exact commit、Linux 回归和全新独立复审 GO 前，真实 CUDA、R0、P0 与
official-60 继续阻断。当前论文阶段严格写作 **R0 前的 R1 证据合同修复**；没有新的 mAP，
也没有主实验 running。

这次 HOLD 不触发开放式 Pro 讨论。修复后只需要一次针对新 exact commit 的有界逐行复审；
R0 产出后再做统计裁决，R4 前再做 legal hard-swap alignment 审查。

## 2026-07-22 06:54 evidence-contract implementation note

`4ec3e078a3aad834ffe504d74d414bf7e2b6fad3` implements the bounded R0/simple-delta/
submission-evidence repairs without changing this model contract. The final artifact remains the
same offline fixed-budget pre-backbone acquisition plugin; no new selector, decoder, fusion path,
dynamic budget or detector backend was introduced. Status is only
`implemented_local_tested_pending_linux_and_independent_max`; R0--R5 empirical gates and V9 remain
unproven.

## 2026-07-22 07:15 current execution position and R0-decision propagation

The clean Linux snapshot for `4ec3e078a3aad834ffe504d74d414bf7e2b6fad3` has now passed the
affected DUCA `109 passed`, mandatory C3/ASFormer `23 passed`, pycompile/bash/exact-HEAD/clean-tree
checks and a no-submit `PRECHECK_ONLY=1`. Therefore the exact status is
`linux_tested_pending_independent_max`, not pending Linux and not experiment running.

One execution-policy mismatch remains under the same independent audit: R0 freezes one weakest
feasible projected family, but the current scripts make Gaussian, R2Q3 and R4Q5 all mandatory and
queue all four official-60 arms. The canonical claim-driven order is stricter and cheaper:

1. R0 freezes one projected burst geometry or kills the current feasible set;
2. R2 trains/evaluates diagnostics, but only the R0-selected burst family is a mandatory learned
   winner; simple delta and Gaussian are controls, not vetoes;
3. R3 first runs matched U versus that selected G0 only;
4. alternate burst/Gaussian arms enter ablation only after the main U/G0 anchor exists.

This routing correction is not yet implemented and must not be applied before the fresh no-context
MAX verdict. It does not authorize changing the model objective, K/G, global DP or detector.

## 2026-07-22 07:30 exact split-evaluator repair

The fresh review of `4ec3e07` returned `HOLD_FIX_REQUIRED`, not method KILL. The split artifact names
were previously interpreted backwards: a consumer-specific block list stores the videos excluded
from that consumer, so the holdout consumer's list contains the train assignment. Exact successor
`f90595d8620e42e8e3d74722f2ab48126c6b65f2` now builds the R0 evaluator blocked JSON from
`frontend_holdout_block_list`, and a semantic test proves blocked IDs equal manifest `train_videos`
while evaluator targets equal `holdout_videos`.

The clean remote snapshot passed affected DUCA `168 passed, 2 skipped`, mandatory C3/ASFormer
`23 passed`, compile/bash/HEAD/clean and a no-submit precheck with manifest SHA-256
`14f345dc53b246b036ba1c80c993454c0d83a1173aae8481b54ac0f8647c8a2c`. A new independent MAX is
running. This still does not authorize CUDA/R0/P0/official-60, does not create V9 and does not alter
the R0--R5 paper plan.

## 2026-07-22 07:54 stage-authorized R0 execution

Independent MAX `019f8701-edaa-7e83-a572-49024b524098` did not grant a global GO, but it explicitly
granted R0 while withholding P0/CUDA downstream unlock and official-60. Accordingly, only R0 Job
`1179392` was submitted from exact commit `f90595d8620e42e8e3d74722f2ab48126c6b65f2`; no dependent
job exists. Its purpose remains the R0 question in section 6.1: whether the projected G2
boundary-burst feasible set has paired-bootstrap detector-mAP headroom over exact uniform, and
whether unrestricted Oracle separates a too-tight coverage constraint from a weak selection idea.

P0 is now blocked only on evidence closure: recompute summaries from per-sample records, bind the
real CUDA P0 gate, verify frozen official-ASFormer source identity, prove matched evaluator/input
identity across arms, preserve `gt_boundary_validity` in the full-model gate and atomically write the
final suite. These fixes do not change the model contract and proceed in parallel with R0.

## 2026-07-22 08:40 R0 failure diagnosis and exact-quota correction

R0 Job `1179392` did not produce detector mAP. The 40-video/124-window export
completed without GT/teacher leakage or detector-backbone execution, but the
Oracle builder stopped on `video_validation_0000206|0` before any detector
evaluation. The failure is not evidence against G2 or boundary clustering:
the old diagnostic solver fixed the nearest Q positions around every endpoint,
whereas the canonical contract only requires jointly choosing at least Q
positions inside the radius. The fixed union is infeasible for dense-action
windows, while the intended center/quota/bilateral/exact-K/G2 constraints have
an exact HiGHS witness.

Exact candidate `22555a4e830ce24f9bb516897b1bb7f44b70c188` repairs only this
privileged R0 feasible-space implementation and the already-authorized evidence
chain. It does **not** change the deployable selector, coarse model, K=384,
G=2, selected-axis detector input or official AdaTAD/ActionFormer backend.
Real-sample replay now succeeds for U, R2Q3, R4Q5 and unrestricted Oracle.
Therefore the execution position is reset to **R0 exact-commit re-audit**:

1. independent MAX must authorize corrected R0 only;
2. corrected R0 must produce paired-bootstrap projected-Oracle mAP headroom;
3. only then may P0 train the existing coarse/transition/burst frontend;
4. only a P0 winner and full-model gate may unlock matched terminal-EMA U/G0;
5. G1/G2 remain forbidden until real legal hard-swap alignment.

This is a bounded correction, not a new method or an invitation to relax G to
10/15. G10/G15 remain possible later ablations only if corrected R0 evidence
shows a scientifically justified coverage trade-off; they must not hide a
solver-contract bug.

## 2026-07-22 current implementation clarification after e49 Pro audit

The canonical design goal still includes Oracle-like bilateral boundary bursts,
but the current `e49ef696` model implements a **soft bilateral/quota objective**.
Its hard structured decoder guarantees exact-K and max-hole only. It does not yet
guarantee a mandatory sample on both sides of every predicted center or a hard
per-boundary quota. Therefore sections describing bilateral/quota behavior are
design objectives unless explicitly labeled soft-loss metrics. A mandatory hard
decoder is a conditional successor and may be introduced only after official mAP
plus center/left-right/quota mismatch diagnostics show that soft-hard mismatch is
the bottleneck.

The current training/research path decodes and transforms the complete low-resolution
sequence and moves it to the device before the selector saves heavy VideoMAE work.
It demonstrates post-decode heavy-backbone frame reduction. The paper deployment
path with low-cost dense proxy plus selected high-resolution random-access
materialization remains `designed`, not `implemented` or `tested`. Until that path
and paired cost accounting exist, the contract forbids claims of K-only decode,
K-only H2D or end-to-end video-I/O savings.

`8d85929ea04dc40f1eb0c3cc806061ce3b071d3f` is an execution/evidence successor of
the same `e49ef696` model. Its independent official-validation jobs are exact-uniform
`1180111`, Gaussian G0 `1180112`, R2Q3 G0 `1180113` and R4Q5 G0 `1180114`.
They have no cross-job dependencies. At this update they are running without
terminal epoch-59 EMA mAP, so model efficacy remains unproven. The former serial
R0--R5 DAG and its contaminated R0 family gate are no longer the canonical
execution route.

Before any future R5 paper aggregate, evidence contracts must additionally:

1. reopen raw predictions and independently rerun the OpenTAD evaluator;
2. enforce matched hardware/session/software/input identities for candidate/dense
   cost pairs;
3. bind the dense baseline through a sealed commit/checkpoint/config/source receipt,
   including the expected source commit rather than accepting any 40-hex SHA.

These are evidence-only requirements and do not modify the current selector,
decoder or official-derived detector.

## 2026-07-23 rate-curriculum training contract

The sampling-rate route now has one canonical two-phase curriculum, not
parallel frontend and joint-training alternatives.

1. **Stage 1 -- uniform full-model warmup.** Train the complete THUMOS
   training subset with exact uniform `K=384`. The detector learns on that
   fixed observation set while the low-cost spatial stem, official-ASFormer
   temporal trunk, actionness head, and transition scorer are jointly trained
   with binary actionness, transition-distribution, and transition-boundary
   supervision. Checkpoints 5/10/15/20/25/30 record actionness AP/AUC/Brier/ECE
   and transition-boundary support on the fixed validation subset. These are
   convergence diagnostics, not selector decisions or paper mAP.

2. **Stage 2a -- low-LR joint adaptation.** Initialize the *complete* model
   from the Stage-1 terminal EMA but reset optimizer, scheduler, AMP scaler,
   and selector curriculum clock. Across the first 3,000 updates, policy
   alpha, contribution distillation, detector-to-density transport gradient,
   and ASFormer policy adaptation increase smoothly from zero. Coarse losses
   remain strong during this handoff.

3. **Stage 2b -- TAD-led joint learning.** The detector loss remains `1.0`;
   actionness, transition, and boundary supervision taper only to
   `0.25/0.10/0.25`, respectively. They are intentionally nonzero so the TAD
   objective cannot erase the coarse action/state representation.

Full-model EMA transfer is strict and hash-bound. It carries neither optimizer
nor scheduler state, so Phase 2 is a new curriculum phase, not a disguised
continuation. The exact implementation at
`codex/duca-density-transport-20260723@04b7df4` passed its remote PyTorch
focused suite (`15 passed`) plus Python and shell syntax checks. Status:
Stage 1 completed all 30 uniform-K=384 epochs under Job `1182391` and sealed
the terminal EMA checkpoint `epoch_29.pth` (SHA-256
`7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e`). Its
five-epoch validation curve ends at `60.39` Avg-mAP at epoch 30; this is a
30-epoch warmup diagnostic, not a matched 60-epoch performance result.

The old Stage-2 process made zero optimizer updates. It inherited the legacy
P0 successful-update switch while declaring no formal P0 protocol, so the
legacy variant binder rejected an empty variant before model construction.
Commit `b554f04c8d58721c7648bb8c5afcc91d63577d6d` declares the curriculum as
non-P0 and adds an SHA-bound Stage-2-only recovery path. The first recovery,
Job `1190439`, failed at zero runtime before Python, model construction, or an
optimizer update because its outer Slurm wrapper sourced `/etc/profile` under
`set -u` and the site profile read an unset `XDG_DATA_DIRS`. This is a launcher
environment failure, not model evidence. The sole replacement, Job `1190528`,
uses the same commit and sealed Stage-1 EMA, safely disables nounset only while
sourcing the site environment, and runs only Stage 2 under
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_b554f04_stage2_recovery_20260726_040504`.
Status: `experiment_running / no_stage2_optimizer_update_yet`. No mAP claim
follows until its Stage-2 terminal EMA evaluation.

At 2026-07-26 04:05 +08:00, `1190528` passed the repaired environment launch,
created the Stage-2 work directory, logged the strict full-model
`state_dict_ema` initialization from the sealed epoch-29 checkpoint, reset the
selector loss-schedule step, and entered epoch 0. No finite-update receipt or
performance result is available yet.

At 04:31 +08:00 it had reached `duca_schedule_step=350` with finite loss,
stable 9.9 GB memory, and fixed requested budget K=384. Two AMP replays each
recovered on attempt 1 of 8; neither exhausted the replay budget. The density
policy alpha is increasing from zero, while detector-gradient weight remains
zero during its configured first-1,000-successful-update warmup. The persistent
actionness, transition, and transition-boundary weights remain nonzero. No
five-epoch diagnostic or mAP is available yet.

The Stage-2 epoch-5 EMA checkpoint `epoch_4.pth` was sealed at 04:42 +08:00
(SHA-256 `4016ca083dd56286b6a32edce92f085945ce415c0b237c1fc43b0cb7ad1cc2a3`).
The live runner's scheduled full validation then reported `60.52` Avg-mAP
after epoch 5. It is a learning-curve diagnostic only and is explicitly barred
from checkpoint selection or any comparison to the terminal 60-epoch result.

The separate read-only evaluator additionally exports the fixed-window
actionness, calibration, transition-boundary, and selection diagnostics.
`1190606` exited before Python or model construction because its wrapper omitted
the canonical environment source. Replacement `1190626` then reached
`tools/test.py` but failed before DDP, model construction, or inference because
the resolved-config hash could not serialize the frozen `CellCFTrainingProtocol`.
Commit `a1bf61d` makes that protocol serialization deterministic and passed the
focused 9-test suite locally and on the remote exact checkout. Job `1190633`
then reached backbone construction but failed before the Stage-2 checkpoint was
loaded because the read-only wrapper had omitted the training launcher's
absolute pretrain override; its relative default did not exist in the remote
repository. Job `1190637` reused the same epoch-5 EMA SHA, state key, and
non-selecting manifest and added only that already-frozen absolute pretrain
path. It successfully constructed the DDP model, loaded the EMA checkpoint,
and began official testing, but was deliberately cancelled after `10/396`
windows when the wrapper was found to omit the required fixed 64-window
actionness/calibration/boundary export. It produced no final metric or quality
artifact and is read-only workflow evidence, not model evidence. The next
replacement must retain all bindings and add that fixed export before starting.
Job `1190643` is that unique replacement: it is bound to the same e5 EMA SHA,
uses no training mutation and `selection_rule=none`, runs full official
validation, and then exports/analyzes exactly 64 fixed validation windows.
Rejected explicit-memory requests created no jobs and are not experiment
evidence.

At 2026-07-26 06:29 +08:00, Job `1190643` completed `0:0` with the full
official e5 result: Avg-mAP `60.521318%`
(`76.917970/71.979260/63.698867/52.930814/37.079679%` at tIoU
`0.3/0.4/0.5/0.6/0.7`). This independently confirms the live e5 curve but is
explicitly non-selecting and not comparable with terminal matched uniform.
The fixed 64-window selector-only export passed its no-leakage provenance
checks. It records coarse actionness AUPRC/AUROC/Brier/ECE
`0.343840/0.577626/0.202376/0.012158`, learned action enrichment `1.003583`
(CI `0.998122--1.008924`), and r0 boundary-recall delta versus matched uniform
`-0.017789` (CI `-0.075717--0.046480`). The stronger same-feasible pure-delta
microcluster coverage is a diagnostic warning that the learned policy has not
yet converted available boundary signal into selected slots; it is not a model
claim.

Job `1190528` subsequently failed closed after 1,000 finite Stage-2 updates.
It sealed `epoch_9.pth`, completed its scheduled e10 diagnostic validation at
Avg-mAP `61.62%` (`77.94/72.84/64.61/54.34/38.39%`), then immediately after
`Epoch 10 started` raised `FloatingPointError` before AMP scaling. The e10
number is a non-selecting curve point, not a terminal result. No terminal EMA
or model verdict exists for this arm. Because the failure occurs at the
detector-gradient/contribution warmup boundary, the only admissible next step
is a hash-bound, read-only e10 single-batch replay from `epoch_9.pth` that
reports finite status of each loss component and schedule weight. A recovery
may be considered only after that diagnosis, must retain strict loading, and
may change only the implicated numerical path.

Job `1190683` is the sole admissible pre-recovery diagnostic. It replays only
batch 0 of epoch 10 under Slurm from `epoch_9.pth/state_dict` SHA
`3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`, exact
commit `3a87132d60b0a328ccbe9d153e795a7ce3987911`, strict sealed Stage-1
initialization, and the same absolute pretrain file. Its manifest fixes
`training_mutation=false`, `selection_rule=none`, no backward, no optimizer,
no scheduler, and no EMA. Pending or completed status is numerical-component
evidence only and cannot be a checkpoint choice or offline TAD performance
claim.

`1190683` completed `0:0`: epoch-10 batch 0 at checkpoint selector step 1000
has finite cost `3.402128`, every recorded loss component is finite, and all
detector-gradient/contribution/utility weights are zero. The original
training log contains a backward-kernel warning before the subsequent
pre-AMP exception, so batch 0 reached its successful post-optimizer schedule
advance. This excludes batch 0 at step 1000 but not the next forward at step
1001. Any follow-up is constrained to a no-parameter-mutation, in-memory
selector-clock probe of batch 1 at step 1001; it cannot emulate the optimizer
step, change checkpoints, or be represented as a recovery or performance
experiment.

Job `1190699` completed `0:0` with no mutation of parameters or checkpoints.
Epoch-10 batch 1 at in-memory selector step 1001 is finite (cost `4.324651`),
including the first nonzero detector-gradient/contribution weights
(`1.542125e-7`/`6.168501e-7`) and cls/reg contribution losses
(`4.095707e-6`/`4.101499e-6`). The single schedule entry cannot alone explain
the historical failure. No Stage-2 recovery is permitted without isolating a
post-batch-0 state or nondeterministic input/kernel cause.

At 2026-07-26 18:07 +08:00, repaired continuation `1191880` was cancelled
after the controller found that its inherited config enabled intermediate mAP
checkpoint selection. Before cancellation it completed 700 finite updates
through epoch 17, with zero non-finite attempts/replays/exhaustions and no
optimizer/selector/scheduler/EMA mismatch. The pointer did not feed back into
training, but the run is still protocol-invalid for this course: it wrote
`best_validation_ema.json` after its epoch-15 curve evaluation. That
non-selecting operational value was `62.403751%` Avg-mAP
(`79.036658/73.799613/64.984514/54.566999/39.630969%`), but is not an offline
TAD performance result and cannot select a checkpoint. No required epoch-15
AP/AUC/Brier/ECE/state-transition-boundary diagnostic was exported. The next
continuation must explicitly enforce diagnostic-only intermediate validation,
reuse sealed e9, and retain terminal epoch-59 EMA OpenTAD official mAP alone.

## Sealed K=384 curriculum evidence

The current complete K=384 course is operationally complete at clean exact
commit `42dba3f90b37243e7965d18b6707e88e81bf7109`. Its sealed epoch-59
`state_dict_ema` OpenTAD receipt reports `65.385724%` Average-mAP and
`80.193191/75.662461/68.607247/58.581766/43.883956%` at tIoU
`0.3/0.4/0.5/0.6/0.7`. However, Stage 1 performs 30 epochs of full-model
training and Stage 2 performs another 60 epochs, so the endpoint consumes a
90-epoch optimization budget. The approximately `+0.896pp` difference from
the 60-epoch exact-uniform `64.49%` anchor is therefore an over-budget
curriculum comparison, not a fair official-60 DUCA gain. The best observed
Stage-2 epoch-50 point is `65.650497%`, but also consumes 80 total epochs.
Both values remain useful diagnostics; neither is the final fair-budget paper
claim.

## Next candidate contract: physical-time budget transport

The next model candidate must address the selected-axis mismatch before adding
more auxiliary plugins:

1. Learn a nonnegative density over the original physical timeline and decode
   exactly K monotone observations by inverse-CDF transport.
2. Pass true timestamps and left/right physical gaps to the post-backbone
   temporal model through a zero-initialized residual or equivalent controlled
   interface.
3. Perform detector prior placement, positive assignment and boundary
   regression in physical coordinates rather than selected rank.
4. Retain the uniform density as an exact special case and preserve exact
   budget conservation.
5. Compare raw contribution-magnitude distillation with normalized
   ranking/transport supervision; do not stack both by default.
6. Use one total 60-epoch budget for the main comparison. Required arms are
   joint-from-scratch, a preregistered short-warmup curriculum, and uniform;
   the old 30+60 course is retained only as an over-budget upper diagnostic.

This candidate is `designed`, not `implemented`, `tested`,
`empirically_supported` or `paper_ready`.
