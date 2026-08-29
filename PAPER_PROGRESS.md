# ZoomToken 论文缩略报告

## 当前论文判断（先读）

- **问题：** 当前观测上的连续 K64 原生支持，能否在保护高 tIoU 定位的同时降低离线 TAD 的真实端到端成本？
- **候选机制：** BPNS-R1 在 VideoMAE 前保留一个连续无孔洞的 `8×8/K64` 支持；所有保留 token 仍完整执行 12 层 VideoMAE-S 与既有 Adapter，不使用历史 cache、carry 或深度跳过。
- **已有证据：** 单个 seed-42 中，K100 与 R1 的 final-EMA 为 `68.51/61.19/46.27` 与 `69.07/61.14/46.57`。v004 同硬件八-pass 成本闭环进一步测得 R1/K100 的 p50、gross energy 与 peak allocated/reserved memory 比值分别为 `0.98493/0.93500/0.75130/0.68966`。
- **不能主张：** 36% 原生 token 减少没有转化为冻结要求的至少 5% p50 改善；端到端 p50 只下降 `1.51%`。因此不能再把 BPNS-R1 作为当前效率 headline，也没有多种子、跨硬件、跨检测器或跨数据集证据。
- **当前实验：** v002 job `1258299` 已永久封存为无效率结果的准入协议失败。Pro 冻结的 v003 clean/pushed revision `8a59d655005b9030d8ea5dc17ee2620844cb587b` 通过 local/remote `21 passed`、fresh Critic `PASS`、fresh result-blind Evaluator `PRE_RUN_READY` 与 precheck `1258524 COMPLETED 0:0`。唯一正式 job `1258526` 在 `g0063` 运行 `05:33:32` 后终态 `FAILED 1:0 / FAILED_PROTOCOL_INVALID`：八个 prediction/evaluator pass 已保存，但短动作 evaluator 配置遗漏 registry 的 `type`，在形成成本、功耗、显存、短动作或边界汇总前确定性终止。
- **Pro 裁决：** v004 的全新 exact-Project `GPT-5.6 Pro` 复盘已返回 `PIVOT`，角色合同 `KEEP`。工程为 `PASS_STRONG`，协议为 `VALID_WITH_DISCLOSED_POWER_UNCERTAINTY`，科学为 `VALID_NEGATIVE_FOR_STANDALONE_FULL_STACK_LATENCY_HEADLINE`。BPNS-R1 永久退出独立效率 headline，只保留为空间支持可行性、局部 GPU/显存/能耗归因和负系统结果。
- **BPNS 主线终态：** 唯一 v004 job `1260095` 已 `COMPLETED 0:0` 并形成完整八-pass 证据。能耗下降 `6.50%` 通过门槛，但 p50 仅下降 `1.51%`，未通过冻结的联合门；终态为 `STOP_BPNS_R1_EFFICIENCY_HEADLINE`。不存在 v005、重放或阈值修订授权；完整结果只送一次 fresh Pro 独立裁决。
- **当前唯一任务：** `ZOOMTOKEN-R1-TAR32-FKV-TERMINAL-VALIDATION-AND-K100-MATCHED-FULL-STACK-COST-CLOSURE-v001`。先只读核验唯一 seed-42 job `1260166` 的 60 轮、epoch-59 EMA、原始 evaluator/prediction 与 `[64,32]×6` route ledger；若终态模型与协议有效，则从 `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7` 建立模型零修改的最小成本后继，并只提交一个 K100/TAR32 八-pass 同 GPU 成本作业。这仍是 composition-first falsifier，不是已成立方法、新颖性或效率证据。
- **证据边界：** v004 的 prediction identity、6,336 cost rows、929,889 power rows、profile、terminal receipt 和离线诊断均完整。R1−K100 准确率与边界变化混合；不能主张边界保护。第 3 号 K100 pass 有一次约 `2.805 s` 功耗采样间隙，必须作为能耗不确定性披露，但它不改变 p50 门失败，也不授权重放。

以下按日期和实验族保留完整证据与负结果；运行标识用于定位原始材料，不替代上述科学判断。

> **2026-08-28 正式成本回放 v002 终止：** 从冻结基线 `b7357817…` 建立的最小 clean/pushed 候选 `e9323448…` 只修正 accuracy-parity 数值合同及 focused tests；`13 passed`，独立 Critic 为 `PASS`，结果盲 Evaluator 为 `PRE_RUN_READY`。唯一正式 job `1258299`（`zt-bpns-r1-pv2-e9323448`）在 `g0048` 运行 `01:13:06` 后以 `FAILED 1:0` 终止：首个 R1 pass 的 `mAP@0.6` 与冻结两位小数 reference 相差 `0.0530390970556900 pp`，严格超过 inclusive `0.05 pp` 门槛。八个 pass 未完成，结果根为空，未发布 profile/终结收据、预测、成本样本或功耗轨迹。因此该轮是协议有效的 replay admission failure，不是模型性能或效率结果；在 fresh Pro 裁决前不放宽门槛、不重提、不追加实验。

> **2026-08-28 Pro 终态复盘：** 同一有效 invocation 在 verified Pro 路由下完成，裁决为 `REVISE`、角色合同为 `KEEP`。Pro 认可 v002 对冻结合同的工程执行，但指出 reported-2dp 点距离不能识别 raw-to-raw `0.05 pp` 一致性；因此 v002 永久封存为 replay-admission 协议失败。BPNS-R1 仅保留“单种子准确率可行、效率未知”的窄主张。唯一 v003 任务以执行身份与测量完整性为硬门，并在八 pass 完整回放后按预注册的延迟、能耗和边界门决定继续、定向修订或停止效率候选。

> **2026-08-27 正式成本回放终止：** `ZoomToken-BPNS-R1` 同硬件 final-EMA 重放 job `1257281` 在 `g0003` 运行 `00:38:04` 后以 `FAILED 1:0` 终止。它完成了 K100 的完整 validation，但在首个 pass 的结果一致性门处发现当前回放 `mAP@0.7=46.246663` 与预填历史值 `46.27` 不一致，profiler 主动抛出 `RuntimeError`；因此未进入后续 R1/ABBA+BAAB 完整测量，也没有发布 `profile.json` 或 `terminal_receipt.json`。这是回放准入/数值绑定失败，不是模型性能或效率结论；现阶段仍不能声称 R1 实际加速、节能或降低显存。

> **2026-08-26 前进路线裁决：** 在完整复核已完成矩阵、固定 `bffff43d…` 代码边界和最近相关工作后，当前主问题从“复用上一时间单位的 hidden/KV/完整表征”转向 `ZoomToken-BPNS-R1`：只使用当前观测，在 VideoMAE 主干前保留一个连续无孔洞的 `8×8/K64` 原生支持，并让全部 K64 token 完整执行 12 层 VideoMAE-S 与既有 Adapter。现有同源 K100 与 R1 的 final-EMA 分别为 `68.51/61.19/46.27` 和 `69.07/61.14/46.57`（Avg-mAP/mAP@0.6/mAP@0.7），说明减少 36% 空间输入没有造成准确率下降，但尚未证明真实效率。下一项唯一优先证据是不新增训练，先在同一硬件重放 K100 与 R1，测完整 decode-to-NMS 的 p50/p95、峰值显存、gross energy、短动作和边界误差；DSR6-KV、MOD32-KV、DROP32 只作为辅助 Pareto 点。若 R1 不能形成真实全栈收益且保护边界，则停止效率论文主张。本轮是实施前路线裁决，没有新增性能或成本结果。

> **2026-08-26 科学路线更新：** 对“在主干前一次确定稳定/变化 token，变化 token 完整通过 12 层，稳定 token 直接复用上一时间单位完整表征”的方案完成了独立 Pro 与代码/文献核验。结论是在当前官方 VideoMAE/AdaTAD 结构下停止该精确方案：16 帧 clip 只是注意力执行单元，每层后的 Adapter 仍沿全窗口时间轴传播；只缓存最终层不能提供逐层上下文，逐层缓存又与 Eventful Transformers、STC-Cacher 的选择性重计算合同高度重叠，并可能破坏现有批处理效率。本结论是实施前设计判断，不是新的性能实验，也不否定全部时序复用或独立 A-MoD。该次裁决当时要求优先使用已有 final checkpoint 完成 FULL64、DSR6-KV、MOD32-KV、DROP32 的同硬件端到端延迟、显存和能耗测量。

> **2026-08-24 运行状态更新：** `DSR6-KV` job `1252521` 已终态 `FAILED 1:0`，运行 2 秒，在 cell 创建、数据读取和模型执行前退出。精确原因是旧 launcher 在 Slurm 非登录 shell 中以 `set -u` source `/etc/profile`，站点脚本引用未定义的 `LC_BYOBU`。因此这不是方法性能或成本结果，仍无 DSR6 准确率证据。该 job 不恢复、不重排、不重提；最小 launcher 修正、独立复核和真实 job-shell 结果盲 PRE_RUN 现已完成，下一动作是请求一个新的独立 epoch。

> **实现与运行状态：** launcher-only 最终后继 `c6327a891809aa30370b3b2d9bedab0dcfe0d326` 已通过 N16R4 focused suite `12 passed`、fresh 独立 Critic `AUDIT_PASS` 和真实双卡 Slurm job-shell 结果盲 PRE_RUN。唯一正式 DSR6-KV job `1252527` 已于 `2026-08-24 10:53:53 CST` 在 `g0041` 终态 `COMPLETED 0:0`，完整运行 `06:07:17`；预注册 `epoch_59.pth` 和最近恢复点 `44/49/54` 均存在，未见 Traceback、显存溢出或非有限 loss。终态不可变日志中的 epoch-59 EMA 官方 validation 为 Avg-mAP/mAP@0.6/mAP@0.7 `67.38/59.34/46.01`，低于预注册的严格准确率保持门，但其 VideoMAE 重块算量代理仅为 FULL64 的 `79.06%`。因此它不支持“近乎无损”的结论，却仍是待真实端到端成本测量的精度—计算 Pareto 候选；不追加结构补救或重复训练。

> **并行方法状态：** 严格 VideoMAE A-MoD 参照已在 scientific revision `a41714e9f9271906a2eb4505e3fedc590c838055` 实现并通过独立 Critic `AUDIT_PASS`。EMA 前缀加载缺陷由 clean successor `2d945e64bdccd09ae2e2916524562e3f388c5a2a` 修复；N16R4 focused suite `20 passed`，fresh 独立 Critic再次为 `AUDIT_PASS`。显式 Bash 与 canonical 411-video 根绑定的 capacity=1 等价性 job `1254040` 已 `COMPLETED 0:0`：mAP@0.3/0.4/0.5/0.6/0.7 为 `83.46/79.45/71.96/61.59/47.20`，Avg-mAP `68.73`；相对官方记录最大绝对差 `0.04` 个百分点，判定实现对齐通过但非逐位一致。这是实现准入证据，不是 A-MoD-50 的方法性能或效率结果；正式 A-MoD-50 尚未训练。

> **2026-08-25 当前正式实验：** `R1-APM-C32/FULL64` 已完成。它只让 32 个可靠匹配位置使用“前一 tubelet 的停止梯度表征 + 当前帧残差”的输入载体；64 个 token 仍在全部 12 个 VideoMAE block 和 Adapter 中完整更新，从而把帧间载体与 K32 深度裁剪解耦。clean revision `bffff43dad28ca1042602ad3a01ba2990b953c13` 已通过 N16R4 focused suite（`22 passed`）、fresh 独立 Critic `AUDIT_PASS` 与结果盲 PRE_RUN。唯一 seed-42、60-epoch job `1254008` 终态 `COMPLETED 0:0`；epoch-59 final EMA 的 Avg-mAP/mAP@0.6/mAP@0.7 为 `68.22/60.43/45.60`，低于预注册门 `68.73/61.58/47.24` 达 `0.51/1.15/1.64` 个百分点。因此按冻结规则停止当前 APM 载体，不启动成本、额外 seed 或结构补救。该结果是当前载体的有效负证据，不是所有时序复用方法的普遍否定；A-MoD-50 仍未训练，selector-inclusive 效率结果仍不存在。

> **2026-08-25 路线裁决更新：** 在完成 Eventful Transformers 最近邻核验和保守算量上界后，fresh ZoomToken Project Pro 复核对 `R1-ACR16-Δ1-FKV` 给出 `STOP`。该方案本质上是 Eventful 式变化证据、条件深度跳过与一次低秩输入差分残差的组合，而不是新的时序复用原理；VideoMAE 主块理论节省上限约 `9.45%`，计入全 K/V、全 Adapter、patch embedding、匹配与动态执行后，已知骨干算术上界约 `8.80%`，不足以为完整 `decode→Soft-NMS` 链路同时获得至少 `5%` 的延迟与能耗改善留下可信余量。因此该方案在实现前停止，不进入 Builder、PRE_RUN 或训练。本裁决只停止 ACR16/Eventful 直接迁移，不否定全部时序去冗余研究，也没有产生新的准确率或效率结果。

> **2026-08-25 动态选择性重计算复核：** 新一轮附件式 ZoomToken Project Pro 讨论进一步比较了 clip 内动态重计算 `IC-DRU`、重叠窗口精确缓存 `OW-ECR` 和 current-proxy 条件深度路由 `PCD-DRU`。理想已知骨干算术上界分别约节省 `50.12%`、`7.66%` 和 `60.16%`，但前两种动态路线要求平均仅刷新约 25% 的 token、仍保留 dense/full-refresh 峰值，并且其变化路由、轻量残差与逐 token 深度分配均可被已有 Eventful、视频缓存和 MoD/A-MoD 工作分解覆盖；窗口路线则因 Adapter 依赖传播和约 `54 MiB/样本` 的十二层缓存缺少全链路余量。Pro 对这三个精确定义候选给出 `STOP_BEFORE_IMPLEMENTATION`，新增正式实验为 0。本裁决是实施前设计判断，不是动态刷新、特征复用或 20%–30% 更新率的经验失败，也没有产生新的准确率或端到端成本结果。

- 更新时间：2026-08-29（BPNS-R1 v004 已形成完整同硬件终态成本证据，fresh Pro 确认 `PIVOT` 并停止独立效率 headline。当前按唯一原子任务核验 TAR32-FKV 训练终态；协议有效才提交冻结的唯一 K100/TAR32 成本测量）。
- 证据等级：已有真实 THUMOS14 validation 与单硬件、单 seed 的八-pass 端到端成本。官方路径 A、全部 token + sparse adapter 的 B、主干前 ROI `K=64` + 同一 adapter 的 C，其 Avg-mAP/mAP@0.7 分别为 `68.73/47.24`、`68.51/46.27`、`68.22/45.35`。v004 证明固定条件下 R1 能耗与显存下降，但延迟收益不足；不能声称总体计算效率提升。

## 1. 一句话问题与应用价值

ZoomToken 研究离线时序动作检测（Temporal Action Detection，TAD）中如何减少 VideoMAE 重骨干对空间 patch 的重复计算，同时保护动作边界和高时间交并比（temporal IoU，tIoU）定位，从而在真实端到端成本下降时保持检测实用性。

## 2. 唯一主路线与直观方法

主路线是 `ZoomToken-BPNS-R1`：减少端到端 TAD 中 VideoMAE 的空间重复计算，同时用连续无孔洞的原生支持保护动作边界。已完成的一变量实验中，B 在每个 tubelet 上保留全部 `100` 个原生空间 token，C 在重 VideoMAE 之前按连续椭圆/高斯分数保留 `64` 个原生 token；二者使用同一 true-ragged 重主干、同一 sparse adapter 和同一官方训练配方。R1 进一步把支持约束为九个合法 `8×8` 原生矩形之一，所有 K64 token 仍执行完整 12 层 VideoMAE-S 与 Adapter，不使用历史 hidden/KV/cache。B→R1 因而直接检验在当前观测上减少 36% 主干输入是否能保持准确率并带来真实全栈收益。R2/R3/R4 与其对照保留为支持拓扑、时间对齐和框外信息的归因证据；ACR16/Eventful 迁移及完整表征直接复用均已在实现前停止。

上一轮 `RC32-KV` 在全部 12 个 VideoMAE block 中只对确定性 K32 子集执行 query/output/MLP，并尝试前一 tubelet 同位置 carry；完整结果证明 carry 在相同理论重块算量下显著劣于无 carry 的 MOD32-KV，而 MOD32-KV 仍明显低于 FULL64。历史上随后选择的最小修订为 `DSR6-KV`（Depth-Staged K64→K32 Refresh with Full K64 Context）：严格 8×8、K64 空间支持和现有 refresh 排序保持不变；blocks 0–5 对 K64 做完整更新，blocks 6–11 只对同一个 K32 子集做 query/output/MLP，全部 K64 保持为不截断梯度的 K/V 上下文并继续通过既有 Adapter。该路线不使用前帧隐藏态、不增加浅层传输模块、不使用逐层动态预算，也不搜索第二个深度切分点。它只检验“完整早期表示形成后，后半深度是否存在可削减的重复更新”。

历史机制对照还包括严格 A-MoD，用于单独检验逐层计算分配：12 个 VideoMAE block 中偶数块完整更新全部 800 个时空 token，奇数块根据紧邻前一完整块的注意力概率列均值稳定选择 400 个 token，仅对其执行注意力与多层感知机更新；未选 token 保持恒等旁路，既有 Adapter 仍作用于全部 token。该参照不含帧间缓存。独立的 `APM32-CTX64` 时序路线保存前一 tubelet 的无位置 patch 表征，在当前 tubelet 的局部邻域内做双向一致匹配；匹配可靠时保留 32 个表征并对其余 32 个重计算，匹配不足则完整重算 64 个 token。这样分别检验逐层分配与相邻 tubelet 的表征复用，避免把二者混称为同一机制。

## 3. 与官方/最强基线的真实差异

唯一可接受的稠密参照是共享的、未修改 OpenTAD/AdaTAD 官方路径。published anchor 为 Avg-mAP `69.03`、mAP@0.7 `48.27`。已发布 checkpoint 尚无可验证副本，因此项目在 AdaTAD release `01c58b9` 的原始 THUMOS14 配置上完成了一次唯一的 60-epoch fallback reproduction；官方 evaluator 的终态日志结果为 `68.73/47.24`，相对发表锚点分别低 `0.30/1.03` 个百分点，应表述为接近锚点的官方路径复现，而非逐数值复现。

既有 `66.42/67.14/65.99` Avg-mAP 与 `45.19/45.84/45.02` mAP@0.7 是修改 source family 的 matched-source dense 输出，不是官方 AdaTAD 复现，不能用来宣称最终方法质量或解释与 `69.03/48.27` 的差距。方法矩阵中的稠密 `DN`、uniform `U`、random `R` 和动态路由 `Q` 必须与同一数据、检测器、损失、优化、评测与完整成本统计相匹配；官方稠密 `DO` 在 Track B 中只以共享官方回执作必报参照，不再强行要求它输出 GeoRoute 路由遥测。

## 4. 已完成实现与当前 clean revision

- 已通过路由层静态 parity 的 `cd6463df…` 证明 F/N/Q/D 开关可到达 production selector；它不证明 detector 输出或性能。已关闭 `b157433d…` detector-fixture 负例，禁止重开该表面。
- 先前真实 P1 矩阵 `5491c580…` 已封存为 `NO_SURVIVOR_INVALID_P1`：DO 的非适配遥测要求、control-path 配置身份约束和 136-window/40-video population 身份漂移导致 0/5 accuracy cells 与 0/8 cost leaves；这是准入实现/协议缺陷，不是 Q 的效果或成本反证。
- ROI60 的最终 ROI-only `G` clean revision 为 `59960255a708c0341baa8104a1d4e120f87435e3`，位于从 AdaTAD release 派生的独立 official-base worktree；同源全计算 `DN` 运行于其 claim-preserving 前序 `d2b5de054bf0a9b5927218517a56b571a4d6ded2`。实现先恢复历史已审的 `GeoRouteSourceViews` 数据变换，随后修正优化器参数组别名和 `ActionFormer.forward_train` 中辅助损失的分布式训练归属；最终改动只使 G 的既定损失图可在真实训练路径中执行，不改变 DN 前向、数据划分、检测器、评测器、NMS、全局预算或 checkpoint 规则。最终 focused 静态套件 17 项通过，独立 Critic 对实际 `ActionFormer` 路径复核通过。
- 为严格回答 adapter 与 ROI 各自带来的准确率变化，clean revision `1a18565bbee5fdb08969b754881d0b06f3429870` 曾实现一组**后主干诊断**：A 复用未修改的官方 AdaTAD；B 保留完整官方 VideoMAE/AdaTAD 前向，只把空间平均聚合替换为对全部 token 的 sparse adapter；诊断 C 与 B 共享同一主干和 adapter，仅把聚合支持改为后主干 ROI `K=64`。该实现通过 21 项静态测试和独立 Critic 审查，但只用于准确率归因，不作为节省主干计算的证据，也不是当前正式 C。
- 代码身份回验确认，旧 seed-3407 的 G、上述 `1a18565b…` 后主干诊断 C 和当前 `70dcbe10…` 主干前 C 是三种不同干预。旧 G 在原生源帧上执行全局 `B=24576`、动态 `K_t` 选择；在常见 `180×320 → 11×20` 原生网格上只保留约 `29.1%` 的重主干 token，并允许 `K_t=0`。后主干诊断 C 先完整执行稠密 VideoMAE，再从 100 个稠密特征中选 64 个。当前正式 C 则在官方 `160×160 → 10×10` 网格上、VideoMAE patch embedding 和 blocks 之前固定选择 64/100 个原生 token。
- **方法边界更正**：最终 ZoomToken 必须在 VideoMAE 重主干之前完成 ROI 选择，使重主干仅处理被选中的原生 token；不接受“先完整运行稠密主干、再做 ROI 聚合”作为最终方法。`1a18565b…` B/C 仅保留为历史后主干诊断；当前正式实现与性能结论来自 `70dcbe10…` 的主干前 B/C。
- 主干前严格候选已在 clean revision `70dcbe1089866f6ee3821176eb41d2dc10ee8d14` 闭合。其父候选遗漏训练端 regularization 生命周期，已在 `c209d582…` 修复；目标环境随后发现 OpenTAD 的 `batch_size` 是作业级全局批量，最终候选据此固定 global batch `2`、world size `2`、local batch `1`。独立 Critic 对两次局部修正均通过，目标双卡无指标前向作业 `1248828` 验证 B/C 分别执行 `38,400/24,576` 个原生 token、单次重主干、零 padding，输出形状一致。
- 严格矩形 R1 已实现于 clean revision `9e25c6d38de8c993948025629181470b858682b4`。它在 `10×10` 原生 token 网格上从九个合法 `8×8` 完整矩形中稳定选择一个，在 patch embedding 前只 gather 该矩形的 64 个 token，并保持单次 true-ragged VideoMAE、零 padding 和同一 sparse adapter。独立 Critic 最终 PASS；N16R4 目标环境 9 项无数据 Torch 检查通过。该检查只建立实现与执行正确性；真实性能由下文终态结果给出。
- BPNS-R1 同硬件成本回放实现为 clean/pushed revision `b7357817d81127ab2d713b5471d008ea893efd35`（分支 `codex/zoomtoken-bpns-r1-cost-v001`）。它不修改模型，只增加 profiler、N16R4 launcher 与 focused test；严格加载 K100/R1 的 epoch-59 EMA，在同一 Slurm GPU 上按 ABBA+BAAB 各运行四次完整 validation。目标环境检查把旧开发 population 更正为 211 个 validation 视频、792 个有序 loader 项，并修复功耗 sidecar 的 CPU affinity 继承。focused pytest `6 passed`，fresh Critic `AUDIT_PASS`，结果盲 Slurm 见证 job `1257250` 完成，fresh Evaluator `PRE_RUN_READY`。正式 job `1257281` 随后在首个 K100 完整 pass 的数值一致性门终止：回放 `mAP@0.7=46.246663`，预填历史值为 `46.27`；没有完整 profile 或终结收据。这证明现有准入检查仍未正确绑定可重放的原始精度值，但不改变模型训练结果，也不产生成本结论。
- R2/R3/R4 多分支候选已实现于 clean revision `b1d9fa7b10209b23c4405b4be3965ee66f3c05f5`。独立 Critic 确认 production selector、主干前执行、SHUF 对照的物理位置置换、R3 连续动态矩形和 R4 7×7+15 支持均符合冻结目的；目标环境 8 项无数据测试通过。该段只记录实现准入；R2/R3/R4 的终态证据与裁决以下文结果为准。
- 已生成同一个 THUMOS14 validation 滑窗上的 token-selection 定性主图（可视化代码 `0b12c68e…`、job `1250245`）。每行使用该方法自己的 checkpoint，原色网格为主干前实际选中的 native token，灰色网格为未选中 token；R4-SHUF15/Q64-GLOBAL 明确使用当前恢复点并仅作定性观察。该图说明严格矩形、框内选择、框外补充、乱序与全局 Top-K 的空间形态确实不同，但不同 checkpoint 的行间差异不是公共 checkpoint 下的反事实比较，也不提供性能或成本结论。
- RC32-KV 原方法实现 revision `836f2ce4…` 的 K64 上下文、确定性 K32 刷新、窗口内同位置 carry 与无泄漏边界已经目标环境测试和独立审查。首个真实部署暴露训练入口仍只允许旧 route surface；修复后的 clean candidate 为 `813012620dca991ff90121d0d9faf688f303d1ef`，仍在 GitHub 分支 `codex/zoomtoken-rc32-kv-v001`。它只修改训练入口白名单和实际配置回归测试；目标 N16R4 环境 `10 passed`，独立审查 PASS，未知 route 仍被拒绝，完整恢复语义未改变。该状态是实现和测试证据，不是性能证据。
- 终态诊断入口位于 `81301262…` 的 clean descendant `4e940b780da5a3cd0ea28ca420c5d1cb879818b5`（分支 `codex/zoomtoken-rc32-boundary-eval-v001`）。它新增离线边界诊断、为 `tools/test.py` 增加标准配置覆盖参数，并修复 EMA 权重应加载到 DDP 内部模型而不是包装器的既有错误；终态脚本只接受 `epoch_59.pth`，在 canonical validation 上保存 `result_detection.json`；它已覆盖 FULL64、DROP32、MOD32-KV 和 RC32-KV 四臂。8 项纯结果/静态测试与 shell 语法检查通过。该 descendant 不改变正在运行的 `81301262…` 模型、训练或 checkpoint。
- `DSR6-KV` 的冻结科学合同记录于 `PRO_TEMPORAL_DEPTH_REDUNDANCY_ROUTE_ADJUDICATION-v001.md`。科学实现根为 clean/pushed `3260cd39154069138c6b1757326372cc3b73754e`（父提交 `4e940b…`，分支 `codex/zoomtoken-dsr6-kv-v001`）：复用同一严格 K64 mask，blocks 0–5 调用既有 FULL64，blocks 6–11 调用既有 MOD32-KV；全部 K64 为不截断梯度的 K/V 上下文，既有 Adapter 仍处理全部 K64，未新增参数、loss、cache 或 transport。当前可执行后继为 `c6327a891809aa30370b3b2d9bedab0dcfe0d326`，相对科学实现仅修复 N16R4 profile 初始化边界及其回归测试。静态合同、Shell/编译检查、小型无数据 PyTorch 前后向、fresh 独立 Critic 与真实 job-shell 结果盲 PRE_RUN 均通过。本机完整 pytest 因 OpenMMLab/Torch DLL 环境不兼容未能收集，不能把该环境限制写成测试通过；唯一正式训练终态 EMA 为 `67.38/59.34/46.01`。它未通过预注册的近无损门，但以 `79.06%` 重块算量代理保留为高精度 Pareto 候选，下一证据应是与 FULL64 同硬件的端到端成本，而不是追加结构。
- 严格 A-MoD 参照的 scientific revision 为 `a41714e9f9271906a2eb4505e3fedc590c838055`；test-only clean/pushed successor 为 `31e4b1e61a23c4f1b319249684c8f05da6734235`（分支 `codex/zoomtoken-amod-v001`）。模型侧精确修改 backbone、配置、launcher、训练入口和 focused test 五个既有路径；后继只增强测试。N16R4 标准 OpenTAD 环境的无数据测试为 `10 passed`，独立 Critic 结论为 `AUDIT_PASS`。首个 wrapper job `1254014` 在 Python 前因 `/bin/sh` 不支持 `pipefail` 失败；第二个 job `1254016` 在推理前暴露官方 EMA 参数名统一带 `module.` 前缀的测试入口兼容问题。两者均封存且没有性能证据。加载修正 clean/pushed revision `2d945e64bdccd09ae2e2916524562e3f388c5a2a` 仅在前缀统一时选择 DDP 外壳或内部模型作为严格加载目标；N16R4 focused suite `20 passed`，fresh 独立 Critic `AUDIT_PASS`。结果盲 job `1254038` 又因 `/bin/sh` wrapper 与不存在的旧视频路径在 Python 前终止并封存；显式 Bash 与 canonical 411-video 根绑定的 `capacity=1.0` dense-parity job `1254040` 已完成 validation 并通过数值对齐。目前仍没有 A-MoD-50 训练性能或端到端成本结果。
- `R1-TAR32-FKV` 的最小 clean/pushed revision 为 `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`（base `2d945e64…`，分支 `codex/zoomtoken-r1-tar32-fkv-v001`）。实现保持 R1 K64 原生支持、全 K64 K/V 与全 K64 Adapter，按 `[K64,K32]x6` 交替更新且没有 cache、新参数、新 loss、动态基数或 fallback。N16R4 focused suites 为 `32 passed, 1 skipped` 与 strict-R1 `9 passed`；fresh Critic `PASS`，fresh result-blind Evaluator `PRE_RUN_READY`，真实 batch CUDA AMP pre-run job `1260163` 完成。唯一正式 seed-42 training job `1260166` 已在 `g0059` 启动；终态前不存在准确率、成本或可发表性结果。
- `APM32-CTX64` 与 matched control `CUR32-CTX64` 的 clean/pushed executable candidate 为 `e92df6a4737a10955722c6aedc2f079e0d285a18`（父提交 `d985dfb8…`，分支 `codex/zoomtoken-apm32-ctx64-v001`，基于 `31e4b1e6…`）。八个既有表面实现前一 tubelet detached patch memory、局部双向一致对齐、K32 refresh/K64 context、K64 fallback、同掩码当前表征对照，以及仅限结果盲准入的生产单批/完整状态恢复入口；没有新增模型参数或损失。批内不同 fallback 总数已按真实 Query 数分桶。N16R4 CPU-only focused suite `19 passed`，fresh Critic `AUDIT_PASS`。该旧方案同时改变帧间载体与深度更新数量，已被后续 `R1-APM-C32/FULL64` 单变量方案取代，不直接训练。
- 完整矩阵审查指出上述 `APM32-CTX64` 同时改变了帧间载体和深层更新数量，因果解释不够单一。修订后的 `R1-APM-C32/FULL64` clean/pushed revision 为 `bffff43dad28ca1042602ad3a01ba2990b953c13`（父提交 `e92df6a4…`，分支 `codex/zoomtoken-r1-apm-c32-full64-v001`）。现有局部双向一致匹配、阈值 `0.80`、半径 `2`、one-tubelet detached memory、clip reset 与 K64 fallback 保持不变；载体应用后，全部 K64 在 12 个 block 及 Adapter 中完整更新。N16R4 focused suite `22 passed`，fresh 独立 Critic `AUDIT_PASS`，结果盲 Evaluator 为 `PRE_RUN_READY`。本机 pytest 因 Windows Torch DLL `WinError 1114` 未能收集，未被写成通过；正式准入依赖通过的目标环境结果。

## 5. 预先规定的比较与证据边界

**数据与评测。** 使用共享、只读的 THUMOS14 规范视频根 `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`（411 个有效 MP4 软链接：training 200、validation 211、0 断链），配套官方标注 `thumos_14_anno.json`、类别表 `category_idx.txt`、VideoMAE-S 预训练权重和 AdaTAD 官方评测器。官方 test 维持关闭；先在冻结的 development/Fit–Gate 划分进行准入和开发比较。

**矩阵。** Track A 的 released-checkpoint artifact 未找到，因此唯一的 clean AdaTAD `01c58b9…` 60-epoch reproduction 已完成训练和官方 validation，保持官方两卡、配置 `batch_size=2` 的配方；在 OpenTAD 的按 world-size 切分语义下，这是 local batch 1、global batch 2。Track B seed `3407` 的 60-epoch 配对训练也已完成终态 validation：`DN` 为同源全计算对照，`G` 为 ROI modifier-only、residual-off 的 exact-`B=24576` 动态路由。两臂使用同一 raw-video 数据、训练时长、优化/EMA 和单卡资源。共享官方结果只作外部锚点，不能把 DN 冒充官方复现。`U/R/Q` 的既有结果用于解释选择机制，但不混入本次两臂训练。

**后主干诊断矩阵。** A 直接使用已完成的官方作业 `1245842`，不重复训练；B job `1247290` 为全部 token 加 sparse adapter；C job `1247291` 为完成稠密 VideoMAE 后的 ROI `K=64` 聚合。B/C 使用 revision `1a18565b…`、seed `42`、双卡 local batch `1`/global batch `2` 和相同官方配方。该矩阵只诊断 adapter 与稠密特征 ROI 聚合的准确率影响，不是计算节省实验。

**主干前严格因果矩阵。** A 仍只读复用官方 job `1245842`；B job `1248835` 在每个 tubelet 上让全部 `100` 个原生 token 进入 true-ragged VideoMAE 与 sparse adapter；C job `1248834` 只在重主干之前将支持改为 ROI fixed `K=64`，其余模型和训练配方与 B 相同。两臂绑定 clean revision `70dcbe10…`、seed `42`、双卡 global/local batch `2/1`、官方增强、AdamW/调度器、AMP、EMA、evaluator、Soft-NMS、60 轮以及每 2 轮 checkpoint/validation。两项均为 `COMPLETED 0:0`，各保存 30 个周期检查点并完成 epoch-59 官方 validation。

**严格矩形 R1 falsifier。** R1 job `1249099` 绑定 clean revision `9e25c6d…`、seed `42`、相同双卡 global/local batch `2/1`、60 轮、官方 training/validation、AdamW/调度器、AMP、EMA、evaluator 与 Soft-NMS。唯一科学差异是把 C 的椭圆/高斯 Top-64 支持换成一个完整 `8×8` 原生矩形；token 数仍为 64。每 5 轮保存完整恢复状态、保留最近 3 份，final checkpoint 同时保存 raw 与 EMA，EMA 为预注册主结果。训练已完成，终态 EMA 为 `69.07/61.14/46.57`（Avg-mAP/mAP@0.6/mAP@0.7）；v004 已完成 R1/K100 配对成本，但 p50 只下降 1.51%，未通过效率主门，且仍无多 seed 结果。

**BPNS-R1 同硬件成本回放。** 不新增训练，严格复用 K100 job `1248835` 与 R1 job `1249099` 的 epoch-59 EMA。正式 population 是完整 official validation loader 的 211 个视频、792 个有序样本项；K100/R1 原计划在同一张 Slurm GPU 内按 `K100,R1,R1,K100,R1,K100,K100,R1` 各执行四次，每次先 warmup 50 个窗口。正式 job `1257281` 在首个 K100 pass 后因 `mAP@0.7=46.246663` 与预填历史值 `46.27` 不一致而 fail closed，未完成 R1 和其余 counterbalanced passes，且未发布完整 profile/终结收据。该 namespace 不提供效率证据；在明确历史数值的原始精度、舍入与容差合同并重新通过独立结果盲检查前，不解释任何局部计时或能耗输出。

**严格矩形 R2/R3/R4 矩阵。** clean revision `b1d9fa7b…` 上的八个 seed-42、60-epoch 单元为：R2、R2-SHUF48、Q48-GLOBAL、R3、R3-AREA-SHIFT、R4、R4-SHUF15、Q64-GLOBAL，对应 jobs `1249125–1249132`。R2 固定 8×8 候选矩形并在框内选择 48；R3 执行连续严格矩形的全部成员并允许自然动态 `K_t`；R4 固定保留 7×7 core49，再从框外选 15 个基础效用最高的 token。SHUF 与全局 Top-K 对照分别隔离内容排序和矩形 eligibility。所有单元沿用相同官方数据、seed、双卡批量、优化器、调度器、增强、AMP、EMA、evaluator/NMS 和恢复策略；不使用 synthetic/subset 作为性能证据。

**RC32-KV 跨帧刷新矩阵。** FULL64 只读复用严格矩形 R1 job `1249099` 及其预注册 final-EMA；不重复训练。revision `836f2ce4…` 上的首次部署 jobs `1250604/1250605/1250606` 在 recovery 入口终止，保持封存且不提供科学结论。修复版 clean revision `81301262…` 的 DROP32、MOD32-KV、RC32-KV jobs `1252179/1252180/1252181` 均为 `COMPLETED 0:0`，共同根为 `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_81301262_seed42_20260823T2100`。三项使用 seed 42、双卡、60 轮和同一官方配方，每 5 轮保存完整恢复点；每个 cell 均存在 `checkpoint/epoch_59.pth`，日志未见 Traceback、OOM 或非有限数值。终态 EMA 的 Avg-mAP/mAP@0.6/mAP@0.7 分别为 DROP32 `66.11/57.83/44.88`、MOD32-KV `66.50/59.24/45.21`、RC32-KV `64.73/57.34/42.91`；FULL64 为 `69.07/61.14/46.57`。按 Q/K/V/output、attention 与 MLP 矩阵乘统计的 VideoMAE 重块理论计算量代理，DROP32、MOD32-KV、RC32-KV 分别约为 FULL64 的 `49.32%/58.11%/58.11%`；该代理不包含解码、H2D、scout、patch embedding、adapter、检测头、后处理和 NMS。三项准确率均未通过存活门，因此不启动 selector-inclusive latency/energy；没有把理论 FLOPs 或训练耗时冒充真实延迟。

**DSR6-KV 唯一新实验。** 实现和独立审查通过后，只新增一个 seed-42、双卡、60-epoch THUMOS14 training→validation cell；FULL64、DROP32、MOD32-KV 全部只读复用。配方、增强、AdamW、调度器、AMP、EMA、detector、loss、evaluator、Soft-NMS 与 FULL64 相同，final epoch-59 EMA 是唯一主结果，每 5 epoch 保存完整恢复状态。其准确率必须同时达到 Avg-mAP `≥68.57`、mAP@0.6 `≥60.64`、mAP@0.7 `≥46.07`；任一失败即停止该时序/深度路线，不追加切分点、种子、MoD 或浅层传输补救，也不测成本。只有准确率通过后，才在完整 canonical validation population 上测量 selector-inclusive decode→NMS 的 p50/p95、gross energy 和峰值显存；机械 block-FLOPs 代理约 `79.06%`，不作为真实速度结论。

该唯一单元现已完成；final EMA 为 `67.38/59.34/46.01`，分别低于阈值 `1.19/1.30/0.06` 个百分点。原预注册结论 `STOP_DEPTH_ROUTE` 对“近乎无损准确率保持”仍成立；经用户明确要求按计算—准确率联合目标重审后，不再把它解释成效率候选的全面终止。DSR6-KV 不追加训练或结构补救，但可与 FULL64、MOD32-KV、DROP32 进行同硬件端到端成本复测。

**A-MoD 与帧间记忆边界。** 严格 A-MoD 参照已具备可执行配置，但尚未启动正式 A-MoD-50 训练。其代码级审计确认 Dense/A-MoD 交替、前一 Dense attention 路由、top-400、MHSA+MLP identity bypass、dense Adapter 和无新增参数均符合冻结设计；`capacity=1.0` 的官方 Dense 等价性 job `1254040` 已 `COMPLETED 0:0`，完整 mAP 向量相对官方记录最大差 `0.04` 个百分点且 Avg-mAP 同为 `68.73`，故实现对齐通过但非逐位一致。旧 parity jobs `1254014/1254016/1254038` 均封存且没有模型证据。帧间路线的旧 `APM32-CTX64` 不再直接训练；解耦后的 `R1-APM-C32/FULL64` job `1254008` 已完成并因三项准确率门失败而停止。

**APM-C32/FULL64 单变量实验。** 该实验只把 32 个可靠匹配 token 的主干输入改为前一 tubelet detached 表征与当前残差之和，未匹配及回退位置使用当前表征；随后所有 64 个 token 完整通过 12 个 VideoMAE block、Adapter、检测头与官方评测链。唯一正式单元使用 seed `42`、双卡 global/local batch `2/1`、官方 THUMOS14 training→validation、60 epoch、每 5 epoch 完整恢复、保留最近 3 份及 final、epoch-59 EMA 主结果。job `1254008` 已 `COMPLETED 0:0`，final EMA 为 `68.22/60.43/45.60`，相对存活门 `68.73/61.58/47.24` 三项均失败。因此当前载体按预注册规则停止，不启动成本、额外 seed 或结构补救。

代码层面的准备已经完成：现有稀疏 Adapter 路径保留 `(tubelet index, spatial index)` 原生坐标，数据样本保留视频名和窗口起点，ChronoTransport 分支已有可复用的状态容器、状态年龄、首块强制重计算和非有限值回退。它们尚不包含可靠的跨帧对应、遮挡/场景切换失效或跨窗口身份管理，因此当前只作为实现基础，不作为已完成方法或性能证据。

**指标和停止规则。** 主要准确率为 Avg-mAP、mAP@0.6、mAP@0.7；同时按已冻结定义报告短动作、起止边界误差和高 IoU 错误分解。离线边界评估器只消费最终 prediction/annotation JSON，不进入训练或选择 checkpoint。历史 P1/Q 成本合同使用 136 windows、40 video clusters 和配对 bootstrap；它不适用于本次 BPNS 回放。v004 已在完整 official validation loader 的 211 个视频、792 个有序样本项上按 ABBA+BAAB 各执行四次，覆盖 decode→H2D→model→postprocess→NMS 的 p50/p95、吞吐率、峰值内存和连续 gross energy。完整性通过，但 p50 联合门失败，因此 R1 的总体效率主张停止。

**checkpoint。** 未来非 untouched-official 的 DN/U/R/Q/R1 完整训练每 5 个 epoch 保存原子可恢复 `.pth`，保留最近 3 个有效恢复点以及预定义 milestone/final；恢复必须恢复 model、EMA、optimizer、scheduler、AMP scaler、epoch/update 与成功更新计数、sampler epoch、Python/NumPy/Torch CPU/CUDA 随机状态。只使用预注册的 final/final-EMA 选择规则，不能据中间验证挑选最好 checkpoint。未修改的共享官方 AdaTAD 配方保持其原生每 2 epoch 保存节奏。

## 6. 已取得的真实性能/成本结果

已有真实 THUMOS14 validation 的 matched-source 方法级负证据：seed 3408 的同源 dense/uniform/random/Q 分别为 Avg-mAP/mAP@0.7 `67.14/45.84`、`60.05/40.17`、`61.53/41.80`、`57.84/36.93`；seed 3409 的同源 Q 为 `53.81/33.40`。因此 Q 在两个独立 seed 中都低于同源 dense、uniform 和 random 对照；这是当前 Q-core 在该修改 source family 下的重复负向观察，不能被忽略或表述为“完全没有真实验证”。

本轮三项 60-epoch 终态 validation 如下（百分数）：

| 路径 | Avg-mAP | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|
| clean official AdaTAD reproduction | 68.73 | 61.58 | 47.24 |
| matched-source full-compute DN | 64.73 | 56.14 | 43.26 |
| ROI-only G | 61.49 | 53.42 | 39.99 |

ROI-only G 在全部五个 tIoU 阈值上均低于 DN；相对 DN 的 Avg-mAP、mAP@0.6、mAP@0.7 分别为 `-3.24/-2.72/-3.27` 个百分点。这是当前 ROI-only exact-B 动态路由实现的明确负结果，不能主张 ROI 提升准确率。它不否定 ROI 作为一般机制，但否定了当前 G 配置作为性能改进方案。三项运行没有完整配对成本结果，因而也不能判断这种精度损失是否换来了足够的端到端效率收益。

ROI 的历史证据也必须单列而不能遗漏：ROI-only 的 20-epoch 开发性诊断为 Avg-mAP `13.18`、mAP@0.7 `8.95`，另一次 ROI 20-epoch 训练在正式评测前因视频解码失败；continuous-ROI 的旧 60-epoch 矩阵仅完成训练，没有检测预测、mAP 或完整成本。当前新增的 ROI-only G 已形成与 DN 同源可比的 60-epoch validation，但仍不是“在未修改官方 AdaTAD 上只加 ROI 插件”的直接一变量比较，也尚无成本闭环。

严格三臂归因矩阵的首个同阶段过程性 validation 已完成。第 42 轮训练开始前，未修改官方 A、全部 token + sparse adapter 的 B、ROI `K=64` + 同一 adapter 的 C，其 Avg-mAP/mAP@0.7 分别为 `67.88/46.19`、`67.06/45.72`、`67.86/46.14`。因此在这一中间节点，A→B 为 `-0.82/-0.47` 个百分点，B→C 为 `+0.80/+0.42` 个百分点，C 与 A 的 Avg-mAP 几乎相同（`-0.02`）。这是单个中间节点，不用于挑选 checkpoint，也不能替代 60 轮 final/final-EMA 结论；其价值是初步显示 adapter 的负影响与 ROI 的相对恢复效应可以被分离观察。

B 的下一次过程性 validation 为 Avg-mAP `67.45`、mAP@0.7 `46.43`，仍属训练中间结果。官方 A 从同阶段 `67.88` 到 epoch-59 终态 `68.73` 仅增加 `0.85` Avg-mAP，因此训练阶段不足以解释旧 G 与历史后主干诊断 C 约六个百分点的差异。该历史诊断 C 执行全部 VideoMAE 重计算，不能据其准确率推断计算节省；当前正式 `70dcbe10…` C 已改为主干前选择，并以独立终态结果报告。

主干前严格三臂的 60 轮终态如下：

| 严格归因臂 | Avg-mAP | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|
| A：未修改官方 AdaTAD | 68.73 | 61.58 | 47.24 |
| B：全部 100 token + sparse adapter | 68.51 | 61.19 | 46.27 |
| C：主干前 ROI K=64 + 同一 adapter | 68.22 | 61.01 | 45.35 |

A→B 为 `-0.22/-0.39/-0.97` 个百分点，说明 native-ragged/sparse-adapter 接入对 Avg-mAP 的代价很小，但在高 tIoU 定位上仍有约 1 点损失。B→C 为 `-0.29/-0.18/-0.92`，说明固定 ROI 删除 36% 空间 token 后，平均性能仅再下降 0.29 点，但高 tIoU 又下降约 0.92 点。C 相对 A 为 `-0.51/-0.57/-1.89`。这组结果否定了“当前主干前 ROI 必然导致旧实验中约 4–7 点 Avg-mAP 大幅下降”的解释；旧 `64.73/61.49` 主要混入了不同 source、训练图与动态预算路径。当前结果仍只有 seed 42，且缺少端到端成本，因此还不是论文级效率结论。

严格矩形 R1 已完成 seed 42 的 60 轮训练，终态 EMA 的 Avg-mAP/mAP@0.6/mAP@0.7 为 `69.07/61.14/46.57`；相对不规则 Top-64 的 C 为 `+0.85/+0.13/+1.22` 个百分点，三项预注册准确率条件全部通过。后续八个单元均已 `COMPLETED 0:0`；在 `Training Over` 前按冻结 EMA 评测路径打印的终态官方 validation 为：R2 `66.56/59.06/45.17`，R2-SHUF48 `66.17/58.53/44.47`，Q48-GLOBAL `65.78/58.62/44.74`，R3 `67.88/60.32/46.41`，R3-AREA-SHIFT `67.50/60.26/45.09`，R4 `68.02/60.32/46.26`，R4-SHUF15 `67.19/60.17/46.20`，Q64-GLOBAL `67.84/60.66/45.39`。R2 相对乱序框内对照为 `+0.39/+0.53/+0.70`，相对全局 Top-48 为 `+0.78/+0.44/+0.43`，在该单一种子上同时支持框内内容排序和矩形 eligibility；R3 相对时间错位面积轨迹为 `+0.38/+0.06/+1.32`，说明动态矩形的时间对齐主要保护高交并比定位。R4−R4-SHUF15 为 `+0.83/+0.15/+0.06`，未达到预注册的 mAP@0.7 `+0.30` 框外内容排序门；R4−Q64-GLOBAL 为 `+0.18/-0.34/+0.87` 的交叉结果。因此不能宣称框外 learned ranking 有效，也不为该机制启动成本或多 seed。上述数值是真实 THUMOS14 validation 准确率证据，但仍没有同硬件端到端成本或多 seed 结论。

固定 K32 深度稀疏系列的 seed-42 终态 EMA 与理论重块算量如下：

| 深度刷新臂 | 重块 FLOPs 代理/FULL64 | Avg-mAP | mAP@0.6 | mAP@0.7 |
|---|---:|---:|---:|---:|
| FULL64 | 100.00% | 69.07 | 61.14 | 46.57 |
| DROP32 | 49.32% | 66.11 | 57.83 | 44.88 |
| MOD32-KV | 58.11% | 66.50 | 59.24 | 45.21 |
| RC32-KV | 58.11% | 64.73 | 57.34 | 42.91 |
| DSR6-KV | 79.06% | 67.38 | 59.34 | 46.01 |

RC32-KV 相对 FULL64 为 `-4.34/-3.80/-3.66`，相对 DROP32 为
`-1.38/-0.49/-1.97`，相对 MOD32-KV 为 `-1.77/-1.90/-2.30`。因此 temporal carry
没有增量价值，并且其损失远超 Pro 允许的非劣门。MOD32-KV 虽比 DROP32 高
`+0.39/+1.41/+0.33`，相对 FULL64 低 `-2.57/-1.90/-1.36`，同时重块算量代理减少
`41.89%`；DROP32 的代理减少 `50.68%`，Avg-mAP/mAP@0.7 下降 `2.96/1.69`；DSR6-KV 的
代理减少 `20.94%`，Avg-mAP/mAP@0.7 下降 `1.69/0.56`。因此三者分别构成激进、中等和保守
预算下的候选 Pareto 点。RC32-KV 在与 MOD32-KV 相同代理成本下三项准确率均更低，仍被严格支配；
APM-C32/FULL64 不减少重块计算且低于其 R1 基线，也不属于效率候选。当前代理尚不是实际加速，
必须通过相同硬件上的完整端到端延迟、峰值显存和能耗测量确认。

## 7. 已否定路线与负证据

- 把原始 SCNR exact-B 的 ROI 与 residual 捆绑为 headline 被否定：会混淆机制，且出现 M2 role collapse（`0/0/3,342,336`）。
- 依赖窗口级可变预算 `B_t` 的更激进联合时空预算被否定：它破坏 exact-B 的公平归因及可比的运行时/能耗核算。
- `b157433d…` 单层/四通道 FPN detector fixture 是结构性不匹配，已关闭。
- 多个 P1/F0 和 `5491c580…` 矩阵的失败均归因于 source/config、container、runtime、work-dir、终结器或 population-identity 等确定性准入实现缺陷；没有视频/模型完整证据时，它们不能被升级为 Q-core 的科学失败。
- RC32-KV 的时序 carry 被 seed-42 完整矩阵否定：在相同 K32 refresh 与 K64 K/V 条件下，其 Avg-mAP/mAP@0.7 比 MOD32-KV 低 `1.77/2.30` 点，且两者代理成本相同。停止的是 RC32 carry，不再把 MOD32-KV 的精度下降单独当成效率路线的否决。
- DSR6-KV 在前六层保持 FULL64、后六层稀疏刷新，得到 `67.38/59.34/46.01`；相对 FULL64 为 `-1.69/-1.80/-0.56`，未通过近无损门，但理论重块算量减少 `20.94%`。因此当前只否定“近乎无损”的表述，不否定其作为保守 Pareto 候选；在端到端测量前不能宣称有效加速。
- `R1-APM-C32/FULL64` 在不减少任何主干计算的条件下只替换 32 个可靠匹配位置的输入载体，终态为 `68.22/60.43/45.60`，相对官方路径复现低 `0.51/1.15/1.64` 个百分点，三项存活门均失败。因此停止当前 detached one-tubelet APM 载体；这不等于否定所有时序复用机制。
- `R1-ACR16-Δ1-FKV` 在实现前被否定：最近邻 Eventful Transformers 已覆盖 reference/buffer、变化 token 选择和稀疏/增量更新；该候选剩余差异主要是条件深度跳过与一次低秩残差的应用组合。其主块理论节省上限约 `9.45%`，计入已知必要开销后约 `8.80%`，无法为完整链路的稳定收益提供足够余量。因此不实现、不训练，也不把这一停止外推为全部时序冗余方向无效。

## 8. 当前未解决问题与下一科学决定

截至当前，共享官方 AdaTAD job `1245842`（运行 `05:47:13`）、同源全计算 DN job `1245907`（运行 `04:24:56`）与 ROI-only G job `1245924`（运行 `05:46:15`）均为 `COMPLETED 0:0`，并完成 epoch 59 后的 validation；训练日志未见 Traceback、显存溢出或非有限数值硬故障。DN/G 结果根各保留最近三个 5-epoch 恢复点（epoch 44/49/54）。官方路径按原配方从后 20 轮起每 2 轮评测一次；DN/G 训练期间只在第 60 轮后评测一次，随后已用 recovery checkpoint 补齐 epoch 44/49/54 的中途曲线。旧 BPNS 回放 jobs `1257281/1258299` 均未形成八 pass、`profile.json` 或 `terminal_receipt.json`。fresh Pro 已确认两位小数历史点值不足以承担 raw parity 硬门，并冻结 v003：硬门只覆盖执行身份与测量完整性，历史准确率仅作区间三态诊断。当前缺的是可接受的端到端配对成本证据，而不是训练准确率。

ROI60 首轮已经完成，为同一环境下的 ROI-only `G` 对同源全计算 `DN`（seed 3407、60 epoch）；`DO` 只消费共享官方回执。`G` 不是硬裁剪，ROI 外 token 仍可由基础效用选中。两臂均从同一预训练起点重新训练，不复用身份不明的旧 checkpoint；每 5 epoch 保存可恢复状态，最终统一使用预注册的 final/final-EMA 规则。

首轮 DN/G 的 epoch 44/49/54 六项补充 validation 已全部 `COMPLETED 0:0` 并加载 EMA；G−DN Avg-mAP 分别为 `-3.08/-3.05/-3.04` 个百分点，与终态 `-3.24` 一致，排除了“差距仅由最后几轮退化造成”的解释。严格三臂现已进一步把 adapter 与固定 ROI 的影响拆开。端到端成本只有在同一硬件和完整测量链成立后才比较，不能以 token 数代理。

主干前正式 B/C jobs `1248835/1248834`、R1 job `1249099`、RC32 修复版 jobs `1252179/1252180/1252181`、DSR6-KV job `1252527` 与 APM-C32/FULL64 job `1254008` 均已终态 `COMPLETED 0:0`。RC32 原 revision `836f2ce4…` 的首次部署 jobs `1250604/1250605/1250606` 仍封存为训练前失败；没有 resume 或重复 FULL64。既有 Q64-GLOBAL job `1249132` 终态 EMA 为 `67.84/60.66/45.39`。APM-C32/FULL64 因不省重块计算且降低准确率而停止，RC32-KV 因被同成本 MOD32-KV 严格支配而停止，ACR16/Eventful 直接迁移因新颖性与全栈收益余量同时不足而在实现前停止。v004 成本 job `1260095` 已终态并经 Pro 冻结；TAR32-FKV job `1260166` 的终态只按新任务读取审计。jobs `1257281/1258299` 只保留为 `FAILED 1:0` 的数值一致性准入证据；后者结果根为空。TAR32 审计完成前不能声称其准确率或效率收益，也不自动创建 successor。
