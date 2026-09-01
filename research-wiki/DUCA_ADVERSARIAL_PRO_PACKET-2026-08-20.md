# DUCA 对抗性 Pro 审查材料包（2026-08-20）

## 审查任务与严格边界

请把 DUCA 当作离线时序动作检测（TAD）的研究项目，以最严厉的审稿人身份先攻击下述解释、代码和实验合同，再给出一条你认为更有创新性、可证伪且可发表的唯一替代路线。不得编造实验结果，不得把历史或开发性结果升级为论文结论。

论文问题保持不变：能否用低成本 scout 的动作性、起点和终点边界语义，经确定性间接采样与逐视频/窗口动态预算 K，减少重型 VideoMAE 计算，同时保持高 IoU 定位。固定 K 只能作为对照、归因与回退；直接学习索引只可作为消融。

用户的新增科学输入：UVT 和 Fovea/Query-Bridge 的失败包不能简单删除。应保留并重新设计其中 Query 前后协同与知识传递的机制，但必须把它们和直接索引、在线 detector feedback、GT 进入部署选择、额外 detector 训练预算严格分开。

## GitHub 代码核验入口（本轮唯一代码来源）

本轮不上传任何模型、配置或测试源码，以避免大附件阻塞提交。请优先通过以下 GitHub 仓库、分支与提交审阅代码；本材料包仅提供已完成实验的数值、边界和审查问题。若某个提交网页不可访问，请如实写为“无法核验”，不要据此编造代码结论。

| 代码对象 | GitHub 地址 | 审阅用途 |
| --- | --- | --- |
| DUCA 仓库 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702 | 唯一仓库入口。 |
| Fovea 当前分支头 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-fovea-query-bridge-20260819 | `46c714249ff444fcc6428dbe95c52aefe55c488f` 仅追加研究记录；训练代码未变。 |
| Fovea 实际训练代码 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ae5067100c4490c7110c00a1ad406230ba603cd | Job `1244851` 所用代码；核验 `configs/adatad/thumos/duca_fovea_qb_thumos.py`、`opentad/models/selectors/fovea_query_bridge_selector.py` 和 `opentad/models/losses/fovea_losses.py`。 |
| UVT 当前分支头 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-uvt-utility-value-20260819 | `59f27d59c322a0e85932eb56448aedc3fb454950` 仅追加 Wiki 记录。 |
| UVT 实际训练代码 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/df544c78ce515d925dc7019f106fce09a53c09f8 | Job `1244840` 所用代码；核验 `configs/adatad/thumos/duca_uvt_value_portal_n16r4.py` 与 `opentad/models/backbones/backbone_wrapper.py`。 |
| 历史 65.385724 | https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/42dba3f90b37243e7965d18b6707e88e81bf7109 | 审阅 `opentad/models/detectors/two_stage.py` 与 `opentad/models/selectors/duca_online_frame_selector.py`，辨析历史非均匀输入的实际合同。 |

不要使用或引用根目录污染的 SparseHead 修订 `a6bdc084…`；它不属于 DUCA 代码身份。

## 已有结果和历史证据

| 证据 | 已知事实 | 允许的解释 |
| --- | --- | --- |
| 历史 K=384 `65.385724` | commit `42dba3f90b37243e7965d18b6707e88e81bf7109`；先 30 epoch exact-uniform full detector，再 60 epoch learned sampling/full detector；后段有 contribution distillation、`density_transport_st` 和 ASFormer adaptation。384 个非均匀 selected RGB 来自 768 帧窗口，输入为 24 temporal chunks/384 VideoMAE frames。 | 这是间接非均匀输入有正信号的真实历史结果，不是 uniform；但它是 90 epoch、多因素课程，不能当作公平 selector 增益。 |
| 历史 uniform/grid | Job `1150701`：64.352，native stride-2/adaptor ActionFormer；Job `1150842`：65.696，grid-aware detector/geometry。 | 都是 protocol-unmatched 的只读背景锚点，不能改称非均匀 DUCA，也不能填入新主表。 |
| UVT 开发训练 | Job `1244840`, 60 epoch, seed 3407：off 57.35、geo 55.93、geo+EMA 55.92 Avg-mAP。 | 当前组合为负诊断；V(t) 同时改变 selection score、geometry/EMA supervision 与 K evidence，不能归因。 |
| Fovea/Query-Bridge 开发训练 | Job `1244851`, 60 epoch, seed 3407：baseline fused 42.94、query only 45.26、query GT-mask 49.16、query cycle 54.67、query fovea 43.77。 | `query_cycle` 是待拆分的正信号，不是 Query 或 cycle 已有效的结论；该波缺同提交 matched controls、两臂和多 seed。 |

### 已完成对照的约束

官方 dense、严格 fixed-K uniform 和 seeded random 的多次历史/VC（版本控制）记录已经存在；本轮**不得为了凑矩阵重跑这些基线**。它们应被只读绑定为对照证据。若某个新 runtime 修改了 heavy backbone 的输入合同，以至旧对照不再同构，必须诚实标为跨运行时诊断，不能把重复 baseline 训练伪装成必要步骤。

## 代码核验事实

1. UVT 配置同时启用 `selection_strategy="dynamic_B"`、`straight_through_detector_loss=True`、256--512 dynamic budget 与多项辅助损失；outer K 只读 actionness，`value_mode` 又变更 alpha、geometry 和 EMA。
2. 当前 UVT wrapper 会先把按物理位置排序但可不连续的 selected frames reshape 成 16-frame VideoMAE clips；physical positions 在 backbone 后才用作插值。静态代码足以证明有伪连续时间风险，不能单独量化其 mAP 后果。
3. Fovea 全部 arm 都有 `dynamic_budget=True`；`query_gt_mask`、`query_cycle`、`query_fovea` 同时改变不止一个变量。现有 loss 将 endpoint=2 与 interior=1 都折叠为动作标签，center 逼近零，width 的均值被最小化，diversity 只有 entropy、没有 query orthogonality。
4. 历史 `42dba3f` 使用 `_gather_time` 按 selected positions 收集 RGB，并在重型 backbone 前替换 dense input；proposal 在 NMS 前回映 physical time。但该 forward 合同也没有把 selected-rank 邻接的真实时间间隔作为 VideoMAE 内部输入。

## 正在考虑的、尚未冻结的方案

不要把以下候选混成一条方法。请先攻击它们的科学性、公平性和可发表性，再只推荐一个。

### H：历史兼容的间接输入参考臂

保持 `65.385724` 的核心输入形态：完整低成本 actionness/transition scout 产生确定性 global structured exact-K 非均匀位置；selected RGB 保持物理排序后按 selected rank 输入 existing VideoMAE。仅移除 90 epoch、直接 detector gradient、contribution distillation、全 ASFormer adaptation 等混杂。它是诊断 reference，不必成为主方法。

### S：语义间接 Query-Bridge 主候选

低成本 scout 输出 action/start/end logits；Query-Bridge 只提供上下文 residual，不能直接输出 index、K、Gumbel、MMR/DPP 或 detector utility。确定性 acquisition 从语义概率选位置，K 使用独立窗口级聚合，不读 detector、GT、训练 loss 或历史 mAP。

需要把以下两项做成严格单变量、训练期机制：

- `S0`：无 Query、无 knowledge transfer 的 semantic scout；
- `SQ`：只加入 Query contextual residual；
- `SQD`：在 `SQ` 上加入 frozen training-population teacher 对 action/start/end logits 的 detached soft target。teacher 不参与部署、位置/K 决策或 detector 反传；额外 P0 成本单列、detector 成功更新数不增加。

### P：物理时间运行时候选

可采用连续 16-frame cliplet，使 VideoMAE tubelet 不跨越不连续物理帧；每个 cliplet 需保留原始 frame indices、timestamps、fps/timebase，重建与 proposal/NMS 均使用 physical time。它是合理候选，但不是已证实唯一修复：它会改变预训练位置编码、跨 block 时序建模和有效采样几何。

## 请进行的对抗性审查

1. 按“事实 / 高优先级假设 / 未验证主张”重排以上所有结论；找出任何偷换、泄漏、过预算、统计或成本不公平。
2. 明确回答：历史 65 的强信号更可能来自非均匀输入、课程预热、知识传递/gradient bridge、ASFormer adaptation、伪连续时间的偶然容忍，还是其他解释？哪些可由最少的新实验区分？
3. 严厉攻击 S0/SQ/SQD：teacher 语义蒸馏是否仍会破坏公平性或创新性？如何让“前后协同”成为可发表的机制而不是额外训练/teacher 的收益？
4. 在不得重复现有 official dense/uniform/random 基线训练的条件下，设计最小的新实验包。要求充分利用已存在 VC baseline receipts；若无法严格同构，写出诚实的证据边界而不是要求重复基线。
5. 在 H、S、P 或你提出的单一更优非固定-K方案中，选择一条唯一推荐路线。它必须保持 dynamic K 为最终核心，fixed K 仅对照/回退，并给出惊喜观察、领域机制、可证伪预测、最便宜真实视频实验、失败时的停止规则和论文 claim 边界。
6. 给出可实现的文件/模块合同、loss/gradient ownership、physical-time path、checkpoint recovery、成本核算和实际 `executed_k` runtime hook。禁止以配置字段或 padding 代替真实 variable VideoMAE compute。
7. 给一个顺序明确的实验计划：先 P0 语义机制，再固定-K TAD 的新方法臂，再 dynamic-K；说明何时只需要现有 VC 对照，何时必须承认不可比。不得声称已有 mAP/成本结果可证明该路线。

输出中文，像顶级会议的反驳审稿意见与可执行替代方案；最后给唯一 `CONTINUE / REVISE / PIVOT / STOP` 裁决，不把路线选择交回用户。

## 研究记录入口（非代码附件）

- 当前论文缩略报告：`PAPER_PROGRESS.md`
- 历史与资源审计：`research-wiki/DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`
- 两份既有审查吸收：`research-wiki/DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md`、`research-wiki/DUCA_SECOND_REVIEW_COMPARISON_AND_ABSORPTION-2026-08-20.md`
- 历史课程：`research-wiki/experiments/duca-rate25-sampling-rate-curriculum.md`
- 代码请只从上表 GitHub 地址读取；本轮不附模型源码。
