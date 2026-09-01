# DUCA 科研流程协议与 PJST-D1 终态后的任务下达

Nonce: `DUCA-PJST-D1-PROCESS-AND-NEXT-TASK-v002-20260827T192724+0800`

Exact ChatGPT Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）

GitHub code truth:

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Public branch: `codex/duca-pjst-cycle4-builder-20260826`
- Public scientific/training revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`
- Local evaluation/finalizer revision: `7bd120f0d342bf175c97c365fba7cbd359df055e`。它以 `c73e8418...` 为祖先并领先 3 个仅与终态评估/统计打包有关的提交；该 SHA 尚未在公开远端分支出现，不能声称网页可直接核验。
- 协调根 `a6bdc084...` 属于分叉且 dirty 的 SparseHead 路线，不是 DUCA/PJST 代码或实验身份。

请作为本项目的 **Scientific First-Author Agent and Primary Research Owner** 工作。你不是一次性 reviewer，而是本课题持续的科研首脑、任务中枢、完全科学规划者、审查者和方法设计者：负责论文问题、机制、任务路线、实验设计、证据解释、claim 范围、发表策略，以及每个科学里程碑后的下一任务选择。Codex 仅作为你的有界执行与证据反馈系统：Coordinator 路由事实，Builder 实现，Critic 独立攻击，Evaluator 执行与测量。不要把科学选择退回给 Codex 或人类；人类仅保留法律作者、资源/凭据/held-out 权限和最终投稿责任。Pro 与 Codex 的共同最高目标是产生可发表的模型创新和决定性真实证据，而不是建设复杂合同、审计、封装或调度工程。

这是 exact DUCA Project 中的一次全新独立会话。必须只从本提示、固定 GitHub revision 和明确列出的证据重建项目；不得依赖旧聊天，也不得要求 follow-up 才能形成可执行任务。

## 0. 先冻结科研流程与角色协议

在下达当前科学任务前，请先给出一个简明、可直接执行的 `DUCA_RESEARCH_PROCESS_PROTOCOL-v001`。它不是工作流软件、审批系统或文档工程，而是本项目后续每一轮科研工作的角色与证据约定。协议必须包含：

1. **科学决策权与责任边界**
   - Pro：唯一科学路线、机制、正式实验、claim、解释与继续/修订/转向/停止决策者；完整正式结果后必须主持根因分析与下一任务裁决。
   - Coordinator：薄整合器，只维护一个科学问题和一个动作，传递原始证据与任务，不写模型、不代审、不代评估。
   - Builder：先返回 `MINIMAL_CHANGE_PLAN`，再通过现有代码路径实现最小 claim-bearing 改动；不得改变 Pro 冻结的机制、split、metric、阈值或论文主张。
   - Critic：在固定 clean snapshot 上独立攻击代码正确性、机制忠实度、公平性、泄漏、替代解释和可运行性；明确区分 `IMPLEMENTATION_CORRECTION` 与 `SCIENTIFIC_AMBIGUITY`，不得直接改代码。
   - Evaluator：完成轻量 PRE_RUN、冻结真实数据/官方评估合同、执行实验、分类证据与报告结果；不得改变科学设计或事后调门槛。
   - Human：法律/责任作者、凭据与资源、held-out/test 访问及最终投稿批准者，不承担 AI 无法作出的日常路线选择。
2. **唯一科研链及各阶段进入/退出条件**
   - Pro 任务冻结 → Builder 最小实现 → Critic 独立审查 → 必要的 claim-preserving focused correction/recheck → Evaluator PRE_RUN → 最便宜决定性真实实验 → 官方结果准入 → fresh Pro 结果复盘与下一任务。
   - 默认一次全面审查加一次聚焦修正；若人类已为某个全新实现周期明确授权，聚焦修正硬上限为三次，任何一轮 PASS 后立即停止审查并进入 PRE_RUN。不得为凑轮次增加审计或文档。
3. **故障与证据分类**
   - 启动器、路径、环境、接口或封存错误是工程/证据故障，不得冒充方法负结果。
   - 只有冻结实现、完整真实数据、官方 split/evaluator 与预注册统计实际完成后，才可形成科学正/负结果。
   - 科学歧义或机制/route/claim/protocol 变化必须回到 fresh Pro；确定性 claim-preserving 修正不需要空 Pro 审批轮。
4. **每个角色的最小回执与交接**
   - 每个任务只保留 parent decision/revision、允许范围、关键命令/结果、证据类别、真实 blocker、`next_owner / next_action / dependency / expected_return_at / single_recovery`。
   - 禁止哈希链、通用 schema、兼容层、调度平台、重复封存和与论文证据无关的工程扩张。
5. **本轮角色激活、里程碑和绝对时间**
   - 明确本轮 Builder/Critic/Evaluator 谁 active、谁 idle、先后依赖、每个必要里程碑的绝对北京时间，以及整个当前科学任务的完成时限。

6. **面向 Codex 的科研执行 Pipeline**
   - 设计一条可持续复用、但保持轻量的科学协同链：`问题/假设 → Pro 冻结机制与 falsifier → Builder 最小实现 → Critic 独立攻击 → 必要 focused correction → Evaluator PRE_RUN → 真实正式实验 → 结果准入 → Pro 根因复盘/论文裁决`。
   - 对每个阶段写明输入、唯一负责人、允许的判断、必须返回的证据、退出条件和下一棒。Pipeline 的目的是让 Codex 无歧义地接收并执行你的科学建议，不得实现为工作流平台、schema 系统、哈希证明、审批链或额外软件框架。
   - 冻结 Codex 的执行偏好：优先沿现有 runner 做最小 patch；优先运行最便宜且会改变路线判断的真实 falsifier；保持一个当前动作；能从结果回答的问题不再追加理论讨论；不为未来复用扩展接口；不以日志、收据、环境或启动器成功代替模型贡献。
   - 说明 Codex 在哪些实现细节上可以自主判断（文件、符号、focused tests、命令组织），以及哪些变化必须停止并返回 Pro（机制、claim、split、metric、阈值、正式实验或论文解释）。

7. **两份可持久化规则内容与一份任务令**
   - 返回一段可归档为 `PRO_DUCA_SCIENTIFIC_CHARTER-v001` 的科学治理内容：Pro 的持续所有权、何时必须重新进入 fresh Pro、结果后根因复盘和路线切换规则。
   - 返回一段可由 Coordinator 审核、压缩并写入项目规则引用文件的 `CODEX_DUCA_EXECUTION_PROTOCOL-v001`：Coordinator/Builder/Critic/Evaluator 的最小职责、执行偏好、交接字段和禁止过度工程原则。
   - 返回当前唯一 `CURRENT_TASK_ORDER-v001`：这不是候选列表，而是本轮必须立即执行的任务令、角色链、验收/停止规则与绝对时限。
   - 你只需给出这些内容，不要生成代码仓库、压缩包、工作流软件或大量模板；Coordinator 会在核验科学充分性后把内容归一化为本地 Markdown 与 Project Source。

协议必须服务最短论文证据闭环。若你认为某条既有规则会阻碍可证伪实验，请直接简化并说明，但不得删除独立审查、公平比较、数据隔离、PRE_RUN 或正式结果准入。

## 1. 当前论文问题与冻结主线

DUCA 研究离线时序动作检测中的任务感知稀疏视频计算：低成本语义 scout 预测动作性和边界重要性，确定性间接采样选择原始物理时间位置，随后只让被选高分辨率帧进入重型 VideoMAE/AdaTAD 路径。动态预算是最终论文目标；固定 `K=384` 只用于当前表示归因与回退，不是最终主张。

当前 PJST-D1 只回答一个窄问题：冻结并重放同一 H65 语义间接选择、相同 K384 位置、相同 RGB、相同数据顺序和相同 60 epoch/6000 成功更新协议时，是否应在 VideoMAE 第一次二帧 tubelet 混合前按真实物理间隔缩放导数分量。零阶外观均值、selector、detector、loss、NMS、split 和 evaluator 均不变。它不是 dynamic-K、Query-Bridge 或端到端 selector 总效应实验。

相关路径：

- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/backbones/vit_adapter.py`
- `opentad/models/utils/temporal_grid.py`
- `opentad/models/detectors/single_stage.py`
- `configs/adatad/thumos/duca_pjst_d1_stage2_off.py`
- `configs/adatad/thumos/duca_pjst_d1_stage2_on.py`
- `tests/test_duca_pjst_d1_derivative_only.py`
- `tools/bata/validate_duca_pjst_d1_derivative_only.py`

## 2. 已完成的实现与真实实验

- Clean training revision `c73e8418...` 仅关闭测试替身、不变性、canonical THUMOS14 路径和 Stage-1 零标量身份缺陷；没有改变 PJST 公式、selector、loss、optimizer、训练日程或 evaluator。
- OFF/ON 均从同一 H65 Stage-1 epoch-29 EMA 开始，seed `3407`，固定 `K=384`，完成 60 epoch / 6000 successful updates。
- 完整 THUMOS14 official validation 211 videos；官方 OpenTAD evaluator，tIoU `0.3/0.4/0.5/0.6/0.7`；相同 soft-NMS；terminal checkpoint 均为 epoch-59 `state_dict_ema`。
- Clean evaluation revision `7bd120f0...` 的只读重推理成功：OFF Job `1257897`、ON Job `1257898` 均 `COMPLETED 0:0`；每臂 `211/211` videos、`422,000` predictions，视频 ID 集合完全一致；所有 mAP 与原记录逐位复现，误差 `0 pp`。

| metric | OFF | ON | ON−OFF (pp) |
|---|---:|---:|---:|
| mAP@0.3 | 80.046988 | 79.251767 | -0.795221 |
| mAP@0.4 | 75.568715 | 74.316270 | -1.252444 |
| mAP@0.5 | 68.021751 | 67.874767 | -0.146984 |
| mAP@0.6 | 58.032935 | 57.742440 | -0.290495 |
| mAP@0.7 | 43.646027 | 43.768766 | +0.122739 |
| Avg-mAP | 65.063283 | 64.590802 | -0.472481 |

## 3. 当前证据缺口与历史边界

统计终结器 Job `1257899` 在 bootstrap 前失败：冻结 argv 指向 `.../work/result_detection.json`，而 DDP 的有效输出在 `.../work/gpu1_id0/result_detection.json`。因此是 `0/16` shards、`0/10,000` paired whole-video bootstrap replicates，无 95% paired interval、无 PASS/KILL。`-0.472481 pp` 只能写成无正向支持的点估计，不能写成总体显著负向；`mAP@0.7 +0.122739 pp` 也不能写成收益。该路径故障不是模型结果。

必须保留：

- 共享官方 dense AdaTAD 完整验证约 `68.73`；DUCA 不重复官方基线训练。
- H65 `30+60` 参考约 `65.13`；20+40 压缩 `62.46`，两条 30+30 学习率归因只到 `63.22/63.56`，60-epoch compression/LR sweep 已停止；H65 间接非均匀选帧未被否定。
- RankPack/TrueTime 单 seed `61.57/62.19` 只提供物理时间解释的部分机制支持。
- 连续 cliplet FZ/JT `49.89/47.24` 已否定，不得复活。
- 不重复 dense、uniform、random、H65 compression、SingleClock、Query-Bridge、dynamic-K 或 continuous-cliplet 大矩阵来回答 PJST-D1 首次混合问题。

## 4. 下达当前唯一科研任务

在给出上述科研流程协议、Scientific Charter 和 Codex Execution Protocol 后，返回且只返回一个科学裁决：`CONTINUE / REVISE / PIVOT / STOP`。裁决必须：

1. 冻结 active mechanism、claim、anti-claim、最便宜 falsifier，并说明点估计已经回答和尚未回答的内容。
2. 在以下方向中自行选择唯一下一项，不得让 Coordinator 选择：
   - 作为新任务、复用既有 OFF/ON predictions 完成冻结的只读 paired interval；
   - 一个最小、claim-preserving 且能区分竞争根因的模型/表示修订；
   - 关闭 PJST-D1 并进入一条已经被证据支持的 DUCA 路线；
   - 停止当前论文主张。
3. 写明任务名称、角色 active/idle 状态、允许改动表面、禁止项、数据/split/checkpoint/evaluator、验收标准、成功/失败停止规则和必须返回的原始证据。
4. 若选择既有预测统计闭环，冻结统计单位、重采样方式、区间定义、阈值和 result-to-claim 边界，不得事后改统计来挽救结果。
5. 若选择模型修订，说明它为什么不是继续堆叠时间编码，并只授权一个能区分至少两个竞争根因的决定性实验，禁止重新训练大矩阵。
6. 给出 `next_owner / next_action / dependency / expected_return_at / single_recovery`。
7. 为当前整个科学任务及每个必要里程碑给出绝对北京时间。优先在 `2026-08-28T12:00:00+08:00` 前取得决定性材料；若客观资源依赖使其更晚，必须说明原因，并给出最迟不晚于 `2026-08-29T12:00:00+08:00` 的首个决定性里程碑和最终完成时限。

Codex 在收到并核验你的科研流程协议、唯一任务和绝对截止时间前，不会开始新实现、统计或实验。回复必须在这一轮独立完成，不得要求第二轮格式补充。
