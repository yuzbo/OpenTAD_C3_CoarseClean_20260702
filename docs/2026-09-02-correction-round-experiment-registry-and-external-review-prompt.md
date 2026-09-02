# 2026-09-02 更正轮实验矩阵、提交台账与外部逐行审查 Prompt

> 文档日期：2026-09-02  
> 目的：记录本轮 DUCA Evidence、DUCA CT-DP-BAMoD、ZoomToken BAFDR、ZoomToken ET-TRC 更正路线的代码提交、实验矩阵、远端状态和可审计证据。  
> 结论口径：代码分支已落地不等于实验已经完成；没有门禁、终态 checkpoint 和有效评测回执，不得填写 mAP。

## 1. 总体裁决

本轮四条路线均已有独立修正分支并推送到同一个 GitHub 仓库，但尚未形成一个合并到根分支的单一发布版本。四个远端 checkout 当前均不是各自分支的最新提交；正在运行或排队的 Slurm 作业大多由旧提交产生，不能直接作为更正后的正式结果。

当前正式结果栏统一为 **NO VALID RESULT**，直到同时满足：

1. 远端 checkout 的 HEAD 等于本表列出的最新 SHA，且工作树干净；
2. 对应数值、几何、预训练覆盖或 CUDA 门禁通过；
3. 训练达到预先规定的成功更新数/终态 epoch；
4. 使用终态 EMA checkpoint 执行一次独立评测并保存 Slurm、checkpoint、评测日志和配置回执。

## 2. GitHub 提交总表

仓库：[yuzbo/OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)

| 路线 | 本地工作树 | 分支 | 最新提交 | GitHub |
|---|---|---|---|---|
| DUCA Evidence 数值/重放更正 | E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902 | codex/duca-evidence-recovery-numerical-correction-20260902 | 4b6df22a | [branch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-numerical-correction-20260902) |
| DUCA CT-DP-BAMoD 几何更正 | E:\DeskTop\TAD\duca_ctdp_revised_20260902 | codex/duca-ctdp-geometry-mechanism-correction-20260902 | b568ca84 | [branch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-geometry-mechanism-correction-20260902) |
| ZoomToken BAFDR 梯度更正 | E:\DeskTop\TAD\zoomtoken_bafdr_correction_20260902 | codex/zoomtoken-bafdr-gradient-correction-20260902 | 42946779 | [branch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-gradient-correction-20260902) |
| ZoomToken ET-TRC 预训练/启动更正 | E:\DeskTop\TAD\zoomtoken_ettrc_correction_20260902 | codex/zoomtoken-et-trc-correction-20260902 | b3be6482 | [branch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-correction-20260902) |

本地四个工作树均已通过 git diff --check，并与各自的 origin/<branch> 对齐。根工作树 E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702 仍有历史文档改动，未将这些无关改动混入更正分支。

## 3. 本轮实验矩阵

### 3.1 DUCA Evidence Recovery：8 臂、单 seed 8261

代码入口：

- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\opentad\models\utils\numerics.py
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\opentad\models\duca\evidence_recovery.py
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\opentad\models\bricks\dense_temporal_recovery.py
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\opentad\models\bricks\temporal_token_merge.py
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\tools\train_engine.py

矩阵：

| 臂 | 配置 | 预期隔离因素 |
|---|---|---|
| C0 | duca_evidence_recovery_matched_h65_60.py | 匹配 H65-60 对齐控制 |
| F | duca_evidence_recovery_full.py | 覆盖、时间调节、鲁棒、Merge、Recovery 全组合 |
| A1 | duca_evidence_recovery_no_coverage.py | 移除覆盖约束，纯语义 Top-K |
| A2 | duca_evidence_recovery_no_time.py | 移除时间条件调节，固定 g=1 |
| A3 | duca_evidence_recovery_no_robust.py | 移除鲁棒训练和蒸馏 |
| A4 | duca_evidence_recovery_no_merge.py | 移除 VideoMAE Token Merge |
| A5 | duca_evidence_recovery_no_recovery.py | 移除特征重建，直通选定坐标 |
| A6 | duca_evidence_recovery_h65_selection.py | 冻结 H65 选择器 |

门禁和启动器：

- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\tests\test_duca_evidence_recovery_numerics.py
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\scripts\submit_duca_evidence_recovery_cuda_gate_n16r4.sh
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\scripts\submit_duca_evidence_recovery_nonfinite_replay_n16r4.sh
- E:\DeskTop\TAD\duca_evidence_numerical_correction_20260902\scripts\submit_duca_evidence_recovery_single_seed_n16r4.sh

实现状态：数值有限性检查、梯度审计、AMP 重放、短序列/全 mask 恢复、Merge 顺序同步、严格零尾部和 K=384 选择约束已落地。更正分支较早提交 ff186be6 在远端通过过 31 passed, 1 skipped；最后的提交绑定脚本变更已进入 4b6df22a，需要重跑门禁。

远端部署：/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_fix_1d8a893d 当前为 ff186be，落后于 4b6df22a。旧 Evidence array 1265077_0,3,4,5,6 仍在运行，但不是本轮更正后的有效正式结果；新的 8 臂 seed 8261 尚未完成提交。

### 3.2 DUCA CT-DP-BAMoD：G0-G3 核心 4 臂 + 机制矩阵

代码入口：

- E:\DeskTop\TAD\duca_ctdp_revised_20260902\opentad\models\selectors\dual_phase_frame_selector.py
- E:\DeskTop\TAD\duca_ctdp_revised_20260902\opentad\models\backbones\vit_adapter.py
- E:\DeskTop\TAD\duca_ctdp_revised_20260902\opentad\models\backbones\backbone_wrapper.py

核心矩阵：

| 臂 | 机制设置 | 目的 |
|---|---|---|
| G0 | Dual-Phase + CT-Tubelet，selected-axis | 几何/骨架基线 |
| G1 | G0 + B-AMoD | 测试边界偏置 Token 路由 |
| G2 | G0 + CT-Conv1d physical-grid | 测试连续物理坐标检测头 |
| G3 | G1 + CT-Conv1d physical-grid | 完整组合 |

后续机制对照：M10（CT-Tubelet only）、M01（B-AMoD only）、M11（CT-Tubelet+B-AMoD），必须在 G0-G3 门禁通过后执行。

测试与启动器：

- E:\DeskTop\TAD\duca_ctdp_revised_20260902\tests\test_duca_ctdp_corrected_geometry.py
- E:\DeskTop\TAD\duca_ctdp_revised_20260902\scripts\submit_duca_ctdp_corrected_campaign_n16r4.sh

实现状态：物理时间元数据、短序列 padding mask、B-AMoD 的有效 token 约束、CT Conv stride/dilation parity、selected-axis 与 physical-grid 配置分流已落地。远端 679b7121 做过旧版本测试；最新 G0/G1 配置修复提交为 b568ca84，尚未在远端重跑完整 focused tests。

远端部署：/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902 当前为 679b7121。作业 1265777/1265778（G0/G1）运行中，1265779/1265780（G2/G3）因依赖等待；这些作业不能替代最新 SHA 的正式矩阵。

### 3.3 ZoomToken BAFDR：5 臂 screen，21-cell 条件开放

代码入口：

- E:\DeskTop\TAD\zoomtoken_bafdr_correction_20260902\opentad\models\backbones\bafdr_wrapper.py
- E:\DeskTop\TAD\zoomtoken_bafdr_correction_20260902\opentad\models\projections\bafdr_asymmetric_proj.py
- E:\DeskTop\TAD\zoomtoken_bafdr_correction_20260902\scripts\submit_zoomtoken_bafdr_screen_n16r4.sh

第一阶段 5 臂：

| 臂 | 设置 |
|---|---|
| G96 | 全局 96 Token 对照 |
| U16-UNIFORM-A0 | 16 Token 均匀采样 |
| BAFDR-K16-LATE | BAFDR K=16，late route |
| BAFDR-K16-NOKD | BAFDR K=16，无 KD |
| BAFDR-K16-FULL | BAFDR K=16 完整路线，必须有终态 Teacher |

BAFDR-K16-FULL 在缺少终态 Teacher checkpoint 或 epoch59/state_dict_ema 时应 fail closed。只有 5 臂 screen 通过成本、梯度和性能门禁后，才允许展开 21-cell 矩阵。

远端部署：/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_correction_20260902 当前为 41d5f252，落后于 42946779。作业 1265796/1265797/1265798（U16/LATE/NOKD）运行中；FULL 未提交，21-cell 未开放。当前无有效 BAFDR 最终 mAP。

### 3.4 ZoomToken ET-TRC：OFF/ON 配对矩阵

代码入口：

- E:\DeskTop\TAD\zoomtoken_ettrc_correction_20260902\opentad\models\backbones\et_trc_videomae.py
- E:\DeskTop\TAD\zoomtoken_ettrc_correction_20260902\opentad\models\backbones\backbone_wrapper.py
- E:\DeskTop\TAD\zoomtoken_ettrc_correction_20260902\tests\test_et_trc_pretrain_parity.py
- E:\DeskTop\TAD\zoomtoken_ettrc_correction_20260902\scripts\submit_zoomtoken_et_trc_n16r4.sbatch

| 臂 | 设置 |
|---|---|
| ET-OFF | 关闭 ET-TRC，匹配基线 |
| ET-ON | 开启 ET-TRC，共享低秩一阶残差近似 |

门禁：必须检查 clean checkout、精确 HEAD、预训练权重覆盖、允许缺失 key 前缀、fused-QKV 拆分、stride-1 parity、full-KV sensitivity 和真实梯度。

远端部署：/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902 当前为 827c2526，落后于 b3be6482。作业 1265818（OFF）运行中，1265819（ON）排队且受 AssocGrpGRES 限制。最新 parity/strict-loading 改动尚未在远端重跑。

## 4. 当前 Slurm 快照

核验时间：2026-09-02（N16R4）。

| 作业 | 状态 | 解释 |
|---|---|---|
| 1265077_0,3,4,5,6 | RUNNING | 历史 Evidence 作业，非最新更正 SHA |
| 1265777/1265778 | RUNNING | CT G0/G1，提交早于最新 b568ca84 |
| 1265779/1265780 | PENDING | CT G2/G3，等待依赖 |
| 1265796/1265797/1265798 | RUNNING | BAFDR screen 的 3 个非 FULL 臂 |
| 1265818 | RUNNING | ET-TRC OFF |
| 1265819 | PENDING | ET-TRC ON，AssocGrpGRES |

这些作业只能作为部署过程观察对象。没有精确 SHA、干净工作树、门禁回执和终态 EMA 的作业，不能写入论文结果表。

## 5. 外部审查 Prompt（可直接复制）

你是负责复现实验和代码审计的严厉外部审查员。请审查以下 2026-09-02 更正轮的四条路线，不要默认作者描述正确，也不要用“代码能 import”替代行为正确性。

### 审查对象

1. DUCA Evidence：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-numerical-correction-20260902，目标 SHA 4b6df22a。
2. DUCA CT-DP-BAMoD：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-geometry-mechanism-correction-20260902，目标 SHA b568ca84。
3. ZoomToken BAFDR：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-gradient-correction-20260902，目标 SHA 42946779。
4. ZoomToken ET-TRC：https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-correction-20260902，目标 SHA b3be6482。

### 必须逐行检查的内容

对每条路线逐个文件、逐个函数和关键配置逐行检查，并给出 file:line 证据：

#### A. 设计一致性

- 代码是否实现了路线声明的数学算子，而不是只添加配置字段或日志字段？
- 配置中的预算、采样率、时间轴、stride、dilation、K、epoch、成功更新数是否真正传递到运行时？
- 消融臂是否只移除目标因素，是否意外改变了优化器、数据、seed、训练量或评测协议？
- Evidence 的 K=384、1000/13200 成功更新和 AMP replay 是否真实成立？
- CT 的 selected-axis、physical-grid、padding mask 和真实物理时间是否语义一致？
- BAFDR 的 late/NOKD/FULL 是否真的只改变声明因素，FULL 是否 fail closed 于缺失 Teacher？
- ET-TRC 的“低秩一阶残差近似”是否被错误宣称为精确 JVP；预训练覆盖报告是否完整且严格？

#### B. 时间、坐标与路线错误

- 检查原始帧号、dense candidate index、tubelet midpoint、selected-axis index、秒级 timestamp 是否发生混用。
- 检查 padding token、短序列、非均匀采样、stride/dilation 对物理时间和 mask 的影响。
- 检查 K 的预算是否在 selector、backbone、detector、loss、评测和 ledger 中保持一致。
- 检查终态 epoch/EMA、resume、seed、AMP replay 是否引入时间或训练量偏差。
- 检查 G0-G3 与 M10/M01/M11 的执行顺序是否遵守门禁；检查 BAFDR 5-arm screen 未通过前是否错误展开 21-cell。
- 检查远端作业提交的 checkout/HEAD 是否就是目标 SHA；旧作业结果不得被归因到新代码。

#### C. 前后矛盾与不可证伪声明

- 搜索配置、脚本、README、测试和日志中互相冲突的默认值、路径、GPU、world size、epoch、seed、commit 或结果。
- 检查脚本是否先加载错误环境再覆盖变量，是否在登录节点训练，是否静默 fallback 到旧 checkpoint。
- 检查测试是否只验证 shape/静态字符串而没有验证梯度、数值、时间语义和真实 detector path。
- 检查“通过测试”“运行中”“PENDING”“checkpoint 存在”和“取得有效 mAP”是否被错误等同。
- 检查任何 mAP 是否具有完整的 config、commit、终态 checkpoint、评测命令、数据划分和日志回执。

### 审查输出格式

请按路线输出：

1. Verdict：PASS、PASS WITH BLOCKERS 或 FAIL。
2. 逐行问题表：严重级别(P0/P1/P2) | file:line | 观察到的代码 | 违反的设计/协议 | 可复现后果 | 修复建议。
3. 矩阵完整性表：每个臂的因素、配置、seed、预算、训练量、门禁、远端 SHA、作业号、终态 checkpoint、有效 mAP。
4. 时间/坐标审计：明确每个 index/timestamp 的单位、来源和消费者，指出任何混用。
5. 因果归因裁决：哪些差异可以归因于目标因素，哪些被训练量、权重、数据或提交差异污染。
6. 结果裁决：没有完整证据时必须写 NO VALID RESULT，不得估算或沿用旧结果。
7. 最小修复计划：按阻塞关系列出下一步，禁止建议无门禁地重跑全部历史实验。

审查时必须区分：

- 代码已提交：GitHub 分支存在目标 SHA；
- 代码已验证：目标 SHA 上的 focused tests/gates 通过；
- 代码已部署：远端 checkout clean 且 HEAD 等于目标 SHA；
- 实验已完成：达到规定终态并有有效 EMA checkpoint；
- 结果可发表：独立评测回执完整且满足 matched-protocol 与无泄漏约束。

任何一层缺失，都不得把该路线标记为“全部完成”。
