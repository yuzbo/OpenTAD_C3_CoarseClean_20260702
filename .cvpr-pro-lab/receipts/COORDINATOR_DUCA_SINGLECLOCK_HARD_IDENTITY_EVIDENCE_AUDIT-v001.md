# COORDINATOR_DUCA_SINGLECLOCK_HARD_IDENTITY_EVIDENCE_AUDIT-v001

- status: `MATERIAL_NEEDS_ATTENTION / EVIDENCE_ADMISSION_BLOCKED`
- audited_at: `2026-08-24T19:45:00+08:00`
- candidate_worktree: `E:/DeskTop/TAD/OpenTAD_DUCA_H65_FirstMixSingleClock_20260824`
- clean_head: `61065a50dc044f54876de5abfcad8c25559fede4`
- training_revision: `08a817e91867839abf3a81e24f8469512b26a6ea`
- formal_evaluation_revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- running_job: `1253090 / RUNNING`
- evidence_class: `READ_ONLY_IMPLEMENTATION_AND_ARTIFACT_ACCOUNTING`
- efficacy_result: `NONE`

## Current scientific question

在固定 `K=384`、保持历史 H65 语义间接非均匀逐帧选择与检测协议不变时，仅在 VideoMAE 第一次时间混合处加入真实物理时间残差，是否既保持 H65 兼容性能，又改善高 IoU 定位。

## Verified implementation boundary

1. `opentad/models/detectors/actionformer.py:232-299` 的当前生产身份审计记录选中位置、有效掩码、选中 RGB 和完整 VideoMAE 输入哈希，并合并同一物理窗口的完全相同重复暴露。
2. `scripts/run_duca_h65_singleclock_terminal_eval_n16r4.sbatch:87-190` 运行 SingleClock final/EMA、gate-zero twins、H65 OFF final/EMA、训练集分层冻结、10,000 次整视频 bootstrap 与分层统计。
3. `tools/bata/finalize_duca_h65_singleclock_terminal.py:1027-1039` 还要求机器可读的独立实现审查回执，schema 为 `duca_h65_singleclock_unit1_gate_implementation_review_v1`、verdict 为 `UNIT1_GATE_IMPLEMENTATION_PASS` 且 `focused_tests_pass=true`，否则 Query 准入保持关闭。

## Missing hard evidence

1. H65 回放五边界身份回执不存在。尤其没有检测器原始选中查询的 reference/replay 对，因此不能从现有位置/RGB/VideoMAE 输入哈希离线推导合法的五边界身份。
2. nominal-uniform 路径逐位身份回执不存在。当前没有 canonical-uniform 位置、零残差/零偏置、首次时间混合输出及最终骨干输出的冻结 reference/replay 对。
3. 独立审查的科研文档 `research-wiki/experiments/DUCA_H65_SINGLECLOCK_UNIT1_GATE_IMPLEMENTATION_REVIEW-v001.md` 记录 focused recheck PASS 和 33 项无数据测试，但当前工作树没有终结器所要求的机器可读 JSON 回执。

## Disposition

- 上述缺失不构成 `KILL_SINGLECLOCK_REPRESENTATION`，也不是效能反证。
- Job `1253090` 的统计完成后，若没有独立冻结的硬身份材料，Evaluator 必须将证据判为不可准入，不能签发正式 PASS 或 KILL。
- 现阶段不得根据运行日志片段推断 mAP、置信区间、分层收益或成本优势。
- 当前冻结边界不授权新的模型、checkpoint、协议、推理包装、训练或调参恢复。

## Handoff

- current_scientific_question: `First-Mixing SingleClock 是否在 H65 固定 K=384 合同下保持性能并改善物理时间表示。`
- next_owner: `Independent Evaluator`
- next_action: `Job 1253090 终态后，读取既有材料并按 Gate-v2 将结果归类为合法 PASS/KILL 或证据不可准入。`
- dependency: `Job 1253090 terminal artifacts；两项硬身份回执若仍缺失则只能输出 evidence invalid。`
- expected_return_at: `Job 1253090 terminal 后立即。`
- single_recovery: `none under the current frozen evidence chain`

