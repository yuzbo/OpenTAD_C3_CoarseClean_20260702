# 六路线最新 GitHub 代码与 Pro 审查附件索引

GitHub 分支核验时间：2026-09-08 01:11:25 CST。
通过 git ls-remote 直接查询远端，不以本地 origin 缓存或历史消息代替。分支后续可移动，审查引用固定40位SHA。

## 官方 AdaTAD 基线优先

768帧只是输入长度，不能证明模型与官方相同。现有 REF-D768 67.58% 是修改配方的参考臂，尚不能称为原始官方 AdaTAD 复现。原始结构和原始训练/评测配方需分别核验。
官方公开69.03%、历史共享68.73%、本次REF-D768 67.58%必须分行；共享旧产物尚需溯源。严格6000更新的匹配对照不能替代官方原配方基线。
[官方结果表](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/configs/adatad/README.md#thumos-14-results)。
附件含官方源码快照346d09d1，未包含官方模型权重和训练日志；官方main快照不等于已证实的论文checkpoint训练SHA。

## 主审路线

| 路线 | 当前实现分支 | 固定源码 | 种子与状态 |
|---|---|---|---|
| H65-Pro：语义相位选帧、时间定位及因素消融 | [codex/h65-pro-physical-time-optimizer-repair-20260904](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-physical-time-optimizer-repair-20260904) | [f2068e18](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7) | 当前单种子3407；F01-F06 已有终态官方结果，但全部 phase=0；不能代表四相完整方案已验证。 |
| CT-DP：非均匀时间嵌入、检测几何与稀疏骨干路由 | [codex/duca-ctdp-successful-updates-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-successful-updates-20260907) | [78cde1be](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/78cde1be1cb8b3acc7d750afc92ea740f2a03d06) | 当前单种子3407；成功更新计数已修复；G0/G1 坐标闭环缺陷仍在；G0-G3 不包含开启 B-AMoD 的机制消融。 |
| DUCA-Unified：manifest 驱动的系统因素控制实验 | [codex/duca-unified-formal-gates-20260903](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-formal-gates-20260903) | [793c4f9c](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/793c4f9cdf7dac4f224bc73012aff8bc93949f87) | 开发种子3407，确认种子4407、5407、6407；41 个配置存在；Taylor P0/P1 和历史 H65 retention/transition 运行时机制仍被显式阻断。 |
| BAFDR：全局低分辨率载体、局部块路由和教师蒸馏 | [codex/zoomtoken-bafdr-successful-updates-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-successful-updates-20260907) | [710ce8a6](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/710ce8a6246c471742c83bf7c180d9ab87c36fac) | 当前单种子4407；修复成功更新后的重新训练与旧 5994-5997 更新结果严格区分；旧 receipt seal 不是新结果。 |
| ET-TRC：保持密集状态的 Transformer 局部残差近似 | [codex/zoomtoken-et-trc-formal-repair-20260903](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-formal-repair-20260903) | [74473c27](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/74473c2775caebf0da9d368ce8009d78e2942098) | 当前单种子4407；正式训练 OFF/ON 已出分；当前是固定 stride anchor + learned low-rank surrogate，不是已完成事件策略。 |
| Evidence-Recovery：预选阶段语义侦察、空洞控制与证据恢复 | [codex/duca-evidence-storage-resume-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-storage-resume-20260907) | [ce767b4d](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ce767b4df49a82b42f4fab4897ac4ea562e4b548) | 当前单种子8261；最新恢复 SHA 与 C0/A2、A1/A6、F/A3-A5 原始训练来源不同；最终 receipt 必须保留恢复谱系。 |

主表不是全矩阵完成报告；结果背景截至 2026-09-08 00:37 +08:00，本次只刷新代码身份和审查材料，没有重新SSH检查队列或重新计算mAP。

## 独立评测与历史来源

| 快照 | 角色 | 分支与精确提交 | 本地位置/状态 |
|---|---|---|---|
| official_adatad | 官方原始源码参考；优先核对模型和完整配方，不能只对比帧数 | [main](https://github.com/sming256/OpenTAD/tree/main) / [346d09d19e2091372cec48172dbe40f7b28bdee6](https://github.com/sming256/OpenTAD/commit/346d09d19e2091372cec48172dbe40f7b28bdee6) | 无对应当前worktree；按Git对象导出 |
| h65_train | 当前 F 臂训练实现 | [codex/h65-pro-physical-time-optimizer-repair-20260904](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-physical-time-optimizer-repair-20260904) / [f2068e18e2c68bdbdd7a607b47f32d05ac3beed7](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7) | `E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission`；tracked clean=True，untracked=0（未导出） |
| h65_reference | 已出分 REF-D768 / REF-U384 的历史训练实现，不是当前 F 臂 | [codex/h65-pro-admission-fix-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-admission-fix-20260902) / [e553a5a4a1063a755900d3dfa4bf8909bf97d466](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e553a5a4a1063a755900d3dfa4bf8909bf97d466) | 无对应当前worktree；按Git对象导出 |
| h65_evaluator | 独立评测器，不替代训练 SHA | [codex/h65-pro-eval-binding-repair-20260903](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-eval-binding-repair-20260903) / [67c8f39fa7d20b865a0f77adf75d418a374c7ff5](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/67c8f39fa7d20b865a0f77adf75d418a374c7ff5) | `E:/DeskTop/TAD/_duca_fix_worktrees/h65_eval_repair`；tracked clean=True，untracked=0（未导出） |
| ctdp_train | 当前 successful-update 修复训练实现；坐标问题尚未修复 | [codex/duca-ctdp-successful-updates-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-successful-updates-20260907) / [78cde1be1cb8b3acc7d750afc92ea740f2a03d06](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/78cde1be1cb8b3acc7d750afc92ea740f2a03d06) | `E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_successful_updates`；tracked clean=True，untracked=0（未导出） |
| ctdp_evaluator | 为 78cde1be 绑定的独立评测器 | [codex/duca-ctdp-terminal-eval-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-terminal-eval-20260907) / [11ced13ac6b72091d26405c5d6f152f09914c22d](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11ced13ac6b72091d26405c5d6f152f09914c22d) | `E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_terminal_eval`；tracked clean=True，untracked=0（未导出） |
| unified_implementation | 当前系统消融实现，缺失机制仍阻塞正式矩阵 | [codex/duca-unified-formal-gates-20260903](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-formal-gates-20260903) / [793c4f9cdf7dac4f224bc73012aff8bc93949f87](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/793c4f9cdf7dac4f224bc73012aff8bc93949f87) | `E:/DeskTop/TAD/OpenTAD_DUCA_Unified_FormalGates_20260903`；tracked clean=True，untracked=0（未导出） |
| bafdr_train | 当前 6000 successful-update 修复训练实现 | [codex/zoomtoken-bafdr-successful-updates-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-successful-updates-20260907) / [710ce8a6246c471742c83bf7c180d9ab87c36fac](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/710ce8a6246c471742c83bf7c180d9ab87c36fac) | `E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_successful_updates`；tracked clean=True，untracked=0（未导出） |
| bafdr_old_receipt | 旧结果独立补封存；不能使 5994-5997 次更新成为合格 6000 | [codex/zoomtoken-bafdr-receipt-seal-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-receipt-seal-20260907) / [84f1f0356766ed9e695ff55244b0a1ad429cd84c](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/84f1f0356766ed9e695ff55244b0a1ad429cd84c) | `E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_receipt_seal`；tracked clean=True，untracked=0（未导出） |
| ettrc_train | OFF/ON 已出分的正式训练实现 | [codex/zoomtoken-et-trc-formal-repair-20260903](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-formal-repair-20260903) / [74473c2775caebf0da9d368ce8009d78e2942098](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/74473c2775caebf0da9d368ce8009d78e2942098) | `E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902`；tracked clean=True，untracked=0（未导出） |
| ettrc_evaluator | 当前独立终态 EMA 评测器；不是新训练模型 | [codex/zoomtoken-ettrc-terminal-eval-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-ettrc-terminal-eval-20260907) / [67d7079d0d1c33e129d31cd3a45aaf93a67db252](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/67d7079d0d1c33e129d31cd3a45aaf93a67db252) | `E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval`；tracked clean=True，untracked=0（未导出） |
| evidence_resume | 当前存储迁移/精确恢复实现；需保留原始训练来源 | [codex/duca-evidence-storage-resume-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-storage-resume-20260907) / [ce767b4df49a82b42f4fab4897ac4ea562e4b548](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ce767b4df49a82b42f4fab4897ac4ea562e4b548) | `E:/DeskTop/TAD/_duca_fix_worktrees/evidence_storage_resume`；tracked clean=True，untracked=0（未导出） |
| evidence_optimizer_source | A1/A6 恢复前训练来源 | [codex/duca-evidence-optimizer-repair-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-optimizer-repair-20260907) / [1570a72507491899a50767700a35a04eee3f5fe9](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/1570a72507491899a50767700a35a04eee3f5fe9) | `E:/DeskTop/TAD/_duca_fix_worktrees/evidence_optimizer_repair`；tracked clean=True，untracked=0（未导出） |
| evidence_remaining_source | F/A3/A4/A5 恢复前训练来源 | [codex/duca-evidence-remaining-arms-20260907](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-remaining-arms-20260907) / [73bdd34ae21c6c675a00d1927b8af7c12f9edc05](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/73bdd34ae21c6c675a00d1927b8af7c12f9edc05) | `E:/DeskTop/TAD/_duca_fix_worktrees/evidence_remaining_arms`；tracked clean=True，untracked=0（未导出） |
| evidence_published_results | C0/A2 已出分来源，不能归到最新恢复代码 | [codex/duca-evidence-fullgrid-repair-20260904](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-fullgrid-repair-20260904) / [246058f2c24edc78818ada60eec26249bbf7d5d2](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/246058f2c24edc78818ada60eec26249bbf7d5d2) | 无对应当前worktree；按Git对象导出 |

## 源码入口

每个主审快照含该SHA下完整opentad源代码、训练/测试入口、路线配置、静态可解析继承base、相关工具/脚本和focused tests；补充快照只含所列审查表面。共享路径在不同快照中不可混用。
从配置实际构建的loader/selector/backbone/optimizer/head/postprocess追踪，不因附件内存在其他模型就把它计入本路线。源码附件用于审查，不是带依赖、数据和权重的可运行环境。

### official_adatad / 官方原始源码参考；优先核对模型和完整配方，不能只对比帧数

- 导出目录：`snapshots/official_adatad__346d09d1/`；169 个源码/配置/测试文件。
- 源码提交时间：2026-07-14T15:56:15+08:00；Enforce long type for labels when using one class (#75)。
- [configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)
- [opentad/models/backbones/vit_adapter.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/opentad/models/backbones/vit_adapter.py)
- [opentad/models/backbones/backbone_wrapper.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/opentad/models/backbones/backbone_wrapper.py)
- [opentad/models/detectors/actionformer.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/opentad/models/detectors/actionformer.py)
- [opentad/cores/optimizer.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/opentad/cores/optimizer.py)
- [opentad/cores/train_engine.py](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/opentad/cores/train_engine.py)

### h65_train / 当前 F 臂训练实现

- 导出目录：`snapshots/h65_train__f2068e18/`；276 个源码/配置/测试文件。
- 源码提交时间：2026-09-04T08:00:27+08:00；Stabilize H65 boundary comparison test。
- [configs/adatad/thumos/h65_pro/base_h65_pro_strict60.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/configs/adatad/thumos/h65_pro/base_h65_pro_strict60.py)
- [tools/bata/generate_h65_pro_fullmatrix.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/tools/bata/generate_h65_pro_fullmatrix.py)
- [opentad/models/duca/acquisition.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/opentad/models/duca/acquisition.py)
- [opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py)
- [opentad/models/backbones/vit_adapter.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/opentad/models/backbones/vit_adapter.py)
- [opentad/cores/optimizer.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7/opentad/cores/optimizer.py)

### h65_reference / 已出分 REF-D768 / REF-U384 的历史训练实现，不是当前 F 臂

- 导出目录：`snapshots/h65_reference__e553a5a4/`；62 个源码/配置/测试文件。
- 源码提交时间：2026-09-03T16:05:59+08:00；fix(h65): pad short-window coordinates to frame budget。
- [configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e553a5a4a1063a755900d3dfa4bf8909bf97d466/configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py)
- [configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/e553a5a4a1063a755900d3dfa4bf8909bf97d466/configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py)

### h65_evaluator / 独立评测器，不替代训练 SHA

- 导出目录：`snapshots/h65_evaluator__67c8f39f/`；49 个源码/配置/测试文件。
- 源码提交时间：2026-09-04T02:39:48+08:00；fix H65 terminal evaluation identity。
- [tools/experiments/run_h65_pro_eval.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/67c8f39fa7d20b865a0f77adf75d418a374c7ff5/tools/experiments/run_h65_pro_eval.sbatch)
- [tools/test.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/67c8f39fa7d20b865a0f77adf75d418a374c7ff5/tools/test.py)

### ctdp_train / 当前 successful-update 修复训练实现；坐标问题尚未修复

- 导出目录：`snapshots/ctdp_train__78cde1be/`；262 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T17:52:11+08:00；test(ctdp): isolate single-process CUDA witness logging。
- [configs/adatad/thumos/duca_ctdp_geometry_g0.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_geometry_g0.py)
- [configs/adatad/thumos/duca_ctdp_geometry_g1.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_geometry_g1.py)
- [configs/adatad/thumos/duca_ctdp_geometry_g2.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_geometry_g2.py)
- [configs/adatad/thumos/duca_ctdp_geometry_g3.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_geometry_g3.py)
- [configs/adatad/thumos/duca_ctdp_mechanism_m00.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_mechanism_m00.py)
- [configs/adatad/thumos/duca_ctdp_mechanism_m01.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_mechanism_m01.py)
- [configs/adatad/thumos/duca_ctdp_mechanism_m10.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_mechanism_m10.py)
- [configs/adatad/thumos/duca_ctdp_mechanism_m11.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/configs/adatad/thumos/duca_ctdp_mechanism_m11.py)
- [opentad/models/selectors/dual_phase_frame_selector.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/opentad/models/selectors/dual_phase_frame_selector.py)
- [opentad/models/dense_heads/anchor_free_head.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/opentad/models/dense_heads/anchor_free_head.py)
- [opentad/models/utils/post_processing/utils.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/opentad/models/utils/post_processing/utils.py)
- [tools/bata/ctdp_training.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/78cde1be1cb8b3acc7d750afc92ea740f2a03d06/tools/bata/ctdp_training.py)

### ctdp_evaluator / 为 78cde1be 绑定的独立评测器

- 导出目录：`snapshots/ctdp_evaluator__11ced13a/`；36 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T21:17:36+08:00；feat(ctdp): evaluate frozen terminal EMA with verified update counts。
- [tools/bata/ctdp_terminal_receipt.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11ced13ac6b72091d26405c5d6f152f09914c22d/tools/bata/ctdp_terminal_receipt.py)
- [scripts/run_ctdp_terminal_eval_n16r4.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/11ced13ac6b72091d26405c5d6f152f09914c22d/scripts/run_ctdp_terminal_eval_n16r4.sbatch)

### unified_implementation / 当前系统消融实现，缺失机制仍阻塞正式矩阵

- 导出目录：`snapshots/unified_implementation__793c4f9c/`；298 个源码/配置/测试文件。
- 源码提交时间：2026-09-03T14:44:56+08:00；fix(unified): keep mechanism gates and make profiling optional。
- [docs/experiments/duca_unified_matrix_manifest.yaml](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/docs/experiments/duca_unified_matrix_manifest.yaml)
- [tools/bata/generate_duca_unified_fullmatrix.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/tools/bata/generate_duca_unified_fullmatrix.py)
- [scripts/duca_unified_fullmatrix/matrix.json](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/scripts/duca_unified_fullmatrix/matrix.json)
- [opentad/models/duca/acquisition.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/opentad/models/duca/acquisition.py)
- [opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py)
- [opentad/models/bricks/scale_adaptive_conv1d.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/793c4f9cdf7dac4f224bc73012aff8bc93949f87/opentad/models/bricks/scale_adaptive_conv1d.py)

### bafdr_train / 当前 6000 successful-update 修复训练实现

- 导出目录：`snapshots/bafdr_train__710ce8a6/`；295 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T16:05:29+08:00；Fix BA-FDR successful optimizer update accounting and replay。
- [configs/adatad/thumos/bafdr_k16_full_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/configs/adatad/thumos/bafdr_k16_full_seed4407.py)
- [tools/bata/bafdr_k16_fullmatrix_train.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/tools/bata/bafdr_k16_fullmatrix_train.py)
- [tools/bata/bafdr_k16_fullmatrix.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/tools/bata/bafdr_k16_fullmatrix.py)
- [tools/bata/bafdr_training_contract.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/tools/bata/bafdr_training_contract.py)
- [opentad/datasets/transforms/bafdr.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/opentad/datasets/transforms/bafdr.py)
- [opentad/models/backbones/bafdr_wrapper.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/opentad/models/backbones/bafdr_wrapper.py)
- [opentad/models/projections/bafdr_asymmetric_proj.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/710ce8a6246c471742c83bf7c180d9ab87c36fac/opentad/models/projections/bafdr_asymmetric_proj.py)

### bafdr_old_receipt / 旧结果独立补封存；不能使 5994-5997 次更新成为合格 6000

- 导出目录：`snapshots/bafdr_old_receipt__84f1f035/`；58 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T12:03:46+08:00；Add retrospective BAFDR receipt sealing without changing evaluation provenance。
- [tools/bata/seal_bafdr_receipt.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/84f1f0356766ed9e695ff55244b0a1ad429cd84c/tools/bata/seal_bafdr_receipt.py)
- [tools/bata/bafdr_k16_fullmatrix_train.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/84f1f0356766ed9e695ff55244b0a1ad429cd84c/tools/bata/bafdr_k16_fullmatrix_train.py)

### ettrc_train / OFF/ON 已出分的正式训练实现

- 导出目录：`snapshots/ettrc_train__74473c27/`；260 个源码/配置/测试文件。
- 源码提交时间：2026-09-03T14:41:15+08:00；fix(et-trc): gate formal pair on real two-GPU update。
- [configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/74473c2775caebf0da9d368ce8009d78e2942098/configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py)
- [configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/74473c2775caebf0da9d368ce8009d78e2942098/configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_off_seed4407.py)
- [configs/adatad/thumos/continuous_roi_s2_v3_d160_seed4407.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/74473c2775caebf0da9d368ce8009d78e2942098/configs/adatad/thumos/continuous_roi_s2_v3_d160_seed4407.py)
- [opentad/models/backbones/et_trc_videomae.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/74473c2775caebf0da9d368ce8009d78e2942098/opentad/models/backbones/et_trc_videomae.py)
- [opentad/models/backbones/backbone_wrapper.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/74473c2775caebf0da9d368ce8009d78e2942098/opentad/models/backbones/backbone_wrapper.py)

### ettrc_evaluator / 当前独立终态 EMA 评测器；不是新训练模型

- 导出目录：`snapshots/ettrc_evaluator__67d7079d/`；53 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T13:07:17+08:00；Write independent ETTRC terminal EMA official evaluation receipts。
- [tools/bata/ettrc_terminal_receipt.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/67d7079d0d1c33e129d31cd3a45aaf93a67db252/tools/bata/ettrc_terminal_receipt.py)
- [scripts/run_ettrc_terminal_eval_n16r4.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/67d7079d0d1c33e129d31cd3a45aaf93a67db252/scripts/run_ettrc_terminal_eval_n16r4.sbatch)

### evidence_resume / 当前存储迁移/精确恢复实现；需保留原始训练来源

- 导出目录：`snapshots/evidence_resume__ce767b4d/`；267 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T19:42:35+08:00；Preserve original workdir spelling in Evidence resume binding。
- [configs/adatad/thumos/duca_evidence_recovery_base.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/configs/adatad/thumos/duca_evidence_recovery_base.py)
- [configs/adatad/thumos/duca_evidence_recovery_full.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/configs/adatad/thumos/duca_evidence_recovery_full.py)
- [opentad/models/duca/evidence_recovery.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/opentad/models/duca/evidence_recovery.py)
- [opentad/models/bricks/dense_temporal_recovery.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/opentad/models/bricks/dense_temporal_recovery.py)
- [opentad/models/bricks/bounded_interval_adapter.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/opentad/models/bricks/bounded_interval_adapter.py)
- [opentad/models/bricks/temporal_token_merge.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/opentad/models/bricks/temporal_token_merge.py)
- [scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)
- [scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch)
- [tools/train.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/tools/train.py)
- [tools/test.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/ce767b4df49a82b42f4fab4897ac4ea562e4b548/tools/test.py)

### evidence_optimizer_source / A1/A6 恢复前训练来源

- 导出目录：`snapshots/evidence_optimizer_source__1570a725/`；56 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T11:49:44+08:00；Isolate injected time-gradient overflow in CUDA test。
- [opentad/cores/optimizer.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1570a72507491899a50767700a35a04eee3f5fe9/opentad/cores/optimizer.py)
- [tools/train.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/1570a72507491899a50767700a35a04eee3f5fe9/tools/train.py)

### evidence_remaining_source / F/A3/A4/A5 恢复前训练来源

- 导出目录：`snapshots/evidence_remaining_source__73bdd34a/`；57 个源码/配置/测试文件。
- 源码提交时间：2026-09-07T12:54:46+08:00；Add real-data optimizer prechecks for remaining Evidence arms。
- [tools/train.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/73bdd34ae21c6c675a00d1927b8af7c12f9edc05/tools/train.py)
- [scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/73bdd34ae21c6c675a00d1927b8af7c12f9edc05/scripts/run_duca_evidence_recovery_train_array_n16r4.sbatch)

### evidence_published_results / C0/A2 已出分来源，不能归到最新恢复代码

- 导出目录：`snapshots/evidence_published_results__246058f2/`；46 个源码/配置/测试文件。
- 源码提交时间：2026-09-04T03:00:58+08:00；fix(evidence): validate normalized ledger positions。
- [configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/246058f2c24edc78818ada60eec26249bbf7d5d2/configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py)
- [configs/adatad/thumos/duca_evidence_recovery_no_time.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/246058f2c24edc78818ada60eec26249bbf7d5d2/configs/adatad/thumos/duca_evidence_recovery_no_time.py)
- [tools/test.py](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/246058f2c24edc78818ada60eec26249bbf7d5d2/tools/test.py)

## 原冻结分支不是最新运行版本

| 路线 | 原冻结分支 | 原审计SHA |
|---|---|---|
| H65-Pro：语义相位选帧、时间定位及因素消融 | [codex/h65-pro-fullmatrix-strict60-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/h65-pro-fullmatrix-strict60-20260902) | [cfb7041d876f6e38e9ef6ce77cef7cee04b79659](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/cfb7041d876f6e38e9ef6ce77cef7cee04b79659) |
| CT-DP：非均匀时间嵌入、检测几何与稀疏骨干路由 | [codex/duca-ctdp-geometry-mechanism-correction-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-ctdp-geometry-mechanism-correction-20260902) | [2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc) |
| DUCA-Unified：manifest 驱动的系统因素控制实验 | [codex/duca-unified-fullmatrix-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-unified-fullmatrix-20260902) | [89b9ea3e8e018b41034917ee14de7f409354a7e9](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/89b9ea3e8e018b41034917ee14de7f409354a7e9) |
| BAFDR：全局低分辨率载体、局部块路由和教师蒸馏 | [codex/zoomtoken-bafdr-gradient-correction-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-gradient-correction-20260902) | [fdeaeb98340bf7070201a02feb8093f50486aeaa](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdeaeb98340bf7070201a02feb8093f50486aeaa) |
| ET-TRC：保持密集状态的 Transformer 局部残差近似 | [codex/zoomtoken-et-trc-correction-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-et-trc-correction-20260902) | [59eab0c6aaacf5039d2ae20969a6dd5772bcb80f](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59eab0c6aaacf5039d2ae20969a6dd5772bcb80f) |
| Evidence-Recovery：预选阶段语义侦察、空洞控制与证据恢复 | [codex/duca-evidence-recovery-numerical-correction-20260902](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-evidence-recovery-numerical-correction-20260902) | [08d425a259fc468dde7c496e77b4c43e953d8d0c](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08d425a259fc468dde7c496e77b4c43e953d8d0c) |

## 容易误用的身份

- 原单分支集成远端仍是 `076f8e9821140e4ad90cc60961f484555a8e6bdf`，本地主目录为 `dd0efcbd466c368e16c9dbcb1779eedb1711ed2c`，tracked clean=False；两者不相同。本任务没有推送、回退或采纳其本地独有修改。
- ET原correction分支已到e173f84e，但formal-repair 74473c27才是已报告OFF/ON的训练身份；e173与744的ET模型/主配置没有差异。863c0e39是更早的诊断脚本提交，不是替代正式训练的最新算法。
- Evidence的evidence_eval_repair文件夹实际HEAD是9739811e（A6数值修复），不能只凭目录名将它冒充246058f2的C0/A2来源。
- GitHub其余同名/旧路线分支列在02_CODE_MANIFEST.json的related_github_heads；它们是来源与历史参考，不自动成为本次负责人新增负责的实验。

## 附件

- 00_PRO_REVIEW_PROMPT.md：可直接提供给Pro的中文任务，官方基线身份列为首要问题。
- 01_CODE_INDEX.md / 02_CODE_MANIFEST.json：代码身份、GitHub精确链接和本地位置。
- 本地源码包：E:/DeskTop/TAD/_pro_review_packets/DUCA_SIX_ROUTES_20260908/DUCA_SIX_ROUTES_PRO_REVIEW_20260908.zip。
- 包内03_SOURCE_FILE_LIST.json：逐快照实际导出的完整文件列表。
- context_prior_reports/：协调提交327df305中的基线和低性能诊断、实验目录，均为待独立复核材料；无checkpoint/原始服务器日志。
- 未调用Pro、未宣称Pro已审阅、未修改模型/部署或结果。
