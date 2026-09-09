# 六路线历史 Checkpoint 清理记录

用户授权：清理历史 checkpoints，仅保留最后一个可用参数文件。停止实验的指令继续有效，本次只进行 CPU 文件验证和存储维护。

## 完成情况

- 执行时间：2026-09-09 14:25:38 CST；独立目录复核：14:27:17 CST。
- 21 个明确归属本任务的输出目录，904 份权重，按 101 个独立运行/版本/实验臂/seed 分组。
- 删除 803 份旧权重，共 501,769,717,837 bytes，约 467.31 GiB（501.77 GB）。
- 每组保留按数字 epoch 排序的最后一份可加载权重，共 101 份。不是仅保留六个模型，也不把不同版本、失败运行或 PRECHECK 混成一个实验。
- 101 份保留文件全部通过 CPU torch.load 和 state_dict/state_dict_ema 全参数有限值检查；无损坏候选，无回退。保留文件未裁剪、重写或移动，原有 EMA/optimizer/恢复状态仍在。
- 清理后重新扫描：101 份权重全部存在；803 份删除目标均不存在；没有额外权重、checkpoint 临时文件或读取错误。
- /data 可用空间从清点时的 34.68 GB 增至 507.36 GB（472.51 GiB），使用率从100%降至92%。共享盘仍有其他任务写入，因此净增可用空间不等于删除文件字节总量。
- 本地相关工作树没有 .pth/.pt/.ckpt 训练权重，未进行本地权重删除。

## 按路线汇总

| 路线（对外名称） | 保留文件 | 删除文件 | 删除体积 GiB |
|---|---:|---:|---:|
| 预选阶段证据补漏 (Evidence-Recovery) | 34 | 198 | 115.66 |
| 分块路由与蒸馏 (BAFDR) | 11 | 121 | 71.46 |
| 物理时间嵌入与稀疏层路由 (CT-DP) | 23 | 296 | 171.97 |
| Transformer 内部低秩近似 (ET-TRC) | 10 | 24 | 13.26 |
| 四相选帧与物理时间定位 (H65-Pro) | 23 | 164 | 94.97 |
| 系统因素消融（DUCA-Unified） | 0 | 0 | 0 |

DUCA-Unified 尚无完整正式模型；其旧 single-seed 目录中的 CT-DP G0 权重按实际模型归入 CT-DP，不算新的统一路线结果。上述101份包含历史版本、失败后的最后快照和预检快照，不代表101个有效正式性能结果。

## 保留与边界

- JSON sidecar、训练审计、配置、stdout/stderr、预测与评测 receipts 全部不删不改。原来的 source/config 身份和历史分数不变。
- 5 份 best_test.pth 独立副本已按“只留最后一个”删除。较早最佳 epoch 的4组权重已不再保留，不能再承诺直接重跑其 best checkpoint；最佳分数和完整曲线仍作为历史记录保存。相位关闭 H65 的最佳 epoch 就是最后一轮，其 epoch59 文件仍保留。
- BAFDR 两个已纳入本任务的 D160 教师运行均保留其 epoch59 checkpoint；教师配置和元数据保留。Evidence 恢复来源运行也各自保留最后快照；resume lineage 与审计不改写。
- 历史 BAFDR 独立任务（含1267920/1267921）、另任务 H65 matched90、GeoSparse、BCR、原始官方 AdaTAD、外部预训练权重及其他未明确归属本任务的旧路线均未清理。
- 可加载且参数有限只表示文件可用，不修复既有实现/协议问题，也不将失败、PRECHECK 或不合格训练升级为最终性能。
- 没有训练、GPU评测、PRECHECK、重新提交或重启监督器。末次队列仅见其他项目的1278774、1281582、1281445。

## 完整记录

- [清理范围](50_CHECKPOINT_CLEANUP_SCOPE_20260909.json)
- [已执行清单，含全部保留和删除路径](50_CHECKPOINT_CLEANUP_20260909.json)
- [删除后独立目录与存储复核](50_CHECKPOINT_CLEANUP_VERIFICATION_20260909.json)
- 远端维护记录：`/data/run01/sczc063/yuzibo/maintenance/checkpoint_cleanup_20260909/`。
- 本地 focused tests：34 passed；py_compile通过。维护脚本与4项回归测试均在协调分支，模型代码未修改。

## 保留文件列表

epoch 为文件内零基编号；epoch59 对应完成60轮，低于59的历史/预检文件不等于完整正式训练。

| # | 保留的完整远端路径 | epoch |
|---:|---|---:|
| 1 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A1/seed_8261/gpu1_id0/checkpoint/epoch_24.pth` | 24 |
| 2 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A2/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 3 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A3/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 4 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A4/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 5 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A5/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 6 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/A6/seed_8261/gpu1_id0/checkpoint/epoch_49.pth` | 49 |
| 7 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/C0/seed_8261/gpu1_id0/checkpoint/epoch_49.pth` | 49 |
| 8 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/F/seed_8261/gpu1_id0/checkpoint/epoch_9.pth` | 9 |
| 9 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/diagnostic_A6_20260902_153411_1265656/gpu1_id0/checkpoint/epoch_0.pth` | 0 |
| 10 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/diagnostic_F_20260902_152341_1265634/gpu1_id0/checkpoint/epoch_10.pth` | 10 |
| 11 | `/data/run01/sczc063/yuzibo/duca_evidence_recovery_single_seed_8261_647151fa_20260902_092503/diagnostic_F_20260902_160223_1265679/gpu1_id0/checkpoint/epoch_10.pth` | 10 |
| 12 | `/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_d160_seed4407/checkpoint/epoch_59.pth` | 59 |
| 13 | `/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_g96_seed4407/checkpoint/epoch_59.pth` | 59 |
| 14 | `/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_late_seed4407/checkpoint/epoch_59.pth` | 59 |
| 15 | `/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_nokd_seed4407/checkpoint/epoch_59.pth` | 59 |
| 16 | `/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_u16_uniform_a0_seed4407/checkpoint/epoch_59.pth` | 59 |
| 17 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal/duca_ctdp_geometry_g0_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 18 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal/duca_ctdp_geometry_g1_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 19 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck/duca_ctdp_geometry_g0_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 20 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck/duca_ctdp_geometry_g1_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 21 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck/duca_ctdp_geometry_g2_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 22 | `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck/duca_ctdp_geometry_g3_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 23 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/formal/duca_ctdp_geometry_g0_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 24 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/formal/duca_ctdp_geometry_g1_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 25 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/formal/duca_ctdp_geometry_g2_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 26 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/formal/duca_ctdp_geometry_g3_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 27 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/precheck/duca_ctdp_geometry_g0_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 28 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/precheck/duca_ctdp_geometry_g1_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 29 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/precheck/duca_ctdp_geometry_g2_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 30 | `/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/precheck/duca_ctdp_geometry_g3_seed3407_precheck/gpu1_id0/checkpoint/epoch_1.pth` | 1 |
| 31 | `/data/run01/sczc063/yuzibo/experiments/duca_ctdp_c0fae67a_seed3407/geometry_g0_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 32 | `/data/run01/sczc063/yuzibo/experiments/duca_ctdp_c0fae67a_seed3407/geometry_g1_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 33 | `/data/run01/sczc063/yuzibo/experiments/duca_ctdp_c0fae67a_seed3407/geometry_g2_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 34 | `/data/run01/sczc063/yuzibo/experiments/duca_ctdp_c0fae67a_seed3407/geometry_g3_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 35 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_77c8d173_seed8261/C0/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 36 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/A1/seed_8261/gpu1_id0/checkpoint/epoch_49.pth` | 49 |
| 37 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/A2/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 38 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/A3/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 39 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/A4/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 40 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/A5/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 41 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/C0/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 42 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2/F/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 43 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/formal/A1/seed_8261/gpu1_id0/checkpoint/epoch_39.pth` | 39 |
| 44 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/formal/A6/seed_8261/gpu1_id0/checkpoint/epoch_39.pth` | 39 |
| 45 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a/formal/A3/seed_8261/gpu1_id0/checkpoint/epoch_29.pth` | 29 |
| 46 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a/formal/A4/seed_8261/gpu1_id0/checkpoint/epoch_29.pth` | 29 |
| 47 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a/formal/A5/seed_8261/gpu1_id0/checkpoint/epoch_29.pth` | 29 |
| 48 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a/formal/F/seed_8261/gpu1_id0/checkpoint/epoch_29.pth` | 29 |
| 49 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/formal/A1/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 50 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/formal/A6/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 51 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/formal/F/seed_8261/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 52 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/A1/seed_8261/gpu1_id0/checkpoint/epoch_40.pth` | 40 |
| 53 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/A3/seed_8261/gpu1_id0/checkpoint/epoch_30.pth` | 30 |
| 54 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/A4/seed_8261/gpu1_id0/checkpoint/epoch_30.pth` | 30 |
| 55 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/A5/seed_8261/gpu1_id0/checkpoint/epoch_30.pth` | 30 |
| 56 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/A6/seed_8261/gpu1_id0/checkpoint/epoch_40.pth` | 40 |
| 57 | `/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/precheck/F/seed_8261/gpu1_id0/checkpoint/epoch_30.pth` | 30 |
| 58 | `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/off/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 59 | `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/on/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 60 | `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/precheck/off/gpu2_id0/checkpoint/epoch_0.pth` | 0 |
| 61 | `/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/precheck/on/gpu2_id0/checkpoint/epoch_0.pth` | 0 |
| 62 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/F01_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 63 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/F02_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 64 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/F03_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 65 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/F04_seed3407/gpu1_id0/checkpoint/epoch_14.pth` | 14 |
| 66 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/F05_seed3407/gpu1_id0/checkpoint/epoch_14.pth` | 14 |
| 67 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/REF-D768_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 68 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/REF-MNV3FC384_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 69 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_fullmatrix_e553a5a4/REF-U384_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 70 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F01_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 71 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F02_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 72 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F03_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 73 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F04_seed3407/gpu1_id0/checkpoint/epoch_19.pth` | 19 |
| 74 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F05_seed3407/gpu1_id0/checkpoint/epoch_14.pth` | 14 |
| 75 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803/F06_seed3407/gpu1_id0/checkpoint/epoch_14.pth` | 14 |
| 76 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F01_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 77 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F02_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 78 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F03_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 79 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F04_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 80 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F05_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 81 | `/data/run01/sczc063/yuzibo/experiments/h65_pro_physical_optimizer_f2068e18/F06_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 82 | `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEOFF_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 83 | `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEON_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 84 | `/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-UNIFORM_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 85 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_d160_seed4407/checkpoint/epoch_59.pth` | 59 |
| 86 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_full_seed4407/checkpoint/epoch_59.pth` | 59 |
| 87 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_g96_seed4407/checkpoint/epoch_59.pth` | 59 |
| 88 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_late_seed4407/checkpoint/epoch_59.pth` | 59 |
| 89 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_nokd_seed4407/checkpoint/epoch_59.pth` | 59 |
| 90 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407/work_dirs/bafdr_k16_u16_uniform_a0_seed4407/checkpoint/epoch_59.pth` | 59 |
| 91 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_ettrc_74473c27_seed4407/admission/gpu2_id0/checkpoint/epoch_0.pth` | 0 |
| 92 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_ettrc_74473c27_seed4407/off/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 93 | `/data/run01/sczc063/yuzibo/experiments/zoomtoken_ettrc_74473c27_seed4407/on/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 94 | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902/exps/thumos/adatad/duca_ctdp_geometry_g0_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 95 | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902/exps/thumos/adatad/duca_ctdp_geometry_g1_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 96 | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902/exps/thumos/adatad/duca_ctdp_geometry_g2_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 97 | `/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902/exps/thumos/adatad/duca_ctdp_geometry_g3_seed3407/gpu1_id0/checkpoint/epoch_59.pth` | 59 |
| 98 | `/data/run01/sczc063/yuzibo/projects/duca_unified_single_seed_20260903/exps/thumos/adatad/duca_ctdp_geometry_g0_seed3407/gpu1_id0/checkpoint/epoch_13.pth` | 13 |
| 99 | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902/exps/thumos/adatad/et_trc_videomae_s_768x1_160_adapter_off_seed4407/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 100 | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_fix_be330c07/off/gpu2_id0/checkpoint/epoch_59.pth` | 59 |
| 101 | `/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_fix_be330c07/on/gpu2_id0/checkpoint/epoch_59.pth` | 59 |

