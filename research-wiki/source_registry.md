# Source Registry

本文件登记新一轮方法判断所直接依赖的原始来源；完整历史覆盖仍见 `source_map.md`。

| ID | 类型 | 来源 | 本轮用途 |
| --- | --- | --- | --- |
| SRC-PT-011 | G1b SDPQ P0 repair | commit to be created after this repair; remote test copy `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_p0fix_test_20260716_004648` | Evidence/assignment mask separation, zero-init query/coverage residuals, explicit offset loss, structured pilot artifact validation; remote focused tests `21 passed in 52.60s`; not a real gate or mAP source |
| SRC-PT-010 | 独立 Max 代码审查 | `research-wiki/reviews/2026-07-13-phystime-g1a-max-code-review.md` | 两轮逐行审查、P1/P2 修复、测试证据与部署门槛 |
| SRC-PT-001 | 外部审查原文 | `docs/methods/reviews/2026-07-13-phystime-performance-drop-pro-audit-response-raw.md`，SHA256 `651C4CA673073D7E4C05746138C82EBBE2E6174C459516FB40B3EFDCA47305AB` | `HOLD AND REBUILD`、SM-PTAF 与 P0 gates |
| SRC-PT-002 | 正式结果 | `docs/evaluation/results.md` | 三头 best-checkpoint mAP、容量、候选与预测分解 |
| SRC-PT-003 | 完整性审计 | `docs/evaluation/EXPERIMENT_AUDIT.md` | real GT、official mAP、证据范围与剩余 WARN |
| SRC-PT-004 | 远端原始作业 | run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800` | jobs `1159491..1159495` 与复算 `1159819..1159821` 状态核验 |
| SRC-PT-005 | 正式代码 | commit `3ac93a12c299012db64513567d5bdedf0c6d5f71` | raw-video K384 三头实现 |
| SRC-PT-006 | 预部署工程验证 | remote scratch `/data/run01/sczc063/yuzibo/projects/scratch/phystime_g1a_green_20260713` | G1a focused/new-old regression `116 passed`；411-video timebase audit；不是正式实验或 mAP 来源 |
| SRC-PT-007 | 失败 gate 原始作业 | run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_8e2b832_pilot_20260713_161418_+0800` | gate `1161304` 因两个未引用 test MP4 缺 annotation 而 fail-closed；pilots `1161305/1161306` 未启动并取消；仅作审计范围修复证据 |
| SRC-PT-008 | 失败 gate 原始作业 | run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_e598bd7_pilot_20260713_162840_+0800` | gate `1161353` 因 scalar state byte-view 兼容性失败；pilots `1161354/1161355` 未启动并取消；仅作状态摘要修复证据 |
| SRC-PT-009 | 失败 gate 原始作业 | run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_d193417_pilot_20260713_164152_+0800` | gate `1161378` 在 selected-axis 首个真实样本因旧逐步 `regression_gradient` 非零合同 fail-closed；pilots `1161379/1161380` 未启动并取消；用于三步聚合梯度合同与 assignment 诊断修复 |
| SRC-LIT-001 | 论文 | `https://arxiv.org/abs/2101.10318` | mTAN，不规则时间 attention 先例 |
| SRC-LIT-002 | 论文 | `https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html` | actual-time TAD 坐标先例 |
| SRC-LIT-003 | 论文 | `https://openaccess.thecvf.com/content/CVPR2022/html/Wang_RCL_Recurrent_Continuous_Localization_for_Temporal_Action_Detection_CVPR_2022_paper.html` | 连续锚表示先例 |
| SRC-LIT-004 | 论文 | `https://arxiv.org/abs/2403.20254` | 缺帧/时序扰动与定位鲁棒性先例 |
| SRC-LIT-005 | 论文 | `https://arxiv.org/abs/2604.18274` | continuous-dynamics TAD 新颖性边界 |
