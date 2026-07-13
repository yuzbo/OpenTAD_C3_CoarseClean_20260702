# Source Registry

本文件登记新一轮方法判断所直接依赖的原始来源；完整历史覆盖仍见 `source_map.md`。

| ID | 类型 | 来源 | 本轮用途 |
| --- | --- | --- | --- |
| SRC-PT-001 | 外部审查原文 | `docs/methods/reviews/2026-07-13-phystime-performance-drop-pro-audit-response-raw.md`，SHA256 `651C4CA673073D7E4C05746138C82EBBE2E6174C459516FB40B3EFDCA47305AB` | `HOLD AND REBUILD`、SM-PTAF 与 P0 gates |
| SRC-PT-002 | 正式结果 | `docs/evaluation/results.md` | 三头 best-checkpoint mAP、容量、候选与预测分解 |
| SRC-PT-003 | 完整性审计 | `docs/evaluation/EXPERIMENT_AUDIT.md` | real GT、official mAP、证据范围与剩余 WARN |
| SRC-PT-004 | 远端原始作业 | run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800` | jobs `1159491..1159495` 与复算 `1159819..1159821` 状态核验 |
| SRC-PT-005 | 正式代码 | commit `3ac93a12c299012db64513567d5bdedf0c6d5f71` | raw-video K384 三头实现 |
| SRC-PT-006 | 预部署工程验证 | remote scratch `/data/run01/sczc063/yuzibo/projects/scratch/phystime_g1a_green_20260713` | G1a focused/new-old regression `116 passed`；411-video timebase audit；不是正式实验或 mAP 来源 |
| SRC-LIT-001 | 论文 | `https://arxiv.org/abs/2101.10318` | mTAN，不规则时间 attention 先例 |
| SRC-LIT-002 | 论文 | `https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html` | actual-time TAD 坐标先例 |
| SRC-LIT-003 | 论文 | `https://openaccess.thecvf.com/content/CVPR2022/html/Wang_RCL_Recurrent_Continuous_Localization_for_Temporal_Action_Detection_CVPR_2022_paper.html` | 连续锚表示先例 |
| SRC-LIT-004 | 论文 | `https://arxiv.org/abs/2403.20254` | 缺帧/时序扰动与定位鲁棒性先例 |
| SRC-LIT-005 | 论文 | `https://arxiv.org/abs/2604.18274` | continuous-dynamics TAD 新颖性边界 |
