# Source Registry

## 2026-07-20 Decode-Cross Failed Deployment Source

- `SRC-PT-017`: runtime commit
  `9bbc6eadf85dd65364223da719d13dd5b3789dda` / tree
  `68b5cc3f68ec1dfedbba82ac1421bf89d88b88d8`，run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_decode_cross_9bbc6ea_20260720_153600_0800_6f75b261e21d4626a7399a248afd6aee`。
  Gate `1175739` 在真实四条件推理前因 focused test 的可选 `solver`
  配置读取缺陷失败；`1175740–1175744` 未启动并取消。原始 gate 日志为
  `slurm_logs/pt_dc_gate_1175739.out/.err`，`jobs.tsv` 保留六个
  token/comment/Job ID。该来源只证明配置兼容性与 Slurm 多依赖规范化缺陷，
  不是 gate、mAP 或方法有效性证据。

## 2026-07-20 Active Review Source

- `SRC-PT-016`: 外部 PhysTime `STOP-Q-LIFT` Pro 严审原文
  `docs/methods/reviews/2026-07-20-phystime-stop-q-lift-pro-review-raw.md`，
  SHA256
  `F08AF135EAC342960929031FE84400144F0ADA55720F9A744203CFF2943A5057`。
  审查对象为文档 commit `21c264b`、可执行 commit `0dc5851` /
  tree `bddc9b9` 及 full60 `41.28/57.57%`。完整独立吸收见
  `docs/methods/2026-07-20-phystime-stop-q-lift-pro-review-absorption.md`。
  用途：暂停训练型 Q-lift，固定 P0 全精度 NMS replay、Q192
  assignment/decode 机制分解、无训练 Q-density 反事实和发布级代码缺口；
  不是新实验、实现或 mAP 来源。原报告主效应公式标签互换的修正以本地
  吸收记录为准。

## 2026-07-19 Active Review Source

- `SRC-PT-015`: 外部 PhysTime Full60/Q-lift Pro 严审原文
  `docs/methods/reviews/2026-07-19-phystime-full60-q-lift-pro-review-raw.md`，
  SHA256
  `BBD48B6BCE5E4AC612A395561D2EABCBB1F6DB5880B329EF21CAC6808CFBD5E0`。
  审查对象为 commit `0dc5851` / tree `bddc9b9` 及
  `41.28/57.57%` full60 结果。完整独立吸收见
  `docs/methods/2026-07-19-phystime-full60-q-lift-pro-review-absorption.md`。
  用途：固定代码作用域、比较公平性、K/J/Q 因果缺口、全精度 NMS 修复、
  support-preserving query-lift 候选与下一轮四臂实验边界；不是新实验或
  新 mAP 来源。

## 2026-07-18 Active Source

- `SRC-PT-014`: commit `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`,
  tree `bddc9b9386604d00d213275a47ce7997b35d3f4c`, clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1_full60_0dc5851_20260718`,
  and run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800`.
  Gate `1170945` and jobs `1170946/1170947` completed `0:0`. Final epoch-59
  selected-axis/physical-metric Avg-mAP is `41.28/57.57%`; both independent
  completion validators pass. Checkpoint SHA256 is `6fd0781b...` /
  `c83a3463...`, and metrics JSON SHA256 is `526274c6...` / `f725e6ca...`;
  full hashes and IoU-wise metrics are in `docs/evaluation/results.md`.

本文件登记新一轮方法判断所直接依赖的原始来源；完整历史覆盖仍见 `source_map.md`。

| ID | 类型 | 来源 | 本轮用途 |
| --- | --- | --- | --- |
| SRC-PT-011 | G1b SDPQ P0 repair | commit to be created after this repair; remote test copy `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_p0fix_test_20260716_004648` | Evidence/assignment mask separation, zero-init query/coverage residuals, explicit offset loss, structured pilot artifact validation; remote focused tests `21 passed in 52.60s`; not a real gate or mAP source |
| SRC-PT-012 | G1b 20轮完成结果 | commit `4a57577193c07cc90ac0867176aa79c76f637c36`; run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_4a57577_gtboundaryfix_medium20_20260716_190900_0800` | 证明 medium-run trainability；预测 mAP 可重算，但旧轻量 checkpoint 缺 EMA，不能重放 evaluated weights |
| SRC-PT-013 | 三臂 matched medium 完成证据 | code commit `5e8a8219c27785c15d720c5ed3c6b37298a2a866`; tree `7dfdf3d1c1e1c681a5df23f5916e2aa53de221ea`; snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1_matched_5e8a821_20260717`; run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_5e8a821_medium20_20260717_132000_0800` | gate `1168484` 和 jobs `1168485..1168487` 全部 `COMPLETED 0:0`；selected/physical/G1b Avg-mAP=`30.42/44.88/30.88`；online/EMA checkpoint 与独立 evaluator 均通过 |
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
