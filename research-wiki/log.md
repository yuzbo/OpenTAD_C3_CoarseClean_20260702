# 2026-07-18 PhysTime G1 matched full60

- User-authorized survivor validation was implemented at commit `0dc5851`.
- The suite fixes K384/J192, seed 42, no feature interpolation, exact
  scheduler/workflow length 60, final-only online+EMA checkpointing, an 8 GiB
  free-space gate, and independent epoch-59 metric recomputation.
- Local no-Torch deployment tests passed `6`; remote Linux focused tests passed
  `54`. The known local Torch `c10.dll` issue remains environmental.
- Clean snapshot and run root are registered as `SRC-PT-014`.
- Real gate `1170945` and selected-axis/physical-metric jobs
  `1170946/1170947` all completed `0:0`.
- First matched epoch-41 validation: selected-axis Avg-mAP `39.84%`
  (`64.54/54.50/40.78/26.53/12.87` at tIoU 0.3:0.7), physical-metric
  `57.32%` (`77.57/71.41/61.44/48.00/28.17`), a `+17.48` Avg-mAP
  interim delta. This is still `experiment_running`; epoch 59, final
  online/EMA checkpoint validation, independent recomputation, and both
  completion artifacts remain pending.
- Second matched validation at epoch 43 remained consistent: selected-axis
  `40.41%`, physical-metric `57.19%`, interim delta `+16.78` Avg-mAP.
  Both jobs continue from epoch 44 without logged anomalies.
- Epochs 45 and 47 retained the ordering. Latest epoch-47 Avg-mAP is
  selected-axis `40.81%`, physical-metric `57.48%`, delta `+16.66`;
  mAP@0.7 is `14.22/27.99`, delta `+13.77`. Both jobs continue from epoch 48
  with no logged anomaly. This remains interim evidence.
- Epoch 49 again retained the ordering: selected-axis `41.16%`,
  physical-metric `57.54%`, delta `+16.37` Avg-mAP. Physical-metric remains
  ahead at every IoU threshold, including mAP@0.7 `28.00%` versus `14.49%`.
  Both jobs continue beyond epoch 49 with zero matched anomaly hits. Epoch 59
  and completion/checkpoint validation remain pending.
- Epoch 51 produced the sixth consistent matched validation: selected-axis
  `41.24%`, physical-metric `57.44%`, delta `+16.19` Avg-mAP. The mAP@0.7
  values are `14.51/27.82`, delta `+13.31`. Both jobs remain active with
  finite losses and zero anomaly hits; terminal artifacts are still absent.
- Epoch 53 produced the seventh consistent matched validation: selected-axis
  `41.36%`, physical-metric `57.56%`, delta `+16.19` Avg-mAP. The mAP@0.7
  values are `14.63/27.94`, delta `+13.31`. The selected-axis arm completed
  epoch 54 while physical-metric finished epoch-53 evaluation; anomaly counts
  and terminal artifacts remain zero/absent.
- Epoch 55 produced the eighth consistent matched validation: selected-axis
  `41.44%`, physical-metric `57.66%`, delta `+16.22` Avg-mAP. The mAP@0.7
  values are `14.77/28.21`, delta `+13.44`. Selected-axis completed epoch 56;
  both jobs remain active with zero anomaly hits and no terminal artifacts.
- Epoch 57 produced the ninth consistent matched validation: selected-axis
  `41.37%`, physical-metric `57.65%`, delta `+16.28` Avg-mAP. The mAP@0.7
  values are `14.82/28.66`, delta `+13.84`. Selected-axis completed epoch 58
  and an unvalidated epoch-59 checkpoint path appeared, but neither completion
  marker exists; both jobs remain active with zero anomaly hits.
- Final epoch 59 completed with selected-axis `41.28%` and physical-metric
  `57.57%` Avg-mAP, delta `+16.29`. IoU-wise deltas at 0.3:0.7 are
  `+12.38/+14.10/+19.90/+21.30/+13.78`. Both independent completion validators
  pass; checkpoints contain 499 finite online and 499 finite EMA entries,
  exclude optimizer/scheduler state, and replay the evaluated weights. Best
  logged validation for both arms is epoch 55 (`41.44/57.66%`). All anomaly and
  GT-boundary audit counts are zero. Status advances to
  `full60-single-seed-supported`, not `paper_ready`.

# 2026-07-13 PhysTime G1a d1747d6 deployment gate

# 2026-07-16 PhysTime G1b SDPQ P0 repair

- Absorbed the `REVISE-BEFORE-FULL-TRAIN` review for commit `372fcbf58d1b2eb895b724f6f040458bde4d636e`. The decision is accepted as a blocking pre-full-train verdict: G1b SDPQ is runnable and gate-passed, but its previous pilot delta over G1a was too small and the implementation lacked clean evidence/assignment separation.
- Implemented the first P0 repair in the PhysTime worktree, not the DUCA/Spatial-Zoom tree: `domain_valid_mask`, `evidence_mask`, and `assignment_mask` are now distinct; projection records `coverage_ratio`; uncovered queries can remain valid candidates but are blocked from positive assignment when evidence is absent.
- Added query-geometry and coverage residual paths to `PhysTimeMeasureProjection` with zero-initialized final layers so the initial pooling contract is preserved while allowing query/support geometry to become learnable after training.
- Reworked SDPQ target/loss logic: center scale and width reference are separated, explicit center/log-width offset loss is added, assignment uses `assignment_mask`, and diagnostics now include positive-uncovered / low-coverage counts plus reservation collision counts.
- Hardened `run_phystime_g1b_sdpq_pilot_slurm.sh`: `PILOT_COMPLETE.json` is now built from structured `evaluation_metrics.json` and final `epoch_5.pth`, with finite metric checks and final-epoch validation instead of fragile train-log regex.
- Verification status: local `py_compile` passed; local pytest is still blocked by the known Windows Torch `c10.dll` loader issue. Remote clean-copy focused verification passed: `21 passed in 52.60s` on `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_p0fix_test_20260716_004648`.
- Current status after this entry is `implemented + remote-focused-tested`; real THUMOS gate, new pilot mAP, same-commit controls, and full train are still pending. No empirical claim is unlocked by this repair alone.
- Commit `698ee4be20b6c3ace4ab168047446fae0e3e9073` was pushed to `codex/phystime-performance-diagnosis-20260712`; clean remote snapshot is `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_698ee4b_20260716_p0fix`, tree `b985813ed37a094c882d06a1e9f99dec931c48e6`.
- New real THUMOS deployment root is `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_698ee4b_p0fix_20260716_005601_+0800`. Jobs: `1165745` G1b SDPQ gate, `1165746` G1b SDPQ 6-epoch pilot afterok. First check: gate `RUNNING`, pilot `PENDING (Dependency)`. Status is `running_gate`, not pilot-supported or paper-ready.
- Gate `1165745` completed successfully (`COMPLETED 0:0`, elapsed `00:01:55`) and released pilot `1165746`, which is now `RUNNING`. Status is `pilot_running`; mAP and pilot-supported status remain pending until `PILOT_COMPLETE.json` and final evaluation metrics exist.
- Pilot `1165746` failed at final checkpoint save, not from model NaN/OOM. It reached epoch 4 evaluation with Avg-mAP `10.17%` and mAP@0.7 `1.66%`, then failed at epoch 5 checkpoint write with `PytorchStreamWriter failed writing file`; residual `epoch_5.pth` is 0 bytes and `PILOT_COMPLETE.json` is absent. The 6-epoch pilot is therefore invalid as a completed artifact and too short for method judgment.
- Following the user's correction, G1b evaluation is being upgraded from short pilot to a 20-epoch medium run. Implemented checkpoint repair: atomic checkpoint writes, configurable `workflow.checkpoint_save_mode=lightweight`, no optimizer/scheduler/EMA in G1b medium checkpoint, dynamic final checkpoint/evaluation epoch validation instead of hard-coded `epoch_5.pth`. Remote focused verification passed on `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_medium_ckpt_test_20260716`: `15 passed in 53.05s`.
- Commit `d5a2136dfc4f9f9936ff2c0843cfc8e6768d4e42` was pushed and deployed from clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_d5a2136_20260716_medium20`, tree `28062d97cf4ab2605f3dacee6b3eb9f7375e3843`. New 20-epoch medium run root is `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_d5a2136_medium20_20260716_131046_+0800`; jobs are `1166413` gate and `1166414` 20-epoch pilot afterok. First queue check: gate `PENDING (Priority)`, pilot `PENDING (Dependency)`.
- 20-epoch medium pilot `1166414` failed during epoch 12 DataLoader, not from NaN/OOM/checkpoint save. Last valid metric was epoch 11 Avg-mAP `24.46%` / mAP@0.7 `5.39%`. Traceback root was `BuildPhysTimeRawFrameGeometry` raising `PhysTime ground truth lies outside the end-exclusive window domain`. This invalidates the run as a completed 20-epoch artifact, while preserving it as diagnostic evidence that SDPQ continued learning through epoch 11.
- Implemented the GT/window boundary repair requested by the user: `BuildPhysTimeRawFrameGeometry` now converts dense-window GT to seconds, clamps any overlapping segment to the physical window `[domain_start, domain_end]`, filters segments that become empty after clamping, keeps `gt_labels` aligned, and records `phystime_gt_boundary_audit` with `video_name`, dense/raw window coordinates, original/kept/clamped/filtered counts, overflow magnitudes, and affected indices. `Collect` now preserves this audit in metas. This is `implemented + remote-focused-tested`, not a completed rerun.
- Verification for the boundary repair: local `py_compile` passed; local pytest remains blocked by the known Windows Torch `c10.dll` loader issue. Remote patched-copy verification on `/data/run01/sczc063/yuzibo/projects/opentad_phystime_gt_boundary_repair_filetest_20260716_183220` passed `30 passed in 43.76s` for raw geometry, native tubelet geometry, G1a configs, and AdaTAD configs. A lightweight train-set repair scan was attempted but stopped because Decord/IO made it slow; future reruns can locate exact repaired samples through the new per-sample audit fields.
- Re-deployed the G1b SDPQ 20-epoch medium run after the GT/window boundary repair. Code commit `4a57577193c07cc90ac0867176aa79c76f637c36`, tree `2d9ae007b7d9cea179a9ec5e08a82bf01ef4cf4c`, clean remote snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_4a57577_20260716_gtboundaryfix`, run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_4a57577_gtboundaryfix_medium20_20260716_190900_0800`. Jobs: `1167109` real gate and `1167110` 20-epoch pilot afterok. First queue check: gate `RUNNING` on `g0017`, pilot `PENDING (Dependency)`. Status is `gate_running`, not medium-run-supported and not paper-ready.
- Gate `1167109` completed successfully (`COMPLETED 0:0`, elapsed `00:01:52`) and released pilot `1167110`, which is now `PENDING (Priority)`. Gate JSON confirms commit/tree match, `K_raw_observations=384`, `J_native_tubelet_tokens=192`, `feature_interpolation=false`, and `gt_without_assigned_query=0`. Status is `gate_passed_waiting_pilot`, not medium-run-supported and not paper-ready.

- Independent Max reviewer returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE` after the two P1 fixes: per-step optimizer-state parameter-name hash validation, and cross-arm recomputation of parameter/initial-state/optimizer-schema matches from `variants`.
- Commit `d1747d6657e185495b4db9eb491fd135d4b90360` was pushed to `codex/phystime-performance-diagnosis-20260712`; clean remote snapshot is `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_d1747d6_20260713_gate`, tree `2651bd30eda5b0e0960518da4060ccfc628b7a58`.
- Formal Slurm queue was submitted: `1161476` real gate, `1161477` selected-axis 6-epoch pilot afterok, `1161478` physical-metric 6-epoch pilot afterok. Current status at submission check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. This is `queued_for_gate`, not yet `experiment_running`, and there is still no new mAP.
- Gate `1161476` then failed before pilot start because the gate incorrectly required every production train sample to have `mask.sum()==384`. Static contract/G0 passed, but real AdaTAD training windows can be shorter/padded; K=384 is the decoded slot count, not a guarantee of 384 valid raw observations per sample. The fix keeps `decoded_frame_count==384`, records `production_train_raw_valid_counts`, and requires each valid count to satisfy `0 < count <= 384` with min/max consistency. Focused remote regression after the fix: gate contract `30 passed`, PhysTime/C3 physical-grid `243 passed`. The fix is under renewed Max review before any requeue.
- Renewed Max review found two more P1 issues before requeue: the gate checked `inputs.shape[2]` instead of six-dimensional time axis `shape[-3]`, and assignment valid-point validation still assumed `batch_size*378`. Both are fixed: `_decoded_temporal_length` now validates `[B,N,C,T,H,W]`, `assignment_valid_point_per_sample` is recorded by the head, and validator checks per-sample valid candidate ranges/sums. Remote regression after the final fix: gate contract `34 passed`, PhysTime/C3 physical-grid `247 passed`; Max returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`, P0/P1 none.
- Commit `56c7e98e54ba83eb32b84dbdbeb74c3b5698eca2` was pushed and deployed from clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_56c7e98_20260713_gate`, tree `d698d451edc165ff4ac6179181157646262002a9`. New run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_56c7e98_gatepilot_20260713_194728_+0800`. Jobs: `1161486` real gate, `1161487` selected-axis pilot afterok, `1161488` physical-metric pilot afterok. First queue check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. Status remains `queued_for_gate`; no gate pass and no mAP yet.
- Gate `1161486` then failed before pilot because `_selected_index_checksum` still required exactly 384 selected raw-frame indices; a real short production window had 269. Fix is local to G1a: variable-valid selected-index checksum accepts strictly increasing lengths in `(0,384]`, includes length in the digest, records `selected_index_lengths` and `tail_selected_index_length`, and validates them. Remote regression: gate contract `34 passed`, PhysTime/C3 physical-grid `247 passed`; Max returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`, P0/P1 none. This is still a gate-contract repair, not method evidence.
- Commit `49fa13c15bb0e4e58428af52598f031e77a69ec2` was pushed and deployed from clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_49fa13c_20260713_gate`, tree `0cd3fa6e376057aa38364fdd93a5121aca187d77`. Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_49fa13c_gatepilot_20260713_200044_+0800`. Jobs: `1161495` real gate, `1161496` selected-axis pilot afterok, `1161497` physical-metric pilot afterok. First queue check: gate `RUNNING`, pilots `PENDING (Dependency)`. No gate pass and no mAP yet.
- Gate `1161495` then failed because `_selected_index_checksum_g1a` used `np.asarray` without importing numpy. This was a coverage hole in the new variable-valid checksum helper. Fix: add `import numpy as np` and a direct regression test for length-269 selected indices plus non-increasing rejection. Remote regression: gate contract `35 passed`, PhysTime/C3 physical-grid `248 passed`; Max returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`, P0/P1 none.
- Commit `a4b7f1db0424966c9f9c5d4304a7619be59661db` was pushed and deployed from clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_a4b7f1d_20260713_gate`, tree `d164816c3ee12946ac40bd1e1446711146cfb1af`. Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_a4b7f1d_gatepilot_20260713_200922_+0800`. Jobs: `1161500` real gate, `1161501` selected-axis pilot afterok, `1161502` physical-metric pilot afterok. First queue check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. No gate pass and no mAP yet.
- Gate `1161500` completed successfully (`COMPLETED 0:0`) and produced `gate_pass=true`. Key evidence: K=384 decoded slots, J=192, Q0=192, Q_total=378, `feature_interpolation=false`, selected-axis and physical-metric each completed 3 optimizer steps, `production_train_raw_valid_counts=[384,384,269,384,316,384]`, `selected_index_lengths=[[384,384],[269,384],[316,384]]`, tail length 253. Pilots `1161501/1161502` are now `RUNNING`; mAP remains NA.
- Pilots `1161501/1161502` later failed at checkpoint save after epoch 2, not from model NaN or evaluator failure. Traceback root: `torch.save` raised `PytorchStreamWriter failed writing file ... file write failed` and `unexpected pos ...`; remote `/data` was 100% full, while each pilot was saving ~595MB checkpoints every epoch. Early epoch-1 evaluation existed but is not a completed-pilot result: selected-axis Avg-mAP `0.10%`, physical-metric Avg-mAP `0.10%`. Fix commit `623a376700c5781a3a54e3c6622ceb2ebc5ffc8e` changes G1a pilot checkpointing to final-only (`workflow.checkpoint_interval=${PILOT_EPOCHS}`), preserving the required final `epoch_5.pth` artifact while avoiding intermediate checkpoint bloat. Local checks: `py_compile` passed and `tests/test_phystime_g1a_deployment.py` `2 passed`; remote checks on clean snapshot passed `40 passed`. New snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_623a376_20260713_checkpointfix`, tree `5e63fdc4d99997137f4e1da691b319ab90533b24`; new run root `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_623a376_ckptfix_20260713_225354_+0800`; jobs: `1162048` gate running at first check, `1162049/1162050` pending afterok. Status returns to `queued/running_gate`; no completed pilot mAP yet.

# Research Wiki Log

- 2026-07-13：第三轮 Max 审查等待期间自查发现 manifest 中的 VideoMAE pretrained checkpoint 与 pilot `epoch_5.pth` 被测试夹具错误合并；已拆分二者并删除错误路径相等约束，completion 仍独立验证 epoch checkpoint 的 EMA/optimizer/scheduler。focused tests 保持 `65 passed`，必须以最新 diff 重新复审。
- 2026-07-13：独立 Max code review 第二轮发现 assignment 伪计数、optimizer state 覆盖、DataLoader `drop_last`、pilot artifact 传递信任等 4 个 P1，并补充固定参数集合、GPU batch 生命周期和显式 seed 风险。现已按测试先行修复；远端 gate/artifact `65 passed`，PhysTime/shared physical-grid `240 passed`。第三轮复审前禁止部署，状态保持 `tested`。

本文件只追加，不回写历史。

- 2026-07-11：初始化 research-wiki。
- 2026-07-11：清点 C3/PAction/GAS-VT、DUCA、MUST、X3D/SlowFast、PIVOT/ChronoTransport、PhysTime 的仓库文档、原始附件、提交历史和实验记录。
- 2026-07-11：建立当前方向、决策台账、时间线、经验禁区、gap map、idea catalog、experiment register 和 query pack。
- 2026-07-11：将 feature-token PhysTime 轨道标记为取消/诊断，将 PhysTime-AdaTAD K384 三头比较标记为当前唯一执行主线。
- 2026-07-11：明确秒坐标可转换回原视频帧号，但禁止 selected-rank GT/预测坐标。
- 2026-07-11：声明当前无 claim 实体；任何论文主张必须等待 matched full run 与 result-to-claim 审计。
- 2026-07-11：完成首轮 lint：31 个实体、10 个 gaps、48 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3348 字符。
- 2026-07-11：第二轮完整性审计发现并修正遗漏：ChronoTransport 在本地分支已实现到 `92029ea`，formal P3 science gate 为负且 15 commits 未推远端；DUCA 另有 `a5e1774` full-stack/structural audit 分支。
- 2026-07-11：新增 DUCA、ChronoTransport、PhysTime 三份完整路线档案、逐主题覆盖矩阵，以及 ResearchClaw 第二组 24 个候选 idea。
- 2026-07-11：迁入主任务用户侧完整导出与跨代理近期记录，固定 SHA256；新增 11-worktree 审计库存，防止单一 checkout 遗忘历史实现。
- 2026-07-11：第二轮 lint：36 个实体、10 个 gaps、55 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3351 字符。
- 2026-07-11：PhysTime-AdaTAD 1.0 在 `549bb81` 完成 raw-video K384 三头 matched pipeline、原帧 same-index 审计、one-step 梯度证明、真实 CUDA gate 工具及 gate-dependent 启动器；远端 focused suite `45 passed`。状态为 `tested`，真实 THUMOS gate、正式训练与 mAP 仍 pending。
- 2026-07-11：首次 raw-video gate `1158528` 在 Python/模型执行前因非登录 shell 无 `module` 命令以 127 退出；依赖训练 `1158529/1158530/1158531` 未启动并取消。分类为 infrastructure failure；GPU launchers 改为可选 module 初始化并新增回归测试，等待新 commit 重跑。
- 2026-07-11：第二次 gate `1158546` 的 matched validator 通过，但 submission 覆盖 Slurm GPU mask，导致模型构建前 `CUDA is not available`；依赖训练 `1158547/1158548/1158549` 未启动并取消。launcher 已改为 Slurm 内保留调度器 mask，专项测试通过，等待新 commit 重跑。
- 2026-07-11：第三次 gate `1158556` 通过 CUDA、真实 THUMOS decode 与 same-frame checksum，但 imgaug 独立 RNG 导致增强后像素不一致；模型未构建，依赖训练 `1158557/1158558/1158559` 未启动并取消。gate 已统一 Python/NumPy/Torch/imgaug/OpenCV seed 并新增确定性测试。
- 2026-07-11：第四次 gate `1158576` 与逐 transform 诊断 `1158591` 将剩余分叉定位到首次 ImgAug 构造改变 ColorJitter 的 NumPy 状态；加入增强库预热后，真实诊断 `1158614` 证明三头 decode、crop、ImgAug、ColorJitter、FormatShape 像素 hash 全部一致。仍需重跑完整 detector gate。
- 2026-07-11：FP32 real gate `1158636` 完成三头真实 raw-video forward/backward/inference 并通过全部梯度/optimizer contract；formal selected-axis `1158637` 启动，physical-grid `1158638` 因 torchrun rendezvous broken pipe 失败，PhysTime `1158639` 因 endpoint probability BCE 不兼容 AMP 失败。已作 event-logit BCE 等价修复并将 gate 升级为 AMP，等待同 commit 重排完整三头。
- 2026-07-11：最终实验 commit `bd27544` 的真实 AMP gate `1158668` 通过；三头 same raw-frame/input、optimizer 与梯度合同全部满足。formal jobs `1158669/1158670/1158671` 已解除依赖并进入 epoch 0，状态提升为 `experiment_running`，mAP 与 claim 仍为 pending。
- 2026-07-11：formal PhysTime `1158671` 在 epoch 0 第 50 步出现全 NaN，定位为未覆盖 logits 在 support-measure masked attention 中先 `exp` 后乘零。`0bbf0e9` 改为 mask-before-exp，新增极值回归测试，远端 `68 passed`；新 AMP gate `1158718` 通过，matched jobs `1158719/1158720/1158721` 已越过第 50 步且 loss 有限，mAP 仍 pending。
- 2026-07-12：`0bbf0e9` matched jobs `1158719/1158720/1158721` 全部 FAILED。selected-axis/physical-grid 训练至 epoch 41 后首次验证因 evaluator GT annotation 相对路径不存在而退出；PhysTime 从 epoch 1 step 99 起持续全 NaN，并叠加相同验证路径错误。三头均无有效 mAP，实验状态改为 `experiment_failed`，禁止使用 checkpoint 填表。
- 2026-07-12：`52b5756` 修复 evaluator 路由与物理时间 FP32 数值路径；gate `1159481` 通过，但 stability gate `1159482` 在正式作业启动前 fail-closed。诊断作业 `1159489` 将问题定位到 epoch 0 iter 47 的 `rpn_head.cls_head.weight`：forward loss 有限，11 个 scaled gradient 为 Inf，无 NaN。
- 2026-07-12：最终 commit `3ac93a1` 将 AMP 初始 scale 设为 1024，限制可恢复 Inf 跳步，关闭单 GPU FP16 DDP compression，并保留 NaN、参数污染与跳步超限硬失败。远端 `102 passed`；gate `1159491` 与两 epoch stability gate `1159492` 均通过且零跳步。formal jobs `1159493/1159494/1159495` 正在运行，mAP pending。
- 2026-07-12：formal jobs `1159493/1159494/1159495` 均越过 epoch 1 step 50，loss 分别为 0.9929、1.0115、1.1880，全部有限；这只提升训练稳定性证据，mAP 与方法 claim 仍 pending。
- 2026-07-12：`3ac93a1` 三头正式训练全部完成；最佳 checkpoint 复算 `1159819/1159820/1159821` 逐项复现官方结果。PhysTime 1.0 未胜两个 sparse controls，状态改为“负结果已验证”，不是 paper-ready。
- 2026-07-12：完成性能下降诊断：排除训练崩溃、evaluator、重复坐标换算与缺失 test window；确认比较存在容量/上下文混杂，并发现 absolute-second query 主导、粗层 attention 坍缩、候选密度和短动作监督不足、单标签 assignment 差异。完整数字只写入 `docs/evaluation/results.md`。
- 2026-07-12：独立 GPT-5.5 xhigh 完整性审计确认 real GT 与 raw mAP 路径有效；发现本地 registry 曾滞后于远端完成状态。结果表与 Wiki 已整改，剩余风险为单数据集单种子和非等容量比较。
- 2026-07-12：形成基于 GitHub 分支 `codex/phystime-performance-diagnosis-20260712`、正式实现 `3ac93a1` 与诊断提交 `d900c7c` 的 Pro 严厉审核 prompt；要求逐文件裁决根因，并交付等容量/同上下文/同候选数的最终模型、核心代码和因果实验 gate，禁止回退 selector 或用调参掩盖结构混杂。
- 2026-07-13：逐字归档 1539 行 PhysTime 性能 Pro 回复，附件与仓库归档 SHA256 均为 `651C4CA673073D7E4C05746138C82EBBE2E6174C459516FB40B3EFDCA47305AB`；审查裁决为 `HOLD AND REBUILD`。
- 2026-07-13：吸收 native tubelet feature-support provenance、capacity/context/candidate/assignment parity、gap-query 与训练态 mass-path 等新约束；新增 `idea:sm-ptaf`，严格标记为 `designed`。PhysTime 1.0 继续冻结为负基线，下一步先做 provenance 与 coordinate-only P0 gates，不创建虚假的实验或结果状态。
- 2026-07-13：重建 Wiki index/query pack/lint；共 38 个实体、10 个 gaps、66 条关系，0 孤立、0 失效引用、0 重复节点/边，query pack 4430 字符。
- 2026-07-13：独立复核远端最终作业与最佳 checkpoint 复算，`1159491..1159495`、`1159819..1159821` 均为 `COMPLETED 0:0`；正式快照关键合同测试 `69 passed`。确认结果可信、PhysTime 1.0 失败、physical-time 假设未被裁决。
- 2026-07-13：分级接受 Pro 审查而非照单全收：锁定 `HOLD AND REBUILD`，但 SM-PTAF 保持 `designed`。新增 tubelet 跨 gap 非线性融合风险，并把下一步拆成 G0 provenance、G1a `Q=J` temporal-metric、G1b 双侧共享 Q384 中性 lift、G2 mass residual。
- 2026-07-13：登记 RCL 为连续锚 TAD 近邻，进一步限制新颖性主张；建立 `source_registry.md` 记录本轮原始审查、正式结果、远端作业、代码和文献来源。
- 2026-07-13：独立核验后的 Wiki 完整性检查：39 个实体、10 个 gaps、67 条关系，0 孤立实体、0 失效引用、0 重复关系，query pack 4684 字符。
- 2026-07-13：实现 PhysTime G1a native-J192 matched control：分离 K384/J192/Q0=192/QΣ=378，补齐全部 patch 输入槽与 padding-repeat provenance、显式秒域起止边界、官方 ActionFormer 梯度/完整后处理 gate，以及 static-contract→G0→real-gate→pilot 哈希链。当前状态仅 `implemented`；远端 PyTorch、真实 THUMOS gate 与 pilot 尚未完成。
- 2026-07-13：G1a 扩展诊断与回归完成。远端 Linux/Torch 新旧相关 suite `100 passed`；修复 `AnchorFreeHead` 的 `dense_valid_len` 残留 NameError，以及 `selected_center` view 被物理中心原地写入污染、从而错误裁剪合法候选的关键 bug。部署合同升级为 commit/tree/config/data/checkpoint 全链绑定、双臂三步 AMP、正式单视频滑窗 NMS/evaluator 和严格 6 epoch artifact 验收。G1a 状态提升为 `tested`，正式 fixed-snapshot gate/pilot 与 mAP 仍 pending。
- 2026-07-13：G1a 预部署收口完成。真实 gate 改用 test split 尾样本和 test evaluator；数据指纹升级为逐文件完整 SHA256/Merkle；checkpoint/metrics 验收升级为真实反序列化与 evaluator 独立重算；VideoMAE/TIA 在 patch、attention、残差、MLP、卷积和 norm 全路径实施严格 padding isolation。全量 411 个 THUMOS14 MP4 的 decoder/annotation timebase 审计确认最大相对 FPS 偏差约 1.12%、帧数偏差为 0，配置容差固定为 1.25%/0.01%。远端新旧回归 `116 passed`；证据仍仅为 `tested`，正式 clean snapshot gate、pilot 和 mAP pending。
- 2026-07-13：首个 G1a clean snapshot `8e2b832` 已部署；gate `1161304` 在模型执行前正确失败，依赖 pilot `1161305/1161306` 未启动并取消。根因是 test 根目录 213 个 MP4 中有 2 个不在 annotation/正式 data_list，旧全量审计错误把目录集合等同于 evaluator 集合。修复后审计范围严格来自 `build_dataset(...).data_list`，消费 200 train+211 test；两个未引用文件显式登记并由目录 Merkle 绑定，被消费文件缺失仍硬失败。真实目录范围 precheck 与远端 `116 passed` 完成，等待新 commit/snapshot 重排。
- 2026-07-13：范围修复 commit `e598bd7` 的 gate `1161353` 越过 timebase 审计后，在模型初始 state 摘要处因 0 维 LongTensor 直接 byte-view 失败；pilot `1161354/1161355` 未启动并取消。已改为 `reshape(-1).view(torch.uint8)` 并加入标量 buffer 摘要回归；证据仍为工程修复，尚无 pilot mAP。
- 2026-07-13：标量摘要修复 commit `d193417` 的 gate `1161378` 越过全量数据、checkpoint、evaluator 与模型构建，在 selected-axis 首个真实样本因旧逐样本 `regression_gradient` 非零合同 fail-closed；pilot `1161379/1161380` 未启动并取消。根因是三步 gate 错把 ActionFormer ReLU 回归头的单样本零参数梯度当成断路。补丁现要求每步正 assignment、正 `reg_loss`、全部有限，adapter/projection/classification 每步非零，regression 三步内至少一次非零，并保存逐步 assignment/梯度证据；远端完整回归 `118 passed`，正在接受独立最高强度逐行审查，状态保持 `tested`。
- 2026-07-13：对上一条“ReLU 根因”作证据纠正：旧 `1161378` artifact 没有 assignment、`reg_loss` 或 pre-ReLU 激活，故只能说现象与 dead zone 一致，不能说已证明。独立 max 审查发现 gate validator 可被顶层字段伪造、buffer 可冒充参数更新、batch/scheduler 不是生产轨迹、`scale` 漏检及 schema/artifact 防御缺口。v3 修复改用正式 batch=2 DataLoader、warmup scheduler、EMA 和生产更新顺序，记录并重算逐步 assignment/pre-ReLU/梯度/LR/optimizer state，以 trainable-only hash+delta 证明更新，并重算 pilot 全部绑定；远端回归 `142 passed`，同一代理第二轮复审中，状态仍为 `tested`。
- 2026-07-17：G1b SDPQ GT-boundary-fix 20-epoch medium run `1167110` 完成，真实 gate `1167109` 与最终轻量 checkpoint/evaluator artifact 均通过；原始结果登记在 `docs/evaluation/results.md`。状态提升为“medium-run trainability empirically supported”，但因旧 G1a controls 只有六轮且 commit 不同，结构优越性仍未裁决。
- 2026-07-17：实现同 commit、同 seed、同 K384/J192、同 20 epochs 的 selected-axis / physical-metric / G1b SDPQ 三臂部署闭环：共享 G1a+G1b real gate、统一 runner、独立 evaluator/checkpoint validator 与 afterok DAG。远端 Linux/PyTorch focused suite `75 passed`；正式 clean snapshot 与 Slurm 作业待提交。
- 2026-07-17：三臂预部署自查发现旧 G1b medium 评价使用 EMA、轻量 checkpoint 却只保存 online 权重，导致指标可由保存预测重算但 evaluated model 不可精确重放。新 suite 改为 final-only 保存 online+EMA、继续排除 optimizer/scheduler，并增加缺失/非有限 EMA 反例测试；远端 focused suite 更新为 `77 passed`。
- 2026-07-17：matched medium code commit `5e8a8219c27785c15d720c5ed3c6b37298a2a866` 已推送；通过 `ghproxy.net` 在远端建立 tree `7dfdf3d1c1e1c681a5df23f5916e2aa53de221ea` 的 clean snapshot，相关远端测试合计 `100 passed`。
- 2026-07-17：正式三臂 run root 为 `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_5e8a821_medium20_20260717_132000_0800`。shared gate `1168484` 已 `COMPLETED 0:0`；selected-axis `1168485`、physical-metric `1168486`、G1b SDPQ `1168487` 均已进入 epoch 0。状态为 `experiment_running`，mAP 仍为 NA。
- 2026-07-17：matched medium 三臂全部完成且 `validation_pass=true`：selected-axis `30.42%`、physical-metric `44.88%`、G1b SDPQ `30.88%` Avg-mAP；三者最佳/最终评价均为 epoch 19。physical-metric 相对 selected-axis 为 `+14.46` Avg-mAP、`+9.82` mAP@0.7；G1b 相对 selected-axis 仅 `+0.46` Avg-mAP，但 mAP@0.7 `+2.42`。三个 final-only checkpoint 均含有限 online/EMA 权重且排除 optimizer/scheduler；异常与 GT boundary repair 扫描为 0。裁决：physical-time metric 达到 `matched-medium-supported`，当前 SDPQ 结构不获支持，整体仍非 `paper_ready`，不自动启动 60-epoch full train。
- 2026-07-19：逐字归档针对 commit `0dc5851` 与 full60 `41.28/57.57%` 的 PhysTime/Q-lift Pro 严审；附件与仓库原文 SHA256 均为 `BBD48B6BCE5E4AC612A395561D2EABCBB1F6DB5880B329EF21CAC6808CFBD5E0`。
- 2026-07-19：完成独立代码与证据核验并分级吸收。接受 head-level physical-metric 作用域、旧随机/dense 非公平、K/J/Q 混杂、全精度跨窗口 NMS 修复、窗内无 GT 措辞和 Q×coordinate 四臂设计；不接受把 cross-attention 写成已证明唯一结构，也不把外部固定 pp/成本阈值、非单调 timestamp shuffle 或未经审计的 ActivityNet 选择写成既定合同。
- 2026-07-19：`exp:phystime-g1-matched-full60` 保持 `full60-single-seed-supported`；`idea:sm-ptaf` 吸收 support-preserving physical query lift 后仍为 `designed`。下一步先关闭 NMS/provenance/反事实与真实 CUDA gate，再做同一新 commit 的 Q192/Q384 × uniform-rank/physical 四臂 20-epoch 因子实验；新 full train 未解锁。
- 2026-07-20：逐字归档第二轮 PhysTime `STOP-Q-LIFT` Pro 严审；原文 1053 行、49884 字节，附件与仓库归档 SHA256 均为 `F08AF135EAC342960929031FE84400144F0ADA55720F9A744203CFF2943A5057`。审查代码锚点仍为 `0dc5851` / tree `bddc9b9`，当前文档 HEAD 相对该锚点没有可执行代码差异。
- 2026-07-20：独立代码核验确认跨窗口 NMS 提前舍入、proposal validity filter 缺失、FPNIdentity LN 后未 remask、per-GT assignment 缺失、query audit 命名混淆、`random_trunc` 静默 fallback 与 float class ID。它们是发布级 blocker/风险，不自动撤销 `57.57%`。
- 2026-07-20：高度认可 `STOP-Q-LIFT` 的研究顺序，但不完全接受原文。修正了 decode/assignment 主效应公式标签互换，并记录 strict inside-GT 仍依赖 decode center，故四臂只分解两个代码开关。Q-lift 是“未知且当前不获训练授权”，不是永久证伪。
- 2026-07-20：当前唯一任务改为 `P0-FULLPRECISION-NMS-REPLAY`；禁止并行加入 Q-lift 或新训练。P0 后依次做冻结 decode cross-replay、Q192 UU/UP/PU/PP 与无训练 Q-density replay。`exp:phystime-g1-matched-full60` 保持 `full60-single-seed-supported`，`idea:sm-ptaf` 保持 `designed`，不创建新 claim。
- 2026-07-20：完成 `P0-FULLPRECISION-NMS-REPLAY` 本地实现。移除模型输出 2/4 位提前舍入，固定 long class id，新增 raw/effective/output 三层 proposal validity 审计、全精度 pre-cross gzip artifact、legacy/fullprecision×filtered/unfiltered 四模式冻结重放、checkpoint epoch 绑定、source/runtime 坐标语义与配置等价哈希、单臂独立 evaluator/hash/delta validator，以及依赖四任务的 CPU suite validator。第一轮独立 max 审查的 HOLD 项已修；纯部署测试 `12 passed`，远端真实 gate、四任务和 suite 尚未运行，状态严格为 `implemented`，mAP 为 NA。
- 2026-07-20：P0 部署前审查闭环完成。后续独立审查发现并修复 source canonical/file hash 混用、artifact/epoch/run-dir 绑定、计数守恒、普通主配置被 P0 舍入策略污染、旧 focused test 断言、提交器错误 raw-video 默认路径和小边界位移误判等问题。两个 P0 overlay 现显式关闭舍入，普通配置保持 legacy 默认；数据路径从冻结 full60 共同 source gate 恢复。本地无 Torch 回归 `25 passed`、四脚本 Bash `-n` 通过，最终限定复核为 `DEPLOY` 且无剩余 P0/P1；远端 GPU gate 尚未运行，状态仍为 `implemented`、mAP 为 NA。
- 2026-07-20：首次 P0 Slurm 提交在 job 创建前被集群 `job_submit_lua` 拒绝，因为该集群要求 `--gpus=1` 而不识别 `--gres=gpu:1`，且唯一公共 partition 为 `gpu`。重试循环已由本任务终止，未生成任何 job。提交器改为所有 DAG 节点申请 1 卡；suite 仍仅运行 CPU 验证逻辑并在 deployment summary 中记录不使用 CUDA。
- 2026-07-20：纠正上一条“未生成 job”：重试期间 gate `1174679` 与四个依赖作业 `1174680–1174683` 已创建，但 suite 未提交。gate 在 focused tests 以 `31 passed / 2 failed` 正确 fail-closed；失败来自 `SimpleNamespace` 测试夹具不支持 ConfigDict 式赋值，以及 DDP gather 原地扩展首 rank 列表污染测试期望。合并实现改为复制列表，夹具改用 `ConfigDict`；四个 `DependencyNeverSatisfied` 作业已定向取消，整批不构成实验结果。
- 2026-07-20：P0 最终 runtime commit `c2cfcfa2470f9f1e0b9d10e397480f6c66aeaf2c` / tree `0b78dd402e8997239ef9d1b4b4cd8bfa4f7a6338` 已推 GitHub并建立 clean snapshot `opentad_phystime_p0_c2cfcfa_20260720`。正式 run root 为 `phystime_p0_fullprecision_c2cfcfa_20260720_025843_+0800`；gate `1174688` 的远端 focused tests 已 `33 passed`，四臂 `1174689–1174692` 与 suite `1174693` 仍依赖阻塞。状态更新为 `experiment_running`，mAP 仍为 NA。
