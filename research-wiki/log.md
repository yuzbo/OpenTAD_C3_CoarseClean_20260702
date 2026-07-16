---
type: wiki_log
append_only: true
---

# Research Wiki Log

- 2026-07-11：初始化 C3/DUCA research-wiki。
- 2026-07-11：逐轮读取主任务 191 轮，归档 158 条用户侧原始消息。
- 2026-07-11：登记实现代理、论文代理和早期目标任务的近期记录。
- 2026-07-11：登记 C3、PAction、GAS-VT、lattice、detector-aware、TrueTime、
  DUCA、MUST、X3D/SlowFast、physical-grid、CFPA、CVCR、ChronoTransport、
  PhysTime 路线。
- 2026-07-11：冻结当前裁决：70aa069 是待裁决 DUCA baseline，a5e1774 是最新
  审计代码；正式论文 claim 尚未闭环。
- 2026-07-11：wiki lint 通过：16 ideas、7 experiments、10 claims、47 edges、
  0 orphan nodes、0 curated broken links；query pack 2825 chars。
- 2026-07-11：纠正 ChronoTransport 过期状态：`92029ea` formal Stage-B P3 science gate 为负，Stage C/P5 未解锁；新增独立 negative experiment 节点，路线暂停。
- 2026-07-12：为无法读取本地工作区的 Pro reviewer 建立 GitHub 固定提交审查入口；仅同步
  ChronoTransport r1 规格、实现表面、原 Pro 记录、两轮独立复核与本地源码审计，明确排除
  数据、checkpoint、GPU 日志和新行为结果。审查仍止于 `REVISE_SPEC_BEFORE_PLAN`，不得借
  GitHub 同步越过到实现、profiling、Gate 1、新 seed 或 Stage C。
- 2026-07-15：纠正活动任务为 Spatial Zoom-only。S1 保持完整时间轴，只比较 matched
  dense160/224/256；DUCA 与时序选帧不属于此执行线。旧 `35204f5` 3x3 任务因每臂均出现
  `upsample_linear1d_backward_out_cuda` nondeterminism warning 而失效并取消，222 个
  checkpoint/sidecar 仅保留为诊断，无 sealed-test、最终 mAP、成本或 GO/KILL。
- 2026-07-15：在 `codex/spatial-zoom-s1-audit-fix-20260715` 实现 exact-2x deterministic
  temporal interpolation、formal strict determinism、真实 full-model AMP backward precheck、
  paired Bayesian video-cluster bootstrap、test-open crash recovery、原子证据发布和成本口径
  限定。本地 S1/train-iteration tests 为 `40 passed, 1 skipped`，required C3 regression 为
  `20 passed`；
  远端 replacement CUDA gate 与 3x3 matrix 尚未产生结果。
- 2026-07-15：提交 `64e71dd` 后创建 exact remote snapshot
  `opentad_spatial_zoom_s1_64e71dd_20260715_ghfast`。Job `1165648` 的 Linux 测试
  `41 passed`，真实 pretrained VideoMAE + AdaTAD AMP forward/backward 成功执行，但门禁因
  `backbone.model.backbone.fc_norm.{weight,bias}` 两个参数无梯度而 fail-closed。源码证明
  它们仅属于 `return_feat_map=False` 的分类池化出口，S1 dense TAD 路径在此前返回。
- 2026-07-15：将 formal precheck 升级为 v6：只允许上述两个精确参数缺梯度，要求观察集合
  与冻结白名单完全相等，并分别统计 trainable 与 gradient-required tensors；新增少项、额外
  断图和 component coverage 漂移测试。本地为 `43 passed, 1 skipped`。替换 CUDA gate 通过
  前保持 3x3 正式训练未排队。
- 2026-07-15：独立 `gpt-5.6-sol/max` 只读审计给出 `PASS_FOR_REMOTE_GATE`，无 P0/P1；
  其唯一 P2 是 component 与 global gradient counts 未闭合。提交前已增加所有 component 的
  trainable/required/unused/finite/nonzero 守恒验证及 under-report 反例测试。该门禁结论仍不
  等价于 S1 empirical GO。
- 2026-07-15：修复提交 `47842427eb373fb1f440b1661971a6a231a95f67` 已推送。
  新 exact snapshot `opentad_spatial_zoom_s1_4784242_20260715_ghfast`、suite
  `spatial_zoom_s1_4784242_20260715_2245` 和 manifest 均重新创建，未修改或复用旧快照。
- 2026-07-15：CUDA gate Job `1165667` COMPLETED 0:0 / PASS。三分辨率均为 339 个
  trainable、337 个 gradient-required/finite tensors；缺梯度集合精确等于两个 `fc_norm`，
  strict deterministic warn-only=false，formal_training_ready=true。
- 2026-07-15：正式 3x3 S1 matrix 已排队：`1165669-1165671`=dense160，
  `1165672-1165674`=dense224，`1165675-1165677`=dense256，各对应 seeds
  3407/3408/3409。状态提升到 `experiment_running`，sealed test/cost/GO-KILL 尚未完成。
- 2026-07-15：九任务均开始 epoch 0，初始 loss 有限。若 GradScaler overflow，S1 会恢复
  同批次 RNG/model buffers 并重放，成功前不推进 scheduler/EMA；单次 `AMP skipped batch`
  日志不是丢失 optimizer update。八次重试耗尽、raw loss 非有限或 successful-update parity
  不满足才判失败，并在 checkpoint/evidence 中记录全部 attempts。
- 2026-07-15 23:03+08:00：`1165669-1165677` 全部 RUNNING；dense160 已进入 epoch 2，
  dense224/256 已进入 epoch 1。最新 loss `1.0760-1.1135`，各臂 2-3 次 AMP attempt 已恢复；
  Traceback/OOM/non-finite/determinism/FAIL 计数均为 0。epoch 40 前无 validation mAP 是预注册行为。
- 2026-07-15 23:08+08:00: updated the two-hour Spatial Zoom S1 heartbeat
  into an idempotent end-to-end state machine: monitor the formal 3x3 matrix,
  validate checkpoint selection, open the sealed test once, run official test
  and trained-checkpoint cost profiling, issue the frozen GO/KILL diagnosis,
  publish only auditable summaries to GitHub, then invoke a verified Pro-tier
  code/protocol/result review and deactivate the automation.
- 2026-07-16 02:55+08:00: all formal jobs `1165669-1165677` failed closed while
  atomically writing checkpoints because the shared `/data` filesystem had only
  about 13 MiB free. Losses remained finite; this is infrastructure failure,
  not S1 performance evidence. Resume is forbidden, so the matrix is invalid.
- 2026-07-16 03:02+08:00: verified the 130 GiB canonical root with exactly 222
  checkpoints belongs to invalid commit `35204f58`. Removed only those forbidden
  `.pth` weights, preserved all 222 sidecars and diagnostics, and wrote purge
  receipt `59f1d9d3499eb3cd105478672805f9a19c15a73c8a747b1249bc7c2372ad9ecf`.
  Began a replacement fix that saves only gate-eligible checkpoints, cleans
  failed atomic temporaries, and enforces a 96 GiB free-space preflight.
- 2026-07-16 03:18+08:00: hash-validated and reclaimed the no-resume
  `47842427` failed matrix: 151 weights (94,195,092,514 bytes) plus nine
  temporaries (134,217,728 bytes). Preserved all sidecars and diagnostics;
  receipt file SHA is `b5237253eaa8d196957da47d5ebd2c07ae6537596b6e53e1e4348286c88d58d9`.
  `/data` free capacity rose to 217 GiB before the fresh replacement gate.
- 2026-07-16 03:29+08:00: storage-safe commit
  `0421a8d9f6982a6d4ec1fb590cd108581fa2bb83` passed CUDA gate Job `1165774`
  (0:0, 47 Linux tests, three-resolution deterministic full-model backward).
  Precheck internal SHA is `3d30ea5489b2ac7f07785dff94ed057ac420aebdd8762ab6df6c76a2ffb003ea`.
- 2026-07-16 03:31+08:00: submitted fresh epoch-0 3x3 Jobs `1165775-1165783`
  to canonical namespace `bf71376e2d57946a3f898d25b7dcc88cfc002549a9ed78656293f1a95316a8f7`.
  All nine received normal Slurm GPUs without physical-index override. No
  resume, test opening, profiling, or performance interpretation occurred.
- 2026-07-16 04:57+08:00: all nine storage-safe S1 cells remain healthy and
  running at epochs 11-15. Latest finite losses are `0.5830-0.7153`; retry
  counts are 2-4 with maximum same-batch retry `2/8`, followed by continued
  progress. Pre-gate checkpoint/evidence counts correctly remain zero and
  storage remains 217 GiB free. No hard-failure signature was found.
- 2026-07-16 06:58+08:00: formal S1 replacement remains healthy at epochs
  27-37. Latest finite losses are `0.4248-0.5028`; AMP retry counts/depth did
  not increase. No checkpoint or gate evidence exists before its frozen epoch,
  no hard failure was found, and `/data` still reports 217 GiB free.
- 2026-07-16 09:05+08:00: all nine S1 jobs remain healthy at epochs 43-55.
  Exactly 37 eligible checkpoints and 37 sidecars exist, with 34 completed gate
  evidence records and three evaluations still publishing; no temporary or
  selection file exists. Completed gate-only Avg-mAP values span 14.85-15.15
  and are not sealed-test evidence. Update/attempt parity, commit binding, and
  `official_test_opened=false` hold; hard-failure scans are clean and `/data`
  retains about 195 GiB free.
- 2026-07-16 11:00+08:00: Jobs `1165775-1165780` completed all 4,800 model
  updates and ten gate artifacts but failed only in post-training selection.
  Every rejected row was a finite zero-length proposal, which official OpenTAD
  retains as a zero-IoU false positive. An exact official-parity probe passed;
  dense160/3407 epoch59 recomputed gate Avg-mAP 64.7391 and mAP@0.6/0.7
  58.0680/46.1726. Fix `cbc63d0` passed 41 focused tests plus 20 C3 regression
  tests. No selection, sealed test, profile, or GO/KILL was allowed; dense256
  remains running and the strict contract requires a new formal namespace.
- 2026-07-16 11:25+08:00: formal replacement commit `18139b9` passed full
  CUDA gate Job `1166358` (`0:0`). Submitted fresh epoch-0 3x3 Jobs
  `1166361-1166369` to canonical namespace `d95a36db...`, without resume or
  physical-GPU override. An `afterany` dependency on old diagnostic Jobs
  `1165781-1165783` prevents old/new overlap. No sealed test, profile, or
  GO/KILL was opened.
