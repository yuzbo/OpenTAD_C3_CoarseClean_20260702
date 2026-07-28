# Research Log

## 2026-07-27 — DUCA-RIME four-stage implementation

- Recorded user approval for direct four-stage execution.
- Marked the earlier total-60 bounded-density plan as superseded by the
  dynamic-budget RIME adjudication.
- Implemented Phase 1 exact-K/geometry/cost evidence, Phase 2 `U-mixed-K` and
  causal gates, Phase 3 candidate/ablation matrix, and Phase 4 formal matrix.
- Corrected the mixed-K schedule to a per-video stateless 60-entry exposure
  with exact mean K=384.
- Corrected K192 to `fixed_floor_budget_position_only`; it cannot support a
  dynamic-budget claim.
- Corrected full-stack cost matching to exact `U-same-K` replay.
- Corrected the profiler to consume `effective_k`.
- Added an explicit Phase-4 authorization → Phase-2 receipt → budget protocol
  path/hash → checkpoint audit → terminal identity binding.
- Independent MAX audit found and corrected a deployment-blocking dense
  evidence SHA variable-name mismatch between the Phase-3 controller and
  Phase-3/4 submitters.
- Deployment preflight also corrected the one legacy Phase-1 gate invocation
  to run through `bash`, because that retained script is intentionally tracked
  without an executable bit.
- Rejected fabrication of a trained commit for the historical exact-uniform
  checkpoint whose surviving log records `git_head=unknown`. The Phase-1
  no-probe/probe cost pair now uses one byte-identical, SHA-bound checkpoint
  trained at `cb89586a92b8b0a8349ecc9551bc50aa97982360`; the launcher and seal
  require identical checkpoint SHA, trained commit, epoch, and EMA state.
  The no-probe arm drops only the registered probe/transition state that its
  configuration does not build, so the common heavy path is weight-identical.
- Retrieved the official released AdaTAD VideoMAE-S/ActionFormer checkpoint
  from the source-linked Google Drive file through its direct user-content
  endpoint. Its size is `200938640` bytes and SHA-256 is
  `21dbb9efe9f62d3089696c3c535edd27e8b8d9c14a06a21aac5738ec82bfab97`.
- Pre-registered the exact K/cost ladder, K384/K192 panel semantics,
  `weak_overlap` decoder, risk rule, O4 calibration gates, and 2 s / 8 s
  duration strata in both the submitting commit and submission manifest.
- The first released transaction at commit `57965bec` failed at the code gate
  before any experiment work because Slurm `--wrap` executes under `/bin/sh`
  and therefore rejected the Bash-only `source` builtin. All downstream jobs
  remained dependency-blocked. The submitter now explicitly enters
  `/bin/bash -lc` before the CUDA/Miniforge bootstrap; the failed root and
  scheduler records are retained as negative deployment evidence.
- The next code gate at commit `c0e7e036` reached the full remote Torch suite
  and stopped on two stale test fixtures: a `protected_e2e` RIME construction
  requested bridge scale `0.0` despite the registered `1.0` contract, and a
  no-padding ledger fixture omitted `irregular_dense_valid_len`. The fixtures
  were corrected to exercise, rather than violate, their production contracts;
  no model behavior or gate threshold was relaxed.
- The diagnostic rerun at commit `8667d057` cleared those failures and exposed
  one final stale floor-protocol fixture that constructed non-uniform RIME
  without the required official trainable ASFormer evidence source. The fixture
  now supplies that registered source contract; production behavior and every
  scientific gate remain unchanged.
- At commit `3f8e3ca1`, the exact remote suite passed all 158 tests. Its first
  unobstructed config-matrix run then exposed an over-broad generic assertion:
  the two evaluation-only Phase-1 paired cost profilers were being checked as
  train/evaluation-result configs. The matrix now keeps batch-size-one and saved
  predictions mandatory for trainable/formal-evaluation configs, while applying
  the stronger relevant contract to those two cost-only configs: test batch
  size one, zero loader workers, no saved accuracy predictions, no accuracy
  claim, and a byte-identical paired checkpoint identity.
- Current state: `implemented/tested`; remote code gate and Slurm deployment
  remain pending. No empirical or paper-ready claim has been made.

## 2026-07-28 — DUCA-RIME final transaction released

- Froze the physical protocol on implementation commit
  `f510741b32075c5c4e729d4207a549886a6dd064`; manifest SHA-256 is
  `5d28d1d37e698b5f17156245f55da62a82dc5b537c32fe70104f1be231e605d8`.
- Released the immutable fail-closed transaction at
  `/data/run01/sczc063/yuzibo/rime_runs/duca_rime_four_phase_f510741b_20260728_094811`.
  Submission-manifest SHA-256 is
  `c74c351bad04dd7bfc6701ca5205e419116694d21736419776cec1f3cdb7ada6`.
- Code-gate job `1197889` completed successfully on the exact clean commit:
  all 158 focused tests passed and all 24 registered configs passed their
  stage-specific matrix.
- Slurm released Phase 1 (`1197890`), dense ActionFormer (`1197891`), and dense
  TriDet (`1197892`) into `RUNNING`. Phase 2 (`1197893`) and the Phase-3/4
  controller (`1197894`) remain dependency-gated as registered.
- Current scientific state is `experiment_running`, not
  `empirically_supported` or `paper_ready`.
- Early monitoring then found both dense jobs failed before their first
  optimizer update: their configs correctly set `dataset.val=None` and disabled
  validation intervals, but omitted the generic trainer's explicit
  `seal_eval_dataloaders_during_training=True` switch. The Phase-3 dependency
  failed closed. Phase 1 and all remaining jobs in this now-uncompletable
  transaction were canceled by exact job ID; the root and logs are retained.
- Added the missing seal to both dense configs and bound it in the launcher
  precheck, config matrix, and focused tests. This is an orchestration fix, not
  a model or scientific-protocol change.
