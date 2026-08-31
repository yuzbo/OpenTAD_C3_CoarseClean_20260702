# ZoomToken ordered-video decode reuse R1/K100 precheck start receipt

## Frozen identity

- Task: `ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`
- Base: `b82441c1aa2663069033d394794298d5c723bbb6`
- Candidate: `6139e793e530033e2af6d992819cbe327d5bbd86`
- Branch: `codex/zoomtoken-ordered-video-reuse-r1-k100-v001`
- Candidate remote ref contains the exact commit and the deployed checkout was clean.
- GitHub commit: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/6139e793e530033e2af6d992819cbe327d5bbd86`

Exactly three new files implement the frozen surface:

- `tools/bata/profile_zoomtoken_ordered_video_reuse_r1_k100_cost.py`
- `scripts/run_zoomtoken_ordered_video_reuse_r1_k100_cost_n16r4.sh`
- `tests/test_zoomtoken_ordered_video_reuse_r1_k100_cost.py`

No dataset, transform, model, backbone, Adapter, detector, evaluator, NMS, checkpoint or training configuration was modified.

## Independent admission evidence

- Local syntax/launcher checks passed.
- Focused tests: `8 passed`.
- First independent Critic found two blockers: evaluator parity was tolerant instead of exact, and `frame_inds` lacked an explicit identity tap.
- Both were minimally fixed: the six raw evaluator values are now exact hard gates, and the complete input-parity scan binds frame-index dtype, shape and values between legacy and rolling paths.
- A fresh independent re-Critic returned `PASS`.
- A fresh result-blind Evaluator returned `PRE_RUN_READY`.

## Remote precheck

- Slurm job: `1262717`
- Job name: `zt-r1reuse-pre`
- Mode: `PRECHECK_ONLY=1`; this is not a scientific submission and does not train, resume, evaluate model accuracy or create a formal result.
- Resources: one Slurm-visible GPU, six CPUs, three-hour ceiling.
- Source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_src_6139e793`
- Reserved formal result root (must remain absent during precheck): `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_r1_k100_6139e793_seed42_20260901`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_logs_6139e793_20260901`
- Formal submission count: `0/1`.

The precheck must complete the exact legacy-vs-rolling comparison over the frozen complete THUMOS14 validation population (`211 videos / 792 ordered windows`), including input tensor, mask, metadata and frame-index identity. Only `COMPLETED 0:0` with `PRECHECK_READY` may admit the sole formal job. Any failure is an engineering/protocol blocker and cannot trigger an automatic replacement.

No live or partial precheck output is scientific evidence. The next observation uses one silent terminal waiter with 10-minute real wall-clock intervals and emits only an authoritative terminal state.
