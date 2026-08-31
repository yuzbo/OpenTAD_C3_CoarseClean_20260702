# ZoomToken ordered-video decode reuse R1/K100 formal start receipt

## Frozen identity

- Task: `ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`
- Base: `b82441c1aa2663069033d394794298d5c723bbb6`
- Candidate: `6139e793e530033e2af6d992819cbe327d5bbd86`
- Branch: `codex/zoomtoken-ordered-video-reuse-r1-k100-v001`
- GitHub commit: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/6139e793e530033e2af6d992819cbe327d5bbd86`
- Source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_src_6139e793`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_r1_k100_6139e793_seed42_20260901`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_logs_6139e793_20260901`

## Precheck admission

- Job `1262717` (`zt-r1reuse-pre`) completed `0:0` on `g0063` after `00:57:10`.
- The result root remained absent through precheck terminal state.
- The complete canonical THUMOS14 validation population was checked: `211` videos and `792` ordered windows.
- Legacy and rolling paths matched exactly for pre-H2D tensors, masks, metadata and frame-index identity over all 792 windows.
- The rolling path opened one reader per video, used bounded CPU `uint8` raw-frame storage, had no asynchronous prefetch, and reported a maximum of 768 buffered frames.
- Both epoch-59 EMA checkpoints, configs, evaluator/NMS contracts, annotation/class map, video inventory and atomic writers passed their frozen checks.
- Precheck emitted `PRECHECK_READY`, read no validation metric, and performed no training or resume.

## Sole formal job

- Slurm job: `1262753`
- Job name: `zt-r1reuse-s42`
- Submitted: `2026-09-01T00:44:40+08:00`
- Started: `2026-09-01T00:45:08+08:00`
- Initial node: `g0030`
- Resources: one Slurm-visible GPU, six CPUs, `16:00:00` walltime.
- Launcher: `scripts/run_zoomtoken_ordered_video_reuse_r1_k100_cost_n16r4.sh`
- Pass order: `K100,R1,R1,K100,R1,K100,K100,R1`; each pass has 50 warmup windows.
- Formal submission count: `1/1`; retry, resume, replacement, extra arm, training and official-test opening are forbidden.

The formal replay must cover all 211 validation videos and 792 ordered windows in every pass. No live or partial latency, energy, memory, prediction, evaluator, short-action or boundary value is consumed. Only complete terminal artifacts or an objective terminal blocker may be interpreted, and either outcome returns once to a fresh exact-Project Pro review.
