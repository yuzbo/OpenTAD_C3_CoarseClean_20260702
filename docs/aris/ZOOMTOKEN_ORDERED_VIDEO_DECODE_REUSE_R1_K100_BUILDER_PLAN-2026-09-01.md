# ZoomToken ordered-video decode reuse R1/K100 Builder plan

## Frozen assignment

- Task: `ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`
- Pro decision: `PIVOT`; role contract: `KEEP`
- Exact Project conversation: `6a958d01-9768-83ea-b163-9b481bb64856`
- Base revision: `b82441c1aa2663069033d394794298d5c723bbb6`
- Branch: `codex/zoomtoken-ordered-video-reuse-r1-k100-v001`
- Scientific variable: evaluation-side ordered per-video decode reuse only. Model, checkpoint, data semantics, detector, evaluator, postprocess and NMS remain frozen.

## Single mechanism

The canonical THUMOS14 validation iterator remains in its original 211-video / 792-window order. For each arm and pass, one synchronous CPU iterator keeps a bounded raw RGB `uint8` frame buffer for the current video. A source frame is decoded at most once within that arm/pass; overlapping windows gather their repeated frame indices from the buffer. The iterator resets at video, ordering discontinuity, warmup-to-measurement, arm and pass boundaries.

The cache is intentionally below preprocessing: resize, center crop, `NCTHW` formatting, tensor conversion and collation still execute for every window. Decode remains inside `next_batch()` and therefore inside the existing timed full-stack boundary. There is no asynchronous prefetch, GPU cache, transformed-frame cache, model-output cache, prediction cache, hidden-state reuse, worker sweep or cross-pass state.

## Allowed implementation surface

Only these three new files may be added to the frozen code branch:

1. `tools/bata/profile_zoomtoken_ordered_video_reuse_r1_k100_cost.py`
2. `scripts/run_zoomtoken_ordered_video_reuse_r1_k100_cost_n16r4.sh`
3. `tests/test_zoomtoken_ordered_video_reuse_r1_k100_cost.py`

No existing dataset, transform, model, backbone, Adapter, detector, evaluator, NMS, checkpoint or training configuration may be edited.

## Fail-closed Builder contracts

The focused tests and precheck must prove all of the following before a formal submission:

- exact clean Git identity and the frozen R1 checkpoint/config identities;
- canonical validation population `211 videos / 792 ordered windows` for both arms;
- fixed eight-pass order `K100,R1,R1,K100,R1,K100,K100,R1`;
- batch size 1, worker count 0 and 50-window warmup per pass;
- reset after warmup so no warmed decode state enters measured windows;
- monotonically ordered per-video access and explicit reset on discontinuity;
- every source frame index decoded no more than once per video/arm/pass;
- bounded raw `uint8` CPU buffer and no transformed/GPU/output cache;
- exact per-window metadata, frame-index, mask and pre-H2D tensor parity against the canonical loader;
- exact prediction SHA and six-value evaluator-vector parity for every pass;
- unchanged production evaluator known-answer check;
- continuous NVML trace with complete power coverage and atomic pass/profile/terminal receipts;
- complete full-stack timing boundary including decode, H2D, model, postprocess and final video NMS.

Any identity, population, ordering, parity, measurement-completeness or artifact failure is an objective engineering/protocol blocker. It cannot be repaired by changing scientific gates or submitting a replacement job.

## Formal protocol and decision gate

- Full frozen THUMOS14 validation population; this is not official-test evidence.
- One RTX 4090, six CPUs, 16-hour walltime, one formal submission only.
- Eight passes in the frozen order above; 50 warmup windows per pass.
- Compare pooled R1 against pooled K100.
- Acceptance requires all of: full identity/artifact completeness, exact input/prediction/evaluator parity, wall-time ratio `<= 0.95`, gross GPU energy ratio `<= 0.95`, allocated-memory ratio `<= 1.05`, reserved-memory ratio `<= 1.05`.
- Valid pass: `SURVIVE_ORDERED_VIDEO_REUSE_DEPLOYMENT_PENDING_FRESH_PRO`.
- Valid failure: `STOP_R1_CURRENT_CONTIGUOUS_SUPPORT_AS_SINGLE_GPU_EFFICIENCY_ROUTE`.
- Objective blocker: no scientific conclusion, no retry/replacement, return the terminal evidence to a fresh exact-Project Pro.

## Builder verification sequence

1. Implement the new synchronous iterator and isolated two-arm profiler.
2. Run `py_compile` and the new focused test file locally.
3. Run an independent code Critic against this frozen plan.
4. Run a fresh result-blind Evaluator; it may return only `PRE_RUN_READY` or a precise blocker.
5. Deploy the exact clean candidate and run one `PRECHECK_ONLY=1` launcher on N16R4.
6. Only after `PRECHECK_READY`, submit the sole formal Slurm job.
7. On any terminal outcome, preserve all evidence and return it to exactly one fresh Pro before any successor experiment.

## Beijing deadlines from Pro

- Builder plan: `2026-09-01 02:00`
- Candidate: `2026-09-01 18:00`
- Critic: `2026-09-01 21:00`
- Evaluator: `2026-09-02 00:30`
- Precheck: `2026-09-02 02:00`
- Formal submission: `2026-09-02 03:00`
- Queue check/blocker return: `2026-09-02 18:00 / 18:15`
- Scientific return: `2026-09-04 12:00`
- Fresh post-result Pro: within one hour of terminal evidence ingestion.
