# ZoomToken ordered-video decode reuse R1/K100 terminal receipt

## Frozen identity and terminal state

- Task: `ZOOMTOKEN-ORDERED-VIDEO-DECODE-REUSE-R1-K100-FULLSTACK-VIABILITY-CLOSURE-v001`
- Candidate: `6139e793e530033e2af6d992819cbe327d5bbd86`
- Branch: `codex/zoomtoken-ordered-video-reuse-r1-k100-v001`
- GitHub implementation: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/6139e793e530033e2af6d992819cbe327d5bbd86`
- Formal Slurm job: `1262753` (`zt-r1reuse-s42`), submission `1/1`, no replacement.
- Node/resource: `g0030`, one RTX 4090, six CPUs.
- Start/end/elapsed: `2026-09-01T00:45:08+08:00` / `2026-09-01T04:35:06+08:00` / `03:49:58`.
- Scheduler terminal: `COMPLETED 0:0`.
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_ordered_video_reuse_r1_k100_6139e793_seed42_20260901`.
- No training, resume or official-test opening occurred.

## Protocol and artifact completeness

- The exact pass order is `K100,R1,R1,K100,R1,K100,K100,R1`; all eight passes completed.
- Every pass covers the complete frozen THUMOS14 validation population: 211 videos and 792 ordered windows, with 50 warmup windows per pass.
- `cost_samples.jsonl` contains 6,336 rows; `power_trace.jsonl` contains 485,950 rows. `acquisition_state.json` reports `COST_AND_DIAGNOSTIC_COMPLETE`, zero anomalies and complete prediction identity.
- All eight prediction JSONs and six-value raw evaluator vectors exist. The four K100 prediction hashes are identical to the frozen K100 anchor; the four R1 hashes are identical to the frozen R1 anchor. Per-arm evaluator vectors are also identical across all four passes.
- Every pass has complete power coverage with coverage ratio `1.0`, coverage of measurement start/end and maximum observed trace gap `66.38012081384659 ms`.
- The measured path remains decode through H2D, detector/model, postprocess and final video NMS. Final NMS is measured once per pass and amortized over its 792 window rows.
- Two fresh independent terminal reviewers reproduced the row counts, order, identity and frozen ratios. Both returned protocol/integrity `PASS`.

## Frozen primary estimator and gate

The primary estimator is the ratio of the median of four complete-pass arm estimates.

| Evidence | K100 median-four | R1 median-four | R1/K100 | Frozen gate | Result |
|---|---:|---:|---:|---:|---|
| end-to-end p50 | `1310.197414345851 ms` | `1285.3026268345238 ms` | `0.9809992087919388` | `<=0.95` | FAIL |
| gross GPU energy/pass | `87851.06299819297 J` | `80918.38455151053 J` | `0.9210860038560372` | `<=0.95` | PASS |
| peak allocated memory | `2053.2802734375 MB` | `1542.54638671875 MB` | `0.7512595365932656` | `<=1.05` | PASS |
| peak reserved memory | `2494 MB` | `1720 MB` | `0.6896551724137931` | `<=1.05` | PASS |

The observed p50 reduction is only `1.900079%`, below the required `5%`. Energy falls by `7.891400%`, and memory falls materially, but the gate is conjunctive. Therefore the frozen terminal decision is:

`STOP_R1_CURRENT_CONTIGUOUS_SUPPORT_AS_SINGLE_GPU_EFFICIENCY_ROUTE`

This is a valid negative systems result, not an engineering or population blocker.

## Accuracy and diagnostics boundary

- Exact raw evaluator vectors are K100 Avg/mAP@0.6/mAP@0.7 `68.506890/61.191118/46.246663` and R1 `69.042148/61.086961/46.499472` percent.
- These values are identity diagnostics for the frozen checkpoints, not a cost-admission gate and not new matched-training evidence.
- Short-action and boundary diagnostics are complete, but they remain descriptive validation diagnostics. They do not establish boundary protection.
- GPU temperature was not measured. Power coverage itself is complete.

## Supported and unsupported claims

Supported: on this single RTX 4090, complete THUMOS14 validation population and frozen checkpoints, symmetric bounded ordered-video decode reuse preserves the frozen predictions/evaluator vectors and reduces R1 gross energy and memory, but does not deliver the preregistered 5% full-stack p50 improvement.

Unsupported: a positive latency headline; official-test, matched-training, multi-seed, cross-hardware or cross-dataset generalization; Online TAD; accuracy preservation as a newly trained method; boundary protection; or a universal failure of decode reuse, dynamic computation or all spatial-support methods.

No retry, threshold revision, local model rescue, cost rerun or successor experiment is authorized before one fresh exact-Project Pro adjudication.
