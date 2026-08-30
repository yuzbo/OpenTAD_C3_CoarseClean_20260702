# ZoomToken R1 depth-sparsity four-arm full-stack Pareto — Builder plan

Date: 2026-08-31 (Asia/Shanghai)

## Frozen task

Implement the read-only four-arm full-stack replay assigned by the fresh GridFuse terminal Pro review:

`ZOOMTOKEN-R1-DEPTH-SPARSITY-READONLY-FOUR-ARM-FULLSTACK-PARETO-CLOSURE-v001`

The replay compares four existing epoch-59 EMA checkpoints on the canonical THUMOS14 validation population. It does not train, resume, modify checkpoints, or change model/data/evaluator/NMS semantics.

| Arm | Frozen model | Source job | Checkpoint role |
| --- | --- | ---: | --- |
| A | R1/FULL64 | 1249099 | control |
| B | DSR6-KV | 1252527 | candidate |
| C | MOD32-KV | 1252180 | candidate |
| D | DROP32 | 1252179 | candidate |

## Minimal implementation surface

Only these new files are permitted:

- `tools/bata/profile_zoomtoken_r1_depth_pareto_cost.py`
- `scripts/run_zoomtoken_r1_depth_pareto_cost_n16r4.sh`
- `tests/test_zoomtoken_r1_depth_pareto_cost.py`

The implementation reuses the already exercised atomic full-stack profiler machinery, while replacing the old two-arm protocol with the frozen four-arm contract.

## Protocol to encode

- One Slurm-visible GPU and five allocated CPUs; no physical GPU override.
- Canonical validation population: 211 videos and 792 ordered loader items.
- Sixteen complete passes in Williams order:
  `A B D C / B C A D / C D B A / D A C B`.
- Fifty untimed warmup windows before each pass.
- One continuous NVML power sidecar, pass-local coverage receipts, and atomic persistence after every pass.
- Per-pass prediction, SHA, raw evaluator vector, ordered cost rows, p50/p95, throughput, allocated/reserved peak memory, gross joules, population identity, and route-audit evidence.
- Diagnostics run only after all timed acquisition is durably saved.
- Primary arm estimate: median of four pass-local p50 and gross-joule estimates.
- Candidate ratios are computed against arm A. B/C/D survive independently only if p50 and gross energy are each at most 0.95, while allocated and reserved peak-memory ratios are each at most 1.05.
- Historical accuracy is a non-blocking diagnostic, not a cost gate.
- Any incomplete execution or measurement yields `BLOCKER_NO_SCIENTIFIC_COST_DECISION`; no replacement job is permitted.

## Verification and handoff

1. Run local syntax checks and the focused test suite.
2. Freeze a clean pushed candidate and record repository, branch, and exact GitHub commit URL.
3. Obtain a fresh independent Critic verdict on the code/protocol and a fresh result-blind Evaluator verdict on executability.
4. Reproduce the focused checks on N16R4 and run exactly one Slurm `PRECHECK_ONLY=1` job.
5. Only after `PRECHECK_READY`, submit exactly one formal 1-GPU/5-CPU/16-hour job and monitor it with one terminal-only machine-side waiter.
6. After terminal evidence is ingested, open exactly one fresh Project Pro discussion. Its prompt must include the latest repository URL, branch URL, and exact commit URL in addition to the full terminal evidence.

## Current Builder status

- Four-arm profiler, launcher, and focused tests implemented.
- Local focused suite: 22 passed, 1 N16R4-only test skipped.
- Python compilation, Bash syntax, and `git diff --check`: passed.
- Clean pushed candidate: `b82441c1aa2663069033d394794298d5c723bbb6`.
- GitHub repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- GitHub branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-r1-depth-pareto-v001>.
- GitHub exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b82441c1aa2663069033d394794298d5c723bbb6>.
- Independent Critic: `PASS`.
- Fresh result-blind Evaluator: `PRE_RUN_READY`.
- N16R4 exact-checkout focused suite: 23 passed.
- Non-scientific Slurm precheck: job `1262119`, `COMPLETED 0:0 / PRECHECK_READY`.
- Sole formal job: `1262120` (`zt-r1dp-s42`), one GPU/five CPUs/16 hours, submission `1/1`; initially `PENDING (Priority)`.
