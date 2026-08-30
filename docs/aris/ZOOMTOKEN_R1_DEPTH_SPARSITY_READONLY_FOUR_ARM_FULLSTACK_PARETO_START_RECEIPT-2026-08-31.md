# ZoomToken R1 depth-sparsity read-only four-arm full-stack Pareto start receipt

## Authority and immutable implementation

- Task: `ZOOMTOKEN-R1-DEPTH-SPARSITY-READONLY-FOUR-ARM-FULLSTACK-PARETO-CLOSURE-v001`
- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-r1-depth-pareto-v001>
- Exact implementation commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b82441c1aa2663069033d394794298d5c723bbb6>
- Execution base: `c6327a891809aa30370b3b2d9bedab0dcfe0d326`
- Clean N16R4 source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_depth_pareto_src_b82441c1`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_depth_pareto_b82441c1_seed42_20260831`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_depth_pareto_logs_b82441c1_20260831`

The implementation changes only the three Pro-authorized files: the read-only profiler, N16R4 launcher, and focused tests. It does not modify or train a model, resume training, alter a checkpoint, change data/evaluator/NMS, add an arm, or change the frozen decision gates.

## Frozen four-arm protocol

- A: R1/FULL64, job `1249099`, epoch-59 EMA.
- B: DSR6-KV, job `1252527`, epoch-59 EMA.
- C: MOD32-KV, job `1252180`, epoch-59 EMA.
- D: DROP32, job `1252179`, epoch-59 EMA.
- Population: 211 validation videos and 792 ordered loader items.
- Pass order: `A B D C / B C A D / C D B A / D A C B`.
- Warmup: 50 windows before each of 16 complete passes.
- Primary evidence: per-pass full decode→H2D→model→postprocess→Soft-NMS p50 and total gross GPU joules; each arm is summarized by the median of four pass-local estimates.
- Survivor gate relative to A: p50 and gross energy ratios each `<=0.95`, allocated and reserved peak-memory ratios each `<=1.05`.
- Historical accuracy is diagnostic only. Incomplete execution or measurement returns `BLOCKER_NO_SCIENTIFIC_COST_DECISION` and does not authorize a replacement.

## Result-blind verification

- Local Python compilation, Bash syntax, and `git diff --check`: pass.
- Local focused tests: `22 passed, 1 skipped` (the skipped check requires the target N16R4 runtime).
- N16R4 exact-checkout focused tests: `23 passed in 12.39s`.
- Independent Critic: `PASS`.
- Fresh result-blind Evaluator: `PRE_RUN_READY`.
- Precheck job `1262119`: `COMPLETED 0:0 / PRECHECK_READY` on `g0063`, elapsed `00:00:59`.
- Precheck verified the clean remote commit, all four checkpoint SHA256 identities, epoch 59, `state_dict_ema`, 527 entries, four config/evaluator/NMS contracts, 411 MP4 inventory, 211/792 population, atomic writers, and production evaluator known answers. It did not read validation performance and did not train or resume.

## Sole formal action

- Job: `1262120` / `zt-r1dp-s42`.
- Submitted: `2026-08-31T06:37:14+08:00`.
- Initial authoritative state: `PENDING (Priority)`.
- Resources: `gpu`, one GPU, five CPUs, `16:00:00`.
- Formal submission count: `1/1`.
- Retry/resume/replacement/second seed/extra arm: forbidden.
- Terminal-only waiter: FastCtx `j-d0saxx`, 300-second real wall-clock sleep, no running metric output.

No running or partial latency, energy, memory, prediction, accuracy, short-action, or boundary value will be read or interpreted. A single terminal-only machine-side waiter owns the next state transition. Complete terminal evidence or an objective blocker will be ingested before exactly one fresh exact-Project Pro review. That review prompt must explicitly include the repository, branch, and exact implementation commit URLs above.
