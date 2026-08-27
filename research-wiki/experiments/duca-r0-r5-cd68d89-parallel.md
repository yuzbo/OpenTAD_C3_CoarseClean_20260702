---
id: exp:duca-r0-r5-cd68d89-parallel
type: experiment
status: experiment_running
updated: 2026-07-22
---

# DUCA R0-R5 cd68d89 parallel formal suite

## Purpose

在离线 TAD、真实 AdaTAD/ActionFormer 与 TemporalMaxer 后端上，一次性回答边界微簇智能选帧是否优于严格匹配的均匀采样、检测器反馈是否有益、K384 降到 K256 后性能如何变化，以及真实端到端成本是否下降。

## Immutable identity

- Branch: `codex/duca-boundary-burst-20260722`
- Exact commit: `cd68d89dcc0854baa3c0107607086e801509b552`
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-boundary-burst-20260722`
- Clean snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_cd68d89_20260722`
- Formal root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506`
- Jobs ledger: `${formal_root}/jobs.tsv` and `${formal_root}/jobs.tsv.sha256`
- Deployment manifest: `${formal_root}/deployment_manifest.json`
- Task identity: offline temporal action detection, never Online TAD.

## Model contract

- Dense low-cost coarse probe predicts binary actionness and hidden temporal evidence.
- Transition/boundary scorer uses state-change evidence to allocate Oracle-like bilateral boundary bursts plus residual global context.
- Hard decoder enforces exact K and max-hole; local bilateral utility is separated from optional global mandatory groups.
- Selected original-time RGB observations feed the real selected-axis official-derived AdaTAD/ActionFormer or TemporalMaxer stack.
- P0 trains coarse/transition/burst supervision; official detector training then evaluates detached/adapted and protected-feedback variants.
- R0 is a detector-seen training-internal non-routing diagnostic. The learned family is preregistered as `R2Q3_privileged_boundary_burst`; only matched official-validation terminal EMA mAP can support continuation or claims.

## Deployed bundles

| R stage | Job | GPU | Dependency | Contents | Primary output |
|---|---:|---:|---|---|---|
| R0/R1 | `1180336` | 1 | none | Four-family frozen-detector reachability diagnostic plus production/no-leak contracts | `bundles/r0_r1/completion.json`, R0 summaries and seals |
| R2/R3 core | `1180337` | 3 | none | Exact-uniform, R2Q3 soft-detached and R2Q3 hard-detached P0/gate/official-60 arms | Per-arm terminal `epoch_59.pth` EMA evaluation |
| R2/R3 adapted | `1180338` | 3 | none | R2Q3 soft-adapted, R2Q3 hard-adapted G0 and R4Q5 diagnostic arms | Per-arm terminal official-validation EMA mAP |
| R4 | `1180339` | 2 | none | Fixed R2Q3 bootstrap, legal hard-swap alignment, protected G1 and G2 | Alignment artifact plus G1/G2 terminal mAP |
| R5 | `1180340` | 4 | none | Real TemporalMaxer gate, all 24 paper cells and applicable paired runtime profiles | 24 terminal evaluations and raw cost summaries |
| Aggregate | `1180341` | 1 | `afterok:1180340` | Fail-closed R5 aggregation | `r5/final_results.json` and sealed summaries |

All five model bundles were independently accepted by Slurm. They do not wait for one another. Only the R5 aggregate waits for R5 completion.

## Complete experiment inventory

The following list records the actual variants executed inside each bundle, not merely the outer Slurm jobs.

| Bundle | Exact experiment variants | Paper role |
|---|---|---|
| R0/R1 | `A_exact_uniform`, `R2Q3_privileged_boundary_burst`, `R4Q5_privileged_boundary_burst`, `Z_unrestricted_gt_oracle`; focused R1 contract tests | Detector-seen train-internal reachability and contract diagnostics only; absolute 93--94 mAP is forbidden from the main table |
| R2/R3 core | `two_stage_exact_uniform`; `boundary_burst_r2q3_soft_detached_g0`; `boundary_burst_r2q3_hard_detached_g0` | Matched uniform anchor; soft-versus-hard frontend supervision; detector-detached G0 comparisons |
| R2/R3 adapted | `boundary_burst_r2q3_soft_adapted_g0`; `boundary_burst_r2q3_g0` (hard-adapted); `boundary_burst_r4q5_g0` | Test whether adaptation improves learned selection and whether wider R4Q5 bursts help |
| R4 | Self-contained current-commit R2Q3 bootstrap, U/G0 terminal evaluation and legal hard-swap alignment; then `boundary_burst_r2q3_g1` and `boundary_burst_r2q3_g2` | Measure protected detector feedback and the training-only 50% uniform companion |
| R5 mechanism | Live DUCA-selected RGB -> VideoMAE -> TemporalMaxer one-step gate | Prove the second backend is real and trainable, not a placeholder |
| R5 ActionFormer | uniform/learned x K384/K256 x seeds 3407/5801/8123 = 12 cells | Primary multi-budget, multi-seed official mAP matrix |
| R5 TemporalMaxer | uniform/learned x K384/K256 x seeds 3407/5801/8123 = 12 cells | Cross-backend generalization mAP matrix |
| R5 cost | ActionFormer seed 3407 x uniform/learned x K384/K256 = four candidate profiles, each paired with dense AdaTAD in the same session | Same-backend end-to-end latency/memory evidence and performance-cost plots |
| Aggregate | `1180341`, after R5 only | Fail-closed collection of all 24 terminal rows and four paired cost records |

R4 and R5 intentionally regenerate their trainable prerequisites inside their own bundles so all five outer jobs can run without inter-bundle dependencies. Those repeated prerequisites are audit-bound execution work, not additional paper rows and must not be double-counted in result tables.

## R5 paper matrix

- Backends: `ActionFormer`, `TemporalMaxer`.
- Policies: exact-uniform and learned R2Q3 boundary burst.
- Budgets: `K=384` with max-hole `2`; `K=256` with max-hole `3`.
- Seeds: `3407`, `5801`, `8123`.
- Total official-mAP cells: `2 x 2 x 2 x 3 = 24`.
- Formal metric: complete THUMOS validation, OpenTAD evaluator, tIoU `0.3/0.4/0.5/0.6/0.7`, terminal `epoch_59.pth/state_dict_ema`.
- TemporalMaxer is cross-backend mAP generalization evidence. It is not paired against an AdaTAD dense runtime baseline.

## Cost and plotting evidence

- Paired runtime backend: ActionFormer/AdaTAD only.
- Candidate profiles: seed-3407 x uniform/learned x K384/K256 = four profiles.
- Each candidate is paired in the same profile session with the sealed dense AdaTAD baseline; cross-backend ratios fail closed.
- Plot CLI: `tools/bata/plot_duca_r5_performance_cost.py`.
- Planned exports: raw CSV/TSV/JSON, Avg-mAP versus latency PNG/PDF, Avg-mAP versus budget PNG/PDF, and a LaTeX figure snippet.
- Missing FLOPs remain explicitly `unavailable`; no synthetic FLOPs or TemporalMaxer/AdaTAD mixed ratio is permitted.

## Superseded startup evidence

- Commit `2645e688368acc36605f0047fb9af10e929eaec0` was fully submitted as Jobs `1180326--1180331`.
- The first real soft-bilateral P0 batch failed before an optimizer update at `build_boundary_burst_utility` because PyTorch 2.0 CUDA mishandled scalar advanced assignment to a two-dimensional row slice (`5920 vs 1184`).
- This is a runtime implementation failure with no mAP, not negative method evidence.
- Commit `cd68d89` replaces the row reset with dimension-explicit `index_fill_`. All `2645e68` Jobs were canceled and must never be reused as paper evidence.

## Current status and acceptance criteria

- Latest observed state (`2026-07-22 21:04 +08:00`): all five independent model bundles are `RUNNING` on separate nodes: `1180336/g0043`, `1180337/g0006`, `1180338/g0045`, `1180339/g0005`, and `1180340/g0067`. Aggregate Job `1180341` is correctly `PENDING (Dependency)` on R5 only.
- A file-scoped scan of top-level and child `.out/.err` logs found no `Traceback`, OOM, non-finite loss, `FAIL`, or recurrence of `5920 vs 1184`.
- The first visible training consumers are R2/R3 core exact-uniform official-60 and R2/R3 adapted R4Q5-G0 P0; both reached `Training Starts` and epoch 0. No terminal mAP, terminal checkpoint, completion artifact, or logged successful optimizer-step line exists yet, so corrected first-batch success is not overclaimed.
- Status remains `experiment_running`, not `empirically_supported` or `paper_ready`.
- Immediate success check: corrected soft/hard P0 arms execute finite optimizer updates without the CUDA indexing failure.
- Main performance check: learned official-validation terminal EMA Avg-mAP must be compared against same-commit exact-uniform, with raw IoU-wise mAP shown first.
- Cost check: report probe, selector, heavy backbone, detector, total latency and peak memory; claim savings only from same-backend paired measurements.
- Final claim check: report multi-seed K384/K256 and second-backend results; no R0 93--94 diagnostic value may enter the main paper table.

## Monitoring protocol

Each evidence-changing check records `squeue`, `sacct`, bundle logs, terminal evaluations, selected K/max-hole, Traceback/OOM/non-finite/FAIL scans, cost summaries and final hashes. Update this page and `research-wiki/log.md` only when Job state, failure, mAP, cost or claim status changes.

- Active heartbeat automation: `duca-21-00-full-progress-report` (`DUCA R0-R5 cd68d89 progress`).
- Frequency: hourly; status `ACTIVE`; target is this task thread.
- The automation is bound to exact commit `cd68d89`, the formal root and Jobs `1180336--1180341`, and must read this page plus `query_pack.md`, `anti_repetition.md`, the final model contract, version registry and log before every check.
- No-change heartbeats remain short. Any state transition, failure, terminal mAP, cost result or claim verdict must be written here and appended to `research-wiki/log.md` in the same turn.

## Monitoring ledger

| Time (+08:00) | Job state | Training/result evidence | Decision |
|---|---|---|---|
| 2026-07-22 20:55 | `1180336--1180340` Slurm-accepted; `1180341` dependency pending | No corrected run output yet | Register as `experiment_running`; wait for execution |
| 2026-07-22 21:04 | Five model bundles `RUNNING`; aggregate dependency pending | Core uniform official-60 and adapted R4Q5-G0 P0 entered epoch 0; scoped error scan empty; no terminal artifact/mAP | Continue monitoring; no empirical claim |
| 2026-07-22 22:13 | Five model bundles remain `RUNNING`; aggregate remains dependency-pending | Exact-uniform official-60 and adapted R4Q5-G0 P0 both entered epoch 9 with finite logged losses and successful updates. The uniform arm used K384 and about 8.6 GB; P0 skipped the detector as intended and used about 3.7 GB. Isolated AMP replay events stayed recoverable. Full scoped scan found no Traceback, OOM, non-finite loss, ValueError or FAIL; no terminal mAP yet. | Healthy training progress only; status remains `experiment_running` |
| 2026-07-22 23:15 | `1180336--1180340` remain `RUNNING`; `1180341` remains dependency-pending | Exact-uniform official-60 and adapted R4Q5-G0 P0 reached epoch 17 with finite losses and checkpoints through epoch 14. All five top-level stderr files contain zero lines and the scoped error scan remains empty. R0/R1 bootstrap reached 400/1000, while the independently submitted R4 and R5 bundles are each still repeating their own R0 bootstrap at 200/1000 before entering later model stages. No terminal epoch-59 EMA mAP exists. | Model runs are healthy, but repeated bootstrap is a real execution bottleneck rather than model evidence; do not interpret the occupied R4/R5 GPUs as completed R4/R5 training. |
