# DUCA Dynamic-K RIME Four-Stage Adjudication

## Status

- Date: `2026-07-27`
- Approval: `user_approved`
- Design: `designed`
- Implementation: `implemented`
- Focused local verification: `tested`
- Remote authoritative verification: `pending`
- Deployment: `pending_submission`
- Empirical support: `not_yet_empirically_supported`
- Paper status: `not_yet_paper_ready`

## Objective

Test whether cheap coarse evidence can allocate real pre-backbone temporal
compute and protect high-IoU localization through risk-aware, exact-K,
monotone physical-time acquisition.

## Frozen experimental DAG

```text
remote code gate
    -> Phase 1 execution/geometry/cost closure
    -> Phase 2 U-mixed-K + O1/O2/O3/O4 + K384/K192 protocols
    -> Phase 3 six train arms + U-same-K + development seal
    -> Phase 4 authorization
    -> 2 detectors x 2 budget panels x 3 fresh seeds
    -> formal matrix seal
```

Dense ActionFormer and TriDet references are trained in parallel after the code
gate and become prerequisites for Phase 3/4 cost evidence.

## Algorithms and trainable outputs

| Phase | Output | Model status |
|---|---|---|
| 1 | exact-K physical decoder, coordinate inverse, controls, ledger/cost profiler | algorithm/infrastructure |
| 2 | `U-mixed-K`, cross-fitted targets, counterfactual gates, frozen price protocols | trainable baseline + protocol |
| 3 | `RIME-full` and causal ablation family | first new model candidate |
| 4 | fresh-seed ActionFormer/TriDet replicas and formal evidence matrix | validation of the same candidate |

## Formal arms

Phase 3 train arms:

- `RIME-full`;
- `U-fixed`;
- `F-bound`;
- `D-no-risk`;
- `AdapTok-TAD`;
- `D-shuffle`.

`U-same-K` is evaluation-only and reuses the RIME checkpoint with an immutable
exact-K replay. It is not a separately trained model.

Phase 4 trains `RIME-full` and `U-fixed` for each formal cell. `U-same-K`
remains evaluation-only from the corresponding RIME source.

## Corrected panel semantics

| Panel | Allocation | Allowed claim |
|---|---|---|
| K384 | frozen-price dynamic K | dynamic allocation and learned positions |
| K192 | forced exact K=192 | learned positions at the floor budget only |

## Corrected cost semantics

Candidate and `U-same-K` must consume the same realized per-video K sequence
within tolerance. `U-fixed` is retained for accuracy, not variable-K cost
matching. Candidate, matched control, and dense reference all use measured
full-stack latency, throughput, energy, and peak memory.

## Provenance requirements

- exact clean Git commit;
- positive Slurm job IDs and explicit dependencies;
- immutable external run roots;
- split manifest and assignment hashes;
- checkpoint SHA-256 plus trained-commit evidence;
- Phase-2 receipt and both formal protocol hashes;
- Phase-4 authorization bound to the same Phase-2 receipt and protocol;
- checkpoint audit and terminal identity;
- official-final video set opened only after Phase-3 GO.

## Pre-registered protocol inputs

These values are frozen in the submitting Git commit rather than selected from
Phase-2 or later outcomes:

| Input | Frozen value | Interpretation |
|---|---:|---|
| candidate K / cost proxy | `192,256,384,512` | exact heavy RGB frames; no Kmax padding |
| primary / floor target | `384 / 192` | K384 dynamic panel; K192 learned-position-only panel |
| decoder family | `weak_overlap` | must pass the independent-oracle regret gate |
| risk weight / threshold | `1.0 / 0.35` | finite-budget utility and feasibility rule |
| O4 max Brier / ECE | `0.25 / 0.10` | pre-test calibration ceilings |
| O4 min low-risk coverage | `0.25` | rejects a vacuous always-fallback risk head |
| O4 max low-risk failure | `0.35` | observed failure may not exceed the accepted-risk ceiling |
| short / medium duration | `2.0 s / 8.0 s` | inherited from the registered DUCA physical P3 strata |

## Deployment ledger

Not yet submitted. Fill this section only with confirmed scheduler output:

- implementation commit: `pending`
- external run root: `pending`
- submission manifest: `pending`
- code-gate job: `pending`
- Phase-1 job: `pending`
- Phase-2 job: `pending`
- Phase-3 controller/job set: `pending`
- Phase-4 controller/matrix: `pending`

## Stop rules

- Any failed hash, split, coordinate, exact-K, no-padding, or cost-match check
  stops the dependent stage.
- Phase-3 NO-GO prevents official-final evaluation.
- Phase-4 formal failure forbids the general-plugin paper claim; negative
  evidence is retained rather than silently rerouted.
