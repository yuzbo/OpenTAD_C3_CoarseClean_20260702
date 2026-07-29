---
type: experiment
node_id: exp:georoute-estimator-representation-pilot-v1
title: "GeoRoute estimator/representation exploratory pilot v1"
idea: idea:geo-route-adatad
stage: experiment_running
status: amp_repair_cuda_pass_five_arms_running_pending_incomplete_closeout
verdict: pending
confidence: medium
commit: cbe0a08218a2f4550960f7c832f88c8cf77757c1
jobs: 1203707-1203720
updated: 2026-07-29
---

# GeoRoute estimator/representation exploratory pilot v1

## Question

On one frozen development-only seed, which effect explains detector utility:
PL versus ST credit assignment, geometry representation at fixed support,
geometry representation at learned ROI support, or ROI versus residual token
support under matched PL and representation-off settings?

## Frozen protocol

Six arms use exact `K=64`, seed `3407`, 20 epochs, absolute pretrained
position enabled, uniform-selected pooling, final-only checkpoints, and the
same immutable inputs. Four contrasts are preregistered. The Slurm DAG is six
parallel P0 leaves, one afterany P0 finalizer, six afterany fail-closed wrappers
that require the sealed PASS P0 suite before cell creation, and one afterany
non-promoting finalizer over all terminal leaves.

## Current state

The first runtime `02b6efe7` passed remote Linux tests `108/108`, but its six P0
Jobs `1203380`--`1203385` all failed mechanically before model reports and no
training ran. Jobs `1203386`--`1203392` were canceled after becoming
impossible; finalizer `1203393` exposed JSON-key-order validation and wrote no
finalization. The namespace is immutable failure evidence. A v2 repair is
implemented: module-mode P0 and early root bootstrap; Slurm-job-scoped
`127/8`, kernel-assigned ports, 120-second readiness and hashed failure
diagnostics with whole-process-group cleanup and same-leaf node revalidation;
arm-set normalization independent of JSON key order; fail-safe P0/final
control receipts; and the all-terminal fail-closed DAG above. N16R4 probes
`1203460/1203461` rejected `--resv-ports=2`, so it is not used.

Fresh runtime `cbe0a08218a2f4550960f7c832f88c8cf77757c1` was synced through
the academic proxy with exact HEAD/origin/clean-tree parity and passed remote
Linux tests `118/118`. Same-node gate Jobs `1203689/1203690` completed `0:0`
concurrently on `g0005` with distinct job-scoped hosts and four distinct
runtime TCPStore ports. Under the fresh run root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`,
P0 Jobs `1203707`--`1203712` and finalizer `1203713` completed `0:0`.
Sealed suite
`00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
is `PASS_MECHANICAL_ONLY`. Six training/evaluation leaves
`1203714`--`1203719` started in parallel. Job `1203715`
(`residual_pl_rep_off`) then hard-failed on real batch 0 after eight AMP
retries (scale `32768` down to `256`) and produced no checkpoint or metric.
The other five leaves remain running; closeout `1203720` must seal INCOMPLETE
after all leaves become terminal. Root cause is the FP16 production-horizon PL
temporal sum overflowing before GradScaler can recover, a case missed by the
float64 `T=1` KAT and synthetic P0. An FP32 likelihood/reduction repair plus
mandatory `T=384/N=220/K=64` AMP P0 KAT is implemented locally for a future
fresh source and full restart. Repair commit
`30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` is now proxy-synced in an exact
clean snapshot and passed the complete remote Linux suite `120/120`.
Standalone CUDA KAT Job `1203873` completed `0:0`: half source, FP32
likelihood/loss, objective magnitude `128637.0234375 > 65504`, and finite
scaled gradients. Receipt internal/file SHA-256 values are
`7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
and
`75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
This is `tested` numerical-correctness evidence only. The current experimental
node still has a hard-failed arm and is not `empirically_supported` or
`paper_ready`.

A fresh history-free agent independently audited the raw Pro review and its
absorption, frozen six-arm contract, no-leak paths, PL AMP repair, P0/finalizer
logic, and all-or-none 14-job deployer. It returned
`DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`. This closes the need for another Pro
discussion before the minimal pilot, but it does not bypass old closeout,
submit capacity, fresh per-arm P0, or all-six completion. The audit is archived
at
`docs/methods/reviews/2026-07-29-georoute-estimator-pilot-independent-agent-audit.md`.

## Claim boundary

For the current namespace, the hard-failed arm means finalizer `1203720` may
emit only `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with an empty contrast
set. A future fresh namespace may emit descriptive single-seed contrasts only
if all six arms pass. No finalizer can emit a winner, reuse the historical
selector, promote P2/P3, open official test, or authorize an efficiency,
Geometry Zoom, or paper claim.

## Provenance

Protocol:
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
