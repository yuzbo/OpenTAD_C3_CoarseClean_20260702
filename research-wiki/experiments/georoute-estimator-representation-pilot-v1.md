---
type: experiment
node_id: exp:georoute-estimator-representation-pilot-v1
title: "GeoRoute estimator/representation exploratory pilot v1"
idea: idea:geo-route-adatad
stage: implemented
status: mechanical_failure_repair_pending_remote_linux_and_fresh_p0
verdict: pending
confidence: medium
commit: 02b6efe71bd9c62de304467adf0981799eba6b1e_failed_repair_commit_pending
jobs: 1203380-1203393
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

Exact runtime `02b6efe7` passed remote Linux tests `108/108`, but its six P0
Jobs `1203380`--`1203385` all failed mechanically before model reports and no
training ran. Jobs `1203386`--`1203392` were canceled after becoming
impossible; finalizer `1203393` exposed JSON-key-order validation and wrote no
finalization. The namespace is immutable failure evidence. A v2 repair is
implemented locally: module-mode P0 and early root bootstrap; Slurm-job-scoped
`127/8`, kernel-assigned ports, 120-second readiness and hashed failure
diagnostics with whole-process-group cleanup and same-leaf node revalidation;
arm-set normalization independent of JSON key order; fail-safe P0/final
control receipts; and the all-terminal fail-closed DAG above. N16R4 probes
`1203460/1203461` rejected `--resv-ports=2`, so it is not used. A fresh clean
commit, remote Linux tests, real same-node gate, and all six P0 leaves are
pending; therefore this node is not `experiment_running`,
`empirically_supported`, or `paper_ready`.

## Claim boundary

The finalizer can emit descriptive single-seed contrasts only. It cannot emit
a winner, reuse the historical selector, promote P2/P3, open official test, or
authorize an efficiency, Geometry Zoom, or paper claim.

## Provenance

Protocol:
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
