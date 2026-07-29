---
type: experiment
node_id: exp:georoute-estimator-representation-pilot-v1
title: "GeoRoute estimator/representation exploratory pilot v1"
idea: idea:geo-route-adatad
stage: implemented
status: pending_remote_linux_test_and_p0
verdict: pending
confidence: medium
commit: pending_clean_pilot_commit
jobs: none
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
parallel P0 leaves, one afterok P0 finalizer, six parallel training/evaluation
leaves, and one afterany non-promoting finalizer.

## Current state

The contract, launchers, receipts, full population/telemetry integrity checks,
and tests are implemented locally. Remote Linux tests, CUDA P0, and all
training arms are pending; therefore this node is not `tested`,
`experiment_running`, `empirically_supported`, or `paper_ready`.

## Claim boundary

The finalizer can emit descriptive single-seed contrasts only. It cannot emit
a winner, reuse the historical selector, promote P2/P3, open official test, or
authorize an efficiency, Geometry Zoom, or paper claim.

## Provenance

Protocol:
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
