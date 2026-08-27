---
type: experiment
node_id: exp:phystime-performance-drop-diagnosis
title: "PhysTime-AdaTAD 1.0 performance-drop diagnosis"
idea: idea:phystime-adatad-1
status: empirically_supported
verdict: architecture_and_resolution_confound_confirmed
confidence: high
metrics: "Raw values and artifact paths are maintained only in docs/evaluation/results.md."
provenance: "docs/evaluation/phystime-performance-drop-diagnosis.md"
added: 2026-07-12T14:20:00+08:00
---

# PhysTime Performance-Drop Diagnosis

## Confirmed

- final official results are reproducible and use real dataset GT;
- no training collapse, double time conversion, missing test windows, or score self-normalization explains the gap;
- current three-head comparison changes detector capacity and temporal context in addition to coordinates;
- raw absolute time dominates learned query representation;
- coarse support attention covers many observations but effectively uses very few;
- PhysTime candidate density and short-action supervision are thinner;
- after a correct high-IoU match, PhysTime boundary quality is competitive with the controls, but coverage and ranking are weaker.

## Decision

Freeze PhysTime 1.0 as a negative baseline. Do not tune around the symptoms. The next scientific gate must keep ActionFormer-level capacity/context and candidate count matched, then change one physical-time component at a time.

## Evidence

- `docs/evaluation/results.md`
- `docs/evaluation/phystime-performance-drop-diagnosis.md`
- `docs/evaluation/EXPERIMENT_AUDIT.md`

## Connections

[AUTO-GENERATED from graph/edges.jsonl]
