---
status: designed
evidence_class: static_contract_only
date: 2026-08-17
---

# DUCA semantic dynamic cycle 2 — minimal change plan

Reuse `PCOTMRASPreBackboneFrameSelector` and its
`PCOTMRASBoundaryDifficultyTemporalFrameScout` action/boundary heads, then
route all six arms through the existing `tools/train.py` parser and official
AdaTAD/ActionFormer runner contract. Modify only the selector, six-arm config,
validator/launcher, focused tests, and this wiki/log entry. Add no new runner,
cache, feature flag, checkpoint, dataset, or remote execution layer.

The semantic-indirect fixed arms differ by acquisition score: actionness-only
uses action probability; actionness+boundary reserves boundary-ranked frames
before action support. The dynamic headline derives per-window K from those
predictions and records actual heavy-path input count. Non-contiguous picks keep
original dense positions and remap before detector threshold/NMS. FIT/CAL/HOLD
are explicit disjoint manifests validated without opening data.

Focused checks: selector/config import and six-arm isolation, physical-time
metadata before NMS, real heavy input/executed-K accounting, manifest
disjointness, checkpoint contract, launcher PRE_RUN parser invocation, and the
repository py_compile/pytest checks. No data/GPU/Slurm/remote training or
performance claim is permitted in this cycle.
