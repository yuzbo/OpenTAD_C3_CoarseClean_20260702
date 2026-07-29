---
type: experiment
node_id: exp:georoute-real-data-amp-stability-v1
title: "GeoRoute real-data AMP stability gate v1"
idea: idea:geo-route-adatad
stage: tested
status: sealed_hold_both_arms_violate_zero_skip
verdict: stability_gate_incomplete_hold
confidence: high
commit: 86ff1dde6ddb058ca9250f968972c255f19dab92
jobs: [1205033, 1205034, 1205035]
updated: 2026-07-30
---

# GeoRoute real-data AMP stability gate v1

## Question

After the localized PL temporal-normalization repair, can matched residual-PL
and residual-ST each complete 32 production-path real-data batches with finite
forward losses and no AMP skip at initial GradScaler scale `65536`?

This is a numerical-only gate. It cannot produce checkpoints, predictions,
metrics, evaluator outputs, official-test evidence or paper claims.

## Frozen execution

- exact source:
  `86ff1dde6ddb058ca9250f968972c255f19dab92`;
- root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_real_data_amp_stability_86ff1dde_20260730_0132`;
- matched arms: `residual_pl_rep_off` and `residual_st_rep_off`;
- seed `3407`, `K=64`, production native source, temporal reduction `mean`;
- 32 batches, initial scale `65536`, retry limit `0`, zero failed attempts;
- exact parent diagnostic and failed-pilot inputs receipt-bound;
- deployment reported all matched diagnostic inputs equal;
- clean remote GeoRoute suite: `124/124`.

## Terminal evidence

| Arm | Job | Batches reached | Successful scale-65536 updates before failure | Failure |
| --- | ---: | ---: | ---: | --- |
| residual PL | 1205033 | 3 | 2 | batch 3, `scout_score_function`, two nonfinite elements in `module.backbone.scout.residual_head.weight` |
| residual ST | 1205034 | 21 | 20 | batch 21, `detector`, 13 nonfinite elements in `module.rpn_head.cls_head.weight` and 77 in `module.rpn_head.reg_head.weight` |

Both arms had finite forward losses. Each observed one failed optimizer attempt
at scale `65536`, after which GradScaler selected `32768`; the frozen retry
limit `0` correctly aborted each stage. The failure receipts are self-hashed,
and there are no checkpoint, prediction, evaluator, metric or official-test
files.

PL receipt internal/file SHA-256:
`86b59bca66a20213e32793ca4a0e332c5958cf8c838f390c721f73bc6e7f2b21`
/
`6a0ebdf03a1aaa78bbc8e144918f21cbe7c5cfe71961877dbf15c57bdb27ed0b`.

ST receipt internal/file SHA-256:
`b417c2adf0268d6909831dfebdf4a50d0671101d3d228ac46d2c006f61815439`
/
`53f5d0c54bc78e66ba6a1390c5e87f2ba98ca2a8668058a6fd7a82a7ba7d9b54`.

Afterany finalizer `1205035` emitted
`INCOMPLETE_REAL_DATA_AMP_STABILITY_GATE /
STABILITY_GATE_INCOMPLETE_HOLD`, with `stability_gate_passed=false`,
`official_protocol_freeze_authorized=false`, empty performance metrics, and all
performance/test/paper guards false. Finalization internal/file SHA-256:
`aca065dc4d3dd32325909105ac461a9c32783a133b643bedfcfa8c48b0be1871`
/
`d62a017c656975495bb55e7059bd77b080c6f83d49a45a14156620566ea2100e`.

## Interpretation

The original PL-only catastrophic scale problem was substantially reduced: the
historical objective failed the first matched batch down through scale `256`,
whereas the temporal-mean candidate completed two batches at `65536` before one
localized backoff. This is useful numerical evidence, not accuracy evidence.

The frozen v1 rule nevertheless failed for both estimators, and ST failed in
the detector rather than its estimator path. Therefore v1 does not support a
PL-versus-ST verdict. It also cannot be treated as an official AdaTAD
comparability gate: exact official AdaTAD uses the default dynamic GradScaler
and does not require every update to remain at scale `65536`.

## Next admissible step

Design a distinct no-metric stability-v2 that matches the intended official AMP
semantics, uses an independent preregistered data order, keeps every
performance/test surface closed, and records bounded scaler adaptation plus a
stable tail. The sealed v1 rule and namespace must never be edited, resumed or
relabeled as passed. Only a passing replacement gate may authorize freezing a
separate official-comparable experiment.

## Paper boundary

No number from this gate belongs in a paper performance table. Paper eligibility
still requires:

1. an exact official AdaTAD reproduction;
2. same-recipe native-source dense and GeoRoute arms;
3. matched split/windows/padding, effective batch and updates,
   optimizer/scheduler/AMP/EMA, checkpoint selection and official evaluator/NMS;
4. preregistered disjoint multi-seed confirmation;
5. one sealed official-test opening;
6. selector-inclusive decode-to-NMS latency, peak memory, energy and actual
   token/route cost.

## Connections

Maintained in `research-wiki/graph/edges.jsonl`.
