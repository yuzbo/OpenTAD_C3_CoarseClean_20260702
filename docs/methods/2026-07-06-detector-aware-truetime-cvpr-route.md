# Detector-Aware TrueTime Sparse TAD Route

This note records the current direction after the Stage0/1 GAS-VT work, the
Stage2 detector-aware selector import, and the Stage3/4 TrueTime joint-selector
smoke route. It is intentionally a claim-locked engineering note, not a result
report.

## Core Position

The current p_action and GAS-VT strict-ledger routes are useful offline evidence
routes, but they are not strict end-to-end TAD training. They answer whether a
cheap actionness/transition signal can preserve enough temporal evidence for a
downstream detector. They do not prove that the detector itself has taught the
selector which observations matter for high-IoU temporal localization.

The CVPR-level story should therefore move from "actionness selects frames" to
"detector-aware temporal acquisition preserves boundary evidence for high-IoU
TAD under sparse pre-backbone observation budgets."

## Role of official_asformer

`official_asformer` is a strong and relatively heavy temporal scout. It should
be used as a strong actionness-transition reference or upper-bound scout, not as
the final low-cost acquisition model and not as the main novelty. Any efficiency
claim must account for total cost:

- scout encoder cost
- acquisition policy cost
- sparse backbone/detector cost
- TrueTime adapter and head cost

The paper story should avoid claiming backbone savings alone.

## Module Boundary

The method should be described as a pluggable temporal acquisition adapter:

```text
TemporalAcquisitionAdapter =
    ScoutEncoder
    + AcquisitionPolicy
    + SparseSampler
    + TrueTimeAdapter
```

The adapter outputs:

- `selected_frames`
- `selected_positions`
- `selected_mask`
- `acquisition_meta`

The detector consumes selected frames and true-time metadata without changing
the official mAP semantics.

## Four Stages

Stage0/1: Offline GAS-VT sparse ledgers to AdaTAD mAP.

- fixed_384, fixed_768, dynamic
- proves whether sparse ledger input can keep detector mAP alive
- still offline and not detector-aware

Stage2: Dense AdaTAD teacher utility to detector-aware acquisition.

- train split only
- teacher signals may include point responsibility, cls/reg utility, proposal
  saliency, and counterfactual utility
- val/test selector must not use teacher, GT, raw prediction cache, or evaluator
  artifacts
- answers whether AdaTAD teacher utility is better than p_action-only

Stage3: ST hard selector plus AdaTAD joint training smoke.

- forward path uses hard TopK sparse selected inputs
- backward path keeps a straight-through selector gradient path
- TrueTime metadata maps selected-axis detector predictions back to dense time
- ActionFormer target assignment must use `physical_grid_actionformer`; otherwise
  selected-axis outputs and dense-time GT are not a closed localization system
- current proof is a gradient smoke, not a full THUMOS mAP proof

Stage4: Curriculum/bilevel stabilization.

- dense teacher to selector pretrain
- frozen or partially frozen detector sparse warmup
- sparse detector full train
- ST joint fine-tune
- bilevel/fulltrain candidate only after sparse mAP evidence is available

## Claim Locks

Until full detector mAP is reported for the relevant variants, the following
claims are locked:

- no formal end-to-end claim
- no paper performance claim
- no runtime/FLOPs claim
- no deployment claim
- no mAP improvement claim

The strict phrase to use for the current implementation is:

> The selector does not localize boundaries; it preserves boundary evidence for
> the detector.

## Required Metrics

Ledger and acquisition integrity:

- selected_count strictness
- max_gap and p95_gap
- boundary_support at r1/r2/r4/r8
- action positive coverage
- dynamic budget distribution
- no uniform fill or uniform scaffold
- no val/test teacher, GT, raw prediction cache, or evaluator leakage

Detector evidence:

- mAP at official tIoU thresholds
- especially high-IoU mAP at 0.6 and 0.7
- dense baseline versus sparse fixed_384/fixed_768/dynamic
- p_action-only versus detector-aware teacher-utility selector
- offline ledger versus runtime/ST selector

Training stability:

- selector entropy and collapse rate
- selected_count variance
- detector loss and selector grad norm
- rank correlation between teacher utility and learned acquisition score
- compute accounting for scout + selector + sparse detector

## Implementation Status

Implemented in this branch:

- Stage2 detector-aware teacher-utility export and selector training route
- Stage2 detector-aware strict ledger validation and AdaTAD full-train config
- Stage3 TrueTime ST selector and ActionFormer detector-gradient smoke route
  with physical-grid ActionFormer assignment required by validator
- Stage4 fail-closed curriculum evidence validator

Not yet proven:

- detector-aware selector mAP beats p_action-only
- GAS-VT or detector-aware sparse ledgers improve high-IoU TAD localization
- ST joint training improves or preserves mAP in full THUMOS training
- runtime compute savings after scout cost is included
