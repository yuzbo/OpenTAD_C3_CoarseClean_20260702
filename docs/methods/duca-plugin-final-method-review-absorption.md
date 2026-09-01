---
updated: 2026-07-08
status: active
scope: Absorbed final-method review for the DUCA-style plug-in temporal acquisition target
out-of-scope: Claiming detector mAP results, implementing code, or marking the method paper-ready
---

# DUCA Plugin Final-Method Review Absorption

Raw record:

- `docs/methods/reviews/2026-07-08-duca-plugin-final-method-review-raw.txt`

Raw record SHA256:

```text
CF8183CEAADE93F9D92570DDD2ED2205E38907A80F7D4D57C0535DB3B0226924
```

## Core Verdict

The final goal should not be C3, p_action top-k, GAS-VT, lattice replacement,
heuristic radius, or gap-repair tuning. Those routes remain useful only as
baselines, diagnostic ablations, initialization signals, or cheap input features.

The final goal should be:

```text
Learn a task-adapted, teacher-free-at-inference detector-utility temporal
acquisition plugin that selects at most 384 detector-consumed original-time
temporal observations from a 768-grid window, improving sparse TAD while
protecting high-tIoU localization.
```

The preferred method name is `DUCA-TAD` or `DUCA-CR`, interpreted as a
detector-utility-calibrated temporal acquisition plugin, not as a ledger
pipeline.

## Chosen Main Method

The recommended main method is a strengthened version of:

```text
Task-adapted detector-utility plug-in selector
```

This corresponds to candidate B in the review. It is preferred over:

- frozen zero-shot selector: elegant, but too weak and hard to beat strong
  uniform baselines;
- end-to-end selector-detector: elegant and innovative, but currently too
  unstable and high-risk as the first paper method;
- offline learned ledger: reproducible, but too pipeline-like to be the method
  story.

The paper should describe a policy/interface:

```text
low-cost observable inputs -> DUCA selector -> selected_positions ->
AdaTAD/ActionFormer single detector forward
```

The ledger should be described as a reproducibility and no-leak audit artifact,
not as the algorithm itself.

## Required Method Contract

The main method is only credible if all of the following hold:

- `selected_positions` are the positions truly consumed by the detector.
- The budget unit is detector-consumed temporal observations, not centers,
  metadata, snippets, or raw frames.
- The selected positions are sorted, unique original dense-time coordinates.
- `len(selected_positions) <= 384` for the main 768-grid setting.
- Validation/test selection does not use GT, teacher utility, raw predictions,
  prediction caches, oracle boundaries, or post-hoc detector outputs.
- The detector runs a single forward pass on the selected observations.
- High-IoU changes are explained with boundary support, short-action recall,
  and action-local hole diagnostics, not only average mAP.
- Selected-rank decoding is an ablation only; the main result must preserve
  original-time coordinates.

## Training Mode

The recommended first-paper training mode is not direct end-to-end training.
It is:

1. Train or use a dense AdaTAD teacher on the training split only.
2. Export train-only detector utility/responsibility.
3. Train the DUCA selector with utility coverage, boundary support, strict
   budget, radius cost, hole, and auxiliary action losses.
4. Freeze the selector and generate no-leak train/val/test deploy
   `selected_positions`.
5. Train AdaTAD or ActionFormer with those selected positions, preserving
   original-time decoding.

End-to-end or straight-through fine-tuning is a second-stage enhancement only.
It should be attempted after the offline plug-in version beats the current
strong sparse baselines, especially the old lattice best, and does not degrade
mAP@0.6 or mAP@0.7.

## THUMOS Training Interpretation

Training the selector on THUMOS is acceptable only if the paper does not claim a
universal frozen plug-and-play selector. The correct wording is:

```text
task-adapted acquisition plugin trained on the training split of each TAD
dataset, analogous to a proposal generator or detector head.
```

The forbidden wording is:

```text
dataset-agnostic plug-and-play selector
```

unless a frozen selector is shown to transfer across datasets and detectors.
The review recommends reporting both a frozen setting and a task-adapted setting
to reduce this criticism.

## Teacher Utility Policy

Dense detector utility is recommended because without it the method risks
collapsing into actionness or boundary heuristics. It is allowed only as
train-only supervision.

Required safeguards:

- a train-only utility manifest;
- recursive no-leak scans for val/test sources and ledgers;
- inference input whitelist;
- explicit reporting of teacher export cost and selector training cost;
- no teacher, GT, raw prediction, prediction cache, or oracle payload in
  validation/test ledgers.

## Experimental Matrix

The minimum main matrix should include:

- dense AdaTAD and dense ActionFormer anchors;
- uniform fixed 384;
- random fixed 384 with multiple seeds;
- p_action top-k;
- delta-p_action or boundary proxy;
- GAS-VT fixed 384;
- lattice best;
- DUCA-Frozen;
- DUCA-Adapted fixed 384;
- DUCA-Adapted dynamic K with hard max budget;
- dynamic matched-average-K fixed control.

Mandatory ablations:

- selected-rank decode versus original-time decode;
- no utility, no boundary, no radius, fixed radius, learned radius;
- no repair versus decoder constraints;
- shuffled utility/source controls;
- cross-detector transfer to ActionFormer.

Mandatory metrics:

- average mAP;
- mAP@0.6 and mAP@0.7;
- selected count and effective budget;
- browser, gather, backbone, detector, NMS, and total latency;
- boundary support;
- short-action recall;
- action-local max hole and p95 hole;
- uniform similarity;
- radius distribution;
- leakage validator pass/fail.

## Routes To Stop Treating As Mainline

Stop treating the following as potential final methods:

- `p_action` top-k variants;
- C3 coarse classifier as the main contribution;
- GAS-VT fixed384 as the main contribution;
- move25/move50/move75 lattice searches;
- heuristic radius written into the ledger but not consumed by the detector;
- gap-repair rule stacking;
- 384 centers plus extra expanded positions as a pseudo-budget;
- selected-rank decode as the main result.

They can remain as baselines and failure evidence.

## First-Week Gate

The minimal next closed loop should be:

```text
DUCA selector checkpoint
-> no-leak deploy selected_positions ledger
-> AdaTAD consumes selected_positions
-> full train/eval
-> mAP + high-IoU + boundary diagnostics
```

The gate requires:

- `selected_positions` are truly consumed;
- selected count is at most 384;
- validation/test ledgers contain no teacher, GT, raw prediction, or cache;
- original-time decoding is used;
- performance beats p_action fixed384;
- preferably performance beats the old lattice best and does not degrade high
  tIoU metrics.

If this fails, the claim should shrink to a diagnostic/no-leak sparse
acquisition protocol or a task-adapted selector that improves over actionness
but not over strong sparse baselines.

## Claim Policy

Allowed cautious wording:

- DUCA is a task-adapted sparse temporal acquisition plugin for TAD.
- It is teacher-free at inference.
- It uses train-only detector utility to supervise acquisition.
- It outputs strict-budget original-time positions consumed by the detector.
- Ledgers are audit artifacts for reproducibility and leakage prevention.

Forbidden until verified:

- dataset-agnostic plug-and-play selector;
- end-to-end selector-detector training as the main claim;
- detector utility if the method is still p_action-only;
- raw-frame or full-backbone compute saving if the method only selects temporal
  observations on the AdaTAD/OpenTAD grid;
- dynamic-budget improvement without hard max K and matched-average-K controls;
- high-IoU protection without mAP@0.6/@0.7 and boundary diagnostics.

## Local Interpretation

This review supersedes any plan that treats C3/GAS-VT/lattice as the final
paper method. The current code may still use strict deploy ledgers to implement
the experiment, but the method narrative must be a plug-in acquisition policy
with a strict selector-detector interface.
