---
updated: 2026-07-08
status: active
scope: Absorbed final-design review for DUCA as an online hard-budget acquisition plugin
out-of-scope: Claiming implementation is complete, reporting detector mAP, or endorsing offline ledger as final method
---

# DUCA Online Plugin Final-Design Review Absorption

Raw record:

- `docs/methods/reviews/2026-07-08-duca-online-plugin-final-design-review-raw.txt`

Raw record SHA256:

```text
A61226E4D33399240C3F52745E8081110E24760260E66E5DACC650EFC3C6EA37
```

## Core Verdict

The final paper should not deliver a new full TAD detector. It should deliver an
online temporal acquisition plugin placed before an existing TAD detector such
as AdaTAD or ActionFormer.

The final method should be:

```text
DUCA-TAD: Detector-Utility-Calibrated Online Acquisition for Sparse Temporal Action Detection.
```

The final plugin must run inside the detector forward path, produce original-time
`selected_positions`, and make the detector consume exactly those positions in a
single forward pass.

Offline strict ledgers are not the final method. They are allowed only as audit
and reproducibility artifacts. If the paper's main story remains
`export ledger -> train AdaTAD`, the method will be judged as an engineering
pipeline rather than an online acquisition plugin.

## Final Method Target

The target final model is an online detector-side adapter:

```text
low-cost observations / frozen actionness source
-> DUCA acquisition adapter
-> hard selected_positions, original-time, <=384
-> sparse gather + SparseTemporalGrid
-> AdaTAD / ActionFormer single forward
-> TAD predictions
```

Required properties:

- `forward_acquire()` exists and is part of the detector path.
- Inference reads no GT, teacher utility, raw prediction, prediction cache, or
  oracle boundary.
- The acquisition budget is measured in detector-consumed temporal observations.
- `selected_positions` are the actual positions gathered by the detector.
- `selected_centers` and `radius` are metadata/intermediate decisions only.
- The same acquisition interface works for AdaTAD and ActionFormer.
- Main forward preserves original-time coordinates rather than selected-rank
  time.

## Final Training Design

The recommended final training design is:

```text
train-only teacher utility warm-up
-> hard-forward straight-through joint fine-tune
```

This is not a pure offline selector pipeline and not a purely soft top-k method.

Rationale:

- Warm-up prevents the selector from starting from degenerate random exploration.
- Hard forward ensures the detector sees only the selected observations used at
  inference.
- Straight-through or surrogate gradients let detector loss affect the selector.
- Joint fine-tuning makes DUCA a detector-facing acquisition adapter rather than
  a frozen external sampler.

The final paper may use train-only detector utility as warm-up or auxiliary
distillation only. It must not be the test-time source of selected positions and
must not appear in validation/test inputs or ledgers.

## Zero-Shot / No-Training Actionness Branch

The review explicitly adds a no-THUMOS-training actionness branch, but does not
allow it to become the main method.

Roles:

- `DUCA-Frozen` frozen input branch;
- actionness top-k baseline;
- auxiliary input or warm-start source for `DUCA-Adapted`;
- not the final main method.

Recommended frozen sources:

- X-CLIP zero-shot prompt actionness: preferred DUCA-Frozen main signal.
- ActionCLIP zero-shot prompt actionness: alternative zero-shot video-text
  source.
- SlowFast / VideoMAE Kinetics logits: medium-cost pretrained actionness proxy.
- Motion energy / frame difference / entropy: lightest source, but weakest.
- InternVideo / large VideoMAE / large video foundation models: teacher or
  upper-bound only because cost weakens efficiency claims.

Zero-shot actionness should be computed from action/background prompt scores or
from Kinetics logits without THUMOS label calibration. Prompt lists, model
checkpoint, temperature, and calibration rules must be frozen and hashed before
experiments.

## Required Model Components

The review expects concrete implementation around these components:

- `ZeroShotActionnessSource`: frozen video-text or pretrained actionness source.
- `DucaAcquisitionAdapter`: lightweight temporal encoder plus selection,
  radius, boundary, utility, and optional budget heads.
- `hard_topk_st`: hard top-k forward with straight-through/surrogate backward.
- `budgeted_center_radius_decode`: deterministic strict-budget decoder.
- `SparseTemporalGrid`: selected positions plus original-time metadata.
- `gather_selected_observations`: detector input gather by selected positions.
- `duca_forward_train`: online acquisition + sparse detector forward + losses.
- `duca_forward_test`: teacher-free online acquisition at inference.
- `duca_losses`: detector, budget, boundary, hole, redundancy, radius, entropy,
  and optional train-only utility distillation losses.

The code skeleton in the raw record is method-design guidance. It is not proof
that the repository currently implements the design.

## Method Contract

The final method contract is stricter than prior ledger-based routes:

- Budget unit: detector-consumed temporal observations.
- Main setting: 768 candidate temporal observations, at most 384 consumed.
- `selected_positions`: sorted, unique original dense-time indices.
- `selected_positions` are the only positions gathered and consumed.
- `selected_centers` and `context_radius` are metadata; they are not the budget.
- Dynamic K is allowed only with hard max K, matched-average fixed baselines,
  selected-count histograms, and violation reporting.
- Short windows use `selected_count <= valid_len`, with no duplicate padding.
- Teacher utility scope is train-only and loss-only.
- Frozen sources must declare model ID, checkpoint hash, prompt list, temperature,
  and no target-label calibration.
- The detector head, assignment, decode, and NMS must not be silently modified.
  If modified, uniform/random baselines must use the same detector changes.
- Cross-detector API must expose `adapter.acquire()` and detector
  `forward_sparse()`/equivalent for AdaTAD and ActionFormer.
- Validator must fail closed on missing fields, budget violations, dense
  fallback, uniform fallback, selected-rank coordinate confusion, or teacher keys
  in validation/test artifacts.

## Experimental Commitments

Final experiments must prove the method is not an engineering pipeline:

- Dense AdaTAD and dense ActionFormer anchors.
- Uniform fixed 384 and random fixed 384 with multiple seeds.
- p_action top-k and delta-p_action/boundary proxy.
- zero-shot X-CLIP/ActionCLIP actionness top-k.
- Kinetics SlowFast/VideoMAE actionness top-k.
- C3, GAS-VT, and lattice baselines.
- DUCA-Frozen.
- DUCA-Adapted without teacher warm-up.
- DUCA-Adapted with teacher warm-up.
- DUCA online plugin without joint fine-tune.
- DUCA online plugin with hard-forward ST fine-tune.
- center-only, center-radius, no radius, fixed radius, learned radius.
- no boundary loss, no hole loss, no teacher utility, no actionness input.
- selected-rank decode versus original-time decode.
- AdaTAD and ActionFormer transfer.
- dynamic K versus matched-average and matched-latency fixed K.

Mandatory metrics:

- average mAP;
- mAP@0.5, mAP@0.6, and mAP@0.7;
- high-IoU drop/gain relative to dense;
- selected count and budget violation rate;
- browser/selector, gather, backbone, detector/NMS, and total inference latency;
- training cost, teacher warm-up cost, and zero-shot source cost;
- boundary support;
- short-action recall;
- action-local max hole and p95 hole;
- uniform similarity;
- actionness-interior over-selection;
- selector entropy and collapse diagnostics;
- no-leak audit pass/fail.

## Claim Limits

If only offline ledger is implemented:

```text
Claim limit: audited offline sparse acquisition protocol.
Do not claim final online plugin or end-to-end acquisition.
```

If online plugin is implemented but not end-to-end/joint:

```text
Claim limit: teacher-free online acquisition adapter with frozen or separately
trained selector.
Do not claim detector-loss-trained acquisition.
```

If online hard-budget joint training is implemented:

```text
Claim: DUCA-TAD is an online, hard-budget, detector-utility-calibrated temporal
acquisition plugin trained with detector loss and strict original-time sparse
consumption.
```

## Routes Demoted

The following must not be treated as final-paper main methods:

- C3 coarse classifier as the primary contribution;
- p_action top-k variants;
- GAS-VT fixed-ledger route;
- lattice/move heuristic searches;
- heuristic radius/gap repair as the main source of gain;
- exported ledger pipeline as the method body;
- selected-rank decoding as the main result;
- large foundation-model actionness as the main efficiency method.

They remain valid as baselines, frozen sources, teacher/upper-bound diagnostics,
or failure analysis.

## Local Interpretation

This absorption supersedes the weaker position that a task-adapted strict ledger
pipeline is enough for the final paper. The current repository can still use
strict ledgers to debug and audit, but the final method target is now an online
plugin with hard selection in the detector forward path and detector-loss-driven
selector training.
