# DUCA CellCF KILL / CARA redesign review absorption

## Source identity

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Model/training source of truth:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`
- Cost/evidence source of truth:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- User attachment:
  `C:/Users/skywalker/.codex/attachments/d5c1b391-e4d6-45bb-b835-7ec95a146b0c/pasted-text.txt`
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-19-1642f26-4ce69c8-cellcf-kill-cara-redesign-pro-review-raw.txt`
- Raw/archive size: `68,339` bytes
- Raw/archive SHA-256:
  `3FB06655193E7CF665BB37CF0701C2708139B15DF40AC2114742C23B19E292E7`

## Overall project verdict

`PARTIAL_ACCEPT_STRONG_DIAGNOSIS / HOLD_NEW_METHOD`

The review is substantially correct about the current CellCF feasible set,
coordinate contract, gradient semantics and evidence boundary. The project
therefore kills the current one-frame-per-uniform-cell CellCF as a paper main
method for boundary-adaptive allocation. It remains a diagnostic
uniform-phase/content-correction control.

This does not kill the broader DUCA question. The retained research objective
is still offline TAD task-aware temporal redundancy reduction:

1. train a low-cost action/background coarse model;
2. infer semantic state changes and uncertainty from its outputs/features;
3. allocate a strict sparse observation budget primarily for boundary and
   high-IoU localization utility;
4. evaluate with actual full-stack cost and a real TAD detector.

The proposed coverage-anchored residual allocation family is recorded as a
candidate named `DUCA-CARA`, not as an implemented or final method.

## Independently verified code and mathematical facts

### 1. CellCF cannot transfer quota across cells

At the exact model commit, `local_cell_deformation()` partitions the valid
window with `exact_uniform_cell_bounds()` and selects exactly one item from
each cell. A high-transition cell cannot receive a second item and a
background cell cannot release its item. It is therefore local phase/content
correction around uniform sampling, not boundary-adaptive budget allocation.

For `T=768, K=384`, an independent reconstruction of the committed rounding
and midpoint rules gives:

- 384 unique anchors;
- cell lengths: 382 cells of length 2, one of length 3 and one of length 1;
- maximum displacement from a cell anchor: 1 dense-grid index;
- exact-uniform maximum unselected hole: 2;
- a legal CellCF realization can have maximum unselected hole 3.

These facts support the review's central KILL argument.

### 2. Acquisition time and detector time are different in current CellCF

At `1642f26`, the selector gathers the actual `selected_positions`, while
local-cell acquisition sets `detector_grid_positions` to
`local_cell_anchor_positions`. The formal CellCF config also uses
`remap_gt_to_selected_axis=True`. Thus the current experiment changes observed
content around a fixed detector time lattice; it does not train/evaluate a
detector directly on the true irregular observation positions.

This is an intentional tested contract, not an accidental tensor bug, but it
confounds any claim that an `@0.7` change proves better physical boundary
placement.

### 3. Current CellCF has detector-derived supervision, not direct detector gradient

The formal config sets `detector_gradient_mode="none"`. Counterfactual
candidate losses are evaluated under `torch.no_grad()`, detached and converted
to a signed policy-ranking/distillation loss. The method is detector-aware,
but it is not direct end-to-end detector-gradient learning.

Any paper or diagram must use the correct wording:

- allowed: `train-time detector-derived hard counterfactual utility`;
- forbidden: `the detector loss directly backpropagates through the hard
  CellCF decision`.

### 4. The matched seed-0 result does not support CellCF utility

Terminal-EMA Avg-mAP:

| Arm | Avg-mAP |
|---|---:|
| exact-uniform | 63.8594 |
| transition-beta0 | 64.2755 |
| CellCF | 64.0610 |

CellCF is `-0.2145` below transition-beta0. Its isolated `mAP@0.7` increase
cannot establish a boundary mechanism with one seed and the coordinate
confound above. Transition-beta0's `+0.4161` over uniform is also only a
one-seed diagnostic, not a robust claim.

### 5. Existing ceiling scripts are not exact family ceilings

The committed feasible-set diagnostic selects the best member from a supplied
finite candidate set and labels itself `not_upper_bound`. The GT selection
path in the decomposition tool is heuristic. Neither may be renamed a CellCF
oracle or a deployable-family upper bound.

### 6. Physical-grid ActionFormer already exists

The repository has a strict physical-grid path in the ActionFormer head. It
reads `irregular_selected_positions`/`selected_dense_indices`, rejects
selected-axis GT remapping and maps priors/strides/ranges onto physical time.
Any redesign should reuse and test that path instead of creating another
irregular-time detector implementation.

This path is an OpenTAD project extension around official AdaTAD/ActionFormer
components. It must not be described as an untouched official AdaTAD model.

### 7. Cost evidence remains incomplete

Commit `4ce69c8` repairs and gates the profiler sample schema. It does not
contain a new trained model or prove full-stack savings. The two-sample schema
gate is compatibility evidence only; the repeated matched dense/uniform/new
method profile and break-even analysis remain missing.

## Accepted recommendations

1. Permanently stop calling one-per-cell CellCF boundary-adaptive allocation.
2. Retain it only as a matched phase/content-correction diagnostic control.
3. Before another long training run, compare exact feasible families:
   exact-uniform, one-per-cell, coverage scaffold plus adaptive residual,
   global exact-K/max-gap allocation and a privileged unrestricted reference.
4. Measure both boundary coverage and background budget release. A selector
   that cannot move quota between regions cannot satisfy the project goal.
5. Make acquisition positions, detector prior positions, GT coordinates and
   output coordinates share one physical-time contract.
6. Reuse the existing physical-grid ActionFormer implementation.
7. Keep fixed-K as the next bounded question. Dynamic MUST stays frozen.
8. Split decode, resize/preprocess, transfer, heavy backbone, head and NMS in
   the cost ledger, and compare against both bare uniform and dense input.
9. Do not unlock multiple detectors, frozen X3D/SlowFast, dynamic budgets or a
   large ablation matrix before the feasible-family and cost gates pass.

## Recommendations accepted only as hypotheses

### `G=3` and `192 + 192`

For `T=768, K=384`, a minimum positive balanced scaffold under a dense-index
maximum-hole target of 3 has 192 points, leaving 192 residual points. This is
a clean and interpretable candidate family, and it genuinely permits budget
concentration beyond uniform.

It is not yet the unique correct setting:

- `G=3` was chosen to match CellCF's theoretical worst hole, not by a measured
  accuracy-cost Pareto study;
- the project previously discussed gaps of 10 or 15 frames, so the unit must
  be frozen as dense-grid index, decoded frame index or original video frame;
- short windows and non-contiguous valid positions need explicit semantics;
- a fixed scaffold may still consume too much budget near easy background.

The exact ceiling must compare at least a small preregistered `G` set before
freezing the final family.

### CP-SAT ceiling

Exact integer optimization is suitable for linear geometric objectives such
as endpoint coverage, distance and background allocation under exact-K and
max-gap constraints. It does not make detector mAP a tractable exact CP-SAT
objective. Frozen-detector evaluation is a secondary diagnostic over generated
candidates, not a proof of a global detector-utility upper bound.

### Coarse/selector gradient ownership

The review proposes detaching the coarse temporal hidden features from every
selector loss so that the coarse model is trained only by binary BCE. This is
a defensible modular control, but it is not accepted as the only final
training design because it removes the collaborative-learning question.

The bounded comparison must preserve binary action/background supervision as
the coarse model's primary task and compare:

1. detached coarse evidence;
2. transition supervision allowed into the temporal trunk with calibration
   monitoring;
3. detector-derived utility updating only the selector scorer.

Direct detector-gradient wording is allowed only if a tested estimator really
propagates that gradient and passes hard one-swap finite-difference alignment.
Detached counterfactual distillation is safer, but it supports a different
claim.

### Numerical weights, cadence and publication gates

The proposed loss weights, temperature, four candidates, eight-update
counterfactual cadence, 600-update warmup, 60 epochs, three seeds and numerical
GO/KILL thresholds are useful preregistration candidates. They are not
deduced by the code or current experiment. They must be justified, frozen
before observing the corresponding result and reported as protocol choices,
not facts.

## Problems in the supplied illustrative patch

The code block is a blueprint, not a merge-ready patch:

1. residual top-k detaches policy scores, so hard selection itself supplies no
   gradient; training depends entirely on auxiliary transition/utility losses;
2. gap compliance is proved only on compact valid-order indices. A
   non-contiguous physical valid mask can violate the physical-time gap;
3. `minimal_balanced_scaffold()` calls one point the minimum when `L <= G`,
   although the mathematical minimum can be zero; the desired positive-anchor
   convention must be stated;
4. actual per-sample physical maximum hole is not recomputed in the returned
   contract;
5. add/remove candidate bounds and the sign convention
   `utility = L_base - L_candidate` need fail-closed validation;
6. the snippet does not implement the complete cadence, AMP replay,
   successful-update or physical-grid integration it specifies.

No code from the response is marked implemented or tested.

## Frozen next decision

The next scientific action is not another full training run. It is a bounded
feasible-set and coordinate audit:

1. implement exact geometric optimization/verification for the registered
   families;
2. verify every selected set is exact-K and physically max-gap compliant;
3. quantify boundary density, distance, both-endpoint coverage, background
   release and short-action coverage;
4. run a frozen-detector physical-grid secondary diagnostic;
5. measure the coarse/selector frontend cost;
6. authorize a small matched train only if the new family has materially more
   useful freedom than CellCF and a plausible full-stack break-even.

Status after absorption:

- current CellCF main-method role: `KILLED`;
- current CellCF diagnostic role: `tested_diagnostic`;
- DUCA research program: `REDESIGN`, not killed;
- DUCA-CARA: `discussed`, not implemented;
- dynamic MUST: `frozen`;
- C3/C4/C7 and paper readiness: unchanged and unproven.
