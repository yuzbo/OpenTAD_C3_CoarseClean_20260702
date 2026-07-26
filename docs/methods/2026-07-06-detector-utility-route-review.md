# Detector Utility Route Review Absorption

Source:
`docs/methods/2026-07-06-detector-utility-route-review.raw.txt`

Status:
recorded and absorbed into the detector-aware TrueTime sparse TAD route.

## Accepted Judgement

GAS-VT and p_action strict ledgers remain necessary engineering guardrails, but
they are not the final paper contribution. The main route should be detector
utility supervised pre-backbone temporal acquisition:

```text
dense detector critic / value-of-observation
    -> low-cost acquisition policy
    -> strict sparse ledger or runtime hard selector
    -> sparse-aware AdaTAD / ActionFormer detector
```

The deployable selector must see only low-cost observable signals at test time.
Dense detector teacher outputs are train-time critic labels, not ground truth
and not deploy-time inputs.

## Route Name Candidate

Working method name:

```text
DUCA-TAD: Detector-Utility-Calibrated Acquisition for Temporal Action Detection
```

Preferred paper claim shape:

```text
Boundary-sensitive value-of-observation learning for sparse temporal action detection.
```

## Method Constraints Absorbed

Teacher utility should be described as detector critic value, not GT:

```text
We distill the dense detector's marginal value-of-observation into a deployable
low-cost acquisition policy.
```

Counterfactual utility should avoid per-frame brute-force masking. Prefer
interval or group-level counterfactuals around proposal supports, boundary
neighborhoods, uncertainty peaks, and process segments, then project utility
back to snippets or frames.

Utility should become signed rather than only positive foreground/action
utility. TAD sparse acquisition must preserve:

- action interior evidence
- boundary bracket evidence
- background suppression evidence
- proposal ranking evidence
- non-redundant evidence across local peaks

Dynamic budget needs calibrated marginal gain, not just per-video ranking.
Ranking is enough for fixed_384/fixed_768; dynamic budget requires comparable
gain across videos.

Sparse detector training must include true-time metadata:

- selected positions
- true dense-time axis
- cell width / visibility
- selected-axis to native-axis mapping
- physical-grid ActionFormer assignment

## Loss Direction

The target loss family should move toward:

```text
L =
  L_rank
  + L_gain_calib
  + L_budget
  + L_boundary_bracket
  + L_interior
  + L_cvar_utility_gap
  + L_sparse_distill
  + 1_stage3 * L_detector
```

Interpretation:

- `L_rank`: high detector-utility frames rank earlier.
- `L_gain_calib`: dynamic budget scores are comparable across videos.
- `L_boundary_bracket`: preserve evidence on both sides of boundaries.
- `L_interior`: preserve interior action support.
- `L_cvar_utility_gap`: penalize tail-risk holes in high-utility regions.
- `L_sparse_distill`: sparse detector mimics dense teacher proposals, logits,
  boundary distributions, or ranking.
- `L_detector`: added only after Stage2/3 are stable.

## Required Ablations Added

Add these to the experiment plan before paper claims:

1. Teacher utility upper bound, marked non-deployable.
   Directly use dense teacher utility top-k to answer whether the utility signal
   itself is valuable.

2. Observable selector versus teacher-feature selector.
   Compare low-cost observable features against dense-feature selector to measure
   the observability gap.

3. Positive-only utility versus signed utility.
   Test whether preserving boundary/background suppression evidence improves
   high-IoU localization stability.

## Stage Policy

Stage1 remains GAS-VT strict ledger baseline for engineering and attribution.

Stage2 is the main innovation:

- dense detector utility export
- low-cost detector-utility selector
- strict deployable ledgers
- fixed/dynamic mAP comparison

Stage3 should first be sparse-aware detector distillation, not premature full
end-to-end training.

Stage4 should use ST hard selector partial joint fine-tuning only after Stage2
and Stage3 beat GAS-VT or produce convincing detector mAP/stability evidence.

## Open Evidence Gaps

Do not claim the route is complete until these are resolved:

- Stage2 detector-aware mAP versus p_action-only and GAS-VT.
- Teacher utility top-k non-deployable upper-bound mAP.
- Observable selector versus teacher-feature selector gap.
- Signed utility versus positive-only utility ablation.
- Sparse detector distillation evidence.
- Stage3/4 full THUMOS high-IoU mAP, not only smoke gradient proof.
- Total compute accounting including scout, selector, sparse backbone, and head.

