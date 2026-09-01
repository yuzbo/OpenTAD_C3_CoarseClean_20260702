---
updated: 2026-07-07
status: active
scope: Absorbed diagnosis shift from uniform-collapse explanation to detector-utility mismatch
out-of-scope: Claiming final causal proof, modifying running experiments, or replacing matched ablation evidence
---

# GAS-VT Diagnosis Shift To Detector Utility

Raw record: `docs/methods/reviews/2026-07-07-gasvt-diagnosis-shift-detector-utility-review-raw.txt`

Related notes:

- `docs/methods/2026-07-07-gasvt-paction-diagnosis-evidence.md`
- `docs/methods/2026-07-07-gasvt-plateau-paction-advantage-review-absorption.md`

## Absorption Verdict

This review materially changes the diagnosis priority.

The stronger interpretation is no longer:

> GAS-VT failed mainly because it became too uniform.

The better current interpretation is:

> GAS-VT selected relatively more action-positive/action-interior regions, but did not reliably preserve the boundary-neighborhood geometry, boundary brackets, and local continuous evidence that AdaTAD needs for high-IoU localization. PAction learned fixed_384 is stronger not because it copies raw p_action top-k more closely, but because it produces better detector-facing selection geometry: stronger boundary r1, much smaller p95 hole, and denser local support near useful transition regions.

I accept this as the current leading diagnosis. It is still not final causal proof until matched ablations are run.

## What Changed

The earlier explanation emphasized possible uniform-like collapse and hard repair. This review reframes the evidence:

- GAS-VT action coverage is higher than PAction learned.
- Yet GAS-VT boundary r1 is lower and p95 hole is much worse.
- PAction learned has lower overlap with raw p_action top-k than GAS-VT in the sampled analysis.
- Therefore PAction's advantage is not simply "more action" or "closer to top-k p_action".
- The core bottleneck is detector utility, especially boundary and high-IoU localization utility.

This is important because it prevents a wrong fix. If we only fight uniformity, we may still keep optimizing the wrong target: action/interior/coverage rather than detector-useful boundary evidence.

## Evidence Interpretation

The current evidence supports the following reading:

1. GAS-VT can reach about 40 mAP early because fixed_384 is half-density and p_action carries enough coarse action signal for low-IoU detection.
2. GAS-VT then plateaus because action-positive coverage is not the same as high-IoU localization utility.
3. PAction learned fixed_384 gains mostly in high-IoU mAP because it better preserves boundary-neighborhood and local support geometry.
4. Uniform-like repair remains a risk, but it is not the complete explanation for the current GAS-VT/PAction gap.
5. The main research route should become detector-utility-calibrated pre-backbone acquisition, not handcrafted GAS priors.

## Claim Boundary

| Claim | Status | Reason |
|---|---:|---|
| GAS-VT is failing only because it is too uniform | FAIL | Current evidence points more strongly to detector-utility mismatch |
| GAS-VT action coverage is enough for TAD | FAIL | Higher action coverage did not produce better high-IoU mAP |
| PAction learned is just p_action top-k imitation | FAIL/HOLD | PAction top-k overlap is lower than GAS-VT while boundary r1 and mAP are higher |
| PAction learned is the final method | FAIL | It is still Stage1 p_action/GT-surrogate supervised selection |
| PAction learned fixed_384 is the current strong Stage1 baseline | WARN/PASS as working baseline | Current logs support this, matched reruns still required |
| Detector-utility-calibrated acquisition is the main route | PASS as research direction | It directly targets the observed failure mode |

## Code-Level Findings To Absorb

### P0: GAS-VT Apply-Time Target-Budget Mismatch

Training conditions GAS-VT features on a target budget. Apply-time checkpoint scoring can build features without that target budget, causing the feature builder to treat the budget as approximately the full valid length.

This contaminates:

- `remaining_budget`
- `remaining_time`
- `budget_pressure`
- `gap_urgency`
- the learned meaning of fixed_384 and dynamic budgets

Required action:

- For fixed variants, pass the concrete budget into apply-time scoring.
- For dynamic variants, decode budget first, then rebuild features with the decoded budget before final frame-value scoring.

### P0: Stage2 Inherits The Same Risk

Stage2 detector-aware policy reuses the GAS-VT feature builder. If detector-aware apply also builds features without target budget, Stage2 inherits the same train/apply mismatch.

Required action:

- Fix Stage2 apply-time feature construction in the same patch family as GAS-VT.
- Run Stage2 precheck only after this is verified.

### P1: GAS-VT Is Not Yet True Sequential VT

Current GAS-VT has sequential-looking features, but the ledger path still behaves like:

> score all frames -> top-k -> constrained max-hole/hard repair

not:

> select one point -> update state/gap/value -> select next point

Required action:

- Demote current GAS-VT to "engineered gap/coverage baseline", or implement a true iterative state-update decoder and test it as an ablation.

### P1: Validator Needs Detector-Geometry Statistics

The validator should absorb the sampled analysis into formal summaries:

- selected-count histogram;
- start/end boundary distance separately;
- boundary r1/r2/r4;
- p_action top-k, delta top-k, and boundary-score overlap;
- gap CV;
- phase-shift uniformity;
- CDF/KS distance to uniform;
- repair inserted/replaced count;
- pre-repair and post-repair max/p95/p99 hole.

## Method Direction Update

The main method should be renamed/reframed away from GAS-VT as the paper center.

Suggested working name:

> DUCA-TAD: Detector-Utility-Calibrated Acquisition for Temporal Action Detection

Core problem:

> Pre-backbone sparse acquisition for TAD should maximize downstream detector utility under a compute budget, not maximize action coverage.

Core objective:

```text
S* = argmax_{|S| <= K} U_detector(S)
```

where `U_detector` must include classification confidence, boundary regression quality, high-IoU matching quality, and false-positive risk, not only action-positive coverage.

## Recommended Selector Design

The selector should become a detector-utility-calibrated module with separate utility heads:

- `u_cls`: useful for action classification;
- `u_start`: useful for start-boundary localization;
- `u_end`: useful for end-boundary localization;
- `u_risk`: likely false-positive or mislocalization risk.

Acquisition should explicitly decompose budget:

```text
K = K_boundary + K_context + K_uncertainty + K_safety
```

The review's key design preference is:

- Boundary utility should be modeled explicitly.
- Safety coverage is allowed, but it must be tagged and ablated.
- Hard repair should not silently convert a learned policy into a uniform-like lattice.

## Stage2 Absorption

Stage2 should train the selector from dense AdaTAD teacher utility, not from action labels alone.

Utility targets should include:

- positive observation gain;
- boundary regression gain;
- high-IoU match gain;
- false-positive risk penalty;
- teacher gradient utility or masked perturbation utility;
- start/end boundary error decomposition.

Stage2 loss should emphasize ranking quality because final selection is top-k:

```text
loss =
  utility BCE
  + pairwise/rank loss
  + boundary pair loss
  + budget loss
  + diversity/redundancy loss
  + small safety coverage loss
```

Stage2 evaluation must report:

- detector utility NDCG@K;
- top-k utility recall;
- boundary r1/r2/r4;
- start/end coverage separately;
- p95/p99 hole;
- selected-count histogram;
- length-bucket mAP;
- mAP@0.6 and mAP@0.7;
- matched comparison to PAction learned fixed_384.

## Stage3 Absorption

Stage3 should prove genuine joint optimization:

> Detector loss changes selector parameters in the same training graph and improves TAD mAP.

Minimum proof gates:

```python
loss_detector.backward()
assert selector_grad_norm > 0
assert selector_param_delta_after_step > 0
```

Recommended curriculum:

1. Initialize selector from Stage2.
2. Freeze AdaTAD and train selector against teacher utility.
3. Freeze selector and train sparse AdaTAD detector.
4. Joint fine-tune with selector LR 10x-100x smaller than detector LR.
5. Add trust-region to prevent selector distribution drift.
6. Add anti-collapse gates: entropy, selected-count, duplicate count, max-hole, boundary support.
7. Log selector gradient norm from detector loss every training phase.

## Detector Interface Warning

Current sparse AdaTAD config uses selected-axis remapping. This is useful for engineering consistency, but may limit high-IoU localization under irregular sparse selection.

The more paper-grade interface should expose:

```text
selected_features: [B, C, K]
selected_time:     [B, K]
cell_width:        [B, K]
valid_mask:        [B, K]
```

The detector should learn true-time start/end prediction rather than only selected-rank-axis prediction.

This is a larger change, but it matches the final story:

> The detector understands sparse non-uniform temporal observations.

## Minimal Experiment Matrix

### Explain PAction

| Experiment | Purpose |
|---|---|
| raw p_action top-k fixed_384 | Check whether PAction beats raw p_action |
| PAction no boundary loss | Test whether boundary supervision drives mAP@0.7 |
| PAction no temporal hole loss | Test whether p95 hole/local continuity drives gain |
| PAction no hard repair | Test repair dependence |
| PAction same max-hole as GAS | Fair repair-effect comparison |

### Explain GAS

| Experiment | Purpose |
|---|---|
| GAS current fixed_384 | Reproduce current baseline |
| GAS apply budget fix | Test train/apply mismatch |
| GAS no CVaR max-hole | Test over-coverage |
| GAS no action_interior_bin | Test whether interior coverage wastes budget |
| GAS no hard repair | Test whether repair hurts boundary peaks |
| GAS true sequential decode | Test whether actual sequential VT matters |

### Promote Main Route

| Experiment | Purpose |
|---|---|
| Stage2 teacher utility selector fixed_384 | Must beat PAction under matched budget |
| Stage2 boundary utility variant | Test mAP@0.7 improvement |
| Stage2 rank loss vs BCE | Test ranking-oriented learning |
| Stage3 ST selector gradient smoke | Prove detector loss reaches selector |
| Stage3 joint fine-tune mAP | Prove end-to-end benefit |

## Paper Story Rewrite

Do not center the paper around:

> GAS-VT with gap-aware CVaR and boundary bracket improves TAD.

Instead:

> Sparse TAD is not solved by action coverage. A selector can preserve many action-interior frames and still fail high-IoU localization because it misses boundary-discriminative evidence. We propose detector-utility-calibrated pre-backbone acquisition: a low-cost selector trained by dense AdaTAD teacher utility and then jointly fine-tuned with detector loss, while preserving true-time sparse geometry for high-IoU localization.

This makes the novelty:

1. problem definition: pre-backbone sparse acquisition for TAD;
2. diagnosis: action coverage is not high-IoU detector utility;
3. method: detector-utility-calibrated selector;
4. geometry: true-time sparse observation interface;
5. training: Stage2 teacher utility distillation plus Stage3 detector-loss joint fine-tuning;
6. evaluation: mAP@0.6/0.7, boundary support, hole distribution, utility recall.

## Immediate Decision

```text
GAS-VT_AS_MAIN_METHOD: NO
GAS-VT_AS_STAGE1_ENGINEERED_BASELINE: YES
PACTION_LEARNED_AS_STAGE1_STRONG_BASELINE: YES
PACTION_LEARNED_AS_FINAL_METHOD: NO
NEXT_MAIN_ROUTE: DETECTOR_UTILITY_CALIBRATED_ACQUISITION
IMMEDIATE_PATCH_PRIORITY:
  1. GAS/Stage2 apply-time target_budget conditioning
  2. validator top-k overlap + boundary distance + selected-count histogram
  3. repair statistics and richer uniformity metrics
  4. PAction/GAS matched ablations
  5. Stage2 dense teacher utility selector
```

## Absorption Status

This review is fully recorded and absorbed. It should be used as the current claim lock:

- Do not over-explain GAS-VT as merely uniform collapse.
- Do not overclaim PAction learned as detector-aware.
- Do not center the final paper on GAS handcrafted priors.
- Move implementation priority toward detector utility, boundary-specific utility heads, true-time sparse geometry, Stage2 utility distillation, and Stage3 joint gradient proof.

No running experiment was changed by this record.
