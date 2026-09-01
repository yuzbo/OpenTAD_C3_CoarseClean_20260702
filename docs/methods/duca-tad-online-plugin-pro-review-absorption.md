---
updated: 2026-07-08
status: active
scope: Absorbed Pro review of the 7-page DUCA-TAD online acquisition plugin paper draft
out-of-scope: Reporting new mAP numbers, implementing missing code, or treating missing experiments as the primary review issue
---

# DUCA-TAD Online Plugin Pro Review Absorption

Raw record:

- `docs/methods/reviews/2026-07-08-duca-tad-online-plugin-pro-review-raw.txt`

Raw record SHA256:

```text
2DFE53622867023AA5310FDD4E46B41F1A48912B80EAE0AE598FF4781E7C41D3
```

## Core Verdict

The Pro review rates the current 7-page draft as:

```text
Reject, close to Borderline.
```

The verdict is not mainly because the detector mAP tables are empty. The review
explicitly says the draft already fixed several dangerous boundaries:

- DUCA-TAD is not a new full TAD detector.
- DUCA-TAD is not an offline ledger pipeline.
- The main contract is hard selected positions from a dense window, e.g.,
  `T=768`, `K<=384`.
- Inference forbids teacher, GT, raw prediction cache, and offline decision
  ledger.
- Zero-shot actionness is a deploy-visible prior or diagnostic, not proof of TAD
  success.

The remaining problem is that the paper still reads like a method contract and
protocol draft rather than a fully credible CVPR method paper. The next paper
revision must tighten method definitions before filling results.

## Three Central Reviewer Attacks

### 1. "Online" Is Ambiguous

The current paper uses `online` to mean selection is generated during the
detector forward/window rather than loaded from a precomputed ledger. However,
reviewers may read `online` as causal streaming detection with no future frames.
The current formula `S = A_theta(x_{1:T})` sees the whole current window, so the
paper must define the term.

Required definition:

```text
We use online acquisition to mean that selected positions are computed on the
fly from the current detector window during the forward pass, rather than loaded
from an offline ledger. It does not imply causal streaming emission or
prefix-only online temporal action localization unless explicitly stated.
```

Possible safer title:

```text
DUCA-TAD: Detector-Utility-Calibrated In-Forward Temporal Acquisition for Efficient Temporal Action Detection
```

or:

```text
DUCA-TAD: Window-Online Detector-Utility-Calibrated Temporal Acquisition for Efficient Temporal Action Detection
```

### 2. Hard-Forward ST Gradient Contract Is Not Proved

The current Method says hard-forward straight-through joint fine-tuning lets
detector loss update the selector. The review identifies this as the most
dangerous method claim.

Problem:

- Algorithm uses `S <- HardTopK`.
- Detector consumes packed subset `D_phi(V_S, G_S)`.
- Standard hard gather and TopK do not give detector loss a gradient path to
  selector logits.
- The current ST mask appears only in a surrogate loss, not in the detector
  input.

Two acceptable routes:

1. Downgrade the claim:

```text
During hard-forward fine-tuning, detector loss updates the detector parameters,
while selector parameters are updated by detector-utility surrogate losses
computed on the same hard-selected forward. We do not claim exact gradients
through TopK.
```

2. Implement and describe a real ST/custom-autograd bridge:

```text
In forward, we gather V_S. In backward, a custom autograd operator scatters the
selected-token detector gradient back to selected dense positions and uses a
surrogate estimator for selector logits; unselected positions receive gradients
through coverage and budget surrogates.
```

Before claiming detector-loss-driven acquisition, the paper needs a gradient
path unit test:

```text
|| d L_det / d theta_selector || > 0 with ST bridge enabled,
and zero or near zero when the bridge is disabled.
```

### 3. Original-Time Sparse Grid Is Still Too Abstract

The draft correctly says selected rank is not physical time, but it does not yet
pin down geometry enough for a TAD reviewer.

Required formalization:

```text
For selected coordinates tau_i = s_i, define support-cell boundaries
l_i = (tau_{i-1}+tau_i)/2 and r_i = (tau_i+tau_{i+1})/2,
clipped to the valid window bounds.

The detector point center is c_i = tau_i.
Regression targets and decoded segments are measured in original dense-time
units:
y_i^left = c_i - t_start,
y_i^right = t_end - c_i,
pred_start = c_i - delta_i^left,
pred_end = c_i + delta_i^right.
```

The paper must state that assignment, regression range checks, score filtering,
decoding, and NMS operate in original-time coordinates after conversion.

Required ablation:

```text
selected-axis decode vs original-time sparse-grid decode
```

This ablation should not be buried as a sentence; it must appear in the main
experiment protocol.

## P0 Rewrite Gates

These must be fixed before the next Pro review.

1. Define `online` as window-online or in-forward selection, not causal
   streaming.
2. Consider title change or add title/abstract limitation so `online` is not
   overclaimed.
3. Remove formal-paper phrasing such as "quantitative mAP cells are locked" from
   abstract and camera-ready prose.
4. Rewrite ST fine-tuning to either prove a detector-loss gradient path or
   downgrade to hard-forward surrogate fine-tuning.
5. Formalize sparse-grid geometry: support cells, point centers, offset units,
   assignment, regression range, decoding, and NMS.
6. Replace any "detector architecture reused" wording with a precise statement:
   detector modules are reused, while the sparse-coordinate interface is
   explicit and may modify input, point generation, assignment, decode, and
   postprocess paths.
7. Add an observation/compute boundary: raw/snippet/pre-backbone selection
   versus feature-stream/head-only selection must be reported separately.
8. Redraw Figure 1 into three lanes: inference graph, training-only supervision,
   audit sidecar.
9. Split zero-shot prompt taxonomy into generic prompts, dataset-taxonomy prompts,
   and supervised learned sources.
10. Remove or rewrite Appendix internal progress notes as implementation
    contracts and invariant checks.

## Figure 1 Absorption

Current Figure 1 is directionally right, but the review says it still needs a
stronger visual contract.

Required three-lane structure:

### Lane 1: Inference Graph

Solid arrows only:

```text
Dense candidate window V_t
-> low-cost deploy-visible descriptors z_t
-> DUCA_theta hard TopK
-> selected_positions S, K<=384
-> Gather V_S
-> SparseGrid G_S: original coordinates + support cells
-> Detector D_phi(V_S, G_S)
-> Predictions in original time
```

Add a forbidden dense bypass indicator:

```text
V_{1:T} -> detector is forbidden.
```

### Lane 2: Training-Only Supervision

Dashed arrows only:

```text
Dense teacher / detector utility on train split
-> u*_t: boundary + responsibility + hard-negative risk
-> warm-up loss / surrogate loss
-> DUCA_theta
```

Add a firewall label:

```text
Removed before validation/test/deployment.
```

### Lane 3: Audit Sidecar

Sidecar written after selection:

```text
selected_positions S and SparseGrid G_S
-> post-decision audit record:
   hashes, selected count, no-leak flags, cost
```

No arrow may go from audit record back to selector.

## Claim and Scope Policy

### Allowed

- DUCA is a window-online or in-forward sparse temporal acquisition plugin.
- DUCA is teacher-free at inference.
- DUCA uses train-only detector utility for calibration.
- DUCA emits strict-budget original-time selected positions.
- The detector consumes hard selected observations.
- The audit record is a reproducibility and no-leak artifact.
- Zero-shot actionness is a deploy-visible prior, frozen branch, baseline, or
  diagnostic.

### Forbidden Until Verified

- Causal streaming online TAD if the selector sees the full current window.
- Detector-loss-trained selector if no ST/custom gradient path exists.
- High-IoU protection without original-time assignment/decode/NMS details.
- Raw-frame or full-backbone efficiency if selection happens after dense feature
  extraction.
- Generic zero-shot claim when using THUMOS class-name prompts.
- Coarse actionness AUROC as evidence for detector mAP improvement.
- Oracle/GT variant as deployable.
- Offline ledger as method input or decision source.

## Zero-Shot Actionness Absorption

The current direction is good, but Table 1 should be reorganized into a prompt
taxonomy:

| Category | Source | Claim Allowed |
|---|---|---|
| Label-free deploy prior | generic action/background prompts | generic zero-shot deploy-visible branch |
| Dataset-taxonomy prior | THUMOS class-name prompts | dataset-aware diagnostic only |
| Trained source | C3/PAction/DUCA-adapted | supervised or adapted baseline |
| Oracle source | GT actionness/boundary | diagnostic upper bound only |

Prompt provenance must include:

- prompt text or prompt ID;
- prompt hash;
- checkpoint hash;
- whether THUMOS class names are used;
- frozen versus adapted status;
- score-generation cost;
- whether target labels are used for calibration.

## Experiment Protocol Absorption

The three-gate structure is correct and should be kept:

1. Coarse zero-shot actionness evaluation.
2. Selection-geometry evaluation.
3. Sparse detector mAP evaluation.

Required improvements:

- Split start-boundary and end-boundary recall.
- Report boundary radii such as +/-1, +/-2, +/-4, +/-8 dense positions.
- Stratify by short, medium, and long action duration.
- Report action-interior over-selection ratio.
- Report largest uncovered GT action fraction.
- Report selected-count drift and budget violation.
- Report action-local max gap and p95 gap separately from background gaps.
- Add selected-axis decode versus original-time decode as a main negative
  control.
- Add detector-training parity columns:
  detector training, selector training, coordinate mode, seeds, and eval config.
- Ensure uniform/random/actionness baselines receive equivalent detector
  adaptation or explicitly report when detector weights are shared/frozen.

Compute accounting must be decomposed into:

- low-cost descriptor cost;
- zero-shot scorer cost;
- selector temporal encoder cost;
- expensive video backbone cost;
- detector temporal head cost;
- postprocess/NMS cost;
- wall-clock latency;
- peak memory;
- whether dense features were precomputed.

## Language Replacement Rules

Use these replacements in the next paper edit:

| Risky Wording | Replacement |
|---|---|
| online temporal acquisition | in-forward/window-online acquisition, with definition |
| stricter problem than offline frame subsampling | deployment stricter than offline subsampling: generated during forward from deploy-visible inputs |
| training uses dense detector utility only as supervision | training may use dense detector utility as train-only supervision; inference never reads it |
| detector loss updates the selector | only use if a real ST/custom gradient path is implemented and verified |
| detector architecture is reused | detector modules are reused; sparse-coordinate interface is made explicit |
| sparse grid is the key object for high-IoU credibility | sparse grid is necessary because selected rank is not metric time |
| zero-shot with class names | dataset-taxonomy zero-shot diagnostic |
| quantitative mAP cells are intentionally locked | remove from formal paper; keep only in internal notes |
| current implementation direction exposes... | implementation contract / invariant checks |
| final arbiter | decisive evaluation metric |

## Related Work Expansion

The next draft should expand Related Work beyond current TAD, efficient TAD,
video frame selection, zero-shot actionness, and provenance.

Add categories for:

- differentiable TopK, hard subset selection, and ST estimators;
- adaptive token/frame pruning in video;
- online or streaming temporal action localization, with a note that this paper
  is window-online unless otherwise stated;
- sparse or irregular temporal grids and non-uniform sampling geometry;
- detector-aware proposal/sample selection if relevant.

The related-work close should clarify what DUCA is not:

```text
DUCA is not a new detector, not an offline ledger method, not zero-shot TAD, and
not causal streaming detection unless the causal setting is explicitly enabled.
```

## Implementation / Audit Invariants To Surface

The paper should list or test these invariants:

- `selected_count <= 384` in the main setting.
- detector temporal input length equals selected count.
- no dense bypass into the detector.
- `ledger_read=false` during inference decision.
- `uses_gt=false`, `uses_teacher=false`, `uses_raw_prediction=false`,
  `uses_cache=false`.
- prompt/checkpoint/source hashes are present.
- selected positions are sorted, unique, and inside the valid prefix.
- decoded proposals and NMS inputs are in original-time units.
- validator fails closed on missing provenance or forbidden keys.

## Local Interpretation

This review supersedes the current 7-page paper draft as the active rewrite
gate. The direction is accepted, but the next revision must tighten four areas
before results are filled:

1. online terminology;
2. ST gradient contract;
3. original-time sparse-grid geometry;
4. compute/observation boundary.

The next paper edit should not primarily add new result placeholders. It should
make the method contract auditable enough that a CVPR reviewer can no longer
attack DUCA as a renamed offline selector, an unproven ST trick, or a
selected-axis decoding hack.
