---
updated: 2026-07-07
status: active
scope: Absorbed Pro/CVPR-style editorial review for the 6-page DUCA-TAD paper draft
out-of-scope: Claiming any issue is fixed, reporting experiment results, or changing code/experiment policy
---

# DUCA-TAD CVPR Pro Paper Review Absorption

Raw record: `docs/methods/reviews/2026-07-07-duca-tad-cvpr-pro-paper-review-raw.txt`

Raw review SHA256:

```text
1B8058662678BF2F515D5C269488F685A4B093603ACD7314CF786FDE4A2E4D8D
```

Reviewed artifact:

```text
paper/main.pdf
```

Review scope:

- Title
- Abstract
- Introduction
- Related Work
- Problem Formulation
- Method
- Figure 1
- Experiment-design skeleton

The review explicitly did **not** treat missing experimental numbers or an unfinished conclusion as the main defect.

## Core Verdict

The completed front half of the paper is **Needs major rewrite** and is not CVPR submission-ready.

Absorbed interpretation:

- The research direction remains worth pursuing.
- The current draft is closer to a system-route description plus experiment plan than to a mature CVPR method paper.
- The main problem is not missing results; the main problem is that the method contract has not converged.
- The draft currently overuses broad terms such as `dynamic`, `utility-calibrated`, `detector-aware`, `TrueTime`, `value-transport`, and `deployable` without enough algorithmic definitions, training targets, inference rules, and evaluation invariants.

## Highest-Risk Reviewer Attack

The paper is currently vulnerable to this reviewer interpretation:

> Is this just a cheap actionness model selecting top-K frames before AdaTAD, with gap repair and metadata bookkeeping?

The rewrite must make the answer clearly no by proving four distinctions in the text:

1. TAD utility is not actionness.
2. Sparse TAD acquisition must preserve metric temporal geometry.
3. Deploy-visible, no-leakage selection is part of the method contract.
4. Dynamic budget is constrained and matched-compute, not simply variable K.

## P0 Absorbed Issues

### P0-1: Remove Proposal-Like Writing

The abstract and experiment section use future/proposal language, especially:

```text
Experiments will evaluate whether ...
```

Required action:

- Replace future-tense internal planning language with method/protocol language.
- Final paper must replace protocol-only language with real quantitative results.
- Internal draft notes such as "results can be inserted later" must not appear in the submission PDF.

### P0-2: Collapse the Method to One Primary Implementation

The current method describes a design space:

- low-resolution pixels
- compressed descriptors
- motion deltas
- coarse actionness probe
- many possible utility targets
- multiple budget mechanisms

Required action:

- Select one primary DUCA-TAD path for the main method.
- Treat all other browser inputs, heads, and budget decoders as ablations or appendix variants.
- Remove `can`, `may`, `optionally`, and `current implementation family` from main-method claims.

Default absorbed primary path unless superseded by later experiments:

```text
low-cost p_action TCN browser
-> deploy-visible value / boundary / redundancy-proxy / budget scores
-> constrained utility ledger decoder
-> sparse AdaTAD/ActionFormer-style detector with original-time metadata
```

### P0-3: Resolve the TrueTime vs Unchanged Detector Contradiction

The draft simultaneously claims:

- TrueTime metadata participates in assignment, downsampling, decoding, and NMS.
- AdaTAD / ActionFormer detector is unchanged.

These cannot both be true.

Required action:

- If original-time metadata affects assignment/decode/NMS, write:

```text
We keep the detector backbone/head architecture but replace temporal-coordinate handling with ledger-aware original-time operators.
```

- If the detector truly remains unchanged, do not claim TrueTime assignment/decode/NMS. Describe only selected-axis training plus final dense-time remapping.

Preferred absorbed direction:

- Use `original-time sparse grid` as the paper term.
- Avoid claiming a fully unchanged detector when temporal-coordinate operators are modified.

### P0-4: Define Detector Utility as an Algorithmic Target

The draft lists possible detector-utility sources but does not define a primary target.

Required action:

- Add a main-method section: `Training-Time Utility Target Construction`.
- Define exactly one primary target.
- State whether it uses a frozen dense teacher, detector loss, responsibility, boundary distance, or counterfactual utility.
- State that train-only targets are computed only on the training split and never serialized into validation/test ledgers.

Claim gate:

- Until this target is defined and implemented, do not write `detector-utility-calibrated` as a completed central claim.
- Safer interim term: `utility-supervised` or `utility-guided`.

### P0-5: Define or Rename Value Transport

The draft calls the module `value-transport`, but currently describes score ranking plus gap constraints.

Required action:

- Either formally define a transport plan with slots, frames, cost, mass constraints, and transport mass; or
- Rename the method component to `constrained utility ledger decoding`.

Absorbed default:

- Use `constrained utility decoder` in the paper until a real transport objective exists.
- Do not keep `transport mass` in the main formula unless it is tied to a defined transport plan.

## P1 Absorbed Issues

### P1-1: Rewrite Contributions as Verifiable Claims

The current contribution list reads like a component inventory.

Required contribution structure:

1. Formulate deployable sparse acquisition for TAD under original-time localization constraints.
2. Propose a utility-guided low-cost browser plus constrained sparse-ledger decoder.
3. Introduce an original-time sparse detector interface and no-leakage evaluation protocol.

Do not present the evaluation protocol alone as a main contribution unless the paper becomes a benchmark/protocol paper.

### P1-2: Expand Related Work

Current related work is too thin for CVPR.

Required categories:

- TAD detectors
- End-to-end and large-backbone TAD
- Efficient TAD and TAD model compression
- Adaptive frame/keyframe selection for video understanding
- Sparse or irregular temporal geometry
- Evaluation leakage and benchmark standardization

Required positioning:

- OpenTAD should be cited for unified/fair TAD comparison.
- Long-video frame selection / video reasoning methods should be contrasted against TAD's need for metric temporal segment regression.

### P1-3: Make Dynamic Budget a Constrained Problem

Dynamic K is currently underspecified.

Required action:

- Define a global expected-budget or latency constraint.
- Record K min/max, requested/effective K, short-window behavior, and total cost.
- Add matched-average-K and matched-latency controls.

Required experiment controls:

- DUCA dynamic vs DUCA fixed at same mean K.
- DUCA dynamic vs uniform fixed at same total latency.

### P1-4: Strengthen the Actionness-Only Failure Mode

The paper currently states that actionness alone is insufficient, but only as intuition.

Required action:

- State the failure mode sharply:

```text
Actionness ranks semantic confidence; TAD utility is boundary-sensitive and non-monotonic in actionness.
```

- Add a timeline figure or diagnostic showing:
  - uniform sampling
  - actionness top-K
  - DUCA selection
  - action interior redundancy
  - boundary support
  - short-action support

### P1-5: Replace Figure 1

Current Figure 1 is an internal pipeline diagram, not a strong CVPR main figure.

Required redesign:

- Panel A: failure-mode timeline comparing dense, uniform, actionness top-K, and DUCA.
- Panel B: deployment pipeline with solid arrows.
- Panel C: train-only utility path and ledger contract with explicit no-GT/no-teacher/no-cache/no-raw-prediction flags.

Also remove or change any `unchanged detector` label if original-time operators are modified.

## P2 Absorbed Issues

### P2-1: Language Surgery

Adopt these safer replacements:

| Current phrase | Absorbed replacement |
|---|---|
| learns which observations should be delivered | predicts deployable acquisition scores |
| value-transport module | constrained utility decoder, unless transport is formalized |
| current implementation family | in our implementation, or delete variants |
| a richer browser can add | the browser predicts, with ablations removing heads |
| easy to overlook | required for valid evaluation |
| no-leakage validation as first-class metrics | leakage audits reported alongside detection and compute metrics |

### P2-2: Title Risk

`Dynamic Utility-Calibrated Acquisition` is too abstract and may overclaim.

Preferred candidate titles:

1. `Detector-Utility-Guided Sparse Acquisition for Temporal Action Detection`
2. `True-Time Sparse Video Acquisition for Efficient Temporal Action Detection`
3. `DUCA-TAD: Deployable Utility-Guided Temporal Acquisition for Action Detection`

Default absorbed title direction:

- Prefer `Detector-Utility-Guided Sparse Acquisition for Temporal Action Detection` unless the original-time grid becomes the dominant contribution.

### P2-3: Terminology Table Required

Add a compact terminology or method-contract table before or inside Method:

- deploy-visible
- train-only utility
- constrained utility decoder
- sparse ledger
- original-time sparse grid
- requested/effective budget
- no-leakage flags

Avoid product-like terms such as `TrueTime` unless they are rigorously defined.

## Experiment Protocol To Lock Before Results

Minimum future matrix:

- dense AdaTAD
- exact uniform fixed K
- random fixed K with at least 5 seeds
- stratified or coverage random
- actionness top-K
- delta-actionness top-K
- boundary proxy top-K
- uniform with same gap constraints
- DUCA fixed K
- DUCA dynamic K
- dynamic matched-average-K fixed control
- selected-rank decode ablation
- oracle boundary upper bound as a ceiling diagnostic only

Required reporting:

- average mAP
- mAP@0.6 and mAP@0.7
- selected frame count
- browser overhead
- gather overhead
- backbone cost
- detector and NMS cost
- total latency
- dynamic-K distribution
- boundary support
- action coverage
- max gap / p95 gap
- max unselected hole / p95 unselected hole
- uniform similarity
- leakage validator pass/fail count

## Paper Claim Policy After Absorption

Forbidden until rewritten and verified:

- CVPR-ready front half.
- Detector-aware or detector-utility-calibrated as a completed claim without a defined utility target.
- TrueTime assignment/decode/NMS while also claiming unchanged detector.
- Value transport without a real transport plan.
- Dynamic budget improvement without matched-compute controls.
- Actionness-only insufficiency without diagnostic or ablation support.

Allowed cautious wording:

- The paper targets deployable sparse acquisition for TAD.
- The current draft identifies important constraints: deploy-visible selection, sparse ledger provenance, original-time localization, and matched-budget evaluation.
- The current manuscript requires a major rewrite to turn a route description into one reproducible CVPR method.

## Required Rewrite Order

1. Choose the primary method path and demote variants to ablations.
2. Rename or formalize value transport.
3. Resolve original-time detector interface wording and implementation contract.
4. Define the train-only detector utility target.
5. Rewrite abstract, introduction thesis, and contribution list.
6. Rewrite Method around the fixed algorithm rather than a design space.
7. Replace Figure 1 with failure-mode plus deployment/train-contract panels.
8. Expand Related Work with efficient TAD, OpenTAD, adaptive frame selection, and leakage/fairness.
9. Replace experiment skeleton prose with a locked baseline/ablation/cost matrix.
10. Recompile and re-review before external circulation.

## Direct Text To Reuse

Absorbed thesis:

```text
Efficient TAD is not frame sampling; it is budgeted acquisition under boundary-sensitive utility and original-time coordinate preservation.
```

Absorbed method opening:

```text
DUCA-TAD consists of four deterministic stages at inference time and one training-only supervision path. Given a dense candidate window, a lightweight temporal browser computes deploy-visible descriptors and predicts per-position acquisition scores. A constrained ledger decoder selects an ordered subset of dense-time indices under a fixed or budget-predicted count while enforcing validity, duplicate, and maximum-gap constraints. The detector loader gathers only the selected observations, but attaches their original dense coordinates and support cells as a sparse temporal grid. Ledger-aware detector operators then perform assignment, decoding, and post-processing in the original time axis. During acquisition training, detector-derived utility targets supervise the browser on the training split; these targets, teacher outputs, raw detector predictions, and caches are never serialized into validation or test ledgers.
```

Absorbed contribution list:

```text
1. We formulate deployable sparse acquisition for TAD.
2. We propose DUCA-TAD, a detector-utility-guided acquisition method.
3. We introduce an original-time sparse detector interface and evaluation protocol.
```

## Local Interpretation

This review should be treated as the controlling paper-writing gate for the current DUCA-TAD draft. It does not invalidate the project direction. It does invalidate the current front-half manuscript as reviewer-ready prose.

The next paper-writing pass must be a **major rewrite and method contraction**, not a light polish.
