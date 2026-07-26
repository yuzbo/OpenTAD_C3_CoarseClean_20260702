---
updated: 2026-07-07
status: active
scope: Absorb the Pro review on whether DUCA-TAD should be framed as raw-frame selection or detector-compatible temporal observation acquisition.
out-of-scope: Reporting new experiment results, changing running jobs, or claiming the current implementation is paper-complete.
---

# Observation Grid Vs Raw Frame Pro Review Absorption

Raw record:
`docs/methods/reviews/2026-07-07-observation-grid-vs-raw-frame-pro-review-raw.txt`

Source attachment:
`C:\Users\skywalker\.codex\attachments\31cb19f7-844f-4f78-bbab-fada1a08ae5f\pasted-text.txt`

## Absorbed Verdict

The review is accepted as a route-level correction and paper-claim guard.

Final absorbed verdict:

> The current method should not be described as raw-frame selection. It should be
> described as detector-compatible sparse temporal observation acquisition:
> selecting up to 384 observations from AdaTAD's 768-position temporal input
> lattice before expensive VideoMAE/AdaTAD backbone computation.

This still supports a pre-backbone acquisition story because the sparse ledger
is applied before `DecordDecode` / VideoMAE backbone processing, not after dense
feature extraction or proposal generation. The claim must, however, report cost
in selected temporal observations, decoded raw-frame accesses, unique raw frames,
VideoMAE chunks/tokens, runtime, and memory. Selected count alone is not compute
evidence.

## Claim Correction

Unsafe or discouraged wording:

- `384 raw frames`
- `raw-frame selection`
- `frame-level selector`
- `select 384 frames from raw video`
- `end-to-end raw video frame selector`

Allowed and preferred wording:

- `temporal observation acquisition`
- `detector-compatible sparse temporal observation acquisition`
- `snippet-grid observation acquisition`
- `pre-backbone sparse observation acquisition`
- `backbone input sample acquisition`

The paper claim must explicitly say that the current selector chooses positions
on a dense detector-compatible temporal lattice. A selected position is a
temporal observation / snippet-grid position, not necessarily one raw video
frame.

## Route Decision

The review separates three possible routes.

### Route A: Detector-Grid Acquisition

Keep the current 768-to-384 detector-grid acquisition as the immediate main
route, but rename and validate it correctly.

Why this is accepted:

- It aligns with AdaTAD/OpenTAD temporal contracts.
- It avoids rewriting detector assignment, postprocess, and time conversion
  before the selector utility problem is solved.
- It allows dense AdaTAD teacher utility and sparse selector training to live in
  the same coordinate system.
- It is the shortest path to testing whether a learned selector can beat
  uniform/PAction baselines under `K<=384`.

Mandatory additions:

- observation-to-raw-frame mapping sidecar;
- cost accounting for raw accesses and VideoMAE units;
- geometry round-trip tests;
- claim-unit validator;
- raw/grid oracle diagnostic.

### Route B: Pure Raw-Frame Acquisition

Do not make pure raw-frame acquisition the immediate main route.

Why it is deferred:

- It changes the unit of selection from detector-grid observation to raw
  timestamp, introducing clip-center alignment, frame interval, chunk packing,
  GT remapping, true-time postprocess, and cost accounting ambiguity.
- It may not reduce VideoMAE compute if each raw center becomes a separate
  16-frame clip.
- It risks lowering mAP before we know whether raw-level freedom has real
  headroom.

Route B is only justified if a raw-oracle diagnostic shows meaningful headroom
over grid-level oracle acquisition.

### Route C: Raw-Aware Scout + Detector-Grid Acquisition

This is the recommended final narrative:

1. A low-cost raw/video-aware scout estimates action, motion, transition, or
   uncertainty signals.
2. These signals are projected to the detector-compatible 768-position lattice.
3. A detector-utility-calibrated selector chooses `K<=384` observations.
4. At test time, no dense teacher, ground truth, or prediction cache is used.
5. The sparse observations are decoded and processed by AdaTAD.

This keeps the method compatible with AdaTAD while preserving a credible
raw-video-aware acquisition story.

## Paper Target

Recommended method identity:

**DUCA-TAD: Detector-Utility-Calibrated Observation Acquisition for Temporal
Action Detection**

The method should be positioned as:

> A detector-aware sparse temporal observation acquisition framework for
> AdaTAD/OpenTAD-style temporal action detection.

It is not yet:

- a raw-frame-level selector;
- a new full TAD detector;
- a universal plug-in for arbitrary TAD models;
- an end-to-end selector-detector model unless Stage3 proves nonzero detector
  loss gradients and selector movement.

## Contribution Boundaries

Accepted contribution structure:

1. Problem formulation:
   pre-backbone sparse temporal observation acquisition for TAD, distinct from
   post-backbone feature/proposal pruning.

2. Detector-utility calibration:
   use train-only dense AdaTAD point responsibility / loss / gradient utility to
   supervise the selector, rather than relying only on actionness or hand-coded
   boundary heuristics.

3. Strict accounting and geometry:
   preserve the mapping from selected detector-grid observations to raw frame
   centers, timestamps, VideoMAE chunks, and true-time detector outputs, so
   efficiency and high-IoU localization claims are not coordinate artifacts.

## Required Experiment Matrix

Matched sparse baselines:

- dense AdaTAD 768;
- uniform 384;
- random 384;
- raw p_action top-k 384;
- PAction learned fixed384;
- GAS-VT fixed384;
- PAction lattice replacement as diagnostic only;
- Stage2 proposal-score surrogate as diagnostic only;
- Stage2 point-responsibility selector as paper-main candidate;
- Stage3 joint selector-detector training only after real gradient and full mAP
  evidence.

Mandatory diagnostics:

- mAP vs budget: 256 / 320 / 384 / 512;
- teacher utility vs selected probability calibration;
- boundary recall and endpoint error, especially at high tIoU;
- p95/max hole on detector-grid axis and raw-time axis;
- raw/grid oracle headroom;
- selected observation timeline with raw timestamp overlays;
- selected observation to VideoMAE chunk/tubelet mapping.

## Required Code And Artifacts

The review requires the following concrete artifacts before paper-level claims:

- `observation_mapping` sidecar for each sparse ledger row;
- `compute_observation_cost.py` for selected observations, decoded raw-frame
  accesses, unique raw frames, VideoMAE chunks/tokens, runtime, and memory;
- `validate_paper_claim_units.py` to forbid raw-frame claims when selected
  positions are local dense indices;
- `test_geometry_roundtrip.py` for selected-axis, dense-grid, raw-frame, and
  seconds conversion;
- `manifest.schema.json` requiring acquisition unit, decode mode, selected
  position unit, raw-frame cost, and VideoMAE cost;
- `raw_grid_oracle_diagnostic.py` to estimate whether raw-frame acquisition has
  enough extra headroom to justify a route shift;
- Stage2 selector manifests with:
  `uses_teacher_train_only=true`, `uses_teacher_at_eval=false`,
  `selected_positions_unit=local_dense_index`.

## Current Implementation Mapping

Already aligned after the latest local route:

- Stage2 claim guards now reject proposal-score surrogate as paper-main utility.
- Selector / ledger metadata now distinguishes temporal observation units from
  raw-frame units.
- The current project direction is explicitly DUCA-style detector-utility
  sparse observation acquisition, not raw-frame selection.

Still missing for paper-level support:

- true dense AdaTAD point-responsibility utility full run;
- observation-to-raw-frame cost table;
- geometry round-trip test suite;
- raw/grid oracle diagnostic;
- matched uniform384 under the same latest detector setup;
- verified Stage2 point-responsibility mAP over PAction learned and uniform;
- Stage3 nonzero selector-gradient evidence and full detector mAP.

## Adopted Immediate Plan

1. Keep detector-grid `K<=384` as the main implementation path.
2. Do not rename the current method as raw-frame selection.
3. Treat lattice replacement as diagnostic, not the final intelligent selector.
4. Prioritize Stage2 true point-responsibility utility and matched uniform384.
5. Add accounting and geometry code before making efficiency or high-IoU claims.
6. Use raw-frame route only as an oracle/scout diagnostic until it shows
   meaningful headroom.

## Reviewer Attack Surface

Expected attack:

> This is only subsampling AdaTAD's temporal grid, not selecting frames from raw
> video.

Defense:

> Correct. The contribution is detector-compatible temporal observation
> acquisition before expensive backbone computation. We explicitly report raw
> frame accesses, unique raw frames, VideoMAE chunks, runtime, and geometry
> round-trip consistency instead of equating 384 observations with 384 raw
> frames.

Expected attack:

> Uniform 384 may be enough.

Defense:

> Matched uniform384, PAction learned384, GAS-VT384, Stage2 responsibility384,
> and raw/grid oracle diagnostics are required before any main claim.

Expected attack:

> Teacher utility leaks dense detector knowledge into test-time selection.

Defense:

> Utility is train-only; deploy ledgers must prove no teacher, GT, prediction
> cache, or oracle boundary is used at val/test.

Expected attack:

> High-IoU improvements may be coordinate-remapping artifacts.

Defense:

> Geometry round-trip tests and true-time diagnostics are mandatory.

## Final Accepted One-Sentence Claim

The safest claim to pursue is:

> We propose detector-utility-calibrated sparse temporal observation acquisition
> for TAD: given a dense detector-compatible temporal input lattice, a lightweight
> train-only-supervised selector acquires at most 384 observations before
> VideoMAE/AdaTAD backbone computation, improving sparse detector performance
> over matched uniform and actionness baselines while reporting strict
> observation, raw-frame, and backbone cost.

The improvement clause is conditional. It can only be used after Stage2
point-responsibility full-run evidence beats the matched sparse baselines.
