# ActionFormer SparseHead DCSR G1 Integrity and Comparability Audit

Date: 2026-07-30

Audit status: `tested`

Overall verdict: `WARN`

Scientific integrity verdict: `PASS`

Official paper-comparability verdict: `FAIL`

The warning is a scope warning, not metric fraud or an engineering failure.
G1 is a valid preregistered internal method-kill experiment. It is not an
official THUMOS test result and cannot be a paper performance row.

## A. Preregistration and stopping rule: PASS

- G1 froze three development seeds, a deterministic 160/40 split derived only
  from official `validation`, paired dense controls and terminal epoch-35 EMA.
- The frozen continuation bounds are Avg delta `>= -0.50 pp` and
  mAP@0.6/mAP@0.7 deltas each `>= -1.00 pp`.
- Observed values are `-7.556202/-11.043134/-11.019821 pp`; all seeds and
  thresholds are negative.
- The frozen kill rule requires route termination. No G2--G4 or official
  five-seed continuation is permitted.

## B. Data and leakage boundary: PASS

- Manifest SHA-256 is
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`.
- It uses 160 development-training and 40 holdout videos from official
  `validation`, with all 20 classes represented.
- The annotation file contains multiple subsets, but selection and evaluation
  use only manifest-bound validation records. The formal receipts and
  diagnostic completion record `test_subset_used=false`.
- No test GT, AP, prediction, teacher signal or test-derived checkpoint choice
  was used.

## C. Pairing, training and checkpoint: PASS

- Frozen paired seeds are `2026073001/2026073002/2026073003`.
- Dense and DCSR arms share data, seed, schedule, evaluator, terminal epoch-35
  EMA rule and environment.
- Pair receipts and aggregate receipt exist and validate. No result was
  reconstructed from an incomplete task.
- G0 independently proves native-grid, tensor, decoder and official Soft-NMS
  identity before the G1 architecture change.

## D. Evaluator and metric normalization: PASS

- The same independent holdout evaluator recomputes every arm from raw
  predictions.
- AP uses raw score ordering and dataset-GT denominators. No prediction-derived
  self-normalization is substituted.
- Counterfactual arms use the same frozen predictions/evaluator. Their
  completion is explicitly `diagnostic_only=true` and
  `paper_performance_row_allowed=false`.

## E. Artifact and execution existence: PASS

- Formal G1 aggregate SHA-256:
  `b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
- Per-seed pair SHA-256:
  `c924ed997a438f14e3d4660906635e2cca90b34b8ac2d2dc7c4170df2a4a5867`,
  `a2ecb27e8485c10fe97a5319b930c1e7d49d5f918f02c64269ec6281d87f88da`,
  `9b85c1f38f5ecb4ba7fbb9e60c39c9e6005dc8201d828aa79cce0a365cdfcd40`.
- Diagnostic aggregate Job `1207441` completed `0:0`; its log contains no
  traceback, OOM or failed-validation finding.
- Diagnostic completion/prediction/checkpoint SHA-256:
  `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53`,
  `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36`,
  `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.

## F. Official comparability: FAIL by design

- G1 trains on 160 validation videos and evaluates a 40-video validation
  holdout. It does not train on the full official validation split or evaluate
  the official test split.
- It is an internal architecture gate, not an official ActionFormer benchmark
  row. Absolute G1 values cannot be compared with historical `63.xx`, released
  `66.833392`, or official S0 `66.583013`.
- The official five-seed protocol was conditional on G1--G4 passage. G1 failed,
  so running it only to obtain a paper-looking number would violate the
  preregistration.

## G. Cost and efficiency: FAIL / not measured

- No synchronized complete feature-to-final-detection latency, memory, FLOPs,
  energy or paired confidence interval was measured.
- Neither G1 nor its diagnostics authorize speedup or efficiency claims.

## H. Mechanistic inference: PASS with strict qualifiers

- Scaffold-only, all-query and K384 counterfactuals support the one-layer
  scaffold/decomposition as the dominant observed factor and K384 support as a
  smaller high-IoU factor.
- Checkpoint norms show residual finals become nonzero and settle, weakening a
  persistent dead-branch explanation.
- The evidence cannot uniquely separate representation capacity from
  optimization, cannot observe first-step gradients, and cannot reconstruct
  suppressed pre-NMS proposals.

## I. Independent review: WARN

Six local independent code/evidence reviews and three final independent
scientific audits were completed. Configured Claude, Gemini, GPT-4o and
MiniMax external reviewer routes were unavailable because credentials were
missing or revoked. This availability warning must not be represented as an
external cross-model PASS.

## Paper-safe statement

“On a preregistered three-seed internal holdout from the official THUMOS
validation split, the exact DCSR G1 architecture failed its paired
non-inferiority gate by `7.56 pp` Avg-mAP. Post-hoc no-training diagnostics
attribute most of the observed gap to the one-layer scaffold/decomposition,
with a smaller high-IoU penalty from K384 residual support.”

This sentence belongs in development/negative-analysis material, not an
official benchmark performance table.

## Forbidden statements

- DCSR achieves an official THUMOS test score.
- DCSR is below or above the historical `63.xx` baseline.
- DCSR is faster, cheaper or accuracy preserving.
- NMS, calibration, zero initialization or optimization is the sole cause.
- All sparse heads or conditional-computation methods fail.
- The audit received an external cross-model PASS.
