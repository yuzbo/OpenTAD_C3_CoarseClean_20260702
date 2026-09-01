# ActionFormer SparseHead S0 Experiment Integrity Audit

Date: 2026-07-30

Audit status: `tested`

Overall integrity verdict: `WARN`

The warning is a scope/reviewer-availability warning, not evidence of metric
fraud or an implementation failure. The frozen artifacts, official evaluator,
data split, metric computation and same-run control pass manual source-level
and receipt-level checks. A required external cross-model file-level reviewer
could not be completed because the configured Claude reviewer returned no
parseable result and the Gemini/GPT-4o/MiniMax routes lacked valid API
credentials. Therefore this record must not claim an independent external
integrity PASS.

## A. Ground-truth provenance: PASS

- The official dense and K384 arms use `train_split: ['validation']` and
  `val_split: ['test']` from the ActionFormer THUMOS config
  (`configs/thumos_i3d.yaml:1-6` and
  `configs/thumos_i3d_sparsehead_k384_uniform.yaml:1-6`).
- `eval.py:52-87` constructs the declared test split, loads the frozen
  annotation JSON through the dataset and gives the dataset split and native
  THUMOS tIoU thresholds to `ANETdetection`.
- `libs/datasets/thumos14.py:78-138` loads annotations from the dataset JSON
  and filters videos by the declared subset. Ground truth is not derived from
  predictions.
- The no-retraining 2x2 and post-NMS diagnostics reuse the same official test
  annotation and frozen raw predictions. They are diagnostic-only.
- The assignment/support audit explicitly constructs the official
  `validation` training split, uses exactly 64 deterministic training windows,
  and asserts `test_gt_used=false`, `loss_computed=false`,
  `backward_called=false` and `optimizer_created=false`
  (`audit_actionformer_s0_assignment_support.py:621-632,735-785`).

## B. Official comparability: PASS with a strict claim qualifier

- The dense arm uses the upstream ActionFormer THUMOS I3D config. The K384 arm
  changes only the declared sparse-head intervention: enabled, total budget
  384, `stratified_uniform`, seed `1234567891`, and selected-query loss support
  (`configs/thumos_i3d_sparsehead_k384_uniform.yaml:18-28`). Dataset,
  optimizer, schedule, loss defaults and test settings remain matched.
- The launcher trains both arms sequentially from scratch, forbids resume,
  requires the terminal epoch-35 EMA checkpoint, runs the native official
  evaluation, and independently recomputes metrics from the raw predictions
  (`run_actionformer_official_matched_pair_n16r4.sbatch:270-369,455-456`).
- Source, data, feature, evaluator, runtime and official Soft-NMS identities
  are receipt-bound. Pair completion explicitly records same commit, seed,
  data, schedule, terminal EMA, evaluator and environment
  (`run_actionformer_official_matched_pair_n16r4.sbatch:482-535`).
- The valid comparator is the same-run dense `66.583013`, not the released
  `66.833392` anchor and not historical PhysTime/OpenTAD values.
- This is an official-comparable single-seed negative screening result. It is
  not a five-seed main result, not a released-number reproduction, and was
  preregistered `paper_main_table_eligible=false`.

## C. Score normalization: PASS

- `ANETdetection.evaluate` computes per-class AP and averages over classes and
  tIoU thresholds (`libs/utils/metrics.py:199-250`).
- Predictions are sorted by their raw scores and matched once to dataset GT;
  recall is normalized by the number of dataset GT instances
  (`libs/utils/metrics.py:253-333`).
- No metric is divided by a statistic of the model's own predictions. The
  post-NMS calibration summaries are descriptive and are not substituted for
  official AP.

## D. Result existence and exact-number binding: PASS

- Job `1205599` is `COMPLETED 0:0`. Pair SHA-256 is
  `545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd`.
  Dense/sparse ARM SHA-256 values are
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
  `fc682cfb01b9ed6639f821938922051edc2afa55490f504170eb7e3a6fd49037`.
- Job `1205701` is `COMPLETED 0:0`. Attribution, diagnostics and suite
  SHA-256 values are
  `d0bffe87cfb582b1b0649da3833e9fe0147db5a0a78500b6700fb78019323afb` /
  `a6b7fa0c4a41aac75ae2fb4cb4fcfbe68cf48bc7d2c813b37485b35998838791` /
  `e71721cb07334f1b6abb09347a7b609e51d6da1ed4be864c190ed60433a197d6`.
- Job `1205799` is `COMPLETED 0:0`. Assignment suite/producer/rows/sample
  SHA-256 values are
  `475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567` /
  `ca7e97a4124e49eb2ac30e949bcd50d4407998e8518eb72c8c6c8c8bb3f86e8b` /
  `a73b6f69c8655fed584774d131388ebf4974cf001f3efd9f492a952251e96b7f` /
  `d02f1de5fe9320cea47011b4af253001db77ecb7aadff83b8185a3350c7c55f4`.
- Local audit copies with duplicate basename files were explicitly renamed to
  preserve both dense/sparse and both off-diagonal cells. Those convenience
  copies are not primary provenance; the remote receipts above are.

## E. Actual invocation / dead-code boundary: PASS

- The official launcher directly calls ActionFormer `train.py` and `eval.py`
  for both arms and calls the independent raw evaluator
  (`run_actionformer_official_matched_pair_n16r4.sbatch:286-369`).
- The 2x2 launcher directly evaluates both off-diagonal cells and constructs
  contrasts from all four frozen cells
  (`run_actionformer_official_s0_attribution_n16r4.sbatch:351-463,505-577`).
- The assignment launcher directly calls the focused tests and audit producer,
  then validates all non-training and claim-boundary fields before writing its
  completion artifact
  (`run_actionformer_official_s0_assignment_audit_n16r4.sbatch:168-249`).

## F. Scope: WARN

- The official S0 contains one paired seed. The 2x2 adds no new seed and no new
  training. The assignment audit contains 64 deterministic training windows.
- No synchronized end-to-end detector-pipeline cost, confidence interval,
  peak-memory or energy result exists for the rejected method.
- Consequently, “robust”, “general”, “efficient”, “accuracy preserving” and
  main-table positive claims are unsupported.

## G. Diagnostic boundary: PASS

- The 2x2, post-NMS and assignment/support results are real, receipt-bound and
  useful for mechanism attribution.
- They remain `paper_main_table_eligible=false`. The assignment audit uses
  training-split GT and may be reported only as a mechanism diagnostic, not as
  test performance or model selection.
- Post-NMS predictions cannot reveal pre-NMS recall, suppressed identities,
  unretained logits or counterfactual NMS behavior.

## H. Mechanistic inference boundary: PASS with qualifiers

- The no-retraining 2x2 makes hard K384 execution/query coverage the dominant
  observed factor (`-20.7082 pp` average main effect), while selected-loss
  training is secondary (`-1.9552 pp`) and their interaction is small
  (`+0.1810 pp`).
- Post-NMS class-aware and class-agnostic recall, fixed-topK recall and overlap
  all collapse, which rejects a calibration-only explanation.
- The assignment audit shows that the exact K384 support retains only
  `16.9423%` of dense positives and leaves `395/804` sampled training GT with
  no candidate and `427/804` with no assignment.
- These observations support structural proposal/support coverage loss as the
  leading explanation for this exact intervention. They do not prove a
  universal theorem about all sparse heads, isolate pre-NMS NMS causality, or
  establish the performance of a redesigned method.

## Paper-safe claim set

1. Under the frozen official-comparable one-seed S0 protocol, the exact
   K384 stratified-uniform selected-loss intervention scores `43.919699`
   versus its same-run dense control `66.583013`, so it is decisively rejected.
2. Frozen-checkpoint 2x2 attribution indicates that K384 query/proposal
   execution is the dominant observed source of degradation, with selected-loss
   training a smaller secondary source.
3. Training-split assignment and post-NMS diagnostics are consistent with a
   severe support/coverage bottleneck.

## Forbidden claim set

- Do not place S0, 2x2 or assignment rows in the positive main table.
- Do not claim speedup, energy saving, robustness, state of the art,
  accuracy preservation or general SparseHead failure/success.
- Do not compare the sparse arm causally with the released `66.833392` or
  historical `63.xx` values; the causal baseline is the paired `66.583013`.
- Do not claim calibration/NMS or selected-loss is the sole cause.
- Do not claim an external cross-model integrity PASS.

