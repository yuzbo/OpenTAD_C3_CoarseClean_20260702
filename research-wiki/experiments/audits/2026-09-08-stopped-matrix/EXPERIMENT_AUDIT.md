# D2S / PA-TAD Implementation Audit

Date: 2026-09-08. Scope: the two user-stopped matrices, D160/G96 controls,
selected final EMA checkpoints, and the new evaluation-only entry. This is not
an audit of every historical ZoomToken/BA-FDR/DUCA/ET-TRC worktree.

## Verdict

The previous claim of implementation completeness was premature. Two execution
blockers and one parameter-reporting error are confirmed and corrected in this
change. No model-equation, physical-skip, or PA feature-flow error was confirmed
in the active training implementation. That is a bounded conclusion, not proof
that all possible inputs or the full GPU evaluation are correct.

Training source: `21aa2945b934a0dba469a517c224efe9b30d3967`.
Failed evaluation source: `1f600f8fa9c696c0d476970b353a05f7e004a05f`.
All original checkpoints and receipts remain untouched. No training resumes.

## Confirmed Findings

### P1: Non-login Slurm shell has no module command

`scripts/submit_zoomtoken_stopped_matrix_eval_n16r4.sbatch:11` originally called
`module load` without initializing Environment Modules. Both real PRECHECK jobs
failed at this exact statement: `1280173` on g0017 and `1280174` on g0087,
exit `127:0`, elapsed one second. Their dependent evaluations `1280175/1280176`
were cancelled without starting. No prediction or metric was produced.

Correction: explicitly source `/usr/share/modules/init/bash` before `module
load`. The same non-login-shell sequence was then exercised successfully on
N16R4, loading CUDA 11.8, Miniforge 24.11, and the OpenTAD environment. Add a
regression assertion for initialization-before-module-load. A fresh GPU PRECHECK
is still required for the complete entry.

Remote error files:
- `/data/run01/sczc063/yuzibo/projects/d2s_user_stop_diagnostic_20260908_1f600f8f_r1/logs/precheck-1280173.err`
- `/data/run01/sczc063/yuzibo/projects/patad_user_stop_diagnostic_20260908_1f600f8f_r1/logs/precheck-1280174.err`

### P1: Task-local AP disagreed with the unchanged official evaluator on ties

`tools/bata/continuous_roi_s2_v3_full200_compute_eval.py:238` ordered equal-score
predictions by video/UID, and resolved equal GT overlaps by GT coordinates/UID.
The official `opentad/evaluations/mAP.py:301` and `:323` use NumPy
`argsort()[::-1]`. Predictions are rounded to four decimal places by the inference
entry, so equal-score predictions are a real supported input, not an exotic case.

Reproduction: one GT `[0,1]`; predictions `[0,1]` and `[2,3]`, both score 0.5.
The old local AP was 100%, but the unchanged official AP was 50%. The mandatory
1e-12 parity assertion would therefore abort a valid test after GT opening.
An equal-IoU GT example independently reproduces the second ordering mismatch.

Correction: local mAP now uses exactly the official reverse-argsort order for
both predictions and GT overlaps. Both examples are regression-tested against
the actual official function. Do not weaken the parity tolerance, perturb scores,
change NMS, or change the official evaluator. The separately frozen UID ordering
for short-action recall and boundary diagnostics remains unchanged.

### P2: Trainable-parameter count was collected before backbone freezing

`tools/bata/continuous_roi_s2_v3_full200_compute_train.py:543` counted all initial
`requires_grad` parameters. VideoMAE freezes its non-adapter layers at forward
time (`opentad/models/backbones/vit_adapter.py:1955`), and the optimizer excludes
the pretrained backbone. The old receipt field therefore overstated parameters
actually optimized. Production CPU construction, strict final EMA loading,
optimizer membership, and stored AdamW state give:

| Architecture | Old reported count | Parameters actually optimized |
| --- | ---: | ---: |
| D160/G96 shared baseline architecture | 49,582,504 | 27,702,568 |
| D2S candidate | 49,878,185 | 27,998,249 |
| PA-TAD candidate | 50,665,641 | 28,785,705 |

Correction for future receipts: preserve the initial count under
`initial_requires_grad_parameters`; populate `total_trainable_parameters` from
the optimizer's actual members after optimizer construction. This is metadata
only. It does not change any optimizer group, gradient, model, or checkpoint.
Old receipt values remain historical evidence and must use this qualification.

The two unoptimized `fc_norm` parameters are not a missing-training defect:
`return_feat_map=True` returns before `fc_norm` is called. Every forward-used
trainable parameter was present exactly once in the reconstructed optimizer.

## Confirmed Correct

- D2S local execution gathers only the selected 16 CPU uint8 chunks before device
  transfer (`d2s_videomae_wrapper.py:214-266`); the native-ragged path executes no
  heavy dummy tokens. It does not evaluate all 48 local crops and mask afterward.
- Global input uses 48 chunks; local input uses 16. The token budget is
  13,824 + 8,192 = 22,016, or 57.333% of 38,400 dense tokens. This is a token
  ratio, not a measured latency or full-operator ratio.
- Physical time/spatial indices survive native-ragged packing, and attention is
  bucketed by original clip, not one 8,192-token all-to-all window.
- PA L0/L1 receive residuals; L2-L5 follow the unmodified global branch
  (`pyramid_aware_asymmetric_proj.py:140-166`). All eight existing D2S/PA model
  architecture tests passed in the actual remote PyTorch environment on CPU.
- Each training epoch checks all 200 identities exactly once across two ranks;
  each epoch requires 100 successful updates. Skipped AMP attempts do not advance
  scheduler/EMA in this registered path.
- All nine selected real checkpoints were read on CPU. Their receipt hashes,
  final checkpoint hashes, EMA/identity/sample-trace bindings, epoch 59 and 6000
  updates agree. Every stored AdamW parameter state has step 6000. All EMA
  floating tensors are finite. This is stronger evidence than log counters alone.
- D2S candidate EMA gamma is 0.1060845256 (4407) and 0.1057929844 (4408); PA-TAD
  gamma is 0.0485219918 (4407). The residual gate did not remain stuck at zero.
- Representative production D160, D2S and PA models strictly loaded their final
  EMA checkpoints on CPU with no missing/unexpected keys. This is not a GPU or
  all-window execution claim.
- Evaluation input uses the GT-free annotation; all selected complete prediction
  bundles precede metric-GT opening. Full 211-video/792-window coverage and the
  original formal nine-cell seals remain enforced. Diagnostic results explicitly
  decline full-three-seed/paper admission.

## Rejected Suspicions

- A reviewer confused `ActionDataPreprocessor.preprocess` with the wrapper's
  separate `pre_processing_pipeline`. The local branch calls only the former;
  the 48-way rearrange is only called for the global branch. Installed MMAAction
  source and active wrapper source confirm this, so no model edit is justified.
- Python/NumPy augmentation RNG is not restored inside AMP retries, but dataset
  sampling/augmentation has already produced the same batch before the retry
  loop. No Python/NumPy randomness is used by these active model forwards.
  Their stochastic layers use the restored Torch RNG. No reachable defect was
  established for these configurations; do not add a general RNG framework.
- EMA value-order copying is not a demonstrated mismatch: it operates on a
  deepcopy of the same fixed module graph. No modules are replaced mid-training.
- Segment voting does not invalidate UID association in these configurations:
  the official NMS explicitly disables voting in its `multiclass=True` branch.

## Cost And Claim Limits

Original profile receipts report candidate/D160 operation-proxy ratios
0.5037660655 (D2S) and 0.5040854723 (PA-TAD). These come from a single fixed-shape
trace scaled by 792, not an empirical distribution of 792 real-video cost
measurements. The profiler explicitly records that trace type. Do not call this
measured end-to-end speedup or a measured cost distribution.

PA-TAD adds Q0/Q1 and changes information flow; it does not remove the high-level
global pyramid operators. Its proxy count is slightly above D2S, not below it.
Claims that PA alone eliminates extra high-level computation or has fewer
parameters than D2S are not supported by this implementation.

D2S has two complete matched seeds and PA-TAD one. These selected complete-data
weights do not complete either original three-seed nine-cell experiment. No new
accuracy result exists yet. GPU PRECHECK and complete diagnostic evaluation must
still finish before results can be reported.

## Verification And Review Limits

- Local shared/diagnostic/C3 tests: 50 passed.
- D2S-context tests: 27 passed; PA-context tests: 23 passed (overlapping suites).
- Remote D2S/PA architecture tests: 8 passed on CPU.
- Nine actual checkpoint states inspected; three production model classes
  constructed and strict-loaded on CPU. No canonical metric GT opened.
- Three fresh default-role read-only reviewers were used. Review is same-family
  and provisional; their initial claims were checked against code or execution.
  The confirmed execution errors above come from concrete reproduction/logs,
  not an unverified reviewer verdict.

Local source: `E:/DeskTop/TAD/zoomtoken_stopped_matrix_eval_20260908`.
GitHub branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-stopped-matrix-eval-20260908
