# D2S / PA-TAD Implementation Audit

Date: 2026-09-08. Scope: the two user-stopped matrices, D160/G96 controls,
selected final EMA checkpoints, and the new evaluation-only entry. This is not
an audit of every historical ZoomToken/BA-FDR/DUCA/ET-TRC worktree.

## Verdict

The previous claim of implementation completeness was premature. Four launch/
evaluation errors and one parameter-reporting error are confirmed and corrected.
No model-equation, physical-skip, or PA feature-flow error was confirmed
in the active training implementation. That is a bounded conclusion, not proof
that all possible inputs or the full GPU evaluation are correct.

Training source: `21aa2945b934a0dba469a517c224efe9b30d3967`.
Failed evaluation source: `1f600f8fa9c696c0d476970b353a05f7e004a05f`.
Subsequently failed diagnostic source: `d34d51d29900532a399c5a1edc0f2563c5b3641c`.
Replacement diagnostic source: `2a8639b6650cf3b04d246fe1efc9380e913ec341`.
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
regression assertion for initialization-before-module-load. Replacement GPU
PRECHECK jobs 1280197/1280199 have now completed successfully.

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

### P1: Formal matrix CLI passes a payload digest to a file-digest check

`tools/bata/continuous_roi_s2_v3_full200_compute_eval.py:1188` selected the seal's
internal `seal_sha256`, but `begin_single_gt_open` at line 1122 compares the digest
of the entire serialized file. These are different for a valid prediction seal.
The original nine-cell `evaluate-matrix` command therefore fails with
`prediction seal hash mismatch` before opening metric GT.

Correction: pass the existing whole-file digest expected by the callee. Extend
the real nine-bundle seal test to exercise the CLI, stopping before GT loading.
The test failed with the original exception before the fix and passes afterward;
the one-shot marker still rejects a second open. This is a later shared formal-CLI
fix, not part of d34d51d2. The active diagnostic has its own opening entry and
does not call this CLI, so it does not need cancellation or resubmission.

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
accuracy result was available at the 18:23 Asia/Shanghai status check. GPU
PRECHECK has passed; full diagnostic evaluation still has to finish.

## Verification And Review Limits

- Local shared/diagnostic/C3 tests: 50 passed.
- D2S-context tests: 27 passed; PA-context tests: 23 passed (overlapping suites).
- Remote D2S/PA architecture tests: 8 passed on CPU.
- Remote d34d51d2 diagnostic/statistics tests: 24 passed on CPU. After the
  additional formal-CLI fix, the corresponding local suite has 25 passing tests.
- Nine actual checkpoint states inspected; three production model classes
  constructed and strict-loaded on CPU. These checks did not open metric GT.
- Three fresh default-role read-only reviewers were used. Review is same-family
  and provisional; their initial claims were checked against code or execution.
  The confirmed execution errors above come from concrete reproduction/logs,
  not an unverified reviewer verdict.

## Deployment Snapshot: 2026-09-08 18:23 Asia/Shanghai

| Route | GPU PRECHECK | Complete-population diagnostic |
| --- | --- | --- |
| D2S, six selected cells | 1280197 COMPLETED 0:0, 1m29s | 1280198 RUNNING on g0006 |
| PA-TAD, three selected cells | 1280199 COMPLETED 0:0, 1m10s | 1280200 RUNNING on g0050 |

Both PRECHECK logs record completion and publish `control/diagnostic_plan.json`.
They verify all selected real checkpoint states, strict EMA loading, complete
loader size and one real label-free GPU forward per cell. This is not yet
all-window evaluation completion or a final accuracy result.

The immutable active source is
`/data/run01/sczc063/yuzibo/projects/zoomtoken_stopped_matrix_eval_d34d51d2_src`.
Output roots are
`/data/run01/sczc063/yuzibo/projects/d2s_user_stop_diagnostic_20260908_d34d51d2_r2`
and
`/data/run01/sczc063/yuzibo/projects/patad_user_stop_diagnostic_20260908_d34d51d2_r2`.
Model/config trees still match the original training commit exactly. The
30-minute monitor is bound to these jobs and must not restart cancelled training.

Local source: `E:/DeskTop/TAD/zoomtoken_stopped_matrix_eval_20260908`.
GitHub branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-stopped-matrix-eval-20260908

## Follow-up P1: Official Zero-Duration Predictions Rejected By The Bundle

The 18:23 running snapshot above is superseded: evaluation jobs 1280198/1280200
both FAILED 1:0 around 19:05, after the first D160 seed-4407 full inference/NMS
pass but before publishing predictions or opening metric GT. The exception at
`continuous_roi_s2_v3_full200_compute_eval.py:92` rejected finite/positive-duration
validation without printing the failing values. No final accuracy exists.

Actual production CPU clipping, two-decimal segment rounding and unchanged NMS
reproduce the exception with finite zero-duration predictions. Their endpoints
collapse at video boundaries or through rounding; the official evaluator keeps
them as false positives. The exact old runtime row is unavailable because the
failed bundle was not published, so this reproduction is not presented as a
recovered original prediction.

The repair accepts finite zero-duration detections unchanged, still rejects
reversed/nonfinite values, and includes UID/values in any remaining error. It
also moves PRECHECK's early return after the real NMS/serializer path. One-window
PRECHECK remains engineering evidence and never publishes a full-population
prediction file. This closes a real verification gap in the previous PRECHECK.
Local diagnostic/statistics/C3 regressions: 53 passed. New tests compare zero-
duration false-positive AP against the unchanged official function. Preserve
all old artifacts; only evaluation repair/redeployment is authorized.

Replacement source 2a8639b6 is pushed and synchronized as an immutable clean
remote worktree. Production postprocessing, diagnostic and statistics tests pass
34/34 on remote CPU. Slurm accepted D2S PRECHECK/evaluation 1280308/1280309 and
PA-TAD 1280310/1280311, with afterok dependencies and fresh 2a8639b6_r3 output
roots. The existing 30-minute monitor now follows these jobs. Their submission
does not constitute completed GPU admission or full-population metric evidence.
