# Current DUCA-TAD / C3 Sparse Acquisition Experiment Map And GPT Review Prompt

> **注意 (Note)**：本文件为 2026-07-06 早期实验地图与外部审阅 Prompt 历史记录。  
> 涵盖 12 大实验体系、全量配置、工具、测试与实测数据的最新全局汇总记录，请参阅：  
> 👉 [docs/CONSOLIDATED_EXPERIMENTS_RECORD.md](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/docs/CONSOLIDATED_EXPERIMENTS_RECORD.md)

Updated: 2026-07-07 +0800

Status: active engineering, deployment, and external-review record

Primary branch: `codex/gas-vt-stage23-detector-aware-20260706`

Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

Review target implementation commit: `3f0041c9847ffc50b43a55d3845ec37ec089c026`

Commit URL: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/3f0041c9847ffc50b43a55d3845ec37ec089c026`

## 1. Research Objective

The project studies a pre-backbone temporal acquisition module for Temporal
Action Detection (TAD). The intended contribution is not a standalone
action/background classifier. The intended contribution is a deployable sparse
frame/snippet acquisition module that decides which temporal observations should
be sent into AdaTAD/OpenTAD so that expensive backbone/detector computation is
reduced while high-IoU temporal localization is preserved.

The current research question is:

> Can a detector-aware, gap-aware, true-time sparse acquisition policy select
> substantially fewer temporal observations before the backbone while preserving
> or improving AdaTAD detector mAP, especially high-IoU mAP?

The long-term target is a pluggable pre-backbone acquisition adapter:

```text
TemporalAcquisitionAdapter =
    low-cost scout / selector inputs
    + acquisition policy
    + hard/ST sparse sampler
    + true-time metadata adapter
    + AdaTAD-compatible detector bridge
```

The strict claim gate is:

- No detector mAP, no performance claim.
- No matched-budget baselines, no acquisition-method superiority claim.
- No train-only dense teacher provenance, no detector-aware utility claim.
- No detector loss backpropagating into selector parameters in a real detector
  path, no end-to-end claim.
- No dynamic-budget train calibration, no dynamic-budget deployment claim.

## 2. Implemented Experiment Families

### 2.1 PAction learned strict ledger -> AdaTAD mAP

Purpose:

- Establish a strict learned `p_action` policy baseline.
- Generate deployable fixed and dynamic ledgers.
- Validate no uniform fill, no uniform scaffold, no test teacher/GT/cache
  leakage.
- Train AdaTAD on selected observations and report detector mAP.

Representative surfaces:

- `tools/bata/train_paction_acquisition_policy.py`
- `tools/bata/run_paction_learned_policy_ledger_pipeline.py`
- `tools/bata/validate_paction_learned_policy_ledger.py`
- `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py`
- `scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh`

Variants:

- `learned_fixed_384`
- `learned_fixed_768`
- `learned_dynamic`

Interpretation:

- This is not end-to-end.
- It is an offline strict ledger route where the detector cannot teach the
  selector.
- It is a necessary baseline, not the final paper method.

### 2.2 GAS-VT Stage0/1 -> AdaTAD mAP

Purpose:

- Move beyond raw `p_action` by adding gap-aware sequential state and value
  transport.
- Test whether sparse ledgers with boundary bracket, action interior, and
  gap/hole control are useful to AdaTAD.
- Provide Stage1 detector mAP evidence for sparse pre-backbone acquisition.

Representative surfaces:

- `tools/bata/gas_vt_paction_policy.py`
- `tools/bata/train_gap_aware_acquisition_policy.py`
- `tools/bata/apply_gap_aware_acquisition_policy.py`
- `tools/bata/run_gap_aware_ledger_pipeline.py`
- `configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py`

Variants:

- `gas_vt_fixed_384`
- `gas_vt_fixed_768`
- `gas_vt_dynamic`

Interpretation:

- This is still offline ledger training.
- It answers whether a stronger sparse ledger helps detector mAP.
- It does not yet answer whether the selector is detector-aware or end-to-end.

### 2.3 Official dense AdaTAD teacher route

Purpose:

- Train or locate a clean selector-free dense AdaTAD teacher under official-like
  settings.
- Provide the teacher checkpoint and config needed by Stage2 detector-aware
  utility export.
- Verify the dense teacher setting before using it as a detector critic.

Implemented surfaces:

- `configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py`
- `scripts/run_c3_dense_adatad_teacher_full_train_gpu.sh`
- `tests/test_c3_dense_adatad_teacher_full_train.py`
- `docs/methods/2026-07-06-dense-teacher-deployment-evidence.md`

Current state:

- Local and remote precheck passed.
- Full dense teacher run is queued and waits for a GPU to free.
- No Stage2 detector-aware utility claim is unlocked until this teacher exists
  and train-only utility export is validated.

### 2.4 Stage2 detector-aware teacher utility / offline selector

Purpose:

- Replace `p_action`-only supervision with detector utility.
- Use dense AdaTAD teacher signals such as point responsibility, signed
  utility, cls/reg contribution, saliency, and counterfactual value to train an
  acquisition policy.
- Keep teacher utility train-only and forbid teacher/GT/cache leakage in
  val/test/deploy artifacts.

Implemented surfaces:

- `tools/bata/detector_teacher_utility.py`
- `tools/bata/detector_aware_acquisition_policy.py`
- `tools/bata/train_detector_aware_acquisition_policy.py`
- `tools/bata/apply_detector_aware_acquisition_policy.py`
- `tools/bata/convert_detector_aware_samples_to_value_transport_ledger.py`
- `tools/bata/run_detector_aware_ledger_pipeline.py`
- `tools/bata/validate_detector_aware_policy_ledger.py`
- `tools/bata/validate_duca_stage23_precheck.py`
- `configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py`
- `configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck.py`
- `configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck_exec.py`
- `scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh`
- `scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh`

Current engineering status:

- Stage2 code path and fail-closed precheck gate are implemented.
- Direct trainer now requires `signed_frame_utility`; it no longer silently
  treats unsigned `frame_utility` as `signed_detector_utility_v1`.
- Stage2 still needs a real dense AdaTAD teacher checkpoint plus train-only
  utility export before the precheck can pass.

Interpretation:

- Offline Stage2 is detector-aware in supervision, but not end-to-end.
- It can answer whether AdaTAD teacher utility trains a better acquisition
  policy than `p_action` only, once the dense teacher exists.

### 2.5 Stage3 TrueTime ST hard selector / AdaTAD joint-training candidate

Purpose:

- Put a hard/ST selector in the detector training graph.
- Preserve selected-axis and true-time metadata so detector predictions can be
  mapped back to physical time.
- Prove detector loss can reach selector parameters in a real ActionFormer
  detector forward/backward path.

Implemented surfaces:

- `opentad/models/selectors/truetime_joint_selector.py`
- `tools/bata/run_truetime_joint_selector_smoke.py`
- `tools/bata/run_truetime_joint_selector_precheck.py`
- `tools/bata/validate_truetime_joint_selector_precheck.py`
- `tools/bata/validate_duca_stage23_precheck.py`
- `configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py`
- `configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py`
- `configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck_exec.py`
- `scripts/run_c3_truetime_joint_selector_adatad_gpu1.sh`
- `scripts/run_duca_stage3_truetime_precheck_gpu1.sh`

Current engineering status:

- The precheck route uses a tiny synthetic real ActionFormer loss path, not a
  384/768 mAP run.
- The full-run wrapper no longer delegates to the older smoke launcher.
- `precheck_only_default=True` is separated from `current_run_precheck_only`,
  avoiding the previous `PRECHECK_ONLY=0` gate contradiction.
- Full-run unlock is bound to the current Stage3 config, execution config, and
  gradient-proof JSON SHA256 values so a stale PASS summary cannot unlock a
  different run.
- Full train remains gated by a PASS precheck summary, an explicit
  `ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN=1`, and Slurm allocation/step.

Interpretation:

- This is the first code path aimed at honest end-to-end selector + detector
  optimization.
- It is still a precheck/full-train candidate until a real sparse AdaTAD run
  reports detector mAP.

### 2.6 Stage4 curriculum / bilevel evidence gate

Purpose:

- Define fail-closed evidence requirements before claiming a stable
  detector-aware curriculum or bilevel training method.
- Require teacher utility provenance, no leakage, selector gradient evidence,
  sparse mAP, and collapse diagnostics.

Implemented surfaces:

- `tools/bata/validate_stage4_detector_aware_truetime_curriculum.py`
- `tests/test_stage4_detector_aware_truetime_curriculum.py`
- `docs/methods/2026-07-06-detector-aware-truetime-cvpr-route.md`

Interpretation:

- This is currently a gate/protocol, not a full curriculum training
  implementation.
- A real curriculum still needs dense warmup, selector pretraining, sparse
  detector training, and joint fine-tuning schedules.

## 3. Currently Deployed Remote Experiments

Remote allocation:

- Job: `1118197`
- Node: `g0030`
- Environment: `/data/run01/sczc063/yuzibo/conda_envs/opentad`

### 3.1 GAS-VT Stage0/1 on GPU0

- RUN_TAG: `c3_gas_vt_stage01_gpu0_provok_20260706_151908_+0800`
- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/gas_vt_adatad/c3_gas_vt_stage01_gpu0_provok_20260706_151908_+0800`
- Driver log: `driver_resume.log`
- Variants: `gas_vt_fixed_384`, `gas_vt_fixed_768`, `gas_vt_dynamic`
- Latest observed state: `gas_vt_fixed_384` still training, past epoch 31.
- Latest observed mAP snapshot: average mAP around 45.87 at the latest eval.

### 3.2 PAction learned strict ledger on GPU1

- RUN_TAG: `c3_paction_learned_g30_gpu1_d413df8_20260706_214309_+0800`
- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/paction_learned_adatad/c3_paction_learned_g30_gpu1_d413df8_20260706_214309_+0800`
- Driver log: `driver.log`
- Code snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_gasvt_d413df8_g30_20260706`
- Commit: `d413df83ad223e56039f7b1530c88d84f288190e`
- Variants: `learned_fixed_384`, `learned_fixed_768`, `learned_dynamic`
- Latest observed state: `learned_fixed_384` was training around epoch 9.

### 3.3 Dense AdaTAD teacher full run queue

- RUN_TAG: `c3_dense_adatad_teacher_full_queued_20260706_2338_+0800`
- Waiter PID: `322183`
- Snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_dense_teacher_366b9951ef39_20260706_233128`
- Driver log: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_queued_20260706_2338_+0800/driver.log`
- State: waiting for either active GPU experiment to finish, then launches full
  dense teacher training inside Slurm job `1118197`.

### 3.4 Stage3 TrueTime precheck queue

- RUN_TAG: `duca_stage3_precheck_3f0041c_queued_20260707_001814`
- Waiter PID: `687773`
- Snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_stage23_3f0041c9847f_20260707_001421`
- Driver log: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_stage3_truetime/duca_stage3_precheck_3f0041c_queued_20260707_001814/driver.log`
- State: waiting for the current GPU1 PAction run to exit, then launches
  `PRECHECK_ONLY=1` in Slurm job `1118197`.

### 3.5 Historical / no longer primary

- Old GPU0 model zoo was intentionally interrupted and must not be restarted
  automatically.
- Old PAction full run `c3_paction_learned_adatad_map_full_direct_20260706_101331_+0800`
  failed earlier and is not a current monitored target.

## 4. Experiments Still Needed

### 4.1 Dense teacher evidence

Need:

- Complete official-like dense AdaTAD teacher training.
- Record teacher config hash, checkpoint hash, and mAP.
- Export train-only detector utility.
- Validate utility split provenance and no val/test leakage.

### 4.2 Stage2 full matrix

Compare under matched budgets:

- Dense AdaTAD
- Uniform 384 / 768
- Random 384 / 768
- `p_action` only 384 / 768 / dynamic
- GAS-VT 384 / 768 / dynamic
- Detector-aware 384 / 768 / dynamic

Required metrics:

- Average mAP
- mAP@0.3/0.4/0.5/0.6/0.7
- Boundary support
- Action coverage
- Max gap / p95 gap / CVaR hole
- Compute, memory, wall-clock

### 4.3 Stage3 online dense-sparse co-training

Goal:

- Avoid relying solely on a pre-trained dense teacher.
- Let AdaTAD and selector co-optimize in one training loop.

Recommended design:

- Dense branch: full or near-full AdaTAD branch computes online
  responsibility/saliency.
- Sparse branch: ST hard selector chooses K or dynamic-budget observations and
  sparse AdaTAD predicts on selected frames.
- Teacher signal: online dense branch with stop-gradient or EMA stabilization.
- Selector losses: sparse detector loss, dense-to-sparse distillation, budget
  loss, CVaR max-hole loss, boundary bracket loss, action interior loss.
- Evaluation path: selector + sparse AdaTAD only; dense branch not used at
  inference.

This is the first route that can be honestly positioned as end-to-end if
detector loss reaches selector parameters in the real training graph.

### 4.4 Stage4 curriculum / bilevel stabilization

Suggested schedule:

1. Dense AdaTAD warmup or EMA teacher warmup.
2. Train-only utility export or online utility extraction.
3. Selector pretraining.
4. Frozen selector sparse detector training.
5. Partial unfreeze sparse detector + selector.
6. Joint ST hard selector fine-tuning.
7. Dynamic budget calibration and matched-average-K comparison.

Collapse diagnostics:

- Selector entropy
- Selected count distribution
- Per-video max hole
- Boundary bracket coverage
- Action interior coverage
- Sparse vs dense logit/proposal agreement
- High-IoU mAP trend

## 5. Known Risks And Claim Locks

Current risks:

- Sparse selector may protect low-IoU recall but damage high-IoU localization.
- Hard gap repair may hide manual rule effects.
- Dynamic budget may be rank-derived rather than calibrated utility-derived.
- Offline detector-aware utility may not translate to deploy-time observable
  selection.
- Tiny synthetic Stage3 gradient precheck may be mistaken for full THUMOS
  end-to-end evidence.
- Stage3 full-run hash binding now prevents stale PASS summary reuse, but it is
  still only a launch gate; it is not detector mAP evidence.

Claim locks:

- No detector mAP, no performance claim.
- No matched-budget baselines, no acquisition-method superiority claim.
- No selector gradient in real detector full training, no end-to-end claim.
- No train-only provenance manifest, no detector-aware utility claim.

## 6. Complete GPT / Pro Review Prompt

Copy the following prompt into GPT/Pro. It uses the latest pushed GitHub commit.

```text
You are a strict senior CVPR/ICCV/NeurIPS reviewer and a code auditor for
Temporal Action Detection systems.

Please review this GitHub repository and commit line by line:

Repository:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702

Branch:
codex/gas-vt-stage23-detector-aware-20260706

Commit:
3f0041c9847ffc50b43a55d3845ec37ec089c026

Commit URL:
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/3f0041c9847ffc50b43a55d3845ec37ec089c026

Context:

We are developing DUCA-TAD / C3 sparse temporal acquisition for OpenTAD/AdaTAD.
The goal is a deployable pre-backbone temporal frame/snippet selection module
for Temporal Action Detection. It should reduce temporal observations before
the expensive backbone while preserving or improving detector mAP, especially
high-IoU localization. The selector should eventually become detector-aware and
end-to-end trainable, so that AdaTAD detector loss or detector utility can teach
the selector what observations matter.

Please check whether the current implementation matches this objective or has
drifted into the wrong problem, such as merely learning actionness, replaying
uniform sampling, leaking teacher/GT/cache information, or adding engineering
gates without proving detector performance.

Important files and directories to inspect:

1. Stage0/1 GAS-VT / p_action strict ledger:
   - tools/bata/gas_vt_paction_policy.py
   - tools/bata/train_gap_aware_acquisition_policy.py
   - tools/bata/apply_gap_aware_acquisition_policy.py
   - tools/bata/run_gap_aware_ledger_pipeline.py
   - configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py
   - tests/test_gas_vt_paction_policy.py
   - tests/test_c3_gas_vt_adatad_full_train.py

2. PAction learned baseline:
   - tools/bata/train_paction_acquisition_policy.py
   - tools/bata/run_paction_learned_policy_ledger_pipeline.py
   - tools/bata/validate_paction_learned_policy_ledger.py
   - configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py
   - scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh
   - tests/test_paction_learned_ledger_pipeline.py

3. Dense AdaTAD teacher:
   - configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py
   - scripts/run_c3_dense_adatad_teacher_full_train_gpu.sh
   - tests/test_c3_dense_adatad_teacher_full_train.py
   - docs/methods/2026-07-06-dense-teacher-deployment-evidence.md

4. Stage2 detector-aware teacher utility and selector:
   - tools/bata/detector_teacher_utility.py
   - tools/bata/detector_aware_acquisition_policy.py
   - tools/bata/train_detector_aware_acquisition_policy.py
   - tools/bata/apply_detector_aware_acquisition_policy.py
   - tools/bata/convert_detector_aware_samples_to_value_transport_ledger.py
   - tools/bata/run_detector_aware_ledger_pipeline.py
   - tools/bata/validate_detector_aware_policy_ledger.py
   - tools/bata/validate_duca_stage23_precheck.py
   - configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py
   - configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck.py
   - configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck_exec.py
   - scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh
   - scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh
   - tests/test_detector_teacher_utility.py
   - tests/test_detector_aware_acquisition_policy.py
   - tests/test_detector_aware_ledger_pipeline.py
   - tests/test_c3_detector_aware_adatad_full_train.py

5. Stage3 TrueTime / ST hard selector:
   - opentad/models/selectors/truetime_joint_selector.py
   - tools/bata/run_truetime_joint_selector_smoke.py
   - tools/bata/run_truetime_joint_selector_precheck.py
   - tools/bata/validate_truetime_joint_selector_precheck.py
   - configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py
   - configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py
   - configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck_exec.py
   - scripts/run_c3_truetime_joint_selector_adatad_gpu1.sh
   - scripts/run_duca_stage3_truetime_precheck_gpu1.sh
   - tests/test_truetime_joint_selector.py
   - tests/test_truetime_joint_selector_config.py
   - tests/test_truetime_detector_selector_integration.py
   - tests/test_truetime_geometry.py

6. Stage4 curriculum/evidence gates:
   - tools/bata/validate_stage4_detector_aware_truetime_curriculum.py
   - tests/test_stage4_detector_aware_truetime_curriculum.py
   - docs/methods/2026-07-06-detector-aware-truetime-cvpr-route.md
   - docs/methods/2026-07-06-current-experiment-map-and-gpt-review-prompt.md

Review tasks:

1. Visibility and reproducibility:
   - Confirm the commit is visible.
   - Confirm which files were actually inspected.
   - Identify any missing scripts/configs/tests needed to reproduce claims.

2. Objective alignment:
   - Does the code implement a pre-backbone sparse temporal acquisition module
     for TAD, or has it drifted into an actionness classifier / ledger
     engineering exercise?
   - Does each implemented stage directly support the final objective?
   - Which parts are only diagnostics, gates, or smoke tests rather than real
     training methods?

3. Leakage and deployment correctness:
   - Check that val/test/deploy ledgers do not contain teacher outputs, GT
     labels, GT segments, proposal caches, dense prediction caches, or any
     forbidden supervision payload.
   - Check that teacher utility is train-only and that split provenance is
     explicit.
   - Check that fixed and dynamic ledgers do not use uniform fill or uniform
     scaffold.

4. Geometry and time-axis correctness:
   - Check selected-axis vs dense/true-time coordinate handling.
   - Verify whether selected-axis proposals are remapped to physical time before
     NMS and mAP evaluation.
   - Look for off-by-one, valid_len, selected_count, fps, feature_stride,
     window_offset, and duration mistakes.

5. Stage0/1 / GAS-VT:
   - Does GAS-VT genuinely improve over p_action-only or mostly encode
     hand-crafted gap/coverage priors?
   - Are boundary bracket, action interior, CVaR max-hole, and gap losses
     implemented in a way that helps detector mAP rather than only ledger
     metrics?
   - What ablations are mandatory?

6. Dense AdaTAD teacher:
   - Is the dense teacher route close enough to official AdaTAD/OpenTAD settings?
   - Are eval/checkpoint epochs, pretrained weights, config inheritance, and
     dataset settings aligned with official practice?
   - What must be logged before the teacher can be used for Stage2 utility?

7. Stage2 / detector-aware utility:
   - Is the teacher utility semantically meaningful for AdaTAD? Does it capture
     proposal responsibility, cls/reg loss, saliency, or counterfactual utility
     correctly?
   - Is signed utility handled correctly, or are harmful negative utilities
     collapsed into absolute high value?
   - Is dynamic budget calibrated using train-split marginal gain and frozen for
     val/test, or merely rank-derived?
   - What exact code changes are needed to make Stage2 scientifically strong?

8. Stage3 / end-to-end selector:
   - Does the current code prove that detector loss reaches selector parameters
     in a real AdaTAD/ActionFormer forward/backward path?
   - If not, identify the exact missing integration points.
   - Review `scripts/run_duca_stage3_truetime_precheck_gpu1.sh` and confirm it
     no longer falls back to the old smoke launcher for full-run mode.
   - Provide key implementation code for an online dense-sparse co-training route
     where selector and AdaTAD are optimized in the same training loop.
   - The design should not require a fully pre-trained dense teacher, although
     it may use an EMA dense branch or warmup for stability.

9. Stage4 / curriculum and bilevel training:
   - Is Stage4 currently implemented as a real training curriculum or just an
     evidence gate?
   - Provide a detailed curriculum plan:
     a. dense warmup or EMA online teacher,
     b. train-only utility / online utility,
     c. selector pretraining,
     d. frozen selector sparse AdaTAD,
     e. joint ST fine-tuning,
     f. dynamic budget calibration,
     g. collapse diagnostics.

10. Experiments:
    - Design the complete experiment matrix needed for a CVPR-level claim.
    - Include dense AdaTAD, uniform, random, p_action-only, delta-p_action,
      GAS-VT, detector-aware offline, and end-to-end online selector variants.
    - Require matched budgets, matched compute, multiple seeds, high-IoU metrics,
      wall-clock, memory, and statistical uncertainty.
    - Specify which experiments must run first and which can wait.

11. Key code:
    - Provide concrete pseudocode or patch-level code for:
      a. online dense-sparse co-training forward pass,
      b. ST/Gumbel hard top-k selector,
      c. selected-axis-to-true-time remap,
      d. detector utility extraction,
      e. dynamic budget calibration,
      f. recursive no-leakage validator,
      g. selector gradient proof in a real detector path,
      h. stale precheck-summary hash binding for Stage3 full-run gate.

Please output:

1. PASS/WARN/HOLD/FAIL verdict.
2. Top P0/P1/P2 issues with file-level references.
3. Whether current implementation supports each claim:
   - sparse acquisition improves TAD mAP,
   - detector-aware acquisition improves over p_action-only,
   - dynamic budget is calibrated,
   - end-to-end selector + AdaTAD is implemented,
   - high-IoU localization is protected.
4. A corrected experiment roadmap from now to a publishable CVPR-style paper.
5. Concrete critical code snippets or patch sketches for the missing pieces.
6. A concise paper story: problem, insight, method, novelty, and evidence
   required.
```
