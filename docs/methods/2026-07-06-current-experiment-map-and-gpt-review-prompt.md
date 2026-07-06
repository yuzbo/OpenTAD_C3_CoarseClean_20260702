# Current DUCA-TAD / C3 Sparse Acquisition Experiment Map And Review Prompt

Updated: 2026-07-06

Status: active engineering and experiment coordination record

Primary branch: `codex/gas-vt-stage23-detector-aware-20260706`

Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

## 1. Research Objective

The project studies a pre-backbone temporal acquisition module for Temporal Action Detection (TAD). The intended contribution is not a standalone action/background classifier. The intended contribution is a deployable sparse frame/snippet acquisition module that decides which temporal observations should be sent into AdaTAD/OpenTAD so that expensive backbone/detector computation is reduced while high-IoU temporal localization is preserved.

The current research question is:

> Can a detector-aware, gap-aware, true-time sparse acquisition policy select substantially fewer temporal observations before the backbone while preserving or improving AdaTAD detector mAP, especially high-IoU mAP?

The strict claim gate is:

- Do not claim sparse acquisition improves TAD until detector mAP is measured.
- Do not claim detector-aware acquisition until it beats p_action-only and GAS-VT controls under matched budgets.
- Do not claim end-to-end training until detector loss demonstrably backpropagates into selector parameters in a real detector forward/backward path.
- Do not claim dynamic budget until the budget policy is calibrated on train only and frozen for val/test/deploy.

## 2. Implemented Experiment Families

### 2.1 PAction learned strict ledger -> AdaTAD mAP

Purpose:

- Establish a strict learned p_action policy baseline.
- Generate deployable fixed and dynamic ledgers.
- Validate no uniform fill, no uniform scaffold, no test teacher/GT/cache leakage.
- Train AdaTAD on selected observations and report detector mAP.

Implemented surfaces:

- `tools/bata/train_paction_acquisition_policy.py`
- `tools/bata/run_paction_learned_policy_ledger_pipeline.py`
- `tools/bata/validate_paction_learned_policy_ledger.py`
- `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py`
- `scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh`
- `tests/test_paction_learned_ledger_pipeline.py`
- `tests/test_c3_asformer_delta_ledger_full_train.py`

Variants:

- `learned_fixed_384`
- `learned_fixed_768`
- `learned_dynamic`

Current interpretation:

- This is not end-to-end.
- It is an offline strict ledger route where the detector cannot teach the selector.
- It is a necessary baseline, not the final paper method.

### 2.2 GAS-VT Stage0/1 -> AdaTAD mAP

Purpose:

- Move beyond raw p_action by adding gap-aware sequential state and value transport.
- Test whether sparse ledgers with boundary bracket, action interior, and gap/hole control are useful to AdaTAD.
- Provide the Stage1 detector mAP evidence for sparse pre-backbone acquisition.

Implemented surfaces:

- `tools/bata/gas_vt_paction_policy.py`
- `tools/bata/train_gap_aware_acquisition_policy.py`
- `tools/bata/apply_gap_aware_acquisition_policy.py`
- `tools/bata/run_gap_aware_ledger_pipeline.py`
- `configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py`
- `tests/test_gas_vt_paction_policy.py`
- `tests/test_c3_gas_vt_adatad_full_train.py`

Variants:

- `gas_vt_fixed_384`
- `gas_vt_fixed_768`
- `gas_vt_dynamic`

Current interpretation:

- This is still offline ledger training.
- It answers whether a stronger sparse ledger helps detector mAP.
- It does not yet answer whether the selector is detector-aware or end-to-end.

### 2.3 Stage2 detector-aware teacher utility / offline selector

Purpose:

- Replace p_action-only supervision with detector utility.
- Use AdaTAD dense teacher signals such as point responsibility, cls/reg loss, saliency, and counterfactual utility to train an acquisition policy.
- Keep train-only teacher utility and forbid teacher/GT/cache leakage in val/test/deploy artifacts.

Implemented surfaces:

- `tools/bata/detector_teacher_utility.py`
- `tools/bata/detector_aware_acquisition_policy.py`
- `tools/bata/train_detector_aware_acquisition_policy.py`
- `tools/bata/apply_detector_aware_acquisition_policy.py`
- `tools/bata/convert_detector_aware_samples_to_value_transport_ledger.py`
- `tools/bata/run_detector_aware_ledger_pipeline.py`
- `tools/bata/validate_detector_aware_policy_ledger.py`
- `configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py`
- `scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh`
- `tests/test_detector_teacher_utility.py`
- `tests/test_detector_aware_acquisition_policy.py`
- `tests/test_detector_aware_ledger_pipeline.py`
- `tests/test_c3_detector_aware_adatad_full_train.py`

Current interpretation:

- The utility and selector route exists, but a clean official dense AdaTAD teacher checkpoint / train-only utility export is still required for a decisive Stage2 experiment.
- Offline Stage2 is detector-aware in supervision, but not end-to-end.

### 2.4 Stage3 TrueTime ST hard selector smoke / integration prototype

Purpose:

- Introduce a selector that can make hard/ST temporal selections inside the detector training graph.
- Preserve true-time coordinate metadata so selected-axis predictions can be remapped to physical time before NMS/evaluation.
- Prove detector loss can reach selector parameters.

Implemented surfaces:

- `opentad/models/selectors/truetime_joint_selector.py`
- `tools/bata/run_truetime_joint_selector_smoke.py`
- `tools/bata/validate_truetime_joint_selector_precheck.py`
- `configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py`
- `tests/test_truetime_joint_selector.py`
- `tests/test_truetime_joint_selector_config.py`
- `tests/test_truetime_detector_selector_integration.py`
- `tests/test_truetime_geometry.py`

Current interpretation:

- The selector smoke path and gradient proof tooling exist.
- The current branch must still be reviewed carefully to confirm whether real AdaTAD detector forward uses the selector in the production training path, not only in a smoke harness.

### 2.5 Stage4 curriculum / bilevel evidence gate

Purpose:

- Define fail-closed evidence requirements before claiming a stable detector-aware curriculum or bilevel training method.
- Require teacher utility provenance, no leakage, selector gradient evidence, sparse mAP, and collapse diagnostics.

Implemented surfaces:

- `tools/bata/validate_stage4_detector_aware_truetime_curriculum.py`
- `tests/test_stage4_detector_aware_truetime_curriculum.py`
- `docs/methods/2026-07-06-detector-aware-truetime-cvpr-route.md`

Current interpretation:

- This is currently a gate/protocol, not a full curriculum training implementation.
- A real curriculum still needs dense teacher warmup, selector pretraining, sparse detector training, and joint fine-tuning schedules.

## 3. Currently Deployed Remote Experiments

Remote allocation:

- Job: `1118197`
- Node: `g0030`
- Environment: OpenTAD conda env under `/data/run01/sczc063/yuzibo/conda_envs/opentad`

### 3.1 GAS-VT Stage0/1 on GPU0

- RUN_TAG: `c3_gas_vt_stage01_gpu0_provok_20260706_151908_+0800`
- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/gas_vt_adatad/c3_gas_vt_stage01_gpu0_provok_20260706_151908_+0800`
- Driver log: `driver_resume.log`
- Variants: `gas_vt_fixed_384`, `gas_vt_fixed_768`, `gas_vt_dynamic`
- Last observed state: `gas_vt_fixed_384` training was active and had passed epoch 20.
- Intermediate detector mAP had reached about 43-44 average mAP at the latest observed evaluation.

### 3.2 PAction learned strict ledger on GPU1

- RUN_TAG: `c3_paction_learned_g30_gpu1_d413df8_20260706_214309_+0800`
- Root: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/paction_learned_adatad/c3_paction_learned_g30_gpu1_d413df8_20260706_214309_+0800`
- Driver log: `driver.log`
- Code snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_gasvt_d413df8_g30_20260706`
- Commit: `d413df83ad223e56039f7b1530c88d84f288190e`
- Variants: `learned_fixed_384`, `learned_fixed_768`, `learned_dynamic`
- Policy checkpoint SHA256: `dbebc39ec8f3c40b15d221d3a88b89871d6848df00dc60ab1bb68571270b0484`
- Last observed state: `learned_fixed_384` detector training had started.

### 3.3 Historical / no longer primary

- Old GPU0 model zoo was intentionally interrupted and must not be restarted automatically.
- Old PAction full run `c3_paction_learned_adatad_map_full_direct_20260706_101331_+0800` failed earlier and is not a current monitored target.

## 4. Experiments Still Needed

### 4.1 Official dense AdaTAD teacher

Need:

- Locate or train a clean official dense AdaTAD teacher using official-like config.
- Export train-only detector utility.
- Verify utility split provenance and no val/test leakage.

Expected config anchor:

- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`

Key evidence:

- Dense teacher mAP under official settings.
- Checkpoint hash.
- Utility export manifest with split, axis, fps, stride, window offset, dense length, proposal coordinate system, and source checkpoint.

### 4.2 Stage2 full matrix

Compare under matched budgets:

- Dense AdaTAD
- Uniform 384 / 768
- Random 384 / 768
- p_action-only 384 / 768 / dynamic
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

- Dense branch: full or near-full AdaTAD branch computes online responsibility / saliency.
- Sparse branch: ST hard selector chooses K or dynamic budget observations and sparse AdaTAD predicts on selected frames.
- Teacher signal: online dense branch with stop-gradient or EMA teacher stabilization.
- Selector losses: sparse detector loss, dense-to-sparse distillation, budget loss, CVaR max-hole loss, boundary bracket loss, action interior loss.
- Evaluation path: selector + sparse AdaTAD only; dense branch not used at inference.

This is the first route that can be honestly positioned as end-to-end if detector loss reaches selector parameters.

### 4.4 Stage4 curriculum / bilevel stabilization

Suggested schedule:

1. Dense AdaTAD warmup.
2. Train-only utility export or online EMA utility.
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
- Offline detector-aware utility may not translate to deploy-time observable selection.
- Smoke gradient proof may not equal real detector integration.

Claim locks:

- No detector mAP, no performance claim.
- No matched-budget baselines, no acquisition-method superiority claim.
- No selector gradient in real detector forward, no end-to-end claim.
- No train-only provenance manifest, no detector-aware utility claim.

## 6. Complete GPT / Pro Review Prompt

Copy the following prompt into GPT/Pro. Replace `<COMMIT_URL>` with the final GitHub commit URL after pushing.

```text
You are a strict senior CVPR/ICCV/NeurIPS reviewer and a code auditor for Temporal Action Detection systems.

Please review this GitHub repository and commit line by line:

Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
Branch: codex/gas-vt-stage23-detector-aware-20260706
Commit URL: <COMMIT_URL>

Core project objective:

We are developing DUCA-TAD / C3 sparse temporal acquisition for OpenTAD/AdaTAD. The goal is a deployable pre-backbone temporal frame/snippet selection module for Temporal Action Detection. It should reduce temporal observations before the expensive backbone while preserving or improving detector mAP, especially high-IoU localization. The selector should eventually become detector-aware and end-to-end trainable, so that AdaTAD detector loss or detector utility can teach the selector what observations matter.

Please check whether the current implementation matches this objective or has drifted into the wrong problem, such as merely learning actionness, replaying uniform sampling, leaking teacher/GT/cache information, or adding engineering gates without proving detector performance.

Important files and directories to inspect:

1. Stage1 GAS-VT / p_action strict ledger:
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

3. Stage2 detector-aware teacher utility and selector:
   - tools/bata/detector_teacher_utility.py
   - tools/bata/detector_aware_acquisition_policy.py
   - tools/bata/train_detector_aware_acquisition_policy.py
   - tools/bata/apply_detector_aware_acquisition_policy.py
   - tools/bata/convert_detector_aware_samples_to_value_transport_ledger.py
   - tools/bata/run_detector_aware_ledger_pipeline.py
   - tools/bata/validate_detector_aware_policy_ledger.py
   - configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py
   - scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh
   - tests/test_detector_teacher_utility.py
   - tests/test_detector_aware_acquisition_policy.py
   - tests/test_detector_aware_ledger_pipeline.py
   - tests/test_c3_detector_aware_adatad_full_train.py

4. Stage3 TrueTime / ST hard selector:
   - opentad/models/selectors/truetime_joint_selector.py
   - tools/bata/run_truetime_joint_selector_smoke.py
   - tools/bata/validate_truetime_joint_selector_precheck.py
   - configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py
   - tests/test_truetime_joint_selector.py
   - tests/test_truetime_joint_selector_config.py
   - tests/test_truetime_detector_selector_integration.py
   - tests/test_truetime_geometry.py

5. Stage4 curriculum/evidence gates:
   - tools/bata/validate_stage4_detector_aware_truetime_curriculum.py
   - tests/test_stage4_detector_aware_truetime_curriculum.py
   - docs/methods/2026-07-06-detector-aware-truetime-cvpr-route.md
   - docs/methods/2026-07-06-current-experiment-map-and-gpt-review-prompt.md

6. Detector and geometry integration:
   - opentad/models/detectors/single_stage.py
   - opentad/models/detectors/two_stage.py
   - any post-processing, NMS, decode, selected-axis-to-true-time conversion code

Review tasks:

1. Visibility and reproducibility:
   - Confirm the commit is visible.
   - Confirm which files were actually inspected.
   - Identify any missing scripts/configs/tests needed to reproduce claims.

2. Objective alignment:
   - Does the code implement a pre-backbone sparse temporal acquisition module for TAD, or has it drifted into an actionness classifier / ledger engineering exercise?
   - Does each implemented stage directly support the final objective?
   - Which parts are only diagnostics, gates, or smoke tests rather than real training methods?

3. Leakage and deployment correctness:
   - Check that val/test/deploy ledgers do not contain teacher outputs, GT labels, GT segments, proposal caches, dense prediction caches, or any forbidden supervision payload.
   - Check that teacher utility is train-only and that split provenance is explicit.
   - Check that fixed and dynamic ledgers do not use uniform fill or uniform scaffold.

4. Geometry and time-axis correctness:
   - Check selected-axis vs dense/true-time coordinate handling.
   - Verify whether selected-axis proposals are remapped to physical time before NMS and mAP evaluation.
   - Look for off-by-one, valid_len, selected_count, fps, feature_stride, window_offset, and duration mistakes.

5. Stage1 / GAS-VT:
   - Does GAS-VT genuinely improve over p_action-only or mostly encode hand-crafted gap/coverage priors?
   - Are boundary bracket, action interior, CVaR max-hole, and gap losses implemented in a way that helps detector mAP rather than only ledger metrics?
   - What ablations are mandatory?

6. Stage2 / detector-aware utility:
   - Is the teacher utility semantically meaningful for AdaTAD? Does it capture proposal responsibility, cls/reg loss, saliency, or counterfactual utility correctly?
   - Is signed utility handled correctly, or are harmful negative utilities collapsed into absolute high value?
   - Is dynamic budget calibrated using train-split marginal gain and frozen for val/test, or merely rank-derived?
   - What exact code changes are needed to make Stage2 scientifically strong?

7. Stage3 / end-to-end selector:
   - Does the current code prove that detector loss reaches selector parameters in a real AdaTAD forward/backward path?
   - If not, identify the exact missing integration points.
   - Provide key implementation code for an online dense-sparse co-training route where selector and AdaTAD are optimized in the same training loop.
   - The design should not require a fully pre-trained dense teacher, although it may use an EMA dense branch or warmup for stability.

8. Stage4 / curriculum and bilevel training:
   - Is Stage4 currently implemented as a real training curriculum or just an evidence gate?
   - Provide a detailed curriculum plan:
     a. dense warmup or EMA online teacher,
     b. train-only utility / online utility,
     c. selector pretraining,
     d. frozen selector sparse AdaTAD,
     e. joint ST fine-tuning,
     f. dynamic budget calibration,
     g. collapse diagnostics.

9. Experiments:
   - Design the complete experiment matrix needed for a CVPR-level claim.
   - Include dense AdaTAD, uniform, random, p_action-only, delta-p_action, GAS-VT, detector-aware offline, and end-to-end online selector variants.
   - Require matched budgets, matched compute, multiple seeds, high-IoU metrics, wall-clock, memory, and statistical uncertainty.
   - Specify which experiments must run first and which can wait.

10. Key code:
   - Provide concrete pseudocode or patch-level code for:
     a. online dense-sparse co-training forward pass,
     b. ST/Gumbel hard top-k selector,
     c. selected-axis-to-true-time remap,
     d. detector utility extraction,
     e. dynamic budget calibration,
     f. recursive no-leakage validator,
     g. selector gradient proof in a real detector path.

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
6. A concise paper story: problem, insight, method, novelty, and evidence required.
```

