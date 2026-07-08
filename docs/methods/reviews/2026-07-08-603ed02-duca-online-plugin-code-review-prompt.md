# GPT Review Prompt: DUCA Online Plugin at commit 603ed02

Please perform a strict, line-by-line code review and research-method review of the current DUCA online temporal acquisition implementation.

Repository branch:

- `codex/gas-vt-stage23-detector-aware-20260706`

Fixed commit to review:

- `603ed0203ef2ebf631523ca42d9061b5ee877de9`

Context:

This commit is an implementation milestone for the final DUCA direction. The intended final paper method is not C3, GAS-VT, PAction top-k, lattice relocation, or an offline ledger pipeline. The intended final method is an online temporal acquisition plugin placed before an existing TAD detector. At inference time it must generate hard `selected_positions <= 384` online, in original dense-time coordinates, and the detector must actually consume only those selected temporal observations in a single forward pass.

The current implementation adds a registry-buildable OpenTAD `frame_selector` path and zero-shot actionness evaluation tools. The purpose is to move from offline ledger experiments toward a deployable online acquisition adapter that can eventually connect to AdaTAD and ActionFormer.

## Files changed in this round

Core DUCA online selector / detector path:

- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/selectors/__init__.py`
- `opentad/models/dense_heads/duca_online_precheck_head.py`
- `opentad/models/dense_heads/__init__.py`
- `opentad/models/duca/acquisition.py`

OpenTAD configs / launchers / validators:

- `configs/adatad/thumos/duca_online_adatad_precheck.py`
- `configs/adatad/thumos/duca_online_zeroshot_actionness_precheck.py`
- `scripts/run_duca_online_adatad_precheck_gpu1.sh`
- `scripts/run_duca_online_zeroshot_actionness_precheck_gpu1.sh`
- `tools/bata/validate_duca_online_adatad_precheck.py`
- `tools/bata/run_duca_online_adatad_wrapper_precheck.py`

Zero-shot actionness and selection evaluation:

- `tools/bata/eval_zero_shot_actionness.py`
- `tools/bata/validate_zero_shot_actionness_eval.py`
- `tools/bata/run_zero_shot_actionness_selection_eval.py`
- `tools/bata/validate_zero_shot_selection_eval.py`

Tests:

- `tests/test_duca_online_frame_selector_integration.py`
- `tests/test_duca_online_precheck_config.py`
- `tests/test_zero_shot_actionness_eval.py`
- `tests/test_zero_shot_actionness_selection_eval.py`

## What this commit claims to implement

1. `DucaOnlineFrameSelector` is registered as an OpenTAD selector and buildable through `build_detector(cfg.model)`.
2. It plugs into `SingleStageDetector.frame_selector`, using standard OpenTAD forward signatures:
   - train: `inputs, masks, metas, gt_segments, gt_labels, return_loss=True`
   - test: `inputs, masks, metas, return_loss=False`
3. It hard-selects detector-consumed temporal observations with budget `<=384`.
4. It writes original-time `selected_positions` and selected-axis remap metadata into `metas`.
5. The detector receives selected inputs and selected masks, not dense observations.
6. GT labels remain available to the detector during training, but teacher/oracle/cache/ledger payloads must not enter selector decision or inference.
7. Detector loss should have a straight-through gradient path back to selector parameters through selected sparse observations.
8. Offline ledger is only an audit/reproducibility artifact, not the final method body.
9. Zero-shot/no-target-label actionness is now implemented as an evaluable source family, with motion/feature-energy fallback and manual JSONL/precomputed support.

## Verification already run

Remote verification environment:

- `/data/run01/sczc063/yuzibo/projects/opentad_duca_online_impl_verify_20260708_1630`
- conda env: `/data/run01/sczc063/yuzibo/conda_envs/opentad`

Remote focused tests:

```bash
python -m pytest \
  tests/test_duca_online_acquisition.py \
  tests/test_duca_online_plugin_smoke.py \
  tests/test_duca_online_adatad_wrapper_precheck.py \
  tests/test_duca_online_frame_selector_integration.py \
  tests/test_duca_online_precheck_config.py \
  tests/test_zero_shot_actionness_eval.py \
  tests/test_zero_shot_actionness_selection_eval.py -q
```

Result:

- `33 passed in 153.48s`

Runtime DUCA online precheck launchers:

```bash
PRECHECK_ONLY=1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_duca_online_adatad_precheck_gpu1.sh
PRECHECK_ONLY=1 CUDA_VISIBLE_DEVICES=1 bash scripts/run_duca_online_zeroshot_actionness_precheck_gpu1.sh
```

Both passed. Runtime validator reported:

- `build_detector=true`
- `standard_forward_train=true`
- `standard_forward_test=true`
- `gt_reaches_detector_train=true`
- `teacher_free_inference=true`
- `uses_ledger_for_decision=false`
- `selected_positions_original_time=true`
- `masks_selected_count=true`
- `remap_metadata_present=true`
- `real_detector_loss_selector_grad_nonzero=true`
- `runtime_selected_count=384`

## Review tasks

Please be severe. Do not rubber-stamp the implementation because tests pass. Treat this as a CVPR reviewer plus senior code auditor review.

### A. Line-by-line code correctness review

Read the files listed above and identify P0/P1/P2 issues. For each issue provide:

- file and line/function;
- why it is wrong or risky;
- what failure it can cause;
- whether it affects paper claims, mAP validity, no-leak validity, online-plugin validity, or only engineering cleanliness;
- exact fix recommendation.

Pay special attention to:

- whether `selected_positions` are truly original-time positions;
- whether detector truly consumes only selected temporal observations;
- whether selected masks prevent padded slots from being consumed;
- whether detector loss really backpropagates into selector parameters;
- whether `selected_mask_st` is aligned with actual consumed selected positions;
- whether train and test both use hard forward selection;
- whether train-only teacher utility is isolated from inference and detector decision;
- whether GT is used only as detector training label and not selector decision input;
- whether `SparseTemporalGrid` and meta fields are internally consistent;
- whether selected-axis prediction remap is sufficient for high-IoU TAD evaluation;
- whether the validator or tests can be fooled by fake metadata;
- whether `DucaOnlinePrecheckHead` is only a smoke head and not overclaimed;
- whether configs are genuinely OpenTAD-buildable and not a fake pipeline.

### B. Zero-shot actionness review

Review the zero-shot actionness tools and selection evaluation tools. Decide whether they are sufficient to support the claim that no-target-label/frozen actionness can be a useful acquisition prior.

Check:

- whether GT labels are only used in evaluation, not score generation;
- whether manual/precomputed source provenance is fail-closed;
- whether unknown source provenance can accidentally be reported as no-THUMOS;
- whether AUROC/AUPRC/Recall@K are computed correctly;
- whether threshold tuning on test split is avoided;
- whether selection metrics separate coarse actionness quality, selection geometry quality, and detector mAP;
- whether oracle baselines are clearly diagnostic-only;
- whether large foundation-model sources would be fairly costed and cannot be hidden inside the efficiency claim.

### C. Research-method judgment

Judge whether this implementation is currently:

1. a useful engineering precheck,
2. a credible experimental prototype,
3. or already a CVPR-level method implementation.

Be explicit. Do not be generous. If the current result is not yet CVPR-level, explain exactly what is missing.

Evaluate the current experiments against these possible claims:

- "online temporal acquisition plugin";
- "pre-backbone sparse temporal observation selection";
- "strict <=384 detector-consumed observations";
- "no-leak teacher-free inference";
- "detector-loss-aware acquisition";
- "zero-shot actionness can replace THUMOS-trained coarse classifier";
- "DUCA improves high-IoU TAD localization";
- "works beyond AdaTAD, e.g. ActionFormer";
- "not merely actionness top-k or uniform sampling with repair";
- "compute-efficient and fair against dense detector baselines".

For each claim, mark:

- supported now;
- partially supported;
- unsupported;
- invalid unless additional experiments are run.

### D. More intelligent / more elegant selection design

The current code is still a hard selected-position plugin with center/radius and zero-shot/actionness sources. Please propose more elegant, less heuristic alternatives that could become the final paper method.

Consider:

- differentiable budgeted sparse acquisition;
- detector-loss-aware hard-forward straight-through training;
- learned adaptive context radius `0..16`;
- boundary-aware utility without oracle boundary leakage;
- uncertainty/actionness/responsibility fusion;
- dynamic K with fair matched-budget reporting;
- ActionFormer-compatible sparse temporal grid;
- raw-frame vs AdaTAD 1/4 downsampled temporal grid issue;
- end-to-end or alternating training that avoids three disconnected stages;
- how to avoid collapsing into uniform, actionness top-k, or boundary prior.

For each proposed method:

- give the model definition;
- explain how it preserves strict budget;
- explain the train/test forward path;
- explain what loss gives detector feedback to the selector;
- explain how it avoids leakage;
- state what code modules would need to change.

### E. CVPR paper potential

Give a direct verdict:

- Does this project have CVPR potential?
- What would be the strongest paper title/claim if it succeeds?
- What result table and figures are mandatory before submitting?
- What ablations are non-negotiable?
- Which current experiments are only engineering scaffolding and should not be sold as the main method?
- What would reviewers attack first?
- What is the shortest path from current commit `603ed02` to a defensible CVPR-level submission?

Mandatory result matrix to discuss:

- Dense AdaTAD;
- Dense ActionFormer;
- Uniform fixed 384;
- Random fixed 384, multi-seed;
- p_action top-K;
- zero-shot X-CLIP / ActionCLIP actionness top-K;
- Kinetics SlowFast / VideoMAE actionness top-K;
- C3 / GAS-VT / lattice baselines;
- DUCA-Frozen;
- DUCA-Adapted without teacher warm-up;
- DUCA-Adapted with teacher warm-up;
- DUCA online plugin without joint fine-tune;
- DUCA online plugin with hard-forward ST fine-tune;
- center-only / center-radius / no radius / fixed radius / learned radius;
- no boundary loss / no hole loss / no teacher utility / no actionness input;
- selected-rank decode vs original-time decode;
- AdaTAD vs ActionFormer;
- dynamic K vs matched-average fixed K.

Mandatory metrics to discuss:

- Average mAP;
- mAP@0.5/@0.6/@0.7;
- high-IoU drop/gain;
- selected count;
- budget violation rate;
- selector/gather/backbone/detector/NMS/total latency;
- training cost;
- teacher warm-up cost;
- zero-shot source cost;
- boundary support;
- short-action recall;
- action-local max hole / p95 hole;
- uniform similarity;
- actionness-interior over-selection;
- selector entropy / collapse diagnostics;
- no-leak audit pass/fail.

Please finish with:

1. a P0 fix list;
2. a P1 fix list;
3. a recommended final method design;
4. a minimal code-change plan;
5. a paper-level experiment plan;
6. a verdict on whether current commit `603ed02` is CVPR-ready, CVPR-promising, or only an engineering scaffold.
