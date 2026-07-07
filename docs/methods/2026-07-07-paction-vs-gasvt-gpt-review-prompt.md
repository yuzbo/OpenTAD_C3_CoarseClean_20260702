# GPT Review Prompt: PAction Learned vs GAS-VT Stage0/1

Updated: 2026-07-07 +0800

Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702

Review branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/gas-vt-stage23-detector-aware-20260706

Primary task: strict code review, experiment-logic audit, and research-route diagnosis.

## Prompt

You are a severe CCF-A / CVPR-level code and experiment reviewer. Please review the public GitHub repository and the branch above. First perform a visibility check: report the exact commit hash you can access, and do not review an inaccessible or stale commit.

Context:

- The project studies sparse temporal acquisition for Temporal Action Detection (TAD), using OpenTAD/AdaTAD.
- The final intended method is not a standalone action classifier. The intended final direction is a deployable pre-backbone selector/acquisition module that reduces input observations while preserving or improving AdaTAD detector mAP, especially high-IoU mAP.
- Current Stage1 baselines include:
  - `PAction learned` strict ledgers: `learned_fixed_384`, `learned_fixed_768`, `learned_dynamic`.
  - `GAS-VT` strict ledgers: `gas_vt_fixed_384`, `gas_vt_fixed_768`, `gas_vt_dynamic`.
- Current later-stage direction includes:
  - Stage2 dense AdaTAD teacher utility -> detector-aware offline selector.
  - Stage3 ST/hard selector + AdaTAD joint training / TrueTime candidate.
  - Stage4 curriculum or bilevel stabilization.

Observed monitoring notes to verify, not to assume as final claims:

- `GAS-VT fixed_384` reached roughly 40 Average-mAP very early, around the first few epochs, but later plateaued around the mid-40s and did not show effective progress.
- One GAS-VT run stopped after `gas_vt_fixed_384` because strict validation reported `uniform similarity above threshold: 1.0`. Latest observed/final GAS-VT fixed_384 eval was approximately Average-mAP 44.90, with tIoU 0.30/0.40/0.50/0.60/0.70 around 60.09/53.83/46.39/37.28/26.92.
- `PAction learned_fixed_384` appears to train better in detector mAP, reaching substantially higher monitored values than GAS-VT fixed_384. Treat this as a hypothesis until you inspect logs or request missing artifacts.
- PAction fixed_384 has shown strong boundary coverage and small p95 unselected hole; dynamic variants may have larger holes.

Please review these code surfaces line by line:

- `tools/bata/paction_acquisition_policy.py`
- `tools/bata/train_paction_acquisition_policy.py`
- `tools/bata/apply_paction_acquisition_policy.py`
- `tools/bata/run_paction_learned_policy_ledger_pipeline.py`
- `tools/bata/validate_paction_learned_policy_ledger.py`
- `tools/bata/gas_vt_paction_policy.py`
- `tools/bata/train_gap_aware_acquisition_policy.py`
- `tools/bata/apply_gap_aware_acquisition_policy.py`
- `tools/bata/run_gap_aware_ledger_pipeline.py`
- `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py`
- `configs/adatad/thumos/c3_gas_vt_ledger_adatad_full_train.py`
- `scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh`
- `scripts/run_c3_gas_vt_policy_adatad_full_train_gpu1.sh`
- Stage2/3 related files:
  - `configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py`
  - `configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py`
  - `configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py`
  - `tools/bata/export_dense_adatad_teacher_utility.py`
  - `tools/bata/train_detector_aware_acquisition_policy.py`
  - `tools/bata/detector_aware_acquisition_policy.py`
  - `opentad/models/selectors/truetime_joint_selector.py`

Core questions:

1. Why might `PAction learned_fixed_384` outperform `GAS-VT fixed_384` even though GAS-VT was designed as the more structured value-transport route?
   - Is PAction genuinely learning a better acquisition policy?
   - Or is it benefiting from simpler direct p_action ranking, larger effective density, weaker constraints, easier optimization, or hidden evaluation/config differences?
   - Does fixed_384 naturally guarantee small p95 holes because it selects about half of a 768-length window?
   - Does PAction's boundary coverage come from deploy-visible p_action/delta signals, from train-time boundary supervision, or from budget density?

2. Why might GAS-VT reach around 40 mAP within the first few epochs and then plateau around 44-46?
   - Is this normal AdaTAD optimization behavior under sparse inputs?
   - Is the detector quickly learning coarse classification while high-IoU localization is bottlenecked by sparse geometry?
   - Is the GAS-VT ledger too similar to uniform sampling due to hard gap repair or max-hole constraints?
   - Could `uniform similarity above threshold: 1.0` indicate policy collapse to uniform-like coverage?
   - Are boundary bracket, action interior, CVaR max-hole, and gap urgency improving acquisition, or are they over-regularizing and suppressing high-value p_action points?

3. Is the current GAS-VT implementation faithful to the original research idea?
   - Original idea: gap-aware sequential state, CVaR max-hole, boundary bracket, action interior, then AdaTAD mAP validation.
   - Current concern: the implementation may be `p_action + handcrafted gap/coverage priors + hard constrained decoder`, not a detector-aware acquisition model.
   - Please judge whether this is acceptable as a Stage1 baseline, and whether it should be demoted from "main method" to "strong engineered baseline".

4. Is the current experiment logic biased or flawed?
   - Are PAction and GAS-VT compared under identical AdaTAD configs, pretraining paths, eval schedules, window sizes, post-processing, seeds, and selected-count handling?
   - Are validation schedules now correctly aligned to epoch 9, 19, 29, 39, 49, 59 under zero-based OpenTAD epoch semantics?
   - Are fixed_384 / fixed_768 / dynamic variants using comparable selected-axis remapping and true-time metadata?
   - Is short-valid-ratio count handled identically?
   - Are the old GAS-VT and current PAction logs enough for any claim, or do we need matched-budget reruns?

5. Check for code correctness bugs.
   - Train/val/test leakage: does val/test ledger generation ever use GT, teacher utility, dense teacher output, raw prediction cache, or training-only fields?
   - Are GT-derived `action_target`, `gt_boundaries`, and action interior bins used only for policy training on train split?
   - Are deploy ledgers stripped of invisible payloads?
   - Is `selected_positions` always on the original dense local index axis, not selected-rank axis?
   - Does AdaTAD remap GT and predictions correctly between selected axis and true dense/physical time?
   - Are duplicate sample IDs, canonical unique source rows, and metric source rows handled correctly?
   - Does the validator's `uniform_similarity` metric correctly measure uniform-like collapse?
   - Is `max_unselected_hole` / `p95_unselected_hole` correctly defined and enforced?
   - Are `uses_uniform_fill=False` and `uses_uniform_scaffold=False` meaningful, or can hard repair mimic uniform fill while bypassing these flags?

6. Check whether the current path has drifted from the original Stage0-4 goal.
   - Stage1 should be an offline strict-ledger baseline, not the final paper method.
   - Stage2 should answer: can dense AdaTAD teacher utility train a better acquisition policy than p_action-only / GAS-VT?
   - Stage3 should answer: can detector loss truly backpropagate into selector parameters in one training graph?
   - Stage4 should stabilize joint training and prevent selector collapse / high-IoU degradation.
   - Does the current code and experiment plan still follow this logic, or has it become a collection of ad hoc baselines?

7. Provide a severe verdict.
   - PASS / WARN / HOLD / FAIL for code correctness.
   - PASS / WARN / HOLD / FAIL for experimental logic.
   - PASS / WARN / HOLD / FAIL for paper-claim readiness.
   - List P0/P1/P2 issues with exact file and line references.

Required output:

1. Visibility check: exact commit hash and branch reviewed.
2. One-page executive verdict.
3. A table comparing PAction learned vs GAS-VT:
   - features
   - training losses
   - decoder behavior
   - validation gates
   - likely reason for performance gap
4. A root-cause analysis for:
   - why PAction learned may be better
   - why GAS-VT may plateau after early progress
   - why uniform similarity can become 1.0
5. A strict bug list with exact file:line references.
6. A fairness checklist for rerunning the matched matrix.
7. A corrected experiment plan:
   - immediate reruns or ablations
   - Stage2 dense teacher utility experiments
   - Stage3 true joint selector experiments
   - Stage4 curriculum/bilevel stabilization
8. Concrete code-level recommendations or patches, including key implementation snippets if needed.

Be skeptical. Do not accept metric claims without logs and result files. If GitHub does not contain the running logs, explicitly separate "code-review conclusions" from "result claims requiring remote artifacts".
