---
updated: 2026-07-07
status: active
scope: Prompt for a Pro-tier reviewer to decide the final research target, acceptable claims, and concrete implementation route for <=384 sparse pre-backbone TAD.
out-of-scope: Treating this prompt as a review result, claiming current code is complete, or relying on unavailable Pro routing.
---

# Pro Discussion Prompt: Final Target And Route

请你作为 CVPR/ICCV 高级审稿人、TAD 方法研究者、以及严厉代码审查者，基于公开 GitHub 仓库和下面的实验上下文，帮助我们裁决这个研究到底应该做到什么程度、最终产出什么结果、以何种具体实现路线得到。

## 0. 可见性与审查对象

请先做 visibility check：

- GitHub repo: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- 固定公开 commit: `46cacc113042fcf0931c70774491d44665246e32`

如果你只能审查 GitHub 可见代码，请明确说“仅基于公开 commit”。如果你不能访问 GitHub，请停止并说明不可见，不要假装审查。

另有本地 pending patches，尚未必然同步到 GitHub，供你在路线建议中考虑：

1. lattice replacement summary 已改为诚实标记：
   `uses_uniform_scaffold=True`,
   `scaffold_type="uniform_lattice_local_replacement"`，
   并记录 protected/replaceable/replaced counts、base uniform Jaccard。
2. lattice formal deploy 默认不再允许 inferred p_action provenance；
   inferred provenance 只允许显式 opt-in 的 diagnostic/migration route。
3. 新增 `<=384` sparse TAD claim-budget gate：
   paper-main 必须 fail-closed 拒绝 fixed_768、dynamic >384、缺 selected-count evidence、diagnostic route 混入 paper-main。
4. Stage2 teacher utility 目前只加强了 utility source manifest；
   本质仍是 proposal-score surrogate，不是严格 point-responsibility/counterfactual utility。
5. Stage3 precheck 加强了 selector grad/param delta/selected-position drift proof；
   但仍不是 THUMOS fulltrain mAP 结果。

## 1. 研究背景

任务是 Temporal Action Detection，基于 OpenTAD/AdaTAD。我们希望在 backbone 前插入一个 sparse temporal acquisition / pre-backbone selector，在每个 768 dense window 内最多选择 384 个 temporal positions，然后把选中的 sparse observations 喂给 AdaTAD detector。

核心目标不是做一个独立的粗分类器，而是证明：

> 在 `K <= 384` 的预算下，一个低成本、可部署、无泄漏的 selector 能选择对 TAD 定位真正有用的 temporal observations，并在 matched AdaTAD 设置下超过 uniform_384 baseline，尤其保护 high-IoU mAP。

当前重要约束：

- 主 claim 必须是 `<=384`，不能用 640/768 或 dynamic >384 作为论文主结果。
- val/test deploy selection 不能使用 GT、teacher prediction cache、raw detector prediction、oracle boundary。
- dense teacher utility 只能 train-only。
- 所有结果必须能被 strict ledger、manifest、checkpoint sha、config diff、Slurm/precheck evidence 追溯。

## 2. 当前实验现象

已观察到的结果，供路线分析，不视作你独立复现：

- PAction learned fixed_384:
  Average-mAP `59.10`;
  tIoU 0.30/0.40/0.50/0.60/0.70 =
  `74.32/68.50/61.71/51.47/39.51`。
- GAS-VT fixed_384:
  Average-mAP `44.90`;
  tIoU =
  `60.09/53.83/46.39/37.28/26.92`。
- Dense AdaTAD teacher:
  epoch 29 Average-mAP `64.39`;
  tIoU =
  `81.78/76.38/67.08/55.58/41.13`;
  dense teacher full run still intended to epoch 59.
- uniform_384 anchor was discussed as approximately `65 mAP`, but this must be rechecked under matched same-commit/same-config conditions.

Important diagnosis so far:

- PAction learned is currently the strongest sparse route, but still below the desired uniform_384 anchor.
- GAS-VT reached around 40 mAP early and plateaued; likely learned action/interior coverage faster than high-IoU boundary/localization utility.
- Lattice replacement can diagnose whether uniform geometry plus local p_action replacement helps, but it is not a satisfying intelligent acquisition method.
- Stage2 detector-aware route currently exports proposal-score-derived utility, not true detector responsibility.
- Stage3 has gradient-path proof but no real THUMOS fulltrain mAP evidence.

## 3. What We Do Not Accept As Final Method

Please explicitly evaluate these non-acceptance decisions. If you disagree, give technical reasons.

1. We do not accept lattice replacement as the final CVPR method.
   Reason: it starts from uniform scaffold and performs local replacement. It may be useful as a diagnostic or performance bridge, but packaging it as “intelligent acquisition” would be weak and potentially misleading.

2. We do not accept 768/dynamic >384 as the paper-main result.
   Reason: the target is `<=384` sparse pre-backbone acquisition; 768/dynamic can be diagnostic ceilings only.

3. We do not accept Stage2 proposal-score utility as true detector-aware utility.
   Reason: proposal score is not point responsibility, cls/reg loss sensitivity, high-IoU localization responsibility, or counterfactual utility.

4. We do not accept Stage3 gradient smoke/precheck as end-to-end training.
   Reason: end-to-end claim requires real THUMOS training, nonzero selector losses/regularization, detector-loss gradient into selector, selected-position movement, mAP/tIoU, and anti-collapse evidence.

5. We do not accept “more engineering rules” as the central novelty.
   Reason: hand-designed slot/lattice/repair rules may improve robustness, but the desired contribution is a learned detector-utility-calibrated acquisition policy.

## 4. Questions For You

### A. Final research target

What should the final paper output be?

- A plug-in pre-backbone selector module for AdaTAD/OpenTAD?
- A full end-to-end sparse TAD model?
- A staged method: train-only detector utility distillation + optional joint fine-tuning?
- Something else?

What is the minimum result needed for a credible CVPR paper?

- Is beating matched uniform_384 by +0.5 to +1.0 mAP enough?
- Must it beat dense teacher or dense AdaTAD? Or only sparse uniform?
- How important is high-IoU mAP@0.6/0.7 compared with Average-mAP?
- What compute/latency/FLOPs evidence is required?

### B. Diagnose current route

Please analyze whether the current route has deviated from the original goal.

- Did GAS-VT optimize the wrong target, i.e. action/interior coverage instead of detector localization utility?
- Why can PAction learned be better than GAS-VT despite being simpler?
- Why did GAS-VT reach around 40 mAP early but fail to improve later?
- Is the current pipeline over-emphasizing no-leak engineering while under-solving utility quality?
- Are there code-level mistakes that could explain the gap, especially selected-axis remapping, target_len/window_size mismatch, eval cadence, short-valid-count exceptions, or variant mixing?

### C. What should be implemented next?

Please propose a concrete implementation route that avoids hand-engineered scaffold dependence while still controls gaps and boundary coverage.

I want a route that can honestly be described as:

> A detector-utility-calibrated sparse temporal acquisition module for TAD.

Please give:

1. Model architecture.
2. Utility target definition.
3. Loss terms.
4. Training schedule.
5. Deployment ledger constraints.
6. Exact files/modules to change in this repo.
7. Core pseudocode or code sketches.
8. Precheck gates.
9. Full run matrix.
10. Claim locks and pass/fail criteria.

### D. Stage2 responsibility utility

How should we implement true AdaTAD teacher utility?

Please go beyond proposal score. Consider:

- point responsibility;
- FPN level/point assignment;
- cls/reg loss contribution;
- start/end boundary utility;
- high-IoU proposal responsibility;
- false-positive risk;
- saliency/gradient sensitivity;
- counterfactual drop/add utility;
- train-only export without val/test teacher leakage.

What is the minimum useful version that can be implemented quickly and has a real chance to beat PAction learned fixed_384 and uniform_384?

### E. Stage3 joint training

How should selector and AdaTAD be trained in the same graph?

Please specify:

- hard/ST top-k or differentiable subset selection;
- true-time coordinate handling vs selected-axis remapping;
- how detector loss reaches selector;
- how to avoid collapse into action interiors;
- how to control max holes without manual scaffold;
- curriculum from dense teacher to sparse joint;
- how to log selector movement, gap distribution, boundary bracket, and mAP.

### F. Experiment plan

Design a complete but prioritized experiment plan.

Must include:

- uniform_384 baseline;
- raw p_action top-k 384;
- PAction learned fixed_384;
- GAS-VT fixed_384;
- lattice diagnostic 384;
- Stage2 proposal-score utility 384;
- Stage2 responsibility utility 384;
- Stage3 joint 384;
- dense AdaTAD teacher as reference;
- optional 768/dynamic only as diagnostic ceilings.

For each experiment, define:

- hypothesis;
- exact budget;
- expected gain/failure mode;
- required ledger metrics;
- required detector metrics;
- whether it can support a paper claim.

### G. Strict code review

Please inspect the public GitHub code line-by-line where possible and produce:

- P0 bugs that invalidate experiments;
- P1 issues that confound attribution;
- P2 quality issues;
- files/functions to patch;
- tests that must be added;
- launch/precheck commands;
- what evidence must be saved before claiming success.

Pay special attention to:

- `tools/bata/run_paction_lattice_replacement_ledger_pipeline.py`
- `tools/bata/validate_sparse_tad_claim_budget.py`
- `tools/bata/export_dense_adatad_teacher_points.py`
- `tools/bata/detector_teacher_utility.py`
- `tools/bata/train_detector_aware_acquisition_policy.py`
- `tools/bata/run_truetime_joint_selector_precheck.py`
- `tools/bata/validate_truetime_joint_selector_precheck.py`
- `tools/bata/validate_duca_stage23_precheck.py`
- `opentad/models/selectors/truetime_joint_selector.py`
- `opentad/models/detectors/actionformer.py`
- `configs/adatad/thumos/c3_*`
- `scripts/run_*stage*`, `scripts/run_c3_*`
- relevant tests.

## 5. Expected Output Format

Please answer in this structure:

1. Visibility check.
2. One-sentence final verdict.
3. What you agree with / disagree with in our non-acceptance decisions.
4. What the final research contribution should be.
5. Minimum result threshold for a credible paper.
6. Current implementation gap percentage:
   - engineering scaffold;
   - detector-aware utility;
   - end-to-end training;
   - experimental evidence;
   - paper claim readiness.
7. P0/P1/P2 code issues with file/function references.
8. Concrete implementation plan with key code sketches.
9. Exact experiment matrix and expected pass/fail interpretation.
10. The shortest route to a `<=384` result that can beat matched uniform_384.
11. If the route is unlikely to work, say so bluntly and propose a pivot.

Be strict. Do not flatter the current route. Do not accept claims that are not supported by code and experiments. The goal is to decide whether this project can become a CVPR-grade method, what exact method it should be, and what code/experiments are needed next.

