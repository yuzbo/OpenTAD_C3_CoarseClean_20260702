# GPT/Pro Prompt: Strict Review of Stage2 Learned Context Radius Selector

请你以严厉的 CCF-A/CVPR 审稿人和资深 PyTorch/OpenTAD 工程审查者身份，基于公开 GitHub 仓库代码进行逐行审查和研究路线判断。

## 可见性与代码来源

- GitHub 仓库: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 分支: `codex/gas-vt-stage23-detector-aware-20260706`
- 请以该分支最新 HEAD 为准，不要审查旧 commit。
- 如果无法访问分支或文件，请先明确报告 visibility failure，不要基于猜测给 PASS。
- 代码重点是 OpenTAD/AdaTAD 上的 C3 sparse temporal acquisition / pre-detector selection 路线。

## 当前研究任务与真实定位

我们要解决的是 Temporal Action Detection (TAD) 中的视频时序观测选择问题：在 AdaTAD/OpenTAD 使用的 768 个 temporal observation/grid positions 内，只选择不超过 384 个 observation 给后续 detector 使用，目标是在降低输入观测数量的同时尽量保持或提升 detection mAP，尤其保护高 tIoU 定位质量。

注意：当前实现不是严格 raw-video frame-level pre-backbone，也不是严格 end-to-end。它更准确地说是:

- 在 AdaTAD/OpenTAD 的 temporal grid 上做 sparse temporal observation acquisition。
- 当前 Stage2 使用 dense AdaTAD teacher/exported responsibility utility 训练一个 detector-aware selector。
- 当前 PAction learned 使用独立训练的 coarse actionness/p_action 信号。
- 当前 Stage3 尝试 selector + AdaTAD joint training，但仍需要严审其训练图是否真正让 detector loss 回传到 selector。

请特别审查我们是否在论文故事中错误地把它说成 raw-frame pre-backbone，或者错误地 claim 为端到端。

## 已知实验锚点

请用这些锚点判断方法是否有价值和是否跑偏:

- Dense AdaTAD upper anchor: Average-mAP about `68.29`.
- PAction learned fixed384: Average-mAP `59.10`, tIoU 0.30/0.40/0.50/0.60/0.70 = `74.32/68.50/61.71/51.47/39.51`.
- GAS-VT fixed384: Average-mAP `44.90`, tIoU = `60.09/53.83/46.39/37.28/26.92`.
- PAction lattice replacement move50 epoch9 diagnostic: Average-mAP `50.75`, still below PAction learned fixed384.
- Earlier Stage2 surrogate run failed before AdaTAD due `p95_unselected_hole above threshold: 96.0`.
- Earlier Stage2 responsibility run also failed before AdaTAD because fixed384 was effectively shrunk by short-valid-ratio budget, causing very small selected counts such as around 34 instead of filling min(384, valid_len).

## 本轮新增/修改的核心代码

请逐行审查以下文件，并检查每处实现是否真的符合设计:

1. `tools/bata/detector_aware_acquisition_policy.py`
   - 新增 exact fixed budget: `fixed_deploy_budget(requested_budget, valid_len) = min(requested_budget, valid_len)`.
   - 新增 learned-score local dilation:
     - fallback radii: `DEFAULT_DETECTOR_AWARE_SCORE_DILATION_RADII = (2, 4)`.
     - learned continuous context radius range: `[2, 16]`.
     - `score_dilated_frame_values(..., context_radii=...)` 应只围绕 learned positive detector utility peaks 扩展，不应变成 uniform scaffold。
   - 新增 loss terms:
     - `learned_spacing`
     - `action_local_hole`
     - `context_radius_cost`
   - `detector_aware_training_objective` 中是否正确用 soft context coverage 计算 spacing/action-local hole。
   - `DetectorAwareSequentialAcquisitionPolicy` 是否正确增加 `context_radius_head`，输出每个 temporal observation 的 `context_radius`。
   - 严查: 这个 radius head 现在是否真的可学习、是否会被 loss 有效驱动、是否存在 collapse 到 16 或 collapse 到 2 的风险。

2. `tools/bata/apply_detector_aware_acquisition_policy.py`
   - `checkpoint_policy_scores(..., return_context_radius=True)` 是否正确兼容新旧 checkpoint。
   - `_checkpoint_budget_conditioned_scores` 是否给 fixed384/fixed768/dynamic 分别使用 target-budget-conditioned scores 和 context radii。
   - 严查: monkeypatch fallback 到 max radius 是否可能掩盖真实 checkpoint 缺失半径头的问题。

3. `tools/bata/train_detector_aware_acquisition_policy.py`
   - `_extract_action_target` 是否合理。当前如果没有 `action_target`，会从 positive gain >= 0.5 推断 action region；这是否合理，是否会把 detector utility 和 action label 混淆？
   - `_batch_to_tensors` 和 `_run_epoch` 是否把 `action_target` 正确传入 objective。
   - 严查: 当前 training loss 是否真的优化 boundary/context，还是只是另一个 p_action/utility fitting。

4. `tools/bata/run_detector_aware_ledger_pipeline.py`
   - default `allow_short_valid_ratio_count=False` 是否正确。
   - CLI `--allow-short-valid-ratio-count` / `--disable-short-valid-ratio-count` 是否语义清晰。
   - conversion/validation 是否再也不会把 fixed384 缩成很小的短视频比例预算。

5. `configs/adatad/thumos/c3_detector_aware_ledger_adatad_full_train.py`
   - `bata_value_transport_allow_short_valid_ratio_count=False` 是否符合 Stage2 exact budget contract。
   - AdaTAD 设置是否仍与官方 AdaTAD 训练设置保持一致。

6. `tools/bata/validate_c3_detector_aware_adatad_full_train.py`
   - validator 是否正确要求 short-ratio gate 关闭。

7. `scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh`
   - launcher 是否不再把 `--allow-short-valid-ratio-count` 传给 ledger validator。
   - full/precheck 路径是否仍 fail-closed。

8. 测试文件:
   - `tests/test_detector_aware_acquisition_policy.py`
   - `tests/test_detector_aware_ledger_pipeline.py`
   - `tests/test_c3_detector_aware_adatad_full_train.py`
   请判断测试是否真正覆盖了原始 bug、r=2/4 fallback、learned radius up to 16、exact budget 和 Stage2 precheck gate。

## 请回答的核心问题

### A. 代码正确性

1. 当前实现是否真正修复了 Stage2 只选几十帧的问题？
2. exact fixed budget 是否会导致训练/验证/测试中 selected_count 与 AdaTAD target_len 不一致，尤其是 valid_len < 384 的视频？
3. `score_dilated_frame_values` 是否可能因为 tie-break 或负/零分数导致选择非预期位置？
4. learned context radius `[2,16]` 是否真的参与训练梯度？是否存在无效 head 或梯度很弱的问题？
5. `context_radius_cost` 的权重和形式是否足以避免全局扩大到 16？
6. `action_local_hole_loss` 是否可能用错 action target 或引入 label leakage？
7. 新旧 checkpoint 兼容路径是否会隐藏错误？是否应该严格要求新 checkpoint 带 `context_radius_head`？
8. 当前 tests 有没有测试不到的重要 failure mode？

### B. 研究路线是否跑偏

我们现在有多个阶段:

- PAction learned fixed384: 独立 coarse actionness selector -> AdaTAD sparse ledger。
- GAS-VT fixed384: gap-aware/state/boundary-inspired selector -> AdaTAD，效果差。
- Stage2 detector-aware utility selector: dense AdaTAD teacher/responsibility utility -> selector -> AdaTAD。
- Stage3 selector + AdaTAD joint training: 尝试 end-to-end，但尚需验证 detector loss 是否真的回传 selector。

请判断:

1. 当前 Stage2 是否仍然太像 teacher distillation/offline reranking，而不是智能 acquisition？
2. 三阶段独立训练 (coarse PAction, dense teacher, detector-aware selector) 是否过于工程化、不优雅？
3. 如果论文目标是 CVPR 级，是否应该把最终方法定位为:
   - detector-aware offline selector,
   - pluggable pre-detector sparse temporal acquisition module,
   - 或真正 end-to-end pre-backbone/frame acquisition model？
4. 当前最有希望超过 PAction learned fixed384 59.10、并逼近/超过 sparse uniform 约 65 mAP 锚点的路线是什么？
5. 如果最终不能超过 65，该路线是否仍有论文价值？需要怎样的 claim 边界？

### C. 更智能、更优雅的选帧方法

请提出比当前实现更优雅的方案，优先考虑真正让后续 AdaTAD detector 告诉 selector 哪些 temporal observations 重要:

1. Detector-aware differentiable acquisition:
   - ST/Gumbel top-k、soft mask、sparse attention、differentiable knapsack 或 subset selection。
   - 如何让 classification/regression/proposal responsibility 回传 selector。

2. Learned context radius / dilation:
   - 当前 `[2,16]` continuous radius head 是否足够？
   - 是否应该预测 multi-scale dilation gates，例如 r=1/2/4/8/16 的 mixture weights？
   - 是否应该按 boundary/action-interior/background 分区域学习不同 context radius？

3. Joint training / curriculum:
   - dense AdaTAD teacher pretrain,
   - selector responsibility distillation,
   - frozen detector sparse distill,
   - partial-unfreeze detector,
   - full joint fine-tune。
   请给出最小可跑版本和完整论文版本。

4. Loss design:
   - boundary responsibility loss,
   - action-local hole loss,
   - high-IoU localization loss,
   - counterfactual utility loss,
   - compute/observation budget regularizer。
   请给出具体公式和 PyTorch 伪代码。

5. Raw-frame vs AdaTAD temporal grid:
   - 当前是在 AdaTAD 768 temporal grid 上选择 observation，不是原始视频帧。
   - 如果要讲 pre-backbone，是否必须从原始 frame 或 Decord decode 前的位置出发？
   - 如果继续在 grid 上做，论文应该如何命名，避免二次加工嫌疑？

### D. 实验计划

请给出一个可执行的实验路线，目标是先得到一个有说服力的 <=384 sparse TAD 结果:

1. 必跑 baseline:
   - dense AdaTAD official setting
   - uniform sparse384
   - random sparse384 multi-seed
   - p_action top-k/fixed384
   - PAction learned fixed384
   - GAS-VT fixed384
   - Stage2 responsibility selector fixed384
   - learned context radius selector fixed384
   - Stage3 joint selector fixed384

2. 必要消融:
   - no dilation
   - fixed r=2
   - fixed r=4
   - learned radius max 4 / 8 / 16
   - no action-local hole loss
   - no context radius cost
   - exact budget vs short-ratio budget
   - responsibility utility vs proposal-score surrogate

3. 必要分析图:
   - mAP@tIoU decomposition
   - boundary error CDF
   - best proposal tIoU CDF
   - action-normalized selected density
   - action-local hole distribution
   - selected context radius distribution by region
   - compute/observation vs mAP Pareto
   - representative success/failure timelines with GT segments, selected observations, detector proposals, and learned radius overlay

4. 请给出关键代码级实现建议:
   - 需要新增/改哪些文件。
   - PyTorch 模型和 loss 该怎么写。
   - AdaTAD/OpenTAD 训练循环该怎么接 detector loss 到 selector。
   - precheck/validator 应如何防止 label leakage 和 teacher utility 泄漏到 val/test deploy。

## 输出格式要求

请按以下结构输出:

1. Visibility check
2. Verdict: PASS / WARN / HOLD / FAIL
3. Blocking code issues with file/line-level references
4. Non-blocking but important code issues
5. Research-route diagnosis: what is elegant, what is not
6. Stronger model proposal with equations or pseudocode
7. Minimal next implementation plan
8. Full CVPR-level experiment plan
9. Claim boundary: what we may and may not claim today
10. Concrete code snippets for the highest-priority fix

请非常严厉，不要为了鼓励而放松标准。如果当前实现只能算 engineering scaffold，请直接说 HOLD，并说明达到论文级还缺什么。
