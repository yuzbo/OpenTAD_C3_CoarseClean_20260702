---
updated: 2026-07-08
status: active
scope: 记录并吸收 Pro/GPT 对 commit fbea37b learned context radius / Stage2 detector-aware acquisition 的 HOLD 审查
out-of-scope: 不声明 Stage2 已通过论文级验证；不把 learned context radius 视为已证明有效的 end-to-end acquisition
---

# fbea37b learned context radius 审查吸收记录

## 原文归档

- 原始审查文本已归档到 `docs/methods/reviews/2026-07-08-fbea37b-learned-context-radius-hold-review-raw.txt`。
- 审查固定对象为 commit `fbea37b878b207f7b93dc7d213b234b43a39a3ea`，分支 `refs/heads/codex/gas-vt-stage23-detector-aware-20260706`。
- 审查结论为 **HOLD，不是 FAIL**：当前分支值得继续推进，但还不能作为论文 claim 或 Stage2/Stage3 完整成功证据。

## 总体吸收

这轮审查确认：`fbea37b` 基本修正了固定预算路径上的关键问题，尤其是 `fixed_deploy_budget=min(requested_budget, valid_len)`、pipeline 默认关闭 short-ratio budget、config loader flag 关闭等。但它同时指出，新增的 learned context radius 仍然是一个 selector surrogate head，并不是 AdaTAD detector loss 真正驱动的端到端采集。

因此当前路线的准确定位应是：

- Stage2 是 dense AdaTAD teacher / point responsibility utility 监督的 offline detector-aware sparse acquisition。
- learned context radius 是一个可训练的上下文膨胀辅助头，用于改善局部覆盖和 action-local hole，而不是已经证明的智能端到端采样策略。
- 当前方法仍工作在 AdaTAD/OpenTAD 的 temporal grid observation 上，不是严格 raw-frame pre-backbone acquisition。
- 在 exact-budget Stage2、lattice、Stage3 joint 结果产出前，论文级 claim 必须保持 HOLD。

## 必须先修的阻塞问题

1. `apply_detector_aware_acquisition_policy.py` 不能对旧 checkpoint fail-open：如果 checkpoint 没有 `context_radius_head` 或 context radius API，现在会退化为默认半径 16，这会把 legacy checkpoint 伪装成 learned radius。应改为 fail-closed，除非显式传入 `--allow-legacy-no-context-radius --legacy-context-radius`。

2. `score_dilated_frame_values` 不能对所有 `center_score > 0` 的点做膨胀。positive utility 不等于局部峰值；这种实现容易退化成广泛平滑或隐式 scaffold。应只对局部峰值、top-percentile 或 high-confidence seeds 膨胀。

3. learned radius 的约束口径需要统一。训练/论文口径是每个 temporal observation 输出 `[2,16]` 范围内的 context radius，但当前 learned 分支存在 `[0,16]` clamp 的不一致。需要统一到 `[2,16]` 或明确 0 的语义。

4. `context_radius_head` 需要梯度与 collapse 测试。当前 surrogate loss 通过 soft context coverage 间接训练 radius，`amax` 可能导致稀疏梯度，sigmoid 可能饱和，radius cost 未必能阻止全局膨胀到 16。必须增加 radius-head nonzero gradient、all-radius-16 collapse guard 等 focused tests。

5. `_extract_action_target` 不能默认用 `gain >= 0.5` 推断 action target。detector responsibility / utility 与 action label 不是同一个语义。应默认要求显式 `action_target`，或者把该项重命名为 `utility_local_hole_loss` 并在 manifest 中写清楚语义。

6. validator 的 exact selected count 应覆盖所有 `valid_len`，不能只在 `valid_len >= 768` 时严查。应统一调用 `paction_budget_contract.expected_selected_count(..., allow_short_valid_ratio_count=False)`。

7. Stage3 / end-to-end 证据仍缺失。当前 commit 的 Stage2 objective 不是 detector loss，不能声称 detector-loss-driven 或 end-to-end。Stage3 必须单独证明 AdaTAD loss backward 后 selector 参数有非零梯度。

## 我接受的判断

我接受这轮审查的核心判断：`fbea37b` 是一个有价值的中间工程和算法推进，但还没有完成论文级模型闭环。尤其是 fail-open legacy radius、positive-score dilation、action target fallback 这三点，如果不修，会直接污染实验结论，让 “learned context radius 有效” 变成不可归因结果。

我也接受它对 claim 边界的约束：目前最多能 claim detector-utility-calibrated sparse temporal grid acquisition 的实现与预检查，不能 claim raw-frame pre-backbone、严格端到端、或者 learned radius 已经带来高 IoU 定位改进。

## 需要保留的研究方向

这轮审查没有否定 learned context radius。相反，它把更优雅的方向收敛为：

- 用 detector responsibility / utility 作为 Stage2 teacher，而不是 proposal-score surrogate。
- 将 radius 从单个连续值进一步升级为多尺度可学习 dilation gates，例如 radii `{1,2,4,8,16}` 的 mixture。
- 在 Stage3 中用 ST/Gumbel top-k、sparsemax/entmax 或 differentiable gap-aware decoding，让 AdaTAD detector loss 能回传到 selector。
- 最小 end-to-end 证明不是 mAP 数字，而是先证明 detector loss 对 selector logits / radius head 有非零梯度，再进行 sparse detector full run。

## 下一步实现顺序

1. 先修 `fbea37b` 的阻塞实现问题：context radius fail-closed、peak-only dilation、radius clamp contract、action target fallback、validator exact selected count。
2. 增加 focused tests：legacy checkpoint 必须失败、显式 legacy flag 才允许；radius head 非零梯度；positive-everywhere utility 不应被全域膨胀；`valid_len < 384` exact budget；val/test deploy 不泄漏 teacher utility。
3. 重新同步新 commit 到远端干净 snapshot，再跑 Stage2 precheck。只有 precheck 通过后才允许 Stage2 full。
4. Stage2 full ablation 按固定 384 预算跑：no dilation、fixed r=2、fixed r=4、learned max=8、learned max=16。
5. Stage3 先跑 gradient proof，再跑 full candidate。若 detector loss 梯度无法稳定进入 selector，不应继续声称端到端。
6. 论文证据链必须补齐：mAP@tIoU 分解、boundary error CDF、best proposal tIoU CDF、action-local hole、selected context radius by region、compute/observation vs mAP Pareto、代表性 timeline/failure gallery。

## 当前 claim 边界

当前可以说：

- 我们已经把 Stage2 从 proposal-score surrogate 推向 point responsibility utility 主线。
- 当前代码具备固定预算、teacher utility provenance、precheck gate 和 learned context radius 的初步实现。
- PAction learned fixed384 已知强于 GAS-VT fixed384，lattice move50 早期结果也显示超过 GAS-VT，但还未超过 PAction learned fixed384 和 dense AdaTAD。

当前不能说：

- 不能说 learned radius 已证明提升 TAD。
- 不能说 Stage2 已经超过 uniform sparse384 或 dense upper bound。
- 不能说当前方法是严格 raw-frame pre-backbone。
- 不能说当前 Stage2 是端到端。
- 不能把旧 `5f21d30` Stage2 failure 或旧 surrogate run 当成新责任主线结论。

## 吸收后的决策

这份审查把下一步从“直接重排 learned-radius Stage2 full”改成“先修阻塞实现，再重排”。尤其要避免使用 fail-open context radius 或全正分数膨胀产生看似更好的结果。短期目标仍是拿到一个在 384 预算下有说服力的 sparse TAD 结果；长期目标是把 offline teacher selector 过渡到 detector-loss-aware joint acquisition。
