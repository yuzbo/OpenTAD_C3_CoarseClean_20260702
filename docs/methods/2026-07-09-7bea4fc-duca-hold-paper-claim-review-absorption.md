---
updated: 2026-07-09
status: active
scope: 吸收 7bea4fc DUCA HOLD 审稿裁决，约束论文主张、最终模型解释、实验排布与后续实现优先级
out-of-scope: 不替代实验结果表；具体 mAP、FLOPs、日志异常仍以 evaluation/results.md 或远端日志为准
---

# 2026-07-09 7bea4fc DUCA HOLD Review Absorption

## 原始记录

- 原始审查文件已保存到 `docs/methods/reviews/2026-07-09-7bea4fc-duca-hold-paper-claim-review-raw.txt`。
- 原文 SHA256：`7C4207B30186179986F37CEF92F92059E6FDA0BB91C416560D58D7139D5555B4`。
- 审查对象：GitHub commit `7bea4fc`，标题为 `Fix DUCA official proof loss config`。
- 仓库链接：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/gas-vt-stage23-detector-aware-20260706`。

## 总裁决

本轮审查必须被吸收为一个明确判断：

**DUCA 当前已经进入论文方法候选状态，但仍是 HOLD，不能直接支撑主论文 claim。**

`7bea4fc` 的新增价值主要是修复 official one-step gradient proof 脚本，而不是完成新的 full training 路线。它可以证明 official ActionFormer/AdaTAD 风格 backend 的 detector loss 在 one-step proof 中能连到 coarse probe、selector、dynamic budget controller；但它还不能证明 selector 真正学到了 detector-utility-calibrated temporal acquisition，也不能证明 DUCA-MUST dynamic adaptive budget policy 有效。

最危险的问题不是“有没有梯度”，而是：

1. `detector_utility` 仍不是 true detector-derived utility。
2. dynamic MUST 当前实验近乎崩溃。
3. 正在跑的主实验不是最新 `7bea4fc`。
4. fixed-384 的较好 mAP 只能说明链路不死，不能单独证明方法成立。

## 已确认可以保留的正确方向

1. **最终方法仍应是 detector 前的 online pre-backbone temporal acquisition plugin。**
   不是完整新 TAD detector，不是离线 ledger，不是 dense X3D 预提取 pipeline。

2. **粗分类 probe 必须在线产生 deploy-visible actionness 和 hidden features。**
   当前 `C3CoarseProbeActionnessSource` 方向正确：粗分类不只是外部 `p_action` 曲线，而应向 selector 提供低维时序 hidden features。

3. **selector 必须 transition/boundary/utility-first。**
   actionness 权重可以存在，但只能是辅助校准；不能把 selected positions 退化成 actionness coverage top-k。

4. **后端必须是 unmodified official AdaTAD/ActionFormerHead。**
   validator 禁止 `DucaOnlinePrecheckHead` 是正确约束。DUCA 只应改前置采集和必要的 selected-axis / coordinate remap，不应魔改 detector head/loss/NMS。

5. **detector loss gradient bridge 是必要贡献，但当前只能称为 surrogate feedback。**
   `soft_to_hard_resample` 可以作为 straight-through soft-to-hard surrogate；论文不能宣称 hard selection 本身完全可微。

6. **X3D/SlowFast 只能作为 frozen prior baseline / appendix / upper-bound diagnostic。**
   dense video prior 如果扫完整视频，会吞掉选帧节省，不能作为主 pre-backbone 插件。

## 必须收紧的论文表述

可以写：

> DUCA is an online pre-backbone temporal acquisition plugin with an online C3 coarse probe, transition/boundary-oriented selection, original-time selected positions, an unmodified official ActionFormer/AdaTAD backend, and detector-loss gradient feedback through a soft-to-hard surrogate. Current utility supervision is a GT boundary-utility proxy; detector awareness is supplied by end-to-end detector loss gradient, not by true detector-derived utility labels.

不能写：

- `true detector-derived utility target`，除非 target 真来自 detector loss sensitivity、proposal quality delta、NMS survival 或 mAP delta。
- `fully differentiable hard frame selection`。
- `dynamic budget already gives real variable-length detector FLOPs saving`，除非 backend 真按 variable K 执行。
- `X3D train-free is the main low-cost pre-backbone module`。
- `fixed-384 40.94% proves the final method`。

## 当前实现的关键风险

### R1: Actionness 隐性主导风险

虽然配置把 `actionness_weight` 压到 0.05，并提升 `transition_weight` / `boundary_weight`，但 transition features 很多仍由 `p_action` 派生，例如 `delta_p_action`、`abs_delta_p_action`、`uncertainty_peak`。因此“transition-first”可能仍是“p_action change-first”。

必须补强的证据：

- `actionness_weight=0`。
- no p_action transition features。
- hidden-only transition。
- p_action top-k baseline。
- full DUCA。
- score component attribution。

### R2: Utility 命名高危

当前所谓 `detector_utility_target` 本质是 GT boundary utility proxy。validator 已经要求它只能作为 deprecated alias / proxy 使用。这一点必须同步到代码、配置、论文和实验表头。

建议统一术语：

- `boundary_utility_proxy_target`
- `utility_proxy_target_kind="gt_boundary_utility_proxy"`
- `detector_loss_feedback` 用于描述 detector gradient，而不是监督 target。

### R3: One-step proof 不等于 full training 成立

`7bea4fc` 的 one-step proof 只证明计算图连通和 optimizer coverage，不证明训练稳定、mAP 有效或 dynamic budget 可用。因此 `7bea4fc` 必须排 full train，不能用旧 commit running jobs 直接替代。

### R4: Dynamic MUST 是当前最大方法风险

当前 dynamic jobs 的 mAP 过低，可能由以下因素叠加导致：

- backend 仍是 `budget_max` cap 下的 padded/capped detector，不是 natural variable length。
- 低预算 target 过激，尤其 128/768 时有效 slots 太少。
- detector-gradient schedule 早期较弱，selector 先被 proxy / max-gap 牵引到坏分布。
- budget controller 有梯度不等于学到 mAP-aware stop policy。
- hard max-gap repair 可能在 low budget 下消耗 slots，掩盖 selector 学习失败。

### R5: Max-gap 需要拆分证明

当前应被描述为 `soft loss + hard repair`。必须区分：

1. no max-gap；
2. soft max-gap loss only；
3. hard repair only；
4. soft loss + hard repair。

否则 reviewer 会认为效果来自 deterministic repair，而不是 policy 学会了避免时序空洞。

## 当前 running experiments 的归类

审查意见明确：当前 running experiments 不能进入论文 main table，只能作为旧版本诊断。

- `1151863 fixed-384 official/JCT, commit 0b0b0f5, Avg 40.94%`：旧版本链路 sanity / diagnostic。
- `1151864 dynamic, commit 0b0b0f5, Avg 7.69%`：dynamic collapse evidence。
- `1151927 fixed-384, commit 009f9d7, Avg 19.40%`：旧版本不稳定诊断。
- `1151928 dynamic, commit 009f9d7, Avg 2.90%`：dynamic collapse evidence。
- `1151929 fixed budget curve budget=128, Avg 5.86%`：低预算旧实现诊断。
- `1151955 MUST target 128, Avg 3.12%`：dynamic low-budget failure diagnosis。

主论文结果必须来自同一 commit、同一 official backend、同一数据管线、同一训练协议的完整实验。

## 必须补的主实验

### Main table

- Dense official AdaTAD / ActionFormerHead, 768 observations。
- Uniform fixed K=384 / 256 / 128。
- Random fixed K=384 / 256 / 128，至少 3 seeds。
- C3 actionness top-k。
- DUCA fixed K=384。
- DUCA fixed K=256。
- DUCA-MUST dynamic target 384 / 256 / 128，并报告 actual average budget。

### Claim-critical ablation

- no detector-gradient bridge。
- no coarse hidden features。
- actionness_weight = 0 / 0.05 / 0.25 / 1.0。
- no p_action transition features。
- hidden-only transition。
- no boundary loss。
- no boundary utility proxy。
- no max-gap / soft only / hard only / soft+hard。
- fixed policy forced K vs dynamic MUST same adapter。
- dynamic fixed-384 warm-start vs no warm-start。
- budget controller on/off。

### Metrics beyond mAP

- Avg-mAP and IoU-wise mAP。
- selected_count mean/std/p5/p95。
- actual detector-consumed observations。
- expected cost vs actual selected count。
- max unselected hole mean/p95/max。
- boundary recall@selected。
- selected-to-nearest-boundary distance。
- selected-in-action-interior ratio。
- repair ratio。
- soft_resample entropy。
- gradient norm by module。
- total FLOPs / latency / memory including coarse probe + selector + detector。

## 多检测头泛化最小闭环

主 backend：ActionFormerHead / AdaTAD official backend。

第二 backend：优先 TriDetHead 或 TemporalMaxerHead，选一个 OpenTAD 已有官方 config、训练成本可控、head/assignment 与 ActionFormer 不完全相同的检测头。

最小矩阵：

| Backend | Dense | Uniform-384 | DUCA fixed-384 | DUCA-MUST target-256 |
| --- | --- | --- | --- | --- |
| ActionFormerHead | 必跑 | 必跑 | 必跑 | 必跑 |
| TriDetHead 或 TemporalMaxerHead | 必跑 | 必跑 | 必跑 | 可选但强烈建议 |

规则：selector、loss weights、max-gap、budget policy 不随 backend 重调；否则会被认为是 per-head hand tuning。

## 立即执行优先级

### P0

1. 立即排最新 `7bea4fc` fixed-384 official full training。
2. 立即排最新 `7bea4fc` DUCA-MUST dynamic official full training，但先从 target 384 / 320 / 256 诊断，不要直接押 128。
3. 对 dynamic 引入 fixed-384 warm-start，再解锁 budget controller，再 anneal target。
4. 在每个 epoch 强制记录 dynamic budget diagnostics。
5. 把旧 commit results 从 main evidence 中隔离，只保留为 diagnostic。
6. 保持 dynamic config 的 `paper_claim_allowed=False`，直到 full result 和 diagnostics 过关。

### P1

1. 补 actionness dominance ablation。
2. 补 hidden feature ablation。
3. 补 max-gap ablation。
4. 补 no-detector-gradient ablation。
5. 补同 commit dense / uniform / random / C3 top-k baselines。
6. 补完整 cost table。

### P2

1. 补一个第二 detector backend。
2. 把 X3D/SlowFast 固定为 appendix prior baseline。
3. 将 0b0b0f5 / 009f9d7 等旧 run 整理成 failure analysis，不混入主结果。

## 吸收后的执行原则

从本轮开始，后续 DUCA 工作必须遵守三个硬门槛：

1. **不再把 one-step proof 当作论文主结果。**
2. **不再把 boundary utility proxy 说成 detector-derived utility。**
3. **不再用旧 commit full run 证明最新最终方法。**

如果 dynamic MUST 不能恢复到合理 mAP，论文主线应及时降级为：

> fixed-budget online DUCA plugin + detector-loss feedback + transition/boundary-first acquisition

而不是强行主打 adaptive dynamic budget。
