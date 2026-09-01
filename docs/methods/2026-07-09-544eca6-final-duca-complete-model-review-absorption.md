---
updated: 2026-07-09
status: active
scope: 吸收 544eca6 最终 DUCA 完整模型审查，约束代码实现必须成为论文主方法最终版
out-of-scope: 不记录实验结果数值；不替代 evaluation/results.md
---

# 2026-07-09 544eca6 Final DUCA Complete Model Review Absorption

## 原始记录

- 原始审查全文：`docs/methods/reviews/2026-07-09-544eca6-final-duca-complete-model-review-raw.txt`
- 审查对象：`codex/gas-vt-stage23-detector-aware-20260706` / `544eca6 Make DUCA selector transition-first`
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

## 必须吸收的总裁决

当前 `544eca6` 已经是正确方向上的 DUCA online selector 主干，但还不是论文最终模型。最终代码不能停留在 precheck、proof-of-concept 或旧路线零件拼贴状态。

最终 DUCA 必须是同一个主干内完整闭环：

```text
raw/low-res video
  -> online lightweight coarse actionness probe
       -> p_action
       -> actionness_logits
       -> transition profile
       -> coarse_hidden_features [B,T,D]
  -> DUCA selector
       -> descriptor + transition profile + projected coarse hidden
       -> transition/boundary/utility-first score
       -> fixed-384 or dynamic MUST budget
       -> soft max-gap loss
       -> hard max-gap repair
  -> hard original-time selected positions
  -> official AdaTAD / ActionFormerHead
  -> detector loss
  -> ST / soft-to-hard surrogate feedback to selector and coarse probe
```

## 当前已有但分散的零件

必须承认：项目里已经实现过大量相关能力，不应重复造轮子。

1. DUCA online skeleton、fixed-384、dynamic MUST、official AdaTAD config 已在 GASVT 主干。
2. ST / soft-to-hard resampling bridge 已在 `DucaOnlineFrameSelector` / `DucaAcquisitionAdapter`。
3. online C3 coarse probe 已接入 DUCA。
4. TrueTime 路线已有真实 ActionFormer selector gradient proof，可迁移 proof 思路。
5. paction/move/lattice 旧路线已有 soft gap loss、large gap repair、max-gap guard，可迁移算法思想。

但上述零件没有全部合入最终 DUCA 主干。

## 必须立即合入主干的 P0 项

1. **coarse hidden feature path**
   - `C3CoarseProbeActionnessSource.forward()` 必须输出 `coarse_hidden_features [B,T,D]` 和 `coarse_hidden_dim`。
   - `DucaAcquisitionAdapter` 必须接收并投影 hidden features。
   - selector input 必须是 `descriptor + transition profile + hidden_proj`。
   - curve-only 只能作为 ablation，不是主配置。

2. **official detector-loss grad proof**
   - 新增 `tools/bata/run_duca_official_adatad_one_step_grad_proof.py`。
   - 必须使用 official `ActionFormerHead`，不能用 `DucaOnlinePrecheckHead`。
   - 必须证明 `losses["cost"].backward()` 后 coarse probe、coarse hidden projection、selector encoder、center/boundary/utility heads、dynamic budget controller、official detector head 都有非零梯度。

3. **soft max-gap loss**
   - 把旧 `paction_acquisition_policy.py` 的窗口级 gap loss 思想合入 DUCA loss。
   - 主配置默认 `max_gap=10`，`soft_max_gap_loss_weight>0`。
   - fixed 和 dynamic MUST 都必须启用。

4. **hard max-gap repair**
   - 把旧 `_repair_large_gaps()` / max-gap guard 思想迁移进 DUCA decode。
   - `budgeted_center_radius_decode()` 后必须保证 `largest_gap_after <= max_gap`，且不超预算。
   - repair 只能作为安全约束，不得退化成 uniform scaffold 主导。

5. **utility target 命名修正**
   - GT boundary proxy 不得再称为 `detector_utility_target`。
   - 主路径改成 `boundary_utility_proxy_target` / `utility_proxy_target_kind="gt_boundary_utility_proxy"`。
   - 真正 detector-derived target 只能留给未来 detector sensitivity / proposal IoU gain / NMS survival / mAP delta。

## 最终训练方案约束

最终 loss：

```text
L =
  L_detector
+ lambda_action(t) * L_actionness_bce
+ lambda_boundary(t) * L_boundary
+ lambda_proxy(t) * L_boundary_utility_proxy
+ lambda_gap * L_soft_max_gap
+ lambda_budget(t) * L_dynamic_budget
+ optional lambda_entropy/redundancy
```

训练要求：

- detector loss 始终训练 official detector backend。
- detector ST surrogate gradient 必须进入 selector 和 coarse probe。
- actionness BCE 只训练 coarse probe 的动作性，不得长期主导 selector。
- transition/boundary/utility/proxy/gap 是 selector 主监督。
- fixed-384 和 dynamic MUST 共用同一 acquisition policy。
- X3D/SlowFast 只能作为 diagnostic/frozen prior baseline，不能作为主方法低成本 selector。

## validators 必须 fail-closed

主配置必须检查：

- `use_coarse_hidden_features=True`
- `max_gap` enabled
- `soft_max_gap_loss_weight > 0`
- `hard_gap_repair=True`
- official detector grad proof JSON 存在且通过
- `DucaOnlinePrecheckHead` 不得作为最终 proof
- rpn head 保持 official `ActionFormerHead`
- no ledger / no external actionness / no raw prediction cache
- X3D/SlowFast 不得是 main method actionness source

## 论文声明边界

可以说：

- online pre-backbone acquisition plugin
- transition/boundary/utility-first
- official AdaTAD / ActionFormerHead backend
- detector-loss ST feedback
- soft max-gap regularized acquisition with hard max-gap repair

不能说：

- fully differentiable hard selection
- detector-derived utility target，除非真的实现 detector-derived target
- X3D/SlowFast 是主方法低成本 selector
- dynamic MUST 真实节省 detector FLOPs，除非 backend variable-length 执行并完成 profiling

## 本轮实现门槛

本轮不是写 prompt 或记录讨论。必须落代码，至少达到：

1. final DUCA hidden-feature path 可测。
2. DUCA soft max-gap loss 可测。
3. DUCA hard max-gap repair 可测。
4. official detector grad proof 脚本存在。
5. validators 和 tests 能阻止半成品配置被当作最终模型。
