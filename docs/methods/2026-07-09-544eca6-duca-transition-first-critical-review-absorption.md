---
updated: 2026-07-09
status: active
scope: 吸收 544eca6 DUCA transition-first 严厉审查结论，约束后续模型实现、实验排布与论文表述
out-of-scope: 不记录新的实验 mAP 数字；实验结果仍以 evaluation/results.md 或运行日志为准
---

# 2026-07-09 544eca6 DUCA Transition-First Review Absorption

## 原始记录

- 原始审查文件已保存到 `docs/methods/reviews/2026-07-09-544eca6-duca-transition-first-critical-review-raw.txt`。
- 审查对象：GitHub commit `544eca6`，标题为 `Make DUCA selector transition-first`。
- 仓库链接：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- commit 链接：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/544eca6`

## 总裁决

本轮审查必须吸收为一个明确结论：

**544eca6 是正确方向上的明显进步，但仍不是论文级闭环。**

它已经从早期 `PAction / GAS-VT / ledger heuristic` 路线转向 online pre-backbone selector plugin，并且建立了重要合同：禁止 ledger、禁止 external actionness、使用 trainable coarse probe、transition/boundary/utility-first、fixed-384 与 dynamic MUST 共用 adapter、官方 backend 不改 head/loss/NMS。

但它仍只能被判为 **MAJOR REVISION / HOLD**。当前实现更准确的状态是：

> online trainable transition-prior selector skeleton + proxy-supervised boundary/utility loss + ST surrogate bridge + official-backend config contract

不能无保留宣称：

> transition-first, jointly trained, detector-loss-calibrated, dynamic online acquisition plugin

## 已确认吸收的正确方向

1. **主方法必须是 online pre-backbone temporal acquisition plugin。**
   不是新 TAD detector，不是离线 ledger，不是 X3D dense export，不是 detector 后处理。

2. **粗分类模型负责产生 deploy-visible actionness。**
   粗分类监督可以是 binary actionness label，但它只能提供动作性与时序表征，不能让 selector 退化为 actionness top-k。

3. **间接选帧模块必须 transition/boundary/utility-first。**
   actionness 只能是小权重辅助输入，不能成为最终 selected positions 的主导因素。

4. **官方 AdaTAD/ActionFormer backend 必须保持原始 head/loss/NMS。**
   DUCA 只应插在 detector 前，最多做 selected-axis / coordinate remap / wrapper，不应魔改检测头。

5. **fixed-384 与 dynamic MUST 应是同一 acquisition policy 的两个特例。**
   fixed 是 `K=384`，dynamic 是 learnable/prefix marginal utility stop，不应是两套互不相干的工程逻辑。

6. **X3D 只能是 diagnostic / appendix / upper-bound reference。**
   dense X3D 不是低成本 pre-backbone probe；若它先扫完整视频，新增计算可能淹没后续省下的 detector 计算。

## 未完全吸收的 P0 问题

### P0-1 selector 仍偏 curve-only

当前 selector 主要看到低成本 descriptor 与 actionness 派生标量曲线：

- `p_action`
- `uncertainty`
- `entropy`
- `delta_p_action`
- `abs_delta_p_action`
- `uncertainty_peak`
- `transition_score`

但它没有看到 coarse classifier 的 `[B,T,D]` hidden temporal features。这样会削弱“粗分类模型 + 间接 selector 协同学习边界”的方法性，容易被审稿人打成：

> actionness derivative heuristic + MLP scorer

必须补：

- `C3CoarseProbeActionnessSource.forward()` 返回 `coarse_hidden_features [B,T,D]`。
- official ASFormer / MobileNet / TCN probe 暴露低维 temporal embedding。
- `DucaAcquisitionAdapter.acquire()` 接收并投影 coarse hidden。
- `DucaOnlineFrameSelector._forward_select()` 将 hidden 传入 adapter。
- 主 config 默认 `use_coarse_hidden_features=True`，curve-only 只作为 ablation。

### P0-2 `detector_utility_target` 命名仍有误导

当前 target 本质是 GT boundary utility proxy，但部分 API / loss 参数仍叫 `detector_utility_target`。这会让 reviewer 追问它是不是 detector-derived utility。

必须改名：

- `boundary_utility_proxy_target`
- `transition_boundary_proxy_target`
- `utility_proxy_target_kind="gt_boundary_utility_proxy"`

只有真正来自 detector loss sensitivity、proposal IoU gain、mAP delta 或 NMS survival 的 target 才能叫 `detector_utility_target`。

论文中也不能写 `detector-derived utility target`。当前路径只能称为：

> Boundary-Utility Proxy Supervision

真正 detector loss 的影响只能称为：

> Detector-loss ST feedback

### P0-3 ST bridge 不是 hard selection 的真实可微优化

当前 hard gather/top-k 本身仍不可微。`soft_to_hard_resample` 或 ST 权重只是 surrogate gradient path。

论文不能写：

> detector loss differentiably optimizes selected positions

应写：

> Detector loss is propagated to the acquisition scorer through a straight-through soft-to-hard resampling surrogate while inference uses strict hard original-time positions.

### P0-4 one-step grad proof 不是 official detector proof

当前 one-step proof 使用 `DucaOnlinePrecheckHead`，不是官方 AdaTAD/ActionFormer detector head。因此它只能证明 precheck/toy loss 下 coarse probe、selector、budget controller 有梯度，不能支撑：

> official AdaTAD detector loss 反向影响 selector

必须新增 official backend 版本：

- 使用官方 AdaTAD/ActionFormer head。
- 使用真实 detector loss。
- `losses["cost"].backward()` 后断言：
  - coarse probe grad > 0
  - selector encoder grad > 0
  - score heads grad > 0
  - dynamic budget controller grad > 0
  - detector head grad > 0

## 未完全吸收的 P1 问题

### P1-1 dynamic MUST 可能没有真实 runtime/FLOPs 节省

如果 detector 实际仍消费 cap-length padded representation，dynamic policy 的平均 K 不能直接换算成 detector FLOPs 节省。

论文必须同时报告：

- policy selected K
- actual detector-consumed observations
- total runtime/FLOPs，包括 coarse probe 与 selector

若 backend pad 到 384，就不能把 dynamic average K 当成真实 detector compute。

### P1-2 缺少显式 max-gap 机制

`max_radius` 不是 `max_gap`。它只控制 center-radius decode 的局部扩展，不能保证任意相邻 selected positions 的最大间隔。

必须新增：

- `max_gap`
- `gap_loss_weight`
- soft gap regularizer
- hard gap repair decoder

主实验建议先用 `max_gap=10`，再做 `8/10/12/15/None` ablation。`max_gap=15` 不应直接作为默认主结果，因为当前存在 clustered but shifted 风险，过大间隔会放大边界 miss。

### P1-3 utility proxy 可能仍把 selector 拉向动作内部

binary actionness label 会鼓励动作内部 plateau。若 boundary utility proxy 仍给动作内部较高 reward，selector 会自然选择动作内部，而不是 start/end transition。

必须把 proxy 改成：

```text
target = boundary_gaussian(start/end)
       + lambda_transition * abs label transition
       + small inside_action_floor
```

其中 inside action floor 应很小，例如 `0.05-0.10`，不能主导 utility。

### P1-4 transition-first 必须变成行为事实

不能只依靠 config 里 actionness weight 小来证明 transition-first。必须记录 score component attribution：

- actionness component
- transition component
- uncertainty component
- boundary component
- utility component

并统计 selected positions 上各 component 的贡献，否则 transition-first 只是宣称。

## move25 / move50 / dilation 聚集但偏移的原因假设

审查结论要求后续必须定位偏移来源，而不是继续加 dilation 或 actionness coverage。

最可能原因：

1. **actionness binary label 太粗。**
   BCE 鼓励动作内部全高分，不鼓励边界尖峰，导致 p_action plateau 或 delta peak 偏向动作内部。

2. **coarse probe temporal calibration / stride / smoothing 造成系统性延迟。**
   ASFormer/TCN smoothing 可能让 start/end transition 向后或向内偏移。

3. **selector soft loss 与 hard decode 不一致。**
   训练期 soft coverage 与推理期 hard top-k/center-radius decode 的 peak 可能错位。

4. **coordinate mismatch / off-by-one。**
   必须测试 GT segment 秒/帧/snippet index 到 original-time selected positions 的映射，特别是 THUMOS 的 inclusive/exclusive end、feature_stride、sample_stride、window offset。

必须新增诊断指标：

- p_action peak 到最近 GT boundary 的 signed offset
- abs_delta peak 到最近 GT boundary 的 signed offset
- start/end 分开统计
- selected-to-nearest-boundary distance
- selected-in-action-interior ratio
- largest unobserved gap
- soft peak 与 hard selected center 的偏差

## 必须补的代码任务

1. `opentad/models/duca/acquisition.py`
   - coarse probe 返回 `coarse_hidden_features [B,T,D]`
   - adapter 支持 `coarse_hidden_dim`
   - adapter 支持 `coarse_hidden_proj_dim <= 64`
   - adapter 支持 `use_coarse_hidden_features`
   - hidden projection FLOPs 计入 cost

2. `opentad/models/selectors/duca_online_frame_selector.py`
   - `_forward_select()` 传递 coarse hidden features
   - proxy target 全面改名，避免 `detector_utility_target` 误导
   - boundary utility proxy 降低动作内部 floor
   - 新增 max-gap regularizer 与 hard gap repair
   - 新增 score component logging

3. `tools/bata/`
   - 新增 official AdaTAD one-step detector grad proof
   - validators 检查 hidden features、score attribution、max_gap、proxy 命名、actual detector consumed length

4. `tests/`
   - `test_duca_coarse_hidden_feature_contract.py`
   - `test_duca_score_component_dominance.py`
   - `test_duca_max_gap_contract.py`
   - `test_duca_official_detector_grad_bridge.py`
   - `test_duca_coordinate_mapping_no_off_by_one.py`
   - `test_duca_dynamic_fixed_unified_policy.py`

## 必须补的实验

主论文最小闭环：

1. Fixed-384 DUCA main result。
2. Dynamic MUST main result。
3. Official AdaTAD backend full train。
4. 至少一个第二 backend，如 ActionFormer。
5. 完整 cost table：coarse probe FLOPs、selector FLOPs、detector/backbone FLOPs、total FLOPs、wall-clock、memory、selected K。
6. boundary coverage metrics：selected-to-boundary distance、boundary recall within ±r、selected-in-action-interior ratio、largest unobserved gap、mAP@0.6/0.7。

核心 ablation：

1. curve-only vs curve+coarse-hidden。
2. actionness-first vs transition-first。
3. boundary proxy vs detector-derived utility / detector-gradient-only。
4. hard only / ST sparse gather / soft-to-hard resample。
5. `max_gap=8/10/12/15/None`。
6. dynamic budget min/target/max。
7. utility proxy inside-action weight。
8. official ASFormer vs MobileNet/TCN。
9. frozen coarse probe vs jointly trainable coarse probe。
10. hidden dim 16/32/64。

Appendix / diagnostic：

1. X3D train-free dense export。
2. cached actionness / ledger replay。
3. oracle boundary utility。
4. move25 / move50 / dilation visualization。
5. failure cases：clustered but shifted、long actions、short actions、dense action sequences。

## 论文表述边界

应该写：

> DUCA is an online pre-backbone temporal acquisition plugin. A lightweight coarse actionness probe produces deploy-visible per-frame actionness logits and low-dimensional temporal hidden features. The acquisition policy uses transition, uncertainty, boundary-proxy, and marginal-utility signals to select strict original-time observations under a hard budget. The selected observations are passed to an unmodified TAD detector. During training, detector loss is propagated to the acquisition policy through a straight-through soft-to-hard surrogate, while inference uses hard selected positions only.

不要写：

- `detector-derived utility target`，除非 target 真来自 detector sensitivity。
- `fully differentiable hard selection`。
- `X3D train-free is our main low-cost selector`。
- `dynamic budget reduces detector compute to average K`，除非 backend 真按 variable K 执行。
- `boundary-first 已被证明`，除非有 selected-to-boundary metrics 和 transition-first ablation。
- `official AdaTAD fully joint training proof`，除非有 official backend detector-loss grad proof。

## 后续执行优先级

P0 必须先做：

1. coarse hidden features 进入 selector。
2. proxy target 重命名并降低动作内部 floor。
3. official detector-loss one-step grad proof。
4. score component attribution。
5. max-gap regularizer/repair。

P1 再做：

1. 第二 backend。
2. cost profile。
3. boundary metrics。
4. max_gap 与 hidden-dim ablation。

暂停：

- 不再将 dense X3D 作为主方法部署。
- 不再把 actionness coverage 当作 selector 主目标。
- 不再部署旧 commit 的重复 full run，除非明确作为 diagnostic。
