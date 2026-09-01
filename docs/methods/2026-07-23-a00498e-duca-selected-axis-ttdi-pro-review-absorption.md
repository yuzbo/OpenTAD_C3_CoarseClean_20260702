# DUCA selected-axis / TTDI Pro 审查吸收与当前代码复核

## 原始记录

- 审查对象：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 审查分支：`codex/duca-boundary-burst-20260722`
- 审查提交：`a00498e15d69294f78d0abeadfb47bc456db0b0e`
- 原始附件：`C:/Users/skywalker/.codex/attachments/61034c80-3c84-4534-a575-3024e7e7a651/pasted-text.txt`
- 仓库原文归档：`docs/methods/reviews/2026-07-23-a00498e-duca-selected-axis-ttdi-pro-review-raw.txt`
- 原文/归档大小：`76,688` bytes
- 原文/归档物理行数：`1,705`
- 原文/归档 SHA-256：`36523b2f1a7456f8d4a4314ea445971f8066eec59611f9632d7bc1d33e31a884`
- 字节一致性：`true`

## 时间边界

该回复实际审查的是 `a00498e`，不是当前正在运行的 `9f97f2c`。远端逐项复核确认：

- `a00498e -> 9f97f2c` 只修改两个 R0-R5 bundle 启动脚本和对应测试；selector、decoder、
  VideoMAE、detector、loss 与训练日程没有改变。
- `9f97f2c -> 4f81299` 只新增四种粗分类后端 P0 启动脚本和测试；DUCA 主模型语义没有改变。

因此，审查中针对模型本体的结论仍适用于 `9f97f2c/4f81299`；针对旧部署拓扑、跨提交预算
合并和旧 Job 的描述必须按当前队列重新解释。

## 项目裁决

```text
SUBSTANTIAL_ACCEPT_MODEL_DIAGNOSIS
PARTIAL_ACCEPT_TTDI_REMEDY
REJECT_UNCONDITIONAL_FULL_TTDI_AS_FINAL_MODEL
REJECT_STALE_A00498E_JOB_PLAN
```

我不完全同意原回复，但认可它抓住了当前最值得优先验证的结构风险。TTDI 是当前结果出来后
的首选候选实验，不是已经证明正确的最终模型，也不能在没有 matched terminal mAP 的情况下
替代正在运行的 `9f97f2c`。

## 逐项核验

| 审查结论 | 当前复核 | 裁决 |
|---|---|---|
| 非均匀 selected frames 在 VideoMAE/projection/head 中被当成等间隔 rank | `duca_r5_paper_matrix.py` 仍固定 `detector_axis="selected_axis_index"`；当前没有 TTDI 文件；ActionFormer 仅在输出端保存 inverse-map metadata | **确认，最高优先级模型风险** |
| posthoc true-time inverse map 不能恢复前面已经发生的 tubelet/Conv/attention 时间扭曲 | 代码只在输出坐标端 remap；当前 heavy feature extractor 未消费真实间隔 | **合理推断，需 U/L/T mAP 证伪** |
| mandatory group 进入 DP 后退化成 bool union | `build_mandatory_bilateral_set` 最终只返回 union mask、center mask、group count；DP 只接收 `required_mask` | **确认** |
| mandatory set 没有接纳前 completeability 检查 | `_required_selection_mask` 只验证 shape 和 `required_count <= K`；不可补全时由后续 Viterbi fail-closed | **确认，属于正确性与低预算稳定性问题** |
| G1 detector bridge 是 expected-position × local RGB slope surrogate | 当前 selector 仍使用 `_add_protected_structured_transport_gradient_path`，日志合同也写明 `expected_position_and_local_temporal_slope_lower_bound` | **确认；非零梯度不等于正确 utility** |
| 五预算成本 parser 仍只认 K384/K256 | 当前 `profile_duca_full_stack_cost.py` 仍硬编码 `{384:2,256:3}` 和 `384|256`，而 R5 矩阵已含五预算 | **确认，影响成本证据，不影响已启动 mAP cell** |
| 五点 aggregate 的跨提交语义等价校验不足 | 聚合器本身仍缺 model-contract hash | **代码问题确认；当前 `9f97f2c` 已统一五预算，因此本轮直接风险已降低** |
| G2 只是 mixed-policy batch，不是逐样本 paired counterfactual | 当前 G2 每个样本只走一种 policy | **确认，应修正论文表述** |
| detector 应称 official-derived components + extended wrapper | head/projection/NMS 主体沿用官方，wrapper、selected-axis 映射和 optimizer 接口有扩展 | **完全同意** |
| `DucaOnlineFrameSelector` 名称误导 | 实际是 offline full-window TAD | **同意，但不是性能阻塞项** |

## 对原始设计初心的判断

当前实现仍保持核心初心：动作 head 由二分类动作性监督；transition descriptor 不使用绝对
actionness top-k；selector 通过状态变化、隐藏特征变化和不确定性间接定位边界；预算可跨区域
转移并允许边界微簇；最终 hard gather 发生在原时间轴、heavy VideoMAE 之前。

需要诚实补充：transition/burst loss 可通过受限 route 更新 ASFormer 最后一层。因此“action head
保持纯二分类”成立，但“整个 coarse representation 只受二分类监督”不成立。更准确的表述是：

> 二分类动作语义是粗模型主任务；ASFormer 最后一层允许弱边界适配；official-60 阶段 coarse
> branch 冻结，detector feedback 只作用于 selector scorer。

## 我不完全认可的部分

1. **不能现在就把完整 TTDI 写成最终模型。** 当前还没有 `9f97f2c` 的 terminal official mAP，
   也没有证明 learned selection quality 已优于 uniform。若 selector 本身没选好，TTDI不会解决根因。
2. **原回复所谓“一次结构修改”实际包含两个变量。** 零初始化 true-time feature residual 与
   physical-coordinate head/GT assignment 是两项不同改动，同时加入会失去归因。
3. **2000/6000 updates、0.10/2.0/0.20 权重、55% sign gate 等是提案，不是实验证实的最优值。**
   可以作为预注册起点，不能写成理论必然或最终配方。
4. **继续 ASFormer 是当前受控默认，不是永久裁决。** 四种粗分类后端 P0 正在运行；在统一
   temporal-hidden 并完成 TAD 对照前，不能宣布 ASFormer 必然最好。
5. **R2Q3 也仍是候选。** 受污染的 R0 内部回放不能把它封为最终 family；最终由完整 validation
   matched mAP 与边界/上下文诊断共同决定。

## 正确的下一步顺序

1. 保持 `9f97f2c` Jobs `1180490--1180496` 继续运行，不取消、不篡改，不把本审查追溯性地
   应用于已经启动的 cell。
2. 收割 terminal epoch-59 EMA 的 U/L 五预算、两后端、三种子 mAP，并同步收割 hard selection
   的 center error、bilateral coverage、gap/Jaccard、动作内部和背景上下文覆盖。
3. 若 hard selection 没有优于 uniform，先修 scorer/mandatory/可行域；不做 TTDI。
4. 若 hard selection 明显更好但高 tIoU mAP 不升，做最小三臂：
   `U = uniform`、`L = current learned rank-time`、`T1 = L + zero-init true-time feature residual`。
5. 只有 T1 仍暴露 assignment/regression 的时间尺度错误时，再做 `T2 = T1 + physical-coordinate head`。
   不把 T1/T2 合并成无法归因的一次实验。
6. legal hard-swap sign/Spearman 不通过时，关闭当前 detector feedback；通过后再比较 current bridge
   与 sparse legal-swap ranking。
7. 成本 parser 的五预算支持和 mandatory completeability 应在下一精确提交修复，但不能伪装成
   当前运行模型已经具备的能力。

## 当前论文边界

当前仍只能声称：offline full-window pre-backbone hard-RGB sampler、状态转变驱动的边界微簇、
exact-K/max-hole、original-time gather、official-derived TAD backend，以及 heavy-backbone
processed-frame reduction。不能声称 learned sampling 已提高 official mAP、完整端到端成本已下降、
protected bridge 等价于离散换帧 utility、或 TTDI 已实现并验证。

## 2026-07-23 性能影响优先级与触发式修复

1. `selected_axis_index` 把不规则间隔当成等间隔，是最高优先级的潜在 mAP 风险，尤其影响
   边界微簇、低预算、短动作与高 tIoU；但只有 hard selection 质量已优于 uniform 而 mAP 未升时，
   才能把它识别为主因。
2. mandatory group 的 bool union 与无接纳前 completeability 是中高优先级 selector 风险，
   在 K256 以下、多边界和重叠 burst 时最严重。下一版应保留 group/quota/bilateral 身份，按组做
   可补全性检查后再接纳，并把剩余预算交给全局 exact-K/G 填充。
3. protected detector bridge 的 hard forward 正确，但 backward 是局部 RGB 斜率代理。若 G1/G2
   低于 G0 或 legal hard-swap sign/Spearman 不通过，应关闭该 bridge，优先比较真实 hard-forward、
   soft-backward 的 selected-RGB 传输与 train-only legal-swap detector utility ranking。
4. 五预算 cost parser 不改变 mAP，但会直接破坏效率曲线和论文成本声明；应动态支持
   K384/320/256/192/128 及其 G，不需要重训模型。
5. G2 当前是 mixed-policy regularization，不是逐样本 paired counterfactual。若保留“反事实”主张，
   必须让同一视频、同一增广、同一 GT 同时走 learned/uniform 两路；否则只按混合策略正则化表述。
6. official-derived backend 命名不影响数值，但影响复现与论文可信度，必须明确 wrapper、selected-axis
   映射和 optimizer 接口的扩展。

定位顺序冻结为：hard selection 不优于 uniform -> 修 scorer/group-aware decoder；selection 已优但
mAP 不升 -> 测 T1 true-time residual；G1/G2 比 G0 低 -> 修或关闭 detector bridge；T1 仍暴露物理
时间尺度错误 -> 才实现 T2 physical-coordinate head。禁止把这些变量一次性合并。
