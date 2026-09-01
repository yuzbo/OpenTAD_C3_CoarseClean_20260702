---
updated: 2026-07-09
status: active
scope: 吸收 Pro 对“THUMOS14 标注性质改变后 DUCA-TAD 任务是否仍有意义”的严厉判定
out-of-scope: 不记录新的实验数值；不声称任何 sparse detector mAP 或 key-event 结果已经完成
---

# DUCA-TAD Task Validity After THUMOS14 Action-State Review

## 原始记录

- 原始审查文件：`docs/methods/reviews/2026-07-09-duca-task-validity-thumos-keyevent-pro-verdict-raw.txt`
- 原始审查 SHA256：

```text
F3F119245847AFD72842E6504C6D37D47DBBA032864879A8382CDBB808287578
```

- 审查触发问题：THUMOS14 的 GT 可视化显示 Shotput、LongJump、HighJump、PoleVault 等动作标注覆盖完整动作状态，而不是脱手、起跳、过杆等单点关键事件。因此需要重新判断当前 DUCA-TAD 任务是否仍然成立。

## 控制性结论

本次审查必须作为后续论文、PPT、实验和代码命名的控制性边界：

**继续，但必须收缩和改名。DUCA-TAD 只能被定义为 efficient sparse Temporal Action Detection / detector-utility-calibrated temporal acquisition，不能在 THUMOS14 上声称 precise key-event localization。**

这不是项目失败，而是错误叙事失败。THUMOS14 支持的是 interval-level action detection，不支持 release/takeoff/impact timestamp spotting。当前项目仍有意义，但必须从“找关键瞬间”改成：

```text
在严格 observation budget 下，选择 original-time sparse observations，
让 TAD detector 仍能恢复完整动作区间、边界和 proposal ranking。
```

## 任务重新定义

### 当前 DUCA-TAD 属于 sparse TAD

DUCA-TAD 的合法任务是：

```text
dense temporal observations
-> hard selected_positions <= 384
-> original-time sparse observations
-> AdaTAD / ActionFormer style detector
-> temporal segments + class + score
-> Avg mAP and high-IoU mAP@0.6/@0.7
```

该任务关注 detector 在稀疏输入下是否仍能保持：

- 完整动作区间覆盖；
- start/end boundary evidence；
- proposal classification；
- boundary regression；
- IoU/ranking；
- original-time coordinate consistency。

### 当前 DUCA-TAD 不属于 key-event localization

以下任务不应再被写成 THUMOS14 主线：

- key event localization；
- critical moment discovery；
- release / takeoff / impact frame localization；
- precise event spotting；
- state-change timestamp localization。

如果后续真要做 key-event，则需要 SoccerNet Action Spotting、Ego4D PNR、FineGym/FineDiving step transition，或在 THUMOS 田径子集上额外人工标注 event timestamp。

## THUMOS14 能支持的 claim

THUMOS14 可以支持：

1. **strict-budget sparse TAD**
   - detector 只消费 `selected_positions <= 384` 时，是否还能保持 TAD mAP。

2. **action-support prior**
   - C3 / ASFormer 粗分类器输出的 `p_action` 可以被解释为 action-support / action-state prior。
   - 它估计“该 snippet 是否位于动作支持区间中”，不是“该 snippet 是否为关键事件瞬间”。

3. **original-time sparse grid 的必要性**
   - selected positions 必须保留 dense original-time index。
   - detector 若在 selected-axis 上预测 proposal，必须 remap 回 original time。

4. **high-IoU interval localization**
   - `mAP@0.6/@0.7` 可以证明稀疏采样是否保住区间边界质量。
   - 但它不能证明模型找到了脱手/起跳等关键帧。

5. **detector utility calibration**
   - 如果 DUCA-joint 超过 actionness-only、uniform-384、feature-energy-384，才可以说 selector 学到的不只是 foregroundness，而是 detector-useful acquisition。

## THUMOS14 不能支持的 claim

以下表述禁止进入论文主张：

1. **不能证明 key-event localization**
   - THUMOS14 没有 release/takeoff/impact timestamp GT。

2. **不能证明动作内部阶段理解**
   - 粗粒度 action interval 不等于 fine-grained phase supervision。

3. **不能把 p_action 说成 detector utility**
   - 高 AUROC/AUPRC 只说明 action/background support prior 有效。
   - 它不说明该位置对 regression、IoU ranking、NMS 或 mAP 有用。

4. **不能自动证明 online deployability**
   - inference 决策必须无 GT、无 teacher、无 oracle、无 raw prediction cache、无 offline ledger decision。

5. **不能自动证明 raw-frame compute saving**
   - 若 selection 发生在 pre-extracted dense features 之后，只能 claim detector-side temporal observation reduction。

## 对 coarse actionness 的最终吸收判断

coarse actionness 仍然有意义，但必须降级为：

```text
deploy-visible action-support prior
```

它不是：

```text
key-event probability
detector utility
boundary correctness
TAD success proxy
```

它的合理作用：

- 避免 sparse acquisition 把预算浪费在纯背景；
- 覆盖完整动作支持区间；
- 为 boundary/change/scaffold/DUCA utility 提供输入特征；
- 作为 actionness-only baseline，帮助证明 DUCA-joint 的必要性。

它的主要风险：

- 过采样 action interior；
- 忽略 start/end boundary；
- 依赖场景捷径；
- AUROC 高但 sparse detector mAP 不升；
- `mAP@0.5` 保持但 `mAP@0.7` 崩。

因此论文中的核心叙事应改为：

```text
Actionness provides action-support evidence;
DUCA calibrates which support, boundary, transition, and context observations
are useful for the downstream detector under a strict budget.
```

## 数据集策略

### 主线

THUMOS14 继续作为主线，但只用于 efficient sparse TAD：

- Avg mAP；
- mAP@0.5/0.6/0.7；
- action segment coverage；
- boundary coverage；
- short-action recall；
- selected-position geometry；
- compute / latency accounting。

### 压力测试

FineAction 可以作为 fine-grained stress test，用于检查 actionness 是否只是 THUMOS 场景捷径。它不必立即替代 THUMOS14，但如果 claim 涉及 fine-grained generalization，则必须加入。

### 关键事件诊断

SoccerNet Action Spotting、Ego4D PNR、FineGym、FineDiving 或 THUMOS 田径子集人工 key-event timestamp，只能作为另一条 diagnostic / future extension。不要把它们的目标混入 THUMOS14 sparse TAD 主线。

## 最小证据链

### 1. Coarse actionness eval

需要报告：

- frame/snippet action-background AUROC；
- AUPRC；
- Recall@K；
- Precision@K；
- per-class breakdown；
- short-action vs long-action breakdown；
- boundary-near vs interior breakdown。

解释边界：

```text
这些指标只证明 action-support prior，不证明 TAD 成功，不证明边界定位，也不证明关键事件定位。
```

### 2. Selection geometry eval

必须报告：

- budget used；
- action segment touched recall；
- full segment coverage；
- boundary coverage within +/-r；
- short-action recall；
- max hole inside GT；
- p95 hole；
- redundancy；
- uniform similarity / Jaccard。

关键失败解释：

- touched recall 高但 full coverage 低：只是碰到动作，不是覆盖动作。
- full coverage 高但 boundary coverage 低：高 IoU TAD 风险大。
- selection density 集中在中段：actionness-only interior bias。
- p95 hole 过大：破坏 detector temporal geometry。

### 3. Sparse detector eval

最终裁决必须来自 detector：

- dense detector；
- uniform-384；
- random-384，多 seed；
- feature-energy-384；
- actionness-only top-k；
- actionness + boundary/change heuristic；
- oracle-actionness，仅作 upper bound；
- DUCA-online teacher-free；
- DUCA-joint hard-forward ST；
- DUCA without detector utility；
- DUCA selected-axis ablation；
- DUCA original-time grid。

必须看：

- Avg mAP；
- mAP@0.5/0.6/0.7；
- per-class mAP；
- short-action mAP；
- latency / FLOPs / memory；
- actual detector-consumed observations。

### 4. Diagnostic visualization

必须新增 GT-normalized diagnostic：

```text
pre-context: -20% to 0%
early: 0% to 20%
middle: 20% to 80%
late: 80% to 100%
post-context: 100% to 120%
```

绘制：

- mean p_action；
- selection density；
- boundary response；
- delta p_action；
- DUCA utility；
- uniform baseline density。

目标是证明 DUCA 不是只选 actionness peak，而是在 start/mid/end 和 boundary 附近保留 detector-useful support。

## 论文重写规则

### 标题方向

推荐：

```text
DUCA-TAD: Detector-Utility-Calibrated Sparse Temporal Acquisition for Efficient Temporal Action Detection
```

或：

```text
Sparse Temporal Acquisition with Detector-Utility Calibration for High-IoU Temporal Action Detection
```

避免：

```text
Key Event Acquisition
Critical Moment Localization
Precise Event Discovery in THUMOS14
```

### Abstract 必须避免

不要写：

- locates key events；
- discovers critical action moments；
- identifies release/takeoff/impact frames；
- frame-level event localization。

应写：

- selects task-informative temporal observations；
- preserves action support and temporal boundaries；
- maintains high-IoU temporal detection under strict observation budgets；
- uses action-support priors but calibrates selection with detector utility。

### Method 里 coarse actionness 的写法

应写：

```text
The coarse probe estimates an action-support prior, not a detection score.
It is deploy-visible and teacher-free at inference.
It provides candidate support evidence, while DUCA calibrates final hard selection
using boundary/change features and train-only detector utility.
```

必须明确：

```text
actionness != detector utility
actionness != boundary correctness
actionness != key-event probability
```

## Fatal risks

1. 继续把 THUMOS14 写成 key-event localization。
2. actionness AUROC 高，但 sparse mAP 不升。
3. `mAP@0.5` 保持，但 `mAP@0.7` 大幅下降。
4. uniform-384 太强，DUCA 无法稳定超过。
5. online claim 被 ledger/teacher/cache 污染。
6. selected-axis 坐标污染导致 sparse detector 结果不可解释。
7. pre-extracted feature selection 被误写为 raw-frame compute saving。
8. dynamic budget 只是 eval-time sweep，而不是模型内部 policy。
9. FineAction / fine-grained stress test 崩溃时仍过度宣称 generalization。

## Go / No-Go

本次吸收后的最终决策：

```text
HOLD -> conditional GO for efficient sparse TAD.
NO-GO for key-event localization on THUMOS14.
```

继续条件：

1. DUCA-online / DUCA-joint 在同预算下稳定不弱于 uniform-384，最好 Avg mAP 和 `mAP@0.7` 都赢。
2. 如果 Avg mAP 只小幅赢，必须在 short actions、boundary coverage 或 latency 上有明确优势。
3. actionness-only 可以输，但 DUCA-joint 必须赢。
4. selected-axis 负控必须差，original-time sparse grid 必须有必要性。
5. inference 必须 teacher-free、ledger-free、GT-free。
6. geometry diagnostics 必须证明不是只选动作内部 peak。
7. FineAction 或其他 fine-grained stress diagnostic 若暂不做，论文必须明确不 claim fine-grained generalization。

## 后续执行约束

从本记录起，后续所有论文、PPT 和实验汇报默认遵守：

- `THUMOS14 = interval-level sparse TAD benchmark`；
- `p_action = action-support prior`；
- `DUCA = detector-facing sparse temporal acquisition plugin`；
- `mAP@0.6/@0.7 = high-IoU interval localization evidence`；
- `key-event localization = out-of-scope unless using timestamp-level event dataset or extra key-event labels`。

允许继续推进 DUCA-TAD，但必须放弃“THUMOS14 关键事件定位”叙事。
