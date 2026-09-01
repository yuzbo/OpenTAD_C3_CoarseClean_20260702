# 唯一裁决：**REVISE**

**Nonce：`DUCA-MARGINAL-ORACLE-GRAY-ZONE-ADJUDICATION-v001-20260831`**

当前灰区诊断**准入**，但只准入为训练侧留出集合上的机制诊断；**现在不允许训练 utility head**。唯一下一任务是：不增加预算档位、不运行新的 detector forward、不训练任何模型，只解除“每视频最多改变 `floor(0.5N)` 个窗口”这一项约束，使用现有密封 K256/K384/K512 产物执行一次只读 oracle 证伪。

---

## 一、证据准入裁决

### 1. 双提交来源足以准入，但必须按双阶段来源表述

当前分支确实指向精确提交 `f67d96fdf68a295eaa7f678f3dfc125530828889`，其直接父提交是 producer 使用的 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。

从 `f87555f7...` 到 `f67d96fd...` 只改动：

* `tools/bata/run_duca_marginal_frozen_h65_probe.py`：把换行文本 block-list 确定性转换成 evaluator 接受的 JSON 数组；
* `tests/test_duca_marginal_budget.py`：增加对应回归测试。

提交没有修改模型前向、三档选择、反事实损失、分配器、配置、检查点、数据、NMS、评估指标或科学阈值。

因此，严格审查提出的“producer commit 与 summary commit 不完全一致”是一个**必须显式披露的来源边界**，但不是足以否定结果的科学 blocker。正确表述是：

> K384/K256/K512 producer 产物生成于 `f87555f7...`；`f67d96fd...` 在模型、配置、数据和 producer 语义不变的条件下，只重新执行身份绑定和汇总评估。

不得把三个 producer 重新标成由 `f67d96fd...` 生成，也不得把整条计算链简写为“全部运行于 f67d96fd”。

### 2. 不需要重跑 producer

最小必要修复本应是：

1. 在 `f67d96fd...` 上重新通过 PRE_RUN；
2. 读取原密封 producer 产物；
3. 用修复后的 JSON block-list 重新执行 `summarize`；
4. 保留 producer receipts 的 `f87555f7...` 来源和产物哈希。

Job `1262098` 已经完成这四项。重新执行三个重型 producer 不会增加科学信息，反而只会引入无意义的重复计算。因此不要求重算、不建设额外来源框架，也不增加新的哈希系统。

### 3. 准入级别

这份结果的证据级别固定为：

* **实现与来源：准入；**
* **训练侧留出集合机制诊断：准入；**
* **正式论文主结果：不准入；**
* **官方 validation/test 性能结论：不准入；**
* **显著性或总体效应结论：不准入。**

附件中的代码库存边界、角色规则、Evaluator 原始证据、项目规则、论文优先流程、旧问询、材料说明、旧研究状态和历史版本表只用于约束证据层级与执行方式；它们不覆盖本轮 `f67d96fd...` 和原始 JSON 的最新事实。        

---

## 二、灰区结果的科学解释

### 1. 已观察到的事实

在固定的 40 个训练侧 utility-holdout 视频、124 个窗口上：

* `Fixed-H65-384`：Avg-mAP `88.131197%`，mAP@0.7 `76.270583%`；
* `Oracle-Reallocate-384`：Avg-mAP `88.856786%`，mAP@0.7 `76.999587%`；
* 增益为 Avg-mAP `+0.725589` 个百分点、mAP@0.7 `+0.729004` 个百分点；
* 相比强门槛，分别还差 `0.074411` 和 `0.270996` 个百分点；
* 结果同时高于无 headroom 边界，因此既不是强通过，也不是近零结果；
* 124 个窗口中，102 个保持 K384，11 个降为 K256，11 个升为 K512；实际改变 22 个窗口，占 `17.74%`；
* 发生非基线分配的 9 个视频中，有 5 个视频恰好达到其 `floor(0.5N)` 改变窗口上限；
* 每视频实际 observation 总预算精确不变，整体实际成本为 `47110`；
* detector、Scout、utility targets 与执行长度合同均通过。

分配器当前确实按每视频精确实际成本运行，并把 `max_changed_fraction` 纳入动态规划；汇总入口则把该参数硬编码为 `0.5`。

### 2. 可以作出的推断

最有证据支持的解释是：

> **冻结 H65 优先序列内部存在局部、有限的跨窗口预算异质性，但在当前三档预算和 50% 改变比例约束下，机制特定 oracle 的收益尚不足以支持学习阶段。**

具体而言：

* 结果明显高于无 headroom 边界，因而不能说“不同窗口完全没有不同的边际计算价值”。
* 只有 22 个窗口发生改变，说明有效信号是**稀疏和局部的**，不是大多数窗口普遍需要动态预算。
* 五个发生重分配的视频达到改变比例上限，使“50% 上限截断了一部分可用转移”成为有原始证据依据的解释。
* 但“达到上限”不等于证明解除上限一定增加 mAP；在无约束重算前，它仍只是一个待证伪解释。
* 当前 oracle 使用真实冻结 detector loss 构造边际效用，而不是直接全局优化 mAP。因此它是当前 utility-head 机制的**操作性上界诊断**，不是所有可能动态预算方法的数学 mAP 上界。
* 由于已实现 utility head 的目标就是近似这组 loss-based utility，一个尚未通过强门槛的真实效用 oracle 不足以支持继续承担预测误差、分配误差和有限样本误差。

### 3. 当前仍未知的事项

现有结果不能判断：

* `+0.726/+0.729` 是否由少量视频驱动，或其整视频配对区间是否跨零；
* 解除 50% 改变上限后是否能够通过原强门槛；
* K256/K384/K512 三档量化本身是否是主要限制；
* Scout 特征能否预测 downgrade penalty 和 upgrade gain；
* 学习分配能恢复多少 oracle 增益；
* official validation/test 上是否存在收益；
* 实际端到端延迟、吞吐量或 FLOPs 是否改善。

特别是，**当前数据不支持把三档预算称为失败根因**。验证更多档位需要新的 producer forward，并会演化成事后预算搜索，因此本轮禁止执行。

---

## 三、utility head 裁决

**不允许训练。**

当前代码只在 Avg-mAP 同时达到 `+0.8`、mAP@0.7 同时达到 `+1.0` 后进入 `_fit_utility_head`；灰区正确地停在训练之前。

现在直接训练 utility head 会产生三个问题：

1. 违背冻结的强 headroom 前置门槛；
2. 在机制特定 oracle 尚不足时，用预测器误差进一步压缩本已有限的收益；
3. 把“是否存在足够预算空间”和“预算空间是否可预测”两个问题混为一次实验。

即使下一项只读证伪通过，也只返回 Pro 再次裁决；本任务内仍不得训练 utility head。

---

## 四、唯一冻结任务

### 任务名称

**解除 50% 改变窗口上限的只读 oracle 证伪**

### 科学问题

当前灰区主要来自跨窗口边际效用本身不足，还是来自每视频最多改变 `floor(0.5N)` 个窗口这一人为约束截断了有价值的预算转移？

### 唯一干预

把 oracle 分配的：

```text
max_changed_fraction = 0.5
```

改为：

```text
max_changed_fraction = 1.0
```

除此之外任何科学变量均不得变化。

### 可证伪预测

若 50% 上限确实是主要限制，那么在：

* 相同 K256/K384/K512 产物；
* 相同真实反事实 utility；
* 相同每视频实际 observation 总预算；
* 相同 tie-break；
* 相同 NMS 和 evaluator；

条件下，解除上限应使 oracle 同时达到原冻结强门槛：

* `ΔAvg-mAP ≥ +0.8` 个百分点；
* `ΔmAP@0.7 ≥ +1.0` 个百分点。

### 相反解释

若解除上限后仍未同时通过两项强门槛，则当前收益上限主要不是由 50% 约束造成，而是来自：

* 冻结 H65 priority sequence 中可交换预算的真实价值有限；
* 三档 loss-based utility 与最终 mAP 的对齐有限；
* 或当前三档机制本身的表达能力有限。

此时终止 `DUCA-Marginal-v1` 当前机制，不追加预算档位、utility-head 训练或比例搜索。

### 任务性质

这是**只读诊断**，不是正式论文实验：

* 不训练任何模型；
* 不访问 official test；
* 不运行新的 detector/Scout forward；
* 不生成 K320 或其他预算；
* 不需要 GPU；
* 只消费已密封的三个 producer 产物。

---

## 五、实现边界

### 权威基座

* Repository：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* Base revision：`f67d96fdf68a295eaa7f678f3dfc125530828889`
* Base branch：`feature/duca-marginal-budget-v1-20260830`
* 新分支：`feature/duca-marginal-cap-release-falsifier-v1-20260831`

### 允许修改的文件与符号

只允许修改两个文件：

1. `tools/bata/run_duca_marginal_frozen_h65_probe.py`

   * 给 `_allocate_rows_by_video` 增加显式 `max_changed_fraction` 参数，默认值必须保持 `0.5`；
   * 原 `run_summary_stage` 的现有结果路径必须保持不变；
   * 增加一个独立只读诊断入口，使用 `max_changed_fraction=1.0`；
   * 输出单独的 `oracle_cap_release_result.json`，不得覆盖原始 `probe_result.json`；
   * 复用 `_official_holdout_metrics`、现有 NMS 和现有 sealed predictions。

2. `tests/test_duca_marginal_budget.py`

   * 验证默认 `0.5` 路径不变；
   * 验证 `1.0` 在合成四窗口样例中允许两组正效用预算转移；
   * 验证实际成本严格等于 all-K384 目标；
   * 验证 tie-break 仍是“最大 utility → 较少改变 → 固定档位字典序”；
   * 验证原 `probe_result.json` 不被写回。

现有 allocator 已经支持任意合法 `max_changed_fraction`，因此不得修改 `allocate_video_budgets_exact` 的数学实现。现有聚焦测试也已经覆盖短窗口别名、真实 observation 成本、部分末包 padding、detached targets 和 evaluator block-list 修复。

### 明确禁止修改

不得修改：

* `opentad/models/duca/dynamic_budget.py`；
* `opentad/models/duca/counterfactual_utility.py`；
* H65 selector、priority sequence 或嵌套集合；
* detector、Scout、loss normalizer、VideoMAE、projection、head；
* detector grid 与物理时间映射；
* K256/K384/K512 三档定义；
* 160/40 split、seed 3407；
* annotation、class map、checkpoint、预训练权重；
* NMS、官方 evaluator、指标；
* 强 headroom 与无 headroom 数值门槛；
* producer artifacts 及其 receipts；
* utility-head 结构、训练轮数或优化器。

不得新建通用 bootstrap 框架、调度系统、来源系统、配置族或兼容层。

---

## 六、评估协议与停止条件

### 阶段 1：原结果复现门

用原 sealed artifacts 重新执行 `max_changed_fraction=0.5` 的汇总，必须满足：

* Fixed 与 capped-oracle 所有 mAP 值相对原 `probe_result.json` 误差不超过 `1e-6` 个百分点；
* K256/K384/K512 计数保持 `11/102/11`；
* 所有 artifact SHA 与原 receipts 一致；
* 每视频预算误差为零。

不满足时属于实现或产物绑定错误，只做一次最小修复，不产生科学解释。

### 阶段 2：解除上限的点估计

在 `max_changed_fraction=1.0` 下报告：

* Fixed 与 cap-release oracle 的 mAP@0.3–0.7；
* Avg-mAP；
* 相对 Fixed 的两项主增益；
* 相对原 capped oracle 的增量；
* K256/K384/K512 窗口数；
* 改变窗口数、改变视频数；
* capped 方案中达到上限的视频数；
* cap-release 后的每视频实际成本误差；
* 原始与新分配的确定性摘要。

### 阶段 3：条件式配对区间

只有 cap-release **点估计同时通过原强门槛**时，才执行：

* 40 个视频的整视频配对 bootstrap；
* 10,000 次；
* seed `3407`；
* Fixed 与 cap-release oracle 使用完全相同的重采样索引；
* 每次重新计算 Avg-mAP 和 mAP@0.7；
* 报告 2.5%/97.5% percentile 区间。

该区间不修改原强门槛，只作为继续投入的证据准入条件。

### 唯一判定规则

| 结果                          | 科学处理                                                           |
| --------------------------- | -------------------------------------------------------------- |
| cap-release 任一主点指标未通过原强门槛   | 终止当前 `DUCA-Marginal-v1`；不运行 bootstrap，不训练 utility head，不测试更多档位 |
| 两项点指标通过，但任一 95% 配对区间下界不高于 0 | 终止当前机制；证据不足以承担 predictor 误差                                    |
| 两项点指标通过，且两项区间下界均高于 0        | 返回 Pro 进行新的 utility-head 任务冻结；本任务仍不得训练                         |
| 实际成本、数据身份、原结果复现或实现门失败       | 仅视为实现失败；最小修复后重做同一只读诊断                                          |

---

## 七、Builder → 独立 Critic → Evaluator 交接

### Builder

交付：

* 从 `f67d96fd...` 派生的单一 clean commit；
* 只包含上述两个文件；
* 原 capped 路径不变；
* 新 cap-release 入口及独立结果文件；
* 聚焦测试通过。

截止：**2026-08-31 18:00:00 +08:00**

### 独立 Critic

只审查：

* 是否真的只改变 `max_changed_fraction`；
* 原 `0.5` 路径是否保持默认及结果兼容；
* 是否没有新 detector forward、utility-head 训练或 official-test 访问；
* 每视频实际成本、短窗口 alias、tie-break 是否保持；
* 是否没有覆盖或重标 producer artifacts。

不得因代码风格、额外日志或通用完备性制造修复循环。

截止：**2026-08-31 22:00:00 +08:00**

### Evaluator

在 Critic 通过的同一 clean commit 上：

1. 验证 sealed artifact SHA；
2. 精确复现 capped 结果；
3. 执行 cap-release 只读汇总；
4. 仅在点门通过时运行配对 bootstrap；
5. 返回原始 JSON、指标、分配和科学判定，不做 utility-head 训练。

截止：**2026-09-01 12:00:00 +08:00**

---

## 八、论文主张边界

### 当前可以写入论文或研究记录的事实

可以写：

1. 在冻结 H65 Scout、epoch-59 EMA detector 和训练侧 40-video 留出集合上，三档真实反事实效用重分配在相同实际 observation 总预算下获得：

   * Avg-mAP `+0.725589` 个百分点；
   * mAP@0.7 `+0.729004` 个百分点。
2. 该重分配只改变 22/124 个窗口，说明边际预算价值集中在少数窗口。
3. 实现保持 K384 selection/prediction 一致、detector/Scout frozen、targets detached，并满足精确实际成本。
4. 结果处于预注册灰区，没有触发 utility-head 训练。
5. producer 来源为 `f87555f7...`，汇总修复和最终结果绑定 `f67d96fd...`。
6. 当前 50% 改变比例约束在五个发生重分配的视频上达到上限，因此值得进行一次有界的约束释放心证。

### 当前不可写入的主张

不可声称：

* 动态预算已在 THUMOS14 官方 validation/test 上提高性能；
* `+0.726/+0.729` 具有统计显著性或稳定总体效应；
* utility head 可以预测真实边际效用；
* learned allocation 有效；
* 50% 改变上限已被证明是收益不足的根因；
* 三档预算已被证明过粗；
* 当前结果是所有动态预算方法的上界；
* 已获得端到端 FLOPs、延迟、吞吐量或能耗收益；
* DUCA-Marginal 已形成可投稿主方法；
* 训练侧 `88.xx%` 可以与官方 validation/test 指标直接比较。

### 当前贡献状态

**本轮尚未形成可投稿贡献。**

它形成的是一个实现严格、来源可解释、但收益不足且统计未闭合的机制诊断。其价值在于约束后续研究：当前 frozen-H65 三档边际重分配并非完全没有空间，但在原 50% 改变比例下尚不足以承担学习与部署阶段。只有唯一 cap-release 证伪通过后，才有理由重新讨论 predictor；否则该机制应作为受控负结果保存。

---

```text
next_owner: Codex Builder
next_action: 在 f67d96fdf68a295eaa7f678f3dfc125530828889 上实现并执行 max_changed_fraction=1.0 的只读 oracle 约束释放心证；原 0.5 路径必须精确复现；禁止 utility-head 训练和新 producer forward
dependency: 三个已密封 producer artifacts、对应 f87555f7 receipts、f67d96fd PRE_RUN/source bindings、原 160/40 split、冻结 evaluator/NMS/checkpoint/data
expected_return_at: 2026-09-01T12:00:00+08:00
```
