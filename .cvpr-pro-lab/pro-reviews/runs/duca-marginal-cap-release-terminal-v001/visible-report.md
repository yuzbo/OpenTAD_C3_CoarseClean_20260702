# PIVOT

**Nonce：`DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831`**

**终止当前 DUCA-Marginal-v1 的“三档预算、逐窗口加性反事实损失、逐视频等总 observation 重分配”机制；不终止 DUCA 的任务感知动态计算方向。论文问题应转向：在重叠滑窗时序动作检测中，动态预算是否必须优化视频级联合预测集，而不能把窗口级检测损失边际简单相加。**

---

## 1. 终态证据核验

GitHub 分支确实指向精确提交 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`，其直接父提交是 `f67d96fdf68a295eaa7f678f3dfc125530828889`。该提交只修改 cap-release runner 和聚焦测试，没有改变 `dynamic_budget.py`、H65 producer、模型、数据、损失、NMS 或评估器。

代码审查与终态 JSON 一致：

* runner 要求干净提交、核对原 capped 结果及密封 K256/K384/K512 产物，重新计算 Fixed-H65、capped 和 released 三个结果，并且不能覆盖原 `probe_result.json`；
* 强门严格为 Avg-mAP 增益至少 `+0.8` 个百分点且 mAP@0.7 增益至少 `+1.0` 个百分点；只有两者同时通过才运行 10,000 次整视频配对 bootstrap；否则终止当前机制；
* allocator 在每个视频内最大化三个预算档位的**加性窗口效用**，严格保持实际 observation 总数等于全 K384 的目标，并使用确定性 tie-break；
* 新测试验证了解除上限能引入更多等成本 K256/K512 转移，同时不改变默认 0.5 路径和原结果文件。

终态比较如下：

| 分配                       |    Avg-mAP |    mAP@0.7 |         相对 Fixed-H65-384 |
| ------------------------ | ---------: | ---------: | -----------------------: |
| Fixed-H65-384            | 88.131197% | 76.270583% |                        — |
| 50% capped loss-oracle   | 88.856786% | 76.999587% | +0.725589 / +0.729004 pp |
| cap-released loss-oracle | 88.558507% | 76.720863% | +0.427310 / +0.450280 pp |

解除上限后：

* 相对 capped allocation 下降 `-0.298279/-0.278724` 个百分点；
* 分配从 `11/102/11` 变成 `17/90/17`；
* 只新增了 6 组等成本转移，即 12 个变化窗口，集中在 5 个视频；
* 总实际 observation 始终为 `47110`，预算误差为零；
* Fixed 与 capped 的全部复现误差为 `0.0`；
* 两项强门均失败，因此不运行 bootstrap 是冻结规则的正确执行，而不是遗漏。

附件中的旧研究状态、库存边界和历史注册表只承担背景、谱系和规则说明，不能覆盖本轮 `d2fad7c0...` 与终态 JSON。PJST-D1 的统计缺口也与本轮 Marginal-v1 终态无关，不重新开启。      项目与角色规则要求把真实负结果、实现成功和论文主张分别处理，并只下达一项能改变科学结论的任务。  

---

## 2. 当前负结果最支持什么

### 直接支持

**第一，当前 Marginal-v1 没有达到预先要求的强 oracle headroom。**

这里的“oracle”只意味着分配器看到了真实的逐窗口 K256/K384/K512 反事实检测损失。即使给予这种不可部署的特权信息，解除改变窗口上限后仍只有 `+0.427/+0.450` 个百分点，明显低于冻结强门。因此，不应再训练一个 predictor 去逼近同一个不足的目标。

**第二，逐窗口反事实检测损失不是最终 mAP 的充分加性效用。**

cap release 扩大了分配可行集。allocator 在扩大的可行集中获得了更高的内部加性效用，却得到更低的最终 mAP。因而被直接否定的不是“优化没找到解”，而是以下隐含假设：

> 在保持逐视频 observation 总成本不变时，独立窗口检测损失的改善可以相加，并能单调转化为视频级 TAD mAP 改善。

**第三，50% cap 更像一个偶然有效的 trust region，而不是得到验证的科学机制。**

它限制了弱效用信号被过度应用，因而在这批数据上优于 unrestricted allocation。但不能据此把 `0.5` 写成有普适意义的最优比例，也不能继续做 0.4、0.6、0.7 等 cap 搜索。

### 仍不能支持

本结果不能推出：

* 所有动态预算或动态计算方法均无效；
* K256/K512 档位本身无用；
* 当前三档可行空间完全没有任何 metric-level headroom；
* coverage、物理时间、边界保护或语义选帧假设无效；
* learned allocator 必然失败；
* 当前差异具有统计显著性；
* 在 official validation/test 上会出现相同结果；
* 已获得任何端到端效率结论。

尤其不能把训练侧 40-video holdout 上的 `88.xx%` 与 official validation 的 `65.xx/68.xx%` 直接比较。

---

## 3. 为什么解除上限后反而变差

### 最可能的主因：效用目标与最终评估的非加性错配

Marginal-v1 对每个窗口使用：

* K256 相对效用：`-(loss256 - loss384)`；
* K384 相对效用：`0`；
* K512 相对效用：`loss384 - loss512`。

随后把这些量在视频内相加并求精确预算最优解。最终 mAP 却是在所有窗口预测合并、滑窗去重、soft-NMS、类别内全局置信度排序和 AP 聚合之后得到。窗口损失和最终 mAP 之间不存在可保证的线性关系。

因此，以下现象完全可能同时发生：

1. 某个 K512 窗口的局部训练损失下降；
2. 与其重叠窗口产生更多重复或竞争 proposal；
3. soft-NMS 改变保留分数与排序；
4. 对应的 K256 窗口丢失另一段动作或背景抑制信息；
5. 最终 AP 下降。

released allocation 在 mAP@0.3 上甚至低于 Fixed-H65-384，而在 mAP@0.5–0.7 上仍保持小幅正值。这更符合 proposal 覆盖、分类排序或窗口去重受到干扰，而不是单纯的高 tIoU 边界问题。

### 次要放大因素：三档预算和精确成本约束过粗

K384 到 K256/K512 是一次 `±128` observation 的离散跃迁。严格等总成本通常要求一个 downgrade 与一个 upgrade 成对出现，而不是允许连续微调。

此外，`int(window_count × max_changed_fraction)` 与成对转移存在奇偶阶跃：

* 3 个窗口、50% cap 时最多允许改变 1 个窗口，但等成本转移至少需要改变 2 个，因此实际上不允许任何转移；
* cap release 后会突然允许一整对 K256/K512 改变。

这解释了为什么解除上限会离散地新增整组转移，但它本身不能解释为什么 mAP 下降；下降仍需要效用误排序或窗口间交互。

### 仍然存活的替代解释：可利用空间本来就很小

capped 和 released 都只显示亚百分点级开发集增益，因此当前固定 H65 priority sequence 加三档预算的真实可用空间可能有限。40-video holdout 也可能放大偶然波动。

但现有实验没有直接优化最终 mAP，也没有搜索联合预测集的 metric-level allocation，所以尚不能把“空间不足”定为唯一原因。

**综合判断：最强证据指向“窗口级加性效用错配”，其中重叠窗口干扰、NMS 和 AP 聚合是主要表现；粗预算档位与等成本组合约束是放大器；固有 headroom 较小是仍未排除的替代解释。**

---

## 4. 修订后的科学问题与路线

新的核心问题应为：

> 在固定逐视频重型 observation 总成本下，滑窗时序动作检测的预算分配是否必须依据多个重叠窗口联合产生的预测集效用，而不是独立窗口的检测损失边际？

这一路线的潜在机制不是再增加一个 utility head，而是把分配单位从“独立窗口”提升为“视频内相互重叠的窗口集合”，显式处理：

* 不同窗口之间的 proposal 重复与互补；
* 边界和动作区域的跨窗口覆盖；
* NMS 前后的置信度竞争；
* downgrade 与 upgrade 的联合后果。

这一问题具有论文价值，因为当前终态揭示了一个反直觉现象：**一个拥有真实窗口反事实损失、内部目标更优且预算完全守恒的 oracle，可能因忽略联合预测集结构而得到更差的最终 mAP。**

但现阶段不授权实现视频级 predictor、图网络、集合模型或新的训练损失。首先必须证明现有 sealed predictions 中确实存在可被联合选择恢复的 metric-level 空间。

---

## 5. 唯一当前任务单

### 任务名称

**cap-release 差分邻域的联合 mAP 穷举诊断**

### 科学问题

解除上限新增的 6 组等成本转移之所以降低 mAP，究竟是因为：

1. 每一组新增转移本身就被窗口损失误排序；
2. 某些转移单独有利，但组合后因窗口重叠、NMS 或 AP 聚合发生负交互；
3. 两者兼有；
4. 即使联合选择，局部 action space 也没有足够强的 headroom。

### 权威代码与允许修改面

* 基座：`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`
* 新分支：`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`
* 只允许修改：

  * `tools/bata/run_duca_marginal_frozen_h65_probe.py`
  * `tests/test_duca_marginal_budget.py`
* `opentad/models/duca/dynamic_budget.py` 必须保持逐字不变。
* 不新增模型类、配置族、启动框架、通用搜索器或 provenance 系统。

### 最小实现

在现有 runner 中增加一个只读分析 stage：

1. 读取并验证原 capped result、cap-release terminal result 和三个密封 producer；
2. 从 capped 与 released allocation 的差异自动导出新增变化：

   * 当前预期为 5 个视频；
   * 12 个新增变化窗口；
   * 6 组净等成本转移；
3. 不得为 `video_validation_0000419` 人为指定两组配对。该视频有两个新增 downgrade 和两个新增 upgrade，必须枚举所有满足实际 observation 成本守恒的平衡子集；
4. 在每个视频内只允许使用 capped→released 差分窗口，并按真实 `actual_cost` 检查成本，不得假设所有窗口都正好是 ±128；
5. 对各视频合法状态做笛卡尔积。按当前终态应得到 **96 个唯一、确定性、逐视频等成本的联合状态**；实现必须由数据推导状态数，再断言当前输入确实为 96，不能只硬编码 96；
6. 对全部 96 个状态使用同一 40-video holdout、同一密封预测、同一 sliding-window NMS 和同一评估器计算：

   * Avg-mAP；
   * mAP@0.3/0.4/0.5/0.6/0.7；
   * 相对 Fixed、capped、released 的差值；
7. 单独报告：

   * 每个最小合法单转移的指标变化；
   * 所有可组合单转移的联合变化；
   * 联合变化减去单项变化之和的交互残差；
   * 96 个状态中的最佳点；
8. 写入独立结果文件，不得覆盖 `probe_result.json` 或 `oracle_cap_release_result.json`。

### 公平性与必须复现的控制

正式枚举前必须再次满足：

* Fixed-H65-384 全部指标复现误差不超过 `1e-6 pp`；
* capped oracle 全部指标复现误差不超过 `1e-6 pp`；
* released oracle 全部指标复现误差不超过 `1e-6 pp`；
* 每个枚举状态逐视频实际 observation 成本均与 K384 目标完全一致；
* 全局实际 observation 始终为 `47110`；
* 只允许 12 个 capped→released 差分窗口变化；
* detector/Scout forward 次数为零；
* 模型训练、utility-head 拟合、梯度计算和 official test 消耗均为零。

这是纯预测重组分析，因此不需要也不允许增加模型形状、梯度或物理坐标新功能。

### 主要指标与最便宜 falsifier

主要指标继续使用冻结的：

* 相对 Fixed-H65-384 的 Avg-mAP 增益；
* 相对 Fixed-H65-384 的 mAP@0.7 增益。

最便宜 falsifier 就是 96 个 CPU-only 官方评估器调用。无需 GPU、无需前向、无需训练。

### 继续与停止门

**联合效用问题获得继续研究资格，当且仅当：**

* 96 个状态中至少一个同时达到：

  * Avg-mAP 相对 Fixed-H65-384 `≥ +0.8 pp`；
  * mAP@0.7 相对 Fixed-H65-384 `≥ +1.0 pp`。

这只能证明当前局部 action space 存在 metric-level 开发集 headroom，并证明加性损失排序遗漏了它。结果必须返回 Pro；不得自动训练任何 predictor。

**若没有任何状态同时通过两项强门：**

* 终止“用视频级联合效用修复本次 cap-release 差分”的路线；
* 不再对当前 H65 priority、K256/K384/K512 和等逐视频成本合同做 cap sweep、档位搜索或 utility predictor 训练；
* broader DUCA、coverage 或其他动态计算问题保持未决，但必须另行重新定义科学问题，不能由 Codex自动选择。

### 根因分类规则

* **单项误排序为主：** 所有最小合法新增转移都不能同时改善 capped 的两项主要指标；
* **窗口交互为主：** 至少两个最小转移单独都改善两项主要指标，但联合后至少一项收益发生符号反转或低于两者中的较好单项；
* **混合原因：** 其余模式。

这只是确定性机制诊断，不是统计总体结论。

### 禁止项

本任务禁止：

* 任何 detector、Scout 或 utility predictor 训练；
* detector/Scout forward；
* official validation/test；
* bootstrap；
* 修改 H65 priority sequence；
* 改变预算档位、split、checkpoint、annotation、类别映射、NMS 或 evaluator；
* 根据 96 个状态事后设计新的阈值；
* 扩展到任意全空间搜索；
* 新增视频级网络、图模型、集合模型或可学习交互项；
* 把最佳 96-state 点写成可部署或可发表性能。

之所以不运行 bootstrap，是因为最佳状态将在同一 40-video development holdout 的 96 个候选中被选择，选择后的区间不能被冒充为预登记确认性统计。

### Builder → Critic → Evaluator

**Builder**

* 在上述两个文件内完成最小实现和聚焦测试；
* 证明状态枚举唯一、完整、确定性且逐视频成本严格守恒；
* 截止：`2026-08-31T10:30:00+08:00`。

**独立 Critic**

只审查：

* 96-state 枚举是否完整；
* `video_validation_0000419` 是否避免任意配对；
* 是否只改变 12 个差分窗口；
* 是否保持逐视频实际成本、NMS、评估器与数据边界；
* 是否存在任何 forward、训练或 official-test 路径；
* 是否把同集 metric oracle 错写成论文证据。

通过即停止，不审查代码风格或扩展性。

截止：`2026-08-31T12:00:00+08:00`。

**独立 Evaluator**

* 在 Critic 通过的唯一干净提交上运行一个 CPU Slurm 作业；
* 先复现 Fixed/capped/released，再运行全部 96 个状态；
* 返回完整状态表、最佳点、单转移效应、交互残差和上述冻结门的机械判定；
* 结果根使用
  `/data/run01/sczc063/yuzibo/duca_marginal_cap_release_neighborhood_<implementation-short-sha>_20260831`；
* 最终截止：`2026-08-31T18:00:00+08:00`。

唯一客观 blocker 是任一密封输入缺失或 SHA 不一致。出现该 blocker 时停止，不重新生成 producer、不重跑 detector，也不建立第二条工程修复链。

---

## 6. 论文可写入边界

### 当前可以准确记录

本结果可以作为**开发阶段负结果、机制淘汰证据或相关论文的附录消融**：

> 在冻结 H65、同一 40-video training-side holdout 和严格相同的逐视频实际 observation 总成本下，基于真实逐窗口反事实检测损失的 K256/K384/K512 重分配在 50% 改变上限时相对固定 K384 提高 Avg-mAP/mAP@0.7 `+0.726/+0.729` 个百分点；解除上限后增益下降为 `+0.427/+0.450` 个百分点，并比 capped allocation 低 `0.298/0.279` 个百分点。该诊断否定了“扩大当前加性窗口损失分配的自由度会单调改善最终 TAD mAP”的假设，因此该 allocator 被停止。

若后续论文采用视频级联合效用机制，这一结果可以作为其问题动机；否则只作为内部路线淘汰记录。

### 当前仍不可声称

不得声称：

* DUCA-Marginal-v1 是有效方法；
* capped allocation 的增益具有统计显著性；
* released allocation 证明动态预算无效；
* 当前结果来自 official validation/test；
* 当前结果能推广到其他 seed、数据集、检测器或预算档位；
* learned allocator 能达到任何 oracle 数值；
* 已证明窗口交互是唯一因果来源；
* 已获得性能—成本 Pareto 改善；
* coverage、物理时间或整个 DUCA 方向已被否定。

**本轮仅授权上述一项联合 mAP 穷举诊断。当前 Marginal-v1 机制已达到终态，不再允许通过训练 predictor、调整 cap 或增加预算档位进行恢复。**
