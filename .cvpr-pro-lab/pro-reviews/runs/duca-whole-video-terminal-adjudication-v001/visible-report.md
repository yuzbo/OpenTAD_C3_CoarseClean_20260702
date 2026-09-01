Nonce: `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`

# 1. 唯一科学裁决

## **STOP**

**零通过结果触发此前冻结的项目级停止边界。**

停止的准确范围是：

> 在当前 THUMOS14 训练侧 controller holdout、冻结 H65 检测器与优先序列、密封的 K256/K384/K512 窗口预测、真实 observation 成本口径和现有资源范围内，停止继续开发基于“在视频或窗口之间转移三档输入 observation 预算”的 DUCA 方法。

因此，不再进行：

* Marginal-v1 的恢复、改写或重新训练；
* capped、released、96-state 或 whole-video 候选空间的扩展；
* 增加第三个视频、多次 transfer、组合更多 donor–recipient 对；
* 改变 K256/K384/K512 档位或事后降低 `+0.8/+1.0` 门槛；
* 训练 utility predictor、controller 或其他对当前 oracle 状态进行模仿的预测头；
* 为这条路线补 bootstrap、访问 official validation/test，或把更大的开发集搜索包装成新机制。

停止边界**不**扩展为以下一般性结论：

* 所有动态计算都无效；
* 所有低成本 Scout 都无效；
* 所有预算空间都缺少 headroom；
* 所有逐视频方法都无效；
* 预算条件训练、token/layer 级动态计算或其他数据集上的动态计算都无效。

这些问题没有被本实验检验。

# 2. 实现与证据是否足以执行停止条件

## 结论：足够，并且没有发现会阻止科学裁决的实现偏差

公开远端分支当前确实指向精确提交 `33e4ed137c33eef07f0452b44506a6993bdf7535`。

该提交相对父提交只修改了两个文件：whole-video runner 和对应聚焦测试；核心修改是删除对密封 proposal 行的额外排序，恢复 producer 原始顺序，并增加顺序保持回归测试。它没有修改预测、候选定义、成本、Soft-NMS、评估器或预算机制。([GitHub][1])

公开 runner 与冻结协议一致：

1. 它从密封 K384、K256、K512 产物合并预测，并保留 K384 producer 的原始行顺序；同时核对三个产物具有相同 sample 集、40 个 holdout 视频和124个窗口。
2. 每个候选严格定义为：一个 donor 视频全部请求 K256，一个不同的 recipient 视频全部请求 K512，其余全部请求 K384。合法性要求两个被改变视频都存在真实非基线执行，并且总实际成本不超过 `47110`。
3. 它完整枚举 `40×39=1560` 个有序视频对，并在候选清单中保留全部候选及其合法性；候选生成只依赖 `sample_id`、`video_id` 和密封 actual-cost accounting。
4. runner 先生成并写出完整 candidate manifest，之后才加载终态指标、annotation 和 evaluator。聚焦测试也显式检查了这一顺序。
5. 正式阶段重新生成候选空间并与 PRE_RUN manifest 的 canonical hash 比较，重新评估 fixed K384 anchor，要求所有指标复现误差不超过 `1e-6` 个百分点，随后逐一评估全部合法候选。
6. 联合通过条件在代码中固定为 `ΔAvg-mAP ≥ +0.8`、`ΔmAP@0.7 ≥ +1.0` 且成本不超过 `47110`；零通过时直接记录 whole-video 项目级停止状态。
7. 聚焦测试检查了完整有序对、真实短窗口成本折叠、视频内统一预算、未调用旧 Marginal allocator、密封 proposal 顺序以及确定性联合排序。

`dynamic_budget.py` 在邻域提交 `46812fac…` 和当前提交 `33e4ed13…` 上具有相同 blob SHA `268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`，确认分配器没有被本轮修改。

物理时间恢复和评估路径也不是由 whole-video runner 重新发明。密封 counterfactual producer 明确保存 selected-axis 到原始真实时间坐标的映射并要求预测在 NMS 前逆映射；终态 runner只是按照密封顺序组合这些已经生成的预测。  `_official_holdout_metrics` 对组合后的 proposals 使用配置中不变的 Soft-NMS，并调用同一 THUMOS14 evaluator。

因此，首次正式作业的节点故障应继续排除在科学证据之外；使用同一快照、manifest、预测和 evaluator 完成 `704/704` 的唯一恢复作业可以作为本轮终态执行。根据你指定为权威的终态 artifact、SHA、`0.0` anchor 复现误差和 `705` 次 evaluator 调用，没有证据表明零通过来自候选遗漏、成本错误、proposal 重排、评估漂移或旧 allocator 介入。

我没有在此界面重新读取集群本地 JSON 的原始字节；其 SHA 和数值按本轮明确指令作为权威终态证据接受。公开代码与所报告的结果结构完全相容。

旧的 2026-08-28 材料包也已完整读取。其 prompt、旧研究状态、库存边界、PJST 原始证据、项目规则、角色规则、通用科研 skill、历史实现注册表和 README 只用于解释历史与证据边界；其中“仍待路线裁决”的内容已被本轮 2026-08-31 终态覆盖。        

# 3. 机制层面的失败诊断

## 3.1 被证据支持的事实

这不是“预算变化完全没有影响”。实际结果显示，预算转移能够改变检测结果，但变化没有形成足够大且方向一致的联合收益：

* 最佳 Avg-mAP 候选达到 `+0.694215`，但 mAP@0.7 为 `-0.043632`；
* 最佳 mAP@0.7 候选达到 `+0.496998`，但 Avg-mAP 为 `-0.235922`；
* 联合门排序最优候选只有 `+0.147383/+0.489786`，其联合门余量仍为 `-0.652617`。

这支持三个机制结论：

**第一，逐窗口预算混合不是失败的充分解释。**
本实验已经把同一视频的全部窗口统一到一个档位，消除了“同一视频内不同窗口预算导致 proposal 分数或质量不一致”这一主要候选解释，但联合 headroom 仍未出现。

**第二，当前三档转移的主要问题不是 allocator 没有找到正确的视频，而是候选动作所产生的效用不一致。**
提升 Avg-mAP 的状态和提升高 tIoU 定位的状态不是同一状态；计算转移有时改善较宽松阈值下的总体检测，却没有同步保护精确边界，反之亦然。

**第三，当前 action space 缺少达到预登记论文价值门槛的联合效用。**
这一结论适用于已完整枚举的 whole-video 单 donor–recipient 状态，以及此前 capped、released 和 96-state 差分邻域。它不是从一个失败策略推断出来，而是 oracle 直接读取开发集指标后仍无法找到通过状态。

因此，被否定的机制命题是：

> 仅把 H65 同一嵌套优先序列上的 K256/K384/K512 observation，在窗口之间或视频之间重新分配，就足以在不增加实际 observation 成本时产生可利用的 Avg-mAP 与高 tIoU 联合 headroom。

## 3.2 仍未被检验的解释

本实验没有检验：

* 多个 donor、多个 recipient 的任意全局组合；
* K256/K384/K512 之外的预算或连续预算；
* 重新训练后的 budget-conditioned detector；
* 不以 H65 同一 priority sequence 构造的 K256/K512；
* 改变重型编码器内部 token、层数、特征刷新频率或空间分辨率的动态计算；
* 其他数据集、检测器或 Scout；
* population-level 稳定性。

但这些未知项不能成为继续扩大当前搜索的理由。尤其是多个 transfer 的更大组合空间，仍只是对同一密封预测进行更强的开发集事后优化；它会增加 oracle 选择自由度，却没有引入新的机制或可部署预测信号。

## 3.3 最强的替代解释

最强的未检验解释是：

> **当前 H65 priority sequence 与只在 K384 下训练的检测器，没有形成跨预算兼容、单调且边界敏感的表示。**

K256 是从同一优先序列中删除 observations，K512 是向同一优先序列尾部增加 observations；但检测器是在 K384 分布下训练的。因而：

* K512 新增 observations 不一定包含检测器能利用的新增边界证据；
* K256 删除的 observations 可能改变 proposal 置信度和边界回归校准；
* 不同档位产生的收益和损失未必可加，也未必在 Avg-mAP 与 mAP@0.7 上同向。

whole-video 一致预算可以消除视频内部的档位混合，却不能修复这种**表示与训练分布不兼容**。这比“allocator 还不够聪明”更符合完整负结果。

这一解释仍然是解释，不是已证实结论。验证它需要新的训练机制或新的计算动作空间；那已经越过当前冻结边界，不能作为 Marginal-v1 的补救实验。

# 4. 最强诚实论文主张

当前最强且可核验的表述是：

> 在40个 THUMOS14 训练侧 controller-holdout 视频、冻结 H65 检测器、密封的嵌套 K256/K384/K512 预测和不高于 fixed K384 实际 observation 成本的条件下，704个合法整视频单 donor–recipient 预算转移状态中，没有状态同时达到预登记的 `+0.8` Avg-mAP 和 `+1.0` mAP@0.7 headroom。结合此前 capped、released 和96-state 邻域结果，当前三档 frozen-detector 预算转移动作空间缺少预登记水平的联合开发集 headroom。

还可以进一步写明：

* whole-video 一致预算没有挽救逐窗口 Marginal-v1 的失败；
* 最优 Avg-mAP 与最优高 tIoU 状态存在明显权衡；
* 当前停止是 action-space headroom 停止，而不是 predictor 训练失败。

## 仍然禁止的主张

不得声称：

* DUCA 或动态预算普遍无效；
* Scout 无法预测预算效用；
* 当前结果具有总体显著性或 population-level 置信区间；
* 当前 oracle 是可部署策略；
* 已获得端到端速度、显存、能耗或吞吐收益；
* 已在 official validation/test 上证明负结果；
* DUCA 优于 dense AdaTAD、保持 dense 性能，或获得性能—成本联合优势；
* 任意多视频组合也不可能通过；
* budget-conditioned 训练或其他动态计算动作空间也会失败。

# 5. 可发表性裁决

该结果具有良好的**内部科学终止价值**：候选集合预先生成、有限空间被完整枚举、anchor 精确复现、停止门预先冻结，而且实现没有明显混杂。它足以结束当前研究路线。

但它目前**不能作为一篇 CVPR 方法论文的独立主要结果**，原因是：

* 只使用训练侧 development holdout；
* oracle 事后读取标签和指标；
* 没有 official validation/test；
* 没有 population uncertainty；
* 没有可部署 controller；
* 没有端到端成本测量；
* 没有跨数据集或跨检测器复现；
* H65 本身仍明显低于共享 dense 参考。

因此，本轮负结果的发表地位应定为：

> **可以作为未来其他主论文中的补充失败分析；在当前项目中主要作为内部终态证据。**

若未来要将其发展为独立的“动态时序采集局限性”论文，需要预先设计跨检测器、跨数据集和多个预注册 action space 的系统研究。那将是一个新的研究项目，不是当前 DUCA 的下一实验。

# 6. 是否还存在应立即启动的全新科学任务

## 当前没有

概念上仍可提出与当前路线不同的问题，例如在不删除 temporal observations 的情况下，对 VideoMAE 内部的 token、层或特征刷新进行条件计算。但当前证据只说明输入 observation 转移动作空间失败，并没有提供足够证据表明某个新的内部计算机制具有较高成功概率。

立即选择并实现另一个机制，会把本次明确的停止边界变成无限连续的路线救援。它需要：

* 新的科学问题；
* 新的 action space；
* 新的 matched control；
* 新的实际计算口径；
* 很可能还需要新的完整训练。

在没有便宜、预注册的机制 headroom 证据之前，其预期信息增益不足以合理化新的 Builder、Critic、Evaluator 周期。

## 重新开放研究所需的证据

未来只有在以下条件出现时，才应以一份全新的项目级科学问询重新开放：

> 一个明确位于当前 H65/K256-K384-K512 observation-transfer 边界之外的新机制，先在不使用 official validation/test 的训练侧独立开发划分上证明：其 oracle action space 在匹配实际计算下能够同时超过预先冻结的 Avg-mAP 与高 tIoU headroom 门，并且这种 headroom 不依赖事后扩张候选集合或改变门槛。

在此证据出现之前，不应让 Codex自行实现候选机制，也不应把历史 ChronoTransport、UVT、Fovea、Query-Bridge 或其他分支自动恢复为主线。

# 7. 当前任务

**没有新的科学实验任务。**

Builder、Critic 和 Evaluator 在当前 DUCA 路线上均无剩余工作：

* Builder 不再修改模型、selector、allocator、runner 或配置；
* Critic 不再为同一机制发起新的审查循环；
* Evaluator 不再运行候选、bootstrap、训练、official validation 或 official test。

唯一剩余动作是由 Coordinator 完成一次简短研究归档：

1. 将本裁决、准确停止边界、最强替代解释和论文主张边界写入 decision history、whole-video 实验页和研究日志；
2. 绑定公开分支、精确提交、终态 JSON 路径与 SHA；
3. 把 Marginal-v1、cap-release、96-state neighborhood 和 whole-video 分支标为只读负证据；
4. 不提交新的 Slurm 作业，不新建候选分支。

`next_owner`: DUCA Coordinator
`next_action`: 归档本次项目级 STOP、冻结准确停止范围并确认无新的 Builder、Critic 或 Evaluator 工作被调度
`dependency`: 无；公开代码身份、终态 artifact、候选枚举和停止门均已完成
`expected_return_at`: `2026-08-31T18:00:00+08:00`

`DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`

[1]: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535 "Fix sealed proposal replay order · yuzbo/OpenTAD_C3_CoarseClean_20260702@33e4ed1 · GitHub"
