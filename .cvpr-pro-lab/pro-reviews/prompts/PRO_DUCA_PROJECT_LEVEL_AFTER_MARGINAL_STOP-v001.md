# DUCA 项目级后继裁决：在 Marginal-v1 终止后决定继续、转向或停止

**Nonce：`DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831`**

你是 DUCA 的独立科学负责人、机制设计者和论文主张审查者。Codex 只负责执行你冻结的最小实现、独立代码
审查、正式实验评估和证据回传。当前没有已授权的后继机制；请不要接受 Codex 预设路线，也不要因为流程需要
“下一项任务”而强行延续研究。

这是一轮新的项目级科学裁决，不是对已经终止的 DUCA-Marginal-v1 的追问、恢复或调参。请基于最新公开代码、
附带的完整阶段报告、终态负结果记录和上一轮 Pro 终态，独立判断 DUCA 是否仍有一条值得投稿的新机制路线。

## 1. 最新公开实现与 GitHub 代码真值

本轮必须以以下已推送对象为最新实现，不能以本地路径、旧 Project Source 或旧聊天中的分支名覆盖：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 最新只读证据分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
- 联合邻域诊断入口：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
- 当前动态预算 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>
- H65 30+60 正式基座提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>

`46812fac...` 的本地 clean HEAD、远端 upstream 和远端分支头已重新核对为一致。它只为终态只读诊断修改
runner 与测试；allocator、模型和训练代码没有随诊断改变。附件中的阶段报告和实验页给出所有历史代码身份与
证据边界。

## 2. 仍然有效的论文目标

研究对象是离线时序动作检测（Temporal Action Detection, TAD）。长期目标是在重型 VideoMAE 路径之前，利用
低成本动作与状态证据，为不同时间位置分配真实不同的高分辨率视觉计算；在相同或更低的真实端到端计算下，
尽量保护或提高 THUMOS14 上的平均检测精度，尤其是高时间交并比下的边界定位。

任何候选都必须满足：

1. 真正改变 VideoMAE 执行的 observation/clip 数，而不是 padding 后的名义稀疏；
2. 使用官方数据划分、注释、类别映射、检测器评估和相同 NMS；
3. 与公平的 dense 或匹配计算基线比较，分别报告性能、真实成本和不确定性；
4. validation/test 标签、teacher 或事后挑选不得参与分配决策；
5. 先用最便宜的决定性 falsifier 验证机制，再考虑完整训练；
6. 保留负结果，避免把协调系统、合同代码或防御性框架当作科研贡献。

## 3. 已经取得的主要证据

请完整读取附件；以下只给出项目级必要摘要：

- 共享官方 dense AdaTAD Avg-mAP 为 `68.73`，论文公开锚点为 `69.03`；当前 H65 30+60 单种子参考为
  Avg-mAP `65.13`、mAP@0.7 `43.31`。两者不构成同一计算预算下的完整论文比较，当前也没有可发表的
  端到端成本曲线。
- 把 H65 的 30+60 压缩为 20+40 或 30+30，并调整第二阶段学习率衰减，均未恢复性能。它终止的是已测试的
  压缩日程，不否定 H65 的低成本语义证据。
- 连续高分辨率片段采样的完整训练明显下降；联合训练没有恢复。这条采样单元已停止。
- TrueTime/PJST-D1 没有显示平均性能收益；配对区间又因统计收尾路径错误缺失，因此不能升级成总体负效应。
- 固定 K=384 原生 tubelet 的任务状态 coreset 相比匹配 uniform 下降 `1.32` Avg-mAP 和 `1.89` mAP@0.7；
  结构化预测和配对区间没有封存。细粒度 coreset 已停止。
- DUCA-Coverage-v1 在训练前机制门失败：集合变化和覆盖增益不足，最大时间空洞反而恶化；没有启动完整训练。
- DUCA-Marginal-v1 冻结 H65、检测器和 K256/K384/K512 真实反事实。50% capped 真实效用 oracle 相对固定
  K384 为 `+0.726/+0.729` 个百分点；解除 cap 后只有 `+0.427/+0.450`，未达到 `+0.8/+1.0` 门。
- 最终 96-state 逐视频等成本联合邻域中通过状态为 `0/96`，联合门最优仅 `+0.554/+0.933`；8 个最小合法
  转移没有一个同时改善 Avg-mAP 与 mAP@0.7。上一轮 Pro 因此终止现有加性 Marginal-v1，并明确该结论
  不能外推为 H65 priority sequence、三档预算或任务感知动态计算总体无效。

目前没有正在运行或已授权的 DUCA Builder、Critic、PRE_RUN、训练、Evaluator 或 official-test 任务。

## 4. 已停止且不得换名重试的内容

不要把以下内容作为新任务：

- 重跑或扩展 Marginal-v1 的 96 状态，改变 `+0.8/+1.0` 门，选择折中状态，更换配对或 tie-break；
- 为事后最佳状态补 bootstrap，训练旧 utility head，继续 cap/预算档位/加性损失小修；
- 恢复 fixed-K coreset 的分数调权、Coverage-v1 的 `M/sigma/K/M` 调参或最大空洞修补；
- 重做连续片段、PJST-D1、60 轮压缩学习率扫描，或在旧负结果上追加无决定性的种子；
- 直接访问 official test 来寻找路线；
- 用新术语包装同一机制，或者设计多个并行候选菜单。

## 5. 请独立完成的项目级裁决

第一行必须且只能是 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。本轮的 `STOP` 表示：在当前任务、证据和
可接受资源边界下停止 DUCA 项目级方法创新，而不只是停止 Marginal-v1。

请依次回答：

1. 当前证据是否已经足以结束 DUCA 的任务感知稀疏重型计算研究？分别说明直接证据、最合理的跨路线失败
   机制，以及仍未排除但也尚无正证据的可能性。
2. 现有结果能否形成可投稿贡献？若不能，缺少的是新机制、严格公平的计算—性能比较、统计证据，还是更根本
   的问题定义；不要把工程完成度当成科学贡献。
3. 如果你选择 `CONTINUE`、`REVISE` 或 `PIVOT`，只提出一个真正新的、最有信息增益的机制假设。它必须解释
   为什么能够越过已经观察到的失败，而不是 Marginal-v1、Coverage、coreset、连续片段或 PJST 的换名重试。
4. 对这唯一机制冻结：一句话科学问题、因果机制、可证伪预测、最便宜的决定性实验、公平对照、真实计算
   定义、数据划分、指标、阈值、停止规则和论文主张边界。
5. 只给完成该 falsifier 所需的最小 Builder 代码表面、必须保持不变的组件、独立 Critic 的少数实质检查、
   Evaluator 的 PRE_RUN 与唯一正式运行。不要生成平台、工作流框架、复杂合同或超参数网格。
6. 明确 `next_owner / next_action / dependency / absolute_deadline`。如果没有一项实验能以合理成本改变论文
   结论，请选择项目级 `STOP`，说明应如何归档已有负结果，而不是虚构下一任务。

## 6. 输出约束

紧接第一行原样输出 nonce。只给一个科学裁决和至多一个当前任务，不给路线菜单。明确区分开发集 oracle、
official validation/test、点估计、置信区间、运行成本和因果解释。不得把未运行的机制写成有效，也不得把当前
负结果外推到证据没有覆盖的研究族。Pro 对问题和路线负责；Codex 只能执行你明确冻结的任务。

