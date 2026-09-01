# DUCA capped→released 联合 mAP 邻域终态：请独立裁决下一项科学任务

**Nonce：`DUCA-MARGINAL-CAP-RELEASE-NEIGHBORHOOD-TERMINAL-ADJUDICATION-v001-20260831`**

你是本项目的独立科学负责人、路线设计者和论文主张审查者。Codex 只负责在你冻结的任务内完成最小实现、
独立代码审查、正式评估和证据回传。请不要接受 Codex 预设的替代路线；基于本轮完整代码与原始终态，独立
给出且只给出一个 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`，并决定是否还有一项值得执行的可证伪任务。

## 1. 最新公开代码真值

本轮讨论必须以以下已经推送的 GitHub 实现为代码真值：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
- 联合邻域 runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
- 保持不变的 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>

精确提交的父提交是 `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`。本轮只修改 runner 与聚焦测试；
`opentad/models/duca/dynamic_budget.py` 相对父提交的 Git blob SHA-1 保持
`268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`，没有修改 allocator、模型、预测、数据、损失、NMS、评估器
或门槛。附件包含完整终态 JSON；runner、allocator 和聚焦测试由上面的精确提交永久 GitHub 链接绑定。
请完整读取附件并审阅链接中的最新代码，不以项目中更旧 Source 或旧聊天覆盖它们。

## 2. 上一轮由你冻结的问题与停止规则

此前 cap release 扩大了三档 K256/K384/K512 allocator 的可行集，并改善其独立窗口加性损失目标，却使
最终 mAP 下降。你据此停止当前加性 Marginal-v1，并冻结唯一后继：只在 capped 与 released allocation
不同的窗口上枚举所有逐视频等实际 observation 成本的联合状态，以判断视频级联合预测集合效用是否仍能
修复这一本次差分。

冻结规则是：

1. 从密封分配与逐窗口实际成本自动导出差分，不硬编码配对；
2. 当前数据应得到 12 个差分窗口、6 个净转移组和 96 个唯一联合状态；
3. 所有状态保持每个视频的实际 observation 总成本和全局成本 `47110`；
4. 复用相同密封预测、Soft-NMS 和评估器；不执行模型前向、训练、official test 或 bootstrap；
5. 只有至少一个状态同时达到相对 Fixed-H65-384 的 `ΔAvg-mAP >= +0.8 pp` 与
   `ΔmAP@0.7 >= +1.0 pp` 才允许继续，而且仍须返回你裁决；否则停止用视频级联合效用修复本次差分邻域；
6. 从同一开发集 96 个状态中选择出的最佳状态不可部署、不可作为确认性或论文主结果，也不得选择后补做
   bootstrap。

## 3. 实现、审查和执行事实

- 实现自动导出 5 个差分视频、12 个差分窗口、6 个净转移组、8 个最小合法转移和 96 个唯一状态。
- `video_validation_0000419` 存在两种完整等成本分解；实现保留四个合法最小配对和两种完整分解，没有任意
  固定一种配对。
- N16R4 exact clean snapshot 上：16 项聚焦测试通过；23 项既有回归测试通过；独立 Critic 返回
  `TERMINATOR_STATIC_PASS`。
- 唯一 Evaluator 是 Slurm Job `1262121`，终态 `COMPLETED 0:0`，共调用 CPU evaluator 96 次。
- 集群只有强制申请 GPU 的分区，所以作业申请 1 张调度占位 GPU；运行时清空 `CUDA_VISIBLE_DEVICES` 并
  固定 `--device cpu`，没有使用 GPU 计算。
- stderr 只有无关的 `requests` 依赖版本警告；没有 evaluator 失败。
- fixed、50% capped、cap-released 三个历史结果复现最大误差均为 `0.0 pp`。
- 没有 detector/Scout forward、模型训练、utility-head fitting、梯度计算、official test 或 bootstrap。

原始终态：

- 本地：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`
- 远端：`/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831/oracle_cap_release_neighborhood_result.json`
- SHA-256：`a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`

## 4. 终态结果

参考结果：

| 状态 | Avg-mAP | mAP@0.7 | 相对 Fixed-H65-384 |
|---|---:|---:|---:|
| Fixed-H65-384 | 88.131197% | 76.270583% | — |
| 50% capped oracle | 88.856786% | 76.999587% | +0.725589 / +0.729004 pp |
| cap-released oracle | 88.558507% | 76.720863% | +0.427310 / +0.450280 pp |

96-state 联合邻域：

- 通过状态数量：`0/96`；
- 按联合门 margin 最优 `state_014`：Avg-mAP `88.685169%`，mAP@0.7 `77.203817%`，相对 fixed
  `+0.553972/+0.933234 pp`；联合门 margin 为 `-0.246028 pp`；
- Avg-mAP 最优 `state_020`：`88.864187%/76.749874%`，相对 fixed `+0.732990/+0.479291 pp`；
- mAP@0.7 最优 `state_001`：`88.679866%/77.204123%`，相对 fixed `+0.548669/+0.933539 pp`；
- 8 个最小合法转移中，没有一个同时提高 Avg-mAP 和 mAP@0.7；
- 按上一轮冻结定义，`interaction_witness_count=0`，root-cause classification 为
  `single-item misranking primary`；
- 终态状态为 `JOINT_NEIGHBORHOOD_GATE_FAILED_STOP_DIFFERENCE_REPAIR`，没有运行 bootstrap。

最小转移的相对 capped 变化（Avg-mAP / mAP@0.7，百分点）为：

1. video 0059：`-0.176920 / +0.204536`；
2. video 0206：`+0.004503 / -0.249650`；
3. video 0267：`-0.106811 / -0.151798`；
4. video 0419 的四种合法最小配对：分别为
   `-0.029264/-0.014718`、`-0.028038/-0.013730`、`-0.006230/-0.015282`、
   `-0.006529/-0.019337`；
5. video 0905：`-0.062820 / -0.132893`。

## 5. 请独立回答的科学问题

请不要因为需要“给下一项任务”而强行延续路线。先判断本轮证据是否足以停止更大的研究族；若不足，再选择
唯一最有信息增益的后继。必须明确区分直接证据、最合理解释和仍未排除的替代解释。

1. 本轮是否忠实执行了你冻结的联合邻域任务？是否存在会改变结论的实现、预算、评估或选择偏差？
2. `0/96` 联合门通过、Avg 最优与 @0.7 最优明显分离、且所有最小转移都不能同时改善两项指标，最直接支持
   什么失败机制？`single-item misranking primary` 的冻结分类应如何转写成论文可理解的科学结论？
3. 这是否只停止“本次 capped→released 差分邻域的联合效用修复”，还是足以停止更广的三档预算、H65
   priority sequence、或任务感知动态计算路线？请准确划定停止边界。
4. 在不重跑已停止诊断、不对 96 状态事后调参、不访问 official test、不扩大工程系统的前提下，是否还存在
   一项能真正改变论文结论的低成本实验？若没有，请明确 `STOP` 并说明论文应如何保存这组负结果。
5. 若选择继续、修订或转向，请只冻结一项当前任务，并给出：
   - 一句话科学问题与机制；
   - 为什么它不是对已失败思路的换名重试；
   - 最小 Builder 修改表面和禁止修改内容；
   - 最便宜的决定性 falsifier、数据划分、公平比较、指标、阈值和停止规则；
   - 独立 Critic 要核验的少数实质风险；
   - Evaluator 的 PRE_RUN 与唯一正式运行；
   - 可发表主张边界、失败后保留的负结果；
   - `next_owner / next_action / dependency / absolute_deadline`。

## 6. 输出要求

输出第一行必须是且只能是 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP` 之一。随后必须原样写出 nonce。
先给终态科学解释与停止边界，再给唯一下一任务（若有）。不要提供多路线菜单、超参数网格、泛化工作流平台、
复杂合同代码或多个并行实验。不要把本轮开发集 oracle 结果写成部署效果、官方验证/测试结果、统计显著性或
论文主性能。Pro 对科学路线负责；Codex 不应替你预选机制。
