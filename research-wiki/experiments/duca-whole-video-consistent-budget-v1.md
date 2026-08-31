---
type: experiment
status: tested
updated: 2026-08-31
project: DUCA
---

# DUCA 整视频一致预算的跨视频转移 oracle

## 科学问题

在全局实际 observation 成本不高于固定 K384 时，如果同一个视频的全部重叠窗口使用同一预算档位，只在
不同视频之间转移计算，能否避免逐窗口混合预算造成的 proposal 质量与置信度不一致，并同时改善 Avg-mAP
与 mAP@0.7？

这是 Pro 在终止加性 DUCA-Marginal-v1 后独立冻结的新机制假设。它不是旧 allocator 的恢复：旧机制在
一个视频内部按窗口损失转移预算；新机制保持视频内预算一致，直接枚举一次跨视频 donor–recipient 转移。

## 代码基座与最新公开真值

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 只读父分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
- 父提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
- 实现分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- 独立 runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
- 未修改的三档预算分配器：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

新分支只允许新增独立 runner、聚焦测试和必要时的薄 CPU Slurm 入口。不得修改旧 Marginal runner、
`opentad/models/duca/dynamic_budget.py`、selector、acquisition、detector、训练器、配置或既有预测。

父提交 `c27d77aa...` 只新增上述 runner 与测试。候选清单由 `sample_id`、`video_id` 和密封的
`budget_accounting.actual_cost` 生成并落盘，此后才读取终态指标、annotation 或调用评估器。新 runner
没有调用旧 Marginal allocator。首次 PRE_RUN Job `1262147` 发现 runner 对密封 proposal 行做了额外
字典序排序；这会在分数并列时改变既有 Soft-NMS 的确定性输入顺序，使 fixed/capped/released 锚点无法
按 `1e-6` 个百分点复现。该问题发生在候选性能计算前，是证据重放顺序错误，不是机制负结果。

最小修正提交 `33e4ed13...` 只保留密封 producer 的原始 proposal 顺序，并增加相应回归测试；没有修改
候选定义、成本、预测值、Soft-NMS、评估器、三档预算分配器或科学门槛。28 项聚焦测试通过，全新独立
Critic 对精确干净提交返回 `PASS`。修正提交已部署为 N16R4 干净快照
`/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831`。

修正后的 PRE_RUN Job `1262161` 已 `COMPLETED 0:0` 并返回 `PRE_RUN_PASS`：40 个视频、124 个窗口，
固定 K384 实际成本 `47110`；在全部 `40×39=1560` 个有序对中有 704 个合法候选、1330 个实际改变
预算的候选；候选生成阶段未读取标签、GT 或指标，fixed/capped/released 三个锚点复现误差均为
`0.0` 个百分点。回执与候选清单分别位于
`/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_pre_run_receipt.json`
和同目录 `whole_video_candidate_manifest.json`，SHA-256 分别为
`734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3` 与
`c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`。

首次正式 Evaluator Job `1262162` 只运行冻结的 `--stage evaluate` 并复用相同密封预测、Soft-NMS 与评估器，
但在完成 `500/704` 个候选后因计算节点 `g0022` 下线而被 Slurm 标为 `NODE_FAIL`。运行器没有异常回执，
也没有生成终态结果，因此它不是性能证据。完全相同的唯一基础设施恢复 Job `1262190` 随后从同一干净
快照、脚本、候选清单和输出目录完成 `704/704` 个候选，Slurm 状态为 `COMPLETED 0:0`。没有创建第三份作业。

## 终态结果

固定 K384 在 40 个训练侧 controller holdout 视频上的 Avg-mAP/mAP@0.7 为 `88.1312%/76.2706%`，实际
observation 成本为 `47110`。全部 704 个合法 donor-recipient 候选均完成相同 Soft-NMS 与评估器计算，
共调用评估器 705 次（固定臂一次、候选 704 次），但没有候选同时达到预先登记的两个性能门槛：

- Avg-mAP 最优候选 `video_validation_0000158=>video_validation_0000173` 的变化为
  `+0.6942/-0.0436` 个百分点（Avg-mAP/mAP@0.7），实际成本 `46982`；
- mAP@0.7 最优候选 `video_validation_0000490=>video_validation_0000173` 的变化为
  `-0.2359/+0.4970` 个百分点，实际成本 `46854`；
- 按联合门余量最优的候选 `video_validation_0000419=>video_validation_0000173` 的变化为
  `+0.1474/+0.4898` 个百分点，实际成本 `45830`，离联合门最近的余量仍为 `-0.6526` 个百分点。

因此，通过候选数为零。该结果触发此前由 Pro 冻结的停止条件：在当前 THUMOS14 训练侧 holdout、H65
priority sequence、K256/K384/K512 三档真实 observation 动作空间和本轮资源边界内，不再扩大搜索、组合更多
视频、修改预算档位或降低门槛。该结果随后已经返回 Pro；下文记录其独立终态裁决，Codex 未从负结果自行选择
替代路线。

终态文件为
`/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`，
SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`。固定臂复现误差在全部六个
指标上均为 `0.0` 个百分点；没有生成 failure receipt。

## 冻结干预

只使用 40 个训练侧 controller holdout 视频、124 个窗口，以及密封 K256/K384/K512 窗口预测和真实成本：

1. 全 K384 是固定基线；
2. 对每个不同视频的有序对 `(d,u)`，donor `d` 的全部窗口请求 K256；recipient `u` 的全部窗口请求
   K512；其余视频全部请求 K384；
3. donor 和 recipient 都必须至少有一个窗口的实际成本不同于 K384；
4. 短窗口继续按 `min(valid_observations,K)` 计费和折叠；
5. 只保留全局实际成本不高于 `47110` 的候选；
6. 候选集合必须在读取标签、GT 或指标前完整生成；标签只用于随后寻找开发集特权 oracle 最优状态。

最多有 `40×39=1560` 个有序候选。所有状态复用相同密封预测、坐标逆映射、Soft-NMS、annotation、类别
映射和评估器；不执行 detector/Scout forward、训练、梯度、bootstrap 或 official test。

## 通过与停止规则

只有至少一个合法候选相对 fixed K384 同时满足：

- `ΔAvg-mAP >= +0.8` 个百分点；
- `ΔmAP@0.7 >= +1.0` 个百分点；
- 全局实际 observation 成本 `<=47110`；

才说明该开发集 whole-video action space 有足够 headroom，并返回 Pro 裁决可预测性。最优候选先最大化
`min(ΔAvg-mAP-0.8, ΔmAP@0.7-1.0)`，再选择成本更低者，最后按 donor、recipient 视频 ID 字典序。

若身份复现误差超过 `1e-6` 个百分点、输入身份不一致、没有合法候选或通过候选为零，则停止本 falsifier。
前述前三种确定性问题只允许一次不改变机制和门槛的最小修复。若通过候选为零，不增加第三个视频、不组合
多个 transfer、不降低门槛、不改变预算档位、不补 bootstrap、不训练控制器、不访问 official test；在当前
THUMOS14、H65 priority sequence、三档真实 observation 动作空间和资源边界下项目级停止 DUCA 方法创新。

## 证据边界

这是从 704 个合法开发集状态中事后选择的 privileged oracle 负结果。它没有执行 detector/Scout forward、
训练、梯度、bootstrap 或 official validation/test，也没有确认性统计区间；因此不能证明可部署控制器、真实
端到端速度收益或 DUCA 优于 dense AdaTAD。它只否定当前冻结边界内的整视频单 donor-recipient 三档转移
是否具有足够的开发集联合性能 headroom，不能外推为所有动态计算、所有预算空间或所有低成本 Scout 无效。

## Pro 终态裁决

Fresh exact DUCA Project 对话完整绑定了最新公开代码：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 实际远端分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
- 未修改的三档 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

Pro 裁决为 `STOP`。在当前 THUMOS14 训练侧 holdout、冻结 H65 detector/priority sequence、密封三档预测、
真实 observation 成本与现有资源边界内，不再开发在窗口或视频之间转移 K256/K384/K512 预算的方法；不再
扩大状态空间、改门、训练 controller、补 bootstrap 或访问 official validation/test。该停止不否定所有动态
计算、Scout、预算空间、budget-conditioned training、内部 token/层级条件计算或其他数据集。

最强诚实结论是：704 个合法整视频单 donor-recipient 状态中没有状态同时达到预登记联合门，且 Avg-mAP 与
高 tIoU 的最优状态分离。最强未验证解释是当前 H65 优先序列与仅在 K384 下训练的 detector 缺少跨预算兼容、
单调且边界敏感的表示。当前没有新的 Builder、Critic、Evaluator 或 Slurm 任务；该分支只作为负证据保留。
