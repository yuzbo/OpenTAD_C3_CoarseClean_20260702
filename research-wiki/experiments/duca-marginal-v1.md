---
type: experiment
status: empirically_supported
updated: 2026-08-31
project: DUCA
---

# DUCA-Marginal-v1：窗口级加性边际效用的终态负结果

## 科学问题

本实验检验：在冻结 H65 时序优先序列、epoch-59 指数移动平均检测器和 K256/K384/K512 三档真实重型
observation 预算后，逐窗口反事实检测损失能否作为加性效用，在每个视频实际 observation 总成本严格不变时，
把计算从低价值窗口转移到高价值窗口，并同时改善 Avg-mAP 与高时间交并比定位。

40 个训练侧 utility holdout 视频只从 utility-head fitting 中留出，并不是 H65 detector 的未见测试集。本实验
从未访问 official test，因此所有百分数都属于开发集机制诊断，不能与 official validation/test 主表直接比较。

## 实现与证据身份

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 最终只读证据分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831>
- 最终精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d>
- 父提交：`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`
- allocator Git blob：`268c26cf41ae8a0d33c5a1b849ebff2adf0b388e`
- runner：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tools/bata/run_duca_marginal_frozen_h65_probe.py>
- 未修改的 allocator：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/opentad/models/duca/dynamic_budget.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/46812facc8773d9b4a9c21833cbe397c8aaa5a2d/tests/test_duca_marginal_budget.py>
- 唯一联合邻域 Evaluator：Slurm Job `1262121`，`COMPLETED 0:0`
- 原始终态：`.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`
- 原始终态 SHA-256：`a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`

最终提交只修改 probe runner 与聚焦测试；allocator、模型、预测生成、损失、数据、Soft-NMS 和评估器均未
改变。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 通过。正式作业完成 96 次 CPU evaluator
调用；集群分区虽要求申请一张 GPU 作为调度资源，但作业清空 CUDA 可见性并固定 CPU。没有 detector/Scout
forward、模型训练、utility-head fitting、梯度、official test 或 bootstrap。

## 结果

固定 K384 为 Avg-mAP `88.131197%`、mAP@0.7 `76.270583%`。50% 改变窗口上限的真实效用 oracle 为
`88.856786%/76.999587%`，相对固定为 `+0.725589/+0.729004` 个百分点；解除上限后降为
`88.558507%/76.720863%`，相对固定仅 `+0.427310/+0.450280`。三项参考结果的复现误差均为 `0.0`
个百分点，全局实际 observation 成本始终为 `47110`。

最终诊断从 capped/released 差分自动导出 5 个视频、12 个窗口、6 个净转移组、8 个最小合法转移和 96 个
逐视频等成本联合状态，并保留 `video_validation_0000419` 的全部四种最小配对与两种完整分解。结果为：

- 联合继续门通过状态：`0/96`；
- 联合门最优 `state_014`：相对固定 `+0.553972/+0.933234` 个百分点；
- Avg-mAP 最优 `state_020`：`+0.732990/+0.479291`；
- mAP@0.7 最优 `state_001`：`+0.548669/+0.933539`；
- 8 个最小等成本转移中，没有一个同时改善 Avg-mAP 与 mAP@0.7。

冻结继续门为 `+0.8/+1.0` 个百分点，因而本实验按预注册规则停止。没有对同一开发集事后最佳状态运行
bootstrap；这类 bootstrap 也不能把未达到实用效应门槛的点估计变成达到门槛的确认性效应。

## Pro 终态裁决与停止边界

精确 DUCA Project 的 Pro 终态报告保存于
`.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`，裁决为
`STOP`。Pro 认为直接支持的科研表述是：

> 窗口级加性反事实检测损失不是视频级联合检测效用的充分排序统计量。失败已经出现在最小等成本预算转移
> 层面：没有一个最小转移能够同时改善平均检测性能和高时间交并比定位，因此不存在一组各自联合有益、只是
> 组合后被交互破坏的转移来挽救该邻域。

这里的“最小项”是一个等成本预算转移，通常包含一降一升两个窗口；论文不使用内部分类
`single-item misranking primary`。`interaction_witness_count=0` 只否定“若干已各自联合有益的转移在组合后发生
反转”这一特定解释，不能证明 Soft-NMS、排序或窗口组合完全没有交互。

本终态足以停止：当前 capped→released 联合邻域修复，以及由同一 H65 priority sequence、三档预算、逐窗口
反事实 detector loss、逐视频等成本加性分配和 cap/配对/tie-break 调整组成的现有 DUCA-Marginal-v1。它不足
以否定 K256/K384/K512 三档本身、H65 priority sequence、所有三档分配或任务感知动态计算的一般问题。

## 只读归档状态

分支 `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831` 从本记录起只作为负证据读取，不再接受恢复性
修改、阈值调整、配对重排、事后状态选择、bootstrap、utility-head 训练或 official-test 运行。本机制没有后继
Builder、Critic、PRE_RUN、Evaluator 或算力任务。若未来重新研究动态计算，必须由 Pro 提出新的机制假设和
独立任务，不能描述成 Marginal-v1 的小修或恢复。

这组结果可进入论文的失败机制分析或补充材料，但不能进入主性能表作为候选方法成绩，也不能表述为动态预算
总体无效、H65 无效、三档预算无效、统计显著的负总体结论或可部署 oracle。
