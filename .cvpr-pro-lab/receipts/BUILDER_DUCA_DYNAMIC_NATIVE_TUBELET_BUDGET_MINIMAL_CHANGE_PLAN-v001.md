# DUCA 窗口级动态原生 tubelet 预算：最小实现计划

## 身份与边界

- 科学裁决：fresh exact-DUCA Project Pro `PIVOT`。
- 干净实现基座：`b33391126eac05e3353d322b973dda91741f0732`。
- worktree：`C:/Users/skywalker/.codex/worktrees/duca-dynamic-native-tubelet-budget-20260829/OpenTAD_C3_CoarseClean_20260702`。
- branch：`codex/duca-dynamic-native-tubelet-budget-20260829`。
- 只实现一个窗口级 16/20/24-clip 动态臂；不重训 uniform 对照，不修改下游检测器、损失、NMS、评价器、split、seed 或 60 轮/6000 次成功更新合同。

## 当前缺口

1. 当前 `ThumosSlidingDataset` 与普通 `DistributedSampler` 逐窗口随机取样，一次前向看不到同一视频的全部窗口，不能在前向中正确计算视频内窗口需求排名。
2. 当前 native-tubelet 路径固定选择 192 个 tubelet，并以 `t1=24` 运行 VideoMAE；没有真实的 16/20/24-clip 分桶执行。
3. 当前物理时间重建可以消费不同数量的有效位置并恢复到固定 384 点 detector 网格，可直接保留。

## 最小实现

1. 使用冻结的 H65 Stage-1 epoch-29 EMA 侦察器，为当前被消费 split 中每个视频的全部滑动窗口生成确定性需求表。逐窗口统计平均动作性、边界重要性的 90th percentile 和时序新颖性的 90th percentile，分别转为视频内 percentile rank，再使用已完成 coreset 的固定分量权重合成需求分数。按需求排序，以更早的物理起始时间打破并列；最低和最高 `floor(W/2)` 个窗口分配 16/24 clips，奇数视频的唯一中位窗口分配 20 clips。训练与官方验证各自只使用该 split 的原始低分辨率输入和冻结侦察器输出；不得消费 GT、teacher、detector predictions、validation metrics、checkpoint-dependent information 或 raw-prediction cache，也不得跨 split 拟合或调参。
2. 数据集保持原有长度、shuffle 与每轮 100 个 batch；训练时按稳定的 `(video_name, window_start_frame)` 查表，避免引入改变 6000-update 暴露合同的 video-group sampler。
3. 每个预算内使用确定性均匀的 native-tubelet positions：16/20/24 clips 分别对应 128/160/192 个 tubelet 和 256/320/384 个高分辨率帧。
4. 在 ActionFormer heavy-backbone 入口按实际 clip 数分桶，分别执行 VideoMAE，随后恢复原 batch 顺序并用既有物理时间重建恢复到固定 384 点 detector 网格。
5. 记录每桶真实窗口数、实际 clips/tubelets/frames、VideoMAE 调用和计时；明确 `padded_to_global_max=false`。历史提交 `36d75c146492a38eb8966c66ff6b2881938cf3c6` 的 `_offline_window_table_backbone` 只作为分桶、顺序恢复和工作量记录的工程参考，不继承其预算公式或 selected-axis loader。

## 聚焦验证

- 视频内窗口排名和并列处理确定且可重复；每个视频的平均预算严格为 20 clips。
- 需求表构建拒绝 GT、teacher、detector prediction、评价指标和跨 split 拟合；训练与官方验证分别只使用各自原始输入和冻结侦察器输出。
- 16/20/24 三组分别实际输入 128/160/192 tubelets；禁止补齐到 24 clips。
- 分桶前后样本顺序、梯度和 384 点物理时间重建一致。
- loader 暴露仍为每轮 100 batches、60 epochs、6000 successful updates。
- checkpoint 每 5 epochs 保存，恢复保留模型、优化器、调度器、AMP scaler、epoch/update 与随机数状态；正式评价固定 epoch-59 EMA。

## 当前状态

最小实现计划完成，模型代码尚未修改，尚无独立 Critic、Evaluator PRE_RUN 或新 Slurm Job。
