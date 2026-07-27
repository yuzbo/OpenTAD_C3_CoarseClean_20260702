# DUCA 总计 60 轮纯前置插件主实验

## Status

- Date: `2026-07-27`
- Stage: `designed`
- Approval: `user_approved`
- Implementation: `not_started`
- Task: offline TAD
- Canonical specification:
  `docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md`

## Scientific question

在保持后续时域动作检测器主体不变、严格固定 K 帧预算和总计 60 轮检测器训练预算的条件下，有界单调密度传输能否稳定超过干净均匀下采样，并以经过真实离散收益验证的分类、边界贡献监督保护高重叠阈值定位？

## Frozen decisions

1. 主方法是 pre-backbone 选帧插件，不改检测器主干、投影、标签分配、检测头、损失和 NMS。
2. 官方 dense、窗口预算匹配 50%/25% 均匀采样、整视频 1/2/1/4 降帧率和 DUCA wrapper 均匀对照必须分开命名。
3. 所有论文主臂最多 60 轮、6000 次成功检测器更新。
4. 核心五臂为：干净均匀、密度从零训练、10+50 课程、归一化 cls/reg 贡献、通过准入门后的检测梯度。
5. 单帧交换只是局部有限差分。正式准入同时使用单帧、1%/5%/10% 分散多帧、连续片段和全局密度步进。
6. 若多尺度真实硬交换与连续代理不对齐，直接检测梯度不得进入最终模型；保留 detached normalized rank/transport teacher。
7. 第一粒种子未清楚超过干净均匀基线时，先修正时间扭曲和贡献教师，不启动多种子或第二检测器。
8. 现有 K=384 65.385724% 为 90 轮超预算诊断；当前 K=192 30+60 路线也不是本实验的公平主结果。

## Multi-scale discrete-alignment gate

- m=1 验证局部梯度方向；
- 约 1%、5%、10% 一进一出集合验证可组合性；
- 连续小片段交换验证边界、微簇和背景覆盖；
- 沿代理方向做 0.25/0.5/1.0 全局密度步进并重新硬解码；
- 报告 Spearman、Kendall、方向正确率、top-10% 收益、随机遗憾和逐视频 bootstrap 置信区间；
- 只使用训练侧固定留出数据，不使用官方测试 GT。

## Next gate

书面规范审阅通过后，实施拆为 P1 基线与路径一致性、P2 多尺度离散对齐与贡献审计、P3 有界密度总计 60 轮、P4 稳定性与免训练模式。第一批只为 P1/P2 生成实施计划，不直接投递新的长训练。
