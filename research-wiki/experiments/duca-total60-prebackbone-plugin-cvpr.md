# DUCA 总计 60 轮纯前置插件主实验

## Status

- Date: `2026-07-27`
- Stage: `designed_major_revision`
- Approval: `user_approved_initial / pro_review_absorbed_with_corrections`
- Implementation: `not_started`
- Task: offline TAD
- Canonical specification:
  `docs/superpowers/specs/2026-07-27-duca-total60-prebackbone-plugin-cvpr-design.md`
- Scientific-review absorption:
  `docs/methods/2026-07-27-duca-total60-prebackbone-pro-review-absorption.md`

## Scientific question

在保持后续时域动作检测器主体不变、严格固定 K 帧预算和总计 60 轮检测器训练预算的条件下，有界单调密度传输能否稳定超过干净均匀下采样，并以经过真实离散收益验证的分类、边界贡献监督保护高重叠阈值定位？

## Frozen decisions

1. 主方法是 pre-backbone 选帧插件，不改检测器主干、投影、标签分配、检测头、损失和 NMS。
2. 官方 dense、窗口预算匹配 50%/25% 均匀采样、整视频 1/2/1/4 降帧率和 DUCA wrapper 均匀对照必须分开命名。
3. 所有论文主臂最多 60 轮、6000 次成功检测器更新。
4. development 因果臂修订为：干净均匀、无界密度加已验证 RDD、有界密度加同一
   RDD、有界密度加总预算内 warmup/ramp、通过独立直接梯度门后的增强臂。
5. 单帧交换只是局部有限差分。正式准入同时使用单帧、1%/5%/10% 分散多帧、连续片段和全局密度步进。
6. 准入拆为 `G_rank` 和 `G_direct`。`G_rank` 失败时，当前连续梯度贡献教师和 RDD
   长训练停止；`G_direct` 失败但 `G_rank` 通过时，仅删除直接梯度臂。
7. 第一粒种子是 development screening，不进入最终统计；结构冻结后使用未参与开发的
   预登记种子，不能把第一粒正向种子混入论文均值。
8. 现有 K=384 `65.385724%` 和 K=192 `57.967272%` 均为 90 轮超预算终端诊断，
   不是本实验的公平主结果，也不能识别 learned selector 相对 clean uniform 的收益。
9. 非线性坐标逆映射必须作用于 raw proposals，并在物理时间执行参数不变的官方 NMS。
10. 审阅提出的密度界、DP 常数、RDD 具体公式和性能/成本数值门槛暂为
    `designed_reviewer_proposal`，在 clean baseline 与训练侧功效分析后、正式结果前冻结。

## Two scientific gates

### `G_rank`

- 使用 m=1、1%/5%/10%、连续片段和全局重新硬解码验证 detached cls/reg contribution
  rank 是否预测真实硬收益；
- 报告逐视频 Spearman/Kendall、方向率、top-decile gain、matched-random regret 和
  cluster bootstrap；
- 统计单位是视频，样本数由视频聚类功效模拟确定；
- 失败时删除当前 RDD 教师，不继续对应 A1/A2/A3 长训练。

### `G_direct`

- 在 `G_rank` 通过后检查多尺度可组合性、共享 selector 参数梯度范数/夹角/有限性；
- 做同一起点 direct on/off 短程开发配对；
- 短程结果不能作为额外 1,000 updates 的主表结果；
- 失败时删除 A4，不降低门槛或更换测试集合。

两道门只使用训练侧固定留出数据，不使用官方测试 GT 或 test mAP 选择机制。

## Review verdict

`major_revision_accepted_with_corrections`：

- 完全接受 clean A0、唯一 exact-K 数学模型、warped-time I/O、双门、视频级统计、
  development seed 排除和 train-free 声明分离；
- 不直接冻结 reviewer 提出的具体常数、RDD 唯一性、`+1.0pp/40% gap` 等数值门槛；
- PR #3 的 133-file aggregate diff 必须另行清理，但不是模型实验阻塞项；
- 当前最终模型仍是 `not_started`，禁止称已实现或 paper-ready。

## Next gate

并行完成：

1. P1 released-weight dense、clean K=384/K=192 uniform 和 clean/wrapper parity；
2. P2 密度可行域、确定性 exact-K decoder、q/t 坐标和 pre-NMS inverse-map tests；
3. 视频级 `G_rank/G_direct` 统计 manifest 与工具。

任何 learned-selector 长训练至少等待 parity 和 decoder/coordinate contract；RDD 臂额外
等待 `G_rank`，A4 额外等待 `G_direct`。当前不投递 A1 至 A4 长训练。

## 2026-07-27 dynamic-K paper-role update

动态 K / AdapTok 接管回复已在
`docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`
中独立审计。该回复重新打开论文主结构，但不使本实验自动升级或失效：

- 本节点继续提供 fixed-K inner policy、clean uniform、坐标和 hard-utility 的必要
  因果门；
- fixed-K 不再被默认冻结为最终 paper main，dynamic K 是用户要求的候选中心；
- dynamic Oracle、decoder-family regret 和 mixed-K matched controls 由
  `research-wiki/experiments/duca-dynamic-k-rime-oracle.md` 单独登记；
- 在用户批准无冲突设计且前置门通过前，本节点与 dynamic 节点均不投递 learned
  long training。
