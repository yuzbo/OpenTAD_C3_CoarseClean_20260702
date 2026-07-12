# Gap Map

## gap:G1 - 不规则观测下的内部时间度量错误

状态：核心未解决。

selected-axis Conv/attention/pyramid 把相邻 token 当等间隔，输出 remap 无法修复内部 receptive field 与 assignment 的物理意义。`idea:phystime-adatad-1` 正在最小化验证这一 gap。

## gap:G2 - 观测支持区间与缺失质量没有严格定义

状态：算子与 raw-video AMP gate 已完成，full-run 证据进行中。

相邻时间戳的 Voronoi 宽度会跨 gap 膨胀支持并虚构观测。PhysTime-TAD 2.0 使用可审计原始 support cell、midpoint clipping 和 overlap mass，但还需 raw-frame pipeline gate。

## gap:G3 - 新颖性与近邻方法碰撞

状态：高风险。

mTAN 已做 irregular continuous-time attention；TE-TAD 已用 actual timeline coordinate；FrameDrop/TRC 已研究缺帧定位鲁棒性；LiquidTAD 已使用 continuous dynamics prior。必须证明 support-integrated physical-time detector 不等于 timestamp embedding、interpolation 或一般 time-series projection。

## gap:G4 - raw-video AdaTAD 的公平三头隔离

状态：matched pipeline 与单步 AMP gate 已实现，但首轮 full run 无效。

需要三个 config 在相同 raw videos、selected indices、VideoMAE-S adapter、checkpoint、schedule、seed 和 NMS 下，只改变时间几何/检测头。

`0bbf0e9` 三头均在 epoch 41 首次验证时因 evaluator annotation 相对路径错误退出；PhysTime 另从 epoch 1 end 起持续全 NaN。修复后必须同 commit 重跑三头，当前无可比较 mAP。

## gap:G5 - 高 tIoU 与短动作证据

状态：未开始。

主结果必须包含 mAP@0.6/0.7、boundary error、短动作分组和最差采样模式。只提高 Avg-mAP 不足以支持物理时间定位主张。

## gap:G6 - robustness 不是简单 augmentation 收益

状态：Phase 2 HOLD。

需要 seen/held-out sampling families、matched K/coverage/observability，并与 FrameDrop/TRC、timestamp embedding 和 interpolation 区分。

## gap:G7 - 完整计算账本

状态：未闭环。

必须计 raw decode、preprocess、H2D、VideoMAE adapter、projection/head、padding 和 NMS，不能只报 K 或 head FLOPs。

## gap:G8 - 第二数据集与跨 FPS 泛化

状态：Phase 2 HOLD。

THUMOS14 单数据集可能奖励特定 action density 和 sampling prior。主方向最终需要至少第二数据集与跨 FPS/采样模式证据。

## gap:G9 - 当前代码与最终 raw-video 目标之间的实现缺口

状态：已关闭。

`BuildPhysTimeRawFrameGeometry`、K384 configs、same-index/同增强 validator、raw-video AMP gate、multi-step stability gate、formal launcher、evaluator 绝对路径和 full-run 结果均已落地。后续性能问题归入新的结构/因果对照缺口，不再误记为部署稳定性缺口。

## gap:G10 - 论文主张审计

状态：无正向 claim 节点；已有负结果与诊断裁决。

`exp:phystime-adatad-k384` 已完成但不支持“优于不规则采样基线”“更鲁棒”或“保持计算节省”等正向主张。只有等容量因果对照、多 seed、高 IoU、成本和第二数据集证据闭环后，才能重新创建正向 claim。
