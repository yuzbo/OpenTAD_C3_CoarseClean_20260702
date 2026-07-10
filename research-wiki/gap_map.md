# Gap Map

## gap:G1 - 不规则观测下的内部时间度量错误

状态：核心未解决。

selected-axis Conv/attention/pyramid 把相邻 token 当等间隔，输出 remap 无法修复内部 receptive field 与 assignment 的物理意义。`idea:phystime-adatad-1` 正在最小化验证这一 gap。

## gap:G2 - 观测支持区间与缺失质量没有严格定义

状态：算子层已实现，raw-video 证据未完成。

相邻时间戳的 Voronoi 宽度会跨 gap 膨胀支持并虚构观测。PhysTime-TAD 2.0 使用可审计原始 support cell、midpoint clipping 和 overlap mass，但还需 raw-frame pipeline gate。

## gap:G3 - 新颖性与近邻方法碰撞

状态：高风险。

mTAN 已做 irregular continuous-time attention；TE-TAD 已用 actual timeline coordinate；FrameDrop/TRC 已研究缺帧定位鲁棒性；LiquidTAD 已使用 continuous dynamics prior。必须证明 support-integrated physical-time detector 不等于 timestamp embedding、interpolation 或一般 time-series projection。

## gap:G4 - raw-video AdaTAD 的公平三头隔离

状态：尚未实现。

需要三个 config 在相同 raw videos、selected indices、VideoMAE-S adapter、checkpoint、schedule、seed 和 NMS 下，只改变时间几何/检测头。

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

状态：P0。

缺少 `BuildPhysTimeRawFrameGeometry`、matched K384 configs、same-index validator、raw-video one-step CUDA gate、formal launcher 和结果记录。

## gap:G10 - 论文主张审计

状态：无 claim 节点。

在 `exp:phystime-adatad-k384` 完成并经过 result-to-claim 之前，不创建或传播“优于不规则采样基线”“更鲁棒”“保持计算节省”等已证实主张。
