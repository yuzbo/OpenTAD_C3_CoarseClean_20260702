---
title: 2026-07-23 DUCA 夜间实现与实验总账
status: experiment_running
updated: 2026-07-23
---

# 本轮目标

在不改 selector、decoder、VideoMAE 和 TAD 后端的前提下，同时收割当前 R0--R5 主线、比较四种官方粗分类器，并实现“稀疏粗分类计算 + 多维 temporal hidden 线性重建 + 完整 TAD mAP”四档实验。

# 唯一实验身份

| 组别 | 精确提交 | Jobs | 当前状态 | 回答的问题 |
| --- | --- | --- | --- | --- |
| R0--R5 主线 | `9f97f2c7f081b10fbf1f63d0602a621c6b43a780` | `1180490--1180496` | R1 完成；R2/R3/shared 运行；R4/R5 依赖等待 | 边界微簇与 selected-axis DUCA 能否在五预算、两后端、三种子下获得 terminal official mAP 与成本收益 |
| 四粗分类器 P0 | `4f81299f826a4d33b18f21af8436ec1bd8cc4f51` | `1180502--1180505` | 全部 `COMPLETED/0:0` | MS-TCN2、ASFormer、FACT、Video-Mamba-ASFormer 哪种低分辨率二分类/状态转变证据更强 |
| 稀疏粗扫描完整 TAD | `dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45` | Gate `1180556`；Suite `1180557` | Gate 完成；四臂运行 | 粗 probe 每 4/8/12/16 个源帧计算一次并插值 hidden 后，最终 TAD mAP 与总成本如何变化 |

# 今晚完成的模型代码

- 分支：`codex/duca-sparse-probe-interpolation-20260723`。
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`。
- 只在规则时间 anchor 上运行空间 stem 与官方 ASFormer；把 action logits、temporal encoder hidden 和 policy hidden 按原始有效时间坐标线性重建到完整候选网格。
- 插值值直接作为 selector 证据，不提供 anchor mask 或距 anchor 距离。
- d=1/2/3/4 只改变 probe 计算密度；保持 R2Q3、K384/G2、VideoMAE、official-derived AdaTAD/ActionFormer、seed 和 official-60 协议一致。
- 每臂按 P0 20 epochs -> 真实 full-model gate -> official-60 60 epochs 执行，只以 terminal epoch-59 `state_dict_ema` 完整 THUMOS validation mAP 裁决。

# 已形成证据

- 新 focused test：`4 passed`。
- 真实 CUDA Gate `1180556 COMPLETED/0:0`：四档输出长度和数值正确，空间/时序梯度非零，估算 MACs 随 d 增大单调下降。
- 四粗分类器终轮 Action AP：MS-TCN2 `0.4078`、ASFormer `0.4087`、FACT `0.3945`、Video-Mamba-ASFormer `0.4161`。
- 四者的最佳间接边界策略均为 `delta_p_action`；对应边界支持@1 为 `0.7225/0.8184/0.7956/0.8302`。
- 稀疏四臂首个共同点均到 P0 epoch 1 batch 20，K384、有限损失；显存为 `3719/2108/1568/1315 MB`。d=1 两次 AMP replay 后正常更新。

# 证据边界

- 四粗分类器结果没有 detector，不是 TAD mAP。
- CUDA 梯度和 MACs 只证明实现与成本趋势，不证明检测性能。
- 稀疏四档目前没有 terminal mAP，状态是 `experiment_running`，不能声称 d=2/3/4 可无损替代密集粗扫。
- combined 回归中的一个 exact-equality 随机性测试在未修改基线 `4f81299` 上同样失败，不是本次稀疏实现回归。

# 运行位置与跟踪

- R root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`。
- 四粗分类器 root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_coarse_backends_4f81299_20260723_0015`。
- 稀疏粗扫 root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329`。
- 每小时 heartbeat `duca-21-00-full-progress-report` 已更新；只在新状态、错误、terminal mAP、成本或裁决出现时更新 Wiki 和通知。

# 下一次必须收割

1. 稀疏四臂 P0 终点粗证据质量和 full-model gate。
2. d=1/2/3/4 terminal epoch-59 EMA Avg-mAP、tIoU 0.3--0.7、短动作和高 tIoU 差异。
3. probe、插值、selector、VideoMAE、detector 与总延时/显存/FLOPs，形成性能-成本 Pareto。
4. 只在统一 temporal-hidden 的完整 TAD 实验中比较四种粗分类器，不能用本轮 P0 指标替代。
