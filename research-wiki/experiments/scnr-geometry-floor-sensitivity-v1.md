---
type: experiment
node_id: exp:scnr-geometry-floor-sensitivity-v1
title: "SCNR-TAD native-cell ROI floor sensitivity v1"
stage: experiment_running
status: m2_g1_g2_running_cost_and_finalizer_dependency_pending
outcome: pending
added: 2026-08-02
updated: 2026-08-02
---

# SCNR-TAD native-cell ROI floor sensitivity v1

## 问题与主张

本实验只回答一个问题：在其余动态 Stage-1 路由协议完全匹配时，连续 ROI 的
最小尺度应采用一个还是两个原生空间 token cell。它不用于重新比较 Free-first、
PL/ST、固定 `8/28/28`，也不单独产生 ROI+TokenSelect 有效性或效率主张。

主设置使用运行时网格推导的独立轴向下限：

`w_min = 1 / W_grid, h_min = 1 / H_grid`。

唯一的尺度敏感性消融使用：

`w_min = 2 / W_grid, h_min = 2 / H_grid`。

两者都使用同一个有界中心参数化，且不加入 Uni-AdaFocus Eq. 15 的
full-frame size penalty，也不加入 area、coverage 或 smoothness loss。

## Claim map

| Claim | 最小可信证据 | 反主张 |
| --- | --- | --- |
| G1：一个 native cell 是合理的主 floor | 1x1/2x2 在相同动态 exact-B 路径、初始化、数据顺序和训练协议下完成，并报告高 tIoU、ROI 尺度分布与真实成本 | 结果仅来自隐藏的较大窗口、额外计算或面积正则 |
| G2：方法不依赖精确的单一 floor 超参 | 1x1 不发生 sub-token collapse，且相对 2x2 不出现跨种子不稳定的灾难性退化 | 单个开发 seed 的偶然优势被当作 floor 最优性证明 |

## 冻结两臂

| Arm | `georoute_roi_extent_floor_mode` | `georoute_roi_extent_floor_cells` | 角色 |
| --- | --- | ---: | --- |
| G1 `native_1cell_main` | `native_cells` | 1 | 主设置 |
| G2 `native_2cell_sensitivity` | `native_cells` | 2 | 匹配尺度敏感性消融 |

必须逐项匹配：同一 source/scout 输入、VideoMAE 初始化、动态全窗口 exact-`B`、
允许 `K_t=0` 的 masked-zero carrier、Scheme-A utility、hard top-B/ST、早期 proxy
schedule、训练 split、样本顺序、seed、成功 optimizer update 数、AMP、EMA、detector、
evaluator/NMS 与成本测量。两臂的 `geometry_smoothness_weight`、
`area_prior_weight` 和任何 full-frame size penalty 都必须为零。

## 指标与诊断

- 决策指标：Avg-mAP、mAP@0.6、mAP@0.7，以及同预算下的实测成本 Pareto；
- 机制指标：逐 tubelet 的 `w/h/area`，floor/ceiling 饱和率，ROI token 占用，
  `K_t` 与角色计数分布；
- 计算账本：全窗口唯一 selected/executed `B`，每个 native clip 的 `b_c`，
  `sum_c b_c^2`，实际 patch embedding、attention、MLP、Adapter 调用，端到端
  p50/p95、峰值显存和 energy；
- 失败诊断：若 1x1 长期贴 floor 且定位下降，而 2x2 稳定改善，说明一格下限过松；
  若二者近似等价，保留更少先验的 1x1；若两者都失败，不能继续调 floor 来掩盖
  allocator、proxy 或 ragged execution 的问题。

## 执行顺序

| Milestone | 内容 | Gate | 状态 |
| --- | --- | --- | --- |
| M0 | 公式、独立宽高、11x20 的 1x1/2x2 known-answer tests | 精确得到 `(1/20,1/11)` 与 `(2/20,2/11)`，in-bounds 且梯度有限 | tested at exact source `4be71844`: focused `36/36`; complete GeoRoute+C3 `194 passed, 1 skipped` |
| M1 | 真实 180x320 → 11x20 P0，验证配置、审计字段和零正则 | 不产生 metric/checkpoint；任一 hidden clamp/penalty/静态 0.20 即失败 | tested: G1 Job `1215358`; G2 Job `1215364` |
| M2 | 匹配 G1/G2 development 训练、完整 accuracy/telemetry 回放、独立全栈成本回放 | 两臂全完成、population/hash/cost ledger 一致后才读结果 | runtime `6ee97336`: G1/G2 Jobs `1216180/1216181` running; cost/finalizer `1216182/1216183` dependency-pending; no result yet |
| M3 | 仅在动态主方法通过总 gate 后进入 disjoint-seed confirmation | 不从单 seed 宣称 floor 最优 | blocked |

## 当前边界

M0 与 M1 已通过。G1 的 corrected P0 Job `1215358` 使用一格 floor，G2 Job
`1215364` 使用两格 floor；两者都在动态全窗口 exact-`B`、masked-zero、真 ragged
路径上完成真实模型 forward/backward，且没有 metric/checkpoint。G2 配置在 source
`8aa8e2a3` 的契约测试中与 G1 逐字段匹配，唯一方法差异是
`georoute_roi_extent_floor_cells: 1 -> 2`。提交 `7e5775e8` 已补齐逐 tubelet
ROI/`K_t`/角色/真 ragged 成本遥测并在 N16R4 Linux 回归中通过 `35/35`。提交
`ec8de9f51f85fc81031d82b79e30019d57a381b4` 已实现并冻结单 seed-3407、60 epoch
两臂 runner、最终 epoch-59 checkpoint/成功更新/EMA 原子 sidecar、完整 Gate
accuracy/telemetry 回放、同一 GPU 上 `G1 -> G2 -> G2 -> G1` 的独立全栈成本回放，
以及 after-any fail-closed finalizer。成本 validator 会从原始 NVML trace 与单调时间窗
重新积分 energy，并绑定完整 population、配置、checkpoint 和 stage-result 谱系。
本地编译、Bash、whitespace 与聚焦契约回归通过（`29 passed`）。干净远端 runtime
`9d6641a6` 进一步通过 Linux/Torch `76/76` 与两臂、成本、终结器四项 precheck；
存储要求 `47,244,640,256` bytes、可用 `225,293,430,784` bytes，提交额度也通过。
但正式部署在任何 Job 创建前被 N16R4 submit Lua 拒绝：CPU-only finalizer 不满足该站点
必须申请 GPU 的规则。旧 root
`/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_9d6641a6_s3407_20260804_0507`
只含 storage preflight，永久不复用。最小修复
`bad14693daa1fe414e56bf697c617e76f96eed48` 将 finalizer 改为 1 GPU/1 CPU 调度
（不进行模型或成本计算，并在 deployment receipt 中披露为 scheduling overhead），
本地 `13/13` 通过。新源的远端回归、新 precheck 与新 namespace 尚待执行；训练、
checkpoint、metric、latency、energy 仍全部未产生。

替换 runtime `6ee97336775a09611f10423e07cafcea375e191a` 已通过相同远端
Linux/Torch `76/76` 与全新四项 precheck，并在新 root
`/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525`
原子提交和释放 G1 `1216180`、G2 `1216181`、paired-cost `1216182`、finalizer
`1216183`。部署 self-hash 为
`a0504e45179957f20580b901e6ef7723d63c7b0ed445d8b3c35c3b5aaa02b89a`，
deployment 文件 SHA-256 为
`188da9dbf8cabffc1ab59cd90822e117adcd1457729c5a08806c916516ac8284`。
G1/G2 新 P0 均为 `PASS_NO_PERFORMANCE_P0`（文件 SHA-256
`90d8cc10b9cb13853d90a8ff2cf4d92115ee3f99ffdcca2c12a4318cdfc08010` /
`30517e05372d4307aa2d40301031d42fdc33b4cd5d205aaa4d14f7234eceb0c1`），
随后两臂进入 Epoch 0；成本与终结器仍按依赖等待。当前状态仅为
`experiment_running`，不能从 live log、同步 AMP replay 或部分产物读取 floor 结论。

P0 中三角色计数、`K_t` 范围、loss 或梯度只能证明路径非退化且可训练，不能用于
判断 1x1/2x2 谁更好。只有完整匹配的 development 训练、相同 population/hash、
决策指标与实测成本全部封存后，才允许给出 floor 敏感性结论；仍不得把旧固定
K64/`8/28/28` checkpoint 换 floor 后混入本实验。
