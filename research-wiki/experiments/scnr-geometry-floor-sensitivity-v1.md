---
type: experiment
node_id: exp:scnr-geometry-floor-sensitivity-v1
title: "SCNR-TAD native-cell ROI floor sensitivity v1"
stage: designed
status: approved_waiting_for_dynamic_stage1_executor
outcome: no_result
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
| M0 | 公式、独立宽高、11x20 的 1x1/2x2 known-answer tests | 精确得到 `(1/20,1/11)` 与 `(2/20,2/11)`，in-bounds 且梯度有限 | implemented locally; remote tensor test pending |
| M1 | 真实 180x320 → 11x20 P0，验证配置、审计字段和零正则 | 不产生 metric/checkpoint；任一 hidden clamp/penalty/静态 0.20 即失败 | blocked on executable dynamic Stage 1 |
| M2 | 匹配 G1/G2 development 训练 | 两臂全完成、population/hash/cost ledger 一致后才读结果 | blocked |
| M3 | 仅在动态主方法通过总 gate 后进入 disjoint-seed confirmation | 不从单 seed 宣称 floor 最优 | blocked |

## 当前边界

当前共享 GeoRoute 几何 primitive 可以实现运行时 axis-specific native-cell floor，
但最终动态全窗口 exact-`B` ragged executor、masked-zero carrier 和完整 proxy/ST
训练路径尚未形成可执行实验。不得把旧固定 K64/`8/28/28` checkpoint 换一个 floor
后称为本实验，也不得为尽快开跑而给每个 tubelet padding dummy heavy tokens。
