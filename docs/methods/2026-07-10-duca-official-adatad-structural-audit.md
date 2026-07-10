---
updated: 2026-07-10
status: active
scope: DUCA 主配置与官方 OpenTAD/AdaTAD 的配置、源码和坐标语义一致性审计
official_repository: https://github.com/sming256/OpenTAD
official_commit: 1aa8ca4ac5e846b1e8ff69298dd6607121a01589
---

# DUCA 与官方 AdaTAD 结构一致性审计

## 裁决

当前 DUCA **不是与官方 AdaTAD 源码完全一致的模型**，也不能表述为
“未修改的官方 AdaTAD”。准确口径是：

> 使用官方 OpenTAD/AdaTAD 派生的 VideoMAE-S、projection、neck 与
> ActionFormerHead 配置，在其前加入 DUCA selector，并由扩展后的
> ActionFormer/SingleStageDetector 完成 selected-axis GT 映射、检测和
> true-time 后映射。

这一区分不是措辞洁癖。DUCA 改变了检测器实际消费的时序长度与坐标系，
即使分类、回归和正负样本分配公式本身保持不变，也不能据此声称整个检测
路径与官方实现完全相同。

## 可复核证据

审计固定官方仓库 `sming256/OpenTAD` 的提交
`1aa8ca4ac5e846b1e8ff69298dd6607121a01589`。

| 对象 | 官方 Git blob | DUCA 分支 Git blob | 结论 |
| --- | --- | --- | --- |
| AdaTAD base config | `e0dd2a0d4ed9c1ecbe2a2c042c0f748cda016266` | `e0dd2a0d4ed9c1ecbe2a2c042c0f748cda016266` | 字节一致 |
| `actionformer_head.py` | `42e78b261686801097592d53d63fc3c4ca0f8d89` | `42e78b261686801097592d53d63fc3c4ca0f8d89` | 字节一致 |
| `anchor_free_head.py` | `fe9c12a049da8b107604e44ffe0dd9d042528352` | `8c877962d9d444c9c9ec115eac53432a046e4123` | 已扩展 |
| `actionformer.py` | `f07e3e8acca94d3899aa72f94c67ad7476e3e558` | `24fed8a70c3ed4535a01ced78e5fef9d1e5a7d81` | 已扩展 |
| `single_stage.py` | `3ba72941c4ce063f7c81c8960b3fefa9f109a328` | `f97cab56c9809eb0d140214430cd6c749d0c1732` | 已扩展 |
| `vit_adapter.py` | `6c505dc31b518c42b83003b206e2eaccfd7d58f3` | `c72bece85b003a8d9c87d5bf7ca580582f17a66a` | 已扩展 |

相对官方提交，上述四个已扩展源码的审计差异量分别为：

- `actionformer.py`: `+595/-13`
- `single_stage.py`: `+185/-10`
- `anchor_free_head.py`: `+214/-17`
- `vit_adapter.py`: `+415/-6`

这些数字证明源码不相同，但不单独证明存在数值错误；是否产生行为变化需按
具体配置判断。

## 保持一致的部分

1. 主配置继承的 AdaTAD base config 与官方固定提交字节一致。
2. `ActionFormerHead` 直接源码字节一致，主配置的 `rpn_head` 配置与官方
   base config 相等。
3. DUCA fixed 主配置没有启用 `physical_grid_actionformer`；已有 physical-grid
   扩展不是当前主实验的活动路径。
4. 当前证据支持“分类、回归、assignment 的公式未主动改写”，不支持
   “整个 assignment 行为与官方完全相同”。

## 有意改变的部分

1. 在 heavy backbone 前加入 `DucaOnlineFrameSelector`。这里的 `Online` 是
   历史类名；任务协议实际是离线全窗口、forward 内即时选择，不是 Online TAD。
2. detector 输入由 dense `T=768` 改为 selected `K`；fixed-384 时为 `K=384`。
3. VideoMAE `total_frames`、chunk 数和 projection `max_seq_len` 随 `K` 改变。
4. 训练 GT 从原始时间轴映射到 selected axis；ActionFormer 在等间隔 selected
   axis 上完成卷积、assignment 和回归。
5. 后处理再通过 selected positions 将检测结果映回原始时间轴。
6. `ActionFormer`/`SingleStageDetector` 增加 selector、损失合并、梯度桥、元数据
   传递、优化器覆盖和坐标后映射逻辑。

因此应区分两句话：

- **正确**：assignment 的数学公式和官方 head 配置保持不变。
- **错误**：assignment 的输入坐标语义和官方 dense AdaTAD 完全不变。

## 已确认的风险与当前错误

已确认的实现错误是旧 validator/contract 的表述范围过大：
`changes_detector_head=False` 与 `changes_loss_assignment=False` 没有说明它们只
表示 head 配置和 assignment 公式未改，容易被误读为源码、输入几何和运行行为
完全一致。新契约必须同时公开：

- `official_detector_source_identical=False`
- `loss_assignment_coordinate_system_changed=True`
- `detector_input_length_changed=True`
- `selected_axis_adapter_active=True`
- `gt_remap_active=True`
- `posthoc_true_time_remap_active=True`

当前没有证据证明这些结构改写必然导致数值错误；但 selected-axis 将不等物理
间隔视为等间隔，确实是尚未解决的建模风险。按当前决策暂不实现 physical-grid，
该风险必须保留在论文限制和后续裁决实验中，不能被“官方后端”字样掩盖。

## 论文与实验口径

论文、图示和配置摘要统一使用：

> official OpenTAD/AdaTAD-derived detector components with a DUCA
> pre-backbone selector and selected-axis coordinate adapter

不再使用：

- “完全未修改的官方 AdaTAD”
- “与官方 AdaTAD 源码完全一致”
- “loss assignment 完全不变”

可使用但必须限定范围：

- “官方 AdaTAD base config 与 ActionFormerHead 配置保持一致”
- “未启用 physical-grid head”
- “assignment 公式不变，但 assignment 所在的坐标轴已改变”
