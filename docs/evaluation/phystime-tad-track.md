---
updated: 2026-07-10
status: active
scope: PhysTime-TAD 实验协议、部署依赖与审计路径
out-of-scope: 实验结果数字与方法正文
---

# PhysTime-TAD Experiment Track

## 代码锚点

- Branch: `codex/phystime-tad-2`
- 首个模型 commit: `5a46ea6`
- 方法契约: [phystime_tad_contract.md](../methods/phystime_tad_contract.md)
- 唯一结果表: [results.md](results.md)

## Phase 0: 数据与真实门禁

1. 从 OpenTAD 官方 THUMOS 页面提供的 ActionFormer I3D two-stream archive 下载特征。
2. 校验至少 300 个 `.npy`、标注、类别表和缺失视频清单。
3. 用正式 config 读取一个真实训练样本，执行 forward/backward/inference。
4. 审计秒坐标 GT、支持区间来源、paired view、optimizer coverage 和四条关键梯度。

任何一步失败，所有训练任务通过 `afterok` 自动保持不运行。

## Phase 1: Matched Pilot

七个 pilot 使用相同 I3D 特征、THUMOS split、60 epoch schedule、分类/回归头宽度与 seed 42。变化项仅为时间几何、K 或一致性监督。

| ID | 唯一变化 |
| --- | --- |
| `phys_support_k384_s42` | 完整 support-integrated PhysTime |
| `phys_point_k384_s42` | 支持质量替换为 point Gaussian measure |
| `phys_nodisc_k384_s42` | 单视图且无 discretization consistency |
| `selected_k384_s42` | selected-rank ActionFormer |
| `timestamp_k384_s42` | selected-rank + 四个显式时间通道 |
| `phys_support_k192_s42` | K=192 |
| `phys_support_k768_s42` | K=768 |

## Phase 2: 结果门控扩展

Phase 2 不在 pilot 结果出现前自动启动。通过门槛后部署：

- seed 43/44 的 support、point、selected-axis；
- uniform/random/bursty/contiguous-gap；
- K=192/384/768；
- 最终报告 mean/std/worst-case、mAP@0.7、短动作、显存和延迟。

## 失败分类

- 数据下载、checksum、样本计数失败：data blocker；
- metadata、GT unit、support provenance 失败：contract blocker；
- OOM：允许一次降低 batch size 重试，但不得改模型；
- 非有限 loss 或连续超过 10 次梯度跳过：training collapse；
- isolated early gradient skip：记录并继续观察。
