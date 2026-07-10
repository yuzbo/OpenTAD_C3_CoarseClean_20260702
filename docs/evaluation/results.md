---
updated: 2026-07-10
status: active
scope: PhysTime-TAD 实验结果的唯一数字来源
out-of-scope: 方法设计、未完成实验的推测性结论
---

# PhysTime-TAD Results

本文档是 PhysTime-TAD 实验数字的唯一权威来源。训练日志、部署清单和讨论文档只链接本文档，不复制结果数字。

## Gate 状态

| Gate | Commit | 状态 | 证据 |
| --- | --- | --- | --- |
| 合成算子与 CUDA Gate 0B | `a8c4234` | passed | Slurm `1156182`，`97 passed`，CUDA precheck passed |
| 官方 I3D 数据准备 | `7098049` | running | Slurm `1156248`，官方归档 `5.26 GB`，`data_ready.json` 待生成 |
| 真实 THUMOS one-batch | `7098049` | pending | Slurm `1156249`，依赖 `afterok:1156248` |

## Matched Pilot

统一 run root：`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_7098049_pilot_20260710_214816_+0800`。

| 实验 ID | Job | 方法 | K | 训练视图 | Seed | 状态 | Avg-mAP | mAP@0.7 | 结果路径 |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| `phys_support_k384_s42` | `1156250` | PhysTime support measure | 384 | random + bursty | 42 | pending | - | - | run root 下同名目录 |
| `phys_point_k384_s42` | `1156251` | physical point-only | 384 | random | 42 | pending | - | - | run root 下同名目录 |
| `phys_nodisc_k384_s42` | `1156252` | PhysTime no consistency | 384 | random | 42 | pending | - | - | run root 下同名目录 |
| `selected_k384_s42` | `1156253` | selected-axis ActionFormer | 384 | random | 42 | pending | - | - | run root 下同名目录 |
| `timestamp_k384_s42` | `1156254` | timestamp-channel selected-axis | 384 | random | 42 | pending | - | - | run root 下同名目录 |
| `phys_support_k192_s42` | `1156255` | PhysTime support measure | 192 | random + bursty | 42 | pending | - | - | run root 下同名目录 |
| `phys_support_k768_s42` | `1156256` | PhysTime support measure | 768 | random + bursty | 42 | pending | - | - | run root 下同名目录 |

## 扩展门槛

只有真实数据 gate 通过且 `phys_support_k384_s42` 没有非有限 loss、持续梯度跳过或明显训练崩溃，才允许启动三种子和 sampling robustness 扩展。程序正常退出不等价于方法门槛通过。
