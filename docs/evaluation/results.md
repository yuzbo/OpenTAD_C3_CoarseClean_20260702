---
updated: 2026-07-11
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
| 官方 I3D 数据准备 | `18cf111` | cancelled | Slurm `1156248` 网络失败；恢复作业 `1157170` 因研究主线改为 raw-video AdaTAD 而主动取消 |
| 真实 THUMOS feature one-batch | `7098049` | cancelled | Slurm `1156249` 及其依赖全部取消；不得作为 PhysTime-AdaTAD 证据 |
| PhysTime-AdaTAD raw-video gate 首次提交 | `2cfdf2e` | infrastructure failed | Slurm `1158528` 在 Python/模型执行前因非登录 shell 无 `module` 命令以 127 退出；依赖 `1158529/1158530/1158531` 未启动并取消，不构成方法证据 |
| PhysTime-AdaTAD raw-video gate 第二次提交 | `5d73b98` | infrastructure failed | Slurm `1158546` 的 matched validator 通过，但 submission 覆盖 Slurm 的 GPU mask 后 `torch.cuda.is_available=false`；模型未构建，依赖 `1158547/1158548/1158549` 未启动并取消，不构成方法证据 |
| PhysTime-AdaTAD raw-video gate 第三次提交 | `92ea441` | determinism gate failed | Slurm `1158556` 通过 raw config、CUDA、真实 THUMOS decode 与 same-frame checksum，但三次独立 train pipeline 的增强后像素 checksum 不同；根因为 imgaug RNG 未纳入统一 seed。模型未构建，依赖 `1158557/1158558/1158559` 未启动并取消，不构成方法证据 |
| PhysTime-AdaTAD raw-video gate 第四次提交 | `c448f1f` | determinism gate failed | Slurm `1158576` 显示 physical-grid/PhysTime 输入一致，而进程内首个构建的 selected-axis 只在 ColorJitter 后分叉；逐 transform 诊断 `1158591` 定位为首次 ImgAug 构造消耗 NumPy 状态。模型未构建，依赖 `1158577/1158578/1158579` 未启动并取消 |
| 三头真实 pipeline 增强确定性诊断 | post-`c448f1f` fix | passed | Slurm `1158614`：三头在 DecordDecode、RandomResizedCrop、ImgAug、ColorJitter、FormatShape 后的像素 SHA256 均逐级相同；这是数据合同证据，不是 mAP 证据 |
| PhysTime-AdaTAD raw-video FP32 gate | `d31e99c` | passed | Slurm `1158636`，`gate_pass=true`；真实 THUMOS MP4、K=384/768、same frame/input checksum、三头 optimizer coverage、adapter 梯度及 PhysTime projection/classification/regression/endpoint 梯度均通过。后续 gate 已升级为 AMP |
| PhysTime-AdaTAD raw-video AMP gate | `bd27544` | passed | Slurm `1158668`，`gate_pass=true`；三头 `amp_enabled=true`，其余 raw-video、same-index/input、optimizer 与梯度合同全部通过，正式三头依赖已释放 |
| PhysTime-AdaTAD masked-attention 修复后 AMP gate | `0bbf0e9` | passed | Slurm `1158718`，`gate_pass=true`；三头 finite loss/prediction、optimizer coverage 与必需梯度全部通过。远端 PhysTime focused suite `68 passed` |

### Raw-video gate `1158636` 原始指标

| Head | cost | train forward+backward (ms) | inference (ms) | peak CUDA memory (MB) | adapter grad | detector grad |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| selected-axis | 0.265581 | 14162.69 | 55.61 | 1296.67 | nonzero | nonzero |
| physical-grid | 0.266731 | 448.40 | 62.89 | 1296.26 | nonzero | nonzero |
| PhysTime | 0.550057 | 1088.00 | 74.32 | 1141.34 | nonzero | nonzero |

selected-axis 的一次性 14.16 s 包含首个 CUDA/CuDNN warm-up，不可当作稳态延时比较；正式成本结论等待 full-run 账本。

### AMP gate `1158668` 原始指标

| Head | cost | train forward+backward (ms) | inference (ms) | peak CUDA memory (MB) | AMP | required gradients |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| selected-axis | 0.265547 | 10364.46 | 32.64 | 1058.75 | on | nonzero |
| physical-grid | 0.266705 | 354.10 | 40.02 | 1062.05 | on | nonzero |
| PhysTime | 0.550022 | 963.10 | 42.93 | 904.36 | on | adapter/projection/cls/reg/endpoint nonzero |

## Raw-video K384 Formal Track

统一 run root：`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_d31e99c_k384_20260711_161413_+0800`。

| Head | Job | Commit | 状态 | 当前证据 | Avg-mAP | mAP@0.7 |
| --- | ---: | --- | --- | --- | ---: | ---: |
| selected-axis | `1158637` | `d31e99c` | cancelled / diagnostic | 已进入 epoch 0 后主动取消；修复后套件必须同 commit 重跑，故本作业不进入最终 matched table | NA | NA |
| physical-grid | `1158638` | `d31e99c` | infrastructure failed | epoch 0 首步后 torchrun rendezvous broken pipe；无方法失败证据 | NA | NA |
| PhysTime | `1158639` | `d31e99c` | implementation failed | epoch 0 首批触发 autocast 不允许 probability BCE；已按等价 event-logit BCE 修复 | NA | NA |

第二套 matched run root：`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_bd27544_k384_20260711_162907_+0800`。

| Head | Job | Commit | 最终状态 | Avg-mAP | mAP@0.7 |
| --- | ---: | --- | --- | ---: | ---: |
| selected-axis | `1158669` | `bd27544` | cancelled / diagnostic；新 commit 需 matched 重跑 | NA | NA |
| physical-grid | `1158670` | `bd27544` | cancelled / diagnostic；新 commit 需 matched 重跑 | NA | NA |
| PhysTime | `1158671` | `bd27544` | implementation failed；epoch 0 第 50 步三项 loss 全 NaN，根因为未覆盖 logits 在 AMP masked softmax 中形成 `inf * 0` | NA | NA |

当前 matched run root：`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_0bbf0e9_k384_20260711_164800_+0800`。

| Head | Job | Commit | 状态（2026-07-11 16:58 +08:00） | epoch 0 step 50 loss | Avg-mAP | mAP@0.7 |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| selected-axis | `1158719` | `0bbf0e9` | running，epoch 1 | 1.7127 | NA | NA |
| physical-grid | `1158720` | `0bbf0e9` | running，epoch 1 | 1.7261 | NA | NA |
| PhysTime | `1158721` | `0bbf0e9` | running，epoch 1；已越过旧 NaN 暴露点 | 1.9106 | NA | NA |

## Matched Pilot

统一 run root：`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_7098049_pilot_20260710_214816_+0800`。

| 实验 ID | Job | 方法 | K | 训练视图 | Seed | 状态 | Avg-mAP | mAP@0.7 | 结果路径 |
| --- | ---: | --- | ---: | --- | ---: | --- | ---: | ---: | --- |
| `phys_support_k384_s42` | `1156250` | PhysTime support measure | 384 | random + bursty | 42 | cancelled | - | - | feature-token 旧路线 |
| `phys_point_k384_s42` | `1156251` | physical point-only | 384 | random | 42 | cancelled | - | - | feature-token 旧路线 |
| `phys_nodisc_k384_s42` | `1156252` | PhysTime no consistency | 384 | random | 42 | cancelled | - | - | feature-token 旧路线 |
| `selected_k384_s42` | `1156253` | selected-axis ActionFormer | 384 | random | 42 | cancelled | - | - | feature-token 旧路线 |
| `timestamp_k384_s42` | `1156254` | timestamp-channel selected-axis | 384 | random | 42 | cancelled | - | - | feature-token 旧路线 |
| `phys_support_k192_s42` | `1156255` | PhysTime support measure | 192 | random + bursty | 42 | cancelled | - | - | feature-token 旧路线 |
| `phys_support_k768_s42` | `1156256` | PhysTime support measure | 768 | random + bursty | 42 | cancelled | - | - | feature-token 旧路线 |

## 扩展门槛

只有真实数据 gate 通过且 `phys_support_k384_s42` 没有非有限 loss、持续梯度跳过或明显训练崩溃，才允许启动三种子和 sampling robustness 扩展。程序正常退出不等价于方法门槛通过。
