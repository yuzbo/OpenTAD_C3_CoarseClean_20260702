---
updated: 2026-07-12
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

| Head | Job | Commit | 最终状态（2026-07-12 01:28 +08:00） | 末次训练诊断 | Avg-mAP | mAP@0.7 |
| --- | ---: | --- | --- | --- | ---: | ---: |
| selected-axis | `1158719` | `0bbf0e9` | failed；epoch 41 后首次正式验证找不到 GT annotation | epoch 41 end loss 0.4684；checkpoint `epoch_41.pth` | NA | NA |
| physical-grid | `1158720` | `0bbf0e9` | failed；epoch 41 后首次正式验证找不到 GT annotation | epoch 41 end loss 0.5285；checkpoint `epoch_41.pth` | NA | NA |
| PhysTime | `1158721` | `0bbf0e9` | failed；epoch 1 end 起持续全 NaN，epoch 41 验证另遇 GT annotation 路径错误 | epoch 41 end loss NaN；checkpoint 不构成有效结果 | NA | NA |

三头验证均因 `evaluation.ground_truth_filename` 仍指向相对路径 `data/thumos-14/annotations/thumos_14_anno.json` 而 `FileNotFoundError`。这是共享部署配置错误。PhysTime 还有独立且更严重的训练稳定性错误：首次记录的全 NaN 位于 epoch 1 step 99，此后分类、回归、端点和总 loss 持续为 NaN；因此修复标注路径后也不能直接把该 checkpoint 用作正式结果。

## Final repaired raw-video K384 track (2026-07-12)

Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_adatad_3ac93a1_k384_final_20260712_023243_+0800`

| Stage / head | Job | Commit | Status | Raw evidence | Avg-mAP | mAP@0.7 |
| --- | ---: | --- | --- | --- | ---: | ---: |
| three-step AMP + evaluator gate | `1159491` | `3ac93a1` | passed | evaluator constructed from runtime absolute annotation; all three heads completed 3 optimizer steps with finite gradients and parameters | NA | NA |
| PhysTime two-epoch stability gate | `1159492` | `3ac93a1` | passed | epoch 0 end loss 1.5824; epoch 1 end loss 1.1674; zero AMP skips; `STABILITY_GATE_COMPLETE` present | NA | NA |
| selected-axis | `1159493` | `3ac93a1` | completed | best epoch 59；无 NaN/OOM/AMP skip | 63.61 | 41.87 |
| physical-grid | `1159494` | `3ac93a1` | completed | best epoch 57；无 NaN/OOM/AMP skip | 59.14 | 32.34 |
| PhysTime | `1159495` | `3ac93a1` | completed | best epoch 59；无 NaN/OOM/AMP skip | 57.21 | 34.96 |

Failure localization retained for audit: commit `52b5756` gate `1159481` passed, but stability job `1159482` failed closed. Diagnostic job `1159489` found the first event at epoch 0 iter 47: finite forward losses and 11 Inf entries in `rpn_head.cls_head.weight` gradient, no NaN. Final protocol lowers AMP initial scale from 65536 to 1024 and bounds recoverable scaler skips; remote regression suite is `102 passed`.

### Final best-checkpoint raw mAP

同协议复算作业 `1159819/1159820/1159821` 均以 exit code 0 完成，并逐项复现正式最佳 checkpoint。结果文件位于 final run root 的 `diagnostics/checkpoint_full_eval_20260712/`。

| Head | Best epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-axis | 59 | 79.87 | 74.15 | 66.12 | 56.02 | 41.87 | 63.61 |
| physical-grid | 57 | 77.09 | 71.80 | 63.74 | 50.74 | 32.34 | 59.14 |
| PhysTime | 59 | 72.70 | 68.38 | 60.94 | 49.06 | 34.96 | 57.21 |

同一仓库的 dense AdaTAD 复现锚点为 Avg-mAP 68.29；它不是本次同采样三头隔离的一部分，只用于说明 K384 accuracy-cost 缺口。

### Capacity and supervision audit

| Head | Total params | Trainable adapter + detector | Detector params | Temporal projection params |
| --- | ---: | ---: | ---: | ---: |
| selected-axis | 49,582,504 | 27,702,568 | 26,695,708 | 23,505,920 |
| physical-grid | 49,582,504 | 27,702,568 | 26,695,708 | 23,505,920 |
| PhysTime | 29,242,288 | 7,362,352 | 6,355,492 | 3,168,774 |

因此当前三头实验不是只改变坐标表示的等容量 head isolation：PhysTime 的可训练 adapter+detector 参数仅为 ActionFormer 两个对照的 26.58%，且删除了 ActionFormer 的跨时间 Transformer 投影栈。

| Geometry / assignment diagnostic | selected-axis | physical-grid | PhysTime |
| --- | ---: | ---: | ---: |
| test mean valid candidates/window | 748.86 | 748.86 | 397.52 |
| train Monte Carlo mean valid candidates/window | 710.34 | 710.34 | 343.78 |
| train mean eligible locations/GT | 3.450 | 3.482 | 3.015 |
| train `<1 s` mean eligible locations/GT | 2.379 | 2.316 | 1.348 |
| train equal-duration multi-GT conflict share | 6.75% | 6.94% | 7.44% |

PhysTime/test 与 selected-axis/test 的候选数比为 0.5308；训练 Monte Carlo 比值为 0.4840。当前 PhysTime target assignment 在多 GT 同最短距离位置只保留一个标签，而 ActionFormer 对照保留同长度标签集合。

### Learned query and attention audit

Epoch-59 EMA checkpoint 的 query embedding 审计显示，原始绝对 `center_sec` 对 query pre-activation 的贡献占比从细到粗层为 95.31%、94.93%、94.65%、93.53%、92.41%、90.53%。绝对秒数项主导了归一化位置、宽度和 Fourier 时间特征。

真实 THUMOS checkpoint attention 诊断作业 `1159823` 使用 8 个等距测试窗口：

| Level | Median covered observations | Mean effective observations | Effective/covered | Mean content-logit span | Mean relative-time span |
| --- | ---: | ---: | ---: | ---: | ---: |
| L0 | 2 | 1.54 | 0.830 | 0.624 | 0.197 |
| L1 | 4 | 2.00 | 0.690 | 1.891 | 0.462 |
| L2 | 8 | 3.52 | 0.592 | 2.956 | 0.634 |
| L3 | 15 | 2.08 | 0.166 | 16.577 | 0.506 |
| L4 | 28 | 1.80 | 0.067 | 59.316 | 0.408 |
| L5 | 56 | 2.07 | 0.039 | 66.975 | 0.199 |

粗层 query 虽覆盖大量观测，但有效聚合约两个观测；content logits 的尺度远大于 relative-time logits。

### Full prediction decomposition

预测分解 artifact：`diagnostics/phystime_prediction_decomposition.json`，schema `phystime_prediction_diagnostic_v1`，真实 THUMOS `validation` GT 共 3325 instances。该分析使用每个最佳 checkpoint 的完整 post-NMS prediction JSON，不做自归一化。

| Head | All class-agnostic R@0.7 | All class-aware R@0.7 | Top-100 class-agnostic R@0.7 | Top-100 class-aware R@0.7 | `<1 s` class-aware R@0.5 | `<1 s` class-aware R@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-axis | 93.20 | 89.95 | 71.58 | 69.89 | 81.42 | 50.00 |
| physical-grid | 90.29 | 85.20 | 62.77 | 60.45 | 79.20 | 50.88 |
| PhysTime | 85.59 | 79.70 | 67.28 | 61.32 | 50.00 | 7.08 |

| Head | matched GT at IoU>=0.7 | start MAE (s) | end MAE (s) | normalized mean boundary error |
| --- | ---: | ---: | ---: | ---: |
| selected-axis | 2991 | 0.325 | 0.318 | 0.0752 |
| physical-grid | 2833 | 0.402 | 0.406 | 0.0905 |
| PhysTime | 2650 | 0.361 | 0.333 | 0.0799 |

这些数字区分了“命中后的边界精度”和“能否覆盖/排序到正确候选”：PhysTime 在已命中的高 IoU 样本上优于 physical-grid、接近 selected-axis，但总候选覆盖、类别排序和短动作召回明显更弱。

### Evidence verdict

- `PhysTime-AdaTAD 1.0` 当前实现没有胜过任一 sparse baseline，也没有达到 dense anchor，主假设为负。
- 该结果不能推出“physical-time detection 无效”，因为当前 PhysTime 同时更换了坐标表示、投影结构、时序上下文和模型容量。
- 证据范围仅为 THUMOS14 单协议、单种子 matched run；尚未达到 `paper_ready`。

### Diagnostic artifact hashes

下列 artifact 均位于 final run root 的 `diagnostics/`，SHA256 用于后续审计锁定：

| Artifact | Schema | SHA256 |
| --- | --- | --- |
| `phystime_attention_checkpoint_diagnostic.json` | `phystime_attention_checkpoint_diagnostic_v1` | `ae39e0158382332696cae7c269d26495e35d749dcd8aeb2a4d00b943a194d504` |
| `phystime_test_geometry_checkpoint_diagnostic.json` | `phystime_performance_geometry_diagnostic_v1` | `937433e72c4c2294f5f8b0df0f980ad744c649bf4b101bc4ecf465a6b0edb616` |
| `phystime_train_geometry_drop_diagnostic.json` | `phystime_performance_geometry_diagnostic_v1` | `70032400eb08fb17629e670b4141c8070328c64dd814b492cb09765556044b67` |
| `phystime_prediction_decomposition.json` | `phystime_prediction_diagnostic_v1` | `2e3a8a3a613036a9a6130656f2aee07ab49cd4f99d0145d0e6751731d4ac64bb` |

## G1b SDPQ 20-Epoch Medium Run

Code commit: `4a57577193c07cc90ac0867176aa79c76f637c36`.

Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_4a57577_gtboundaryfix_medium20_20260716_190900_0800`.

| Stage / variant | Job | Status | Evaluation epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| real THUMOS gate | `1167109` | completed / gate passed | NA | NA | NA | NA | NA | NA | NA |
| G1b SDPQ | `1167110` | completed / validation passed | 19 | 52.60 | 43.54 | 31.82 | 18.24 | 8.19 | 30.88 |

Artifact evidence:

- `PILOT_COMPLETE.json` and `evaluation_metrics.json` are present and finite.
- Final lightweight checkpoint `epoch_19.pth` is 132,393,307 bytes and passed online-state deserialization/finite-value validation. It did not retain `state_dict_ema`, while the epoch evaluation used EMA weights; therefore the saved prediction metrics are independently recomputable, but the exact evaluated weights cannot be replayed from this checkpoint.
- The run completed all 20 epochs without Traceback, OOM, non-finite loss, AMP skip, or GT boundary repair event.
- This run supports G1b trainability and continued learning beyond the old six-epoch pilots.
- It does **not** establish superiority: the available G1a selected-axis and physical-metric controls were trained for only six epochs under older commits. A same-commit, same-seed, same-schedule 20-epoch three-arm comparison is required before any method claim or full-train decision.
- The new matched suite closes the replay gap by retaining both online and EMA state dicts while still excluding optimizer and scheduler state.

## G1 Native-J192 Matched Three-Arm 20-Epoch Comparison

Code commit: `5e8a8219c27785c15d720c5ed3c6b37298a2a866`; Git tree: `7dfdf3d1c1e1c681a5df23f5916e2aa53de221ea`.

Run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_5e8a821_medium20_20260717_132000_0800`.

All arms use the same THUMOS data, K384/J192 observations, sampler, seed 42, 20-epoch schedule, optimizer, evaluator, and no feature interpolation.

| Variant | Job | Status | Best/final epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shared real gate | `1168484` | completed / gate passed | NA | NA | NA | NA | NA | NA | NA |
| selected-axis | `1168485` | completed / validation passed | 19/19 | 56.43 | 43.58 | 30.19 | 16.13 | 5.77 | 30.42 |
| physical-metric | `1168486` | completed / validation passed | 19/19 | 68.94 | 59.73 | 47.60 | 32.52 | 15.59 | 44.88 |
| G1b SDPQ | `1168487` | completed / validation passed | 19/19 | 52.60 | 43.54 | 31.82 | 18.24 | 8.19 | 30.88 |

Matched deltas in percentage points:

| Comparison | Avg-mAP | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| physical-metric minus selected-axis | +14.46 | +12.51 | +16.15 | +17.41 | +16.40 | +9.82 |
| G1b SDPQ minus selected-axis | +0.46 | -3.83 | -0.04 | +1.63 | +2.11 | +2.42 |
| G1b SDPQ minus physical-metric | -14.00 | -16.34 | -16.19 | -15.78 | -14.28 | -7.40 |

Artifact and stability evidence:

| Variant | First logged loss | Final logged loss | Peak memory | Predictions | Checkpoint contract |
| --- | ---: | ---: | ---: | ---: | --- |
| selected-axis | 1.7229 | 0.6651 | 3596 MB | 422000 | 401,895,677 bytes; online/EMA 499/499; SHA256 `d1fbaf990253bfd1ff384e724f395f5781639f47cba21180376495cefe3dbbca` |
| physical-metric | 1.7205 | 0.5879 | 3596 MB | 422000 | 401,895,677 bytes; online/EMA 499/499; SHA256 `72a3e001805209ea70b68dd72bce89d2d9344d098a153d874b67b08f08ab2ab5` |
| G1b SDPQ | 1.9153 | 0.9530 | 3134 MB | 420280 | 264,786,987 bytes; online/EMA 432/432; SHA256 `2debe94c006042d5268f0553d4104c90fbda8bd6c9cb10c58091d39232e9407a` |

- All three `MEDIUM_COMPLETE.json` artifacts report `validation_pass=true`; final checkpoints are lightweight, finite, replayable, and exclude optimizer/scheduler state.
- No Traceback, OOM, non-finite loss/gradient, AMP skipped step, checkpoint write failure, dependency failure, or GT boundary clamp/filter event was found.
- The physical-metric control is the clear matched-medium survivor. Its gain is large across every IoU threshold, so the evidence supports modeling ActionFormer assignment/regression in the real physical-time metric rather than selected rank.
- G1b SDPQ does not improve Avg-mAP meaningfully over selected-axis. Its small high-IoU gains coexist with lower mAP@0.3, consistent with sharper localization for some matched actions but weaker coverage/classification recall.
- This single-seed, 20-epoch THUMOS result is `matched-medium-supported`, not `paper_ready`. It does not justify claiming the SDPQ operator as the main contribution, and it does not by itself authorize an automatic 60-epoch run.

## G1 Native-J192 Matched Two-Arm 60-Epoch Validation (Running)

Code commit: `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`; Git tree:
`bddc9b9386604d00d213275a47ce7997b35d3f4c`.

Run root:
`/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1_matched_full60_0dc5851_20260718_112053_+0800`.

Both arms use K384/J192, seed 42, no feature interpolation, the same data,
sampler, backbone, optimizer, evaluator, and an exact 60-epoch cosine schedule.
Gate `1170945` passed; jobs `1170946/1170947` remain running.

First matched validation after epoch 41, as recorded in the immutable training
logs:

| Variant | Eval epoch | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selected-axis | 41 | 64.54 | 54.50 | 40.78 | 26.53 | 12.87 | 39.84 |
| physical-metric | 41 | 77.57 | 71.41 | 61.44 | 48.00 | 28.17 | 57.32 |
| physical minus selected | 41 | +13.03 | +16.91 | +20.66 | +21.47 | +15.30 | +17.48 |
| selected-axis | 43 | 64.83 | 55.15 | 41.58 | 27.13 | 13.38 | 40.41 |
| physical-metric | 43 | 77.29 | 70.93 | 61.57 | 48.16 | 28.01 | 57.19 |
| physical minus selected | 43 | +12.47 | +15.78 | +19.99 | +21.03 | +14.63 | +16.78 |
| selected-axis | 45 | 64.76 | 55.73 | 42.19 | 26.95 | 13.90 | 40.71 |
| physical-metric | 45 | 77.35 | 71.10 | 62.02 | 48.33 | 27.77 | 57.31 |
| physical minus selected | 45 | +12.59 | +15.37 | +19.83 | +21.38 | +13.87 | +16.60 |
| selected-axis | 47 | 64.85 | 55.85 | 42.15 | 26.99 | 14.22 | 40.81 |
| physical-metric | 47 | 77.36 | 71.27 | 62.09 | 48.65 | 27.99 | 57.48 |
| physical minus selected | 47 | +12.51 | +15.42 | +19.95 | +21.66 | +13.77 | +16.66 |
| selected-axis | 49 | 64.86 | 56.18 | 42.72 | 27.58 | 14.49 | 41.16 |
| physical-metric | 49 | 77.29 | 71.17 | 62.28 | 48.95 | 28.00 | 57.54 |
| physical minus selected | 49 | +12.43 | +14.99 | +19.56 | +21.38 | +13.50 | +16.37 |
| selected-axis | 51 | 64.83 | 56.35 | 42.96 | 27.58 | 14.51 | 41.24 |
| physical-metric | 51 | 77.26 | 70.95 | 62.40 | 48.75 | 27.82 | 57.44 |
| physical minus selected | 51 | +12.44 | +14.59 | +19.45 | +21.17 | +13.31 | +16.19 |
| selected-axis | 53 | 65.14 | 56.28 | 43.16 | 27.62 | 14.63 | 41.36 |
| physical-metric | 53 | 77.34 | 70.92 | 62.29 | 49.30 | 27.94 | 57.56 |
| physical minus selected | 53 | +12.20 | +14.64 | +19.13 | +21.69 | +13.31 | +16.19 |
| selected-axis | 55 | 65.19 | 56.46 | 43.07 | 27.70 | 14.77 | 41.44 |
| physical-metric | 55 | 77.31 | 70.88 | 62.70 | 49.20 | 28.21 | 57.66 |
| physical minus selected | 55 | +12.12 | +14.42 | +19.63 | +21.50 | +13.44 | +16.22 |

This is an interim, same-epoch comparison. It strengthens the survivor signal
from the 20-epoch matched run, but it is not the full60 result: epoch 59,
final online/EMA checkpoint validation, independent mAP recomputation, and both
`FULL_COMPLETE.json` artifacts are still pending. Current status remains
`experiment_running`, not `full60-single-seed-supported` or `paper_ready`.
The latest exact epoch-55 evaluator JSON SHA256 values are
`d163dedfda9b0bc353d176b3da1ae69b5f6b419985533752f5f721a13c0661f5`
for selected-axis and
`c235f9860828d801f9d415c3f5494ac9ba7a2a56da3bacef7fabd48a22e927f7`
for physical-metric.

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
