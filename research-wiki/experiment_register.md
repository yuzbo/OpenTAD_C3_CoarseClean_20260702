# 实验台账

实验数值唯一来源仍是 `docs/evaluation/results.md` 或各正式 run artifact。本页只记录实验身份、证据等级和裁决，避免把旧 run 当成当前结果。

| Experiment ID | 路线 | 状态 | 证据等级 | 结论用途 |
| --- | --- | --- | --- | --- |
| `exp:c3-stage1-selector-matrix` | PAction/GAS-VT/uniform/random 等 | 历史完成/部分完成 | baseline/diagnostic | 证明 actionness 选帧可行，暴露 coverage 与 high-IoU 错配 |
| `exp:move25-move50-geometry` | lattice replacement/dilation | 历史诊断 | geometry diagnostic | 证明聚集和位置偏移并存；不作为主方法 mAP 证据 |
| `exp:duca-joint-old-commits` | 旧 DUCA fixed/MUST/JCT | 取消、失败或旧 commit | diagnostic only | 不得进入最终论文主表；用于训练崩溃和 contract 诊断 |
| `exp:x3d-trainfree-grid` | X3D grid/export/downstream | 终止 | appendix diagnostic | 运行过慢，不能支持低成本主模块；不再排 dense X3D |
| `exp:slowfast-fast-diagnostic` | Fast-side frozen prior | 代码/诊断候选 | appendix only | 只验证 motion prior，不代表主方法 |
| `exp:duca-repaired-final` | CFPA/full-window repaired DUCA | 软件 gate 通过，正式效果未建立 | engineering proof | 128 tests、official one-step 等证明合同；无新 full-run 不能证明论文效果 |
| `exp:duca-cost-structural-audit` | DUCA full-stack cost/source parity | 工程完成、科学裁决未完成 | engineering audit | `a5e1774` 分支补齐成本/结构事实，但无决定性结果 |
| `exp:chronotransport-engineering-track` | Chrono Stage-A/replay/formal Stage-B | P3 负结果后暂停 | negative scientific gate | 真实链路可运行，但 risk 排序/尺度与 feature transport gate 失败；Stage C/P5 未解锁，commits 未推远端 |
| `exp:phystime-feature-track` | I3D feature-token PhysTime pilots | 已取消 | software/feature diagnostic | 不得作为 raw-video PhysTime-AdaTAD 证据 |
| `exp:phystime-adatad-k384` | raw-video 三头 matched comparison | completed / negative | matched full run | `3ac93a1` 三头完成并由只读 checkpoint replay 复现；PhysTime 1.0 未胜两个 sparse controls |
| `exp:phystime-performance-drop-diagnosis` | 容量、几何、attention、预测分解 | completed | diagnostic + integrity audit | 排除 evaluator/训练崩溃；确认架构混杂、query 尺度、粗层聚合、候选密度和短动作问题 |
| `exp:phystime-g1a-native-j192` | selected-axis / physical-metric native-J192 controls | six-epoch pilots completed | early matched diagnostic | 两臂差异很小，但轮数过短；不能与 G1b medium run 横比 |
| `exp:phystime-g1b-sdpq-medium20` | support-decoupled physical query sparse head | completed | medium-run trainability evidence | 20轮稳定闭环成立；优越性仍需 same-commit 三臂 20轮对照 |
| `exp:phystime-g1-matched-medium20` | selected-axis / physical-metric / G1b SDPQ | completed | matched medium evidence | physical-metric 44.88 明显胜 selected-axis 30.42；G1b 30.88 未证明结构优势 |
| `exp:phystime-g1-matched-full60` | selected-axis vs physical-metric, K384/J192 | completed / validation passed | final epoch-59 Avg-mAP `41.28/57.57`, delta `+16.29`; finite online/EMA checkpoints pass | Full60 single-seed support for physical-time metric; not paper-ready |

## 证据等级定义

1. `contract test`：张量、坐标、泄漏和预算合同。
2. `one-step proof`：真实/缩放模型可 forward/backward，梯度与 optimizer 覆盖成立。
3. `diagnostic run`：解释失败机制，不能直接支持主张。
4. `matched full run`：同 commit、同数据、同采样、同 schedule 的可比 mAP。
5. `claim evidence`：matched full run + 多 seed/置信区间 + 高-IoU/成本/泛化审计。

当前没有达到第 5 级的 PhysTime 实验。

下一阶段不是继续添加新结构，而是复现并分解 physical-metric survivor。代码 commit `5e8a821` 的三臂 20轮对照已全部完成且 artifact 通过；当前仍只有单 seed、单数据集、medium schedule，不能创建论文 claim，也不能自动启动 60轮 full train。
