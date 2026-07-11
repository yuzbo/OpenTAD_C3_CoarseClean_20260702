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
| `exp:phystime-adatad-k384` | raw-video 三头 matched comparison | formal run failed | invalid run | 三头 evaluator path 错误；PhysTime 另有持续 NaN；无 mAP，修复后必须整套重跑 |

## 证据等级定义

1. `contract test`：张量、坐标、泄漏和预算合同。
2. `one-step proof`：真实/缩放模型可 forward/backward，梯度与 optimizer 覆盖成立。
3. `diagnostic run`：解释失败机制，不能直接支持主张。
4. `matched full run`：同 commit、同数据、同采样、同 schedule 的可比 mAP。
5. `claim evidence`：matched full run + 多 seed/置信区间 + 高-IoU/成本/泛化审计。

当前没有达到第 5 级的 PhysTime 实验。
