# 讨论覆盖矩阵

## 覆盖口径

“全面纳入”指：每个重要问题、idea、选择/否定理由、实现事实、实验分类和恢复条件都有稳定文档位置，并能回溯到原始附件、仓库文档或 Git commit。路线档案不逐字重复数十万字原始回复，也不复制实验数值；可获得的任务原文保存在 `sources/`，其余原文由 `source_map.md` 的路径、SHA 和 `docs/methods/reviews/*.raw.txt` 固定。

主任务用户原文和相关代理近期导出已进一步纳入 `sources/`；全部本地 checkout 的审计 HEAD 见 `worktree_inventory.md`。

## DUCA 覆盖

| 讨论主题 | Wiki 位置 | 主要来源 | 状态 |
| --- | --- | --- | --- |
| C3 coarse actionness 初心 | `routes/duca-complete-record.md` §1-2 | C3 docs、线程讨论 | 完整 |
| PAction 为什么强、为何不是最终方法 | DUCA §2.1 | GAS/PAction diagnosis | 完整 |
| GAS-VT plateau、非 sequential、budget shift | DUCA §2.2 | 07-07 raw/absorption | 完整 |
| detector teacher utility 与 signed risk | DUCA §2.3 | detector utility review | 完整 |
| plugin vs complete detector | DUCA §2.4 | 86b/d008 attachments | 完整 |
| offline/full-window vs online/streaming | DUCA §2.5 | 88e audit/thread | 完整 |
| official ASFormer 与 hidden features | DUCA §3.1-3.2 | a885/a1d/391 attachments | 完整 |
| transition/boundary-first selector | DUCA §3.2 | 544eca6 review | 完整 |
| center/radius、max-gap 10/15、scaffold repair | DUCA §3.3/6 | move reviews、7e3a508 | 完整 |
| ST/soft-resample/structured hard-soft bridge | DUCA §3.4 | 13761fa、CFPA audit | 完整 |
| official AdaTAD 是否真的未修改 | DUCA §3.5 | a5e1774 structural audit | 完整 |
| 三阶段为何被建议、为何用户否定 | DUCA §4 | 多轮 thread/attachments | 完整 |
| JCT curriculum/optimizer-step schedule | DUCA §4.2 | JCT deployment docs | 完整 |
| loss duplication、optimizer 漏参、uint8、DDP | DUCA §4.3 | c8a/repair status | 完整 |
| fixed budget curve 与 MUST dynamic | DUCA §5 | 60cb/0ce/MUST code | 完整 |
| actionness vs boundary、move 聚集偏移 | DUCA §6 | a1d/geometry reviews | 完整 |
| X3D train-free 的定义、速度与终止 | DUCA §7.1 | 1705/0ce/thread | 完整 |
| SlowFast Fast 先验 | DUCA §7.2 | f705dda/88e50b1/thread | 完整 |
| 核心代码 API 和 configs/tests | DUCA §8 | 当前源码 | 完整 |
| 关键 commit 演化 | DUCA §9 | git log | 完整 |
| 远端作业谱系与 evidence category | DUCA §10 | thread/heartbeats/docs | 完整 |
| ResearchClaw 第二套 24 ideas | DUCA §11 + idea catalog | a5e1774 raw review | 完整 |
| full-stack cost 与 source parity audit | DUCA §12 | a5e1774 branch | 完整 |
| 为什么 pivot、保留什么、禁止什么 | DUCA §13-16 | 全部裁决 | 完整 |

## ChronoTransport 覆盖

| 讨论主题 | Wiki 位置 | 主要来源 | 状态 |
| --- | --- | --- | --- |
| PIVOT/ChronoTransport/DCRT 的来源 | Chrono §1 | 1fc attachment/thread | 完整 |
| 是否是插件 | Chrono §2 | thread + design spec | 完整 |
| 离线还是在线 | Chrono §3 | corrected design spec | 完整 |
| 以帧/16帧/token/layer 为单位 | Chrono §4 | 2bb3456/627c5ab | 完整 |
| 768 detector vs 384 tubelets | Chrono §4 | 78d4c00 spec | 完整 |
| RECOMPUTE/TRANSPORT/HOLD | Chrono §5 | actions/cache/runtime code | 完整 |
| 哪些计算 dense、哪些动态 | Chrono §6 | design spec | 完整 |
| counterfactual regret/risk/cost | Chrono §7 | risk/scheduler/cost code | 完整 |
| Stage A/replay/formal B/C/P5 | Chrono §8 | implementation plan | 完整 |
| 实际源码、configs、tests | Chrono §9 | `git ls-tree 92029ea` | 完整 |
| formal P3 负结果与停止 | Chrono §8/11 | `92029ea` implementation record | 完整 |
| 15 个本地提交未推远端 | Chrono §0/10 | branch audit | 完整 |
| 工程证据与科学证据边界 | Chrono §11 | implementation record | 完整 |
| kill criteria | Chrono §12 | spec | 完整 |
| 与 MoD/feature reuse 接近性 | Chrono §13 | thread/external audit | 完整 |
| 用户最终否定理由 | Chrono §14 | thread | 完整 |
| 与 PhysTime 关系、恢复条件 | Chrono §15-17 | final decision | 完整 |

## PhysTime 覆盖

| 讨论主题 | Wiki 位置 | 主要来源 | 状态 |
| --- | --- | --- | --- |
| 从 DUCA selected-axis 风险转向新 detector | PhysTime §1 | thread/DUCA audits | 完整 |
| mTAN/TE-TAD/FrameDrop/TRC/LiquidTAD 碰撞 | PhysTime §2 | web + e8d attachment | 完整 |
| PhysTime-TAL 1.0 架构与损失 | PhysTime §3.1-3.2 | 8abc521/spec | 完整 |
| 1.0 八项概念错误与 HOLD | PhysTime §3.3 | e8d attachment | 完整 |
| 2.0 input/support/measure/query/head/loss | PhysTime §4 | 2.0 spec | 完整 |
| 当前 geometry/projection/head/detector/transforms | PhysTime §5 | current source | 完整 |
| focused tests 证明范围 | PhysTime §5 | tests/test_phystime_* | 完整 |
| feature-token track 与 jobs | PhysTime §6 | manifest/results/thread | 完整 |
| 为什么取消特征下载/训练 | PhysTime §6 | user decision/results | 完整 |
| raw-video AdaTAD 研究问题 | PhysTime §7.1 | 9266ebc | 完整 |
| 相同无学习无 GT 采样 | PhysTime §7.2-7.4 | user decision/spec | 完整 |
| selected-axis/physical-grid/PhysTime 三头 | PhysTime §7.3 | AdaTAD spec | 完整 |
| raw seconds/support metadata | PhysTime §7.5 | design/plan | 完整 |
| official VideoMAE adapters/optimizer | PhysTime §7.6 | design/plan | 完整 |
| primary 不加 consistency | PhysTime §7.7 | design decision | 完整 |
| 为什么可转原帧号但不能 selected rank | PhysTime §8 | latest thread decision | 完整 |
| 已实现/未实现清单 | PhysTime §9 | current tree/commit | 完整 |
| Phase 0/1/2 实验 | PhysTime §10 | plan | 完整 |
| 成功/降级/停止条件 | PhysTime §11 | spec/review | 完整 |
| 与 DUCA/Chrono 的关系 | PhysTime §12-13 | final synthesis | 完整 |
| 禁止重复与唯一下一步 | PhysTime §14-15 | decisions | 完整 |

## 刻意去重或单一来源保留的内容

1. 主任务用户侧原文已经逐条归档；重复的“检查远端实验”所对应的瞬时输出没有在路线档案中重复展开，其 job 与状态转折按实验谱系归档。
2. 原始回复中的大段建议代码没有再次复制到路线档案；代码意图、缺陷、commit 和最终实现 API 已归档，可获得原文保留在 `sources/` 或由 SHA/path 固定。
3. 实验 mAP 曲线和数值不进入 Wiki，继续服从 `docs/evaluation/results.md` 单一数字来源原则。

这三项是刻意的去重与单一来源设计，不是讨论遗漏。
