# Idea 总目录

## A. 实际推进过的主路线

| Idea ID | 核心 | 当前裁决 | 关键原因 |
| --- | --- | --- | --- |
| `idea:c3-coarse-actionness` | 低成本二分类 `p_action` | 保留资产 | 可作粗监督/基线，但不能独立代表 detector utility |
| `idea:paction-selector` | p_action 衍生学习式选帧 | 强 baseline | 比复杂 GAS-VT 更直接，但仍是 actionness/GT surrogate |
| `idea:gas-vt` | gap/value transport | 否定主线 | 非真正 sequential、budget shift、coverage 与 detector utility 错配 |
| `idea:lattice-center-radius` | uniform scaffold + local replacement/dilation | 诊断 | 几何可解释，但 scaffold/repair 可能是主要有效成分 |
| `idea:duca` | detector-utility acquisition plugin | 已 pivot | 工程合同强，研究中心仍近似 score+top-k+scaffold |
| `idea:duca-jct` | coarse probe、selector、detector 单作业协同 | 旧路线资产 | 修复了梯度和 optimizer，但效果、hard/soft 一致性与创新不足 |
| `idea:duca-must` | learned dynamic K | appendix/失败 | padded cap 不是真 variable compute，K 语义不统一且训练跳变 |
| `idea:cfpa-structured-policy` | exact-K/max-gap hard/soft 同构 | DUCA 修复资产 | 解决策略正确性，不改变 frame-selection 主问题创新风险 |
| `idea:trainfree-x3d` | frozen X3D action prior | appendix | 推理过慢、外部预训练与类别重叠、可能吞没节省 |
| `idea:slowfast-fast-prior` | frozen Fast pathway motion prior | appendix | 强运动先验但重、非独立预训练、易受相机运动干扰 |
| `idea:chronotransport-dcrt` | time x layer recompute/transport/reuse | 用户否决 | 接近 MoD/feature reuse，系统面和归因风险高 |
| `idea:phystime-tal-1` | continuous-time irregular TAD | 被 2.0 取代 | support/time/hazard/consistency 定义不够严格且撞近邻 |
| `idea:phystime-tad-2` | support-integrated physical-time detector | 长期目标 | 独立 detector，直接解决不规则时间几何 |
| `idea:phystime-adatad-1` | raw-video matched head isolation | 当前唯一主线 | 最小公平实验，直接验证物理时间头价值 |

## B. 2026-07-10 发散产生的 23 个候选

| ID | 名称 | 问题定义 | 裁决 |
| --- | --- | --- | --- |
| C1 | ChronoTransport / DCRT | time x layer 重算、传输或复用 | Pro 首推，后被用户否决为当前主线 |
| C2 | CoDeR-TAL | codec decode 与定位 distortion 联合优化 | 高风险备线；依赖 codec/API/hardware |
| C3 | ACTAL / Compute-to-Resolve | streaming 中购买计算缩小 endpoint belief | 高风险新任务；偏离离线 TAD |
| C4 | PhysTime-TAL | 真实 timestamp 上建模不规则观测 | 被选择并演化为 PhysTime-TAD 2.0 |
| C5 | NyquistBound | 用 boundary spectrum/anti-alias 约束 sensing | 并入 PhysTime observability/采样协议，不独立成文 |
| C6 | ProposalRefine | 预算分配给未解决 proposal | 与已有 refinement 路线碰撞强，淘汰 |
| C7 | SparseNeck | dense backbone、稀疏 neck/head | 近似 token/head pruning，淘汰 |
| C8 | LayerClocks | 不同 backbone layer 异步更新 | 并入 ChronoTransport 机制 |
| C9 | ConformalCache | cache 风险越界才刷新 | 并入 ChronoTransport 风险校准 |
| C10 | PhaseState | event-driven phase state update | thesis 不够聚焦，暂存 |
| C11 | TailRisk | CVaR 保护短动作/高-IoU tail | 作为目标项并入其他方法，不独立成文 |
| C12 | UniTVI | 跨 TAD/TAS/AQA 的 temporal information value | 范围过大，淘汰 |
| C13 | ParetoCompiler | 依硬件 profile 编译多资源 policy | 并入 No-Free-Frames/CoDeR，非当前主线 |
| C14 | GOPScheduler | codec reference graph 决定刷新/partial decode | 并入 CoDeR-TAL |
| C15 | DenseLatent | 稀疏观测但显式重建 dense latent | 与 feature propagation/latent reconstruction 碰撞密集 |
| C16 | MarginalCompute | counterfactual 衡量计算块真实价值 | 强诊断工具，非独立方法 |
| C17 | No-Free-Frames | 全栈效率 benchmark | 必要伴随线，不单独作为当前方法 |
| C18 | RetroZoom | streaming ring buffer 触发回看 | 并入 ACTAL；不适合当前离线任务 |
| C19 | MoDepth-TAD | 不同时间点采用不同 temporal depth | 接近 MoD，淘汰 |
| C20 | EnergyMemory | energy-risk 优化 streaming memory | 系统面过大，淘汰 |
| C21 | EdgeSplit-TAL | uncertainty-triggered edge/cloud 切分 | 系统面过大，淘汰 |
| C22 | Detector-aware SOFT-TopK | OT 可微 top-k | 正是应停止的旧结构，淘汰 |
| C23 | Uncertainty-K | entropy 直接决定 K | 过于简单且不是真实 variable execution，淘汰 |

## C. 明确停止重复的 idea 形态

- 新的 frozen actionness prior；
- 新的 boundary/actionness/uncertainty 权重组合；
- 新的 differentiable top-k；
- 更多 hole/gap/radius 修补；
- offline ledger 作为论文中心；
- 不实现 variable execution 的 dynamic budget；
- 只在 selected-axis 上继续叠 remap patch；
- 只报 backbone FLOPs 或 selected count；
- 把 one-step gradient 当作 end-to-end 论文证据。
