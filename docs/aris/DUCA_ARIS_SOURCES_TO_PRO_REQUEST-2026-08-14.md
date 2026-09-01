# DUCA ARIS Sources-to-Pro Request — 2026-08-14

- 类型：本地 Sources-to-Pro 请求（无 browser、无 Pro 调用、无角色 dispatch、无执行）
- 状态：PREPARED_ONLY / BLOCKED_PRE_RESULT（本轮不发送）
- 目标：fresh exact-Project Pro 终态决策

---

## 供 Pro 阅读的本地材料（最多两份）

1. `docs/aris/ARIS_CPR_PLAN-2026-08-14.md`
   — 三条非等价动态预算路线（A 原始动态选帧恢复 / B 层级 Dynamic DUCA / C 价格校准密度）、
   条件推荐、P 阶段 Builder 包、R 阶段角色链与门、以及 dirty-base / non-DUCA-SHA 证据边界。

2. `docs/aris/ARIS_DECISION_LOG-2026-08-14.md`
   — 环境/边界裁决、代码表面 parent-fidelity 分类、科学边界（sealed U/O/R 终态负证据保留）、
   路由裁决与不确定性。

（参考但非必需：`research-wiki/ideas/duca-rime.md` 与 `research-wiki/experiments/duca-dynamic-k-rime-oracle.md`，为 Route B 的既有讨论/规划来源。）

---

## 冻结问题（仅一个）

**授权有界动态预算 Builder → Critic → Evaluator PRE_RUN 序列，还是 STOP/REVISE？**

具体裁决对象（Route B，层级 Dynamic DUCA）：
- 外层 = per-video/window K 的 train-only hard detector utility − paired-boundary/high-IoU risk − λ·measured cost；
- 内层 = 复用 frozen `density_decode`（fixed-K 有界密度分位解码器）作 per-K exact-K 物理帧传输 + pre-NMS 物理时间逆映射；
- 前置门 = O1 动态 headroom / O2 decoder-family regret（strict nested vs independent vs weak-overlap）/ O3 `G_rank` / O4 pair-risk；
- 训练合同 = 总计 6000 次成功 detector update，development seed 排除，official validation/test 防火墙。

**期望裁决输出**：
- `AUTHORIZE_PRE_RUN`（进入有界 Builder→Critic→Evaluator 静态链，仍不授权数据/GPU/Slurm/training/metrics），或
- `STOP`（终止本动态预算路线，保留负证据），或
- `REVISE`（给出唯一、有界、可检验的路线/合同修正；不得第三次修正 sealed U/O/R 表面）。

**前置约束（本轮与裁决前均不改变）**：
- sealed U/O/R execution-surface 终态失败仅作负证据；
- Fixed K 仅作 baseline/attribution/fallback；
- 所有阈值在 clean baseline 方差与视频级功效后、正式结果前一次性冻结。
