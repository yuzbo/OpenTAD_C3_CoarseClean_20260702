# Research Wiki Index

## Active Experiment (2026-07-18)

- `exp:phystime-g1-matched-full60`: gate passed; selected-axis job `1170946`
  and physical-metric job `1170947` are running. Metrics remain NA.

更新时间：2026-07-18

这不是论文草稿，而是本项目的长期研究记忆。所有后续方法修改、实验部署和论文主张，必须先读取本页指向的当前方向、决策和失败记录。

## 必读入口

1. [压缩上下文](query_pack.md)：供新 agent 或新一轮工作首先读取的 8000 字以内摘要。
2. [反重复契约](anti_repetition.md)：禁止回退路线、证据混级和已知数值错误。
3. [当前唯一方向](current_direction.md)：现在究竟要实现什么，哪些内容尚未实现。
4. [决策台账](decision_register.md)：每次路线选择、否定理由和恢复条件。
5. [经验与禁区](lessons.md)：已经用代码、实验或评审代价换来的教训。
6. [讨论时间线](timeline.md)：C3、GAS-VT、DUCA、PIVOT/ChronoTransport 到 PhysTime 的演进。
7. [Idea 总目录](idea_catalog.md)：所有主要 idea、两轮 23+24 个发散候选及其去留。
8. [实验台账](experiment_register.md)：已完成、取消、诊断、待部署实验的统一分类。
9. [证据来源](source_map.md)：原始附件、评审记录、代码提交与文献来源。
10. [记忆维护协议](memory_protocol.md)：怎样保证 Wiki 不再次失效。
11. [三条路线完整档案](routes/index.md)：DUCA、ChronoTransport、PhysTime 的讨论、代码、实验和裁决。
12. [讨论覆盖矩阵](discussion_coverage.md)：逐主题说明内容归档位置与原始来源。
13. [原始讨论归档](sources/README.md)：主任务用户原文与跨代理近期记录。
14. [跨 Worktree 库存](worktree_inventory.md)：所有历史 checkout、分支 HEAD 与证据边界。
15. [原始来源登记](source_registry.md)：本轮审查、正式结果、远端作业、代码与外部文献锚点。

## 当前实体

### Ideas

- `idea:c3-coarse-actionness`：低成本粗二分类产生 `p_action`。
- `idea:paction-selector`：PAction 学习式选帧基线。
- `idea:gas-vt`：gap/value-transport 路线。
- `idea:lattice-center-radius`：move25/move50 与膨胀/半径诊断。
- `idea:duca`：detector-utility-calibrated acquisition。
- `idea:duca-jct`：单作业协同训练 DUCA。
- `idea:duca-must`：动态预算 MUST。
- `idea:cfpa-structured-policy`：exact-K/max-gap 同构结构化策略。
- `idea:trainfree-x3d`：冻结 X3D 动作先验。
- `idea:slowfast-fast-prior`：冻结 SlowFast Fast 侧先验。
- `idea:chronotransport-dcrt`：按 time x layer 重算/传输/复用。
- `idea:cvcr-tad`：counterfactual value-of-compute routing。
- `idea:bcft`：boundary-certified feature transport。
- `idea:codetad`：GOP-dependent partial decode。
- `idea:coder-tal`：codec-native rate-distortion TAD。
- `idea:actal`：streaming compute-to-resolve。
- `idea:no-free-frames`：全栈效率审计协议。
- `idea:phystime-tal-1`：第一版连续/物理时间 TAD 规格。
- `idea:phystime-tad-2`：support-integrated physical-time detector。
- `idea:phystime-adatad-1`：已完成的 raw-video AdaTAD 三头负基线。
- `idea:sm-ptaf`：native tubelet support-measure physical-time ActionFormer 重建候选，当前仅 `designed`。

### Experiments

- `exp:c3-stage1-selector-matrix`
- `exp:move25-move50-geometry`
- `exp:duca-joint-old-commits`
- `exp:x3d-trainfree-grid`
- `exp:slowfast-fast-diagnostic`
- `exp:duca-repaired-final`
- `exp:duca-cost-structural-audit`
- `exp:chronotransport-engineering-track`
- `exp:phystime-feature-track`
- `exp:phystime-adatad-k384`
- `exp:phystime-performance-drop-diagnosis`
- `exp:phystime-g1a-native-j192`
- `exp:phystime-g1b-sdpq-medium20`
- `exp:phystime-g1-matched-medium20`
- `exp:phystime-g1-matched-full60`

### Papers

- `paper:zhang2022_actionformer`
- `paper:liu2024_adatad`
- `paper:kim2024_te_tad`
- `paper:zeng2024_temporal_robustness`
- `paper:shukla2021_mtan`
- `paper:sun2026_liquidtad`
- `paper:wang2022_rcl`

### Claims

当前不创建 claim 实体。研究 Wiki 约定 claim 只能由严格的 proof/claim audit 产生；目前只有候选主张和实验门槛，记录在当前方向与 gap map 中，不能冒充已证实结论。

## 当前状态一句话

PhysTime-AdaTAD 1.0 的 K384 三头 full run 已冻结为负基线；same-commit selected-axis / physical-metric / G1b SDPQ 三臂 20轮对照已全部完成。physical-metric `44.88%` 明显胜 selected-axis `30.42%`，G1b `30.88%` 未证明结构优势。当前结论是 physical-time metric 获得 matched-medium 支持；多 seed、完整 schedule、机制分解和第二数据集完成前仍不是 paper-ready。
