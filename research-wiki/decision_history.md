---
type: decision_history
updated: 2026-07-11
---

# 路线演化与选择理由

## 1. C3 粗分类出发点

最初洞察：低成本模型即使不能直接精确选帧，也可能可靠地区分动作/背景并暴露
状态变化。选择“粗分类 → 间接选择”，而不是让小模型直接承担 TAD。

保留原因：动作性提供低成本候选证据。限制：动作内部高分不等于边界有用。

## 2. PAction 与严格 ledger

建立 train-time p_action 监督、strict budget、no-leak ledger 和 AdaTAD 消费链路。
它证明工程路径可行，并成为固定预算安全锚点。

未选为最终方法：多阶段、detector-unaware、硬 gap decoder，创新性与联合优化不足。

## 3. GAS-VT

尝试学习 frame value、边界覆盖、预算与 gap，再生成 value-transport ledger。

降级理由：apply-time 不是忠实 sequential value transport；训练/应用特征存在风险；
多重覆盖损失和 repair 可能推向均匀；PAction 的简单策略反而更强。保留为诊断 baseline。

## 4. Detector-aware teacher 与 TrueTime

为了让选择服务 TAD，引入 train-only dense detector utility、selected-axis 到原时间映射
和 detector-loss gradient proof。

未作为最终三阶段方案：teacher 预提取仍然多阶段；toy gradient 和 wrapper 不足以证明
联合检测收益；selected-axis 的不等物理间隔语义尚未解决。

## 5. Lattice、move25/move50 与 learned radius

为了避免大 gap，用 uniform scaffold/local replacement 和边界附近膨胀保护覆盖。

降级理由：它是工程启发式；repair/膨胀消耗预算并可掩盖分数偏移；move 分析显示聚集
存在但位置仍可能偏离边界。只保留为几何诊断和对照。

## 6. DUCA 插件化

从 ledger 转向 detector 前 forward 内生成选择：coarse probe、selector、hard selected
positions、official detector backend、teacher-free inference。固定 K 是归因锚点，MUST
尝试动态预算。

选择理由：统一模型接口，允许 detector 梯度反馈，避免测试时 ledger/cache。

## 7. X3D / SlowFast frozen prior

尝试 train-free 视频动作先验，验证视频预训练特征是否比图像 MobileNet 更懂动作。

降级理由：密集视频推理过慢，可能吞掉 heavy backbone 节省；Kinetics 预训练还带类别
先验和 overlap 风险。仅作为 frozen-prior diagnostic/upper baseline。

## 8. Transition/boundary-first 修正

move 分析和早期 DUCA 低性能表明 actionness coverage 再次主导。于是 selector 改为读取
`delta_p_action`、绝对变化、不确定性和粗分类隐藏特征，以 transition/boundary/utility
优先，actionness 降为小权重辅助。

这是当前必须保留的设计初心。

## 9. Progressive joint 与 structured bridge

为协调 probe、selector、detector，采用 progressive joint schedule：先稳定 coarse/boundary
目标，再逐步打开 detector gradient；detector loss 始终训练 backend。structured
zero-forward bridge 保持 hard forward，同时向 selector 传递近似梯度。

未闭环问题：梯度非零不等于 hard utility 对齐，必须做 finite-difference one-swap 审计。

## 10. Offline full-window 语义纠正

严厉审查曾把目标误认为 causal online acquisition，提出 prefix invariance/CFPA。用户明确
纠正：项目从未要求流式因果，完整窗口可见是允许的。CFPA 只保留给未来 streaming 版本。

## 11. 70aa069 冻结与 a5e1774 审计

`70aa069` 修复 full-window structured joint training 并启动 fixed-384 full run。
ResearchClaw 审查指出总成本、hard/soft utility 与 selected-axis geometry 是决定性门槛。
结论不是立即停止，也不是继续堆模块，而是冻结候选并裁决。

`a5e1774` 修复完整成本统计和官方 AdaTAD 表述：base/head 配置一致，但 detector 源码、
输入长度、GT 坐标与后映射并非完全官方不变。

## 12. 当前选择

主科学问题保持“任务感知时序去冗余并保护高 tIoU”。当前先完成 DUCA fixed-384 的
决定性证据；MUST、physical-grid、X3D 主方法和更多 loss 暂停。ChronoTransport 与
PhysTime 是并行新假设，必须独立记账，不能用来改写 DUCA 结果。
