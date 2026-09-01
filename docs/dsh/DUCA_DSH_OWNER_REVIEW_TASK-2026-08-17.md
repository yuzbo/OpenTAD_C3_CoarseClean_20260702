# DUCA 外部正式审查任务（只读）

你是独立的外部审查者。请只读审查当前 DUCA 项目，不修改任何文件，不启动训练、评估、下载、远端连接、浏览器或 GPU 任务。最后直接输出一份完整、面向外部读者的中文审查报告；不要把报告写回仓库。

## 审查对象与证据边界

- 协调根 `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702` 是脏的研究记忆根，`a6bdc084...` 属于 SparseHead 污染，不得当作 DUCA 代码或性能身份。
- 当前可审查的 DUCA 方法实现树是 `C:/Users/skywalker/.codex/worktrees/duca-indirect-dynamic-20260817`，冻结提交 `6125654b946cc30c614428ce1141f1903b015867`，应先核对其 HEAD 与 clean 状态。
- 首先阅读根目录的 `RTK.md`、`PAPER_PROGRESS.md`、`research-wiki/query_pack.md`、`research-wiki/anti_repetition.md`、`research-wiki/decision_history.md`、`research-wiki/DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`、`research-wiki/duca_model_version_registry.md` 和 `research-wiki/log.md`；随后直接阅读该冻结工作树中涉及 selector、配置、启动器和 focused tests 的源码。
- 历史约 65 的数值并非当前干净 official AdaTAD 结果：必须分清固定 K、物理网格、80/90 epoch、direct controller、诊断和正式对照，不能将它们升级为本方法性能。
- 共享的原始 AdaTAD official baseline 由 ZoomToken 负责人只执行一次；DUCA 不得重复训练或把既有 `66.xx` matched-source 数字伪装成该 shared official baseline。当前未绑定 released checkpoint evaluation，故没有已通过的 official dense 数字。

## 方法的硬语义

DUCA 主线不是小模型直接学习帧索引。低成本 scout 在训练期学习逐帧 `0/1` 动作/背景语义和边界重要性；确定性 acquisition 仅从这些预测及冻结规则导出重要性、物理时间位置与逐视频/窗口动态 outer-K。选择后的每帧必须带原始 timestamp，重型 VideoMAE/AdaTAD 的 decode、IoU 与 NMS 不得把稀疏序号伪装成均匀时间。动态 outer-K 是主张核心；fixed-K 只能是基线、归因控制或失败回退；直接 index selector 只能是消融。

预注册的首个正式矩阵是：official dense、native uniform fixed-K、间接 actionness-only fixed-K、间接 actionness+boundary fixed-K、同一间接预测器的 dynamic outer-K、direct-selection ablation。六臂须共享官方数据/视频级 split、detector、loss、NMS、evaluator、updates、seeds 与完整成本定义。

## 已知当前实现事实，要求逐行复核而非照抄

冻结工作树实现了名为 `DucaIndirectSemanticAcquisition` 的静态包及与 PC-OT-MRAS pre-backbone selector 的接入。已知的真实 PRE_RUN 阻断是：batch 级 `dynamic_budget_meta` 字典在 selector 中被按样本索引，无法为每个视频封存 requested/effective/executed K；请确认具体文件、符号、行号和实际后果。先前“`forward_train` 没有把 GT 传给 selector/semantic loss”的报告已被直接重读推翻；如源码同样显示 `gt_segments` 已传入，明确写出该旧指控为误报。不得把静态测试、Python 编译或历史远端日志说成真实视频效能证据。

此前 dynamic-B physical transport 路线的 locality 缺陷属于旧实现/审查链；请区分它与本次间接语义 acquisition 包，不能跨身份套用结论。

## 需要回答的问题

请给出清晰中文报告，先给结论，后给可复查的 `文件:行号`/commit 证据，并包含：

1. 当前唯一可继续验证的科学路线是否忠实于“动作性与边界预测 → 确定性间接选帧 → 动态预算”；它的可证伪价值和固定 K/direct policy 的正确角色。
2. 历史中最关键的正、负、空和协议不匹配发现；尤其说明 65.696、80/90 epoch 65.xx、旧 direct controller 与当前方法分别意味着什么。
3. 冻结实现是否真的在 train-only 侧形成 `0/1` actionness 与 boundary target，并通过确定性而非直接 index policy 派生分数、位置与动态 K；是否存在 padding/metadata 假动态或 timestamp/GT leakage 风险。
4. 与官方 OpenTAD/AdaTAD 的公平性差异。明确哪些是已核验的 source/config/evaluator 差异，哪些仍缺 shared official baseline receipt，不能猜测 2–3 个点的来源。
5. 按严重程度列出阻断问题、实际后果、最小修复与验证方式。必须独立判断 `dynamic_budget_meta` 是否为实质 PRE_RUN blocker，不能只重复本任务文字。
6. 给出 `PASS`、`CONDITIONAL_PASS` 或 `FAIL` 的明确审查终态，并区分方向、实现与经验效能三个层级。
7. 共享 official baseline 仍由其他负责人准备时，DUCA 可以并行完成的最小实际工作；以及下一项完整真实数据官方对比的准入条件。
8. 对未来完整训练的 checkpoint 合同：除非未改 official recipe 更频繁，5 个 epoch 一个可恢复 `.pth`；模型选择仍是预注册 final/final-EMA；至少保留最近 3 个恢复点，恢复必须包含 model、optimizer、scheduler、scaler、epoch/update 与 RNG 状态。

不得自行选新科研路线、声称实验成功或请求人工选择。最终报告必须仅基于你实际读取到的材料，并清楚标识“设计/静态实现/静态测试/真实实验”的证据等级。
