# Research Wiki Log

本文件只追加，不回写历史。

- 2026-07-11：初始化 research-wiki。
- 2026-07-11：清点 C3/PAction/GAS-VT、DUCA、MUST、X3D/SlowFast、PIVOT/ChronoTransport、PhysTime 的仓库文档、原始附件、提交历史和实验记录。
- 2026-07-11：建立当前方向、决策台账、时间线、经验禁区、gap map、idea catalog、experiment register 和 query pack。
- 2026-07-11：将 feature-token PhysTime 轨道标记为取消/诊断，将 PhysTime-AdaTAD K384 三头比较标记为当前唯一执行主线。
- 2026-07-11：明确秒坐标可转换回原视频帧号，但禁止 selected-rank GT/预测坐标。
- 2026-07-11：声明当前无 claim 实体；任何论文主张必须等待 matched full run 与 result-to-claim 审计。
- 2026-07-11：完成首轮 lint：31 个实体、10 个 gaps、48 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3348 字符。
- 2026-07-11：第二轮完整性审计发现并修正遗漏：ChronoTransport 在本地分支已实现到 `92029ea`，formal P3 science gate 为负且 15 commits 未推远端；DUCA 另有 `a5e1774` full-stack/structural audit 分支。
- 2026-07-11：新增 DUCA、ChronoTransport、PhysTime 三份完整路线档案、逐主题覆盖矩阵，以及 ResearchClaw 第二组 24 个候选 idea。
- 2026-07-11：迁入主任务用户侧完整导出与跨代理近期记录，固定 SHA256；新增 11-worktree 审计库存，防止单一 checkout 遗忘历史实现。
- 2026-07-11：第二轮 lint：36 个实体、10 个 gaps、55 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3351 字符。
- 2026-07-11：PhysTime-AdaTAD 1.0 在 `549bb81` 完成 raw-video K384 三头 matched pipeline、原帧 same-index 审计、one-step 梯度证明、真实 CUDA gate 工具及 gate-dependent 启动器；远端 focused suite `45 passed`。状态为 `tested`，真实 THUMOS gate、正式训练与 mAP 仍 pending。
