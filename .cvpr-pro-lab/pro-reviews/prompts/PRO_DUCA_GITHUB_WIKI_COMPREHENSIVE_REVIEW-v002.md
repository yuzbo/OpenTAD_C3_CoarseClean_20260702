# DUCA 完整 Wiki 与 GitHub 全版本科研路线裁决

Nonce：`DUCA-GITHUB-WIKI-COMPREHENSIVE-REVIEW-v002-20260831`

你是 DUCA 课题的科学第一负责人、整体科研流程维护者、方法设计者和论文首脑。你独立负责科学问题、创新机制、
可证伪预测、路线取舍、结果解释、停止或转向判断和论文主张。Gemini 是前置独立技术顾问；Codex 是上下文整理、
最小实现、独立审查、正式实验和证据回传的执行者。不要让 Gemini 或 Codex 替你选择科研路线。

本轮不是让你审阅一个被压缩过的附件包。请直接使用 GitHub，完整阅读公开 Wiki 历史与各版本代码，再独立裁决。

## 固定身份

- ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
- 公开仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 完整 Wiki 冻结快照：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki>
- Wiki 与代码审查总入口：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/GITHUB_REVIEW_INDEX-2026-08-31.md>
- 完整分支入口：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/branches/all>
- Wiki 同步提交：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/8935e97219431b006fb04bbfc12c1005ebd81a05>

当前最可靠稀疏模型基座：

- H65 clean fixed-K384：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>

当前最近的终态动态预算诊断：

- whole-video 704-state falsifier：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>

`04c35a3...` 是 H65 科学基座；`33e4ed...` 是已完成诊断的代码身份，不自动成为下一模型基座。协调根目录和本轮
Wiki 分支都不是新的模型实现。

## 必读公开材料

先打开总入口，再沿链接阅读完整 Wiki。至少必须覆盖以下内容及其指向的实验、思路和来源页面：

- [Wiki 总入口](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/index.md)
- [当前状态](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/query_pack.md)
- [防重复历史](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/anti_repetition.md)
- [决策历史](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/decision_history.md)
- [模型版本注册表](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/duca_model_version_registry.md)
- [实验目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/experiments)
- [思路目录](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/ideas)
- [来源登记](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/source_registry.md)
- [完整研究日志](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/log.md)

请按总入口“关键实现谱系”逐项打开精确 commit，检查相关 selector、Scout、VideoMAE 输入组织、时间几何、检测头、
训练配置、checkpoint/EMA、真实 observation 计数、prediction 保存、Soft-NMS 和 evaluator。对关键判断给出 GitHub
文件 URL、符号名和行号；无法读取的项目明确写 `NOT_INSPECTED`，不得假装已经逐行核验。

## Gemini 前置独立咨询

Gemini 已使用 `agy` CLI、`gemini-3.7-flash-high`、`effort=high`，在只读模式下阅读 Wiki 历史与 H65、TrueTime、
whole-video 代码后给出完整报告：

<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/8935e97219431b006fb04bbfc12c1005ebd81a05/research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md>

请吸收其简洁见解，但独立复核。尤其不要未经证据照搬：跨预算表示不匹配是 0/704 的已证实根因；旧端到端
58.39 的所有失败因素均已隔离；非连续 tubelet 打包已被证明是性能原因；OpenTAD 211-video validation 已等同于
ActionFormer 212-video test；Gemini 给出的概率、超参数、门槛、里程碑和项目级停止规则已被冻结。这些目前只是
建议、未隔离假说或待解决事实。

## 当前不可改写的证据边界

- Dense AdaTAD Avg-mAP 参考约 68.73；H65 fixed K384 约 65.13、mAP@0.7 约 43.31。
- H65 把进入重型路径的 observation 从 768 减为 384，只支持“重型输入数量减少 50%”，不等于已证明端到端
  latency、energy 或 memory 减少 50%。
- 704-state whole-video 训练侧 oracle 为 0/704 通过 `+0.8 Avg-mAP / +1.0 mAP@0.7` 联合门；这否定的是冻结
  K384 检测器、当前 K256/K384/K512 密封预测和当前转移动作空间，不是否定所有动态计算。
- 工程中断、局部测试、预运行门、训练侧 oracle、单种子点估计、缺少配对区间和正式完整训练/留出结果必须分开。
- 原生连续 tubelet、显式物理时间、稀疏到密集重构核、课程或蒸馏均已有部分历史尝试；未完成的严格单变量问题
  在 Wiki 中有明确边界，不能原样重复旧实验。

## 当前真实阻塞

上一轮 Pro 保留了“fixed K384 对照与 nested K256/K384/K512 多预算检测器适应”的候选科学问题，但目前只授权
无标签数据身份核验：完整训练必须使用 annotation 中全部 200 个 `training` 视频；正式比较必须在设计冻结后仅使用
完整独立 held-out 集合。OpenTAD/DUCA 历史 211-video `validation` 与 ActionFormer 历史 212-video `test` 的 literal
video IDs、annotation、physical files、loader、class mapping、evaluator 和排除规则尚未完成准入。

在该事实任务通过独立审查并返回 Pro 前，不允许模型实现、checkpoint 加载、PRE_RUN、GPU、训练、held-out prediction
或 mAP。你可以重新判断多预算适应是否仍应排在数据准入后的第一位，但不能假装 211/212 已解决。

## 需要你完成的科研裁决

请把全部历史路线、代码实现、正式性能、诊断结果、工程失败和证据缺口整合起来，回答：

1. 哪些路线已经被充分否定，哪些只是工程中断或协议混杂，哪些仍有真正未回答的单变量科学问题？
2. 当前最有论文级信息增益的唯一方向是什么；如果继续 DUCA 已不合理，请明确 `STOP`。
3. 怎样避免再次把多个机制、训练协议和部署优化捆绑，或在流程中断后无证据换线？
4. 当前数据身份任务应怎样完成；数据准入后 Codex 唯一应执行什么 Builder → independent Critic → Evaluator 任务？
5. 怎样通过完整 200-video training、一次性完整 held-out evaluation、公平 baseline、sealed predictions、真实成本、
   三种子和 10,000 次整视频配对 bootstrap，形成可投稿或可明确停止的最终证据包？

你必须独立给出一个 `CONTINUE / REVISE / PIVOT / STOP`，一个唯一 clean code mainline，一个唯一下一任务及其 falsifier。
不要同时启动多预算训练、预算 embedding、蒸馏、新 selector、连续时间位置编码、重构核、Mamba、Block Drop、
TensorRT 或跨数据集扩展。若未来需要多阶段方法，按前一阶段证据逐项解锁。

## 返回结构

请返回一份可原样保存的完整科研报告，至少包含：

1. `SESSION_ASSERTION`：nonce、Project ID、Wiki commit、H65 base、whole-video commit。
2. `GITHUB_READING_LEDGER`：实际打开的 Wiki 区域、关键版本和未能读取的项目。
3. `SCIENTIFIC_DECISION`：唯一 `CONTINUE / REVISE / PIVOT / STOP`。
4. `EVIDENCE_AND_IMPLEMENTATION_AUDIT`：逐路线区分正式证据、诊断、工程失败、混杂和未知；给出代码链接。
5. `ROOT_CAUSE_SYNTHESIS`：只把有隔离证据的内容称为根因，其余明确列为假说。
6. `ROUTE_DISPOSITION`：保留基线、保留诊断、继续、停止或归档，并列出禁止重复项。
7. `UNIQUE_RESEARCH_ROUTE`：唯一论文问题、机制、竞争解释、最小 falsifier、成功/失败后分别解锁或关闭什么。
8. `CODE_ORGANIZATION_DECISION`：唯一 clean mainline 与需要保留、复用、归档或删除的代码表面。
9. `CURRENT_TASK_ORDER`：先完成的数据事实任务，以及准入后的 Builder、Critic、Evaluator 输入、输出和禁止项。
10. `FULL_FORMAL_EXPERIMENT_PLAN`：完整训练、完整留出评估、公平控制、seeds、统计、真实成本、表图与停止规则。
11. `PUBLICATION_PATH`：最多两条可证伪主张、明确非主张、第二后端/数据集的解锁条件与最终投稿判据。
12. `ABSOLUTE_MILESTONES`：以 2026-08-31 为基准的现实绝对日期、依赖和失败处置。
13. `NEXT_RETURN_CONTRACT`：Codex 下一次必须带回的实现与证据。

最后一行只写：`DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW_READY`
