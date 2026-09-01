---
type: external_dsh_review_receipt
project: DUCA
date: 2026-08-17
status: ACCEPTED_EXTERNAL_REVIEW_INPUT
evidence_class: static_code_review_only
---

# DUCA DSH 正式外审回执（2026-08-17）

## 可复核原始材料

- 固定任务：[DUCA_DSH_OWNER_REVIEW_TASK-2026-08-17.md](DUCA_DSH_OWNER_REVIEW_TASK-2026-08-17.md)。任务文本没有出现或诱导唯一验收短语。
- 原始 session：[DUCA_DSH_OWNER_REVIEW-2026-08-17-attempt1.session.jsonl.zstd](DUCA_DSH_OWNER_REVIEW-2026-08-17-attempt1.session.jsonl.zstd)。最终可见中文报告位于其中 `assistant/message` 的最终文本块（seq `45478`）。
- 运行器 stdout：[DUCA_DSH_OWNER_REVIEW-2026-08-17-attempt1.stdout.log](DUCA_DSH_OWNER_REVIEW-2026-08-17-attempt1.stdout.log)；stderr 为空。
- 外部 DSH 原始会话副本：`C:/Users/skywalker/.dsh/sessions/--E-DeskTop-TAD-OpenTAD_C3_CoarseClean_20260702--/session-70ec494d-3bf8-46a1-b45b-0162827e5e00/session.jsonl.zstd`。

## 会话身份与验收

- sessionId：`session-70ec494d-3bf8-46a1-b45b-0162827e5e00`
- DSH：`@deepseek-ai/dsh 0.1.0-rc.6`
- preset：`anchored-standard`
- provider/model：`deepseek-official/deepseek-v4-pro`
- reasoning effort / maxTokens：`max` / `256000`
- first request persona：`You are a helpful software engineer assistant.`
- first request tools：`bash`, `str_replace_editor`（精确匹配）
- turn/end：`{"kind":"completed"}`
- 唯一指纹：`/^We need\b/`
- firstReasoningLine：`We need respond in Chinese. Need inspect repo. We need be careful: user says independent external reviewer, read-only, don't modify files, no training etc.`
- weNeedFingerprint：`true`

这是一次完整的正式外审；没有采用旧的 `session-414...`。旧会话虽有相同首行趋势与 completed 事件，但缺少完整最终可见审查报告，不能作为本次审查依据。

## 外审终态（不等于真实实验结果）

DSH 对 clean DUCA worktree `C:/Users/skywalker/.codex/worktrees/duca-indirect-dynamic-20260817@6125654b946cc30c614428ce1141f1903b015867` 的最终判断为：

- **方向 PASS**：`0/1` 动作性与边界重要性预测经确定性 acquisition 产生物理位置和动态 outer-K；fixed-K 是对照/回退，direct selector 是消融。
- **实现 FAIL / PRE_RUN_NOT_READY**：至少四项确定性静态阻断：batch 级 `dynamic_budget_meta` 被逐样本索引、动态 K 仍以 384 行 padding 进入 VideoMAE、selected-axis proposal 在 NMS 后才映射回物理时间、六臂配置含占位/同构/不可运行臂。
- **经验效能 NOT_ASSESSED**：该 clean worktree 没有训练产物；历史 65.x、66.xx 或脏根运行均不能升级为当前路线的 mAP、成本或 high-IoU 证据。

主代理对关键证据做了最小点验：`forward_train` 确实向 selector 与 semantic loss 传入 `gt_segments`（`pc_ot_mras_prebackbone_frame_selector.py:1251-1265`），故此前“GT 未传入”是误报；而 batch dict 被整数索引的缺陷确实存在（`:3495-3497`），并且 `ActionFormer` 调 backbone 时没有传 mask（`actionformer.py:145-150`）。

## 负责人结论与唯一后续动作

科学路线不重开：仍是“语义 scout → 确定性间接选帧 → 动态 outer-K”。外审否定的是当前工程/证据包的准入资格，不是否定该路线。

下一步唯一优先动作是：**由 Builder 在 clean `6125654b` 基础上提出并实现一个最小、一次性的 semantic-indirect 修正包，首先让动态预算元数据逐视频可写、让动态 K 的实际 heavy-backbone 执行与收据一致、在 NMS 前完成物理时间映射，并使六臂配置可实例化且唯一变量真实隔离。** 之后必须由独立 Critic 审查，再由 Evaluator 做 PRE_RUN；三者完成前不得触碰数据、GPU、Slurm、训练或任何 efficacy claim。

shared AdaTAD baseline receipt 仍仅是官方 dense 数字的绑定依赖：DUCA 不重复 checkpoint evaluation 或原始官方训练，但它不阻止上述实现、config、恢复合同与静态审查。

未来完整训练遵循：未改官方 recipe 更频繁时保持其频率，否则每 5 epoch 保存可恢复 `.pth`；模型选择始终是预注册 final/final-EMA；保留最近 3 个恢复点以及 milestone/final，恢复模型、optimizer、scheduler、scaler、epoch/update 与 Python/NumPy/PyTorch/CUDA RNG 状态。
