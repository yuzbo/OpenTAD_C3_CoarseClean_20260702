# ZoomToken 科研流程连续性 Pro 复盘收据（2026-08-28）

## 复盘身份

- Request ID：`PRO_RESEARCH_PROCESS_CONTINUITY_AND_V003_NEXT_TASK_REQUEST-v001`
- Nonce：`ZOOMTOKEN-PROCESS-CONTINUITY-V003-v001-20260828T121545+0800`
- Exact Project：`g-p-6a79701398bc8191a9ef61db6302b24b`
- Conversation：`6a910cca-f580-83e9-b41a-975f37f4489d`
- URL：`https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a910cca-f580-83e9-b41a-975f37f4489d`
- 浏览器可见模型：`GPT-5.6 Pro`
- 提交：一次全新对话、一次 prompt、七个附件、零 follow-up、未修改 Project Sources
- 提交时间：`2026-08-28T12:21:13+08:00`
- 完成时间：`2026-08-28T12:35:02+08:00`
- Transcript SHA-256：`811f876d07d3febe5900051708d97a2048510f202242901cbb2670e3645dfb37`

## 提交给 Pro 的事实范围

请求如实覆盖了 `2026-08-27T12:00:00+08:00` 至 `2026-08-28T12:00:00+08:00` 的工作：v002 的最小实现、独立 Critic/Evaluator、job `1258299` 的终态协议失败、终态到有效 Pro 复盘之间约 6 小时 54 分钟的浏览器与协调延迟、Pro 的 v003 委派、candidate `8a59d655005b9030d8ea5dc17ee2620844cb587b`、21 项 focused tests、Critic PASS、Evaluator PRE_RUN_READY、precheck job `1258524` 完成，以及 formal job `1258526` 于 `2026-08-28T11:58:39+08:00` 启动。没有读取或解释 v003 的 live/partial 性能。

## Pro 的流程裁决

- 总体裁决：`CONTINUE`。
- 角色合同：`REVISE`。
- 科学工作有推进，但尚未形成新的决定性论文结果；v001/v002 的主要问题是用不足精度的历史数值构造了不可识别硬门。
- 最大可避免耗时来自浏览器和协调：零提交排队、重复确认，以及 v002 终态后约 6 小时 54 分钟才形成有效 Pro 请求。GPU/Slurm 不是当前主瓶颈；权限明确后，v003 从 Pro 裁决到正式启动约 3 小时 13 分钟。
- 有效一轮固定为：`Pro 唯一任务 → Codex 最小实现 → 一次 Critic → 一次结果盲 Evaluator → 一次正式实验 → 后端静默等待 → 终态证据摄取 → fresh post-result Pro 裁决`。作业提交、运行中或收据存在均不等于完成。
- 已授权冻结任务由 Codex 连续推进到终态证据与 fresh Pro 裁决；普通 Git、远端部署、Slurm、后台等待和证据摄取不再逐项请示。
- 长时等待后端化，前台只在状态变化、正式终态、硬 blocker 或固定报告节点通知。
- Pro 请求以 Project ID、request ID 和 nonce 唯一标识。未实际提交时恢复同一请求；已有 submission/conversation 时只等待或回取；禁止 duplicate 和 follow-up。
- 默认一次 Critic 与一次 Evaluator；只有决定性缺陷允许一次最小修复和针对改动表面的复审，禁止无限审计。
- 工程或协议失败没有科学方向。正式作业终态不得自行重跑、恢复或创建 successor，除非冻结任务明确授权。

上述最小规则已经同步到 `RTK.md` 与 `docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`。

## 当前唯一任务与期限

唯一任务：`ZOOMTOKEN-BPNS-R1-V003-TERMINAL-CLOSURE-AND-FRESH-PRO-ADJUDICATION-v001`。

只允许对 formal job `1258526` 后端静默等待。终态后摄取冻结的八 pass、prediction/evaluator、identity、完整实测成本、功耗覆盖、显存、short-action/boundary、profile 与 terminal receipt；随后发起恰好一次全新的 post-result Pro 复盘。不得新增训练、第二次 replay、辅助 arm、阈值修改、路线转换或 partial-result 选线。

北京时间时限：

- 作业按 8 小时时限预计不晚于 `2026-08-28T19:58:39+08:00` 进入终态；
- post-result Pro 必须在终态后 30 分钟内提交，且绝对不晚于 `2026-08-28T20:30:00+08:00`；
- 包含 Pro 裁决回取在内的本轮完整返回，绝对不晚于 `2026-08-28T22:00:00+08:00`。

本次流程复盘不替代 v003 的 fresh post-result Pro 复盘。

## 证据位置

- 请求：`.cvpr-pro-lab/reviews/PRO_RESEARCH_PROCESS_CONTINUITY_AND_V003_NEXT_TASK_REQUEST-v001.md`
- 完整回答：`.cvpr-pro-lab/reviews/PRO_RESEARCH_PROCESS_CONTINUITY_AND_V003_NEXT_TASK_RESPONSE-v001.md`
- Streaming receipt：`.cvpr-pro-lab/reviews/PRO_RESEARCH_PROCESS_CONTINUITY_AND_V003_NEXT_TASK_STREAMING_RECEIPT-v001.json`
- Terminal receipt：`.cvpr-pro-lab/reviews/PRO_RESEARCH_PROCESS_CONTINUITY_AND_V003_NEXT_TASK_TERMINAL_RECEIPT-v001.json`
- Raw Oracle transcript/meta：`.cvpr-pro-lab/reviews/runs/zoomtoken-process-continuity-v003-pro-20260828t121545/oracle-home/sessions/zoomtoken-process-continuity-v003-pro/`
