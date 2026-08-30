# ZoomToken-RACER24 Iteration-0 终态独立裁决与唯一下一任务请求

request_id=`PRO_RACER24_ITERATION0_TERMINAL_ADJUDICATION-v001`

nonce=`ZOOMTOKEN-RACER24-ITERATION0-TERMINAL-PRO-v001-20260831T031000+0800`

exact Project=`g-p-6a79701398bc8191a9ef61db6302b24b`

请把自己视为本项目整体科研流程的设计者、维护者与科学负责人。请基于附件和 Project Sources 独立判断，不接受 Codex 预设路线，也不要默认继续 RACER24、回到旧路线或从材料中已有候选挑选。你可以拒绝当前 framing、指出遗漏冲突，并提出附件未列出的科学方向；但最终只下达一个下一任务。

## 一、需要你裁决的终态事实

1. 此前用户手工转交了一份 Pro 风格 `PIVOT / KEEP` 响应，提出 `ZoomToken-RACER24`。该响应没有可审计的 exact Project conversation、nonce、模型/effort、附件、提交计数、Oracle transcript/meta 或 terminal receipt；还包含两项错误事实：job `1258299` 已终态而非仍运行，base 中也没有现成 selected-Q/full-KV helper。项目始终把它标为 `USER_MANUAL_TRANSFER_NOT_BROWSER_AUDITED`。用户随后明确确认其可作为 Iteration-0 实现与冻结微基准的执行权威，但这不补造浏览器 provenance。
2. 另一条 K100-TAR50 formal job `1261680` 在首个成功 optimizer update 前因 non-GeoRoute successful-update hook 失败，且冻结 prose 的 full-800 K/V、per-tubelet K50 与实际 global selected-400 K/V 实现冲突。它是工程/协议 blocker，不是科学负结果；没有自动修复或重跑。
3. RACER24 clean candidate 为 `5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`，base `2d945e64bdccd09ae2e2916524562e3f388c5a2a`，branch `codex/zoomtoken-racer24-v001`。GitHub 已核验：branch [`codex/zoomtoken-racer24-v001`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-racer24-v001)，exact commit [`5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/5ebaa74f611bb3a43c3042700a78b92a9e5e74fb)；GitHub ref API 在 `2026-08-31T03:03:44+08:00` 返回同一 SHA。机制保持 BPNS native K64 dense carrier，在 blocks `{4,6,8,10}` 每 tubelet 选择 24/64，以总 Q192 对 full K/V512，parameter-free completion 恢复未选位置，随后既有 Adapter 处理全部 512 token。没有新参数、loss、teacher、cache或跨 clip state。
4. Builder 完成四个 clean commits；独立 Critic 在 full-KV layout 与 durable receipt 两个确定性缺口修正后给出最终 `PASS`。N16R4 focused tests 为 `16 passed`；result-blind Evaluator 为 `PRE_RUN_READY`。
5. 首次 scheduler job `1262067` 因 Slurm `/bin/sh` wrapper 不支持 `source/module` 在 1 秒内退出，Python、模型、CUDA、profile 都未执行。独立 Evaluator分类为 `PRE_EXECUTION_OPERATIONAL_BLOCKER`，准入一次只把 wrapper 改为显式 `/bin/bash -lc` 的 replacement。
6. 唯一有效 scientific-attempt job `1262068` 在单张 N16R4 GPU 上完成冻结 real-shape matched block measurement：B=1、8 tubelets、64 token/tubelet、24 selected/tubelet、Q192/KV512、embed384、6 heads、warmup50、每臂200次。
7. 终态数值：dense/RACER24 p50=`1.3347729109/5.3468430415 ms`，p95=`1.3655385468/5.5123582017 ms`；p50 speedup=`0.2496375713x`。RACER24/dense peak allocated=`1.9888417957x`，peak reserved=`1.8461538462x`。冻结门 speedup `>=1.08x` 与两个 memory ratio `<=1.05` 全部失败。raw profile 和 terminal receipt 完整；exit `2:0` 来自门失败。
8. 独立 result-to-claim 结论为 `claim_supported=no`、confidence high、same-family/provisional：当前 exact RACER24 实现不具备最低 block-path 工程效率可行性，不支持训练、accuracy、full-stack TAD、energy 或发表级主张，也不能外推否定所有 selected-Q/full-KV 或 completion 方法。
9. 执行偏差必须显式裁决：微基准运行前，本机 GitHub HTTPS push 因 `SSL_ERROR_SYSCALL` 失败，随后使用增量 Git bundle 将当时尚未推送的 exact commit 部署到 remote clean checkout；这违反 RTK 的“bundle 只传递已推送对象”同步纪律。运行后从 N16R4 推送又因缺 GitHub credential 失败。本轮 Pro 提交前已通过有效 `gh` 凭据把同一 branch/commit 推到 GitHub，并由 ref API 和 commit API 双重核验 exact SHA；但事后补推不改变“实验运行时对象尚未在 GitHub”的时间顺序。local、remote 与当前 GitHub object/HEAD 均精确一致。请明确这一已纠正但真实存在的时序偏差是否降低该 block microbenchmark 的证据等级，以及后续应如何处置，不能默认为无影响。

## 二、请独立完成的科学判断

请严格区分：

- 工程实现与机械合同是否可信；
- 微基准协议和上述 Git 同步偏差是否使结果有效、仅诊断或无效；
- 当前 exact RACER24 的最低效率可行性是否应停止；
- 该负结果对论文主张、selected-Q/full-KV、completion 与更广 CPTC 范围能否外推；
- K100-TAR50 的工程/规格 blocker 与 RACER24 的有效负结果应怎样共同改变项目 framing；
- 角色合同 `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES` 应 `KEEP` 还是 `REVISE`。

不要为了“继续”而补救当前 RACER24，不要把 FLOPs、随机初始化、代码测试或结构直觉当作 full-stack/accuracy 证据。你也不必从 FARM24、PairLatent32 或历史路线中选择；它们在本次裁决前均未获授权。

## 三、必须返回的唯一执行订单

请用中文返回：

1. `A. 最终裁决`：对当前路线给出明确裁决及证据等级。
2. `B. 事实/来源/解释/提案边界`：纠正任何冲突或不当外推。
3. `C. 角色合同`：明确 `KEEP` 或 `REVISE`；如 REVISE，给出可直接同步的精确规则文本。
4. `D. 唯一下一任务`：只给 Codex 一个论文实验开发任务。你可以完全拒绝现有 framing 并提出未列方向，但必须说明科学问题、最小机制、允许文件面、matched control、证伪预测、主要指标、停止门与禁止事项。
5. `E. 精确北京时间节点`：至少给出 Builder MCL、candidate、fresh Critic、result-blind Evaluator/PRE_RUN、formal action、terminal scientific return、mandatory fresh post-result Pro review 的具体北京时间。
6. `F. 防止流程中断的执行规则`：只写能直接约束本任务连续执行的最小规则；不要建立与科学任务无关的调度框架。

在你返回之前，Codex 不运行训练、成本、successor、FARM24、PairLatent32 或任何新实验。你下达任务后，其 terminal result 仍必须回到一次全新的 Project Pro 复盘，不能由 Codex自行选线。
