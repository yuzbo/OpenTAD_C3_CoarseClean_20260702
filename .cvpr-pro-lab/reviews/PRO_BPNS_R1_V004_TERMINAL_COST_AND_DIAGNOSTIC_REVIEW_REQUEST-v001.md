# ZoomToken BPNS-R1 v004 完整成本与诊断终态：post-result Pro 独立裁决请求

Request ID：`PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_REQUEST-v001`

Nonce：`ZOOMTOKEN-BPNS-R1-V004-TERMINAL-PRO-v001-20260829T130000+0800`

Exact Project：`g-p-6a79701398bc8191a9ef61db6302b24b`

请继续以本项目的科研首脑、总体规划者、方法设计者、流程维护者和最终审查者身份进行一次独立 post-result 裁决。Codex 仅作为执行、实现与证据整理者。本请求给出完整权威上下文，但不预设继续、停止、改写论文、转向 TAR32-FKV 或任何未列路线；你可以拒绝当前 framing、提出材料之外的解释，并独立下达唯一下一任务。

## 1. 论文问题与此前冻结合同

ZoomToken 面向离线 TAD，考察能否在保护动作边界和高 tIoU 定位时降低真实端到端成本。BPNS-R1 在 VideoMAE 前保留当前观测上的连续无孔洞 `8×8/K64` 原生支持；全部 K64 token 仍执行 12 层 VideoMAE-S 与既有 Adapter，不使用 cache、carry 或深度跳过。

此前固定 seed-42 final-EMA 的 K100/R1 为 `68.51/61.19/46.27` 与 `69.07/61.14/46.57`（Avg-mAP/mAP@0.6/mAP@0.7）。v001/v002 为数值准入失败，v003 为离线诊断构造导致的 measurement-completeness 协议失败，均没有效率结果。你在上一轮只授权一次 v004：逐 pass 原子保存 raw cost/power/prediction/receipt，八 pass 后才做非计时诊断；主判据为每臂四 pass 中位数的 p50 和完整 pass gross energy 比值都不高于 `0.95`，任一失败即停止 BPNS-R1 作为效率 headline。

## 2. v004 实施、准入和终态

v004 candidate `a4694019fd4cbbdc74885e160163e23d947dc05f` 是 base `8a59d655005b9030d8ea5dc17ee2620844cb587b` 的最小 clean descendant；只修改冻结的 profiler、focused test 和 launcher。N16R4 `27 passed`，fresh Critic `PASS`，fresh result-blind Evaluator `PRE_RUN_READY`；有效 precheck job `1260092` 为 `COMPLETED 0:0 / PRECHECK_READY`。

唯一正式 job `1260095` 在 `g0059` 从 `2026-08-29 00:37:38` 运行至 `06:10:03 +08:00`，`COMPLETED 0:0`。八 pass 顺序、每 pass 792 项 population、6,336 cost rows、929,889 power rows、每臂四次一致 prediction SHA、六项未舍入 evaluator vector、short-action/boundary、profile 和 terminal receipt 全部存在；没有训练、resume、official test 或第二次提交。

## 3. 冻结主估计结果

从 raw rows 独立重算的四-pass 中位数如下：

- K100：p50 `2481.957453 ms`，p95 `4375.123301 ms`，gross energy `144194.443195 J`，throughput `0.384916 item/s`，peak allocated/reserved `2053.593/2494 MiB`。
- R1：p50 `2444.551777 ms`，p95 `4278.951752 ms`，gross energy `134821.840551 J`，throughput `0.391503 item/s`，peak allocated/reserved `1542.859/1720 MiB`。
- R1/K100：p50 `0.9849289616`，p95 `0.9780185512`，energy `0.9350002508`，throughput `1.0171116288`，peak allocated `0.7512973880`，peak reserved `0.6896551724`。

能耗门通过，p50 门失败；因此 Codex 按冻结合同记录 `STOP_BPNS_R1_EFFICIENCY_HEADLINE`。这不是请你确认 Codex 的选线，而是请你独立裁决其科学含义、论文位置及唯一下一任务。

## 4. 准确率、边界、异常和未知

R1−K100 的 Avg-mAP 与 mAP@0.3–0.7 为 `+0.5353/+0.7520/+0.1518/+1.6238/-0.1042/+0.2528 pp`。short-action Avg-mAP 几乎不变；边界 start error 略改善、end error 略变差，matched 数略减少，不能支持统一的边界保护主张。

必须披露：第 3 号 K100 pass 的 in-pass power trace 最大采样间隙约 `2804.82 ms`，其余 pass 约 `52–71 ms`。coverage ratio 为 1.0，协议未冻结 gap 上限，能耗以线性插值积分；本轮形式上完整，但能耗存在这项不确定性。该异常不能改变 p50 门失败，也不自动授权重跑。

当前只有单 seed、单 checkpoint、单硬件证据；GPU temperature 未测量。能耗和显存的固定条件观察不能改写成跨硬件、跨 seed、跨数据集或总体效率结论。

另有一条严格隔离的 Pro 冻结组合探针 `R1-TAR32-FKV` 正在等待终态验收。它不是 v004 的补救、不会改变 v004 判据，且本请求不提供其终态性能来诱导你选线。Codex 将在本轮 v004 Pro 裁决完整回取后，才验证其训练协议；若协议有效且有有效模型输出，则依原冻结规则提交唯一 matched full-stack cost，不能据此预设论文方向。

## 5. 请独立回答并只下达一个任务

请用中文先给独立科学裁决，再给唯一任务，并至少回答：

1. 分别判定 v004 的工程证据、协议证据和科学证据；`STOP_BPNS_R1_EFFICIENCY_HEADLINE` 是否是冻结合同的正确直接结论？
2. p50、p95、能耗、吞吐和显存观察分别能支持什么最窄表述，绝不能支持什么？如何处理 2.805 秒功耗 gap 的不确定性？
3. BPNS-R1 是否应完全停止、降级为论文中的负结果/归因证据，或由你提出材料中未列出的其他处置？请不要默认 TAR32-FKV 是答案。
4. 准确率与 short-action/boundary 的混合结果是否支持任何边界保护陈述？当前论文主张和可发表性应如何改写？
5. 角色合同请明确为 `KEEP` 或 `REVISE`；若修订，只给最小科研流程修订，不扩张为复杂工程合同。
6. 只下达一个下一项论文—实验—开发任务，给出任务 ID、科学问题、允许/禁止改动、Builder 交付物、最小 focused tests、一次 Critic、一次结果盲 Evaluator、实验/停止规则、必须保存的终态证据，以及精确北京时间期限。
7. 冻结：任何新结果完成后必须进入一次全新 post-result Pro 复盘；在该裁决前不得追加实验。

附件中的 durable receipt、原始 `profile.json` 与 `terminal_receipt.json` 是本轮主要证据。请不要把结构代理当作实测成本，不要用 protocol success 替代科学 success，也不要用本次固定条件结果外推一般性。
