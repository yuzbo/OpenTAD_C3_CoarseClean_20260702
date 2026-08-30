# ZoomToken RACER24 Iteration-0 fresh Pro 终态复盘回执

- 时间（北京时间）：`2026-08-31T03:41:51+08:00`
- request：`PRO_RACER24_ITERATION0_TERMINAL_ADJUDICATION-v001`
- nonce：`ZOOMTOKEN-RACER24-ITERATION0-TERMINAL-PRO-v001-20260831T031000+0800`
- exact Project：`g-p-6a79701398bc8191a9ef61db6302b24b`
- conversation：`6a94842b-1370-83ea-a13c-2cc492170597`
- URL：`https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a94842b-1370-83ea-a13c-2cc492170597`
- Oracle session：`zoomtoken-racer24-iteration0-terminal-pro`
- 浏览器可验证路由：requested `gpt-5-pro`，model picker `Pro`，verified `true`
- transport：attachment-only，`browserInlineFiles=false`
- submission：`promptSubmitted=true`，实际科研提交 `1`，follow-up `0`
- response：`.cvpr-pro-lab/reviews/PRO_RACER24_ITERATION0_TERMINAL_ADJUDICATION_RESPONSE-v001.md`
- transcript：`.cvpr-pro-lab/reviews/runs/zoomtoken-racer24-iteration0-terminal-pro-v001/oracle-home/sessions/zoomtoken-racer24-iteration0-terminal-pro/artifacts/transcript.md`
- meta：`.cvpr-pro-lab/reviews/runs/zoomtoken-racer24-iteration0-terminal-pro-v001/oracle-home/sessions/zoomtoken-racer24-iteration0-terminal-pro/meta.json`

## 代码身份

- repository：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- branch：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-racer24-v001`
- exact commit：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`

上述 branch 与 exact commit URL 均进入唯一 Pro prompt，并由 Pro 在回答开头回显。

## 传输审计

Oracle 浏览器日志逐一确认实际上传了 6 个文件：主请求、`PAPER_PROGRESS.md`、用户转交材料、RACER24 MCL、终态回执、角色合同。计划中的 raw profile 位于 Codex `tmp` 路径，被 Oracle 默认忽略规则跳过；其决定性数值已经完整写入终态回执。Pro 回答声称“实际读取七个 attachment-only 文件”，并额外列出旧 BPNS 回执；浏览器上传日志不支持该第七项，因此该说法保留为 Pro 的来源陈述，不能覆盖实际附件计数 `6`。此差异不触发第二次提交或 follow-up。

## Pro 裁决

- 总裁决：`PIVOT`
- exact candidate：`STOP_RACER24_ITERATION0`
- 证据等级：`DECISION_GRADE_VALID_NEGATIVE_NOT_CLAIM_GRADE`
- Git 时序偏差：降低 provenance 等级，但不使内部停止结论失效；RACER24 不重跑。
- 角色合同：`KEEP`，不修改 `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`。
- 唯一下一任务：`ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`
- next owner：Codex Builder

RACER24 的合法结论只覆盖当前 exact eager block-path：在冻结 N16R4 真实形状微基准上，候选比 dense 慢约 4.006 倍，并使 peak allocated/reserved 分别约为 `1.98884x/1.84615x`。它不支持准确率、全栈 TAD、能耗、跨硬件或整个 selected-Q/full-KV/completion/CPTC 家族结论。

## 唯一任务与门

`GridFuse32-L6` 保持 R1 的 8 个 temporal tubelet 和每 tubelet K64；blocks 0–5 dense，blocks 6–11 对每 tubelet 固定相邻 pair 做 `64→32` 平均，在总计 N256 上执行完整 Q/K/V/MLP，把 block residual 广播回 N512 后再运行既有 dense Adapter。不得新增 router、top-k、teacher、cache、新参数或额外 loss。

- G0：六-block segment p50 speedup `>=1.35x`，peak allocated/reserved ratio 均 `<=1.05`。
- G1（仅 G0 通过）：Avg-mAP `>=68.57`，mAP@0.6 `>=60.64`，mAP@0.7 `>=46.07`，short-action delta `>=-0.75 pp`，start/end error ratio 均 `<=1.05`。
- G2（仅 G1 通过）：candidate/R1 p50 与 gross-energy ratio 均 `<=0.95`，peak allocated/reserved ratio 均 `<=1.05`。

任一前门失败即任务终态并在两小时内 fresh Pro；不调 pair 比例、方向、block schedule、编译模式或自定义 kernel，不启动第二候选、额外 seed 或 official test。

## 北京时间节点

- Builder MCL：`2026-08-31T07:30:00+08:00`
- clean/pushed candidate：`2026-08-31T20:00:00+08:00`
- fresh Critic：`2026-08-31T23:00:00+08:00`
- result-blind Evaluator：`2026-09-01T01:00:00+08:00`
- G0 formal action：`2026-09-01T02:00:00+08:00`
- conditional G1：`2026-09-01T06:00:00+08:00`
- conditional G2：G1 终态后两小时内，且不晚于 `2026-09-02T08:00:00+08:00`
- terminal scientific return：`2026-09-03T12:00:00+08:00`
- mandatory fresh post-result Pro：`2026-09-03T18:00:00+08:00`
