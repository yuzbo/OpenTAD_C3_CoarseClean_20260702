# DUCA 完整数据身份准入与唯一下一任务裁决

**Nonce：`DUCA-FULL-DATA-IDENTITY-ADMISSION-v001-20260831`**

你是 DUCA 课题的独立科研负责人、总体设计者与最终科学审查者。Codex 只负责忠实执行你冻结的最小实现、独立代码审查、正式实验与证据回传。本轮不是请 Codex 或提示词替你选择路线，而是请你根据公开、精确版本的原始证据独立作出数据准入裁决，并在必要时签发唯一下一项任务。

## 公开代码与证据身份

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Wiki/证据分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831>
- 本轮 Wiki/证据精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/68690dbbbd8c44a8b2434e8d6f353c29d14f3824>
- 数据身份审计说明：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/experiments/duca-full-data-identity-audit-v1.md>
- 完整机器可读报告：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/sources/2026-08-31-duca-full-data-identity-audit-fdd2bcdd/split_identity_report.json>
- 字面 ID manifests 与集合差分：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/sources/2026-08-31-duca-full-data-identity-audit-fdd2bcdd>
- 你上一轮的完整综合路线裁决：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/sources/2026-08-31-pro-duca-comprehensive-route-integration-v001.md>
- 当前论文进展：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/PAPER_PROGRESS.md>
- 当前研究检索摘要：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/query_pack.md>
- 防重复实验记录：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/68690dbbbd8c44a8b2434e8d6f353c29d14f3824/research-wiki/anti_repetition.md>

审计实现是 H65 干净提交的直接子提交：

- H65 基座：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- 审计分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-full-data-identity-audit-v1-20260831>
- 审计提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837>
- 审计工具：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/fdd2bcdddf3f23f3546244adf90c4427ed022837/tools/bata/audit_duca_thumos14_split_identity.py>
- 聚焦测试：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/fdd2bcdddf3f23f3546244adf90c4427ed022837/tests/test_audit_duca_thumos14_split_identity.py>

请实际打开并检查上述精确提交和关键文件，不要只根据本提示词复述结论。

## Codex 中立回传的事实

1. 审计提交 `fdd2bcdd...` 的父提交精确为 H65 `04c35a3b...`，只增加审计工具和一个聚焦测试；工作树干净，已推送 GitHub。
2. 本地相关测试共 `29 passed`，N16R4 CPU 聚焦测试 `6 passed`；独立只读 Critic 返回 `PASS`。
3. 完整训练侧 annotation、正式 loader replay 与 canonical physical media 都是相同的 200 个视频，manifest SHA-256 均为 `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0`。
4. OpenTAD 留出侧 annotation、loader、physical media、evaluator 与历史正式 prediction-key 集合都是相同的 211 个视频，manifest SHA-256 均为 `5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e`。
5. ActionFormer 原始 annotation 的字面 `Test` 集合为 212。相对 OpenTAD 211 的唯一额外 ID 是 `video_test_0000270`；OpenTAD 源码说明它因错误标注被排除。`video_test_0001292` 不在 ActionFormer 的 212 条 annotation 中，只是额外物理/特征文件，不属于评估集合。
6. 411 个预期 canonical 视频全部通过基本 `ffprobe` 解码；无缺失、坏链接、重复、未分配或训练—留出交集。
7. 审计未读取留出动作类别或时间边界，未加载 checkpoint/model，未使用 GPU，未生成预测或计算 mAP；历史预测文件仅读取顶层视频 ID 键。
8. 首次 CPU 命令把 ActionFormer 区分大小写的 `Test` 误写成小写 `test`，所以得到空集合和调用层阻断。原输出被保留；随后只修正该参数，代码、数据与来源不变。有效报告位于 `result_v2`，SHA-256 为 `d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`，结论为 `DATA_IDENTITY_PASS_211`。
9. 当前没有多预算模型代码、PRE_RUN、GPU 作业、训练、留出预测、mAP 或成本结果。数据身份事实不能被写成模型有效性证据。

## 请你独立裁决

请先判断上述证据是否足以把 DUCA 的正式数据边界冻结为：完整 200-video `training` 用于训练，完整 211-video OpenTAD `validation` 作为一次性 held-out evaluation；ActionFormer 的 212-video `Test` 仅作为来源差异，不与 OpenTAD 211 静默合并。若证据不足，请指出唯一、具体、会改变正式比较解释的缺口，并给出最小补证任务。若证据充分，请明确签发数据准入，同时说明 held-out 集合在后续训练、模型选择、阈值选择、规则选择和最终一次性读取中的边界。

随后请独立复核你上一轮条件冻结的 H65 系统多预算暴露适应实验是否应当正式解锁、修订或停止。不要把该实验称为纯 detector-only 适应：H65 Stage-2 的可训练集合还包含 Scout/selector 相关路径和 detector feedback。不要自动恢复旧三档 frozen-detector oracle、Coverage-v1、Gumbel-Softmax、Mamba、Block Drop、DFT、TensorRT 或其他未被本轮证据授权的路线。

两份最新 Pro 裁决对后续种子顺序有一处明确冲突：较早报告写为完整执行 `3407/3408/3409`，较新综合裁决写为 seed `3407` 先回答机制问题，只有全部门通过后才按冻结方法复制 `3408/3409`。请你明确选择并解释一种顺序；Codex 不替你选择。

若你决定继续，请只签发一个当前任务，给出：科学问题、唯一实验变量、完整 200/211 数据边界、代码基座与允许移植的最小表面、两个 matched arms、可训练参数集合、训练更新/EMA/seed 顺序、预测封存和配对不确定性协议、真实成本口径、停止门、Builder → 独立 Critic → Evaluator 的职责以及北京时间绝对截止时间。工程要求应保持最小，服务于论文可证伪实验，不建设新的合同或工作流平台。

输出必须以一个清楚的 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP` 开头；明确区分数据事实、模型假说和尚未获得的性能证据。结尾逐字写出：

`DUCA_FULL_DATA_IDENTITY_ADMISSION_READY`
