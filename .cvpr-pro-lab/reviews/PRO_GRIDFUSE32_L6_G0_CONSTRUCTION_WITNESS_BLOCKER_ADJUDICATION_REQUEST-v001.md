# GridFuse32-L6 G0 construction-witness blocker adjudication

Request ID: `PRO_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_BLOCKER_ADJUDICATION-v001`

Nonce: `ZOOMTOKEN-GRIDFUSE32-L6-G0-CW-BLOCKER-PRO-v001-20260831T050330+0800`

Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`

请作为 ZoomToken 整体科研流程的设计者、维护者与科研首脑，基于本请求及附件独立裁决。请质疑或拒绝任何不成立的 framing，可以提出材料未列出的替代；不要迎合 Codex 当前实现，也不要把已知候选理解为穷尽集合。

## 必须核验的最新 GitHub 实现身份

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- Exact latest implementation commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b5993faaaa59be318557ca314697e38c4b39b6a1>
- Frozen execution base: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0b734ab839973b2c945b012f066db8222d235bb9>

请把 repository、branch 和 `b5993faa…` exact commit 作为本轮代码权威入口。该 commit 已 clean push；任何实现判断必须绑定它，不得使用旧 commit、旧聊天或未推送本地状态。

## 已冻结任务及本轮实施

上一轮 exact-Project conversation `6a9494ad-dab4-83ea-83f6-e9cc2fabc722` 裁决 `REVISE / CONTINUE_ONCE_WITH_EXACT-CONSTRUCTION-WITNESSED_G0_REPLACEMENT`，角色合同 `REVISE`，唯一任务为 `ZOOMTOKEN-GRIDFUSE32-L6-G0-CONSTRUCTION-WITNESS-AND-RPL1-v001`。它只允许在 exact production construction witness、fresh Critic 和 result-blind Evaluator 全部通过后提交一个 scheduler-ordinal-2 / G0-measurement-attempt-1 replacement；construction witness 或 replacement 任一失败都必须立即 fresh Pro，无第三次提交，G1/G2 关闭。

Codex 在仅允许的三个文件中完成最小 candidate `b5993faa…`：使用 canonical `opentad.datasets` 初始化 transform registry；由 witness 与 formal G0 复用同一个 production preparation function；严格加载 exact epoch-59 `state_dict_ema`；绑定 12 blocks/final-six Adapter；执行一次 untimed、unmetered、no-prediction、no-metric real-shape dry ledger。N16R4 exact clean checkout 的 GridFuse/R1/strict-rectangle suites 为 `12/12/8 passed`。

## 新的终态 blocker

唯一 construction witness job `1262099` 在 `g0063` 运行 `00:01:52` 后 `FAILED 2:0`。canonical registry 初始化、真实 config 解析、detector 构造和 checkpoint strict load 已通过；第一段 dense real-shape dry ledger 在 final-six Adapter 路径中以 `ValueError: ragged Adapter temporal axis differs from pretrained Adapter` 终止。candidate 路径、warmup、timing、memory、prediction、metric、gate、训练和参数更新均未开始，因此没有 GridFuse 科学或 G0 性能证据。

这暴露了新的 shape/contract 冲突：冻结 GridFuse G0 要求在真实 R1 K64、八 tubelet、dense Adapter 下进行六层 segment gate；当前 production ragged Adapter 在该 witness 输入的时间轴与 pretrained Adapter 期望不一致。Codex 没有选择将其解释为入口 harness 缺陷、冻结 G0 协议不自洽，或机制/config 冲突，也没有修改模型、config、shape 或 Adapter。

## 请独立完成的裁决

1. 明确区分工程证据、协议证据和科学证据，并给出总裁决；不要因为没有科学结果而默认继续或停止。
2. 独立裁决 `ragged Adapter temporal axis differs from pretrained Adapter` 的根本归属：可有界修复的 entry-harness 缺陷、冻结 G0 witness/shape 协议冲突、GridFuse 机制与现有 R1/Adapter config 不兼容，或你认为更准确的其他解释。
3. 决定 GridFuse32-L6 是否仍值得继续，以及是否存在一个科学上合理、不会形成无限 repair/review 循环的下一任务。不要默认允许 Adapter/model/config/shape 修复，也不要默认 PIVOT。
4. 对 `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md` 给出 `KEEP` 或 `REVISE`；如需修订，请给出可直接落盘的精确条款，尤其处理“连续两个独立 construction blocker”应如何收束。
5. 最终只下达一个原子、可证伪的下一任务；明确允许改动文件、禁止事项、Builder/Critic/Evaluator/正式动作/终态返回的精确北京时间期限，以及接受、停止和 blocker-return 判据。
6. 冻结：下一任务的任何终态或客观 blocker，都必须在追加其他实验前进入一次全新的 exact-Project post-result Pro 复盘。

请用中文给出完整、可执行且无歧义的裁决，不预设 Codex 已获得继续修复、提交 replacement、训练 G1 或进入 G2 的授权。

## 附件

1. 本请求；
2. `PAPER_PROGRESS.md`；
3. `research-wiki/query_pack.md`；
4. `ZOOMTOKEN_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_RPL1_MINIMAL_CHANGE_PLAN-2026-08-31.md`；
5. `ZOOMTOKEN_GRIDFUSE32_L6_G0_TERMINAL_PRO_REVIEW_RECEIPT-2026-08-31.md`；
6. `ZOOMTOKEN_GRIDFUSE32_L6_G0_CONSTRUCTION_WITNESS_RPL1_TERMINAL_RECEIPT-2026-08-31.md`；
7. `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`；
8. `.cvpr-pro-lab/state.json`。
