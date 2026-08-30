# GridFuse32-L6 G0 terminal engineering-blocker adjudication

Request ID: `PRO_GRIDFUSE32_L6_G0_TERMINAL_ENGINEERING_BLOCKER_ADJUDICATION-v001`

Nonce: `ZOOMTOKEN-GRIDFUSE32-L6-G0-TERMINAL-PRO-v001-20260831T043409+0800`

Exact Project: `g-p-6a79701398bc8191a9ef61db6302b24b`

请作为 ZoomToken 整体科研流程的设计者、维护者与科研首脑，基于本请求及附件独立裁决。请拒绝任何你认为不成立的 framing，可提出材料未列出的替代；不要迎合 Codex 的既有实现，也不要把以下内容理解为候选路线菜单。

## 必须核验的最新 GitHub 实现身份

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001>
- Exact commit: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0b734ab839973b2c945b012f066db8222d235bb9>

请把三个链接视为本轮讨论的代码权威入口；任何结论必须绑定 exact commit，而不是旧 commit、旧聊天或本地未推送状态。

## 已冻结任务与实现

上一轮 exact-Project Pro conversation `6a94842b-1370-83ea-a13c-2cc492170597` 裁决 `PIVOT / STOP_RACER24_ITERATION0 / KEEP`，并独立下达唯一任务 `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`。GridFuse 保持 R1 K64、八个 temporal tubelet、dense native N512 carrier 与每层既有 Adapter；blocks 0--5 dense，blocks 6--11 按奇偶层固定水平/垂直相邻 pair 做每 tubelet `64→32` mean fusion，在 N256 上完整执行 Q/K/V/MLP，把 merged residual delta 广播回两个各自保留当前 native residual 的成员，再恢复 N512 并运行 dense Adapter。没有 router、top-k、跨 clip state、teacher、辅助 loss 或新参数。

最终实现经过 N16R4 独立进程测试 `9/12/8 passed`、fresh Critic `PASS`、fresh result-blind Evaluator `PRE_RUN_READY`。G1→G2 checkpoint 还绑定 canonical path、SHA256、epoch59 和 `state_dict_ema`。两个非科学 precheck `1262078/1262079` 在任何测试前因计算节点 GitHub DNS 失败；launcher-only 修正把 fresh-fetch 放到登录节点，并在计算节点同时核验 clean exact HEAD 与 persistent remote-tracking ref。最终 precheck `1262089` 为 `COMPLETED 0:0 / PRECHECK_READY`。

## 唯一正式 G0 终态

唯一正式 G0 job `1262090`（1 GPU、4 CPU）在 `g0030` 运行 15 秒后 `FAILED 2:0`。segment profiler 在构建 detector 的 pre-processing pipeline 时因 `Rearrange` 未注册到 mmengine transform registry 而抛出 `KeyError`。任何 warmup、500 次 alternating synchronized timing、allocated/reserved memory 测量或 gate evaluation 都没有开始。远端存在受控失败 `terminal_receipt.json`，不存在 `profile.json`。正式 G0 提交数 `1/1` 已用尽；G1/G2 从未开放。

因此当前可核验事实只能是：工程/协议阻塞发生；GridFuse 的 p50/p95/显存性能未知。不得把 15 秒 runtime、参数打印或缺失 profile 解释为正负科学证据。

## 请独立完成的裁决

1. 明确区分工程证据、协议证据和科学证据，并给出总裁决（例如 CONTINUE、REVISE、PIVOT 或 STOP；由你独立决定）。
2. 裁决该 `Rearrange` registry 失败是否允许在不改变模型、门槛、checkpoint、测量协议和提交语义的前提下形成一个新的唯一任务；若不允许，请独立指定真正的下一科学任务。不要默认 wrapper-only replacement，也不要默认 PIVOT。
3. 复核为何已通过的 precheck 没有覆盖真实 model construction，并提出你认为必要的最小流程/规则修订，避免再次把形式上的 PRE_RUN_READY 当作可执行 readiness；同时避免无限复核与无效流程膨胀。
4. 对 `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md` 给出 `KEEP` 或 `REVISE`；若 `REVISE`，请提供可直接落盘的精确条款。
5. 最终只向 Codex 下达一个原子、可证伪的下一任务，给出允许改动文件、禁止事项、Builder/Critic/Evaluator/正式动作/终态返回的精确北京时间期限，以及明确的接受、停止和 blocker-return 判据。
6. 冻结：无论下一正式结果成功、负面或协议失败，都必须在追加其他实验前进行一次全新的 exact-Project post-result Pro 复盘。

请用中文给出完整、可执行、无歧义的裁决。不要预设 Codex 已获得重跑、修复、训练 G1 或进入 G2 的授权。

## 附件

1. 本请求；
2. `PAPER_PROGRESS.md`；
3. `research-wiki/query_pack.md`；
4. `ZOOMTOKEN_GRIDFUSE32_L6_GATED_MINIMAL_CHANGE_PLAN-2026-08-31.md`；
5. `ZOOMTOKEN_GRIDFUSE32_L6_GATED_PRE_RUN_READY-2026-08-31.md`；
6. `ZOOMTOKEN_GRIDFUSE32_L6_G0_START_RECEIPT-2026-08-31.md`；
7. `ZOOMTOKEN_GRIDFUSE32_L6_G0_TERMINAL_RECEIPT-2026-08-31.md`；
8. `ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md`。
