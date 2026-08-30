# ZoomToken RACER24 用户转交 Pro 裁决摄取

## 1. 摄取身份

- 摄取时间：`2026-08-31T01:11:01+08:00`
- 原始材料：`C:/Users/skywalker/.codex/attachments/557675fe-15b5-4574-9d1f-b374016b9278/pasted-text.txt`
- 材料内裁决：`PIVOT`
- 材料内角色合同：`KEEP`
- 材料内订单：`ZOOMTOKEN-COMPOSITE-SPRINT-AGENTS-ORDER-v001`
- 材料内候选：`ZoomToken-RACER24`
- 材料内建议 base：`2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- 材料内建议 branch：`codex/zoomtoken-racer24-v001`

这份材料由用户手工转交。仓库中没有与它对应的 exact Project ID、conversation ID/URL、nonce、浏览器可见模型与 effort、附件清单、提交计数、Oracle transcript/meta 或 terminal receipt。因此它不是浏览器审计完成的 Project Pro 证据；本文只保存其可执行提案、事实纠正与权限边界，不补造来源。

## 2. 已核验的项目事实

1. K100-TAR50 formal job `1261680` 已 `FAILED 1:0`。它在 epoch 0 首个成功 optimizer update 前因 `successful update indexing requires a GeoRoute backbone` 退出，没有 checkpoint、official prediction/vector、短动作、边界或成本结果。该终态是 `ENGINEERING_OR_PROTOCOL_BLOCKER`，不是科学负结果，正式提交 `1/1` 已用尽。
2. 冻结文案声称 full-800 K/V 与 per-tubelet K50，但 candidate 继承的 strict A-MoD odd block 实际在全局 flattened selected top-400 上执行 attention，K/V 也是 selected-400。这一规格—实现冲突已存在于终态证据中。
3. BPNS-R1 v002 job `1258299` 早已终态 `FAILED 1:0`，且结果根为空；它并非仍在运行。后续 v004 job `1260095` 已完成八-pass 成本闭环，并以 p50 仅改善 `1.51%` 停止 BPNS-R1 独立效率 headline。
4. 当前 `vit_adapter.py` 没有现成的 selected-query/full-KV helper，也没有 RACER residual-completion 实现。材料中“base 已有该 helper”的陈述不能作为代码事实。

## 3. 作为来源陈述保存的 RACER24 合同

以下内容只标记为用户转交 Pro 材料中的 proposed contract：

- 保持 BPNS 连续原生 `8x8/K64` 支持和 512-token dense carrier；RACER blocks 为 `{4,6,8,10}`。
- 每个 tubelet 精确选择 `24/64`，每 clip `192`；禁止全局 top-k。
- RACER block 使用 selected-Q/full-KV：`Q=192`、`K/V=512`，MLP 只对 selected token 执行。
- router 只使用紧邻 dense block 的 pre-Adapter residual relative magnitude 与相邻 tubelet residual surprise；stop-gradient，按原生索引稳定打破并列；无跨 clip/window/video state。
- 未选 token 不丢弃，而以当前 selected residual、当前 key 与前一 dense residual进行 parameter-free completion；每层后恢复 dense K64 carrier，既有 Adapter 仍处理全部 512 token。
- 不新增参数、loss、teacher、cache；只允许一个 config、focused tests、一个 launcher、必要的 `tools/train.py` allowlist，以及一个 real-shape GPU microbenchmark。
- Iteration 0 只做最小实现、focused parity 与 microbenchmark；Iteration 1 至多一次 seed-42、60-epoch正式训练。FARM24 或 PairLatent32 最多只能在冻结触发条件后选择一个，不得并行或扫 capacity。

完整公式、门限与停止条件仍以原始材料为来源；在浏览器 Project provenance 或用户明确授权范围扩展前，不把它们写成已验证模型事实或实验结果。

## 4. 当前权限边界

- 接受 `PIVOT`、`KEEP` 与 RACER24 作为下一候选的 **Iteration-0 Builder 规划输入**。
- 当前只允许生成可审计的 `MINIMAL_CHANGE_PLAN`；不得以本摄取替代 clean candidate、fresh Critic、result-blind Evaluator 或真实 microbenchmark。
- 不授权数据读取、GPU、Slurm、60-epoch训练、full-stack cost、FARM24、PairLatent32 或第二候选。
- 材料没有给出新的精确北京时间 deadline；旧 K100-TAR50 deadline 不迁移到 RACER24，也不由 Codex编造。
- 后续若用户提供 exact Project conversation/receipt，则在同一记录上补充 provenance；不得把手工转交追溯改写成浏览器已审计。

## 5. Builder MCL 回执

只读 MCL 已完成并保存为：

`docs/aris/ZOOMTOKEN_RACER24_ITERATION0_MINIMAL_CHANGE_PLAN-2026-08-31.md`

MCL 确认 selected-Q/full-KV 与 parameter-free completion 都是新增机制，不是现有 helper；Iteration-0
最小面为 `vit_adapter.py`、一个 RACER24 config、一个 focused test、一个 task-specific profiler和一个
microbenchmark launcher。用户已于 `2026-08-31T01:23:36+08:00` 明确确认：将这份手工转交材料提升为
RACER24 Iteration-0 实施授权。该确认解冻最小实现、focused checks、独立 Critic、结果盲 Evaluator 与
冻结的 real-shape microbenchmark；它不追溯补齐缺失的 Project provenance，也不授权 60-epoch训练、
正式 full-stack cost、FARM24 或 PairLatent32。
