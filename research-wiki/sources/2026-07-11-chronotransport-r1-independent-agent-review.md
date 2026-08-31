---
type: source_record
title: "ChronoTransport r1 independent agent review"
reviewer_route: "secondary Codex agent with empty conversation fork"
reviewer_task: "/root/independent_ct_r1_review"
verdict: LOCAL_CORRECTED_R1
status: independent_review_complete
updated: 2026-07-11
---

# ChronoTransport r1 Independent Agent Review

## Independence certificate

用户要求把当前裁决交给一个完全独立的 agent。reviewer 以 `fork_turns=none` 启动，
只获得仓库路径、原始 Pro review、原规格、固定上游版本和待回答问题；禁止修改文件、
连接远端或运行 GPU。它被要求先形成 provisional findings，再读取主 agent 的 local audit
做第二轮反证。reviewer task 为 `/root/independent_ct_r1_review`。

本 review 没有重跑实验；所有内容均为 repository fact、source inference 或 design proposal。

## Round 1: sealed provisional review

独立 provisional verdict 为 `LOCAL_CORRECTED_R1`，不是全盘接受 Pro，也不是终止路线。
reviewer 在读取主 agent local audit 前独立确认：

1. Stage B 使用独立 AdamW，当前不存在 base optimizer `lr=0` 冻结。
2. head 在 paired replay 中为 eval，RNG 被恢复，当前没有 loss-normalizer 顺序污染证据；
   仍应加入 schedule permutation regression test。
3. CT config 已关闭 frame selector 与 packed route，当前不存在 DUCA/packed 同时激活事实。
4. 当前 adapter 只写回 RECOMPUTE rows，改变了官方 AdaTAD block 语义。
5. 正确边界是每个 block 缓存 pre-adapter heavy attention/MLP 输出；动作只门控 heavy
   子路径；恢复完整 48-row heavy surrogate 后，对全部 rows 应用原 AdaTAD adapter；
   adapter 输出不回灌同一 block rolling cache。
6. 当前 library、formal schedules、split、statistics、profiler 和 Stage C 均不足以执行 r1。

## Round 2: adversarial comparison

reviewer 读取主 agent local audit 后没有发现 material disagreement，但给出以下新增问题。

### P0: cache-age and frozen-library contradiction

`hold_only` 与 `transport_only` 只有首 clip RECOMPUTE，此后连续 47 个 HOLD/TRANSPORT；
生产配置却固定 `max_cache_age=8`。结果是 learned scheduler 把这两个候选判为 infeasible，
forced runtime 又会在 age>8 时静默修成 RECOMPUTE。候选名、冻结动作矩阵、library hash、
profiled cost 与 executed schedule 因而不一致。

reviewer 推荐在 r1 中拆分两个概念：

- `hard_cache_validity_age=47`：允许 48-clip 控制按名字原样执行；
- `transport_age_embedding_cap=8`：保留当前 transport age embedding 结构并对输入 clamp。

另一合法选择是从 schedulable/training library 删除 only schedules 并给 repaired controls
重新命名，但这会改变已冻结 candidate library，因此不作为首选。

### P0: requested cost is not executed cost

非线性 cost lookup 当前按 nominal library row counts 查询；runtime repair、transport NaN
fallback 或 whole-window dense fallback 会修改实际动作，但 summary 仍携带请求 schedule 的
`estimated_cost`。r1 必须同时保存 requested/executed schedule 与 cost；任何 repair 后，
nominal cost 失效，必须以 executed-shape lookup 或实际 total sample 记账。

### P1: Stage-B precision and exposure

当前 bespoke Stage-B trainer 没有 autocast/GradScaler，所以 AMP skip 不是当前 bug。
r1 必须明确冻结为 FP32，或明确引入 AMP；不能在 artifact 中伪造不存在的 AMP skips。

140/16 无法整除：固定顺序会让前 12 个 candidate 每 seed 暴露 9 次、后 4 个暴露 8 次；
而 loader `shuffle=False` 会把 candidate 与 video order 绑定。reviewer 建议为三 seed 冻结
不同 round-robin start offset，并完整登记 candidate×video exposure；在此之前不能称 balanced。

### P2: EMA alias equality

`risk_predictor` 同时以 `scheduler.predictor` 注册。当前 checkpoint/校准代码看似同步两个
别名，没有确认 bug；r1 仍应在 raw/EMA/calibration state 上断言 canonical/alias tensor
完全相等，并只对逻辑 predictor 计算一次 hash。

## Final consensus

最终 verdict：`LOCAL_CORRECTED_R1`。

必须在新 spec SHA 前冻结：pre-adapter heavy cache + all-row TIA；cache validity 与 embedding
age cap 分离；exact 16-candidate actions/hash/exposure；seed-3407 共享精确 140/30/30 split；
window quantile/statistics；dense risk=0 与 B*；requested/executed cost 分离；full-stack total
samples；显式 Stage-C optimizer/matched control；EMA alias equality。完成这些定义前，不得
写实现代码或运行 Gate 1。

reviewer 对主 agent 的最强反对意见是：只补齐 16-candidate library 仍不够；若不先解决
`max_cache_age=8` 与 only schedules 的矛盾，所谓冻结 library、成本和 provenance 仍是假的。
