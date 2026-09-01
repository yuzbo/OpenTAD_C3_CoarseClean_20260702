---
type: source_record
title: "ChronoTransport CT-P3R-3S-r1 local source audit"
reviewed_commit: "375094d"
reviewed_spec: "b74101d"
status: local_source_audited_before_r1_design
updated: 2026-07-11
---

# ChronoTransport CT-P3R-3S-r1 Local Source Audit

## Scope and evidence boundary

本页把 Pro review 中未见本地源码的推测，与 commit `375094d` 工作树中的实际实现逐项
对照。它是静态源码审计，不是新的 GPU 行为测试或实验结果；按既定约束，真实行为、
profiling、replay、训练和 detector evaluation 仍全部留到远端执行。

上游语义同时复核到固定版本：OpenTAD
`1aa8ca4ac5e846b1e8ff69298dd6607121a01589`、AdaTAD
`25e06c720e450298ca5267fda6927f3591dcdfef`、VideoMAE
`14ef8d856287c94ef1f985fe30f958eb4ec2c55d`。

## Confirmed blocking defects

1. `risk.py` 仍对 `48×3` 个非负 cell risk 直接求和；失败的 cell-sum/window-target
   尺度没有被新 window-level head 替换。
2. `formal_stage_b.py` 仍以 candidate row 校准 residual，并把所有 candidate rows
   pooled 成一个 Spearman；没有 simultaneous per-window max residual、window-vector
   ranking 或 unique-window cluster bootstrap。
3. formal runner 为每个 seed 生成/校验自己的 split manifest；它没有实现 seed 3407
   一次冻结、3408/3409 复用的共享 `140/30/30` manifest。
4. production schedule library 只有 9 个 non-dense candidates，formal runner 只轮转 6 个；
   缺 P4/P8 HOLD、三组 layer/joint HOLD 和 reverse-T/H，未达到冻结的 16 个 non-dense
   candidate library。
5. Stage B 的一次 epoch 在 200-video THUMOS split 下可形成 140 次 `optimizer.step()`，
   但 artifact 没有 attempted/successful/skipped/LR/EMA/per-schedule exposure 计数，也没有
   对 16-candidate 冻结顺序的训练合同。
6. profiler 只保存内部 stage samples；没有仓库内 end-to-end total-sample 采集 runner。
   cost lookup builder 能消费外部 `latency_ms`，但没有证明这些值来自完整全栈计时，也没
   绑定 software fingerprint、commit/spec/library/split/cost provenance。
7. dense fallback 没有记录 `safety_override_budget_violation`，会把安全回退与预算可行性
   混在一起。
8. calibrated checkpoint 只绑定 split hashes、offset 和旧 P3 status；缺 source/spec/
   library/cost/calibration hashes、successful updates、seed 和完整 claim/unlock chain。
9. Gate 1、三 seed Gate 2/3 hierarchical statistics、Stage C、matched dense control、
   post-Stage-C recalibration、P5/full-stack Gate 4 都尚未实现；Stage-C config 只是声明。
10. endpoint/high-IoU/short-action proxy、fit-only duration quartile、NVML secondary energy
    仍无闭合的代码定义或 runner。

## Pro risks that are not current Stage-B bugs

1. Stage B 不使用基座 `optimizer.backbone.lr=0` 的通用 optimizer；factory 在
   `configure_stage_b()` 后直接对 requires-grad transport/risk 参数创建 `AdamW(lr=1e-4)`。
   因而“当前 Stage B 静默冻结 transport/risk”不成立。Stage C 若直接走通用 optimizer
   则仍会冻结这些非 adapter 名称参数，必须另建显式 optimizer groups。
2. `set_stage_b_module_modes()` 先将 detector/head 设为 eval，仅 transport/risk 为 train；
   `AnchorFreeHead.loss_normalizer` 因此不会在 paired replay 间更新。replay 还会在 dense/
   counterfactual 前后恢复 Python/NumPy/Torch/CUDA RNG，所以 reviewer 担心的 head EMA
   candidate-order 污染当前没有源码依据。r1 仍需 permutation regression test 固化该合同。
3. ChronoTransport config/validator 已显式保证 `frame_selector=None`、官方 rpn head 相等，
   并把 packed-tubelet route 设为 `None`；backbone constructor/forward 也有双重互斥保护。
   因而 DUCA/physical-grid/packed-route 同时激活不是当前 CT 配置事实。

## Additional local defect missed by the reviewer

当前 `_run_group()` 对每个 block 都在完整 provisional tensor 上计算 AdaTAD adapter，
但只把 adapter 输出写回 RECOMPUTE rows；HOLD/TRANSPORT rows 保留 adapter 前的 provisional。
因此 `adatad_adapter_innovation_remains_dense=True` 只在“计算发生”意义上成立，在实际状态
更新语义上不成立。它还使 skipped rows 绕过官方 TIA，混淆了“节省 heavy attention/MLP”
与“改变 adapter 语义”两个因素。

r1 应把 action 明确定义在 heavy attention/MLP 子路径：先按动作恢复完整、顺序正确的
heavy-output tensor，再对所有 rows 应用原 AdaTAD adapter。HOLD 的 bitwise 语义限定为
“heavy 子路径输出复用 latest cache”，不能再误写为“完整 block 最终输出不变”。

## Local verdict

不完全同意 Pro。统计与成本协议修订基本接受；三个 Stage-B 本地 P0 推测被源码否定；
同时本地实现比 reviewer 所见风险更不完整，尤其是 adapter 有效语义、共享 split、冻结
candidate library、全栈计时、provenance 和 Stage C/P5。唯一推荐路线仍是 Route B，
但必须采用吸收上述本地事实的 `CT-P3R-3S-r1-local-corrected`，不能直接照抄 Pro patch，
也不能直接执行 `b74101d`。
