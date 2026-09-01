---
type: source_record
title: "ChronoTransport 02199f8 independent written-spec review"
reviewer_route: "new secondary Codex agent with empty conversation fork"
reviewer_task: "/root/independent_ct_r1_spec_audit"
reviewed_commit: "02199f8"
verdict: REVISE_SPEC_BEFORE_PLAN
status: independent_review_complete
updated: 2026-07-12
---

# ChronoTransport `02199f8` Independent Written-Spec Review

## Independence certificate

按用户要求，启动了一个不复用 Sartre、`fork_turns=none` 的新 reviewer。它直接从 Git
commit `02199f8` 读取书面规格，先形成 sealed provisional critique，之后才读取原 Pro
review、`b74101d` 与前两份本地/独立审计。全程 read-only，未连接远端或运行 GPU。

## Final verdict

`REVISE_SPEC_BEFORE_PLAN`。禁止对 `02199f8` 调用 writing-plans。reviewer 确认 all-row
TIA、pre-adapter heavy cache、hard validity 47/embedding cap 8、requested/executed cost
分离均正确，但书面协议仍有六个 material blockers。

## P0 amendments

### 1. Candidate/video exposure confounding

`02199f8` 的 `candidate=(update+offset)%16` 与 offsets `0/4/8` 使 candidate index mod 4
始终等于 canonical video index mod 4；每个 candidate 只观察一个固定视频位置子群。

替换为：对 canonical fit-video index `j`，令 `b=floor(j/16)`、`p=j mod16`：

`candidate=(p+5*b+seed_offset) mod16`，seed offsets 仍为 `0/4/8`。

本地复算与 reviewer 均验证：每 seed 为 12×9+4×8；汇总 candidates 0–3 各 27 次、
4–15 各 26 次；每个完整 block 是 permutation；每个 candidate 在四种 `p mod4` 上各
出现 6 次；每个视频跨三 seed 获得三个不同 candidates。validator 必须断言这些性质并
hash candidate×video matrix。

### 2. Video/window sample unit

split 定义的是 200 video IDs，但 train pipeline 使用 `random_trunc`，val/test 使用
`sliding_window`；`02199f8` 直接把 30 videos 写成 30 windows，conformal 交换单位未闭合。

最小修复固定 option A：Gate 1–3 前，以 label-free SHA-256 规则为每个 train video 冻结
恰好一个 768-frame window。window manifest 记录 video ID、source length、temporal start、
全部 sampled/padded indices、valid mask、data/annotation hash 与 window hash。三 seed 和
所有 schedules 使用同一 temporal window；paired branches 共享 materialized tensors/RNG。

Stage B 固定 batch size 1、world size 1、shuffle false、140 updates。calibration/evaluation
各恰好 30 manifested windows；outer bootstrap unit 是该 unique window（本协议中一一对应
video）。Gate 4 明确属于不同的 official full-video/sliding-window population；Gate 3 的
conformal/coverage 不得自动转移成 Gate 4 deploy guarantee。

### 3. Executable Stage-C gradient ownership

定义 object-identity-disjoint sets：`A`=AdaTAD adapter parameters、`T`=transport、`R`=risk。
同一 forward 产生 detector loss `LD`、feature loss `LF`、risk loss `LR`，使用同一 scaler：

- `autograd.grad(scale(LD), A∪T, retain_graph=True)`；
- `autograd.grad(scale(0.1*LF), T, retain_graph=True)`；
- `autograd.grad(scale(0.1*LR), R)`。

最终 `A.grad=gD`、`T.grad=gD+gF`、`R.grad=gR`，然后一次 unscale、finite audit、global
clip=1、step/update。这样 LF 可穿过 trainable adapter Jacobian 到 T，却不写 A.grad。
全-HOLD schedule 下 T grad 为 None/0 是结构预期；TRANSPORT exposures 汇总 T grad 必须
nonzero finite。generic name-substring optimizer 不可用于 Stage C。

### 4. Matched AMP retry

Stage C 固定 global batch size 2、world size 1、no accumulation、drop_last false，即每 epoch
70 successful updates、60 epochs 共 4200。两个 arms 不要求相同 skip vector；任一 arm
overflow 时，不推进 batch/sampler/augmentation RNG/schedule/LR/EMA/successful index，保留
GradScaler backoff 并重试同一 materialized batch，最多三次，第四次为 INVALID。

retry 还必须恢复所有 forward-mutated model buffers/Python state，尤其 train-mode
`AnchorFreeHead.loss_normalizer`；清空 grads，但不恢复 scaler。回归测试要求除 scaler 外
完整 model state bitwise unchanged。arms 以相同 ordered successful batch/augmentation hashes
与 4200 common-adapter updates 匹配，而不是 attempted/skip-vector 相等。

### 5. Gate-1 comparator

原 shuffle hard test 对 per-window minimum oracle 近乎必然成立，应删除。evaluation-best
static 可以作为只读 evaluation adjudicator 中的 hard scientific oracle comparator，因为
Gate 1 只声称 oracle headroom；它仍不得进入 deployment、checkpoint 或后续拟合。

在每个 bootstrap replicate 内重新选择 evaluation-best static 与其他 adaptive comparators；
joint oracle 相对最强 comparator 的 mean relative reduction≥10%，paired 5000-window
bootstrap absolute improvement CI lower>0，同时保留 B* 相对 dense p50 saving≥20%。
Gate 3 才负责 deploy-visible input dependence。

### 6. Immutable identity registration

profile/replay 前必须冻结 dense checkpoint registry ID+SHA-256、annotation SHA-256、video/
window manifest hash 与 config hash。允许使用独立 pre-Gate1 registration artifact，但该
artifact 必须先完成、不可读取任何 profile/replay/evaluation 结果。

## P1 clarifications

- 140 updates 保留；失败只否定固定 140-update/head/library 的 bounded protocol，不否定
  所有未来 transport ideas。只增加 fit-side integrity diagnostics，不增加训练步数。
- Gate 3 每 seed non-dense≥6/30，pooled selected rows≥18，至少 10 个 distinct evaluation
  windows 被任一 seed 选择。empirical coverage point≥0.85；cluster lower CI 必报但不作
  0.85 hard gate；>0.95 仅记 OVERCOVERED。constant baseline 明确定义为每 schedule 的
  fit-only empirical tau=.9 quantile。
- Gate 4 冻结 hashed matched invocation set，并以 deterministic balanced crossover 交错
  CT/dense/static 顺序。bootstrap outer official video，再 matched invocation，再 seed；
  latency 15% 用 one-sided 95% LCB≥0.15，mAP drop 1.5 用 one-sided UCB≤1.5，overhead 用
  paired margin LCB>0。
- 冻结 action/group embedding dimensions 为 8/8。
- motion threshold 必须定义 aggregate calibration count/tie rule，不能声称一个 global
  threshold 保证每窗口 exact count。
- Gate 4 只解锁 metric+latency；deploy/paper 保持 false，等待外部验证与新颖性复核。

## Consensus and strongest objection

本轮 reviewer 与前两轮一致接受 repaired cache/adapter/age/cost 语义。它新增识别了简单
offset 的 mod-4 混淆、video/window 单位错位、Stage-C loss-specific autograd、AMP retry
buffer mutation、Gate-1 shuffle tautology和 identity freeze 缺口。

最强反对意见：`02199f8` 可能把 deterministic video-order effects 误判为 schedule effect，
同时没有闭合 30 个 conformal units 或 Stage-C owned-gradient algorithm；因此不能进入计划。
