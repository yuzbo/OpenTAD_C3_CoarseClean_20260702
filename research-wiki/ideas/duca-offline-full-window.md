---
type: idea
node_id: idea:duca-offline-full-window
title: "DUCA 离线全窗口 pre-backbone 插件"
stage: active_candidate
outcome: pending
tags: ["duca", "offline-full-window", "pre-backbone"]
added: 2026-07-11
updated: 2026-07-22
---

# DUCA 离线全窗口 pre-backbone 插件

> Canonical contract: `research-wiki/duca_final_model_contract.md`. 本页保留方法演化、
> 外部审查和负证据；最终模型身份、训练日程与实验停止条件以 canonical contract 为准。

## One-line thesis

同一 forward 内由可训练 coarse probe 和结构化 selector 生成 hard sparse observations，送入 official-derived TAD detector，并接受 detector feedback。

## 为什么提出

将粗分类、间接选择和检测放入统一训练图，同时保持推理无 ledger、无 teacher、无 cache。

## 已有证据

70aa fixed-384 已完成并稳定达到 58.39；a5e selector、official-derived backend、exact-K/
max-gap structured DP、joint graph 与 profiler 已实现。Pro 与本地复核确认 hard/soft DP 属于
同一可行族，但 current hidden/direct-head 机制不符合间接边界假设，full-model optimizer
coverage、one-swap、geometry、matched baseline 和正式成本仍缺。

## 当前选择或否定理由

冻结 a5e 为 direct-boundary joint baseline。唯一下一候选是 Shared-ASFormer Transition-Only；
MobileNet、MUST 和新增 heads 继续冻结。通过 mechanism、成本、one-swap、geometry 与
matched baseline 后才可成为主方法。

## 风险与失败模式

full-window probe 成本、selected-axis geometry、hard utility 方向、effective K 波动、external
ASFormer provenance、direct-head bypass 和只在 AdaTAD 上验证。

## 下一次允许采取的动作

先完成 G18/G19/G20 的 transition-only implementation gate，再完成 G1/G2/G3/G4/G6/G8。

## Connections

## 2026-07-11 Transition-Only implementation update

- Status advanced from `designed` to `tested`, not to `empirically_supported`.
- Isolated branch `codex/duca-transition-only-20260711` preserves `a5e1774` as
  the direct-boundary baseline.
- Implemented true official-ASFormer encoder hidden capture, transition-only
  relational descriptors, one shared scorer, protected gradient routing,
  continuous exact-DP alpha/beta homotopy, balanced binary BCE, fixed sigma=2
  truncated-radius=4 transition targets, and audited component learning rates.
- Slurm gate `1159350` completed: 26 focused tests passed; official
  ActionFormerHead train/test and optimizer step passed; exact-K/max-hole passed;
  transition/coverage/detector-only gradient ownership passed.
- The pre-backbone cost smoke is selector-only and cannot support a full-stack
  efficiency claim.
- P0 evidence remains exact-uniform, matched direct-a5, transition beta=0, and
  transition beta=0.25. Dynamic MUST, MobileNet, and new selector heads remain
  frozen.

由 `research-wiki/graph/edges.jsonl` 维护。

## 2026-07-13 Pro audit intake

`0ea4e15` exact-commit review 维持 `HOLD`。它发现 direct/legacy stable route 仍有 midpoint
uniform 残留，并把 current raw-pixel bridge、selected-axis geometry、short-window zero
padding 和 dense decode/H2D 成本列为 P0/P1 风险。

reviewer 推荐把下一候选改写成 DUCA-FSU：用 feasible hard one-swap detector gain 蒸馏
utility difference。项目仅登记该建议为 `discussed`；在 common uniform、coarse quality、
one-swap、geometry 和 cost gates 通过前，不视为当前主路线已被替换。

## 2026-07-21 conditional coarse-heavy fusion contract

- 当前 V8 没有把 coarse hidden 与 VideoMAE detection feature 融合。coarse hidden
  只进入 transition scorer；选中索引随后 gather 原始帧，VideoMAE/AdaTAD 只消费
  选中帧路径。这是已确认的结构事实，不得把 selector 看见 coarse hidden 写成 detector
  已复用 coarse context。
- 两种特征不能直接相加。VideoMAE 是动作外观/运动的主检测表征，coarse hidden 是
  动作性、状态变化和低成本全局上下文。首选有界融合是 timestamp-aware gated context：
  VideoMAE selected tokens 作 query，完整 coarse sequence 经独立 adapter 后作 key/value，
  再以零初始化门控残差回到 VideoMAE 主流。门控为零时必须逐值等于现有 VideoMAE
  baseline，官方 AdaTAD projection/head 不改。
- 梯度所有权必须隔离：coarse action/transition/boundary losses 训练 coarse 前端与
  scorer；TAD loss 训练 VideoMAE/AdaTAD 和 fusion adapter，但默认不能进入 action head
  或 coarse trunk；可选 feature distillation 只能以 stop-gradient VideoMAE 为 teacher
  训练 coarse-to-detector adapter。只有独立 gate 证明无破坏后，才允许极小 rho 进入
  ASFormer 最后一块。
- 该 context fusion 能补充全局信息，但本身不填补物理时间空洞。只有后续 canonical
  physical-grid 版本把 coarse context 放到规则时间锚点、把 sparse heavy feature 按
  timestamp transport 回同一网格后，才有依据放宽 max-hole。两者不得混成一个实验。
- 该合同当前状态仅为 `discussed_conditional_post_v8`。Job `1178989` 结束前不实施、
  不新建 selector/decoder/worktree，也不改变 U/G0/G1/G2。

### Three bounded fusion hypotheses

1. **Late prediction fusion (low risk baseline).** Coarse action/transition
   logits only calibrate TAD class/boundary logits. It requires no shared
   feature space, but cannot restore a proposal that the sparse heavy stream
   never represented.
2. **Timestamp-aware gated context (preferred first test).** Selected VideoMAE
   tokens query the complete coarse sequence through independent projections
   and physical-time relative bias. A zero-initialized residual gate makes the
   initial forward exactly equal to the existing VideoMAE/AdaTAD baseline.
   This reuses global context without claiming the coarse feature is a weaker
   VideoMAE feature.
3. **Canonical-grid coarse fallback (higher-risk successor).** Transport sparse
   VideoMAE tokens back to regular physical-time anchors and use an adapted
   dense coarse feature only where heavy-token coverage is low. Uniform
   selection must reduce exactly to the original VideoMAE baseline. This is
   the only candidate that could justify materially relaxing max-hole, but it
   changes the representation contract and needs its own matched experiment.

Direct addition/concatenation before normalization and unrestricted shared
backpropagation are rejected. The preferred training ownership is: coarse
losses update coarse/action/scorer; optional stop-gradient VideoMAE feature
distillation updates only the coarse-to-detector adapter; TAD loss updates the
fusion adapter, VideoMAE/AdaTAD and optionally the scorer through the existing
protected bridge, but not the coarse action head/trunk. A later tiny-rho update
to the final ASFormer block requires a separate gradient-conflict and coarse-
calibration gate.

Because feature fusion occurs after VideoMAE, the resulting method is honestly
an acquisition-and-fusion adapter with a pre-backbone selector and a small
post-backbone hook, not a strictly pre-backbone-only plugin.

## 2026-07-21 V8 supervision audit and nontrivial-event requirement

- Exact-commit review confirms a semantic defect in the active transition
  supervision: one Gaussian target is reused by a mass-based coverage loss that
  continues rewarding repeated occupancy around the same endpoint. This is a
  plausible explanation for learned clustering, but remains a hypothesis until
  a single-variable objective comparison is complete.
- The proposed replacement cannot be copied verbatim. Under current
  `max_unselected_hole=2`, a radius-one endpoint event contains three
  consecutive positions and is already forced to be covered by every feasible
  path. Its exact event loss is therefore zero-gradient for almost every
  endpoint.
- Any successor must first prove that its event definition has policy headroom
  under the frozen K/G family. The bounded candidate is an exact rounded
  `radius=0` unique-endpoint event plus a separately audited event-distribution
  objective; no implementation or experiment is authorized by this note.
- G1/G2 remain surrogate-gradient hypotheses until a real legal hard-swap
  alignment artifact passes the preregistered thresholds. Plumbing connectivity
  and `formal_training_unlocked=True` metadata are insufficient.

## 2026-07-21 Oracle-calibrated boundary-burst target

The desired selector is allowed and expected to cluster observations around
state-transition boundaries. The historical Oracle selects each GT endpoint
center and its radius-two neighbors before globally filling the remaining
budget. The failure to avoid is not clustering itself, but a shifted,
one-sided or unlimited broad-band pile-up that misses other endpoints.

An audit of registered V0--V8 and the inherited PAction/GAS-VT/lattice code
found partial components only: legacy center/radius decoding, learned context
radius, binary left/right bracket loss, symmetric score dilation, global
exact-K/max-hole DP and bracket diagnostics. No version currently predicts a
deploy-visible transition center together with explicit bilateral multi-frame
allocation, capped per-endpoint reward, overlap deduplication and residual
global budget inside the V8 official-AdaTAD joint graph.

The post-V8 bounded design target is therefore:

```text
coarse state evidence
  -> transition-center evidence
  -> bounded q_left / q_center / q_right boundary burst
  -> deduplicate overlapping endpoint bursts
  -> allocate remaining exact-K budget with the existing global DP
  -> official AdaTAD selected-axis detector
```

`q_left/q_center/q_right` denotes the required semantics, not a pre-authorized
new head design. Before implementation, Oracle train-split statistics must fix
the useful radius/count range, and a no-training headroom test must show that
the frozen K/G family can distinguish candidate profiles. Radius-zero unique
coverage is only the center-anchor term. Status is `designed_not_implemented`;
running V8 Job `1178989` remains immutable.

## 2026-07-22 canonical final product contract

### Final deliverable

The intended paper artifact is one **offline-TAD pre-backbone acquisition
plugin**, not a new TAD detector, not Online TAD, and not three separately
deployed models. Its first formal backend is the existing official-derived
AdaTAD/ActionFormer path. Detector generality is a later evidence requirement,
not a reason to fork the selector before the AdaTAD result is sound.

The plugin must consume a cheap dense low-resolution view, select an exact
budget of original observations, and expose original-time coordinates and an
auditable cost record. The heavy VideoMAE/AdaTAD path consumes only the hard
selected observations. Inference uses no GT, teacher, ledger, cache or
pre-extracted actionness JSONL.

The primary paper method is fixed-budget allocation; K=384 is the first anchor
and K=256 is the first efficiency extension. Dynamic budget remains out of the
main claim until the fixed-budget method is empirically sound.

### Canonical model graph

```text
dense cheap low-resolution observations
  -> coarse spatial stem + official ASFormer temporal trunk
  -> binary p_action + coarse hidden sequence
  -> indirect transition descriptors
       (delta p_action, uncertainty/entropy change, hidden change)
  -> transition-center scorer
  -> bounded bilateral boundary-burst profile
       (q_left / q_center / q_right semantics)
  -> overlap-aware saturating union of endpoint profiles
  -> residual global context utility
  -> existing global exact-K / max-hole structured DP
  -> chronological hard gather + original-time metadata
  -> official VideoMAE/AdaTAD + original ActionFormerHead
  -> TAD predictions and mAP
```

`transition-center` is an indirect state-change center inferred from the
coarse sequence. It is not a class-specific start/end detector head. Train-only
GT endpoints may supervise the state-change evidence, but validation/test GT
must never enter the decision path.

The first bounded parameterization should extend the existing V8 scorer with
one small burst-profile output rather than create another selector family. A
normalized finite offset profile represents left/center/right allocation; its
total useful mass is capped by an Oracle-calibrated local quota. Overlapping
profiles combine through a saturating union so duplicate endpoints cannot
multiply reward. The existing global DP still makes the final exact-K hard
decision and spends all residual budget. The exact radius/quota and max-hole
are frozen only after a train-split Oracle reachability audit; `G=2` is not a
permanent scientific truth.

### Objective and gradient ownership

The final objective has distinct semantics:

- `L_action`: binary action/background supervision; updates the coarse spatial
  stem, ASFormer trunk and action head.
- `L_anchor`: centers each predicted transition burst on a rounded endpoint or
  another proven nontrivial center event; updates scorer/burst parameters only.
- `L_bilateral`: requires useful observations on both sides of each endpoint;
  updates scorer/burst parameters only.
- `L_quota`: rewards the Oracle-calibrated local burst only until its useful
  quota is reached, then saturates; it must not reward unlimited broad-band
  occupancy.
- `L_fair/context`: prevents a strong endpoint from consuming all local budget
  and preserves residual global context without one-frame-per-cell geometry.
- `L_TAD`: trains the official detector and may reach scorer/burst parameters
  only through a protected bridge that passed a real legal hard-swap alignment
  gate. It must not rewrite the coarse action head/trunk by default.

The hard forward and differentiable backward must use the same exact-K/max-hole
family. Radius-zero coverage is only `L_anchor`; it cannot replace the
bilateral burst and quota terms.

### Training contract

This is one model with a two-stage curriculum, not a three-model pipeline:

1. **Frontend P0.** Train the cheap coarse branch and boundary-burst selector
   on a sealed train-only split. Action loss trains coarse semantics;
   anchor/bilateral/quota/context losses train the selector. The heavy detector
   is skipped.
2. **Official detector stage.** Start inside the unchanged official training
   budget with exact-uniform detector warmup. Then switch to the learned hard
   policy, freeze the coarse action model, decay but retain selector auxiliary
   supervision, and train AdaTAD plus the selector through the protected
   detector bridge. Inference uses only the learned hard policy.

### Decisive evidence sequence

1. **Reachability before learning:** compare unrestricted GT Oracle, the same
   Oracle projected into candidate K/G families, and exact uniform. Use
   train-only statistics to freeze burst radius/quota and max-hole. If the
   feasible Oracle has no detector-mAP headroom over uniform, change the
   feasible family before training rather than tune losses.
2. **Mathematical/code gates:** brute-force small exact-K/G cases; nonzero
   gradient for every active event; exact K; no leak; deterministic overlap
   deduplication; hard/soft family equality; loss-to-parameter ownership.
3. **Frontend mechanism gate:** under one frozen split compare the old V8
   Gaussian-mass objective, simple `abs(delta p_action)`, and the corrected
   burst objective. Proxies license training only; they never replace mAP.
4. **Matched U versus G0:** run exact-uniform and corrected learned policy with
   detector feedback disabled. If terminal-EMA G0 does not exceed U, stop this
   K/G learned-allocation route before adding detector feedback.
5. **Feedback gate and G1/G2:** run the real legal hard-swap alignment test.
   Only a passing bridge licenses G1; G2 tests the training-only uniform
   companion as a stability ablation, not as part of inference.
6. **Paper closure:** repeat U and the best learned arm for three seeds, add a
   fixed-budget curve, one second detector backend, short-action/high-tIoU
   analysis, selection visualizations and full-stack latency/FLOPs/memory/
   energy including the dense cheap probe.

The primary paper claim is supported only when the learned plugin beats the
same-commit exact-uniform detector at matched budget and lowers measured total
inference cost relative to the dense route. Detector-gradient usefulness is a
separate supporting claim and exists only if G1 beats G0 after alignment.
For the current THUMOS/AdaTAD anchor, the preregistered GO line is terminal-EMA
Avg-mAP at least `65.00`, improvement over matched U of at least `0.20`, no more
than `0.20` loss at mAP@0.6 or mAP@0.7, and lower measured end-to-end inference
cost than the dense route.

Status: final product contract `designed`; implementation
`designed_not_implemented`; empirical status `unproven`. This section does not
rename V8, create V9, or authorize modification of the frozen V8 experiment.

## 2026-07-22 EU-CRR Pro 审查吸收边界

精确提交 `63e25eb` 的复核再次确认：coarse hidden 当前只进入 transition
descriptor/scorer，hard selected RGB 再进入 VideoMAE/AdaTAD；detector 没有消费
coarse hidden。该事实不等于“缺少 feature fusion 是 learned selection 低于 uniform 的
首因”。当前更强的解释仍是 hard/surrogate utility 不一致、selected-rank 与 physical
time 语义、coarse evidence 可辨识性和 K/G 可行域。

外部审查提出的 EU-CRR 只登记为
`discussed_conditional_diagnostic_not_authorized`：在 exact-uniform K384 下，把 hard
positions 对应的 frozen/detached coarse hidden 经 `LayerNorm + Conv1d` 映射后，以
全零 channel gate 残差加到 post-VideoMAE、pre-projection feature。它只能回答 frozen
coarse representation 是否对 detector 有增量价值，不能回答 learned boundary-burst
selection 是否有效。

主线优先级不变：V8 终局 -> R0 train-split Oracle K/G reachability -> G23
boundary-burst R1--R3。EU-CRR U0/U1 若以后获准，必须作为正交单变量实验，并同时报告
`U1-U0`、`L1-L0`、`L0-U0`、`L1-U1` 四个 contrast。U1 失败只否定
post-VideoMAE coarse residual；不得据此否定 G23。U1 成功则方法需诚实改称
acquisition-and-fusion adapter，不能继续称 strict pre-backbone-only plugin。
