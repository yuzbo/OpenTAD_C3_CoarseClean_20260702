---
type: idea
node_id: idea:chronotransport
title: "ChronoTransport 动态特征刷新"
stage: spec_approved
outcome: bounded_appeal_pending
implementation_status: partial_repair_in_progress_registration_blocked
registration_status: NOT_READY
tags: ["feature-refresh", "transport", "parallel-route"]
added: 2026-07-11
updated: 2026-07-13
---

## r2 foundation repair batch (2026-07-13)

### Manifest/protocol implementation slice

The approved r2 label-free protocol slice now has remotely tested implementations for the exact
200-video/200-window manifest and deep re-derivation validator, canonical exact-byte serialization,
hash-bound Stage-B exposure artifact, complete Stage-C exposure/cursor validation, frozen control
algorithm identities, and deep canonical-library validation. A canonical atomic manifest builder was
added, and the legacy six-schedule formal Stage-B `run`/`main` route is unconditionally disabled.
The tests-first remote RED was the expected missing `r2_control_algorithm_identity` import; later REDs
exposed trust in the registry-provided source index vector and strict canonical/type gaps. The repaired
focused suite passed 27/27 remotely in 53.81 seconds, and the broader protocol/control/legacy-runner
matrix passed 55/55 in 91.90 seconds. This slice is `tested`, but registration and every formal Gate
remain blocked.
Two independent reviewers then returned `APPROVE_PROTOCOL_SLICE` and
`APPROVE_PROTOCOL_QUALITY`; the bounded repair was committed and pushed as `33378af`. This is an
intermediate implementation commit, not immutable implementation `I` or registration `R`.

The first Gate-1 cost/profile draft then passed 36/36 remote focused checks and got the B*/20% formula
right, but two independent reviewers rejected its evidence boundary. The rejected draft could accept
registration-external factories/provenance and arbitrary 30+30 record IDs, did not freeze the exact
23-profile candidate/order contract, and retained coercive scalar/safety-override paths. The selected
repair is deep registration binding and strict artifacts; weakening registration or trusting caller
hashes is explicitly rejected.

The remote deployment audit also rejected the earlier launcher assumption that
`CUDA_VISIBLE_DEVICES=1` proves physical GPU1. With Slurm task/cgroup isolation, the physical index is
carried by `SLURM_STEP_GPUS`/`SLURM_JOB_GPUS` and a single assigned device is remapped to local ordinal
0. The selected contract is physical ID `1` plus local `CUDA_VISIBLE_DEVICES=0`; no currently running
allocation is reusable. Reserving two GPUs merely to expose local ordinal 1 is rejected as wasteful.

The repaired Gate-1 implementation now uses deep registration v2, exact 23x200 profiling and exact
manifest-bound record artifacts; remote focused and complete ChronoTransport matrices passed 24/24
and 203 passed/1 CUDA-only skip, respectively. Separately, Stage-C ownership/AMP rollback reached
15 passed/1 protected-CUDA skip plus 27/27 compatibility checks. Both slices remain pending independent
approval and do not unlock profiling, a Gate, Stage C, or a claim. The Stage-C review subsequently
returned `BLOCK_STAGEC_SLICE`: omitted adapters, wrong optimizer hyperparameters, infinite aggregate
norm, hidden Python state and caller-asserted action identity all had reproducible fail-open paths. The
primitive is being repaired; the matched-dense 4,200-update runner remains a separate unimplemented
surface.

Formal Stage B separately reached remote 20/20 focused and 71/71 compatibility verification with
exact 140-success/retry/resume semantics, but independent review rejected its formal boundary: an
arbitrary factory could bypass real OpenTAD construction, the dense checkpoint was not strict-loaded,
paired replay/order checks were self-reported, and fit-only 140x16 rank-127 baseline generation was
missing. Gate 1 was likewise rejected until its registration reads and verifies real Git/filesystem,
manifest, checkpoint and paired-replay evidence instead of accepting self-described hashes. Both
slices are under repair; no formal experiment is unlocked.

The first bounded repair batch corrected the r2 Stage-B/Stage-C resolved-config path so the overlay
only reaches `model.backbone.backbone.chronotransport`; added window-level Gate-3 conformal logic
that reduces each complete `30 x 16` residual matrix to 30 window maxima before rank 28; kept the
fit-only per-schedule 140-target rank-127 statistic as a separate API; and made formal profiler
validation require direct samples for every required stage plus `total_ms`. Missing placeholders now
carry `p50/p95=None` and cannot validate as measurements. The remote config regression completed a
valid RED-to-GREEN cycle (1 failed, then 1 passed). The Torch-focused risk/profiler suite was launched
remotely: the four risk checks and two profiler checks passed, while one profiler test failed only
because its expected error-message regex did not match the fail-closed missing-`total_ms` error. The
message was aligned afterward, but no retry result is claimed. Therefore the risk/profiler production
changes remain `implemented` rather than newly `tested`. This batch does not complete r2, unlock
registration, or create experiment facts.

Independent follow-up review tightened this foundation slice before re-verification: r2 now uses
dedicated no-coverage-override helpers with fixed ranks 28 and 127, while the legacy generic conformal
API remains backward compatible;
the resolved r2 overlay explicitly neutralizes inherited legacy `max_cache_age` with `None` while the
two r2 age fields carry the executable contract; profiler accounting now
separates `innovation` from `dense_adatad_adapter`, requiring direct samples for both in a formal
summary under the explicit `chronotransport_profile_v2` schema. Tests enumerate the specification stage
set rather than deriving expectations from the production constant. The repaired focused remote matrix
then passed 13/13 in 37.12 seconds; after the compatibility/schema follow-up the same focused matrix
again passed 13/13 in 37.30 seconds. Formal per-invocation aggregation and peak-memory reset remain later full-stack
profiler work and are not claimed by this slice.
The complete remote ChronoTransport regression surface then passed 128/128 in 199.80 seconds after
the scratch checkout was populated with its required `scripts/` and `docs/` files. This foundation
slice is remotely tested, but it does not upgrade the overall r2 status or unlock registration.

# ChronoTransport 动态特征刷新

## One-line thesis

保持外部 detector 网格，仅在 VideoMAE time×layer 上选择 RECOMPUTE/TRANSPORT/HOLD，减少 heavy subpath 重算。

## 为什么提出

避免 pre-backbone 删除帧引起 selected-axis 几何和 full decode 争议。

## 已有证据

Stage-A、paired replay 和正式 Stage-B fit/calibration/evaluation 已落地。`92029ea` 的预注册 P3 science gate 为 FAIL：risk-regret 排序为负，cell-risk/window-target 尺度错配，feature transport 改善不稳定；Stage C/P5 未解锁。

## 当前选择或否定理由

历史 P3 保持负结论，Stage C/P5 仍未解锁。用户批准一次有界上诉，但 GitHub-visible
Pro 复核裁决 `REVISE_SPEC_BEFORE_PLAN`：必须先完成 r2 规格、精确 SHA 与 spec-only
复核，不能把书面修订误写成已经实现或实验支持。

## 风险与失败模式

transport 可能不优于 HOLD；cache 状态与校准；真实 kernel cost。

## 下一次允许采取的动作

修订规格最终冻结为 commit `e4422f5`，exact-byte SHA-256 为
`87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`；空白上下文
spec-only reviewer 已返回 `APPROVE_SPEC_FOR_PLAN`。当前只解锁 implementation plan；实现
完成后仍必须先通过 pre-Gate1 registration 与 Gate 1，才可按 stop chain 继续 Gate 2--4。

## Connections

## r2 实现状态（2026-07-12）

第一批达到局部 `tested`：canonical split/window/exposure protocol、固定 16-candidate library、
motion/random exact-count controls，以及 hard-cache age 与 transport embedding age 分离。远端
focused checks 分别为 7/7 和 36/36；这只是实现证据，不是 Gate 或 science evidence。

implementation plan commit 为 `18cc1c0`；下一项是 runtime live-tensor、all-row adapter 与
requested/executed contract。完整实现和 pre-Gate1 registration 之前不得运行或声称 Gate 1。

runtime 与唯一 window-risk head 随后达到局部 `tested`：runtime/integration focused suite
35/35，risk/core focused suite 30/30。当前下一项转为 paired replay、正式 Stage B 与 Gate
adjudication；仍未注册、未运行任何新 Gate。

独立 implementation audit 裁决 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。110 个远端测试
通过只覆盖已实现子集，不能抵消 Gate 3/4、正式 Stage B/C/matched dense、overflow retry、
B* 与 exact cost、full-stack profiler 及严格 registration input-chain 仍缺失。禁止冻结 I/R。

外部 Pro 随后对固定 GitHub 快照 `4b07020acb2611c3f085488d2f678f3be037f1be` 复核，维持
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`，同意上述七项 blocker，并新增两个 P0：r2 config
overlay 写在 wrapper 层而非 inner ViT runtime；Gate-3 conformal 错把 `30×16` candidate rows
展平，正确单位应是 30 个 window maxima 后取 rank 28。完整 manifest、per-window Spearman、
Stage-C exposure/resume ledger、requested/executed exact cost 和 profiler zero-placeholder 也需
fail closed。Route B 继续保留，但 I/R、profile、Gate 1、新 Stage-B seed、Stage C 和 Gate 4
全部锁定，直到修复后第二次独立审查返回 `APPROVE_IMPLEMENTATION_FOR_REGISTRATION`。

精确状态：科学规格 `designed/approved`；实现和测试均为 `partial`；`experiment_running=false`；
r2 `empirically_supported=false`；`paper_ready=false`；`deploy=false`。审计不产生实验事实，
也不推翻历史 `92029ea` 的负结果。

由 `research-wiki/graph/edges.jsonl` 维护。

## r2 当前实现裁决补记（2026-07-13）

当前路线没有被科学判死刑，但实现仍未取得注册资格。Gate-1 oracle-headroom 的科学问题尚未
真正运行；现在被阻止的是证据链和事务语义，而不是实验结果。已选择继续修复真实 OpenTAD
full-stack backend、formal repository context、Stage-B phase completion 与 Stage-C transactional
retry，不接受用 toy callback、caller hash、payload-only validation 或测试通过数替代。

Gate-1 `random_p{2,4,8}` 存在规格缺口：算法 digest 含 seed，但 unsuffixed comparator schema
没有写明用 3407/3408/3409 中哪一个。推荐最小修订为固定 3407，以保持六 comparator schema、
与 split root seed/已有 control test 一致，并避免把 3 seeds 作为 multiple-comparator search；
在用户批准和 spec-only 复核前，正式代码保持缺 seed 即 fail closed。

精确状态：protocol slice `tested+approved`；Gate-1 backend/registration `implemented_under_repair`；
formal Stage B 与 Stage-C primitive 均为 `tested_then_rejected_under_repair`；
`experiment_running=false`、`empirically_supported=false`、`paper_ready=false`、`deploy=false`。

## r2 最小协议修订提案（2026-07-15，未批准）

`docs/methods/2026-07-15-chronotransport-r2-minimal-protocol-amendment-proposal.md` 把四个不可唯一
执行的问题写成 A1--A4：unsuffixed random controls 固定 seed 3407；改为 Slurm 分配单 GPU、
不覆盖 visibility、进程使用 `cuda:0`；train-mode `loss_normalizer` 仅在成功更新推进并要求两
arms 精确同轨；Stage C 明确一次 no-grad dense reference 加一次 differentiable counterfactual
forward，并从同一 batch-two head 结果产生逐窗口 regret target。状态仅为 `discussed/proposed`，
SHA-256 为 `30371FFC17B02DF615FF0D772B93BADF30CF0A3AB84E36325CBF5A71EFD8469F`；它没有修改冻结 spec，
也未解锁 runner、I/R 或 formal job。

## Gate-4 证据边界补记（2026-07-15）

纯统计 adjudicator 已降为显式 test-only：caller-owned timing/metric/regret mappings 不再能够用
`formal=True` 铸造 Gate-4 schema。正式路线仍必须另行实现 repository-owned official-population
producer、Stage-C/post-Gate3/checkpoint/static/profiler/R 绑定和原子 terminal。当前状态只是
经 `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL` 独立复核的 bounded 完整性修复，不是 Gate-4
implementation completion，也没有产生 formal evidence 或实验结果。

## Stage-C 测量成本证据补记（2026-07-15）

formal Stage-C 不能把 proxy cost 的 `cost_is_measured=False` 当成可接受的布尔字段；该路径已
RED-first 修复，并经 `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK` 独立限定复核。测试夹具现在使用
显式 `_TEST_MEASURED_COST`，只用于验证正常 formal runtime 后的篡改拒绝，不能冒充正式 profile。
真实路线仍须绑定 profile artifact hash、硬件/环境、producer、requested/executed cost 与 I/R；
因此状态只是 bounded primitive lock，不是 Stage-C workflow、registration 或实验完成。

## `b854adb` GitHub Pro 审计吸收（2026-07-15）

review-only immutable snapshot 已通过外部快照门，但总体仍为
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。该结论被接受：当前缺真实 ActionFormer Stage-C
桥、4,200 successful-update candidate/matched-dense workflow、repository-owned Gate-4 producer、
完整 source vector、A1--A4 与 profile provenance。路线没有被科学判死，也没有新 Gate 结果。

具体实现建议不盲从。registration 应立即承认两份已确认 integration-test omission，但完整性检查
必须基于显式 formal/legacy/nonformal 分类，不能让裸 `test_chronotransport*.py` glob 自动决定 R。
Pro 提议的 Stage-C evidence dataclass 仅作为需求清单吸收，必须等 A3/A4 唯一化 official loss、
per-window regret、forward order 与 `loss_normalizer` 后才可冻结。未来 launcher 也必须服从 Slurm
分配单设备和逻辑 `cuda:0`，不能沿用 physical-GPU1/CVD=1 语义。

## A1--A4 规范冻结与下一轮 Pro 严审（2026-07-15）

用户已授权 exact A1--A4，spec-only commit `537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`
把它们写入权威规范；新规范 SHA-256 为
`E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`。状态只从
`discussed` 迁移到 `designed/pending_external_spec_diff_review`，没有迁移到 `implemented` 或
`tested`。当前生产代码仍须逐项证明 A1 seed、A2 Slurm identity、A3 normalizer transaction 与
A4 paired official loss/regret forward；此前缺失 workflow/source/provenance blocker 也没有因规范
提交消失。

下一轮 GitHub-only Pro prompt 固定要求：fresh resolve `537f692` 的严格后代；先给独立
`APPROVE_SPEC_FOR_PLAN`/`REVISE_SPEC_BEFORE_PLAN`，再给 implementation verdict、RED tests 与完整
实现建议。该 prompt 是只读审计输入，不创建 I/R、PRECHECK、Job、Gate 或论文数字。
