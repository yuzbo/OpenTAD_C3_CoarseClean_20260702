---
type: source_absorption
node_id: source:chronotransport-r2-pro-review-1b6366d-20260716
source_file: research-wiki/sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-verbatim.txt
source_sha256: C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08
review_sha: 1b6366d0acb712e8096c2cceb0f05e66b16d30d4
tree_sha: 3fc64c72cf26b77f041d059f51385f29e5e85462
parent_sha: 537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37
spec_verdict: APPROVE_SPEC_FOR_PLAN
implementation_verdict: REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
disposition: partially_accepted_with_explicit_corrections
updated: 2026-07-16
---

# ChronoTransport r2 Pro review absorption at `1b6366d`

## Bottom line

不完全认可具体建议，但完全认可两项顶层裁决及其主要事实依据：A1--A4 规范差异通过
`APPROVE_SPEC_FOR_PLAN`；当前实现仍为 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。
这意味着规范状态可记为 `spec_approved`，但实现、registration、PRECHECK、正式作业和任何
科学 claim 均未获批准。

本文区分三种内容：固定快照直接证明的事实、可采纳的实现要求、以及 reviewer 给出的非权威
示例补丁。接受总体裁决不等于授权逐字复制接口、文件分类、launcher 或 kill 语义。

## Source and snapshot integrity

- 用户附件已逐字节归档为
  `research-wiki/sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-verbatim.txt`。
- 附件与归档均为 86,871 bytes、2,019 lines，SHA-256 均为
  `C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`。
- review certificate 固定 `REVIEW_SHA=1b6366d0acb712e8096c2cceb0f05e66b16d30d4`、
  `TREE_SHA=3fc64c72cf26b77f041d059f51385f29e5e85462`，唯一 parent 为
  `537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`。
- reviewer 明确没有执行 tests、CUDA、Slurm、training、profiling 或 evaluation。因此它是代码与
  规范审计来源，不是实验结果。
- `b854adb..1b6366d` 没有 ChronoTransport production/tool/test/config/launcher 字节变化；新增内容
  是 A1--A4 规范、审计 prompt、审计记忆与 `.gitattributes`。旧发现只有经当前源码重新读取后才
  可作为当前事实。

## Accepted without reservation

1. A1--A4 是最小且一致的规范修订，未改变 seeds、split、candidate library、threshold、更新数、
   bootstrap unit、official population 或 stop-chain；`APPROVE_SPEC_FOR_PLAN` 只批准实现合同。
2. 当前 registration 仍绑定旧 `e4422f5`/旧 spec hash；formal random lock 同时拒绝 random
   candidates 和 `control_seed`，所以 A1 在当前实现中不可达。
3. A2 尚未实现：当前 Gate-1 launcher/backend 仍固定 physical GPU1/CVD=1，与已批准的
   Slurm-assigned single-device、unmodified visibility、logical `cuda:0` 合同冲突。
4. 当前 Stage-C primitive 要求一次 top-level model forward 和顶层 differentiable Tensor，且成功
   路径禁止任何 registered buffer 变化；真实 `ActionFormer.forward_train` 返回 loss dictionary，
   `AnchorFreeHead.losses` 在 train mode 推进 `loss_normalizer`。因此当前 primitive 与 A3/A4 及真实
   detector 不兼容。
5. 正式 Stage-C、matched-dense 与 Gate-4 workflows 缺失，不能创建 I/R 或启动 formal chain。
6. registration source vector 与所有 ChronoTransport tests 的分类尚未闭合；正式 source set 必须
   明确、无默认分类并覆盖自身的 runner/launcher/validator/hardening tests。
7. lexical-path、parent-symlink、TOCTOU 和 pathname reload 风险原则成立；正式证据必须由不可替换
   的已打开对象、inode 和 exact bytes 绑定。
8. Gate-4 pure adjudicator 的 bounded `APPROVE_FROZEN_SLICE` 只适用于 test-only 统计切片；它不批准
   official producer、formal workflow 或 Gate result。
9. 当前没有 r2 science Gate、训练或论文数字；路线尚未因本轮审计发生 science FAIL。

## Accepted as requirements, not as verbatim patches

### A1 and registration

更新 `APPROVED_SPEC_COMMIT`、spec hash 和 random-control lock 是必要条件，但 reviewer 的 bounded
diff 不是完整修复。还必须同步 registration factory/generator、candidate factory config、action
hash/recomputation identity、claim consumer、source classification、all formal entrypoints 和对应
negative tests。任何一个旧 consumer 继续读取旧 alias 都应 fail closed。

### A2 Slurm identity

删除 physical-GPU1 pin 是必要条件，但 reviewer 自己也把 artifact identity 标为
`PATCH_BLOCKED_BY_MISSING_FACT`。正式实现必须重新观察 OS/Slurm/torch/NVML 可见状态，不能把
caller-provided `CHRONOTRANSPORT_OBSERVED_*` 或 JSON 当作证据。registration 冻结 required GPU
model 与 driver/CUDA/PyTorch/cuDNN/precision policy；每个 run artifact 再绑定实际 job/step、
scheduler-visible identity 与 GPU UUID。run-specific Job/Step 不应被预先伪造进 R。

reviewer 的 launcher patch 覆盖不完整：当前 full-stack profiler CLI、Stage-B CLI、Gate-1 profile
backend、Gates-2/3 claim key 以及旧 implementation plan 仍含 GPU1 语义。必须做 repository-wide
classification 后迁移，不能只修一份 shell script。

### Real ActionFormer and Stage C

同一 logits/targets 产生 exact shape `[2]` 的 per-window task-loss vector、保持现有 batch aggregate
objective、一次 dense reference、一次 differentiable counterfactual 和一次 risk forward，是 A3/A4
的强制要求。但 reviewer 展示的 `ActionFormerPerWindowTrainOutput` 只是需求草图：规范没有冻结
exact dataclass 名、`detector_features` 字段或 loss dictionary 只能等于
`{cls_loss, reg_loss, cost}`。当前 ActionFormer 还可能合并 selector/reader auxiliary losses，必须从
registered config 和实际 head reduction 推导、冻结精确 namespace 与等式，不能删掉合法 loss 或
发明 feature boundary。

reviewer 使用 `clamp/max(counterfactual-dense, 0)` 并非臆造；它与规范第 13.4 节一致。保留意见只
针对示例接口和未冻结的字段集合，不针对该 regret 公式。

### Source classification

显式 classification manifest 的原则完全接受，但 reviewer 只枚举到 19 个 matching tests，并把
剩余文件标为 missing fact。当前 clean checkout 实际有 21 个；额外两个是
`tests/test_chronotransport_opentad_replay.py` 与
`tests/test_chronotransport_stage_a_smoke.py`。不能仅因文件名就把任何文件自动标为 REQUIRED；全部
21 个必须逐一裁为 `formal_r2_source`、`legacy` 或 `nonformal`，新出现的未分类文件必须 fail
closed。classification manifest 自身也必须进入 immutable source vector。

### Filesystem integrity

component-wise `lstat`/`O_NOFOLLOW`、same-fd hash/deserialization 与 inode-bound publication 是必要
方向，但仍不足以单独证明 Python 实际执行的是同一组已哈希 bytes。最终设计还必须统一 clean
detached worktree、module import resolution、loaded module origins/bytes 和 formal entrypoint，使
验证对象与执行对象一致；否则 descriptor-safe data load 仍可能与 import-time pathname race 并存。

### Permanent kill semantics

接受 Gate 1/2/3/4 的冻结科学失败，以及事后改 protocol、扩 seed/comparator、泄漏 GT/teacher、
隐藏 overhead 等不可接受行为。对 reviewer permanent-kill list 的第 14/15 项作窄化：无合法 I/R
启动作业或 provenance 不闭环，首先使该 run 成为 `INVALID_IMPLEMENTATION`，立即停止、隔离、修复、
重新审计并重跑；它不自动证明科学假设失败。只有故意/事后改变 frozen protocol、不可恢复污染，
或正式 science Gate FAIL，才触发路线永久冻结。该区分与规范多处“implementation violation 不是
science FAIL”一致。

## Additional current surfaces the bounded patch did not cover

- `tools/bata/profile_chronotransport_r2_full_stack.py:65-66` 仍要求 CVD=1/physical GPU1。
- `tools/bata/train_chronotransport_r2_stage_b.py:273-274` 仍要求 CVD=1/physical GPU1。
- `tools/bata/chronotransport_r2_opentad_profile_backend.py:348-349` 仍要求 CVD=1/physical GPU1。
- `opentad/models/chronotransport/gates23.py:1691` 与对应 test 仍使用
  `latency_gpu1_fixed_stack` claim key。
- `docs/superpowers/plans/2026-07-12-chronotransport-ct-p3r-3s-r2-implementation.md` 仍绑定旧
  `e4422f5`/旧 hash、physical GPU1 与 `_gpu1.sh` workflow；该计划已经 stale，禁止原样执行。
- formal Stage-C、matched-dense、Gate-4 的五个预期 workflow/launcher paths 当前均不存在。
- registration 在 repository context 和多个 external path 上先 `.resolve()`；leaf-only symlink check
  与后续 pathname read 不能闭合 parent alias/TOCTOU 风险。
- formal Stage-C 还必须单独计量并披露 transaction/audit overhead；不能只证明 heavy actions
  减少而忽略完整端到端开销。

## Adopted implementation order

0. 保留本原文、吸收记录和精确状态；继续禁止 I/R、PRECHECK、formal job。
1. 先替换 stale implementation plan；不得按旧 `e4422f5`/GPU1 计划继续。
2. 完成 A1 registration authority、factory/consumer 迁移和 21-test/source classification。
3. 完成 repository-wide A2 Slurm migration及环境/artifact schema，所有观测由 producer 现场重算。
4. 建立共享 lexical/source/runtime identity primitive，闭合验证 bytes 与实际执行 bytes。
5. 从 registered ActionFormer config/head 实现并验证 exact per-window loss API，保持 aggregate objective。
6. 重写 Stage-C A3/A4 transaction primitive及真实 batch-two integration tests。
7. 实现 Stage-C 4,200 successful updates、matched-dense、post-Stage-C Gate-3 与 Gate-4 official workflows。
8. 完整 exact-byte 独立审查通过后才可考虑创建 I；R 仍必须是 I 的单父、只新增 canonical
   registration artifact。随后才允许 PRECHECK 和 stop-chain。

## Status after absorption

- A1--A4 specification: `spec_approved`。
- implementation: `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。
- registration: `NOT_READY`。
- `experiment_running=false`；没有新 Gate/report/unlock/training/result。
- `empirically_supported=false`、`paper_ready=false`、`deploy=false`。
- 本轮只归档和吸收审计，没有修改生产实现，也没有启动实验。
