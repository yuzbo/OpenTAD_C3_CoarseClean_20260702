# ChronoTransport r2 GitHub Pro 严审 — 第二次：Stage C、Gate 4 与最终裁决

这是两次连续 Pro 讨论的第二份 prompt。调用者必须同时提供：

1. 可选但推荐的 `EXPECTED_REVIEW_SHA=<40-hex>`；
2. 第一次讨论的**完整原文输出**，包括 `PART1_AUDIT_PACKET`，不得只给摘要。

本轮独立复核同一 immutable SHA 的后半实现，并把第一次的 findings 与本轮结果合并为唯一整体
registration verdict。全程只读；不得修改仓库、创建 commit/PR、启动 CUDA/Slurm、训练、Gate、
profiling 或生成任何实验 artifact。

## 1. Input and same-snapshot gate

Repository：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

Branch：`codex/chronotransport-r2-implementation`

Anchors：implementation floor `6c3606cc5161d415909a42741b3bc402278bf332`；equivalent-certificate
snapshot `702c67b4e38e80d307722a275a00b47f89cbfbf8`；prior two-part-prompt snapshot / required direct
parent `049923bbbaf6f664e985b4fb1cc96a2c06cdc810`。

First validate the attached Part-1 output：

- it contains exactly one complete `PART1_AUDIT_PACKET` and a permitted Part-1 verdict；
- if caller supplied `EXPECTED_REVIEW_SHA`, packet `review_sha` equals it；otherwise packet `review_sha` is
  the mandatory expected identity for this session；
- `snapshot_pass=true` and spec SHA equals
  `E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`；
- it is not `GITHUB_SNAPSHOT_INCOMPLETE` and does not omit its coverage/finding ledgers.

If missing or inconsistent, output the compact diagnostic and stop：

```text
PART2_INPUT_INCOMPLETE
packet_review_sha: <40-hex>|UNAVAILABLE
caller_expected_sha: <40-hex>|NOT_SUPPLIED
first_failed_condition: <one concrete condition>
END_PART2_INPUT_DIAGNOSTIC
```

Then independently fresh-resolve the branch once. It must equal the packet SHA and, if supplied, caller expected
SHA. All subsequent reads must be SHA-pinned. Independently prove：

- strict compare from `6c3606c`, `702c67b` and `049923b` to `REVIEW_SHA` with `behind_by=0` and exact
  merge bases；
- `REVIEW_SHA^1=049923bbbaf6f664e985b4fb1cc96a2c06cdc810` and no `REVIEW_SHA^2`；
- the complete `6c3606c...REVIEW_SHA` path list contains only audit/research documents under
  `docs/methods/` or `research-wiki/` after the implementation floor；
- every Part-2 mandatory file is retrieved through `ref=REVIEW_SHA` or equivalent immutable endpoint.

Full Git Data object is preferred. If tree/timestamps are not exposed, label them
`UNAVAILABLE_NONBLOCKING_AFTER_EQUIVALENT_CERTIFICATE`; do not copy project-reported values. Any SHA,
parent, ancestry, path or frozen authority-content mismatch requires this diagnostic and stop；never return
only the bare marker：

```text
GITHUB_SNAPSHOT_INCOMPLETE
resolved_sha: <40-hex>|UNAVAILABLE
packet_review_sha: <40-hex>|UNAVAILABLE
caller_expected_sha: <40-hex>|NOT_SUPPLIED
snapshot_route_attempted: A|B|NONE
first_failed_condition: <one concrete condition>
failure_class: ACTUAL_MISMATCH|INTERFACE_UNAVAILABLE|CALLER_INPUT_INVALID
END_GITHUB_SNAPSHOT_DIAGNOSTIC
```

## 2. Carryover discipline and evidence classes

Part-1 output is external reviewer evidence, not a repository fact. Preserve every Part-1 P0/P1 as blocking in
the final synthesis unless this session re-reads the exact SHA-pinned lines and demonstrates a factual error;
record any reversal explicitly. `PART1_FOUNDATION_BLOCKED` automatically prevents overall APPROVE, but this
session must still complete its own audit so all implementation work is identified in one response.

Label facts as：

- `REPOSITORY_FACT`
- `REVIEWER_EXECUTED`
- `PROJECT_REPORTED_NOT_INDEPENDENTLY_VERIFIED`
- `REVIEWER_INFERENCE`
- `PART1_REVIEW_EVIDENCE`

Do not claim tests you did not run. Do not assume data, checkpoints, CUDA/Slurm state, hidden artifacts or local
attachments. Project-reported CPU tests are not Gate/scientific evidence.

## 3. Authority and mandatory reading

Verify the governing spec SHA-256 independently：

`docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md`

`E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`

Read the complete Stage-C/Gate-4/A3/A4 and stop-chain sections. Also read `AGENTS.md`, `RTK.md`, query pack,
anti-repetition, execution tracker, implementation-verification, source classification/inventory and
registration. Read the SHA-pinned Part-1 prompt itself and validate the attached output against its exact packet
contract. Recheck classification/registration coverage for every Part-2 file; do not redo Part-1 code line-by-line
unless needed for cross-boundary verification.

### Runtime, detector and Stage C

- `actions.py`, `cache.py`, `transport.py`, `scheduler.py`, `runtime.py`
- `losses.py`, `stage_c.py`, `formal_stage_c.py`, `post_stage_c.py`
- `profiler.py`, `environment.py`, `filesystem.py` at their Stage-C/Gate-4 call boundaries
- `opentad/models/detectors/actionformer.py`
- `opentad/models/detectors/single_stage.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- registered Stage-B/Stage-C ChronoTransport configs and their complete base-config inheritance

### Stage-C tools and entrypoints

- Stage-C factory, CT trainer, matched-dense trainer and validator
- post-Stage-C factory, Gate-3 replay runner and validator
- Stage-C and post-Stage-C Slurm launchers
- every helper/config imported by these entrypoints

### Gate 4 and official evaluation

- `gate4_population.py`, `gate4.py`, `formal_gate4.py`, `full_stack_profiler.py`
- official population builder, Gate-4 factory, producer/finalizer and validator
- Gate-4 Slurm launcher
- `opentad/evaluations/mAP.py`
- `opentad/cores/test_engine.py`
- official post-processing/NMS modules reached by the registered config

### Part-2 tests

Read all relevant production-boundary tests, at least：

- ChronoTransport core, ActionFormer per-window, Stage-C, runtime, pipeline and ViT-adapter integration tests
- Gate-4 and formal-Gate-4 tests
- registration/filesystem/environment tests that exercise Part-2 entrypoints
- repository-contract tests for these surfaces

At the end, union the Part-1 and Part-2 coverage ledgers. Any `REQUIRED` production/tool/script/test not covered
by either session is a blocking completeness failure. OUT_OF_SCOPE/test-only code must be unreachable from every
formal Part-2 entrypoint.

## 4. Stage C and matched-dense audit

Reconstruct the actual data/gradient/state flow through the registered ActionFormer/AnchorFreeHead：

1. Batch-two aggregate task loss and both per-window LD/LF/LR values come from the same forward, logits,
   targets, reduction and `loss_normalizer`; no dummy/second hidden forward, caller loss or detached surrogate.
2. The sole canonical runtime exposes the executed action after forward. `[1,T,L]→[T,L]` is allowed only after
   proving batch size exactly one. Requested/executed action bytes, hashes and measured cost agree.
3. Trainable CT, risk predictor and frozen VideoMAE/head/other parameters have exact object-identity ownership,
   optimizer groups, gradients and EMA. `.data` writes, equal-byte storage rebinds, views/aliases, module swaps,
   ordinary Tensor attributes and persistent/nonpersistent buffers are all audited.
4. Each seed reaches exactly 4200 successful updates. CT and matched dense share batch, augmentation, LR, EMA,
   attempt order and successful-update identity.
5. Overflow/retry restores model/optimizer/scaler (except registered backoff), scheduler, EMA, RNG/sampler,
   normalizer, exposure and ledger, then retries the same batch without advancing success count.
6. Checkpoint/resume from every legal interruption point is exact-equivalent to uninterrupted execution and
   rejects impossible partial publication/no-clobber states.
7. Measured cost is bound to exact registered profile/environment/producer/requested/executed bytes; a boolean,
   caller table, proxy cost or test fixture cannot satisfy formal provenance.
8. Every public runner/validator independently checks clean detached R, exact registration/source/import bytes,
   upstream unlocks and canonical output roots.

Trace shortest attacks that would create fake gradient ownership, fake shared batch, fake success count, fake
rollback or fake resume. Verify RED tests hit real production entrypoints rather than a self-consistent helper.

## 5. Post-Stage-C Gate 3

Verify that the replay uses only legal Stage-C checkpoints/artifacts and the frozen calibration protocol. The
report/unlock must be reconstructed independently and cannot be minted from caller payload, old pre-Stage-C
report, test-only schema or preaggregated rows. Any Stage-C mismatch/invalid artifact must lock Gate 4 without
being mislabeled a scientific Gate FAIL.

## 6. Gate 4 official population and inference integrity

### Population and decision leakage

- exact official full-video/sliding-window population, order, media/config/checkpoint bytes；
- no missing/duplicate/post-hoc filtered videos or fixture substitution；
- dense, calibration-frozen best static and learned CT identities remain frozen；
- learned decisions consume only deploy-visible whitelist signals—never evaluation GT, teacher, replay ledger,
  raw-prediction cache, future window or evaluation-selected comparator；
- overlapping windows aggregate through official full-video post-processing/NMS；
- metric/regret count each unique invocation once; timing repetitions only enter latency blocks.

Media must be hashed/read/decoded from the retained no-follow descriptor and rechecked immediately after decode.
Import/config/checkpoint identity and precheck/producer override locks must remain exact.

### Timing and real heavy computation

- total latency includes decode, preprocess, H2D, model, postprocess and full-video NMS；
- six D/C/S counterbalanced orders and warmup/measured separation match the spec；
- deferred CUDA events flush only after the registered outer synchronization; no hidden mid-forward sync,
  CPU conversion or changed schedule；
- heavy computation reflects executed heavy paths; TRANSPORT/HOLD cannot retain dense compute under a new label；
- cache hits, requested/executed actions, repair/fallback and invocation identities are auditable.

### Official AP and bootstrap

- AP mirror is exact to `opentad/evaluations/mAP.py`, including NumPy `argsort()[::-1]` quicksort equal-score
  behavior；no stable-sort substitute, approximate evaluator or preaggregated per-video AP；
- latency bootstrap resamples official videos then complete invocation blocks and recomputes arm p50 from raw
  totals—never sums bootstrapped stage percentiles；
- mAP bootstrap resamples official videos then three seeds; every sampled seed rebuilds from raw predictions/GT
  on the same video multiset; never merge predictions/NMS across seeds；
- detector-regret uses unique official invocations and the registered video/seed hierarchy；
- 5000 replicates, seed 20260711 and percentile one-sided 95% bounds are exact.

### Seven hard conditions

Check exact direction, unit and strictness：

1. latency-saving 95% LCB `>=0.15`；
2. mAP@0.7-drop 95% UCB `<=1.5`；
3. shortest-Q1 drop `<=1.5`；
4. per matched invocation
   `heavy_saving_i=dense_heavy_i-selected_heavy_i`,
   `overhead_i=total_CT_i-selected_heavy_i`,
   `margin_i=0.40*heavy_saving_i-overhead_i`; median heavy saving `>0`, median-margin 95% LCB `>0`；
5. `p50_CT-p50_static` 95% UCB `<=0`；
6. CT versus calibration-frozen static detector-regret improvement 95% lower `>0`；
7. every seed separately passes point latency saving, mAP@0.7 drop, shortest-Q1 drop, median margin and
   CT-static; pooled success cannot hide a reversing seed.

Do not replace per-invocation margin with arm-p50 arithmetic. Calibration-frozen static stays fixed; an
evaluation-selected diagnostic comparator is reselected per replicate but never replaces the hard comparator.

### Energy

Energy is only 10-Hz NVML power integrated over a long complete official-population timed block. Verify requested
and observed cadence, sample count, median/max gap and boundaries. Single-inference/sparse estimates are not
energy evidence and never replace primary conditions.

## 7. Slurm, stop-chain and no-result boundary

- GPU work requires a Slurm allocation/step; no login-node training；
- launchers never override scheduler visibility and process code uses only logical `cuda:0`；
- stop-chain is Gate1 → StageB → Gates2/3 → StageC+matched → post-Stage-C Gate3 → Gate4；
- any missing/FAIL/invalid upstream artifact atomically locks downstream execution；
- `INVALID_IMPLEMENTATION` is distinct from a scientific Gate FAIL；
- tests/synthetic artifacts are never labeled formal Gate results；
- even APPROVE only permits designating exact REVIEW_SHA as I. It does not create registration-only R,
  authorize PRECHECK/experiments or support a scientific claim.

## 8. Findings, tests and concrete fixes

For every invariant, map spec → production boundary → positive test → adversarial RED → residual risk. Check
mocks, shared/relabelable fixture schemas, self-consistent hashes and tests that remain green after deleting the
production check.

Use P0/P1/P2/P3. Each finding needs exact SHA/file/line/function, shortest attack trace, why current tests miss
it, minimal RED test, concrete compilable patch for P0/P1 and required rerun commands. Do not alter approved
thresholds/statistical units, introduce GT/teacher/replay leakage or invent a new method. Mark any true spec
change as requiring amendment rather than silently patching it.

## 9. Final synthesis and exact output

Output in this order：

1. Part-2 Input/Snapshot Certificate；
2. evidence-class table；
3. Part-1 blocker carryover/reversal table；
4. Part-2 line-coverage ledger and union coverage closure；
5. Stage-C/matched verdict；
6. post-Stage-C Gate-3 verdict；
7. Gate-4 population/timing/statistics/energy verdict；
8. Slurm/stop-chain verdict；
9. P0/P1 findings with patches；
10. P2/P3 findings；
11. combined test-adequacy matrix；
12. executable next plan with exact commands, artifacts and stop conditions；
13. residual unknowns requiring only a future valid R-bound CUDA/Slurm PRECHECK；
14. one exact overall verdict.

The overall verdict must be exactly one：

```text
APPROVE_IMPLEMENTATION_FOR_REGISTRATION
```

or

```text
REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
```

Any Part-1 or Part-2 P0/P1, missing mandatory file/coverage, source/registration gap, unreachable formal workflow,
non-equivalent ActionFormer loss/state transaction, or non-exact Gate-4 timing/metric/bootstrap requires REVISE.
No conditional approval. APPROVE means only that exact `REVIEW_SHA` may be designated implementation commit I;
it proves no scientific result and unlocks no experiment by itself.
