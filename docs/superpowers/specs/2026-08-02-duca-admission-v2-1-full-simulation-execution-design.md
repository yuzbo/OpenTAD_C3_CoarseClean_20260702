# DUCA Admission v2.1 Candidate-Free Full-Simulation Execution Design

Date: `2026-08-02`

Status: `EXECUTION_PROTOCOL_REVIEW_REQUIRED / FULL_SIMULATION_NOT_RELEASED`

Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`

Branch: `codex/duca-rime-20260727`

Audited HEAD: `77acd054965d4e5527c170cedf3073a3fe7cb04b`

## 1. Decision

The scientific core of `U-PRO-V21-FINAL-REPAIR-1` remains accepted. Stage A,
Stage B, the Stage-C registry/evaluators, and the Stage-D receipt/publication
libraries remain valid candidate-free protocol work. Production Admission v2.1
remains `NO_GO`.

The complete `52 x 500` simulation and `24 x 200` MC calibration must not be
released from the current HEAD. Repository and runtime audit found that the
statistical registry is frozen, but the distributed experiment itself is not:

1. no full-simulation runner, task manifest, reducer, resumable shard contract,
   or terminal receipt writer exists;
2. `run_simulation_outer` defaults every outer dataset to primary stream zero
   and diagnostic stream one unless a caller invents a mapping;
3. the five shift profiles deterministically regenerate the same stream and
   repeat the same jackknife, but the protocol does not say whether common
   random numbers and result reuse are required, allowed, or forbidden;
4. the 24-by-200 operational-stream producer and the two-2M-to-one-4M reference
   producer are absent, including their exact stream namespace;
5. the frozen reference runtime is Python `3.11.7` plus NumPy `1.23.5`, while
   the currently registered OpenTAD environment is Python `3.10.20` plus NumPy
   `1.23.5`;
6. N16R4 has `MaxArraySize=1001`, account `MaxSubmitJobs=16`, a GPU-only
   submission partition, and a default eight CPUs per requested GPU. A naive
   one-task-per-outer DAG cannot be submitted.

These are execution-protocol questions, not model-performance results. Choosing
answers locally would create a new stochastic/numerical protocol after the
acceptance gates were frozen. The correct action is one narrow execution review,
not a new architecture or innovation discussion.

## 2. Measured engineering feasibility

A local, non-authoritative microbenchmark on the current scalar implementation
used synthetic candidate-free cells only. It measured approximately:

```text
100 multiplier replicates                     0.0868 s
10,000 multiplier replicates                  8.85 s
10,000-replicate maxT finalization             0.063 s
10 delete-1000 jackknife recomputations         0.559 s
```

Linear extrapolation gives roughly 89 seconds to generate one 100k stream and
roughly 63 seconds for its 100 delete-1000 recomputations, before Python object,
filesystem, scheduler, extension-to-200k, and contention costs. The current
outer executor does this for both the primary and diagnostic streams and repeats
the work for five shift profiles. This is an `ENGINEERING_STATUS` capacity
estimate only; it is not a statistical result and cannot enter the paper.

The full protocol may still be feasible after deterministic sharding and
semantics-preserving reuse, but feasibility must be demonstrated in the exact
Python/NumPy runtime by a registered nonzero-scenario benchmark before releasing
all tasks.

## 3. Alternatives considered

### A. Deploy the current scalar API verbatim

Rejected. It would silently choose stream IDs in the launcher, regenerate
identical streams five times, offer no terminal task identity, and cannot fit a
one-task-per-outer DAG under the account limit. A successful process exit would
not prove completion of the frozen registry.

### B. Vectorize or approximate the kernel immediately

Rejected before review. NumPy matrix products, pairwise reductions, partition
selection, or GPU kernels may be much faster, but they do not necessarily match
the current `math.fsum`, Python ordering, type-1 quantile, and delete-one-batch
semantics bit for bit. An optimization cannot be introduced as an unrecorded
implementation detail.

### C. Freeze a content-addressed distributed runner and an equivalence contract

Recommended. The review must first freeze the stream namespace, common-random-
number policy, reference construction, and acceptable numerical equivalence.
Then implement deterministic shards, immutable task artifacts, strict reducers,
and an optional optimized kernel that must pass the frozen equivalence suite.

## 4. Proposed execution contract for adjudication

The following is a recommendation, not yet an accepted protocol.

### 4.1 Stream namespace

Use disjoint unsigned-32-bit ranges and register every mapping in the task
manifest:

```text
outer primary:
  2 * (scenario_index * 500 + outer_index)
outer diagnostic:
  outer_primary + 1

MC operational:
  1_000_000 + calibration_scenario_index * 200 + stream_index

MC reference half 0:
  2_000_000 + 2 * calibration_scenario_index
MC reference half 1:
  reference_half_0 + 1
```

The manifest validator must prove global uniqueness, exact task counts, bounds,
and stable mapping from task identity to stream ID. No CLI default may select a
production stream.

### 4.2 Shift-profile common random numbers

Recommended policy: all five shift profiles for one outer dataset use the same
primary and diagnostic multiplier draws. The profiles differ only by the frozen
added truth vector. This is a deliberate common-random-number design, not five
independent MC experiments.

Replicate generation may be cached once per outer. Reusing maxT scales,
criticals, or jackknife half-widths requires a proof and golden equivalence
tests because profile-specific observed values and binary certificates still
need recomputation. Until that proof passes, only the generated replicate
prefix/suffix may be cached; each profile's registered finalizer and certificate
must run independently.

### 4.3 MC calibration producer

For each of the 24 registered scenarios, generate the DGP once at
`outer_index=0` and hold its cells, contrast, residuals, exact-zero flags, and
registry hashes fixed.

For each of 200 operational streams:

1. generate the exact first 100k prefix;
2. compute the 26 registered estimates and their delete-1000 99% half-widths;
3. append, never regenerate, the 100k suffix;
4. recompute the same 26 estimates and half-widths at 200k;
5. seal one content-bound stream artifact.

For the reference, generate two independent 2M streams from the two registered
reference IDs. Seal the estimate from each half and the estimate obtained by
concatenating the halves in the fixed order `half0 || half1`. The reducer must
not synthesize the 4M estimate from two 2M summary statistics.

The review must decide whether the 4M calculation must use the current scalar
Python object representation, an out-of-core binary64 array, or an optimized
equivalent finalizer.

### 4.4 Task and artifact identity

Freeze one content-hashed task manifest containing:

- exact Git commit and tree;
- simulation registry artifact and semantic SHA-256;
- role manifest, incidence, and metric registry content SHA-256;
- exact Python/NumPy/Philox environment receipt;
- every outer, operational, and reference task ID and stream ID;
- output relative path for every task;
- reducer order and expected terminal receipt paths;
- `authorization_scope=NONE`, all authorization booleans false, and
  `official_final_sealed=true`.

Every task artifact binds the task-manifest content hash, exact task row, input
hashes, and result content hash. Publication is exclusive. Resume may only skip
an existing artifact after full independent validation; an invalid, truncated,
extra, duplicate, or mismatched artifact fails closed and is never overwritten.

Reducers require exact set equality, not just counts:

```text
observed task IDs == registered task IDs
```

They consume artifacts in the frozen canonical order and reject missing or
extra files before evaluating any gate.

### 4.5 Slurm topology

Do not submit 30,824 individual jobs or array elements. After a completed
exact-clean code gate, submit at most the currently available account capacity
as long-lived deterministic shards. A recommended first topology is:

```text
outer stage:       <= 15 one-GPU jobs, 8 CPU workers per job
MC operational:   <= 15 one-GPU jobs, 8 CPU workers per job
MC reference:     <= 15 one-GPU jobs, one reference scenario per process
reducer/audit:       1 one-GPU job after all registered artifacts validate
```

The exact shard count must account for unrelated active jobs because the account
limit is 16. Task assignment is `task_ordinal mod shard_count`; it cannot depend
on scheduler order. A preemption/requeue may fill only missing validated task
artifacts under the same manifest. A scientific or content-integrity failure
terminates the transaction and requires a new root.

The GPU request is a cluster submission constraint, not evidence that the
candidate-free scalar workload uses GPU compute. Receipts must report this
truthfully.

## 5. Mandatory implementation and release gates

1. Exact task-count/order/stream-uniqueness tests.
2. Scalar replay equality for every task class on small frozen fixtures.
3. Common-random-number and 100k-prefix equality tests.
4. Optimized-kernel versus scalar golden tests, including q+, q-, 12 scales,
   24 bounds, type-1 indices, jackknife half-widths, exact zero, tiny scale,
   heavy tail, and near-collinear scenarios.
5. Deterministic resume, invalid-existing-file, missing, extra, duplicate,
   truncated, hash-drift, and no-overwrite tests.
6. POSIX concurrent publication and injected-crash cleanup tests.
7. Exact Python 3.11.7/NumPy 1.23.5/Philox environment gate.
8. One exact-runtime S050 500-outer end-to-end rehearsal.
9. One registered nonzero-scenario benchmark that produces no acceptance claim
   but proves bounded memory and wall-time feasibility.
10. Independent read-only audit of the clean commit and task manifest.
11. Only then release the complete candidate-free registry.
12. Stop after terminal receipts. Real-video workers and every later phase still
    require a separate explicit authorization.

## 6. Narrow Pro discussion prompt

```text
你现在接管的是 DUCA Admission v2.1 的“候选无关全量仿真执行协议”窄裁决，
不是新一轮模型创新讨论，也不得启动真实视频、训练、holdout 或 official-final。

仓库地址：
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

必须检查分支：
codex/duca-rime-20260727

开始前必须从 GitHub 读取该分支最新 HEAD，报告 commit、tree、remote、clean
checkout 证据；不得沿用本 prompt 中的旧 HEAD。请完整阅读并交叉核对：

- research-wiki/query_pack.md
- research-wiki/anti_repetition.md
- docs/superpowers/plans/2026-08-01-duca-admission-v2-1-final-repair-implementation.md
- docs/superpowers/specs/2026-08-02-duca-admission-v2-1-full-simulation-execution-design.md
- tools/bata/duca_admission_v2_1_simulation.py
- tools/bata/duca_admission_v2_1_statistics.py
- tools/bata/duca_admission_v2_1_mc.py
- tools/bata/duca_admission_v2_1_runtime_receipt.py
- tools/bata/duca_safe_publication.py
- configs/protocols/duca_admission_v2_1_simulation_registry_v1.json
- configs/protocols/duca_admission_v2_1_control_registry_v1.json
- 所有 Admission v2.1 focused tests

不可改变的边界：

1. U-PRO-V21-FINAL-REPAIR-1 的 Stage A-D 科学核心已接受；本轮只裁决完整
   candidate-free simulation 的随机流、数值和分布式执行合同。
2. production Admission v2.1 仍为 NO_GO。
3. authorization_scope 必须为 NONE；phase1_v2_authorized=false；
   holdout_open_authorized=false；paper_claim_allowed=false；
   official_final_sealed=true。
4. 禁止真实视频 worker、scale-fit、calibration、holdout、Phase 1-4、learned
   H-RIME、full-200 refit、exact-211 official evaluation 和任何性能解释。
5. 当前没有 paper-admissible empirical result。

已经核验的仓库事实：当前只有 registry、单 outer executor、aggregator 和
validator；没有全量 runner/task manifest/reducer/terminal writer。当前
run_simulation_outer 默认 stream_id=0，并为五个 shift profiles 重复生成确定性
相同的 multiplier 流。24×200 operational streams 与两个 2M halves 组成 4M
reference 的 producer 未实现。冻结 runtime 要求 Python 3.11.7 + NumPy
1.23.5 + Philox golden。N16R4 MaxArraySize=1001，账户 MaxSubmitJobs=16，唯一
partition 要求 GPU，默认每 GPU 8 CPU。

请从 CVPR 第一作者、统计协议负责人和系统实现负责人三个视角，逐项作出
唯一裁决，并给出可直接合入仓库的核心代码：

A. 冻结全局无冲突的 uint32 stream-ID namespace。明确 52×500 outer primary/
diagnostic、24×200 operational、24×2 reference halves 的精确公式，并证明
唯一性、prefix 性和与任务身份的绑定。

B. 裁决五个 shift profiles 是否必须采用 common random numbers。若共享，明确
允许复用的最小对象（仅 replicate bytes，还是 scales/q/jackknife），证明不改变
profile-specific bounds、numeric-tail decision 和 MC certificate；若不共享，给出
新的独立流公式及统计理由。

C. 完整定义 MC calibration producer：outer_index=0 的数据是否固定；每个
operational stream 的 100k/200k estimate 与 delete-1000 half-width 如何形成；
两个独立 2M 流如何按固定顺序 concatenate 为真正的 4M reference；26 参数
q_plus/q_minus/12 lower/12 upper 如何展开和封存。

D. 裁决可接受的数值实现。当前标量实现使用 Python dict/list、math.fsum、完整
排序和重复 jackknife。若允许 NumPy/out-of-core/GPU 优化，请给出必须逐位一致、
ULP 有界一致或仅 gate-decision 一致中的唯一标准，误差预算、边界场景、golden
fixtures 和失败规则。不得用“数值应当差不多”代替合同。

E. 给出 content-addressed runner 的闭世界 schema、task enumeration、exclusive
atomic publication、resume/no-overwrite、missing/extra/duplicate/corrupt fail-closed
reducer、terminal receipts 和独立 audit。请提供核心 Python 代码，不只给伪代码。

F. 在 MaxSubmitJobs=16、MaxArraySize=1001、GPU-only partition、8 CPU/GPU 下给出
可执行 Slurm DAG：每阶段 shard 数、每 shard 内 worker 数、任务分配、依赖、重排/
抢占恢复、资源与预计 walltime 的保守上界。不得提交一 task 一 job 的不可运行方案。

G. 给出红灯先行测试：stream collision、five-profile semantics、100k prefix、
4M concatenation、scalar/optimized equivalence、resume、并发 publication、截断/
hash drift、exact set reducer、授权不变量、S050 全 500，以及一个非零场景的
exact-runtime 可行性 benchmark。

最终必须输出：

1. GO / CONDITIONAL GO / NO-GO；
2. 逐项 ACCEPT / REJECT / MODIFY 表；
3. 唯一推荐算法和替代方案拒绝理由；
4. 文件级修改清单与核心可运行代码；
5. machine-readable execution decision manifest；
6. full simulation 释放前的硬门禁；
7. 明确声明本裁决不授权任何真实视频或模型实验。

如果任一随机流、4M reference、数值等价或恢复语义仍未闭合，必须 NO-GO，
不得建议“先跑起来再看结果”。
```

## 7. Current evidence statement

```text
full_simulation_runner = not_implemented
full_simulation_execution_protocol = under_narrow_review
complete_52x500 = not_run
complete_24x200_mc_calibration = not_run
production_admission_v2_1 = NO_GO
paper_admissible_empirical_result = NONE
```

No paper-admissible empirical conclusion is available yet.
