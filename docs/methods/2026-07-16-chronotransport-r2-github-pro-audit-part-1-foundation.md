# ChronoTransport r2 GitHub Pro 严审 — 第一次：快照、注册与 Gates 1–3

这是两次连续 Pro 讨论的第一份 prompt。只审本文件规定的前半证据链；不要在本轮审 Stage C/Gate 4
细节，也不要给整体 implementation verdict。你的完整输出将原样附给第二次讨论。

调用者可在消息中给出 `EXPECTED_REVIEW_SHA=<40-hex>`（推荐）。若未提供，reviewer 必须只
fresh-resolve 分支一次，将该 40-hex 立即冻结为 `REVIEW_SHA`，并在证书与 packet 中标明
`caller_expected_sha=NOT_SUPPLIED`。全程只读：不得修改仓库、创建 commit/PR、启动 CUDA/Slurm、
训练、Gate、profiling 或产生可被误认作实验结果的 artifact。

## 1. 本轮目标与允许的结论

以零信任、fail-closed 方式判断以下前半链条是否实现完整：

1. GitHub immutable snapshot 与规范字节；
2. A1/A2、registration、source/import/filesystem/environment identity；
3. Gate 1 的 23 schedules、200 invocations、真实成本和 oracle headroom；
4. Stage B 的 140 successful updates；
5. Gates 2/3 的独立 split、统计、calibration 与 unlock；
6. 上述表面的 adversarial tests 与 formal reachability。

本轮只能以以下一行之一结束：

```text
PART1_FOUNDATION_PASS
```

或

```text
PART1_FOUNDATION_BLOCKED
```

它们都不是 `APPROVE_IMPLEMENTATION_FOR_REGISTRATION`。任何整体 verdict 留给第二次讨论。

## 2. Snapshot certificate

Repository：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

Branch：`codex/chronotransport-r2-implementation`

Anchors：

- approved spec：`537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`
- previous reviewed spec snapshot：`1b6366d0acb712e8096c2cceb0f05e66b16d30d4`
- implementation floor：`6c3606cc5161d415909a42741b3bc402278bf332`
- metadata-blocked snapshot：`92a18bec2f5f247446083a8eb50fe889f367c23e`
- equivalent-certificate snapshot：`702c67b4e38e80d307722a275a00b47f89cbfbf8`
- prior two-part-prompt snapshot / required direct parent：
  `049923bbbaf6f664e985b4fb1cc96a2c06cdc810`

Fresh-resolve branch HEAD once. If caller supplied `EXPECTED_REVIEW_SHA`, it must match; otherwise the resolved
SHA becomes the frozen review identity. It must be a strict descendant of every anchor. Thereafter every
compare/file/raw read must explicitly use that SHA; never return to the moving branch.

Preferred Route A：obtain full Git Data commit object and report SHA, message, author/committer timestamps,
complete parents and tree SHA, plus compares to all anchors.

Route B is allowed only if the reviewer explicitly proves its GitHub interface does not expose the full object.
Then all of the following are mandatory:

- compare every anchor to `REVIEW_SHA`: `status=ahead`, `behind_by=0`, `ahead_by>0`, exact merge base；
- prove `REVIEW_SHA^1=049923bbbaf6f664e985b4fb1cc96a2c06cdc810` and `REVIEW_SHA^2` absent；
- enumerate the complete `6c3606c...REVIEW_SHA` changed-path list；
- every post-floor path must be an audit/research document under `docs/methods/` or `research-wiki/`；
- bind every later mandatory-file read to `ref=REVIEW_SHA` or an equivalent SHA-pinned endpoint. Full file
  reading belongs to §§3–8 and must not be completed before declaring the snapshot route；
- unavailable tree/timestamps must be labeled
  `UNAVAILABLE_NONBLOCKING_AFTER_EQUIVALENT_CERTIFICATE`, never copied from project prose.

An actual mismatch cannot use Route B. Missing ancestry, non-doc post-floor path, moving-ref read, or failure to
obtain the frozen prompt/spec authority bytes requires the compact diagnostic below and immediate stop. Do not
return a bare marker without the diagnostic fields：

```text
GITHUB_SNAPSHOT_INCOMPLETE
resolved_sha: <40-hex>|UNAVAILABLE
caller_expected_sha: <40-hex>|NOT_SUPPLIED
snapshot_route_attempted: A|B|NONE
verified_conditions: [short item, ...]
first_failed_condition: <one concrete condition>
failure_class: ACTUAL_MISMATCH|INTERFACE_UNAVAILABLE|CALLER_INPUT_INVALID
END_GITHUB_SNAPSHOT_DIAGNOSTIC
```

If Route A or B passes, continue; unavailable tree/timestamps alone must not stop the audit.

## 3. Authority and evidence discipline

Verify these exact SHA-256 values from SHA-pinned contents：

- governing spec
  `docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md`：
  `E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`
- prior Pro review
  `research-wiki/sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-verbatim.txt`：
  `C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`
- prior snapshot-only response
  `research-wiki/sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be.md`：
  `990E84F1D09116257D684090163BACB3F579ACA7290BADCB4D9FC6CFDA151FD1`
- its absorption
  `research-wiki/sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be-absorption.md`：
  `EB7A5767C7274C5F22F3625FC993853C44D39E0F20B9E12C71A4C9222EE078B1`
- source classification：
  `2713C84DD63B596C6EC522820D625A4271654E73F437F5A736E85A06BA108D2F`

The prior review approved A1–A4 as specification but returned
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` for the old code. Do not transfer either verdict automatically.

Label every claim as one of：

- `REPOSITORY_FACT`
- `REVIEWER_EXECUTED`
- `PROJECT_REPORTED_NOT_INDEPENDENTLY_VERIFIED`
- `REVIEWER_INFERENCE`

Do not claim tests you did not run. Do not assume datasets, checkpoints, Slurm, CUDA, hidden artifacts or local
attachments exist. Unit tests and project-reported `441 passed, 1 skipped` are implementation claims, not
scientific evidence.

## 4. Mandatory reading for Part 1

Read completely, not by filename inference：

### Rules, memory, specification

- `AGENTS.md`, `RTK.md`
- governing spec, especially §§1–12 and A1/A2
- prior Pro review and its absorption
- prior `92a18be` snapshot response and absorption
- `research-wiki/query_pack.md`, `research-wiki/anti_repetition.md`
- execution tracker and implementation-verification record

### Registration and model-side foundation

- `__init__.py`, `source_classification.json`, `source_inventory.py`, `registration.py`
- `filesystem.py`, `environment.py`, `protocol.py`
- `actions.py`, `controls.py`, `cost_lookup.py`, `replay.py`
- `adjudication.py`, `gate1_unlock.py`, `full_stack_profiler.py`
- `formal_stage_b.py`, `risk.py`, `training.py`, `gates23.py`
- `runtime.py` only for signal whitelist/leakage and formal reachability; its Stage-C mechanics belong to Part 2

### Formal tools and entrypoints

- manifest builder, registration CLI, precheck validator
- r2 profile factory, OpenTAD profile backend, full-stack profiler CLI
- Gate-1 replay factory, Gate-1 runner and Gate-1 Slurm launcher
- Stage-B factory/trainer and Gates-2/3 factory/runner
- every imported helper/config/base config actually reached by those entrypoints

### Part-1 tests

Read the production boundary and all relevant positive/negative tests, at least：

- registration, filesystem, environment, protocol and manifest-protocol tests
- actions/cache, profile-backend, Gate-1 cost/hardening and adjudication tests
- Stage-B, risk and Gates-2/3 tests
- repository-contract tests that cover these surfaces

Enumerate all 65 classified paths and confirm 49 `REQUIRED` / 13 `OUT_OF_SCOPE` / 3
`TEST_ONLY_NON_FORMAL`. Part 1 need not line-review Part-2-owned files, but must prove they are classified and
cannot be reached through a Part-1 formal entrypoint under an out-of-scope/test-only identity.

## 5. A1/A2, registration, source and filesystem audit

Prove or refute with exact file/line/function evidence：

1. A1 fixes unsuffixed `random_p2/p4/p8` to integer seed 3407. Replacing invocation order regenerates every
   per-window requested-action hash and order digest from exact window IDs; no stale template hash survives.
2. Registration generation reads no result/profile/replay/evaluation data. Future I must be clean; R must be
   I's unique single-parent child; `I..R` may add exactly one canonical regular-blob registration artifact.
3. In-memory registration equals the exact R:path bytes. Required-source set equals explicit classification;
   new/unclassified paths fail closed, while OUT_OF_SCOPE/test-only code cannot mint/rebuild formal evidence.
4. Every public formal mint/report/unlock validates clean detached R, source/import bytes, registration bytes,
   random locks and R-derived output roots at its own boundary—not only in a launcher.
5. All existing lexical path components use no-follow/lstat. Hash/read/load use the same retained descriptor;
   parent/leaf symlink, inode swap, import shadowing, concurrent no-clobber and interrupted partial publication
   are rejected.
6. Imported module origins/bytes are bound to registered sources; `PYTHONPATH`, editable installs and namespace
   aliases cannot execute unregistered code.
7. A2 registration freezes required model/software but not a future UUID. Producers re-observe current PID,
   raw Slurm allocation, CUDA visibility/runtime and full GPU UUID. Exactly one visible device and logical
   `cuda:0`; no launcher overwrites scheduler visibility or pins a physical index.
8. Precheck and producers share the same config-override lock and exact canonical bytes.

Trace adversarial cases rather than merely listing checks. Any formal API that accepts caller-built evidence,
relies on private Python names as authority, or validates only self-consistent hashes is blocking.

## 6. Gate 1 audit

Check exact protocol and execution ownership：

- exactly 23 registered schedules × 200 registered invocations in frozen order；
- all media/registry bytes validated before any candidate warmup/timing；
- warmup excluded; D/C/S counterbalancing and raw invocation observations retained；
- full-stack cost includes the registered components and uses real fixed repository backend, not caller rows,
  proxy cost, zero-count placeholders or test-only reconstruction；
- requested/executed action/cost, repair/fallback, registration/environment and replay identities are bound；
- dense/candidate regret comes from the same materialized batch/RNG inside repository-owned execution；
- evaluation-best static and strongest comparator are reselected inside each unique-window bootstrap replicate
  exactly as specified；
- Gate-1 FAIL/invalid/missing artifact atomically prevents every downstream phase.

Distinguish `INVALID_IMPLEMENTATION` from a scientific Gate FAIL.

## 7. Stage B and Gates 2/3 audit

### Stage B

- exactly 140 **successful** updates per registered seed, not attempts；
- overflow/retry does not advance optimizer, scheduler, EMA, normalizer, sampler/RNG, exposure or ledger；
- checkpoint, 140×16 exposure/rank evidence, predictor identity and completion marker publish atomically；
- three seeds preserve registered batch/augmentation/LR/EMA identities；
- predictor/inference consumes only deploy-visible whitelist signals—never validation/evaluation GT, teacher,
  raw-prediction cache, replay ledger or future-window information.

### Gates 2/3

- fit/calibration/evaluation windows and artifacts remain disjoint and exact；
- bootstrap outer unit is unique manifested window; seed is a global cluster draw, not redrawn per sampled row；
- formulas, strict inequalities, empty-selection behavior, 5000 replicates and seed 20260711 match the spec；
- Gate-3 simultaneous residual uses each window's candidate maximum before the registered quantile；
- coverage/ranking/regret do not claim selected-conditional or Gate-4 guarantees；
- reports/unlocks are produced from the fixed formal session and cannot be minted from caller JSON, simplified
  PASS payload, test-only schemas or preaggregated statistics.

Confirm formal CLI/launcher reachability. A missing production launcher or an r2 path that silently falls back
to a legacy GPU1/six-schedule/old-split runner is blocking.

## 8. Tests and findings contract

For every invariant, map：spec clause → production boundary → positive test → adversarial RED test → residual
risk. Check whether mocks bypass the real entrypoint, fixtures share relabelable schemas, or deleting the
production check would leave the test green.

Findings use P0/P1/P2/P3 and include exact SHA, file, line/function, shortest failure trace, missing/insufficient
test, minimal RED test, and a concrete compilable patch for every P0/P1. Keep patches minimal; do not rewrite
unaffected files or propose a new method/spec unless explicitly labeled as requiring amendment.

## 9. Mandatory output and handoff packet

Output in this order：

1. Snapshot Certificate and selected Route A/B；
2. Evidence Classification；
3. Part-1 Line-Coverage Ledger；
4. Previous-finding closure for Part-1 scope；
5. P0/P1 findings and minimal patches；
6. P2/P3 findings；
7. registration/source/filesystem/A1/A2 verdict table；
8. Gate-1 verdict table；
9. Stage-B/Gates-2/3 verdict table；
10. test-adequacy matrix；
11. the exact machine-readable-style block below；
12. final one-line Part-1 verdict.

```text
PART1_AUDIT_PACKET
review_sha: <40-hex>
caller_expected_sha: <40-hex>|NOT_SUPPLIED
snapshot_route: A|B
snapshot_pass: true|false
spec_sha256: <64-hex>
part1_verdict: PART1_FOUNDATION_PASS|PART1_FOUNDATION_BLOCKED
blocking_finding_ids: [P1-...]
nonblocking_finding_ids: [P2-..., P3-...]
classified_counts: {required: 49, out_of_scope: 13, test_only_non_formal: 3}
registration_closure: PASS|BLOCKED
a1_a2: PASS|BLOCKED
gate1: PASS|BLOCKED
stage_b: PASS|BLOCKED
gates23: PASS|BLOCKED
files_read_sha_pinned: [path, ...]
part2_owned_files_not_line_reviewed: [path, ...]
mandatory_carryover: [finding-or-constraint, ...]
END_PART1_AUDIT_PACKET
```

Do not omit discovered blockers to shorten the packet. `PART1_FOUNDATION_PASS` requires no P0/P1, complete
mandatory reading and executable formal reachability for this scope. End with exactly one permitted Part-1
verdict line; never emit an overall registration verdict in this discussion.
