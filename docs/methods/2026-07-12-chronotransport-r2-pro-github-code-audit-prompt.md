# ChronoTransport r2 GitHub-only Pro Code Audit and Discussion Prompt

用途：直接提交给 GPT-5 Pro / GPT-5.5 Pro / Oracle Pro。审查者无法读取作者本地文件，必须
仅使用下面固定的 GitHub commit、GitHub 官方上游和本 prompt 中的事实边界。

以下分隔线后的正文可原样复制。

---

# Role

You are the final adversarial reviewer for a research-code appeal named ChronoTransport
`CT-P3R-3S-r2`. Act simultaneously as:

- a senior CVPR/ICCV/NeurIPS reviewer for temporal action detection;
- a PyTorch/OpenTAD/AdaTAD/VideoMAE implementation expert;
- a conditional-computation, cache/feature-transport, AMP and systems-profiling expert;
- a conformal prediction and clustered-bootstrap statistician;
- a research-integrity auditor who is allowed to reject the entire implementation.

Use maximum reasoning effort. Do not reward engineering effort or agree by default. Reconstruct what
the pinned GitHub code actually does, compare it against the complete approved written specification,
decide whether the existing independent audit is correct, identify every additional blocker, and
provide implementation-grade patch proposals.

# 0. Hard execution boundary

This task is `READ_ONLY_REVIEW_DISCUSSION_AND_PATCH_PROPOSAL_ONLY`.

You may:

- browse and quote the pinned GitHub fork and primary official upstream repositories;
- inspect Git history, files, tests, configs, launchers, and research-wiki pages visible on GitHub;
- reason mathematically about algorithms, statistical units, gradients, retry semantics, costs, and
  experimental gates;
- provide complete unified diffs, replacement functions/classes, tests, validators, and commands;
- state that the route should be frozen.

You must not:

- modify, commit, push, merge, tag, or open a PR;
- use SSH, remote servers, Slurm, GPUs, training, calibration, profiling, or detector evaluation;
- claim that proposed code has been executed;
- invent missing checkpoints, data manifests, GPU logs, registration artifacts, Gate reports, or
  experimental results;
- silently change the approved seeds, candidates, exposure formula, head, loss, quantile, budgets,
  thresholds, population, bootstrap units, or stop-chain;
- call ChronoTransport an Online TAD method;
- introduce C3/DUCA `p_action`, action-frame selection, or pre-backbone frame deletion.

Stop after: strict review, route discussion, complete patch proposal, verification plan, and next-step
plan. Do not perform any experiment or deployment.

# 1. Authoritative GitHub snapshot

Repository:

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

Review branch:

`codex/chronotransport-r2-implementation`

The code snapshot to audit is immutable commit:

`4b07020acb2611c3f085488d2f678f3be037f1be`

Pinned tree:

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4b07020acb2611c3f085488d2f678f3be037f1be`

Do not audit the repository default branch, a newer moving branch head, or an old ChronoTransport
commit by accident. Every local-code citation must use a permalink containing the full pinned SHA.
If the pinned tree is inaccessible, return `GITHUB_VISIBILITY_BLOCKED` and list exactly what could not
be opened; do not infer file contents from this prompt.

# 2. Required reading order

Read these GitHub files completely before giving a verdict:

1. Repository rules:
   - `AGENTS.md`
   - `RTK.md`
2. Current memory:
   - `research-wiki/query_pack.md`
   - `research-wiki/anti_repetition.md`
   - `research-wiki/ideas/chronotransport.md`
   - `research-wiki/experiments/chronotransport-formal-stage-b.md`
   - `research-wiki/experiments/chronotransport-r2-implementation-verification.md`
   - `research-wiki/source_registry.md`
3. Normative specification and implementation plan:
   - `docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md`
   - `docs/superpowers/plans/2026-07-12-chronotransport-ct-p3r-3s-r2-implementation.md`
4. Independent reviews:
   - `research-wiki/sources/2026-07-12-chronotransport-r2-spec-only-independent-agent-review.md`
   - `research-wiki/sources/2026-07-12-chronotransport-r2-independent-implementation-audit.md`
5. All implementation surfaces listed below and every additional file found by searching the pinned
   tree for `chronotransport`.

Normative spec identity:

- approval commit: `e4422f5`;
- exact committed/worktree SHA-256:
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`;
- the specification at pinned commit `4b07020...` must hash to the same value before it is treated as
  normative.

# 3. Evidence taxonomy and current status

Keep four evidence classes separate:

- `REPOSITORY_FACT`: directly visible in the pinned GitHub source.
- `PROJECT_REPORTED_TEST_FACT`: a command/result recorded in the wiki, but not independently rerun by
  you in this read-only review.
- `EXPERIMENT_FACT`: requires raw registered artifacts; none are supplied for the r2 appeal.
- `INFERENCE_OR_PROPOSAL`: your reasoning or proposed patch.

Current state that must not be inflated:

- Historical commit `92029ea` has a formal negative Stage-B/P3 record. It does not validate r2.
- The r2 specification is approved for implementation, not empirically supported.
- The project reports a remote combined regression of 110 passing tests in 84.58 seconds. Treat this
  only as `PROJECT_REPORTED_TEST_FACT` unless independently reproducible from permitted evidence.
- No clean implementation commit `I`, registration commit `R`, registered full-stack profile, formal
  r2 Gate 1 result, new three-seed Stage B, Stage C, or Gate 4 result exists.
- Claim flags `deploy` and `paper` must remain false even if all four gates later pass.

# 4. Non-negotiable method semantics

ChronoTransport is an offline, full-window TAD conditional-computation route. It does not select action
frames and does not delete frames before the backbone.

It preserves:

- 768 external input points and detector grid;
- 48 VideoMAE 16-frame chunks;
- 384 internal tubelet points (`tubelet_size=2`);
- dense patch embedding;
- the original full-row AdaTAD temporal adapter after each block;
- projection/head/NMS and official full-video population where required.

Only heavy VideoMAE attention/MLP work may be controlled at `chunk × layer_group` granularity. The
three actions are `RECOMPUTE`, `TRANSPORT`, and `HOLD`. Cache history is detached, but the current
RECOMPUTE/TRANSPORT row remains live through the adapter and downstream loss. Actual validity age is
47; only the transport embedding index is capped at 8. Any formal repair, fallback, action-hash change,
or missing exact cost invalidates that sample/run rather than silently redefining the candidate.

# 5. Files that must be audited

At minimum inspect:

- `opentad/models/chronotransport/actions.py`
- `opentad/models/chronotransport/cache.py`
- `opentad/models/chronotransport/controls.py`
- `opentad/models/chronotransport/protocol.py`
- `opentad/models/chronotransport/transport.py`
- `opentad/models/chronotransport/risk.py`
- `opentad/models/chronotransport/scheduler.py`
- `opentad/models/chronotransport/runtime.py`
- `opentad/models/chronotransport/replay.py`
- `opentad/models/chronotransport/formal_stage_b.py`
- `opentad/models/chronotransport/training.py`
- `opentad/models/chronotransport/stage_c.py`
- `opentad/models/chronotransport/adjudication.py`
- `opentad/models/chronotransport/profiler.py`
- `opentad/models/chronotransport/cost_lookup.py`
- `opentad/models/chronotransport/registration.py`
- `tools/bata/chronotransport_opentad_factory.py`
- `tools/bata/train_chronotransport_stage_b.py`
- `tools/bata/run_chronotransport_stage_b_formal.py`
- `tools/bata/profile_chronotransport_schedules.py`
- `tools/bata/register_chronotransport_r2.py`
- `tools/bata/run_chronotransport_r2_gate1.py`
- `configs/adatad/thumos/c3_chronotransport_r2_stage_b.py`
- `configs/adatad/thumos/c3_chronotransport_r2_stage_c.py`
- `scripts/run_chronotransport_r2_gate1_gpu1.sh`
- every `tests/test_chronotransport_r2_*.py` file;
- existing ChronoTransport integration/formal tests and repository-contract tests.

Record missing Stage-B/C/matched-dense/Gate-3/Gate-4/profiler/launcher surfaces explicitly. Do not
interpret a declarative config, primitive function, test fixture, or plan as an executable workflow.

# 6. Re-audit the seven existing blocking findings independently

The previous independent implementation audit returned
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` and reported:

1. Gate 3 and Gate 4 adjudicators and launchers are absent.
2. No executable r2 Stage B, Stage C, or matched-dense control exists; the old formal runner still uses
   six legacy schedules and the obsolete split protocol.
3. Transactional AMP overflow retry, backoff-preserving restore, same-batch retry and resume validation
   are absent.
4. The learned scheduler does not enforce registered `B*` and exact measured requested cost.
5. The profiler/cost key cannot produce provenance-complete 50-warmup/200-invocation full-stack
   `total_ms` evidence with six-order crossover.
6. Registration accepts a caller-authored identity object instead of deriving and deeply revalidating
   every source/spec/config/checkpoint/data/window/library/exposure/environment identity.
7. Gate-1 inputs and fixed statistical constants are caller-controlled rather than registration-bound;
   exact 30/30 populations and complete frozen library are not enforced.

For each finding return:

- `AGREE`, `PARTLY_AGREE`, or `DISAGREE`;
- exact pinned GitHub file/line evidence;
- violated normative spec section;
- false-positive/false-negative risk;
- minimal mandatory repair;
- exact regression test;
- whether the repair changes scientific specification or only implements it.

Do not merely repeat this list. Try to falsify every finding and search for additional P0/P1 defects.

# 7. Full specification-compliance audit

Build a section-by-section matrix for all 16 normative spec sections. Use only:

- `PASS_IMPLEMENTED_AND_TESTED`;
- `IMPLEMENTED_NOT_INDEPENDENTLY_TESTED`;
- `PARTIAL`;
- `MISSING`;
- `CONTRADICTS_SPEC`;
- `CANNOT_VERIFY`.

Audit at least the following contracts:

## Protocol and manifests

- NFC/UTF-8 exact-byte split/window digests;
- exactly 200 unique videos and shared 140/30/30 split independent of run seed;
- label-free one-window-per-video construction and edge padding;
- canonical manifest hashes and complete identity fields;
- Stage-B `candidate=(p+5*b+seed_offset)%16` exact tails/balance;
- Stage-C 8,400 exposures/seed and exactly 525/candidate.

## Candidate/control semantics

- exact 16 non-dense canonical order and dense safety candidate outside fit/calibration vectors;
- exact 48×3 matrices and library/action hashes;
- motion/random exact RECOMPUTE counts, deterministic ties/digests, non-finite invalidation;
- no evaluation oracle enters checkpoint, training, calibration or scheduler.

## Runtime and risk

- original dense block path parity and block order;
- heavy gather affects only attention/MLP;
- adapter output writes all rows;
- live current row vs detached historical state;
- actual age 47 vs embedding cap 8;
- requested/executed action and exact cost separation;
- fixed D=23 encoder, mean/max pooling and one window scalar;
- dense risk exact zero outside fit/calibration/ranking.

## Stage B, Gate 1, Gate 2, Gate 3

- exactly 140 successful FP32 updates, no AMP, no extra epochs;
- paired RNG/materialized-pixel replay and permutation regression;
- full 16-candidate fit replay for the rank-127 constant baseline;
- Gate-1 equal-cost HOLD library, all hard comparators and registered B*;
- bootstrap resamples unique windows and reselects evaluation-best/strongest comparator per replicate;
- Gate-2 matched masks and hierarchical window/seed vector preservation;
- Gate-3 rank-28 simultaneous marginal calibration, per-window Spearman, selected support/coverage,
  coverage-margin definitions, pinball improvement, and constant-baseline denominator;
- scheduler feasibility includes `requested_p50<=B*`, upper risk, finite/hash validity, canonical tie
  order, and dense fallback outside non-dense success.

## Stage C and Gate 4

- object-identity A/T/R ownership and alias uniqueness;
- exact loss-specific scaled `autograd.grad` algorithm;
- transactional overflow retry restoring every mutable state except scaler backoff;
- 4,200 successful updates and matched-dense common-A exposure;
- post-Stage-C recalibration and Gate-3 rerun;
- official full-video population, matched timing, six-order crossover;
- seed-level mAP bootstrap and approved official-video detector-regret hierarchical bootstrap;
- all Gate-4 metric, latency, overhead and Pareto hard conditions.

## Registration/provenance/stop chain

- clean implementation commit `I`, detached generation, sole registration commit `R`;
- generator cannot read result/profile/replay/evaluation paths;
- full deep hash derivation and validation, not placeholder acceptance;
- launcher loads registration, requires HEAD=R, clean tree, exact GPU1/Slurm step and allowed output
  root, then revalidates every content identity;
- Gate 1 → Gate 2 → Gate 3 → Stage C → Gate 4 hard stop chain;
- honest status/claim transitions and permanently false `deploy`/`paper`.

# 8. Tensor, gradient, state and cost reconstruction

Provide an implementation map that follows one 768-point input through:

`dataset/window → patch embedding → 48 chunks → each layer group/block → heavy action → cache →
full-row adapter → projection/head/loss → detector grid/NMS`.

For every action show:

- current tensor and cache source;
- anchor/latest/age mutation;
- detach/live-gradient boundary;
- requested vs executed identity;
- invalidation/fallback behavior;
- how the next block/chunk observes the result.

Then reconstruct:

- Stage-B dense/counterfactual RNG and loss flow;
- Stage-C A/T/R gradient ownership and scaler lifecycle;
- scheduler risk/cost feasibility and tie-breaking;
- registration → profile → Gate artifacts and claim-state flow.

Flag every path where an in-place write, alias, detach, repair, proxy cost, missing hash, or stateful
buffer can violate the written protocol.

# 9. Statistics red-team

Verify mathematically and in code:

- conformal ranks 28/30 and 127/140;
- simultaneous marginal vs selected empirical coverage distinction;
- all-candidate-covered and window-all-selected-covered definitions;
- support denominator and dense-fallback exclusions;
- per-window Spearman degeneracy handling;
- hierarchical bootstrap outer window and inner seed units;
- Gate-1 replicate-time re-selection;
- Gate-4 seed-level mAP vector and official-video regret bootstrap statistic;
- one-sided vs two-sided confidence bounds exactly where specified;
- undefined relative improvement when denominator `<=1e-12`;
- no candidate row, invocation, overlapping window, or seed is falsely treated as an independent
  experimental unit.

For every defect provide a minimal numerical counterexample.

# 10. Official upstream verification

Use only primary sources: official OpenTAD/AdaTAD/VideoMAE repositories, releases, papers, or official
documentation. Give permanent upstream commit URLs and file/line ranges. Compare:

- VideoMAE patch/tubelet embedding and transformer block order;
- AdaTAD temporal adapter insertion and temporal reshape;
- official OpenTAD `random_trunc`, short-vector edge padding, detector/head/loss/NMS behavior;
- activation checkpoint behavior;
- optimizer, AMP, EMA and scheduler semantics relevant to Stage B/C.

For each item label the pinned fork as `SAME`, `WRAPPED`, `STRUCTURALLY_MODIFIED`, `CUSTOM`,
`MISSING`, or `CANNOT_VERIFY`, and state the scientific consequence. Do not call this fork “official
unmodified AdaTAD”.

# 11. Required code findings format

List findings from P0 to P3. Each finding must contain:

- concise title;
- pinned permalink and line range;
- repository fact;
- violated spec clause;
- concrete failure trace or counterexample;
- scientific/operational impact;
- minimal patch boundary;
- regression test name and exact assertions.

Distinguish:

- a missing surface;
- a wrong implementation;
- a weak test;
- an unverified but plausible path;
- a specification ambiguity.

Do not recommend widening the model or tuning the frozen protocol to make it pass.

# 12. Patch-generation contract

After the review, provide implementation-grade unified diffs for every registration-blocking repair
that can be responsibly specified from GitHub evidence.

Patches must:

- have no `TODO`, placeholder, `pass`, omitted core logic or pseudocode;
- preserve C3/DUCA and all unrelated repository paths;
- remain under `opentad/models/chronotransport/`, focused tools/configs/scripts/tests and required wiki
  documents;
- use canonical serialization and schema versions;
- fail closed on missing/non-finite/mismatched identities;
- avoid caller-controlled formal constants after registration;
- use exact measured cost lookup, never linear action-count estimates for formal gates;
- keep result-derived data out of registration and inference;
- preserve formal stop-chain and claim flags;
- include focused tests before production changes in TDD order.

At minimum discuss and, where feasible, provide complete patches for:

1. registered-B* scheduler/exact-cost enforcement;
2. transactional Stage-C retry/snapshot/restore;
3. executable r2 Stage-B and Stage-C/matched-dense runners;
4. Gate-3 and Gate-4 pure adjudicators;
5. full-stack profiler and provenance-complete cost key;
6. derived immutable registration generator/deep validator/formal-launch verifier;
7. Gate-specific GPU1 launchers and upstream unlock validation.

If a complete patch depends on unavailable repository facts, output `PATCH_BLOCKED_BY_MISSING_FACT`,
name the exact missing fact, and provide the complete interface/test contract instead of inventing it.

# 13. Verification plan

For every proposed patch list:

- red test and expected failure;
- green test and exact command;
- existing regression suites affected;
- CPU vs CUDA requirement;
- what passing proves and does not prove.

The project requires behavior tests on the remote environment, but this review must not run them.
Commands may be proposed only. The minimum eventual matrix includes:

- `git diff --check`;
- `py_compile` for touched entrypoints;
- all r2 focused tests;
- existing ChronoTransport core/integration/formal/repository tests;
- `tests/test_c3_coarse_classifier_model_matrix.py`;
- `tests/test_c3_asformer_delta_ledger_full_train.py`;
- remote CUDA forced-dense FP32/AMP gradient parity;
- overflow/backoff/retry/resume determinism;
- launcher `PRECHECK_ONLY=1` before formal submission.

# 14. Route discussion and mandatory verdict

Compare exactly three routes:

- Route A: accept current commit for registration and run Gate 1.
- Route B: implement the missing spec-preserving surfaces, re-audit, then create I/R.
- Route C: reject and permanently freeze the bounded appeal now.

For each give scientific validity, false-positive risk, engineering cost, expected information gain and
kill criterion. Then choose exactly one final verdict:

- `APPROVE_IMPLEMENTATION_FOR_REGISTRATION`;
- `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`;
- `REJECT_AND_FREEZE`.

Do not answer “run experiments and see”. If choosing revision, enumerate the smallest mandatory patch
set and an order that reaches a second independent review without opening formal data early.

# 15. Mandatory output order

Return these sections in order:

1. Executive Verdict, at most 300 Chinese characters.
2. GitHub Visibility Certificate: pinned fork files, upstream sources, inaccessible items.
3. Evidence Table: repository facts, project-reported tests, experiment facts, inferences/proposals.
4. Independent-Audit Recheck: AGREE/PARTLY_AGREE/DISAGREE for all seven findings.
5. Additional P0/P1 Findings.
6. End-to-End Implementation Map.
7. Sixteen-Section Spec Compliance Matrix.
8. Statistical Red-Team Audit with numerical counterexamples.
9. Official Upstream Verification Matrix with permanent links.
10. Code Findings P0→P3.
11. Route A/B/C Discussion and one mandatory verdict.
12. Minimal Mandatory Patch Architecture.
13. Complete Unified Diffs / Complete Replacement Code.
14. TDD and Verification Matrix; clearly mark all commands `NOT_EXECUTED_BY_REVIEWER` unless actually
    executed within the read-only environment.
15. Registration Readiness Checklist.
16. Next-Step Plan with inputs, outputs, stop conditions and wiki state transitions.
17. Result-to-Claim Matrix.
18. Final Kill Criteria.

# 16. Final discipline

- Code is implementation fact; the spec is the target contract; tests are not science results.
- Do not weaken the historical negative result or treat r2 as a fresh unconstrained search.
- Do not treat 110 reported passing tests as proof that missing workflows exist.
- Do not allow placeholder hashes, arbitrary identity JSON, or result-aware registration.
- Do not let dense risk zero count as a budget-feasible non-dense success.
- Do not use repaired/executed proxy cost to validate a requested formal candidate.
- Do not treat candidate rows, invocations, overlapping windows, or seeds as independent videos.
- Do not transfer frozen-window conformal claims to official full-video Gate 4.
- Do not call precheck, smoke, unit tests or synthetic adjudication a passed Gate.
- Do not generate or recommend a registration commit until every mandatory surface exists and remote
  CPU/CUDA verification is recorded.
- It is acceptable—and required when justified—to reject and freeze ChronoTransport.

Be severe, precise, source-grounded and implementation-complete. The purpose is to prevent another
incomplete registration or invalid GPU experiment, not to preserve the route.
