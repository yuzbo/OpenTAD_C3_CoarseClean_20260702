---
type: implementation_inventory
status: active
updated: 2026-07-28
---

# OpenTAD Worktree Inventory

## 2026-07-28 SparseHead superseding ownership

用户已指定当前仓库为 SparseHead/PhysTime 唯一可写路线。
`E:\DeskTop\TAD\OpenTAD_SparseHeadClean_20260702` 保留在磁盘上作为只读历史证据，
封存 HEAD 为 `dce2c66d1053d53dfcc40b051399cd4c2ecde9ad`。它与当前仓库无共同
Git 祖先，不能作为可继续开发的第二主线，也不能做 whole-tree merge/cherry-pick。

本轮选择性吸收 irregular bridge/point generator/assignment audit，以及
`codex/phystime-performance-diagnosis-20260712@e05f6231` 的
raw/native geometry、physical-metric 与 SDPQ 代码和关键实验记录。旧仓未提交的
repair/retrain launch matrix 不迁入。精确清单与 claim 边界见
[`experiments/sparsehead-route-consolidation-20260728.md`](experiments/sparsehead-route-consolidation-20260728.md)。
本节只变更 SparseHead 所有权，不改变 ChronoTransport、DUCA 或 Spatial-Zoom 的独立路线边界。

## Why this inventory exists

The local `E:\DeskTop\TAD` directory contains many historical OpenTAD trees.
They are valuable implementation evidence, but their names do not establish
recency, correctness or route ownership. Before any DUCA method edit, consult
this page and `anti_repetition.md`; never copy a whole tree or edit an adjacent
route in place.

## 2026-07-21 superseding reconciliation

The 2026-07-20 statement that the nested Protected-E2E clone was the unique
DUCA construction surface is now historical. Three later isolated trees exist:
`OpenTAD_DUCA_UniCompanion_20260721`, `OpenTAD_DUCA_TwoStage_20260721`, and
`OpenTAD_DUCA_LocalResidual_20260721`. Their model files form a shared commit
lineage rather than three independent architectures.

The complete model-family and reuse decision is frozen in
[`duca_model_version_registry.md`](duca_model_version_registry.md). Its key
verdict supersedes any older wording on this page: the global exact-K/max-gap
selector already exists in the selected-axis lineage, while one-frame-per-cell
is diagnostic only. Future work must reuse the global implementation and may
merge only the repaired P0, curriculum, and protected-gradient contracts.

## Current count and risk

- Top-level disk directories matching `E:\DeskTop\TAD\OpenTAD*`: 41.
- Including the protected-physical and global-curriculum isolated clones under
  `.codex_tmp`, the audit covers 43 relevant OpenTAD Git trees in total.
- Trees currently registered by the primary repository: 20. This count
  includes the detached Codex inspection tree
  `C:\Users\skywalker\.codex\worktrees\c961\OpenTAD_C3_CoarseClean_20260702`;
  it is not an implementation target.
- Several historical directories are valid standalone worktrees but no longer
  appear in the primary `git worktree list`.
- The primary repository is on the Spatial-Zoom branch. The 20:50 refreshed
  snapshot reports 189 dirty or untracked files when untracked directories
  are expanded recursively; earlier values counted an older filesystem state
  or top-level porcelain
  entries. It is research memory/co-ordination state, not the DUCA
  implementation surface.
- `OpenTAD_DUCA_ProtectedE2E_20260720` is at `b3222af` with five local gate
  hardening changes. Those changes are backed up in
  `.codex_tmp/duca-protected-e2e-b3222af-local-gate-hardening.patch`; that
  worktree remains untouched.

## Historical protected-physical construction tree

The Protected-E2E physical implementation was isolated in:

```text
E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702\.codex_tmp\
OpenTAD_DUCA_ProtectedE2E_Final_20260720
```

- Branch: `codex/duca-physical-protected-e2e-20260720`
- Current exact head: `ee05f610133fc37f8f1ee67b7225bb38ae917cc5`
- Remote: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Current state: clean exact implementation with physical selector/config/gate
  assets. It remains a component source rather than the current final method.
- The physical exact-K hard/soft DAG draft passed its first independent
  mathematical gate on the remote OpenTAD PyTorch environment: `9 passed`.
  The suite checks exhaustive irregular-axis Viterbi/log-partition/slot
  marginals, gradients, deterministic ties, mixed `K_eff`, invalid axes and
  hard/soft graph-contract equality. The test imports the module directly to
  isolate an unrelated remote NumPy-2/old-OpenCV binary conflict.
- This is only `tested` evidence for the standalone selection graph. It is
  not P1 completion, does not prove Allocation-Ceiling oracle parity at
  production scale, and does not authorize P0-P3 or official-60.

No code may be copied from this clone into the dirty primary tree. Reuse must
name exact files and semantics in the model registry; whole-tree copying is
forbidden.

## Reusable reference assets

| Reference tree | Exact head | Reusable asset | Boundary |
| --- | --- | --- | --- |
| `OpenTAD_DUCA_AllocationCeiling_20260720` | `db11aee` | `PhysicalAxis`, exact-uniform physical cap, source/internal/sink hard DP, exhaustive solver tests | Training-side diagnostic only; no differentiable Gibbs path |
| `OpenTAD_DUCA_ProtectedE2E_20260720` | `b3222af` | Official-ASFormer source, gradient-scope capture/replay, real-model gate patterns, manifest/hash binding | Candidate-hole, local-slope bridge and selected-axis contract are superseded |
| `OpenTAD_PhysTime_DeployFix_20260710` | `2c9840c` | Physical-grid ActionFormer target/decode implementation and roundtrip tests | Existing strict flag is incomplete; historical short-action support and mAP degraded |
| `OpenTAD_TrueTimeJointSelector_Worktree_20260706` | `05baa48` | Official ActionFormer detector-loss-to-selector connectivity proof pattern | Legacy relaxed/selected-axis selector is not the final feasible family |
| `OpenTAD_DUCA_TransitionOnly_20260711` | `4ce69c8` | Mature training/evidence tooling and immutable CellCF/transition results | Completed CellCF route is diagnostic and must not be relabelled |
| Legacy `AxisRoundtrip`, `AuditTargetDecodeIOU`, `Bridge*`, `PointGenVerifier` trees | heads recorded below | Adversarial test patterns for axis conversion, hard-forward equality and generated-point checks | Most target the old sparse/selected-axis head; reuse tests conceptually, not route code |

## Historical physical-route audit verdict (superseded)

The following paragraph records the 2026-07-20 physical-route audit and is no
longer the current DUCA construction decision. The superseding decision is the
global V8 reconciliation at the top of this page and in
`duca_model_version_registry.md`; it forbids rebuilding the physical or
local-cell selector as the current mainline.

Two independent read-only audits reached the same conclusion:

- no tree already contains the complete frozen method contract;
- no tree combines a physical exact-K DAG, hard Viterbi and soft Gibbs on the
  same graph, the fixed probability coverage floor, explicit selector adapter
  and score head, strict native-time detector semantics, and the complete P0-P3
  gates;
- the old Protected-E2E implementation still uses selected-axis GT semantics,
  a candidate-hole/local-slope bridge and ambiguous physical metadata;
- therefore the final route must be assembled narrowly in the isolated clone,
  using Allocation-Ceiling only as the hard-solver oracle and reusing
  PhysTime/TrueTime assets only as test ideas.

This is an implementation inventory verdict, not empirical support. It does
not authorize official-60.

## Explicit no-edit boundaries

The following routes are independent and remain read-only during this DUCA
task:

- `OpenTAD_SparseHeadClean_20260702` and all sparse-head audit worktrees.
- `OpenTAD_SpatialZoom_S1_*` and the dirty primary Spatial-Zoom branch.
- `OpenTAD_ChronoTransport_R2_20260712`.
- GAS-VT, signed-utility, Stage23/Stage234 and detector-aware historical
  branches.

Do not edit their sparse heads, spatial crop modules, configs, launchers,
tests, wiki claims or experiment artifacts. Reuse only a narrowly identified
test or helper after a line-by-line semantic comparison.

## Complete disk-tree snapshot

Snapshot refreshed on 2026-07-21 during the full DUCA lineage reconciliation. Dirty
counts below expand untracked directories to individual files
(`git status --porcelain --untracked-files=all`) so the metric remains
unambiguous across future checkpoints.

The table covers the 41 top-level `E:\DeskTop\TAD\OpenTAD*` directories.
Nested historical copies under `temrefuse-tad` are intentionally excluded:
repository policy forbids treating them as current implementation candidates.
The registered detached Codex inspection tree is also excluded from this disk
table because it lives under `C:\Users\skywalker\.codex\worktrees`.

| Tree | HEAD | Branch | Dirty paths |
| --- | --- | --- | ---: |
| `OpenTAD_AuditTargetDecodeIOU_Worktree_20260706` | `350129a` | `codex/audit-target-decode-iou-20260706` | 0 |
| `OpenTAD_AxisRoundtripTests_Worktree_20260706` | `5f65268` | `codex/axis-roundtrip-tests-20260706` | 0 |
| `OpenTAD_BridgeDenseEquivalence_Worktree_20260706` | `46b73ee` | `codex/bridge-dense-equivalence-sanity-20260706` | 0 |
| `OpenTAD_BridgeFallbackContract_Worktree_20260706` | `80a0df5` | `codex/bridge-fallback-contract-20260706` | 0 |
| `OpenTAD_BridgeGeneratedVerifier_Worktree_20260706` | `2c9a0c8` | `codex/bridge-generated-verifier-20260706` | 0 |
| `OpenTAD_BridgeScaleContract_Worktree_20260706` | `41e7629` | `codex/bridge-scale-contract-20260706` | 0 |
| `OpenTAD_C3_CoarseClean_20260702` | `35204f5` | `codex/spatial-zoom-s1-formal-20260715` | 189 |
| `OpenTAD_ChronoTransport_R2_20260712` | `b1dc482` | `codex/chronotransport-r2-implementation` | 13 |
| `OpenTAD_DetectorAwareSelector_Worktree_20260706` | `c799c48` | `codex/detector-aware-selector-20260706` | 0 |
| `OpenTAD_DUCA_AllocationCeiling_20260720` | `db11aee` | `codex/duca-allocation-ceiling-20260720` | 0 |
| `OpenTAD_DUCA_LocalResidual_20260721` | `6c56e11` | `codex/duca-local-residual-20260721` | 0 |
| `OpenTAD_DUCA_ProtectedE2E_20260720` | `b3222af` | `codex/duca-protected-e2e-20260720` | 5 |
| `OpenTAD_DUCA_SignedUtility_Worktree_20260706` | `edaf589` | `codex/duca-signed-utility-20260706` | 0 |
| `OpenTAD_DUCA_Stage234_Worktree_20260706` | `b15c278` | `codex/duca-stage234-owner-20260706` | 0 |
| `OpenTAD_DUCA_Stage23Owner_Worktree_20260706` | `679f194` | `codex/duca-stage23-owner-20260706` | 0 |
| `OpenTAD_DUCA_Stage23Runners_Worktree_20260706` | `3ce6bae` | `codex/duca-stage23-runners-20260706` | 8 |
| `OpenTAD_DUCA_Stage3E2E_Worktree_20260706` | `36c92d4` | `codex/duca-stage3-e2e-20260706` | 0 |
| `OpenTAD_DUCA_TransitionOnly_20260711` | `4ce69c8` | `codex/duca-cellcf-evidence-20260717` | 0 |
| `OpenTAD_DUCA_TwoStage_20260721` | `6f2ed48` | `codex/duca-two-stage-curriculum-20260721` | 0 |
| `OpenTAD_DUCA_UniCompanion_20260721` | `7f9ad10` | `codex/duca-selected-axis-diagnostics-20260721` | 15 |
| `OpenTAD_EvalLeakageFailClosed_Worktree_20260706` | `9c3c26f` | `codex/eval-leakage-failclosed-20260706` | 0 |
| `OpenTAD_GASVT_CostAudit_20260710` | `a5e1774` | `codex/gas-vt-stage23-detector-aware-20260706` | 0 |
| `OpenTAD_GASVT_Worktree_20260706` | `696f77d` | `codex/phystime-tad-2` | 15 |
| `OpenTAD_NativeAbsrangeShortGate_Worktree_20260706` | `a4b7655` | `codex/native-absrange-expanded-shortgate-20260706` | 4 |
| `OpenTAD_OfficialDenseSelectedAxis_Worktree_20260706` | `d6c3f00` | `codex/official-dense-selected-axis-sanity-20260706` | 0 |
| `OpenTAD_OnlineTADClean_20260702` | `b974f3d` | `codex/online-tad-clean-20260702` | 47 |
| `OpenTAD_PhysTime_DeployFix_20260710` | `2c9840c` | `codex/phystime-performance-diagnosis-20260712` | 0 |
| `OpenTAD_PointGenVerifier_Worktree_20260706` | `3377e98` | `codex/pointgen-verifier-contract-20260706` | 0 |
| `OpenTAD_PostprocessAxis_Worktree_20260706` | `80a0df5` | `codex/postprocess-axis-contract-20260706` | 0 |
| `OpenTAD_PreflightAudit_Worktree_20260706` | `80a0df5` | `codex/preflight-audit-contract-20260706` | 0 |
| `OpenTAD_SelectedAxisControls_Worktree_20260706` | `bcaaf80` | `codex/selected-axis-random-uniform-controls-20260706` | 0 |
| `OpenTAD_SparseHeadClean_20260702` | `dce2c66` | `codex/sparse-head-clean-20260702` | 25 |
| `OpenTAD_SpatialZoom_S1_AuditFix_20260715` | `8ebe5f0` | `codex/spatial-zoom-s1-audit-fix-20260715` | 0 |
| `OpenTAD_SpatialZoom_S1_de81fbd_20260715` | `911448a` | detached | 0 |
| `OpenTAD_SpatialZoom_S1_MatrixReceipt_20260717` | `f65ce5c` | detached | 0 |
| `OpenTAD_Stage1_AuditGuard_Worktree_20260706` | `66dbf07` | `codex/sparse-stage1-audit-guard-20260706` | 2 |
| `OpenTAD_Stage23_Runners_Worktree_20260706` | `66dbf07` | `codex/sparse-stage23-runners-20260706` | 3 |
| `OpenTAD_StrictFailClosed_Worktree_20260706` | `06dce3d` | `codex/strict-fail-closed-20260706` | 18 |
| `OpenTAD_TrueTimeJointSelector_Worktree_20260706` | `05baa48` | `codex/truetime-joint-selector-20260706` | 0 |
| `OpenTAD_WARN578Fixes_Worktree_20260706` | `578fe0d` | `codex/warn578-fixes-20260706` | 5 |
| `OpenTAD_WARNReviewFixes_Worktree_20260706` | `2213388` | `codex/warn-review-fixes-20260706` | 18 |

## Required update cadence

Update this page at every one of these boundaries:

1. physical DAG draft retained or replaced;
2. P0 protocol/config freeze;
3. P1 mathematical and hard-forward gates;
4. P2 per-loss gradient ownership gate;
5. P3 stratified hard-swap alignment gate;
6. exact commit pushed;
7. remote gate or official experiment submission.

Every update must state the exact tree, branch, commit, dirty count, test
evidence and whether another route was touched.

## 2026-07-21 global-curriculum isolated clone

- Tree:
  `.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721`
- Type: isolated Git clone under the ignored coordination area; it is not an
  additional registered worktree and must not be counted as a new model.
- Branch: `codex/duca-global-curriculum-20260721`
- Exact clean commit: `63e25eb17e523d369f73434ed4d9b6446608861a`
- Base lineage: `9442b94`, before the `56c2683` local-cell derivation.
- GitHub:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`
- Remote clean snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_63e25eb_20260721`
- Verification: the same V8 model line passed remote full focused regression
  `158 passed, 3 skipped`; the EMA group-audit revision passed `21` affected
  Linux tests, `py_compile`, diff check, exact-HEAD and clean-tree checks.
- Deployment: the only active serial Job is `1178989`, run root
  `duca_global_63e25eb_serial_20260721_2120`. Jobs `1178911`, `1178927`,
  `1178933`, `1178947`, and `1178975` are immutable protocol/launcher/evidence-
  gate history and are not alternate models.
- Route isolation: only DUCA selector/config/gate/launcher/test files changed
  in the isolated clone. SparseHead, Spatial-Zoom, ChronoTransport and the
  dirty coordinating tree's model code were not edited.

The 21:20 refresh still counts 43 relevant trees and 12 DUCA-module trees.
`63e25eb` changes only the existing P0 evidence gate and its focused test;
model-code hashes and the V0-V8 lineage are unchanged. No new worktree,
selector, decoder, detector wrapper, SparseHead, SpatialZoom or ChronoTransport
file was created or modified.

## 2026-07-20 physical DAG retention checkpoint

- Tree:
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720`
- Branch: `codex/duca-physical-protected-e2e-20260720`
- Base commit: `b3222af0895e23eca83113977c1bcfad75258c9e`
- Dirty implementation entries: 2
- Remote disposable test copy:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_physical_dag_draft_20260720_02`
- Evidence: `py_compile` passed and focused remote pytest is
  `9 passed, 1 environment warning`.
- Decision: retain the physical DAG draft for selector integration, subject to
  production-scale oracle parity and full P1/P2/P3 gates.
- Route isolation: no SparseHead, Spatial-Zoom, ChronoTransport, spatial-crop
  or other historical route file was modified. The helper PowerShell script
  lives only under the primary repository's ignored `.codex_tmp` coordination
  area and writes only to the disposable remote copy.
