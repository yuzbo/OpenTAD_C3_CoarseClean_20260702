---
type: experiment
node_id: exp:duca-protected-e2e-gates
idea: idea:duca-protected-e2e
status: tested_diagnostic_nonconforming
verdict: incomplete
updated: 2026-07-20
---

# DUCA Protected-E2E P0-P3 Gate Diagnostics

## Scope

This node records the post-`280631a` protected-E2E implementation and gate
attempt. It is not an official-60 result and does not contain mAP.

## Implementation chain

- `0477c55`: protected detector-gradient bridge and initial gates.
- `c2226dc`: exact gradient ownership and hard-forward evidence hardening.
- `e7aa881`: pretrained/checkpoint evidence binding.
- `b3222af0895e23eca83113977c1bcfad75258c9e`: Slurm repository-root fix.

The implementation used candidate-hole `G=2`, selected-axis detector
semantics, a local protected structured-transport surrogate, detector bridge
endpoint `0.25`, rho `0.05`, seed 0 and a planned 6000-update official-60
protocol.

## Verification

- Clean Linux focused tests at the earlier exact commits: `45 passed, 2
  skipped`.
- Slurm Job `1176948`: `FAILED/1:0` after 2:58.
- Main and rho exact full-model gates ran on real THUMOS data with real
  VideoMAE-S adapters, projection, neck, ActionFormerHead, FP16 and one-rank
  DDP.
- Main detector-only gradients reached detector parameters and selector
  scorer while ASFormer/action head remained zero.
- Rho detector-only gradients additionally reached only the final ASFormer
  encoder layer.
- Detector backbone input exactly matched hard gather; `K=384` and the
  candidate-hole interval contract were satisfied.
- P3 never evaluated hard-soft alignment. It fail-closed while parsing the
  old `transition_beta0` evidence because the gate required a
  `training_profile` field absent from the hash-bound manifest.

## Superseding audit

The Pro adjudication archived under SHA-256
`f91db53a83d79f56927b04d38b1e886d2e4260e4528e7882ddd49adbda97ccb0`
requires a physical exact-K DAG, Gibbs slot marginals, native physical
target/decode and a much larger stratified P3. The implemented local bridge,
selected-axis contract and small P3 population therefore do not conform even
if the manifest parser is repaired.

## Evidence boundary

This experiment supports only real-model gradient connectivity for the
superseded candidate. It does not support hard-decision utility alignment,
physical-coordinate correctness, mAP gain, cost saving, C3, C4 or paper
readiness. No official-60 arm is authorized.

## 2026-07-20 worktree isolation checkpoint

A complete local tree inventory is recorded in
`research-wiki/worktree_inventory.md`. The disk has 37 top-level `OpenTAD_*`
directories and 19 registered worktrees; the additional registered entry is a
detached Codex inspection tree outside `E:\DeskTop\TAD`, not an implementation
target. The dirty Spatial-Zoom primary tree, SparseHead trees, ChronoTransport
and all historical route trees are read-only for this task.

New work is isolated in
`.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` on branch
`codex/duca-physical-protected-e2e-20260720`, based on `b3222af`. It currently
contains one untested draft modification to
`opentad/models/duca/structured_selection.py`. This is inventory/implementation
work only: P0-P3 remain incomplete and no training is authorized.

Two independent read-only cross-tree audits found no hidden complete
implementation. The reusable pieces are distributed across Allocation-Ceiling,
Protected-E2E, PhysTime and TrueTime, while their old selected-axis, sparse-head
and local-bridge route code is explicitly ineligible for transplantation.

## 2026-07-20 physical exact-K mathematical gate

The isolated construction tree now has a shared physical exact-K graph
implementation for deterministic hard Viterbi and differentiable Gibbs
forward-backward slot marginals. The source and test are still uncommitted on
base `b3222af`; no historical route was edited.

The disposable remote copy
`opentad_duca_physical_dag_draft_20260720_02` passed `py_compile` and
`tests/test_duca_protected_e2e_selection.py` with `9 passed`. The tests compare
hard paths, log partitions and slot marginals against exhaustive enumeration
on an irregular physical axis, verify nonzero finite gradients, deterministic
lexicographic ties, `K_eff=min(K,valid_len)`, `-1` inactive slots, and
fail-closed invalid axes. A NumPy-2/old-OpenCV warning exists in the remote
environment; the focused test loads the standalone module by file so this
unrelated package-import conflict does not hide the dynamic-programming result.

Status is `tested` for the standalone mathematical component only. P0
protocol, selector integration, detector hard-forward/soft-backward identity,
per-loss gradient ownership, native-time detector semantics, P3 hard-swap
alignment and official-60 remain incomplete and unauthorized.

## 2026-07-20 protected selector integration draft

The isolated construction tree now also contains an uncommitted explicit
protected selector draft:

- `DucaProtectedTransitionScorer` exposes a separate 197-to-64
  `selector_adapter` and one-dimensional `selector_score_head` while leaving
  the historical scorer unchanged.
- `coverage_floor_distribution` freezes the 0.10 uniform floor and 0.70 score
  temperature.
- `DucaProtectedE2EFrameSelector` implements the four registered arms:
  `exact_uniform`, `transition_no_bridge`, `protected_e2e`, and
  `protected_e2e_rho001`.
- The learned arms use official-ASFormer action logits plus relational hidden
  changes, the shared physical exact-K graph, exact hard gather forward and a
  soft Gibbs backward path. The main arm detaches action logits and ASFormer
  hidden from detector loss; the rho arm admits only a 0.01 restricted hidden
  route.
- Dense GT is returned unchanged. Metadata declares native dense/physical
  detector coordinates and no selected-axis inverse remap.

Local `py_compile` passed. Local pytest cannot load the Windows PyTorch
`c10.dll`, which is an existing host-environment failure. Focused tests were
authored for exact-uniform metadata, hard-forward identity, per-loss gradient
ownership, rho routing and hard-only inference, but the integrated remote test
was not run before the turn was interrupted.

This advances only to `implemented_draft`. ActionFormer RNG isolation and
optimizer grouping, strict AnchorFreeHead native-axis validation, formal
configs/pipeline metadata, P0-P3 gates and official-60 are still missing. No
experiment is deployed and no claim status changes.

The interrupted integrated test was then resumed in disposable remote copy
`opentad_duca_protected_selector_draft_20260720_01`. The first wrapper
invocation failed before Python because a PowerShell two-column upload map was
flattened to characters. The corrected wrapper reached pytest; its first run
reported `10 passed, 4 failed`, with all four failures caused by a
three-index typo in the synthetic fake-ASFormer test source. After correcting
that test-only tensor index, the same focused suite completed with
`14 passed in 38.71s`.

The passing suite covers the standalone physical DAG plus exact-uniform
bypass/native metadata, protected detector-gradient stop at selector
adapter/head, auxiliary action versus transition gradient ownership, rho=0.01
restricted hidden routing and hard-only inference. This upgrades the selector
component to `tested_focused`; it still does not prove official full-model
optimizer ownership, detector RNG isolation, native physical head semantics,
P3 alignment, mAP or cost.

An independent read-only P0-P3 audit returned `HOLD`. It confirmed that
successful-update AMP replay/scheduler/EMA machinery is reusable, while
identifying remaining blockers: train currently constructs val/test loaders,
loader length is not derived and hash-sealed, old official-60 configs and
validator still require selected-axis semantics, protected parameter LR
ownership is implicit, and P3 still targets the old checkpoint/population.

## 2026-07-20 protected detector-contract focused gate

The isolated construction tree now includes route-scoped integration in the
official `ActionFormer` and `AnchorFreeHead` surfaces. The changes activate
only when the selector declares `protected_e2e_physical` and the head declares
contract `duca_protected_e2e_physical_v1`; historical physical-grid routes are
otherwise unchanged.

The focused contract now fail-closes on float selected positions, disagreement
between duplicate position/count fields, unordered or duplicate positions,
selected-axis output declarations, dense-valid-length violations, detector
mask/count disagreement and any GT object replacement. The official detector
RNG stream is snapshotted before selector execution and restored before the
backbone so extra selector stochastic work cannot change the matched
detector-dropout sequence. `ActionFormer.get_optim_groups()` now explicitly
assigns selector adapter/head to `1e-4`, official-ASFormer trunk/stem to
`2.5e-5`, and official-ASFormer encoder/decoder action output heads to `5e-5`,
while rejecting unexpected trainable protected-selector parameters.

The disposable real OpenTAD remote copy
`/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_selector_draft_20260720_01`
passed `py_compile` and the three focused suites with `24 passed in 36.19s`.
The new assertions cover dense-physical proposal centers, fail-closed
coordinate metadata, full protected-parameter optimizer coverage and LR
ownership, detector RNG equivalence, exact detector mask count and dense-GT
identity.

Status remains `tested_focused`, not `empirically_supported`. A built official
VideoMAE-S/AdaTAD model, four independent backward ownership checks, P0
train-only data sealing, P3 hard-swap alignment and official-60 terminal EMA
mAP are still required. The performance objective is to exceed the strictly
matched approximately 65 Avg-mAP uniform baseline, but no gain is claimed or
guaranteed before the formal run.

## 2026-07-20 P0-P3 implementation closure focused gate

The isolated construction tree
`.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` now contains an
uncommitted end-to-end evidence implementation for the frozen physical route:

- P0 derives and hashes the exact train dataset, sampler, drop-last loader
  length, four resolved configs, VideoMAE-S checkpoint, official ASFormer
  source, THUMOS annotation/class map and the 48-window P3 population.
- P1/P2 builds the real VideoMAE-S `ActionFormer` under one-rank DDP/FP16 and
  independently backpropagates action, transition, detector and total losses.
  It verifies exact optimizer coverage, per-parameter gradient ownership,
  exact-hard backbone input, native physical head execution and unselected
  input invariance. Both main and rho gates are explicitly hash-bound to P0.
- P3 preregisters 16 short, 16 medium and 16 long train windows, including
  full and padded windows with `K_eff=min(384,valid_len)`. It enumerates legal
  physical one-swaps, samples 3 from each predicted-gradient quartile, freezes
  detector loss normalization, restores buffers/module modes/custom selector
  replay state/all RNG streams, and aggregates exactly 576 rows with the
  preregistered video-cluster bootstrap thresholds.
- The formal endpoint is 60 epochs with checkpoint-every-5 preserved, sealed
  test loaders during training and only `epoch_59.pth:state_dict_ema` evaluated.
  The post-run finalizer reopens checkpoint/audit/sidecar/predictions and
  recomputes official OpenTAD mAP before accepting a terminal artifact.

The disposable N16R4 copy
`/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_selector_draft_20260720_01`
passed the expanded compile/config/selector/detector/protocol/P3 focused suite:
`37 passed in 53.62s`. Local Windows PyTorch remains unusable because of the
pre-existing `c10.dll` initialization failure; pure P3 tests pass locally.

Status is `tested_focused` only. The tree is not yet a clean exact commit, the
real CUDA main/rho gates and three P3 shards have not run, authorization does
not yet exist, and no official-60 arm or terminal mAP is running. The `>65`
Avg-mAP threshold remains a bounded GO criterion rather than a promised
result.

## 2026-07-20 protected evidence-chain and official-60 closure

The isolated construction tree now also contains fail-closed transactional
submission for both the P0-P3 gate DAG and the four-arm official-60 suite.
The latter accepts only four terminal `epoch_59.pth:state_dict_ema` artifacts
from the same commit, seed, P0 manifest, authorization and successful-update
exposure, then reuses the official OpenTAD recomputation evidence to report
the preregistered differences and whether the best learned arm is strictly
above 65 and its matched exact-uniform arm. Intermediate checkpoints cannot
enter the aggregate.

The Slurm submission rollback was corrected so job IDs are registered in the
parent shell rather than lost in command-substitution subshells. Formal test
evaluation now independently requires an exact clean commit, one process,
seed 3407, epoch-59 EMA and structured metrics even if an outer launcher is
bypassed.

The disposable N16R4 copy
`/data/run01/sczc063/yuzibo/projects/opentad_duca_protected_selector_draft_20260720_01`
passed the expanded selector/detector/config/protocol/P3/evidence suite with
`50 passed in 49.09s`; local pure evidence tests also pass. Status remains
`tested_focused`. This does not establish P3 alignment, terminal mAP or that
rank-packed nonuniform frames preserve VideoMAE tubelet semantics. A clean
exact commit and real P0-P3 CUDA authorization are still required before
official-60 may be submitted.

## 2026-07-21 exact physical candidate and submission state

The complete exact candidate is pushed at
`ee05f610133fc37f8f1ee67b7225bb38ae917cc5` with tree
`a190e399bb1fdfdac230c0a4305c4b08946a8ec1`. The clean N16R4 snapshot
passes 84 matched focused-plus-legacy tests. Exact P0 manifest SHA-256 is
`a02b6e690804d574d7929a408c17b396cc3cca4887a352be6c55270846e46a7e`;
it binds 100 steps/epoch, 60 epochs, 6000 updates/arm, 48 P3 windows and
576 swaps.

No formal evidence job existed when this checkpoint was recorded. Early
submission attempts were rejected before admissible Job creation by a stale
cluster name, stale GPU resource syntax, and finally the account's
`AssocMaxSubmitJobLimit`. The exact commit includes a scheduling-only
single-allocation wrapper that preserves the same main/rho/P3/completion
order and fail-closed authorization while requiring one queue slot. Its
precheck passed. Status remains `tested_focused`; CUDA/P3,
`experiment_running`, official-60, mAP, and claims are still absent.

At 2026-07-21 02:23 +0800, one queue slot opened and the exact serial gate was
accepted as Slurm Job `1177681` (`dp_all_ee05f61`) under
`duca_protected_physical_ee05f61_gate_single_20260721_021705`. Initial state
is `PENDING (AssocGrpGRES)`. Gate-stage status is now
`experiment_running`; this is not P1/P2/P3 success or official-60
authorization.
