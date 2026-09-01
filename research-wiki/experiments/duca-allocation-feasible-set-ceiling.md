---
type: experiment
node_id: exp:duca-allocation-feasible-set-ceiling
title: "DUCA allocation-family feasible-set ceiling"
status: tested
outcome: hold_pending_hash_bound_deployable_map
tags: ["duca", "oracle", "boundary-coverage", "physical-time"]
added: 2026-07-19
---

# DUCA allocation-family feasible-set ceiling

## Question

在同一 `T`、`K` 和物理最大间隔合同下，当前 CellCF 与可跨区域分配预算的新
family 各自最多能达到怎样的边界覆盖和背景释放能力？

## Registered families

1. exact-uniform;
2. current one-frame-per-uniform-cell CellCF;
3. fixed minimum coverage scaffold plus global residual allocation,
   diagnostic only;
4. global exact-K/physical-max-interval variable-quota allocation, primary
   ceiling;
5. unrestricted privileged GT reference.

## Verified design constraints

For the formal stride-4 grid at `L=768,K=384`:

- exact uniform has maximum selected-center interval 12 original frames;
- an original-frame cap of 10 is incompatible with uniform inclusion;
- an original-frame cap of 15 has an effective discrete cap of 12;
- the minimum arbitrary fixed scaffold under that cap has 255 positions;
- the minimum exact-uniform-subset scaffold has 382 positions;
- therefore fixed `192+192` is not admissible under this physical contract.

These consequences apply only if the project explicitly freezes the unit as
original decoded frame index. The earlier "10 or 15 frames" wording is still
unit-ambiguous and must not be silently resolved by implementation.

## Primary geometry outputs

- exact-K and physical maximum-interval compliance;
- dense-hole, decoded-frame interval and seconds interval reported
  separately;
- selected-to-nearest-boundary distance;
- radius 0/1/2/4 boundary density;
- any-endpoint and both-endpoint coverage;
- per-instance minimum selected count;
- short-action coverage;
- background selected fraction and released background quota.

Exact optimization applies only to explicitly encoded geometric objectives.
Detector mAP is not claimed as an exact CP-SAT objective.

## Secondary diagnostic

Evaluate preregistered candidate sets with a frozen checkpoint through the
existing physical-grid ActionFormer path. Keep dense-time GT unchanged and
forbid selected-axis remapping. This is a necessary-condition diagnostic, not
a globally exact detector-utility oracle.

## Preconditions

- freeze coordinate units and valid-mask semantics;
- bind each valid dense position to the actual decoded frame index and
  timestamp rather than relying only on an intended stride formula;
- require the proposed deployable family to contain exact uniform as a
  reproducible feasible point and to have strictly more regional quota freedom
  than one-per-cell CellCF;
- prove exact-K and max-gap for every family;
- use deterministic tie-breaking;
- record solver status and accept exact-oracle language only for `OPTIMAL`;
- state that additive-score DP exactness is with respect to a frozen,
  explicitly serialized score quantization;
- keep train/validation/test boundaries explicit and prevent GT leakage into
  any deployable selector.

## Authorization boundary

This experiment is the immediate bounded task after the project explicitly
freezes the physical unit and cap. It may add diagnostic/solver code,
coordinate exporters/validators, focused tests and frozen-model evaluation
only. It does not authorize selector training changes or remote long runs.
CARA model implementation begins only after a preregistered GO result
demonstrates privileged family headroom, deploy-visible recoverability,
physical compliance and a plausible frontend/heavy-backbone break-even.

## Current implementation and formal run

The bounded Allocation-Ceiling diagnostic is implemented on branch
`codex/duca-allocation-ceiling-20260720` at exact commit
`1d51379d5feb32c8dfb11ec9a2ef238f4c3f7bbe`. It is not a trained selector and
does not authorize a paper-method claim. The code exports deploy-visible
transition evidence from the frozen `1642f26` transition-beta0 checkpoint,
solves exact privileged global exact-K physical-gap families on training
windows, evaluates frozen physical-grid AdaTAD/ActionFormer detector loss, and
profiles the exact family-D solver.

The initial `b18dd8f` formal gate Job `1174706` failed before producing a
ceiling result because the exporter incorrectly required decoder FPS and
annotation FPS to accumulate less than one frame of clock drift. Jobs
`1174707-1174710` were cancelled without runtime. A complete metadata audit
then showed that all 200 training videos have exactly equal decoded and
annotated frame counts, while FPS-clock drift has median about 3.00 frames and
maximum 3.69 frames. Commit `1d51379` therefore makes decoded frame index the
canonical alignment axis: frame-count alignment remains fail-closed within one
frame, while FPS-clock drift is serialized as a diagnostic.

Local pure-contract verification is `49 passed`. The clean Linux snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_allocation_1d51379_20260720`
passed `105` focused tests, including real PyTorch, physical-grid
ActionFormer/AdaTAD candidate evaluation and required C3 regressions.
Precheck root
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_1d51379_precheck_20260720_041207`
passed.

The superseded formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_1d51379_training_20260720_041247`.
Jobs are `1174711` gate, `1174712` export, `1174713` diagnostics, `1174714`
frozen detector candidate loss and `1174715` completion. Gate `1174711`
completed `0:0` in 2:18 and export `1174712` completed `0:0` in 25:55. The
exported 670-window deploy-visible recoverability artifact passed strict
validation with output SHA-256
`e23fdf6ee72341c74944592abea57970de65a63b50975b6c77cdb2d7d9365968`
and summary SHA-256
`999b38bf7413b849d75a49c853bf933b3f995dfc669fc48409941c6863af848f`.
It reports transition-policy AP/F1/AUC
`0.219662/0.304830/0.579192` against the radius-4 transition proxy, but this is
training-side coarse-signal diagnosis, not detector mAP or privileged GT
headroom.

GT diagnostic Job `1174713` failed `1:0` after 12:43 on first GT32 sample
`video_validation_0000983|1016` at
`lex_block_0210_0240`. The mathematical objectives had solved; the old code
computed a `2^29`-weighted integer objective directly from floating binary
values, so a tiny HiGHS residual became a false ambiguity. Jobs `1174714` and
`1174715` were cancelled at zero runtime. The chain has no admissible
completion artifact and is not negative method evidence.

Replacement commit
`8ebdd2a11ea5cc0644979324872a3b1cae5a2170` uses rounded integer variables only
after strict integrality checks; verifies every subproblem's optimal status,
numeric zero MIP gap, finite primal objective and dual bound; recomputes all
boundary, distance, short-action, background, uniform-overlap and
lexicographic objectives exactly from selected positions; and replays every
pinned objective on the terminal solution. Local focused contracts are
`55 passed`. Two independent review rounds first found the terminal-certificate
gap and then returned `P0=0/P1=0/GO` after its repair. The clean Linux snapshot
`/data/run01/sczc063/yuzibo/projects/opentad_duca_allocation_8ebdd2a_20260720`
passed `111` relevant tests and exact precheck passed.

The exact old failure sample `video_validation_0000983|1016` was replayed under
Slurm by valid Job `1175393`, which completed `0:0` in 1:48. Both privileged
families solved with `lex_block_size=30`, and the independent validator
reported `validation_passed=true`. Output/summary/validation SHA-256 values are
`b877bf...4b97`, `224256...7294`, and `19aeae...b0eb`. Earlier temporary Jobs
`1175380/1175392` never entered the solver and remain deployment diagnostics.

The replacement formal root is
`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_8ebdd2a_training_20260720_1320`.
Jobs are `1175395` gate, `1175396` export, `1175397` diagnostics, `1175398`
candidate detector loss, and `1175399` completion. Transactional scheduler
validation passed. Gate `1175395` completed `0:0` in 4:02; its artifact
SHA-256 is
`6030d9fb7110aa7c73b2df244eff50136d1342c5e2e90bd86db485d38faafc61`.
Exact solver replay, candidate-loss, solver-cost, submission and scheduler
validation all passed. Export `1175396` is released and priority-pending;
every later job remains strictly `afterok` gated. Validation/test is not
consumed, selector training is not authorized, and C3/C4/C7/paper readiness
remain unproven.

## Sealed result

All five replacement Jobs `1175395-1175399` completed `0:0`; every stderr is
empty. Final evidence SHA-256 is
`8232f2f0889bc5e0579abcf82d42ab4009397366c5c4b0e6bfd71d0c658ad6d6`.
The full 670-window deploy-score artifact, exact 32-window privileged subset,
32-window frozen-detector loss artifact and 100-sample solver-cost artifact
all pass independent replay.

On 670 matched training windows, exact uniform has mean endpoint distance
`0.464399`, while the deploy-visible transition policy has `0.508843`.
Deploy-score both-endpoint recall changes by `-0.003532` at radius 0,
`-0.151970` at radius 1 and `-0.000595` at radius 2. Its transition-proxy
AP/F1/AUC is only `0.219662/0.304830/0.579192`. Therefore the current frozen
transition score does not recover boundary geometry better than uniform.

On the exact 32-window privileged subset, constrained GT allocation reduces
mean endpoint distance from `0.446381` to `0.238764` and increases
both-endpoint radius-0 recall from `0.035037` to `0.063952`. This establishes
modest coordinate-exact geometry headroom, but radius-1 recall is already
`1.0` for uniform.

Frozen physical-grid detector loss contradicts the boundary-only geometry
objective. Mean losses are `0.224627` uniform, `0.275252` deploy score,
`0.318532` constrained privileged GT and `0.487739` unrestricted privileged
GT. The three nonuniform families beat uniform on only `25.0%`, `21.875%` and
`18.75%` of the 32 paired windows. Exact CPU decoder latency is median
`347.835 ms` and p95 `363.880 ms`; this is decoder-only diagnostic cost, not
full-stack deployment cost.

The diagnostics raise two strong warnings: deploy-visible recoverability is
negative and even privileged boundary concentration worsens the frozen
detector objective. However, frozen training loss is not the TAD task metric:
ranking, decoding and NMS can make loss and final mAP disagree. The current
route is therefore HOLD, not KILL, until a single-use hash-bound official mAP
replay compares exact uniform and the deploy-visible transition policy under
the same checkpoint and post-processing. Privileged GT selection must not
enter the deployable mAP comparison. Selector training and paper claims remain
unauthorized.

## Status

Code, runtime gate and formal training-side diagnostic are `tested`. The
result is a strong warning but not a route verdict. It supplies no detector
mAP, no full-stack cost saving and no final-method support. Next status is
`hold_pending_hash_bound_deployable_map`.

## Connections

由 `research-wiki/graph/edges.jsonl` 维护。
