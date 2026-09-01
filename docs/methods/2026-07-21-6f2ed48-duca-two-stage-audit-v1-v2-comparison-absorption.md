# DUCA two-stage audit V1/V2 comparison and absorption

## Evidence boundary

- Reviewed repository: `yuzbo/OpenTAD_C3_CoarseClean_20260702`.
- Reviewed branch: `codex/duca-two-stage-curriculum-20260721`.
- Reviewed commit: `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`.
- Audit V1 raw record:
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-pro-audit-raw.txt`.
- Audit V1 SHA-256:
  `0b265d08b811b821b1014cf7c52b579a759ee79e637710260a48cfc284367379`.
- Audit V2 raw record:
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-route-audit-v2-raw.txt`.
- Audit V2 SHA-256:
  `bca69084bfb1c09f5fe92d49aa10362b18fecf69ff8d2fa754c1d53335734703`.
- Both external reviews are read-only GitHub source audits. Neither review ran
  CUDA, Slurm, training, terminal mAP evaluation or cost profiling.
- Coordinator verification used the clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`
  at the exact reviewed commit.

## Are the two recommendations identical?

No. Their diagnosis is highly consistent, and their broad replacement family
is consistent, but their detector-feedback mechanism, local selection geometry,
P0 schedule and numerical protocol are materially different.

### Shared diagnosis

Both audits agree that the current implementation must be placed on HOLD:

1. The declared P0 three-loss contract is false because nonzero Python
   defaults enter the returned selector loss set and therefore `cost`.
2. Transition-distribution supervision reaches live ASFormer hidden features,
   so the coarse branch is not learning only action/background semantics.
3. The first 1000 official updates are not a real frontend freeze while the
   frontend remains in shared AdamW, global clipping, train-mode buffers and
   whole-model EMA.
4. A 197-dimensional learned global scorer can overwhelm the reliable
   `abs(delta p_action)` signal and move selections within a very large
   exact-K/G2 feasible family.
5. The current expected-position/RGB-slope bridge proves autograd connectivity,
   not alignment with the loss change caused by a legal hard frame swap.
6. Selected-axis mapping is internally coherent but changes detector temporal
   geometry and must not be described as source-identical AdaTAD semantics.
7. Job `1178591` is useful only as a diagnostic of the pre-repair protocol and
   cannot authorize paper-grade official-60 experiments.

Both audits also favor the same high-level successor:

```text
exact-uniform coverage anchor
  + immutable pure abs(delta p_action) evidence
  + detached, zero-initialized bounded residual scorer
  + local rather than global selection freedom
  + coarse semantics isolated from selector/detector objectives
  + independent optimizer transactions
```

### Material differences

| Dimension | Audit V1 | Audit V2 | Coordinator decision |
| --- | --- | --- | --- |
| Detector feedback | Detached legal hard-swap detector-loss utility teacher; direct bridge disabled | Local-cell hard-forward/soft-RGB-backward bridge; hard-swap alignment used as a gate | Do not combine them blindly. Build one shared alignment harness. Promote the direct bridge only if it predicts actual legal hard-swap loss changes; otherwise use detached swap distillation as a bounded fallback. |
| Local geometry | Each uniform anchor chooses from approximately `{-1,0,+1}` under exact-K/G2 DP | One selection per exact-uniform cell | Run a no-training reachability/oracle audit before freezing either family. Neither local radius is proven optimal. |
| P0 transition schedule | Action-only for updates 0-199, then transition/boundary ramp | Transition/boundary ramp from update 0 | Preserve an action-only bootstrap so the coarse semantics exist before residual supervision starts. |
| P0 weights | Action/transition/boundary `1/0.25/1` after normalization | `1/0.5/2` | Neither scalar set is accepted as evidence. Normalize the objectives, isolate their parameter ownership, then freeze one setting using a train-only gradient ledger rather than test mAP or a weight grid. |
| P0 checkpoint | Only update 1600 is eligible | Earliest passing checkpoint among 400/800/1200/1600 with multiplicity correction | Use one preregistered sequential rule. The conservative default is terminal update 1600 only; intermediate checkpoints remain diagnostics unless an explicit sequential stopping protocol is implemented before training. |
| Scorer optimizer | Peak LR `5e-5`, clip `0.5` in official training | Starts at `1e-4`, later decays; clip `0.25` | Treat these as implementation proposals, not facts. Freeze after a real one-step and short train-only stability gate. |
| GO/KILL thresholds | Stricter seed-3407 mAP and latency thresholds | Different continuation, high-IoU and cost thresholds | Neither set is statistically derived. Terminal matched mAP, high-IoU mAP and total cost are mandatory, but exact cutoffs require variance and power justification. |

## Additional V2 facts independently confirmed

The exact remote snapshot confirms the additional code facts emphasized by V2:

1. `entropy_anti_collapse_loss` is negative entropy times a positive weight;
   minimizing it pushes action probabilities toward higher entropy and can
   oppose coarse binary discrimination.
2. `balanced_binary_actionness_loss()` defaults to `positive_prior=0.5`, which
   produces `pos_weight=1`; it is not batch-balanced BCE.
3. Transition target and boundary radii default to four while the desired
   discriminative mechanism gate is radius one.
4. The transition descriptor has 197 dimensions when hidden size is 96, of
   which 192 are signed and absolute hidden differences.
5. The official-ASFormer probe stem contains two `BatchNorm2d` layers. Frames
   are flattened across `B*T` and passed through the stem before invalid time
   positions are masked, so padding can affect train-mode BN statistics.

These confirmations strengthen the HOLD. They do not establish that V2's
exact residual scale, bridge coefficient, learning rates or statistical
thresholds are optimal.

## Absorbed next model contract

The next implementation should preserve the original DUCA purpose: a cheap
coarse model learns action/background state, state changes provide indirect
boundary evidence, and the detector may optimize only the residual selection
decision. Detector gradients must not rewrite the coarse action semantics.

```text
low-resolution RGB
  -> coarse action model trained only by balanced action BCE
  -> immutable pure abs(delta p_action) base
  -> detached transition descriptors
  -> zero-init bounded local residual scorer
  -> exact-K local coverage-preserving hard selection
  -> VideoMAE-S/AdaTAD/ActionFormerHead
  -> true-time inverse mapping
```

### Phase 0: reachability before learning

Before implementing a specific cell radius, compare on the frozen training
holdout only:

1. canonical exact uniform;
2. pure-delta selection inside the candidate family;
3. GT-boundary oracle inside the same local family;
4. GT-boundary oracle under the existing global exact-K/G2 family.

Report r0/r1 boundary recall, short-action both-endpoint recall, endpoint
distance, physical gap distribution and the local-versus-global oracle gap.
If the local oracle has no material headroom over uniform, the local family is
too restrictive and must be rejected before full training.

### Phase 1: repair P0 semantics and mechanics

1. Add a strict complete loss inventory. Undeclared keys fail closed and zero
   weights do not construct gradient paths.
2. Replace the misleading BCE with separate positive and negative means.
3. Detach logits and hidden descriptors for every selector objective.
4. Replace padding-sensitive BN with GroupNorm or execute the stem only on
   valid frames; require a padding metamorphic test.
5. Use a dedicated frontend-only trainer and selector-only EMA/checkpoint.
6. Train coarse and residual scorer with disjoint optimizers and independent
   clipping inside one atomic AMP transaction.
7. Count successful optimizer updates, not nominal epochs. Use one frozen P0
   selection rule and no test mAP.

### Phase 2: decide detector feedback instead of merging it

Implement a common real-model alignment harness that, for legal local swaps,
measures the actual hard detector cls/reg loss after selected-axis GT remapping.
Compare two candidates:

1. local-cell hard-forward/soft-RGB-backward gradient;
2. detached hard-swap utility distillation.

The direct bridge is promoted only if its rank/sign direction agrees with the
actual hard-swap loss with a preregistered video-cluster confidence interval.
If it fails, it remains diagnostic and the detached utility teacher is the
only allowed detector-aware candidate. If both fail, keep a no-bridge residual
baseline and drop the protected detector-feedback claim.

### Phase 3: official detector training

1. Coarse stem/ASFormer/action head stay frozen and in `eval()` for all
   official detector updates.
2. Updates 0-999 bypass the complete frontend and feed canonical uniform
   frames to AdaTAD.
3. Detector and scorer use disjoint optimizers, schedulers, clips and EMA
   state, with atomic skip/replay behavior.
4. The local residual policy starts only after the uniform warmup. The chosen
   detector-feedback route starts later and only after its alignment gate.
5. Formal verdict uses terminal EMA Avg-mAP and mAP at 0.6/0.7. Mechanism
   metrics and detector loss cannot replace mAP.

### Bounded experiment order

For model development, run these matched arms before rerunning explanatory
global-policy variants:

1. `U`: exact uniform;
2. `D`: immutable pure-delta local selection;
3. `R0`: bounded residual without detector feedback;
4. `R1`: bounded residual with the alignment-approved feedback route.

Only if `R1` beats matched `U` on terminal mAP without high-IoU or total-cost
regression should the historical scratch/joint/frozen global arms be rerun as
paper ablations. Historical `65.696`, `68.97` and `76.67` remain unmatched
context, not causal controls.

## Final adjudication

```text
shared diagnosis: accepted
current implementation: HOLD
current Job 1178591: diagnostic only
bounded local residual family: designed, not implemented
V1 hard-swap teacher: candidate/fallback, not proven
V2 local-cell direct bridge: candidate pending alignment, not proven
next action: reachability audit, then P0 contract repair, then feedback gate
```

The two reviews therefore support one repair family but not one already proven
implementation. The coordinator does not accept the words `unique`, `final`
or any fixed numerical threshold until exact-commit gates and matched terminal
mAP/cost evidence exist.
