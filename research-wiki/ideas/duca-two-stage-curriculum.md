---
id: idea:duca-two-stage-curriculum
type: idea
status: experiment_running
updated: 2026-07-21
---

# DUCA two-stage curriculum

## Objective

Recover the stable coarse action/transition supervision seen in historical
separate training without giving up the final offline-TAD joint objective.
The detector must not consume an immature learned sampling policy at startup.

## Pre-audit design (protocol-invalidated)

1. P0 trains only the low-cost RGB stem, official ASFormer coarse branch and
   transition/boundary selector. The complete VideoMAE/AdaTAD detector path is
   frozen and skipped. Three preregistered loss scales are tested for 20
   epochs, with checkpoints at 5/10/15/20 and a deterministic training-only
   80/20 split (seed 3407).
2. The official-60 stage uses exactly 6000 successful detector updates. During
   the first 1000 updates, AdaTAD consumes canonical exact-uniform K=384
   samples while all frontend auxiliary and detector-to-selector weights are
   zero. This is detector warmup inside the fixed training budget, not extra
   detector epochs.
3. From update 1000 to 2500 the hard policy moves from uniform to learned
   selection. From update 2500 to 4000 the protected detector-gradient bridge
   ramps to 0.25 and updates the transition scorer only. Late auxiliary
   weights are action 0.1, transition distribution 0.02 and boundary coverage
   2.0.
4. Four matched arms compare exact uniform, scratch joint training, P0
   initialized joint training, and P0 initialized training with the complete
   coarse probe frozen. All keep exact K=384 and the same official AdaTAD
   backend/protocol.

This design is retained as historical memory, not as the current approved
method contract. Exact-commit audit found hidden selector losses, transition
gradients into the coarse trunk, shared-AdamW warmup drift and an unaligned
direct detector bridge. These defects prevent causal interpretation even if a
terminal arm later improves mAP.

## 2026-07-21 two-audit verdict

- Implementation verdict: `HOLD / diagnostic-only`.
- Research-question verdict: continue after one bounded repair.
- Required principles: strict explicit loss schema; gradient ownership;
  optimizer/scheduler/EMA isolation; half-open schedules; exact-K/per-sample
  max-hole gates; matched terminal mAP and decode-to-output cost.
- Leading proposal: exact-uniform and robust pure-delta anchors plus a
  zero-initialized bounded local residual, with coarse features detached and
  frozen during official detector training.
- Audit V1 prefers a detached legal hard-swap detector-loss teacher and
  disables direct detector gradients. Audit V2 instead proposes a local-cell
  hard-forward/soft-RGB-backward bridge. These are materially different
  feedback mechanisms, not equivalent descriptions of one method.
- Coordinator decision: build one real hard-swap alignment harness. Promote
  the local-cell direct bridge only if its sign/rank predicts actual legal
  hard-swap loss changes; otherwise use detached swap distillation as a
  bounded fallback. Do not optimize both simultaneously by default.
- The proposed `{-1,0,+1}` radius and numerical GO/KILL thresholds are not yet
  frozen facts. Reachability, variance and cost audits are required first.

## Next bounded optimization

1. Run a no-training local-family reachability audit: exact uniform, pure
   delta, local GT-boundary oracle and global exact-K/G2 GT-boundary oracle.
2. Repair P0 before new full training: strict loss allowlist, balanced
   positive/negative BCE, detached coarse descriptors, padding-safe spatial
   normalization, a dedicated frontend-only trainer and disjoint optimizers.
3. Keep the coarse action model frozen/eval throughout official detector
   training. Detector feedback may update only the bounded residual scorer.
4. Compare matched `U` exact uniform, `D` pure delta, `R0` residual without
   detector feedback and `R1` residual with the alignment-approved feedback.
5. Terminal EMA Avg-mAP and high-IoU mAP decide the method. Mechanism scores,
   training loss and gradient connectivity cannot substitute for mAP.

## Why detector warmup is not executed in P0

It is possible to run uniform AdaTAD concurrently with P0, but the repository
uses one aggregate loss and global gradient clipping. Even with zero selector
bridge, detector gradients can change frontend effective optimization through
the shared clipping norm, AMP state and optimizer schedule. The current
implementation therefore keeps P0 genuinely frontend-only, then places the
uniform AdaTAD warmup at the start of official-60. This provides the requested
detector warm start without adding detector updates or creating a hidden
coupling in frontend model selection.

## Evidence boundary

Exact implementation commit `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`
passed 83 focused and regression tests, but those tests did not establish the
objective, optimizer-isolation or discrete-utility contracts above. Current
Job `1178591` may provide diagnostic P0 behavior only and must not authorize a
paper-grade official-60 claim. There is no selected P0 checkpoint, terminal
matched mAP, greater-than-65 result or paper-ready claim.

## 2026-07-21 repaired P0 implementation state

The repair now has a concrete code contract in an isolated worktree. P0 uses a
complete 19-key loss mapping and only three active objectives: binary
actionness, transition distribution and transition-boundary coverage. Binary
actionness uses separate positive and negative means. Both transition routes
see detached coarse evidence, so their gradients update the shared transition
scorer but cannot rewrite the official ASFormer representation. The spatial
stem uses GroupNorm, the detector remains frozen and skipped, and global
gradient clipping is disabled to avoid coupling the coarse and selector
objectives through a shared norm.

This is `tested_local_contract`, not `empirically_supported`. A pure config
validator now emits hash-bound P0 evidence before training, and downstream
candidate aggregation requires that evidence. Linux gradient ownership,
padding metamorphic behavior, a real optimizer step and training-only holdout
selection must still pass on an exact clean commit.

The local-family oracle audit is running first. Pilot evidence suggests the
one-per-cell local family can attain the global GT boundary ceiling under the
matched K/G constraints, so the current priority remains fixing learning, not
expanding the selector to an unconstrained global policy. Full-holdout evidence
is not yet available.

## 2026-07-21 full reachability verdict and P0 boundary

The exact 120-record training-holdout audit completed. The privileged local
one-per-cell oracle equals the global privileged oracle on all reported
boundary-recall and both-endpoint-coverage radii. Its mean endpoint distance is
`0.2484` versus `0.2462` for the global oracle and `0.4775` for exact uniform.
The local family therefore has exact-boundary headroom and is not the current
failure cause.

The learned evidence is the weak point: coarse actionness is only
`AUROC=0.6161/AUPRC=0.3750`, and the invalidated checkpoint trails both uniform
and pure delta. Uniform already saturates radius-one coverage, so P0 must retain
exact-distance/r0 diagnostics and downstream matched mAP remains decisive.

P0 repair is now `implemented_and_tested_local_pure`, not yet a CUDA or
empirical result. It makes the three declared objectives exact, isolates their
gradient ownership, replaces batch-composition-sensitive spatial BN and keeps
the detector frozen and skipped. Deployment is deliberately small: one real
one-step gate and three sequential candidates. `DUCA_FRONTEND_ONLY=1` stops
before the old official-60 matrix.

## 2026-07-21 P0 learning-rate and selection-contract correction

The global-curriculum source at `4c777a6` correctly restores the global
`global_structured_topk` feasible family, but a fresh pre-runtime audit found
two P0 optimization defects that are independent of the selector architecture:

1. `select_duca_frontend_checkpoint.py` makes radius-one boundary recall a hard
   gate and its first ranking key, even though the matched uniform reference is
   already approximately `0.9998` at radius one. This contradicts frozen rule
   203 and can select or reject checkpoints using a saturated proxy rather than
   exact-boundary quality.
2. The P0 grid keeps coarse-trunk/action-head/transition-scorer learning rates
   at `2.5e-5/5e-5/1e-4`. The scorer therefore learns four times faster than the
   random-initialized coarse trunk. The running diagnostic still shows
   actionness BCE near `0.68` at epoch 16, while historical learned selection
   clustered strongly at inaccurate locations.

The bounded successor keeps the same global selector and detector. It compares
one old learning-rate control with two coarse-first groups, fixes the auxiliary
losses at `action=1, transition=0.10, transition_boundary=16`, and selects on a
training-only holdout using radius-zero boundary gain, short-action radius-zero
both-endpoint gain, endpoint distance, transition AUROC and coarse
AUROC/AUPRC. Radius one remains diagnostic only. This is inspired by
Uni-AdaFocus's explicit component learning-rate multipliers; it is not a claim
that Uni-AdaFocus backpropagates through hard temporal indices.

## 2026-07-27 training-budget recovery

The original two-stage idea did **not** allocate 30 detector-training epochs
before a fresh 60-epoch detector course. Optional P0 was frontend-only and
skipped the detector; the paper-comparable detector course itself remained
exactly 6,000 successful updates. Its intended internal phases were uniform
warmup, policy release, detector-signal release and stabilization, all inside
one total 60-epoch budget.

The later K=384 rate curriculum instead trained the full detector for 30
epochs in Stage 1 and another 60 epochs in Stage 2. Its `65.385724%` endpoint
is therefore a 90-epoch over-budget candidate, and its epoch-50
`65.650497%` point consumes 80 total epochs. Neither is the recovered
official-60 answer. A fair successor must compare joint-from-scratch and
uniform-warmup curriculum under the same 6,000 detector updates. Intermediate
best selection is allowed only with the same maximum budget, evaluation
frequency and selection rule for every arm.

Canonical details now live in
`research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`.
