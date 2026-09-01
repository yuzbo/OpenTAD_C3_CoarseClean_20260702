# DUCA two-stage curriculum Pro audit absorption

## Source and evidence boundary

- Raw review:
  `docs/methods/reviews/2026-07-21-6f2ed48-duca-two-stage-curriculum-pro-audit-raw.txt`
- Raw SHA-256:
  `0b265d08b811b821b1014cf7c52b579a759ee79e637710260a48cfc284367379`
- Reviewed repository commit:
  `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9`
- External review mode: read-only GitHub static audit. It did not execute CUDA,
  Slurm, training, cost profiling or terminal evaluation.
- Independent local-side verification used the clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_twostage_7431543_20260721`,
  whose HEAD is the exact reviewed commit. The coordinator worktree does not
  contain this Git object and was not used as a substitute.

## Coordinator verdict

The central verdict is accepted:

```text
implementation: HOLD before paper-grade official-60
research question: continue after repair
current two-stage run: diagnostic only
```

This is not full acceptance of every proposed architecture choice or numeric
threshold. The code defects and causal-identifiability concerns are strongly
supported; the claim that one particular bounded-residual design is the
unique final architecture is a proposal that still needs matched mAP and cost
evidence.

## Independently confirmed code facts

1. Formal P0 does not have a strict three-loss objective. `duca_losses()`
   starts from nonzero defaults for budget, boundary, hole, redundancy,
   radius and entropy, then partially overwrites them. The P0 config only
   names actionness, transition and transition-boundary weights.
2. `ActionFormer.forward_train()` records every selector loss key returned by
   the selector, and selector-only mode sums that complete set into `cost`.
   The hidden defaults are therefore optimization terms, not log-only values.
   Running P0 logs also show nonzero boundary, redundancy and entropy terms.
3. The auxiliary transition descriptor detaches action logits but not ASFormer
   hidden features. Transition-distribution supervision therefore updates the
   ASFormer trunk, while the policy path at hidden-gradient scale zero is
   detached. This violates the intended separation between coarse action
   semantics and indirect boundary selection.
4. Official warmup sets only selected frontend losses to zero while keeping
   frontend parameters in a shared AdamW and global gradient clip. It does not
   prove parameter, optimizer-state, buffer or EMA invariance. Hidden losses
   make the problem stronger than weight-decay-only drift.
5. The schedule uses `step <= warmup`, so a nominal 1000-update warmup spans
   update indices 0 through 1000. The half-open schedule contract is not met.
6. P0 skips the detector forward, but still constructs and places the complete
   detector, DDP wrapper and whole-model EMA on the GPU. This is verified
   inefficiency, although it is not by itself a scientific-invalidity proof.
7. The protected direct bridge is autograd-connected to the scorer, but its
   local RGB-slope backward surrogate does not recompute hard selected RGB,
   selected-axis GT remapping, assignment or regression distances. Gradient
   connectivity is not evidence of correct discrete swap utility.
8. Selected-axis remapping and inverse mapping are statically coherent, but
   the detector operates on selected-rank coordinates. The method must not be
   described as preserving the original AdaTAD coordinate semantics exactly.

## Where the review overreaches

1. The deployed P0 log establishes 80 batches per epoch (`0..79`) for the
   current data snapshot, so the concrete run is not currently ambiguous
   between 79 and 80 batches. The missing hash-bound successful-update
   contract remains a valid reproducibility defect.
2. Selecting among 12 train-holdout candidates creates multiple-comparison
   risk, but it is not automatically invalid if the test split is untouched,
   the search space is preregistered and uncertainty is reported. Reducing to
   one recipe is cleaner; it is not the only statistically defensible option.
3. A dedicated frontend-only model is desirable for memory, speed and artifact
   clarity, but complete detector construction is not alone a reason to reject
   the scientific hypothesis when detector execution and updates are proven
   absent.
4. Dual optimizers are the clearest isolation mechanism, not a mathematical
   necessity. Equivalent explicit parameter exclusion and independent
   transactional state machines could satisfy the same contract.
5. Exact-uniform plus fixed pure delta, a zero-initialized residual and local
   `{-1,0,+1}` swaps is well motivated by the observed misplaced clustering,
   but the radius and residual bound are not proven optimal. They require a
   feasibility, reachability and oracle-headroom audit before being frozen.
6. Detached hard-swap detector-loss differences are better aligned than the
   current RGB-slope surrogate, but remain a training surrogate rather than
   mAP. They may be expensive and noisy. Final adjudication must use matched
   terminal mAP, high-IoU metrics and decode-to-output cost.
7. The proposed `+0.80`, absolute `65.0`, high-IoU and cost thresholds are
   useful preregistration candidates, not truths derived from the static code
   audit. They need variance and power justification before becoming a kill
   rule.

## Absorbed architecture direction

The preferred repaired candidate is recorded as a bounded experiment, not as
an empirically supported final method:

```text
low-resolution coarse action model
  -> fixed robust |delta p_action| anchor evidence
  -> detached transition descriptors
  -> zero-initialized bounded residual scorer
  -> local exact-K/max-hole dynamic program around uniform anchors
  -> hard selected frames
  -> AdaTAD/ActionFormer detector
```

During official detector training, the coarse model should be frozen. The
scorer should receive explicit transition/boundary supervision and, only after
alignment gates, detached legal hard-swap detector-utility distillation. The
current direct hard-forward/soft-backward bridge is diagnostic-only.

## Required repair gates

1. Strict formal loss allowlist with every weight explicit and undeclared or
   missing keys failing closed.
2. Per-loss/per-module gradient-ownership tests and FP32 gradient ledgers.
3. Coarse hidden detachment from transition, boundary, utility and detector
   objectives unless a separately named ablation enables it.
4. Independent optimizer, scheduler, clip, AMP-replay and EMA transactions for
   detector and scorer; byte-invariance through the uniform warmup.
5. Half-open update schedule tests at every transition boundary.
6. Exact-K, per-sample max-hole, hard/soft support and selected-axis roundtrip
   tests for the bounded residual decoder.
7. Real one-swap utility recomputation including GT remapping and detector
   assignment; direct-bridge sign/Spearman alignment remains diagnostic.
8. Decode-to-output cost ledger including dense decoding/preprocessing,
   frontend, selector, VideoMAE, detector head, inverse mapping and NMS.
9. Matched terminal-EMA mAP remains the primary method verdict. P0 mechanism
   metrics may gate entry but cannot establish method superiority.

## Running-experiment interpretation

- Job `1178591` tests the pre-repair two-stage implementation. Its outputs may
  diagnose optimization and mechanism behavior but must not unlock a
  paper-grade official-60 claim under the repaired contract.
- Job `1178642` resumes immutable legacy selected-axis arms at commit
  `cb89586`. This review did not audit that exact commit. Those arms remain
  historical matched diagnostics, not evidence for the proposed repaired
  bounded-residual method.
- No current result proves Avg-mAP above 65 or paper readiness.

## Decision

Accept the HOLD and repair requirement. Accept bounded residual selection,
strict objective isolation and hard-swap utility as the leading next design.
Reject the words `unique`, `final` and fixed numeric kill thresholds until
implementation gates, matched terminal mAP and total-cost evidence exist.
