---
updated: 2026-07-07
status: active
scope: Absorb the Pro-tier route review for commit 46cacc113042fcf0931c70774491d44665246e32 and lock the final DUCA-TAD research target.
out-of-scope: Treating the review as reproduced experiment evidence, claiming current code is complete, or changing running jobs.
---

# 46cacc1 Pro Final Route Review Absorption

Raw record:
`docs/methods/reviews/2026-07-07-46cacc1-pro-final-route-review-raw.txt`

Reviewed target:

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`
- Commit: `46cacc113042fcf0931c70774491d44665246e32`
- Visibility: PASS for public commit. Pending local patches were considered only
  as route context, not public-code evidence.

## Absorbed Verdict

The Pro review is accepted as a valid route-level critique and planning input.
It is not an independent reproduction of experiment numbers.

Final absorbed verdict:

> The public commit is not CVPR-ready. The final target should be a staged
> detector-utility-calibrated selector: train-only AdaTAD responsibility utility
> distillation, fixed `K=384` true-time sparse acquisition, and optional joint
> fine-tuning only after real fulltrain evidence.

The review confirms that the current public repository is stronger as an
engineering and diagnostic base than as a final paper method.

## Non-Acceptance Decisions Confirmed

The review explicitly supports the five decisions already made in this thread:

1. Do not use lattice replacement as the final method.
   It starts from a uniform scaffold and performs local replacement. It is a
   diagnostic bridge, not the intelligent acquisition contribution.

2. Do not use 768 or dynamic `>384` as paper-main evidence.
   The paper target is fixed `K<=384` sparse pre-backbone acquisition.

3. Do not call current Stage2 proposal-score utility detector-aware completion.
   The current public utility is proposal-score-derived and lacks point
   responsibility, loss sensitivity, high-IoU responsibility, or counterfactual
   utility.

4. Do not call current Stage3 smoke/precheck end-to-end training.
   Stage3 must show real THUMOS fulltrain, nonzero selector losses, selector
   parameter movement, selected-position drift, mAP/tIoU, and anti-collapse
   evidence before any end-to-end claim.

5. Do not make engineering rules the central novelty.
   Strict ledgers and repair/scaffold rules are necessary defenses, but the
   CVPR-level method must be detector-utility-calibrated acquisition.

## Final Research Target

The final method should be framed as:

**DUCA-TAD: Detector-Utility-Calibrated Sparse Temporal Acquisition for Temporal
Action Detection**

The target contribution:

- a trainable pre-backbone sparse temporal acquisition module;
- fixed `K=384` selected dense temporal positions per 768-window;
- train-only dense AdaTAD teacher utility;
- deploy-time selector using only low-cost p_action-like descriptors, without
  GT, teacher cache, raw detector prediction, or oracle boundary;
- true-time sparse metadata preserved into detector training/evaluation;
- optional joint fine-tuning only after Stage2 fixed_384 is credible.

The method should be described as plug-in after detector-aware training, not as
zero-shot plug-and-play.

## Minimum Paper Threshold

The sparse method does not need to beat dense AdaTAD teacher. Dense teacher is a
reference and utility source.

The main comparison is matched exact uniform_384 under the same commit, same
config, same detector, same eval schedule, same split, and same selected-count
contract.

Absorbed thresholds:

- Minimum submission threshold: `Avg-mAP >= uniform_384 + 0.7`, with mAP@0.6
  and mAP@0.7 not lower.
- Credible CVPR threshold: `Avg-mAP >= uniform_384 + 1.0`, and mAP@0.6 or
  mAP@0.7 improves by at least `+1.0`.
- Strong claim threshold: `Avg-mAP >= uniform_384 + 1.5`, mAP@0.7 `+1.5`, plus
  real sparse compute evidence.

Selected count alone is not compute evidence. The sparse path must prove that
backbone/projection does not pad back to 768, or it must disable compute claims.

## Implementation Gap Estimate

Absorbed public-commit readiness estimate:

- engineering / ledger / provenance scaffold: about 70%;
- detector-aware utility: about 25%;
- end-to-end training: about 20%;
- experimental evidence: about 30%;
- paper claim readiness: about 15%;
- CVPR novelty readiness: about 35%, rising only if true responsibility utility
  is implemented and validated.

Local pending patches may improve claim-gate and scaffold truthfulness, but they
do not change the key method gap: Stage2 utility quality and Stage3 fulltrain
evidence remain incomplete.

## P0 Issues Absorbed

1. Public commit lacks a visible `<=384` paper-main claim-budget gate.
   Local pending gate must be committed and pushed before it can count as public
   evidence.

2. Public lattice summary is not truthful enough.
   It must say it uses a uniform scaffold, or lattice cannot be included even as
   a diagnostic without misleading metadata.

3. Stage2 teacher utility manifest conflicts with actual generation path.
   The public exporter uses `forward_test` proposal scores, while metadata
   suggests train-forward detector utility. This must be renamed or fixed.

4. Current Stage2 utility is not detector responsibility.
   A true method needs assignment/loss/gradient/boundary/false-positive utility,
   not proposal center score projection.

5. Stage3 cannot claim end-to-end.
   Selector losses are zero placeholders in public code, and detach semantics
   can sever detector-loss learning depending on mode.

6. True-time versus selected-axis coordinates need detector-head-level proof.
   Metadata alone is insufficient if the head/evaluator ignores the true-time
   mapping.

7. Matched uniform_384 anchor is not locked.
   Without re-running exact uniform_384 under the same conditions, no method can
   claim to beat the sparse baseline.

## P1/P2 Risks Absorbed

- `allow_short_valid_ratio_count=True` is too broad for paper-main exact 384.
- Stage2 training still contains dynamic-budget behavior unless fixed_384 is
  explicitly locked.
- Stage3 precheck protects against false claims, but therefore cannot be mAP
  evidence.
- FLOPs/latency/memory must be profiled, not inferred from selected count.
- Utility source manifest must distinguish proposal-score surrogate, point-loss
  responsibility, gradient sensitivity, saliency, and counterfactual utility.
- Naming should be simplified for paper: `DUCA-384`, `DUCA-Joint-384`, and
  `Lattice-Diag-384`.

## Required Implementation Route

### Stage A: Lock Evidence And Baselines

- Commit and push local `<=384` claim-budget gate.
- Commit and push lattice truthfulness/provenance gate.
- Re-run matched exact uniform_384.
- Re-run or preserve PAction learned fixed_384 under the same commit/config
  family.
- Treat GAS-VT fixed_384 and lattice replacement as diagnostics only.

### Stage B: Responsibility Utility Export

Add a new responsibility utility path instead of overloading the proposal-score
surrogate:

- `tools/bata/export_adatad_responsibility_utility.py`
- `tools/bata/validate_adatad_responsibility_utility.py`
- updates to `tools/bata/detector_teacher_utility.py`
- detector/head hooks for per-level/per-point assignment and loss export.

Each teacher point should include level, point index, true-time center/support,
assigned GT id, boundary role, cls/reg/quality loss, predicted IoU, grad norm,
false-positive score/risk, positive gain, negative risk, and
`utility_source_type="point_loss_gradient_responsibility_v1"`.

The first useful version should implement assignment + loss + gradient
sensitivity before expensive counterfactual drop/add.

### Stage C: Fixed-384 Selector Training

- Train selector with fixed top-k `K=384`.
- Disable or lock dynamic budget classifier for paper-main.
- Use utility distillation, pairwise ranking, false-positive risk penalty,
  boundary bracket loss, soft max-hole loss, and entropy annealing.
- Emit strict train/val/test ledgers with no teacher/GT at deploy.

### Stage D: AdaTAD Fulltrain

- Run AdaTAD on strict DUCA-384 ledger with matched settings.
- Report mAP curve, high-IoU mAP, selected-count histogram, max/p95 holes,
  boundary bracket/support, utility capture, short-valid exceptions, and compute
  profiling.

### Stage E: Optional Stage3 Joint Fine-Tuning

Only after Stage2 responsibility utility approaches or beats uniform_384:

- use true-time sparse selector in the detector graph;
- add nonzero selector losses;
- prevent action-interior collapse;
- log selector parameter movement, selected-position drift, gap distribution,
  boundary bracket, high-IoU mAP, and collapse metrics.

## Experiment Matrix

Paper-main candidates:

- matched exact uniform_384;
- Stage2 responsibility utility DUCA-384;
- optional Stage3 DUCA-Joint-384 after Stage2 succeeds.

Required baselines:

- raw p_action top-k384;
- PAction learned fixed_384;
- GAS-VT fixed_384;
- Stage2 proposal-score utility 384.

Diagnostics only:

- lattice diagnostic 384;
- 768 and dynamic ceilings;
- no-CVaR/no-repair variants;
- post-hoc repair/scaffold variants.

Every run must save ledger metrics and detector metrics. Ledger metrics include
selected count min/max/mean, valid length, short-valid rows, max/p95 gap and
hole, boundary bracket recall, utility capture@384, uniform Jaccard, p_action
coverage, and foreground/interior/background selection ratio. Detector metrics
include Avg-mAP, all tIoU APs, prediction/NMS counts, per-class AP, checkpoint
epoch, eval script sha, wall-clock, FLOPs, memory, and whether compute claim is
enabled.

## Pivot Conditions

The review sets hard stop lines:

- If matched uniform_384 is around 65 and Stage2 responsibility utility remains
  below 62, do not package as a beat-uniform paper.
- If Stage2 responsibility utility fails to beat PAction learned fixed_384 by at
  least +1.0, revisit utility target rather than adding Stage3 complexity.
- If Stage3 joint collapses into action interiors, hurts mAP@0.7, or worsens
  p95 gap, do not make it the main method.
- If sparse path still pads back to 768 and has no compute gain, disable
  efficiency/pre-backbone compute claims.
- If all learned K=384 selectors fail to beat uniform, pivot to explaining why
  uniform is hard to beat, or move toward post-backbone detector-aware token
  pruning / proposal-level compute allocation.

## Immediate Action Items

1. Commit and push local scaffold-truthfulness and claim-budget gate patches.
2. Re-sync remote clean snapshot after the push.
3. Re-run focused pytest/py_compile on remote.
4. Re-establish matched uniform_384 under same config and commit.
5. Keep current dense teacher run as utility source, but do not use proposal
   score surrogate as the final Stage2 method.
6. Implement `export_adatad_responsibility_utility.py`.
7. Run fixed_384 DUCA responsibility selector before expanding Stage3.

