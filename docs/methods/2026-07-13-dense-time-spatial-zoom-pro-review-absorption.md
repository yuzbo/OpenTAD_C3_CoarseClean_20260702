# Dense-Time Spatial Zoom Pro Review Absorption

## Source certificate

- Original attachment:
  `C:/Users/skywalker/.codex/attachments/5f8506a8-5538-4099-9074-a633872a238b/pasted-text.txt`
- Exact raw archive:
  `docs/methods/reviews/2026-07-13-dense-time-spatial-zoom-pro-review-raw.txt`
- SHA-256:
  `667A319CA2ABB0601EE0D6A76DF9D8D139D1F116A7BD93D55B48CFA2DC655650`
- Archive verification: source and archive are byte-identical, 84,533 bytes.
- Reviewer verdict: `HOLD`; only S1/S2 falsification is recommended before a
  learned Zoom model.

## Accepted conclusions

1. Dense-time spatial allocation is structurally better aligned with high-tIoU
   TAD than aggressive temporal deletion because it can preserve the regular
   detector time grid and boundary observations.
2. Direct Uni-AdaFocus/AdaSpot transplantation is insufficient novelty.
   AdaSpot is the strongest localization-near baseline.
3. The current runtime baseline is 160x160; `img_size=224` is not evidence that
   the actual input is 224. Native 96/112 crops must remain native to save heavy
   spatial tokens.
4. S1 (matched dense-resolution headroom) and S2 (ROI sufficiency at matched
   full cost) must precede any learned routing model. Either failure kills the
   spatial-Zoom main route.
5. Complete cost must include decode, preprocessing, H2D, scout, policy, crop,
   local backbone, fusion, detector, postprocess, memory, and energy.
6. Current learned temporal selection should be frozen as a development
   mainline, while exact-uniform/random/periodic and historical DUCA remain
   baselines. This is not a universal refutation of temporal sparsification.
7. If S1/S2 pass, hard discrete ROI-tube alternatives with train-only detector
   regret are more defensible than treating a nonzero `grid_sample` gradient as
   hard crop utility.

## Corrections and reservations

1. The review says branch `codex/chronotransport-pro-review` is not public.
   Local verification with `git ls-remote origin` returns
   `1f5f7254a390f183121e6c4b7cebcebd2f2954d1`; the branch is public now.
2. AdaTAD's published 160-to-224 result (68.8 to 70.7 at 768 frames) is a useful
   prior, not a matched result for this fork. It justifies S1 but cannot replace
   it.
3. The proposed S1/S2 numerical thresholds, seeds, scout width, 80-pixel global
   view, J=16, K<=2, 96/112 buckets, exact loss weights, teacher cadence, and
   15% latency threshold are proposed preregistration values, not validated
   constants or an approved final specification.
4. The S2 `label-free oracle` is still a privileged dense-teacher reference.
   It may be used on a frozen gate split for headroom only, but repeated route
   decisions must not inspect the official test set. It is neither deployable
   nor a selector baseline and its own compute must be disclosed separately.
5. The full DART-Zoom design is over-specified before S1/S2. In particular,
   one-clip regret additivity, boundary-risk-to-spatial-capacity correlation,
   detached scout inputs, shared global/local/fused heads, and EMA stability
   remain unproved.
6. A new detector subclass is not yet justified. S1 requires only matched
   configs and the existing official-derived detector. After S2, a backbone or
   wrapper interface must be chosen by a minimal-diff audit rather than by the
   proposed filename list.
7. Keeping a 768-frame 224/256 canvas can dominate memory and H2D. The gate
   implementation must test uint8/pinned-memory and CPU/GPU crop paths before
   assuming that smaller local tokens produce a system speedup.
8. Claiming an unchanged ActionFormer head is only valid for the main fused
   path. Reusing the same head for auxiliary global/local losses can change
   optimizer exposure and internal state and therefore requires its own test.

## Project verdict

`PARTIAL_ACCEPT / HOLD_FULL_MODEL`.

The gate protocol is sufficiently concrete to authorize S1 experiment-code
implementation now. It does not authorize DART-Zoom model code, S2 deployment,
or paper claims.

## Authorized implementation boundary

### Allowed now: S1 infrastructure only

- matched dense-224 and dense-256 configs derived from dense-160;
- config-diff validator proving temporal sampling, masks, targets, backbone,
  detector, evaluator, optimizer-update count, and checkpoint rule are matched;
- one-batch memory/shape/positional-interpolation precheck;
- frozen fit/gate/test manifest and seed schema;
- trained-checkpoint full-stack profiler and paired result/CI aggregator;
- no ROI policy, no teacher oracle, no new detector architecture.

### Locked until S1 GO

- source-resolution and native-crop microbenchmarks;
- fixed/random/motion/person/AdaSpot-like candidates;
- the privileged dense-teacher ROI-sufficiency gate S2.

### Locked until S2 GO and a new written-spec audit

- DART-Zoom scout, policy, Viterbi route, EMA counterfactual regret, fusion,
  auxiliary losses, second detector, and second dataset.

## Status mapping

- Dense-Time Spatial Zoom route: `designed` at gate level.
- S1 code: authorized but not yet implemented or tested.
- S1/S2 experiments: not running.
- DART-Zoom: design proposal only, not selected as final.
- Paper status: `HOLD`.
