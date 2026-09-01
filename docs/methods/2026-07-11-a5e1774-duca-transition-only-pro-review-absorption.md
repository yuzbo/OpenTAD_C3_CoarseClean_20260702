# a5e1774 DUCA Transition-Only Pro Review Absorption

## Source

- User attachments:
  - `258e6bbc-6c27-421c-8c78-2cc36326a728/pasted-text.txt`
  - `bca8f4a3-af2c-4542-834b-b9a43de88600/pasted-text.txt`
- Both attachments are byte-identical.
- SHA256: `011EBB67CC52D943248D18E4638E2220763DED44329BEF8EB78DBD77973BE863`
- Raw archive: `docs/methods/reviews/2026-07-11-a5e1774-duca-transition-only-pro-review-raw.txt`
- Reviewed branch HEAD: `a5e1774b9941312569ca645341da1abad339db61`
- Formal-result commit: `70aa069b895322c2307ffbb13dfdef9fac0d1305`

## Reviewer Verdict

`HOLD`. The current code passes the offline full-window, inference no-leak, single-checkpoint,
fixed-K and same-feasible-family hard/soft structured-selection contracts. It does not yet pass
the claimed indirect-boundary mechanism, hard detector-utility alignment, selected-axis geometry,
matched-baseline or trained full-stack cost contracts.

## Independently Rechecked Repository Facts

The following high-impact claims were rechecked locally against `a5e1774` rather than merely copied
from the review:

1. `C3OfficialActionSegmentationProbe(return_hidden=True)` returns
   `features.transpose(1, 2)`, where `features` is produced by the two-layer spatial stem before the
   official temporal model. It is not an ASFormer encoder/decoder temporal state.
2. The selector concatenates the raw descriptor, action-state features and absolute coarse hidden.
   Its `start_head`, `end_head`, `context_head` and `utility_head` all contribute directly to
   `center_scores`.
3. GT segments create train-only start/end/context/boundary-utility targets. This is not inference
   leakage, but it makes the overall selector a direct boundary-supervised mini-localizer and bypasses
   the claimed indirect state-transition mechanism.
4. `global_structured_topk` uses detached hard Viterbi and non-detached soft forward-backward in the
   same exact-K/max-hole DP state space, followed by `hard + soft - soft.detach()`. Feasible-family
   hard/soft isomorphism is a pass; utility-direction validity remains unproven.
5. The progressive curriculum uses a `step % 100` duty-cycle switch, not continuous score homotopy.
6. Optimizer tests cover `frame_selector.*`, but excluded parameters are skipped without necessarily
   setting `requires_grad=False`; exact full-model optimizer coverage remains open.

## Absorbed Final Method Candidate

The only allowed next DUCA main candidate is **Shared-ASFormer Transition-Only DUCA**:

```text
low-resolution full-window RGB
  -> current spatial stem
  -> official ASFormer
  -> actual ASFormer encoder state + binary action logits
  -> delta hidden / semantic change / delta logits / delta entropy
  -> one transition-utility scorer
  -> exact-K/max-gap structured policy
  -> selected heavy observations
  -> official-derived AdaTAD
```

The main candidate removes absolute-hidden fusion, raw RGB-mean fusion, direct start/end/context/
utility/radius heads and GT boundary-utility proxy. GT boundaries may supervise a single transition
utility distribution and soft boundary coverage during training only. MobileNet is not added now;
it remains a stem-replacement diagnostic after the mechanism has been fairly tested.

## Absorbed Training Contract

- One model, one optimizer and one checkpoint; no standalone checkpoint handoff in the main method.
- Match approximately 13,200 optimizer updates and drive all schedules by successful optimizer steps.
- Binary action head receives only the unscaled action objective.
- Spatial stem and ASFormer trunk receive action and transition supervision, not detector loss.
- Transition scorer receives transition/coverage supervision and a ramped detector bridge.
- Detector always receives its full detector loss; bridge weight only controls the upstream selector
  path.
- Replace duty-cycle policy switching with continuous score homotopy from a feasible uniform reference
  to the learned transition policy.
- Preserve the existing constrained DP; do not replace it with unconstrained dense top-k/resampling.
- Use raw coarse/transition health metrics and full-model optimizer coverage tests.

## Required Gates Before Deployment Claims

1. Same-commit, same-loader, same-step and same per-sample effective-K dense/uniform/periodic/random
   baselines.
2. Transition-only `beta=0` before the protected detector-gradient version.
3. One-swap finite-difference alignment versus transition, actionness and random.
4. Same-selected-frames selected-rank versus physical-time geometry control.
5. Trained-checkpoint full-stack decode/preprocess/H2D/probe/selector/backbone/head/NMS latency,
   energy and memory.
6. Fixed-384 must pass before fixed-256/128 or dynamic MUST. Current MUST remains diagnostic because
   padded detector length is still 384 and real variable-length compute is not realized.

## Reviewer-Proposed, Not Yet Empirical Truth

The review proposes concrete defaults and gates such as 13,200 steps, `beta_max=0.25`, transition
temperature 0.7, coarse AUROC 0.65, one-swap Spearman 0.15 and p50 saving 15%. These are useful initial
specifications, not repository facts or validated universal thresholds. They require implementation,
ablation and calibration before becoming paper gates.

The review's MAC estimates for the current stem and MobileNet are analytical estimates, not measured
latency. Its ASFormer hidden-exposure patch also depends on the external ASFormer snapshot, whose exact
bytes and upstream commit are not in the reviewed Git tree. Implementation must first pin that external
source and verify wrapper logits are numerically equivalent to the original forward.

## Frozen Decisions

- Freeze `a5e1774` as the current direct-boundary joint baseline.
- Do not deploy dynamic MUST, add MobileNet, add selector heads or tune more loss weights before the
  transition-only fixed-384 gates.
- Do not call the existing coarse hidden an ASFormer temporal hidden.
- Do not claim detector-aware utility from nonzero gradient alone.
- If only absolute hidden or direct endpoint heads work, the original indirect-transition hypothesis is
  falsified and the method must be renamed/reframed rather than rescued by wording.

## Status

The review is fully archived and absorbed as a `designed` correction. No transition-only implementation,
test, formal run or empirical support is implied by this document.
