---
id: idea:duca-protected-e2e
type: idea
status: designed
updated: 2026-07-20
---

# DUCA Protected-E2E

## Objective

Train an offline-TAD pre-backbone fixed-budget frame-selection policy with the
real official AdaTAD/ActionFormer classification and regression losses while
preserving the semantics of the low-cost binary action/background probe.

## Frozen candidate

- Low-resolution official-ASFormer coarse probe returns binary action logits
  and hidden features.
- The protected main arm detaches both before the policy route.
- The existing equal-capacity scorer is exposed as a separately auditable
  selector adapter and score head.
- A lightweight selector adapter and score head receive direct detector
  gradients through exact-hard-forward, soft-backward assignment.
- Hard Viterbi and soft Gibbs slot marginals must use the same physical
  exact-K DAG, including source/internal/sink edges and a per-sample
  exact-uniform-reference cap in seconds.
- The detector uses dense/native physical coordinates through the existing
  physical-grid ActionFormer path; selected-axis GT remap is forbidden.
- The main arm uses rho 0. The only coarse-trunk ablation uses fixed rho 0.01
  into the final ASFormer temporal block.
- Inference consumes actual hard frames and uses no GT, teacher,
  counterfactual detector or answer cache.
- The backend remains the repository's official AdaTAD/ActionFormer path.

## Status boundary

This is `designed` only. The earlier `b3222af` local-slope/selected-axis
implementation is a nonconforming diagnostic, not P1 completion. The required
order is P0 protocol freeze, physical-DAG P1 implementation, P2 gradient
ownership, P3 stratified hard-soft alignment, then one four-arm official-60
matrix. Failure at any gate stops the route.

## Four arms

1. exact-uniform
2. transition-no-bridge
3. protected-E2E
4. protected-E2E with a small fixed gradient scale into the final ASFormer
   temporal block

No broader experiment is authorized before this matrix finishes.

## 2026-07-20 Pro adjudication absorption

Raw review:
`docs/methods/reviews/2026-07-20-280631a-duca-protected-e2e-pro-adjudication-raw.txt`
with SHA-256
`f91db53a83d79f56927b04d38b1e886d2e4260e4528e7882ddd49adbda97ccb0`.

The review read only prompt commit `280631a`, so its claim that no protected
implementation or real detector gradient existed is stale. Later Job
`1176948` established real full-model gradient connectivity for the
superseded candidate. Its structural objections remain active: the current
surrogate is not a Gibbs relaxation of the physical hard feasible family,
selected-axis remap is still enabled, rho/bridge/P3 protocols differ, and no
terminal mAP exists.

The proposed 5940-update count is not accepted without exact-loader evidence.
P0 must derive and hash the real loader length; historical 200-video,
batch-size-2 runs used 100 steps per epoch.

## 2026-07-20 pre-backbone design and paper-readiness review

The route remains `designed`; focused components are `tested_focused`, but the
full method is not. The design review accepts physical exact-K, one hard/soft
DAG, exact-hard forward, protected gradient ownership and native-axis
detector semantics as a coherent hypothesis. It does not accept correctness,
empirical support or paper readiness.

The new highest-priority structural risk is upstream of the physical head:
nonuniform selected frames are packed by selected rank into nominal
16-frame VideoMAE clips/tubelets. Reassigning ActionFormer points to dense
physical coordinates after feature extraction may not repair the temporal
semantics already used by the backbone. P1 must therefore include a complete
384-frame configuration build, chunk/feature/mask contract, exact-uniform
end-to-end parity, short-action support, timestamp-spacing counterfactual and
raw-gather-to-head roundtrip. This is a gate on the frozen candidate, not a
new method arm. Failure stops official training and requires a separately
adjudicated representation revision.

The review also freezes three wording boundaries: fixed `K=384` is adaptive
placement rather than dynamic budget; the action head is binary-supervised
but the shared ASFormer trunk also sees transition/boundary auxiliary
gradients; and low-cost/efficiency claims require decode-to-output
full-stack measurements. Current evidence is insufficient for a full paper.
See
`docs/methods/reviews/2026-07-20-duca-prebackbone-design-paper-readiness-review.md`.
