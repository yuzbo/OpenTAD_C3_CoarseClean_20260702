# GeoRoute-AdaTAD Theory Package

**Status:** `COHERENT AFTER REFRAMING`
**Scope:** offline temporal action detection (TAD), native `2 x 16 x 16`
VideoMAE tubelets, and a fixed per-tubelet token budget.
**Purpose:** explain what the routing mechanism can establish analytically and
what must remain an empirical claim.

## Target

The target is not a theorem about mAP.  The analytical target is the following
conditional statement:

> Under a fixed token budget and the implemented packed attention/MLP path,
> GeoRoute can reduce the token-dependent heavy-backbone operations while
> retaining a structured approximation family that contains both a continuous
> region prior and non-region residual evidence.

Whether that approximation improves high-tIoU detection at lower *measured*
end-to-end cost is a falsifiable P1/P2 experimental question.  It is not
deduced below.

## Invariant Object

Let `x` be a complete offline video window, `y` its detection labels, `theta`
the detector and heavy-backbone parameters, and `phi` the scout/router
parameters.  The invariant training object is the expected detector risk under
the route distribution:

\[
  J(theta, phi) =
  E_{(x,y)} E_{S \sim p_phi(\cdot\mid x)}
  [ L_det(f_theta(x,S), y) ] + lambda R(phi; x).
\]

`L_det` is the unmodified AdaTAD-derived Focal plus DIoU detector loss.  `R`
is only an explicitly logged router regularizer.  A latency or energy term is
not silently placed in this objective: it is measured on the deployed path and
reported separately.  With an exact fixed `K`, the leading selected-token count
does not vary between matched routes; their real system costs can nevertheless
vary because the scout, gather, packing, adapter, and hardware behavior vary.

## Assumptions

The propositions below require all of the following.

1. A window contains `T` temporal tubelet positions and `N` native spatial
   tubelets at each position.  `N` is the boundary-padded source lattice size,
   not a resized image resolution.
2. Every matched route selects exactly `K` distinct native tubelets at every
   temporal position, with `0 < K <= N`; deterministic context tokens are
   counted inside `K`.
3. The implementation executes selected tokens through the packed attention
   and MLP operations.  Any dense operation, including a dense adapter, is
   accounted for separately rather than declared sparse by association.
4. The detector-facing tensor remains `[B, 384, 768]`.  No temporal detector
   position is deleted; spatially selected features are aggregated per
   temporal position.
5. For the score-function identity, `p_phi(S|x)` has differentiable positive
   probability on its sampled support, the required expectations exist, and
   the detector loss is treated as a sample reward with respect to `phi`.
6. The approximation discussion assumes useful spatial evidence decomposes
   approximately into a coherent region-correlated component plus a sparse
   non-region innovation component.  This is a modelling assumption, not a
   property guaranteed by TAD annotations.

## Notation

- `P_t = {p_(t,j)}_{j=1}^N`: native spatial tubelets at temporal position `t`.
- `g_t = (c_x,t, c_y,t, w_t, h_t)`: bounded continuous ROI geometry in source
  coordinates.  It defines a score over existing patch centres; it does not
  create a resized crop tensor.
- `a_(t,j)(g_t)`: continuous ROI-support score for patch `p_(t,j)`.
- `r_(t,j)`: residual free-token score predicted by the scout.
- `C_t`, `G_t`, and `R_t`: deterministic context, geometry, and residual
  selections, respectively; `S_t = C_t union G_t union R_t`.
- `K_c + K_g + K_r = K`: exact allocation of context, geometry, and residual
  tokens.
- `F_t`: dense native token features before routing; `Fhat_t(S_t)` denotes the
  selected/aggregated approximation delivered to the detector.
- `d`: token width and `L`: number of VideoMAE blocks.

## Derivation Strategy

The derivation follows one fixed object, `J(theta, phi)`, through four separate
claims:

1. exact-K selection gives an operation-count comparison for the packed
   attention/MLP subpath;
2. the hard stochastic route admits a score-function gradient identity;
3. a straight-through (ST) gate is a biased surrogate rather than that
   identity;
4. a region-plus-residual route is an approximation family motivated by a
   stated decomposition assumption.

No step turns a lower operation count into lower wall-clock latency, or a
smaller approximation residual into higher mAP.

## Derivation Map

1. Define the exact-K selected support `S_t` and the corresponding packed
   token length `K`.
2. Count dense and packed attention/MLP operations under Assumptions 1--3.
3. Differentiate the expected detector risk by the log-derivative identity.
4. Compare that identity with the gradient produced by an ST replacement.
5. Introduce a conditional geometry-plus-innovation approximation model and
   derive the error decomposition it suggests.
6. Map each mathematical quantity to a required runtime measurement or
   ablation.

## Main Derivation

### 1. Exact-K packed operation property

For one attention block with sequence length `M`, token width `d`, and a
fixed head configuration, the leading self-attention interaction term is
`O(M^2 d)` and token-wise projections/MLP terms are `O(M d^2)`.  The dense
native spatial path therefore has the per-window leading form

\[
  C_dense^{attn+mlp} =
  O\left(L T (N^2 d + N d^2)\right).
\]

Under Assumptions 1--3, exact-K packed attention/MLP has

\[
  C_packed^{attn+mlp} =
  O\left(L T (K^2 d + K d^2)\right).
\]

This is an **operation-count proposition**, conditional on the packed path
actually being executed and excluding unselected-token work from that path.
It supports the ratios `K/N` and `(K/N)^2` as explanatory quantities for
projection/MLP and attention interaction terms, respectively.

It does *not* imply the complete model has this ratio.  Let

\[
  C_total = C_scout + C_decode + C_gather + C_patch +
            C_packed^{attn+mlp} + C_adapter + C_detector + C_system.
\]

If the VideoMAE adapter remains dense, `C_adapter` may still scale with `N`.
Similarly, decoding, host-to-device transfer, padding, gathering, kernel
launches, and NMS may dominate at small `K`.  The paper may only claim a total
cost reduction after the prescribed end-to-end p50/p95, memory, and energy
measurements are available.

### 2. Score-function estimator for hard exact-K routing

For a stochastic exact-K route sampled from `p_phi(S|x)`, differentiate the
invariant object while holding the sampled detector computation fixed with
respect to `phi`:

\[
\begin{aligned}
  \nabla_phi E_S[L_det(f_theta(x,S),y)]
  &= \sum_S \nabla_phi p_phi(S|x) L_det(f_theta(x,S),y) \\
  &= E_S[L_det(f_theta(x,S),y) \nabla_phi \log p_phi(S|x)].
\end{aligned}
\]

For a sequential without-replacement Plackett--Luce sample
`S=(s_1,...,s_K)`, the required log probability is

\[
  \log p_phi(S|x) =
  \sum_{i=1}^K \left[z_{s_i} -
  \log \sum_{j \notin \{s_1,...,s_{i-1}\}} \exp(z_j)\right],
\]

where `z` are the policy logits.  Subtracting any baseline `b(x)` independent
of the sampled route leaves the expectation unchanged:

\[
 E_S[(L_det-b(x))\nabla_phi\log p_phi(S|x)]
 = E_S[L_det\nabla_phi\log p_phi(S|x)].
\]

This is an **identity under Assumption 5**, not proof that variance is useful
at the target `K`, video length, or batch size.  The P0 known-answer test must
compare this estimator with enumerated expected gradients on a tiny finite
case, and P1 must report variance and optimization behavior.

### 3. Why the straight-through gradient is biased

Let `H` be a hard binary exact-K membership vector and `Q_phi` be a smooth
relaxed support.  A common ST construction is

\[
  S_ST = H + Q_phi - stopgrad(Q_phi).
\]

The forward value equals `H`, while automatic differentiation uses
`nabla_phi Q_phi`.  In general,

\[
  E[\nabla_phi L_det(f_theta(x,S_ST),y)]
  \ne \nabla_phi E_S[L_det(f_theta(x,S),y)].
\]

The mismatch comes from replacing the discontinuous change in selected support
with a pathwise derivative of a different relaxed computation.  The ST route
is therefore a **biased optimization surrogate**.  It may be retained only if
the code labels it as such and P0/P1 show a useful stability/accuracy/cost
trade-off against a stop-gradient and score-function control.  It must never
be described as an unbiased hard-policy gradient.

### 4. Structured approximation interpretation

For an analysis slice, suppose the dense feature contribution at time `t` can
be represented as

\[
  F_t = F_t^{geo}(g_t^*) + E_t^{res},
\]

where `F_t^{geo}` is concentrated near an unknown coherent support associated
with geometry `g_t^*`, and `E_t^{res}` has relatively sparse, non-region
innovations.  A geometry-only approximation has residual

\[
  e_t^{roi} = \|F_t - Fhat_t(G_t)\|.
\]

Allowing `K_r` residual selections expands the feasible selected support from
`G_t` to `G_t union R_t`; therefore, for the *same aggregation family* and a
residual selector allowed to choose the best additional support,

\[
  \min_{R_t: |R_t|=K_r} \|F_t-Fhat_t(G_t\cup R_t)\|
  \le \|F_t-Fhat_t(G_t)\|.
\]

This inequality is a set-inclusion **proposition** about the approximation
family.  It says nothing about what the learned residual scorer finds, whether
the selected token features are sufficient for a detector, or whether the
larger family generalizes.  It motivates the direct ablation `ROI-only` versus
`ROI+residual`, and the decisive comparison against `free TokenSelect`:
if free selection wins the high-IoU/cost Pareto, the structured ROI claim is
not supported.

### 5. Temporal continuity is a regularized prior, not proof of tracking

When knots are interpolated across tubelet positions, a total-variation term

\[
  R_{tv}(g) = \sum_{t=2}^{T} \|g_t-g_{t-1}\|_1
\]

encodes a preference for slowly varying geometry.  It does not prove that the
ROI follows an actor, action object, or annotated boundary.  Fast motion,
camera cuts, and multi-actor actions are explicit counterexamples; residual
tokens and the spatial/temporal diagnostic plots are intended to expose those
regimes rather than hide them.

## Falsification Map

| Analytical statement | Observable required to challenge it |
| --- | --- |
| Packed attention/MLP uses `K`, not `N` tokens | P0 runtime audit: selected count, packed count, one-heavy-forward count, and shape checks |
| Whole-model cost declines | Same-node end-to-end p50/p95, peak memory, and energy including scout/gather/adapter/detector |
| Score-function implementation matches its identity | Enumerated finite known-answer gradient and variance report |
| ST is only a surrogate | Explicit estimator label, stop-gradient control, and empirical comparison |
| ROI+residual is a useful structural prior | Matched free/ROI/hybrid high-IoU-cost Pareto and structure diagnostics |
| Geometry is temporally coherent without collapse | ROI area, centre velocity, coverage, and residual-allocation distributions |

## Boundaries and Non-Claims

1. There is no theorem from token approximation error to tIoU mAP.
2. There is no latency theorem: operation counts omit hardware and dense
   components until profiling measures them.
3. The theory does not prove the learned ROI is semantically meaningful,
   tracks actors, or outperforms free TokenSelect.
4. The score-function identity applies to the stated stochastic distribution;
   it does not validate a deterministic staged hybrid likelihood unless that
   likelihood is implemented and tested.
5. ST's forward hard selection does not make its backward derivative exact.
6. A-MoD/depth routing is outside these statements until its independent P0
   numerical and cost gate passes.

## Open Risks

- The dense adapter can erase the expected total-cost advantage at modest `K`.
- Native token routing may lose contextual information that free selection or
  dense computation preserves.
- A rectangular geometry prior can be harmful for multiple disjoint actors.
- Score-function variance can make the unbiased estimator impractical.
- The selected-feature aggregation can be the failure point even when routing
  geometry is visually plausible.
