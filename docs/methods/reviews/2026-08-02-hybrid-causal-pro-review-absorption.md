# Hybrid-centered causal pilot: Pro review absorption

Date: 2026-08-02

Source: user attachment `pasted-text.txt`, 1,101 lines, SHA-256
`b1a39b0869d03b50de9743df388c01665496ffebbb63bcb22b2efe908b196133`.

## Executive decision

The project accepts the central verdict `RUN_HYBRID_CAUSAL_PILOT_FIRST`.
It does not accept every sentence as already established evidence.

The earlier Hybrid result is strong descriptive motivation, not proof that
ROI and residual support are complementary. The old matrix mixed the
8/28/28 role split, two scorers, selected-only ST, detector-visible geometry
representation, and a single training seed. It also lacked valid matched
component arms. The correct response is therefore neither to abandon Hybrid
nor to optimize it directly from the old number. The response is a new,
support-only, Hybrid-centered causal study.

No further Pro discussion is required before this P0 and exploratory pilot.
Another discussion becomes useful only if P0 reveals a new contract failure,
or after all nine exploratory arms produce a sealed result that requires a
confirmatory freeze.

## What Fixed means

`Fixed K64` is the deterministic non-learned uniform lattice control produced
by `_deterministic_uniform_valid_indices()`. For every native 11x20 spatial
lattice at every one of 384 tubelets it selects the same row-major, uniformly
spread 64 valid positions. It does not inspect video content and is not claimed
to be optimal coverage or a lower bound on learned routing.

Fixed is useful because it holds source-native support, K, packed VideoMAE,
pretrained absolute position, Adapter, detector, and uniform pooling constant
while removing learned membership. Its role is to reveal whether learning a
route adds value beyond a stable spatial coverage anchor. The eight context
tokens in Hybrid are a smaller deterministic coverage floor; they do not
inherit every property of the full Fixed K64 route.

## Why ROI + TokenSelect was not scientifically abandoned

The old hierarchy required Free TokenSelect to pass before Hybrid could be
promoted. Free performed poorly, and the ROI-only cell failed to publish a
valid stage result, so the selector correctly refused promotion. That closed
the old Free-first decision path. It did not causally refute the Hybrid
mechanism.

Hybrid remained the strongest descriptive signal, especially at high tIoU,
but could not be interpreted because of the confounds above. The new study
therefore faces the signal directly. Free-ST is not rerun. The unstructured
control is residual-PL K64, and the decisive comparisons are Hybrid against
context+residual, context+ROI, matched Hybrid-ST, and temporally shifted ROI
geometry.

## Frozen method under test

The fixed-budget probe name is Structured Complementary Native Routing for TAD
(SCNR-TAD); `GeoRoute` remains the code-history namespace. It is not the final
paper architecture. Its purpose is to identify useful evidence roles under a
matched budget before designing the intended temporally adaptive continuous ROI
and token allocation. This probe does not resample source coordinates and must
not be called Online TAD.

Input and carrier:

- source input: uint8 `[B,1,3,768,180,320]`;
- scout input: uint8 `[B,1,3,768,96,96]`;
- native VideoMAE support: 384 tubelets, 11x20 = 220 patches per tubelet;
- exact selected budget: 64 patches per tubelet;
- detector-facing output: `[B,768,768]` after deterministic temporal 2x;
- one packed heavy VideoMAE forward, full temporal axis preserved.

The main route has fixed roles:

- 8 deterministic context tokens;
- 28 ROI tokens scored by a continuous in-bounds geometry trajectory;
- 28 residual tokens scored by a separate scout head;
- exact union 64, valid-only, no duplicates.

These fixed counts are an experimental intervention, not the final-method
requirement. The geometry head already predicts in-bounds `(cx,cy,w,h)` at all
384 tubelets, so ROI location and extent are temporally adaptive. What remains
absent is adaptive token quantity: the current probe fixes both total K64 and
the ROI/residual role quotas. The confirmed final objective requires dynamic
total `K_t` and dynamic role allocation; changing only the role split under
fixed K is not sufficient. The confirmed policy unit is the native two-frame
VideoMAE tubelet, hence 384 dynamic ROI/budget decisions for a 768-frame input.
The dynamic-budget route is staged: first enforce exact configurable
`sum_t K_t=B` per window and learn only redistribution; content-dependent
window-level B is deferred to a separately frozen successor.
The intended allocator also makes context count fully dynamic. It does not
retain the probe's deterministic context8 or introduce another fixed context
floor.

The learned hard policy is the explicit sequential conditional distribution

`p(ROI order | context) * p(residual order | context, complete ROI set)`.

Residual denominators exclude invalid tokens, context, all sampled ROI tokens,
and previously sampled residual tokens. The joint ordered log probability is
the sum of the two branch log probabilities. PL sampling uses independent
route-only generators keyed by study seed, successful-update index,
distributed rank, and role. It does not advance global CPU/CUDA RNG state and
replays the same route on an AMP retry.

The policy minimizes detector risk with

`L_det = cls_loss + reg_loss`

and positive score-function objective

`(stopgrad(L_det) - EMA_baseline) * mean_t(log p_joint,t)`.

The EMA momentum is 0.95, policy weight is 1, temperature is 0.7, and local
batch is exactly one. No auxiliary, geometry, policy, or unknown loss key may
enter the risk. Scout and likelihood arithmetic remain FP32; DDP uses default
FP32 reduction with `fp16_compress=false`.

## Representation and mechanism isolation

All nine arms retain pretrained VideoMAE absolute positional embeddings. All
disable external absolute source coordinates, ROI-relative coordinates,
geometry projection, and geometry side channels. Pooling is uniform-selected,
and geometry smoothness/area losses are zero. Geometry can affect only hard
membership.

The negative control rolls the geometry trajectory over the temporal tubelet
axis according to `pi(t)=(t+127) mod 384` before ROI logits are formed. It is not
an 11x20 spatial roll. It preserves the geometry-value multiset and role quota
while changing video-tubelet alignment.

## Frozen exploratory matrix

Study `georoute_hybrid_causal_pilot_v1`, result-blind seed 5227, one seed,
20 epochs, development-only, all-nine completion required:

1. Dense native 220.
2. Fixed lattice K64.
3. Stateless data-independent Random K64.
4. Residual sequential-PL K64.
5. Context8 + residual56 PL.
6. Context8 + ROI56 PL.
7. Hybrid context8/ROI28/residual28 selected-only ST.
8. Hybrid context8/ROI28/residual28 sequential PL.
9. The same Hybrid-PL with geometry temporal shift 127.

Only the preregistered descriptive contrasts may be emitted. Admission to a
separately frozen confirmatory study requires A7 to exceed both component arms,
Fixed, Random, and the shifted control on the registered high-IoU composite;
not fall below the simple controls on mAP@0.7; not fall below matched ST; and
have isolated model+postprocess p50 below Dense. These are screen conditions,
not effect-size claims or multiple-comparison-adjusted evidence.

## Qualified disagreements and corrections

1. Fixed does not literally prove a smaller combinatorial state space for the
   complete Hybrid policy. It supplies deterministic coverage and reduces the
   number of learned slots. That is the defensible claim.
2. A geometry-shift loss supports content alignment only if the router has not
   collapsed to a time-invariant trajectory. A tie is mechanism-ambiguous, not
   proof against all structured ROI support.
3. A single seed may generate a confirmatory protocol but cannot establish
   complementarity, estimator superiority, accuracy preservation, or a paper
   method. No fixed post-hoc pp margin is introduced.
4. If the eventual paper claims a PL-over-ST estimator choice, matched Hybrid-ST
   must remain in the multi-seed confirmatory matrix. The supplied eight-arm
   confirmatory proposal omitted it even though the exploratory admission rule
   uses it. The project therefore retains ST in any estimator-claiming
   confirmatory freeze, or explicitly drops the estimator-superiority claim.
5. Accuracy/telemetry inference and cost replay must be separate. Hashing routes
   and transferring diagnostic tensors to CPU inside the timed model forward
   would bias p50, especially Dense versus K64.

## Implementation and evidence status

Implemented in the current working tree:

- isolated structured routing schema and conditional ordered-PL sampler;
- exact context/ROI/residual role IDs and joint likelihood;
- update/rank/role-keyed private RNG;
- strict `{cls_loss, reg_loss}` detector-risk binding;
- support-only wrapper checks and temporal geometry-shift control;
- per-role span, adjacent Jaccard, lineage, branch conditional entropy,
  observed-path log probability, branch-gradient, RNG, and geometry hashes;
- frozen nine-arm config binder and all-complete contract;
- separate accuracy/telemetry and cost-profile replays;
- single structured-model P0 plus world-size-two DDP KAT;
- held Slurm deployments, storage/capacity gates, and all-terminal finalizers.

Current stage is `experiment_running`, not `empirically_supported`. Exact clean
runtime `0f64218d8f404ef652934844dcd97a3f9607c580` passed remote Linux/CUDA
pycompile, required C3 tests (`20 passed`), complete GeoRoute tests
(`171 passed, 1 skipped`), and the real data/config binder. No-performance P0
Jobs `1213665--1213667` completed `0:0` and sealed `PASS_MECHANICAL_ONLY`,
including exact roles, private role RNG, finite nonzero ROI/residual branch
gradients, and the world-size-two FP32-DDP KAT. The nine performance leaves are
Jobs `1213694--1213702`; all-terminal finalizer `1213703` remains responsible
for fail-closed interpretation. No complete checkpoint/prediction population,
metric contrast, empirical result, official-test result, or paper evidence
exists yet.

## Next executable sequence

1. Completed: clean remote Linux focused and complete GeoRoute tests.
2. Completed: no-performance P0 DAG.
3. Completed: structured A7 full-graph P0 and world-size-two FP32 DDP KAT sealed
   `PASS_MECHANICAL_ONLY`.
4. Completed: all nine held performance leaves and their after-any finalizer
   were released in one immutable namespace.
5. In progress: interpret nothing until the finalizer validates all nine results,
   artifact hashes, common population, exact-K, one-forward, and no-leak rules.
6. On a complete screen, either freeze a disjoint-seed confirmatory study or
   issue `HOLD_MECHANISM_AMBIGUOUS`; do not open official test or write a paper
   claim.
