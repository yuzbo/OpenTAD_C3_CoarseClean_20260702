I used the canonical repository at commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, whose commit record freezes the pre-backbone density design as designed but not implemented, together with `CURRENT_RESEARCH_STATE-v005(1).md` and `MODEL_EXPERIMENT_HISTORY-v005.md`. I also used the foundational v001 Sources, `PRO_INITIAL_REVIEW-v002.md`, `PRO_P0_BLOCKER_DECISION-v001.md`, and `EVALUATOR_DUCA_DENSITY_P0P1-v001.md`.    The attached DUCA source bundle and repository evidence contained no other-project scientific material. Unrelated project summaries were visible in the platform-level recent-conversation context; they were excluded entirely and are not evidence for this decision.

`DUCA-P0-ORACLE-FRESH-20260811::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::8b12d6f4a90c4b9aa602f1d9b3e75c18`

# PRO_P0_ROUTE_ADJUDICATION-v002

## CONTEXT_USED

**Coordinator task:** `019fa3db-42bf-7f30-a0ab-2b8171ab33ed`
**Project ID:** `g-p-6a796fef9a00819194024cf1de3bd697`
**Fresh turn:** `duca-p0-oracle-20260811-8b12d6f4a90c4b9a`
**Fresh nonce:** `8b12d6f4a90c4b9aa602f1d9b3e75c18`
**Code truth:** `63a726a4aaf48ecbf6780bb196de43a890c6b4df`

The accepted route is still fixed-requested-`K=384` physical acquisition before the heavy backbone, with an unchanged detector and raw selected-coordinate proposals transported into physical coordinates before unchanged NMS. The latest P0 evidence establishes that the uniform endpoint mismatch and post-NMS mapping are deterministic correctness defects, while the absent density tensor and inverse-CDF decoder are a scientific method ambiguity.  

The repository’s current selector computes deploy-visible dense descriptors, optionally coarsens them through `_candidate_grid`, invokes the multi-head reader, and then dispatches among legacy slot, top-k, rank, packet, or quota transport plans. The current reader exposes a shared temporal representation named `browser_memory`, followed by slot allocation and numerous task-specific heads. That shared representation is a valid attachment point for a new scalar density head; none of its existing output heads is itself the required density.

Evidence remains **`BLOCKED_PRE_RESULT`**. There is no admitted metric, cost, checkpoint, formal experiment, model-quality evidence, or paper result.  

## SCIENTIFIC_DECISION

**SCIENTIFIC_DECISION: REVISE**

Retain bounded monotone physical-density acquisition, but supersede the ambiguous v001 method definition with one new, explicitly named density-only interface and decoder. No existing slot allocation, `frame_selection_logits`, actionness score, soft transport matrix, top-k result, rank, or quota policy may be reinterpreted as the density.

This decision defines a falsifiable hard-forward mechanism. It does **not** admit a straight-through estimator, direct detector-gradient path, contribution teacher, training experiment, or model-quality evaluation.

## AUTHORITATIVE_MECHANISM

### Mechanism identity

**`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`**

Hard-decoder API identity:

```text
decode_duca_density_positions_v001(
    density_logits_valid: FloatTensor[T_v],
    requested_k: int = 384
) -> Int64Tensor[K_eff]
```

The API is applied independently per sample. It must not normalize across a batch or depend on batch size, ordering, or duplication.

### 1. Authoritative learned density input

The sole decoder input is a newly named tensor:

```text
duca_density_logits[b, t]
```

It contains one finite scalar for every valid physical candidate time `t ∈ {0,…,T_v−1}`.

Its insertion point is frozen as follows:

1. `PCOTMRASPreBackboneFrameSelector._select` constructs deploy-visible dense scout descriptors.
2. The primary density route uses `selection_unit=1`; therefore its candidate grid is the identity physical grid.
3. A dedicated density-only temporal reader forms the shared dense temporal representation corresponding to the existing `browser_memory` attachment point.
4. One learned scalar projection emits `duca_density_logits`.
5. The density decoder runs **before** `_sparse_transport_plan`, bypassing every legacy selection-strategy dispatch.
6. Only the returned hard integer positions gather heavy-backbone inputs. No soft weighted frame mixture or soft transport weights may enter the heavy backbone.

`browser_memory` is an attachment feature, not the density. The density-only primary configuration must not instantiate or consume slot queries, allocation matrices, role/process heads, actionness heads, boundary heads, top-k policies, rank policies, or dynamic-budget logic.

If the implementation cannot provide this dense identity-grid attachment without changing the detector or silently interpolating a coarse legacy policy, Builder must return a blocker rather than select another mechanism.

### 2. Positivity and continuous cumulative density

For finite valid-prefix logits (z_t),

[
\rho_t = 10^{-6} + \operatorname{softplus}(z_t).
]

The floor `1e-6` is fixed and non-tunable in this route.

Define positive density mass on each physical interval between adjacent dense candidate frames:

[
m_i=\frac{\rho_i+\rho_{i+1}}{2},
\qquad i=0,\ldots,T_v-2.
]

Normalize these interval masses and form the piecewise-linear cumulative function (F:[0,T_v-1]\rightarrow[0,1]), with (F(0)=0) and (F(T_v-1)=1). Normalization is strictly per sample and over the valid prefix only.

### 3. Endpoint-inclusive inverse-CDF rule

For

[
K_{\mathrm{eff}}
================

\min!\left(384,;16\left\lfloor T_v/16\right\rfloor\right),
]

`T_v < 16` fails closed.

Use endpoint-inclusive quantiles

[
\alpha_j=\frac{j}{K_{\mathrm{eff}}-1},
\qquad j=0,\ldots,K_{\mathrm{eff}}-1,
]

and continuous targets

[
x_j=F^{-1}(\alpha_j).
]

At a cumulative knot, inverse lookup chooses the lowest interval index whose upper cumulative mass is at least the quantile. The endpoints are exact:

[
x_0=0,\qquad x_{K_{\mathrm{eff}}-1}=T_v-1.
]

The earlier midpoint expression ((j+0.5)/K) is **replaced** for the primary route. It is incompatible with the accepted endpoint-inclusive uniform identity unless followed by an additional repair rule. The new endpoint quantiles make the density definition, endpoints, and constant-density limit one coherent mechanism.

### 4. Exact constant-density specialization

When all valid `duca_density_logits` are exactly equal under tensor equality—not approximately equal—the hard decoder directly invokes the single canonical uniform generator:

[
u_j=
\left\lfloor
\frac{2j(T_v-1)+(K_{\mathrm{eff}}-1)}
{2(K_{\mathrm{eff}}-1)}
\right\rfloor .
]

There is no tolerance-based constant test. Near-constant learned density follows the ordinary inverse-CDF and constrained-projection path.

At `T_v=768`, `K_eff=384`, the first and last positions are exactly `0` and `767`.

### 5. Integer conversion, collisions, and geometry

First form half-up integer targets:

[
r_j=\lfloor x_j+0.5\rfloor .
]

Do **not** sort, clip, deduplicate, greedily fill, or substitute uniform positions after collisions. Instead, return the globally projected sequence (p) from the feasible set:

[
\begin{aligned}
p_j &\in \mathbb Z,\
p_0&=0,\
p_{K_{\mathrm{eff}}-1}&=T_v-1,\
1 &\le p_{j+1}-p_j \le 4,\
|p_j-u_j|&\le16.
\end{aligned}
]

The upper adjacent-span bound of `4` means there may be at most **three unselected dense positions** between neighboring selected positions.

The deterministic projection minimizes, in order:

1. (\sum_j (p_j-r_j)^2);
2. among primary minimizers, (\sum_j (p_j-x_j)^2);
3. among remaining ties, the lexicographically smallest position vector.

The canonical uniform vector is always a feasible witness. Solver failure, an unresolved tie, non-finite input, an incorrect count, or any constraint violation fails closed. A hidden interpolated frame cannot count as one of the `K_eff` heavy frames.

No backward estimator is authorized here. The hard-forward API must be implemented and closed at P0 before a later decision may choose or test any learning surrogate.

## DETERMINISTIC_CORRECTIONS

Both corrections **may proceed without the learned-density wrapper**. They must be separable patch units with independent receipts, and they remain valid even if the learned-density route is later replaced or stopped.

### A. Shared canonical exact-uniform generator

**Allowed scope**

Implement one shared endpoint-inclusive integer-half-up generator using the accepted formula above. Replace the clean data-path generator and the wrapper-uniform generator with calls to that same implementation or immutable fixture identity.

The generator may also provide:

* the exact-constant density specialization;
* the reference vector (u) used by the learned decoder’s displacement bound.

It may not be used as a post-hoc filler or hidden fallback for nonconstant learned density.

**What it fixes**

It resolves the `766` versus `767` terminal-index disagreement and establishes one exact-uniform meaning across clean and wrapper paths. The current repository’s wrapper helper uses floating `linspace(...).round()` plus duplicate repair, which is not the accepted identity. 

**What it does not prove**

It does not prove clean/wrapper end-to-end parity, learned acquisition, gradient fidelity, accuracy, cost efficiency, detector invariance, or paper novelty.

**Claim effect**

Claim-neutral correctness correction. It changes no paper claim and creates no result evidence.

### B. Selected-q to physical-dense transport before NMS

**Allowed scope**

Add one coordinate-state-aware adapter at the entry to each per-sample `SingleStageDetector.post_processing` path:

```text
selected_q raw proposals
    -> exactly-once physical_dense mapping
    -> thresholding / top-k / IoU / NMS
```

For selected coordinates, segment endpoints use the end-exclusive domain `[0,K_eff]`. Mapping knots are:

```text
(q=j, t=p_j), j=0,…,K_eff−1
(q=K_eff, t=T_v)
```

The physical domain is `[0,T_v]`.

A clean physical-coordinate path is a strict no-op. Unknown state, missing knots, non-increasing knots, or double mapping fails closed. Scores, labels, head outputs, filtering thresholds, NMS callable and configuration, evaluator, split, and class map remain unchanged.

The current generic path performs filtering and local NMS before `convert_to_seconds`, while the selected-to-dense mapping presently resides inside that later conversion; therefore the current order is not admissible.

**What it fixes**

It ensures overlap suppression is computed in physical time rather than selected-rank geometry and gives both uniform and future learned wrappers the same coordinate contract.

**What it does not prove**

It does not prove wrapper parity, accuracy improvement, unchanged raw detector predictions, cost savings, or density-mechanism validity. Prior outputs produced under post-NMS mapping cannot be repaired or promoted retrospectively.

**Claim effect**

Claim-neutral correctness correction. It is a prerequisite for the paper contrast, not a contribution or positive result.

## FALSIFIABLE_CLAIM_AND_GATES

### Claim 1 — mechanism identity

The new route is genuinely a positive temporal-density, endpoint-inclusive inverse-CDF acquisition mechanism rather than a renamed legacy selector.

**Gate:** source-to-code trace plus synthetic property fixtures must establish the dedicated density tensor, dense identity grid, exact constant degeneration, deterministic exact `K_eff`, strict ordering, endpoints, adjacent span, displacement bound, and batch independence. They must also prove that no legacy slot, top-k, rank, quota, or soft transport output reaches the decoder.

**Cheapest falsifier:** one `T_v=768`, `K=384` constant-logit fixture that does not exactly equal the canonical vector ending at `767`, or one source path in which `duca_density_logits` aliases an existing selector signal. Either blocks the route immediately.

### Claim 2 — deterministic control semantics

After the two corrections, identical canonical indices can in principle enter clean and wrapper paths with one uniform identity and with physical proposals supplied to unchanged NMS.

**Gate:** a five-boundary sentinel parity harness must compare indices, gathered RGB, masks and mapped labels, raw proposals/scores/labels, physical proposals entering NMS, and final output. An order-sensitive synthetic pair must prove that the harness detects selected-axis versus physical-axis NMS.

**Cheapest falsifier:** any clean/wrapper index mismatch, or a call trace showing NMS receives `selected_q` coordinates. Either keeps P0 blocked.

These are mechanism and correctness claims only. No accuracy, efficiency, novelty, or publication claim is admitted.

## CODEX_DISPATCH

### Builder — `BUILDER_DUCA_P0_DENSITY_API_AND_CORRECTIONS-v001`

Use one clean isolated worktree at the exact commit. Produce one patch with three independently reviewable units:

1. shared canonical uniform utility and replacement of both existing generators;
2. coordinate-state-aware pre-NMS transport;
3. dedicated density-only reader, `duca_density_logits`, and `decode_duca_density_positions_v001`.

The density route must freeze `selection_unit=1`, bypass `_sparse_transport_plan`, and perform hard physical frame gathering only. Builder may author fixtures and tests but may not execute them under this decision.

Stop if the implementation requires a changed backbone interface, projection, assignment, detector head, detector losses, NMS, evaluator, split, or class map; if it requires aliasing an existing selector signal; or if the constrained decoder cannot satisfy its exact contract.

No dataset access, local or remote test execution, GPU initialization, Slurm, metrics, checkpoints, Git push, PR, result promotion, or paper edits are authorized.

### Critic — `CRITIC_DUCA_P0_DENSITY_CLOSURE-v001`

Begin only after receiving the complete Builder diff.

Perform a read-only audit of:

* the new density tensor’s provenance and sole-use property;
* the `browser_memory` attachment boundary;
* dense-grid identity and `selection_unit=1`;
* positivity, interval masses, endpoint quantiles, inverse-CDF ties;
* exact-constant specialization;
* constrained integer projection and deterministic tie-breaking;
* absence of post-hoc clipping, sorting, deduplication, or uniform filling;
* hard-only heavy-frame gathering;
* canonical-uniform single-source identity;
* exactly-once physical mapping before every NMS;
* unchanged detector and claim-neutral correction scope.

Return only `P0_STATIC_PASS` or `P0_BLOCKED`, with exact file/line and minimal counterexample for every blocker. No execution, mutation, metrics, or route substitution is allowed.

### Evaluator — `EVALUATOR_DUCA_P0P1_DENSITY_AMENDMENT-v002`

Amend the existing preregistration without executing it.

Freeze:

* endpoint quantiles `j/(K_eff−1)`;
* trapezoidal interval masses;
* exact-constant equality rule;
* half-up raw targets;
* constrained-projection objective and tie rules;
* adjacent span `≤4`;
* displacement `≤16`;
* dense identity-grid and batch-independence witnesses;
* canonical uniform and pre-NMS parity receipts;
* future remote CPU commands and schemas.

Keep P1 execution and every `PRE_RUN_READY` record blocked. P2 statistical questions and resource authorization remain outside this adjudication.

## NEXT_RETURN_CONTRACT

The next fresh scientific decision requires all of the following.

### Builder return

* `BUILDER_DUCA_P0_DENSITY_API_AND_CORRECTIONS-v001.md`
* `DUCA_P0_DENSITY_API_AND_CORRECTIONS-v001.patch`
* `DUCA_DENSITY_DECODER_SPEC-v001.json`
* `DUCA_CANONICAL_UNIFORM_SPEC-v001.json`
* `DUCA_P0_CHANGED_FILES-v001.txt`
* `DUCA_P0_TESTS_AUTHORED_NOT_RUN-v001.txt`
* `BUILDER_DUCA_P0_NO_EXECUTION_ATTESTATION-v001.json`
* SHA-256 manifest

The report must identify the first point where the new route diverges from legacy selector dispatch and prove by source trace that the detector is unchanged.

### Critic return

* `CRITIC_DUCA_P0_DENSITY_CLOSURE-v001.md`
* `CRITIC_DUCA_P0_DENSITY_FINDINGS-v001.json`
* one order-sensitive NMS counterexample
* final `P0_STATIC_PASS` or `P0_BLOCKED`
* SHA-256 manifest

### Evaluator return

* `EVALUATOR_DUCA_P0P1_DENSITY_AMENDMENT-v002.md`
* `DUCA_DENSITY_DECODER_FIXTURES-v001.json`
* `DUCA_P1_DENSITY_REMOTE_CPU_COMMANDS-v001.json`
* `DUCA_P1_DENSITY_RECEIPT_SCHEMA-v001.json`
* explicit `P1_STATUS: NOT_AUTHORIZED`
* SHA-256 manifest

### Joint reconciliation

It must state, without metrics, whether:

1. one and only one density tensor exists;
2. no legacy signal is aliased as density;
3. constant density exactly equals canonical uniform;
4. endpoint-inclusive quantiles are implemented;
5. integer projection is exact and deterministic;
6. geometry and batch-independence contracts hold by inspection;
7. clean and wrapper paths share one uniform generator;
8. selected proposals are mapped exactly once before every NMS;
9. detector semantics remain unchanged;
10. all Critic P0 findings are closed.

This adjudication authorizes no experiment or execution. Evidence remains `BLOCKED_PRE_RESULT`; P1, P2, GPU, Slurm, data traversal, metric access, result promotion, and paper claims remain blocked.
