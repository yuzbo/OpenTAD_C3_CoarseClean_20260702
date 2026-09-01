`DUCA-PROJECTION-POLICY-20260812::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::48a111ed756749671ffa8976364a9b52`

# PRO_P0_PROJECTION_POLICY-v001

## 1. CONTEXT_USED and routing assertion

I used the exact repository commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, verified as the commit freezing the DUCA total-60 pre-backbone design; the required Sources `CURRENT_RESEARCH_STATE-v005(1).md` and `MODEL_EXPERIMENT_HISTORY-v005.md`; the accepted `PRO_P0_ROUTE_ADJUDICATION-v002`; and the supplied fresh-turn manifest.    

* **Project ID:** `g-p-6a796fef9a00819194024cf1de3bd697`
* **Fresh turn ID:** `duca-projection-policy-20260812-48a111ed75674967`
* **Nonce:** `48a111ed756749671ffa8976364a9b52`
* **Coordinator task:** `019fa3db-42bf-7f30-a0ab-2b8171ab33ed`
* **Other-project material:** none was used in this adjudication.
* **Evidence state:** `BLOCKED_PRE_RESULT`. No result, metric, dataset traversal, test execution, checkpoint, cost, CPU/GPU/Slurm run, or paper claim is admitted.

The v005 Sources establish that the missing projection definition is a scientific ambiguity rather than an implementation defect. The accepted v002 route binds `duca_density_logits`, positive trapezoidal density, endpoint-inclusive inverse-CDF targets, and the geometric constraints, but intentionally leaves the nonconstant integer-projection objective and tie policy for this decision.   

---

## 2. SCIENTIFIC_DECISION

**SCIENTIFIC_DECISION: `CONTINUE`**

Continue only with `DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`, using the exact projection policy below.

This decision completes the method definition without changing its route, claim scope, fixed requested budget, detector, training protocol, dataset, split, evaluator, metric, NMS, checkpoint rule, or the separately frozen deterministic corrections. The projection is an **exact lexicographic Euclidean projection onto the bounded integer-feasible set**, with conservative, fully deterministic tie breaking.

P0 remains blocked until implementation fidelity and cross-implementation identity are returned. No execution is authorized here.

---

## 3. NONCONSTANT_INTEGER_PROJECTION_POLICY

### 3.1 Domain and canonical uniform reference

For one valid-prefix sample, let:

* (T=T_v), with physical candidate indices (0,\ldots,T-1);
* (K=K_{\mathrm{eff}}=\min(384,16\lfloor T/16\rfloor));
* (T<16) fail closed;
* this policy apply only after the exact-constant-logit specialization has been rejected.

The canonical uniform positions are

[
u_j=
\left\lfloor
\frac{2j(T-1)+(K-1)}{2(K-1)}
\right\rfloor,
\qquad j=0,\ldots,K-1.
]

This is integer half-up endpoint arithmetic. Floating `linspace`, banker rounding, tolerance repair, clipping, deduplication, and alternative uniform generators remain forbidden.

### 3.2 Continuous inverse-CDF targets

The accepted density route remains unchanged:

[
\rho_t=10^{-6}+\operatorname{softplus}(\ell_t),
\qquad
m_t=\frac{\rho_t+\rho_{t+1}}{2},
]

where (\ell_t=\texttt{duca_density_logits}[t]), all logits are finite, and (m_t>0).

Let

[
A_0=0,\qquad A_{r+1}=A_r+m_r,\qquad M=A_{T-1}.
]

For (j=0,\ldots,K-1), define (h_j=jM/(K-1)). The endpoint-inclusive inverse-CDF target (x_j) is:

* (x_j=r) when (h_j=A_r) exactly;
* otherwise, for the unique (r) satisfying (A_r<h_j<A_{r+1}),

[
x_j=r+\frac{h_j-A_r}{m_r}.
]

Thus (x_0=0), (x_{K-1}=T-1), and the internal targets are strictly increasing.

The projection receives the serialized IEEE-754 binary64 target vector (x). This decision does not require different softplus libraries to produce cross-platform-identical (x); the identity obligation begins from the identical serialized projection input defined next.

### 3.3 Exact fixed-point comparison representation

Set

[
Q=2^{20}=1{,}048{,}576.
]

For every internal target, convert the exact real represented by its binary64 bit pattern to

[
a_j=\left\lfloor Qx_j+\frac12\right\rfloor.
]

This is exact nonnegative half-up conversion performed from the binary64 sign, exponent, and significand—not the host language’s `round`. Its maximum target quantization error is less than (5\times10^{-7}) dense-frame units.

Set the endpoints exactly:

[
a_0=0,\qquad a_{K-1}=Q(T-1).
]

No tolerance or approximate equality is permitted after this conversion.

### 3.4 Exact feasible set

The admissible integer sequences are

[
\mathcal F(T,K)=
\left{
p\in\mathbb Z^K:
\begin{array}{l}
p_0=0,\quad p_{K-1}=T-1,\
p_{j+1}-p_j\in{1,2,3,4}\quad\forall j,\
|p_j-u_j|\le16\quad\forall j
\end{array}
\right}.
]

These constraints mean:

* exact count (K);
* integer and in-range positions;
* unique, strictly increasing positions;
* both physical endpoints included;
* at most three unselected valid candidates between adjacent selected positions;
* displacement at most 16 dense positions from canonical uniform.

The canonical uniform vector (u) must itself witness non-emptiness for every supported ((T,K)). Failure of that witness is a protocol error and fails closed.

### 3.5 Ordered objective

For each (p\in\mathcal F(T,K)), define, over internal indices (j=1,\ldots,K-2),

[
e_j(p)=Qp_j-a_j.
]

The candidate key is the ordered tuple

[
\Phi(p)=
\left(
E_2(p),;
E_\infty(p),;
E_1(p),;
U_1(p),;
p_1,\ldots,p_{K-2}
\right),
]

where

[
E_2(p)=\sum_{j=1}^{K-2} e_j(p)^2,
]

[
E_\infty(p)=\max_{1\le j\le K-2}|e_j(p)|,
]

[
E_1(p)=\sum_{j=1}^{K-2}|e_j(p)|,
]

[
U_1(p)=\sum_{j=1}^{K-2}|p_j-u_j|.
]

The output is

[
p^\star=\operatorname*{arg,min}_{p\in\mathcal F(T,K)}
\Phi(p)
]

under exact lexicographic tuple comparison.

The terms have the following binding order and meaning:

1. **(E_2), primary:** the actual constrained Euclidean projection; it preserves the inverse-CDF targets globally.
2. **(E_\infty), first tie term:** among equal global projections, minimize the largest local quantile miss.
3. **(E_1), second tie term:** among equal squared and worst-case fits, minimize total unsigned target displacement.
4. **(U_1), conservative tie term:** among density-equivalent candidates, choose the sequence closest to the exact-uniform control.
5. **Position-vector lexicography:** at the first differing internal index, the smaller physical position wins. This is a convention only and carries no performance interpretation.

No weighted sum, learned coefficient, stochastic choice, tolerance, approximate solver, or post-hoc repair may replace this ordering.

### 3.6 Deterministic candidate ordering

The production implementation must use an exact dynamic program or equivalent exact shortest-path solver. Greedy rounding, clipping, deduplication, local exchanges, heuristic fallback, and approximate optimization are nonconforming.

At internal index (j), given predecessor (r=p_{j-1}), define

[
L_j(r)=
\max\left(
0,,
r+1,,
u_j-16,,
T-1-4(K-1-j)
\right),
]

[
R_j(r)=
\min\left(
T-1,,
r+4,,
u_j+16,,
T-1-(K-1-j)
\right).
]

Candidate integers are visited in strictly ascending order

[
L_j(r),L_j(r)+1,\ldots,R_j(r).
]

A state with (L_j(r)>R_j(r)) is infeasible. States are processed by increasing (j), then increasing current physical position. Incumbents are replaced only when the complete exact comparison key is strictly smaller.

All objective arithmetic must use arbitrary-precision integers or checked signed arithmetic of at least 128 bits. Overflow, saturation, floating accumulation, unordered parallel reduction, or comparator inconsistency fails closed.

### 3.7 Tie resolution and mandatory fail-closed behavior

The appended position vector makes (\Phi) a total order: two distinct feasible sequences cannot remain tied mathematically.

The decoder must fail closed if any of the following occurs:

* a logit, target, mass, cumulative value, or fixed-point target is non-finite or outside its declared domain;
* the canonical uniform feasibility witness fails;
* the feasible set is empty;
* checked integer arithmetic overflows;
* no exact minimizer is returned;
* two distinct candidates are reported as equal after the full tuple comparison;
* a returned sequence violates any constraint;
* an objective certificate cannot be recomputed exactly;
* production and the accepted independent reference disagree once the identity gate is executed.

For a nonconstant input, failure must raise a typed projection error and invalidate the sample/run identity. It must **not** silently return uniform positions, invoke a legacy selector, loosen a bound, clip, deduplicate, retry with another solver, or apply a second decoder.

---

## 4. EVIDENCE_OBLIGATION

### Decision

**`CROSS_IMPLEMENTATION_IDENTITY_REQUIRED`**

Properties and same-process repeatability alone are insufficient. Two implementations could satisfy count, endpoints, order, span, and displacement while selecting different physical frames. Because the discrete frame sequence is the scientific method output, identical serialized projection inputs must yield an identical integer vector.

The identity requirement is scoped to the integer projector receiving the same serialized tuple

```text
(T, K, canonical_uniform_u, fixed_point_targets_a)
```

It does not, in this turn, require cross-library bit identity for softplus or inverse-CDF floating arithmetic.

### Minimum discriminating evidence

The later authorized evidence package must contain:

1. **Normative specification:**
   `DUCA_P0_NONCONSTANT_PROJECTION_SPEC-v001.json`, encoding (Q), feasible constraints, objective order, candidate order, integer widths, and fail conditions.

2. **Independent reference:**
   `DUCA_P0_PROJECTION_REFERENCE-v001.py`, which may share the serialized specification and fixtures but must not import the production solver, comparator, candidate generator, or backpointer code. It is a verification oracle, not a second runtime decoder.

3. **Exact exhaustive witness:**
   At supported (T=385,K=384), exhaustively enumerate the feasible sequences for nonconstant target fixtures, including at least one exact tie fixture. Verify that both implementations return the unique smallest (\Phi).

4. **Full-scale identity fixtures:**
   At (T=768,K=384), include:

   * smooth monotone nonconstant density;
   * alternating density;
   * single-boundary and dual-boundary concentration;
   * a constraint-saturating density;
   * a fixed-point half-up tie case.

5. **Exact receipts:**
   Both implementations must emit identical (p^\star), identical objective terms, and an independently recomputed feasibility certificate. One mismatch is `P0_BLOCKED`.

6. **Critic independence audit:**
   Verify that the two implementations do not share the solver/comparator logic and that no fallback or legacy selector path exists.

No broad SHA-tree manifest is required. Each receipt must bind the exact commit, patch identity, fixture identity, command, environment, exit status, and deviation field.

### What this evidence cannot establish

Even a complete identity pass cannot establish:

* useful learning of `duca_density_logits`;
* correctness or portability of the density reader or floating inverse-CDF implementation beyond the frozen fixtures;
* gradient validity or trainability;
* accuracy, high-IoU localization, efficiency, latency, or novelty;
* detector invariance or pre-NMS transport closure;
* superiority over uniform sampling or AdaTAD;
* readiness for P1, P2, training, or formal evaluation.

Infrastructure correctness remains non-empirical evidence. 

---

## 5. FALSIFIABLE_CLAIM_AND_ANTI_CLAIM

**Active claim:** For every supported ((T,K)) and valid serialized nonconstant inverse-CDF target vector, this policy produces the unique feasible integer sequence minimizing the frozen exact comparison tuple, independently of conforming implementation.

**Anti-claim:** A correct projection does not imply that the learned density is useful, that DUCA improves detection or cost, that the route is trainable, or that any paper claim is supported.

**Cheapest decisive falsifier:** Use one nonconstant (T=385,K=384) fixture whose feasible set can be exhaustively enumerated. If exhaustive enumeration finds a sequence with a smaller (\Phi) than the production output—or the independent reference returns a different sequence—the policy implementation is falsified and P0 remains blocked.

---

## 6. CODEX_DISPATCH

All work below begins only after this decision is accepted into intake. **Tests, data access, local or remote execution, CPU/GPU initialization, Slurm, metrics, and experiments remain not authorized.**

### Builder — bounded projection implementation

1. First return `BUILDER_DUCA_P0_PROJECTION_PLAN-v001`, mapping every policy clause to an exact file/symbol and identifying any conflict before editing.
2. After literal conformance review, produce one bounded patch containing:

   * one production exact projector;
   * checked exact comparator and objective certificate;
   * fixed-point conversion;
   * typed fail-closed errors;
   * an independent test-only reference oracle;
   * authored-but-not-run fixtures and tests.
3. Do not introduce a second runtime decoder, alter the density reader, reinterpret legacy selector outputs, or modify detector/NMS/evaluator behavior.
4. No unrelated refactor, optimization, GPU kernel, gradient path, or fallback is permitted.

### Critic — static closure after complete diff

Return only `P0_STATIC_PASS` or `P0_BLOCKED`. Audit:

* exact transcription of the feasible set and objective order;
* independence of the reference oracle;
* integer overflow and comparison semantics;
* candidate reachability bounds;
* absence of greedy/clip/dedup/uniform fallback;
* one minimal tie counterexample;
* preservation of the route and claim boundaries.

No execution is permitted.

### Evaluator — no-execution protocol amendment

Prepare:

* the canonical fixture schema;
* exact identity and objective-certificate receipt schema;
* the exhaustive (T=385,K=384) oracle contract;
* the full-scale fixture matrix;
* future command/environment fields;
* explicit `not_authorized` status for every execution field.

The two prior deterministic corrections remain governed by `PRO_P0_ROUTE_ADJUDICATION-v002`; this decision neither expands nor revisits their scope.

---

## 7. NEXT_RETURN_CONTRACT

Before the next fresh Pro decision, the coordinator must return:

1. `BUILDER_DUCA_P0_PROJECTION_PLAN-v001`
2. The complete bounded projection patch and changed-file list
3. `DUCA_P0_NONCONSTANT_PROJECTION_SPEC-v001.json`
4. `DUCA_P0_PROJECTION_REFERENCE-v001.py`
5. `DUCA_P0_PROJECTION_FIXTURES-v001.json`, clearly marked `AUTHORED_NOT_RUN`
6. Builder no-test/no-data/no-CPU-GPU/no-remote attestation
7. `CRITIC_DUCA_P0_PROJECTION_CLOSURE-v001.md`
8. `EVALUATOR_DUCA_P0_PROJECTION_PROTOCOL-v001.md`
9. `DUCA_P0_PROJECTION_IDENTITY_RECEIPT_SCHEMA-v001.json`
10. A consolidated unresolved-blocker record stating whether the patch is ready for a separate bounded execution-authorization decision.

The package must contain no executed test receipt, dataset access, metric, checkpoint, cost, result comparison, Git push, route expansion, or claim promotion.

The next fresh Pro decision may decide only whether to authorize the bounded CPU identity/optimality gate. Actual cross-implementation identity receipts are mandatory before P0 closure, but cannot be generated under this decision.

`DUCA-PROJECTION-POLICY-20260812::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::48a111ed756749671ffa8976364a9b52`
