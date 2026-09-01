---
doc_id: PRO_P0_IDENTITY_GATE_AUTHORIZATION
version: v001
stage: DRAFT
author_role: pro
intake_role: coordinator
status: accepted
scientific_decision: CONTINUE
evidence_class: BLOCKED_PRE_RESULT
project_id: g-p-6a796fef9a00819194024cf1de3bd697
project_url: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project
fixed_commit: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
turn_id: duca-p0-evidence-gate-20260812-d91f630e4c2a7b8f5e0d193a6c84f27b
nonce: d91f630e4c2a7b8f5e0d193a6c84f27b
submitted_at: 2026-08-12T18:26:34.563Z
completed_at: 2026-08-12T18:44:27Z
model: gpt-5.5-pro
effort: MAX_EFFORT_NOT_SEPARATELY_EXPOSED
sources:
  - CURRENT_RESEARCH_STATE-v005(1).md
  - MODEL_EXPERIMENT_HISTORY-v005.md
  - PRO_P0_ROUTE_ADJUDICATION-v002.md
  - PRO_P0_PROJECTION_POLICY-v001.md
  - CURRENT_RESEARCH_STATE-v007.md
  - MODEL_EXPERIMENT_HISTORY-v007.md
supersedes: PRO_P0_PROJECTION_POLICY-v001
raw_transcript: C:/Users/skywalker/.fastctx/jobs/j-1c18sd/output.log#L46-L363
---

# P0 cross-implementation identity/optimality gate — accepted intake

The fresh Project-bound Pro decision verified the exact Project, nonce, frozen
commit and required Sources, and reported no other-project material. Pro remains
the acting Scientific First-Author and Primary Research Owner. The sole decision
is `CONTINUE`: authorize one finite P0 projector identity/optimality evidence
gate. This is not a model, data, training, evaluation, latency, cost, or paper
result.

## Frozen scope and claim boundary

The gate starts after inverse-CDF conversion. Production and the independent
Evaluator reference receive identical canonical JSON bytes containing integer
`T`, `K`, `Q=2^20`, and full integer arrays `u` and `a`.

- `K=min(384,16*floor(T/16))`; `T<16` returns `INVALID_T_LT_16`.
- `u_j=floor((2j(T-1)+(K-1))/(2(K-1)))` for `0<=j<K`.
- `a_0=0` and `a_(K-1)=Q(T-1)`.
- A successful `p` has length `K`, endpoints `0,T-1`, strides in `{1,2,3,4}`
  and `|p_j-u_j|<=16` using checked integers.
- The unique result minimizes the exact ascending-candidate lexicographic key
  `(E2,E_infinity,E1,U1,p_1,...,p_(K-2))`.

No tolerance, clipping, deduplication, heuristic, uniform fallback, second
decoder, legacy selector, detector/data import, metric, accelerator, network,
or training environment is permitted. A pass establishes only bounded
cross-implementation conformance on the registered fixture domain; it does not
establish density quality, learnability, detector correctness, mAP, cost,
latency, P1 admission, or paper readiness.

## Independent reference requirement

Evaluator owns the reference projector. It may share only canonical fixture JSON
and this mathematical specification; it must import no production projector,
helper, objective, certificate, candidate generator, or selector module. It uses
exhaustive ascending enumeration for the `T<=385` witnesses and an independently
structured exact shortest-path/DP solver at `T=768`, all in integer arithmetic.

## Closed fixture matrix

`p^(-m)` means the ascending sequence omitting interior integer `m`; quarter-grid
vectors use `a_j=(Q/4)v_j`.

| ID | Frozen input / required witness |
| --- | --- |
| `G16-U` | `T=16,K=16,a=Q*u`; singleton `p=u=[0,...,15]`. |
| `G17-E2` | `T=17,K=16,a=Q*u`; all 15 feasible sequences; `E2` selects `p^(-8)`. |
| `G17-EINF` | `v=[0,1,6,17,18,23,24,28,36,37,42,52,53,54,57,64]`; `E2` tie `p^(-3),p^(-8)`, `E_infinity` selects `p^(-3)`. |
| `G17-E1` | `v=[0,7,15,16,18,19,21,35,36,37,43,49,50,52,55,64]`; scalar tie through `E_infinity`, `E1` selects `p^(-1)`. |
| `G17-U1` | `v=[0,5,7,17,18,19,29,30,39,40,44,53,54,55,60,64]`; scalar tie through `E1`, `U1` selects `p^(-6)`. |
| `G17-PLEX` | `v=[0,1,12,13,14,18,28,29,34,37,47,48,49,59,63,64]`; all scalar terms tie for `p^(-10),p^(-6)`, final vector lex selects `p^(-10)`. |
| `G31-U` | `T=31,K=16,a=Q*u`; non-unit-stride uniform optimum. |
| `G32-U` | `T=32,K=32,a=Q*u`; first next-`K` singleton. |
| `G383-U` | `T=383,K=368,a=Q*u`; last pre-cap band. |
| `G384-U` | `T=384,K=384,a=Q*u`; first capped singleton. |
| `G385-X` | `T=385,K=384`; `a_j=Qj` for `j<191`, `a_191=Q(191+3/4)`, `a_192=Q(192+1/4)`, `a_j=Q(j+1)` for `j>192`; exhaustive 383 candidates, scalar tie `p^(-191),p^(-193)`, vector lex selects `p^(-193)`. |
| `G767-U` | `T=767,K=384,a=Q*u`; canonical stride-two uniform. |
| `F768-U` | `T=768,K=384,a=Q*u`; exact uniform ending at 767. |
| `F768-PERIODIC` | gaps `g_i=1` for even `i<382`, `3` for odd `i<382`, `g_382=3`; cumulative `p*`, `a=Q*p*`. |
| `F768-DISP16` | canonical uniform gaps, add one at gaps `32..47`, subtract one at `300..315`; cumulative `p*`, `a=Q*p*`; reaches displacement 16. |
| `F768-CONVEX` | `a_j=floor((2Q*767*j^2+383^2)/(2*383^2))`; reference freezes the constraint-active winner. |
| `F768-CONCAVE` | `a_j=Q*767-a_CONVEX_(383-j)`; mirrored late-density witness. |
| `F768-ALT` | `b_j=floor((2Q*767*j+383)/(2*383))`; internal `a_j=b_j+Q/2` for even `j`, `b_j-Q/2` for odd `j`. |

No random fixtures, fuzzing, parameter sweep, or post-failure witness addition is
authorized under this decision.

Negative fixtures must fail with exactly: `N-T15` -> `INVALID_T_LT_16`; `N-K`
-> `K_EFF_MISMATCH`; `N-U-LEN` -> `U_LENGTH_MISMATCH`; `N-A-LEN` ->
`A_LENGTH_MISMATCH`; `N-U-CANON` -> `U_CANONICAL_MISMATCH`; `N-A-END` ->
`A_ENDPOINT_MISMATCH`; `N-A-ORDER` -> `A_ORDER_MISMATCH`; `N-INFEASIBLE`
(`T=1534,K=384,a=Q*u`) -> `INFEASIBLE`; and `N-ARITH` ->
`INTEGER_RANGE_OR_OVERFLOW`. Certificate mutations—duplicate position, stride
five, displacement 17, objective increment, scalar-tied losing vector, and
reversed candidate sequence—must be rejected as `CERTIFICATE_REJECTED` or
`CANDIDATE_ORDER_VIOLATION`.

## Success, failure, and role order

The global pass receipt must bind fixture bytes, production/reference identities,
non-importing-reference and zero-forbidden-access attestations, exact statuses,
`p`, feasibility, every objective component, ascending candidate order, complete
`T=17/T=385` witness counts and ties, independent `T=768` root optimum, all
certificate mutations, Critic `PASS`, and `scope_deviation="none"`.

At the first discrepancy, stop with `P0_IDENTITY_GATE_BLOCKED`, reporting the
fixture, phase, bytes, statuses, first differing field, feasibility/certificate
checks, candidate-order trace, reference independence and forbidden-access
status. There is no repair, rerun, third implementation, tolerance, fallback,
fixture deletion, or expected-output update under the same identity. A failure
only falsifies conformance of the current package, not the density-acquisition
hypothesis or any performance proposition.

1. Evaluator first authors and freezes the normative specification, matrix and
   independent reference package.
2. Builder freezes the already-authored production diff and output interface;
   after Evaluator expectations are frozen it may execute the production
   projector exactly once on the sealed matrix and return its receipt.
3. Evaluator then runs the independent exhaustive/full-scale comparison and
   sole machine-readable global receipt.
4. Critic acts only after complete Builder and Evaluator packages, returning
   `P0_IDENTITY_GATE_PASS` or `P0_IDENTITY_GATE_BLOCKED`.

The required bundle is exactly the eleven artifacts stated in the Pro response:
normative specification, fixture matrix, independent reference, Builder
production receipt, T17/T385/T768 receipts, negative/mutation receipt, Evaluator
global receipt, Critic closure, and unresolved-blockers receipt. No benchmark,
metric, checkpoint, cost, GPU/Slurm receipt, Git push, or paper claim may appear.
