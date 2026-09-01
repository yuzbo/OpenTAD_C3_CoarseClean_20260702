---
doc_id: PRO_P0_PROJECTION_POLICY
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
turn_id: duca-projection-policy-20260812-48a111ed75674967
conversation_id: 6a7bba83-4550-83ea-83db-ff6d03a255a1
nonce: 48a111ed756749671ffa8976364a9b52
submitted_at: 2026-08-12T00:12:47.192Z
completed_at: 2026-08-12T00:24:53.692Z
model_route: Pro / chatgpt-model-picker / already-selected
effort: MAX_EFFORT_NOT_SEPARATELY_EXPOSED
sources:
  - CURRENT_RESEARCH_STATE-v005(1).md
  - MODEL_EXPERIMENT_HISTORY-v005.md
supersedes: null
raw_transcript: .cvpr-pro-lab/pro-reviews/runs/duca-projection-policy-v001/raw-response.md
---

# P0 nonconstant projection policy — accepted intake

The raw Pro response named above is the authoritative, verbatim scientific
decision. Its Project, fixed commit, fresh turn, nonce, required Sources, and
absence of other-project material were all matched during intake. Pro remains
the acting Scientific First-Author and Primary Research Owner for scientific
route, experimental planning, claim scope, and paper narrative; human authors
retain legal authorship and final submission authority.

## Binding route decision

`CONTINUE` the already selected
`DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002` route only. The decision
freezes the missing nonconstant hard-decoder policy; it changes no detector,
loss, split, metric, budget, evaluator, NMS, checkpoint rule, or paper claim.

For a valid prefix, use `K_eff=min(384, 16*floor(T/16))`; `T<16` fails closed.
The integer-half-up endpoint-inclusive uniform vector remains the sole control.
From the serialized binary64 inverse-CDF targets, convert each target using
fixed point `Q=2^20` and exact half-up conversion. The selected sequence is the
unique feasible sequence minimizing the exact lexicographic key

`(E2, E_infinity, E1, U1, p1, ..., pK-2)`.

The feasible set fixes both endpoints, restricts each adjacent stride to
`{1,2,3,4}`, and bounds every displacement from uniform by 16. Production must
use an exact dynamic program or exact shortest-path equivalent, visit candidate
positions in ascending order, use checked integer arithmetic, and fail closed
on malformed inputs, infeasibility, overflow, comparison inconsistency,
certificate failure, or reference mismatch. No heuristic, clipping,
deduplication, tolerance, second decoder, legacy selector, or uniform fallback
is allowed for a nonconstant input.

## Evidence and claim boundary

The obligation is `CROSS_IMPLEMENTATION_IDENTITY_REQUIRED` for equal serialized
`(T, K, u, a)` projector inputs. It is not a cross-library softplus/inverse-CDF
bit-identity claim. Required later evidence is a normative JSON specification,
a non-importing reference projector, an exhaustive `T=385,K=384` witness,
full-scale `T=768,K=384` fixtures, exact objective/feasibility receipts, and a
Critic independence audit.

The active claim is solely uniqueness of the frozen constrained integer
projection for valid serialized inputs. This decision admits no result, metric,
data access, test execution, CPU/GPU/Slurm run, checkpoint, cost, performance
comparison, Git push, or paper claim. P0 remains `BLOCKED_PRE_RESULT` until a
separate Pro decision authorizes the bounded identity/optimality gate.

## Authorized next role work

Builder first returns `BUILDER_DUCA_P0_PROJECTION_PLAN-v001` mapping every
policy clause to an exact file/symbol and identifying conflicts before editing.
Critic acts only after the complete bounded diff. Evaluator prepares only
no-execution fixture and receipt schemas. Their returns must be brought back to
a fresh Pro decision before any execution authorization.
