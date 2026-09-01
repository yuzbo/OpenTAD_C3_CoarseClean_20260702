# Raw Pro response — DUCA P0 blocker decision

Captured verbatim from the fresh DUCA Project conversation after generation completed. No editorial correction has been applied.

```text
PROJECT_ID=g-p-6a796fef9a00819194024cf1de3bd697; TURN_ID=duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd; NONCE=51c88fd75537120ce96a417beb7e81dd; GIT_COMMIT=63a726a4aaf48ecbf6780bb196de43a890c6b4df; CURRENT_STATE=CURRENT_RESEARCH_STATE-v002; HISTORY=MODEL_EXPERIMENT_HISTORY-v002

PRO_P0_BLOCKER_DECISION-v001
1. SESSION_ASSERTION

Conversation status: fresh DUCA Project decision turn.

Project ID: g-p-6a796fef9a00819194024cf1de3bd697

Project URL: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project

Turn ID: duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd

Nonce: 51c88fd75537120ce96a417beb7e81dd

Canonical repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

Canonical Git revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df

Parent decision: PRO_INITIAL_REVIEW-v002

Parent artifact: art-20260811T045704Z-2d13c25e41

New evidence: EVALUATOR_DUCA_DENSITY_P0P1-v001

Evidence level: preparatory_preregistration_no_execution

Other Project material seen: no.

Builder/Critic partial state: not admitted. No statement, unsealed file, or partial worktree from those timed-out tasks is used as evidence.

The repository commit was verified read-only. Its two P0 defects agree with the Evaluator’s static findings: the data path and selector path use different uniform constructions, and generic non-sliding post-processing performs NMS before the selected-to-physical inverse map. 

EVALUATOR_DUCA_DENSITY_P0P1-v001

2. MODEL_EFFORT_ASSERTION

Actual selected model/tier: GPT-5.6 Pro.

Available browser thinking-effort observation: no trustworthy live browser effort-selector label is exposed to this response. I therefore do not assert Pro, 5 of 5, Effort Pro, or any equivalent browser-visible value.

Any control-plane acceptance rule requiring a particular live effort label must rely on a contemporaneous browser receipt for this exact Project, turn, and nonce rather than an invented observation.

3. ROLE_ACKNOWLEDGMENT

I accept the role of acting Scientific First-Author Agent and Primary Research Owner for the DUCA-RIME method route, experimental ordering, falsification criteria, claim scope, and publication logic.

Codex remains coordinator and evidence transport only. Builder, Critic, and Evaluator may execute only the bounded queues below. The human remains the legal and accountable author, access and spending authority, remote-execution approver, result-promotion authority, and final submission approver.

4. CONTEXT_USED
Project Sources actually used

PROJECT_CHARTER-v001

LITERATURE_AND_GAP-v001

ROUTE_DECISION-v001

EXPERIMENT_PLAN-v001

IMPLEMENTATION_STATUS-v001

RESULTS-v001

RESULT_ANALYSIS-v001

FAILURES_AND_PIVOTS-v001

CLAIM_MAP-v001

PAPER_DRAFT-v001 

PROJECT_CHARTER-v001
 

LITERATURE_AND_GAP-v001
 

ROUTE_DECISION-v001
 

EXPERIMENT_PLAN-v001
 

IMPLEMENTATION_STATUS-v001
 

RESULTS-v001
 

RESULT_ANALYSIS-v001
 

FAILURES_AND_PIVOTS-v001
 

CLAIM_MAP-v001
 

PAPER_DRAFT-v001

CURRENT_RESEARCH_STATE-v002

MODEL_EXPERIMENT_HISTORY-v002

PRO_INITIAL_REVIEW-v002

EVALUATOR_DUCA_DENSITY_P0P1-v001 

CURRENT_RESEARCH_STATE-v002
 

MODEL_EXPERIMENT_HISTORY-v002
 

PRO_INITIAL_REVIEW-v002
 

EVALUATOR_DUCA_DENSITY_P0P1-v001

Canonical repository surfaces actually inspected

opentad/datasets/transforms/end_to_end.py

opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py

opentad/models/detectors/single_stage.py

opentad/models/utils/post_processing/utils.py

No local checkout, dirty coordinator tree, unpushed patch, GPU output, data sample, prediction, checkpoint, metric, or held-out artifact was used.

5. HISTORY_SYNTHESIS
5.1 Accepted route

The accepted PRO_INITIAL_REVIEW-v002 route is:

DUCA_FIXEDK_BOUNDED_MONOTONE_DENSITY_ACQUISITION-v001

Its scientific identity is:

fixed requested K=384;

physical-frame acquisition before the heavy backbone;

deploy-visible low-cost scout;

positive temporal density and monotone inverse-CDF acquisition;

exact, unique, increasing positions;

bounded coverage and coordinate displacement;

clean exact-uniform control;

external physical-to-selected training-label mapping;

external selected-to-physical raw-proposal mapping before NMS;

unchanged detector architecture, assignment, head, losses, NMS algorithm, and evaluator;

no dynamic K and no dynamic-budget claim.

That route remains the active candidate. The v002 state explicitly excludes dynamic K, quota-policy stacks, physical-grid-head modifications, and detector changes from the primary route. 

CURRENT_RESEARCH_STATE-v002
 

PRO_INITIAL_REVIEW-v002

5.2 Historical routes

The following remain historical engineering or negative evidence and are not revived:

local-cell selection;

naive actionness top-k;

selected-rank decoding;

hard-K query deletion and SparseHead;

DCSR-G1 and ODF-CR-G2;

PC-OT-MRAS quota stacks and auxiliary-head proliferation;

heuristic dynamic-budget thresholds;

detector-internal physical-grid heads;

failed or unmatched-budget runs;

partial, subset, over-budget, unsealed, proxy, or failed-root metrics.

Those routes helped expose physical-coordinate and attribution failures, but they do not define the accepted method and cannot be used as evidence for it. 

FAILURES_AND_PIVOTS-v001
 

MODEL_EXPERIMENT_HISTORY-v002

5.3 Status of the new evidence

EVALUATOR_DUCA_DENSITY_P0P1-v001 is an unexecuted preregistration and static source inspection. It establishes no implementation pass, model behavior, loss behavior, performance, cost, or paper claim. It identifies two real, fail-closed P0 blockers:

For normal T=768, K=384, the data generator ends at physical index 766, while the selector’s endpoint-inclusive generator ends at 767.

The generic non-sliding detector path calls batched NMS before the existing selected-to-dense inverse map.

The accepted route requires endpoint preservation and pre-NMS mapping, so neither current behavior is admissible. 

EVALUATOR_DUCA_DENSITY_P0P1-v001

The evidence state therefore remains BLOCKED_PRE_RESULT. There is still no formal DUCA result, cost receipt, or paper-ready claim. 

RESULTS-v001
 

RESULT_ANALYSIS-v001
 

CLAIM_MAP-v001

6. SCIENTIFIC_DECISION

SCIENTIFIC_DECISION: REVISE

This is a narrow P0 contract revision, not a change of scientific route.

Permitted

This decision permits only:

a new isolated worktree based exactly on 63a726a4aaf48ecbf6780bb196de43a890c6b4df;

implementation of one shared canonical uniform generator;

replacement of the two conflicting uniform implementations with calls to that generator;

implementation of one detector-agnostic raw-proposal coordinate adapter before all NMS;

removal of selected-axis inverse mapping from the post-NMS seconds-conversion path;

clean uniform and density-wrapper configuration reconciliation;

preparation of focused synthetic tests, fixtures, and audit receipts;

static Critic review and Evaluator protocol amendment.

Forbidden

This decision forbids:

resuming, cherry-picking, or trusting the timed-out Builder/Critic worktree;

GPU initialization, GPU training, or GPU evaluation;

local or remote test execution under the present authorization;

Slurm submission or any paid/remote experiment;

dataset traversal, official validation/test access, or metric computation;

dynamic K or budget-controller work;

detector-head, assignment, classification-loss, regression-loss, prior-generator, NMS-algorithm, evaluator, class-map, or split changes;

post-evaluation deduplication;

tolerance-based repair of uniform mismatch;

reusing a baseline trained with the old 766 endpoint contract;

Git push, branch publication, PR creation, result promotion, or paper claim expansion.

The P0 defects must be repaired as external acquisition/transport semantics. They must not be “fixed” by altering detector behavior.

7. P0 RESOLUTION A — CANONICAL UNIFORM POSITIONS
7.1 Domains and effective budget

Let:

T_v be the number of prefix-valid dense candidate positions before padding;

valid frame indices be the integers 0, …, T_v−1;

requested budget be K_req=384;

backbone temporal quantum be Q=16.

A valid mask must be prefix-contiguous. Non-prefix masks fail closed.

The effective budget is:

K_eff=min(384,16 floor(T_v / 16)).

Rules:

T_v < 16 fails closed.

K_eff is used identically by uniform and learned arms.

requested_K, effective_K, unique_K, backbone_input_K, and padded_K are separate ledger fields.

No arm may process 384 heavy slots while reporting a smaller K_eff.

Any unavoidable padding is recorded and charged as actual backbone work.

7.2 Canonical endpoint-inclusive integer formula

For K_eff > 1, the canonical uniform positions are:

u_j = floor((2j(T_v−1)+(K_eff−1))/(2(K_eff−1))), j=0,…,K_eff−1.

Equivalent interpretation:

u_j = round_half_up(j(T_v−1)/(K_eff−1)).

The tie rule is round half toward the larger physical index. Python banker’s round, floating linspace, device-dependent rounding, and approximate equality are prohibited.

A reference integer implementation is:

Python
Run
numerator = 2 * j * (valid_len - 1) + (effective_k - 1)
denominator = 2 * (effective_k - 1)
u_j = numerator // denominator

Consequences:

u_0 = 0;

u_{K_eff−1} = T_v−1;

positions are strictly increasing because T_v ≥ K_eff;

no clipping, deduplication, sorting repair, or fallback is required;

for T_v=768, K_eff=384, the terminal position is 767, not 766.

A generic K_eff=1 helper may return (0,), but that case is outside the admitted route because T_v<16 already fails and all admitted K_eff values are multiples of 16.

7.3 Shared API

Create one dependency-neutral module, for example:

opentad/utils/temporal_sampling.py

with exactly one source of truth:

Python
Run
def duca_effective_k(
    valid_len: int,
    requested_k: int = 384,
    quantum: int = 16,
) -> int:
    ...

def duca_canonical_uniform_positions(
    valid_len: int,
    effective_k: int,
) -> tuple[int, ...]:
    ...

Requirements:

the generator itself uses integer arithmetic only;

it has no Torch/CUDA dependency;

the data transform and selector wrapper both call this function;

consumers convert the returned tuple to numpy.int64 or torch.long without recomputation;

the function source hash and serialized fixtures are bound in DUCA_CANONICAL_UNIFORM_SPEC-v001.json;

the old data and selector formulas cease to be executable sources of truth.

The pinned data implementation currently follows a round(j*T/K) construction, while the selector follows rounded endpoint-inclusive linspace; the new shared function supersedes both.

7.4 Density degeneration rule

The continuous acquisition coordinate is x∈[0,T_v−1]. For strictly positive density ρ(x),

F(x)= integral_0^x rho(s) ds / integral_0^(T_v−1) rho(s) ds.

The hard decoder uses endpoint quantiles:

r_j=j/(K_eff−1), x_j=F^-1(r_j).

Midpoint quantiles such as (j+0.5)/K are not the final hard-position convention for this route because they do not naturally preserve both physical endpoints.

For constant density:

x_j=j(T_v−1)/(K_eff−1),

and the hard integer projection must produce exactly the canonical u_j above.

Constant density must be bit-identical to canonical uniform. This requirement covers shape, integer dtype, values, order, serialization, CPU/GPU placement, and batch composition.

To avoid floating-point drift, an exactly constant valid density/logit vector must use the integer canonical specialization in the hard forward path. The equality check is exact, not tolerance-based. Near-constant learned density must not trigger this specialization. This is a disclosed exact implementation of the mathematical constant-density case, not a hidden uniform scaffold or repair path.

7.5 Geometry terminology resolved

The frozen maximum-gap rule is:

max_j(p_{j+1}−p_j−1)≤3,

equivalently:

p_{j+1}−p_j≤4.

Thus “three” refers to the maximum number of unselected valid positions between adjacent selected positions, not an adjacent-span bound of three.

The maximum displacement rule remains:

max_j |p_j−u_j|≤16.

7.6 Required canonical fixtures

The serialized fixture must include at least:

T_v ∈ {16, 17, 31, 32, 383, 384, 385, 767, 768}

For every case it records:

requested_K;

effective_K;

full canonical integer vector;

first and last positions;

uniqueness and monotonicity;

maximum adjacent span;

fixture SHA-256.

Any difference between data path, wrapper path, constant-density path, or serialized fixture is a P0 failure with zero tolerance.

8. P0 RESOLUTION B — DETECTOR-AGNOSTIC PRE-NMS TRANSPORT
8.1 Raw representation

The canonical external adapter consumes detector-native raw proposals:

Python
Run
segments_q: Tensor[N, 2]   # selected-axis boundaries
scores:     Tensor[N]      # unchanged
labels:     Tensor[N]      # unchanged

with an explicit coordinate-state tag:

coordinate_space = "selected_q"

The segment domain is end-exclusive:

q∈[0,K_eff].

Frame indices remain in [0,T_v−1], while temporal segment boundaries use the end-exclusive physical domain:

t∈[0,T_v].

The selected-to-physical knots are:

(q=j, t=p_j), j=0,…,K_eff−1,

plus the terminal boundary knot:

(q=K_eff, t=T_v).

This resolves the Evaluator’s coordinate-domain question.

8.2 Mapping

For q∈[j,j+1], linear interpolation between adjacent knots gives:

t(q)=(1−alpha)t_j+alpha t_{j+1}, alpha=q−j.

For the terminal interval, t_K_eff=T_v.

The inverse training-label map t→q uses the same strictly increasing knots. Exact knots map exactly; values between knots use piecewise-linear interpolation.

Properties:

order preserving;

injective because selected positions are strictly increasing;

labels, scores, and proposal order unchanged;

no dependence on GT during inference;

no use of scores or labels in coordinate calculation;

no batch-dependent state.

8.3 Canonical API and hook

Add one generic utility, for example:

opentad/models/utils/post_processing/coordinate_transport.py

with an API equivalent to:

Python
Run
def transport_raw_segments_to_physical_dense(
    segments: torch.Tensor,
    *,
    selected_positions: torch.Tensor,
    valid_len: int,
    coordinate_space: str,
) -> tuple[torch.Tensor, str]:
    """
    Input:
        segments: [N, 2], finite selected-axis or physical-dense boundaries.
        coordinate_space: "selected_q" or "physical_dense".
    Output:
        mapped segments and coordinate_space == "physical_dense".
    """

Contract:

"selected_q" requires valid, strictly increasing positions and performs exactly one map;

"physical_dense" is an identity only for genuinely clean physical proposals;

an unknown tag fails;

a second attempted selected-to-physical map fails rather than silently remapping;

duplicated, unsorted, out-of-range, or count-mismatched selected positions fail before proposal processing.

For this repository, the canonical call site is the entry of per-sample SingleStageDetector.post_processing, immediately after raw head outputs are available and before any score filtering, top-k, clipping, IoU computation, voting, local NMS, or serialization.

Mapping before score-only filtering is stricter than mathematically necessary, but it provides one simple auditable rule:

No post-head operation may observe selected-axis proposal coordinates.

The current generic path performs batched_nms and only later calls convert_to_seconds, where selected-axis inverse mapping currently occurs. That order must be reversed.

8.4 Preservation of detector and NMS

The only permitted post-processing changes are:

insert the external coordinate adapter before current post-processing;

remove selected-axis inverse mapping from convert_to_seconds;

leave convert_to_seconds responsible only for dense-window offset, snippet stride, FPS, and duration conversion.

The following must remain hash-identical between clean uniform and density-wrapper routes:

detector backbone interface;

projection;

point/prior generator;

target assignment;

detection head;

classification and regression losses;

proposal scores and labels;

NMS callable;

NMS configuration;

evaluator.

NMS must receive physical coordinates but otherwise run unchanged.

8.5 Sliding-window and global NMS

For every window:

raw selected-axis proposals are produced;

selected-axis boundaries are mapped to dense physical window coordinates;

window offsets and FPS convert them to video time;

any local suppression, if present in the clean route, operates only after mapping;

global/sliding-window merge NMS receives physical video-time coordinates.

No selected-axis proposal may cross a window merge boundary.

8.6 Duplicate-coordinate leakage safeguards

Selected positions must already be unique; there is no post-hoc deduplication.

The strictly monotone map cannot merge distinct coordinates.

Identical raw proposals remain identical; they are handled only by unchanged NMS.

The adapter may not combine proposals, average scores, change labels, sort by geometry, or use GT.

No proposal cache, counterfactual ledger, teacher output, or evaluator result may enter the map.

A coordinate-state tag and call trace must prove that mapping happened exactly once.

9. SMALLEST THREE-ROLE QUEUES

No extra role, subagent, probe, or parallel implementation is admitted.

9.1 Builder

Queue: BUILDER_DUCA_P0_CANONICAL_TRANSPORT-v001

Hard bound: one fresh isolated worktree, one patch attempt, at most 6 elapsed task-hours and at most 10 changed production/test files. Stop at the first scope overflow or scientific invariant conflict.

Work:

implement the shared integer uniform generator;

replace both conflicting call sites;

implement exact constant-density degeneration;

implement the coordinate-state adapter;

place it before every NMS in the admitted path;

remove selected-axis mapping from post-NMS seconds conversion;

prepare canonical fixture, parity, order-sensitive NMS, short-window, and double-map tests;

prepare resolved clean-uniform versus density-wrapper config diff.

No execution is authorized. Builder writes tests and patch artifacts only.

Required return:

DUCA_P0_CANONICAL_TRANSPORT-v001.patch

DUCA_P0_CHANGED_FILES-v001.txt

DUCA_CANONICAL_UNIFORM_SPEC-v001.json

DUCA_P0_RESOLVED_CONFIG_DIFF-v001.json

DUCA_P0_TEST_PLAN-v001.md

patch SHA-256 and no-execution attestation.

Stop conditions:

a detector/head/loss/NMS/evaluator change is required;

canonical constant-density identity requires a tolerance;

positions require post-hoc deduplication;

the adapter cannot precede all NMS;

more than the bounded patch scope is needed.

9.2 Critic

Queue: CRITIC_DUCA_P0_CLOSURE-v001

Hard bound: read-only review, at most 3 elapsed task-hours, one pass over the complete Builder diff plus one counterexample file.

Scope only:

canonical formula and tie rule;

short-window K_eff;

constant-density identity;

resolved detector/config equality;

pre-NMS order in non-sliding and sliding paths;

coordinate-state and double-map failure;

duplicate-position and hidden-padding failure;

five-boundary uniform-wrapper parity specification.

No broad new architecture review and no model-quality interpretation.

Required return:

CRITIC_DUCA_P0_CLOSURE-v001.md

CRITIC_DUCA_P0_FINDINGS-v001.json

one order-sensitive synthetic NMS counterexample;

final status P0_STATIC_PASS or P0_BLOCKED.

9.3 Evaluator

Queue: EVALUATOR_DUCA_P0P1_AMENDMENT-v001

Hard bound: protocol-only work, at most 3 elapsed task-hours, no data, model, CPU execution, GPU, or metric access.

Work:

amend the preregistration with the accepted uniform formula;

bind endpoint quantiles and half-up tie semantics;

bind the end-exclusive [0,T_v] segment domain;

bind the canonical pre-NMS hook;

define exact remote-CPU P1 commands and expected receipt schemas without running them;

preserve PRE_RUN_READY.status="BLOCKED".

Required return:

amended P0/P1 protocol;

expected canonical fixture hash;

remote-CPU command manifest;

receipt schema for parity, geometry, round trip, and NMS-order tests;

explicit zero-execution attestation.

9.4 Timed-out worktree disposition

All timed-out Builder/Critic worktree state is discarded for scientific and implementation continuation:

do not resume it;

do not cherry-pick it;

do not infer its contents;

do not use it to reduce the new review scope.

A read-only forensic snapshot may be retained solely to document that it was excluded. The new Builder starts from a clean exact-commit worktree.

10. P0/P1 STOP RULES
10.1 P0 pass conditions

P0 remains blocked until all of the following have durable receipts:

Canonical uniform identity: one generator and one fixture hash for all listed normal and short-window cases.

Endpoint identity: T=768, K=384 ends at 767 in every route.

Constant-density identity: bit-identical to canonical uniform with zero tolerance.

Detector invariance: resolved detector, assignment, head, losses, NMS, evaluator, split, class map, and augmentation identities match.

Pre-NMS mapping: static call graph shows physical mapping before every NMS.

Coordinate-state safety: selected, physical, unknown, and double-map cases are explicit and fail closed.

Critic closure: no unresolved P0 finding.

Evaluator amendment: all conventions and expected receipts are frozen.

Any violation stops P0. No threshold widening, endpoint compromise, fallback generator, NMS modification, or deduplication is allowed.

10.2 P1 pass conditions

P1 is a later remote CPU, synthetic-only gate. It is not presently authorized.

It must establish:

constant-density degeneration and deterministic serialization;

batch-size, permutation, and duplication invariance;

exact count, endpoints, range, uniqueness, and monotonicity;

maximum unselected run ≤3;

maximum uniform displacement ≤16;

physical-to-selected-to-physical error ≤1e−5 dense units;

selected-to-physical-to-selected error ≤1e−5 selected units;

exact selected-knot round trips;

scores and labels unchanged;

actual first NMS argument bit-identical to inverse-mapped physical proposals;

unchanged NMS callable and configuration;

clean/wrapper five-boundary parity;

zero dataset access and zero GPU initialization;

exact revision, patch, config, environment, command, output path, and deviation receipt.

One failure stops P1 and leaves P2 blocked. No partial pass is admitted.

11. FULL PRE_RUN_READY CHECKLIST FOR A LATER P2

A P2 readiness record must remain BLOCKED until every item below is concrete and non-null.

11.1 Authority

new immutable P2 record ID;

fresh Pro instruction artifact ID and message ID;

human authorization ID;

authorized experiment ID;

authorized stage exactly P2_ONLY;

authorization expiry;

run-equivalent cap;

spend cap.

11.2 Code and configuration

exact Git revision;

accepted Builder patch artifact ID and SHA-256;

resolved config path and SHA-256;

mechanism harness path and SHA-256;

canonical uniform specification ID and SHA-256;

deviation receipt, using the literal "none" when no deviation exists.

11.3 P0/P1 gates

Artifact IDs and pass status for:

canonical uniform and wrapper parity;

detector invariance;

density geometry properties;

coordinate round trip;

pre-NMS transport order;

Critic P0/P1 closure.

11.4 Training-side data boundary

scope exactly training_side_video_disjoint_utility_only;

dataset identity;

training-media manifest hash;

training-annotation hash;

utility-set manifest path and hash;

main-training exclusion receipt;

at least 64 videos and 128 windows;

stratification specification hash;

grouping key video_id;

validation media access false;

validation annotation access false;

test media access false;

test annotation access false;

forbidden-payload attestation.

11.5 Fixed models

fixed detector checkpoint path and hash;

fixed detector config hash;

fixed selector checkpoint path and hash;

complete checkpoint-provenance receipt.

11.6 Sampling and perturbations

dense candidate length 768;

requested K=384;

accepted K_eff rule;

endpoint-inclusive canonical generator;

maximum unselected run 3;

maximum adjacent span 4;

maximum displacement 16;

exact/unique/strictly-increasing requirement;

perturbation specification ID and hash;

registered perturbation families only:

single-frame swap;

dispersed 5% swap;

dispersed 10% swap;

contiguous swap;

global-density steps 0.25, 0.5, and 1.0.

11.7 Statistics

Prospectively frozen, before execution:

total detector loss change;

classification loss change;

boundary-regression loss change;

video-grouped bootstrap;

confidence level 0.95;

bootstrap replicate count and seed;

tie handling;

missing/failed perturbation handling;

single-frame Spearman lower-bound rule;

5% Spearman lower-bound rule;

direction-accuracy lower-bound rule;

exact top-decile statistic;

number and seeds of random-control replications;

minimum top-decile effect;

confidence rule for top-decile versus random;

numerical “no stable inverse relation” rule for 10% and global steps;

classification/regression component-admission rule;

failure action stop_direct_detector_gradient_route.

The currently unresolved numerical details must be frozen by a later P2-specific first-author decision. Their absence alone keeps readiness blocked.

11.8 Execution identity and resources

remote-only mode;

local execution disabled;

remote cluster and Slurm partition;

hardware model;

GPU count;

CPU count;

memory and wall-time limits;

environment-lock path and hash;

working directory;

exact command argv;

output root;

unique job name;

concurrent jobs forbidden.

Any change to code, config, data, checkpoint, command, environment, or authority requires a new run identity.

11.9 Outputs and access

Allowed outputs only:

registered training-side loss deltas;

perturbation identities;

registered alignment statistics;

resource, failure, and access receipts.

Required prohibitions:

official evaluator not invoked;

official metrics not computed;

all mAP fields forbidden;

held-out predictions forbidden;

raw prediction caches forbidden as selection inputs;

result path, seal path, and access-log path fixed before execution.

11.10 Final readiness predicates

READY requires all four:

all_required_fields_non_null = true
all_p0_p1_gates_pass = true
all_fresh_pro_questions_resolved = true
no_scope_deviation = true

Otherwise:

status = BLOCKED
ready = false
12. REMOTE GPU AUTHORIZATION

No concrete remote GPU experiment is authorized.

Accordingly:

no GPU config is authorized;

no split or baseline is opened for execution;

no seed is activated;

no command is approved;

no GPU-hour, run-equivalent, or spending budget is granted;

no Slurm partition or output path is approved;

no validation/test access is permitted.

P0 patching and static audit may proceed only within the stated queues. Remote CPU P1 also requires a separate human authorization and accepted patch package. GPU use before a later explicit decision is prohibited.

13. PUBLICATION, FAIRNESS, AND NOVELTY RISKS
13.1 Publication status

The project remains pre-result. The P0 findings are correctness evidence only; resolving them would not show that learned acquisition is useful. No mAP, cost, convergence, or superiority statement is admitted. 

RESULTS-v001
 

PAPER_DRAFT-v001

13.2 Fairness

The endpoint revision changes the actual baseline sample set. Therefore:

no old model trained or evaluated with the 766 endpoint contract may serve as the clean baseline;

all paired future arms must be regenerated and trained under the shared canonical generator;

uniform and learned arms must have identical K_eff sequences;

detector, initialization, augmentation, successful updates, terminal checkpoint, NMS, evaluator, and window identities must match;

equal-heavy-frame attribution and equal-full-stack-cost comparison remain separate analyses;

short-window padding and actual backbone work must be disclosed.

These requirements are consistent with the registered fixed-K-first experiment plan. 

EXPERIMENT_PLAN-v001

13.3 Leakage and coordinate risk

Physical transport may use only:

selected positions;

valid length;

raw segment coordinates.

It may not use GT, class labels, proposal scores, prediction caches, teacher outputs, validation information, or batch-level utility. Any such dependency is a P0 scientific failure.

13.4 Novelty

The current Sources do not support a final priority or novelty claim. Broad adaptive temporal sampling, token allocation, or saliency selection is not by itself sufficient. The only potentially defensible contribution remains the conjunction of:

real physical-frame removal before the heavy backbone;

exact fixed K;

bounded temporal geometry;

unchanged detector;

external pre-NMS coordinate transport;

high-IoU and short-action evaluation;

full decode-to-proposal cost accounting.

A dedicated contemporary literature audit remains mandatory before manuscript novelty language is promoted. 

LITERATURE_AND_GAP-v001

13.5 Negative-evidence preservation

The endpoint conflict and post-NMS mapping defect must be preserved as versioned P0 failures. Future parity, gradient, pilot, high-IoU, or cost failures must likewise be retained. They may not be erased by renaming the route, adding unregistered modules, or relabeling infrastructure success as a model contribution. 

FAILURES_AND_PIVOTS-v001

14. DRIFT_CHECKLIST
14.1 Is the implementation currently testing the accepted hypothesis?

No. The accepted hypothesis requires one shared uniform reference and physical mapping before NMS. Both are violated at the pinned commit. The route becomes faithful only after P0/P1 closure.

14.2 Is the contribution a model idea or infrastructure?

The bounded physical-acquisition rule is the candidate model idea. Canonical generators, coordinate adapters, validators, fixtures, and receipts are supporting infrastructure. Infrastructure correctness cannot support an accuracy claim.

14.3 Which parts are unnecessary or overengineered?

For the primary route, dynamic K, quota policies, role/process/value/risk heads, altered physical-grid detectors, auxiliary GT selector losses, and broad controller stacks remain deferred. P0 must not reintroduce them.

14.4 Are baseline, compute, data, tuning, stopping, and evaluator fair?

Not yet. Endpoint mismatch breaks sampling parity, and post-NMS mapping breaks coordinate parity. Future fairness requires the shared generator, identical downstream configuration, actual heavy-work ledgers, paired windows/seeds, terminal checkpoints, and an immutable evaluator.

14.5 Are leakage, cherry-picking, post-hoc drift, and relabeling controlled?

They are controlled only if the adapter is metadata-only, all thresholds are preregistered, validation/test information remains inaccessible, failed routes are preserved, and no old endpoint baseline or best-seed/checkpoint is reused.

14.6 Is the route formal, reproducible, serious, and publishable?

It is scientifically serious but not formal or publishable. P0 and P1 have not passed, P2 is not ready, and no model-quality evidence exists.

14.7 What evidence would falsify the route, and will it be preserved?

Immediate falsifiers include:

any uniform fixture mismatch;

non-bit-identical constant density;

duplicate or non-monotone positions;

hidden padded work;

detector/config drift;

NMS observing selected coordinates;

double mapping;

clean/wrapper parity failure;

future gradient-alignment failure;

future nonpositive or high-IoU-damaging model evidence.

Every falsifier must be retained as a versioned failure artifact.

14.8 Should the project continue unchanged, simplify, change direction, or stop?

It must revise narrowly. The scientific route is preserved, but execution remains blocked until uniform and coordinate semantics are corrected. It must not continue unchanged and does not yet warrant a broader pivot.

14.9 Is this presently a publishable model idea or an engineering system?

Presently it is a candidate model idea surrounded by preparatory engineering. It becomes publication-relevant only if the minimal unchanged-detector implementation later produces sealed, fair, cost-aware evidence.

14.10 What exact return is required?

The next admissible package is only:

Builder’s clean P0 patch, canonical fixture, config diff, and no-execution receipt;

Critic’s P0 closure and order-sensitive counterexample;

Evaluator’s amended P0/P1 protocol and unexecuted remote-CPU manifest.

A later fresh Pro decision may then decide whether to authorize remote CPU P1. No P2 or GPU decision may be inferred from this artifact.

FINAL_DECISION: REVISE; P0=BLOCKED_PENDING_CLEAN_PATCH_AND_CLOSURE; P1=NOT_AUTHORIZED; P2_PRE_RUN_READY=BLOCKED; REMOTE_GPU=PROHIBITED
```
