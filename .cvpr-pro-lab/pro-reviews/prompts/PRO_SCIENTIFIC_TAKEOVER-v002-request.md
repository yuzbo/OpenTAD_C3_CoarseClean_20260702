PARALLEL-PRO-STRESS-20260811::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::bed5d343023bcce60a35a9233e18f7aa

# DUCA-RIME — Fresh Pro Scientific Takeover Request v002

## Session identity

- ChatGPT Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`
- Project URL: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project`
- Canonical GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Exact review commit: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`
- Branch carrying that remote commit: `codex/duca-total60-plugin-cvpr-20260727`
- Coordinator task ID: `019fa3db-42bf-7f30-a0ab-2b8171ab33ed`
- Required fresh-turn nonce: `bed5d343023bcce60a35a9233e18f7aa`
- Parallel stress marker: `PARALLEL-PRO-STRESS-20260811::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::bed5d343023bcce60a35a9233e18f7aa`
- Required response artifact: `PRO_INITIAL_REVIEW-v002`

You are hereby appointed the **acting Scientific First-Author Agent and Primary Research Owner** for this exact model, experiment program, and CVPR 2027 paper project. You—not Codex—own model innovation, candidate-route comparison and selection, experiment ordering, decisions after failure (continue/revise/pivot/stop), result interpretation, claim scope, paper narrative, and the next bounded assignments for Builder, Critic, and Evaluator.

Codex is only the coordinator/control-plane transport and evidence unit. It must not choose the research route, next experiment, claims, or paper conclusion. Human approval remains mandatory for GPU/paid work, remote/Slurm execution, held-out/test access, formal result promotion, final claims, authorship, and submission.

## Mandatory context to use

First read and reconcile all twelve confirmed Project Sources in this exact order:

1. `PROJECT_CHARTER-v001.md`
2. `LITERATURE_AND_GAP-v001.md`
3. `ROUTE_DECISION-v001.md`
4. `EXPERIMENT_PLAN-v001.md`
5. `IMPLEMENTATION_STATUS-v001.md`
6. `RESULTS-v001.md`
7. `RESULT_ANALYSIS-v001.md`
8. `FAILURES_AND_PIVOTS-v001.md`
9. `CLAIM_MAP-v001.md`
10. `PAPER_DRAFT-v001.md`
11. `CURRENT_RESEARCH_STATE-v001.md`
12. `MODEL_EXPERIMENT_HISTORY-v001.md`

Then inspect the canonical GitHub repository only at the exact commit above. At minimum audit these paths and any directly required dependencies/tests/configs:

- `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
- `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py`
- `opentad/models/selectors/pc_ot_mras_reader.py`
- `opentad/models/necks/pc_ot_mras_detector_bridge.py`
- `opentad/models/utils/pc_ot_mras_raw_prediction_guard.py`
- relevant `configs/adatad/thumos/*pc_ot_mras*`
- corresponding validators, focused tests, and launch contracts

Do not use the dirty/unpushed coordinator worktree as code truth. State exactly which commit, Source versions, code paths, and external literature you actually used, and state every access limitation.

## Scientific boundary and present evidence

- Task: offline Temporal Action Detection, not Online TAD or causal streaming.
- Primary benchmark: THUMOS14 under the repository's official OpenTAD train/validation protocol.
- Primary paper metrics: official Avg-mAP over tIoU 0.3:0.1:0.7, mAP@0.6 and mAP@0.7, plus measured full-stack cost.
- Core hypothesis: deploy-visible low-cost video evidence may allocate exact physical-time heavy-frame positions and eventually a per-video budget better than cost-matched uniform sampling while preserving high-IoU localization.
- Current formal evidence state: `BLOCKED_PRE_RESULT`. There is no accepted full-split, multi-seed, terminal-checkpoint, full-stack-cost DUCA result tied to the pinned revision.
- Do not claim current superiority over the historical AdaTAD/uniform 0.5 reference near mAP 65. That value is only a baseline target to reproduce under the locked official protocol.
- Do not promote intermediate, subset, single-seed, failed-root, over-budget, train-domain, Oracle, synthetic, proxy, upstream, validator, or infrastructure numbers.
- Current registered role tasks are Builder `019febf1-d382-7013-bf49-aab6b7d887e4`, Critic `019febd9-0e77-7a12-bdfb-e6c9d9892856`, and Evaluator `019febf2-690b-7093-bcf5-8eb69636770e`. They are independent and waiting for your versioned decision.

No GPU, remote/Slurm, held-out/test, formal experiment, Git push, result promotion, or submission is authorized by this request. Your plan must stop at explicit Human gates for those actions.

## Required first-author work

1. Perform `HISTORY_SYNTHESIS`: reconstruct in your own words the paper objective, offline-TAD boundary, scientific gap, method evolution, rejected routes, protocol repairs, current implementation, experiment design, result state, and unresolved questions. Identify contradictions, missing evidence, and uncertainty rather than silently resolving them.
2. Decide whether the pinned implementation actually tests the paper hypothesis. Audit detector invariance, physical-time semantics, q-to-t mapping before NMS, differentiability, exact/unique/monotone K, short-window behavior, masks, budget enforcement, batch independence, leakage, cost measurement, and control plumbing.
3. Judge novelty and CVPR publishability against current primary literature. Distinguish generic adaptive token/frame selection from the TAD-specific scientific contribution; identify occupied claims and the smallest defensible novelty statement.
4. Audit fairness and reproducibility: official data/splits, evaluator, initialization, exposure, updates, checkpoint rule, seeds, hyperparameter parity, stopping, uncertainty, full-stack cost, leakage, and post-hoc drift.
5. Choose exactly one route status: `CONTINUE`, `REVISE`, `PIVOT`, `STOP`, or `ESCALATE_HUMAN`.
6. Provide the complete next-stage scientific plan, but authorize only work within the stated human boundary. Every task must bind to a paper claim or falsification test.
7. Give the cheapest decisive falsification, strongest required baseline, minimal isolating ablations, preregisterable success/failure thresholds, resource classes/estimates where supportable, early-stop rules, and a strict do-not-do list. Do not invent numeric thresholds when evidence is absent; instead define how they must be frozen before metric access.
8. Assign separately bounded tasks to Builder, Critic, and Evaluator. Builder may implement only the frozen decision; Critic attacks it; Evaluator preregisters/measures it. Do not ask Codex coordinator to make a scientific choice.

## Mandatory response schema

Return a single structured document titled `PRO_INITIAL_REVIEW-v002` with all of the following sections:

### SESSION_ASSERTION

At the very beginning of the response, reproduce verbatim the coordinator task ID, Project ID, nonce, Git commit, `CURRENT_RESEARCH_STATE-v001`, and `MODEL_EXPERIMENT_HISTORY-v001`. Then state the new-conversation status, whether this response is independent of old Pro conversations, and explicitly state whether you saw any other project marker. Do not reproduce any marker other than this DUCA marker.

### MODEL_EFFORT_ASSERTION

State the exact web model label/tier and reasoning effort actually used. Do not claim a tier or effort you cannot verify.

### ROLE_ACKNOWLEDGMENT

Explicitly accept or reject appointment as acting Scientific First-Author Agent and Primary Research Owner; if accepted, state that Codex is not the scientific route owner.

### CONTEXT_USED

List all twelve Source filenames/versions in order, the exact GitHub commit, inspected paths, external literature, and access limitations.

### HISTORY_SYNTHESIS

Reconstruct the full research and experiment history, negative evidence, corrections, and present uncertainty.

### PAPER_OBJECTIVE

State the task boundary, scientific problem, intended contribution, falsifiable hypotheses, and what would count as a publishable result.

### CURRENT_JUDGMENT

Audit novelty, implementation fidelity, experiment validity, fairness, leakage, cost, reproducibility, overengineering, and CVPR readiness.

### SCIENTIFIC_DECISION

Give exactly one of `CONTINUE`, `REVISE`, `PIVOT`, `STOP`, or `ESCALATE_HUMAN`, followed by the decisive reasons.

### ROUTE_AND_CLAIMS

Select/freeze the scientific route or explicitly stop it. Define claim scope, excluded claims, falsification conditions, and route-change conditions.

### CODEX_DISPATCH

Provide three separate queue-ready briefs: Builder, Critic, Evaluator. Each must include claim/falsification target, exact inputs, allowed paths/actions, forbidden changes, concrete output, acceptance criterion, stop condition, evidence class, dependencies, and whether separate Human authorization is required.

### EXPERIMENT_PLAN

Specify the complete official plan from cheapest falsification through any conditional later stage: data/splits, cells, baselines, ablations, seeds, exposure, hyperparameters, checkpoints, evaluator, uncertainty, costs, thresholds, early stops, resources, leakage controls, and sealed-analysis rules.

### PUBLICATION_PLAN

Give the intended novelty statement, contribution list, required tables/figures/analyses, related-work burden, claim-to-evidence map, and the minimum result package needed for CVPR submission.

### DRIFT_CHECKLIST

Answer all ten items explicitly:

1. Is the implementation faithful to the scientific hypothesis?
2. Is current progress science or infrastructure?
3. Is there unnecessary complexity or platform building?
4. Are baseline, compute, data, tuning, stopping, and evaluator fair?
5. Are leakage, cherry-picking, post-hoc drift, and result relabeling controlled?
6. Would a skeptical CVPR reviewer find the work formal, reproducible, and publishable?
7. Is negative/falsification evidence preserved?
8. Should the project continue, simplify/revise, pivot, stop, or escalate?
9. Is the core a publishable scientific idea rather than a large engineering system?
10. Does every next action stay inside the current human authorization boundary?

### NEXT_RETURN_CONTRACT

Define the exact evidence package each role must return to you, which decision questions return only to Pro, the next fresh Pro review trigger, and the Human approvals required before execution or evidence promotion.

## Rejection conditions

Your response is incomplete and must not be executed if it does not use all twelve Sources, does not cite the exact commit, omits any required section, fails to choose one route status, delegates scientific choice back to Codex, invents evidence/numbers/literature/access, promotes nonformal metrics, omits the required identity echo, or contains another project's marker/content. End with the exact DUCA stress marker on the penultimate line and only the chosen route status on the final line.

PARALLEL-PRO-STRESS-20260811::019fa3db-42bf-7f30-a0ab-2b8171ab33ed::g-p-6a796fef9a00819194024cf1de3bd697::bed5d343023bcce60a35a9233e18f7aa
