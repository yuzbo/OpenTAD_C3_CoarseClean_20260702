---
request_id: PRO_INITIAL_REVIEW-v001
project_id: g-p-6a796fef9a00819194024cf1de3bd697
nonce: 122c44f96c0c4397a394a6738c0a2259
date: 2026-08-10
author_role: coordinator
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
---

# Request: PRO_INITIAL_REVIEW-v001

Act as the senior first-author research lead for a CVPR 2027 model-innovation project. Fully take over the scientific context, audit it skeptically, and return one prioritized research plan. This is a research review, not permission to run experiments or mutate the repository.

## Immutable identity and evidence boundary

- ChatGPT Project ID: `g-p-6a796fef9a00819194024cf1de3bd697`
- Nonce: `122c44f96c0c4397a394a6738c0a2259`
- Canonical repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`
- Fixed accessible revision: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`
- Branch at that revision: `codex/duca-total60-plugin-cvpr-20260727`
- The local coordinator worktree is one commit ahead and heavily dirty/unpushed. It is explicitly excluded from your code evidence. Do not cite or assume local-only implementation.
- No GPU run, held-out test, formal experiment, result promotion, Git push, or submission is authorized by this request.

## Key code paths at the fixed revision

1. `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
2. `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py`
3. `opentad/models/selectors/pc_ot_mras_reader.py`
4. `opentad/models/necks/pc_ot_mras_detector_bridge.py`
5. `opentad/models/utils/pc_ot_mras_raw_prediction_guard.py`
6. `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py`
7. `tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py`
8. `tests/pc_ot_mras_test_utils.py`

Inspect the GitHub revision and cite the exact commit and paths actually used. If GitHub access or a path is unavailable, state that explicitly rather than pretending to have read it.

## Confirmed Project Sources, in reading order

All ten are version `v001`, dated 2026-08-10, tied to the fixed GitHub revision, and confirmed in this exact Project:

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

In your response, list the specific Source filenames/versions actually used. If you cannot access them, say so and treat the relevant conclusion as blocked.

## Paper objective and task boundary

The task is offline Temporal Action Detection, not Online TAD and not causal streaming. The research question is whether a cheap, deploy-visible full-video scout can allocate both (a) a discrete per-video heavy-frame budget under a global average-cost constraint and (b) monotone physical-time acquisition positions, outperforming strong cost-matched uniform controls while protecting high-IoU localization and reducing real full-stack cost.

Primary benchmark/protocol: THUMOS14 with the repository's official OpenTAD train/validation split. Evaluation must use the official Avg-mAP over tIoU 0.3:0.1:0.7, mAP@0.6 and 0.7, short-action/boundary analysis, multiple registered seeds, terminal checkpoints, and end-to-end cost including scout/transport overhead. Held-out test access is not currently authorized.

## Candidate model route and falsifiable hypotheses

Working route: DUCA-RIME, a pre-backbone task-aware allocator. It should use deploy-visible low-cost evidence to predict per-video K and exact unique physical-time positions, enforce coverage/max-gap constraints, run the heavy backbone only at the realized positions, and map detector proposals from selected coordinate q to physical coordinate t before unchanged official NMS.

Candidate allocation families to adjudicate narrowly:

- independent positions per candidate K;
- strict nested acquisition across K;
- at most one weak-overlap compromise, only if train/utility-only Oracle regret shows strict nesting is too restrictive.

Falsifiable hypotheses:

- H1: learned positions beat exact-uniform positions at the same realized fixed K and otherwise matched training/detector conditions.
- H2: per-video K improves the accuracy-cost frontier beyond the identical K sequence with uniform positions and beyond a shuffled K-histogram control.
- H3: benefits persist at high IoU and for short/boundary-sensitive actions.
- H4: gains remain after including scout, selection, transport, memory, and latency overhead.
- H5: all inference decisions are leakage-free and physical-time geometry is correct.

## Current implementation, experiment, and result status

The fixed GitHub revision implements substantial PC-OT-MRAS/DUCA scaffolding, including pre-backbone scouts, slot transport, selection plans, auxiliary losses, a discrete local dynamic-budget controller, payload denylisting, bridge components, configs, validators, launchers, and tests. It is not yet accepted as a complete paper-ready DUCA-RIME implementation.

There is no current formal result. `RESULTS-v001`, `RESULT_ANALYSIS-v001`, and `PAPER_DRAFT-v001` are explicitly `BLOCKED_PRE_RESULT`. Do not report or infer a DUCA mAP, do not claim improvement over the AdaTAD uniform 0.5 reference near mAP 65, and do not elevate any historical subset, single-seed, intermediate, failed-root, over-budget, Oracle, synthetic, upstream, or infrastructure number.

The intended first decisive Stage-A contrast is full official data, multi-seed: dense T768; exact-uniform fixed K384; uniform mixed-K training with exact-uniform K384 evaluation; DUCA learned fixed K384. Dynamic K is admitted only after fixed-K feasibility. This is a plan, not an authorized run.

## Preserved negative evidence and blockers

- Earlier direct DUCA variants did not fairly establish learned selection over cost-matched uniform sampling.
- Selected-rank decoding caused physical-coordinate mismatch; q-to-t mapping before official NMS is an invariant.
- Local-cell, actionness top-k, selected-rank decode, hard K384 query deletion/SparseHead, DCSR-G1, and ODF-CR-G2 were rejected or negative in their evaluated scopes.
- Prior missing symlink/mask propagation/static temporal-axis/checkpoint receipt failures are engineering failures, not model-quality evidence.
- A fixed-budget selector cannot by itself support a dynamic-budget claim.
- Precise novelty and AdapTok comparison require literature verification; do not invent citations.

## Role/process status

Builder, Critic, and Evaluator are not yet registered as independent Codex processes. Your plan must assign them distinct, bounded responsibilities, but do not pretend that they already exist. Builder may plan the smallest claim-bearing change; Critic independently attacks novelty/fairness/leakage/overengineering; Evaluator freezes the preregistration and result-admission rules. GPU and held-out actions remain human-gated.

## Required output: exactly one structured `PRO_INITIAL_REVIEW-v001`

Return a self-contained Markdown report with all sections below:

1. **Context reconstruction.** Restate in your own words the paper purpose, offline-TAD boundary, core scientific problem, and plausible contribution.
2. **Evidence reconstruction.** Rebuild the current method/candidate routes, implementation state, experiment plan, result status, negative evidence, and blockers. Separate known facts, inferences, and unknowns.
3. **Hypothesis alignment audit.** Decide whether the current implementation and experiment plan truly test the frozen paper hypothesis, and identify proxy tasks or engineering drift.
4. **Publishability audit.** Judge novelty, conceptual clarity, TAD specificity, CVPR significance, and the strongest likely reviewer objections. Compare carefully with AdaTAD, AdapTok, token/frame selection, and adaptive computation; do not invent literature facts.
5. **Protocol/fairness audit.** Check baselines, official split, metrics, cost matching, tuning budgets, checkpoint/stop rules, multi-seed statistics, leakage, evaluator integrity, and short-window/physical-time semantics.
6. **Route adjudication.** Decide the smallest model route worth implementing now. Explicitly adjudicate fixed-K before dynamic-K and independent versus nested versus optional weak-overlap allocation.
7. **Prioritized next-stage plan.** Give an ordered plan in which every item is bound to a candidate paper claim or a falsification test. For each item specify owner (`Builder`, `Critic`, `Evaluator`, or `Human`), input artifact, concrete output, acceptance criterion, stop condition, and rough resource class (CPU/read-only, single-GPU pilot, or formal multi-seed). Do not authorize gated work.
8. **Cheapest decisive tests.** Name the cheapest falsification experiment, strongest required baselines/controls, minimal isolating ablations, success/failure thresholds expressed without fabricated outcomes, rough resource estimate, and early-stop rule.
9. **Do-not-do list.** Identify engineering/platform work, broad sweeps, detector expansion, admission simulations, or paper writing that should be deferred.
10. **Mandatory Pro drift checklist.** Answer all nine questions with citations to the actual commit paths and/or Source versions used:
   1. Is the implementation still testing the frozen paper hypothesis?
   2. Which work is scientific contribution, and which is only infrastructure?
   3. Is any component unnecessary or overengineered relative to the claim?
   4. Are baselines, compute, data, tuning, stopping, and evaluator treatment fair?
   5. Is there test leakage, cherry-picking, post-hoc hypothesis drift, or result relabeling?
   6. Would a skeptical CVPR reviewer consider the experiment formal, reproducible, serious, and publishable?
   7. What evidence would falsify the current route, and has that evidence been preserved?
   8. Should the route continue, be simplified, pivot, or stop?
   9. Are we spending effort on a publishable model idea, or drifting into building a complete engineering system?
11. **Sources and code actually used.** State the exact GitHub commit, paths opened, confirmed Project Sources used, and access limitations.
12. **Final route status.** End with exactly one standalone status token from: `CONTINUE`, `REVISE`, `PIVOT`, `STOP`, or `ESCALATE_HUMAN`. Do not include more than one token or qualify it.

Do not fabricate experimental numbers, citations, file access, model identity, or permissions. A missing artifact must remain an explicit blocker. Optimize for a publishable model idea and the shortest decisive evidence path, not for a comprehensive engineering platform.
