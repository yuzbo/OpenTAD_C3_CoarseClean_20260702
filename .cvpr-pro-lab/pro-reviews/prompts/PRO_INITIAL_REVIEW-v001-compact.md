Project ID `g-p-6a796fef9a00819194024cf1de3bd697`; nonce `122c44f96c0c4397a394a6738c0a2259`; routing handshake passed for DUCA with 10 Sources and GPT-5.6 Pro.

Act as senior first author and fully take over this CVPR 2027 research project. Use the ten confirmed Project Sources in this order: PROJECT_CHARTER-v001, LITERATURE_AND_GAP-v001, ROUTE_DECISION-v001, EXPERIMENT_PLAN-v001, IMPLEMENTATION_STATUS-v001, RESULTS-v001, RESULT_ANALYSIS-v001, FAILURES_AND_PIVOTS-v001, CLAIM_MAP-v001, PAPER_DRAFT-v001. Inspect the canonical GitHub repository `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git` at fixed commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, especially the selector, dynamic-budget controller, reader, detector bridge, raw-prediction guard, main PC-OT-MRAS config, validator, and focused tests. State exactly what you actually accessed. The dirty/unpushed local tree is excluded.

The task is offline Temporal Action Detection, not Online TAD. The hypothesis is that a cheap deploy-visible full-video scout can allocate both a per-video heavy-frame budget under an average-cost constraint and exact monotone physical-time positions, beating strong cost-matched uniform controls while preserving high-IoU/short-action localization and reducing full-stack cost. Inference must be leakage-free; proposals map selected coordinate q to physical t before unchanged official NMS.

Primary protocol: THUMOS14 official OpenTAD train/validation, official Avg-mAP tIoU 0.3:0.1:0.7 plus mAP@0.6/0.7, short-action/boundary analysis, terminal checkpoints, multiple registered seeds, and cost including scout/transport. No held-out access is authorized.

Candidate allocation choices are independent-per-K, strict nested acquisition, and at most one weak-overlap compromise if train/utility-only Oracle regret shows nesting is restrictive. First establish fixed-K learned placement against exact uniform and mixed-K exposure controls; only then test dynamic K against the identical K sequence with uniform positions and a shuffled K-histogram control.

No formal result exists. RESULTS/RESULT_ANALYSIS/PAPER_DRAFT are BLOCKED_PRE_RESULT. Do not infer DUCA mAP or improvement over the AdaTAD uniform 0.5 reference near mAP 65. Reject historical subset, single-seed, intermediate, failed-root, over-budget, Oracle, synthetic, upstream, or infrastructure numbers. Preserve negative evidence: earlier DUCA did not fairly beat uniform; selected-rank decode was wrong; local-cell/actionness-top-k/selected-rank/hard-query-deletion SparseHead/DCSR-G1/ODF-CR-G2 were rejected or negative in scope; prior launcher/mask/axis/receipt failures are engineering only.

Builder, Critic, and Evaluator are being created as independent Codex processes but have not yet received substantive queues. No GPU, remote run, formal experiment, Git push, result promotion, or submission is authorized.

Return a Markdown report titled `PRO_INITIAL_REVIEW-v001` with:

1. context reconstruction and task boundary;
2. facts/inferences/unknowns for method, implementation, experiment, results, negatives, blockers;
3. hypothesis-alignment and engineering-drift audit;
4. novelty/publishability audit versus AdaTAD, AdapTok, adaptive token/frame selection, without invented literature;
5. protocol/fairness/leakage/evaluator/cost/short-window/physical-time audit;
6. route adjudication: fixed-K before dynamic-K and independent vs nested vs optional weak-overlap;
7. prioritized plan. Every item must specify owner (Builder/Critic/Evaluator/Human), claim or falsification test, input, concrete output, acceptance criterion, stop condition, and resource class; do not authorize gated work;
8. cheapest decisive falsification, strongest baselines, minimal isolating ablations, non-fabricated success/failure thresholds, rough resources, early stop;
9. explicit do-not-do list to avoid a general engineering platform;
10. all nine cvpr-pro-lab drift questions with commit/Source citations: hypothesis fidelity; science vs infrastructure; unnecessary complexity; fairness of baseline/compute/data/tuning/stopping/evaluator; leakage/cherry-picking/post-hoc drift/result relabeling; skeptical CVPR formality/reproducibility/publishability; falsification evidence preservation; continue/simplify/pivot/stop; publishable idea vs complete engineering system;
11. exact commit, paths and Source versions actually used plus access limitations;
12. final route status.

End with exactly one standalone token: CONTINUE, REVISE, PIVOT, STOP, or ESCALATE_HUMAN. Do not fabricate numbers, citations, access, model identity, or permission.
