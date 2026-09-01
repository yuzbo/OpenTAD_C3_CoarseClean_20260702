# DUCA fixed-budget native-tubelet terminal scientific adjudication

Nonce: `DUCA-NATIVE-TUBELET-CORESET-TERMINAL-v001-20260829`

You are the independent scientific lead and primary research owner for this DUCA turn. Make the scientific choice yourself; do not hand route selection back to Codex or the human. Use publication-oriented reasoning and ordinary temporal action detection terminology. Do not create workflow machinery or additional theory rounds in place of one decisive next action.

## Authoritative identity and scientific question

- Repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Frozen experiment revision: `b33391126eac05e3353d322b973dda91741f0732`
- Branch: `codex/duca-native-tubelet-coreset-20260828`
- Experiment record: `research-wiki/experiments/duca-native-tubelet-coreset-fixed384.md`
- Formal jobs: uniform `1260184`, task-state coreset `1260185`; both use seed `3407`.

Scientific question: under the same fixed `K=384` high-resolution RGB budget, does an H65-derived task-state native-tubelet coreset preserve THUMOS14 boundary, short-action and high-temporal-IoU localization better than deterministic uniform native-tubelet selection when context recycling, physical-time reconstruction, VideoMAE-S, Adapter, ActionFormer, losses, non-maximum suppression, split, optimizer schedule and official evaluator are matched?

This fixed-budget comparison is an attribution experiment for “where to compute.” It is not the final DUCA definition and is not evidence for dynamic budgeting.

## Frozen implementation

- Each 768-frame window is represented as 384 native two-frame VideoMAE tubelets; both arms select exactly 192 tubelets, so the heavy path reads 384 high-resolution frames and produces 192 heavy temporal tokens.
- Uniform selects 192 native tubelets deterministically on the valid grid.
- Coreset freezes the H65 Stage-1 epoch-29 exponential-moving-average scout. It averages two-frame actionness, takes the maximum two-frame boundary importance, derives temporal novelty from frozen hidden features, combines percentile scores with fixed weights, forces the two endpoints, limits the longest unselected gap to seven tubelets and breaks ties toward earlier physical time.
- Both arms use the same nearest-anchor low-resolution context recycling, physical-time residual and reconstruction to the 384-point native tubelet grid.
- Selected tubelets remain ordered by physical time, but packing nonadjacent tubelets into 24 conventional 16-frame VideoMAE clips may still create artificial adjacency across tubelet boundaries. This is a mechanism risk, not an established cause.
- Independent static review and N16R4 execution-and-resume checks passed. Each full run completed 60 epochs and wrote an epoch-59 checkpoint containing the exponential-moving-average state.

## Terminal evidence and missing evidence

The official evaluator logs cover all 211 THUMOS14 validation videos and 422,000 predictions per arm:

| arm | Avg-mAP | mAP@0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 64.13 | 79.26 | 74.89 | 67.44 | 56.61 | 42.45 |
| task-state coreset | 62.81 | 78.91 | 74.09 | 65.90 | 54.58 | 40.56 |
| coreset minus uniform, percentage points | -1.32 | -0.35 | -0.80 | -1.54 | -2.03 | -1.89 |

Both jobs failed only after these metrics were computed. The configuration used `post_processing.save_dict=False`, so no prediction file existed for the structured metric writer. Consequently:

- no `metrics_epoch59_ema.json` was written;
- no sealed paired predictions or paired whole-video interval exists;
- no matched realized-compute or end-to-end cost artifact exists.

The point estimates are therefore diagnostic and unfavorable to this exact coreset, especially at high temporal IoU. They are not a training failure, a statistically closed population effect, an efficiency result, or evidence that all low-cost-scout plus sparse-heavy-compute designs fail. Do not infer a unique cause from the aggregate metrics alone.

The shared official dense AdaTAD reference is Avg-mAP `68.73%`, produced once by the separate shared baseline responsibility. It is context, not a same-commit causal control for this fixed-budget selector comparison.

## Required independent decision

Issue exactly one leading decision: `CONTINUE`, `REVISE`, `PIVOT` or `STOP`. Then:

1. State what the completed fixed-budget experiment does and does not falsify.
2. Decide whether one evaluation-only sealing pass from the existing immutable epoch-59 checkpoints is scientifically decision-changing. If yes, freeze its exact outputs, paired-video statistic, interval convention and stop rule; it may not retrain or change model, selector, data, split, checkpoint, NMS or evaluator. If no, explicitly say why the point estimates already settle the next scientific action.
3. Rank the supported mechanism explanations for the high-tIoU loss, clearly separating evidence from hypotheses. Consider selection-score quality, endpoint/max-gap coverage, artificial cross-tubelet adjacency in VideoMAE clip packing, context recycling and physical-time reconstruction. Do not claim causality without a discriminating observation.
4. Freeze one smallest decision-changing next scientific task. It must not repeat completed dense, uniform, random, arbitrary-frame, continuous-cliplet, H65-compression, UVT, Query-Bridge or PJST matrices.
5. Decide explicitly whether true dynamic budgeting remains scientifically justified now. If authorized, freeze one mechanism that changes the number of actually executed heavy clips and constrains mean realized VideoMAE compute to be no higher than the fixed-budget control. Fixed `K=384` remains a matched control, not the headline method. If not authorized, stop or revise it explicitly.
6. Define the result-to-claim boundary, the cheapest falsifier, one Builder → independent Critic → Evaluator handoff, the required official-data evidence, and an absolute deadline. Keep engineering to the minimum needed for a correct executable experiment.

Return the unique decision first, followed by the scientific interpretation, the frozen next task and `next_owner`, `next_action`, `dependency`, `expected_return_at`.
