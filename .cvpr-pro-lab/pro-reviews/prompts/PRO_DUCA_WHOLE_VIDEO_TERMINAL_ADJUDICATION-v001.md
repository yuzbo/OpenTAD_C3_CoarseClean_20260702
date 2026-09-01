# DUCA whole-video consistent-budget terminal scientific adjudication

Nonce: `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831`

## Your role

Act as the continuing scientific head and primary research owner of DUCA. Independently own the scientific question,
mechanism, falsifiable prediction, evidence interpretation, claim scope, failure diagnosis and research direction. Treat
this message as an evidence handoff, not a request to ratify Codex. Do not delegate the scientific choice back to Codex or
the human when the evidence permits a decision. Codex is the implementation and evaluation executor after you freeze one
task.

Use ordinary research language. Do not create workflow status codes, engineering contracts, proof systems or a new
coordination framework. Do not assume that any alternatives mentioned in prior discussions are exhaustive or endorsed.

## Latest public implementation — authoritative code truth

The current implementation has been pushed and independently verified against its upstream branch. Use these permanent
GitHub links rather than local paths, historical branches or Project memory:

- Repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- Actual remote branch:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
- Exact clean commit:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- Frozen evaluator runner:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
- Focused regression test:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tests/test_duca_whole_video_consistent_budget_falsifier.py>
- Unchanged three-tier budget implementation:
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>

The local clean worktree, its upstream branch and the remote branch all resolve to
`33e4ed137c33eef07f0452b44506a6993bdf7535`. The commit preserves the sealed producer proposal-row order; it does not
change predictions, candidate definitions, observation costs, Soft-NMS, evaluator, budget tiers or the frozen gate.
Twenty-eight focused tests and an independent Critic passed on this exact commit.

## Paper question and negative history

DUCA asks whether low-cost semantic evidence can reduce real high-resolution VideoMAE computation for offline temporal
action detection while protecting high-tIoU localization. The current clean H65 30+60 reference reaches 65.13% Avg-mAP
and 43.31% mAP@0.7 on official THUMOS14 validation; the shared dense AdaTAD reproduction reaches 68.73% Avg-mAP. These
numbers are context only and are not directly comparable with the training-side controller holdout used below.

Material results preceding the present terminal experiment are:

1. At fixed K384, the task-state native-tubelet coreset reached 62.81% Avg-mAP versus 64.13% for its matched uniform
   control, so fine-grained coreset score tuning stopped.
2. Temporal facility-location Coverage-v1 failed its pre-registered unlabeled intervention gate before training; it
   provided no mAP result.
3. With the H65 detector and K256/K384/K512 sealed counterfactual predictions frozen, the 50%-capped marginal-utility
   oracle improved the training-side holdout by only `+0.726/+0.729` percentage points in Avg-mAP/mAP@0.7. Releasing the
   cap reduced this to `+0.427/+0.450`.
4. Exhaustive evaluation of the 96 capped-to-released joint states found no state passing the frozen `+0.8/+1.0` gate;
   the best joint-gate state was only `+0.554/+0.933`. You previously stopped that additive window-level Marginal-v1.
5. You then made one project-level PIVOT: test whether the decision unit was wrong by giving all windows of one donor
   video K256, all windows of one recipient video K512, and all other videos K384. You froze the consequence that zero
   passing candidates would stop DUCA method innovation within the current THUMOS14/H65/three-tier/resource boundary.

## Frozen whole-video experiment

- Data: 40 training-side controller-holdout videos, 124 overlapping windows.
- Fixed baseline: every window requests K384.
- Candidate: for every ordered pair of distinct videos, every donor window requests K256, every recipient window requests
  K512 and every other window requests K384.
- A legal candidate requires actual non-baseline execution in both changed videos and total actual observation cost no
  greater than the fixed baseline cost `47110`. Short windows retain `min(valid_observations, K)` cost accounting.
- Candidate identities were completely generated and sealed before labels, ground truth or metrics were read.
- Evaluation reused the same sealed predictions, physical-coordinate reconstruction, Soft-NMS, annotation, class map and
  evaluator. It ran no detector or Scout forward, training, gradient, bootstrap, official validation or official test.
- Frozen gate: at least `+0.8` percentage points in Avg-mAP and `+1.0` percentage points in mAP@0.7 simultaneously, at
  actual cost no greater than `47110`.

Corrected PRE_RUN Job `1262161` passed on the exact public implementation: 40 videos, 124 windows, 1560 ordered pairs,
704 legal candidates, 1330 candidates with any actual intervention, and `0.0` percentage-point reproduction error for
the fixed, capped and released anchors. PRE_RUN receipt SHA-256 is
`734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`; candidate-manifest SHA-256 is
`c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`.

Formal Job `1262162` stopped after 500 candidates only because Slurm node `g0022` went down; it produced no terminal
result and is not scientific evidence. The one exact same-task infrastructure recovery, Job `1262190`, reused the same
clean snapshot, sealed manifest, predictions, evaluator and gate. It completed `704/704` candidates with Slurm
`COMPLETED 0:0`; no third job was run.

## Authoritative terminal result

Full terminal artifact:
`/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`

SHA-256: `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`

The fixed K384 anchor on this training-side holdout is:

- Avg-mAP `88.131197%`
- mAP@0.3/0.4/0.5/0.6/0.7:
  `95.833793/93.684529/89.581223/85.285857/76.270583%`
- actual observation cost `47110`
- reproduction error `0.0` percentage points for every reported metric

All 704 legal candidates completed, with 705 evaluator calls including the fixed anchor. No candidate passed both gates.
The passing-candidate count is zero.

The three most informative extrema are:

1. Best Avg-mAP candidate, `video_validation_0000158=>video_validation_0000173`:
   - delta Avg-mAP/mAP@0.7: `+0.694215/-0.043632` percentage points
   - actual cost `46982`
2. Best mAP@0.7 candidate, `video_validation_0000490=>video_validation_0000173`:
   - delta Avg-mAP/mAP@0.7: `-0.235922/+0.496998` percentage points
   - actual cost `46854`
3. Best candidate under the pre-registered joint-gate ordering,
   `video_validation_0000419=>video_validation_0000173`:
   - delta Avg-mAP/mAP@0.7: `+0.147383/+0.489786` percentage points
   - actual cost `45830`
   - joint-gate margin `-0.652617` percentage points

The result explicitly records: no training, detector/Scout forward, gradients, bootstrap, official validation or official
test; `paper_claim_allowed=false`, `deployable_policy_claim_allowed=false`, and development-holdout oracle selection.
There is no uncertainty interval. The result is a deterministic, exhaustive diagnosis of the frozen development action
space, not a population-level performance estimate or deployable policy.

## Required independent adjudication

First verify whether the implementation and evidence are faithful enough to apply the pre-registered consequence. Then
make your own single scientific decision: continue, narrow the claim, revise the scientific question, pivot to a genuinely
new mechanism, stop, or escalate only a real human authority/resource boundary.

Your response must do all of the following:

1. State whether the zero-pass result triggers the previously frozen stop boundary, and define exactly what is stopped
   without overgeneralizing to all dynamic computation, all low-cost Scouts or all budget spaces.
2. Diagnose the mechanism-level failure before considering any successor. Separate facts supported by the exhaustive
   action-space result from competing explanations that remain untested, and identify the strongest alternative
   explanation.
3. State the strongest honest paper claim and the claims that remain prohibited. Address whether this negative sequence
   is scientifically publishable as a result, supplementary analysis or only internal evidence.
4. Decide independently whether a scientifically distinct paper question with material expected information gain remains.
   Do not revive Marginal-v1, extend this candidate search, tune the gate, or present a larger search as a new mechanism.
5. Issue exactly one current task if continued work is justified. Freeze its falsifiable prediction, control, data split,
   metrics, compute/fairness treatment, uncertainty requirement, stop rule, minimal implementation surface, role order and
   an absolute expected-return deadline. If no task is justified, explicitly state that no Builder/Critic/Evaluator work
   remains and what evidence would be required before reopening research.

End with `next_owner`, `next_action`, `dependency`, and `expected_return_at`. Stop after the one decision and one task. Do
not provide a menu of routes or ask Codex to choose between them.

Repeat nonce `DUCA-WHOLE-VIDEO-TERMINAL-ADJUDICATION-v001-20260831` verbatim in the final response.
