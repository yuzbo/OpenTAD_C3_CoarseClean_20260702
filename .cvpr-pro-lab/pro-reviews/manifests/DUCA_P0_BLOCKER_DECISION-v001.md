---
turn_id: duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd
nonce: 51c88fd75537120ce96a417beb7e81dd
project_id: g-p-6a796fef9a00819194024cf1de3bd697
project_url: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697/project
github_url: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
github_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
parent_decision: PRO_INITIAL_REVIEW-v002
parent_artifact: art-20260811T045704Z-2d13c25e41
evidence_source: EVALUATOR_DUCA_DENSITY_P0P1-v001.md
evidence_level: preparatory_preregistration_no_execution
---

# Fresh P0-blocker scientific decision request

You are the acting Scientific First-Author Agent and Primary Research Owner
for DUCA-RIME. Codex is only the coordinator and evidence transport. This is a
new Project conversation; do not rely on a previous conversation and do not
delegate scientific decisions back to Codex.

At the beginning of your answer, echo verbatim:

`PROJECT_ID=g-p-6a796fef9a00819194024cf1de3bd697; TURN_ID=duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd; NONCE=51c88fd75537120ce96a417beb7e81dd; GIT_COMMIT=63a726a4aaf48ecbf6780bb196de43a890c6b4df; CURRENT_STATE=CURRENT_RESEARCH_STATE-v002; HISTORY=MODEL_EXPERIMENT_HISTORY-v002`

Also state the actual selected model/tier and available browser thinking effort,
whether you saw any material from another Project (must be no), and list the
Project Sources actually used.

The accepted initial decision was `REVISE`, selecting
`DUCA_FIXEDK_BOUNDED_MONOTONE_DENSITY_ACQUISITION-v001`: fixed K=384
pre-backbone physical-frame acquisition; learned nonnegative density +
inverse-CDF positions; a clean unchanged-detector uniform control; external
selected-to-physical coordinate transport before unchanged NMS/evaluation; no
dynamic K or detector/head/loss change. The exact GitHub source is the URL and
revision above. Current Project Sources include the confirmed initial review,
state/history v002, and newly confirmed
`EVALUATOR_DUCA_DENSITY_P0P1-v001.md`.

Only formal new evidence is the Evaluator preregistration. No model/test/data,
metric, local CPU, remote, GPU, Slurm, held-out, Git push, or code mutation was
run. It identifies two fail-closed P0 blockers in the frozen tree: (1) the
current data and selector uniform generators disagree on the normal T=768,
K=384 terminal endpoint (766 versus 767); (2) generic SingleStageDetector
performs batched NMS before the existing selected-to-dense inverse mapping.
Critic and Builder exceeded their bounded task time and produced no durable
returns; treat all partial statements or worktree state from them as
non-evidence and do not infer unseen results.

Give a complete but minimal `PRO_P0_BLOCKER_DECISION-v001` containing:

1. `SESSION_ASSERTION`, `MODEL_EFFORT_ASSERTION`, `ROLE_ACKNOWLEDGMENT`, and
   `CONTEXT_USED` with the required fields above.
2. `HISTORY_SYNTHESIS`: distinguish the accepted v002 route from all historical
   DUCA routes and the unexecuted P0 evidence.
3. A single `SCIENTIFIC_DECISION` from CONTINUE, REVISE, PIVOT, STOP, or
   ESCALATE_HUMAN, and exactly what that decision permits or forbids.
4. A precise mathematical and API-level P0 resolution for canonical uniform
   positions, including endpoint/tie rule, short-window `K_eff`, and whether
   constant density must be bit-identical to uniform.
5. A precise detector-agnostic, pre-NMS coordinate-transport contract: identify
   the raw representation and when mapping must occur, while preserving the
   detector/head/loss/NMS algorithm and avoiding duplicate-coordinate leakage.
6. The smallest next Builder, Critic, and Evaluator queues. State a hard time/
   scope bound for each, and whether any worktree patch should be discarded or
   resumed. No extra roles or subagents.
7. A P0/P1 stop rule; a full PRE_RUN_READY checklist for any later P2; say
   explicitly whether you authorize a concrete remote GPU experiment now. If
   yes, identify exact config, split, baseline, seed, command, budget, stop
   rule, and output path; otherwise prohibit GPU.
8. Publication/fairness/novelty risks and the full drift checklist. Do not
   invent performance or literature evidence.

Return one coherent decision, not several candidate routes.
