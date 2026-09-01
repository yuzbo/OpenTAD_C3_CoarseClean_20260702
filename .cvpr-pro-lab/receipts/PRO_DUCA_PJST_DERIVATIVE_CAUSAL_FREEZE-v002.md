# PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002

- status: COMPLETED / ACCEPTED
- scientific_decision: REVISE
- frozen_route: PJST-D1 — Derivative-Only Physical-Jacobian Scaled Tubelet
- causal_estimand: fixed-selector representation attribution
- project_id: g-p-6a796fef9a00819194024cf1de3bd697
- nonce: DUCA-PJST-DERIVATIVE-CAUSAL-FREEZE-v002-20260825
- model: gpt-5.5-pro
- oracle_job: j-7rkw7z
- oracle_session: duca-pjst-derivative-v2
- conversation_id: 6a8dace6-e798-83ea-b112-d5636c54fb62
- conversation_url: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8dace6-e798-83ea-b112-d5636c54fb62
- submitted_at: 2026-08-25T14:55:22.291Z
- completed_at: 2026-08-25T15:25:36.609Z
- frozen_revision: b2ccfccab5b4912b59954afcc9b0364955327f7c

## Routing and artifact proof

- Saved report: `.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md`
- Saved metadata: `.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/oracle-home/sessions/duca-pjst-derivative-v2/meta.json`
- Saved transcript: `.cvpr-pro-lab/pro-reviews/runs/duca-pjst-derivative-causal-freeze-v002/oracle-home/sessions/duca-pjst-derivative-v2/artifacts/transcript.md`
- User attachment and saved report are text-identical after newline normalization.
- Metadata binds the completed turn, nonce and conversation to the exact DUCA Project.

## Frozen implementation boundary

The transform preserves the pair mean and scales only the pair difference by canonical gap divided by
actual physical gap. Exact-uniform rows bypass the transform before cast or division. Support/Voronoi
quantities are audit-only and cannot affect forward or gradients.

The first formal comparison must train matched frozen-selector OFF and PJST-D1 ON from the same Stage-1
terminal checkpoint. Selected positions, RGB, masks, random exposure order and executed K must be identical;
the sole intervention is the pre-PatchEmbed PJST-D1 transform.

## Evidence boundary and handoff

This receipt freezes a design and authorizes its bounded implementation chain. It is not PRE_RUN, training,
efficacy, cost or paper evidence. Next owner is a clean Builder at the frozen revision; an independent Critic
must pass the frozen snapshot before Evaluator PRE_RUN. No formal experiment starts before those gates.
