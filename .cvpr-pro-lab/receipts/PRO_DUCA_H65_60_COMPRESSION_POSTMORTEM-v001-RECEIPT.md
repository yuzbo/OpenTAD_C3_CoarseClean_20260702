# PRO_DUCA_H65_60_COMPRESSION_POSTMORTEM-v001 — FINAL RECEIPT

- status: `COMPLETED / ACCEPTED`
- dispatch_id: `CENTRAL-ARIS-PRO-DUCA-H65-60-COMPRESSION-POSTMORTEM-v001`
- exact_project: `DUCA-RIME: Dynamic-Budget Task-Aware Temporal Acquisition for Offline TAD` / `g-p-6a796fef9a00819194024cf1de3bd697`
- nonce: `DUCA-H65-60-COMPRESSION-POSTMORTEM-v001-20260824`
- github_revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
- oracle_job/session: `j-vh5n9f` / `duca-h65-60-postmortem-v1`
- conversation_id: `6a8c2526-9864-83ea-b1cc-12ddeb13a906`
- conversation_url: `https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c2526-9864-83ea-b1cc-12ddeb13a906`
- runtime: profile `61`, CDP `127.0.0.1:14106`, target `EBB19E1BEC9C7FD1FFC489B345337F90`
- model: requested `gpt-5-pro`; resolved `Pro`; picker verified; `MAX_EFFORT_NOT_SEPARATELY_EXPOSED`
- submitted_at: `2026-08-24T19:03:59+08:00`
- completed_at: `2026-08-24T19:19:38+08:00`
- hard_timeout_at: `2026-08-24T21:03:59+08:00`
- actual_attempt_count: `1`
- raw_log: `C:/Users/skywalker/.fastctx/jobs/j-vh5n9f/output.log`
- prompt: `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_60_COMPRESSION_POSTMORTEM-v001.md`
- final_visible_report: `.cvpr-pro-lab/pro-reviews/runs/duca-h65-60-compression-postmortem-v001/PRO_DUCA_H65_60_COMPRESSION_POSTMORTEM-v001.md`
- oracle_meta: `C:/Users/skywalker/.codex/oracle/duca-h65-60-compression-postmortem-v001/sessions/duca-h65-60-postmortem-v1/meta.json`

## Scientific decision

`STOP` applies only to the claim that the frozen H65 `30+60` training can be
compressed without material loss into `60` total epochs by changing the shared
Stage-2 learning-rate decay. It does not reject H65 semantic indirect frame
selection.

The terminal evidence closes that hypothesis: restoring the mature 30-epoch
Stage-1 and testing both AM-RPCH25 and a higher-exposure LongCosine schedule
produced `63.22/41.25` and `63.56/41.01` Avg-mAP/mAP@0.7, both below the frozen
recovery neighborhood and both without a rising terminal Avg-mAP tail. The higher
LongCosine exposure improved Avg-mAP by only `0.34 pp` and reduced mAP@0.7 by
`0.24 pp` relative to AM-RPCH25. This is sufficient to reject further LR-decay,
warmup, hold, terminal-factor, or Stage-ratio sweeps for this subproblem.

The strongest current explanation is missing Stage-2 joint exposure: the
compressed 30+30 route removes 3000 successful minibatch/AdamW/EMA updates,
shortens the semantic/policy and detector-feedback transitions, and reduces the
fully joint tail from about 3000 to about 1000 updates. EMA lag and boundary
support are admissible secondary hypotheses but remain unmeasured and may not be
assigned numerical causal shares.

## Bounded handoff

- frozen performance recipe: Stage-1 `30 epochs / 3000 successful updates` plus Stage-2 `60 epochs / 6000 successful updates`, terminal epoch-59 EMA, no intermediate checkpoint selection
- Builder: `IDLE`; no new schedule/config/launcher or training
- next_owner: `Independent Evaluator / read-only analysis owner`
- next_action: align existing 30+60, 20+40 and 30+30 artifacts at Stage-2 updates 2000/2500/3000; examine historical trajectory, online-versus-EMA, unweighted boundary losses, parameter-group movement, and selector boundary coverage; then issue a terminal causal postmortem
- dependency: immutable existing checkpoints/logs/update audits plus official evaluator, split, NMS, annotation and state-key identities
- expected_return_at: after the bounded read-only artifact analysis; before any new model/config/training action
- single_recovery: `none`

No new training or experiment is authorized by this receipt. A future physical-
time or representation experiment must use the frozen 30+60 recipe and isolate
that mechanism from schedule, EMA, and dynamic-K changes.
