---
queue_name: EVALUATOR_DUCA_P0P1_AMENDMENT-v001
parent_decision: PRO_P0_BLOCKER_DECISION-v001
stage: DRAFT/P0
fixed_revision: 63a726a4aaf48ecbf6780bb196de43a890c6b4df
execution: prohibited
---

# Evaluator P0/P1 amendment

Write a protocol-only amendment. Do not inspect data or models, execute CPU/GPU
commands, compute metrics, modify training code, access validation/test, or
start remote work. Keep `PRE_RUN=BLOCKED`.

Specify the exact future *remote-only* P1 commands and receipt schema required
to establish: canonical endpoint identity; constant-density bit identity;
valid length / `K_eff` contract; strict monotonicity/uniqueness; five boundary
parity cases; coordinate-tag and double-map failures; pre-NMS order for all
non-sliding paths; and detector/config invariance. State the command inputs,
expected receipt fields, failure signatures, evidence category, and stop rule.
Do not authorize or execute P1. Return a durable amendment and a concise
`PRE_RUN_BLOCKED` conclusion.
