---
artifact_id: CRITIC_DUCA_DENSITY_UOR_FOCUSED_IMPLEMENTATION_RECHECK-v001
role: Critic
kind: UOR_FOCUSED_IMPLEMENTATION_RECHECK_V001
status: REACHABILITY_PRE_RUN_BLOCKED
loop_disposition: NEEDS_ATTENTION
finding_classification: IMPLEMENTATION_CORRECTION
scientific_ambiguity: NONE
queue_message: msg-20260814T010419Z-3798229343dc
parent_artifacts:
  - CRITIC_DUCA_DENSITY_UOR_IMPLEMENTATION_INDEPENDENT_PRE_RUN_REVIEW-v001
  - BUILDER_FOCUSED_CORRECTION_DUCA_DENSITY_UOR-v001
reviewed_revision: 6576789468c1a7692d49b2ba94a638e01e7970f4
review_binding: CLEAN_PASS_REBOUND_AS_DELIVERED_BY_COORDINATOR
review_cwd: C:/Users/skywalker/.codex/worktrees/24ef/OpenTAD_C3_CoarseClean_20260702-critic-7f07e454
execution_state: NOT_EXECUTED
scientific_evidence_status: BLOCKED_PRE_RESULT
next_owner: Coordinator
---

# Verdict

`REACHABILITY_PRE_RUN_BLOCKED`.

An equivalent deterministic launch-bypass defect remains in correction area `DUCA-REACH-IMP-002`. This is the one permitted focused correction recheck, so the implementation loop terminates as `NEEDS_ATTENTION`. No further Builder correction cycle is requested or authorized by this receipt.

# Remaining finding

## DUCA-REACH-IMP-002-R1 — DUCA `--cfg-options` bypasses the corrected entrypoint gate

Classification: `IMPLEMENTATION_CORRECTION`.

The corrected DUCA configs define strict entrypoint gates, but their gate `route` and `stage` values are `duca_density_reachability_v001` / `duca_density_*` (`configs/adatad/thumos/duca_density_detector_fit_n16r4.py:44-68`; `duca_density_reader_fit_n16r4.py:45-69`; `duca_density_cal_uor_n16r4.py:52-76`). The helper deciding whether CLI overrides require protection recognizes a gate only when `pc-ot-mras` occurs in `route` or `pc_ot_mras`/`pc-ot-mras` occurs in `stage` (`opentad/utils/training_guard.py:133-140,300-301`). None of the DUCA gate values match that predicate.

Consequently, `assert_safe_cfg_options_for_gated_config` returns immediately for every corrected DUCA config (`opentad/utils/training_guard.py:304-307`). Both entrypoints call that helper before merging arbitrary `--cfg-options`, then merge the overrides before the later gate check (`tools/train.py:69-78,108-109`; `tools/test.py:61-70,131-132`). A direct invocation can therefore supply, for example, an override that sets `pc_ot_mras_prebackbone_e2e_acquisition_gate.entrypoint_gate_context.required=False`. The later `_entrypoint_gate_context_block_reason` explicitly returns no block when `required` is false (`opentad/utils/training_guard.py:392-395`), bypassing the gate JSON, resolved-config identity, manifest, argv/cwd, Slurm/rank, annotation/root/checkpoint/workdir/artifact, and arm equalities before dataset/checkpoint/output access.

This is a reachable equivalent of the prior launch-bypass finding. It also makes the otherwise corrected FIT/CAL loader firewall mutable through the same pre-gate override path, so the package is not PRE_RUN-admissible.

# Focused disposition of the three prior areas

- `DUCA-REACH-IMP-001`: the declared config and loader-branch correction is structurally present: FIT train/val/test bindings use the FIT annotation/root/class map and `tools/train.py` skips inactive validation loaders; CAL val/test use the CAL annotation/root/class map. End-to-end closure is withheld because the remaining CLI-gate bypass can mutate those bindings.
- `DUCA-REACH-IMP-002`: not closed. `DUCA-REACH-IMP-002-R1` is an equivalent second deterministic defect and triggers terminal `NEEDS_ATTENTION`.
- `DUCA-REACH-IMP-003`: within this focused static scope, the declared correction is present: predecessor receipts are required sealed; CAL roots are recursively sealed; evaluator inputs are derived from exact declared sealed receipts/artifacts; and the final receipt is sealed in a temporary root before atomic rename (`tools/bata/duca_density_reachability.py:57-98,277-351,353-390,676-692,777-821`). No separate equivalent defect was identified before the first-fail disposition.

# Minimal simplification proposal

Do not extend the generic gate taxonomy. For this experiment, make `tools/train.py` and `tools/test.py` reject every nonempty `--cfg-options` whenever `duca_reachability_phase.protocol == "DUCA_DENSITY_REACHABILITY_PROTOCOL-v001"`, before any merge. The authored launcher already supplies the frozen runtime values through gate-bound environment variables, so DUCA needs no CLI override surface. This proposal preserves the frozen protocol and scientific route, but this terminated loop does not authorize its implementation.

Dependency: Coordinator records `NEEDS_ATTENTION` and decides whether to stop or obtain explicit authority for a simplified replacement implementation/review sequence. Do not dispatch another correction round under the current loop identity.

# Review boundary and commands

The Coordinator-delivered clean rebind at `6576789468c1a7692d49b2ba94a638e01e7970f4` was consumed as the frozen identity. The review read only the queue, required governing documents, prior Critic receipt, Builder correction receipt, and declared changed source surfaces. Commands/tests executed: none. No Git operation, Python/import/test, data or dataset/root access/listing, official validation, checkpoint/model/prediction, remote, GPU/CUDA/Slurm, training, inference, evaluator/bootstrap, metric, browser/Sources/Pro, claim, route, repository edit, cleanup, or role dispatch occurred. This durable Critic receipt is the sole write.

CRITIC_DECISION: REACHABILITY_PRE_RUN_BLOCKED.

LOOP_DISPOSITION: NEEDS_ATTENTION.
