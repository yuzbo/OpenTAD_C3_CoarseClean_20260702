---
artifact_id: CRITIC_DUCA_UOR_SEALED_EXECUTION_SURFACE_TERMINAL_REVIEW-v001
role: Critic
kind: DUCA_UOR_SEALED_EXECUTION_SURFACE-v001-TERMINAL_REVIEW
status: SEALED_REPLACEMENT_BLOCKED
terminal_disposition: STOP
queue_message: msg-20260814T025512Z-f89eee3c9f47
parent_artifacts:
  - PRO_DUCA_UOR_TERMINAL_DISPOSITION-v001
  - BUILDER_DUCA_UOR_SEALED_EXECUTION_SURFACE-v001
reviewed_revision: 6515ebf501b3c49be57ad4c37fc84d2fb4ae66d7
review_binding: CLEAN_READ_ONLY_AS_DELIVERED
execution_state: NOT_EXECUTED
evidence_class: STATIC_READ_ONLY_TERMINAL_REVIEW
next_owner: Coordinator
next_action: RECORD_TERMINAL_STOP_AND_KEEP_EVALUATOR_DORMANT
---

# Verdict

`SEALED_REPLACEMENT_BLOCKED`.

The additive replacement keeps the old generic train/test/launcher path unreachable, uses a closed top-level manifest, exposes literal dedicated entrypoints, and imports no Torch/mmengine/OpenTAD/data/evaluator surface. It nevertheless fails the required phase-order and evaluator-receipt attack: read-only mode is treated as receipt authenticity while the receipt's closed semantics are not validated.

# Terminal finding — incomplete sealed-receipt binding

`tools/duca_uor_sealed_run.py:108-128` accepts a predecessor after checking only that its path is a non-writable file and that four fields match: `protocol`, `schema_version`, `phase`, and `manifest_sha256`. It does not require the exact receipt fields and invariant values emitted by `build_phase_receipt` at lines 80-105, including:

- `execution_state="NOT_EXECUTED"`;
- `pre_run_state="PRE_RUN_BLOCKED"`;
- `official_validation_state="INACCESSIBLE_FORBIDDEN"`;
- `metric_state="EMBARGOED_NOT_COMPUTED"`;
- `decision="NOT_APPLIED"`, `result_state="NOT_AVAILABLE"`, and `claim_allowed=false`;
- the manifest-bound `artifact` with `artifact_state="NOT_PRODUCED"`;
- every no-data/no-GPU/no-remote/no-dataset/no-metric attestation;
- the phase-specific CAL arm and frozen evaluator identity.

Therefore a sealed predecessor JSON containing the four accepted identifiers but contradictory, missing, or substituted evidence fields advances the next phase and causes a new sealed receipt to be emitted. This is a direct phase-order/output-receipt and no-data-boundary bypass, not a formatting issue.

The dedicated evaluator has the equivalent terminal defect. `tools/duca_uor_sealed_evaluate.py:70-91` checks each arm receipt only for read-only mode, protocol, schema version, arm, evaluator identity, and manifest digest. It does not require the receipt's expected CAL phase, manifest-bound artifact, `NOT_EXECUTED/PRE_RUN_BLOCKED/NOT_PRODUCED` states, embargo/decision/claim fields, or no-access attestations. Three sealed minimal or contradictory arm receipts can consequently produce the atomic final `NOT_EXECUTED` receipt at lines 94-126. The final receipt can thus launder evidence that was never validated as the exact output of the dedicated runner.

The focused tests do not close this path: `tests/test_duca_uor_sealed_run.py:97-101` attacks only an absent predecessor, and `tests/test_duca_uor_sealed_evaluate.py:77-120` attacks a missing arm and evaluator-identity mismatch, not a sealed malformed/substituted/contradictory receipt.

# Other bounded attacks

- Argument/environment injection: the dedicated parsers accept only their named manifest/phase options and reject the tested generic config/checkpoint/arm/NMS/threshold/help surfaces; no environment-to-config path exists in the replacement.
- Closed manifest: the top-level, phase-binding, arm, phase-order, forbidden-key, and canonical-digest checks are present in `tools/duca_uor_sealed_manifest.py:55-206`.
- Literal launcher and old-path reachability: `tools/slurm/duca_uor_sealed_v001.sh` contains only the five literal ordered phase calls and one dedicated evaluator call. The new Python modules import only standard-library modules and the local sealed-manifest guard; they do not import generic `tools/train.py`, `tools/test.py`, the old DUCA helper, Torch, mmengine, OpenTAD, datasets, checkpoints, or the official evaluator.
- Scope fidelity: all generated receipts remain explicitly `NOT_EXECUTED`, `PRE_RUN_BLOCKED`, `EMBARGOED_NOT_COMPUTED`, and `NOT_APPLIED`; no scientific, metric, or claim output is implemented. The terminal receipt-validation defect prevents those labels from being trusted end to end.

# Terminal handoff and boundary

Next owner is the Coordinator. It must record terminal `STOP` and keep the registered Evaluator dormant. Under `PRO_DUCA_UOR_TERMINAL_DISPOSITION-v001`, this block permits no Builder response, correction, replacement amendment, Critic recheck, Evaluator intake, PRE_RUN admission, or execution.

This review consumed only the exact queue, governing documents, Builder intake, and the eight additive frozen files. Commands/tests executed: none. No Git operation, repository patch/clean, data or root access/listing, dataset, checkpoint/model/prediction, GPU/CUDA/Slurm, remote, training, inference, CAL, evaluator/metric/bootstrap, browser/Pro/Sources, claim, scientific-route change, or role dispatch occurred. This durable Critic receipt is the sole write.

CRITIC_TERMINAL: SEALED_REPLACEMENT_BLOCKED.

NEXT_OWNER: Coordinator.

NEXT_ACTION: RECORD_TERMINAL_STOP_AND_KEEP_EVALUATOR_DORMANT.
