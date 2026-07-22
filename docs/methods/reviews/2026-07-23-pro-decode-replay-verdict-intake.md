# PhysTime Decode-Replay Pro Verdict Intake

## Source and status

- External review attachment: `69e90dee-f3f3-4276-a89d-97be17ac92ba/pasted-text.txt`
- Attachment SHA256: `28C6D00404B7530D5A85E27538FA3EFBE07021ACDC54AEA713E0E4222EA79CC1`
- Intake date: 2026-07-23
- Scope: frozen decode cross-replay only. This is not a training result, and it
  does not change the status of the full60 result.

## Accepted decision

The review is accepted as `REVISE_BEFORE_REQUEUE` / `HOLD`. The next formal
implementation must use **A-STRICT-SOURCE-DTYPE**:

1. Preserve `cls_scores` in the capture artifact at its source dtype. The
   observed failing production condition is `torch.float16 -> numpy.float16`.
2. Replay thresholding, flattening, sorting, and pre-NMS top-k with that same
   CPU source score dtype. The direct path performs these operations after
   `.detach().cpu()`.
3. Do not change `SingleStageDetector.post_processing`, Soft-NMS, evaluator,
   checkpoint, model, or training protocol in this repair.
4. Continue to reconstruct proposals from captured score/regression/point
   tensors. Captured `native_proposals` remain audit-only and must never
   substitute for reconstructed proposals.

The alternative of a shared stable total order is rejected for this repair.
It is a new production inference semantic and would require fresh P0/full60
inference anchors before its numbers could be compared to legacy results.

## Local verification of the review

Verified against the current frozen source:

- `AnchorFreeHead._capture_decode_replay_state` records source dtype metadata
  but explicitly casts `cls_scores` to CPU `float32`.
- `replay_phystime_decode_cross.validate_array_semantics` and the independent
  validator require captured scores to be `float32`.
- `SingleStageDetector.post_processing` moves scores to CPU, filters, calls
  `pred_prob.sort(descending=True)`, and applies `pre_nms_topk` before the
  sliding-window cross-video NMS stage.
- The failed real gate used a sliding-window path; therefore the first observed
  mismatch is a pre-NMS score-order/top-k contract failure, not evidence of a
  Soft-NMS implementation failure.

Existing remote counts, hashes, and first-difference indices are retained as
project forensic evidence, not re-measured by this intake. The original remote
NPZ and Slurm artifacts must remain available for an independently reproducible
read-only verification.

## Narrowing applied during intake

The following are accepted as required for the requeue:

- schema-v2-style per-tensor source/stored/replay dtype provenance;
- hard ordered-result equality, plus diagnostic top-k candidate-id order/set
  hashes and failure artifacts written before a gate raises;
- v1 rejection when an ordering-sensitive source-fp16 score was stored as
  fp32;
- a fixed runtime fingerprint for the legacy CPU tie-order semantics;
- source-score-dtype/tie-boundary focused tests and P0 direct re-anchoring.

The following are good hardening recommendations, but are not independently
established as the minimal scientific fix: requiring source-dtype transport for
every non-score float tensor, cross-endian canonical hashes, and full
filesystem-durability fsync. They must be implemented only if they can be
covered by focused tests without changing frozen production semantics.

## Required deployment sequence

1. Implement the narrow source-score-dtype repair and diagnostic contract in a
   new commit/tree. No model or training change.
2. Run local focused tests, then target-cluster CPU preflight.
3. Run capture-off versus capture-on direct invariance for selected/physical x
   online/EMA.
4. Run the four-condition real CUDA native exact gate using a new snapshot,
   run root, and DAG token.
5. For each formal replay, re-anchor capture-enabled direct output to the
   reviewed P0 output before cross-axis decode.
6. Run the formal replay suite only after all four gates pass.

Failure at any point remains fail-closed: formal decode-cross mAP stays `NA`,
and Q192 UU/UP/PU/PP, Q-lift, new sampling, loss, assignment, NMS, and training
remain frozen.
