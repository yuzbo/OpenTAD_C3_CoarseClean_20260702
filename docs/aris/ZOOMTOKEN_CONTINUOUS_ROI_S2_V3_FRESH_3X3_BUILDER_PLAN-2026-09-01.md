# ZoomToken Continuous-RoI S2-v3 fresh 3×3 Builder plan

## Frozen assignment

- Task: `ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FRESH-3X3-MATCHED-TRAIN-REFERENCE-AND-COST-FALSIFIER-v001`
- Pro decision: `PIVOT`; role contract: `KEEP`
- Exact Project conversation: `6a961c52-1b5c-83e9-b04c-6b0bda18892f`
- Execution base: `10aed28659a08fa703def278fc0f5f1422dcad89`
- Branch: `codex/zoomtoken-continuous-roi-s2-v3-fresh-3x3-v001`
- Scientific identity: a new D160-V3/G96-V3/U128-V3 × seeds 4407/4408/4409 campaign. It is not recovery, continuation or reconstruction of the permanently closed historical exact-nine route.

## Minimal implementation

The existing D160/G96/U128 transforms, model, VideoMAE, Adapter, ActionFormer, detector, evaluator, postprocess and NMS remain read-only. The nine new source configs inherit the three frozen family configs and change only the fresh task identity and seed-bound work namespace.

`tools/bata/run_continuous_roi_s2_v3_fresh_3x3.py` is the task-local control plane. It will:

1. validate the new protocol self-hash, exact Git identity, complete 160/40 development split, official-test seal and immutable fresh namespace;
2. bind the existing successful-update-aware training machinery to the v3 seed set and the nine new source configs inside the same training process, without changing existing training or model source;
3. submit or execute exactly one named cell at a time, require 60 epochs × 80 successful updates = 4,800 updates, and retain only the epoch-59 final EMA checkpoint plus sidecar, rendered config and completion receipt;
4. strict-finalize all nine cells before any reference work and publish one immutable training-matrix completion receipt;
5. enforce the single-campaign submission budget and refuse retry, resume, replacement, second seed set or second campaign.

The task-local binding bridge must be self-auditing: the rendered config and checkpoint metadata carry both the inherited Continuous-RoI training-contract identity and the fresh v3 protocol/task/cell identity. A new process must reconstruct the same binding byte-for-byte before accepting a checkpoint. No generic unbound `tools/train.py` execution is permitted.

## Reference and claim closure

`tools/bata/evaluate_continuous_roi_s2_v3_reference.py` inherits the v2.2 known-answer semantics:

- Torch `2.0.1` Sobol, dimension 48, scramble enabled, seed `20260720`, one anchor plus sixteen candidates;
- twelve knots interpolated to 48 tubelets, common physical centers for fixed-size and variable-size arms;
- complete frozen fit/gate 160/40 development split and sanitized 129 ordered reference windows;
- raw execution with no annotation, GT, teacher, preferred ID or raw-prediction shortcut;
- canonical raw seal before a separate privileged CPU join;
- paired two-level bootstrap with 20,000 replicates and unchanged `S_CR`, `H`, `F`, boundary, short-action and missing-evidence rules.

Three raw-reference submissions correspond to the three fresh seeds. One privileged join/reference finalizer may run only after all three raw receipts and all nine training completions validate. Partial cells, partial windows and intermediate checkpoints are not interpreted.

## Conditional full-stack cost

`tools/bata/profile_continuous_roi_s2_v3_cost.py` is admitted only after a valid reference finalizer returns `S_CR=true` and `H=true`. The single cost submission uses all three final EMA seeds on the same physical GPU and the same complete frozen evaluation population for every arm. It measures:

`decode → crop/resize → H2D → full model/detector → postprocess → full-video Soft-NMS`

It reports p50, p95, throughput, peak allocated/reserved memory and gross GPU energy. Exhaustive 17-candidate reference-search cost is separately labeled and cannot be substituted for deployable-policy cost. `F=false` stops efficiency continuation even if crop science survives.

## Allowed candidate surface

Only the following paths may be added on the code branch:

1. `docs/methods/continuous_roi_s2_v3_fresh_3x3_protocol.json`
2. nine files under `configs/adatad/thumos/continuous_roi_s2_v3_fresh/`
3. `tools/bata/run_continuous_roi_s2_v3_fresh_3x3.py`
4. `tools/bata/evaluate_continuous_roi_s2_v3_reference.py`
5. `tools/bata/profile_continuous_roi_s2_v3_cost.py`
6. `scripts/run_continuous_roi_s2_v3_fresh_3x3_n16r4.sh`
7. `tests/test_continuous_roi_s2_v3_fresh_3x3.py`

No existing model, dataset, transform, VideoMAE, Adapter, ActionFormer, evaluator, NMS, training module or old Continuous-RoI artifact is edited or reused as a fresh result.

## Fail-closed verification

Focused tests and the result-blind pre-run review must prove:

- exact base/candidate identity and clean checkout;
- exactly the nine frozen family/seed cells and no other cell;
- full official training population prescribed by the frozen development manifest, 60 complete epochs, 80 successful updates per epoch and 4,800 updates per cell;
- epoch-59 `state_dict_ema`, optimizer/scheduler closure, sidecar/config/completion identity and strict real-model load;
- complete matched reference population/order and raw/privileged no-leak boundary;
- no official-test opening and no validation/test GT in raw selection;
- atomic receipts and the exact 9+1+3+1+conditional-1 submission budget;
- objective failure produces a terminal blocker without replacement or scientific extrapolation.

Formal training or evaluation on a subset, short run, truncated loader, smoke population or intermediate checkpoint is engineering-only and cannot satisfy this task.

## Decision and waiting rules

- Acceptance: `S_CR=true && H=true && F=true`, all three seeds complete and every identity/no-leak/population/evidence gate passes.
- `S_CR=false`: stop this exact S2-v3 representation; no S3.
- `H=false`: stop variable-size/S3.
- `F=false`: preserve valid crop science, stop efficiency continuation.
- Incomplete evidence: `NO_DECISION_INVALID_EVIDENCE`.
- Formal campaign limit: 9 training + 1 training finalizer + 3 raw reference + 1 privileged join/reference finalizer + at most 1 conditional cost = at most 15 submissions; replacement `0`, second campaign `false`.
- Once a long job is accepted and no immediate step remains, use the single terminal waiter or a real silent 600-second terminal sleep. During the timer window there is no output, status query, file/browser action or second waiter. Each wake performs one authoritative terminal check only.

## Beijing deadlines

- Builder plan and role-rule sync: `2026-09-01 12:00`
- Clean candidate: `2026-09-02 06:00`
- Critic: `2026-09-02 09:00`
- Evaluator: `2026-09-02 12:00`
- Formal action: `2026-09-02 14:00`
- Terminal evidence: within four hours of the last authorized terminal and no later than `2026-09-06 12:00`
- Mandatory fresh post-result Pro: `2026-09-06 16:00`

The role contract is `KEEP`; no role-file revision is made.
