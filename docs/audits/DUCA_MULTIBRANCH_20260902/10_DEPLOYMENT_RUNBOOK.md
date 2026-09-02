# DUCA Multi-branch Deployment Runbook

This runbook is the execution contract for the six frozen routes. It is
fail-closed: an admission receipt is not a training result, and a result from
an old, later, dirty, or patched checkout is not assigned to a frozen SHA.

## Immutable sources

| Route | Branch | Frozen SHA | GitHub commit | Current release state |
|---|---|---|---|---|
| H65-Pro Full Matrix | `codex/h65-pro-fullmatrix-strict60-20260902` | `cfb7041d876f6e38e9ef6ce77cef7cee04b79659` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/cfb7041d876f6e38e9ef6ce77cef7cee04b79659> | Conditional; P0 signature-routing failure found at exact SHA |
| DUCA Unified Full Matrix | `codex/duca-unified-fullmatrix-20260902` | `89b9ea3e8e018b41034917ee14de7f409354a7e9` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/89b9ea3e8e018b41034917ee14de7f409354a7e9> | Blocked until Taylor/H65/cost mechanisms are real |
| DUCA Evidence Recovery | `codex/duca-evidence-recovery-numerical-correction-20260902` | `08d425a259fc468dde7c496e77b4c43e953d8d0c` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08d425a259fc468dde7c496e77b4c43e953d8d0c> | C0/parity admission only |
| DUCA CT-DP-BAMoD | `codex/duca-ctdp-geometry-mechanism-correction-20260902` | `2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc> | Geometry/gradient admission only; G0/G1 factorization conflict found |
| ZoomToken BAFDR | `codex/zoomtoken-bafdr-gradient-correction-20260902` | `fdeaeb98340bf7070201a02feb8093f50486aeaa` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdeaeb98340bf7070201a02feb8093f50486aeaa> | Five-arm screen only |
| ZoomToken ET-TRC | `codex/zoomtoken-et-trc-correction-20260902` | `59eab0c6aaacf5039d2ae20969a6dd5772bcb80f` | <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59eab0c6aaacf5039d2ae20969a6dd5772bcb80f> | Checkpoint and real two-GPU DDP admission only |

The local audit worktrees are under `E:/DeskTop/TAD/_duca_audit_worktrees/`.
The corresponding clean remote worktrees are under
`/data/run01/sczc063/yuzibo/projects/duca_multibranch_audit_20260902/`.
Every submission must check the full SHA, required branch name where the
launcher requires it, and an empty `git status --porcelain`.

The review-only correction branches are also pushed and are intentionally
separate from the frozen sources: H65 signature routing
`codex/h65-pro-admission-fix-20260902` at
`78cde6aa5335b2e399e597ce9229d8657e6760a5`, CT-DP factorization
`codex/duca-ctdp-admission-fix-20260902` at
`d62cab763c8e0478e73c6c47a4c185db45164dda`, and Unified fail-closed gates
`codex/duca-unified-admission-gates-20260902` at
`98d559ee414504caaa480294ce4d066276cdebe6`.

## N16R4 environment

Use an allocated Slurm job, never the login node for training:

```bash
source /etc/profile
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
export YUZIBO_ROOT=/data/run01/sczc063/yuzibo
export ETTRC_PRETRAIN=$YUZIBO_ROOT/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
```

The observed checkpoint SHA256 is
`4b96b7f403f8ae0396437855b785af6a0064f11a9d76e2268e5a76a04e0de251`.
The observed annotation hashes are recorded in
`01_ADATAD_REFERENCE_CONTRACT.json`. Train/held-out list hashes and the full
AdaTAD inheritance/evaluator contract are still required before formal release.

## Supervisors

The remote minute supervisor is `scripts/duca_remote_supervisor.py` with the
empty, operator-populated `scripts/duca_remote_supervisor_queue.json`. It is
installed at
`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902/`
and runs as a single `nohup` process with `--interval 60`. It writes
`supervisor_state.json`, `latest_receipt.json`, timestamped receipts, and
`logs/supervisor.log`; `supervisor.pid` identifies the active process. The
queue is intentionally empty while the dispatcher is blocked. Adding an entry
does not bypass the dispatcher, exact-SHA, clean-tree, or route-specific
launcher checks. Retryable scheduler failures are bounded; code or contract
failures become `NEEDS_REPAIR`.

The local Codex heartbeat automation
`monitor-duca-recent-remote-experiments` runs every 30 minutes. It collects the
remote supervisor receipt, `squeue`/`sacct`, owned logs/checkpoints/receipts,
and the result ledger. It may perform a minimal correction only after
preserving the first failure, creating a new correction SHA, running focused
tests and prechecks, and updating the remote queue. It must never rewrite a
frozen SHA or adopt pre-existing jobs.

## Mandatory failure-handling rule

Every failed submission, failed run, and protocol/invocation error follows the
same ordered contract:

1. Preserve the attempt and read complete launcher/Slurm `stdout` and `stderr`
   before any retry. Record attempt/job ID, exact command, checkout, source
   SHA, work directory, and log paths in `08_SLURM_LEDGER.json` and the failure
   diagnosis receipt.
2. Classify the cause as code, protocol/invocation, resource/scheduler,
   numerical, data, or environment. Code/protocol/environment repairs use a
   separate correction branch and commit; frozen SHAs are immutable and old or
   dirty checkouts cannot donate results.
3. Run the route-specific focused tests and corresponding `PRECHECK_ONLY` or
   admission precheck from a clean checkout with the documented N16R4
   environment. A passing precheck authorizes a launch attempt only; it never
   supplies a metric or scientific result.
4. Resubmit only after the repaired precheck passes and Slurm resources are
   available. Retries are bounded and use the same corrected SHA. Account,
   association, or partition limits produce `BLOCKED_RESOURCE`, not duplicate
   submissions or code changes.
5. Keep every failure, cancellation, non-finite loss, missing checkpoint,
   missing terminal receipt, and wrong-checkout completion in the ledger. A
   Slurm `COMPLETED` state or an epoch/`Training Over` line without terminal
   receipts is not a result. No mAP, speedup, cost, or bootstrap claim is
   allowed without the exact-SHA, clean-tree, terminal checkpoint, evaluator,
   and aggregation receipts.

The supervisor and 30-minute heartbeat must report unresolved failures and the
next action; silently dropping, relabeling, or adopting a pre-existing job is
forbidden.

## Ordered admission plan

1. **Shared identity gate.** Resolve the data lists, annotation, checkpoint,
   model/config inheritance, optimizer groups, evaluator commit, physical
   coordinate unit, and resource identity. Emit one immutable contract and a
   per-cell parity receipt. A missing field is `BLOCKED_GATE`.
2. **H65-Pro.** On the frozen SHA run the validator, CUDA forward/backward,
   physical-coordinate reduction, optimizer coverage, strict `(60, 6000)`
   counter/resume, Slurm port smoke, and cost preflight. The exact-SHA P0
   admission currently fails because an x-only proof backbone receives
   `masks`. Validate the signature-aware patch in the separate
   `codex/h65-pro-admission-fix-20260902` worktree, then create and re-audit a
   new frozen SHA; do not attach patched results to `cfb7041d`.
3. **Evidence Recovery.** Run the exact-branch precheck for seed `8261`, then
   C0 numerical parity (indices, physical positions, features, logits, loss,
   decode, final predictions), all-mask/mixed-mask CUDA, non-finite replay, and
   resume. Only a terminal C0 plus CUDA receipt can release downstream cells.
4. **CT-DP-BAMoD.** First prove G0--G3 geometry and uniform reduction, then
   finite-difference CT/eta gradients and batch/DDP behavior. The frozen
   factorization is inconsistent with the declared G0/G1 contract; the
   correction branch `codex/duca-ctdp-admission-fix-20260902` at
   `d62cab763c8e0478e73c6c47a4c185db45164dda` is a candidate correction, not a
   result for the frozen SHA. Re-freeze after review.
5. **BAFDR.** Run the gradient gate and the five-arm selection screen on the
   disjoint selection split. `BAFDR-K16-FULL` requires a same-seed terminal
   D160 epoch-59 EMA Teacher containing `state_dict_ema`, its config SHA, and
   the Teacher source commit. Until the screen finalizer returns `PASS`, the
   21-cell matrix and all metrics remain closed.
6. **ET-TRC.** Verify VideoMAE checkpoint/module coverage and one-GPU load,
   then run the actual OFF/ON pair with `torchrun --standalone
   --nproc_per_node=2`, global batch 2, and a resume test. Static launcher tests
   alone are insufficient. No OFF/ON metric is valid without terminal EMA
   checkpoints and independent evaluator receipts.
7. **DUCA Unified.** Implement and test real Taylor P0/P1 objective wiring,
   original H65 retention/transition semantics, and measured end-to-end cost.
   The generator and submitter must report `READY` before any of the 41 cells
   can be submitted.

## Formal DAG release

The canonical manifest is `06_FULL_DAG_MANIFEST.json`. It uses at most eight
in-flight submissions, `afterok` for scientific dependencies, Slurm-provided
`CUDA_VISIBLE_DEVICES`, a master port derived from job/array IDs, and terminal
receipts that cancel descendants on submission failure. The current dispatcher
(`tools/bata/dispatch_duca_multibranch.py`) returns `BLOCKED` and no job IDs;
this is the expected behavior while any admission or implementation gate is
closed.

After all gates pass, release in this order: H65 and Evidence anchors; CT-DP
geometry then mechanism; BAFDR screen then 21 cells; ET-TRC OFF/ON; finally the
Unified 41-cell matrix. Every cell records source SHA, config/data/checkpoint
hashes, declared changed variables, world size/global batch, successful-update
target, dependency IDs, run root, terminal EMA path, evaluator output, and
cost receipt.

## Evaluation and analysis

Training is exactly 60 completed epochs and 6000 successful optimizer updates
for strict-60 cells. Evaluation uses terminal EMA checkpoints and the official
held-out split only after selection is sealed. Metrics are rerun by the pinned
evaluator; bootstrap resamples paired videos/seeds and archives indices. Cost
is measured end to end with synchronized hardware/software identity. Until
those artifacts exist, the result ledger remains `NO_VALID_RESULTS` and no mAP,
speedup, bootstrap interval, or cost claim is reported.

## Current execution ledger

Successful non-training admissions already recorded on N16R4:

- `1266242`: generic CUDA probe;
- `1266268`: H65 exact-SHA focused CUDA tests, `15 passed`;
- `1266269`: Evidence exact-SHA focused CUDA tests, `35 passed`;
- `1266271`: CT-DP exact-SHA focused CUDA tests, `7 passed`;
- `1266317`: BAFDR/ET protocol tests, `11` and `10 passed`.

H65 P0 job `1266323` failed on the real signature mismatch. The corrected
rerun was attempted but Slurm returned `AssocMaxSubmitJobLimit`; no formal
training job was submitted by this audit. Existing BAFDR, ET-TRC, CT-DP, and
Evidence jobs use other heads or dirty checkouts and are explicitly not owned
by this audit; they are not cancelled and cannot contribute results here.

The authoritative machine-readable status is in `00_IDENTITY.json`,
`05_ADMISSION_MATRIX.json`, `06_FULL_DAG_MANIFEST.json`,
`08_SLURM_LEDGER.json`, and `09_RESULT_LEDGER.json`.
