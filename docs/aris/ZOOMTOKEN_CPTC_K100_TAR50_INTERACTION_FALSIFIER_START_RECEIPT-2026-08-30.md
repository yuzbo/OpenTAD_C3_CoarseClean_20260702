# ZoomToken CPTC K100-TAR50 interaction falsifier start receipt

## Frozen authority

- Task: `ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`
- Pro conversation: `6a930db4-fb90-83ea-ae8b-16e5028b6a45`
- Base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- Clean/pushed candidate: `fac88624723aed08175a947025a7f1d8a2af3171`
- Branch: `codex/zoomtoken-k100-tar50-interaction-v001`
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_src_fac88624`
- Frozen result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_interaction_fac88624_seed42_20260830`
- Frozen log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_k100_tar50_logs_fac88624_20260830`

The candidate is a minimal descendant: it does not change the inherited model implementation. It adds a task-specific identity alias, one precheck/formal launcher and focused protocol tests around the existing strict A-MoD capacity-0.5 route. The executed method remains native K100, `[K100,K50]x6`, flattened 800/400 tokens, immediately preceding dense attention-column ranking, full K/V, full Adapter, dense detector input and exact identity bypass, with no new parameter, loss, cache, dynamic K, fallback or residual predictor.

## Reference binding

Strict A-MoD capacity=1 job `1254040` is frozen as the route-matched reference:

- Checkpoint SHA-256: `3aca10bc3593e301b7d7e77271419b8bb557d8f8b29bead195fa2aa350e34ddd`
- Prediction SHA-256: `0d09e3fec839449923db1158a18ead631e813b9d00cdab051328cb2b407485f3`
- Reference config SHA-256: `81c805838502639d4fb0e6fcdd0848c53ccbd8eeccf7d1501562af2e84d9ac87`
- Candidate task-alias config SHA-256: `bf40b67a52ef35c58e425634ca8d888d6787570a7fe33a669f9acdb27055c110`
- Official Average mAP/mAP@0.6/mAP@0.7: `68.73/61.59/47.20`
- Canonical population: 211 videos, 792 ordered windows, 411 MP4 files

The four SHA values were recomputed on N16R4 after the exact Git revision and remote-tracking ref were verified. They match the frozen Pro assignment.

## Builder, Critic and Evaluator

- Local static protocol tests: `2 passed`.
- Local Torch suite: unavailable because the documented Windows user-site Torch fails to load `c10.dll`; this is not treated as a candidate failure.
- N16R4 focused suite: `14 passed in 42.77s`, covering the inherited exact A-MoD route and the task-specific launcher/config contract.
- Fresh independent Critic: `PASS` on candidate `fac88624...`.
- Fresh result-blind Evaluator: `PRE_RUN_READY` on the same candidate.

The first Evaluator pass correctly identified that the inherited generic config named upstream job `1245842` rather than the required route-matched job `1254040`. Candidate `fac88624...` resolves that ambiguity through a task-specific alias while retaining the original reference-config hash separately. The final Critic and Evaluator both accepted the corrected identity.

## Slurm disposition

The first non-scientific `PRECHECK_ONLY=1` submission was rejected by Slurm before job creation with `AssocMaxSubmitJobLimit`. It consumed no scientific execution and did not increment the formal submission count. No unrelated job was modified or cancelled. A single machine-side submitter then retried only this non-scientific precheck at real 60-second intervals.

The same frozen precheck was admitted as job `1261670` and reached `COMPLETED 0:0` with `PRECHECK_READY`:

- Submitted: `2026-08-30 12:34:29+08:00`
- Started/ended: `2026-08-30 12:59:38+08:00` / `12:59:45+08:00`
- Node/resources: `g0087`, 2 GPUs, 8 CPUs
- Scientific execution count: `0`

After that successful precheck, the sole authorized formal submission was created:

- Formal Job ID / name: `1261680` / `zt-k100-tar50-s42`
- Submitted: `2026-08-30 13:00:57+08:00`
- Requested resources: 2 GPUs, 8 CPUs, 16-hour walltime
- Current authoritative state at receipt update: `PENDING`, reason `AssocGrpGRES`, node unassigned
- Frozen launcher: `scripts/run_zoomtoken_k100_tar50_interaction_n16r4.sh`
- Formal `sbatch` count: `1 / 1`
- Result root created: `false`
- Training/evaluation started: `false`
- Terminal-only waiter: FastCtx `j-cz81o5`, real `sleep 300`, bound only to job `1261680`

The formal job is now the only authorized scientific execution. Queue state is not a scientific result. The terminal waiter emits only an authoritative terminal state; live loss, accuracy, prediction and intermediate validation are not read. The frozen job performs seed-42, 60-epoch, two-GPU training followed by epoch-59 `state_dict_ema` official validation. Any formal failure returns to fresh Pro without retry, resume or replacement; no cost is authorized.
