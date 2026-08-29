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

The non-scientific `PRECHECK_ONLY=1` submission was rejected by Slurm before job creation with `AssocMaxSubmitJobLimit`. It therefore has no Job ID, consumed no scientific execution and did not increment the formal submission count. At detection time the shared account had 16 visible queue entries, including an unrelated 15-element running array and unrelated pending jobs. No unrelated job is modified or cancelled.

- Precheck state: `WAITING_SHARED_SUBMIT_SLOT`
- Formal `sbatch` count: `0 / 1`
- Result root created: `false`
- Training/evaluation started: `false`
- Machine-side slot waiter: FastCtx `j-00wllb`, real `sleep 300`, emits only after the visible queue count decreases

When a slot becomes available, the same exact candidate will run one Slurm precheck. Only `PRECHECK_READY` authorizes the sole formal seed-42, 60-epoch, two-GPU training plus epoch-59 `state_dict_ema` official validation submission. A formal failure returns to fresh Pro without retry, resume or replacement; no cost is authorized.
