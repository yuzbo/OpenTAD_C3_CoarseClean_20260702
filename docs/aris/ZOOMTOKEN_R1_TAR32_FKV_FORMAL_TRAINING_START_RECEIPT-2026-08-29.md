# ZoomToken R1-TAR32-FKV formal training start receipt

## Scientific authority and boundary

- Pro decision: `CONTINUE_COMPOSITE_PROBE`
- task: `ZT-CODEX-COMP-PROBE-001`
- experiment: `ZT-R1-TAR32-FKV-S42-E60-v001`
- status: composition-first probe; method novelty, efficacy and efficiency are not established
- isolation: this task does not replace, modify, delay or consume partial evidence from BPNS-R1 v004 job `1260095`

The frozen mechanism keeps current-frame R1 contiguous `8x8/K64` support and the existing
full-K64 Adapter. Blocks `0/2/4/6/8/10` update K64. Each odd block uses only the immediately
preceding dense block's attention-column mean to select exact stable K32 per tubelet for
Query/output/MLP, while all K64 remain non-detached Key/Value context. No cache, temporal
state, new parameter, new loss, dynamic cardinality, fallback or rescue was added.

## Source identity

- exact base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- branch: `codex/zoomtoken-r1-tar32-fkv-v001`
- clean/pushed candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_src_b0a1ca11`
- remote source verification: exact HEAD, remote-tracking ref and clean worktree matched

The candidate changes only the frozen implementation surfaces:

- `opentad/models/backbones/vit_adapter.py`
- `configs/adatad/thumos/georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py`
- `scripts/run_zoomtoken_r1_tar32_fkv_n16r4.sh`
- `tests/test_zoomtoken_r1_tar32_fkv.py`
- one literal route allowlist addition in `tools/train.py`

## Verification and independent review

- N16R4 TAR32 plus inherited R1/MOD32/A-MoD focused suite: `32 passed, 1 skipped`
- independent strict-R1 regression process: `9 passed`
- fresh independent Critic: `PASS`
- fresh result-blind Evaluator: `PRE_RUN_READY`
- pre-run job `1260163`: `COMPLETED 0:0` on `g0048`
  - canonical THUMOS14 training dataset: 200 videos
  - real batch tensor: `[1,1,3,768,160,160]`; mask: `[1,768]`
  - CUDA AMP forward/backward finite
  - frozen route ledger: `[64,32,64,32,64,32,64,32,64,32,64,32]`
  - fallback/failure: `0/0`
  - no accuracy, prediction or evaluator result was read

Job `1260162` is separately sealed as an operational wrapper failure: `/bin/sh` exited 127
before candidate Python/CUDA/data execution. It is not a pre-run or scientific result and
was replaced once by the successful result-blind job above.

## Formal training

- Slurm job: `1260166`
- JobName: `zt-r1-tar32-fkv-s42-e60-v001`
- actual formal submissions: `1`
- submitted: `2026-08-29 02:29:35 +08:00`
- started: `2026-08-29 02:29:38 +08:00`
- node: `g0059`
- resources: 2 GPUs, 8 CPUs, 16-hour walltime
- seed: 42
- local/global batch: 1/2
- actual epochs / scheduler horizon: 60 / 100
- AMP/EMA: enabled; frozen epoch-59 EMA is the only primary accuracy result
- result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_v001_seed42_20260830`
- log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_tar32_fkv_logs_b0a1ca11_20260830`

Machine-side terminal-only waiting is delegated to FastCtx background job `j-ucv5ag` at a
300-second interval. Codex does not poll the formal job in the foreground and does not read,
summarize or interpret live/partial accuracy, loss, prediction, cost or route values.

## Terminal decision boundary

The `79.06%` heavy-block figure is architecture accounting only. Formal training terminal
accuracy and a later probe-grade matched full-stack cost package are both required before
any efficacy or efficiency statement. Every terminal state, including a controlled or hard
failure, must be preserved and returned to one fresh Pro review. There is no automatic
successor, retry, resume, second seed, sweep or auxiliary arm.
