---
updated: 2026-07-07
status: active
scope: Local auditable experiment queue and launch checklist for DUCA-TAD after SSH access recovers.
out-of-scope: Running remote commands, modifying code, claiming final results, or starting new training from a login node.
---

# DUCA-TAD Experiment Queue

This note is intentionally local-only. The current SSH entry may fail with
`Permission loginCluster denied`, so no remote command should be attempted while
that condition persists. Its purpose is to make the next remote session
mechanical: first confirm read-only state, then run prechecks, then launch only
the highest-priority unlocked jobs inside a Slurm allocation.

Default remote environment after SSH recovers:

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
cd /data/run01/sczc063/yuzibo/projects/OpenTAD_GASVT_Worktree_20260706
```

Default write boundary:

- `~/run/yuzibo`
- `/data/run01/sczc063/yuzibo`
- route outputs under `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/`

Default GPU rule:

- C3 mainline launches use physical GPU1.
- GPU1 launchers must see `CUDA_VISIBLE_DEVICES=1`.
- Do not train directly on the login node.

## Known Remote State To Reconfirm

These are references from existing notes and past log reads, not fresh facts.
Reconfirm them with read-only commands after SSH recovers before using them as
launch dependencies or result claims.

| Item | Current referenced status | Reconfirm before action |
|---|---|---|
| Dense AdaTAD teacher | Was running / had early logs. It is needed by Stage2. | Latest process state, latest checkpoint, config path, `train.out`, final epoch, mAP curve, checkpoint sha256. |
| Stage2 waiter | Was waiting for dense teacher `epoch_59`. | Whether waiter is still alive, whether it launched, whether it wrote a precheck summary. |
| PAction learned fixed_384 | Referenced best Avg mAP `59.10`. | Same commit, same source, train/val/test ledgers, `train.out`, final mAP table, checkpoint and manifest sha256. |
| GAS-VT fixed_384 | Referenced Avg mAP `44.90`. | Same commit status, whether apply-time budget conditioning was pre/post fix, ledgers, validation summaries, mAP table. |

Do not promote any of these numbers to paper evidence until the remote paths,
commit, sha manifest, selected-count histograms, and matched settings are
recorded.

## Priority Queue

### P0: Matched `uniform_384`

- Role: paper-main control.
- Purpose: establish the clean same-commit fixed_384 baseline that every learned
  selector must beat. This is the most important denominator for the next paper
  claim gate.
- Budget: fixed 384 selected positions over dense 768.
- GPU: GPU1 unless the existing uniform launcher explicitly uses a separate
  allocation and is documented.
- Dependencies:
  - THUMOS annotations and class map under `/data/run01/sczc063/yuzibo/thumos14/annotations/`.
  - AdaTAD pretrain `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`.
  - Exact uniform C3 config, expected candidate from older N16R4 path:
    `configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py`.
  - Existing full-train sbatch or launcher should be inspected read-only before
    use; do not invent a new remote launcher during recovery.
- Precheck command draft:

```bash
CUDA_VISIBLE_DEVICES=1 PRECHECK_ONLY=1 \
  CONFIG=configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py \
  bash scripts/run_pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.sbatch
```

- Full run command draft:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
  CONFIG=configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py \
  RUN_TAG=uniform_384_matched_$(date +%Y%m%d_%H%M%S_%z) \
  bash scripts/run_pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.sbatch
```

- Success standard:
  - Same commit and pretrain as learned baselines.
  - Selected count is exactly 384 except valid-ratio-short videos.
  - No learned selector, no GT, no teacher, no prediction cache.
  - mAP table includes tIoU 0.30, 0.40, 0.50, 0.60, 0.70.
  - Evidence includes selection manifest, config sha, and result logs.
- Stop conditions:
  - Precheck cannot prove exact uniform budget protocol.
  - Launcher would run outside Slurm.
  - Config is not same-commit comparable to PAction/Stage2.

### P1: PAction Learned `fixed_384` Recheck

- Role: paper-main Stage1 baseline.
- Purpose: verify the strong `p_action`-supervised selector under the same
  commit and source ledger contract used for `uniform_384` and Stage2.
- Budget: fixed 384 selected positions over dense 768.
- GPU: GPU1 only.
- Dependencies:
  - Source samples for train/val/test.
  - `scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh`.
  - `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train.py`.
  - `configs/adatad/thumos/c3_paction_learned_ledger_adatad_full_train_exec.py`.
  - validators:
    `tools/bata/validate_paction_learned_policy_ledger.py` and
    `tools/bata/validate_c3_paction_learned_adatad_full_train.py`.
- Precheck command draft:

```bash
CUDA_VISIBLE_DEVICES=1 \
PRECHECK_ONLY=1 \
PACTION_ADATAD_VARIANTS="learned_fixed_384" \
RUN_TAG=paction_learned_fixed384_recheck_precheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh
```

- Full run command draft:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
PRECHECK_ONLY=0 \
ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN=1 \
PACTION_ADATAD_VARIANTS="learned_fixed_384" \
RUN_TAG=paction_learned_fixed384_recheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_c3_paction_learned_policy_adatad_full_train_gpu1.sh
```

- Success standard:
  - Reproduces or explains the referenced `59.10` Avg mAP.
  - Validation summaries exist for train, val, and test.
  - Policy checkpoint sha256 is recorded and bound into ledger validation.
  - Selected-count histogram, boundary r1/r2/r4, action coverage, max/p95
    holes, top-k overlap, and uniform similarity are recorded.
  - High-IoU mAP, especially 0.60 and 0.70, does not regress relative to the
    referenced run without an attributable cause.
- Stop conditions:
  - Any ledger uses GT, teacher, oracle boundary, raw prediction cache, or
    val/test-derived signal for deployment selection.
  - `CUDA_VISIBLE_DEVICES` is not `1`.
  - Full run is attempted without `ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN=1`
    and a Slurm allocation.

### P2: Lattice Diagnostic 384

- Role: diagnostic, not paper-main method.
- Purpose: test whether geometry-compatible local replacement explains some of
  the PAction advantage. This must be described as lattice replacement over a
  scaffold, not as pure learned intelligence.
- Budget: fixed 384.
- GPU: GPU1 only.
- Dependencies:
  - PAction source samples and policy training path.
  - `scripts/run_c3_paction_lattice_replacement_adatad_full_train_gpu1.sh`.
  - `tools/bata/run_paction_lattice_replacement_ledger_pipeline.py`.
  - `tools/bata/validate_paction_lattice_replacement_ledger.py`.
- Precheck command draft:

```bash
CUDA_VISIBLE_DEVICES=1 \
PRECHECK_ONLY=1 \
PACTION_LATTICE_FIXED_BUDGET=384 \
PACTION_LATTICE_ADATAD_VARIANTS="paction_lattice_fixed_384" \
RUN_TAG=paction_lattice_fixed384_diag_precheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_c3_paction_lattice_replacement_adatad_full_train_gpu1.sh
```

- Full run command draft:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
PRECHECK_ONLY=0 \
ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN=1 \
PACTION_LATTICE_FIXED_BUDGET=384 \
PACTION_LATTICE_ADATAD_VARIANTS="paction_lattice_fixed_384" \
RUN_TAG=paction_lattice_fixed384_diag_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_c3_paction_lattice_replacement_adatad_full_train_gpu1.sh
```

- Success standard:
  - Ledger identifies lattice/replacement provenance and checkpoint sha.
  - Counts, gap growth, boundary support, uniform similarity, and replacement
    counts are reported.
  - mAP is compared only as diagnostic evidence against matched PAction and
    uniform.
- Stop conditions:
  - Precheck cannot distinguish replacement/scaffold behavior from learned-only
    selection.
  - It requires code changes during the SSH recovery window.
  - It is about to consume the only GPU slot before PAction/uniform are secured.

### P3: Stage2 Proposal-Score Utility `fixed_384`

- Role: diagnostic bridge toward detector utility.
- Purpose: run the currently available Stage2 detector-aware route using the
  existing public utility path, while explicitly labeling it proposal-score
  utility rather than full point responsibility.
- Budget: fixed 384.
- GPU: GPU1 only.
- Dependencies:
  - Dense AdaTAD teacher checkpoint and config, preferably final epoch 59 after
    read-only confirmation.
  - Teacher utility export summary and generator manifest.
  - `scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh`.
  - `scripts/run_c3_detector_aware_selector_adatad_full_train_gpu1.sh`.
  - `configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck.py`.
  - `configs/adatad/thumos/c3_duca_stage2_detector_aware_precheck_exec.py`.
- Precheck command draft:

```bash
CUDA_VISIBLE_DEVICES=1 \
C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH=/path/to/reconfirmed/dense_teacher_epoch_59.pth \
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH=/path/to/reconfirmed/dense_teacher_config.py \
RUN_TAG=stage2_proposal_score_fixed384_precheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh
```

- Full run command draft:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
DUCA_STAGE2_FULL_RUN=1 \
C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH=/path/to/reconfirmed/dense_teacher_epoch_59.pth \
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH=/path/to/reconfirmed/dense_teacher_config.py \
RUN_TAG=stage2_proposal_score_fixed384_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh
```

- Success standard:
  - Precheck summary decision is `DUCA_STAGE2_PRECHECK_PASS`.
  - Teacher evidence is train-only and bound by checkpoint/config sha.
  - Utility manifest states actual signal source; if it is proposal-score from
    `forward_test`, label the run diagnostic.
  - Ledgers are deployable, fixed_384, no uniform fill, no val/test teacher
    signal.
  - Detector mAP beats matched `uniform_384`; it must beat PAction fixed_384 to
    become a main route candidate.
- Stop conditions:
  - Teacher checkpoint/config cannot be confirmed.
  - Utility provenance conflicts with the manifest.
  - Precheck emits anything other than `DUCA_STAGE2_PRECHECK_PASS`.
  - Run would be described as responsibility utility before responsibility code
    exists.

### P4: Stage2 Responsibility Utility `fixed_384`

- Role: intended paper-main Stage2, but blocked.
- Purpose: replace proposal-score-only utility with train-only AdaTAD
  responsibility utility: classification, start/end boundary, regression/loss
  sensitivity, uncertainty/context, false-positive suppression, and ideally
  interval-level counterfactual evidence.
- Status: waiting for code.
- Budget: fixed 384 first; no 768-first claim.
- GPU: GPU1 after precheck.
- Dependencies:
  - Code path that exports responsibility utility from dense teacher training
    artifacts, not val/test inference caches.
  - Generator manifest with `generator_source=dense_detector_forward_train` or
    equivalent.
  - Validator requiring signed utility, train-only split scope, checkpoint sha,
    config sha, output jsonl sha, and no teacher/GT/cache use in deployment
    selection.
- Precheck command draft:

```bash
# Placeholder until responsibility-utility exporter and validator are landed.
CUDA_VISIBLE_DEVICES=1 \
C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH=/path/to/reconfirmed/dense_teacher_epoch_59.pth \
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH=/path/to/reconfirmed/dense_teacher_config.py \
C3_DETECTOR_AWARE_UTILITY_MODE=responsibility \
RUN_TAG=stage2_responsibility_fixed384_precheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh
```

- Full run command draft:

```bash
# Placeholder until precheck supports responsibility utility and passes.
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
DUCA_STAGE2_FULL_RUN=1 \
C3_DETECTOR_AWARE_UTILITY_MODE=responsibility \
C3_DETECTOR_AWARE_TEACHER_CHECKPOINT_PATH=/path/to/reconfirmed/dense_teacher_epoch_59.pth \
C3_DETECTOR_AWARE_TEACHER_CONFIG_PATH=/path/to/reconfirmed/dense_teacher_config.py \
RUN_TAG=stage2_responsibility_fixed384_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage2_detector_aware_precheck_gpu1.sh
```

- Success standard:
  - Responsibility utility evidence passes a fail-closed validator.
  - It reports utility coverage, utility NDCG, boundary responsibility support,
    signed positive/negative utility coverage, and false-positive suppression
    evidence.
  - fixed_384 detector mAP beats `uniform_384` and preferably PAction fixed_384
    under matched settings.
- Stop conditions:
  - Responsibility exporter is absent or only wraps proposal scores.
  - Utility uses val/test GT, dense val/test predictions, cached proposals, or
    teacher data at deploy selection time.
  - Boundary/high-IoU metrics collapse even if Avg mAP improves.

### P5: Stage3 Joint 384

- Role: future paper-main end-to-end evidence, blocked by Stage2 result.
- Purpose: prove selector-detector joint optimization under strict fixed_384,
  with detector loss moving the selector and preserving true-time localization.
- Status: wait for Stage2 result and Stage3 precheck proof.
- Budget: fixed 384 over dense 768.
- GPU: GPU1 only.
- Dependencies:
  - Stage2 fixed_384 ledger and policy evidence.
  - Stage2 result beating `uniform_384`; ideally beating PAction fixed_384.
  - `scripts/run_duca_stage3_truetime_precheck_gpu1.sh`.
  - Stage3 config and exec config:
    `configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck.py` and
    `configs/adatad/thumos/c3_truetime_joint_selector_adatad_precheck_exec.py`.
  - Gradient proof JSON from `tools/bata/run_truetime_joint_selector_precheck.py`.
- Precheck command draft:

```bash
CUDA_VISIBLE_DEVICES=1 \
RUN_TAG=stage3_joint384_precheck_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage3_truetime_precheck_gpu1.sh
```

- Full run command draft:

```bash
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=24:00:00
CUDA_VISIBLE_DEVICES=1 \
DUCA_STAGE3_FULL_RUN=1 \
ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN=1 \
RUN_TAG=stage3_joint384_$(date +%Y%m%d_%H%M%S_%z) \
bash scripts/run_duca_stage3_truetime_precheck_gpu1.sh
```

- Success standard:
  - Precheck summary decision is `DUCA_STAGE3_PRECHECK_PASS`.
  - Selector gradient proof is bound to config/proof sha.
  - Full-run gate records clean git tree, active sha manifest, resolved config
    sha, and proof json sha.
  - Selector parameters or selected-position distribution change due to detector
    loss.
  - Count, duplicate, max-hole, entropy, boundary support, mAP@0.60, and
    mAP@0.70 do not collapse.
- Stop conditions:
  - Stage2 fixed_384 is not credible.
  - Precheck proof is stale or config hash mismatches.
  - Git tracked tree is dirty before formal full train.
  - Any result is about to be called end-to-end without selector gradient proof.

## SSH Recovery Read-Only Checklist

Run only read-only checks first. Do not launch, kill, rsync large outputs, or
edit remote files until this inventory is complete.

```bash
hostname
pwd
date
whoami
git rev-parse HEAD
git status --short --untracked-files=no
ls -lah /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe -maxdepth 3 -type f \( -name 'train.out' -o -name '*.summary.json' -o -name '*manifest*.json' -o -name '*.pth' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
squeue -u "$USER"
nvidia-smi
```

Targeted read-only checks:

```bash
# Dense teacher status and epoch/checkpoint inventory.
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher -maxdepth 5 -type f \( -name 'train.out' -o -name 'epoch_*.pth' -o -name '*.summary.json' -o -name '*manifest*.json' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort

# Stage2 waiter / detector-aware route inventory.
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_stage2_detector_aware -maxdepth 5 -type f \( -name '*.summary.json' -o -name 'train.out' -o -name '*manifest*.json' -o -name '*.jsonl' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort

# Existing PAction and GAS-VT result inventory.
find /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/paction_learned_adatad /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/gas_vt_adatad -maxdepth 5 -type f \( -name 'train.out' -o -name '*.validation.json' -o -name 'pipeline.summary.json' -o -name '*.pth' \) -printf '%TY-%Tm-%Td %TH:%TM %s %p\n' | sort
```

After inventory, compute sha only on specific candidate artifacts, not the whole
tree:

```bash
sha256sum /path/to/candidate/config.py /path/to/candidate/epoch_59.pth /path/to/candidate/manifest.json
```

## Do Not Do After SSH Recovers

- Do not retry remote login in a loop if `Permission loginCluster denied`
  persists.
- Do not train on the login node.
- Do not launch full runs before the matching precheck passes.
- Do not kill the dense teacher, waiter, or existing training jobs until their
  state is read and recorded.
- Do not overwrite existing run directories; always use a new `RUN_TAG`.
- Do not copy historical wiki, logs, checkpoints, generated figures, datasets,
  or compressed artifacts into this repository.
- Do not claim Stage2 responsibility utility from proposal-score-only evidence.
- Do not use 640/768 as the main sparse claim before fixed_384 is settled.
- Do not report PAction `59.10` or GAS-VT `44.90` as current truth until paths,
  commit, and sha evidence are reconfirmed.

## Evidence Checklist

Every queued run should leave a small evidence packet with these items:

- Run identity: route name, `RUN_TAG`, commit hash, git cleanliness, host,
  Slurm job/step id, GPU id, seed, command line, environment overrides.
- Config evidence: config path, exec config path, config sha256, active
  manifest sha256, pretrain path and sha256.
- Input provenance: source jsonl paths and sha256, split identity, train-only
  teacher scope when applicable, no val/test GT or teacher use in deployment
  selection.
- Policy evidence: checkpoint path and sha256, training summary, utility mode,
  loss weights, budget settings.
- Ledger evidence: train/val/test ledger paths and sha256, selected-count
  histogram, short-valid-ratio handling, exact budget exceptions, repair or
  replacement counts.
- Geometry metrics: boundary support r1/r2/r4, start/end boundary distance,
  max/p95/p99 hole, gap CV/CDF if available, uniform similarity, top-k overlap.
- Utility metrics for Stage2: detector utility coverage, utility NDCG, proposal
  score coverage, point responsibility coverage, signed positive/negative
  utility coverage, boundary utility, false-positive suppression utility, and
  generator manifest.
- Detector metrics: Avg mAP and mAP at tIoU 0.30/0.40/0.50/0.60/0.70, eval
  epoch curve, best and final checkpoint distinction, high-IoU regression note.
- Compute profiling: wall time, GPU memory, utilization snapshot, data loading
  bottleneck notes, estimated sparse-vs-dense compute and any profiler output.
- Failure packet if stopped: failing command, exact error, last 100 log lines,
  whether artifacts are reusable, and the next safe retry condition.

## Launch Order Once Read-Only Checks Pass

1. If dense teacher is unfinished but healthy, let it finish. Do not preempt it.
2. Confirm or produce matched `uniform_384`.
3. Recheck PAction learned `fixed_384`.
4. Run lattice diagnostic only if it does not block the paper-main queue.
5. Run Stage2 proposal-score fixed_384 as diagnostic if dense teacher evidence
   is valid.
6. Land and precheck Stage2 responsibility utility before making any main
   detector-utility claim.
7. Run Stage3 joint 384 only after Stage2 fixed_384 produces credible detector
   evidence and the Stage3 gradient proof is fresh.

