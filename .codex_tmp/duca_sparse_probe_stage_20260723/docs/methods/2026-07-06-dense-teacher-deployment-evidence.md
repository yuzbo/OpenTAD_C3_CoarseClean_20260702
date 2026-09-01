# Dense AdaTAD Teacher Deployment Evidence

Date: 2026-07-06
Branch: `codex/gas-vt-stage23-detector-aware-20260706`
Commit: `366b9951ef39`

## Purpose

This route trains a selector-free dense AdaTAD teacher under the official dense 768-frame AdaTAD setting. It is not a sparse-acquisition result and does not unlock any detector mAP improvement claim by itself. Its role is to provide a clean dense baseline checkpoint and, later, train-only detector utility for Stage2 detector-aware selector training.

## Local Verification

- `python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py tools/bata/detector_teacher_utility.py tools/bata/train_detector_aware_acquisition_policy.py`
- `python -m pytest tests/test_c3_dense_adatad_teacher_full_train.py -q`
- `bash -n scripts/run_c3_dense_adatad_teacher_full_train_gpu.sh`

Focused dense-teacher pytest passed with `2 passed, 1 warning` before remote deployment.

## Remote Snapshot

- Snapshot: `/data/run01/sczc063/yuzibo/projects/opentad_dense_teacher_366b9951ef39_20260706_233128`
- Source commit marker: `.source_commit = 366b9951ef39`
- Script: `scripts/run_c3_dense_adatad_teacher_full_train_gpu.sh`
- Config: `configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py`

The remote GitHub clone path failed due a TLS transport problem, so the snapshot was synced by local `git archive` + `scp`, without touching active experiment snapshots.

## Precheck Evidence

Remote command used the independent snapshot and `PRECHECK_ONLY=1`.

Precheck output:

```json
{
  "checkpoint_epochs_zero_based": [9, 19, 29, 39, 49, 59],
  "config": "configs/adatad/thumos/c3_dense_adatad_teacher_full_train.py",
  "decision": "C3_DENSE_ADATAD_TEACHER_PRECHECK_PASS",
  "dense_teacher_axis": "dense_768_frame_axis",
  "eval_epochs_zero_based": [9, 19, 29, 39, 49, 59],
  "full_train_requires_slurm": true,
  "pretrain": "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
  "selector_free_dense_teacher": true
}
```

Precheck file:

`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_precheck_20260706_2335/dense_teacher_precheck.json`

## Pretrain Resolution

Primary requested location was not present:

`/data/run01/sczc063/yuzibo/retrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`

Fallback exists and was used:

`/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`

The launcher now prefers `retrained/` but falls back to `pretrained/` and exports the resolved `C3_DENSE_TEACHER_ADATAD_PRETRAIN_PATH`.

## Full Run Queue

Full training was not launched immediately because both current GPUs were occupied by active experiments. A wait-only launcher was deployed to avoid GPU contention.

- Dense teacher run tag: `c3_dense_adatad_teacher_full_queued_20260706_2338_+0800`
- Waiter PID: `322183`
- Waiter script: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_queued_20260706_2338_+0800/launch_wait.sh`
- Waiter log: `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/dense_adatad_teacher/c3_dense_adatad_teacher_full_queued_20260706_2338_+0800/driver.log`

The waiter checks:

- GPU0 current PID: `1469296` (`GAS-VT Stage0/1`)
- GPU1 current PID: `3694332` (`PAction learned strict ledger`)

It sleeps while both are alive. When either exits, it starts the dense teacher inside Slurm allocation `1118197` with the freed `CUDA_VISIBLE_DEVICES` and `ALLOW_C3_DENSE_TEACHER_FULLTRAIN=1`.

## Claim Boundary

This evidence proves only:

- Dense teacher route has a selector-free official AdaTAD 768-frame config.
- Local and remote precheck gates pass.
- Full training is queued in a non-contentious way inside the existing allocation.

It does not prove:

- Dense teacher detector mAP.
- Detector-aware selector quality.
- Sparse ledger improvement.
- End-to-end selector + AdaTAD training.
