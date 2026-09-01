# BUILDER_PJST_D1_CYCLE3_CHECKPOINT_BINDING_CORRECTION-v001

## Identity / binding

- Same implementation Builder, second bounded claim-preserving correction of the clean
  PJST-D1 Cycle-3 package.
- Worktree: `C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle3-builder-20260826`
- Branch: `codex/duca-pjst-cycle3-builder-20260826`
- Clean starting commit (parent): `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`
- Independent review consumed:
  `E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.cvpr-pro-lab/role-returns/CRITIC_PJST_D1_CYCLE3_FOCUSED_REVIEW-v001.md`
- Only the authorized validator / launcher / focused-test correction was made. No model,
  selector, detector, data, loss, optimizer, schedule, evaluator, scientific mechanism, or
  experiment-matrix change. No browser/data/GPU/Slurm/remote access, training, inference,
  evaluation, metric, or claim.

## Parent / head / diff

- parent: `5a4fef786f71f191cd9bb4fe3cb32334a0ba61b5`
- head (new clean commit): `cbefa51563adce5c512403695259f2fcb3da16fa`
- commit message: `Fix PJST-D1 validator/launcher Stage-1 checkpoint binding fail-closed (static, no-data)`
- diff (parent..head), limited to the three authorized files:

```
 scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch   |  9 ++-
 tests/test_duca_pjst_d1_derivative_only.py             | 91 +++++++++++++++++++++-
 tools/bata/validate_duca_pjst_d1_derivative_only.py    | 79 ++++++++++++++-----
 3 files changed, 152 insertions(+), 27 deletions(-)
```

- `git status --porcelain=v1` after commit: empty (clean; the local ARIS session dir
  `.claude/` was removed and is not part of any commit).

## What was changed (exact deterministic defect)

### Validator (`tools/bata/validate_duca_pjst_d1_derivative_only.py`)
- Removed the fabricated fallback (`/nonexistent/duca_stage1_epoch29.ckpt` + all-zero
  SHA-256 via `os.environ.setdefault`).
- `--stage1` and `--sha256` are now `required=True`; no admission-capable PASS can occur
  without both.
- Before any config admission: epoch must be exactly 29; the checkpoint must be a readable
  regular file (`Path.is_file()` + `os.access(..., os.R_OK)`); the digest must match
  `^[0-9a-fA-F]{64}$`.
- The file's actual SHA-256 is computed with a streaming `hashlib` read and compared
  case-insensitively to the supplied digest; a well-formed but wrong digest fails.
- Both resolved configs are now verified to retain the exact supplied checkpoint path,
  digest, and epoch 29 under `workflow.model_initialization` (no coercion/override), in
  addition to the existing `state_dict_ema` / epoch-29 checks.

### Launcher (`scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`)
- `precheck()` now independently fails closed before the validator / PRECHECK / training:
  epoch `== 29`; checkpoint `-f` and `-r`; SHA matches `^[0-9a-fA-F]{64}$` (replacing the
  prior length-only check).
- The launcher still calls the validator with the explicit `--stage1 "$STAGE1_CHECKPOINT"
  --sha256 "$STAGE1_SHA" --epoch 29`; it does not compute or replace the declared digest —
  the validator performs authoritative content equality.

### Focused tests (`tests/test_duca_pjst_d1_derivative_only.py`)
- Extended (existing model-path tests untouched) to distinguish: missing file;
  non-regular path (directory); malformed digest; well-formed wrong digest; correct small
  fixture file/digest; wrong epoch; config retention (via the correct-fixture PASS);
  launcher precheck fail-closed (structural check of the guards preceding the validator
  call + explicit path/digest/epoch pass-through); plus an unreadable-file case
  (POSIX-only, root-aware skip).

## Verification (exact commands / results)

Host: Windows (unknown), Python 3.11.7. All commands run inside the worktree.

1. `python -m py_compile tools/bata/validate_duca_pjst_d1_derivative_only.py tests/test_duca_pjst_d1_derivative_only.py`
   → exit 0 (PASS).
2. Validator negative cases (all expected non-zero):
   - no args → exit 2 (argparse: `--stage1, --sha256` required)
   - missing file → exit 1 (`Stage-1 checkpoint is not a readable regular file`)
   - malformed digest (`not-a-sha`) → exit 1 (`sha256 must match ^[0-9a-fA-F]{64}$`)
   - well-formed wrong digest (`a`*64) → exit 1 (`sha256 mismatch: supplied=… actual=…`)
   - wrong epoch (`--epoch 30`) → exit 1 (`epoch must be exactly 29, got 30`)
3. Validator positive case: temporary regular file with computed SHA-256, `--epoch 29`
   → exit 0, `PASS PJST-D1 matched configs: sole_distinction=[work_dir, pjst_derivative_only]
   seed=3407 epochs=60 updates=6000 checkpoint_interval=5 roots=[…off, …on]
   stage1_epoch29_checkpoint=<exact supplied path>` (confirms retention of exact path/digest/epoch).
4. `python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q`
   → `1 skipped in 1.35s` (honest record: local Windows Torch `c10.dll` load failure —
   the module-level skip; no tests executed locally). Full focused run remains required on
   Linux/N16R4 before PRE_RUN.
5. `bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch` → exit 0 (PASS).
6. `git diff --check` → exit 0 (PASS).
7. `git status --porcelain=v1` → empty (clean worktree).

## Evidence class

`STATIC_NO_DATA_IMPLEMENTATION`

## No-efficacy boundary

- This correction only makes the validator/launcher fail closed on the Stage-1 epoch-29
  checkpoint binding and extends the focused tests accordingly. No model result, metric,
  or claim is produced, touched, or implied. No training/inference/evaluation ran.

## next_owner

`the same independent Critic focused recheck` (consume clean commit
`cbefa51563adce5c512403695259f2fcb3da16fa`; then Evaluator runs the Linux/N16R4
Torch-dependent focused tests and PRE_RUN).
