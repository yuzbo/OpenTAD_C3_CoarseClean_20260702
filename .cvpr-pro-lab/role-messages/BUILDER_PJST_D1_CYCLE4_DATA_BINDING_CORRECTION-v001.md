# Builder focused correction — PJST-D1 Cycle 4 data binding

You are the sole Builder for one claim-preserving runtime correction. Work only in:

`C:/Users/skywalker/.codex/worktrees/duca-pjst-cycle4-builder-20260826`

Frozen parent: `c195b97c46acae166e0721fcb412b70221ae7d49`, clean branch `codex/duca-pjst-cycle4-builder-20260826`.

## Observed failure

Formal jobs 1256281/1256282 both passed the focused precheck, then failed before the first optimizer update while building `cfg.dataset.train`:

`FileNotFoundError: /data/run01/sczc063/yuzibo/tmp/home/run/yuzibo/thumos14/annotations/thumos_14_anno.json`

The launcher defines canonical read-only paths but does not export the environment variables explicitly consumed by `configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py`. Its `data.*` cfg-options create an unused tree; `tools/train.py` reads `cfg.dataset.*`.

## Exact scope

Make the smallest runtime-only correction:

1. In `scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch`, bind the already-verified canonical paths through the config's real supported interface before config loading:
   - `THUMOS14_ANNOTATION_PATH=$ANNOTATION_PATH`
   - `THUMOS14_CLASS_MAP=$CATEGORY_PATH`
   - `THUMOS14_TRAIN_DATA_PATH=$VIDEO_ROOT`
   - `THUMOS14_TEST_DATA_PATH=$VIDEO_ROOT`
2. Remove or replace the stale `data.*` options so the final training command cannot silently leave `cfg.dataset.*` unchanged. Prefer the environment interface above; do not add duplicate override layers without necessity.
3. Keep `model.backbone.custom.pretrain=$ADATAD_PRETRAIN`: the actual base config defines the pretrain field at `model.backbone.custom.pretrain`; do not move it to `data_preprocessor`.
4. Add only the smallest focused regression in `tests/test_duca_pjst_d1_derivative_only.py` proving the launcher exports the four exact dataset variables before `tools/train.py`, contains no `data.train/data.val/data.test` overrides, and retains the actual custom pretrain binding.
5. Do not modify production model code, configs, selector, PJST math, optimizer, schedule, seed, split, evaluator, loss, NMS, Stage-1 binding, or scientific documents.

Run the feasible focused checks (`bash -n`, `py_compile`; pytest if the local environment permits). Commit the change cleanly on the existing branch, push that branch, and report commit, changed files, exact checks, and next_owner=Critic. Do not launch remote jobs or experiments.
