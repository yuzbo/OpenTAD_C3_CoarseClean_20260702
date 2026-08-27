# BUILDER PJST-D1 canonical pretrain binding v003

status: MATERIAL_READY
start_revision: 4204937a933c7a48854b623efefc7fd662e98805
end_revision: HEAD (the clean commit containing this receipt; its exact immutable SHA is reported in the Builder return because a commit cannot contain its own SHA)
scientific_training_revision: c73e8418de31cdcb2a445ff58a1e33ab9ab6a508
next_owner: independent Critic
next_action: one static PASS/BLOCKED review
dependency: clean commit
expected_return_at: 2026-08-27T22:00:00+08:00
single_recovery: none

## Exact diff

- `scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch`: define `ADATAD_PRETRAIN` with environment override and canonical default `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`; include it in the existing readable-file gate; bind `model.backbone.custom.pretrain="$ADATAD_PRETRAIN"` in the real `tools/test.py --cfg-options` invocation.
- `tests/test_duca_pjst_d1_terminal_finalizer.py`: add three minimal static assertions covering the canonical default, readable-file gate, and real cfg-option binding.
- This receipt is the only other added file.

## Verification

- `bash -n scripts/run_duca_pjst_d1_terminal_eval_n16r4.sbatch`: PASS.
- `python -m pytest tests/test_duca_pjst_d1_terminal_finalizer.py -q`: PASS, 4 passed.
- `python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py`: PASS.
- `python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q`: PASS, 23 passed with one non-failing `lib2to3` deprecation warning.
- `git diff --check`: PASS before receipt creation.

## Scientific state unchanged

PJST-D1 epoch-59 EMA point estimates remain OFF 65.063 and ON 64.591. This change does not alter the model, selector, training, checkpoint, data, NMS, evaluator, bootstrap, seed, or result interpretation. No retraining, data/GPU access, Slurm operation, remote submission, or remote write occurred.
