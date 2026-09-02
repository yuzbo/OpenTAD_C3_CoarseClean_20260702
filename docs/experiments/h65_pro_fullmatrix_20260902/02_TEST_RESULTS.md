# Test Results

Local structural check on final implementation commit `bd8623754a4375c39eb5c941893c606cffbcd6de`:

- `python tools/bata/validate_h65_pro_fullmatrix.py`: PASS, 28 unique train jobs, configs, factors, and strict60 identities.
- `python -m py_compile tools/train.py tools/test.py tools/bata/train_lowres_action_probe.py tools/bata/duca_selected_axis_training.py tools/bata/generate_h65_pro_fullmatrix.py tools/bata/validate_h65_pro_fullmatrix.py tools/bata/h65_pro_hard_one_swap_diagnostic.py opentad/models/bricks/scale_adaptive_conv1d.py opentad/models/bricks/conv.py opentad/models/dense_heads/anchor_free_head.py opentad/models/duca/acquisition.py opentad/models/selectors/duca_online_frame_selector.py opentad/models/backbones/vit_adapter.py opentad/models/backbones/backbone_wrapper.py opentad/models/detectors/single_stage.py`: PASS.
- `python -m pytest tests/test_h65_pro_fullmatrix.py -q`: PASS, 6 passed and 3 skipped because this Windows host cannot load PyTorch `c10.dll`.
- `python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q`: PASS, 23 passed.
- `bash -n tools/experiments/run_h65_pro_train.sbatch tools/experiments/run_h65_pro_eval.sbatch tools/experiments/submit_h65_pro_fullmatrix.sh`: PASS.
- `git diff --check`: PASS.

Remote N16R4 checks on clean worktree `/data/run01/sczc063/yuzibo/OpenTAD_H65Pro_FullMatrix_20260902_bd862375`:

- `python -m pytest tests/test_h65_pro_fullmatrix.py -q`: PASS, 9 passed.
- `python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q`: PASS, 23 passed.
- `H65_PRO_EXPECTED_COMMIT=bd8623754a4375c39eb5c941893c606cffbcd6de PRECHECK_ONLY=1 bash tools/experiments/submit_h65_pro_fullmatrix.sh`: PASS, full 28-row train/eval precheck completed.
- Formal Slurm submission is blocked by current account/QOS state: `sbatch` returned `AssocMaxSubmitJobLimit`; final-commit registry is not present because no complete 28-row DAG was accepted.
