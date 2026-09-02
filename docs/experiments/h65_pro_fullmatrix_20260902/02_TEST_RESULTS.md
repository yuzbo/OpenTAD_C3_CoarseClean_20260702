# Test Results

Post-review local structural check:

- `python tools/bata/validate_h65_pro_fullmatrix.py`: PASS, 28 unique train jobs, configs, factors, and strict60 identities.
- `python -m py_compile tools/train.py tools/test.py tools/bata/duca_p0_training.py tools/bata/duca_selected_axis_training.py tools/bata/generate_h65_pro_fullmatrix.py tools/bata/validate_h65_pro_fullmatrix.py opentad/models/detectors/actionformer.py opentad/models/dense_heads/anchor_free_head.py opentad/models/duca/acquisition.py`: PASS.
- `python -m pytest tests/test_h65_pro_fullmatrix.py -q`: PASS, 8 passed and 7 skipped because this Windows host cannot load PyTorch `c10.dll`.
- `python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q`: PASS, 23 passed.
- `bash -n tools/experiments/run_h65_pro_train.sbatch tools/experiments/run_h65_pro_eval.sbatch tools/experiments/submit_h65_pro_fullmatrix.sh`: PASS.
- `git diff --check`: PASS.

Remote Torch regression checks, full `PRECHECK_ONLY=1`, and Slurm submission remain pending until the clean pushed fix commit is available on the N16R4/Slurm host.
