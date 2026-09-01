# EVALUATOR_PJST_D1_CYCLE4_RAW_FILE_EXECUTION-v004

The prior syntax report is invalidated by direct remote byte inspection: validator line 100 contains bytes `66 22` (`f"` in JSON display means Python `f"..."` without a literal backslash), and `tools/train.py` is likewise normal. Local exact-commit `py_compile` passes. Do not patch, copy, render, serialize, or reconstruct source text.

- source checkout: `/data/run01/sczc063/yuzibo/projects/duca_pjst_d1_cycle4_c195b97c_recovery_20260826`
- exact revision: `c195b97c46acae166e0721fcb412b70221ae7d49`
- execute commands directly against files in that checkout only.

After canonical environment activation, run the raw commands from the checkout:

```bash
python -m py_compile tools/train.py tools/bata/validate_duca_pjst_d1_derivative_only.py
python -m pytest tests/test_duca_pjst_d1_derivative_only.py -q -rs
python tools/bata/validate_duca_pjst_d1_derivative_only.py --stage1 /data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth --sha256 bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3 --epoch 29
bash -n scripts/run_duca_pjst_d1_matched_cycle3_n16r4.sbatch
```

No inline Python, heredoc, temporary source, JSON-decoded source, or copied file is allowed. If these exact commands pass with zero focused-test skips/failures and all v001 resource/duplicate checks pass, immediately submit the matched OFF/ON formal jobs exactly as v001 specifies. Otherwise return the raw command stderr and stop.

