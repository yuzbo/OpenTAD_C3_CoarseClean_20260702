# Builder PJST-D1 implementation receipt v001

- Commit: `877d893f` (`Implement frozen PJST-D1 derivative transport`)
- Changed files: `opentad/models/utils/temporal_grid.py`, `opentad/models/backbones/backbone_wrapper.py`, `opentad/models/backbones/vit_adapter.py`, `opentad/models/detectors/single_stage.py`, `configs/adatad/thumos/duca_pjst_d1_stage2.py`, `tests/test_pjst_d1_contract.py`.
- Implemented: global int64 pair validation and detached scale/audit metadata; pre-PatchEmbed mean/odd transport with mixed-row uniform bypass; metadata passthrough; pre-filter exactly-once proposal remap ordering; frozen-selector config and focused algebra/padding tests.
- Syntax evidence: `python -m py_compile opentad/models/utils/temporal_grid.py opentad/models/backbones/backbone_wrapper.py opentad/models/backbones/vit_adapter.py opentad/models/detectors/single_stage.py` passed.
- Focused test evidence: `python -m pytest tests/test_pjst_d1_contract.py -q` could not collect because this Windows environment aborts loading `torch` (`c10.dll`, WinError 1114). No training, PRE_RUN, evaluator, or metrics were run.
- Evidence boundary: no claim of identity, finite-gradient, production CUDA trace, or empirical effectiveness; those require the Linux/OpenTAD environment and remain unverified.
