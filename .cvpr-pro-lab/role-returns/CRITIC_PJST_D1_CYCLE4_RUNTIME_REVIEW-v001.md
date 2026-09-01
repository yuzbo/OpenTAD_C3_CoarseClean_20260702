# CRITIC_PJST_D1_CYCLE4_RUNTIME_REVIEW-v001

- verdict: `PJST_D1_CYCLE4_RUNTIME_STATIC_PASS`
- frozen_candidate: `c195b97c46acae166e0721fcb412b70221ae7d49`
- parent: `a16a67c4f74ce19de640704c357850c0e7b85ba3`
- evidence_class: `READ_ONLY_STATIC_NO_DATA_REVIEW`

The frozen diff changes only `tests/test_duca_pjst_d1_derivative_only.py`; no production, model, config, launcher, schedule, split, metric, or science surface changed.

Findings:

- `test_constant_pair_invariance` now constrains frames 0/1 to one constant pair, checks that pair only for exact invariance, and separately checks that the remaining nonconstant pairs are not vacuous.
- `_RecordingBackbone.forward(self, x, masks=None, metas=None, **kwargs)` matches the supported detector call shape and records both `masks` and `metas`; inference and training tests assert metadata propagation.
- `SingleStageDetector._call_backbone` remains unchanged and retains signature-aware dispatch.
- Identity, shape, gradient, physical-coordinate, exactly-once pre-filter/NMS remap, and metadata-forwarding coverage were not weakened.

Checks:

- `python -m py_compile tests/test_duca_pjst_d1_derivative_only.py opentad/models/detectors/single_stage.py opentad/models/backbones/backbone_wrapper.py`: PASS.
- Local Windows pytest reaches the known Torch `c10.dll` module-level skip; this is not code evidence and must be replaced by the Linux/N16R4 focused run.

Classification:

- `IMPLEMENTATION_CORRECTION`: none.
- `SCIENTIFIC_AMBIGUITY`: none.
- fairness/leakage: PASS; no data, GT, teacher, prediction, checkpoint-content, or evaluator access occurred.

- next_owner: `Evaluator`
- next_action: run the exact Linux/N16R4 focused suite, then the frozen PRE_RUN; on `PRE_RUN_READY`, submit the matched H65 OFF/PJST-D1 ON formal jobs immediately.
- dependency: clean exact candidate `c195b97c46acae166e0721fcb412b70221ae7d49`.

