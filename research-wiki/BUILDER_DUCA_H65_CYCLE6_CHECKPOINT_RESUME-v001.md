# DUCA H65 Cycle-6 Builder Receipt

## MINIMAL_CHANGE_PLAN

1. Inspect the existing training/checkpoint path and preserve its current DUCA metadata contract.
2. Add only the missing checkpoint/resume bookkeeping needed for successful optimizer updates and deterministic RNG round-trip; do not invent sampler/selector state.
3. Add focused CPU-only tests for checkpoint state and random-stream/update-count restoration.
4. Run focused static/tests, record boundaries, and hand off to Critic.

Status: implemented.

## Changed files and symbols

- `opentad/utils/checkpoint.py`: `save_checkpoint` now serializes optional top-level `successful_optimizer_updates`.
- `tools/train.py`: checkpoint call passes the real cumulative `update_audit["successful_optimizer_updates"]`.
- `tests/test_duca_checkpoint_resume_roundtrip.py`: CPU round-trip coverage for model/optimizer/scheduler, update count, and Python/NumPy/Torch RNG streams.

No sampler or selector state was added: the current formal loader uses epoch-boundary saves, calls dataset `set_epoch`, and the selector schedule is a model buffer already covered by `state_dict` and the existing DUCA training metadata contract.

## Verification

- `python -m py_compile tools/train.py opentad/utils/checkpoint.py tools/bata/duca_p0_training.py tests/test_duca_checkpoint_resume_roundtrip.py` — passed.
- `python -m pytest tests/test_duca_checkpoint_resume_roundtrip.py -q` — not executed due to local Windows PyTorch DLL initialization failure (`WinError 1114`, `c10.dll`); no data/GPU/remote training used.

## Unexecuted boundaries

Stage-1 one-batch checkpoint audit and dummy epoch29 same-start OFF/ON smoke remain for Evaluator. No Slurm, data, GPU, or training run was performed.

next_owner=Critic
