# DUCA semantic dynamic cycle2 arm/split focused recheck — 2026-08-18

Frozen target: `cf7207301d0f29204fd64d704f90a5cac6f305c3`.

## Verdict

`STATIC_PASS / RUNTIME_BLOCKED`. This closes only the focused constructor and
split-contract repair. It is not PRE_RUN admission and it does not supply a
training or fairness result.

## Evidence

- The six declared arms are in
  `configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py:6-12`.
  `tools/train.py:63-80` accepts `--duca-arm` and `--dry-run`, validates
  manifests, and constructs each non-dense arm through `build_arm`.
- Uniform, actionness-only, actionness-plus-boundary, dynamic, and direct
  controls have distinct policies in the config (`:8-12`) and selector
  (`opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py:1566-1600`).
- `tools/bata/duca_semantic_cycle2_contract.py:14-27,41-46` implements
  pairwise FIT/CAL/HOLD ID disjointness and rejects deploy-time GT, teacher,
  and raw-cache fields. The validator applies these checks per arm
  (`tools/bata/validate_duca_semantic_indirect_n16r4.py:14-25`).
- The launcher PRECHECK argv is parser-supported, while future full/pilot
  argv is deliberately non-executing (`scripts/run_duca_semantic_indirect_n16r4.sbatch:3-12`).

## Remaining deterministic execution defects

1. The shared detector/loss/NMS/evaluator/update/seed contract is currently
   metadata in the config (`...six_arm...py:3`); `build_arm`
   (`tools/bata/duca_semantic_cycle2_contract.py:29-39`) does not bind it to
   the actual training runtime.
2. Checkpoint/resume still lacks the required scaler, RNG, DataLoader state,
   five-epoch interval, latest-three retention, milestone/final/final-EMA
   artifacts, and executable resume verification.

The local `PRECHECK_ONLY` and focused tests could not import PyTorch because
of `torch/lib/c10.dll` `WinError 1114`; no runtime construction is claimed.
The next minimal repair may address these training-runtime admission defects
without changing DUCA's mechanism, data split, metric, or scientific claim.

