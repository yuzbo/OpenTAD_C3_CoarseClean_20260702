# DUCA semantic dynamic cycle2 final critic review — 2026-08-18

Frozen snapshot: `d80022e963a8ad21d390c785cbd8a4c23f41484a`.

## Verdict

`BLOCKED_PRE_RUN`. Three permitted claim-preserving Builder corrections are
complete; no further Builder patch is admissible in this implementation cycle.
This is an implementation/evidence blocker, not an efficacy result or a
scientific refutation of indirect semantic acquisition with dynamic K.

## Evidence

- `py_compile` passed for the changed training, checkpoint, selector,
  ActionFormer, and configuration surfaces. Runtime pytest cannot collect on
  this Windows host because `torch/lib/c10.dll` raises `WinError 1114`.
- The configuration still calls the dense arm `dense_placeholder`
  (`configs/adatad/thumos/duca_semantic_indirect_six_arm_n16r4.py:6-12`), and
  `build_arm` returns a placeholder rather than a dense runtime builder
  (`tools/bata/duca_semantic_cycle2_contract.py:63-66`).
- Shared detector/loss/NMS/evaluator/update/seed fields are strings in config
  (`...six_arm...py:3-5`); the contract never constructs the actual
  ActionFormer/VideoMAE/optimizer/scheduler/data runtime (`...contract.py:56-66`).
- The checkpoint state now contains more recovery state, but artifact lifecycle
  still only writes `epoch_<epoch>.pth` (`opentad/utils/checkpoint.py:90`) and
  does not produce the required `latest3`, milestone, final, and final-EMA
  artifacts (`tools/train.py:305-340`; `checkpoint.py:140-153`).
- Dynamic K bucket dispatch and deploy firewall have static code evidence, but
  the unavailable Torch runtime prevents promotion to a PRE_RUN-ready execution
  contract. Physical-time ordering also has no executable verification here.

No data, GPU, Slurm, remote run, evaluator metric, or performance conclusion
was produced. Next owner is the Coordinator for terminal reporting; a fresh
implementation cycle would require new user authorization. `single_recovery`:
none.

