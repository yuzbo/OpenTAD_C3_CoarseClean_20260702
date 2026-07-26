# DUCA Online Plugin Contract

DUCA is an online temporal acquisition plugin/adapter for Temporal Action Detection. It is not a new detector. The detector remains AdaTAD, ActionFormer, or another OpenTAD detector; DUCA decides which original-time observations are consumed before the detector sees the temporal sequence.

## Method Boundary

- The plugin consumes deploy-visible dense observations and low-cost actionness/browser signals.
- The plugin emits sparse observations plus `selected_positions`.
- The detector trains and tests on the sparse detector input produced by the plugin.
- Validation, test, and inference decisions must not read GT, dense teacher utility, cached detector predictions, or any offline selection ledger.

Train-only dense teacher utility may supervise/calibrate the acquisition model during training. That payload must not be present in val/test/inference metadata, exported decision payloads, or online selector inputs.

## Position And Budget Contract

`selected_positions` are original-time detector-consumed positions. They are not selected-axis coordinates, not proposal coordinates, and not row ids from a ledger.

Required invariants:

- `selected_positions` are sorted, unique, original dense-time indices for each sample.
- `selected_count <= budget`.
- Paper-main fixed-budget runs use `budget <= 384`.
- The detector input length equals the selected count for the consumed sparse sequence.
- Any post-processing that predicts on a selected axis must explicitly inverse-map predictions back through `selected_positions`.

The ledger is an audit artifact only. It may record what an online selector did after the fact, but it must not be the decision source for the final method.

## OpenTAD Integration Points

AdaTAD integration can happen in either of two equivalent places:

- Detector wrapper: call DUCA before the detector backbone/projection path, then forward sparse observations and metadata to the existing detector.
- Dataset transform path: call DUCA immediately before the gather/subsample transform that creates detector input, then pass sparse observations and `selected_positions` downstream.

ActionFormer uses the same contract. The adapter must run before ActionFormer feature/projection/RPN consumption, and the head or post-processing path must know whether predictions are in selected-axis coordinates and need inverse mapping.

## What The Baselines Mean

C3, GAS-VT, value-transport ledgers, and lattice replacement are baselines, ablations, diagnostics, or failure-analysis tools. They can explain why simple actionness, gap priors, or ledger-based replacement help or fail, but they are not the final paper-main decision mechanism.

The paper-main claim should be phrased as online detector-utility-calibrated temporal acquisition under a strict budget, with train-only teacher utility and no val/test/inference leakage.
