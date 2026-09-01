# DUCA Shared-ASFormer Transition-Only Contract

## Status

This is an offline TAD pre-backbone candidate. It is not Online TAD, not a
streaming method, and not paper-ready until the P0 evidence gates pass.

The existing `a5e1774` direct-boundary architecture remains a frozen-definition
matched baseline: its architecture/hidden semantics are preserved, but its
weights are retrained under the matched 132-epoch protocol. Dynamic MUST and
additional spatial backbones are out of scope for this decision.

## Forward Path

1. A low-resolution spatial stem and the official ASFormer implementation
   predict binary action/background logits over the full 768-observation window.
2. The wrapper captures the encoder hidden state from the same official forward
   call that produced the logits. It does not replay the encoder.
3. The selector receives only temporal changes of detached binary logits,
   entropy, and encoder hidden state. Absolute hidden state and raw RGB means are
   not ranking inputs.
4. One shared transition-utility scorer produces equal-valued auxiliary and
   policy scores with different gradient ownership.
5. The existing structured dynamic program chooses exactly 384 positions while
   enforcing a maximum unselected hole of 15. Hard inference and soft training
   use the same feasible family.
6. Selected raw frames are consumed by the unchanged official-config
   ActionFormerHead path. Selected-axis geometry is remapped to true time by the
   existing wrapper contract.

## Gradient Ownership

- Balanced binary actionness loss updates the spatial stem, ASFormer trunk, and
  binary output layers.
- Transition distribution loss updates the spatial stem, ASFormer encoder, and
  shared transition scorer. Binary output logits are detached on this route.
- Soft boundary coverage and detector gradients update the shared scorer through
  detached transition descriptors. They must not update the coarse probe.
- The detector always receives its full classification and regression losses.
  Beta scales only the detector-to-selector bridge.

Audited learning rates are `2.5e-5` for the spatial/ASFormer trunk, `5e-5` for
binary output layers, `1e-4` for the transition scorer and detector modules, and
`2e-4` for existing VideoMAE adapters.

## Continuous Schedule

All modules exist in one model and one optimizer from step zero. There is no
checkpoint handoff and no modulo-step policy switching.

- Steps 0-660: exact-uniform reference policy; direct detector bridge is 0.
- Steps 660-4620: cosine policy alpha from 0 to 1; direct detector bridge remains 0.
- The counterfactual arm uses detached hard one-swap utility distillation with
  `lambda_cf=0.25`; this is not a detector-gradient beta.
- Steps 4620-13200: learned policy remains active with the same detached
  counterfactual distillation weight.

The 13200 value is an expected horizon from 132 epochs and 100 batches per
epoch. The actual successful optimizer-step counter is authoritative under AMP.
Formal mAP validation begins at epoch 47, after policy alpha has reached one;
final inference also uses alpha one. Early mixed-policy mAP is not reported as
the learned selector's performance.

## Supervision Provenance

The task-adapted coarse probe is trained with THUMOS14 action/background labels
and GT segment-derived transition targets. This training history is recorded as
`trained_with_thumos_labels=true` and `trained_with_gt_segments=true`. At
validation/test inference it consumes no labels, GT segments, teacher outputs,
prediction cache, or ledger state; those inference-use fields are explicitly
false. The method must not be described as train-free or THUMOS-free.

## P0 Matrix

- Exact-uniform fixed-384 in the same structured feasible family.
- Definition-frozen direct-boundary `a5e1774` architecture retrained for the
  matched horizon, validation timing, and component learning rates.
- Transition-only fixed-384 without counterfactual distillation.
- Transition-only fixed-384 with detached counterfactual distillation
  (`lambda_cf=0.25`, direct detector-gradient bridge 0).

All four use the same data, detector head, physical detector input length,
optimizer horizon, and evaluation protocol.

## Required Gates

- Coarse AUROC/AUPRC/ECE and transition recall pass their health thresholds.
- Exact-K and max-hole violations are zero.
- Three-seed fixed-384 improves over matched exact-uniform, especially at high
  tIoU, or establishes a supported non-inferiority/efficiency claim.
- One-swap counterfactual utility agrees with the detector ST direction before
  detector-guided selection is claimed.
- Trained selector-only and full-stack latency, memory, and FLOPs are reported
  on the same hardware.
- A second detector backend is attempted only after fixed-384 clears the primary
  gate.
- Before any P0 full run, a clean-commit CUDA gate must build the unmodified
  VideoMAE Adapter/AdaTAD model, execute the real 768-to-384 selector path at
  160x160, backpropagate only ActionFormerHead classification plus regression
  losses, and verify non-zero transition-scorer/backbone-adapter/head gradients
  with zero coarse-probe leakage. The gate JSON is bound to commit, config,
  official ASFormer source, and checkpoint hashes.
