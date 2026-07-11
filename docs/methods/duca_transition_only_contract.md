# DUCA Shared-ASFormer Transition-Only Contract

## Status

This is an offline TAD pre-backbone candidate. It is not Online TAD, not a
streaming method, and not paper-ready until the P0 evidence gates pass.

The existing `a5e1774` direct-boundary implementation remains a frozen matched
baseline. Dynamic MUST and additional spatial backbones are out of scope for
this decision.

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

- Steps 0-660: exact-uniform reference policy, beta 0.
- Steps 660-4620: cosine policy alpha from 0 to 1, beta 0.
- Steps 4620-7920: learned policy, cosine beta from 0 to 0.25.
- Steps 7920-13200: learned policy, beta 0.25.

The 13200 value is an expected horizon from 132 epochs and 100 batches per
epoch. The actual successful optimizer-step counter is authoritative under AMP.

## P0 Matrix

- Exact-uniform fixed-384 in the same structured feasible family.
- Frozen direct-boundary `a5e1774` architecture retrained for the matched horizon.
- Transition-only fixed-384 with beta 0.
- Transition-only fixed-384 with protected beta ramp to 0.25.

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
