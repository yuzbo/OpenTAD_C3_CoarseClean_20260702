# DUCA Uniform-Policy Homotopy Official-60

## Status

`gate_failed/stop_and_revise_representation`

## Purpose

The learned protected selector previously starts from a random hard path and
can disrupt the detector before its transition supervision becomes useful.
This successor keeps one end-to-end training run but starts the actual hard
detector input at canonical exact-uniform sampling. It then changes the shared
physical exact-K node potential continuously toward the learned policy.

## Frozen design candidate

- Offline TAD, full-window selection, fixed `K=384`.
- Official ASFormer coarse binary action/transition supervision remains.
- Warmup: first 5% of successful optimizer updates use alpha zero.
- Transition: next 30% use cosine alpha from zero to one.
- Remaining updates and all inference use alpha one.
- Alpha zero contract is `hard_forward_exact_uniform`; no soft-distribution
  equivalence claim is allowed.
- Detector gradient bridge scale is 0.25 and updates the selector policy, not
  the coarse ASFormer trunk/action head.
- The schedule advances only after a successful optimizer step and is stored
  in model/EMA checkpoints; AMP replay restores its pending state.

## Required matched experiment

One exact commit/P0 must run four 60-epoch, seed-3407 arms with 6000 successful
updates each: exact uniform, direct bridge 1.0, bridge 0.25, and homotopy bridge
0.25. Primary evidence is terminal epoch-59 EMA official THUMOS mAP. The GO
criterion remains strictly above matched uniform and strictly above 65 Avg-mAP.

## Exact deployment (2026-07-21)

- GitHub branch: `codex/duca-uniform-homotopy-20260721`.
- Exact commit/tree: `b987c8c6bd2b9f83027354adaaf6f338a205798a` /
  `d33d91941578e16cbf5a8cdc67b8b58471a29411`.
- Clean remote snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_homotopy_b987c8c_20260721`.
- Exact Linux focused and legacy regression suite: `155 passed`.
- Independent final code audit: `PASS`; artifact-chain recomputation,
  homotopy phase boundary and forced full-model AMP replay checks are closed.
- P0 root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_homotopy_b987c8c_gate_20260721_064211`.
- P0 file SHA-256:
  `a246dc8c3fbc6f6e4a65a3a706a1259e54421f93a4707a922c567db1c92f9b99`.
- P0 freezes loader length 100, 60 epochs and 6000 successful updates per
  arm. The homotopy schedule is bound to the same 6000 updates.
- Superseded gate `1177713` ran for 2m33s and failed on a control-plane dtype
  defect: a float64 physical cap was narrowed to AMP float16 before metadata
  verification. It is a failed gate, not model evidence.
- Superseded gate `1177714` also ran for 2m33s and failed in a gate-only
  perturbation helper that called floating-point random noise on uint8 RGB.
  It crossed the prior physical-cap check but remains diagnostic only.
- Formal replacement CUDA/P3 gate Job: `1177715` (`dp_all_b987c8c`), submitted
  on `g0003` at 2026-07-21 06:46 +0800 and failed closed after `00:02:40`.
- Exact failure: the physical/native and selected-axis detector losses were
  not equivalent under endpoint-inclusive exact-uniform selection. Read-only
  diagnostic Job `1177719` measured objective `0.072007999` versus
  `0.094866410`, a 24.10% relative difference. This is a representation
  mismatch, not a tolerance or AMP issue.

The frozen P1 rule therefore stops this chain. CUDA/P3 authorization and the
four-arm official-60 training were not produced; no optimizer update,
checkpoint or mAP exists. The next admissible action is an explicit
representation revision, not loosening the parity threshold or resubmitting
the same method.
