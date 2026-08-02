---
type: experiment
node_id: exp:scnr-dynamic-stage1-executor-v1
title: "SCNR-TAD dynamic exact-budget ragged executor v1"
stage: tested
status: d4_m2_protocol_implemented_local_tested_remote_precheck_pending
outcome: pass_no_performance
added: 2026-08-02
updated: 2026-08-02
---

# SCNR-TAD dynamic exact-budget ragged executor v1

## Purpose

Build the first executable Stage-1 form of the approved ROI + TokenSelect
Hybrid.  The experiment is an implementation and no-performance admission
gate; it cannot produce mAP, efficiency, floor-optimality, or paper evidence.

## Frozen model contract

- The decision unit is one native two-frame VideoMAE tubelet patch.  A window
  has `T=384` tubelets and the production source lattice has `N=11*20=220`
  physical candidates per tubelet.
- One global constrained projection selects exactly the configured window
  budget `B` unique physical `(t,n)` candidates.  There is no independent
  count head, fixed context quota, per-tubelet quota, padding repair, or dummy
  heavy token.
- The policy has one shared base utility and two modifiers:
  `u_hard=q_base+max(0,delta_roi,delta_res)`.  The same argmax assigns context,
  ROI, or residual role; role IDs do not alter heavy execution or pooling.
- The backward relaxation is
  `u_soft=q_base+tau*logsumexp((0,delta_roi,delta_res)/tau)`.  A global sigmoid
  threshold projection over valid physical candidates produces strict soft
  probabilities with `sum p=B`.  Hard forward membership remains exact top-B.
- The main route permits `K_t=0`.  Its carrier is an exactly zero heavy feature
  plus an explicit boolean heavy-valid mask.  Empty tubelets and empty clips
  execute no patch embedding, attention, MLP, or Adapter token.
- The ROI floor is runtime `(1/W_grid,1/H_grid)` with no full-frame, area,
  coverage, smoothness, expected-cost, fixed-context, or fixed-`K_t` loss.

## Ragged execution contract

Selected physical indices are sorted only after hard top-B.  Each selected
token carries `(batch,tubelet,clip,local_tubelet,spatial_index)` provenance.
Patch embedding runs once on the flat selected-token tensor.  Within every
VideoMAE block, non-empty clips are grouped only with clips having the same
true token count; every bucket executes attention and MLP without padding.
The coordinate-lineage Adapter operates on flat selected tokens and looks up
only identical spatial indices in adjacent global tubelets.  Missing neighbors
contribute zero.

The executor records per-window `K_t`, per-clip `b_c`, `sum_c b_c^2`, the
number and sizes of real ragged buckets, executed patch tokens, attention
pairs, MLP tokens, Adapter tokens, and the requested/unique/padded/executed
counts separately.  `padded=0` is a hard invariant, not a reporting choice.

## Compatibility and failure behavior

The new route is opt-in under a new dynamic mode.  Legacy dense, fixed-K,
structured `8/28/28`, packed Adapter, P0, checkpoint, and audit schemas retain
their existing code paths.  Dynamic execution fails closed on an invalid
budget, duplicate or unsorted physical index, out-of-grid provenance, a soft
budget residual above tolerance, a nonzero padded count, an unmasked empty
tubelet, or any mismatch between selected and executed tokens.

## Implementation milestones

| Milestone | Scope | Required evidence | State |
| --- | --- | --- | --- |
| D0 | Pure global allocator and differentiable exact-sum soft projection | known answers, uniqueness, dynamic roles, finite dense gradients | tested |
| D1 | Native ragged VideoMAE + coordinate-lineage Adapter | zero-padding, empty-clip, full-token parity, exact ledger KATs | tested |
| D2 | Wrapper integration, masked-zero aggregation, scout stop-gradient and proxy schedule | one real detector backward, successful-step schedule, no-leak audit | tested |
| D3 | Clean N16R4 Linux/CUDA no-performance P0 | exact source, clean tree, Slurm GPU, zero metric/checkpoint | passed, Job `1215358` |
| D4 | Matched development G1/G2 floor arms | D3 PASS, sample-level dynamic telemetry, frozen full-stack cost scope | protocol implemented at `ec8de9f5`; local `29 passed`; remote precheck and performance not started |

## Executed evidence

- The opt-in dynamic allocator, native no-padding ragged VideoMAE/Adapter path,
  masked-zero carrier, support-only wrapper, successful-step proxy schedule and
  fail-closed ledgers landed in commits `536e05b7` through `dfcbe692`.  Exact
  source `0c29a5e5` passed the relevant clean N16R4 Linux/Torch regression set:
  `92 passed` with one dependency warning.
- The first CUDA P0, Job `1215355` at source `2829c59e`, exposed a semantic bug
  rather than usable route evidence.  Decoded `w,h` are full rectangle extents,
  but the signed ellipse divided offsets by those values as if they were
  semi-axes.  Its role attribution collapsed to context/ROI/residual
  `0/24573/3`.  The report is preserved as a non-promoting diagnostic.  Commit
  `dfcbe692` changed the ellipse semi-axes to `w/2,h/2` and receipts that
  interpretation explicitly.
- Corrected G1 CUDA P0 Job `1215358` completed `0:0` from exact clean source
  `dfcbe692678c0bb53ec9a23d18623464c8e378c0`.  It sealed
  `PASS_NO_PERFORMANCE_P0`, exact selected/executed `B=24576`, zero padding,
  `K_t=48..83`, and context/ROI/residual counts `4713/14292/5571`.  Report file
  SHA-256 is
  `285116b1ae02826f060d700b435253043a945e49aaecd5903aa0499cfb4abdb6`;
  internal report hash is
  `d2adbded39668f9945422e8e06dc5515d3d5ff20595b1e7bf234905e5bd0048d`.
- The real-data Fit-only policy-health gate then passed as Job `1215363` from
  exact clean source `7cf589f0ff583160c8e45b103e8ea4c316c10339`.  It made 64
  successful updates in 66 attempts, replayed two AMP skips, ended at scale
  `16384`, and observed nonzero gradients for every one of the nine required
  components on `64/64` updates.  Aggregate context/ROI/residual counts were
  `297230/984020/291614`; observed `K_t` ranged `17..218`; maximum soft-budget
  residual was `0.005859375`; peak allocated CUDA memory was `7705451008`
  bytes.  The artifact audit found no checkpoint, prediction, evaluator,
  metric, test, teacher, oracle or route-GT surface.  Report file/internal
  SHA-256 are
  `09e6b03fa747865cce0e1ed0ee54702f89723ae1b64c6b66ee5fdba7f8c3f3d8`
  and
  `fc457ba928743df12d68dcf3713128577d6b8cc175fe1196e0c2b730dfe5ac94`.
- The matched G2 two-cell configuration was added at exact source
  `8aa8e2a3c6649eca94d3ab714d0b122e4f7a5f97`.  CUDA P0 Job `1215364`
  completed `0:0`, sealed exact `B=24576` with zero padding, `K_t=49..85`,
  and context/ROI/residual counts `3899/15853/4824`.  Report file/internal
  SHA-256 are
  `5103e024c7543de52946ae883b79c992096027a9d013b87a41912e3852957464`
  and
  `cb41492ea6723bbebb8beb3add8c515f2ab06f9dbfcaba482a67da48db222bbc`;
  an independent validator replay passed.
- Commit `7e5775e89c0e02428f9af2f6e13c4637a76c7850` added the previously
  missing sample-level dynamic evaluation telemetry.  It serializes every
  tubelet's decoded `(cx,cy,w,h)`, independent width/height/area distributions,
  floor/ceiling saturation, complete `K_t` values and histogram, per-tubelet
  and aggregate role counts, `b_c`, `sum_c b_c^2`, and requested/unique/
  padded/executed receipts.  It fails closed on multi-sample attribution,
  physical/tubelet lineage mismatch, duplicate physical tokens, invalid ROI
  bounds, non-ragged execution, padding, dense Adapter execution, or a ledger
  that differs from exact `B`.  Accuracy telemetry is explicitly excluded from
  timed cost replay.  The exact clean N16R4 Linux source passed the dynamic,
  P0-contract and policy-health regression set `35 passed`; this was a CPU/
  tensor regression, not a CUDA performance or real-data evaluation run.
- Commit `ec8de9f51f85fc81031d82b79e30019d57a381b4` implements the immutable
  M2 execution boundary.  G1/G2 each train from scratch for 60 epochs at seed
  3407 with successful-update-only scheduler/EMA accounting and accept only the
  atomic epoch-59 checkpoint plus sidecar.  Complete Gate accuracy/telemetry
  replays are followed, only after both arms pass, by one same-GPU serial
  `G1 -> G2 -> G2 -> G1` replay that times decode/preprocess/H2D/model/
  postprocess/NMS, records peak memory and an independently sampled raw NVML
  power trace, and retains monotonic energy/NMS windows for validator
  recomputation.  The after-any finalizer emits only descriptive deltas and
  requires M3 before any confirmatory floor claim.  Local compile/Bash/
  whitespace checks and focused contracts passed (`29 passed`); this is
  implementation evidence only, pending exact-source remote precheck.

## Current boundary

The dynamic route is now `tested` at implementation, synthetic CUDA P0, real
Fit-prefix health and sample-level telemetry-unit levels.  D3 has passed and
the two floor configurations are mechanically admissible.  The M2 matched
training/evaluation and separate full-stack cost/energy protocol is implemented
and locally tested, but exact-source remote regression/precheck and the actual
DAG are still pending.  None of the evidence above contains
mAP, end-to-end latency/energy, checkpoint utility or a floor comparison; it is
not `empirically_supported` or `paper_ready`.  `K_t=0` is permitted and covered by
contract tests, but no zero-count tubelet happened to occur in the recorded P0
or 64-update health traces; absence in those traces is neutral rather than a
failed capability gate.
