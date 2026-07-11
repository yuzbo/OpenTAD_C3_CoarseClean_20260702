# DUCA Shared-ASFormer Transition-Only Implementation Plan

## Objective

Implement an isolated fixed-budget DUCA candidate that preserves the official
ASFormer and AdaTAD forward paths while replacing the current direct-boundary
selector with a single transition-only scorer. Keep commit `a5e1774` behavior as
the default baseline.

## Non-Negotiable Contracts

1. The coarse hidden state must be the official ASFormer encoder state, not the
   pre-ASFormer spatial stem output.
2. Selection utility may consume only temporal changes of coarse logits,
   entropy, and encoder state. It must not consume absolute hidden state, RGB
   means, or dedicated start/end/context/utility/radius heads.
3. Auxiliary transition supervision may update the shared ASFormer trunk and
   transition scorer. Boundary coverage and detector losses may update the
   scorer through the differentiable structured selector, but must not update
   the coarse classifier through the selector branch.
4. Hard inference and soft training use the existing exact-K/max-gap dynamic
   program. Fixed-384 is one policy, not a separate implementation.
5. The learned policy is introduced continuously from a feasible uniform
   reference; no modulo-step duty-cycle switching is allowed for this variant.
6. Official AdaTAD/ActionFormer behavior outside the frame-selector boundary is
   unchanged and all trainable selector parameters are covered exactly once by
   the optimizer.

## Tasks

1. Add failing tests for true ASFormer hidden extraction and logit equivalence.
2. Add failing tests for transition descriptor invariance, one shared scorer,
   protected gradients, and continuous policy interpolation.
3. Implement provenance-aware official-ASFormer encoder-state extraction using
   the original model forward plus a temporary encoder hook.
4. Add an opt-in `transition_only` adapter path and keep `direct_boundary` as
   the default baseline.
5. Add transition distribution and differentiable boundary-coverage losses;
   disable legacy direct-head losses in the new configuration.
6. Add continuous homotopy scheduling and expose audit fields for alpha,
   selected count, max gap, and repair status.
7. Add exact optimizer-coverage validation for selector parameters.
8. Add the fixed-384 official AdaTAD configuration, validator, GPU1 launcher,
   and focused full-model one-step tests.
9. Run local compile/unit checks, remote Torch tests and precheck, then commit,
   push, and deploy a matched fixed-384 experiment only after all gates pass.

## Evidence Gates

- ASFormer logits are bitwise/effectively identical with hidden capture enabled.
- Captured hidden tensor is produced by the official encoder and has non-zero
  gradient under transition supervision.
- Detector-only backward gives non-zero gradient to the transition scorer and
  zero gradient to the coarse probe/action head through the selector branch.
- Hard selected indices satisfy exact K and max-gap; soft marginals are finite
  and policy alpha is monotonic over training progress.
- Official detector one-step train/test passes and every required trainable
  selector parameter appears in exactly one optimizer group.
- No full run is submitted from a failed precheck.
