# DUCA full-window final-method repair status

## Method scope

The final method is a cache-free, runtime-generated, full-window, jointly trained pre-backbone frame-selection plugin. The low-cost C3 coarse probe and selector may inspect the complete input window before the expensive VideoMAE/AdaTAD backbone consumes the selected frames. It is not a streaming, causal, or prefix-invariant model.

The paper-main route is currently fixed-budget DUCA with the official OpenTAD AdaTAD/VideoMAE-S/ActionFormerHead backend. The existing MUST route remains a padded-cap dynamic-budget diagnostic and is not paper-main evidence.

## Verified repairs

| Finding | Repair | Verification |
| --- | --- | --- |
| Selector leaf losses were counted through aggregate and alias entries | Removed `total_loss` and the utility alias; detector wrappers reject aggregates, aliases, non-scalar, non-finite, or non-loss entries | Gradient-multiplicity and detector integration tests |
| Hard inference and detector-gradient policies differed | Added a full-window exact-K/max-gap structured policy with a shared hard Viterbi and entropy-regularized feasible-state relaxation | Brute-force optimum, exact budget, max-gap, hard-forward identity, and detector-gradient tests |
| Stable warmup did not stabilize hard detector input | Added a deterministic structured reference path during warmup and a short optimizer-step-driven transition | Warmup invariance and post-transition gradient tests |
| Boundary utility accumulated action-interior mass | Added normalized start/end endpoint targets and narrow boundary context; utility proxy is endpoint dominated | Short/long instance mass, action-body suppression, and head-gradient tests |
| Boundary and utility score scales could bypass configured weights | Added separate start, end, context, and utility heads with bounded score contributions and direct leaf supervision | Direct nonzero gradient tests for all supervised heads |
| Dense/selected/original-time mapping clipped head/tail boundaries and mixed point conventions | Added sentinel knots, half-open targets, and integer temporal points consistent with ActionFormer | Synthetic head/tail round-trip and GT remap tests |
| Frozen C3 sources could be switched back to train mode by the parent model | Overrode `train()` so frozen sources remain in eval mode | Parent-train regression test |
| The official gradient gate still accepted a precheck detector | Formal suite now runs the real ActionFormerHead one-step proof and optimizer coverage gate | Official model build, `losses["cost"].backward()`, and nonzero probe/selector gradients |
| Dynamic cost mixed hard K, ST K, and soft expected K | Split canonical hard requested/effective/unique, padded/backbone, ST, and soft expected budget fields; normalized dual update | Dynamic accounting and anti-bang-bang tests |
| MUST claimed main-method status without variable detector compute | Marked the current route `diagnostic_only`, `main_method_candidate=False`, and `dynamic_compute_realized=False` | Fail-closed MUST config validator |
| “online” was used as a streaming claim | Formal contracts now use runtime-generated, cache-free, full-window terminology and declare `streaming=False` | Fixed and MUST contract validators |

## Structured-policy scale check

For `T=768`, `K=384`, and maximum unselected hole 15 on remote CPU, the structured policy kernel took approximately 0.47 s forward and 0.43 s backward after Torch import. The full process peak RSS was about 721 MB; the incremental policy footprint over a hard-only process was about 62 MB. Slot-wise normalization keeps the numerical soft budget equal to K on longer sequences.

## Evidence status

- Existing runs based on `7e3a508` or earlier commits are diagnostic only and cannot validate the repaired method.
- No new full training should be submitted until the final focused suite, official validator, official one-step backward proof, and shell/config prechecks all pass from one clean commit.
- After the gate, the first paper runs should be fixed K=384/256/128 from that single commit.
- MUST must remain appendix/diagnostic until detector/backbone compute actually changes with K and matched-actual-K accuracy plus latency gates pass.
- X3D and SlowFast frozen priors remain train-free appendix baselines, not the jointly trained main method.
- The legacy JCT suite is fail-closed by default because it mixes padded-cap MUST and X3D appendix jobs. Diagnostic replay requires an explicit `ALLOW_LEGACY_DIAGNOSTIC_SUITE=1` override.

## Final verification

- Local syntax gate: 29 Python files compiled and `git diff --check` passed.
- Remote unified focused gate: 128 tests passed.
- Remote fixed and MUST config validators passed; MUST reports diagnostic-only and no realized dynamic compute.
- Remote official AdaTAD/ActionFormer one-step proof passed with complete frame-selector optimizer coverage and nonzero detector-loss gradients into the trainable coarse probe and selector.
- Remote formal fixed launcher `PRECHECK_ONLY=1` passed, including pretrained checkpoint resolution and 19 config-contract tests.

## Remaining risks

1. A full VideoMAE-S GPU precheck and at least one short training smoke still need to confirm end-to-end memory and throughput beyond the scaled official-head proof.
2. The structured relaxation is exact in budget and feasible-state definition, but its detector-utility direction still needs a hard one-swap finite-difference surrogate audit.
3. The endpoint/context target is a GT boundary-utility proxy, not detector-derived utility. The paper must retain that name unless a train-only detector sensitivity teacher is implemented and audited.
4. A true dynamic-compute MUST implementation is not present in this repair and must not be implied by padded masks or soft expected K.

## First GPU smoke finding

The first clean-snapshot fixed-384 GPU smoke (`1154930`) reached the real training loop and failed before the first logged iteration with `baddbmm_cuda not implemented for Byte`. The full-train loader correctly supplied a `uint8 [B,N,C,T,H,W]` window, but the new structured detector-gradient bridge used that tensor directly in `einsum`. The bridge now follows the existing DUCA bridge contract: non-floating dense and hard tensors are promoted to floating point before soft assignment, while preserving the exact hard forward value. A 6D uint8 official-ASFormer regression test covers forward dtype and detector-to-selector backward.
