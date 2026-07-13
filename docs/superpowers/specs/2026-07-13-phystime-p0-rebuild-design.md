# PhysTime P0 Rebuild: G0 Provenance and G1a Matched Metric Control

## Status

Approved for implementation on 2026-07-13. This specification supersedes the
K=384 feature-count assumption in PhysTime-AdaTAD 1.0. It does not claim that
SM-PTAF, mass residuals, or a new paper method have been validated.

## Question

Before adding capacity, can physical time help a raw-video sparse AdaTAD model
when observation count, native VideoMAE token count, query count, trainable
parameters, and supervision are correctly matched?

## G0: Provenance Before Training

The system must distinguish three counts:

- `K=384`: selected raw RGB observations entering VideoMAE;
- `J=192`: native temporal tubelet tokens emitted by a tubelet-size-2
  VideoMAE path;
- `Q0=192`: the base ActionFormer candidate grid, equal to native J in G1a;
- `Qsum=378`: total candidates across the official six-level J-based pyramid.

The native `J=192` features must never be interpolated to 384 in G1a. Each
patch-embedding tubelet consumes two raw-frame input slots. Those atoms are
exact patch-input lineage only: chunk-wide ViT attention and adapter mixing
mean the final token can depend on the complete 16-frame chunk. The audit must
not relabel two patch atoms as the final feature's exact support.

The G0 audit must fail closed when:

- K, J, tubelet size, mask prefix, or metadata counts disagree;
- a token support is silently replaced by its convex envelope;
- sampling provenance is missing or claims GT-dependent selection;
- any model path labels K raw frames as J feature tokens;
- the two compared configurations have different parameter names, shapes, or
  trainable counts.

G0 emits JSON containing K/J/Q, per-level Q, selected-index checksum, atom-gap
statistics, envelope inflation, parameter parity, and inference-metadata
checks. G0 does not train a model and cannot establish effectiveness.

## G1a: Matched Temporal-Metric Control

Both arms use the same:

- the same standard AdaTAD GT-aware `random_trunc` training-window crop,
  followed by deterministic, no-GT within-window `random_fixed_subsample`
  K=384 observations; validation/test windowing and subsampling use no GT;
- VideoMAE-S backbone and adapter;
- native J=192 temporal feature sequence;
- ActionFormer projection, pyramid, head towers, losses, optimizer, seed, and
  post-processing;
- candidate count Q and all trainable parameter shapes.

GT, candidates, regression, decode, NMS, and evaluation remain in canonical
seconds for both arms. Only the common temporal-coordinate tensor differs:

1. `uniform_rank_seconds`: J ranks are placed uniformly over the accepted
   physical-time window.
2. `physical_time_seconds`: J positions are the native tubelet atom-mean
   timestamps on the original timeline.

Both arms use the same metadata key, coordinate-mapping code, physical-grid
head, and seconds-domain post-processing. Patch atoms remain available for
audit, but G1a does not add a mass-residual branch or pretend that an atom mean
is a complete support measure. This limitation is explicit.

## Implementation Boundary

The implementation may add a reusable native-tubelet geometry transform and a
small post-backbone mask alignment hook. It must reuse the official
ActionFormer projection/head and existing physical-grid machinery. It must not
duplicate ActionFormer, add a learned selector, use actionness, use a teacher,
add paired consistency, change K dynamically, or introduce train-only
information into validation/test decisions.

## Deployment Gate

Deployment order is fixed:

1. focused unit tests and config parity;
2. G0 synthetic audit;
3. real THUMOS raw-video one-step CUDA gate for both arms;
4. a short matched pilot for both arms;
5. inspect loss stability, candidate counts, and preliminary mAP before any
   full run or G1b/G2 implementation.

G1b (a shared neutral J192-to-Q0=384 lift, yielding Qsum=756) and G2
(support-mass residuals) remain
blocked until G1a produces interpretable evidence. One experiment changes one
uncertainty.

## Success Criteria

- no temporal interpolation from J=192 to K=384;
- identical selected-index and decoded-RGB checksums between arms;
- identical parameter schema and Q counts;
- identical initialized state and optimizer-group schemas;
- both real-data gates complete with finite forward/backward and non-zero
  detector gradients;
- both pilots train and evaluate without metadata leakage or coordinate-unit
  mismatch;
- results are recorded as diagnostic evidence, not promoted to a paper claim.
