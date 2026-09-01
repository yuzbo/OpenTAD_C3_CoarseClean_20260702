# PJST-D1 focused correction plan v001

Scope is limited to the critic's deterministic bounded set. I will:

1. complete the `VisionTransformerAdapter.forward` PJST keyword interface and propagate pair tensors through normal and temporal-checkpointing calls;
2. pass selector metadata from `SingleStageDetector` to `BackboneWrapper`, where existing bridge metadata is normalized to int64 positions, dense length, and mask; expose explicit PJST OFF/ON backbone config wiring without touching selector/RGB/K/optimizer;
3. make `build_pjst_pair_metadata` reuse the sole `exact_uniform_positions` generator;
4. add focused contract tests for K384/global pair layout, mixed-batch uniform bypass, padding/partial pairs, interface and metadata reachability, checkpointing, finite gradients/no new state, and pre-filter remap trace;
5. run syntax and focused tests only (no training, evaluation, or PRE_RUN), recording any local Torch DLL failure verbatim.

No support forward, second convolution, new trainable state, dynamic-K, Query, or SingleClock expansion is in scope.
