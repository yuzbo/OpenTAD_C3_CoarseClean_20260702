# GeoRoute correctness Pro review absorption

## Source

- Exact audited commit:
  `df3e54e0c6776544dba20807b2ec100e1a399654`
- Verdict: `HOLD_FOR_CORRECTNESS_FIX`
- Raw review:
  `2026-07-28-georoute-correctness-pro-review-raw.txt`
- Raw SHA-256:
  `e71e1964b75c68c3b05467ba571112e2bd540afa2ce791f991c5cf68ee078600`

## Accepted findings

The project accepts the six blocking findings in full: replicated native
padding without validity, dense-lattice Adapter execution, a geometry-
conditioned `free` control, unmatched route-score pooling, branch-misaligned
hybrid gradients, and unbounded checkpoint/storage behavior. It also accepts
the repaired selector hierarchy: prove an ROI-free NativeTokenSelect base
against fixed/random/geometry-side-channel controls before asking whether
geometry contributes.

The replacement design and experiment contract are recorded in
`docs/methods/2026-07-28-native-token-select-first-correctness-and-parallel-experiment-design.md`.
All listed P1R correctness fixes are implemented locally. This is not yet a
remote test pass or empirical result.

## Deferred recommendations

Soft-to-hard curriculum, FlashVID-inspired relevance/diversity/motion
residuals, A-MoD, dynamic token/depth budgets, and a measured-cost Lagrangian
remain conditional P2 research. They must not enter P1R or obscure a failed
NativeTokenSelect base. Component-level p95/energy and multi-seed
generalization remain required for a publishable claim.

## Clarification of the model claim

The current code performs exact-K selection and packed processing of
source-native tubelets. Geometry, when enabled, changes membership evidence;
it does not create a second source-pixel crop or resized zoom view. Thus
“NativeTokenSelect first” is an experimental evidence order. A “Geometry
Zoom” label is conditional on a strict geometry add-on result and would still
need wording that matches the actual native-token implementation.
