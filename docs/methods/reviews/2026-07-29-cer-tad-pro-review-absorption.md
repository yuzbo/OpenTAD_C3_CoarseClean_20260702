# CER-TAD Pro review absorption

Date: 2026-07-29

Project verdict: `ACCEPT_WITH_MAJOR_REVISION / READY_PREEXPERIMENT_ONLY`

The user-provided Pro review is preserved as design-review evidence, not as an
implemented method or experiment result. It correctly identifies the current
method as native-token routing rather than Geometry Zoom and correctly closes
Free NativeTokenSelect v1 as the primary candidate: the exact-source run gives
strong descriptive negative evidence, even though the ROI decode failure
prevented the old selector from issuing a formal decision.

## Accepted

- Do not promote Free v1, Hybrid v1, the old Free-first selector, or the failed
  namespace.
- Do not call the current method Geometry Zoom.
- Diagnose the complete old checkpoints before inventing another large model.
- Use a hard without-replacement likelihood if a future hard policy is trained.
- Separate token-support changes from geometry/coordinate representation.
- Require exact-index decode integrity, numerical known-answer tests,
  multi-seed accuracy, and full-stack measured cost before a paper claim.
- Treat EVAD and scout-to-local-crop work as important direct novelty risks;
  Plackett-Luce/Gumbel-top-k is standard machinery, not the contribution.

## Accepted with conditions

- Phase M may replay only immutable completed checkpoints in a new diagnostic
  namespace. Every old prediction-bearing arm must reproduce its prediction
  SHA-256 exactly; replay output is diagnostic and cannot repair or complete
  the failed seven-arm experiment.
- The existing single-family Gumbel-top-k sampler and ordered
  Plackett-Luce log likelihood are mathematically coherent, and the positive
  `(detector risk - baseline) * log probability` sign is correct for risk
  minimization. They still require numerical selected/unselected-gradient and
  variance tests before training is scientifically justified.
- CER's context / geometry / residual role decomposition is a plausible new
  hypothesis, but only after estimator and representation isolation pass.

## Rejected or unresolved

- Immediate implementation and deployment of the eleven-arm P1 is rejected.
  It conflicts with the frozen seven-arm contract and old selector.
- The proposed `+0.50 pp` and `+0.30 pp` gates are rejected as confirmatory
  thresholds because they were proposed after observing the old development
  results and have no independent variance or power basis.
- Dynamic role allocation is underspecified: the review does not fully define
  or implement the role-count likelihood, and changing role allocation at fixed
  total K is not dynamic total compute.
- The critic, boundary head, coverage penalty, temporal-stability objective,
  advantage normalization, and their weights are not reproducibly specified.
- Current code cannot isolate geometry support from representation:
  `use_absolute_coordinates` jointly controls absolute and ROI-relative
  coordinates, while `geometry_projection` is always added.
- The review's linked implementation framework was not delivered as a
  repository artifact and therefore cannot be treated as code.

## Authorized next action

Only the development-only estimator/representation preexperiment in
`docs/methods/2026-07-29-georoute-estimator-representation-preexperiment.md`
is authorized. It consists of a full decode census, numerical KATs, and six
prediction-hash-preserving Phase M replays. Full CER remains `discussed`.

An additional independent Pro round was attempted but could not be executed:
Oracle API mode has no `OPENAI_API_KEY`, Rosetta has no active authenticated CDP
endpoint, Claude review authentication is revoked, and Gemini review lacks a
valid API/trusted CLI session. These unavailable routes are not counted as an
approval.
