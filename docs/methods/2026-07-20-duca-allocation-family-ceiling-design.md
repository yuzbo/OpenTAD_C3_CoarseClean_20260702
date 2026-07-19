# DUCA allocation-family ceiling design

## Decision

Implement a bounded, read-only ceiling package before any new DUCA selector
training. The package measures whether a globally adaptive exact-K family has
useful physical boundary headroom, whether current deploy-visible coarse
signals can recover that headroom, and whether the candidate can plausibly
save full-stack cost.

This is offline full-window TAD. It is not Online TAD.

## Source of truth

- Repository: `yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Base commit: `4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- Immutable trained model ancestor:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`
- Current CellCF is a tested local phase/content control, not a global
  allocation method.

## Scope

### In scope

- exact family definitions and deterministic solvers;
- valid-prefix and physical-coordinate export;
- geometric GT ceilings;
- deploy-visible coarse-score recovery through the same family-D decoder;
- frozen physical-grid detector candidate evaluation;
- dense/uniform/candidate full-stack cost evidence;
- fail-closed artifacts, manifests, hashes and focused tests.

### Out of scope

- training a new CARA selector;
- Dynamic MUST;
- X3D/SlowFast;
- new detector heads;
- selected-axis GT remapping;
- changing completed CellCF checkpoints or results;
- claiming detector mAP is an exact combinatorial oracle.

## Physical coordinate contract

Each sample carries one contiguous valid prefix and three aligned coordinates:

1. dense candidate ordinal;
2. actual decoded source-frame index;
3. actual timestamp in seconds.

The exporter must retain actual `frame_inds`, decoder FPS and annotation FPS.
It may reconstruct the expected regular grid from
`window_start_frame + dense_index * snippet_stride`, but it must compare that
grid with decoded indices and fail closed on an unregistered mismatch.
Padding is excluded from all families.

The primary cap policy is `uniform_reference`:

```text
Delta(sample) = max physical interval of exact-uniform(L, K_eff)
```

The candidate selection must not leave a larger physical interval than the
same-budget exact-uniform reference. This guarantees uniform inclusion without
an arbitrary 10/15-frame hyperparameter and remains meaningful when FPS,
window start or valid length changes.

The implementation also supports explicit caps in source frames or seconds.
The fixed stride-4 `cap_frames=15` fixture must reproduce effective interval
12 and minimum scaffold cardinalities 255/382 at `L=768,K=384`.

Every artifact reports dense hole, source-frame interval and seconds interval
separately.

## Registered families

### A: Exact uniform

The unique endpoint-inclusive, round-half-to-even exact-uniform selection.

### B: Current one-per-uniform-cell family

Exactly one position from every canonical uniform Voronoi cell. Additive
scores are optimized independently inside each cell. It cannot transfer quota
between cells.

### C: Fixed scaffold plus residual

The scaffold is the exact minimum-cardinality subset of uniform anchors that
satisfies the physical cap. Remaining positions are selected globally by the
frozen additive score. This family is diagnostic because a strict physical cap
may consume almost all K slots.

### D: Global exact-K physical-gap family

Select exactly `K_eff=min(requested_K, valid_L)` positions on a source-to-sink
physical path whose edge intervals respect the frozen cap. No cell quota is
fixed. Additive scores are solved exactly by dynamic programming.

This is the primary deployable-family ceiling and candidate generator.

### E: Privileged unrestricted GT family

Select exact-K without the coverage cap using GT objectives. This family is
evaluation-only, marked `privileged=true, deployable=false`, and can never feed
deployment selection.

## Solver design

### Additive family solver

Use a source-to-sink DAG dynamic program with state `(selected_count,
last_position)`. Scores are integer-quantized with a manifest-bound scale.
Exactness applies to the serialized quantized vector. Ties choose the
lexicographically smallest selected-position tuple.

### Canonical GT solver

Use a deterministic mixed-integer path model when SciPy HiGHS MILP is
available. It encodes exact-K, path flow, endpoint-hit variables and
both-endpoint variables. Sequentially pin the preregistered objective vector:

1. both endpoints at dense radii 0, 1, 2, 4;
2. distinct endpoint hits at radii 0, 1, 2, 4;
3. minimum total endpoint-to-selection distance;
4. short-action support;
5. minimum selected background count;
6. exact-uniform overlap;
7. deterministic lexicographic tie break.

Only solver status `OPTIMAL` supports exact language. A missing solver,
`FEASIBLE`, timeout, numerical ambiguity or contract violation fails closed.
Metric-wise upper envelopes are emitted separately from the one canonical
lexicographic solution.

Detector loss and mAP are evaluated only as frozen empirical diagnostics.

## Components

Add:

```text
tools/bata/duca_allocation_families.py
tools/bata/duca_exact_physical_solver.py
tools/bata/export_duca_allocation_ceiling_inputs.py
tools/bata/diagnose_duca_allocation_family_ceiling.py
tools/bata/validate_duca_allocation_ceiling_artifact.py
tools/bata/evaluate_duca_allocation_candidates.py
tests/test_duca_allocation_families.py
tests/test_duca_exact_physical_solver.py
tests/test_duca_allocation_ceiling_contract.py
tests/test_duca_allocation_candidate_evaluator.py
```

Modify the default `Collect` metadata only to retain actual decoded
`frame_inds`, `avg_fps` and `total_frames`. Existing finite-candidate and
CellCF tools remain unchanged.

## Artifact contract

Every run binds:

- exact Git commit and clean-tree state;
- resolved config and checkpoint SHA-256;
- annotation, class map, split and video/window manifest hashes;
- physical-coordinate input hash;
- family, cap and score-quantization specification;
- solver identity/version/options;
- objective specification;
- output JSONL and summary hashes.

Per-sample output includes family status, exact/deployable/privileged flags,
selection, optional scaffold/residual decomposition, objective vector, all
three gap reports and boundary/allocation metrics.

Existing outputs are never overwritten. Unknown fields in a strict artifact,
non-finite values, duplicate selections, noncontiguous masks, GT in a
deployable score path or hash mismatch fail closed.

## Data and split protocol

- Family/cap/objective implementation is fixed using deterministic
  training-subset windows only.
- THUMOS validation is consumed once by a terminal, preregistered diagnostic.
- Current config `val` and `test` both use the validation subset; different
  overlap ratios do not create an independent held-out population.
- GT is available only inside privileged diagnostic code. The deploy-visible
  D decoder receives only frozen coarse score vectors and physical metadata.

## Experiment order

1. Synthetic mathematical fixtures and exhaustive small-instance parity.
2. Real training-loader coordinate/export gate.
3. Training-side A-E geometry and coarse recoverability diagnostic.
4. One sealed validation geometry/recoverability diagnostic.
5. Frozen physical-grid detector finite-candidate evaluation.
6. Dense/uniform/candidate full-stack cost profile.
7. GO/HOLD/KILL decision.

No new selector training is authorized by steps 1-6.

## Decision rule

- Kill the global selector route if D and E have no useful boundary or frozen
  detector headroom over uniform.
- Hold/replace coarse evidence if privileged D has headroom but deploy-visible
  scores cannot recover it.
- Kill the pre-backbone efficiency claim if measured end-to-end savings do not
  remain positive after the coarse frontend.
- Only if all three gates pass may the project design and audit a trainable
  global-D selector.

## Acceptance criteria

- all registered synthetic contracts and small exhaustive parity tests pass;
- every exact sample reports `OPTIMAL`;
- exact-K, valid-prefix, uniform inclusion and physical cap compliance are
  100%;
- no selected-axis GT remapping is present in physical-grid evaluation;
- no validation/test GT reaches deployable score generation;
- raw detector and cost evidence is hash-bound and reproducible;
- no model-training job is launched before the ceiling decision.
