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
- Primary frozen score source: the matched `transition_beta0`
  `epoch_131.pth/state_dict_ema` checkpoint, SHA-256
  `f4ac9891b7cfffd1ab482f28a43086a6e862112f6ffbcb79c7b86c3d2ed935ac`.
  It is used because it is the strongest terminal matched deploy-visible
  transition arm (`64.2755`), while CellCF is lower (`64.0610`) and remains a
  killed local-allocation diagnostic.
- Current CellCF is a tested local phase/content control, not a global
  allocation method.

## Scope

### In scope

- exact family definitions and deterministic solvers;
- valid-prefix and physical-coordinate export;
- geometric GT ceilings;
- deploy-visible coarse-score recovery through the same family-D decoder;
- frozen physical-grid detector candidate evaluation;
- exact family-D decoder incremental-cost evidence;
- code-complete, authorization-gated validation replay and later full-stack
  cost evidence;
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

GT segments are stored explicitly as
`dense_ordinal_aligned_to_exported_physical_axis`. They are never silently
clipped: any endpoint outside the valid prefix fails closed. Training-side GT
is independently reconstructed from the hashed THUMOS annotation, class map,
window start, stride and IoA rule; a self-consistent rewritten JSONL is not a
trust root. Validation/test records must use the validation/testing dataset
subset with `test_mode=True`, and both runtime batches and serialized records
must contain no GT. The canonical alignment axis is decoded frame index:
the decoder frame count and the annotation frame count reconstructed from
`annotation_fps * annotation_duration` must agree within one frame, and every
decoded frame index must lie inside `[0,total_frames-1]`. Decoder FPS and
annotation FPS are different clocks in THUMOS and their full-video clock drift
is retained as an explicit diagnostic, not used to reject an otherwise exact
frame-index alignment. Boundary radii remain dense detector-grid radii;
physical source-frame and seconds caps are reported separately.

The allocation exporter also removes only exact duplicate sliding-window
identities produced by the legacy end-window loop, retaining the first copy.
The removed count is provenance-bound. A repeated identity with different
coordinates fails closed; the base OpenTAD dataset implementation is not
modified.

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

Only solver status `OPTIMAL` with a non-boolean, numerically exact zero MIP
gap supports exact language. A missing solver,
`FEASIBLE`, timeout, numerical ambiguity or contract violation fails closed.
Metric-wise upper envelopes are an optional analysis output, not part of the
formal 32-window run.

Generation solves each privileged D/E MILP once. One independent validation
pass re-solves it with the summary-bound cap, objective and solver options,
then compares positions and the complete objective payload. Candidate and
finalization stages consume the hash-bound validation receipt instead of
solving the same MILP again. Rehashing a feasible but non-optimal selection is
therefore not sufficient to pass. Each full GT solve has a total 300-second
deadline across all sequential objectives. The one-window gate measures
generation and independent-validation time only as a smoke diagnostic. The
formal gate uses the analytical worst case `32 windows x 2 privileged
families x 2 independent passes x 300 seconds = 38,400 seconds` and fails
closed above the preregistered 12-hour allowance. A fast first window cannot
lower that formal bound.

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
tools/bata/validate_duca_allocation_candidate_loss_artifact.py
tools/bata/profile_duca_allocation_solver_cost.py
tools/bata/validate_duca_allocation_solver_cost_artifact.py
tools/bata/finalize_duca_allocation_ceiling_gate.py
tools/bata/finalize_duca_allocation_training_suite.py
tools/bata/authorize_duca_allocation_validation.py
opentad/models/selectors/duca_allocation_artifact_replay.py
scripts/run_duca_allocation_validation_export.sh
tests/test_duca_allocation_families.py
tests/test_duca_exact_physical_solver.py
tests/test_duca_allocation_ceiling_contract.py
tests/test_duca_allocation_candidate_evaluator.py
```

Do not modify the default `Collect` contract. Only the allocation-ceiling
training and validation configs retain decoded `frame_inds`, `avg_fps` and
`total_frames`, so completed CellCF and official baseline pipelines remain
unchanged.

## Artifact contract

Every run binds:

- exact Git commit and clean-tree state;
- resolved config and checkpoint SHA-256;
- annotation, class map, split and video/window manifest hashes;
- the content hash of every raw video file, including the bytes and target
  identity of canonical dataset symlinks;
- terminal checkpoint, backbone pretrain, exact config and suite-manifest
  hashes at every DAG node;
- the exact `n16r4` Slurm cluster identity;
- all DAG jobs held until a scheduler-side `scontrol` pre-release snapshot
  proves Job IDs, names, commands, working directory, GPU requests and exact
  `afterok` dependencies; a second post-release snapshot must prove that the
  holds were removed without changing the DAG;
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
  training-subset windows only. The route-specific training dataset uses
  sliding windows with `ioa_thresh=1e-8`; OpenTAD therefore truncates GT to
  each local window, retains every genuinely action-intersecting window and
  excludes background-only windows from privileged geometry.
- The exact GT and frozen detector diagnostics use a deterministic
  hash-ranked, cross-video round-robin subset of 32 training windows. This
  subset rule is outcome-independent and fixed before results exist.
- THUMOS validation is consumed once by a terminal, preregistered diagnostic.
- Current config `val` and `test` both use the validation subset; different
  overlap ratios do not create an independent held-out population.
- The validation replay uses the official test-style overlap `0.5`, includes
  background windows and carries no runtime GT into the selector. It requires
  a human-issued, hash-bound `GO` receipt over the completed training-side
  evidence. The receipt is atomically consumed once by validation export;
  replay must use the resulting sealed export manifest rather than an
  unstructured environment flag.
- GT is available only inside privileged diagnostic code. The deploy-visible
  D decoder receives only frozen coarse score vectors and physical metadata.

## Experiment order

1. Synthetic mathematical fixtures and exhaustive small-instance parity.
2. Real training-loader coordinate/export gate.
3. Full training-side A-D geometry and coarse recoverability diagnostic.
4. Bounded 32-window training-side privileged D/E ceiling.
5. Frozen physical-grid AdaTAD loss on uniform, deploy-score D, privileged D
   and unrestricted E, using dense-axis GT without selected-axis remapping.
6. Exact family-D CPU decoder incremental-cost profile.
7. Training-side geometry/recoverability GO/HOLD/KILL decision.
8. Only after GO: one sealed validation replay and matched full-stack cost
   profile.

No new selector training is authorized by steps 1-8. Steps 1-7 do not claim
validation mAP or full-stack savings.

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
- all export and frozen-detector evidence is bound to the immutable
  `epoch_131` `state_dict_ema` checkpoint;
- every ceiling, candidate-loss and solver-cost finalizer reopens raw JSONL,
  recomputes selections/metrics and rejects summary-only evidence;
- a hash-bound scheduler receipt and every raw `scontrol` record remain
  unchanged from gate execution through finalization;
- privileged GT MILPs are generated once and independently replayed once;
  later stages verify the receipt and artifact hashes without redundant
  re-solving;
- the finalizer recomputes the preregistered 32-window
  hash-video-round-robin subset from the full training export;
- no model-training job is launched before the ceiling decision.
