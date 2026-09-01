# DUCA physical allocation-family Pro review absorption

## Source identity

- Repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Audited evidence commit:
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- Immutable trained-model ancestor:
  `1642f265e48391418a7c8a4a087e33e2b7bf6899`
- User attachment:
  `C:/Users/skywalker/.codex/attachments/6190aa05-a97e-4c41-82a7-7c74bed997ad/pasted-text.txt`
- Byte-identical raw archive:
  `docs/methods/reviews/2026-07-20-4ce69c8-duca-physical-allocation-family-pro-review-raw.txt`
- Raw/archive size: `75,867` bytes, `2,001` lines
- Raw/archive SHA-256:
  `E40A69BD2DA9EBE32B41B45A136C2AA1A9FB8109A4875A16E2E3ABB7AF8FCC14`

Local Git independently verifies that both commits exist, that `1642f26` is
the merge base, and that `4ce69c8` is exactly 12 commits ahead and zero
commits behind.

## Project verdict

`SUBSTANTIAL_ACCEPT / NOT_FULL_ACCEPT / HOLD_AND_REVISE_FAMILY`

The central diagnosis and the immediate stop condition are accepted:

1. current one-frame-per-uniform-cell CellCF is local phase/content
   substitution, not cross-region adaptive budget allocation;
2. fixed `G=3, 192 scaffold + 192 residual` must not be promoted into the
   final method;
3. the next bounded scientific question is an exact allocation-family ceiling
   on real physical coordinates, not another long selector training run;
4. the clean primary ceiling family is global exact-K under a physical maximum
   interval, with exact uniform explicitly feasible;
5. existing physical-grid ActionFormer should be reused with dense-time GT and
   selected-axis GT remapping disabled;
6. current CellCF remains a historical local-content control, and Dynamic MUST
   remains frozen.

The response is not accepted verbatim as a production specification. Its
physical cap, statistics, proposed code and split protocol still need the
corrections below.

## Independently verified facts

### Current CellCF semantics

At `4ce69c8`, the formal CellCF configuration has
`max_unselected_hole=None`, `detector_gradient_mode="none"`,
`detector_output_coordinate_space="selected_axis_index"` and
`remap_gt_to_selected_axis=True`.

The selector gathers actual acquisition positions, while local-cell decoding
passes uniform cell anchors as detector-grid positions. The result therefore
measures nonuniform content substitution under uniform detector geometry. Its
detector-derived counterfactual utility is detached supervision, not direct
detector-loss backpropagation.

### Exact-uniform and local-cell geometry

Independent reconstruction of the committed round-half-to-even and midpoint
rules for `L=768, K=384` gives:

- 384 unique endpoint-inclusive uniform anchors;
- anchor gaps: 382 gaps of 2 dense indices and one gap of 3;
- cell lengths: 382 cells of length 2, one cell of length 3 and one cell of 1;
- exact-uniform maximum unselected dense hole: 2;
- a legal one-per-cell CellCF realization can attain dense hole 3.

Thus CellCF has large within-cell combinatorial freedom but zero regional
quota-vector freedom.

### Dense-hole `G=3` is not a physical 15-frame contract

For the code's dense-hole definition, the minimum scaffold cardinality at
`L=768, G=3` is 192. A 192-anchor subset of exact uniform does exist, but its
ordinal convention must be explicit. With zero-based anchor ordinals it keeps:

```text
1, 3, ..., 191, 192, 194, ..., 382
```

This subset has 192 positions and maximum unselected dense hole 3. Writing the
same formula with unspecified or one-based subscripts is ambiguous and can
produce an invalid tail hole of 4.

The formal data path uses `feature_stride=4` and `sample_stride=1`, so one
dense index represents four source-frame indices. Dense hole 3 permits a
four-step selected-position interval, namely 16 source frames. It is not a
15-source-frame guarantee.

### Consequence of an original-frame cap of 15

If the scientific contract is explicitly:

- first selected center to the first valid source frame at most 15 frames;
- every adjacent selected-center interval at most 15 frames;
- last selected center to the last valid source frame at most 15 frames;

then legal coordinates `0,4,8,...` make the effective discrete cap 12 frames.
Independent dynamic programming confirms:

- exact uniform is feasible and its maximum source-frame interval is 12;
- a cap of 10 source frames excludes exact uniform;
- the minimum arbitrary fixed scaffold has 255 positions;
- the minimum fixed scaffold restricted to exact-uniform anchors has 382
  positions, obtained by dropping only the first and last anchors.

Consequently a fixed mandatory scaffold that both guarantees this physical
cap and remains a subset of exact uniform leaves only two residual slots. It
cannot support the intended `192+192` adaptive-allocation claim.

### Dataset and evaluation split

The audited base config uses:

- `train.subset_name="training"`;
- `val.subset_name="validation"` with overlap 0.25;
- `test.subset_name="validation"` with overlap 0.5;
- evaluation subset `"validation"`.

Therefore `val` and `test` are not independent semantic splits. Different
window overlap does not create a new held-out population.

## Accepted design revision

The allocation ceiling should compare:

1. A: exact uniform;
2. B: current one-per-uniform-cell CellCF;
3. C: fixed scaffold plus global residual, retained as a diagnostic family;
4. D: global exact-K under an explicitly frozen physical maximum interval;
5. E: privileged unrestricted GT reference, diagnostic only.

Family D is the primary ceiling because it contains uniform while allowing
background cells to release quota and transition regions to receive multiple
observations. A scaffold/residual representation may be computed after solving
for a selected set, but it must not silently become a mandatory fixed
scaffold.

The exact package must bind valid-prefix positions to actual decoded source
frame indices and timestamps, report dense hole, source-frame interval and
seconds interval separately, and fail closed on padding, coordinate mismatch,
non-optimal exact-solver status, GT leakage or selected-axis remapping.

Detector mAP is not an exact combinatorial objective. Exact language applies
only to encoded geometric/additive objectives. Frozen physical-grid detector
evaluation is a secondary empirical diagnostic.

## Reservations and required corrections

### 1. The value 15 is still a proposed scientific contract

The code proves the conversion after a unit is chosen; it does not prove that
the user's earlier phrase "10 or 15 frames" meant original decoded frames
rather than dense candidate-grid positions. Before implementation, the
project must explicitly choose one of:

- original decoded frame index;
- dense candidate-grid index;
- seconds.

If original frame index is chosen, the `15 -> effective 12` and `255/382`
results above are binding. If dense-grid index was intended, those numbers do
not define the requested family. No code or paper text may silently choose
between these interpretations.

### 2. The supplied code is a blueprint, not merge-ready code

- The canonical GT oracle shown in the response omits its own declared
  endpoint-distance, short-action-support and background-count objectives.
- Floating scores are rounded to integers at a fixed scale. The DP is exact
  only for the quantized score vector, not automatically for the original
  floating ordering.
- The lexicographic CP-SAT tie break may invoke hundreds of full solves per
  sample and needs a measured runtime bound or a more efficient exact pinning
  implementation.
- GT radii are expressed on dense indices while the hard cap is expressed in
  source frames. Both coordinate systems must be named and serialized.
- Annotation FPS, decoder FPS and actual decoded frame indices are not yet
  bound in the current ceiling artifact.
- No proposed module or listed test has been integrated or executed in this
  repository.

### 3. Statistical gates require a paired design specification

The response mixes paired video bootstrap, simultaneous confidence bounds,
MDE and a two-independent-group seed formula. Before preregistration it must
define:

- the multiplicity correction for the simultaneous bound;
- whether `sigma_seed` is per-arm variance or paired-difference variance;
- whether seeds are paired across methods;
- the practical effect threshold separately from statistical significance;
- a seed budget based on pilot or external variance rather than seed-0.

MDE is a planning quantity, not a post-hoc natural constant.

### 4. The validation set cannot be repeatedly called sealed

Because current `val` and `test` both use the validation subset, family design,
cap selection, objective order and score calibration must use only a frozen
training-side partition. The validation subset may be consumed once under a
predeclared terminal protocol. It cannot serve repeatedly as both model
selection data and a sealed final test.

### 5. Family D is a ceiling, not yet the final learned method

Even if privileged D has geometric headroom, the paper route survives only if
deploy-visible low-cost actionness, transition, uncertainty and temporal
features can recover a useful part of that headroom at a real full-stack cost
saving. A GT-optimal D solution does not by itself justify CARA training or a
paper claim.

## Frozen next action and status

The only newly authorized work from this review is a bounded, read-only
allocation-family ceiling package after the physical unit is explicitly
frozen. It may implement deterministic diagnostic solvers, coordinate
export/validation, tests and frozen-model evaluation. It does not authorize:

- fixed `192+192` CARA training;
- a new detector implementation;
- selected-axis GT remapping;
- Dynamic MUST, X3D/SlowFast, more detector heads or broad ablations;
- any claim that DUCA is empirically supported or paper-ready.

Status after absorption:

- current CellCF adaptive-allocation main claim: `killed`;
- CellCF local phase/content control: `tested_diagnostic`;
- DUCA-CARA working idea: `discussed / hold_and_revise_family`;
- allocation-family ceiling: `designed`, not implemented or run;
- physical cap unit/value: `unfrozen`;
- C3/C4/C7 and paper readiness: `unproven`.
