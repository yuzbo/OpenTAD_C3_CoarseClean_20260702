# GeoRoute estimator and representation-isolation preexperiment

Date: 2026-07-29

Status: `tested_complete_go_pilot_design_only`

Authorization boundary: development diagnostics only. This protocol does not
authorize a new P1/P2/P3 study, official test, efficiency claim, Geometry Zoom
claim, or CER-TAD paper claim.

## Research decision

The completed exact-source run is strong descriptive negative evidence against
the current Free NativeTokenSelect v1, although the ROI decode failure prevented
the old seven-arm selector from issuing a formal receipt. The next scientific
question is therefore not whether to add more heads to Free v1. It is whether:

1. the hard-route estimator can assign detector-risk credit to swaps involving
   unselected native tokens; and
2. any apparent geometry benefit changes token membership rather than merely
   entering the detector through coordinate or geometry projections.

The proposed full CER-TAD model remains `discussed`. Dynamic role allocation,
boundary conditioning, a learned critic, temporal-stability penalties, and an
eleven-arm matrix are deferred.

## Hypotheses and falsifiers

### H1: hard-route credit reachability

For a single-family residual exact-K route, ordered Plackett-Luce (PL)
score-function training exposes nonzero direct gradients to both selected and
unselected valid logits. The current selected-only straight-through (ST)
amplitude surrogate does not expose direct unselected membership gradients.

Falsifier: any exact-likelihood, probability-normalization, sign, selected
gradient, or unselected-gradient known-answer test fails.

Passing H1 is only a mathematical authorization for a matched estimator pilot.
It is not evidence that PL improves detection.

### H2: support/representation separability

The sparse temporal adapter can independently enable:

- absolute source coordinates;
- ROI-relative coordinates; and
- the geometry projection.

With all three disabled, changing geometry and coordinates while holding
selected features fixed must leave the output bitwise unchanged and must expose
no gradient path to geometry or coordinates.

Falsifier: the disabled representation path changes output or carries a
geometry/coordinate gradient; or the legacy all-enabled path is not numerically
equivalent to the pre-change formula.

### H3: diagnostic replay non-interference

Opt-in route telemetry can replay a completed development checkpoint without
changing its prediction JSON bytes.

Falsifier: the replay prediction SHA-256 differs, the development population is
incomplete, the checkpoint/config hash is not the recorded source artifact, or
any replay uses GT, teacher, oracle, raw-prediction cache, or official test.

## Phase D: full exact-index decode census

Use one original bound development config because all seven old arms share the
same data population and pipeline. Instantiate the exact sliding-window dataset
and retrieve every item for two complete passes. Record:

- source and replay commits;
- bound-config SHA-256;
- dataset length and population digest;
- pass count and successful item count;
- video ID, window-center endpoints, and exception fingerprint for every
  failure;
- explicit `official_test_opened=false`.

Any failed item makes Phase D `FAIL_DATA_DECODE` and prevents Phase M and all
training. The retry ceiling is not increased as a substitute for the census.

## Phase K: numerical known-answer tests

The receipt passes only if all conditions hold:

1. ordered PL log probability matches a manual sequential softmax within
   `1e-7`;
2. probabilities of every length-K ordered sample for `N=4, K=2` sum to one
   within `1e-6`;
3. the exact expected score-function gradient matches the exact risk gradient
   within `1e-6`;
4. selected and unselected PL-logit gradients are finite and have absolute
   magnitude greater than `1e-8` in the frozen known-answer case;
5. the matched ST case has finite selected gradients greater than `1e-8` and
   exactly zero direct unselected membership gradient;
6. the representation-disabled adapter is invariant to geometry/coordinate
   perturbations and has zero geometry/coordinate gradients;
7. each individually enabled representation channel is effective; and
8. the legacy all-enabled computation matches the pre-change formula within
   `1e-7`.

The KAT receipt is `PASS_MECHANICAL_ONLY`; it contains no mAP result.

## Phase M: instrumentation-only replay

Replay only the six old cells that have both a unique final checkpoint and a
recorded prediction hash:

- dense;
- fixed;
- fixed plus geometry representation;
- random;
- Free NativeTokenSelect v1; and
- Hybrid v1.

ROI has a final checkpoint but no source prediction and therefore cannot prove
bitwise non-interference. It is excluded from Phase M parity. A later
diagnostic-only ROI replay may be considered only after Phase D passes, and it
must not be pooled with the six parity cells.

Each replay uses a new immutable namespace and records compact per-window,
no-GT telemetry:

- selected-set adjacent-tubelet intersection, union, Jaccard, and lineage
  retention;
- selected native-coordinate mean, span, and quadrant occupancy;
- geometry area and temporal motion;
- selected versus unselected ROI/residual score separation;
- selected versus unselected soft-surrogate statistics; and
- frozen role counts and exact-K integrity.

The replay must reproduce the source `result_detection.json` SHA-256 exactly.
All six parity replays and the exact telemetry population count must pass before
any estimator/geometry training pilot is designed as `implemented`.

## Deferred matched training pilot

No training arm is launched by this protocol. If and only if D, K, and all six
M leaves pass, freeze a new study ID and a new selector. The smallest causal
pilot is:

1. residual-only ST, all geometry/coordinate representation disabled;
2. residual-only PL, otherwise identical;
3. fixed uniform support, learned geometry representation disabled;
4. fixed uniform support, learned geometry representation enabled;
5. ROI/geometry support with PL, geometry representation disabled; and
6. ROI/geometry support with PL, geometry representation enabled.

Dynamic role counts, boundary supervision, stability losses, and a context /
geometry / residual union are absent. They may enter a later study only after
the corresponding single-intervention comparisons pass.

Pilot results are exploratory variance estimates, not confirmatory paper
evidence. Numerical superiority/equivalence margins must be derived
result-blind from independent historical variance or frozen using pilot
variance and then tested on disjoint confirmatory seeds. The review-proposed
`+0.50 pp` / `+0.30 pp` gates are not adopted.

## STOP / GO

- `STOP_DECODE`: any Phase D failure. Fix or replace the data backend under a
  new source and rerun D; do not resume an old experiment cell.
- `STOP_ESTIMATOR`: any Phase K likelihood, sign, or gradient failure. Do not
  train PL.
- `STOP_INSTRUMENTATION`: any Phase M hash or population mismatch. Telemetry is
  not observationally neutral and cannot be used.
- `GO_PILOT_DESIGN_ONLY`: D, K, and all six M leaves pass. This authorizes
  freezing the six-arm exploratory pilot, not launching full CER.

The sealed remote result is `GO_PILOT_DESIGN_ONLY`; see
`docs/methods/2026-07-29-georoute-estimator-preexperiment-results.md`. The
independent pilot is frozen in
`docs/methods/2026-07-29-georoute-estimator-pilot.md`.

## Claim boundary

The current implementation is source-native pre-backbone token routing. It is
not Geometry Zoom because it performs no continuous source-coordinate crop or
increased source-density resampling before the heavy backbone. PL is a standard
estimator and is not a novelty claim. Until matched multi-seed accuracy,
decode-to-NMS latency, memory, and energy form a Pareto improvement, no
efficiency or paper-method claim is allowed.
