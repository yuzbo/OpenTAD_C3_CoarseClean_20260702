# DUCA Admission v2.1 Corrigendum: Independent Audit and Final Repair Contract

Date: `2026-07-31`

Status: `DESIGNED / PROTOCOL_IMPLEMENTATION_BLOCKED_PENDING_ONE_NARROW_CORRECTION`

Repository:
`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git`

Required branch: `codex/duca-rime-20260727`

Reviewed repository commit:
`d3e9814afd16739dadc273f181deb9a065c151d4`

Reviewed repository tree:
`c2f74b963bc6291b96a3d18133fb90c9eb3e3901`

Reviewed response:
`U-PRO-V21-CORRIGENDUM-1`

Response SHA-256:
`12a324b2eb43086397a0e54d5c64dae84c86fec96cef03ab0c8decc095cc7f37`

Response size: `67,868 bytes`

## 1. Executive adjudication

The response is **not accepted verbatim**.

The correct decision is:

```text
core_direction = ACCEPT
protocol_implementation_as_written = NO_GO
next_action = ONE_NARROW_STATISTICAL_AND_SCHEMA_CORRECTION
production_admission_v2_1 = NO_GO
holdout_opening = UNAUTHORIZED
phase1_or_later_training = UNAUTHORIZED
learned_hrime = UNAUTHORIZED
full_200_refit = UNAUTHORIZED
official_final = SEALED
paper_admissible_empirical_result = NONE
```

The response substantially improves the previous proposal. It correctly:

1. replaces sparse multinomial draws with positive factor multipliers;
2. preserves the fixed equal-video estimating equation and removes
   replicate-specific denominators;
3. names the method `single_step_fixed_scale_standardized_maxT`;
4. replaces binary stream agreement with a numerical Monte Carlo error
   contract and deterministic 100k-to-200k extension;
5. separates structural catastrophes from numeric-tail inference;
6. narrows the runtime threat model and separates enforced, attested,
   observed and unavailable controls;
7. keeps Admission, Phase 1, H-RIME, full-200 refit and official-final
   unauthorized.

Those decisions remain frozen. They do not, however, resolve the five P0
problems below.

## 2. P0-1: manifest order and permutation invariance contradict each other

The response assigns process slots from manifest rank:

```text
p1(i) = i mod 8
p2(i) = (i + floor(i / 8) + 1) mod 8
```

It also requires a test asserting that reordering manifest rows does not change
the result when video IDs are preserved. Both statements cannot be true unless
the implementation first derives a canonical, candidate-independent rank from
immutable video identity.

### Required correction

Freeze one exact rule:

```text
canonical_video_rank =
    rank of video_id under a fully specified, domain-separated hash ordering
    bound to protocol_id + source_split_sha256 + role_id + video_id
```

The correction must specify:

- byte encoding and separator rules;
- Unicode policy;
- exact hash function;
- tie handling;
- whether the role identifier enters the domain;
- whether incidence hashes encode canonical rank or input row rank;
- process-label permutation semantics.

Required invariant:

```text
permuting input JSON rows while preserving the same immutable video records
must produce byte-identical role membership, canonical ranks, incidence,
incidence SHA-256, multiplier streams and final statistics.
```

## 3. P0-2: the calibration/holdout triplet changes the statistical unit

The response defines 22 long triplets and 10 short triplets, with one
calibration and one holdout video per triplet. The third video is only implied
to be `scale_fit`. The pseudocode then creates:

```text
Cell(video_id=block_id, ...)
```

and indexes calibration and holdout scores by the same `block_id`. This silently
replaces two different videos with one pseudo-video/block. It conflicts with
the explicit statement that the statistical unit is video.

The following are also unspecified:

- how the four unused long videos are selected;
- how long and short videos are canonically grouped into triplets;
- how the three videos in each triplet are assigned to
  `scale_fit`, `calibration` and `admission_holdout`;
- the exact seeded permutation and tie-breaking bytes;
- whether physical process effects are paired across roles;
- which exchangeability/randomization assumption licenses subtracting two
  different videos inside one cell.

### Required correction

The next adjudication must select exactly one of these estimands.

#### Option A — block-randomized paired contrast

Use an explicit `pair_id` or `triplet_id`, never a fake `video_id`. The response
must admit that the resampling block for the tail contrast is the preregistered
triplet/pair. It must define a design-based or model-based justification for
the paired contrast and keep physical process identities role-specific.

Required record:

```text
pair_id
calibration_video_id
holdout_video_id
scale_fit_video_id
length_stratum
calibration_process_id
holdout_process_id
assignment_hash
```

#### Option B — independent role contrast

Keep calibration and holdout as two disjoint 32-video samples. Estimate the
difference of role means/tail summaries without manufacturing video pairs.
Length-stratified triplets may be used for allocation only. The bootstrap or
randomization procedure must independently represent the video and role-specific
process factors.

### Current recommendation

Prefer Option B unless a complete block-randomization derivation shows why the
paired estimator targets the intended population quantity and how process
effects enter its covariance. It is simpler, preserves the declared video
unit, and avoids treating a logical process label as a shared physical process.

## 4. P0-3: the finite-level correction is not derived and worsens known
conservatism

The response multiplies every perturbation by:

```text
kappa_VP = sqrt(32 / 31) * sqrt(8 / 7)
kappa_VP^2 = 256 / 217
```

Owen and Eckles support independent factor reweighting with product observation
weights and explicitly state that no exact bootstrap exists for the general
crossed problem. Their result describes the product method as generally mildly
conservative under its assumptions. It does not derive the response's product
of two Bessel-style factors.

An exact covariance calculation on the frozen 32-by-8 incidence gives the
following expected multiplier-variance gain relative to the true variance of
the equal-video estimator:

| Random-effects component | Product multiplier, `kappa=1` | With proposed `kappa_VP` |
|---|---:|---:|
| video/row only | `117/64 = 1.828125` | `468/217 = 2.156682...` |
| process only | `69/64 = 1.078125` | `276/217 = 1.271889...` |
| observed-cell interaction only | `181/64 = 2.828125` | `724/217 = 3.336406...` |

For two observed cells `e=(i,p)` and `f=(j,q)`, let
`M_e=A_i B_p-1`. The registered multiplier moments imply:

```text
Cov(M_e, M_f) =
    3, e=f
    1, i=j and p!=q
    1, i!=j and p=q
    0, otherwise.
```

Let `Omega` be this 64-by-64 covariance matrix, let scalar `a=1/64`,
let `a_vec=a*1`, and let `H=I-1 a_vec^T` be the residual-centering operator.
For random-effects covariance `Sigma`, the expected uncorrected multiplier
variance and true estimator variance are:

```text
E_X Var_G(Z | X) = a^2 trace(Omega H Sigma H^T)
Var_X(delta_hat) = a_vec^T Sigma a_vec.
```

Evaluating these expressions for the exact frozen incidence and, respectively,
the row, process and observed-cell identity covariance matrices produces the
fractions above.

Thus the proposed correction inflates bootstrap variance by another
`256/217 = 1.179723...`; it does not calibrate any of the three registered
variance components to one.

This does not prove that the product bootstrap must be discarded. It proves
that `kappa_VP` cannot be described as a theoretically established finite-level
correction.

### Required correction

The next adjudication must choose exactly one:

1. set `kappa=1`, align with the established product-weight construction, and
   certify coverage plus non-vacuity on the frozen incidence;
2. derive a design-specific correction from the exact covariance/gain
   coefficients and validate it without candidate data;
3. replace the product perturbation with another completely specified
   two-way-cluster procedure and provide its small-sample contract.

No scalar may be selected only because it improves coverage. A method that
always produces very wide upper bounds can pass a coverage-only gate while
having no useful admission power.

## 5. P0-4: the simulation contract is not reproducible and checks only
coverage

The proposed simulation names nine high-level scenarios and outer/inner sample
counts, but does not completely specify:

- the distributions and parameters of video, process and interaction effects;
- the exact count-linked heteroskedastic law;
- skew-lognormal centering/scaling;
- the complete metric family and planned family size;
- cross-metric joint generation;
- tiny-scale and leverage boundaries;
- outer versus inner random-stream domains;
- whether `q`, `s`, `U` and the Monte Carlo certificate are recomputed for every
  outer data set;
- the exact number of registered scenario combinations.

Coverage alone is insufficient. An arbitrarily conservative method can satisfy
the lower coverage bound.

The delete-one-1000-replicate-batch jackknife is also tested only against hand
fixtures. Its claimed 99% half-width must be calibrated for the nonsmooth
type-1 order statistic, maxT critical value and joint upper bounds.

### Required correction

Create an immutable simulation registry that binds:

```text
scenario_id
effect distributions and all parameters
family metric IDs and exact C
cross-metric covariance construction
count/leverage law
outer seed domain
inner multiplier seed domain
exact recomputation algorithm
coverage endpoint
width/non-vacuity endpoint
power/sensitivity endpoint
MC-half-width calibration endpoint
acceptance thresholds
```

The gate must report at least:

1. simultaneous coverage;
2. normalized bound width or variance inflation;
3. power/sensitivity under preregistered finite shifts;
4. false numeric-tail alarm under the null;
5. MC-half-width calibration against independent repeated streams.

Claims must be limited to:

```text
scenario-conditional empirical calibration on the registered 32x8 grid
```

not general finite-sample validity.

## 6. P0-5: runtime receipt semantics are not closed

The response's threat model is directionally acceptable, but the receipt schema
still needs:

- exact enums for `status`, `classification` and `claim_state`;
- closed-world, unique control IDs;
- planned-cell manifest and hash;
- writer path and fresh output root;
- parent receipt bindings;
- status-to-authorization invariants;
- exact Slurm job/step, PID start time, cgroup, host and CUDA allocation
  attestations;
- a same-filesystem atomic-publication self-test;
- canonical allowlisted roots and rejection of pre-existing symlink parents.

The current evidence writer resolves a path before publication and creates its
parent. That is useful atomic publication behavior but is not, by itself, a
root allowlist or symlink-escape proof.

The local audit has now resolved the previously unavailable Git tree:

```text
c2f74b963bc6291b96a3d18133fb90c9eb3e3901
```

This removes the report connector's identity uncertainty. It does not create a
runtime isolation receipt or authorize execution.

## 7. P1 editorial corrections

The final correction should also repair:

1. the estimand formula where a comma appears between `I_ip` and
   `E[X_ipc]` instead of multiplication;
2. malformed set braces in the incidence definition;
3. the placeholder status value `passed_or_failed_closed`;
4. all uses of `block_id` where the actual object is a pair/triplet;
5. the exact type-1 order-statistic behavior when `ceil((B+1)(1-alpha)) > B`;
6. fail-closed behavior when `r_c^MC` is zero or nonfinite.

## 8. Repository implementation status

No v2.1 production protocol module has been implemented by this audit.

Current code remains correctly fail-closed:

- old Admission v2 cannot authorize Phase 1;
- the current v2.1 auditor still records the obsolete per-video
  full-plus-short infeasibility and authorizes nothing;
- current H-RIME finalization does not implement the proposed future maxT/tail
  contract;
- Phase 4 and official-final remain sealed.

The existing v2.1 test fixture also represents 70 long and 26 short videos,
not the immutable 70/30 source inventory. It must be replaced only after the
role generator contract is corrected.

## 9. Post-correction implementation order

Only after the next response resolves all five P0s:

### Stage A — deterministic primitives

1. implement canonical role/triplet/reserve generation;
2. implement canonical 32-by-8 incidence;
3. implement closed schemas and validators;
4. keep every producer authorization field false.

### Stage B — pure statistics

1. implement the selected equal-video/role contrast;
2. implement the selected crossed perturbation with no unapproved scalar;
3. implement fixed-scale standardized single-step maxT;
4. implement exact structural failures and the corrected numeric-tail
   procedure;
5. implement deterministic stream extension and MC diagnostics.

### Stage C — candidate-free certification

1. run deterministic unit and property tests;
2. run the complete immutable simulation registry;
3. validate coverage, non-vacuity, power and MC-error calibration;
4. fail closed on any registered scenario.

### Stage D — exact-clean runtime gate

1. collect exact commit/tree/dirty-state identities;
2. run Linux/PyTorch and DDP focused gates;
3. run truthful Slurm/PID/cgroup/GPU/filesystem attestations;
4. publish content-bound, exclusive receipts from a fresh root.

### Stage E — production Admission

Production workers, scale-fit, calibration and disjoint holdout remain a
separate authorization decision. Passing code and simulation gates is necessary
but not sufficient to open the holdout.

## 10. Ready-to-send final Pro prompt

```text
You are the final statistical protocol adjudicator for a CVPR-level offline
Temporal Action Detection project. Act as an unusually rigorous first-author
PhD student, statistical reviewer, reproducibility auditor, and code reviewer.
Do not brainstorm a new model. Resolve one narrow Admission-v2.1 protocol
contract so that deterministic candidate-free protocol code can be implemented.

AUTHORITATIVE REPOSITORY ADDRESS
https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git

REQUIRED BRANCH
codex/duca-rime-20260727

INSTRUCTIONS FOR REPOSITORY IDENTITY
1. Fetch the repository from the address above.
2. Resolve and report the live branch HEAD, HEAD^{tree}, remote URL, dirty
   status, and path diff from commit
   d3e9814afd16739dadc273f181deb9a065c151d4.
3. Read the latest versions of:
   - research-wiki/query_pack.md
   - research-wiki/anti_repetition.md
   - research-wiki/source_registry.md
   - docs/superpowers/plans/2026-07-29-duca-admission-v2-1-repair-plan.md
   - docs/superpowers/plans/2026-07-30-duca-v2-1-pro-response-adjudication.md
   - docs/superpowers/plans/2026-07-31-duca-v2-1-corrigendum-independent-audit.md
   - tools/bata/audit_duca_acquisition_v2_1_feasibility.py
   - tools/bata/duca_evidence_io.py
   - tools/bata/hrime_stage1_oracle.py
   - all focused Admission-v2/v2.1 tests
4. Do not rely on pasted code or an older snapshot when the repository differs.

NON-NEGOTIABLE EVIDENCE BOUNDARY
- No candidate output, decoded holdout video, checkpoint result, development
  metric, model-performance number, mAP, official-final artifact, or training
  run may be consumed.
- Production Admission, holdout opening, Phase 1, learned H-RIME, full-200
  refit and exact-211 evaluation remain unauthorized.
- The only allowed output is a corrected deterministic protocol design,
  formulas, schemas, pseudocode, tests and a file-level implementation plan.
- Preserve the offline TAD pure selected-axis pre-backbone plugin as the paper
  mainline. This is not an architecture discussion.

ACCEPTED DECISIONS THAT MUST NOT BE REOPENED
1. Three disjoint roles of 32 videos, each containing 22 natural-full and
   10 natural-short videos; four remaining long videos are metadata-only
   reserves.
2. Eight logical process clusters per role, video degree two, process degree
   eight, connected incidence.
3. Equal-video estimating coefficients remain fixed; no replicate-specific
   denominator, video deletion, redraw-until-nonempty, or window-as-sample.
4. Exact method name:
   single_step_fixed_scale_standardized_maxT.
5. Structural catastrophes and numeric-tail inference are separate.
6. Deterministic 100k-to-200k stream extension; binary stream agreement is
   diagnostic only.
7. Explicit trusted-program threat model with enforced/attested/observed/
   unavailable controls.
8. No production or later-stage authorization follows from this discussion.

YOU MUST RESOLVE FIVE P0 QUESTIONS

P0-1 CANONICAL ORDER
The proposed incidence uses manifest rank i, while a mandatory test demands
input-row permutation invariance. Give one exact canonical ranking algorithm:
domain-separated bytes, encoding, hash, role binding, tie handling, incidence
serialization and hash. Prove or test that reordered input rows give byte-
identical roles, ranks, incidence, streams and statistics.

P0-2 ROLE/TRIPLET AND STATISTICAL UNIT
The previous pseudocode used Cell(video_id=block_id) for a difference between
two distinct calibration/holdout videos. This is invalidly ambiguous. Select
exactly one:
A. an explicit block-randomized paired estimator with pair/triplet as the
   resampling block and a complete exchangeability/randomization derivation; or
B. an independent calibration-role versus holdout-role estimator that preserves
   video as the statistical unit and uses role-specific process effects.
Specify reserve selection, triplet formation, assignment of the third member to
scale_fit, role permutation bytes, disjointness, physical process identity,
estimand, estimator, multiplier factors, covariance assumptions and all IDs.
Never store pair_id in a video_id field.

P0-3 CROSSED MULTIPLIER AND FINITE CORRECTION
The prior proposal used
kappa=sqrt(32/31)*sqrt(8/7).
An exact calculation on the frozen incidence gives variance gains:
- row only: 117/64 without kappa; 468/217 with kappa;
- process only: 69/64 without kappa; 276/217 with kappa;
- cell interaction only: 181/64 without kappa; 724/217 with kappa.
Independently verify or refute this derivation. Owen-Eckles supports product
factor weights but does not appear to derive this kappa. Select one exact
procedure: kappa=1, a fully derived design-specific correction, or a different
fully specified two-way-cluster method. Explain coverage and conservatism.
Coverage-only selection is forbidden.

P0-4 COMPLETE SIMULATION AND MC CERTIFICATION
Return an immutable, executable registry of every synthetic scenario, including
all effect distributions/parameters, heteroskedastic law, family metric IDs and
exact C, correlations, count/leverage law, outer/inner seed domains, complete
recomputation algorithm and exact number of scenarios. In addition to
simultaneous coverage, freeze non-vacuity/width and power/sensitivity criteria.
Validate the delete-one-batch jackknife for the type-1 maxT quantile and joint
bounds against independent repeated streams. State only scenario-conditional
empirical calibration, not universal finite-sample validity.

P0-5 CLOSED RUNTIME RECEIPT
Return exact enums and cross-field invariants for status, control
classification, claim state and authorization. Bind the planned-cell manifest,
parent receipts, writer/root, commit/tree/cleanliness, role/incidence/metric/
config/checkpoint/executable hashes, Slurm job and step, PID start time, cgroup,
host, CUDA allocation, filesystem semantics and exclusive publication.
Repository controls, cluster attestations, observations and limitations must
remain distinct. Missing required attestations fail closed; optional unavailable
network/mount strengthening does not become an enforced claim.

REQUIRED OUTPUT
1. Repository identity receipt.
2. GO / CONDITIONAL GO / NO-GO for protocol implementation only.
3. A table of accepted, rejected and corrected statements from the July 31
   independent audit.
4. Exact role/triplet/reserve algorithm and schema.
5. Exact incidence and canonical-order algorithm.
6. Exact estimands and complete crossed/tail formulas.
7. Independent verification of the three variance-gain fractions.
8. Exact multiplier/correction decision with derivation.
9. Fully enumerated simulation registry, RNG contract, coverage, width, power,
   false-alarm and MC-calibration gates.
10. Closed runtime receipt schema and validation invariants.
11. Deterministic pseudocode detailed enough to translate line by line.
12. A file-by-file implementation plan that touches protocol-only modules and
    focused tests; do not modify model architecture, losses, budgets,
    thresholds, splits, checkpoints or official metrics.
13. Mandatory tests, including reordered input, role disjointness, reserve
    handling, pair/video-ID separation, exact hand calculations, degenerate
    scales, MC extension, MC calibration, missing attestations and permanent
    old-v2 rejection.
14. Explicit remaining blockers and the only actions authorized after the
    response.

QUALITY BAR
- Do the mathematics; do not answer with general advice.
- Every symbol, denominator, quantile index, seed byte and failure state must be
  closed.
- Point out any contradiction instead of silently choosing an interpretation.
- Do not claim implementation or tests that are not present in the live repo.
- Do not return model-performance analysis; no paper-admissible empirical result
  exists yet.
```
