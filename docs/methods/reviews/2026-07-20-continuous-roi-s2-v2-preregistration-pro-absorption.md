# Continuous-RoI S2 Crop-Sufficiency Preregistration v2: Absorption

## Source Identity

- Reviewed repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Reviewed code/research-state commit:
  `6118cd50a3601d044dab690427ad9c756ce7d827`
- Raw review:
  `2026-07-20-continuous-roi-s2-v2-preregistration-pro-raw.txt`
- Raw line count: `2457`
- Raw SHA-256:
  `9adbd388ad41f79e9323612c25be493332127b226eb2aa968832d14c5446582b`
- Reviewer verdict: `V2_READY`
- Project verdict:
  `ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`

## Independent Checks

The response is not rejected for arithmetic or visibility errors:

- the repository and immutable commit were actually visible;
- the stated protocol-core canonical JSON reproduces
  `3aac17439277fa213f02c5c61f1c2429d6c631abe76e4ce163419ae088b62653`;
- the proposed ROI head, fusion, and three auxiliary heads reproduce
  `277125 + 594049 + 23100 = 894274` trainable parameters;
- the source-area/aspect decoder analytically keeps boxes in bounds over the
  registered range; and
- the checkpoint, manifest, fit/gate/test, and pretrained hashes agree with
  the existing S1 contract.

These checks establish internal arithmetic consistency. They do not establish
scientific validity, implementation feasibility, or permission to queue the
matrix.

## Bottom Line

We agree with the response's research object and most of its implementation
discipline, but we do not accept `V2_READY_FOR_IMPLEMENTATION`.

The document correctly replaces fixed resolution and fixed 21-box selection
with continuous `(cx,cy,w,h)` tubes, preserves the 768-point temporal axis,
uses one shared VideoMAE parameter instance with two real evaluations, keeps
the original AdaTAD-derived ActionFormer projection/head/NMS/evaluator path,
and separates raw no-GT inference from privileged temporal-GT analysis.

However, the proposed experiment does not isolate the claims it says it will
test. It trains a learned ROI policy inside S2, derives fixed controls by
overriding that adaptive model only at inference, and then compares a
per-window GT-privileged continuous selection against controls that do not
receive matched training or matched privileged selection. This can create
apparent "continuous headroom" from distribution shift and information
privilege rather than variable width/height. The no-GT search adequacy rule
also certifies convergence of a confidence-maximizing objective, not coverage
of useful spatial representations.

There are additional direct deployment contradictions with the audited
cluster: the proposed 40 ms power-trace maximum is stricter than the validated
100 ms infrastructure and would reject known-valid traces; one-GPU 128 GB jobs
do not match the N16R4 memory-allocation contract; the 512 GiB free-space gate
is not currently satisfiable; and deleting a contaminated namespace violates
the project's immutable-failure policy.

S2 therefore remains `designed`. No Continuous-RoI implementation, CUDA gate,
training job, reference sweep, official-test opening, or paper claim is
authorized until a v2.1 corrigendum closes the P0/P1 items below.

## Accepted Core

The following should survive v2.1:

- offline TAD semantics and the complete 768-frame time axis;
- continuous normalized source boxes with variable center, area, width,
  height, and source-pixel aspect ratio;
- an analytic in-bounds decoder without post-hoc box repair;
- fixed local output size as a batching/compute contract, not a fixed source
  window or native-density claim;
- source-coordinate crop before local resize;
- a temporally coherent ROI tube rather than per-frame jitter or one box for
  the whole video;
- differentiable training crop and an explicitly matched CPU runtime crop;
- one shared VideoMAE parameter instance while charging both view evaluations;
- an auditable original ActionFormer projection/head/NMS/evaluator backend;
- detector-gradient connectivity, exact optimizer coverage, AMP transaction
  replay, successful-update schedules, and final-EMA discipline;
- fit/gate separation, official-test sealing, raw no-GT generation, immutable
  sealing, and post-seal privileged analysis;
- the 21-box library only as D0;
- separate estimands for geometry, search/reachability, representation
  sufficiency, adaptive headroom, and cost viability;
- separate video/seed detection inference and ABBA window cost inference;
- full-stack latency, memory, energy, MAC/FLOP, training, and search-cost
  accounting; and
- the explicit statement that finite search is not an oracle or global upper
  bound.

The exact `96+128 versus 160`, 12-knot, area/aspect bounds, loss weights, and
4,800-update schedule are acceptable as preregistered hypotheses only after
the corrected static/real-loader feasibility gates pass. They are not
empirically validated design choices.

## Blocking Revisions

### P0-1: S2 and S3 are conflated

The roadmap defines S2 as crop-representation sufficiency before a deployable
learned policy, but the response trains the ROI head jointly from update 1
(`raw.txt:828-832`) and includes its cost in the prospective deployment path
(`raw.txt:1615-1633`). The final matrix never reports the learned policy's
standalone predicted-box row; it only uses that policy as training machinery
and a search initialization (`raw.txt:1053-1065`, `raw.txt:1314-1326`).

Required repair:

- either restore a true S2 representation gate whose continuous geometry
  distribution is registered independently of a deployable learned policy;
- or explicitly merge S2/S3, rename the stage, add the learned predicted-box
  row, and revise the paper roadmap and all authorization rules.

The project chooses the first option: keep S2 and S3 separate. S2 may train a
geometry-conditioned representation under a registered continuous crop
distribution, but it must not claim or silently embed the final learned
deployment policy.

### P0-2: Fixed and location-only controls are not training matched

The response trains only D160, G, and one adaptive GL model, then derives
`C/R/D0/LC/CR` by changing inference geometry on the same GL checkpoint
(`raw.txt:1302-1326`, `raw.txt:2388-2413`). A model trained under adaptive
variable boxes is not a fair fixed-center, random, discrete, or fixed-size
location model. Performance differences can be caused by train/inference crop
distribution mismatch.

Required repair:

- every decision-critical geometry family must be trained under its own
  registered geometry distribution, or all compared families must share a
  common geometry-conditioned training distribution that makes inference
  interventions valid;
- the allowed-differences table must isolate exactly one factor for each
  headroom contrast; and
- fixed-center sufficiency must use a model actually trained with fixed-center
  crops, not a post-hoc override of the adaptive checkpoint.

### P0-3: Continuous headroom comparisons have unequal GT privilege

`CR-PREF` uses gate temporal GT to choose a restart per window after sealing
(`raw.txt:1231-1256`), but the headroom conjunction compares it with
unprivileged `C`, `R`, and `LC` (`raw.txt:1564-1576`). Only `D0-PREF` receives
comparable privileged selection. The resulting gain cannot be attributed to
continuous width/height.

Required repair:

- add a fixed-size continuous-center reference with the same initialization
  count, optimization budget, raw seal, and privileged join as the
  variable-size reference;
- compare variable-size versus fixed-size under the same trained
  representation and the same privilege level;
- keep learned no-GT policy comparisons in a separate deployability family;
  and
- never mix privileged reference rows with deployable rows in one mechanism
  contrast.

### P0-4: Search convergence is not spatial-reference adequacy

The registered no-GT objective maximizes detector confidence, temporal
coverage of confidence, and branch agreement (`raw.txt:1067-1129`). It can
prefer confident false positives, action interiors, background shortcuts, or
calibration artifacts. The `Q` rule then checks finite gradients and
convergence of that same objective (`raw.txt:1278-1296`), not coverage of
spatial representations useful for TAD.

Required repair:

- label confidence optimization as a no-GT policy diagnostic only;
- define reference adequacy using result-independent geometry coverage and
  matched fixed-size/variable-size candidate populations;
- report restart diversity and spatial support before objective convergence;
- specify the exact proposal population, score threshold/top-k, window-GT
  truncation, and matching rules used by the privileged join; and
- allow a reference pass to establish sufficiency, but continue to treat a
  failure as inconclusive.

### P0-5: The formal queue contract is not executable on the audited cluster

Known infrastructure facts conflict with the proposed queue:

- the response requires one-GPU jobs with 96-128 GB memory
  (`raw.txt:2297-2313`), while the audited N16R4 route needs a site-compliant
  outer allocation and an exact one-GPU inner step to obtain that memory;
- it requires `>=512 GiB` free shared storage (`raw.txt:2317-2324`) while the
  latest audited state had roughly 30 GiB free;
- it rejects any NVML gap over 40 ms (`raw.txt:1658-1669`), whereas the
  validated S1 persistent sampler contract used a 100 ms maximum and observed
  valid maxima above 40 ms; and
- on pre-seal GT exposure it orders deletion of the namespace
  (`raw.txt:2375-2384`), contradicting the immutable failed-campaign rule.

Required repair:

- use the cluster's audited outer-allocation/inner-step memory contract;
- estimate and cap raw artifact volume before freezing a storage floor;
- retain the validated 20 ms target/100 ms fail-closed maximum unless a
  no-result pilot justifies a stricter bound;
- preserve every failed/contaminated namespace as immutable evidence and
  create a new namespace; never delete or reuse it.

## Important P1 Revisions

### P1-1: Selector reserve currently double-counts an ROI head

The cost ledger already charges an ROI policy head
(`raw.txt:1615-1633`), then adds a future-selector reserve
(`raw.txt:1681-1694`). This follows from the S2/S3 conflation. v2.1 must either
measure the actual policy once or reserve an absent future policy, never both.

### P1-2: ABBA invocation arithmetic is ambiguous

One `A-B-B-A` block contains two A and two B invocations. The response states
`129x4x3=1548` measured invocations "per cell"
(`raw.txt:1635-1656`), which is either the combined A+B total or a twofold
overcount for each arm. The profiler population and resampling unit must be
rewritten unambiguously.

### P1-3: The power audit uses in-sample fit residuals

The models are trained on fit160, and the power audit reads fit predictions
and fit GT (`raw.txt:1470-1490`). This can understate deployment variance.
Use out-of-fold fit predictions, a predeclared inner calibration split, or
historical independent variance evidence. With only three seeds, do not
present a resampled seed distribution as strong population-level inference;
retain raw per-seed results and a worst-seed guardrail.

### P1-4: The state machine conflates "no headroom" with "bad reference"

A continuous representation can be sufficient while adaptive headroom is
absent. If `S_CR=true`, `H=false`, and `F=false`, the current table falls into
`CONTINUOUS_REFERENCE_INSUFFICIENT` (`raw.txt:1713-1736`). v2.1 needs a
distinct `SUFFICIENT_CONTINUOUS_NO_ADAPTIVE_HEADROOM` outcome.

### P1-5: Privileged matching is under-specified

The lexicographic join defines matched counts and false-evidence mass but does
not freeze the candidate proposal threshold, top-k population, duplicate
handling before selection, or exact treatment of actions crossing overlapping
windows (`raw.txt:1231-1256`). These choices can change the selected restart
and must be part of the protocol core hash.

### P1-6: Hard-coded geometry/schedule numbers are design hypotheses

The area range `[0.18,0.36]`, aspect range `[0.75,2.25]`, 12 knots, 16 search
steps, loss weights, and 4,800 updates are internally coherent, but the
response does not supply empirical or analytical evidence that they provide
adequate TAD spatial/temporal support. They may be frozen only after
result-blind geometry, memory, gradient, and runtime feasibility checks; a
failed feasibility gate must revise the protocol before formal training, not
silently alter values inside a run.

## Correct Status And Unique Next Step

- S1 infrastructure remains `tested`.
- Continuous-RoI S2 remains `designed`.
- The Pro response is archived and absorbed, but its `V2_READY` verdict is not
  adopted.
- No implementation, formal training, reference search, Slurm matrix,
  official-test access, learned-policy authorization, or paper claim follows
  from this response.
- Unique next step: issue a narrow v2.1 corrigendum that repairs the stage
  boundary, matched training arms, matched privilege/reference comparisons,
  search adequacy, cluster/storage/power contracts, and state machine. Only
  after a static protocol validator and result-blind feasibility precheck pass
  may implementation begin.
