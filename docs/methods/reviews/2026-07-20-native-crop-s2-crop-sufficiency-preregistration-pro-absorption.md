# Native-Crop S2 Crop-Sufficiency Preregistration v1: Absorption

## Source Identity

- Reviewed repository:
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- Reviewed commit:
  `f9eca5ab81ee0429469789cfa697e7851dce4bd4`
- Raw review:
  `2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-raw.txt`
- Raw SHA-256:
  `e14abfab41fafa3c3f411df87d3148170872a190c274ed9b7eb2dd44c520c7d5`
- Project verdict:
  `ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`

## Bottom Line

We do not fully accept v1 as an immediately executable preregistration.
It is a serious and unusually complete protocol draft, and most of its
implementation, provenance, no-leak, evaluator, and cost-accounting structure
should be retained. However, the current mechanical decision rule conflates
three different questions:

1. whether a finite native-crop representation is sufficient for TAD;
2. whether adaptive crop selection has measurable headroom over fixed crops;
3. whether a future deployable selector can preserve a real cost advantage.

The draft also treats one GT-visible lexicographic assignment rule as if its
failure could reject the entire registered library, although the draft itself
correctly states that this rule is not a global-mAP oracle. These are decision-
semantic defects, not wording details. S2 remains `designed`; implementation,
formal training, sealed evaluation, and Slurm queueing are not authorized until
a corrected v1.1 protocol is frozen.

## Accepted Core

The following parts are accepted and should survive v1.1:

- S2 is a crop-sufficiency experiment, not a learned crop-policy experiment.
- Crops are taken in source coordinates before full-frame resize while the
  complete 768-point temporal axis is retained.
- The source-letterbox `D160` baseline must be implemented in the same S2
  runtime rather than reusing the historical dense-resize R0 result.
- The registered object uses one shared VideoMAE-S parameter instance, two
  actual view computations, a fixed detector feature contract, and the
  AdaTAD-derived ActionFormer path already closed by S1.
- The 21 static `128x128` candidates have explicit coordinates, no-padding
  checks, a source-area union certificate, and a same-size crop-IoU geometry
  certificate.
- Fit/gate identities, official test sealing, final-only checkpoint policy,
  optimizer coverage, exact update exposure, deterministic schedules, and
  immutable receipts are correctly treated as protocol invariants.
- Raw gate predictions must be generated without GT, teacher, target cache, or
  reference IDs and sealed before detached utility analysis.
- Official evaluator parity must retain finite zero-length proposals as
  zero-IoU false positives.
- Full-stack latency, peak memory, and gross GPU energy must be measured;
  exhaustive reference-search cost must be disclosed separately and cannot be
  presented as deployable cost.
- A finite library cannot establish a claim about all continuous crops, and S2
  cannot claim that a learned policy is trainable or successful.

## Blocking Revisions

### P0-1: The GT-visible rule is not a library upper bound

The draft defines `NC2-GL-LIB-REF` as a per-window GT-visible lexicographic
rule (`raw.txt:361-566`) and explicitly says it is not a global-mAP oracle
(`raw.txt:92`). Nevertheless, failure of its registered metrics falls through
to `KILL_THIS_LIBRARY` (`raw.txt:1278-1286`).

This implication is invalid. The selected per-window assignment need not
maximize final video-level official mAP after overlapping-window aggregation,
NMS, class competition, and score ordering. Another assignment from the same
21 candidates could outperform it.

Required v1.1 repair:

- rename it `GT_VISIBLE_HEURISTIC_REFERENCE`;
- treat a pass as sufficient evidence for representational sufficiency;
- do not treat its failure as proof that the 21-candidate library is
  insufficient; and
- use `REFERENCE_RULE_INSUFFICIENT` unless a genuine certified optimization
  upper bound or valid relaxation is implemented.

`KILL_THIS_LIBRARY` is not authorized by the current reference.

### P0-2: Crop sufficiency and learned-policy headroom need separate outcomes

The current 14-endpoint conjunction makes insufficient headroom over
`BESTFIXED`, center, or random lead to `KILL_THIS_LIBRARY`
(`raw.txt:1167-1286`). If a fixed crop already matches D160, however, the crop
representation may be sufficient even though an adaptive selector is
unnecessary. That is a positive crop result and a negative learned-policy
motivation, not a failed library.

Required v1.1 decision axes:

- `crop_sufficiency`;
- `adaptive_selection_headroom`;
- `representation_cost_viability`.

Required terminal interpretations:

- `SUFFICIENT_AND_POLICY_HEADROOM`: crop sufficiency passes and adaptive
  selection plus reserved selector-cost headroom are supported;
- `SUFFICIENT_FIXED_CROP_ONLY`: a fixed crop is sufficient, but adaptive
  selection is not justified;
- `REFERENCE_RULE_INSUFFICIENT`: the registered heuristic fails without
  rejecting the library;
- `INCONCLUSIVE_GEOMETRY_OR_SUPPORT`: a genuine geometry or subgroup-support
  precondition fails.

Only the first state may authorize a learned-policy preregistration.

### P0/P1-3: Gate GT cache ordering is internally inconsistent

The permission table allows a sealed gate target cache during candidate/cache
generation (`raw.txt:177-181`), while the reference section requires raw gate
outputs to be hashed and closed before gate targets are read
(`raw.txt:493-566`). Creating gate targets in the shared pre-training cache
namespace leaves an avoidable leakage surface even if the model API later
claims not to consume them.

Required v1.1 DAG:

```text
geometry/candidate cache without GT
-> training
-> raw gate sweep without GT/cache/reference IDs
-> immutable raw-prediction receipt
-> privileged gate-GT join and detached reference analysis
```

Fit GT may remain available to training. Gate GT artifacts must not exist in
the model/inference namespace before the raw receipt is sealed.

### P1-4: Detection and cost cannot share one bootstrap family unchanged

The draft puts detection endpoints and latency/energy endpoints in one
14-score max-T family (`raw.txt:917-959`, `raw.txt:1241-1256`). Detection
uncertainty is paired by gate video and seed. Cost uncertainty is paired by
ABBA block/window, execution order, hardware state, and seed. Video-cluster
weights are not the sampling unit for the cost measurements.

Required v1.1 repair:

- detection family: paired video-cluster resampling plus seed hierarchy;
- cost family: paired ABBA block/window resampling plus seed hierarchy and
  frozen order/drift accounting;
- simultaneous correction within each family; and
- final authorization by intersection of the corrected detection and cost
  families.

Before numerical margins are frozen, perform a result-blind power and
Monte-Carlo precision audit using synthetic data or historical variance only.
Do not tune margins after observing S2 outcomes.

### P1-5: Geometry coverage and model-conditioned reachability are different

The 21-grid geometry certificate is a property of source coordinates.
`CandidateUnionRecall@0.7`, however, depends on the trained GL-LIB detector,
proposal scores, and fusion path (`raw.txt:333-347`). A low value can be caused
by optimization or representation failure even when geometric candidate
coverage is complete.

Required v1.1 repair:

- keep `geometric_library_coverage` as a deterministic certificate;
- rename union recall to
  `model_conditioned_candidate_union_reachability`; and
- reserve `INCONCLUSIVE_GEOMETRY_OR_SUPPORT` for geometry or subgroup-support
  failure, not ordinary detector underperformance.

### P1-6: The estimand must disclose the training-distribution confound

GL-LIB receives epoch-varying crop positions while D160 is a deterministic
letterbox. Turning off other spatial augmentation does not make these
representations training-distribution matched. The experiment therefore
estimates the registered representation plus its registered crop schedule, not
pure crop information alone.

Required v1.1 repair: state this estimand explicitly. A full-frame matched-
jitter control may be preregistered if the project wants a pure information
claim, but it must not be added after observing results.

### P1-7: Reserve cost for the future selector

`NC2-GL-LIB-R` is policy-shaped but selector-free. It can establish
representation-path cost headroom, not the cost of a deployable learned
selector. v1.1 must either reserve an explicit selector latency/energy budget
inside the GO margin or limit the conclusion to representation-path headroom.

## Nonblocking Improvements

- P2: Stage the 21 training cells so the decision-critical D160/GL-C/GL-LIB
  block closes before descriptive D224/D256 and single-view ceilings consume
  resources. The frozen decision rule must not change between stages.
- P2: CUDA/AMP cached-global parity should be bitwise only where deterministic
  kernels prove that contract; otherwise use a frozen tolerance plus tensor
  hashes and deterministic execution trace.
- P2: Preserve the proposed numerical margins only after the result-blind
  feasibility/power audit shows that 40 gate videos, three seeds, and the
  simultaneous endpoints can produce a meaningful GO decision.

## Status And Unique Next Step

- S1 implementation status remains `tested`.
- S2 protocol status is `designed`,
  `pro_protocol_reviewed_revision_required`.
- No S2 model implementation, training job, sealed-test opening, profile
  campaign, learned crop policy, or paper claim is authorized.
- Unique next step: revise the archived v1 into Native-Crop S2
  Preregistration v1.1, close the P0/P1 items above, run a result-blind
  statistical feasibility audit, and freeze the corrected decision semantics
  before writing or queueing the formal experiment.
