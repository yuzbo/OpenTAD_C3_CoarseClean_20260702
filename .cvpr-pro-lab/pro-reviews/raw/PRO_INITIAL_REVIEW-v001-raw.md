# PRO_INITIAL_REVIEW-v001

## 1. Context reconstruction and task boundary

### 1.1 Review identity

The takeover contract identifies ChatGPT Project `g-p-6a796fef9a00819194024cf1de3bd697`, nonce `122c44f96c0c4397a394a6738c0a2259`, project title DUCA, ten Project Sources, and the GPT-5.6 Pro route. I treated those routing fields as contract-supplied context. I independently accessed the ten supplied Project Sources and the pinned GitHub commit; I did not independently inspect the ChatGPT browser UI or Project metadata.

The canonical code identity for this review is:

* Repository: `yuzbo/OpenTAD_C3_CoarseClean_20260702`
* Commit: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`
* Resolved Git tree: `b50b7d986448657ba2debbaac0ef2610fce1bdfb`

The commit object and recursive tree were accessible through the GitHub connector.

### 1.2 Scientific task

The task is **offline Temporal Action Detection**, not Online TAD, causal streaming, video classification, reconstruction, video-language reasoning, or a general adaptive-computation platform.

The falsifiable paper hypothesis is:

> A cheap, deploy-visible full-video scout can jointly allocate a per-video heavy-frame budget and exact monotone physical-time acquisition positions, under a dataset-level average-cost constraint, so that an otherwise controlled offline detector achieves a better accuracy–cost trade-off than strong uniform controls while preserving high-IoU localization, short actions, and boundary accuracy.

The intended processing contract is:

1. A low-cost scout observes only deploy-visible video input.
2. It selects a discrete heavy-frame budget (K_v) and unique monotone physical positions.
3. The heavy backbone processes only those positions.
4. The detector predicts on selected coordinate (q).
5. Segment endpoints are mapped from (q) to physical time (t) **before** unchanged official NMS and evaluation.
6. Full-stack cost includes scout, transport, backbone, detector, memory, latency, and throughput.

These boundaries are explicit in the charter and route decision.  

### 1.3 Evidence and authorization boundary

The project is **pre-result**. No sealed, full-split, multi-seed result matrix exists for this revision. Consequently:

* No DUCA mAP is established.
* No improvement over exact uniform sampling is established.
* No dynamic-budget advantage is established.
* The historical AdaTAD/uniform reference is only a target for controlled reproduction.
* No subset, intermediate, single-seed, failed-root, over-budget, Oracle, synthetic, upstream, or infrastructure metric may be promoted.

RESULTS, RESULT_ANALYSIS, and PAPER_DRAFT are correctly marked `BLOCKED_PRE_RESULT`.   

This review authorizes **no** GPU run, remote run, formal experiment, Git push, result promotion, held-out access, or submission. The plan below defines queues and gates; it does not activate them.

---

## 2. Facts, inferences, and unknowns

### 2.1 Method

**Facts**

* The intended contribution combines dynamic per-video budget allocation, exact-budget monotone physical acquisition, coverage/gap safeguards, and physical-coordinate transport into an AdaTAD/OpenTAD detector. 
* The candidate allocator families are independent-per-(K), strict nested acquisition, and at most one weak-overlap compromise admitted only by preregistered train/utility-only evidence. 
* The first scientific contrast is fixed-(K) learned placement versus exact-uniform placement, with mixed-(K) exposure controlled. Dynamic (K) is downstream of that gate. 
* A generic importance scorer plus top-(K) is explicitly insufficient as the intended novelty. The gap concerns physical coverage, short actions, boundaries, coordinate transport, and actual full-stack computation. 

**Inferences**

* The scientific core is viable only if “joint allocation” is a real optimized method rather than a fixed-slot differentiable reader combined with a disconnected hand-coded budget heuristic.
* Same-(K) learned-versus-uniform comparison isolates placement, but it is not automatically a **full-stack cost-matched** comparison because the learned arm pays scout and transport overhead.
* The most defensible initial method is a narrow selector-only fixed-(K) plugin. Dynamic budgeting should remain unimplemented or quarantined until the fixed-(K) hypothesis survives.

**Unknowns**

* Whether a jointly optimized scout can actually learn useful TAD placement.
* Whether learned placement beats exact uniform after all training exposure and detector differences are controlled.
* Whether high-IoU and short-action localization survive.
* Whether scout overhead leaves a genuine end-to-end saving.
* Whether strict nesting is materially restrictive.
* Whether the idea remains novel after an exhaustive 2026 literature audit.

### 2.2 Implementation

**Facts**

The pinned tree contains the requested surfaces:

* `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`
* `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py`
* `opentad/models/selectors/pc_ot_mras_reader.py`
* `opentad/models/necks/pc_ot_mras_detector_bridge.py`
* `opentad/models/utils/pc_ot_mras_raw_prediction_guard.py`
* the principal PC-OT-MRAS physical-grid config
* an exact-uniform physical-grid control
* a validator
* focused coordinate, candidate, and execution tests.

The implementation status Source accurately describes these as implemented surfaces, while explicitly withholding paper-readiness. 

The inspected dynamic-budget controller:

* accepts a discrete budget set;
* computes utility and summary statistics from deploy-visible inputs;
* applies fixed hand-written score weights and thresholds;
* initially assigns budgets from thresholds;
* then adjusts assignments across the current batch to drive that batch toward a target average;
* constructs endpoint-anchored, sorted positions from utility;
* rejects a broad set of forbidden metadata names.

The full-video reader uses a lightweight temporal network and differentiable monotone slot transport, but exposes a fixed number of slots.

The main pinned configuration is fixed at dense length 768 and selected length 384, uses a frame-score top-(K) policy, and explicitly records that detector-head geometry and loss assignment are changed. It therefore does not instantiate the charter’s clean “selector before an unchanged downstream detector” route.

The exact-uniform control exists and removes reader influence by allocating all 384 positions to its uniform component. It nevertheless inherits the altered physical-grid detector route and labels its changed surface as input sampling plus head temporal geometry.

The bridge supports selected-axis physical metadata and raw-prediction guards, but also contains broad interpolation, context, projection, and conditioning functionality.

The raw-prediction guard recursively rejects a narrow set of raw prediction/cache key tokens.

The focused tests provide meaningful regression protection for:

* selected-coordinate to physical-coordinate interpolation;
* rejection of selected-rank semantics;
* use of physical geometry in target construction;
* retention of official batched NMS;
* fixed-(K) candidate configuration identity;
* exact-uniform control identity;
* a local execution/gradient smoke.

**Inferences**

* The batch-level budget correction makes a video’s chosen (K) depend on the other videos in its inference batch. That is not a stable per-video deployment policy and can become a transductive evaluation mechanism even without labels.
* The controller currently enforces average **nominal frame budget**, not a measured average full-stack cost.
* The reader and controller represent different allocation mechanisms: a fixed-slot differentiable transport and a hard variable-length heuristic. Their joint optimization and paper-route integration are not established.
* The main configuration is better characterized as a fixed-(K), physical-grid detector candidate than as the registered DUCA-RIME joint dynamic-budget method.
* The broad selector/bridge/validator surface is significantly larger than the minimum paper mechanism and creates engineering-drift risk.
* The raw guard is useful but insufficient as a complete provenance policy; forbidden information can arrive under names not containing its small token set.

**Unknowns**

* Whether the dynamic controller is exercised end to end by any paper-intended config at this commit. It is not referenced in the inspected main config, and repository symbol search did not establish an integration path.
* Whether repeated padded positions for short valid sequences ever cause duplicate heavy-frame computation or are always excluded by masks.
* Whether the full training graph propagates useful detector gradients into the exact hard acquisition policy under production settings.
* Whether all official detector-head and loss semantics can be retained while putting coordinate conversion entirely outside the detector.
* Whether current cost metadata corresponds to measured deployment cost or only nominal tensor dimensions.

### 2.3 Experiment

**Facts**

The registered Stage A cells are:

1. dense T768 reference;
2. exact-uniform fixed K384;
3. mixed-(K) training with exact-uniform K384 evaluation;
4. learned fixed K384 positions.

The protocol requires complete official training and validation data, frozen evaluation, terminal checkpoints, multiple registered seeds, and full-stack cost accounting. 

The pinned tree contains the exact-uniform control. I did not identify the complete registered mixed-(K) exposure cell or a paper-ready dynamic-(K) matrix in the inspected tree.

**Inferences**

* The current code can support engineering smokes but not the registered attribution matrix without revision.
* Running only the current learned and exact-uniform physical-grid configs could answer a narrower comparison, but it would not establish the charter’s unchanged-detector claim.
* A fixed-K same-heavy-budget comparison and a full-stack cost-matched comparison must be reported separately.

**Unknowns**

* Registered seed identities and the final uncertainty estimator are not supplied in these Sources.
* No numerical superiority, non-inferiority, cost-matching, or futility margins have been frozen.
* Actual GPU memory, latency, throughput, and training cost at this exact revision are unknown.

### 2.4 Results

**Fact**

No formal result exists. All C1–C6 candidate claims remain unsupported or only partially implemented. 

**Inference**

No conclusion about feasibility, convergence, superiority, paper readiness, or final novelty can be drawn from tests, validators, historical jobs, or implementation completeness.

**Unknown**

All model-quality and accuracy–cost outcomes.

### 2.5 Negative evidence

**Facts to preserve**

* Earlier DUCA variants did not fairly establish superiority over cost-matched uniform sampling.
* Selected-rank decoding was semantically wrong.
* Local-cell selection, actionness top-(K), hard query deletion/SparseHead, DCSR-G1, and ODF-CR-G2 were rejected or negative in their evaluated scopes.
* Launcher, mask, temporal-axis, receipt, and checkpoint failures are engineering evidence, not model-quality evidence. 

**Inference**

The current frame-score top-(K) route bears enough resemblance to previously weak or rejected saliency routes that it cannot be accepted merely because physical-coordinate plumbing is now better.

### 2.6 Principal blockers

1. **Detector-invariance blocker:** the canonical main config changes detector-head geometry and loss assignment.
2. **Dynamic-policy blocker:** the budget controller is heuristic, batch-coupled, and not established as an integrated learned route.
3. **Fairness blocker:** same (K) is not necessarily same full-stack cost once scout and transport are included.
4. **Control blocker:** the complete Stage A exposure-matched matrix is not established in the pinned tree.
5. **Short-window blocker:** a unique-(K_{\text{eff}}), masking, and no-duplicate-heavy-read contract is not frozen.
6. **Cost blocker:** no sealed full-stack cost receipt or target-hardware measurement exists.
7. **Literature blocker:** C6 remains open.
8. **Evidence blocker:** no formal result exists.

---

## 3. Hypothesis-alignment and engineering-drift audit

| Component                    | Registered requirement                                        | Pinned implementation                                                   | Adjudication                                                                             |
| ---------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| Full-video scout             | Cheap, deploy-visible, pre-heavy-backbone                     | A lightweight full-video reader exists                                  | **Partially aligned**; actual end-to-end cost is unknown                                 |
| Per-video dynamic budget     | Independent deploy-time (K_v) under average-cost constraint   | Heuristic budget scores followed by batch-wide correction               | **Material drift**; batch composition affects (K_v)                                      |
| Budget objective             | Average full-stack cost                                       | Target average nominal frame count                                      | **Material drift**                                                                       |
| Position allocation          | Exact, unique, monotone physical positions with safeguards    | Sorted hard selections and endpoint/coverage machinery exist            | **Partially aligned**; short-window uniqueness and heavy-read behavior remain unresolved |
| Joint optimization           | Scout allocates budget and positions for TAD utility          | Fixed-slot differentiable reader and separate hard heuristic controller | **Unestablished**                                                                        |
| Fixed-K Stage A              | Learned K384 versus exact uniform and mixed-exposure controls | Learned and exact-uniform configs exist; full matrix not established    | **Incomplete**                                                                           |
| Unchanged detector           | Official detector/head/loss retained                          | Main config explicitly changes head geometry and loss assignment        | **Direct contradiction**                                                                 |
| Physical coordinate contract | Map (q\rightarrow t) before official NMS                      | Implemented and covered by focused tests                                | **Strongly aligned**                                                                     |
| Leakage prevention           | Deploy-visible inputs only                                    | Controller denylist and raw-prediction guard exist                      | **Partially aligned**; provenance certification remains missing                          |
| Full-stack efficiency        | Include scout, transport, backbone, detector                  | No formal measured cost receipt                                         | **Unestablished**                                                                        |
| Evidence handling            | Full split, terminal, multi-seed, sealed                      | Sources correctly block results                                         | **Aligned at protocol level**                                                            |
| Project shape                | Minimal falsifiable method                                    | Large selector/bridge/validator platform surface                        | **Engineering-drift risk**                                                               |

The most important contradiction is not cosmetic. The charter defines an unchanged downstream detector, while the principal config declares changes to detector-head geometry and loss assignment. 

There are only two scientifically legitimate responses:

1. **Preferred revision:** externalize all coordinate conversion. Map physical training targets (t) to selected coordinate (q) before the official detector, retain the official detector head and loss unchanged, then map predicted (q) endpoints back to (t) before official NMS.
2. **Alternative reframing:** declare the method to include a physical-grid detector adaptation, then rebuild every dense, uniform, mixed-exposure, and learned control with the same adapted head. This is broader, less attributable, and inconsistent with the present charter.

The first is the recommended route.

The dynamic controller should also be replaced, after fixed-(K) feasibility, by a per-video decision such as

[
K_v=\arg\max_{K\in\mathcal K}
\left[s_\theta(x_v,K)-\lambda,C(K)\right],
]

where:

* (x_v) is deploy-visible input for video (v);
* (C(K)) is a training-side measured target-hardware cost table including relevant scout/transport components;
* (\lambda) is calibrated on training data only and frozen before validation;
* inference for video (v) is invariant to batch composition and order;
* the validation average is **measured**, not forced by using other validation videos.

---

## 4. Novelty and publishability audit

### 4.1 AdaTAD

AdaTAD addresses scalable end-to-end TAD by reducing training memory through a temporal-informative adapter and enabling substantially larger backbones and longer inputs. Its central contribution is not content-dependent heavy-frame acquisition or per-video dynamic pre-backbone budgeting. ([arXiv][1])

DUCA-RIME therefore should not be presented as another detector adapter. Its distinct burden is to establish that adaptive physical acquisition adds value over the strong AdaTAD/OpenTAD uniform route at matched cost.

### 4.2 AdapTok

AdapTok allocates variable numbers of latent video tokens using a causal scorer and constrained allocation for video reconstruction and generation. Its task, token semantics, objective, and evaluation are not equivalent to boundary-sensitive offline TAD. ([arXiv][2])

However, AdapTok raises the novelty bar for claims such as “adaptive video token allocation under a global budget.” DUCA-RIME cannot claim novelty from dynamic allocation alone. The differentiating content would have to be:

* TAD-task utility rather than reconstruction quality;
* exact monotone physical-time acquisition;
* boundary and coverage protection;
* selected-to-physical coordinate transport;
* full-stack detector cost;
* evidence on high-IoU and short-action localization.

### 4.3 Adaptive frame selection

AdaFrame already established per-input adaptive frame usage for efficient video recognition. ([arXiv][3]) Flexible Frame Selection learns to choose a flexible number of informative frames for video-language reasoning, and Adaptive Keyframe Sampling explicitly combines relevance and temporal coverage for long-video question answering. ([Open Access CVF][4]) ([arXiv][5])

Therefore, none of the following is sufficient novelty:

* “select informative frames”;
* “use fewer frames for easy videos”;
* “combine relevance and coverage”;
* “predict variable frame counts”;
* “apply top-(K) to scout scores.”

### 4.4 Plausible publishable wedge

A publishable contribution remains plausible if the final method and evidence jointly establish:

1. **TAD-specific joint allocation:** per-video heavy budget and physical acquisition are optimized for localization rather than recognition, reconstruction, or question answering.
2. **Geometry preservation:** exact monotone positions, endpoint/coverage/gap safeguards, and correct (q\rightarrow t) transport are necessary for high-IoU detection.
3. **Strong attribution:** learned placement beats exact uniform at the same heavy (K), and also survives a genuine full-stack cost match.
4. **Dynamic attribution:** dynamic (K) beats both the identical (K)-sequence uniform-position control and a shuffled video-to-(K) assignment.
5. **Deployment evidence:** scout overhead does not erase the saving.
6. **Negative-results discipline:** failure of any one of these narrows or kills the corresponding claim.

This is a plausible research gap, **not an established novelty claim**. The inspected literature slice is not an exhaustive contemporary audit, so C6 remains blocked exactly as the Source states.  

---

## 5. Protocol, fairness, leakage, evaluator, cost, short-window, and physical-time audit

### 5.1 Data and evaluation protocol

The intended protocol is sound at the design level:

* official OpenTAD THUMOS14 training and validation;
* official Avg-mAP over tIoU 0.3–0.7;
* explicit mAP@0.6 and mAP@0.7;
* short-action and boundary analyses;
* terminal checkpoints;
* multiple registered seeds;
* immutable evaluator;
* no held-out access. 

Before execution, the protocol manifest must freeze:

* dataset manifests and sample counts;
* augmentation and window-generation settings;
* initialization artifact;
* optimizer, schedule, update count, precision, batch construction, and hardware class;
* seed list;
* terminal-checkpoint rule;
* evaluator source hash and arguments;
* cost-measurement procedure;
* result-to-claim rules.

No arm may receive additional tuning, a different stopping rule, a different evaluator, or a different checkpoint-selection rule.

### 5.2 Baseline fairness

Two different notions must not be conflated.

#### Same-heavy-(K) attribution

Learned K384 versus exact-uniform K384 answers:

> At the same number of heavy frames and identical downstream training exposure, are learned positions better than uniform positions?

This is the correct first position-selection test.

#### Full-stack cost matching

The learned arm pays for a scout and transport. Uniform sampling may not. Therefore K384 versus K384 is not necessarily equal full-stack cost.

A paper-level cost comparison requires an additional uniform frontier or a preregistered cost interpolation:

[
C_{\text{uniform}}(K_u)\approx
C_{\text{DUCA}}(K_d)+C_{\text{scout}}+C_{\text{transport}}.
]

The cost-matched (K_u) must be selected from a cost table frozen before DUCA validation metrics are read. It must not be chosen post hoc to improve the visual frontier.

For the Stage B “identical (K) sequence with uniform positions” control, the same scout/controller cost should be retained when the purpose is causal attribution. A second deployment-oriented uniform baseline may omit the scout, but the two interpretations must be reported separately.

### 5.3 Training-exposure fairness

The mixed-(K)-training/exact-uniform-K384-evaluation arm is necessary because a selector trained under variable temporal resolutions may benefit from exposure or regularization independently of learned placement.

All fixed-(K) arms must share:

* data order and augmentation distribution;
* effective updates and global batch;
* initialization;
* detector and backbone;
* target construction;
* loss weights other than the minimum selector-specific term under study;
* terminal stopping;
* evaluator.

The current custom physical-grid head violates the intended “unchanged detector” contract. Even where learned and uniform controls share that head, the result would support a different method definition than the charter’s one.

### 5.4 Leakage and deploy validity

Positive implementation evidence includes:

* a controller-level forbidden-payload denylist;
* a recursive raw-prediction/cache guard;
* config flags disabling existing runtime detections and paper claims;
* tests excluding selected-rank decoding.

Required revision:

* Enumerate every scout/controller input by source, shape, production availability, and creation time.
* Reject objects by typed provenance, not only substring matching.
* Freeze any budget threshold, dual variable, normalization statistic, or cost model on training data only.
* Prohibit validation-wide sorting, quantiles, or budget correction.
* Make each video’s output invariant to batching with different peers.
* Keep teacher predictions, detector caches, raw predictions, counterfactual ledgers, validation labels, and result payloads outside the deploy graph.

The current batch-wide average-budget correction fails the desired per-video invariance even though it does not directly consume labels.

### 5.5 Evaluator and coordinate transport

This is the strongest part of the pinned implementation.

The focused tests require physical-axis metadata, reject direct selected-rank semantics, exercise selected-to-physical segment interpolation, and retain official batched NMS.

The final detector contract should be:

[
[t_s,t_e]*{\text{GT}}
\xrightarrow{\text{physical-to-selected}}
[q_s,q_e]*{\text{GT}}
\xrightarrow{\text{unchanged official detector}}
[\hat q_s,\hat q_e]
\xrightarrow{\text{selected-to-physical}}
[\hat t_s,\hat t_e]
\xrightarrow{\text{unchanged official NMS/evaluator}}.
]

Required fail-closed properties:

* physical positions strictly increase over valid selected slots;
* endpoint interpolation is monotone;
* no use of selected rank as physical time;
* no coordinate conversion after NMS;
* raw predictions and mapped predictions have separate immutable receipts;
* evaluator code and parameters are identical across arms.

### 5.6 Cost

Full-stack reporting must separate and sum:

* full-video decode/input transport attributable to the scout;
* scout compute;
* budget and position planning;
* frame gathering and host/device transfer;
* heavy backbone;
* detector;
* coordinate transport;
* NMS/evaluation where relevant;
* peak memory;
* latency;
* throughput.

Nominal selected-frame count is not a sufficient cost metric. Measurements must use the same hardware, precision, batch policy, warm-up protocol, synchronization policy, and repeated-run estimator.

Because no measured cost data exist, no numeric GPU-hour, latency, or saving claim is admissible.

### 5.7 Short-window contract

No Project Source freezes a complete short-window execution rule. The selector’s reviewed path can retain a shorter valid length while padding the target tensor by repeating a valid position.

The required revision is:

* define (K_{\text{eff}}=\min(K_{\text{requested}},L_{\text{valid}})), with any backbone quantum rule stated explicitly;
* gather exactly (K_{\text{eff}}) unique physical frames;
* never perform duplicate heavy-frame reads merely to reach a tensor length;
* permit post-backbone tensor padding only with an explicit mask and excluded cost;
* record requested (K), effective (K), unique heavy reads, padded slots, and backbone input count separately;
* fail closed for invalid or empty windows;
* apply the identical rule to uniform and learned arms.

### 5.8 Result handling

Metrics must remain unread or non-promotable until identity, split, checkpoint, evaluator, (K), and cost receipts pass. The predeclared analysis order in RESULT_ANALYSIS is appropriate. 

No claim may be based on:

* an intermediate checkpoint;
* a best seed;
* a subset;
* a failed or resumed root without immutable lineage;
* a partial IoU range;
* an over-budget arm;
* an Oracle or synthetic diagnostic;
* infrastructure success.

---

## 6. Route adjudication

### 6.1 Fixed-(K) before dynamic-(K)

**Mandatory.**

Dynamic (K) must remain blocked until learned fixed-K placement has passed a clean Stage A. This is not merely a resource-saving preference. If task-aware positions do not beat exact uniform at a fixed heavy budget, dynamic budgeting creates an underidentified mixture of:

* position quality;
* video-to-budget assignment;
* training exposure;
* cost differences;
* detector adaptation;
* scout overhead.

The project should first establish or falsify C1.

### 6.2 Recommended fixed-(K) route

Use a single K384 learned-position implementation with:

* a deploy-visible lightweight scout;
* exact unique monotone physical positions;
* explicit endpoint, coverage, and maximum-gap protections;
* unchanged official detector head, loss, and NMS;
* target conversion (t\rightarrow q) outside the detector;
* prediction conversion (q\rightarrow t) before NMS;
* no dynamic controller;
* no broad context bridge unless demonstrably necessary.

Compare against exact-uniform K384 and the mixed-(K)-exposure control under the same detector and training contract.

### 6.3 Independent versus nested allocation

If Stage A passes and dynamic (K) is admitted:

1. **Independent-per-(K) is the first reference allocator.** It is the least restrictive and gives the cleanest estimate of achievable placement quality at each budget.
2. **Strict nested acquisition is the sole structured competitor.** It may offer reuse and coherent refinement but introduces an explicit restriction (S_{K_1}\subset S_{K_2}).
3. **Weak-overlap is blocked by default.** It may be admitted only if a preregistered train/utility-only regret analysis shows strict nesting is materially restrictive. The admission test and threshold must be frozen before validation results.
4. No fourth allocator family should be added in this phase.

The current heuristic utility-based position builder is not sufficient evidence for any of these families.

### 6.4 Dynamic-budget control contract

After fixed-(K) success, the dynamic comparison must include:

* dynamic learned (K) plus learned positions;
* the identical per-video (K) sequence with uniform positions;
* a deterministic shuffled video-to-(K) assignment preserving the (K) histogram;
* strong fixed-(K) controls;
* mixed-(K) exposure controls;
* independent versus strict nested allocation, only at the smallest decisive scale.

The shuffle seed and video ordering must be registered before metrics are read.

### 6.5 Adjudication

The scientific direction should not be stopped or replaced. It should be **revised and simplified** before any substantive queue is activated.

---

## 7. Prioritized plan

Resource classes:

* **R0:** static source review, protocol work, or literature work; no model execution.
* **R1:** CPU unit tests and tiny synthetic/tiny-fixture integration checks.
* **R2:** single-device integration smoke; gated and not authorized here.
* **R3:** formal Stage A multi-seed training/evaluation; gated and not authorized here.
* **R4:** formal dynamic-budget Stage B; gated and not authorized here.

### P0 — Freeze canonical evidence identity

**Owner:** Evaluator
**Claim or falsification test:** Every future artifact can be tied to the authorized Sources and immutable code rather than the dirty local tree.
**Input:** Ten v001 Sources, commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, resolved tree and inspected paths.
**Concrete output:** `EVIDENCE_MANIFEST-v001` recording Source versions/hashes, repository, commit, tree, config identities, evaluator identity, authorization flags, and excluded local state.
**Acceptance criterion:** All ten Sources and the exact commit resolve; dirty/unpushed code is explicitly excluded; all GPU/push/result-promotion flags remain false.
**Stop condition:** Any identity mismatch, unavailable canonical dependency, or undocumented local input.
**Resource class:** R0.

### P1 — Complete the dedicated novelty and claim audit

**Owner:** Critic
**Claim or falsification test:** The proposed combination is distinct from contemporary TAD, adaptive tokenization, frame selection, and dynamic inference work.
**Input:** LITERATURE_AND_GAP-v001, candidate mathematical contract, primary papers and official code where available.
**Concrete output:** A claim-by-claim comparison matrix covering task, supervision, inference inputs, budget granularity, position semantics, geometry, cost boundary, and evaluation.
**Acceptance criterion:** Every novelty sentence has a primary citation; unsupported priority language is removed; nearest competing formulation is identified; C6 is either admitted with a precise delta or returned to Human for pivot.
**Stop condition:** A prior method already implements the same offline-TAD joint budget/physical-position formulation and evaluation contract without a defensible methodological difference.
**Resource class:** R0.

### P2 — Build a minimal selector-only fixed-K route

**Owner:** Builder
**Claim or falsification test:** Learned physical positions can be tested without changing the official detector head or loss.
**Input:** Pinned selector, main config, physical-grid tests, charter invariants.
**Concrete output:** An isolated worktree patch containing one learned K384 selector, external (t\leftrightarrow q) adapters, and no custom detector-head or loss-assignment modification.
**Acceptance criterion:** Detector head, loss class, NMS, evaluator, backbone, and training contract are identical to the corresponding official control; valid selected positions are unique and strictly monotone; CPU coordinate/gradient tests pass.
**Stop condition:** Correct physical geometry cannot be externalized without detector modification. In that case, implementation stops and Human decides whether to reframe the paper method.
**Resource class:** R1.

### P3 — Materialize the complete Stage A control matrix

**Owner:** Builder
**Claim or falsification test:** Learned placement can be isolated from heavy budget, detector adaptation, and mixed-resolution exposure.
**Input:** P2 route and EXPERIMENT_PLAN-v001.
**Concrete output:** Immutable configs and a machine-readable diff manifest for dense T768, exact-uniform K384, mixed-(K)-train/exact-uniform-K384-eval, and learned-position K384.
**Acceptance criterion:** The learned-versus-uniform pair differs only in position policy and required selector optimization; all other settings are byte-identical or manifest-equivalent.
**Stop condition:** Any arm needs special tuning, a different checkpoint rule, different evaluator behavior, or unmatched updates.
**Resource class:** R1.

### P4 — Freeze leakage, short-window, physical-time, and cost receipts

**Owner:** Evaluator
**Claim or falsification test:** The route is deploy-valid and the evaluator cannot silently consume invalid geometry or cost accounting.
**Input:** P2–P3 configs, raw guard, controller denylist, coordinate tests, cost requirements.
**Concrete output:** Typed provenance schema, short-window execution contract, physical-time round-trip tests, evaluator hash receipt, and full-stack cost receipt schema.
**Acceptance criterion:** Forbidden payloads fail closed; no validation-wide adaptation is possible; (K_{\text{requested}}), (K_{\text{eff}}), unique reads, padding, and backbone count are distinct; (q\rightarrow t) occurs before official NMS; all cost components are mandatory.
**Stop condition:** Any raw prediction, teacher output, held-out label, selected-rank coordinate, duplicate heavy read, or mutable evaluator enters the route.
**Resource class:** R1.

### P5 — Independent pre-execution red team

**Owner:** Critic
**Claim or falsification test:** The candidate implementation actually answers C1 rather than a detector-head, cost, or data-plumbing question.
**Input:** P0–P4 outputs and code diff.
**Concrete output:** `STAGE_A_ADMISSION_REVIEW-v001` with pass/fail findings for attribution, fairness, leakage, cost, stopping, and evidence promotion.
**Acceptance criterion:** No unresolved scientific confound; no unnecessary dynamic or platform module; every Project invariant maps to code or a fail-closed test.
**Stop condition:** Any P0 scientific-confound issue remains.
**Resource class:** R0–R1.

### P6 — Explicit authorization gate

**Owner:** Human
**Claim or falsification test:** The exact allowed execution is knowingly authorized rather than inferred from repository flags or historical validator output.
**Input:** P0–P5 outputs.
**Concrete output:** A signed allow/deny decision specifying commit, configs, seeds, data, resource ceiling, permissible outputs, and forbidden actions.
**Acceptance criterion:** Authorization is explicit and narrower than or equal to the reviewed manifest.
**Stop condition:** No explicit authorization or any identity mismatch.
**Resource class:** R0.

### P7 — Execute and seal Stage A

**Owner:** Evaluator
**Claim or falsification test:** C1—learned physical-time placement improves over exact uniform under controlled exposure and survives high-IoU, short-action, and full-stack-cost scrutiny.
**Input:** Human-authorized immutable Stage A matrix.
**Concrete output:** Complete terminal-checkpoint, multi-seed metric and cost matrix with per-seed receipts, paired differences, uncertainty, short-action strata, and boundary analysis.
**Acceptance criterion:** Every receipt passes before metrics are read; the predeclared C1 success rule in Section 8 is met.
**Stop condition:** Immediate stop on identity, leakage, evaluator, coordinate, (K), or cost failure. After a complete valid matrix, stop the route before dynamic (K) if C1 fails.
**Resource class:** R3, gated and not authorized by this review.

### P8 — Implement the minimum dynamic-budget route

**Owner:** Builder
**Claim or falsification test:** A frozen train-calibrated per-video policy can allocate (K) without batch coupling or validation adaptation.
**Input:** A passing Stage A result, training-only fixed-(K) utility evidence, measured cost table, registered budget set.
**Concrete output:** Shared scout plus independent-per-(K) allocation heads, frozen training-calibrated cost dual, batch-invariance tests, and one strict-nested comparator.
**Acceptance criterion:** A video’s (K) and positions are invariant to batch composition and order; no validation statistic changes the policy; realized average cost is measured rather than forced; all positions are exact, unique, and monotone.
**Stop condition:** Scout overhead removes the relevant saving, training-only calibration is unstable, or C1 has not passed.
**Resource class:** R1–R2, gated by P7.

### P9 — Execute and seal Stage B

**Owner:** Evaluator
**Claim or falsification test:** C2–C4—video-specific budget assignment and learned positions improve the full-stack accuracy–cost frontier rather than merely reproducing a favorable (K) histogram.
**Input:** Human-authorized dynamic route and frozen Stage B controls.
**Concrete output:** Sealed dynamic, same-(K)-sequence uniform, shuffled-(K)-histogram, fixed-(K), mixed-(K), independent, and nested result matrix.
**Acceptance criterion:** The predeclared C2–C4 rules in Section 8 are met with complete receipts.
**Stop condition:** Gains are explained by the (K) distribution alone, disappear under full-stack cost, violate high-IoU/short-action preservation, or fail leakage/fairness.
**Resource class:** R4, gated and not authorized by this review.

---

## 8. Cheapest decisive falsification, baselines, ablations, thresholds, resources, and early stopping

### 8.1 Cheapest decisive scientific falsification

The cheapest decisive test is **not** dynamic (K). It is:

> Learned physical positions at fixed K384 versus exact-uniform K384, using the identical official detector, training exposure, updates, terminal checkpoint rule, data, evaluator, and heavy-frame count.

The mixed-(K)-training/exact-uniform-evaluation arm is required to test whether any apparent benefit comes from mixed temporal-resolution exposure rather than learned positions.

Dense T768 is a reference, not the primary C1 comparator.

A single registered seed may be used only as a formally labeled feasibility or integrity screen after authorization. It is not a paper result and must not replace the complete registered seed set.

### 8.2 Strongest required baselines

For fixed (K):

1. exact-uniform K384;
2. mixed-(K) training with exact-uniform K384 evaluation;
3. dense T768 reference;
4. full-stack cost-matched uniform frontier in addition to same-(K) attribution.

For dynamic (K), only after C1:

1. identical per-video (K) sequence with uniform positions;
2. shuffled video-to-(K) assignment preserving the histogram;
3. fixed-(K) controls;
4. mixed-(K) controls;
5. independent allocation;
6. strict nested allocation.

The historical uniform reference must be reproduced under the locked protocol rather than cited as evidence for or against DUCA. 

### 8.3 Minimal isolating ablations

Before C1 passes, only these are justified:

1. **Position policy:** learned physical positions versus exact uniform.
2. **Training exposure:** fixed-(K) training versus mixed-(K) exposure, both evaluated with exact-uniform K384 where appropriate.
3. **Coverage safeguards:** full registered safeguards versus a single safeguard-removed variant, only if needed to explain high-IoU/short-action behavior.
4. **Scout supervision:** only if the final implementation uses an auxiliary selector target; compare with that selector-specific term removed while retaining all detector supervision.

The following are not valid ablations:

* selected-rank decoding, because it is wrong;
* reintroducing actionness-top-(K) as a nominally new route without a scientific delta;
* changing the detector head only in the learned arm;
* adding multiple weak-overlap or routing families before independent/nested evidence.

### 8.4 Non-fabricated success and failure rules

No numerical margin or baseline variance is supplied in the Sources. This review therefore does not invent one.

Before DUCA metrics are unblinded, Evaluator and Human must freeze:

* the paired-seed estimator;
* interval construction;
* cost-measurement tolerance;
* high-IoU and short-action non-inferiority margins derived from repeatability of the control, not from DUCA outcomes.

Minimum C1 success rule:

1. The lower bound of the preregistered uncertainty interval for the paired official Avg-mAP difference, learned minus exact uniform, is above zero.
2. High-IoU and short-action results satisfy the frozen preservation/non-inferiority rules.
3. The result remains valid under full-stack cost accounting.
4. No coordinate, leakage, evaluator, or receipt violation occurs.

C1 failure rule:

* the complete valid matrix does not satisfy all four conditions;
* learned placement is dominated by uniform on both accuracy and measured cost;
* high-IoU or short-action damage exceeds the frozen margin;
* any scientific-integrity contract fails.

Minimum C2 success rule:

1. Dynamic DUCA improves the preregistered accuracy–cost measure over the identical-(K)-sequence uniform-position control.
2. It improves over the shuffled video-to-(K) assignment, demonstrating value in assigning budgets to the correct videos.
3. It is not dominated by fixed-(K) or mixed-(K) controls at measured full-stack cost.
4. High-IoU and short-action preservation rules pass.

C2 failure means dynamic budgeting is removed from the main claim. A fixed-(K) paper route may continue only if C1 itself is strong and the novelty audit supports it.

Minimum C4 success rule:

* repeated target-hardware measurements show a strictly lower full-stack cost with the preregistered interval excluding no saving.

### 8.5 Rough resources

No numeric GPU-hour estimate is supportable before baseline throughput is measured.

* **R0:** one CPU/web-capable review environment; no data or model execution.
* **R1:** CPU test suite with tiny tensors and tiny fixtures.
* **R2:** one approved accelerator for shape, memory, and cost instrumentation smoke; no paper evidence.
* **R3:** four Stage A cells multiplied by all registered seeds, each trained and evaluated to its terminal checkpoint on the same accelerator class.
* **R4:** only the minimum admitted Stage B cells after C1.

A resource receipt should express formal cost as:

[
\sum_{\text{cells}}\sum_{\text{registered seeds}}
(\text{terminal training wall time}+\text{evaluation wall time}),
]

using measured baseline wall time rather than a fabricated estimate.

### 8.6 Early stopping

Immediate integrity stop:

* wrong commit, tree, data, config, seed, initialization, evaluator, checkpoint, (K), mask, coordinate mapping, or cost receipt;
* forbidden input or held-out access;
* NaN, numerical failure, duplicate heavy reads, or invalid physical positions.

Scientific stop:

* no metric-driven intermediate-checkpoint stopping;
* no best-seed stopping;
* no expansion to dynamic (K) unless the complete sealed Stage A passes;
* stop dynamic (K) if the scout cost already removes the saving under the frozen cost protocol;
* stop after Stage B if the shuffled-(K) control explains the gain.

A performance-based futility rule may be used only if it is mathematically frozen from independent baseline variability before DUCA metrics are observed.

---

## 9. Explicit do-not-do list

1. Do not build a general admission simulator, scheduling platform, Monte Carlo framework, or broad context-routing system before C1.
2. Do not implement dynamic (K) before the selector-only fixed-(K) route is admitted.
3. Do not silently retain a custom detector head while claiming an unchanged official detector.
4. Do not compare learned K384 against uniform K384 and call it full-stack cost-matched without accounting for the scout.
5. Do not use selected rank as physical time.
6. Do not move (q\rightarrow t) conversion after NMS.
7. Do not reintroduce rejected local-cell, naive actionness-top-(K), hard-query-deletion/SparseHead, DCSR-G1, or ODF-CR-G2 routes without new falsifying evidence.
8. Do not use validation-wide sorting, quantiles, budget balancing, or adaptive dual updates.
9. Do not consume GT, teacher outputs, raw predictions, prediction caches, result payloads, counterfactual ledgers, or held-out labels at inference.
10. Do not duplicate physical frame reads to satisfy a padded tensor shape.
11. Do not tune arms independently on validation.
12. Do not select seeds, checkpoints, IoU thresholds, short-action definitions, cost tolerances, or uncertainty methods after reading DUCA results.
13. Do not promote subset, intermediate, single-seed, failed-root, over-budget, Oracle, synthetic, upstream, or infrastructure metrics.
14. Do not treat validator or unit-test success as model evidence.
15. Do not consume the dirty/unpushed coordinator tree.
16. Do not admit weak-overlap allocation without its preregistered train/utility-only nesting-regret gate.
17. Do not expand to additional detectors, datasets, or budgets before the smallest decisive Stage A is sealed.
18. Do not run GPU, remote, Slurm, formal evaluation, push, result promotion, or submission without a new explicit Human authorization.

---

## 10. Nine cvpr-pro-lab drift questions

### 10.1 Is the implementation faithful to the scientific hypothesis?

**Partially, but not sufficiently.**

The pinned route supports deploy-visible scouting, physical positions, and tested (q\rightarrow t) mapping. It does not yet support a clean joint learned dynamic-budget method, and the main config contradicts the unchanged-detector invariant by changing detector-head geometry and loss assignment. The dynamic controller is also batch-coupled and heuristic. 

### 10.2 Is current progress science or infrastructure?

Most verified progress is **infrastructure and semantic correctness**:

* physical-coordinate transport;
* NMS preservation;
* config identity checks;
* validator and local gradient smoke;
* payload guards.

The scientific claims C1–C6 remain unsupported. Validator success must not be relabeled as model feasibility.     

### 10.3 Is there unnecessary complexity?

**Yes.**

The selector exposes many strategies, auxiliary objectives, transport modes, and route variants; the bridge contains interpolation, selected-axis, context, projection, and conditioning machinery; the validator includes broad execution infrastructure. Much of this is not needed to falsify C1. The project Source already warns against a general admission or scheduling platform.    

The paper route should retain only the scout, exact physical acquisition, minimal coordinate adapters, official detector, and receipt instrumentation.

### 10.4 Are baseline, compute, data, tuning, stopping, and evaluator fair?

**Designed to be fair, but not yet implementation-ready.**

The Source specifies strong Stage A controls, full official data, terminal checkpoints, multiple seeds, and full-stack cost. The exact-uniform control exists. However:

* the complete mixed-(K) exposure cell was not established;
* both inspected sparse routes inherit an altered physical-grid detector;
* same (K) does not imply same full-stack cost;
* numerical thresholds and cost tolerances are not frozen.

These must be repaired before execution.    

### 10.5 Are leakage, cherry-picking, post-hoc drift, and result relabeling controlled?

**Partially.**

Positive controls include explicit forbidden-input rules, a raw-prediction guard, no held-out authorization, terminal checkpoints, and BLOCKED_PRE_RESULT documents.   

Residual risks are:

* batch-wide validation budget adjustment;
* substring-based rather than typed provenance enforcement;
* incomplete freezing of thresholds and uncertainty rules;
* potential relabeling of physical-grid-head results as selector-only evidence;
* historical validator flags being mistaken for current execution permission.

### 10.6 Would a skeptical CVPR reviewer find the work formal, reproducible, and publishable?

**Not at the pinned state. Potentially yes after revision and evidence.**

A skeptical reviewer would currently challenge:

* the detector-change confound;
* the heuristic, batch-coupled budget controller;
* absence of a formal result matrix;
* absence of measured full-stack cost;
* incomplete contemporary novelty verification;
* the large engineering surface relative to the core idea.

The plausible differentiator is the TAD-specific combination of joint budget/physical acquisition, boundary-sensitive geometry, correct coordinate transport, and rigorous cost-matched evidence—not adaptive frame scoring by itself.   ([arXiv][1])

### 10.7 Is falsification evidence being preserved?

**Yes at the Source level; it must now be enforced mechanically.**

FAILURES_AND_PIVOTS preserves prior negative scientific routes and correctly separates infrastructure failures. RESULTS and CLAIM_MAP refuse to promote unsupported outcomes. The pinned commit gives an immutable code anchor.   

Future receipts must retain failed cells, invalidation reasons, and complete lineage rather than overwriting them.

### 10.8 Should the project continue, simplify, pivot, or stop?

**Revise and simplify.**

Continue the scientific question, but:

* freeze dynamic (K);
* remove detector-head drift;
* implement the complete fixed-(K) control matrix;
* certify leakage, short-window, evaluator, and cost contracts;
* run only after explicit Human authorization.

A pivot is warranted only if the clean fixed-(K) test fails or the literature audit finds the claimed combination already occupied.   

### 10.9 Is this a publishable idea or a complete engineering system?

The potentially publishable idea is narrow:

> task-aware offline-TAD allocation of heavy-frame budget and exact physical positions, with geometry-preserving transport and demonstrated full-stack advantage.

The complete selector/bridge/validator platform is not itself the paper contribution. Building the latter before proving the former would dilute attribution, increase failure surface, and consume resources without advancing C1–C6.   

---

## 11. Exact commit, paths, Source versions, and access limitations

### 11.1 Project Sources used, in requested order

1. `PROJECT_CHARTER-v001` — active draft; canonical hypothesis, benchmark, evidence, and authorization boundary. 
2. `LITERATURE_AND_GAP-v001` — active draft; relevant families, gap, novelty burden, and need for dedicated audit. 
3. `ROUTE_DECISION-v001` — active draft; independent/nested/conditional weak-overlap routes and deploy invariants. 
4. `EXPERIMENT_PLAN-v001` — active draft; Stage A, conditional Stage B, measurements, and stop rules. 
5. `IMPLEMENTATION_STATUS-v001` — active draft; canonical revision, implemented surfaces, exclusions, and audit targets. 
6. `RESULTS-v001` — `BLOCKED_PRE_RESULT`; no formal performance evidence. 
7. `RESULT_ANALYSIS-v001` — `BLOCKED_PRE_RESULT`; predeclared analysis and prohibited interpretations. 
8. `FAILURES_AND_PIVOTS-v001` — active draft; scientific negatives, engineering failures, pivot, and anti-repetition rules. 
9. `CLAIM_MAP-v001` — active draft; C1–C6 evidence requirements and unsupported state. 
10. `PAPER_DRAFT-v001` — `BLOCKED_PRE_RESULT`; paper skeleton and writing boundary. 

All ten are dated August 10, 2026 and identify the same canonical GitHub revision.

### 11.2 GitHub identity used

* Exact commit: `63a726a4aaf48ecbf6780bb196de43a890c6b4df`
* Exact tree: `b50b7d986448657ba2debbaac0ef2610fce1bdfb`
* Commit message observed: `docs: freeze DUCA total-60 plugin experiment design`
* The commit and recursive tree were directly resolved through the GitHub connector.

### 11.3 Code paths actually inspected

| Path                                                                                             | Access and review scope                                                                                                                                        |
| ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `opentad/models/selectors/pc_ot_mras_prebackbone_frame_selector.py`                              | Targeted inspection of helper logic, main selector initialization, hard sparse-plan construction, fixed-budget scout, masks, metadata, and padding behavior.   |
| `opentad/models/selectors/pc_ot_mras_dynamic_budget_controller.py`                               | Direct inspection of configuration, feature extraction, scoring, batch-level budget adjustment, position construction, forbidden-input handling, and receipts. |
| `opentad/models/selectors/pc_ot_mras_reader.py`                                                  | Direct inspection of monotone slot transport and deploy-visible full-video reader.                                                                             |
| `opentad/models/necks/pc_ot_mras_detector_bridge.py`                                             | Targeted inspection of projection, interpolation/selected-axis modes, metadata validation, runtime context, and prediction guards.                             |
| `opentad/models/utils/pc_ot_mras_raw_prediction_guard.py`                                        | Complete guard logic inspected.                                                                                                                                |
| `configs/adatad/thumos/pc_ot_mras_prebackbone_c3_physical_grid_actionformer_full_train_n16r4.py` | Complete principal config inspected.                                                                                                                           |
| `configs/adatad/thumos/pc_ot_mras_exact_uniform_c3_physical_grid_actionformer_n16r4.py`          | Complete exact-uniform control config inspected.                                                                                                               |
| `tools/bata/validate_pc_ot_mras_prebackbone_c3_physical_grid_full_train_gate.py`                 | Inspection of identity, config, local smoke, gradient, physical-axis, and gate checks.                                                                         |
| `tests/test_c3_physical_grid_round_trip.py`                                                      | Coordinate round-trip, range, metadata, selected-rank exclusion, and NMS tests inspected.                                                                      |
| `tests/test_c3_physical_grid_actionformer_candidate.py`                                          | Candidate config, head, loss, and physical-grid tests inspected.                                                                                               |
| `tests/test_c3_physical_grid_full_train_execution.py`                                            | Config identity, exact-uniform control, validator, and execution-contract tests inspected.                                                                     |

### 11.4 Access limitations

* I accessed GitHub through direct connector reads at the exact commit.
* A local repository clone was not available, so I did not execute Git commands, inspect a local worktree, or run tests.
* I did not execute Python, training, evaluation, Slurm, GPU, or cost measurement.
* The very large selector file was inspected in targeted relevant sections, not exhaustively line by line.
* GitHub symbol search did not establish a main-route reference to the dynamic controller, but this is classified as **integration unestablished**, not proof that no indirect integration exists.
* I did not inspect the excluded dirty/unpushed local tree.
* I did not access held-out data or formal result artifacts.
* The literature review used a limited set of primary paper records sufficient for an initial novelty-risk audit; it is not the dedicated exhaustive C6 audit.
* No performance or resource number was inferred from code, historical context, or infrastructure.

---

## 12. Final route status

Preserve the DUCA-RIME scientific question, the fixed-(K)-before-dynamic ordering, the physical-time coordinate invariant, and the strong negative-evidence discipline.

Before any substantive queue or experiment:

* replace the custom physical-grid detector route with an external coordinate adapter around the unchanged official detector;
* freeze a complete Stage A control matrix;
* separate same-heavy-(K) attribution from full-stack cost matching;
* freeze exact short-window and cost contracts;
* defer and redesign dynamic budgeting as a training-calibrated, batch-independent per-video policy;
* complete the dedicated novelty audit;
* obtain explicit Human authorization.

The project is neither ready to continue unchanged nor scientifically compelled to pivot or stop.

REVISE

[1]: https://arxiv.org/abs/2311.17241 "https://arxiv.org/abs/2311.17241"
[2]: https://arxiv.org/abs/2505.17011 "https://arxiv.org/abs/2505.17011"
[3]: https://arxiv.org/abs/1811.12432 "https://arxiv.org/abs/1811.12432"
[4]: https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html "https://openaccess.thecvf.com/content/CVPR2025/html/Buch_Flexible_Frame_Selection_for_Efficient_Video_Reasoning_CVPR_2025_paper.html"
[5]: https://arxiv.org/abs/2502.21271 "https://arxiv.org/abs/2502.21271"
