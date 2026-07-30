---
type: idea
node_id: idea:geo-route-adatad
title: "NativeTokenSelect-first routing for offline TAD"
stage: tested
status: free_v1_closed_cer_discussed_stability_v2_running
tags: ["offline-tad", "native-token", "token-selection", "geometry", "adatad"]
added: 2026-07-22
updated: 2026-07-30
---

# NativeTokenSelect-first Routing for Offline TAD

## One-line thesis

First establish detector-supervised, ROI-free exact-K selection of valid
source-native tubelets with coordinate-lineage packed temporal adaptation.
Only if that base survives matched controls may continuous geometry be retained
as a structured add-on. The unchanged AdaTAD-derived detector loss is the
primary learning signal.

## Design status and boundary

Exact clean source `7be8363e` passed Linux tests `82/82`; gate-only Job
`1200602` demonstrated two concurrent, node-bound c10d stores with distinct
dynamic ports and preserved long-worker lifetime after complete short-parent
exit. P0R Jobs `1200611`--`1200613` then passed and sealed suite
`693034b276697e92ae915ea5f40cebdd5d01a76bad65f46e5639844654f210e9`
as `PASS_MECHANICAL_ONLY`. The P0 finalizer wrote that valid receipt before a
submit-cap guard stopped automatic P1 submission, so no partial matrix was
created. Supported sealed-parent bootstrap `1200652` subsequently launched the
unchanged seven-arm P1R matrix as Jobs `1200663`--`1200669`, with automatic
selector `1200670`, under fresh root
`georoute_nativefirst_7be8363e_p1p3_20260728_2225`. Six arms completed with
valid development-only results, including free and hybrid, but ROI-only failed
after training during development video decode and emitted no stage result.
The afterok selector is consequently dependency-unsatisfied and emitted no
decision. The idea is now `tested`, not `empirically_supported`; P2/P3,
official test, efficiency claims, and paper claims remain closed.

The correctness replacement is implemented; clean-commit remote focused tests
pass and replacement P0R sealed `PASS_MECHANICAL_ONLY`. Its first P1R matrix,
however, is not a scientific test of the idea. Concurrent single-GPU leaves
used implicit `torch.distributed.run --standalone` localhost port `29400`.
Free NativeTokenSelect `1199869` shared node `g0043` with fixed lattice,
attached to its TCPStore after a bind collision, and failed in Epoch 8 when
that store closed. Hybrid `1199871` had the same collision with random on
`g0048` and failed in Epoch 6 when random's store closed. The selector is
consequently `DependencyNeverSatisfied`. Dense, fixed,
fixed-plus-geometry, random, and ROI-only produced development-only diagnostic
cells,
but they cannot establish the required native-base comparison without a valid
free result and selector receipt. Descriptively, the fixed geometry
side-channel changed fixed's single-seed Avg-mAP by `+0.21` while reducing
mAP@0.6/0.7 by `0.35/0.08`; this is not authorization to interpret geometry.
The failures are deployment-isolation evidence only, not support for or
against NativeTokenSelect, and they do not authorize geometry, P2/P3, official
test, an efficiency claim, or a paper claim. A valid future test must repeat
the complete frozen matrix in a new namespace with a unique or kernel-assigned
per-leaf rendezvous endpoint and an explicit concurrent-node isolation gate.
That historical replacement used c10d `127.0.0.1:0` with a
Slurm/job/stage/variant/seed/phase-bound ID; a P0 same-node concurrent gate
verifies the two actual runtime ports, run IDs, and independent lifetimes; and
each P0 model report is hash-bound to the isolation receipt from the same
Slurm leaf. This changes no model or selector decision. It is remotely
mechanically validated but remains not empirically supported until the complete
new seven-arm matrix and frozen selector finish.
The first remote gate source `a2ebd060` passed `82/82` Linux tests but its
three P0 leaves `1200510`--`1200512` failed before model execution: a fixed
0.5/2.0-second lifetime contrast measured torchrun parent teardown latency and
could falsely report that the long worker was no longer alive. The replacement
uses a deterministic peer-exit handshake instead: the long worker waits until
the controller observes full short-parent exit and publishes a marker. This is
an infrastructure-gate correction only and creates no model evidence.
Source `bfee5790` subsequently proved that deterministic handshake, but its P0
leaves failed another over-strict assertion before model execution: Torch
exposed `MASTER_ADDR` as the allocated hostname rather than literal loopback.
Slurm diagnostic `1200560` observed the correct unique run ID and dynamic port
alongside `MASTER_ADDR=g0024`. The gate now binds both probe hostname and
master address to the exact current node; it must pass a standalone Slurm gate
before any further P0 deployment; Job `1200602` satisfied that condition under
source `7be8363e`.
ROI-only reached Avg-mAP/mAP@0.6/mAP@0.7 `13.18/11.28/8.95`, descriptively
above fixed, random, and fixed-plus-geometry at high tIoU, but it is not the
ROI-free native base and cannot satisfy the hierarchical gate. In the
replacement run, free NativeTokenSelect reached Avg-mAP/mAP@0.6/mAP@0.7
`10.03/7.80/5.27`, below fixed `12.42/10.75/7.17`, random
`12.68/10.76/7.53`, and fixed-plus-geometry `12.63/10.40/7.09`. This is strong
descriptive negative evidence for the current free selector, but the ROI
decode failure prevented the frozen selector receipt, so it is not promoted to
a formal empirical verdict. Hybrid reached `13.23/11.35/8.81`; it cannot rescue
or authorize geometry because the native base did not pass first. The
replacement idea remains not empirically supported.

The later independent estimator/representation pilot did not change this
scientific verdict. Its first runtime `02b6efe7` passed Linux tests but all six
P0 leaves `1203380`--`1203385` failed before model evidence: two exposed a
script-mode import bug after a passing rendezvous gate and four co-located
leaves exceeded an under-instrumented 30-second readiness bound. No training
ran. Finalizer `1203393` additionally exposed JSON-key-order validation and
wrote no finalization. The repair is implemented with module-mode P0, early
source-root bootstrap, Slurm-job-scoped `127/8` plus kernel port, 120-second
diagnostic readiness with whole-process-group cleanup,
key-order-independent arm binding, fail-safe control receipts, and an
all-terminal fail-closed DAG. `srun --resv-ports` is explicitly unavailable on N16R4 by Jobs
`1203460/1203461`. Until a fresh Linux suite, same-node gate, six P0 leaves, and
all six exploratory arms complete, CER stays `discussed`, the pilot stays
non-empirical, and Geometry Zoom remains unauthorized.

## Why geometry is conditional

Geometry parameterizes a structured spatial support distribution with
continuous center, scale, aspect ratio, and temporal trajectory. It may provide
contiguous context and temporal coherence, but those benefits are hypotheses.
The primary base is a truly geometry-free selector because one rectangle may
miss disjoint actor, object, and scene evidence. Geometry is retained only if a
corrected hybrid strictly adds under the same validity, pooling, adapter,
budget, and total-cost contract.

## Proposed minimal model

1. A lightweight dense global scout observes the full offline video at reduced
   spatial resolution and predicts ROI knots every fixed temporal stride.
2. The native support follows the pretrained floor Conv3d semantics. For
   180x320 input it uses 176x320, an 11x20 grid, and a boolean validity mask;
   it creates no replicated or interpolated patch support.
3. The ROI-free base fixes geometry to the full frame and freezes the geometry
   head. Its residual scorer selects exact-K valid tubelets. Geometry-enabled
   variants use interpolated `(cx, cy, w, h)` knots to define a structured
   field over the same absolute native grid.
4. In hybrid, the geometry branch receives a fixed share `K_geo` of the exact input token
   budget. A residual scorer operating on the scout receives `K_res` tokens
   outside or complementary to that support. A small fixed global-context
   allowance `K_ctx` prevents all scene evidence from vanishing. The union is
   packed to `K_geo + K_res + K_ctx = K_input`.
5. Attention, MLP, and the original temporal Adapter operate on packed selected
   tokens. The Adapter preserves absolute spatial lineage across adjacent
   tubelets; absent lineage neighbors are zero and full-K matches dense.
6. Every P1 arm uses uniform-selected aggregation, producing one feature per
   temporal tubelet and preserving the detector-facing `[B,384,768]` contract.

A-MoD is not part of the minimal model or P1R. It is a conditionally gated P2
extension only after a learned route survives the primary controls.

ToMe-style merging is not part of the initial main model. It is an important
matched baseline and may be restricted to low-value global-context tokens in a
later ablation; merging ROI-local tokens risks blurring the fine spatial detail
needed for high-tIoU temporal localization.

## Learning and cost rules

- The primary objective is the audited AdaTAD-derived Focal classification and
  DIoU regression loss; this configuration has no independent quality loss.
  There is no spatial GT, teacher, oracle, test signal, or manual ROI in the
  policy path.
- Native-token selection uses differentiable soft support only for the relaxed
  dense path. Hard exact-K membership has no ordinary pathwise ROI derivative;
  P0 must compare a score-function hard-policy estimator with a clearly marked
  biased straight-through surrogate. Nonzero gradients alone do not establish
  a useful routing mechanism.
- Geometry constraints only prevent degenerate boxes and implausible temporal
  jitter. They may not replace detector supervision or encode a hand-designed
  action prior.
- The full cost ledger includes scout, geometry/residual routing, native patch
  embedding, packed backbone and Adapter, detector, NMS, decode,
  preprocessing, H2D, latency, memory, and energy. Patch-token or FLOP counts
  alone cannot establish an efficiency claim.

## Required matched comparison matrix

All variants must share source frames, pretrained VideoMAE initialization,
AdaTAD detector/head/loss, training updates, seeds, token budget, and real
end-to-end cost protocol:

1. P0R: dense native numerical/reference route plus score-function and corrected
   hybrid
   routes; it checks one real CUDA step, real detector losses, exact-K,
   validity, full-K parity, packed Adapter accounting, gradients, memory,
   component trace, storage profile, and one-heavy-forward accounting only.
2. P1R: dense native, fixed lattice, fixed lattice plus learned geometry
   side-channel, random, free native TokenSelect, geometry/ROI-only, and
   geometry plus residual TokenSelect. Pooling and Adapter execution are
   identical across arms.
3. P2: promote only the P1 winner to three seeds and budgets. ToMe and
   A-MoD are separately gated extensions, never assumptions embedded in the
   P1 primary claim.
4. P3: freeze a surviving configuration before a second detector/dataset and
   a one-time sealed official test.

Report Avg-mAP and mAP at tIoU 0.3--0.7, especially 0.6/0.7, short-action and
boundary diagnostics, exact token/depth utilization, per-tubelet coverage,
ROI temporal stability, and measured end-to-end latency/memory/energy.

## Paper evidence package

The theory package establishes conditional operation-count, score-function,
and structured-approximation statements only; it makes no theorem about mAP
or wall-clock speed. The paper figure tooling renders source-bound
architecture, Pareto, high-IoU, budget, ablation, diagnostic and stability
figures plus raw-seed and LaTeX tables from validated records outside the
repository. The new fixed-lattice-plus-geometry control is mandatory before
attributing a gain to changing native-token support rather than merely
injecting a learned geometry embedding.

## Decision rule

First require `free` to beat fixed lattice, random, and the
fixed-lattice-plus-geometry side-channel on the frozen high-tIoU accuracy rule
and cost less than dense. Failure stops learned routing. Only after this base
passes may hybrid geometry be considered; it must beat free, random, and the
geometry side-channel without greater total cost. A pass promotes Route A;
otherwise the simpler Route B advances. Even a Route A pass is native-token
geometry routing, not generic dynamic cropping or a sequential pixel zoom.

## A-MoD correction

A-MoD is a valid pretrained-model adaptation baseline, not merely a
from-scratch language-model technique. Its attention-derived routing reports
no additional trainable routing parameters and adaptation from pretrained
transformers. This motivates a VideoMAE compatibility experiment, but does not
prove compatibility, speed, or localization benefit in AdaTAD/TAD.

For this candidate, the required schedule is an initial dense prefix followed
by alternating Dense-MoD pairs. A-MoD must score a MoD block from the full
attention state of the preceding dense block; a consecutive all-MoD tail would
not preserve that paper-level premise and is not the intended comparison.

## FlashVID transfer boundary (2026-07-23)

FlashVID is now recorded as a relevant video-token-compression reference, but
not as an empirical precedent for GeoRoute. Its reported 90% visual-token
reduction with 99.1% retained score is a LLaVA-OneVision VLLM result: 57.9
versus 58.4 average score under an aligned 10% retention-budget protocol. It
is neither 99.1% absolute accuracy nor TAD mAP.

Its useful hypothesis is narrower: a selector should jointly preserve task
relevance, feature diversity, and motion-tolerant cross-frame correspondence.
FlashVID itself runs a full vision encoder to obtain features and attention,
then compresses for the LLM under `torch.no_grad()`. It therefore cannot be
ported as a pre-backbone AdaTAD efficiency method and cannot support our
detector-gradient or native-token claims.

P1 is deliberately unchanged. If and only if the P1 hybrid survives against
free TokenSelect, P2 may test a scout-only, FlashVID-inspired
relevance-diversity-correspondence residual baseline with exact-K lineage and
one-heavy-forward accounting. It must be labelled an adaptation, not a
FlashVID reproduction, and is removed if it loses on high-tIoU or total cost.

## External v1 review absorption (2026-07-23)

The archived external review is `HOLD`, not an implementation acceptance. Its
code findings are accepted: the current U128 route uses fixed-output
`grid_sample`, evaluates VideoMAE twice, contains no learned policy, and the
current ActionFormer configuration exposes Focal plus DIoU rather than a
quality loss. Its insistence on a one-heavy-forward native-tubelet P0 and the
Dense-MoD interval requirement is accepted.

The following implementation choices remain hypotheses rather than frozen
model facts: a review-proposed 48-knot / 16-source-frame cadence, `K=64`, a fixed 96-pixel scout,
a CPU-pinned source gather, a dense-scatter sparse adapter, a 4,800-update
schedule, and numerical latency or mAP thresholds. A score-function estimator
is mathematically honest for a stochastic hard policy, but must be measured
against a labelled straight-through surrogate for variance, detector utility,
and total cost before it is made the main algorithm. Semantic violations kill
the claim; an early gather or adapter bottleneck is a HOLD/pivot condition, not
automatic evidence that the research hypothesis is false.

## CER-TAD review absorption and narrowed next hypothesis (2026-07-29)

The new Pro review is accepted with major revision, not wholesale. It correctly
closes the present Free v1 as the primary candidate on strong descriptive
negative evidence, forbids promoting Hybrid through a failed hierarchy, and
identifies the current path as token routing rather than Geometry Zoom. Its
context / geometry / residual role decomposition is a plausible successor
hypothesis, provisionally named boundary-conditioned complementary
pre-backbone routing.

The full successor is only `discussed`. Dynamic role counts at fixed total K,
the count likelihood, critic, boundary auxiliary, temporal-stability loss,
coverage penalty, and their weights are not reproducibly specified. The
eleven-arm matrix also cannot reuse the old seven-arm selector. Moreover, the
current sparse adapter confounds support and representation: one switch controls
both absolute and ROI-relative coordinates, while geometry is always projected
into the pooled feature.

The estimator/representation preexperiment is now
`tested_complete_go_pilot_design_only`: its full exact-index decode census,
numerical PL/ST and representation-isolation KATs, and all six
prediction-hash-preserving replays passed. It still authorizes no CER or paper
claim.

The separately frozen six-arm exploratory pilot is now
`old_namespace_sealed_incomplete_replacement_ready_capacity_gate`.
Its first runtime failed before model evidence and is not resumed. Fresh
runtime `cbe0a082` passed the full remote Linux suite, concurrent same-node
rendezvous gate, and all six P0 leaves, but residual-PL Job `1203715`
hard-failed on real batch 0 after eight AMP retries and produced no checkpoint
or metric. The other five leaves completed only to preserve complete failure
provenance; they cannot be interpreted. Finalizer `1203720` completed `0:0`
and sealed `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with an empty contrast
set and self-hash
`738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`.
The root cause is FP16 temporal PL
accumulation at the real `T=384/N=220/K=64` horizon. An estimator-equivalent
FP32 likelihood/reduction repair and a production-grid-bound AMP KAT are
now `tested` for numerical correctness at exact commit `30f9ca6f`: remote
Linux `120/120` and CUDA KAT Job `1203873` passed with an FP32 objective above
the FP16 range and finite scaled gradients. The old run has now sealed
INCOMPLETE; the repair must still pass the all-at-once capacity check, fresh
per-arm P0, and a full new six-arm namespace. The design remains the smallest
way to identify estimator, support,
and representation effects, but no single-seed contrast or CER/paper method
exists from the failed run or the KAT.

A fresh independent agent, given only the raw review and repository paths,
audited the absorption, six-arm causal contract, no-leak binding, PL AMP repair,
and all-or-none DAG. Its verdict is
`DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`: the minimal exploratory pilot is
specified well enough that another Pro discussion is unnecessary, but the old
namespace had to seal INCOMPLETE and the new exact `30f9ca6f` namespace must
pass all six P0/stage paths. The old-closeout condition is now satisfied. This
review does not advance CER beyond
`discussed` and is not empirical support.

The complete replacement is now `experiment_running` from exact source
`30f9ca6f`, not a new method version. Its all-at-once capacity preflight passed,
six schema-v4 P0 Jobs `1204015`--`1204020` plus finalizer `1204021` passed, and
six frozen stages `1204022`--`1204027` run in parallel under new closeout
`1204028`. This may identify estimator/support/representation effects only if
all six finish; it still does not implement or establish full CER or Geometry
Zoom.

The replacement exposed a second, narrower numerical gap: Job `1204023`
exhausted full-model AMP retries even though isolated PL arithmetic passed P0.
Independent review showed P0 never combined autocast, GradScaler, the real
model graph, and an optimizer update. This does not falsify PL versus ST;
it invalidates the execution evidence. The next method-neutral repair is an
FP32 scout/score-function route branch plus full-graph AMP P0, with the same six
arms and contrasts.

This repair is now numerically `tested` at exact source `c822add3`: the clean
Linux suite passed `121/121`, and full-graph CUDA P0 Job `1204087` passed
schema v5 with FP32 scout execution, finite detector-plus-score-function
gradients and an unchanged GradScaler floor. It changes precision and gate
coverage only; it is not a new CER component and supplies no empirical support.
The failed `30f9ca6f` namespace has now sealed INCOMPLETE with empty contrasts.
The complete fresh `c822add3` six-arm namespace passed all six schema-v5 P0
leaves plus its finalizer, but residual-PL stage `1204309` again exhausted all
eight AMP retries on real batch 0 and hard-failed without a checkpoint or
metric. The other five stages completed only for terminal provenance. ROI-PL
Jobs `1204312/1204313` each accumulated 11 failed optimizer attempts before
successful replay and reached GradScaler `64`; this is a numerical-stress
signal, not a formal failure, because the registered source rule is eight
exhausted retries within one batch rather than cumulative count `>10`.
Closeout `1204314` sealed
`PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, false all-six, empty contrasts,
and all promotion guards false. This falsifies the sufficiency of the current
synthetic full-graph P0 as a real-batch stability certificate, not the
scientific PL hypothesis itself. GeoRoute/CER remains without empirical
support; no further repair is designed until a new real-batch cause analysis
and experiment decision are recorded.

That cause analysis is now `implemented` at exact local source
`832caedd3713f477cb4b2f29a692acba9cd5a836` as
`exp:georoute-real-batch-amp-diagnostic-v1`. The chosen route observes the
unchanged production train engine for paired residual-PL/ST real batches,
fingerprints inputs/RNG, and localizes scaled, unscaled and clipped gradients.
It may authorize only a minimal cause-matched numerical repair followed by a
fresh real-data stability gate. It does not add CER/Geometry components or
produce accuracy evidence. Local pure checks and required C3 regressions pass;
remote Linux/CUDA validation and the three-job no-metric DAG remain pending.
The first exact-source DAG proved only one missing execution contract:
`mmengine.Config` rejected `del cfg[key]` before either arm reached data/model
execution. Both stage jobs failed symmetrically and the afterany finalizer
sealed `DIAGNOSTIC_INCOMPLETE_NO_REPAIR`. Minimal `Config.pop` repair
`64d991f9` changes no estimator, representation, data, seed, K or decision rule;
its fresh namespace instead exposed that `block_list` is an exclusion list.
Corrected source `047f643f` passed remote `149/149` and brought both arms to
the same real batch with matching input/RNG hashes and finite forward losses.
Both nevertheless stopped before backward because the diagnostic accidentally
used strict deterministic error mode while the failed pilot used deterministic
warn-only mode. Finalizer `1204910` sealed
`DIAGNOSTIC_INCOMPLETE_NO_REPAIR`, so no model hypothesis was updated. Candidate
`861e9b1e` bound the historical warn-only seed policy without changing the
model or scientific intervention. Its matched PL/ST run localized the original
failure to PL score-function gradient scale: PL required scale `128`, while ST
succeeded at `65536` on the identical batch/RNG state. Per-tubelet temporal
mean was therefore implemented as the minimal model-component normalization.
The subsequent 32-batch zero-skip stability-v1 did not pass: PL backed off at
batch 3 in the score-function head and ST backed off at batch 21 in the detector
head. Because exact official AdaTAD uses dynamic loss scaling rather than a
zero-skip-at-65536 rule, this is a numerical gate HOLD, not a scientific
estimator or performance verdict. The next admissible step is a separately
versioned, no-metric stability gate matching official AMP semantics on an
independent data order.

That v2 numerical gate is now implemented and locally `tested`. It uses the
official default dynamic-scaler and per-batch scheduler/EMA transition cadence,
but explicitly does not match the official scheduler hyperparameters or full
training recipe. Exact runtime source `27fba03c` passed the clean remote
Linux/Torch suite `168/168`; the terminal two-arm run consists of PL/ST Jobs
`1205588/1205589` and afterany finalizer `1205590`. Passing it could authorize
only protocol freezing, not an accuracy, estimator, efficiency or paper
conclusion.

PL crossed the frozen numerical rule: its third nonconsecutive skip at batch
`29` reduced scale to `8192`, below the `16384` floor. PL eventually completed
61/64 updates and ST 62/64; both had finite forward losses and stable final-16
tails. Finalizer `1205590` nevertheless correctly sealed
`OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`. It is a numerical stability finding
only and does not rank PL versus ST, change the model claim, or authorize the
formal paper protocol.

The route therefore advances only to a matched mechanism decomposition, not to
an estimator choice. The user-provided Pro audit recommends
`NEW_MATCHED_DIAGNOSIS_BEFORE_REPAIR`; the project accepts the central verdict
and implements a seed-7367 PL/ST study that observes analytic policy gradients,
actual residual-logit gradients, FP32 pre-hook GradBuckets, detached FP16 casts,
and post-unscale/clip states while retaining the standard hook. This tests
whether PL-specific pressure is intrinsically nonfinite, amplified in the scout
VJP, or merely crosses FP16 communication range, and whether detector failures
are shared.

The implementation corrects an impossible literal matchedness requirement:
after batch zero, PL and ST cannot be required to share CUDA RNG state because
only PL samples Gumbel noise. Data and CPU RNG remain matched across all 64
batches; CUDA RNG is matched initially and later divergence is recorded without
replay/reset. This is a protocol correction, not a model intervention. Formal
accuracy/cost experiments remain blocked until one mechanism-specific repair
passes a fresh no-performance gate.

The publication path is explicitly separate: the one-seed 20-epoch pilot is
exploratory. A paper result must first reproduce the official AdaTAD recipe and
compare against a matched native-source dense control under identical training,
EMA, evaluator/NMS and full decode-to-NMS cost, followed by disjoint multi-seed
confirmation and one sealed official-test opening.

## Connections

Relations are maintained only in `research-wiki/graph/edges.jsonl`.
