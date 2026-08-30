---
type: query_pack
updated: 2026-08-31
max_chars: 8000
---

# Research Query Pack

除下方首节外，本文其余部分均是按日期保存的历史证据；历史标题中的 “current” 或“当前”只描述当时状态，不覆盖首节的现时判断。

## Current paper question: BPNS-R1 (updated 2026-08-29)

- **Question.** Can current-frame contiguous native support remove substantial spatial
  VideoMAE work in offline TAD while preserving high-tIoU localization and reducing
  measured end-to-end cost?
- **Mechanism.** `ZoomToken-BPNS-R1` selects one strict contiguous `8x8/K64` native
  support before VideoMAE. Every retained token executes all 12 VideoMAE-S blocks and
  the existing Adapter. There is no hidden/KV cache, stable/changed gate, temporal
  carry, layer bypass or new loss.
- **Current evidence.** Same-source seed-42 final EMA is
  `68.51/61.19/46.27` for K100 and `69.07/61.14/46.57` for R1
  (Avg-mAP/mAP@0.6/mAP@0.7). R1 therefore removes 36% of native spatial inputs while
  matching or improving these three metrics in this one run. This supports an
  accuracy-feasibility claim, not a multi-seed or efficiency claim.
- **Strongest alternative explanation.** Native-token reduction may not lower full
  decode-to-NMS latency, energy or peak memory because decoding, transfers, Adapter,
  detector, postprocessing and sparse-execution overhead remain. The observed
  accuracy difference may also be seed-specific.
- **Terminal replay diagnosis.** The first two same-hardware K100/R1 replays are admission
  failures, not cost evidence. Job `1257281` exposed a raw-versus-rounded reference
  binding defect. Minimal candidate
  `e9323448f6cd78b99bb3de53fd9ffb55f3676d65` correctly implemented the subsequently
  frozen inclusive `0.05 pp` point-distance gate, but its sole formal replay, job
  `1258299`, stopped in the first R1 pass: unrounded `mAP@0.6=61.0869609029443100 pp`
  differs from reported-2dp `61.14 pp` by `0.0530390970556900 pp`. The eight passes
  did not complete, the result root is empty, and no cost or boundary artifact exists.
  The identity-gated v003 job `1258526` later completed all eight frozen prediction/
  evaluator passes, but terminated `FAILED_PROTOCOL_INVALID` in the profile phase because
  the short-action evaluator configuration omitted registry key `type`. It emitted no
  `profile.json`, cost rows, power trace, measured latency/energy/memory, short-action or
  boundary summary. This deterministic evaluator-construction omission is also not a
  model result or efficiency evidence.
- **Previous Pro adjudication.** The exact-Project post-result review classified v003 as
  engineering-defective, protocol-invalid and scientifically directionless, then returned
  `CONTINUE_ONCE_WITH_DECOUPLED_COST_CLOSURE`; the role contract remains `KEEP`. The exact
  raw diagnostic differences are mixed rather than uniformly positive: R1−K100 is
  `+0.5353 pp` Avg-mAP and `+0.7520/+0.1518/+1.6238/-0.1042/+0.2528 pp` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. This supports only fixed-seed accuracy feasibility; efficiency,
  memory, energy and boundary protection remain unknown.
- **v004 decision-changing result.** The sole v004 job `1260095`, candidate
  `a4694019fd4cbbdc74885e160163e23d947dc05f`, completed all eight frozen passes and
  emitted complete cost, power, prediction, evaluator, short-action, boundary, profile and
  terminal evidence. Independent pass-level recomputation gives R1/K100 ratios of
  `0.9849289616` for p50 and `0.9350002508` for gross energy. Energy passes the 5% gate,
  but p50 improves only `1.51%` and fails. Under the frozen conjunctive rule this is
  `STOP_BPNS_R1_EFFICIENCY_HEADLINE`: BPNS-R1 cannot be the current end-to-end efficiency
  headline. Peak allocated/reserved memory ratios are `0.7513/0.6897`, but these fixed
  single-hardware observations do not override the latency failure. No v005, replay,
  threshold change, seed or auxiliary arm is authorized.
- **v004 terminal Pro adjudication.** One fresh exact-Project `GPT-5.6 Pro` turn returned
  `PIVOT`, with engineering `PASS_STRONG`, protocol
  `VALID_WITH_DISCLOSED_POWER_UNCERTAINTY`, science
  `VALID_NEGATIVE_FOR_STANDALONE_FULL_STACK_LATENCY_HEADLINE`, and role contract `KEEP`.
  `STOP_BPNS_R1_EFFICIENCY_HEADLINE` is confirmed. BPNS-R1 is frozen only as an
  accuracy-feasible support primitive, local GPU/memory/energy attribution and a valid
  negative systems result. The 2.805 s power gap remains disclosed uncertainty; future
  tools must compute pass-local rather than cumulative gap statistics.
- **CPTC final route and TAR32 replacement evaluation (user-confirmed 2026-08-29).** The
  final route is `CPTC-vFinal-20260829`, which treats TAR32 as a context-preserved
  transformation-compression probe rather than state reuse: current dense token identity,
  full current K/V context and all-K64 Adapter execution remain, while only odd-block
  Query/output/MLP transformation updates are restricted to K32. Formal training job
  `1260166` completed with a valid epoch-59 EMA checkpoint, but emitted no official final
  validation. The sole evaluation-only submission, job `1261121`, stopped after four
  seconds in the external launcher's video-inventory gate: it incorrectly counted only
  top-level regular files, while the canonical inventory is 411 recursively nested MP4
  symbolic links. No model/checkpoint load, validation item, prediction, metric, training,
  resume or parameter update occurred. This is an engineering/protocol blocker, not a
  TAR32 result. A fresh exact-Project `GPT-5.6 Pro` turn returned
  `REVISE_AND_CONTINUE` with role contract `REVISE` and authorized exactly one
  `RPL1_EVALUATION_ONLY_COMPLETION`: scheduler ordinal 2, scientific-attempt ordinal 1,
  with no third submission. The one-line `find -L` correction passed Builder checks,
  independent Critic `PASS`, and result-blind Evaluator `PRE_RUN_READY_REPLACEMENT`.
  Replacement job `1261142` completed `0:0` on `g0067` after `00:14:58`. Its official
  unrounded Average-mAP/mAP@0.6/mAP@0.7 are `64.98114/57.37074/43.66910`; all three
  main admission thresholds fail. Reconstructed short-action mAP falls by `3.31759 pp`
  and fails its guard, while start/end normalized median-error ratios
  `1.09259/1.01788` pass. Identity, official prediction/evaluator and diagnostic evidence
  are complete, so the frozen classification is `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`,
  not a protocol blocker. Cost and the residual successor remain frozen pending exactly
  one fresh terminal Pro turn.
- **Parallel Pro-frozen composite probe (user-confirmed 2026-08-29).** The latest Pro
  instruction independently opens `ZoomToken-R1-TAR32-FKV`: retain R1 contiguous K64
  support and full-K64 Adapter, run dense K64 updates in even VideoMAE blocks, and in each
  odd block use the immediately preceding dense attention column mean to select exact K32
  per tubelet for Query/output/MLP while all K64 remain Key/Value context. It is a
  composition-first research probe, not an established method or novelty claim. Its exact
  base is `2d945e64bdccd09ae2e2916524562e3f388c5a2a`; the minimal clean/pushed candidate is
  `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7` on
  `codex/zoomtoken-r1-tar32-fkv-v001`. N16R4 focused checks report `32 passed, 1 skipped`
  plus `9 passed` for strict-R1 regressions; fresh Critic is `PASS`, fresh result-blind
  Evaluator is `PRE_RUN_READY`, and real-shape CUDA AMP pre-run job `1260163` completed
  `0:0` with the frozen `[64,32]x6` route ledger and zero fallback/failure. The only formal
  seed-42 training is job `1260166`, started on `g0059` with 2 GPUs. It was strictly
  isolated from v004. A Slurm terminal signal is available, but its model, checkpoint,
  route-ledger, accuracy and protocol evidence are intentionally not interpreted until the
  complete v004 package was adjudicated. The unique next task is now
  `ZT-CPTC-TAR32-TERMINAL-001`, superseding the old K100-labelled closure for this
  current arm. It first binds the original authority and audits job `1260166`; if a valid
  final model output later passes the frozen gate, it conditionally performs exactly one
  matched R1/FULL64 versus R1-TAR32-FKV full-stack cost job.
  No retry, resume, second seed or auxiliary arm is allowed.
- **Current execution state.** v004 is terminal `COMPLETED 0:0` and fully ingested. Its
  eight-pass order is `K100,R1,R1,K100,R1,K100,K100,R1`; all four prediction SHAs per
  arm match their frozen anchors, and 6,336 cost rows plus 929,889 power rows are present.
  R1−K100 accuracy deltas are mixed, while short-action/boundary changes are small and
  mixed, so no boundary-protection claim is supported. K100 pass 3 has a disclosed
  `2804.82 ms` in-pass power sampling gap; coverage remains complete and the frozen
  protocol had no gap threshold, so this is uncertainty rather than replay authority.
  The v004 post-result Pro turn is complete. TAR32 training identity, checkpoint and the
  single authorized replacement evaluation are now terminal and valid. No live or partial
  result was consumed. The valid negative result fails Average-mAP, mAP@0.6, mAP@0.7 and
  short-action guards; start/end boundary guards pass. It supports only the statement that
  this exact single-seed composition did not preserve frozen R1 accuracy. It provides no
  cost, novelty, multi-seed or general CPTC-mechanism conclusion. A third submission is
  forbidden, and fresh Pro adjudication is mandatory before any successor or cost work.
- **Frozen primary aggregation (instantiated by v004).** The profiler's top-level `comparison` pools all
  windows from four passes per arm. It is descriptive but does not implement the
  Pro-frozen primary estimate. Terminal analysis must group `cost_samples.jsonl` by
  `(arm, pass_index)`, compute decode-to-Soft-NMS p50 and total GPU joules for each
  complete pass, then take the median of the four pass estimates per arm before
  forming R1/K100 ratios. v004 provides the required complete raw rows. The reconstructed
  K100/R1 median p50 values are `2481.9575/2444.5518 ms`; median complete-pass energies
  are `144194.4432/134821.8406 J`. These produce the frozen ratios above. The pooled
  profiler summary remains descriptive and cannot replace this primary estimate.
- **Known terminal-evidence coverage deviations.** The current profiler persists one
  canonical prediction file per arm and checks repeated-pass prediction equality in
  memory; per-pass evaluator vectors live in `profile.pass_receipts`, rather than in
  eight separate prediction/vector files. The accuracy-reference contract records its
  source revision/path/symbol/SHA, but prediction-file hashes are not emitted.
  `cost_samples.jsonl` preserves the arm,
  pass, ordered dataset-item identity and cost rows needed to reconstruct population
  identity and pass-level estimates, but no separate population receipt is emitted.
  Missing or invalid power samples abort integration, yet the artifacts expose no
  quantitative coverage/gap summary and measure no temperature; observed order drift
  must not be called thermal drift. Final video-level NMS time and energy are amortized
  across window rows. The profile records Python/Torch/CUDA summaries, but not the full
  command, annotation/class-map/video/checkpoint paths, full package lock, or an explicit
  success-side anomaly list; those identities are reconstructible only from the start
  receipt, launcher and Slurm logs and must be labelled as external evidence. Do not
  synthesize absent files or metadata after the run. At
  terminal state, disclose produced, reconstructible and unmeasured evidence separately
  and ask the fresh Pro review to judge whether it is claim-valid, diagnostic-only, or
  requires a new bounded measurement.

## Full-representation temporal reuse adjudication (2026-08-26)

- A fresh exact-Project Pro review at revision `bffff43dad28ca1042602ad3a01ba2990b953c13`
  returned `STOP` for the narrowly defined route that reuses the preceding temporal
  unit's complete VideoMAE representation while recomputing only changed tokens.
- The decisive structural fact is that a 16-frame clip is an attention bucket but not
  an independent full-backbone state: every VideoMAE block is followed by an AdaTAD
  Adapter operating on the global 384-tubelet lineage. A block-11-only cache cannot
  supply changed tokens with valid per-layer context; a per-layer cache becomes an
  approximate event-driven network whose fixed pre-backbone mask is not closed under
  attention and Adapter propagation.
- Eventful Transformers and CVPR 2026 STC-Cacher already cover the broad gate/cache/
  stable-token reuse/dynamic-token recompute operation family. This is a material
  novelty warning, not empirical proof that all temporal reuse is impossible.
- Independent code audit supports the state-dependency argument. It qualifies the
  systems claim: the Transformer attention buckets admit 48-clip batching, while the
  Adapter already couples clips, so the exact latency loss from sequential cache
  execution remains a measurement question rather than a proven number.
- Project disposition: `STOP_BEFORE_IMPLEMENTATION` for this exact full-representation
  route; zero new config, Builder patch, PRE_RUN or 60-epoch cell. No new accuracy or
  efficiency evidence was produced. Reopening requires a genuinely different state
  dependency contract, an operation-level comparison against Eventful/STC, and a
  credible full-stack execution lower bound.

## R-PADT-v0 user-report intake (2026-08-26)

- The user supplied a complete external `R-PADT-v0` report and a pasted rendering.
  They are scientifically coherent, but the project receipt for the referenced Pro
  conversation remains `TERMINAL_INCOMPLETE_NO_SCIENTIFIC_DECISION`; nonce and
  attachment claims also conflict. Treat the report as a user-provided candidate,
  not an accepted same-session Pro verdict or execution authority.
- R-PADT is actually prefix-conditioned suffix token compression: two dense
  VideoMAE blocks, periodic K64 anchors, non-anchor delta top-16 plus four summaries,
  a shorter dense suffix, then anchor-copy restoration. It is not direct reuse of
  the preceding frame's complete representation, exact KV cache or dense-equivalent
  temporal reuse.
- Fixed-revision audit confirms T=384, K64 on a 10x10 native grid, D=384 and 12
  blocks, but all 12 blocks also carry an Adapter and the standard Adapter expects a
  full temporal lattice. R1 independently chooses one of nine 8x8 blocks per
  tubelet, so consecutive K64 sets are not guaranteed to have the report's required
  one-to-one anchor mapping. Direct anchor-copy also has stale temporal-identity
  risk, and Q=4 summaries confound selection, transport and aggregation.
- Prior-art audit adds STA and PVC as omitted close neighbors. Eventful/ToMe are
  correctly recognized in principle, but the pasted references are materially
  misassigned and `VideoZip` is unverified. Novelty can only be narrow TAD/ROI-grid/
  suffix-compression/restoration integration.
- Project verdict is `PARTIAL_ACCEPT_REVISE_BEFORE_G4`. Status remains `discussed`:
  no Builder, PRE_RUN, experiment or paper claim. See
  `sources/2026-08-26-r-padt-v0-user-report-intake-audit.md`.

## Dynamic selective recompute / light-update adjudication (2026-08-25)

- A fresh, attachment-based exact-Project Pro review compared three complete
  candidates: clip-internal dynamic recompute plus rank-32 current update
  (`IC-DRU`), exact overlap-window dependency-cone caching (`OW-ECR`), and a
  current-proxy nested-depth route (`PCD-DRU`). The top-level verdict was
  `STOP_BEFORE_IMPLEMENTATION` for these exact candidates.
- Under an optimistic 25% refresh assumption, known-backbone arithmetic ceilings
  were about `50.12%` saving for IC-DRU and `60.16%` for PCD-DRU. These are not
  measured speedups: first-tubelet/full-refresh fallbacks, bucket fragmentation,
  dense Adapter, decode/H2D/detector/NMS and peak-memory behavior remain.
- OW-ECR has only about `7.66%` known-backbone saving in the most favorable exact
  dependency-cone bound, while twelve cached overlap layers add roughly
  `54 MiB/sample`; it lacks credible full-stack headroom.
- Pro judged IC-DRU/PCD-DRU reducible to known change routing, video feature cache,
  stable-token residual and MoD/A-MoD depth-allocation components. Therefore no
  Builder, PRE_RUN or 60-epoch cell is opened, and no seed/K/rank/threshold or
  teacher/distillation rescue is allowed for these three designs.
- This is pre-execution design evidence, not an empirical failure of dynamic
  20–30% refresh or all temporal reuse. Any future route needs a distinct error-
  control/execution principle, legal bidirectional VideoMAE semantics and a
  conservative full-stack saving margin before implementation.

## Post-APM ACR16/Eventful adjudication (2026-08-25)

- After the closest-prior-art correction and conservative arithmetic bound, a
  fresh exact-Project Pro review returned `STOP` for `R1-ACR16-Delta1-FKV`.
  Eventful Transformers (ICCV 2023) already covers token references/buffers,
  temporal change selection, gather/scatter identity restoration, and sparse or
  incremental Transformer updates.
- ACR16 does not reuse old hidden states, Q/K/V, attention products or MLP
  outputs. It recomputes current full-K64 K/V, lets stable tokens bypass selected
  middle-depth residual branches, and applies one low-rank input-delta residual.
  Its remaining distinction is therefore an application combination of
  Eventful-style evidence and MoD-style conditional depth skipping, not a new
  temporal-reuse principle.
- Verified arithmetic gives a `9.446%` maximum saving over the 12 VideoMAE main
  blocks. After current full K/V, dense Adapter, patch embedding, matching and
  other known backbone arithmetic, the upper bound is about `8.80%`; it lacks a
  credible margin for selector-inclusive decode-to-Soft-NMS p50 latency and gross
  energy to both improve by at least `5%`.
- Status is terminal `STOP_BEFORE_IMPLEMENTATION`: zero new training cells, no
  Builder, no PRE_RUN and no accuracy/efficiency claim. This stops only the
  ACR16/Eventful-transfer G4 route, not all temporal-redundancy research. A future
  route must introduce a distinct mechanism and show a conservative full-stack
  saving margin before implementation; do not tune or rescue ACR16.

## Strict A-MoD reference and temporal-memory continuation (2026-08-24)

- Strict VideoMAE A-MoD reference is implemented and clean/pushed at
  `a41714e9f9271906a2eb4505e3fedc590c838055` on
  `codex/zoomtoken-amod-v001`. Exactly five existing paths changed: backbone,
  official-derived config, N16R4 launcher, training entry and focused test.
- Blocks `0/2/4/6/8/10` are dense. Blocks `1/3/5/7/9/11` use the immediately
  preceding dense block's attention-probability column mean, stable top-400 of
  800 tokens, selected Attention+MLP update and identity bypass for unselected
  tokens. The existing Adapter remains dense on all 800 tokens. There are no
  router parameters, extra loss, hidden-state cache or cross-frame carry.
- N16R4 no-data verification is `8 passed`; independent Critic verdict is
  `AUDIT_PASS`. This is implementation/review evidence only: no A-MoD formal
  training, accuracy, latency or energy result exists yet.
- Test-only clean/pushed successor `31e4b1e61a23c4f1b319249684c8f05da6734235`
  closes both nonblocking Critic coverage notes. It directly proves every sparse
  block receives its immediately preceding dense block's score and every Adapter
  receives `[1,800,C]` in the official token geometry. N16R4 CPU-only suite is now
  `10 passed`; no model/config/launcher/scientific behavior changed.
- Cross-frame feature storage and mapping remains an active primary direction
  and is not stopped by the strict A-MoD reference. The fresh Project Pro review
  selected `APM32-CTX64`: strict R1 K64 support, one-tubelet detached pre-position
  patch memory, deterministic Chebyshev-radius-2 mutual-nearest alignment at
  similarity `>=0.80`, K32 refresh/K64 K/V context, and exact K64 fallback when
  fewer than 32 matches are valid. `CUR32-CTX64` uses the identical mask and
  fallback but retains the current embedding, isolating memory substitution.
- The clean/pushed executable successor is
  `e92df6a4737a10955722c6aedc2f079e0d285a18` on
  `codex/zoomtoken-apm32-ctx64-v001`. It preserves the frozen model and adds only
  a result-blind one-production-batch preflight plus full model/EMA/optimizer/
  scheduler/scaler/counter/sampler/RNG recovery verification; temporal memory is
  forbidden from serialization. N16R4 CPU-only result-free tests are `19 passed`,
  and a fresh independent Critic returned `AUDIT_PASS`. Result-blind PRE_RUN is
  `NOT_READY` only because APM and CUR have not yet executed their actual two-GPU
  single-batch witnesses. No formal APM/CUR job or performance/cost result exists.
- The terminal DSR6 job `1252527` is independent. It must not be rerun, resumed
  into this route, or reinterpreted as temporal-memory evidence. It completed
  `0:0`; the immutable terminal log gives epoch-59 EMA
  Avg-mAP/mAP@0.6/mAP@0.7 `67.38/59.34/46.01`, so all three frozen near-lossless
  thresholds fail. The scientific interpretation was revised on 2026-08-25:
  at `79.055%` VideoMAE-block proxy cost it remains a conservative Pareto
  candidate for matched end-to-end cost measurement, not an accuracy-preserving
  result or demonstrated speedup.
- Read-only implementation mapping is complete. Native lineage already exists as
  `(tubelet_index, spatial_index)` in the packed/ragged Adapter path; THUMOS samples
  already carry `video_name` and `window_start_frame`. ChronoTransport provides a
  reusable per-stream cache container, detach policy, state age, first-chunk dense
  repair, non-finite fallback and current-state delta/cosine signals. These are
  primitives only: they do not provide correspondence, occlusion/scene-cut
  invalidation or cross-window identity. A valid implementation must never key a
  persistent cache by batch position alone; it must reset on video change,
  unexpected window order or failed alignment. Live activations/cache tensors are
  not checkpoint artifacts; epoch-boundary resume should clear memory and force a
  warm-start recomputation unless Pro explicitly freezes a different protocol.

## DSR6-KV implementation and PRE_RUN state (2026-08-24)

- Fresh Project Pro selected exactly one revised depth route: strict R1 K64 support,
  FULL64 updates in blocks 0–5, and one fixed per-tubelet K32 query/output/MLP set in
  blocks 6–11 with all K64 retained as non-detached K/V context. Existing Adapter
  computation remains on all K64. No hidden carry/cache, shallow transport, new module,
  loss, second split, K24/K18 or extra seed is allowed.
- The scientific implementation is rooted at `3260cd39154069138c6b1757326372cc3b73754e`.
  Its launcher-only final clean successor is
  `c6327a891809aa30370b3b2d9bedab0dcfe0d326` on
  `codex/zoomtoken-dsr6-launcher-profile-v001`. Relative to `3260cd39…`, only the
  N16R4 launcher profile boundary and its focused regression changed; the final
  `c6327a89…` step itself adds only explicit probe success termination. The exact
  N16R4 suite is `12 passed`; fresh independent Critic verdict is `AUDIT_PASS`.
- Result-blind PRE_RUN is renewed as `READY`. A real 2-GPU Slurm job-shell witness
  `1252525` completed `0:0` and proved the exact `/etc/profile` boundary succeeds
  with `LC_BYOBU`/`XDG_DATA_DIRS` unset and restores nounset before later launcher
  work. It accessed no training data/model/result. Canonical 411 MP4/0 broken links,
  annotation, class map, VideoMAE-S pretrain, OpenTAD environment, exact clean ref,
  capacity, and the absence of proposed root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_c6327a89_seed42_20260824`
  and job `zt-dsr6-kv-s42-c6327a89` were reconfirmed. Old job `1252521` and its root
  remain sealed pre-data infrastructure failure and must not be resumed/requeued.
  PRE_RUN witness `1252525` had consumed the initially proposed job name, so the first
  formal dispatch stopped before `sbatch` with attempt count zero. Central then froze the
  distinct formal name `zt-dsr6-train-s42-c6327a89`; exactly one formal job `1252527`
  was submitted at `2026-08-24T04:46:31+08:00` under the same root and unchanged tuple.
  It started on `g0041` at `04:46:36` and completed `0:0` at `10:53:53` after
  `06:07:17`. The epoch-59 checkpoint and retained recoveries 44/49/54 exist, with
  no Traceback, OOM or non-finite loss. Although no standalone metric JSON was
  produced, the terminal stdout is auditable and its final evaluation runs after
  epoch 59 through the EMA-loading evaluator path. Final-EMA is
  `67.38/59.34/46.01`, below the frozen `68.57/60.64/46.07` all-of gate by
  `1.19/1.30/0.06` points. The decision is `STOP_DEPTH_ROUTE`: no cost, extra
  seed or structural rescue. The `79.055%` block-FLOPs proxy is not measured
  latency or energy.

## Terminal RC32-KV seed-42 decision (2026-08-24)

- Clean revision `813012620dca991ff90121d0d9faf688f303d1ef` completed the full
  DROP32/MOD32-KV/RC32-KV 60-epoch matrix as jobs `1252179/1252180/1252181` under
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_81301262_seed42_20260823T2100`.
  All three are `COMPLETED 0:0`; each has `checkpoint/epoch_59.pth`, and logs are
  clear of Traceback, OOM and non-finite errors. FULL64 remains read-only job
  `1249099`; no baseline was duplicated.
- Final-EMA Avg-mAP/mAP@0.6/mAP@0.7 are FULL64 `69.07/61.14/46.57`, DROP32
  `66.11/57.83/44.88`, MOD32-KV `66.50/59.24/45.21`, and RC32-KV
  `64.73/57.34/42.91`. RC32-KV fails every frozen D−A/D−B/D−C accuracy gate;
  MOD32-KV also exceeds the original near-lossless allowance against FULL64.
  RC32 temporal carry is stopped because it is strictly dominated by MOD32-KV
  at identical proxy cost. MOD32-KV and DROP32 remain candidate Pareto points.
- The final compute–accuracy figure uses only the declared VideoMAE-block matmul
  proxy: DROP32/MOD32-KV/RC32-KV are `49.32%/58.11%/58.11%` of FULL64. Under the
  user-confirmed accuracy–efficiency objective, DROP32, MOD32-KV and DSR6-KV
  represent aggressive, medium and conservative proxy-compute operating points.
  No new training or rescue is justified; the next discriminating evidence is a
  matched FULL64/DSR6/MOD32/DROP32 end-to-end latency, memory and energy replay.
  The proxy alone is not a real speed or energy result.
- The offline boundary/final-prediction tooling remains implemented in clean
  descendant `4e940b780da5a3cd0ea28ca420c5d1cb879818b5`, but the terminal decision does
  not depend on additional GPU evaluation: the primary accuracy gate already
  fails by large margins. Preserve R1/FULL64 K64 as the last accuracy-supported
  spatial route; do not revive RC32-KV without a new scientific decision.

## Current strict-rectangle R1 experiment (2026-08-22)

- The current paper question is pre-backbone spatial recomputation reduction in
  end-to-end TAD. Existing seed-42 A/B/C validation is `68.73/47.24`,
  `68.51/46.27`, `68.22/45.35` (Avg-mAP/mAP@0.7). C really gathers 64/100
  native tubelets before VideoMAE, but its support is ellipse/Gaussian Top-64,
  not a complete rectangle.
- Accepted R1 changes only support topology: one of nine complete `8x8` blocks
  on the `10x10` native grid, K64, one true-ragged heavy forward, no padding,
  same sparse adapter and official recipe. Clean revision is
  `9e25c6d38de8c993948025629181470b858682b4`; independent Critic and target
  runtime PRE_RUN passed.
- R1 seed-42 60-epoch job is `1249099`, run root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_official_prebackbone_r1_9e25c6d3_seed42_20260822T080108Z`.
  It completed `0:0` after 60 epochs. The terminal EMA validation is
  `69.07/61.14/46.57` (Avg-mAP/mAP@0.6/mAP@0.7), which is
  `+0.85/+0.13/+1.22` points relative to C and passes all three preregistered R1
  accuracy conditions. This establishes seed-42 accuracy support for complete
  rectangular support, not yet an efficiency or multi-seed claim. All eight jobs
  `1249125–1249132` are terminal `COMPLETED 0:0`. R2/R2-SHUF48/Q48-GLOBAL final
  Avg-mAP/mAP@0.6/mAP@0.7 are `66.56/59.06/45.17`, `66.17/58.53/44.47`, and
  `65.78/58.62/44.74`; R3/R3-AREA-SHIFT are `67.88/60.32/46.41` and
  `67.50/60.26/45.09`; R4/R4-SHUF15/Q64-GLOBAL are respectively
  `68.02/60.32/46.26`, `67.19/60.17/46.20`, and `67.84/60.66/45.39`.
  Thus R4−R4-SHUF15 is `+0.83/+0.15/+0.06`: the preregistered high-tIoU
  ordering margin of `+0.30` is not met, so frame-outside content ordering is
  not established. R4−Q64-GLOBAL is `+0.18/-0.34/+0.87`, a crossed result.
  No strict-rectangle cost or multi-seed result exists.

## Terminal ROI60 result (2026-08-21)

- The final ROI-only G implementation is clean revision
  `59960255a708c0341baa8104a1d4e120f87435e3`. It inherits the reviewed
  `GeoRouteSourceViews` restoration and adds only deterministic optimizer/DDP
  execution corrections needed for the existing ActionFormer production path.
  The independent Critic passed that exact path; model science is unchanged.
- The admitted real THUMOS14 seed-3407 60-epoch runs are terminal: matched-source
  full-compute DN `1245907` gives Avg-mAP/mAP@0.7 `64.73/43.26`; ROI-only G
  `1245924`, residual off, gives `61.49/39.99`. G is lower than DN by `3.24/3.27`
  points and is a method-level negative result for this configuration.
- Shared untouched AdaTAD reproduction `1245842` gives `68.73/47.24`, close to
  but below the published `69.03/48.27` anchor. It is separate from DN.
- Official AdaTAD evaluated every two epochs over the final 20 epochs. DN/G were
  deliberately final-only (`val_eval_interval=60`, `val_start_epoch=59`), so no
  intermediate curve was emitted during training. Validation-only evaluation of
  epoch 44/49/54 EMA recovery checkpoints gives DN Avg-mAP
  `65.50/65.06/64.84` and G `62.42/62.00/61.80`; the stable
  `-3.08/-3.05/-3.04` gap shows the final negative result is not a last-epoch
  fluctuation. No run contains a complete end-to-end cost result; do not infer
  efficiency from token budget.
- Earlier `1245897/1245898` and G-only `1245908/1245909/1245910` ended before an
  admissible result because of optimizer-group aliasing or DDP auxiliary-loss
  graph ownership. Preserve them as implementation diagnostics; do not use them
  as model evidence or resume them.

## Baseline-first correction (2026-08-17)

- Current ZoomToken work must not call matched-source dense seeds
  `3407/3408/3409 = 66.42/67.14/65.99 Avg-mAP` an exact official AdaTAD
  reproduction. The published upstream AdaTAD anchor is Avg-mAP `69.03` and
  mAP@0.7 `48.27`; the released-checkpoint evaluation has not been performed.
- First read [WIKI_MEMORY_AUDIT-2026-08-17.md](WIKI_MEMORY_AUDIT-2026-08-17.md).
  It supersedes this file's old “current” pointer only; all historical
  anti-repetition constraints below remain preserved.

## Shared official AdaTAD baseline rule (2026-08-17)

- ZoomToken alone may execute one exact released-checkpoint evaluation, then—only if genuinely
  necessary—one clean untouched official reproduction. Every related TAD project consumes the
  final durable receipt read-only; do not duplicate it or label 66.xx matched-source dense as it.
- This shared-number gate does not stop accepted local method preparation: keep Q-matrix entry,
  conditional ROI/residual controls, recovery checkpoints, review and PRE_RUN work moving while
  treating the dense official value as `UNBOUND_SHARED_INPUT`.

## Current decision: Hybrid-centered causal pilot (2026-08-02)

- Accept the Pro review's central verdict `RUN_HYBRID_CAUSAL_PILOT_FIRST`, with
  explicit evidence corrections. The old Hybrid result is strong descriptive
  motivation, not proof of ROI/residual complementarity. The old Free-first
  selector remains closed and is not rerun.
- Current candidate: support-only Structured Complementary Native Routing
  (SCNR-TAD; GeoRoute code namespace). The running exact-K64
  context8+ROI28+residual28 study is a causal mechanism probe, not the final
  architecture. The user-restated final objective is a temporally adaptive
  continuous `(cx,cy,w,h)` ROI together with dynamic total heavy-token budget
  `K_t` at the native 384-step, two-frame VideoMAE tubelet granularity; the user
  explicitly confirmed both this temporal unit and that changing only the
  ROI/residual split under fixed K is insufficient. The role split must therefore be dynamic too.
  Budgeting is staged: first require a configurable hard per-window budget
  `sum_t K_t=B` while learning the 384-way redistribution; only a passing first
  stage may introduce content-dependent window-level B under a separately
  frozen cost constraint.
  All three role counts are dynamic, including context: there is no fixed
  context floor and no hidden uniform scaffold.
  Stage 1 permits `K_t=0`; a matched `K_t>=1` setting is required as an
  explicit ablation rather than a hidden runtime repair. The main zero-budget
  representation is a masked zero carrier: zero heavy feature at that tubelet
  plus an explicit heavy-valid mask, with no content-bearing substitute and no
  executed-heavy-K charge. `learned-null` and `scout-projection` are separately
  trained carrier ablations, never inference-time switches on the main
  checkpoint.
  The user approved the Stage-1 allocator family: one global constrained
  exact-`B` projection over all physical `(tubelet,patch)` candidates, with at
  most one dynamic context/ROI/residual role per physical token. `K_t` and all
  role counts are induced by the selected set rather than predicted as fixed
  quotas or repaired afterward. The Stage-1 ragged execution/cost boundary is
  now designed: exact `B` means exactly `B` unique physical tokens are
  patch-embedded and heavy-executed over the full window, not that every route
  has equal FLOPs. Full-window temporal adaptivity is retained; no fixed
  16-frame-clip quota may be imposed as the final main method. For native clip
  `c`, record its executed-token count `b_c` and attention-pair ledger
  `P=sum_c b_c^2`, plus actual patch-embedding, attention, MLP and
  coordinate-lineage Adapter calls and end-to-end p50/p95. Advancement requires
  a measured cost/Pareto gate. The current `[B,T,K]` packed path requires
  constant K and cannot be relabeled as this execution through dummy/padded
  tokens; role-utility and estimator details remain under design review.
  The user approved the Uni-AdaFocus-informed role/estimator revision as
  `designed`. Keep the cheap full-window scout feature as policy context, remove
  the independently parameterized `q_ctx` head, and use one shared physical-
  token base utility with ROI/residual modifiers. Hard routing uses
  `u_hard=q_base+max(0,delta_roi,delta_res)` and the winning modifier defines the
  operational context/ROI/residual role; context is the zero-modifier outcome.
  Backward uses a temperature-controlled log-sum-exp relaxation while forward
  remains the approved unique global exact-B physical top-B. The main estimator
  family is a Uni-inspired stop-gradient coarse-feature surrogate plus ST; PL is
  a separately trained matched ablation whose fixed-pilot recovery may inform
  analysis but cannot choose the dynamic main method. Uni-AdaFocus's
  classification proxy, fixed focus count, resized crop, full-frame size
  regularizer and early-exit budget are not transplanted. The user also approved
  the exact training boundary as `designed`: the cheap scout feature `Z` is
  trained jointly by a train-only auxiliary TAD head, while the route policy
  consumes `stopgrad(Z)`. The detector loss is active for every successful
  optimizer step on the exact hard top-B/ragged path and reaches route heads only
  through selected-token ST. A backward-only global soft-budget projection has
  `0<p<1` and `sum p=B`; it aggregates detached scout features to provide dense
  counterfactual TAD supervision, updates the policy and auxiliary head but not
  the scout or heavy backbone, never enters inference, and never counts as
  executed B. Its weight is enabled early and annealed to zero by optimizer step
  before the final hard-only phase. The main setting adds no area, coverage,
  expected-cost or fixed-context loss; PL remains a separately trained ablation.
  The formula-level re-audit is now `designed`. Uni-AdaFocus's
  paper describes centre/height/width geometry, while official commit `88464883`
  actually emits four sigmoid actions and maps source sizes to `96..224` pixels
  on a 224-pixel input, then maps normalized top-left fractions into the
  remaining in-bounds interval. In normalized coordinates this is algebraically
  the same bounded-interval family as the proposed
  `w=w_min+(1-w_min)sigmoid(a_w)` and
  `c_x=w/2+(1-w)sigmoid(a_x)`; the material differences are the minimum size,
  native-token membership instead of resized crops, and loss semantics. Uni adds
  Eq. 15's full-frame-seeking size regularizer (implemented as a penalty toward
  size action one) because its feature-interpolation classification proxy shrinks
  deformable crops. The proposed `w_min=1/W_grid`, `h_min=1/H_grid` and omission
  of that regularizer are now approved for the main route: the floor is derived
  independently at runtime as `(1/W_grid,1/H_grid)`, with no full-frame, area,
  coverage or smoothness penalty. A separately trained matched 2x2-cell floor is
  the frozen sensitivity ablation. The shared geometry primitive implements the
  new native-cell mode while preserving static normalized behavior for historical
  configs. Exact clean source `4be71844` passed focused geometry/routing `36/36`
  and complete GeoRoute plus required C3 regressions `194 passed, 1 skipped` on
  N16R4 Linux/Torch. The complete opt-in dynamic Stage-1 path is now also
  implementation-level `tested`: global unique exact-B, fully dynamic `K_t`,
  masked-zero, true no-padding ragged VideoMAE/Adapter, support-only ST and the
  backward-only exact-sum proxy are executable. The initial P0 Job `1215355`
  diagnosed a full-extent/semi-axis mismatch that collapsed ROI attribution;
  corrected source `dfcbe692` uses `w/2,h/2` ellipse semi-axes and G1 P0 Job
  `1215358` sealed `PASS_NO_PERFORMANCE_P0`. Fit-only health Job `1215363` at
  exact source `7cf589f0` completed 64 successful updates with all nine required
  gradient groups nonzero on `64/64`, two bounded AMP replays, dynamic
  `K_t=17..218`, exact B, and zero performance artifacts. Matched G2 source
  `8aa8e2a3` changes only the ROI floor to two native cells; P0 Job `1215364`
  also passed. These are mechanical/health receipts only: no mAP, measured
  floor comparison, Hybrid efficacy, complementarity, efficiency or paper claim
  has been established. Exact clean source `7e5775e8` now emits fail-closed,
  sample-level dynamic ROI/`K_t`/role/ragged telemetry and passed the relevant
  N16R4 Linux suite `35 passed`; telemetry CPU copies are excluded from timed
  cost replay. Exact implementation commit
  `ec8de9f51f85fc81031d82b79e30019d57a381b4` now freezes the single-seed M2
  runner: matched 60-epoch G1/G2 training and complete accuracy/telemetry replay,
  followed by one same-GPU `G1 -> G2 -> G2 -> G1` counterbalanced full-stack
  decode/H2D/model/postprocess/NMS latency, peak-memory and gross-energy replay,
  then an all-terminal fail-closed finalizer. Local compile/Bash/whitespace and
  focused contract checks passed (`29 passed`); exact clean remote runtime
  `9d6641a6` passed the expanded Linux/Torch suite (`76 passed`) and all four
  deployment prechecks. Its first formal submission stopped before any Job was
  created because N16R4 rejected the CPU-only finalizer reservation. Failed root
  `scnr_dynamic_floor_m2_9d6641a6_s3407_20260804_0507` contains only the storage
  preflight and is not reusable. Exact infrastructure fix
  `bad14693daa1fe414e56bf697c617e76f96eed48` gives the finalizer one GPU/one CPU
  solely as disclosed scheduling overhead; local focused checks pass `13/13`.
  Replacement runtime `6ee97336775a09611f10423e07cafcea375e191a` passed the
  same remote `76/76` suite and all four fresh prechecks, then atomically deployed
  G1/G2/cost/finalizer Jobs `1216180/1216181/1216182/1216183` under
  `scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525` (deployment self-hash
  `a0504e45179957f20580b901e6ef7723d63c7b0ed445d8b3c35c3b5aaa02b89a`).
  Both 60-epoch arms completed `0:0` and sealed valid stage results, but paired
  cost Job `1216182` failed on the first timed audit because the profiler read the
  nonexistent legacy key `packed.attention_pairs` instead of the executor's
  receipted `packed.attention_pairs_per_window`. Finalizer `1216183` correctly
  sealed `INCOMPLETE_NO_FLOOR_INFERENCE`, with no contrasts, floor selection,
  official-test opening, or paper claim. A minimal recovery reads and validates
  the current per-window ragged ledger, records the unchanged model/runtime
  commit separately from its clean execution-repair commit, preserves both
  completed arms, and must rerun the entire counterbalanced paired-cost pass plus
  finalizer before any metric is interpreted. First recovery execution `c67e13e8`
  passed remote `50/50`, but cost Job `1222672` failed because dynamic SCNR calls
  `sparse_adapter.forward_ragged` directly and therefore bypassed the profiler's
  module-forward hook; finalizer `1222673` again sealed incomplete with every
  promotion guard false. Exact execution-only repair
  `6341927f099bd59e0be6aff9b4b1062b4f76150e` instruments the real ragged method,
  names invalid stages, passes local `16/16`, remote `51/51` and cost precheck,
  and changes no model/config. After revalidating unchanged arm states/hashes,
  cost Job `1222700` and finalizer `1222701` were immutably receipted and released
  without retraining. Cost completed all four timed passes but failed only at final
  profile validation: the producer forced `post_processing.sliding_window=True`
  before hashing its cost config while the validator reconstructed the config
  without that mutation, so `pass_receipt.cost_config_sha256` mismatched.
  Finalizer again sealed incomplete. Its four raw cost files are preserved under
  `cost_failed_job1222700/`, but cannot be manually assembled because the required
  in-memory pass receipts and complete profile-level provenance were never
  published. Exact execution-only repair
  `011d2943c698bb8a3727de9163034a7153779b64` makes producer, validator and tests
  call one shared cost-config builder; it changes no model/config/training code and
  passes local `16/16`, remote `51/51`, and cost precheck. After revalidating the
  unchanged arms, replacement cost/finalizer Jobs `1222869/1222870` were submitted
  held, bound to recovery-v3 receipt self/file SHA-256
  `4cc8b7649d82cfd89530453df6be02609af5a30334f0f9f44a3efa447bf584e2` /
  `623c2f958d0a86fdba5ffd81f14c413e652c4d788b2fc089151b48bbf9ce81fe`,
  validated with the original finalizer and released. Job `1222869` then failed
  before creating `cost/`: one population-preflight call site still referenced the
  removed `_cost_config` helper and raised `NameError`; finalizer `1222870` sealed
  incomplete with every promotion guard false. Exact repair
  `42923d9f7aaddb14368f82aacda5c77e1f857a24` changes that call to the shared
  builder and adds an AST regression that rejects a loaded legacy name. It passes
  local `16/16`, remote `51/51` and cost precheck. Replacement cost/finalizer Jobs
  `1222889/1222890` were held, bound to recovery-v4 receipt self/file SHA-256
  `5fe63bce1811abddadb5dda60bc67385b07693f7642c4de016616cd8756c1e1c` /
  `5bd504a60668eeb204035d25e4853d601c67fc5097b474987e718562c7b51226`,
  validated and released. Cost `1222889` completed `0:0` and published a profile
  that passes the `42923d9f` validator, but finalizer `1222890` ran from frozen
  runtime source `6ee97336`; that old validator reconstructed pre-repair configs
  without `sliding_window=True`, rejected the otherwise valid pass receipts, and
  sealed incomplete. Exact finalizer repair
  `75e2adc86877f002e10626ee4011104b60b0ce49` binds model runtime, cost execution
  and finalizer execution separately. Remote `52/52`, finalizer precheck and a
  no-number existing-profile/descriptive-finalization dry run pass. Finalizer-only
  Job `1223310` reused the immutable profile without cost replay or retraining and
  was bound to deployment self/file SHA-256
  `8fe36543b2e0b4f74f9c5fbdb77100e204f46e81c3e72385c5c62de2e08b9f0c` /
  `1eeff523c3a676701e4a688aff0a3eb3ee95718653fb6eb15c5ae1c82e87f1ee`.
  It completed `0:0` and sealed
  `PASS_COMPLETE_DESCRIPTIVE_FLOOR_SENSITIVITY /
  COMPLETE_DESCRIPTIVE_ONLY_M3_REQUIRED_FOR_FLOOR_SELECTION` with no errors.
  Seed-3407 G1 minus G2 is `+5.78 pp` Avg-mAP and `+6.22 pp` high-IoU composite;
  model-forward p50 differs by only `+0.438%`, while aggregate end-to-end p50 is
  `+2.845%` and is dominated by a cold first G1 host/input pass. This is complete
  descriptive M2 evidence, not a floor selection.
  The decisive M2 mechanism finding is selected-role collapse: G1
  context/ROI/residual `0/7/3,342,329`; G2 `0/0/3,342,336`. Both floor-saturation
  rates are zero, so neither operational Hybrid complementarity nor floor
  causality is established. Replay integrity is now resolved: same-GPU OFF/ON
  proves observer neutrality; legacy OFF-A/OFF-B/ON exposes baseline CUDA replay
  drift after routing; strict math-SDPA makes OFF-A=OFF-B=ON byte-identical for
  both arms. Strict backend differs from historical source, so continuous scores,
  margins, geometry, predictions and performance remain closed. A field-minimized
  categorical bridge validates exact hard-role equality for all 136 windows.
  Authorized all-valid counts are G1 `0/2,671/11,486,609` and G2
  `0/984/11,488,296`; residual dominates `136/136` windows before global top-B.
  Thus branch-offset identifiability, not top-B squeezing diverse roles, is the
  first repair target. Exact source `091f9f9b` subtracts only the all-valid
  full-window mean of `delta_residual` and leaves Scheme A/exact B/dynamic K/
  ragged/masked-zero/ROI/context unchanged. Clean N16R4 regression passed
  `90/90` GeoRoute plus `20/20` required C3 tests. Frozen-checkpoint Jobs
  `1223783/1223784` then passed strict duplicate prediction/route parity and the
  preregistered structural gate in both arms. Selected context/ROI/residual
  counts are G1 `168,733/421,121/2,752,482` and G2
  `186,976/429,896/2,725,464`; maximum post-centering residual-mean error is
  `3.04e-7`. This proves structural role reachability only and authorizes
  freezing a matched development `none` versus centering training protocol. It
  does not prove accuracy, operational complementarity, cost or floor effects;
  M3 and official test remain held. No Pro discussion is needed for the matched
  single-variable protocol.
  That protocol is now terminal as
  `exp:scnr-residual-centering-matched-training-v1` at stage `tested`.
  It uses two fresh G1 `native_1cell_main` cells, `none_control` and
  `residual_window_center`, with the same pretrained initialization, seed 3407,
  data/order, exact B, fully dynamic K/roles, ragged/masked-zero execution,
  60 epochs and exactly 9,600 successful updates. Old M2 checkpoints are not
  reused. Each final EMA receives same-GPU strict math-SDPA duplicate Gate
  replay; any lineage/prediction/route/population/invariant failure leaves empty
  contrasts. The single-seed accuracy screen requires centered mAP@0.6 and
  mAP@0.7 both strictly higher and Avg-mAP non-lower. PASS authorizes only a new
  ABBA+BAAB paired full-stack cost design; disjoint seeds, M3, official test and
  every paper/efficiency/complementarity claim remain held. The implementation
  hashes the normalized complete training recipe across cells, atomically binds
  both held stage Jobs plus the `afterany` finalizer, and fails closed on any
  malformed or incomplete artifact. Local pure-contract/inherited-M2/required-C3
  regression was `57/57`. Exact runtime `16137484c5cc` passed clean N16R4
  Torch/CUDA regression `93/93`; deployment SHA `71b10681...` atomically bound
  fresh control/center Jobs `1223819/1223820` plus after-any finalizer `1223821`
  under `scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`.
  All three Jobs completed `0:0`. Both cells completed 60 epochs/9,600 updates,
  published one epoch-59 EMA, and passed byte-identical strict duplicate Gate
  replay across 40 videos/80,000 candidates. Control Avg/mAP@.6/mAP@.7 is
  `10.52/8.90/6.98`; centered is `12.57/11.04/8.14`, for centered-minus-control
  `+2.05/+2.14/+1.16 pp`. Centered selected context/ROI/residual is
  `210,925/1,613,683/1,517,728`; control remains `0/0/3,342,336`.
  Finalization `2a9351a3...` passes the registered signs and authorizes only
  `exp:scnr-residual-centering-paired-cost-v1`. Cost v1 is frozen as one Slurm
  Job/GPU with one continuous 20-ms NVML sidecar and eight serial
  `A B B A B A A B` passes. Primary center/control end-to-end-p50 and
  energy/sample ratios use a 10,000-replicate video-cluster/pass-pair bootstrap;
  both 95% upper bounds must be `<=1.05` to open seeds 3408/3409. Exact
  execution `2eca86cf` passed remote `65/65` and the frozen precheck; deployment
  `3e12809c...` released one cost Job `1233097` under
  `scnr_residual_centering_paired_cost_2eca86cf_from16137484_s3407_20260809_112558`.
  It is currently scheduler-pending, so no cost result exists. Official test,
  general Hybrid/complementarity, efficiency and paper claims remain closed.
  No further Pro discussion is needed before this frozen cost study completes.
  It is source-native token membership, not Online TAD. Pretrained VideoMAE
  absolute position stays on; all external
  coordinate, ROI-relative, geometry-projection/side-channel and
  weighted-pooling paths are off in the current probe.
- Learned hard policy: sequential conditional ordered PL,
  `p(ROI|context) p(residual|context,complete ROI)`, detector risk exactly
  `cls_loss+reg_loss`, temporal mean, EMA .95, local batch1, route-private RNG
  keyed by seed/successful-update/rank/role, default FP32 DDP reduction.
- Frozen exploratory study `georoute_hybrid_causal_pilot_v1`: seed5227,
  20 epochs, nine all-complete arms A0 Dense, A1 Fixed64, A2 Random64, A3
  residual-PL64, A4 context8+residual56, A5 context8+ROI56, A6 Hybrid-ST,
  A7 Hybrid-PL, A8 A7 with temporal geometry trajectory shift127.
- State is `tested_incomplete_finalizer_input_failure`, not empirically
  supported. Exact clean runtime
  `0f64218d` passed remote Linux/CUDA checks (`20/20` required C3 and `171/171`
  GeoRoute, one skipped) plus real binder SHA
  `202d8d75b024ae6f080caba461ba05c33edd0790b99d683c9751b4f449f2e78d`.
  No-performance P0 Jobs `1213665--1213667` sealed
  `PASS_MECHANICAL_ONLY`; all nine held leaves `1213694--1213702` completed
  `0:0` and each published a stage result and final checkpoint. After-any
  finalizer `1213703` failed closed before artifact/metric interpretation because
  it compared the insertion order of the canonically sorted `jobs.stages` JSON
  mapping with the frozen arm order. Every other deployment predicate passed,
  including the canonical hash and stage-key set. Its sealed failure status is
  `FAIL_UNTRUSTED_FINALIZER_INPUT`, with empty contrasts. Accuracy/telemetry and
  cost timing remain uninterpreted until a separately versioned, immutable-input
  recovery finalizer validates the full population; the old namespace is not
  rerun or edited.
- A single-seed pass authorizes only a separately frozen disjoint-seed study.
  If that future study claims PL over ST, matched Hybrid-ST must remain across
  seeds; otherwise estimator-superiority language is removed. No official test,
  paper, accuracy-preservation, complete-efficiency, or mechanism claim is open.
- Full review absorption:
  `docs/methods/reviews/2026-08-02-hybrid-causal-pro-review-absorption.md`.

## Current Active Route: NativeTokenSelect-first GeoRoute-AdaTAD (2026-07-28)

- 2026-07-29 Pro-review absorption verdict:
  `ACCEPT_WITH_MAJOR_REVISION / READY_PREEXPERIMENT_ONLY`. Free
  NativeTokenSelect v1 is closed as the primary candidate on strong descriptive
  negative evidence; this does not manufacture the missing old selector receipt.
  Hybrid v1, the failed namespace, and the old Free-first selector are not
  promoted or reused. The current implementation is source-native token routing,
  not Geometry Zoom. The proposed CER-TAD context/geometry/residual model remains
  `discussed`: dynamic role allocation, critic, boundary head, coverage/stability
  objectives, and the review-proposed eleven-arm matrix are not implementation
  ready. The immediate independent preexperiment is only (D) a two-pass
  full-development exact-index decode census, (K) numerical PL/ST and
  representation-isolation KATs, and (M) prediction-SHA-preserving diagnostic
  replay of the six old arms that have both a final checkpoint and prediction.
  ROI lacks a source prediction and is excluded from parity. Any decode, KAT,
  prediction-hash, or population mismatch stops before training. The
  review-proposed `+0.50/+0.30 pp` margins are not accepted as confirmatory
  thresholds because they lack an independent variance/power basis. Full design:
  `docs/methods/2026-07-29-georoute-estimator-representation-preexperiment.md`.
  D/K/M is now `tested_complete_go_pilot_design_only` at runtime commit
  `0c20f2e89e6af8bac0e3612776e03f80c0a9f3fb`. Jobs
  `1203105`--`1203113` all completed `0:0`; the two-pass census retrieved
  `272/272` items, every KAT passed, and all six Phase-M leaves preserved
  prediction SHA and a common 136-window population. Sealed finalization
  `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
  emitted `GO_PILOT_DESIGN_ONLY`, with training/P2/P3/official test/paper claim
  all false. Results:
  `docs/methods/2026-07-29-georoute-estimator-preexperiment-results.md`.
  A new independent study,
  `georoute_estimator_representation_pilot_v1`, is
  `c822add3_sealed_incomplete_no_performance_inference`.
  Six single-seed, 20-epoch, K=64 arms isolate PL versus ST, fixed-support
  representation, ROI-support representation, and ROI versus residual support.
  The first clean runtime `02b6efe7` passed remote Linux tests `108/108`, but
  P0 Jobs `1203380`--`1203385` all failed before a model result: the two
  single-tenant leaves passed the rendezvous gate and then exposed a script-mode
  `tools` import defect, while four co-located leaves exceeded the old
  30-second readiness threshold. No training leaf ran. Descendants
  `1203386`--`1203392` were canceled only after becoming impossible; finalizer
  `1203393` then exposed a JSON-key-order validation defect and produced no
  pilot finalization. This namespace is immutable deployment-failure evidence,
  not model evidence. The v2 repair launches P0 by module, installs the source
  root before dynamic imports, uses a decimal-Slurm-job-scoped `127/8` address
  plus a kernel-assigned port and unique run ID, raises readiness to 120 seconds
  with a hashed failure sidecar, terminates the entire torchrun process group on
  failure, validates the current node and arm maps independent of sorted JSON
  order, and changes all gating dependencies to `afterany` while stage wrappers
  fail before cell creation unless the sealed P0 suite passed. P0-finalizer and
  final-closeout prevalidation exceptions also produce hashed mechanical
  failure/INCOMPLETE receipts. Thus both success and failure paths reach a
  terminal non-promoting record. N16R4 Jobs
  `1203460/1203461` proved `srun --resv-ports=2` is unavailable
  (`Requires more ports than can be reserved`), so that mechanism is explicitly
  rejected. Fresh runtime
  `cbe0a08218a2f4550960f7c832f88c8cf77757c1` was proxy-synced with exact
  HEAD/origin/clean-tree parity and passed remote Linux tests `118/118`.
  Independent same-node Jobs `1203689/1203690` completed `0:0` concurrently
  on `g0005` with distinct job-scoped loopback hosts and four distinct actual
  TCPStore ports. The fresh pilot root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_cbe0a082_20260729_1849`.
  P0 Jobs `1203707`--`1203712` and finalizer `1203713` all completed `0:0`;
  sealed suite
  `00b7c0e3251f3d384df91cf900267694918d1245b4a5803150e8e2e1465210d2`
  is `PASS_MECHANICAL_ONLY`. Six leaves `1203714`--`1203719` started the
  frozen study in parallel. Residual-PL Job `1203715` hard-failed on real
  batch 0 after eight AMP retries from scale `32768` to `256`, with no
  checkpoint or metric. The other five leaves completed `0:0` only for terminal
  provenance. Afterany closeout `1203720` completed `0:0` and sealed schema-v2
  status `INCOMPLETE_EXPLORATORY_PILOT` with decision
  `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, `all_six_arms_passed=false`, an
  empty contrast set, and all selector/P2/P3/official-test/paper-claim guards
  false. Its canonical self-hash is
  `738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`;
  file SHA-256 is
  `63f73a353e356bc77a7a701972f22f62620b35e46b0c8f3eba0fc3c9816db0cc`.
  The four contrasts cannot be completed from this namespace.
  Root cause is a finite per-tubelet PL likelihood being temporally summed in
  FP16 over the 384-tubelet production horizon; the float64 `T=1` KAT and
  synthetic P0 missed it. A numerical-only FP32 likelihood/reduction repair
  and mandatory AMP-shaped `T=384/N=220/K=64` P0 KAT are now implemented
  in exact commit `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62`.
  The validator binds this to the real decoded `180x320`, floor-native `11x20`
  source grid rather than a mismatched synthetic capacity. The commit was
  proxy-synced to an exact clean remote snapshot, passed the complete Linux
  suite `120/120`, and standalone CUDA KAT Job `1203873` completed `0:0`.
  Its objective magnitude was `128637.0234375 > 65504`, all scaled gradients
  were finite, and receipt internal/file SHA-256 values are
  `7d0ccc346b95180d02a5ddcf4253ac0278e83f39a6f7e434357c86067e3c8e84`
  /
  `75ef280473f5032fd734fb86f1f58207702c1999d34c5c7132d40ff5017ae4a4`.
  The numerical repair plus the sealed old closeout permits only an all-at-once
  capacity check and a future fresh full six-arm P0/run, not a result. There is
  no resume, partial-performance inference,
  old selector, automatic promotion, P2/P3, official test, efficiency claim,
  Geometry Zoom claim, or paper claim.
  A fresh history-free agent independently audited the Pro absorption,
  six-arm contract, numerical repair, no-leak paths, all-six finalizer, and
  14-job fail-closed deployer. Its verdict is
  `DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY`: no further Pro discussion is needed.
  The old-closeout condition is now satisfied; Slurm must still admit all 14
  new jobs at once. This is code/protocol review evidence, not a model result.
  That capacity gate has now passed atomically: active jobs `2` plus all `14`
  new jobs exactly matched `MaxSubmitJobs=16`. Exact clean source
  `30f9ca6fff1572e2eabc6c1b6636c4cc23595a62` deployed a new namespace at
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_30f9ca6f_20260729_2023`.
  P0 Jobs `1204015`--`1204020` and afterany finalizer `1204021` completed
  `0:0`; suite
  `2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`
  is self-hash-valid `PASS_MECHANICAL_ONLY`. The repaired score-function report
  is schema v4 and binds Job `1204016`, `180x320 -> 11x20/N=220`,
  `T=384/K=64`, FP16 source, FP32 likelihood/loss,
  `|objective|=128637.0234375 > 65504`, and finite scaled gradients. All six
  frozen stage Jobs `1204022`--`1204027` are running in parallel; afterany
  closeout is `1204028`. No selector, P2/P3, official test, or claim is open.
  The repaired residual-PL stage `1204023` nevertheless hard-failed on real
  batch 0 after eight AMP retries, again at floor scale `256`, with no
  checkpoint or metric. Its failure self/file SHA-256 values are
  `f70b0a541cbfbbcf6595e8dfe7d7ef46ce16426d09a5ff7bc9fc921273c9eb81`
  /
  `e36556f20a5cdbf138779fd46efc2f31462aaec19dc49296bb23fa94180edb5e`.
  A single fresh independent agent audited the discrepancy and returned
  `HOLD -> REPAIR`: the P0 model backward was FP32 without autocast/GradScaler,
  while its AMP KAT differentiated only isolated route logits. It therefore
  missed the scaled full graph through detector, scout, adapter, and backbone.
  The other five stages completed only for terminal provenance; no partial
  result is valid. Closeout Job `1204028` completed `0:0` and sealed
  `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with `all_six_arms_passed=false`,
  an empty contrast set, and selector/P2/P3/official-test/paper guards false.
  Its self/file SHA-256 values are
  `60c9dab575e65830b7b849437963de2c7f789743caedb130b499c142c49c76ab`
  /
  `6ad32b7822042685b353f378c6eb9ea14be061e7f22d7db7288a129cbe080f06`.
  The next source must keep the six-arm science fixed, execute the
  score-function scout/likelihood path safely in FP32, and pass a full-model
  autocast+GradScaler optimizer-update P0 before any further replacement.
  Exact repair source
  `c822add335c38a9f6c63e609237c4bfa9b9f468d` is now `tested` for the
  numerical gate: backbone schema v4 keeps the complete scout/route forward and
  backward in FP32 outside autocast; P0 schema v5 runs the actual score-function
  model graph under FP16 autocast and GradScaler at floor scale `256`, unscales
  and audits required parameter gradients, then requires a successful
  zero-learning-rate optimizer step. The exact clean Linux snapshot passed
  `121/121`. Standalone Slurm Job `1204087` completed `0:0`; its self-hash-valid
  report records `T384/N220/K64`, FP16 source, FP32 likelihood/loss,
  `|objective|=128637.0234375 > 65504`, full-graph scale `256 -> 256`, finite
  required gradients, FP32 scout execution, a successful optimizer update and
  zero checkpoints. The estimator loss, weight, six arms and contrasts are
  unchanged. The old closeout and all-at-once capacity gates then passed.
  With `active=2`, the full 14-job DAG exactly filled `MaxSubmitJobs=16`.
  Exact source `c822add3` created fresh root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_estimator_representation_pilot_c822add3_20260729_2149`.
  P0 Jobs `1204301`--`1204306`, P0 finalizer `1204307`, six stage wrappers
  `1204308`--`1204313`, and afterany closeout `1204314` were submitted
  atomically. All six P0 leaves and the finalizer completed `0:0`; sealed suite
  `f6f423670c9c2417aadfca97c67d794427ee337c359ba2d2509faee53a5ccdb6`
  is self-hash-valid `PASS_MECHANICAL_ONLY`. Residual-PL stage `1204309`
  nevertheless hard-failed on real batch 0 after exhausting all eight AMP
  retries from scale `32768` through `256`. Its failure receipt is
  self-hash-valid
  (`5e55619291285a36a8410be9582a79f293b33cd135053b4c6a3967e9d8beb5c8`;
  file
  `77c9eff76c5881f8daa45ef68dd0d0d71bec2f5e171f31e2a8a97fc734b9f3c5`)
  and the cell contains no checkpoint, metric, or stage result. The other five
  stages and closeout `1204314` completed `0:0`; each surviving stage has one
  final epoch-19 checkpoint, one stage result, and zero temporary checkpoints,
  but all are terminal provenance only. ROI-PL Jobs `1204312/1204313` each
  logged 11 cumulative failed AMP optimizer attempts before later successful
  updates, with scale reaching `64`. The earlier monitor classified cumulative
  count `>10` as a protocol hard failure; this is corrected. The source contract
  sets `max_amp_retries_per_batch=8`, and hard-fails only when one batch cannot
  produce a successful update after those retries. The cumulative counter is a
  numerical-stress alert, not a sealed experiment decision threshold.
  Closeout `1204314` sealed schema-v2
  `INCOMPLETE_EXPLORATORY_PILOT /
  PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, with
  `all_six_arms_passed=false`, empty contrasts, and all promotion guards false.
  Its canonical self/file SHA-256 values are
  `a02e551ba9007b49670103e2e4db3bf1c1d917cb5a7bb5c4dd724274b9379a2a`
  /
  `c95c1694dccbda2687b1b9e6e07bb9016ebe80181e2288d172874afa791d8f1c`.
  Thus `1204309` is the sole formal stage hard failure. The run shows that the
  current synthetic full-graph P0 is insufficient evidence of real-batch AMP
  stability; it does not yield a PL/ST, representation, efficiency, or Geometry
  Zoom verdict. No rerun, partial contrast, selector, P2/P3, official test, or
  claim is authorized.
  The approved next step is now implemented locally at exact source
  `832caedd3713f477cb4b2f29a692acba9cd5a836` as
  `exp:georoute-real-batch-amp-diagnostic-v1`: matched residual-PL/ST on the
  production data/training path, with input/RNG hashes and scaled/unscaled/
  clipped gradient localization. The old job lacks sample indices and RNG
  state, so this is not described as bitwise replay. It emits no metrics,
  checkpoint, prediction or test evidence. Only a localized cause can authorize
  a minimal repair and a fresh real-data stability gate. The observer is
  opt-in; parent pilot runtime/file hash, stage and wrapper-failure
  self-hashes, Slurm IDs and rendezvous are bound. Local pure checks pass
  `50/50`, required C3 regressions pass `20/20`, and Python/Bash/whitespace
  checks pass. Clean proxy-synced N16R4 Linux/CUDA validation and all three
  Slurm jobs are still pending, so no numerical diagnosis exists yet.
  The first exact-source attempt is now sealed as an infrastructure failure:
  clean `832caedd` passed the remote Linux/Torch suite `98/98`, but both PL
  `1204847` and ST `1204848` failed before observer/model execution because
  `mmengine.Config` has no `__delitem__`. Afterany finalizer `1204849` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR` with no performance artifacts. Minimal
  source `64d991f96981a3e60b10f47d6d093d5457da9c60` uses `Config.pop` and adds a
  real binder regression; local combined checks pass `71/71`. Its clean replay
  passed remote `99/99`, but PL/ST Jobs `1204864/1204865` stopped before the
  observer because the binding inverted exclusion-list semantics; finalizer
  `1204866` sealed no-repair. Corrected source `047f643f` passed remote
  `149/149`; Jobs `1204908/1204909` reached an identical real batch with matched
  input/CPU/CUDA RNG hashes and finite forward losses, then both stopped before
  backward because the diagnostic enabled strict deterministic error mode,
  unlike the historical pilot's warn-only policy. Finalizer `1204910` sealed
  `DIAGNOSTIC_INCOMPLETE_NO_REPAIR` (self-hash `7755f777...`). Exact candidate
  `861e9b1edba5baf1b96fe0d4ed1c3c08d1e2da58` restored and receipt-bound the
  historical deterministic warn-only seed policy. Its clean matched run
  `1204944/1204945/1204946` sealed
  `ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED`: PL failed nine scaled attempts from
  `65536` through `256` only in `scout_score_function` and first succeeded at
  `128`, whereas matched ST had zero failures and succeeded at `65536`; the
  batch/input/CPU/CUDA RNG hashes matched and no performance artifact existed.
  Source `768e1a30` then implemented an explicit per-tubelet temporal mean for
  the PL score-function loss, and exact source
  `86ff1dde6ddb058ca9250f968972c255f19dab92` fail-closed the real-data gate's
  parent/input bindings; clean remote GeoRoute checks passed `124/124`.
  Stability-v1 Jobs `1205033/1205034/1205035` are now sealed HOLD. Repaired PL
  passed its first two batches at `65536` but had two nonfinite
  `residual_head.weight` gradients on batch 3; ST passed twenty batches at
  `65536` but had detector-head nonfinite gradients on batch 21. Both therefore
  violated the preregistered 32-batch zero-skip rule. This is a failure of that
  numerical gate, not a performance or PL/ST verdict. The exact official AdaTAD
  path uses dynamic GradScaler and does not require zero skip, so stability-v1
  cannot be called an official-comparability criterion; an official-semantics,
  no-metric replacement gate must be separately frozen on an independent data
  order before any performance run.
  That replacement is now implemented and locally `tested` as
  `exp:georoute-real-data-amp-stability-v2`: seed/order `4417`, 64 batches,
  default dynamic GradScaler, zero retry/replay, official scheduler/EMA
  per-batch transitions, at most two nonconsecutive skips, scale floor
  `16384`, and a fully successful final-16 tail. Its binding explicitly records
  that only transition cadence is matched: scheduler hyperparameters and the
  full official recipe are not matched, so the gate is not performance
  comparable. Exact runtime source
  `27fba03cb6d4932ee10cb4545b97984dff28c28c` passed the clean remote
  Linux/Torch suite `168/168`; PL/ST/finalizer Jobs
  `1205588/1205589/1205590` are now terminal. PL consumed 64 batches with 61
  successful updates and skips at `11/20/29`, ending at scale `8192`; ST
  consumed 64 batches with 62 successful updates and skips at `20/29`, ending
  at `16384`. Both had finite forward losses and a successful final-16 tail,
  but PL violated the frozen two-skip limit and scale floor. The canonical
  finalizer emitted
  `INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2 /
  OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`; protocol freeze and every
  performance/test/paper guard are false. Finalization internal/file SHA-256:
  `ab7ea3e5fca378532b689f8dce8d3ed57631ca78eec99b91a77a96a5e8e29d56` /
  `c7f59dbcec609430bdf4aafe99cc5ef3272ef93362b7f44ba74bcbc337c85ab0`.
  This is terminal numerical-gate evidence only: do not rerun, supplement,
  change thresholds, infer model performance or launch the formal protocol.
  The next registered step is
  `exp:georoute-gradient-decomposition-diagnostic-v1`, following the
  user-provided Pro verdict `NEW_MATCHED_DIAGNOSIS_BEFORE_REPAIR` (attachment
  SHA-256
  `22f5802f62689f687667f56ddd6aacb35e07242c213a591cf93a4e50942c6e83`).
  Candidate `664180b6` passed remote `161/161` and its KAT, but its first DAG
  admission stopped before namespace creation/`sbatch` on the wrong sealed-v2
  provenance key (`failed_batch_indices` instead of
  `skipped_batch_indices`). Corrected exact source
  `33f721be83e0ad7f7a36e853491e7a14f148814b` passed clean remote
  `162/162`; same-commit CUDA/DDP KAT `1207480` completed `0:0` with
  self/file SHA-256
  `d31d34144e60bdde6103acc36cff42301ba7fbd80a40eb6f04ead63ddb6901b4`
  /
  `09e2ed0ec6f6e3372871ea00f0aa610027bbedd81d098b60ea6a2529aed0e6f4`.
  The sealed diagnostic root is
  `georoute_pl_gradient_decomposition_v1_33f721be_s7367_20260730_2300`;
  PL/ST/finalizer Jobs `1207484/1207485/1207486` all completed `0:0`.
  Both arms consumed 64 matched batches with finite forwards, zero retry/replay
  and 64 scheduler/EMA advances. PL skips at `2/20/29` were all uniquely
  `DDP_FP16_CAST_OVERFLOW`: analytic and actual residual-logit gradients plus
  pre-hook FP32 buckets were finite, while the detached FP16 cast introduced
  the first nonfinite. ST skips at `14/29` were detector-only and already
  nonfinite in FP32 pre-hook, so they do not explain the PL scout failures.
  Finalizer classified
  `PL_NUMERICAL_MECHANISM_LOCALIZED_REPAIR_CLASS_IDENTIFIED`; self/file SHA-256
  are
  `52d4dfd698ed0679a976e6d468fb4b0d1ede9ea630df32f808115c9f118f681e`
  /
  `816819086374f964264d3a8bb4810842f97ef554d5661d2ec4a6b85fd135bc9c`.
  All hashes and the zero-artifact audit independently pass. The study froze seed `7367`, matched
  residual-PL/ST, 64 consumed batches, `T/N/K=384/220/64`, temperature `0.7`,
  temporal mean, default GradScaler, the production FP16 compression hook, zero
  retry/replay, and no performance artifacts. It observes the analytic PL
  score, actual residual-logit gradient, pre-hook FP32 buckets, detached FP16
  shadow casts, unscaled/clipped gradients, and scaler/scheduler/EMA transitions.
  The Pro recommendation is accepted with one necessary matchedness correction:
  data and CPU RNG must match across all 64 batches and CUDA RNG at batch zero;
  later CUDA RNG equality is not required because PL Gumbel sampling consumes
  CUDA RNG while ST does not. No replay/reset may hide that divergence. A
  identified class authorizes only one minimal numerical repair and a fresh
  no-performance gate. The preregistered next repair is to disable DDP FP16
  compression across the matched native family; no estimator/objective change
  or performance run is authorized.
  That successor is now implemented as
  `exp:georoute-ddp-fp16-cast-repair-gate-v1`: independent mechanically derived
  seed `2307`, the same 64-batch official-prefix PL/ST protocol and inherited
  bounded-scaler thresholds, with `solver.fp16_compress=false` as the sole
  intervention in both arms. A same-commit real CUDA/DDP KAT must first prove
  that a finite scaled FP32 gradient above `65504` survives default FP32 DDP
  reduction, while a detached FP16 shadow cast overflows. Exact clean source
  `685f935e759d5d78f94e5f208997644e07bf4654` passed focused local
  AMP/repair tests `32/32` and the complete remote GeoRoute suite `145/145`.
  Same-commit Slurm KAT Job `1207542` completed `0:0` and sealed
  `PASS_DDP_FP16_CAST_REPAIR_CUDA_KAT_ONLY`: a finite FP32 scaled gradient of
  `70000` survived default NCCL/DDP reduction without a compression hook, its
  detached FP16 shadow overflowed, unscale remained finite, and the optimizer
  update succeeded. KAT self/file SHA-256 are
  `257436d617b79413b4b790cda754d6dec56602d52edb07e50c03cdcd28f78b4f`
  /
  `d957514816f660a8eb43b922dfb3325baf36f1bbb706f398d0a54cc0a37df3ae`.
  The fresh gate is now terminal at
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_ddp_fp16_cast_repair_gate_v1_685f935e_s2307_20260730_2314`;
  PL/ST/finalizer Jobs `1207554/1207555/1207556` all completed `0:0`.
  Both arms consumed 64 matched batches with finite forward losses, default
  dynamic GradScaler, zero retry/replay and exactly the registered
  `solver.fp16_compress=false` intervention. Both skipped only batches `20/29`,
  made 62 successful updates, reached final/minimum scale `16384`, and completed
  a successful final-16 tail. Data and CPU RNG sequences, initial CUDA RNG,
  seed and immutable input bindings match; cross-arm skip delta is `0` and
  final-scale ratio is `1.0`. Finalizer sealed
  `COMPLETE_DDP_FP16_CAST_REPAIR_GATE_ONLY /
  DDP_FP16_CAST_REPAIR_GATE_PASS_MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED`;
  finalization self/file SHA-256 are
  `ad556812454f2ff02161587979ac99c33d9a4983b5c8fcd97d26efe47a936185`
  /
  `f8ef174c934b42ef4efb98e91f16ee4a0a79d0b9f0bbc9c3e174ad3b64bd77e3`.
  This validates the no-compression numerical repair and authorizes freezing a
  later matched formal protocol only. It explicitly leaves
  `official_protocol_freeze_authorized=false` and all metric, checkpoint,
  prediction, evaluator/NMS, official-test and paper guards closed.
  The single-seed 20-epoch pilot is not an official paper result. A future
  confirmatory study must include both an exact official AdaTAD reproduction
  and a matched native-source dense control, then match updates, effective batch
  size, EMA, evaluator/NMS, sealed-test policy and decode-to-NMS
  latency/memory/energy across disjoint seeds. The current GeoRoute development
  recipe uses a development-only population, batch size `1`, warmup `2`, no
  validation evaluator/NMS and final-only checkpointing, whereas the official
  AdaTAD anchor uses its official split/evaluator recipe, effective batch
  `2`, warmup `5`, and scheduled validation/checkpointing. Their numbers must
  never share a paper comparison row.
  The authorized successor is now implemented locally as
  `exp:georoute-official-comparable-protocol-v1`, with remote execution still
  pending. Its pinned upstream anchor is OpenTAD commit
  `01c58b9f2370e914150cf94d392208a4e211c053`; the byte-identical THUMOS
  VideoMAE-S Adapter config has SHA-256
  `5521b6ce28cc6770e662d3dfdd4621479bc228be6131e300a92285fb4961a49c`.
  A key implementation correction is frozen: OpenTAD divides the configured
  batch by world size, so official config batch `2` under two ranks means
  global batch `2` and local batch `1`, not global batch `4`. F0 is a
  no-performance admission gate with two parallel 32-batch real-data
  single-rank stress leaves plus a world-size-two FP32-DDP KAT. Only F0 PASS
  may release F1: dense/fixed/random/ST/PL x seeds `3407/3408/3409`, 60 epochs,
  exact K=64, official scheduler 5/100, AMP/EMA/static graph, no FP16
  communication, final EMA only and complete Fit/Gate evaluation. F1 metrics
  remain development-only. Each selector must beat fixed and random at
  mean(mAP@0.6,mAP@0.7) on every seed and cost less than dense on every seed;
  ST versus PL additionally requires strict paired-seed accuracy/cost Pareto
  dominance. Ambiguity is `HOLD_NO_OFFICIAL_TEST`; Geometry is excluded. A
  later F2 still requires the official reproduction/bridge stack, complete
  decode-to-NMS cost and one separately sealed official-test open.
  Exact source `3d8c2b487fa983d6d6240b347177cc423a37748b` passed remote Linux
  focused `25/25` and all GeoRoute `154/154`; source and pinned upstream
  `01c58b9` snapshots have full HEAD/origin-ref/clean/config parity. Fresh F0
  root `georoute_official_comparable_preflight_v1_3d8c2b48_20260731_122316`
  completed PL/ST/KAT/finalizer Jobs `1209309`--`1209312` at `0:0`. Both
  real-data leaves consumed 32 matched batches, recorded one official-semantic
  scaler skip, ended at scale `32768`, and passed the final-16 stable tail;
  world-two default FP32 DDP reduction/update passed. Finalizer
  `313da95faeae9e600965fe4ac5c7ad5816f652d5ff2c97cf9734f7028d888a3c`
  authorizes only the complete F1 development matrix. It emitted no
  checkpoint, prediction, metric, evaluator or official-test artifact and
  supplies no mAP/model evidence. F1 remains unsubmitted because its immutable
  16-submission gate currently sees two unrelated active jobs and its
  conservative training storage gate sees `31,646,543,872` free versus
  `130,996,502,528` required bytes. Do not split the matrix, cancel unrelated
  jobs, or relax either gate.
- Objective: first test whether detector-supervised, ROI-free exact-K selection
  of source-native VideoMAE tubelets protects high-tIoU offline TAD at lower
  measured total cost. Only after that base passes may continuous geometry be
  tested as a strict add-on.
- Active replacement status:
  `p1r_data_decode_failure_no_selector_exact_source_7be8363e`
  (`tested`, not `empirically_supported`). The external Pro audit of exact
  commit `df3e54e0c6776544dba20807b2ec100e1a399654` returned
  `HOLD_FOR_CORRECTNESS_FIX`. The local replacement now implements floor-native
  176x320 support with a validity mask, mask-aware exact-K, a
  coordinate-lineage packed Adapter, a truly geometry-free `free` control,
  common uniform-selected pooling, branch-aligned hybrid gradients,
  final-only atomic checkpoints, and same-commit aggregate storage preflight.
  Clean GitHub snapshot `45f5cca2e6b003478327511e3f38c8871b77084f`
  was synced through the frozen academic proxy; GeoRoute remote focused tests
  passed `58/58` and required C3 regressions passed `20/20`. P0R Jobs
  `1199838`--`1199840` plus afterok finalizer `1199841` were submitted under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_nativefirst_45f5cca2_p0p3_20260728_1630`.
  Aggregate storage preflight passed with 214,831,312,896 free versus
  79,456,894,976 required bytes. P0R Jobs `1199838`--`1199840` and finalizer
  `1199841` all completed `0:0`; sealed suite
  `a0073394c3f0f505679797a4c22afeefda1d32adea7b4615e2eec4bbeed35077`
  is `PASS_MECHANICAL_ONLY`, with full-token parity, packed
  attention/MLP/Adapter execution, zero dense Adapter calls, real detector
  backward, route gradients, zero checkpoints, and same-commit storage
  measurement verified. It automatically submitted seven P1R leaves
  `1199865`--`1199871` and selector `1199872`. Dense `1199865` and fixed
  `1199866`, fixed-plus-geometry `1199867`, and random `1199868` completed
  `0:0`, each with one final checkpoint, zero temporary files, passing storage
  receipt, and a development-only stage result. Their diagnostic
  Avg-mAP/mAP@0.6/mAP@0.7 are respectively `13.90/11.83/8.74`,
  `12.42/10.75/7.17`, `12.63/10.40/7.09`, and `12.68/10.76/7.53`.
  Fixed-plus-geometry and random used exact unique `K=64` of 220, zero
  duplicates, one heavy forward, 12 packed attention/MLP/Adapter calls, and
  zero dense Adapter calls. Their profiles are development-only
  model-and-postprocess diagnostics, not paper-grade full-stack or energy
  evidence. Primary free NativeTokenSelect `1199869` failed `1:0` in Epoch 8
  without checkpoint or stage result because its implicit torchrun localhost
  port `29400` collided with fixed on shared node `g0043`. Hybrid `1199871`
  independently confirmed the same defect on `g0048`: random ended training
  at 19:04:37, then hybrid lost the shared store and failed `1:0` in Epoch 6
  with no checkpoint or stage result. Neither failure showed OOM or non-finite
  loss/cost. ROI-only `1199870` completed `0:0` with
  Avg-mAP/mAP@0.6/mAP@0.7 `13.18/11.28/8.95`, exact unique `K=64`, one heavy
  forward, and p50/p95/peak `905.40 ms/4360.95 ms/1818.21 MB`. All five
  completed cells have one final checkpoint, zero temporary files, passing
  storage receipts, and development-only stage results. The cost scope is
  model-and-postprocess only, excludes the evaluator, has no energy receipt,
  and is not paper-grade end-to-end evidence. Free and hybrid have no
  checkpoint or stage result. Selector `1199872` is
  `DependencyNeverSatisfied` and has no receipt; P2/P3 and official test are
  absent. Thus this P1R matrix is infrastructure-invalid, supplies no
  NativeTokenSelect or conditional-geometry verdict, and supports no
  efficiency or paper claim. The approved replacement now removes implicit
  standalone, binds train/test to kernel-assigned `127.0.0.1:0` endpoints and
  unique Slurm/cell/phase rendezvous IDs, adds a real same-node concurrent
  lifetime gate using observed `TORCHELASTIC_RUN_ID` and `MASTER_PORT`, and
  hash-binds each P0 model report to its same-leaf isolation receipt. Local
  non-Torch compile/focused/C3 checks pass `59/59`.
  Clean source `a2ebd0604b4e5648b4f9bc4b3432541fae070393` passed remote
  Linux tests `82/82`, but P0R `1200510`--`1200512` all failed before model
  execution because the first gate's fixed 0.5/2.0-second probe durations
  conflated torchrun parent teardown with store lifetime. Finalizer `1200513`
  was dependency-unsatisfied; no P0/P1 result exists. A deterministic
  replacement now keeps the long worker blocked until the controller observes
  complete short-parent exit and publishes a peer-exit marker; it still
  requires a fresh commit, namespace, and P0R.
  Deterministic source `bfee57904b3919480ce56b72429314eda508bf8e` also passed
  `82/82`, but P0R `1200550`--`1200552` failed before model execution because
  the gate required literal `MASTER_ADDR=127.0.0.1`. Slurm diagnostic `1200560`
  observed the correct dynamic port `57695` and run ID but
  `MASTER_ADDR=g0024`, the allocated node hostname. The local validator now
  binds master address to exact `socket.gethostname()`; a gate-only Slurm pass
  was mandatory before another P0 namespace.
  Exact clean source `7be8363ea6e26b320bffafeb03f0e82d8b660779`
  passed remote Linux tests `82/82`. Gate-only Job `1200602` then passed
  concurrent rendezvous isolation on `g0053` with exact run IDs, distinct
  dynamic ports `54013/34325`, and the long worker alive after complete
  short-parent exit. P0R Jobs `1200611`--`1200613` all completed `0:0`; their
  three same-leaf isolation receipts and CUDA reports sealed suite
  `693034b276697e92ae915ea5f40cebdd5d01a76bad65f46e5639844654f210e9`
  as `PASS_MECHANICAL_ONLY`. Finalizer `1200614` failed only after writing that
  receipt because obsolete dependency-dead jobs made the submit-cap preflight
  report `active=11, required_additional=8, MaxSubmitJobs=16`; no P1 job was
  partially submitted in that namespace. After cancelling only obsolete
  GeoRoute Jobs `1199872`, `1200513`, and `1200553` while leaving DUCA/RIME
  untouched, sealed-parent bootstrap `1200652` completed `0:0` into fresh root
  `georoute_nativefirst_7be8363e_p1p3_20260728_2225`. Dense `1200663`, fixed
  `1200664`, fixed-plus-geometry `1200665`, random `1200666`, free
  NativeTokenSelect `1200667`, and hybrid `1200669` completed `0:0`, each with
  one final checkpoint, zero temporary files, passing storage receipt, and a
  development-only stage result. Their Avg-mAP/mAP@0.6/mAP@0.7 are
  `13.90/11.83/8.74`, `12.42/10.75/7.17`, `12.63/10.40/7.09`,
  `12.68/10.76/7.53`, `10.03/7.80/5.27`, and `13.23/11.35/8.81`.
  Free is descriptively worse than fixed, random, and fixed-plus-geometry by
  `2.39/2.65/2.60` Avg-mAP and by `2.95/2.96/2.60` at mAP@0.6, so it does not
  satisfy the preregistered native-base accuracy condition. Hybrid's descriptive
  gain over free cannot authorize geometry because the native base did not
  pass. ROI-only `1200668` finished training and wrote its unique final
  checkpoint, then failed `1:0` during development testing when decord could
  not retrieve final video frames before `DECORD_EOF_RETRY_MAX=10240`; it has no
  prediction or stage result. This is a data/video-decode I/O failure, not OOM,
  non-finite loss/cost, storage, model, or rendezvous failure. Selector
  `1200670` is `DependencyNeverSatisfied` with no receipt. Therefore the matrix
  is protocol-incomplete and supplies no formal selector verdict; P2/P3 and
  official test remain closed. Available timing is model-and-postprocess-only,
  excludes evaluator and energy, and permits no paper-grade efficiency claim.
- Historical P1 status remains
  `failed_p1_infrastructure_storage_exhaustion_no_metric`. The sealed P0 parent from
  [`4a9358d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4a9358d1fba4bde9aa7693a94f7e4dfc95d31ecc)
  remains `PASS_MECHANICAL_ONLY`. Clean dispatcher snapshot
  [`6a9bba62`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/6a9bba6222c18a468c3bd410edac89a4afdea189)
  completed bootstrap Job `1196062` (`0:0`) and atomically submitted the seven
  matched P1 leaves `1196071`--`1196077`. All seven later failed `1:0` while
  publishing per-epoch checkpoints after `/data` reached 100% usage; the
  namespace had accumulated 63 GB. Result-blind selector `1196078` remains
  dependency-held and emitted no decision. This is immutable infrastructure
  failure evidence only: no P1 mAP, cost, A-MoD result, empirical support,
  official-test evidence, or paper claim exists.
- P0R contains three mechanical CUDA leaves. A one-shot Slurm dependency graph
  submits all seven P1R arms automatically and concurrently only after the P0R
  finalizer emits `PASS_MECHANICAL_ONLY`. Parallel scheduling does not alter the
  causal selector order.
- P1R is the first scientific screen: matched dense, fixed lattice,
  lattice-plus-geometry side-channel, random, ROI-free NativeTokenSelect
  (`free`), ROI-only, and corrected hybrid, all with uniform pooling and the
  packed Adapter. The native base must beat fixed, random, and the geometry
  side-channel while costing less than dense. Geometry is considered only
  afterward and must strictly improve on free, random, and the geometry
  side-channel without higher total cost. Otherwise Route B advances or learned
  routing stops. P2 promotes only the authorized route to seeds/budgets; P3 is
  frozen second-detector/dataset and sealed-test closure.
- The prior quota hold was cleared, but P1 is now storage-held. P0 replacement Gate `1181172` passed the
  real uint8 180x320 path; roots `1181007` and `1181177` remain immutable
  scheduler diagnostics only. The fresh namespace is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_6a9bba62_p1p3_20260727_222913`.
  Its bootstrap and submission receipts bind the sealed P0 suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`.
  The replacement uses a new namespace, result-blind aggregate
  storage-capacity preflight, and one atomic final checkpoint per cell.
  The failed namespace was pruned conservatively on 2026-07-28: seven
  highest-loadable per-cell checkpoints were retained and 58.370 GiB of
  intermediate/corrupt epoch files were removed. This does not make the
  namespace resumable or create P1 evidence.
  A subsequent user-authorized root-wide retention pass validated the highest
  loadable checkpoint in 48 additional multi-epoch directories and deleted 273
  lower epoch files (144.455 GiB) plus 166 matching metadata/temp companions.
  Together the two passes removed 380 epoch checkpoint files
  (202.825 GiB); a bound post-verification found zero multi-epoch directories
  among the 278 still-existing inventoried checkpoint directories, and `/data`
  reported 205 GB available. Pretrained weights, `best.pth`, configs, logs, and
  single-checkpoint directories were not changed.
  P2/P3 remain absent and result-gated. The code is native-token evidence
  routing, not a sequential second crop/resized zoom; “Geometry Zoom” remains
  unauthorized unless the conditional geometry gate and later paper evidence
  close.
- FlashVID was audited as a VLLM reference, not a GeoRoute result. Its 10%
  retention result is 57.9/58.4 = 99.1% relative score after a full vision
  encoder, so it cannot support native-pre-backbone or detector-gradient
  claims. Its relevance-diversity-motion correspondence principle is only a
  conditional P2 scout-side comparator after a P1 hybrid win.

## Continuous-RoI / Native-Crop Record (Frozen or Held)

- The original spatial goal is source-coordinate, variable `(cx, cy, w, h)`
  crop tubes at native local pixel density while retaining the full temporal
  axis. Dense 160/224/256 resizing is R0 headroom control, not a crop result.
- Continuous-RoI S2 exact-nine training (`1177668`--`1177676`) is sealed as
  `PASS_TRAINING_ONLY`: 60 epochs, 4,800 successful updates per cell,
  final-EMA-only, and no official-test opening. It is neither crop
  sufficiency, cost, mAP, nor a learned-policy result.
- Its fixed/variable reference protocol is `HOLD`: common physical centers,
  Sobol generator identity, candidate-ID authority, no-GT raw entrypoint and
  privileged join/tie/statistics are not jointly frozen. Only a result-blind
  v2.2 corrigendum is allowed; no official test or S3 learned policy follows.
- Native-Crop S1 merely established source-native crop data/model/gradient and
  no-leak mechanics. The fixed 128 candidate library is D0 diagnostic only.
  Historical S1/R0 campaigns must not be resumed, combined, or presented as
  crop GO/KILL evidence.

## C3 / DUCA Historical Baselines and Negative Memory

- Project-wide target remains offline, task-aware redundant-computation
  removal with protected mAP@0.6/0.7 and full decode-to-NMS cost. It is not
  causal or Online TAD.
- DUCA is a frozen, unproven full-window candidate. Its honest contract is
  `offline_full_window + runtime_generated + cache_free + jointly_trained`.
  It uses a low-cost coarse probe, transition/boundary-sensitive selection,
  fixed-K positions and AdaTAD-derived components; it cannot be called an
  unmodified official AdaTAD plugin or a paper method before matched evidence.
- Do not revive these mistakes: actionness top-k as the final selector;
  post-hoc gap repair/uniform scaffolds that hide learning failure; old-commit
  mAP as current evidence; smoke/gradient checks as utility evidence; dense
  X3D as a low-cost main probe; dynamic MUST as a main contribution; or FLOPs
  without trained end-to-end cost.
- Known failure mechanisms remain valuable: actionness focuses action interiors
  rather than boundaries; complex coverage constraints can collapse toward
  uniform; GAS-VT train/apply mismatch and hard repair invalidated its main
  reading; selected-axis geometry can damage high-tIoU; and requested,
  effective, unique, padded and actual backbone budgets must be logged
  separately.

## Non-Negotiable Evidence Rules

1. Match commit, data, pretrained initialization, updates, effective batch
   size, AMP, EMA, seeds, token budget, detector/head, evaluator and effective
   NMS before comparing selection methods. Pair an exact official reproduction
   with a matched native-source dense control whenever preprocessing changes.
2. Report high-tIoU, short-action/boundary diagnostics and measured full-stack
   p50/p95 latency, memory and energy. FLOPs or random-init profiling alone
   cannot establish efficiency.
3. Training-only, smoke, precheck, pending or failed jobs never become
   empirical support. Test/validation GT, teacher signals and raw prediction
   caches must never participate in inference decisions.
4. Any route that fails its matched control narrows or dies instead of gaining
   extra selector heads or loss weights. Preserve failed evidence in the
   experiment record and `anti_repetition.md`.

## Current Decision Question: K100-TAR50 Interaction Falsifier

- TAR32 is a valid single-seed negative result for the exact R1/K64 plus
  `[K64,K32]x6` composition: `STOP_R1_TAR32_FKV_EXACT_COMPOSITION`. It does not
  establish latency, energy, memory, multi-seed generalization or a universal
  failure of context-preserved transformation compression.
- The only open causal ambiguity is whether fixed half transformation fails on
  native K100 by itself or only after it is stacked with R1/K64 spatial
  compression. The unique next task is
  `ZT-CPTC-K100-TAR50-INTERACTION-FALSIFIER-001`.
- The frozen candidate is a minimal clean descendant of `2d945e64...`: native
  K100, `[K100,K50]x6` (800/400 flattened tokens), immediately preceding dense
  attention-column ranking, full K/V, full Adapter, dense detector input and
  exact identity bypass. No new parameters, loss, cache, fallback, dynamic K,
  residual predictor or selector sweep is allowed.
- Strict A-MoD capacity=1 job `1254040` is the route-matched reference:
  Average mAP/mAP@0.6/mAP@0.7 `68.73/61.59/47.20`. One seed-42, 60-epoch,
  epoch-59-EMA official-validation Slurm submission is allowed. All six frozen
  accuracy/short-action/boundary gates must pass; any valid failure stops the
  fixed-half-update attention-column identity-bypass family. No cost is
  authorized before a fresh Pro review.
- Formal job `1261680` exhausted the single authorized submission and failed
  before its first successful optimizer update with `ValueError: successful
  update indexing requires a GeoRoute backbone`. It produced no checkpoint,
  official vector, short-action/boundary diagnostic or cost evidence and is
  therefore `ENGINEERING_OR_PROTOCOL_BLOCKER`, not a scientific negative.
- A material specification conflict was found during terminal audit. The frozen
  prose says full K/V over all 800 tokens and `[K100,K50]x6`, but the unchanged
  inherited strict A-MoD odd block gathers a global flattened top 400 and runs
  attention on that selected tensor, so its K/V are selected-400 rather than
  full-800 and it is not an explicit per-tubelet K50 contract. This conflict and
  the hook failure must be adjudicated by a fresh Project Pro; no corrected run,
  route decision or family-level claim is authorized locally.

## User-Supplied Pro Pivot Intake: RACER24

- The user manually transferred a Pro-style response with verdict `PIVOT`, role
  contract `KEEP`, and a proposed unique candidate `ZoomToken-RACER24`. No exact
  Project/conversation/nonce/model/attachment/submission/Oracle receipt exists in
  the repository for that response, so it is preserved as a provenance-warning
  source statement rather than browser-audited Project Pro evidence.
- The proposed mechanism keeps BPNS contiguous native K64 support and a dense
  carrier, applies RACER at blocks `{4,6,8,10}`, selects exactly 24/64 tokens per
  tubelet, computes selected queries against all 512 keys/values, and completes
  unselected residuals without new trainable parameters before the existing
  Adapter processes all 512 tokens.
- Two source statements are explicitly rejected as project facts: job `1258299`
  is terminal rather than running, and the current base has no ready-made
  selected-query/full-KV or RACER-completion helper.
- The user separately authorized the frozen Iteration-0 implementation and
  matched block microbenchmark. Clean candidate `5ebaa74f...` passed the focused
  implementation checks, fresh Critic and result-blind Evaluator PRE_RUN.
- The only valid microbenchmark job `1262068` measured dense/RACER24 p50 as
  `1.33477/5.34684 ms`; speedup was only `0.24964x`. RACER24 peak allocated and
  reserved memory were `1.98884x/1.84615x` the dense control. The frozen
  `>=1.08x` speed and `<=1.05x` memory gates all failed.
- This is a valid negative result for the current RACER24 block-path
  implementation, not accuracy, full-stack TAD, energy or family-universal
  evidence. Iteration-0 is stopped. Training, K/block tuning, full-stack cost,
  FARM24 and PairLatent32 remain forbidden pending a fresh Pro decision.

## Fresh Pro Adjudication and Unique Task: GridFuse32-L6

- The exact-Project fresh Pro conversation
  `6a94842b-1370-83ea-a13c-2cc492170597` returned `PIVOT`, role contract `KEEP`,
  and permanently stopped the exact RACER24 Iteration-0 implementation. The
  `1262068` result is a decision-grade valid negative with a disclosed pre-push
  deployment deviation, but it is not claim-grade evidence and is not rerun.
- Browser evidence records one prompt, zero follow-ups and six uploaded files.
  The Pro response says it read seven attachment-only files and names an old BPNS
  receipt that the upload log does not contain. Preserve this as an attachment
  provenance discrepancy; do not rewrite the browser count or resubmit the turn.
- The only next task is `ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001`: keep all eight
  temporal tubelets and the R1 K64 carrier, keep blocks 0–5 dense, and in blocks
  6–11 use fixed adjacent spatial pairs to run full Q/K/V/MLP on N256, broadcast
  the block residual back to N512, and then run the existing dense Adapter.
- G0 is a pre-training six-block real-shape gate requiring p50 speedup `>=1.35x`
  and allocated/reserved ratios `<=1.05`. Only a full G0 pass opens the single
  seed-42 G1 training cell; only a full G1 pass opens matched full-stack G2.
  A failed gate is terminal and returns to a fresh Pro without rescue or sweep.
- The minimal implementation is clean and pushed at
  `0b734ab839973b2c945b012f066db8222d235bb9` on
  `codex/zoomtoken-gridfuse32-l6-v001`. Exact N16R4 checks passed in separate
  processes: GridFuse `9`, R1 regression `12`, and strict-rectangle `8` tests.
  A fresh Critic returned `PASS`; after binding G2 to the canonical G1 terminal
  checkpoint path, SHA256, epoch 59 and `state_dict_ema`, a fresh result-blind
  Evaluator returned `PRE_RUN_READY`. These are implementation/protocol facts,
  not G0 performance evidence.
- Precheck jobs `1262078` and `1262079` both stopped before tests because compute
  nodes could not resolve GitHub. The final launcher-only correction requires a
  login-node fresh-fetch and verifies exact clean HEAD plus the persistent
  remote-tracking ref on the compute node. Fresh Critic/Evaluator on the final
  candidate returned `PASS/PRE_RUN_READY`; no G0 action has run yet.
- Final precheck job `1262089` completed `0:0` with `PRECHECK_READY`. Exactly one
  formal G0 job, `1262090` (`zt-gf32-l6-g0`, one GPU, four CPUs), was submitted
  against exact candidate `0b734ab8…`; G1/G2 remain closed pending its terminal
  frozen gate evidence.
- Formal G0 job `1262090` is terminal `FAILED 2:0` after 15 seconds on `g0030`.
  Model construction stopped before warmup or any timing/memory measurement because
  `Rearrange` was absent from the mmengine transform registry. The exclusive terminal
  receipt exists, `profile.json` does not, and no G0 gate is evaluated. This is an
  engineering/protocol blocker, not GridFuse performance evidence. G1/G2 remain closed;
  it is not repaired under the old task.
- Fresh exact-Project Pro conversation `6a9494ad-dab4-83ea-83f6-e9cc2fabc722`
  returned `REVISE / CONTINUE_ONCE_WITH_EXACT-CONSTRUCTION-WITNESSED_G0_REPLACEMENT`
  and role contract `REVISE`. The old scheduler submission remains ordinal 1; the new
  task may submit exactly one scheduler-ordinal-2 / G0-measurement-attempt-1 replacement
  only after the exact production construction witness and fresh Critic/Evaluator pass.
  The unique task is `ZOOMTOKEN-GRIDFUSE32-L6-G0-CONSTRUCTION-WITNESS-AND-RPL1-v001`;
  G1/G2 remain forbidden and every terminal returns to a fresh Pro.
- The authorized minimal descendant is clean and pushed at
  `b5993faaaa59be318557ca314697e38c4b39b6a1`. It uses the canonical transform
  registry and one shared production preparation function; exact N16R4 GridFuse,
  R1-regression and strict-rectangle suites passed `12/12/8`. Construction-witness
  job `1262099` confirmed real detector construction and strict checkpoint load,
  then failed in the first dense real-shape ledger with `ragged Adapter temporal
  axis differs from pretrained Adapter`. No timing, memory, prediction, metric,
  gate or parameter update began. This is a second independent construction/shape
  blocker, not G0 performance evidence. The frozen task forbids another repair,
  Critic/Evaluator, replacement, G1 or G2 before a fresh exact-Project Pro decision.
- Fresh exact-Project conversation `6a949bec-1334-83ea-b410-a47ecdd451f7`
  verified the latest repository/branch/`b5993faa…` commit and returned `REVISE`.
  It permanently closes the incorrect 8-tubelet segment-G0 protocol and attributes
  the failure to a protocol-construction mismatch embodied as a harness/test defect,
  not to GridFuse science. The only task is now
  `ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-ATOMIC-G0-v001`: an actual
  B1/T384/K64/N24576, 48-bucket, dense-Adapter full-window atomic construction-plus-
  measurement job. It preserves the model/config/checkpoint/mechanism and original
  speed/memory gates. One final scheduler-ordinal-2/scientific-attempt-1 submission
  exists, with no replacement. Any construction/execution/artifact or valid gate
  failure permanently stops this exact route; a pass only returns to fresh Pro and
  does not automatically open G1.
- The sole production-full-window atomic G0 job `1262108` is terminal. Its
  construction witness, checkpoint strict load, exact full-window shape, dry/final
  ledgers, 100 warmups per arm, 500 alternating samples per arm, memory profile and
  terminal receipt are complete. Dense/candidate p50 is
  `178.500099/314.885696 ms`; frozen speedup is `0.56687268x`, far below `1.35x`.
  Allocated and reserved memory ratios are both `1.0`. Slurm `FAILED 3:0` is the
  launcher's fail-closed encoding of the complete gate failure, not a construction
  or artifact blocker. Status is
  `STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE`; no G1/G2, repair, replacement,
  ordinal 3, training or sweep is allowed. Fresh Pro review must bind the latest
  repository, branch and exact GitHub commit `dde46aee…` and independently choose
  the one next scientific task.

## Pointers

- GeoRoute implementation and gates:
  `research-wiki/experiments/georoute-adatad.md`.
- Current GeoRoute hypothesis and decisions:
  `research-wiki/ideas/geo-route-adatad.md`.
- Native-Crop S2 hold:
  `research-wiki/experiments/native-crop-s2-crop-sufficiency.md`.
- Full historical source and decision record: `research-wiki/log.md`,
  `research-wiki/decision_history.md`, `research-wiki/anti_repetition.md`,
  and `research-wiki/source_registry.md`.
