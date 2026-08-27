---
type: anti_repetition
updated: 2026-08-28
---

## GeoRoute deployment anti-repetition

本节首条给出当前 BPNS-R1 边界；其余内容按日期保留为失败路线与运行教训，其中“当前”“active”等字样只描述当时状态。

0. The current paper route is `ZoomToken-BPNS-R1`, not temporal reuse:
   current-only strict contiguous `8x8/K64` native support, full 12-layer VideoMAE-S
   and the existing Adapter, with no hidden/KV/cache/carry or depth bypass. Do not
   relabel it as preceding-frame reuse, resurrect stopped cache routes inside it, or
   start A-MoD-50/new seeds/K24/K18 as a prerequisite. Do not retrain K100 job
   `1248835`, R1 job `1249099`, DSR6 job `1252527`, MOD32 job `1252180` or DROP32
   job `1252179`. The next primary comparison is a same-hardware final-EMA K100-R1
   replay with full decode-to-NMS latency, memory, gross energy and boundary quality;
   theoretical token/block FLOPs are not speed evidence. DSR6/MOD32/DROP32 are
   secondary points and must not delay K100-R1. Formal replay job `1257281` is not
   cost evidence: it completed only the first K100 validation pass, then stopped when
   raw `mAP@0.7=46.246663` was compared with rounded history `46.27`. R1 and the
   remaining counterbalanced passes did not run. Do not interpret partial timing,
   memory or energy, and do not call this a model failure. The only admissible repair
   is to bind the raw accuracy and explicit rounding/tolerance rule, independently
   review it, and obtain new run authorization. Keep identity checks minimal: exact
   job/config/path/EMA and numerical parity are sufficient; do not add a new
   hash/provenance framework merely because a review listed one.
   That repair is now clean/pushed at `e9323448f6cd78b99bb3de53fd9ffb55f3676d65`;
   focused tests, independent Critic and result-blind PRE_RUN passed. Exactly one fresh
   formal replacement exists: job `1258299`, name `zt-bpns-r1-pv2-e9323448`, result
   root `/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_parity_e9323448_seed42_20260828`.
   It is running under the unchanged eight-pass protocol. Do not submit a duplicate,
   resume job `1257281`, read live/partial metrics, or infer cost before complete
   `profile.json`, `terminal_receipt.json`, predictions and power evidence exist.
   Do not apply the primary 5% efficiency gate directly to the profiler's pooled
   `comparison.r1_over_k100` fields. The Pro-frozen primary estimator first computes
   one end-to-end p50 and one total-energy value for each complete pass, then takes
   the four-pass median per arm. Recompute those values from `cost_samples.jsonl`
   grouped by `(arm, pass_index)`; retain the pooled profiler summary only as a
   separately labelled descriptive statistic. This uses existing raw evidence and
   does not authorize a rerun or code change while job `1258299` is active.
   Do not fabricate eight per-pass prediction files, eight standalone evaluator-vector
   files, a separate population receipt, quantitative power-coverage metadata, or
   temperature measurements after terminal state. The current code produces one
   canonical prediction per arm, verifies repeated predictions in memory, embeds
   per-pass evaluator vectors in `profile.pass_receipts`, and retains ordered raw rows
   sufficient to reconstruct the pass population and frozen estimator. Treat these as
   direct versus reconstructible evidence, not as identical artifacts. The reference
   source SHA exists, but prediction hashes do not; do not manufacture them post hoc.
   The complete command, data/checkpoint paths, package identity and success-side anomaly
   status are not fully serialized in `profile.json`; use the frozen start receipt,
   launcher and Slurm logs as separately labelled external provenance rather than
   pretending those fields were emitted by the profiler. Do not relabel
   observed latency or power order drift as thermal drift because temperature is not
   measured. Disclose that final video-level NMS cost is amortized across window rows.
   A fresh Pro review, not Codex post-hoc completion, must decide whether the terminal
   package is sufficient for a paper claim, diagnostic only, or needs a new bounded
   measurement.

0. The exact `FULL_REPRESENTATION_TEMPORAL_REUSE-v001` route is terminal
   `STOP_BEFORE_IMPLEMENTATION` at revision `bffff43d...`. Do not implement a
   block-11-only old-hidden copy, a fixed-mask per-layer KV/attention/MLP cache, or a
   whole-clip final-output memoizer and call it exact preceding-frame reuse. A 16-frame
   clip is only an attention bucket; the 12 Adapters operate on the global 384-tubelet
   lineage, so exact cache state is not clip-local. Eventful Transformers and
   STC-Cacher also occupy the broad gate/cache/selective-recompute contract. Do not
   create a config, PRE_RUN or 60-epoch cell for this stopped nonce, and do not rescue
   it with mapping MLPs, summary tokens, K24/K18, extra seeds or threshold sweeps.
   This is a design-level stop, not empirical evidence against all temporal reuse.

0. The user-provided `R-PADT-v0` report is not an accepted same-session Pro
   adjudication. The captured session receipt remains
   `TERMINAL_INCOMPLETE_NO_SCIENTIFIC_DECISION`; do not cite its external
   `CONTINUE` as Builder/PRE_RUN/experiment authority. Do not relabel R-PADT as
   direct preceding-frame full-representation reuse or exact KV cache: it runs a
   dense prefix, compresses only the suffix and copies anchor suffix outputs during
   restoration. Do not implement the exact `L_p=2/R=4/m=16/Q=4` tuple before a
   corrected human-frozen specification closes temporal identity, dense Adapter
   cost and the omitted STA/PVC prior art. Do not infer speed, energy or memory from
   `N'/N` or analytical FLOPs. Intake audit:
   `sources/2026-08-26-r-padt-v0-user-report-intake-audit.md`.

0. `IC-DRU`, `OW-ECR`, and `PCD-DRU` are terminal
   `STOP_BEFORE_IMPLEMENTATION` as currently defined. Their optimistic known-
   backbone saving bounds are about `50.12%`, `7.66%`, and `60.16%`, but the two
   dynamic routes require an unsupported ~25% average deep-refresh regime and
   reduce to known change-routing/cache/light-residual/MoD components; the exact
   overlap route lacks full-stack margin and adds about `54 MiB/sample` of
   twelve-layer state. Do not implement, train, tune K/rank/threshold, add gates,
   or rescue them with teacher/distillation. Do not claim that dynamic 20–30%
   refresh or all temporal reuse has been empirically disproved: no such model was
   trained. A future route must introduce a distinct error-control or execution
   principle and show conservative full-stack headroom before Builder work.

0. `R1-ACR16-Delta1-FKV` is terminal `STOP_BEFORE_IMPLEMENTATION`. The corrected
   Pro review found that Eventful Transformers (ICCV 2023) already covers token
   references/buffers, temporal change selection, identity scatter and sparse or
   incremental Transformer updates. ACR16's remaining low-rank delta plus
   conditional depth skip is an application combination, not a new reuse
   principle. Its verified main-block saving ceiling is `9.446%` and its known
   backbone arithmetic ceiling is about `8.80%`, leaving insufficient credible
   margin for both full-stack p50 latency and gross energy to improve by `>=5%`.
   Do not implement, train, tune, add ID/Delta/SHUF cells, or transfer Eventful
   directly as a rescue. Do not claim first temporal token cache or dynamic-region
   recomputation. This stop is local to ACR16/Eventful-transfer and must not be
   generalized to all temporal-redundancy research.

0. The strict-rectangle R2/R3/R4 matrix jobs `1249125–1249132` are all terminal
   `COMPLETED 0:0`; do not treat the old R4-SHUF15 intermediate
   `66.27/59.02/44.59` as its final result. Final R4-SHUF15 is
   `67.19/60.17/46.20`, versus R4 `68.02/60.32/46.26`, so
   R4−R4-SHUF15 is `+0.83/+0.15/+0.06`. It fails the preregistered
   `mAP@0.7 >= +0.30` ordering gate. Do not claim that learned frame-outside
   token ranking is effective, launch seeds/cost from that mechanism, or rerun
   the matrix. Q64-GLOBAL is terminal `67.84/60.66/45.39`; the crossed
   R4−Q64 result does not rescue the failed R4-SHUF15 gate.

0. The strict A-MoD reference is clean/pushed at
   `a41714e9f9271906a2eb4505e3fedc590c838055` and has N16R4 `8 passed` plus
   independent Critic `AUDIT_PASS`. Do not call it temporal reuse: it alternates
   dense and selected-token VideoMAE blocks within one forward pass and stores no
   state across frames or tubelets. Do not stop the temporal-memory objective
   because this reference exists, and do not retrofit temporal memory into it.
   The temporal route is now frozen as `APM32-CTX64` with one-tubelet detached
   pre-position patch memory, radius-2 mutual-nearest alignment at `>=0.80`, K32
   refresh/K64 context and exact K64 fallback; `CUR32-CTX64` is its matched
   current-embedding control. Do not substitute old same-index RC32 carry,
   hidden/KV cache, shallow transport, new loss or trainable router. The exact
   one-batch/full-state preflight implementation is clean at `e92df6a4…`; do not
   substitute CPU static tests for the still-missing APM and CUR two-GPU witnesses,
   and do not submit either 60-epoch arm until those witnesses close result-blind
   PRE_RUN and explicit run authority. Do not rerun, resume or relabel terminal
   DSR6 job `1252527`.
   For any later temporal arm, do not bind state to DDP batch index or assume that
   the same spatial index means the same physical content. Use explicit video/window
   identity and frozen alignment confidence; reset on video change, discontinuity,
   scene invalidation or missing state. Do not serialize live activation graphs or
   raw per-batch cache into the five-epoch recovery checkpoint.

0. `DSR6-KV` has one current executable candidate,
   `c6327a891809aa30370b3b2d9bedab0dcfe0d326`, on branch
   `codex/zoomtoken-dsr6-launcher-profile-v001`; the scientific implementation ancestor is
   `3260cd39154069138c6b1757326372cc3b73754e`. Fresh independent Critic is
   `AUDIT_PASS`, and result-blind PRE_RUN is `READY` after the no-data/no-model 2-GPU
   Slurm job-shell witness `1252525` completed `0:0`. The first seed-42 attempt job
   `1252521` and root
   `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_3260cd39_seed42_20260824`
   are sealed pre-data infrastructure failure: never resume, requeue, reuse in place or infer
   results from them. The proposed distinct root
   `/data/run01/sczc063/yuzibo/projects/zoomtoken_dsr6_c6327a89_seed42_20260824` and
   witness `1252525` used the earlier proposed job name and is never reused. The sole
   formal job is `1252527`, exact name `zt-dsr6-train-s42-c6327a89`, at the distinct root.
   It completed `0:0` after `06:07:17`, with epoch-59 checkpoint and retained
   recoveries `44/49/54`. Its immutable terminal stdout records the EMA evaluation
   `67.38/59.34/46.01`, below the frozen `68.57/60.64/46.07` thresholds on all
   three metrics. This is `STOP_DEPTH_ROUTE` for the original near-lossless claim,
   not an infrastructure failure. The 2026-08-25 user-directed joint
   accuracy–compute review retains DSR6 as a conservative Pareto candidate at
   `79.055%` block proxy. Do not submit another training cell, resume/requeue either
   job, reuse the witness, add a second split/seed, revive RC32 carry, or launch
   K24/K18. A future cost replay may only reuse the immutable final checkpoints
   for matched FULL64/DSR6/MOD32/DROP32 measurement; do not use the proxy as speed.

0. The corrected RC32-KV epoch at clean revision
   `813012620dca991ff90121d0d9faf688f303d1ef`, root
   `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_refresh_rc32_81301262_seed42_20260823T2100`,
   jobs DROP32/MOD32-KV/RC32-KV `1252179/1252180/1252181`, is terminal
   `COMPLETED 0:0`. Final-EMA Avg-mAP/mAP@0.7 are `66.11/44.88`, `66.50/45.21`,
   and `64.73/42.91`, versus read-only FULL64 `69.07/46.57`. RC32-KV fails the
   D−A/D−B/D−C gates and MOD32-KV also fails the original near-lossless gate.
   RC32-KV remains stopped because MOD32-KV is strictly more accurate at the same
   proxy cost. Do not submit a duplicate, add K24/K18 or another seed, resume the
   sealed `836f2ce4` cells, or revive carry. MOD32-KV and DROP32 may be included
   with DSR6/FULL64 in a read-only final-checkpoint cost replay; FULL64 job
   `1249099` and all three training cells must not be retrained.

0. The terminal ROI60 runs are exactly DN job `1245907` at clean revision
   `d2b5de05…` and ROI-only G job `1245924` at clean revision `59960255…`, seed
   3407, 60 epochs. Their final Avg-mAP/mAP@0.7 are `64.73/43.26` and
   `61.49/39.99`; G is a negative result against DN. Do not submit a duplicate,
   reuse an old DN checkpoint, add residual without a new causal rationale, or
   infer efficiency without full-stack cost evidence. Earlier jobs
   `1245897/1245898/1245908/1245909/1245910` are terminal implementation
   diagnostics and must not be resumed. The shared official job `1245842`
   is a separate untouched AdaTAD reproduction with terminal `68.73/47.24` and
   is not the DN cell.

0. Baseline-first correction: never label the matched-source dense outputs
   `66.42/67.14/65.99` (or any Q/U/R outputs compared against them) as an exact
   AdaTAD official reproduction. First verify the released AdaTAD checkpoint
   on the clean release config/evaluator against the published `69.03/48.27`
   anchor; do not explain the gap as seed or routing beforehand. See
   `WIKI_MEMORY_AUDIT-2026-08-17.md`.
   ZoomToken is the sole shared-baseline executor: other related TAD projects
   must consume its final receipt read-only and must not launch a duplicate
   checkpoint evaluation or clean official training. The shared number remains
   unbound while Q implementation/review/PRE_RUN preparation continues.

0. The active route is now the Hybrid-centered causal pilot, not Free-first.
   Do not submit another Free-ST selector cell or treat the old hierarchy's
   failure to promote Hybrid as a causal refutation of Hybrid.
0. Do not call fixed context8/ROI28/residual28 or exact K64 the final method.
   The user-restated target is temporally adaptive continuous ROI geometry,
   dynamic total `K_t`, and consequently dynamic role allocation—not merely a
   changing ROI/residual split under fixed K. The frozen nine-arm study is only a matched-budget
   mechanism probe; its result may select or reject components but cannot close
   the dynamic allocation objective.
0. The confirmed dynamic decision unit is the native two-frame VideoMAE
   tubelet: 384 ROI/budget decisions over 768 input frames. Do not redesign the
   heavy patch embedding for 768 independent raw-frame budgets or describe the
   tubelet-level policy as a raw-frame policy.
0. Dynamic budgeting is staged. Stage 1 must keep an exact configurable
   per-window total `sum_t K_t=B` and test only redistribution across tubelets
   and evidence roles. Do not add a Lagrangian, dataset-average expected-K
   target, or content-dependent window total until this stage passes and a new
   protocol is frozen.
0. Context allocation is fully dynamic. Do not reserve a fixed context count,
   enforce a deterministic per-tubelet context floor, or repair learned budgets
   with an undisclosed uniform scaffold. Any lower bound on total `K_t` is a
   separate design choice and must not be mislabeled as fixed context evidence.
0. Stage 1 permits zero-heavy-token tubelets. Do not silently clamp `K_t` to
   one, manufacture a fallback selected token, or count a scout/null carrier as
   heavy execution. The `K_t=0` main setting and a matched `K_t>=1` ablation
   must be separately named and trained.
0. The main `K_t=0` carrier is masked zero: a zero heavy feature accompanied by
   an explicit heavy-valid mask. Do not leak content through the main carrier,
   let a bias turn it into an unmarked pseudo-token, or charge it as executed
   heavy K. `learned-null` and `scout-projection` are independent, separately
   trained ablations; never enable either only at inference or mix their
   checkpoints with masked-zero results.
0. The approved dynamic allocator is a global exact-`B` physical-token
   projection. Do not implement it as independent per-tubelet top-K, a separate
   rounded count head, or top-B over role copies that can select one physical
   patch more than once. `K_t` and role counts must emerge from one constrained
   union with exact physical-token uniqueness.
0. The existing native packed VideoMAE path assumes a rectangular `[B,T,K]`
   route and equal selected-token counts across chunk batches. Do not pad empty
   or short tubelets/chunks with dummy heavy tokens and still report
   `sum_t K_t=B`, one exact-B compute path, or `K_t=0`. Requested, unique,
   padded, executed patch-embedding, attention, MLP, and Adapter counts must be
   separately receipted until a true ragged executor is validated.
0. Exact selected/executed `B` is a token-count contract, not an equal-FLOPs or
   equal-latency claim. Native VideoMAE attention also depends on the within-clip
   distribution: for every 16-frame clip record `b_c` and
   `P=sum_c b_c^2`, together with actual ragged bucket calls, executed
   patch-embedding/attention/MLP/coordinate-lineage-Adapter counts, and measured
   p50/p95. Do not hide padding or dummy tokens in those ledgers. Do not impose a
   fixed per-clip quota on the final main method merely to equalize `P`; retain
   full-window allocation and require a separate measured cost/Pareto gate.
0. Do not call Uni-AdaFocus training Plackett-Luce, REINFORCE, or straight-
   through routing. Its journal method replaces AdaFocusV1's three-stage RL with
   one-stage auxiliary supervision: detached global feature maps provide a
   differentiable spatial interpolation loss, and a decomposed Monte Carlo
   expected frame loss trains temporal weights. Hard local crop coordinates and
   sampled focus indices are detached in the official implementation.
0. An independent `q_ctx(t,n)` is not automatically required merely because
   context allocation is dynamic. In a support-only union where role IDs do not
   change heavy execution or pooling, unconstrained context and residual heads
   are non-identifiable. The approved main design therefore has no independent
   `q_ctx`: use one shared base utility with ROI/residual modifiers and treat
   context as the zero-modifier outcome. Reopening a separate context head
   requires a separately approved role-specific representation and ablation.
0. In the approved Scheme-A hard forward, compute
   `u_hard=q_base+max(0,delta_roi,delta_res)`, assign the operational role by the
   same argmax, and apply the unique global physical top-B. The backward-only
   soft score is a temperature-controlled log-sum-exp relaxation. Do not use the
   soft mass, expected count or duplicate role copies as executed B, and do not
   let a proxy-only route replace the exact hard detector forward.
0. Do not transplant Uni-AdaFocus's fixed glance/focus counts, resized local
   crop, classification frame proxy, full-frame-seeking size penalty, random
   second heavy branch, or validation-tuned early exit as the dynamic exact-B
   TAD method. Any borrowed surrogate must retain native physical-token hard
   execution, Stage-1 exact B, `K_t=0`, train-only supervision, and the existing
   measured cost gate; it must be separately named and ablated.
0. Scheme A fixes the Uni-inspired surrogate plus ST as the main dynamic
   estimator family. PL remains a separately trained matched ablation, not an
   inference switch, warm-start source, or automatic winner selected by the
   fixed-quota pilot. The immutable recovery may report A6/A7 evidence but
   cannot override the approved dynamic estimator family without a new design.
0. The approved main policy consumes `stopgrad(Z)` from the cheap scout. A
   train-only auxiliary TAD head may train the scout jointly from fit/train GT,
   but this is not an independently pretrained/cached coarse classifier and may
   not become actionness top-k. Detector loss remains active on the true hard
   exact-B path for every successful optimizer step; disabling its route bridge
   must not disable detector learning.
0. The backward-only soft-budget projection must operate on the same unique
   global physical candidates, satisfy `0<p<1` and `sum p=B`, and aggregate only
   detached scout features. Its proxy loss may update the route policy and
   training-only auxiliary head, but not the scout or heavy backbone. Soft mass
   is neither a hard selection, an execution receipt, nor a carrier for
   `K_t=0`, and it must be absent at inference.
0. Advance the proxy schedule by successful optimizer steps, enable it only in
   the early training phases, and anneal it to zero before the final hard-only
   phase. Do not keep an undisclosed permanent proxy objective, schedule it by
   raw forward/AMP retry count, or use its output as validation/test inference
   evidence. The main Stage-1 objective has no area, coverage, expected-cost,
   fixed-context, or fixed-`K_t` loss; exact hard B and measured cost remain the
   compute contract.
0. Do not describe the proposed in-bounds ROI mapping as mathematically
   unrelated to Uni-AdaFocus. At official commit `88464883`, Uni emits sigmoid
   top-left/size actions, maps height and width into `96..224` pixels for a
   224-pixel input, and maps the top-left coordinate into the residual legal
   interval. Rewritten in centre coordinates, this is algebraically the same
   bounded-interval family. The differences that require evidence are the
   minimum-size rule, native-token membership versus resized source crops, and
   the associated loss/gradient contract.
0. Uni-AdaFocus Eq. 15 explicitly pushes deformable crops toward the full frame;
   its official code implements this by penalizing size actions away from one.
   Do not transplant that classification-specific anti-collapse regularizer into
   exact-B native token routing: ROI area is not executed heavy-token count and a
   full-frame bias can erase ROI selectivity. The approved main setting derives
   `w_min=1/W_grid`, `h_min=1/H_grid` independently at runtime and uses no
   size/area/coverage/smoothness penalty. Do not replace the two axis-specific
   floors by `max(1/W_grid,1/H_grid)`, which silently enlarges the rectangle.
   Do not claim the one-cell floor is proven optimal: the only approved scale
   sensitivity intervention is a separately trained, otherwise matched 2x2-cell
   floor arm. Never select 2x2 post hoc on a main checkpoint or use either floor
   to repair a collapsed allocator at inference.
0. Decoded SCNR `w,h` are full rectangle extents because the bounded centre uses
   `w/2,h/2`.  A signed ellipse modifier must therefore normalize offsets by the
   semi-axes `w/2,h/2`, not by `w,h`.  Job `1215355` used the latter mismatch and
   collapsed operational roles to context/ROI/residual `0/24573/3`; preserve it
   as a diagnostic and never promote its nominal P0 PASS.  Corrected source
   `dfcbe692` receipts
   `signed_ellipse_with_semiaxes_half_decoded_full_extent`.
0. Nonzero context/ROI/residual counts, a broad `K_t` range, exact B, finite
   losses and nonzero gradients in P0 or policy-health are admission evidence,
   not ROI/residual complementarity, localization quality, cost saving or model
   performance.  Jobs `1215358`, `1215363` and `1215364` opened no such claim.
0. `K_t=0` is an allowed state, not a required event in every finite trace.  Do
   not invalidate an otherwise exact dynamic run because its observed minimum
   is positive; prove zero handling with known-answer/empty-clip tests and report
   the observed zero count without post-hoc pressure to manufacture zeros.
0. The paired 1-cell/2-cell CUDA P0 passes are not an empirical floor verdict.
   Job `1215364` proves that the matched G2 configuration is mechanically
   executable only.  Floor selection requires complete matched M2 development
   results and cost ledgers; role counts or synthetic losses cannot choose it.
0. Dynamic diagnostic telemetry from source `7e5775e8` is an accuracy-replay
   mechanism receipt, not a timed-cost instrument or a performance result.  It
   must run with local batch one on the complete matched evaluation population;
   keep it disabled during training and the separate timed replay.  M2 protocol
   source `ec8de9f51f85fc81031d82b79e30019d57a381b4` freezes identical data/order/
   successful updates/EMA/evaluator plus the full decode-to-NMS p50/p95,
   peak-memory and gross-energy scope.  Do not submit it until that exact clean
   source passes remote Linux tests and every deployment `PRECHECK_ONLY` gate.
0. M2 cost evidence is valid only when both complete trained-arm receipts feed
   one physical-GPU serial `G1 -> G2 -> G2 -> G1` replay with accuracy telemetry
   disabled, identical full-center population hashes, one continuously sampled
   20-ms NVML trace, and raw monotonic energy/NMS windows that the validator
   reintegrates.  A model-only timer, self-reported energy scalar, partial pass,
   different GPU/job, or hand-combined profile is not M2 cost evidence.
0. Never reuse M2 deployment root
   `scnr_dynamic_floor_m2_9d6641a6_s3407_20260804_0507`. N16R4 rejected its
   CPU-only finalizer during `sbatch --test-only`; zero Jobs were created and the
   root contains only `control/storage_preflight.json`. On this GPU-only site an
   M2 finalizer must request one GPU/one CPU, disclose that GPU as scheduling
   overhead, and perform no model or cost computation. Replacement source
   `bad14693daa1fe414e56bf697c617e76f96eed48` requires a fresh exact-source
   remote regression, precheck and namespace; the old root is not resumable.
0. The active M2 namespace is
   `scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525`, exact runtime
   `6ee97336775a09611f10423e07cafcea375e191a`, Jobs
   `1216180/1216181/1216182/1216183`. G1/G2 completed successfully; paired-cost
   failed only because the profiler requested nonexistent
   `packed.attention_pairs`, and finalizer sealed
   `INCOMPLETE_NO_FLOOR_INFERENCE`. Do not retrain, resume, replace, or tune either
   arm; do not hand-read their metrics or treat the incomplete finalizer as a
   floor result. Preserve the failed cost/finalization attempt. The only
   admissible recovery reuses the hash-valid G1/G2 stage artifacts, runs the full
   same-GPU `G1 -> G2 -> G2 -> G1` replay from a separately clean, non-model
   execution-repair commit that records model runtime `6ee97336`, and then runs a
   fresh fail-closed finalizer. The original clean runtime checkout must remain
   untouched.
0. Recovery Jobs `1222672/1222673` are terminal incomplete, not evidence: the
   profiler attached an `nn.Module` forward hook to the sparse adapter while the
   real route directly called `forward_ragged`, so `sparse_adapter_ms` was zero.
   Preserve their artifacts as `cost_failed_job1222672/` and
   `control/finalization_incomplete_job1222673.json`; do not reuse execution
   commit `c67e13e8` for cost.
0. Recovery Jobs `1222700/1222701` are terminal incomplete, not cost evidence.
   Although all four timed passes ran, the final validator reconstructed a cost
   config without the producer's forced `post_processing.sliding_window=True`, so
   `pass_receipt.cost_config_sha256` mismatched. Preserve
   `cost_failed_job1222700/` and
   `control/finalization_incomplete_job1222701.json`; their raw samples, power
   trace and sidecar cannot be hand-assembled because the fail-closed protocol did
   not persist the pass receipts and complete profile-level provenance. Never
   patch hashes, manually synthesize `paired_cost_profile.json`, or interpret its
   raw pass values.
0. Recovery Jobs `1222869/1222870` are terminal incomplete, not cost evidence.
   Population preflight retained one call to the removed `_cost_config` helper,
   raising `NameError` before the `cost/` directory was created. Preserve
   `logs/dfm2_cost_recovery3.1222869.err` and
   `control/finalization_incomplete_job1222870.json`; there are no pass samples or
   cost artifacts to salvage. Do not reuse execution commit `011d2943` for cost.
0. Cost Job `1222889` is terminal `COMPLETED 0:0` and its immutable profile passes
   the validator from cost execution `42923d9f`; do not rerun or replace this cost
   pass. Finalizer `1222890` is terminal incomplete because it ran the frozen model
   runtime's old validator, whose reconstructed cost configs omitted
   `sliding_window=True`. Preserve
   `control/finalization_incomplete_job1222890.json`; this failure invalidates only
   that finalization, not the current profile. Do not interpret profile values
   before a fresh valid finalizer.
0. The final M2 recovery was finalizer-only Job `1223310` from exact clean
   execution source `75e2adc86877f002e10626ee4011104b60b0ce49`. Its receipt binds
   model runtime `6ee97336`, cost execution `42923d9f`, and finalizer execution
   `75e2adc8` as distinct identities; the existing cost profile is validated
   against its own execution commit. No arm retraining/resume or cost replay is
   allowed. It completed and sealed descriptive-only PASS. Do not collapse these
   three commits, run another finalizer/cost replay, or treat the terminal PASS as
   a single-seed floor selection, official-test opening, or paper result.
0. M2 does not validate an operational three-role Hybrid. G1 selected
   context/ROI/residual `0/7/3,342,329`, while G2 selected
   `0/0/3,342,336`. Never describe those runs as ROI/residual complementarity or
   use the large G1 accuracy lead to hide role collapse. `K_t=0` is operational,
   but the role mechanism is effectively residual-only.
0. Neither M2 floor was active in the accuracy replay: both width and height
   floor-saturation rates are zero and all observed extents exceed the bound.
   Therefore `+5.78 pp` Avg-mAP and `+6.22 pp` high-IoU for G1 over G2 are
   single-seed descriptive training contrasts, not causal proof that the 1-cell
   lower bound is better. Do not launch floor-selection M3 before modifier-scale
   and role-margin diagnosis restores an identifiable dynamic Hybrid.
0. Do not repair M2 role collapse with fixed context/ROI/residual counts, target
   role fractions, post-hoc token reassignment, or a new independent `q_ctx`
   head. First measure all-valid versus selected `delta_roi`, `delta_residual`,
   zero-baseline wins and top1-top2 margins out of band, preserve exact B and
   prediction hashes, then test minimal scale-identifiability interventions as
   separately trained ablations.
0. Do not salvage role telemetry from Jobs `1223615/1223616`, `1223625/1223626`,
   or `1223640/1223641`. The first pair closed against the wrong output root, the
   second against the wrong schema, and the third failed exact source prediction
   SHA parity. Earlier Jobs `1223595/1223596` and `1223601/1223602` failed before
   inference. None supplies valid modifier-scale or role-margin evidence.
0. The `2c39ce58` source mismatch is not a JSON-order-only difference: both
   replays have the same 40-video key set and 80,000 records, but exact candidate
   identity overlaps only `76,660/80,000` (G1) and `78,387/80,000` (G2). Do not
   inspect the sealed role telemetry or blame instrumentation without a matched
   causal control.
0. Do not repeat the same-GPU neutrality or strict triplet integrity runs. Pair
   Jobs `1223686/1223687` establish OFF/ON neutrality. Legacy triplet Jobs
   `1223707/1223708` show OFF-A/OFF-B prediction drift despite exact route hashes;
   strict math-SDPA Jobs `1223727/1223728` make OFF-A=OFF-B=ON exactly. The cause
   is downstream memory-efficient CUDA SDPA replay nondeterminism, not role-route
   mutation. Strict backend output is not historical prediction parity.
0. Only the `ede8af53` field-minimized categorical bridge may cross that backend
   boundary. It authorizes hard role categories, for which all 136 legacy/strict
   windows match exactly. Never use it to recover continuous modifiers, margins,
   geometry, predictions, mAP, cost, floor causality, or complementarity.
0. Authorized categorical evidence shows pre-top-B residual dominance:
   all-valid G1 `0/2,671/11,486,609`, G2 `0/984/11,488,296`; selected counts retain
   the M2 collapse. Do not blame global top-B for squeezing an already diverse
   role partition.
0. The first model-repair candidate may change only the additive residual offset
   by subtracting its differentiable all-valid full-window mean. Do not combine
   the first probe with per-tubelet centering, RMS/temperature/bounding,
   ROI-conditioned complement, fixed quotas, target fractions, reassignment,
   `q_ctx`, a new loss, or a budget change.
0. A frozen-checkpoint centering probe is mechanism evidence only. It must use
   strict deterministic replay and preserve exact B, fully dynamic K including
   zero, true ragged execution, masked-zero carrier and no-leak routing. If both
   ROI/context are not reachable among valid candidates or no non-residual token
   is selected, stop before training. A pass authorizes only a new matched
   development-training protocol, not M3, official test, efficiency or a claim.
0. Do not repeat the residual-window-centering frozen-checkpoint probe. Exact
   source `091f9f9b` and Jobs `1223783/1223784` passed duplicate strict replay,
   exact-B/ragged/no-leak and structural context/ROI reachability in both frozen
   M2 arms. This is terminal structural evidence, not an mAP result.
0. The only authorized next performance question is fresh G1-anchor
   `none_control` versus fresh G1-anchor `residual_window_center`. Do not relabel
   centering as G2, select a floor, reuse an old M2 checkpoint as a control,
   resume either cell, add a role-fraction target, or combine a second repair.
0. Matched centering training requires both complete 60-epoch/9,600-update cells
   and same-GPU strict math-SDPA duplicate Gate replay. Any config, population,
   checkpoint, raw-prediction, route-payload, exact-B, ragged, role, or no-leak
   failure yields empty contrasts; partial results are not interpretable.
0. The seed-3407 centering screen passes only if centered mAP@0.6 and mAP@0.7
   are each strictly greater than control and centered Avg-mAP is not lower.
   A pass authorizes only a separately frozen ABBA+BAAB paired full-stack cost
   study. It does not directly authorize seeds 3408/3409, M3, official test,
   complementarity, efficiency, Hybrid efficacy or a paper claim.
0. Do not define matchedness by a hand-picked field list. The centering study
   must derive both cells from one complete G1 config and match a normalized
   complete-recipe hash covering optimizer, losses, augmentation, scheduler,
   detector/head, data and execution settings. Only path/receipt identity and
   the registered calibration mode may differ.
0. Do not submit the two training leaves independently or bolt on a finalizer
   afterward. The atomic deployer must first submit both stages held, submit one
   held `afterany` finalizer over their exact Job IDs, persist and revalidate the
   immutable deployment receipt, and only then release all three Jobs.
0. Do not inspect or contrast one surviving centering cell. Any absent, failed,
   malformed, nonterminal or duplicate-divergent arm forces empty contrasts and
   `INCOMPLETE_NO_PERFORMANCE_INFERENCE`; no resume or replacement inside that
   namespace is permitted.
0. The only active matched-centering namespace is
   `scnr_residual_centering_matched_training_16137484_s3407_20260806_061352`,
   exact runtime `16137484c5cc`, stage Jobs `1223819/1223820` and after-any
   finalizer `1223821`, deployment SHA `71b10681...`. Both P0 gates passed and
   both trainings entered epoch 0. Do not submit a duplicate namespace, replace
   one leaf, treat a later documentation commit as runtime, or read live/partial
   metrics as evidence.
0. The aggregate G1 end-to-end p50/energy disadvantage is cold-order-sensitive:
   model-forward p50 differs by only `+0.438%`, and G1 pass 0 is the host/input
   outlier. Never claim that the 1-cell model intrinsically costs `+2.845%` from
   this ABBA aggregate. A later cost confirmation must mirror ABBA with BAAB (or
   otherwise randomize first-arm cold state) and keep the same physical GPU.
0. Do not say “Hybrid has been proved effective” or “ROI and residual are
   complementary.” The old Hybrid result is single-seed descriptive evidence
   confounded by role split, scorer family, ST, and representation. Only the new
   all-nine study may generate exploratory mechanism evidence.
0. `Fixed K64` means deterministic row-major uniform lattice selection. It is a
   non-learned coverage control, not an optimal support, learned lower bound, or
   proof that Hybrid's complete combinatorial state space is smaller.
0. Structured residual likelihood must condition on the complete sampled ROI
   set and exclude invalid, context, all ROI, and preceding residual choices.
   Never reuse a single-family likelihood or sum independent unconditioned
   top-K terms and call it the Hybrid joint policy.
0. Structured route randomness is private and keyed by study seed, successful
   update, rank, and role. It must not advance global CPU/CUDA RNG. AMP retries
   reuse the successful-update key; never key by optimizer attempt.
0. Policy risk accepts exactly finite scalar `cls_loss` and `reg_loss` at local
   batch one. Never iterate arbitrary loss tensors or include cost, policy,
   geometry, coverage, or unknown auxiliary losses in the advantage.
0. Support-only means pretrained VideoMAE absolute position ON, external
   absolute coordinates OFF, ROI-relative coordinates OFF, geometry projection
   and side channel OFF, uniform pooling, and zero geometry regularization in
   every arm. Do not reintroduce Fixed+geometry or representation-on Hybrid in
   this matrix.
0. Geometry shift127 is a cyclic permutation over the 384-tubelet temporal
   trajectory, `pi(t)=(t+127) mod 384`, before ROI logits. It is not an 11x20
   spatial shift. A tie is mechanism-ambiguous, especially if geometry is
   time-invariant; it is not a universal disproof of ROI support.
0. Never time diagnostic route telemetry inside the admission p50 forward.
   Accuracy/telemetry and cost profiling are separate complete replays; route
   hashing or CPU transfer in the timed pass confounds Dense versus K64.
0. `georoute_hybrid_causal_pilot_v1` is seed5227, 20 epochs, exactly nine arms.
   Do not run a subset, resume a cell, reuse an old checkpoint/prediction, or
   interpret survivors. Missing/invalid cells require empty contrasts.
0. The nine stage Jobs `1213694--1213702` completed, but finalizer `1213703`
   sealed `FAIL_UNTRUSTED_FINALIZER_INPUT` because it compared JSON mapping
   insertion order against arm order after canonical sorted-key serialization.
   Do not hand-read metrics, rewrite the deployment receipt, remove the sealed
   failure, or rerun inside the old namespace. A recovery must be a new
   versioned exact source and output namespace, bind every immutable old
   artifact/hash/Job, compare the stage key set and then explicitly iterate the
   frozen `arm_order`, and preserve empty contrasts on any mismatch.
0. The single-seed screen has no automatic scientific winner and no
   multiple-comparison-adjusted claim. It can only admit a separately frozen
   confirmatory protocol. Official test and paper claims remain closed.
0. If a future confirmatory result is used to claim PL-over-ST, retain matched
   Hybrid-ST across all confirmatory seeds. Omitting ST requires deleting the
   estimator-superiority claim even if the exploratory A7 >= A6 check passed.

0. Do not treat a Pro proposal as an implementation receipt. The 2026-07-29
   CER-TAD review correctly diagnoses Free v1 and motivates complementary
   evidence routing, but its dynamic role-count likelihood, critic, boundary
   head, coverage/stability losses, and numerical weights are underspecified.
   They remain `discussed` until the estimator/representation preexperiment
   passes.
0. Do not reuse the old seven-arm Free-first selector or its failed namespace
   for a new CER or estimator study. A changed arm set requires a new study ID,
   contract, selector, source commit, and namespace.
0. Do not claim a support-selection gain while absolute coordinates,
   ROI-relative coordinates, or the geometry projection differ between arms.
   These three representation paths must be independently switchable and
   matched.
0. Do not adopt the review-proposed `+0.50 pp` / `+0.30 pp` accuracy margins as
   confirmatory gates. They were proposed after the old development results and
   lack independent variance/power justification. Pilot estimates are
   exploratory and must be separated from confirmatory seeds.
0. Instrumentation replay is valid only in a new diagnostic namespace with
   exact prediction-SHA and population parity. It cannot repair the old ROI
   decode failure, complete the old selector, or create paper evidence.
0. D/K/M finalization
   `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
   authorizes only the independent six-arm exploratory pilot. Never reinterpret
   `GO_PILOT_DESIGN_ONLY` as CER implementation, P2/P3 authorization, official
   test permission, or paper evidence.
0. In `georoute_estimator_representation_pilot_v1`, keep
   `absolute_position_enabled=true` in all arms. Representation-off means only
   the three new detector-visible paths are off: absolute source coordinates,
   ROI-relative coordinates, and geometry projection.
0. The exploratory pilot has no automatic winner. Do not use one seed or
   post-result margins to promote an arm. First report the four frozen
   contrasts; only then may its variance inform a new protocol with disjoint
   confirmatory seeds.
0. Never resume or interpret
   `georoute_estimator_representation_pilot_02b6efe7_20260729_1805`.
   P0 Jobs `1203380`--`1203385` failed mechanically before any model result;
   no training leaf ran and no performance conclusion exists. A repair requires
   a new commit and namespace.
0. The fresh pilot namespace
   `georoute_estimator_representation_pilot_cbe0a082_20260729_1849` is bound
   to runtime `cbe0a08218a2f4550960f7c832f88c8cf77757c1` and its sealed P0
   suite. Job `1203715` hard-failed at real batch 0 after eight AMP retries and
   produced no checkpoint or metric. Never replace that arm, resume any
   arm, or interpret an epoch log or the other five completed leaves as a
   partial result. All-terminal finalizer `1203720` emitted only
   `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with an empty contrast set
   (self-hash `738e9875de2e9e08408263fd7d359e60f5ba1ca1912d0fbb9062a462c58cbf3a`);
   the four contrasts require a new exact commit and full six-arm namespace.
0. Do not repeat the FP16 production-horizon PL reduction. A finite
   per-tubelet likelihood can overflow when summed over `T=384`. Half/bfloat
   likelihood evaluation and the unchanged sum-then-batch-mean reduction must
   run in FP32. Before a replacement pilot, a clean remote P0 must bind its AMP
   KAT to the actual `180x320`, floor-native `11x20` grid (`N=220`) and `K=64`,
   prove an objective magnitude above FP16 range, and prove finite scaled
   gradients. A float64 `T=1` KAT or a mismatched `160x160` source gate is not
   sufficient.
   Exact repair `30f9ca6f` has now passed remote Linux `120/120` and standalone
   CUDA KAT Job `1203873`; do not reinterpret that mechanical/numerical pass as
   detector utility. Only a fresh six-arm namespace whose own schema-v4 P0 and
   all six stage receipts pass can supply the four descriptive contrasts.
0. The independent-agent verdict
   `DEPLOY_AFTER_OLD_CLOSEOUT_AND_CAPACITY` is a code/protocol audit, not a
   runtime authorization that bypasses either gate. Never submit fewer than
   the complete 14-job DAG to fit current capacity. The old `cbe0a082` closeout
   has now sealed INCOMPLETE, but that does not bypass the fresh capacity or P0
   gates; never treat reviewer approval as a P0 or performance receipt.
0. The replacement namespace
   `georoute_estimator_representation_pilot_30f9ca6f_20260729_2023` is sealed
   incomplete. It is bound to exact runtime `30f9ca6f`,
   P0 Jobs `1204015`--`1204020`, suite self-hash
   `2aea448be4c8d72957b3c904bb22c5ae39689cb0010c3b18a4914bd71f5265ec`,
   stage Jobs `1204022`--`1204027`, and closeout `1204028`. Never mix any old
   `cbe0a082` artifact into it, manually replace a stage, interpret a partial
   stage, or bypass the all-six closeout. Job `1204028` emitted only
   `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE` with empty contrasts
   (self-hash `60c9dab575e65830b7b849437963de2c7f789743caedb130b499c142c49c76ab`).
0. The latest candidate performance namespace is sealed incomplete:
   `georoute_estimator_representation_pilot_c822add3_20260729_2149`, exact
   runtime `c822add335c38a9f6c63e609237c4bfa9b9f468d`. It contains P0
   `1204301`--`1204306`, P0 finalizer `1204307`, stages
   `1204308`--`1204313`, and closeout `1204314`. Its six P0 leaves and finalizer
   passed mechanically, but residual-PL stage `1204309` exhausted all eight AMP
   retries on real batch 0 and failed with no checkpoint, metric, or stage
   result. Closeout `1204314` completed `0:0` and sealed
   `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, false all-six, empty contrasts,
   and all promotion guards false (self/file SHA-256
   `a02e551ba9007b49670103e2e4db3bf1c1d917cb5a7bb5c4dd724274b9379a2a`
   /
   `c95c1694dccbda2687b1b9e6e07bb9016ebe80181e2288d172874afa791d8f1c`).
   Never mix either earlier namespace into it, rerun or resume the failed arm,
   bypass the all-six decision, or interpret the other five arms.
0. Do not confuse cumulative AMP retry telemetry with the per-batch hard-fail
   rule. ROI-PL representation-on/off Jobs `1204313/1204312` each logged 11
   cumulative failed optimizer attempts, and both reached scale `64`, but each
   replayed the affected batch successfully and completed all 20 epochs. The
   experiment source hard-fails only when one batch exhausts all
   `max_amp_retries_per_batch=8` retries without a successful optimizer update.
   The generic monitor heuristic `count>10` is an alert for clustered numerical
   stress, not a preregistered or finalizer-enforced invalidation threshold.
   Their outputs remain unusable here because the all-six namespace is
   incomplete, not because cumulative retry count itself formally failed them.
0. Schema-v4 P0 from source `30f9ca6f` did not test full-graph AMP. Its model
   forward/backward was FP32, and the `T384/N220/K64` AMP KAT used an isolated
   logits leaf. Job `1204023` then exhausted all eight real-batch AMP retries at
   scale `256`. Never accept an isolated logits KAT as an optimizer-update gate.
   A replacement must run the actual model under autocast plus GradScaler,
   unscale and inspect all required trainable gradients, and prove a successful
   optimizer step at the registered floor scale before releasing any stage.
   The implemented replacement is backbone schema v4 plus P0 schema v5:
   scout/route FP32 outside autocast and a full-model GradScaler-256
   zero-learning-rate step. Never weaken this back to an isolated logits gate
   or silently add normalization/clipping to make it pass.
0. Full-graph CUDA P0 Job `1204087` from exact source `c822add3` passed schema
   v5, but it remains a synthetic numerical gate with zero checkpoints and no
   metric. Never promote it to real-batch stability or performance evidence.
   Do not start a replacement six-arm DAG until old closeout `1204028` seals
   `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`, and never reuse any of the five
   surviving `30f9ca6f` arm outputs in the future namespace.
0. The fresh namespace's own schema-v5 suite also passed
   `PASS_MECHANICAL_ONLY` (self-hash
   `f6f423670c9c2417aadfca97c67d794427ee337c359ba2d2509faee53a5ccdb6`),
   yet real-batch residual-PL Job `1204309` still failed after scales
   `32768,16384,8192,4096,2048,1024,512,256`. Therefore neither the standalone
   nor per-arm synthetic full-graph gate may be treated as a real-batch
   stability certificate. The namespace is preserved and closeout `1204314`
   has emitted only `PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE`. Make no further
   numerical repair or replacement run without a new real-batch cause analysis
   and explicit experiment decision.
0. Do not claim an exact replay of residual-PL Job `1204309`. The failed run
   did not persist sample indices, full input tensors, sampler state or complete
   pre-forward RNG state. A new run can be a deterministic same-config,
   same-data-path reproduction only, and must fingerprint its own input and RNG
   state before interpretation.
0. A diagnosis-only retry below loss scale `256` is not a training protocol or
   performance result. It may localize whether finite unscaled gradients
   overflow only when scaled, but its retry limit/floor cannot silently enter a
   scientific run. Any resulting repair requires a fresh production-path
   real-data stability gate.
0. Do not put the 20-epoch, seed-3407 estimator pilot or its model-only cost in
   a paper table. Paper eligibility requires an exact official AdaTAD
   reproduction plus matched native-source dense control; matched
   split/windows/padding, effective batch/updates/schedule/AMP/EMA,
   evaluator/NMS; disjoint multi-seed confirmation; sealed official test; and
   selector-inclusive decode-to-NMS latency, memory and energy.
0. Real-batch AMP telemetry is opt-in diagnostic instrumentation. Never enable
   `georoute_amp_diagnostic_enabled` or attach its observer in a production or
   official-comparable run, and never reinterpret a diagnostic loss scale,
   gradient group, retry count or repair authorization as accuracy evidence.
0. A diagnostic child must explicitly bind both its own exact runtime commit
   and the sealed failed parent's distinct runtime commit plus canonical file
   hash. Wrapper-failure evidence is usable only when its self-hash, arm,
   expected runtime, observed runtime, Slurm Job ID and zero-performance guards
   all validate.
0. Do not place the current GeoRoute development config beside an official
   AdaTAD number. Development-only population, batch `1`, warmup `2`, disabled
   evaluator/NMS and final-only checkpointing are not the official recipe.
   Paper comparison requires a separately frozen official reproduction and
   same-recipe native-source dense/GeoRoute arms.
0. Do not interpret OpenTAD config `batch_size=2` as two samples per GPU.
   `build_dataloader` divides that job-global value by world size; the frozen
   two-rank official-comparable recipe is local batch `1`, global batch `2`.
   Any run or table using local batch `2`/global batch `4` is a different
   optimization protocol. The F0 single-rank batch-2 leaves are deliberately
   stronger resource stresses and explicitly not a full world-size-two recipe.
0. Do not put F0 preflight/KAT output, an individual F1 cell, or a partial F1
   matrix in a performance table. F1 inference exists only if all 15
   dense/fixed/random/ST/PL x three-seed cells and all artifact/population
   bindings pass. Even a complete F1 result is Fit/Gate development selection,
   not official-test or paper evidence. It cannot establish end-to-end
   efficiency because its profiler excludes decode, evaluator, energy and
   full-system orchestration.
0. Do not select NativeTokenSelect from averages alone. Every paired seed must
   beat fixed and random at mean(mAP@0.6,mAP@0.7), and every paired seed must
   have lower matched development model-plus-postprocess p50 than dense. ST
   versus PL additionally requires non-inferiority on both axes for every seed
   and strict improvement on both mean axes. Any tie, crossing, missing cell or
   population mismatch means `HOLD_NO_OFFICIAL_TEST`. Geometry is not an F1 arm.
0. Do not resume or reinterpret F0 namespace
   `georoute_official_comparable_preflight_v1_4a03339b_20260731_1145`.
   World-two KAT Job `1209274` never entered Python/CUDA; Slurm rejected its
   inner 192-GB step because the outer allocation did not reserve matching
   memory. PL/ST terminal outputs in that namespace cannot compensate for the
   missing KAT. N16R4's submit Lua rejects explicit `--mem` overrides and
   assigns 55 GB per requested GPU, so the only admissible successor is a
   fresh exact-source/fresh-namespace F0 that requests two GPUs once and lets
   the inner KAT inherit that allocation. This is a deployment repair, not
   evidence about ST, PL or mAP.
0. Do not turn the successful replacement F0 into model evidence. Exact source
   `3d8c2b48`, Jobs `1209309`--`1209312`, and finalization
   `313da95faeae9e600965fe4ac5c7ad5816f652d5ff2c97cf9734f7028d888a3c`
   authorize only one complete all-at-once F1 development matrix. The F0 has no
   metric/checkpoint/prediction/evaluator/test surface. F1 must not be split or
   partially submitted to evade `MaxSubmitJobs`, and its 15-cell conservative
   training storage gate must not be replaced by F0's no-artifact profile.
0. Do not mistake “zero AMP skip at initial scale 65536” for an official
   AdaTAD requirement. The official config uses the default dynamic
   `GradScaler`; with no formal binding, the legacy train path permits a skipped
   update and advances normally after the scaler backs off. Stability-v1
   `1205033/1205034` deliberately imposed a stricter 32-batch zero-skip rule and
   rejected both PL (batch 3, scout score-function) and ST (batch 21, detector).
   That HOLD cannot reject either estimator or authorize a paper protocol.
   Any replacement numerical gate must be versioned, use an independent data
   order, match the intended official AMP semantics, keep all performance/test
   surfaces closed, and never silently relax the sealed v1 rule.
0. Stability-v2 is a new contract, not a corrected v1 receipt. Its frozen
   profile uses seed/order `4417`, 64 consumed batches, the default dynamic
   GradScaler, zero retry/replay, scheduler and EMA advancement on every batch,
   at most two nonconsecutive skips, scale floor `16384`, and a successful
   final-16 tail. Never edit v1, reuse paper seeds `3407/3408/3409`, tune these
   thresholds after seeing v2, or call a 64-batch prefix a full official run.
   The implementation deliberately records
   `official_scheduler_hyperparameters_matched=false`,
   `full_official_recipe_matched=false` and
   `official_performance_comparable=false`. Never cite a v2 PASS beside an
   official AdaTAD mAP number; it can authorize only freezing a later
   same-recipe paper experiment.
   The only v2 namespace is the sealed terminal run
   `georoute_official_semantics_amp_stability_v2_27fba03c_20260730_0800`
   from runtime source `27fba03c`, Jobs `1205588/1205589/1205590`. Do not
   resume, replace one arm, tune thresholds from its live telemetry, or use
   later docs-only commits as its runtime source.
   PL crossed the frozen rule with skips at batches `11/20/29` and final scale
   `8192`; ST ended at its limit with skips at `20/29` and scale `16384`.
   Finalizer `1205590` sealed
   `INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2 /
   OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD`. Do not reinterpret PL's later
   successful final-16 tail as recovery, treat ST's arm-only PASS as a pairwise
   gate PASS, or launch a formal performance run from this HOLD.
0. Do not repair PL, drop PL, freeze ST, lower the initial loss scale, disable
   FP16 communication, clip the advantage, or change temperature/weight/K from
   the sealed stability-v2 HOLD. The only authorized successor is the new
   seed-7367 matched gradient-decomposition diagnosis. It must retain the
   authoritative standard FP16 hook and observe it through a detached wrapper;
   any observer mutation of a bucket or replacement of the hook Future makes
   the result incomplete.
0. Do not require PL and ST CUDA RNG states to remain identical after their
   first matched batch start. PL Gumbel sampling consumes CUDA RNG while ST does
   not; all-batch equality would require forbidden replay/reset or an estimator
   change. For the gradient-decomposition study require all-batch data and CPU
   RNG equality plus batch-zero CUDA equality, then record the expected later
   divergence. Do not use that corrected gate outside this versioned study
   without a new preregistration.
0. The gradient-decomposition KAT is a hard parent, not an optional unit test.
   It must run under Slurm on one visible GPU at the exact clean runtime commit,
   exercise a real world-size-one DDP/GradScaler/standard-FP16-hook backward,
   prove the observer leaves the pre-hook bucket unchanged, detect an FP16
   shadow-cast overflow, and seal a self-hashed no-performance receipt. Do not
   submit the two-arm DAG if this KAT is missing or failed.
0. The sealed stability-v2 receipt names dynamic-scaler non-update batches
   `summary.skipped_batch_indices`; it does not expose
   `summary.failed_batch_indices`. Any successor provenance reader must validate
   the former as a sorted unique nonnegative list. The `664180b6` deployment
   admission failure occurred before namespace creation and `sbatch`; never
   fabricate, alias, or silently default this parent field.
0. The only gradient-decomposition DAG is the sealed exact runtime
   `33f721be83e0ad7f7a36e853491e7a14f148814b`, root
   `georoute_pl_gradient_decomposition_v1_33f721be_s7367_20260730_2300`,
   PL/ST/finalizer Jobs `1207484/1207485/1207486`, all `COMPLETED 0:0`.
   Finalizer uniquely identified `DDP_FP16_CAST_OVERFLOW` from all three
   PL-specific failures; its self/file hashes are
   `52d4dfd698ed0679a976e6d468fb4b0d1ede9ea630df32f808115c9f118f681e`
   /
   `816819086374f964264d3a8bb4810842f97ef554d5661d2ec4a6b85fd135bc9c`.
   Do not resume, replay,
   supplement one arm, change the 64-batch order, infer performance from live
   telemetry, or treat a later documentation commit as this runtime source.
0. The unique class authorizes exactly one communication repair. The selected
   repair is disabling DDP FP16 compression for every arm in the matched native
   family, followed by a new independent-seed no-performance stability gate.
   Do not combine it with BF16 compression, lower initial scale, clipping,
   temperature/weight/K changes, PL dropping, performance training or official
   test; those would add unregistered variables.
0. The only authorized repair study is
   `georoute_ddp_fp16_cast_repair_gate_v1`, seed `2307`, two arms and 64
   batches. Its inherited skip/scale/tail/pairwise thresholds were frozen before
   execution and cannot be relaxed after observing telemetry. Both arms must
   set `solver.fp16_compress=false`; an arm-specific hook choice is invalid.
0. The no-compression CUDA/DDP KAT is a hard same-commit parent. It must use
   real NCCL DDP without registering a communication hook, preserve a finite
   scaled FP32 gradient of `70000`, observe a nonfinite detached FP16 shadow,
   unscale finitely and complete an optimizer update. A unit-test mock or
   synthetic receipt cannot replace the Slurm KAT.
   Exact-source KAT Job `1207542` has now satisfied this contract; its self/file
   hashes are
   `257436d617b79413b4b790cda754d6dec56602d52edb07e50c03cdcd28f78b4f`
   /
   `d957514816f660a8eb43b922dfb3325baf36f1bbb706f398d0a54cc0a37df3ae`.
   Never reuse this KAT with a different runtime commit or treat it as
   real-batch stability or performance evidence.
0. The no-compression gate namespace
   `georoute_ddp_fp16_cast_repair_gate_v1_685f935e_s2307_20260730_2314`,
   exact source `685f935e759d5d78f94e5f208997644e07bf4654`, with PL/ST/finalizer
   Jobs `1207554/1207555/1207556`, is terminal and immutable. All jobs completed
   `0:0`; both arms skipped only batches `20/29`, made 62 updates, ended at
   scale `16384`, and passed the final-16 tail, so the registered pairwise repair
   gate passed. Do not resume, replace one arm, relax thresholds, or submit a
   duplicate gate. The PASS authorizes freezing a matched formal protocol only;
   it is not official AdaTAD reproduction, mAP/cost evidence, an official-test
   opening, or permission to claim Geometry Zoom/paper readiness.
0. `mmengine.Config` is not a normal dict and does not implement
   `__delitem__`. Never use `del cfg[key]` in a bound-config builder; use the
   tested `Config.pop` API and execute a real Config materialization regression.
   Jobs `1204847/1204848` proved that pure receipt tests alone do not cover this
   launcher boundary.
0. `SlidingWindowDataset.block_list` excludes videos; it is not the included
   population. The historical estimator pilot blocks Gate for train and Fit for
   val/test, so its actual populations are Fit-train and Gate-development.
   Bind included and blocked IDs separately. Jobs `1204864/1204865` proved that
   swapping these meanings prevents observer execution even when all input-file
   hashes match.
0. Do not make the real-batch diagnostic stricter than the failed pilot and
   call the result a same-config reproduction. The pilot ran through the
   ordinary train path with deterministic algorithms in warn-only mode. Jobs
   `1204908/1204909` used identical real-batch and RNG hashes and finite forward
   losses, but strict error mode rejected nondeterministic CUDA bilinear
   backward before any optimizer attempt. The diagnostic must receipt-bind
   `deterministic_warn_only=true`; this is an execution-contract correction,
   not a model repair or PL/ST result.
0. JSON object key order is not experimental arm order. Deployment validators
   must compare the exact arm-key set, normalize it back to the frozen arm
   order, require unique numeric Slurm IDs, and then bind by arm. Never reject
   a valid receipt merely because atomic `sort_keys=True` serialization changed
   insertion order.
0. Never put an `afterany` closeout behind descendants that can remain
   `DependencyNeverSatisfied`. For the estimator pilot, P0 finalization runs
   `afterany` over all P0 leaves; every training wrapper runs `afterany` over
   that finalizer but must verify the sealed PASS P0 suite before creating its
   cell or launching training; the final closeout then runs `afterany` over all
   terminal leaves. A failed P0 must end in an INCOMPLETE receipt without
   partial-performance inference. P0-finalizer and final-closeout
   prevalidation/sealing exceptions must write hashed fail-safe receipts before
   re-raising.
0. Never launch a multi-cell 60-epoch GeoRoute matrix with per-epoch full
   model/optimizer/EMA checkpoint retention and no aggregate storage
   preflight. Jobs `1196071`--`1196077` accumulated 63 GB, filled `/data`, and
   all failed during checkpoint publication before any result JSON. A
   replacement must use a new namespace, prove aggregate headroom, and retain
   final EMA only or an explicitly bounded result-blind checkpoint set. Partial
   checkpoints and epoch logs are not P1 evidence and the failed namespace must
   not be resumed.
0. Never create synthetic native support by replicating a spatial remainder
   and then route it without a validity invariant. Match the pretrained
   floor-Conv3d support, propagate a boolean validity mask, and fail closed when
   the valid count is below K.
0. Packed attention/MLP is not a packed backbone claim while the original
   Adapter still executes on a dense carrier or lets unselected positions mix
   into selected lineages. P1R requires the coordinate-lineage packed Adapter
   and a full-K numerical parity gate.
0. A `free` NativeTokenSelect control must use fixed full-frame geometry and a
   frozen geometry head. Never call a learned-geometry route ROI-free.
0. Membership comparisons must use identical uniform-selected pooling.
   Route-logit pooling is a separately named ablation, not a hidden learned-
   route advantage.
0. Deterministic context may not receive hybrid route gradients. The
   straight-through or score-function surrogate must match the hard staged
   branches and temporal route likelihood.
0. Parallel Slurm scheduling is not parallel causal interpretation. All seven
   P1R arms may run concurrently after P0R, but NativeTokenSelect must pass its
   fixed/random/geometry-side-channel controls before geometry is interpreted.
0. Never revive the failed `1196071`--`1196078` namespace, infer utility from
   its partial checkpoints, or call the current native-token router a
   source-pixel crop. Do not use “Geometry Zoom” unless the native base and the
   strict conditional geometry add-on both pass, followed by multi-seed,
   cost, diagnostic, and generalization closure.
0. Every N16R4 GitHub clone/fetch/pull/ls-remote must use the academic
   acceleration proxy frozen in `RTK.md` from the first network attempt. After
   syncing, bind full HEAD and remote-tracking SHA and require a clean tree.
   Direct GitHub attempts, uncommitted source copies, or rsync-overwritten
   snapshots are not valid experiment provenance.
0. Never launch concurrent single-node GeoRoute leaves with an implicit
   `torch.distributed.run --standalone` rendezvous endpoint. Slurm may place
   independent one-GPU jobs on the same node, where the default localhost port
   aliases their TCPStore lifetimes. Job `1199869` attached to fixed lattice's
   port `29400` and died when that store closed; hybrid `1199871` repeated the
   failure on `g0048`, terminating nine seconds after random logged
   `Training Over`. Every leaf must use a unique or kernel-assigned endpoint
   such as the already audited `127.0.0.1:0` pattern, bind a unique rendezvous
   ID, and pass an intentional same-node concurrent isolation gate. A bind
   collision invalidates the cell; never resume or infer model utility from it.
0. On N16R4, do not rely on `srun --resv-ports`: co-located diagnostic Jobs
   `1203460/1203461` both failed immediately with `Requires more ports than can
   be reserved`. The repaired GeoRoute contract instead derives a distinct
   `127/8` loopback address from the decimal Slurm job ID, retains a
   kernel-assigned port and cell/phase-bound rendezvous ID, and records the
   actual runtime port. Keep the audited 120-second readiness bound and hashed
   per-probe failure sidecar; terminate the whole torchrun process group on
   failure so no worker or TCPStore survives its parent. A bare timeout string
   is not sufficient evidence.

## Continuous-RoI S2 deployment anti-repetition

0. Never pass a Windows/CRLF-derived final CLI value into a Slurm
   `--export` string without fail-closed validation. Every export key/value
   must reject ASCII controls, leading/trailing whitespace and commas before
   `sbatch`, and the launcher must repeat the check before nested `srun`.
0. A deployment launcher is experiment code. It must resolve to the tracked
   canonical path, match the expected Git blob, and be rehashed immediately
   before every submission. A caller-selected launcher is not an auditable
   alternative.
0. A Gate-passing campaign whose training launcher fails is immutable
   deployment-failure evidence. Never edit, resume or reinterpret it as a
   model result; use a new commit, Gate authorization and campaign namespace.
0. `SUBMITTED`, `PENDING`, `RUNNING`, `Training Starts`, and epoch-0 logs are
   distinct states. None is crop-sufficiency evidence; only complete registered
   development results can advance the scientific claim.

# 禁止重走清单

## Native-Crop S2 协议反例

0. 最终方法不得是固定分辨率、固定窗口大小或从 21 个固定 `128x128`
   位置中离散选一个。目标是连续回归 source-coordinate
   `(cx,cy,w,h)`；中心、宽、高、尺度和纵横比均可变化。
0. 固定的是 local backbone 的批处理 tensor shape 时，必须明确它只是对可变
   source ROI 的重采样规格，不得把它误写为固定 source crop，也不得继续声称
   strict native-pixel-density/no-resize 是最终方法特征。
0. 21-candidate fixed library 只能作为 D0 sanity/baseline。它的通过或失败均
   不能代替连续 variable-RoI sufficiency，更不能 KILL 连续回归路线。
0. 连续宽高学习必须防止 `w,h -> 0` 的退化，并对 in-bounds、面积/尺度、纵横比和
   时间平滑给出可微参数化及测试；不得依赖推理后硬裁剪掩盖训练退化。
0. 不得把 GT 可见、逐窗口词典序选择的 reference 称为 21-candidate library
   的上界或 global-mAP oracle。它通过可作为充分证据；它失败只能否定该规则，
   不能据此 `KILL_THIS_LIBRARY`。
0. 不得把 crop sufficiency、adaptive-selection headroom 和 deployable cost
   viability 合并成一个二元 GO/KILL。固定 crop 足够但无选择 headroom 时，应记录
   `SUFFICIENT_FIXED_CROP_ONLY`，而不是宣判 crop library 失败。
0. gate raw predictions 封存前不得创建可被训练或推理命名空间访问的 gate GT
   target cache。顺序必须是 no-GT raw sweep、不可变 receipt、特权 GT join。
0. 不得用 video-cluster bootstrap 同时代表检测 mAP 与 ABBA latency/energy
   不确定性。检测和成本必须按各自采样单位校正，再做交集裁决。
0. 不得把确定性的 candidate geometry coverage 与训练模型产生的
   `CandidateUnionRecall` 混为一谈；后者应称 model-conditioned reachability。
0. 不得声称 selector-free 的 policy-shaped path 已证明 learned selector 的部署成本。
   必须预留 selector 成本预算，或只主张 representation-path headroom。
0. 不得在看过 S2 结果后调整等效、headroom 或成本 margin。冻结前只能用 synthetic
   或历史方差做 result-blind power/Monte-Carlo feasibility audit。

## Native-Crop S1 新增反例

0. 不得把 `320x180` 称为原始摄像机采集分辨率；它只是当前数据副本中
   ffprobe/Decord 可见的解码源分辨率。
0. 不得沿用会漏掉 development 身份的滑窗设置。旧 `0.25` overlap
   会遗漏 `video_validation_0000054` 的末尾短动作；Native-Crop 的
   population audit 必须覆盖冻结 manifest 的 fit 160 / gate 40，
   当前隔离配置使用 `0.5` overlap。
0. 不得把共享 VideoMAE 权重写成一次 backbone 计算。global/local
   两个视图复用同一参数实例，但仍产生两次前向计算；成本证据必须分开记录
   global backbone 与 local backbone。
0. 不得把 source-pixel equality、no-padding census、nonzero gradient
   或 `[B,384,768]` shape parity 当成 crop sufficiency。它们只授权进入
   development crop-sufficiency 协议讨论。

## Spatial Zoom 当前边界

0. 不得把整图 `Resize + CenterCrop/RandomResizedCrop` 实验称为空间选择、原生分辨率
   crop 或 Zoom。旧 `Dense-160/224/256` 仅是 R0 分辨率控制，不再是 Native-Crop S1
   的逻辑必要前置门槛。
0. Native crop 必须先在源帧坐标中选择区域，再以明确记录的局部像素密度进入重分支。
   不得把低分辨率 crop 放大回完整重模型输入后仍声称节省了像素、FLOPs 或端到端成本。
0. 在 oracle/teacher-reference crop sufficiency 通过预注册 GO 条件前，不得实现 learned
   ROI policy；也不得继续为旧 dense-resize recovery matrix 消耗 GPU 来替代 crop 验证。
0. 不得把有限的八候选或其他固定 candidate library 称为整个连续空间的 oracle。没有
   coverage certificate 时，候选库失败只能否定该库，不能直接 KILL continuous crop。
0. 不得声称 final masked mean 已消除 padded token 污染。ViT self-attention 会在 pooling
   前混合 token；优先将固定 crop 平移回图内，仅在源帧小于 crop 时 padding。
0. 不得在 source-geometry 与统计审计前冻结 `96/128` crop、48 knots、速度/尺度约束或
   `+1 pp/30%` 等 GO/KILL 门槛。
0. 不得先物化或 H2D 整段 768 帧 native-resolution float tensor 再裁剪。必须尽可能在
   decoded uint8 source frames 上完成 global/local crop 后再 format 与传输。
0. 下一垂直切片不得使用 teacher、GT、oracle 或 official-test evidence。teacher split、
   cache、候选训练分布与 formal test exception 必须在 oracle 实验前另行冻结。

1. 不得把旧 R0 称为 Zoom/crop 模型。R0 只有 matched dense spatial-resolution matrix。
2. 不得在 Native-Crop S1 GO 前实现 learned ROI policy，或用 oracle ROI 结果倒推修改
   已冻结的预注册门槛。
3. 不得把 DUCA、时序选帧、dynamic budget、max-gap 或 X3D/SlowFast prior 混入当前任务。
4. 不得恢复 `35204f5` 的 warning-bearing partial checkpoints 作为正式结果；替换矩阵必须从
   新 exact commit、新 precheck 和新 canonical experiment namespace 全量重跑。
5. 不得把 precheck、pilot、checkpoint 数量或中间 epoch 当作 S1 性能结果。
6. S1 的正式统计不得拒绝缺少稀有类的 bootstrap replicate；使用正权重 paired Bayesian
   video-cluster bootstrap，并保持 baseline/candidate 同 replicate 配对。
7. 成本只允许表述为同节点同 GPU 的 warm serial per-window latency 与 gross GPU energy，
   不得冒充 cold-start、whole-video p95、incremental energy 或完整系统能耗。
8. VideoMAE `return_feat_map=True` 会绕过分类出口 `fc_norm`；formal gradient gate 只能
   精确允许 `backbone.model.backbone.fc_norm.{weight,bias}` 两个参数无梯度。不得用前缀、
   正则或宽泛白名单掩盖新的断图。
9. S1 只持久化预注册的 gate-eligible checkpoints；不得保存不会参与选择的 pre-gate
   周期权重耗尽共享存储。任何存储故障后的矩阵不得 resume，必须新 commit、门禁和 namespace。
10. S1 selector must follow the official evaluator's prediction-domain policy:
    finite zero-length proposals remain zero-IoU false positives. Do not reject
    or delete them, because either action diverges from or inflates official AP.
    The in-training evaluator log is not a gate score when its GT population is
    broader than the frozen gate prediction population.
11. Post-processing repair code must not reconstruct a historical bound config
    against its own current `ROOT` or commit. It must derive the original clean
    repository from the recorded audited config path, verify its exact Git HEAD
    and config matrix, and validate the original precheck there. Never copy or
    rewrite bound configs merely to make a repair snapshot accept them.
12. A clean repair clone does not own the training snapshot's ignored `data/`
    mount. Repair entrypoints that instantiate the official dataset must run
    with the historical clean training snapshot as the working directory while
    importing the certificate-bound repair code explicitly. Do not add hidden
    symlinks or Git excludes to make relative dataset paths appear available.
13. Do not certify a long formal power profile from a short synthetic cadence
    test while the sampler remains a Python thread inside the detector process.
    Job `1167538` proved that native NVML can still suffer a `2413.519` ms
    observed gap under the full memory-heavy inference/NMS path despite passing
    a ten-second Gate. Keep the 20 ms target and 100 ms limit unchanged; require
    an independently scheduled UUID-bound sampler process, preserve the raw
    failure trace, and pass a representative long-duration no-open stress Gate
    before any replacement matrix.
14. A locally passing sidecar implementation is not a passed Gate. Do not
    submit a replacement matrix until a clean remote snapshot completes the
    full 792-exposure dense256/seed3408 path with the frozen 20/100 ms cadence,
    4+1 CPU isolation, UUID parity, unchanged test-evidence hash, and no formal
    profile publication. Submit exactly one serial matrix only after that Gate.
15. Do not require a separately scheduled Gate and matrix to receive the same
    physical GPU UUID. The Gate must bind its own actual UUID; the matrix must
    match the Gate's stable hardware/software class, bind its own actual UUID,
    and keep all nine cells in one allocation on one physical GPU.
16. Do not pair a sidecar report with an independently selected trace. Every
    consumer must use the shared attempt validator and recompute trace hash and
    cadence. Partial salvage may publish a missing hash-matching counterpart
    but must never overwrite an existing report or trace.
17. A matrix namespace is single-use. The persistent atomic matrix lock and
    start/completion receipts are evidence, not temporary scheduling files.

## Native-Crop provenance anti-repetition

1. Do not accept `expected_commit` while ignoring untracked files or merely
   recording working-tree hashes. Every audited executable/configuration file
   must be tracked and byte-equal to its `HEAD` blob before the gate runs.
2. Do not treat a self-hashed geometry census as source-bound evidence. The
   gate must re-probe every frozen development video and match containment,
   path, file size, dimensions, rotation, frame count, and frame rate.
    Never remove a failed lock to resume or duplicate the same campaign.
18. Do not lower the formal 90,000 MiB memory floor to fit N16R4's 55 GB
    one-GPU outer-job default, and do not override `CUDA_VISIBLE_DEVICES`.
    Reserve the site's two-GPU outer resource only when required to obtain
    sufficient memory, then run the entire Gate or frozen matrix in one exact
    Slurm step with one GPU, five CPUs, and 96,000 MiB. Record the step-scoped
    GPU and finite cgroup limit. The idle outer GPU is scheduling overhead,
    not model compute or measured cost, and must be disclosed.
19. Do not acquire or consume a formal matrix namespace before all no-write
    preflights pass. This includes source/artifact checks, the representative
    cell, finite cgroup v2 memory, step-GPU membership, logical-CUDA/NVML UUID,
    Gate hardware/software class, and in-memory matrix-start receipt
    validation. Never hand-write an alternative receipt in a launcher.
20. Do not combine profiles from different Slurm jobs, steps, or GPUs. Every
    profile, marker, and descriptor must bind the same canonical start receipt;
    the analyzer requires one completion receipt that seals exactly nine
    frozen-order descriptors. A directory count is not equivalent evidence.
21. Do not open any new sealed-test cell before all nine frozen cells pass the
    matrix no-write dry-run and the current start receipt passes runtime
    hardware/software validation. A per-cell preflight after another cell has
    opened the test is not a substitute for the all-cell dry-run.
22. Do not accept unbound official-test evidence from a prior matrix. The only
    exception is the historical dense256/seed3408 evidence whose canonical
    path, file hash, internal hash, and cell identity are frozen by the active
    recovery certificate. Every newly opened cell must publish and validate a
    canonical test-to-matrix binding before profiling.
23. Do not treat marker, stdout, or stderr files as proof that the NVML sidecar
    started. Salvage is authorized only by sidecar PID/ready/raw-power/result
    evidence, and a failed salvage must remain a hard failure.
24. Do not run the historical training snapshot's `tools/test.py` unchanged
    inside the high-memory two-level Slurm allocation. Job `1170468` proved
    that its old guard rejects the valid exact one-GPU step because it inspects
    the two-GPU outer `SLURM_JOB_GPUS`. Never falsify Slurm variables or edit
    that snapshot in place. A replacement must use a recovery-certificate-bound
    runtime entrypoint while proving the model/config/evaluator tree is unchanged.
25. One completed descriptor in a failed matrix is diagnostic cell evidence,
    not a partial 3x3 result. Without the exact-nine completion receipt it
    cannot select a resolution, drive GO/KILL, or be pooled with another
    campaign.
26. Do not recompute S1 evidence with ambient Python user-site packages.
    Runtime commit `6524e1b` proved NumPy `2.2.6` changes tied-score AP ordering
    relative to the formal Conda NumPy `1.23.5`. This is not grounds for metric
    tolerance: formal Gate/matrix/cell launchers must set
    `PYTHONNOUSERSITE=1`, and exact stored metrics must reproduce.
27. A failed post-`sbatch` receipt writer does not authorize another
    submission. Reconcile the existing Job by name/accounting first, then
    atomically bind that Job ID with the certificate-bound Conda Python.
    Job `1170765` was recovered this way after the login-node bare `python`
    rejected an f-string; no duplicate Gate was submitted.
28. Do not use a recovery schema name as a proxy for inherited runtime
    capability. Job `1170765` failed before sidecar startup because v5 carried
    the exact buffered-sidecar contract while the profiler accepted only the
    literal v4 reason. Validate backend, atomic publication, no loop I/O,
    20/100 ms cadence, 4+1 CPUs, and long-Gate fields together. A descendant
    recovery must also bind the failed parent inventory and prove no sidecar or
    new test evidence appeared; the failed campaign remains immutable.
29. Do not let recovery evidence roles alias or mix. Parent certificates and
    Gate stdout/stderr must use their canonical campaign paths; stdout/stderr
    cannot share a path or inode. Legacy, matrix, Gate, and power-diagnostic
    evidence are mutually exclusive for a schema transition, and incomplete
    role sets must fail closed rather than be ignored.
30. Do not merge the pre-policy S2 representation gate with S3 learned-policy
    training while retaining the old stage names. A learned ROI head inside S2
    changes the scientific question and cost semantics.
31. Do not derive fixed-center, random, discrete, or fixed-size controls by
    overriding a variable-box-trained checkpoint only at inference. A
    decision-critical comparison requires matched training distributions.
32. Do not compare a per-window GT-privileged continuous reference with
    unprivileged fixed or location-only controls. Privilege and search budgets
    must match before attributing gains to variable width/height.
33. Do not certify continuous spatial-reference adequacy merely because a
    detector-confidence objective converges. Confidence optimization can select
    false positives or action interiors and is only a no-GT policy diagnostic.
34. Do not double-count both a measured ROI policy head and a future-selector
    reserve in the same cost path.
35. Do not delete a failed or contaminated formal namespace. Preserve it
    immutably and create a new recursively bound campaign.
36. Do not freeze one-GPU high-memory Slurm requests, storage floors, or NVML
    gap thresholds that contradict the audited N16R4 allocation and validated
    20/100 ms sampler contract.
37. Do not call equal `sx,sy` an equal physical center trajectory when the
    decoder also conditions `cx,cy` on `w,h`. A fixed-size versus variable-size
    contrast must pair physical centers explicitly or state the center change
    as part of the intervention.
38. Do not launch the Continuous-RoI S2 reference sweep until the Sobol engine,
    dtype, transform serialization, stable-hash bytes and known-answer hash are
    frozen. A seed and draw shape alone are not an auditable generator identity.
39. Do not interpret the raw no-leak ban on a preferred/GT-selected reference
    ID as a ban on a result-blind enumerated candidate ID. Freeze the typed raw
    schema and object-graph audit before inference; never let the privileged
    join run in the raw GPU process.
40. Exact-nine training completion proves only training/exposure integrity.
    It is not development mAP, reference adequacy, crop sufficiency, cost
    viability, official-test evidence, or authorization for S3.
41. Do not infer equal runtime or energy from equal exact window budget. SCNR
    centering preserves `B=24576` but changes the induced `K_t` distribution and
    therefore can change ragged attention pairs and full-stack cost.
42. Do not reuse an ABBA-only cost profile as the residual-centering decision.
    The prior M2 cost run exposed a cold first-arm host/input outlier. The frozen
    successor requires both ABBA and BAAB in one Slurm Job/GPU, retains all eight
    passes, and uses one continuous power sidecar.
43. Do not open seeds 3408/3409 from the seed-3407 accuracy pass alone. The
    sealed finalizer authorizes only paired cost; both primary cost-ratio 95%
    upper bounds must be at most 1.05 under the preregistered analysis.
44. Do not profile control and centered checkpoints in separate Jobs or GPUs
    and call the result paired. Hardware UUID, software identity, population,
    pass order, source checkpoints, and power trace must be recursively bound to
    one allocation.
45. Do not reinterpret the centered Gate observation `K_t min=5` as a new
    minimum-K rule. The method still allows `K_t=0`; the observed centered
    distribution is an empirical consequence, and the approved `K_t>=1`
    alternative remains a separate ablation.

## 任务与叙事

1. **不要再称 Online TAD。** 当前方法观察完整离线窗口；`online` 仅表示 forward
   内生成且不查 ledger/cache。
2. **不要把 THUMOS14 解释成 key-event timestamp spotting。** 它监督动作区间；项目
   应表述为边界敏感的稀疏 interval detection。
3. **不要把插件泛化当作已证明。** 当前只有 AdaTAD-derived 主路径，第二 detector
   仍缺正式结果。

## 模型

4. 不再回到“粗分类器独立训练 → selector 独立训练 → detector 独立训练”作为最终
   方法；它只能是归因 baseline。
5. 不允许 `asformer_lite` 冒充官方 ASFormer。
6. actionness 必须由二分类 GT 校准，但 selector 必须以 transition/boundary/utility
   为首要目标；不能再次退化为 actionness top-k。
7. 不允许用硬膨胀、uniform scaffold、max-gap repair 把坏分数修成看似合理的网格而
   不披露 repair 数量和影响。
8. `detector_utility_target` 若来自 GT 边界，只能叫 boundary-utility proxy。
9. 不得声称“完全未修改官方 AdaTAD”；源码 wrapper、selected-axis 和 GT remap 已变。

## 训练与梯度

10. nonzero grad 只证明连通，不证明梯度方向等价于 hard frame utility。
11. loss schedule 必须按 optimizer step 推进，不能按 raw forward 次数。
12. detector backend loss 与 selector gradient bridge 必须分开：关闭 bridge 不得关闭
    detector 学习。
13. dynamic budget 不得只优化 expected K；必须记录真实执行 K 与实测成本。

## 实验

14. 不再重复排同一 X3D dense export/grid；它计算过慢且可能吞掉节省。
15. 不再用旧 commit、失败 suite、重复 job 或缺失 checkpoint 的运行填论文表格。
16. 不再把 smoke、precheck、toy wrapper、geometry-only 指标称为主实验。
17. 不再跳过 exact-uniform/random/dense 等同提交基线后继续扩新方法。
18. 不再只看 Avg-mAP；必须看 mAP@0.6/0.7、短动作和边界误差。
19. 不再只报模型 FLOPs；必须报告完整数据和系统通路的 p50/p95、显存、energy。
20. 不允许 validation/test GT、teacher、oracle、raw prediction cache 或外部隐式 JSONL
    参与主方法选择。

## 决策纪律

21. 讨论提出的 CVCR/BCFT/CoDeTAD/physical-grid/CFPA 不等于已经实现或更优。
22. 决定性实验未完成前，不宣布 DUCA 成功；同样也不宣布其必然失败。
23. 每次部署前必须记录 commit、配置、checkpoint、数据、Job ID 和 run root。
24. 新结果必须先更新 experiment/claim 节点，再改论文叙事。
