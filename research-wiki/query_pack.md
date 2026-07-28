---
type: query_pack
updated: 2026-07-28
max_chars: 8000
---

# Research Query Pack

## Current Active Route: NativeTokenSelect-first GeoRoute-AdaTAD (2026-07-28)

- Objective: first test whether detector-supervised, ROI-free exact-K selection
  of source-native VideoMAE tubelets protects high-tIoU offline TAD at lower
  measured total cost. Only after that base passes may continuous geometry be
  tested as a strict add-on.
- Active replacement status: `p1r_submitted_priority_pending`
  (`experiment_running`). The external Pro audit of exact
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
  `1199865`--`1199871` and selector `1199872`; leaves are priority-pending and
  selector dependency-pending. These remain implementation/test/deployment
  facts, not accuracy or cost evidence.
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

1. Match commit, data, pretrained initialization, updates, seeds, token budget
   and detector/head before comparing selection methods.
2. Report high-tIoU, short-action/boundary diagnostics and measured full-stack
   p50/p95 latency, memory and energy. FLOPs or random-init profiling alone
   cannot establish efficiency.
3. Training-only, smoke, precheck, pending or failed jobs never become
   empirical support. Test/validation GT, teacher signals and raw prediction
   caches must never participate in inference decisions.
4. Any route that fails its matched control narrows or dies instead of gaining
   extra selector heads or loss weights. Preserve failed evidence in the
   experiment record and `anti_repetition.md`.

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
