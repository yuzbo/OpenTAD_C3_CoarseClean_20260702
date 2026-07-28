# NativeTokenSelect-first correctness replacement and parallel experiment design

Date: 2026-07-28
Status: `implemented_local_pending_remote_p0r`
Source verdict: `HOLD_FOR_CORRECTNESS_FIX`

## Research objective

The first publishable question is not whether a learned crop looks plausible.
It is whether detector-supervised, exact-budget selection of source-native
VideoMAE tubelets can protect high-tIoU offline TAD while reducing measured
total cost. Continuous geometry is a conditional second hypothesis: it is
retained only if it adds to a valid NativeTokenSelect base under the same
training, budget, aggregation, adapter, and cost protocol.

The implemented model is therefore a native-token evidence router. It does not
perform a second source-pixel crop, a second resized view, or a sequential
TokenSelect-then-Geometry forward. “NativeTokenSelect first” denotes the causal
evidence hierarchy and selector rule. “Geometry Zoom” is not an authorized
paper name unless the conditional geometry gate passes.

## Absorbed correctness verdict

The external Pro audit of exact commit
`df3e54e0c6776544dba20807b2ec100e1a399654` identified six interpretation
blockers. The replacement implements all six in one code batch:

1. Source support follows the pretrained Conv3d floor rule. A 180x320 frame is
   cropped to 176x320, yielding an 11x20 native patch grid and explicitly
   ignoring the bottom four rows. A boolean validity mask is carried through
   exact-K selection; `valid_count < K` fails closed before heavy execution.
2. Attention, MLP, and the original VideoMAE temporal Adapter all execute only
   on selected spatial lineages. The Adapter uses the original learned
   projection, depthwise 3x1 temporal convolution, up-projection, and gamma,
   but looks up neighbors only at the same absolute spatial index. Missing
   lineage neighbors contribute zero; unselected carrier positions are exact
   identity. Full-K must equal the dense Adapter numerically.
3. The `free` control is genuinely ROI-free: geometry is fixed to the full
   frame `(0.5, 0.5, 1, 1)`, its geometry head is frozen, and its membership is
   decided only by residual NativeTokenSelect scores.
4. Every primary P1 arm uses identical uniform-selected `1/K` aggregation.
   Route-score pooling remains available only as an explicitly named ablation,
   never as a hidden advantage for learned membership.
5. Hybrid straight-through gradients are aligned with the hard staged route:
   deterministic context has zero route gradient, ROI fills its assigned valid
   support, and residual selection acts on the remaining support. The
   score-function likelihood sums temporal decisions before batch averaging.
6. GeoRoute training uses atomic final-only checkpoint publication. Deployment
   performs aggregate, same-commit storage preflight before namespace
   acquisition and again before every stage. A completed cell must contain
   exactly one final checkpoint and no temporary checkpoint.

The correctness batch also repairs the decision contract: random is a required
control, Route B can advance without geometry, and geometry cannot advance
merely because hybrid exceeds an already weak free route.

## Fixed P0R/P1R experiment

P0R has three one-step CUDA leaves:

- dense full-token numerical parity;
- score-function learned routing through real AdaTAD losses;
- branch-aligned straight-through hybrid routing.

P0R checks implementation facts only: floor-native support, validity
propagation, exact unique K, full-K parity, packed attention/MLP/Adapter
accounting, real detector backward, finite route gradients, component trace,
memory, zero training checkpoints, and a same-commit storage profile. Its only
positive status is `PASS_MECHANICAL_ONLY`.

P1R contains seven matched seed-3407 arms:

| Arm | Learned selection | Learned geometry | Purpose |
|---|---:|---:|---|
| `dense_native` | no | no | accuracy and total-cost upper-compute anchor |
| `fixed_lattice` | no | no | deterministic exact-K control |
| `fixed_lattice_geometry` | no | yes | geometry side-channel without learned membership |
| `random` | no | no | stochastic exact-K control |
| `free` | yes | no | ROI-free NativeTokenSelect base |
| `roi` | geometry-only | yes | structured-support attribution |
| `hybrid` | geometry plus residual | yes | conditional geometry candidate |

All arms share source windows, initialization, detector/head/loss, optimizer
updates, seed, K, native grid, uniform pooling, coordinate-lineage packed
Adapter, evaluator, and cost instrumentation. A-MoD, FlashVID-inspired
residuals, dynamic K/depth, curricula, and auxiliary selection losses are
excluded from P1R.

## One-shot parallel deployment

The complete dependency graph is submitted once:

```text
P0R dense ─────┐
P0R score-fn ──┼─ afterok → P0R finalizer
P0R hybrid ────┘                    │
                                    ├─ PASS_MECHANICAL_ONLY
                                    │
                                    ├─→ seven P1R arms, concurrently
                                    │          │
                                    │          └─ afterok(all) → P1 selector
                                    │                               │
                                    │                               └─ result-gated P2/P3
                                    └─ failure → no P1R namespace
```

This is parallel scheduling without weakening the evidence order. P0R is a
mechanical parent, all seven P1R scientific arms are independent siblings, and
the result-blind selector is the only process allowed to instantiate later
stages.

## Hierarchical decision rule

NativeTokenSelect is established only if both conditions hold:

1. `free` exceeds `fixed_lattice`, `random`, and
   `fixed_lattice_geometry` on the frozen high-tIoU accuracy criterion; and
2. `free` has lower measured total cost than `dense_native`.

If this native gate fails, learned routing stops. A superficially strong hybrid
cannot rescue a failed NativeTokenSelect base; that outcome is ambiguous
geometry/selection interaction, not proof of Geometry Zoom.

If the native gate passes, geometry is tested conditionally. Hybrid must exceed
`free`, `random`, and `fixed_lattice_geometry` on high-tIoU accuracy while its
measured total cost is no greater than all three. A pass promotes the corrected
hybrid as Route A. Otherwise the simpler ROI-free `free` route advances as
Route B.

P2 repeats the selected route and its complete controls over three seeds and
pre-registered budgets. P3 opens only after P2 and freezes second-dataset or
second-detector generalization before one sealed official test.

## Required evidence and claim boundary

P1R is a selector screen, not a paper-ready result. The publication package
requires raw-seed Avg-mAP and mAP@0.3--0.7, especially 0.6/0.7; paired
short-action and boundary diagnostics; exact requested/effective/unique token
counts; route coverage and stability; full decode-to-NMS p50/p95 latency;
peak memory; energy; and component-level cost accounting. P2 supplies
multi-seed and budget evidence; P3 supplies generalization and sealed-test
closure.

Until those stages complete, allowed statements are limited to
`implemented`, remotely `tested` mechanical facts, and
`experiment_running`. Neither P0R nor a single P1R seed authorizes
`empirically_supported`, `paper_ready`, a speedup claim, or the method name
“Geometry Zoom.”
