---
type: experiment
node_id: exp:georoute-adatad
title: "GeoRoute-AdaTAD native spatial routing"
stage: tested
status: tested_p0_mechanical_only
updated: 2026-07-23
---

# GeoRoute-AdaTAD native spatial routing

## Question

At a fixed native VideoMAE token budget, can a continuous geometry prior plus
free-token residual evidence protect high-tIoU offline TAD better than
unstructured free TokenSelect at lower measured end-to-end cost?

## Current evidence

- Local implementation exists for native `2 x 16 x 16` tubelet routing,
  ROI-only, free-token, hybrid, the fixed-lattice geometry-side-channel
  control, P0 CUDA checks, a result-blind P0/P1/P2/P3 dispatcher, theory, and
  external paper plotting/table tools.
- Pure Python contract, DAG, result-schema and paper-tool checks passed
  `20` tests on 2026-07-23.
- The local Windows Torch runtime fails while loading `c10.dll`; this is an
  environment failure, not CUDA evidence. The Torch-dependent routing tests
  and the only meaningful P0 verdict remain pending on N16R4 CUDA.
- The first N16R4 P0 submission created no Slurm jobs because the deployment
  wrapper requested one outer GPU with `96G`, above the site's `55G/GPU`
  outer-allocation rule. The model code did not execute and emitted no P0
  report. The deployment code now requests a site-compliant two-GPU outer
  allocation and the existing launcher still executes exactly one GPU, five
  CPUs, and `96G` in an inner Slurm step.
- The second N16R4 P0 attempt created jobs `1180859`--`1180861`, but all
  three failed before a P0 JSON after reaching real AdaTAD detector forward:
  `_gather_selected_native_tubelets` expanded `[B,T,K]` to eight rather than
  seven native-video dimensions. The failed namespace and logs are preserved
  as diagnostics. A separate scheduler defect also rejected the CPU-only P0
  finalizer because this site requires every batch job to declare a GPU. Both
  issues are now under a minimal code fix. The accompanying remote routing
  suite also exposed missing uniform/random role accounting and a roundoff
  residue in the ST hard-forward gate; both are patched and still await the
  corrected remote suite and CUDA P0. No metric, cost, official test, or P0
  pass claim resulted.
- The next corrected snapshot passed its full remote focused suite, but the
  scheduler admitted only two of the three P0 leaves. `1180874` (hybrid,
  straight-through) independently passed its one-step CUDA gate: exact
  `K=32`, one packed VideoMAE forward, real AdaTAD classification/regression
  backward, finite nonzero scout/router gradients, and 5.15 GB peak allocated
  memory. This is a leaf-level implementation fact, not a suite verdict. The
  all-token dense leaf `1180873` fail-closed before a JSON because its P0
  reference used `no_grad`, which may dispatch CUDA SDPA differently from the
  real training forward (`max_abs_error=6.5612793e-4`). The repair keeps the
  `1e-4` criterion unchanged, executes the reference with matching autograd
  dispatch, immediately detaches it, and records that mode in the P0
  contract. It still requires a fresh CUDA verification.
- No development metric, cost result, official-test record, paper claim, or
  A-MoD experiment exists.
- **Authoritative current P0 evidence:** the model-path commit
  [`4a9358d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4a9358d1fba4bde9aa7693a94f7e4dfc95d31ecc)
  produced three CUDA one-step reports in
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/georoute_adatad_4a9358d_p0_20260723_102943`:
  dense all-token parity Job `1180906`, score-function ROI Job `1180907`, and
  straight-through hybrid Job `1180927`, all `COMPLETED 0:0`.  The dense report
  uses matching autograd dispatch and passed its frozen combined numerical
  comparison; hybrid has exact unique `K=32`, one packed heavy VideoMAE forward,
  real AdaTAD classification/regression loss, and finite nonzero geometry and
  residual scout gradients.  The score-function report binds real detector
  losses to the geometry scout.
- The P0-only finalizer first failed closed as Job `1180963` before writing a
  report because its file-path Python entrypoint could not import the repository
  `tools` package.  It is preserved as diagnostic evidence.  The minimal module
  entrypoint repair
  [`c2a3c69`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/c2a3c69d9006b32b1ca18f4ce66222b59550c45f)
  passed remote shell/focused checks; replacement Job `1180966` completed
  `0:0` and sealed `p0_finalization.json` as `PASS_MECHANICAL_ONLY`, suite SHA
  `a6f8ea041345cdc400c7f8a4f478c037cb66c8cfd3c19edb09d454ff363ce0b1`.
  It did not dispatch P1.  This is not accuracy, total-cost, or paper evidence.
- The P1/P2/P3 activation repair is implemented locally: a new P1 bootstrap
  recomputes the sealed P0 suite before submitting the seven matched P1 cells,
  freezes P2/P3 successor policy in its receipt, and never replays P0.  The
  dispatcher now uses the valid hyphenated action names and an N16R4-compatible
  control allocation without an explicit `--mem` override.  Focused contract
  checks pass `18/18`.  This remains `implemented` until the fresh committed
  remote snapshot has passed its Linux checks and submitted P1.

## Frozen decision logic

P0 proves only implementation facts. P1 compares all matched primary
controls. If free TokenSelect beats ROI-plus-residual on the high-IoU/cost
Pareto, the ROI primary claim is killed rather than tuned after the fact. P2
and P3 are result-blind descendants; they cannot start direct full training
until their parent decision receipt authorizes them.

## Evidence outputs

- Theory: `docs/methods/georoute_adatad_theory.md`
- Figure/claim plan: `docs/methods/georoute_adatad_paper_evidence_and_figures.md`
- Runtime P0: `tools/bata/run_georoute_p0_gate.py`
- Result schema/analysis: `tools/bata/georoute_result_schema.py` and
  `tools/bata/analyze_georoute_results.py`
- External figures/tables: `tools/bata/plot_georoute_paper.py` and
  `tools/bata/render_georoute_paper_tables.py`
