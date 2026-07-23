---
type: experiment
node_id: exp:georoute-adatad
title: "GeoRoute-AdaTAD native spatial routing"
stage: implemented
status: implementation_fix_pending_cuda_p0
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
  issues are now under a minimal code fix; no metric, cost, official test, or
  P0 pass claim resulted.
- No development metric, cost result, official-test record, paper claim, or
  A-MoD experiment exists.

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
