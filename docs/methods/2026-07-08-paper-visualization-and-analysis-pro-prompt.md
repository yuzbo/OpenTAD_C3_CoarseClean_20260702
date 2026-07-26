# Pro Prompt: Paper-Level Visualization and Analysis Plan for Sparse Pre-Backbone TAD

You are reviewing a public GitHub repository for a C3/OpenTAD/AdaTAD sparse temporal acquisition project. Please inspect the latest HEAD of:

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Branch: `codex/gas-vt-stage23-detector-aware-20260706`

Before reviewing, resolve and report the exact commit SHA you inspected. If the branch is not visible or the commit cannot be loaded, stop and report a visibility failure.

## Research Context

The project targets Temporal Action Detection (TAD), using AdaTAD/OpenTAD as detector. The current research question is:

Can a low-cost, deployable pre-backbone temporal acquisition/selection module choose at most 384 observations from an AdaTAD 768-position temporal grid while preserving or improving detector mAP, especially high-IoU localization?

Important caveat: the current selector operates on the 768 AdaTAD temporal grid after the video/features have already been downsampled, not directly on raw original video frames. Please discuss whether this is acceptable for a paper story, whether it looks like secondary processing, and what additional raw-frame or pre-feature evidence would be needed if the method is claimed as "pre-backbone."

## Current Empirical Anchors

Known detector mAP anchors:

- Dense AdaTAD teacher/checkpoint anchor: about `68.29` Average-mAP.
- PAction learned fixed384: `59.10` Average-mAP.
- GAS-VT fixed384: `44.90` Average-mAP.
- Lattice move50/move75: geometry analysis exists, detector mAP still pending.
- Stage2 detector-aware responsibility route: running/being debugged; not yet paper evidence.
- Stage3 joint selector + detector: precheck only; not a completed end-to-end mAP result.

Known selector geometry on val split:

| method | Avg mAP | both endpoint coverage @ r16 | boundary recall @ r1 | p95 unselected hole | comment |
|---|---:|---:|---:|---:|---|
| GAS-VT fixed384 | 44.90 | 0.17164 | 0.22794 | 92.54 | large holes, poor endpoint coverage |
| PAction learned fixed384 | 59.10 | 0.26471 | 0.42148 | 2.18 | much smaller holes, better boundary coverage |
| lattice move50 | pending | 0.26520 | 0.42895 | 2.00 | geometry diagnostic, not final method |
| lattice move75 | pending | 0.26520 | 0.42727 | 2.00 | geometry diagnostic, not final method |

The core observed issue is that the first visualization set was not visually direct enough: stacked bars and generic coverage curves made differences hard to see. We added paper-facing direct summary plots, but we want a senior reviewer to tell us what is still missing.

## Code Paths to Inspect

Please inspect these files line by line for correctness, missing assumptions, and whether the outputs support the paper story:

- `tools/bata/analyze_selector_geometry.py`
- `tools/bata/export_selector_paper_tables.py`
- `tools/bata/plot_selector_geometry.py`
- `tools/bata/plot_selector_paper_summary.py`
- `tools/bata/plot_selector_timeline.py`
- `tools/bata/plot_selector_dashboard.py`
- `tools/bata/generate_selector_failure_gallery.py`
- `tools/bata/validate_selector_geometry_metrics.py`
- `tests/test_analyze_selector_geometry.py`
- `tests/test_selector_geometry_outputs.py`
- `tests/test_plot_selector_geometry.py`
- `tests/test_plot_selector_paper_summary.py`

The new direct paper-facing script is:

```bash
python tools/bata/plot_selector_paper_summary.py \
  --analysis-root <analysis_root_with_geometry_and_tables> \
  --out-dir <analysis_root>/paper_direct_figures \
  --formats pdf png
```

It generates:

- `paper_gap_boundary_quadrant.{pdf,png}`: p95 unselected hole vs both-endpoint coverage, with mAP labels when available.
- `paper_delta_vs_gasvt.{pdf,png}`: delta mAP, delta endpoint coverage, and p95-hole reduction relative to GAS-VT.
- `paper_selector_scorecard.{pdf,png}`: method-by-metric heatmap with raw values.

## Review Questions

Please answer rigorously:

1. Are the current analysis metrics sufficient to explain why PAction learned fixed384 strongly beats GAS-VT fixed384, or are we still missing a decisive diagnostic?
2. Are `p95_unselected_hole`, `both endpoint coverage`, `boundary recall @ r1/r16`, and region share the right geometry metrics for TAD localization? What metrics should be added?
3. What figure set would make this a mature, visually convincing CVPR-style paper? Please propose the exact figure list, each figure's message, required data, and preferred visualization type.
4. Which figures should be main-paper figures versus appendix figures?
5. How should we visualize per-video selection behavior so that it is not anecdotal? Should we show representative timelines, failure galleries, density maps aligned to normalized action time, or counterfactual utility maps?
6. How should we visualize high-IoU localization preservation? What mAP@0.5/0.6/0.7 decomposition, boundary error, or proposal-quality plot is required?
7. How should we compare against uniform_384, random_384, p_action top-k, GAS-VT, lattice diagnostic, Stage2 detector-aware, and dense AdaTAD without confusing diagnostic methods with paper-main methods?
8. Does the current code risk misleading plots, for example by mixing rows from short videos, mixing 768/dynamic with fixed384, using missing mAP as zero/NA in a confusing way, or normalizing metrics in the heatmap in a way that hides absolute magnitudes?
9. What exact additional CSV/JSON schema should be exported to support stronger plots, such as action-length-stratified performance, boundary-length normalized distance, per-class breakdown, and compute/speed tradeoff?
10. Given the current evidence, what would be the minimum set of additional experiments and figures required before claiming a paper-level contribution?

## Desired Output

Please provide:

1. A verdict: PASS/WARN/HOLD for the current visualization and analysis suite.
2. A line-by-line code review of the relevant scripts, focusing on correctness and possible misleading visualization logic.
3. A concrete paper figure plan with:
   - Figure ID
   - Claim supported
   - Data source
   - Plot type
   - Required additional code
   - Whether it belongs in main paper or appendix
4. Key code snippets or pseudocode for the missing analyses/plots.
5. A revised experimental evidence plan from the current state to a logically complete paper.
6. A frank assessment of whether the current 768-grid selection setting is enough for a pre-backbone paper claim, or whether raw-frame acquisition evidence is required.
