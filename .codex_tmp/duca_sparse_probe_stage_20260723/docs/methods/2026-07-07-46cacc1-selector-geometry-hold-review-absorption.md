# 2026-07-07 Pro Review Absorption: Selector Geometry HOLD

Source raw record:

- `docs/methods/reviews/2026-07-07-46cacc1-selector-geometry-hold-review-raw.txt`
- Reviewed visible commit: `46cacc113042fcf0931c70774491d44665246e32`
- Overall verdict from Pro review: `HOLD`

## Absorbed Core Judgment

The current repository has useful ledger validation, provenance, summaries, and preliminary diagnostics, but it does not yet support strong paper claims such as:

- selected frames concentrate near action boundaries;
- PAction learned is better than GAS-VT because it is more boundary-aware;
- Stage2/Stage3 are already proven detector-aware acquisition methods.

Existing `boundary_support@r` is a boundary recall-style metric. It does not prove selected-frame concentration around boundaries. Existing `action_positive_coverage` is frame/action-positive coverage, not per-action instance coverage. These metrics must not be over-claimed.

## Missing Evidence

The review identifies missing analysis needed before strong claims:

- selected-frame distance distribution to nearest GT boundary;
- selected-frame region share: boundary band / action interior / background / invalid;
- per-action endpoint coverage: start hit, end hit, both endpoints hit, neither hit;
- holes by region: whole video, action interior, boundary band, background, long/short actions;
- normalized action-time density over aligned actions;
- p_action / uncertainty / detector utility calibration against boundary/interior/background;
- matched-budget method comparison dashboards and failure galleries.

## Code Correctness Risks To Address

- Coordinate-system contract is not explicit enough. All analysis must specify dense frame axis, valid length, frame stride, segment convention, selected position convention, and whether segment ends are inclusive or half-open.
- Boundary support must not be interpreted as selected concentration.
- Bracket metrics should be split into boundary hit, boundary bracket, and exact boundary hit.
- Action coverage should be separated from per-action instance coverage.
- Short-video selected-count behavior must not be mixed with fixed-384 paper claims.

## Required New Analysis Suite

The review asks for a reproducible selector geometry suite:

- `tools/bata/analyze_selector_geometry.py`
  - outputs frame/video/action/method-level tables;
  - computes boundary distance, region share, endpoint coverage, holes by region, normalized action-time density, and score calibration.
- `tools/bata/export_selector_paper_tables.py`
  - converts geometry and mAP outputs into paper-ready CSV tables.
- `tools/bata/validate_selector_geometry_metrics.py`
  - validates coordinate contract, off-by-one behavior, padding, selected count fields, and toy edge cases.
- `tools/bata/plot_selector_geometry.py`
  - global figures: boundary-distance CDF/histogram, region share bars, endpoint coverage, holes by region, calibration, normalized density.
- `tools/bata/plot_selector_timeline.py`
  - per-video raster with GT segments, boundary bands, selected ticks, p_action/utility curves.
- `tools/bata/plot_selector_dashboard.py`
  - side-by-side per-video method dashboard.
- `tools/bata/generate_selector_failure_gallery.py`
  - representative failure/success cases for GAS-VT, PAction, Stage2, Stage3.

## Experimental Implication

Do not keep adding training variants before the analysis contract is frozen. The immediate correction is:

1. Build the selector geometry analysis suite.
2. Run matched-budget geometry analysis for `uniform_384`, `paction_topk_384`, `PAction learned_fixed_384`, `GAS-VT fixed_384`, Stage2, and Stage3 when available.
3. Only claim boundary-aware improvement if both mAP and geometry support it.
4. If geometry does not support boundary concentration, weaken the claim to: PAction learned is empirically stronger than GAS-VT under this implementation, but the advantage is not yet proven to arise from boundary concentration.

## Updated Local Priority

This review shifts the next implementation priority from more full-train variants to a reproducible geometry analysis layer. Model optimization remains important, but without this suite we cannot know whether gains come from boundary-aware acquisition, action-interior oversampling, short-video behavior, decoder differences, or detector/config differences.
