---
updated: 2026-07-08
status: active
scope: 记录并吸收 Pro/GPT 对 selector geometry visualization suite 的 HOLD 审查
out-of-scope: 不把该审查误当成最新 44ee8e7 代码的最终判定；不在本文声明 paper claim 已成立
---

# 46cacc1 Visualization HOLD Review Absorption

Raw review archived at:

- `docs/methods/reviews/2026-07-08-46cacc1-visualization-hold-review-raw.txt`

## Review Target

The review inspected public branch `codex/gas-vt-stage23-detector-aware-20260706` at commit:

```text
46cacc113042fcf0931c70774491d44665246e32
```

Its verdict was:

```text
HOLD_CURRENT_VISUALIZATION_SUITE_NOT_PUBLIC_AT_INSPECTED_HEAD
```

Important context: this is a visibility verdict for the inspected `46cacc1` HEAD. After that review, local work added and pushed the missing paper-facing visualization code in commit `44ee8e7` (`Add paper-facing selector geometry plots`). Therefore:

- The review's **visibility failure** is stale for the latest pushed branch head.
- The review's **methodological critique** remains valid and should guide the next analysis/plotting work.

## Absorbed Findings

1. Current selector-side metrics are useful but not decisive.

   The review accepts the correlation:

   - GAS-VT fixed384: `44.90` Avg-mAP, both-endpoint coverage @ r16 `0.17164`, boundary recall @ r1 `0.22794`, p95 unselected hole `92.54`.
   - PAction learned fixed384: `59.10` Avg-mAP, both-endpoint coverage @ r16 `0.26471`, boundary recall @ r1 `0.42148`, p95 unselected hole `2.18`.
   - lattice move50/move75: geometry close to PAction, detector mAP pending in that review.

   But this only shows strong correlation. It does not prove that holes or endpoint coverage causally drive detector mAP.

2. The decisive missing evidence is matched instance-level causal accounting.

   For every GT action instance, we need to join:

   - selector geometry,
   - nearest selected start/end distance,
   - bracketed endpoint coverage,
   - detector best proposal tIoU,
   - TP/FN state at high tIoU,
   - start/end localization error.

   The target chain is:

   ```text
   coverage geometry -> proposal quality -> high-IoU mAP
   ```

3. Existing geometry metrics should be retained but expanded.

   Keep:

   - `p95_unselected_hole`,
   - `both endpoint coverage`,
   - `boundary recall @ r1/r16`,
   - region share as auxiliary evidence.

   Add:

   - endpoint nearest distance,
   - bracketed endpoint coverage,
   - action-interior max/p95 hole,
   - boundary-band hole,
   - action-length-stratified geometry,
   - per-class geometry,
   - detector-coupled proposal quality,
   - compute/latency contract.

4. Figure risks to avoid.

   - Missing mAP must stay `NA`, never become zero or participate in misleading heatmap normalization.
   - Diagnostic-only methods such as lattice must not be ranked as paper-main completed methods.
   - Stage2/Stage3 in-progress routes must not appear in completed detector-mAP leaderboards.
   - 768-grid selection must not be described as raw-frame pre-backbone acquisition without raw-frame/pre-feature evidence.
   - Whole-video p95 hole may hide action-local failure; action-local and boundary-local holes are required.
   - Aggregation level must be explicit: action instance, video, window, or row.

5. Paper-level figure plan.

   Main paper should eventually include:

   - task/pipeline figure showing the actual 768-grid acquisition setting,
   - p95-hole vs both-endpoint coverage quadrant with mAP labels,
   - mAP@tIoU decomposition, especially @0.6/@0.7,
   - normalized action-time selection density,
   - geometry-to-proposal-quality/failure plot,
   - compute-quality Pareto plot.

   Appendix should include:

   - raw-value scorecard,
   - delta-vs-baseline bars,
   - per-class breakdown,
   - action-length breakdown,
   - deterministic failure gallery,
   - method taxonomy separating completed/diagnostic/in-progress routes.

6. Strong pre-backbone claim is not currently justified.

   The review is explicit: selecting from the AdaTAD 768 temporal grid is weaker than raw-frame pre-backbone acquisition. The safer phrasing is:

   ```text
   sparse temporal-grid acquisition for AdaTAD/OpenTAD
   ```

   A real pre-backbone claim needs raw-frame or pre-feature evidence showing the expensive backbone is only run on selected frames/snippets.

## Implementation Consequences

The next analysis implementation should prioritize:

1. `selector_geometry_per_instance.csv`

   One row per GT instance and method, with endpoint nearest distances, normalized distances, bracketed coverage, action-local holes, boundary-band holes, and selected density around start/end/interior.

2. `detector_match_per_instance.csv`

   One row per GT instance and method, joining detector output to best proposal tIoU, start/end errors, and TP/FN at 0.5/0.6/0.7.

3. `method_registry.json`

   It must separate:

   - completed detector-mAP methods,
   - diagnostic-only methods,
   - in-progress / not-paper-evidence methods.

4. Updated plots

   - NA-safe figures,
   - diagnostic methods marked as hollow/gray,
   - action-instance-level aggregation by default,
   - no hidden mixing of 384/768/dynamic budgets.

## Current Status After Absorption

Already addressed after the reviewed `46cacc1` commit:

- `tools/bata/plot_selector_paper_summary.py` added.
- `tests/test_plot_selector_paper_summary.py` added.
- `docs/methods/2026-07-08-paper-visualization-and-analysis-pro-prompt.md` added.
- Commit `44ee8e7` pushed to GitHub.

Still open:

- line-by-line Pro review should be rerun against latest `44ee8e7` or newer HEAD;
- per-instance detector matching is not yet implemented;
- raw-frame/pre-feature acquisition evidence is still absent;
- lattice detector mAP and Stage2 full mAP are still running/pending;
- paper claim remains HOLD.
