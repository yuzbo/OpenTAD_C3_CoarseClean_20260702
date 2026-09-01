# Selector Geometry Analysis Suite

This suite implements the Pro-review requirement that sparse selector claims must be supported by selected-centric geometry evidence, not only ledger validation.

## Goal

Given matched-budget selector ledgers, produce reproducible evidence for:

- selected-frame distance to nearest GT boundary;
- selected-frame region share: boundary band / action interior / background / invalid;
- per-action endpoint coverage: start, end, both endpoints;
- region-specific holes;
- normalized action-time selected positions;
- p_action calibration against boundary distance.

## Coordinate Contract

Default convention:

- coordinate system: `dense_frame_index`;
- selected positions: local dense frame centers;
- GT segments: half-open `[start,end)`;
- discrete start boundary: `start`;
- discrete end boundary: `end - 1`;
- radius metrics use frame-center distance.

This must be kept fixed when comparing `uniform_384`, `p_action_topk_384`, `PAction learned_fixed_384`, `GAS-VT fixed_384`, Stage2, and Stage3.

## Core Command

```bash
python tools/bata/analyze_selector_geometry.py \
  --run-tag matched_val_geometry_v1 \
  --split validation \
  --selector-ledger uniform_384=/path/to/uniform_384.jsonl \
  --selector-ledger paction_learned_fixed_384=/path/to/paction.jsonl \
  --selector-ledger gas_vt_fixed_384=/path/to/gasvt.jsonl \
  --common-sample-jsonl /path/to/source.canonical_unique.jsonl \
  --boundary-band-radius 4 \
  --radii-frames 1 2 4 8 16 \
  --out-dir outputs/analysis/matched_val_geometry_v1
```

If methods use different source JSONL files, pass method-specific sample inputs:

```bash
--sample-jsonl uniform_384=/path/to/uniform_samples.jsonl \
--sample-jsonl paction_learned_fixed_384=/path/to/paction_samples.jsonl
```

## Outputs

- `manifest.json`
- `frame_metrics.csv`
- `selected_frame_metrics.csv`
- `video_summary.csv`
- `action_summary.csv`
- `method_summary.csv`
- `holes_by_region.csv`
- `paction_calibration.csv`

## Claim Rule

Do not claim boundary concentration from `boundary_recall@r` alone.

Strong claim requires both:

1. mAP improvement under matched budget and detector settings;
2. selected-centric geometry improvement, such as lower selected-frame boundary distance, higher boundary-band selected ratio, higher endpoint coverage, and lower boundary-region holes.

If geometry does not support this, restrict the claim to empirical selector strength under the tested implementation.
