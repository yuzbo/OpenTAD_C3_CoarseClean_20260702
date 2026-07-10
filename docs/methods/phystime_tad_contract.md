# PhysTime-TAD Runtime Contract

PhysTime-TAD is an offline detector for irregular feature observations. It is
not a DUCA selector, dynamic-budget policy, or online TAD model.

## Required metadata

- `phystime_timestamps_sec`
- `phystime_support_intervals_sec`
- `phystime_duration_sec`
- `phystime_domain_start_sec`
- `phystime_domain_end_sec`
- `phystime_support_provenance`
- `gt_time_unit="seconds"` during training
- `prediction_time_unit="seconds"`
- `irregular_native_axis=True`

The observation mask is padding only and must be a valid prefix. Missing
regions are represented by absent support mass, not holes in the padding mask.

## Fail-closed conditions

The model rejects selected-axis GT remapping, unverified support provenance,
teacher/oracle fields, prediction caches, offline ledgers, actionness inputs,
and budget/selector metadata. Raw sparse frames are not accepted as a single
feature token unless a contiguous decoded support interval is auditable.

## Coordinate behavior

Targets, query cells, decoded proposals, NMS inputs, and evaluation outputs use
absolute video seconds. `convert_to_seconds` performs only duration clamping for
PhysTime predictions and never applies snippet-stride or selected-axis inverse
mapping.

## Evidence status

The Gate 0B configuration is the first deployable feature-token geometry track.
Passing it establishes software-contract correctness only. Paper claims require
matched robustness and accuracy experiments across observation counts and gap
patterns.
