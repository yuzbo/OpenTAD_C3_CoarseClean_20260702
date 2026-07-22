# GeoRoute-AdaTAD Result Schema v1

**Machine-readable schema:** `georoute-paper-result-v1`
**Validator:** `tools/bata/georoute_result_schema.py`
**Analyzer:** `tools/bata/analyze_georoute_results.py`

This schema is a result-ingestion contract, not a model configuration.  Its
purpose is to prevent a figure or table from silently combining incompatible
budgets, split roles, cost scopes, estimators, or evidence identities.

## Record Envelope

Each JSON or JSONL record must contain the following top-level fields.

| Field | Required value | Reason |
| --- | --- | --- |
| `schema_version` | `georoute-paper-result-v1` | Reject ambiguous historical output |
| `study_id` | one non-empty study identifier per input package | Prevent mixing campaigns |
| `stage` | `P0`, `P1`, `P2`, or `P3` | Preserve the evidence stage |
| `split_role` | `development` or `official_test` | Enforce sealed-test separation |
| `dataset`, `detector`, `variant`, `seed` | explicit identity | Prevent cross-method/seed aliasing |
| `budget` | source, selected, and declared exact-K values | Make same-budget claims auditable |
| `metrics` | Avg-mAP and mAP@0.3--0.7 | Prevent loose-IoU-only reports |
| `cost` | p50/p95, memory, energy, and complete scope flags | Prevent FLOPs-only efficiency claims |
| `evidence` | runtime/config/checkpoint/prediction/receipt hashes | Bind values to artifacts |
| `diagnostics` | native-route and packed-path audit values | Prevent accidental crop-resize/two-backbone claims |
| `policy_estimator` | estimator semantics | Keep score-function and ST claims separate |

## Exact-K Semantics

For all sparse matched variants, `selected_tokens_per_tubelet` must equal
`tokens_per_tubelet`.  `dense_native` is an upper-compute reference and must
set all three quantities consistently:

```text
tokens_per_tubelet == selected_tokens_per_tubelet
                    == source_tokens_per_tubelet
```

Consequently, a dense-native point is intentionally not drawn as an equal-K
competitor to a `K=64` sparse route.  It may provide an upper-cost reference
but cannot be used to attribute an adaptive-routing gain.

## Mandatory Cost Scope

Every boolean in `cost.scope` must be true:

```text
decode, preprocess, host_to_device, scout, route, patch_embed,
backbone, adapter, detector, nms
```

The schema does not accept a partial FLOP ledger as an end-to-end result.  The
units are milliseconds per window for latency, MiB for peak memory, and joules
per window for gross GPU energy.  Hardware/device provenance belongs in the
run receipt referred to by `run_receipt_sha256`.

## Estimator Guard

- `score_function` requires `score_function_kat_passed: true`; the finite
  known-answer test is necessary but not sufficient evidence for training
  usefulness.
- `straight_through` requires
  `estimator_bias_label: "biased_surrogate"`; no downstream table may relabel
  it as an unbiased policy gradient.
- `none` is valid for deterministic controls.

## Development/Test Separation

Use the analyzer as follows before the official test is opened:

```bash
python tools/bata/analyze_georoute_results.py \
  --input development_records.jsonl \
  --output /outside/repository/georoute_dev_analysis.json \
  --development-only
```

The flag causes the validator to reject every `official_test` record.  The
plotter verifies that `analysis.input_records_sha256` equals the canonical hash
of the supplied validated records.  A plot cannot be regenerated from a
different input under an old analysis file.

## Minimal Example

```json
{
  "schema_version": "georoute-paper-result-v1",
  "study_id": "georoute-p1-dev-commit",
  "stage": "P1",
  "split_role": "development",
  "dataset": "THUMOS14-dev",
  "detector": "AdaTAD-derived",
  "variant": "roi_residual",
  "seed": 3407,
  "budget": {
    "tokens_per_tubelet": 64,
    "source_tokens_per_tubelet": 240,
    "selected_tokens_per_tubelet": 64
  },
  "metrics": {
    "average_map": 0.0,
    "map_by_tiou": {"0.3": 0.0, "0.4": 0.0, "0.5": 0.0, "0.6": 0.0, "0.7": 0.0}
  },
  "cost": {
    "end_to_end_p50_ms": 1.0,
    "end_to_end_p95_ms": 1.0,
    "peak_memory_mb": 1.0,
    "gross_gpu_energy_j": 1.0,
    "scope": {
      "decode": true, "preprocess": true, "host_to_device": true,
      "scout": true, "route": true, "patch_embed": true,
      "backbone": true, "adapter": true, "detector": true, "nms": true
    }
  },
  "evidence": {
    "runtime_commit": "...", "config_sha256": "...",
    "checkpoint_sha256": "...", "prediction_sha256": "...",
    "run_receipt_sha256": "..."
  },
  "diagnostics": {
    "one_heavy_backbone_forward": true,
    "uses_grid_sample": false,
    "uses_resized_local_crop": false,
    "packed_attention_tokens": 64,
    "packed_mlp_tokens": 64
  },
  "policy_estimator": "score_function",
  "score_function_kat_passed": true
}
```

The numeric zeros in this example are schema placeholders, not reported model
results.  Real records require finite measured values and artifact hashes.
