# CPTC TAR32 terminal evidence and candidate identity manifest

## Candidate diff identity

- Base: `2d945e64bdccd09ae2e2916524562e3f388c5a2a`
- Candidate: `b0a1ca113bec1d8ca66b355f83dbb272bb7b3cb7`
- Branch: `codex/zoomtoken-r1-tar32-fkv-v001`
- Commits: `8eaf68c2` (minimal TAR32 implementation), `b0a1ca11` (pre-run contracts)
- Aggregate diff: 5 files, 1,024 insertions, 7 deletions
- Changed surfaces:
  - `configs/adatad/thumos/georoute_official_r1_tar32_fkv_prebackbone_seed42_v001.py`
  - `opentad/models/backbones/vit_adapter.py`
  - `scripts/run_zoomtoken_r1_tar32_fkv_n16r4.sh`
  - `tests/test_zoomtoken_r1_tar32_fkv.py`
  - `tools/train.py`

The implementation preserves contiguous R1 K64 input support, K64 K/V context and the
existing K64 Adapter. It applies dense K64 updates in even blocks and exact K32
Query/output/MLP updates in odd blocks using the preceding dense attention-column mean.
Unselected tokens take identity residual bypass. There is no cache, new router, new loss,
dynamic cardinality or fallback.

## Pre-run evidence

- N16R4 focused checks: `32 passed, 1 skipped`; strict-R1 regressions: `9 passed`
- Fresh Critic: `PASS`
- Fresh result-blind Evaluator: `PRE_RUN_READY`
- CUDA AMP pre-run job `1260163`: `COMPLETED 0:0`
- Formal training job `1260166`: `COMPLETED 0:0`; valid epoch-59 EMA, but no official final validation
- First evaluation-only job `1261121`: pre-model launcher blocker, scientific attempt 0
- Replacement authority: fresh Pro `REVISE_AND_CONTINUE`, exactly one scheduler replacement
- Replacement Builder/Critic/Evaluator: one-line `find -L` correction / `PASS` / `PRE_RUN_READY_REPLACEMENT`

## Terminal evidence manifest

1. `docs/aris/ZOOMTOKEN_CPTC_TAR32_EVAL_ONLY_REPLACEMENT_TERMINAL_RECEIPT-2026-08-29.md`
2. `docs/aris/ZOOMTOKEN_CPTC_TAR32_TERMINAL_DIAGNOSTICS-2026-08-29.json`
3. remote `launch_receipt.tsv`, SHA-256 `20b83360bd4b76aaf1f9994d94ef42c92b2a8d423022a894b1871013c9027604`
4. remote `terminal_receipt.tsv`, SHA-256 `b07b9644c6b7d99e2c611243a9be8136b2941109f7584d007ca48e16698f73d3`
5. remote `evaluation.log`, SHA-256 `1ecc94904017c0d8f0b7763c50c41dd73fcf86649aa5c2570d0440e6524f0381`
6. remote `gpu2_id0/result_detection.json`, SHA-256 `a70bae55e05219ed1c03b918fb57a4e560c19f0c8c6770908628b9b08ea767df`
7. `docs/aris/ZOOMTOKEN_CPTC_TAR32_EVAL_ONLY_REPLACEMENT_EVALUATOR_RECEIPT-2026-08-29.md`
8. `docs/aris/ZOOMTOKEN_CPTC_TAR32_EVAL_BLOCKER_PRO_REVIEW_RECEIPT-2026-08-29.md`

## Evidence boundaries and anomalies

- Direct: official prediction, official full-validation mAP, run/identity receipts.
- Reconstructed: unrounded official AP vector from the frozen prediction; short-action
  mAP and start/end boundary diagnostics from both frozen predictions.
- Structural: fixed `[64,32]x6` schedule and runtime route assertions.
- Unmeasured: latency, energy, memory, cost, temperature, multi-seed and cross-dataset behavior.
- Missing as a standalone artifact: per-window route ledger and independently counted
  K64/K32/fallback rows. The fixed configuration and terminal receipt assert the route,
  but this is not relabelled as a ledger.
- The original request template said “six-threshold mAP vector”; the official THUMOS
  evaluator emits five thresholds (`0.3`–`0.7`) plus Average mAP. The final request binds
  those actual six scalar values without inventing a sixth threshold.
