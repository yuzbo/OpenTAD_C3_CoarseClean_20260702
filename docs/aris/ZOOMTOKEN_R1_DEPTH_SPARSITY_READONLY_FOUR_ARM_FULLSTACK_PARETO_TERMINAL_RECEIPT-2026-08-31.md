# ZoomToken R1 depth/sparsity four-arm full-stack Pareto terminal receipt

## 1. Frozen identity

- Task: `ZOOMTOKEN-R1-DEPTH-SPARSITY-READONLY-FOUR-ARM-FULLSTACK-PARETO-CLOSURE-v001`.
- GitHub repository: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>.
- Branch: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-r1-depth-pareto-v001>.
- Exact implementation: <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b82441c1aa2663069033d394794298d5c723bbb6>.
- Formal job: `1262120` / `zt-r1dp-s42`; submission `1/1`; no retry, resume, replacement, extra seed, arm or training.
- Slurm terminal: `COMPLETED 0:0`; node `g0063`; start `2026-08-31T06:37:20+08:00`; end `2026-08-31T21:33:40+08:00`; elapsed `14:56:20`.
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_depth_pareto_b82441c1_seed42_20260831`.
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_r1_depth_pareto_logs_b82441c1_20260831`.
- Hardware/software: NVIDIA RTX 4090 on `g0063`, GPU UUID `GPU-aa004f54-2d8d-1907-022f-e79fe4bbe4e1`, driver `550.135`, 450 W limit; Python `3.10.20`, PyTorch `2.0.1`, CUDA `11.8`, MMEngine `0.10.7`, CUDA autocast fp16, cuDNN deterministic enabled.

## 2. Protocol completeness

- Four frozen epoch-59 EMA arms were replayed without training or resume:
  - A: R1/FULL64 (`job 1249099`);
  - B: DSR6-KV (`job 1252527`);
  - C: MOD32-KV (`job 1252180`);
  - D: DROP32 (`job 1252179`).
- Complete THUMOS14 validation population: 211 videos / 792 ordered windows in every pass. Official test remained unopened.
- Sixteen passes completed in the frozen Williams order `A B D C / B C A D / C D B A / D A C B`; 50 warmup windows preceded every pass.
- `12,672 = 16 × 792` full-stack cost rows were written. Each pass has the same ordered population and unique item identities.
- Sixteen prediction files, sixteen raw evaluator vectors, 16 pass receipts, `profile.json`, `terminal_receipt.json`, raw cost rows, raw power trace, short-action diagnostics and boundary diagnostics are present. Each arm's four prediction SHA256 values are internally identical.
- Raw power evidence contains `2,527,501` finite, monotonic samples. Every pass reports `COMPLETE` coverage, zero invalid samples and zero uncovered fraction; the largest raw sampling gap is `78.9527 ms`, below the frozen `3000 ms` gate.
- Recomputing pass-local summaries from raw rows and then taking the median over each arm's four passes reproduces `profile.json` exactly. Terminal decision and profile decision agree.
- `terminal_receipt.json` reports no anomalies. The formal stderr contains only environment/launcher notices; targeted scans found no Traceback, OOM, non-finite value, RuntimeError, ValueError, assertion failure, kill or NCCL error.

Prediction SHA256 by arm:

- A: `ffc78393e4097a578def8fdd62ffe4f36dd87c2dddd52de9b3ae248cb108c734`.
- B: `1283bd695645250b3f88a0501c8a8015f69462545b209da27b67ab154a33abca`.
- C: `d85c2c14ba78a1f6ac91baf3b79043a93d860c8315914e5cf0f8da12a5621917`.
- D: `ed70976256ae35d929ebec5c94d43cc18643a7d70d15251d3dec0d0d71640b98`.

## 3. Directly measured full-stack result

The arm estimate is the median of four complete pass-local estimates. Ratios are relative to A/R1-FULL64.

| Arm | p50 (ms) | Gross energy (J) | Peak allocated (MB) | Peak reserved (MB) | Throughput (windows/s) | p50 ratio | Energy ratio | Allocated ratio | Reserved ratio | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A R1/FULL64 | 2848.736980 | 165094.263331 | 1544.179199 | 1722 | 0.328072 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | control |
| B DSR6-KV | 3168.720036 | 182052.613843 | 1447.131836 | 1650 | 0.297955 | 1.112325 | 1.102719 | 0.937153 | 0.958188 | FAIL |
| C MOD32-KV | 3162.700302 | 181582.699748 | 1542.694824 | 1724 | 0.297900 | 1.110211 | 1.099873 | 0.999039 | 1.001161 | FAIL |
| D DROP32 | 3141.607514 | 175727.816833 | 1141.896973 | 1430 | 0.301083 | 1.102807 | 1.064409 | 0.739485 | 0.830430 | FAIL |

Frozen survivor rule: relative to A, median-four p50 `<= 0.95`, complete-pass gross energy `<= 0.95`, peak allocated memory `<= 1.05`, and peak reserved memory `<= 1.05`, all conjunctively. No candidate survives. B and C reduce little or no memory while worsening latency and energy; D reduces memory materially but still worsens latency and energy.

Deterministic terminal decision:

`STOP_R1_FIXED_DEPTH_SPARSITY_FOURPOINT_AS_CURRENT_EFFICIENCY_ROUTE`

## 4. Diagnostic quality evidence, not a cost gate

Historical-checkpoint raw validation diagnostics for Avg-mAP / mAP@0.6 / mAP@0.7 are:

- A: `69.042148 / 61.086961 / 46.499472`;
- B: `67.367708 / 59.334273 / 46.021908`;
- C: `66.526569 / 59.257047 / 45.231457`;
- D: `66.108813 / 57.823637 / 44.883521`.

Short-action Avg-mAP and normalized start/end boundary errors are:

- A: `0.411555 / 0.149101 / 0.133124`;
- B: `0.397457 / 0.149862 / 0.134573`;
- C: `0.392901 / 0.153364 / 0.134618`;
- D: `0.383943 / 0.158420 / 0.135457`.

These values only diagnose the four frozen historical checkpoints. They do not establish matched-training accuracy, official-test performance, multi-seed generalization, boundary protection, or a positive paper claim.

## 5. Evidence classes and claim boundary

- Directly measured: decode-to-Soft-NMS pass-local latency, throughput, allocated/reserved CUDA memory, raw NVML power and complete-pass gross energy on this GPU and complete validation population.
- Reconstructible: arm medians, ratios, power coverage and the deterministic conjunction of the frozen gates.
- Structural/diagnostic: checkpoint-specific validation vectors, short-action and boundary diagnostics, and known architectural sparsity. They do not replace measured systems cost.
- Unmeasured: GPU temperature. Final video-level NMS is performed once per video and amortized over that video's window rows; this remains an explicit limitation of the row-level full-stack accounting.
- Supported conclusion: none of these three fixed depth/sparsity points is a latency-energy-memory Pareto survivor against R1/FULL64 under the frozen same-GPU, complete-validation protocol.
- Not supported: official-test, training, Online TAD, multi-seed, cross-hardware, cross-dataset, cross-detector, universal depth-sparsity failure, or boundary-protection claims.

## 6. Independent result-to-claim review

A fresh read-only independent reviewer classified the intended positive survivor claim as unsupported and the fixed-route STOP as a high-confidence negative systems conclusion, conditional on this hardware/configuration/population. It separately confirmed that D's memory reduction cannot conceal its p50 and energy failures, and that this result must not be generalized beyond the frozen four points.

## 7. Required next action

No experiment, repair, replacement or successor is authorized by this receipt. The complete terminal evidence, negative result, limitations, independent review and exact GitHub repository/branch/commit must be submitted once to a fresh exact ZoomToken Project Pro conversation. Pro must independently adjudicate the scientific meaning, KEEP or REVISE the role contract, and return exactly one next task with explicit Beijing deadlines.
