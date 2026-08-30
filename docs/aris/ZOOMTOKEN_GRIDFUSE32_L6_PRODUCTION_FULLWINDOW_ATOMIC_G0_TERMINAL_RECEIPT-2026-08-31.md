# ZoomToken GridFuse32-L6 production-full-window atomic G0 terminal receipt

## 1. Immutable identity

- Task: `ZOOMTOKEN-GRIDFUSE32-L6-PRODUCTION-FULLWINDOW-ATOMIC-G0-v001`
- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-gridfuse32-l6-v001
- Exact implementation commit: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dde46aee17f10bf793e5407055fc7b3416d93205
- Source commit: `dde46aee17f10bf793e5407055fc7b3416d93205`
- Remote source: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_src_dde46aee`
- Result root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_production_fullwindow_atomic_g0_final_20260831`
- Log root: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_production_fullwindow_atomic_g0_logs_20260831`
- Formal Slurm job: `1262108` / `zt-gf32-fw-g0-final`
- Frozen ordinals: scheduler action `2`; scientific measurement `1`
- Formal submission count: `1/1`; replacement and ordinal 3 are forbidden.

The repository, branch and exact commit URLs above are the only code-review identity for the mandatory fresh Pro adjudication. A moving branch head, local worktree or later commit must not substitute for the exact commit.

## 2. Scheduler terminal state

Authoritative `sacct` terminal row:

```text
1262108|zt-gf32-fw-g0-final|gpu|FAILED|3:0|00:05:33|2026-08-31T05:38:35|2026-08-31T05:44:08|g0041|billing=4,cpu=4,gres/gpu=1,mem=62200M,node=1
```

The launcher deliberately returned a non-zero exit after writing a complete failed-gate profile and terminal receipt. Therefore Slurm `FAILED 3:0` is not an execution or artifact blocker. It is a protocol-complete, valid G0 negative.

## 3. Construction, checkpoint and shape witness

- Detector construction: passed.
- Canonical transforms registered: `Rearrange`, `Reduce`, `Interpolate`.
- Checkpoint: epoch 59 `state_dict_ema`.
- Checkpoint SHA256: `3977e890b6f3990225d7d6818670aef1243fd024f4eb0f0f7e82344a48807599`.
- Strict load: passed with no missing or unexpected keys.
- Real shape: `B=1`, `T=384` tubelets, `K=64`, `N=24576`, `D=384`, 6 heads.
- Blocks measured: `6,7,8,9,10,11`.
- Buckets: 48; eight tubelets per bucket.
- Dense/candidate tokens per bucket: `512/256`.
- Adapter: dense, temporal length 384.
- Dtype: `fp16_autocast`.
- Construction witness did not execute timing, memory measurement, predictions, metrics, training or resume.

Frozen dry and final ledgers agree:

| quantity | dense | candidate |
|---|---:|---:|
| bucket calls | 288 | 288 |
| attention tokens | 147,456 | 73,728 |
| K/V tokens | 147,456 | 73,728 |
| MLP tokens | 147,456 | 73,728 |
| attention pairs | 75,497,472 | 18,874,368 |
| Adapter tokens | 147,456 | 147,456 |
| restored native tokens per block | 24,576 | 24,576 |

## 4. Atomic G0 measurement

- Protocol: synchronized, alternating arms, no candidate-only compilation.
- Warmup: 100 iterations per arm.
- Timed samples: 500 per arm.

| measurement | dense | candidate |
|---|---:|---:|
| p50 latency | 178.500099 ms | 314.885696 ms |
| p95 latency, report-only | 179.724905 ms | 316.611830 ms |
| mean latency | 178.497078 ms | 314.812821 ms |
| min latency | 176.684036 ms | 312.095734 ms |
| max latency | 180.188293 ms | 317.785095 ms |
| peak allocated memory | 262,924,800 B | 262,924,800 B |
| peak reserved memory | 327,155,712 B | 327,155,712 B |

Frozen gates:

- p50 speedup `dense/candidate = 0.5668726817907567`, required `>=1.35`: **failed**. The candidate latency is approximately `1.764x` the dense latency.
- allocated-memory ratio `1.0`, required `<=1.05`: passed.
- reserved-memory ratio `1.0`, required `<=1.05`: passed.
- Combined G0: **failed**.

## 5. Terminal classification and claim boundary

Terminal status is exactly:

`STOP_GRIDFUSE32_L6_EXACT_ROUTE_VALID_G0_NEGATIVE`

This directly rejects the production-full-window GridFuse32-L6 exact implementation as an efficiency-feasible route under the frozen G0 contract. The structural ledger reduction did not translate to physical latency or memory improvement in this implementation. The result does not establish accuracy, boundary, end-to-end decode-to-NMS, energy, training, multi-seed, cross-hardware, cross-detector or family-wide claims. It does not prove that all token fusion or all structured sparse execution is slow.

Per the frozen Pro contract, no repair, rerun, replacement, ordinal 3, G1, G2, training, sweep or successor is authorized. The only next action is one fresh exact-Project Pro adjudication with the complete terminal evidence and exact GitHub code identity.

## 6. Raw evidence

- Remote profile: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_production_fullwindow_atomic_g0_final_20260831/g0/profile.json`
- Remote terminal receipt: `/data/run01/sczc063/yuzibo/projects/zoomtoken_gridfuse32_l6_production_fullwindow_atomic_g0_final_20260831/g0/terminal_receipt.json`
- Local attachment copy of profile: `.cvpr-pro-lab/reviews/ZOOMTOKEN_GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_ATOMIC_G0_PROFILE-2026-08-31.json`
- Local attachment copy of terminal receipt: `.cvpr-pro-lab/reviews/ZOOMTOKEN_GRIDFUSE32_L6_PRODUCTION_FULLWINDOW_ATOMIC_G0_RAW_TERMINAL-2026-08-31.json`

