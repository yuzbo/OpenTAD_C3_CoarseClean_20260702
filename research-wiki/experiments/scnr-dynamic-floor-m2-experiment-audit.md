---
type: experiment_audit
node_id: audit:scnr-dynamic-floor-m2-terminal-v1
title: "SCNR dynamic floor M2 terminal experiment audit"
stage: tested
status: pass_complete_descriptive_only
added: 2026-08-06
updated: 2026-08-06
---

# SCNR dynamic floor M2 terminal experiment audit

## Verdict

`PASS_COMPLETE_DESCRIPTIVE_ONLY`.

M2 的两个训练臂、完整 accuracy/telemetry 回放、同 GPU 四遍全栈成本回放和
fresh finalizer 均已终局完成，允许读取 seed-3407 的描述性开发结果。该审计不允许
单种子 floor 选择、ROI/residual 互补性、官方测试或论文主张。

本审计是同模型家族的 Type-A 完整性复核；本轮没有可用的 cross-family reviewer
overlay，因此不能表述为 Type-B 独立模型无罪审计。

## Terminal lineage

- model runtime: `6ee97336775a09611f10423e07cafcea375e191a`;
- cost execution: `42923d9f7aaddb14368f82aacda5c77e1f857a24`;
- finalizer execution: `75e2adc86877f002e10626ee4011104b60b0ce49`;
- G1/G2/cost/finalizer Jobs: `1216180/1216181/1222889/1223310`, all
  `COMPLETED 0:0`;
- run root:
  `/data/run01/sczc063/yuzibo/scnr_dynamic_floor_m2_6ee97336_s3407_20260804_0525`;
- finalization status/decision:
  `PASS_COMPLETE_DESCRIPTIVE_FLOOR_SENSITIVITY /
  COMPLETE_DESCRIPTIVE_ONLY_M3_REQUIRED_FOR_FLOOR_SELECTION`;
- finalization internal/file SHA-256:
  `716faa0e354c2d3281cb4be5b033b3540ff03602a0aaa43e4d885ea24dbc1f7f` /
  `17ad8532d1358a10f221b657b42ac783f0e67b5317f81317507bfbf90aa564aa`;
- cost profile internal/file SHA-256:
  `df5529b09cf4e5fba748f11e22df4ee3f1fc68a327eee1ee4b1885d7b29ec443` /
  `2a556d2a4ddc5822b66a6c3efe1bcddd27a7d7add8e0d7af50e3aec320802983`.

## A--F integrity gates

| Gate | Result | Evidence |
| --- | --- | --- |
| A. Population and no-leak | PASS | Same complete 136-window development population; route GT/teacher/oracle/raw-cache flags false |
| B. Training and checkpoint lineage | PASS | Both 60-epoch arms ended at epoch 59 with 9,600 successful updates and one final EMA checkpoint |
| C. Exact dynamic execution | PASS | `B=24576`, true ragged, zero padding, one heavy forward, dynamic `K_t`, masked-zero carrier |
| D. Accuracy and telemetry | PASS | Both stage-result hashes and raw accuracy/telemetry receipts reproduce |
| E. Full-stack cost | PASS | One GPU, `G1->G2->G2->G1`, 272 samples/arm, raw 20-ms NVML trace and energy reintegration |
| F. Atomic finalization and claim ceiling | PASS | Fresh finalizer has `errors={}` and leaves floor selection/test/paper guards closed |

## Descriptive development results

| Metric | G1 1-cell | G2 2-cell | G1-G2 (pp) |
| --- | ---: | ---: | ---: |
| Avg-mAP | 10.95 | 5.17 | +5.78 |
| mAP@0.3 | 13.79 | 9.40 | +4.39 |
| mAP@0.4 | 12.79 | 7.64 | +5.15 |
| mAP@0.5 | 11.51 | 4.60 | +6.91 |
| mAP@0.6 | 9.41 | 2.79 | +6.62 |
| mAP@0.7 | 7.25 | 1.43 | +5.82 |
| high-IoU composite | 8.33 | 2.11 | +6.22 |

Aggregate cost uses two passes per arm:

| Metric | G1 | G2 | G1-G2 |
| --- | ---: | ---: | ---: |
| end-to-end p50 | 3453.349 ms | 3357.831 ms | +95.518 ms (+2.845%) |
| end-to-end p95 | 4420.017 ms | 4324.377 ms | +95.640 ms (+2.212%) |
| model-forward p50 | 699.738 ms | 696.687 ms | +3.051 ms (+0.438%) |
| gross GPU energy | 58517.466 J | 57608.674 J | +908.793 J (+1.578%) |
| peak allocated | 1521.784 MB | 1523.436 MB | -1.652 MB |

The aggregate end-to-end/energy difference is not a clean model-cost effect.
G1 pass 0 was a cold host/input outlier, while both middle G2 passes were stable;
G1 pass 3 versus G2 pass 2 differs by only `18.315 ms` at end-to-end p50, and the
aggregate model-forward difference is only `0.340%` by mean. A future matched
cost confirmation must mirror `ABBA` and `BAAB` order or otherwise randomize
first-arm cold state.

## Mechanism finding and claim impact

- Dynamic allocation is operational: G1/G2 both have mean `K_t=64`, with
  `K_t=0` rates `1.4668%` and `1.1853%` respectively.
- The hard role policy collapsed. Across `3,342,336` selected tokens per arm,
  G1 context/ROI/residual counts are `0/7/3,342,329`; G2 counts are
  `0/0/3,342,336`.
- Neither floor bound was active during evaluation. Width and height floor
  saturation are `0` in both arms; observed minimum extents remain above their
  configured floors.
- Therefore the large G1 descriptive accuracy lead is real within this seed but
  cannot be causally attributed to the 1-cell floor. M2 does not demonstrate an
  operational ROI+TokenSelect Hybrid or ROI/residual complementarity; it mostly
  evaluated a dynamic residual selector under two separately trained runs.
- M3 floor confirmation is required by the frozen finalizer but is not yet
  scientifically admissible. First diagnose modifier scales/margins and recover
  role identifiability without fixed quotas; only then freeze disjoint seeds.

## Audit trace

The first Type-A pass preserved the incomplete-finalizer finding in
`.aris/traces/experiment-audit/2026-08-06_run01`. The terminal re-audit passed
A--F in `.aris/traces/experiment-audit/2026-08-06_run02`; its only warning was
the stale `experiment_running` wiki state, resolved by the terminal documentation
update that accompanies this report.
