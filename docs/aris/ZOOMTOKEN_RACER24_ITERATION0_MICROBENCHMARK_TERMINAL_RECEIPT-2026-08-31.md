# ZoomToken-RACER24 Iteration-0 微基准终态回执

## 终态

- exact candidate：`5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`
- scientific-attempt job：`1262068`
- Slurm：`FAILED 2:0`，node `g0041`，elapsed `00:01:21`
- 协议终态：`VALID_COMPLETE_NEGATIVE_GATE_RESULT`
- terminal receipt：`status=failure`、`step=matched_block_profile`；退出码来自冻结门失败，而非执行缺失。
- focused target-environment tests：`16 passed in 54.26s`
- raw profile：`/data/run01/sczc063/yuzibo/results/zoomtoken_racer24_i0_5ebaa74f_20260831/profile.json`
- raw terminal receipt：`/data/run01/sczc063/yuzibo/results/zoomtoken_racer24_i0_5ebaa74f_20260831/terminal_receipt.json`

执行偏差：微基准运行前，本机 GitHub HTTPS push 失败后，未推送的 exact commit 通过增量 Git bundle 部署到 remote clean checkout；这违反 RTK 的“bundle 只传递已推送对象”同步纪律。运行后、fresh Pro 提交前，同一 branch/commit 已推到 GitHub，ref API 与 commit API 在 `2026-08-31T03:03:44+08:00` 均核验为 `5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`：branch [`codex/zoomtoken-racer24-v001`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-racer24-v001)，exact commit [`5ebaa74f611bb3a43c3042700a78b92a9e5e74fb`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/5ebaa74f611bb3a43c3042700a78b92a9e5e74fb)。事后补推不抹去实验运行时尚未推送的时间顺序；该偏差继续交 fresh Pro 裁决证据等级。

首次 job `1262067` 仅因 `/bin/sh` wrapper 无 `source/module/python` 而在 1 秒内退出；没有 Python、模型或 CUDA 执行。独立 Evaluator 将其定性为 `PRE_EXECUTION_OPERATIONAL_BLOCKER` 并准入一次显式 `/bin/bash -lc` replacement。它不计科学尝试，也不提供性能证据。

## 冻结测量

| 指标 | matched dense R1 | RACER24 | RACER24 / dense 或 speedup |
| --- | ---: | ---: | ---: |
| p50 latency | `1.3347729109 ms` | `5.3468430415 ms` | speedup `0.2496375713x` |
| p95 latency | `1.3655385468 ms` | `5.5123582017 ms` | — |
| peak allocated | `17,757,696 B` | `35,317,248 B` | `1.9888417957x` |
| peak reserved | `27,262,976 B` | `50,331,648 B` | `1.8461538462x` |

协议形状为 `B=1`、8 tubelets、64 token/tubelet、24 selected/tubelet、总 `Q=192`、`K/V=512`、embed 384、6 heads；warmup 50，200 timed repetitions。scope 是 `matched_block_only_not_full_stack_tad`。

## 冻结门与证据含义

- p50 speedup 门 `>=1.08x`：**失败**。
- allocated/reserved memory ratio 门 `<=1.05`：**两项均失败**。
- `result-to-claim` 独立判断：`claim_supported=no`、confidence `high`、`review_independence=same-family`、`acceptance_status=provisional`。

该结果直接否定“当前 RACER24 实现在 matched real-shape block path 上具备最低工程效率可行性”。它不提供准确率、训练、full-stack TAD、能耗、跨硬件或发表级方法优越性证据，也不能外推为所有 selected-Q/full-KV 或 residual-completion 方法必然失败。

冻结动作：`STOP_RACER24_ITERATION0_AND_RETURN_TO_PRO`。不得调 K、改 blocks、训练、测 full-stack cost、实现 FARM24/PairLatent32 或自行设计 successor。下一科学责任人是 Pro。
