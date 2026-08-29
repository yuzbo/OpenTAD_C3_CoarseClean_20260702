# ZoomToken BPNS-R1 v004 终态证据收据

## 1. 冻结身份与终态

- 任务：`ZOOMTOKEN-BPNS-R1-DECOUPLED-DIAGNOSTIC-AND-COST-CLOSURE-v004`
- base：`8a59d655005b9030d8ea5dc17ee2620844cb587b`
- candidate：`a4694019fd4cbbdc74885e160163e23d947dc05f`
- branch：`codex/zoomtoken-bpns-r1-decoupled-cost-v004`
- source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_a4694019_v004`
- result：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_decoupled_v004_a4694019_seed42_20260829`
- Slurm job：`1260095`（`zt-bpns-v004-a4694019`）
- node/time：`g0059`，`2026-08-29 00:37:38–06:10:03 +08:00`
- Slurm：`COMPLETED 0:0`，elapsed `05:32:25`
- precheck：job `1260092`，`COMPLETED 0:0 / PRECHECK_READY`
- focused tests：N16R4 `27 passed`；fresh Critic `PASS`；fresh result-blind Evaluator `PRE_RUN_READY`。

正式提交次数恰好为 1。没有 retry、resume、第二 seed、辅助臂、阈值修改、official test 访问或训练。

## 2. 完整性与身份核验

八个 pass 严格按 `K100,R1,R1,K100,R1,K100,K100,R1` 完成。每个 pass 均包含 792 个有序 population 项、raw cost、prediction/evaluator、prediction SHA、pass receipt 与功耗覆盖；总计 6,336 条成本行和 929,889 条功耗行。每臂四次 prediction SHA 完全一致，并等于冻结锚点：

- K100：`008daf5a55af90318506e913c13a4bd2d6ce8ff17a45cc8e856f5619eaa45eb7`
- R1：`ffc78393e4097a578def8fdd62ffe4f36dd87c2dddd52de9b3ae248cb108c734`

`terminal_receipt.json` 标记 `COMPLETED_FINAL_EMA_REPLAY`，cost/acquisition/diagnostic/prediction identity 均为 true；`measurement_completeness.status=COMPLETE`。GPU temperature 未测量，不能据功耗顺序推断热漂移。

## 3. 冻结主估计的独立重算

下表直接从 `cost_samples.jsonl` 按 `(arm, pass_index)` 分组重算。延迟单位为毫秒，能耗为完整 pass 的 gross joules；吞吐率为 items/s，显存为 MiB。

| pass | arm | p50 | p95 | gross energy | throughput | peak allocated | peak reserved |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | K100 | 2527.7848 | 4454.7541 | 147714.0313 | 0.376879 | 1860.1543 | 2268 |
| 1 | R1 | 2422.8567 | 4233.6837 | 134177.1895 | 0.395134 | 1542.6636 | 1802 |
| 2 | R1 | 2374.3897 | 4232.3048 | 132201.3231 | 0.399818 | 1349.4282 | 1622 |
| 3 | K100 | 2477.9377 | 4396.3738 | 144212.6770 | 0.384647 | 2052.8740 | 2494 |
| 4 | R1 | 2466.2469 | 4324.2198 | 135466.4916 | 0.387872 | 1543.0542 | 1720 |
| 5 | K100 | 2482.3977 | 4340.1174 | 144143.7687 | 0.385185 | 2054.5928 | 2494 |
| 6 | K100 | 2481.5172 | 4353.8728 | 144176.2094 | 0.386059 | 2054.3115 | 2494 |
| 7 | R1 | 2487.6161 | 4497.1248 | 137921.9607 | 0.380231 | 1543.5854 | 1720 |

每臂四次 pass 的中位数：

| arm | p50 | p95 | gross energy | throughput | peak allocated | peak reserved |
|---|---:|---:|---:|---:|---:|---:|
| K100 | 2481.9575 | 4375.1233 | 144194.4432 | 0.384916 | 2053.5928 | 2494 |
| R1 | 2444.5518 | 4278.9518 | 134821.8406 | 0.391503 | 1542.8589 | 1720 |

R1/K100 比值：p50 `0.9849289616`、p95 `0.9780185512`、gross energy `0.9350002508`、throughput `1.0171116288`、peak allocated `0.7512973880`、peak reserved `0.6896551724`。

冻结判据要求 p50 与 gross energy 两项比值均 `<=0.95`。能耗通过（下降 `6.50%`），p50 失败（仅下降 `1.51%`）。因此协议规定的终态分类是：

> **`STOP_BPNS_R1_EFFICIENCY_HEADLINE`**

BPNS-R1 不能作为当前论文的真实端到端效率 headline；不得把 36% token 减少、显存下降或能耗单项通过改写为总体加速成立。

## 4. 准确率、短动作与边界诊断

同一固定 seed/checkpoint 下，R1−K100 的 Avg-mAP 与 mAP@0.3/0.4/0.5/0.6/0.7 差值依次为：

`+0.5352577894 / +0.7520311665 / +0.1518204238 / +1.6237860133 / -0.1041569533 / +0.2528082968 pp`。

它支持“未出现广泛准确率崩塌”的固定种子诊断，但结果混合且不能替代多 seed 证据。

K100/R1 的 short-action Avg-mAP 为 `0.4112924396/0.4115554712`；matched boundary 为 `3293/3287`（总数 3325）。start error 为 `0.1499703291/0.1491006470`，end error 为 `0.1309378935/0.1331244103`。短动作 matched 为 `2176/2172`（总数 2208），short recall@0.7 为 `0.6512681159/0.6548913043`。差异细小且方向混合，不能主张边界保护成立。

## 5. 异常与证据边界

第 3 号 K100 pass 的 in-pass power trace 最大采样间隙为 `2804.8198968172073 ms`；其余 pass 为约 `52–71 ms`。覆盖比仍为 1.0，能耗通过线性插值积分；冻结协议没有最大 gap 阈值，因此该轮形式上有效，但这是一项必须向 Pro 披露的能耗不确定性。它不改变 p50 门失败，也不授权重放。

本轮只建立单 seed、单 checkpoint、单硬件上的固定条件观察。显存和能耗下降可作为测量事实报告，但不能外推到多硬件、吞吐批量、数据集或总体论文效率。结构代理、直接测量、可重建量和未测量温度必须分开表述。

## 6. 原始产物身份

- `terminal_receipt.json`：SHA-256 `ba769e578f2b998e5dc32d9c62eb6bee2046fb8307979eaf6a5b18891e8d0283`，29,938 bytes。
- `profile.json`：`bf798346bcfb5afdf8f459631ff5bc86465eafee6e12b3b9a5a3735949293694`，253,948 bytes。
- `acquisition_state.json`：`0462e7348d654e428d9588143aa1d597dc5fd6f592482068cb75098d011697e4`，966 bytes。
- `cost_samples.jsonl`：`2b6039178536c0cd679d5e182f2585d32f28ae02c916b38244ca80266ab4c6ee`，4,673,655 bytes / 6,336 lines。
- `power_trace.jsonl`：`494b2ec2192224f0ea2c3e4afd51fed90baa6a9531fab7a36fc6c30b3c35bc5e`，96,911,043 bytes / 929,889 lines。
- `pass_receipts.json`：`732c075d99019479d4d3e10b735c67d36a2bf0153f2becc7181df4c239d9331d`，122,502 bytes。

终态后未修改阈值、重排、恢复、重跑或创建 v005。下一动作只允许一次 fresh exact-Project Pro 独立复盘。
