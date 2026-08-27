# ZoomToken BPNS-R1 同硬件成本回放 v002 终态回执

- 日期：2026-08-28
- 科学状态：`REPLAY_ADMISSION_FAILURE_NO_COST_RESULT`
- Slurm Job：`1258299`（`zt-bpns-r1-pv2-e9323448`）
- 候选：`e9323448f6cd78b99bb3de53fd9ffb55f3676d65`
- 节点与时间：`g0048`，`2026-08-28 00:20:24–01:33:30 +08:00`，`01:13:06`
- Slurm 终态：`FAILED 1:0`

## 精确终止原因

首个 R1 pass 在 accuracy-parity gate 处抛出 `RuntimeError`：

```text
R1 final-EMA replay differs from its historical result:
mAP@0.6=61.0869609029443100 pp,
expected 61.14 pp,
difference 0.0530390970556900 pp exceeds 0.05 pp
```

候选代码以未舍入 percentage points 做比较，只在 `difference > tolerance` 时失败；focused
test 明确验证 `0.05` 通过、`0.050001` 失败。因此该异常符合已冻结的 inclusive `0.05 pp`
合同，不是单位、比较方向或展示舍入错误。

## 终态产物核验

- 八 pass 顺序 `K100,R1,R1,K100,R1,K100,K100,R1` 未完成；失败发生在首个 R1 pass。
- exact result root
  `/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_cost_parity_e9323448_seed42_20260828`
  存在但为空。
- `profile.json`、`terminal_receipt.json`、两臂 predictions、
  `short_action_validation_gt.json`、`cost_samples.jsonl`、`power_trace.jsonl` 均不存在。
- 日志根只存在 `.out`（1130 bytes）与 `.err`（3300 bytes）；它们是执行/异常证据，不是性能或成本证据。

## 证据边界与下一动作

该轮只证明 replay 没有通过冻结的数值一致性准入门。它不是 K100/R1 模型性能失败，也没有形成
延迟、吞吐、显存、能耗、短动作或边界结果。不得读取或拼接失败前局部数值，不得补造终态产物、
放宽门槛、resume 或 duplicate。按照既有 Pro post-result 规则，下一动作是一次全新的中性 Pro
科学复盘：提交冻结合同、候选/准入证据、精确异常、空产物清单和全部已知协议偏差，由 Pro 独立
决定当前效率主张、协议解释与唯一下一任务。在其裁决前不追加实验。
