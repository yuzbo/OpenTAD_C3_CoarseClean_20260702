# ZoomToken BPNS-R1 v003 身份门控完整栈回放终态回执

- 日期：2026-08-28
- 冻结任务：`ZOOMTOKEN-BPNS-R1-IDENTITY-GATED-FULL-STACK-REPLAY-v003`
- candidate：`8a59d655005b9030d8ea5dc17ee2620844cb587b`
- formal Slurm job：`1258526`（`zt-bpns-v003-8a59d655`）
- 节点与时间：`g0063`，`2026-08-28 11:58:39–17:32:11 +08:00`
- Slurm 终态：`FAILED 1:0`，耗时 `05:33:32`
- 协议终态：`FAILED_PROTOCOL_INVALID`

## 终止原因

作业完成了冻结顺序 `K100,R1,R1,K100,R1,K100,K100,R1` 的八个 validation pass，并分别保存了 prediction 与六项未舍入 evaluator vector。随后在 profile 阶段构造短动作评估器时失败：

```text
KeyError: `cfg` or `default_args` must contain the key "type"
tools/bata/profile_zoomtoken_bpns_r1_cost.py:1632 -> profile:1400
-> build_evaluator -> EVALUATORS.build
```

candidate 在 `profile()` 中向 `build_evaluator()` 传入了 prediction、短动作 GT、subset 与 tIoU thresholds，但没有传入注册表要求的 `type="mAP"`，也没有合并既有 `cfg.evaluation`。focused tests 覆盖了边界统计函数，却没有执行这条真实 evaluator 构造路径。因此这是确定性的评估器接口/协议遗漏，不是模型失败、Slurm/OOM 故障或科学负结果。

终态收据直接记录：`status=FAILED_PROTOCOL_INVALID`、`phase=profile`、`error_type=KeyError`、`execution_commit=8a59d655...`、`training_or_resume_executed=false`、`official_test_opened=false`。日志无 OOM、non-finite 或训练/恢复行为。

## 已直接保存的证据

- 八个 pass 的顺序、prediction 文件、prediction SHA-256 与 evaluator vector 完整存在。
- 四个 K100 prediction 的 SHA-256 均为 `008daf5a55af90318506e913c13a4bd2d6ce8ff17a45cc8e856f5619eaa45eb7`。
- 四个 R1 prediction 的 SHA-256 均为 `ffc78393e4097a578def8fdd62ffe4f36dd87c2dddd52de9b3ae248cb108c734`。
- K100 四个 pass 的 evaluator vector 完全一致：Avg-mAP `0.6850689034352214`；mAP@0.3/0.4/0.5/0.6/0.7 为 `0.8361032283895863 / 0.7975732565520736 / 0.71729021957962 / 0.6119111785629083 / 0.46246663409191857`。
- R1 四个 pass 的 evaluator vector 完全一致：Avg-mAP `0.6904214813293945`；mAP@0.3/0.4/0.5/0.6/0.7 为 `0.8436235400542165 / 0.799091460790171 / 0.7335280797130965 / 0.6108696090294431 / 0.4649947170600461`。
- `short_action_validation_gt.json` 与受控失败 `terminal_receipt.json` 存在。

这些 accuracy 向量是终态前已持久化的直接诊断证据。由于完整测量协议没有闭合，它们不能独立升级为新的论文准确率主结果，也不能替代原有 final-EMA 证据。

## 未形成的证据

失败发生在成本采样与短动作/边界汇总之前。结果根没有：

- `profile.json`
- `cost_samples.jsonl`
- `power_trace.jsonl`
- pass-level 或 arm-level p50/p95、throughput、peak memory、gross energy
- short-action 与 boundary diagnostics

因此无法计算冻结的四-pass-median 主估计，也没有真实效率、能耗、显存或边界保护结论。结构上的 36% token 减少仍只是结构代理，不是实测成本。

## 解释与执行边界

该轮只裁定为“八 pass 准确率诊断已保存，但 measurement completeness 硬门未通过的协议无效终态”。不修改阈值、不补造缺失产物、不恢复或重跑 job `1258526`，也不由 Codex 自行创建 successor 或选择科学路线。下一动作仅是把完整终态、根因、已有与缺失证据交给 exact ZoomToken Project 的一次全新 post-result Pro 复盘，由 Pro 独立裁决可发表性、是否允许最小协议修正，以及唯一下一任务和北京时间期限。

## 原始位置

- source：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_cost_src_8a59d655_v003`
- result root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_r1_identity_v003_8a59d655_seed42_20260828`
- log root：`/data/run01/sczc063/yuzibo/projects/zoomtoken_bpns_v003_logs_8a59d655_20260828`
- terminal receipt：`.../terminal_receipt.json`（原始失败收据保留于远端结果根）
