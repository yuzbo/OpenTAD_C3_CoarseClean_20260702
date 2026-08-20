# DUCA-UVT value-portal development seed

## Status

- Date: `2026-08-19`
- Stage: `tested`（诊断性发展种子完成；结论为负，不是 empirically_supported）
- Branch: `codex/duca-uvt-utility-value-20260819`
- Deployed commit: `df544c78ce515d925dc7019f106fce09a53c09f8`
- Job: `1244840` (array 0-2: `off` / `geo` / `geo_ema`, seed 3407)
- Run root: `/data/run01/sczc063/yuzibo/duca_uvt_official_df544c78_wv2_20260819T171000Z`
- Prior failed attempt: `1244775` failed at optimizer-group construction before training; fixed by
  `06b0d514` (classify `value_head.queries` and `cross_attention.in_proj_weight`).
- Current matrix `1244133`: untouched.

## Question

单头可学习 signed V(t) 残差、GT geometry utility、value-head self-EMA，以及
boundary-foveated exact-K decoder 是否能在诊断性发展种子中分别证明增量有效性？
Portal detector feedback 保持关闭，直到四层有限差分 gate 通过。

## Scope and constraints

- 三臂共享同一 runtime：`V_off` 为 legacy 等价控制臂；`V_geo` 开启 GT geometry；
  `geo_ema` 在 V_geo 基础上开启 self-EMA 稳定性正则。
- 本种子只产诊断结果，不产 mAP/efficiency/paper claim。
- 推理不得出现 GT/teacher/cache/EMA；hard frame indices 始终 detach。
- 完整训练结束、五轮恢复生命周期通过前，不得跨级声称完成。

## Observed final results

- 三任务完整训练 60 epoch 并完成 test，Job `1244840` exit 0。
- 最终 test mAP（checkpoint epoch_59）：

| arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---|---|---|---|---|---|
| off | 57.35 | 74.00 | 68.76 | 60.84 | 49.31 | 33.84 |
| geo | 55.93 | 74.63 | 68.67 | 59.84 | 46.50 | 30.02 |
| geo_ema | 55.92 | 74.15 | 68.41 | 59.57 | 46.99 | 30.49 |

- 结论：在当前设置下，开启 GT geometry V(t) 与 self-EMA 的 `geo`/`geo_ema`
  均未超过 legacy `off`；差值约为 Avg-mAP −1.4、@0.6 −2.3~−2.8、@0.7 −3.3~−3.8。
- 注意混杂因素：`geo`/`geo_ema` 同时改变了 selection score（α·V）和
  dynamic-K evidence（mean sigmoid(V)），不能把损失单独归因于哪一个；
  boundary-foveated decoder 与 portal detector feedback 本种子均关闭。
- 首轮 EMA distill loss 为 0（EMA 与 online 同初始化）；后续 EMA distill
  loss 已进入非零路径，但终端 mAP 仍为负。
