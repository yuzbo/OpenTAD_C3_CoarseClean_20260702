# DUCA-UVT value-portal development seed

## Status

- Date: `2026-08-19`
- Stage: `experiment_running`
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

## Observed so far

- 三任务 RUNNING，Epoch 0 已打印迭代 50/100/150 loss 行，总 loss 与各项 loss 有限。
- `geo` / `geo_ema` 的 `selector_value_geometry_loss` 约 0.15 且有限；
  `geo_ema` 首轮 EMA distill loss 为 0（EMA 由 online head 初始化，预期行为）。
