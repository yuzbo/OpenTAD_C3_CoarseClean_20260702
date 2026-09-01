# EVALUATOR_DUCA_H65_SINGLECLOCK_JOB1253023_FAILURE_CLASSIFICATION-v001

- verdict: `DETERMINISTIC_EVALUATOR_CONFIG_MISMATCH / RECOVERABLE_WITHOUT_MODEL_CHANGE`
- evidence boundary: evaluation-only; no code edit, checkpoint edit, training or claim
- frozen implementation: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- evaluated job: `1253023`

`final_on`、`final_gate_zero`、`ema_on`、`ema_gate_zero` 四个 family 已完成，并具有可接受的逐窗口身份和终态结构。作业在 H65 OFF checkpoint 严格加载处停止，尚未执行 OFF 推理：checkpoint 含注册键 `module.backbone.model.backbone.blocks.0.relative_physical_time_scale`，而原 OFF config 未构造该参数。

checkpoint 审计证明 final 与 EMA 的该标量均严格为零，且 H65 OFF 身份为 `single_clock_admission=False`。因此该失败属于 evaluator/config 构造身份不匹配，不是 checkpoint 损坏、模型失败或科学反证。允许的最小恢复只能注册既有零参数并保持 admission 关闭；禁止 `strict=False`、checkpoint 改写、权重变换、阈值改变或重新训练。

- next_owner: Coordinator
- next_action: submit one computation-preserving H65 OFF evaluator/config compatibility recovery, then return all terminal artifacts to an independent Evaluator
- dependency: strict loading and unchanged OFF forward computation must both be demonstrated
- single_recovery: one exact operational compatibility recovery

