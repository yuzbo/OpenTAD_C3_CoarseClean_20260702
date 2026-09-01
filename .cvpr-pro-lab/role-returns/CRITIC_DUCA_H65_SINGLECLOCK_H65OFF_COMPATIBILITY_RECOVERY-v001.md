# CRITIC_DUCA_H65_SINGLECLOCK_H65OFF_COMPATIBILITY_RECOVERY-v001

- verdict: `PASS_FOR_COMPUTATION_PRESERVING_OFF_EVALUATION_RECOVERY`
- evidence boundary: read-only code/config/checkpoint review
- frozen implementation: `b2ccfccab5b4912b59954afcc9b0364955327f7c`

允许的兼容绑定为：在原 H65 OFF config 上仅以运行时配置覆盖启用 `model.backbone.backbone.relative_physical_time_residual=True`，同时保持 `model.single_clock_admission=False`。前者仅使模型注册 checkpoint 中已经存在且值为零的标量；后者使 ActionFormer 不向骨干传递物理坐标。VideoMAE 中该标量只有在非空相对物理坐标进入时才参与计算，因此此绑定的 H65 OFF 有效前向与原实现一致。

严格加载必须保持开启；不允许使用 `strict=False`、删除键、编辑 checkpoint、改变模型权重、数据、检测头、损失、NMS 或评估器。训练 optimizer 等字段不参与只读推理，不构成恢复范围。

- next_owner: Coordinator -> Evaluator
- next_action: run final/EMA H65 OFF full validation with authoritative `tools/test.py`, then combine with the four frozen SingleClock/twin families for preregistered statistics
- dependency: immutable OFF checkpoint and admission-false runtime identity
- single_recovery: no additional custom preflight wrapper after this binding

