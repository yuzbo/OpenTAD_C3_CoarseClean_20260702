# 执行证据范围

JSON来自真实本地保存的服务器查询和实际run receipts，数值未改写；绝对机器路径替换为 `<MACHINE_PATH>/文件名`，便于公开审查。它们是时间点快照，不是实时状态。`live_protocols/`保留采集时的包装字段和早期预检记录，当前GPU能力以current_status为准。

`official_released_checkpoint/launcher_executed.py`是已经完成的公开权重复测实际保存的外围脚本；`../launchers/official_job.py`是当前原版训练所用的新外围脚本，保留官方验证周期。请比较两者而非假定完全相同。早期eval配置中的训练验证频率不会影响该次固定权重test入口，但不能用它宣称当前官方训练配置。

不包含raw训练日志、视频、weights或完整predictions。正式mAP可从本地留存的211视频预测重新计算；缺少逐实例原始资料时应限制结论。
