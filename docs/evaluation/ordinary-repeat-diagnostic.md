# B20普通反向自身差异的独立定位

245730已测得：ordinary repeat本身在源AMP/FP32的相对L2差分别为
0.000529515/0.000104144；全部梯度有限。这个观察不支持将差异单独归因于分量测量器。
原容差和旧失败保留，生产M、receiver、normalizer恢复语义和训练均不改变。

新入口ordinary_repeat_diagnostic.py仍读取同一B20普通checkpoint与下一轮首批。
它在每次原forward_train后保存实际完整RoutePlan、PL顺序、各loss、acquisition记录，
再执行普通total backward并保存未缩放参数梯度。每种精度重复三次，每次恢复相同状态/RNG。
计划以实际张量比较，不能用选中数量相同替代；浮点gain/log_prob与执行选择分开报告。

source与strict分两个全新Python进程运行。strict在CUDA初始化前设置
CUBLAS_WORKSPACE_CONFIG=:4096:8，并启用deterministic_algorithms（warn_only=False）、
cudnn.deterministic及关闭benchmark；TF32、精度、模型算子和loss不变。后端设置正常/异常
退出均恢复。源AMP和FP32分别记录，不默认不同精度或不同后端模式使用了相同选择。

strict若遇到不支持确定性执行的算子，保存实际错误栈和已完成的forward产物并停止该进程；
这是诊断结果，不能写成模型训练失败或自动修改生产实现。三个重复仍不一致时继续以实际
计划/损失/梯度差异定位，不放宽attribution门槛。即使严格模式一致，也不能直接量化原训练
mAP影响，或据此宣布loss分量归因通过。

CPU观察会同步设备；每次重复采用同一观察方式，不能将其当无观察器的原训练吞吐或调度。
不运行optimizer、EMA、scheduler或dual更新，不创建新训练。实际CPU/GPU状态在外部执行包。

```bash
python /ABS/RESEARCH/geosparse_research/ordinary_repeat_diagnostic.py \
  --training-run /ABS/EXISTING_B_RUN --completed-epochs 20 \
  --mode source --output /ABS/NEW_OUTPUT/source
python /ABS/RESEARCH/geosparse_research/ordinary_repeat_diagnostic.py \
  --training-run /ABS/EXISTING_B_RUN --completed-epochs 20 \
  --mode strict --output /ABS/NEW_OUTPUT/strict
```

两个命令各保存独立退出状态。source失败不能吞掉完整日志，strict也不得以warn_only替代。
