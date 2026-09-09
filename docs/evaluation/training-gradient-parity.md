# B20梯度等价失败的独立定位

旧245705使用fdfdc6a0测量冻结M的B动态第20轮普通权重、下一轮首批；total与ordinary范数
相对差0.000162777超过rtol1e-4。旧入口在断言前没有保存梯度，不能据此归因actor或critic。
保留该失败，不改容差、不修M、不重复训练。

新入口`geosparse_research/training_gradient_parity.py`复用原输入准备过程；模型仍从训练
bindings指定的M导入。恢复的是M的普通权重/RNG/AMP/normalizer语义，包括已知normalizer截断。
它不是连续训练未中断的精确回放，更不是EMA或全训练分布分析。

每种精度从相同保存状态依次测ordinary、ordinary repeat、分量测量的total、ordinary after。
保存全部未缩放CPU参数梯度以及逐参数L2/max-abs/连通性比较；失败等价也保留原始向量和
`decomposition.unverified.json`。相同标量范数不等于向量相同。原rtol1e-4/atol1e-5保留，
新报告额外要求逐参数比较，`attribution_equivalence_passed=false`不能用作分量归因结论。

源AMP模式和FP32模式分别测量。二者各自恢复原状态/RNG，但精度可能改变计划，所以不把
跨精度差异自动解释为单独AMP误差，也不声称两种精度强制使用相同计划。
正常及异常退出均恢复状态；不执行optimizer、EMA、dual或scheduler更新，不改原global clip。

```bash
python /ABS/RESEARCH/geosparse_research/training_gradient_parity.py \
  --training-run /ABS/EXISTING_B_RUN --completed-epochs 20 --output /ABS/NEW_PARITY_OUTPUT
python -m pytest tests/test_sci3_training_gradient_parity.py tests/test_sci3_training_gradient_diagnostic.py -q
```

该入口是独立测量修订，旧诊断准入未放宽。实际CPU验证、GPU排队和现场证据记录于外部
执行包；没有真实GPU产物前不称已定位B20差异。部署必须使用新stage，不覆盖245705的旧输出。
