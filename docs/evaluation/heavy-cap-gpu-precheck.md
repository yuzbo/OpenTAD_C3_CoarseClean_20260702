# 首个真实窗口的Heavy cap GPU预检

`geosparse_research/heavy_cap_precheck.py`作为文件执行，读取一个已完成60轮且选点完整的
A训练目录，从其bindings固定加载生产M与best EMA；独立研究目录只提供cap工具。
读取原typed resolved config、完整211视频/792窗口test loader的首窗768×160。

先由原source selector在fixed q=1下构造合法有效位置的full参考，再使用相同signed
Scout分数执行MAC cap=0、0.5、1三个强制计划。比较真实encoder.trace的QKV/MLP
计数与各parent二次MAC，不使用TorchDispatch或OperationMacCounter。
cap=1必须与独立source full参考特征逐位一致；所有cap保持原检测输入网格、有效mask
及有限特征，cap=0必须实际Heavy为0。临时budget配置在正常/异常退出恢复。

这是FP32、单窗口、到检测头输入特征为止的正确性检查，没有检测loss、mAP、梯度或
硬件计时结论。它不替代原1280030的AMP观察器失败定位，也不解决原训练的动态控制问题。
只能说明这一cap工具是否被原执行器实际遵守，不能把CPU测试或预检耗时当作性能。
首个生产入口限已完成A；B/C cap原语已有独立CPU测试，未借用A的GPU证据。

```bash
python /ABS/RESEARCH/geosparse_research/heavy_cap_precheck.py \
  --training-run /ABS/COMPLETED_A_RUN --output /ABS/NEW_PRECHECK_OUTPUT
```

必须在原许可Slurm GPU1→CUDA0分配内执行。既有主方法独立评价和FineDiving优先；
准备不等于提交，提交不等于PASS。输出包括measurement.json、实际逐层trace/cap的result.json，
失败写failure.json并保留stderr；不覆盖旧precheck输出、不改生产M或已有训练结果。
