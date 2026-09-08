# 固定预算诊断的统计路径隔离

2026-09-09新增诊断，针对Slurm1280030的GPU严格一致性失败。原失败完整保留，
不修改b70ae056模型、EMA或旧测量输出，不降低误差阈值，不启动全量推理。

对同一实际首窗口，分别运行普通前向、只加SelectionObserver、只加OperationMacCounter、
同时加两者；各做两次，最后再普通前向一次。每次恢复RNG/buffers/执行状态，
首先记录选择计划差异，再记录proposal/score的dtype、形状、元素差异和最大绝对/相对差。
生产AMP与FP32分开，二者都使用原模型权重。所有原始输出和实际Heavy trace保留。

这能区分普通重复本身不稳定、被动记录器影响、TorchDispatch计数路径影响及精度相关性。
如果多个因素一起出现，保留完整结果后继续定位，不将某一个假说直接写成结论。
完成差分诊断不等于原一致性预检PASS，也不自动放行792窗口科学评价。

```bash
python geosparse_research/observer_parity_diagnostic.py \
  --training-run /ABS/tr-0afe8e4fc09b \
  --failed-precheck /ABS/sci3-a-dynamic-best-fixed-q050-20260908/precheck \
  --output /ABS/observer-parity-new-output
```

入口先从训练绑定的冻结快照导入模型，独立研究目录只提供测量函数。
新输出目录必需；输入元数据与旧失败首窗口核对。结果不含mAP/延迟，不进行optimizer更新。
CPU针对性检查为tests/test_sci3_observer_parity.py；GPU仍需合法Slurm资源，不能借登录节点执行。
