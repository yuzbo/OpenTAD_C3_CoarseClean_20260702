# 训练损失分量梯度诊断

此离线接口用于定位当前 B 的较大 preclip 梯度，以及任务与路由梯度是否相互抵消。它不修改生产 M，不训练新模型，也不根据 loss 标量大小预判原因。

`geosparse_research.training_gradients.measure_training_gradients` 接受已经加载的真实 GeoSparseDetector 和训练 batch。调用者负责指定来源 checkpoint、epoch、minibatch、实际 microbatch 划分、AMP 和当前 GradScaler scale；真实分析只使用200视频训练池。沿用源 forward_train 的采样、探索、反事实时机及历史正样本 normalizer。不能在 EMA 推理权重上测一次就宣称复现了当时普通权重的训练梯度。

每个 microbatch 只执行一次主 forward，真实 acquisition 如适用会执行源代码规定的额外 forward。task、actor、critic、acquisition、actionness 和合计梯度来自同一个主图；通过逐项 backward(retain_graph=True) 保留生产 reentrant checkpoint 机制。没有为诊断关闭 checkpoint，也没有为各分量分别抽新计划。激活重计算仍会产生额外算子执行，诊断耗时不是正常训练吞吐。

返回未缩放的分量梯度范数、参数模块范数、方向余弦、分量和对实际 total 梯度的重构误差，以及合并梯度在原 max_norm=1 全局裁剪下的系数。不会分别裁剪分量或改变 PL 概率归一化。actor/critic/acquisition 都可能更新共享 Scout CNN；critic 不是只更新最后一个 critic 头。B 的任务梯度还经过 cheap query 的共享 Scout。

函数临时进入训练模式，并在正常或异常退出时恢复 buffers、原梯度对象、随机状态、混合 module mode、requires_grad、minibatch、pending_cost、计划、反事实记录、执行 trace 和 TIA 时间尺寸。没有 optimizer、scheduler、EMA 或 dual 更新。调用者应使用独立加载的诊断模型，勿插入正在进行的异步训练。

多 microbatch 时要求其大小整除 effective batch，与生产训练器一致；按原训练器的 micro/effective 权重累积，并保持各 microbatch 的 loss normalizer 顺序。此顺序不等价于单次大 batch 的正样本归一化，接口不声称两种划分数学等价。FP16 必须在实际 CUDA 环境核验；CPU 检查不标成 AMP 验证。非有限梯度单独记录，不从中生成正常 cosine 或 clip 结论。

复现 focused 检查：

```bash
python -m pytest tests/test_sci3_training_gradients.py -q
```

测试使用真实的小规模 VideoMAE/TIA/ActionFormer 与合成输入，覆盖 A/B/C、空GT、奇数有效尾部、真实 acquisition、动态 pending cost、暖启动、actionness、microbatch、恢复和普通总损失 backward 的一致性。通过这些检查只表示数据生产器正确性；真实 checkpoint/batch 的分量结果与生产 GPU 验证仍待采集，不能据此宣称已定位 B 的梯度主因。

## 2026-09-08 18:57 验证记录

实际代码 `18281bacd7695f49a22c00a6ba6c01905d235ef0` 已在独立、干净的远端 CPU checkout `/data/run01/sczc063/yuzibo/geosparse_official_20260908/sci3_evidence_18281bac/repo` 通过全部10项 focused 检查（pytest 48.63秒，进程返回0）。操作员保存完整日志于 `official_adatad_audit/sci3_verification_18281bac/remote.stdout.txt`。没有提交训练或 GPU 诊断。

独立源码复核提出非整除 microbatch 与生产训练器不一致；已加入相同拒绝规则并验证。其余已审阅范围没有发现具体错误。此前8项检查通过的 `5d616a2d` 留作实现轨迹，实际采用上述最终版本；不重复测试未改动的原124项生产检查。

## 普通检查点到真实训练 batch 的独立入口

新增 `geosparse_research/training_gradient_diagnostic.py`，必须作为文件执行。它先读取训练目录 bindings，再从其中冻结的模型 snapshot 导入 geosparse_ext；诊断代码可来自独立研究分支。两者源码身份分别记录，不加载新修复分支的模型冒充 M。

```bash
python /ABS/RESEARCH/geosparse_research/training_gradient_diagnostic.py \
  --training-run /ABS/TRAINING_RUN \
  --completed-epochs 20 \
  --output /ABS/NEW_DIAGNOSTIC_DIRECTORY
```

入口读取不可变 `epoch_19.pth` 的普通 state_dict，恢复其 loss_normalizer buffer、minibatch、CPU/CUDA/Python/NumPy RNG和GradScaler scale，按原种子与epoch20建立原训练加载器的首个batch（第21轮第一批）。有效batch、microbatch、workers、AMP、数据增强和模型选择规则不修改；只允许全200训练池，不使用测试视频。训练本身不必已经完成60轮。

本诊断不重现任意历史中间batch，而是测量该检查点状态的原下一步输入。只测一个真实batch，若该次探索没有actor或没有触发acquisition，按实际缺项记录，不改counter或重新抽计划来制造完整分量。输出不是全训练总体或失败原因的确定结论。

同一batch先进行分量测量，再用原普通总损失backward复核总梯度范数。状态和RNG在两次之间恢复；不执行optimizer、scheduler、EMA或dual更新。FP16非有限结果单列，不能当正常clip/cosine。测量目录包含measurement.json、gradients.json、result.json或failure.json，记录普通权重身份、GPU UUID、输入视频、有效长度、GT数量及下一步上下文；不复制视频或训练检查点。

新增入口检查：`python -m pytest tests/test_sci3_training_gradient_diagnostic.py -q`。部署状态和精确提交测试凭证记录于外部执行包；没有GPU作业或真实输出前不称已经完成梯度诊断。

2026-09-08 20:59核验：实现 `6519073707db556c9dac1d5495289d374add3fe3` 在独立远端干净CPU checkout通过6项新增检查（pytest 56.05秒），外部凭证 `sci3_verification_65190737/remote.stdout.txt`。原10项核心梯度检查未重复；模型M没有修改。下一实际目标为B动态完成20轮普通检查点的第21轮首批梯度；该单批若没有acquisition/actor分量，保留缺项，不制造分支。
