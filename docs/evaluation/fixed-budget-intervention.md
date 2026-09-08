# 动态A相同权重、固定q=.5的独立诊断

2026-09-08 amendment；ID `sci3-a-dynamic-best-fixed-q050-20260908`。这是追加的具体诊断，不借用旧1545任务或缺失的SCI3 805请求ID。

动态A `tr-0afe8e4fc09b` 已完成60轮及全部12次全测试集验证，最终EMA best为E40。已完成的预算审计发现该E40在792窗口全部选择q=1。因此以其同一权重，在推理时固定q=.5，运行全部211视频/792窗口，测量性能是否保留。原动态训练和独立评价结果均保留；本实验不新增训练，不把该干预称为已训练的动态成本约束方法。

只暂时改变已加载模型的 `budget_mode` 和 `budget`，沿用原Scout位置分数、quota、native分组、TIA、检测器、EMA和推理随机性；退出时恢复配置。模型和官方评价代码从训练记录的干净M快照导入，研究入口有独立版本。q=.5是选择原子比例，不是Heavy MAC=.5。逐窗口在产生预测的同一次前向记录实际QKV trace、选择计划、Conv/Linear/Matmul计数；同时记录同分辨率eligible-full与padded-full分母及分布。带计数器的前向不用于延迟结论。

保留每窗口raw proposals/scores、选择计划、帧支持和成本。使用官方窗口后处理与跨窗口NMS、完整官方mAP和高tIoU指标。短动作及miss-aware边界评价复用已完成独立评价在200个训练视频上冻结的阈值，不在测试集上重新拟合。两种推理的比较绑定同一checkpoint身份；它衡量该权重对预算干预的响应，不证明新Router有效。

执行前在实际GPU做一窗口预检，要求同一输入带/不带观察器的原始预测和选择计划完全一致，固定q确实进入执行路径。预检不能充当792窗口结果。正式目录必须为新目录；失败保留部分产物，没有完整覆盖不写完成。

```bash
python geosparse_research/fixed_budget_diagnostic.py \
  --training-run /ABS/tr-0afe8e4fc09b \
  --reference-evaluation /ABS/ev-301df5cdf563 \
  --output /ABS/sci3-a-dynamic-best-fixed-q050-20260908/precheck \
  --precheck-only

python geosparse_research/fixed_budget_diagnostic.py \
  --training-run /ABS/tr-0afe8e4fc09b \
  --reference-evaluation /ABS/ev-301df5cdf563 \
  --output /ABS/sci3-a-dynamic-best-fixed-q050-20260908/full
```

代码 `7ea094f40064dfd32259beb068548610514d1654` 的6项focused CPU检查在独立远端干净checkout通过（60.83秒）。检查用真实小规模VideoMAE/TIA/ActionFormer和合成输入，覆盖实际Heavy缩小、观察器前后原始输出相等、尾部有效域、两种分辨率成本分母及异常恢复。这不是生产尺寸GPU或全量精度验证。

已提交独立Slurm作业 **1280030**，在g0087等待现有benchmark **1279571** 结束释放资源（`afterany`，不依赖计时成功或性能值）。同一分配先运行一窗口GPU预检，通过后运行全量诊断；仍要求授权物理GPU1→CUDA0。2026-09-08 17:03北京时间观察为PENDING/Dependency，无precheck或full结果。测试日志、提交及观察收据位于外部执行包 `official_adatad_audit/sci3_verification_7ea094f4/` 和 `fixed_budget_diagnostic_20260908/`。

原六项seed0主方法优先，计时实验同卡独占；不重复提交1280030，不覆盖M或原任务产物。该作业尚未产生固定50%推理成绩，动态成本可行分配器仍未实现。

17:07资源优先级记录：固定A仍QUEUED_RESOURCE，因此1280030追加Slurm hold，保留其原ID及afterany依赖。待1279571实际释放分配、固定A已获得真实运行分配或完成后，释放同一ID；不按mAP或路线胜负放行。`resource_hold.json`记录实际操作。不能只靠nice值假设主方法一定先获得GPU1。
