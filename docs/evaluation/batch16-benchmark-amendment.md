# batch32改为batch16的硬件补测

2026-09-08用户明确要求：reference batch32过大，调小并重测。本次选择batch16，同时测reference与optimized，保留原batch1/8结果和batch32 OOM；不更改模型、训练、checkpoint或重跑已有小batch。

独立入口geosparse_research/batch16_benchmark.py作为文件执行，读取原benchmark运行目录的job/config/provenance，再从原训练bindings中的冻结M导入模型。独立measurement文件只扩展明确的batch32-to16-20260908协议；原50 warm-up、200 samples、100视频轮转、三输入边界、CUDA同步/UUID隔离及逐case记录均不变。模型源码与measurement源码分别记录，不能把研究分支模型当作M。

仅新增原benchmark ID加-b16-20260908的一项补测，batches=[16]。GPU作业先以真实batch16比较同plan的reference/bucket输出（原容差），通过后写precheck.json再执行六项正式case。该预检不是科学结果；OOM/FAILED保留并拒绝整体成功，不自动改batch再试。全量测试集训练验证规则不变；计时仍是原100视频子集的窗口pipeline，不是整段长视频墙钟。

原1279571继续完成其余已启动测量；新补测不与它同GPU计时。资源调度保留主方法优先，等待原benchmark释放GPU并让固定A获得运行资源或完成后放行。新任务是否已提交/运行以外部submission与Slurm为准，不能把本文件当部署凭证。

```bash
python /ABS/MEASUREMENT/geosparse_research/batch16_benchmark.py \
  --original-benchmark-run /ABS/ORIGINAL_BENCHMARK_RUN \
  --output /ABS/NEW_BATCH16_OUTPUT
```
