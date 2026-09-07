# 源码地图

以下行号针对根目录模型 commit `902fa05b5c64452ff1c94b82cabce801943d3484`。导航不代表审查已经完成；旧版本在 historical 下另查，不能混用行号。

|审查面|入口与关键位置|需要沿调用链确认|
|---|---|---|
|官方原版|`configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py`及继承S配置；`opentad/models/backbones/vit_adapter.py`；`tools/train.py`、`tools/test.py`|原版模型/训练器不变，解析配置和外围记录仍须核查|
|原版外围执行|`review/launchers/official_job.py:36`、`:50`、`:73`、`:99`|实际数据、配置覆盖、双卡batch、全量预测、best保存时序|
|任务矩阵|`geosparse_ext/matrix.py:37`、`:58`、`:90`、`:233`|F00–F13全部注册、去重、种子、依赖和checkpoint字段|
|配置准入|`geosparse_ext/protocol.py:29`、`:76`；`entry.py:12`、`:34`、`:59`|每个字段被使用、拒绝还是忽略；dataset/kind能力并非仅route能力|
|解析协议|`protocol.py:6`|160px、768/384、TIA原生作用域与cost声明是否符合真实张量|
|A/B/C组合|`geosparse_ext/model.py:14`、`:45`|DENSE、COARSE、DEPTH、BCR也在这里组装；Bfull是否仍经过receiver|
|真实稀疏/粗细|`geosparse_ext/sparse.py:208`|NativeEncoder、选中张量Heavy、C的coarse/fine及TIA、reference/bucket两实现|
|几何|`geosparse_ext/geometry.py:7`、`:29`、`:47`|parent/tubelet/PE/实际支持union，padding和空间分组|
|数据变换|`geosparse_ext/data.py:17`、`:39`|保留官方frame_inds、PTS与原图空间映射|
|Receiver|`geosparse_ext/receivers.py`|五类receiver、null、support/distance、rank与physical对照|
|Scout与路由|`geosparse_ext/routing.py:49`、`:118`、`:130`、`:164`、`:233`|ordered PL、执行排序、配额、warm-up、动态K、actor/critic|
|检测器/在线反事实|`geosparse_ext/detector.py:75`、`:149`|signed acquisition、RNG/buffers恢复、强制plan、确定性推理|
|训练器|`geosparse_ext/runtime.py:45`、`:59`、`:93`、`:120`、`:134`|GPU映射、EMA包括buffers、optimizer覆盖、checkpoint/RNG、AMP和scheduler|
|训练中全量验证|`geosparse_ext/training_validation.py:18`、`:67`、`:128`|全量覆盖、恢复训练状态、5epochs与best、失败处理|
|独立最终评估|`geosparse_ext/evaluation.py:35`、`:47`|与训练中验证是不同路径；internal_dev迁移遗留|
|独立预测导出|`geosparse_ext/prediction_export.py:8`、`:20`、`:55`|60epoch/EMA/provenance要求，split消费者与实际dataset subset|
|选择可视化导出|`geosparse_ext/selection_export.py:73`|是否必须等待60epochs；当前阶段能否观察真实选择|
|选择/成本记录|`geosparse_ext/analysis_capture.py:29`|MAC算子覆盖、单位、unsupported ops下界，原生选择计划|
|硬件|`geosparse_ext/benchmark.py`|独占、batch1/8/32、warm-up50/repeat200、三种计时边界与OOM|
|统计图表|`geosparse_ext/figures.py:339`|MEASURED过滤、window/video单位、seed要求、Pareto配对与误差条|
|状态与来源|`geosparse_ext/records.py:15`、`:36`、`:44`|多进程保存、固定snapshot、全200split及所有消费者|
|资源调度|`geosparse_ext/slurm_queue.py`|不重复作业、单配置GPU能力、恢复、资源等待不等于科研门槛|
|实际部署及focus变换|`review/launchers/deploy_corrected.py:94`、`:104`、`:108`、`:124`|唯一归属、full到focus的字段修改、真实batch、能力和快照绑定|
|证据及选择查看器|`geosparse_ext/evidence.py`、`selection_viewer.py`|实际evidence表示、选择可视化输入、空结果和缺数据|
|风险及GPU认证|`geosparse_ext/metrics.py`、`gpu_precheck.py`、`capabilities.py`|风险阈值与漏检、source-limit证明范围、逐配置能力认证|

重点测试文件：`tests/test_geosparse_native_execution.py`、`test_geosparse_detector.py`、`test_geosparse_runtime.py`、`test_geosparse_training_validation.py`、`test_geosparse_official_protocol.py`、`test_geosparse_risk_metrics.py`、`test_geosparse_figures.py`、`test_geosparse_analysis_capture.py`、并发记录测试。请用实际存在的测试和断言评价覆盖，不按测试文件名认定验证完成。

实际GPU形状/梯度/EMA/数据管道检查凭证在 `evidence/current_status.json`。其中单窗口评估只验证执行路径，不是完整211视频科学指标，也没有覆盖训练60epochs后的全部子任务。
