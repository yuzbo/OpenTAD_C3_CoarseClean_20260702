# 主方法阶段状态快照

查询时间UTC：`2026-09-07T17:39:52.360066+00:00`。此表不是实时刷新页面。

|服务器|训练ID|Slurm ID|状态|GPU预检|动态策略检查|
|---|---|---|---|---|---|
|N16R4|tr-c43d3e9cad58|1277994|RUNNING|PASS|固定预算，不适用|
|N16R4|tr-32754eef72d1|尚未分配|QUEUED_RESOURCE|PASS|PASS|
|A100|tr-be89cd6cb2ca|244973|PENDING|PASS|固定预算，不适用|
|A100|tr-a45686a40afa|244975|PENDING|PASS|固定预算，不适用|
|A100|tr-8a922c4e6ca7|244989|PENDING|PASS|PASS|
|A100|tr-72ba800e6f08|244991|PENDING|PASS|PASS|

三条动态策略检查均使用真实 `[2,1,3,768,160,160]` 输入，预算头与gain头非零梯度、acquisition、预算头更新及成本控制器均通过。完整receipt见corrected_status.json。

固定50%的A已有5epochs全量验证，早期Avg-mAP为4.3061%，当前继续训练；这是学习路由接管前的早期结果，不能冒充最终性能，也没有因此停止其他路线。

B-full244976与额外稠密对照244977在PENDING且0epochs时延期；官方244947不变。动态新增检查初次因外部脚本import路径失败，修复后1277998/244986/244987通过；失败预检没有正式训练结果。
