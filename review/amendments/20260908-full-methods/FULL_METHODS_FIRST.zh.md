# 本轮执行修订：完整方法与固定50%先行

2026-09-08，北京时间。依据用户本轮指令：先验证A/B/C完整创新方法的性能，暂不开发或执行模块消融；完整方法必须包含动态预算；固定50%实验也继续。

## 当前六项主实验

|路线|固定50%任务|完整动态预算任务|服务器|
|---|---|---|---|
|A：规则状态＋稀疏重更新|tr-c43d3e9cad58|tr-32754eef72d1|N16R4，实际分配的物理GPU1/容器GPU0|
|B：Dense Query＋Sparse Evidence|tr-be89cd6cb2ca|tr-8a922c4e6ca7|A100|
|C：Mixed-scale Refinement|tr-a45686a40afa|tr-72ba800e6f08|A100|

六项均为seed0、60epochs，全部200官方训练视频、每次正式验证全部211测试视频/792窗口。GeoSparse每5epochs完整验证并保留EMA best。已进行的固定50%训练不停止、不更名、不从头重复；动态版本从同一官方识别预训练独立初始化，不能把固定版本checkpoint续训后冒充动态方法。

官方原版AdaTAD seed0训练244947继续，保留其40/2/2验证与checkpoint调度。已完成的发布权重复测244949实测71.1387948970%，直接复用；它不是从头训练完成结果。

## “完整方法”的确切含义

共同启用Scout、任务价值硬路由PG、在线signed acquisition校准，以及各自的原生支持几何和实际Heavy执行。动态版本另外启用预算头、内容相关K选择、预算log-prob和成本控制器；固定50%版本保留其余核心组件，只固定预算。

A保持原生规则状态、真实selected-token Heavy更新与原TIA。B保留全量cheap query、实际selected evidence、support-aware receiver和null处理。C执行互斥coarse/fine token及原生回写。B专用receiver不是A/C缺失的组件，C的coarse/fine也不是A/B必须串联的模块。

`B_full`只是全Heavy evidence架构对照，selector和estimator关闭；其中full指全部证据，不代表完整创新方法。本轮先延期，不占主方法资源。ROI、第二轮获取、其他估计器等扩展不在本轮新增实施。

动态版本使用已预注册的目标0.5，仅一个目标，不展开0.25/0.75预算曲线。A/B预算菜单为0/0.25/0.5/0.75/1；C为0.25/0.5/0.75/1，全coarse仍支付Heavy计算。固定“50%”为登记的token预算口径：C的fine比例由(4r-1)/3换算，r=0.5对应约1/3区域fine。动态成本控制目标是相对于dense的Heavy QKV/attention/MLP MAC比例0.5，为软约束，不保证每个视频刚好50%，更不等于完整模型延迟或FLOPs必减半。完整成本包括Scout/TIA/packing/解码等，必须另行实测。

前6epochs为原定随机探索课程，随后学习策略逐渐接管并执行在线校准；这不代表正式训练关闭了Router。部署前增加真实训练microbatch上的动态预算头梯度、gain梯度、acquisition与控制器更新检查；这些是丢弃模型的正确性检查，不是消融训练或科学结果。

## 延期与执行边界

- B-full的待运行244976、额外稠密对照244977在PENDING且无训练progress时取消，记录为用户调整阶段而延期，不是实验失败；原始登记和GPU凭证保留。
- 其他预算档位、种子1/2、模块删减、receiver/ROI/训练估计器消融和相关诊断开发暂缓。
- 主方法的完整评价、必要成本/延迟测量、训练故障修复和正确性检查继续；不得把它们误判为用户禁止的消融。
- 六条主实验之间没有mAP/Oracle门槛。先完成主方法与官方基线的性能和成本判断，再安排消融；负结果如实保留，不自动删除或掩盖。
- 本轮改变执行优先级，不覆盖原始1538项或修订1545项历史注册清单。动态活动子集在experiments.full_methods.dynamic.jsonl；固定版本沿用其原manifest和运行记录。

运行源码保持902fa05，两个阶段使用独立work_root与能力凭证，复用同一已验证软件环境和数据。旧审查快照55d5429保持可追溯；本轮执行变更以本文件、deploy_full_methods.py和full_methods_phase_operations.json为准。
