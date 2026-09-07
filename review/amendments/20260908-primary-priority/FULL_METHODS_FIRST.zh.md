# 本轮执行修订：主方法优先，部署后推进补充实验

2026-09-08，北京时间。最新用户指令：A/B/C完整创新方法仍是首要性能验证目标；在其部署或开始训练后，推进其他实验部署；4090/N16R4与A100同时推进。完整方法包含动态预算，固定50%继续，当前仍仅seed0。本修订取代早先“得到主方法性能判断之后才安排消融”的阶段门槛；不等待mAP结果。

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

`B_full`只是全Heavy evidence架构对照，selector和estimator关闭；其中full指全部证据，不代表完整创新方法。它进入较低优先级的补充队列。ROI、第二轮获取、其他估计器等扩展不在第一批补充部署范围。

动态版本使用已预注册的目标0.5，仅一个目标，不展开0.25/0.75预算曲线。A/B预算菜单为0/0.25/0.5/0.75/1；C为0.25/0.5/0.75/1，全coarse仍支付Heavy计算。固定“50%”为登记的token预算口径：C的fine比例由(4r-1)/3换算，r=0.5对应约1/3区域fine。动态成本控制目标是相对于dense的Heavy QKV/attention/MLP MAC比例0.5，为软约束，不保证每个视频刚好50%，更不等于完整模型延迟或FLOPs必减半。完整成本包括Scout/TIA/packing/解码等，必须另行实测。

前6epochs为原定随机探索课程，随后学习策略逐渐接管并执行在线校准；这不代表正式训练关闭了Router。部署前增加真实训练microbatch上的动态预算头梯度、gain梯度、acquisition与控制器更新检查；这些是丢弃模型的正确性检查，不是消融训练或科学结果。

## 主方法优先与第一批补充队列

- 原B-full待运行244976、额外稠密对照244977在PENDING且无训练progress时取消，历史记录仍是阶段延期，不是实验失败。此次在独立secondary work_root恢复，复用902fa05对应的原PASS凭证，不重复预检或已执行训练。
- 第一批补充：N16R4的A uniform（tr-f90ea0ae2ce4）；A100的B-full（tr-c2d93c4c3979）、同结构稠密对照（tr-bacdbb058eca）、B uniform（tr-6f26bfd4ff96）、C uniform（tr-067daa13036b）。共5项seed0训练及各自evaluate/benchmark，共15条登记。uniform必须各自通过真实预检。
- 两端补充协调器已经部署；它们使用Slurm nice=10000及after（主作业开始）依赖，优先让本服务器主方法取得分配，不使用afterok、性能阈值或等待主训练结束。A100固定B/C原ID244973/244975分别等待动态244989/244991先开始，正在训练的A固定50%不动。
- 其余预算/模块研究后续按实现和资源就绪推进；种子1/2继续延期，不自动恢复多种子全矩阵。新Oral文件的旧340a3541、224px形状和开发留出划分不适用于当前运行。
- 主方法的完整评价、必要成本/延迟测量、训练故障修复和正确性检查继续；不得把它们误判为用户禁止的消融。
- 六条主实验与后续补充实验均无mAP/Oracle门槛。主方法是首要性能验证对象，部署完成后即可准备补充实验；负结果如实保留，不自动删除或掩盖。
- 本轮改变执行优先级，不覆盖原始1538项或修订1545项历史注册清单。动态活动子集在experiments.full_methods.dynamic.jsonl；固定版本沿用其原manifest和运行记录。

运行源码保持902fa05，两个阶段使用独立work_root与能力凭证，复用同一已验证软件环境和数据。旧审查快照55d5429保持可追溯；本轮执行变更以本文件、deploy_full_methods.py和full_methods_phase_operations.json为准。

新增部署入口为deploy_secondary.py/secondary_queue.py；primary_start_priority.json记录保留原作业ID的开始顺序调整，secondary_deployment.json记录两端补充部署。不能再运行旧deploy_full_methods.py的阶段收缩操作来撤销本轮补充队列。

02:00前后核查：N16共享盘仅余约8 GiB、低于新训练20 GiB要求，A动态因此BLOCKED_STORAGE；A固定50%继续。A100可运行节点GPU全部占用，主作业已提交但排队，调度预计开始时间不作承诺。保留检查点并复制到A100存储；不删除数据、不改epoch/分辨率/有效batch来绕过资源问题。实时状态以corrected_status.json为准。
