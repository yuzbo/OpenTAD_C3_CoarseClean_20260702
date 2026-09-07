# 全部实验的执行、统计与验收协议

## 1. 真正的并行含义

全部F00–F13训练配置同时登记，每个都从同一识别预训练初始化，不等待其他实验的精度结果。初始化数据、实现和本路线正确性测试属于不可消除的技术依赖；evaluation只能读取自己的checkpoint。准确率、Oracle headroom和speedup不准出现在ready条件中。

并行的是实现、训练队列和独立测量工作；不是让多个作业争抢同一张GPU，也不是让延迟测试与训练同时污染同一GPU。

资源有限时排队仍然必要。调度器按族/路线/任务类型轮转，周期性为隔离benchmark排空本机自有任务。禁止以“主路线更重要”为由饿死B/C。多个节点使用平台真实调度器的array jobs和共享artifact状态，不能把本地脚本伪装成远程集群调度器。

## 2. 任务类型

- train：每个manifest任务60 epochs、seed确定，从识别预训练开始；不引用其他train任务。
- evaluate：依赖自身train的epoch59；所有标准与风险指标统一计算。
- benchmark：seed0对应配置；每任务内batch1/8/32，50warmup+200计时重复，100个按训练长度分布分层固定的真实视频；不同seed的accuracy都评估，但硬件主测试不重复三遍训练seed。每个时间重复保存原始sample，报告GPU/CPU/编码视频路径。
- diagnostic：只依赖自己的train；某些读取同run的epoch5/20/40/59。

同一配置出现于多个实验族时复用同一训练任务，避免重复训练。模型参数或评估协议不同则不复用。

## 3. F00–F13的具体解释

F00：official dense、A-full、B-full、C-all-fine、cheap-only，两主要数据集三seed。
F01：A/B的T-only、ST及固定.25/.5/.75；动态预算三个目标；C-fixed三档和dynamic .5/.75。
F02：uniform、random、motion、actionness、uncertainty、CDF-native、PG-only；whole-clip；legacy DUCA。如没有原DUCA代码，legacy项BLOCKED，CDF-native照常实现，但不得称完整复现DUCA。
F03：固定4/6/8/10层；teacher-free PBD-inspired；BCR前缀0/2/4。PBD-inspired必须显式标modified，不冒充需要不同训练流程的原论文PBD。
F04：时间原子1/2/4/8 tubelets，空间组1×1/2×2/7×7；总token相同不等于实际cost相同。
F05：rank插值、物理插值、concat-scatter、timestamp-only和support-aware；A错误rank-TIA负对照；C中心/支持/尺度信息。
F06：每clip配额、global但不许空、global允许0、budget controller/threshold、硬件LUTcost、global-zero禁止。
F07：scout分辨率80/112/160、宽64/128/256、时间stride1/2/4、3.125% detail probe、actionness辅助头；一次只改变一个轴。
F08：PG、PG+真实新增反事实、保留梯度、zero-gate、新增估计器、ST；探索比例及probe频率8/32/128；warmup0。
F09：T×S完整2×2因子；原图ROI112/160/224和1/2 ROI；fullframe fallback、平滑轨迹、低分辨率放大负对照。
F10：B slots1/4/8、receiver层1/2/4、无null/L2对齐/直接覆盖；C coarse聚合和无尺度、无TIA。
F11：同总预算的一次/两次获取，random/single-boundary/pair-boundary；parent上下文、A每层重新route。内部两轮必须计算全部成本，不声称缓存始终合法。
F12：FineAction与VideoMAE-L；资源不存在时透明BLOCKED，不影响其余实验。
F13：ActionFormer/TriDet、query384/768，并包含稠密检测头控制。

## 4. 必须执行的六个诊断suite

### D01：价值与有限菜单
在internal diagnostic split上按长度、背景比例、训练duration分组固定128个窗口。每例固定其余计算，仅选8个合法原子组成菜单。固定K=4时枚举C(8,4)=70个集合；动态菜单仅在该有限集合内枚举可行budget。

报告名字：**finite-menu loss-best**，只说明当前模型、当前窗口、当前loss、当前候选集的最优。视频级loss最优不能称数据集mAP最优。

A/C必须重跑改变选择会影响的上下文路径；B缓存需通过D06。标签不用作eval-window训练数据。

Utility评估：acquisition、retention、swap分开；Spearman/Kendall、符号准确率、top-budget regret、simple-policy regret、epoch分层；使用相同classification/localization normalizer。AP仅作为后验关联，不能当可分解逐位置真值。

### D02：几何与gap
固定证据manifest，receiver不同模型读取相同native支持。执行：存储顺序排列、坐标正确/错误、native pairing正确/错误、gap/block deletion、support union vs hull负对照、实际PTS与index假设。

gap强度按maxgap/GT duration及gap到边界距离报告；固定K的均匀与聚集选择都纳入。模拟retiming必须同步重采样内容、PTS和GT；只改metadata的测试标为metadata-corruption，不称VFR鲁棒性。

### D03：动态预算
在internal dev校准mean实际cost；在诊断集比较：learned budget、同histogram跨视频打乱、固定budget菜单、budget按长度分层打乱。
打乱只替换K，位置仍使用同一模型score重新选。使用输入产生的budget可两遍收集，但这额外流程不能冒充部署latency。
报告mean/p95 K、cost、性能；不在held-out labels上校准lambda或cost匹配。

### D04：时间/空间交互
同一context与视频比较四状态(无追加、T、S、TS)，I_TS=u(TS)-u(T)-u(S)。同时报告平滑task loss、分类、start/end误差与AP阈值变化；控制成本。
两端acquisition比较single-start/single-end/pair，排除两端落在同packet造成的重复计费。pair interaction为负或不稳定也保留。

### D05：捷径、伪边界与证据内容
coarse-only；coarse+selection mask；coarse+mask+真实evidence；同support替换evidence为零/均值/另一视频内容；错误时间坐标（诊断负对照）。替换内容不能改cost口径。
selection mask来自输入时不是自动标签泄漏；该实验只用于拆出heavy-content的净贡献。
统计预测边界与compute-switch距离，分真实GT附近/远离GT；控制router天然靠近动作造成的相关性。对matched边界误差之外报漏检和false positive。

### D06：缓存合法性
直接全网络新forward vs缓存构造，固定随机状态；检查中间states和最终predictions。A/C若依赖已变则缓存无效；明确记录哪些prefix/packet可复用，失效依赖从首个受影响层重算。禁止把invalid-cache快实现放入Pareto主表。

## 5. 统计、数据与选择政策

全部主训练配置三seed，报告mean±sd；按视频配对bootstrap=1000次，不把同视频多个窗口当独立样本。对训练seed和视频不确定性分别说明。全部metric在同一个official evaluation code上计算。

协议默认从官方train留10%内部开发，训练90%，因此不能直接把论文作者的全train分数当同协议复现。训练数据比例必须列在表头；全train复现作为一致的补充协议需要所有主方法同样重训并形成amendment，不悄悄只训练自己方法。

短动作定义：train-duration bottom quartile固定阈值，同时报绝对秒bins。视频长度、overlap、长背景按训练分位数冻结。低运动只能叫代理slice；没有人工物体标签时小物体slice为NA。VFR slice必须真实PTS证据；人工stress test单独标注。

GT不输入deploy policy。oracle诊断有GT访问但不算部署方法。禁用基于test成绩决定是否完成其他实验的晋级逻辑。

## 6. 性能计时与真实计算

three lanes：device-model（GPU tensor已就绪）；decoded tensor→detections；encoded video→detections。只有第三类能称包含decode的端到端；没有该路径报告NA。

必须CUDA同步、真实predictions postprocess、相同数据顺序和warm-cache协议、记录冷启动及compile另表。均值、p50、p95及95%CI，峰值allocated和reserved VRAM，batch throughput按有效视频/秒。OOM不改变输入分辨率偷偷重测；返回OOM状态及原因。

reference与optimized必须同算子数学含义。记录每层每parent：候选token、有效selected、padding、执行QKV/MLP token、attention形状；记录dense TIA、PE、Scout、Router、crop、packing、head成本。MAC和FLOP用统一口径（1MAC=2FLOPs或另一选择明确全表一致）。

原图ROI含decode/crop成本；训练probe、encoder replay、反事实搜索和policy calibration单独计GPU-hours与总forward次数。所有方法统一精度、设备、batch和kernel路径，不给baseline故意更差实现。

## 7. 状态与交付

PLANNED / BLOCKED_EXTERNAL_ASSET / BLOCKED_CAPABILITY / BLOCKED_ARTIFACT / QUEUED_RESOURCE / RUNNING / DONE / FAILED。
DONE只表示真实输出和receipt通过格式检查，不代表科学主张成立。缺资源任务不删除、负结果不隐藏。结果汇总页面总显示分母=完整注册矩阵。

每run：job.json、resolved_config.json、source_commits.json、split_hash、seed、train.log、metrics.json、cost_trace.jsonl、raw_predictions、checkpoint（train）、result.json。科学记录另有status per measurement（OK/NA/OOM/INVALID），不以零填充NA。
