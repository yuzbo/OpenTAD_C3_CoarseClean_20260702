# 按真实Heavy MAC限制的前缀计划接口

这是SCI3独立推理计划工具。生产M和原六个训练没有改变，没有新增训练或精度结果。
`allocate_heavy_prefix`接收原RoutePlan、native validity、A/B/C路由以及实际width/layers，
返回可通过`forced_plan`执行的计划和成本凭证。

候选为完整合法原生原子，按原signed predicted_gain稳定降序排序。每次纳入原子时，
使用它所在parent当前容量计算真实增量：
`layers * [12*delta*width^2 + 2*width*(2*k*delta + delta^2)]`。
取在每个窗口Heavy cap内的最长前缀。A/B从0开始；C从全部合法coarse开始，
fine升级增加members−1个token。C不可行的低预算明确拒绝，不把coarse写成0计算。

这是预算上限下的排序控制，不保证signed utility最优；负分不截断，但仍可能为了填充前缀被纳入。
它没有学会不同候选的互补性，也没有新增训练swap、R1或可训练的约束策略。
相同token数可能不同MAC；相同cap可能选择不同数量的token。支持当前逐层固定计划的
默认完整Heavy，不能把depth/BCR/逐层reroute等执行套用到同一层数公式。

`max_fraction`相对当前有效位置的同分辨率全量Heavy，逐窗口约束，
不是全数据均值约束、token比例、总模型FLOPs或延迟上限。返回plan.requested_budget表示
这一MAC cap，budget_id=−1，log_prob=0且learned_sample=False；不得作为PG策略样本训练。
完整状态、物理位置与parent分组由原模型执行，检测mask继续来自原frame mask。

计划使用CPU排序与计数，部署比较必须计入其开销。这里只有可行规划原语，不能把它称作
已经修正了现有动态训练或已经取得精度—延迟收益。新科学运行必须使用新诊断/配置身份，
保留原权重、完整数据和真实trace，不能覆盖原动态结果。首次生产GPU验证仍待实际提交和产物。

检查：`python -m pytest tests/test_sci3_heavy_cap.py -q`，含真实小型A/B/C执行trace、
检测mask、原生768×160尺寸计划、尾部、全空有效域、非零coarse下限和跨parent二次成本。
这些检查不是真实视频性能实验；实际执行和测试结果由外部receipt记录。

2026-09-09验证：代码提交`bffc90becca72da43b90fdd11e17dbbf6ca3c221`在N16独立干净
checkout中通过上述17项CPU检查（49.24秒）。输入为合成数据；包含实际小型A/B/C算子，
没有提交GPU实验、运行完整视频评价或修改正在训练的模型。单独只读源码核验未发现当前
默认A/B/C范围内的成本公式、C下限及计划消费错误；这不替代生产GPU与性能证据。
