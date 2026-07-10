# ChronoTransport 完整模型设计规格（第二版）

日期：2026-07-10

## 1. 研究目标

ChronoTransport 面向密集时序动作定位，研究在保持完整物理时间轴的前提下，哪些 VideoMAE 特征必须使用当前输入刷新，哪些可以从缓存传递或短暂复用。

模型不删除时间点，不把不规则采样位置压缩为等距序列。AdaTAD/ActionFormer 始终接收 dense temporal feature，并在原始 768 帧时间轴上完成分类和边界回归。

优化对象不是“重要帧”，而是：

> 对当前时间单元和网络深度，采用某种更新动作会产生多少 detector counterfactual localization regret，以及该动作具有多少真实系统成本。

## 2. 与旧路线的隔离边界

ChronoTransport 主路径明确禁止依赖：

- 动作/背景二分类器；
- 动作概率及其时间差分；
- 显式边界代理或边界导向 selector；
- pre-backbone frame selection；
- offline selection ledger；
- DUCA acquisition、budget controller 或 selected-axis remap；
- validation/test GT、teacher、oracle schedule 或 raw-prediction cache。

旧 C3/DUCA 资产仅作为外部 baseline、no-leak 检查和成本比较工具，不向 ChronoTransport scheduler 提供输入。

ChronoTransport 的部署可见信号只描述“缓存是否仍可信”，而不预测“当前位置是否像动作或边界”。允许信号包括 cache age、低成本视觉变化、codec motion/residual、transport residual、轻量 feature drift、历史动作以及候选 schedule 元数据。

## 3. 可证伪假设

### 3.1 主假设

VideoMAE 的 time × depth 重算价值高度不均匀，并可由部署可见的缓存可靠性信号预测。相比固定周期刷新和简单运动阈值，风险调度能在相同实测成本下更好地保护高 tIoU、短动作和边界定位。

### 3.2 必须预注册的停止条件

满足任一条件即停止扩大方法：

1. 完整系统 p50/p95 延迟节省低于 15%；
2. periodic refresh 在三种子 95% CI 内与学习调度持平；
3. mAP@0.7 或最短 duration quartile 下降超过 1.5 absolute points；
4. scheduler、transport 和 cache movement 的成本超过重算节省的 40%；
5. predicted risk 与真实 detector regret 的相关性接近零；
6. transport error 随 cache age 不可控累积，且安全回退使有效节省消失；
7. 换 backbone 或数据域后，不重新调启发式权重便失去 Pareto 优势。

## 4. 当前 VideoMAE 时间几何

当前 AdaTAD 配置以 768 帧为窗口，将其组织为 48 个 16-frame clip。16 帧是数据和执行容器，不是 ChronoTransport 的固定调度单位。

VideoMAE 使用 `tubelet_size=2`。每个 16-frame clip 包含 8 个 temporal tubelet；整个 768 帧窗口包含 384 个 temporal tubelet。160×160 输入和 16×16 spatial patch 使每个 temporal tubelet 包含 10×10 个空间 token。

因此定义：

- observation 时间分辨率：允许逐帧或逐 tubelet；
- 最小 backbone 刷新单位：一个 2-frame temporal tubelet；
- 默认执行单位：连续 temporal tubelet group；
- 第一版空间粒度：保留一个 tubelet 内的全部空间 token；
- 16-frame clip：兼容现有 AdaTAD 的容器和基线，不是方法约束。

## 5. 动态时间与深度粒度

### 5.1 时间执行组

允许将连续 tubelet 动态组织为：

- 1 tubelet：2 帧；
- 2 tubelets：4 帧；
- 4 tubelets：8 帧；
- 8 tubelets：16 帧。

稳定区域可使用较大的执行组，快速变化或缓存不可靠区域使用较小执行组。第一版 schedule library 使用上述有限集合，避免任意长度造成不可控 kernel fragmentation。

### 5.2 深度执行组

12 个 ViT block 的层分组完全配置化。必须支持并比较：

- 整网一组：`[0:12]`；
- 两组：`[0:6] [6:12]`；
- 三组：`[0:4] [4:8] [8:12]`；
- 六组：每两层一组；
- 逐层：每层一组，仅用于上界和开销分析。

最终粒度由 accuracy–latency–memory Pareto 决定，不预设越细越好。

### 5.3 调度单元

调度单元是：

`contiguous temporal tubelet group × configurable layer group`

而不是单帧、固定 16 帧 clip 或单个空间 patch token。

## 6. 三种状态更新动作

### 6.1 `RECOMPUTE`

使用当前 tubelet group 的真实输入执行目标层组，生成新特征并刷新缓存。它是最可靠但最昂贵的动作。

### 6.2 `TRANSPORT`

使用最近一次可信缓存，通过低成本 temporal transport 和 residual correction 估计当前特征：

`h_new = Transport(h_cache, innovation) + ResidualCorrection`

Transport 第一版只处理时间维度和完整空间 token 集，不进行空间 token crop。

### 6.3 `HOLD`

直接沿用最近缓存状态。只有 predicted upper risk、最大 cache age 和数值完整性同时满足时才允许使用。

### 6.4 强制刷新与安全回退

以下情况必须回退到 `RECOMPUTE`：

- 窗口首个时间单元；
- cache 缺失、shape 不匹配或包含非有限数值；
- 无候选 schedule 满足风险阈值；
- cache age 超过硬上限；
- OOD/innovation guard 触发；
- scheduler 自身异常。

## 7. Transformer 执行语义

VideoMAE 使用 self-attention，不能把某个 tubelet 从上下文中孤立取出并独立通过 block。ChronoTransport 必须明确 partial execution 语义。

第一阶段采用 packed temporal-tubelet execution：保持每个被选 tubelet 的全部空间 token，将连续时间组打包执行并 scatter 回 dense lattice。现有 `PackedTubeletRuntimeRoute` 作为底层可复用入口，但必须扩展为训练可用、schedule-driven 且具备 dense scatter provenance 的实现。

第二阶段评估 selected-query/dense-KV 执行：缓存 token 保留为 attention context，只对需要刷新的 query 计算新输出。只有真实 kernel profiling 证明其收益后才进入主路径。

禁止把“跳过 Python 循环”或“构造 mask”计作真实计算节省；必须证明 attention/MLP 的实际 kernel workload 减少。

## 8. 模型组件

新增 `opentad/models/chronotransport/`：

- `actions.py`：动作枚举、schedule schema、层组与时间组验证；
- `cache.py`：按样本、时间组和层组维护 feature、age、来源时间与 detach 策略；
- `observation.py`：构造与动作语义无关的 deploy-visible cache reliability signal；
- `transport.py`：temporal transport 与 residual correction；
- `risk.py`：schedule-conditioned quantile regret predictor；
- `scheduler.py`：在有限 schedule library 中选择满足风险阈值的最低成本方案；
- `losses.py`：task、transport consistency 与 quantile pinball loss；
- `runtime.py`：协调 dense、packed、recompute、transport、hold 与 scatter；
- `profiler.py`：记录阶段耗时、显存、cache movement 和动作分布。

`VisionTransformerAdapter` 只增加默认关闭的 runtime hook。关闭时必须与原 dense forward 数值兼容；具体 cache 和调度逻辑不进入通用 `Block`。

## 9. Observation 与风险输入

允许的 observation 必须满足 test-time 可见、无语义 teacher、成本可测量：

- cache age 和距上次 `RECOMPUTE` 的时间；
- 当前低成本 stem 与缓存 stem 的 feature distance；
- raw/tubelet difference 的低维统计；
- 可选 codec motion vector 与 residual energy；
- transport residual 与历史 correction magnitude；
- 当前层组、时间组长度和候选动作；
- 最近动作序列与累计 transport/hold 次数。

每类 observation 都必须单独测量成本并做 remove-one ablation。若 observation 本身需要运行接近完整 backbone 的网络，则不允许作为主路径输入。

## 10. Counterfactual localization regret

对时间区域 `g` 和 schedule `p`，以 dense execution 为参考：

`R(g,p) = L_detector(F_schedule(g,p)) - L_detector(F_dense(g))`

结构化风险由以下可解释分量组成：

- detector classification loss difference；
- endpoint regression loss difference；
- high-tIoU degradation；
- short-action/high-IoU tail risk。

训练阶段 dense reference 可用于生成 counterfactual target；validation/test scheduler 不得访问 dense teacher、GT 或 reference prediction。

## 11. 风险预测与调度

Risk predictor 对每个候选 schedule 预测 regret quantile。训练使用 pinball loss；在独立 calibration split 上估计 correction，得到经验 upper-risk。

推理选择：

`argmin measured_cost(schedule)`，约束为 `upper_risk(schedule) <= epsilon`。

“Risk-certified”只表示在明确 calibration assumptions 下的经验风险控制，不宣称 domain shift 下无条件保证。OOD guard 触发时回退到 dense recompute。

## 12. 训练流程

### Stage A：Frozen-feature counterfactual replay

冻结 VideoMAE 与 detector，导出必要的层组 reference feature。在 replay 上模拟 dense、periodic、hold、transport 和有限 time × depth schedules，建立 regret map，验证重算价值是否非均匀且可预测。

该 replay/ledger 仅为训练与分析产物，不参与部署决策。

### Stage B：Transport 与 risk 学习

冻结主 backbone，训练 transport adapter、residual correction 和 risk predictor：

`L = L_task + lambda_transport * L_transport + lambda_risk * L_pinball`

Scheduler 第一版通过有限候选枚举，不使用 RL，不使用不可解释的多代理 loss 堆叠。

### Stage C：Runtime integration

将学习组件接入 VideoMAE packed runtime，验证真实 kernel、cache movement 和 GPU latency。先做 inference/smoke，再开放训练模式。

### Stage D：AdaTAD adapter 联调

保持 VideoMAE 主干冻结，联合微调 AdaTAD adapter、transport correction 和 risk predictor。正式结论至少使用 3 个随机种子并报告 mean、std 和置信区间。

## 13. 基线与消融

主要基线：

- dense 768；
- periodic refresh：2/4/8/16 帧或等价 tubelet周期；
- motion/residual threshold；
- HOLD-only；
- TRANSPORT-only；
- DFF-style key refresh；
- 原始 MoD/packed tubelet compute-skip；
- 当前旧路线 fixed384，仅作外部系统对照；
- oracle counterfactual schedule，仅作诊断上界。

核心消融：

- 无 residual correction；
- 无 cache age；
- 无 feature drift；
- 无 codec signal；
- 无 calibration；
- time-only；
- depth-only；
- joint time × depth；
- 固定 16 帧组 versus 动态 2/4/8/16 帧组；
- 固定层组 versus 多种可配置层组。

## 14. 指标与完整成本

任务指标：THUMOS14 mAP@0.3:0.7、mAP@0.7、start/end error、duration quartile、short-action tail。

系统成本分项：

- storage/I/O；
- decode；
- preprocess；
- H2D；
- observation；
- scheduler；
- recompute；
- transport/correction；
- cache read/write/movement；
- neck/head；
- postprocess；
- amortized offline counterfactual generation。

报告 p50/p95 latency、throughput、peak GPU/CPU memory、GPU utilization、可用时的 energy、cold/warm cache、batch 1 与训练 batch。

ChronoTransport 第一版主要降低 backbone recompute cost，不宣称自动降低 decode 成本。

## 15. 配置、验证与部署

新增最小 ChronoTransport 配置、validator、focused tests 和 `scripts/run_chronotransport_adatad_gpu1.sh`。

启动器要求：

- 默认 `PRECHECK_ONLY=1`；
- 默认物理 GPU1；
- GPU1 保护检查 fail closed；
- 正式运行必须处于 Slurm allocation/step 或已有明确授权的保护分配；
- 输出只写入 `/data/run01/sczc063/yuzibo`；
- checkpoint、数据、日志和 counterfactual cache 不进入仓库。

部署顺序：本地静态测试 → N16R4 validator/precheck → GPU1 Stage-A smoke → kill-gate 审核 → Stage B/C → 第二次 kill-gate → Stage D 三种子。

## 16. TDD 与测试合同

每项生产行为先写失败测试并确认失败原因，再写最小实现。

必须覆盖：

- 2-frame tubelet 几何；
- 动态 1/2/4/8-tubelet group；
- 可配置 layer group 完整覆盖且不重叠；
- 窗口首单元强制刷新；
- cache age、来源时间和样本隔离；
- transport shape、梯度与数值稳定性；
- HOLD 不修改缓存内容；
- 非法 schedule 和异常状态 fail closed；
- dense-disabled 数值兼容；
- dense scatter 后时间轴完整；
- inference 无 GT、teacher、oracle 和旧路线信号；
- 风险阈值选择与无可行候选回退；
- deterministic seed；
- profiler 成本字段完整；
- GPU1、Slurm 与写入边界保护。

## 17. 完成定义

工程实现完成要求：核心组件、VideoMAE runtime hook、配置、训练/评测工具、validator、启动器和 focused tests 全部存在并通过。

部署完成要求：远端 precheck 通过，并在 GPU1 合法分配中启动可观察的 Stage-A smoke 作业，产出 counterfactual regret、动作成本和 profiler 结果。

科学主张成立要求：至少三种子结果通过预注册停止条件，风险调度在完整成本下优于 periodic/motion/MoD 基线，并保护高 tIoU 与短动作。未达到这些条件时，只能报告机制失败或负结果，不能宣称 ChronoTransport 有效。
