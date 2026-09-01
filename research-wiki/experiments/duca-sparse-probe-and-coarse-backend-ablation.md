---
title: DUCA 稀疏粗扫、插值与粗分类器架构消融
status: experiment_running
updated: 2026-07-23
---

# 核心问题

在保持重型选帧预算 K、边界优先 selector、exact-K/max-hole decoder、官方派生 TAD backend 和训练协议完全不变时，粗分类证据能稀疏到什么程度，仍能保护完整 validation terminal-EMA mAP 与高 tIoU 边界定位？卷积、注意力和 SSM 时序编码器中，哪一种形成最佳性能-成本 Pareto？

# 网格合同

当前 dense candidate stride 是 4 个源帧。粗扫因子固定比较 `d=1/2/3/4`，对应源帧间隔 `4/8/12/16`；其中 16 是超过现有 15 帧请求上限的压力测试。selector 始终在 768 点物理时间网格上决策，不能把粗扫 anchor 数直接当成 K，否则 `384 anchors + K384` 会退化为全选。

重建主设置是在粗分类 hidden feature 上做线性插值到 768 点，插值后的时序隐藏特征直接作为 selector 的完整粗粒度证据；不额外输入 observed-anchor mask，也不输入到最近 anchor 的距离。这样检验的是稀疏粗分类和连续特征重建本身能否支撑间接边界定位，而不是让 selector 依赖“哪些点真实计算过”的旁路标记。禁止只插值最终 `p_action`，因为单一概率曲线会丢失状态转变、语义变化和不确定性所需的多维信息。对照为 nearest、hidden-linear，以及最佳稀疏间隔上的 transition/uncertainty peak 局部补扫。

# 粗分类器架构

只使用仓库已有官方实现适配，不新增 lite 仿制：

- 卷积/TCN：`official_ms_tcn2`
- 注意力+TCN：`official_asformer`，当前主线
- SSM 混合：`official_video_mamba_asformer`
- 强但可能较贵的参考：`official_fact`

所有架构共享相同低分辨率空间输入、二分类 actionness/transition/boundary supervision、输出时间网格和下游 selector。先报告原生官方设置，再补参数量或 MACs 近似匹配设置；不能只比较粗分类 F1 而忽略总成本和最终 TAD mAP。

# 两级实验

1. P0 筛选：四个粗扫间隔 x hidden-linear，统计 actionness AUPRC/F1/ECE、transition peak F1、端点召回、短动作召回、dense-vs-sparse hidden/logit 误差、probe latency/MACs。MS-TCN2、ASFormer、Video-Mamba-ASFormer、FACT 四个官方架构必须全部进入同协议粗分类实验。
2. 完整 TAD：每个粗扫间隔至少对当前 ASFormer 跑同 commit、同 seed、同 K/G 的 official-60 terminal mAP；四种架构在统一“时序隐藏特征”输出合同后都必须进入完整 TAD 对照。不得让 ASFormer 输出 temporal encoder hidden、其他架构却输出 spatial stem hidden 后直接比较。为控制算力，可先跑共同 seed/K384，再对 Pareto 前沿补三种子和预算曲线，但四架构的共同 seed 结果不能被粗分类 F1 替代。

主判据是完整 THUMOS validation、OpenTAD tIoU 0.3--0.7、terminal epoch-59 EMA mAP，重点看 mAP@0.6/0.7 和短动作。选择最大可接受间隔时必须预先固定非劣界，并同时报告 decode/resize/H2D/probe/selector/VideoMAE/detector/total latency；平滑度、边界召回和选帧图只能解释 mAP，不能替代 mAP。

# 边界分布轴

当前 R2Q3 表示端点中心半径 2 个 candidate step、每簇配额 3；按 stride 4 换算为中心前后约 8 个源帧。R4Q5 是半径 4、配额 5，即前后约 16 个源帧。K384 时最多 25% 预算进入 mandatory bilateral burst，其余预算维持全局上下文；G2 使相邻选中点最大间隔为 12 个源帧。

固定 K/G 后比较 exact-uniform、action-interior-heavy、single-transition-peak、R2Q3、R4Q5，以及边界预算比例 `0/0.125/0.25/0.5`。每个分布都必须重新完成同协议 TAD 训练；GT Oracle 只做诊断。最终根据 mAP、短/中/长动作和高 tIoU 结果判断“更集中、更多上下文或更均匀”哪种分布真正有利。

# 当前落地状态

- 生产分支新增提交 `codex/duca-boundary-burst-20260722@4f81299`，只加入四后端粗分类实验入口，不改变正在运行的 `9f97f2c` R 系列模型。
- 干净远端快照：`/data/run01/sczc063/yuzibo/projects/opentad_duca_coarse_4f81299_20260723`。
- 运行根：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_coarse_backends_4f81299_20260723_0015`。
- Linux focused：`11 passed in 8.23s`；四个官方后端均通过真实单卡 CUDA 小前向。
- 独立无依赖 Jobs：MS-TCN2 `1180502`、ASFormer `1180503`、FACT `1180504`、Video-Mamba-ASFormer `1180505`；均已进入 `RUNNING`。
- 当前运行协议为 64x64、768 点窗口、seed 3407、20 epochs、只在终轮评估、无 early stopping。其 actionness/边界结果只是 P0 诊断，不能写入论文 TAD mAP 主表。
- 四后端 Jobs `1180502--1180505` 均已 `COMPLETED/0:0`。终轮 P0 原始诊断见下表；它们没有 detector 训练和 TAD mAP。
- ASFormer 稀疏间隔 4/8/12/16、temporal-hidden 线性重建与完整 TAD 对照已在
  `codex/duca-sparse-probe-interpolation-20260723@dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`
  实现并推送。GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`。
- 实现只在规则 anchor 上运行空间 stem 与官方 ASFormer，再按原始有效时间坐标把 action logits、encoder hidden 和 policy hidden 线性重建到完整网格；selector 不接收 anchor mask 或距离。四臂保持同一 R2Q3、K384/G2、VideoMAE、official-derived AdaTAD/ActionFormer 和训练协议。
- 新 focused test 为 `4 passed`；真实 CUDA Gate Job `1180556` 为 `COMPLETED/0:0`，确认 d=1/2/3/4 均输出完整时间长度、数值有限、空间和时序模块梯度非零，估算 MACs 随间隔单调下降。
- 四卡正式套件 Job `1180557` 已在 `g0053` 上 `RUNNING`，四个独立 step 分别绑定一张 GPU，并已同时进入 P0 epoch 0。每臂执行 P0 20 epochs、真实 full-model gate、official-60 60 epochs，并只以 terminal epoch-59 EMA 完整 validation mAP 裁决。
- 运行根：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329`；部署回执已记录 Gate/Suite sbatch 与 jobs.tsv 哈希。当前没有 terminal mAP，状态只能是 `experiment_running`。

外部官方源码固定为：MS-TCN2 `f423a9e65f4ccb1cd7322eb9f94946a19e787993`、ASFormer `e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`、FACT `7bd81bda2b84618a3e23876a2637a82f24881122`、Video-Mamba suite `ec9108b72d5db59f6d634c94cd0e008228a7b918`。

## 2026-07-23 00:25 进度

四臂均运行约 9 分钟并进入 epoch 4，日志错误扫描为空。当前累计训练 loss 约为 ASFormer `0.61`、FACT `0.61`、MS-TCN2 `0.62`、Video-Mamba-ASFormer `0.61`；这些仅证明训练健康，不能用于架构排序。尚无终轮粗分类指标，更无完整 TAD mAP。

## 2026-07-23 四粗分类器终轮 P0 结果

| 后端 | Action AP | ROC-AUC | 最佳 F1 | Balanced Acc | `delta_p_action` 边界支持@1 | 终轮验证秒数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MS-TCN2 | 0.4078 | 0.5992 | 0.4999 | 0.5000 | 0.7225 | 471.7 |
| ASFormer | 0.4087 | 0.6237 | 0.5158 | 0.5326 | 0.8184 | 465.0 |
| FACT | 0.3945 | 0.6030 | 0.5064 | 0.5099 | 0.7956 | 456.2 |
| Video-Mamba-ASFormer | 0.4161 | 0.6179 | 0.5104 | 0.5484 | 0.8302 | 471.1 |

信息增益是：Video-Mamba-ASFormer 在 Action AP、balanced accuracy 与当前间接边界支持上领先，ASFormer 的 ROC-AUC/F1 更高，四者不存在单一粗分类指标上的绝对赢家。所有后端的最佳边界策略均为 `delta_p_action`，继续支持“状态转变优先”而非 actionness top-k；但只有统一 temporal-hidden 后的完整 TAD mAP 才能决定主后端。

## 2026-07-23 稀疏粗扫门禁与运行状态

真实 CUDA 门禁在 T=32 上得到 anchor 数 `32/17/12/9`，对应估算 MACs
`771.9M/409.7M/289.1M/216.8M`；四档 spatial gradient norm 为
`3.62/4.18/4.47/4.21`，temporal gradient norm 为
`29.77/30.50/31.32/31.25`。这只证明实现、梯度和成本趋势成立，不证明 mAP 保持。

套件最初在生成预检查回执时误用了登录环境旧 Python，因缺少 `pathlib` 停在正式 suite sbatch 提交之前；已经确认 Gate `1180556` 完成后直接提交唯一 Suite `1180557`，没有重复实验。当前四个训练日志只见既有预训练权重 key 提示与 DDP warning，无 Traceback、OOM、non-finite 或 FAIL。

01:24 的首个共同训练点显示四臂均到 P0 epoch 1 batch 20，loss 为
`3.7978/3.6695/3.3860/3.5707`，K 均为 384，显存为
`3719/2108/1568/1315 MB`。d=1 在 batch 20 出现两次 AMP replay 后成功形成有限更新，
没有耗尽 replay 或训练塌缩；当前 detector path 为 P0 预训练阶段的预期 `skipped`，不能据此
评价端到端检测梯度，后续 full-model gate 与 official-60 才负责该证据。

## 2026-07-23 04:20 P0 后置门禁恢复

- Suite `1180557` 的 d=1/2/3/4 均已完成全部 20-epoch P0；失败发生在每臂随后的真实整模门禁，
  四臂堆栈均是同一组六个 BatchNorm 非参数 buffer 被旧初始化器误判为可训练状态缺失。因此
  没有任何 official-60 optimizer update 或 mAP，这不是稀疏间隔的性能负结果。
- 四个封存 epoch-19 P0 的 SHA-256 分别为：d1
  `acb6e30673e811f34ce84d710442581bec8a74ca68e9187eb71e005e01536c9b`、d2
  `e479df3cd29ba4cdde55ee6a08a125741333befc8377b161bd39396a3eb3fc82`、d3
  `aeec4297bf1229b9e4084cf02c91bdd538226d9e8bf87691e1912d1969940b76`、d4
  `c23e370286e3e45bac3f6a48847e3178226cab01cc99921936fbc5c02c8f2c3c`。
- 修复分支为 `codex/duca-sparse-probe-gatefix-20260723`，精确提交
  `cee4ccd33fb20e11978e4a2a6eaa3f5845b51489`。它只移植已验证的非参数 buffer 初始化合同和
  哈希校验 P0 恢复入口，不修改 stride、hidden-linear 插值、R2Q3、K/G 或 detector。
  远端不可变快照为 `/data/run01/sczc063/yuzibo/projects/opentad_duca_sparse_cee4ccd_20260723`，
  Linux focused 为 `10 passed in 39.45s`，pycompile、bash 语法与 clean-tree 均通过。
- 唯一恢复 Job 为 `1180696`，无串行依赖，一次申请四卡并行运行四个 stride；run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_cee4ccd_recovery_20260723_042025`，
  deployment manifest SHA-256 为
  `6c077f3f7ff9076982005ce6ecbb8f1db64cabf42de8e10d26239a920bf88fc5`。当前
  `PENDING(AssocGrpGRES)`，属于账户 GPU 配额等待。
- 状态保持 `experiment_running / no_terminal_map`；禁止重跑四个 P0，禁止把 `1180557` 写成
  d=1/2/3/4 性能失败。

## 2026-07-23 04:25 自动巡检

- 唯一恢复 Job `1180696` 仍为 `PENDING(AssocGrpGRES)`，尚未获得四卡 allocation；这是资源
  等待，不是代码或模型失败，也不触发重复提交。
- 当前没有新的稀疏粗扫日志或 terminal mAP；旧 `1180557` 的四个 Traceback 仍只对应已修复的
  BatchNorm buffer 门禁。

## 2026-07-23 09:28 恢复作业运行时映射失败

- `1180696` 获得四卡后，四个已封存 P0 分支均在 official-60 第一个 optimizer
  update 前失败；` sparse_probe_hidden_linear_d1/d2/d3/d4` 没有登记到
  selected-axis runtime `VARIANT_CONFIGS` 映射，触发 fail-closed `ValueError: invalid
  selected-axis variant`。
- 这是运行时配置合同错误，不是数值崩溃，也不是稀疏 hidden 插值的性能负证据。
  四个 P0 checkpoint 仍可复用；当前没有任何稀疏粗扫 official terminal mAP。
