# DUCA 综合路线与实验中立证据索引

本文件供 Pro 定位原始材料。它不是路线裁决，也不把缺失证据补写成事实。所有数值必须回查随附 Wiki、原始回执或
精确 Git 提交；冲突处保持未知。

## 1. 当前科研边界

- 任务：离线时序动作检测（Temporal Action Detection, TAD）中的 pre-backbone 任务感知时序去冗余。
- 长期目标：使用低成本动作/状态证据分配真实重型视觉计算，在降低端到端成本时保护高 tIoU 定位。
- 当前可靠稀疏基线：H65 fixed K384，低成本 Scout 扫描 768 时间位置，经预算校准系统采样选出 384 个物理位置，
  VideoMAE-S/Adapter/ActionFormer/AdaTAD 后端完成检测。
- 当前数据阻塞：完整 200-video training 与完整 held-out 协议已经由 Pro 冻结；211-video OpenTAD validation 与
  212-video ActionFormer test 的 literal identity 尚未核验，模型工作暂停。

## 2. 主要参考点

| 路线 | 结果 | 成本/身份 | 证据边界 |
|---|---:|---|---|
| Shared Dense AdaTAD | Avg-mAP 68.73 | 768 heavy observations | 参考上界；缺统一 @0.7 与完整同硬件成本 |
| H65 fixed K384 | Avg 65.1257；@0.7 43.3137 | 384/768 heavy observations | 当前最可靠稀疏参考；不是完整 E2E 成本结论 |
| Exact-uniform K384 | Avg 64.49；@0.7 42.45 | matched historical arm | H65 约 +0.90 Avg；训练预算解释见课程部分 |
| Old E2E fixed K384 | Avg 58.39；@0.7 34.53 | commit `70aa069` / Job 1154971 | loader exposure 不匹配，不可裁决 E2E 一般问题 |
| Native stride-2 historical | Avg 64.352 | Job 1150701 | 非当前 matched 主表 |
| Grid-aware historical | Avg 65.696 | Job 1150842 | 非当前 matched 主表 |

## 3. 路线演化与终态

### 3.1 Transition-only、CellCF 与 utility 对齐

- Transition-only/FSU 最初试图用状态变化间接监督选帧，但多次审计发现 direct-boundary mini-localizer、raw-pixel
  bridge、midpoint residue 与 hard one-swap utility 不对齐。
- CellCF 正式结果：exact-uniform 63.8594、transition-beta0 64.2755、CellCF 64.0610 Avg-mAP。CellCF 没有超过
  transition 控制；one-per-cell 只局部移动预算，detached detector utility 不是 detector gradient 直接穿过 hard
  selection。该主方法已终止，不得原样重跑。
- CellCF 公开实现：commit `1642f265e48391418a7c8a4a087e33e2b7bf6899`。

### 3.2 Protected E2E、homotopy、boundary-burst 与 R0

- Protected-E2E 要求 hard forward 与 soft/protected gradient 在同一 physical exact-K DAG 上对齐。
- Selected-axis 60-epoch matched 结果：exact uniform 64.4580；direct-0.25 63.7102；homotopy-0.25 63.0601；
  homotopy+uniform companion 63.6931。所有 learned arms 低于 uniform。
- Boundary-burst 的 R0 是 training-internal holdout oracle：uniform/R2Q3/R4Q5/unrestricted Avg 为
  93.5871/94.1905/93.9992/93.9701，但 paired CI 下界均未过预注册选择门。它否定该冻结 projected feasible set
  能稳定选出 learned family，不是 official validation 模型负结果。
- Fast-only SlowFast Fast pathway + R2Q3：63.5297/42.0937，对 matched uniform 64.49/42.45，Avg -0.9603。
  该具体先验为负，不否定低成本 Scout 一般问题。

### 3.3 训练课程、压缩与蒸馏

- H65 30-epoch exact-uniform Stage-1 + 60-epoch learned Stage-2 terminal EMA：Avg 65.385724，@0.6 58.581766，
  @0.7 43.883956。相对 60-epoch exact-uniform 64.49/42.45，点估计约 +0.896 Avg、+1.312/+1.434
  @0.6/@0.7，但总训练是 90 对 60 轮，不能公平归因于课程或 selector。
- 20+40 压缩：62.4648/@0.7 39.9434；30+30 学习率归因约 63.22/63.56，均未恢复 H65；该压缩族 STOP。
- K192 30+60 历史 terminal 约 57.9673，缺 matched native-K192 uniform，且总训练同样 90 轮。
- 两阶段 coarse/selector warmup 与 joint training 已实现；Job 1178591 到 epoch 8/step 700 后因隐藏 loss、监督泄漏、
  warmup 未隔离 optimizer/clip/EMA 和 bridge 语义而 HOLD。
- CellCF detached utility distillation 64.0610，低于 transition-beta0 64.2755；仅非零 distill loss 不能证明收益。
- 尚无成功 update 总数一致的 scratch-vs-short-warmup，或 feature distillation-vs-no-distillation 因果对照。

### 3.4 物理时间与时间几何

- PhysTime v1：selected-axis/physical-grid/PhysTime Avg 63.61/59.14/57.21，@0.7 41.87/32.34/34.96；该实现为负。
- 后续 physical-metric full60 在特定低性能架构中 selected-axis 41.28 对 physical 57.57，但不能直接作为 H65/Dense
  公平比较。
- RankPack/TrueTime matched：相同 RGB、K、Scout、课程、检测器和 evaluator；RankPack 61.5722/@0.7 37.1003，
  TrueTime 62.1930/37.8918，TrueTime +0.6208 Avg、+0.7915 @0.7。单种子、paired bootstrap 与最终证据包未闭合，
  属部分机制支持。
- TTDI/SingleClock 多个周期因未接入生产 forward、metadata 打包、资源路径或 launcher 问题在 PRE_RUN 前阻塞；这些
  是工程失败，不否定物理时间假说。
- CT-RoPE 仅为外部提案，未实现。

### 3.5 Native tubelet、continuous cliplet 与重构

- 连续 16-frame cliplet commit `8be817...`、Job 1245927 证明连续采集、selected-only VideoMAE 与物理时间重构可运行，
  但只完成 Scout S0，没有 AdaTAD mAP。
- Native-tubelet fixed K384：768 帧组成 384 个原生两帧 tubelet，选 192 个。Uniform Job 1260184 为
  64.13/42.45；task-state coreset Job 1260185 为 62.81/40.56，后者 -1.32 Avg、-1.89 @0.7。两臂都使用
  native tubelet，故该结果检验选择策略，不是 pairing continuity。预测未保存，缺 paired CI 与正式成本。
- FZ_CONTIG/JT_CONTIG 连续 bundle 完整训练 Avg 49.89/47.24，同时改变采集与打包系统，不能隔离配对连续性。
- Sparse hidden-linear bridge 已通过 focused tests 与 CUDA Gate 1180556，证明完整长度、有限值、非零梯度和 MACs
  随稀疏度下降；后续 formal suite 在 mAP 前因 BatchNorm buffer/variant mapping 工程问题停止。没有 nearest/linear/
  Gaussian 在冻结 sparse input/detector 下的正式边界质量对照。

### 3.6 Coverage、Marginal 与动态预算

- Dynamic native-tubelet 16/20/24 clips 路线 commit `d127c2...` 完成实现和静态审查，没有正式训练/mAP/CI/cost。
- Coverage-v1 在 K384 下把 H65 采样替换为 facility-location coverage。PRE_RUN 1261679：集合变化中位数
  48.05%<80%，anchor coverage 增益 3.32%<10%，max-gap P95 2→8；机制门失败，训练前停止，无 mAP。禁止通过
  降低阈值、调 M/sigma/K/M 或 gap repair 继续该版本。
- Marginal/cap-release/96-state：96/96 状态无一过 +0.8 Avg/+1.0 @0.7 联合门；最好接近 +0.554/+0.933，
  additive window loss 与 whole-video AP/Soft-NMS 不一致。该动作空间 STOP。
- Whole-video K256/K384/K512 transfer：704 个合法状态联合门 0/704。Avg 最优约 +0.694/-0.044 @0.7；@0.7
  最优约 -0.236/+0.497；联合最优约 +0.147/+0.490，actual observations 47110→45830（-2.72%）。这否定
  冻结 K384 detector 上的当前整视频 transfer 空间，不否定动态预算一般问题。
- 最新未执行假说：固定控制只用 K384；候选在训练时暴露 nested K256/K384/K512，保持 selector/位置构造/架构/
  loss/time map/NMS/evaluator 不变，以检验检测器预算适应是否重新建立等成本 headroom。尚无代码、训练或结果。

### 3.7 PJST-D1 与其他未闭环证据

- PJST-D1 official inference 的点估计 ON-OFF Avg 为 -0.47248126 个百分点；@0.3--0.7 分别
  -0.7952/-1.2524/-0.1470/-0.2905/+0.1227。
- 两臂各有 211/211 视频与 422,000 predictions，但 finalizer 在 bootstrap 前因预测路径错误失败，0/10,000
  replicates，因此没有 CI、gate 或 population-effect 结论。不能写为“显著负向”。
- Native sparse ActionFormer 诊断中 dense 66.5830、sparse K384 43.9197；2x2 attribution 显示 K384 主效应
  -20.7082、selected-axis loss -1.9552、交互 +0.1810。它属于特定 native sparse head/geometry 的强负证据，
  不等同 H65 主线。

## 4. 当前正式数据协议

- 两条论文主比较臂必须在完整 200-video `training` 上训练，每臂 6,000 successful optimizer updates，从同一 H65
  Stage-1 epoch-29 `state_dict_ema` 起点，匹配 optimizer、LR、EMA、seed、trainable set、augmentation 和 checkpoint。
- held-out 只作一次最终评价；在打开指标前冻结方法、成本匹配、候选 manifest、checkpoint、阈值和停止规则。
- 保存每个预算/每臂的 sealed predictions、video identity、actual observation 与完整成本；统一 official evaluator、
  class map、Soft-NMS；执行 10,000-replicate paired whole-video bootstrap。
- 当前缺失的是 annotation/loader/physical/evaluator/historical 211/source-backed 212 literal identity；该事实审计优先于
  模型代码和 GPU。

## 5. 论文证据缺口

1. 完整 211/212 identity 与无静默排除证据；
2. 当前候选的 full-train terminal EMA 结果；
3. sealed prediction/cost 和 paired bootstrap；
4. 公平 dense/H65/exact-uniform/random/simple-priority baseline；
5. 三种子正式结果；
6. actual observation 之外的 end-to-end FLOPs/latency/throughput/memory/energy；
7. 必要时第二 detector 与第二数据集，但只在 THUMOS14 机制通过后；
8. 与最接近文献的创新边界和论文叙事仍需 Pro 重新冻结。

## 6. 防重复边界

- 不重跑 CellCF、homotopy、Fast-only、Coverage-v1、Marginal/cap-release/96-state 或 704-state transfer 的同一配置。
- 不把 30+60 对 60 的差异当公平 curriculum 证据。
- 不从头重复 generic selected-axis vs physical-time；新方法必须明确区别于 TrueTime。
- 不重复搭建 hidden-linear bridge；如果研究重构，只做冻结输入/检测器的 kernel-only 对照。
- 不把 native-tubelet uniform/coreset 当 pairing continuity 实验；如果研究连续性，只改 tubelet 内物理配对。
- 不在一轮同时加入多预算训练、embedding、蒸馏、新 selector、CT-RoPE、Gaussian splatting、Mamba、Block Drop、
  TensorRT 或跨数据集扩展。

## 7. 需要 Pro 独立裁决的冲突

- 最新 Pro 顺序认为多预算 detector adaptation 是当前唯一未测试单变量；四方向历史核验又显示 pairing-only、
  kernel-only 和 total-update-matched curriculum 仍未闭环，TrueTime 有部分正向证据。哪一个信息增益最高必须由 Pro 判断。
- H65 的 65.13/65.3857 受到 30+60 训练预算与历史 identity 解释影响；应如何构造公平论文基线需重新冻结。
- 论文目标究竟是“等成本性能提升”还是“等精度效率提升”，以及什么门槛足以发表，尚需统一。
- 211/212 事实冲突尚未解决；任何模型路线都不能绕过当前数据准入。

