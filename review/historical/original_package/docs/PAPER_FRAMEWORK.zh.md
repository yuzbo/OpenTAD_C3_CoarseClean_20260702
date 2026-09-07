# GeoSparse-TAD：论文框架与预注册研究主张

版本：v1.0；日期：2026-09-06。GeoSparse-TAD 为工作名称，未做名称唯一性声明。
本文是研究设计，不包含已实现模型或已完成训练的声明，也不包含虚构结果。

## 1. 论文标题

**GeoSparse-TAD: Task-Value Routing with Geometry-Preserving Sparse Refinement for End-to-End Temporal Action Detection**

中文：**GeoSparse-TAD：面向端到端时序动作检测的任务价值路由与几何保持稀疏细化**。

一句话主张：**保留完整时间位置的低成本状态，只在小模型预测值得的时空区域执行昂贵表征更新，且不把非均匀的计算次序误当成物理时间。**

主路线 A：native token 底座 + 规则 TIA + 稀疏 ViT 更新。
挑战路线 B：dense cheap query + sparse native packet/ROI evidence + support-aware receiver。
挑战路线 C：mixed-scale tokens + 原坐标回写 + 规则时间状态。
三条路线全部实现并启动注册实验；B/C 不是在 A 失败之后才启动的备选任务。

## 2. 可直接用于开题/初稿的摘要（无结果版本）

端到端时序动作检测需要兼顾细粒度语义与精确起止边界，但现有高容量视频编码器通常在冗余背景与困难动作区域投入近似一致的计算。输入级稀疏采样能够减少早期计算，却可能同时丢失短动作证据，并破坏后续时间适配器和多尺度检测器的几何假设。我们研究任务价值驱动的稀疏时空细化：由低成本模型预测追加计算的预期收益，在保留原生时间位置和低成本状态的同时，仅对选中的时空单元执行重型 Transformer 更新。模型以原始采样支持和物理时间为依据进行状态交互，从而将昂贵观测/计算网格与检测输出网格解耦。训练采用同一检测模型产生的任务反馈与有限在线反事实校准，不依赖独立的稠密 TAD Teacher。我们预注册主干内稀疏更新、双分支证据接收及混合尺度计算的并行实验，分别检验路由可预测性、时间几何一致性、动态预算价值和真实硬件收益。实验将报告标准检测指标、短动作召回、高 tIoU、边界误差及端到端延迟，并明确区分架构性能损失、证据遗漏与策略选择误差。

完成实验后方可加入：数据集、mAP 变化、speedup、置信区间和适用限制。没有结果时不得使用“实现无损加速”“显著优于”等措辞。

## 3. 引言的五段逻辑

1. TAD 不只是寻找能判别视频类别的若干证据，还必须覆盖多个动作实例及双端边界。
2. 早删输入能够省早期成本，但风险不可逆；晚删层保留信息，却已支付前缀计算。区分观察缺失、计算跳过和状态缺失。
3. 核心科学问题一不是预测动作标签，而是预测什么追加计算能降低任务风险；该预测受廉价输入的可辨识性限制。
4. 核心科学问题二不是加一个 timestamp embedding，而是在稀疏观测下保持支持范围、局部算子、金字塔和回归坐标的一致性。
5. 提出几何保持稀疏细化，给出三种架构实现的受控比较，并以 Accuracy–Latency 而非仅 FLOPs 评价。

## 4. 研究问题、假设与证据表

| 编号 | 待检验假设 | 正证据 | 反证据/限制 | 对应矩阵 |
|---|---|---|---|---|
| H1 | 同成本下，任务收益路由优于不依赖任务的选择 | A/B 在 uniform、random、motion、actionness 之上改善；低 selection regret | 强简单基线同样好，或收益无法从 cheap 输入预测 | F01,F02,F07,F08,D01 |
| H2 | 保留规则低成本状态比直接删除时间位置更安全 | 匹配预算和训练后短动作/边界更好 | 仅来自额外参数或额外成本 | F00,F04,F05,D02 |
| H3 | 支持范围/真实时间处理改善不规则输入检测 | 同证据集合下 support-aware 优于 rank；gap 增大时优势扩大 | timestamp-only 已经足够，或仅是重新训练收益 | F05,D02 |
| H4 | 跨视频的预算异质性有真实价值 | 动态预算优于最佳固定预算和同直方图打乱 | 差别不显著或来自总成本不同 | F06,D03 |
| H5 | 时间与空间计算收益不完全同序 | 完整 2×2 因子与等成本菜单显示非零交互 | 只有 AP 阈值跳变，没有连续损失/边界收益 | F09,D04 |
| H6 | 理论稀疏可转化成真实加速 | Batch=1 端到端与 Batch=8/32 吞吐改善，p95 不恶化 | decode/TIA/gather 成为瓶颈 | 全部 benchmark,D05 |

这些是研究假设，不是论文已证明的结论。

## 5. 贡献的合规表述

**贡献 1：问题与测量。** 将任务收益可预测性、观测支持几何、计算执行成本放在同一个 TAD 评价框架中，提供误差来源拆分。

**贡献 2：方法。** 给出保留规则时空状态、允许局部零重型更新的路由架构，并建立全量选择时与原模型一致的计算极限；支持集/网格接口适用于三条路线。

**贡献 3：实证与系统。** 提供固定预算、动态预算、粒度、ROI、表示融合、训练估计器和真实硬件的预注册对照。

不得把以下单独称为新颖性：dense-light/sparse-heavy；Top-K；CDF；cross-attention；timestamp embedding；梯度敏感性；一般条件执行。这些分别与 CoDA、Coarse-Fine、mTAN、MSViT 等已有思路重合。

只有 H1/H2/H3 与实际收益得到证据后，才能将方法贡献写成肯定式；结果不支持时应报告负结果、删去不成立的主张，不删除注册实验。

## 6. 方法章节结构

### 3.1 问题定义
给定源视频 V、实际显示时间戳 PTS、标签 Y={(c,s,e)}。定义规则检测网格 G_D、原生 tubelet 网格 G_N、被选计算集合 S。所有 GT 和最终预测使用秒；selected rank 只能作为存储索引。

### 3.2 任务价值与预算
u(a|C,b,theta)=E[L(f_theta(V;S),Y)-L(f_theta(V;S∪{a}),Y)|C,b,a未被选]。
该值依赖模型与已有集合；一阶段 Router 学习对集合分布边缘化后的预期价值，而非完整高阶集合函数。
主优化目标 E[L_TAD]+lambda E[cost]；固定预算与动态预算分别训练并报告。

### 3.3 几何保持的重型更新（A 主路线）
原生 patch embedding 后保持 dense state。重型 MHSA/MLP 在真实压缩的 selected tensors 上执行，结果按 native index 回写。TIA 的原生作用域和位置不随 selected rank 改变。必须在实际源码的每一个计算边界实现，而非随意把原 block 改写成“ViT后接TIA”。

### 3.4 原始支持范围与检测
时间支持采用原生 tubelet 的实际 PTS/曝光单元列表；不连续观测不能用一个包络区间假装连续覆盖。保持原检测器的规则时间 stride 与标签映射。B 在 receiver 处理 irregular evidence；C 在 coarse/fine 变换中保存映射。

### 3.5 单次联合训练
60 epochs、同一个当前模型、无独立 Dense Teacher；硬执行路由通过策略梯度训练，周期性单候选反事实用于收益校准。基线同步实现 PG-only、梯度代理、明确标记有偏的 ST estimator。所有额外 probe 和搜索计入训练成本。

### 3.6 硬件实现
原生 clip attention partition、packing、bucket 与 qkv/MLP 实际形状、dense TIA、端到端数据流分别计量。

## 7. 两个可陈述而不过度宣称的性质

**性质一：dense-limit 一致性（有条件）。** 若 S=全部有效 native tokens，gather/scatter 互逆，attention partition、PE、normalization、residual、TIA 和 masking 均与原模型一致，且评估时关闭随机算子，则 A 与原模型具有相同计算图输出（浮点允许误差）。这不是准确率保证，且必须通过数值及梯度测试。

**性质二：存储排列不变性（有条件）。** 对证据存储顺序作排列，同时排列相应坐标、支持、mask 和原生位置编码，使用集合式 receiver/同组 attention 的输出应保持一致。任意改动 parent clip 分组、或改变位置编码不属于该性质。

不能声称稀疏证据恢复所有未观察动作；完整位置覆盖不等于信息无损。

## 8. 实验章节与论文图表

- Table 1：两数据集的主结果；A/B/C、dense、DUCA、uniform、静态浅层；mAP、high-tIoU、short recall、端到端 p50/p95、throughput、显存。
- Table 2：选择原子、编码 packet 与执行分组，区分 equal-token 与 equal-latency。
- Table 3：receiver/time-geometry 2×因素与同一证据计划测试。
- Table 4：Router 训练与可预测性；Spearman、符号准确率、regret、probe 开销。
- Table 5：固定/动态预算及 histogram shuffle。
- Table 6：时间×空间细化、原生 patch block 与原图 ROI。
- Table 7：系统成本分解及不同 batch/硬件。
- Figure 1：动机：错误邻接和早删信息风险。
- Figure 2：A 主结构和 B/C 比较接口。
- Figure 3：Accuracy–Latency Pareto 与置信区间。
- Figure 4：短动作/gap/边界误差剖面。
- Figure 5：真实收益与预测收益、预算分布。
- Appendix：全部 manifest、失败/缺资源条目、估计器公式、seed、测试、复现命令。

## 9. 结果选择与负结果政策

主架构预指定 A；B/C 平行挑战。不得看 held-out test 后更换主架构并继续声称预指定。可根据内部开发集提出 amendment v2，但必须保留 v1 结果与时间戳。训练主比较使用最后 epoch59 checkpoint；best-dev 仅作次要报告且各方法同规则。

拟定工程参考目标而非保证：在 token 更新占比约0.5处，主 mAP 下降≤1.0个百分点、预注册 high-tIoU 指标下降≤1.5个百分点、短动作 recall 下降≤2.0个百分点，同时端到端 batch1 p50 加速≥1.30倍且 p95 不恶化。完整 Pareto 必须公开，即使达不到这些参考线。

## 10. 局限性必须提前承认

A 不省全部解码、Patch Embedding 和 TIA；B 有异构融合/局部证据不足风险；C 粗 token 并非零计算。低分辨率不可辨识、小动作在所有分支不可见、真实 VFR 证据有限、标注边界噪声、p95 调度以及 GPU 固定开销均可能限制收益。
