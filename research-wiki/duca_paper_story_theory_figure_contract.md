---
type: paper_evidence_contract
title: "DUCA 论文叙事、理论与图表合同"
status: designed_waiting_terminal_evidence
canonical: true
updated: 2026-07-23
---

# DUCA 论文叙事、理论与图表合同

本页规定 DUCA 应该如何成为一篇可审稿的离线 TAD 论文。它不改变模型实现，也不把
内部 R0--R5 编号包装成论文贡献。所有结论必须以官方 validation/test TAD mAP、真实总
成本和同协议消融为准。

## 1. 当前编辑裁决

若按当前草稿直接投稿，结论是 **HOLD**。主要原因不是“方法简单”，而是：

1. 当前 `paper/` 仍描述 zero-shot actionness、teacher utility、通用 Top-K、audit ledger 和
   `window-online`，与当前 offline TAD boundary-burst 主线不一致；
2. 核心 C1/C3/C4/C7 仍未由 terminal official mAP、完整成本和多种子闭合；
3. R0--R5、G0--G2、四种粗分类器和多种扫描间隔若直接平铺，会被理解为工程枚举；
4. 内部 frozen-detector holdout 的 93--94 mAP 不是官方论文主结果，不得进入主表或摘要；
5. selected-axis true-time、mandatory-group completeability 与 detector-gradient surrogate
   仍是可能影响性能或主张可信度的风险。

简单方法可以发表，但必须有一个尖锐问题、一个必要而非堆叠的结构化解法，以及能够排除
替代解释的因果证据。

## 2. 唯一论文故事

一句话主张：

> 密集 TAD 把昂贵计算近似均匀地分配给整段视频，而普通 actionness 采样偏向动作内部；
> DUCA 先用低成本粗动作状态估计状态转变，再在固定预算下把观测组织成边界双侧微簇和
> 少量全局上下文，从而在减少重 backbone 观测时保护高 tIoU 定位。

主方法应称为 **transition-calibrated boundary-burst acquisition**，而不是泛化的
“detector-utility-calibrated Top-K”。论文只保留三项贡献：

1. 将高效离线 TAD 表述为 exact-K、最大空洞约束下的任务感知时间观测分配问题；
2. 用粗动作状态的变化间接发现边界，并以双侧、饱和、公平的微簇效用分配预算；
3. 在官方 TAD mAP、高 tIoU、总成本、多个预算和后端上验证准确率-成本收益。

训练课程、审计字段、Slurm 启动器、ledger、X3D/zero-shot 历史路线不属于贡献。
detector feedback 只有在 G1/G2 的真实 hard-swap alignment 和 mAP 均为正时才可列为贡献；
否则只作为消融或负结果。

## 3. 可承担的理论分析

不追求与 mAP 脱节的收敛定理。理论只解释设计为什么必要，并明确适用条件。

### 3.1 受约束观测分配

定义二值选择 `z_t`，在 `sum z_t = K` 和最大未观测空洞 `G` 下最大化结构化效用：

```text
U(z) = transition-anchor utility
     + bilateral boundary-burst utility
     + residual context utility
     - redundant concentration penalty
```

证明或给出命题：对加性状态效用和有限状态约束，当前动态规划返回 exact-K/G 可行域内的
全局最优解；mandatory boundary groups 在接纳前必须满足可补全性。

### 3.2 为什么 actionness Top-K 不等价于边界采样

将粗动作状态近似为分段平稳信号。动作内部 `p_action` 可形成宽平台，因此 Top-K 会把预算
堆在平台内部；状态转变对应 `|delta p_action|`、隐藏特征变化和不确定性峰。该分析只能说明
transition evidence 更符合边界定位，不直接保证 mAP，必须由同协议消融验证。

### 3.3 为什么边界需要双侧微簇

在边界左右观测可辨别“状态前/状态后”，单点或单侧观测只能给出较弱的转变证据。可在明确
观测假设下给出边界区间宽度与左右最近观测距离的上界。微簇收益采用凹函数/饱和配额，表达
前几帧有价值、继续无限堆帧收益递减；endpoint fairness 防止强边界吞掉全部预算。

### 3.4 成本盈亏条件

完整成本写为：

```text
C_total(d,K) = C_decode/materialize
             + C_probe(T/d) + C_interpolate(T) + C_selector(T)
             + C_heavy(K) + C_head(K) + C_post
```

由实测各项给出 DUCA 相对 dense/uniform 的 break-even 区域。只减少 VideoMAE 输入帧但没有
降低完整路径成本时，只能声称减少 heavy-backbone processed observations。

### 3.5 非均匀时间轴风险

相邻 selected positions 的真实间隔 `Delta t_i` 不恒定。把 selected rank 当作规则时间会产生
尺度扭曲；其风险应由 gap variation 与高 tIoU mAP 的关系诊断。只有出现“选帧几何优于
uniform、但高 tIoU 不升”的证据时，才测试零初始化 true-time residual；不得提前把完整
TTDI 写成主方法。

## 4. 主张与必要证据

| 论文主张 | 必要直接证据 | 当前状态 |
|---|---|---|
| transition evidence 比 actionness 更适合边界 | actionness / delta / hidden-change 的同模型消融；边界距离、短动作召回与 official mAP | 粗分类诊断已有，TAD mAP 未闭合 |
| boundary burst 优于均匀采样 | 同 commit、同 K/G、同 detector 的 uniform / simple-delta / DUCA，三种子 terminal EMA | 未证明 |
| DUCA 有准确率-成本优势 | K=384/320/256/192/128 官方 mAP + 完整 p50/p95 latency、FLOPs、能耗、显存 | 未证明 |
| 可作为通用 pre-backbone 插件 | 至少两个真实 detector backend；若声称跨数据集，还需第二数据集 | 第二后端在跑，跨数据集未证明 |
| detector feedback 有效 | no-feedback 与 aligned-feedback 的合法 hard-swap、梯度归属和 mAP | 未证明，不得主打 |

历史 `63.8594/64.2755/64.0610` 和 `64.4580/63.7102/63.0601/63.6931` 仅是负证据与
协议演化背景。R0 内部 93--94 mAP 只回答几何可达性，不回答论文性能。

## 5. 论文实验组织

内部 R0--R5 必须翻译为五个科学问题，而不是方法遍历：

1. **可达性**：固定预算下，合理边界聚集是否存在 mAP 上限；
2. **可学习性**：deploy-visible coarse evidence 能否逼近该分布；
3. **有效性**：learned DUCA 是否在 official mAP 上优于 matched uniform/simple baselines；
4. **归因性**：transition、双侧 burst、饱和配额、全局覆盖和 feedback 分别贡献什么；
5. **效率与泛化**：不同预算、扫描间隔、coarse backend、detector backend 下是否保持 Pareto 优势。

主表至少需要同协议 dense、exact-uniform、random、actionness Top-K、simple transition 和
DUCA；每个 learned 主结果报告三种子均值/标准差、tIoU 0.3--0.7 与 terminal EMA。不得用
历史不同协议数字代替 matched baseline。

## 6. 主文图表合同

### 图 1：问题与方法总览

左侧对比 dense/uniform 的计算浪费与 actionness Top-K 的动作内部偏置；中间展示低成本
状态曲线、转变峰、双侧边界微簇和剩余全局覆盖；右侧展示 exact-K/G hard observations
进入 official-derived detector。该图不再展示旧三阶段、teacher、ledger 或在线任务叙事。

### 图 2：机制可视化

同一视频对齐绘制 GT segments、`p_action`、`|delta p|`、hidden change、selection score，
以及 uniform/actionness/simple-transition/DUCA/Oracle 的硬选帧。至少覆盖短动作、密集相邻
边界、长动作三类；突出边界左右聚集和背景预算转移。

### 图 3：官方准确率-总成本 Pareto

横轴使用完整路径 latency/GFLOPs/energy，纵轴使用 Avg-mAP；另给高 tIoU mAP。绘制 dense、
uniform、random、actionness、simple-transition、DUCA 在多个 K 下的点和三种子误差条。

### 图 4：预算曲线

横轴 K/T 或实际 selected ratio，纵轴分别为 Avg-mAP 与 mAP@0.7；uniform 与 DUCA 同图，
并报告 384/320/256/192/128。不能用 selection proxy 替代 mAP。

### 图 5：机制与失败分析

包含 selected-to-boundary 距离分布、每个 endpoint 的 burst size/左右平衡、短中长动作分层
收益，以及 selection-quality 与 mAP 的关系。相关图只说明联系，因果仍由消融建立。

### 可选图 6：真实时间修正

仅在 TTDI 触发后绘制采样间隔变异度与高 tIoU 损失，以及 rank-time / T1 / T2 的控制实验。

### 主表

1. 主准确率与总成本表；
2. transition/burst/context/feedback/true-time 因果消融表；
3. 两后端及可选第二数据集泛化表；
4. decode、probe、interpolation、selector、VideoMAE、head、NMS 的成本分解表。

所有图使用官方 raw results、统一 K/G 和同协议 detector；机制 proxy 与 mAP 分开；输出矢量
PDF、色盲友好配色、误差条和可复现脚本。禁止雷达图、20 个变体的大柱状图和内部 93--94
mAP headline。

## 7. 投稿前 GO 条件

满足以下条件才从 HOLD 转为可写主方法论文：

1. DUCA 在至少主预算上稳定优于 matched exact-uniform，并保护 mAP@0.6/0.7；
2. 多预算形成优于 uniform 的准确率-总成本 Pareto，而非只减少理论 backbone FLOPs；
3. transition 与 boundary-burst 的关键消融支持设计，而非仅从大量组合中挑最好结果；
4. 第二 detector backend 方向一致；广泛 plug-and-play 主张需要第二数据集；
5. 当前 selected-axis、group completeability 与 feedback alignment 风险得到结果驱动裁决；
6. 全文删除 online/window-online、zero-shot 主方法、teacher-utility Top-K 等旧叙事。

若最终增益不稳定或不足以超过 uniform，正确选择是把工作降级为诊断研究或重新设计核心
采样目标，而不是依靠更多变体、更多工程合同或选择性报告包装成方法论文。

## 8. 当前实验到五类图的证据映射

不得为“每个实验”机械复制五张图。所有实验共享同一绘图脚本、配色、方法顺序和原始产物
schema；每张图只在对应证据真实存在时生成。

- 四粗分类器 P0：只生成同视频 evidence 曲线和选帧/边界诊断；没有 detector，禁止生成 TAD
  Pareto、预算 mAP 或把 AP/AUC 写成 mAP。
- 稀疏粗扫 d=1/2/3/4：terminal official-60 与实测成本齐备后生成总览、同视频机制、扫描成本-
  mAP Pareto、间隔曲线和失败分析；当前运行未结束，不能预画结论。
- R 系列 U/G0/G1/G2 与五预算：终局后生成全部五图，主要检查 learned 方法是否在边界两侧形成
  Oracle 式微簇、是否在 mAP@0.6/0.7 和短动作上获益，以及总成本是否形成 Pareto 优势。
- PhysTime/T1：固定同一 selected positions，优先生成采样间隔变异度、高 tIoU 差值和失败案例；
  只有完整成本齐备才进入主 Pareto 图。
- 免目标域训练候选：必须把冻结 encoder 成本计入横轴，并与 exact-uniform、可训练 DUCA 使用
  相同 K/G 和 detector；若只改善 proxy 而不改善 official mAP，图中明确标为诊断 baseline。

设计预期的可证伪链为：粗 evidence 改善 -> 硬选帧更靠近真实边界且双侧微簇更合理 -> 高 tIoU
mAP/短动作收益 -> 计入全部成本后仍占优。若第一步失败，修 coarse/scorer；若前两步成功而 mAP
不升，才触发 T1；若 mAP 上升但总成本不降，则不能主张高效插件。
