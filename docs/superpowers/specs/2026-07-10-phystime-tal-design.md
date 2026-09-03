# PhysTime-TAL 设计规格

## 1. 方法定位

PhysTime-TAL 是一个独立的离线时序动作检测方法。它接收任意不规则视频观测及其真实时间戳，直接在物理时间轴上完成动作分类和起止边界定位。

它不是：

- DUCA 的补丁或配套检测头；
- 选帧、动态预算或在线 TAD 方法；
- selected-axis 预测后的坐标映射器；
- MoD 式 token 路由或条件计算模块。

方法的核心研究问题是：

> 当视频观测具有不同帧率、随机缺失、连续空洞和非均匀密度时，如何使 TAD 检测器在真实物理时间上保持稳定的感受野、边界语义和定位结果？

目标贡献为：

1. 物理时间原生的时序编码器；
2. 按真实时间跨度构造的多尺度特征金字塔；
3. 直接在物理时间上预测起止边界的 hazard 检测头；
4. 面向不规则观测的重采样等变训练与评测协议。

## 2. 输入输出契约

单个视频的输入为：

\[
\mathcal O=\{(x_i,t_i,\Delta t_i)\}_{i=1}^{K},
\]

其中：

- `x_i` 是第 `i` 个观测的视觉特征；
- `t_i` 是真实帧号、秒数或归一化视频时间；
- `delta_t_i` 是该观测代表的物理时间跨度；
- `K` 可以随视频和采样模式变化；
- `valid_mask` 只表示 padding，不改变物理坐标。

必须满足：

- `t_i` 严格单调递增；
- GT 始终保留在原始物理时间轴；
- 禁止 selected-axis GT remap；
- 推理输出直接使用原始帧号、秒数或归一化时间；
- 不需要 selected-axis inverse remap。

模型输出为动作区间集合：

\[
\hat{Y}=\{(\hat{s}_j,\hat{e}_j,\hat{c}_j,\hat{p}_j)\}_{j=1}^{N}.
\]

## 3. 候选架构与裁决

### 3.1 Timestamp ActionFormer

在现有 ActionFormer 中加入 timestamp embedding，并将 point center/stride 映射到物理时间。

优点是改动小，可作为强基线。缺点是 projection、neck、Conv1d 和 attention 仍在 selected-rank 上运行，不能作为最终方法。

### 3.2 Resampling-Equivariant PhysTime-TAL

采用连续相对时间编码、物理时间邻域、多尺度物理 query 和重采样等变监督。

这是最终推荐方案。它直接修改检测器内部的时间几何，而不是只修正输出坐标。

### 3.3 Neural CDE/ODE-TAL

通过连续动力系统演化隐状态。其训练成本、数值稳定性和实现风险较高，仅保留为后续对照，不进入第一版主方法。

## 4. 最终网络结构

### 4.1 视觉特征接口

第一版复用标准视频特征或 AdaTAD/VideoMAE 特征提取流程，但 PhysTime-TAL 本身不依赖特定采样器。输入适配器输出：

```text
features:       [B, K, C]
timestamps:     [B, K]
support_widths: [B, K]
valid_mask:     [B, K]
duration:       [B]
```

时间统一采用归一化坐标 `[0, 1]` 参与网络计算，同时在 metadata 中保留帧号、秒数、FPS 和视频时长用于审计与结果转换。

### 4.2 物理时间嵌入

每个观测的输入表示为：

\[
h_i^0=W_xx_i+E_t(t_i)+E_w(\Delta t_i).
\]

`E_t` 使用多频 Fourier 特征与小型 MLP；`E_w` 编码观测支持区间，避免相同时间中心但不同覆盖范围被视为等价。

### 4.3 连续相对时间注意力

任意两个观测的相对时间偏置为：

\[
b_{ij}=\phi(t_i-t_j,|t_i-t_j|,\log(1+|t_i-t_j|),\Delta t_i,\Delta t_j).
\]

注意力写为：

\[
A_{ij}=\operatorname{softmax}_j
\left(\frac{q_i^\top k_j}{\sqrt d}+b_{ij}+\log w_j\right),
\]

其中 `w_j` 是由相邻时间戳计算的时间覆盖权重。该权重用于抵消局部采样密度差异，防止密集区域仅因 token 数量更多而获得额外质量。

第一版采用稠密 attention 作为正确性锚点；后续只有在 profiling 证明必要时才引入按物理半径构造的稀疏邻域。

### 4.4 物理时间多尺度 query 金字塔

禁止通过 selected-token `stride=2` 建立金字塔。每一层在归一化物理时间上建立规则 query：

```text
Level 0: M 个 query
Level 1: M/2 个 query
Level 2: M/4 个 query
Level 3: M/8 个 query
```

每个 query 通过 continuous-time cross-attention 从不规则观测中聚合特征。query 的中心、支持区间和有效范围均为物理时间量。

`M` 是检测查询分辨率，不等同于输入观测数量 `K`。因此相同视频的不同采样模式可以产生相同定义的检测输出网格。

### 4.5 物理时间边界头

每个 query 预测：

- 动作类别概率；
- start hazard；
- end hazard；
- 左右物理时间距离。

边界解码为：

\[
\hat{s}=t_q-\operatorname{softplus}(d_s),\qquad
\hat{e}=t_q+\operatorname{softplus}(d_e).
\]

距离以归一化视频时间训练，评测时直接转换为帧号或秒数。assignment、回归损失、NMS 和评测全部使用物理时间坐标。

## 5. 训练方法

### 5.1 双视图不规则采样

同一个 dense 训练视频在线生成两种独立的不规则观测：

\[
\mathcal O^{(1)}\sim S_1(V),\qquad
\mathcal O^{(2)}\sim S_2(V).
\]

训练采样族包括：

- 不同固定帧率；
- 均匀降采样；
- 独立随机丢帧；
- 连续时间空洞；
- 局部密集与局部稀疏；
- 时间戳 jitter；
- 最大观测间隔 5、10、15 帧。

采样器不得读取 GT 边界或类别。GT 只用于检测监督，不参与观测决策。

### 5.2 精简损失

总损失保持为三个可归因项：

\[
L=L_{\mathrm{TAD}}+\lambda_{eq}L_{\mathrm{resample}}+\lambda_hL_{\mathrm{endpoint}}.
\]

- `L_TAD`：标准分类与物理时间区间回归；
- `L_resample`：两个不规则视图在物理 query grid 上的预测一致性；
- `L_endpoint`：start/end hazard 监督。

`L_resample` 只比较共同有效物理时间区域内的 hazard、类别分布和匹配区间，不要求输入 token 特征逐点一致。

禁止重新引入 actionness、budget、gap、radius、entropy 或 selector utility loss。

### 5.3 训练阶段

第一版采用单次端到端训练，不训练或导出独立 selector：

```text
双不规则视图
  -> 共享 PhysTime-TAL
  -> 两份 TAD 监督
  -> 跨采样等变监督
  -> 一次反向传播
```

允许先用 dense-only 配置做数值稳定性 smoke，但最终模型不能依赖分阶段 checkpoint 拼接。

## 6. 当前代码复用与新增边界

可以复用：

- physical positions 与 dense-valid-length metadata；
- dense-axis GT fail-closed contract；
- physical point assignment 与坐标 round-trip 测试；
- OpenTAD 的分类、回归、NMS 与 THUMOS 评测接口；
- 现有 AdaTAD/ActionFormer 配置作为 matched baseline。

不能把以下现有实现当作最终方法：

- `AnchorFreeHead._build_physical_points_and_masks()` 只修正 point center/stride；
- selected-rank Conv1d/attention；
- selected-axis inverse remap；
- DUCA 或 PC-OT-MRAS selector；
- offline ledger、JSONL actionness 或 X3D/SlowFast prior。

建议新增边界：

```text
opentad/models/projections/phystime_projection.py
opentad/models/necks/phystime_pyramid.py
opentad/models/dense_heads/phystime_head.py
opentad/models/utils/phystime_geometry.py
configs/adatad/thumos/phystime_tal_*.py
tests/test_phystime_*.py
```

具体文件名可在实现计划中根据现有 registry 和模块归属调整。

## 7. 实验设计

### 7.1 必要基线

1. 原始 dense ActionFormer/AdaTAD；
2. selected-axis ActionFormer；
3. selected-axis + timestamp embedding；
4. 线性插值回 dense grid + ActionFormer；
5. 当前 physical point assignment；
6. PhysTime encoder，不使用重采样等变损失；
7. 完整 PhysTime-TAL。

### 7.2 评测矩阵

每个方法在以下条件下评测：

- 不同观测数量 `K`；
- 最大间隔 5、10、15 帧；
- uniform、random、bursty、motion-biased 和 selector-produced observations；
- 不同原始帧率；
- THUMOS14 主实验，第二数据集用于泛化验证。

报告：

- Avg-mAP 与各 tIoU mAP；
- mAP@0.7；
- 按动作时长分组的 mAP；
- 不同采样策略的均值、标准差和最差值；
- 同一视频跨采样预测一致性；
- 参数量与 FLOPs/MACs；p50/p95 延迟、吞吐量与显存仅作为可选系统诊断，不作为训练或性能结论门禁。

### 7.3 核心主张与证据

| 主张 | 必要证据 |
| --- | --- |
| selected-axis 破坏时间几何 | 相同 K 下不同 gap pattern 导致显著性能方差 |
| PhysTime-TAL 具有重采样鲁棒性 | 多采样策略均值提高且方差、最差值下降 |
| 连续时间机制不是 timestamp embedding | 完整方法显著超过 timestamp embedding 基线 |
| 高 tIoU 和短动作受益 | mAP@0.7 与短动作分组结果 |
| 方法与采样器无关 | 至少三种采样策略和一种 learned selector 输入 |

## 8. 测试与失败处理

必须提供：

- timestamp 单调性和合法区间测试；
- padding 不改变物理坐标测试；
- dense 等间隔输入退化为标准时间轴测试；
- batch 内不同 `K` 测试；
- GT 永不映射到 selected-axis 测试；
- 输出不需要 inverse remap 测试；
- 两种采样视图共享权重且产生有限梯度测试；
- physical query 与 endpoint assignment 测试；
- 预测区间始终满足 `start <= end` 测试；
- 一个真实 OpenTAD 配置的 one-step backward 与 GPU smoke。

所有缺失 timestamp、非单调时间、坐标单位冲突和 GT remap 都必须 fail-closed。

## 9. 停止条件

出现任一情况，应停止把 PhysTime-TAL 作为论文主方法：

1. 线性插值 + timestamp embedding 与完整方法在置信区间内持平；
2. 完整方法不能显著降低跨采样策略的性能方差或最差性能；
3. mAP@0.7 与短动作没有稳定收益；
4. 只在单一采样器上有效；
5. 连续时间模块的额外成本明显超过其精度收益；
6. 第二数据集无法复现重采样鲁棒性。

## 10. 第一阶段完成标准

第一阶段只建立最小但完整的科学闭环：

1. 从任意不规则特征和 timestamp 构建物理 query grid；
2. 完成连续相对时间编码和物理时间边界预测；
3. 完成双视图重采样等变训练；
4. 跑通真实 THUMOS14 OpenTAD full train；
5. 完成 dense、timestamp embedding、linear interpolation 和 selected-axis 四个 matched baseline；
6. 得到按 K、gap、采样策略和动作时长分组的完整结果。

在第一阶段结果通过停止条件前，不扩展到 codec、动态计算、在线 TAD、CDE/ODE 或多任务任务族。
