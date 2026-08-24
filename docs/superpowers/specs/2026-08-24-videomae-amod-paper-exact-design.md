# VideoMAE A-MoD 论文语义严格移植设计

日期：2026-08-24

状态：设计已形成，待用户规格复核；尚未实现、测试或训练

基线提交：`c6327a891809aa30370b3b2d9bedab0dcfe0d326`

目标分支：`codex/zoomtoken-amod-v001`

## 1. 研究问题

在不改变官方 AdaTAD 的数据、检测器、损失、优化器、调度器、EMA、评测器和 NMS 的前提下，将 Gadhikar 等人的 A-MoD（Attention routing for Mixture-of-Depths）按论文语义移植到预训练 VideoMAE-S 主干，检验动态深度计算能否在保护 THUMOS14 高 tIoU 定位性能的同时降低真实主干计算量。

本设计只回答 A-MoD 本身的作用，不同时引入 ROI、固定 K64、RC32 特征搬运、KV 全上下文、TokenSelect 或新的可训练路由器。

## 2. 权威方法边界

依据：

- A-MoD：<https://arxiv.org/html/2412.20875>
- 原始 MoD：<https://arxiv.org/html/2404.02258>

论文语义如下：

1. Dense 与 MoD block 交替，每个 MoD block 的路由分数来自紧邻的前一个 Dense block。
2. 对前一层每个 attention head 的 softmax attention map `A[h,j,i]`，按 query 行 `j` 和 head `h` 求平均：

   `r[i] = (1 / (H*N)) * sum_h sum_j A[h,j,i]`。

3. 每个样本按固定 capacity 选择得分最高的精确 `K` 个 token。
4. 选中 token 构成 MoD block 的完整输入，因此其 Q、K、V、输出投影和 MLP 都只在所选子序列上执行。
5. 未选 token 对 VideoMAE Attention+MLP 子块执行恒等旁路。
6. A-MoD 不新增路由参数，不把路由分数乘到 block 输出，也不使用路由辅助损失。

当前 DSR6-KV 的“短 Q、完整 K/V”以及 RC32-KV 的跨 tubelet carry 均不符合第 4 条，不能复用或重命名为 A-MoD。

作者未提供可核验的 VideoMAE 官方实现。本工作只能称为“依据论文方程的 VideoMAE 移植”，不能称为复用了官方 A-MoD VideoMAE 代码。

## 3. 输入与层级调度

官方 AdaTAD 输入为 `16 x 160 x 160`。VideoMAE-S 使用 temporal tubelet size 2 和 spatial patch size 16，因此每个样本产生：

- 8 个时间 tubelet；
- 每个 tubelet 为 `10 x 10 = 100` 个空间 token；
- 总序列长度 `N = 8 x 100 = 800`。

首个验证配置使用论文主设置 `capacity=0.5`，即每个 A-MoD block 每个样本动态处理精确 `K=400` 个 token。

12 个 VideoMAE block 固定为：

```text
Dense-0 -> A-MoD-1 -> Dense-2 -> A-MoD-3 -> Dense-4 -> A-MoD-5
        -> Dense-6 -> A-MoD-7 -> Dense-8 -> A-MoD-9 -> Dense-10 -> A-MoD-11
```

路由 mask 随样本和层动态变化。它不是每 tubelet 固定 K，也不是跨层复用一个 mask。

## 4. 前向数据流

### 4.1 Dense block

Dense block 在全部 800 个 token 上执行现有预训练 VideoMAE Attention+MLP。其 attention 同时返回：

- 正常 block 输出；
- softmax 后、dropout 前的 attention column mean，形状 `[B,N]`。

column mean 只供紧邻的下一 A-MoD block 做 hard top-K。它不写入 loss，不乘到输出，也不读取标签、预测、未来窗口或评测信息。

### 4.2 A-MoD block

1. 对每个样本从上一 Dense block 的 `[N]` 分数取精确 top-400。
2. 将索引按原序排序，gather 为 `[B,400,C]`，保证时空位置顺序确定。
3. 使用该 A-MoD block 原有的预训练 `norm1/qkv/proj/norm2/mlp` 参数，在 400-token 子序列上执行原生 self-attention 与 MLP。
4. 将结果 scatter 回 800-token carrier；未选位置保持进入该 VideoMAE 子块前的值。
5. 路由分数不参与幅值缩放。

稳定 tie-break 使用原 token 索引，仅用于确保精确 capacity 和可复现性，不改变正常非并列分数排序。

### 4.3 AdaTAD adapter 边界

A-MoD 只稀疏化论文定义的 VideoMAE `MHSA+MLP`。现有 AdaTAD temporal adapter 不是预训练 VideoMAE block 的组成部分，继续按官方配方在完整 800-token 网格上执行。

这样可使唯一处理变量成为“预训练 VideoMAE 深度计算是否被 A-MoD 跳过”，避免同时改变 AdaTAD 的任务适配器。成本报告必须把稠密 adapter 的成本保留在总成本中，不能只报告稀疏 VideoMAE FLOPs。

## 5. 预训练参数与训练配方

以下权重从现有 VideoMAE-S checkpoint 原样加载：

- patch embedding；
- 12 个 block 的 LayerNorm；
- QKV 权重和 q/v bias；
- attention output projection；
- MLP；
- 最终 normalization。

位置编码不是 checkpoint 参数：沿用当前 VideoMAE 实现按 token 网格生成并注册的固定正弦位置编码 buffer，并校验其长度与 `8 x 10 x 10` token 布局一致。

A-MoD 不创建任何新 Parameter。官方 AdaTAD adapter、projection 和 ActionFormer head 继续按原配置初始化和训练。

训练严格复用已完成官方 AdaTAD job `1245842` 的数据、train->validation split、seed 42、双卡 global/local batch 2/1、数据增强、AdamW、warmup/cosine scheduler、AMP、60 epochs、EMA、评测器和 NMS。每 5 epochs 保存完整恢复状态，final epoch-59 EMA 为预注册主结果。

## 6. Attention 统计实现策略

当前 PyTorch SDPA 只返回 attention output，不返回论文所需的 column mean。不得用第二次完整 attention 重算分数，因为这会把节省的 QK 计算重新加回来。

官方 AdaTAD 配置使用 `attention dropout=0`。第一版把这一点设为显式合同：A-MoD arm 遇到非零 attention dropout 直接拒绝启动，不在首版引入不同的 dropout/路由统计随机语义。

第一版采用单次、分块的精确 attention：在 query 维分块计算 softmax、输出和 column sum，同一次 QK/softmax 同时产生正常输出与 `r[i]`，避免保存完整 `[B,H,N,N]` attention map。column mean 取 softmax 后的 attention probability；由于 dropout 固定为零，它同时也是实际 attention 输出使用的 probability。该路径必须以数值测试证明：

- 与未分块显式 attention 输出一致；
- column mean 与论文方程一致；
- capacity=1.0 时 12 层输出与原始 dense VideoMAE 路径一致。

若分块 PyTorch 路径没有真实延迟收益，它仍可作为论文语义正确的参考实现，但不能据 FLOPs 宣称加速。只有后续 selector-inclusive 实测通过，才允许提出 fused/Triton 优化。

## 7. 最小实现表面

预计只修改或新增以下项目内表面：

1. `opentad/models/backbones/vit_adapter.py`：attention column mean、交替 Dense/A-MoD 执行、gather/scatter、账本。
2. 一个新的 THUMOS14 A-MoD seed-42 配置，继承未修改官方 AdaTAD 配方，只增加 A-MoD 设置。
3. 现有 pre-backbone N16R4 launcher 的一个新 arm 绑定，不创建第二套 launcher。
4. focused tests：论文方程、top-K、identity bypass、C=1 parity、预训练 state_dict 无新增参数、无泄漏和恢复合同。
5. Wiki、PAPER_PROGRESS 与启动/结果回执。

不修改 detector、loss、NMS、split、官方 baseline 配置，也不复用任何已封存结果目录。

## 8. 验证与实验

### 8.1 实现验收

- 输入/输出 shape 与官方 dense 一致；
- 预训练 state_dict 无 missing/unexpected A-MoD 参数；
- Dense block 全 800 token，A-MoD block 精确 400 token；
- 六个 A-MoD mask 由各自紧邻 Dense attention 独立生成；
- 未选 token 对 VideoMAE MHSA+MLP 严格 identity bypass；
- 无 ROI/K64/RC32/DSR6 语义进入该 arm；
- 同一 seed/config 下 top-K 可重复；
- resume 恢复 model、adapter/head、optimizer、scheduler、scaler、epoch/update 与 RNG。
- attention dropout 精确为零；非零配置 fail closed。

### 8.2 最小正式比较

- Dense 对照：只读复用 untouched official AdaTAD job `1245842`，终态 `68.73 Avg-mAP / 61.58 mAP@0.6 / 47.24 mAP@0.7`，不重复训练。
- 在正式 A-MoD 训练前，使用 job `1245842` 的 final-EMA checkpoint 对新代码的 `capacity=1.0` 路径做一次同配置、同 evaluator 的只读 validation。其输出必须与原 dense 评测在数值容差内一致；这只验证代码路径等价，不产生新的训练基线。
- 新实验：A-MoD-50，seed 42，60 epochs，final EMA。
- 先报告 Avg-mAP、mAP@0.3:0.7、尤其 mAP@0.6/0.7；再报告 selector-inclusive decode-to-NMS p50/p95、峰值显存、gross energy 与真实执行账本。
- 在终态 accuracy 尚未形成前，不启动 12.5% capacity、ROI+A-MoD、多 seed 或成本扩展。

成本人口与计时边界必须和 dense control 完全相同，并计入：视频解码、预处理、H2D、patch embedding、Dense block 的 attention-stat 计算、top-K/稳定排序、gather、A-MoD Attention+MLP、carrier clone/scatter、完整网格 AdaTAD adapter、projection、detector、后处理和 NMS。连续 gross energy 与峰值显存覆盖同一完整区间；不得从账本中扣除路由或数据搬运开销。

## 9. 方案比较与最终选择

| 方案 | 定义 | 是否论文语义严格 | 首轮采用 |
| --- | --- | --- | --- |
| A0 | 完整 800-token 序列，全局 top-400，Dense/A-MoD 交替 | 是，属于 VideoMAE 移植 | 是 |
| A1 | 每 tubelet 各选 50/100 | 否，增加时序覆盖约束 | 否 |
| A2 | ROI K64 后再做 A-MoD K32 | 否，复合 ROI 与深度路由 | 否 |

最终选择 A0。它是回答“A-MoD 本身能否适配预训练 VideoMAE/AdaTAD”的最小、可解释实验。

## 10. 声明边界

- A-MoD 依据图像 ViT/DeiT 论文移植到 VideoMAE；VideoMAE/TAD 效果尚无证据。
- 理论 token/FLOPs 降低不等于端到端延迟或能耗下降。
- 当前运行的 DSR6-KV 是独立深度诊断，不是 A-MoD，不得合并结果。
- 本规格批准前状态为 `designed`；代码、测试和正式实验分别需要后续独立证据才能升级状态。
