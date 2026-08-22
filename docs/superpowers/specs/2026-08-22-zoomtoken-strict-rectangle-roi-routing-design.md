# ZoomToken 严格矩形 ROI 与补充 Token 选择设计

- 日期：2026-08-22
- 状态：`designed`，尚未实现、测试或运行实验
- 基线代码：`70dcbe1089866f6ee3821176eb41d2dc10ee8d14`
- 科学问题：在 VideoMAE 重主干之前，严格矩形空间支持能否比当前基于连续 ROI 分数的 Top-K 更好地保护 TAD 准确率；矩形内部选择、动态矩形面积和矩形外少量关键 token 分别带来什么影响。

## 1. 已有证据与本轮边界

当前正式主干前三臂结果为：未修改 AdaTAD `68.73/47.24`、全部 100 token 加相同 sparse adapter `68.51/46.27`、当前 ROI Top-64 `68.22/45.35`（Avg-mAP/mAP@0.7）。当前 ROI 路径已经在 VideoMAE patch embedding 和全部 blocks 之前删除未选 token，但它使用连续 ROI 分数排序后取 Top-64；最终支持不保证形成完整矩形。

本轮不改变官方数据、增强、优化器、scheduler、EMA、检测头、NMS、seed、双卡 global/local batch `2/1` 或 60 轮训练配方。ROI/residual 仍是机制实验，不改写“减少端到端 TAD 中 VideoMAE 空间重计算”的论文主问题。Residual、动态全窗 exact-B、后主干 ROI 和旧 G 路径不混入本轮。

## 2. 共同几何定义

每个两帧 VideoMAE tubelet 由低成本 scout 预测归一化矩形

`g_t = (c_x, c_y, w, h)`，

其中 `w,h` 是完整边长，矩形始终位于图像内。10×10 原生 patch 网格中，第 `n` 个 patch 中心为 `(x_n,y_n)`。严格矩形支持定义为

`R_t = {n : |x_n-c_x| <= w/2 且 |y_n-c_y| <= h/2}`。

硬前向只认上述集合；任何框外 patch 都不能通过 ROI 分支进入重主干。训练使用同一矩形的连续 signed margin

`m_tn = min(w/2-|x_n-c_x|, h/2-|y_n-c_y|)`

构造 straight-through soft membership；hard forward 仍由 `m_tn >= 0` 决定。这样既保持严格矩形，又允许检测损失训练位置和大小。不得复用现有椭圆高斯分数并将其重命名为矩形。R1/R2/R4 为了保证固定格子数，将 scout 中心确定性投影为合法的离散矩形左上角；R3 保留连续中心和连续宽高，并以 patch-center membership 得到自然的离散执行集合。

## 3. 四条待实现方法

### R1：严格固定矩形 ROI

Scout 预测每个 tubelet 的矩形位置，矩形尺寸固定为网格上的 8×8 连续区域；scout 中心确定性映射为 10×10 网格中合法的 8×8 左上角，重主干执行该矩形内全部 64 个 token。矩形位置随视频内容和时间变化，但执行支持始终是无孔洞矩形。训练使用相同中心的 soft window surrogate，硬前向和执行计数只认离散 8×8 block。

作用：与当前 Top-64 ROI 在相同 token 数下比较，直接判断“完整矩形结构”是否更适合保护动作边界。

### R2：ROI 内 Token Select

先形成与 R1 完全相同的离散 8×8 严格矩形候选集，再仅在矩形内部按任务相关基础效用选择 token。第一版执行其中 48 个 token；框外 token 永不可选。选择结果允许在矩形内部有孔洞，因此该臂必须表述为“矩形约束 Token Select”，不能表述为完整矩形 ROI。

作用：判断矩形内部是否仍存在可安全删除的空间冗余。它与 R1 的计算量不同，结果必须同时报告真实成本，不能只比较 mAP。

### R3：动态严格矩形 ROI

Scout 同时预测中心和连续宽高；每个 tubelet 执行矩形内全部 patch，因此

`K_t = |R_t|`

由矩形覆盖自然产生。前向不再在矩形内部 Top-K。训练按成功 optimizer step 计算 train-only 软计数损失

`L_budget = |mean_t(sum_n sigmoid(m_tn/tau)) - 64| / 64`，

仅使平均 token 数接近当前 `64/tubelet` 锚点，不使用 validation/test GT，也不改变 hard forward。运行和成本统计只使用 hard membership 的真实 `K_t`；必须记录每窗总 token、`K_t` 分布、每 16 帧 clip 的 token 数和 attention-pair 数。不得用期望 token 数代替真实执行量；若 hard 平均预算明显偏离 64，该模型只能作为不同成本点报告，不能冒充 K64 对照。

作用：检验“简单场景用小框、复杂场景用大框”的动态空间预算是否优于固定 64 token。

### R4：严格矩形核心 + 框外关键 Free Token

每个 tubelet 先保留一个随内容移动、宽 8 格×高 6 格的严格矩形核心（48 个连续 token）；scout 中心确定性映射为合法离散 block 左上角。随后从矩形外仅按 ROI 无关的基础任务效用选择 16 个关键 token，总数保持 64。框内与框外候选互斥；框外名额严格不超过 16，ROI modifier 不得参与框外排序。

作用：检验当前高 tIoU 损失是否来自严格 ROI 遗漏了框外边界、上下文或相关物体信息。该方法不是旧版“ROI 只加分、全图统一 Top-K”，因为 48 个矩形核心 token 是不可被框外 token 替换的硬支持。

## 4. 备选方案与取舍

1. **只做严格矩形。** 最易解释，但无法判断矩形外关键信息是否是高 tIoU 损失来源。
2. **矩形仅作 eligibility，再在所有候选中全局 Top-K。** 改动最小，但会在矩形中形成孔洞，不能回答用户要求的严格矩形有效性；不作为主臂。
3. **四臂机制矩阵（采用）。** R1 提供严格矩形基准；R2 测框内冗余；R3 测动态面积；R4 测框外补充。四臂共享相同 scout、重主干、adapter、检测器和训练配方，变量边界清楚。

## 5. 代码结构

最小实现复用 `GeoRouteBackboneWrapper._forward_official_fixed_support` 和现有 true-ragged VideoMAE，不建立新 launcher 或模型族：

- `georoute_routing.py`：新增矩形 signed margin、硬 membership、矩形内选择和 core/free 互斥选择纯函数；
- `georoute_wrapper.py`：增加显式 route mode，调用上述函数并把真实 selected native tubelets 送入现有 `forward_native_ragged`；
- 配置：从 70dc 的官方同配方 common config 派生四个 arm config；
- focused tests：验证矩形完整性、框外不可达、core/free 配额、动态 `K_t`、唯一 physical indices、零 padding、单次 heavy forward、无 GT/teacher/oracle/raw-prediction 泄漏。

普通 B/C 路径和旧历史 route mode 默认行为保持不变。新增模式必须显式配置，不能通过修改默认值影响已有结果。

## 6. 数据流与错误处理

`uint8 source/scout -> scout geometry/base utility -> hard rectangle membership -> arm-specific selection -> gather selected native tubelets -> one true-ragged VideoMAE forward -> same sparse adapter -> unchanged ActionFormer`。

任何以下情况直接终止当前 batch/配置检查，不允许静默退回全图或补 dummy token：矩形越界、选择重复、框外 token 进入 R1/R2 core、R4 框外超过 16、执行 token 与 receipt 不一致、padding 非零、heavy forward 次数不是 1。R3 的 hard token 数可以变化，但不能用固定 K 或 padding 修复。

## 7. 测试与实验判读

实现阶段先做纯函数 known-answer、梯度有限性、wrapper production-path 和配置差异测试。静态或无数据测试只能证明实现正确，不能证明性能。

真实实验沿用官方 THUMOS14 validation、60 epoch、seed 42、双卡、相同训练配方。第一阶段至少并列报告 Avg-mAP、mAP@0.6、mAP@0.7、实际 token 数和峰值显存；完整结论还需要同硬件 decode→H2D→model→postprocess→NMS p50/p95 和能耗。R1 与当前 C 是最直接的等 token 因果比较；R2/R3 必须在准确率—真实成本曲线上判断；R4 与 R1 同为 64 token，可直接判断框外 16 个关键 token 是否恢复高 tIoU。

## 8. Claim 与反例边界

可支持的初始 claim 只有：严格矩形、框内稀疏、动态矩形和框外补充在主干前原生 token 路径上的相对效果。若 R1 不优于当前 C，则“矩形完整性本身保护定位”的假设被否定；若 R4 显著恢复 mAP@0.7，则支持“框外关键证据解释固定 ROI 的边界损失”；若 R3 退化到全图或真实成本无下降，则动态面积路线失败。

单 seed、token 数、FLOPs、静态测试或训练可运行性均不能升级为论文级效率结论。
