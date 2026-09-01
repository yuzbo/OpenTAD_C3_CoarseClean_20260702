# DUCA / CT-DP-TAD 高效时序动作检测技术全景与科学路线报告

> **报告版本**：v1.0 (2026-09-01)  
> **代码库主线**：[OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)  
> **审查基准提交**：[`a2039858`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a2039858)  
> **预注册实验矩阵**：4 臂 2×2 正交消融方阵（Job `1264291` ~ `1264294` @ N16R4）

---

## 一、 科学问题与研究背景 (Scientific Problem & Motivation)

### 1. 端到端时序动作检测的计算瓶颈
在未修剪的长视频中，动作通常只占据局部时序区间，背景冗余度极高。然而，主流的高精度端到端模型（如以 VideoMAE-S 为骨干网络的 ActionFormer / AdaTAD）在处理长视频窗口（例如 768 帧稠密时序窗口）时，**超过 75% 的 FLOPs、显存与端到端延迟消耗在重型视频骨干网络的前向计算上**。

### 2. 现有去冗余方案的三大根本性缺陷
针对长视频计算瓶颈，现有探索存在三大核心矛盾：
1. **简单均匀降采样（Uniform Downsampling）**：若将 768 帧均匀抽取至 384 帧甚至 192 帧，会严重平滑动作的起止边界（Boundary Transition），导致高交并比阈值下的定位性能（如 $\text{mAP@0.7}$）发生断崖式下跌；
2. **传统语义 Top-K 选帧（Semantic Top-K Selection）**：选帧策略容易过度聚集在动作高潮部分（波峰），产生两大派生病态：
   * **时间覆盖空洞（Coverage Hole）**：长背景与动作过渡段被完全漏选，造成大范围漏检；
   * **时序物理速度畸变（Temporal Velocity Distortion）**：非均匀跳帧破坏了时空连续性，使固定参数的标准 3D 卷积将跳跃帧误判为“超高速运动”；
3. **检测头坐标错位截断（Coordinate Truncation Bug）**：检测头若停留在离散序号网格上，后半段动作（$[384, 768]$）会被静默截断为背景。

---

## 二、 科学路线演进与设计哲学 (Roadmap Evolution & Scientific Philosophy)

整个研究路线遵循**“单变量因果消融、严格等价性退化、零假设可复现验证”**的科学演进法则：

```
 [历史探索期: C3 / PAction / GAS-VT / Lattice]
   │  探索粗粒度筛选与特征缓存，确立端到端直接在 RGB 像素级去冗余的大方向
   ▼
 [DUCA 雏形期: H65 实验 (K=384, Avg-mAP 65.13%)]
   │  确认 384 帧 Top-K 选帧与 768 Dense 基线 (68.73%) 存在 3.6 个百分点差距
   │  确立核心科学假设：时间覆盖空洞 + 物理速度畸变是主要性能瓶颈
   ▼
 [物理时间探索期: PJST-D1]
   │  初步验证基于时序导数归一化的物理时间变换可行性
   ▼
 [当前系统主线: CT-DP-TAD / DUCA-Coverage-v1]
   │  1. 选帧层: 双相正交预算 (Scaffold 128 + Burst 256) 与边界敏感次模覆盖优化
   │  2. 骨干层: CT-Tubelet 3D 卷积速度归一化 + B-AMoD ViT 奇偶稀疏调度 (恒等直通)
   │  3. 检测层: 连续物理尺度自适应卷积 (CT-Conv) + [0, 768] 物理坐标全闭环
   ▼
 [下阶段储备: 连续几何与最优传输]
      Time-Aligned RoPE (TARoPE) + 连续物理 GIoU 损失 + Sinkhorn 最优传输梯度
```

---

## 三、 端到端系统架构与技术机制 (System Architecture)

系统由四大紧密协同的子模块构成，形成自底层输入到检测输出的完全物理闭环：

```
 原始视频输入 [B, 1, 3, 768, 160, 160] (768 帧稠密时序窗口)
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. 任务感知选帧层 (Task-Aware Keyframe Selection)             │
 │    • Dual-Phase: 128 全局骨架 (Scaffold) + 256 相变微簇 (Burst)│
 │    • Submodular: 变化能量质量项 + 饱和高斯衰减时间覆盖核      │
 │    ──────────────────────────────────────────────────────── │
 │    双链解耦输出:                                             │
 │    • tubelet_delta_t [B, 192]: 原始帧对物理间隔 (专供骨干网)  │
 │    • synced_temporal_positions [B, 384]: 连续时间戳 (专供检测头)│
 │    • detector_delta_t [B, 384]: 检测网格连续步长 (专供 CT-Conv)│
 │    • boundary_prior [B, 384]: 边界调制先验 (专供 B-AMoD)     │
 └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 2. 物理速度归一化与稀疏骨干网络 (CT-Tubelet VideoMAE-S)       │
 │    • 24 Chunks × [B, 3, 16, 160, 160]                       │
 │    • CT-Tubelet 3D Patch 嵌入 (T=2 正交速度归一化):          │
 │      W_mean*(x0+x1) + W_diff*((x1-x0) * 1/tubelet_delta_t)   │
 │      -> 8 Tubelets × 100 空间 Token = 800 Tokens/Chunk      │
 │    • 12 层 ViT-Adapter with B-AMoD (边界偏置自适应稀疏路由):  │
 │      - 6 层 Dense (层 0,2,4,6,8,10) 保持全局时空感知         │
 │      - 6 层 Sparse@50% (层 1,3,5,7,9,11):                    │
 │        Top-50% 关键 Token 经 Heavy MHSA/MLP/Adapter          │
 │        未选 50% Token 严格恒等直通 (Zero Mutation Bypass)   │
 │      -> 等效 9 层计算，直接降低骨干网络 25%~35% FLOPs       │
 │    • 空间池化 + 1D 线性插值 (align_corners=False)           │
 │      -> 输出 [B, 384, 384] 同步时序特征                      │
 └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 3. 投影与多尺度时序金字塔 (Projection & FPN Pyramid)         │
 │    • Conv1D + Temporal Transformer Stem                     │
 │    • 6 级 FPN 金字塔: L0: 384, L1: 192, L2: 96, ..., L5: 12 │
 └─────────────────────────────────────────────────────────────┘
                         │
                         ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 4. 连续物理坐标检测头 (Physical-Grid ActionFormer Head)      │
 │    • CT-ScaleAdaptiveConv1d: 依据局部步长动态调制采样偏移     │
 │    • Physical PointGenerator: 检测锚点直接映射至 [0, 768]    │
 │    • 零截断 Target Assignment 与回归损失计算                 │
 │    • 原生物理轴秒数无失真还原 (irregular_native_axis=True)   │
 └─────────────────────────────────────────────────────────────┘
```

---

## 四、 核心创新机制深度拆解

### 1. 任务感知选帧器 (Keyframe Selector)
当前系统支持两套互补的选帧方案：

* **方案 A：双相正交预算选帧器（`DualPhaseFrameSelector`）**：
  * **全局骨架相（Scaffold Phase, $K_{\text{scaffold}}=128$）**：在 $[0, 768]$ 稠密网格上均匀锚定采样，步长恒为 6 帧，作为全视频宏观时序上下文底线，彻底杜绝背景空洞；
  * **相变微簇相（Burst Phase, $K_{\text{burst}}=256$）**：基于无参数的低分辨率像素差分变化能量 $E(t) = \operatorname{mean}(|I_{t+1}-I_t|)$，在动作发生与起止边界周围采样半径 $R=2$ 的密集偶极微簇。
* **方案 B：边界敏感次模覆盖选帧器（`SubmodularCoverageFrameSelector`）**：
  * 基于次模目标函数：$\max_{|S|=K} \sum_{t\in S} Q(t) + \beta \sum_{t\in V}\left(1 - \exp\left(-\frac{\min_{s\in S} d(t,s)^2}{2\sigma^2}\right)\right)$；
  * $Q(t) = E(t) + \alpha |E(t+1)-E(t)|$ 显式强化边界梯度，饱和指数核防止在长动作内部过度冗余堆叠，GPU 上 $O(K\cdot T)$ 增量贪心求解。
* **物理时间解耦输出**：
  * 输出原始帧对间隔 $\Delta t^{\text{pair}}_j = t_{2j+1} - t_{2j}$ (`[B, 192]`) 专供骨干网；
  * 输出插值连续时间戳 $\tau^{\text{detector}}$ (`[B, 384]`) 专供检测头。

---

### 2. CT-Tubelet 物理速度归一化 3D 卷积
* **解决的物理问题**：标准 3D 卷积核固定假设帧率均匀（$\Delta t \equiv 1$）。当稀疏采样导致帧对物理间隔变为 $\Delta t > 1$ 时，物体的表观运动差分会在特征空间被错误放大。
* **数学正交分解**：
  VideoMAE 3D 卷积核在 $T=2$ 维度上正交拆解为稳态分量与动态差分分量：
  $$W_{\text{mean}} = \frac{1}{2}(W_0 + W_1), \quad W_{\text{diff}} = \frac{1}{2}(W_1 - W_0)$$
  前向输出计算为：
  $$Y = W_{\text{mean}} * (X_0 + X_1) + W_{\text{diff}} * \left((X_1 - X_0) \cdot \frac{1}{\Delta t^{\text{pair}}}\right) + b$$
  * **退化性质**：在均匀连续采样（$\Delta t^{\text{pair}} = 1.0$）下，代数上严格退化为原生 3D 卷积；
  * **物理一致性**：非均匀跳帧时，动态分量显式除以真实物理时间差，使得单位时间内的表观运动速度在特征流中保持严格守恒。

---

### 3. B-AMoD 边界偏置自适应混合专家 (Boundary-Biased A-MoD)
* **零额外参数路由**：
  复用自注意力矩阵的列均值（代表 Token 的全局影响力）并调制侦察器边界先验：
  $$s_i = \left(\frac{1}{N}\sum_{j=1}^{N} A_{i,j}\right) \cdot (1.0 + \alpha_{\text{prior}} \cdot p_{\text{boundary}}[i])$$
* **严格恒等直通（Zero Mutation Bypass）**：
  $$Y_i = \begin{cases} \operatorname{Block}(X_i), & i \in \operatorname{TopK}(s, K=0.5N) \\ X_i, & i \notin \operatorname{TopK}(s, K=0.5N) \end{cases}$$
  未选中的 50% Token 绕过昂贵的多头自注意力和 MLP，保留原始特征，消除时空断裂。

---

### 4. 连续物理坐标检测头 (Physical-Grid ActionFormer Head)
* **感受野尺度自适应**：
  标准 Conv1d 的采样点为 $u \pm \text{stride}$。而在非均匀时间轴上，`ContinuousTimeScaleAdaptiveConv1d` 依据目标物理时间 $\tau_{\text{target}} = \tau(u) \pm \Delta \tau_{\text{ref}}$ 进行逆分段线性插值计算分数偏移，使检测头在真实物理秒维度上维持恒定的感受野大小。
* **物理坐标闭环**：
  PointGenerator 锚点通过 `irregular_selected_positions` 直接提升到 $[0, 768]$ 物理域，正样本分配在统一物理度量空间进行，后处理直接换算秒数，杜绝了坐标截断与二分之一失真。

---

## 五、 正式 2×2 消融实验矩阵与协议规范 (Comparable Matrix)

为彻底阻断任何“未消融拼凑”质疑，系统预注册了标准的 2×2 全因子正交消融矩阵：

| 实验臂 | 配置文件 | CT-Tubelet | B-AMoD (6D+6M) | CT-Conv1d | 科学验证目的 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Arm 1 (Full)** | `duca_ct_dual_phase_bamod_thumos.py` | **ON** | **ON** | **ON** | 完整主方法性能上限 |
| **Arm 2** | `duca_dual_phase_bamod_thumos.py` | **ON** | **ON** | **OFF** | **单一消融 CT-Conv**，验证检测头连续几何贡献 |
| **Arm 3** | `duca_ct_dual_phase_densevit_thumos.py` | **ON** | **OFF** | **ON** | **单一消融 B-AMoD**，验证骨干稀疏路由贡献 |
| **Arm 4 (Control)** | `duca_dual_phase_densevit_stdconv_thumos.py` | **ON** | **OFF** | **OFF** | **双消融基准对照**，提供无进阶算子的严谨底线 |

* **冻结可比性协议**：
  * 数据集与评测：THUMOS14（官方 200 训练 / 211 验证），评测 IoU 阈值 $[0.3, 0.4, 0.5, 0.6, 0.7]$；
  * 训练参数：`seed=3407`，AdamW，Base LR 1.8e-4 (Backbone) / 9.0e-5 (Det)，Cosine Annealing，总计 60 轮；
  * 评测协议：第 42 轮起评，每 2 轮评测一次（`val_start_epoch=40, val_eval_interval=2`），密封预测结果一次性读取。

---

## 六、 进阶储备机制 (Next-Stage Reserve Mechanisms)

1. **时间对齐旋转位置编码（Time-Aligned RoPE / TARoPE）**：
   * 模块路径：[`opentad/models/bricks/time_aligned_rope.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/opentad/models/bricks/time_aligned_rope.py)
   * 将标准整数位置索引 $m \in \{0, \dots, T-1\}$ 推广至连续物理时间戳 $\tau \in \mathbb{R}$：
     $$\phi_j(\tau) = \tau \cdot \text{base}^{-2j/d}$$
   * 严格满足相对时间平移不变性：$\langle R(\tau_q + \Delta) q, R(\tau_k + \Delta) k \rangle = \langle R(\tau_q) q, R(\tau_k) k \rangle$，天然适配非均匀时序注意力。

2. **连续物理时间 GIoU 损失（`ContinuousPhysicalGIoULoss`）**：
   * 模块路径：[`opentad/models/losses/iou_loss.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/opentad/models/losses/iou_loss.py)
   * 在连续物理秒坐标系下直接优化预测区间与真值区间的 GIoU 及中心距离惩罚。

3. **Sinkhorn 最优传输选帧损失（`SinkhornOptimalTransportLoss`）**：
   * 模块路径：[`opentad/models/losses/sinkhorn_ot_loss.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/opentad/models/losses/sinkhorn_ot_loss.py)
   * 基于熵正则 Sinkhorn-Knopp 算法，为选帧概率分布提供全局连续的最优传输梯度。

---

## 七、 实施与验证状态 (Current Status & Deliverables)

1. **工程与单测状态**：
   * 本地 20 项全链路单元测试 **100% 通过**（覆盖解耦时间路由、坐标对齐、空间 Token 完整性、CT-Conv 等价性与梯度反向传播）；
   * 最新代码已同步推送到 GitHub 仓库：[Commit `a2039858`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a2039858)。
2. **集群消融实验监控**：
   * N16R4 集群 4 臂消融矩阵（Job `1264291` ~ `1264294`）正按计划平稳推进，严格遵循第 42~60 轮评估协议。
3. **论文最终交付物（Deliverables）**：
   * **主性能表**：$K=384$ 下的 Avg-mAP、mAP@0.7 相比 Dense 768 及 Uniform 384 的对比；
   * **Pareto 效率图**：端到端 GFLOPs、显存峰值（VRAM）、推理延迟 vs Avg-mAP 的 Pareto 前沿；
   * **2×2 消融因果分析**：CT-Conv、B-AMoD 与 CT-Tubelet 的独立增益与协同效应。
