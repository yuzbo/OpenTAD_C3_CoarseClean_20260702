# 面向高精度时序动作检测的连续时空自适应与双相 B-AMoD 计算体系 (CT-DP-BAMoD)

## 状态

- `implemented_provisional` (已完成算法实现与单元测试，待正式集群实验检验)
- 2026-09-01 已完成核心算法代码实现、张量对齐与多维单元测试套件。
- 遵循第一性原理，旨在解决非均匀选帧引起的物理时序几何失真、粗 Scout 边界空间混叠、平滑密度与双相动力学错配、以及 VideoMAE 全层统一计算冗余四大瓶颈。

## 核心机制设计

1. **双相正交预算分配 (Dual-Phase Orthogonal Budget Allocation)**:
   - 将总预算 $K$ 解耦为正交的全局稳态骨架 $S_{\text{scaffold}}$（$K_{\text{scaffold}}=128$ 均匀覆盖）与相变微簇 $S_{\text{burst}}$（$K_{\text{burst}}=256$ 锁定边界峰值）。
   - 目标是在保证全局底线召回率的同时实现起止点高物理分辨率聚集。
2. **零参数边界偏置 A-MoD (Boundary-Biased A-MoD, B-AMoD)**:
   - 复用 Dense 层自注意力分块列均值 $r_i$ 作为注意力接收度指标，并与 Scout 边界先验调制：$r_{\text{fused}} = r \cdot (1 + \alpha p_{\text{boundary}})$。
   - 奇偶交替调度 12 层 VideoMAE，A-MoD 层对 Top-50% 时空 Token 计算 MHSA+MLP，未选 Token 恒等残差旁路。理论分析等效 9 层计算量，设计目标为降低骨干计算负荷，真实加速与显存需经集群基准实测确认。
3. **连续时序尺度自适应 1D 卷积 (Continuous-Time Scale-Adaptive Conv1d, CT-Conv1d)**:
   - 局部物理采样步长 $\Delta t_i = t_{i+1} - t_i$ 动态调制 1D 卷积物理采样偏移，目标消除非均匀选帧引起的 FPN 感受野漂移与物理几何速度畸变。

## 待验证的科学假设与实验目标

- 检验双相预算分配能否在保持 $K=384$ 预算不变下提升高 tIoU（@0.7）定位质量；
- 检验 B-AMoD 在引入时空稀疏性时能否无损保持 VideoMAE 特征判别力；
- 检验 CT-Conv1d 是否能在连续时间轴上保护多尺度检测精度。
- 所有结论必须等待正式 200 视频训练集与 211 视频官方留出评估集的预注册实验终态结果。
