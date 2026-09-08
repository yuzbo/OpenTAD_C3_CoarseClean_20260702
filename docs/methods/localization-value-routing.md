---
updated: 2026-09-08
status: current-implementation-plus-proposed-extension
scope: Route A 与 MoD 的共同执行定义及定位价值增强的语义
out_of_scope: 声称增强已经实现、冻结未指定超参数、替代实际源码
---

# 定位价值路由：方法与实现边界

## 已核对的当前 A

现行 M=b70ae056c495b43ca3f305fe438926b97b2723b5：

|机制|源码依据与含义|
|---|---|
|前置低分辨率 Scout|[model.py](../../geosparse_ext/model.py#L61) 从原视频产生 gain、预算与 baseline。|
|一次规划、跨层复用|[model.py](../../geosparse_ext/model.py#L77) 生成一个 plan；[sparse.py](../../geosparse_ext/sparse.py#L256) 各层复用 selection。|
|真实稀疏 Heavy|[sparse.py](../../geosparse_ext/sparse.py#L45) 在原 parent 内 gather 后计算再回写，保留完整状态。|
|原生全窗口 TIA|A 在回写后调用源 Adapter；源 Adapter 沿 native384 重排并做时间卷积，不因 Heavy 的 48 个 parent 分组而变成 48 个独立 TIA。|
|PG 与 acquisition|[detector.py](../../geosparse_ext/detector.py#L75) 和 [routing.py](../../geosparse_ext/routing.py#L233) 使用任务成本反馈，反事实增加未选原子并重新编码。标签是标量检测 loss 的 signed difference。|
|动态预算|改变此次选择量；不等于逐层重新选择、逐层动态深度或新的区间风险目标。|

当前没有等成本 swap 标签、实例平衡 interval risk 或 M0–M3 的实现证据。使用原检测 loss 的边界项不等于已经实现本文建议的新定位风险校准。

## 公共执行骨架

令 H_l 包含原层 Heavy 的归一化、attention、MLP 及其残差，定义纯增量 ΔH_l(z)=H_l(z)−z，避免将已有 residual 再加一次。用 P 选择原生位置：

    X̃_l = X_l + P_Sl^T D_l ΔH_l(P_Sl X_l)
    X_(l+1) = T_l(X̃_l)

T_l 表示源 block 规定位置的规则时间更新。此式是结构抽象，不授权改动 AdaTAD 的实际算子顺序。Heavy attention 保持 parent 内；TIA 保持完整 native384 时间轴上的源卷积。当前 A 的选中增量没有 MoD 风格分数缩放，D_l=I；原风格 MoD 与统一缩放控制须分列。

~~~mermaid
flowchart LR
    X["完整原生状态"] --> S["路由选择"]
    S --> H["parent 内 Heavy 更新"]
    X --> R["保留原状态"]
    H --> W["原位置回写"]
    R --> W
    W --> T["native384 TIA"]
    T --> N["下一层完整状态"]
~~~

MoD 的逐层路由与 A 的前置规划都能使用这个公共基础。A 的未选位置仍受轻路径影响，但本次固定计划不会在更深层新获得 Heavy 资格；MoD 可以逐层改变资格。二者的信息量、规划成本和适应能力属于实验问题。

## 拟新增：定义实际操作和有符号价值

令 R(S) 为固定视频、标签及当前模型下执行计划 S 的检测风险。

    acquisition(a | S) = R(S) − R(S ∪ {a})
    retention(b | S)   = R(S \ {b}) − R(S)
    swap(a ← b | S)   = R(S) − R((S \ {b}) ∪ {a})

这里 retention 的正值表示保留 b 有用；不同定义的方向必须写明。三种标签均允许为负，互不混名。尤其 acquisition(a|S)−retention(b|S) 一般不等于 swap；交互依赖改变了条件集合。

固定预算的 swap 监督须定义候选 a、被移走的 b 以及已有集合 S。单独的候选分数并不天然知道移走谁。最初可通过同 parent、相同实际有效 token 数的替换控制 MAC；供预测器使用的 donor/集合摘要及动作采样规则仍待实现规格冻结。

动态预算使用 acquisition 与真实 ΔC，目标可考察 u_add−λΔC；这不自动构成最优子集保证。新增监督是否采用此成本修正、λ 如何与当前 dual 一致，须在新配置中明确，不能事后给旧 gain 换名。

**操作作用的层必须显式记录。** A 的一个原子可以表示跨全部 Heavy 层的一次计划修改；逐层 MoD 的原子可以表示某一层的修改。两者标签与 probe 成本不同，不能复制 A 的跨层标签后声称已校准 MoD 的单层决策。若 MoD 后续 Router 重新决策，要注明是允许下游策略响应，还是冻结其计划的受控干预；二者不可混合。

## 拟新增：区间风险

第一控制始终为 R=L_TAD。实例平衡候选为：

    R_interval = mean_j(L_cls,j + β L_boundary,j) + γ L_background

这只是设计定义，不是已冻结的新 loss。开始新训练前须落实：正样本到实例的分配、实例内归约、β/γ、边界归一化尺度、空 GT 时仅背景项、没有成功 proposal 的 GT 如何仍贡献训练风险。不得只对成功匹配预测计算风险，再宣称保护漏检。所有方法获得同一风险定义；不能只给 A 更有利的目标。

在配对反事实中保持目标分配规则、归约与 loss-normalizer 状态一致，防止标签包含归一化状态变化。GT 仅用于训练与正式评价；Scout 的推理输入不得包含 GT、动作真实边界或事后收益。

## 重计算及噪声

改变较早选择会改变后续 attention、TIA 与检测输出，因此重新计算完整受影响路径。当前 M 的 acquisition 重新编码是这一原则的已有实现；它不是 swap 或逐位置效应图的实现。

额外 probe 使用当前模型、配对输入与状态；不引入独立 Dense TAD Teacher。记录新增 Heavy forward 和训练时间。恢复 RNG 起点不保证不同 selected 张量形状获得逐原生位置相同的随机掩码：图中用于机制解释的干预应明确采用确定性 eval 状态，或实现并验证按原生位置配对的随机性。不能将训练态一次噪声差异直接称为某个位置的因果贡献。

## 成本定义及可匹配范围

令 k_(l,p) 为第 l 层、第 p 个 parent 的实际有效 Heavy token 数，宽度 d_l，MLP ratio=4：

    C_heavy ≈ Σ_l Σ_p [12 k_(l,p) d_l² + 2 k_(l,p)² d_l]

这是 QKV/output、MLP 和两次 attention 矩阵乘的 MAC 计数，忽略小项；不含 Scout、TIA、embedding、packing、解码与检测头，不是完整 FLOPs 或实测时间。

同一层在 parent p 加入 s 个有效 token：

    ΔC = 12 s d² + 2d(2 k_p s + s²)

多层动作对全部受影响层求和。相同全局 K 只固定线性项，Σ k_p² 仍可变化。同 parent、等有效大小的 swap 保持此计数，但不保证硬件耗时逐次相等；packing、布局及内核会影响时延。尾部有效性、未重叠原子和原生 parent 容量必须按实际计划处理。

例如 d=768 的单层，两种合法 parent 分配 (100,100) 与 (200,0) 都有 K=200，Heavy MAC 分别为 1,446,297,600 和 1,477,017,600。这个差异是公式算例，不是训练或硬件结果。

交替全量/稀疏层的 MoD 成本包含全量层；其可达到的预算有下限，不能给它一个不可达到的低成本点并判输。固定 token、固定 Heavy MAC、固定总延迟是三个不同控制。

## 定位假设的数学边界

若等长区间整体平移 e，且 |e|<d，则 tIoU=(d−|e|)/(d+|e|)，达到 θ 需 |e|≤d(1−θ)/(1+θ)。该算例说明短区间容许的绝对平移更小，不证明所有真实边界附近都值得增加计算，也不证明 A 已保护短动作。

计算位置与受益位置可不同。U_(a→q)=ell_q(S)−ell_q(S∪{a}) 是固定模型和计划下的干预差分，不能用 attention 权重替代，不能不注明条件集合就推广为某一帧永久不变的重要性。
