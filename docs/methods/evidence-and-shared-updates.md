---
updated: 2026-09-08
status: current-implementation-separated-from-hypotheses
scope: 当前 B/C 执行、支持语义、已有细化收益与新增假设
out_of_scope: 新模块实现声明、无损保证、重命名正在训练的模型
---

# B/C：不完整证据与共享更新

本说明与[A/公共价值定义](localization-value-routing.md)共同维护方法语义，不更改运行中的 M=b70ae056c495b43ca3f305fe438926b97b2723b5。近邻事实见[原文核对](../project/prior-art-bc-lite.md)，实验见[B/C/LITE 比较计划](../evaluation/bc-lite-comparison-plan.md)。

## B 的当前数据流

|步骤|当前操作与源码|
|---|---|
|状态与选择|Scout 提供完整低成本时间状态；前置生成一个选择计划。Dense PatchEmbed 仍支付全部输入 embedding 成本，见 [sparse.py](../../geosparse_ext/sparse.py#L229)。|
|Heavy|parent 内选中原生 token 执行冻结 VideoMAE，B 的 Heavy 中禁用源 TIA；不是只对某些原始 clips 完成全部前端编码。|
|Evidence|按 tubelet 与空间槽汇聚选中特征，保存锚点时间支持、parent、有效性及原图映射，见 [evidence.py](../../geosparse_ext/evidence.py#L6)。|
|Receiver|native384 cheap query 接收 evidence；已有五类 receiver、残差及 null，见 [receivers.py](../../geosparse_ext/receivers.py#L69)。|
|规则时间与检测|先在完整 native384 上执行 regular TIA，再插值到 query768，见 [model.py](../../geosparse_ext/model.py#L109)。旧 regular-TIA8 描述已不适用。|

B-full 是同一新 query/receiver 架构的全 Heavy evidence 控制；不是官方原版 AdaTAD。现行原图 ROI 几何记录不等于已实现按原图 ROI 主动获取和编码。

### 支持必须分层解释

1. **锚点与槽的直接成员**：空间池化槽对应的 tubelet、所选空间成员及其真实采样位置。当前 EvidenceBatch 的 support 主要表示这一锚点时间支持。
2. **完整直接观测集合**：当前 Heavy 调用实际读取的原生输入集合，可由选择计划恢复；不能把不连续区间的 hull 说成连续观察。
3. **编码依赖集合**：该槽经过 parent 内多层 self-attention 后可能依赖同 parent 的其他选中成员。它不等于槽自己的锚点支持，也不等于最终经过 receiver/TIA/head 的全部影响范围。

当前代码同时拥有 plan 和 parent 标识，但尚未将这三层语义分别完整导出为 evidence 字段。不能宣称已经完成来源可信度校准或独立 packet 缓存。固定锚点而改变同 parent 其他所选内容，仍可能改变该特征。缓存合法性须按实际依赖判断。

例如实际支持 [1.9,2.1]∪[5.9,6.1] 不代表 [1.9,6.1] 连续可见。远处证据可以提供语义上下文，cheap query 也拥有本地观察；不能据此规定没有 Heavy support 的位置都不可推断。

### 当前 support receiver 做到了哪里

[EvidenceAttention](../../geosparse_ext/receivers.py#L24)使用时间差、支持时长、fidelity 与 ROI 描述构造几何 bias，并以到实际支持并集的距离和半径限定候选；保留 null/零更新。它尚不是专门训练的“边界可信度概率校准器”。要证明更少的错误扩张或同类动作粘连，必须进行同 evidence、同元数据及合理容量的 receiver 对照。

局部 mask 在完整 Q×M attention score 构造之后应用，见 [receivers.py](../../geosparse_ext/receivers.py#L54)。因此当前几何半径不会自动减少 Q×M 的计算。这里 Q=384；slots=4 时过滤前 M 可达384×4=1536，不能假设 M≪Q。成本除 Heavy 外还包括证据投影、槽汇聚、receiver、dense TIA 和检测头。

## C 的当前数据流

当前 C 对每个原生空间2×2组使用1个 coarse 或4个 fine token；不合并时间轴。每层从该层完整状态重建 mixed tokens，但 fine 选择计划跨层复用，见 [sparse.py](../../geosparse_ext/sparse.py#L263)。**逐层重建不是逐层重新路由。**

对 coarse 组 g，在略去掩码和尺度编码的结构示意下：

    z_g = CoarseProject(mean_{i∈g} x_i)
    delta_g = H(z)_g − z_g
    x'_i = x_i + delta_g, i∈g

H 在同 parent 的整个 mixed token 集合上计算，因此 delta_g 不是只依赖 g 的独立函数。具体 coarse 构造、Heavy、回写见 [mixed_block](../../geosparse_ext/sparse.py#L96)；分桶实现沿用相同语义。Fine 路径使用对应更新，之后调用原生全窗口 TIA。all-fine 极限使用源路径。

同组共享本次 delta 时，x'_i−x'_j=x_i−x_j。这一代数性质属于共享残差更新，不是 C 独有；不保证后续网络等价、不保证全部信息无损，也不证明边界质量。

160px 下每个 native slice 有100个原生空间 token/25个2×2组。注册 mixed-token 比例 q 与 fine-group 比例 p 满足 p=(4q−1)/3；q=.5 对应 p=1/3。all-coarse 仍有25个 Heavy token/slice。q、fine 比例、有效 token 数、Heavy MAC 和总延迟分别报告。

## C 已有普通检测损失下的 refinement gain

不能把“coarse→fine 有符号收益”全部列为尚未实现。现行 [detector.py 的 acquisition probe](../../geosparse_ext/detector.py#L97)选取未选原子、强制加入计划并重新前向；C 在 [model.py](../../geosparse_ext/model.py#L92)把这些原子作为 fine groups。对 C，这已对应：

    u_refine(g | S) = L_TAD(S, g=coarse) − L_TAD(S, g=fine)

当前模型的有限在线 probe 保留符号并重算受影响路径。这是注册 acquisition 在 C 上的实际含义；它同时改变 Heavy 成本，并不是等成本替换。它也尚不包含新实例平衡区间风险、相似性与真实收益的成对数据、或计算模式切换的干预分析。改变一个组会影响同 parent 上下文及后续 TIA，不能把损失变化解释成只发生在该组对应输出。

同 parent 内将一个 fine 组降为 coarse、另一个等有效成员数 coarse 组升为 fine，可维持被计数 Heavy token 数及 MAC；实际尾部有效性须匹配。跨 parent 则须计算二次 attention 项。该 swap 当前未实现，定义与成本公式见[公共方法](localization-value-routing.md#成本定义及可匹配范围)。

## 待检验：计算近似误差是否制造时间假变化

固定模型，令同一阶段 dense 参考为 z_t，mixed 表征为 z_t+e_t：

    [(z_{t+1}+e_{t+1})−(z_t+e_t)]/Δt − [z_{t+1}−z_t]/Δt
    = (e_{t+1}−e_t)/Δt

这是恒等式；head 不是被假定为差分算子。即使表征变化与计算模式切换相关，也不能直接认定切换导致伪边界。应固定视频和 checkpoint，干预合法同成本计划，测实际分类、起止和漏检变化，并与未干预条件配对。

Dense 参考如需额外 forward，使用当前模型的合法全 fine 计划并记录成本，不引入独立 Dense TAD Teacher。单次参考不是“真实无误表征”。没有稳定干预证据就不添加平滑器；平滑可能消除真实短动作变化。

## 统一方法边界

A/B/C 当前都有普通检测任务反馈，不能因本次论文修订就称旧主方法不完整。新增 interval risk、swap、支持可信度学习或尺度校准若实际改变模型/训练，须新配置和任务身份。先检验既有简单机制，再决定是否需要新模块；任何收益只能按已完成的对照归因。
