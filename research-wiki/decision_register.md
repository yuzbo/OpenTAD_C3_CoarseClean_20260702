# 决策台账

每条决策包含“为什么选/为什么否/何时允许恢复”。后续不得只凭印象推翻已记录决策。

| ID | 决策 | 状态 | 选择或否定理由 | 恢复条件 / 证据 |
| --- | --- | --- | --- | --- |
| DR-001 | PAction/GAS-VT 不再是最终目标 | 已锁定 | 它们证明低成本动作信号可用于稀疏输入，但 action coverage 与高-IoU detector utility 不等价；GAS-VT 还有 train/apply budget shift、非真正 sequential、repair 近似 uniform 等问题 | 仅保留为 baseline、诊断和失败机制证据 |
| DR-002 | 三阶段独立 checkpoint/ledger pipeline 不能作为最终模型 | 已锁定 | 多阶段 teacher export、selector train、detector train 容易 stale，训练/推理不同构，也不符合用户要求的优雅协同训练 | 仅作归因/上界；最终模型若含 selector 必须同一 forward 和联合梯度 |
| DR-003 | fixed K 是安全锚点，不等于最终动态预算 | 已锁定 | fixed K 可公平归因；dynamic K 若只有 padded cap、soft expected K 或 budget sweep，不是动态计算 | 只有实际 backbone kernel/length/latency 随 K 变化且优于 fixed Pareto 才恢复主张 |
| DR-004 | actionness 只能辅助 selector，边界/状态转换优先 | 已锁定 | move 与 DUCA 现象表明动作内部覆盖可能很好但边界偏移；二分类标签天然粗糙 | selector 路线只在显式 start/end、transition、hard-policy utility 和 detector gradient 一致时恢复 |
| DR-005 | selector 必须看到 coarse hidden features | 已锁定 | 只输入 `p_action`、delta 和不确定性曲线会把模型退化为复杂 top-k；隐藏特征含语义与局部状态信息 | 适用于任何未来 selector baseline |
| DR-006 | detector loss 必须真实影响 selector | 已锁定 | one-step 非零梯度只证明可导，不证明 hard 决策方向正确；soft bridge 与 hard policy 必须同构 | 需要 hard one-swap finite difference、参数移动、选点变化和 full mAP |
| DR-007 | max-gap 是覆盖安全约束，不是核心创新 | 已锁定 | hard repair 能保证可行，但可能把 learned policy 变成 uniform scaffold；soft gap loss 不保证硬几何 | 必须报告 repair 次数、前后 gap、no-gap/soft/hard 消融 |
| DR-008 | X3D 和 SlowFast Fast 不作主 pre-backbone 模块 | 已锁定 | dense frozen video prior 运行过慢，可能吞没后续 backbone 节省；Kinetics 类别重叠和外部预训练也扩大审稿面 | 只作 appendix frozen prior/上界；必须全栈计时和 class-overlap audit |
| DR-009 | “online DUCA”术语被纠正 | 已锁定 | 项目是离线 TAD；当时的 online 只是同一 forward 内 runtime-generated、cache-free、full-window，不是 streaming/prefix-causal | 统一使用 offline、in-forward、full-window、cache-free |
| DR-010 | DUCA 主线 pivot | 已锁定 | 长期迭代发现核心仍是复杂 score + top-k + scaffold，资源决策空间过窄，真实定位 regret 和全栈成本未被直接优化 | DUCA 代码保留为 baseline、contract 和测试资产，不继续堆权重/loss/prior |
| DR-011 | ChronoTransport/DCRT 不作为当前主线 | 用户否决 | 虽然 Pro 推荐 time x layer 的 recompute/transport/reuse，但它接近 MoD/feature reuse，层级动作僵硬、系统工程面大、创新归因和三个月闭环风险高；不符合用户希望的新 TAD 检测方法 | 只有 profiling 证明 feature recompute 是绝对主瓶颈，且 risk-certified transport 明显胜 periodic refresh，才可作为独立未来项目 |
| DR-012 | CoDeR-TAL 与 ACTAL 暂不进入主线 | 已锁定 | CoDeR 依赖 codec/partial decode 和硬件；ACTAL 是 streaming 新任务，均偏离当前离线 TAD 问题 | 只在解码瓶颈或在线任务被明确选择时恢复 |
| DR-013 | 选择独立 PhysTime-TAL/TAD 方向 | 已锁定 | 它直接解决所有不规则采样都会遇到的 selected-rank 几何错误，输出仍是完整 TAD 区间，不是插件工程叠加 | 必须处理 mTAN、TE-TAD、robustness benchmark、LiquidTAD 的新颖性碰撞 |
| DR-014 | PhysTime-TAL 1.0 不直接实现为论文最终版 | 已锁定 | continuous embedding、reference query、双视图一致性都已有强邻居；support width、固定 M、hazard 语义和真实缺失定义不严谨 | 被 PhysTime-TAD 2.0 support-integrated measure operator 取代 |
| DR-015 | 秒是规范时间坐标 | 已锁定 | 归一化时间会丢失物理尺度；selected rank 会改变距离语义；秒坐标可稳定 assignment、decode、NMS、跨 FPS | 允许 `round(t*fps)` 导出原帧号，不允许映射到 selected rank |
| DR-016 | feature-token PhysTime pilot 取消 | 已锁定 | 下载 I3D 特征并训练只能验证 feature geometry，不能证明原始帧稀疏计算和真实 AdaTAD 端到端有效性 | 代码仅保留算子与单元测试；不得作为主实验数字 |
| DR-017 | PhysTime-AdaTAD 1.0 先做 head isolation | 当前执行 | 相同无学习、无 GT 的不规则采样只比较检测头，能把收益归因给物理时间几何，避免又把 selector 混进来 | Phase 1 完成后再考虑 learned selector 或 sampling robustness |
| DR-018 | primary comparison 不加 paired consistency | 当前执行 | consistency 会给 PhysTime 额外监督，破坏三头公平性；先证明 architecture 本身 | 仅 Phase 2 作为消融/鲁棒训练扩展 |
| DR-019 | 所有老 commit/job 均按证据类别管理 | 已锁定 | 不同 commit、config、预训练和修复状态的 mAP 不可混为论文主证据 | 只有同 commit、同 selected indices、同 schedule 的 matched runs 可进主表 |
| DR-020 | 全栈成本必须计入 | 已锁定 | selected count 或 backbone FLOPs 会遗漏 decode、preprocess、scout、H2D、padding、cache/ledger generation | 报告 p50/p95 latency、显存、吞吐、decode 与 backbone 分解 |

## 不得静默推翻

恢复任何已否决路线前，必须在本表新增一条 superseding decision，写明新证据、对应 commit、实验 ID 和为什么旧否定理由已经失效。禁止直接改写历史条目。
