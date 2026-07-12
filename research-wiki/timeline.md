# 研究路线时间线

## 2026-07-02：C3 粗分类与严格 ledger 起点

- 初心：低成本粗二分类模型先产生逐时刻动作/背景概率 `p_action`，再间接选择对 TAD 有用的观测，减少 VideoMAE/AdaTAD 重计算。
- 设计底线：粗分类以动作性二分类 GT 监督；选帧不能只做 actionness top-k，而要保护状态转换和动作边界；val/test 不读 GT、teacher、oracle 或 detector cache。
- 固定 384/768 被定义为归因和安全锚点，不是最终动态预算贡献。

## 2026-07-05 至 07-07：PAction、GAS-VT 与 detector utility

- PAction learned fixed384 成为较强 Stage1 工作基线；它说明简单 p_action 衍生信号可保留有用证据，但不是最终 detector-aware 方法。
- GAS-VT 尝试 value transport、gap urgency、boundary bracket、action interior、CVaR hole 与 hard repair。
- 诊断发现：GAS-VT 早期 mAP 可以由半密度输入和 p_action 解释，后期 plateau 与 detector utility 错配、train/apply budget shift、非 sequential decode、过强覆盖约束有关。
- 方向转为 DUCA：train-only dense detector responsibility/utility -> deployable selector -> sparse AdaTAD。
- 同时建立严格 no-leak、selected position、original-time、预算、provenance 和 geometry contracts。

## 2026-07-07 至 07-08：lattice、move 系列与几何分析

- lattice replacement 从 uniform scaffold 局部替换，因而只能是诊断，不得包装成无 scaffold 智能采样。
- move25/move50、膨胀与 learned radius 显示选点确实聚集，但聚集位置偏移；原因候选包括粗二分类标签过粗、probe stride/smoothing 时延、score 与 hard decode 不同构、radius 扩散正分数、repair 改写策略。
- 明确 `boundary_support@r` 不能证明边界聚集，更不能证明对 detector 的因果贡献。
- 要求 action-instance 级 geometry -> best proposal tIoU -> high-IoU mAP 证据链。

## 2026-07-08：插件目标与 train-free prior

- 论文目标一度被定义为 detector 前的 DUCA acquisition plugin，而不是新 detector。
- 反复否定三阶段离线 export/checkpoint 拼接，要求同一 forward 内粗分类、selector 与 detector 协同训练。
- X3D frozen prior 被设计为 train-free baseline；实际 grid/export 极慢，且 dense X3D 计算可能吞没后续节省。
- SlowFast Fast 侧被讨论为更强运动先验，但它仍依赖 Kinetics 预训练、类别重叠审计和密集视频推理，最终只保留为 appendix 诊断。

## 2026-07-09：DUCA-JCT、MUST 与最终 selector contract

- DUCA-JCT 尝试单次训练作业中的渐进协同：早期 actionness 辅助、detector 从头训练、随后逐步打开 detector-to-selector bridge。
- selector 被要求观察 coarse hidden features，并从 actionness 主导改为 transition/start/end/boundary utility 优先。
- fixed384 与 dynamic MUST 统一 acquisition policy；fixed 是 K=384 特例。
- MUST 暴露 expected K、hard K、unique K、padded K、实际 backbone K 不一致和 64/384 跳变；动态计算主张被降级。
- 修复 max-gap scaffold：先保证硬可行骨架，再用剩余预算进行边界优先选择。
- 正式 one-step gate 要求 official ActionFormerHead、完整 optimizer coverage，以及 probe/selector/budget controller 非零梯度。

## 2026-07-10：DUCA 严厉审计与 full-window 澄清

- 发现 leaf losses、aggregate 和 alias 重复聚合；boundary proxy 实际仍偏 action body；soft-resample 与 hard center/radius policy 不同构。
- 纠正“online”：项目始终是离线 TAD。所谓 online probe 只是同一 forward 中即时产生、无 JSONL/ledger/cache，不是 streaming 或 causal。
- CFPA/structured exact-K/max-gap 尝试统一 hard Viterbi 与 soft relaxation，修复 uint8 bridge、DDP static graph、坐标 round trip 和 loss multiplicity。
- 即便工程合同明显变强，DUCA 仍面临复杂 score+top-k+scaffold、全栈成本不清、动态计算未实现和创新碰撞问题。

## 2026-07-10：PIVOT 发散与 ChronoTransport 建议

- Pro 审计判定：重要问题是长视频真实计算与高-IoU 风险，但把它压缩成 pre-backbone frame subset selection 过窄。
- 产生 23 个候选方向，幸存者为 ChronoTransport/DCRT、CoDeR-TAL、ACTAL、PhysTime-TAL 和 No-Free-Frames。
- Pro 首推 ChronoTransport：time x layer 选择 recompute/transport/reuse，以 counterfactual localization regret 约束。
- ChronoTransport 并非只停在 idea：另一 local worktree 已完成 Stage-A runtime、OpenTAD replay 和 formal Stage-B fit/calibration/evaluation，方法提交链为 `6e4bc54..92029ea`。
- 正式 seed-3407 的 P3 science gate 为 FAIL：risk-regret 排序为负、cell-risk/window-target 尺度错配、feature transport 改善不稳定；Stage C 与 P5 因此未解锁。
- 该本地分支比 origin 多 15 个提交，只能称为“工程闭环已形成、科学 gate 已失败”，不能误记为未实现或已验证。
- 用户否定该主线：它接近更复杂的 MoD/feature reuse，层级动作与 token/tubelet 单位僵硬，系统工程和创新归因风险过高。

## 2026-07-10：DUCA 全栈成本与结构审计分支

- commit `a5e1774` 在另一 DUCA 分支加入 full-stack profiler、官方 OpenTAD source-parity audit 和第二轮 ResearchClaw 24-idea 发散审计。
- 结构审计确认：部分 AdaTAD base config/ActionFormerHead 可保持一致，但 single-stage wrapper、ActionFormer、anchor-free head 和 ViT adapter 已为 selected-axis/selector 路线扩展；论文只能称 official-derived components，不能笼统称“完全未修改 official AdaTAD”。
- 该轮提出 CVCR-TAD、CoDeTAD、BCFT、continuous-time physical head 和 full compute ledger 等候选，但未直接替换 DUCA；当时先要求完成 full-stack trace、hard one-swap utility audit 与 same-selected-frames geometry 对照。
- 这些审计实验没有形成完整论文闭环，后续 PhysTime 转向吸收了 physical-time geometry 问题，而非把所有 DUCA 主张视为已证实。

## 2026-07-10：选择 PhysTime-TAL/TAD

- 用户明确选择“独立的新 TAD 检测方法”：输入任意不规则观测与真实时间戳，直接在物理时间上分类和定位。
- 新颖性审查指出 mTAN、TE-TAD、Temporal Robustness Benchmark 和 LiquidTAD 已占据宽泛 continuous/irregular/robustness 叙事。
- PhysTime-TAL 1.0 因 support width、归一化时间、固定 query 数、hazard 定义和 paired consistency 过宽而被 HOLD。
- PhysTime-TAD 2.0 将核心收敛为 support-integrated measure attention、global seconds query pyramid 和 physical-time head。

## 2026-07-10 至 07-11：feature track 取消，转向 raw-video PhysTime-AdaTAD

- 首版 PhysTime-TAD 2.0 代码在 I3D feature-token 轨道上完成算子和 CUDA gate，并部署特征下载/训练队列。
- 用户否定下载预提取特征作为主证据，要求直接验证 AdaTAD 端到端 raw-video 稀疏头。
- 取消 feature-token 正式 jobs；保留代码作为几何算子单测资产。
- 冻结 PhysTime-AdaTAD 1.0：同一无学习、无 GT 不规则 K=384 采样，只比较 selected-axis、physical-grid 和 PhysTime 三种头。
- 明确秒坐标可以映射回原视频帧号用于展示，但不能映射到 selected-rank 轴。

## 2026-07-11：raw-video 实现阶段

- `PhysTime-TAD 2.0` feature-geometry 核心已实现。
- `PhysTime-AdaTAD 1.0` raw-video 集成形成规格与 implementation plan，随后进入真实 gate 与正式训练。

## 2026-07-12：首个 full run 与性能下降诊断

- 经过 evaluator、masked attention 和 AMP 稳定性修复，`3ac93a1` 真实 gate、两 epoch stability gate 和三头 K384 full run 全部完成。
- 最佳 checkpoint 只读重放复现正式 mAP；PhysTime 1.0 未胜 selected-axis 或 physical-grid，首个方法结果为负。
- 诊断排除 evaluator、训练崩溃、重复秒转换和缺失 test windows；确认当前比较同时改变容量/上下文，且存在 absolute-second query 主导、粗层 attention 坍缩、候选密度和短动作监督不足。
- PhysTime 1.0 冻结为负基线。下一步不扩多 seed/第二数据集，先构建 capacity/context/candidate-matched physical-time 因果对照。
