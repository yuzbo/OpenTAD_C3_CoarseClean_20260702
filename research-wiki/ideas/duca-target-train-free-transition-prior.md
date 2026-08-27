# DUCA 免目标域训练的冻结转变先验

## 状态

`implemented / tested / experiment_running / not_empirically_supported`

当前“免训练”只限定 **pre-backbone 选择前端**：冻结外部预训练编码器、无目标域标签、
无目标域微调、无测试时优化、无参数证据融合和固定分配器。后接 AdaTAD/ActionFormer 仍在
THUMOS training split 上训练；因此禁止把完整 TAD 系统称为 training-free。

## 精确定义

本候选只主张“目标数据集免训练”：不使用 THUMOS 等目标数据集标签，不在目标训练集微调，
推理时不做梯度更新，也不依据目标 validation/test mAP 调提示词、融合权重或阈值。允许使用在
外部数据上预训练后完全冻结的视觉/视觉语言编码器。一个真正从未训练过的模型不能提供可靠
语义动作证据，因此不得把“冻结外部预训练”写成“模型无训练”。

## 最小模型

1. 在低分辨率、低频时间锚点上运行冻结编码器。MobileNetV3 是低成本候选；SlowFast-R50
   **Fast pathway only** 只作高成本冻结视频先验诊断，Slow pathway 与 lateral fusion 均不执行。
   Fast-only 的完整成本必须进入 Pareto，不能包装成轻量主方法。
2. 将多维冻结特征线性重建到完整候选时间网格；重建特征直接作为证据，与现有稀疏 probe
   合同一致，不额外输入 anchor mask 或距 anchor 距离。
3. 构造三类目标域无参数证据：通用动作/背景文本相似度、相邻特征余弦变化或分布变化、
   语义不确定性。优先把“变化”作为间接边界证据，把动作性只作为低权重辅助。
4. 每个视频内使用固定的稳健秩/分位数归一化和事先冻结的等权或几何均值融合，随后做固定
   peak-NMS；禁止在目标数据上学习融合器。
5. 复用当前 exact-K/max-hole 与 R2Q3 边界微簇分配：在转变中心两侧聚集少量帧，同时保留
   全局上下文。冻结编码器和无参数 evidence 不改变后接 official-derived AdaTAD/ActionFormer。

## 两种必须区分的口径

- 严格免优化模式：冻结外部编码器 + 无参数证据融合 + 固定分配器；目标训练和测试均无梯度。
- 跨数据集通用模式：可以在源数据集训练 selector，再完全冻结到目标数据集；它是
  target-train-free，不是 training-free。两者不得合并报告。

## 决定性实验

P0 诊断固定比较低层变化、冻结语义动作性、冻结特征变化、语义+变化+不确定性融合；报告
action AP/AUC 仅作诊断，并报告边界支持、端点距离、短动作端点召回和完整 probe 成本。

完整 TAD 固定同 K/G、同 detector、同训练与评估协议，比较 exact-uniform、变化-only、
通用动作性 top-K、冻结转变融合+R2Q3、可训练 DUCA。主结论只能使用完整 validation、terminal
EMA、OpenTAD tIoU 0.3--0.7 mAP 和端到端成本。广泛即插即用主张至少还需第二数据集完全不
调参迁移。

## GO/KILL

- GO：同预算且计入冻结 encoder 全成本后，免训练模式稳定优于 exact-uniform，最好接近可训练
  DUCA，并在第二数据集无需调参保持方向一致。
- KILL/降级 baseline：只提高粗分类或边界 proxy，却不提高官方 TAD mAP；或 encoder 成本抵消
  重 backbone 节省；或必须用目标 mAP 选择 prompt/权重/阈值才能成立。

## 研究边界

T3AL 类方法虽不使用目标标签训练，却会对每个测试视频更新投影器，不能归入本项目的严格
免优化模式。冻结 VLM 可以提供零样本语义，但“动作/背景”二分类并非天然稳定的开放词汇任务；
因此最可信的核心仍是冻结特征的状态变化，语义动作性只作辅助。该候选当前只适合作为独立
baseline/模式，除非跨数据集官方 mAP 与总成本证据足以支持论文主线转向。

## 2026-07-23 实现与运行证据

- 分支：`codex/duca-t1-trainfree-20260723`；当前合同修复提交
  `4c5604b4a0abde9e59f625d519934e855bfe1519`。
- Linux focused：`29 passed in 43.19s`，并通过 py_compile、shell syntax 与 clean-tree。
- MobileNetV3-Small 使用 ImageNet-1K 官方冻结权重，SHA-256
  `047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f`。
- SlowFast 使用 Kinetics-400 官方 SlowFast-R50 权重，但只执行 Fast pathway；真实 CUDA
  preflight 为 `pathway=fast_only`、`slow_path_executed=false`、
  `lateral_fusion_executed=false`、`hidden_shape=[1,16,256]`。权重 SHA-256 为
  `454f39e1c1f985df2bee2aa27887ed53ff56e74ed8b8cca11203a1a1264d7cc2`。
- Fast-only 正式 Job `1180653` 已完成 epoch 0 并进入 epoch 1；MobileNet 三臂最终重试 Job
  `1180654` 已无依赖提交，当前等待账户 GPU 配额。当前没有 terminal epoch-59 EMA mAP。
- Jobs `1180639/1180644/1180652` 分别记录首次权重下载、旧 official-60 合同和计算节点
  MobileNet 下载失败，均在 optimizer update 前退出，不是模型性能负结果。

## 2026-07-27 terminal evidence and plug-and-play boundary

后续正式结果已形成。Fast-only 为 `63.5297%`，相对同协议 exact-uniform
`64.49%` 为 `-0.9603pp`；MobileNet 特征变化、语义和固定融合分别为
`63.27%/62.78%/64.33%`；T1 actual-time residual 为 `64.0200%`。因此现有
target-train-free 前端没有超过均匀采样，状态应解释为
`implemented / tested / negative terminal evidence`，不能再写成“尚无终端结果”。

这些实验仍然训练了后续 THUMOS detector，所以只证明“目标域免训练选择前端”，没有证明
真正 frozen-detector 即插即用。后者要求 selector 与已训练 detector 都不做目标域重训，
目前尚未验证。下一版免训练模式应复用主方法的有界密度解码器，并优先使用低成本帧差、
边缘/压缩域变化或稀疏冻结图像特征；高成本 SlowFast 只保留为负面对照。

统一口径见
`research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`。
