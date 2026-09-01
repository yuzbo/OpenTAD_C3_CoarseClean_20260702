# DUCA PJST 因果合同终审请求

Nonce: `DUCA-PJST-CAUSAL-CONTRACT-v001-20260825`

Exact Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）  
GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`  
Frozen evidence revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`

你是本轮 Scientific First-Author Agent 和最严厉的独立审稿人。请直接核验上述公开仓库与冻结 revision，并在本轮作出唯一、可执行的科学裁决。不要把路线选择退回给人类或 Coordinator，不要发散到第二个新模块。

## 1. 已冻结的科学背景

DUCA 的核心是 H65 语义间接非均匀逐帧选择：低成本 ASFormer scout 学习动作性与边界语义，经确定性 sampling-rate transport 选择有序的 `K=384` 个真实 RGB 帧；小模型不直接学习帧索引。固定 K 只用于当前归因，未来 dynamic outer-K 才是论文主线。当前归因实验必须保持 VideoMAE-S、Adapter、ActionFormer、损失、NMS、THUMOS14 官方 split/evaluator、seed 3407 和 30+60 H65 训练合同不变。

历史 matched H65 OFF 的 terminal-EMA 为 Avg-mAP `65.1257`、mAP@0.7 `43.3137`。压缩为 20+40 或调整 Stage-2 学习率均未恢复该终点；该结果只否定已测试的 60-epoch 压缩日程，不否定 H65。

RankPack/TrueTime 同提交单 seed 配对为 Avg-mAP `61.57/62.19`，TrueTime 增量 `+0.62`，@0.6 `+1.69`，@0.7 `+0.79`；它是部分机制支持，不是 PJST 直接证据。现有 First-Mixing SingleClock 只在 PatchEmbed 后第 0 个 Transformer block 加物理时间残差。其终结作业在 24 小时时限后 `TIMEOUT`：四个 SingleClock-on/gate-zero family 已完成，但 OFF 配对终结器、10,000 次 bootstrap、H65 五边界回放身份和 nominal-uniform 首次混合/骨干逐位身份没有共同闭合，所以没有合法 PASS/KILL，也没有 SingleClock 论文效能结论。

## 2. 候选机制：PJST-v0.2

候选名：`DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v0.2`（PJST）。它只修复 VideoMAE 最早的 temporal kernel=2 PatchEmbed 混合，不增加可学习参数：

```text
原 tubelet: z = W^- * x_i + W^+ * x_{i+1} + b

m = (q_i*x_i + q_{i+1}*x_{i+1}) / (q_i + q_{i+1})
v = canonical_delta / (2*actual_delta) * (x_{i+1} - x_i)

PJST: z = (W^- + W^+) * m + (W^+ - W^-) * v + b
```

其中物理中心、Voronoi 支撑和 `delta` 必须先在全局 K384 上形成，再按原 rank 切成 `24×16`；不能在每个 16 帧容器内重建局部时间。exact canonical-uniform 输入在任何浮点支撑计算前直接旁路原 PatchEmbed，要求 forward、input gradient、参数对象和 optimizer membership 严格身份一致。非均匀路径保持 selected RGB、rank 顺序、token 数与重型计算量不变。proposal 必须在 filtering/top-k/IoU/NMS 前恰好一次进入 physical coordinate，禁止后处理二次映射。

当前核验只把 PJST 定义为 `designed_candidate`：它是归一化支撑加权外观与物理间隔差分，不是严格时间积分；constant-content 下没有纯 gap 加性捷径，但一般内容下网络仍可利用 gap；它只修复 pair 内首次混合，不能声称完成整个骨干的连续时间建模。TDN、TAdaConv、Run-Length Tokenization、ToMe 与 TE-TAD 构成明确 prior-art 压力。

## 3. 必须裁决的唯一科学分叉

H65 Stage-2 中 selector/scout、贡献蒸馏和 detector loss 共同训练。PJST 改变首次重型表示后，反向梯度可能改变 selector，随后 selected positions/RGB 也会变化。因此以下两种因果口径不能混称：

### A. 系统级端到端总效应

- 从同一 30-epoch Stage-1 checkpoint、同一 seed 执行一条 PJST-ON 60-epoch/6000-successful-update Stage-2；
- 允许 selector 因 PJST 梯度而漂移，但 selector 算法、监督、K384 和全部训练合同不变；
- 与既有 matched H65 OFF terminal-EMA 比较；
- 报告 selected-position Jaccard、边界覆盖、gap 分布、duplicate rate 和 selector 参数位移，把选择变化视为中介变量；
- 允许的 claim 只能是 PJST 对完整 H65 系统的总效应，不能声称同一 selected RGB 的纯表示归因。

### B. 冻结/重放 selector 的纯表示归因

- 冻结或确定性重放 selector/selected positions，使两臂看到相同 RGB；
- 这改变原始 H65 联合训练合同；若不能合法使用既有 OFF 作为对照，必须明确是否需要同时训练一个 matched frozen-selector OFF；
- 允许的 claim 只能是首次 tubelet 表示效应，不能直接外推到完整联合 H65。

请在 A、B 或 `STOP/PIVOT` 中冻结唯一正式口径。若选择 B，必须明确额外 OFF 训练是否不可避免；不能一边复用端到端 OFF，一边声称相同 selected RGB。若认为存在严格更好的单一 H65-compatible 口径，可 `REVISE`，但不得引入 Query、Bridge、dynamic-K、continuous cliplet、新 selector 或新的 detector。

## 4. 你必须返回的终稿

第一行给出且只给出一个：`CONTINUE / REVISE / PIVOT / STOP`。随后完整冻结：

1. **唯一因果 estimand 与论文表述**：系统总效应或纯表示效应，哪些量是干预、结果与中介变量。
2. **精确 PJST 合同**：张量 shape/dtype/device，global-K384 support，短视频/padding/duplicate/mixed-batch 处理，canonical-uniform identity，非均匀公式、数值稳定和 exactly-once physical decode。
3. **最小 Builder patch**：允许的文件、符号、禁止修改面。优先限制为：
   - `opentad/models/utils/temporal_grid.py`
   - `opentad/models/backbones/backbone_wrapper.py`
   - `opentad/models/backbones/vit_adapter.py`
   - 单一配置、validator、focused tests、launcher
   不得修改 selector/ASFormer、ActionFormer 训练语义、loss、NMS/evaluator、dynamic-K、Query/UVT/cycle 或训练日程。
4. **可区分实现测试**：shape/layout、uniform byte identity、显式代数参考、constant-content geometry invariance、gap 倍增的一阶幅度、global support partition、padding、执行 K384、无新参数/optimizer group、有限梯度、生产调用轨迹和 pre-NMS 单次物理解码。
5. **最便宜兼容性 falsifier**：说明是否允许在既有 checkpoint 上同输入只读前向诊断、它能/不能支持什么结论，以及触发停止的错误。
6. **唯一正式真实视频实验**：是否复用 Stage-1，新增几条训练、60 epoch/6000 updates、每 5 epoch 完整恢复 checkpoint、latest-3+milestone+final、terminal final/final-EMA、THUMOS14 官方 evaluator、seed、N16R4 资源、结果根和 full-stack cost。
7. **预先冻结的指标与停止规则**：Avg-mAP、@0.6、@0.7、短动作/相邻动作/gap strata、selection mediation、10,000 次整视频 paired cluster bootstrap 的作用和局限、明确的通过/终止阈值。不得把单 seed 视频 bootstrap 写成训练 seed 稳健性。
8. **既有 SingleClock TIMEOUT 的处置**：是否必须先补齐其缺失证据，还是可作为封存负向预警而不阻塞 PJST；禁止重复 SingleClock 训练。
9. **prior-art invalidator 与最窄新颖性 claim**。
10. **下游交接**：`next_owner / next_action / dependency / expected_return_at / single_recovery`。默认顺序应为 clean Builder 最小实现 → 独立 Critic → Evaluator PRE_RUN → 只有准入后才运行唯一正式实验。

## 5. 硬边界

- 不重复 dense、uniform、random、RankPack/TrueTime、UVT、Fovea 或 H65 OFF 训练，除非你选择 B 且证明 matched frozen-selector OFF 在因果上不可省略。
- 不引入 dynamic-K；K=384 是当前表示归因门。
- 不用本地 CPU、synthetic、subset 或早期 checkpoint 形成效能结论。
- 不使用 official validation/test GT、teacher、cache 或结果做 selector 决策。
- 不把当前设计、测试、TIMEOUT 或跨运行根算术差写成 mAP 改善、成本收益或论文支持。
- 不能要求人类再选择路线；你必须给出唯一可执行裁决。

