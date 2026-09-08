---
updated: 2026-09-08
status: primary-sources-checked-claims-constrained
scope: B/C 与 LITE 的机制重叠、任务边界和待证增量
out_of_scope: 全领域新颖性证明、迁移性能结论、宣布对照已经实现
---

# B/C 与 LITE：已核对的近邻

本次核对截至 2026-09-08。以下判断来自列出的原论文；当前实现依据 M=b70ae056c495b43ca3f305fe438926b97b2723b5，见[B/C 方法说明](../methods/evidence-and-shared-updates.md)。旧 340a3541 的实现描述不能覆盖当前修复。结构近邻成立，不等于已经证明其 TAD 迁移优于本项目；任务名称不同也不足以证明新颖。

## B：密集状态、压缩表征与重对齐

|工作与主来源|原文支持的机制|对本研究的约束|
|---|---|---|
|[Coarse-Fine Networks for Temporal Activity Detection in Videos，CVPR 2021](https://arxiv.org/html/2103.01302v2)，§3.1–3.2|Grid Pool 学习非均匀时间采样；Grid Unpool 在 logits 端重对齐；Fine 分支保留完整时间，融合显式校准时间对应。|不能声称首次非均匀采样后密集定位或时间校准。原文包含 Charades 逐帧活动定位，不能直接当作 THUMOS 区间 high-tIoU 比较。|
|[Multi-Time Attention Networks，ICLR 2021](https://arxiv.org/abs/2101.10318)，方法部分|连续时间 attention 把不规则观测映射到固定参考时间。|规则时间 query 接收不规则 evidence 已有先例；其通用映射并未自动证明任何 TAD 边界机制。|
|[Perceiver IO，ICLR 2022](https://arxiv.org/abs/2107.14795)，架构部分|输入、潜在压缩表示与输出 query 解耦，输出大小和结构可独立设定。|观测与输出网格解耦不是本项目独有。|
|[TokenLearner: What Can 8 Learned Tokens Do for Images and Videos?，NeurIPS 2021](https://arxiv.org/abs/2106.11297)，TokenFuser 与视频架构|少量学习 token 做高成本建模，TokenFuser 将信息返回完整表示。|不能把压缩—处理—恢复一概说成只适合分类或只能进行一次。|
|[LookupViT: Compressing Visual Information to a Limited Number of Tokens，ECCV 2024](https://arxiv.org/abs/2407.12753)，Lookup block 与视频实验|少量 compressed tokens 与较大 lookup 表示通过双向交互结合，包含视频分类扩展。|完整轻状态与少量重状态反复交互已有直接视觉近邻；不是已有 TAD 区间有效性证据。|
|[AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.pdf)，方法部分|低分辨率完整视频分支与高分辨率 ROI、时间一致区域选择。|是实际存在的定位近邻；目标为事件点，不能把它写成已解决区间双端定位。当前 B 也尚非该类原图 ROI 获取器。|

以上各工作不完全等价于 B；组合这些已有概念也不自动形成新的架构贡献。B 应检验：给普通 receiver 同样的坐标、有效性、null、时间适配及合理容量后，完整支持信息是否仍减少证据空洞附近的区间错误。该现象及其修正收益目前是待证假设。

## C：混合尺度、共享更新与密集恢复

|工作与主来源|原文支持的机制|必须保留的区别|
|---|---|---|
|[MSViT: Dynamic Mixed-Scale Tokenization for Vision Transformers，2023](https://arxiv.org/html/2307.02321v1)，§3 与分割实验|输入处学习 coarse/fine 选择，支持将粗 token 恢复到密集布局。|当前 C 每层重建表征，但沿用一次路由计划；不能称当前 C 每层重新路由。覆盖每个区域不等于表征信息无损。|
|[Token Merging for Fast Stable Diffusion，2023](https://arxiv.org/html/2303.17604v1)，§3.1–3.2、§4.2/Table 3|重型子层前合并，之后展开，再加原 skip；比较 self-attention、cross-attention、MLP 的压缩组合。|最终设计偏好只压缩 self-attention；不能把所有子层压缩说成唯一原版，或只用该版本作弱对照。|
|[ALGM: Adaptive Local-then-Global Token Merging for Efficient Semantic Segmentation，CVPR 2024](https://arxiv.org/abs/2406.09936)，方法部分|局部再全局合并，利用对应关系恢复密集分割表示。|已有 token merging 直接研究密集预测。|
|[CubistMerge，2025 预印本](https://arxiv.org/abs/2509.21764)|研究保留二维空间结构的 token 合并。|结构保持不是本项目可以独占的泛化主张；不据此宣称其 TAD 迁移已经有效。|
|[StructSAM: Structure- and Spectrum-Preserving Token Merging for Segment Anything Models，2026 预印本](https://arxiv.org/abs/2603.07307)|SAM 场景下保护结构、边界/提示相关信息并合并、恢复 token。|边界敏感和恢复也已有近邻；空间分割边界与时间动作边界的差异须落实为可测问题。|
|[VidToMe: Video Token Merging for Zero-Shot Video Editing，2023 预印本](https://arxiv.org/abs/2312.10656)|视频编辑中的跨帧 token 合并与时间一致性。|不能泛称首次考虑合并的时间一致性；需检验 TAD 特定的时间判别误差。|

ToMeSD 的展开加 residual 与 C 的共享 delta 不必逐算子相同，但已足以否定“首次恢复完整状态、首次保留成员差异”的独占主张。新的比较须保留合理残差、恢复关系与 TIA，不能通过永久删去对手的状态制造区别。

## LITE：价值预测与预算的直接近邻

[Principles of Visual Tokens for Efficient Video Understanding，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Hao_Principles_of_Visual_Tokens_for_Efficient_Video_Understanding_ICCV_2025_paper.pdf)及[arXiv v2](https://arxiv.org/html/2411.13626v2)的 §3、式(1)–(2)、§4.2–4.3、§5.2 已核对：

- 以真实类别 logit 对后层 MLP 激活的梯度构造 Grad-CAM；聚合后 ReLU，再缩放到 [0,1]。这是类别价值代理，并非逐个删掉负梯度，也不是 signed 实际操作损失差。
- 三层 MLP 从冻结 patch embedding 预测代理分数，以 BCE 训练。LITE++ 用 MoviNet 置信度驱动预算规则，不等同于学习预期边际任务收益。
- 使用 VideoMAE，包含从分类训练的 selector 向 AVA 检测迁移。因此不能将其排除为“只有分类”。在上述已读方法与实验中，未找到本项目所提等成本 swap、双端区间风险或改计划后完整路径反事实校准的实现证据；这不是对全部相关文献的不存在证明。

“低成本模型预测 token 价值，并按难度给预算”不能继续单独作为本项目创新。强对照应包括原风格类别代理与用相同 TAD 风险重新训练的 LITE-inspired 代理；后者是有明确定义的迁移控制，不冒称官方复现。若其达到相同效果，应把解释收敛到任务目标或适配，而非坚持新的价值估计优越。

## 当前能成立的研究位置

|路线|已有执行范式|待建立的具体增量|
|---|---|---|
|A|MoD/CoDA 条件执行|实际合法更新的区间收益是否更可预测、可校准，是否优于同 TIA 强路由。|
|B|粗细分支及不规则观测接收|在同一 evidence 下，是否避免把远端语义误外推为连续局部观察。|
|C|混合尺度及 merge/process/unmerge|任务收益是否比相似性更能决定共享更新；计算模式变化是否产生可干预的时间误差。|

三者作为同一研究原则的三个执行器；A 仍为预指定主线。当前没有证据允许宣称上述增量已经成立，也不因近邻存在判定路线不可行。
