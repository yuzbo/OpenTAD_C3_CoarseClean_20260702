# ZoomToken 文献研究地图与空白分析（证据绑定版）

- 日期：2026-08-25
- 状态：待用户详细审阅（DRAFT-FOR-REVIEW）
- 范围：LITERATURE_AND_GAP-v001 注册论文 + 2025–2027 独立扫描
- 证据纪律：每条判断附来源；只拿到摘要未拿到正文/局限节的维度标【未知】

**证据边界说明**：本次核验证据 = 12 篇 arXiv 原文摘要 + 6 篇正文 limitations 原文引句（MoD、A-MoD、EVAD、STTS、AdaTAD、AdaSpot）。DynamicViT 正文 HTML 转换失败，其作者自认局限【未知】。TokenLearner、ToMe、Glance&Focus、AdaFocusV2、Uni-AdaFocus、FlashVID 的作者自认局限节未取得【未知】。

---

## 一、研究地图

### A. 识别/分类类（视频级或图像级一个标签）

| 论文 | 研究对象 | 方法核心 | 选择变量/粒度 | 数据来源 | 实验条件 | 主要结论（原文数字） |
|---|---|---|---|---|---|---|
| **Glance & Focus** (NeurIPS'20, 2010.05300) | 图像分类 | RL 顺序选择小patch序列，置信度够即早停 | 像素级 crop 位置+步数 | ImageNet | MobileNet/EfficientNet/RegNet 骨干；真机 iPhone XS Max | "reduces the average latency of the highly efficient MobileNet-V3 on an iPhone XS Max by 20% without sacrificing accuracy" |
| **AdaFocusV2** (CVPR'22, 2112.14238) | 视频分类 | 可微插值 patch 选择替代 V1 的 RL 三阶段；conditional-exit 时序自适应 | 像素级 crop 位置/大小 + 早退帧 | ActivityNet、FCVID、Mini-Kinetics、SSv1&v2、Jester | 端到端单阶段训练 | "significantly outperforms the original AdaFocus...while being considerably more simple and efficient to train"；V1 被自述为 "a complicated three-stage training pipeline (involving reinforcement learning)" |
| **Uni-AdaFocus** (TPAMI, 2412.11228) | 视频分类 | 全局轻量编码器→policy net 定位 patch→大网络推理；空间+时序+样本级三级动态 | 连续 ROI（形状/大小/位置平滑移动）+帧分配+难易样本 | 7 个基准、3 种应用场景 | 兼容 TSM/X3D 现成骨干 | "considerably more efficient than the competitive baselines"；具体数值【未知：摘要无数，正文未取】 |
| **DynamicViT** (NeurIPS'21, 2106.02034) | 图像分类 | 轻量预测模块打分，逐层递进剪枝；attention masking 可微化 | token 保留/丢弃，分层 | ImageNet | 多种 ViT；端到端 | "pruning 66% of the input tokens...reduces 31%~37% FLOPs and improves the throughput by over 40% while the drop of accuracy is within 0.5%"；局限【未知】 |
| **TokenLearner** (NeurIPS'21, 2106.11297) | 图像+视频分类 | 学习挖掘 8–16 个重要 token，仅对其做 pairwise attention | 学习到的少量 token 身份 | ImageNet、K400、K600、Charades、AViD | 端到端 | "comparable results to the state-of-the-arts on ImageNet while being computationally more efficient"；局限【未知】 |
| **STTS** (ECCV'22, 2111.11591) | 视频分类 | 排序式轻量 scorer，时序选帧+空间选区；perturbed-max 可微 Top-K | 帧级 + 特征图区域级 | K400 主、SSv2；MViT-B16、VideoSwin-B | 端到端 | "keeping only 50% of the input tokens...reduces...GFLOPs by more than 33% with a drop of accuracy within 0.7%"；**"temporal redundancy is more significant than spatial redundancy in videos"**，"temporal selection exceeds the spatial selection by a large margin" |
| **ToMe** (ICLR'23 Oral, 2210.09461) | 图像/视频/音频识别 | 免训练，轻量匹配算法逐层合并相似 token | 合并对（bipartite 匹配） | 图像 ViT-L@512/ViT-H@518、视频 ViT-L、音频 ViT-B | **off-the-shelf 免训练**；也可训中用 | "2x the throughput...on video with only a 0.2-0.3% accuracy drop"；定性："merges object parts into one token, even over multiple frames"；局限【未知】 |
| **MoD** (2024, 2404.02258) | 语言模型 | 每层 top-k 路由，容量 k 先验固定→静态计算图 | 每层每 token 过/跳过 | LM（具体基准【未知：摘要未列】） | **从零训练**，isoFLOP 对照 | "match baseline performance for equivalent FLOPS and wall-clock times...upwards of 50% faster to step during post-training sampling" |
| **A-MoD** (2024, 2412.20875) | 图像分类为主 | 用前层 attention 图做 MoD 路由，**零新增路由参数** | 同 MoD（隔层、容量 50%/12.5%） | ImageNet-1k；迁移到 Cars/Pets/Flowers；DETR 检测 | DeiT-T/S、ViT-B/L；含 post-hoc 适配 | "up to 2% higher accuracy on ImageNet compared to standard routing and isoFLOP ViT baselines"；"reduce the number of FLOPs by up to 18% without dropping performance"；"up to 2× faster transfer learning" |
| **FlashVID** (ICLR'26 Oral, 2602.08024) | 视频大语言模型 | 免训练：ADTS 注意力+多样性选 token，TSTM 树状时空合并 | VLLM prefill 的视觉 token | 3 个 VLLM、5 个基准 | **post-vision-encoder，无下游梯度** | 保留 10% 视觉 token 维持 "99.1% of the performance of LLaVA-OneVision"（**相对分**，非绝对精度）；Qwen2.5-VL 同算力 10× 帧数 +8.6% 相对增益 |

### B. 检测/定位类（密集预测）

| 论文 | 研究对象 | 方法核心 | 选择变量/粒度 | 数据来源 | 实验条件 | 主要结论（原文数字） |
|---|---|---|---|---|---|---|
| **AdaTAD** (CVPR'24, 2311.17241) | **离线 TAD**（本项目检测器） | TIA 轻量适配器+冻结骨干，端到端扩到 1B 参数/1536 帧 | 无 token 选择（全密集） | THUMOS14、ActivityNet-1.3、EK-100、Ego4D-MQ | VideoMAE-S → VideoMAEv2-g 多档 | THUMOS14：VideoMAE-S avg **68.8** / @0.7 **46.9**；最佳 1536帧-g 型 avg **76.9** / @0.7 **56.1**；摘要称 75.4% mAP。**全文无任何 GFLOPs/速度句**（仅内存与解码时间讨论） |
| **EVAD** (ICCV'23, 2304.08451) | **时空动作检测**（AVA，关键帧框+分类，非区间定位） | 关键帧 token 全保留，非关键帧按与关键帧相关度递进剪枝（ρ=0.7×3 次）+ 上下文精炼解码器 | 帧角色（key/non-key）+ token 保留 | AVA | ViT-B；**需重训** | "reduces the overall GFLOPs by 43% and improves real-time inference speed by 40% with no performance degradation"；编码器输出仅 34% token |
| **AdaSpot** (CVPR'26, 2602.22073) | **精确事件定位 PES**（帧级事件，非区间） | **非监督**显著度 RoI 选择器（每帧一个区域，时空一致性约束），低分全局+高分 RoI 双路 | 每帧一个 RoI（分辨率分配） | Tennis、FineDiving、FineGym、F³Set | 双分辨率特征提取+时序建模 | "+3.96 and +2.26 mAP@0 frames on Tennis and FineDiving"；vs E2E-Spot：6× 更少参数、1.5× 更少 FLOPs |

**2025–2027 独立扫描结论**（注册表要求的碰撞审计）：以 "efficient offline TAD / token pruning TAL / adaptive computation TAD 2025–2026" 检索，**未发现在离线 TAD backbone 内部做 token 级自适应计算的同题工作**。最近邻：WACV'26 蒸馏离线检测器到在线流式（4× 延迟改善，任务为在线化）、OZ-TAL（在线零样本）、MDVLM-TAL（扩散式 TAL，THUMOS 72.7，非效率方法）。证据强度：web 检索级，非系统综述；未索引会议/工业报告【未知】。

---

## 二、空白分析（五类，逐项对应原文）

### 1. 结论互相矛盾之处

- **矛盾C1：可学习 ROI 裁剪到底稳不稳？**
  - AdaFocusV2/Uni-AdaFocus：端到端可微 ROI 学习成功（"enabling efficient end-to-end optimization"，2112.14238）。
  - AdaSpot（CVPR'26）：**"most existing methods rely on learnable cropping mechanisms, which can be unstable to train"**，且在 PES 上此类方法 "perform poorly when transferred to the PES setting"，原因是 "limited supervision, low input diversity, and training instability"（2602.22073）。
  - 本项目内部证据站在 AdaSpot 一侧：ROI-only G 61.49 vs DN 64.73 负结果 + 角色坍塌 0/0/3342/336（decision_history）。**三方对同一技术路线的可训练性结论相反，且各自任务不同——这是地图上最深的裂缝。**
- **矛盾C2：MoD 层路由到底行不行？**
  - MoD 原文：从零训练 LM 上匹配基线、采样快 50%（2404.02258）。
  - A-MoD 原文：ImageNet 上比标准路由高 2%（2412.20875）；**但同文自认 "MoD models are unable to match the isoFLOP model performance on transfer tasks"（迁移任务追不平），且检测任务上 "MoD and A-MoD achieve comparable results"（对 dense 无明确收益）**。
  - 本项目内部：DROP32 −2.96、MOD32-KV −2.57、RC32-KV −4.34、DSR6-KV −1.69，全部终止（PAPER_PROGRESS）。即：**MoD 家族在"从零训练 LM/分类"成立，在"适配预训练骨干 + 密集定位"上目前全部证据为负——包括 A-MoD 作者自己的迁移与检测两组实验。**
- **矛盾C3：高比例剪枝是否无损？**
  - DynamicViT：剪 66% token、掉点 <0.5%（2106.02034）；STTS：留 50%、掉点 <0.7%（2111.11591）；ToMe：免训练 2× 加速、掉 0.2–0.3%（2210.09461）。
  - EVAD 作者自己证伪其普遍性是有条件的：**"token pruning drops a high percentage of tokens (66%), resulting in poor performance on categories with small motion or interaction with small objects"**（2304.08451）。
- **矛盾C4：哪种冗余主导？**
  - STTS：K400 上 "temporal redundancy is more significant than spatial redundancy"。
  - AdaFocus 全系 + AdaSpot：空间冗余为主轴取得成功（视频分类与 PES）。
  - FlashVID：批评 "compress spatial and temporal redundancy separately...yielding suboptimal results"，主张时空必须联合（2602.08024）。
  - 三者的冗余排序结论互不相同，但**没有任何一篇在定位任务上测过冗余排序**。

### 2. 方法局限（原文可证的结构性约束）

- MoD：top-k 路由 "present difficulties in post-training autoregressive sampling"；只验证 decoder-only LM；"We did not study this extensively"（部分设计）；容量 "appears to be empirically determinable"（无原理）。
- A-MoD：迁移差距（见 C2）；"attention maps do not always learn semantically meaningful scores...especially for larger models, where the attention scores tend to concentrate on a single patch"；ViT-B 12.5% 容量下 "marginally worse"。
- EVAD："requires re-training once"；"Further reducing the keep rate degrades the performance because the number of tokens kept is insufficient to contain complete semantic information"。
- STTS：VideoSwin 上只做了时序选择（窗口 shuffle 太复杂）；**"multi-step selection leads to frequent changes of the spatio-temporal structures of the videos and is more difficult to train"**。
- AdaTAD：Ego4D-MQ 上 VideoMAE-L 7200 帧单视频 60GB；"loading, decoding, and processing such long videos take much longer time than pre-extracted features"；无速度报告。
- AdaSpot："generalization beyond sports remains to be evaluated"；"scenarios involving simultaneous actions...require further study"。
- FlashVID：post-encoder、无检测器梯度、无高 tIoU 证据（repo 评审注 + 摘要一致）。

### 3. 只在特定条件成立的结论

| 结论 | 成立条件 | 条件外证据 |
|---|---|---|
| MoD 匹配 dense | 从零训练、LM、isoFLOP 对照 | 迁移追不平（A-MoD 自认）；检测无收益（A-MoD 自认）；TAD 适配失败（内部 4 臂） |
| 高剪枝 <1% 掉点 | 图像/视频**分类**，单标签 | 小动作/小物体类别差（EVAD 自认）；TAD @0.7 内部衰减 0.9–1.9pp（A→C） |
| 时序冗余主导 | K400/MViT 分类 | 定位任务上【未知：无人测过】 |
| 可学习 ROI 端到端有效 | 视频分类（AdaFocus 系） | PES 上不稳定（AdaSpot）；TAD 上 ROI-only 负（内部 G 臂） |
| 免训练合并无损 | 分类/VLLM，下游冗余容忍强 | 检测器梯度下【未知：ToMe/FlashVID 均未碰检测】 |
| AdaTAD 75.4/76.9 | VideoMAEv2-g、1536 帧大配置 | VideoMAE-S 档即降到 68.8（原文表） |

### 4. 缺少验证的场景

- **离线 TAD 的 backbone 内部 token 级自适应计算：全表为零。** EVAD 是关键帧检测且全留 keyframe token；AdaSpot 是帧级事件；DynamicViT/STTS/TokenLearner 是分类；MoD/A-MoD 是 LM/分类/DETR；FlashVID 是 VLLM。注册表 candidate gap 的六性质组合在检索范围内无碰撞（证据强度见扫描结论）。
- **高 tIoU 区间的效率-精度曲线**：所有剪枝/合并论文均只报分类精度或 @0.5 级检测指标；TAD @0.7+ 的 trade-off 形状在文献中无可引用数据【未知】。
- **种子方差**：13 篇均未见多种子误差棒报告【未知：正文未逐篇核验，但摘要级均无】；而内部证据显示 0.2–0.9pp 差异在 TAD 种子噪声量级内。
- **选择器计入成本的端到端账**：DynamicViT/STTS/EVAD 报 GFLOPs/吞吐；AdaTAD 无速度；MoD 报 LM 训练/采样 wall-clock；**无任何论文在 TAD 上报 selector-inclusive 的 decode→NMS 延迟/能耗**。

### 5. 作者自认 limitations（原文引句，已全部并入上文第 2 节）

覆盖：MoD、A-MoD、EVAD、STTS、AdaTAD、AdaSpot（6 篇原文核验）。DynamicViT、TokenLearner、ToMe、G&F、AdaFocusV2、Uni-AdaFocus、FlashVID 的作者自认局限【未知：正文局限节未取得】。

---

## 三、为什么同一个问题，得到不同结果？

把"视频冗余可被安全压缩"视为同一问题，各论文结论分歧的可识别来源（每条有原文支撑）：

1. **监督粒度不同 → "冗余"的定义不同。** 分类任务最终只需一个标签，多数 token 本就是可弃的（DynamicViT："the final prediction...is only based on a subset of most informative tokens"）；而离线 TAD 的 ActionFormer 头需要**每个时序位置**的特征做边界回归——"丢帧"在分类里是时序选择（STTS 的主力收益来源），在 TAD 里直接切断该位置的监督通路。EVAD 之所以 43% 剪枝无损，前提恰是 "we preserve all keyframe tokens for accurate actor localization in the keyframe"——被检测的那帧一个 token 都没剪。
2. **训练制度不同 → 路由可学性不同。** MoD 成功条件是"从零训练"（2404.02258）；A-MoD 自认迁移场景 MoD 追不平 isoFLOP（2412.20875）；AdaSpot 自认可学习裁剪在弱监督定位任务上不稳（2602.22073）。本项目是"适配预训练 VideoMAE + 稀疏边界监督"，恰好落在两个失败条件的交集里——内部四次层路由失败与文献相容，不是异常。
3. **冗余主导性随数据时间尺度变化。** K400 是秒级动作分类（时序冗余主导，STTS）；体育 PES 是帧级瞬发事件（空间分辨率主导，AdaSpot）；VLLM 多轮 QA 需要时空联合（FlashVID）。**没有跨任务统一的冗余排序——"哪种冗余大"本身是数据属性，不是方法属性。**
4. **效率度量口径不同。** GFLOPs（DynamicViT/STTS/EVAD）≠ 吞吐（DynamicViT/ToMe）≠ 真机延迟（G&F 的 iPhone）≠ 训练 wall-clock（MoD）≠ 相对性能保持（FlashVID 的 99.1% 是相对分）。跨论文比较"节省"在度量层面就不可通约。
5. **"免训练成功"与"训练内成功"不矛盾，因为它们作用的下游不同。** ToMe/FlashVID 的下游（分类/VLLM）对特征扰动鲁棒；检测器边界回归对特征质量敏感（本项目 A→C 臂 @0.7 衰减与 EVAD 小动作类别失败均为证据）。

---

## 四、5 个可验证研究假设

**H1：剪枝类方法在 TAD 上的掉点主要由"边界回归对特征质量的敏感度"驱动，而非分类信息不足。**
- 依据：DynamicViT/STTS 的分类无损结论（2106.02034；2111.11591）+ EVAD "poor performance on categories with small motion"（2304.08451）+ 内部 A/B/C 臂 @0.7 衰减（68.73/47.24 → 68.22/45.35）大于 Avg 衰减。
- 需要数据：THUMOS14 上同一选择器、同一保留率，分 tIoU 阈值（0.3/0.5/0.6/0.7/0.75）的 mAP 衰减曲线。
- 如何验证：比较 @0.5 衰减幅度 vs @0.75 衰减幅度；并对检测结果按"边界偏移量"分桶看特征保留率的影响。
- 推翻条件：若各 tIoU 阈值同步等量衰减 → 掉点是分类头信息不足驱动，H1 被推翻。

**H2：MoD 层路由在 TAD 上的失败主因是"预训练骨干迁移"，而非路由机制本身。**
- 依据：A-MoD "MoD models are unable to match the isoFLOP model performance on transfer tasks"（2412.20875）+ MoD 从零训练成功（2404.02258）+ 内部 MOD32-KV −2.57。
- 需要数据：THUMOS14；strict A-MoD 参考臂（零路由参数、前层 attention 列均值路由，内部 a41714e9 分支已实现）训练至 terminal。
- 如何验证：在匹配容量（0.5）下比较 A-MoD 臂 vs MOD32-KV 臂的 Avg/@0.6/@0.7。若零参数路由显著优于可学路由 → 支持"可学路由器的训练噪声是主因"（A-MoD："routing introduces noise into the training process"）；若两者同样失败 → "迁移"解释不足，H2 被推翻，转向任务结构解释（H1）。
- 注：此臂本来就是路线图内欠账（A-MoD 必须训练成对照行）。

**H3：相邻 tubelet 特征相似度分布可以跨任务预测"时序复用"的可行性上限。**
- 依据：STTS 时序冗余主导（K400，2111.11591）vs AdaSpot 空间 ROI 有效（帧级事件，2602.22073）vs FlashVID 时空必须联合（2602.08024）——三者的分歧若由数据相似度结构解释，则该度量应有跨任务预测力。
- 需要数据：THUMOS14 全训练集的相邻 tubelet 余弦相似度分布（APM 已在 0.80 阈值上统计）；K400 子集的同口径分布作对照。
- 如何验证：THUMOS14 高相似（≥0.8）tubelet 比例应显著高于 K400（STTS 结论反向预测 K400 时序冗余其实也大——这本身是 H3 内部的张力，见下）；然后检验 APM32-CTX64 臂的 gate 结果是否与相似度预测方向一致。
- 推翻条件：若 THUMOS14 相似度分布显示高冗余但 APM32-CTX64 仍过不了 gate（对 MOD32-KV 无优势）→ "相似度⇒可复用"链条被推翻；若两数据集分布无差异但文献结论相反 → 度量的跨任务预测力被推翻。

**H4：可学习 ROI 的不稳定性来自"稀疏监督下的像素级裁剪梯度"，而非 ROI 概念本身；在原生 token 网格 + 硬检测损失 + 直通估计下可稳定。**
- 依据：AdaSpot "learnable cropping...unstable to train"（2602.22073）vs AdaFocusV2 分类任务端到端成功（2112.14238）vs 内部 G 臂（连续 ROI）61.49 失败 vs R1 臂（8×8 硬矩形 + 直通软 margin）gate 通过。
- 需要数据：THUMOS14；R1（已训练）+ G（已训练）+ 逐因素消融臂（像素裁剪↔token 网格；直通↔PL/RL；分类损失↔检测损失三轴）。
- 如何验证：沿三轴分解 G→R1 的差异，观察哪个轴翻转稳定性。
- 推翻条件：若 token 网格 + 直通 + 检测损失下仍出现 G 式坍塌 → "ROI 概念本身在 TAD 不可学"，H4 被推翻（对项目含义：退守固定/显著度几何，放弃可学位置）。

**H5：免训练 token 合并在 TAD 上与训练内选择存在系统性差距，因为合并特征缺乏检测器梯度校准。**
- 依据：ToMe 免训练 0.2–0.3% 掉点（2210.09461）、FlashVID 99.1% 相对保持（2602.08024）均为分类/VLLM 下游；repo 评审注："no detector gradient, no high-tIoU evidence"。
- 需要数据：THUMOS14；冻结 VideoMAE-S 特征 + AdaTAD 头 + ToMe 式逐层合并（无需重训 backbone，只需训练/微调检测头）。
- 如何验证：比较免训练合并 vs R1（训练内选择）在匹配 token 预算下的 Avg 与 @0.7。
- 推翻条件：若免训练合并达到 R1 的 ≥99% 相对 mAP（FlashVID 口径）→ H5 被推翻，且这将直接威胁本项目方法的必要性，需重新定位贡献。

---

## 五、现有证据里解释不通的

1. **EVAD 之谜**：EVAD 在 AVA 剪掉 66% token "no performance degradation"（2304.08451），本项目在 THUMOS14 同量级剪枝掉 1.7–4.3pp。候选解释有四个（关键帧全保留设计 / 关键帧检测 vs 区间边界任务 / AVA vs THUMOS 数据特性 / 重训预算差异），**没有一篇原文同时控制这些变量，无法在文献内仲裁【未知】**。这同时是最有价值的一个对照实验设计空间。
2. **A-MoD 检测实验的基线语义不明**："MoD and A-MoD achieve comparable results in this case" 无法从引句判定是"两者都=dense（无损）"还是"两者都<dense（无收益）"【未知：需读 DETR 实验节原表】。这决定 MoD 家族对密集预测的中性/负性结论。
3. **STTS 的时序>空间结论从未在定位任务上检验**，向 TAD 的外推证据为零【未知】。
4. **R1 已通过 gate，但"scout 学习位置"的价值未被排除**：中心固定 8×8 矩形对照臂不存在；文献中"TAD 上学习位置 vs 中心先验"的证据为零【未知】。若中心固定达 ~69，方法的几何学习部分无贡献。
5. **AdaTAD 原文无任何速度/效率句**（仅内存讨论）——本项目要讲的"效率"故事在检测器母论文层面就没有可对接的度量传统，TAD 的 efficiency frontier 在文献中无可引用曲线【未知】。
6. **种子噪声 vs 方法效应**：13 篇文献均无误差棒传统【未知：正文未逐篇核验】，而内部所有臂间差异（0.2–0.9pp）处于 TAD 种子噪声量级——这意味着**文献中所有 <1pp 的"无损"结论在本项目的统计标准下都不可直接采信**，反之亦然。

---

## 六、关键提醒

1. 第四节假设全部是"可推翻"的，且 H2/H3/H4 的验证臂在现有路线图/分支里已经存在或已排队——文献分析没有改变实验优先级，而是给每个已排队实验补上了"推翻条件"的文献锚点。
2. "缺少验证场景"只是候选空白，不是创新声明；碰撞审计仅为 web 检索级，正式投稿前需系统综述复核。
3. 本文档中所有内部实验数字引用自 PAPER_PROGRESS.md 与 decision_history.md（截至 2026-08-25）。

---

## Sources

- [AdaTAD (arXiv:2311.17241)](https://arxiv.org/abs/2311.17241)
- [A-MoD (arXiv:2412.20875)](https://arxiv.org/abs/2412.20875)
- [MoD (arXiv:2404.02258)](https://arxiv.org/abs/2404.02258)
- [DynamicViT (arXiv:2106.02034)](https://arxiv.org/abs/2106.02034)
- [TokenLearner (arXiv:2106.11297)](https://arxiv.org/abs/2106.11297)
- [STTS (arXiv:2111.11591)](https://arxiv.org/abs/2111.11591)
- [ToMe (arXiv:2210.09461)](https://arxiv.org/abs/2210.09461)
- [FlashVID (arXiv:2602.08024)](https://arxiv.org/abs/2602.08024)
- [Uni-AdaFocus (arXiv:2412.11228)](https://arxiv.org/abs/2412.11228)
- [AdaFocusV2 (arXiv:2112.14238)](https://arxiv.org/abs/2112.14238)
- [Glance and Focus (arXiv:2010.05300)](https://arxiv.org/abs/2010.05300)
- [EVAD (arXiv:2304.08451)](https://arxiv.org/abs/2304.08451) · [EVAD GitHub](https://github.com/MCG-NJU/EVAD)
- [AdaSpot (arXiv:2602.22073)](https://arxiv.org/abs/2602.22073) · [AdaSpot GitHub](https://github.com/arturxe2/AdaSpot)
- [Distilling Offline Action Detection (WACV 2026)](https://openaccess.thecvf.com/content/WACV2026/papers/Patel_Distilling_Offline_Action_Detection_Models_into_Real-Time_Streaming_Models_WACV_2026_paper.pdf)
- [OZ-TAL (arXiv:2605.09976)](https://arxiv.org/html/2605.09976v1)
- [MDVLM-TAL](https://www.emergentmind.com/papers/2605.29858)
