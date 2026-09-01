# DUCA 稀疏 Token 物理时间表示跨领域迁移：最终科学裁决请求

## 路由身份

- Exact Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）
- Nonce: `DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825`
- GitHub: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 当前可远程核验实现 revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
- 当前模型分支（定位用）: `codex/duca-h65-firstmix-singleclock-20260824`
- 这是一段全新的 Project 会话；不得依赖旧聊天中的未陈述信息。
- Project Sources 中较新的状态文件存在容量与确认历史，本轮不把任何未确认 Source 当作依据；以下完整内联材料是本轮权威上下文。

## 你的职责

请担任本项目的 **Scientific First-Author Agent and Primary Research Owner**，同时先以极其严厉的顶会审稿人攻击当前方法，再以具有视频理解、视觉 Transformer、稀疏计算、长上下文大语言模型与时序动作检测经验的科研同行身份给出唯一可执行裁决。

你必须返回恰好一个总决策：`CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。不得把路线选择交回给人类或 Codex。若继续或修订，必须只冻结一个主机制和一个最便宜、能改变下一步决策的真实 falsifier；不要提出无界模型矩阵。

## 一、当前科学问题

DUCA 的 H65 路线先用低成本 ASFormer scout 学习动作性和边界语义，再用确定性 sampling-rate transport 选择 `K=384` 个非均匀、严格递增的原始 RGB 帧。它确实只把这 384 帧送入重型 VideoMAE-S，因此不是在 dense 768 帧之后做掩码的伪稀疏。

然而，当前代码把这 384 个选中帧按 selected rank 重排为 `24 × 16`，并由 VideoMAE 的 Conv3D PatchEmbed 以 temporal kernel/stride `2` 先混合相邻 selected-rank 帧。两帧在 tensor 中相邻，并不表示它们在原视频物理时间上相邻。随后再把 proposal 回映到物理时间，只能修复输出坐标，不能恢复 backbone 已经发生的错误局部时间解释。

当前 SingleClock 只在 Conv3D PatchEmbed **之后**，向第 0 个 Transformer block 的 attention 加入真实时间相对偏置。训练后可学习标量约为 `-0.0018`，实际 bias 极小；其 terminal-EMA 相对 H65 OFF 的描述性差值为 Avg-mAP `-0.6596`、mAP@0.6 `-0.4037`、mAP@0.7 `-0.1720` 个百分点。该比较尚缺预注册硬身份材料，正式效能裁决仍为 `EVIDENCE_ADMISSION_BLOCKED`，不能称 PASS 或 KILL；但“真实时间信息作用得太晚且机制几乎未被使用”已构成下一轮表示设计的明确科学歧义。

本轮唯一问题是：

> 如何吸收 video、视觉 token 稀疏化/合并与 LLM 稀疏 token/KV 保留中的原始位置保持思想，设计一个 H65-compatible、在 VideoMAE 首次时序混合之前或内部保留真实物理时间/支撑区间、且适合 TAD 高 IoU 边界定位的稀疏表示机制？

## 二、已知代码真值

请直接核验 GitHub revision 和下列符号；不要只复述本摘要。

1. `opentad/models/backbones/backbone_wrapper.py`
   - 当前在调用 VideoMAE 前根据全局 `[B,384]` 原始位置形成 clip/tubelet 坐标，并把实际/规范坐标传给 backbone。
2. `opentad/models/utils/temporal_grid.py`
   - `global_rank_clip_coordinates`：将 `[B,384]` 按原始 rank 变为 `[B,24,16]`，再按 tubelet size 2 得到每 clip 8 个 tubelet 中心。
   - `clip_relative_physical_time_mask`：当前计算 `(actual-canonical)/(T-1)` 的 pairwise residual，并扩展到 temporal-major spatial tokens。
3. `opentad/models/backbones/vit_adapter.py`
   - `PatchEmbed` 为 Conv3D，temporal kernel/stride 均为 tubelet size 2。
   - `self.patch_embed(x)` 是当前最早的时序混合。
   - SingleClock 物理时间偏置在 PatchEmbed 之后构造，只进入 block 0 attention。
4. `opentad/models/dense_heads/anchor_free_head.py`
   - detector point/proposal 在 head 中按 selected positions 映到 dense physical frame axis；NMS 前已经是物理时间。
5. `opentad/models/detectors/actionformer.py`、`single_stage.py`
   - H65 身份封存与最终后处理入口。

请明确区分三个位置：

```text
原始 RGB 帧及真实 timestamp/support
    -> Conv3D tubelet PatchEmbed（当前最早时序混合）
    -> Transformer block 0 attention（当前 SingleClock 才在这里生效）
    -> Adapter / ActionFormer / pre-NMS physical decode
```

## 三、必须保留的实验与主张边界

本轮表示归因门必须保持：

- H65 的同一 ASFormer 语义间接非均匀选帧；
- 同一组 selected RGB，`K=384`；
- 同一 Stage-1 30 epoch 起点、Stage-2 60 epoch / 6000 successful updates；
- 同一 VideoMAE-S/Adapter/ActionFormer、检测损失、NMS、THUMOS14 split、seed `3407`、官方 evaluator 与 terminal epoch-59 EMA；
- proposal 在 filtering/top-k/IoU/NMS 前恰好一次进入物理时间；
- 真实 heavy compute 仍只处理 384 帧；
- 每 5 epoch 保存完整可恢复 checkpoint，禁止用 validation 挑中间最好点。

本轮禁止：

- 重复 dense、uniform、random 或 fixed-K 旧控制；
- 把 continuous cliplet 恢复为主路线（其 FZ/JT 已为 `49.89/47.24` 的完整负结果）；
- 在表示归因门加入 dynamic-K、Query-Bridge、UVT value、direct selector 或新的采样器；
- 改变 selected RGB、K、训练长度、detector、loss、NMS、evaluator、split 或 seed；
- 用本地 CPU/synthetic 结果宣称效能；
- 把 SingleClock 的证据准入缺口误写成科学 KILL。

固定 K 只用于这次表示归因，不得重新成为论文最终主线。若表示门成功，论文主路线仍应回到“语义预测 -> 确定性间接采样 -> dynamic outer-K”。

## 四、已有真实结果，不得跨协议误归因

- shared official dense AdaTAD: Avg-mAP `68.73`（背景锚，不是同提交因果基线）。
- matched H65 30+60 terminal EMA: Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
- 历史 H65 30+60: Avg-mAP `65.3857`；是复合协议结果，不能归因于单模块。
- compressed H65 20+40: Avg-mAP `62.4648`；压缩日程已停止。
- RankPack/TrueTime，同提交、同 K、同 seed、同 60 epoch：`61.57 -> 62.19` Avg，mAP@0.6 `+1.69`、@0.7 `+0.79`，只构成单 seed 部分机制支持。
- SingleClock ON terminal EMA: Avg-mAP `64.4661`；当前只作描述性负向预警，正式裁决尚未准入。
- UVT `off/geo/geo_ema`: `57.35/55.93/55.92`，混杂 selection score 与 budget evidence，不能否定 value representation 一般价值。
- Fovea/Query-Bridge 第一波 `baseline_fused/query_only/query_gt_mask/query_cycle/query_fovea`: `42.94/45.26/49.16/54.67/43.77`；动态预算、策略与多损失混杂，不能否定 Query 表示本身。

## 五、必须吸收并核验的跨领域思想

不要只列论文名。请查阅可核验的原始论文/官方实现，并对每一类回答：它保留的是 tensor 行号、原始 coordinate、support interval，还是仅压缩内容？若迁移到 VideoMAE/TAD，具体在哪一层生效，是否破坏预训练兼容，是否会跨边界平均化？

至少覆盖：

1. **视觉 token pruning / merging / latent tokenization**
   - DynamicViT、EViT、A-ViT、TokenLearner、ToMe，以及视频 token/tubelet pruning。
   - 重点区分删除、合并、摘要与“保留原始时空坐标”。
2. **LLM 长上下文与稀疏 token/KV**
   - token dropping/KV eviction 后保持 original position IDs；RoPE 的绝对相位、重旋转或位置压缩问题；relative bias、ALiBi/T5 bias、StreamingLLM/H2O/EQUIP 等相关机制。
   - 重点判断“删 token 但不把位置重新压成 0..K-1”对视频 token 的可迁移性与局限。
3. **不规则采样与连续时间建模**
   - continuous/relative temporal encoding、Fourier/Time2Vec、Neural CDE/Latent ODE、irregularly sampled transformer/video transformer。
   - 不得为了形式复杂而引入 ODE；只有它能解决当前 Conv3D 首次混合问题且成本可控时才可选。
4. **稀疏视觉的坐标感知交互**
   - deformable attention、sparse query/dense support、support-aware token aggregation、原始空间/时间坐标的 gather/scatter。

请给出主要 prior-art / novelty invalidators。抽象的“token pruning + position encoding”很可能并不新；可发表创新必须落在 TAD 特有的边界支撑、H65 语义采样、首次重型时序混合与真实计算合同上。

## 六、需要你对抗性裁决的候选机制

下面只是候选，不是要求全部实现。你必须选择一个，或提出一个严格更好的单一替代。

### A. Physical-Support-Conditioned Tubeletization

Conv3D 仍接收成对 selected RGB，但 temporal mixing kernel 同时消费两帧的真实时间、间隔及 support interval；规范 uniform 输入时严格退化为原 VideoMAE PatchEmbed。需要说明如何从预训练 Conv3D 权重初始化、如何避免 gap 成为动作强度捷径。

### B. Spatial-First, Time-Aware Tubelet Aggregation

先对每个 selected RGB 独立执行空间 patch embedding，不做伪连续 temporal convolution；再用真实 timestamp/support 对相邻或局部 sparse frame tokens 做时间感知聚合。必须说明如何等价或近似复用原 `(2,16,16)` Conv3D 权重、计算是否仍与 384 帧 H65 同量级、是否破坏 VideoMAE 预训练分布。

### C. Original-Coordinate Sparse Tokens

保留每个 sparse token 的原始物理时间/position ID，不重新编号为 selected rank；在首次 token interaction 使用 continuous physical RoPE、relative bias、ALiBi-like distance 或 Fourier/gap embedding。若仍保留 Conv3D temporal kernel 2，必须解释为何它不是重复当前 SingleClock 的“作用太晚”问题。

### D. Sparse Support -> Dense Physical Query Bridge

把 K384 sparse observation tokens 作为 support，用固定的 canonical dense physical queries 读取它们；detector 在物理 query grid 上工作。必须证明 dense queries 不重新执行昂贵 VideoMAE，也不退化为当前低性能 Query-Bridge；解释边界 support、缺口、padding 和短视频。

## 七、输出必须直接解决的十个问题

1. 当前最严重的表示错误是否确实是 Conv3D 对 selected-rank 相邻帧的伪连续混合？还有没有更优先的机制性解释？
2. 视频/视觉/LLM 的哪些方法能够真实迁移，哪些只是表面类比？给出原始论文或官方代码依据。
3. 选择唯一主机制，并说明为何它优于其他三个候选。
4. 给出精确 tensor contract：RGB、frame timestamps、tubelet support、valid/padding mask、token positions、输出 token layout。
5. 给出 canonical-uniform identity 和 identity-at-init：何时必须与原 VideoMAE bit-identical，何时只允许数值等价；不允许用模糊容差掩盖结构变化。
6. 说明如何复用 VideoMAE-S 预训练权重，以及新参数的初始化、学习率、正则和防塌缩方式。特别回答为何当前 SingleClock scale 收缩到约 `-0.0018`，新机制如何避免再次被网络忽略。
7. 给出 TAD 专用边界合同：长 gap、短动作、相邻动作、padding、尾窗、变帧率、相同外观不同实例；physical decode 必须在 NMS 前。
8. 给出最小 Builder patch：允许修改的文件/符号、禁止修改的表面、focused shape/identity/gradient/coordinate tests。
9. 给出唯一最便宜 falsifier 和正式实验：统计量、阈值、停止规则、是否需要 paired whole-video bootstrap；不得重复旧基线矩阵。
10. 给出论文创新边界：什么可以 claim，什么只是借鉴；哪个 prior art 会直接使路线失去新颖性。

## 八、期望的最终结构

请以外部评审可读的中文完成：

1. 严厉审稿人攻击；
2. 跨领域方法核验与可迁移性；
3. 唯一 `CONTINUE/REVISE/PIVOT/STOP`；
4. 唯一冻结机制及公式/张量合同；
5. 最小实现与测试；
6. 最便宜真实 falsifier 与结果到 claim 边界；
7. prior-art/novelty invalidators；
8. `next_owner / next_action / dependency / expected_return_at / single_recovery`。

若你认为所有候选都不足以形成有趣、可发表且可执行的路线，请明确 `STOP`；不要用增加模块数量来掩盖基本机制不成立。若认为可以继续，优先给出最小、优雅、能在一次真实 H65-compatible 训练中裁决的方案。
