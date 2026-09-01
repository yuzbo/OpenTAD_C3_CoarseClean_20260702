# DUCA 稀疏 Token 物理时间表示 Pro 终稿：核验、吸收与项目裁决

日期：2026-08-25  
状态：`PARTIAL_ACCEPT / DESIGN_CANDIDATE / NO_EXECUTION_AUTHORITY`  
对应 DUCA revision：`b2ccfccab5b4912b59954afcc9b0364955327f7c`  
Pro 裁决：`REVISE`  
候选名：`DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v001`（PJST）

## 1. 来源与证据边界

- 用户提供的完整可见终稿已逐字归档为
  `docs/methods/reviews/2026-08-25-b2ccfcca-duca-pjst-pro-response-user-supplied-raw.md`。
- 归档文件 SHA-256：
  `d1dce144eeff2b2bc474154df948b20b82536252df1f24cc40e4d84b62a02160`；
  与原附件逐字节相同，共 32,254 bytes。
- 终稿自述 exact Project 为
  `g-p-6a796fef9a00819194024cf1de3bd697`，nonce 为
  `DUCA-H65-SPARSE-PHYSICAL-TOKEN-TRANSFER-v001-20260825`。
- 但本地正式 browser receipt 记载该次自动调用实际落入另一个 Project，状态为
  `NEEDS_ATTENTION`，没有被项目系统验收为 exact-DUCA Project 的完成回执。因此，本文件把该终稿
  分类为“用户提供的独立科学审查原文”，不回写或伪造此前失败的浏览器路由证据。
- 本轮只完成科学与代码可实现性核验。没有实现 PJST、没有 PRE_RUN、没有训练、没有产生新的
  mAP、成本或论文结论。

## 2. 我对 Pro 结论的总体判断

我**部分认可**，但不完全认可。

认可的核心判断是：H65 的 384 个非均匀选中帧在进入 VideoMAE 后，首个 temporal kernel=2 的
Conv3D tubelet 化仍把相邻 selected rank 当作等时间间隔。后置的第 0 个 Transformer 时间偏置或
proposal 物理回映，都不能撤销已经发生的二帧混合。因此，把物理时间干预前移到 PatchEmbed 内部，
比继续堆叠 post-PatchEmbed 的 SingleClock、Query 或 Bridge 更符合当前负证据。

我也认可 PJST 的三个设计优点：

1. 它精确复用 VideoMAE 的两组时间抽头，不增加可学习参数；
2. 它为 canonical uniform 输入保留原始 PatchEmbed 直接旁路，能建立严格预训练身份；
3. 它把 H65 语义非均匀选帧、固定 K=384、检测头、损失、NMS 和官方评估器固定下来，原则上能把
   下一项科学问题压缩为“首次重型时间聚合是否需要物理尺度”。

但我不接受以下强表述：

- PJST 目前只是一个合理且可证伪的表示假设，不是已被代码或实验支持的方法；
- 一次新的 30+60 端到端训练不能同时保证“selector/scout 参数相同、selected RGB 完全相同”并允许
  detector loss 继续影响 selector；
- 单 seed 的整视频 bootstrap 只刻画验证视频总体的不确定性，不刻画训练 seed 不确定性；
- `+0.50 pp`、`1.02×` 成本和短动作 `−0.50 pp` 等阈值是本次审查提出的新建议，不是项目此前已
  登记的门，不能未经项目冻结便回溯性宣称为预注册标准；
- 现有材料没有完成 prior-art 排除，因此“新颖性成立”仍是开放问题。

## 3. 与真实代码的对应关系

本节所有代码行号均绑定到干净、只读的 DUCA worktree
`E:/DeskTop/TAD/OpenTAD_DUCA_H65_FirstMixSingleClock_Critic_20260824`，其 HEAD 为
`b2ccfccab5b4912b59954afcc9b0364955327f7c`。当前项目根目录的 `a6bdc084...` 与大量未提交修改
属于错误/污染身份，只用于保存研究记忆，不能据此判断 PJST 的生产代码状态。

### 3.1 首次重型时间混合确实发生在 PatchEmbed

`opentad/models/backbones/vit_adapter.py:796-804` 建立
`kernel_size=(tubelet_size, patch_size, patch_size)` 的 Conv3D PatchEmbed；当前 H65/VideoMAE-S 的
`tubelet_size=2`。同文件 `866-968` 的真实 forward 在 `889` 行先执行
`self.patch_embed(x)[0]`，之后才在 `890-927` 行构造物理时间残差，并在 `958-968` 行把它送入第 0 个
Transformer block。这支持 Pro 的主要诊断：只在后续 attention 增加物理时间，不能修复 PatchEmbed
已经完成的 selected-rank 二帧混合。

### 3.2 H65 选中帧以 rank-packed 形式进入骨干

`pc_ot_mras_prebackbone_frame_selector.py:3230-3290` 先把时间轴展平，再按 dense indices gather，
最后恢复为长度 K 的时间轴；因此骨干看到的是按选中次序排列的 RGB，而不是带空洞的原 768 帧轴。
`backbone_wrapper.py:70-99` 随后把 `[B,T,3,clip_len,H,W]` 的片段维展平后送入 VideoMAE。
PJST 在工程上有明确插入点，但必须把全局 384 个位置的 support 先计算完，再切成 24×16；若在每个
16 帧 clip 内独立构造端点 support，会在 clip 边界制造人为物理边界。

### 3.3 物理检测坐标已有代码基础，但必须避免二次映射

`anchor_free_head.py:177-261,325-473` 已能从严格递增的物理位置构造 ActionFormer points、stride 和
regression range，并明确禁止把训练 GT 改到 selected axis。与此同时，
`opentad/models/utils/post_processing/utils.py:119-184,223-275` 仍保留 selected-axis 到 dense-axis 的
后处理转换。PJST 不应修改检测器训练语义，但实现审查必须沿真实调用链证明：proposal 在
filtering/top-k/IoU/NMS 之前恰好转为一次 physical axis，已经在 native physical head 解码的结果不能
再次按 selected rank 插值。

### 3.4 冻结权重本身不是障碍

`vit_adapter.py:1097-1112` 会冻结 PatchEmbed 参数。PJST 可以从已有权重视图构造
`W_A=W^-+W^+` 与 `W_V=W^+-W^-`，且不注册新参数；冻结状态与 optimizer membership 可以保持不变。
不过，非 uniform 分支中的加减法会改变浮点运算顺序，不能要求与原路径逐位相同。逐位身份只应通过
exact canonical uniform 的直接旁路建立。

## 4. PJST 数学设计中成立的部分

令原 temporal kernel 的两个抽头为 `W^-` 和 `W^+`。在 uniform 条件下，定义

```text
m = (x_i + x_{i+1}) / 2
v = (x_{i+1} - x_i) / 2
W_A = W^- + W^+
W_V = W^+ - W^-
```

则 `W_A*m + W_V*v` 代数上严格等于
`W^-*x_i + W^+*x_{i+1}`。因此，零阶外观项与一阶变化项的分解本身正确；原文还正确指出第二个
权重必须是 `W^+-W^-`，不能误写成退化的 `W^+-W^+`。

用 `bar_delta/delta` 缩放差分也有清楚的机制含义：它把跨越大物理间隔的外观差异从“单步局部运动”
降权，避免 VideoMAE 把相隔很远的两帧当成预训练时的相邻帧。但它依赖一个可证伪假设：局部变化可由
两个稀疏样本的差分近似。若大 gap 中存在短动作或相位反转，PJST 无法恢复未观测信息。

## 5. 需要修订的科学与实现问题

### 5.1 “纯表示归因”与“完整联合训练”存在冲突

H65 Stage-2 不是固定索引的数据管线；selector/scout、贡献蒸馏与检测损失共同训练。PJST 改变第一层
表示后，反向梯度会改变 selector 参数，进而改变之后的 selected positions 与 RGB。即使起点、seed、
代码路径相同，完整 30+60 训练也只能得到**系统级干预**，不能自动得到“同一 selected RGB 的纯表示
因果效应”。原终稿要求同时比较 selector/scout hash 和 selected positions，但没有解决这一逻辑冲突。

因此，项目不能在完整联合训练后使用“相同 selected RGB”这一强 claim，除非采取下面二者之一：

- 冻结或确定性重放 H65 selector，使 selected positions 对两臂完全相同；这得到表示归因，但不再是
  原始完整 H65 联合训练合同；
- 保持端到端联合训练，允许 selected positions 漂移，并把实验解释为“PJST 对完整 H65 系统的总效应”；
  同时报告 selection Jaccard、边界覆盖率、gap 分布和 selector 参数漂移，把选择变化视为中介变量。

不能把两种解释混在一个主实验中。

### 5.2 support 是归一化插值，不是完整的物理积分

原式

```text
m = (q_i*x_i + q_{i+1}*x_{i+1}) / (q_i + q_{i+1})
```

在 `q_i=q_{i+1}=c` 时与普通均值相同。因此它利用的是 pair 内相对 support，不保留共同放大的区间
“质量”。这不是数学错误：RGB 是点采样值，归一化可避免长 support 人为放大亮度；但论文应称其为
“归一化支撑加权外观”，不能声称完成了对不规则时间区间的严格积分。

### 5.3 “不存在 gap shortcut”需要缩窄

当 `x_i=x_{i+1}` 时，差分项为零，归一化外观项也与共同 gap 无关；这一性质成立。但当两帧内容不同，
`q` 与 `1/delta` 会改变特征幅值，网络仍可间接识别 gap。严谨表述应是：

> PJST 不注入与视觉内容无关的加性 gap token；在常量内容 pair 上几何变化不改变输出。

不能扩张为“一般情况下网络无法使用 gap”。

### 5.4 它只修复 pair 内第一次混合，不是完整时间表示

当前输入为 24 个独立 16-frame clip，每个 clip 产生 8 个 tubelet。PJST 修复每个二帧 pair 的首次
局部混合，但后续固定 rank positional embedding、跨 tubelet attention、Adapter 和检测特征仍可能把
rank distance 当成时间距离。因此 PJST 是一个必要性 falsifier，而不是整个 sparse physical-time
问题的完备解。若它失败，不能推出“物理时间无用”；若它成功，也不能推出后续所有层都已时间一致。

### 5.5 padding、重复位置和 mixed-batch identity 必须先冻结

- `delta` 要求位置严格递增；任何 duplicate/填充位置进入有效 pair 都会破坏公式。现有 selector 能记录
  duplicate rate，但正式 H65 路径必须先证明实际有效 K384 无重复。
- support 必须在全局 K384 上构造，再按原 clip 排列切片；不能在每个 clip 内重建 `[0,T_b]`。
- exact-uniform 旁路在 batch=1 下最简单。若 batch 中同时存在 uniform 和 non-uniform 样本，按样本拆分
  再拼接可能改变卷积执行与逐位身份。合同需明确正式训练 batch=1，或把 identity 定义为整批 canonical
  uniform 才进入旧路径。
- 变帧率视频只有在可信 PTS 存在时才能使用 seconds；本轮 THUMOS14/CFR 主张不能外推到 VFR。

### 5.6 新颖性尚未闭合

已有工作已经覆盖了相邻构件：TDN 显式建模短期与长期时间差分；TAdaConv 按时间上下文校准卷积并强调
预训练兼容；Run-Length Tokenization 为压缩视频 token 保存时间长度；TE-TAD 已在 TAD 中使用实际
时间线坐标。因此，“时间差分”“时间自适应卷积”“token 长度”“真实时间坐标”都不能单独构成贡献。

可继续核验的窄 claim 只能是：

> 面向语义非均匀物理帧 acquisition，在 pretrained VideoMAE 的首次二帧 tubelet 化中，以零新增参数
> 进行归一化支撑外观与物理间隔变化分解，并对 canonical uniform 输入保持原 PatchEmbed 严格恒等，
> 再在 unchanged high-IoU TAD 合同下进行验证。

### 5.7 原终稿的 95% bootstrap 区间有明确索引错误

原终稿规定对 10,000 个有序 bootstrap 样本取第 500 与第 9500 个值作为“95% percentile bounds”。
这对应约 5% 与 95% 分位数，只形成约 90% 的中心区间。若冻结双侧 95% percentile interval，应使用
2.5% 与 97.5% 分位数；实现可调用明确冻结插值规则的 quantile API，或按预先声明的 0/1-based 索引
约定选择约第 250 与第 9750 个顺序统计量。提交前必须把 quantile 定义、索引基准与插值规则写入合同，
不能沿用 500/9500，也不能在看到结果后修改。

### 5.8 数值通过门尚无功效依据

`+0.50 pp`、全栈时延/显存 `1.02×`、short-action `−0.50 pp` 等门具有工程直觉，但当前材料没有提供
最小可检测效应、方差或资源预算推导。它们可作为审稿人建议，不能直接升级为项目既有预注册门。更稳妥
的做法是先冻结主统计量、容许误差和决定后续行动的规则，再由历史整视频方差或预算约束给出阈值依据。
此外，单 seed 的视频级 bootstrap 只回答验证视频总体的不确定性，不能覆盖训练随机性。

### 5.9 support-weighted 外观项引入了不必要的第二个科学变量

原 PJST 同时改变两件事：以 `bar_delta/delta` 重标定差分项，并以 `q_i` 改写零阶外观平均。后者是新的
quadrature 假设；它不由“首次混合应看到真实时间尺度”唯一推出，还会使一般非恒定 pair 通过特征幅值
间接携带 gap 信息。若下一步目标是最便宜地判断首次二帧混合是否需要物理尺度，应先移除这一混杂：

```text
m = (x_i + x_{i+1}) / 2
v = (bar_delta / delta) * (x_{i+1} - x_i) / 2
z = (W^- + W^+) * m + (W^+ - W^-) * v + b
```

该 `PJST-v0.2-derivative-only` 保留原外观零阶项，只校正物理变化率；support/Voronoi 区间仅作为审计
元数据，不进入 forward。它不是已经冻结或获执行授权的新方法，而是本次核验后推荐的最小、可归因修订。
若这一门通过，再把 support-weighted zero-order term 作为后续独立消融，而不能与物理差分首轮合并。

## 6. 吸收后的优化方向：PJST-v0.2 证据链

### Gate A：现有证据闭合，不训练

1. 先完成既有 SingleClock/RankPack/TrueTime artifact 的身份与终态核算；不得把运行中、部分或跨运行根
   的数值写成新机制结论。
2. 对既有 RankPack/TrueTime prediction JSON 做整视频配对 bootstrap，只回答 `+0.6208` 的视频总体
   不确定性；明确它不是训练 seed 置信区间，也不是 PJST 直接证据。
3. 固定 canonical H65 OFF checkpoint、Stage-1 起点、official evaluator、split、NMS 和 30+60 更新账本。

### Gate B：静态实现与坐标合同

只允许在干净、可验证的 DUCA worktree 上实现；当前项目根 HEAD 为 SparseHead 污染身份
`a6bdc084...` 且工作区很脏，不能作为 PJST 生产基座。应从已核验 DUCA revision `b2ccfcca...` 建立新的
clean worktree，再由 Builder 提交最小计划与 patch。

最小实现面保持为：

- `opentad/models/utils/temporal_grid.py`：全局 K384 actual/canonical pair interval 与 valid metadata；
- `opentad/models/backbones/backbone_wrapper.py`：只传递时间、pair interval 和 valid mask；
- `opentad/models/backbones/vit_adapter.py`：PatchEmbed 内部 derivative-only PJST 路径；
- 一份 PJST ON 配置与 focused tests。

不修改 selector、ASFormer、ActionFormer 训练语义、loss、NMS、dynamic-K、Query、UVT 或训练日程。

静态验收除 Pro 的十项测试外，增加三项：

1. actual/canonical pair metadata 必须在全局 K384 上构造后再切 24×16；
2. 捕获真实调用链，证明 native physical proposal 不会在 post-processing 二次映射；
3. 对有效位置执行 duplicate-free、strict-increasing、clip-boundary interval 一致性，以及
   derivative-only 公式、constant-pair invariance 与 canonical-uniform byte identity 检查。

### Gate C：先做同输入兼容性 falsifier，再决定唯一正式训练

用已经存在的 H65 terminal checkpoint，对完全相同的逐窗口 selected RGB/positions 执行一次 PJST ON
只读重推理。这个检查不用于声称性能有效，只回答两个问题：

- PJST 是否造成灾难性表示错配或坐标错误；
- 高 IoU、短动作和大 gap strata 的变化方向是否与机制预测一致。

它不能代替训练，也不能用来选择 checkpoint。若静态或兼容性门失败，停止，不投入 30+60 训练。

### Gate D：唯一正式 falsifier 采用纯表示归因

本轮问题是“首次 tubelet mixing 是否因错误时间尺度损失性能”，因此优先采用固定/重放 selector 的纯表示
归因，而不是让表示梯度同时改变采样决策。OFF 与 derivative-only PJST 必须从同一 Stage-1 起点出发，
接收逐窗口完全相同的 selected positions、RGB、valid mask 与 K384，并使用相同训练更新、检测器、损失、
NMS、official evaluator 和 terminal-EMA 规则。若既有 OFF artifact 不能证明这一训练期同输入合同，则不能
把它当作该归因实验的 matched OFF；需要另行冻结是否允许一对 matched Stage-2 作业，不能用“同 seed”
代替同输入证明。

结果报告：Avg-mAP、mAP@0.6、mAP@0.7、short-action/相邻动作/gap strata、executed RGB、VideoMAE
token/MAC、全栈时延与显存。该实验只能支持“固定 H65 acquisition 下的表示效应”。只有它通过且另获执行
授权后，才考虑一条允许 selector 漂移的端到端 PJST 总效应实验，并报告 selection Jaccard、边界覆盖率、
gap 分布与 selector 参数漂移；二者不得在同一结果中混称。

### Gate E：统计与停止规则

- 单 seed paired whole-video bootstrap 只用于视频总体 CI；若通过单 seed 工程门，仍需后续独立 seeds 才能
  形成稳定性结论。
- 双侧 95% percentile CI 必须冻结为 2.5%/97.5% quantiles；禁止把第 500/9500 顺序统计量称为 95%
  中心区间。
- `+0.50 pp`、成本与短动作门只有在作业提交前由历史方差、最小可检测效应或资源预算给出依据并写入
  独立 frozen contract 后才生效；不能在看到结果后修改。
- 任一身份、坐标、official evaluator、K384、terminal-EMA 或 PRE_RUN 门失败，均不得读 mAP。
- 若 PJST 正式 falsifier 失败，停止继续堆叠 SingleClock/Bridge/Query/gap gate；转回 H65 selector 质量、
  训练成熟度和 selected-rank 输入合同的证据分析。

## 7. 当前项目裁决

1. 将 PJST 保留为 `designed_candidate`，不是 `implemented` 或 `empirically_supported`。
2. 接受“first heavy mixing 必须看到物理时间”作为当前最强表示假设。
3. 不把用户提供的 Pro 终稿当作已经完成的 exact-Project browser receipt，也不由此自动启动 Builder 或训练。
4. 在执行前必须先冻结 PJST-v0.2 的因果口径：
   `system-level end-to-end` 或 `fixed-selector representation attribution`，二者不可混称。
5. 本次核验推荐把正式首门缩为 derivative-only + fixed-selector representation attribution；support-weighted
   zero-order term 与端到端 selector mediation 均后置，避免一次实验同时改变三个机制。
6. 当前 `PAPER_PROGRESS.md` 不更新：没有新的实现、PRE_RUN、实验或真实性能变化。

## 8. 相关外部工作

- TDN: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html
- TAdaConv: https://arxiv.org/abs/2110.06178
- Run-Length Tokenization: https://arxiv.org/abs/2411.05222
- TE-TAD: https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html
