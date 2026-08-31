---
type: decision_history
updated: 2026-08-25
---

## 2026-08-25 PJST 因果口径等待唯一科学裁决

现有代码核验支持把物理时间干预从 PatchEmbed 后的 SingleClock 前移到 VideoMAE 首次二帧
tubelet 混合；PJST 以原 Conv3D 两个时间抽头构造归一化支撑外观项与物理间隔变化项，并对 exact
canonical-uniform 输入直接旁路原 PatchEmbed。它当前只是 `designed_candidate`，尚无实现或效能证据。

执行前存在一个会改变实验 estimand 的真实分叉：保持 H65 Stage-2 端到端联合训练会允许 selector
因表示梯度而漂移，只能估计 PJST 对完整系统的总效应；冻结或重放 selector 才能估计同一 selected
RGB 的纯表示效应，但会改变原始 H65 训练合同，并可能要求重新训练 matched frozen-selector OFF。
两种口径不能用同一个实验混称。项目不自行选择，已将其压缩为唯一 exact-DUCA Pro 裁决请求；在
该裁决前不启动 Builder、PRE_RUN 或训练。

SingleClock 的统一终结作业已 `TIMEOUT`。四个 ON/gate-zero family 存在，但 OFF 配对终结器、
bootstrap 与两项硬身份未共同闭合，所以无正式 PASS/KILL；该终态只作为 PJST 前的负向预警与
证据缺口保存，不要求重复 SingleClock 训练。

## 2026-08-24 H65 60轮压缩终止与只读训练动力学归因

在成熟30轮 Stage-1 的同一起点上，两条只改变 Stage-2 学习率尾部的正式归因实验均完成。AM-RPCH25 的终态 EMA Avg-mAP/mAP@0.7 为 `63.22/41.25`，LongCosine-H6000 为 `63.56/41.01`，均低于预注册恢复门 `64.6257/42.8137`。LongCosine 保留约高18.33%的累计相对学习率面积和更高终点因子，但总体性能只小幅改善，高 IoU 反而不占优；两臂 epoch 29 Avg-mAP 又都低于 epoch 24，因而不满足唯一延长分支的上升条件。

项目接受 exact DUCA Project 的终态 Pro 裁决 `STOP_60_EPOCH_COMPRESSION`：不再启动第三个30+30调度器，不提高峰值学习率，不做无梯度证据的参数组学习率微调，不执行1000-update延长，也不以中间 checkpoint 挽救。历史30+60继续作为当前 H65 训练参考。这一裁决只终止已测试的60轮压缩族，不是否定H65语义间接选帧，也不证明所有60轮轨迹在理论上不可能成功。

下一阶段只利用既有 artifacts 做 matched-successful-update 尸检：比较实际学习率与模块参数位移、课程/feedback 时钟、selector 动态、terminal online/EMA 和高 IoU 边界误差。只有尸检发现历史轨迹不存在、压缩轨迹可重复出现的单一机制签名，未来才可另立新身份提出一个机制实验；当前不创建训练配置或 Slurm 作业。

## 2026-07-20 CARA 固定骨架撤回与物理全局可行集修订

对 `4ce69c8` 的第二轮 exact-commit Pro 审核已逐字归档并做独立源码与数学
复核。项目接受 `HOLD_AND_REVISE_FAMILY`：不实现固定
`G=3, 192 scaffold + 192 residual`，先把主 ceiling 修订为真实物理坐标上的
global exact-K/max-interval family D。coverage scaffold 只允许作为一个已选集合
的事后规范分解，不再预先占用固定预算。

关键反例已复算。正式数据网格每个 dense index 对应 4 个 source-frame indices。
若最大间隔明确为 15 个原始解码帧，则离散有效 cap 为 12；exact uniform 可行，
但任意固定 scaffold 至少 255 点，若还要求是 exact-uniform 子集则至少 382 点。
因此只剩 2 个 residual，无法表达原先想要的跨区域预算集中。相反，dense-hole
`G=3` 的 192 点结论属于另一套坐标合同，不能与物理 15 帧混写。

项目不完全照收回复中的规格：此前“10/15 帧”仍未明确是 dense index、原始解码帧
还是秒；附件代码尚未集成，GT oracle 也缺少其声明的距离、短动作和背景目标；统计
门槛尚未明确 paired seed variance 与多重比较；当前 `val/test` 又共同使用
THUMOS validation subset。物理单位、训练侧冻结集和统计合同明确之前，不运行
CARA full train。CellCF 主方法身份保持终止，DUCA broader route 保持
`REDESIGN`，C3/C4/C7 不变且未证实。

## 2026-07-19 CellCF 主方法终止与 DUCA 可行集合重设计

对模型提交 `1642f26` 和成本证据提交 `4ce69c8` 的外审已完整归档并做独立
源码/数学复核。当前 CellCF 的 one-per-uniform-cell 可行集合不能跨区域转移预算；
在 `T=768,K=384` 时只允许每个均匀锚点在本 cell 内最多移动一个 dense index。
同时，实际采集位置与检测器使用的均匀 anchor 坐标不一致，正式配置使用
selected-axis GT remap；其检测监督是 detached hard counterfactual utility，
不是直接检测损失梯度。

结合 matched seed-0 的 `uniform=63.8594`、`transition=64.2755`、
`CellCF=64.0610`，项目永久终止 CellCF 的“边界自适应预算分配主方法”身份，
仅保留为 uniform phase/content correction 诊断对照。这不是终止粗动作状态变化
驱动的离线 TAD 间接选帧目标。

下一候选 `DUCA-CARA` 采用覆盖骨架加全局 residual quota，并要求实际位置直接进入
仓库已有 physical-grid ActionFormer。它当前仅为 `discussed`。先实现 exact
allocation-family ceiling 和坐标/成本门禁，再决定是否小规模训练。`G=3`、
`192+192`、全部损失权重、梯度 detach 方案、训练轮数和数值阈值均未被当作事实
接受；Dynamic MUST 继续冻结，C3/C4/C7 状态不变。

## 2026-07-17 CellCF 双训练协议与证据代码裁决

正在运行的 `1642f26` 三臂实验保持冻结，仍用于回答“在充分且匹配的
13,200 次更新下，transition 与 CellCF 是否优于 exact-uniform”。它不因
后处理工具升级而改变提交身份。`2a0f848` 只负责把 `exposure132` 与
`official60` 明确分开，并为收敛曲线、训练 GPU 时、完整推理时延和
break-even 提供可重放证据。只有当前 132 轮终点通过 GO 门槛，才允许启动
同提交、三臂匹配的 official-60 正式训练；若终点失败，则先停止扩展，不用
更多轮数或新骨干掩盖核心假设失败。

# 路线演化与选择理由

## 1. C3 粗分类出发点

最初洞察：低成本模型即使不能直接精确选帧，也可能可靠地区分动作/背景并暴露
状态变化。选择“粗分类 → 间接选择”，而不是让小模型直接承担 TAD。

保留原因：动作性提供低成本候选证据。限制：动作内部高分不等于边界有用。

## 2. PAction 与严格 ledger

建立 train-time p_action 监督、strict budget、no-leak ledger 和 AdaTAD 消费链路。
它证明工程路径可行，并成为固定预算安全锚点。

未选为最终方法：多阶段、detector-unaware、硬 gap decoder，创新性与联合优化不足。

## 3. GAS-VT

尝试学习 frame value、边界覆盖、预算与 gap，再生成 value-transport ledger。

降级理由：apply-time 不是忠实 sequential value transport；训练/应用特征存在风险；
多重覆盖损失和 repair 可能推向均匀；PAction 的简单策略反而更强。保留为诊断 baseline。

## 4. Detector-aware teacher 与 TrueTime

为了让选择服务 TAD，引入 train-only dense detector utility、selected-axis 到原时间映射
和 detector-loss gradient proof。

未作为最终三阶段方案：teacher 预提取仍然多阶段；toy gradient 和 wrapper 不足以证明
联合检测收益；selected-axis 的不等物理间隔语义尚未解决。

## 5. Lattice、move25/move50 与 learned radius

为了避免大 gap，用 uniform scaffold/local replacement 和边界附近膨胀保护覆盖。

降级理由：它是工程启发式；repair/膨胀消耗预算并可掩盖分数偏移；move 分析显示聚集
存在但位置仍可能偏离边界。只保留为几何诊断和对照。

## 6. DUCA 插件化

从 ledger 转向 detector 前 forward 内生成选择：coarse probe、selector、hard selected
positions、official detector backend、teacher-free inference。固定 K 是归因锚点，MUST
尝试动态预算。

选择理由：统一模型接口，允许 detector 梯度反馈，避免测试时 ledger/cache。

## 7. X3D / SlowFast frozen prior

尝试 train-free 视频动作先验，验证视频预训练特征是否比图像 MobileNet 更懂动作。

降级理由：密集视频推理过慢，可能吞掉 heavy backbone 节省；Kinetics 预训练还带类别
先验和 overlap 风险。仅作为 frozen-prior diagnostic/upper baseline。

## 8. Transition/boundary-first 修正

move 分析和早期 DUCA 低性能表明 actionness coverage 再次主导。于是 selector 改为读取
`delta_p_action`、绝对变化、不确定性和粗分类隐藏特征，以 transition/boundary/utility
优先，actionness 降为小权重辅助。

这是当前必须保留的设计初心。

## 9. Progressive joint 与 structured bridge

为协调 probe、selector、detector，采用 progressive joint schedule：先稳定 coarse/boundary
目标，再逐步打开 detector gradient；detector loss 始终训练 backend。structured
zero-forward bridge 保持 hard forward，同时向 selector 传递近似梯度。

未闭环问题：梯度非零不等于 hard utility 对齐，必须做 finite-difference one-swap 审计。

## 10. Offline full-window 语义纠正

严厉审查曾把目标误认为 causal online acquisition，提出 prefix invariance/CFPA。用户明确
纠正：项目从未要求流式因果，完整窗口可见是允许的。CFPA 只保留给未来 streaming 版本。

## 11. 70aa069 冻结与 a5e1774 审计

`70aa069` 修复 full-window structured joint training 并启动 fixed-384 full run。
ResearchClaw 审查指出总成本、hard/soft utility 与 selected-axis geometry 是决定性门槛。
结论不是立即停止，也不是继续堆模块，而是冻结候选并裁决。

`a5e1774` 修复完整成本统计和官方 AdaTAD 表述：base/head 配置一致，但 detector 源码、
输入长度、GT 坐标与后映射并非完全官方不变。

## 12. 当前选择

主科学问题保持“任务感知时序去冗余并保护高 tIoU”。当前先完成 DUCA fixed-384 的
决定性证据；MUST、physical-grid、X3D 主方法和更多 loss 暂停。ChronoTransport 与
PhysTime 是并行新假设，必须独立记账，不能用来改写 DUCA 结果。

## 13. ChronoTransport 正式负结果与查新裁决

外部 ResearchClaw 审查建议把论文中心从 pre-backbone 选帧转向 backbone 内部计算
复用，于 2026-07-10 形成 ChronoTransport。其正确任务语义是离线全窗口 TAD；缓存
不等于在线，旧 `p_action/Δp_action` 不得进入主路径。16 帧是 AdaTAD clip 容器，
内部时间格来自 2 帧 tubelet；v1 实际调度为 48 chunks × 3 layer groups。

正式 Stage-B seed 3407 完成工程闭环，但 P3 科学 gate 失败。失败不是“训练没有跑”，
而是风险尺度错误且 feature transport 优势不稳定，因此 Stage C/P5 继续锁定。

2026-07-11 查新发现 Eventful Transformers、ResidualViT、Progressive Block Drop、
Adaptive Temporal Refinement、SCOPE 与 Conformal Thinking 分别覆盖变化 token 重算、
dense residual encoding、TAD 深度压缩/连续深度、三模 cache-predict-recompute 和风险
控制。ChronoTransport 只能守住 TAD-specific dense physical-time lattice + structured
localization regret + measured full-stack cost 的组合 delta，综合新颖性暂评 `4.5/10`。

当前裁决：路线暂停，只允许一次有界 P3 修复；若再次失败则冻结为 baseline。新候选
优先级为 Boundary-Adaptive Temporal Multigrid > Counterfactual Value-of-Computation
> 高风险 Spectral Innovation Operator。

## 14. DUCA 联合训练优化提案（designed，未实现）

针对 70aa 的 selector chance-level loss、5940/13080 step 不匹配、随机 coarse stem 和
hard-forward/soft-backward 偏差，首选方案不是恢复三阶段训练，而是单模型、单 checkpoint
的连续同伦式联合训练。统一 structured decoder 从均匀可行 score prior 平滑过渡到完全
learned score；detector-gradient bridge 同步按 optimizer step 逐渐打开，最终训练与推理
均只使用 learned hard selection。

梯度按模块路由：detector loss 始终训练 detector；binary actionness loss 只直接训练 coarse
head/trunk；boundary/gap 训练 selector；detector 梯度进入 selector 与 coarse temporal trunk，
但不直接污染 actionness head，并在共享 trunk 上做 detector-priority conflict projection。
hard MAP 与 backward marginals 必须来自同一个 exact-K/max-gap 可行集合，替换当前 dense
soft-slot surrogate。coarse 架构不再预先锁定为 MobileNetV3 + ASFormer：默认低成本候选
是通用预训练 MobileNetV3 spatial encoder 直接产生逐帧 hidden/actionness，由 selector
承担时间建模；若使用官方 ASFormer，则它必须替换而不是叠加现有 selector temporal
encoder，作为共享 temporal trunk，并通过 full-stack 成本/边界收益消融后才能晋升。
两种候选均不加载 THUMOS 分离 coarse checkpoint，在一次 TAD 训练中协同更新。

立即执行顺序仍受证据门槛约束：先匹配 13080 optimizer steps、effective K 和 uniform
baseline；再做冻结选择重训、one-swap gradient alignment 和 geometry control。fixed-384
未超过 matched uniform 前，不开放 MUST dynamic，也不新增 selector head。selected-axis
若被证实伤害 high tIoU，仅靠优化器无法修复，需采用 post-backbone physical-time regrid
或其他显式时间几何方案。

## 15. Joint ASFormer 优先修训练而非换 MobileNet（designed，未实现）

代码审计确认 standalone probe 与 DUCA joint path 都可构造同一个
`C3OfficialActionSegmentationProbe`。standalone 每步只优化未缩放 binary BCE，并按验证
指标保存最佳 checkpoint；joint 默认 actionness 权重仅 0.05，同时承受 boundary、budget
与尚未证明 hard-utility 对齐的 detector surrogate，且总 optimizer exposure 只有历史
分离训练的 45.4%。因此分离 ASFormer 更强首先归因于干净监督、稳定输入、更多更新和
checkpoint selection，不应直接归因于缺少 MobileNet。

下一版首选保持相同两层 spatial stem + 官方 ASFormer 做受控修复：匹配 optimizer steps；
actionness head 仅接 binary supervision；shared ASFormer trunk 对 action/boundary/detector
梯度做冲突投影；detector 梯度按 step 连续打开；hard MAP 与 soft marginals 来自同一
exact-K/max-gap 可行集合。官方 ASFormer 应成为共享 temporal trunk，避免再叠加一个
功能重复的 selector temporal encoder。MobileNetV3 不是必需项，只在修复后 coarse
AUROC/AUPRC 仍显著低于 standalone 时作为 spatial-front-end cost/accuracy 消融。

## 16. 间接边界初心与职责纠正（designed，未实现）

纠正上一版“ASFormer shared trunk 直接挂 boundary head”的歧义。主方法 coarse probe
只能学习 binary action/background state，输出 `p_action` 与 coarse state embedding；它
不得直接从 RGB/absolute hidden 预测 GT start/end。部署时的边界证据必须由
`delta logits / abs delta / entropy / uncertainty change / delta hidden` 等状态变化量产生。

GT boundary 只在训练期监督 selector 的 transition-utility ranking、boundary-neighborhood
coverage 和 detector-aware selection，不得作为推理输入，也不得把 coarse probe 训练成
mini-TAD boundary detector。selector 可观察 coarse 语义，但主路径应使用 state-change
representation（特别是 `delta hidden`），避免 unrestricted absolute hidden 绕过间接机制。
因此正确职责是 `ASFormer action-state probe -> explicit transition descriptor -> indirect
transition-utility selector -> exact-K/max-gap decoder -> TAD detector`。

## 17. ChronoTransport Pro 复核：先修规格，不先写代码

基于 `b74101d` 的 Pro 红队裁决为 `REVISE_SPEC_BEFORE_CODE`。它没有推翻一次 bounded
appeal 的资格，但确认原规格不能原样执行：Gate 1 的集合 oracle 只能证明 headroom，
不能单独证明 input dependence；Gate 3 的双边 coverage 上限会在小样本下制造纯采样
假失败；candidate rows 不能 pooled 成 Spearman；evaluation-best static 只能 diagnostic；
阶段 p50 不能相加成 full-stack p50。

当前选择 Route B：在不改变 window quantile head、seeds 3407/3408/3409、candidate
library、140 successful optimizer updates、quantile、epsilon 和 Gate 1/2/4 数值门槛的
前提下，形成 `CT-P3R-3S-r1`。r1 使用 window candidate-vector ranking、unique-window
cluster bootstrap、coverage≥0.85 单边门、OVERCOVERED 诊断、candidate 自身完整 total
cost、fit/calibration 冻结 comparator、dense safety budget-violation 显式记账和
successful-update exposure ledger。

Pro 环境看不到本地源码，因此其 optimizer LR=0、packed route、selected-axis remap、
loss normalizer 与 Stage C runner 风险必须由本地逐文件审计确认。review 中引用的
generic patch/测试仅有 sandbox 链接与 SHA，未随附件提供，不能当成已获得或已集成代码。
在 r1 新 SHA 经用户复核前，禁止 profiler、Gate 1、新 seed 训练或 Stage C。

## 17. Pro HOLD 与 Transition-Only 唯一上诉版本

2026-07-11 Pro 审查确认 current a5e hidden 实为 pre-ASFormer stem feature，absolute hidden
与 GT-supervised start/end/context/utility heads 已把整体 selector 变成 direct boundary
mini-localizer。审查同时纠正：current hard Viterbi 与 soft marginals 已来自同一个
exact-K/max-gap DP，可行族同构不是主要错误；缺口是 hard utility 方向未被 one-swap 证明。

冻结 `a5e1774` 为 direct-boundary joint baseline。下一版只允许 Shared-ASFormer
Transition-Only：暴露真实 ASFormer encoder state，只从 delta hidden/logit/entropy 等变化量
产生单一 transition score，删除 absolute/raw/direct heads，采用 protected gradient routing
和连续 score homotopy。MobileNet、MUST 与新增 heads 在 fixed-384 matched gates 前继续冻结。

## 18. `0ea4e15` Pro HOLD 与 DUCA-FSU 条件候选

2026-07-13 exact-commit 审计确认 transition-only uniform fix 本身成立，但 direct/legacy
stable route 仍有 midpoint 残留；旧 P0 矩阵全部只能作为诊断。审计进一步指出 current
score homotopy 会放大微弱噪声、raw-pixel bridge 未与 hard replacement utility 对齐、
selected-axis 内部几何和 dense decode/H2D 成本仍未闭环。

reviewer 唯一推荐是 DUCA-FSU：用训练期 feasible hard one-swap detector gain 监督
transition utility difference，删除 soft RGB bridge，并以 physical-time reconstruction
恢复 detector 的等物理时间轴。项目接受它作为值得 falsify 的条件候选，但不接受其为
既定最终模型；尤其 reconstruction 必须面对 PhysTime v1 负结果，FSU 也必须先证明
one-swap detector gain 可由 additive `u_t-u_s` 预测。当前合法动作是 P0 机制审计，
不是继续 beta 调参或直接启动论文 full matrix。

## 19. `1fc7037` REDESIGN 与 DUCA-CellCF 有界最终申诉

2026-07-13 新审查结合 selection-quality 诊断，把 DUCA-FSU 进一步收缩为
DUCA-CellCF：exact-uniform anchors 划分保序 cells，每个 cell 恰选一帧，只学习
uniform 的局部位移；训练期以 EMA detector 对少量 hard cell alternatives 的真实
`cls+reg` loss 蒸馏 preference。当前 global structured top-k、`G=15` 和
`structured_zero_forward` 不再允许通过调权重继续承担最终主方法。

项目基本接受该收缩方向，因为它把 coverage 变成可行集不变量，并让 utility target
与推理时可执行的 hard action 对齐；但不接受它已经正确或最终。固定 detector anchor
可能把 `s_j` 观测错误标成 `u_j` 时间，per-cell utility 也可能无法表达跨 cell 非加性。
因此 CellCF 状态仅为 `discussed/design proposed`。先修 uniform/target/coverage/diagnostic
四项确定性问题，再过 coarse、pure-vs-compound、hard one-swap、coverage 和 geometry
机制门；只有这些门通过才允许 same-commit fixed-384 pilot。pilot 不优于 exact uniform
即终止 DUCA 主方法，不解锁 MUST、X3D/SlowFast、多 detector 或新 selector loss。

## 20. 空间 Zoom 路线：值得验证，但不原样搬移 Uni-AdaFocus

用户提出放弃选帧、转向帧内 ROI 裁剪。当前判断是：从问题结构看，dense-time spatial
zoom 比继续 aggressive frame deletion 更契合高 tIoU TAD，因为它保留全部时间位置，
可以直接绕开局部 coverage、max-gap 和 selected-axis 几何三类已观察到的失败；但它只
是候选假设，尚未替代 DUCA-CellCF 的有界裁决。

Uni-AdaFocus 的轻量全局观察、高分辨率局部分支、时间平滑 crop 与全局特征复用值得
借鉴，但其主要证据是视频分类。2026 年 AdaSpot 已把相似范式用于 Precise Event
Spotting，并发现 training-free saliency ROI 比 learnable crop 更稳定，因此直接移植
Uni-AdaFocus/AdaSpot 到 AdaTAD 只能作为 baseline，论文贡献不足。

路线先经过两个 kill gate：dense 224/256 必须相对当前 dense 160 在 mAP@0.7/短动作上
存在可靠 headroom；oracle ROI 必须在匹配 heavy-token 成本下接近 dense high-resolution。
通过后才设计 TAD-specific 方法，优先比较稳定 saliency zoom、boundary-risk-conditioned
ROI scale/count 和 detector-counterfactual ROI-tube utility。完整成本必须包含高分辨率
decode/H2D 与 scout/crop/fusion，不能只计算局部 backbone FLOPs。状态为 `discussed`。

## 21. 空间 Zoom Pro HOLD：只解锁 S1 代码

2026-07-13 Pro review 给出 `HOLD`：冻结 learned temporal frame selection 的继续扩展，
但保留 exact-uniform/random/periodic 和历史 DUCA 为基线；空间路线必须先经过 S1 dense
160/224/256 matched headroom 与 S2 equal-total-cost ROI sufficiency。项目接受这一顺序、
AdaSpot novelty collision、规则 768 点 detector grid 与完整成本记账。

项目不全盘接受 reviewer 的 DART-Zoom 细节。80px scout、J=16、K<=2、96/112、固定
loss 权重、EMA cadence、三 seed 数值门和 15% latency threshold 都仍是 proposed；S2 的
无 GT dense-teacher oracle 仍是 privileged diagnostic，只能在冻结 gate split 使用，不能
反复查看 official test。branch `codex/chronotransport-pro-review` 也已由 `git ls-remote`
确认公开可见，纠正 reviewer 的可见性判断。

当前状态升级为 `designed at gate level`：立即只允许实现 S1 configs、config-diff validator、
shape/memory precheck、frozen manifests、统计与 full-stack profiler。S1 未 GO 不写 S2，
S2 未 GO 不写 DART-Zoom。完整模型仍为 `HOLD_FULL_MODEL`，没有实验正在运行。

## 22. `7525efb` Round-2 REDESIGN：CellCF 升为 designed

第二轮方法/论文 Pro 裁决选择 exact-uniform anchored local-cell deformation 作为 DUCA
唯一有界申诉：每格恰选一帧，step 0 显式回到 exact uniform；只从 deploy-visible state
changes 学局部 preference；训练期用不同 cells 的 detached hard flips 与 weighted signed
logistic 对齐官方 detector 策略效用。项目接受 coverage 成为结构不变量、删除 global
homotopy/Gram proximal、先 gate 后 seed-0 的核心方向。

项目不完全接受回复细节：TAPS 被误写为 TAPOS；detached utility 不能继续冒充 direct-
gradient C4；local-cell 只能主张 uniform residual learning；固定 loss/schedule/阈值仍是
proposal；单 seed 可停止当前配置但不是所有 DUCA 假设的科学反证；direct-boundary
attribution 仅延期，若最终保留间接机制主张仍需补做。

因此 CellCF 从 `discussed` 升为 `designed`，但没有代码、gate、pilot、mAP 或成本证据。
唯一允许的下一动作是实现 local-cell/logistic 及 focused tests；在 synthetic、real-loader
CUDA 和 forced-overflow DDP pilot 全通过前，不得启动 full train或解锁此前冻结扩展。

## 23. CellCF 训练预算裁决

当前 `epoch_131.pth` 实际对应 132 个 epoch 和 13,200 次成功更新。采用
这一长度的原始目的，是匹配历史分离训练约 13,080 次更新，排除旧 DUCA
仅 5,940 次更新造成的欠训练混杂。因此，正在运行的三臂正式实验继续按冻结
协议完成，不中途缩短，也不使用中间检查点挑选结果。

但 132 epoch 只被定义为“充分训练的匹配诊断”，不自动成为论文主训练方案。
仓库 AdaTAD 基础协议为 60 epoch；若 CellCF 通过当前终点门，必须再进行
same-commit 的 60-epoch exact-uniform、transition-beta0、CellCF 严格匹配，
并报告成功更新数、单模型 GPU 小时、峰值显存、反事实监督额外开销、完整推理
时延及训练成本与推理节省的收支平衡点。

现有 epoch 59/89/131 只允许作为预先声明的收敛曲线节点。epoch 59 使用的是
132-epoch 学习率日程，不能冒充官方 60-epoch 训练，也不能用于事后挑选最佳
checkpoint。若 CellCF 只在 132 epoch 获益，而在 60 epoch 或等计算预算下
没有优势，则高效训练主张不成立，132-epoch 结果只能进入充分训练或收敛性分析。

## 24. Protected-E2E REVISE：转向同一物理可行域的 hard/soft DAG

2026-07-20 Pro 裁决读取 prompt commit `280631a` 后给出 `REVISE`。项目接受其
核心方法判断：受保护 detector gradient 不能继续建立在 detached hard positions
周围的局部 soft bridge 上；hard Viterbi 与 soft slot marginals 必须来自同一个
physical exact-K DAG，且 GT/target/decode/NMS/evaluator 必须使用 dense/native
physical coordinate。主臂只允许 detector loss 更新 selector adapter/head，rho
消融只开放最后一个 ASFormer 时序块。

项目没有机械接受审查中的全部事实。它没有看到后续 `0477c55` 至 `b3222af`
实现，所以“尚无 protected 实现/真实 detector 梯度为零”已过时；Job `1176948`
已在真实 full model 上通过主臂与 rho 臂 P1/P2 connectivity/ownership，但 P3
因错误要求旧 manifest 的不存在字段而在数值审计前停止。该旧实现仍因 local
surrogate、candidate-hole、selected-axis、rho=0.05、bridge=0.25 ramp 和小规模 P3
而不符合新裁决，禁止运行 official-60。

审查提出的 seed 3407、rho 0.01、48-window/576-swap P3 可冻结为本次实现合同。
`99 steps/epoch = 5940 updates` 暂不接受为事实：P0 必须从 exact loader manifest
推导；历史 200-video/batch-2 正式运行是 100 steps/epoch。physical-grid 虽为必要
坐标修复，也必须针对历史短动作 positive-support 退化通过独立 gate。状态保持
`designed`，C3/C4 与论文准备度不变。

## 25. DUCA 最终模型与论文闭环冻结

2026-07-22 将散落在 V8、Oracle、Uni-AdaFocus 与 Pro 审查记录中的结论收束为
`research-wiki/duca_final_model_contract.md`。最终产品选择 strict fixed-budget
offline-TAD pre-backbone acquisition，而不是 post-backbone fusion、新 detector、Online
TAD、local-cell 或 dynamic budget。

选择理由是：历史 Oracle 证明有价值的分布是边界中心及其双侧微簇加剩余全局覆盖；
V5/V8 负证据则显示 Gaussian mass、direct bridge、homotopy 和 companion 尚未把粗状态
变化转成更好的 detector positions。最终结构因此保留 V8 的全局 exact-K/max-hole DP，
只补齐 G23 的 transition-center、bilateral burst、quota saturation、endpoint fairness、
overlap deduplication 和 residual context 语义。

训练固定为两个阶段：P0 跳过 detector，binary action loss 训练 coarse，边界微簇损失
只训练 scorer/burst；official-60 先 exact-uniform 预热，再冻结 coarse 并训练 detector
与 selector，TAD surrogate 只有通过 hard-swap alignment 后才进入 selector。最终判据
不是 detector loss，而是 same-commit terminal-EMA mAP 与完整部署成本。

实验顺序固定为 R0 Oracle/KG 可达性、R1 数学/代码门、R2 P0 机制、R3 U/G0、R4
G1/G2、R5 三种子/预算曲线/第二 detector/完整成本。任一 kill gate 失败即删除对应
claim，不再通过新 selector、更多 epoch、fusion 或动态预算延长路线。

## 26. 恢复纯 Pre-Backbone 插件、原始总 60 轮合同与官方基线

2026-07-27 重新逐份复核本地保存的 DUCA 多轮与 Pro 原始讨论后，确认当前
“30 轮 full-model exact-uniform + 60 轮 full-model joint”的 K=384/K=192
实现不是最初冻结的多课程训练结构。原始结构允许 detector-free 的前端 P0，
但 detector 的正式优化总预算只有 6,000 次更新：先 exact-uniform，再在同一
60 轮预算内逐步释放采样率、贡献监督和 detector-to-selector 梯度。当前 K=384
终点 `65.385724%` 因此是 90 轮过预算候选，不再作为公平 official-60 最终结果；
Stage-2 epoch 50 的 `65.650497%` 也已消耗 80 轮。允许最佳中间检查点，但所有
比较臂必须共享相同最大更新数、评估频率和选择规则。

同时恢复官方 AdaTAD 基线事实：上游 VideoMAE-S、768 点输入、160 帧设置报告
Average-mAP `69.03%`。当前约 `68.29%` 的 dense 结果尚未证明等价复现该官方
权重/选择规则；`64.49%` 的 K=384 exact-uniform 使用 DUCA selected-axis
wrapper、目标映射和稀疏 detector 扩展，只能作为该 wrapper 的均匀控制，不能称
原生官方 1/2 下采样基线。必须在上游干净表面补齐 released-weight dense、
clean reproduction、native uniform K=384 和 native uniform K=192。

论文主方法重新限定为 detector-agnostic 的 pre-backbone acquisition plugin：
固定预算、单调原时间选点、只向后端交付采样后的特征及通用坐标映射，后端
backbone、projection、检测头、损失、分配和后处理保持不变。向检测头注入真实
时间、修改 assignment/regression 或设计专用 head，可以作为坐标诊断或增强版，
但不能再作为“纯可插拔选帧插件”的主证据。

理论主线改为有界、单调、恒定预算的任务感知时间重参数化：uniform 是严格
恒等特例；局部压缩/扩张和累计偏移有界；贡献监督使用分类/边界效用的归一化
排序或传输，而不是不稳定的原始幅值拟合；离散 hard 选点与 soft 反传必须通过
真实 hard-swap 效用对齐。下一轮只先做同一总 60 轮预算下的 native uniform、
joint-from-scratch、短 uniform warmup 后释放，以及逐项增加归一化贡献监督和
检测梯度；第一种子出现清晰增益后才扩展独立运行和第二种 detector。

免训练路线继续保留，但必须区分“目标数据无 selector 训练、detector 仍训练”
与“selector 和冻结 detector 均不训练的真正 plug-and-play”。现有 SlowFast、
MobileNet 和固定先验均未超过 K=384 wrapper uniform；真正冻结 detector 的
即插即用实验尚未完成。主合同记录于
`research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`。

## 27. 吸收总计 60 轮 Pro 大修，但不冻结未经验证的公式和阈值

2026-07-27 完整归档并逐项裁决 886 行 Pro 审阅。项目接受 `major revision` 的核心
科学判断：inverse-CDF 和学习选帧不是独立新意；必须先恢复 clean official dense、
native K=384/K=192 uniform 和 wrapper parity；主模型必须形成唯一、可测试的
`e -> p -> F -> y -> S` exact-K 有界时间重参数化；连续贡献排序与直接检测梯度分别
通过 `G_rank` 和 `G_direct`；第一枚 development seed 不进入最终统计。

纯插件的坐标合同进一步明确为：训练 GT 在插件接口映到 warped time，detector raw
proposals 在 NMS 前逆映射回物理时间，官方 NMS 算法和参数保持不变。因为非线性变换
不保持 IoU，先在 warped time 做 NMS 再逆映射不能作为严格纯插件等价实现。

项目没有逐字接受审阅方案。`1/(2T)..2/T`、`4/K`、DP 间隔/anchor 常数、线性 rank
RDD、`+1.0pp/40% dense-gap`、跨 detector 和成本阈值均保留为
`designed_reviewer_proposal`，须在 clean baseline、可达性和视频级功效分析后、
正式结果前一次冻结。DP 是实现候选而非贡献；禁止的是未登记 post-hoc repair。
`G_rank` 失败会杀死当前连续梯度贡献教师，但不逻辑性禁止一次明确的硬效用监督重设计。

checkpoint 采用条件合同：没有独立训练侧选择集合时统一第 6,000 次更新 terminal EMA；
有严格无泄漏留出集合时允许所有臂共享同一中间选择规则。A4 的短程分叉只作开发门，
正式 A4 仍受总计 6,000 updates 约束。PR #3 的 133-file aggregate diff 需另行清理，
但不是模型科学 blocker；最小 frozen-detector train-free 基线在共享解码器和坐标合同
通过后可并行，不与 task-adapted 主结果混称。

审阅时点后 K=192 已封存 `57.967272%` terminal 结果，因此原审阅的 D 级中间状态已
过时；但它仍是 90 轮超预算且缺 clean K=192 uniform，科学结论不变。公平 total-60
最终模型状态保持 `implementation_not_started / not_paper_ready`。

## 28. 动态 K 恢复为候选论文中心，但 RIME 与 nestedness 暂不冻结

2026-07-27 完整读取并归档 4,589 行 dynamic-K / AdapTok 研究接管回复。项目接受其
科学中心：在 heavy backbone 前，以 train-only hard counterfactual 估计不同真实
帧预算对 cls/reg/high-IoU 和 paired endpoints 的价值，用训练侧冻结的 per-video
dual policy 选择 `K`，并以有界 exact-K 物理帧集合和 pre-NMS 物理时间映射闭环。
dynamic K 因用户优先级成为必须裁决的候选中心，但不是已验证的必要机制或可单独
声称的新颖性。

项目拒绝把回复逐字冻结。短版明确要求各 K 共享证据但独立 exact-K 解码，扩展版却
要求 strict nested ladder；两者分别对应 budget-policy value 与 group-add marginal
value。当前采用 regret-gated decision protocol：先在 train-only Oracle 上比较
independent、strict nested 和一次 weak-overlap 备选，在 formal training 前冻结
唯一 decoder family。不得依据 official test mAP 切换。

回复中的 Oracle `+0.75/+1.0pp`、20/40% gap recovery、25/30% 成本下降和其他门槛
均为 reviewer proposals。clean baseline 视频级方差、cluster power、split manifest、
per-K uniform/U-same-K、K-histogram shuffle、null controls 和 hard-label GPU cost
闭环后，才预注册一套门槛。

旧 `duca-must` 继续保持 `paused/negative`；现有 online/selected-axis controller、
greedy center-radius decoder 和历史 `.codex_tmp` 不构成 RIME 实现。当前状态为：

```text
paper_structure = reopened
dynamic_k = required_candidate
duca_rime = discussed
decoder_family = unresolved
implementation = not_started
training = not_authorized
paper_claim = not_allowed
```

原始记录、独立审计、idea 与 experiment 分别位于：

- `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`
- `docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`
- `research-wiki/ideas/duca-rime.md`
- `research-wiki/experiments/duca-dynamic-k-rime-oracle.md`

## 29. 动态预算 C/P/R 材料准备（transport-only correction，2026-08-14）

唯一替换 ARIS DeepSeek V4 Pro Executor 完成动态预算路线的 C/P/R 材料准备，非科学重试。

**环境/边界事实（固化）**：cwd=`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`；
git common-dir=`.git`；HEAD=`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`（SparseHead
提交 `docs: freeze sparsehead evidence-first diagnostics design`，**非 DUCA revision**）；
branch=`codex/duca-total60-plugin-cvpr-20260727`；工作树脏约 275 条。本工作树绝不可称
clean frozen DUCA revision。DUCA 证据 = 命名 untracked working-tree 表面 +
`research-wiki/`。a6bdc084 / SparseHead Route-T / 任何 SparseHead 结果不得作为 DUCA 证据。

**parent-fidelity 分类**：`PrefixMarginalUtilityBudgetController` + `budgeted_center_radius_decode`
= 真实已有生产 dynamic-budget 工作（`paused/negative`）；`density_decode.py` = frozen fixed-K
有界密度分位解码器（`PROTOTYPE_ONLY`，只作 baseline/attribution/fallback 与内层复用原语）；
RIME / outer-budget-inner-transport = 仅讨论/规划、无实现；`pc_ot_mras_*` selectors/configs =
SparseHead 基础（tracked，非 DUCA）。

**科学边界**：sealed U/O/R execution-surface 终态失败（`--cfg-options` launch-bypass →
旧包 `657678946`；sealed replacement `6515ebf5` 未闭合 receipt 语义 → Critic
`SEALED_REPLACEMENT_BLOCKED`）仅作负证据，绝不作 efficacy 证据，不提议第三次修正。
Fixed K 仅作 baseline/attribution/fallback。动态 K 为 `required_candidate`，非已验证。

**C 阶段三路线**：
- A 原始动态选帧恢复（prefix-marginal-utility 控制器）→ REJECTED as final（`paused/negative`；
  selected-rank + greedy center-radius 已诊断失败；`lambda_dual` 留作 Route C 原语）。
- B 层级 Dynamic DUCA（outer per-video K + inner physical exact-K transport + hard utility +
  paired-boundary/high-IoU risk）→ **RECOMMENDED default，仅条件默认**。
- C realized-cost Lagrangian density acquisition（复用 `lambda_dual` + frozen
  `density_decode`）→ strongest deliverable fallback。

**推荐保留**：B 仅为默认，须先过 O1/O2/O3/O4 与 clean 方差+视频级功效冻结阈值，并经过一次
全新无上下文独立 Pro 攻击；任一 kill gate 触发即收缩（回退 fixed-K 归因或 Route C）。

**交付**：`docs/aris/ARIS_CPR_PLAN-2026-08-14.md`、`ARIS_DECISION_LOG-2026-08-14.md`、
`DUCA_ARIS_SOURCES_TO_PRO_REQUEST-2026-08-14.md`、终态 receipt；`research-wiki/log.md` 已更新；
`PAPER_PROGRESS.md` 已更新。无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim。

## 30. 独立 Pro 对动态预算路线的决定（2026-08-14）

全新 Project Pro 以 `REVISE` 冻结唯一计划级路线
`DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION-v001`（Candidate B）。它的可发表
前提是可独立检验的 TAD 特异机制：外层离散 K 决定每窗口投入多少重型计算，K-independent
inner physical exact-K transport 决定该预算落在何处；外层必须体现 paired start/end
high-IoU risk，而不退化为 motion、actionness、visual salience 或 generic clip difficulty。

- A prefix-marginal controller 为 `paused/negative` 历史对照，不得作为最终方法、pilot arm 或
  Route-B 代码来源。
- C realized-cost Lagrangian 不被准入为自动 fallback；若 B 无 oracle headroom 或 utility/risk
  不可学习，必须保留失败证据而非转写为泛化的 adaptive-computation 论文。
- O1 dynamic oracle headroom、O2 exact-K transport、O3 train-only hard-utility predictability、
  O4 paired-risk 相对 actionness/transition/no-risk/K-shuffle 的因果优势均为 hard kill gates。
  任何失败、K-collapse、hidden dense/padded Kmax、leakage、cost mismatch 或高 IoU/short-action
  退化终止 B；不得阈值放宽、自动转 C 或修补旧 U/O/R 包。
- 仅许可后续的 no-code plan sequence：Builder 作者产出 provenance/API/tensor/六臂/O1--O4/
  FIT-CAL-HOLD/cost/N16R4 argv 规范；Critic 给出唯一 static PASS/BLOCKED；仅 PASS 后 Evaluator
  可写 `PRE_RUN_NOT_READY` 结构协议。它不许可 patch、test、数据、GPU、Slurm、训练、指标或 claim。

证据状态：B/C、utility/risk、outer policy、reader、six-arm matrix 和 full-stack cost 合同均为
`DESIGNED_ONLY`；A 为已实现但负向历史；fixed-K decoder 为 `PROTOTYPE_ONLY`；没有 Route-B
实现、PRE_RUN、pilot、正式训练、性能、成本或论文证据。原始 Pro 回应：
`C:/Users/skywalker/.fastctx/jobs/j-n1xm7e/output.log`。

## 31. Route-B plan 的终态静态阻塞（2026-08-14）

`CRITIC_DUCA_DYNAMIC_ROUTE_B_PLAN_STATIC_REVIEW-v001` 返回
`DYNAMIC_ROUTE_B_PLAN_BLOCKED`。该 dispatch 的一次 Critic 审查与零 correction/recheck
边界已经耗尽，Evaluator 不得启动，Route B 进入 terminal hold。

1. **Canonical-uniform identity 不一致。** 一般 decoder 同时规定 midpoint quantile、首个
   `cdf[t] >= u` bin、`r=x-0.5` 与 squared-error 的 lexicographic-smaller tie。对于
   `T_v=768,K=384` 常量质量，这一规则选 `(0,2,...,766)`；同一规范却要求 bit-for-bit
   `floor((2j+1)T_v/(2K))=(1,3,...,767)`。这使未来的 fixed-K baseline 与 constant-density
   route 没有唯一共同 identity，不能实现或归因。
2. **K-shuffle control 不可复现。** 仅称“nonce-seeded permutation”而没有冻结 nonce/seed、
   canonical row order 与 algorithm，故不同实现可得到不同内容到 K 的映射并改变 O4/pilot
   结论。

上述均是 `STATIC_PLAN_REVIEW_ONLY / NOT_EXECUTED` 的负面计划证据，不是性能、成本或论文
结果。不能将 A、C、fixed-K prototype、旧 U/O/R、sealed replacement、SparseHead 路线或
`pc_ot_mras_*` 作为自动补丁或 fallback。任何未来的 simplification/repair/pivot 需要新的
独立科学决定；当前 action 为 Coordinator terminal hold。

## 32. 最终动态预算 Pro 裁决材料（2026-08-14）

用户要求动态预算保持为核心，而 Route-B 的一次 Critic/零修正角色环已经结束。因此 §31 的两项
确定性发现不被改写成“动态预算失败”，而被保留为最终 fresh Project Pro 必须裁决的科学歧义：

1. Pro 必须给出一个同时覆盖 CDF bin 约定、bin 内坐标、整数投影/平局规则及常量密度 special
   case 的唯一 canonical-uniform identity；或说明这种统一不能维持可隔离的 physical exact-K
   机制并 REVISE/PIVOT/STOP。不能由 Builder 在既有合同下选择 `(0,2,...,766)` 或
   `(1,3,...,767)`。
2. Pro 必须冻结 K-shuffle 的 nonce/seed、canonical row order、permutation algorithm、适用
   stratum 与无效/塌缩输入处置；或拒绝该对照不能支持 O4 因果归因。不同实现不可再有不同的
   content-to-K mapping。

这构成当前最大三轮中的最后一轮：DeepSeek round-1 proposal 与第一次 fresh Pro Route-B freeze
已经完成。唯一准备物是 `CURRENT_RESEARCH_STATE-v012.md` 和
`DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION-v001.md`，待 Sources 后交由全新 exact-Project Pro。
该决定仅可冻结定义或 REVISE/PIVOT/STOP；fixed K 仍是 baseline/control/fallback，不能成为
最终路线。官方 THUMOS14/OpenTAD-AdaTAD 六臂、FIT/CAL/HOLD、O1--O4 与 N16R4 仅是未执行的
后续合同。没有代码、角色、PRE_RUN、数据、GPU、Slurm、pilot、训练、指标、成本或论文证据。

## 33. 内层机制二选一：最终科学问题的重构（2026-08-14，local epoch v013）

§32 把 `CURRENT_RESEARCH_STATE-v012.md` / `DUCA_DYNAMIC_FINAL_PRO_ADJUDICATION-v001.md`
（经 `PROJECT_SOURCE_SYNC_REQUEST-v008.json`）当作唯一准备物；该 v008 source batch 现为
`UNKNOWN_REMOTE_STATE / quarantined`。本轮裁决：**不 retransmit、不依赖 v008**，以独立本地
material epoch `LOCAL_ONLY-v013-20260814T172500+0800` 重新准备，唯一交付是
`.cvpr-pro-lab/role-returns/BUILDER_DUCA_DYNAMIC_INNER_MECHANISM_MINIMAL_CHANGE_PLAN-v001.md`。

不裁决路线，只把最后一轮 fresh Project Pro 的科学问题重构为**内层机制二选一**：

- **A**：outer dynamic K + 任意非连续选帧 + 逐帧精确原始时间戳/physical-coordinate 逆映射。
- **B**：outer dynamic K + monotone/local physical exact-K transport（沿用有界密度 inverse-CDF
  整数投影，常量密度→canonical uniform）。

两条都必须保留真实时间戳；**禁止把 selected ordinal 当作 uniform time**（历史 selected-rank/
selected-axis 扭曲，属结构性禁止，不是阈值）。A 的代价是时间戳必须进入 detector 时间语义
（true-time positional/adapter），属 enhanced integration、非纯插件；B 是纯插件（detector
ordinal-grid 近似成立、只在 proposal→physical 边界修正）。因此 A-vs-B 同时是“纯插件 vs
enhanced integration”的科学裁决；RTK 的 pure pre-backbone 合同要求 A 若胜必须单列为 enhanced
integration，B 若胜则 F1（decoder identity）必须被冻结。

六臂合同：dense / uniform_k384 / dynamic_A / dynamic_B / k_shuffle(F2) / no_risk，全臂共享
detector/loss/NMS/evaluator/6k updates/terminal EMA，official val/test 不可达；`dynamic_A` vs
`dynamic_B` 是干净 A-vs-B（outer policy 相同，仅 inner 不同），fixed-K 的 A-vs-B 归因由 O2
在 matched K 下以 frozen detector 完成。证伪链：F-O1 动态 headroom、F-O2 内层几何+decoder
identity（F1 在此决定性）、F-O3 `G_rank`、F-O4 pair-risk、F-INV 时间戳不变量。

F1（canonical decoder identity）与 F2（K-shuffle nonce/order/permutation）**保持为未决 Pro
问题**：对 B，F1 是单调解码器 constant→uniform 的同一 identity；对 A，F1 重构为
canonical-uniform-subset + exact-timestamp 的 round-trip-bounded transport 合同；F2 对两者都
必须冻结。novelty 无效化含 AdaFrame/MGSampler/AdapTok/AdaFocusV3/SMART/TAPS/Progressive Block
Drop/keyframe/semantic-boundary wavelet，A 另有 TE-TAD/PhysTime/TrueTime 时间对齐先验，B 另有
Hartley systematic sampling 与 Uni-AdaFocus inverse-CDF 近邻。

无实现、无 PRE_RUN、无数据、无 GPU/Slurm、无训练/推理/评估/指标/成本/claim。停在
`MATERIAL_READY`，next_owner=central dispatcher → fresh exact-Project Pro（须另授
Sources-to-Pro lease，v008 不得复用）。

## 34. Final Pro：B 的机制冻结与 A 的 ceiling 身份（2026-08-15）

新鲜 exact-Project Pro（nonce `DUCA-DYNAMIC-INNER-FINAL-v004-20260814T203230Z`）在 v013
权威 Sources 上给出 `REVISE`。它不允许 fixed-K 取代动态预算核心，也不引入第三条自动路线：

- **B**=`DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION_B-v001` 是唯一 claim-bearing 路线：
  deploy-visible scout → batch-independent outer K → positive physical density → bounded monotone
  exact-K integer transport → physical-time decode/NMS/serialization。
- **A** 保留为同 outer-K 的 timestamp-aware arbitrary-frame enhanced-integration freedom ceiling，
  只能证伪“B 的局部约束无实质损失”的假设；A 优势不自动成为论文方法。
- **F1** 为 endpoint-inclusive integer-half-up canonical uniform generator；**F2** 为 routing-nonce
  派生种子、canonical row order 和 Fisher--Yates permutation 的 K-shuffle control。

该决定冻结 prospective FIT/CAL/HOLD、六臂、O1--O4、full-stack-cost 与 pilot→formal stop contract，
但其证据类仍为 `STATIC_PLAN_ONLY / NOT_EXECUTED`。路线没有实现、PRE_RUN、数据访问、远端/GPU、
pilot、训练、评估、指标、成本或效果/新颖性结果。旧 U/O/R、sealed replacement、历史 prefix
budget 与 prototype decoder 不能成为此路线的 implementation evidence；`a6bdc084...` 不能用作
DUCA identity。未来 Builder→独立 Critic→Evaluator 链需另有 downstream authority 和新的 clean
DUCA identity，尚未开始。完整 Pro 原文：
`C:/Users/skywalker/.codex/oracle/duca-dynamic-inner-final-v004/final.md`。

## 35. B 路线的首个干净实现快照（2026-08-15）

在用户已有的后续执行授权下，Coordinator 从 DUCA 的 `63a726a4...` 创建了隔离、干净的
`codex/duca-dynamic-physical-b-v001`，而不是复用 SparseHead `a6bdc084...` 或任何脏工作树。
Builder 的静态提交 `9eb328f9...` 实现并声明：dynamic outer K、B 的有界单调/局部 physical
exact-K transport、F1 canonical uniform、F2 nonce/order/Fisher--Yates，以及 future
`dense/uniform_k384/dynamic_A/dynamic_B/k_shuffle/no_risk` FIT/CAL/HOLD/N16R4 PRE_RUN 契约。

该提交只代表 `infrastructure_evidence`：它没有访问数据或 held-out，也没有训练、推理、评估、
指标、成本、GPU、Slurm、远端或效能结论。B 仍是唯一 claim-bearing 候选，A 仍只是 freedom
ceiling control。下一步骤是对 `9eb328f9...` 的一次独立只读 Critic 静态审查；PASS 后才允许
Evaluator 做结构性 PRE_RUN 绑定。

## 36. B 的 static review：局部 transport 必须 fail closed（2026-08-15）

Critic 对 `9eb328f9...` 返回 `DYNAMIC_ROUTE_B_STATIC_BLOCKED / IMPLEMENTATION_CORRECTION`。它证明 `dynamic_B` 的 transport helper 在局部锚点无法满足时会退回不保证 locality 的候选前缀，并把 `local_radius=0` 解释成无界；该路径的 F2 metadata 也领先于其实际 shuffle。由于 B 的科学定义要求有界单调/局部 physical exact-K，这些是首个实现快照不可接受的确定性偏差。

这不改变 Pro 冻结的科学路线，也没有新增效能、数据或 PRE_RUN 证据。接受的唯一工程动作是一次 focused correction：用合约保持或 fail-closed 行为替代无界回退，严格对齐 F2 执行和 metadata，并补充命中这些路径的静态测试。修正后仅允许原 Critic 一次 focused recheck；若有等价第二缺陷则终止该修正环，不能以更多修订链代替证据。

## 37. B 的 focused recheck：锚点仍破坏 locality（2026-08-15）

Builder 的唯一修正提交 `3e551595...` 修复了无界 prefix fallback、zero-radius bypass 和 dynamic-B F2 metadata；九项 focused static tests 通过。原 Critic 的 focused recheck 却在 `bounded_monotone_local_exact_k([0,1,0,1],2,radius=1)` 上得到 `[0,3]`。helper 在验证 nonzero locality 之前先接受了 anchors，因此 `[0,3]` 相隔 3 仍可返回，违背 B 的 bounded monotone/local physical exact-K 契约。

这构成第二个等价确定性缺陷，耗尽 one-correction 预算。该实现环终止于 `DYNAMIC_ROUTE_B_FOCUSED_STATIC_BLOCKED / PRE_RUN_NOT_READY`：不能再开第三次 Builder 修正或 Critic 链，也不能将 B 送往 Evaluator、真实 THUMOS14、N16R4 或任何训练/评估。科学路线未据此获得或失去效能证据；若要考虑真正简化的替代，必须在此冻结结论之外取得新的显式科学权限。

## 38. 实验优先续期：动态预算完整训练启动（2026-08-16）

用户新的明确授权将执行优先级改为完整官方远端实验，不再以 160/40 子集或先导结果裁决路线。主线仍是动态外层 K；固定 K 仅作官方可比对照。新的干净实现把任意选中的帧按原始时间排序，真实缩短 VideoMAE 计算序列，并在主干后按物理位置恢复到 768 步检测轴。完整矩阵冻结为五组（官方完整计算、均匀固定 384、学习式固定 384、动态预算、动态预算去边界风险）×三种子（3407/3408/3409）×60 轮，统一使用官方 THUMOS14 training/validation、同一 AdaTAD/ActionFormer、损失、后处理和 evaluator。

首个 Slurm epoch `1239577` 的 15 个单元均到达真实模型/数据读取后，因带空格的视频路径在配置覆盖时被拼接错误而失败。无空格的同一官方视频入口消除该数据绑定问题。随后 dynamic-budget seeds 3407/3408 在首次参数更新前暴露短窗口 budget clamp 破坏 16 帧 clip 对齐；`06b02be1…` 只修正该对齐并通过远端定向回归。重交 `1239627` 已进入 epoch 0 并完成至少 50 次优化更新；`1239628` 等待 GPU。以上故障均发生在效果测量前，不能解释为 DUCA 有效或无效；只有完整矩阵的官方指标与全栈成本可以回答科学问题。

截至 14:03，`1239627` 已进入 epoch 4，detector 与 selector 各损失持续有限；动态 seeds 3408/3409、official dense seed3407 与 uniform-k384 seed3407（`1239638`）均已登记等待 GPU。动态预算不再是只到达入口的作业，而是在真实视频上持续优化；尚无 official validation 结果。

## 39. 动态预算已持续训练，并加入边界风险因果对照（2026-08-16）

截至 14:20，dynamic-k seed3407 已进入 epoch 6并写出epoch-5 checkpoint。新的远端任务 `1239646` 对同一种子运行 `dynamic_k_no_risk`，只移除边界风险监督，用于判断未来任何收益是否来自边界建模而非动态预算本身。learned-k384 是下一待提交对照，但当前账户作业上限拒绝了该提交，未产生远端 job。路线不因排队改变；只有相同官方训练/验证下的完整结果可以决定动态预算与边界风险是否成立。

## 40. 完成 DSH 外部复核：方向保留、冻结语义包不可 PRE_RUN（2026-08-17）

新的空白 DSH session `session-70ec494d-3bf8-46a1-b45b-0162827e5e00` 完成了对 clean
`codex/duca-indirect-dynamic-20260817@6125654...` 的只读审查。它的首请求身份、唯一
`/^We need\b/` 指纹与 completed turn 均已由原始 session 核验；完整 receipt 在
`docs/dsh/DUCA_DSH_OWNER_REVIEW_RECEIPT-2026-08-17.md`。这次外部审查不选新路线，也不替代
后续 Critic/Evaluator。

审查判断间接语义 scout + 确定性 acquisition + dynamic outer-K 的方向仍正确，却发现当前
semantic package 有四项 deterministic implementation-contract blocker：batch-level budget metadata
被逐视频索引、动态 K 实际仍以 384 padding 行送入重型 backbone、selected-axis proposal 在 NMS
之后才映射 physical time、六臂配置尚未形成可实例化且变量隔离的比较。此前“GT 未传入”是误报：
`forward_train` 确实向 `_select` 和 `_losses` 传递 `gt_segments`。因此没有实际运行、mAP、成本或
claim，下一步不是重开科学路线，而是在 clean identity 上闭合这些实现合同，再走独立 Critic 与
Evaluator PRE_RUN。历史段落中任何尚未在 clean identity 上重新核对的远端训练日志只保留为线索，
  不得抵消本次代码审查结论或代替 shared official baseline receipt。

## 41. 历史 65 的输入链与 Query 协同/知识传递的受控保留（2026-08-20）

对 `42dba3f90b37243e7965d18b6707e88e81bf7109` 的冻结代码复核确认，历史 K=384 的
`65.385724` 不是 uniform：`global_structured_topk` 在原始 768 时间轴上选取非均匀的 384 个位置，
`_gather_time` 直接按这些 selected positions 收集 RGB，而 `TwoStageDetector.forward_test` 在重型
backbone 前用 selector 返回的 `inputs` 替换 dense 输入。该课程的记录为 384 selected frames、24 个
temporal chunks 和 384 VideoMAE frames。selected-axis 到原时间的 remap 在 proposal 的 NMS 前执行，
但该冻结 forward 合同没有把相邻 selected positions 的真实时间间隔送入重型 VideoMAE。因此它既是
“间接非均匀输入曾有正信号”的历史事实，也是可能把不连续物理帧作为相邻 selected-rank 观测的实现
风险；90 epoch、多组件课程仍使其不能成为公平主结果。

用户要求保留并继续研究 UVT/Fovea 中 Query 前后协同和知识传递的价值。项目据此不删除这些思想，
但不把现有 `query_cycle`、straight-through detector bridge、contribution distillation 或 `V(t)` 复合作为
主方法。新的干净周期应先建立 semantic-only 间接 selector；随后只允许两项可区分、训练期受控的扩展：

1. Query context ablation：同容量 scout 的 action/start/end 语义预测，与加入 Query contextual residual
   的版本一一比较，采样规则、K 和 detector 全部不变；
2. detached knowledge-transfer ablation：冻结的训练集 teacher 只向 scout 的 action/start/end logits 提供
   明示的软语义目标，teacher 不参与部署、K/索引决策或 detector 反向传播；其训练成本单列，检测器更新数
   不得增加。

只有这两项均与 no-Query/no-distillation 对照隔离，才可判断“协同”是提升语义预测，还是引入 detector
反馈、额外预算或时间轴混杂。已有 dense/uniform/random 与 VC 历史记录应被绑定为只读对照证据，不能因
这项复核而重复训练。未启动新的训练，也未新增性能、成本或论文 claim。

## 42. 以同预算配对实验隔离重型编码器的物理时间解释（2026-08-22）

历史 `65.3857%` 路径已经证明 train-only ASFormer 可以间接产生非均匀 K384 帧集合，但重型
VideoMAE 把相邻 selected-rank 观测当作等时间间隔。为避免重复 dense/uniform/random 实验，也避免把
动态预算、Query、蒸馏或额外训练轮次混入归因，当前决定只比较两个完整训练臂：

1. `RANKPACK_K384`：保留历史重型编码器内部的 selected-rank 时间解释；
2. `TRUETIME_K384`：RGB 帧、K、scout、课程、检测器和评估协议完全相同，但在第一次重型时间混合前
   消费原物理位置，并在共享 ActionFormer 头中继续使用物理坐标直至 pre-NMS 解码。

新 clean revision `11126684af779aa2916a68ecf617c4f14c805478` 已闭合旧候选未执行物理算子、短视频
metadata 长度和 padding attention NaN 三类实现问题。独立审查、N16R4 的 26 项 focused tests，以及
两臂 full/padded/short-padded 三步全模型门均通过。正式 Jobs `1248822/1248823` 已运行，状态为
`experiment_running`。这一决定只授权用终态 epoch-59 final-EMA 官方 validation 与完整成本账本回答
物理时间解释是否有用；在结果完成前，不升级或否定历史逐帧路线，也不把运行证据写成性能证据。

## 43. 首个终态裁决：CONTIG 停止，True-Time 保留为部分机制证据（2026-08-22）

固定 `M=24` 的 FZ_CONTIG/JT_CONTIG 在完整 THUMOS14、60 epoch、epoch-59 EMA 官方 validation 上
得到 Avg-mAP `49.89/47.24`。执行账本证明两臂真实只向重型 patch embedding 输入 K384，并非 padding
假稀疏；两臂的巨大定位损失因此触发合同停止条件。决定停止把连续 cliplet 获取扩展到 Query/cycle 或
dynamic-M，不以额外组件补救基础门，也不重复历史 dense/uniform/random。

同提交、同帧集合、同 K384 的 RankPack/TrueTime 配对得到 Avg-mAP `61.5722/62.1930`，TrueTime
在 tIoU 0.6 提升 `1.6885` 点。两份 prediction 已由冻结官方 evaluator 独立重算并精确复现记录值，
所以当前 seed 下“物理时间解释优于 selected-rank”具有机制证据；但单 seed 只允许 `partial`，不能写成
稳定或论文级主张。终态封存器还因 raw/normalized evaluation 配置哈希不同而失败，虽不改变指标，仍使
完整证据包未闭合。决定先对既有产物修复封存并运行 10,000 次逐视频配对 bootstrap；不得在该门之前
重训、增加模型组件或把不同协议的 65.xx/68.73 当成直接配对基线。

## 44. H65 兼容 SingleClock 获得科学准入，但首个实现周期关闭（2026-08-22）

exact DUCA Project 的独立 Pro 对 H65、RankPack/TrueTime、UVT 和 Query-Bridge 作了最终对抗性审查，
裁决为 `REVISE`：先恢复 H65 的完整语义采样与优化合同，只在首个 VideoMAE 时空自注意力加入一个
零初始化的相对物理时间残差；Query 与 dynamic outer-K 必须等待该表示门。代码核验接受这一主判断，
但把“第一时间注意力”限定为首层时空自注意力中的共享时间对偏置，并保留 H65 的 score-only
threshold/top-k→物理映射→NMS 顺序，以避免同时改变第二个变量。

新 clean worktree 从 H65 `42dba3f9` 建立，经过 Builder、独立 Critic、三次有界确定性修正和终态复核。
候选 `87d9a1ae` 已具有零初始化 scalar、uniform 无 mask 快路径和 24×16 位置分段，但它在每个 clip 内
重建 `exact_uniform_positions(768,16)`，没有先生成全局 K384 canonical 再按全局 rank 切片，因而测试的
不是冻结机制。实现周期以 `IMPLEMENTATION_PACKAGE_CLOSED_BLOCKED` 关闭，未进入 PRE_RUN 或 Slurm。

只读资源核验同时确认历史 Stage-1 epoch-29 EMA 与 Stage-2 回放 checkpoint 不在声明路径。科学路线没有
被效能实验否定；若继续，必须以新的 clean 实现周期修复全局 canonical clock，并从权威备份恢复 exact
Stage-1 SHA 身份。不得用相似 checkpoint、重新训练权重或当前错误候选冒充 H65 回放。

## 45. 以模型不变的总 60 轮候选检验 H65 课程可压缩性（2026-08-23）

历史 H65 的模型和间接非均匀逐帧机制已经冻结；当前不再用 TrueTime、SingleClock、Query、dynamic-K
或其他结构变化回答“90 轮能否压缩为 60 轮”。决定建立一个唯一的 schedule-only 候选：Stage-1 使用
20 轮 exact-uniform K384 预热，Stage-2 在一次连续 40 轮调用中先用 20 轮余弦过渡到 learned H65，再用
20 轮完整联合训练。只允许 Stage-1→Stage-2 的一次优化器、调度器和混合精度状态重建，总 epoch 40
处不得再次重置。H65 ASFormer、exact-K sampling-rate transport、density transport、贡献蒸馏、selected-rank
VideoMAE、检测器、损失、NMS、评估器和 seed `3407` 均保持不变。

该设计不重复 dense、uniform 或 random 的既有性能对照；原始 30+60 H65 复现是直接参照。冻结实现
revision `84acc15e948c213db48d1bc74a23d66ac868f7ca` 已通过目标环境测试、两阶段合同检查和真实 GPU
非零学习率更新，正式 Stage-1 Job `1250200` 已运行。只有 epoch-19 EMA 来源身份通过后才可启动 40 轮
Stage-2。预注册判定是最终 epoch-39 EMA 相对原始 H65 90 轮终点不低于 `-0.20` mAP 且高 IoU 不退化；
在终态官方 validation 前，本决定不构成课程压缩有效、性能等价或训练成本更优的结论。

## 46. DUCA-TAS 先冻结固定预算机制门，再决定动态预算准入（2026-08-23）

50Salads 的逐帧标注完整覆盖动作流程，原 TAD 的前景/背景 actionness 二分类在该任务上会退化为全正。
因此 TAS 迁移将低成本 scout 的学习目标冻结为 19 类粗动作识别和“相邻时刻是否发生类别转换”，再由
语义置信度与转换概率通过确定性逆累计规则生成有序原始帧位置；不得把直接预测帧索引作为主方法。
非均匀 RGB 必须在 VideoMAE-S patch embedding 前真实抽取，并在首次时间混合前保留物理时间信息。

第一阶段只运行 `uniform_2x(K=384)`、`uniform_4x(K=192)` 与 `h65_fixed384` 三个完整五折控制，统一
使用真实 50Salads 官方 split、EAST 检测头与 evaluator、seed 42、200 epoch 和 epoch-199 EMA。
既有 dense ViT-S 五折结果只读引用，不重复训练。只有 `h65_fixed384` 相对同预算 `uniform_2x` 在完整
五折上证明语义非均匀采样具有机制增量后，才准入共享动态集合 `{192,256,320,384}` 的
`dynamic_uniform/h65_dynamic`；否则停止动态扩展并分析语义监督、采样分布和时间重建误差。该决定避免
在固定选择机制尚未成立时，把动态 K 的预算变化与帧位置变化混为一个效应。

## 47. 以 First-Mixing SingleClock 作为 H65 物理时间的唯一表示门（2026-08-24）

最终科学裁决选择 `H65_FIRST_CROSS_TUBELET_BOUNDED_SINGLECLOCK_WITH_GATE_ZERO_TWIN-v001`。本阶段不改
H65 的语义间接非均匀逐帧选择、固定 K384、选中 RGB、训练课程、重型骨干、检测头、损失、NMS、split
或 evaluator，只允许 VideoMAE 第 0 个注意力块使用一个共享、零初始化、有界的物理时间残差。Bridge、
Query 和 dynamic-K 继续后置，不能用来修补本表示门；H65 OFF 只读重推理，不重复训练。

决定性证据不是单独 ON 分数，而是同一个 epoch-59 checkpoint 的 final/EMA ON 与 gate-zero twin。两者必须
逐窗口证明选中 RGB、原始位置和有效掩码完全一致，同时封存关闭门的配置路径、配置哈希和门状态。统计固定
为 10,000 次整视频 cluster bootstrap；短动作和时间畸变阈值只从训练 population 冻结；成本固定为同节点
三次顺序平衡的完整 validation workload 配对。只有身份、旧 RankPack/TrueTime 证据、H65 OFF 成熟度、
主要 mAP/高 IoU、短动作、畸变交互和成本门全部通过，才允许进入 replication；否则只能停止或在不增加
新时间模块的前提下修订。当前正式训练仍在运行，尚无 SingleClock 效能结论。

## 48. H65 总 60 轮压缩转为学习率时钟归因（2026-08-24，Pro 终稿修订）

H65 原 `30+60` 日程的终态 EMA 为 Avg-mAP `65.1257`、mAP@0.7 `43.3137`；模型不变的
`20+40` 压缩日程为 `62.4648/39.9434`。两者模型参数键、选择器、固定 K384 输入、检测器、损失、
NMS 与 evaluator 相同，因此下降不能归因于模型容量变化。最强直接异常是 Stage-1 30轮与20轮终点
`59.4231/49.5389`；同时 Stage-2 更新从6000降至4000，semantic/policy transition 从3000压至2000，
feedback 从 `1000+2000` 压至 `667+1333`，cosine horizon 与 full-joint 尾段也同步缩短。共同绝对
`eta_min=1e-8` 还会在末段破坏参数组相对 LR 比例。现有证据不能把 `-2.6609` 点定量拆给这些因素。

独立 Pro 裁决 `REVISE`：不增加峰值 LR，不改变 H65 模型，复用成熟 Stage-1 epoch-29 EMA。严格总60轮
只剩 3000 次 Stage-2 更新，无法生成历史缺失的3000次梯度、optimizer、EMA和联合适应机会，因此只能测试
日程恢复，不能保证无损压缩。下一门只比较两个30轮 Stage-2：主臂 `AM-RPCH25` 为500-step warmup、
1000-step峰值平台、1000-step相对 cosine 到`0.25×`、500-step `0.25×` hold；归因臂
`LongCosine-H6000` 使用500-step warmup和历史6000-step horizon，在3000 updates停止。两臂共享
2000-step semantic/policy transition、`1000+1000` feedback和1000-step full-joint tail。两臂之后停止
scheduler 搜索；达到原终点0.50点工程保真带才增加种子，否则按预注册恢复比例决定接受部分恢复或承认
60轮预算不足。当前状态为 `designed`，尚未实现、审查、PRE_RUN或训练。

## 49. H65 学习率归因进入正式执行（2026-08-24）

冻结实现 `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f` 保持 §48 的科学变量隔离，只加入两个相对
学习率调度器、完整 epoch-boundary 恢复状态和规范 THUMOS14 路径绑定。目标环境测试、两臂 fresh
PRE_RUN、跨作业 resume 与规范路径 smoke 均通过；恢复后 optimizer/scheduler/EMA/selector 累计时钟
从 2 连续到 4，没有重置。独立复核据此给出正式训练准入。

两条 30-epoch Stage-2 作业 `1252979/1252980` 从同一 Stage-1 epoch-29 EMA、同一 seed 和相同基础 LR
提交。它们各自必须完成 3000 次成功更新并以 epoch-29 EMA 裁决；每5轮 checkpoint 仅用于恢复与诊断，
中间 validation 不改变模型选择。两臂终态前不增加 scheduler 搜索，也不把运行状态写成课程恢复证据。

## 50. 全矩阵同资源审查只准入 SingleClock Gate-v2 既有证据闭环（2026-08-25）

全矩阵独立审查将官方 dense、H65、20+40 压缩、两条30+30学习率归因、RankPack/TrueTime、UVT、
Fovea/Query-Bridge、连续 cliplet 与 First-Mixing SingleClock 按设计、实现和结果边界重新对齐。审查确认
H65 的 `65.1257/65.3857` 是训练暴露、语义 selector、贡献蒸馏、ASFormer 适配、selected-axis 检测和
物理回映的复合配方结果，不能归因给单个模块。60轮压缩的当前最高值 `63.56` 未恢复 H65；停止学习率与
课程比例搜索，保留30+60 terminal-EMA 为唯一同预算基座。

SingleClock 的唯一合法主比较冻结为 terminal EMA `ON−H65 OFF/replay` 的 Avg-mAP、mAP@0.6、mAP@0.7
三个 point delta，均需不低于 `−0.20` 个百分点；配对 bootstrap 置信区间只报告，不进入硬门。现有影子差值
为 `−0.6596/−0.4037/−0.1720` 点，属于负向预警，但跨运行根且缺五边界 H65 replay、canonical-uniform
骨干逐位身份与正确终结器，故状态为 `EVIDENCE_ADMISSION_BLOCKED`。下一步只允许从既有作业 artifact
恢复这些身份并终结；无法确定性恢复则标为 `EVIDENCE_INVALID`，不得重训或用 Query/Bridge/dynamic-K 绕过。

## 51. PJST-D1 冻结为首次重型时间混合表示门（2026-08-25）

早期用户提供的独立 Pro 终稿裁决 `REVISE`，建议以零新增参数的
`DUCA_PHYSICAL_JACOBIAN_SUPPORT_TUBELET-v001` 取代 post-PatchEmbed SingleClock：将 VideoMAE temporal
kernel=2 分解为支撑加权外观项与按真实间隔归一化的变化项，并在 exact canonical uniform 输入上直接
旁路原 PatchEmbed。代码核验确认该候选作用于 H65 尚未修复的第一次二帧重型混合，且可在不改变
selector、K384、ActionFormer、loss、NMS 和 evaluator 的条件下实现。

项目只做 `PARTIAL_ACCEPT`。完整 H65 联合训练会让表示变化经梯度改变 selector/scout，故不能同时声称
“selected RGB 完全相同”的纯表示因果效应。执行前必须二选一冻结：允许选择中介变化的系统级总效应，
或冻结/重放 selector 的纯表示归因；二者不得混称。support 应表述为归一化支撑插值而非严格时间积分，
“无 gap shortcut”只成立为常量内容下无加性纯几何信号。PJST 只修复 pair 内第一次混合，也不代表后续
rank positional semantics 已全部解决。

随后完成的 fresh exact-DUCA v002 Pro 终态裁决仍为 `REVISE`，并正式冻结
`PJST-D1: Derivative-Only Physical-Jacobian Scaled Tubelet`。唯一首门因果口径是固定/重放
selector 的表示归因：matched OFF 与 ON 必须共享 K384 positions、RGB、valid mask、起点 checkpoint、
RNG 与 exposure ledger；端到端 selector mediation 不属于本轮。实现从干净 DUCA `b2ccfcca...`
worktree 开始，先闭合 global pairing、canonical byte identity、duplicate/padding、exactly-once pre-NMS
physical mapping、same-input ledger 和 no-new-parameter 静态门。现有项目根 `a6bdc084...` 为
SparseHead 污染身份，不能作为生产实现基座。

进一步核验发现早期终稿的统计合同有一处确定性错误：10,000 个 bootstrap 样本取第 500/9500 个值约为
90% 中心区间，不是双侧 95%；正式定义必须改为预先冻结的 2.5%/97.5% quantiles。科学上也不应在
首门同时引入 support-weighted 外观与物理差分缩放。fresh v002 已冻结最小修订为 derivative-only：保留
普通 pair mean，只以 `bar_delta/delta` 校正差分项，support 只作审计元数据；首个正式 falsifier 优先
固定/重放 selector 以隔离表示效应。当前状态为 `designed_frozen / implementation_starting`；
尚无 PRE_RUN、训练、mAP 或成本结果。

实现周期先后形成 `877d893f...` 与一次聚焦修正 `84325205...`。独立 focused recheck 仍发现
生产桥把全局 `[B,192]` pair metadata 错误重复成 `[B*24,192]`，而每个 VideoMAE clip 只接受
`[B*24,8]`；temporal checkpointing 也没有随 chunk 切分 pair metadata，且实现缺 matched OFF 配置。
这是第二个等价确定性实现缺陷，故该实现包按冻结边界关闭，不进入 Evaluator 或 PRE_RUN。PJST-D1
科学合同保持 `designed_frozen`；由于没有效能实验，不能据此判定机制无效。

## 52. PJST-D1 点估计后只允许一次统计终结（2026-08-27）

完整匹配训练得到 OFF/ON epoch-59 EMA Avg-mAP `65.063/64.591`，即 ON−OFF `-0.472` 个百分点。
该单 seed、完整 validation 点估计没有正向支持，但在缺少预注册整视频成对区间时也不能升级为正式负结论。
独立 Pro 结果审阅因此裁决 `REVISE`：不改 PJST-D1、selector、训练合同、NMS 或 evaluator，也不重训；
只允许一次终结评估启动器的 canonical absolute VideoMAE-S 预训练权重路径绑定，随后复用既有 OFF/ON
epoch-59 EMA，在同一 211-video official validation 上只读重推理、封存逐视频预测并完成冻结的 10,000 次
whole-video paired bootstrap。只有结果身份与点估计复现通过，区间和预注册门才可用于科学处置。

本次网页会话曾与同一浏览器 profile 的先发任务并发，且 Oracle 未正常写出终态文件；但 exact DUCA Project、
conversation、nonce、提交材料和完整可见回复均一致，未发现跨项目内容。该运行与捕获警告保留在科研日志，
不改变上述科学裁决，也不作为论文证据。当前唯一下一步是最小 launcher 绑定修复，经一次独立静态审查后
执行既有只读终结；禁止把附带的重型哈希或证明系统扩展成新门禁。

## 53. PJST-D1 点复现闭合但成对区间未执行（2026-08-27）

最小预训练路径绑定在 clean `7bd120f0...` 上通过独立静态审查。随后 OFF/ON epoch-59 EMA 均完成完整
211-video 官方 validation 重推理：两臂视频集合完全相同、各 422,000 条预测，所有 mAP 与原训练记录逐项
精确复现。Avg-mAP 仍为 `65.063283/64.590802`，即 ON−OFF `-0.472481` 个百分点；这闭合了点估计身份，
但没有改变其“无正向支持、尚非正式负向总体结论”的证据级别。

唯一统计终结作业沿用了平铺的 `work/result_detection.json` 参数，而官方单卡评估实际把预测写入
`work/gpu1_id0/result_detection.json`。作业在任何抽样前退出，故为 `0/16` shards、`0/10,000` replicates，
没有冻结置信区间或 PASS/KILL。该错误属于结果封存路径绑定，不是模型效能证据；本轮按既定边界关闭，
不修改、不重提、不重训，也不据此否定 PJST-D1。

## 54. 动态预算改为完整视频滑窗总体上的匹配预算重分配（2026-08-29）

初始动态预算合同要求在同一视频的窗口间排名，但 H65 训练每轮只随机截取一个窗口，使训练时窗口数退化为一。Pro 因此修订而非放弃动态预算：training 与 official validation 统一使用长度 768、步长 384 的确定性全视频滑窗；冻结 H65 侦察器一次性生成逐窗口语义与 K256/K384/K512 的选帧表；一个视频的全部窗口作为一个训练单位，两个视频构成逻辑批次。

正式比较冻结为同一新合同下的 fixed384、semantic 和 content-independent 三臂。semantic 与 content-independent 在每个视频上拥有完全相同的 K 多重集和 `384×窗口数` 总预算，唯一差别是 K 与窗口的对应关系是否读取冻结语义。实际 VideoMAE 必须按 K 分桶执行，不能补齐到 K512；分桶只能产生一次逻辑 optimizer、scheduler 和指数移动平均更新，并在 NMS 前恢复物理时间。

这一修订把论文问题收缩为“相同平均高分辨率预算下，语义跨窗口重分配是否有效”，不再把总帧节省混入第一项因果检验。历史 H65 65.13 只作背景参考；新 fixed384 伴随臂负责同合同安全比较。当前没有实现完成、运行前核验、训练、mAP、配对区间或成本结果，下一动作是 clean `04c35a3b...` 上的最小实现、独立代码审查和正式运行前核验。

## 55. 先用固定预算检验原生 tubelet 时序 coreset，再进入真实动态预算（2026-08-29）

后续 Pro 裁决将当前问题转向一个更基础的表示与选择归因：在每个 768 帧窗口上构造 384 个 VideoMAE 原生两帧 tubelet，并固定选择 192 个。候选臂根据冻结侦察器的动作性、边界强度和时序新颖性确定性选择，同时约束端点与最大未选择空洞；对照臂在同一原生 tubelet 网格上均匀选择。两臂共享低分辨率上下文回收、物理时间残差、384 点 tubelet 网格重建和完全相同的后端检测合同。

这一固定 `K=384` 实验只回答时序 coreset 的整套拼接系统是否有生命力，不把固定预算提升为最终方法。用户进一步冻结研究顺序：如果真实 THUMOS14 结果没有直接否定低成本侦察与稀疏重型计算假设，下一阶段必须让动作状态、边界密度、新颖性和冗余度决定实际执行的 heavy clip 数；平均真实 VideoMAE 计算量必须与固定预算相同或更低，fixed384 只作匹配控制，padding 后的名义动态 K 不计。

实现最终冻结于 `codex/duca-native-tubelet-coreset-20260828@b33391126eac05e3353d322b973dda91741f0732`。第一组运行前检查在任何数据迭代前发现配置同时要求学习曲线中间验证和 terminal EMA 唯一选模；第二组则发现新实验误启用历史 P0 的变体、gate、pilot 和哈希证明链。两项均属训练前协议/环境绑定失败，没有产生科学结果。最终修复关闭无关旧证明链，保留有限重试、成功更新计数、每 5 轮检查点和 epoch-59 EMA，并用实际保存—恢复—继续训练验证模型、EMA、优化器、调度器、AMP、训练轮次、更新数与随机状态。N16R4 的 20 项相关测试与最终独立只读审查通过；最终运行前检查 uniform `1260182` 与 coreset `1260183` 均完成，完整 60 轮训练 uniform `1260184` 与 coreset `1260185` 已自动开始。当前没有本轮 mAP、置信区间或成本结果，运行状态不能替代真实性能裁决。

## 56. 当前归因收窄为 H65 优先级下的时间覆盖分配（2026-08-29）

最新裁决不再同时调整任务分数、边界梯度或重型表示，而是冻结 clean H65 `04c35a3b...` 的逐帧优先级、固定 `K=384`、VideoMAE-S/Adapter/ActionFormer、训练和评估合同，只把 Top-K 分配替换为 96 个物理时间锚点上的确定性设施位置贪心选择。该问题检验 H65 性能是否受过度集中的时间采样限制；它不是动态预算结论。

最小实现冻结于 `feature/duca-coverage-only-v1-20260829@a8a0514b00c3528fcf201e6a042b6056429346e1`。聚焦测试、严格 Stage-1 加载、启动器语法和独立只读审查通过，并部署到 N16R4 干净目录。首次 PRE_RUN 提交在创建作业前因 Slurm 用户提交数上限被拒绝；没有作业、训练、mAP 或成本证据。下一动作是在额度可用后提交相同 PRE_RUN，不修改科学合同，也不擅自取消共享作业。

## 57. Coverage 中间机制失败后转向边际计算价值（2026-08-30）

修正后的 `DUCA-Coverage-v1` PRE_RUN `1261679` 在 200 个无标签训练样本上确认了 selection identity、
合法位置与 H65 优先级保留，但集合变化、锚点覆盖和最大空洞均未达到预注册门槛；尤其最大空洞第 95
百分位从 `2` 恶化到 `8`。与此同时，代码核验确认 H65 对照是预算校准系统采样而非字面 Top-K。
因此 Pro 给出 `PIVOT`：停止 Coverage-v1 的两个 60 轮训练臂，不再把固定 K 内时间覆盖作为当前主问题。

新的论文问题是跨窗口的重型计算边际价值。唯一当前任务从 clean H65 `04c35a3b...` 开始，冻结
H65 Scout、epoch-59 EMA detector、VideoMAE、检测头、损失、NMS 和官方评估器；在训练侧生成
K256/K384/K512 的 detached 反事实检测损失，先判断同视频 equal-budget reallocation 是否存在
oracle headroom，再判断低成本 Scout 是否能预测该 headroom。只有实现、headroom 与 predictability
三类门全部通过，才允许一次官方 test；当前没有 DUCA-Marginal mAP、配对区间或成本结果。

## 58. 联合邻域穷尽后终止现有 DUCA-Marginal-v1（2026-08-31）

DUCA-Marginal 的 50% capped 真实效用 oracle 相对固定 K384 只有 `+0.725589/+0.729004` 个百分点；解除
改变窗口上限后降至 `+0.427310/+0.450280`。Pro 随后只允许一次 capped→released 差分邻域的联合 mAP
诊断，以区分独立窗口效用误排序与若干有益转移组合后的负交互。

最终公开提交为
`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`。
实现由真实分配和成本导出 5 个视频、12 个窗口、6 个净转移组、8 个最小合法转移和 96 个唯一逐视频等成本
状态，没有硬编码多解视频的配对。独立审查、16 项聚焦测试、23 项既有回归测试和 Job `1262121` 的 96 次
CPU evaluator 调用均通过；fixed/capped/released 复现误差为零，原始 JSON SHA-256 已独立核验为
`a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`。

没有状态同时达到 `+0.8 pp` Avg-mAP 与 `+1.0 pp` mAP@0.7；联合门最优仅为
`+0.553972/+0.933234`。Avg 最优和 @0.7 最优分离，八个最小等成本转移也没有一个同时改善两项指标。
精确 DUCA Project 的 Pro 因此裁决 `STOP`：窗口级加性反事实检测损失不是该邻域中视频级联合检测效用的
充分排序统计量，现有加性 DUCA-Marginal-v1 已获得终局负证据，不再执行代码修改、重评、bootstrap、
utility-head 训练或 official test。

停止边界不扩展到 K256/K384/K512 三档本身、H65 priority sequence 或任务感知动态计算的一般问题。最终
分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为
Marginal-v1 的恢复性实验。

## 59. 项目级转向整视频一致预算的跨视频转移（2026-08-31）

在现有加性 Marginal-v1 被终止后，精确 DUCA Project 的新 Pro 对话读取最新公开
`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`
以及完整阶段证据，裁决 `PIVOT`，而不是停止整个 DUCA 问题。

唯一未直接检验的机制是：同一视频全部重叠窗口保持统一预算档位，仅在不同视频之间移动计算，使决策单位与
视频级预测合并、Soft-NMS 和 AP 更一致。冻结任务只枚举一次 donor 整视频 K256、recipient 整视频 K512、
其余整视频 K384 的有序候选；候选在读取标签或指标前完整生成，只保留实际 observation 总成本不高于
`47110` 的状态，并复用密封预测和相同评估器。

只有至少一个候选同时达到相对 fixed K384 的 `+0.8` Avg-mAP 与 `+1.0` mAP@0.7 才允许返回 Pro 继续讨论。
若通过候选为零，则在当前 THUMOS14、H65 优先序列、K256/K384/K512 动作空间与资源边界内项目级停止 DUCA
方法创新；不得扩大组合搜索、改变门槛、训练控制器或访问 official test。

## 60. 整视频枚举后终止当前三档 observation-transfer 路线（2026-08-31）

最新实现已经公开于
[`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)，
精确提交为
[`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)。
终态作业 `1262190` 完成全部 704 个合法整视频 donor-recipient 状态；通过预登记 `+0.8` Avg-mAP 与
`+1.0` mAP@0.7 联合门的状态数为零。终态 JSON 位于
`/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`，
SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`。

Fresh exact-Project Pro 对话读取了上述最新 GitHub 仓库、实际分支、精确提交、runner、聚焦测试和未修改的
三档 allocator，并裁决 `STOP`。准确停止范围是当前 THUMOS14 训练侧 controller holdout、冻结 H65 detector
与 priority sequence、密封 K256/K384/K512 窗口预测、真实 observation 成本口径和现有资源范围内，基于在
窗口或视频之间转移这三档 observation 预算的 DUCA 方法。Marginal-v1、cap-release、96-state 邻域和
whole-video 分支均转为只读负证据；不再扩张候选集合、改变预算档位或门槛、训练 utility/controller、补做
bootstrap、访问 official validation/test，亦不自动恢复历史路线。

证据支持的机制结论是：视频内混合预算不是失败的充分解释，且提升 Avg-mAP 与提升高 tIoU 的最优状态并不
一致；当前动作空间没有达到预登记联合效用门。最强但尚未验证的替代解释是，H65 priority sequence 与只在
K384 下训练的 detector 未形成跨预算兼容、单调且边界敏感的表示。验证这一解释需要新的训练机制或新的计算
动作空间，属于新项目而非当前路线修补。

当前没有新的科学实验任务。重新开放只允许从当前边界之外的新机制开始，并须先在独立训练侧开发划分上、匹配
真实计算且不事后扩张候选或改门的条件下展示预登记的 Avg-mAP 与高 tIoU oracle headroom。

## 61. REVISE：以多预算检测器适应检验跨预算表示假说（2026-08-31）

用户提供的最新裁决保留第 60 项对冻结检测器三档转移的 `STOP`，同时把跨预算表示不匹配改写为一个边界之外的
新科学问题。第一轮只改变检测器训练时是否同时暴露于 K256/K384/K512；现有嵌套位置构造、Scout、检测器结构、
损失、物理时间映射、Soft-NMS、评价器和实际成本口径全部冻结。较早附件提出的预算原生选点不进入第一轮。

固定控制只训练 K384；候选三档名义概率为 `0.25/0.50/0.25`，并按短窗口后的实际 observation 成本校准。
两臂必须匹配起点、成功更新数、优化器、学习率日程、随机种子、可训练参数和最终 EMA 规则。K384 安全门为
Avg-mAP 与 mAP@0.7 相对匹配控制均不低于 `-0.2` 个百分点；独立训练侧开发集的等成本 oracle 继续门仍为
`+0.8/+1.0` 个百分点且不增加实际 observation 成本。

模型基座冻结为 `04c35a3b...`；`33e4ed...` 只作为变长真实执行、packet 对齐、成本计数、K384 parity、
whole-video 评价与原始顺序保持的功能来源。当前裁决没有唯一冻结额外成功更新数/训练轮数，也没有给出新的独立
训练侧开发视频清单，因此状态是已设计而非已实现；在两项补齐前不启动 Builder 或算力任务。

## 62. 正式比较使用完整训练集与完整官方留出评估集（2026-08-31）

人类新增正式实验约束：第 61 项的固定 K384 控制臂与多预算适应臂在设计冻结后都必须使用完整训练集完成匹配训练，
并在完整官方 held-out evaluation split 上使用相同 annotation、类别映射、Soft-NMS 和官方 evaluator 作最终比较。
训练侧子集、40-video holdout、pilot 或 shortened run 只能承担前置诊断，不能成为论文主比较。官方留出评估不得
用于训练、checkpoint 选择、阈值/规则选择、路线选择或反复调试。

当前研究记录存在不能由 Codex 擅自消解的协议差异：OpenTAD/DUCA 的部分运行使用 `training/validation` 并报告
211 个评估视频；ActionFormer 官方运行使用 `validation/test`，历史记录报告 212 个评估视频。Pro 必须在 Builder
开始前冻结两侧精确 subset 名称、完整视频 ID、annotation/类别映射、评价器和留出评估使用边界。

正在生成的 Pro 对话早于这一约束，继续保持原会话，不追问、不打断、不重提。终态返回后若其训练侧开发划分或
训练日程与完整训练要求冲突，或没有明确完整留出评估身份，则通过新的 fresh Pro 裁决解决；在此之前不执行代码、
PRE_RUN 或训练。

## 63. Pro v001 继续多预算适应，但其 160/40 协议暂不准入（2026-08-31）

精确 Project 中的 Pro v001 选择 `CONTINUE`，冻结了现有嵌套 K256/K384/K512 下的单变量训练预算分布实验。
它指定两臂从 H65 Stage-1 terminal `state_dict_ema` 开始，各完成 6,000 次成功 update，并匹配 optimizer、日程、
随机种子、可训练参数和 terminal EMA；多预算概率按实际 observation 成本校准。旧 704-state 冻结检测器路线的
`STOP` 不变。

但该回答把规范训练侧 200 个视频划为 160 train / 40 development，且本轮不访问 official test。由于 prompt
提交后人类冻结了完整训练与完整官方留出评测要求，这一数据协议不能直接执行，也不能成为论文主比较。下一步是
新的独立 Pro 裁决，唯一解决完整 split 身份、211/212 差异、一次性留出评估和诊断到正式全量训练的关系；此前
不建立 Builder、PRE_RUN 或训练。

## 64. REVISE：完整训练与完整官方留出评测先经过数据身份准入（2026-08-31）

新的精确 DUCA Project Pro turn 已验证 Project、conversation、nonce 和 Pro 模型选择，并撤销第 63 项的 160/40
正式协议、旧 40-video holdout、有标签训练侧 mAP 门和 whole-video oracle。单变量科学问题保持不变：固定 K384
控制与嵌套 K256/K384/K512 多预算适应只在训练预算暴露上不同；两臂都使用完整 200-video `training` 集合，
从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始，各完成 6,000 次成功 update。

完整留出评测冻结为同一 annotation 的 `validation` subset，但 211/212 必须由只读事实核验解决，不能由 Codex 按
命名惯例选择。当前唯一任务是在 `04c35a3b...` 基座上物化 annotation、config、loader、物理媒体、evaluator、历史
211 IDs 和 ActionFormer 212 来源，只允许读取身份层字段。Builder 的精确提交必须经独立 Critic；Critic 通过后，
独立 Evaluator 在 N16R4 CPU 上运行一次。无论结论通过或阻断，都先返回 Pro；此前不得触碰模型、checkpoint、
GPU、训练、held-out temporal labels、预测或 mAP。

未来若获得 Pro 数据身份准入，两臂的全部固定预算和同一无标签 fixed mixed-budget 预测先密封，再一次性开放 held-out
标签，执行统一评测和 10,000 次整视频配对 bootstrap。K384 安全门保持 `-0.2/-0.2` 个百分点；mixed 直接差异门
保持 `+0.8/+1.0` 且成本不高于全 K384，并要求两项区间下界均大于零。该预冻结不构成当前模型实现授权。
