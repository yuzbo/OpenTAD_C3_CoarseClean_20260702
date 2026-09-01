DUCA 完整 Wiki 与 GitHub 全版本科研路线裁决报告
1. SESSION_ASSERTION
nonce:
DUCA-GITHUB-WIKI-COMPREHENSIVE-REVIEW-v002-20260831

ChatGPT Project:
g-p-6a91061f789881918ccd8357ca3d6c92

Wiki frozen revision:
8935e97219431b006fb04bbfc12c1005ebd81a05

H65 scientific base:
04c35a3b76897e6c1569eeede41ed3aecaf7f854

Whole-video diagnostic revision:
33e4ed137c33eef07f0452b44506a6993bdf7535

Adjudication date:
2026-08-31

Scientific decision:
REVISE

Unique clean model mainline:
04c35a3b76897e6c1569eeede41ed3aecaf7f854

Unique current task:
full-data identity audit only

Conditionally unlocked scientific experiment:
fixed-K384 training exposure versus matched K256/K384/K512
multi-budget training exposure

8935e972... 是 Wiki 同步提交，不是模型实现；33e4ed... 是冻结检测器三档预算诊断的终态代码，不是下一模型基座。H65 的模型科学身份继续由 04c35a3b... 承担。Wiki 总入口也明确将这三种身份分开。

本报告中的路线选择由我独立作出。Gemini 报告被作为前置批评材料吸收，但其中关于跨预算表示不匹配、非连续 tubelet 打包、项目成功概率、超参数和停止阈值的判断，没有在缺少隔离证据时被升级为项目事实。

2. GITHUB_READING_LEDGER
2.1 实际打开并用于裁决的 Wiki

已完整读取或分段完整读取：

区域	实际用途
research-wiki/GITHUB_REVIEW_INDEX-2026-08-31.md	核对关键提交谱系、当前审查边界与 211/212 未解决状态
research-wiki/index.md	Wiki 导航和研究记忆结构
research-wiki/query_pack.md	当前状态、结果与不可扩张的证据边界
research-wiki/anti_repetition.md	已停止路线和禁止原样重复项
research-wiki/decision_history.md	从早期 C3、CellCF、全局选择、物理时间、动态预算到 704-state 终态和完整数据协议的决策链
research-wiki/duca_model_version_registry.md	历史模型、分支、提交、能力与结果索引
research-wiki/source_registry.md	外部来源及其证据地位
research-wiki/log.md	关键终态运行和协议变更的时间顺序
research-wiki/experiments/	枚举目录并重点读取 native tubelet、Coverage、Marginal、whole-video 等与当前裁决直接相关页面
research-wiki/ideas/	枚举目录并用决策历史追溯设计来源
Gemini 全量报告	吸收其代码审阅线索，但重新检查其因果措辞和路线偏好

当前 Wiki 的中心历史明确记录：当前三档冻结预测动作空间已经终止，但“同一检测器是否因只在 K384 上训练而不能适应多预算输入”仍是动作空间之外的新问题；之后的完整数据协议又把当前动作收缩为 200-video training 和 211/212 身份核验。

本轮附带的九份本地材料均已完整读取，用于交叉核对角色边界、原始 PJST 证据、代码库存边界和历史版本索引；它们没有取代 GitHub 冻结提交和公开 Wiki。

02_CODE_INVENTORY_BOUNDARY

 

05_RESEARCH_ROLE_RULES

 

03_EVALUATOR_RAW_EVIDENCE

 

04_PROJECT_RULES

 

06_PAPER_FIRST_RESEARCH_SKILL

 

00_PROMPT

 

README

 

01_CURRENT_RESEARCH_STATE

 

07_HISTORICAL_MODEL_VERSION_REG…

2.2 实际打开的关键实现谱系

下列提交均核对了提交身份；对当前裁决有直接作用的模型、配置、数据、评估和诊断入口进一步读取了源码。

谱系	精确提交	阅读结论
Transition / CellCF	4ce69c852bdbd902046b47bc6019ae11e850dbe4	结果和全栈成本表面已追溯；末端提交主要是 profiler 证据修复，不是新 selector
Protected end-to-end	b3222af0895e23eca83113977c1bcfad75258c9e	末端提交只绑定 Slurm repository root；不能把运行入口修复解释成模型创新
Global curriculum	63e25eb17e523d369f73434ed4d9b6446608861a	末端提交修复 EMA 参数组审计；全局 exact-K、课程与梯度归属需按更早模型主体理解
H65	04c35a3b76897e6c1569eeede41ed3aecaf7f854	当前唯一可靠稀疏科学基座
TrueTime	11126684af779aa2916a68ecf617c4f14c805478	核对显式时间选择、直通梯度和时间元数据；不是单纯位置编码开关
PJST-D1	c73e8418... / 7bd120f0...	训练干预与终态只读评估分别核验
Native tubelet	b33391126eac05e3353d322b973dda91741f0732	固定 192 个原生 tubelet 的 matched uniform/coreset 配置核验
Dynamic native tubelet	d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610	真实变长执行、短窗口 fail-closed 逻辑核验
Coverage	048143124e2a36a76575200ae17d6f42ec79ea3a	中间机制门负结果
Marginal	f67d96f...、46812fac...	训练侧反事实预算与 96-state 邻域终态
Whole-video	33e4ed137c33eef07f0452b44506a6993bdf7535	704 个合法状态枚举、原始 prediction 顺序和评价路径核验

关键谱系由 Wiki 总入口公开列出。

2.3 关键代码文件、符号和审阅范围

以下 URL 均绑定精确提交。

文件与符号	审阅范围	能支持的判断
duca_online_frame_selector.py
 — DucaOnlineFrameSelector	约 L500–L1320	H65 不是字面普通 Top-K；它从完整时间轴上的动作状态、变化与选择合同形成位置
同文件 _remap_train_targets_to_selected_axis / _write_metas
	约 L2890–L3070	训练目标映射与推理 inverse-map 元数据属于 H65 实际检测几何
acquisition.py
 — DucaAcquisitionAdapter	约 L1345–L1385	采集适配器是 selector 到真实 RGB gather 的接口
同文件 forward_acquire
、gather_selected_observations	约 L2745–L3135、L3522–L3561	选中原始观察真实进入重型路径，而不是只在损失中模拟稀疏
vit_adapter.py
 — VideoMAE adapter forward	约 L850–L1030	代码支持物理时间或 packed route，但是否启用必须由具体配置决定，代码存在不能代表 H65 实验使用了该路径
actionformer.py
 — forward_train	约 L136–L302	检测器训练输入、projection、neck、head 的调用关系
同文件 forward_test
	约 L461–L519	推理路径与 detector 输出
single_stage.py
 — forward_test / post_processing	L110–L230	selector 在 backbone 前执行；proposal 在 Soft-NMS 前调用时间逆映射
同文件 _remap_selector_segments_for_post_processing
	约 L250–L330	selected-axis proposal 到物理时间的 fail-closed 检查
padding_dataset.py
	全文件	subset、GT 为空、test mode 等因素可改变实际 loader 总体
sliding_dataset.py
	全文件	滑窗 IOA、背景窗口和窗口归属也会改变有效评估集合
thumos.py
	全文件	THUMOS wrapper 自身的 annotation 与 class-map 解释
tools/test.py
	全文件	checkpoint/EMA、raw prediction 保存和 evaluator 调用
nms.py
 — batched_nms, soft_nms	全文件	prediction 顺序和 Soft-NMS 参数是结果身份的一部分
mAP.py
 — ANETdetection	全文件	官方式 tIoU 0.3–0.7 mAP 评价过程
run_duca_whole_video_consistent_budget_falsifier.py
	全文件	whole-video 只重组密封预测并调用 evaluator，没有模型训练或梯度
truetime_joint_selector.py
 — TrueTimeRelaxedHardTopKSelector	全文件	TrueTime 同时带有可学习选择、直通软路径和时间元数据，不能概括成一个纯时间编码消融

H65 的实际前向链是“完整低成本观察→语义与变化证据→有序原始位置→真实 RGB gather→VideoMAE→ActionFormer→pre-NMS 物理时间逆映射”。这支持“重型输入观察数从 768 降为 384”，但不支持“端到端延迟、能耗或显存降低 50%”。

2.4 NOT_INSPECTED

以下项目没有被伪装成已经核验：

ChatGPT Project 网页内部的所有历史对话 UI：NOT_INSPECTED。本报告只使用当前会话提供的 Project 身份和公开 GitHub。

N16R4 上真实的 annotation、视频文件、ActionFormer feature 文件、checkpoint、sealed prediction JSON 和历史 211/212 ID 清单：NOT_INSPECTED。这正是当前数据任务。

仓库所有分支中的每一个 blob：NOT_INSPECTED。我检查了总入口列出的关键谱系、当前决策所依赖的模型文件和终态提交，但没有声称逐行阅读所有并行 SparseHead、Spatial-Zoom、ChronoTransport 或已归档工具代码。

d80022e... semantic dynamic cycle 和 46c714... Query-Bridge 谱系的每一个父提交模型 diff：PARTIALLY_INSPECTED。其版本身份、正式结果和混杂边界来自 Wiki、注册表和终态记录；未把单个组件归因建立在未读源码上。

ActionFormer 服务器端实际 thumos14.json、实际 .npy feature inventory 和历史 212 prediction ID：NOT_INSPECTED。

真实硬件上的当前 full-stack latency、energy 和 peak-memory 原始样本：NOT_INSPECTED。

这些缺口不会阻止当前路线裁决，但会阻止模型实现和正式训练。

3. SCIENTIFIC_DECISION
REVISE

不是 CONTINUE，因为当前正式比较仍缺数据总体准入，不能把候选直接送入训练。

不是 PIVOT，因为最可靠的正资产仍是 H65，下一问题继续使用同一 Scout、同一嵌套 K256/K384/K512 位置、同一 VideoMAE 和同一检测器。

不是 STOP，因为 704-state 结果没有训练一个同时暴露于 K256/K384/K512 的检测器，因而没有回答以下严格单变量问题：

在选择位置、模型结构、损失、优化器、成功更新数、物理时间逆映射、Soft-NMS 和评价器全部不变时，仅让同一 H65 检测系统在训练期间见到 K256/K384/K512，是否能够改善同一个密封混合预算 workload，并且不损害 K384？

这是当前最高信息增益问题。它是所有后续动态预算方法的必要基座检验：若失败，继续设计预算 embedding、utility head、蒸馏或新 selector 没有科学依据；若通过，才说明适应后的多预算表示值得承载一个后续语义预算策略。

当前唯一任务不是模型实验，而是数据身份审计。

4. EVIDENCE_AND_IMPLEMENTATION_AUDIT
4.1 可靠基线与总体状态

共享 dense AdaTAD 参考约为 68.73 Avg-mAP；H65 clean fixed-K384 约为 65.13，mAP@0.7 约为 43.31。因此当前可靠稀疏基座与 dense 参考仍相差约 3.6 个百分点。该差距是系统级差距，不是某个单独 selector、时间编码或训练部件的已隔离效应。

H65 是复合配方：训练期语义 Scout、确定性非均匀位置、30+60 两阶段训练、贡献相关监督、selected-axis 检测和物理时间回映同时存在。它证明“这套复合系统在单种子上达到约 65.13”，不证明任一部件单独贡献了该结果。

4.2 路线级证据分类
路线	证据类别	科学处置
H65 30+60	完整单种子训练和官方式验证；当前最强 clean 稀疏参考	保留为基座和匹配控制
H65 20+40、AM-RPCH25、LongCosine-H6000	完整训练；分别约 62.46/39.94、63.22/41.25、63.56/41.01	充分否定“简单压缩日程或只改 Stage-2 LR 尾部即可恢复 H65”
CellCF/local-cell	匹配训练中 CellCF 未超过 transition，并受 one-per-cell 可行集限制	主方法身份终止；可保留为历史局部位移消融
V5 direct/homotopy/uniform-companion	全部低于 matched exact-uniform	旧梯度桥与 companion 训练合同停止，不原样重复
连续 FZ/JT cliplet	完整单种子训练约 49.89/47.24	当前连续 16 帧片段实现充分否定，不转成 dynamic-M 或 Query 修补
Native tubelet coreset	uniform 约 64.13、coreset 约 62.81；结构化封存失败	内部停止该整套 coreset 拼接方法；不可作为完整统计论文结果
RankPack/TrueTime	单种子约 61.57/62.19	只有局部正向机制线索，不能证明物理时间是 H65 缺口根因
PJST-D1	OFF/ON 65.063/64.591；Avg -0.472，@0.7 +0.123；无 bootstrap	没有平均正向支持；也没有总体显著负结论；不再列为当前任务
SingleClock	终态身份、OFF 配对和统计闭环未共同完成	工程/证据阻塞，不是方法负结果
UVT	57.35/55.93/55.92	首版整体明显弱，但同时改变选择、预算证据和几何，不能归因到单一组件
Fovea/Query-Bridge	最佳约 54.67	首版复合系统负结果；Query、知识传递和 selector 不能分拆归因
Coverage-v1	真实 training 样本无标签重放门失败，最大空洞反而恶化	终止该设施位置干预；没有训练 mAP，因此不是效能负结果
Marginal-v1	capped、released 和 96-state 邻域均未通过联合门	终止当前加性窗口效用与邻域修复
Whole-video 704-state	704 个合法 donor-recipient 状态，0 个通过 +0.8/+1.0	终止冻结 K384 检测器、当前三档预测和当前转移动作空间
动态 native-tubelet	代码、测试和短窗口 fail-closed；无正式结果	只作变长执行代码参考，不作模型候选
旧端到端约 58.39 路线	复合模型、训练、几何和实现变量未完全隔离	只能作为混杂负结果；不得称所有失败因素均已定位

这些处置与当前公开 Wiki 中的正式结果和终态边界一致。

4.3 Whole-video 结果究竟否定了什么

33e4ed... 的 runner 读取已经密封的 K256/K384/K512 prediction，保持 producer 原始顺序，枚举 704 个训练侧整视频 donor→recipient 状态，再用相同 Soft-NMS 和 evaluator 重新计算指标。它没有加载模型训练图，没有更新参数，也没有让 detector 在多个 K 上适应。

因此，0/704 支持的准确结论是：

在当前训练侧 controller holdout、冻结 H65 K384 detector、冻结 H65 priority sequence、密封 K256/K384/K512 predictions、当前真实 observation 成本和单次整视频转移空间中，没有状态达到预登记的 Avg-mAP 与高 tIoU 联合门。

它不支持以下扩大结论：

所有动态计算无效；

多预算训练适应无效；

跨预算表示不匹配已被证明；

非连续输入是唯一根因；

预算 embedding 或蒸馏一定必要；

H65 Scout 本身无效。

4.4 工程失败不能替代科学失败

PJST-D1 的 bootstrap 在任何采样前失败，是因为 finalizer 读取 work/result_detection.json，而真实单卡输出位于 work/gpu1_id0/result_detection.json。这是一条已隔离的证据运输根因，不是模型根因。两臂 211/211 prediction 和点估计已经精确复现。

Native tubelet 的两个 60 轮训练完成并输出日志点值，但结构化指标文件和 prediction 封存没有闭合；这足以让项目停止继续投入该候选，却不足以包装成带配对区间的论文结论。

Coverage 初次 Slurm 用户作业上限、早期动态预算的路径拼接、短窗口 packet 对齐、SingleClock 身份不全等，均发生于结果成立之前，只能触发最小修复或归档，不能成为路线效能判断。

5. ROOT_CAUSE_SYNTHESIS
5.1 可以称为已隔离根因的事项

PJST-D1 统计未执行的根因：prediction 路径层级错误；0/10000 bootstrap，不是模型效果。

Whole-video 首次锚点复现错误的根因：额外排序改变了 Soft-NMS 输入顺序；33e4ed... 恢复 producer 原始顺序后，704-state 结果才是有效终态。

Coverage-v1 中间机制门失败：冻结 H65 优先级的设施位置干预没有达到预注册的集合变化、覆盖和最大空洞目标；训练因此正确停止。

当前所测 60 轮压缩族失败：在冻结模型下，20+40 以及两种 30+30 LR 日程均未恢复 30+60。准确说法是这些日程不足，不是所有 60 轮训练理论上都不可能。

CellCF 主方法偏离原问题：one-per-cell 可行集不能进行真正跨区域预算转移，这一代码/数学约束与其方法主张不一致。

5.2 只有支持性线索、不能称为根因的事项

跨预算训练分布不匹配：是当前要检验的假说，不是 0/704 的既成解释。

非连续物理帧被包装成相邻 tubelet：H65 确有该风险；TrueTime 出现小幅正信号，PJST 平均点值却为负。证据互不支持单一根因结论。

selected-axis 几何造成 dense gap：代码显示 H65 使用 selected-axis 训练和 pre-NMS 逆映射，但没有一个干净实验将整个约 3.6 点差距定量归因到这一因素。

Scout 排序不够边界敏感：Coverage、Marginal 和 native coreset 的负结果提供怀疑理由，但 selector、检测适应和预算分配没有在一个严格单变量链中全部隔离。

预算变化导致检测头统计漂移：合理但未被正式训练实验直接测量。

5.3 仍未知

H65 的三种子稳定性；

完整 held-out 总体究竟是 211、212，还是两个来源在排除规则上对应不同 estimand；

H65 多预算适应能否保住 K384；

多预算适应后是否存在可利用的同成本动态分配 headroom；

H65 的真实端到端延迟、能耗和峰值显存；

约 3.6 点 dense 差距中由稀疏观察、表示几何、训练暴露和 selector 质量分别贡献多少。

6. ROUTE_DISPOSITION
路线/资产	处置	约束
Dense AdaTAD 68.73	保留只读参考	数据总体未核准前不直接计算正式性能差
H65 04c35a3...	唯一 clean 主线	不修改 selector 语义、检测结构、loss、NMS 或 evaluator
Fixed-K384 新匹配控制	条件继续	必须和多预算臂使用同一新代码路径、同一 200-video training、同一成功更新数
多预算训练暴露	唯一继续方向	第一轮不加入预算 embedding、蒸馏或新 policy
33e4ed... whole-video	保留只读负诊断和有限代码 donor	不作为模型 parent，不恢复 704-state 搜索
Marginal/cap-release/96-state	归档终态	不补 bootstrap、不改门、不训练 utility head
Coverage-v1	停止	不重新调锚点、覆盖权重或最大空洞
CellCF/local-cell	停止主方法	仅历史消融
连续 FZ/JT、native coreset	停止	不用 Query、dynamic-M 或新重构核补救
PJST-D1	归档为不完整统计的负向点估计	不重训、不再优先补区间
RankPack/TrueTime	归档部分机制线索	不晋升主线
SingleClock	归档工程未闭合周期	不恢复实现修正链
UVT/Fovea/Query-Bridge	归档混杂首版	不从中挑一个组件直接加入下一实验
Dynamic native-tubelet	仅作变长执行参考	不采用其 coreset、重构器或窗口 policy
ChronoTransport、Spatial Zoom、Mamba、Block Drop	与本任务隔离	不合并进 DUCA 当前候选
禁止原样重复

再次运行 capped、released、96-state 或 704-state 搜索；

再次进行 H65 20+40、AM-RPCH25、LongCosine-H6000；

再次将 CellCF/local-cell 升级为主方法；

再次将连续 cliplet 作为主要采样单元；

补做 PJST bootstrap 并借此延长物理时间路线；

在多预算适应结果前加入预算 embedding、Gumbel-Softmax、蒸馏、新 Scout、Query、Mamba、Block Drop、TensorRT 或跨数据集实验；

用工程失败、局部测试或训练侧 oracle 替代完整训练和完整 held-out 结果。

7. UNIQUE_RESEARCH_ROUTE
7.1 唯一论文问题

一个只在固定 K384 上训练的稀疏时序动作检测系统，是否因为缺少跨预算训练暴露而无法利用真实 K256/K384/K512 计算变化？在不改变位置构造、模型结构和评价协议的情况下，多预算训练暴露能否提高同一混合预算 workload 的检测质量，同时保持 K384 能力？

这个问题有论文级信息增益，但本阶段本身仍是一个终局机制门。仅证明“多预算训练比固定预算训练更稳健”未必单独构成完整 CVPR 论文；它决定是否值得继续唯一一次语义预算分配实验。

7.2 唯一机制

唯一机制是训练分布适应：

控制臂的每次训练前向只使用 K384；

候选臂在训练期间外生地使用 K256、K384、K512；

两臂使用同一嵌套 H65 priority sequence；

不学习 K；

不向模型输入预算 embedding；

不改变 Scout、VideoMAE、adapter、projection、ActionFormer、loss 或后处理；

候选臂和控制臂具有完全相同的可训练参数集合。

因此，任何差异只能解释为“多预算输入暴露对现有系统参数适应的作用”，不能解释为动态预算 policy 的作用。

7.3 最强竞争解释

当前三档动作空间本身没有足够可利用的检测效用；K256 只是删除重要证据，K512 增加的证据也没有一致边界价值。即使通过多预算训练，模型也不会在同成本混合 workload 上产生具有实际意义的 Avg-mAP 与 mAP@0.7 联合提升。

7.4 可证伪预测

候选相对控制必须同时满足：

在同一密封 mixed-budget manifest 上：

Avg-mAP 至少 +0.8 个百分点；

mAP@0.7 至少 +1.0 个百分点；

两项整视频配对 95% 区间下界均大于 0。

在固定 K384 workload 上：

Avg-mAP 不低于控制超过 0.2 个百分点；

mAP@0.7 不低于控制超过 0.2 个百分点。

mixed workload 的实际重型 observation 总数不高于全 K384。

不得通过 padding 到 K512 伪造变长执行。

这些门沿用此前冻结的实际意义尺度，但本报告重新将其解释为终局可发表性门，而不是 Gemini 建议自动生效。

7.5 成功和失败后只解锁一件事

成功后：只解锁一个后续问题——在已经通过的多预算适应模型上，用冻结 Scout 语义生成一个预登记、无 held-out 标签的预算分配，并与同成本、相同预算直方图的 content-independent shuffle 比较。

失败后：在 H65、K256/K384/K512 和当前 VideoMAE/ActionFormer 后端上停止 DUCA 动态预算主线。不得用预算 embedding、蒸馏、新 selector 或内部层级动态计算对本结果进行恢复性修补。

8. CODE_ORGANIZATION_DECISION
8.1 唯一 clean mainline
repository:
yuzbo/OpenTAD_C3_CoarseClean_20260702

base revision:
04c35a3b76897e6c1569eeede41ed3aecaf7f854

current audit branch:
feature/duca-full-data-identity-audit-v1-20260831

不得以 Wiki 同步提交、协调根目录、代码库存提交或 33e4ed... 作为模型 parent。

8.2 必须保留且默认不修改
表面	科学作用
DucaOnlineFrameSelector	冻结 H65 Scout 语义、priority sequence 和 nested positions
DucaAcquisitionAdapter	真实 RGB gather 与实际 observation 计数
H65 Stage-1 config	共同起点 epoch_29/state_dict_ema
H65 Stage-2 detector path	共同训练参数、loss、EMA 和 6000-update 参考
vit_adapter.py / VideoMAE-S	重型视觉编码
ActionFormer、projection、neck、head	下游检测结构
SingleStageDetector pre-NMS remap	物理时间逆映射
nms.py	原 Soft-NMS 算法及参数
mAP.py / tools/test.py	最终统一评价
checkpoint/EMA/resume 逻辑	相同起点和 terminal EMA 选择
8.3 可从 33e4ed... 有限复用的代码能力

只能逐文件审查后复制以下能力，不能 merge 整个分支：

K256/K384/K512 真实变长 packet 组织；

K384 parity；

实际 observation 计数；

producer 原始 prediction 顺序保持；

whole-video prediction ID 和 evaluator 结果核对；

变长执行不 padding 到 Kmax 的检查。

33e4ed... 的状态枚举、训练侧 oracle、candidate 搜索和联合门实现不得进入新模型分支。

8.4 当前数据任务允许修改的文件

当前只允许新增：

tools/bata/audit_duca_full_data_identity.py
tests/test_duca_full_data_identity.py
docs/data/DUCA_FULL_DATA_IDENTITY_AUDIT.md

允许生成但不提交大型数据：

data_identity_manifest.json
data_identity_set_diff.csv
data_identity_source_provenance.json

当前禁止修改：

opentad/models/**
configs/adatad/thumos/**
tools/train.py
tools/test.py
任何 checkpoint、prediction、NMS、evaluator 或模型启动器
8.5 数据准入后模型 Builder 才允许的最小表面

两个 matched config：K384 control、multi-budget exposure；

一个外生、无标签、可重放的 budget occurrence table builder；

同一 variable-K dispatcher，控制臂也通过该 dispatcher 但每行固定 K384；

actual observation counter；

focused shape、nestedness、gradient ownership、K384 parity、resume 测试；

一个最小 Slurm launcher。

禁止大型重构、通用动态计算框架、兼容层、schema 平台或新的 selector 类。

9. CURRENT_TASK_ORDER
当前唯一任务：完整数据身份审计
9.1 Builder

Builder 从 04c35a3b... 建立上述 audit branch。只读取身份层，不读取 held-out 时序区间、类别标签内容、prediction scores 或 mAP。

必须物化以下字面集合：

T_annotation:
annotation 中 subset == training 的所有 ID

T_physical:
training 对应可读取、可解码的物理视频 ID

T_loader:
使用正式 H65 training config 和真实 dataset class 构造出的 ID

H_annotation:
候选 held-out subset 的 annotation ID

H_physical:
候选 held-out 物理视频 ID

H_loader:
正式 OpenTAD held-out loader 实际产生的完整视频 ID

H_evaluator:
evaluator ground truth 实际接纳的 ID

H_prediction_211:
历史 H65/PJST 密封 prediction 的视频 ID

AF_annotation:
ActionFormer 实际 thumos14.json 的 test ID

AF_feature:
ActionFormer loader 可找到 feature 文件的 test ID

AF_loader:
ActionFormer `THUMOS14Dataset` 实际接纳的 ID

必须检查：

annotation 文件路径和 SHA-256；

database key、subset 字符串及其大小写；

重复 ID、扩展名、前后缀和规范化规则；

200 个 training 视频的物理文件存在性与 ffprobe 可解码性；

fps、duration、frame count 是否足以支持正式 loader；

PaddingDataset、SlidingWindowDataset 的 test_mode、空 GT、背景窗口和 IOA 行为；

class-map 顺序、类别数、annotation label 与 evaluator mapping；

211 prediction IDs 与 loader/evaluator IDs 的精确集合相等性；

ActionFormer 的实际 feature-missing 静默跳过行为；

211/212 的所有对称差集及每个 ID 的排除来源。

公共资料提供了一个待核验线索：OpenTAD 的数据说明称删除了 video_test_0000270 和 video_test_0001292 后使用 211 个视频；常见 THUMOS14 口径则报告 200 个 validation 训练视频和约 212 个带目标标注的 test 视频。这个线索不能替代本地字面集合审计。
GitHub
+1

ActionFormer 官方配置明确使用 validation 训练、test 评估；其 loader 还会在 feature 文件不存在时跳过视频，因此“配置写 test”不等于实际 loader 一定有 212 个样本。

9.2 Builder 输出
data_identity_manifest.json
  - every literal ID
  - source path
  - subset
  - physical file
  - loader membership
  - evaluator membership
  - prediction membership
  - exclusion reason

data_identity_set_diff.csv
  - left source
  - right source
  - left-only IDs
  - right-only IDs
  - source-backed explanation

data_identity_source_provenance.json
  - repository revision
  - config paths
  - source file SHAs
  - annotation/class-map SHAs
  - physical-root identity
  - ActionFormer source revision

DUCA_FULL_DATA_IDENTITY_AUDIT.md
  - plain-language conclusion
  - PASS or BLOCK
  - remaining NOT_INSPECTED items
9.3 独立 Critic

Critic 使用独立上下文，只审查：

Builder 是否读取了被禁止的 held-out 语义或指标；

是否使用精确 ID equality，而非 substring 或数量相等；

是否覆盖 test_mode、空 annotation、背景窗口和 missing-feature skip；

是否把 annotation membership、physical availability、loader membership 和 evaluator membership错误合并；

是否完整解释每个 211/212 差集；

class-map 和 evaluator 是否具有相同语义；

audit 是否能在 clean commit 上重放。

Critic 只能返回 PASS 或带精确 ID/代码位置的 BLOCK。不得要求代码风格重构或通用审计框架。

9.4 独立 Evaluator

Evaluator 只在 N16R4 CPU 上运行一次：

clean exact commit；

无 GPU；

无 checkpoint；

无 PRE_RUN；

无模型构造；

无 prediction；

无 mAP；

无 held-out segment 值输出。

9.5 数据准入通过条件

必须同时满足：

T_annotation = T_physical = T_loader，且数量严格为 200；

training 与 held-out ID 完全不相交；

held-out annotation、physical、loader、evaluator 和正式 prediction population 均有明确集合关系；

211/212 的每一个差异 ID 均被 source-backed 解释；

class map、评价阈值和 evaluator 语义一致；

任何排除规则对所有未来实验臂完全相同；

Critic 和 CPU Evaluator 都通过。

任一未解释差集、缺失物理视频、类别映射冲突或 evaluator population 冲突都返回 BLOCK。不允许为了形成 211 或 212 而按数量删除视频。

9.6 数据准入后的唯一条件任务

数据 PASS 返回本 Pro 后，才解锁：

Builder:
实现固定 K384 与多预算暴露的单变量 matched 训练

Critic:
审查唯一变量、真实变长执行、梯度归属、K384 parity、
成本匹配和 held-out 隔离

Evaluator:
PRE_RUN → 三种子完整训练 → 先密封全部 prediction →
一次性 held-out unseal/evaluation → 10,000 次配对 bootstrap
10. FULL_FORMAL_EXPERIMENT_PLAN
10.1 两个且只有两个训练臂
Arm A：固定 K384 控制

每个训练 occurrence 请求 K384；

使用 variable-K dispatcher 的同一代码路径；

不使用旧 H65 已训练 Stage-2 checkpoint作为终点；

从共同 Stage-1 epoch_29/state_dict_ema 开始重新训练。

Arm B：K256/K384/K512 多预算暴露

使用与 Arm A 完全相同的 nested H65 priority sequence；

只改变每个 occurrence 的 K；

不训练预算 policy；

不输入预算 embedding；

不增加 loss；

不改变可训练参数集合。

10.2 外生预算 occurrence 表

在不读取 GT、Scout score 或 held-out 信息的条件下，先枚举完整训练 occurrence：

(seed, epoch, canonical_video_id, window_start, valid_length)

固定 canonical row order。

候选表约束：

p(K384) 目标为 0.5；

K256 和 K512 的数量按每行实际可执行 observation 成本校准；

完整 6000-update 计划的实际 heavy observations 必须与全 K384 控制严格相等；

短窗口、packet 对齐或有效长度不足时使用预先固定的 eligible-row 规则；

若不能在不使用语义信息的情况下形成精确成本匹配，任务 fail closed，返回 Pro，不用近似成本继续；

三个种子各有独立、预登记且可重放的 occurrence 表；

Arm A 和 Arm B 的数据次序、augmentation RNG 和 optimizer update 次序保持配对。

在所有窗口都满足对称成本时，该规则会接近 0.25/0.50/0.25；真实执行以实际 observation 总数为准，而不是名义概率。

10.3 数据与训练

训练：审计通过的完整 200-video training population；

种子：3407, 3408, 3409；

成功 optimizer updates：每臂每种子严格 6000；

若 audit 确认每 epoch 100 updates，则对应 60 epochs；更新数是优先合同；

两臂共享 optimizer、基础学习率、scheduler、AMP、gradient clipping 和 EMA；

不运行中间 validation 选模；

最终模型：第 6000 次成功更新对应的 terminal state_dict_ema；

checkpoint：每 500 次成功更新或至多每 5 epochs 一次；

checkpoint 包含 model、EMA、optimizer、scheduler、GradScaler、epoch、成功更新数和 RNG；

至少保留最近三个恢复点、预登记里程碑和终态；

确定性环境/路径错误只做最小修复并从同一 checkpoint 恢复，不建立新科学版本。

10.4 公平基线

不重复训练：

shared dense AdaTAD；

历史 uniform/random；

PJST、TrueTime、Coverage、native tubelet；

old Marginal/whole-video。

正式因果比较只使用新 Arm A 与 Arm B。Dense 68.73 和历史 H65 65.13 仅作为背景坐标，只有数据总体审计证明完全可比时才可并列展示，不作为 Arm B 增益的分母。

10.5 预先冻结的 held-out prediction 矩阵

每个种子、每个模型都生成：

A@K256
A@K384
A@K512
A@M

B@K256
B@K384
B@K512
B@M

其中 M 是同一 label-free、content-independent、预登记 mixed-budget manifest：

两个模型读取完全相同的 M；

M 的预算直方图与实际 observation 总数预先冻结；

M 的实际成本不高于全 K384；

M 不依据 held-out GT、prediction、confidence 或 mAP；

生成所有 prediction 后先记录 SHA-256、video IDs、result counts、配置和 checkpoint；

所有 prediction 均密封后，才一次性开放 held-out GT 给 evaluator。

10.6 三个明确 estimand

主要因果 estimand

B@M − A@M

只回答多预算训练暴露是否改善同一个 mixed workload。

K384 安全 estimand

B@K384 − A@K384

回答多预算暴露是否破坏标准预算能力。

次要部署 estimand

B@M − A@K384

同时包含训练适应和 mixed workload 差异，只能作为部署结果，不能替代主要因果比较。

10.7 评价和统计

报告：

mAP@0.3、0.4、0.5、0.6、0.7；

Avg-mAP；

每种子值、三种子均值和种子间离散度；

短动作、边界误差和预算档位分层结果作为预登记次要分析；

不从多个指标中事后挑选“正”的一个。

Bootstrap：

10,000 次整视频配对重采样；

相同 replicate 同时重采样两个臂的同一视频；

分别报告每种子的配对区间；

主结论再使用 seed×video 的分层配对 bootstrap；

双侧 95% 区间使用 2.5%/97.5% quantiles；对严格排序的 10,000 个样本使用预先实现并测试的 nearest-rank 250/9750 口径；

不再使用历史错误的 500/9500 索引；

bootstrap nonce、RNG 算法、canonical video order 和输出 schema 在 prediction unseal 前冻结。

10.8 真实成本

在读取 held-out GT 前完成成本测量：

实际 heavy observation count；

scout；

selector/packet organization；

decode 与 H2D；

VideoMAE；

temporal adapter/projection；

ActionFormer head；

pre-NMS inverse mapping；

Soft-NMS；

完整 workload wall-clock；

latency p50/p95；

peak GPU memory；

GPU energy，仅在采样稳定且设备绑定可核验时报告；

训练 GPU-hours作为附加成本。

测量要求：

同一节点、GPU 型号、软件栈、精度、batch 和进程布局；

平衡执行顺序；

固定 warm-up；

完整 held-out workload 重复运行，而不是阶段 p50 相加；

observation count 是机制事实，不能代替 latency 或 energy；

不声称 384/768 自动等于 50% 端到端节省。

10.9 终态停止规则

以下任一发生即停止当前路线：

B@M − A@M 的 Avg-mAP 小于 +0.8；

mAP@0.7 小于 +1.0；

任一主要指标的 95% 配对区间下界不大于 0；

B@K384 相对 A@K384 在 Avg-mAP 或 mAP@0.7 下降超过 0.2；

mixed workload 实际 observation 超过全 K384；

variable-K 实际 padding 到 K512；

三种子方向明显不一致；

数据、prediction 或 evaluator population 不同；

需要新增预算 embedding、蒸馏或新 selector 才能继续。

若点估计为正但低于 +0.8/+1.0，结论只能是“存在轻微适应信号但不足以支撑动态 DUCA 主线”，项目仍停止扩展。

11. PUBLICATION_PATH
11.1 最多两条可证伪主张
主张一：多预算适应

在相同 H65 稀疏选择、模型结构、训练更新数和 mixed-budget workload 下，多预算训练暴露相对固定 K384 训练产生统计可靠的 Avg-mAP 与高 tIoU 提升，同时保持 K384 能力。

只有正式三种子、一次性完整 held-out 和真实成本全部通过后才可写。

主张二：语义预算分配

在已经通过多预算适应的检测系统上，低成本动作状态与边界证据能够比相同预算直方图的 content-independent 分配产生统计可靠的性能—成本 Pareto 改善。

本主张尚未解锁，当前不得实现或书写为已有方法。

11.2 明确非主张

即使当前实验成功，也不能声称：

DUCA 已优于 dense AdaTAD；

H65 的历史 3.6 点差距由跨预算不匹配造成；

非连续 tubelet 是历史失败的唯一根因；

384/768 带来 50% 端到端延迟、能耗或显存节省；

方法适用于在线/流式 TAD；

方法已经跨 detector 或跨数据集泛化；

Query、蒸馏、物理时间、Mamba 或 Block Drop 有效；

旧 704-state negative 被“推翻”。它仍然是冻结 K384 detector 的有效负证据。

11.3 第二后端和第二数据集解锁条件

只有在 THUMOS14 上同时满足以下条件后，才解锁一个第二后端：

三种子主要门通过；

K384 安全门通过；

无 hidden Kmax padding；

full-stack cost 可复现；

数据和 evaluator population 完整一致；

语义预算分配主张二也获得正结果。

顺序固定为：

先一个第二 TAD 后端；

后端复现通过后，才选择一个第二数据集；

不并行启动第二后端与第二数据集；

不因 THUMOS14 失败而换数据集寻找正结果。

11.4 最终投稿判据

DUCA 只有在以下条件全部满足时才进入论文结果冻结：

主张一和主张二均通过；

matched sparse control、semantic allocation 和 content-independent control 同代码身份；

三种子和整视频配对区间完整；

真实成本形成可复现 Pareto；

dense 差距被诚实报告；

至少一个第二后端结果方向一致，或论文清楚限定为单后端机制研究并有足够强的因果消融；

所有负路线作为设计边界保留，不被从论文叙述中删除。

若仅主张一通过而主张二失败，本项目得到一项有价值的预算鲁棒性结论，但不足以继续包装成 DUCA 动态采集主论文。

12. ABSOLUTE_MILESTONES

以下为冻结目标日期，不是对算力排队或未来结果的保证。时间以 America/Chicago 为主，括号内为北京时间。

日期	交付	依赖与失败处置
2026-09-01 23:59 CDT（09-02 12:59 北京）	数据身份 Builder clean commit 和全部 raw manifests	若物理文件或历史 212 来源不可读，明确 BLOCK，不做替代
2026-09-02 18:00 CDT（09-03 07:00 北京）	独立 Critic PASS/BLOCK	只允许一次集中修正确定性 identity defect
2026-09-03 12:00 CDT（09-04 01:00 北京）	N16R4 CPU Evaluator 终态和 Pro 返回包	数据 BLOCK 时，所有后续模型日期自动取消
2026-09-05 23:59 CDT（09-06 12:59 北京）	条件性模型 Builder exact commit、两臂 config、occurrence table 和 focused tests	仅在 Pro 接受数据 PASS 后
2026-09-06 18:00 CDT（09-07 07:00 北京）	模型独立 Critic verdict	若发现科学变量不止一个，退回 Builder；不得改变问题
2026-09-07 18:00 CDT（09-08 07:00 北京）	Evaluator PRE_RUN、resume 和 K384 parity	PRE_RUN 不是效能证据
2026-09-08 23:59 CDT（09-09 12:59 北京）	六个正式训练单元全部提交并记录依赖	队列失败只修启动，不建新模型版本
2026-09-18 23:59 CDT（09-19 12:59 北京）	六个 terminal EMA 或客观基础设施阻塞报告	不使用中间 validation 选终点
2026-09-20 23:59 CDT（09-21 12:59 北京）	全部 label-free predictions、成本样本和 SHA 密封	任一 video ID 集不同即停止 unseal
2026-09-21 23:59 CDT（09-22 12:59 北京）	一次性 held-out evaluation 和 10,000 次 bootstrap	不允许第二次查看 held-out 后调整规则
2026-09-22 18:00 CDT（09-23 07:00 北京）	Pro 终态：解锁唯一语义 policy 或 STOP	不增加第三种解释性实验
2026-09-23 23:59 CDT（09-24 12:59 北京）	终态 Wiki、证据表和论文边界冻结	无论正负都保存原始证据
13. NEXT_RETURN_CONTRACT

Codex 下一次只返回数据身份任务，不返回模型建议或训练计划改写。

必须带回：

1. repository / branch / exact commit / clean status

2. audit script and focused tests

3. annotation SHA and class-map SHA

4. exact set sizes and literal IDs:
   T_annotation
   T_physical
   T_loader
   H_annotation
   H_physical
   H_loader
   H_evaluator
   H_prediction_211
   AF_annotation
   AF_feature
   AF_loader

5. exact set differences:
   every left-only and right-only ID

6. source-backed explanation for:
   video_test_0000270
   video_test_0001292
   every additional 211/212 difference

7. loader semantics:
   subset
   test_mode
   empty annotations
   missing files
   background windows
   sliding-window inclusion
   class mapping
   evaluator filtering

8. physical media evidence:
   existence
   readability
   ffprobe success
   fps/duration/frame-count availability

9. independent Critic report

10. N16R4 CPU Evaluator report

11. one final conclusion:
    DATA_IDENTITY_PASS
    or
    DATA_IDENTITY_BLOCKED

12. remaining NOT_INSPECTED items

返回包中不得出现：

模型改动；

checkpoint 加载；

PRE_RUN；

GPU；

Slurm 训练；

held-out predictions；

mAP；

对 211 或 212 的无来源选择；

因为“看起来像官方”而进行的 silent reconciliation。

最终科学裁决归纳：

当前 DUCA 不应整体停止；

当前冻结检测器的三档预算重分配路线已经停止；

当前唯一仍值得检验的机制是多预算训练暴露；

该机制尚未获得模型实现授权；

当前唯一动作是完整数据身份准入；

数据通过后才执行一项两臂、三种子、完整训练、一次性完整 held-out 的终局实验；

该实验失败后，停止 H65/K256-K384-K512 上的 DUCA 动态预算主线，不再通过附加机制恢复。

DUCA_GITHUB_WIKI_COMPREHENSIVE_REVIEW_READY
