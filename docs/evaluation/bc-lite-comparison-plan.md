---
updated: 2026-09-08
status: research-design-not-dispatchable
scope: B/C/LITE 强比较、因子控制及四张新增证据图
out_of_scope: 伪造已实现对照、扩大正式任务计数、修改现有训练
---

# B/C/LITE：比较与证据计划

采用用户本轮的三类问题：价值如何学习、证据如何接收、更新何时可共享。以下是设计组，不是新训练 ID；正式 manifest 仍为1545项。模型语义以[B/C 方法说明](../methods/evidence-and-shared-updates.md)为准；来源见[近邻核对](../project/prior-art-bc-lite.md)。

继续优先执行已部署的 A/B/C 动态与固定 .5、seed0。B-full、dense、uniform 等已提交任务保留。新比较按资产、实现正确性和资源就绪推进，不等待 A/B/C 或 Oracle 获得好成绩；本轮文档修订没有提交新的 Slurm 作业。

## 最小机制比较与当前状态

|设计组|要做的控制|当前可复用部分与缺项|
|---|---|---|
|B receiver|物理插值＋残差＋TIA；普通时间 attention＋valid/null；mTAN-style 内容条件 receiver；实际 support receiver。|现有五 receiver 分支可复用，但同一 evidence 的训练/评价对照未完成。mTAN-style、同元数据增强 attention 须实现并冻结配置。|
|B support|anchor-only、hull、真实不连续支持；另分锚点与完整编码上下文；geometry-only 对比拟议任务校准。|真实支持并集算子存在；三层来源导出与配对干预尚缺。任务校准机制尚未定义，不能将现行 geometry bias 改名。|
|B architecture|Coarse-Fine、LookupViT-style 的合理 TAD 适配；AdaSpot-style 原图 ROI 如展开需完整像素获取路径。|未实现，不能用当前 B 更名冒充。先确保关键 receiver 控制，再按工程/资源依赖开发完整迁移；没有成绩门槛。|
|C selector|同一 C 执行器下 random/uniform、局部相似性、显著性、梯度代理、真实 refinement gain。|当前 C 已有普通 TAD-loss 的 signed refinement probe；uniform 任务可核对复用。新增分数分支、同成本比较与真实收益数据待落实。|
|C merge/recovery|MSViT-style gate；ToMeSD-style merge/unmerge＋TIA，分别 attention-only 与 attention＋MLP；ALGM/CubistMerge-style 结构控制。|未实现。必须恢复原位置、保留 residual/TIA，注明原方法与 TAD 适配差异，不只比较不利的压缩子层组合。|
|LITE transfer|原风格类别 Grad-CAM 代理；在相同 TAD 风险下重新训练的 LITE-inspired 代理；当前 signed operation gain。|均需精确实现映射；不得把现有 acquisition 分支命名为 LITE，或把原版 AVA selector 当作已完成区间风险控制。|

LITE-inspired 的 TAD 目标须冻结：由哪个风险标量求梯度、在哪层激活、如何聚合到共同原子、是否保留 ReLU/归一化、selector 输入与容量、PG/排序梯度路径、训练标签更新频率。以同一当前模型产生反馈的控制遵守无独立 Dense TAD Teacher 的约束；原风格分类代理另列，不能混称完全复现。动态预算再区分原风格置信度规则与收益/成本策略，不能把分类阈值直接挪入 TAD 并宣称公平。

## 让“同 evidence”真实成立

仅冻结选择索引不够：不同路线/训练的权重、前置状态或 packet 依赖会产生不同特征。应区分两个实验：

1. **接收机制控制**：冻结同一 Heavy/cheap-query 生产器、同一像素变换和计划，向各 receiver 提供相同特征及元数据；分别按共同预算训练 receiver。固定权重下只替换 metadata 的干预另报，不能替代训练后的架构比较。
2. **完整方法比较**：允许各自端到端训练；结果解释为整体方法效果，不宣称严格分离 receiver。Receiver×Router 组合先核对是否已有等价配置/训练再展开，不预造所有交叉任务。

元数据控制向标准 attention 提供相同的时间、支持描述、valid/null及合理 FFN/gate机会，并核算容量和开销；不能删除对手有用模块制造优势。若要归因实际支持集合，特征、计划与其他输入保持相同，明确改的是 bias、mask 或输入字段，不能同时换一套 evidence。

## 成本与协议

沿用[公共比较协议](mod-tia-comparison-plan.md#公共协议)：全200训练、全211测试/792窗口、768×160、同一识别预训练和检测器、native384 TIA、Heavy parent约束、60轮seed0。保持已授权 full-test EMA best，并补固定60轮/共同50、60轮节点分析；不按测试集拟合风险权重、预算阈值或选择诊断样例。

Heavy MAC 使用每层每 parent 的实际有效 k，计入 k² 项。相同总 token 不一定同 MAC。C 的 coarse→fine 添加3个 token只适用于四成员均有效的组；同 parent 等有效组的 fine/coarse 对调才保持此计数。B 的 Q×M、投影、槽操作必须计入；几何 mask 不自动带来稀疏 attention 内核加速。

完整延迟纳入 Scout、路由、mixed 构造/回写、receiver、TIA、embedding、传输与所声明的其他阶段；训练额外 probe成本另报。同GPU同batch同边界同窗口比较，4090/A100分面。相同训练60轮不等于相同训练算力。

## 四张新增图：生产器尚待实现

以下图共用[已有证据图规范](localization-value-figures.md)的 checkpoint、计划、实际成本、视频及窗口身份；不再创建第二套结果收据。原117项诊断仍不得记为完成。

|图|必须新增的真实数据|可回答与不可外推|
|---|---|---|
|B-1 空洞附近错误|同 producer/evidence 下的各 receiver 原始预测、GT、anchor/实际支持/编码依赖；预先定义 gap/action-duration 比、边界扩张及同类实例粘连规则。|同条件错误是否随空洞增大；只画相关性不能证明远端语义导致粘连。|
|B-2 Receiver×Router|每个实际完成组合的配置、正式指标与真实成本；未运行格子留空。另标同 evidence 控制与完整训练结果。|分开接收/选择因素及交互；不能混合不同成本、不同 producer 的格子直接归因。|
|C-1 相似性与细化收益|同模型、同基准计划、候选组的局部相似性、各选择器分数、signed coarse→fine 损失差及 ΔMAC。|相似性/梯度代理能否预测真实操作收益；图不是把现有训练日志中的单个 probe 直接叫总体校准。|
|C-2 模式切换与边界漂移|固定视频、checkpoint，在同 parent 等有效组交换 fine/coarse 计划；前后实际成本、各阶段表征差、start/end预测、匹配及漏检。|检验计算计划变化是否引起错误；与真实内容变化分开，单凭切换点邻近错误不能作因果结论。|

诊断候选和阈值来自训练数据或事先冻结的窗口/规则；完整测试分析注明事后用途，不作为反复调参反馈。定性图按固定抽样规则选例；没有可靠细小交互标注就标 NA，不能凭看起来像小物体临时造切片。

干预采用确定性 eval 状态或已验证的配对随机性，保持检测 loss normalizer 状态；记录额外 forward。图 C-2 先完成表征与实际预测两级干预，再判断是否有理由增加尺度校准，不能看到相关性就默认新增时间平滑模块。

## 何种结果允许何种论文结论

- 普通时间 attention 已与完整 support 相当：B 的复杂几何机制未建立独立收益。
- LITE-inspired 同风险代理达到当前 gain：增量可能来自任务适配，不能将全部提升归给反事实定义。
- C 相似性与 refinement 表现相同：收益监督的必要性未建立；保留更简单解释。
- 计算模式干预没有稳定边界影响：撤回“尺度切换制造伪边界”的核心动机。
- 同一价值原则改进 A、B、C 或近邻执行器：可以讨论跨执行范式的适用性，但逐项标出实际完成证据。

这些结论决定解释，不决定其他已就绪实验是否有资格运行。新增训练只在目标、执行器或监督确有变化时创建独立任务；相同已有配置、诊断重算和计时补测优先复用。
