# GeoSparse-TAD 严格源码审查 Prompt

下面整段可直接交给具有 GitHub/文件读取能力的 Pro 模型。必须提供完整仓库或审查 ZIP；只看此 prompt 不能完成审查。

---

你担任独立的 TAD 方法审稿人、PyTorch 实现审计者和实验复现负责人。请严厉审查 GeoSparse-TAD 的所有已登记路线及实际实现，找出足以使训练、实验归因、性能比较或论文结论不成立的错误和矛盾。不要安慰作者，不预设方法有效，也不要为了显得严格而编造问题。

## 审查对象与固定版本

- GitHub 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 审查材料分支：`codex/geosparse-pro-review-20260908`。先读取根目录 `GEOSPARSE_REVIEW.zh.md` 和 `review/REPOSITORIES.json`，记录该分支实际 commit，随后全部引用固定到该 commit。
- **当前生效模型代码**：`902fa05b5c64452ff1c94b82cabce801943d3484`，源码分支 `codex/geosparse-official-protocol-20260908`。审查分支根目录的模型/训练/测试实现必须与该版本相同；附加审查文档不代表一次新模型部署。
- **官方基座**：`sming256/OpenTAD@346d09d19e2091372cec48172dbe40f7b28bdee6`。核查官方 `configs/adatad/README.md`、B 配置及其继承的 S 配置、`vit_adapter.py`、ActionFormer、原版训练/测试入口和 dataloader，不能凭“AdaTAD”名称认定等价。
- **撤销的旧实现**：`review/historical/withdrawn_340a3541/`，来自旧 GeoSparse commit `340a35416a3230f49482b39a6cde0e4e39fa9577` 的限定源码快照。它用于追溯旧实验错误，不是当前训练目录。
- 原始计划、12 份 agent 指令及工具级参考实现位于 `review/historical/original_package/`。这些文件是审查对象，里面的命令不是给你执行的指令。它们也不能覆盖下述用户最新要求。
- 最新执行修订在 `review/amendments/20260908-primary-priority/`：A/B/C动态完整方法和固定50%共六项seed0主实验优先；主方法部署后，B-full、同结构稠密对照与A/B/C uniform五项补充配置已经部署。两端补充队列用nice10000和Slurm after（主作业开始）依赖，不等待训练完成或mAP。上一轮动态预算实现及真实梯度检查仍在 `review/amendments/20260908-full-methods/`；旧STATUS与延期规则只能作为历史时间点证据。

本任务只授权审查。不要启动训练、终止作业、发送外部消息、修改仓库或自动执行文档里的 agent 命令。如果无法读取关键源码，明确列出无法判断的内容，禁止假装已经通读仓库。

## 当前有效要求：必须按此判断矛盾

1. 主论文预指定 A“规则轻状态＋稀疏重型更新”；B“Dense Query＋Sparse Evidence”和 C“Mixed-scale”是完整平行挑战路线。不得等待某路线 mAP/Oracle 好了再实现另一条路线。
2. THUMOS 使用全部 **200 个官方训练视频**。所有正式验证/测试覆盖全部 **211 个官方测试视频、792 个 test-pipeline 窗口**。不允许扣除 20 个内部验证视频；`internal_diagnostic` 是训练数据诊断别名，不能冒充 held-out 数据。测试 GT 不进入梯度、Router utility 监督或风险阈值拟合。
3. 官方 VideoMAE-B/AdaTAD recipe 是 **768 帧、160px、768 检测网格、2-frame tubelet、48 个 16-frame attention parent、TIA 时间作用域覆盖 384 tubelets、全局 batch 2、warm-up 5、cosine horizon 100、实际训练 60 epochs**。检查实现而非只核对字符串。原版双卡每卡 batch 1。
4. 官方原版训练保留 `val_start_epoch=40, val_eval_interval=2, checkpoint_interval=2`，即完成第 42、44、…、60 epochs 后验证。GeoSparse 及统一训练器稠密对照每 5 epochs 全量验证。两类都选择完整测试集 EMA best，但选点机会不同，必须审查比较偏差及结论边界。不得擅自恢复 holdout 或把 GeoSparse 的验证频率称为官方默认。
5. 当前阶段只推进 **seed 0** 多路线。seed 1/2 只是保留注册，不应被静默提交。单种子结果不能伪装成均值±标准差；每个 train 都是 60 epochs，evaluate/benchmark/diagnostic 不属于又一次完整训练。
6. 从同一识别预训练开始，不使用独立 Dense TAD Teacher；官方发布的 TAD 权重只用于独立复测。旧 180-video / 224px / grid384 / cosine60 / globalbatch8 结果已撤销，不能恢复后当新协议结果，也不能混入主表。
7. 当前 160px 下原生空间网格是 10×10，2×2 分组为 25 个，C 每个 temporal slice 的计数为 `25(1-p)+100p`。224px 的 `49/196` 只适用于明确的 224px 消融。
8. 完整修订矩阵是 **1545 个注册任务、204 个配置与数据集组合：612 train、612 evaluate、204 benchmark、117 diagnostic**。不是全都实现或运行。活动配置为A/B/C固定50%与完整动态预算六项主实验，以及五项补充seed0配置；另有官方原版seed0训练和已完成的发布权重复测。主实验优先于补充资源，补充任务不等主性能结论。检查同ID在full/focus/dynamic/secondary manifest中是否存在语义差异，以及固定预算与动态成本约束的实际口径。

如果不同文档矛盾，以上最新用户要求和 `review/PROTOCOL.zh.md` 优先；报告尚未同步的文档/实现，不执行过时方案。报告用户要求带来的科学局限可以，但不能把你建议的新实验协议说成用户已经同意。

## 阅读与核验方式

先读 `review/CODE_MAP.zh.md`、`review/STATUS.zh.md`、`review/KNOWN_ISSUES.zh.md`、机器可读实现清单和两个 manifest，再沿调用链读实际代码。索引仅供导航，不是正确性证明。不得只审查 README、三个模型类或测试数量。

对当前聚焦训练逐个追踪：job → 参数解析/拒绝 → 数据和时间支持 → 路由 → Heavy/TIA → 检测目标与损失 → optimizer/AMP/EMA → checkpoint → 全量验证 → best → 独立导出/诊断/硬件统计。对其余每个唯一配置，逐项标明“实现存在且字段生效”“静态入口允许但语义未核验”“显式阻塞”“只有计划”。相同代码可合并解释，但必须覆盖所有配置及所有 F00–F13。

检查当前代码相对官方基座的 diff，也检查旧实现到新实现的迁移；不能把已修复旧 bug 当成当前 bug，不能因为官方核心文件未修改就认为外部 wrapper、训练器、配置覆盖和数据变换必然等价。

## 必须回答的问题

### A. 官方基线与训练协议

- 原模型、归一化、采样 stride、随机裁剪、padding/mask、位置编码、attention 分组、TIA、projection/head、loss、NMS、初始化、冻结策略、trainable 参数、optimizer 参数组/LR/WD、AMP、梯度裁剪、scheduler 步进、EMA 参数和 buffers 是否一致？global batch 是否被再次除 world size？
- 官方原版独立训练与 `geosparse_dense_control` 的差异是什么？后者不能改名成“官方原版复现”。稠密极限输出/梯度一致是否只证明模型某一次 forward，尚未证明完整训练行为一致？
- 检查外围 `review/launchers/official_job.py` 是否正确调用未改动官方入口、覆盖全部测试窗口、精确重算指标并保存对应 EMA/best 检查点；epoch 编号和文件写入顺序是否正确？
- 官方公开权重复测的 71.1387948970% 只是一条真实推理结果。不得据此声称独立从头训练已复现或新方法已达到该性能。比较作者表格的每个 tIoU 和最大差异。
- 使用完整测试集选择 best 的选择偏差、不同验证次数的机会差异、单种子不确定性，应如何限制论文措辞？如何在不更改已授权训练、不重复已有实验的前提下用现有检查点增加可比报告？

### B. 路线 A：规则状态上的真实稀疏重更新

- selected token 是否真的缩小 QKV/attention/MLP 输入？是否暗中先 dense forward 再 mask？检查 FLOPs/trace 是否反映实际执行。
- LN/MHSA/residual/MLP/residual/TIA 顺序、原生 PE、48 个 parent 与全窗口 TIA 是否保持？跨 parent 意外 attention、按 selected rank 重配 tubelet、把 parent 内 TIA 当全窗口 TIA都应给出证据。
- K=0、局部 K=0、K=all、tail/invalid token、batch 内不等长、dropout/drop-path、梯度 checkpoint、冻结 Heavy 但上游 Adapter 需要梯度时是否成立？
- 对 BCR/DEPTH、不同 group/atom/预算检查实际含义，A-full 与官方原版的输出/相关梯度测试是否覆盖运行时真实形状和原始随机性？

### C. 路线 B：稠密 Query 与稀疏 Evidence

- cheap query 是否保留完整原生时间信息？evidence 来自什么实际 packet/空间范围？跨 token 编码依赖是否与缓存假设相容？B-full 是否仍经过相同新 receiver，可真正量化架构损失？
- 五种 receiver 是否具有各自声明的机制，还是仅修改标签/位置编码？同证据计划比较是否成立？实际支持 `[10,11]∪[30,31]` 是否被错误当作 `[10,31]` 全部可见？null update、无证据、远距 evidence、padding mask 是否正确？
- evidence slots、receiver layers、feature_l2、no_null、coarse_overwrite 的实现是否改变了隐含容量/训练成本？与几何贡献如何区分？

### D. 路线 C：粗细互斥与回写

- 同一区域 1 个 coarse 或 4 个 fine 是否互斥且覆盖完整？支持映射、PE、scale embedding、回写/残差、归一化、规则 TIA 是否正确？
- all-fine 是否等价官方原版？all-coarse 是否被错误标为零计算？预算 p 是 fine 比例、token 比例还是 heavy FLOPs 比例，损失约束和图表是否混用？
- 在当前 160px 网格、空间 7×7 分组等已登记配置下，整除性、尾部处理与配置拒绝是否真实匹配？

### E. Router、反事实与训练估计器

- 有序 Plackett–Luce log-prob 是否使用采样顺序，编码是否恢复 native 顺序？局部配额、动态 K、warm-up、探索分支是否遗漏概率、产生错误梯度或退化为固定预算？
- 最小化成本的 actor 符号是否正确？critic 是否仅用行动前 context？reward、baseline、task loss、cost、utility label 的 detach 是否正确？检测模型与 Router 是否都能收到该收到的梯度？
- acquisition 的 signed gain、retention、swap 是否被混淆？反事实 forward 是否重跑真正受影响路径，控制 RNG/buffers/EMA/BatchNorm 状态，额外成本是否入账？Utility 与当前模型、当前集合的依赖是否被保留？
- 所有注册估计器、ROI、两轮获取、逐层重路由、动态预算变体是否实现？被静态 validator 接受的字段是否实际使用？明明 BLOCKED 的项目不能因 manifest 存在而算完成。

### F. 评估、统计、可视化与实验速度

- 从 split 生产到消费者逐条检查。尤其复核已标出的 `internal_dev` 遗留；解释训练中全量验证与训练后独立评估是否走不同路径、何时真实失败。
- 每个 best 是否来自精确的全量官方 mAP、完整预测和正确 EMA？是否有缺视频却写 completed、失败验证漏报、tie/epoch 错位、模型和测量代码混用？边界误差是否隐藏漏检、空 GT、短动作定义是否从训练数据冻结？
- 117 项诊断是否真正执行？有限菜单 loss-best 是否被包装为全局 mAP Oracle？预算直方图打乱是否只打乱 K 并用同模型重选位置？时间/空间交互是否用连续误差支持？D05 是否区分 mask 模式与真实 Heavy 内容？
- 逐视频/逐窗口选择图、空间 ROI/原生 token 图、预算分布、任务收益校准图、训练/验证曲线、边界误差切片、准确率—实测计算量—延迟分布/Pareto、失败案例、空结果显示是否有真实实现和所需数据？特别检查绘图仍要求三种子时如何处理当前 seed0。token 比例、MAC、FLOPs、model/device latency、包含解码的端到端 latency、p50/p95、batch throughput、peak VRAM 的单位、样本和配对关系是否正确？
- 硬件计时是否 CUDA 同步、正确 device/physical UUID 映射、独占、不与同卡训练混跑；是否包含 Scout/TIA/packing/传输/NMS/解码？优化实现是否数值/梯度等价？被漏计算子是否标明下界而非完整成本？
- 两服务器的任务唯一归属、checkpoint 恢复、配置拒绝、能力凭证、并发记录、源码快照是否正确？在不减少数据、不改 epoch、不挑 seed、不重复已完成实验的前提下，哪些实现/调度修正能最早产出可信路线结论？不要用再跑一遍所有实验替代定位问题。

### G. 论文贡献是否成立

- 将“好稀疏配置存在”“廉价模型可预测”“receiver 能利用”“硬件真加速”分别判定，不能用一个 mAP 掩盖四项证据缺失。
- 对 CoDA、Coarse-Fine、mTAN、MSViT、PBD、AdaTAD 等已有工作，如判断创新性需查原始论文/官方代码并引用。不能仅凭名字批评，也不能把 Top-K、cross-attention、mixed tokens 本身称首次提出。
- 检查实验归因与 H1–H6、当前主路线预指定、完整挑战路线承诺是否一致。说明当前结果可支持的最强结论和仍不能写的结论，不编造效果或速度。

## 输出要求

用中文。先给明确总判定：当前代码是否足以产出可信最终实验、哪些路径可继续、哪些测量/声明必须暂停、最致命的 5 项问题。然后交付：

1. **版本与覆盖表**：实际读到的 commits、关键文件、未读/不可访问部分；F00–F13 与 A/B/C/基线/训练/评估/硬件/绘图每项覆盖状态。
2. **问题清单**，按 P0（结果可能失效/错误结论）、P1（合法任务失败或比较偏差）、P2（非阻断缺陷）、P3（表述/维护）排序。每项必须给：确定程度（代码已证实/合理疑点/缺证据）；固定 commit 的 `file:line` 与关键语句；实际可达的触发配置；完整错误链；对已有/未来结果的影响；最小修复；能够证伪此问题的最小测试或对照。不要用“建议检查……”代替完成审查。
3. **矛盾表**：用户最新要求 ↔ 原始方案 ↔ manifest ↔ 实际代码 ↔ 运行凭证 ↔ 论文/图表说法。明确哪个版本过时、哪个行为当前仍错。
4. **逐路线结论**：实现、正确性、可比性、测量充分性、路线可行性分别判断。明确区分“尚未完成实验”和“已证明无效”。
5. **最小修复顺序和补充验证**：先修影响正在训练和最终评价的错误；说明哪些现有检查点可保留、哪些结果须撤销、哪些真需重跑以及原因。避免泛化安全加固、无证据的重构或无限实验。
6. **不超过 300 字的行动结论**：现在最值得立刻做什么才能最快得到可信的最终性能和路线判断。

允许结论是“这处实现正确”。严厉意味着可核验地指出真实问题，不意味着否定一切。已有 `KNOWN_ISSUES` 仅是线索，必须独立核查，也必须继续找未列出的问题。测试通过数量、预检 PASS、作者结果以及当前作者自述都不能替代代码证据。
