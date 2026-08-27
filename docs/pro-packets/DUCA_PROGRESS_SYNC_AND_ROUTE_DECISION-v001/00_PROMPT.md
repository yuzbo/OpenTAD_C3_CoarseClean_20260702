# DUCA 进度同步、路线裁决与代码整理问询

你是本项目的首席科学负责人、模型与实验设计者、证据审阅者和当前任务决策者。Codex 是执行者：负责按你的科学决定完成代码实现、独立审查调度、实验执行、证据整理和研究记忆更新。人类保留论文作者确认、资源与凭据授权以及最终投稿决定。

本轮不是请你批准 Codex 已选好的方案。Codex 只提供权威上下文和可核验代码。你可以否定当前问题表述、指出材料冲突、提出严格更好的路线，并独立下达唯一下一任务。下面列出的历史路线不是未来候选的穷尽集合，也不代表偏好顺序。

请先完成**路线决策与代码整理**，再下达实验或实现任务。不要把选择交回人类或 Codex；不要用多个并列方案代替判断。

## 一、权威 GitHub 材料

Repository：

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702

### 1. 当前代码与 Wiki 库存快照

固定提交：

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/5136011ed57df8a639427a633a488a592ba95924

分支：

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-research-sync-20260828

这个分支同步了当前 `opentad/`、`configs/`、`tools/`、`scripts/`、`tests/`、DUCA Wiki 和论文进展。它是用于整理的**代码库存**，不是已经验证可运行的实验候选。它从 PJST-D1 终态评估提交 `7bd120f0…` 建立，再叠加项目根当前代码；项目根包含多条历史与并行路线，因此共享文件可能存在科学身份冲突。

先读：

- 当前研究状态：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/DUCA_CURRENT_RESEARCH_STATE-2026-08-28.md
- 论文缩略报告：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/PAPER_PROGRESS.md
- 代码库存边界：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/docs/pro-packets/DUCA_PROGRESS_SYNC_AND_ROUTE_DECISION-v001/01_CODE_INVENTORY_BOUNDARY.md
- 研究 Wiki 入口：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/index.md
- 角色与科研协作规则：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/PRO_RESEARCH_ROLE_RULES.md
- 项目规则：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/RTK.md

需要追溯时再读：

- DUCA 全量记忆审计：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md
- 决策历史：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/decision_history.md
- 防重复记录：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/anti_repetition.md
- 实验流水：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/log.md
- 模型版本登记：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/duca_model_version_registry.md
- pre-backbone 方法与 baseline 边界：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md
- 数据与远端资源：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/research-wiki/REMOTE_DATA_RESOURCE_MAP-2026-08-17.md
- PJST 外部审查的本地核验：
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/5136011ed57df8a639427a633a488a592ba95924/docs/methods/2026-08-25-b2ccfcca-duca-pjst-pro-review-absorption.md

### 2. 不能被库存分支替代的干净实验提交

请直接检查这些固定提交的代码，而不是只读摘要：

1. H65 30+60 参考：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854
2. H65 第二阶段学习率诊断：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f
3. First-Mixing SingleClock：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b2ccfccab5b4912b59954afcc9b0364955327f7c
4. PJST-D1 匹配训练：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/c73e8418de31cdcb2a445ff58a1e33ab9ab6a508
5. PJST-D1 终态只读评估：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7bd120f0d342bf175c97c365fba7cbd359df055e
6. UVT：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/df544c78ce515d925dc7019f106fce09a53c09f8
7. Fovea/Query-Bridge：
   https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ae5067100c4490c7110c00a1ad406230ba603cd

这些实现可能修改相同文件，不能机械合并。请把“代码库存可见”与“实验身份可归因”严格区分。

## 二、科学问题和人类给出的边界

DUCA 的长期问题是：在离线时序动作检测中，能否用低成本侦察模型预测逐时刻二元动作性和动作起止边界重要性，再由确定性规则间接选择非均匀原始帧，并根据逐视频或逐窗口语义证据分配动态预算，从而真实减少 VideoMAE 高分辨率计算，同时保护高时间交并比下的边界定位。

人类已明确的研究边界是：

1. 学习任务应是动作性和边界语义预测；直接学习帧索引只能作为消融或控制，不是主方法。
2. 动态预算是长期主问题；固定 `K` 只能作为归因、公平对照和回退。
3. 非连续选帧可以研究，但每个选中帧的原始时间戳或物理坐标必须贯穿重型编码与 NMS 前的检测解码；不能把稀疏序号伪装成均匀物理时间。
4. 共享的未修改官方 AdaTAD 复现全局只训练一次；DUCA 不得重复相同官方 dense 训练。
5. 正式实验使用完整 THUMOS14 官方训练/验证、官方评估器、相同检测器/损失/NMS、公平更新数和实际端到端成本。subset 或 synthetic 只能调试，不能支撑效能主张。
6. 完整训练至少每 5 个 epoch 保存可恢复检查点；最终或最终指数移动平均模型的选择规则必须预先固定，不能事后挑最好。
7. 已经多次完成的 dense、uniform、random 和固定 K 基础对照不能因流程惯性重复；只有新的科学变量需要它们且现有结果不满足公平身份时，才说明必要性。

这些是研究目标和公平性约束，不是已经获得证据支持的结论。如果你判断长期主问题在科学上不成立，应明确给出证据和更好的论文问题，而不是默默改写目标。

## 三、完整进度与实验结果

### 1. 共享官方基线

- 未修改 AdaTAD 共享复现：Avg-mAP `68.73`；论文公开锚点 `69.03`。
- DUCA 只读引用该结果，不重复训练。
- 历史 65.xx 或 66.xx 不能冒充官方 dense 结果。

### 2. H65 语义间接非均匀逐帧选择

H65 使用历史 ASFormer 语义预测和确定性非均匀逐帧选择，固定 `K=384`。它不是均匀采样。

| 实验 | Avg-mAP | mAP@0.7 | 解释边界 |
|---|---:|---:|---|
| H65 30+60 | 65.13 | 43.31 | 当前最强干净 H65 参考；单种子 |
| H65 20+40 | 62.46 | 39.94 | 比 30+60 低 2.66/3.37 点 |
| H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
| H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |

60 轮压缩搜索已经停止。现有结果否定了“简单压缩训练或只改第二阶段学习率即可无损恢复”的假设，但没有否定 H65 的语义间接选帧机制。

历史 `65.3857` 是 H65 30+60 语义间接非均匀逐帧结果。历史 `65.696` 同时改变了物理检测网格，因此不是与官方原生检测器严格匹配的纯采样结果；后续应以 `65.13` 的干净 H65 身份作匹配参考。

### 3. 物理时间表示

| 实验 | Avg-mAP | mAP@0.7 | 解释边界 |
|---|---:|---:|---|
| RankPack `K=384` | 61.57 | 37.10 | 单种子表示对照 |
| TrueTime `K=384` | 62.19 | 37.89 | 相对 RankPack +0.62/+0.79；无配对区间 |
| PJST-D1 OFF | 65.063283 | 43.646027 | 固定并重放 H65 选择；211/211 视频 |
| PJST-D1 ON | 64.590802 | 43.768766 | 只启用 PJST-D1；211/211 视频 |

PJST-D1 ON 相对 OFF：Avg-mAP `-0.47248126` 个百分点；mAP@0.3/0.4/0.5/0.6/0.7 分别为 `-0.79522098/-1.25244448/-0.14698444/-0.29049525/+0.12273884`。

两臂各产生 422,000 条预测，视频集合一致，epoch-59 指数移动平均检查点的重推理精确复现点值。预登记的 10,000 次整视频配对自助法没有开始：终结器寻找 `work/result_detection.json`，而真实输出为 `work/gpu1_id0/result_detection.json`，因此 `0/16` shards、`0/10000` replicates，没有置信区间。

据此只能说平均点估计没有正向支持，不能说总体效应已经显著为负；mAP@0.7 的小幅正差也不是可靠收益。路径故障是证据生成失败，不是模型效能结果。上一执行链已关闭，当前材料不自动授权重试。

### 4. UVT 与 Fovea/Query-Bridge

- UVT legacy / geometry / geometry+EMA Avg-mAP：`57.35/55.93/55.92`。
- Fovea/Query-Bridge 第一波最佳 `query_cycle` Avg-mAP：`54.67`。
- UVT 同时改变选择分数与动态预算证据；Fovea/Query-Bridge 也改变了选择机制和训练信息流。
- 它们与 H65 不在同一提交、同一严格归因合同下，不能把跨版本性能差归因于某个 Query、价值头或协同训练组件。
- 它们的 Query 前后协同、知识传递和低成本语义建模仍可作为设计材料，但不得因为概念吸引力而绕过 H65 匹配归因。

### 5. 已停止的连续片段路线

- FZ：Avg-mAP `49.89`，mAP@0.7 `29.68`。
- JT：Avg-mAP `47.24`，mAP@0.7 `26.52`。

连续 16 帧片段采样在完整单种子训练中造成明显定位损失，联合训练没有恢复，当前不应重新作为主线。它不否定低成本语义侦察或物理时间一致性的一般问题。

### 6. 尚无证据的事项

- 没有动态预算带来性能或效率收益的正式证据；
- 没有与主路线匹配的完整端到端成本结论；
- 没有多 seed 或有效配对区间支持 PJST-D1；
- 没有证据证明 UVT 或 Query-Bridge 的单个组件独立有效；
- 当前没有正在运行的 DUCA 训练。

## 四、你必须先完成的路线与代码裁决

### A. 核对事实

直接检查固定提交和库存代码，逐条指出：

1. 上述结果、代码身份或实现解释是否有错误；
2. 哪些事实被确认，哪些仍未知，哪些历史页面已经过时；
3. 当前库存中哪些文件属于 H65/PJST/UVT/Fovea/物理时间/其他并行路线，哪些共享文件存在语义冲突；
4. 当前结果是否受模型结构、训练日程、选择器、物理时间、检查点、评估或跨提交比较混杂。

若材料不足，不得猜测；请明确指出必须读取的具体 GitHub 文件或缺失原始证据。

### B. 独立决定论文路线

用一句明确裁决开头：`CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。

随后独立回答：

1. 当前最值得发表的科学问题究竟是什么；它应让领域审稿人意外在哪里？
2. 哪个机制解释预期现象，并产生官方 dense、uniform 或普通固定 K 不会自然产生的新预测？
3. H65 的 3.6 点左右 dense 差距、PJST-D1 的负向点估计、UVT/Fovea 的大幅下降分别说明什么，又不说明什么？
4. 现有 DUCA 问题应保持、修订、转向还是停止？如提出替代路线，必须说明它如何继承现有正负证据并避免重复旧实验。
5. 当前是否值得完成既有 PJST 配对统计，仅由你判断；不要因为它“最短”就默认执行，也不要因为点估计为负就跳过必要的证据闭环。

你可以讨论多个解释，但最终必须选择一个主路线和一个当前任务，不把选择交还给人类或 Codex。

### C. 指定权威代码主线与整理方案

请把科学判断落实为可执行的代码整理决定：

1. 指定一个权威干净基座提交；如果现有提交都不合适，说明如何从最接近的提交建立新分支。
2. 列出必须保留或复用的文件、类和函数，以及其科学作用。
3. 列出只应保留为历史对照、不应进入主候选的实现。
4. 指出库存分支中会造成机制混杂或重复实现的共享表面，并给出最小整理动作。
5. 冻结下一候选允许修改的文件/符号和禁止改变的 detector、loss、NMS、split、evaluator、checkpoint 与成本口径。
6. 不要设计大型重构、通用框架、兼容层、schema、证明系统或流程平台。代码整理只服务于一个清晰科学候选和一个决定性实验。

## 五、下达唯一下一任务

在完成路线与代码裁决后，请向 Codex 下达一个、且只有一个当前任务。任务必须包含：

1. 科学问题、机制、可证伪预测与相反解释；
2. 具体负责人顺序：Builder 实现、独立 Critic 审查、独立 Evaluator 运行前检查和实验；
3. 精确的基座 revision、分支、允许修改的文件和关键符号；
4. 最小实现内容、必须通过的形状/梯度/物理坐标/数据隔离测试；
5. 是否需要训练；若需要，给出完整 THUMOS14 配置、seed、更新数、检查点、最终模型选择、资源、结果根和停止规则；
6. 公平对照和指标，但不得机械重复已有 dense/uniform/random 对照；
7. 哪个结果支持、反驳或使假设未决，以及失败后如何定位根因；
8. 明确的依赖、客观 blocker 和北京时间绝对截止时间。

任务应优先产生能改变论文判断的真实结果，而不是增加合同代码、状态文档、审计脚手架或新的空讨论。确定性启动器/路径错误应采用最小修复，不得被写成科学失败。

## 六、输出格式

请使用外部评审可以直接理解的中文，按以下顺序返回：

1. **唯一裁决**；
2. **证据核对与错误更正**；
3. **当前论文问题、机制、创新性与反解释**；
4. **权威代码主线和整理清单**；
5. **唯一当前任务单**；
6. **决定性实验、指标与停止规则**；
7. **角色分工、依赖与绝对截止时间**；
8. **可写入论文与仍不可声称的边界**。

不要使用内部队列语言或自造状态码；不要复述所有材料来代替判断；不要以“建议人类选择 A/B/C”结束。
