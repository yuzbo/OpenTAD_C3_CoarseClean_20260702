# DUCA 全历史、代码与论文路线综合裁决

Nonce：`DUCA-COMPREHENSIVE-ROUTE-INTEGRATION-v001-20260831`

你是本课题的科学第一负责人、整体科研流程维护者、方法设计者和论文首脑。你独立负责科学问题、创新机制、
可证伪预测、实验路线、结果解释、停止/转向决策与论文主张。Codex 只负责忠实整理上下文、执行你冻结后的
最小实现、独立代码审查、正式实验评估和证据回传。本轮必须由你自己作科学判断；不要让 Codex 在候选路线中
代选，也不要因为当前材料提出了某条路线就默认继续它。

## 精确身份与公开代码

- ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
- GitHub 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 当前最可靠 H65 科学基座与完整代码提交：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
- H65 分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-h65-60-curriculum-20260823>
- 最近的真实变长三档预算和整视频诊断提交：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
- 该诊断分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>

其他需要逐项核验的公开历史提交：

- CellCF：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/1642f265e48391418a7c8a4a087e33e2b7bf6899>
- Protected E2E：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b3222af0895e23eca83113977c1bcfad75258c9e>
- Sparse hidden-linear reconstruction：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45>
- K192 curriculum：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ed0d4900bffe3546997ea1f00ae806d82cad55f2>
- TrueTime：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11126684af779aa2916a68ecf617c4f14c805478>
- Native-tubelet coreset：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b33391126eac05e3353d322b973dda91741f0732>
- Dynamic native-tubelet window budget：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610>
- Coverage-v1：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/048143124e2a36a76575200ae17d6f42ec79ea3a>
- Marginal-v1：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889>

`04c35a3...` 是当前模型科学基座；`33e4ed...` 是已经完成的诊断代码身份，不得把诊断分支自动升级为下一模型
基座。协调根目录是脏工作树，不是模型身份。请以精确公开提交、随附文件和原始实验记录为依据。

## 本轮任务

DUCA 已经尝试了很多方向。请对随附材料逐文件、逐行阅读，并对上述公开提交中的相关实现做实现忠实度检查。
你需要把各路线、代码、训练协议、正式或诊断结果、失败原因和证据质量整合成一份可以直接指导后续科研的裁决，
回答两个核心问题：

1. 哪一个仍未被已有证据否定、最可能产生论文级信息增益的方向值得继续优化？
2. 从当前状态出发，怎样通过最少但决定性的完整实验，得到一个最终可投稿的 DUCA 模型与证据包？

这不是让你“罗列所有可能方案”。你必须判断、收缩并给出唯一当前科学路线；若证据要求停止 DUCA，也应明确
STOP，而不是为了继续项目而制造新机制。

## Gemini 前置独立咨询

本轮随附一份由 `agy` CLI 的 `gemini-3.7-flash-high`、`effort=high` 在只读模式下完成的全历史与代码咨询报告，
以及 Codex 对该报告的事实校正说明。请完整阅读两者，把 Gemini 当作简洁的独立技术意见，而不是第二个科学首脑。
你必须自行检查其引用和推理，并独立作出最终裁决。

尤其不得照搬 Gemini 的以下过度结论：跨预算表示不匹配尚未被证明为 0/704 的根因；旧 E2E 的随机 stem 与
surrogate-gradient 解释未被完全隔离；native-tubelet 连续性仍是未做严格同 RGB 对照的假说；211/212 held-out
身份仍未准入。Gemini 给出的概率、优化器参数、门槛、实验矩阵、期限和停止规则都是建议，是否冻结由你决定。

## 必须核验的事实边界

### 当前可靠参考

- Shared Dense AdaTAD 参考 Avg-mAP 为 68.73；其计算为重型输入 100% 的参考上界，但当前材料没有统一的
  mAP@0.7 和完整同硬件成本表。
- H65 fixed K384 的登记结果约为 Avg-mAP 65.13、mAP@0.7 43.31。它将重型输入 observation 从 768
  降为 384，因此支持“重型输入数量减少 50%”；没有完整端到端 latency、energy、memory 证据时，不能写成
  “端到端计算减少 50%”。
- H65 不是简单全局 Top-K，而是由低成本动作/边界状态调制、带均匀覆盖下限的预算校准系统采样，hard forward
  为 exact-K、严格递增物理位置。
- 旧 fixed-K384 end-to-end 结果 58.39/34.53 存在 loader exposure 不匹配，只能说明旧实现失败，不能直接
  证明联合训练一般不可行。

### 已有结果及其限制

随附 `DUCA_COMPREHENSIVE_ROUTE_EVIDENCE-v001.md` 是中立索引，不是科学裁决。至少核验以下类别：

1. fixed-K selector/utility：transition-only、CellCF、protected-E2E、boundary-burst、R0 oracle、Fast-only；
2. 训练课程：30+60 H65、60-epoch exact-uniform、压缩 20+40/30+30、K192、homotopy、CellCF distillation；
3. 时间几何：PhysTime、physical metric、TTDI/SingleClock、RankPack/TrueTime；
4. 采样单元与重构：continuous cliplet、native tubelet uniform/coreset、sparse hidden-linear bridge；
5. 动态计算：dynamic native tubelet、Coverage-v1、Marginal/cap-release、96-state、704-state whole-video transfer；
6. PJST-D1 和其他未闭环路线；
7. 当前尚未实现的多预算检测器适应。

必须把以下几类内容分开：

- 正式模型性能证据；
- 单种子或诊断性点估计；
- 训练内部 oracle 或无标签机制门；
- 实现/启动器/封存失败；
- 尚未执行的设计假说；
- 只有 observation 数而没有端到端成本的效率陈述。

不得把工程失败写成科学负结果，不得把局部测试或运行成功写成模型有效性，不得把目标值写成已达成结果。

## 当前数据与执行阻塞

上一轮同一 Project 的 Pro 裁决为 `REVISE`。它保留“固定 K384 控制与 K256/K384/K512 多预算检测器适应”的
单变量科学问题，但先要求完成 THUMOS14 数据身份核验：

- 正式训练必须使用 annotation 中完整的 200-video `training` 集合；
- 正式最终评估必须使用完整独立 held-out 集合；
- OpenTAD/DUCA 历史 211-video `validation` 与 ActionFormer 历史 212-video `test` 尚未完成 literal ID、loader、
  physical file、evaluator 和排除规则核验；
- 身份核验与独立审查、CPU Evaluator、Pro 数据准入前，不允许加载 checkpoint、建立候选模型、提交 PRE_RUN/GPU/
  训练、生成 held-out predictions 或计算 mAP。

本轮可以重新审视多预算适应是否仍应排在第一位，但不得假装 211/212 已解决，也不得用 held-out 结果反复选线。
请把“当前必须先完成的数据事实任务”与“数据准入后的模型科学任务”明确分开。

## 四个最新防重复问题

请特别检查以下四项，不要误判为空白路线：

1. **原生连续 tubelet**：连续 cliplet、native-tubelet uniform/coreset、CONTIG bundle 已有实现或训练；尚无同一 RGB
   集合只改变 tubelet 内连续性的严格对照。
2. **显式物理时间**：PhysTime 有负结果，physical-metric 在低性能架构中有正结果，RankPack/TrueTime 的匹配点估计
   为 61.5722 对 62.1930，TrueTime +0.6208 Avg-mAP，但单种子、paired CI 未闭合。CT-RoPE 未实现。
3. **稀疏到密集重构核**：hidden-linear bridge 有测试和 CUDA gate，但没有正式 TAD kernel comparison；nearest/linear/
   Gaussian 对边界质量的单变量实验未完成。
4. **课程/蒸馏**：30+60 H65 达到 65.385724，但总训练 90 轮，不能与 60 轮控制作公平课程归因；homotopy 与
   CellCF utility distillation 未超过相关控制；等成功 update 的 scratch/warmup 或 distill/no-distill 尚未闭环。

如果你选择其中任何一项继续，必须明确它与历史实验的唯一差异，禁止原样重复。

## 逐行代码检查要求

请完整阅读随附的 H65 `acquisition.py`、Stage-1/Stage-2 configs、TrueTime residual、whole-video
`dynamic_budget.py` 和 falsifier runner，并结合精确 GitHub 提交检查：

1. 论文/文档描述是否与真实 forward、训练可训练参数、预算单位、物理时间映射、checkpoint/EMA 和 evaluator 一致；
2. 哪些代码表面是当前可靠方法核心，哪些只是历史诊断、重复实现、未调用代码或会导致路线歧义；
3. 哪些旧路线应保留为 baseline/归因工具，哪些应归档、从当前主线移除或禁止继续修改；
4. 当前 H65 与三档预算诊断之间能够安全复用的最小代码是什么；
5. 是否存在会让下一决定性实验无法回答其科学问题的多变量混杂、训练分布偏移、成本伪动态、时间几何错误、
   prediction/evaluator 身份或 checkpoint 恢复问题。

请给出 `file:line` 或 GitHub 文件 URL 与符号名。若你无法访问某个 GitHub 文件，请明确写 `NOT_INSPECTED`，
不要假装逐行检查已经完成。

## 科学整合与路线判断

请对每一主要路线给出明确处置：`RETAIN_BASELINE / RETAIN_DIAGNOSTIC / CONTINUE_CANDIDATE / STOP / ARCHIVE`。
不要用总分表代替判断。对仍可继续的候选，说明：

- 未被旧证据回答的精确科学问题；
- 论文级创新点，而不是工程组件堆叠；
- 最强竞争解释和最小 falsifier；
- 为什么它比其他尚未闭环方向更先做；
- 失败后会关闭什么，成功后才解锁什么。

随后只选择一个唯一方向。不要在第一轮同时加入多预算训练、预算 embedding、蒸馏、新 selector、CT-RoPE、
Gaussian reconstruction、Mamba、Block Drop、TensorRT 或跨数据集扩展。若最终方法需要多个阶段，请按单变量证据
依赖顺序解锁，而不是一次合并。

## 论文可发表路径

请设计一条从当前状态到投稿证据包的最短闭环，至少明确：

1. 一句话论文问题、核心机制、最多两条可证伪主张和明确非主张；
2. 当前数据身份核验如何结束，以及模型工作何时才可开始；
3. 唯一下一项 Builder -> independent Critic -> Evaluator 任务；
4. 完整 200-video training、完整 held-out evaluation、同起点、同 successful updates、同 optimizer/LR/EMA/seed/
   trainable-set/evaluator/NMS 和真实 observation 成本口径；
5. 所有正式候选必须保存 sealed predictions、per-video identity、完整原始指标和成本；
6. 单种子机制门、三种子正式主结果、10,000 次 paired whole-video bootstrap 各自何时使用；
7. Dense、H65、exact-uniform、random、actionness-only/simple-transition 和最终候选中的必要 baseline，避免无意义矩阵；
8. Avg-mAP、mAP@0.3--0.7、短中长动作、proposal recall、起止误差、校准/NMS、actual observation、FLOPs、
   latency p50/p95、throughput、memory、energy 中哪些是决定性输出；
9. 何时才允许第二 detector、第二数据集和部署优化；
10. 以当前日期 2026-08-31 为基准，给出你认为现实的绝对里程碑与最终“可投稿/停止”判定点。

不能把完整 held-out 用于设计、checkpoint 选择、阈值、超参数、规则或路线选择。若需要训练侧诊断，必须说明它
只回答什么，并在正式完整训练前冻结。最终提交仍由人类确认。

## 返回格式

请返回一份可以原样保存的完整报告，并严格包含：

1. `SESSION_ASSERTION`：回显 nonce、Project ID、H65 base、whole-video diagnostic commit 和实际使用的材料；
2. `SCIENTIFIC_DECISION`：只选 `CONTINUE / REVISE / PIVOT / STOP` 之一；
3. `EVIDENCE_INTEGRITY_AUDIT`：逐路线校正结果、代码身份、正式性和不可声称内容；
4. `LINE_BY_LINE_IMPLEMENTATION_REVIEW`：按文件/符号给出实现忠实度、错误、未调用表面与复用边界；
5. `ROOT_CAUSE_SYNTHESIS`：区分已证实根因、最强假说、工程失败和未知项；
6. `ROUTE_DISPOSITION`：对全部主要路线作 retain/continue/stop/archive 判断，并明确禁止重复项；
7. `UNIQUE_RESEARCH_ROUTE`：唯一论文问题、机制、falsifier、成功/失败门和为何优先；
8. `CODE_ORGANIZATION_DECISION`：唯一 clean mainline、保留/复用/归档/移除的代码表面；
9. `CURRENT_TASK_ORDER`：当前数据任务与数据准入后的唯一 Builder、Critic、Evaluator，写清输入、输出和禁止项；
10. `FULL_EXPERIMENT_AND_STATISTICS_PLAN`：完整训练/留出评估、baseline、seeds、sealed predictions、成本与不确定性；
11. `PUBLICATION_PLAN`：论文主张、表/图、创新风险、必需证据、局限性和停止点；
12. `ABSOLUTE_MILESTONES`：绝对时间、依赖、失败时的处置；
13. `NEXT_RETURN_CONTRACT`：Codex 下一次必须带回的代码、正式实验和证据；
14. 最后一行只写：`DUCA_COMPREHENSIVE_RESEARCH_DECISION_READY`。

不要要求构建新的工作流平台、通用审计系统、复杂合同代码或为了“完整”而加入大量并行实验。科研与模型创新优先，
但公平性、无泄漏、证据真实性和完整训练/完整留出评估不可牺牲。
