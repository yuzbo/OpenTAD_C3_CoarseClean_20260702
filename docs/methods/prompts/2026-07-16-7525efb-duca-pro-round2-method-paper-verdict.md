# DUCA `7525efb` Pro 第二轮：最终方法、训练与论文路线裁决

你现在是 **CVPR/CCF-A 方法领域主席、离线 TAD 研究负责人和最强反方审稿人**。第一轮已经负责代码、数学和训练合同审计。本轮不要重新逐行扫描整个仓库；只在第一轮交接包指出矛盾时打开对应文件核验。

本轮唯一任务是：**基于第一轮已核验的实现事实，裁决 DUCA 是否值得继续，以及论文最终模型、监督、训练和最小实验闭环到底应该是什么。**

## 1. 固定代码对象

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 精确提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/7525efb2e07214615a59c482443246174a6adaf1>
- SHA：`7525efb2e07214615a59c482443246174a6adaf1`
- 上游 AdaTAD/OpenTAD：`sming256/OpenTAD@1aa8ca4ac5e846b1e8ff69298dd6607121a01589`
- 上游 ASFormer：`ChinaYi/ASFormer@e1bbe4f3ed083748f91467c51a63ac2a8b9277ad`

## 2. 第一轮交接包

将第一轮回答末尾的完整 YAML 放在这里：

```yaml
<<PASTE_ROUND1_HANDOFF_PACKET_HERE>>
```

若交接包缺失、commit 不一致或 `visibility=BLOCKED`，只输出 `ROUND1_EVIDENCE_REQUIRED`。若第一轮存在 P0 blocker，仍可讨论最终设计，但必须把“修复 P0”放在所有训练之前，不得假设它已修复。

## 3. 不可改变的研究边界

任务是 **离线 TAD 的 pre-backbone 时序去冗余**，不是 Online TAD。模型可观察完整低成本窗口，再决定哪些原始 RGB 帧进入重型 backbone。推理禁止 GT、teacher、oracle、ledger、预测缓存和外部 JSONL。

研究初心是把小模型不擅长的精确边界回归，转换为“binary action/background state -> 状态转变与不确定性 -> 间接边界证据 -> 预算约束选帧”。actionness 只能是粗状态监督；selector 必须以 transition/boundary coverage 和下游效用为中心，不能退化成 actionness top-k。

当前核心问题固定为 `T=768 -> K=384`、`max_unselected_hole=15`。在 fixed-384 机制被证实前，不得用 X3D、SlowFast、MUST、动态预算、第二检测头或 physical-grid 转移问题。

## 4. 最终方法裁决

结合交接包，必须在以下路线中只选一个，或直接 KILL：

1. 保留 `binary state -> transition evidence -> structured fixed-K -> signed hard-swap utility`，但做必要简化；
2. 保留 transition-only，删除 counterfactual teacher；
3. 改为 coverage-preserving local-cell deformation；
4. 用更简单、可证明的 detector-aware 目标替换 signed proximal；
5. `KILL DUCA`：说明智能选帧为何无法可靠超过 uniform，并给出停止依据。

不要给五个并列 idea。先构造当前路线最强失败反例，再给唯一推荐。若推荐替换 detector-aware 目标，只允许在 baseline-anchored logistic、signed regression、当前 proximal、local-cell counterfactual 中比较；明确目标公式、候选生成、梯度去向、推理时删除项和为何比当前实现更简单。

## 5. 初心一致性与模块边界

明确回答：

- ASFormer encoder 是否应只受 binary actionness 更新，还是允许 transition loss 更新共享表示；
- boundary/endpoint GT 应监督哪个模块，怎样避免把粗分类器变成直接边界网络；
- selector 应看到哪些 hidden/state-difference 特征，哪些绝对特征应禁止；
- detector-aware 信号应采用 detached hard utility、直接可微代理还是不使用；
- exact-K/max-gap 应作为统一结构化可行族还是工程 repair；
- selected-axis detector geometry 是否足以做主方法，若不足，最小解决方案是什么；
- 训练时额外 detector passes 是否可以接受，推理成本优势是否仍有意义。

输出一份最终模块表：`模块 / 输入 / 推理时存在 / 训练监督 / 可接收梯度 / stop-grad 边界 / 参数量与主要成本`。

## 6. 最终训练方案

给出一个完整但简洁的最终训练合同，不要再提出三阶段独立训练作为主方法。至少说明：初始化、是否使用 ImageNet/Kinetics/ASFormer 预训练、哪些模块从零训练、loss 公式和权重角色、gradient routing、uniform-to-learned curriculum 是否保留、何时启用 detector-aware supervision、AMP/DDP/successful-update、EMA/checkpoint 和终止条件。

必须区分：

- `binary actionness supervision`；
- `transition/boundary coverage supervision`；
- `detached detector utility`；
- `official detector cls/reg loss`。

不得把 proxy 命名为真实 detector utility，不得把 nonzero gradient 当作方向正确。

## 7. 最小论文实验闭环

只围绕三个可检验 claim 设计：

- `C3`：transition policy 是否优于同协议 matched exact-uniform；
- `C4`：detector-aware 监督是否优于 transition beta=0；
- `C7`：probe + selector + sparse heavy stack 的真实端到端推理成本是否优于 dense stack。

请裁决以下臂是否保留：bare-uniform cost baseline、matched exact-uniform、transition beta=0、transition + 最终 detector-aware 目标、direct-boundary attribution。所有臂必须同 commit、seed、data order、成功更新数、EMA、evaluator 和 terminal checkpoint 规则。

为每个 claim 给出：最小 seed 数、主指标、mAP@0.6/0.7、short-action、boundary recall/distance、max-hole、utility sign/rank、p50/p95、显存与能耗，以及 preregistered `GO/HOLD/KILL`。历史 learned 64.34、uniform 64.352/65.696、oracle 约 78 只能作为不匹配背景或 privileged ceiling，不得进入 matched 主表。禁止用 THUMOS test 中间 mAP 选 checkpoint。

给出严格执行 DAG：`P0 code fixes -> exact-commit synthetic contract gate -> real-loader CUDA gate -> forced-overflow/mixed-batch pilot -> matched seed-0 -> result-to-claim -> additional seeds -> cost -> 才允许扩预算/检测头`。若 seed-0 不超过 exact-uniform，给出停止规则，不得继续无界调参。

## 8. 新颖性和可发表性

检索并引用可核验的一手论文或官方代码，至少比较 AdaFrame、Action Sensitivity Learning、AdaTAD、TE-TAD、TAPS、Progressive Block Drop 及近期 task-aware/adaptive video computation。不要泛泛列 related work；建立“已有工作解决什么 / DUCA 真正新增什么 / 是否只是工程组合”的差异表。

分别给出最强 novelty attack、methodology attack、efficiency attack。判断唯一推荐方法在什么实证门槛下可达到 CVPR/CCF-A；若达不到，应把论文主张降为什么，或是否应 KILL。

## 9. 当前证据边界

- 当前精确提交没有 CUDA gate、真实 loader gate、pilot、full train、mAP 或成本结果。
- `160 passed, 7 skipped` 与 Job `1165646/1165650/1165654` 均为项目方自报、非论文性能证据。
- 旧 Jobs `1164700-1164703` 因成功更新数不足失效。
- 第一轮若判定 signed proximal 数学通过，也只表示实现方向合同通过，不表示 C4 有效。

## 10. 强制输出格式

正文控制在 **7000 个中文字符以内**，按顺序输出：

1. 读取的 `HANDOFF_PACKET` 摘要与证据边界；
2. 一句话总裁决：`KEEP / REDESIGN / KILL`；
3. 当前路线最强失败反例；
4. 唯一最终架构与模块表；
5. 唯一训练目标、公式与梯度路由；
6. 相对当前 `7525efb` 的最小实现变更清单，必要时给核心代码片段，但不要再做全仓逐行审计；
7. real-loader/CUDA/pilot/matched 实验 DAG；
8. C3/C4/C7 preregistered result-to-claim 表；
9. 真实成本与新颖性裁决；
10. 最终结论：再次训练前必须满足什么，以及什么结果将永久停止 DUCA。

任何 proposal 都必须明确标为 proposal。不要因第一轮 `GO_TO_REAL_GATE` 就宣布方法可发表；不要因当前没有 mAP 就凭直觉 KILL。最终回答必须给出一个能直接落实的唯一方法，而不是新的 idea 清单。
