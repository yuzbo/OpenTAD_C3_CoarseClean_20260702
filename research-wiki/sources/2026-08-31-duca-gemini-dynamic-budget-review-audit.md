# DUCA 动态预算外部分析的证据审计与吸收记录

日期：2026-08-31  
来源：用户在当前 Codex 对话中粘贴的 Gemini 分析报告  
来源身份边界：报告正文可见，但本记录未独立验证其 Gemini 具体模型档位、生成会话或引用文献  
权威代码：[`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)  
权威分支：[`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)

## 结论

这份外部报告可以作为未来机制候选和审查问题清单，但不能作为当前 DUCA 的实验结论、根因证明或直接实现计划。

它正确抓住了三个重要问题：稀疏重型输入不等于等比例端到端加速；当前三档预算转移空间没有同时改善平均检测精度和高时间交并比定位精度的充分余量；跨预算表示兼容性值得在新的科学问题中检验。与此同时，报告把若干尚未验证的解释写成了既成事实，并把多个高风险机制捆绑为一条路线。当前权威科学状态仍是 Pro 对现有三档预算转移路线的 `STOP`，没有新的实现或实验授权。

## 与权威证据一致的内容

1. H65 使用低成本侦察信号指导进入重型 VideoMAE 路径的时域观察分配。
2. 从 768 个候选时间位置向重型路径提供 384 个观察，表示重型输入数量减少 50%；这不能直接写成端到端时延、吞吐量、显存或能耗减少 50%。
3. 当前项目记录的参考点包括：
   - H65 30+60：Avg-mAP 65.13%，mAP@0.7 43.31%；
   - 原生 tubelet 均匀选择 K=384：64.13%，42.45%；
   - 任务状态 coreset K=384：62.81%，40.56%；
   - 共享 Dense AdaTAD 参考 Avg-mAP 68.73%，但它不是已经证明只由稀疏化造成差异的严格配对因果对照。
4. 整视频一致预算诊断在 40 个训练侧 controller holdout 视频、124 个窗口上枚举了 704 个合法候选；没有候选同时达到预登记的 Avg-mAP `+0.8` 个百分点和 mAP@0.7 `+1.0` 个百分点门槛。
5. 平均检测精度最优状态和高时间交并比定位精度最优状态发生分离，说明当前动作空间不能通过一次简单预算转移获得预登记的联合收益。
6. 多预算条件训练或跨预算兼容表示是合理的未来假设；Pro 也把“仅在 K=384 训练的检测器与 H65 priority sequence 缺少跨预算兼容表示”列为当前最强但尚未验证的解释。

## 必须修正的实现描述

### H65 不是字面意义的全局 Top-K

报告把 H65 描述为“按重要性全局降序后截取 K=384”。当前代码中的 `budget_calibrated_sampling_rate` 实际执行的是：

- 将逐位置优先级校准为总和等于 K 的 capped sampling rate；
- 混入均匀覆盖下限；
- 对累计 rate 使用固定阈值的 systematic sampling；
- 产生严格递增的原始时间索引。

因此，未来正文应使用“由 H65 优先级调制、带均匀覆盖下限的预算校准系统采样”，不能继续简化成普通全局 Top-K。相关实现位于 [`opentad/models/duca/structured_selection.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/structured_selection.py)。

### 704 个候选不是训练出的动态控制器

终态实验由 [`run_duca_whole_video_consistent_budget_falsifier.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py) 使用已经密封的 K=256、384、512 预测进行训练侧开发集 privileged oracle 枚举。它没有：

- 模型前向传播；
- 新训练或梯度更新；
- 学习一个可部署控制器；
- 官方 validation/test；
- bootstrap 或不确定性区间；
- 端到端硬件成本测量。

因此，“离线控制器过拟合导致 704 个候选失败”不是证据支持的结论。更准确的结论是：即使允许使用密封预测进行事后 privileged oracle 搜索，当前 K256/K384/K512 整视频预算转移动作空间仍没有表现出足够的联合 oracle headroom。

### 当前动态预算不是 padding 后的名义动态 K

项目规则已经明确禁止把所有窗口 padding 到最大预算后声称节省计算。现有 producer 与成本记录使用不同实际观察数量，并在终态诊断中核算实际 observation cost。报告关于动态长度可能引入 padding 和 kernel 调度开销的警告在一般意义上成立，但它不是当前终态实验已经观察到的失败原因。

## 仍只是解释性假设的内容

下列说法在论文直觉上合理，但当前证据没有直接测量，不能写成根因：

- K=256 必然因边界帧缺失导致 mAP@0.7 下降；
- K=512 必然因新增背景噪声破坏检测头感受野；
- 分类天然只需要低预算，而边界定位天然需要高预算；
- 检测头的相对位置编码是跨预算退化的主要来源；
- 当前性能差距主要由输入分布偏移而非侦察优先级、预算档位、训练目标或其他因素造成；
- Dense 与 H65 的 3.60 个百分点差距可完全归因于稀疏化。

如果 Pro 以后重新开放这一问题，上述解释必须被改写为可证伪预测，并由单一变量的配对实验区分，而不是作为实现理由预先成立。

## 对建议路线的取舍

### 可保留为候选科学问题

- 多预算条件训练是否能建立跨预算兼容、单调且边界敏感的表示；
- 在匹配真实计算下，局部预算重分配是否比整视频三档转移更有联合 oracle headroom；
- 真实端到端成本与 observation 数、重型骨干计算、数据搬运和检测头开销之间的关系。

### 不应直接进入当前实现

- 同时引入 Gumbel-Softmax、双通道分类/边界 Scout、预算条件嵌入、知识蒸馏、Mamba、Block Drop、自定义 CUDA Gather 和 TensorRT；
- 把 DFT Scout、Mamba 或层级剪枝写成当前失败的必然解；
- 在没有测量前承诺 40%–45% 骨干时延收益、15%–20% TensorRT 收益、60% 端到端时延下降或 68.0% Avg-mAP；
- 立即扩展到 THUMOS14、ActivityNet-1.3 与 EPIC-Kitchens-100 的多数据集联合验证。

这些组合会同时改变选择机制、训练分布、表示、检测器、骨干深度和部署栈，无法定位收益来源，也不符合当前论文优先、单一可证伪任务的执行原则。

### Gumbel-Softmax 的边界

可微选择只解决梯度估计问题，不自动保证推理时真正减少重型观察数量，也不自动保证匹配真实计算。如果以后采用这一机制，仍需证明：

- 推理执行的实际 heavy clip 数发生变化；
- 没有通过 padding 或全量骨干前向隐藏计算；
- 比较使用匹配的真实平均计算；
- 离散推理策略与训练时软选择之间没有不可接受的落差。

## 当前科学处置

1. 保留 Pro 的现有 `STOP`：停止当前 THUMOS14 训练侧 holdout、冻结 H65 detector/priority sequence、K256/K384/K512 密封预测和 observation-transfer 动作空间内的继续搜索。
2. 本外部报告不改变 `PAPER_PROGRESS.md`，因为它没有产生新实验事实、独立评估结果或新的 Pro 裁决。
3. 当前没有 Builder、Critic、Evaluator 或 Slurm 任务；不从本报告直接建立新分支或启动实验。
4. 若人类决定继续，下一次应向 Pro 提交中立材料，由 Pro 独立选择是否把“跨预算兼容表示”或其他单一机制改写为新的科学问题，并冻结一项最小、可证伪、匹配真实计算的任务。
5. 新路线必须位于本次 `STOP` 边界之外，并首先在独立训练侧开发划分上展示预登记的联合 oracle headroom；不能通过事后更换候选、阈值或指标重新解释旧结果。

## 追溯材料

- 权威 Pro 报告：`.cvpr-pro-lab/pro-reviews/runs/duca-whole-video-terminal-adjudication-v001/visible-report.md`
- Gemini 审查 Prompt：`.cvpr-pro-lab/gemini-reviews/prompts/GEMINI_DUCA_CURRENT_EXPERIMENT_CODE_REVIEW-v001.md`
- 终态实验结果：`/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`
- 终态结果 SHA-256：`40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`

