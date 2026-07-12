# DUCA Transition-Only `0ea4e15` Pro 严格代码审查与最终方法讨论 Prompt

> 下面内容可直接提交给 GPT-5 Pro / GPT-5.5 Pro。请保留全部仓库、提交、实验和输出约束。

你现在同时扮演以下角色：

1. CCF-A/CVPR 级别的 TAD/TAL 审稿人；
2. 熟悉 OpenTAD、AdaTAD、ActionFormer、VideoMAE、ASFormer 的高级研究工程师；
3. 熟悉离散选择、结构化预测、动态规划、straight-through estimator、双层优化和多任务梯度冲突的优化专家；
4. 对实验协议、成本核算、泄漏、基线公平性和论文 claim 极度严格的复现审计者。

你的任务不是替作者辩护，也不是给出泛泛建议，而是基于下列 exact commit **逐文件核验当前 DUCA 是否在定义、代码、梯度、训练、推理、检测器几何、成本和论文主张上自洽**，然后给出一个真正可实现、可证伪、足够优雅的最终模型方案。

## 1. 强制可见性证书

请先实际打开并阅读：

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 分支：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-transition-only-20260711>
- 当前审计提交：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d>
- 缺陷修复差异：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/compare/8bfc0e549434591b9bf1a9cd5563deb0da388f92...0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d>

必须重点阅读以下 exact-commit 文件，不允许只看 README 或 commit message：

### 方法核心

- [`opentad/models/duca/transition_only.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/models/duca/transition_only.py)
- [`opentad/models/duca/structured_selection.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/models/duca/structured_selection.py)
- [`opentad/models/duca/acquisition.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/models/duca/acquisition.py)
- [`opentad/models/selectors/duca_online_frame_selector.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/models/selectors/duca_online_frame_selector.py)
- [`tools/bata/train_lowres_action_probe.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tools/bata/train_lowres_action_probe.py)

### Detector 与优化器集成

- [`opentad/models/detectors/actionformer.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/models/detectors/actionformer.py)
- [`opentad/cores/optimizer.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/opentad/cores/optimizer.py)
- [`tools/train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tools/train.py)

### 正式配置与对照

- [`duca_transition_only_fixed384_official_adatad_backend_full_train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py)
- [`duca_exact_uniform_fixed384_official_adatad_backend_full_train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/configs/adatad/thumos/duca_exact_uniform_fixed384_official_adatad_backend_full_train.py)
- [`duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py)
- [`duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/configs/adatad/thumos/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py)
- [`duca_online_official_adatad_backend_full_train.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py)

### Gate、validator 与测试

- [`run_duca_transition_only_formal_full_model_gate.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tools/bata/run_duca_transition_only_formal_full_model_gate.py)
- [`validate_duca_transition_only_p0_variant.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tools/bata/validate_duca_transition_only_p0_variant.py)
- [`test_duca_transition_only.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tests/test_duca_transition_only.py)
- [`test_duca_transition_only_p0_matrix.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tests/test_duca_transition_only_p0_matrix.py)
- [`test_duca_transition_only_fixed384_official_adatad_backend.py`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d/tests/test_duca_transition_only_fixed384_official_adatad_backend.py)

回复开头必须给出“可见性证书”，逐项列出：仓库、分支、commit、实际读取文件、关键函数及行号。若你无法访问 exact commit，必须明确停止代码事实判断；禁止根据下述摘要假装已经逐行读过代码。

## 2. 不可误解的研究任务

本任务是**离线 Temporal Action Detection**。模型可以观察完整离线窗口，但必须在昂贵 detector backbone 之前减少实际送入 backbone 的帧数。类名中的 `Online` 是历史命名，只表示 actionness/selection 在同一次 forward 内产生，不表示 streaming、causal 或 Online TAD。

最终研究初心是：

> 小模型通常难以直接精确回归语义动作边界，因此先把问题降维为逐帧“动作/背景”二分类，再从粗状态的变化、置信度变化和隐藏状态变化中**间接定位状态转换点**，优先保护动作边界附近和下游检测有用的观测，而不是执行 actionness top-k，也不是让粗分类器直接变成另一个小型 TAD detector。

必须遵守：

- actionness 用 THUMOS14 train GT segments 生成二分类监督；
- GT start/end 只能作为 train-only transition/boundary supervision，不能作为 selector 输入；
- validation/test 不得读取 GT、teacher、oracle、ledger、raw prediction cache 或外部 JSONL；
- fixed-384 是决定性固定预算候选，不是动态预算最终 claim；
- 当前方法不是 train-free；X3D/SlowFast 不是主方法；
- detector 是 official-derived OpenTAD/AdaTAD components with DUCA wrapper，不得声称源码完全未修改；
- 任何 smoke、nonzero gradient、gate 或单 seed 都不能替代最终 mAP 与成本证据。

## 3. 当前实现的事实架构，请逐项核验而非直接相信

### 3.1 Coarse probe

- 输入为 T=768 的完整低分辨率 RGB 窗口，每帧缩放到 64x64；
- 两层自定义 Conv2d spatial stem 将 RGB 转成 96 维时序特征；
- temporal module 使用官方 ASFormer `MyTransformer` 的 2-layer 配置；
- 输出逐帧 binary actionness logits 和通过 forward hook 捕获的 official ASFormer encoder hidden；
- 无 checkpoint，随整个模型从随机初始化联合训练；
- actionness BCE 由 train-only GT action intervals 监督。

请判断：“自定义随机 spatial stem + 官方 ASFormer temporal module”是否能诚实称作 official ASFormer coarse probe；它是否真的足够低成本；是否比已经验证的分离 ASFormer probe 更难优化；是否需要预训练、冻结再解冻、teacher-free distillation 或更简单的 spatial encoder。

### 3.2 Transition-only selector

每个时间点的 descriptor 当前由以下量组成：

- `delta_actionness_logit` 与绝对值；
- binary entropy 的变化与绝对值；
- ASFormer encoder hidden 的差分与绝对值；
- 相邻 hidden 的 cosine change。

不允许 absolute hidden、raw RGB mean、direct start/end/context heads 作为主路径。共享 MLP `DucaTransitionUtilityScorer` 输出 transition/policy score。

请核验以下梯度所有权：

- `actionness_logits.detach()` 阻止 transition loss 和 detector loss 更新 action head；
- auxiliary transition route 可更新 spatial stem、ASFormer encoder 和 scorer；
- policy route 对 descriptor 再次 `detach()`，因此 detector gradient 只更新 scorer，不更新 coarse stem/ASFormer；
- actionness BCE 单独更新 coarse stem、ASFormer encoder/decoder/action head。

请严厉回答：这是否仍能称为“联合协同训练”？保护 coarse semantics 是否合理，还是把 detector 对 coarse representation 的任务适配切断了？若应放开一部分梯度，请给出精确、稳定、不导致 coarse collapse 的梯度路由和损失设计，而不是一句“端到端训练即可”。

### 3.3 Structured policy

当前 policy 为 full-window exact-K/max-unselected-hole structured DP：

- T=768，requested K=384，`max_unselected_hole=15`；
- hard forward 使用 Viterbi MAP；
- backward 使用同一可行集合上的 forward-backward occupancy/slot marginals；
- detector 实际消费 hard gathered frames；
- training-only `structured_zero_forward` bridge 用
  `hard + beta * (soft - stop_grad(soft))` 将 detector loss 传给 policy scorer；
- inference 只走 hard path；无 post-hoc repair；
- short valid windows 的 effective K 可以小于 384，detector axis 再 pad 到 384。

请检查 hard MAP、soft occupancy、soft slot assignment、slot 顺序、valid-prefix、padding、AMP、梯度尺度和数值稳定性是否严格一致。尤其需要回答：soft raw-pixel mixture 的梯度是否真正近似“替换一个 hard frame 对 detector loss 的影响”，还是一个数值连通但语义错误的 surrogate？

### 3.4 Policy homotopy

当前训练使用：

```text
z_alpha = (1-alpha) * normalize(z_uniform) + alpha * normalize(z_learned)
```

- alpha 前 660 optimizer steps 保持 0；
- 随后 3960 steps 以 cosine ramp 到 1；
- detector bridge beta 在 step 4620 后启动，再用 3300 steps ramp 到 0.25；
- inference alpha=1；
- schedule 只在 successful optimizer step 后推进。

请判断“score 连续”是否足以称为 policy 连续。由于 hard Viterbi 路径会在临界 alpha 处离散跳变，这个 curriculum 是否仍存在严重 optimization shock？请比较并给出更优选择：perturb-and-MAP、Gumbel/SoftSort、Fenchel-Young structured loss、policy distillation、bilevel one-swap utility、continuation with margin、冻结 coarse 后再联合微调，或其他更合适方案。不要为了新颖而重造已有成熟结构化推断库。

### 3.5 Detector route

- hard selected raw frames 进入 VideoMAE-S adapter backbone；
- detector head 为 ActionFormerHead，head loss/NMS 配置宣称保持官方；
- detector 在 selected-axis index 上预测；
- train GT 被 remap 到 selected axis；
- test prediction 再 post-hoc remap 到 original time；
- detector loss 与 selector losses 最终在 `ActionFormer.forward_train()` 中求和。

请逐行检查这条路径是否确实是正确的 AdaTAD/OpenTAD 实现。重点审查：

1. selected-axis 是否错误地把不均匀物理时间压成等间隔，从而伤害 high-tIoU；
2. GT remap、mask、padding、projection max length、backbone total frames 是否一致；
3. `ActionFormerHead` 是否真的未被语义修改；
4. selector loss 与 detector `cls_loss/reg_loss` 是否漏加、重复加、错误缩放或改变了官方优化；
5. optimizer 是否覆盖所有且仅覆盖 `requires_grad` 参数，组件 LR 是否真实生效；
6. 第二检测头接入前，当前实现究竟是通用 plugin，还是 AdaTAD-specific wrapper。

## 4. 已确认的 uniform 缺陷与修复，必须据此重置实验结论

旧提交 `8bfc0e5` 的 `_uniform_reference_scores` 使用半整数中心
`0.5, 2.5, ...`。在 T=768/K=384 时，每个整数位置到最近中心的距离都为 0.5，归一化后 reference logits 全零，Viterbi 按平局规则输出退化路径。

复现结果：

- 与 `round(linspace(0,767,384))` 的位置重合率仅 47.135%；
- rank-aligned mean absolute error 为 179.695 帧；
- gap histogram 为 `{1:358, 10:1, 16:24}`；
- 因此 Job `1159414` 不是 exact-uniform。

当前提交 `0ea4e15` 已改为 rounded endpoint linspace，并在 formal gate/P0 validator 中要求最终 decoded positions 逐点相等、`uniform_reference_max_rank_error==0`。远端 focused verification 为 `26 passed, 2 skipped`。

但必须注意：

- 修复后尚未运行新的 formal CUDA full-model gate；
- 尚未提交 corrected replacement full-train jobs；
- 旧 gate `1159395` 不能用于新提交；
- 旧 beta0/beta0.25 同样从退化 alpha=0 起步，不能证明预期 homotopy 有效。

请继续搜索是否还有同类 midpoint/tie-breaking/stable-reference 逻辑残留，不能因为 transition-only 路径修复就假设整个仓库已干净。

## 5. 当前实验事实，严格区分诊断与正式证据

旧 `8bfc0e5` 四个作业已全部完成且 exit code 为 0，但整个矩阵已被协议失效：

| 旧 Job | 名称 | best Avg-mAP | 当前解释 |
|---:|---|---:|---|
| 1159414 | invalid alpha0 control | 55.67 | 不是 uniform，只是 DP tie-break diagnostic |
| 1159415 | direct-a5 | 57.71 | 不受 uniform 函数直接影响，但不再构成完整 matched matrix |
| 1159416 | transition beta=0 | 64.34 | learned-policy diagnostic，错误 homotopy 起点 |
| 1159417 | transition beta=0.25 | 63.55 | learned-policy/bridge diagnostic，错误 homotopy 起点 |

beta=0 最佳 checkpoint 的 IoU-wise mAP 为 `79.92/75.18/67.83/56.81/41.97`；
beta=0.25 最佳 checkpoint 为 `79.17/74.37/66.17/55.75/42.28`。beta=0.25
低 0.79 Avg-mAP，其 best-checkpoint @0.7 只高 0.31，final @0.7 反而低 0.52。
所有作业均完成 17 次评估，stderr 为空且无 Traceback/OOM/runtime non-finite skip。
这些数字不得用于“优于 uniform”或“detector gradient 有效”的论文结论。

历史上“uniform 约 65”有真实日志来源，但不是当前同协议结果：

- Job `1150701`，native stride-2 + adapter + ActionFormer：best Avg-mAP 64.352，IoU-wise `79.40/74.57/67.98/57.17/42.64`；
- Job `1150842`，grid-aware detector：best Avg-mAP 65.696，IoU-wise `80.88/76.62/68.50/58.43/44.05`。

因此当前最诚实的结论是：corrected same-commit uniform 尚未得到；transition-only 尚未证明超过强 uniform；beta=0.25 目前也未证明优于 beta=0。

训练日志还暴露出以下待解释信号：

- transition runs 中 logged transition distribution loss 约 3.15，权重为 0.5，raw 约 6.3，接近 `log(768)` 的均匀交叉熵尺度；
- balanced actionness BCE 常约 0.93–0.96，但使用了动态 positive weight，不能脱离 AUROC/AUPRC/ECE 直接判断；
- effective K 在短 valid window 上可出现 322、353 等，其他 batch 为 384；
- beta=0.25 并未稳定提高 Avg-mAP 或 mAP@0.7。

请判断性能问题更可能来自：coarse state 未学好、transition score 未学好、surrogate gradient 错向、selected-axis geometry、训练 exposure/LR、detector 被稀疏输入破坏、还是这些因素的组合。必须提出可区分根因的最小实验，而不是继续加 loss。

## 6. 必须逐行回答的核心审查问题

### A. 初心一致性

1. 当前代码是否真的实现“粗二分类状态变化 -> 间接边界定位 -> 选帧”？
2. 它是否暗中退化为 direct boundary detector、actionness top-k 或 absolute-feature shortcut？
3. transition target 使用 start/end 高斯是否仍属于合理 train-only supervision，还是已经把 coarse probe 变成隐式边界回归器？

### B. 表征与 coarse probe

1. selector 是否应该看到 ASFormer hidden，而不仅是 `p_action` 曲线？当前看到的是哪一层 hidden，语义是否正确？
2. 差分 hidden 是否足够表达长动作、渐变边界、重复动作和相机运动？
3. 两层 spatial stem + 2-layer ASFormer 的实际成本、容量和归纳偏置是否合理？
4. 为什么分离训练 ASFormer/均匀采样曾达到约 63–65，而当前联合路线不稳定？
5. 是否必须使用 MobileNet？若不必须，最简单且成本可信的空间编码器是什么？

### C. 梯度与优化

1. `policy_scores = scorer(descriptors.detach())` 是否过度保护 coarse trunk？
2. detector gradient 只更新 scorer 是否足以称作 task-adapted acquisition？
3. 是否需要让 detector gradient 以小权重进入 ASFormer encoder，但对 action head stop-grad？
4. 如何处理 actionness、transition、detector 三种梯度冲突：PCGrad、GradNorm、orthogonal projection、交替更新、EMA teacher，还是不需要复杂方法？
5. beta=0.25 的绝对尺度是否有理论或实证依据？如何归一化不同 loss 的梯度范数？
6. 当前 schedule 是否只是隐藏的多阶段训练？单 checkpoint、连续 schedule 是否足够“优雅”？

### D. 离散选择与 surrogate

1. hard/soft 是否来自完全相同的 feasible family？
2. slot marginal 是否保留顺序并与 hard slot 对齐？
3. detector 对 soft RGB mixture 的梯度是否与 one-swap hard replacement utility 正相关？
4. max-gap=15 是科学约束还是工程保险？能否改成可解释的 risk/coverage prior？
5. fixed K=384 是否应是主方法，dynamic budget 是否必须暂时降级？

### E. Detector 与时间几何

1. selected-axis remap 是否是当前高 tIoU 上限的主要瓶颈？
2. 应使用 original timestamps/physical-grid-aware head，还是保持官方 ActionFormer 并仅做输入重采样？
3. 如何设计 same-selected-frames control，把“选帧好坏”和“detector 几何损失”拆开？
4. 当前实现能否诚实称为 plugin？第二 detector 需要哪些最小接口合同？

### F. 成本与论文价值

1. 64x64 dense probe + ASFormer + exact-K DP + preprocessing/H2D 是否吞掉 50% VideoMAE 节省？
2. `structured_zero_forward` 的 training cost 为何近似 `O(numel(dense RGB)*K)`，是否不可接受？
3. inference 必须报告哪些 p50/p95 latency、MACs/FLOPs、峰值显存、energy 和吞吐？
4. uniform 不需要 probe，DUCA 需要 probe，怎样做真正公平的 accuracy-total-cost Pareto？
5. 当前组合创新性是否足够 CVPR，还是只是“轻量 probe + structured top-k + detector wrapper”的组合式工程？

## 7. 要求提出三个互斥、可落地的新方案

在完成代码审查后，请至少提出三条互斥路线，而不是在当前实现上继续堆补丁：

1. **最小修复路线**：保留 Shared-ASFormer Transition-Only，修正梯度、curriculum、geometry 和成本；
2. **更优雅的最终 DUCA 路线**：重新定义 selector utility 或联合训练，使 detector task signal 与间接边界先验协调；
3. **若 DUCA 基本假设不成立的替代路线**：例如 train-only counterfactual one-swap utility distillation、residual innovation、boundary-adaptive multigrid 或其他更有发表价值的任务感知去冗余方法。

每条路线必须包含：

- 精确输入、输出和 tensor shape；
- 哪些模块复用当前代码，哪些删除；
- 梯度流向图和 stop-gradient 位置；
- 完整数学目标；
- train/eval 同构关系；
- 推理时可见信息；
- 预期成本；
- 最大失败风险；
- 一周内可证伪的决定性实验；
- 与近期已有工作的关键区别，并只引用论文/官方代码等一手来源。

最后必须选择一条推荐路线。不能用“都可以尝试”回避裁决。

## 8. 推荐方案必须达到“最终实现规格”级别

请直接给出推荐最终模型，而不是初级逐步实验计划。至少包括：

1. 模块框图的文字版；
2. forward_train 与 forward_test 伪代码；
3. 每项 loss 的公式、权重或自适应归一化；
4. coarse、selector、detector 的梯度归属；
5. 单一训练过程的 schedule；
6. exact-K/max-gap 或替代预算约束的数学定义；
7. original-time 与 detector-time 的坐标定义；
8. optimizer param groups、LR、冻结/解冻规则；
9. 必须修改的文件和函数；
10. 关键 PyTorch 实现代码片段；
11. fail-closed tests 与 one-step/full-model tests；
12. 成功、降级、终止标准。

禁止提出新的 `asformer_lite`、伪官方模型、外部推理 JSONL 或测试 GT shortcut。优先复用当前 official ASFormer、structured DP、OpenTAD detector 和已有测试，不要重复造轮子。

## 9. 必须给出的实验裁决矩阵

请将实验按“claim -> 最小证据”组织，并明确哪些必须立即运行、哪些只有 P0 通过后才能解锁。

最低限度应审查：

- corrected exact-uniform K=384；
- direct-a5 matched control；
- transition beta=0；
- transition beta>0；
- dense 768；
- random/periodic K=384；
- same-selected-frames selected-axis vs physical-time geometry；
- one-swap finite-difference vs surrogate gradient correlation；
- no hidden / no action logits / no transition supervision；
- coarse AUROC、AUPRC、ECE；
- transition peak recall、selected-to-boundary distance、short-action recall；
- trained-checkpoint full-stack total cost；
- 至少三个 seed；
- 只有 fixed-384 通过后才允许 K=256/128、dynamic budget 和第二 detector。

对每个实验给出：配置差异、控制变量、指标、预期判别、失败后结论。不要用巨大的网格搜索掩盖主假设不成立。

## 10. 强制输出格式

请严格按以下顺序回复：

1. **可见性证书**：实际访问的 commit、文件、函数、行号；
2. **一句话总裁决**：`GO / HOLD / KILL`，只选一个；
3. **当前实现事实图**：forward、loss、梯度、坐标与成本；
4. **逐行代码问题表**：严重度 P0/P1/P2、文件:行号、问题、后果、修复、回归测试；
5. **uniform 缺陷复核**：确认修复是否完整，列出残留同类风险；
6. **实验结果合法性表**：哪些有效、哪些仅诊断、哪些必须撤回；
7. **性能低于强 uniform 的根因排序**：给概率和可证伪实验；
8. **三个互斥方案**：完整比较；
9. **唯一推荐最终模型**：达到实现规格级别；
10. **核心代码**：只给关键修改，不复制整仓库；
11. **决定性实验矩阵与运行顺序**；
12. **真实成本核算模板与 break-even 条件**；
13. **论文可发表性裁决**：当前能 claim 什么、不能 claim 什么、需要什么证据；
14. **停止条件**：出现哪些结果就应终止 DUCA，而不是继续调参。

## 11. 审查纪律

- 不要因为测试通过就称方法正确；测试只证明指定合同。
- 不要因为 detector gradient 非零就称 utility 对齐；必须要求 one-swap 或 counterfactual 证据。
- 不要把旧 55.50 当 uniform，也不要把历史 64.352/65.696 直接填入当前 matched 表。
- 不要将“score 连续”自动等同于“hard policy 连续”。
- 不要假设 max-gap 必然提高边界覆盖；检查它是否迫使策略接近 uniform。
- 不要假设 official config 相同就代表 detector 源码完全相同。
- 不要建议先跑更多昂贵实验再理解代码。
- 不要为了照顾作者语气降低批评强度。
- 所有判断必须标注为：代码事实、实验事实、合理推断、尚未验证。

最终目标不是证明当前 DUCA 一定正确，而是裁决：**这个“粗状态变化驱动的 pre-backbone 离线 TAD 去冗余”假设是否值得继续；若值得，最终模型究竟应如何实现；若不值得，哪条替代路线最有科学价值。**
