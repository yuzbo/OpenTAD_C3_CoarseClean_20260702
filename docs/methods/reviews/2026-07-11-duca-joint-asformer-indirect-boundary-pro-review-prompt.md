# DUCA Joint ASFormer / Indirect Boundary Selection Pro Review Prompt

你是一名同时具备以下身份的严格审查者：CVPR/ICCV/NeurIPS 级别资深审稿人、TAD/TAL
研究者、PyTorch/OpenTAD 工程专家、离散结构化选择与多任务优化专家。请使用最高推理强度，
不要顺着作者预设结论。你的任务不是给泛泛建议，而是实际阅读仓库代码、逐项核验实验事实、
裁决当前 DUCA 是否实现了正确科学假设，并给出一个可落地、可证伪、可部署的最终方案。

## 0. 仓库与版本

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- DUCA branch: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/gas-vt-stage23-detector-aware-20260706
- 产生正式 fixed-384 结果的 commit `70aa069b895322c2307ffbb13dfdef9fac0d1305`:
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/70aa069b895322c2307ffbb13dfdef9fac0d1305
- 当前远端分支 HEAD、最新审计 commit `a5e1774b9941312569ca645341da1abad339db61`:
  https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/a5e1774b9941312569ca645341da1abad339db61

请先给出“代码可见性证书”：确认你实际打开了仓库、分支和两个 commit，列出 commit hash、
commit title、你实际阅读的文件及关键行号。若无法访问，必须明确停止代码事实判断，不能凭
prompt 内容假装已逐行审查。

必须重点阅读：

- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/dynamic_budget.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `tools/bata/train_lowres_action_probe.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/cores/optimizer.py`
- `opentad/cores/train_engine.py`
- `configs/adatad/thumos/duca_online_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_must_dynamic_official_adatad_backend_full_train.py`
- `tools/bata/validate_duca_official_adatad_backend.py`
- `tests/test_duca_joint_training_contract.py`
- `tests/test_duca_jct_one_step_grad_proof.py`
- `tests/test_duca_structured_selection.py`
- `tests/test_duca_full_stack_cost.py`
- `docs/methods/2026-07-10-70aa069-researchclaw-duca-divergent-audit-absorption.md`
- `docs/methods/2026-07-10-88e50b1-duca-final-method-audit-review-absorption.md`

## 1. 任务与原始科学假设

这是离线 full-window TAD，不是 causal/streaming/Online TAD。`online` 只是历史类名，不能
用于论文任务定义。目标是在昂贵 VideoMAE/AdaTAD backbone 之前减少时序观测，在保护尤其
是 mAP@0.6/0.7 的同时降低 decode-to-output 总延迟、能耗和显存。

项目的核心初心不是让小模型直接完成边界定位，而是：

1. 小模型更容易可靠学习 frame/segment 级 action-vs-background 粗状态；
2. 动作状态的变化、置信度变化和不确定性峰值比 actionness top-k 更接近语义边界；
3. selector 应从这些状态变化证据间接估计帧的边界/检测效用；
4. GT boundary 只能在训练期监督 selector，不能成为推理输入；
5. 下游 TAD detector loss 应影响 selection policy，但必须证明梯度与 hard frame utility
   对齐，而不只是 nonzero；
6. 推理时只允许完整窗口 RGB、模型参数和运行时生成的中间量，不允许 GT、teacher、
   oracle、外部 JSONL、ledger 或 raw-prediction cache 决策。

请把这条科学假设与当前代码逐行对照。若当前 selector 可以通过 absolute hidden feature
直接学习 GT boundary，或 coarse probe 直接挂 start/end predictor，请指出它是否已经绕过
“粗分类驱动的间接边界定位”，并给出结构性修正。

## 2. 不可偷换的最终架构合同

请以以下合同审查，而不是沿用历史文档中可能错误的 `online` 或 direct-boundary 表述：

```text
low-resolution full-window RGB
        -> official-ASFormer coarse action-state probe
        -> p_action / logits / coarse state embedding
        -> explicit state-transition descriptors
           (delta logits, abs delta, entropy, uncertainty change, delta hidden)
        -> indirect transition-utility selector
        -> exact-K, max-gap-constrained hard original-time positions
        -> selected expensive observations
        -> official-derived AdaTAD/ActionFormer detector
```

硬约束：

- coarse ASFormer 只直接接受 binary action/background supervision；不得把它变成 mini-TAD
  start/end detector；
- `boundary-first` 表示预算优先保护由状态变化推断出的边界，不表示 coarse probe 直接回归边界；
- selector 可以使用 coarse semantic state，但必须解释如何防止 unrestricted absolute hidden
  绕过 transition mechanism；主路径至少应显式依赖 `delta hidden` 等变化量；
- GT start/end 只能是 train-only selection supervision，不能是 inference input；
- actionness 只能是粗状态校准和辅助，不得退化为 actionness top-k；
- 不允许 `asformer_lite`，若用 ASFormer 只能用官方实现代码；
- MobileNetV3 不是预设必需项。ASFormer 需要 deploy-visible spatial features，但应先判断
  当前两层 spatial stem + official ASFormer 的 joint optimization 是否已修好，再决定是否
  值得增加 MobileNet 成本；
- 主方法必须是单次训练、单 checkpoint 的联合模型；分离 coarse/selector/detector 仅可作
  归因 baseline；
- fixed-K 是主方法锚点，dynamic MUST 只有在 fixed-K 超过 matched uniform 后才可晋升；
- detector 应诚实称 official-derived components + DUCA wrapper，不能声称源码/坐标语义
  完全未修改的官方 AdaTAD。

## 3. 当前已知实现事实，请逐项核验而不是直接相信

1. standalone coarse probe 与 joint DUCA 都可以构造同一个
   `C3OfficialActionSegmentationProbe`。
2. 当前 official-ASFormer probe 是两层随机 Conv2d spatial stem、约 96 维 temporal feature、
   official `MyTransformer`，默认 `num_layers=2`；70aa 主实验未加载 coarse checkpoint。
3. standalone probe 每步直接优化未缩放 binary BCE，并可按 validation metric 保存最佳
   checkpoint。
4. joint selector 默认 actionness 权重约 0.05，同时叠加 boundary、coverage、budget、
   detector surrogate 等目标。
5. current hard structured policy 与 detector forward 消费 hard selected positions；backward
   通过 ST/soft-context/soft-resample 近似路径传播。
6. hard Viterbi/MAP 是否对 logits `detach`，soft backward 是否属于同一个 exact-K/max-gap
   可行集合，必须以代码给出明确结论。
7. detector 在 irregular selected-rank axis 上运行，GT/prediction 做 true-time remap；但
   Conv/FPN stride、regression range、center sampling 是否仍假设等物理时间，请审查。
8. `a5e1774` 已加入 full-stack profiler 和更诚实的 AdaTAD contract，但只有 random-init
   smoke，没有 trained-checkpoint 正式成本矩阵。

## 4. 当前实验现象

正式 DUCA fixed-384：

- commit: `70aa069`
- Slurm Job: `1154971`
- 60 epoch 正常完成，无 OOM、NaN 或训练崩溃；曲线持续上升，epoch 59 最佳；
- Avg-mAP: **58.39**
- mAP@0.3/0.4/0.5/0.6/0.7:
  **76.26 / 71.06 / 61.20 / 48.90 / 34.53**
- 每 epoch 约 99 optimizer steps，总计约 **5940 steps**；
- requested K=384，但 120 个 budget summary 中 effective K 均值约 **360.55**、最低
  **214**，35 个 summary 小于 384。

历史、协议尚未完全匹配的锚点：

- dense AdaTAD: Avg-mAP 约 **68.29**；
- separated lattice/move50 best: Avg-mAP **63.18**，IoU-wise
  **78.00 / 73.28 / 66.52 / 55.89 / 42.19**；
- PAction separated best: **61.02**，final **59.10**；
- 用户记录的 uniform-384 约 **65**，但目前没有定位到与 70aa 同 commit、同 loader、
  同 effective-K 的成功 provenance，不能把它伪装成 matched baseline；
- 历史 separated pipeline 约 218 steps/epoch、60 epoch 共 **13080 steps**。

关键现象：

- DUCA 相对 lattice 在 tIoU 0.3 只低约 1.74，但在 tIoU 0.7 低约 7.66，差距随定位
  严格度增大，疑似边界/时间几何问题；
- 按最终权重还原后，joint actionness raw BCE 约 `0.034/0.05 ≈ 0.68`，接近随机
  `log(2)=0.693`；endpoint/context/boundary-proxy raw distribution losses 约 6.38，接近
  `log(T)`，selector/coarse 可能仍近 chance-level；
- 因此 weighted loss 下降可能主要来自 schedule 权重下降，不能直接证明 selector 学会；
- 分离 ASFormer 的优势可能来自纯 BCE、稳定输入、更多 optimizer steps 和 best-val
  checkpoint，而不是 MobileNet 缺失；
- 当前结果既不能证明 joint training 天生更差，也不能证明 learned selection 优于 uniform；
- 当前成本 claim 仍是 unproven：768->384 只提供约 50% heavy-backbone 理论空间，dense
  probe、decode、H2D、selector 和 wrapper overhead 都必须计费。

请检查这些数字的可比性；把“事实”“合理推断”“尚未验证假设”分栏，不得把历史锚点
写成同协议结论。

## 5. 必须回答的核心问题

### A. 当前代码到底实现了什么

逐文件重建真实 tensor/data/gradient flow。回答：

1. coarse probe 的实际输入、shape、spatial stem、official-ASFormer 层数和输出是什么？
2. selector 到底看到 `p_action`、logits、absolute hidden、delta hidden 中哪些量？
3. 当前是否存在 absolute-hidden bypass 或 direct boundary predictor？
4. GT boundary 在何处进入，train/val/test 是否严格 no-leak？
5. detector loss 实际更新 detector、selector、ASFormer trunk、actionness head 中哪些参数？
6. 所有 trainable parameters 是否被 optimizer 覆盖？
7. current AdaTAD path 与官方 OpenTAD AdaTAD 在 backbone、adapter、neck/head、GT 坐标、
   selected axis、postprocessing 上有哪些具体差异？

### B. 为什么 separated ASFormer/ledger pipeline 比 joint DUCA 更强

给出按证据强度排序的 root-cause table，至少审查：

- 5940 vs 13080 optimizer exposure；
- actionness=0.05 与多任务梯度稀释；
- standalone best-checkpoint selection；
- hard selected positions 的非平稳输入分布；
- detector surrogate 梯度是否方向错误；
- ASFormer smoothing 是否把 transition peak 向动作内部或时间上偏移；
- duplicate ASFormer temporal trunk + selector encoder；
- effective-K mismatch；
- irregular selected-axis geometry；
- loader/window/data augmentation 差异。

每个原因必须给出：代码/日志证据、置信度、最小反证实验、若为真应观察到的现象。

### C. 最终结构如何保持“间接边界定位”而不退化成 direct boundary detector

请提出至少三种结构候选并强制选出一个主方案：

1. 保持 current two-layer stem + official ASFormer，只修联合优化；
2. official ASFormer 作为共享 state trunk，移除重复 selector temporal encoder；
3. 增加 MobileNetV3 或其他 spatial frontend；
4. 你认为更优但仍满足低成本、间接定位和 no-leak 的方案。

讨论 selector 是否应看到 absolute hidden。若允许，必须提出可审计机制证明它没有绕过
transition hypothesis；若不允许，说明 `delta hidden`、pairwise semantic change、uncertainty
如何保留足够语义。禁止只写“加一个 boundary head”。

### D. 如何设计真正稳定、优雅的单次联合训练

请直接给出最终训练算法，而非只说“调权重”：

- module-specific gradient routing；
- actionness head 是否只接 `L_action`；
- ASFormer trunk 如何接 `L_action/L_transition/L_detector`；
- detector gradient 是否需要 ramp、stop-grad、PCGrad/GradNorm、two-timescale optimizer；
- 如何用连续 homotopy 从稳定 feasible policy 过渡到 fully learned hard policy，同时保持
  单次训练、单 checkpoint，而不是三阶段伪装；
- scheduler 必须按 optimizer step，并匹配至少 13080 updates；
- 如何避免 coarse action calibration 被 detector gradient 破坏；
- 如何处理 class imbalance、ASFormer over-smoothing 和多个动作实例；
- checkpoint selection 应同时满足 detector mAP 与 coarse/transition health gates。

请给出清晰数学目标、参数集合、每项 loss 的梯度接收者、默认权重/调度原则及伪代码。

### E. hard/soft selection 是否科学成立

审查 current ST/soft-resample。讨论是否应改为：hard MAP forward 与 entropy-regularized
soft marginals 来自同一个 exact-K/max-gap feasible family，例如
`z_ST = z_hard + z_soft - stopgrad(z_soft)`。如果该方案在 slot ordering 或复杂度上不可行，
必须给出更可靠替代，而不是继续使用未经验证的 dense surrogate。

设计 one-swap finite-difference 审计：比较 selector/ST gradient 与真实 hard replacement
detector-loss utility 的 Spearman/Top-k agreement，并与 actionness、transition、random 对照。

### F. 时间几何是否构成不可由训练修复的上限

解释 irregular selected-rank axis 对 ActionFormer convolution、FPN stride、regression range、
center sampling 和 high-tIoU 的影响。设计 same-selected-frames 对照，区分：

- 位置质量差；
- joint optimization 非平稳；
- selected-axis geometry 错误。

若确实需要 physical-time-aware regrid，给出最小、保持 official detector head 的方案；不要
未经对照就把完整 physical-grid 工程当成既定答案。

### G. MobileNet 是否必要以及真实成本能否成立

不能用“MobileNet 很轻”作结论。请比较 current stem、MobileNet-only、MobileNet+shared
official-ASFormer 三种设计。使用近似 break-even：

`T*C_probe + K*C_heavy + C_selector < T*C_heavy`

给出需要实测的 decode、preprocess、H2D、probe、selector、gather、heavy backbone、head、
NMS、p50/p95、energy 和 peak memory。判断何种条件下 fixed-384 仍能形成有效
accuracy-latency Pareto；若 probe 吞掉节省，应明确否决。

### H. 实验与发表性

给出最小但足够严厉的 paper experiment matrix，必须包含：

- 同 commit/loader/optimizer-step/effective-K 的 dense、exact-uniform、periodic、random；
- current joint、no-detector-gradient、protected-gradient、frozen standalone probe、
  standalone-init-then-joint（后两者只作诊断，不得冒充单次主方法）；
- raw coarse AUROC/AUPRC/F1/ECE；
- transition peak recall、selected-to-boundary distance、short-action recall、max/p95 hole；
- one-swap gradient alignment；
- same-selected-frames geometry control；
- trained-checkpoint full-stack cost；
- fixed-384 通过后才允许 fixed-256/128 与 dynamic MUST；
- 至少三 seed；
- 第二 detector，或诚实收缩为 AdaTAD-specific。

最后以 CVPR/ICCV 审稿标准裁决：当前组合创新是否足够；哪些结果出现时可称主方法成功；
哪些结果出现时必须降级、删除 detector-gradient bridge、回到 simpler residual selection，
或终止 DUCA 路线。

## 6. 强制输出格式

请严格按以下顺序输出：

1. **可见性证书**：commit 和实际阅读文件/行号。
2. **总裁决**：PASS / HOLD / REJECT，最多 300 字，不许含糊。
3. **当前真实实现图**：不是作者想象图，标出 tensor shape、detach、GT 和梯度路径。
4. **原始假设一致性审计**：逐条判断是否违背“粗分类间接边界定位”。
5. **根因排序表**：证据、置信度、反证实验、预期观察。
6. **三种候选架构**：优缺点、成本、风险；强制选择一个最终方案。
7. **最终架构与数学目标**：模块职责、loss、gradient routing、hard/soft policy。
8. **关键实现代码**：给出可落入本仓库的 PyTorch 核心代码/patch 级伪代码，列出要修改
   的准确文件、类、函数；禁止只写抽象概念。
9. **测试合同**：optimizer coverage、no-leak、gradient recipients、hard/soft feasibility、
   one-swap、train/test isomorphism、cost accounting。
10. **正式实验矩阵**：主表/消融/诊断/附录分开，给出优先级、变量控制和停止规则。
11. **结果到 claim 矩阵**：不同可能结果分别允许写什么、不允许写什么。
12. **最终 kill criteria**：何时继续、何时降级、何时停止 DUCA。

## 7. 审查纪律

- 必须区分 repository fact、experiment fact、inference、proposal。
- 不得编造缺失日志、uniform 65 provenance 或 a5e full-train 结果。
- 不得把 smoke/precheck/nonzero gradient 当成科学有效性。
- 不得把同 epoch 不同 optimizer steps 当作公平比较。
- 不得因为 GT boundary 用于 train supervision 就误判为泄漏；真正禁止的是它参与
  val/test/inference decision。
- 不得建议 `asformer_lite`。
- 不得默认 X3D/SlowFast dense inference 是低成本主方案。
- 不得把三阶段训练重新包装成“联合模型”。
- 不得为了显得复杂而继续增加 selector head/loss；优先消除不一致、错误梯度和重复模块。
- 可以推翻我们的前提。如果“粗状态变化足以支持高 tIoU 选帧”本身不成立，请明确说明，
  给出最小证伪实验与更好的替代科学问题。
- 不要给逐日项目管理式计划；直接给最终目标结构、训练算法、关键代码和决定性实验。

请严厉、具体、可执行。我们需要的是能够阻止错误路线继续消耗 GPU 的技术裁决，而不是
礼貌性建议。
