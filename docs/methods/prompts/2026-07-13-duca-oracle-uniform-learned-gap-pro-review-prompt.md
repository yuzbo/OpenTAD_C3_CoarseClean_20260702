# DUCA Oracle-Uniform-Learned Gap: Strict Pro Review Prompt

你现在不是来鼓励项目，也不是来给一份泛泛的实验清单。你是负责决定该方法是否值得继续作为 CVPR/ICCV/NeurIPS 主方法的高级审稿人、时序动作定位研究者和 PyTorch/OpenTAD 工程审计者。

请使用 GitHub 连接器直接读取下面固定提交中的真实代码。不得只根据本 prompt 的摘要作答，不得把类名、config 名称、smoke test 或非零梯度测试当成行为事实。

## 1. 固定审计对象

- 仓库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 分支：`codex/duca-transition-only-20260711`
- 固定代码与诊断提交：`1fc7037358e1141f7555ad87d1edd9128ce2e6a5`
- 固定提交入口：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/1fc7037358e1141f7555ad87d1edd9128ce2e6a5`

必须至少逐行阅读：

1. 仓库规则与方法口径
   - `AGENTS.md`
   - `RTK.md`
   - `docs/methods/duca_transition_only_contract.md`
2. 主模型与结构化选择
   - `opentad/models/selectors/duca_online_frame_selector.py`
   - `opentad/models/duca/acquisition.py`
   - `opentad/models/duca/transition_only.py`
   - `opentad/models/duca/structured_selection.py`
   - `opentad/models/detectors/actionformer.py`
   - `opentad/models/detectors/single_stage.py`
   - `opentad/models/dense_heads/actionformer_head.py`
3. 正式配置与训练合同
   - `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
   - `configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py`
   - `configs/adatad/thumos/duca_exact_uniform_fixed384_official_adatad_backend_full_train.py`
   - `tools/bata/run_duca_transition_only_formal_full_model_gate.py`
   - `tools/bata/validate_duca_transition_only_p0_variant.py`
4. 新增质量分析、测试和结果报告
   - `tools/bata/export_duca_selection_quality.py`
   - `tools/bata/analyze_duca_selection_quality.py`
   - `tests/test_duca_selection_quality_analysis.py`
   - `docs/methods/2026-07-13-duca-selection-quality-diagnostic.md`
5. 既有严格审计背景
   - `docs/methods/prompts/2026-07-12-duca-transition-only-0ea4e15-pro-audit-prompt.md`
   - `docs/methods/2026-07-10-duca-official-adatad-structural-audit.md`

请在回答开头给出“可见性证书”：列出实际读取的 commit、文件和关键行号。看不到文件就明确写不可见，不允许假装读过。

## 2. 任务口径，禁止误解

本研究是完整视频可用条件下的离线 TAD pre-backbone 时序去冗余，不是 Online TAD。代码中的历史 `online` 命名不能改变任务定义。

方法初心是：

1. 低成本粗分类器只学习 action/background 状态，不直接承担精确边界回归。
2. selector 观察 deploy-visible 的 `p_action`、不确定性、状态变化和粗分类隐藏特征，通过状态转变间接定位语义边界。
3. GT action segments/boundaries 只在训练时提供监督，不能作为推理输入。
4. fixed-K/max-gap 选择器在 original-time 中产生 detector 实际消费的位置。
5. 后端尽可能保持官方 AdaTAD/ActionFormerHead，不允许用简化 detector 掩盖选择问题。
6. detector 监督若参与 selector 学习，必须明确梯度所有权、软硬一致性和不可见信息边界。

审查时必须判断当前实现是否仍然遵守这个初心，还是已经退化为 noisy actionness top-k、手工 gap repair 或 selected-axis 上的伪联合训练。

## 3. 当前已知结果，先审计协议再解释数字

用户提出的核心现象是：

> 为什么 oracle 大约能达到 78 Avg-mAP，均匀采样大约 65，而学习采样只有大约 63？

不得默认这三个数字同协议。现有可审计信息是：

| 结果锚点 | 数值 | 当前证据等级 |
|---|---:|---|
| 用户报告的 oracle | 约 78 Avg-mAP | 本轮尚未定位同协议原始 artifact；必须追查，否则只能标记 unverified upper-bound anchor |
| 历史 grid-aware exact-uniform，Job `1150842` | 65.696 | 有日志来源，但不是当前 same-commit/same-training protocol |
| 历史 native-stride exact-uniform，Job `1150701` | 64.352 | 有日志来源，但 detector geometry 与当前矩阵不完全相同 |
| learned transition beta=0，Job `1159416` | 64.34 | 旧 `8bfc0e5` homotopy 起点失效，只是 learned-policy diagnostic |
| learned transition beta=0.25，Job `1159417` | 63.55 | 同样是失效协议；比 beta=0 低 0.79，不能证明 detector bridge 有效 |
| 历史分离训练 lattice | 63.18 | 跨协议历史参考，不是 matched joint-vs-separated 因果实验 |

旧提交 `8bfc0e5` 的 alpha=0 uniform reference 在 `T=768,K=384` 时产生全相同 logits，Viterbi tie-break 路径并非 exact uniform。因此 Job `1159414` 的 55.67 绝对不能作为均匀采样基线，Jobs `1159416/1159417` 也没有执行预期的连续 uniform-to-learned homotopy。`0ea4e15` 修复了该问题，但尚没有替代 full train 结果。

第一项任务不是解释，而是审计：

1. 在仓库、报告和引用路径中寻找 oracle≈78 的原始定义与来源。
2. 查清它究竟是 Avg-mAP、mAP@0.3、dense detector、GT-boundary selector、teacher utility、GT actionness、post-hoc best-of-candidates，还是其他指标。
3. 查清 oracle 是否使用 validation/test GT、dense detector loss、teacher features 或 raw predictions作决策。
4. 核对 oracle、uniform、learned 的数据 split、K/effective-K、输入分辨率、detector、坐标轴、训练步数、LR schedule、checkpoint selection 和评估脚本。
5. 如果找不到原始 artifact，必须写“oracle 78 未验证”，禁止替它编造机制或把差距量化成严格结论。

只有协议完全匹配时，才允许使用：

`P_oracle - P_learned = (P_oracle - P_uniform) + (P_uniform - P_learned)`。

否则必须把三项分别称为 privileged-information upper-bound、historical regular-grid reference 和 invalidated learned-policy diagnostic。

## 4. 新完成的选帧质量诊断

真实 EMA epoch-89 checkpoint、211 个验证视频、487 个窗口的 selector-only 诊断如下。GT 只在选择完成后用于评估，按原视频做 2,000 次 cluster bootstrap。

### 4.1 粗分类质量

| 指标 | 数值 |
|---|---:|
| pooled AUROC | 0.6214 |
| pooled AUPRC | 0.4111 |
| action prevalence | 0.3250 |
| AUPRC lift | 1.265x |
| Brier / ECE | 0.2401 / 0.1710 |
| balanced accuracy / F1@0.5 | 0.5866 / 0.4930 |

动作帧平均/中位 `p_action=0.5268/0.5417`，背景帧为 `0.4811/0.4940`。阈值 0.5 时模型预测 53.62% 帧为动作，而真实动作比例为 32.50%。历史但协议不匹配的 standalone official-ASFormer 在 epoch 90 达到 AP 0.4569、AUROC 0.6494。

### 4.2 间接 transition 排序

| 边界半径 | learned AP/AUROC | raw `abs(delta p_action)` AP/AUROC |
|---|---:|---:|
| r0 | 0.0075 / 0.5775 | 0.0082 / 0.6079 |
| r1 | 0.0619 / 0.5841 | 0.0654 / 0.6069 |
| r2 | 0.1107 / 0.5785 | 0.1178 / 0.6070 |
| r4 | 0.1916 / 0.5711 | 0.2019 / 0.5982 |
| r8 | 0.3038 / 0.5542 | 0.3145 / 0.5770 |

学习型 transition scorer 在所有半径均弱于最简单的原始动作性变化。

### 4.3 最终 selected positions

| Policy | recall r1 | both endpoints r1 | mean endpoint distance | mean max hole | action enrichment |
|---|---:|---:|---:|---:|---:|
| learned structured | 0.8437 | 0.7108 | 0.6755 | 11.86 | 1.0436x |
| exact uniform | 0.9991 | 0.9982 | 0.4800 | 1.83 | 0.9988x |
| stratified random | 0.9082 | 0.8280 | 0.5481 | 1.86 | 0.9990x |
| learned utility top-k diagnostic | 0.8438 | 0.7110 | 0.6814 | 15.33 | 1.0445x |
| raw-change top-k diagnostic | 0.8377 | 0.7059 | 0.8559 | 28.91 | 0.9953x |

Learned 相对 exact uniform：

- exact-r0 recall +1.53 个百分点，95% CI `[+0.50,+2.54]`；
- r1 recall -15.55 个百分点，CI `[-17.04,-13.95]`；
- endpoint distance 恶化 +0.195 帧，CI `[+0.159,+0.233]`；
- 487 个窗口中 113 个更好、66 个持平、308 个更差；
- learned decoder 与 unconstrained learned-utility top-k 平均重合 99.80%，264/487 窗口完全相同；
- hard max-gap 将最坏空洞从 60 降到 15，但没有实质改变边界指标；
- coarse AUROC 与 selection gain 的窗口级 Pearson 相关仅 0.182；即使 top coarse-AUROC quartile，平均 gain 仍为 -0.088 帧。

这些数字来自旧失效 homotopy checkpoint，因此只能诊断该 checkpoint，不能直接否定修正方法。但它们足以证明“当前已训练 selector 选得好”这一说法不成立。

## 5. 必须回答的核心科学问题

### Q1. Oracle 为什么可能远高于 uniform？

请区分至少四种完全不同的 oracle：

1. GT boundary/actionness oracle；
2. dense detector-loss/teacher-utility oracle；
3. best-of-candidate counterfactual oracle；
4. 同可见输入下可学习 Bayes oracle。

判断约 78 属于哪一种。若使用 GT 或 dense downstream output，它只证明“有信息时可以选好”，不证明当前低成本粗分类输入能够预测该选择。请判断 oracle gap 是真正可学习 headroom，还是 privileged-information gap。

### Q2. Uniform 为什么异常强？

必须结合 `K≈T/2`、ActionFormer/AdaTAD 的时间几何和卷积感受野解释，而不是只说“覆盖更均匀”。逐项检查：

1. 每 1-2 个 dense positions 一个样本是否几乎保证任意小数 GT endpoint 在 ±1 内被覆盖；
2. official projection/neck/head 是否默认 selected-axis 等间距，并把 irregular positions 当作均匀时间；
3. GT target assignment、offset regression、NMS 和 inverse map 是否真正消费 original-time timestamps；
4. irregular sampling 是否破坏局部运动、上下文连续性、短动作和边界两侧证据；
5. uniform 是否同时充当 coverage prior、anti-alias filter 和 detector geometry prior；
6. 在 50% budget 下，自由 top-k 是否本质上只有很小的可获益空间，却有很大的造洞风险。

### Q3. Learned 为什么落后 uniform？

请对下面候选根因逐项给出代码证据、概率排序和可证伪实验，不得全部列为“可能”：

1. coarse probe 分离度和校准不足；
2. transition head 没有超过 raw `abs(delta p_action)`，甚至抹掉了原有状态变化信号；
3. boundary labels 极稀疏、窗口裁剪和小数坐标造成监督噪声；
4. coarse BCE、transition/boundary coverage、soft max-gap 和 detector loss 的尺度或梯度冲突；
5. detector gradient 只更新 scorer、不更新 coarse representation，或 soft-to-hard bridge 的梯度方向与 hard one-swap utility 不一致；
6. learned utility 过度集中在少数高分区，获得少量 exact hits但丢失大量边界和上下文；
7. max-gap=15 在 K=384/T=768 下过松，合法不等于高覆盖；
8. structured decoder 几乎等于 utility top-k，约束没有形成有效 inductive bias；
9. selected-axis 与 physical/original-time 几何不一致；
10. invalid uniform homotopy 使训练起点和课程完全错误；
11. standalone 与 joint 的 optimizer steps、LR schedule、checkpoint criterion 和数据暴露不匹配；
12. official ASFormer hidden feature是否真的被 scorer 有效利用，还是最终主要依赖 noisy p_action 曲线；
13. selector 优化的是 boundary proxy，而 oracle 优化的是 detector utility，目标本身不一致；
14. oracle 78 可能根本不是同一 metric/protocol。

### Q4. 当前梯度桥到底有没有学习正确的 hard decision utility？

不能用“参数有非零梯度”作答案。请检查：

1. hard selected positions 与 soft surrogate 的 support 是否一致；
2. 对一个可行 one-swap `(selected s -> unselected t)`，soft gradient 的符号是否与真实 detector loss 差 `L(S-s+t)-L(S)` 一致；
3. beta=0.25 比 beta=0 低 0.79 Avg-mAP 是否说明 bridge 有害、无效，还是旧 homotopy confound；
4. detector loss 是否通过 raw-pixel/feature resampling改变了 detector input分布，却没有给 selector可解释的局部 credit；
5. 当前方法应继续直接 detector backprop，还是改为 train-only counterfactual utility distillation；若改，必须承认论文机制已经改变。

## 6. 必须进行的逐行代码审查

请按严重程度列出 P0/P1/P2 findings，每条必须有：

- `file:line`；
- 当前代码实际行为；
- 为什么会导致 oracle/uniform/learned gap；
- 是确定 bug、协议混淆、目标错配还是待验证假设；
- 最小修复；
- 能直接触发失败的测试。

至少审查：

1. exact uniform helper 是否在所有 legacy/direct/transition routes 统一；
2. `structured_selection.py` 的 MAP、logZ、marginal、occupancy 和数值稳定性；
3. hard max-gap 的可行集定义是否与 soft hole loss同构；
4. `p_action`、ASFormer hidden、delta、uncertainty 到 utility score 的真实数据流；
5. detector input 的时间坐标、mask、GT remap 和 inference inverse map；
6. ActionFormer optimizer groups 是否完整覆盖 selector参数且没有重复/遗漏；
7. leaf loss 聚合是否每项只加一次；
8. coarse、scorer、bridge 的梯度所有权；
9. train/val/test 是否完全无 GT/teacher/cache决策泄漏；
10. epoch/step schedule 是否真正匹配历史 uniform与 standalone ASFormer；
11. exporter/analysis 是否存在 GT来源、score normalization、窗口重复统计或指标定义问题；
12. 当前所谓 official AdaTAD backend 与官方结构的确切差异。

## 7. 不要只修补，给出最终模型裁决

在读完代码后，从下面三类路线中裁决一个主路线，也可以提出更好的第四路线。不要同时推荐所有路线。

### 路线 A：coverage-preserving deformable sampling

将 exact-uniform/stratified anchors视为可行集参数化，每个 temporal cell 必选一个点，selector只学习 cell 内单调 offset 或少量跨 cell交换。这样不是 post-hoc uniform fill，而是从定义上保证 coverage，再用 transition/detector utility决定偏移。

### 路线 B：monotonic structured transport

用带 coverage lower bound 的 monotonic optimal transport/soft-DP直接学习 K 个有序位置，使 soft distribution与 hard MAP来自同一可行集，避免 global top-k + repair。

### 路线 C：counterfactual detector-utility distillation

训练时用真实 hard one-swap detector-loss差构造可行位置效用，selector从 deploy-visible coarse state证据预测该效用；推理无 teacher、GT或 dense detector。必须明确这是 utility distillation，不是 detector loss直接穿过 hard选择。

最终设计必须回答：

1. coarse 模块是否保留 official ASFormer，是否需要预训练 spatial stem；
2. selector 能看到哪些粗分类隐藏特征，哪些绝对特征会违背“间接状态转变定位”初心；
3. fixed-384 policy 的数学可行集；
4. 如何在一个训练运行中协调 coarse state、transition utility、coverage和 detector任务，而不是重新变成三个独立 checkpoint；
5. 如何避免 detector loss摧毁粗分类状态表示；
6. 如何保证 train/inference同构；
7. 后端如何保持官方 AdaTAD/ActionFormerHead；
8. 为什么该方案理论上应保留 uniform 65，并向 oracle headroom逼近。

## 8. 要求给出核心实现，不接受伪代码口号

请直接给出可以落地到本仓库的关键 patch，至少覆盖：

1. 最终 selector policy/module API；
2. coverage-preserving或 monotonic decoder核心；
3. coarse-to-transition特征构造；
4. loss定义和权重归一化；
5. detector utility或梯度路由；
6. ActionFormer/AdaTAD integration；
7. optimizer groups；
8. one-step full-model gradient test；
9. hard one-swap gradient/utility alignment test；
10. exact-uniform identity、max-gap、timestamp和 no-leak tests。

优先复用仓库已有 official ASFormer、structured DP、OpenTAD registry、ActionFormerHead和 dataset/evaluator，不要再造一个简化 detector、ASFormer-lite或假模型。

## 9. 最小决定性实验，不要给无限矩阵

给出不超过 8 个、按 kill-gate排序的实验。每个实验必须写：

- 唯一要验证的因果假设；
- 固定项与变量；
- metric及统计单位；
- 通过阈值；
- 失败后停止或转向什么。

至少覆盖：

1. same-commit exact-uniform vs corrected learned fixed-384；
2. raw delta vs learned transition scorer；
3. coverage-preserving policy vs global top-k；
4. hard one-swap detector utility alignment；
5. matched standalone-vs-joint coarse quality；
6. selected-axis vs original-time-aware detector geometry；
7. oracle 78 的同协议复现或证伪；
8. trained-checkpoint full-stack accuracy-cost结果。

在 fixed-384 未超过 same-commit uniform 或没有形成更优 accuracy-cost Pareto 前，不允许开放 dynamic MUST、多 detector泛化、X3D/SlowFast或大规模消融。

## 10. 强制输出格式

请严格按以下顺序输出：

1. **可见性证书**：commit、实际读取文件、不可见项。
2. **总裁决**：`CONTINUE_CURRENT / REDESIGN / PIVOT / KILL` 四选一。
3. **三数字协议表**：oracle、uniform、learned每个数字的真实来源、可比性和证据等级。
4. **差距分解**：只有同协议才能计算；否则明确哪些差值无效。
5. **P0/P1/P2代码 findings**：带精确 `file:line`。
6. **根因排序**：概率、支持证据、反证、最小证伪实验。
7. **为什么 uniform 强、learned 弱的机制解释**：必须联系 K/T、边界覆盖、时间几何、优化和 detector inductive bias。
8. **唯一推荐的最终架构**：模块、张量形状、前向数据流、反向梯度流、推理合同。
9. **数学目标**：完整 loss、可行集、soft/hard关系和必要约束。
10. **核心代码 patch**：可直接映射到本仓库文件。
11. **不超过 8 个决定性实验与 kill criteria**。
12. **论文口径**：现在能说什么、不能说什么；若结果不翻转，DUCA应降级为什么。

最后必须用一句非常直接的话回答：

> 在现有证据下，oracle≈78、uniform≈65、learned≈63 最可能说明的是“可学习方法尚未利用 oracle headroom”，还是“oracle headroom本身不可由 deploy-visible输入学习”，抑或“这三个数字根本不可比较”？

不允许回答“都有可能”。必须给出当前最可信裁决，并说明哪一个最小实验可以推翻它。
