# PhysTime-TAD 最终架构 Pro 严厉审核与重设计 Prompt

请把下面整段原样交给具备联网浏览 GitHub 能力的 GPT-5 Pro / GPT-5.5 Pro。不要只粘贴摘要；必须保留仓库、分支、commit 和文件清单。

---

你现在同时扮演以下角色：

1. CVPR/ICCV/NeurIPS 级别的严厉领域主席或高级审稿人；
2. 熟悉 OpenTAD、AdaTAD、ActionFormer、VideoMAE、稀疏/不规则时间序列的 PyTorch 研究工程师；
3. 负责发现实验归因错误、训练/推理不同构、隐藏 GT 泄漏、评估口径错误和伪公平对照的实验完整性审计员；
4. 必须提出可真正实现、可公平验证且具有论文新颖性的最终方法设计者。

你的任务不是顺着已有诊断说“合理”，也不是给出泛泛调参建议。你必须先独立逐文件审计，再决定现有判断是否成立；如果现有方向根本不值得修，请明确建议 PIVOT，并说明证据。

## 0. 唯一代码与证据范围

GitHub 仓库：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

当前审核分支：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/phystime-performance-diagnosis-20260712`

正式训练模型代码锚点：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/3ac93a12c299012db64513567d5bdedf0c6d5f71`

性能诊断、预测分解、独立审计与 Research Wiki 锚点：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d900c7ce12081de3b7932fab5af8cabe4278abbd`

必须区分：

- `3ac93a1` 是正式三头训练所使用的最终稳定模型/训练实现；
- `d900c7c` 增加诊断工具、结果登记和研究裁决，没有改变三头正式训练行为；
- 不得引用其他 DUCA、X3D、SlowFast、ChronoTransport 或旧 PhysTime commit 的 mAP 来替代本次 matched run；
- 如果你无法打开仓库或某个文件，必须明确写“不可验证”，不得凭文件名猜实现。

## 1. 开始审核前必须完整阅读

### 研究目标与禁止回退

- `AGENTS.md`
- `RTK.md`
- `research-wiki/query_pack.md`
- `research-wiki/anti_repetition.md`
- `research-wiki/current_direction.md`
- `research-wiki/decision_register.md`
- `research-wiki/experiments/phystime-adatad-k384.md`
- `research-wiki/experiments/phystime-performance-drop-diagnosis.md`

### 唯一结果与诊断证据

- `docs/evaluation/results.md`
- `docs/evaluation/phystime-performance-drop-diagnosis.md`
- `docs/evaluation/EXPERIMENT_AUDIT.md`
- `docs/evaluation/EXPERIMENT_AUDIT.json`

实验数字只以 `docs/evaluation/results.md` 为准。请直接读取其中的 raw mAP、参数量、候选数、短动作召回、attention 行为和 artifact SHA256，不要从本 prompt 转抄数字。

### 三头正式配置与数据合同

- `configs/adatad/thumos/selected_axis_adatad_sparse_k384.py`
- `configs/adatad/thumos/physical_grid_adatad_sparse_k384.py`
- `configs/adatad/thumos/phystime_adatad_sparse_k384.py`
- `opentad/datasets/transforms/phystime_raw.py`
- `opentad/datasets/transforms/phystime.py`
- `tools/bata/validate_phystime_adatad_track.py`
- `tools/bata/run_phystime_adatad_real_gate.py`

### 核心模型与正确对照

- `opentad/models/detectors/phystime_tad.py`
- `opentad/models/projections/phystime_projection.py`
- `opentad/models/dense_heads/phystime_head.py`
- `opentad/models/utils/phystime_geometry.py`
- `opentad/models/projections/actionformer_proj.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- `opentad/cores/train_engine.py`

### 诊断实现与回归测试

- `tools/bata/analyze_phystime_performance_drop.py`
- `tools/bata/analyze_phystime_prediction_diagnostics.py`
- `tools/bata/run_phystime_attention_diagnostic.py`
- `tests/test_phystime_performance_diagnostics.py`
- `tests/test_phystime_prediction_diagnostics.py`
- `tests/test_phystime_measure_attention.py`
- `tests/test_phystime_head.py`
- `tests/test_phystime_adatad_configs.py`
- `tests/test_phystime_adatad_one_step.py`
- `tests/test_phystime_adatad_gate_contract.py`

## 2. 不可误解的研究目标

这是**完全离线的 TAD**，不是 Online TAD、streaming TAD 或 causal prefix detection。

长期目标是一个独立检测器：输入任意数量、任意间隔、可能存在连续空洞的 raw-video observations，以及真实时间戳/支持区间，直接在物理时间轴上完成动作分类和边界定位。

当前 Phase-1 使用：

- THUMOS14 raw RGB；
- 逻辑时间网格 T=768；
- 相同、确定性、无学习、无 GT 的不规则 K=384 采样；
- `DecordDecode` 和 VideoMAE-S 只消费选中帧；
- 三头共享 selected indices、backbone、预训练、增强、schedule、seed、NMS 和 evaluator；
- selected-axis ActionFormer、physical-grid ActionFormer、PhysTime 三种系统进行比较。

这不是 selector/plugin 论文，也不允许把 actionness、learned selector、teacher、oracle、ledger、动态 K 或预提取 I3D/X3D 特征重新塞回主方法。

秒坐标必须继续用于几何、GT assignment、proposal decode、NMS 和 evaluation；允许从秒导出原视频帧号，但禁止把 GT/预测压回 selected-rank。表征内部可以使用窗口归一化坐标，但不得因此抹掉物理持续时间和 FPS 语义。

## 3. 已观察现象，但你必须独立复核

请不要默认以下判断正确。逐文件核验后，对每条给出 `成立 / 部分成立 / 不成立 / 证据不足`：

1. 最终性能下降不是 NaN、OOM、错误 checkpoint、重复秒转换、缺失测试窗口或官方 evaluator 造成的。
2. 当前所谓“三头 head isolation”不公平：PhysTime 同时更换了坐标、projection、跨 query 时序上下文和可训练容量。
3. `PhysicalQueryEmbedding` 中未归一化 `centers_sec`/`widths_sec` 导致绝对秒数主导 content query。
4. coarse-level support attention 虽覆盖许多 observations，effective observation count 却坍缩；content logits 压过 relative-time logits。
5. PhysTime query pyramid 的候选密度显著低于 ActionFormer 两个对照，尤其损害短动作和高 tIoU。
6. PhysTime 的单一 `min_index` target assignment 与 ActionFormer 的 `filter_similar_gt` 多标签行为不同构。
7. endpoint loss 可能改善“已经命中后的边界精度”，但无法补偿候选覆盖和排序问题。
8. physical-grid 的下降主要来自“selected-rank 特征邻接关系”和“physical coordinate assignment”之间的几何错配，而不只是候选不足。

对每个假设必须给出：

- 直接代码证据，精确到 `file:line`；
- 与 `docs/evaluation/results.md` 中现象的对应关系；
- 置信度；
- 一个只改变单一变量的证伪实验；
- 如果该假设不成立，最可能的替代根因。

## 4. 严厉逐文件代码审查

请逐行检查上述核心文件，至少覆盖以下问题：

### 数据与坐标

- K384 是否真的在 decode/backbone 前生效，还是先 dense decode 再丢帧；
- train/test 是否使用完全相同的时间坐标定义；
- crop、offset、FPS、snippet stride、duration 和原帧号是否存在 off-by-one 或单位混用；
- support cells 是否偷偷用 Voronoi 扩张填满真实缺失区；
- padding mask、query mask 和真实观测 mask 是否同构；
- validation/test 是否有任何 GT、teacher、oracle 或 cache 参与推理决策。

### 模型结构

- PhysTime 是否真的缺少与 `Conv1DTransformerProj` 等价的跨时间/跨 query 上下文；
- 各层 query 数、spacing、regression range 和 receptive field 是否与 K384、视频长度和短动作尺度匹配；
- query/key/content/relative logits 的尺度是否稳定、可辨识且具有时间平移/缩放合理性；
- support-overlap mass、softmax 和 dropout 是否保持测度语义；
- 是否需要保留显式 mass average 作为不会被 content logits 覆盖的残差；
- endpoint branch 与 regression tower 是否发生梯度竞争或重复监督；
- 参数量差异究竟是公平效率优势，还是导致主实验无法归因的容量混杂。

### 监督与优化

- classification、regression、endpoint、normalizer 和正样本 assignment 是否与 ActionFormer 公平同构；
- endpoint target 是否在多尺度重复计数；
- short-action GT 是否因 spacing/range/center sampling 被系统性稀释；
- loss 的单位和权重是否会随视频 duration、FPS、query 数变化；
- VideoMAE adapter、projection、head 是否都在 optimizer 中并收到预期梯度；
- AMP FP32 islands 是否只解决数值问题，还是无意改变梯度尺度；
- train/inference 是否真正同构。

### 诊断与评估

- `analyze_phystime_*` 是否使用真实 dataset GT、是否存在自归一化、top-k 定义错误或把“所有 2000 predictions recall”误当实际 mAP；
- matched-boundary MAE 是否存在条件选择偏差，应如何正确解释；
- 当前结果能支持哪些主张、不能支持哪些主张。

所有问题按 `P0 阻断 / P1 高风险 / P2 次要 / 非问题` 排序。不要为了显得全面而罗列没有行为后果的风格问题。

## 5. 必须讨论至少三种下一版架构

请提出至少三条真正不同、但都遵守当前离线 raw-video/K384/无 selector 合同的路线，并进行审稿风险、创新性、可实现性和归因清晰度比较。至少包括：

### 候选 A：Capacity-matched Physical-Time ActionFormer

保留 ActionFormer 同等级 projection 深度、参数量和跨时间上下文，只把位置编码、attention bias、point generator、assignment、decode 改成物理时间感知。讨论它能否作为最干净的科学主方法，而不只是 control。

### 候选 B：Mass-Residual PhysTime

使用显式 support-mass pooling 作为保底路径，再叠加归一化 content correction，例如讨论但不要盲从：

```text
z_mass(q) = sum_i m_qi v_i / sum_i m_qi
logit_qi = alpha_l * cosine(q_q, k_i) + beta_l * r(delta_t_qi / width_q)
z_q = z_mass(q) + gamma_l * sum_i softmax(logit_qi + log(m_qi)) v_i
```

其中 `gamma_l` 是否应零初始化、`alpha_l/beta_l` 如何约束、是否还需跨 query encoder，都要给出理由。

### 候选 C：你认为更前沿且更审稿安全的方案

可以是 continuous relative-time Transformer、set-to-interval detector、time-warp equivariant detector 或其他方案，但必须：

- 明确区别于 mTAN、TE-TAD、FrameDrop/TRC、LiquidTAD 和普通 timestamp embedding；
- 不能退化成更复杂的 interpolation/resampling；
- 不能依赖 learned selector 掩盖检测头问题；
- 给出可以落到当前 OpenTAD/AdaTAD 代码树的具体模块接口。

最终必须明确只推荐一条主路线。若推荐组合方案，说明最小不可分核心，不允许把三个 idea 全部堆叠成无法归因的大系统。

## 6. 最终推荐模型必须达到的详细程度

请直接给出论文最终模型规格，而不是“先试试看”的草案：

1. 输入张量、时间戳、support interval、mask 的形状与单位；
2. VideoMAE/AdaTAD backbone 输出到新 projection 的数据流；
3. 每一级 query/candidate 数如何由有效 K 或物理 duration 推导，并保证与 baseline 候选数公平；
4. 时间表征、relative-time bias、support mass 和跨 query encoder 的准确公式；
5. 分类、左右边界回归、可选 endpoint 的输出和 decode；
6. 训练 loss、权重/normalizer 的单位不变性，以及梯度如何回到 adapter、projection、head；
7. train/test 同构、no-leak 和原始秒坐标合同；
8. 参数量与计算量如何与 ActionFormer control 对齐，哪些差异属于方法贡献；
9. 对短动作、长空洞、不同 FPS、视频时长平移/缩放的预期行为。

必须给出关键伪代码或核心 PyTorch 代码，至少覆盖：

- time-aware projection/attention；
- candidate/query grid 构造；
- multi-label target assignment；
- loss aggregation；
- seconds decode 与 inference；
- one-step gradient/optimizer coverage test；
- candidate parity 与 attention-collapse regression test。

不要重写整个 OpenTAD；优先复用 `Conv1DTransformerProj`、`AnchorFreeHead`、现有 registry、loss、NMS 和 evaluator。对每个新增类说明为什么不能复用现有实现。

## 7. 文件级最终修改表

请给出可直接执行的 patch map：

| 文件 | 保留/修改/删除/新增 | 类或函数 | 精确修改 | 目的 | 必测行为 |
| --- | --- | --- | --- | --- | --- |

至少覆盖 configs、projection、head、geometry、detector、optimizer/train engine、tests、validator 和 launcher。指出哪些 PhysTime 1.0 代码应冻结为 baseline，哪些可以安全复用，哪些必须删除或禁止进入主方法。

## 8. 因果实验与停止条件

不要直接建议再跑一套昂贵 full matrix。先设计最小、逐因素、能证伪根因的 P0 gates：

1. 同一 ActionFormer 容量下，仅 selected-axis vs physical-time coordinate；
2. raw absolute seconds on/off；
3. candidate-count matched/unmatched；
4. mass residual on/off；
5. content logits on/off 或 bounded/unbounded；
6. single-label vs ActionFormer-equivalent multi-label assignment；
7. endpoint weight off/on，但必须排在结构修复之后。

每项说明：唯一变化、预期指标、失败时推翻什么假设、是否值得进入 full train。随后再定义正式主实验、消融、多 seed、第二数据集、不同 sampling family 和完整成本账本。

给出明确的 `GO / HOLD / STOP / PIVOT` 条件。不要使用拍脑袋的绝对阈值；优先用与 matched selected-axis/physical-grid control 的差值、置信区间、候选同构和 attention 行为合同定义。

## 9. 新颖性与论文叙事审查

必须回答：

1. “不规则采样下在物理时间上检测”本身是否足够新？若不足，真正不可替代的贡献是什么？
2. 与 mTAN 的 continuous-time attention、TE-TAD 的 actual timeline coordinate、Temporal Robustness Benchmark 的 FrameDrop/TRC、LiquidTAD 的 continuous dynamics 分别有何实质区别？
3. 推荐架构是否只是 timestamp embedding、插值后 ActionFormer 或更复杂 resampling？
4. 哪个实验能证明贡献来自 detector，而不是 sampling prior、容量或额外监督？
5. 论文应当声称新 TAD detector、robustness method、irregular-observation benchmark，还是其中严格限定的组合？
6. 最强审稿攻击是什么？在代码和实验上如何提前关闭？

请给出一句可以写进摘要的、不过度声称的最终贡献表述；如果目前无法形成，请明确说无法形成。

## 10. 禁止给出的答案

- “继续训练更久看看”；
- “多调几个 loss weight/NMS threshold”；
- “换更大 backbone”；
- “重新使用 X3D/SlowFast/ASFormer/learned selector”；
- “三阶段预训练后分别冻结”；
- “PhysTime 结果低，所以 physical time 无效”；
- “加更多模块可能会更好”，但没有公式、接口和因果实验；
- 把 smoke、one-step、gradient proof 当成 mAP；
- 把单数据集单 seed 写成泛化或 SOTA；
- 编造 GitHub 中不存在的文件、日志或实验。

## 11. 强制输出格式

请严格按以下顺序回答：

1. **仓库可见性与审查范围证书**：列出实际读到的分支、commit、关键文件；
2. **总裁决**：`ACCEPT DIRECTION / HOLD AND REBUILD / PIVOT` 三选一；
3. **P0/P1/P2 代码问题表**：每项带 `file:line`、行为后果和证据；
4. **八条现有根因假设裁决**：成立程度、置信度、证伪实验；
5. **三种候选架构对比**：创新性、风险、复杂度、归因、公平性；
6. **唯一推荐的最终模型**：完整公式、张量、模块、数据流和推理合同；
7. **训练与监督方案**：loss、梯度、normalizer、课程是否必要、train/test 同构；
8. **核心实现代码**：可以落到当前仓库的关键 PyTorch 代码；
9. **文件级 patch map 与测试合同**；
10. **最小因果 gates、正式实验矩阵和 GO/HOLD/STOP 条件**；
11. **新颖性、相关工作边界和最强审稿攻击**；
12. **最终论文主张**：一句贡献表述，以及当前证据距离 paper-ready 还缺什么。

语气必须严厉、具体、可执行。不要为了照顾作者而淡化负结果；也不要因为当前实现失败就否定尚未被公平测试的科学假设。

---
