# 1. 仓库可见性与审查范围证书

## 1.1 实际读取范围

我通过 GitHub 连接器直接读取了公开仓库：

* Repository：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* 审核分支 ref：`codex/phystime-performance-diagnosis-20260712`
* 正式训练实现锚点：`3ac93a12c299012db64513567d5bdedf0c6d5f71`
* 诊断与研究裁决锚点：`d900c7ce12081de3b7932fab5af8cabe4278abbd`

分支 ref 可以直接解析文件。对比两个提交后，没有发现 `d900c7c` 改变三头正式训练的 projection/head/detector 配置或行为；其主要新增内容是诊断脚本、结果台账、审计报告和 Wiki 裁决。`3ac93a1` 对训练路径的相关修订集中于优化器参数枚举与非有限梯度处理。

实际读取了提示中要求的全部文件组，包括：

* `AGENTS.md`、`RTK.md`
* 六个 `research-wiki/` 目标、决策和实验页面
* `docs/evaluation/results.md`、性能诊断、Markdown/JSON 审计
* 三个正式配置
* `phystime_raw.py`、`phystime.py`、`end_to_end.py`
* `phystime_tad.py`、`phystime_projection.py`、`phystime_head.py`、`phystime_geometry.py`
* `actionformer_proj.py`、`anchor_free_head.py`
* `vit_adapter.py`、`optimizer.py`、`train_engine.py`、`test_engine.py`
* 两个 validator/gate 脚本
* 三个诊断脚本
* 提示列出的全部 focused tests

仓库自身规定不提交 checkpoint、服务器日志、run root 或数据集。因此，我能够核验提交中的代码、配置、结果登记、artifact SHA256 字符串和诊断实现，但**不能独立重新计算远端 checkpoint/artifact SHA256，也不能从仓库外原始 Slurm 日志重新证明“无 OOM/无 NaN/最佳 checkpoint 正确”**。这些部分只能标记为“由提交结果台账支持，但不可独立复验”。

另有一个文档一致性问题：`AGENTS.md` 第 33 行仍写着 raw-video 1.0 尚无 mAP，而正式结果页已经登记完成结果；这是过期状态文本，不影响模型执行，但必须修正。

## 1.2 正式结果复核

`docs/evaluation/results.md` 的正式最佳 checkpoint 结果为：

| 系统                         | mAP@0.3 |  @0.4 |  @0.5 |  @0.6 |  @0.7 |   Avg-mAP |
| -------------------------- | ------: | ----: | ----: | ----: | ----: | --------: |
| selected-axis ActionFormer |   79.87 | 74.15 | 66.12 | 56.02 | 41.87 | **63.61** |
| physical-grid ActionFormer |   77.09 | 71.80 | 63.74 | 50.74 | 32.34 | **59.14** |
| PhysTime 1.0               |   72.70 | 68.38 | 60.94 | 49.06 | 34.96 | **57.21** |

因此：

* physical-grid 相对 selected-axis：Avg `−4.47`，mAP@0.7 `−9.53`。
* PhysTime 相对 selected-axis：Avg `−6.40`，mAP@0.7 `−6.91`。
* PhysTime 相对 physical-grid：Avg `−1.93`，但 mAP@0.7 `+2.62`。

这说明 PhysTime 1.0 的总体排序/覆盖更差，但不能简单概括成“边界回归全面更差”。

---

# 2. 总裁决

## **HOLD AND REBUILD**

不是 `ACCEPT DIRECTION`，因为当前 PhysTime 1.0 存在至少五个阻断科学归因的问题：

1. backbone 原生 192 个 tubelet 时序特征被插值为 384，再被强行绑定到 384 个 raw-frame support，feature–time provenance 没有成立；
2. “三头 head isolation”实际上同时更换了约 2050 万可训练参数、projection 深度、时序上下文、候选密度和监督语义；
3. PhysTime 候选数只有 ActionFormer 对照约一半；
4. target assignment 与 ActionFormer 不同构；
5. 未归一化 absolute seconds 和无界 content logits 已经出现明显尺度支配和 coarse-level attention collapse。

也不是 `PIVOT`，因为当前负结果没有公平隔离 physical-time modeling。现有证据只否定：

> **PhysTime 1.0 这一具体实现优于 matched sparse ActionFormer。**

它没有否定：

> **在 feature provenance、容量、候选数和 assignment 同构后，物理时间原生检测器能否优于 selected-rank 或 post-hoc physical-grid。**

必须冻结 PhysTime 1.0 为负基线，停止在其上继续调 endpoint weight、NMS、spacing 或训练时长。下一步应从 capacity-matched control 重建，而不是局部修补。

---

# 3. P0/P1/P2 代码问题表

| 优先级    | 问题                                                                                 | 直接代码证据                                                                                                                                                                                                                                                                                          | 行为后果与裁决                                                                                                                                                                                                                         |
| ------ | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0** | **384 个 backbone feature 与 384 个 raw supports 的一一绑定没有 feature provenance**         | 配置先把 384 帧切成 24 个 16-frame chunk，再把 VideoMAE 输出拼成 tubelet 序列，最后 `Interpolate(size=384)`；VideoMAE temporal patch stride/tubelet size 为 2，TIA adapter 在 `total_frames // tubelet_size=192` 的 rank 序列上卷积。`phystime_adatad_sparse_k384.py:115-129`；`vit_adapter.py:779-824`；`vit_adapter.py:21-77`。 | 原生语义是 192 个 tubelet token，随后按 selected-rank 线性插值为 384。PhysTime 却要求 feature 长度与 raw-frame timestamp/support 数完全相等。现有 gate 只检查“长度为 384”，没有检查语义对应。由此 support-overlap attention 可能在错误的 anchor 上积分。**阻断所有“support-measure 正确实现”主张。** |
| **P0** | **所谓 head isolation 是架构、容量、候选和监督的联合替换**                                            | PhysTime 配置 `_delete_=True`，整体替换 detector、projection、head；ActionFormer projection 是 `(2 conv, 2 stem Transformer, 5 branch Transformer)`，PhysTime 只是六个独立 cross-attention。`phystime...py:86-165`；`actionformer_proj.py:74-174`。                                                                  | 对照有 27.70M 可训练 adapter+detector 参数，PhysTime 只有 7.36M，即 26.58%；projection 为 23.51M 对 3.17M。这个实验不能称“只换 head/坐标”。                                                                                                                  |
| **P0** | **候选数不匹配，且 query mask 把“无观测覆盖”错误等同于“无合法候选”**                                       | PhysTime 固定 `base_spacing_sec=0.5`；query 数由固定秒 spacing 决定。projection 最终使用 `output_mask = query_mask & coverage>eps`。`phystime...py:134-143`；`phystime_projection.py:136-152`。                                                                                                                   | 测试候选数 397.52 对 748.86，训练 343.78 对 710.34；短动作 `<1s` eligible locations/GT 只有 1.348，对照为约 2.3。长空洞中心的 query 又被直接删除，模型甚至不能利用周围上下文预测跨 gap 动作。                                                                                         |
| **P0** | **PhysTime target assignment 与 ActionFormer 不同构**                                  | PhysTime 对每个位置使用单一 `min_index`；ActionFormer 对相同最短持续时间的 GT 形成 `min_len_mask`，classification 可同时保留多标签，而 regression 才使用一个 `min_len_inds`。`phystime_head.py:150-192`；`anchor_free_head.py:415-497`。                                                                                                 | 约 7% 的 equal-duration conflict 位置使用了不同监督语义。分类、候选、坐标和架构已经不同，再叠加 assignment 差异，无法归因。                                                                                                                                              |
| **P0** | **raw absolute seconds 与无界 content dot product 破坏可辨识性和时间平移合理性**                    | query embedding 同时输入 `centers_sec`、normalized center、duration、raw width 等；content logits 是未经归一化的 query-key dot product。`phystime_projection.py:28-45,113-134`。                                                                                                                                  | scale proxy 显示 raw center 对第一层 preactivation 的贡献占比为 90.53%–95.31%；coarse levels content-logit span 达 59–67，而 relative span 约 0.2–0.4。绝对视频时间几乎成为 query identity。                                                                 |
| **P1** | **PhysTime 缺少 ActionFormer 等价的跨 query 时序上下文**                                      | 六个 level 都直接、独立地从原始 observations 做 attention；level 间没有父子聚合，query 间没有 stem/branch Transformer。`phystime_projection.py:161-228`。                                                                                                                                                                  | 每个 query 只得到局部 support 内若干 observation 的加权和；动作阶段关系、跨 gap 上下文和候选间竞争主要留给两个局部卷积 head，能力明显不等价。                                                                                                                                      |
| **P1** | **physical-grid 是 selected-rank feature geometry 与 physical point geometry 的后置拼接** | physical-grid 保留完整 selected-axis projection，只在 head 生成 points 后把 selected-rank center/stride 插值到 dense physical axis。`anchor_free_head.py:129-205`；配置只开启该映射。                                                                                                                                    | projection 的局部卷积、Transformer window 和 pooling 仍把相邻 selected ranks 当等间隔；监督/解码却使用不均匀物理距离。它是有意构造的错误几何 control，不应被当成合理 physical-time detector。                                                                                      |
| **P1** | **正式 dropout 破坏 measure-normalization 合同**                                         | `weights` 已归一化后直接执行 `Dropout`，正式配置 `dropout=0.1`。`phystime_projection.py:136-146`。                                                                                                                                                                                                              | 训练时每个 query 的权重和不再为 1；只有一两个支持 observation 时甚至可能把全部 measure path 清零。现有 measure-invariance test 使用默认 `dropout=0`，未覆盖正式配置。                                                                                                         |
| **P1** | **endpoint 是不完整且可能冲突的辅助任务**                                                        | endpoint 使用 regression tower 特征；每个 GT start/end 在每一级重复标正；loss 按全部 valid candidates 平均；推理只返回 cls score 与 proposal，不使用 endpoint probability。`phystime_head.py:114-147,193-256,272-280`。                                                                                                           | endpoint 对 mAP 只能通过共享 regression tower 间接起作用，不能增加候选、改善直接 score ranking 或在推理时校正边界。它可能帮助“已经命中”的 proposal，也可能与距离回归竞争。                                                                                                              |
| **P1** | **训练 sampling 的“No GT”表述过宽**                                                       | `random_fixed_subsample` 的 K 子采样由视频名和 dense-window geometry 确定；但训练时 dense 768 window 先通过带 GT 的 `random_trunc` 选取，代码明确要求 `gt_segments/gt_labels`。`end_to_end.py:699-754`。                                                                                                                        | 不构成 PhysTime 专属泄漏，因为三头完全共享；但准确表述必须是“**K384 子采样无 GT**”，不能说“整个训练窗口采样无 GT”。validation/test 确实不使用 GT。                                                                                                                               |
| **P1** | **预测诊断的 all-pred recall 和 matched-boundary MAE 容易被过度解释**                           | all-pred recall 对每个 GT 独立寻找任一 post-NMS proposal，同一 prediction 可服务多个 GT；boundary MAE 只在已经达到阈值的最佳匹配上统计。attention 诊断只取 8 个等距测试窗口。`analyze_phystime_prediction_diagnostics.py`；`run_phystime_attention_diagnostic.py:49-84,212-255`。                                                                | all-pred recall 是候选覆盖上界，不是 mAP；matched MAE 有成功条件选择偏差，只能解释“命中后边界质量”。8-window attention 是机制证据，不是总体统计结论。                                                                                                                           |
| **P1** | **远端证据不可由仓库独立重放**                                                                  | 结果页登记了 job、checkpoint epoch 和 artifact SHA，但仓库规则禁止提交原始文件。                                                                                                                                                                                                                                       | 不是造假证据，但 paper-ready artifact bundle 必须在可访问的归档中包含 checkpoint、prediction JSON、manifest 和原始日志。                                                                                                                                    |
| **P2** | **滑窗实现可能重复最后一个完整窗口**                                                               | `snippet_num // window_stride` 的循环中，最后一个超界窗口会向后贴齐；某些整除情形可能产生与前一窗口相同的末窗。`sliding_dataset.py:111-159`。                                                                                                                                                                                            | 三头共享，且全视频 NMS 会大幅抑制重复，因此不是 PhysTime 降点主因；仍应加 window manifest 去重测试。                                                                                                                                                              |
| **P2** | **global t=0 对齐和末端 clipped cell 造成 phase/edge asymmetry**                          | query cell 用 `floor(domain_start/width)` 对齐全局零点；末 cell 宽度被 duration 截短。`phystime_geometry.py:128-157`。                                                                                                                                                                                          | 同一局部动作在不同视频绝对起点附近可能落入不同 cell phase；末 cell 又使用缩短宽度归一化回归。影响次于容量、候选和 feature provenance。                                                                                                                                           |
| **P2** | **过期文档状态**                                                                         | `AGENTS.md:33` 与正式结果页冲突。                                                                                                                                                                                                                                                                        | 不影响执行，但破坏研究台账可信度，应立即修正。                                                                                                                                                                                                         |

## 已排除为主因的事项

以下检查通过或没有发现行为性错误：

* **K384 确实在 decode/backbone 前生效。** `LoadFrames` 先选 `frame_idxs=dense_window[keep_positions]`，配置随后才调用 `DecordDecode`；不是 dense decode 后丢帧。
* support cell 是每个 dense logical cell 的真实半 stride 支持区间；`clip_to_ownership_intervals` 只裁剪重叠，不跨 sparse gap 扩张。
* padding mask 要求有效 observation 是 prefix；真实空洞通过 support geometry 表示，而非伪装成 padding。
* test pipeline 不收集 GT，detector 显式拒绝 teacher、oracle、ledger、cache、actionness 等 metadata。
* PhysTime proposal 已经是秒时，通用后处理只执行 `[0,duration]` clamp，不会再次乘 snippet stride/FPS。
* optimizer builder 会同时加入 adapter custom group 和 detector group；real gate 也检查重复/缺失参数及非零梯度。
* 官方 mAP evaluator 读取 annotation JSON，并按原始 prediction score 排序，没有 per-video 自归一化。

---

# 4. 八条现有根因假设裁决

## 4.1 “下降不是 NaN、OOM、错误 checkpoint、重复秒转换、缺失测试窗口或 evaluator 错误”

**裁决：部分成立。置信度：代码路径 0.97；远端 artifact 0.70。**

代码可以直接排除：

* 重复秒转换；
* validation/test 使用 GT；
* evaluator 自归一化；
* K384 在 decode 后才生效；
* optimizer 漏掉 detector/adapter。

结果页登记正式三头完成，并写明无 NaN/OOM/AMP skip；审计页也确认最终官方 GT/evaluator 路径有效。

但 checkpoint 和原始日志不在仓库，故“最佳 checkpoint 与日志完全正确”不可独立验证。

**单变量证伪实验：**
从固定 `3ac93a1` clean snapshot，以已登记 checkpoint SHA 为输入，重新生成三头完整 prediction JSON；要求 JSON SHA、每类 prediction count、原始 mAP 五个阈值逐项一致。任何不一致都推翻该假设。

**若不成立，替代根因：** checkpoint/EMA 选择错误、外部数据 manifest 漂移或远端配置未与提交一致。

---

## 4.2 “三头 head isolation 不公平”

**裁决：成立。置信度：0.995。**

PhysTime 同时改变：

* 坐标；
* projection；
* 跨 query 上下文；
* 参数量；
* 候选数；
* assignment；
* endpoint 辅助监督；
* feature 与 timestamp 的绑定解释。

可训练 adapter+detector 只有 ActionFormer 的 26.58%，候选数约一半。

**单变量证伪实验：**
建立完全相同参数量、projection 深度、候选拓扑、head 和 assignment 的两个 ActionFormer，仅切换：

* selected-rank coordinates；
* physical-second coordinates/relative bias/seconds decode。

若差距仍存在，才可归因于坐标表达。

**若不成立，替代根因：** physical coordinate 本身与现有 backbone 特征不兼容，而非容量差异。

---

## 4.3 “未归一化 centers_sec/widths_sec 导致绝对秒主导”

**裁决：部分成立。置信度：0.94。**

代码明确把 raw `centers_sec` 和 raw `widths_sec` 直接送入 MLP。结果中 90.53%–95.31% 是一个**输入尺度 × 权重列范数 proxy**，不是严格的因果 attribution；它强烈证明 center scale 不健康，但尚未单独证明 raw width 的影响。

**单变量证伪实验：**

* 唯一变化：删除 raw center，保留 normalized center、raw width 和其他结构；
* 参数量通过同维度零常量输入保持不变；
* 比较 translation-shift test、attention logit span、Avg-mAP 和 @0.7。

**若不成立，替代根因：** key norm 或 unbounded query-key dot product，而不是 raw center 本身。

---

## 4.4 “coarse support attention 覆盖多 observation，但 effective count 坍缩；content 压过 relative”

**裁决：成立，但总体性证据有限。置信度：0.90。**

L3–L5 覆盖 observation 中位数分别为 15、28、56，但有效数约 2；content span 达 16.6、59.3、67.0，relative span只有约 0.5、0.4、0.2。

诊断正确地用 inverse participation ratio `1/Σw²` 计算 effective count，但只覆盖 8 个等距窗口。

**单变量证伪实验：**

1. content logits 关闭；
2. content logits cosine-normalized 且有界；
3. 当前 unbounded content。

保持相同 mass、candidate、projection depth 和 seed。若 bounded 版本不能恢复 effective fraction 或性能，则“content collapse 是主因”被推翻。

**替代根因：** observation features 本身过度同质、feature–support 错配，或 coarse query 缺少跨 query encoder。

---

## 4.5 “候选密度低主要损害短动作和高 tIoU”

**裁决：部分成立。置信度：0.89。**

直接证据：

* 测试候选数约 53%；
* `<1s` eligible locations/GT 为 1.348，对照约 2.3；
* `<1s` class-aware R@0.7 为 7.08%，对照约 50%；
* PhysTime all-pred class-aware R@0.7 为 79.70%，低于 selected 89.95%。

但这仍与容量、assignment 和 feature provenance 混杂。

**单变量证伪实验：**
只把 PhysTime query count 改成 `[384,192,96,48,24,12]`，保持每个 query 的物理秒坐标、attention 和 head 不变。若短动作 eligible count 恢复而短动作 R@0.7 不恢复，则候选数不是主要根因。

**替代根因：** feature–support 对齐、assignment 或 score calibration。

---

## 4.6 “单一 min_index assignment 与 ActionFormer 多标签行为不同构”

**裁决：成立。置信度：0.995。**

这是直接代码事实。ActionFormer classification 对 tied shortest GT 保留标签集合；PhysTime 只选一个 GT。当前诊断也记录了 equal-duration conflicts。

**单变量证伪实验：**
在 PhysTime 现有 head 中仅替换 assignment，完全复用 ActionFormer 的 `min_len_mask @ one_hot`；regression 仍选一个 shortest GT。比较 conflict locations 的 classification loss、class-aware recall 和 mAP。

**替代根因：** 若无提升，主要问题是候选覆盖或 representation，而非标签冲突。

---

## 4.7 “endpoint 可改善已命中边界，但无法补偿覆盖与排序”

**裁决：结构判断成立；实际收益证据不足。置信度：0.96。**

endpoint：

* 不创建候选；
* 不进入 inference score；
* 不直接参与 proposal decode；
* 只通过共享 regression tower 间接影响特征。

PhysTime 相对 physical-grid 的 @0.7 更高、matched-boundary error 更低，但 Avg-mAP 和总覆盖更低，这与该解释一致，不过不是 endpoint 的因果证明。

**单变量证伪实验：**
必须排在 provenance、容量、候选、assignment 修复之后，比较 endpoint weight `0` 与 `0.25`，且记录：

* candidate recall；
* top-100 recall；
* conditional boundary error；
* mAP@0.7。

若只改善 conditional MAE、不改善 mAP，则不能作为论文主贡献。

**替代根因：** PhysTime 高 @0.7 可能来自秒距离回归尺度，而非 endpoint。

---

## 4.8 “physical-grid 的下降主要是 selected-rank 邻接与 physical assignment 错配，而不是候选不足”

**裁决：部分成立。置信度：0.91。**

physical-grid 与 selected-axis 候选数完全相同，因此下降不能归因于候选数量。它在 projection 中继续使用 selected-rank 邻接和 rank-local Transformer，直到 head 才映射 point center/range/stride。

但“主要”仍未被单因素实验证明。

**单变量证伪实验：**

* 保持 physical points/GT/decode；
* 仅将 projection 的输入先通过一个无参数的、按 physical interval 聚合的 level-0 representation；
* head、候选、参数、assignment 不变。

若 physical-grid 恢复，则 adjacency mismatch 成立。

**替代根因：** local physical stride 对 regression range 的缩放、center-radius assignment 或前述 192→384 feature provenance。

---

# 5. 三种候选架构对比

| 候选                                                        | 核心                                                                                                                                                              |    创新性 |   审稿风险 | 实现复杂度 |  归因清晰度 |           公平性 |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | -----: | -----: | ----: | -----: | ------------: |
| **A. Capacity-matched Physical-Time ActionFormer**        | 完整保留 ActionFormer `(2,2,5)` projection、512 维、head 和候选拓扑，只把 grid、relative bias、assignment 与 decode 改成秒坐标                                                         |      中 | **最低** |     中 | **最高** |        **最高** |
| **B. Mass-Residual PhysTime**                             | 显式 support-mass average 为保底路径，叠加有界 content correction，再加跨 query encoder                                                                                         |     中高 |      中 |    中高 |      中 | 若不匹配候选/容量则不公平 |
| **C. Support-Measure Physical-Time ActionFormer，SM-PTAF** | 使用原生 tubelet feature provenance；不做 192→384 feature interpolation；以不跨 gap 的 support atoms 做 measure-preserving set-to-query lift；后接 ActionFormer 等容量物理时间 pyramid | **最高** |      中 |     高 |      高 |             高 |

## 对候选 A 的判断

A 是**必须实现的科学 control**。它最适合回答：

> 相同容量、候选数、监督和跨时序上下文下，physical coordinates 到底比 selected-rank coordinates 好还是坏？

但单独把 ActionFormer 加 timestamp/relative bias，论文创新性不足，容易被评价为 TE-TAD 式 actual timeline coordinate 的局部改造。

## 对候选 B 的判断

B 修复了当前 attention collapse 的核心缺陷。推荐：

[
z_{\text{mass}}(q)=
\frac{\sum_i m_{qi}v_i}{\sum_i m_{qi}+\epsilon}
]

[
\ell_{qi}
=========

\alpha_l\cos(q_q,k_i)
+\beta_l\tanh r_l(\Delta t_{qi}/w_q)
+\log\left(\frac{m_{qi}}{w_q}+\epsilon\right)
]

[
z_q=\operatorname{LN}\left(
z_{\text{mass}}(q)+
\gamma_l W_o\sum_i \operatorname{softmax}(\ell_{qi})v_i
\right).
]

其中：

* `gamma_l` 不应精确零初始化；精确为零会让第一步 q/k correction 梯度为零。应初始化为约 `1e-3`；
* `alpha_l/beta_l` 必须通过 sigmoid/tanh 限制在有限范围；
* dropout 只能作用于 correction output，不能作用于归一化后的 mass weights；
* 必须加跨 query encoder，否则仍只是独立 cell pooling。

## 唯一推荐

## **候选 C：SM-PTAF**

A 作为必要对照，B 的 mass residual 作为 C 的核心算子。最小不可分贡献不是“三个模块堆叠”，而是：

> **经过审计的 native feature support provenance，加上 measure-preserving、无插值的 set-to-physical-query pyramid。**

ActionFormer 等容量 encoder、候选数对齐和多标签 assignment 是公平性控制，不应包装成贡献。

---

# 6. 唯一推荐的最终模型：SM-PTAF

## 6.1 输入和原生 feature geometry

输入：

* RGB：`frames ∈ R[B, N=1, 3, Kmax, H, W]`
* raw observation mask：`Mraw ∈ {0,1}[B,Kmax]`
* timestamps：`T ∈ R[B,Kmax]`，单位秒
* raw supports：`S ∈ R[B,Kmax,2]`，单位秒
* physical domain：`[a_b,b_b]`
* full-video duration：`D_b`，单位秒

Phase-1：`Kmax=384`。

VideoMAE-S tubelet size 为 2。因此在删除最后的 `Interpolate(size=384)` 后，原生时序输出为：

[
F\in\mathbb{R}^{B\times384\times J},\qquad
J=\left\lceil K_{\text{valid}}/2\right\rceil.
]

K=384 时 `J=192`。

每个 tubelet token `j` 绑定两个**不连续也不扩张**的 support atoms：

[
A_{j,0}=S_{2j},\qquad A_{j,1}=S_{2j+1}.
]

数据结构：

* `atoms ∈ R[B,J,2,2]`
* `atom_mask ∈ {0,1}[B,J,2]`
* `token_mask = atom_mask.any(-1)`

如果两帧之间存在大 sparse gap，禁止把两个 atom 合成 `[left_0,right_1]`，因为那会伪造未观测区间。

需要明确：support atom 是 feature 的**时间 anchor provenance**，不是该 feature 完整神经感受野。TIA 和 ViT 已经混入上下文，但 anchor 仍必须与原生 tubelet 生成过程一致。

## 6.2 候选数和物理网格

为实现与 ActionFormer 完全一致的候选拓扑：

[
Q_l=\left\lceil \frac{K_{\text{valid}}}{2^l}\right\rceil,
\qquad l=0,\ldots,5.
]

K=384 时：

[
[Q_0,\ldots,Q_5]=[384,192,96,48,24,12],
]

总数 756，与 ActionFormer 的理想候选拓扑完全相同。

定义：

[
\delta_b=\frac{b_b-a_b}{K_{\text{valid},b}}.
]

第 `l` 层第 `q` 个物理 cell：

[
I_{blq}=
\left[
a_b+q,2^l\delta_b,;
\min\left(a_b+(q+1)2^l\delta_b,b_b\right)
\right].
]

注意：

* `K` 只决定**计算预算和候选个数**；
* center、width、GT assignment、decode 全部是秒；
* 不允许把 GT 或预测投回 selected rank；
* query domain mask 与 observation coverage mask 必须分开；
* 即使 cell 内没有 observation，也保留候选，用 gap token 和周围 query context 预测。

建议修改 `RTK.md:25`：原“query grid 不由 K 定义”应改为：

> query 坐标和回归几何必须由物理时间定义；在 matched comparison 中，候选基数允许由 K 决定，以保证 detector capacity 和 candidate topology 与对照相同。

## 6.3 support-measure lift

对 query `q` 和 native tubelet token `j`：

[
m_{qj}
======

\sum_{a=1}^{2}
\left|I_q\cap A_{ja}\right|
\cdot M_{ja}.
]

显式 mass path：

[
z_{\text{mass},q}
=================

\frac{
\sum_j m_{qj}W_vF_j
}{
\sum_j m_{qj}+\epsilon
}.
]

如果 `Σ_j m_qj=0`，使用学习的 `gap_token`，而不是删除 query。

content correction 使用 mass-pooled local content 生成 query，而不是从 absolute center 生成：

[
\hat q_q=
\operatorname{norm}
\left(
W_q[z_{\text{mass},q};e_{\text{scale},q}]
\right),
\qquad
\hat k_j=\operatorname{norm}(W_kF_j).
]

尺度 embedding 只允许：

[
e_{\text{scale},q}
==================

\operatorname{MLP}
\left[
\operatorname{clip}\log(w_q/1s),;
\operatorname{clip}\log(D/1s),;
\frac{\sum_jm_{qj}}{w_q}
\right].
]

禁止 raw absolute center。

relative features：

[
r_{qj}=
\left[
\frac{\bar t_{qj}-c_q}{w_q},
\left|\frac{\bar t_{qj}-c_q}{w_q}\right|,
\log\frac{\bar w_{qj}}{w_q},
\frac{m_{qj}}{w_q}
\right].
]

有界 logits：

[
\ell_{qj}
=========

\alpha_l\cos(\hat q_q,\hat k_j)
+
\beta_l\tanh(g_l(r_{qj}))
+
\log\left(\frac{m_{qj}}{w_q}+\epsilon\right),
]

其中：

[
\alpha_l=4\sigma(\tilde \alpha_l),\qquad
\beta_l=4\sigma(\tilde \beta_l).
]

最终：

[
z_q=
\operatorname{LN}
\left(
z_{\text{mass},q}
+e_{\text{scale},q}
+\gamma_l\operatorname{Dropout}
\left[
W_o\sum_j\operatorname{softmax}(\ell_{qj})W_cF_j
\right]
\right).
]

取：

[
\gamma_l=\sigma(\tilde \gamma_l),
\qquad \gamma_l(0)\approx10^{-3}.
]

## 6.4 跨 query encoder

使用与 ActionFormer 对齐的：

* `d_model=512`
* `n_head=4`
* local window size 19
* `arch=(2,2,5)`
* path dropout 0.1
* 两个 stem blocks
* 五个 downsampling branch blocks

但在 query-query attention 中加入有界 physical relative bias：

[
b_{qp}
======

\eta_l\tanh h_l
\left[
\frac{c_p-c_q}{\sqrt{w_pw_q}},
\log\frac{w_p}{w_q},
\frac{|I_p\cap I_q|}{\sqrt{w_pw_q}}
\right].
]

高层 query feature 从低层 query 按 interval-overlap mass 聚合，不再让六个 level 各自直接访问 raw observations。

参数量要求：

* projection+head 与 ActionFormer control 差异不超过 1%；
* 不允许加入未参与 forward 的 dummy parameters；
* 可通过调整 observation pointwise FFN expansion ratio 实现参数匹配；
* 参数、MACs、候选数必须由 validator 自动登记。

## 6.5 Head 与 decode

复用 `AnchorFreeHead`：

* 2 层 classification tower；
* 2 层 regression tower；
* Focal Loss；
* DIoU Loss；
* tied-shortest multi-label classification assignment；
* one-shortest regression target。

物理 points：

[
p_{lq}=
[c_{lq},R_l^{\min},R_l^{\max},s_{lq}],
]

其中 nominal stride：

[
s_{lq}=2^l\delta,
]

regression ranges：

[
\delta\times
[(0,4),(4,8),(8,16),(16,32),(32,64),(64,\infty)].
]

解码：

[
\hat s_{lq}=c_{lq}-d^L_{lq}s_{lq},
\qquad
\hat e_{lq}=c_{lq}+d^R_{lq}s_{lq}.
]

输出保持秒坐标，只做 duration clamp，再复用现有 NMS/evaluator。

**主模型不保留 endpoint branch。**
只有在结构修复后 endpoint-off/on 因果实验明确改善 mAP@0.7 时，才允许作为附加消融；不得先进入主模型。

## 6.6 预期行为

* **短动作：** level-0 候选由约 200 恢复到 384，物理 cell 宽度约减半；与 ActionFormer 的 eligible density 同构。
* **长空洞：** 空洞 query 不再被 mask；gap token 可通过跨 query encoder利用左右上下文。
* **时间平移：** 平移所有 timestamp/support/domain/GT 后，内部相对表示不变，proposal 同幅平移。
* **不同 FPS：** 同一真实时间观测及 support 下，几何不依赖 frame rank 或 FPS。
* **时长缩放：** 不声称严格 scale invariance。normalized relative path具有协变性，同时 bounded log-width channel保留 0.5 秒与 5 秒的真实语义差异。
* **不同 K：** 候选数量随有效观测预算变化，但坐标始终是秒。

---

# 7. 训练与监督方案

## 7.1 主损失

不需要课程学习，不需要三阶段预训练，不需要先冻结 projection/head。

[
N_t=0.9N_{t-1}+0.1\max(n_{\text{pos}},1).
]

[
L_{\text{cls}}
==============

\frac{
\sum_{\text{valid}} L_{\text{focal}}
}{
N_t
},
]

[
L_{\text{reg}}
==============

\frac{
\sum_{\text{positive}}
L_{\text{DIoU}}
(\hat y,y)
}{
N_t
}.
]

[
L=L_{\text{cls}}+L_{\text{reg}}.
]

DIoU 是无量纲的，因此不会因秒、FPS 或 duration 直接改变 loss 单位。分类和回归共享与 ActionFormer 相同的 positive normalizer。

## 7.2 Assignment

对每个 point：

1. 计算到所有 GT 左右边界的秒距离；
2. 应用 physical center sampling；
3. 应用秒单位 regression range；
4. 找最短 eligible GT duration；
5. classification 保留与最短 duration 相差不超过 `1e-3` 的全部 GT 标签；
6. regression 使用其中一个最短 GT。

这与 ActionFormer 当前语义同构。

## 7.3 梯度路径

梯度应为：

```text
Focal/DIoU
  -> AnchorFreeHead
  -> physical query Transformer
  -> support-measure lift
  -> native VideoMAE tubelet features
  -> TIA adapters
```

冻结：

* VideoMAE 原始 backbone 参数；

训练：

* TIA adapters；
* support-measure projection；
* query encoder；
* classification/regression head。

physical interval、overlap mass 和 relative features 在 FP32 计算；feature/value projections 可继续 AMP。FP32 island 只应解决时间数值精度，不改变 loss scaling 或 detach 梯度。

## 7.4 Train/test 同构和 no-leak

训练和测试必须共用：

* native feature extraction；
* atom geometry；
* candidate grid；
* mass lift；
* query encoder；
* head；
* seconds decode。

唯一差异只能是：

* dropout/train mode；
* 训练时提供 GT 并计算 loss。

推理 metadata 继续拒绝：

* GT；
* teacher；
* oracle；
* selector；
* actionness；
* ledger；
* prediction cache；
* dynamic budget。

---

# 8. 核心实现代码

下面是应进入仓库的核心逻辑，而不是完整重写。

## 8.1 Native tubelet support 与 candidate grid

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor


@dataclass
class PhysicalLevel:
    intervals_sec: Tensor      # [B, Q, 2]
    centers_sec: Tensor        # [B, Q]
    widths_sec: Tensor         # clipped cell width [B, Q]
    nominal_stride_sec: Tensor # [B, Q]
    points_sec: Tensor         # [B, Q, 4]: center, min_range, max_range, stride
    valid_mask: Tensor         # [B, Q]


def build_tubelet_support_atoms(
    raw_supports_sec: Tensor,  # [B, K, 2]
    raw_mask: Tensor,          # [B, K]
    tubelet_size: int = 2,
) -> tuple[Tensor, Tensor, Tensor]:
    """Map raw supports to native VideoMAE tubelet atoms without filling gaps."""
    if tubelet_size != 2:
        raise NotImplementedError("The audited VideoMAE-S configuration uses tubelet_size=2.")
    if raw_supports_sec.ndim != 3 or raw_supports_sec.shape[-1] != 2:
        raise ValueError("raw_supports_sec must have shape [B,K,2].")
    if raw_mask.shape != raw_supports_sec.shape[:2]:
        raise ValueError("raw_mask shape mismatch.")

    bsz, kmax, _ = raw_supports_sec.shape
    if kmax % 2:
        raw_supports_sec = torch.cat(
            [raw_supports_sec, raw_supports_sec.new_zeros(bsz, 1, 2)], dim=1
        )
        raw_mask = torch.cat(
            [raw_mask, raw_mask.new_zeros(bsz, 1)], dim=1
        )

    atoms = raw_supports_sec.reshape(bsz, -1, 2, 2)
    atom_mask = raw_mask.reshape(bsz, -1, 2).bool()
    token_mask = atom_mask.any(dim=-1)
    return atoms, atom_mask, token_mask


def build_physical_candidate_pyramid(
    domain_start_sec: Tensor,  # [B]
    domain_end_sec: Tensor,    # [B]
    valid_k: Tensor,           # [B], integer observation budget
    num_levels: int = 6,
) -> List[PhysicalLevel]:
    range_factors = (
        (0.0, 4.0),
        (4.0, 8.0),
        (8.0, 16.0),
        (16.0, 32.0),
        (32.0, 64.0),
        (64.0, float("inf")),
    )
    if num_levels != len(range_factors):
        raise ValueError("SM-PTAF currently matches the six-level ActionFormer topology.")

    valid_k = valid_k.to(torch.long)
    duration = domain_end_sec - domain_start_sec
    if torch.any(valid_k <= 0) or torch.any(duration <= 0):
        raise ValueError("Every sample needs a positive K and physical domain.")

    base_stride = duration / valid_k.to(duration.dtype)
    levels: List[PhysicalLevel] = []

    for level, (rmin_factor, rmax_factor) in enumerate(range_factors):
        scale = 2**level
        count = torch.div(valid_k + scale - 1, scale, rounding_mode="floor")
        qmax = int(count.max().item())
        slot = torch.arange(qmax, device=duration.device)

        nominal_stride = base_stride * float(scale)
        left = domain_start_sec[:, None] + slot[None, :] * nominal_stride[:, None]
        right = torch.minimum(
            left + nominal_stride[:, None],
            domain_end_sec[:, None],
        )
        valid = (slot[None, :] < count[:, None]) & (right > left)

        intervals = torch.stack((left, right), dim=-1)
        intervals = torch.where(valid[..., None], intervals, torch.zeros_like(intervals))
        centers = 0.5 * (intervals[..., 0] + intervals[..., 1])
        widths = (intervals[..., 1] - intervals[..., 0]).clamp_min(0)

        stride = nominal_stride[:, None].expand_as(centers)
        lower = base_stride[:, None] * rmin_factor
        if rmax_factor == float("inf"):
            upper = torch.full_like(centers, 1.0e8)
        else:
            upper = base_stride[:, None] * rmax_factor

        points = torch.stack(
            (centers, lower.expand_as(centers), upper.expand_as(centers), stride),
            dim=-1,
        )
        points = torch.where(valid[..., None], points, torch.zeros_like(points))
        levels.append(
            PhysicalLevel(
                intervals_sec=intervals,
                centers_sec=centers,
                widths_sec=widths,
                nominal_stride_sec=stride,
                points_sec=points,
                valid_mask=valid,
            )
        )
    return levels
```

## 8.2 Mass-residual lift

```python
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _inverse_sigmoid(probability: float) -> float:
    probability = min(max(probability, 1e-6), 1.0 - 1e-6)
    return math.log(probability / (1.0 - probability))


class SupportMassResidualLift(nn.Module):
    """No-imputation set-to-query lift with an explicit measure-preserving path."""

    def __init__(
        self,
        in_channels: int = 384,
        out_channels: int = 512,
        attention_channels: int = 128,
        dropout: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.eps = float(eps)

        # Pointwise observation encoder: no selected-rank temporal convolution.
        self.obs_encoder = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.query_proj = nn.Linear(out_channels, attention_channels)
        self.key_proj = nn.Linear(out_channels, attention_channels)
        self.correction_value = nn.Linear(out_channels, out_channels)
        self.correction_out = nn.Linear(out_channels, out_channels)

        self.scale_embed = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )
        self.relative_mlp = nn.Sequential(
            nn.Linear(4, attention_channels),
            nn.GELU(),
            nn.Linear(attention_channels, 1),
        )

        self.gap_token = nn.Parameter(torch.zeros(out_channels))
        self.alpha_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(0.5 / 4.0)))
        self.beta_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(1.0 / 4.0)))
        self.gamma_raw = nn.Parameter(torch.tensor(_inverse_sigmoid(1.0e-3)))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)

    def forward(
        self,
        observations: Tensor,    # [B, J, Cin]
        atoms_sec: Tensor,       # [B, J, A, 2]
        atom_mask: Tensor,       # [B, J, A]
        query_intervals: Tensor, # [B, Q, 2]
        domain_duration_sec: Tensor,  # [B]
    ) -> tuple[Tensor, dict[str, Tensor]]:
        qleft = query_intervals[:, :, None, None, 0]
        qright = query_intervals[:, :, None, None, 1]
        aleft = atoms_sec[:, None, :, :, 0]
        aright = atoms_sec[:, None, :, :, 1]

        atom_mass = (
            torch.minimum(qright, aright) - torch.maximum(qleft, aleft)
        ).clamp_min(0)
        atom_mass = atom_mass * atom_mask[:, None].to(atom_mass.dtype)

        mass = atom_mass.sum(dim=-1)  # [B,Q,J]
        coverage = mass.sum(dim=-1)  # [B,Q]
        covered_query = coverage > self.eps

        obs = self.obs_encoder(observations)
        mass_weights = mass / coverage[..., None].clamp_min(self.eps)
        z_mass = torch.einsum("bqj,bjc->bqc", mass_weights, obs)

        qwidth = (query_intervals[..., 1] - query_intervals[..., 0]).clamp_min(self.eps)
        qcenter = 0.5 * (query_intervals[..., 0] + query_intervals[..., 1])
        coverage_fraction = (coverage / qwidth).clamp(0.0, 1.0)

        scale_features = torch.stack(
            (
                torch.log(qwidth).clamp(-8.0, 8.0),
                torch.log(domain_duration_sec[:, None]).clamp(-8.0, 8.0),
                coverage_fraction,
            ),
            dim=-1,
        )
        scale_embedding = self.scale_embed(scale_features)

        base = torch.where(
            covered_query[..., None],
            z_mass,
            self.gap_token[None, None, :],
        )
        base = base + scale_embedding

        query = F.normalize(self.query_proj(base), dim=-1)
        key = F.normalize(self.key_proj(obs), dim=-1)
        cosine = torch.einsum("bqd,bjd->bqj", query, key)

        atom_center = 0.5 * (atoms_sec[..., 0] + atoms_sec[..., 1])
        atom_width = (atoms_sec[..., 1] - atoms_sec[..., 0]).clamp_min(self.eps)
        signed = (
            atom_center[:, None] - qcenter[:, :, None, None]
        ) / qwidth[:, :, None, None]

        relative_atom = torch.stack(
            (
                signed,
                signed.abs(),
                torch.log(
                    atom_width[:, None] / qwidth[:, :, None, None]
                ).clamp(-8.0, 8.0),
                atom_mass / qwidth[:, :, None, None],
            ),
            dim=-1,
        )
        atom_conditional = atom_mass / mass[..., None].clamp_min(self.eps)
        relative = (
            relative_atom * atom_conditional[..., None]
        ).sum(dim=-2)

        alpha = 4.0 * torch.sigmoid(self.alpha_raw)
        beta = 4.0 * torch.sigmoid(self.beta_raw)
        relative_logit = beta * torch.tanh(self.relative_mlp(relative).squeeze(-1))
        normalized_mass = mass / qwidth[..., None]

        logits = (
            alpha * cosine
            + relative_logit
            + torch.log(normalized_mass.clamp_min(self.eps))
        )
        supported = mass > 0
        logits = logits.masked_fill(~supported, torch.finfo(logits.dtype).min)

        correction_weights = torch.softmax(logits, dim=-1)
        correction_weights = torch.where(
            covered_query[..., None],
            correction_weights,
            torch.zeros_like(correction_weights),
        )
        correction = torch.einsum(
            "bqj,bjc->bqc",
            correction_weights,
            self.correction_value(obs),
        )

        gamma = torch.sigmoid(self.gamma_raw)
        output = self.norm(
            base + gamma * self.dropout(self.correction_out(correction))
        )
        return output, {
            "mass": mass,
            "coverage_sec": coverage,
            "mass_weights": mass_weights,
            "correction_weights": correction_weights,
            "alpha": alpha,
            "beta": beta,
            "gamma": gamma,
        }
```

## 8.3 ActionFormer-equivalent target assignment

```python
@torch.no_grad()
def assign_physical_targets(
    points: Tensor,       # [Q,4]: center,min_range,max_range,stride, all seconds
    gt_segments: Tensor,  # [N,2], seconds
    gt_labels: Tensor,    # [N]
    num_classes: int,
    center_radius: float = 1.5,
) -> tuple[Tensor, Tensor]:
    q = points.shape[0]
    n = gt_segments.shape[0]
    if n == 0:
        return (
            points.new_zeros(q, num_classes),
            points.new_zeros(q, 2),
        )

    center = points[:, 0, None]
    stride = points[:, 3, None]
    gt = gt_segments[None].expand(q, n, 2)

    left = center - gt[..., 0]
    right = gt[..., 1] - center
    distances = torch.stack((left, right), dim=-1)

    gt_center = 0.5 * (gt[..., 0] + gt[..., 1])
    sample_left = torch.maximum(
        gt_center - center_radius * stride, gt[..., 0]
    )
    sample_right = torch.minimum(
        gt_center + center_radius * stride, gt[..., 1]
    )
    inside_center = (
        torch.stack(
            (center - sample_left, sample_right - center), dim=-1
        ).amin(dim=-1)
        > 0
    )

    max_distance = distances.amax(dim=-1)
    inside_range = (
        (max_distance >= points[:, 1, None])
        & (max_distance <= points[:, 2, None])
    )
    eligible = inside_center & inside_range

    duration = (gt_segments[:, 1] - gt_segments[:, 0])[None].expand(q, n)
    duration = duration.masked_fill(~eligible, float("inf"))
    min_duration, min_index = duration.min(dim=1)

    tied_shortest = (
        (duration <= min_duration[:, None] + 1e-3)
        & torch.isfinite(duration)
    )
    one_hot = F.one_hot(
        gt_labels.long(), num_classes
    ).to(points.dtype)
    cls_target = tied_shortest.to(points.dtype) @ one_hot
    cls_target.clamp_(0.0, 1.0)

    reg_target = distances[torch.arange(q, device=points.device), min_index]
    reg_target = reg_target / points[:, 3, None].clamp_min(1e-6)
    reg_target[~torch.isfinite(min_duration)] = 0.0
    return cls_target, reg_target
```

## 8.4 Seconds decode 与核心 regression tests

```python
def decode_seconds(
    points_sec: Tensor,  # [B,Q,4]
    reg_pred: Tensor,    # [B,Q,2], non-negative
    duration_sec: Tensor,
) -> Tensor:
    center = points_sec[..., 0]
    stride = points_sec[..., 3]
    start = center - reg_pred[..., 0] * stride
    end = center + reg_pred[..., 1] * stride
    segments = torch.stack((start, end), dim=-1)
    segments[..., 0].clamp_(min=0.0)
    segments[..., 1] = torch.minimum(
        segments[..., 1], duration_sec[:, None]
    )
    return segments


def test_candidate_parity_k384():
    start = torch.tensor([0.0])
    end = torch.tensor([102.4])
    levels = build_physical_candidate_pyramid(
        start, end, torch.tensor([384])
    )
    assert [int(level.valid_mask.sum()) for level in levels] == [
        384, 192, 96, 48, 24, 12
    ]


def test_tubelet_atoms_do_not_fill_sparse_gap():
    supports = torch.tensor(
        [[[0.0, 0.1], [10.0, 10.1]]]
    )
    atoms, atom_mask, token_mask = build_tubelet_support_atoms(
        supports, torch.ones(1, 2, dtype=torch.bool)
    )
    assert atoms.shape == (1, 1, 2, 2)
    assert torch.equal(atoms[0, 0, 0], supports[0, 0])
    assert torch.equal(atoms[0, 0, 1], supports[0, 1])
    # No synthetic [0.0,10.1] convex hull is ever constructed.
    assert atom_mask.all() and token_mask.all()


def test_mass_path_is_preserved_in_training_mode():
    lift = SupportMassResidualLift(dropout=0.5).train()
    # gamma is near zero, but non-zero; dropout never touches mass_weights.
    assert 0.0 < float(torch.sigmoid(lift.gamma_raw)) < 0.01


def test_translation_equivariance(model, batch):
    delta = 17.0
    logits_a, seg_a = model.forward_raw(**batch)

    shifted = shift_all_physical_metadata(batch, delta)
    logits_b, seg_b = model.forward_raw(**shifted)

    torch.testing.assert_close(logits_a, logits_b, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(seg_b, seg_a + delta, atol=1e-4, rtol=0)


def assert_optimizer_coverage(model, optimizer):
    optimized = {
        id(p)
        for group in optimizer.param_groups
        for p in group["params"]
    }
    required = {
        id(p): name
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    missing = [name for pid, name in required.items() if pid not in optimized]
    assert not missing, missing
    assert sum(len(g["params"]) for g in optimizer.param_groups) == len(optimized)
```

---

# 9. 文件级 patch map 与测试合同

| 文件                                                                | 操作   | 类/函数                                              | 精确修改                                                                   | 目的                       | 必测行为                                       |
| ----------------------------------------------------------------- | ---- | ------------------------------------------------- | ---------------------------------------------------------------------- | ------------------------ | ------------------------------------------ |
| `configs/adatad/thumos/selected_axis_adatad_sparse_k384.py`       | 保留冻结 | 全配置                                               | 不改变正式 baseline                                                         | 固定 selected-axis anchor  | config SHA 不变                              |
| `physical_grid_adatad_sparse_k384.py`                             | 保留冻结 | 全配置                                               | 继续作为 post-hoc geometry mismatch control                                | 保留负 control              | 候选数与 selected 完全一致                         |
| `phystime_adatad_sparse_k384.py`                                  | 保留冻结 | 全配置                                               | 明确标记 `PhysTime 1.0 negative baseline`                                  | 防止覆盖历史结果                 | 禁止主实验引用为 2.0                               |
| `phystime_af_capacity_selected_coord_k384.py`                     | 新增   | 配置                                                | 等容量 selected-coordinate control                                        | 隔离坐标变量                   | 参数、候选、assignment hash                      |
| `phystime_af_capacity_physical_coord_k384.py`                     | 新增   | 配置                                                | 只切换 physical grid/bias/decode                                          | 候选 A                     | 与 selected control 仅坐标差异                   |
| `sm_ptaf_adatad_sparse_k384.py`                                   | 新增   | 配置                                                | native 192 tubelets、SM-PTAF projection、AF head                         | 最终主方法                    | 禁止 feature Interpolate                     |
| `opentad/datasets/transforms/phystime_raw.py`                     | 修改   | `BuildPhysTimeRawFrameGeometry`                   | 保留 raw supports；增加 `window_crop_uses_gt` 与 `subsample_uses_gt` 两个分离字段  | 修正 no-GT 表述              | test 必须两者均 false；train 只允许前者 true          |
| `opentad/models/utils/phystime_feature_geometry.py`               | 新增   | `build_tubelet_support_atoms`                     | raw supports → multi-atom native tubelet supports                      | 关闭 feature provenance P0 | K384→J192；不跨 gap                           |
| `opentad/models/backbones/backbone_wrapper.py`                    | 修改   | forward/postprocess                               | 可选 `return_native_temporal=True`，跳过最终 Interpolate；生成 tubelet mask      | 暴露原生 feature             | 输出长度 J，不是 K                                |
| `opentad/models/backbones/vit_adapter.py`                         | 小改   | metadata export                                   | 暴露 `tubelet_size`、native temporal length；不改变正式 baseline forward        | 可审计 provenance           | adapter temporal size 与 J 一致               |
| `opentad/models/projections/support_measure_actionformer_proj.py` | 新增   | `SupportMassResidualLift`、`PhysicalRelativeBlock` | measure lift + `(2,2,5)` physical query encoder                        | 主核心                      | mass preservation、bounded logits、gap query |
| `opentad/models/projections/phystime_projection.py`               | 冻结   | 现有类                                               | 不继续修主线；只供 1.0 baseline                                                 | 防止隐式改历史                  | 原有测试继续通过                                   |
| `opentad/models/dense_heads/anchor_free_head.py`                  | 修改   | point override/target assignment                  | 正式支持 batched physical points；保持现有 AF assignment                        | 最大化复用                    | old AF 数值回归不变                              |
| `opentad/models/dense_heads/phystime_head.py`                     | 冻结   | 现有类                                               | 不进入 2.0 主配置                                                            | 保留负 baseline             | endpoint 历史行为不变                            |
| `opentad/models/detectors/sm_ptaf.py`                             | 新增   | `SMPTAF`                                          | 路由 raw geometry、native token geometry、projection、AF head               | 最小 detector glue         | train/test 同构、no-leak                      |
| `opentad/models/detectors/phystime_tad.py`                        | 冻结   | `PhysTimeTAD`                                     | 仅修文档，不改变正式行为                                                           | 保存 1.0                   | checkpoint 可加载                             |
| 各 `__init__.py`                                                   | 修改   | registry                                          | 注册新 projection/detector/util                                           | 构建支持                     | registry smoke                             |
| `opentad/cores/train_engine.py`                                   | 保留   | train loop                                        | 不引入特殊多阶段；可加入 optimizer coverage fail-closed hook                       | 避免新训练异构                  | AMP/FP32/grad finite                       |
| `tools/bata/validate_sm_ptaf_track.py`                            | 新增   | validator                                         | 检查 native J、atom provenance、params±1%、candidate parity、assignment hash | 主科学合同                    | 任一混杂 fail closed                           |
| `tools/bata/run_sm_ptaf_real_gate.py`                             | 新增   | CUDA gate                                         | raw decode、native feature、one-step、inference、seconds decode            | 真数据 gate                 | 不把 gate 当 mAP                              |
| `tools/bata/run_sm_ptaf_p0_gates.py`                              | 新增   | causal launcher                                   | 按序执行单因素 gates                                                          | 避免 full matrix           | 每次只允许一个 diff                               |
| `tests/test_sm_ptaf_feature_geometry.py`                          | 新增   | tests                                             | tubelet atom、padding、gap                                               | provenance               | 不生成 convex hull                            |
| `tests/test_sm_ptaf_candidate_parity.py`                          | 新增   | tests                                             | `[384,192,96,48,24,12]`                                                | 公平候选                     | 对照逐层相同                                     |
| `tests/test_sm_ptaf_measure_attention.py`                         | 新增   | tests                                             | mass residual、dropout、bounded logits                                   | 防 collapse               | train mode 下 mass path不丢                   |
| `tests/test_sm_ptaf_assignment.py`                                | 新增   | tests                                             | tied multi-label 与 AF bitwise 对照                                       | 监督公平                     | cls target 完全一致                            |
| `tests/test_sm_ptaf_equivariance.py`                              | 新增   | tests                                             | 时间平移、FPS重参数化                                                           | 几何合同                     | logits不变、proposal平移                        |
| `tests/test_sm_ptaf_one_step.py`                                  | 新增   | tests                                             | adapter/projection/head grad                                           | optimizer coverage       | 所有必需梯度非零有限                                 |
| `RTK.md`                                                          | 修改   | physical contract                                 | 允许 K 只决定 matched candidate cardinality；撤销“只改变 head”描述                  | 修正规格                     | 文档与 validator 一致                           |
| `AGENTS.md`                                                       | 修改   | status                                            | 删除“尚无 mAP”过期文本                                                         | 台账一致                     | docs consistency test                      |
| `docs/evaluation/results.md`                                      | 追加   | 新实验表                                              | 只写真实数字、hash、manifest                                                   | 唯一数字源                    | 无 Wiki 数字复制                                |

---

# 10. 最小因果 gates、正式实验矩阵和 GO/HOLD/STOP 条件

## 10.1 P0 gates

| Gate                      | 唯一变化                                                             | 首要观测                                                           | 推翻什么                                            | 是否进入 full train                   |
| ------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------------- | ----------------------------------------------- | --------------------------------- |
| **G0 feature provenance** | 384 插值 feature+raw support 绑定 → native 192 tubelet+atom supports | native length、Jacobian/provenance、mass cells、one-step gradient | 若二者语义等价且输出无差异，则 provenance 不是主因                 | 先静态/CUDA gate，不直接 full train      |
| **G1 coordinate-only**    | 同一 capacity-matched AF，仅 selected coordinates ↔ physical seconds | Avg、@0.7、short R@0.7、eligible/GT                               | 物理坐标本身是否有价值                                     | **第一个允许 full single-seed 的 gate** |
| **G2 raw seconds**        | raw absolute center on/off；其余全同                                  | translation test、content span、mAP                              | raw center dominance                            | 只跑 survivor                       |
| **G3 candidate count**    | 当前约 398 candidates ↔ AF-matched 756                              | short eligible、all/top100 recall、@0.7                          | 候选不足是否主因                                        | 只跑 survivor                       |
| **G4 mass residual**      | 无显式 mass base ↔ mass residual                                    | effective count、coverage、gap behavior、mAP                      | attention collapse 是否由 learned logits覆盖 measure | 只跑 survivor                       |
| **G5 content logits**     | off ↔ bounded cosine ↔ unbounded dot product                     | coarse effective fraction、logit span、mAP                       | content correction 是否必要/是否过强                    | 只保留 bounded 或 off                 |
| **G6 assignment**         | single-label ↔ AF-equivalent tied multi-label                    | conflict-location loss、class-aware recall                      | assignment 混杂                                   | 修复后固定，不再作为可调参数                    |
| **G7 endpoint**           | weight 0 ↔ 0.25                                                  | candidate recall、top100、conditional MAE、@0.7                   | endpoint 是否有净检测收益                               | 必须最后执行                            |

G0 还应增加一个 feature Jacobian 诊断：

[
J_{ji}=\left|\frac{\partial F_j}{\partial X_i}\right|,
]

用来确认 native tubelet token 与 raw frame pair 的实际依赖，以及 rank-TIA 对邻居的扩散范围。它不是训练指标，但能关闭“support anchor 完全凭猜测”的攻击。

## 10.2 Pilot 判定方式

不使用任意绝对 mAP 阈值。每个训练型 gate 使用相同：

* 数据 manifest；
* 初始化；
* update 数；
* scheduler；
* augmentation RNG；
* checkpoint selection rule。

先使用预注册的短 pilot；只有满足以下条件才进入正式 full run：

* 机制合同通过；
* 相对 matched control 的 `ΔAvg` 与 `ΔmAP@0.7` 不能同时出现 95% paired-video bootstrap 上界 `<0`；
* short-action class-aware recall 不能出现明确反向；
* 参数量、候选数、训练更新数保持同构。

## 10.3 正式主实验

### THUMOS14

三种主系统：

1. selected-coordinate capacity-matched control；
2. physical-coordinate capacity-matched control；
3. SM-PTAF。

至少三 seeds。统计：

* 五个 tIoU mAP 与 Avg；
* class-aware/class-agnostic recall；
* top-100 recall；
* `<1s`、duration quartiles；
* boundary error，但明确标注 conditional；
* paired video bootstrap；
* 跨 seed hierarchical bootstrap。

### 第二数据集

推荐 ActivityNet-v1.3 raw-video 子协议，原因是持续时间和视频长度分布与 THUMOS 显著不同。第二数据集不是为了追 SOTA，而是验证：

* 不同 duration；
* 不同 FPS；
* 较长空洞；
* candidate count 随 K 的行为。

### Sampling families

全部固定 K、无学习、无 GT 子采样：

* deterministic random；
* stratified jitter；
* periodic drop；
* burst-hole；
* 混合密度。

每个系统必须读取同一 selected-index manifest。

### 成本账本

必须登记：

* decoded frames；
* native tubelet count；
* projection/head MACs；
* total/trainable params；
* candidates/window；
* peak VRAM；
* p50/p95 inference latency；
* GPU-hours；
* NMS 前后 proposals；
* support-mass operator overhead。

## 10.4 最终条件

### GO

* feature provenance、candidate parity、assignment、no-leak、translation contracts 全通过；
* projection+head 参数差异 ≤1%；
* 三 seeds 下 SM-PTAF 相对 physical-coordinate control 的 paired CI 不显示明确劣化；
* 至少在高 tIoU、短动作或 gap robustness 中有一项显著、可重复改善；
* 改善不能由候选数或额外监督解释。

### HOLD

* 机制合同通过，但主要指标 CI 跨零；
* 或 Avg 改善而 @0.7/short action 明确下降；
* 或只有单一 sampling family 有效。

此时只允许一个针对性 gate，不允许扩充模块。

### STOP 某组件

某组件在相同 seeds 下同时满足：

* `ΔAvg` 的 95% CI 上界 `<0`；
* `ΔmAP@0.7` 或 short R@0.7 的 CI 上界 `<0`；
* 且其预期机制指标没有改善。

则删除该组件，不继续调 weight。

### PIVOT 整个方向

只有在下列全部成立时才应 PIVOT：

1. provenance、容量、候选、assignment 已完全同构；
2. THUMOS 至少三 seeds；
3. 第二数据集完成；
4. 多种固定 sampling family；
5. 机制合同全部通过；
6. SM-PTAF 同时被 selected-coordinate 和 physical-coordinate controls 支配；
7. `ΔAvg` 和 `Δ@0.7` 的 hierarchical-bootstrap 95% CI 均严格低于 0。

当前证据远未达到该 PIVOT 条件。

---

# 11. 新颖性、相关工作边界和最强审稿攻击

## 11.1 “不规则采样下在物理时间上检测”本身不够新

不够。

TE-TAD 已经强调 actual timeline coordinate 和随视频时长调整 query；RCL 也使用连续时间坐标构造连续 anchor。因此“把坐标换成秒”不能构成核心贡献。([arXiv][1]) ([arXiv][2])

真正不可替代的贡献必须是：

> **在不插值、不填补缺失区间的情况下，把不规则 raw observations 的显式支持集合提升为物理时间候选金字塔，并保持 measure provenance、候选公平性和秒坐标监督。**

## 11.2 与 mTAN 的区别

mTAN：

* 面向不规则多变量时间序列；
* 学习 continuous-time embeddings；
* 通过 attention 得到固定长度 representation；
* 主要用于 interpolation 和 classification。([arXiv][3])

SM-PTAF：

* 不重建或插值 dense signal；
* 输出多个动作 interval 和 class；
* support overlap 是显式 base measure；
* 候选、assignment、decode、NMS、evaluation 全在秒坐标；
* 具有多尺度 detection pyramid 和 per-location regression。

但如果最终实现只是“把 observation attention 到均匀 grid，再跑 ActionFormer”，审稿人仍会称其为 mTAN-style regridding。因此必须保留：

* native tubelet support atoms；
* no-gap-fill；
* explicit mass residual；
* gap query；
* provenance tests。

## 11.3 与 TE-TAD 的区别

TE-TAD 解决：

* normalized coordinate 不适配长短视频；
* actual timeline coordinate；
* adaptive query count。([arXiv][1])

SM-PTAF 解决：

* raw observations 本身不规则；
* observation support 是离散、带空洞的 measure；
* feature token 与物理支持的 provenance；
* 不通过 learned selector 或 dense resampling掩盖缺失；
* K384 主实验保持固定 candidate parity。

因此不能把“actual seconds”本身写成新颖点。

## 11.4 与 FrameDrop/TRC 的区别

Temporal Robustness Benchmark 建立 THUMOS14-C/ActivityNet-C，并提出 FrameDrop augmentation 与 Temporal-Robust Consistency 来提高 corruption robustness。它主要是 robustness benchmark 和训练正则，不是一个原生 support-aware detector。([arXiv][4])

SM-PTAF 的差异是：

* observation geometry 是模型输入合同；
* 缺失位置不被当作普通 corruption augmentation；
* detector 不假设 dense equal-spacing feature；
* proposal geometry 直接由真实 supports 与秒坐标定义。

论文中仍必须把 FrameDrop/TRC 作为强 baseline，而不是只与普通 ActionFormer 比。

## 11.5 与 LiquidTAD 的区别

LiquidTAD 是 2026 年 arXiv 预印本，核心是并行化 liquid/CfC-style temporal dynamics、线性复杂度和 learned time constants，重点在效率和连续动力学，不是显式 irregular support measure。([arXiv][5])

SM-PTAF 不应声称“首个 continuous-time TAD”，而应限定为：

* explicit observation supports；
* no-imputation set-to-interval detection；
* measure-preserving candidate construction。

## 11.6 是否只是 timestamp embedding 或更复杂 resampling

当前 PhysTime 1.0 很容易被攻击成：

> timestamp MLP + attention pooling + 一个更小的 head。

推荐的 SM-PTAF 只有满足以下条件才不是普通 timestamp embedding：

* 删除 raw center embedding；
* relative time 直接改变 attention operator；
* support overlap 是归一化 base measure；
* feature provenance 与 tubelet 生成过程一致；
* 不生成 dense interpolated feature sequence；
* zero-coverage query 保留但显式标注 uncertainty；
* assignment/decode 全在秒。

物理 query cells 是**预测候选域**，不是对 feature signal 的插值重建。

## 11.7 哪个实验能证明收益来自 detector

最关键实验不是最终 SM-PTAF 对现有 PhysTime，而是：

```text
同一 raw frames
同一 selected-index manifest
同一 VideoMAE/TIA
同一参数量
同一候选数
同一 assignment
同一 head
同一训练更新
```

仅比较：

1. selected-coordinate AF；
2. physical-coordinate AF；
3. physical-coordinate AF + support-measure lift。

其中 1→2 证明坐标贡献，2→3 证明 detector operator 贡献。

## 11.8 论文定位

最安全的定位是：

> **一个面向不规则 raw-video observations 的新 TAD detector，加一个严格限定的 irregular-observation evaluation protocol。**

不应声称：

* 通用 temporal robustness；
* continuous-time modeling 首创；
* SOTA；
* 对任意 sampling 的泛化；
* backbone 原生物理时间建模。

ActionFormer 已经用多尺度 local self-attention 和轻量 per-moment decoder建立了强检测骨架；AdaTAD 的 TIA 又会聚合相邻 temporal features。SM-PTAF 的论文必须明确，其主创新发生在 observation-support 到 physical candidate pyramid 的检测器接口，而不是重新发明这些模块。([arXiv][6]) ([arXiv][7])

## 11.9 最强审稿攻击

最强攻击会是：

> “这只是 mTAN 风格的 regridding 加 ActionFormer；提升来自恢复 756 个候选和 20M 参数。VideoMAE/TIA 仍按 selected rank 混合特征，所以所谓 physical-time-native 只发生在 backbone 之后。”

关闭方式：

1. 删除 192→384 feature interpolation；
2. 明确 tubelet support atoms；
3. candidate/parameter matched controls；
4. coordinate-only control；
5. mass-residual on/off；
6. bounded content on/off；
7. TIA rank mixing 的 Jacobian/provenance 诊断；
8. 论文主张限定为 detector，而非 full-stack physical-time backbone；
9. 若后续修改 TIA，必须作为单独第二阶段实验，不能与 detector 主贡献捆绑。

---

# 12. 最终论文主张

## 一句不过度声称的贡献表述

> **我们提出 SM-PTAF，一种无需特征插值、容量与候选数对齐的物理时间动作检测器：它将稀疏 RGB 观测的显式支持集合视为时间测度，以测度守恒的 set-to-pyramid 运算建立多尺度动作候选，并始终在秒坐标中完成监督、解码与评测。**

这句话目前只能作为**目标贡献表述**，不能作为已经被实验支持的摘要结论。

## 距离 paper-ready 仍缺少

1. 关闭 native tubelet feature–support provenance P0；
2. 完成 capacity/candidate/assignment matched coordinate-only control；
3. 实现并验证 SM-PTAF；
4. 至少三 seeds；
5. 第二数据集；
6. 多种固定无 GT sampling families；
7. paired/hierarchical confidence intervals；
8. 参数、候选、延时、显存、GPU-hours 完整成本账本；
9. 可访问的 checkpoint、prediction、manifest 和 artifact hash bundle；
10. 明确证明收益不是来自候选恢复、容量增加或额外 endpoint 监督。

**当前可写入论文的结论只有：PhysTime 1.0 在单个 THUMOS14/K384/单 seed matched run 中失败；由于 detector capacity、候选、assignment 和 feature provenance 混杂，这个负结果尚未裁决 physical-time-native TAD 的科学假设。**

[1]: https://arxiv.org/abs/2404.02405 "https://arxiv.org/abs/2404.02405"
[2]: https://arxiv.org/abs/2203.07112 "https://arxiv.org/abs/2203.07112"
[3]: https://arxiv.org/abs/2101.10318 "https://arxiv.org/abs/2101.10318"
[4]: https://arxiv.org/abs/2403.20254 "https://arxiv.org/abs/2403.20254"
[5]: https://arxiv.org/abs/2604.18274 "https://arxiv.org/abs/2604.18274"
[6]: https://arxiv.org/abs/2202.07925 "https://arxiv.org/abs/2202.07925"
[7]: https://arxiv.org/abs/2311.17241 "https://arxiv.org/abs/2311.17241"
