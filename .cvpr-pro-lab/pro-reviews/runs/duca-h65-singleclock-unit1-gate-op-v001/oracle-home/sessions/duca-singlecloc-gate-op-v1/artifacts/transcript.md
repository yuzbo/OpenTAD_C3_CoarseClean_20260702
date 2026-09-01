# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8c0f1c-cdc0-83ea-915e-2af71e390288

## Prompt

请完整阅读已接受 Pro 原文、当前终结器和统计代码。本轮只操作化 Unit-1 终态门，不重开路线。Nonce: DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_H65_SINGLECLOCK_UNIT1_GATE_OPERATIONALIZATION-v001.md
Lines: 1-47
```md
 1 | # DUCA H65 First-Mixing SingleClock：Unit-1 终态门操作化裁决
 2 | 
 3 | Nonce: `DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824`
 4 | 
 5 | 你是 DUCA 项目的独立 Scientific First-Author Agent 与统计审稿人。本轮只解决已经接受的 `DUCA-H65C-SINGLECLOCK-DYNAMIC-v002` Unit-1 终态门如何被代码无歧义执行；不得重开路线、提高门槛、要求新训练、引入 Query/dynamic-K，或把成本变成未写入原合同的前置 kill gate。
 6 | 
 7 | ## 已接受、不得修改的 Unit-1 合同
 8 | 
 9 | - H65 的固定 K384、语义间接非均匀逐帧选择、选中 RGB、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split 和官方 evaluator 保持不变。
10 | - 唯一表示变化是在第一个 temporal attention mixing 中加入零初始化 SingleClock relative physical-time residual。
11 | - H65 replay 身份先验证 selected indices、gathered RGB、VideoMAE input tensor、raw proposals/scores 和 official evaluator JSON；uniform positions 必须 bit-identical。
12 | - 相对 H65 replay 的主门原文为：`ΔAvg >= -0.20 pp`、`Δ@0.6 >= -0.20 pp`、`Δ@0.7 >= -0.20 pp`；高 gap-CV / 高 boundary-density 窗口中 paired boundary error 不恶化。任一主指标低于 margin 或 uniform identity 失败即 `KILL_SINGLECLOCK_REPRESENTATION`。
13 | - 当前训练和推理已完成/正在封存；不允许为补诊断重训。
14 | 
15 | ## 当前实现问题
16 | 
17 | 现有 `finalize_duca_h65_singleclock_terminal.py` 使用了旧的正增益门：ON-vs-gate-zero Avg-mAP 至少 `+0.50 pp`，并把 ON-vs-H65 OFF `+0.50 pp`、coadaptation CI 和 cost 硬门混入 `main_pass`。这与已接受 Unit-1 的 H65 replay `-0.20 pp` 非劣门不一致。
18 | 
19 | 现有 `analyze_duca_h65_singleclock_strata.py` 已能：
20 | 
21 | - 只用 training population 冻结 per-video distortion q25/q50/q75 与 short-action duration q25；
22 | - 在 validation 上对 low/high distortion 和 short actions 做 10,000 次 whole-video cluster paired official-mAP bootstrap；
23 | - 计算 high-low distortion interaction。
24 | 
25 | 但它没有 boundary-density 定义，也没有 proposal-vs-GT boundary error。仓库中存在两个候选 boundary estimator：
26 | 
27 | 1. 每个 GT 取同类最高 IoU proposal，报告 start/end MAE seconds 与归一化误差；
28 | 2. 按 score 排序做同类 IoU>=0.5 的一对一匹配，报告 matched recall 和 start/end MAE seconds。
29 | 
30 | 二者均不是官方 THUMOS evaluator 的原生指标，也没有既有 Unit-1 freeze。已有 10,000 次 PCG64 whole-video cluster paired bootstrap 基础设施可复用。
31 | 
32 | ## 你必须冻结的唯一可执行规格
33 | 
34 | 请给出唯一 `REVISE_GATE_IMPLEMENTATION / STOP_UNDERDEFINED_GATE`，并明确：
35 | 
36 | 1. Unit-1 主对照究竟是 SingleClock ON EMA vs H65 OFF/replay EMA，还是 ON vs same-checkpoint gate-zero；final 与 gate-zero 应各自承担 identity、机制诊断还是主门角色。
37 | 2. 三个 `-0.20 pp` 主指标使用 point estimate 还是 paired-bootstrap CI；若用 CI，请给 exact bound。不得事后提高原合同。
38 | 3. `boundary error` 的精确定义：proposal-GT matching、类别约束、IoU cutoff、score/top-k/NMS 状态、seconds 还是 duration-normalized、unmatched GT 如何计入、每视频如何聚合。
39 | 4. `gap-CV` 与 `boundary-density` 的精确定义；threshold 必须来自 training population。请给 exact q25/q75 或其他 cutpoint、window/video 粒度、短视频/padding处理。
40 | 5. “不恶化”的精确定义：point delta `<=0`、绝对容忍 margin、还是10k paired video-cluster bootstrap CI；给 exact sign、bound 与 resampling unit。说明高 gap-CV 和高 boundary-density 是 AND、OR 还是分别过门。
41 | 6. 现有训练已经结束，若当前 prediction/identity/GT annotations 足以离线计算，请要求只做离线统计；若无法合法计算，请说明 Unit-1 是否应只按主指标+identity裁决，而把未操作化条款降为诊断，禁止要求重训。
42 | 7. cost 在 Unit-1 中是报告项、后置准入项还是硬 kill gate。原文没有 Unit-1 cost margin，不得凭空发明。
43 | 8. `paper_claim_admissible` 在单 seed Unit-1 是否必须保持 false；Unit-2 Query residual 何时才允许进入 Builder。
44 | 9. 给出终结器应输出的最小字段、唯一 PASS/KILL token、以及可由 focused unit tests 覆盖的例子。
45 | 10. 输出 `next_owner / next_action / dependency / expected_return_at`。
46 | 
47 | 要求：保留原 Unit-1 科学意图，选择最小、可复现、不会利用 validation 调阈值的定义。不要把本轮变成新方法或新实验矩阵。
```

### File: C:/Users/skywalker/.codex/oracle/duca-h65-truetime-pro-20260822-v002/final.md
Lines: 1-745
````md
  1 | # DUCA H65 / TrueTime / UVT / Query-Bridge 最终对抗性科学裁决
  2 | 
  3 | **Project**：`g-p-6a796fef9a00819194024cf1de3bd697`
  4 | **Routing nonce**：`DUCA-H65-TRUETIME-UVT-FOVEA-PRO-v002-20260822`
  5 | **v001 状态**：传输失败、未接触 Project、未提交科学问题；不构成科学轮次，本裁决未复用其内容。
  6 | 
  7 | ---
  8 | 
  9 | ## 1. 唯一裁决：`REVISE`
 10 | 
 11 | 当前方法不能按原样继续。必须停止沿用“现有 TrueTime 全套表示改造 + 新 selector + physical-grid detector + 20/20/20 curriculum”的路线。
 12 | 
 13 | 冻结的唯一主路线是：
 14 | 
 15 | > **`DUCA-H65C-SINGLECLOCK-DYNAMIC-v002`：H65 兼容语义采样、单一物理时间度量、Query 仅作语义残差、最终由 dynamic outer-K 决定真实重计算量。**
 16 | 
 17 | 该路线不是退回旧 H65，而是：
 18 | 
 19 | 1. 先把 H65 中尚未被公平拆开的**语义—采样—优化兼容合同**完整恢复；
 20 | 2. 用最小的“单时钟残差”替换当前 TrueTime 的多部件表示改造；
 21 | 3. Query/Cycle 只能修正 scout 的语义表示，不能直接生成索引、K、proposal 或 NMS score；
 22 | 4. 只有前两门通过后，才接入最终论文核心 dynamic outer-K；
 23 | 5. 固定 K=384 只用于因果归因、控制和故障回退，不得成为最终方法。
 24 | 
 25 | 这与 DUCA 已冻结的“dynamic budget 必须是论文核心、fixed K 只能作对照”的边界一致。 
 26 | 
 27 | 我不选择 `CONTINUE`，因为当前实现混杂过多；不选择 `PIVOT`，因为 H65 和 TrueTime 的高-IoU相对信号仍表明该科学问题有可证伪价值；不选择 `STOP`，因为存在一个不超过三个新增训练单元、能够直接解释 65.385724 的最小路线。
 28 | 
 29 | ---
 30 | 
 31 | ## 2. 代码核验边界
 32 | 
 33 | ### 2.1 实际核验过的 GitHub 代码
 34 | 
 35 | **H65，提交 `42dba3f...`**
 36 | 
 37 | 我实际阅读了：
 38 | 
 39 | * Stage-1 exact-uniform 配置；
 40 | * Stage-2 joint curriculum；
 41 | * cls/reg contribution distillation 与 full-ASFormer adaptation 配置；
 42 | * sampling-rate 累积密度采样配置；
 43 | * `duca_online_frame_selector.py`；
 44 | * `acquisition.py`、`structured_selection.py`；
 45 | * `two_stage.py`；
 46 | * Stage-2 recovery launcher。
 47 | 
 48 | 代码确认：Stage-2 从 Stage-1 epoch-29 EMA 初始化，重置优化器、调度器和 AMP 状态，再训练完整 60 epoch；policy、detector gradient、contribution distillation 和 ASFormer adaptation 分别渐进开启。 H65 的 utility 明确同时使用分类与回归 contribution，并让梯度进入完整 ASFormer encoder。
 49 | 
 50 | sampling-rate 路径使用 per-frame retention rate、coverage floor、平滑、确定性累积采样、50% uniform companion 和 `density_transport_st`。
 51 | 
 52 | **RankPack / TrueTime，提交 `11126684...` 与 `ca83be0f...`**
 53 | 
 54 | 我实际阅读了：
 55 | 
 56 | * 共同 20/20/20 curriculum；
 57 | * RankPack 与 TrueTime 配置；
 58 | * protected physical fixed-K 基配置；
 59 | * `physical_time.py`；
 60 | * `backbone_wrapper.py`；
 61 | * `vit_adapter.py`；
 62 | * 配对实现提交差异。
 63 | 
 64 | 代码确认：两臂共享的已经不是 H65 detector/selector 合同，而是 `DucaProtectedE2EFrameSelector`、physical-grid ActionFormer assignment/regression、无 uniform companion 的新体系。
 65 | 
 66 | **UVT，提交 `df544c78...` / 结果头 `59f27d59...`**
 67 | 
 68 | 我实际阅读了配置、Value head、主 selector 中 V(t) 的接线和结果页。结果页确实记录了 `off=57.35`、`geo=55.93`、`geo_ema=55.92`，并明确承认 V(t) 同时改变了 selection score 和 dynamic-K evidence。
 69 | 
 70 | **Fovea / Query-Bridge，提交 `4ae50671...` / 结果头 `46c71424...`**
 71 | 
 72 | 我实际阅读了配置、Query-Bridge、Fovea selector、Gumbel/greedy-MMR sampler、detached cycle feedback 和结果页。结果页记录了 `query_cycle=54.67`，但也承认是单种子、缺少同提交强基线和完整成本的 development observation。
 73 | 
 74 | ### 2.2 没有被我独立重放的证据
 75 | 
 76 | 下列内容作为本轮所给材料中的真实实验收据接受，但没有被我重新执行或从集群原始文件独立复核：
 77 | 
 78 | * H65 Job `1191957` / `1193610` 的原始 Slurm 日志、checkpoint 字节、prediction JSON；
 79 | * `65.385724` 的独立 evaluator 重算；
 80 | * RankPack / TrueTime Job `1248822/1248823` 的原始 checkpoint 与 evaluator JSON；
 81 | * 连续 cliplet `49.89/47.24` 的原始执行账本；
 82 | * 所有 GPU 端到端性能与能耗收据。
 83 | 
 84 | 因此，本报告可以裁决代码合同和因果可识别性，但不会把上述数字升级为我独立复现的结果。
 85 | 
 86 | ### 2.3 一个额外发现
 87 | 
 88 | H65 的 `two_stage.py` 确实在 NMS 前进行 q→physical-time remap，但当前顺序是：先按 score threshold/top-k 筛选，再做坐标映射，然后 NMS。
 89 | 
 90 | 由于 threshold/top-k 本身只看 score，这未必改变旧指标；但它不满足本轮更严格的合同——物理坐标必须在 **threshold、top-k、IoU、NMS** 之前存在。因此新路线必须把坐标适配器移动到 post-processing 的最前端。
 91 | 
 92 | ---
 93 | 
 94 | ## 3. H65 的因果解释
 95 | 
 96 | ### 3.1 最可能因素排序
 97 | 
 98 | | 因素                                                      | 科学判断                 | 可能因果链                                                                          | 最便宜 falsifier                                                                                            | 处置                                |
 99 | | ------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | --------------------------------- |
100 | | **30 epoch Stage-1 EMA warm start**                     | **最大混杂之一**           | detector、VideoMAE adapter、ASFormer 和 uniform K384 表示先进入稳定盆地，随后才开放非均匀策略         | 对同一 Stage-1 EMA 做只读权重分区审计；随后用 Unit 1 恢复完整 H65 合同，只改变时间表示                                                 | 作为共同初始化保留；不能称作采样贡献                |
101 | | **额外 60 epoch joint optimization**                      | **最大混杂之一**           | 相对 30/60 epoch 基线，多出完整 detector exposure；低 LR 二次优化本身即可提升 mAP                   | 同一 Stage-1 EMA、同一 60-epoch schedule 下比较 H65 RankPack replay 与 SingleClock                                | 保留为因果恢复合同；训练成本必须完整披露              |
102 | | **ASFormer action/transition hidden adaptation**        | **高概率真实贡献**          | scout 不只是 actionness，而是学习动作状态变化、边界附近隐藏表示，并在 Stage-2 继续适配                       | 冻结 checkpoint，对相同窗口分别回放 adapted hidden、冻结 hidden、actionness-only utility；不训练 detector                    | 与 contribution 一起保留               |
103 | | **classification/regression contribution distillation** | **高概率真实贡献**          | rate utility 受到分类和边界回归实际 detector contribution 的约束，而不是泛化 saliency              | 对同一 checkpoint 和同一 frame set 做 `none/cls/reg/both` policy replay，比较选中集合、coverage 和 detector loss         | 保留 `both`；不得拆成论文独立贡献              |
104 | | **density/rate systematic sampling**                    | **中等概率贡献**           | 将连续 utility 转成覆盖全时轴的 retention density，避免纯 top-k 聚集                            | 固定 H65 utility，进行 uniform、top-k 和 systematic 的无训练 counterfactual replay                                  | 作为唯一 inner selector 保留            |
105 | | **50% uniform companion**                               | **较高概率稳定器**          | 防止 detector 只适应不断漂移的稀疏分布，维持 VideoMAE 和 ActionFormer 的 uniform manifold         | 利用同一模型分别统计 companion/learned 行的特征漂移和 loss；不需要完整重训                                                        | 保留                                |
106 | | **`density_transport_st`**                              | **贡献未证实**            | 可能让 detector loss 影响 utility；也可能只是与真实 hard swap 不一致的 surrogate                 | 在 training-side CAL 上做 1%、5%、10% hard swap，检验 surrogate 与真实 loss delta 的 Spearman、方向率和 top-decile regret | 默认关闭 detector→selector 梯度；通过门后才启用 |
107 | | **selected-rank VideoMAE 预训练兼容性**                       | **很可能解释 H65 高而新路线低** | 虽然物理间隔被隐藏，但每个 16-frame pack 仍符合 VideoMAE 的局部有序输入先验，没有改变 Conv3D 和 position path | Unit 1：相同 H65 semantic contract、相同 frame set，仅把时间解释改成 SingleClock                                        | 必须作为 RankPack replay 控制           |
108 | | **proposal physical-time remap**                        | **必要正确性，不是主要性能创新**   | 避免 selected rank 被当作真实秒数，主要保护 boundary 和高 tIoU                                 | 对同一 raw proposals 比较 q-axis NMS 与 physical-axis NMS；无需训练                                                 | 强制保留，但禁止包装成核心创新                   |
109 | 
110 | H65 的 Stage-2 并不是简单地“接着多训”：它从完整 Stage-1 EMA 初始化，随后在低学习率下同步开启 policy、contribution、detector bridge 和 ASFormer adaptation；其 launcher 还明确绑定 terminal epoch-59 EMA，而不是中间 best checkpoint。
111 | 
112 | ### 3.2 哪些应共同保留，哪些必须隔离
113 | 
114 | **共同保留为一个不可再拆的 H65 compatibility contract：**
115 | 
116 | * ASFormer action/transition hidden；
117 | * cls/reg contribution distillation；
118 | * sampling-rate systematic decoder；
119 | * 50% uniform companion；
120 | * Stage-1 EMA → Stage-2 低 LR 渐进开放；
121 | * selected-axis ActionFormer detector；
122 | * q→t pre-NMS transport。
123 | 
124 | **必须单独隔离：**
125 | 
126 | * SingleClock 时间机制；
127 | * Query semantic residual；
128 | * dynamic outer-K；
129 | * detector-gradient `density_transport_st`。
130 | 
131 | 也就是说，下一步不再训练“ASFormer-only”“distill-only”“companion-only”等完整长臂。它们的单项分析优先使用冻结 checkpoint、policy replay、hard-swap diagnostics 和相同权重的 counterfactual forward。
132 | 
133 | ---
134 | 
135 | ## 4. 当前 3–4 pp 下降的解释
136 | 
137 | ### 4.1 首要原因：H65 的语义—课程—表示兼容合同被整体替换
138 | 
139 | 这是**代码事实支持下的科学推断**。
140 | 
141 | 当前 RankPack/TrueTime 共同基线具有以下变化：
142 | 
143 | * H65 的 Stage-1 30-epoch full-model EMA warm start 被换成 20/20/20 curriculum；
144 | * H65 的 contribution-rate utility 被换成新的 protected selector；
145 | * 50% uniform companion 变为 0；
146 | * detector-gradient ownership、selector supervision 和 policy homotopy 全部变化；
147 | * selected-axis detector 被换成 physical-grid assignment/regression；
148 | * optimizer group 与可训练 backbone 参数发生变化。
149 | 
150 | 因此，H65 与 RankPack/TrueTime 之间的约 3–4 pp 差距，**首要解释是整套训练与语义合同丢失，而不是物理时间本身造成的**。
151 | 
152 | ### 4.2 次要原因一：detector geometry 同时改变
153 | 
154 | RankPack 与 TrueTime 虽然彼此共享 physical-grid detector，因而两者之间有一定配对意义；但它们与 H65 不再共享同一个 ActionFormer assignment、regression coordinate contract 和 detector geometry。
155 | 
156 | 这使得：
157 | 
158 | > H65 vs TrueTime 不是“旧时间表示 vs 新时间表示”，而是 selector、训练、backbone 时间表示和 detector geometry 的联合变化。
159 | 
160 | 当前下降中有多少来自 head/assignment，现有结果无法识别。
161 | 
162 | ### 4.3 次要原因二：当前 TrueTime 不是“只改第一次时间混合”
163 | 
164 | 该描述不成立。
165 | 
166 | 当前 TrueTime 同时做了：
167 | 
168 | 1. 重写 Conv3D 两个 temporal kernel slice 的组合；
169 | 2. 加入 gap-dependent residual；
170 | 3. 加入 midpoint/log-gap coordinate MLP；
171 | 4. 按物理间隔修改 adapter 的 depthwise temporal convolution；
172 | 5. 传播物理 mask、padding 和 source positions；
173 | 6. 改变 optimizer 中可训练的 physical-time 参数组。
174 | 
175 | 而且标准 VideoMAE `pos_embed` 仍然执行 `x = x + pos_embed`。
176 | 
177 | 因此当前模型同时存在：
178 | 
179 | * selected-rank absolute positional prior；
180 | * physical midpoint absolute embedding；
181 | * physical gap residual；
182 | * gap-aware adapter convolution。
183 | 
184 | 这不是一个“单一真实时间”机制，而是多个时间解释叠加。它可能造成预训练分布漂移，也让任何正负结果无法归因。
185 | 
186 | ### 4.4 次要原因三：selector 被 detector 过早扰动
187 | 
188 | H65 的 detector bridge 是在已有 Stage-1 EMA 上延迟、低权重开启的；当前新路线则使用另一套 gradient ownership 和 homotopy。selector 尚未形成稳定的 action/boundary semantics 时就被 detector loss 改写，是合理的下降来源。
189 | 
190 | ### 4.5 已被证伪或明显削弱的解释
191 | 
192 | **“物理时间本身必然有害”已被削弱。**
193 | 按材料中的同快照单种子结果，TrueTime 比 RankPack 高 `+0.6208 pp`，@0.6 高 `+1.6885 pp`。这至少排除了“只要重模型看到物理时间就一定下降”的强说法。它不能证明现有 TrueTime 设计正确，但说明 3–4 pp 总下降不能主要归咎于物理时间。
194 | 
195 | **“连续 cliplet 自然修复时间语义”已被证伪。**
196 | 材料中的 `49.89/47.24` 表明，仅将输入改成连续片段不是解决方案。
197 | 
198 | **“archive/hash 冲突导致模型精度下降”不成立。**
199 | 若 epoch-59 EMA checkpoint 已完成并被只读 evaluator 正常加载，训练后的归档冲突影响证据封存，不会回溯改变已完成的权重和预测。现有 recovery launcher 也明确将 checkpoint、evaluation 与后续质量收据分开。
200 | 
201 | ---
202 | 
203 | ## 5. 唯一冻结路线
204 | 
205 | ### 5.1 科学问题
206 | 
207 | > 非均匀物理采样改变的不只是“看哪些帧”，也改变了相邻视觉证据之间的实际时间度量。能否在不破坏 VideoMAE 预训练先验、不修改 ActionFormer detector 的情况下，以一个单一时间尺度解释这些间隔，并利用 boundary-sensitive semantic utility 动态分配每个窗口的重计算预算？
208 | 
209 | ### 5.2 核心机制：Canonical-Time Residual Single Clock
210 | 
211 | 不采用当前 TrueTime 的 absolute coordinate MLP、gap-conditioned tubelet residual 和 gap-scaled adapter convolution。
212 | 
213 | 冻结以下单时钟定义。
214 | 
215 | 对 selected rank (q_i)：
216 | 
217 | * (u_i)：canonical exact-uniform K384 对应的物理时间；
218 | * (t_i)：实际选中帧的物理时间；
219 | * canonical elapsed time 由原 VideoMAE rank/position prior 表示；
220 | * 物理时间只通过相对残差进入：
221 | 
222 | [
223 | r_{ij}
224 | ======
225 | 
226 | \log
227 | \frac{|t_i-t_j|+\epsilon}
228 | {|u_i-u_j|+\epsilon}.
229 | ]
230 | 
231 | 在 VideoMAE **第一处 temporal attention mixing** 中加入零初始化相对 bias：
232 | 
233 | [
234 | B_{ij}=\alpha,h(r_{ij}),\qquad
235 | \alpha_0=0,\qquad h(0)=0.
236 | ]
237 | 
238 | 冻结约束：
239 | 
240 | * 原始 Conv3D tubelet projection 完全不改；
241 | * 原始 VideoMAE `pos_embed` 保留并冻结为 canonical uniform prior；
242 | * 不再加入第二个 absolute physical coordinate embedding；
243 | * 不改后续 adapter convolution；
244 | * 当实际采样等于 canonical uniform 时，`r=0`，SingleClock 路径必须与 H65 RankPack bit-identical；
245 | * 物理时间因此不是第二套绝对坐标，而是对 canonical clock 的唯一残差修正。
246 | 
247 | **明确回答本轮问题：**
248 | 
249 | * 应当把 gap 放在 relative-time / attention-bias 路径；
250 | * 不应粗暴删除整个 pretrained rank `pos_embed`，否则会制造新的预训练分布漂移；
251 | * 应删除的是当前独立的 physical absolute coordinate MLP 和其他重复时间机制；
252 | * rank prior 只充当 canonical prior，实际时间由唯一 residual 完成校正。
253 | 
254 | ### 5.3 Selector 与 detector
255 | 
256 | 固定保留 H65：
257 | 
258 | * ASFormer action/transition hidden；
259 | * cls/reg contribution distillation；
260 | * systematic rate sampler；
261 | * 50% uniform companion；
262 | * K384 fixed attribution gate。
263 | 
264 | ActionFormer 继续在 selected axis (q) 上训练与预测；不使用 current physical-grid head。
265 | 
266 | 训练时：
267 | 
268 | [
269 | t_{\text{GT}}\rightarrow q_{\text{GT}}
270 | ]
271 | 
272 | 只用于 detector label mapping。
273 | 
274 | 推理时：
275 | 
276 | [
277 | q_{\text{proposal}}\rightarrow t_{\text{physical}}
278 | ]
279 | 
280 | 必须在 threshold、top-k、IoU、NMS、voting 和 serialization 前恰好执行一次。
281 | 
282 | ### 5.4 Query semantic residual
283 | 
284 | Query-Bridge 只能产生：
285 | 
286 | [
287 | h'(t)=h_{\text{H65}}(t)+\gamma r_Q(t),
288 | ]
289 | 
290 | 其中：
291 | 
292 | * (\gamma) 零初始化、有界；
293 | * `r_Q` 由 detached detector proposal/cycle target 训练；
294 | * 它只能进入 H65 已有的 semantic/contribution representation；
295 | * 它没有自己的 index head；
296 | * 它不能输出 K；
297 | * 它不能直接改 proposal、classification score 或 NMS；
298 | * 推理时无 teacher、无 cycle、无 cache。
299 | 
300 | Query-Bridge 当前代码本身已将 Q1 标记为 selector-internal，但 contribution 随后被用于 selection；新路线必须进一步收回其直接 selection ownership。
301 | 
302 | ### 5.5 最终 dynamic outer-K
303 | 
304 | 仅在固定 K 两门通过后开启。
305 | 
306 | 冻结 cliplet 预算：
307 | 
308 | [
309 | M\in{16,20,24,28,32},\qquad K=16M,
310 | ]
311 | 
312 | 即：
313 | 
314 | [
315 | K\in{256,320,384,448,512}.
316 | ]
317 | 
318 | 约束：
319 | 
320 | * 训练总体 realized mean (K=384)；
321 | * K 是 action/start/end/uncertainty、H65 contribution 和已通过的 Query residual 的确定性函数；
322 | * dynamic controller 使用训练人口 FIT/CAL 校准；
323 | * inference 禁止 GT、teacher、prediction cache、official validation statistics；
324 | * 每个样本 batch-independent；
325 | * requested、effective、unique、executed、padded K 全部记账；
326 | * dynamic K 必须真实改变 VideoMAE 调用的 clip 数，不得 Kmax padding 后冒充动态。
327 | 
328 | ### 5.6 真正有趣的科学预测
329 | 
330 | 该路线的核心不是“物理时间 + 语义 sampler”两个模块，而是一个可证伪的交互命题：
331 | 
332 | > **非均匀采样只有在重模型的时间度量仍位于预训练 canonical manifold、并用物理残差而非第二套绝对时钟修正时，才会改善边界定位。**
333 | 
334 | 它产生三个 uniform/fixed-K 不会自然产生的预测：
335 | 
336 | 1. 当实际 gap 接近 canonical uniform 时，SingleClock 与 RankPack 必须近乎相同；
337 | 2. SingleClock 的收益应随 gap heterogeneity、短动作密度和 boundary uncertainty 增大，并主要出现在 @0.6/@0.7；
338 | 3. 在 actionness 相同的窗口中，dynamic K 应更多分配给边界密集、短动作多或起止不确定的窗口，而不是简单分配给“动作更多”的窗口。
339 | 
340 | ---
341 | 
342 | ## 6. 最小改动面
343 | 
344 | | 文件 / symbol                                                                       | 允许的改动                                                                             | 明确禁止                                                                        |
345 | | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
346 | | `opentad/models/selectors/duca_online_frame_selector.py::DucaOnlineFrameSelector` | 恢复 H65 rate utility、uniform companion、metadata；增加 semantic-residual 接口            | 替换为 FoveaSampler、PCOTMRAS dynamic_B 或任意新 top-k                              |
347 | | `opentad/models/duca/acquisition.py`                                              | 保留 cls/reg contribution → retention-rate → systematic sampling；允许接收外部 effective K | Query/Value 直接提供 indices 或 K                                                |
348 | | `opentad/models/duca/structured_selection.py`                                     | 保留 deterministic cumulative exact-K sampler                                       | 引入新的连续 cliplet decoder、MMR、DPP、quota stack                                  |
349 | | `opentad/models/backbones/physical_time.py`                                       | 新增 `build_canonical_time_residual_bias(...)`                                      | 主路线调用 `PhysicalTimeTubeletEmbedding`、coordinate MLP、gap-scaled adapter conv |
350 | | `opentad/models/backbones/vit_adapter.py`                                         | 给第一处 temporal attention 增加可选 relative bias；uniform 时 bit-identical                | 第二个 absolute physical embedding；修改所有 block                                  |
351 | | `opentad/models/backbones/backbone_wrapper.py`                                    | 传递 physical positions、valid mask、requested/effective/executed K                   | 用复制最后一帧掩盖真实 heavy compute；把 padded K 当 executed K                           |
352 | | `opentad/models/detectors/two_stage.py`                                           | 把 q→t adapter 移到 per-sample post-processing 最前端；加入 coordinate-state guard         | physical-grid ActionFormer；映射后再映射；NMS 后映射                                   |
353 | | 新建 `duca_query_semantic_residual.py`                                              | 复用 query decoder 与 detached cycle target，输出 semantic residual                     | 引入 Fovea sampler、dynamic budget、proposal head                               |
354 | | `opentad/models/duca/dynamic_budget.py`                                           | 新增 FIT/CAL calibrated outer-M controller、平均成本约束、K-shuffle receipt                 | validation/test calibration；batch-relative K；固定阈值事后调参                       |
355 | | 三个正式 config                                                                       | Unit 1–3                                                                          | 新增 dense/uniform/random 长训练臂                                                |
356 | 
357 | ### 数据流
358 | 
359 | ```text
360 | dense low-resolution scout
361 | → ASFormer action / start / end / transition representation
362 | → optional Query semantic residual
363 | → existing cls/reg contribution utility
364 | → deterministic rate density
365 | → dynamic outer M or fixed M=24
366 | → exact-K physical frame indices
367 | → VideoMAE canonical prior + single relative physical-time residual
368 | → unchanged selected-axis ActionFormer
369 | → raw q proposals
370 | → exactly-once q→physical-t transport
371 | → threshold / top-k / IoU / NMS / evaluator
372 | ```
373 | 
374 | ### 梯度流
375 | 
376 | * action/start/end/transition losses更新 scout 与 ASFormer；
377 | * contribution distillation 更新 semantic utility；
378 | * Query cycle target 全部 detach；
379 | * Query residual 可更新 semantic encoder，但没有独立 selection loss；
380 | * detector→selector 的 `density_transport_st` 默认关闭；
381 | * 只有 training-side hard-swap alignment gate 通过后，才允许以原 H65 上限权重恢复；
382 | * outer-K 不从 official evaluator、validation mAP 或 test prediction 获得梯度。
383 | 
384 | ---
385 | 
386 | ## 7. 最多 3 个新增完整训练单元
387 | 
388 | 三个单元全部使用：
389 | 
390 | * THUMOS14 完整 official training split；
391 | * official validation evaluator；
392 | * N16R4 单 GPU allocation；
393 | * seed=`3407`；
394 | * 同一待冻结 descendant commit；
395 | * 相同数据顺序、augmentation RNG、detector、loss、optimizer、NMS 和 evaluator；
396 | * 每 5 epoch 保存完整可恢复状态；
397 | * 保留 latest-3、全部 5-epoch milestones、final、final-EMA；
398 | * **primary = terminal final-EMA**，同时报告 final；
399 | * 禁止按中间 validation 选择 checkpoint；
400 | * 60 epoch、6000 successful optimizer updates；
401 | * H65 Stage-1 EMA 只读复用，不重新训练 Stage-1。
402 | 
403 | ### Unit 1 — H65 compatibility × SingleClock
404 | 
405 | **新增完整训练：1 个**
406 | 
407 | 配置：
408 | 
409 | ```text
410 | duca_h65c_singleclock_k384_seed3407.py
411 | ```
412 | 
413 | 只读控制：
414 | 
415 | ```text
416 | duca_h65c_rankpack_replay_k384.py
417 | ```
418 | 
419 | 控制直接加载历史 H65 epoch-59 EMA，并在新 commit 的 compatibility-off 模式下重放；不进行新训练。
420 | 
421 | 唯一新训练臂：
422 | 
423 | * H65 Stage-1 EMA 初始化；
424 | * 完整 H65 Stage-2 semantic/contribution/companion/rate contract；
425 | * 相同 K384；
426 | * 唯一变化是 SingleClock relative-time bias。
427 | 
428 | **进入条件**
429 | 
430 | H65 RankPack replay 必须在以下五个边界与原提交一致：
431 | 
432 | 1. selected indices；
433 | 2. gathered RGB；
434 | 3. VideoMAE input tensor；
435 | 4. raw proposals/scores；
436 | 5. official evaluator JSON。
437 | 
438 | 若 replay 不一致，Unit 1 直接 `INVALID_IDENTITY`，不得解释 mAP。
439 | 
440 | **通过条件**
441 | 
442 | 相对重放 H65：
443 | 
444 | * `ΔAvg ≥ −0.20 pp`；
445 | * `Δ@0.6 ≥ −0.20 pp`；
446 | * `Δ@0.7 ≥ −0.20 pp`；
447 | * 高 gap-CV / 高 boundary-density 窗口中，SingleClock 的 paired boundary error 不恶化；
448 | * uniform positions 时 backbone 输出 bit-identical。
449 | 
450 | **Kill rule**
451 | 
452 | 任一主指标低于上述 non-inferiority margin，或 nominal-uniform 不能 bit-identical：
453 | 
454 | ```text
455 | KILL_SINGLECLOCK_REPRESENTATION
456 | ```
457 | 
458 | 这一个单元直接区分：
459 | 
460 | * 若恢复到 H65 水平：当前 3–4 pp 下降主要来自 H65 contract 丢失；
461 | * 若仍落到约 62：SingleClock/时间表示本身构成主要伤害；
462 | * 若 H65 replay 本身失败：历史结果身份不能被当前代码继承。
463 | 
464 | ### Unit 2 — Query semantic residual
465 | 
466 | **新增完整训练：1 个**
467 | 
468 | 配置：
469 | 
470 | ```text
471 | duca_h65c_singleclock_queryres_k384_seed3407.py
472 | ```
473 | 
474 | 与 Unit 1 唯一区别：
475 | 
476 | * 加入 zero-init Query semantic residual；
477 | * detached cycle supervision；
478 | * K 固定为 384；
479 | * sampler、indices、detector、SingleClock 全部不变。
480 | 
481 | **通过条件**
482 | 
483 | 相对 Unit 1：
484 | 
485 | * `ΔAvg ≥ +0.30 pp`；
486 | * `Δ@0.6 ≥ 0`；
487 | * `Δ@0.7 ≥ 0`；
488 | * video-cluster paired bootstrap 的 `LCB95(ΔAvg) > 0`；
489 | * 在控制 H65 actionness/transition utility 后，Query residual 对 boundary miss 或短动作 miss 仍有独立预测力；
490 | * selected-frame coverage 的改善不是由 K 改变造成，因为 K 固定。
491 | 
492 | **Kill rule**
493 | 
494 | 任一条件失败：
495 | 
496 | ```text
497 | KILL_QUERY_SEMANTIC_RESIDUAL
498 | ```
499 | 
500 | 并且根据本轮合同，**不允许继续 Unit 3**。不得把 Query 改成直接索引器或 budget head 进行补救。
501 | 
502 | ### Unit 3 — Dynamic outer-K
503 | 
504 | **仅在 Unit 1、Unit 2 均通过时执行。**
505 | 
506 | 配置：
507 | 
508 | ```text
509 | duca_h65c_singleclock_queryres_dynamicM_seed3407.py
510 | ```
511 | 
512 | 预算：
513 | 
514 | ```text
515 | M = {16, 20, 24, 28, 32}
516 | K = {256, 320, 384, 448, 512}
517 | target mean K = 384
518 | ```
519 | 
520 | 同一个训练 checkpoint 必须支持以下无新增训练的 evaluator modes：
521 | 
522 | 1. learned dynamic M；
523 | 2. forced fixed M=24；
524 | 3. identical learned M sequence + exact-uniform positions；
525 | 4. K-shuffle；
526 | 5. actionness-only dynamic。
527 | 
528 | **K-shuffle 冻结定义**
529 | 
530 | * canonical row order：`(video_name, window_start_frame, window_end_frame, occurrence_index)`；
531 | * stratum：valid-length / padding-equivalence class；
532 | * 不按 actionness、boundary 或 difficulty 分层；
533 | * seed：
534 | 
535 | ```text
536 | SHA256(
537 |   "DUCA_H65C_KSHUFFLE_V1\n"
538 |   + "DUCA-H65-TRUETIME-UVT-FOVEA-PRO-v002-20260822"
539 | )
540 | ```
541 | 
542 | * 使用 PCG64 + deterministic Fisher–Yates；
543 | * stratum 内 K histogram 完全不变；
544 | * stratum 少于 2 行时按相邻 valid-length bucket 确定性合并；
545 | * 若所有 K 最终相同，dynamic mechanism 被判定不可识别。
546 | 
547 | **最终通过条件**
548 | 
549 | 同时满足：
550 | 
551 | * Avg-mAP `≥ 65.685724`，即至少超过历史 H65 `+0.30 pp`；
552 | * @0.6 `≥ 58.581766`；
553 | * @0.7 `≥ 43.883956`；
554 | * learned dynamic 相对 fixed M24：`ΔAvg ≥ +0.30 pp` 且 `LCB95>0`；
555 | * learned dynamic 相对 K-shuffle：`LCB95(ΔAvg)>0`；
556 | * learned dynamic 相对 actionness-only：@0.6/@0.7 的 paired difference 均不为负；
557 | * identical-K exact-uniform positions 不能解释全部收益；
558 | * realized mean K 在 384 的 ±1% 内；
559 | * requested、executed M 逐窗口一致；
560 | * VideoMAE 实际执行 clip 数等于 M，无 Kmax padding 假动态；
561 | * 短动作层不下降超过 0.50 pp；
562 | * scout、outer scorer、inner sampler、heavy backbone、detector、NMS 的成本全部计入。
563 | 
564 | **Kill rule**
565 | 
566 | 出现以下任一项即终止 dynamic paper route：
567 | 
568 | * 不优于 K-shuffle；
569 | * 不优于 actionness-only；
570 | * 高-IoU或短动作明显下降；
571 | * realized cost 不匹配；
572 | * dynamic K 未真实减少或增加 VideoMAE 调用；
573 | * 最终未超过 65.385724；
574 | * 任何 official-validation leakage 或 checkpoint cherry-picking。
575 | 
576 | ### 证据边界
577 | 
578 | 这三个单元均为 seed-3407 的**决定性 development gate**。
579 | 
580 | 即使 Unit 3 达标，也只能声称：
581 | 
582 | > “在预注册单种子完整 official protocol 下超过历史混杂诊断。”
583 | 
584 | 不能立即声称论文级显著优势。论文主张仍需后续至少三粒配对种子、完整 full-stack cost 和独立 seal。
585 | 
586 | ---
587 | 
588 | ## 8. 禁止复用或重复的历史实验
589 | 
590 | 以下内容不得作为新增完整训练臂重复：
591 | 
592 | * official dense；
593 | * exact-uniform；
594 | * random；
595 | * actionness-only fixed-K；
596 | * actionness+boundary fixed-K；
597 | * 已有连续 cliplet frozen/joint；
598 | * 现有 RankPack/TrueTime 20/20/20 pair；
599 | * UVT `off/geo/geo_ema`；
600 | * Fovea `baseline/query_only/query_gt_mask/query_cycle/query_fovea`；
601 | * 旧 U/O/R reachability package；
602 | * 任何失败 root、半程 checkpoint 或 best-validation checkpoint。
603 | 
604 | 允许的复用方式仅限：
605 | 
606 | * immutable historical receipt；
607 | * frozen checkpoint read-only replay；
608 | * 相同权重下的 inference counterfactual；
609 | * policy/utility component replay；
610 | * hard-swap mechanism diagnostic；
611 | * raw-proposal coordinate/NMS counterexample。
612 | 
613 | 下列数字不得改名为论文正证据：
614 | 
615 | * `64.352`；
616 | * `65.696`；
617 | * 当前 `65.385724` 对 uniform 的 `+0.896`；
618 | * UVT/Fovea 单种子 bundle 排名；
619 | * TrueTime 的 `+0.6208`；
620 | * 连续 cliplet 的负结果。
621 | 
622 | 尤其禁止：
623 | 
624 | * 用不同 commit 的 mAP 直接给模块排序；
625 | * 把 Stage-1 额外 detector exposure 隐藏成“预训练免费”；
626 | * 将 selected rank 称为 physical time；
627 | * 用 FLOPs 推断 latency/energy；
628 | * 用 intermediate validation 选择最好 checkpoint；
629 | * Unit 1/2 未过就训练 dynamic outer-K。
630 | 
631 | ---
632 | 
633 | ## 9. Claim / anti-claim / falsifier / novelty invalidator
634 | 
635 | ### 9.1 允许追求的核心 claim
636 | 
637 | > 在 H65 兼容的 boundary-sensitive semantic acquisition 下，非均匀采样引起的时间扭曲可以通过一个保持 VideoMAE canonical prior 的单时钟物理残差修正；该修正使 dynamic outer-K 在相同平均重计算量下优于固定预算和 K-shuffle，并保护高 tIoU 边界定位。
638 | 
639 | ### 9.2 Anti-claims
640 | 
641 | 即使实验成功，也不得声称：
642 | 
643 | * H65 的 `+0.896 pp` 是非均匀采样的单项贡献；
644 | * physical time 单独带来提升；
645 | * Query-Bridge 是 index/K generator；
646 | * generic adaptive frame selection 是新颖点；
647 | * relative-time bias 本身是新颖点；
648 | * dynamic token budget 本身是新颖点；
649 | * fixed K384 是最终方法；
650 | * 单种子超过 65.385724 即为论文级结论；
651 | * nominal K 等于端到端成本；
652 | * 当前方法已经优于所有 adaptive video/token 方法。
653 | 
654 | AdaFrame 已经做过 per-input adaptive frame selection；MGSampler 已经使用累计运动分布实现覆盖式非均匀采样；AdapTok 和 EVATok 已经实现内容依赖的动态 token allocation；TE-TAD 已经把 actual timeline coordinate 用于 TAD。因此“动态预算”“累积分布采样”“真实时间坐标”都不能单独支撑 novelty。([Open Access CVF][1])
655 | 
656 | ### 9.3 最便宜的真实视频 falsifier
657 | 
658 | 在 training-side video-disjoint CAL 上构造同内容、同 frame set 的时间重参数化测试：
659 | 
660 | * 保持 selected RGB 顺序和像素不变；
661 | * 只改变合法 physical timestamp/gap；
662 | * 按 gap-CV 与 boundary density 分层；
663 | * 用同一冻结 detector 比较 RankPack 与 SingleClock 的 boundary error 和 @0.6/@0.7 proxy。
664 | 
665 | 该机制预测：
666 | 
667 | * RankPack 对 timestamp 改变近乎不敏感，并产生物理边界漂移；
668 | * SingleClock 应随真实 gap 改变其 temporal interaction；
669 | * SingleClock 相对收益应随 gap heterogeneity 增大。
670 | 
671 | 如果收益与 gap heterogeneity、boundary density 无交互，或只是所有窗口上的统一平移，则“单时钟物理残差”机制被证伪。
672 | 
673 | ### 9.4 立即杀死 novelty 的事实
674 | 
675 | 以下任一情况足以杀死主论文叙事：
676 | 
677 | 1. 已有工作在 TAD 中同时实现并验证了：
678 | 
679 |    * deploy-visible per-video/window dynamic K；
680 |    * heavy-backbone 前的真实物理帧删除；
681 |    * nonuniform exact-K acquisition；
682 |    * heavy encoder 内真实 gap/time 表达；
683 |    * unchanged detector；
684 |    * pre-NMS physical remap；
685 |    * matched realized full-stack cost；
686 |    * high-IoU/short-action 改善。
687 | 2. Unit 3 不优于 K-shuffle，说明收益只来自 K histogram，而非内容—预算对应关系。
688 | 3. Unit 1 的收益不随 gap heterogeneity 或 boundary risk 变化，说明 SingleClock 只是普通 adapter。
689 | 4. Query residual 的作用可被简单 actionness、transition magnitude 或更多参数完全解释。
690 | 5. full-stack scout/selector 开销消除 heavy-backbone 节省。
691 | 
692 | ---
693 | 
694 | ## 10. 下一步立即动作
695 | 
696 | **只能做一件事时，立即实施并封存 Unit 1。**
697 | 
698 | ### Builder
699 | 
700 | 从 `42dba3f...` 创建唯一干净 descendant：
701 | 
702 | 1. 保留 H65 RankPack 路径不变；
703 | 2. 将 q→t 映射移动到 `TwoStageDetector.post_processing` 的最前端；
704 | 3. 实现 canonical-time residual attention bias；
705 | 4. 主路线禁用：
706 | 
707 |    * `PhysicalTimeTubeletEmbedding`；
708 |    * physical coordinate MLP；
709 |    * gap-scaled adapter convolution；
710 |    * physical-grid ActionFormer；
711 | 5. 增加 H65 checkpoint replay mode；
712 | 6. 生成 Unit 1 config；
713 | 7. 不实现 Query，不实现 dynamic K。
714 | 
715 | ### Critic
716 | 
717 | 只审查以下五点：
718 | 
719 | * H65 replay 是否 bit-identical；
720 | * SingleClock 是否是唯一差异；
721 | * uniform gap 时 relative bias 是否严格为零；
722 | * 是否仍存在第二个 absolute physical clock；
723 | * q→t 是否早于 threshold/top-k/IoU/NMS。
724 | 
725 | 任一失败即 `UNIT1_PRE_RUN_BLOCKED`。
726 | 
727 | ### Evaluator
728 | 
729 | 在任何训练前封存：
730 | 
731 | * exact commit；
732 | * H65 checkpoint hash；
733 | * resolved config；
734 | * seed 3407；
735 | * official split/evaluator/NMS；
736 | * 6000-update rule；
737 | * final/final-EMA rule；
738 | * 五个 replay parity 边界；
739 | * requested/effective/executed K ledger；
740 | * full-stack cost schema；
741 | * Unit 1 pass/kill rule。
742 | 
743 | **绑定结论：先解释 H65，再测试 Query，最后才允许 dynamic outer-K。不得再开启理论分叉或新增第四个完整训练单元。**
744 | 
745 | [1]: https://openaccess.thecvf.com/content/CVPR2026/html/Li_AdapTok_Learning_Adaptive_and_Temporally_Causal_Video_Tokenization_in_a_CVPR_2026_paper.html?utm_source=chatgpt.com "CVPR 2026 Open Access Repository"
````

### File: ../OpenTAD_DUCA_H65_FirstMixSingleClock_20260824/tools/bata/finalize_duca_h65_singleclock_terminal.py
Lines: 1-567
```python
  1 | from __future__ import annotations
  2 | 
  3 | import argparse
  4 | import hashlib
  5 | import json
  6 | from pathlib import Path
  7 | from typing import Any, Mapping
  8 | 
  9 | import numpy as np
 10 | 
 11 | from tools.bata.duca_p0_training import atomic_write_json
 12 | 
 13 | 
 14 | METRICS = ("average_mAP", "mAP@0.6", "mAP@0.7")
 15 | 
 16 | 
 17 | def _require(condition: bool, message: str) -> None:
 18 |     if not condition:
 19 |         raise ValueError(message)
 20 | 
 21 | 
 22 | def _load(path: str | Path) -> dict[str, Any]:
 23 |     payload = json.loads(Path(path).read_text(encoding="utf-8"))
 24 |     _require(isinstance(payload, dict), f"expected a JSON object: {path}")
 25 |     return payload
 26 | 
 27 | 
 28 | def _metric(metrics: Mapping[str, Any], key: str) -> float:
 29 |     aliases = {
 30 |         "average_mAP": ("average_mAP", "Avg-mAP", "avg_mAP"),
 31 |         "mAP@0.6": ("mAP@0.6", "0.6"),
 32 |         "mAP@0.7": ("mAP@0.7", "0.7"),
 33 |     }
 34 |     for alias in aliases[key]:
 35 |         if alias in metrics:
 36 |             value = float(metrics[alias])
 37 |             _require(np.isfinite(value), f"non-finite metric {key}")
 38 |             return value
 39 |     raise ValueError(f"missing metric {key}")
 40 | 
 41 | 
 42 | def _exact_interval(values: np.ndarray) -> tuple[float, float]:
 43 |     ordered = np.sort(np.asarray(values, dtype=np.float64))
 44 |     _require(ordered.shape == (10000,), "terminal bootstrap must contain exactly 10,000 draws")
 45 |     return float(ordered[249]), float(ordered[9749])
 46 | 
 47 | 
 48 | def _paired_delta(
 49 |     bootstrap: Mapping[str, Any], lhs: str, rhs: str, metric: str
 50 | ) -> dict[str, float]:
 51 |     _require(bootstrap.get("samples") == 10000, "bootstrap sample count must be 10,000")
 52 |     _require(
 53 |         (bootstrap.get("lower_rank"), bootstrap.get("upper_rank")) == (250, 9750),
 54 |         "bootstrap interval ranks must be 250/9750",
 55 |     )
 56 |     sampled = bootstrap.get("sampled_metrics")
 57 |     points = bootstrap.get("point_estimates")
 58 |     _require(isinstance(sampled, Mapping) and isinstance(points, Mapping), "bootstrap evidence is incomplete")
 59 |     lhs_draws = np.asarray(sampled[lhs][metric], dtype=np.float64)
 60 |     rhs_draws = np.asarray(sampled[rhs][metric], dtype=np.float64)
 61 |     _require(lhs_draws.shape == rhs_draws.shape == (10000,), "paired bootstrap families do not align")
 62 |     delta = lhs_draws - rhs_draws
 63 |     lower, upper = _exact_interval(delta)
 64 |     point = _metric(points[lhs], metric) - _metric(points[rhs], metric)
 65 |     return {
 66 |         "point": float(point),
 67 |         "ci_lower_exact_rank": lower,
 68 |         "ci_upper_exact_rank": upper,
 69 |         "point_pp": float(point * 100.0),
 70 |         "ci_lower_pp": float(lower * 100.0),
 71 |         "ci_upper_pp": float(upper * 100.0),
 72 |     }
 73 | 
 74 | 
 75 | def _identity_equal(on: Mapping[str, Any], zero: Mapping[str, Any]) -> bool:
 76 |     def valid_records(payload: Mapping[str, Any]) -> bool:
 77 |         records = payload.get("records")
 78 |         sample_count = payload.get("sample_count")
 79 |         if not isinstance(records, list) or isinstance(sample_count, bool) or not isinstance(sample_count, int):
 80 |             return False
 81 |         if len(records) != sample_count or sample_count <= 0:
 82 |             return False
 83 |         required = ("sample_id", "video_name", "window_start_frame", "selected_valid_len", "dense_valid_len", "selected_positions", "selected_rgb_sha256", "selected_positions_sha256", "selected_mask_sha256")
 84 |         ids = []
 85 |         for row in records:
 86 |             if not isinstance(row, Mapping) or any(key not in row for key in required):
 87 |                 return False
 88 |             if not isinstance(row["sample_id"], str) or not row["sample_id"] or not isinstance(row["video_name"], str) or not row["video_name"]:
 89 |                 return False
 90 |             if any(not isinstance(row[key], str) or not row[key] for key in ("selected_rgb_sha256", "selected_positions_sha256", "selected_mask_sha256")):
 91 |                 return False
 92 |             fields = ("window_start_frame", "selected_valid_len", "dense_valid_len")
 93 |             if any(isinstance(row[key], bool) or not isinstance(row[key], int) for key in fields):
 94 |                 return False
 95 |             if row["window_start_frame"] < 0 or row["selected_valid_len"] <= 0 or row["dense_valid_len"] <= 0 or row["selected_valid_len"] > row["dense_valid_len"]:
 96 |                 return False
 97 |             positions = row["selected_positions"]
 98 |             if not isinstance(positions, list) or len(positions) != row["selected_valid_len"] or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= row["dense_valid_len"] for value in positions) or positions != sorted(set(positions)):
 99 |                 return False
100 |             if row["sample_id"] != f"{row['video_name']}|window_start_frame={row['window_start_frame']}":
101 |                 return False
102 |             ids.append(row["sample_id"])
103 |         return len(set(ids)) == len(ids) and ids == sorted(ids)
104 | 
105 |     def valid_accounting(payload: Mapping[str, Any]) -> bool:
106 |         required = ("sample_count", "total_input_exposure_count", "unique_physical_window_count", "duplicate_exposure_count", "duplicate_samples")
107 |         if any(key not in payload for key in required):
108 |             return False
109 |         integers = tuple(payload[key] for key in required[:4])
110 |         if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in integers):
111 |             return False
112 |         unique = payload["unique_physical_window_count"]
113 |         if payload["sample_count"] != unique:
114 |             return False
115 |         duplicates = payload["duplicate_exposure_count"]
116 |         if payload["total_input_exposure_count"] != unique + duplicates:
117 |             return False
118 |         if not valid_records(payload):
119 |             return False
120 |         rows = payload["duplicate_samples"]
121 |         if not isinstance(rows, list):
122 |             return False
123 |         normalized = []
124 |         for row in rows:
125 |             if not isinstance(row, Mapping) or set(row) != {"sample_id", "duplicate_exposure_count"}:
126 |                 return False
127 |             sample_id = row["sample_id"]
128 |             count = row["duplicate_exposure_count"]
129 |             if not isinstance(sample_id, str) or not sample_id:
130 |                 return False
131 |             if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
132 |                 return False
133 |             normalized.append((sample_id, count))
134 |         if len({sample_id for sample_id, _ in normalized}) != len(normalized):
135 |             return False
136 |         if normalized != sorted(normalized):
137 |             return False
138 |         if sum(count for _, count in normalized) != duplicates:
139 |             return False
140 |         if any(sample_id not in {row["sample_id"] for row in payload["records"]} for sample_id, _ in normalized):
141 |             return False
142 |         return duplicates != 0 or not normalized
143 | 
144 |     if not valid_accounting(on) or not valid_accounting(zero):
145 |         return False
146 |     if on["sample_count"] != zero["sample_count"]:
147 |         return False
148 |     accounting_keys = (
149 |         "total_input_exposure_count",
150 |         "unique_physical_window_count",
151 |         "duplicate_exposure_count",
152 |         "duplicate_samples",
153 |     )
154 |     if any(on[key] != zero[key] for key in accounting_keys):
155 |         return False
156 |     on_records = on["records"]
157 |     zero_records = zero["records"]
158 |     keys = (
159 |         "sample_id",
160 |         "video_name",
161 |         "window_start_frame",
162 |         "selected_valid_len",
163 |         "dense_valid_len",
164 |         "selected_positions",
165 |         "selected_rgb_sha256",
166 |         "selected_positions_sha256",
167 |         "selected_mask_sha256",
168 |     )
169 |     return [tuple(row.get(key) if key != "selected_positions" else tuple(row.get(key, ())) for key in keys) for row in on_records] == [
170 |         tuple(row.get(key) if key != "selected_positions" else tuple(row.get(key, ())) for key in keys)
171 |         for row in zero_records
172 |     ]
173 | 
174 | 
175 | def _config_hash_ok(row: Mapping[str, Any], expected_suffix: str) -> bool:
176 |     config_path = Path(str(row.get("config_path", "")))
177 |     expected_hash = str(row.get("config_sha256", ""))
178 |     if not _file_hash_matches(config_path, expected_hash):
179 |         return False
180 |     if config_path.as_posix().endswith(expected_suffix) is False:
181 |         return False
182 |     return True
183 | 
184 | 
185 | def _file_hash_matches(path: str | Path, expected_hash: str) -> bool:
186 |     artifact = Path(path)
187 |     expected = str(expected_hash)
188 |     return bool(
189 |         artifact.is_file()
190 |         and len(expected) == 64
191 |         and hashlib.sha256(artifact.read_bytes()).hexdigest() == expected
192 |     )
193 | 
194 | 
195 | def _same_path(lhs: str | Path, rhs: str | Path) -> bool:
196 |     return Path(lhs).resolve() == Path(rhs).resolve()
197 | 
198 | 
199 | def _family_execution_contract_ok(
200 |     row: Mapping[str, Any],
201 |     metrics: Mapping[str, Any],
202 |     *,
203 |     expected_config_suffix: str,
204 |     expected_gate_zero: bool | None,
205 |     expected_checkpoint_path: str | Path,
206 |     expected_checkpoint_sha256: str,
207 |     expected_state_key: str,
208 |     require_identity: bool,
209 | ) -> bool:
210 |     if row.get("single_clock_gate_zero") is not expected_gate_zero:
211 |         return False
212 |     if not _config_hash_ok(row, expected_config_suffix):
213 |         return False
214 |     if not _same_path(metrics.get("checkpoint_path", ""), expected_checkpoint_path):
215 |         return False
216 |     if metrics.get("checkpoint_sha256") != expected_checkpoint_sha256:
217 |         return False
218 |     if metrics.get("checkpoint_epoch") != 59:
219 |         return False
220 |     if metrics.get("checkpoint_state_key") != expected_state_key:
221 |         return False
222 |     identity_path = row.get("selected_input_identity_path")
223 |     identity_hash = row.get("selected_input_identity_sha256")
224 |     if require_identity:
225 |         return _file_hash_matches(identity_path or "", identity_hash or "")
226 |     return identity_path is None and identity_hash is None
227 | 
228 | 
229 | def _twin_execution_contract_ok(
230 |     on_row: Mapping[str, Any], zero_row: Mapping[str, Any]
231 | ) -> bool:
232 |     on_suffix = "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
233 |     zero_suffix = (
234 |         "configs/adatad/thumos/"
235 |         "duca_h65_first_singleclock_cycle4_gate_zero.py"
236 |     )
237 |     return bool(
238 |         on_row.get("single_clock_gate_zero") is False
239 |         and zero_row.get("single_clock_gate_zero") is True
240 |         and _config_hash_ok(on_row, on_suffix)
241 |         and _config_hash_ok(zero_row, zero_suffix)
242 |         and on_row.get("config_sha256") != zero_row.get("config_sha256")
243 |     )
244 | 
245 | 
246 | def _audit_ok(clock: Mapping[str, Any], off: Mapping[str, Any]) -> bool:
247 |     clock_values = clock.get("single_clock_values")
248 |     off_values = off.get("single_clock_values")
249 |     scalar_ok = (
250 |         isinstance(clock_values, Mapping)
251 |         and isinstance(off_values, Mapping)
252 |         and set(clock_values) == {"state_dict", "state_dict_ema"}
253 |         and set(off_values) == {"state_dict", "state_dict_ema"}
254 |         and all(
255 |             isinstance(clock_values[key], Mapping)
256 |             and len(clock_values[key]) == 1
257 |             and np.isfinite(float(next(iter(clock_values[key].values()))))
258 |             and float(next(iter(clock_values[key].values()))) != 0.0
259 |             and isinstance(off_values[key], Mapping)
260 |             and len(off_values[key]) <= 1
261 |             and all(
262 |                 np.isfinite(float(value)) and float(value) == 0.0
263 |                 for value in off_values[key].values()
264 |             )
265 |             for key in ("state_dict", "state_dict_ema")
266 |         )
267 |     )
268 |     common = (
269 |         clock.get("checkpoint_epoch") == off.get("checkpoint_epoch") == 59
270 |         and clock.get("successful_optimizer_updates") == off.get("successful_optimizer_updates") == 6000
271 |         and clock.get("scheduler_last_epoch") == off.get("scheduler_last_epoch") == 6000
272 |         and clock.get("stage1_checkpoint_sha256") == off.get("stage1_checkpoint_sha256")
273 |         and clock.get("stage1_checkpoint_epoch") == off.get("stage1_checkpoint_epoch") == 29
274 |     )
275 |     return bool(common and scalar_ok and clock.get("family") == "clock_on" and off.get("family") == "h65_off")
276 | 
277 | 
278 | def finalize(
279 |     *,
280 |     receipt: Mapping[str, Any],
281 |     clock_audit: Mapping[str, Any],
282 |     off_audit: Mapping[str, Any],
283 |     final_bootstrap: Mapping[str, Any],
284 |     ema_bootstrap: Mapping[str, Any],
285 |     old_pair_bootstrap: Mapping[str, Any],
286 |     strata: Mapping[str, Any],
287 |     cost: Mapping[str, Any],
288 |     stage1_average_map: float,
289 |     expected_eval_commit: str,
290 | ) -> dict[str, Any]:
291 |     _require(
292 |         receipt.get("schema_version")
293 |         == "duca_h65_singleclock_terminal_eval_receipt_v1",
294 |         "terminal receipt schema mismatch",
295 |     )
296 |     _require(
297 |         receipt.get("git_commit") == expected_eval_commit
298 |         and len(str(expected_eval_commit)) == 40,
299 |         "terminal evaluation commit mismatch",
300 |     )
301 |     families = receipt.get("families")
302 |     required = {
303 |         "final_on",
304 |         "final_gate_zero",
305 |         "ema_on",
306 |         "ema_gate_zero",
307 |         "h65_off_final",
308 |         "h65_off_ema",
309 |     }
310 |     _require(isinstance(families, Mapping) and set(families) == required, "terminal receipt families differ from the frozen six readouts")
311 | 
312 |     clock_checkpoint = receipt.get("clock_checkpoint", "")
313 |     clock_checkpoint_sha256 = receipt.get("clock_checkpoint_sha256", "")
314 |     off_checkpoint = receipt.get("h65_off_checkpoint", "")
315 |     off_checkpoint_sha256 = receipt.get("h65_off_checkpoint_sha256", "")
316 |     stage1_checkpoint = receipt.get("stage1_checkpoint", "")
317 |     stage1_checkpoint_sha256 = receipt.get("stage1_checkpoint_sha256", "")
318 |     _require(
319 |         _file_hash_matches(clock_checkpoint, clock_checkpoint_sha256),
320 |         "SingleClock checkpoint binding failed",
321 |     )
322 |     _require(
323 |         _file_hash_matches(off_checkpoint, off_checkpoint_sha256),
324 |         "H65 OFF checkpoint binding failed",
325 |     )
326 |     _require(
327 |         _file_hash_matches(stage1_checkpoint, stage1_checkpoint_sha256),
328 |         "Stage-1 checkpoint binding failed",
329 |     )
330 | 
331 |     clock_suffix = "configs/adatad/thumos/duca_h65_first_singleclock_cycle4.py"
332 |     zero_suffix = (
333 |         "configs/adatad/thumos/"
334 |         "duca_h65_first_singleclock_cycle4_gate_zero.py"
335 |     )
336 |     off_suffix = (
337 |         "configs/adatad/thumos/"
338 |         "duca_sampling_rate_curriculum_stage2_joint384.py"
339 |     )
340 |     family_specs = {
341 |         "final_on": (clock_suffix, False, clock_checkpoint, clock_checkpoint_sha256, "state_dict", True),
342 |         "final_gate_zero": (zero_suffix, True, clock_checkpoint, clock_checkpoint_sha256, "state_dict", True),
343 |         "ema_on": (clock_suffix, False, clock_checkpoint, clock_checkpoint_sha256, "state_dict_ema", True),
344 |         "ema_gate_zero": (zero_suffix, True, clock_checkpoint, clock_checkpoint_sha256, "state_dict_ema", True),
345 |         "h65_off_final": (off_suffix, None, off_checkpoint, off_checkpoint_sha256, "state_dict", False),
346 |         "h65_off_ema": (off_suffix, None, off_checkpoint, off_checkpoint_sha256, "state_dict_ema", False),
347 |     }
348 |     loaded_metrics = {}
349 |     execution_contract = {}
350 |     for family, spec in family_specs.items():
351 |         row = families[family]
352 |         _require(isinstance(row, Mapping), f"terminal family row is invalid: {family}")
353 |         metrics_path = row.get("metrics_path", "")
354 |         _require(
355 |             _file_hash_matches(metrics_path, row.get("metrics_sha256", "")),
356 |             f"terminal metrics hash mismatch: {family}",
357 |         )
358 |         metrics = _load(metrics_path)
359 |         loaded_metrics[family] = metrics
360 |         execution_contract[family] = _family_execution_contract_ok(
361 |             row,
362 |             metrics,
363 |             expected_config_suffix=spec[0],
364 |             expected_gate_zero=spec[1],
365 |             expected_checkpoint_path=spec[2],
366 |             expected_checkpoint_sha256=spec[3],
367 |             expected_state_key=spec[4],
368 |             require_identity=spec[5],
369 |         )
370 |     _require(all(execution_contract.values()), "terminal family execution binding failed")
371 | 
372 |     identity = {}
373 |     twin_execution_contract = {}
374 |     for prefix in ("final", "ema"):
375 |         on_row = families[f"{prefix}_on"]
376 |         zero_row = families[f"{prefix}_gate_zero"]
377 |         twin_execution_contract[prefix] = _twin_execution_contract_ok(
378 |             on_row, zero_row
379 |         )
380 |         on_metrics = loaded_metrics[f"{prefix}_on"]
381 |         zero_metrics = loaded_metrics[f"{prefix}_gate_zero"]
382 |         for key in ("checkpoint_path", "checkpoint_sha256", "checkpoint_epoch", "checkpoint_state_key"):
383 |             _require(on_metrics.get(key) == zero_metrics.get(key), f"{prefix} twin differs on {key}")
384 |         on = _load(on_row["selected_input_identity_path"])
385 |         zero = _load(zero_row["selected_input_identity_path"])
386 |         identity[prefix] = _identity_equal(on, zero)
387 | 
388 |     family_names = {
389 |         "final": (final_bootstrap, "final_on", "final_gate_zero", "h65_off_final"),
390 |         "ema": (ema_bootstrap, "ema_on", "ema_gate_zero", "h65_off_ema"),
391 |     }
392 |     estimates: dict[str, Any] = {}
393 |     for prefix, (artifact, on_name, zero_name, off_name) in family_names.items():
394 |         estimates[prefix] = {
395 |             "single_clock": {
396 |                 metric: _paired_delta(artifact, on_name, zero_name, metric)
397 |                 for metric in METRICS
398 |             },
399 |             "external": {
400 |                 metric: _paired_delta(artifact, on_name, off_name, metric)
401 |                 for metric in METRICS
402 |             },
403 |             "coadaptation": {
404 |                 metric: _paired_delta(artifact, zero_name, off_name, metric)
405 |                 for metric in ("average_mAP", "mAP@0.7")
406 |             },
407 |         }
408 | 
409 |     old = {
410 |         metric: _paired_delta(old_pair_bootstrap, "truetime", "rankpack", metric)
411 |         for metric in METRICS
412 |     }
413 |     old_pair_partial_representation_evidence = (
414 |         old["average_mAP"]["ci_lower_pp"] > 0.0
415 |         and old["mAP@0.6"]["ci_lower_pp"] > 0.0
416 |     )
417 |     old_pair_no_explicit_harm = (
418 |         old["average_mAP"]["ci_upper_pp"] > 0.0
419 |         and old["mAP@0.7"]["ci_upper_pp"] >= -0.20
420 |     )
421 | 
422 |     _require(strata.get("schema_version") == "duca_h65_singleclock_strata_v1", "strata artifact schema mismatch")
423 |     _require(
424 |         strata.get("primary_checkpoint_state_key") == "state_dict_ema",
425 |         "strata evidence must use the frozen EMA checkpoint state",
426 |     )
427 |     _require(cost.get("schema_version") == "duca_h65_singleclock_cost_pair_v1", "cost artifact schema mismatch")
428 |     cost_pass = (
429 |         float(cost["median_latency_ratio_on_over_gate_zero"]) <= 1.01
430 |         and float(cost["p90_latency_ratio_on_over_gate_zero"]) <= 1.02
431 |         and float(cost["peak_memory_ratio_on_over_gate_zero"]) <= 1.02
432 |     )
433 |     strata_pass = (
434 |         float(strata["short_action_delta_pp"]) >= -0.50
435 |         and float(strata["distortion_interaction_point_pp"]) > 0.0
436 |     )
437 | 
438 |     final_sc = estimates["final"]["single_clock"]
439 |     ema_sc = estimates["ema"]["single_clock"]
440 |     ema_ext = estimates["ema"]["external"]
441 |     ema_co = estimates["ema"]["coadaptation"]
442 |     final_off_avg = _metric(final_bootstrap["point_estimates"]["h65_off_final"], "average_mAP")
443 |     ema_off_avg = _metric(ema_bootstrap["point_estimates"]["h65_off_ema"], "average_mAP")
444 |     baseline_mature = (ema_off_avg - float(stage1_average_map)) * 100.0 >= 0.50
445 |     clock_recovery_contract_pass = bool(
446 |         clock_audit.get("recovery_state_complete")
447 |     )
448 |     h65_off_recovery_contract_pass = bool(
449 |         off_audit.get("recovery_state_complete")
450 |     )
451 | 
452 |     direction_consistent = all(
453 |         final_sc[metric]["point"] * ema_sc[metric]["point"] >= 0.0
454 |         for metric in METRICS
455 |     )
456 |     main_pass = (
457 |         ema_sc["average_mAP"]["point_pp"] >= 0.50
458 |         and ema_sc["average_mAP"]["ci_lower_pp"] > 0.0
459 |         and ema_sc["mAP@0.6"]["point_pp"] >= 0.0
460 |         and ema_sc["mAP@0.7"]["point_pp"] >= 0.0
461 |         and ema_sc["mAP@0.7"]["ci_lower_pp"] > -0.20
462 |         and ema_ext["average_mAP"]["point_pp"] >= 0.50
463 |         and ema_ext["mAP@0.7"]["point_pp"] >= 0.0
464 |         and all(
465 |             row["ci_lower_pp"] >= -0.20 and row["ci_upper_pp"] <= 0.20
466 |             for row in ema_co.values()
467 |         )
468 |     )
469 |     hard_fail = (
470 |         not old_pair_no_explicit_harm
471 |         or not baseline_mature
472 |         or not all(identity.values())
473 |         or not all(twin_execution_contract.values())
474 |         or not _audit_ok(clock_audit, off_audit)
475 |         or not clock_recovery_contract_pass
476 |         or not cost_pass
477 |         or any(
478 |             sc["average_mAP"]["point_pp"] <= 0.0
479 |             or sc["average_mAP"]["ci_upper_pp"] <= 0.0
480 |             or sc["mAP@0.7"]["point_pp"] <= -0.50
481 |             or sc["mAP@0.7"]["ci_upper_pp"] < -0.20
482 |             for sc in (final_sc, ema_sc)
483 |         )
484 |     )
485 |     if hard_fail:
486 |         decision = "PIVOT_TO_ACQUISITION_OR_TRAINING_MATURITY"
487 |     elif main_pass and strata_pass and direction_consistent:
488 |         decision = (
489 |             "CONTINUE_TO_REPLICATION"
490 |             if h65_off_recovery_contract_pass
491 |             else "REVISE_WITHOUT_MORE_TIME_MODULES"
492 |         )
493 |     else:
494 |         decision = "REVISE_WITHOUT_MORE_TIME_MODULES"
495 | 
496 |     return {
497 |         "schema_version": "duca_h65_singleclock_terminal_adjudication_v1",
498 |         "decision": decision,
499 |         "identity_gate_pass": all(identity.values()),
500 |         "twin_execution_contract_pass": all(twin_execution_contract.values()),
501 |         "family_execution_contract_pass": all(execution_contract.values()),
502 |         "checkpoint_audit_gate_pass": _audit_ok(clock_audit, off_audit),
503 |         "clock_recovery_contract_pass": clock_recovery_contract_pass,
504 |         "h65_off_recovery_contract_pass": h65_off_recovery_contract_pass,
505 |         "h65_off_recovery_protocol_deviation": list(
506 |             off_audit.get("recovery_protocol_deviation", ())
507 |         ),
508 |         "old_pair_representation_gate_pass": old_pair_no_explicit_harm,
509 |         "old_pair_partial_representation_evidence": old_pair_partial_representation_evidence,
510 |         "h65_off_training_maturity_gate_pass": baseline_mature,
511 |         "cost_gate_pass": cost_pass,
512 |         "strata_gate_pass": strata_pass,
513 |         "final_ema_direction_consistent": direction_consistent,
514 |         "identity": identity,
515 |         "old_pair": old,
516 |         "estimates": estimates,
517 |         "cost": dict(cost),
518 |         "strata": dict(strata),
519 |         "stage1_average_map": float(stage1_average_map),
520 |         "h65_off_final_average_map": final_off_avg,
521 |         "h65_off_ema_average_map": ema_off_avg,
522 |         "claim_boundary": "single_seed_representation_gate_only",
523 |         "paper_claim_admissible": bool(
524 |             decision == "CONTINUE_TO_REPLICATION"
525 |             and clock_recovery_contract_pass
526 |             and h65_off_recovery_contract_pass
527 |         ),
528 |         "bridge_authorized": False,
529 |         "dynamic_k_authorized": False,
530 |     }
531 | 
532 | 
533 | def parse_args():
534 |     parser = argparse.ArgumentParser(description="Finalize the frozen H65 SingleClock terminal gate")
535 |     parser.add_argument("--terminal-receipt", required=True)
536 |     parser.add_argument("--clock-audit", required=True)
537 |     parser.add_argument("--off-audit", required=True)
538 |     parser.add_argument("--final-bootstrap", required=True)
539 |     parser.add_argument("--ema-bootstrap", required=True)
540 |     parser.add_argument("--old-pair-bootstrap", required=True)
541 |     parser.add_argument("--strata", required=True)
542 |     parser.add_argument("--cost", required=True)
543 |     parser.add_argument("--stage1-average-map", type=float, default=0.594231)
544 |     parser.add_argument("--expected-eval-commit", required=True)
545 |     parser.add_argument("--output", required=True)
546 |     return parser.parse_args()
547 | 
548 | 
549 | def main():
550 |     args = parse_args()
551 |     payload = finalize(
552 |         receipt=_load(args.terminal_receipt),
553 |         clock_audit=_load(args.clock_audit),
554 |         off_audit=_load(args.off_audit),
555 |         final_bootstrap=_load(args.final_bootstrap),
556 |         ema_bootstrap=_load(args.ema_bootstrap),
557 |         old_pair_bootstrap=_load(args.old_pair_bootstrap),
558 |         strata=_load(args.strata),
559 |         cost=_load(args.cost),
560 |         stage1_average_map=args.stage1_average_map,
561 |         expected_eval_commit=args.expected_eval_commit,
562 |     )
563 |     atomic_write_json(args.output, payload)
564 | 
565 | 
566 | if __name__ == "__main__":
567 |     main()
```

### File: ../OpenTAD_DUCA_H65_FirstMixSingleClock_20260824/tools/bata/analyze_duca_h65_singleclock_strata.py
Lines: 1-343
```python
  1 | from __future__ import annotations
  2 | 
  3 | import argparse
  4 | import json
  5 | from pathlib import Path
  6 | import tempfile
  7 | from typing import Any, Mapping
  8 | 
  9 | import numpy as np
 10 | 
 11 | from tools.bata.duca_allocation_families import exact_uniform_positions
 12 | from tools.bata.bootstrap_duca_h65_official_map import bootstrap_h65_official_map, exact_interval
 13 | from tools.bata.duca_p0_evaluation import EXPECTED_TIOU_THRESHOLDS, prediction_results, sha256_file
 14 | from tools.bata.duca_p0_training import atomic_write_json
 15 | 
 16 | 
 17 | def _require(condition: bool, message: str) -> None:
 18 |     if not condition:
 19 |         raise ValueError(message)
 20 | 
 21 | 
 22 | def _load(path: str | Path) -> dict[str, Any]:
 23 |     payload = json.loads(Path(path).read_text(encoding="utf-8"))
 24 |     _require(isinstance(payload, dict), f"expected a JSON object: {path}")
 25 |     return payload
 26 | 
 27 | 
 28 | def _uniform_positions(length: int, count: int) -> np.ndarray:
 29 |     _require(length > 0 and 0 < count <= length, "invalid exact-uniform dimensions")
 30 |     return np.asarray(exact_uniform_positions(length, count), dtype=np.int64)
 31 | 
 32 | 
 33 | def _record_distortion(record: Mapping[str, Any]) -> float:
 34 |     positions = np.asarray(record["selected_positions"], dtype=np.float64)
 35 |     length = int(record["dense_valid_len"])
 36 |     count = int(record["selected_valid_len"])
 37 |     _require(positions.shape == (count,) and count > 0, "identity record positions are malformed")
 38 |     canonical = _uniform_positions(length, count).astype(np.float64)
 39 |     # A trailing partial tubelet has no Clock support and must not fabricate a
 40 |     # time center; the unchanged RGB/mask padding contract still retains it.
 41 |     complete_count = count - (count % 2)
 42 |     if complete_count == 0:
 43 |         return 0.0
 44 |     actual_centers = positions[:complete_count].reshape(-1, 2).mean(axis=1)
 45 |     canonical_centers = canonical[:complete_count].reshape(-1, 2).mean(axis=1)
 46 |     return float(np.mean(np.abs(actual_centers - canonical_centers)) / max(length - 1, 1))
 47 | 
 48 | 
 49 | def _annotations(annotation: Mapping[str, Any], subset: str) -> list[Mapping[str, Any]]:
 50 |     database = annotation.get("database")
 51 |     _require(isinstance(database, Mapping), "annotation has no database mapping")
 52 |     rows = []
 53 |     for video in database.values():
 54 |         if isinstance(video, Mapping) and str(video.get("subset")) == subset:
 55 |             anns = video.get("annotations", [])
 56 |             _require(isinstance(anns, list), "video annotations must be a list")
 57 |             rows.extend(item for item in anns if isinstance(item, Mapping))
 58 |     return rows
 59 | 
 60 | 
 61 | def freeze_training_strata(
 62 |     *, training_identity_path: str | Path, annotation_path: str | Path
 63 | ) -> dict[str, Any]:
 64 |     identity = _load(training_identity_path)
 65 |     annotation = _load(annotation_path)
 66 |     records = identity.get("records")
 67 |     _require(isinstance(records, list) and records, "training identity contains no records")
 68 |     by_video: dict[str, list[float]] = {}
 69 |     for record in records:
 70 |         _require(isinstance(record, Mapping), "identity record must be a mapping")
 71 |         video = str(record["video_name"])
 72 |         by_video.setdefault(video, []).append(_record_distortion(record))
 73 |     database = annotation["database"]
 74 |     training_videos = {
 75 |         str(video_id)
 76 |         for video_id, row in database.items()
 77 |         if isinstance(row, Mapping) and str(row.get("subset")) == "training"
 78 |     }
 79 |     _require(set(by_video) <= training_videos, "training identity contains a non-training video")
 80 |     video_distortion = {
 81 |         video: float(np.mean(values)) for video, values in sorted(by_video.items())
 82 |     }
 83 |     distortion_values = np.asarray(list(video_distortion.values()), dtype=np.float64)
 84 |     _require(distortion_values.size >= 4, "at least four training videos are required")
 85 | 
 86 |     durations = []
 87 |     for item in _annotations(annotation, "training"):
 88 |         segment = item.get("segment")
 89 |         if isinstance(segment, (list, tuple)) and len(segment) == 2:
 90 |             duration = float(segment[1]) - float(segment[0])
 91 |             if np.isfinite(duration) and duration > 0.0:
 92 |                 durations.append(duration)
 93 |     _require(durations, "training annotation contains no positive action durations")
 94 |     q25, q50, q75 = np.quantile(
 95 |         distortion_values, (0.25, 0.50, 0.75), method="linear"
 96 |     )
 97 |     short_q25 = float(
 98 |         np.quantile(np.asarray(durations, dtype=np.float64), 0.25, method="linear")
 99 |     )
100 |     return {
101 |         "schema_version": "duca_h65_singleclock_training_strata_freeze_v1",
102 |         "source_subset": "training",
103 |         "validation_or_test_used": False,
104 |         "training_identity_path": str(Path(training_identity_path).resolve()),
105 |         "training_identity_sha256": sha256_file(training_identity_path),
106 |         "annotation_path": str(Path(annotation_path).resolve()),
107 |         "annotation_sha256": sha256_file(annotation_path),
108 |         "distortion_definition": "per_video_mean_of_window_mean_abs_tubelet_center_displacement_divided_by_dense_len_minus_one",
109 |         "distortion_quantile_method": "numpy_linear",
110 |         "distortion_q25": float(q25),
111 |         "distortion_q50": float(q50),
112 |         "distortion_q75": float(q75),
113 |         "training_video_distortion": video_distortion,
114 |         "training_video_count": len(video_distortion),
115 |         "short_action_definition": "training_ground_truth_duration_seconds_at_or_below_q25",
116 |         "short_action_quantile_method": "numpy_linear",
117 |         "short_action_duration_q25_seconds": short_q25,
118 |         "training_action_instance_count": len(durations),
119 |     }
120 | 
121 | 
122 | def _validation_video_distortion(identity: Mapping[str, Any]) -> dict[str, float]:
123 |     records = identity.get("records")
124 |     _require(isinstance(records, list) and records, "validation identity contains no records")
125 |     by_video: dict[str, list[float]] = {}
126 |     for record in records:
127 |         video = str(record["video_name"])
128 |         by_video.setdefault(video, []).append(_record_distortion(record))
129 |     return {video: float(np.mean(values)) for video, values in sorted(by_video.items())}
130 | 
131 | 
132 | def _write_annotation(
133 |     path: Path,
134 |     annotation: Mapping[str, Any],
135 |     *,
136 |     videos: set[str] | None = None,
137 |     max_duration: float | None = None,
138 | ) -> None:
139 |     database = annotation["database"]
140 |     output = {}
141 |     for video_id, row in database.items():
142 |         if not isinstance(row, Mapping) or str(row.get("subset")) != "validation":
143 |             continue
144 |         if videos is not None and str(video_id) not in videos:
145 |             continue
146 |         copied = dict(row)
147 |         annotations = list(copied.get("annotations", []))
148 |         if max_duration is not None:
149 |             annotations = [
150 |                 item
151 |                 for item in annotations
152 |                 if isinstance(item, Mapping)
153 |                 and isinstance(item.get("segment"), (list, tuple))
154 |                 and len(item["segment"]) == 2
155 |                 and float(item["segment"][1]) - float(item["segment"][0]) <= max_duration
156 |             ]
157 |         copied["annotations"] = annotations
158 |         output[str(video_id)] = copied
159 |     _require(output, "stratified annotation is empty")
160 |     path.write_text(json.dumps({"database": output}, sort_keys=True), encoding="utf-8")
161 | 
162 | 
163 | def _write_predictions(path: Path, source: str | Path, videos: set[str] | None) -> None:
164 |     results = prediction_results(source)
165 |     if videos is not None:
166 |         results = {video: rows for video, rows in results.items() if video in videos}
167 |     _require(results, "stratified prediction is empty")
168 |     path.write_text(json.dumps({"results": results}, sort_keys=True), encoding="utf-8")
169 | 
170 | 
171 | def _delta_row(bootstrap: Mapping[str, Any], family: str, metric: str) -> dict[str, float]:
172 |     row = bootstrap["comparisons"][family][metric]
173 |     return {
174 |         "point_pp": float(
175 |             (
176 |                 bootstrap["point_estimates"][family][metric]
177 |                 - bootstrap["point_estimates"][bootstrap["baseline_family"]][metric]
178 |             )
179 |             * 100.0
180 |         ),
181 |         "ci_lower_pp": float(row["ci_lower_exact_rank"] * 100.0),
182 |         "ci_upper_pp": float(row["ci_upper_exact_rank"] * 100.0),
183 |         "delta_samples": row["delta_samples"],
184 |     }
185 | 
186 | 
187 | def evaluate_strata(
188 |     *,
189 |     frozen_path: str | Path,
190 |     validation_identity_path: str | Path,
191 |     annotation_path: str | Path,
192 |     on_prediction_path: str | Path,
193 |     gate_zero_prediction_path: str | Path,
194 |     nonce: str,
195 |     workers: int,
196 |     evaluator_thread: int = 16,
197 |     chunksize: int = 1,
198 | ) -> dict[str, Any]:
199 |     frozen = _load(frozen_path)
200 |     identity = _load(validation_identity_path)
201 |     annotation = _load(annotation_path)
202 |     _require(
203 |         frozen.get("schema_version") == "duca_h65_singleclock_training_strata_freeze_v1"
204 |         and frozen.get("validation_or_test_used") is False,
205 |         "strata freeze is not training-only",
206 |     )
207 |     video_distortion = _validation_video_distortion(identity)
208 |     low = {
209 |         video for video, value in video_distortion.items()
210 |         if value <= float(frozen["distortion_q25"])
211 |     }
212 |     high = {
213 |         video for video, value in video_distortion.items()
214 |         if value >= float(frozen["distortion_q75"])
215 |     }
216 |     _require(low and high and low.isdisjoint(high), "validation distortion strata are invalid")
217 | 
218 |     def cfg(path: Path) -> dict[str, Any]:
219 |         return {
220 |             "type": "mAP",
221 |             "ground_truth_filename": str(path),
222 |             "subset": "validation",
223 |             "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
224 |             "top_k": None,
225 |             "blocked_videos": None,
226 |             "thread": int(evaluator_thread),
227 |         }
228 | 
229 |     with tempfile.TemporaryDirectory(prefix="duca-h65-singleclock-strata-") as directory:
230 |         root = Path(directory)
231 |         predictions = {}
232 |         for stratum, videos in (("all", None), ("low", low), ("high", high)):
233 |             predictions[stratum] = {}
234 |             for family, source in (("on", on_prediction_path), ("gate_zero", gate_zero_prediction_path)):
235 |                 path = root / f"{stratum}_{family}.json"
236 |                 _write_predictions(path, source, videos)
237 |                 predictions[stratum][family] = path
238 |         short_annotation = root / "short_annotation.json"
239 |         low_annotation = root / "low_annotation.json"
240 |         high_annotation = root / "high_annotation.json"
241 |         _write_annotation(
242 |             short_annotation,
243 |             annotation,
244 |             max_duration=float(frozen["short_action_duration_q25_seconds"]),
245 |         )
246 |         _write_annotation(low_annotation, annotation, videos=low)
247 |         _write_annotation(high_annotation, annotation, videos=high)
248 |         short = bootstrap_h65_official_map(
249 |             predictions["all"], cfg(short_annotation), baseline_family="gate_zero",
250 |             nonce=nonce, namespace="SHORT_ACTION_Q25_V1", workers=workers,
251 |             chunksize=chunksize,
252 |         )
253 |         low_result = bootstrap_h65_official_map(
254 |             predictions["low"], cfg(low_annotation), baseline_family="gate_zero",
255 |             nonce=nonce, namespace="DISTORTION_LOW_Q1_V1", workers=workers,
256 |             chunksize=chunksize,
257 |         )
258 |         high_result = bootstrap_h65_official_map(
259 |             predictions["high"], cfg(high_annotation), baseline_family="gate_zero",
260 |             nonce=nonce, namespace="DISTORTION_HIGH_Q4_V1", workers=workers,
261 |             chunksize=chunksize,
262 |         )
263 | 
264 |     short_row = _delta_row(short, "on", "average_mAP")
265 |     low_row = _delta_row(low_result, "on", "average_mAP")
266 |     high_row = _delta_row(high_result, "on", "average_mAP")
267 |     interaction = np.asarray(high_row.pop("delta_samples")) - np.asarray(low_row.pop("delta_samples"))
268 |     interaction_lower, interaction_upper = exact_interval(interaction, lower_rank=250, upper_rank=9750)
269 |     short_row.pop("delta_samples")
270 |     return {
271 |         "schema_version": "duca_h65_singleclock_strata_v1",
272 |         "primary_checkpoint_state_key": "state_dict_ema",
273 |         "training_freeze_path": str(Path(frozen_path).resolve()),
274 |         "training_freeze_sha256": sha256_file(frozen_path),
275 |         "validation_identity_path": str(Path(validation_identity_path).resolve()),
276 |         "validation_identity_sha256": sha256_file(validation_identity_path),
277 |         "validation_or_test_used_for_cutpoints": False,
278 |         "short_action_delta_pp": short_row["point_pp"],
279 |         "short_action": short_row,
280 |         "distortion_low": low_row,
281 |         "distortion_high": high_row,
282 |         "distortion_interaction_point_pp": high_row["point_pp"] - low_row["point_pp"],
283 |         "distortion_interaction_ci_lower_pp": float(interaction_lower * 100.0),
284 |         "distortion_interaction_ci_upper_pp": float(interaction_upper * 100.0),
285 |         "low_distortion_videos": sorted(low),
286 |         "high_distortion_videos": sorted(high),
287 |         "all_predictions_retained_as_potential_false_positives_for_short_action": True,
288 |         "official_evaluator_reexecuted_per_resample": True,
289 |         "samples": 10000,
290 |         "nonce": nonce,
291 |         "bootstrap_execution": {
292 |             "workers": int(workers),
293 |             "evaluator_thread": int(evaluator_thread),
294 |             "chunksize": int(chunksize),
295 |         },
296 |     }
297 | 
298 | 
299 | def parse_args():
300 |     parser = argparse.ArgumentParser(description="Freeze/evaluate H65 SingleClock strata")
301 |     sub = parser.add_subparsers(dest="command", required=True)
302 |     freeze = sub.add_parser("freeze")
303 |     freeze.add_argument("--training-identity", required=True)
304 |     freeze.add_argument("--annotation", required=True)
305 |     freeze.add_argument("--output", required=True)
306 |     evaluate = sub.add_parser("evaluate")
307 |     evaluate.add_argument("--frozen", required=True)
308 |     evaluate.add_argument("--validation-identity", required=True)
309 |     evaluate.add_argument("--annotation", required=True)
310 |     evaluate.add_argument("--on-prediction", required=True)
311 |     evaluate.add_argument("--gate-zero-prediction", required=True)
312 |     evaluate.add_argument("--nonce", required=True)
313 |     evaluate.add_argument("--workers", type=int, default=1)
314 |     evaluate.add_argument("--evaluator-thread", type=int, default=16)
315 |     evaluate.add_argument("--chunksize", type=int, default=1)
316 |     evaluate.add_argument("--output", required=True)
317 |     return parser.parse_args()
318 | 
319 | 
320 | def main():
321 |     args = parse_args()
322 |     if args.command == "freeze":
323 |         payload = freeze_training_strata(
324 |             training_identity_path=args.training_identity,
325 |             annotation_path=args.annotation,
326 |         )
327 |     else:
328 |         payload = evaluate_strata(
329 |             frozen_path=args.frozen,
330 |             validation_identity_path=args.validation_identity,
331 |             annotation_path=args.annotation,
332 |             on_prediction_path=args.on_prediction,
333 |             gate_zero_prediction_path=args.gate_zero_prediction,
334 |             nonce=args.nonce,
335 |             workers=args.workers,
336 |             evaluator_thread=args.evaluator_thread,
337 |             chunksize=args.chunksize,
338 |         )
339 |     atomic_write_json(args.output, payload)
340 | 
341 | 
342 | if __name__ == "__main__":
343 |     main()
```

### File: ../OpenTAD_DUCA_H65_FirstMixSingleClock_20260824/tools/bata/bootstrap_duca_h65_official_map.py
Lines: 1-433
```python
  1 | from __future__ import annotations
  2 | 
  3 | import argparse
  4 | from concurrent.futures import ProcessPoolExecutor
  5 | import hashlib
  6 | import json
  7 | from pathlib import Path
  8 | from typing import Any, Mapping
  9 | 
 10 | import numpy as np
 11 | 
 12 | from tools.bata.duca_p0_evaluation import (
 13 |     EXPECTED_TIOU_THRESHOLDS,
 14 |     _metrics_from_evaluator,
 15 |     canonical_sha256,
 16 |     evaluation_video_ids,
 17 |     normalize_evaluation_config,
 18 |     official_evaluator_identity,
 19 |     prediction_results,
 20 |     recompute_official_map,
 21 |     sha256_file,
 22 | )
 23 | from tools.bata.duca_p0_training import atomic_write_json
 24 | 
 25 | 
 26 | _WORKER_STATE: dict[str, Any] | None = None
 27 | _METRIC_KEYS = ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")
 28 | 
 29 | 
 30 | def seed_from_nonce(nonce: str, namespace: str) -> tuple[int, str]:
 31 |     nonce = str(nonce).strip()
 32 |     namespace = str(namespace).strip()
 33 |     if not nonce or not namespace:
 34 |         raise ValueError("bootstrap nonce and namespace must be nonempty")
 35 |     digest = hashlib.sha256(f"{nonce}\n{namespace}".encode("utf-8")).digest()
 36 |     return int.from_bytes(digest[:8], byteorder="big", signed=False), digest.hex()
 37 | 
 38 | 
 39 | def exact_interval(values: list[float] | np.ndarray, *, lower_rank: int, upper_rank: int) -> tuple[float, float]:
 40 |     ordered = np.sort(np.asarray(values, dtype=np.float64))
 41 |     if ordered.ndim != 1 or ordered.size == 0:
 42 |         raise ValueError("exact interval requires a nonempty vector")
 43 |     lower_rank = int(lower_rank)
 44 |     upper_rank = int(upper_rank)
 45 |     if not 1 <= lower_rank <= upper_rank <= ordered.size:
 46 |         raise ValueError("exact interval ranks lie outside the bootstrap sample")
 47 |     return float(ordered[lower_rank - 1]), float(ordered[upper_rank - 1])
 48 | 
 49 | 
 50 | def resolve_sample_range(
 51 |     samples: int, sample_start: int = 0, sample_stop: int | None = None
 52 | ) -> tuple[int, int]:
 53 |     """Resolve one deterministic half-open shard of the frozen draw matrix."""
 54 |     samples = int(samples)
 55 |     sample_start = int(sample_start)
 56 |     sample_stop = samples if sample_stop is None else int(sample_stop)
 57 |     if not 0 <= sample_start < sample_stop <= samples:
 58 |         raise ValueError(
 59 |             "bootstrap sample shard must satisfy 0 <= start < stop <= samples"
 60 |         )
 61 |     return sample_start, sample_stop
 62 | 
 63 | 
 64 | def _evaluate_draw(
 65 |     draw: tuple[str, ...],
 66 |     *,
 67 |     families: tuple[str, ...],
 68 |     database: Mapping[str, Any],
 69 |     predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
 70 |     evaluation_config: Mapping[str, Any],
 71 |     ground_truth_path: Path,
 72 | ) -> tuple[tuple[float, ...], ...]:
 73 |     from opentad.evaluations.mAP import mAP
 74 | 
 75 |     synthetic_database: dict[str, Any] = {}
 76 |     synthetic_predictions = {family: {} for family in families}
 77 |     for draw_index, video_id in enumerate(draw):
 78 |         synthetic_id = f"bootstrap_{draw_index:05d}_{video_id}"
 79 |         synthetic_database[synthetic_id] = dict(database[video_id])
 80 |         for family in families:
 81 |             synthetic_predictions[family][synthetic_id] = [
 82 |                 dict(item) for item in predictions[family].get(video_id, [])
 83 |             ]
 84 |     ground_truth_path.write_text(
 85 |         json.dumps({"database": synthetic_database}, sort_keys=True), encoding="utf-8"
 86 |     )
 87 |     kwargs = dict(evaluation_config)
 88 |     kwargs.pop("type")
 89 |     kwargs["ground_truth_filename"] = str(ground_truth_path)
 90 |     kwargs["blocked_videos"] = None
 91 |     rows = []
 92 |     for family in families:
 93 |         evaluator = mAP(prediction_filename={"results": synthetic_predictions[family]}, **kwargs)
 94 |         metrics = _metrics_from_evaluator(evaluator)
 95 |         rows.append(tuple(float(metrics[key]) for key in _METRIC_KEYS))
 96 |     return tuple(rows)
 97 | 
 98 | 
 99 | def _evaluate_draw_in_memory(
100 |     draw: tuple[str, ...],
101 |     *,
102 |     families: tuple[str, ...],
103 |     database: Mapping[str, Any],
104 |     predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
105 |     evaluation_config: Mapping[str, Any],
106 | ) -> tuple[tuple[float, ...], ...]:
107 |     """Re-execute the official AP core without per-draw JSON or subprocess setup.
108 | 
109 |     The table construction below mirrors ``mAP._import_ground_truth`` and
110 |     ``mAP._import_prediction`` exactly.  Synthetic video ids retain cluster
111 |     multiplicity when the same source video is drawn more than once.
112 |     """
113 |     import pandas as pd
114 | 
115 |     from opentad.evaluations.builder import remove_duplicate_annotations
116 |     from opentad.evaluations.mAP import compute_average_precision_detection
117 | 
118 |     activity_index: dict[str, int] = {}
119 |     gt_video: list[str] = []
120 |     gt_start: list[float] = []
121 |     gt_end: list[float] = []
122 |     gt_label: list[int] = []
123 |     synthetic_ids: list[str] = []
124 |     for draw_index, video_id in enumerate(draw):
125 |         synthetic_id = f"bootstrap_{draw_index:05d}_{video_id}"
126 |         synthetic_ids.append(synthetic_id)
127 |         annotations = remove_duplicate_annotations(database[video_id]["annotations"])
128 |         for annotation in annotations:
129 |             label = annotation["label"]
130 |             if label not in activity_index:
131 |                 activity_index[label] = len(activity_index)
132 |             gt_video.append(synthetic_id)
133 |             gt_start.append(float(annotation["segment"][0]))
134 |             gt_end.append(float(annotation["segment"][1]))
135 |             gt_label.append(activity_index[label])
136 |     if not activity_index:
137 |         raise ValueError("formal DUCA bootstrap draw contains no evaluation classes")
138 |     ground_truth = pd.DataFrame(
139 |         {
140 |             "video-id": gt_video,
141 |             "t-start": gt_start,
142 |             "t-end": gt_end,
143 |             "label": gt_label,
144 |         }
145 |     )
146 |     thresholds = np.asarray(evaluation_config["tiou_thresholds"], dtype=np.float64)
147 |     rows = []
148 |     for family in families:
149 |         pred_video: list[str] = []
150 |         pred_start: list[float] = []
151 |         pred_end: list[float] = []
152 |         pred_label: list[int] = []
153 |         pred_score: list[float] = []
154 |         for synthetic_id, video_id in zip(synthetic_ids, draw):
155 |             for prediction in predictions[family].get(video_id, []):
156 |                 pred_video.append(synthetic_id)
157 |                 pred_start.append(float(prediction["segment"][0]))
158 |                 pred_end.append(float(prediction["segment"][1]))
159 |                 pred_label.append(
160 |                     activity_index.get(prediction["label"], len(activity_index))
161 |                 )
162 |                 pred_score.append(prediction["score"])
163 |         prediction = pd.DataFrame(
164 |             {
165 |                 "video-id": pred_video,
166 |                 "t-start": pred_start,
167 |                 "t-end": pred_end,
168 |                 "label": pred_label,
169 |                 "score": pred_score,
170 |             }
171 |         )
172 |         ap = np.zeros((len(thresholds), len(activity_index)), dtype=np.float64)
173 |         for cidx in activity_index.values():
174 |             gt_idx = ground_truth["label"] == cidx
175 |             pred_idx = prediction["label"] == cidx
176 |             ap[:, cidx] = compute_average_precision_detection(
177 |                 ground_truth.loc[gt_idx].reset_index(drop=True),
178 |                 prediction.loc[pred_idx].reset_index(drop=True),
179 |                 tiou_thresholds=thresholds,
180 |             )
181 |         maps = ap.mean(axis=1)
182 |         metrics = {"average_mAP": float(maps.mean())}
183 |         for threshold, value in zip(thresholds, maps):
184 |             metrics[f"mAP@{float(threshold)}"] = float(value)
185 |         rows.append(tuple(float(metrics[key]) for key in _METRIC_KEYS))
186 |     return tuple(rows)
187 | 
188 | 
189 | def _initialize_worker(
190 |     families: tuple[str, ...],
191 |     database: Mapping[str, Any],
192 |     predictions: Mapping[str, Mapping[str, list[dict[str, Any]]]],
193 |     evaluation_config: Mapping[str, Any],
194 | ) -> None:
195 |     global _WORKER_STATE
196 |     _WORKER_STATE = {
197 |         "families": families,
198 |         "database": database,
199 |         "predictions": predictions,
200 |         "evaluation_config": evaluation_config,
201 |     }
202 | 
203 | 
204 | def _evaluate_draw_in_worker(draw: tuple[str, ...]) -> tuple[tuple[float, ...], ...]:
205 |     if _WORKER_STATE is None:
206 |         raise RuntimeError("H65 bootstrap worker was not initialized")
207 |     return _evaluate_draw_in_memory(draw, **_WORKER_STATE)
208 | 
209 | 
210 | def bootstrap_h65_official_map(
211 |     prediction_paths: Mapping[str, str | Path],
212 |     evaluation_config: Any,
213 |     *,
214 |     baseline_family: str,
215 |     nonce: str,
216 |     namespace: str,
217 |     samples: int = 10000,
218 |     lower_rank: int = 250,
219 |     upper_rank: int = 9750,
220 |     workers: int = 1,
221 |     chunksize: int = 1,
222 |     sample_start: int = 0,
223 |     sample_stop: int | None = None,
224 | ) -> dict[str, Any]:
225 |     families = tuple(str(key) for key in prediction_paths)
226 |     if baseline_family not in families or len(families) < 2:
227 |         raise ValueError("bootstrap requires a baseline and at least one comparison")
228 |     samples = int(samples)
229 |     if samples != 10000:
230 |         raise ValueError("H65 terminal bootstrap is frozen at exactly 10,000 resamples")
231 |     if (int(lower_rank), int(upper_rank)) != (250, 9750):
232 |         raise ValueError("H65 terminal bootstrap interval is frozen at ranks 250/9750")
233 |     workers = int(workers)
234 |     if workers < 1 or workers > 64:
235 |         raise ValueError("bootstrap workers must lie in [1,64]")
236 |     chunksize = int(chunksize)
237 |     if chunksize < 1:
238 |         raise ValueError("bootstrap chunksize must be positive")
239 |     sample_start, sample_stop = resolve_sample_range(
240 |         samples, sample_start=sample_start, sample_stop=sample_stop
241 |     )
242 |     shard_samples = sample_stop - sample_start
243 | 
244 |     cfg = normalize_evaluation_config(evaluation_config, expected_subset="validation")
245 |     evaluator_thread = int(cfg["thread"])
246 |     if evaluator_thread < 1:
247 |         raise ValueError("official evaluator thread count must be positive")
248 |     expected = evaluation_video_ids(cfg, expected_subset="validation")
249 |     annotation = json.loads(Path(cfg["ground_truth_filename"]).read_text(encoding="utf-8"))
250 |     database = annotation.get("database")
251 |     if not isinstance(database, Mapping) or any(video not in database for video in expected):
252 |         raise ValueError("bootstrap annotation does not cover the validation videos")
253 |     predictions = {family: prediction_results(path) for family, path in prediction_paths.items()}
254 |     expected_set = set(expected)
255 |     for family, rows in predictions.items():
256 |         extras = set(rows) - expected_set
257 |         if extras:
258 |             raise ValueError(f"{family} prediction contains out-of-scope videos: {sorted(extras)[:4]}")
259 | 
260 |     seed, seed_sha256 = seed_from_nonce(nonce, namespace)
261 |     generator = np.random.Generator(np.random.PCG64(seed))
262 |     indices = generator.integers(0, len(expected), size=(samples, len(expected)), dtype=np.int32)
263 |     draws = [
264 |         tuple(expected[int(index)] for index in row)
265 |         for row in indices[sample_start:sample_stop]
266 |     ]
267 |     sampled = {
268 |         family: {metric: [] for metric in _METRIC_KEYS}
269 |         for family in families
270 |     }
271 | 
272 |     if workers == 1:
273 |         kwargs = {
274 |             "families": families,
275 |             "database": database,
276 |             "predictions": predictions,
277 |             "evaluation_config": cfg,
278 |         }
279 |         iterator = (_evaluate_draw_in_memory(draw, **kwargs) for draw in draws)
280 |         _collect(iterator, sampled, families, shard_samples)
281 |     else:
282 |         with ProcessPoolExecutor(
283 |             max_workers=workers,
284 |             initializer=_initialize_worker,
285 |             initargs=(families, database, predictions, cfg),
286 |         ) as executor:
287 |             iterator = executor.map(
288 |                 _evaluate_draw_in_worker,
289 |                 draws,
290 |                 chunksize=chunksize,
291 |             )
292 |             _collect(iterator, sampled, families, shard_samples)
293 | 
294 |     common = {
295 |         "paired_video_cluster_bootstrap": True,
296 |         "rng": "numpy.random.PCG64",
297 |         "nonce": nonce,
298 |         "namespace": namespace,
299 |         "seed_uint64": seed,
300 |         "seed_sha256": seed_sha256,
301 |         "samples": samples,
302 |         "interval_rank_convention": "one_based_order_statistics",
303 |         "lower_rank": int(lower_rank),
304 |         "upper_rank": int(upper_rank),
305 |         "baseline_family": baseline_family,
306 |         "family_order": list(families),
307 |         "video_ids": list(expected),
308 |         "prediction_paths": {
309 |             key: str(Path(value).resolve()) for key, value in prediction_paths.items()
310 |         },
311 |         "prediction_sha256": {
312 |             key: sha256_file(value) for key, value in prediction_paths.items()
313 |         },
314 |         "evaluation_config": cfg,
315 |         "evaluation_config_sha256": canonical_sha256(cfg),
316 |         "evaluator": official_evaluator_identity(),
317 |     }
318 |     execution = {
319 |         "workers": workers,
320 |         "evaluator_thread_metadata": evaluator_thread,
321 |         "evaluator_thread_used_by_ap_core": False,
322 |         "chunksize": chunksize,
323 |         "result_order": "executor_map_input_order",
324 |         "engine": "official_compute_average_precision_detection_in_memory_v1",
325 |         "elided_operations": ["per_draw_json_roundtrip", "mAP_constructor"],
326 |     }
327 |     if (sample_start, sample_stop) != (0, samples):
328 |         return {
329 |             "schema_version": "duca_h65_official_pcg64_video_bootstrap_shard_v1",
330 |             "official_evaluator_reexecuted_per_resample": True,
331 |             **common,
332 |             "sample_start": sample_start,
333 |             "sample_stop": sample_stop,
334 |             "shard_samples": shard_samples,
335 |             "sampled_metrics": sampled,
336 |             "execution": execution,
337 |         }
338 | 
339 |     comparisons: dict[str, Any] = {}
340 |     for family in families:
341 |         if family == baseline_family:
342 |             continue
343 |         metric_rows = {}
344 |         for metric in _METRIC_KEYS:
345 |             delta = np.asarray(sampled[family][metric]) - np.asarray(sampled[baseline_family][metric])
346 |             lower, upper = exact_interval(delta, lower_rank=lower_rank, upper_rank=upper_rank)
347 |             metric_rows[metric] = {
348 |                 "delta_samples": [float(value) for value in delta],
349 |                 "delta_mean": float(delta.mean()),
350 |                 "ci_lower_exact_rank": lower,
351 |                 "ci_upper_exact_rank": upper,
352 |             }
353 |         comparisons[family] = metric_rows
354 | 
355 |     point_estimates = {
356 |         family: recompute_official_map(path, cfg, expected_subset="validation")["metrics"]
357 |         for family, path in prediction_paths.items()
358 |     }
359 |     return {
360 |         "schema_version": "duca_h65_official_pcg64_video_bootstrap_v1",
361 |         "official_evaluator_reexecuted_per_resample": True,
362 |         **common,
363 |         "execution": execution,
364 |         "point_estimates": point_estimates,
365 |         "sampled_metrics": sampled,
366 |         "comparisons": comparisons,
367 |     }
368 | 
369 | 
370 | def _collect(iterator, sampled, families, samples):
371 |     for sample_index, rows in enumerate(iterator, start=1):
372 |         for family, values in zip(families, rows):
373 |             for metric, value in zip(_METRIC_KEYS, values):
374 |                 sampled[family][metric].append(float(value))
375 |         if sample_index % 1000 == 0 or sample_index == samples:
376 |             print(f"[DUCA_H65_BOOTSTRAP] {sample_index}/{samples}", flush=True)
377 | 
378 | 
379 | def _parse_prediction(value: str) -> tuple[str, str]:
380 |     if "=" not in value:
381 |         raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
382 |     family, path = value.split("=", 1)
383 |     if not family or not path:
384 |         raise argparse.ArgumentTypeError("prediction must be FAMILY=PATH")
385 |     return family, path
386 | 
387 | 
388 | def parse_args(argv: list[str] | None = None):
389 |     parser = argparse.ArgumentParser(description="Exact H65 paired official-mAP bootstrap")
390 |     parser.add_argument("--prediction", action="append", type=_parse_prediction, required=True)
391 |     parser.add_argument("--baseline", required=True)
392 |     parser.add_argument("--annotation", required=True)
393 |     parser.add_argument("--nonce", required=True)
394 |     parser.add_argument("--namespace", required=True)
395 |     parser.add_argument("--workers", type=int, default=1)
396 |     parser.add_argument("--evaluator-thread", type=int, default=16)
397 |     parser.add_argument("--chunksize", type=int, default=1)
398 |     parser.add_argument("--sample-start", type=int, default=0)
399 |     parser.add_argument("--sample-stop", type=int)
400 |     parser.add_argument("--output", required=True)
401 |     return parser.parse_args(argv)
402 | 
403 | 
404 | def main():
405 |     args = parse_args()
406 |     prediction_paths = dict(args.prediction)
407 |     if len(prediction_paths) != len(args.prediction):
408 |         raise ValueError("prediction family names must be unique")
409 |     evaluation_config = {
410 |         "type": "mAP",
411 |         "ground_truth_filename": args.annotation,
412 |         "subset": "validation",
413 |         "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
414 |         "top_k": None,
415 |         "blocked_videos": None,
416 |         "thread": args.evaluator_thread,
417 |     }
418 |     payload = bootstrap_h65_official_map(
419 |         prediction_paths,
420 |         evaluation_config,
421 |         baseline_family=args.baseline,
422 |         nonce=args.nonce,
423 |         namespace=args.namespace,
424 |         workers=args.workers,
425 |         chunksize=args.chunksize,
426 |         sample_start=args.sample_start,
427 |         sample_stop=args.sample_stop,
428 |     )
429 |     atomic_write_json(args.output, payload)
430 | 
431 | 
432 | if __name__ == "__main__":
433 |     main()
```

### File: ../OpenTAD_DUCA_H65_FirstMixSingleClock_20260824/tests/test_duca_h65_singleclock_finalizer.py
Lines: 1-346
```python
  1 | import json
  2 | import hashlib
  3 | 
  4 | import pytest
  5 | 
  6 | from tools.bata.finalize_duca_h65_singleclock_terminal import (
  7 |     _identity_equal,
  8 |     _twin_execution_contract_ok,
  9 |     finalize,
 10 | )
 11 | 
 12 | 
 13 | def _bootstrap(names, points):
 14 |     sampled = {}
 15 |     point_estimates = {}
 16 |     for name in names:
 17 |         sampled[name] = {}
 18 |         point_estimates[name] = {}
 19 |         for metric, value in points[name].items():
 20 |             sampled[name][metric] = [value] * 10000
 21 |             point_estimates[name][metric] = value
 22 |     return {
 23 |         "samples": 10000,
 24 |         "lower_rank": 250,
 25 |         "upper_rank": 9750,
 26 |         "sampled_metrics": sampled,
 27 |         "point_estimates": point_estimates,
 28 |     }
 29 | 
 30 | 
 31 | def _identity(path):
 32 |     path.write_text(
 33 |         json.dumps(
 34 |             {
 35 |                 "sample_count": 1,
 36 |                 "total_input_exposure_count": 1,
 37 |                 "unique_physical_window_count": 1,
 38 |                 "duplicate_exposure_count": 0,
 39 |                 "duplicate_samples": [],
 40 |                 "records": [
 41 |                     {
 42 |                         "sample_id": "v|window_start_frame=0",
 43 |                         "video_name": "v",
 44 |                         "window_start_frame": 0,
 45 |                         "selected_valid_len": 384,
 46 |                         "dense_valid_len": 768,
 47 |                         "selected_positions": list(range(384)),
 48 |                         "selected_rgb_sha256": "r",
 49 |                         "selected_positions_sha256": "p",
 50 |                         "selected_mask_sha256": "m",
 51 |                     }
 52 |                 ],
 53 |             }
 54 |         ),
 55 |         encoding="utf-8",
 56 |     )
 57 | 
 58 | 
 59 | def _sha256(path):
 60 |     return hashlib.sha256(path.read_bytes()).hexdigest()
 61 | 
 62 | 
 63 | def _metrics(path, checkpoint, state_key):
 64 |     path.write_text(
 65 |         json.dumps(
 66 |             {
 67 |                 "checkpoint_path": str(checkpoint.resolve()),
 68 |                 "checkpoint_sha256": _sha256(checkpoint),
 69 |                 "checkpoint_epoch": 59,
 70 |                 "checkpoint_state_key": state_key,
 71 |             }
 72 |         ),
 73 |         encoding="utf-8",
 74 |     )
 75 | 
 76 | 
 77 | def _configs(tmp_path):
 78 |     root = tmp_path / "configs" / "adatad" / "thumos"
 79 |     root.mkdir(parents=True)
 80 |     on = root / "duca_h65_first_singleclock_cycle4.py"
 81 |     zero = root / "duca_h65_first_singleclock_cycle4_gate_zero.py"
 82 |     off = root / "duca_sampling_rate_curriculum_stage2_joint384.py"
 83 |     on.write_text("single_clock_gate_zero = False\n", encoding="utf-8")
 84 |     zero.write_text("single_clock_gate_zero = True\n", encoding="utf-8")
 85 |     off.write_text("single_clock = False\n", encoding="utf-8")
 86 |     return on, zero, off
 87 | 
 88 | 
 89 | def _config_row(path, gate_zero):
 90 |     return {
 91 |         "config_path": str(path),
 92 |         "config_sha256": _sha256(path),
 93 |         "single_clock_gate_zero": gate_zero,
 94 |     }
 95 | 
 96 | 
 97 | def _family_row(metrics, config, gate_zero, identity=None):
 98 |     row = {
 99 |         "metrics_path": str(metrics),
100 |         "metrics_sha256": _sha256(metrics),
101 |         **_config_row(config, gate_zero),
102 |     }
103 |     if identity is not None:
104 |         row["selected_input_identity_path"] = str(identity)
105 |         row["selected_input_identity_sha256"] = _sha256(identity)
106 |     return row
107 | 
108 | 
109 | def test_twin_execution_contract_rejects_a_mislabeled_gate_zero(tmp_path):
110 |     on, zero, _ = _configs(tmp_path)
111 |     assert _twin_execution_contract_ok(
112 |         _config_row(on, False), _config_row(zero, True)
113 |     )
114 |     assert not _twin_execution_contract_ok(
115 |         _config_row(on, False), _config_row(zero, False)
116 |     )
117 | 
118 | 
119 | def _finalizer_fixture(tmp_path):
120 |     on_config, zero_config, off_config = _configs(tmp_path)
121 |     clock_checkpoint = tmp_path / "clock_epoch_59.pth"
122 |     off_checkpoint = tmp_path / "off_epoch_59.pth"
123 |     stage1_checkpoint = tmp_path / "stage1_epoch_29.pth"
124 |     clock_checkpoint.write_bytes(b"clock")
125 |     off_checkpoint.write_bytes(b"off")
126 |     stage1_checkpoint.write_bytes(b"stage1")
127 |     final_on = tmp_path / "final_on.json"
128 |     final_zero = tmp_path / "final_zero.json"
129 |     ema_on = tmp_path / "ema_on.json"
130 |     ema_zero = tmp_path / "ema_zero.json"
131 |     for path in (final_on, final_zero, ema_on, ema_zero):
132 |         _identity(path)
133 |     final_on_metrics = tmp_path / "final_on_metrics.json"
134 |     final_zero_metrics = tmp_path / "final_zero_metrics.json"
135 |     ema_on_metrics = tmp_path / "ema_on_metrics.json"
136 |     ema_zero_metrics = tmp_path / "ema_zero_metrics.json"
137 |     off_final_metrics = tmp_path / "off_final_metrics.json"
138 |     off_ema_metrics = tmp_path / "off_ema_metrics.json"
139 |     _metrics(final_on_metrics, clock_checkpoint, "state_dict")
140 |     _metrics(final_zero_metrics, clock_checkpoint, "state_dict")
141 |     _metrics(ema_on_metrics, clock_checkpoint, "state_dict_ema")
142 |     _metrics(ema_zero_metrics, clock_checkpoint, "state_dict_ema")
143 |     _metrics(off_final_metrics, off_checkpoint, "state_dict")
144 |     _metrics(off_ema_metrics, off_checkpoint, "state_dict_ema")
145 |     eval_commit = "e" * 40
146 |     receipt = {
147 |         "schema_version": "duca_h65_singleclock_terminal_eval_receipt_v1",
148 |         "git_commit": eval_commit,
149 |         "clock_checkpoint": str(clock_checkpoint.resolve()),
150 |         "clock_checkpoint_sha256": _sha256(clock_checkpoint),
151 |         "h65_off_checkpoint": str(off_checkpoint.resolve()),
152 |         "h65_off_checkpoint_sha256": _sha256(off_checkpoint),
153 |         "stage1_checkpoint": str(stage1_checkpoint.resolve()),
154 |         "stage1_checkpoint_sha256": _sha256(stage1_checkpoint),
155 |         "families": {
156 |             "final_on": _family_row(final_on_metrics, on_config, False, final_on),
157 |             "final_gate_zero": _family_row(final_zero_metrics, zero_config, True, final_zero),
158 |             "ema_on": _family_row(ema_on_metrics, on_config, False, ema_on),
159 |             "ema_gate_zero": _family_row(ema_zero_metrics, zero_config, True, ema_zero),
160 |             "h65_off_final": _family_row(off_final_metrics, off_config, None),
161 |             "h65_off_ema": _family_row(off_ema_metrics, off_config, None),
162 |         }
163 |     }
164 |     stage1_sha = _sha256(stage1_checkpoint)
165 |     clock = {
166 |         "family": "clock_on", "checkpoint_epoch": 59,
167 |         "successful_optimizer_updates": 6000, "scheduler_last_epoch": 6000,
168 |         "stage1_checkpoint_sha256": stage1_sha, "stage1_checkpoint_epoch": 29,
169 |         "single_clock_values": {"state_dict": {"clock": 0.1}, "state_dict_ema": {"clock": 0.1}},
170 |         "recovery_state_complete": True,
171 |         "recovery_protocol_deviation": [],
172 |     }
173 |     off = dict(
174 |         clock,
175 |         family="h65_off",
176 |         single_clock_values={
177 |             "state_dict": {"registered_clock": 0.0},
178 |             "state_dict_ema": {"registered_clock": 0.0},
179 |         },
180 |     )
181 |     metrics = ("average_mAP", "mAP@0.6", "mAP@0.7")
182 |     final_points = {
183 |         "final_on": dict.fromkeys(metrics, 0.66),
184 |         "final_gate_zero": dict.fromkeys(metrics, 0.65),
185 |         "h65_off_final": dict.fromkeys(metrics, 0.65),
186 |     }
187 |     ema_points = {
188 |         "ema_on": dict.fromkeys(metrics, 0.67),
189 |         "ema_gate_zero": dict.fromkeys(metrics, 0.66),
190 |         "h65_off_ema": dict.fromkeys(metrics, 0.66),
191 |     }
192 |     old_points = {
193 |         "truetime": dict.fromkeys(metrics, 0.62),
194 |         "rankpack": dict.fromkeys(metrics, 0.61),
195 |     }
196 |     return {
197 |         "receipt": receipt,
198 |         "clock_audit": clock,
199 |         "off_audit": off,
200 |         "final_bootstrap": _bootstrap(tuple(final_points), final_points),
201 |         "ema_bootstrap": _bootstrap(tuple(ema_points), ema_points),
202 |         "old_pair_bootstrap": _bootstrap(tuple(old_points), old_points),
203 |         "strata": {
204 |             "schema_version": "duca_h65_singleclock_strata_v1",
205 |             "primary_checkpoint_state_key": "state_dict_ema",
206 |             "short_action_delta_pp": 0.1,
207 |             "distortion_interaction_point_pp": 0.2,
208 |         },
209 |         "cost": {
210 |             "schema_version": "duca_h65_singleclock_cost_pair_v1",
211 |             "median_latency_ratio_on_over_gate_zero": 1.0,
212 |             "p90_latency_ratio_on_over_gate_zero": 1.0,
213 |             "peak_memory_ratio_on_over_gate_zero": 1.0,
214 |         },
215 |         "stage1_average_map": 0.594231,
216 |         "expected_eval_commit": eval_commit,
217 |     }
218 | 
219 | 
220 | def test_finalizer_accepts_frozen_positive_gate(tmp_path):
221 |     result = finalize(**_finalizer_fixture(tmp_path))
222 |     assert result["decision"] == "CONTINUE_TO_REPLICATION"
223 |     assert result["identity_gate_pass"] is True
224 |     assert result["twin_execution_contract_pass"] is True
225 |     assert result["family_execution_contract_pass"] is True
226 |     assert result["clock_recovery_contract_pass"] is True
227 |     assert result["h65_off_recovery_contract_pass"] is True
228 |     assert result["paper_claim_admissible"] is True
229 |     assert result["bridge_authorized"] is False
230 | 
231 | 
232 | def test_finalizer_rejects_metrics_changed_after_receipt(tmp_path):
233 |     kwargs = _finalizer_fixture(tmp_path)
234 |     metrics_path = kwargs["receipt"]["families"]["ema_gate_zero"]["metrics_path"]
235 |     with open(metrics_path, "a", encoding="utf-8") as stream:
236 |         stream.write("\n")
237 |     with pytest.raises(ValueError, match="metrics hash mismatch"):
238 |         finalize(**kwargs)
239 | 
240 | 
241 | def test_finalizer_keeps_positive_mechanism_diagnostic_but_blocks_replication_for_legacy_off_recovery_gap(tmp_path):
242 |     kwargs = _finalizer_fixture(tmp_path)
243 |     kwargs["off_audit"]["recovery_state_complete"] = False
244 |     kwargs["off_audit"]["recovery_protocol_deviation"] = [
245 |         "rng_state",
246 |         "data_loader_state",
247 |     ]
248 |     result = finalize(**kwargs)
249 |     assert result["decision"] == "REVISE_WITHOUT_MORE_TIME_MODULES"
250 |     assert result["checkpoint_audit_gate_pass"] is True
251 |     assert result["h65_off_recovery_contract_pass"] is False
252 |     assert result["h65_off_recovery_protocol_deviation"] == [
253 |         "rng_state",
254 |         "data_loader_state",
255 |     ]
256 |     assert result["paper_claim_admissible"] is False
257 | 
258 | 
259 | def test_finalizer_hard_fails_when_clock_recovery_state_is_incomplete(tmp_path):
260 |     kwargs = _finalizer_fixture(tmp_path)
261 |     kwargs["clock_audit"]["recovery_state_complete"] = False
262 |     kwargs["clock_audit"]["recovery_protocol_deviation"] = [
263 |         "rng_state",
264 |         "data_loader_state",
265 |     ]
266 |     result = finalize(**kwargs)
267 |     assert result["decision"] == "PIVOT_TO_ACQUISITION_OR_TRAINING_MATURITY"
268 |     assert result["checkpoint_audit_gate_pass"] is True
269 |     assert result["clock_recovery_contract_pass"] is False
270 |     assert result["paper_claim_admissible"] is False
271 | def test_identity_accounting_mismatch_rejected():
272 |     from tools.bata.finalize_duca_h65_singleclock_terminal import _identity_equal
273 | 
274 |     base = {"sample_count": 1, "records": [_record()], "total_input_exposure_count": 2,
275 |             "unique_physical_window_count": 1, "duplicate_exposure_count": 1,
276 |             "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}]}
277 |     changed = dict(base)
278 |     changed["duplicate_exposure_count"] = 0
279 |     assert not _identity_equal(base, changed)
280 | 
281 | 
282 | def test_identity_accounting_requires_explicit_fields_and_consistency():
283 |     from tools.bata.finalize_duca_h65_singleclock_terminal import _identity_equal
284 | 
285 |     valid = {
286 |         "sample_count": 2,
287 |         "total_input_exposure_count": 3,
288 |         "unique_physical_window_count": 2,
289 |         "duplicate_exposure_count": 1,
290 |         "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}],
291 |         "records": [_record(), _record("v|window_start_frame=1", start=1)],
292 |     }
293 |     assert _identity_equal(valid, dict(valid))
294 |     missing = dict(valid)
295 |     del missing["duplicate_samples"]
296 |     assert not _identity_equal(valid, missing)
297 |     malformed = dict(valid, total_input_exposure_count=2)
298 |     assert not _identity_equal(valid, malformed)
299 |     malformed = dict(valid, duplicate_samples=[{"sample_id": "", "duplicate_exposure_count": 1}])
300 |     assert not _identity_equal(valid, malformed)
301 | 
302 | 
303 | def _record(sample_id="v|window_start_frame=0", video_name="v", start=0, positions=None):
304 |     positions = list(range(2)) if positions is None else positions
305 |     return {
306 |         "sample_id": sample_id, "video_name": video_name, "window_start_frame": start,
307 |         "selected_valid_len": len(positions), "dense_valid_len": 4,
308 |         "selected_positions": positions, "selected_rgb_sha256": "r",
309 |         "selected_positions_sha256": "p", "selected_mask_sha256": "m",
310 |     }
311 | 
312 | 
313 | @pytest.mark.parametrize("mutate", [
314 |     lambda p: p.update(records=[]),
315 |     lambda p: p["records"].append(_record("v|window_start_frame=2", start=2)),
316 |     lambda p: p["records"].reverse(),
317 |     lambda p: p["records"][0].update(sample_id="wrong"),
318 |     lambda p: p["records"][0].update(selected_positions=[0, 0]),
319 | ])
320 | def test_identity_records_validator_rejects_malformed_payload(mutate):
321 |     payload = {"sample_count": 2, "total_input_exposure_count": 2,
322 |                "unique_physical_window_count": 2, "duplicate_exposure_count": 0,
323 |                "duplicate_samples": [],
324 |                "records": [_record(), _record("v|window_start_frame=1", start=1)]}
325 |     mutate(payload)
326 |     assert not _identity_equal(payload, payload)
327 | 
328 | 
329 | def test_identity_records_accept_valid_duplicate_payload():
330 |     payload = {"sample_count": 1, "total_input_exposure_count": 2,
331 |                "unique_physical_window_count": 1, "duplicate_exposure_count": 1,
332 |                "duplicate_samples": [{"sample_id": "v|window_start_frame=0", "duplicate_exposure_count": 1}],
333 |                "records": [_record()]}
334 |     assert _identity_equal(payload, dict(payload))
335 | def test_identity_records_validator_rejects_position_at_dense_boundary():
336 |     payload = {"sample_count": 1, "total_input_exposure_count": 1,
337 |                "unique_physical_window_count": 1, "duplicate_exposure_count": 0,
338 |                "duplicate_samples": [], "records": [_record(positions=[0, 4])]}
339 |     assert not _identity_equal(payload, payload)
340 | 
341 | 
342 | def test_identity_records_validator_accepts_last_dense_position():
343 |     payload = {"sample_count": 1, "total_input_exposure_count": 1,
344 |                "unique_physical_window_count": 1, "duplicate_exposure_count": 0,
345 |                "duplicate_samples": [], "records": [_record(positions=[0, 3])]}
346 |     assert _identity_equal(payload, payload)
```

### File: ../OpenTAD_DUCA_H65_FirstMixSingleClock_20260824/tests/test_duca_h65_singleclock_strata.py
Lines: 1-66
```python
 1 | import json
 2 | 
 3 | from tools.bata.analyze_duca_h65_singleclock_strata import (
 4 |     _record_distortion,
 5 |     _uniform_positions,
 6 |     freeze_training_strata,
 7 | )
 8 | 
 9 | 
10 | def test_uniform_distortion_is_exactly_zero():
11 |     positions = _uniform_positions(768, 384)
12 |     record = {
13 |         "selected_positions": positions.tolist(),
14 |         "selected_valid_len": 384,
15 |         "dense_valid_len": 768,
16 |     }
17 |     assert _record_distortion(record) == 0.0
18 | 
19 | 
20 | def test_partial_trailing_tubelet_does_not_create_clock_support():
21 |     record = {
22 |         "selected_positions": [0, 4, 9],
23 |         "selected_valid_len": 3,
24 |         "dense_valid_len": 10,
25 |     }
26 |     assert _record_distortion(record) == 0.0
27 | 
28 | 
29 | def test_freeze_uses_training_only_q1_and_video_distortion(tmp_path):
30 |     identity = {
31 |         "records": [
32 |             {
33 |                 "video_name": f"train_{index}",
34 |                 "selected_positions": _uniform_positions(8, 4).tolist(),
35 |                 "selected_valid_len": 4,
36 |                 "dense_valid_len": 8,
37 |             }
38 |             for index in range(4)
39 |         ]
40 |     }
41 |     annotation = {
42 |         "database": {
43 |             **{
44 |                 f"train_{index}": {
45 |                     "subset": "training",
46 |                     "annotations": [{"segment": [0.0, float(index + 1)]}],
47 |                 }
48 |                 for index in range(4)
49 |             },
50 |             "validation_0": {
51 |                 "subset": "validation",
52 |                 "annotations": [{"segment": [0.0, 100.0]}],
53 |             },
54 |         }
55 |     }
56 |     identity_path = tmp_path / "identity.json"
57 |     annotation_path = tmp_path / "annotation.json"
58 |     identity_path.write_text(json.dumps(identity), encoding="utf-8")
59 |     annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
60 |     frozen = freeze_training_strata(
61 |         training_identity_path=identity_path,
62 |         annotation_path=annotation_path,
63 |     )
64 |     assert frozen["short_action_duration_q25_seconds"] == 1.75
65 |     assert frozen["distortion_q75"] == 0.0
66 |     assert frozen["validation_or_test_used"] is False
```

## Answer

# DUCA H65 First-Mixing SingleClock：Unit-1 终态门操作化裁决

**Nonce：** `DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824`

## 0. 唯一裁决

# `REVISE_GATE_IMPLEMENTATION`

不选择 `STOP_UNDERDEFINED_GATE`。

已接受的 Unit‑1 科学门本身足够明确：主对照、三项 `−0.20 pp` 非劣界、terminal EMA 主检查点、H65 replay 身份和 nominal-uniform bit identity 均已冻结。当前问题不是科学路线缺失，而是终结器把另一个更苛刻的“正增益、coadaptation、cost、旧配对证据”门错误写进了 `main_pass`。

本轮只应修正离线终结逻辑和统计定义，不改模型、不改训练、不重跑任何训练、不引入 Query 或 dynamic‑K。当前研究边界仍是固定 K384 的 H65-compatible 表示归因门。 

---

## 1. 当前终结器为何不符合已接受合同

| 当前实现                                                         | 判定                                                                                                   |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 以 `EMA ON − same-checkpoint gate-zero` 至少 `+0.50 pp` 为主门     | **错误因果估计量**。gate-zero 是对已经与 SingleClock 共同训练的 checkpoint 做推理期 post-treatment 消融，不能替代 H65 OFF/replay。 |
| 要求 `EMA ON − H65 OFF >= +0.50 pp`                            | **擅自把非劣门提高为优效门**。已接受界限是 `>= −0.20 pp`。                                                               |
| 要求 Avg-mAP 的 bootstrap `LCB95 > 0`                           | **事后提高门槛**。原合同未要求 CI 通过。                                                                             |
| 将 gate-zero 与 H65 OFF 的 coadaptation CI 限定在 ±0.20 pp         | **新增未注册等价门**，不得进入 Unit‑1 决策。                                                                         |
| `cost_pass` 使用 1.01/1.02/1.02 阈值并进入 `hard_fail`              | **凭空发明 Unit‑1 cost kill gate**。                                                                      |
| 用旧 RankPack/TrueTime 配对、Stage‑1 maturity 决定当前 Unit‑1         | **估计量越界**。它们不是该 Unit‑1 的通过条件。                                                                        |
| 要求 final 与 EMA 方向一致                                          | **新增门槛**。原合同指定 terminal final‑EMA 为 primary，final 仅需报告。                                              |
| 要求 SingleClock 标量在 final 和 EMA 中非零                           | **新增“参数必须被使用”门**。参数为零可削弱机制解释，但不违反非劣合同。                                                               |
| `strata_pass` 使用短动作 `−0.50 pp` 和 distortion interaction `>0` | **不是已接受的 boundary-error 条款**。现有 distortion 也不是 gap‑CV。                                               |
| 单种子通过后可令 `paper_claim_admissible=True`                       | **明确错误**。Unit‑1 永远只是单种子 development representation gate。                                             |

因此，现有 `CONTINUE_TO_REPLICATION / REVISE_WITHOUT_MORE_TIME_MODULES / PIVOT...` 三分支全部应从 Unit‑1 终结器移除。它们重新打开了路线，而本轮只允许产生 Unit‑1 的 PASS/KILL。

---

## 2. 冻结的比较角色

### 2.1 唯一主对照

主估计量固定为：

[
\Delta_m^{EMA}
==============

100\left[
m(\texttt{ema_on})
------------------

m(\texttt{h65_off_ema})
\right]\ \text{pp}
]

其中：

* `ema_on`：SingleClock ON 新训练的 epoch‑59 `state_dict_ema`；
* `h65_off_ema`：历史 H65 epoch‑59 EMA checkpoint 在当前兼容代码中的 OFF/replay；
* (m\in{\text{Avg-mAP},\text{mAP@0.6},\text{mAP@0.7}})。

**Unit‑1 主门只能读取这三个 EMA ON-vs-H65 OFF/replay point delta。**

### 2.2 same-checkpoint gate-zero 的角色

`ema_gate_zero` 和 `final_gate_zero` 必须：

* 加载与 ON 完全相同的 SingleClock checkpoint；
* 使用相同 `state_dict` 或 `state_dict_ema`；
* 只在推理时令 SingleClock residual gate 为零；
* selected indices、gathered RGB、mask 和 VideoMAE 输入保持相同。

它们只回答：

> 已共同训练的模型在推理时是否实际依赖 SingleClock 路径，以及非时钟参数发生了多少 coadaptation。

它们**不是**主基线，不承担非劣门、正增益门、等价门或 Unit‑2 准入门。ON−gate-zero 即使为负，也不能单独 KILL；即使为 `+1 pp`，也不能弥补 ON−H65 OFF 低于 `−0.20 pp`。

### 2.3 final 的角色

* `state_dict_ema` 是唯一 primary；
* `state_dict` final 只报告 checkpoint sensitivity、EMA 滞后或训练末端不稳定；
* 不要求 final 与 EMA 同方向；
* final 低于门槛不能覆盖 EMA PASS，final 高于门槛也不能挽救 EMA KILL。

---

## 3. 三类身份检查必须分开

### 3.1 H65 replay 身份：主比较的证据准入条件

当前 `h65_off_ema` replay 必须与历史冻结 H65 EMA reference 在以下五个边界一致：

1. selected integer indices；
2. gathered RGB tensor；
3. VideoMAE input tensor；
4. detector raw selected‑q proposals、scores、labels；
5. canonicalized official evaluator JSON。

要求：

* shape、dtype、ordering 和 canonical bytes 一致；
* tensor 使用 SHA‑256 或逐字节比较；
* evaluator JSON 先按冻结 canonical serialization 规范化，再比较字节；
* checkpoint、config、split、class map、NMS、evaluator 和 annotation hash 全部绑定。

H65 replay 身份失败时，**不得解释任何 ON−OFF mAP**。终结器应非零退出并写一个非科学判决错误码：

`INVALID_H65_REPLAY_IDENTITY`

这不是 `KILL_SINGLECLOCK_REPRESENTATION`，因为它没有证明 SingleClock 有害；它只说明基线身份不可继承。该错误不得被压缩成第三个科学 decision token。

### 3.2 ON 与 gate-zero 的输入身份：执行合同

对 final 和 EMA 分别要求：

* 同一 checkpoint 路径/hash/epoch/state key；
* ON 与 gate-zero 的 selected indices、RGB、positions、valid mask、VideoMAE input tensor 完全相同；
* 唯一运行差异是 clock residual 是否启用。

失败时使用：

`INVALID_SAME_CHECKPOINT_GATE_ZERO_EXECUTION`

同样不产生 PASS/KILL，因为这意味着消融臂未正确构造。

### 3.3 nominal-uniform bit identity：真正的硬 KILL 条件

在 exact canonical-uniform positions 下：

* relative clock residual 必须严格为零；
* relative bias tensor 必须严格为零；
* 修改后的第一处 temporal-mixing 输出必须与 gate-zero/H65 path bit-identical；
* final VideoMAE backbone output 必须 bit-identical；
* 不允许使用数值容忍、近似零或“mAP 相同”替代 bit identity。

若该条件失败，证据本身可以有效，且直接输出：

`KILL_SINGLECLOCK_REPRESENTATION`

因为这证明所谓“零残差保持 H65 identity”的表示合同不成立。

---

## 4. 三项 `−0.20 pp` 主指标只使用 point estimate

### 4.1 精确门限

三个条件均为包含等号的 point-estimate 非劣门：

[
\Delta_{\text{Avg}}^{EMA}\ge -0.20\text{ pp}
]

[
\Delta_{0.6}^{EMA}\ge -0.20\text{ pp}
]

[
\Delta_{0.7}^{EMA}\ge -0.20\text{ pp}
]

* 恰好 `−0.20 pp`：PASS；
* `−0.2001 pp`：KILL；
* 比较前不得四舍五入到两位小数；
* 建议将官方 JSON 中的规范十进制文本转为 `Decimal` 后计算 pp delta，并与 `Decimal("-0.20")` 比较，避免二进制浮点边界歧义。

### 4.2 bootstrap 的角色

保留现有 10,000 次 paired whole-video cluster bootstrap：

* RNG：`numpy.random.PCG64`；
* paired resampling unit：whole video；
* 同一 draw 同时用于 ON 与 H65 OFF；
* interval：一基排序统计量 rank 250 和 9750；
* CI 仍写入终态收据。

但是：

> **CI 完全不进入 Unit‑1 PASS/KILL。**

因此，即使 point delta 为 `−0.10 pp` 而 `LCB95 < −0.20 pp`，主指标仍通过；即使 point delta 为 `−0.21 pp` 而 CI 包含零，仍直接 KILL。

要求 `LCB95 >= −0.20` 或 `LCB95 > 0` 都会提高已接受门槛，禁止采用。

---

## 5. boundary error 的唯一冻结定义

现有两个候选中，选择：

# **同类、score-ranked、IoU≥0.5、一对一匹配**

它比“每个 GT 独立选最高 IoU proposal”更能防止同一个 proposal 被重复用于多个 GT，也能显式处理 matched recall。

### 5.1 使用哪一状态的 predictions

边界分析使用：

* `ema_on` 的最终 official per-video prediction JSON；
* `h65_off_ema` 的最终 official per-video prediction JSON。

两者必须已完成冻结的：

1. selected-q → physical-seconds 映射；
2. score filtering；
3. top-k；
4. IoU/NMS/voting；
5. official serialization。

不再施加额外 score cutoff、额外 top-k 或第二次 NMS。`gate_zero` prediction 不参与 boundary hard gate。

### 5.2 proposal–GT matching

对每个 video、每个 class 独立执行：

1. GT 使用与 official evaluator 相同的 duplicate-removal 规则；
2. predictions 按以下稳定顺序排序：

   * score 降序；
   * start second 升序；
   * end second 升序；
   * 原始 serialized row index 升序；
3. 逐 prediction 处理；
4. 在尚未匹配的同类 GT 中寻找最大 temporal IoU；
5. 仅当最大 IoU `>=0.5` 时建立匹配；
6. 最大 IoU 并列时，按 GT 的 `(start, end, canonical_occurrence_index)` 升序决胜；
7. 每个 prediction 和每个 GT 至多使用一次。

匹配先在该 video/class 的**全部 GT**上完成，再从中抽取高 gap‑CV 或高 boundary-density stratum 的 GT，避免一个 prediction 在不同子集分析中被重复分配。

### 5.3 unmatched GT 的处理

不能只报告 matched-pair MAE，否则模型可以通过漏掉困难 GT 人为降低误差。

对 GT (g=[s_g,e_g])，令 (d_g=e_g-s_g>0)。

若匹配 prediction (p=[s_p,e_p])：

[
e_s(g)=\min\left(1,\frac{|s_p-s_g|}{d_g}\right)
]

[
e_e(g)=\min\left(1,\frac{|e_p-e_g|}{d_g}\right)
]

若 GT 未匹配：

[
e_s(g)=e_e(g)=1
]

单 GT boundary error：

[
e(g)=\frac{e_s(g)+e_e(g)}{2}
]

这是一个包含漏检惩罚的、duration-normalized、lower-is-better 指标。它是 Unit‑1 hard boundary statistic。

同时必须报告但不单独过门：

* matched recall；
* matched start MAE seconds；
* matched end MAE seconds；
* matched mean boundary MAE seconds。

False positives 不直接进入该 error，但已经受到三项 official mAP 主指标惩罚，并通过 score-ranked matching 影响匹配过程。

### 5.4 每视频聚合

对某一 stratum (S)，每个拥有至少一个 (S)-GT 的视频计算：

[
E_v^S=\frac{1}{|G_v^S|}\sum_{g\in G_v^S}e(g)
]

全体 point statistic 为视频等权平均：

[
E^S=\frac{1}{|V_S|}\sum_{v\in V_S}E_v^S
]

配对差定义为：

[
\Delta E^S=E_{\mathrm{ON}}^S-E_{\mathrm{H65\ OFF}}^S
]

正值表示 SingleClock 边界误差恶化。

---

## 6. gap‑CV 与 boundary-density 的冻结定义

### 6.1 共同数据边界

所有 cutpoint 必须在读取 validation prediction 或 validation metric 之前，由 official **training population** 冻结。

使用：

* H65 OFF/replay 的 training selected-position identity；
* official training annotations；
* 冻结的 training window ledger；
* unique physical windows，重复 exposure 按 `sample_id` 去重；
* `numpy.quantile(..., method="linear")`。

冻结并保存 q25、q50、q75；hard high stratum 只使用 q75。validation/test 不能参与 cutpoint。

### 6.2 gap‑CV

粒度：**window-level**。

对一个 window 的 valid selected positions：

[
p_0<p_1<\cdots<p_{K_v-1}
]

VideoMAE temporal tubelet size 固定为 2。仅使用完整 tubelet：

[
c_i=\frac{p_{2i}+p_{2i+1}}{2}
]

[
g_i=c_{i+1}-c_i
]

定义 population CV：

[
\operatorname{gapCV}(w)
=======================

\frac{
\sqrt{\frac{1}{n}\sum_i(g_i-\bar g)^2}
}{
\bar g
}
]

约束：

* 计算使用 `float64`；
* 只使用 valid prefix；
* padding、重复填充帧和 invalid mask slots 全部排除；
* trailing incomplete tubelet 排除；
* 位置不严格递增或平均 gap 非正时，身份证据无效；
* 由于 CV 对统一尺度不变，可直接使用 physical dense frame indices，不依赖 FPS。

training high threshold：

[
\tau_{\mathrm{gap}}=
Q_{0.75}^{train}(\operatorname{gapCV})
]

validation high-gap window：

[
\operatorname{gapCV}(w)\ge\tau_{\mathrm{gap}}
]

阈值相等的窗口全部进入 high stratum；不得为了接近 25% 数量而打破 ties。

### 6.3 boundary-density

粒度：**window-level**。

对 window 的真实有效物理时间区间 ([t_w^s,t_w^e))：

* 只计 original GT start/end boundaries；
* 不把跨窗口动作裁剪点制造为新 boundary；
* 每个动作最多贡献一个 start 和一个 end；
* 位于 video final endpoint 的 end boundary，只在该视频最后一个有效窗口中计入；
* padding 时长不进入分母。

定义：

[
\operatorname{BD}(w)
====================

\frac{
#{b:\ b\text{ 是 GT start/end 且位于 window 有效支持}}
}{
t_w^e-t_w^s
}
]

单位为 boundaries/second。

training high threshold：

[
\tau_{\mathrm{BD}}=
Q_{0.75}^{train}(\operatorname{BD})
]

validation high-boundary-density window：

[
\operatorname{BD}(w)\ge\tau_{\mathrm{BD}}
]

### 6.4 从 window stratum 到 GT stratum

一个 GT instance 属于 high-gap stratum，当且仅当它的 start 或 end boundary 至少落入一个 high-gap window。

同理，一个 GT instance 属于 high-boundary-density stratum，当且仅当它的 start 或 end boundary至少落入一个 high-BD window。

同一 GT 被多个重叠窗口覆盖时，在 GT 集合中只保留一次。

### 6.5 AND、OR 还是分别过门

冻结为：

# **两个 stratum 分别过门，并且二者为合取关系。**

即：

[
\Delta E^{high\ gapCV}\le0
]

且

[
\Delta E^{high\ BD}\le0
]

两者都满足才算 boundary gate PASS。

不使用：

* high-gap 与 high-BD 的 OR 联合集；
* 二者的 AND 交集作为唯一门；
* validation 上选择表现更好的那个 stratum。

二者交集可以作为附加机制诊断，但不影响 PASS/KILL。

---

## 7. “不恶化”的统计规则

hard boundary rule 使用 **point delta**：

[
\Delta E^S \le 0
]

* 等于 0：PASS；
* 任意严格正值：KILL；
* 不引入 `+0.01`、`+0.05` 等新容忍 margin；
* 不要求 bootstrap `UCB95 <= 0`，因为这会提高原合同；
* 不用 bootstrap CI 挽救 point worsening。

仍执行 10,000 次 paired video-cluster bootstrap：

* eligible video 为不可拆分 cluster；
* 每次重采样视频后，对抽中的 per-video errors 取等权平均；
* ON 与 H65 OFF 使用同一视频 draw；
* PCG64；
* seed 由本 nonce 与固定 namespace 派生；
* 95% interval 使用 rank 250/9750。

建议固定 namespace：

* `UNIT1_BOUNDARY_HIGH_GAPCV_Q75_V1`
* `UNIT1_BOUNDARY_HIGH_BD_Q75_V1`

CI 只用于报告不确定性和后续论文设计，不进入 Unit‑1 判定。

---

## 8. 当前 artifact 不足时的唯一合法降级

### 8.1 可离线计算 boundary gate 的充分条件

必须同时已有并封存：

1. `ema_on` final official prediction JSON；
2. `h65_off_ema` final official prediction JSON；
3. H65 OFF/replay training selected-position identities；
4. H65 OFF/replay validation selected-position identities；
5. training/validation window 的真实有效物理时间 ledger；
6. official training/validation GT annotations；
7. prediction、annotation、identity、checkpoint、config 和 evaluator hash 绑定。

若全部存在，只做离线统计；不得重训，不得改变 checkpoint，不得重新选择阈值。

### 8.2 不足时禁止的动作

若任一必要 artifact 缺失：

* 不得为补 boundary 诊断重训；
* 不得用 gate-zero prediction 替代 H65 OFF prediction；
* 不得用 ON training identity 替代 H65 OFF training identity；
* 不得从 validation 分布估计 q75；
* 不得重新跑模型推理只为了补充未预先封存的 window-level diagnostic；
* 不得从现有 mAP 或 strata 结果反推阈值。

### 8.3 降级结果

此时输出：

`boundary_gate_status = NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP`

并将 boundary 条款降为 diagnostic-only。

Unit‑1 的科学判定只使用：

* H65 replay 身份；
* same-checkpoint execution identity；
* nominal-uniform bit identity；
* 三项 EMA ON−H65 OFF point non-inferiority metrics。

这不是把缺失证据当成 boundary PASS。必须同时输出：

* `boundary_mechanism_claim_supported = false`
* `boundary_gate_used_for_decision = false`
* 明确列出缺失 artifact。

在这一降级分支中，只要三项主指标与 hard identity 通过，Unit‑1 仍可 PASS；但不得声称 SingleClock 已证明保护高 gap-CV 或高 boundary-density 边界。

---

## 9. cost、paper claim 和 Unit‑2 准入

### 9.1 cost

Unit‑1 cost 的唯一角色是：

# **报告项，不是 hard kill gate，也不是 Unit‑2 前置门。**

因此：

* 删除 `median<=1.01`、`p90<=1.02`、`memory<=1.02` 对 decision 的影响；
* `--cost` 应改为可选输入；
* 有合法 cost receipt 时原样报告；
* cost 缺失或比例较高不能改变 Unit‑1 PASS/KILL；
* cost 可在以后 full-stack/paper admission 中单独审查，但不得追溯重写 Unit‑1。

### 9.2 paper claim

Unit‑1 无论 PASS 或 KILL：

```text
paper_claim_admissible = false
claim_boundary = single_seed_unit1_development_representation_gate_only
```

不得根据 `CONTINUE_TO_REPLICATION`、checkpoint recovery 或 cost receipt 把它设为 true。

### 9.3 Unit‑2 Query residual 何时允许进入 Builder

只有以下条件全部成立时，才可设置：

`unit2_query_builder_eligible = true`

条件：

1. Unit‑1 输出 `PASS_UNIT1_SINGLECLOCK_GATE`；
2. H65 replay identity 有效；
3. same-checkpoint gate-zero execution 有效；
4. nominal-uniform bit identity 通过；
5. 三项 EMA point non-inferiority 通过；
6. boundary gate 可计算时，两项 boundary delta 均 `<=0`；
7. boundary 不可计算时已按本规格显式降级，且没有伪造 PASS；
8. focused unit tests 全部通过；
9. 独立 Critic 返回 `UNIT1_GATE_IMPLEMENTATION_PASS`；
10. terminal v2 artifact、输入 hashes 和 first-failure ledger 已封存。

这只授权实现已接受的 Unit‑2 Query semantic residual。它不授权 dynamic‑K：

```text
dynamic_k_authorized = false
```

若 Unit‑1 KILL，Query residual 不得用于“修复”SingleClock。

---

## 10. 终结器的唯一决策算法

```python
# Evidence-binding failures do not produce a scientific PASS/KILL token.
require(h65_replay_identity_valid)
require(same_checkpoint_on_gatezero_execution_valid)
require(checkpoint_config_evaluator_bindings_valid)
require(primary_state_key == "state_dict_ema")

metric_pass = (
    delta_avg_pp >= Decimal("-0.20")
    and delta_06_pp >= Decimal("-0.20")
    and delta_07_pp >= Decimal("-0.20")
)

if boundary_evaluable:
    boundary_pass = (
        high_gapcv_boundary_error_delta_point <= 0.0
        and high_boundary_density_error_delta_point <= 0.0
    )
else:
    boundary_pass = True  # diagnostic downgrade, not positive boundary evidence

kill = (
    not nominal_uniform_backbone_bit_identical
    or not metric_pass
    or (boundary_evaluable and not boundary_pass)
)

decision_token = (
    "KILL_SINGLECLOCK_REPRESENTATION"
    if kill
    else "PASS_UNIT1_SINGLECLOCK_GATE"
)
```

以下变量不得出现在 `kill` 或 `metric_pass` 中：

* ON−gate-zero 增益；
* coadaptation CI；
* old RankPack/TrueTime evidence；
* Stage‑1 average mAP 或 maturity；
* final-vs-EMA direction；
* clock parameter 是否非零；
* short-action `−0.50 pp`；
* high-low interaction；
* cost；
* H65 historical recovery-state completeness；
* paper claim readiness。

---

## 11. 终态 JSON 的最小字段

```json
{
  "schema_version": "duca_h65_singleclock_unit1_terminal_gate_v2",
  "nonce": "DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824",
  "evidence_status": "VALID",
  "decision_token": "PASS_UNIT1_SINGLECLOCK_GATE",
  "primary_checkpoint_state_key": "state_dict_ema",
  "primary_comparison": "ema_on_minus_h65_off_ema",

  "identity": {
    "h65_replay_five_boundary_pass": true,
    "same_checkpoint_gatezero_execution_pass": true,
    "nominal_uniform_backbone_bit_identical": true
  },

  "thresholds_pp": {
    "average_mAP": -0.20,
    "mAP@0.6": -0.20,
    "mAP@0.7": -0.20,
    "comparison": "inclusive_point_estimate"
  },

  "primary_metrics": {
    "average_mAP": {
      "point_delta_pp": 0.0,
      "point_gate_pass": true,
      "ci_lower_pp_report_only": 0.0,
      "ci_upper_pp_report_only": 0.0
    },
    "mAP@0.6": {},
    "mAP@0.7": {}
  },

  "boundary_gate": {
    "status": "EVALUABLE",
    "used_for_decision": true,
    "comparison": "ema_on_minus_h65_off_ema",
    "high_gapcv_delta_point": 0.0,
    "high_gapcv_pass": true,
    "high_boundary_density_delta_point": 0.0,
    "high_boundary_density_pass": true,
    "bootstrap_samples": 10000,
    "bootstrap_cluster": "whole_video",
    "ci_role": "report_only"
  },

  "diagnostics": {
    "final_on_vs_h65_off": {},
    "ema_on_vs_same_checkpoint_gatezero": {},
    "final_on_vs_same_checkpoint_gatezero": {},
    "gatezero_vs_h65_off_coadaptation": {},
    "short_action": {},
    "gapcv_interaction": {}
  },

  "cost": {
    "decision_role": "report_only",
    "status": "AVAILABLE"
  },

  "claim_boundary": "single_seed_unit1_development_representation_gate_only",
  "paper_claim_admissible": false,
  "unit2_query_builder_eligible": true,
  "dynamic_k_authorized": false,
  "first_failure": null
}
```

若证据身份无效：

* `evidence_status="INVALID"`；
* `decision_token` 必须为 `null`；
* 进程非零退出；
* 写明唯一 first failure；
* 不得输出一个伪装成科学结果的第三 decision token。

合法科学判决只有两个：

1. `PASS_UNIT1_SINGLECLOCK_GATE`
2. `KILL_SINGLECLOCK_REPRESENTATION`

---

## 12. 必须覆盖的 focused unit tests

| 测试                                                             | 预期                                                                |
| -------------------------------------------------------------- | ----------------------------------------------------------------- |
| 三项 point delta 都恰好 `−0.20 pp`                                  | PASS                                                              |
| 任一 point delta 为 `−0.2001 pp`                                  | KILL                                                              |
| point `−0.10 pp`，但 bootstrap LCB 为 `−0.50 pp`                  | PASS，CI 仅报告                                                       |
| point `−0.21 pp`，但 CI 包含零                                      | KILL                                                              |
| ON−gate-zero 为 `−2 pp`，ON−H65 OFF 三项通过                         | 不影响 PASS                                                          |
| ON−gate-zero 为 `+2 pp`，ON−H65 OFF 任一为 `−0.21 pp`               | KILL                                                              |
| final 三项失败但 EMA 三项通过                                           | PASS，final 仅诊断                                                    |
| EMA/final 方向不一致                                                | 不影响 decision                                                      |
| cost ratios 为 1.5/1.8/1.3                                      | cost 报告，不能 KILL                                                   |
| coadaptation CI 超出 ±0.20                                       | 只报告                                                               |
| old TrueTime/RankPack gate 失败                                  | 不影响 Unit‑1                                                        |
| clock alpha 恰好为 0 且有限                                          | 可报告机制未激活，但不能单独 KILL                                               |
| H65 replay selected indices 或 evaluator JSON 不同                | INVALID evidence，无 PASS/KILL                                      |
| ON 与 gate-zero checkpoint/state key 不同                         | INVALID evidence                                                  |
| ON 与 gate-zero selected RGB 不同                                 | INVALID execution                                                 |
| nominal-uniform first-mixing 或 backbone output 非 bit-identical | KILL                                                              |
| high-gap boundary delta `0`，high-BD delta `−0.01`              | boundary PASS                                                     |
| high-gap delta `+1e-12`                                        | boundary KILL，不加容忍 margin                                         |
| unmatched GT                                                   | start/end normalized error 各为 1                                   |
| 同一 prediction 可匹配两个 GT                                         | 测试必须证明第二个 GT 保持 unmatched                                         |
| score tie、IoU tie                                              | 按冻结稳定顺序产生唯一匹配                                                     |
| training q75 与 validation 分布变化                                 | q75 保持 training freeze，不重算                                        |
| boundary artifacts 缺失                                          | 主指标+identity 决策，boundary diagnostic unavailable                   |
| boundary 不可计算但代码将其默认为“通过证据”                                    | 测试必须失败；只能是 decision downgrade                                     |
| 单种子 PASS                                                       | `paper_claim_admissible` 仍为 false                                 |
| Unit‑1 KILL                                                    | `unit2_query_builder_eligible=false`、`dynamic_k_authorized=false` |

---

## 13. 最小代码改动

### `finalize_duca_h65_singleclock_terminal.py`

必须：

1. 将 primary delta 改为 `ema_on − h65_off_ema`；
2. 将三个 hard metric 条件改为 point `>=−0.20 pp`；
3. 删除 `+0.50 pp`、`LCB>0`、coadaptation equivalence、old-pair、Stage‑1 maturity、direction consistency 和 cost 对 decision 的影响；
4. final、gate-zero、old pair、short-action、interaction 和 cost 只放入 `diagnostics`；
5. `paper_claim_admissible` 固定为 false；
6. 输出唯一 PASS/KILL token；
7. 证据身份错误非零退出，不伪装为科学 KILL；
8. cost 参数改为 optional/report-only；
9. 不要求 clock scalar 非零；
10. H65 historical recovery-state 缺口不得替代 replay identity。

### `analyze_duca_h65_singleclock_strata.py`

必须：

1. hard comparison 从 `ON − gate-zero` 改为 `ON EMA − H65 OFF/replay EMA`；
2. 新增 training-only gap‑CV q25/q50/q75；
3. 新增 training-only boundary-density q25/q50/q75；
4. 实现上述 score-ranked same-class IoU≥0.5 one-to-one matcher；
5. 实现 unmatched-penalized duration-normalized boundary error；
6. 分别输出 high-gap 和 high-BD point delta；
7. 现有 distortion、short-action official-mAP 和 high-low interaction保留为 diagnostics，不生成 Unit‑1 hard `strata_pass`；
8. schema 升级为 v2，并显式记录 cutpoint 未使用 validation/test。

### `bootstrap_duca_h65_official_map.py`

三项 official mAP bootstrap 可保持不变。

boundary error 不是 official mAP，建议新增一个很小的离线 helper，复用：

* `seed_from_nonce`；
* PCG64；
* 10,000 draws；
* whole-video paired resampling；
* ranks 250/9750。

不得修改 official AP 核心来伪装 boundary metric 为官方指标。

### 测试文件

更新现有 finalizer/strata tests，并新增：

* boundary matching；
* unmatched penalty；
* q75 training-only freeze；
* separate conjunctive strata；
* artifact-gap downgrade；
* cost/CI/gate-zero 不进入 decision。

---

## 14. 独立审查与执行链

### Builder

只修改终结器、离线统计器和 focused tests。不得修改：

* SingleClock 模型；
* selector；
* K；
  -训练 config；
* checkpoint；
* prediction；
* evaluator；
* Query；
* dynamic‑K；
* cost threshold。

### Critic

只审查以下六项：

1. primary estimand 是否唯一为 EMA ON−H65 OFF/replay；
2. 三项门是否为 inclusive point `−0.20 pp`；
3. gate-zero、final、CI、cost 是否完全退出 hard decision；
4. H65 replay invalid 与 representation KILL 是否正确区分；
5. boundary strata 是否 training-only、分别过门且 sign 正确；
6. 单 seed 是否永远 `paper_claim_admissible=false`。

唯一 closure token：

`UNIT1_GATE_IMPLEMENTATION_PASS`

或

`UNIT1_GATE_IMPLEMENTATION_BLOCKED`

### Evaluator

在不重训、不改 prediction 的条件下：

* 验证输入 hashes；
* 运行已有 primary bootstrap 或读取已封存 artifact；
* 若充分 artifacts 存在，只运行新的离线 boundary statistics；
* 若不足，封存 diagnostic downgrade；
* 生成 v2 terminal artifact；
* 不选择阈值，不解释 Query，不授权 dynamic‑K。

---

## 15. 最终返回合同

```text
next_owner:
  BUILDER_DUCA_H65_SINGLECLOCK_UNIT1_GATE_V2

next_action:
  按本裁决修订 finalizer、离线 boundary analyzer 和 focused tests；
  不改模型或训练；随后交给独立 Critic 做单次 focused closure。

dependency:
  已封存的 six-family terminal receipt；
  H65 epoch-59 EMA replay 五边界身份；
  EMA ON/H65 OFF official metrics；
  nominal-uniform bit-identity receipt；
  boundary 可计算时还需 ON/H65 OFF EMA predictions、
  H65 OFF training/validation identities、window physical-time ledger 和 GT annotations。

expected_return_at:
  BUILDER_UNIT1_GATE_V2_PATCH_TEST_RECEIPT
  +
  CRITIC_UNIT1_GATE_IMPLEMENTATION_CLOSURE
  +
  UNIT1_TERMINAL_ADJUDICATION_V2
```

## 绑定结论

Unit‑1 不是“SingleClock 必须产生显著正增益”的门，而是：

> 在恢复 H65 replay 身份后，SingleClock terminal EMA 是否在 Avg-mAP、mAP@0.6 和 mAP@0.7 上均未低于 H65 超过 `0.20 pp`，是否保持 canonical-uniform bit identity，并在当前已有 artifact 足以合法计算时，不恶化两个预注册高风险边界 stratum。

因此必须删除现有正增益、CI、coadaptation、cost 和旧证据硬门。完成这一有限修订后，Unit‑1 可由代码无歧义地产生唯一 `PASS_UNIT1_SINGLECLOCK_GATE` 或 `KILL_SINGLECLOCK_REPRESENTATION`，而不重开任何 DUCA 路线。
