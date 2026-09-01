# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/c/6a952a19-9294-83ea-b09f-5524e7825316

## Prompt

Read every provided file completely. Independently reconcile the verified prior Pro freeze with the later human requirement for complete matched training and complete official held-out evaluation. Freeze exact data identities, the no-leakage sequence, and one executable current task without delegating scientific choices to Codex. Preserve nonce DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831 verbatim in the response.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_FULL_DATA_COMPARABLE_PROTOCOL-v001.md
Lines: 1-104
```md
  1 | # DUCA 多预算适应：完整训练与完整留出评测协议裁决
  2 | 
  3 | Nonce：`DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`
  4 | 
  5 | 你是本课题的科学第一负责人、整体科研流程维护者和论文首脑。你独立负责科学问题、创新机制、可证伪预测、
  6 | 实验路线、结果解释和论文主张。Codex 只执行你冻结后的最小实现、独立代码审查、正式实验评估和证据回传。
  7 | 本轮不要沿用 Codex 的预设选择；请独立解决一项会改变论文证据有效性的正式数据协议冲突。
  8 | 
  9 | ## 精确项目与最新 GitHub 代码身份
 10 | 
 11 | - ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
 12 | - Project URL：<https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92/project?tab=chats>
 13 | - GitHub 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
 14 | - H65 干净训练基座：
 15 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854>
 16 | - 最新 whole-video 诊断分支：
 17 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831>
 18 | - 最新诊断提交：
 19 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535>
 20 | - 真实变长执行与 whole-video runner：
 21 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/tools/bata/run_duca_whole_video_consistent_budget_falsifier.py>
 22 | - 三档预算实现：
 23 |   <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/33e4ed137c33eef07f0452b44506a6993bdf7535/opentad/models/duca/dynamic_budget.py>
 24 | 
 25 | `33e4ed...` 只提供已经验证的真实变长 VideoMAE 执行、packet 对齐、actual-observation 计数、K384 parity、
 26 | whole-video 评价和 producer 原始 proposal 顺序保持；新模型科学基座仍是 `04c35a3b...`。
 27 | 
 28 | ## 已完成的上一轮 Pro 裁决
 29 | 
 30 | 请完整读取随本 prompt 提供的上一轮 Pro 报告。它在同一精确 Project 中由 Pro 模型生成，nonce 为
 31 | `DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`。上一轮选择 `CONTINUE`，冻结了一个单变量实验：
 32 | 
 33 | - 固定控制只训练 K384；候选使用当前嵌套 K256/K384/K512 位置构造进行多预算训练；
 34 | - Scout、位置构造、检测器架构、损失、物理时间映射、Soft-NMS 和评价器不变；
 35 | - 两臂从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始；
 36 | - 每臂 6,000 次成功 optimizer update，训练状态重新初始化，terminal `state_dict_ema` 是唯一结果模型；
 37 | - 多预算概率按实际 observation 成本校准；
 38 | - 旧 704-state 冻结检测器路线继续 `STOP`；不加入 embedding、蒸馏、Gumbel、新 Scout/head、Mamba、
 39 |   Block Drop 或部署优化。
 40 | 
 41 | 这些机制和训练选择可以保留、修订或停止，但请自己判断。没有代码、训练或模型结果由上一轮报告自动产生。
 42 | 
 43 | ## 必须解决的后到人类约束
 44 | 
 45 | 上一轮 prompt 提交后，人类明确要求：
 46 | 
 47 | 1. 两个正式可比训练臂都必须使用项目定义的**完整 THUMOS14 训练集**，不能把 160/40、旧 40-video holdout、
 48 |    pilot、smoke 或 shortened run 作为论文主比较。
 49 | 2. 设计、超参数、规则、模型选择和停止条件冻结后，两臂必须在**完整官方 held-out evaluation/test split** 上使用
 50 |    相同数据语义、annotation、类别映射、Soft-NMS、评价器和结果保存协议完成最终比较。
 51 | 3. 官方留出集只用于冻结方案的最终评价，不能参与训练、checkpoint 选择、阈值/规则选择、路线选择、候选设计或
 52 |    反复窥视后修改方法。
 53 | 4. 训练侧子集若仍有必要，只能作为正式实验前的诊断，且之后必须用完整训练集重新训练两条匹配臂；诊断结果不得
 54 |    冒充论文主结果。
 55 | 
 56 | 上一轮 Pro 报告却冻结了 160-video train / 40-video development，并明确本轮不访问 official test。该数据协议因而
 57 | 不能直接交给 Builder。
 58 | 
 59 | ## 真实 split 身份冲突
 60 | 
 61 | 当前仓库和历史记录至少存在两套不能由 Codex静默等同的口径：
 62 | 
 63 | - OpenTAD/DUCA 实验常把 THUMOS14 `training` 用于训练、`validation` 用于官方评估，并记录 211 个成功评估视频；
 64 | - ActionFormer 官方协议常把 `validation` 用于训练、`test` 用于评估，历史材料记录 212 个带标注测试视频。
 65 | 
 66 | 请基于随附完整材料和代码身份，独立冻结：
 67 | 
 68 | - 两臂完整训练所使用的精确 config subset 名称、annotation 文件、完整视频 ID 集合和数量；
 69 | - 最终完整 held-out evaluation/test 所使用的精确 subset 名称、annotation、类别映射、完整视频 ID 集合和排除规则；
 70 | - 为什么这一选择与当前 H65/AdaTAD 公平基线可比；
 71 | - 对 211/212 差异的事实解释和唯一处理，不允许把未知信息写成已知；若材料不足，冻结一个在实现前必须完成的
 72 |   只读身份核验和明确阻断条件，而不是让 Codex猜测。
 73 | 
 74 | ## 你必须独立冻结的正式实验顺序
 75 | 
 76 | 请设计最小、论文优先且无测试集泄漏的顺序。至少明确：
 77 | 
 78 | 1. 是否保留任何训练侧诊断。如果保留，说明它只回答什么机制问题、何时封存，以及为什么不影响最终全量训练；
 79 | 2. 两条正式臂如何使用完整训练集，从哪个 checkpoint/state key 开始，是否保留每臂 6,000 次成功更新、相同
 80 |    optimizer/LR/EMA/seed/trainable-set 与 observation 成本匹配；
 81 | 3. 在打开完整 held-out 标签或指标之前，哪些代码、配置、checkpoint、概率、门槛和停止规则必须不可变；
 82 | 4. 完整 held-out 评估是一次性最终比较还是还需要另一外部确认集；不得要求用 held-out 结果继续调参；
 83 | 5. 对单种子结果、配对不确定性、可发表性和后续是否训练 controller 的证据边界；
 84 | 6. 明确的停止条件：什么结果关闭当前 K256/K384/K512 适应路线，什么结果才允许下一轮 controller 研究。
 85 | 
 86 | 不要为了形式完整而加入工作流平台、复杂合同代码、通用审计框架、额外哈希系统、Mamba、Gumbel、蒸馏、
 87 | 新 selector、Block Drop、TensorRT 或跨数据集实验。本轮只解决正式数据与证据协议，并给出唯一当前任务。
 88 | 
 89 | ## 返回合同
 90 | 
 91 | 请输出一份可直接保存的裁决，并包含：
 92 | 
 93 | 1. `SESSION_ASSERTION`：原样回显 nonce、Project ID、H65 base 和诊断提交；
 94 | 2. `SCIENTIFIC_DECISION`：只选 `CONTINUE / REVISE / PIVOT / STOP` 之一；
 95 | 3. `FULL_TRAIN_IDENTITY`：精确训练 subset、annotation、完整 ID 集、数量和 H65 可比性；
 96 | 4. `HELD_OUT_EVALUATION_IDENTITY`：精确留出 subset、annotation、ID 集、数量、211/212 处置和禁止用途；
 97 | 5. `DIAGNOSTIC_TO_FORMAL_SEQUENCE`：训练侧诊断与全量正式训练的唯一关系，或明确删除诊断；
 98 | 6. `MATCHED_TRAINING_FREEZE`：两臂起点、成功更新数、优化、随机性、成本匹配、checkpoint/EMA；
 99 | 7. `FINAL_EVALUATION_AND_STATISTICS`：一次性完整评估、指标、不确定性、继续/停止门和论文证据边界；
100 | 8. `CODEX_TASK_ORDER`：只给一个当前 Builder、一个独立 Critic、一个独立 Evaluator 的最小顺序与禁止项；
101 | 9. `NEXT_RETURN`：Codex 必须返回哪些实现、正式全量训练和完整评测证据，才能再次请求科学裁决。
102 | 
103 | 若现有材料不足以诚实冻结完整 ID 集，请列出唯一只读身份核验任务、确切输入和通过/阻断输出；除此之外不要把
104 | 路线选择交回 Codex。不要把文档、代码完成或作业成功当作模型有效性证据。
```

### File: .cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md
Lines: 1-769
```md
  1 | 以下为可直接保存并交付 Codex 的冻结版科学裁决。
  2 | 
  3 | # DUCA 多预算检测器适应：科学裁决与执行冻结
  4 | 
  5 | ## 1. `SESSION_ASSERTION`
  6 | 
  7 | **Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`**
  8 | 
  9 | * ChatGPT Project ID：`g-p-6a91061f789881918ccd8357ca3d6c92`
 10 | * 仓库：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
 11 | * 新实验的唯一模型与代码基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
 12 | * 只读 whole-video 诊断提交：`33e4ed137c33eef07f0452b44506a6993bdf7535`
 13 | 
 14 | `04c35a3b...` 是 H65 训练身份。它的 Stage‑1 合同为 30 个训练轮次、每轮 100 次成功更新、终态 `epoch_29/state_dict_ema`；Stage‑2 合同明确使用一个重新开始的 6,000 次成功更新日程，并从 Stage‑1 EMA 初始化。
 15 | 
 16 | `33e4ed...` 不是新模型基座。该提交只修改 whole-video runner 和聚焦测试，把密封 proposal 的重放顺序恢复为 producer 原始顺序，避免 Soft‑NMS 分数并列时因额外排序产生确定性漂移；它没有修改模型、预测值、预算候选、损失、NMS 或评价器。
 17 | 
 18 | 本轮材料权重冻结如下：
 19 | 
 20 | 1. 用户本轮提供的 2026‑08‑31 prompt、`PAPER_PROGRESS.md`、两个实验页和来源记录是当前科学状态；
 21 | 2. 2026‑08‑28 材料包已完整读取，只用于代码库存边界、角色约束、原始检查点证据和历史谱系，不覆盖本轮更新后的停止边界与科学问题。库存提交不是实验身份，Pro 负责科学选择，Builder、Critic、Evaluator 只执行冻结任务。  
 22 | 3. PJST 原始评估材料用于确认 H65 Stage‑1 检查点路径、epoch、SHA‑256 和状态键；旧项目 prompt、README、旧研究状态和历史版本注册表仅作历史背景。     
 23 | 
 24 | ---
 25 | 
 26 | ## 2. `SCIENTIFIC_DECISION`
 27 | 
 28 | **唯一裁决：`CONTINUE`**
 29 | 
 30 | 批准执行一次有界、单变量、完整匹配的多预算训练实验。
 31 | 
 32 | ### 2.1 为什么值得执行
 33 | 
 34 | 此前 704 个 whole-video 候选的终态负结果只检验了：
 35 | 
 36 | > 一个只在固定 `K=384` 协议下训练并冻结的模型，在密封的 `K256/K384/K512` 预测之间重新分配预算时，是否已经具有足够的开发集联合性能空间。
 37 | 
 38 | 该实验没有重新训练 detector，没有执行梯度更新，也没有让重型表示在训练阶段见到 `K256` 或 `K512`。因此，“训练输入支持不足导致跨预算不兼容”仍是一个尚未接受直接检验、但可以通过单次匹配实验裁决的机制假说。
 39 | 
 40 | 本轮批准不表示该假说已经成立，也不表示为了延续项目而放宽门槛。批准的理由只有一个：它改变了旧实验从未改变过的变量——**训练时的预算分布**——并能以两条完整匹配训练臂直接产生判别性证据。
 41 | 
 42 | ### 2.2 旧 `STOP` 完整保留
 43 | 
 44 | 以下旧路线继续永久只读：
 45 | 
 46 | * Marginal‑v1；
 47 | * cap‑release；
 48 | * 96-state 联合邻域；
 49 | * whole-video 704-state 分支；
 50 | * 旧 40-video controller holdout 上的密封预测与 oracle 结果。
 51 | 
 52 | 不得：
 53 | 
 54 | * 重跑或扩大旧候选空间；
 55 | * 修改旧门槛；
 56 | * 加入第三个视频或组合多个 transfer；
 57 | * 训练旧 controller 或 utility head；
 58 | * 在旧结果中事后选择候选再补 bootstrap；
 59 | * 使用旧 40 个视频作为本轮“未参与规则选择”的开发集；
 60 | * 访问 official test；
 61 | * 把新结果写成旧 Marginal 路线的恢复。
 62 | 
 63 | 新实验从 `04c35a3b...` 建立独立分支。旧分支唯一可移植的是已经验证的真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始 proposal 顺序保持。
 64 | 
 65 | ---
 66 | 
 67 | ## 3. `CAUSAL_ISOLATION`
 68 | 
 69 | ### 3.1 冻结的唯一干预变量
 70 | 
 71 | 第一轮继续使用当前**嵌套**的 `K256/K384/K512` 位置构造。
 72 | 
 73 | * `K256`、`K384`、`K512` 必须来自同一 H65 priority sequence；
 74 | * 不改成三个预算分别重新运行的“预算原生 H65 选点”；
 75 | * 不修改 Scout 输入、评分字段、优先级调制、覆盖下限、确定性累计采样或短窗口折叠；
 76 | * 不修改物理时间逆映射、VideoMAE 语义、Temporal Adapter、ActionFormer/AdaTAD 检测头、损失、Soft‑NMS 或评价器。
 77 | 
 78 | **唯一干预变量是：训练时送入相同 H65 模型的预算分布。**
 79 | 
 80 | 两臂为：
 81 | 
 82 | 1. **固定预算控制臂**
 83 | 
 84 |    * 每个训练窗口请求 `K384`；
 85 |    * 短窗口实际成本为 `min(valid_observations,384)`。
 86 | 
 87 | 2. **多预算适应臂**
 88 | 
 89 |    * 每个训练窗口请求 `K256`、`K384` 或 `K512`；
 90 |    * 使用下文冻结的、按实际 observation 成本校准的概率；
 91 |    * 短窗口继续按 `min(valid_observations,K)` 折叠。
 92 | 
 93 | ### 3.2 不进入第一轮的变量
 94 | 
 95 | 明确禁止：
 96 | 
 97 | * 预算条件嵌入；
 98 | * 显式输入 `K`、相邻物理时间间隔或局部采样密度；
 99 | * 蒸馏；
100 | * Gumbel‑Softmax；
101 | * 新 Scout、新分类头或边界头；
102 | * 新 selector 或预算原生位置构造；
103 | * DFT、Mamba、Block Drop；
104 | * CUDA、TensorRT 或其他部署优化；
105 | * 新数据集；
106 | * controller 训练；
107 | * 根据开发结果改变预算概率。
108 | 
109 | ### 3.3 可归因结论的精确边界
110 | 
111 | 两臂使用完全相同的 H65 Stage‑2 可训练参数集合。不得只在候选臂冻结或解冻某些参数。
112 | 
113 | 因此，正结果支持的是：
114 | 
115 | > 在相同 H65 架构、初始化、监督、优化和嵌套位置构造下，训练时接触多种预算输入可以建立更好的跨预算兼容性。
116 | 
117 | 它不能被写成：
118 | 
119 | * “ActionFormer 检测头单独是根因”；
120 | * “Scout 完全没有参与适应”；
121 | * “不可微选点已被解决”；
122 | * “动态预算一般问题已经成立”。
123 | 
124 | ---
125 | 
126 | ## 4. `TRAINING_FREEZE`
127 | 
128 | ### 4.1 训练定位
129 | 
130 | 选择：
131 | 
132 | > **从冻结的 H65 Stage‑1 终态进行两臂匹配的完整 Stage‑2 训练。**
133 | 
134 | 拒绝从 H65 terminal detector 进行短期继续训练。
135 | 
136 | 原因是：短期继续训练从一个已经被 `K384` 输入分布塑形的优化盆地出发。若候选失败，无法区分“多预算训练无效”和“学习率太小、更新不足或无法离开既有盆地”；若候选成功，也只能说明事后微调可部分修复，不能形成干净的完整训练比较。
137 | 
138 | ### 4.2 唯一起点
139 | 
140 | 两臂共同加载：
141 | 
142 | * 路径：
143 |   `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
144 | * 期望 epoch：`29`
145 | * 状态键：`state_dict_ema`
146 | * SHA‑256：
147 |   `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`
148 | 
149 | 该检查点身份在已有终态评估材料中已绑定；H65 Stage‑1 配置也明确指定 terminal `epoch_29/state_dict_ema`。
150 | 
151 | 加载模型权重后，两臂都必须重新初始化：
152 | 
153 | * optimizer state；
154 | * learning-rate scheduler state；
155 | * AMP scaler；
156 | * EMA accumulator；
157 | * successful-update counter；
158 | * `frame_selector._loss_weight_schedule_step`；
159 | * 数据采样与随机增强随机状态。
160 | 
161 | 不得从 H65 Stage‑2 terminal optimizer、scheduler 或 EMA 状态恢复。
162 | 
163 | ### 4.3 训练长度
164 | 
165 | 每臂：
166 | 
167 | * **恰好 6,000 次成功 optimizer update**；
168 | * 非有限损失或 AMP 重放不计为成功更新；
169 | * 两臂成功更新数必须完全相等；
170 | * 以成功更新索引作为训练主时钟。
171 | 
172 | 由于新 Stage‑2 训练集为 160 个视频，不能把“60 个实际数据轮次”静默当作与原 H65 相同的训练量。训练器应循环数据直到完成 6,000 次成功更新。
173 | 
174 | 为保持 H65 原日程：
175 | 
176 | * 每 100 次成功更新定义一个 H65 **schedule epoch**；
177 | * 总计 60 个 schedule epoch；
178 | * warmup 5 个 schedule epoch，即前 500 次成功更新；
179 | * Stage‑2 前后半程边界保持在第 3,000 次成功更新；
180 | * 所有原来按 step 定义的 loss、policy 和 detector-gradient 日程保持原值。
181 | 
182 | H65 Stage‑2 原配置明确冻结 `max_updates=6000`、`duca_stage2_half_steps=3000`，并从 `state_dict_ema` 初始化后重置训练状态。
183 | 
184 | ### 4.4 优化器、调度器与数值合同
185 | 
186 | 两臂逐项相同：
187 | 
188 | * optimizer：`AdamW`
189 | * 主学习率：`1e-4`
190 | * weight decay：`0.05`
191 | * VideoMAE 主体保持 H65 参数分组；基础 backbone 的冻结规则不变，adapter 学习率为 `2e-4`
192 | * H65 Stage‑2 Scout 相关学习率：
193 | 
194 |   * coarse trunk：`1e-5`
195 |   * action head：`2e-5`
196 |   * transition scorer：`5e-5`
197 | * scheduler：`LinearWarmupCosineAnnealingLR`
198 | * warmup：前 500 次成功更新
199 | * 总日程：6,000 次成功更新
200 | * gradient clipping：global norm `1`
201 | * AMP：开启
202 | * EMA：开启
203 | * batch size：每卡 `2`
204 | * seed：`3407`
205 | * DDP、unused-parameter、损失归一化和梯度累积规则不得因预算档位而改变。
206 | 
207 | 这些数值来自 `04c35a3b...` 所继承的 H65/AdaTAD 配置；精确配置优先于任何文字摘要。
208 | 
209 | 若 `K512` 在 batch size 2 下发生显存不足，只允许对**两臂同时**采用 microbatch 1、累计 2 个样本后完成一次 optimizer update。不得只降低候选臂 batch size，也不得改变总成功更新数。
210 | 
211 | ### 4.5 随机性匹配
212 | 
213 | 两臂必须共享：
214 | 
215 | * 视频与窗口采样次序；
216 | * random truncation；
217 | * resize/crop/flip/color augmentation；
218 | * 模型初始化；
219 | * dropout/drop-path 随机种子；
220 | * optimizer update 次序。
221 | 
222 | 候选臂的预算随机性使用独立命名空间，不得消耗数据增强或模型随机流：
223 | 
224 | `SHA256("DUCA-MBDA-BUDGET-v001|3407|<successful_update>|<sample_id>|<occurrence>")`
225 | 
226 | 固定预算臂忽略该值并始终使用 `K384`。
227 | 
228 | ### 4.6 多预算概率与训练成本校准
229 | 
230 | 先在**160-video Stage‑2 训练集合**上重放完整的无标签训练窗口计划，按短窗口折叠计算：
231 | 
232 | * `μ256`：请求 K256 时的平均实际 observation；
233 | * `μ384`：请求 K384 时的平均实际 observation；
234 | * `μ512`：请求 K512 时的平均实际 observation。
235 | 
236 | 不得读取动作类别、边界或检测指标。
237 | 
238 | 保持中央预算概率固定为：
239 | 
240 | `p384 = 0.50`
241 | 
242 | 保持两侧总暴露为：
243 | 
244 | `p256 + p512 = 0.50`
245 | 
246 | 并唯一求解：
247 | 
248 | `p256 = 0.5 × (μ512 − μ384) / (μ512 − μ256)`
249 | 
250 | `p512 = 0.5 × (μ384 − μ256) / (μ512 − μ256)`
251 | 
252 | 这样候选臂的期望实际 observation 成本等于固定 K384：
253 | 
254 | `p256·μ256 + p384·μ384 + p512·μ512 = μ384`
255 | 
256 | 当 `μ384` 正好位于两端均值中点时，概率自然恢复为 `0.25/0.50/0.25`。
257 | 
258 | 阻断条件：
259 | 
260 | * `μ512 == μ256`；
261 | * 任一概率不在 `[0,1]`；
262 | * 实际成本不是逐档单调；
263 | * 概率是在读取开发标签或模型指标后调整。
264 | 
265 | 6,000 次更新中的离散预算数量采用最大余数法从上述概率一次性确定，再由冻结哈希打乱。候选臂最终实际训练 observation 总量与控制臂的偏差必须报告；预运行时若超过 `0.5%`，只允许在不改变概率公式和不读取标签的情况下重新排列预算与样本 occurrence，使差值最小。仍超过 `0.5%` 时返回阻断，不开始正式训练。
266 | 
267 | ### 4.7 checkpoint 与中间验证
268 | 
269 | 保存点：
270 | 
271 | * 每 500 次成功更新保存可恢复 checkpoint；
272 | * 强制保留 update 3,000；
273 | * 保留最近三个有效 checkpoint；
274 | * 强制保留 update 6,000 terminal checkpoint。
275 | 
276 | 恢复包必须包含：
277 | 
278 | * model；
279 | * `state_dict_ema`；
280 | * optimizer；
281 | * scheduler；
282 | * AMP scaler；
283 | * successful-update index；
284 | * 数据 sampler 和全部随机状态。
285 | 
286 | **唯一结果模型：update 6,000 的 `state_dict_ema`。**
287 | 
288 | 不得：
289 | 
290 | * 按中间 Avg-mAP 选择 checkpoint；
291 | * early stopping；
292 | * 事后选择 best validation EMA；
293 | * 因某一预算曲线较好而修改概率或损失。
294 | 
295 | 新的 40-video 开发集在两臂 terminal checkpoint 和配置全部密封前不得进行任何有标签评价。训练中可以查看损失、梯度、数值有限性和无标签实际成本，但不能查看开发 mAP。
296 | 
297 | ---
298 | 
299 | ## 5. `DEVELOPMENT_SPLIT_FREEZE`
300 | 
301 | ### 5.1 为什么不能继续使用旧 40 个视频
302 | 
303 | 旧 40-video controller holdout 已经参与：
304 | 
305 | * capped oracle；
306 | * released oracle；
307 | * 96-state 联合邻域；
308 | * 704-state whole-video 枚举；
309 | * 门槛与路线裁决。
310 | 
311 | 因此它不再是未参与规则选择的开发集。
312 | 
313 | 本轮将这 40 个视频明确归入训练可用集合，不再承担评价功能。
314 | 
315 | ### 5.2 可复现划分规则
316 | 
317 | 输入：
318 | 
319 | 1. THUMOS14 规范训练侧的 200 个视频 ID；
320 | 2. 旧 whole-video 实验实际读取的 40 个 `holdout_videos` ID；
321 | 3. split seed：`20260831`。
322 | 
323 | 步骤：
324 | 
325 | 1. 从 annotation 中读取规范训练侧视频 ID，去重后按完整视频 ID 字典序排序；
326 | 
327 | 2. 要求数量严格为 `200`；
328 | 
329 | 3. 读取旧 40-video manifest，要求数量严格为 `40`，且全部属于上述 200；
330 | 
331 | 4. 从 200 中移除旧 40，得到 160 个新的开发候选；
332 | 
333 | 5. 对每个候选视频 ID 计算：
334 | 
335 |    `SHA256("DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-DEV-v001|20260831|" + video_id)`
336 | 
337 | 6. 按 `(完整 SHA‑256 十六进制值, video_id)` 升序排序；
338 | 
339 | 7. 前 `40` 个视频作为新开发集；
340 | 
341 | 8. 其余 `120` 个新候选与旧 `40` 个历史 holdout 合并，形成 `160` 个 Stage‑2 训练视频。
342 | 
343 | 最终数量：
344 | 
345 | * Stage‑2 train：`160`
346 | * Stage‑2 development：`40`
347 | * official evaluation/test：`0` 个被访问
348 | 
349 | 必须落盘：
350 | 
351 | * `train_video_ids.txt`
352 | * `development_video_ids.txt`
353 | * `historical_40_excluded_from_dev.txt`
354 | * 一个简短 JSON，记录 seed、生成字符串、数量和来源 annotation。
355 | 
356 | 不要求新的哈希治理框架；普通文本清单和一个 JSON 足够。
357 | 
358 | ### 5.3 使用边界
359 | 
360 | 160-video train 可用于：
361 | 
362 | * 两臂参数学习；
363 | * 多预算概率校准；
364 | * 实际训练成本估计；
365 | * 动作时长分层阈值；
366 | * 数值与运行检查。
367 | 
368 | 40-video development 仅用于：
369 | 
370 | * 两臂 terminal EMA 的三档预算评价；
371 | * 预登记的诊断指标；
372 | * 预登记的等成本 whole-video oracle；
373 | * 本轮最终科学门槛。
374 | 
375 | development 不得用于：
376 | 
377 | * 超参数、概率、损失权重或训练长度选择；
378 | * checkpoint 选择；
379 | * NMS 或分数阈值选择；
380 | * 修改 K 档位；
381 | * 选择不同的 whole-video 候选定义；
382 | * 决定是否补训某一臂。
383 | 
384 | ### 5.4 证据独立性的诚实边界
385 | 
386 | 该 40-video development 不参与**本轮 Stage‑2 参数更新或本轮规则选择**。
387 | 
388 | 但是，现有 H65 Stage‑1 checkpoint 并不是针对该新划分重新训练和封存的。现有材料不能证明新开发视频从未影响过历史初始化。因此，本划分是：
389 | 
390 | > 条件于冻结 H65 Stage‑1 初始化的独立 Stage‑2 机制开发集。
391 | 
392 | 它不是完全独立于所有历史训练的外部确认集。
393 | 
394 | 这是在不重训 Scout、不改变另一项科学变量、又不访问 official test 的前提下可接受的最小证据设计。即使实验通过，也只能进入下一轮确认性实验裁决，不能直接形成最终泛化或显著性主张。
395 | 
396 | 若旧 40-video manifest 无法精确恢复、200 个训练 ID 数量不符、train/dev 发生交集，Evaluator 必须返回阻断，不得自行换 seed、改变视频数量或重新切分。
397 | 
398 | ---
399 | 
400 | ## 6. `EVALUATION_AND_GATES`
401 | 
402 | ### 6.1 实验准入检查
403 | 
404 | 正式训练前必须通过：
405 | 
406 | 1. 基座精确为 `04c35a3b...`；
407 | 2. `dynamic_budget.py` 的三档预算语义未修改；
408 | 3. Stage‑1 checkpoint 路径、SHA‑256、epoch 和状态键完全匹配；
409 | 4. 新 train/dev ID 无交集；
410 | 5. 两臂初始化参数逐张量相同；
411 | 6. 两臂 optimizer 参数组、学习率、trainable flags 和 EMA 配置相同；
412 | 7. 固定控制的 K384 前向与基座 K384 在相同 checkpoint、相同输入和相同 RNG 下：
413 | 
414 |    * selected positions bit-exact；
415 |    * detector tensor 输出在现有数值容差内一致；
416 |    * proposal 原始生成顺序一致；
417 |    * actual observation 数一致；
418 | 8. `K256/K512` 真正执行变长 VideoMAE，不得在重型骨干前补齐为 K384 或 K512；
419 | 9. mixed-budget packet 分组后，输出重新对齐到原 batch/sample 顺序；
420 | 10. 候选训练期预期成本满足上文校准合同。
421 | 
422 | 任何一项失败都属于实现或协议问题，不是模型负结果。
423 | 
424 | ### 6.2 三档终态指标
425 | 
426 | 两臂 terminal EMA 都必须分别在 `K256/K384/K512` 上产生密封预测并报告：
427 | 
428 | * mAP@0.3、0.4、0.5、0.6、0.7；
429 | * Avg-mAP；
430 | * pre-NMS class-agnostic proposal recall：
431 | 
432 |   * Recall@100、Recall@200；
433 |   * tIoU 0.5 和 0.7；
434 | * 每个 GT 与同类别最高 tIoU pre-NMS proposal 匹配后的：
435 | 
436 |   * 起点绝对误差；
437 |   * 终点绝对误差；
438 |   * 物理秒误差的均值和中位数；
439 |   * 相对 GT duration 归一化误差；
440 | * 每视频 pre-NMS、post-NMS proposal 数的均值和中位数；
441 | * top‑200 proposal 中的假阳性：
442 | 
443 |   * 若不存在同类别、tIoU≥0.3 的 GT 匹配则记为假阳性；
444 |   * 分别报告 NMS 前后；
445 | * 短、中、长动作的 Avg-mAP 与 mAP@0.7。
446 | 
447 | 短、中、长阈值只能由 160-video train 的 GT duration 三分位数预先确定，不得用 development duration 调整。
448 | 
449 | 上述指标为解释性诊断，不新增通过门槛。
450 | 
451 | ### 6.3 K384 安全门
452 | 
453 | 在新 40-video development 上，比较多预算模型与同更新数固定控制，二者都以 K384 推理：
454 | 
455 | * `ΔAvg-mAP >= -0.2` 个百分点；
456 | * `ΔmAP@0.7 >= -0.2` 个百分点。
457 | 
458 | 两项必须同时满足。
459 | 
460 | 该门防止多预算模型通过降低自身 K384 锚点来人为放大后续 oracle 增益。
461 | 
462 | ### 6.4 等成本 whole-video oracle
463 | 
464 | 对两臂分别使用各自密封的 K256/K384/K512 terminal predictions。
465 | 
466 | 候选空间保持旧 whole-video 定义，但使用新的 40-video development：
467 | 
468 | 1. 全 K384 为该模型自己的固定基线；
469 | 2. 对每个不同视频的有序对 `(donor,recipient)`：
470 | 
471 |    * donor 的全部窗口请求 K256；
472 |    * recipient 的全部窗口请求 K512；
473 |    * 其余视频全部请求 K384；
474 | 3. donor 和 recipient 都必须至少有一个窗口的实际成本不同于 K384；
475 | 4. 总实际 observation 成本不得超过全 K384；
476 | 5. 最多生成 `40×39=1560` 个有序状态；
477 | 6. 候选清单只依赖 video ID、sample ID 和 actual observation 成本，必须在读取标签和指标前生成；
478 | 7. 两臂使用完全相同的候选清单；
479 | 8. proposal 合并必须保持 producer 原始生成顺序。
480 | 
481 | `33e4ed...` 的必要价值仅在于已经证明额外重排 proposal 会破坏 Soft‑NMS 的确定性重放，因此新 Evaluator 必须保留原顺序。
482 | 
483 | 每个模型的最优候选先最大化：
484 | 
485 | `min(ΔAvg-mAP − 0.8, ΔmAP@0.7 − 1.0)`
486 | 
487 | 再选择实际成本更低者，最后按 donor、recipient ID 字典序。
488 | 
489 | ### 6.5 Oracle 继续门
490 | 
491 | 多预算模型相对其自身全 K384 基线必须同时满足：
492 | 
493 | * `ΔAvg-mAP >= +0.8` 个百分点；
494 | * `ΔmAP@0.7 >= +1.0` 个百分点；
495 | * actual observation cost `<=` 其全 K384 成本。
496 | 
497 | 同时必须通过 K384 安全门。
498 | 
499 | 固定控制也运行相同 oracle，但只作为因果对照，不改变预登记阈值。
500 | 
501 | 结果解释冻结为：
502 | 
503 | 1. **多预算通过、固定控制不通过**
504 | 
505 |    * 支持“训练时多预算暴露建立了新的跨预算联合 headroom”；
506 |    * 返回 Pro 决定是否进入可预测 budget controller 与确认性实验；
507 |    * 不自动训练 controller，不访问 official test。
508 | 
509 | 2. **多预算与固定控制都通过**
510 | 
511 |    * 说明新 development 本身存在 whole-video headroom；
512 |    * 不能把通过归因于多预算训练；
513 |    * 结果为因果未决，返回 Pro，不自动继续。
514 | 
515 | 3. **多预算未通过**
516 | 
517 |    * 无论固定控制结果如何，当前多预算检测器适应假说未达到预登记目标；
518 |    * 停止当前 H65 嵌套 `K256/K384/K512` detector-adaptation 路线；
519 |    * 不补训练、不换 seed、不改概率、不降低门槛。
520 | 
521 | 4. **K384 安全门未通过**
522 | 
523 |    * 即使 oracle 通过，也判定本路线失败；
524 |    * 不接受以损害标准预算性能换取 oracle 增益。
525 | 
526 | 5. **Avg-mAP 达门而 mAP@0.7 未达门**
527 | 
528 |    * 当前路线仍停止；
529 |    * 只允许下一轮由 Pro 重新判断是否值得单独检验 K、相邻物理间隔或局部采样密度条件；
530 |    * 本轮不得预埋这些输入。
531 | 
532 | ### 6.6 统计与论文证据边界
533 | 
534 | 本轮使用一个 seed `3407` 和一个条件于历史 Stage‑1 初始化的训练侧 development。
535 | 
536 | 因此：
537 | 
538 | * 不作总体显著性主张；
539 | * 不把 oracle 最优候选称为可部署策略；
540 | * 不把通过结果称为 official validation/test 结果；
541 | * 不补事后 bootstrap 来放大结论；
542 | * 不声称优于 dense AdaTAD；
543 | * 不声称已经获得端到端性能—成本优势。
544 | 
545 | 通过结果最多是强机制开发证据；失败结果可以在冻结范围内停止本三档适应路线，但不能外推否定所有动态计算或预算条件模型。
546 | 
547 | ---
548 | 
549 | ## 7. `CODEX_TASK_ORDER`
550 | 
551 | ### 7.1 Builder
552 | 
553 | 建立唯一分支：
554 | 
555 | `feature/duca-multi-budget-detector-adaptation-v1-20260831`
556 | 
557 | 基于：
558 | 
559 | `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
560 | 
561 | #### Builder 允许完成的最小实现
562 | 
563 | 1. 新增两份 matched Stage‑2 配置：
564 | 
565 |    * fixed K384 control；
566 |    * calibrated K256/K384/K512 adaptation。
567 | 2. 新增一个确定性 train/dev ID 生成入口；
568 | 3. 新增训练侧实际 observation 概率校准；
569 | 4. 移植并接入：
570 | 
571 |    * 真实变长 VideoMAE packet 执行；
572 |    * packet 输出回填；
573 |    * actual observation 计数；
574 |    * K384 parity；
575 |    * whole-video terminal evaluator；
576 |    * producer 原始 proposal 顺序保持；
577 | 5. 新增一个最小 Slurm launcher；
578 | 6. 新增聚焦测试。
579 | 
580 | #### Builder 禁止修改
581 | 
582 | * H65 Scout 结构或输入；
583 | * H65 priority、selector、位置嵌套关系；
584 | * `dynamic_budget.py` 的三档语义；
585 | * detector head；
586 | * 分类或回归损失；
587 | * 物理时间逆映射；
588 | * Soft‑NMS；
589 | * annotation、类别映射或评价器；
590 | * official test 入口；
591 | * 任何旧 Marginal、cap-release、96-state 或 whole-video 产物；
592 | * 新 controller、部署优化或额外模型组件。
593 | 
594 | #### Builder 必须证明
595 | 
596 | * split 数量与无交集；
597 | * 旧 40 个视频不进入新 development；
598 | * 两臂初始 state_dict 一致；
599 | * optimizer/trainable/EMA 配置一致；
600 | * K384 bit-exact parity；
601 | * 三档实际 observation 单调；
602 | * 短窗口正确折叠；
603 | * mixed packet 对齐；
604 | * proposal 顺序不被排序；
605 | * 6,000 次成功更新是唯一停止时钟；
606 | * development 标签在 terminal 密封前不可访问。
607 | 
608 | #### Builder 失败返回
609 | 
610 | 遇到以下任一情况立即返回客观阻断，不自行设计替代协议：
611 | 
612 | * Stage‑1 checkpoint 或 SHA 不匹配；
613 | * 旧 40-video ID 清单无法恢复；
614 | * 规范训练视频不是 200 个；
615 | * K384 无法复现；
616 | * 变长执行必须修改 selector、loss、NMS 或 evaluator；
617 | * 成本校准无法满足合同；
618 | * K512 只能通过候选臂专属 batch size 才能运行。
619 | 
620 | Builder 不提交正式训练。
621 | 
622 | ### 7.2 独立 Critic
623 | 
624 | Critic 在独立上下文中只读审查 Builder 的唯一精确提交。
625 | 
626 | 审查范围仅限：
627 | 
628 | * 科学变量是否仍只有训练预算分布；
629 | * 位置集合是否仍嵌套；
630 | * Scout、selector、detector、loss、NMS、评价器有无隐性变化；
631 | * train/dev 泄漏；
632 | * checkpoint 与状态重置；
633 | * 两臂 trainable 参数、优化器和更新数公平性；
634 | * variable-length forward 是否真实；
635 | * K384 parity；
636 | * packet 和 proposal 顺序；
637 | * 训练成本口径；
638 | * development 是否可能在 terminal 前参与选择。
639 | 
640 | Critic 输出只能是：
641 | 
642 | * `PASS`；或
643 | * 一次有界 blocker 清单。
644 | 
645 | 不得因代码风格、日志美化、通用完备性、缺少工作流框架或假设性边角条件制造修复循环。角色边界应继续服从论文优先、最短科学闭环原则。 
646 | 
647 | ### 7.3 独立 Evaluator
648 | 
649 | Critic `PASS` 后，Evaluator 依次执行：
650 | 
651 | 1. 真实数据 PRE_RUN：
652 | 
653 |    * checkpoint 身份；
654 |    * split；
655 |    * K384 parity；
656 |    * 每档至少两个真实 batch；
657 |    * 梯度、AMP、EMA、actual cost；
658 |    * 不计算 development mAP。
659 | 2. 两臂正式 6,000-update 训练：
660 | 
661 |    * 相同 seed；
662 |    * 相同硬件类别；
663 |    * 相同样本与增强流；
664 |    * 独立输出根；
665 |    * 不访问 official test。
666 | 3. 两臂 terminal `state_dict_ema` 的六次推理：
667 | 
668 |    * control × K256/K384/K512；
669 |    * adaptation × K256/K384/K512。
670 | 4. 密封三档预测后才开放 development 标签；
671 | 5. 运行预登记诊断、K384 安全门和两臂 whole-video oracle；
672 | 6. 生成一次终态结果，不根据结果回改训练或规则。
673 | 
674 | 确定性的启动器、路径、节点或存储故障只允许一次不改变模型、数据、split、门槛和结果选择的最小恢复。它们必须与科学结果分开报告。
675 | 
676 | ---
677 | 
678 | ## 8. `NEXT_RETURN`
679 | 
680 | Codex 下一次必须返回以下证据，缺一项不得要求新的科学路线裁决。
681 | 
682 | ### 8.1 实现与身份
683 | 
684 | * branch；
685 | * exact commit；
686 | * parent commit；
687 | * clean-tree 状态；
688 | * 修改文件清单；
689 | * 从 `33e4ed...` 移植的具体符号；
690 | * `dynamic_budget.py` 未修改证明；
691 | * 两份最终配置全文或稳定链接。
692 | 
693 | ### 8.2 数据划分
694 | 
695 | * 规范 200-video ID 清单；
696 | * 旧 40-video 清单；
697 | * 新 40-video development 清单；
698 | * 新 160-video train 清单；
699 | * seed 和完整 SHA‑256 生成字符串；
700 | * train/dev 交集为空的测试结果。
701 | 
702 | ### 8.3 起点与训练合同
703 | 
704 | * Stage‑1 checkpoint 实际路径；
705 | * SHA‑256；
706 | * epoch；
707 | * state key；
708 | * 两臂初始化逐张量一致性结果；
709 | * optimizer 参数组；
710 | * trainable 参数名集合；
711 | * scheduler；
712 | * AMP、EMA、gradient clipping；
713 | * 预算概率 `p256/p384/p512`；
714 | * `μ256/μ384/μ512`；
715 | * 预期与实际训练 observation 成本；
716 | * 6,000 次成功更新审计。
717 | 
718 | ### 8.4 Builder 与 Critic 证据
719 | 
720 | * focused test 结果；
721 | * K384 parity 数值；
722 | * 三档真实 batch 的 selected count 与 packet shape；
723 | * Critic 对 exact commit 的完整终态结论；
724 | * 所有未关闭 blocker。
725 | 
726 | ### 8.5 正式作业与终态模型
727 | 
728 | * PRE_RUN Job ID、状态和输出根；
729 | * 两臂训练 Job ID、状态和输出根；
730 | * 基础设施恢复记录；
731 | * update‑6,000 checkpoint 路径与 SHA‑256；
732 | * `state_dict_ema` 存在性；
733 | * 两臂实际成功更新数；
734 | * 两臂最终 observation 总量。
735 | 
736 | ### 8.6 结果
737 | 
738 | 对两臂、三档预算逐项返回：
739 | 
740 | * 五个 tIoU mAP；
741 | * Avg-mAP；
742 | * proposal recall；
743 | * 起终点误差；
744 | * proposal 数；
745 | * NMS 前后假阳性；
746 | * 短、中、长动作结果；
747 | * 原始预测和结构化指标路径。
748 | 
749 | ### 8.7 Oracle 与门槛
750 | 
751 | * 无标签候选 manifest；
752 | * 合法候选数；
753 | * 固定 K384 成本；
754 | * 每臂最优 Avg-mAP 候选；
755 | * 每臂最优 mAP@0.7 候选；
756 | * 每臂最优联合门余量候选；
757 | * K384 安全门两项差值；
758 | * 多预算 oracle 两项差值和成本；
759 | * fixed-control oracle 结果；
760 | * 根据本裁决规则得到的唯一分类：
761 | 
762 |   * 支持适应；
763 |   * 因果未决；
764 |   * 停止当前路线；
765 |   * 证据无效。
766 | 
767 | 在这些实现与结果证据返回前，不得训练 controller，不得访问 official test，不得另行提出预算条件嵌入或新 selector，也不得把代码完成、测试通过或训练作业成功当作模型有效性证据。
768 | 
769 | 以上即为本轮冻结版科学裁决，可直接作为 Builder、Critic 与 Evaluator 的唯一执行依据。
```

### File: research-wiki/sources/2026-08-31-duca-full-train-official-test-human-constraint.md
Lines: 1-44
```md
 1 | ---
 2 | type: source
 3 | source_type: human_constraint
 4 | status: discussed
 5 | updated: 2026-08-31
 6 | project: DUCA
 7 | ---
 8 | 
 9 | # 正式可比实验必须使用完整训练集与完整官方留出评估集
10 | 
11 | ## 人类约束
12 | 
13 | DUCA 后续能够进入论文比较的正式实验必须满足以下条件：
14 | 
15 | - 固定 K384 控制臂与多预算检测器适应臂都使用项目最终冻结的完整 THUMOS14 训练集，不得把训练子集或
16 |   pilot 结果提升为正式主比较；
17 | - 设计、超参数、训练日程、模型选择规则和评价协议冻结后，两臂只在完整官方 held-out evaluation split 上进行
18 |   可比评估；
19 | - 两臂使用完全相同的数据语义、类别映射、官方评价器、Soft-NMS 和结果保存协议；
20 | - 官方留出评估只用于冻结方案的评价，不参与训练、超参数调整、checkpoint 选择、阈值或规则选择、路线选择，
21 |   也不得通过反复查看结果进行迭代；
22 | - 训练侧子集或开发划分只能作为正式训练前的诊断、机制证伪或设计选择材料，不能替代完整训练与完整官方留出
23 |   评估形成的论文证据。设计冻结后，正式两臂必须重新在完整训练集上完成匹配训练。
24 | 
25 | ## 当前协议冲突
26 | 
27 | 项目记录中并存两种 THUMOS14 命名口径：部分 OpenTAD/DUCA 运行把 `training` 作为训练入口、把 211 个视频的
28 | `validation` 作为官方评估；ActionFormer 官方协议则使用 `validation` 训练并在 `test` 上评估，历史记录包含 212 个
29 | 带标注评估视频。不能仅凭名称把两者视为同一个 split，也不能静默删除或补入视频。
30 | 
31 | 因此，Pro 必须在 Builder 开始前冻结：
32 | 
33 | 1. 完整训练 split 的配置名称和完整视频 ID 集合；
34 | 2. 完整官方 held-out evaluation split 的配置名称和完整视频 ID 集合；
35 | 3. annotation、类别映射、官方 evaluator 与排除规则；
36 | 4. 正式评估的单次使用边界，以及任何训练侧开发诊断与最终完整训练之间的关系。
37 | 
38 | 在这些身份明确前，不启动实现或训练。当前正在生成的 Pro 对话早于本约束，不能通过追问、打断或同会话追加来
39 | 修改；其终态返回后，应先核对是否明确区分开发诊断与完整训练/完整官方留出评估。若没有，则另开一次新的 Pro
40 | 裁决，不由 Codex 自行选择 split。
41 | 
42 | ## 证据边界
43 | 
44 | 本文件记录人类对正式实验公平性与完整性的约束，不是模型性能结果，也不证明任何路线有效。
```

### File: research-wiki/sources/2026-08-31-duca-multi-budget-pro-freeze-v001.md
Lines: 1-48
```md
 1 | ---
 2 | type: source
 3 | status: discussed
 4 | updated: 2026-08-31
 5 | project: DUCA
 6 | ---
 7 | 
 8 | # Pro 多预算检测器适应冻结裁决 v001
 9 | 
10 | ## 会话身份
11 | 
12 | - Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`
13 | - Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
14 | - Conversation：`6a9521de-d020-83e9-a0b9-19045c8d5390`
15 | - 模型选择器：Pro，已由浏览器运行记录验证；没有单独暴露额外 effort 控件。
16 | - H65 模型基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
17 | - whole-video 诊断功能来源：`33e4ed137c33eef07f0452b44506a6993bdf7535`
18 | - 完整原始回答：
19 |   `.cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md`
20 | 
21 | 首次浏览器控制尝试在提交前因登录探测失败终止，没有创建科学对话。随后同一 nonce 的一次传输恢复成功提交，
22 | Project、conversation URL、nonce 和 Pro 模型选择均通过终态记录核验；因此科学尝试计数仍为一次。
23 | 
24 | ## Pro 的科学裁决
25 | 
26 | Pro 选择 `CONTINUE`，只批准比较固定 K384 训练和嵌套 K256/K384/K512 多预算训练。Scout、位置构造、检测器
27 | 结构、损失、物理时间映射、Soft-NMS 和评价器保持不变；预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout、
28 | Mamba、Block Drop 和部署优化均不进入第一轮。
29 | 
30 | Pro 还冻结了以下执行选择：
31 | 
32 | - 两臂都从 H65 Stage-1 `epoch_29/state_dict_ema` 开始，重新初始化训练状态；
33 | - 每臂恰好完成 6,000 次成功 optimizer update，使用相同优化器、日程、随机种子、可训练参数与 terminal EMA；
34 | - 多预算概率按实际 observation 均值校准，保持 `p384=0.5` 且期望成本等于固定 K384；
35 | - terminal EMA 才是唯一结果模型，中间指标不得用于 checkpoint 或规则选择；
36 | - Builder、独立 Critic、独立 Evaluator 依次执行，旧冻结检测器的 704-state 路线继续只读。
37 | 
38 | ## 与后到人类约束的冲突
39 | 
40 | 该 Pro prompt 在人类新增正式数据要求之前已提交。回答冻结的是：从 200 个训练侧视频中重切 160 个训练视频和
41 | 40 个开发视频，先在该 40-video development 上完成门控，并且本轮不访问 official test。这不是人类要求的正式
42 | 可比实验：两臂必须使用完整训练集完成匹配训练，并在完整官方 held-out evaluation split 上进行一次冻结后的
43 | 最终比较。
44 | 
45 | 因此本回答的机制选择、6,000-update 建议和代码边界可作为下一轮 Pro 的输入，但其 160/40 数据协议不是当前可
46 | 执行授权。Codex 不建立 Builder、不提交 PRE_RUN 或训练，也不把 40-video 结果提升为论文主结果。需要一个新的
47 | 独立 Pro turn 冻结完整训练与完整留出评测的精确 subset 名称、视频 ID、annotation、类别映射、评价器、一次性
48 | 使用边界，以及训练侧诊断如何与正式全量训练隔离。
```

### File: research-wiki/experiments/duca-multi-budget-detector-adaptation.md
Lines: 1-72
```md
 1 | ---
 2 | type: experiment
 3 | status: designed
 4 | updated: 2026-08-31
 5 | project: DUCA
 6 | ---
 7 | 
 8 | # DUCA 多预算检测器适应
 9 | 
10 | ## 科学问题
11 | 
12 | 在保持 H65 的当前嵌套 K256/K384/K512 位置构造、Scout、检测器结构和评价协议不变时，仅让检测器在三种预算
13 | 分布上共同训练，能否恢复每档预算自身的检测质量，并重新建立匹配真实 observation 成本的 Avg-mAP 与高时间
14 | 交并比联合 oracle headroom？
15 | 
16 | 这是位于此前项目级 `STOP` 边界之外的新机制检验。旧结论仍成立：冻结 K384 检测器后进行三档预算转移没有
17 | 达到预登记联合门。新实验不重跑旧 oracle，也不把跨预算表示不匹配预先写成已证实根因。
18 | 
19 | ## 单一干预
20 | 
21 | - 固定预算控制只在 K384 上训练。
22 | - 候选在 K256/K384/K512 上共同训练；名义抽样概率从 `0.25/0.50/0.25` 开始，并按短窗口折叠后的实际
23 |   observation 成本校准。
24 | - 两臂使用同一 H65 起点、相同成功更新数、优化器、学习率日程、随机种子、可训练参数集合和最终指数移动平均
25 |   模型选择规则。
26 | 
27 | 第一轮保留当前嵌套位置构造。不得同时改成预算原生 H65 采样；否则将同时改变选点和训练分布，无法把结果归因
28 | 于检测器预算适应。
29 | 
30 | ## 代码边界
31 | 
32 | - 模型基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
33 | - `33e4ed137c33eef07f0452b44506a6993bdf7535` 仅提供真实变长执行、packet 对齐、实际 observation 计数、K384
34 |   parity、whole-video 评价和原始生成顺序保持等已验证功能。
35 | - 禁止预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout/head/selector、DFT、Mamba、Block Drop 和 TensorRT。
36 | 
37 | ## 决定性输出
38 | 
39 | 1. K256、K384、K512 各自的 Avg-mAP、mAP@0.3--0.7、proposal recall、起终点误差、proposal 数、NMS 前后
40 |    假阳性和动作长度分层结果。
41 | 2. 多预算模型在 K384 上相对同更新数固定控制满足 `ΔAvg-mAP >= -0.2` 和 `ΔmAP@0.7 >= -0.2` 个百分点。
42 | 3. 在未参与当轮参数学习和规则选择的训练侧开发划分上，等成本 oracle 同时满足 `+0.8/+1.0` 个百分点且实际
43 |    成本不高于固定 K384。这一项仅是正式训练前的机制门，不能成为论文主比较。
44 | 4. 设计冻结后的正式比较必须让两臂重新使用完整训练集完成匹配训练，并在完整官方 held-out evaluation split 上
45 |    使用同一 evaluator、Soft-NMS、数据语义和结果保存协议作最终评估。训练子集、40-video holdout 或其他 pilot
46 |    均不能替代这一证据。
47 | 
48 | 若第三项不通过，则停止当前三档动态转移路线。若仅高时间交并比仍失败，后续是否增加预算和采样密度条件必须由
49 | Pro 另行冻结，不能在本实验中预埋。
50 | 
51 | ## 执行前未决项
52 | 
53 | - 上一轮 Pro 已选择从 H65 Stage-1 terminal EMA 开始、每臂 6,000 次成功 update 的匹配 Stage-2 日程；但该
54 |   回答同时把 200 个训练侧视频切成 160 train / 40 development，因此这一日程在完整训练集上的正式用途仍需
55 |   新 Pro 裁决确认；
56 | - 完整训练 split 与完整官方 held-out evaluation split 的确切配置名称和完整视频 ID；
57 | - 当前记录中 OpenTAD `training/validation`（211 个评估视频）与 ActionFormer `validation/test`（历史记录为
58 |   212 个评估视频）口径如何统一；
59 | - 若保留训练侧开发诊断，其视频 ID、使用时点和与最终完整训练之间的隔离方式。
60 | 
61 | 这些项目未冻结前不建立 Builder 分支、不提交 PRE_RUN 或训练。官方留出评估不得用于训练、checkpoint 选择、
62 | 阈值或规则选择、路线选择或反复调试。当前没有实验结果。
63 | 
64 | ## Pro v001 终态与当前处置
65 | 
66 | 精确 Project 中的 Pro turn `DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831` 已验证完成并选择
67 | `CONTINUE`。其机制隔离、H65 起点、6,000-update 日程、实际 observation 成本校准、terminal EMA 和角色顺序均
68 | 已明确；完整报告保存在 `.cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md`。
69 | 
70 | 该报告在后到人类约束之前提交，并冻结了 160-video train / 40-video development，且不访问 official test。
71 | 这与本页第 44--46 行的正式全量证据要求冲突。当前不把它视为可执行 Builder 授权；一个新的独立 Pro turn 正在
72 | 专门裁决完整训练、完整官方留出评测、211/212 身份差异以及诊断到正式实验的隔离关系。
```

### File: PAPER_PROGRESS.md
Lines: 1-131
```md
  1 | # DUCA 论文缩略报告
  2 | 
  3 | - 更新日期：2026-08-31
  4 | - 名称说明：DUCA 是本项目沿用的方法路线名称。
  5 | - 当前结论：固定 `K=384` 的任务状态时序 coreset 低于匹配均匀选择，`DUCA-Coverage-v1` 也未通过预注册中间机制门。冻结 H65 的三档边际预算诊断、96-state 联合邻域和最终 704-state 整视频枚举均未达到 `+0.8/+1.0` 联合门；Pro 对这一冻结检测器的旧动作空间作出的 `STOP` 继续有效。新的 Pro 裁决在该边界之外选择 `CONTINUE`：保留现有嵌套 K256/K384/K512 位置构造，比较固定 K384 训练与三档多预算适应，并建议从同一 H65 Stage-1 EMA 完成每臂 6,000 次成功更新。但其 160-train/40-development、无 official-test 协议与后到人类要求的完整训练集和完整官方留出评测集冲突，因此当前仍只有设计，没有可执行 Builder 授权、新代码、作业或性能结果；完整 split 身份和正式证据顺序正交由新的独立 Pro turn 裁决。
  6 | 
  7 | ## 1. 论文问题与应用价值
  8 | 
  9 | 离线时序动作检测（Temporal Action Detection, TAD）通常对长视频进行密集的重型视频编码。DUCA 研究一个更节省计算的问题：能否先用低成本模型预测逐时刻动作性和边界重要性，再由确定性规则选择少量高分辨率帧，并进一步为不同视频或窗口分配不同预算，在真实端到端计算下降时保持高时间交并比（temporal Intersection over Union, tIoU）下的边界定位性能。
 10 | 
 11 | 这个问题的应用价值在于：如果低成本语义证据能够可靠地替代大量冗余帧，就可以减少 VideoMAE 等视频骨干网络的实际输入，而不必修改下游检测头或官方评估器。
 12 | 
 13 | ## 2. 当前科学路线
 14 | 
 15 | 长期路线包含两个层次，二者不能混写：
 16 | 
 17 | 1. **语义间接选帧。** 低成本侦察模型学习二元动作性与动作起止边界；确定性采集规则根据这些预测产生有序、非均匀的原始帧位置。侦察模型不直接把帧索引当作主要学习目标。
 18 | 2. **动态预算。** 长期论文主张要求根据逐视频或逐窗口的语义证据决定保留帧数，使重型 VideoMAE 路径真实执行不同工作量。固定 `K=384` 仅用于机制归因、公平对照和回退。
 19 | 
 20 | 已完成的归因实验把 768 帧输入组成 384 个 VideoMAE 原生两帧 tubelet，并固定选择其中 192 个。对照臂在原生 tubelet 网格上均匀选择；候选臂使用冻结侦察器的动作性、边界强度和时序新颖性进行确定性选择。该候选在所有报告阈值上均低于均匀选择，现已终止。
 21 | 
 22 | 固定 K 内的 coverage 和其后的三档预算转移均已结束。后者冻结 H65 Scout 与 detector，并从同一非连续 H65 priority sequence 构造 K256/K384/K512 的嵌套集合，在逐窗口、差分联合邻域和整视频跨视频三个层次搜索不增加真实 observation 成本的重分配。完整开发集 oracle 枚举没有产生预登记的 Avg-mAP 与高 tIoU 联合 headroom。Pro 因而关闭这一动作空间；它不是完整训练后的论文主模型，也不授权 Codex 自动恢复历史路线或自行选择新机制。
 23 | 
 24 | 最新 `REVISE` 将下一项机制检验限定为检测器的多预算适应：第一轮继续使用上述嵌套位置构造，只改变训练时检测器是否同时见到 K256/K384/K512。它不把输入分布不匹配当作已证实根因，也不同时加入预算条件嵌入、蒸馏、Gumbel-Softmax、新 Scout、Mamba、Block Drop 或部署优化。较早附件提出的“预算原生选点 + 多预算训练”与该单变量要求冲突，因而不进入第一轮。
 25 | 
 26 | ## 3. 与官方基线的真实差异
 27 | 
 28 | 共享的未修改 AdaTAD 基线使用官方代码 revision `01c58b9f2370e914150cf94d392208a4e211c053`、seed 42、60 个训练轮次和官方评估器，平均检测精度（Avg-mAP）为 `68.73`；论文公开锚点为 `69.03`。DUCA 只读引用这一共享复现，不重复训练官方 dense 模型。
 29 | 
 30 | DUCA 与官方 dense 模型的目标差异只应来自输入采样与预算：前者先以低成本语义模型选帧，再让 VideoMAE 只处理被选中的高分辨率帧。下游 ActionFormer 检测头、损失、NMS 和官方评估器原则上不变。历史 65.xx 或 66.xx 结果不属于官方 dense 复现，不能代替 `68.73`，也不能在协议不匹配时直接计算方法增益。
 31 | 
 32 | ## 4. 已完成实现与代码身份
 33 | 
 34 | - H65 干净复现使用历史 ASFormer 语义预测、确定性非均匀逐帧选择和固定 `K=384`。当前可审计的 30+60 训练参考冻结于源码 revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。
 35 | - PJST-D1 是当前物理时间表示候选的代码名称。它不改变选择器，只在 VideoMAE 首次二帧 tubelet 混合前校正导数分量所使用的时间间隔。匹配训练冻结于 clean revision `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`；只读重推理与统计入口冻结于 `7bd120f0d342bf175c97c365fba7cbd359df055e`。
 36 | - UVT 诊断分支为 `codex/duca-uvt-utility-value-20260819@df544c78ce515d925dc7019f106fce09a53c09f8`；Fovea/Query-Bridge 诊断分支为 `codex/duca-fovea-query-bridge-20260819@4ae5067100c4490c7110c00a1ad406230ba603cd`。它们与 H65/PJST 不是同提交匹配比较，不能用跨版本差值归因。
 37 | - 物理连续片段路线已完成真实训练，但性能明显下降，现作为负结果保存，不再是当前论文主线。
 38 | - 为支持下一次科学路线与代码主线裁决，当前模型、配置、启动器、测试和研究记忆已同步到 `codex/duca-research-sync-20260828`；代码库存固定提交为 `5136011ed57df8a639427a633a488a592ba95924`。该快照用于检查实现重叠与历史分支，不是经过运行前检查的实验候选，也不能替代各实验的 clean revision。
 39 | - 先前的动态窗口预算候选冻结于 `codex/duca-semantic-budget-matched-20260828@36d75c146492a38eb8966c66ff6b2881938cf3c6`。它在取得效能结果前被后续科学裁决暂缓，保留为下一阶段的实现参考，不能作为动态预算证据。
 40 | - 当前原生 tubelet 时序 coreset 候选位于独立干净分支 `codex/duca-native-tubelet-coreset-20260828`，冻结提交为 `b33391126eac05e3353d322b973dda91741f0732`。实现包含固定 192 个原生 tubelet、任务状态驱动选择、端点/空洞覆盖、低分辨率上下文回收、物理时间重建、严格匹配的均匀对照、配置、聚焦测试与 N16R4 启动器。N16R4 环境的 20 项相关测试通过，独立只读审查未发现阻断性缺陷；两臂均完成 60 轮训练和官方验证，但结构化证据保存失败。
 41 | - 窗口级动态预算候选位于独立干净分支 `codex/duca-dynamic-native-tubelet-budget-20260829`，冻结提交为 `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`。实现按视频生成需求排序和 16/20/24-clip 预算，在各预算内执行确定性均匀 tubelet 选择，并按真实 clip 数分组调用 VideoMAE；较短预算不会在重型骨干前补齐到 24。短窗口若不能容纳分配预算会明确退出。Python 编译、启动器语法、纯启动器测试和独立静态审查已通过；尚无运行前检查或正式实验结果。
 42 | - 当前 `DUCA-Coverage-v1` 候选位于独立干净分支 `feature/duca-coverage-only-v1-20260829`，当前提交为 `048143124e2a36a76575200ae17d6f42ec79ea3a`，基于 H65 正式提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`。实现新增固定预算设施位置选择器，并提供 matched H65 对照、真实训练样本无标签重放门、60 轮配置、恢复合同和 N16R4 启动器。修正后的 PRE_RUN Job `1261679` 已执行 27 项测试和 200 个真实 training 样本重放，但因预注册覆盖/空洞干预条件未满足而在 smoke 前停止。代码已同步 GitHub并部署到 N16R4 干净目录 `/data/run01/sczc063/yuzibo/duca_coverage_v1_04814312_20260830`。
 43 | - 最新 Pro 裁决指定从 H65 clean revision `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 建立 `feature/duca-marginal-budget-v1-20260830`。吸收短窗口修订后的实现为 `be5bb8033c0b11c628394d268c1923ab398c04ed`；测试合同修复为 `f87555f7da362fe1a20d4ca08f7a68c975ed8280`。该提交完成 K384、K256 和 K512 冻结反事实产物后，汇总器因把换行文本 block-list 当作 JSON 读取而退出。最新提交 [`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889) 只为官方评估器生成确定性的 JSON block-list 适配文件并加入回归测试；没有修改模型、分配器、预测、损失、数据、NMS、指标或科学门槛。恢复作业没有重跑三个冻结 producer 阶段，只在当前 clean commit 上重做运行前身份核验并汇总既有产物。producer 产物保留其 `f87555f7...` 来源，汇总与最终结果绑定 `f67d96fd...`；两者的配置、checkpoint、annotation、类别映射、预训练权重及其哈希一致，这一跨提交来源会在后续 Pro 材料中明确披露。
 44 | - Pro 对灰区的后继实现位于 [`feature/duca-marginal-cap-release-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831)，精确提交为 [`d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f)。它只增加独立的 `max_changed_fraction=1.0` 只读汇总入口和聚焦测试，默认 `0.5` 汇总路径、三档 producer、模型、数据、NMS 与门槛均不变。N16R4 的 14 项聚焦测试和独立 Critic 已通过。
 45 | - 最新的联合邻域诊断实现位于 [`feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831)，精确提交为 [`46812facc8773d9b4a9c21833cbe397c8aaa5a2d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d)。它只修改 probe runner 与聚焦测试；`dynamic_budget.py` 相对父提交逐字不变。实现从密封分配和真实 observation 成本自动导出 8 个最小合法转移、6 个净转移组及 96 个唯一联合状态，没有为多解视频硬编码配对。N16R4 上 16 项聚焦测试、23 项既有回归测试和独立 Critic 均通过。
 46 | - 整视频最终 falsifier 位于 [`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-whole-video-consistent-budget-falsifier-v1-20260831)，权威公开提交为 [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535)。父提交 `c27d77...` 的首次 PRE_RUN `1262147` 暴露密封 proposal 被额外排序、从而改变 Soft-NMS 并破坏锚点复现的确定性证据错误；`33e4ed...` 只恢复密封 producer 原始顺序并增加回归测试，不改变候选、成本、预测、NMS、评估器或三档预算。28 项聚焦测试与独立 Critic 通过。修正后 PRE_RUN `1262161` 复现全部锚点；唯一恢复 Job `1262190` 完成 704 个候选。终态 JSON SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`，通过候选数为零。Pro 随后使用该最新 GitHub 身份作出项目级 `STOP`；该分支与 Marginal-v1、cap-release、96-state 分支均只作为负证据读取。
 47 | - 新设计的多预算检测器适应实验以 H65 干净提交 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 为模型基座，并且只允许从 `33e4ed...` 移植真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始生成顺序保持。当前尚未建立实现分支或提交；`33e4ed...` 不是新模型的科学基座。
 48 | 
 49 | 项目根目录承担多路线协调，工作区可能包含尚未归档的修改。论文实验身份必须引用上述独立 clean revision，而不能用项目根当前 `HEAD` 代替。
 50 | 
 51 | ## 5. 正式实验协议
 52 | 
 53 | 正式比较要求两臂使用完整 THUMOS14 训练集，并在设计与全部选择规则冻结后，于完整官方 held-out evaluation
 54 | split 上使用同一检测器、损失、Soft-NMS、评估器和预登记模型选择规则作最终比较。训练侧子集只可用于前置诊断，
 55 | 不能替代正式证据；官方留出评估不得参与调参、checkpoint/阈值/规则选择或路线迭代。当前项目同时存在 OpenTAD
 56 | `training/validation`（211 个评估视频）和 ActionFormer `validation/test`（历史记录为 212 个评估视频）两种命名与
 57 | 视频集合口径，具体 subset 名称和完整视频 ID 必须由 Pro 冻结，不能由 Codex 静默等同。当前已有实验主要使用
 58 | seed `3407`；单种子结果不能支持稳定性或显著性结论。
 59 | 
 60 | 当前与后续实验矩阵按以下顺序组织：
 61 | 
 62 | 1. 只读引用共享的官方 dense AdaTAD 结果；
 63 | 2. H65 固定 `K=384` 的 30+60 参考训练；
 64 | 3. PJST-D1 固定/重放选择结果的物理时间表示 OFF/ON 配对已完成点估计，但平均点差为负且配对区间缺失；它不再是当前优先开发项；
 65 | 4. 固定 `K=384` 的原生 tubelet 均匀选择与任务状态驱动时序 coreset 已完成；细粒度 coreset 候选因一致的负向点估计终止；
 66 | 5. 同一 H65 代码基座上的 matched allocation control 与 `DUCA-Coverage-v1` 已完成真实训练样本无标签重放门；该门未通过，因此两个 60 轮完整训练臂没有启动。当前对照实现是预算校准系统采样，而冻结设计曾概括为 Top-K；在 Pro 裁决该基线身份和失败机制前不得重提正式训练。
 67 | 6. 冻结 H65 的 K256/K384/K512 反事实边际预算实验、cap-release 诊断和 capped→released 差分邻域的 96 个联合状态枚举均已完成。50% 上限 oracle 的 Avg-mAP/mAP@0.7 增益为 `+0.726/+0.729` 个百分点；解除上限后反而降为 `+0.427/+0.450`。96-state 中没有状态同时达到 `+0.8/+1.0`；当前加性 Marginal-v1 及其本次视频级联合效用修复均按冻结规则停止。
 68 | 7. 整视频一致预算的跨视频单次转移 oracle 已完成：donor 视频所有窗口请求 K256，recipient 视频所有窗口请求 K512，其余视频保持 K384；候选总实际 observation 成本不超过固定 K384 的 `47110`。候选集合在读取标签或指标前完整生成，随后仅复用密封预测和相同评估器。1560 个有序对中 704 个合法候选完成评估，没有候选通过联合门；没有执行模型前向、训练、bootstrap 或 official test。
 69 | 8. 新设计的单变量实验比较固定 K384 训练与 K256/K384/K512 多预算检测器适应。Pro v001 已选择两臂从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始、各完成 6,000 次成功更新，并让候选概率按短窗口后的实际 observation 成本匹配 K384；但同一报告的 160/40 划分不能满足正式证据要求。正式两臂仍必须使用完整训练集，并在完整官方留出评估集上完成可比评估；任何训练侧开发划分只承担前置机制诊断。完整 split 身份、211/212 差异和诊断到正式训练的隔离方式尚未冻结，所以当前不进入实现或训练。
 70 | 
 71 | 主要指标为 tIoU 0.3、0.4、0.5、0.6、0.7 下的 mAP，以及五个阈值的平均值；同时报告短动作、边界定位和完整端到端计算成本。任何计算节省主张都必须来自实际执行的 VideoMAE 工作量和相同硬件条件下的测量，不能由 padding 后的名义帧数推断。
 72 | 
 73 | 完整训练至少每 5 个训练轮次保存一次可恢复的 PyTorch `.pth` 检查点；如果未修改的官方配置保存更频繁，则保留官方间隔。恢复包应包含模型、指数移动平均模型、优化器、学习率调度器、混合精度缩放器、训练轮次或更新计数及随机状态，并至少保留最近三个有效恢复点、预定义里程碑和最终检查点。最终结果使用预先登记的最终模型或最终指数移动平均模型，不能根据中间验证结果事后挑选。
 74 | 
 75 | ## 6. 已取得的性能结果
 76 | 
 77 | 除特别说明外，Avg-mAP 是 tIoU 0.3 至 0.7 五个阈值下 mAP 的平均值。
 78 | 
 79 | | 实验 | Avg-mAP | mAP@0.7 | 证据边界 |
 80 | |---|---:|---:|---|
 81 | | 共享官方 dense AdaTAD | 68.73 | 未在本报告重复摘录 | 一次共享复现；论文公开 Avg-mAP 为 69.03 |
 82 | | H65 30+60 | 65.13 | 43.31 | 单种子、完整训练；当前 H65 参考 |
 83 | | H65 20+40 | 62.46 | 39.94 | 同 seed 的训练压缩诊断；比 30+60 低 2.66/3.37 点 |
 84 | | H65 30+30，AM-RPCH25 | 63.22 | 41.25 | 只改变第二阶段学习率日程 |
 85 | | H65 30+30，LongCosine-H6000 | 63.56 | 41.01 | 更慢衰减仍未恢复 30+60 |
 86 | | RankPack K384 | 61.57 | 37.10 | 单种子物理时间表示对照 |
 87 | | TrueTime K384 | 62.19 | 37.89 | 相对 RankPack 为 +0.62/+0.79 点；尚无配对区间 |
 88 | | PJST-D1 OFF | 65.063 | 43.646 | 211/211 视频，冻结 H65 选择结果 |
 89 | | PJST-D1 ON | 64.591 | 43.769 | 211/211 视频；相对 OFF 为 -0.472/+0.123 点 |
 90 | | 原生 tubelet uniform K384 | 64.13 | 42.45 | 单 seed；60 轮训练和官方验证完成；结构化指标文件未封存 |
 91 | | 原生 tubelet coreset K384 | 62.81 | 40.56 | 单 seed；相对匹配 uniform 为 -1.32/-1.89 点；结构化指标文件未封存 |
 92 | | 连续片段 FZ | 49.89 | 29.68 | 完整单种子训练；明显负结果 |
 93 | | 连续片段 JT | 47.24 | 26.52 | 完整单种子训练；联合训练未恢复性能 |
 94 | 
 95 | PJST-D1 两臂各产生 422,000 条预测，视频集合一致，重推理逐项复现了原始点估计。预先登记的 10,000 次整视频配对自助法在任何抽样前退出，因为统计程序指向了错误的预测文件目录。因此当前只有点估计，没有置信区间；不能据此宣布总体效应显著为负，也不能把 mAP@0.7 的 `+0.123` 点解释为真实收益。
 96 | 
 97 | UVT 的 legacy/geometry/geometry+EMA 三臂 Avg-mAP 为 `57.35/55.93/55.92`。Fovea/Query-Bridge 第一波中最佳 `query_cycle` 为 `54.67`。这些结果来自不同代码版本与实验合同，只能说明相应首版设计未恢复 H65 性能，不能用来量化某一个组件的因果影响。
 98 | 
 99 | 历史 `65.3857` 是 H65 语义间接非均匀逐帧选择的 30+60 诊断结果；`65.696` 来自改变物理检测网格的探索实现。后者同时改变了检测器时间几何，因此不是与官方原生检测器严格匹配的输入采样对照。当前干净 H65 复现 `65.13` 是更适合作为后续同代码、同训练协议比较的参考。
100 | 
101 | 当前还没有与主路线匹配、可用于论文的完整端到端成本结果，也没有动态预算保持性能或提高效率的实验证据。`DUCA-Coverage-v1` 已完成代码实现、独立审查和真实 training 数据运行前门；该门给出不满足预注册干预条件的诊断结果，但没有进入训练，因此不是 mAP 或成本结果。
102 | 
103 | `DUCA-Marginal-v1` 的冻结 detector 诊断在训练侧 40 个 utility holdout 视频、124 个窗口上比较了固定 K384 与使用真实反事实效用的等预算 oracle。固定臂 Avg-mAP/mAP@0.7 为 `88.131/76.271`，50% 上限 oracle 为 `88.857/77.000`。解除上限后，分配从 K256/K384/K512 的 `11/102/11` 变为 `17/90/17`，但结果降为 `88.559/76.721`；相对固定臂只有 `+0.427/+0.450` 个百分点。总 observation 预算仍精确为 `47110`。这里的百分数只描述训练侧 controller holdout，不可与 official validation/test 表直接比较。强 headroom 门未通过，因此 utility predictor、正式测试、配对区间和端到端成本均未运行。
104 | 
105 | 随后只在 capped 与 released 分配不同的 12 个窗口上枚举了全部 96 个逐视频等成本联合状态。最佳 Avg-mAP 状态相对固定 K384 为 `+0.733` 个百分点，但 mAP@0.7 仅 `+0.479`；最佳 mAP@0.7 状态为 `+0.549/+0.934`；按两项联合门最优的状态为 `+0.554/+0.933`。没有状态同时满足 `+0.8/+1.0`，也没有单个最小合法转移同时改善 Avg-mAP 与 mAP@0.7。该诊断没有执行模型前向、训练、official test 或 bootstrap；从 96 个开发集状态中事后选出的最优状态不能当作可部署策略或论文主结果。
106 | 
107 | 整视频一致预算的最终开发集 falsifier 在相同固定 K384 锚点 `88.1312%/76.2706%`（Avg-mAP/mAP@0.7）和真实成本 `47110` 下完成 704 个合法状态。Avg-mAP 最优状态的变化为 `+0.6942/-0.0436` 个百分点，mAP@0.7 最优状态为 `-0.2359/+0.4970`，联合门余量最优状态为 `+0.1474/+0.4898`；通过候选数为零。该结果只属于训练侧 controller holdout 的事后 privileged oracle 证伪，没有模型前向、训练、官方验证/测试、配对区间或可部署策略含义。
108 | 
109 | ## 7. 结果解释与已停止的方向
110 | 
111 | - 20+40 训练压缩、AM-RPCH25 和 LongCosine-H6000 均未恢复 H65 30+60 的性能。现有证据说明简单压缩预热或只修改第二阶段学习率尾部不足以保持性能；它不否定 H65 的语义间接选帧机制。
112 | - 连续 16 帧片段采样在真实训练中造成大幅定位损失，联合训练也未恢复，因此该采样单元不再作为当前主路线。这个结果不否定低成本语义侦察或物理时间一致性的一般问题。
113 | - TrueTime 相对 RankPack 有小幅单种子提升，但证据不足以形成论文主张。
114 | - PJST-D1 的当前点估计没有显示平均性能收益；缺少配对置信区间意味着总体效应仍未完成统计裁决。统计程序的路径错误是证据生成失败，不是模型的科学失败。
115 | - UVT 与 Fovea/Query-Bridge 同时改变了选择分数、预算证据或训练信息流，且缺少与 H65 同提交的严格隔离，因此其性能下降不能归因于单一组件。
116 | 
117 | ## 8. 当前证据缺口与下一动作
118 | 
119 | THUMOS14 原始视频、注释、类别映射、VideoMAE-S 预训练权重、H65 Stage-1 侦察器检查点和共享官方 AdaTAD 结果均已核验。PJST-D1 的 OFF/ON 完整推理已经结束。
120 | 
121 | 当前固定预算归因已经得到负向点估计：在完全相同的 `K=384` 高分辨率帧预算、训练日程和检测器下，任务状态驱动 coreset 没有优于原生 tubelet 均匀选择，并在 tIoU 0.5 至 0.7 下降 `1.54/2.03/1.89` 个百分点。这个结果优先要求分析端点覆盖、最大空洞、跨 tubelet 打包、低分辨率上下文回收和选择分数是否共同损害重型表示，而不能直接把固定预算重命名为最终方法。
122 | 
123 | PJST-D1 的配对区间仍属未完成证据，但它不会改变当前路线，因此不作为当前任务补齐。原生 tubelet 两臂已从干净 H65 基座完成实现、N16R4 聚焦测试、独立审查和 60 轮训练。uniform `1260184` 与 coreset `1260185` 都写出 epoch-59 检查点并完成官方 211 视频评估，日志点估计分别为 `64.13%` 和 `62.81%`。两臂随后因同一个证据封存错误退出：配置没有保存预测，结构化指标入口因此拒绝写出 `metrics_epoch59_ema.json`。当前没有配对区间或成本结果；训练成功、日志点估计和结构化证据缺失必须分别陈述。
124 | 
125 | Pro 冻结的 96-state 联合 mAP 邻域诊断已经在公开提交 `46812fac...` 上完成。原 fixed/capped/released 三结果复现误差为 `0.0` 个百分点，96 次评估保持逐视频预算和全局成本 `47110`；没有状态通过 `+0.8/+1.0` 联合门，也没有单个最小合法转移同时改善两个门指标。Pro 已据此最终裁决 `STOP`：现有加性 Marginal-v1 及本次邻域修复关闭，不再重跑、改门、补 bootstrap、训练 utility head 或访问 official test。当前分支只作为负证据读取；未来若重新研究动态计算，必须由 Pro 以新的机制假设和独立任务启动，不能作为 Marginal-v1 的恢复。
126 | 
127 | 项目级 Pro 对旧三档预算转移动作空间的 `STOP` 继续有效。Pro v001 已把“检测器只适应 K384”冻结成一项新的、边界之外的可证伪实验，并选择同起点 6,000-update 的匹配训练，但其 160-train/40-development、无 official-test 协议与完整数据要求冲突，因而未交给 Builder。下一步是新的独立 Pro turn 冻结完整训练 split、完整 official held-out evaluation split、各自完整视频 ID、211/212 差异、一次性评测边界，以及训练侧诊断与最终正式训练的隔离方式。冻结后才建立 Builder；独立 Critic 必须验证只有训练预算分布发生变化，Evaluator 再执行完整训练和预登记的完整留出评估。K384 安全门为相对同更新数控制的 Avg-mAP 与 mAP@0.7 均不低于 `-0.2` 个百分点；等成本 oracle 继续门仍为 `+0.8/+1.0` 且实际成本不高于固定 K384。
128 | 
129 | ## 9. 可发表性边界
130 | 
131 | 目前可以写入论文的事实是：H65 的 30+60 训练参考明显优于已测试的 60 轮压缩日程；连续片段采样是明确负结果；PJST-D1 的匹配点估计没有平均收益；原生 tubelet coreset 的单种子点估计比匹配均匀选择低 `1.32` 个百分点；当前 96-anchor Coverage 干预没有达到预注册中间机制目标。DUCA-Marginal 的训练侧 oracle 在 50% 上限下仅有灰区增益，解除上限后增益进一步降至 `+0.427/+0.450` 个百分点；其 96-state 联合差分邻域和最终 704-state 整视频单次转移 falsifier 均没有达到 `+0.8/+1.0` 的预注册门，因此当前冻结动作空间没有足够开发集 headroom。Pro 将其定位为可用于未来论文补充失败分析的内部终态证据，而不是独立 CVPR 主结果。它没有证明 Scout 能预测预算效用，也没有 official test、统计区间、可部署 controller 或端到端成本结果。不能把该停止扩展为所有 coverage、动态预算、budget-conditioned training、内部 token/层级条件计算或物理时间方法无效，也不能声称 DUCA 有效、优于 dense，或已经获得性能—成本联合优势。
```

## Answer

以下为可直接保存并交付 Codex 的终态科学裁决。

# DUCA 多预算适应：完整训练与完整官方留出评测协议裁决

## 1. `SESSION_ASSERTION`

**Nonce：`DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`**

* ChatGPT Project ID：`g-p-6a91061f789881918ccd8357ca3d6c92`
* GitHub 仓库：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* H65 新实验模型与训练基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
* whole-video 诊断功能来源：`33e4ed137c33eef07f0452b44506a6993bdf7535`
* 被核验的上一轮 Pro 裁决：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`

身份关系冻结如下：

1. `04c35a3b...` 是固定 K384 H65 的正式模型与训练基座。
2. `33e4ed...` 不是新模型基座，只允许移植真实变长 VideoMAE 执行、mixed-budget packet 对齐、actual-observation 计数、K384 parity、whole-video 输出组织和 producer 原始 proposal 顺序保持。
3. 代码库存分支、项目根工作区和历史版本注册表仅用于追溯组件，不能替代干净实验提交。项目材料也明确区分了“代码可见”与“实验可归因”。 
4. Pro 继续独立负责科学变量、数据语义、门槛、停止条件与论文主张；Builder、Critic 和 Evaluator 只执行冻结后的事实核验、实现与评测。
5. 旧材料包和历史问询用于确认项目谱系与原有科研边界，不自动决定本轮数据协议。 

当前没有：

* 新的多预算适应 Builder 分支；
* 正式 PRE_RUN；
* 两臂完整训练；
* 完整官方留出评测；
* 新模型性能或成本结果。

代码存在、局部测试通过、作业成功和模型有效性必须继续分别陈述。

---

## 2. `SCIENTIFIC_DECISION`

**唯一裁决：`REVISE`**

不是 `PIVOT`，因为上一轮提出的单变量科学问题仍然成立：

> 在保持 H65 Scout、嵌套 K256/K384/K512 位置构造、下游检测器、损失、物理时间映射、Soft-NMS 和评价器不变时，仅改变 Stage-2 训练期间模型接触的预算分布，能否改善跨预算检测兼容性？

也不是 `STOP`，因为冻结 K384 检测器后进行预算重分配的旧负结果，没有直接检验完整多预算训练。

本轮只修订上一轮已经失效的数据与证据协议：

### 2.1 保留的科学和训练选择

继续冻结：

* 固定 K384 训练控制臂；
* 嵌套 K256/K384/K512 多预算适应臂；
* 唯一干预变量为训练预算分布；
* 两臂共同从 H65 Stage-1 `epoch_29/state_dict_ema` 开始；
* 每臂恰好 6,000 次成功 optimizer update；
* 相同优化器、学习率日程、可训练参数集合、随机种子和 EMA 规则；
* 多预算概率按 actual-observation 成本校准；
* update 6,000 的 terminal `state_dict_ema` 为唯一结果模型。

### 2.2 撤销的上一轮协议

以下内容全部撤销，不得进入 Builder：

* 160-video train / 40-video development 正式划分；
* 将旧 40-video controller holdout 重新用于评价；
* 任何训练侧有标签 mAP 门；
* 40-video whole-video oracle；
* 根据开发结果选择 donor、recipient、概率、阈值或路线；
* “本轮不访问完整官方留出集”的旧限制。

训练侧 160/40 或旧 40-video 结果只能作为历史诊断，不能成为当前论文比较的一部分。真实 benchmark 上的完整训练和完整评测才可能形成论文证据；子集、pilot、smoke 和基础设施成功不能替代它。

### 2.3 继续有效的旧停止边界

以下路线继续永久只读：

* Marginal-v1；
* cap-release；
* 96-state 联合邻域；
* 704-state whole-video 枚举；
* 旧 40-video holdout 上的 privileged oracle；
* 冻结 K384 detector 的旧预算重分配动作空间。

不得重跑、扩张、改门、补事后 bootstrap、训练旧 utility head，或把本实验写成旧路线的恢复。

---

## 3. `FULL_TRAIN_IDENTITY`

### 3.1 冻结的数据语义

正式两臂采用 **H65/OpenTAD 项目原生数据语义**，而不是把 ActionFormer 的 split 名称直接替换进 H65 配置：

* 配置 subset 字面值：`training`

* annotation：

  `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`

* class map：

  `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`

* video root：

  `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`

完整训练 ID 集的权威定义为：

```text
T_full =
sort_unique({
  video_id |
  thumos_14_anno.json[video_id].subset == "training"
})
```

冻结要求：

* `|T_full|` 必须严格为 `200`；
* 两个正式训练臂都使用全部 `T_full`；
* 排除规则为空；
* 旧 40-video holdout 必须回到训练集合中；
* 不允许因视频短、无某类别、窗口少、解码较慢或历史上属于 holdout 而排除；
* 任一视频文件缺失、不可解码或被 loader 静默丢弃，都属于数据身份 blocker，而不是合法排除。

### 3.2 尚不能伪造的文字清单

现有材料确认了“规范训练侧为 200 个视频”，但没有逐项提供：

* `04c35a3b...` 中实际 H65 Stage-1/Stage-2 config 的精确文件路径；
* 200 个完整视频 ID 的逐行清单；
* config 继承后最终解析出的 dataset 字段。

因此，本裁决不编造配置路径或视频名称。科学选择已经冻结为上述 `training` 集合；唯一当前任务必须从精确代码和本地 annotation 中只读物化：

* `full_train_video_ids.txt`
* 精确 config 路径；
* 解析后的 subset 字段；
* loader 实际输出 ID 集；
* 200 个物理视频文件的一一对应结果。

### 3.3 与 H65/AdaTAD 的公平可比性

选择 H65/OpenTAD 的 `training` 语义，是因为它保持：

* H65 原有数据加载语义；
* 相同 annotation；
* 相同类别映射；
* 相同窗口生成与 GT 坐标处理；
* 相同 AdaTAD/ActionFormer 下游；
* 相同损失、Soft-NMS 和 evaluator。

两臂的训练数据、监督、采样 occurrence、更新数和优化完全匹配，只改变预算暴露。因此结果可以归因于多预算训练，而不能归因于换 split 或换 annotation。

---

## 4. `HELD_OUT_EVALUATION_IDENTITY`

### 4.1 冻结的项目入口

正式 held-out evaluation 采用：

* 配置 subset 字面值：`validation`

* annotation：

  `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`

* class map：

  `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`

* video root：

  `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`

* evaluator：现有 `tools/test.py` THUMOS14 evaluator

* tIoU 阈值：`0.3, 0.4, 0.5, 0.6, 0.7`

* Soft-NMS：

  * `sigma=0.7`
  * `max_seg_num=2000`
  * `multiclass=True`
  * `voting_thresh=0.7`

此前 PJST-D1 终态评测已经证明：两臂使用同一 annotation、class map、评价配置和 evaluator source，均产生 211/211 个完全一致的视频 ID 集。 

但这只证明该次执行评了 211 个一致视频，**不能证明 211 就是完整官方 held-out 集，也不能证明它与历史 ActionFormer 212-video `test` manifest 完全等同。**

完整 held-out ID 集的权威定义先冻结为：

```text
E_annotation =
sort_unique({
  video_id |
  thumos_14_anno.json[video_id].subset == "validation"
})
```

最终 `E_full` 必须通过下述唯一规则确定。

### 4.2 211/212 差异的唯一处置

Codex 不得选择 211 或 212。只读身份核验必须得到以下四种事实之一：

#### 情况 A：annotation 有 212 个 `validation` ID，现有 loader 或历史预测只有 211 个

必须列出缺失 ID 和丢失原因。

* 若原因是缺文件、解码失败、路径错误、loader 静默过滤或预测缺失：不得排除该视频；先修复数据或 loader，正式 `E_full=212`。
* 历史 211-video 指标只能保留为旧协议结果，不能与新的 212-video正式结果直接做点差。

#### 情况 B：权威本地 annotation 本身只有 211 个 `validation` ID

若 loader、物理视频和 evaluator 恰好输出同一 211 个 ID，则：

* `E_full=211`；
* ActionFormer 的 212-video 记录必须标为来自不同 annotation 或不同 manifest；
* 不把其额外视频静默并入 H65/OpenTAD 协议。

#### 情况 C：annotation 有 212 个 ID，但官方来源明确规定排除一个精确视频

只有同时满足以下条件，才允许 `E_full=211`：

* 找到可核验的官方或项目上游 source；
* source 明确给出排除规则；
* 列出被排除的完整视频 ID；
* 证明 H65、两个新训练臂和 evaluator 使用同一规则；
* 排除与模型预测、类别表现或 mAP 无关。

#### 情况 D：无法得到来源支持的解释

若出现以下任一情况：

* 找不到历史 212-video manifest 的来源；
* 无法解释具体差集；
* loader 有未声明过滤；
* annotation、物理文件、loader 或 evaluator ID 集不一致；
* 数量不是 211 或 212；
* 所谓排除只能由经验猜测解释；

则输出 `BLOCK`，不得开始模型实现、PRE_RUN 或训练。

缺失信息必须继续保持未知，不能以“通常 THUMOS14 是……”补齐。科研规则明确要求冲突信息保持可见，并由精确证据而非方便叙述解决。

### 4.3 明确禁止的 held-out 排除

不得因为以下原因删除视频：

* 无目标类别实例；
* annotation 为空；
* 视频较短；
* proposal 为空；
* 模型没有输出；
* 指标较差；
* 解码失败；
* 历史运行没有包含；
* 为了复现 211 这个数字。

只有上节情况 C 中的明确官方排除才有效。

### 4.4 与历史基线的关系

最终两条新臂在同一 `E_full` 上的直接因果比较始终成立。

但是：

* 只有当共享 AdaTAD、H65 65.13 和 PJST 结果的逐视频 manifest 与 `E_full` 完全相同时，才允许计算直接点差；
* 若最终 `E_full` 与历史 211-video manifest 不同，历史结果只能作为上下文锚点；
* 不重复训练共享 dense AdaTAD；
* 本轮也不因历史 baseline 身份不匹配而改变两臂主比较。

### 4.5 held-out 禁止用途

`E_full` 不得用于：

* 模型训练；
* checkpoint 选择；
* early stopping；
* 预算概率选择；
* K 档位选择；
* NMS、分数阈值或 proposal 数选择；
* mixed-budget manifest 修改；
* donor/recipient 搜索；
* whole-video oracle；
* controller 结构、阈值或监督设计；
* 路线选择后再回改方法；
* 根据第一次结果补训某一臂。

项目规则已经明确禁止 validation/test GT 参与选择、teacher leakage 和 raw-prediction shortcut。

---

## 5. `DIAGNOSTIC_TO_FORMAL_SEQUENCE`

### 5.1 删除有标签训练侧开发门

本轮不保留任何 160/40 或旧 40-video 的有标签开发评价。

删除：

* development mAP；
* development K384 安全门；
* development whole-video oracle；
* 根据 development 结果决定是否完整训练；
* 根据 development 结果修改概率、门槛、损失或训练长度。

这样可以让两个正式训练臂都使用完整 200-video 训练集。

### 5.2 只保留无标签运行前诊断

正式训练前只允许回答实现正确性和成本问题：

* split ID 与物理视频身份；
* train/held-out 无交集；
* K384 parity；
* K256/K384/K512 真实变长执行；
* packet 回填顺序；
* selected-position 数量；
* actual-observation 单调性；
* 数值有限性；
* 梯度、AMP、EMA 是否工作；
* 两臂初始化、trainable 参数和优化器是否匹配；
* 训练预算序列的 actual-observation 成本；
* held-out inference manifest 是否已在无标签条件下冻结。

这些检查不产生模型有效性证据。

### 5.3 唯一无泄漏顺序

正式顺序冻结为：

1. **只读数据身份核验**
   物化完整训练和 held-out ID 清单，解决 211/212。

2. **冻结全部选择规则**
   在任何 held-out 动作类别、时间段或 mAP 可见之前，冻结：

   * 代码；
   * 两臂配置；
   * train/held-out manifests；
   * annotation、class map 和 evaluator identity；
   * 训练预算概率；
   * 训练预算 occurrence 序列；
   * held-out fixed mixed-budget manifest；
   * 训练长度；
   * terminal EMA 规则；
   * 所有指标；
   * paired bootstrap；
   * 安全门、继续门和停止条件。

3. **Builder 完成最小模型实现**
   只实现上一轮冻结的多预算训练差异。

4. **独立 Critic 审查同一精确提交**
   审查机制隔离、公平性、数据泄漏、真实变长执行和评价语义。

5. **Evaluator 无标签 PRE_RUN**
   允许读取视频 ID、subset、媒体和必要的时长元数据；不得读取 held-out 动作类别、时间边界或指标。

6. **两个正式训练臂使用全部 `T_full` 完成 6,000 次成功更新**。

7. **密封 terminal 模型和全部预测**
   两臂、三个固定预算及预注册 mixed-budget 输出全部生成并保存后，冻结：

   * checkpoint SHA；
   * config；
   * prediction SHA；
   * 每个输出的视频 ID；
   * proposal 原始顺序；
   * actual-observation 成本。

8. **一次性开放 held-out 动作标签并完成统一评测**
   “一次性”指一次预注册评测事件，可以同时计算全部预注册臂、预算、指标和 bootstrap；不是只允许调用一个指标函数。

9. **永久退出当前 held-out 的方法开发用途**
   结果只交回 Pro 解释；不得根据结果修改当前方法或继续在同一 held-out 上搜索。

确定性的保存、路径或调度故障若不改变密封预测、数据、评价器、门槛和结果选择，只允许最小恢复，并必须与科学结果分开报告。此前 PJST 的 0/10,000 bootstrap 正是证据生成失败，不能被写成模型结论。

---

## 6. `MATCHED_TRAINING_FREEZE`

### 6.1 两个正式训练臂

#### 固定预算控制臂

* 每个训练窗口请求 `K384`；
* 短窗口 actual observation 为 `min(valid_observations,384)`。

#### 多预算适应臂

* 每个训练窗口请求 `K256`、`K384` 或 `K512`；
* 三档位置来自同一 H65 priority sequence，保持嵌套；
* 短窗口按 `min(valid_observations,K)` 折叠；
* 不分别重新运行“预算原生”选点。

两臂只在训练预算暴露上不同。

不得修改：

* Scout 输入或结构；
* priority scoring；
* selector；
* 嵌套位置关系；
* VideoMAE 语义；
* Temporal Adapter；
* ActionFormer/AdaTAD 检测器；
* 分类和回归损失；
* 物理时间逆映射；
* Soft-NMS；
* evaluator；
* annotation 或 class map。

### 6.2 共同起点

两臂共同加载：

```text
/data/run01/sczc063/yuzibo/
duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/
gpu1_id0/checkpoint/epoch_29.pth
```

冻结身份：

* expected epoch：`29`
* state key：`state_dict_ema`
* SHA-256：

  `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`

加载权重后，两臂都重新初始化：

* optimizer；
* scheduler；
* AMP scaler；
* EMA accumulator；
* successful-update counter；
* selector loss-weight schedule step；
* sampler state；
* 数据增强随机状态；
* dropout/drop-path 随机状态。

不得恢复 H65 Stage-2 terminal optimizer、scheduler 或 EMA。

### 6.3 训练长度和停止时钟

每臂：

* 恰好 `6,000` 次成功 optimizer update；
* 非有限 loss、AMP overflow 或未完成梯度更新不计数；
* 两臂成功更新数必须相同；
* 第 `3,000` 次成功更新保持为 Stage-2 前后半程边界；
* 前 `500` 次成功更新为 warmup；
* 训练集 iterator 循环使用全部 200-video dataset，直到完成 6,000 次成功更新；
* sampler trace 必须证明 200 个训练 ID 都实际产生过训练 sample。

“完整训练集”指训练数据域是完整 200-video 集合，不允许把固定更新数误写成 60 个物理数据轮次。

### 6.4 优化与数值合同

两臂逐项相同：

* optimizer：`AdamW`
* 主学习率：`1e-4`
* weight decay：`0.05`
* adapter 学习率：`2e-4`
* Scout coarse trunk：`1e-5`
* Scout action head：`2e-5`
* transition scorer：`5e-5`
* scheduler：`LinearWarmupCosineAnnealingLR`
* warmup：500 次成功更新
* 总日程：6,000 次成功更新
* gradient clipping：global norm `1`
* AMP：开启
* EMA：开启
* batch size：每卡 `2`
* seed：`3407`
* 相同 DDP、unused-parameter、损失归一化和梯度累积规则。

若 K512 在 batch size 2 下显存不足，只允许两臂同时改为：

* microbatch `1`；
* 累积 2 个样本；
* 每两个样本完成一次 optimizer update。

不得只改变多预算臂。

### 6.5 随机性匹配

两臂使用相同的：

* sample ID 与 occurrence 次序；
* random truncation；
* resize/crop/flip/color augmentation；
* 模型初始化；
* optimizer update 次序；
* sample-keyed augmentation seed。

多预算臂额外预算随机流使用独立命名空间：

```text
SHA256(
  "DUCA-MBDA-BUDGET-v001|3407|"
  + successful_update + "|"
  + sample_id + "|"
  + occurrence
)
```

该随机流不得消耗数据增强或模型随机流。

### 6.6 完整训练集上的成本校准

在全部 `T_full` 的冻结 6,000-update sample occurrence 计划上，使用无标签重放计算：

* `μ256`
* `μ384`
* `μ512`

其中 `μK` 是请求预算 K 时的平均 actual observation。

固定：

```text
p384 = 0.50
p256 + p512 = 0.50
```

唯一解为：

```text
p256 = 0.5 × (μ512 − μ384) / (μ512 − μ256)
p512 = 0.5 × (μ384 − μ256) / (μ512 − μ256)
```

从而：

```text
p256·μ256 + p384·μ384 + p512·μ512 = μ384
```

阻断条件：

* `μ512 == μ256`；
* actual-observation 成本不满足逐档单调；
* 任一概率不在 `[0,1]`；
* 概率曾读取类别、边界或检测指标；
* 训练后根据结果调整概率。

6,000 次更新中的离散预算数量按最大余数法一次性确定，再由上述哈希流排列。预运行时，多预算臂与控制臂的总 actual-observation 差异必须不超过 `0.5%`。只允许在训练开始前、无标签条件下重新排列同一预算 multiset；不得改变概率或 sample 集。

### 6.7 checkpoint 与模型选择

* 每 500 次成功更新保存可恢复 checkpoint；
* 强制保留 update 3,000；
* 强制保留 update 6,000；
* 至少保留最近三个有效恢复点；
* 恢复包包含 model、EMA、optimizer、scheduler、AMP scaler、successful-update index、sampler 和随机状态。

**唯一结果模型：update 6,000 的 `state_dict_ema`。**

禁止：

* 中间 held-out mAP；
* best checkpoint；
* early stopping；
* 根据某档预算表现补训；
* 事后切换 EMA 或非 EMA。

---

## 7. `FINAL_EVALUATION_AND_STATISTICS`

### 7.1 密封预测矩阵

两个 terminal EMA 都在完整 `E_full` 上产生：

* fixed control × K256
* fixed control × K384
* fixed control × K512
* multibudget adaptation × K256
* multibudget adaptation × K384
* multibudget adaptation × K512

此外，每臂产生一个预注册、无标签、相同预算 manifest 的 mixed-budget 输出。

所有固定预算预测先独立密封，mixed 输出只按 manifest 选择对应窗口的密封预测；不得重新前向、重新排序 proposal 或读取标签。

### 7.2 无标签 fixed mixed-budget manifest

该 manifest 不是 oracle，也不声称是动态 controller。

设完整 held-out 窗口集合为 `W`。在打开动作标签前：

1. 使用训练侧冻结的 `p256/p384/p512`；
2. 以最大余数法确定 `W` 中三档预算的精确窗口数量；
3. 余数相同时优先顺序固定为 `K384 → K256 → K512`；
4. 对每个窗口计算：

```text
SHA256(
  "DUCA-FULL-DATA-EVAL-MIX-v001|3407|"
  + video_id + "|"
  + window_index + "|"
  + sample_id
)
```

5. 按完整哈希值和窗口身份排序；
6. 前 `n256` 个窗口使用 K256，随后 `n384` 个使用 K384，其余使用 K512；
7. 两个模型使用完全相同的 manifest；
8. 读取 held-out 结果后不得重新排列。

报告实际 observation 成本。若 mixed manifest 的总实际成本高于全 K384，则成本条件失败，不允许根据结果重新生成 manifest。

该输出只回答：

> 在相同、无标签、等价预算混合负载下，多预算训练是否比固定 K384 训练建立了更好的跨预算兼容性？

它不回答语义 controller 是否有效。

### 7.3 指标

对六个固定预算输出和两个 mixed 输出统一报告：

* mAP@0.3、0.4、0.5、0.6、0.7；
* Avg-mAP；
* pre-NMS class-agnostic proposal Recall@100、Recall@200；
* proposal recall 的 tIoU 0.5、0.7；
* 同类别最佳 proposal 的起点和终点绝对误差；
* 物理秒误差均值和中位数；
* 相对 GT duration 归一化误差；
* 每视频 pre-NMS、post-NMS proposal 数；
* top-200 proposal 的 NMS 前后假阳性；
* 短、中、长动作的 Avg-mAP 和 mAP@0.7；
* actual observations；
* VideoMAE 实际输入量。

短、中、长动作阈值由完整训练集 GT duration 三分位数预先计算并冻结，不得使用 held-out duration 分布调节。

诊断指标只解释结果，不新增通过条件。

### 7.4 K384 安全门

多预算适应模型相对固定控制模型，二者均在 K384 推理：

```text
ΔAvg-mAP >= -0.2 个百分点
ΔmAP@0.7 >= -0.2 个百分点
```

两项必须同时满足。

### 7.5 跨预算适应继续门

在相同 fixed mixed-budget manifest 下：

```text
adaptation mixed − control mixed
```

必须同时满足：

```text
ΔAvg-mAP >= +0.8 个百分点
ΔmAP@0.7 >= +1.0 个百分点
mixed actual-observation cost <= 全 K384 actual-observation cost
```

保留 `+0.8/+1.0`，避免因数据协议修订而降低原有实际效应门槛；但估计量由有标签 oracle 增益改为更严格的、无标签固定负载下两臂直接差异。

### 7.6 配对不确定性

使用全部 `E_full` 视频执行：

* 10,000 次整视频配对 bootstrap；
* 每次对视频有放回重采样；
* 同一 replicate 对两个模型和全部模式使用相同视频样本；
* 每次从密封预测重新计算官方 mAP；
* bootstrap RNG 命名空间：

```text
DUCA-FULL-DATA-COMPARABLE-PAIRED-BOOTSTRAP-v001|3407
```

* 95% percentile interval；
* 延续项目最近秩第 250 / 9750 个值的约定。

至少报告：

* mixed `ΔAvg-mAP` 区间；
* mixed `ΔmAP@0.7` 区间；
* K384 两项安全差值区间。

该 bootstrap 只描述 held-out 视频采样不确定性，不包含训练随机种子不确定性。

### 7.7 结果分类与停止条件

#### 允许请求下一轮 controller 科学裁决

必须同时满足：

1. K384 两项安全门；
2. mixed 两项实际效应门；
3. mixed actual cost 不高于全 K384；
4. mixed `ΔAvg-mAP` 的 95% 区间下界大于 0；
5. mixed `ΔmAP@0.7` 的 95% 区间下界大于 0；
6. 无协议偏差、泄漏或 ID 缺失。

该结果支持的精确结论为：

> 条件于 H65 Stage-1 初始化、当前嵌套位置构造和单个训练 seed，在相同训练数据、监督、优化及固定无标签预算混合下，多预算训练暴露改善了跨预算检测兼容性，同时没有实质损害 K384 锚点。

它只允许 Pro 另行判断是否值得设计 controller；**不自动授权 controller 训练。**

#### 停止当前三档适应路线

以下任一项发生即停止：

* K384 安全门失败；
* mixed 任一实际效应门失败；
* mixed actual cost 高于全 K384；
* 多预算模型出现数据、数值或训练不公平；
* 某一预算只能通过候选臂专属 batch 或训练规则运行。

停止后：

* 不补训练；
* 不换 seed；
* 不改概率；
* 不降低门槛；
* 不添加 embedding、蒸馏、新 Scout/head/selector；
* 不以 official held-out oracle 抢救。

#### 点估计通过但区间包含零

判定为证据未决但不前进：

* 不训练 controller；
* 不在当前任务中补 seed；
* 不重新打开 held-out；
* 当前实验终结；
* 任何后继都需要新的 Pro 科学问题和新的确认协议。

#### 协议失败

split、标签泄漏、评价器变化、视频缺失、预测不完整或 proposal 顺序变化，均使证据无效。这不是模型负结果。

### 7.8 单种子和可发表性边界

本轮只冻结 seed `3407`。

因此，即使通过：

* 不作训练稳定性主张；
* 不作一般总体显著性主张；
* 不称 mixed manifest 为 learned controller；
* 不声称语义证据能够预测预算；
* 不声称取得动态预算性能—成本优势；
* 不声称优于 dense AdaTAD；
* 不声称跨数据集成立。

一个通过结果是决定是否研究 controller 的强机制证据，不是完整 DUCA 论文主张。项目现有正式状态也明确指出单种子结果不能支持稳定性或显著性结论。

本轮不要求另一外部确认集，也不授权跨数据集实验；官方 held-out 是当前 detector-adaptation 假说的一次性终态评价。

---

## 8. `CODEX_TASK_ORDER`

### 8.1 唯一当前任务

**只读完成 THUMOS14 完整 train/held-out 身份核验，解决 211/212 差异。**

这是当前唯一可执行任务。

在该任务 `PASS` 前，禁止：

* 建立多预算模型 Builder 分支；
* 修改模型；
* PRE_RUN；
* 加载 Stage-1 checkpoint；
* 提交 GPU；
* 开始训练；
* 生成 held-out 模型预测；
* 计算 held-out mAP；
* 读取 held-out temporal annotations。

科学选择已经由本裁决完成：正式协议使用 H65/OpenTAD 的 `training → validation` 语义。Codex 只核验精确事实，不选择 split。

### 8.2 Builder

建立唯一分支：

```text
feature/duca-full-data-identity-audit-v1-20260831
```

基于：

```text
04c35a3b76897e6c1569eeede41ed3aecaf7f854
```

只允许新增或修改：

```text
tools/bata/audit_duca_thumos14_split_identity.py
tests/test_audit_duca_thumos14_split_identity.py
```

若已有等价只读工具，可直接复用，不得另建框架。

#### 精确输入

1. `04c35a3b...` 中 H65 Stage-1 和 Stage-2 的精确 config；
2. config 继承链最终解析结果；
3. `thumos_14_anno.json`；
4. `category_idx.txt`；
5. video root 文件名；
6. `04c35a3b...` 的 dataset loader、filter 和 evaluator 源码；
7. PJST 211-video sidecar 或 prediction ID manifest；
8. 历史“ActionFormer 212 个带标注测试视频”的原始 annotation、manifest 或可核验来源。

#### 允许读取的字段

* video ID；
* subset 字面值；
* annotation 顶层 key；
* 媒体文件名；
* loader 输出 ID；
* 明确 exclusion 代码；
* 文件可读性和基本解码结果。

#### 禁止读取的内容

* held-out 动作类别；
* held-out temporal segment；
* held-out proposal；
* checkpoint；
* 模型预测；
* mAP；
* 每视频效用或 oracle 结果。

脚本即使加载 JSON，也不得遍历 held-out `annotations` 内容。

#### 必须输出

* 精确 Stage-1/Stage-2 config 路径；
* 最终解析的 train/eval subset；
* `annotation_training_ids.txt`
* `loader_training_ids.txt`
* `physical_training_ids.txt`
* `annotation_validation_ids.txt`
* `loader_validation_ids.txt`
* `physical_validation_ids.txt`
* `historical_211_prediction_ids.txt`
* `actionformer_212_source_ids.txt`，或明确的 `SOURCE_NOT_FOUND`
* 所有集合的数量；
* 所有成对 set difference；
* train/held-out 交集；
* 每项 loader 排除及对应 source line；
* annotation、class map、evaluator source 的普通 SHA-256；
* `full_train_video_ids.txt`
* `full_heldout_video_ids.txt`
* 一份简短 `split_identity_report.json`。

不建立通用数据治理、manifest framework 或新审计系统。

#### Builder 通过条件

训练侧：

* annotation `training` 恰好 200；
* loader 输出恰好同一 200；
* 物理视频恰好覆盖同一 200；
* 无排除；
* 全部可解码。

held-out：

* annotation、loader、物理文件和 evaluator 集合一致；
* 数量为 211 或 212；
* 211/212 差异有精确 source-backed 解释；
* 无静默丢弃；
* 与 training 交集为空。

否则返回客观 blocker，不自行修订 split。

### 8.3 独立 Critic

Critic 在独立上下文中只读审查 Builder 的唯一精确提交。

只审查：

* 是否实际从 `04c35a3b...` 出发；
* 是否只读取身份层字段；
* 是否访问 held-out 动作标签；
* config 继承是否解析完整；
* loader 实际过滤是否被捕获；
* 211/212 差集是否来自真实 ID 集；
* 是否把缺文件或解码失败误写成合法排除；
* 输出清单是否确定、排序且无交集；
* Builder 是否擅自选择 split。

输出只能是：

* `PASS`；或
* 一次有界 blocker 清单。

不因日志、代码风格或缺少通用框架阻塞。

### 8.4 独立 Evaluator

Critic `PASS` 后，Evaluator 在 N16R4 上运行一次只读身份核验：

* CPU 即可；
* 不申请 GPU；
* 不加载模型；
* 不加载 checkpoint；
* 不生成预测；
* 不计算 mAP。

Evaluator 返回：

* exact commit；
* clean-tree 状态；
  -命令；
* 输出根；
* literal manifests；
* counts；
* set differences；
* exclusion source；
* 文件可读性和解码结果；
* report SHA；
* 唯一结论：`PASS` 或 `BLOCK`。

### 8.5 当前任务的明确终点

* `PASS`：把 literal manifests 和 211/212 事实返回 Pro，等待数据身份准入；不得自动进入模型实现。
* `BLOCK`：保留当前多预算科学路线为“未执行”，返回缺失事实；不得自行换 annotation、补视频、删视频或选择 ActionFormer 口径。

Builder、Critic、Evaluator 的工作只解决一项会改变实验有效性的真实 blocker，符合论文优先和反过度工程原则。 

---

## 9. `NEXT_RETURN`

### 9.1 下一次立即返回：数据身份准入证据

Codex 必须返回：

#### 代码身份

* branch；
* exact commit；
* parent commit；
* clean-tree；
* 修改文件；
* focused test 结果；
* Critic 对 exact commit 的完整结论。

#### 配置身份

* H65 Stage-1 config 精确路径；
* H65 Stage-2 config 精确路径；
* config 继承链；
* 最终 train subset；
* 最终 evaluation subset；
* dataset loader 源码路径及相关 source line；
* evaluator 源码路径和 SHA。

#### 完整训练集

* 200 个 literal video IDs；
* annotation、loader、物理文件三集合；
* 三组集合差异；
* 解码失败清单；
* 排除清单，期望为空；
* `full_train_video_ids.txt` 全文或稳定路径。

#### 完整 held-out 集

* annotation 中全部 `validation` IDs；
* loader 输出 IDs；
* 物理视频 IDs；
* 历史 211-video prediction IDs；
* ActionFormer 212-video 来源 IDs，或来源未找到的客观证据；
* 全部 set differences；
* 最终 211 或 212 的事实解释；
* 若有官方 exclusion，完整 ID、规则和 source line；
* `full_heldout_video_ids.txt` 全文或稳定路径。

#### 隔离与完整性

* train/held-out 交集；
* annotation/class map SHA；
* video coverage；
* no-label-access 测试；
* Evaluator 命令、输出和唯一 `PASS/BLOCK`。

在这些证据返回前，不得请求模型路线变化，也不得开始训练。

### 9.2 数据身份通过后，正式终态返回必须最终包含

本节预先冻结后续证据要求，但不构成当前执行授权。

#### 实现

* 多预算适应分支、exact commit 和 parent；
* 从 `33e4ed...` 移植的精确符号；
* `dynamic_budget.py` 三档语义未改变；
* 两份正式完整训练配置；
* K384 parity；
* real variable-length forward；
* packet 对齐；
* proposal 原始顺序；
* Critic 对正式模型提交的结论。

#### 完整训练

* 两臂 full-train manifests；
* Stage-1 checkpoint 路径、epoch、state key 和 SHA；
* 两臂初始参数逐张量一致性；
* optimizer parameter groups；
* trainable 参数名；
* scheduler、AMP、EMA、gradient clipping；
* `μ256/μ384/μ512`；
* `p256/p384/p512`；
* 训练预算 occurrence manifest；
* 两臂 actual-observation 总量；
* 两臂 6,000 次成功 update 证明；
* update-6,000 checkpoint 路径和 SHA；
* terminal `state_dict_ema` 存在性；
* 200 个训练视频实际 occurrence 覆盖。

#### 完整 held-out 预测

* 两臂 × 三档预算的预测；
* 两个 mixed 输出；
* 每份预测的 ID manifest、数量和 SHA；
* fixed mixed-budget manifest；
* producer proposal 顺序证明；
* 所有预测在标签开放前密封的时间与证据；
* actual-observation 成本。

#### 完整评测与统计

* 五个 tIoU mAP；
* Avg-mAP；
* proposal recall；
* 边界误差；
* proposal 数；
* NMS 前后假阳性；
* 动作长度分层；
* 10,000 次 paired whole-video bootstrap；
* K384 安全差值及区间；
* mixed 两项差值及区间；
* mixed 与全 K384 actual cost；
* 所有协议偏差；
* 根据本裁决得到的唯一自然语言结论：

  * 允许请求 controller 科学裁决；
  * 停止当前三档适应路线；
  * 单种子证据未决且不前进；
  * 协议失败、证据无效。

在完整实现、完整训练和完整官方 held-out 证据返回前：

* 不得训练 controller；
* 不得添加预算 embedding；
* 不得添加蒸馏、Gumbel、新 Scout/head/selector、Mamba 或 Block Drop；
* 不得打开 held-out 进行探索；
* 不得把代码完成、审查通过、训练完成或单个点估计写成方法有效性证据。

以上裁决完整保留 nonce：

`DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`

建议保存文件名：`PRO_DUCA_FULL_DATA_COMPARABLE_PROTOCOL-v001.md`。
