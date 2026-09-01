以下为可直接保存并交付 Codex 的冻结版科学裁决。

# DUCA 多预算检测器适应：科学裁决与执行冻结

## 1. `SESSION_ASSERTION`

**Nonce：`DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`**

* ChatGPT Project ID：`g-p-6a91061f789881918ccd8357ca3d6c92`
* 仓库：`yuzbo/OpenTAD_C3_CoarseClean_20260702`
* 新实验的唯一模型与代码基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
* 只读 whole-video 诊断提交：`33e4ed137c33eef07f0452b44506a6993bdf7535`

`04c35a3b...` 是 H65 训练身份。它的 Stage‑1 合同为 30 个训练轮次、每轮 100 次成功更新、终态 `epoch_29/state_dict_ema`；Stage‑2 合同明确使用一个重新开始的 6,000 次成功更新日程，并从 Stage‑1 EMA 初始化。

`33e4ed...` 不是新模型基座。该提交只修改 whole-video runner 和聚焦测试，把密封 proposal 的重放顺序恢复为 producer 原始顺序，避免 Soft‑NMS 分数并列时因额外排序产生确定性漂移；它没有修改模型、预测值、预算候选、损失、NMS 或评价器。

本轮材料权重冻结如下：

1. 用户本轮提供的 2026‑08‑31 prompt、`PAPER_PROGRESS.md`、两个实验页和来源记录是当前科学状态；
2. 2026‑08‑28 材料包已完整读取，只用于代码库存边界、角色约束、原始检查点证据和历史谱系，不覆盖本轮更新后的停止边界与科学问题。库存提交不是实验身份，Pro 负责科学选择，Builder、Critic、Evaluator 只执行冻结任务。  
3. PJST 原始评估材料用于确认 H65 Stage‑1 检查点路径、epoch、SHA‑256 和状态键；旧项目 prompt、README、旧研究状态和历史版本注册表仅作历史背景。     

---

## 2. `SCIENTIFIC_DECISION`

**唯一裁决：`CONTINUE`**

批准执行一次有界、单变量、完整匹配的多预算训练实验。

### 2.1 为什么值得执行

此前 704 个 whole-video 候选的终态负结果只检验了：

> 一个只在固定 `K=384` 协议下训练并冻结的模型，在密封的 `K256/K384/K512` 预测之间重新分配预算时，是否已经具有足够的开发集联合性能空间。

该实验没有重新训练 detector，没有执行梯度更新，也没有让重型表示在训练阶段见到 `K256` 或 `K512`。因此，“训练输入支持不足导致跨预算不兼容”仍是一个尚未接受直接检验、但可以通过单次匹配实验裁决的机制假说。

本轮批准不表示该假说已经成立，也不表示为了延续项目而放宽门槛。批准的理由只有一个：它改变了旧实验从未改变过的变量——**训练时的预算分布**——并能以两条完整匹配训练臂直接产生判别性证据。

### 2.2 旧 `STOP` 完整保留

以下旧路线继续永久只读：

* Marginal‑v1；
* cap‑release；
* 96-state 联合邻域；
* whole-video 704-state 分支；
* 旧 40-video controller holdout 上的密封预测与 oracle 结果。

不得：

* 重跑或扩大旧候选空间；
* 修改旧门槛；
* 加入第三个视频或组合多个 transfer；
* 训练旧 controller 或 utility head；
* 在旧结果中事后选择候选再补 bootstrap；
* 使用旧 40 个视频作为本轮“未参与规则选择”的开发集；
* 访问 official test；
* 把新结果写成旧 Marginal 路线的恢复。

新实验从 `04c35a3b...` 建立独立分支。旧分支唯一可移植的是已经验证的真实变长执行、packet 对齐、实际 observation 计数、K384 parity、whole-video 评价和原始 proposal 顺序保持。

---

## 3. `CAUSAL_ISOLATION`

### 3.1 冻结的唯一干预变量

第一轮继续使用当前**嵌套**的 `K256/K384/K512` 位置构造。

* `K256`、`K384`、`K512` 必须来自同一 H65 priority sequence；
* 不改成三个预算分别重新运行的“预算原生 H65 选点”；
* 不修改 Scout 输入、评分字段、优先级调制、覆盖下限、确定性累计采样或短窗口折叠；
* 不修改物理时间逆映射、VideoMAE 语义、Temporal Adapter、ActionFormer/AdaTAD 检测头、损失、Soft‑NMS 或评价器。

**唯一干预变量是：训练时送入相同 H65 模型的预算分布。**

两臂为：

1. **固定预算控制臂**

   * 每个训练窗口请求 `K384`；
   * 短窗口实际成本为 `min(valid_observations,384)`。

2. **多预算适应臂**

   * 每个训练窗口请求 `K256`、`K384` 或 `K512`；
   * 使用下文冻结的、按实际 observation 成本校准的概率；
   * 短窗口继续按 `min(valid_observations,K)` 折叠。

### 3.2 不进入第一轮的变量

明确禁止：

* 预算条件嵌入；
* 显式输入 `K`、相邻物理时间间隔或局部采样密度；
* 蒸馏；
* Gumbel‑Softmax；
* 新 Scout、新分类头或边界头；
* 新 selector 或预算原生位置构造；
* DFT、Mamba、Block Drop；
* CUDA、TensorRT 或其他部署优化；
* 新数据集；
* controller 训练；
* 根据开发结果改变预算概率。

### 3.3 可归因结论的精确边界

两臂使用完全相同的 H65 Stage‑2 可训练参数集合。不得只在候选臂冻结或解冻某些参数。

因此，正结果支持的是：

> 在相同 H65 架构、初始化、监督、优化和嵌套位置构造下，训练时接触多种预算输入可以建立更好的跨预算兼容性。

它不能被写成：

* “ActionFormer 检测头单独是根因”；
* “Scout 完全没有参与适应”；
* “不可微选点已被解决”；
* “动态预算一般问题已经成立”。

---

## 4. `TRAINING_FREEZE`

### 4.1 训练定位

选择：

> **从冻结的 H65 Stage‑1 终态进行两臂匹配的完整 Stage‑2 训练。**

拒绝从 H65 terminal detector 进行短期继续训练。

原因是：短期继续训练从一个已经被 `K384` 输入分布塑形的优化盆地出发。若候选失败，无法区分“多预算训练无效”和“学习率太小、更新不足或无法离开既有盆地”；若候选成功，也只能说明事后微调可部分修复，不能形成干净的完整训练比较。

### 4.2 唯一起点

两臂共同加载：

* 路径：
  `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`
* 期望 epoch：`29`
* 状态键：`state_dict_ema`
* SHA‑256：
  `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`

该检查点身份在已有终态评估材料中已绑定；H65 Stage‑1 配置也明确指定 terminal `epoch_29/state_dict_ema`。

加载模型权重后，两臂都必须重新初始化：

* optimizer state；
* learning-rate scheduler state；
* AMP scaler；
* EMA accumulator；
* successful-update counter；
* `frame_selector._loss_weight_schedule_step`；
* 数据采样与随机增强随机状态。

不得从 H65 Stage‑2 terminal optimizer、scheduler 或 EMA 状态恢复。

### 4.3 训练长度

每臂：

* **恰好 6,000 次成功 optimizer update**；
* 非有限损失或 AMP 重放不计为成功更新；
* 两臂成功更新数必须完全相等；
* 以成功更新索引作为训练主时钟。

由于新 Stage‑2 训练集为 160 个视频，不能把“60 个实际数据轮次”静默当作与原 H65 相同的训练量。训练器应循环数据直到完成 6,000 次成功更新。

为保持 H65 原日程：

* 每 100 次成功更新定义一个 H65 **schedule epoch**；
* 总计 60 个 schedule epoch；
* warmup 5 个 schedule epoch，即前 500 次成功更新；
* Stage‑2 前后半程边界保持在第 3,000 次成功更新；
* 所有原来按 step 定义的 loss、policy 和 detector-gradient 日程保持原值。

H65 Stage‑2 原配置明确冻结 `max_updates=6000`、`duca_stage2_half_steps=3000`，并从 `state_dict_ema` 初始化后重置训练状态。

### 4.4 优化器、调度器与数值合同

两臂逐项相同：

* optimizer：`AdamW`
* 主学习率：`1e-4`
* weight decay：`0.05`
* VideoMAE 主体保持 H65 参数分组；基础 backbone 的冻结规则不变，adapter 学习率为 `2e-4`
* H65 Stage‑2 Scout 相关学习率：

  * coarse trunk：`1e-5`
  * action head：`2e-5`
  * transition scorer：`5e-5`
* scheduler：`LinearWarmupCosineAnnealingLR`
* warmup：前 500 次成功更新
* 总日程：6,000 次成功更新
* gradient clipping：global norm `1`
* AMP：开启
* EMA：开启
* batch size：每卡 `2`
* seed：`3407`
* DDP、unused-parameter、损失归一化和梯度累积规则不得因预算档位而改变。

这些数值来自 `04c35a3b...` 所继承的 H65/AdaTAD 配置；精确配置优先于任何文字摘要。

若 `K512` 在 batch size 2 下发生显存不足，只允许对**两臂同时**采用 microbatch 1、累计 2 个样本后完成一次 optimizer update。不得只降低候选臂 batch size，也不得改变总成功更新数。

### 4.5 随机性匹配

两臂必须共享：

* 视频与窗口采样次序；
* random truncation；
* resize/crop/flip/color augmentation；
* 模型初始化；
* dropout/drop-path 随机种子；
* optimizer update 次序。

候选臂的预算随机性使用独立命名空间，不得消耗数据增强或模型随机流：

`SHA256("DUCA-MBDA-BUDGET-v001|3407|<successful_update>|<sample_id>|<occurrence>")`

固定预算臂忽略该值并始终使用 `K384`。

### 4.6 多预算概率与训练成本校准

先在**160-video Stage‑2 训练集合**上重放完整的无标签训练窗口计划，按短窗口折叠计算：

* `μ256`：请求 K256 时的平均实际 observation；
* `μ384`：请求 K384 时的平均实际 observation；
* `μ512`：请求 K512 时的平均实际 observation。

不得读取动作类别、边界或检测指标。

保持中央预算概率固定为：

`p384 = 0.50`

保持两侧总暴露为：

`p256 + p512 = 0.50`

并唯一求解：

`p256 = 0.5 × (μ512 − μ384) / (μ512 − μ256)`

`p512 = 0.5 × (μ384 − μ256) / (μ512 − μ256)`

这样候选臂的期望实际 observation 成本等于固定 K384：

`p256·μ256 + p384·μ384 + p512·μ512 = μ384`

当 `μ384` 正好位于两端均值中点时，概率自然恢复为 `0.25/0.50/0.25`。

阻断条件：

* `μ512 == μ256`；
* 任一概率不在 `[0,1]`；
* 实际成本不是逐档单调；
* 概率是在读取开发标签或模型指标后调整。

6,000 次更新中的离散预算数量采用最大余数法从上述概率一次性确定，再由冻结哈希打乱。候选臂最终实际训练 observation 总量与控制臂的偏差必须报告；预运行时若超过 `0.5%`，只允许在不改变概率公式和不读取标签的情况下重新排列预算与样本 occurrence，使差值最小。仍超过 `0.5%` 时返回阻断，不开始正式训练。

### 4.7 checkpoint 与中间验证

保存点：

* 每 500 次成功更新保存可恢复 checkpoint；
* 强制保留 update 3,000；
* 保留最近三个有效 checkpoint；
* 强制保留 update 6,000 terminal checkpoint。

恢复包必须包含：

* model；
* `state_dict_ema`；
* optimizer；
* scheduler；
* AMP scaler；
* successful-update index；
* 数据 sampler 和全部随机状态。

**唯一结果模型：update 6,000 的 `state_dict_ema`。**

不得：

* 按中间 Avg-mAP 选择 checkpoint；
* early stopping；
* 事后选择 best validation EMA；
* 因某一预算曲线较好而修改概率或损失。

新的 40-video 开发集在两臂 terminal checkpoint 和配置全部密封前不得进行任何有标签评价。训练中可以查看损失、梯度、数值有限性和无标签实际成本，但不能查看开发 mAP。

---

## 5. `DEVELOPMENT_SPLIT_FREEZE`

### 5.1 为什么不能继续使用旧 40 个视频

旧 40-video controller holdout 已经参与：

* capped oracle；
* released oracle；
* 96-state 联合邻域；
* 704-state whole-video 枚举；
* 门槛与路线裁决。

因此它不再是未参与规则选择的开发集。

本轮将这 40 个视频明确归入训练可用集合，不再承担评价功能。

### 5.2 可复现划分规则

输入：

1. THUMOS14 规范训练侧的 200 个视频 ID；
2. 旧 whole-video 实验实际读取的 40 个 `holdout_videos` ID；
3. split seed：`20260831`。

步骤：

1. 从 annotation 中读取规范训练侧视频 ID，去重后按完整视频 ID 字典序排序；

2. 要求数量严格为 `200`；

3. 读取旧 40-video manifest，要求数量严格为 `40`，且全部属于上述 200；

4. 从 200 中移除旧 40，得到 160 个新的开发候选；

5. 对每个候选视频 ID 计算：

   `SHA256("DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-DEV-v001|20260831|" + video_id)`

6. 按 `(完整 SHA‑256 十六进制值, video_id)` 升序排序；

7. 前 `40` 个视频作为新开发集；

8. 其余 `120` 个新候选与旧 `40` 个历史 holdout 合并，形成 `160` 个 Stage‑2 训练视频。

最终数量：

* Stage‑2 train：`160`
* Stage‑2 development：`40`
* official evaluation/test：`0` 个被访问

必须落盘：

* `train_video_ids.txt`
* `development_video_ids.txt`
* `historical_40_excluded_from_dev.txt`
* 一个简短 JSON，记录 seed、生成字符串、数量和来源 annotation。

不要求新的哈希治理框架；普通文本清单和一个 JSON 足够。

### 5.3 使用边界

160-video train 可用于：

* 两臂参数学习；
* 多预算概率校准；
* 实际训练成本估计；
* 动作时长分层阈值；
* 数值与运行检查。

40-video development 仅用于：

* 两臂 terminal EMA 的三档预算评价；
* 预登记的诊断指标；
* 预登记的等成本 whole-video oracle；
* 本轮最终科学门槛。

development 不得用于：

* 超参数、概率、损失权重或训练长度选择；
* checkpoint 选择；
* NMS 或分数阈值选择；
* 修改 K 档位；
* 选择不同的 whole-video 候选定义；
* 决定是否补训某一臂。

### 5.4 证据独立性的诚实边界

该 40-video development 不参与**本轮 Stage‑2 参数更新或本轮规则选择**。

但是，现有 H65 Stage‑1 checkpoint 并不是针对该新划分重新训练和封存的。现有材料不能证明新开发视频从未影响过历史初始化。因此，本划分是：

> 条件于冻结 H65 Stage‑1 初始化的独立 Stage‑2 机制开发集。

它不是完全独立于所有历史训练的外部确认集。

这是在不重训 Scout、不改变另一项科学变量、又不访问 official test 的前提下可接受的最小证据设计。即使实验通过，也只能进入下一轮确认性实验裁决，不能直接形成最终泛化或显著性主张。

若旧 40-video manifest 无法精确恢复、200 个训练 ID 数量不符、train/dev 发生交集，Evaluator 必须返回阻断，不得自行换 seed、改变视频数量或重新切分。

---

## 6. `EVALUATION_AND_GATES`

### 6.1 实验准入检查

正式训练前必须通过：

1. 基座精确为 `04c35a3b...`；
2. `dynamic_budget.py` 的三档预算语义未修改；
3. Stage‑1 checkpoint 路径、SHA‑256、epoch 和状态键完全匹配；
4. 新 train/dev ID 无交集；
5. 两臂初始化参数逐张量相同；
6. 两臂 optimizer 参数组、学习率、trainable flags 和 EMA 配置相同；
7. 固定控制的 K384 前向与基座 K384 在相同 checkpoint、相同输入和相同 RNG 下：

   * selected positions bit-exact；
   * detector tensor 输出在现有数值容差内一致；
   * proposal 原始生成顺序一致；
   * actual observation 数一致；
8. `K256/K512` 真正执行变长 VideoMAE，不得在重型骨干前补齐为 K384 或 K512；
9. mixed-budget packet 分组后，输出重新对齐到原 batch/sample 顺序；
10. 候选训练期预期成本满足上文校准合同。

任何一项失败都属于实现或协议问题，不是模型负结果。

### 6.2 三档终态指标

两臂 terminal EMA 都必须分别在 `K256/K384/K512` 上产生密封预测并报告：

* mAP@0.3、0.4、0.5、0.6、0.7；
* Avg-mAP；
* pre-NMS class-agnostic proposal recall：

  * Recall@100、Recall@200；
  * tIoU 0.5 和 0.7；
* 每个 GT 与同类别最高 tIoU pre-NMS proposal 匹配后的：

  * 起点绝对误差；
  * 终点绝对误差；
  * 物理秒误差的均值和中位数；
  * 相对 GT duration 归一化误差；
* 每视频 pre-NMS、post-NMS proposal 数的均值和中位数；
* top‑200 proposal 中的假阳性：

  * 若不存在同类别、tIoU≥0.3 的 GT 匹配则记为假阳性；
  * 分别报告 NMS 前后；
* 短、中、长动作的 Avg-mAP 与 mAP@0.7。

短、中、长阈值只能由 160-video train 的 GT duration 三分位数预先确定，不得用 development duration 调整。

上述指标为解释性诊断，不新增通过门槛。

### 6.3 K384 安全门

在新 40-video development 上，比较多预算模型与同更新数固定控制，二者都以 K384 推理：

* `ΔAvg-mAP >= -0.2` 个百分点；
* `ΔmAP@0.7 >= -0.2` 个百分点。

两项必须同时满足。

该门防止多预算模型通过降低自身 K384 锚点来人为放大后续 oracle 增益。

### 6.4 等成本 whole-video oracle

对两臂分别使用各自密封的 K256/K384/K512 terminal predictions。

候选空间保持旧 whole-video 定义，但使用新的 40-video development：

1. 全 K384 为该模型自己的固定基线；
2. 对每个不同视频的有序对 `(donor,recipient)`：

   * donor 的全部窗口请求 K256；
   * recipient 的全部窗口请求 K512；
   * 其余视频全部请求 K384；
3. donor 和 recipient 都必须至少有一个窗口的实际成本不同于 K384；
4. 总实际 observation 成本不得超过全 K384；
5. 最多生成 `40×39=1560` 个有序状态；
6. 候选清单只依赖 video ID、sample ID 和 actual observation 成本，必须在读取标签和指标前生成；
7. 两臂使用完全相同的候选清单；
8. proposal 合并必须保持 producer 原始生成顺序。

`33e4ed...` 的必要价值仅在于已经证明额外重排 proposal 会破坏 Soft‑NMS 的确定性重放，因此新 Evaluator 必须保留原顺序。

每个模型的最优候选先最大化：

`min(ΔAvg-mAP − 0.8, ΔmAP@0.7 − 1.0)`

再选择实际成本更低者，最后按 donor、recipient ID 字典序。

### 6.5 Oracle 继续门

多预算模型相对其自身全 K384 基线必须同时满足：

* `ΔAvg-mAP >= +0.8` 个百分点；
* `ΔmAP@0.7 >= +1.0` 个百分点；
* actual observation cost `<=` 其全 K384 成本。

同时必须通过 K384 安全门。

固定控制也运行相同 oracle，但只作为因果对照，不改变预登记阈值。

结果解释冻结为：

1. **多预算通过、固定控制不通过**

   * 支持“训练时多预算暴露建立了新的跨预算联合 headroom”；
   * 返回 Pro 决定是否进入可预测 budget controller 与确认性实验；
   * 不自动训练 controller，不访问 official test。

2. **多预算与固定控制都通过**

   * 说明新 development 本身存在 whole-video headroom；
   * 不能把通过归因于多预算训练；
   * 结果为因果未决，返回 Pro，不自动继续。

3. **多预算未通过**

   * 无论固定控制结果如何，当前多预算检测器适应假说未达到预登记目标；
   * 停止当前 H65 嵌套 `K256/K384/K512` detector-adaptation 路线；
   * 不补训练、不换 seed、不改概率、不降低门槛。

4. **K384 安全门未通过**

   * 即使 oracle 通过，也判定本路线失败；
   * 不接受以损害标准预算性能换取 oracle 增益。

5. **Avg-mAP 达门而 mAP@0.7 未达门**

   * 当前路线仍停止；
   * 只允许下一轮由 Pro 重新判断是否值得单独检验 K、相邻物理间隔或局部采样密度条件；
   * 本轮不得预埋这些输入。

### 6.6 统计与论文证据边界

本轮使用一个 seed `3407` 和一个条件于历史 Stage‑1 初始化的训练侧 development。

因此：

* 不作总体显著性主张；
* 不把 oracle 最优候选称为可部署策略；
* 不把通过结果称为 official validation/test 结果；
* 不补事后 bootstrap 来放大结论；
* 不声称优于 dense AdaTAD；
* 不声称已经获得端到端性能—成本优势。

通过结果最多是强机制开发证据；失败结果可以在冻结范围内停止本三档适应路线，但不能外推否定所有动态计算或预算条件模型。

---

## 7. `CODEX_TASK_ORDER`

### 7.1 Builder

建立唯一分支：

`feature/duca-multi-budget-detector-adaptation-v1-20260831`

基于：

`04c35a3b76897e6c1569eeede41ed3aecaf7f854`

#### Builder 允许完成的最小实现

1. 新增两份 matched Stage‑2 配置：

   * fixed K384 control；
   * calibrated K256/K384/K512 adaptation。
2. 新增一个确定性 train/dev ID 生成入口；
3. 新增训练侧实际 observation 概率校准；
4. 移植并接入：

   * 真实变长 VideoMAE packet 执行；
   * packet 输出回填；
   * actual observation 计数；
   * K384 parity；
   * whole-video terminal evaluator；
   * producer 原始 proposal 顺序保持；
5. 新增一个最小 Slurm launcher；
6. 新增聚焦测试。

#### Builder 禁止修改

* H65 Scout 结构或输入；
* H65 priority、selector、位置嵌套关系；
* `dynamic_budget.py` 的三档语义；
* detector head；
* 分类或回归损失；
* 物理时间逆映射；
* Soft‑NMS；
* annotation、类别映射或评价器；
* official test 入口；
* 任何旧 Marginal、cap-release、96-state 或 whole-video 产物；
* 新 controller、部署优化或额外模型组件。

#### Builder 必须证明

* split 数量与无交集；
* 旧 40 个视频不进入新 development；
* 两臂初始 state_dict 一致；
* optimizer/trainable/EMA 配置一致；
* K384 bit-exact parity；
* 三档实际 observation 单调；
* 短窗口正确折叠；
* mixed packet 对齐；
* proposal 顺序不被排序；
* 6,000 次成功更新是唯一停止时钟；
* development 标签在 terminal 密封前不可访问。

#### Builder 失败返回

遇到以下任一情况立即返回客观阻断，不自行设计替代协议：

* Stage‑1 checkpoint 或 SHA 不匹配；
* 旧 40-video ID 清单无法恢复；
* 规范训练视频不是 200 个；
* K384 无法复现；
* 变长执行必须修改 selector、loss、NMS 或 evaluator；
* 成本校准无法满足合同；
* K512 只能通过候选臂专属 batch size 才能运行。

Builder 不提交正式训练。

### 7.2 独立 Critic

Critic 在独立上下文中只读审查 Builder 的唯一精确提交。

审查范围仅限：

* 科学变量是否仍只有训练预算分布；
* 位置集合是否仍嵌套；
* Scout、selector、detector、loss、NMS、评价器有无隐性变化；
* train/dev 泄漏；
* checkpoint 与状态重置；
* 两臂 trainable 参数、优化器和更新数公平性；
* variable-length forward 是否真实；
* K384 parity；
* packet 和 proposal 顺序；
* 训练成本口径；
* development 是否可能在 terminal 前参与选择。

Critic 输出只能是：

* `PASS`；或
* 一次有界 blocker 清单。

不得因代码风格、日志美化、通用完备性、缺少工作流框架或假设性边角条件制造修复循环。角色边界应继续服从论文优先、最短科学闭环原则。 

### 7.3 独立 Evaluator

Critic `PASS` 后，Evaluator 依次执行：

1. 真实数据 PRE_RUN：

   * checkpoint 身份；
   * split；
   * K384 parity；
   * 每档至少两个真实 batch；
   * 梯度、AMP、EMA、actual cost；
   * 不计算 development mAP。
2. 两臂正式 6,000-update 训练：

   * 相同 seed；
   * 相同硬件类别；
   * 相同样本与增强流；
   * 独立输出根；
   * 不访问 official test。
3. 两臂 terminal `state_dict_ema` 的六次推理：

   * control × K256/K384/K512；
   * adaptation × K256/K384/K512。
4. 密封三档预测后才开放 development 标签；
5. 运行预登记诊断、K384 安全门和两臂 whole-video oracle；
6. 生成一次终态结果，不根据结果回改训练或规则。

确定性的启动器、路径、节点或存储故障只允许一次不改变模型、数据、split、门槛和结果选择的最小恢复。它们必须与科学结果分开报告。

---

## 8. `NEXT_RETURN`

Codex 下一次必须返回以下证据，缺一项不得要求新的科学路线裁决。

### 8.1 实现与身份

* branch；
* exact commit；
* parent commit；
* clean-tree 状态；
* 修改文件清单；
* 从 `33e4ed...` 移植的具体符号；
* `dynamic_budget.py` 未修改证明；
* 两份最终配置全文或稳定链接。

### 8.2 数据划分

* 规范 200-video ID 清单；
* 旧 40-video 清单；
* 新 40-video development 清单；
* 新 160-video train 清单；
* seed 和完整 SHA‑256 生成字符串；
* train/dev 交集为空的测试结果。

### 8.3 起点与训练合同

* Stage‑1 checkpoint 实际路径；
* SHA‑256；
* epoch；
* state key；
* 两臂初始化逐张量一致性结果；
* optimizer 参数组；
* trainable 参数名集合；
* scheduler；
* AMP、EMA、gradient clipping；
* 预算概率 `p256/p384/p512`；
* `μ256/μ384/μ512`；
* 预期与实际训练 observation 成本；
* 6,000 次成功更新审计。

### 8.4 Builder 与 Critic 证据

* focused test 结果；
* K384 parity 数值；
* 三档真实 batch 的 selected count 与 packet shape；
* Critic 对 exact commit 的完整终态结论；
* 所有未关闭 blocker。

### 8.5 正式作业与终态模型

* PRE_RUN Job ID、状态和输出根；
* 两臂训练 Job ID、状态和输出根；
* 基础设施恢复记录；
* update‑6,000 checkpoint 路径与 SHA‑256；
* `state_dict_ema` 存在性；
* 两臂实际成功更新数；
* 两臂最终 observation 总量。

### 8.6 结果

对两臂、三档预算逐项返回：

* 五个 tIoU mAP；
* Avg-mAP；
* proposal recall；
* 起终点误差；
* proposal 数；
* NMS 前后假阳性；
* 短、中、长动作结果；
* 原始预测和结构化指标路径。

### 8.7 Oracle 与门槛

* 无标签候选 manifest；
* 合法候选数；
* 固定 K384 成本；
* 每臂最优 Avg-mAP 候选；
* 每臂最优 mAP@0.7 候选；
* 每臂最优联合门余量候选；
* K384 安全门两项差值；
* 多预算 oracle 两项差值和成本；
* fixed-control oracle 结果；
* 根据本裁决规则得到的唯一分类：

  * 支持适应；
  * 因果未决；
  * 停止当前路线；
  * 证据无效。

在这些实现与结果证据返回前，不得训练 controller，不得访问 official test，不得另行提出预算条件嵌入或新 selector，也不得把代码完成、测试通过或训练作业成功当作模型有效性证据。

以上即为本轮冻结版科学裁决，可直接作为 Builder、Critic 与 Evaluator 的唯一执行依据。
