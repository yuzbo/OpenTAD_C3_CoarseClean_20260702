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
