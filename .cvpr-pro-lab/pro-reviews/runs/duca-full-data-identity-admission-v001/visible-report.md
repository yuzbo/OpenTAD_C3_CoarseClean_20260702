CONTINUE

**Nonce：`DUCA-FULL-DATA-IDENTITY-ADMISSION-v001-20260831`**

## 一、唯一裁决

我作出三项绑定裁决：

1. **正式准入完整数据身份。** DUCA 的正式数据边界冻结为：完整 **200-video `training`** 用于训练；完整 **211-video OpenTAD `validation`** 作为 held-out evaluation。ActionFormer 字面 **212-video `Test`** 只记录为上游来源差异，不与 OpenTAD 211 静默合并。
2. **正式解锁 H65 系统级多预算暴露适应实验。** 它不是纯 detector-only 适应，而是复用 H65 第二阶段既有的 Scout/coarse、action head、transition scorer/selector 相关路径和受控 detector feedback。数据准入只使实验具备执行资格，不提供任何模型有效性证据。
3. **种子顺序采用完整 `3407 → 3408 → 3409`，三枚种子全部完成并封存预测后，才一次性读取 211-video held-out 指标。** 我否决“先读取 3407，再根据结果决定是否运行 3408/3409”的条件顺序。

这仍然是一个当前任务，而不是三个任务：三种子是同一冻结实验的必要重复。

---

## 二、数据身份证据：准入

### 2.1 代码和审计身份成立

审计提交 `fdd2bcdddf3f23f3546244adf90c4427ed022837` 是 H65 干净基座 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` 的直接子提交；其科学表面没有模型、训练配置或评估器变化，只增加数据身份审计工具及对应聚焦测试。因此它可以作为数据证据提交，但不能替代 H65 模型基座。

完整机器报告确认：

* 训练 annotation、正式 loader replay、canonical physical media 均为相同的 **200 个视频**，共同 manifest SHA-256 为
  `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0`。
* OpenTAD held-out annotation、loader replay、physical media、evaluator 集合及历史正式 prediction-key 集合均为相同的 **211 个视频**，共同 manifest SHA-256 为
  `5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e`。
* 训练与 held-out 交集为空；没有缺失、重复、未分配 ID 或错误链接。
* 预期的 411 个 canonical 视频均通过基本 `ffprobe` 解码检查。

### 2.2 211 与 212 的差异已得到唯一解释

ActionFormer 字面 `Test` annotation 的 212 个 ID 相对 OpenTAD 211 只有一个额外 ID：

* `video_test_0000270`：ActionFormer 有、OpenTAD 无；OpenTAD 源码将其排除，因为标注错误。
* `video_test_0001292`：不是 ActionFormer 212 annotation 的成员。它只是额外存在的物理或特征文件，不能被提升为评估视频。

因此不存在两个不同的“合法 held-out 集合”可供结果后选择。正式评估必须固定为 OpenTAD 211，ActionFormer 212 只作为来源追踪事实。

### 2.3 首次大小写错误不否定有效审计

首次 CPU 调用把 ActionFormer 区分大小写的 `Test` 写成 `test`，导致空集合并被调用层 fail-closed 阻断。这不是被静默覆盖的结果：

* 原始失败输出被保留；
* 只修正了命令参数的大小写；
* 代码、数据和来源均未变化；
* 有效 `result_v2` 报告 SHA-256 为
  `d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`；
* 终态为 `DATA_IDENTITY_PASS_211`，且 `blockers=[]`。

该事件反而验证了审计在集合为空或来源差异无法解释时会阻断，而不是自动“修成通过”。聚焦测试还覆盖了重复 ID、无法解释的 211/212 差异、checkpoint 禁止加载、预测 payload 禁止读取等关键失效情形。

### 2.4 无 held-out 语义泄漏

本审计没有读取 held-out 动作类别或时间边界，没有加载模型或 checkpoint，没有使用 GPU，没有生成预测或计算 mAP；历史预测文件只读取顶层视频 ID 键。由此可以准入的是**数据集合身份与隔离边界**，而不是 selector 质量、检测性能或成本收益。

---

## 三、正式冻结的 200/211 数据边界

### 3.1 200-video `training` 可以用于

完整 200 个训练视频可用于：

* 两个 matched arms 的全部梯度更新；
* Scout/coarse、action head、transition scorer/selector 路径及 H65 原有 detector-feedback 路径的训练；
* 训练损失和数值稳定性监控；
* 不涉及 held-out 标签的执行正确性检查；
* 预先冻结的预算概率、更新日程和终态 checkpoint 规则。

因为本实验不保留训练侧验证子集，所以任何模型选择都必须由预先登记规则完成，不能事后从训练过程或 held-out mAP 中挑选 checkpoint。

### 3.2 211-video OpenTAD `validation` 禁止用于

在全部训练和预测封存完成前，211-video held-out 不得用于：

* 训练、反向传播或教师监督；
* selector、预算控制器、采样规则或任何阈值拟合；
* budget mixture、NMS、置信度阈值或后处理规则选择；
* checkpoint、epoch、EMA/non-EMA、模型臂或种子选择；
* 根据初步 mAP 决定是否修改代码、重训、追加 seed 或停止某个 seed；
* 按视频观察结果后修改预算分配；
* 使用隐藏 raw-prediction cache 反复试验。

项目规则本身也要求 validation/test GT 不得参与选择，并明确区分代码存在、运行成功、点估计和统计结论。

### 3.3 “一次性 held-out 读取”的精确定义

本轮将“一次性”冻结为以下不可拆分顺序：

1. Builder 完成冻结实现；
2. Critic 对一个 exact revision 给出通过或阻断；
3. Evaluator 完成 `3407`、`3408`、`3409` 的全部两臂训练；
4. 所有 seed 均使用 terminal update-6000 `state_dict_ema`；
5. 使用 held-out ID 清单生成预测，但不读取类别、边界或聚合指标；
6. 所有预测及其 arm、seed、checkpoint、配置身份被封存；
7. **只有此后才打开官方 held-out 标签一次**，统一计算官方 mAP、配对区间和成本结果。

一旦第 7 步开始，禁止改变方法、预算概率、阈值、规则、checkpoint、统计量或重新生成预测。

---

## 四、种子冲突裁决

### 4.1 选择

**执行顺序冻结为：`3407 → 3408 → 3409`，三者全部执行；期间不读取 held-out 指标。**

我不采用较新综合裁决中的“3407 先读取，过门后再复制 3408/3409”。

### 4.2 理由

同一个 211-video population 既承担 3407 的路线门，又承担 3408/3409 的最终确认时，条件执行会使后续实验是否发生依赖 held-out 结果。即使承诺不修改模型，它仍然把宝贵的 held-out 集合用于研究路径选择，与本轮要求的“最终一次性读取”不一致。

完整三种子盲执行具有四个优点：

* 所有训练和种子身份在看到 held-out 结果前已经冻结；
* 不把单一随机种子当作机制裁决；
* 避免只在 3407 偶然正向时才展示多种子结果；
* 三种子聚合与配对不确定性可以在同一次读取中完成。

因此，较早的完整三种子顺序在本轮数据边界下更科学。此前综合裁决中的训练机制、matched arms、概率公式和性能门继续保留，但其条件种子顺序被本裁决取代。此前综合路线本身也明确，多预算实验尚未产生代码、训练、mAP 或成本证据。 

---

## 五、唯一当前任务单

### 任务名称

**H65 系统级多预算暴露适应：完整 200/211、三种子、一次性 held-out 裁决**

这是唯一获准的当前任务。不得并行恢复 frozen-detector oracle、Coverage-v1、Gumbel-Softmax、Mamba、Block Drop、DFT、TensorRT、旧三档 oracle 或新的内容条件预算控制器。

### 5.1 科学问题

在不改变 H65 的语义间接选帧机制、检测头、损失、NMS 和官方评估器的条件下，第二阶段同时暴露于嵌套的 `K256/K384/K512`，能否使整个 H65 系统学习到更稳定的跨预算表示，从而：

* 在固定混合预算清单上优于只接受 `K384` 暴露的 H65；
* 保持强制 `K384` 推理性能；
* 不增加实际进入 VideoMAE-S 的平均高分辨率观测量？

### 5.2 模型假说与反解释

**假说：** 当前 H65 在单一 `K384` 上完成第二阶段适应，Scout、selector、时序适配器与检测反馈可能共同过拟合这一暴露分布。保持位置集合嵌套并让既有系统在三个预算上联合适应，可能产生预算稳定性，而不需要新的动态预算模型。

**最强反解释：** 任何表面增益可能来自更多实际 VideoMAE 观测、不同数据或随机数消费顺序、改变 detector 训练日程、改变后处理，或者使用 held-out 结果调参。当前实验必须逐项排除这些解释。

### 5.3 唯一实验变量

唯一变量是 **H65 第二阶段的预算暴露分布**：

* **Control：** 全部逻辑更新固定 `K384`。
* **Candidate：** 在嵌套位置集合上使用 `K256/K384/K512`。

两臂之外不增加第三个训练臂。Candidate checkpoint 允许进行两个预先登记的只读视图：

* 强制 `K384`，用于安全门；
* 固定混合预算 manifest，用于主效应门。

### 5.4 数据身份

* 训练：manifest hash 为 `5b11…f4c0` 的完整 200-video `training`。
* held-out：manifest hash 为 `5f9a…746e` 的完整 211-video OpenTAD `validation`。
* 禁止使用 ActionFormer 212 进行并列表现选择或补算。

### 5.5 代码基座与允许移植面

**唯一模型基座：**

`04c35a3b76897e6c1569eeede41ed3aecaf7f854`

从该提交建立新的干净分支：

`feature/duca-h65-system-multibudget-exposure-v1-20260831`

`fdd2bcdd...` 只作为数据证据，不作为模型基座；代码库存分支也不得作为实验身份，因为库存中包含多路线重叠表面。

只允许移植或最小扩展以下行为：

1. `K256/K384/K512` 的变长 packet/tensor 对齐；
2. 保持同一 H65 打分和 producer order 的嵌套位置生成，必须满足
   `S256 ⊂ S384 ⊂ S512`；
3. 实际进入 VideoMAE-S 的有效观测量计数；
4. 强制 `K384` parity 推理；
5. 全 211-video 预测封存、官方评估和成本统计所需的最小入口；
6. 对应 focused tests 和最小 Slurm launcher。

禁止移植或新增：

* 新 selector、预算预测头或内容条件 controller；
* frozen-detector oracle 逻辑；
* Gumbel、Mamba、block dropping 或频域模块；
* 新 detector wrapper；
* detector architecture、loss、NMS 或 evaluator 修改；
* 通用 schema、工作流平台、复杂 provenance 框架或大规模重构。

### 5.6 可训练参数集合

严格复用 `04c35a3...` 的 H65 Stage-2 可训练 mask，不扩大到新的模块。该集合必须被解释为系统级适应，至少保留 H65 原有的：

* coarse/Scout trunk；
* action head；
* transition scorer 和 selector 相关路径；
* temporal adapter 与原有受控 detector-feedback 路径。

不得把任务重写成“冻结 Scout，仅训练 detector”。同样不得让 detector 梯度越过原有边界污染未授权参数。

### 5.7 训练与种子顺序

每枚 seed 内，Control 与 Candidate 共享数据顺序、增强、初始化规则、成功更新定义及随机数 producer 顺序；只有预算暴露变量不同。

冻结设置：

* seed 顺序：`3407 → 3408 → 3409`；
* 每臂每 seed：**6,000 次成功 optimizer updates**；
* warmup：前 500 updates；
  -阶段边界：update 3,000；
* optimizer：AdamW；
* base learning rate：`1e-4`；
* weight decay：`0.05`；
* adapter learning rate：`2e-4`；
* coarse trunk learning rate：`1e-5`；
* action head learning rate：`2e-5`；
* transition scorer learning rate：`5e-5`；
* global gradient clipping norm：`1.0`；
* AMP 与 EMA 均开启；
* EMA 在每个成功更新后更新；
* 每 500 updates 保存恢复点；
* update 3,000 和 6,000 必须保存；
* 唯一结果 checkpoint：update-6000 `state_dict_ema`；
* 不允许使用中间 held-out mAP 选择 checkpoint。

Candidate 的预算概率冻结为：

$$
p_{384}=0.5
$$

在 PRE_RUN 中测得三个预算下的实际高分辨率观测量
\(\mu_{256}<\mu_{384}<\mu_{512}\) 后：

$$
p_{256}
=
0.5\frac{\mu_{512}-\mu_{384}}
{\mu_{512}-\mu_{256}},
\qquad
p_{512}
=
0.5\frac{\mu_{384}-\mu_{256}}
{\mu_{512}-\mu_{256}}.
$$

若观测量不满足严格单调，或概率不在 `[0, 0.5]`，PRE_RUN 直接阻断，不得人工调整概率。

---

## 六、运行前准入、预测封存与决定性停止门

### 6.1 Builder 必须证明

* 三种 K 的 tensor、mask、position、packet 长度一致；
* exact-K 和嵌套关系成立；
* 不重复、不越界、不打乱 H65 producer order；
* `K384` 路径与原 H65 前向在容差内一致；
* 实际 observation counter 等于真正送入 VideoMAE-S 的有效输入；
* 两臂的非预算配置完全相同；
* 授权参数有预期梯度，未授权参数无梯度；
* 200-video training loader 完整；
* held-out 类别和边界不能进入训练；
* 不使用 raw-prediction cache；
* detector、loss、NMS 和 evaluator 未改变。

### 6.2 Critic 的独立审查

Critic 只审查一个 exact revision，并只可因以下问题阻断：

* 预算变量以外存在模型或优化差异；
* 梯度归属错误；
* K384 parity 不成立；
* producer/RNG 顺序破坏公平比较；
* 数据泄漏或集合身份变化；
* observation cost 统计不对应实际 VideoMAE 执行；
* evaluator、NMS、checkpoint 规则或统计量被改变；
* 正式路径不能运行。

代码风格、日志美化、通用完备性和非必要抽象不得形成修复循环。Builder、Critic、Evaluator 必须保持上下文独立，Pro 保留最终科学解释权。 

### 6.3 预测封存

全部六个训练单元完成后：

* 每个 seed 的 Control-K384、Candidate-K384 和 Candidate-mixed 均生成完整 211-video 预测；
* 预测视频 ID 集必须与 held-out manifest 精确相等；
* 每个预测文件只记录一个 SHA-256 和最小 sidecar：commit、config、arm、seed、checkpoint、state key、budget view；
* 封存后不得重新推理；
* 所有文件封存完毕后，Evaluator 才能打开官方标签计算结果。

### 6.4 配对不确定性

使用 **10,000 次整视频配对 bootstrap**：

* 三个 seed、所有 arms 使用相同的 10,000 组视频重采样索引；
* 每个 replicate 内分别重新计算官方五个 tIoU 阈值 mAP；
* 先计算每个 seed 的 Candidate-Control 差值，再取三 seed 平均差值；
* 主要区间为第 250 与第 9,750 顺序统计量；
* 同时报告每个 seed 的点值、三 seed 平均值和 seed 间标准差；
* Avg-mAP 与 mAP@0.7 为预先登记的共同主判据，不得只选择有利阈值。

### 6.5 真实成本口径

主要成本量是：

* 实际进入 VideoMAE-S 的有效高分辨率帧或 tubelet 数；
* 全 211-video 总量及逐视频分布。

同一硬件、相同 batch 和精度设置下同时报告：

* 端到端逐视频 wall-clock p50/p95；
* GPU 峰值显存；
* Scout、VideoMAE-S 和 detector 分项时间。

padding 后名义长度、理论 FLOPs 或配置中的 K 不能替代真实成本。

### 6.6 最终一次性门

全部门在三 seed 联合读取后一次应用。

**K384 保持门：**

Candidate-K384 相对 Control-K384 的三 seed 平均差值必须同时满足：

* Avg-mAP ≥ `−0.2 pp`；
* mAP@0.7 ≥ `−0.2 pp`。

**混合预算主效应门：**

Candidate-mixed 相对 Control-K384 必须同时满足：

* Avg-mAP ≥ `+0.8 pp`；
* mAP@0.7 ≥ `+1.0 pp`；
* 两项三 seed 配对区间下界均大于 `0`；
* 实际进入 VideoMAE-S 的总观测量不大于 Control-K384；
* 不存在数据、代码、评估或成本有效性违例。

任意条件失败，即停止 **H65 多预算暴露适应假说**。不得通过换阈值、挑 seed、改 manifest、补训、改 checkpoint 或恢复旧路线补救。

在 held-out 打开前发生的确定性启动器、路径或环境故障，只允许在同一科学设计内进行一次最小修复并重新通过 Critic；它不构成模型负结果。在 held-out 打开后，不再授权修复或重跑。

---

## 七、负责人和北京时间绝对截止时间

这是一条连续任务链，不是多个并列任务：

* **Builder 截止：2026 年 9 月 4 日 23:59:59，北京时间。**
  提交最小实现、两臂配置、focused tests 和 exact clean revision。
* **独立 Critic 截止：2026 年 9 月 6 日 23:59:59，北京时间。**
  对 exact revision 返回通过或唯一具体 blocker。
* **Evaluator PRE_RUN 截止：2026 年 9 月 8 日 23:59:59，北京时间。**
  冻结 \(\mu\)、预算概率、manifest、统计重采样序列、成本入口和所有运行身份。
* **全部三种子训练与预测封存截止：2026 年 10 月 1 日 23:59:59，北京时间。**
* **一次性官方评估、bootstrap、成本报告及科学证据回传截止：2026 年 10 月 2 日 23:59:59，北京时间（UTC+8）。**

资源或集群若形成客观阻塞，Evaluator 只回传阻塞事实，不得提前打开 held-out 结果或缩减为单种子结论。

---

## 八、当前可写入与不可写入论文的边界

### 现在可以写入

* DUCA 正式训练集合是完整 200-video `training`。
* 正式 held-out evaluation 是完整 211-video OpenTAD `validation`。
* annotation、loader、media 和 evaluator 身份一致。
* ActionFormer 212 的唯一 annotation 差异是 `video_test_0000270`。
* `video_test_0001292` 不属于 ActionFormer annotation evaluation set。
* 411 个 canonical 视频通过基本解码检查。
* 本次身份审计没有访问 held-out 动作类别或时间边界。
* H65 系统级多预算暴露实验已经获得科学执行许可。

### 仍然不能声称

* 多预算模型已经实现或通过 PRE_RUN；
* 多预算暴露提高 mAP；
* 动态预算已经有效；
* DUCA 已取得性能—成本联合收益；
* H65 优于官方 dense AdaTAD；
* 该任务是 detector-only adaptation；
* 数据身份证据证明 selector、Scout 或 detector 有效；
* 任何旧 oracle、Coverage、Gumbel、Mamba、Block Drop、DFT 或 TensorRT 路线已恢复；
* 在正式结果产生前，任何 seed 或预算设置已经成功。

数据身份事实、模型假说和尚未获得的性能证据必须在后续所有报告中保持这三层分离。

DUCA_FULL_DATA_IDENTITY_ADMISSION_READY
