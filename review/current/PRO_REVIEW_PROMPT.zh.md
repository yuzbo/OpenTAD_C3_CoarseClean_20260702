# 给 ChatGPT Pro：当前 GeoSparse 代码、矩阵和性能的独立严格审查

请用中文审查本文件所在**固定GitHub提交**。先报告实际能够读取的提交、文件与材料缺口，再做判断；读不到代码时不能假装已经逐行审计。仓库默认分支是无关C3，审查必须使用提供的GeoSparse permalink。需要全面、具体、可证伪的意见，不需要为了严厉而制造错误，也不要把方案文档中的待实现机制当作现有代码。

入口：[START_HERE_PRO_REVIEW.md](../../START_HERE_PRO_REVIEW.md)。生产模型M=`b70ae056c495b43ca3f305fe438926b97b2723b5`；官方O=`346d09d19e2091372cec48172dbe40f7b28bdee6`；研究与执行工具已在当前审查提交合并。模型和官方核心相同不等于外围训练、EMA、检测mask、评价或成本一定正确。

## 审查目标

回答：A/B/C实际实现与登记方法是否一致；目前哪些性能差异来自已证实的代码错误、训练/推理不一致、预算不匹配、架构损失或尚未验证的假设；实际矩阵能否按当前代码完成；最少哪些补测或改动能产生可信的路线判断。**不要给出“完全正确”的无条件保证。**

对每项发现给出：当前提交的`file:line`与调用链、可达触发条件、错误机制、影响边界、最小修复、能证伪该判断的检查，以及旧checkpoint应保留/补测/重训哪一种。正确部分也明确说明。把已证实BUG、合理疑点、未实现研究、方法负结果和历史已修复分别标注。

## 不可偷换的实验事实

- 全200 THUMOS训练；正式验证全211视频/792窗口；VideoMAE-B识别预训练，768输入、160px、tubelet2、48个Heavy parent、native384 TIA、检测768。warmup5、cosine horizon100、训练终点60、B adapter LR1e-4。不得凭文件名误判LoadFrames stride。
- 官方原版独立训练与`geosparse_dense_control`分开：原版双卡每卡1与统一单卡effective2的局部positive normalizer、AMP overflow后scheduler/EMA推进、RNG轨迹可能不同。官方独立训练已完成60轮，best E52=70.1805820128%，E60=69.8879407657%；发布权重全量推理71.1387948970%是另一实验。
- 用户明确选择full-test EMA best并允许之后按测试集做超参数开发。保留用户规则；新调参结果标test-tuned development，不能称独立泛化。测试GT不能进入训练梯度或utility标签。不自行恢复180/20 holdout或三种子。Geo每5轮12次选点与官方42/44/.../60共10次不同；固定60及共同50/60节点可补充比较。
- 现有A/B/C fixed .5及dynamic seed0保持冻结源码与身份。首要验证完整主方法，控制按资源运行。不要建议在正在训练目录热改后仍沿用同一任务ID，也不要无证据重跑官方训练或整个矩阵。

## 先解释已有实测，禁止把高精度直接当低预算收益

详细记录见`results/`，mAP内部存储为0–1，本文百分数为×100。

1. A dynamic已完成60：best E40=68.0196%，E60=63.7006%；独立同best导出68.0156%，微小差异保留。**E30、E40的全部792窗口均argmax q1，eligible-full Heavy=100%。** padded-full均值96.97%来自有效尾部，不是路由节省。A fixed q=.5 E50=61.2806%，完成55轮待续训。没有正式同GPU配对的总延迟比较。
2. B dynamic E20=30.2768%；E10全部q=.25，约22.43%eligible-full Heavy，E20未做预算回放。B fixed E15=24.2322%。B-full同新query/receiver/TIA结构全Heavy的原任务已开始但无结果，不能预设B损失具体归因。
3. C dynamic E20=53.4452%，mAP@.7=29.4846%，高于A同期，尚非最终失败。**C E10全部q1、所有原生组all-fine、100%eligible Heavy。** E20账本缺失，已单独提交245503完整回放，当前PENDING；不要用E10冒充E20。C fixed E15=40.3499%。C的q=.5指mixed token比例，对应fine组约1/3，并非50%fine或50%总MAC。
4. A E40→60训练任务loss .50850→.42866，而测试下降4.319个百分点。可能有过拟合或路由计划变化，但现有证据不能唯一确定原因。A实际5994个成功optimizer update、6次AMP overflow；续训重复日志已去重，不能算作额外训练。
5. B dynamic E20全模型preclip梯度p50=13.09、p95=111.71；A/C同期p50=3.80/3.48。全模型global clip1。actor标量约−4000不证明actor梯度支配；尚缺分量梯度及方向测量。
6. A60共157次去重probe，|gain|中位3.73e-5、约72%<1e-4、正号约51%。这可能使校准信号弱，但不能仅以数值小断言浮点噪声或断梯度。gain head确实参与实际acquisition辅助损失。

## 必须阅读和核查的调用链

### 一、模型与原生几何

阅读`geosparse_ext/model.py, sparse.py, detector.py, geometry.py, contracts.py`及调用的官方ViT/TIA/head。

- A是否真的在QKV/attention/MLP前gather并按原生位置scatter；parent attention是否跨组；原LN/residual/TIA顺序及全窗口作用域是否正确；冻结权重是否保留上游adapter梯度。
- C每层coarse/fine互斥、coarse投影与delta回写、all-fine极限、all-coarse非零计算是否成立；是否有PE/支持/尾部或共享更新的实际错误。不能把其与MSViT/ToMeSD相近直接当实现错误。
- B实际为dense cheap query＋稀疏Heavy evidence＋receiver。Heavy内12层TIA禁用；4slot/tubelet、768→256投影、Scout128→query256、2receiver层，最终一次native384规则TIA后插值768。原生全窗口TIA现已修复；不要重复旧TIA8指控。分别检查这一架构是否符合当前B规格及为何可能丢性能，区分设计代价与bug。
- Heavy tubelet eligibility与detector frame mask是否真正分离，767有效帧不得扩大到768；query384映射、padding、empty GT、K0/Kall、DropPath/checkpoint、AMP是否有实际未覆盖路径。
- B anchor/source/context支持是否被混淆，disjoint union与hull的使用是否正确；recorded ROI是否足以解释真实augmentation。未实现原图ROI、缓存或provenance gate不得宣称完成。

### 二、路由目标与预算退化——最高优先级

阅读`routing.py, model.py, runtime.py`以及`docs/evaluation/dynamic-budget-audit.md`。

- 训练categorical采样预算，验证argmax；目标中的actualMAC、critic、dual、cost EMA各按什么粒度和时机更新？EMA验证是否还改变了控制器参数/菜单？训练期望成本约束为何允许确定性部署全q1或全q.25？请用真实代码说明，而非只说“加大lambda”。
- 有序PL logprob、位置执行排序、探索时actor/critic禁用、warmup及训练curriculum、正负号、detach、loss规模、全模型clip是否一致？不要随意按K平均logprob作为“无偏修复”；说明目标是否改变。
- 当前acquisition是signed source-loss添加收益，C是coarse→fine；不是retention/swap/R1。反事实应重跑受影响Heavy/TIA/head，并恢复RNG/buffers/mode。值依赖整个计划；不能删末端cached feature冒充真实少算。
- 固定预算需要同parent同成本swap，动态预算需要真实边际cost；指出目前已实现与尚缺的连接。现有SCI3 pair runner只在eval收集R0，不等于训练swap模型。
- 给出使目标预算、训练策略、部署策略一致的最小方案与可行性条件；比较硬可行分配、可校准阈值、期望约束/对偶优化等具体取舍。不能事后按测试成绩删q1档、把q=.5说成MAC=.5，或把约束成本移到cheap/TIA/receiver外后不报告总成本。
- Heavy近似MAC为每层每parent `12*k*d*d + 2*k*k*d`，正attention项。相同token总数不等于相同二次cost。160使用10×10基准，不能再除224的14×14分母。区分eligible-full、padded-full、整个模型MAC和实测延迟。

### 三、数据、训练状态、评价与选点完整性

阅读`data.py, records.py, runtime.py, training_validation.py, prediction_export.py, evaluation.py, metrics.py, entry.py`。

- 当前internal_diagnostic必须映射全部200 training视频及正确root/pipeline；正式211覆盖包括空预测；GT、短动作阈值和校准不能从测试进入训练；不能只测成功匹配边界而隐去miss。
- EMA进入/退出正常与异常路径是否完整恢复被冻结和可训练状态、RNG与modes；checkpoint、reported EMA score、best epoch与权重身份是否匹配。
- 训练完成与所有正式选点完成是否区分；某epoch验证失败能否错误DONE/benchmark？已保存epoch补测是否可恢复完整best，不需要重训。
- 官方训练和统一trainer的optimizer groups、局部normalizer、effective batch、scheduler/EMA/AMP步进、恢复轨迹等差异应如实命名。不能以官方子树相同推出训练等价。

### 四、矩阵、执行和硬件

读取实际`manifests/experiments.full.jsonl`及primary/secondary，结合`protocol.py, capabilities.py, matrix.py, slurm_queue.py`和`status/`。

- 1545个ID实际为612训练/612评价/204benchmark/117diagnostic，204组合；所有训练登记60轮三seed，当前seed0执行。逐项判断准入是否真的消费所有variant字段；静态接受与运行/完整语义不能混淆。
- full/focus共有ID当前整理检查无字段差异；仍要检查其产物依赖、epoch零基映射、checkpoint保存列表、blocked字段和恢复逻辑。旧I10不能未经比较直接复用。
- 最新123静态准入/81阻塞、117diagnostic阻塞；v3原始805清单缺失，新30工单/22图/6表不能称已运行。明确哪些关键主方法、强基线、消融和诊断尚无实现/产物。
- benchmark的CUDA UUID与Slurm映射、独占/同步/warmup、OOM失败状态、来源权重与accuracy精确配对、batch1/8/32、p50/p95、模型/device/decoded/encoded计时边界是否正确。4090与A100不能混合排名；窗口吞吐不是视频吞吐。
- `figures.py`如何处理same epoch/source/weight、single-seed、缺失数据与成本分母。不得将方案预设40%/67mAP、CPU合成测试或attention热图当科学结果。

### 五、新研究代码与旧审计修复

阅读`geosparse_research/`和三个SCI3测试，核查forced-plan成对干预、同parent等成本swap、状态恢复、真实官方AP的video bootstrap，包括重复视频、空预测与missing-class draw。17+6项CPU测试仅是已有有限检查，不免除源码阅读和GPU/真实视频证据。

用`reference/prior_independent_audit.zh.md`的I01–I15逐项对照当前M：分为已修复且有证据、部分修复、仍存在、需要现场验证、已显式阻塞。该报告针对旧55d5429e/902fa05，不代表当前M；不要把修复前截图当现状。另给出旧清单之外的新问题。

## 输出要求

1. 先给可以支持/尚不能支持的总体结论，列实际阅读范围和无法验证的资产。
2. 按严重度列真实发现，每项有源码证据、触发配置、影响边界、修复和证伪检查。对于不存在的指控明确撤销，不拼凑数量。
3. 给A/B/C与dynamic/fixed/full控制的可用性表：实现状态、当前性能成熟度、同成本证据是否存在、还缺什么。尤其直接回答：**C动态是否退化成全量，哪些epoch有直接证据？**
4. 分析性能差的前三个最值得检验的解释及各自反证；区分共享训练问题、B架构损失、预算失配和真实方法劣势。
5. 给按收益排序的最小优化计划：先无需训练的配对诊断/补测，再有依据的代码修复，再新命名空间的匹配训练；列哪些checkpoint可复用、哪些才需重训。不能用“重跑全部”代替定位。
6. 审核研究贡献边界：A面对MoD/CoDA/LITE；B面对mTAN/Coarse-Fine/LookupViT；C面对MSViT/ToMeSD等。相关工作构成研究挑战，不自动证明实现错误或优劣。新增模块前先设计能推翻其必要性的强对照。

请在文末列出尚不能回答的问题。收到你的建议后，我们会核对源代码与必要实测再采纳；此次讨论本身不是生产验收。
