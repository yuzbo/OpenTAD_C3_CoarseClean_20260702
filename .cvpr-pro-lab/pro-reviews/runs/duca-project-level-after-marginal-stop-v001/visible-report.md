PIVOT
DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831

## 一、项目级科学裁决

**不恢复 DUCA-Marginal-v1，也不停止整个 DUCA 问题；项目只转向一次“整视频预算一致、跨视频转移计算”的终局低成本证伪。**

这是目前唯一同时满足以下条件的新机制假设：

1. 没有被现有固定预算、Coverage、coreset、连续片段、PJST-D1 或 Marginal-v1 直接检验；
2. 能解释为什么逐窗口预算转移改善局部代理，却破坏最终检测指标；
3. 可以完全复用已经密封的 K256/K384/K512 预测，不训练、不访问 official validation/test、不执行新的 VideoMAE 前向；
4. 若在这种对假设极为有利的开发集 oracle 条件下仍失败，就没有合理依据继续投入完整训练。

远端证据分支头确为 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d`；它比父提交只前进一个提交，唯一修改面是联合诊断 runner 与聚焦测试，没有修改 allocator、模型或训练代码。

## 二、当前证据是否足以终止整个研究问题

### 1. 已被直接证据否定的内容

本轮附件支持以下直接结论：

* 连续高分辨率片段作为采样单元已经得到明显负结果，联合训练没有恢复；
* 固定 K384 的任务状态 coreset 低于匹配 uniform，不能再把更细粒度的语义重排默认视为收益来源；
* Coverage-v1 连最基本的集合变化、覆盖改善和最大空洞机制门都没有通过；
* PJST-D1 的匹配点估计没有 Avg-mAP 正向支持；由于整视频配对区间没有生成，它不是已确认的总体负效应，但也不再提供路线级正证据。
* Marginal-v1 的 50% capped oracle 只有 `+0.725589/+0.729004` 个百分点；解除 cap 后下降到 `+0.427310/+0.450280`；
* capped→released 的全部 96 个合法联合状态中，`0/96` 达到 `+0.8/+1.0` 门，八个最小等成本转移没有一个同时改善 Avg-mAP 与 mAP@0.7。

因此，**逐窗口反事实 detector loss、逐视频内部等成本、加性预算分配**已经终结。当前 allocator 确实逐视频独立求解，目标成本固定为该视频全 K384 的实际 observation 总和；短窗口按照 `min(valid,K)` 计费，无法通过名义 padding 制造虚假稀疏。

### 2. 最合理的跨路线失败机制

以下是基于结果作出的因果解释，而不是已被直接证明的事实：

**第一，动作性或边界显著性不等于重型表示的边际价值。**
Scout 可以指出“哪里像动作或边界”，却未必能判断减少哪些上下文不会改变类别置信度、重复 proposal、边界回归和最终排序。H65 与 dense 之间仍约有 3.6 点 Avg-mAP 差距，也表明当前稀疏观察本身已经损失了相当的表示容量；旧状态报告同样明确，现有实验还没有形成动态预算或端到端成本优势。

**第二，逐窗口预算决策与最终检测单位错位。**
THUMOS14 滑窗检测中，同一动作常由多个重叠窗口共同产生 proposal。一个视频内部同时存在 K256、K384 和 K512 窗口，会让重叠预测具有不同的表示质量、置信度校准和边界偏差。最终 Avg-mAP 又经过视频内预测合并、Soft-NMS 以及跨数据集排序，不是窗口 loss 的可加函数。cap release 改善加性目标却降低最终 mAP，正是这种错位的直接迹象。

**第三，旧路线反复在“提高边界密度”和“保护时间覆盖”之间交换误差。**
连续片段过度聚集，coreset 和 Coverage 又没有找到同时保护上下文与边界的固定预算结构；动态窗口转移则在两个最终指标之间产生冲突。这更像决策单位与目标失配，而不是尚缺一个学习率、cap 或 tie-break。

### 3. 尚未排除、但也没有正证据的部分

现有 Marginal allocator禁止视频之间转移计算：每个视频必须独立守恒自身 K384 成本。
因此，它没有检验：

> **一个视频的全部重叠窗口保持同一预算档位，同时把总计算从低价值视频转移到高价值视频。**

这是本轮唯一保留的未决机制。其他更宽泛的类别感知、proposal 集合效用或新 Scout 训练方式只是逻辑上未被否定，并没有足够正证据或一个同样便宜的决定性实验，本轮不授权。

## 三、当前结果能否形成可投稿贡献

**不能形成一篇以 DUCA 为有效方法的主论文贡献。**

欠缺并非主要是工程完成度，而是以下科学要件：

1. **缺少通过机制门的新方法。** 当前可写的是一组约束明确的负结果，而不是一个有效候选。
2. **缺少严格匹配的性能—真实成本曲线。** 共享 dense AdaTAD 为 `68.73`，H65 为 `65.13`，但现有材料明确指出它们尚未构成完整的同计算预算论文比较。
3. **缺少可部署动态策略。** Marginal 只完成训练侧 40 视频 oracle，没有证明 Scout 能预测最终预算价值。
4. **缺少足够统计证据。** 多项历史结果是单种子；PJST-D1 没有完成预注册配对区间；Marginal 的事后 oracle 状态也不能补 bootstrap 后冒充确认性结果。
5. **缺少 official validation/test 主结果和实测端到端开销。**

现有材料最多适合作为论文的失败机制分析、补充材料或一篇以系统性负证据为主题的独立研究记录。代码、测试、Job 和归档完整度不能替代方法贡献；项目规则也明确要求只有能改变论文判断的实验进入关键路径。 

## 四、唯一新机制：整视频预算一致、跨视频计算转移

### 一句话科学问题

> 在相同或更低的全局实际 observation 成本下，给一个视频的全部重叠窗口使用同一预算档位，并在视频之间转移计算，能否避免逐窗口混合预算造成的 proposal 不一致，从而同时改善 Avg-mAP 和 mAP@0.7？

### 因果机制

* **视频内一致性：** 一个视频的所有窗口统一请求 K256、K384 或 K512，不再让同一动作的不同重叠视图使用不同表示容量。
* **视频间动态性：** 从整体较容易的视频释放计算，交给整体边界风险更高的视频。
* **与检测单位对齐：** 计算决策单位由局部窗口提升为完整视频 proposal 集合，更接近 Soft-NMS 和最终 AP 所作用的单位。
* **H65 保持不变：** 各预算档仍使用同一密封 H65 priority sequence 的嵌套集合，因此不把新 selector 混入本轮。

### 为什么不是 Marginal-v1 换名重试

本机制同时改变了三个根本合同：

* Marginal-v1 在**同一视频内部**移动预算；新机制只在**视频之间**移动预算；
* Marginal-v1 允许同视频窗口混用三档；新机制要求整视频统一请求档位；
* Marginal-v1 以窗口 detector loss 为加性效用；本轮直接用最终 proposal 集合的 Avg-mAP 与 mAP@0.7 作**开发集特权 oracle 裁决**。

它不调用旧 `allocate_equal_budget_marginal_reallocation`，不改变 cap，不扩展 96-state，不训练旧 utility head，也不修改 K 档位或 H65 priority sequence。

## 五、冻结的唯一当前任务：整视频双向转移 oracle falsifier

### 1. 数据和证据等级

只使用现有密封的：

* 40 个训练侧 controller holdout 视频；
* 124 个窗口；
* K256、K384、K512 三套窗口预测；
* 每个窗口的真实 observation 成本；
* 固定 split、annotation、class map、epoch-59 EMA detector、Soft-NMS 和评估器。

这是**开发集机制 oracle**，不是 official validation，不是 official test，也不是部署策略。

### 2. 唯一干预

固定全 K384 为基线。对每一对不同视频 `(d,u)` 构造一个候选：

* donor 视频 `d` 的全部窗口统一请求 K256；
* recipient 视频 `u` 的全部窗口统一请求 K512；
* 其余视频全部保持 K384；
* 因短窗口而与 K384 实际成本相同的请求，继续按现有 sealed accounting 折叠；
* donor 与 recipient 都必须至少包含一个实际非基线窗口。

候选生成不读取标签、GT 或 mAP。标签只在随后调用既有评估器、从所有候选中找开发集 oracle 最优状态时使用。

最多有 `40×39=1560` 个有序候选；只保留满足

$$
C(d,u)\leq C_{\mathrm{fixed}}=47110
$$

的候选，其中成本始终是

$$
\sum_i \min(V_i,K_i)
$$

而不是 requested K、execution slots 或 padding。

### 3. 公平对照与不变量

必须保持逐字或逐对象不变：

* H65 priority sequence 和所有密封三档预测；
* detector、Scout、checkpoint、模型参数和配置；
* 数据划分、annotation、类别映射；
* detector grid、坐标逆映射；
* loss、Soft-NMS、官方评估器；
* K256/K384/K512；
* `+0.8` Avg-mAP 与 `+1.0` mAP@0.7 项目级实用门。

本任务不重复 dense、uniform、coreset 或完整训练。

### 4. 指标与唯一通过规则

报告：

* tIoU 0.3、0.4、0.5、0.6、0.7 的 mAP；
* Avg-mAP；
* mAP@0.7；
* 每个候选的 actual observation 总数及相对 K384 的变化；
* legal candidate 数与 passing candidate 数。

**只有存在至少一个合法 `(d,u)` 候选同时满足**

* `ΔAvg-mAP ≥ +0.8` 个百分点；
* `ΔmAP@0.7 ≥ +1.0` 个百分点；
* actual observation 总成本不高于 `47110`；

才判定开发集 whole-video action space 有足够 headroom，并返回 Pro。

最佳状态按以下顺序确定：

1. 最大化
   `min(ΔAvg-mAP − 0.8, ΔmAP@0.7 − 1.0)`；
2. 成本更低；
3. donor、recipient 视频 ID 字典序。

### 5. 停止规则

以下任一情况直接终止本 falsifier：

* 密封 fixed/capped/released 锚点复现误差超过 `1e-6` 个百分点；
* 任何输入 SHA、视频集合、窗口集合或成本口径不一致；
* 没有合法的 donor–recipient 候选；
* `passing_candidate_count=0`。

前三类身份或确定性实现问题只允许在**不改变机制与门槛**的情况下作一次最小修复。

若正式结果为 `0` 个通过候选，裁决自动转为：

> **项目级 STOP：在当前 THUMOS14、H65 priority sequence、三档真实 observation 动作空间和可接受资源边界下，停止 DUCA 方法创新。**

不得随后增加第三个视频、组合多个 transfer、降低门槛、改变 K 档位、访问 official test、补 bootstrap 或训练控制器。那会把一个失败的 privileged falsifier 变成同开发集上的组合搜索。

本任务不做 bootstrap。若点门通过，最佳候选是从最多 1560 个开发集状态中事后选出，普通 bootstrap 也不能使它成为确认性或可部署结果；它只能证明值得另行讨论 predictability。若点门失败，bootstrap 同样不能改变实用效应门失败。

## 六、最小 Builder、Critic 与 Evaluator 表面

### Builder

从只读证据提交 `46812facc8773d9b4a9c21833cbe397c8aaa5a2d` 建立新分支：

`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`

只允许新增：

1. `tools/bata/run_duca_whole_video_consistent_budget_falsifier.py`
2. `tests/test_duca_whole_video_consistent_budget_falsifier.py`
3. 一个只负责绑定输入和启动该 CPU evaluator 的薄 N16R4 sbatch 文件；若现有通用提交方式足够，则不新增该文件。

不得修改：

* `tools/bata/run_duca_marginal_frozen_h65_probe.py`
* `opentad/models/duca/dynamic_budget.py`
* selector、acquisition、detector、训练器或配置；
* 既有预测与结果文件。

当前提交的测试已经覆盖 K384 精确保持、真实短窗口成本、packet padding、cap tie-break、全部 96 状态自动推导及旧结果不可覆盖，因此新测试只补 whole-video 候选特有的不变量，不再复制旧合同。

### 独立 Critic 只检查四项

1. 每个 changed video 是否真正对全部窗口使用同一 requested tier，且没有调用旧 Marginal allocator；
2. 候选集合是否在读取 GT 或指标前完整生成；
3. 是否使用真实 observation 成本并满足 `≤47110`；
4. 是否完全复用密封预测、相同 Soft-NMS/评估器，且没有 official validation/test、模型前向或训练。

通过即停止审查，不增加风格、schema 或通用完备性修复。

### Evaluator PRE_RUN

只执行：

* exact HEAD、clean tree 和两项 focused tests；
* 三个 producer、receipts、split 与终态 JSON 的身份检查；
* 原 fixed/capped/released 指标零误差复现；
* `40` 视频、`124` 窗口、`47110` 基线实际成本核对；
* 生成但不评价完整候选清单，确认候选唯一且非空。

### 唯一正式运行

一次 CPU-only evaluator 作业：

* 先评估 fixed K384；
* 再评估全部合法 donor–recipient 候选；
* 写出每个候选的预算、成本、六项指标、相对基线点差和最终通过数；
* 不执行 detector/Scout forward、梯度、训练、bootstrap 或 official test。

即使调度分区要求申请 GPU，该作业也必须清空 CUDA 可见性；调度资源不能被描述为模型计算成本。

## 七、负责人、依赖与截止时间

`next_owner`: **Builder**

`next_action`: 在新分支上实现上述独立 whole-video falsifier 与聚焦测试；完成后依次交给独立 Critic 和独立 Evaluator，Evaluator 只提交一次正式运行。

`dependency`:

* exact commit `46812facc8773d9b4a9c21833cbe397c8aaa5a2d`；
* 三个密封 K256/K384/K512 producer 产物及 receipts；
* `probe_result.json`、`oracle_cap_release_result.json`；
* 终态 `oracle_cap_release_neighborhood_result.json`，SHA-256
  `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`；
* 已绑定的 split manifest、block list、annotation、class map 和 evaluator；
* 无新数据、无模型训练、无 official-test 权限、无 GPU 科学计算依赖。

`absolute_deadline`: **2026-09-02T18:00:00+08:00**

## 八、论文主张边界

### 若点门通过

只允许写：

> 在训练侧密封开发 holdout 上，一个从全部整视频 donor–recipient 转移中事后选择的特权 oracle，在不超过固定 K384 实际 observation 成本时达到了预注册的 Avg-mAP 与 mAP@0.7 headroom 门。这支持继续检验“视频内预算一致性”是否可由低成本 Scout 预测。

仍不得声称：

* 存在可部署控制器；
* Scout 能预测 donor 或 recipient；
* official validation/test 有收益；
* 结果具有确认性置信区间；
* 已获得真实端到端速度或能耗收益；
* DUCA 优于 dense AdaTAD。

### 若点门失败

应归档：

* 新 exact commit、作业、输入 SHA；
* legal candidate 总数和全部成本；
* fixed 锚点；
* `passing_candidate_count=0`；
* 最优 Avg-mAP、最优 mAP@0.7 和最优联合门候选；
* 没有模型前向、训练、bootstrap 或 official test 的事实。

论文可陈述：

> 在窗口级加性分配失败后，我们进一步穷尽了全部合法的单次整视频 donor–recipient 转移。即使保持视频内预算一致并允许跨视频移动计算，也没有开发集 oracle 状态达到预注册的联合实用门。

这足以在当前资源边界下停止 DUCA 项目级方法创新，但仍不能外推为“所有任务感知动态计算、所有 priority sequence 或所有视频骨干上的动态预算均无效”。
