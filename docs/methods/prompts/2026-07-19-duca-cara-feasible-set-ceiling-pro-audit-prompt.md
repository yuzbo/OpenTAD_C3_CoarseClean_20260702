# DUCA-CARA 可行集合上限实验 Pro 严厉审核 Prompt

请以最高推理强度完成一次只读、证据优先、严格限定范围的代码与数学审核。

本轮不是让你继续发散新模型，也不是让你直接设计整篇论文或长训练矩阵。唯一任务是：

> 审核当前 DUCA/CellCF 的真实可行集合，并给出一个可实现、可证明、可审计的
> allocation-family ceiling 实验规格，用它裁决 coverage scaffold + adaptive
> residual allocation 是否值得进入模型训练。

不要修改仓库，不要提交代码，不要运行长训练。可以给出核心实现代码、伪代码、测试代码
和精确文件级修改建议，但必须明确它们是 `[PROPOSAL]`，不能写成已实现事实。

---

## 1. 固定仓库与精确提交

仓库：

`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

本轮代码 source of truth：

- 分支：`codex/duca-cellcf-evidence-20260717`
- 精确提交：
  `4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- commit：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- exact tree：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/4ce69c852bdbd902046b47bc6019ae11e850dbe4`

冻结模型提交：

- `1642f265e48391418a7c8a4a087e33e2b7bf6899`
- commit：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/1642f265e48391418a7c8a4a087e33e2b7bf6899`
- exact tree：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/1642f265e48391418a7c8a4a087e33e2b7bf6899`

已本地核验：`1642f26` 是 `4ce69c8` 的祖先。`4ce69c8` 只增加后处理/成本数据契约修复，
不得把它说成重新训练得到的新模型。

### 强制可见性证书

回复开头必须输出 `VISIBILITY_CERTIFICATE`，列出：

1. 实际解析到的仓库和完整 SHA；
2. 实际打开并完整阅读的文件；
3. 无法读取、被截断或只看到摘要的对象；
4. 是否看到模型、配置、数据坐标、诊断工具、测试和成本代码；
5. Prompt 叙述与代码不一致的地方。

若不能读取精确提交，输出 `VISIBILITY_BLOCKED` 后停止代码裁决。不得根据文件名、
commit message 或本 Prompt 猜测实现。所有 `[CODE_FACT]` 必须带 GitHub 文件及行号链接。

---

## 2. 不得改写的研究目标

这是完整窗口可见的离线 TAD，不是 Online、Streaming 或 Causal TAD。

最终目标是：在昂贵视频 backbone 前，使用低成本动作/背景状态证据和其时间变化，
把有限帧/片段预算优先分配给高 IoU 定位真正需要的位置，在真实 decode-to-output
总成本下降时尽量保持完整输入 TAD 性能。

设计初心：

1. coarse 模型只承担较容易的动作/背景二分类；
2. selector 根据 `p_action` 的状态变化、不确定性和 temporal hidden 做间接边界定位；
3. selector 的首要目标是边界覆盖和下游检测效用，不是动作内部 actionness top-k；
4. validation/test GT、teacher、oracle、ledger 和 prediction cache 不得参与部署选择；
5. 固定 `K=384,T=768` 是当前归因锚点，Dynamic MUST 继续冻结；
6. 当前阶段不引入 MobileNet、X3D、SlowFast、第二 detector 或新的动态预算模块。

---

## 3. 已冻结的代码与实验事实

以下内容请先独立核验，再分别标记为确认、修正或无法确认。

### 3.1 当前 CellCF

当前 `local_cell_deformation`：

- 由 exact-uniform anchors 划分 Voronoi cells；
- 每个 cell 恰好选择一帧；
- 不允许背景 cell 释放 quota；
- 不允许高 transition cell 获得第二个 quota。

在 `T=768,K=384` 下，本地按提交中的 rounding/midpoint 规则重建得到：

- 382 个长度 2 的 cell；
- 一个长度 3 的 cell；
- 一个长度 1 的 cell；
- 每个 anchor 最大只移动 1 个 dense-grid index；
- exact uniform 最大未选 hole 为 2；
- CellCF 理论最大未选 hole 为 3。

请逐行核验这些结论，并判断它们是否足以证明：

> 当前 CellCF 只能进行 uniform phase/content correction，不能进行跨区域
> boundary-adaptive budget allocation。

### 3.2 坐标合同

当前 CellCF 实际从 acquisition `selected_positions` gather 输入，但 local-cell 路径把
`detector_grid_positions` 设为 uniform `anchor_positions`；正式配置还设置：

- `detector_output_coordinate_space="selected_axis_index"`；
- `remap_gt_to_selected_axis=True`。

请核验实际 observation time、detector prior time、GT time 和 prediction time 是否一致，
并判断当前结果能否被解释成真实物理时间边界选择。

仓库已有 physical-grid ActionFormer 路径。请核验它是否：

- 读取 `irregular_selected_positions` 或 `selected_dense_indices`；
- 禁止 selected-axis GT remap；
- 把 point center、stride、regression range 映射到实际物理时间；
- 能被新 selector 复用，而不必重写另一个 detector。

### 3.3 梯度语义

正式 CellCF 使用：

- `detector_gradient_mode="none"`；
- hard counterfactual detector losses 在 `torch.no_grad()` 下产生；
- detached signed utility 训练 policy scorer。

请核验它是 detector-derived supervision，而不是 direct detector-loss backpropagation。
不要把这两种机制混为一谈。

### 3.4 matched seed-0 结果

用户提供的 immutable terminal-EMA 原始结果：

| arm | Avg-mAP | @0.3 | @0.4 | @0.5 | @0.6 | @0.7 |
|---|---:|---:|---:|---:|---:|---:|
| exact-uniform | 63.8594 | 78.8009 | 73.4968 | 66.5040 | 56.8974 | 43.5978 |
| transition-beta0 | 64.2755 | 78.9614 | 74.4893 | 67.2996 | 57.4936 | 43.1336 |
| CellCF | 64.0610 | 78.8992 | 74.6776 | 66.6185 | 56.2856 | 43.8241 |

这些是用户提供的远端实验事实。如果 GitHub tree 不含原始远端 artifact，请标记为
`[USER_SUPPLIED_EXPERIMENT_FACT]`，不要伪称从 GitHub 独立重建。

允许的当前解释：

- CellCF 比 transition-beta0 低 `0.2145` Avg-mAP；
- transition-beta0 比 uniform 高 `0.4161`，但只有一个 seed；
- CellCF 单点 `@0.7` 较高，不能在一个 seed 和坐标混杂下证明边界机制。

### 3.5 现有 ceiling 与成本工具

请核验：

- 当前 `diagnose_duca_feasible_set_ceiling.py` 是否只在调用者给出的有限候选中选优，
  并明确声明 `not_upper_bound`；
- selection decomposition 中的 GT 方法是否为 heuristic，而非 optimized oracle；
- `4ce69c8` 是否只修复 profiler producer/consumer schema；
- 当前成本协议是否仍把 decoder 与 preprocess 合并，且缺少可支持论文的正式 dense
  full-stack 对照。

---

## 4. 必须完整阅读的代码表面

至少阅读以下文件；若文件在精确提交中不存在，必须明确报告。

### 选择与训练

- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/transition_only.py`
- `opentad/models/duca/counterfactual_utility.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/detectors/single_stage.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/dense_heads/anchor_free_head.py`

### 配置

- `configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`

### 诊断、坐标与成本

- `tools/bata/diagnose_duca_feasible_set_ceiling.py`
- `tools/bata/diagnose_duca_coarse_boundary_ceiling.py`
- `tools/bata/diagnose_duca_selection_decomposition.py`
- `tools/bata/duca_ceiling_utils.py`
- `tools/bata/export_duca_selection_quality.py`
- `tools/bata/analyze_duca_selection_quality.py`
- `tools/bata/duca_full_stack_cost.py`
- `tools/bata/profile_duca_full_stack_cost.py`
- `tools/bata/summarize_duca_cellcf_cost.py`

### 重点测试

- `tests/test_duca_feasible_set_ceiling.py`
- `tests/test_duca_coarse_boundary_ceiling.py`
- `tests/test_duca_selection_decomposition.py`
- `tests/test_duca_local_cell_selection.py`
- `tests/test_duca_local_cell_counterfactual.py`
- `tests/test_duca_cellcf_contract.py`
- `tests/test_c3_physical_grid_round_trip.py`
- `tests/test_duca_full_stack_cost.py`

还必须追踪 dense-grid index、原始帧编号、视频时间戳、valid prefix、短窗口 padding 在数据
pipeline 中的产生和换算位置。不得只读 selector 而跳过数据坐标来源。

---

## 5. 本轮唯一核心问题：如何定义精确可行集合上限

请严格定义并比较以下五个 family。对每个 family 给出集合定义、变量、约束、自由度、
是否包含 exact uniform、是否允许跨区域 quota 转移、最坏物理 gap 和可精确优化的指标。

### A. Exact uniform

当前提交中的 canonical rounded-endpoint exact-uniform。

### B. Current one-per-cell CellCF

每个 exact-uniform cell 恰选一帧。不得偷偷允许跨 cell quota。

### C. Coverage scaffold + adaptive residual

要求：

1. 总预算 exact `K`；
2. scaffold 单独保证指定物理最大间隔；
3. residual 可以在所有非 scaffold 合法位置全局分配；
4. 同一边界邻域可以获得多个 residual；
5. 简单背景区域可以释放 residual quota；
6. exact uniform 必须是该 family 中一个可精确复现的可行解；
7. short window 和 `K_eff=min(K,L)` 必须有明确合同。

此前有人建议 `G=3`、192 scaffold + 192 residual。请严厉判断：

- 这是否只是匹配 CellCF worst-hole 的任意选择；
- scaffold 是否真是最小 cardinality；
- 它是否包含 exact uniform；
- 是否会因固定 scaffold 消耗过多预算；
- 是否应该用 exact-uniform anchor 的一个子集构造 scaffold；
- 是否存在更简洁、自由度更高而仍可证明覆盖的构造。

不要未经论证把 `G=3` 或 `192+192` 定为最终值。

### D. Global exact-K/max-gap variable quota

允许任意位置选择，只要求 exact-K 和最大物理 hole。请判断：

- 它是否严格包含或支配 C；
- 动态规划、min-cost flow、CP-SAT 中哪种求解/解码最合适；
- 如何保证 deterministic tie-break；
- 如何处理每样本不同 `L`、短窗口和 padding；
- 是否会因可行集合过大退化为 noisy global top-k。

### E. Privileged unrestricted GT reference

只作为上限诊断，不得进入部署。必须明确它与历史 label-dependent Oracle、GT remap Oracle
的区别，避免把标签编码进 selected-axis geometry。

---

## 6. 物理坐标与最大间隔必须先冻结

用户此前希望最大选帧间隔可放宽到原始视频的 10 或 15 帧；此前代码又使用 dense-grid
`G=15`。这两个量不一定相同。

请从数据 pipeline 逐行重建：

1. `T=768` 的一个 index 对应什么；
2. 它如何映射到 decoded frame、原始 frame 和秒；
3. 不同视频、FPS、sampling rate、snippet stride 下是否一致；
4. valid mask 是否始终是 contiguous prefix；
5. short/padded window 如何定义 endpoint holes；
6. max-gap 应在什么坐标中成为硬合同；
7. 应如何同时报告 dense-index gap 和 original-frame/time gap。

若无法从代码确定，必须列出缺失 metadata 和最小修复，不得自行假设。

---

## 7. 精确优化问题

请给出一个不会夸大结论的 exact ceiling 设计。

必须区分：

1. 可被 CP-SAT/DP 精确求解的几何目标；
2. metric-wise upper envelope；
3. 一个可复现的 canonical lexicographic oracle；
4. frozen-detector candidate diagnostic；
5. 无法宣称全局最优的 detector mAP。

至少考虑：

- 选中位置二进制变量；
- exact-K；
- endpoint hole 和 interior hole；
- 每个 GT start/end 的 radius 0/1/2/4 coverage；
- both-endpoint coverage；
- selected-to-boundary distance；
- short-action minimum support；
- background selected count；
- deterministic secondary objective；
- solver `OPTIMAL/FEASIBLE/TIMEOUT/INFEASIBLE` 状态。

请判断 CP-SAT 是否必要，还是现有 `T<=768,K<=384` 结构可用更简单的精确 DP、最小费用流
或区间图算法。避免为求解器而求解器，也不要手写一个实际上不精确的 heuristic。

GT 只能用于 train/validation 设计诊断。必须在查看 test-set ceiling/result 前冻结 family、
`G`、目标和 GO/KILL 阈值。

---

## 8. 指标与 GO/KILL 条件

请给出一组最小、不可事后挑选的指标：

- exact-K 合规率；
- 物理 max-gap 合规率；
- radius 0/1/2/4 boundary hit/density；
- any-endpoint 与 both-endpoint coverage；
- selected-to-nearest-boundary distance；
- short/medium/long action 分层；
- background quota 与相对 uniform 的释放量；
- 每个边界附近额外得到的 residual 数；
- family 相对 exact uniform/CellCF 的自由度；
- frozen-detector physical-grid secondary diagnostic；
- coarse/frontend latency、heavy-backbone saving 与 break-even。

请严厉审查数值阈值。不要凭审稿口味任意发明 `+0.5 mAP`、`+10pp r0` 等门槛。
阈值必须来自：

1. 已知 uniform 的饱和程度；
2. family 的数学自由度；
3. 可观测效应大小；
4. 测量噪声与 bootstrap 单位；
5. 后续训练成本。

最后必须给出：

- `GO_TO_CARA_IMPLEMENTATION`
- `HOLD_AND_REVISE_FAMILY`
- `KILL_DUCA_SELECTOR_ROUTE`

三选一的可执行判据。

---

## 9. 要求给出核心实现蓝图

请输出文件级实现方案，但不要把建议冒充当前代码。

至少说明：

1. 应新增还是扩展哪个 ceiling 工具；
2. family 数据结构和统一接口；
3. exact solver 的输入输出；
4. canonical oracle 与 metric-wise envelopes 如何分开；
5. physical coordinate metadata 如何进入 solver；
6. JSON/JSONL 证据 schema；
7. config、commit、dataset split、solver version 和参数如何哈希绑定；
8. 如何 fail closed；
9. 如何复用现有 physical-grid ActionFormer 做 secondary diagnostic；
10. 哪些现有代码应保持不动。

请给出关键 Python 代码，至少覆盖：

- exact-uniform family；
- one-per-cell family；
- coverage-residual family；
- global exact-K/max-gap family；
- physical max-hole validator；
- canonical lexicographic objective；
- solver-status validation；
-结果 schema；
- 不少于 12 个关键单元测试。

代码必须处理：

- `1 <= L <= 768`；
- `0 <= K_eff <= min(384,L)`；
- all-short/mixed-length；
- contiguous 与非法 non-contiguous valid mask；
- equal-score deterministic ties；
- infeasible gap；
- empty GT、重叠 GT、极短 GT、边界在窗口端点；
- CPU/GPU tensor 输入边界；
- 不允许 test GT 泄漏到部署 selector。

---

## 10. 禁止事项

本轮禁止：

- 直接开始 CARA full training；
- 重新训练或重命名已完成 CellCF arms；
- 解锁 Dynamic MUST；
- 加入 X3D、SlowFast、MobileNet；
- 扩展第二 detector 或第二数据集；
- 用 actionness top-k 替代 transition/boundary-first 初心；
- 把 detached counterfactual utility 写成 direct detector gradient；
- 把 physical-grid ActionFormer 写成未经修改的官方 AdaTAD；
- 把有限候选 best-of 称为 exact oracle；
- 用 test GT 调 family、`G`、阈值或损失；
- 因为 Prompt 推荐 C 就默认 C 必须胜出。

如果审查证明 C 不优雅或不包含 exact uniform，应明确否定并给出最小替代 family，但不要
扩展到另一篇完全不同的论文。

---

## 11. 强制输出格式

请严格按以下章节输出：

1. `VISIBILITY_CERTIFICATE`
2. `EXECUTIVE_VERDICT`
3. `VERIFIED_CODE_FACTS`
4. `CORRECTIONS_TO_PROMPT_ASSUMPTIONS`
5. `CELL_FAMILY_MATHEMATICAL_AUDIT`
6. `PHYSICAL_COORDINATE_CONTRACT`
7. `FAMILY_A_TO_E_SPECIFICATION`
8. `SET_INCLUSION_AND_FREEDOM_PROOFS`
9. `EXACT_OPTIMIZATION_DESIGN`
10. `METRICS_AND_PREREGISTERED_GATES`
11. `IMPLEMENTATION_BLUEPRINT`
12. `CORE_CODE`
13. `TEST_MATRIX`
14. `GO_HOLD_KILL_TABLE`
15. `FINAL_NEXT_ACTION`

每条关键结论必须标记为：

- `[CODE_FACT]`
- `[MATHEMATICAL_CONSEQUENCE]`
- `[USER_SUPPLIED_EXPERIMENT_FACT]`
- `[INFERENCE]`
- `[PROPOSAL]`
- `[UNKNOWN]`

最终必须用一句话回答：

> 当前是否应该开始实现并训练 DUCA-CARA，还是应该先修订可行集合，或终止 literal
> pre-backbone selector 路线？

不要给出泛泛建议，不要用“可以尝试更多实验”回避裁决。优先找出一个足以停止错误路线的
反例、集合包含关系错误、坐标错误或成本不可能性，再讨论实现。
