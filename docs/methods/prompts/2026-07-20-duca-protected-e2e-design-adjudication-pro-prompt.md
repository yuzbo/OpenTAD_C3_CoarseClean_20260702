# DUCA 受保护端到端选帧最终设计裁决 Prompt

你现在不是开放式头脑风暴顾问，而是负责给出一次性设计裁决的 CCF-A/CVPR 级时序动作检测方法审稿人、PyTorch 自动微分专家和实验协议审计员。

本轮只允许审查并裁决一个问题：

> 在离线时序动作检测中，如何让官方 AdaTAD/ActionFormer 检测分类与回归损失真实影响 pre-backbone 选帧策略，同时保护动作二分类粗分类器的语义，并在严格固定预算与物理覆盖约束下产生真实硬帧？

禁止继续扩展新方向。禁止引入 X3D、SlowFast、MobileNet、动态预算、强化学习、新检测头、新数据集或新的多阶段流水线。本轮输出必须收敛到一个可实现方案，或明确判定当前设想不可实现。

## 1. 可见性与精确审查对象

首先给出 `VISIBILITY_CERTIFICATE`：

1. 是否能打开仓库；
2. 是否能定位以下精确提交；
3. 实际读取到的提交哈希；
4. 实际逐行读取的文件；
5. 若不能读取精确代码，输出 `VISIBILITY_BLOCKED` 并停止，不得依赖本 Prompt 的叙述假装完成代码审查。

仓库：

- `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

当前同步分支：

- `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-protected-e2e-20260720`

本轮代码基线：

- 最新诊断/协议提交：`db11aeeab464b655104d1265e4d166bb5976c569`
- Allocation-Ceiling 实现提交：`8ebdd2a11ea5cc0644979324872a3b1cae5a2170`
- CellCF 证据提交：`4ce69c852bdbd902046b47bc6019ae11e850dbe4`
- 正式 CellCF 模型训练提交：`1642f265e48391418a7c8a4a087e33e2b7bf6899`

`db11aee` 相对 `8ebdd2a` 仅修订最终裁决口径。不得把文档提交误称为新模型实现。

任务必须始终称为“离线时序动作检测”，不得称为 Online TAD。

## 2. 必须读取的代码

至少逐行检查：

- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py`
- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/transition_only.py`
- `opentad/models/selectors/duca_online_frame_selector.py`
- `opentad/models/detectors/actionformer.py`
- `opentad/models/dense_heads/actionformer_head.py`
- `opentad/models/dense_heads/anchor_free_head.py`
- `opentad/models/projections/actionformer_proj.py`
- `opentad/models/utils/temporal_grid.py`
- `opentad/models/utils/sampling_contract.py`
- `tests/test_duca_transition_only.py`
- `tests/test_duca_cellcf_contract.py`
- `tests/test_duca_detector_gradient_bridge.py`
- `tools/bata/run_duca_official_adatad_one_step_grad_proof.py`

同时读取 Allocation-Ceiling 结果与当前决策边界：

- `docs/methods/prompts/2026-07-20-duca-allocation-negative-result-detector-utility-pro-audit-prompt.md`
- `tools/bata/duca_allocation_families.py`
- `tools/bata/duca_exact_physical_solver.py`
- `tools/bata/evaluate_duca_allocation_candidates.py`

每个关键判断必须给出文件、符号和具体行号。

## 3. 已冻结实验事实

以下是历史证据，不得篡改或混用协议：

- exact-uniform：终点 EMA Avg-mAP `63.8594`
- transition-beta0：`64.2755`
- CellCF：`64.0610`
- CellCF 不能跨均匀小格转移预算，并未超过 transition-beta0。
- 当前正式 CellCF 配置中：
  - `detector_gradient_bridge="disabled"`
  - `detector_gradient_final_weight=0.0`
  - `detector_gradient_mode="none"`
  - 反事实检测效用是 `detach()` 后的教师监督，不是直接检测梯度。
- 当前策略路径对 `actionness_logits` 和策略描述子执行梯度隔离。
- 旧 `structured_zero_forward` 局部代理未通过真实硬换帧方向对齐，禁止仅重新打开权重。
- Allocation-Ceiling 在训练侧显示有限 GT 边界几何空间，但当前可部署 transition score 的边界覆盖差于均匀采样，冻结检测损失也更差。
- 冻结检测损失不是最终 mAP。当前 Allocation-Ceiling 只能记为诊断。
- 当前没有新的受保护端到端模型实现，也没有对应训练结果。

## 4. 唯一候选设计

请只审查以下候选，不得发散出第二条主路线。

### 4.1 粗分类与梯度所有权

低分辨率完整时间序列经过当前官方 ASFormer 粗分类前端，得到：

- 动作二分类 logits `a_t`
- 粗分类隐藏特征 `z_t`

动作二分类头只接受动作/背景二分类监督。

初始受保护版本规定：

```python
a_policy = a.detach()
z_policy = z.detach()
```

选择适配器和转变评分器读取 `a_policy`、`z_policy` 的状态变化、不确定性和隐藏特征差分。官方检测损失只允许更新选择适配器和选择评分器，不允许直接更新动作二分类头或 ASFormer 共享干。

可选的协同消融允许：

```python
z_policy = z.detach() + rho * (z - z.detach())
```

但 `rho>0` 只能作为第四实验臂，不得默认进入主版本。

请裁决：

1. 该梯度所有权是否仍可诚实称为“端到端选择策略学习”；
2. 是否足以保护动作二分类语义；
3. 是否需要增加一个轻量 selector adapter；
4. 哪些参数应该分别由动作损失、转变损失和检测损失更新；
5. 当前代码中必须保留和必须删除的 `detach()` 位点。

### 4.2 全局覆盖约束的固定预算选择

固定 `K=384`。选择器输出全局分数 `s_t`，构造带均匀覆盖底座的选择分布：

```text
p_t = (1-lambda) * softmax(s_t / tau) + lambda / T
```

目的不是固定每格一帧，而是在保证全局覆盖的同时允许剩余预算跨区域向状态转变和语义边界集中。

请裁决：

1. 如何在离散物理时间上严格实现 exact-K、位置唯一、有序和最大间隔；
2. `lambda` 只能作为可行域参数还是也可学习；
3. 如何避免多个分位点落到同一帧；
4. 最大间隔应使用候选索引、解码原始帧还是秒，并如何在三种单位间审计；
5. 硬约束、软覆盖损失和 fail-closed 验证各自承担什么职责；
6. 该方案是否真的允许跨区域预算转移，且将精确均匀采样包含为显式可行成员。

### 4.3 硬前向、软反向检测梯度桥

目标前向必须只消费真实选中的硬帧：

```python
hard = gather(dense_video, hard_positions)
soft = differentiable_resample(dense_video, soft_assignment)
selected = soft + (hard - soft).detach()
```

`selected` 的前向数值必须逐元素等于 `hard`，检测分类和回归损失通过 `soft` 的梯度更新选择评分器。

请在以下两者中只裁决一个实现：

1. 修订现有 `soft_to_hard_resample`，建立与全局 exact-K/覆盖可行域一致的软分配；
2. 新建单调分位传输或受约束软分配，但必须复用当前张量、网格和正式 AdaTAD 路径，不得重造检测器。

禁止采用已被证伪的 `structured_zero_forward` 原样复活。必须说明训练软路径与推理硬路径的偏差如何测量，以及为什么新梯度代理比旧代理更可信。

### 4.4 官方检测器与时间坐标

必须追踪真实调用链：

```text
低分辨率粗分类
-> 选择器
-> 真实硬帧
-> 官方 AdaTAD 视频骨干/适配器
-> ActionFormer projection/neck/head
-> 官方分类与回归损失
```

请裁决：

1. 当前后端是否确实构建并运行完整官方 AdaTAD/ActionFormer 组件；
2. 非均匀真实帧被压到 selected axis 后是否产生时间几何扭曲；
3. P0-P3 是否可以继续使用当前 selected-axis remap 来隔离梯度桥变量；
4. 若其会使命题无效，必须给出最小 physical-time 修复，而不是新建检测器；
5. 训练目标、输出坐标、NMS 和最终 mAP 是否处于同一时间语义。

## 5. P0-P3 强制门禁

在任何 official-60 训练前，给出可直接实现的门禁规范。

### P0 协议冻结

必须冻结：

- 离线 TAD；
- 官方 THUMOS 训练/测试划分，禁止 val/test 共用同一 subset；
- `K=384`；
- 物理最大间隔单位及转换；
- official 60 epoch；
- 终点 EMA 为主结果，不用中间验证选择 checkpoint；
- 完整系统成本口径；
- 推理 teacher-free、GT-free、cache-free。

### P1 前向与优化器合同

必须测试：

- exact-K、唯一、有序、合法短视频退化；
- 最大物理间隔；
- `selected` 前向逐元素等于真实 hard gather；
- detector 只消费 hard forward；
- 所有可训练 selector 参数进入 optimizer；
- 官方 AdaTAD 主干、projection、neck、head 未被静默替换；
- 训练与推理的硬选择同构。

### P2 梯度所有权

在同一个真实官方配置和同一个 batch 上分别反传：

1. 仅 `cls_loss + reg_loss`
2. 仅动作二分类损失
3. 仅转变/边界辅助损失

主版本必须满足：

| 仅反传的损失 | AdaTAD | selector adapter/scorer | ASFormer trunk | action head |
|---|---:|---:|---:|---:|
| detector | 非零 | 非零 | 零 | 零 |
| action BCE | 零 | 零 | 非零 | 非零 |
| transition auxiliary | 零 | 非零 | 非零或按裁决冻结 | 零 |

不得用总损失一次 backward 后“大家都有梯度”代替来源归因。

### P3 软梯度与真实硬决策对齐

必须设计真实硬换帧有限差分门禁：

- 在固定 batch、固定模型状态、固定 RNG 下；
- 根据软梯度提出合法单帧或少量帧移动；
- 实际重新执行 hard gather 和官方 detector loss；
- 报告梯度预测方向与真实 hard loss 变化的符号一致率、秩相关、有效非零样本数和置信区间；
- 报告不同动作长度、边界距离和窗口类型；
- 全零、单值、不可行候选不得计为通过；
- 未通过时必须 `STOP`，不得进入 official-60。

请给出不依赖事后挑阈值的预注册通过规则。

## 6. 通过后唯一允许的 official-60 四臂

所有实验必须同提交、同数据、同 seed、同 K、同 successful optimizer updates、同 EMA、同后处理和同最终评测：

1. `exact_uniform`
2. `transition_no_bridge`
3. `protected_e2e`
4. `protected_e2e_rho`，只允许极小固定 `rho` 进入 ASFormer 最后一个时序块

主指标只能是终点 EMA 的官方 THUMOS Avg-mAP 和 mAP@0.3/0.4/0.5/0.6/0.7。检测损失、边界覆盖和动作二分类指标只能解释机制。

裁决规则：

- `protected_e2e <= transition_no_bridge`：直接检测梯度假设失败；
- `protected_e2e > transition_no_bridge` 但 `< exact_uniform`：梯度有作用，但选择方法尚不具备论文竞争力；
- `protected_e2e > transition_no_bridge` 且 `> exact_uniform`：才允许三种子、预算曲线和泛化；
- `protected_e2e_rho` 若损害动作二分类或最终 mAP，不得进入主方法。

禁止在四臂完成前运行动态预算、其他 K、其他检测器、其他数据集或新选择策略。

## 7. 强制输出格式

严格按以下顺序回答：

1. `VISIBILITY_CERTIFICATE`
2. `ONE_SENTENCE_VERDICT`
3. `CURRENT_CODE_GRADIENT_AUDIT`
4. `DESIGN_VERDICT: GO / REVISE / KILL`
5. `SINGLE_APPROVED_ARCHITECTURE`
6. `EXACT_GRADIENT_OWNERSHIP_MATRIX`
7. `P0_PROTOCOL_FREEZE`
8. `P1_IMPLEMENTATION_PATCH_PLAN`
9. `P2_GRADIENT_TESTS`
10. `P3_HARD_SOFT_ALIGNMENT_GATE`
11. `OFFICIAL60_FOUR_ARM_PREREGISTRATION`
12. `BLOCKERS_BEFORE_IMPLEMENTATION`
13. `BLOCKERS_BEFORE_TRAINING`
14. `FINAL_GO_NO_GO`

实现计划必须给出：

- 修改文件；
- 具体类/函数；
- 核心张量形状；
- 关键 PyTorch 代码或伪代码；
- 必须新增的测试；
- 哪些现有代码复用；
- 哪些旧桥梁禁止复用；
- 失败后停止条件。

不得以“可以尝试”“建议调参”“以后可研究”结束。只能输出一个主设计和明确执行顺序。
