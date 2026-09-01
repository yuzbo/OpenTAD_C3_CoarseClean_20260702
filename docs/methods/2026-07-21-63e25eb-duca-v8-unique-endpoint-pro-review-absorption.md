# DUCA V8 `63e25eb` Pro 审查吸收与独立复核

## 来源绑定

- 审查对象：`yuzbo/OpenTAD_C3_CoarseClean_20260702`，精确提交
  `63e25eb17e523d369f73434ed4d9b6446608861a`。
- 原始附件：
  `C:/Users/skywalker/.codex/attachments/e62d0b32-a5d2-4b44-bc30-6c43ec3f8d0c/pasted-text.txt`。
- 字节一致归档：
  `docs/methods/reviews/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-raw.txt`。
- SHA-256：
  `DF19960D0B3158CE7F31E0FE4A92F8CD22C7B2AAFD5FB78D13E91DDACEA8EC70`；
  `46,310` bytes，`1,126` 行。

## 总体裁决

项目吸收结论是：

> **实质认可代码诊断，但不完全认可首选修复；状态为
> `SUBSTANTIAL_ACCEPT_DIAGNOSIS / REVISE_OBJECTIVE_BEFORE_IMPLEMENTATION`。**

审查正确识别了当前 V8 的主要监督语义缺口、detector bridge 的 surrogate
边界以及 formal gate 的缺证据问题。但它提出的 `radius=1`
exact event coverage 在当前 `max_unselected_hole=2` 合同下对绝大多数端点是
恒等事件，不能产生有效梯度，因此不得原样实现。

## 已由精确代码确认的事实

1. `transition_only` 分支用 `sigma=2`、截断半径 `4` 的 Gaussian
   `transition_target`，并把同一个张量直接赋给 `boundary_target`。
2. active boundary loss 实际调用
   `local_boundary_mass_coverage_loss(soft_occupancy, transition_target, ...)`；
   其核心为 `exp(-neighborhood_mass)`，在同一边界邻域加入第二、第三个
   occupancy 仍会继续降低损失。
3. 仓库已有 `structured_local_coverage_probability`，它按同一
   exact-K/max-hole 配分函数计算事件至少命中一次的概率，但当前 V8 active loss
   未使用它。
4. hard Viterbi 与 soft forward-backward 复用同一 exact-K/max-hole family，且
   `max_gap_repair.enabled=False`；没有证据把当前问题归因于 post-hoc repair
   覆盖 hard/soft 不一致。
5. G1/G2 的 detector-to-scorer 路径是 hard-forward、soft-backward 的局部
   structured transport surrogate，不是 hard index 的直接导数。
6. `hard_soft_alignment.py` 已定义预注册统计门，但当前串行 gate 只运行单元测试和
   full-model plumbing gate，随后直接写入 `formal_training_unlocked=True`；没有生成
   current bridge 的真实 legal hard-swap alignment artifact。
7. `duca_protected_e2e_fixed384_official60.py` 的
   `detector_gradient_is_direct=True` 与真实实现不一致，必须改为 surrogate 口径。
8. schedule 使用已完成 update 计数，却以 `step <= warmup` 判断，存在一拍偏移；
   这是确定性合同问题，不是当前低 mAP 的已证实主因。
9. G2 companion 替换 hard positions、slot assignment 和 detector input；它没有替换
   transition auxiliary 使用的 `soft_coverage`。因此它是 detector-input/bridge
   companion，不是完整 selector-training companion。
10. U 仍执行 coarse probe/scorer，是 compute-matched control；部署成本表还需要
    单独的 `U-bypass`，且 selector-only profile 不能代表全栈成本。

## 对根因推断的态度

- “Gaussian target + occupancy mass 导致宽边界带重复聚集”与已有 adjacency、
  endpoint-recall 现象方向一致，属于**高可信假设**，尚不是因果证明。必须通过单变量
  objective 对照后才能升级为实验事实。
- 当前 V8 P0 的 BCE 接近 `ln 2` 只能说明早期分离度有限，不能在 holdout 导出前
  直接判定本轮 coarse branch 失败。历史/局部路线的 AUROC 不得冒充当前 P0 终局指标。
- V5 learned arms 全部低于 matched U 是真实负证据，但不能单独证明 V8 的 P0 冻结和
  scorer-only gradient 一定失败。

## 对首选修复的关键否决

审查定义端点事件 `E_i={t: |t-c_i|<=1}`，即内部端点对应三个连续位置；同时固定
`max_unselected_hole=2`。这两个合同合在一起意味着：

```text
任意三个连续位置不可能全部未选
=> 对绝大多数内部端点，P(S intersects E_i) = 1
=> -log P = 0，且对 scorer 的梯度为 0
```

只有紧贴有效序列首尾、事件被截成少于三个位置的极少数端点可能非平凡。因此
`16 * exact-cover(radius=1)` 不能成为当前 G=2 下的主要边界监督。审查本身同时承认
radius-1 coverage 近乎饱和，这与其首选公式互相矛盾。

即使保留它提出的 event-distribution loss，事件内部的分数平台仍可能让结构化解码
在同一边界附近选择多个位置；“唯一事件”必须通过真实 set-level marginal gain、
去重或精确 `radius=0` 命中目标来证明，而不能只靠名称成立。

## 修订后的有界下一步

1. 不修改正在运行的 `1178989`，其 P0/U/G0/G1/G2 只能按冻结提交作为诊断证据；
   在 objective 与 alignment 缺口关闭前，不把 P0 winner 解释为论文主方法解锁。
2. 先增加一个小规模 brute-force/解析门禁，证明 `G=2,r=1` 的 exact event coverage
   对内部端点恒为 1、梯度为 0，防止照抄错误公式。
3. 下一版 objective 必须采用非平凡事件，首选候选是 rounded endpoint 的
   `radius=0` unique-event probability，或明确给出在 G=2 下仍有 headroom 的事件定义；
   event-distribution 与 structured coverage 要分别验证梯度和重复聚集行为。
4. objective 修复必须是单变量：不同时改 coarse probe、scorer、K、G、DP、
   VideoMAE/AdaTAD、loss 权重或训练长度。先以 train-only holdout 证明 scorer 至少不劣于
   `abs(delta p)`，并改善 exact-endpoint/短动作/adjacency 指标。
5. G1/G2 前必须运行真实 legal hard-swap alignment gate；失败时只允许 U/G0，不能声称
   detector-aligned gradient。
6. 独立修复 `detector_gradient_is_direct` 元数据和 schedule 一拍偏移；这两项不应包装为
   新模型贡献。
7. 终局仍由 matched terminal-EMA mAP 裁决。若 corrected G0 不超过 U，则停止在
   `K=384,G=2` 上继续调 scorer/bridge/curriculum；这不等于否定所有时序智能采样或
   后续 coarse-context fusion 路线。

## Oracle 边界聚集语义修正

“unique endpoint”只能作为防止漏掉某个起点或终点的锚点约束，不能被解释成
“每个边界只允许选一帧”或“覆盖后禁止继续聚集”。历史 GT Oracle 对每个动作起点和
终点优先选择中心及其前后半径 2 的多个位置，然后才用均匀点补足剩余预算；其优势
正包含边界附近的局部密集观测。

当前应抑制的是无中心、无左右结构、无每边界配额上限的宽带 mass 堆积，而不是
oracle-like boundary burst。后续目标应拆成三个语义互补的部分：每个端点的精确锚定、
端点附近左右两侧的有界多帧聚集，以及剩余预算的全局覆盖。局部聚集奖励必须在达到
预注册配额后饱和，并对相邻/重叠端点去重；其半径、配额和左右分布应先由 Oracle
选帧统计固定，再做单变量验证。`radius=0` exact event 因而只是 anchor 项候选，不能
单独代表最终边界分配目标。

## 状态边界

- 外部审查：`reviewed`。
- 代码事实复核：`verified_static_exact_commit`。
- unique-endpoint 修订：`discussed_needs_mathematical_revision`。
- 当前 V8：`experiment_running`，无 P0 winner、无 V8 terminal mAP。
- 论文主张：仍未达到 `empirically_supported` 或 `paper_ready`。
