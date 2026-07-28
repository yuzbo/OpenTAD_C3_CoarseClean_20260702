# H-RIME v1 预算守恒分层采样设计

**日期**：2026-07-28

**状态**：`user_approved / designed / implementation_start_authorized`

**证据状态**：尚未 `implemented`、`tested`、`empirically_supported` 或
`paper_ready`

## 1. 裁决

本设计吸收外部 Pro 接管报告的主路线，但不是逐字接受。

最终裁决为 **CONDITIONAL ACCEPT WITH CORRECTIONS**：

- 接受唯一主架构：共享全视频廉价扫描 → 视频总预算规划 →
  多重选择背包精确分配窗口预算 → 复用现有 exact-K 窗内位置选择 →
  同 K 分桶执行 → 保持 AdaTAD 检测头与 NMS 不变。
- 接受执行顺序：先修复旧四阶段基础设施，再做同总成本 oracle，只有
  oracle 通过后才训练 H-RIME，正式集继续封存。
- 不把报告中的 `0.5 mAP`、`0.2 margin`、`15% savings`、`ECE 0.05`、
  `25% overhead` 等数值当作已验证事实或自动生效的论文门槛。
- 修正不可达预算、重叠窗口成本、代理效用与官方 NMS 的偏差、统计端点、
  浮点确定性和风险校准等未闭环问题。
- 报告所引用但未出现在仓库中的 `sandbox:/mnt/data/...` 补丁、哈希和
  synthetic test 计数均只属于 `PARTNER_CLAIM`，不会被当作本仓库实现。

本设计已经得到用户对主路线和实施的明确批准，因此可进入实现；任何大规模
训练和正式集评估仍受本文的实验门禁约束。

## 2. 研究问题与论文定位

研究问题是：

> 对一个完整离线视频，能否用一次低成本扫描预测长度归一化的总重计算预算，
> 再把该预算联合分配给视频所属的重叠 768-candidate AdaTAD 窗口，并在每个
> 窗口中选择 exact-K 物理位置，从而在真实总成本下降时保护高 IoU 与短动作
> 定位？

方法定位是 **offline TAD pre-backbone acquisition plugin**，不是 Online
TAD。论文贡献不能写成“首次按视频分配 token/帧数”，因为 AdapTok 和
EVATok 已经分别展示了自适应 token 数量或按视频分配/路由的思想。可辩护的
目标贡献是：

1. 面向 TAD 定位风险而不是视频生成重建质量的全视频预算规划；
2. 在重叠滑窗、离散可达 K 和 exact physical-time remapping 下的预算守恒
   联合分配；
3. 对高 IoU、短动作与边界覆盖进行显式保护，并用官方检测器/NMS 复算验证；
4. 以同 realized-cost 控制和不可篡改收据证明真实重计算、时延与准确率关系。

## 3. 决策、执行、统计和成本单位

必须永久区分以下单位：

1. **规划单位：完整视频。** 视频规划器只输出一个总配额，而不是给每个窗口
   复制同一个 K。
2. **分配单位：同一视频的全部窗口。** 联合求解每个窗口的 `K_vw`。
3. **位置选择和检测单位：一个 768-candidate 窗口。** 现有 exact-K 选择器、
   AdaTAD adapter/head、物理时间映射和 NMS 保持窗口级语义。
4. **统计单位：视频。** cross-fit、校准、bootstrap 和 paired test 不能把
   同一视频的窗口当独立样本。
5. **默认成本单位：重执行总数。** 在没有真实跨窗 cache 时，
   `E_v = sum_w K_vw`；重叠区域在两个窗口中被重算就计两次。
6. **诊断单位：唯一物理帧并集。** `U_v` 与 `E_v/U_v` 只描述重复度，不可
   冒充实际计算节省。

## 4. 不可破坏的模型与证据约束

- 外部 detector grid 仍为 768 个候选位置。
- nominal budget panel 为 `(192, 256, 384, 512)`，量子 `q=16`。
- 重 VideoMAE 输入长度必须等于 ledger 中的 `effective_k`，不允许补到
  Kmax 后宣称节省。
- ActionFormer/TriDet 的 backbone 后续投影、temporal adapter、head、loss、
  decoding 和 NMS 保持注册版本。
- 推理决策只使用 inference-visible cheap evidence。GT、teacher、验证/测试
  标签、raw-prediction cache 和 counterfactual ledger 禁止进入决策。
- 所有输出先恢复到物理时间，再由原官方 evaluator/NMS 处理。
- 正式集在完整 development receipt 放行前保持封存。
- partial、training-domain、single-seed、intermediate、unmatched 或
  missing-receipt 数字只能是 `ENGINEERING_STATUS`，不得进入论文论证。

## 5. 离散可达预算语义

### 5.1 窗口有效 K

对有效长度 `L_vw`、nominal K `k` 和量子 `q=16`，定义

`e(k, L_vw) = q * floor(min(k, L_vw) / q)`。

若 `e=0`，该窗口/样本必须 fail closed；不得构造虚假重输入。每个窗口先把
nominal panel 映射到 effective K，再去重。多个 nominal K 映射到同一个
effective K 时，使用最小 nominal K 作为 canonical label。

例如 `L_vw=231` 时：

`(192, 256, 384, 512) -> (192, 224, 224, 224)`，

所以可行集合是 `{192, 224}`，`224` 的 canonical nominal label 是 `256`。
这修复旧 Phase-1 中“ledger 记 231、重 backbone 实际吃 384”的错误。

### 5.2 视频总预算的可达投影

令每个窗口的去重可行集合为 `F_vw`，视频可达总成本集合为

`C_v = {sum_w k_w | k_w in F_vw}`。

规划器先产生 raw cap `B_v^raw`。定义

`B_v^reach = max {c in C_v | c <= B_v^raw}`。

若 raw cap 低于所有窗口最小 K 的总和，样本必须 fail closed；不得静默提高
预算。正式账本同时记录：

- `raw_budget_cap`
- `reachable_budget`
- `realized_budget`
- `projection_unused_budget = raw_budget_cap - reachable_budget`
- `solver_unused_budget = reachable_budget - realized_budget`
- `budget_feasible`

v1 的预算守恒模式要求 `realized_budget == reachable_budget`。若求解器不能
精确填满已知可达目标，则 fail closed。由不可达 raw cap 产生的 projection
余量必须单独报告，不能被隐藏成计算收益。

所有 same-total-cost 对照使用 **effective K 的 realized total**，不能用
nominal K、平均 K 或 raw cap。

## 6. H-RIME v1 架构

### 6.1 共享全视频廉价扫描

数据层先形成一个 `VideoWindowGroup`，包含稳定排序的窗口、有效长度、物理
起点、视频标识和只读 cheap feature 索引。廉价扫描对完整视频执行一次，输出：

- 视频摘要 `h_v`
- 与每个窗口对齐的局部摘要 `h_vw`
- 扫描版本、输入身份和排序哈希

共享扫描必须是真实复用；若实现仍逐窗重复计算，就必须按实际执行计费并标注
为 fallback，不能称 shared scan savings。

### 6.2 VideoBudgetPlanner

视频规划器在长度/窗口机会归一化的密度面板

`rho in {0.00, 0.25, 0.50, 0.75, 1.00}`

上预测效用与风险曲线。面板点映射到该视频的 feasible minimum/maximum total
cost 区间，再投影到 `C_v`，而不是把一个固定 raw K 用于所有视频。

网络结构通过累计非负增量约束预测曲线的单调性；这只证明**模型输出结构**
单调，不证明真实风险随预算单调。真实校准仍需独立验证。

预算价格、温度、风险权重和任何保守裕量只允许在 training/calibration video
groups 上拟合并冻结。

### 6.3 WindowOptionHead

对每个窗口和每个去重 effective-K 选项预测：

- 定位效用 `u_vw(k)`
- 高 IoU/短动作/边界失败风险 `r_vw(k)`

曲线使用单调参数化，并输出 canonical nominal/effective 对应表。窗口头不能
看到其他窗口的 GT、teacher 或 counterfactual 结果；训练目标只可由
video-grouped cross-fit 生成。

### 6.4 Exact MCKP Allocator

分配器求解

`max sum_w [u_vw(k_w) - beta * r_vw(k_w)]`

subject to

`k_w in F_vw` and `sum_w k_w = B_v^reach`。

v1 使用确定性精确 multiple-choice knapsack dynamic programming：

- 不使用 Gumbel、straight-through、RL 或可微近似；
- solver stop-gradient；
- 固定整数化分值尺度、数值 dtype、实现版本和 tie-break；
- tie-break 顺序为：更高目标值 → 更低累计风险 → 按窗口顺序字典序更小的
  effective-K assignment；
- 收据记录 solver version、输入 hash 和 assignment hash。

分数整数化误差必须小于 manifest 注册容差；同一输入重复求解必须 bit-exact。

### 6.5 复用现有 exact-K selector

分配得到 `K_vw` 后，直接调用现有 `decode_rime_exact_k` 和
`DucaRimeFrameSelector`。不得复制第二套 selector 或改变物理时间重映射。
推理 replay 仍以 `(video_id, window_start_frame)` 为键，新增视频级分配收据
只负责产生这张 exact replay map。

### 6.6 Homogeneous-K dispatch

同一视频或 batch 内的窗口按 effective K 稳定分桶。每个桶以真实 K 形状调用
重 backbone；桶内恢复原窗口顺序后进入原检测合并/NMS。空桶不执行，短窗口
不补 inactive tail。

## 7. 训练与推理路径

### 7.1 训练

v1 采用 factorized training：

1. 使用注册的 mixed-K detector 产生可比较窗口执行能力；
2. 在 training-role 视频上形成 cross-fitted counterfactual targets；
3. 训练 VideoBudgetPlanner 和 WindowOptionHead；
4. 在 calibration-role 视频上冻结 price、beta、温度、置信界和阈值；
5. exact solver 不反传梯度；
6. development-role 视频只做完整矩阵评估，不重新调参。

训练、校准和开发角色按视频互斥。若历史 checkpoint 无法证明某视频未参与
训练，其结果只能作为工程 sanity，不可用于 oracle 或论文结论。

### 7.2 推理

推理为确定性两遍：

1. cheap full-video scan；
2. 预算规划与 exact MCKP；
3. 生成 hash-bound per-window effective-K replay；
4. homogeneous-K heavy dispatch；
5. 原 AdaTAD 窗口检测、物理时间重映射、合并和 NMS；
6. 输出视频级成本、风险和 provenance ledger。

任何输入身份、排序哈希、budget protocol 或 solver hash 不一致都必须 fail
closed。

## 8. 四个实施/实验阶段

这些阶段不同于已经失败的旧 Phase 1–4 DAG，不得混写。

### Stage 0 — 基础设施恢复

目标：

- 修复短窗口 true effective-K 执行与账本；
- 修复 compactor 的 clean-cwd 模块导入；
- 用新的 hash-bound salvage transaction 处理两个 raw epoch-59 checkpoint；
- 原失败根保持不可变；
- 旧 Phase 4 controller 默认禁用，只有验证过的 development release 才能
  显式开启；
- 重新完成 window-local Phase-1 closure，恢复必要 baseline DAG。

已核验待恢复输入：

| Backend | Source job | Raw checkpoint SHA-256 |
|---|---:|---|
| ActionFormer | `1198115` | `cd92f3d499360c834f7ddd6ccfd5cba172c870bf6922de566b2b7e3878680e11` |
| TriDet | `1198116` | `8940dbe756e8abfa3f7c8b042f3c658b26898d5c805d2876011a4e7510d11e12` |

两份 checkpoint 内部都没有完整的 commit/variant/seed 审计元数据；salvage
receipt 必须诚实地区分“checkpoint 内嵌事实”和“由旧 immutable manifest/log
外部绑定的 provenance”。当前无法取得可信 GPU energy 字段，所以不作能耗
声明。

### Stage 1 — 同总成本 allocation oracle

在真正 held-out 的 development videos 上比较：

1. uniform per-window allocation；
2. independent-window RIME allocation；
3. joint whole-video allocation；
4. 必要的 allocation shuffle/null controls。

所有策略使用完全相同的 per-video realized effective-K total、同 detector
checkpoint、同位置选择规则、同 evaluator 和同 NMS。oracle 先用窗口级
counterfactual/replay 搜索分配，再必须对选中分配执行完整官方 prediction
merge/NMS/evaluator replay。

加性窗口效用只是求解 surrogate。必须报告其与官方视频级 delta 的
rank correlation、sign agreement 和 worst-case error。若 surrogate 排名
不能稳定预测官方结果，joint oracle 不通过，即使代理目标看似提高。

### Stage 2 — Factorized H-RIME development

只有 Stage 1 通过才允许大规模训练。比较：

- best fixed/uniform baseline；
- exact U-same-total-K；
- independent-window RIME；
- H-RIME full；
- no-video-prior、no-risk、shuffle-allocation、uniform-allocation 等归因臂；
- ActionFormer 为主，TriDet 做迁移；
- 注册的 budget-density panel 和 seeds。

开发矩阵必须同时产出准确率、高 IoU、短动作、校准、真实时延、显存、扫描
overhead、duplicate ratio、不可达预算率和完整 provenance。

### Stage 3 — Publication admission

只有完整 development receipt 明确授权后才能冻结候选、预注册正式矩阵并
打开 official-final。正式结果必须覆盖注册 backend/budget/seed 单元、paired
video bootstrap、同成本控制和 full-stack latency。任何单元缺失都不能写成
完整论文结论。

## 9. 实验门槛冻结规则

结构性门槛立即冻结：

- exact-K/no-padding、坐标、无泄漏、排序、求解可行性和 hash 绑定必须全过；
- same-total-cost 必须以 realized effective K 精确相等；
- Stage 0/1/2/3 不得越级；
- official-final 必须保持封存直到 development authorization。

效果数值不沿用 Pro 报告的建议常数。冻结流程是：

1. 在不含 development/final 的 training/calibration roles 上重复测量 detector
   与 profiler 噪声；
2. manifest 显式给出一个 primary endpoint、方向、非劣界、最小实际重要
   差异、family-wise multiplicity 规则和 registered seeds；
3. 同一 manifest 同时注册 high-IoU、short-action、cost 和 calibration
   guardrails；
4. manifest 在读取完整 development matrix 前写入 commit/hash；
5. 无 manifest 或事后改值时 fail closed。

三个 seed 只能提供方向性重复，不足以声称可靠估计正态方差。必须逐 seed
报告，并以视频 cluster bootstrap 为主。不得采用“任意一个 endpoint 显著、
另外两个为正”这种可选择端点的规则。

## 10. 校准与风险报告

ECE 不是唯一门槛。至少报告：

- Brier score；
- reliability diagram；
- risk-coverage curve；
- coverage/violation rate；
- duration、action density、window count 等预注册 worst-group 指标；
- 预算增加时真实风险的 empirical monotonicity violations。

模型输出的单调结构不能替代这些验证。

## 11. v1 明确排除

以下内容不进入 v1 主线：

- CBCG learned edge head；
- cross-window heavy-feature cache；
- pairwise chain DP 或高阶窗口交互；
- two-round sequential acquisition；
- end-to-end detector/planner 联合微调；
- Gumbel/ST/RL allocator；
- 把能耗或唯一帧并集当作未经测量的成本收益。

它们只有在 v1 oracle、校准、因果和 full-stack cost 证据闭环后才可成为独立
extension。

## 12. 实现边界

允许新增或修改的最小表面：

- `opentad/models/duca/hrime.py`：可达预算、planner 输出协议、exact MCKP；
- dataset/collate/sampler 的 video-grouped 只读适配层；
- SingleStageDetector 外部的两遍 planner/dispatch adapter；
- 复用现有 selector/replay/ledger 的接口；
- Stage-0 compactor/salvage/launcher 修复；
- oracle、ledger、validator 和 focused tests；
- 当前 research wiki、design、implementation plan。

不建设通用工作流平台，不复制 selector，不改变 AdaTAD 检测语义，不把旧日志、
checkpoint 或服务器产物提交进仓库。

## 13. 完成定义

“开始实现”只表示进入 `implementation_started`。只有满足下列条件才能提升
状态：

- `implemented`：对应模块、工具和启动器已存在且接口闭合；
- `tested`：focused local/remote checks 与 precheck 全部通过；
- `experiment_running`：新的 immutable Slurm transaction 已提交并有身份收据；
- `empirically_supported`：完整 development experiment 通过预注册门槛；
- `paper_ready`：正式矩阵、统计、成本和 provenance 全部闭环。

在此之前，唯一正确的性能结论是：

`No paper-admissible empirical conclusion is available yet.`
