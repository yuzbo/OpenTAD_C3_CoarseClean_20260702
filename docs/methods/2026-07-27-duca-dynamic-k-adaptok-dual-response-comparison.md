# DUCA 动态 K / AdapTok 两份接管回复的完整吸收、比较与项目裁决

## 1. 原始记录

本文件比较并吸收两份独立回复。两份原文均已字节一致归档；原文是完整记录，
本文件是项目层面的结构化裁决。

### 回复 A：DUCA-METER / METER-TAD

- 原附件：
  `C:/Users/skywalker/.codex/attachments/8dd661a0-1596-4394-ba09-e293fb3c9169/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-a-2032fca-raw.txt`
- 大小：`96,650` bytes
- 物理行数：`2,694`
- SHA-256：
  `2032fcaeddbd4f758ac1be024dd3f867e8dbc6baacd9955de40241ce35595127`
- `cmp` 校验：`byte_identical=true`

### 回复 B：MERTAD

- 原附件：
  `C:/Users/skywalker/.codex/attachments/38deddb7-5b11-45e5-9f30-e8ecfe25a557/pasted-text.txt`
- 字节一致归档：
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-takeover-response-b-e2231c0-raw.txt`
- 大小：`122,113` bytes
- 物理行数：`2,667`
- SHA-256：
  `e2231c0928c7dd345a4c7a0cf8b55afe4de95270b710b95602ddd6b5c3fb4bf5`
- `cmp` 校验：`byte_identical=true`

## 2. 一句话裁决

```text
TOP_LEVEL_DIRECTION = HIGHLY_ALIGNED
EXECUTABLE_SPEC = NOT_ALIGNED
BLANKET_ACCEPTANCE = NO
SCIENTIFIC_CORE = ACCEPT_WITH_MAJOR_CORRECTIONS
DYNAMIC_K = REQUIRED_DECISIVE_CANDIDATE, NOT EMPIRICAL FACT
CANONICAL_INTERNAL_NAME = DUCA-RIME (UNFROZEN)
STRICT_NESTING = UNRESOLVED UNTIL TRAIN-ONLY ORACLE REGRET
MODEL_IMPLEMENTATION_OR_LONG_TRAINING = NOT AUTHORIZED BY THIS REVIEW
```

两份回复在论文问题、AdapTok 边界、证据优先级和最终停止逻辑上高度一致；但在
解码器约束、风险治理、训练成本合同、候选预算、损失、统计门槛和部署 fallback 上
存在不能同时成立的实质差异。因此它们不是两份等价实现说明，也不能选一份逐字执行。

## 3. 回复 A 的完整结构化吸收

回复 A 在 `142--193` 给出唯一裁决：停止把现有密度/inverse-CDF 作为论文中心，
转向外层动态预算、内层严格 exact-K 嵌套物理传输的 DUCA-METER；动态 K 被设为
主模型必要组成，但必须经过 Oracle、utility、定位风险和真实成本门。

其主要内容为：

1. 完整审视 DUCA 快照、现有 90 轮 K=384/K=192 诊断、坐标/NMS 问题和公平性；
2. 逆向 AdapTok 的 block-causal prefix、随机长度训练、多预算质量标签、
   `TransformeScorer`、Fixed/BiThr/BiDelta/ILP、scorer/solver/padding 成本；
3. 明确 AdapTok 已占据“多预算质量曲线 + scorer + 离散平均预算分配”，直接把
   reconstruction loss 换成 detector loss 只能作为迁移 baseline；
4. 评估十二条模型路线，最终在 `985--1003` 选择
   `R1 + R2 + R3 + R5 + R8`：
   marginal detection utility、high-tIoU risk、paired boundaries、
   strict nested ladder、structured exact-K solver；
5. 在 `1025--1356` 定义任务效用、几何/双端风险、内层嵌套传输和外层固定对偶价；
6. 提出 batch-composition invariance、预算随价格非增、动态平均预算必须允许
   平均值两侧候选、物理映射误差和端点误差到 tIoU 等命题候选；
7. 在 `1584--1763` 提出 total-60 mixed-K 训练、hard utility、pair-risk、
   curve calibration 和可选 direct gradient；
8. 在 `1769--1867` 给出逐视频 frozen-dual 推理、`risk infeasible -> Kmax`、
   物理映射后 NMS、K-bucket/ragged execution 和完整成本协议；
9. 在 `1957--2121` 设计多尺度 hard counterfactual、
   `G_rank/G_direct` 和 dynamic Oracle 门；
10. 在 `2127--2243` 给出 baseline、Oracle、首粒 60 轮矩阵、扩展和停止条件；
11. 在 `2249--2469` 给出 METER-TAD 论文标题、三项贡献、主图/主表和三类拒稿攻击；
12. 在 `2478--2685` 给出 30 天行动顺序与最终 MAJOR REDESIGN 裁决。

回复 A 的主要优点是主命题强、动态 K 地位明确、AdapTok 竞争边界清楚，并把
高 tIoU、双端风险、物理时间和真实成本同时放进论文中心。主要缺点是过早把 strict
nesting、候选 K、门槛、训练日程和风险 fallback 冻结成唯一方案。

## 4. 回复 B 的完整结构化吸收

回复 B 在 `169--231` 将裁决分层为：DUCA 科学问题继续、当前 density/inverse-CDF
主方法停止、dynamic K 进入决定性验证、最终候选为 MERTAD；若动态失败，可预注册
退回固定 K 边界保持插件，而不是事先把动态 K 放到附录。

其主要内容为：

1. 更明确地区分 paper/code fact、partner claim、design proposal、
   empirical hypothesis、theorem candidate 和 unresolved；
2. 完整说明 AdapTok 的 nested latent prefix、多预算 label/scorer、batch ILP、
   offline/online scorer、padding 和 solver 成本；
3. 在 `449--453` 承认 nesting 可能损害单 K 最优，但仍在主方法中先验要求 nesting；
4. 在 `611--942` 评估十二条路线，最终将 R1、R2/R11、R3、R5、R8 统一为 MERTAD，
   并给出 detached hard-rank 的 MERTAD-Lite 和高风险 conformal-causal 扩展；
5. 在 `958--1301` 定义真实端到端成本、任务效用、pair/gap/warp 风险、稳健效用、
   exact-K 可行域、固定对偶价、层级贪心/DP/OT/ST 角色；
6. 在 `1305--1420` 给出 batch invariance、价格单调、Hoeffding 预算校准、
   凹边际阈值、端点/tIoU、最大洞和 anytime 一致性等命题候选；
7. 在 `1424--1662` 给出五阶段 total-60、半批双分支 hard counterfactual、
   `L_rank + L_delta + L_risk + L_cal`、cross-fit 与 checkpoint 规则；
8. 在 `1666--1824` 明确三种预算制度，以训练侧 fixed dual 为主、test-batch ILP
   只作 AdapTok baseline，并要求 q→t 后再 unchanged NMS；
9. 在 `1828--1906` 严格区分 fixed、task-adapted dynamic、unlabeled calibration、
   strict zero-shot 和 uniform；
10. 在 `1910--2091` 细化 hard intervention、视频级 cross-fitting、
    `G_rank/G_direct` 和失败降级；
11. 在 `2095--2235` 给出依赖图、五个训练臂、推理消融、资源上限和 69–70 裁决；
12. 在 `2239--2483` 给出 MERTAD 论文叙事、claim 绑定和三类审稿攻击；
13. 在 `2487--2633` 给出 30 天计划、不做清单和最终 Go/Major Redesign/Stop。

回复 B 的主要优点是证据标签更严格、训练公平性和 fallback 更清楚、cross-fit 与
成本记账更完整。主要缺点仍是把 strict nesting 放进胜出方法、把尚未校准的预测风险
同时写成软目标和部分硬约束，并把多个未经功效分析的数值写得过于接近正式门槛。

## 5. 两份回复的共同结论

下列内容实质一致，项目完全吸收：

| 主题 | 共同结论 | 项目裁决 |
|---|---|---|
| 任务定义 | offline TAD 的 pre-backbone 真实帧采集 | 接受；不得称 Online TAD |
| 当前旧主线 | actionness/density/inverse-CDF 不能继续作论文中心 | 接受 |
| dynamic K | 必须正面验证，不能只作无关附录 | 接受为 required candidate |
| AdapTok | 多预算/scorer/ILP/nested prefix 已被占据 | 接受；直接迁移作 baseline |
| 效用监督 | soft gradient 不能证明 hard selection utility | 接受 |
| TAD 特异风险 | cls/reg/high-IoU 与 start/end pair 必须进入证据 | 接受 |
| 时间合同 | raw proposal 必须 q→physical t 后再 NMS | 接受为 P0 合同 |
| 预算协议 | test 前冻结 per-video dual，禁止主方法 test-batch ILP | 接受 |
| 训练公平 | 主臂 total-60 / 6,000 detector updates / 同 checkpoint 规则 | 接受 |
| 真实成本 | decode/H2D/probe/selector/solver/backbone/head/NMS/padding 均入账 | 接受 |
| hard gates | dynamic Oracle、G_rank、G_direct、pair-risk、第二 detector | 接受 |
| 失败处理 | dynamic 失败退固定；fixed inner 也失败则停止 learned acquisition | 接受 |
| 论文证据 | high-IoU、短动作、双端、真实 Pareto 与第二 detector | 接受 |

## 6. 两份回复并不一致的地方

### 6.1 命名与论文中心

- A 使用 `DUCA-METER / METER-TAD`；
- B 使用 `MERTAD`，并提供 `MERTAD-Lite` 与 conformal 扩展。

这不是科学矛盾，但继续增加名字会造成路线漂移。项目保留已有内部节点
`DUCA-RIME`，直到 Oracle 和 development seed 通过后再做投稿命名检索与冻结。

### 6.2 strict nesting 的地位

- A 在 `148--152`、`985--1003`、`1055`、`1254--1356` 把 strict nested
  写成主方法合同；
- B 在 `449--453` 承认其可能损害单 K 最优，却在 `914--926`、
  `992--1001`、`1233--1267` 仍将其冻结为主内层。

两者共同偏向 nesting，但都没有给出相对 independent-per-K optimum 的实测 regret。
离线 pre-backbone 插件是在 heavy execution 前一次性决定 K，不天然需要 progressive
cache reuse。因此 nesting 只能是待验证结构偏置，不能先验成为唯一可行域。

### 6.3 风险是硬约束还是软目标

- A 更偏向 pair/gap/drift 的硬可行性筛选，并在全不安全时回退 Kmax；
- B 把 `pair + gap + warp` 合并到 `Q=U-κσ-R`，同时在可行域中保留部分硬几何。

项目统一为：

1. exact-K、唯一、有序、坐标范围和可实现的最大洞/扭曲界是确定性硬约束；
2. learned pair/high-IoU risk 在未完成独立校准前是经验风险分数，不是安全保证；
3. 若风险阈值使候选不可行，fallback 频率、realized K 和成本必须进入预算证书；
4. 是否把 calibrated risk 作为硬 chance constraint，需在 O4 后冻结。

### 6.4 total-60 训练合同

- A：每步一个主 K 重型 forward，约 25% step 增加 no-grad alternative forward；
- B：半批视频双分支，使两分支总 heavy frames 等于完整 batch 单分支；
- A/B 的阶段边界、K 集合、loss 拆分和梯度比例也不同。

两套合同不能同时声称唯一公平方案。项目暂不冻结具体阶段。优先顺序是：

1. 先用 train-only、video-disjoint、cross-fitted frozen-detector hard labels 通过门；
2. development total-60 默认一 update 一次 detector optimizer step；
3. 若采用在线双分支，所有 trainable 对照必须采用相同半批双分支、相同 unique-video
   exposure 和相同 cumulative heavy frames；
4. hard-label/refresh 的额外 GPU-hours 单列，不伪装成免费；
5. direct detector gradient 默认关闭，只有 `G_direct` 通过才作为正式臂。

### 6.5 候选 K 与平均预算

- A 建议更密的 K grid，并在 mean-K=384 时允许 K>384；
- B 首版建议 `{128,192,256,320,384}`，并不适合在 mean-K=384 下产生非平凡动态。

K 集合不能由叙事先定。它必须满足 detector 可运行、成本可分辨、覆盖目标 mean cost，
并明确报告平均、p95、Kmax 和 tail latency。若 mean-K=384 需要 cap=512，必须承认
动态方法获得更高逐视频峰值预算，并同时提供相同 cap/SLA 的公平解释；不能只匹配均值。

### 6.6 数值门槛

A/B 对 Oracle、G_rank、G_direct、mAP@0.7、gap recovery、cost overhead 和第一粒种子
给出不同数值。它们全部保留为 reviewer proposal，不是正式门槛。正式阈值必须在
clean baseline 方差、视频簇相关性、最小可检测效应和 multiplicity 方案确定后，
在 official final result 出现前一次性冻结。

### 6.7 推理 fallback

- A 明确 `all risk infeasible -> Kmax`；
- B 的主伪代码直接在所有 K 上 argmax，没有完整定义全不可行行为。

项目要求 fail-closed 行为显式、可测试、可计费。若 Kmax fallback 频繁，dynamic policy
不满足平均成本或风险校准，必须失败而不是事后调阈值。

### 6.8 zero-shot / train-free

两者都试图统一 fixed、dynamic、unlabeled 和 strict zero-shot，但监督边界不同。
项目只保留“共享几何/坐标/成本表面”的统一；task-adapted utility 与 strict train-free
不是同一证据制度。train-free 只有在不读取目标标签、teacher、detector cache 或
test-time gradient，且跨 detector/数据集超过强规则基线后，才可能成为核心贡献。

## 7. 我是否完全同意两份回复

答案是：**不完全同意，但同意其科学中心和证据优先级。**

### 7.1 完全同意

1. 停止把 density/inverse-CDF、action enrichment 或动态 K 本身当核心创新；
2. 把 hard budget-conditional TAD utility、paired endpoints/high-IoU risk、
   physical-time exact-K acquisition 和 batch-invariant cost allocation 组成一个问题；
3. AdapTok direct transfer 必须进入强 baseline；
4. q→t 必须发生在 unchanged NMS 前；
5. 90 轮 K=384/K=192 只作诊断，不进入公平 total-60 主表；
6. clean native uniform、wrapper parity、真实成本、G_rank/G_direct 和第二 detector
   必须先后闭环；
7. 动态 K 必须通过或失败于明确门，而不能悄悄降为装饰性附录。

### 7.2 有条件同意

1. nested ladder：只有 Oracle regret 可接受时才采用；
2. fixed dual：接受 per-video batch invariance，但只保证校准分布上的期望成本，
   不保证任意测试集精确平均；
3. pair-risk：接受为核心候选，但在校准前只能称 risk surrogate；
4. mixed-K one-model：接受，但必须与 identical mixed-K uniform exposure 对照；
5. uncertainty/conformal：只在校准质量和样本假设通过后保留；
6. train-free mode：作为后置扩展，不与 task-adapted 主证据混称。

### 7.3 明确不同意

1. 在 Oracle 比较前冻结 strict nesting；
2. 把离散 Lagrangian 写成无条件强对偶或严格平均预算保证；
3. 用同一 calibration sample 选择 λ 后直接套单一固定 λ 的 Hoeffding 界；
4. 把 q→t before NMS 解释为 detector 内部非均匀时间语义已经解决；
5. 把 endpoint/tIoU 几何界解释为 mAP 保证；
6. 把任何具体 K grid、epoch 分段、loss 权重、25% alternative cadence 或
   G_rank/G_direct 数字视为已冻结；
7. 在未通过 strong baselines 与第二 detector 前使用 “first”、
   “detector-agnostic”、“localization-preserving”、“strict plug-and-play”；
8. 用 Kmax fallback 同时宣称严格平均预算；
9. 立即训练完整 METER/MERTAD/RIME；
10. 让附件中的固定快照判断覆盖后续 research-wiki 的项目决策与当前工作树事实。

## 8. 数学合同的项目修正版

候选模型在视频 `v` 上使用训练前固定的候选预算集合 `K_set`：

```text
cheap evidence e_v
    -> candidate exact-K sets S_v(K), K in K_set
    -> predicted task value U_hat_v(K)
    -> calibrated localization risk R_hat_v(K)
    -> measured realized cost C_hat_v(K)
```

训练侧 calibration split 冻结 `lambda`。测试期逐视频独立决策：

```text
K_v = argmax_K [
    U_hat_v(K)
    - kappa * uncertainty_v(K)
    - calibrated_risk_v(K)
    - lambda * measured_cost_v(K)
]
```

确定性 tie-break 选择较低成本。该形式支持 batch-composition invariance；当候选成本
单调且 tie-break 固定时，选择成本随 `lambda` 非增。它不自动提供：

- 离散非凸问题的零 duality gap；
- 任意测试集合的精确平均预算；
- distribution shift 下的风险/成本覆盖；
- detector 对非均匀 selected-axis 输入的定位保持。

这些必须通过独立 calibration、有限 λ 候选校正/样本分裂、预算违反率、坐标测试、
第二 detector 和实测结果补足。

最终 decoder 语义由 O2 决定：

- independent：预测 `budget-policy value`；
- strict nested：预测 `group-add marginal value`；
- weak overlap：只允许一个预注册候选。

## 9. 唯一收敛路线：DUCA-RIME（内部候选名）

在设计冻结前，不采用 METER-TAD/MERTAD 新名字。项目已有的 `DUCA-RIME` 仅作内部
候选标识，其目标是：

```text
raw offline video
  -> counted low-cost full-window evidence
  -> train-only hard utility + calibrated paired/high-IoU risk
  -> frozen per-video realized-cost dual selects K
  -> one Oracle-admitted exact-K physical-frame decoder selects positions
  -> ragged/K-bucket heavy backbone + unchanged detector/head
  -> raw selected-axis proposals q -> physical time t
  -> unchanged official filtering/NMS
  -> detections + complete realized-cost ledger
```

“一个 Oracle-admitted decoder”是关键：论文最终只部署 independent、strict nested
或 weak overlap 中的一种，不携带三套任意切换模型。

## 10. 后续实验计划

### Phase 0：事实、split、测量零点

1. 冻结完整视频 ID 级 detector-train、hard-label、utility-fit、dual/risk-calibration、
   certification/development 和 final-evaluation manifest；
2. 复评 released dense，恢复 local dense、clean native uniform K=384/K=192；
3. candidate K panel 使用 clean per-K uniform，或至少提供 identical per-video K
   sequence 的 uniform-position control；
4. clean/wrapper 做 tensor、mask、raw proposal、coordinate、mAP parity；
5. 实现并测试 raw q→physical t→official NMS；
6. 建立 no-probe uniform 与 probe+uniform 两套成本零点；
7. 做 baseline 视频簇方差、ICC/MDE 与功效分析，再冻结门槛。

停止门：任一 parity/坐标/成本身份不闭合，只修基础设施，不解释 learned gain。

### Phase 1：几何与执行内核

只实现可单测的：

- exact-K、unique、ordered、in-range；
- constant-evidence exact-uniform identity；
- deterministic tie-break；
- independent/nested/one weak-overlap Oracle decoder；
- q/t round-trip 与 boundary-inclusive max-gap；
- requested/effective/unique/backbone/padded K ledger；
- variable-K K-bucket/ragged execution，禁止全局 pad 到 Kmax。

这一步没有 learned selector 长训练。

### Phase 2：最高信息增益的四个决定门

1. `O1 Dynamic headroom`：同一 frozen/mixed-K-compatible detector、同一 inner family、
   同 realized mean cost，比较 all fixed K、best fixed、per-video Oracle、K-shuffle；
2. `O2 Decoder regret`：independent、strict nested、one weak-overlap 逐 K 比较；
3. `O3 G_rank`：video-disjoint cross-fit，测试 1/2/4/8/16、比例 swap、clip、start/end
   pair，并加入 random/shuffle/reverse/actionness/transition/motion nulls；
4. `O4 Pair-risk`：Brier/ECE/coverage/sharpness 与真实 start/end joint failure、
   mAP@0.7 proxy；所有 risk fallback 计费。

停止门：

- O1 无 headroom：删除 dynamic K 主线；
- O2 nesting regret 大：删除 strict nested，按预注册顺序试一次 weak overlap；
- O3 近随机：删除 utility head，不用 ST 补救；
- O4 不优于简单信号：删除 pair-risk contribution。

### Phase 3：一枚 development seed 的 total-60 因果矩阵

最小训练/推理矩阵：

| Arm | 隔离的问题 |
|---|---|
| U-fixed | clean fixed-K uniform 底线 |
| U-same-K | 同 full 的逐视频 K，位置 uniform |
| F-bound | fixed K 的最终内层 decoder + admitted utility/risk |
| D-shuffle | 同 K histogram，打乱 K 与视频配对 |
| D-no-risk | 动态模型删除 paired/high-IoU risk |
| AdapTok-TAD | detector quality curve + test-batch ILP direct transfer |
| RIME-full | 唯一 decoder + hard utility/risk + frozen per-video dual |

训练合同在运行前只冻结一版；所有 trainable 主臂共享：

- 6,000 successful detector updates；
- 同初始化、effective batch、unique-video exposure；
- identical mixed-K exposure、warmup/ramp；
- 同 cumulative heavy training frames；
- 同 EMA/checkpoint/evaluation rule；
- utility-label/refresh/scorer/solver 额外成本单列；
- development seed 不进入正式统计。

第一粒种子只作 Go/Stop。门槛由 Phase 0 功效分析冻结，不沿用 A/B 任一现成数字。

### Phase 4：正式论文证据

仅在 Phase 3 同时证明 inner policy、content-conditioned dynamic assignment、
high-IoU/paired risk 和真实成本均正向后：

1. 冻结方法、K set、thresholds、dual、统计方案；
2. 三枚未参与开发的新种子；
3. 第二 detector；
4. 第二平均预算面板；
5. Avg-mAP、mAP@0.6/0.7、short/medium/long、boundary error、pair support、
   max-gap、K distribution、p50/p95 latency、throughput、energy、memory；
6. train-free 模式最后单列。

## 11. 最终实现目标

最终目标不是“一个会预测密度的 sampler”，也不是 AdapTok 的 TAD 改名版，而是一个
可移植的 offline-TAD pre-backbone acquisition plugin：

1. 只用计费的低成本全窗口证据；
2. 在真实平均成本和逐视频 cap 下联合选择 `K` 与物理帧位置；
3. 选择目标由 cross-fitted hard TAD value 与 paired/high-IoU risk 定义；
4. 位置由唯一、经 Oracle regret 选择的 exact-K decoder 决定；
5. 后端 backbone/detector/head/loss/NMS 算法不改，坐标 adapter 明确披露；
6. raw proposals 在 NMS 前恢复物理时间；
7. variable K 产生真实执行节省，不被 padding/probe/solver 成本吞掉；
8. dynamic full 在 matched realized cost 下超过 best fixed 与 U-same-K，
   同时不损害高 tIoU/短动作/双端；
9. 第二 detector 方向一致后，才允许形成通用插件 claim。

若动态 Oracle 失败，最终目标降为 fixed-K boundary-preserving acquisition；
若 fixed-K inner policy 也不能超过 clean uniform，则终止当前 learned pre-backbone
selection 命题，而不是继续增加 loss、后处理修补或训练轮数。

## 12. 当前状态

```text
dual_response_read = complete
raw_archival = complete
comparison = complete
scientific_direction = accepted_with_major_corrections
canonical_candidate = DUCA-RIME
decoder_family = unresolved
mathematical_contract = revised_but_not_frozen
implementation = not_started_for_RIME
experiment = not_started_for_RIME
training_authorized = false
paper_claim_allowed = false
next_required_action = user_approval_of_design_then_phase_0_only
```

