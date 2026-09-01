# DUCA H65 First-Mixing SingleClock：Unit-1 终态门操作化裁决

**Nonce：** `DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824`

## 0. 唯一裁决

# `REVISE_GATE_IMPLEMENTATION`

不选择 `STOP_UNDERDEFINED_GATE`。

已接受的 Unit‑1 科学门本身足够明确：主对照、三项 `−0.20 pp` 非劣界、terminal EMA 主检查点、H65 replay 身份和 nominal-uniform bit identity 均已冻结。当前问题不是科学路线缺失，而是终结器把另一个更苛刻的“正增益、coadaptation、cost、旧配对证据”门错误写进了 `main_pass`。

本轮只应修正离线终结逻辑和统计定义，不改模型、不改训练、不重跑任何训练、不引入 Query 或 dynamic‑K。当前研究边界仍是固定 K384 的 H65-compatible 表示归因门。 

---

## 1. 当前终结器为何不符合已接受合同

| 当前实现                                                         | 判定                                                                                                   |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 以 `EMA ON − same-checkpoint gate-zero` 至少 `+0.50 pp` 为主门     | **错误因果估计量**。gate-zero 是对已经与 SingleClock 共同训练的 checkpoint 做推理期 post-treatment 消融，不能替代 H65 OFF/replay。 |
| 要求 `EMA ON − H65 OFF >= +0.50 pp`                            | **擅自把非劣门提高为优效门**。已接受界限是 `>= −0.20 pp`。                                                               |
| 要求 Avg-mAP 的 bootstrap `LCB95 > 0`                           | **事后提高门槛**。原合同未要求 CI 通过。                                                                             |
| 将 gate-zero 与 H65 OFF 的 coadaptation CI 限定在 ±0.20 pp         | **新增未注册等价门**，不得进入 Unit‑1 决策。                                                                         |
| `cost_pass` 使用 1.01/1.02/1.02 阈值并进入 `hard_fail`              | **凭空发明 Unit‑1 cost kill gate**。                                                                      |
| 用旧 RankPack/TrueTime 配对、Stage‑1 maturity 决定当前 Unit‑1         | **估计量越界**。它们不是该 Unit‑1 的通过条件。                                                                        |
| 要求 final 与 EMA 方向一致                                          | **新增门槛**。原合同指定 terminal final‑EMA 为 primary，final 仅需报告。                                              |
| 要求 SingleClock 标量在 final 和 EMA 中非零                           | **新增“参数必须被使用”门**。参数为零可削弱机制解释，但不违反非劣合同。                                                               |
| `strata_pass` 使用短动作 `−0.50 pp` 和 distortion interaction `>0` | **不是已接受的 boundary-error 条款**。现有 distortion 也不是 gap‑CV。                                               |
| 单种子通过后可令 `paper_claim_admissible=True`                       | **明确错误**。Unit‑1 永远只是单种子 development representation gate。                                             |

因此，现有 `CONTINUE_TO_REPLICATION / REVISE_WITHOUT_MORE_TIME_MODULES / PIVOT...` 三分支全部应从 Unit‑1 终结器移除。它们重新打开了路线，而本轮只允许产生 Unit‑1 的 PASS/KILL。

---

## 2. 冻结的比较角色

### 2.1 唯一主对照

主估计量固定为：

[
\Delta_m^{EMA}
==============

100\left[
m(\texttt{ema_on})
------------------

m(\texttt{h65_off_ema})
\right]\ \text{pp}
]

其中：

* `ema_on`：SingleClock ON 新训练的 epoch‑59 `state_dict_ema`；
* `h65_off_ema`：历史 H65 epoch‑59 EMA checkpoint 在当前兼容代码中的 OFF/replay；
* (m\in{\text{Avg-mAP},\text{mAP@0.6},\text{mAP@0.7}})。

**Unit‑1 主门只能读取这三个 EMA ON-vs-H65 OFF/replay point delta。**

### 2.2 same-checkpoint gate-zero 的角色

`ema_gate_zero` 和 `final_gate_zero` 必须：

* 加载与 ON 完全相同的 SingleClock checkpoint；
* 使用相同 `state_dict` 或 `state_dict_ema`；
* 只在推理时令 SingleClock residual gate 为零；
* selected indices、gathered RGB、mask 和 VideoMAE 输入保持相同。

它们只回答：

> 已共同训练的模型在推理时是否实际依赖 SingleClock 路径，以及非时钟参数发生了多少 coadaptation。

它们**不是**主基线，不承担非劣门、正增益门、等价门或 Unit‑2 准入门。ON−gate-zero 即使为负，也不能单独 KILL；即使为 `+1 pp`，也不能弥补 ON−H65 OFF 低于 `−0.20 pp`。

### 2.3 final 的角色

* `state_dict_ema` 是唯一 primary；
* `state_dict` final 只报告 checkpoint sensitivity、EMA 滞后或训练末端不稳定；
* 不要求 final 与 EMA 同方向；
* final 低于门槛不能覆盖 EMA PASS，final 高于门槛也不能挽救 EMA KILL。

---

## 3. 三类身份检查必须分开

### 3.1 H65 replay 身份：主比较的证据准入条件

当前 `h65_off_ema` replay 必须与历史冻结 H65 EMA reference 在以下五个边界一致：

1. selected integer indices；
2. gathered RGB tensor；
3. VideoMAE input tensor；
4. detector raw selected‑q proposals、scores、labels；
5. canonicalized official evaluator JSON。

要求：

* shape、dtype、ordering 和 canonical bytes 一致；
* tensor 使用 SHA‑256 或逐字节比较；
* evaluator JSON 先按冻结 canonical serialization 规范化，再比较字节；
* checkpoint、config、split、class map、NMS、evaluator 和 annotation hash 全部绑定。

H65 replay 身份失败时，**不得解释任何 ON−OFF mAP**。终结器应非零退出并写一个非科学判决错误码：

`INVALID_H65_REPLAY_IDENTITY`

这不是 `KILL_SINGLECLOCK_REPRESENTATION`，因为它没有证明 SingleClock 有害；它只说明基线身份不可继承。该错误不得被压缩成第三个科学 decision token。

### 3.2 ON 与 gate-zero 的输入身份：执行合同

对 final 和 EMA 分别要求：

* 同一 checkpoint 路径/hash/epoch/state key；
* ON 与 gate-zero 的 selected indices、RGB、positions、valid mask、VideoMAE input tensor 完全相同；
* 唯一运行差异是 clock residual 是否启用。

失败时使用：

`INVALID_SAME_CHECKPOINT_GATE_ZERO_EXECUTION`

同样不产生 PASS/KILL，因为这意味着消融臂未正确构造。

### 3.3 nominal-uniform bit identity：真正的硬 KILL 条件

在 exact canonical-uniform positions 下：

* relative clock residual 必须严格为零；
* relative bias tensor 必须严格为零；
* 修改后的第一处 temporal-mixing 输出必须与 gate-zero/H65 path bit-identical；
* final VideoMAE backbone output 必须 bit-identical；
* 不允许使用数值容忍、近似零或“mAP 相同”替代 bit identity。

若该条件失败，证据本身可以有效，且直接输出：

`KILL_SINGLECLOCK_REPRESENTATION`

因为这证明所谓“零残差保持 H65 identity”的表示合同不成立。

---

## 4. 三项 `−0.20 pp` 主指标只使用 point estimate

### 4.1 精确门限

三个条件均为包含等号的 point-estimate 非劣门：

[
\Delta_{\text{Avg}}^{EMA}\ge -0.20\text{ pp}
]

[
\Delta_{0.6}^{EMA}\ge -0.20\text{ pp}
]

[
\Delta_{0.7}^{EMA}\ge -0.20\text{ pp}
]

* 恰好 `−0.20 pp`：PASS；
* `−0.2001 pp`：KILL；
* 比较前不得四舍五入到两位小数；
* 建议将官方 JSON 中的规范十进制文本转为 `Decimal` 后计算 pp delta，并与 `Decimal("-0.20")` 比较，避免二进制浮点边界歧义。

### 4.2 bootstrap 的角色

保留现有 10,000 次 paired whole-video cluster bootstrap：

* RNG：`numpy.random.PCG64`；
* paired resampling unit：whole video；
* 同一 draw 同时用于 ON 与 H65 OFF；
* interval：一基排序统计量 rank 250 和 9750；
* CI 仍写入终态收据。

但是：

> **CI 完全不进入 Unit‑1 PASS/KILL。**

因此，即使 point delta 为 `−0.10 pp` 而 `LCB95 < −0.20 pp`，主指标仍通过；即使 point delta 为 `−0.21 pp` 而 CI 包含零，仍直接 KILL。

要求 `LCB95 >= −0.20` 或 `LCB95 > 0` 都会提高已接受门槛，禁止采用。

---

## 5. boundary error 的唯一冻结定义

现有两个候选中，选择：

# **同类、score-ranked、IoU≥0.5、一对一匹配**

它比“每个 GT 独立选最高 IoU proposal”更能防止同一个 proposal 被重复用于多个 GT，也能显式处理 matched recall。

### 5.1 使用哪一状态的 predictions

边界分析使用：

* `ema_on` 的最终 official per-video prediction JSON；
* `h65_off_ema` 的最终 official per-video prediction JSON。

两者必须已完成冻结的：

1. selected-q → physical-seconds 映射；
2. score filtering；
3. top-k；
4. IoU/NMS/voting；
5. official serialization。

不再施加额外 score cutoff、额外 top-k 或第二次 NMS。`gate_zero` prediction 不参与 boundary hard gate。

### 5.2 proposal–GT matching

对每个 video、每个 class 独立执行：

1. GT 使用与 official evaluator 相同的 duplicate-removal 规则；
2. predictions 按以下稳定顺序排序：

   * score 降序；
   * start second 升序；
   * end second 升序；
   * 原始 serialized row index 升序；
3. 逐 prediction 处理；
4. 在尚未匹配的同类 GT 中寻找最大 temporal IoU；
5. 仅当最大 IoU `>=0.5` 时建立匹配；
6. 最大 IoU 并列时，按 GT 的 `(start, end, canonical_occurrence_index)` 升序决胜；
7. 每个 prediction 和每个 GT 至多使用一次。

匹配先在该 video/class 的**全部 GT**上完成，再从中抽取高 gap‑CV 或高 boundary-density stratum 的 GT，避免一个 prediction 在不同子集分析中被重复分配。

### 5.3 unmatched GT 的处理

不能只报告 matched-pair MAE，否则模型可以通过漏掉困难 GT 人为降低误差。

对 GT (g=[s_g,e_g])，令 (d_g=e_g-s_g>0)。

若匹配 prediction (p=[s_p,e_p])：

[
e_s(g)=\min\left(1,\frac{|s_p-s_g|}{d_g}\right)
]

[
e_e(g)=\min\left(1,\frac{|e_p-e_g|}{d_g}\right)
]

若 GT 未匹配：

[
e_s(g)=e_e(g)=1
]

单 GT boundary error：

[
e(g)=\frac{e_s(g)+e_e(g)}{2}
]

这是一个包含漏检惩罚的、duration-normalized、lower-is-better 指标。它是 Unit‑1 hard boundary statistic。

同时必须报告但不单独过门：

* matched recall；
* matched start MAE seconds；
* matched end MAE seconds；
* matched mean boundary MAE seconds。

False positives 不直接进入该 error，但已经受到三项 official mAP 主指标惩罚，并通过 score-ranked matching 影响匹配过程。

### 5.4 每视频聚合

对某一 stratum (S)，每个拥有至少一个 (S)-GT 的视频计算：

[
E_v^S=\frac{1}{|G_v^S|}\sum_{g\in G_v^S}e(g)
]

全体 point statistic 为视频等权平均：

[
E^S=\frac{1}{|V_S|}\sum_{v\in V_S}E_v^S
]

配对差定义为：

[
\Delta E^S=E_{\mathrm{ON}}^S-E_{\mathrm{H65\ OFF}}^S
]

正值表示 SingleClock 边界误差恶化。

---

## 6. gap‑CV 与 boundary-density 的冻结定义

### 6.1 共同数据边界

所有 cutpoint 必须在读取 validation prediction 或 validation metric 之前，由 official **training population** 冻结。

使用：

* H65 OFF/replay 的 training selected-position identity；
* official training annotations；
* 冻结的 training window ledger；
* unique physical windows，重复 exposure 按 `sample_id` 去重；
* `numpy.quantile(..., method="linear")`。

冻结并保存 q25、q50、q75；hard high stratum 只使用 q75。validation/test 不能参与 cutpoint。

### 6.2 gap‑CV

粒度：**window-level**。

对一个 window 的 valid selected positions：

[
p_0<p_1<\cdots<p_{K_v-1}
]

VideoMAE temporal tubelet size 固定为 2。仅使用完整 tubelet：

[
c_i=\frac{p_{2i}+p_{2i+1}}{2}
]

[
g_i=c_{i+1}-c_i
]

定义 population CV：

[
\operatorname{gapCV}(w)
=======================

\frac{
\sqrt{\frac{1}{n}\sum_i(g_i-\bar g)^2}
}{
\bar g
}
]

约束：

* 计算使用 `float64`；
* 只使用 valid prefix；
* padding、重复填充帧和 invalid mask slots 全部排除；
* trailing incomplete tubelet 排除；
* 位置不严格递增或平均 gap 非正时，身份证据无效；
* 由于 CV 对统一尺度不变，可直接使用 physical dense frame indices，不依赖 FPS。

training high threshold：

[
\tau_{\mathrm{gap}}=
Q_{0.75}^{train}(\operatorname{gapCV})
]

validation high-gap window：

[
\operatorname{gapCV}(w)\ge\tau_{\mathrm{gap}}
]

阈值相等的窗口全部进入 high stratum；不得为了接近 25% 数量而打破 ties。

### 6.3 boundary-density

粒度：**window-level**。

对 window 的真实有效物理时间区间 ([t_w^s,t_w^e))：

* 只计 original GT start/end boundaries；
* 不把跨窗口动作裁剪点制造为新 boundary；
* 每个动作最多贡献一个 start 和一个 end；
* 位于 video final endpoint 的 end boundary，只在该视频最后一个有效窗口中计入；
* padding 时长不进入分母。

定义：

[
\operatorname{BD}(w)
====================

\frac{
#{b:\ b\text{ 是 GT start/end 且位于 window 有效支持}}
}{
t_w^e-t_w^s
}
]

单位为 boundaries/second。

training high threshold：

[
\tau_{\mathrm{BD}}=
Q_{0.75}^{train}(\operatorname{BD})
]

validation high-boundary-density window：

[
\operatorname{BD}(w)\ge\tau_{\mathrm{BD}}
]

### 6.4 从 window stratum 到 GT stratum

一个 GT instance 属于 high-gap stratum，当且仅当它的 start 或 end boundary 至少落入一个 high-gap window。

同理，一个 GT instance 属于 high-boundary-density stratum，当且仅当它的 start 或 end boundary至少落入一个 high-BD window。

同一 GT 被多个重叠窗口覆盖时，在 GT 集合中只保留一次。

### 6.5 AND、OR 还是分别过门

冻结为：

# **两个 stratum 分别过门，并且二者为合取关系。**

即：

[
\Delta E^{high\ gapCV}\le0
]

且

[
\Delta E^{high\ BD}\le0
]

两者都满足才算 boundary gate PASS。

不使用：

* high-gap 与 high-BD 的 OR 联合集；
* 二者的 AND 交集作为唯一门；
* validation 上选择表现更好的那个 stratum。

二者交集可以作为附加机制诊断，但不影响 PASS/KILL。

---

## 7. “不恶化”的统计规则

hard boundary rule 使用 **point delta**：

[
\Delta E^S \le 0
]

* 等于 0：PASS；
* 任意严格正值：KILL；
* 不引入 `+0.01`、`+0.05` 等新容忍 margin；
* 不要求 bootstrap `UCB95 <= 0`，因为这会提高原合同；
* 不用 bootstrap CI 挽救 point worsening。

仍执行 10,000 次 paired video-cluster bootstrap：

* eligible video 为不可拆分 cluster；
* 每次重采样视频后，对抽中的 per-video errors 取等权平均；
* ON 与 H65 OFF 使用同一视频 draw；
* PCG64；
* seed 由本 nonce 与固定 namespace 派生；
* 95% interval 使用 rank 250/9750。

建议固定 namespace：

* `UNIT1_BOUNDARY_HIGH_GAPCV_Q75_V1`
* `UNIT1_BOUNDARY_HIGH_BD_Q75_V1`

CI 只用于报告不确定性和后续论文设计，不进入 Unit‑1 判定。

---

## 8. 当前 artifact 不足时的唯一合法降级

### 8.1 可离线计算 boundary gate 的充分条件

必须同时已有并封存：

1. `ema_on` final official prediction JSON；
2. `h65_off_ema` final official prediction JSON；
3. H65 OFF/replay training selected-position identities；
4. H65 OFF/replay validation selected-position identities；
5. training/validation window 的真实有效物理时间 ledger；
6. official training/validation GT annotations；
7. prediction、annotation、identity、checkpoint、config 和 evaluator hash 绑定。

若全部存在，只做离线统计；不得重训，不得改变 checkpoint，不得重新选择阈值。

### 8.2 不足时禁止的动作

若任一必要 artifact 缺失：

* 不得为补 boundary 诊断重训；
* 不得用 gate-zero prediction 替代 H65 OFF prediction；
* 不得用 ON training identity 替代 H65 OFF training identity；
* 不得从 validation 分布估计 q75；
* 不得重新跑模型推理只为了补充未预先封存的 window-level diagnostic；
* 不得从现有 mAP 或 strata 结果反推阈值。

### 8.3 降级结果

此时输出：

`boundary_gate_status = NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP`

并将 boundary 条款降为 diagnostic-only。

Unit‑1 的科学判定只使用：

* H65 replay 身份；
* same-checkpoint execution identity；
* nominal-uniform bit identity；
* 三项 EMA ON−H65 OFF point non-inferiority metrics。

这不是把缺失证据当成 boundary PASS。必须同时输出：

* `boundary_mechanism_claim_supported = false`
* `boundary_gate_used_for_decision = false`
* 明确列出缺失 artifact。

在这一降级分支中，只要三项主指标与 hard identity 通过，Unit‑1 仍可 PASS；但不得声称 SingleClock 已证明保护高 gap-CV 或高 boundary-density 边界。

---

## 9. cost、paper claim 和 Unit‑2 准入

### 9.1 cost

Unit‑1 cost 的唯一角色是：

# **报告项，不是 hard kill gate，也不是 Unit‑2 前置门。**

因此：

* 删除 `median<=1.01`、`p90<=1.02`、`memory<=1.02` 对 decision 的影响；
* `--cost` 应改为可选输入；
* 有合法 cost receipt 时原样报告；
* cost 缺失或比例较高不能改变 Unit‑1 PASS/KILL；
* cost 可在以后 full-stack/paper admission 中单独审查，但不得追溯重写 Unit‑1。

### 9.2 paper claim

Unit‑1 无论 PASS 或 KILL：

```text
paper_claim_admissible = false
claim_boundary = single_seed_unit1_development_representation_gate_only
```

不得根据 `CONTINUE_TO_REPLICATION`、checkpoint recovery 或 cost receipt 把它设为 true。

### 9.3 Unit‑2 Query residual 何时允许进入 Builder

只有以下条件全部成立时，才可设置：

`unit2_query_builder_eligible = true`

条件：

1. Unit‑1 输出 `PASS_UNIT1_SINGLECLOCK_GATE`；
2. H65 replay identity 有效；
3. same-checkpoint gate-zero execution 有效；
4. nominal-uniform bit identity 通过；
5. 三项 EMA point non-inferiority 通过；
6. boundary gate 可计算时，两项 boundary delta 均 `<=0`；
7. boundary 不可计算时已按本规格显式降级，且没有伪造 PASS；
8. focused unit tests 全部通过；
9. 独立 Critic 返回 `UNIT1_GATE_IMPLEMENTATION_PASS`；
10. terminal v2 artifact、输入 hashes 和 first-failure ledger 已封存。

这只授权实现已接受的 Unit‑2 Query semantic residual。它不授权 dynamic‑K：

```text
dynamic_k_authorized = false
```

若 Unit‑1 KILL，Query residual 不得用于“修复”SingleClock。

---

## 10. 终结器的唯一决策算法

```python
# Evidence-binding failures do not produce a scientific PASS/KILL token.
require(h65_replay_identity_valid)
require(same_checkpoint_on_gatezero_execution_valid)
require(checkpoint_config_evaluator_bindings_valid)
require(primary_state_key == "state_dict_ema")

metric_pass = (
    delta_avg_pp >= Decimal("-0.20")
    and delta_06_pp >= Decimal("-0.20")
    and delta_07_pp >= Decimal("-0.20")
)

if boundary_evaluable:
    boundary_pass = (
        high_gapcv_boundary_error_delta_point <= 0.0
        and high_boundary_density_error_delta_point <= 0.0
    )
else:
    boundary_pass = True  # diagnostic downgrade, not positive boundary evidence

kill = (
    not nominal_uniform_backbone_bit_identical
    or not metric_pass
    or (boundary_evaluable and not boundary_pass)
)

decision_token = (
    "KILL_SINGLECLOCK_REPRESENTATION"
    if kill
    else "PASS_UNIT1_SINGLECLOCK_GATE"
)
```

以下变量不得出现在 `kill` 或 `metric_pass` 中：

* ON−gate-zero 增益；
* coadaptation CI；
* old RankPack/TrueTime evidence；
* Stage‑1 average mAP 或 maturity；
* final-vs-EMA direction；
* clock parameter 是否非零；
* short-action `−0.50 pp`；
* high-low interaction；
* cost；
* H65 historical recovery-state completeness；
* paper claim readiness。

---

## 11. 终态 JSON 的最小字段

```json
{
  "schema_version": "duca_h65_singleclock_unit1_terminal_gate_v2",
  "nonce": "DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824",
  "evidence_status": "VALID",
  "decision_token": "PASS_UNIT1_SINGLECLOCK_GATE",
  "primary_checkpoint_state_key": "state_dict_ema",
  "primary_comparison": "ema_on_minus_h65_off_ema",

  "identity": {
    "h65_replay_five_boundary_pass": true,
    "same_checkpoint_gatezero_execution_pass": true,
    "nominal_uniform_backbone_bit_identical": true
  },

  "thresholds_pp": {
    "average_mAP": -0.20,
    "mAP@0.6": -0.20,
    "mAP@0.7": -0.20,
    "comparison": "inclusive_point_estimate"
  },

  "primary_metrics": {
    "average_mAP": {
      "point_delta_pp": 0.0,
      "point_gate_pass": true,
      "ci_lower_pp_report_only": 0.0,
      "ci_upper_pp_report_only": 0.0
    },
    "mAP@0.6": {},
    "mAP@0.7": {}
  },

  "boundary_gate": {
    "status": "EVALUABLE",
    "used_for_decision": true,
    "comparison": "ema_on_minus_h65_off_ema",
    "high_gapcv_delta_point": 0.0,
    "high_gapcv_pass": true,
    "high_boundary_density_delta_point": 0.0,
    "high_boundary_density_pass": true,
    "bootstrap_samples": 10000,
    "bootstrap_cluster": "whole_video",
    "ci_role": "report_only"
  },

  "diagnostics": {
    "final_on_vs_h65_off": {},
    "ema_on_vs_same_checkpoint_gatezero": {},
    "final_on_vs_same_checkpoint_gatezero": {},
    "gatezero_vs_h65_off_coadaptation": {},
    "short_action": {},
    "gapcv_interaction": {}
  },

  "cost": {
    "decision_role": "report_only",
    "status": "AVAILABLE"
  },

  "claim_boundary": "single_seed_unit1_development_representation_gate_only",
  "paper_claim_admissible": false,
  "unit2_query_builder_eligible": true,
  "dynamic_k_authorized": false,
  "first_failure": null
}
```

若证据身份无效：

* `evidence_status="INVALID"`；
* `decision_token` 必须为 `null`；
* 进程非零退出；
* 写明唯一 first failure；
* 不得输出一个伪装成科学结果的第三 decision token。

合法科学判决只有两个：

1. `PASS_UNIT1_SINGLECLOCK_GATE`
2. `KILL_SINGLECLOCK_REPRESENTATION`

---

## 12. 必须覆盖的 focused unit tests

| 测试                                                             | 预期                                                                |
| -------------------------------------------------------------- | ----------------------------------------------------------------- |
| 三项 point delta 都恰好 `−0.20 pp`                                  | PASS                                                              |
| 任一 point delta 为 `−0.2001 pp`                                  | KILL                                                              |
| point `−0.10 pp`，但 bootstrap LCB 为 `−0.50 pp`                  | PASS，CI 仅报告                                                       |
| point `−0.21 pp`，但 CI 包含零                                      | KILL                                                              |
| ON−gate-zero 为 `−2 pp`，ON−H65 OFF 三项通过                         | 不影响 PASS                                                          |
| ON−gate-zero 为 `+2 pp`，ON−H65 OFF 任一为 `−0.21 pp`               | KILL                                                              |
| final 三项失败但 EMA 三项通过                                           | PASS，final 仅诊断                                                    |
| EMA/final 方向不一致                                                | 不影响 decision                                                      |
| cost ratios 为 1.5/1.8/1.3                                      | cost 报告，不能 KILL                                                   |
| coadaptation CI 超出 ±0.20                                       | 只报告                                                               |
| old TrueTime/RankPack gate 失败                                  | 不影响 Unit‑1                                                        |
| clock alpha 恰好为 0 且有限                                          | 可报告机制未激活，但不能单独 KILL                                               |
| H65 replay selected indices 或 evaluator JSON 不同                | INVALID evidence，无 PASS/KILL                                      |
| ON 与 gate-zero checkpoint/state key 不同                         | INVALID evidence                                                  |
| ON 与 gate-zero selected RGB 不同                                 | INVALID execution                                                 |
| nominal-uniform first-mixing 或 backbone output 非 bit-identical | KILL                                                              |
| high-gap boundary delta `0`，high-BD delta `−0.01`              | boundary PASS                                                     |
| high-gap delta `+1e-12`                                        | boundary KILL，不加容忍 margin                                         |
| unmatched GT                                                   | start/end normalized error 各为 1                                   |
| 同一 prediction 可匹配两个 GT                                         | 测试必须证明第二个 GT 保持 unmatched                                         |
| score tie、IoU tie                                              | 按冻结稳定顺序产生唯一匹配                                                     |
| training q75 与 validation 分布变化                                 | q75 保持 training freeze，不重算                                        |
| boundary artifacts 缺失                                          | 主指标+identity 决策，boundary diagnostic unavailable                   |
| boundary 不可计算但代码将其默认为“通过证据”                                    | 测试必须失败；只能是 decision downgrade                                     |
| 单种子 PASS                                                       | `paper_claim_admissible` 仍为 false                                 |
| Unit‑1 KILL                                                    | `unit2_query_builder_eligible=false`、`dynamic_k_authorized=false` |

---

## 13. 最小代码改动

### `finalize_duca_h65_singleclock_terminal.py`

必须：

1. 将 primary delta 改为 `ema_on − h65_off_ema`；
2. 将三个 hard metric 条件改为 point `>=−0.20 pp`；
3. 删除 `+0.50 pp`、`LCB>0`、coadaptation equivalence、old-pair、Stage‑1 maturity、direction consistency 和 cost 对 decision 的影响；
4. final、gate-zero、old pair、short-action、interaction 和 cost 只放入 `diagnostics`；
5. `paper_claim_admissible` 固定为 false；
6. 输出唯一 PASS/KILL token；
7. 证据身份错误非零退出，不伪装为科学 KILL；
8. cost 参数改为 optional/report-only；
9. 不要求 clock scalar 非零；
10. H65 historical recovery-state 缺口不得替代 replay identity。

### `analyze_duca_h65_singleclock_strata.py`

必须：

1. hard comparison 从 `ON − gate-zero` 改为 `ON EMA − H65 OFF/replay EMA`；
2. 新增 training-only gap‑CV q25/q50/q75；
3. 新增 training-only boundary-density q25/q50/q75；
4. 实现上述 score-ranked same-class IoU≥0.5 one-to-one matcher；
5. 实现 unmatched-penalized duration-normalized boundary error；
6. 分别输出 high-gap 和 high-BD point delta；
7. 现有 distortion、short-action official-mAP 和 high-low interaction保留为 diagnostics，不生成 Unit‑1 hard `strata_pass`；
8. schema 升级为 v2，并显式记录 cutpoint 未使用 validation/test。

### `bootstrap_duca_h65_official_map.py`

三项 official mAP bootstrap 可保持不变。

boundary error 不是 official mAP，建议新增一个很小的离线 helper，复用：

* `seed_from_nonce`；
* PCG64；
* 10,000 draws；
* whole-video paired resampling；
* ranks 250/9750。

不得修改 official AP 核心来伪装 boundary metric 为官方指标。

### 测试文件

更新现有 finalizer/strata tests，并新增：

* boundary matching；
* unmatched penalty；
* q75 training-only freeze；
* separate conjunctive strata；
* artifact-gap downgrade；
* cost/CI/gate-zero 不进入 decision。

---

## 14. 独立审查与执行链

### Builder

只修改终结器、离线统计器和 focused tests。不得修改：

* SingleClock 模型；
* selector；
* K；
  -训练 config；
* checkpoint；
* prediction；
* evaluator；
* Query；
* dynamic‑K；
* cost threshold。

### Critic

只审查以下六项：

1. primary estimand 是否唯一为 EMA ON−H65 OFF/replay；
2. 三项门是否为 inclusive point `−0.20 pp`；
3. gate-zero、final、CI、cost 是否完全退出 hard decision；
4. H65 replay invalid 与 representation KILL 是否正确区分；
5. boundary strata 是否 training-only、分别过门且 sign 正确；
6. 单 seed 是否永远 `paper_claim_admissible=false`。

唯一 closure token：

`UNIT1_GATE_IMPLEMENTATION_PASS`

或

`UNIT1_GATE_IMPLEMENTATION_BLOCKED`

### Evaluator

在不重训、不改 prediction 的条件下：

* 验证输入 hashes；
* 运行已有 primary bootstrap 或读取已封存 artifact；
* 若充分 artifacts 存在，只运行新的离线 boundary statistics；
* 若不足，封存 diagnostic downgrade；
* 生成 v2 terminal artifact；
* 不选择阈值，不解释 Query，不授权 dynamic‑K。

---

## 15. 最终返回合同

```text
next_owner:
  BUILDER_DUCA_H65_SINGLECLOCK_UNIT1_GATE_V2

next_action:
  按本裁决修订 finalizer、离线 boundary analyzer 和 focused tests；
  不改模型或训练；随后交给独立 Critic 做单次 focused closure。

dependency:
  已封存的 six-family terminal receipt；
  H65 epoch-59 EMA replay 五边界身份；
  EMA ON/H65 OFF official metrics；
  nominal-uniform bit-identity receipt；
  boundary 可计算时还需 ON/H65 OFF EMA predictions、
  H65 OFF training/validation identities、window physical-time ledger 和 GT annotations。

expected_return_at:
  BUILDER_UNIT1_GATE_V2_PATCH_TEST_RECEIPT
  +
  CRITIC_UNIT1_GATE_IMPLEMENTATION_CLOSURE
  +
  UNIT1_TERMINAL_ADJUDICATION_V2
```

## 绑定结论

Unit‑1 不是“SingleClock 必须产生显著正增益”的门，而是：

> 在恢复 H65 replay 身份后，SingleClock terminal EMA 是否在 Avg-mAP、mAP@0.6 和 mAP@0.7 上均未低于 H65 超过 `0.20 pp`，是否保持 canonical-uniform bit identity，并在当前已有 artifact 足以合法计算时，不恶化两个预注册高风险边界 stratum。

因此必须删除现有正增益、CI、coadaptation、cost 和旧证据硬门。完成这一有限修订后，Unit‑1 可由代码无歧义地产生唯一 `PASS_UNIT1_SINGLECLOCK_GATE` 或 `KILL_SINGLECLOCK_REPRESENTATION`，而不重开任何 DUCA 路线。
