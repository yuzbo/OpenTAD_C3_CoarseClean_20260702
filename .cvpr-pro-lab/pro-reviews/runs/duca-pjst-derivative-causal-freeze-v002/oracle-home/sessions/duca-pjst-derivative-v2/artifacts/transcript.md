# Oracle Browser Transcript

Conversation: https://chatgpt.com/g/g-p-6a796fef9a00819194024cf1de3bd697-duca/c/6a8dace6-e798-83ea-b112-d5636c54fb62

## Prompt

Execute the complete attached DUCA scientific adjudication exactly once. The attached prompt is authoritative and includes nonce DUCA-PJST-DERIVATIVE-CAUSAL-FREEZE-v002-20260825.

### File: .cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002.md
Lines: 1-105
````md
  1 | # DUCA PJST 最终机制与因果合同裁决
  2 | 
  3 | Nonce: `DUCA-PJST-DERIVATIVE-CAUSAL-FREEZE-v002-20260825`
  4 | 
  5 | Exact Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）  
  6 | GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`  
  7 | Frozen code/evidence revision: `b2ccfccab5b4912b59954afcc9b0364955327f7c`
  8 | 
  9 | 你是本轮 Scientific First-Author Agent 和最严厉的独立审稿人。请直接核验公开仓库的冻结 revision，
 10 | 给出唯一 `CONTINUE / REVISE / PIVOT / STOP`。不能把机制或实验选择退回给人类或 Coordinator。
 11 | 
 12 | ## 已冻结背景
 13 | 
 14 | DUCA/H65 使用低成本 ASFormer scout 学习动作性与边界语义，再通过确定性 transport 间接选出有序、
 15 | 非均匀的真实 RGB 帧；小模型不直接学习帧索引。本轮固定 `K=384` 只用于表示归因，未来 dynamic outer-K
 16 | 仍是论文主线。不得改变 selector/ASFormer、VideoMAE-S/Adapter、ActionFormer、loss、NMS、THUMOS14
 17 | 官方 split/evaluator、seed 3407 或 H65 的 30+60 训练合同。
 18 | 
 19 | 代码核验已确认：`vit_adapter.py:889` 在任何真实物理时间残差进入第 0 个 Transformer block 之前执行
 20 | temporal-kernel=2 Conv3D PatchEmbed。因此，相邻 selected rank 被当作等间隔帧进行首次重型混合；
 21 | post-PatchEmbed SingleClock 和 pre-NMS physical decode 均不能撤销这次混合。
 22 | 
 23 | 历史 H65 terminal-EMA `65.1257` 是单 seed、30+60 的诊断锚点。SingleClock 终结作业为
 24 | `TIMEOUT / EVIDENCE_ADMISSION_BLOCKED`，没有合法 PASS/KILL；不得重训 SingleClock。RankPack/TrueTime 的
 25 | 单 seed `+0.6208 Avg` 是相关机制信号，不是 PJST 证据。当前没有 PJST 实现、PRE_RUN、训练或结果。
 26 | 
 27 | 此前 PJST 浏览器调用因 authoritative terminal metadata 落入其他 Project 而被隔离；用户提供的完整可见
 28 | 回复只能作为独立审查，不能授权实施。本轮是新的 exact-DUCA 科学冻结请求。
 29 | 
 30 | ## 必须裁决的机制分叉
 31 | 
 32 | ### A. 原 support-aware PJST
 33 | 
 34 | ```text
 35 | m = (q_i*x_i + q_{i+1}*x_{i+1}) / (q_i + q_{i+1})
 36 | v = (canonical_delta / actual_delta) * (x_{i+1} - x_i) / 2
 37 | z = (W^- + W^+) * m + (W^+ - W^-) * v + b
 38 | ```
 39 | 
 40 | 它同时改变零阶外观 quadrature 与一阶变化率。support/Voronoi 在全局 K384 上形成，再切为 24×16。
 41 | 
 42 | ### B. derivative-only PJST（Coordinator 核验后的最小提案）
 43 | 
 44 | ```text
 45 | m = (x_i + x_{i+1}) / 2
 46 | v = (canonical_delta / actual_delta) * (x_{i+1} - x_i) / 2
 47 | z = (W^- + W^+) * m + (W^+ - W^-) * v + b
 48 | ```
 49 | 
 50 | 它保持原零阶外观平均，只校正真实物理间隔下的一阶变化率；support 仅作审计元数据，不进入 forward。
 51 | 其目的不是声称完整连续时间建模，而是最小地证伪“首次 tubelet mixing 的错误时间尺度造成损失”。
 52 | 
 53 | 两者都必须：零新增参数/optimizer group；canonical uniform 在任何浮点时间计算前直接旁路原 PatchEmbed；
 54 | 保持 selected RGB、rank、K384、heavy token 数；有效位置严格递增且无 duplicate；physical decode 在
 55 | filtering/top-k/IoU/NMS 前恰好一次；不允许 post-processing 二次映射。
 56 | 
 57 | 请在 A、B 或一个严格更小且不改变 H65 selector/detector 的替代中冻结唯一机制。若 support-weighted
 58 | zero-order term 不能由当前因果问题唯一推出，应明确删除或后置为独立消融，而不是为了复杂度保留。
 59 | 
 60 | ## 必须裁决的因果口径
 61 | 
 62 | 完整 Stage-2 联合训练会让表示梯度改变 selector，不能同时声称“端到端训练”和“两个实验臂 selected RGB
 63 | 完全相同”。请冻结唯一首个正式 falsifier：
 64 | 
 65 | 1. `fixed/replayed-selector representation attribution`：OFF 与 PJST 从同一 Stage-1 起点出发，训练期间
 66 |    接收相同逐窗口 positions/RGB/mask/K384；若既有 OFF 不满足该合同，说明 matched frozen-selector OFF
 67 |    是否必须新增。只允许声称首次表示效应。
 68 | 2. `end-to-end system total effect`：只新增 PJST ON，允许 selector 漂移；报告 Jaccard、边界覆盖、gap
 69 |    分布与 selector 位移。只允许声称完整系统总效应，不能声称同一 RGB。
 70 | 
 71 | 选择能最便宜、最干净地回答首次混合因果问题的一项。不要把两个 estimand 合并进同一结论。
 72 | 
 73 | ## 统计修正与实验边界
 74 | 
 75 | - paired whole-video bootstrap 必须重跑 pooled official evaluator，不能先算 per-video AP 再平均。
 76 | - 10,000 个样本的双侧 95% percentile interval 冻结为 2.5%/97.5% quantiles，并明确索引/插值规则；
 77 |   第 500/9500 个顺序统计量约为中心 90%，不得称为 95%。
 78 | - 单 seed 视频 bootstrap 不是训练 seed 稳健性。
 79 | - `+0.50 pp`、时延/显存 `1.02x`、short-action `-0.50 pp` 等门若保留，必须给出历史方差、最小可检测
 80 |   效应或资源预算依据；否则冻结更合适的决策规则。
 81 | - 不重复 dense/uniform/random、RankPack/TrueTime、SingleClock、H65 60-epoch compression、UVT、Fovea、
 82 |   Query/Bridge、continuous cliplet 或 dynamic-K 矩阵。
 83 | - 不用 subset、synthetic、本地 CPU 或早期 checkpoint 形成效能结论；不接触 held-out/test 反馈。
 84 | 
 85 | ## 必须返回的终稿
 86 | 
 87 | 第一行仅给出 `CONTINUE / REVISE / PIVOT / STOP`，随后冻结：
 88 | 
 89 | 1. 唯一 PJST 公式、机制名、论文直觉、falsifiable prediction、anti-claim；
 90 | 2. 唯一 causal estimand，以及干预、结果与中介变量；
 91 | 3. 精确 tensor/shape/dtype/device、global-K384 pair metadata、padding/short-video/mixed-batch、identity、
 92 |    numerical-stability 与 exactly-once physical decode 合同；
 93 | 4. 最小 Builder 修改面，优先只允许 `temporal_grid.py`、`backbone_wrapper.py`、`vit_adapter.py`、一份配置、
 94 |    focused tests 和既有 launcher；列明禁止修改面；
 95 | 5. shape/layout、uniform byte identity、显式代数参考、constant-pair invariance、gap scaling、全局坐标、
 96 |    padding、K384、无新参数、有限梯度、production trace 与 pre-NMS 单次映射测试；
 97 | 6. 同 checkpoint、同输入只读兼容检查能排除什么、不能证明什么、停止条件；
 98 | 7. 唯一真实 THUMOS14/N16R4 实验：是否需要 matched OFF、60 epoch/6000 successful updates、每 5 epoch
 99 |    完整恢复 checkpoint、latest-3+milestone+final、terminal final/final-EMA、seed、成本、结果根；
100 | 8. Avg-mAP、mAP@0.6/@0.7、short/adjacent/gap strata、bootstrap 与预先冻结的通过/停止规则；
101 | 9. 最窄新颖性 claim 及 TDN、TAdaConv、Run-Length Tokenization、ToMe、TE-TAD 的 invalidator；
102 | 10. `next_owner / next_action / dependency / expected_return_at / single_recovery`。默认链为 clean Builder
103 |     最小实现 → 独立 Critic → Evaluator PRE_RUN → 仅在准入后立即运行唯一正式实验。
104 | 
105 | 不能声称 PJST 已实现、已 PRE_RUN、有效、提高 mAP、降低成本或 paper-ready。
````

## Answer

REVISE

**Nonce：`DUCA-PJST-DERIVATIVE-CAUSAL-FREEZE-v002-20260825`**

冻结 revision `b2ccfccab5b4912b59954afcc9b0364955327f7c` 已核验存在，对应提交为 `Validate SingleClock physical position identity`。 代码核验支持本轮问题定义：VideoMAE-S 的 PatchEmbed 是 temporal kernel/stride 均为 2 的 Conv3D；`x = self.patch_embed(x)[0]` 发生在任何 `actual_positions/canonical_positions` 残差被构造并送入第 0 个 Transformer block 之前。 全局 K384 坐标当前先生成 canonical/actual 序列，再按 24 个 16-frame clip 打包为每 clip 8 个 tubelet。

因此，PJST 的首个正式机制必须只修复**首次 2-frame tubelet mixing 的一阶时间单位**。原 A 方案把 support/Voronoi 权重同时写入零阶外观平均，不能由这一因果问题唯一推出。另一个必须修正的实现事实是：冻结代码中的 selected→physical remap 虽在 NMS 前，但目前位于 confidence filtering/top-k 之后；它不满足本轮更严格的“在 filtering/top-k/IoU/NMS 前恰好一次”合同。

---

## 1. 唯一冻结机制

### 1.1 机制名

**`PJST-D1: Derivative-Only Physical-Jacobian Scaled Tubelet`**

选择 **B：derivative-only PJST**。原 support-aware A 从首个正式实验中删除；其 support-weighted zero-order 项不得作为“更完整所以保留”的默认模块。只有 PJST-D1 通过、且以后出现独立的 quadrature 假设时，A 才可在新的科学决定下成为独立消融。

### 1.2 全局物理坐标

对样本 (b)：

* 实际 selected positions：
  [
  P_b=(p_{b,0},\ldots,p_{b,383})
  ]
* canonical exact-uniform positions：
  [
  U_b=(u_{b,0},\ldots,u_{b,383})
  ]

`U_b` 必须调用冻结 revision 中唯一的 `exact_uniform_positions(T_b,n_b)`，保持其显式 round-half-to-even 语义；不得另写第二个 uniform generator。当前全局坐标工具已经以该函数生成 canonical anchors。

对 clip (c\in[0,23])、tubelet (r\in[0,7])，全局首索引为：

[
g(c,r)=16c+2r.
]

冻结：

[
\delta^{\mathrm{act}}*{bcr}
=p*{b,g+1}-p_{b,g},
\qquad
\delta^{\mathrm{can}}*{bcr}
=u*{b,g+1}-u_{b,g},
]

[
s_{bcr}
=\frac{\delta^{\mathrm{can}}*{bcr}}
{\delta^{\mathrm{act}}*{bcr}}.
]

两种 delta 均以 **dense physical frame tick** 为单位。若转换到秒，两者乘同一 `seconds_per_dense_tick`，比值不变。

### 1.3 唯一前向公式

令一个 2-frame tubelet 的输入为 (x^-=x_g)、(x^+=x_{g+1})：

[
m=\frac{x^-+x^+}{2},
\qquad
v=s\frac{x^+-x^-}{2},
]

[
y^-=m-v,
\qquad
y^+=m+v.
]

然后只调用一次原始、冻结的 Conv3D PatchEmbed：

[
z=W^-*y^-+W^+*y^+ + b.
]

它严格等价于：

[
z=(W^-+W^+)*m +(W^+-W^-)*v+b.
]

这比另算一次 correction convolution 更小：只在 PatchEmbed 前构造同 shape 的伪输入 (Y)，随后仍只运行原始 Conv3D 一次。不得增加第二个空间卷积、第二个 tubelet kernel 或可学习 calibration 参数。

### 1.4 关键不变量

由定义直接得到：

[
y^-+y^+=x^-+x^+.
]

因此零阶平均保持不变。仅 odd/derivative mode 被重标度。

* 若 (\delta^{\mathrm{act}}=\delta^{\mathrm{can}})，则 (s=1)，(y^-=x^-)、(y^+=x^+)。
* 若 (x^-=x^+)，则 (v=0)，任意 gap 下结果不变。
* 大实际 gap 导致 (s<1)，抑制被“单位 rank”错误放大的差分。
* 小实际 gap 导致 (s>1)，恢复按 canonical 间隔计量的一阶变化率。
* 不 clipping、不加 learned gate、不加 epsilon 修补。严格递增的整数位置已保证 (\delta^{\mathrm{act}}>0)。

### 1.5 uniform identity

若一个样本的有效整数前缀满足：

```text
P_valid == U_valid
```

则必须在任何 cast、除法或浮点时间计算之前直接把原始 (X) 送入原 PatchEmbed。不是令 (s) 浮点近似为 1，而是**整条 PJST 分支不执行**。

### 1.6 support 的地位

Voronoi support 只作审计元数据。对有效位置 (p_0,\ldots,p_{n-1})，可冻结连续 dense-index 边界：

[
e_0=-\frac12,\quad
e_n=T_b-\frac12,\quad
e_j=\frac{p_{j-1}+p_j}{2},
]

[
q_j=e_{j+1}-e_j.
]

`q` 可用于 gap/support 分层和日志，但：

```text
q 不进入 m
q 不进入 v
q 不进入 Y
q 不接收梯度
```

### 1.7 论文直觉、预测与反主张

**最窄直觉：** 冻结的两帧 tubelet kernel 可分解为 even appearance mode 与 odd temporal-difference mode；非等间隔采样只改变后一项的物理单位，因此只应向后一项施加时间 Jacobian。

**可证伪预测：**

1. exact-uniform 输入逐字节不变；
2. constant frame pair 不变；
3. PJST 效应随 (\left|\log_2(\delta^{\mathrm{act}}/\delta^{\mathrm{can}})\right|) 增大；
4. 若该错误主要损害边界，改善应优先出现在 high-gap、short-action、adjacent-action 和高 tIoU 分层；
5. 若总体变化与 gap distortion 无关，则“错误首次时间尺度”机制解释失败。

**anti-claim：** PJST-D1 不是完整连续时间模型、不是数值积分 quadrature、不是 support-aware appearance reconstruction、不是新 selector、不是 dynamic-K、不是成本降低机制，也不是端到端 H65 总效应。

---

## 2. 唯一 causal estimand

### 2.1 选择

冻结：

**`fixed-selector representation attribution`**

不选择首个 `end-to-end system total effect`。完整联合训练会使表示梯度改变 selector；那样可以估计系统总效应，但不能同时宣称两个实验臂接收相同 RGB。

### 2.2 是否必须新增 matched OFF

**必须新增。**

现有历史 H65 terminal、SingleClock timeout 或其他 OFF 收据没有证明：在 Stage-2 的每一次训练 exposure 中，它们与未来 PJST ON 使用完全相同的 positions、RGB、mask 和 K384。因此不得拿历史 65.1257 直接充当本因果实验的 OFF。

冻结两臂：

| 臂             | acquisition                                      | PatchEmbed              |
| ------------- | ------------------------------------------------ | ----------------------- |
| `MATCHED_OFF` | 同一冻结 Stage-1 selector、同一输入与 RNG；positions/RGB 固定 | 原 PatchEmbed            |
| `PJST_D1_ON`  | 与 OFF 逐 exposure 相同                              | derivative-only PJST-D1 |

### 2.3 干预

唯一干预变量：

[
A\in{0,1}
]

* (A=0)：原始 PatchEmbed；
* (A=1)：在原始 PatchEmbed 前执行 PJST-D1 输入变换。

不得同时打开 SingleClock、support bridge、TrueTime residual、Query residual 或任何 selector 变化。

### 2.4 结果变量

主结果：

[
\tau_{\mathrm{repr}}
====================

## \mathrm{Avg\text{-}mAP}_{\mathrm{finalEMA}}(A=1,S=S^*)

\mathrm{Avg\text{-}mAP}_{\mathrm{finalEMA}}(A=0,S=S^*),
]

其中 (S^*) 是两臂共享的冻结 acquisition exposure 序列。

次结果：

* mAP@0.6；
* mAP@0.7；
* short-action、adjacent-action、gap-distortion strata；
* start/end boundary diagnostic；
* final 非 EMA 结果，仅作预注册的次级诊断。

### 2.5 中介变量

只记录、不在主估计中条件化：

* (s=\delta^{can}/\delta^{act})；
* (|Y-X|)；
* PatchEmbed token displacement；
* raw proposal start/end displacement；
* gap/support strata。

Selector Jaccard 固定应为 `1.0`；任何不为 `1.0` 的 exposure 都是身份失败，不是待解释的中介效应。

### 2.6 因果口径限制

本实验最多支持：

> 在固定 acquisition 下，首次 tubelet derivative 的物理尺度校正是否改变最终 TAD 表示性能。

它不能支持：

* 端到端 selector+representation 总效应；
* PJST 改善了选帧；
* PJST 与 dynamic-K 协同；
* 同一机制在多个 seed 上稳健；
* paper-ready 性能结论。

---

## 3. tensor、shape、dtype、device 与物理时间合同

### 3.1 冻结 tensor

| 名称                        |                Shape | dtype/device                | 合同                          |
| ------------------------- | -------------------: | --------------------------- | --------------------------- |
| `X_sel`                   |    `[B,1,3,384,H,W]` | 当前 backbone 输入 dtype/device | H65 selected RGB，不改变        |
| `selected_positions` (P)  |            `[B,384]` | `torch.int64`，与输入同 device   | 有效前缀严格递增；无 duplicate        |
| `selected_mask` (M)       |            `[B,384]` | `torch.bool`，同 device       | 只允许 contiguous valid prefix |
| `dense_valid_len`         |                `[B]` | `torch.int64`，同 device      | 正整数，且有效 position `< T_b`    |
| `canonical_positions` (U) |            `[B,384]` | `torch.int64`，同 device      | 唯一冻结 generator              |
| `actual_delta`            |           `[B,24,8]` | `torch.int64`，同 device      | 有效 pair 必须 `>0`             |
| `canonical_delta`         |           `[B,24,8]` | `torch.int64`，同 device      | 有效 pair 必须 `>0`             |
| `pair_valid`              |           `[B,24,8]` | `torch.bool`                | 两个 frame slot 都有效           |
| `pair_scale` (s)          |           `[B,24,8]` | `float32`，同 device          | 有效 pair 为 ratio；无效 pair 为 1 |
| `support_width` (q)       |            `[B,384]` | 审计时 `float64`               | 不进入 forward                 |
| packed input (X)          |    `[B·24,3,16,H,W]` | 原 dtype/device              | 当前 VideoMAE clip layout     |
| pair view                 |   `[B·24,3,8,2,H,W]` | 原 dtype/device              | 与 Conv3D kernel=2 对齐        |
| transformed (Y)           |    与 packed input 相同 | 与 (X) 完全相同                  | 只改 irregular、valid pair     |
| PatchEmbed output         | `[B·24,8H_pW_p,C_e]` | 原输出 dtype/device            | token 数、layout 不变           |

冻结代码已表明输入按 16-frame clip 送入、每 clip 形成 8 个 tubelet，并在 wrapper 中将全局坐标按同样布局展开。

### 3.2 短视频与 padding

对每个样本：

1. `valid_count = selected_mask.sum()`；
2. 有效 position 必须是长度 `valid_count` 的严格递增前缀；
3. 无效 suffix 的 position 使用现有 sentinel，不参与 delta；
4. 任一不完整 pair（仅一侧有效或两侧均无效）设置 `pair_valid=false`，并保持原 (X)，不得推断 gap；
5. 不允许把 padding 当成新物理位置；
6. 不允许通过复制最后一帧形成“有效 duplicate”；
7. 若有效位置违反范围、严格递增或 mask-prefix 合同，立即 fail closed。

冻结 revision 已对 SingleClock metadata 增加 position 范围与严格递增检查；PJST 复用而不是放宽这些约束。

### 3.3 mixed batch

为同时保证 mixed batch 中 uniform 样本的 byte identity：

```text
Y = X.clone()
仅对 irregular 且 pair_valid 的 row/pair 写入 y-, y+
所有样本随后共同调用一次原 PatchEmbed
```

uniform row 不经历浮点时间计算，也不经历 `+ zero correction`。这避免不同 batch-size 的 Conv3D algorithm 选择破坏 identity。

### 3.4 numerical stability

* delta 先以 int64 相减；
* 仅在确认两个 delta 均正后转换为 float；
* 不使用 `clamp(delta, eps)` 掩盖非法位置；
* 不 clipping (s)，因为 clipping 会引入第二个机制；
* (m,v,Y) 在 `float32` 工作精度形成，再 cast 回原输入 dtype；
* uniform row 不 cast、不重算；
* 任一 `nonfinite X/s/Y/output/gradient` 都 fail closed；
* metadata 全部 detach，PJST 不向 positions 或 selector 反传。

### 3.5 exactly-once physical decode

唯一允许的状态转换：

```text
selected_axis_index
    -- exactly once -->
true_time_dense_index
    -- existing seconds conversion -->
seconds
```

对 PJST 正式臂，在每个样本的 `post_processing` 入口：

1. raw segments 必须声明 `selected_axis_index`；
2. 立即执行现有 piecewise-linear selected→true-time map；
3. 将状态改为 `true_time_dense_index`；
4. 然后才允许 confidence filtering、top-k、IoU、NMS、voting、serialization；
5. 若入口已经是 `true_time_dense_index`、状态未知、metadata 缺失或后续再次请求 inverse map，均 fail closed。

当前 `TrueTimeMap` 已检查位置有限、范围合法和严格递增，并实现 selected/true 两轴映射。

---

## 4. 最小 Builder 修改面

### 4.1 允许修改

**PJST 主机制：**

1. `opentad/models/utils/temporal_grid.py`

   * 增加一个纯函数生成 global-K384 pair delta、pair scale、pair-valid 和 audit support；
   * 不改 selector 或 canonical generator。

2. `opentad/models/backbones/backbone_wrapper.py`

   * 透传 `pair_scale/pair_valid/exact_uniform_identity`；
   * 保持现有 24×16 打包和输出 pooling。

3. `opentad/models/backbones/vit_adapter.py`

   * PatchEmbed 前构造 (Y)；
   * 原 PatchEmbed 调用、权重、bias、冻结策略和输出 layout 不变。

4. 一份新配置：

   * `pjst_derivative_only=True`；
   * `single_clock_admission=False`；
   * selector fixed；
   * 其余从 H65 30+60 合同继承。

5. focused tests；

6. 既有 H65 launcher，仅允许换配置路径、experiment ID 和结果根。

**唯一 claim-neutral 例外：**

7. `opentad/models/detectors/single_stage.py`

   * 只允许把现有 `_remap_selector_segments_for_post_processing` 调用移动到每样本 raw `segments/scores` 取出之后、任何 filtering/top-k 之前；
   * mapping 函数、NMS、阈值、排序、输出格式一律不改；
   * OFF 与 ON 共用同一修正。

该例外是必要的，因为冻结代码目前先做 score filtering/top-k，后做坐标 remap，再做 NMS。

### 4.2 禁止修改

不得修改：

* `pc_ot_mras_prebackbone_frame_selector.py`；
* ASFormer scout/selector；
* selected RGB、rank、K384；
* `structured_selection.py` 的 selection 机制；
* VideoMAE Transformer block、attention、Adapter；
* ActionFormer projection/head/prior/assignment；
* classification/regression/selector loss；
* optimizer 参数组定义、LR、scheduler；
* H65 30+60 epoch/update 合同；
* data split、augmentation、seed 3407；
* NMS callable/config；
* evaluator/class map；
* GT segment transform；
* dynamic-K、TrueTime/RankPack/SingleClock、Bridge、UVT、Fovea、Query；
* 任何 post-NMS 二次坐标映射；
* 任何新参数、buffer 或 checkpoint key。

---

## 5. 必须通过的 focused tests

在 Critic 之前，以下测试必须全部通过：

1. **shape/layout：** (X\rightarrow Y\rightarrow Z) 的 shape、stride、flatten order 与 OFF 一致。
2. **uniform byte identity：** canonical uniform 在 all-uniform 和 mixed-batch 中均 `torch.equal`；PatchEmbed 输出、raw proposals/scores 和最终 JSON hash 相同。
3. **显式代数参考：** 直接计算 (W^-y^-+W^+y^++b)，与 sum/difference 公式一致；float64 CPU reference `max_abs_error ≤ 1e-9`，生产 float32 使用预冻结 `rtol=1e-5, atol=1e-6`。
4. **constant-pair invariance：** (x^-=x^+) 时任意合法 gap 下输出不变。
5. **gap scaling：** synthetic pair 中实际 gap 加倍时 derivative contribution 精确缩半。
6. **global-coordinate pairing：** pair 必须是全局 `(16c+2r,16c+2r+1)`，不得在每 clip 重新生成 canonical grid。
7. **padding：** invalid/partial pair 完全保持 OFF 输入，不计算 gap。
8. **K384：** selected positions、RGB、heavy token 数与 OFF 完全一致。
9. **no-new-parameter：** ON/OFF `state_dict.keys()`、参数数量和 optimizer group schema 相同。
10. **finite gradients：** fp32 和生产 AMP 下，输入、Adapter 和 detector 的梯度均有限；selector 梯度必须为零。
11. **production trace：** 证明 (Y) 在 Conv3D PatchEmbed 前形成，且没有第二次 temporal/spatial convolution。
12. **physical decode once：** trace 证明 remap 是 raw proposal 后第一个坐标相关操作，NMS 实参为 physical-dense segments；double/unknown mapping 必须报错。
13. **same-input identity ledger：** exposure-level positions/RGB/mask hash 在两臂逐项相同。冻结代码已有 selected RGB、positions 和 mask SHA-256 审计基础，可扩展 exposure key，但不得把审计当作性能证据。

任一失败都阻断 Critic PASS 和 PRE_RUN。

---

## 6. 同 checkpoint、同输入只读兼容检查

### 6.1 检查内容

使用同一个 Stage-1 terminal checkpoint、同一 selected RGB：

1. OFF 正常前向；
2. ON 加载同一权重；
3. 给 ON canonical-uniform metadata；
4. 比较：

   * state_dict；
   * selected input；
   * PatchEmbed 输入/输出；
   * 各 backbone block 输出；
   * raw proposals/scores；
   * pre-NMS physical proposals；
   * final JSON。

要求所有 identity surface 逐字节相同。

另用一个合法 irregular fixture 检查：

* shape；
* finite forward；
* finite backward；
* coordinate trace；
* 无新参数；
* 不读取 GT/teacher/result。

### 6.2 它能排除

该检查可以排除：

* 开关 OFF 仍改变 baseline；
* layout/packing 错位；
* uniform 路径浮点近似；
* checkpoint key 漂移；
* metadata 误接；
* double mapping；
* 非有限算术；
* 隐藏新增参数或额外 Conv。

### 6.3 它不能证明

它不能证明：

* PJST 对 irregular sampling 正确；
* 60 epoch 可优化；
* mAP 改善；
* 训练 seed 稳健性；
* short/high-IoU 改善；
* paper novelty；
* full-stack 成本可忽略；
* dynamic-K 适配。

### 6.4 停止条件

任何 identity、shape、finite-gradient、state-dict、pre-filter remap 或 exposure-hash 失败，立即：

```text
PJST_P0_BLOCKED
```

不得通过 tolerance 放宽、fallback、换 checkpoint 或重训 SingleClock 修复。

---

## 7. 唯一真实 THUMOS14 / N16R4 实验

### 7.1 实验矩阵

只运行两臂：

```text
R0 = MATCHED_FROZEN_SELECTOR_OFF
R1 = MATCHED_FROZEN_SELECTOR_PJST_D1_ON
```

不运行 A、不运行 ON/OFF 2×2、不重训 SingleClock、不重复 RankPack/TrueTime、dense、uniform、random 或 dynamic-K。

### 7.2 共同起点与 selector 合同

* 同一个 Stage-1 30-epoch terminal checkpoint；
* checkpoint bytes/hash 完全相同；
* Stage-1 selector、ASFormer 和 acquisition rule 固定；
* selector 置 `eval`，参数不更新；
* 数据 sampler 与 augmentation 使用独立、预冻结且两臂相同的 RNG stream；
* 每个 exposure 的 key 至少包含：

  ```text
  epoch
  successful_update
  batch_slot
  video_id
  window_start
  augmentation_seed
  ```
* 两臂必须记录并比较：

  ```text
  selected_positions
  selected_rgb_sha256
  selected_mask_sha256
  dense_valid_len
  K
  ```
* 任一 exposure 不同，整个 paired experiment 无效。

### 7.3 训练合同

* seed：`3407`；
* Stage-2：`60 epochs`；
* successful optimizer updates：**恰好 6000**；
* 6000 updates 前结束或超过 6000 均无效；
* optimizer、scheduler、LR、loss、batch、augmentation 与冻结 H65 合同一致；
* selector 不更新是本 causal estimand 的明确控制，不得写成 end-to-end H65；
* PatchEmbed 仍冻结；PJST 无参数；
* EMA 与 OFF 完全一致。

### 7.4 checkpoint

每 5 epoch 写一次完整可恢复 checkpoint，包括：

* model；
* EMA；
* optimizer；
* scheduler；
* scaler；
* sampler/RNG；
* successful-update count；
* exposure ledger cursor。

保留：

```text
rolling latest-3
stage2_start
epoch-30 milestone
epoch-60 milestone
final
final-EMA
```

主结果预先冻结为 `terminal final-EMA`。`final` 必须报告，但不得在二者间择优。

### 7.5 结果根

逻辑结果根冻结为：

```text
DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002/
  seed3407/
    matched_off/
    pjst_d1_on/
```

Evaluator PRE_RUN 必须把它解析为一个启动前不存在的绝对 N16R4 路径，并冻结 code/config/checkpoint/data hashes。不得复用、追加或覆盖旧 root。

### 7.6 成本

预算只允许：

* 两个 Stage-2 full-run equivalents；
* 每臂 6000 successful updates；
* 每臂一次 final 与一次 final-EMA 官方 evaluation；
* bootstrap 为离线 evaluator 计算。

PJST-D1 必须仍只调用一次原 PatchEmbed；新增成本仅为 pair reshape、均值/差分、标量缩放和写回。报告：

* wall-clock；
* GPU-hours；
* peak memory；
* per-iteration latency；
* PJST transform latency；
* heavy token count；
* executed K。

不冻结 `1.02×` 之类硬成本门，因为本轮没有历史方差、MDE 或资源预算依据。若实现需要第二个 Conv、降低 batch size、增加参数或改变 heavy token 数，则属于**结构性合同失败**，无需等待性能结果。

---

## 8. 指标、bootstrap 与通过/停止规则

### 8.1 主指标

官方完整 validation split：

* Avg-mAP over 0.3:0.1:0.7；
* mAP@0.6；
* mAP@0.7。

主比较为 final-EMA paired difference；final 同样报告但不用于选择。

### 8.2 strata

所有 threshold 只从 training split 冻结，validation 不参与选择：

* `short`：training GT duration 的 q25；
* `adjacent`：training 中相邻 action 非负 physical gap 的 q25，overlap 自动属于 adjacent；
* `gap distortion`：
  [
  D=\max \left|\log_2
  \frac{\delta^{act}}{\delta^{can}}\right|
  ]
  其 low/mid/high 边界取 training replay 的 q25/q75。

stratified AP 使用预注册的 ignored-GT 语义：不属于该 stratum 的 GT 标为 ignore；只与 ignored GT 匹配的 prediction 不计 FP。它们是诊断指标，不替代 official primary evaluator。

### 8.3 paired whole-video bootstrap

* 恰好 10,000 个 resamples；
* cluster unit 为整视频；
* 每次从 N 个 validation video 中有放回抽 N 个；
* 两臂、annotations、predictions 使用同一 video multiplicity；
* 重复视频使用唯一临时 namespace；
* **每个 resample 重新运行 pooled official evaluator**；
* 禁止先计算 per-video AP 再平均。

设 10,000 个差值升序为：

[
d_{(1)}\le\cdots\le d_{(10000)}.
]

冻结 empirical inverse-CDF nearest-rank、无插值：

[
LCB_{95}=d_{(250)},\qquad
UCB_{95}=d_{(9750)}.
]

这是双侧 95% percentile interval。第 500/9500 个顺序统计量明确禁止作为 95% CI。

单 seed 视频 bootstrap 只表示 validation-video sampling uncertainty，不是训练 seed 稳健性。

### 8.4 决策规则

由于没有历史训练方差、功效分析或 MDE，本轮删除未经依据的：

* `+0.50 pp`；
* short-action `−0.50 pp`；
* latency/memory `1.02×`。

冻结三种结果处置。

#### `MECHANISM_PASS`

必须同时满足：

1. 所有身份、坐标、K、checkpoint 和 evaluator seals 通过；
2. `LCB95(ΔAvg-mAP) > 0`；
3. `ΔmAP@0.6 >= 0`；
4. `ΔmAP@0.7 >= 0`；
5. short 或 adjacent stratum 不得出现 `UCB95 < 0`；
6. high-gap stratum 的点估计改善大于 low-gap stratum，方向符合机制预测。

该 PASS 只允许后续**考虑** end-to-end total-effect 实验；不支持 paper claim。

#### `MECHANISM_KILL`

任一成立即停止 PJST：

1. `UCB95(ΔAvg-mAP) <= 0`；
2. `UCB95(ΔmAP@0.7) < 0`；
3. uniform identity、同 RGB、pre-filter physical decode、finite gradient 或 K384 合同失败；
4. 改善只能通过 selector 漂移、第二个 Conv、额外参数或不同输入解释。

#### `INCONCLUSIVE_STOP_EXPANSION`

其他任何结果，包括 Avg CI 跨 0、high-IoU 点估计为负但 CI 不确定、或 gap-stratum 预测不成立，均：

```text
NO CLAIM
NO TUNING
NO SECOND SEED
NO SUPPORT-WEIGHTED A
NO END-TO-END EXPANSION
```

必须回到 acquisition/training-maturity 假设，而不是继续堆叠时间模块。

---

## 9. 最窄新颖性 claim 与 invalidators

### 9.1 允许的最窄 claim

仅在机制实验通过且后续多 seed 证据另行建立后，候选新颖性表述为：

> 对非均匀真实帧 acquisition，PJST-D1 在任何重型时间混合之前，将冻结两帧 tubelet convolution 分解为保持不变的 appearance/even mode 与按实际物理间隔校准的 derivative/odd mode；在固定帧集合和未改变 detector 下，直接检验首次时间单位错误对高精度 TAD 的影响。

不得简化成“使用 temporal difference”“时间自适应卷积”“时长感知 token”“token merging”或“真实时间坐标 TAD”。

### 9.2 具体 invalidators

| 先行工作                    | 已覆盖的一般思想                                                                    | 对 PJST 的 invalidator                                                                                                          |
| ----------------------- | --------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| TDN                     | 显式利用相邻/跨段 temporal difference 进行动作识别建模。([CVF Open Access][1])               | 若其或直接后继已按真实 irregular timestamp 对 frozen tubelet odd mode 做 (\delta^{can}/\delta^{act}) 校正，则“derivative correction”新颖性失效。     |
| TAdaConv                | 根据局部/全局时间上下文校准卷积核，增强视频时序建模。([arXiv][2])                                     | 若其校准可严格化简为本公式，并已在不规则物理间隔、首个 tubelet mixing、固定帧集合下验证，则 PJST 只是 TAdaConv 特例。                                                    |
| Run-Length Tokenization | 在 Transformer 前删除重复视频 token，并用长度/时长编码表示被压缩运行。([NeurIPS Proceedings][3])     | 若其 length encoding 实际恢复了同一局部物理导数算子，而非仅表示 token duration，则“pre-transformer physical duration”不能作为差异点。                          |
| ToMe                    | 在 Transformer 内逐步合并相似 token，包括跨视频帧 token。([ICLR][4])                        | token merging/效率本身绝非 PJST 新颖性；若 PJST 被写成一种 token merge，claim 立即失败。                                                            |
| TE-TAD                  | 使用实际 timeline 的 time-aligned coordinate expression 完成 TAD 坐标建模。([arXiv][5]) | physical-time proposal expression、pre-NMS remap 本身不是新颖性；若 TE-TAD 或直接后继已在重型 backbone 首个 tubelet kernel 内完成同一导数校正，则核心 claim 失败。 |

目前这些一手来源足以否定宽泛 claim，但不足以证明绝对优先权。最终 novelty 仍需 Critic 做逐公式、逐代码的完整文献核验。

---

## 10. 交接合同

```text
next_owner:
  clean Builder

next_action:
  在 b2ccfccab5b4912b59954afcc9b0364955327f7c 的全新 clean worktree
  实现 PJST-D1、pre-filter exactly-once remap reorder、配置和 focused tests；
  不实现 support-weighted A，不运行训练或 official evaluator。

dependency:
  1. 本裁决全文；
  2. 冻结 Stage-1 checkpoint/hash；
  3. H65 selector 与数据 RNG 身份；
  4. canonical exact_uniform_positions 唯一实现；
  5. exposure-level RGB/positions/mask identity ledger；
  6. existing TrueTimeMap 与官方 NMS/evaluator；
  7. 无 SingleClock 重训、无历史结果复用。

expected_return_at:
  Builder 完整 diff、resolved config、全部 focused-test receipt、
  state-dict/optimizer identity、production trace、
  no-data/no-training/no-metric attestation 完成后；
  在独立 Critic 开始前返回。

single_recovery:
  Critic 首次审查若仅发现一个确定性的 P0 实现错误，
  允许一次 Builder focused correction 和一次 Critic focused recheck。
  科学歧义、第二个实现错误、identity failure 或坐标合同失败
  均终止该链并返回新的 Pro 决定；不得形成开放式 correction loop。
```

后续顺序冻结为：

```text
clean Builder minimal implementation
→ independent Critic
→ Evaluator PRE_RUN
→ 仅在 PRE_RUN_READY 后运行唯一 matched OFF / PJST-D1 实验
```

本裁决没有声称 PJST 已实现、已通过测试、已 PRE_RUN、有效、提高 mAP、降低成本或达到 paper-ready。历史 RankPack/TrueTime 的单 seed `+0.6208 Avg` 仍仅是相关机制信号，不进入 PJST 的阈值、置信区间或效果主张。

[1]: https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html "https://openaccess.thecvf.com/content/CVPR2021/html/Wang_TDN_Temporal_Difference_Networks_for_Efficient_Action_Recognition_CVPR_2021_paper.html"
[2]: https://arxiv.org/abs/2110.06178 "https://arxiv.org/abs/2110.06178"
[3]: https://proceedings.nips.cc/paper_files/paper/2024/hash/3181db351fd3ced43cd589b0b572675d-Abstract-Conference.html "https://proceedings.nips.cc/paper_files/paper/2024/hash/3181db351fd3ced43cd589b0b572675d-Abstract-Conference.html"
[4]: https://iclr.cc/virtual/2023/poster/12173 "https://iclr.cc/virtual/2023/poster/12173"
[5]: https://arxiv.org/abs/2404.02405 "https://arxiv.org/abs/2404.02405"
