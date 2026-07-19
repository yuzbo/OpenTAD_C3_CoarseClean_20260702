# 1. RESEARCH_VERDICT

**RESEARCH_VERDICT：REVISE**

当前 `57.57 Avg-mAP / 28.64 mAP@0.7` 可以作为可信的 **`full60-single-seed-supported`** 结果；我没有发现足以将其降级为无效结果的 P0 代码错误、GT 泄漏、秒坐标二次换算、checkpoint/evaluator 错绑或 artifact 伪造问题。冻结运行确实绑定了 commit `0dc5851a…`、tree `bddc9b9…`、K384/J192、seed 42、无 feature interpolation 和真正的 60-epoch cosine horizon；epoch-59 指标由独立 evaluator 重算，online/EMA 两套 499 项状态均可反序列化且有限。

但这份结果只证明：

> 在同一 J192/Q192 ActionFormer 结构下，把统一 rank 秒轴改成真实物理秒轴，对 assignment、regression metric 和 decode 有决定性正作用。

它**没有**证明当前稀疏检测表示合理，也没有证明相对旧 `63.61` 的差距来自某一个单独因素。旧随机 ActionFormer 同时使用了 `J192→Q384` feature interpolation、约两倍候选、selected-rank GT、不同 scheduler horizon、不同代码提交和不同 padding/geometry 路径，因此不能与当前 `57.57` 做因果比较。

## 五项布尔裁决

| 裁决项                |        结论 | 精确定义                                                                         |
| ------------------ | --------: | ---------------------------------------------------------------------------- |
| `CODE_CORRECT`     |  **true** | 当前代码正确实现了**检测头后段的物理时间度量**，不等于整个 VideoMAE–projection 表示是物理时间感知的               |
| `COMPARISON_FAIR`  | **false** | 当前 selected-axis vs physical-metric 两臂是公平的；与旧 `63.61`、dense `68.29` 的横向比较不公平 |
| `Q_LIFT_NEEDED`    |  **true** | 必须做同 commit 的 Q192/Q384 因子实验，才能定位剩余高 IoU 缺口；这不预设 Q384 一定更好                   |
| `FULL_TRAIN_READY` | **false** | 对下面推荐的新 query-lift 路线而言，在真实 CUDA gate 和四臂 20-epoch gate 前不得直接跑 60 epochs     |
| `PAPER_READY`      | **false** | 尚缺多 seed、Q 因果隔离、成本、机制诊断和第二数据集                                                |

---

# 2. K、J、Q 的真实调用链、形状、mask 与候选数

先统一定义：

* **K**：送入视频 backbone 的原始 RGB 观测槽位数，不等于有效观测必定为 K。
* **J**：VideoMAE tubelet 输出的原生时序 token 数。
* **Q0**：ActionFormer 最细层 detection feature/query 位置数。
* **QΣ**：六层候选位置总数。
* 一个位置有 20 个类别分数，因此分类组合数为 `20 × QΣ`；这不等于有 `20 × QΣ` 套独立特征。

## 2.1 Dense AdaTAD

实际形状链：

```text
imgs                         [B, 1, 3, 768, 160, 160]
Rearrange + flatten          [48B, 3, 16, 160, 160]
VideoMAE patch tokens        [48B, 384, 8, 10, 10]
spatial mean + concatenate   [B, 384, J=384]
linear feature interpolation [B, 384, Q0=768]
ActionFormer pyramid         [768, 384, 192, 96, 48, 24]
QΣ                           1512
cls logits                   [B, 20, 1512]
regression                   [B, 2, 1512]
decoded proposals            [B, 1512, 2]
```

`window_size=768`、48 个 16-frame chunks、VideoMAE tubelet size 2，以及 backbone 后 `Interpolate(size=768)` 均在配置中直接固定。

完整窗口下：

[
Q_\Sigma=768+384+192+96+48+24=1512.
]

分类展开后是 `1512×20=30,240` 个 class-location scores，再由 `pre_nms_topk=2000` 截断，不应把 30,240 写成 30,240 个独立 query。

## 2.2 旧随机采样 ActionFormer

实际形状链：

```text
logical dense window         768 slots
random_fixed RGB observations K=384
imgs                         [B, 1, 3, 384, 160, 160]
Rearrange + flatten          [24B, 3, 16, 160, 160]
VideoMAE native support      [B, 384, J=192]
linear feature interpolation [B, 384, Q0=384]
ActionFormer pyramid         [384, 192, 96, 48, 24, 12]
QΣ                           756
cls logits                   [B, 20, 756]
decoded proposals            [B, 756, 2]
```

这一路明确在 backbone 后对 `[B,384,192]` 做 `F.interpolate(..., size=384, mode="linear")`。它还把 GT 映射到 selected-rank 轴。

完整窗口下：

[
Q_\Sigma=384+192+96+48+24+12=756,
]

分类组合数为 `15,120`。

这里的 mask 是 `[B,384]`。插值后的 feature 长度被恢复为 384，因而可直接乘这个 raw-slot mask；但旧路径没有当前 native 路径的逐层 strict temporal padding isolation，这也是旧结果与当前结果不能严格横比的一个附加差异。

## 2.3 当前 native selected-axis 与 physical-metric

两臂共同链路：

```text
logical dense window         768 slots
random_fixed RGB capacity    K=384
raw mask                     [B, 384]
VideoMAE native support      [B, 384, J=192]
native mask                  [B, 192]
no feature interpolation
ActionFormer pyramid         [192, 96, 48, 24, 12, 6]
Q0                           192
QΣ                           378
cls logits                   [B, 20, 378]
decoded proposals            [B, 378, 2]
```

`align_native_tubelet_geometry` 强制 `K=J×tubelet_size`，并通过

```python
raw_masks.reshape(B, J, 2).any(dim=-1)
```

将 `[B,K]` mask 收缩为 `[B,J]`。因此尾窗口的有效数满足

[
J_v=\left\lceil K_v/2 \right\rceil,
]

而不是把 padding tubelet 当作有效候选。

当前 physical config 明确移除了 interpolation，将 projection 的 `max_seq_len` 设为 192，并启用真实秒轴 point rewrite。

当前所谓 selected-axis 也不是旧版 selected-rank GT：GT 仍是绝对秒，唯一变化是 `coordinate_mode="uniform_rank_seconds"`，即在窗口秒域上构造均匀 rank 对照轴。

完整窗口下：

[
Q_\Sigma=192+96+48+24+12+6=378,
]

分类组合数为 `7,560`。因此当前方案相对旧随机基线不仅 Q0 减半，**多尺度候选位置总数也正好减半**。

---

# 3. physical-metric 是否真正贯穿训练、推理、NMS 和秒坐标

## 裁决：列出的后段环节全部覆盖；VideoMAE 和 projection 表示没有覆盖

实际训练调用链为：

```text
ActionFormer.forward_train
  -> BackboneWrapper(inputs, raw_mask)
  -> VisionTransformerAdapter
  -> align_native_tubelet_geometry
  -> Conv1DTransformerProj(x, J-mask)
  -> FPNIdentity
  -> AnchorFreeHead.forward_train(..., metas)
  -> PointGenerator
  -> _build_physical_points_and_masks
  -> prepare_targets
  -> focal classification + DIoU regression
```

推理链为：

```text
ActionFormer.forward_test
  -> same backbone / native align / projection
  -> AnchorFreeHead.forward_test
  -> physical point rewrite
  -> continuous offset decode
  -> window-domain clamp
  -> seconds-domain post-processing / cross-window NMS
  -> evaluator
```

`metas` 是在 projection 之后才传到 head 的；projection 本身只接收 `x, mask`。

逐项核验如下。

| 环节                 | 是否物理秒域 | 代码依据                                                                                                       |
| ------------------ | -----: | ---------------------------------------------------------------------------------------------------------- |
| GT                 |      是 | `BuildPhysTimeRawFrameGeometry` 将 dense-window GT 通过 `(gt*stride+dense_origin)/fps` 转为绝对秒，并审计 clamp/filter |
| point center       |      是 | rank point 经分段线性映射写为物理中心                                                                                   |
| regression range   |      是 | 原范围乘 `physical_stride / nominal_stride`                                                                    |
| center sampling    |      是 | 使用 assignment point 的物理中心和物理 stride                                                                        |
| inside-GT 判定       |      是 | 物理 point 与秒 GT 直接相减                                                                                        |
| regression target  |      是 | 秒距离除以物理 stride                                                                                             |
| DIoU loss          |      是 | 预测和 target 都重新 decode 为秒 segment 后计算                                                                       |
| inference decode   |      是 | `start=center-left×physical_stride`，`end=center+right×physical_stride`                                     |
| proposal clamp     |      是 | clamp 到当前绝对秒 window domain                                                                                 |
| seconds conversion |      是 | `prediction_time_unit=="seconds"` 时只截断，不再次缩放                                                               |
| NMS                |      是 | proposal 已是秒；滑窗路径汇总后在秒坐标执行 cross-window NMS                                                                |

GT 转秒和 timebase 校验见 `phystime_raw.py:145-357`。

物理 point、range 和 stride 的改写见 `anchor_free_head.py:223-250, 326-351`。

assignment、center sampling、regression range 和 target normalization 见 `anchor_free_head.py:692-796`。

decode 与 DIoU 使用同一套 physical points。

`convert_to_seconds` 在 `prediction_time_unit=="seconds"` 时没有二次乘 stride 或除 fps。

所以初步怀疑第 1 点应修正为：

> physical-metric 确实只在 backbone/projection 之后进入，但从 head point construction 开始，它正确覆盖了 GT、assignment、range、center sampling、regression、decode、clamp、NMS 和 evaluator 坐标。

---

# 4. FPN 粗层物理中心与 physical stride 是否数学正确

设原生 J-token 的单调秒坐标为

[
p_0<p_1<\cdots<p_{J-1},
]

代码构造一个带显式窗口边界的分段线性映射

[
f(-0.5)=t_{\text{start}},\quad
f(i)=p_i,\quad
f(J-0.5)=t_{\text{end}}.
]

对于 level (l) 的 nominal stride (s_l=2^l)，rank point 为 (r)，代码使用

[
c_l=f(r),
]

[
\Delta_l=f(r+s_l/2)-f(r-s_l/2),
]

并将 point 变换为：

[
[;c_l,;
R_{\min}\Delta_l/s_l,;
R_{\max}\Delta_l/s_l,;
\Delta_l;].
]

这正是规则 ActionFormer cell 在 (f) 下的**分段线性 push-forward**。当采样均匀、(f(r)=ar+b) 时，它严格退化为标准 ActionFormer：

[
\Delta_l=a,s_l,\qquad
R\Delta_l/s_l=aR.
]

因此它不是明显的单位错误，也不是把 level stride 重复乘了两次。代码还在改写前 clone 了 rank center，避免了早期实现中“先写物理秒、再拿它与 selected count 比较”的静默 mask 错误。

但它只在以下意义上正确：

> 它是**候选坐标网格的正确 push-forward**，不是 irregular feature receptive field 的精确物理几何。

有三个限制。

第一，粗层 feature 是通过 rank-domain depthwise convolution、attention 和 stride-2 downsampling 产生的；它不是在不规则秒域按 (\Delta t) 做积分或卷积。

第二，非线性 (f) 下通常有

[
f(r)-f(r-s/2)\neq f(r+s/2)-f(r).
]

当前只保留总宽度 (\Delta_l)，丢失左右不对称 cell width。连续的 left/right 回归仍能表达非对称 segment，但 center sampling 和 range normalization 使用的是单一平均尺度。

第三，代码自身明确把最终 J-token 的支持写成“结构上界而非精确支持”。`native_temporal_geometry.py:65-159` 会拒绝“最终 feature 精确对应两个输入帧”的表述。

结合 12 层 chunk attention 与每层 temporal adapter 的 lineage 递推，中心 token 的结构依赖上界可以覆盖全部 192 个 tubelet；边缘 token 至少也可能覆盖约 96 个 tubelet。这里是**结构依赖上界**，不是声称所有依赖在训练后都同等有效。结论是：

* 作为 detection grid metric：**数学自洽**。
* 作为 feature support metric：**不精确**。
* 作为高 IoU 最终结构：**仍存在可证伪的模型错配**。

---

# 5. VideoMAE、TIA、ActionFormer projection 是否看见时间戳

## 结论：都没有

### VideoMAE

`BackboneWrapper.forward` 只向 backbone 传递 RGB tensor 和 temporal mask，没有传 `metas`、timestamp 或 (\Delta t)。

VideoMAE patch embed 使用规则的 `(tubelet_size=2, patch=16, patch=16)` kernel/stride，并添加固定 rank sinusoidal embedding。

### Temporal adapter / TIA

Adapter 的 temporal operation 是 kernel-3 depthwise `Conv1d`。它把相邻 rank token 当作等间距位置，不接收秒坐标。

### ActionFormer projection

`Conv1DTransformerProj.forward(self, x, mask)` 只有两个输入：

* 两层 kernel-3 Conv1d；
* 两层 stem Transformer；
* 五层 stride-2 branch Transformer。

没有 `metas`、timestamp 或 (\Delta t)。

当前 `n_mha_win_size=-1` 使 attention 是全局的，但 query/key/value 前仍有规则 rank-domain depthwise Conv1d，stride-2 branch 也仍按 rank 下采样。

## 后置几何修正的能力上限

它能够：

1. 防止 GT 被错误地压到 selected-rank；
2. 让长空洞对应更大的物理回归尺度；
3. 让 center sampling、level range 和 decode 使用真实秒距离；
4. 通过改变 assignment，间接改变分类和回归梯度。

它不能：

1. 让相隔 0.04 秒与相隔 4 秒的 rank-adjacent token 在 backbone 中受到不同处理；
2. 撤销已经发生的 rank-domain feature mixing；
3. 恢复未采样边界处的 RGB 证据；
4. 把一个最终全局混合 feature 精确归属到某个两帧 support；
5. 直接让分类 tower在推理时读取 gap、timestamp 或 coverage。

所以当前 physical-metric 更准确的定义是：

> **物理度量的 assignment/decode head correction**，而不是完整的 physical-time representation。

---

# 6. `J192→Q384` interpolation 到底改变了什么

它不只是共享 query-grid lift。

旧路径在 `Conv1DTransformerProj` **之前**对 feature tensor 做线性插值。

因此它同时改变：

1. **feature values**
   新的 384 个位置是相邻 J-token 的线性组合。

2. **后续卷积感受野**
   kernel-3 Conv1d 现在在 384-grid 上卷积；其对应的原始 J-token 范围与 J192 直接卷积不同。

3. **attention normalization 和 token multiplicity**
   projection 的全局 attention 从 192 个位置变为 384 个相关位置，softmax 分母和可交互状态数都变了。

4. **stride-2 pyramid**
   六层由 `[192,96,48,24,12,6]` 变为 `[384,192,96,48,24,12]`。

5. **正样本密度与 assignment opportunities**
   GT 周围可参与 center sampling 的位置数大致增加。

6. **候选覆盖和分类排序容量**
   QΣ 从 378 变为 756，class-location scores 从 7,560 变为 15,120。

7. **激活和动态模型容量**
   learnable parameter 数没有随长度增加，但 attention states、candidate multiplicity、优化路径和后处理竞争数量增加。因此“参数容量相同”不等于“有效模型容量相同”。

另外，ActionFormer 的边界回归是连续 offset：

[
\hat s=c-\hat l\Delta,\qquad
\hat e=c+\hat r\Delta.
]

所以 Q192 不会把边界硬锁在 192 个网格点上。Q 减半主要造成的是：

* 正样本和候选稀疏；
* 分类排序冗余减少；
* regression conditioning 改变；
* projection 上下文改变；

而不是简单的“边界只能量化到 1/192”。

---

# 7. 剩余 mAP@0.7 缺口目前能归因到什么

从数值形态看，当前 physical 相对旧随机的差距随 IoU 提高而扩大：

| IoU |   旧随机 | 当前 physical |     差值 |
| --: | ----: | ----------: | -----: |
| 0.3 | 79.87 |       77.20 |  -2.67 |
| 0.4 | 74.15 |       70.49 |  -3.66 |
| 0.5 | 66.12 |       62.53 |  -3.59 |
| 0.6 | 56.02 |       49.01 |  -7.01 |
| 0.7 | 41.87 |       28.64 | -13.23 |

旧结果见冻结记录。 当前结果见 full60 记录。

这与高 IoU localization/coverage 问题一致，但由于协议混杂，不能直接断言“主要就是 Q”。

## 当前可以证明的部分

### 已证明：规则 rank 秒轴是重大错误来源

同 commit、同 J/Q、同 seed、同 schedule 下，physical 相对 uniform-rank-seconds：

* Avg-mAP `+16.29`
* mAP@0.7 `+13.78`

因此 assignment/metric 不是边缘因素，而是主要机制之一。

### 不能证明：剩余 13.23pp 全来自候选数

因为旧路径同时改变了 feature interpolation、projection context、GT 轴、padding 路径和 scheduler。

### 观测缺失是 dense 上限问题，但不能单独解释当前对旧随机的差距

旧随机和当前方案都使用 K384 观测，因此未观测 RGB 是二者共享约束；它可以解释为什么 K384 难以达到 dense K768，但不能在没有 matched experiment 时解释当前 `57.57` 对旧 `63.61` 的全部差距。

## 必须注册的可证伪诊断

| 假设               | 必须计算的量                                                                       | 判定条件                                                              |
| ---------------- | ---------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| 观测缺失主导           | 按 GT start/end 到最近真实采样帧距离、support gap quartile 分组的 pre-NMS recall            | 最远 gap quartile 解释超过 50% 的 R@0.7 缺口，且 D-B 恢复不足三分之一                |
| 候选/assignment 主导 | 每 GT eligible/positive query 数、zero-positive GT、pre-NMS class-agnostic R@0.7 | D-B 提升至少 5pp recall，zero/low-positive GT 减少至少 40%                 |
| 分类排序主导           | class-agnostic 与 class-aware recall、oracle class-score replay                | class-agnostic 接近参考但 class-aware 低超过 5pp，oracle score 恢复至少 70% 缺口 |
| 回归主导             | oracle exact-boundary replay、matched boundary MAE                            | 保留分数而替换边界后恢复至少 70% 的 mAP@0.7 缺口                                   |
| NMS 主导           | pre-NMS、no-NMS、hard/soft-NMS、full-precision replay                           | pre-NMS recall 接近但 post-NMS 损失超过 5pp                              |
| 规则网格编码主导         | timestamp uniformization、timestamp shuffle、gap-CV 分层                         | 物理优势随 gap irregularity 增大，shuffle 后至少消失 50%                       |
| 仅 Q 数量主导         | 四臂 interaction                                                               | `C-A ≈ D-B` 且 timestamp counterfactual 不影响收益                      |

目前更合理的**待证伪优先级**是：

[
\text{Q/projection topology与排序}

>

\text{rank-time representation mismatch}

>

\text{纯NMS误差}.
]

但在四臂完成前，不应把这个优先级写成实验结论。

---

# 8. P0/P1/P2/P3 代码发现

## P0：0 项

没有发现足以使 `57.57` 失信的 P0。

* 运行脚本绑定 clean commit/tree、固定 config、dataset manifest、预训练 checkpoint hash 和 60-epoch scheduler。`scripts/run_phystime_g1_matched_full60_slurm.sh:26-90,121-175`。
* validator 独立加载 checkpoint、检查 online/EMA、重建 evaluator 并重算指标。`tools/bata/validate_phystime_g1_matched_full60_artifacts.py:44-207`。
* checkpoint validator 检查所有 tensor 有限、EMA 与 online keys 完全一致。
* 秒预测没有二次缩放。

## P1-1：物理时间在 projection 前完全不可见

**文件与行：**

* `opentad/models/detectors/actionformer.py:251-305, 345-382`
* `opentad/models/projections/actionformer_proj.py:128-174`
* `opentad/models/backbones/vit_adapter.py:59-111, 914-1003`

**形状：**

```text
backbone output     [B,384,J=192]
projection input    [B,384,192]
physical metadata   未进入 projection
head input levels   [B,512,192/96/48/24/12/6]
```

**影响：**

分类和上下文表示仍将不规则时间压缩为规则 rank sequence。physical-metric 只能纠正 head metric，不能让 feature aggregation 对 (\Delta t) 敏感。

**修复：**

在现有 support stem 与 branch pyramid 之间加入 deterministic physical-query lift；时间只用于 query geometry、relative-time attention 和 coverage，不更改采样器。

---

## P1-2：旧 `63.61` 不是当前 `57.57` 的公平对照

**文件与行：**

* `selected_axis_adatad_sparse_k384.py:31-52,137-157`
* `e2e_thumos_videomae_s_768x1_160_adapter.py:126-159`
* `phystime_g1a_physical_metric_native_j192.py:13-184`
* `run_phystime_g1_matched_full60_slurm.sh:127-139,183-193`

**不一致：**

1. `J192→Q384` feature interpolation vs 无 interpolation；
2. QΣ 756 vs 378；
3. selected-rank GT vs绝对秒 GT；
4. inherited 100-epoch scheduler horizon vs明确 60-epoch horizon；
5. 不同 commit；
6. strict temporal padding isolation 不同；
7. head geometry和 metadata contract 不同。

**影响：**

`57.57→63.61` 的 6.04pp 不能归因于 Q、插值、GT 轴或表示中的任何单一变量。

**修复：**

冻结一个新 commit，全部四臂重跑。旧 Q192 只能作为历史外部锚点。

---

## P1-3：coarse physical stride 是坐标 cell 的近似，不是 feature support

**文件与行：**

* `anchor_free_head.py:223-250,326-351`
* `native_temporal_geometry.py:65-159`
* `phystime_raw.py:362-575`

**影响：**

当一个 coarse rank cell 横跨多个不规则 gap 时，单一 `physical_stride` 丢失左右不对称和真实 feature support。它可能造成：

* center sampling 尺度过宽或过窄；
* level regression range 分配不稳定；
* 高 IoU 边界 conditioning 不佳。

这不是 silent unit bug，但属于结构上限。

**修复：**

第一轮四臂保持现有数学不变以隔离 Q；只有 Q 因子完成后，才允许单独测试显式

```text
left_cell_width_sec
right_cell_width_sec
```

不能把这项同时塞入四臂。

---

## P1-4：旧 feature interpolation 改变了整个 projection，而不是中性 query lift

**文件与行：**

* `end_to_end.py:950-965`
* `selected_axis_adatad_sparse_k384.py:149-153`
* `actionformer_proj.py:74-117,128-174`

**影响：**

旧结果无法回答“只是候选数是否不足”，因为 feature values、卷积 RF、attention states、branch topology 和 Q 均被改变。

**修复：**

Q384 必须从 J192 support 构造独立 detection queries；禁止在 support tensor 上调用 `F.interpolate`。

---

## P2-1：cross-window NMS 使用了提前四舍五入的 segment

**文件与行：**

* `opentad/models/detectors/single_stage.py:216-223`
* `opentad/cores/test_engine.py:80-106`

单窗口结果在返回前把 segment 保留到 `0.01s`、score 保留到 `1e-4`；滑窗汇总后再做 cross-window NMS。

**影响：**

它是共享 evaluator 协议，因此不使 physical-vs-selected 差值失效；但可能轻微改变 NMS tie、voting 和接近 IoU=0.7 的边界结果。

**修复：**

内部汇总和 NMS 保持 float32 全精度，只在最终 JSON 展示层进行可选格式化。新四臂必须在修复后的同一 commit 全部重跑。

---

## P2-2：“无 GT 固定采样”需要更精确表述

**文件与行：**

* `end_to_end.py:417-460,699-769`
* `phystime_raw.py:145-160`

`random_fixed_subsample` 在**已接受窗口内部**不读取 GT，并由稳定 sample key 产生固定的 sorted subset；但训练的 `random_trunc` window crop 本身会利用 GT 保证动作相交。

**影响：**

没有推理泄漏，且两臂共享；但不能写成“整个训练采样过程完全不接触 GT”。

**正确主张：**

> GT-free、non-learned、within-window fixed subsampling；训练窗口裁剪仍属于标准监督型 random truncation。

**修复：**

artifact 中分别记录：

```text
window_crop_uses_gt = true/false
within_window_subsample_uses_gt = false
```

当前 metadata 已有这两个维度，应在论文和 validator 中继续保留。

---

## P2-3：当前 “selected-axis” 名称容易与旧 selected-rank 混淆

**文件与行：**

* `phystime_g1a_selected_axis_native_j192.py:37-45,81-89,121-129`
* `tests/test_phystime_native_tubelet_geometry.py:122-138`

当前对照的 GT 和 prediction 仍在秒域，只是候选轴为 `uniform_rank_seconds`。

**影响：**

论文若继续写 selected-axis，审稿人可能误以为它复用了旧版 selected-rank GT remap。

**修复：**

实验表中写：

```text
A/C: uniform-rank-seconds control
B/D: physical-time-seconds
```

配置文件名可保留，但正文必须消歧。

---

## P2-4：现有 G1b 不能否定 support-query decoupling

G1b 替换了：

* `ActionFormer` detector；
* `Conv1DTransformerProj`；
* `ActionFormerHead`；
* regression parameterization；
* assignment；
* endpoint head；
* capacity 和 context。

`SupportDecoupledPhysicalQueryHead` 使用 center/log-width regression 和额外 endpoint head，并非当前 head 的 Q-lift。

其 matched 20-epoch 结果为 `30.88`，明显低于 physical-metric `44.88`，只能否定这一个完整 G1b 组合，不能否定“保留 ActionFormer 后进行 support-to-query lift”。

**修复：**

推荐方案必须保留 ActionFormer branch、head、loss、decode 和 NMS，仅在 support 与 query 之间增加一个受控桥。

---

## P3-1：metadata 中仍把 query count 命名为 selected count

**文件与行：**

* `anchor_free_head.py:49-70,104-130`
* `anchor_free_head.py:304-351`

**影响：**

未来 Q384 时容易再次把 Q 与 J/K 混写。

**修复：**

新增通用字段：

```text
phystime_support_valid_count
phystime_query_valid_count
phystime_query_count
phystime_query_positions_sec
phystime_query_provenance
```

旧字段只作为 backward-compatible alias。

---

## P3-2：当前测试缺少 Q-lift provenance 和全精度 NMS

现有测试已经覆盖：

* K/J contract；
* 无 interpolation；
* 秒域 domain edges；
* disconnected support 不填 gap；
* physical 与 uniform-rank 两臂只差 coordinate mode；
* full60 部署合同。

但没有覆盖：

* `J=192,Q=384` 且 observation count 仍为 K384；
* Q-independent parameter hash；
* Q query 不进入 support provenance；
* full-precision cross-window NMS；
* timestamp shuffle counterfactual。

这些必须在新 commit 中补齐。

---

# 9. 四套候选结构与唯一推荐

| 结构                                                                                   | 优点                                                        | 主要问题                                              | 创新风险                              | 裁决                  |
| ------------------------------------------------------------------------------------ | --------------------------------------------------------- | ------------------------------------------------- | --------------------------------- | ------------------- |
| 共享 feature lift：`F.interpolate(J→Q)`                                                 | 改动最小；能恢复 QΣ                                               | 同时改 feature、RF、attention、candidate； provenance 最差 | 极高，容易被判“恢复旧插值”                    | 只作辅助 baseline，不作主方案 |
| 纯 query-only grid + nearest/barycentric copy                                         | K/J/Q 可分开；便宜                                              | 本质仍是 feature copying；无明确 support aggregation      | 高，容易被判“换名字的插值”                    | 不推荐                 |
| **稀疏 support → deterministic query 的 cross-attention，随后复用 ActionFormer branch/head** | Q 与观测解耦；时间在 head 前进入；保留现有强 projection/head；可显式记录 coverage | 需要严格防止把 query state 称为观测；成本增加                     | 中等，需面对 DETR/continuous-query 先验工作 | **唯一主方案**           |
| 全时间感知 projection/VideoMAE/TIA                                                        | 从表示层解决 (\Delta t)                                         | 修改范围大；无法再做 coordinate/Q 因果隔离；成本和预训练错配高            | 中等，但工程与归因风险最高                     | 当前不启动               |

## 唯一推荐：Support-Preserving Physical Query Lift

定义：

1. K384 RGB 只产生 J192 support features；
2. J192 support 的内容和 provenance 不变；
3. 建立 Q192 或 Q384 的 deterministic detection-query coordinate；
4. 每个 query 通过 masked cross-attention 读取 J support；
5. query 是**检测状态**，不是新增 RGB、frame、tubelet 或 observation；
6. 输出进入原 ActionFormer branch pyramid、原 head、原 loss、原 decode、原 NMS。

这条路线比“直接全时间感知 projection”更适合作为下一步，因为它同时满足：

* 能独立操纵 Q；
* 不恢复 dense evidence；
* 时间在候选 head 之前进入；
* 可以保留已有 physical-metric 的 assignment/decode；
* 四臂具备完整的 2×2 因子可识别性。

连续 anchor、真实时间轴和 adaptive temporal queries 已有相邻工作，例如 RCL、TE-TAD，以及更近期的 continuous-time dynamics TAD。因此不能主张“首个连续时间 TAD”。可防守的主张只能是：

> 固定稀疏原始观测下，显式分离 observation support、feature token 与 detection query，不进行 dense RGB evidence imputation，并在物理时间上完成 query construction、assignment 和 decode。 ([arXiv][1])

---

# 10. 推荐方案的具体计算

## 10.1 Support encoding

保留现有 projection 的 embed 与 stem：

```python
support = projection.encode_support(x, support_mask)
# x       [B, 384, J=192]
# support [B, 512, J=192]
```

现有 J-domain stem 参数在四臂中完全共享。

## 10.2 Query coordinate

对每个样本，已有 J 个单调 axis anchors (p_j) 和 domain edges。构造同一个分段线性 (f(r))。

对于 `Qv` 个有效 query：

[
r_i=\frac{(i+0.5)J_v}{Q_v}-0.5,\qquad
q_i=f(r_i).
]

* 当 `Qv=Jv` 时，(r_i=i)，query positions 精确等于当前 J-axis；
* 当 `Qv=2Jv` 时，在 rank 间插入 detection query；
* 插入的是**坐标位置**，不是 feature observation。

query cell width：

[
w_i=f(r_i+J_v/(2Q_v))-f(r_i-J_v/(2Q_v)).
]

## 10.3 Cross-attention

```python
# support: [B, C=512, J]
support_value = support.transpose(1, 2)        # [B,J,C]

q_coord = normalized_query_embedding(
    center=(q_sec - domain_start) / duration,
    width=q_width / duration,
)                                             # [B,Q,C]

relative = (
    support_sec[:, None, :] - q_sec[:, :, None]
) / duration[:, None, None]                   # [B,Q,J]

logits = content_attention(q_coord, support_value)
logits += relative_time_bias(relative, support_width, query_width)
logits = logits.masked_fill(
    ~support_mask[:, None, :], float("-inf")
)

weights = softmax(logits.float(), dim=-1).to(support.dtype)
query = q_coord + output_proj(weights @ value_proj(support_value))
query = query * query_mask[..., None]
query = query.transpose(1, 2)                  # [B,C,Q]
```

约束：

* mask-before-softmax；
* 坐标输入必须 domain-normalized；
* 禁止把 raw absolute seconds 直接送入 content MLP；
* query 数改变时 parameter 数和 state-dict shape 不变；
* support tensor hash 在 A/B/C/D 间必须一致；
* metadata 单独报告 evidence coverage；
* query state 不计入 K 或 J。

随后复用现有五个 branch blocks：

```python
levels = [query]                               # [B,512,Q]
for branch in branches:
    query, query_mask = branch(query, query_mask)
    levels.append(query)
```

---

# 11. 同 commit 四臂最小实验

## 11.1 四臂定义

A/C 的名称在表中保留，但其严格定义是 `uniform-rank-seconds`，不是旧 selected-rank GT。

| 臂 | 支持轴                   |  Q0 | level lengths       |  QΣ | 唯一配置变量                                                   |
| - | --------------------- | --: | ------------------- | --: | -------------------------------------------------------- |
| A | uniform-rank-seconds  | 192 | 192/96/48/24/12/6   | 378 | `coordinate_mode=uniform_rank_seconds`, `query_ratio=1`  |
| B | physical-time-seconds | 192 | 192/96/48/24/12/6   | 378 | `coordinate_mode=physical_time_seconds`, `query_ratio=1` |
| C | uniform-rank-seconds  | 384 | 384/192/96/48/24/12 | 756 | `coordinate_mode=uniform_rank_seconds`, `query_ratio=2`  |
| D | physical-time-seconds | 384 | 384/192/96/48/24/12 | 756 | `coordinate_mode=physical_time_seconds`, `query_ratio=2` |

所有臂共同：

```text
RGB tensor capacity       K=384
native VideoMAE tokens    J=192
backbone                  identical
support encoder           identical
query-lift parameters     identical
ActionFormer branches     identical
head/loss/decode/NMS      identical
optimizer                  identical
schedule                   exact 20 epochs
seed                       42
sample/augmentation trace  hashed and matched
```

## 11.2 各层 tensor shape

### A/B

```text
input RGB          [B,1,3,384,160,160]
VideoMAE support   [B,384,192]
support stem       [B,512,192]
query lift         [B,512,192]
pyramid features   [B,512,{192,96,48,24,12,6}]
cls                [B,20,{192,96,48,24,12,6}]
reg                [B,2,{192,96,48,24,12,6}]
concat proposals   [B,378,2]
concat scores      [B,378,20]
```

### C/D

```text
input RGB          [B,1,3,384,160,160]   # unchanged
VideoMAE support   [B,384,192]           # unchanged
support stem       [B,512,192]           # unchanged
query lift         [B,512,384]
pyramid features   [B,512,{384,192,96,48,24,12}]
cls                [B,20,{384,192,96,48,24,12}]
reg                [B,2,{384,192,96,48,24,12}]
concat proposals   [B,756,2]
concat scores      [B,756,20]
```

C/D 的 384 query 绝不能写成：

* 384 observations；
* 384 VideoMAE tokens；
* 恢复了 384 个缺失帧；
* dense feature reconstruction。

它们只是由 J192 observed-support features 条件化的 384 个 detection states。

## 11.3 参数和成本

参数数目在 A/B/C/D 间必须完全相同；Q 不允许控制任何 `ModuleList` 长度或 learned embedding table 长度。

理论 Q-dependent 成本：

* support encoding：四臂相同；
* support→query cross-attention：Q384 是 Q192 的约 2 倍；
* head convolution与候选数：约 2 倍；
* global branch attention 的长度平方项：Q384 相对 Q192 约 4 倍；
* VideoMAE raw-video 成本：完全相同。

但不能把这些局部比率直接写成端到端 wall-time。raw-video backbone 很可能占主要成本，必须由真实 CUDA profile 给出 p50/p95。

## 11.4 旧 Q192 结果能否复用

**不能复用。四臂必须全部重跑。**

原因：

1. 新 projection 中加入了 support-query bridge；
2. NMS full-precision 修复改变 evaluator 内部路径；
3. 新 state dict、初始随机数消费和梯度路径都不同；
4. 若只重跑 C/D，B-A 与 D-C 不再是同一架构的因子比较。

旧 `41.28/57.57` 只能作为 historical external anchor，不能进入新四臂的统计主表。

---

# 12. 逐文件最小 patch 设计

| 文件                                                             | 类/函数                                   | 修改                                                                                    |
| -------------------------------------------------------------- | -------------------------------------- | ------------------------------------------------------------------------------------- |
| `opentad/models/projections/actionformer_proj.py:13-174`       | `Conv1DTransformerProj`                | 拆分 `encode_support`、`lift_queries`、`build_pyramid`；新增 `metas` 可选输入；禁止 `F.interpolate` |
| 同文件                                                            | 新 `SupportPreservingPhysicalQueryLift` | 构造 deterministic Q axis；masked cross-attention；输出 coverage diagnostics                |
| `opentad/models/projections/__init__.py:1-27`                  | module exports                         | 注册新 projection/lift                                                                   |
| `opentad/models/detectors/actionformer.py:251-305,345-382`     | train/test projection call             | 新增 `_call_projection(x,masks,metas)`；兼容 `(features,masks)` 与 `(features,masks,metas)` |
| `opentad/models/utils/native_temporal_geometry.py:125-189`     | native metadata                        | 明确 support fields，不允许 Q 覆盖 K/J 字段                                                     |
| 新建 `opentad/models/utils/query_temporal_geometry.py`           | query builder                          | 生成 Q positions、cell widths、mask、provenance；无 GT/teacher/selector 输入                   |
| `opentad/models/dense_heads/anchor_free_head.py:44-70,252-451` | physical-grid config                   | `selected_count` 泛化为 `axis_count/query_count`；验证 query provenance；保留现有 loss/decode 数学 |
| `opentad/models/detectors/single_stage.py:156-223`             | post-processing                        | NMS 前不 round segment/score                                                            |
| `opentad/cores/test_engine.py:80-106`                          | cross-window NMS                       | 全精度聚合和 NMS，最终 serialization 才格式化                                                      |
| 四个新 config                                                     | A/B/C/D                                | 只允许 `coordinate_mode`、`query_ratio`、`work_dir` 有差异                                    |
| 新 submit/run scripts                                           | deployment                             | exact commit/tree、四臂、seed42、20 epochs、共享 gate                                         |
| 新 artifact validator                                           | completion                             | 四臂 config diff whitelist、K/J/Q、参数 hash、sample trace、cost 和 evaluator 重算               |

不修改：

* `random_fixed_subsample`；
* K；
* logical dense window；
* GT 来源；
* teacher/oracle；
* learned selector；
* dynamic budget。

---

# 13. 必须新增的测试

## 单元测试

1. `test_query_ratio_one_is_identity_in_coordinate_space`

   * `Q=J`
   * `query_positions == current_axis_positions`
   * 不要求 feature bitwise identity，因为新 bridge仍参与四臂。

2. `test_q384_does_not_change_observation_or_support_count`

   * `K=384,J=192,Q=384`
   * K/J metadata 和 support tensor 不变；
   * query provenance 明确为 detection queries。

3. `test_parameter_state_is_q_independent`

   * Q192/Q384 `state_dict.keys/shapes/numel` 完全一致；
   * 相同 seed 的初始化 hash 一致。

4. `test_physical_and_uniform_modes_share_rgb_and_support`

   * 固定真实 batch；
   * A/B/C/D 的 raw indices、decoded RGB、VideoMAE support bitwise 相同。

5. `test_irregular_query_coordinates_are_monotone_and_edge_bounded`

   * Q positions 严格递增；
   * 位于显式 domain 内；
   * Q=2J 不产生 duplicate。

6. `test_zero_or_masked_support_never_enters_softmax`

   * mask-before-exp；
   * 无 `inf*0`；
   * FP16/AMP 下有限。

7. `test_no_gt_teacher_selector_dependency`

   * test metas 注入 GT/teacher/oracle 字段必须 fail closed；
   * 移除 GT 后 prediction 不变。

8. `test_full_precision_cross_window_nms`

   * 两个相差小于 0.01s 的 segment；
   * 验证 NMS 使用未 round 值；
   * 最终 JSON rounding 不反向改变 NMS。

9. `test_tail_window_k_j_q_masks`

   * 多个 `K_v`：1、2、3、383、384；
   * `J_v=ceil(K_v/2)`；
   * `Q_v=query_ratio×J_v`；
   * 各 level mask 与 tensor shape 一致。

10. `test_head_seconds_contract_for_q384`

    * GT、assignment、range、center sampling、decode、clamp 全在秒域。

---

# 14. 真实 CUDA gate

必须在固定 Slurm 单 GPU 环境中执行，不能用 synthetic-only smoke 替代。

## Gate 输入

* 真实 THUMOS train DataLoader；
* 至少一个含短动作、一个长动作、一个 tail window；
* 真实 test split 样本；
* production AMP、optimizer、scheduler、EMA、NMS、evaluator；
* A/B/C/D 同一 sample trace。

## Gate 必须完成

1. 每臂 **20 个 successful optimizer updates**；
2. 零 NaN、零 Inf、零 AMP skipped step；
3. 每步有有限 cls/reg loss；
4. query-lift 参数在 20 步聚合中至少一次非零梯度；
5. backbone/support tensor 在四臂同输入下 bitwise 相同；
6. Q384 的 K/J 不变；
7. query metadata 不含 GT、teacher、selector；
8. train/test prediction 均在绝对秒；
9. official evaluator 能构建并运行；
10. online 与 EMA 参数有限；
11. 记录 warmup 后 train/infer p50、p95、peak memory；
12. 记录每层 Q、有效 mask、候选数、coverage。

## 成本通过线

相对同 coordinate mode 的 Q192：

```text
Q384 train-step p50/p95   <= 1.40×
Q384 inference p50/p95    <= 1.40×
Q384 peak allocated memory <= 1.35×
OOM                        = 0
```

这些是准入阈值，不是当前代码已经达到的实测结论。

---

# 15. Artifact validator

每个 arm 的 completion artifact 至少绑定：

```text
git_commit
git_tree
canonical_config_sha256
effective_config_sha256
pretrained_checkpoint_sha256
dataset_file_content_manifest_sha256
annotation_sha256
class_map_sha256
seed
scheduler_horizon
optimizer
AMP scaler policy
K_capacity / K_valid histogram
J_capacity / J_valid histogram
Q0 / QΣ / Q_valid histogram
model parameter count
initial_state_dict_sha256
online_state_dict_sha256
ema_state_dict_sha256
sample_trace_sha256 per epoch
raw_selected_indices trace hash
query_coordinate trace hash
prediction JSON hash
full-precision pre-NMS artifact hash
post-NMS artifact hash
cost/profile artifact hash
```

Validator 必须：

* 重新加载 online/EMA；
* 检查所有 tensor 有限；
* 检查四臂 parameter keys/shapes/numel 一致；
* 检查 A/B/C/D 的 sample trace 完全一致；
* 检查 C/D 没有新增 observations/support；
* 独立重算 official mAP；
* 独立重跑 full-precision cross-window NMS；
* 拒绝任何 config whitelist 外差异；
* completion JSON 原子写入。

---

# 16. 20-epoch 停止条件

四臂均必须跑到 epoch 19；不能因某一臂早期差就只保留幸存者。

定义：

[
M_{\text{physical}}
=\frac{(B-A)+(D-C)}{2},
]

[
M_Q
=\frac{(C-A)+(D-B)}{2},
]

[
I_{Q\times physical}
=(D-B)-(C-A).
]

## 允许进入多 seed 的全部条件

1. `M_physical Avg-mAP >= +6.0pp`
2. `M_physical mAP@0.7 >= +4.0pp`
3. `D-B mAP@0.7 >= +3.0pp`
4. `D Avg-mAP >= B Avg-mAP - 0.5pp`
5. `D` 是最高 mAP@0.7 臂
6. `I_Q×physical mAP@0.7 >= +1.5pp`
7. `D-B` pre-NMS class-agnostic R@0.7 `>= +5pp`
8. timestamp shuffle/uniformization 至少消除 `D-C` 收益的 50%
9. 成本 gate 全通过
10. 无 protocol、artifact 或 evaluator 失败

## 立即终止 Q-lift 路线

满足任一项即终止：

* `D-B mAP@0.7 < +1.0pp` 且 pre-NMS R@0.7 `< +2pp`；
* `D Avg-mAP < B - 1.0pp`；
* Q384 成本超过准入线；
* Q384 的收益仅在 post-NMS 出现，而 pre-NMS recall 不变；
* 新架构使 `B-A Avg-mAP < +3pp`，说明 bridge 破坏了已验证的 physical mechanism。

## 降级为“通用 Q 密度工程”，终止物理方法主张

若：

[
|D-B-(C-A)|<1.0\text{pp},
]

且 timestamp shuffle 不削弱收益，则 Q384 改善只是通用 query-density 效应。此时：

* 可以保留为 ActionFormer 工程 ablation；
* 不得称为 physical-time 新方法；
* 不进入论文主路线。

---

# 17. 后续实验 DAG

```text
G0  静态审计
    ├─ config diff whitelist
    ├─ no F.interpolate
    ├─ identical state-dict shapes/hash
    └─ K/J/Q provenance

G1  单元与反事实测试
    ├─ Q identity / Q384 provenance
    ├─ padding tail
    ├─ timestamp shuffle
    └─ full-precision NMS

G2  真实 CUDA gate
    ├─ A/B/C/D 各 20 successful updates
    ├─ real train/test/evaluator
    ├─ gradient/AMP/EMA
    └─ p50/p95/memory

G3  四臂 20 epochs，seed 42
    ├─ final epoch 19 official mAP
    ├─ pre/post-NMS recall
    ├─ assignment diagnostics
    ├─ oracle score / oracle boundary
    └─ observability gap stratification

G4  四臂 20 epochs，seeds 3407、3408
    ├─ 共三 seeds：42/3407/3408
    ├─ paired seed statistics
    └─ nested video bootstrap

G5  四臂 60 epochs，三 seeds
    ├─ 12 个 formal runs
    ├─ exact 60-epoch scheduler
    ├─ full cost accounting
    └─ sealed final evaluation

G6  第二数据集
    ├─ ActivityNet-v1.3
    ├─ 同 K:J:Q 比例和固定 sampler
    ├─ 同四臂因子设计
    └─ 至少三 seeds

G7  paper gate
    ├─ mechanism survives
    ├─ cross-dataset survives
    ├─ cost acceptable
    └─ claim audit
```

60-epoch 多 seed 前，三 seed 20-epoch 平均效果必须满足：

* 所有 seed 的 `B-A` 为正；
* 所有 seed 的 `D-B mAP@0.7` 不低于 `-0.5pp`；
* 三 seed 平均 `D-B mAP@0.7 >= +3pp`；
* paired 95% interval 的下界大于 0。

---

# 18. 对十个问题的最终简明裁决

1. **K/J/Q**：dense 为 `768/384/768`、QΣ1512；旧随机为 `384/192/384`、QΣ756；当前两臂为 `384/192/192`、QΣ378。旧和 dense 都在 projection 前做 feature interpolation；当前没有。

2. **physical-metric 覆盖面**：GT、assignment、range、center sampling、regression、decode、clamp、秒转换和 NMS 均正确覆盖；backbone/projection 表示不覆盖。

3. **粗层中心与 stride**：是规则 rank cell 经分段线性秒轴映射后的正确近似；不是最终 feature receptive field 的精确物理支撑，且丢失左右不对称。

4. **VideoMAE/TIA/projection 是否看时间**：否。能力上限是纠正 metric 和 supervision，不能恢复缺失 RGB，也不能纠正前段 rank-domain mixing。

5. **旧 interpolation 是什么**：不是纯 query lift；同时改变 feature、感受野、attention state 数、pyramid、assignment density、候选覆盖和有效动态容量。

6. **如何诚实增加 Q**：保持 K384/J192 不变，单独构造 deterministic detection-query grid，通过 masked support cross-attention获得 query state；query 不计作 observation。

7. **优先路线**：稀疏 support 到 deterministic physical-query grid 的 cross-attention，之后复用 ActionFormer branch/head。共享 feature interpolation、纯复制 query 和全时间感知 projection 均不作为当前主方案。

8. **mAP@0.7 缺口来源**：当前只能证明 metric/assignment 很重要；剩余缺口尚不能在 Q、排序、回归、表示和 NMS间分解。四臂和 oracle replay 是决定性实验。

9. **57.57 是否可信**：可信。没有足以推翻它的隐藏 P0。需要修复 full-precision cross-window NMS，并纠正“整个采样无 GT”的过宽表述，但这两点不使现有物理对照失效。

10. **独立创新价值**：当前 head-only physical correction 本身不足以成为独立顶会方法；“support–token–query 三者解耦、无 dense evidence imputation 的物理时间 TAD”仍有研究价值，但只有四臂 interaction、多 seed、成本和第二数据集通过后才成立。

---

# 19. 最严厉审稿人攻击与可防守性

## 攻击一：“这只是复杂版 interpolation”

当前旧插值路线无法防守这一攻击。推荐方案只有同时满足以下证据才能防守：

* support tensor 从 J192 起不插值；
* C/D Q384 不增加 K/J；
* Q-independent parameter count；
* A/B/C/D 因子设计；
* shared interpolation 作为独立辅助 baseline；
* query coverage 和 null/no-evidence 状态显式披露；
* `I_Q×physical` 为正；
* timestamp counterfactual 能摧毁物理收益。

否则审稿人的指控成立。

## 攻击二：“只是给 ActionFormer 修坐标”

对当前 `57.57`，这个攻击基本成立：时间只在 head point geometry 中出现。

推荐方案把时间引入 support→query projection，位于 branch pyramid 和 head 之前，因而比当前方案更强。但由于 VideoMAE/TIA 仍是 rank-based，它只能把攻击从“纯 head patch”降为“partial time-aware projection”。要完全消除该攻击，必须证明：

* 物理 query interaction 在 matched 四臂中成立；
* 收益随不规则 gap 增强；
* timestamp shuffle 后消失；
* 不是单纯 Q 数量；
* 高 IoU 和短动作改善来自 pre-NMS proposal/assignment，而非 NMS 巧合。

## 攻击三：“Q384 只是更多计算和候选”

参数相同不能单独反驳，因为激活和候选容量确实增加。必须报告：

* C-A：通用 Q 收益；
* D-B：物理 Q 收益；
* interaction：物理时间是否让额外 query 更有效；
* wall-time、memory、QΣ 和 pre-NMS top-k；
* 固定 top-N replay，排除仅因更多候选进入 NMS。

## 攻击四：“缺失边界帧无法被时间戳恢复”

这个攻击正确。论文必须主动承认：

> 时间戳只描述观测几何，不恢复未观测视觉证据。

需要用 endpoint observability 分层说明在哪些 gap 下方法仍有效，在哪些 gap 下出现不可恢复上限。

## 攻击五：“continuous-time/query 不是新颖概念”

同样正确。RCL、TE-TAD 等已经占据连续 anchor、真实时间坐标和 query 邻域。推荐方案不能以“continuous-time TAD”作为主要新颖性。([arXiv][1])

可防守的新颖性边界只能是：

> 在固定、GT-free within-window 稀疏 RGB 观测下，严格分离 observation support、native feature tokens 和 detection queries，并以可审计的物理时间桥接完成候选生成，而不把 query 伪装成新增观测。

## 最终论文级裁决

* **当前实现**：可信的正机制结果，但仍是 coordinate-level correction；不足以成为独立方法论文。
* **推荐方案**：有机会避免“复杂版插值”质疑，但必须通过四臂 interaction、timestamp counterfactual 和全精度 proposal 诊断。
* **路线终止条件**：若 D-B 不改善高 IoU，或者 C-A 与 D-B 无差别，则应终止 PhysTime 独立方法路线，不得转而用更多 loss、NMS 调参或延长训练包装负结果。

[1]: https://arxiv.org/abs/2203.07112?utm_source=chatgpt.com "RCL: Recurrent Continuous Localization for Temporal Action Detection"
