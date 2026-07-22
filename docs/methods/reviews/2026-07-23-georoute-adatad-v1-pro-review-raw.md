# GeoRoute-AdaTAD v1 实施裁决

## 1. VISIBILITY_CERTIFICATE

```text
repository_access: ACCESSIBLE
repository: yuzbo/OpenTAD_C3_CoarseClean_20260702
branch: codex/spatial-zoom-s1-audit-fix-20260715
requested_commit: 8ebe5f069494dc2efb3d4f9dc1ea3a2fbb51f89c
resolved_branch_head: 8ebe5f069494dc2efb3d4f9dc1ea3a2fbb51f89c
branch_head_matches_requested_commit: true
snapshot_drift: false

research-wiki/ideas/geo-route-adatad.md: ABSENT_AT_COMMIT
docs/methods/2026-07-23-georoute-adatad-design-and-experiment-plan.md:
  ABSENT_AT_COMMIT

design_report_status: 设计报告未被 GitHub commit 固化
georoute_implementation_present: false
visibility_result: ACCESSIBLE_BUT_DESIGN_ONLY
```

仓库、分支和指定 commit 均可访问，且分支 HEAD 精确等于指定 SHA，不存在“实际最新分支实现已经变化”的情况。

但两个 GeoRoute 文件在该快照中都不存在。尤其是日期为 **2026-07-23** 的设计报告，在当前日期 **2026-07-22** 尚未进入指定 GitHub commit。本轮只能把用户消息中的 GeoRoute 描述当作外部设计提案，不能给它虚构 GitHub 行号，也不能把它写成已有实现。

### 实际展开阅读的仓库文件

以下范围是我在 `8ebe5f…` 上实际展开的源文件范围；较短文件在 EOF 处结束。

| 类别                                | 实际读取文件与范围                                                                                                                                                                                                                                                                                                                                                                                            |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 仓库规则与研究状态                         | `AGENTS.md:1-64`；`RTK.md:1-113`；`research-wiki/query_pack.md:1-181`；`research-wiki/anti_repetition.md:1-292`                                                                                                                                                                                                                                                                                         |
| 配置与数据通路                           | `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py:1-159`；`continuous_roi_s2_d160_videomae_s_768x1_adapter.py:1-72`；`continuous_roi_s2_g96_videomae_s_768x1_adapter.py:1-49`；`continuous_roi_s2_u128_videomae_s_768x1_adapter.py:1-142`；`configs/_base_/models/actionformer.py:1-43`；`opentad/datasets/transforms/native_crop.py:1-500`                                               |
| VideoMAE 与 ROI backbone           | `opentad/models/backbones/vit_adapter.py:1-260,430-1060`；`native_crop_wrapper.py:1-260`；`continuous_roi_wrapper.py:1-514`；`continuous_roi_geometry.py:1-320`；`continuous_roi_sampler.py:1-157`；`backbone_wrapper.py:1-320`；`builder.py:1-98`                                                                                                                                                         |
| Detector、projection、head、loss、NMS | `opentad/models/detectors/actionformer.py:1-255`；`single_stage.py:1-280`；`opentad/models/projections/actionformer_proj.py:1-186`；`opentad/models/dense_heads/actionformer_head.py:1-420`；`anchor_free_head.py:1-520`；`opentad/models/losses/focal_loss.py`；`iou_loss.py:1-220`；`opentad/models/utils/post_processing/nms/nms.py:1-260`                                                               |
| 训练和 focused tests                 | `opentad/cores/train_engine.py:1-360`；`tools/bata/run_continuous_roi_s2_one_step_gate.py:130-760`；`tests/test_continuous_roi_geometry_sampler.py:1-260`；`test_continuous_roi_representation.py:1-220`；`test_continuous_roi_s2_one_step_gate.py:1-260`；`test_continuous_roi_source_views.py:1-180`；`test_continuous_roi_s2_implementation_static.py:1-140`；`test_continuous_roi_s2_training.py:1-520` |

外部参照实际阅读了 Uni-AdaFocus 论文及官方仓库中的 `models/uni_adafocus.py`、`opts.py`，以及 A-MoD、MoD、ToMe、DynamicViT、TokenSelect 的原始论文或官方项目页。

---

## 2. 唯一裁决

# **HOLD**

含义不是否定研究假设，而是：

> **冻结所有完整训练、A-MoD 集成和论文主张；只授权一个 P0 原生 token 单重主干垂直切片。**

不能给 `ACCEPT_WITH_FIXES`，因为指定快照中没有 GeoRoute 代码，而且当前 S2 实现与拟议模型存在四个根本差异：

1. 当前 U128 使用固定 `128×128` 的 `grid_sample`，不是原生 patch；
2. 当前 U128 每个样本执行两次 VideoMAE；
3. 当前几何来自外生 common-support/anchor，不含 learned policy；
4. 当前 ActionFormerHead 只有分类和 DIoU 回归，没有独立 quality loss。

也不应立即 `KILL`，因为存在一种不引入第二次重主干、且数学上诚实的训练方式：**训练时对硬路由使用 score-function detector-loss estimator，部署时使用确定性 exact-K**。但这必须先通过 P0 parity、梯度和成本门，不能先排完整 detector training。

仓库本身也明确要求区分 `designed / implemented / tested / empirically_supported`，并明确记载当前 S2 exact-nine receipt 只证明训练完整性，不是 development mAP、成本可行性、learned ROI 或论文证据。

---

## 3. 当前代码的独立事实链

### 3.1 当前 Continuous ROI 不是 GeoRoute

U128 配置把 source 固定为 `180×320`，另建 `96×96` 全局视图，并注册固定 `128×128` local view；同时明确关闭 learned crop policy、official test 和 paper claim。

`continuous_roi_sampler.py:59-146` 对每个 16-frame clip 复制同一个 box，并通过：

```python
F.grid_sample(
    source_chunk,
    grid,
    mode="bilinear",
    padding_mode="zeros",
    align_corners=False,
)
```

输出固定局部尺寸。因此它是连续 ROI 重采样，不是原生 `2×16×16` patch 选择。

`ContinuousRoiBackboneWrapper.forward()` 先编码 global view，再采样 local view，再编码 local view。它自己的审计记录明确写着：

```text
videomae_evaluations = 2
contains_selector = False
policy_head_parameters = 0
```

正式 one-step Gate 同样强制检查 `videomae_evaluations == 2`、`policy_head_parameters == 0`，并拒绝任何名为 selector/ROI-policy 的参数。因此该 Gate 证明的是 global/local 特征、adapter、fusion、projection 和 RPN head 的图连通，不是 learned ROI policy 获得 detector 梯度。

### 3.2 当前 VideoMAE 不支持原生 packed sparse token

`vit_adapter.py:782-793` 的 patch embedding 是无 padding、步幅等于 kernel 的：

```python
Conv3d(
    in_channels=3,
    out_channels=384,
    kernel_size=(2, 16, 16),
    stride=(2, 16, 16),
)
```

所以一个 token 精确覆盖相邻两帧中同一绝对空间 patch。

但当前 forward 在任何路由之前就执行完整 dense patch embedding，并基于完整矩形 `h×w` 网格插值位置编码。最终 `return_feat_map` 也要求把 token 重新 reshape 成完整矩形。

现有 temporal adapter 也把 token reshape 成：

```text
[..., temporal_size, h, w, channels]
```

然后在每个固定绝对空间位置上运行 temporal Conv1d。任意 packed token 数和随时间改变的空间成员会直接破坏这一假设。

### 3.3 Detector 接口可保留，但“质量损失”不存在

原始 dense AdaTAD-derived 配置将 768 帧分成 48 个 16-frame clip，经 VideoMAE 得到 384 个 tubelet 特征，再线性插值回 `[B,384,768]` 后送入 projection。

Projection 本身只要求 dense temporal tensor `[B,C,T]` 和 mask，并继续生成 ActionFormer feature pyramid；因此只要新 backbone 严格输出 `[B,384,768]`，projection 可以保持不变。

但当前 head 配置只有：

```python
cls_loss = FocalLoss
reg_loss = DIOULoss
```

没有 quality branch 或 quality loss。

实际 `AnchorFreeHead.losses()` 也只返回 `cls_loss` 和 `reg_loss`。因此“保持 ActionFormerHead 和 losses 不变”与“以分类、边界回归、质量三损失训练 policy”不能同时成立。

**v1 必须删除独立 quality-loss 表述。** 若新增 quality head，则必须停止声称 head/loss unchanged，并重新做 detector baseline。

---

## 4. 十项核心问题裁决

| 级别        | 核心问题                   | 代码事实与可复现理由                                                                                                                                                                                                                                | 唯一最小修复                                                                                                                                               |
| --------- | ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0**    | 1. 可微性诚实性              | 连续 ROI 到整数 native patch exact-K 的支持集是分段常数。hard gather 对 `(cx,cy,w,h)` 没有路径导数；普通反传只更新已选 token 内容通路。当前 finite-difference test 验证的是 `grid_sample` 的像素插值梯度，不适用于 native exact-K。`soft warm-up → hard STE` 的 backward 不是 hard 模型梯度，且无法正确评价未选成员。 | soft dense gate 只作 warm-up；正式 sparse 阶段用单样本 score-function estimator。STE 只能保留为标注为 biased 的诊断臂。                                                       |
| **P0**    | 2. 真正原生裁剪              | 当前 U128 固定 `180×320`、固定 local128、bilinear `grid_sample`；dense160 baseline 更直接对整图 resize/crop。                                                                                                                                             | 以动态 `H_b,W_b` 的 CPU uint8 source list 为输入；在 Conv3d 前 gather 原始 `[3,2,16,16]` patch；禁止 sparse 路径出现 `grid_sample`、全量 patch embed 或 source padding。     |
| **P0**    | 3. 时空 token 语义         | Conv3d kernel/stride 为 `(2,16,16)`。若两帧分别使用不同空间位置，就不再是 pretrained VideoMAE 的 tubelet。                                                                                                                                                      | 每个 tubelet 只允许一个离散空间成员集合，供其两帧共同使用；48 个连续几何 knot 插值到 384 个 tubelet box，而不是 768 个逐帧 box。                                                               |
| **P0**    | 4. AdaTAD 完整性          | `[B,384,768]` 可以保留 projection/head/NMS；但 hard membership 没有路径梯度，且 head 无 quality loss。现有 Gate 只证明已选分支特征梯度。                                                                                                                                | 保持现有 Focal+DIoU、projection、target assignment、NMS 数学不变；只增加 detached per-sample detector-cost 输出供 policy gradient。表述改为 `AdaTAD-derived detector path`。 |
| **P0**    | 5. 稀疏 adapter          | `vit_adapter.py:21-77` 和 `908-977` 假设完整矩形网格；直接把 packed tokens 塞进去会 reshape 错误或混淆绝对位置。                                                                                                                                                     | 新增 absolute-coordinate scatter 的 bottleneck sparse adapter；full-K 时必须数值退化为现有 adapter。v1 不引入 deformable temporal alignment。                           |
| **P1**    | 6. ROI 与 free selector | 当前代码没有任何 learned selector。ROI 的结构优势尚无证据；连续框也可能仅约束模型去选择一个大致规则的区域，而自由 scorer 可能更适合多主体、背景上下文和边界线索。                                                                                                                                           | 同预算比较 random、ROI-only、free、ROI+residual；若 hybrid 不能可靠胜过 free，删除 ROI 主贡献并改名为 native TokenSelect route。                                                |
| **P2**    | 7. A-MoD               | 当前 attention 使用 `F.scaled_dot_product_attention`，不返回 attention map；不能直接得到前层列均值。                                                                                                                                                           | 先做显式 attention KAT，再实现 fused column-sum kernel；不能保留 fused attention 或没有真实加速时，从主模型删除 A-MoD。                                                           |
| **P2**    | 8. ToMe/merging        | ToMe 在 Transformer token 已生成后逐步合并相似 token，不减少完整 decode、source H2D 或初始 patch projection，因而不是 pre-backbone native selection 的等价对照。([arXiv][1])                                                                                              | 报告 total wall/energy 和“backbone 内部 token FLOPs”两列；不得只用后者宣称与 GeoRoute 等价。                                                                             |
| **P1/P3** | 9. 真实成本                | 当前 Gate 自己把 latency 标成 diagnostic，并明确禁止 paper latency claim。                                                                                                                                                                              | 强制 full-stack ledger；P0 先证明 packed heavy path 有 headroom，P1 再证明 end-to-end 真实收益。                                                                     |
| **P0/P3** | 10. 实验完整性              | 当前 S2 是 training-only exact-nine receipt；`test=None`、learned policy 禁止、无 dev mAP 和成本。                                                                                                                                                     | development-only 完成 P0–P2；final EMA only；official test 只在冻结 P3 条件满足后单次开启。历史 S2 receipt 不得并入 GeoRoute 结果。                                             |

---

## 5. 外部方法核查

### 5.1 Uni-AdaFocus：可借鉴的是连续几何与平滑，不是其训练合同

Uni-AdaFocus 确实使用轻量 global encoder、连续 patch 中心，并在 deformable 版本中输出中心、宽和高。但论文也明确说明，所有可变大小 patch 会被 resize 到同一个局部输入尺寸，接受几何失真以便并行计算。([arXiv][2])

其 end-to-end 可微性来自 bilinear interpolation。论文随后又指出，像素插值产生的 policy 梯度过于局部、语义含糊，因此最终用 global deep-feature patch 的分类代理来更新 policy，而不是简单地让 local heavy path 的最终损失直接决定 policy。([arXiv][2])

论文还采用 stop-gradient 稳定 global encoder/policy 关系。([arXiv][2])

官方代码中的关键含义是：

* `num_segments = 48`；
* `num_glance_segments = 16`；
* policy 在 16 个 global observation 位置产生动作；
* 动作被插值到 48 个 segment 位置；
* local crop 动作在送入重模型前 detach；
* crop 使用固定输出尺寸的 `affine_grid/grid_sample`。

因此 **“16 observation / 48 segment”不是“每 16 帧一个 ROI”**。它是 48 个候选 segment 时间位置中的 16 个 global observation，再将连续动作插值到完整 48-position timeline。

GeoRoute 可以借鉴：

* `(cx,cy,w,h)` 的 in-bounds 参数化；
* knot interpolation；
* trajectory smoothness；
* geometry anti-collapse。

不能原样迁移：

* 固定 local resize；
* classification proxy 作为 TAD detector utility；
* detach 后宣称重 detector 对 hard support 提供直接路径梯度；
* dynamic frame sampling，因为本任务必须保留完整 768 时间轴。

### 5.2 A-MoD：必须是 paper-exact reproduction

A-MoD 的定义是使用**紧邻前一层**的完整 attention，将各 head、各 query row 对同一 key/token 的注意力做平均，即 attention 的列均值，作为下一 MoD 层的 token importance。未选 token 走 identity bypass；与标准 MoD 不同，A-MoD 不把 routing score 乘回 block 输出。([arXiv][3])

论文采用 Dense/MoD 交替，而不是后半段连续 all-MoD。([arXiv][3])

FlashAttention/SDPA 下，论文建议在 tile 内累计 token 列统计，避免物化完整 `N×N` attention map；这需要 attention kernel 级实现，不是一个 Python hook。([arXiv][3])

A-MoD 的 score 无额外参数，并不意味着 hard top-k 排序获得了完整直接梯度。hard membership 仍不可微；attention 参数主要通过前层正常 attention 输出被训练，不能夸大为“detector loss 直接穿过 hard rank”。

截至 **2026-07-22**，我在论文页、作者/机构项目页和 GitHub exact-title/author 检索中未找到作者公开的官方实现；项目页面只提供论文入口，第三方索引显示 “Request Code”。因此必须写：

> **paper-exact reproduction**

不得写：

> official A-MoD code alignment。([Collaborative AI][4])

### 5.3 MoD、ToMe、DynamicViT 与 free TokenSelect 的公平边界

MoD 的核心是每个 routing block 固定 capacity `k`，仅让 top-k token 执行 attention/MLP，其余走 residual identity；它是在模型深度内部减少计算，不是从 decoded source 前删除空间 patch。([arXiv][5])

DynamicViT 在 tokenization 后逐层用 predictor 和 differentiable attention mask 删除 token；它是合理的 post-embedding learned pruning 对照，但仍支付完整 patch embedding 和进入首个 pruning 点之前的计算。([arXiv][6])

“TokenSelect”这个正式方法名目前对应的是 LLM 长上下文中的 KV-cache token selection，而不是视频原生 patch 选择。GeoRoute 实验中的 `free TokenSelect` 必须被操作性定义为：

> **同 scout、同 decoded source、同 K、同训练更新数的自由 native-patch scorer baseline**

不得错误引用 LLM TokenSelect 作为视觉实现来源。([arXiv][7])

---

## 6. 冻结的 GeoRoute-AdaTAD v1 最小模型

```text
decoded uint8 source list: V_b ∈ [3,768,H_b,W_b]
        │
        ├─ global96 letterbox ─> light scout
        │                         ├─ 48 geometry-knot distributions
        │                         └─ 384 × H_p × W_p free-token logits
        │
        └─ sampled geometry + exact-K indices
                                  │
                    CPU pinned native tubelet gather
                                  │
                 raw [3,2,16,16] patches, no resize
                                  │
              equivalent linearized VideoMAE PatchEmbed
                                  │
             exact-K packed tokens + gathered abs position
                                  │
            12-block VideoMAE, one heavy forward only
              + absolute-coordinate sparse adapters
                                  │
                    K-token masked spatial mean
                                  │
                       [B,384,384]
                                  │
                deterministic temporal 2× interpolation
                                  │
                       [B,384,768]
                                  │
         unchanged projection → FPN → ActionFormerHead
                    → Focal + DIoU → existing NMS
```

### 6.1 输入和 native grid

每个样本保留独立动态 source tensor：

```text
V_b: uint8 [3, 768, H_b, W_b]
```

不再把不同 `H_b,W_b` 的 source pad 成统一矩形 batch，而是在 CPU 侧保留 list。

原生 patch grid 为：

[
T_p=768/2=384,\qquad
H_p=\lfloor H_b/16\rfloor,\qquad
W_p=\lfloor W_b/16\rfloor .
]

右侧和底部不足 16 像素的 remainder 与当前无 padding Conv3d 一致地忽略，不增加零 padding。协议前置条件冻结为：

```text
H_p >= 6
W_p >= 10
H_p * W_p >= 64
```

不满足的样本是 preflight failure，不通过 padding 或 resize 偷偷修复。

### 6.2 轻量 scout

固定输入：

```text
global_view: uint8 [B,3,768,96,96]
```

scout 只负责路由，不能作为 detector global branch。冻结结构：

```text
Conv3d 3→32, kernel=(2,7,7), stride=(2,4,4)
DWConv3d + PWConv 32→64, spatial stride 2
DWConv3d + PWConv 64→96, spatial stride 2
2 × temporal depthwise-conv residual block
```

输出：

```text
S_map: [B,384,96,6,6]
S_temporal: [B,384,96]
```

每 8 个 tubelet 聚合成一个 clip feature，得到 48 个 knot features。geometry head 输出：

```text
mu:        [B,48,4]
log_sigma: [B,48,4]
```

48 个 knot 在线性插值后形成 384 个 tubelet box。这里的 48 来自当前 `48 × 16-frame clip` 计算合同，不冒充 Uni-AdaFocus 的 16/48 语义。

### 6.3 几何参数化

在 patch-cell 坐标中参数化：

[
w_c=10+(W_p-10)\sigma(q_w),\qquad
h_c=6+(H_p-6)\sigma(q_h),
]

[
c_x=\frac{w_c}{2}+(W_p-w_c)\sigma(q_x),\qquad
c_y=\frac{h_c}{2}+(H_p-h_c)\sigma(q_y).
]

这保证 box in-bounds，并保证 6×8 ROI canonical grid 有足够空间展开。

在每个 box 内放置 6×8 个连续 canonical target，共 48 个。对 native patch center (p)，定义：

[
s_{\rm roi}(p)=
\log\sum_{j=1}^{48}
\exp!\left(
-\frac{|p-q_j(b_t)|_2^2}{2(0.35)^2}
\right).
]

这比单纯按 normalized center-distance 排序更合适，因为 box 的绝对宽高改变 canonical targets 的展开范围，`w,h` 不会因 exact-K 而完全失去作用。

### 6.4 exact-K 分配

冻结：

```text
K = 64 native spatial patches / tubelet
```

它与历史 local128 的 `8×8=64` spatial token 数对齐，但不进行 local128 重采样。

四个固定 context anchors 取最接近归一化位置：

```text
(0.25,0.25), (0.75,0.25), (0.25,0.75), (0.75,0.75)
```

所有路由臂必须保留这 4 个 context token，并在其余候选中排除重复。

| 路由臂               |                       每个 tubelet 的 exact-K 构成 |
| ----------------- | --------------------------------------------: |
| random            |                  4 context + 60 seeded random |
| ROI-only          |                  4 context + ROI score top-60 |
| free              |                 4 context + free score top-60 |
| **GeoRoute main** | 4 context + ROI top-48 + free residual top-12 |

free scorer 使用相同 scout feature：

```text
upsample(S_map[t]) to [H_p,W_p]
concat normalized (x,y)
1×1 MLP → u_free[t,y,x]
```

free scorer不读取 ROI-relative 坐标，确保 free baseline 真正独立；hybrid 只在选择 residual 时排除已经占用的 ROI/context token。

stable tie-break 固定为原生线性索引：

```text
index = y * W_p + x
```

### 6.5 原生 patch gather 与位置编码

hard sparse 路径不得先调用完整 `PatchEmbed.forward()`。

逻辑 raw packed shape：

```text
[B, 48, 8, 64, 3, 2, 16, 16]
```

实现中压成：

```text
[B*48, 512, 3*2*16*16]
```

现有 Conv3d weight reshape 后用 `F.linear`：

[
z = \operatorname{Linear}
\bigl(\operatorname{vec}(V_{t:t+2,y:y+16,x:x+16})\bigr).
]

这与 kernel=stride 的无 padding Conv3d 对单个 patch 数学等价。

位置编码必须先按实际 `H_p,W_p` 执行现有空间插值，再按 `(tubelet,y,x)` gather；禁止按 packed ordinal 0…511 赋位置。

### 6.6 sparse temporal adapter

v1 不采用 deformable temporal alignment。最小 adapter：

1. packed token `384→96` down projection；
2. 加入 zero-init geometry embedding：

   * absolute `t/383, x/W_p, y/H_p`；
   * ROI-relative `dx/w, dy/h`；
   * `log w, log h`；
3. bottleneck feature scatter 到：

   ```text
   [B,384,H_p,W_p,96]
   ```

   并生成 binary mask；
4. 对每个绝对空间 cell 做 kernel-3 depthwise temporal Conv1d；
5. 缺失邻居用 support-count correction：
   [
   y=\operatorname{Conv}(x\odot m)
   \frac{3}{\max(1,\operatorname{CountConv}(m))}
   ]
   并加 zero-init mask embedding；
6. gather 回 packed token，`96→384` up projection，残差相加。

当 full-K、mask 全一、geometry/mask 新增参数为零时，该 adapter 必须退化为现有 dense adapter。这是 P0 full-K parity 的必要条件。

### 6.7 Detector 输出

第 12 block 后：

```text
packed tokens: [B,384,64,384]
masked mean over K
→ [B,384 time,384 channel]
→ transpose [B,384 channel,384 time]
→ deterministic_linear_2x
→ [B,384,768]
```

其后保持：

```text
Conv1DTransformerProj
FPNIdentity
ActionFormerHead
FocalLoss + DIOULoss
existing soft-NMS
```

不保留当前 Continuous ROI 的 global/local fusion、global auxiliary head、local auxiliary head。

---

## 7. 梯度估计、损失和冻结训练日程

### 7.1 对 STE 的裁决

```text
soft dense gate warm-up → hard exact-K straight-through fine-tune
```

**不足以作为正式“真实 detector-gradient 学习”证据。**

原因：

* STE backward 对应人为指定的 soft surrogate；
* hard forward 的成员变化不服从该导数；
* 未选 token 的反事实效用没有被真实计算；
* exact-K 排序边界附近的梯度高度依赖温度和 surrogate。

因此 STE 仅允许作为 `[BIASED_DIAGNOSTIC]`，不进入主模型证据。

### 7.2 正式 estimator

#### 阶段 A：dense soft-gate warm-up

使用完整 native patch grid，但仍只有一次 VideoMAE forward。对每个 tubelet 求 soft capacity gate：

[
g_i=\sigma((s_i-\lambda)/\tau),
\qquad
\sum_i g_i=60,
]

其中 (\lambda) 用固定 24 次 bisection 求解；4 个 context gate 固定为 1。

gate 作用于 block 输入和最终 spatial pooling。该阶段产生的是**soft surrogate 的 pathwise detector gradient**，不做 sparse-cost 声明。

#### 阶段 B：hard stochastic exact-K

ROI knot 使用 diagonal Gaussian：

[
a_j\sim\mathcal N(\mu_j,\sigma_j^2)
]

再经前述 bounded transform 得到 box。

free residual 使用 Gumbel top-k 采样，其 ordered sample 按 Plackett–Luce 计算精确 log probability。

每个视频只执行一次 hard packed VideoMAE。正式 detector-policy loss：

[
A_b=
\operatorname{clip}
\left(
\frac{L_{{det},b}-b_\phi(z_b)}
{\operatorname{EMAStd}(L_{det})+\epsilon},
-5,5
\right),
]

[
L_{pg}=
\frac{1}{M}
\sum_b
\operatorname{stopgrad}(A_b)
\log p_\theta(a_b\mid z_b),
]

其中 (M) 是固定的随机决策数，只改变整体梯度尺度。

这是对**随机 hard policy 期望 detector loss** 的 score-function estimator；它不是 deterministic exact-K gather 的路径导数。该表述必须保持。

### 7.3 Detector cost

在 `anchor_free_head.py:286-410` 增加可选 per-sample instrumentation：

```text
policy_cls_cost[b]
policy_reg_cost[b]
policy_detector_cost[b]
```

它们必须使用与正式 aggregate loss 完全相同的：

* valid masks；
* focal 项；
* DIoU 项；
* 全局 detached loss normalizer；
* 全局 detached regression loss weight。

正式 aggregate `cls_loss/reg_loss` 的位级数学不得变化。

### 7.4 总损失

[
L_{\rm det}=L_{\rm focal}+L_{\rm DIoU},
]

[
L_{\rm total}
=============

L_{\rm det}
+L_{\rm pg}
+0.5L_{\rm critic}
+0.02L_{\rm velocity}
+0.01L_{\rm acceleration}
-\lambda_H H(\pi).
]

无独立 quality loss，无 GT spatial box，无 teacher/oracle ROI，无 spatial classification proxy。

### 7.5 4,800 successful-update 日程

当前冻结 S2 训练合同为 60 epochs、每 epoch 80 successful updates、共 4,800 updates、final EMA only；GeoRoute matched arms沿用这一更新单位。

|   Successful updates | 模式                      | 关键设置                                                                                     |
| -------------------: | ----------------------- | ---------------------------------------------------------------------------------------- |
|                0–799 | dense soft gate         | gate temperature `2.0→0.5`；所有 sparse arms使用对应 soft mask                                  |
|             800–3199 | stochastic hard exact-K | geometry std floor `0.08→0.03`；PL temperature `1.0→0.3`；entropy coefficient `0.01→0.002` |
|            3200–4799 | low-noise hard exact-K  | geometry std `0.03`；PL temperature `0.3→0.1`；entropy `0.002→0`                           |
| validation/inference | deterministic           | geometry用 `mu`；free 使用稳定 top-k；无采样、无 GT                                                  |

VideoMAE core继续冻结，只有 adapter、新 scout/router、projection、neck、head训练；optimizer schedule、EMA 和 AMP retry 都按 successful optimizer update 推进。

---

## 8. Dense–MoD 精确层级调度

A-MoD **不进入 P0/P1 主模型**。它是 P2 条件扩展。

12 个 block 的唯一合法调度：

```text
Block 0: Dense
Block 1: Dense
Block 2: Dense
Block 3: Dense
Block 4: Dense  ── attention-column statistic ──> Block 5: A-MoD
Block 6: Dense  ── attention-column statistic ──> Block 7: A-MoD
Block 8: Dense  ── attention-column statistic ──> Block 9: A-MoD
Block10: Dense  ── attention-column statistic ──> Block11: A-MoD
```

禁止：

```text
Dense 0-5 → MoD 6-11
```

也禁止 MoD7 读取 Dense4 或 MoD5 的残缺 attention。

packed clip 中：

```text
N_valid = 8 tubelets × 64 spatial tokens = 512
```

容量冻结：

```text
C=1.00 KAT: k=512
C=0.75 production candidate: k=384
```

规则：

* padding 不计入 `N_valid`；
* 每个 MoD 精确处理 `k` 个 token；
* stable tie-break 用 native packed provenance；
* 未选 token bitwise identity bypass；
* 不把 A-MoD score 乘回输出；
* 每个 score 必须来自紧邻前一 Dense block 的完整 attention；
* A-MoD 不被称为 detector hard-rank 的直接梯度。

### A-MoD KAT

必须同时通过：

1. `C=1` 输出、loss、梯度 parity；
2. exact capacity 与 stable tie；
3. unselected identity bitwise parity；
4. random exact-K MoD；
5. linear-router MoD；
6. 显式 attention 列均值与 fused statistic：

   ```text
   max_abs <= 1e-5 fp32
   ```
7. provenance：

   ```text
   M5←D4, M7←D6, M9←D8, M11←D10
   ```
8. production profile 中禁止 materialize `N×N` attention。

无法保持 fused attention，或者 A-MoD 的总延迟没有实测收益时，直接从主模型删除，不再增加 workaround。

---

## 9. 最小代码改动

### P0 必需

| 优先级  | 文件                                                                | 修改                                                                                               |
| ---- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| P0-1 | 新建 `opentad/datasets/transforms/georoute.py`                      | 动态 source list、global96、无 source resize/pad、原生几何 ledger                                          |
| P0-2 | 新建 `opentad/models/backbones/native_token_gather.py`              | CPU pinned raw tubelet gather、Conv3d-equivalent linear patch projection、absolute position gather |
| P0-3 | 新建 `opentad/models/backbones/georoute_wrapper.py`                 | scout、geometry/free policy、exact-K、one-forward audit、384→768 输出                                  |
| P0-4 | `opentad/models/backbones/vit_adapter.py:21-77,782-997`           | 增加 packed forward 和 sparse adapter；保留原 dense path不变                                              |
| P0-5 | `opentad/models/builder.py:1-43`                                  | 注册 `georoute_native_token_v1` wrapper                                                            |
| P0-6 | `opentad/models/dense_heads/anchor_free_head.py:286-410`          | 只增加 detached per-sample policy cost；不得改 aggregate loss                                           |
| P0-7 | `opentad/models/detectors/actionformer.py:126-214`                | 消费 policy loss 和 audit；projection/head顺序不变                                                       |
| P0-8 | 新建 `configs/adatad/thumos/georoute_adatad_v1_k64.py`              | 冻结 K、schedule、trainable params、sealed test、cost schema                                           |
| P0-9 | `opentad/models/backbones/__init__.py`、dataset transform registry | 只做新模块导入                                                                                          |

### P0 focused tests

```text
tests/test_georoute_native_patch_projection_parity.py
tests/test_georoute_full_k_detector_parity.py
tests/test_georoute_exact_k_and_tubelet_semantics.py
tests/test_georoute_policy_gradient_known_answer.py
tests/test_georoute_one_heavy_forward.py
tests/test_georoute_no_grid_sample_no_dense_patch_embed.py
tests/test_georoute_dynamic_source_geometry.py
tests/test_georoute_cost_schema.py
```

### P2 才允许

| 文件                                                   | 修改                                             |
| ---------------------------------------------------- | ---------------------------------------------- |
| `vit_adapter.py:462-542`                             | dense attention column-statistic接口、A-MoD block |
| 新建 `opentad/models/backbones/amod_attention_stat.py` | explicit KAT 与 fused kernel wrapper            |
| `tests/test_georoute_amod_kat.py`                    | C=1、capacity、identity、provenance、fused parity  |

不修改 `actionformer_proj.py:126-172`、ActionFormer target assignment、Focal/DIoU 数学和 NMS。

---

## 10. 唯一 P0 → P1 → P2 → P3 门序

## P0：实现正确性与单重主干门

基线：

```text
D160 original
native-dense full-grid one-pass
GeoRoute full-K parity mode
random exact-K64 vertical slice
```

必须通过：

1. CPU float64 selected patch projection：

   ```text
   max_abs <= 1e-10
   ```
2. CUDA fp32 Conv3d/linear selected-token parity：

   ```text
   max_abs <= 2e-5
   ```
3. full-K feature/logit/loss：

   ```text
   max_abs <= 2e-5
   ```
4. 共享参数梯度：

   ```text
   cosine >= 0.99999
   ```
5. 每个 tubelet：

   ```text
   exactly 64 unique native coordinates
   ```
6. 两帧共享同一 patch 坐标；
7. projection input 精确 `[B,384,768]`；
8. `videomae_evaluations == 1`；
9. sparse runtime graph 无 `grid_sample`、无 dense patch embedding、无 `180×320` 判断；
10. 小型可枚举 toy policy 上，Monte Carlo estimator 对 exact expected gradient：

    ```text
    cosine >= 0.95
    relative_norm_error <= 0.15
    ```
11. K64 isolated heavy path p50：

    ```text
    <= 0.75 × D160 heavy path
    ```
12. 完整 vertical slice p50：

    ```text
    <= 1.05 × D160 full path
    ```

任一 full-K parity、tubelet semantics、单重主干或 estimator KAT 失败：

# **KILL GeoRoute**

不得用完整训练补救。

## P1：matched-budget development-only 科学门

六臂，三种子，均从同一初始化开始：

```text
D160
native-dense full-grid
random64 = context4 + random60
ROI-only64 = context4 + ROI60
free64 = context4 + free60
hybrid64 = context4 + ROI48 + free12
```

四个 sparse 臂必须同：

* decoded source；
* scout；
* K；
* context allowance；
* update count；
* optimizer；
* augmentation；
* AMP；
* final EMA；
* development split。

通过真实效率门：

```text
hybrid mean Avg-mAP drop vs D160 <= 0.30 absolute points
no single-seed drop > 0.60
full-stack p50 latency <= 0.85 × D160
full-stack p95 latency <= 0.90 × D160
energy/video <= 0.90 × D160
```

学习有效性门：

```text
max(free, hybrid) - random >= 0.50 Avg-mAP
```

ROI 主张门：

```text
hybrid - free >= +0.30 Avg-mAP mean
and hybrid > free in at least 2/3 seeds
and cost difference <= 2%
```

裁决：

* `hybrid ≥ free +0.30`：保留 GeoRoute；
* `|hybrid-free| <0.30`：ROI 无独特证据，删除 ROI 主贡献，路线收缩并改名 `NativeTokenSelect-AdaTAD`；
* `free ≥ hybrid +0.30` 且 2/3 seeds成立：**KILL GeoRoute ROI claim**；
* learned arms不能胜 random：**KILL learned routing**；
* 无真实总成本收益：**KILL efficiency claim**。

## P2：A-MoD 条件门

仅使用 P1 获胜的路由模型。

四臂：

```text
packed all-dense
random exact-K MoD, C=0.75
linear-router MoD, C=0.75
A-MoD, C=0.75
```

通过条件：

```text
全部 KAT PASS
A-MoD vs packed all-dense:
  total p50 latency improvement >= 5%
  packed-backbone latency improvement >= 8%
  Avg-mAP drop <= 0.20

A-MoD vs linear MoD:
  Avg-mAP gain >= 0.20
  or same accuracy with lower measured routing overhead
```

失败只删除 A-MoD，不反向篡改 P1 GeoRoute 结果。

## P3：冻结证据与 official-test 门

只有 P1 route 和可选 P2 extension 全部冻结后才能进入：

* immutable commit/config/checkpoint；
* 三种子 development 完整结果；
* final EMA only；
* 无 best-val checkpoint selection；
* 阈值和 NMS 不根据 official test 调节；
* official test 单次开启；
* raw predictions 不供 policy 使用；
* 完整成本和能耗 ledger；
* 公开 negative/random/free 对照。

若 official test 前发生配置、K、capacity、checkpoint 或 claim 改动，必须重新封存，旧 test 结果不得拼接。

---

## 11. 必须保持不变的合同

1. 任务始终是 **offline full-window TAD**；
2. 保留完整 768 detector 时间轴，不做 frame selection；
3. VideoMAE tubelet 始终是原生 `2×16×16`；
4. detector backbone output 始终为 `[B,384,768]`；
5. projection、FPN、ActionFormer target assignment、Focal、DIoU、NMS 数学不变；
6. VideoMAE pretrained core继续冻结，adapter保持当前训练权限；
7. 同一 THUMOS split、window overlap、augmentation 和三种子；
8. training-only GT 只进入 detector loss；
9. validation/test 禁止 GT、teacher、oracle、raw-prediction cache；
10. final EMA only；
11. official test 在 P3 前保持 sealed；
12. matched sparse arms必须运行同一个 scout，random arm也不能省略 scout 成本；
13. dense baseline按自身真实通路计费，不能人为加 scout；
14. hardware、batch、AMP、CUDA、PyTorch、clock/power setting固定。

---

## 12. 强制成本 ledger

每个 arm 都必须分项记录：

```text
video decode
CPU source validation
global96 letterbox
scout H2D
scout forward
geometry/free router
Gaussian / Gumbel / PL sampling
top-k / stable sort
GPU→CPU index copy
synchronization
CPU native patch gather
pinned packed-buffer construction
packed H2D
linearized patch embedding
position/geometry embedding
VideoMAE attention
VideoMAE MLP
sparse-adapter scatter
mask-count correction
sparse-adapter gather
spatial pooling
384→768 interpolation
projection
FPN
ActionFormer head
prediction D2H
window aggregation
NMS
all explicit synchronizations
peak allocated memory
peak reserved memory
energy/video
```

必须同时报告：

```text
p50 / p95 wall latency
total energy
peak memory
internal token FLOPs
```

不能用 token FLOPs 替代 total cost。

最可能令 sparse route 反而更慢的项包括：

* scout 后的 GPU→CPU index synchronization；
* CPU 非连续 raw patch gather；
* pinned buffer 构建；
* Gumbel/PL top-k 与排序；
* packed shape 导致的 kernel 利用率下降；
* scatter/gather 和小 kernel 启动；
* sparse adapter 的 bottleneck dense lattice；
* dynamic source shape 导致 graph break；
* batch 内 padding或分桶；
* 无法继续使用 fused attention；
* full video decode 和 NMS 成为新的主瓶颈。

仓库研究合同本身也要求将 decode、预处理、H2D、selector、backbone、head、后处理、显存和能耗全部纳入真实成本。

---

## 13. 最窄且诚实的论文 claim

### 允许

> 我们研究一种离线 TAD 空间计算分配方法，在保持 768-step AdaTAD-derived detector 接口的前提下，于单次 VideoMAE 前向之前，从 decoded-source native `2×16×16` tubelets 中选择固定数量 token。连续 ROI 仅作为结构化选择先验，并在相同输入、scout、预算和训练合同下与自由 native-token selection 比较。

若 score-function 门通过，可以写：

> The router is optimized with a one-sample score-function estimator of the expected hard-policy detector loss.

### 禁止

* “deterministic exact-K gather 获得真实 pathwise detector gradient”；
* “STE 等价于 hard selector 梯度”；
* “保持 official AdaTAD 完全不变”；
* “质量损失保持不变”，因为当前没有 quality loss；
* “camera-original resolution”，只能写 decoded-source-native；
* “Uni-AdaFocus official implementation alignment”；
* “A-MoD official-code alignment”；
* “16 observations 意味着每 16 帧一个 ROI”；
* “ToMe/DynamicViT 与 pre-backbone native crop 等价”；
* “FLOP 降低即真实加速”；
* “ROI 优于自由 selection”，除非通过 P1 ROI 主张门；
* 把当前 Continuous-RoI S2 exact-nine training receipt 当作 GeoRoute 实证；
* 在 P3 前使用 official-test 结果或 paper-ready 表述。

---

## 14. 唯一下一步

> **在 `8ebe5f…` 派生的干净 worktree 中，只实现并运行 P0：dynamic-source raw native tubelet gather、Conv3d/linear patch parity、full-K detector parity、single-VideoMAE audit、score-function known-answer test 和端到端 vertical-slice cost KAT；P0 全部通过前，不排完整训练，不实现 A-MoD。**

[1]: https://arxiv.org/abs/2210.09461 "https://arxiv.org/abs/2210.09461"
[2]: https://arxiv.org/html/2412.11228 "https://arxiv.org/html/2412.11228"
[3]: https://arxiv.org/html/2412.20875 "https://arxiv.org/html/2412.20875"
[4]: https://www.collaborative-ai.org/publications/gadhikar24_arxiv/ "https://www.collaborative-ai.org/publications/gadhikar24_arxiv/"
[5]: https://arxiv.org/abs/2404.02258 "https://arxiv.org/abs/2404.02258"
[6]: https://arxiv.org/abs/2106.02034 "https://arxiv.org/abs/2106.02034"
[7]: https://arxiv.org/abs/2411.02886 "https://arxiv.org/abs/2411.02886"
