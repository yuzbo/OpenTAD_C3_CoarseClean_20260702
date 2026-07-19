# 仓库可见性证书

**证书状态：`PASS_EXACT_SNAPSHOT_WITH_BRANCH_REF_LIMITATION`**

| 项目         | 认证结果                                                                                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repository | `yuzbo/OpenTAD_C3_CoarseClean_20260702`，GitHub 连接器返回 `visibility=public`、`archived=false`，仓库内容可读。                                                                         |
| 文档快照       | 精确提交 `21c264b85690c05ee7fe27e054d3b84eda1fc02c` 成功解析，提交信息为 `docs: absorb PhysTime full60 Pro review`。                                                                     |
| 可执行代码快照    | 精确提交 `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132` 成功解析；冻结实验记录绑定 tree `bddc9b9386604d00d213275a47ce7997b35d3f4c`。                                                            |
| 分支指针       | GitHub 连接器能按分支 URL 和精确提交读取文件，但 branch-search 接口没有枚举出该长分支名。因此我认证两个精确提交对象及其文件内容，**不额外声称当前移动分支仍指向 21c264b**。                                                                 |
| 读取范围       | 已读取六份指定文档、两份 matched config、`ActionFormer`、projection、head、`SingleStageDetector`、test engine、raw/native geometry、相关测试、real CUDA gate、full60 Slurm 部署和 artifact validator。 |
| 执行纪律       | 全程只读；没有修改仓库、提交代码、启动训练、CUDA/Slurm 作业或生成实验 artifact。                                                                                                                        |

---

# 1. RESEARCH_VERDICT

## **RESEARCH_VERDICT：STOP-Q-LIFT**

准确含义是：

> **立即停止把“增加 Q”作为下一项训练型主方法。继续保留 physical-metric 路线，但下一阶段必须先做 Q192 下的物理机制分解、评估修复和无训练的 Q-density 反事实。**

这不是终止 PhysTime。它是否定当前证据链中的这一跳：

```text
physical Q192 很好
→ 剩余差距一定来自 Q 不够
→ 因而需要 cross-attention / Q384
```

代码和结果都没有证明中间两步。

更直接地说：

* 当前最强事实是 **物理时间度量有效**；
* 当前没有证据表明 **检测位置密度不足**；
* 更没有证据表明 **学习式 support-to-query bridge** 是解决方案；
* 此时实现 B/C/D/E 中任何训练型结构，都会把额外容量、上下文、候选密度、优化难度和时间机制重新混在一起。

唯一获批路线是：

> **A 作为生产结构保留；F 采用“assignment metric × regression/decode metric”的无参数因子化，先查清 57.57 到底来自哪里。**

---

# 2. 六项布尔裁决

| 裁决项                    |        结论 | 精确范围                                                                                    |
| ---------------------- | --------: | --------------------------------------------------------------------------------------- |
| `CODE_CORRECT`         | **FALSE** | 核心 physical-grid 数学与执行链正确，但发布级实现仍有跨窗口 NMS 提前舍入和尾窗 mask 污染风险。                            |
| `COMPARISON_FAIR`      |  **TRUE** | 仅指当前 `uniform-rank-seconds Q192` 与 `physical-seconds Q192` 两臂；不包括旧 63.61 或 dense 68.29。 |
| `Q_LIFT_NEEDED`        | **FALSE** | 尚未证明需要 Q-lift；当前只能说“未知”，在布尔裁决中必须 fail-closed 为 false。                                   |
| `IMPLEMENTATION_READY` | **FALSE** | 新实验所需的全精度 NMS、严格正时长过滤、尾窗隔离测试、机制 artifact 尚未闭合。                                          |
| `PILOT_READY`          | **FALSE** | 当前 gate 能证明可运行和身份一致，不能证明 Q bottleneck、每 GT 可分配性或真实成本。                                   |
| `PAPER_READY`          | **FALSE** | 仍缺机制隔离、多 seed、成本、合法时间反事实和第二数据集。                                                         |

**重要限定：**`CODE_CORRECT=false` 不表示 57.57 作废。它表示当前快照不能直接作为下一阶段论文级实现基线。

---

# 3. P0 / P1 / P2 问题表

我没有发现会直接推翻 57.57 的历史 P0。下表中的 P0 是**进入下一研究阶段前必须修复的 prospective blocker**。

| 级别     | 文件、类、函数、行号                                                                                                                                                                             | 代码证据                                                                                                                                       | 影响                                                                                                                   | 必须修复                                                                                                |
| ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **P0** | `opentad/models/detectors/single_stage.py`，`SingleStageDetector.post_processing`，约 155–232，特别是 218–221；`opentad/cores/test_engine.py`，`apply_sliding_window_nms`，约 112–139             | 每个窗口先把 segment 舍入到 0.01 秒、score 舍入到 1e-4，再由 test engine 重建 tensor 做跨窗口 NMS；NMS 后又舍入一次。                                                     | 能改变 Soft-NMS IoU、投票边界和分数排序；高 IoU、Q-density 和 boundary-error 结论都可能被污染。它是两臂共享的，所以不自动抹掉 +16.29，但必须用冻结 checkpoint 全精度重放。 | scientific artifact 全程保留 float32/float64；禁止模型输出层和跨窗口 NMS 前舍入。格式化只能发生在不参与 evaluator 的展示副本。           |
| **P1** | `opentad/models/dense_heads/anchor_free_head.py`，`_clamp_physical_proposals_to_domain`，430–443；`forward_test`，536–560                                                                  | clamp 独立限制左右端点，没有过滤 `end <= start + eps`、非有限 proposal。                                                                                     | 边界处 ReLU 零回归可能产生零时长 proposal，并进入 score 展开和 NMS。                                                                      | clamp 后、NMS 前统一执行 finite 与严格正时长过滤；artifact 记录过滤数量和样本。                                               |
| **P1** | `opentad/models/necks/fpn.py`，`FPNIdentity.forward`，约 110–127；`opentad/models/bricks/conv.py`，`ConvModule.forward`，约 87–109；`AnchorFreeHead._init_cls_convs/_init_reg_convs`，约 456–487 | FPN LayerNorm 后没有重新乘 mask；LayerNorm bias 训练后可使无效 tail 位置非零。head 的首个 kernel-3 卷积先读邻域、后乘 mask，因此最后一个有效位置可能读到无效位置的 LN bias。                   | 不等同于重复 RGB 泄漏，也不破坏两臂相对 matched；但会削弱“整个 detector 严格 padding-isolated”的主张，并可能影响短尾窗。                                    | `FPNIdentity.forward` 在 norm 后再次乘对应 mask；添加 tail padding 内容替换不变性和 invalid-gradient-zero 测试。         |
| **P1** | `AnchorFreeHead._build_physical_points_and_masks`，约 250–397；`prepare_targets`，690–797；`get_refined_proposals`，562–575                                                                  | 当前 physical intervention 同时改变 point center、局部 stride、regression range、center sampling、target normalization 和 decode。                       | 57.57 的收益来源不可识别。此时把剩余误差归于 Q 是无依据的。                                                                                   | 先运行 Q192 下 `decode metric × assignment metric` 2×2；必要时再分 center 与 scale，不得先加 bridge。                |
| **P1** | `AnchorFreeHead.losses`，610–649；`tools/bata/run_phystime_g1a_real_gate.py`，assignment validator                                                                                        | 只记录 batch/sample aggregate positive 数、valid point 数和 GT 总数；没有每 GT eligible、positive、zero-positive。                                         | 无法判断性能差距是无可分配 query、分类排序、边界回归还是 NMS。                                                                                 | 输出每 GT JSONL/Parquet：eligible、positive、最佳 IoU、最近 query/observation 距离、duration/gap strata。          |
| **P1** | `ActionFormer._update_native_temporal_query_audit`，187–204                                                                                                                             | 同一个 `sum(level_lengths)` 同时写入 `query_tensor_count` 和 `query_count`，只有另一个字段记录每样本有效数。                                                        | 容易把容量 378、tail 有效位置、以及 `378×20` class scores 混称为 query 数。                                                            | 改名为 `Q_location_capacity`、`Q_location_valid_per_sample`、`Q_class_score_count`；禁止只报一个 `query_count`。 |
| **P2** | `opentad/datasets/transforms/end_to_end.py`，`LoadFrames.random_trunc`，约 417–460                                                                                                        | 最多尝试 200 次寻找有动作 crop；若均失败，函数会使用最后一次 crop，而不是显式失败或记录 fallback。                                                                              | 极端数据下训练 crop 合同不够 fail-closed。                                                                                       | 记录 trial count；耗尽时明确 fallback policy 或抛错。                                                           |
| **P2** | `opentad/cores/test_engine.py`，`apply_sliding_window_nms`，约 129                                                                                                                        | class ID tensor 被构造成 `float32`。                                                                                                            | 当前小类别数通常不改变结果，但类型合同错误。                                                                                               | 使用 `torch.long`。                                                                                    |
| **P2** | full60 manifest/validator                                                                                                                                                              | checkpoint 报告 499 个 online/EMA state entries，但没有独立登记 `total_numel`、`trainable_numel`、`optimizer_covered_numel`。现有验证侧重 schema/hash，而非标量参数量。 | 后续 B/D/E 的参数量或优化器覆盖可能被误报。                                                                                            | artifact 中同时登记参数名集合 hash、tensor count、总 numel、trainable numel、optimizer-covered numel 和每模块 numel。   |
| **P2** | `phystime_g1a_selected_axis_native_j192.py`                                                                                                                                            | 实际模式是 `uniform_rank_seconds`，不是旧式 selected-rank GT remap。                                                                                  | “selected-axis”名称容易让读者误以为它复用了旧 selected-rank 监督。                                                                     | 论文、artifact 和图表统一称 `uniform-rank-seconds`。                                                          |

---

# 4. 当前 57.57 能够证明什么、不能证明什么

## 4.1 能够证明

### 4.1.1 当前两臂是严格有意义的 matched 因果比较

两臂共同使用：

* K=384 原始 RGB 观测；
* J=192 native tubelet token；
* projection 起始长度 192；
* 六层长度 `[192,96,48,24,12,6]`；
* 位置候选容量总数 (Q_\Sigma=378)；
* 同一 VideoMAE-S、projection、neck、head、optimizer、seed 42、训练长度和 evaluator；
* 均无 backbone 后 feature interpolation。

physical config 明确设置 K384/J192、禁用 post-backbone `Interpolate`，并把 head 位置轴设为秒域；uniform arm 继承同一模型，只把三套 pipeline 的 `coordinate_mode` 改为 `uniform_rank_seconds`。

最终十次 matched validation 都维持相同排序；epoch 59 为：

* uniform-rank-seconds：41.28 Avg-mAP，14.86 @0.7；
* physical-seconds：57.57 Avg-mAP，28.64 @0.7；
* Avg-mAP 差值 +16.29。

两个 60 轮 checkpoint 的 online/EMA 状态可重放、有限，artifact validator 通过。

因此可以写：

> 在固定 K384/J192/QΣ378、无 feature interpolation 的同构稀疏 ActionFormer 中，把规则 rank 秒轴替换为真实物理秒轴，在 THUMOS14、seed 42、完整 60 轮训练下带来稳定且显著的性能提升。

### 4.1.2 它证明 physical metric，而不是 physical representation

真实调用链是：

```text
RGB/mask
→ BackboneWrapper / VideoMAE
→ native J192 对齐
→ Conv1DTransformerProj
→ FPNIdentity
→ AnchorFreeHead 先产生 cls/reg tensor
→ 再从 metas 构造 physical points
→ assignment / normalization / decode / clamp
```

`ActionFormer.forward_train/forward_test` 调 backbone 时只传 `inputs,masks`；projection 也只接收 `(x, masks)`。metas 在 head 调用时才被传入。

`Conv1DTransformerProj.forward(self, x, mask)` 没有 timestamp/metas 参数。

而 `AnchorFreeHead.forward_train/forward_test` 是先计算分类和回归张量，随后才调用 `_build_physical_points_and_masks`。

所以 57.57 证明的是：

> **共享表征之上的秒域 point/assignment/regression/decode 修正有效。**

它不证明 VideoMAE、TIA/adapter 或 ActionFormer projection 已经理解真实时间 gap。

### 4.1.3 当前 point 几何没有半格 offset 错误

`PointGenerator` 默认 `use_offset=False`，基础 point 为：

[
0,s_\ell,2s_\ell,\ldots
]

而不是 ((i+0.5)s_\ell)。因此 `_selected_axis_to_physical_axis` 把整数 rank anchor 映射到 token 时间戳是自洽的，不存在我最初优先排查的半格偏移 bug。

### 4.1.4 采样没有推理 GT 泄漏，但训练 crop 使用 GT

必须精确表述：

* `random_trunc` 使用 GT intersection ratio，并尝试找到至少包含一个动作的 crop；
* crop 接受以后，`_select_random_fixed_positions` 只由 window/sample key 初始化固定随机数，不读取 GT；
* val/test 使用 sliding window，再执行同样的固定子采样，不读取 GT。

因此正确主张是：

> **训练窗口选择 GT-aware；已接受窗口内的 K384 固定不规则子采样 GT-free；推理无 GT。**

不能简称“整个采样过程无 GT”。

## 4.2 不能证明

57.57 不能证明：

1. Q192 是剩余性能差距的瓶颈；
2. Q384 会提高高 IoU；
3. cross-attention 优于非学习 copy 或无 bridge；
4. classification representation 已利用物理 gap；
5. physical stride 等于 final VideoMAE feature 的真实支持域；
6. 相对旧 63.61 或 dense 68.29 更优或更差；
7. 结果可跨 seed、跨 sampler、跨数据集复现；
8. 方法具有足够的顶会新颖性。

尤其第 5 点，代码和测试主动承认：

```text
phystime_native_final_feature_support_is_exact = False
```

它只保证 patch-embed 输入 atom 的 provenance，final feature support 是经过 chunk attention 和 adapter mixing 后的结构上界。

旧 63.61 分支同时采用 GT remap、J192→Q384 feature interpolation 和 Q0=384；dense 路线又使用 K/J/Q=768 级别的观测和表示，因此都不是当前物理时间干预的公平对照。

---

# 5. 候选结构逐项淘汰与唯一推荐

| 候选                                                 | 裁决                                      | 原因                                                                                                              |
| -------------------------------------------------- | --------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| **A. 保持 physical-metric，不增加 Q**                    | **保留**                                  | 当前唯一被完整 60 轮支持的结构；零新增参数、零新增 observation、保持 ActionFormer 上下文。                                                    |
| **B. J→Q feature interpolation**                   | **只能作辅助负/工程基线**                         | 无论放在 projection 前还是后，都会改变 latent states、局部感受野、head 输入和候选密度。它不是“只改 Q”。旧分支已经证明这种比较不可归因。                           |
| **C. nearest / barycentric copy**                  | **nearest 只准作无训练诊断；barycentric 不准作主方法** | nearest 会复制同一证据并改变 proposal multiplicity/训练梯度权重；barycentric 会在 gap 内制造插值 latent state。两者都不能自动说明新的 query 获得了新证据。 |
| **D. masked sparse-support→query cross-attention** | **当前拒绝**                                | 增加参数、上下文汇聚能力、优化路径和可能的 learned query prior；即使成功也难区分“Q 数量”“额外 attention 容量”和“物理时间”。还必须处理 all-masked softmax。      |
| **E. 轻量 Δt/gap-conditioned projection**            | **未来条件候选，不是下一步**                        | 它比 Q-lift 更贴合“时间戳当前对 projection 不可见”的代码事实，但只有在残差错误确实随 gap、短动作或 classification ranking 系统变化时才有依据。                |
| **F. 轴因子化 physical metric，保持 Q192**                | **唯一推荐**                                | 直接把当前捆绑干预拆成 assignment metric 与 regression/decode metric；无新参数、无新 observation、几乎无成本，且可直接回答 57.57 来自何处。           |

## 唯一推荐：`Q192 Axis-Factorized Physical Metric`

不是新 cross-attention 类，而是利用当前 `AnchorFreeHead` 已存在的独立 assignment-axis 支持：

* decode/regression 轴 (z^D\in{u,p})；
* assignment 轴 (z^A\in{u,p})；
* 固定 K384、J192、QΣ378；
* 形成四臂：

| Arm | regression/decode 轴 | assignment 轴 |
| --- | ------------------- | ------------ |
| UU  | uniform             | uniform      |
| UP  | uniform             | physical     |
| PU  | physical            | uniform      |
| PP  | physical            | physical     |

`AnchorFreeHead.__init__` 已支持 `assignment_positions_key`，并据此选择 `physical_axis` 或 `separate_axis`；`_build_physical_points_and_masks` 已经分别生成 regression points 与 assignment points。

这比用户给出的 axis×Q 2×2 更早、更必要。因为当前未回答的是：

> 57.57 来自训练时 positive/scale eligibility 改变，还是来自 target normalization/decode 进入真实秒域？

在这个问题未回答前，Q-lift 是不可识别的。

从新颖性看，也必须降温：TE-TAD 已明确使用 actual timeline coordinate 并讨论自适应 query 数；RCL 已研究连续时间坐标条件下的 continuous anchoring；DualDETR 和 DiGIT 已分别使用边界级 query 和更强 cross-attention decoder。因此“真实时间”“更多 query”或“cross-attention”本身均不是可防守的新颖性。([CVF Open Access][1]) 2026 年的 LiquidTAD 还已把 continuous-time dynamics 引入 TAD，因此也不能声称“首个 continuous-time TAD”。([arXiv][2])

可争取的窄主张只能是：

> 固定稀疏原始观测下，显式区分 observation、native support token 和 detection location；不把 query 计作观测，不在 gap 中补 RGB/latent evidence，并在真实物理时间上因子化 assignment 与 regression/decode metric。

---

# 6. 数学定义、tensor shape 与最小代码方案

## 6.1 当前物理几何

设接受窗口的原始帧域为 ([a,b]) 秒，原始稀疏观测数为 (K_v\le384)。

第 (k) 个选中原始帧索引为 (r_k)，FPS 为 (f)：

[
\tau_k = \frac{r_k}{f}.
]

对应原始 ownership atom 为：

[
I_k=
\left[
\max\left(\tau_k-\frac{h}{2f},a\right),
\min\left(\tau_k+\frac{h}{2f},b\right)
\right],
]

其中 (h) 是 dense sampling 的原始帧步长。该实现没有把相邻稀疏帧之间的 gap 填成连续证据。

tubelet size 为 2。第 (j) 个 native token 的位置：

[
p_j=
\frac{1}{|A_j|}
\sum_{k\in A_j}\tau_k ,
]

其中 (A_j) 仅包含该 tubelet 中真实有效的 atom；padding repeat 不参与语义均值。

规则轴位置为：

[
u_j=a+\left(j+\frac12\right)\frac{b-a}{J_v},
\qquad j=0,\ldots,J_v-1.
]

代码同时保存 (p_j) 和 (u_j)。

对任一轴 (z=(z_0,\ldots,z_{J_v-1}))，定义分段线性映射 (F_z)，经过：

[
(-1/2,a),\quad (j,z_j),\quad (J_v-1/2,b).
]

FPN 层 (\ell) 的 nominal stride 为 (s_\ell\in{1,2,4,8,16,32})，第 (i) 个基础位置：

[
q_{\ell i}=i s_\ell.
]

其秒域中心和局部宽度为：

[
c_{\ell i}^{z}=F_z(q_{\ell i}),
]

[
w_{\ell i}^{z}
==============

## F_z\left(q_{\ell i}+\frac{s_\ell}{2}\right)

F_z\left(q_{\ell i}-\frac{s_\ell}{2}\right).
]

regression range 由 nominal range ([R_\ell^-,R_\ell^+]) 缩放为：

[
\left[
R_\ell^-\frac{w_{\ell i}^z}{s_\ell},
R_\ell^+\frac{w_{\ell i}^z}{s_\ell}
\right].
]

这正是 `_build_physical_points_and_masks` 当前执行的 center、left/right、physical stride 与 range scaling。

对 GT (g=[g_s,g_e])，当前回归 target 为：

[
d_L=\frac{c^D-g_s}{w^D},\qquad
d_R=\frac{g_e-c^D}{w^D},
]

预测解码为：

[
\hat g_s=c^D-\hat d_Lw^D,\qquad
\hat g_e=c^D+\hat d_Rw^D.
]

assignment eligibility 使用 assignment point 的 center、stride 和 regression range，而严格 inside-GT 条件仍使用 decode point center。

因此推荐四臂的精确定义是：

[
z^D,z^A\in{u,p},
]

* (z^D)：回归 target、normalization、proposal decode；
* (z^A)：center sampling radius 和 regression-range eligibility；
* classification features/logits始终相同结构；
* Q 始终不变。

这能分离 assignment metric 与 regression/decode metric，但仍不能完全分离 decode center 与 local scale。只有四臂结果显示 scale 仍是关键歧义时，才批准第二阶段进一步拆 center/scale。

## 6.2 当前 tensor shape

| 阶段                          | Tensor shape            | 语义                                          |
| --------------------------- | ----------------------- | ------------------------------------------- |
| 输入 RGB                      | `[B,1,3,384,H,W]`       | 384 个原始 RGB observation slots               |
| raw mask                    | `[B,384]` bool          | 严格有效前缀                                      |
| chunk rearrange             | `[24B,1,3,16,H,W]`      | 24 个 VideoMAE chunk                         |
| backbone 实际输入               | `[24B,3,16,H,W]`        | flatten `num_segs` 后                        |
| VideoMAE/native 输出          | `[B,384,192]`           | J=192，384 是 VideoMAE-S embed dim            |
| native mask                 | `[B,192]`               | 每两个 raw slots 做 any reduction，奇数尾 atom 仍可有效 |
| projection/FPN level (\ell) | `[B,512,T_\ell]`        | `T=[192,96,48,24,12,6]`                     |
| level mask                  | `[B,T_\ell]`            | prefix/downsample mask                      |
| cls logits                  | `[B,20,T_\ell]`         | 20 THUMOS 类                                 |
| reg logits                  | `[B,2,T_\ell]`          | 左右归一化距离                                     |
| physical/assignment points  | `[B,T_\ell,4]`          | `[center, min_range, max_range, stride]`    |
| location capacity           | (192+96+48+24+12+6=378) | `Q_location_capacity`                       |
| class-location scores       | (378\times20=7560)      | 不能称 7560 个 query                            |

输入和 chunk 变换来自 config 与 wrapper；projection 输出通道、层数、stride 和类别数来自 ActionFormer base config。

## 6.3 推荐 F 的精确实现规格

### 不新增 production 类

继续使用：

```text
opentad/models/dense_heads/anchor_free_head.py
class AnchorFreeHead
```

利用现有配置字段：

```python
physical_grid_actionformer=dict(
    enabled=True,
    required=True,
    positions_key=DECODE_AXIS_KEY,
    assignment_positions_key=ASSIGNMENT_AXIS_KEY,
    selected_count_keys=["phystime_native_valid_count"],
    assignment_count_keys=["phystime_native_valid_count"],
    axis_start_key="phystime_g1a_axis_start_sec",
    axis_end_key="phystime_g1a_axis_end_sec",
)
```

四臂 key：

| Arm | `positions_key`                        | `assignment_positions_key`             |
| --- | -------------------------------------- | -------------------------------------- |
| UU  | `phystime_uniform_rank_timestamps_sec` | `phystime_uniform_rank_timestamps_sec` |
| UP  | `phystime_uniform_rank_timestamps_sec` | `phystime_native_token_timestamps_sec` |
| PU  | `phystime_native_token_timestamps_sec` | `phystime_uniform_rank_timestamps_sec` |
| PP  | `phystime_native_token_timestamps_sec` | `phystime_native_token_timestamps_sec` |

所有 config 应继承一个共同 base；dataset pipeline 固定为同一份，因为 `BuildPhysTimeNativeTubeletGeometry` 已同时写出两套轴。只有上述两个字符串和 `work_dir` 可以变化。

### 初始化、参数与梯度

* 新参数：**0**
* 新 buffer：**0**
* 参数初始化：不适用
* 四臂 state-dict schema：必须逐名字、shape、dtype 完全一致
* optimizer parameter set：必须完全一致
* backbone/projection/head 梯度：保持当前路径
* 时间轴是 metadata，不可学习
* assignment 离散选择在 `@torch.no_grad()` 的 `prepare_targets` 内，不接受梯度
* regression/decode 通过相同 cls/reg 网络训练
* query provenance：

```text
state_type = detection_location
source_level = FPN level
source_index = level index
is_frame = false
is_tubelet = false
is_observation = false
observation_count_increment = 0
```

## 6.4 Q192 / Q384 的裁决性定义

### Q192

当前结构：

```text
Q0 = 192
QΣ = 378
```

每个 detection location 与一个 projection/FPN state 对应，但它不等于 observation。K384、J192、QΣ378 必须分别登记。

### Q384

**不批准训练型 Q384。**

只允许先做一个 evaluation-only 的 `SubcellQueryReplay`：

```text
opentad/models/dense_heads/physical_query_replay.py
def replay_subcell_queries(
    cls_logits: Sequence[Tensor],
    reg_offsets: Sequence[Tensor],
    points: Sequence[Tensor],
    masks: Sequence[Tensor],
    phases: tuple[float, ...] = (-0.25, +0.25),
) -> ReplayOutput
```

规则：

* 每个有效 parent location 生成两个 detection child；
* child 复用 parent cls/reg 输出；
* center rank 分别平移 (\pm 0.25s_\ell)；
* 不改变 parent stride、range 或 evidence；
* 不训练、不回传梯度、不增加参数；
* level capacity 变为 `[384,192,96,48,24,12]`；
* (Q_\Sigma=756)；
* `observation_count_increment=0`；
* artifact 记录 `(level,parent_index,phase)`。

它只回答：

> 在完全不增加表示能力的条件下，增加几何 readout 密度是否能提高 pre-NMS/oracle 高 IoU recall？

它不是主方法。

### Padding 和 all-masked

当前 raw/native geometry 明确要求非空严格前缀；生产数据中的全 mask sample 应 fail closed，而不是产生输出。

未来任何 cross-attention bridge 若获批，必须定义：

```text
support_valid_count == 0
→ query_mask 全 false
→ query_state 全零
→ logits/reg 不进入 loss/NMS
→ 禁止对全 -inf attention 做 softmax
```

当前 `MaskedMHCA` 对 key mask 后直接 softmax；如果未来任意 query 面对全 masked support，必须在 softmax 前显式分支，否则存在 NaN 风险。

## 6.5 最小 patch map

1. `single_stage.py::post_processing`

   * 删除 segment/score 的 `round`；
   * 返回 full-precision Python float 或 tensor-backed record。

2. `test_engine.py::apply_sliding_window_nms`

   * labels 改 `torch.long`；
   * NMS 前 finite/positive-duration filter；
   * NMS 后 scientific result 保持全精度；
   * 展示舍入另存 human-readable artifact。

3. `fpn.py::FPNIdentity.forward`

   * norm 后重新乘 mask。

4. `anchor_free_head.py`

   * artifact 中显式输出 decode/assignment axis key；
   * 每 GT eligible/positive 诊断；
   * clamp 后记录并过滤零时长 proposal。

5. 新增四个 config

   * `phystime_g1a_uu_native_j192.py`
   * `phystime_g1a_up_native_j192.py`
   * `phystime_g1a_pu_native_j192.py`
   * `phystime_g1a_pp_native_j192.py`

6. 可选诊断文件

   * `opentad/models/dense_heads/physical_query_replay.py`
   * 仅工具调用，不注册为训练 head。

---

# 7. 测试、真实 CUDA gate 与 artifact schema

## 7.1 必须新增的单元测试

### NMS 与 proposal 完整性

1. 两个 segment 在 0.01 秒舍入后 IoU 跨越 Soft-NMS 阈值；修复后结果必须等于直接 tensor NMS。
2. 两个 score 在四位舍入后改变排序；修复后排序保持原始 float。
3. `end==start`、`end<start`、NaN、Inf 全部在 NMS 前被拒绝并计数。
4. DDP 单 rank 与多 rank gather 后结果 bitwise/容差一致。

### Padding

5. 修改无效 tail RGB 内容，在 strict mask 下所有有效 backbone token、projection states、head logits 必须不变。
6. 把 FPN invalid positions 替换为随机极大值，修复后有效 head 输出必须不变。
7. invalid positions 对 loss 的梯度必须为零。

### 物理几何和因子化

8. UU/UP/PU/PP 对人工时间轴产生解析可验证的 point center/stride/range。
9. 当 physical axis 等于 uniform axis 时，四臂输出必须完全相同。
10. assignment key 只能改变 positive eligibility，不能改变 cls feature tensor。
11. decode key 改变 proposal 秒边界，但不能改变 cls logits。
12. domain edge、单 token、奇数 valid atom、极端 gap、短尾窗全部覆盖。

### 采样与反事实

13. 同一个接受窗口，不同 GT 内容不得改变 `keep_positions`。
14. `random_trunc` 的 GT-aware provenance 必须为 true；val/test crop 与 subsample uses-GT 必须为 false。
15. uniformization 和 gap permutation 后时间轴严格递增、端点不变、gap multiset hash 正确。
16. Q replay 的 child provenance、QΣ756、参数增量 0、梯度禁用必须验证。

已有测试已经覆盖无 feature interpolation、K/J mismatch、严格 prefix、padding atom provenance 和 explicit domain edge；新测试必须补的是 head 后段、跨窗口 NMS 和机制分解。

## 7.2 真实 CUDA gate

每一臂必须在同一 clean SHA 下执行：

### 身份

```text
git_commit
git_tree
canonical_config_sha256
effective_config_sha256
dataset_manifest_sha256
pretrained_checkpoint_sha256
initial_state_dict_sha256
selected_indices_manifest_sha256
```

### 真实执行

* 真实 CUDA，单 visible GPU，`cuda:0`；
* full-valid batch 和 partial-tail batch；
* AMP 与 EMA 打开；
* 至少 3 次 successful optimizer updates；
* 检查 backbone adapter、projection、neck、cls、reg 各参数族：

  * parameter numel；
  * gradient finite；
  * gradient nonzero；
  * update nonzero；
* 四臂初始 state、参数名、shape、optimizer coverage 完全一致；
* UU/UP/PU/PP 的 RGB tensor、raw mask、K/J、selected-index checksum 完全一致；
* 单独测量 backbone、projection、neck、head、decode、跨窗口 NMS、总时延及峰值显存；
* fail on：

  * early rounding；
  * NaN/Inf；
  * zero-duration proposal 未过滤；
  * tail invariance 失败；
  * 参数集合漂移；
  * GT-dependent within-window subsampling；
  * Q 或 observation 计数错误。

现有 full60 脚本已经绑定 Slurm、commit/tree、config、dataset、checkpoint、60 epochs、seed42 并禁止 feature interpolation；这些身份合同应保留。

## 7.3 Artifact schema

建议 schema：

```json
{
  "schema_version": "phystime_metric_factorization_v1",
  "identity": {
    "commit": "...",
    "tree": "...",
    "config_sha256": "...",
    "checkpoint_sha256": "...",
    "dataset_manifest_sha256": "...",
    "environment_sha256": "..."
  },
  "sampling": {
    "K_capacity": 384,
    "K_valid_per_sample": [],
    "window_crop_uses_gt": true,
    "within_window_subsample_uses_gt": false,
    "selected_indices_sha256": "..."
  },
  "representation": {
    "J_capacity": 192,
    "J_valid_per_sample": [],
    "feature_interpolation": false
  },
  "queries": {
    "Q0_location_capacity": 192,
    "Qsum_location_capacity": 378,
    "Qsum_valid_per_sample": [],
    "class_score_count_per_sample": [],
    "is_observation": false,
    "provenance_sha256": "..."
  },
  "metric": {
    "decode_axis": "uniform|physical",
    "assignment_axis": "uniform|physical",
    "domain_unit": "seconds"
  },
  "parameters": {
    "state_tensor_count": 0,
    "total_numel": 0,
    "trainable_numel": 0,
    "optimizer_covered_numel": 0,
    "parameter_name_shape_sha256": "..."
  },
  "assignment": {
    "per_gt_artifact": "...",
    "eligible_query_count": [],
    "positive_query_count": [],
    "zero_positive_gt_count": 0
  },
  "predictions": {
    "pre_nms_full_precision_sha256": "...",
    "post_nms_full_precision_sha256": "...",
    "rounding_stage": "display_only",
    "nonfinite_filtered": 0,
    "nonpositive_duration_filtered": 0
  },
  "diagnostics": {
    "pre_nms_recall": {},
    "post_nms_recall": {},
    "oracle_score": {},
    "oracle_boundary": {},
    "fixed_top_n": {},
    "short_action": {},
    "gap_strata": {},
    "boundary_error": {}
  },
  "counterfactual": {
    "uniformized_axis_sha256": "...",
    "gap_permuted_axis_sha256": "...",
    "strictly_increasing": true,
    "endpoint_preserved": true
  },
  "cost": {
    "latency_p50_ms": {},
    "latency_p95_ms": {},
    "peak_memory_bytes": 0,
    "candidate_count": 0
  },
  "validation": {
    "pass": true,
    "violations": []
  }
}
```

每 GT artifact 至少包含：

```text
video_id, window_id, gt_id, class_id,
duration_sec, start_sec, end_sec,
nearest_observation_start/end_distance,
local_gap_quantile,
eligible_Q, positive_Q, zero_positive,
best_pre_nms_iou, best_post_nms_iou,
best_score_rank, boundary_start_error, boundary_end_error
```

---

# 8. 最小实验 DAG、统计与停止条件

## 8.1 用户提出的 axis×Q 四臂目前不足

原计划：

```text
regular/physical × Q192/Q384
```

只有在以下前提成立时才足够：

1. 已经选定唯一 bridge；
2. bridge 在 Q192/Q384 中参数形状相同；
3. Q192 也经过同一 bridge；
4. 已证明当前主要残差与 Q coverage 相关。

这些前提目前全部未满足。

## 8.2 更好的最小 DAG

### Stage 0：评估正确性修复

对冻结 epoch-59 online/EMA checkpoint：

```text
旧 rounded NMS
vs
新 full-precision NMS
```

两臂全部重放。

必须输出：

* mAP@0.3:0.7；
* proposal 数；
* suppression decisions 差异；
* boundary change 分布；
* 短动作和高 IoU strata；
* online/EMA 一致性。

若 physical-minus-uniform 的方向反转，暂停所有结构研究并审计 evaluator。若方向保持，57.57 被升级为 full-precision-replayed anchor。

### Stage 1：现有 checkpoint 的 decode cross-replay

由于 timestamps 不进入 backbone/projection，可以对两个冻结 checkpoint 分别用 uniform/physical decode 轴重放：

```text
train-U / decode-U
train-U / decode-P
train-P / decode-U
train-P / decode-P
```

这不会完全隔离训练 assignment，但能廉价判断：

* 增益主要来自 inference geometry；
* 还是必须依赖 physical-axis 训练。

### Stage 2：Q192 assignment×decode 2×2

在同一新提交下训练 UU/UP/PU/PP，固定所有其他变量。

主因子：

[
\Delta_D =
\frac12[(UP-UU)+(PP-PU)]
]

表示 decode metric 主效应；

[
\Delta_A =
\frac12[(PU-UU)+(PP-UP)]
]

表示 assignment metric 主效应；

[
\Delta_{DA}=PP-PU-UP+UU
]

表示交互。

只有这个阶段能决定 physical-metric 机制是否值得作为论文主线。

### Stage 3：无训练 Q-density replay

对最佳 Q192 checkpoint 运行 QΣ378→756 的 `SubcellQueryReplay`。

检查：

* pre-NMS class-aware recall@0.7；
* oracle-score recall；
* fixed-top-N recall；
* boundary coverage；
* NMS 后 recall；
* 实际 decode/NMS 成本。

若连 oracle/pre-NMS 指标都没有实质改善，则 **永久停止 Q-lift**。

### Stage 4：条件式 Q-lift

只有 Stage 3 显示 Q-density 是明确瓶颈，才选择一个 bridge。此时才运行：

```text
uniform/physical × Q192/Q384
```

而且四臂必须在同一新 bridge 和同一提交下重训。旧 41.28/57.57 只作外部 anchor。

## 8.3 主指标与机制指标

### 唯一主性能指标

[
\text{mAP@0.7}
]

### 必须通过的 guardrail

[
\text{Avg-mAP}_{0.3:0.7}
]

不得以提升 @0.7 但大幅破坏 Avg-mAP 的方式通过。

### 机制指标

* 每 GT eligible query；
* 每 GT positive query；
* zero-positive GT rate；
* pre-NMS class-agnostic recall@0.7；
* pre-NMS class-aware recall@0.7；
* post-NMS recall@0.7；
* oracle score；
* oracle boundary；
* fixed top-N recall，至少报告 (N={100,500,2000})，生产主值为配置中的 2000；
* start/end absolute error；
* short-action quantiles；
* local support-gap quantiles；
* distance-to-nearest-observation strata。

### 成本指标

* end-to-end p50/p95；
* backbone、projection、head、decode、NMS 分项；
* peak allocated/reserved memory；
* train step p50/p95；
* candidate count；
* total/trainable/optimizer-covered parameters。

计时必须：

* CUDA synchronize；
* 随机交错执行不同 arms；
* warm-up 直到 rolling median 稳定；
* 继续测量至 latency bootstrap 的相对半宽低于预注册误差，或达到硬样本上限并披露未收敛。

## 8.4 合法反事实

### Uniformization

[
p_j \longrightarrow
a+\left(j+\frac12\right)\frac{b-a}{J_v}
]

保持 observation、feature、顺序、端点和 J 不变，只消除不规则 gap。

### Gap permutation

定义内部正 gap：

[
\Delta_j=p_{j+1}-p_j,\quad j=0,\ldots,J_v-2.
]

按固定、无 GT 的 window seed 置换 (\Delta_j)，然后：

[
p'*0=p_0,\qquad
p'*j=p_0+\sum*{k<j}\Delta*{\pi(k)}.
]

这样：

* (p'_j) 严格递增；
* 首尾 token 时间不变；
* 内部 gap multiset 不变；
* observation 顺序不变；
* 只破坏“哪个位置对应哪个 gap”。

禁止直接随机打乱 timestamp。

## 8.5 统计方法和非拍脑袋停止条件

### 配对单位

* 同 seed、同 video、同 selected-index checksum；
* 顶层科学单位是 seed；
* video 用于 seed 内 paired bootstrap；
* window 和 proposal 不能作为独立样本。

### 分层估计

多 seed 后采用 hierarchical paired bootstrap：

1. 对 seed 重采样；
2. 在每个 seed 内对 video 重采样；
3. 保持 arms 配对；
4. 报告均值、中位数、95% CI 和 seed-level effect。

单 seed 的 video bootstrap 只能说明数据集视频异质性，不能代替训练随机性。

### 最小有意义效应

设可负担的最大 seed 数为 (n_{\max})，校准阶段得到 paired seed SD (s_d)。由预注册的 (\alpha) 和 power (1-\beta) 计算成本预算下的最小可检测效应：

[
\delta_{\text{MDE}}
\approx
\left(
t_{1-\alpha/2,n_{\max}-1}
+
t_{1-\beta,n_{\max}-1}
\right)
\frac{s_d}{\sqrt{n_{\max}}}.
]

定义最小有意义效应：

[
\delta_{\text{MES}}
===================

\max(
\delta_{\text{MDE}},
\delta_{\text{evaluator-noise}},
\delta_{\text{cost-justified}}
).
]

其中：

* `evaluator-noise` 由重复全精度 replay 和 online/EMA variation 估计；
* `cost-justified` 是在不抹掉 K384 相对 dense K768 实测计算节省时，研究方愿意接受的最小性能收益；
* 不允许预先写死“+1.5pp”或“+4pp”。

### 停止规则

* 若效应 95% CI 上界 `< δMES`：**futility stop**；
* 若 95% CI 下界 `> 0` 且成本 guardrail 通过：可升级；
* 若区间跨 0，只在新增 seed 的 conditional power 足够且预算允许时继续；
* 若 Q replay 的 oracle/pre-NMS 指标无改善：无需训练 Q384；
* 若 Q384 只提升 uniform 和 physical 相同幅度：降级为通用 query-density 工程，不是 PhysTime 主张；
* 若 physical interaction 仅发生在一个 seed：不进 60 轮正式矩阵。

## 8.6 何时进入多 seed、60 轮和第二数据集

### 多 seed

只有在：

* full-precision NMS 修复通过；
* UU/UP/PU/PP real CUDA gate 全通过；
* 至少一个 physical 主效应或交互在校准实验中方向一致；
* MDE 在最大成本预算内小于 MES。

### 60 轮

20 轮只能是 trainability/learning-curve gate。只有：

* 多 seed 短程方向一致；
* 没有明显 schedule-dependent 反转；
* primary/guardrail/cost 同时通过；

才运行完整 60 轮。

### 第二数据集

方法和停止规则必须先在 THUMOS14 冻结，第二数据集不能参与路线选择。

数据集选择需审计：

1. raw RGB 是否完整可获得；
2. frame count、decoder FPS、annotation duration 是否能闭环；
3. 高 IoU 边界质量；
4. 短动作与密集并发程度；
5. 不规则采样后 gap 分布是否补充 THUMOS14；
6. 训练成本与许可。

FineAction 包含大量细粒度、密集和共现实例，适合短动作及边界压力测试；HACS-Segments 更大、更广，但原始视频和总算力成本显著更高。([arXiv][3])

不能未经 raw-video/timebase 审计就默认 ActivityNet 是第二数据集。

---

# 9. 最严厉审稿人攻击与防守状态

| 攻击                                                |             当前能否防守 | 裁决                                                                                         |
| ------------------------------------------------- | -----------------: | ------------------------------------------------------------------------------------------ |
| “这只是把错误坐标换成秒，不是新方法。”                              |             **部分** | 57.57 支持物理 metric 的必要性，但 novelty 需靠固定稀疏 observation、无证据填补、support/token/query 分离及机制反事实来建立。 |
| “TE-TAD/RCL 已经使用真实或连续时间坐标。”                       |         **不能完全防守** | 必须主动引用并把主张收窄；不能宣称首个 physical/continuous-time TAD。                                          |
| “cross-attention 是 DETR/TAD decoder 的常规拼装。”       |           **不能防守** | 这正是拒绝 D 作为下一步主方法的原因。                                                                       |
| “Q384 只是复制同一证据并增加候选。”                             |         **当前不能防守** | 先做无训练 replay；若没有 oracle/pre-NMS 改善，Q 路线直接终止。                                               |
| “你把 query 当成了额外帧或 dense observation。”             | **可防守，需 artifact** | 明确 `is_observation=false`、K/J/Q 分开、provenance 完整、observation increment=0。                  |
| “训练采样用了 GT，你声称 no-GT。”                            |      **可防守，必须改措辞** | 训练 random_trunc crop GT-aware；已接受窗口内固定子采样 GT-free；推理无 GT。                                  |
| “时间戳根本没进入 backbone/projection。”                   |      **完全属实，不能回避** | 论文只能称 physical-metric head，不能称 physical-time encoder。                                      |
| “你把 token support envelope 当成真实 feature support。” |            **可防守** | 代码明确记录 exact patch atoms 和 non-exact final feature support，不应做更强主张。                        |
| “提前舍入改变了高 IoU 和 Soft-NMS。”                        |       **当前不能完全防守** | 两臂共享使历史差值未自动作废，但修复并重放前不能提交论文。                                                              |
| “只有 THUMOS14 单 seed。”                             |           **不能防守** | 当前证据只能称 `full60-single-seed-supported`。                                                    |
| “旧 63.61 已经比你高，方法没有价值。”                           |             **部分** | 旧系统不是公平因果对照，但绝对性能缺口仍是真实审稿压力；不能只用“不公平”回避。                                                   |
| “稀疏节省被更大的 head/NMS 吃掉。”                           |           **不能防守** | 当前没有完整 Pareto profile；任何 Q-lift 必须先证明没有抹掉 K384 的实测节省。                                      |
| “physical 的收益其实只是 assignment 数量变多。”               |         **当前不能防守** | 这正是 UU/UP/PU/PP 因子化的首要问题。                                                                  |
| “短动作提升来自跨窗口舍入偶然效应。”                               |         **当前不能防守** | 需要全精度 replay、短动作 strata 和 pre/post-NMS recall。                                             |

总体论文防守状态：

> **可以防守“当前结果是真实 matched 证据”；不能防守“Q-lift 必要”“cross-attention 新颖”“方法已达到顶会完整性”。**

---

# 10. 下一步只能执行的一项具体任务

## **执行 `P0-FULLPRECISION-NMS-REPLAY`，禁止同时实现任何 Q-lift**

该任务必须在一个新的 clean commit 中只完成：

1. 删除 `SingleStageDetector.post_processing` 中 NMS 前的 segment/score 舍入；
2. `apply_sliding_window_nms` 使用 full-precision segment/score 和 `torch.long` labels；
3. NMS 前过滤非有限和非正时长 proposal，并记录计数；
4. 添加至少两个能证明旧舍入会改变 suppression/ranking 的对抗性单元测试；
5. 用冻结的 epoch-59 uniform/physical online 和 EMA checkpoint 全部重放；
6. 输出一个 validator-checked artifact，比较旧 rounded 与新 full-precision 的：

   * 五个 IoU mAP；
   * Avg-mAP；
   * proposal/NMS 决策差异；
   * boundary displacement；
   * 短动作与高 IoU strata；
   * physical-minus-uniform 差值；
7. 该 commit **不得**包含 Q384、interpolation、copy、cross-attention、gap projection、新 loss 或延长训练。

在这项任务通过前，任何 Q-lift 实现或 GPU 训练都属于错误的研究顺序。

[1]: https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html "https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html"
[2]: https://arxiv.org/abs/2604.18274 "https://arxiv.org/abs/2604.18274"
[3]: https://arxiv.org/abs/2105.11107 "https://arxiv.org/abs/2105.11107"
