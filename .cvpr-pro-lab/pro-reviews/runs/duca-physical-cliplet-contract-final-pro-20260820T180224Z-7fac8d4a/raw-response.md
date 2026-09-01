## 1. `FINAL_DECISION`

**CONTINUE**

以本合同作为第二轮且最终的科学裁决，取代原冻结合同中不完整的算法条款，以及首轮答复中错误的 hash、不充分的 GAPPACK、错误的三臂估计量和无依据数值门槛。后续按本合同执行 Builder → 独立 Critic → Evaluator PRE_RUN → 首个完整 seed；**不再请求第三次 Pro 讨论，也不允许 Builder自行补充科学定义**。

当前证据状态仍为：

```text
designed_not_implemented_not_tested
BLOCKED_PRE_RESULT
```

没有实现、PRE_RUN、训练、指标、成本或论文结果。项目既有 Claim Map 也明确表示 learned acquisition、dynamic budget、高 IoU、真实成本和新颖性均未达到 paper-ready。 

---

## 2. `ACCEPTED_SCIENTIFIC_KERNEL`

以下科学内核原样保留：

1. **标题机制**是低成本、部署可见的 S0 scout 预测逐时刻 actionness、start、end；确定性 acquisition 再由这些语义预测导出连续 cliplet 和 per-window outer budget。Scout 不直接预测 frame index、proposal、NMS score 或最终类别。
2. **主重型输入**是原视频中连续的 16-frame cliplet；不同 cliplet 之间允许存在物理时间空隙，但 cliplet 内部必须是连续源帧。
3. **物理时间先于检测后处理**：source frame ID、时间戳和时间支持在 acquisition 前存在；proposal 必须在 filtering、pre-NMS top-k、IoU、NMS、voting 和序列化前进入物理坐标。
4. **实际 VideoMAE 计算必须减少**：`executed_K` 来自真正进入 temporal patch embedding 的 RGB 帧数。先跑 dense backbone 再 mask、padding 到 dense/Kmax 后只改 metadata、或使用缓存 heavy feature，均无效。
5. **固定 `M=24, K=384` 只是控制、归因截面和故障回退**。最终论文主线仍必须通过真实 dynamic M；这与项目最新动态路线中“dynamic budget 必须是核心、fixed K 不能成为最终路线”的边界一致。 
6. Dense、equidistant-uniform 和 random 的既有训练不得重复。只能只读审计 immutable receipt；协议不兼容时，论文级比较保持阻断。
7. Query-Bridge 只能在 S0 基础路线通过后改善 action/start/end 语义。Cycle 仅在 SQ 通过后增加一个条件臂。**Semantic distillation 永久删除**。
8. Detector architecture、assignment、classification/regression loss、NMS、class map 和 official evaluator 不得改变。
9. 旧 65.xx、UVT、Fovea/Query-Bridge、density prototype、U/O/R 与 prefix-budget 路线都是历史或负面诊断，不构成本路线正证据。U/O/R 的第二个可达 firewall bypass 已终止旧修复循环，不得复用或追加第三次修复。 

---

## 3. `CORRECTED_ALGORITHM`

### 3.1 S0 输出、训练标签与损失

对每个有效窗口，S0 只输出：

```text
action_logits[B, Ts]
start_logits[B, Ts]
end_logits[B, Ts]
scout_bin_frame_support[B, Ts, 2]       # 左闭右开
scout_bin_time_support_ns[B, Ts, 2]     # int64 纳秒，左闭右开
```

`scout_bin_frame_support` 必须按时间排序、互不重叠，并完整划分 `[0,T_valid)`。最后一个 bin 的右端必须等于 `T_valid`。不满足时立即失败。

训练标注均来自 canonical training annotation：

* `y_t^a=1`：bin 的物理时间支持与任一 GT 动作区间有正长度交集。
* `y_t^s=1`：任一 GT start 被唯一分配到该 bin。
* `y_t^e=1`：任一 GT end 被唯一分配到该 bin。
* 点边界使用左闭右开分配；恰等于窗口物理右边界的终点分配给最后一个 bin。
* 重叠动作仍为二元标签，不累加类别或事件数。

不训练第四个 uncertainty head。每个语义通道的预测不确定性由 Bernoulli 熵确定：

[
p_t^q=\sigma(z_t^q),\qquad
u_t^q=\frac{-p_t^q\log p_t^q-(1-p_t^q)\log(1-p_t^q)}{\log 2},
\quad q\in{a,s,e},
]

其中 (0\log0=0)。

对每个通道，在完整 canonical training label ledger 上预先计算正例比例 (\pi_q)。要求 (0<\pi_q<1)，否则失败。使用全局平衡 BCE：

[
L_q=\frac1{N_{\rm valid}}\sum_t
\left[
\frac{y_t^q}{2\pi_q}\operatorname{softplus}(-z_t^q)
+
\frac{1-y_t^q}{2(1-\pi_q)}\operatorname{softplus}(z_t^q)
\right],
]

[
L_{\rm S0}=\frac{L_a+L_s+L_e}{3}.
]

不得加入 frame-index loss、K loss、proposal loss、teacher loss、distillation loss 或 detector-output loss。

### 3.2 语义质量与“纯歧义不能升权”

将 bin 预测按其显式 frame support 逐帧展开。对源帧 (i)：

[
g_i^q=\max(0,2p_i^q-1),
]

[
c_i^q=g_i^q(1-u_i^q),\qquad
r_i^q=g_i^q u_i^q.
]

其中：

* (c_i^q) 是置信语义质量；
* (r_i^q) 是仅在存在正语义证据时才生效的歧义风险；
* 当 (p=0.5) 时 (g=0)，因此纯歧义既不能产生 acquisition 质量，也不能单独增加预算；
* (p<0.5) 不产生正证据；
* uncertainty 只能重新分配已有正证据的“置信质量/风险质量”，不能凭空创造重要性。

所有 ([0,1]) 标量以 float64 计算，再用

[
Q=2^{20},\qquad q(x)=\lfloor Qx+0.5\rfloor
]

转换为非负定点整数。候选比较只使用这些整数和 int64 时间，不使用随机容差。

### 3.3 CONTIG 候选与上下文

有效帧数为 (T)。要求 (T\ge16)，否则失败。

每个候选是：

[
C_c={c,c+1,\ldots,c+15},\qquad c=0,\ldots,T-16.
]

候选必须满足：

```text
source_frame_index[c+r] == source_frame_index[c] + r
timestamp_ns[c+r+1] > timestamp_ns[c+r]
```

已选 cliplet 两两不重叠。

对证据帧 (i) 和候选 (C_c)，定义四帧双侧上下文：

[
L_i=\min(4,i),\qquad R_i=\min(4,T-1-i).
]

若 (i\notin C_c)，则 (\chi(i,C_c)=0)。否则：

[
\ell=\min(L_i,i-c),\qquad
r=\min(R_i,c+15-i),
]

[
\phi(x,d)=
\begin{cases}
1,&d=0,\
x/d,&d>0,
\end{cases}
\qquad
\chi(i,C_c)=\min(\phi(\ell,L_i),\phi(r,R_i)).
]

因此，窗口边缘只要求窗口中实际存在的上下文；不存在的左侧或右侧上下文不会导致除零或不可满足条件。

对选择集合 (S)：

[
h_i(S)=\mathbf1[i\text{ 被任一 }C\in S\text{ 覆盖}],
\qquad
\chi_i(S)=\max_{C\in S}\chi(i,C).
]

### 3.4 完全归一化的 coverage certificate

定义七个分量：

[
B_s(S)=
\frac{\sum_i q(c_i^s)h_i(S)}
{\sum_i q(c_i^s)},
\qquad
B_e(S)=
\frac{\sum_i q(c_i^e)h_i(S)}
{\sum_i q(c_i^e)},
]

[
X_s(S)=
\frac{\sum_i q(c_i^s)\chi_i(S)}
{\sum_i q(c_i^s)},
\qquad
X_e(S)=
\frac{\sum_i q(c_i^e)\chi_i(S)}
{\sum_i q(c_i^e)}.
]

Actionness 使用真实帧支持时长 (d_i)（纳秒）：

[
A(S)=
\frac{\sum_i q(c_i^a)d_i h_i(S)}
{\sum_i q(c_i^a)d_i}.
]

边界歧义风险：

[
R(S)=
\frac{\sum_i [q(r_i^s)+q(r_i^e)]\chi_i(S)}
{\sum_i [q(r_i^s)+q(r_i^e)]}.
]

对任一分量，如果分母为零，则该分量定义为 1，表示约束在该窗口上真空满足；它在所有 (S) 上保持常数，不影响边际选择。

每帧必须提供 int64 物理支持：

```text
frame_time_support_ns[i] = [left_ns, right_ns)
```

不得由 nominal FPS 反推。窗口支持为 `[window_left_ns, window_right_ns)`。

将已选 cliplet 的物理支持并集排序，计算：

* 窗口左边界到第一个 cliplet；
* 相邻 cliplet 之间；
* 最后一个 cliplet 到窗口右边界

的所有未覆盖间隔。令最大间隔为 (G_{\rm ns}(S))，窗口时长为 (D_{\rm ns}>0)：

[
P(S)=1-\frac{G_{\rm ns}(S)}{D_{\rm ns}}.
]

空集合的 (P=0)。

最终证书为：

[
V(S)=(B_s,B_e,X_s,X_e,A,R,P).
]

每个分量用 half-up 量化到 ([0,Q])，目标整数为：

[
J_Q(S)=
q(B_s)+q(B_e)+q(X_s)+q(X_e)+q(A)+q(R)+q(P).
]

### 3.5 确定性、可扩展的 nested CONTIG acquisition

[
M_{\rm cap}=\lfloor T/16\rfloor.
]

当前阶段所需最大序列长度记为 (M_{\rm seq})。

选择一个候选后，将 `[0,T)` 中未选择的帧划分成连续 free runs。定义：

[
\operatorname{capacity}(S)
==========================

\sum_{\text{free run }R}\left\lfloor |R|/16\right\rfloor.
]

在第 (m) 步，候选 (C_c) 只有同时满足下列条件才可行：

1. 与 (S_m) 不重叠；
2. 不越界；
3. 加入后仍可完成整个序列：

[
\operatorname{capacity}(S_m\cup{C_c})
\ge M_{\rm seq}-(m+1).
]

从 (S_0=\varnothing) 开始：

[
c_m=
\arg\max_{\text{feasible }c}
\left[
J_Q(S_m\cup{C_c})-J_Q(S_m)
\right].
]

整数增益完全相同时，选择更小的源帧 start index。不得使用随机 tie、GT、teacher、detector prediction、proposal 或 learned direct-index 输出。

[
S_{m+1}=S_m\cup{C_{c_m}}.
]

若不存在可行候选、完成后的 cliplet 数不足、或任一 nested 前缀不满足直接数组检查，则失败。

### 3.6 固定 M

[
M_{\rm fixed}(T)=\min(24,M_{\rm cap}),
\qquad
K_{\rm fixed}=16M_{\rm fixed}.
]

基础阶段设置：

```text
M_seq = M_fixed
selection = S[M_fixed]
```

正常 768-frame 窗口得到 `M=24, K=384`。

### 3.7 后续 dynamic M

唯一候选 support 为：

[
\mathcal M_0={16,20,24,28,32}.
]

它只有在 training-side CAL 的结构/成本预检确认五个值都满足以下条件后才被 seal：

* 真正的不同 heavy execution；
* 无 Kmax padding；
* N16R4 内存可达；
* 每个值的完整成本可测；
* `executed_K=16M`。

任一值失败则整个 dynamic stage 阻断，不替换 support、不缩小 support、不事后另选范围。

短窗口使用：

[
\mathcal M(T)=
\operatorname{sorted_unique}
{\min(m,M_{\rm cap}):m\in\mathcal M_0}.
]

生成到 (\max\mathcal M(T)) 的同一 nested 序列。

定义：

[
D_m(x)=
\max_{v\in V(S_m)}(1-v).
]

在 video-disjoint training-side CAL 上，候选阈值集合是所有实际出现的量化 (D_m/Q)，外加 0 和 1。对每个候选 (\theta)：

[
M_\theta(x)=
\min{m\in\mathcal M(T):D_m(x)\le\theta},
]

不存在满足项时取 (\max\mathcal M(T))。

使用 CAL 上实际测得的 per-row full-stack cost，选择与 fixed `M=24` 平均成本最接近的 (\theta)。完全相同时依次选择：

1. 更小的 (\theta)；
2. 更小的平均 executed M；
3. 更小的数值阈值序号。

只有成本差位于第 6 节定义的实测 repeatability band 内时才 seal；否则 dynamic matched-cost stage 失败。

最终：

[
M_{\rm dynamic}(x)=M_{\theta^*}(x).
]

该规则使所有激活的 certificate 分量都必须达到统一的 (1-\theta^*) 水平，而不是由 Builder 任意设置七个阈值。

---

## 4. `CORRECTED_GAPPACK_CONTROL`

### 4.1 必要前提

从**实例化后的 official VideoMAE-S config 和 module**直接读取 temporal patch/tubelet kernel 与 stride，记为 (\tau)。

GAPPACK 只在下列条件全部成立时存在：

1. temporal kernel = temporal stride = (\tau)；
2. temporal padding 不使 atom 跨越输入块；
3. (16\bmod\tau=0)；
4. (L=16/\tau\ge2)；
5. (M\ge2)；
6. patch embedding 后每个输入块精确产生 (L) 个有序 temporal slots；
7. backbone 到共享 reconstruction 前不进行 temporal pooling、merging、dropping、排序或长度压缩；
8. 每个输出 temporal slot 可一一绑定到输入 `(block, atom_slot)` 的**位置支持**。

Transformer attention 可以混合内容；这里要求的是位置槽一一对应，不声称输出特征的因果感受野只属于单个 atom。

任一条件无法由真实实例证明，或必须改变官方 backbone/detector 语义才能满足，则：

```text
GAPPACK_STATUS = DROPPED_FAIL_CLOSED
```

不得使用近似 atom、事后时间戳猜测、额外 temporal head 或替代 backbone。基础实验退化为 `FZ_CONTIG / JT_CONTIG` 两臂。

### 4.2 一般 M、L 下的精确非恒等双射

CONTIG 第 (m) 个 cliplet 的第 (r) 个 atom 为：

[
A_{m,r}
=======

[c_m+r\tau,\ldots,c_m+(r+1)\tau-1],
]

其中：

[
m=0,\ldots,M-1,\qquad r=0,\ldots,L-1.
]

定义 forward permutation：

[
\pi(m,r)=((m+r)\bmod M,\ r).
]

因此 GAPPACK block (b) 的 slot (r) 放置：

[
G_{b,r}=A_{(b-r)\bmod M,\ r}.
]

inverse permutation 为：

[
\pi^{-1}(b,r)=((b-r)\bmod M,\ r).
]

证明：

[
\pi^{-1}(\pi(m,r))
==================

((m+r-r)\bmod M,r)
=(m,r),
]

[
\pi(\pi^{-1}(b,r))
==================

((b-r+r)\bmod M,r)
=(b,r).
]

所以它对任意正整数 (M,L) 是双射。且当 (M\ge2,L\ge2) 时，所有 (r=1) 的 atom 都从 block (m) 移到 block ((m+1)\bmod M)，因此必为非恒等变换。

atom 内部帧顺序保持正向连续；不同 atom 之间允许物理跳跃。不得再按物理时间重新排序，否则会改变上述定义。

### 4.3 直接数组不变量；禁止 hash/checksum

准入只使用直接 tensor/array equality：

1. `sort(flat(frame_ids_CONTIG)) == sort(flat(frame_ids_GAPPACK))`；
2. `sort(flat(atom_ids_CONTIG)) == sort(flat(atom_ids_GAPPACK))`；
3. `sort(forward_permutation) == arange(M*L)`；
4. `inverse[forward[i]] == i` 对所有 (i) 成立；
5. `forward[inverse[j]] == j` 对所有 (j) 成立；
6. heavy 输出执行 inverse permutation 后，ordered atom ID、frame support、timestamp support 与 CONTIG 的 ordered arrays 完全相等；
7. A/B 的 M、K、RGB multiset、preprocessing、weights、optimizer、loss、detector、physical timestamp 输出完全相同；
8. GAPPACK 的 packed atom matrix 必须与 CONTIG atom matrix不相等，否则判定为恒等控制并失败。

**不生成、不要求、不比较任何 manifest hash、atom hash、permutation hash、inverse-permutation hash 或 checksum。**

### 4.4 唯一合法估计量

若 GAPPACK 准入：

```text
FZ_CONTIG − FZ_GAPPACK_ATOM
```

只估计：

> 16-frame bundle 中 temporal atom 分组、temporal slot presentation、positional context 和跨 atom 非线性交互的合并效应。

它不是纯 continuity 唯一效应，也不估计 selector 质量、dynamic-budget 质量或 detector 质量。

---

## 5. `FINAL_EXPERIMENT_LADDER`

### 5.1 第一阶段：最小、非冗余、完整数据基础门

实验身份：

```text
DUCA-PHYSICAL-BASE-GATE-v001
```

先训练一次共享的 `S0_TERMINAL`：

* 使用完整 canonical THUMOS14 training population；
* deterministic low-resolution scout view；
* 仅 `L_S0`；
* 60 个完整 semantic epochs；
* terminal checkpoint；
* 无 official evaluator、无 best checkpoint；
* 该共享 S0 不按 detector 臂重复训练。

随后以 seed `3203700` 运行完整 60-epoch/full-schedule detector matrix：

| 臂                 | 定义                                                                           | 唯一估计内容                               |
| ----------------- | ---------------------------------------------------------------------------- | ------------------------------------ |
| `FZ_CONTIG`       | `S0_TERMINAL` 冻结；每个训练窗口的 `S24` 直接数组冻结；CONTIG heavy path                      | 基础输入合同                               |
| `FZ_GAPPACK_ATOM` | 仅在第 4 节准入；与 `FZ_CONTIG` 使用完全相同的逐窗口 frame/atom arrays                         | temporal-presentation bundle         |
| `JT_CONTIG`       | 与 FZ 相同初始化；CONTIG；训练中继续由 `L_S0` 更新 scout；hard selection 不接 detector gradient | frozen-vs-joint semantic-scout drift |

所有 scout 输入均使用确定性 view；随机 spatial augmentation 只在选定 RGB materialization 之后执行，并跨臂共享 RNG。

正确估计量：

* `FZ_CONTIG − FZ_GAPPACK_ATOM`：时间呈现 bundle；
* `JT_CONTIG − FZ_CONTIG`：冻结与联合语义 scout 漂移/优化效应。

三者都使用 S0 和固定 `M=24`，因此：

* **不估计 selector 质量**；
* **不估计 dynamic-budget error/value**；
* **不估计 Query、cycle 或 distillation**。

若 GAPPACK 被 fail-closed 删除，则第一阶段只运行两个 CONTIG 臂，不替换控制。

### 5.2 Selector 质量门：只读 compatible uniform receipt

Selector 质量只由第一阶段选出的最佳 CONTIG 臂与 compatible immutable exact-uniform K384 receipt 比较。

Receipt 必须直接证明以下完全一致：

* canonical training/evaluation split；
* ordered window ledger；
* VideoMAE-S 与 detector config；
* pretrained initialization；
* assignment 和 detector losses；
* 60 epochs 与 successful optimizer-update count；
* optimizer、scheduler、AMP、EMA；
* 相同 seed 或完整公共 seed 集；
* final-EMA 主规则和 final 诊断；
* physical-time mapping；
* soft-NMS、voting、class map、official evaluator；
* effective/executed K384；
* full-stack cost边界。

任一项不兼容，第一阶段仍可作为内部机制实验，但：

```text
PAPER_LEVEL_SELECTOR_NONINFERIORITY = BLOCKED
SQ_STAGE = BLOCKED
DYNAMIC_STAGE = BLOCKED
```

不得以此授权重训 uniform。项目既有实验计划与 anti-repetition 规则也要求先做公平 fixed-K 归因，并禁止用协议不匹配或部分结果代替正式证据。 

### 5.3 第二阶段：条件性 SQ

只有基础 CONTIG 相对 compatible uniform receipt 通过第 6 节 non-inferiority，并且 S0 calibration/coverage 合格时，增加一个：

```text
SQ_CONTIG_M24
```

SQ 规则：

* class-agnostic query；
* 只与 scout 表征交叉注意；
* 最终仍只输出 action/start/end logits；
* query tensor 不出现在 acquisition API；
* 不输出 index、M、proposal、NMS score；
* 使用与入选基础臂相同的 frozen 或 joint 优化方式；
* 固定 CONTIG、M=24。

SQ 必须同时改善：

1. training-side semantic Brier/ECE；
2. certificate coverage；
3. official localization；

否则删除 SQ，并由 S0 进入后续阶段。

### 5.4 第三阶段：条件性 cycle

只有 SQ 同时通过语义门和定位门，才增加一个：

```text
SQC_CONTIG_M24
```

Cycle target：

* 只来自同一 training sample 的 post-heavy semantic representation；
* target 与产生 target 的整条路径 detach；
* 不含 detector proposal、score、NMS、GT geometry 或 teacher；
* inference 完全删除；
* 只能优化 action/start/end logits；
* 不输出 index 或 M。

Cycle 不通过即删除。**不得恢复 SQD 或任何 semantic distillation。**

### 5.5 第四阶段：dynamic-budget 唯一效应

使用最终保留的 S0、SQ 或 SQC fixed-M24 receipt，且不得重训该 fixed arm。新增：

1. `SEMANTIC_DYNAMIC`；
2. `K_SHUFFLE`；
3. `ACTIONNESS_ONLY_DYNAMIC`。

#### `SEMANTIC_DYNAMIC`

使用第 3.7 节的七分量 certificate、nested CONTIG sequence 和 CAL-sealed (\theta^*)。

#### `K_SHUFFLE`

每行 canonical key：

```text
(video_id,
 window_start_source_frame,
 window_end_source_frame,
 valid_length)
```

初始 stratum：

```text
(M_cap,
 floor((valid_length-1)/16),
 decode_cost_decile)
```

`decode_cost_decile` 在 training-side CAL 上按 nearest-rank 十分位冻结；等于边界的值进入较低 bin。

按 stratum ID 和 row key 词典序排序。使用一个全局：

```text
PCG64(seed=20260820180224)
```

依次处理 strata，并对每个 stratum 执行 Sattolo：

```text
for i = n-1 ... 1:
    j = rng_integer_in_[0, i-1]
    swap(order[i], order[j])
```

若 stratum 行数小于 2 或只有一个 M 值：

1. 先与同一 `M_cap/valid_length_bin` 下最近的 decode-cost bin 合并，距离相同取较低 bin；
2. 仍无效时，与相邻 valid-length bin 合并，距离相同取较低 bin；
3. 同一 `M_cap` 全部合并后仍只有一个 M 值，则 dynamic stage 因不可识别而失败。

目标行接收 donor 行的 M，但使用目标行自身的 nested sequence 前缀。必须直接比较每个 stratum 的原始/置换 M arrays 和 histogram；不使用 hash。

#### `ACTIONNESS_ONLY_DYNAMIC`

* acquisition objective 只保留 `A` 和 `P`；
* start/end、`X_s/X_e` 和 `R` 完全禁用；
* 在每个最终 cost stratum 内，按

[
D_{AP}(S_{24})=\max(1-A(S_{24}),1-P(S_{24}))
]

从大到小排序，tie 使用 canonical row key；

* 将同 stratum 中 `SEMANTIC_DYNAMIC` 的 M multiset 从大到小逐项分配；
* 每行执行 actionness-only nested sequence 的对应前缀。

因此它与 semantic dynamic 在每个 cost stratum 中具有完全相同的 M histogram，却不使用 boundary evidence。

#### Dynamic 阶段估计量

* `SEMANTIC_DYNAMIC − FIXED_M24`：内容相关动态预算在 matched full-stack cost 下的总价值；
* `SEMANTIC_DYNAMIC − K_SHUFFLE`：content-to-budget 对应关系的价值；
* `SEMANTIC_DYNAMIC − ACTIONNESS_ONLY_DYNAMIC`：start/end、上下文和边界歧义风险的独立价值。

只有 semantic dynamic 同时超过后三者要求中的 fixed、K-shuffle 和 actionness-only，才能支持最终标题。

---

## 6. `STATISTICAL_AND_COST_GATES`

### 6.1 删除无依据旧常数

首轮中的：

```text
+0.50 pp
-0.20 pp
±1% wall-clock
```

全部删除。现有材料没有提供足以支持这些精度的 compatible seed variance、功效或目标硬件重复性证据。

### 6.2 官方指标的不确定性

单 seed 使用 exactly 10,000 次 paired video-cluster bootstrap：

* cluster 是完整 video；
* 同一 resample 同时应用于两臂；
* 每次从 pooled predictions 重新运行 official evaluator；
* 禁止平均 per-video AP；
* 报告 point、LCB95、UCB95。

三 seed 使用 exactly 10,000 次层次 bootstrap：

1. 外层有放回采样 seed；
2. 每个被采样 seed 内有放回采样 video；
3. 每次重新计算 pooled official evaluator；
4. paired arm 使用相同 seed/video multiplicity。

### 6.3 Practical-equivalence margin 的唯一冻结程序

当前不直接授权任何 pp margin。

在 official evaluation 解封前，Evaluator 只能从**既有、compatible、immutable、training-side CAL exact-uniform K384 receipts**冻结 margin；禁止新训练 uniform。

每个指标 (k\in{\text{Avg},0.6,0.7,\text{short}})：

1. 至少需要三个 compatible seed；
2. 对同一 uniform protocol 的所有无序 seed pair，在 CAL pooled predictions 上计算点差；
3. 收集绝对差值；
4. 定义：

[
\delta_k=
\text{nearest-rank 95th percentile of absolute null seed differences}.
]

该 (\delta_k) 在 official evaluation 前写入 PRE_RUN，并永久冻结。

若不足三个 compatible receipts，或缺少 training-side CAL predictions：

```text
delta_k = UNDEFINED
```

后果是：

* 可以运行内部 A/B/C 机制实验；
* 可以报告 paired uncertainty；
* 不得声称 practical superiority 或 paper-level non-inferiority；
* 不得进入 SQ、cycle 或 dynamic；
* 不得因此重训 uniform。

### 6.4 决策规则

#### Presentation

仅当：

```text
LCB95(FZ_CONTIG − FZ_GAPPACK, Avg-mAP) > 0
且
LCB95(FZ_CONTIG − FZ_GAPPACK, mAP@0.7) >= 0
```

才允许写“CONTIG presentation 有定位价值”。

否则只把 CONTIG 保留为预训练兼容性的工程默认，不写机制增益。

#### Frozen vs joint

只有当：

```text
LCB95(JT − FZ, Avg-mAP) > 0
且
LCB95(JT − FZ, mAP@0.7) >= 0
且
semantic calibration/coverage 改善
```

才保留 JT。其余所有情况，包括 CI 跨零，均选更简单的 FZ。

#### Selector non-inferiority

若 (\delta) 已冻结：

```text
LCB95(DUCA − uniform, Avg-mAP) > -delta_Avg
且
LCB95(DUCA − uniform, mAP@0.7) > -delta_07
```

才通过基础门。

Practical superiority 还要求：

```text
point_difference > delta_metric
且
LCB95 > 0
```

高-IoU 改善必须分别对 mAP@0.6、mAP@0.7 满足 `LCB95>0`。

#### First-seed expansion

第一 seed 只是完整数据/full-schedule 扩展门，不是论文证据。

若 margin 已定义，任一主对比出现：

```text
UCB95(Delta_Avg) < -delta_Avg
或
UCB95(Delta_07) < -delta_07
```

则停止该机制。

margin 未定义时，只在 `UCB95<0` 的明确负效应下停止；否则可扩展，但仍不能形成 practical claim。

### 6.5 成本 repeatability margin

在读取官方指标前，在最终 N16R4 资源上：

1. 使用同一个 fixed-M24 reference arm；
2. 同一 ordered CAL window workload；
3. 一次完整 warm-up；
4. 十次完整 measured repetitions；
5. 无并发任务、无动态频率策略变化；
6. 记录每个窗口的 full-stack cost。

对相邻 repetition 的同一窗口计算：

[
d=|\log C_{r+1}-\log C_r|.
]

定义：

[
\epsilon_{\rm cost}
===================

\text{nearest-rank 95th percentile of all }d.
]

A/B 被认为成本等价，仅当其 paired 90% bootstrap CI 满足：

[
\log(C_A/C_B)\subseteq[-\epsilon_{\rm cost},+\epsilon_{\rm cost}].
]

Dynamic 的 CAL cost matching 也使用同一 band。没有可测 repeatability receipt 时，不得使用任意百分比替代。

### 6.6 真实效率门

Full-stack cost 必须包括：

```text
video open/decode
dense low-resolution scout preparation
scout forward
certificate/acquisition
selected full-resolution materialization
CPU transform
H2D
temporal patch embedding
VideoMAE backbone
GAPPACK inverse permutation（若存在）
physical feature reconstruction
detector
physical-coordinate conversion
filtering/top-k/NMS/voting
serialization
```

真实节省要求：

1. `executed_K` 由 patch-embedding 实际输入计数；
2. dense/Kmax padding 工作全部入账；
3. 相对 dense 的 cost log-ratio 90% UCB 小于 `-epsilon_cost`；
4. scout 与 reconstruction 后仍有净节省。

Energy 只有在功率计、采样频率和 idle-baseline receipt 已校准时报告；否则字段为 `NOT_MEASURED`。

---

## 7. `BUILDER_CONTRACT`

### 7.1 唯一 clean candidate

立即实现的候选身份：

```text
DUCA_PHYSICAL_CLIPLET_CONTIG_S0-v001
```

Coordinator 必须先绑定共享 official AdaTAD receipt 的 exact clean repository revision。该 base 必须：

* clean worktree；
* 不继承 SparseHead Route-T；
* 不启用 `pc_ot_mras_*`；
* 不使用旧 U/O/R、density prototype 或 prefix-budget；
* 不启用 modified physical-grid detector head；
* 解析后 detector 与 official AdaTAD/ActionFormer 一致。

Exact clean revision 未绑定时 Builder 不得自行选择 base，直接返回 `BLOCKED_CLEAN_BASE_UNBOUND`。这不是新的科学讨论。

### 7.2 输入与输出 tensor

每个窗口显式提供：

```text
video_id
window_id
window_start_source_frame[int64]
window_end_source_frame[int64]
valid_length[int64]
source_frame_index[int64, T]
timestamp_ns[int64, T]
frame_time_support_ns[int64, T, 2]
window_time_support_ns[int64, 2]
valid_mask[bool, T]
scout_input
scout_bin_frame_support[int64, Ts, 2]
scout_bin_time_support_ns[int64, Ts, 2]
coord_state = PHYSICAL_SOURCE
```

Acquisition 输出：

```text
nested_cliplet_start[int64, B, M_seq]
nested_cliplet_frame_index[int64, B, M_seq, 16]
nested_cliplet_timestamp_ns[int64, B, M_seq, 16]
requested_M[int64, B]
executed_M[int64, B]
requested_K[int64, B]
executed_K[int64, B]
coverage_vector[int64, B, M_seq, 7]   # Q=2^20
coverage_deficit[int64, B, M_seq]
```

不输出任何 hash/checksum 字段。

### 7.3 Provenance labels

每个可进入 scout/acquisition 的 tensor 必须携带 source class：

```text
DEPLOY_VISIBLE_VIDEO
TRAIN_ANNOTATION_SEMANTIC_ONLY
FORBIDDEN_GT_GEOMETRY_AT_INFERENCE
FORBIDDEN_TEACHER
FORBIDDEN_DETECTOR_OUTPUT
FORBIDDEN_METRIC
FORBIDDEN_CACHE
```

Inference 只允许 `DEPLOY_VISIBLE_VIDEO`。任一 forbidden provenance 到达 scout、certificate、M 或 cliplet index API 即失败。

### 7.4 梯度和 detach

* `L_S0` 更新 S0；
* hard cliplet choice、M choice 和 integer arrays 无梯度；
* detector gradient 不穿过 index、M 或 certificate；
* detector 不读取 semantic labels；
* FZ 的全部 S0 参数 `requires_grad=false`；
* JT 只由 `L_S0` 更新 S0；
* Query 只由相同 semantic losses 更新；
* cycle target 与生成 target 的路径全部 detach；
* 不允许 straight-through direct-index、Gumbel top-k、teacher 或 detector-utility surrogate。

### 7.5 Heavy execution

CONTIG 输入必须是：

```text
[N_executed_cliplets, C, 16, H, W]
```

或等价 bucketed tensor，其中 bucket 只包含同一 executed M。

禁止：

* 完整 dense VideoMAE 后 mask；
* padding 所有样本到 `max(M)` 后不计 padding；
* 对 padding 运行 patch embedding/backbone；
* 使用缓存 heavy features；
* 复制 requested K 生成 executed K。

`executed_K` 必须由 patch embedding 接收到的真实 RGB frame count产生。

### 7.6 物理时间状态机与 reconstruction

合法状态：

```text
PHYSICAL_SOURCE
  -> PHYSICAL_CLIPLET
  -> PHYSICAL_ATOM
  -> PHYSICAL_DENSE
  -> PHYSICAL_SECONDS
```

`SELECTED_RANK` 不得进入 detector 后处理。

每个 heavy temporal atom 输出携带：

```text
atom_id
physical_support_start_ns
physical_support_end_ns
physical_center_ns
```

GAPPACK 在 reconstruction 前先 inverse permutation。

重建到 official detector temporal grid 时：

1. 按 `physical_center_ns` 严格排序；
2. 对每个 official physical target-grid center，使用相邻两个 atom feature 线性插值；
3. 左右凸包外使用最近 atom feature；
4. 相同时间、逆序时间或缺少 support 时失败；
5. 所有臂使用完全相同的 reconstruction；
6. reconstruction 不调用 VideoMAE。

Detector raw proposal 在任何 filtering/top-k/IoU/NMS 前，通过 official physical-grid timestamp map 转为 `PHYSICAL_SECONDS`。映射恰好一次；unknown state 或 double mapping 失败。

### 7.7 Checkpoint 与 resume

每 5 epoch 保存完整恢复点：

```text
epoch_005, 010, ..., 060
```

保留：

```text
latest-3
epoch_020
epoch_040
epoch_060
final
final-EMA
```

必须恢复：

* detector；
* S0/Query；
* optimizer；
* scheduler；
* AMP scaler；
* EMA；
* epoch 和 successful update；
* Python/NumPy/Torch/CUDA RNG；
* sampler/DataLoader state；
* frozen selection arrays；
* dynamic support/θ/strata；
* cost-ledger cursor。

`final-EMA` 是唯一 primary checkpoint；`final` 是预注册 secondary diagnostic。不得二选一，不得使用中间 best。

### 7.8 Config、launcher 与 tests

基础交付至少包含：

```text
duca_s0_semantic_pretrain_v001.py
duca_fz_contig_m24_v001.py
duca_jt_contig_m24_v001.py
duca_fz_gappack_atom_m24_v001.py   # 仅在 precondition PASS 时启用
duca_physical_base_gate_n16r4_v001.sh
```

DUCA launcher 必须在 config merge 前拒绝所有非空 `--cfg-options`。只允许显式 `--resume-from` 指向同 identity checkpoint；其他 runtime override 全部失败。

必须运行无数据或 synthetic property tests：

* label 唯一分配；
* entropy、纯歧义零质量和 zero-denominator；
* context 边界；
* physical-gap 纳秒计算；
* fixed-point coverage；
* nested determinism；
* future-capacity feasibility；
* cliplet 连续、唯一、无重叠；
* batch size/permutation/duplication invariance；
* real executed-K；
* hidden dense path rejection；
* pre-filter/pre-NMS physical mapping；
* unknown/double mapping rejection；
* full-state resume；
* GAPPACK 前提、双射、non-identity 和 direct equality；
* config/detector/NMS/evaluator equality；
* nonempty `--cfg-options` rejection。

### 7.9 Exact fail-closed 条件

任一项出现即阻断候选：

* dirty/contaminated base；
* unresolved config inheritance；
* semantic bin support 不构成完整分区；
* non-finite logits；
* `T<16`；
* cliplet 非连续、重叠、重复或越界；
* nested sequence 无法完成；
* tie 非确定；
* hidden dense/Kmax path；
* padding compute 未入账；
* source time/support 缺失；
* selected-rank 到达 filtering/NMS；
* detector、loss、assignment、NMS、evaluator 改变；
* forbidden provenance；
* GAPPACK 无一一 atom support；
* GAPPACK permutation/round-trip direct array失败；
* resume state 不完整；
* official metric 用于 threshold、checkpoint、seed、M 或机制选择。

---

## 8. `CRITIC_AND_EVALUATOR_CONTRACT`

### 8.1 独立 Critic

Critic 必须在冻结 Builder snapshot 上逐项验证：

1. clean base 与污染代码隔离；
2. S0 labels、loss、entropy 和证据质量公式；
3. zero-mass 行为；
4. 七分量 certificate 的分母、context 和物理 gap；
5. fixed-point objective 与 lower-start tie；
6. future-capacity feasibility；
7. FZ/JT 梯度边界；
8. heavy path 是否真实稀疏；
9. executed-K 是否来自 patch embedding；
10. reconstruction 和 pre-NMS physical time；
11. detector/NMS/evaluator 不变量；
12. checkpoint/resume；
13. launcher override firewall；
14. 三臂估计量是否只解释 presentation 和 scout drift；
15. compatible uniform receipt 是否逐字段满足；
16. GAPPACK resolved-(\tau) 前提、双射和 direct equality；
17. 不存在任何新增 hash/checksum 依赖；
18. K-shuffle 和 actionness-only 的确定性定义。

Critic 返回且只能返回：

```text
DUCA_FINAL_CONTRACT_STATIC_PASS
或
DUCA_FINAL_CONTRACT_BLOCKED
```

若 GAPPACK 单独失败但 CONTIG 主路径完全通过，Critic应记录：

```text
GAPPACK_DROPPED_FAIL_CLOSED
DUCA_FINAL_CONTRACT_STATIC_PASS
```

不得把 GAPPACK 失败扩大为 CONTIG 路线失败。

### 8.2 Evaluator PRE_RUN 必须绑定

Evaluator 在任何 N16R4 submission 前冻结：

* exact Project、final contract 和 clean revision；
* exact resolved config文本；
* Builder changed-file list；
* Critic disposition；
* GAPPACK `ENABLED` 或 `DROPPED_FAIL_CLOSED`；
* canonical THUMOS14 training/evaluation manifests；
* ordered window ledger；
* annotation/category map；
* official evaluator、soft-NMS 和 voting；
* seeds `3203700 / 1677630095 / 1453526567`；
* exact arm matrix；
* complete 60-epoch schedule和successful-update integer；
* final-EMA/final rule；
* five-epoch recovery；
* full-state resume；
* N16R4 resource tuple；
* output root和metric embargo；
* compatible uniform receipt逐字段结论；
* (\delta_k) 数值或 `UNDEFINED`；
* (\epsilon_{\rm cost}) 的 CAL 测量程序；
* full-stack ledger字段；
* stop rules。

PRE_RUN 必须确认：

```text
official evaluation has not been read
no duplicate dense/uniform/random training is scheduled
no subset efficacy run exists
all new arms use full canonical data/full schedule
```

任一字段缺失时：

```text
PRE_RUN = BLOCKED
```

全部通过时：

```text
PRE_RUN = PASS_FIRST_FULL_SEED
```

该 PASS 直接触发第 9 节的 first-seed submission，不再请求 Pro 或人类进行路线选择。

---

## 9. `REMOTE_EXPERIMENT_CONTRACT`

### 9.1 数据与评估

每个新训练臂必须使用：

* 完整 canonical THUMOS14 training population；
* 完整 canonical official evaluation population；
* official annotation、category map、blocked-video 规则；
* official pooled evaluator；
* official tIoU `0.3:0.1:0.7`；
* official soft-NMS/voting。

不得使用 subset、160/40、小样本或 local CPU 结果作为 efficacy/cost claim。

Training-side CAL 只用于在 official metric 解封前冻结 dynamic threshold、cost repeatability 和统计 margin；official evaluation 不参与这些选择。

### 9.2 第一 seed

第一 seed 固定为：

```text
3203700
```

运行第 5.1 节完整 arm matrix。每个臂都是 full-data、full-60-epoch/full-schedule run，不称为 subset pilot。

所有臂的 predictions、cost ledger 和 identity receipts 完成并封存前，不显示任何 arm metric。指标同时解封。

第一 seed 只能决定是否扩展；不能形成论文 claim。

通过扩展门后，追加：

```text
1677630095
1453526567
```

不得在看到结果后换 seed。

### 9.3 N16R4 资源

每个训练 job：

```text
cluster_class = N16R4
Slurm only
gpus = 1
cpus_per_task = 6
source /etc/profile before module/environment setup
in_process_device = cuda:0
do_not_override_CUDA_VISIBLE_DEVICES = true
login_node_training = forbidden
```

GPU model、GPU memory、host memory、partition、walltime 和 software environment 必须与 compatible official AdaTAD resource receipt 完全一致，由 Evaluator写入 PRE_RUN；缺少该 receipt 时不得由 Builder猜测，PRE_RUN 阻断。

成本测量期间禁止并发 job，固定环境和频率策略。

### 9.4 训练公平性

所有 paired arms 共享：

* pretrained weights；
* detector initialization；
* ordered windows；
* augmentation RNG；
* optimizer、LR、scheduler；
* AMP、EMA；
* batch/global batch；
* successful update count；
* 60 epochs；
* detector losses；
* NMS/evaluator；
* checkpoint rule。

Infrastructure failure后的恢复只允许同一 code/config/data/seed/output identity。Identity 改变必须创建全新 root，并使整个 paired seed 重跑；不得只补失败臂。

### 9.5 输出 receipts

每个臂必须输出：

```text
exact code revision and changed-file list
resolved config text
ordered data/window manifest
seed and RNG state
checkpoint inventory
final checkpoint
final-EMA checkpoint
official prediction JSON for final
official prediction JSON for final-EMA
requested/executed M and K per window
nested cliplet starts and frame arrays
physical timestamp/support arrays
GAPPACK forward/inverse arrays if enabled
coordinate-state trace
full-stack per-window cost ledger
p50/p95 latency
throughput
peak GPU/host memory
resume receipts
forbidden-access receipt
official evaluator output
first-failure field
```

身份与公平性通过直接文本/array equality 审计；不新增 manifest、atom 或 permutation checksum。

### 9.6 远端停止规则

立即停止并不解封科学结果：

* structural/Critic/PRE_RUN 不一致；
* hidden dense path；
* executed-K 或 physical-time 失败；
* detector/NMS/evaluator drift；
* checkpoint/resume 不完整；
* official metric被提前读取；
* paired seed 有一臂无法以同 identity 完成。

第一阶段机制停止：

* GAPPACK 显示明确负效应时，只删除 GAPPACK claim；
* JT 未严格通过时固定 FZ；
* compatible uniform receipt 缺失或不兼容时，基础 paper gate、SQ 和 dynamic 阻断，不重训 uniform；
* 选择臂相对 compatible uniform 明确劣于 frozen non-inferiority margin 时停止后续机制。

SQ/cycle 停止：

* 只改善辅助 loss、attention 或 calibration，而不改善 certificate 和定位；
* high-IoU 明确受损；
* query/cycle 影响 index/M API。

Dynamic 停止：

* (\mathcal M_0) 任一预算不能真实执行；
* CAL 无法在 (\epsilon_{\rm cost}) band 内匹配 fixed M24；
* semantic dynamic 不超过 K-shuffle；
* semantic dynamic 不超过 actionness-only；
* mAP@0.6/0.7 或 short-action 明确受损；
* M 最终只由长度、平均 actionness 或 decode cost解释；
* scout/reconstruction 抹去 dense→sparse 的净 full-stack 节省；
* fixed/dynamic 比较依赖重训已存在 dense/uniform/random。

负结果必须保存为该机制的终止证据，不允许静默调 M support、阈值、loss、seed、split 或 checkpoint。

---

## 10. `PUBLICATION_BOUNDARY`

### 10.1 唯一论文主张

只有全部基础、selector、semantic 和 dynamic gates 通过后，论文才允许提出一个合并主张：

> 对离线 TAD，部署可见的 actionness、start 和 end 语义可以形成一个显式归一化的端点—上下文—动作内部—边界歧义—物理间隔覆盖证书；由该证书确定的 nested、物理连续 cliplet 与 matched-cost per-window dynamic budget，能够在不改变 AdaTAD/ActionFormer detector、损失和 official post-processing 的情况下，真实减少重型 VideoMAE 输入计算，并相对 fixed、K-shuffle 和 actionness-only controls 更好地保护高-tIoU 定位。

该主张是组合创新，不得拆成“智能选帧”“动态 token”“物理时间”“coverage”中的任一单点优先权。现有 literature/gap 文件也明确指出，generic importance scorer 加 top-k 不足以构成新颖性，必须依赖 TAD 边界、物理 acquisition 和强 cost-matched controls 的组合。

### 10.2 Anti-claims

不得声称：

* 直接 index learning 是主方法；
* Query-Bridge 或 cycle 是标题贡献；
* GAPPACK 是主方法；
* CONTIG/GAPPACK 的差异是纯 continuity 唯一效应；
* nominal K 等同于真实效率；
* single seed、subset、训练侧 CAL 或 infrastructure test 是论文证据；
* 历史 65.xx/UVT/Fovea 支持本路线；
* 无 compatible uniform receipt 时仍达到非劣性；
* 动态预算优于 fixed，而未超过 K-shuffle/actionness-only；
* 修改 detector 后仍称 pure pre-backbone plugin；
* state of the art、跨数据集或跨 detector 泛化。

### 10.3 Novelty invalidators

以下结果分别使对应主张失效：

* `SEMANTIC_DYNAMIC ≈ K_SHUFFLE`：content-to-budget 映射无价值；
* `SEMANTIC_DYNAMIC ≈ ACTIONNESS_ONLY`：boundary-aware 创新失效；
* 收益只在低 tIoU：高精度 TAD 机制失效；
* M 只随长度、背景比例或总 actionness 变化：退化为普通 adaptive compute；
* dense/Kmax hidden compute：效率主张失效；
* physical time 晚于 filtering/NMS：定位证据失效；
* 必须改变 detector/head/loss：pure plugin 边界失效；
* Query/cycle 只降低 auxiliary loss：语义协同主张失效；
* CONTIG/GAPPACK 无差异：只删除 temporal-presentation 子主张，不自动否定 semantic dynamic；
* selector 未通过 compatible uniform：整个后续论文路线阻断。

### 10.4 Result-to-claim 限制

* `FZ_CONTIG vs GAPPACK` 只支持 presentation bundle；
* `JT vs FZ` 只支持 semantic-scout drift；
* DUCA vs compatible uniform 才支持 selector 质量；
* SQ/SQC 还必须有 semantic calibration/coverage；
* dynamic/fixed/K-shuffle/actionness-only 才支持 dynamic-budget；
* full-stack receipts 才支持效率；
* 三 seed、冻结 margin 和层次 uncertainty 才支持 paper-level 结论；
* official result promotion 前，所有结论保持 candidate evidence。

---

## 11. `NEXT_ACTION`

**立即交给 Builder，不再提交 plan-only 文档。**

唯一任务：

```text
BUILDER_DUCA_PHYSICAL_CLIPLET_FINAL-v001
```

Builder 在 Coordinator 绑定的 clean official AdaTAD base 上直接实现：

1. S0 三语义 head、标签和 balanced BCE；
2. entropy-derived uncertainty 与 gated confidence/risk；
3. 七分量 normalized certificate；
4. fixed-point deterministic nested CONTIG；
5. fixed M24；
   6.真实 sparse VideoMAE execution；
6. physical-time reconstruction 和 pre-filter/pre-NMS mapping；
7. FZ/JT 两个基础 configs；
8. resolved-(\tau) precondition 检查；
9. 仅在 precondition PASS 时实现精确 GAPPACK 双射；
10. direct tensor/array equality tests；
11. full-state checkpoint/resume；
12. full-stack ledger；
13. reject-all-nonempty-`--cfg-options` launcher。

Builder 返回：

```text
BUILDER_DUCA_PHYSICAL_CLIPLET_FINAL-v001.md
DUCA_PHYSICAL_CLIPLET_FINAL-v001.patch
DUCA_PHYSICAL_CLIPLET_CHANGED_FILES-v001.txt
DUCA_PHYSICAL_CLIPLET_RESOLVED_CONFIGS-v001/
DUCA_PHYSICAL_CLIPLET_PROPERTY_TESTS-v001/
DUCA_PHYSICAL_CLIPLET_DIRECT_ARRAY_RECEIPTS-v001/
DUCA_PHYSICAL_CLIPLET_UNRESOLVED_BLOCKERS-v001.json
```

不得包含 mAP、训练日志、GPU/Slurm execution、official predictions、PRE_RUN 声明、Git push 或性能主张。

完整 Builder snapshot 随后自动进入独立 Critic；Critic PASS 后进入 Evaluator PRE_RUN；PRE_RUN PASS 后按第 9 节提交 seed `3203700` 的完整 full-data/full-schedule 基础矩阵。**该链路由本最终合同直接授权，不再发起第三次科学路线讨论。**
