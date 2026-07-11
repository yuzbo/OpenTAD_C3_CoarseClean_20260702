# ChronoTransport CT-P3R-3S-r2 书面修订候选规格

日期：2026-07-12

协议编号：`CT-P3R-3S-r2`

决策状态：`REVISE_SPEC_BEFORE_PLAN`。只有本文件通过一次不读取实验结果的 spec-only
diff review，且 reviewer 输出 `APPROVE_SPEC_FOR_PLAN`，才允许进入 implementation plan。

实现状态：`designed`，不是 `implemented`、`tested`、`experiment_running` 或
`empirically_supported`

替代关系：本文件 supersede `02199f8` 的 r1 规格与 commit `b74101d` 中的 bounded-rescue
规格；两份旧文件只保留为历史记录，均不可执行。

## 1. 裁决、证据边界与目标

ChronoTransport seed-3407 formal Stage-B/P3 已正式失败：旧 risk head 把 144 个
`chunk × layer-group` 非负 cell scalar 相加，却用窗口级 detector regret 监督；旧校准与
Spearman 又把同一窗口的 candidate rows 当独立样本。该 checkpoint、cell-sum head 和旧
P3 裁决协议已经死亡，Stage C/P5 从未解锁。

最初 Pro review 的统计修订总体成立，但当时 reviewer 没看到本地源码；后续本地审计、
两次空白上下文复核与 GitHub-visible Pro 终审进一步确认：

- Stage B 使用独立非零 LR AdamW，不存在 base optimizer `lr=0` 冻结；
- head 在 paired replay 中为 eval，RNG 会恢复，没有已证实的 loss-normalizer 顺序污染；
- CT 有效 config 已关闭 frame selector、physical-grid 行为和 packed route；
- 当前 runtime 虽计算 dense AdaTAD adapter，却只写回 RECOMPUTE rows，实际改变了官方
  AdaTAD block 语义；
- `max_cache_age=8` 与 `hold_only/transport_only` 的 47-clip 连续复用矛盾；
- runtime repair/fallback 后仍沿用 requested schedule cost，会产生错误的成本成功记录。

因此，本协议不是对旧 P3 的结果后调参，而是一次预注册、有限、不可再次修改的协议修复。
任一科学 Gate 失败后，ChronoTransport 永久降级为 frozen engineering baseline；不得更换
head、loss、pooling、candidate library、seed、训练长度、预算或 gate 后复活。

本协议只回答四个问题：

- H1：冻结的 time×depth schedule library 在同一实测预算下是否存在逐窗口 oracle
  headroom？Gate 1 不证明 deploy-visible input dependence。
- H2：在完全相同的 RECOMPUTE mask 下，TRANSPORT 是否稳定优于 HOLD？
- H3：部署可见信号能否在窗口内部排序候选，并以 simultaneous marginal calibration
  支持真实 scheduler？
- H4：通过 H1–H3 后，完整系统能否在保护高 tIoU 与短动作时获得真实 full-stack 加速？

执行顺序固定为 Gate 1 → Gate 2 → Gate 3 → Gate 4；前一道 FAIL 时后续全部锁定。

## 2. 不变任务与官方骨架

以下合同不可修改：

- 方法是离线完整窗口 TAD，不是 Online、causal 或 streaming TAD。
- 输入为 768 帧；VideoMAE-S 接收 48 个独立 16-frame clips；tubelet size 为 2，内部
  temporal tubelet grid 为 384，外部 detector grid 仍为 768。
- 12 个 VideoMAE blocks 固定分为 `[0:4] / [4:8] / [8:12]` 三个连续层组。动作单位是
  `clip × layer-group`，即 48×3 cells；不是逐帧、逐 tubelet 或逐 spatial token routing。
- patch embedding、AdaTAD temporal adapter、post-backbone interpolation、projection、
  ActionFormer head、GT assignment、NMS 与外部时间坐标保持官方 AdaTAD-derived 语义。
- validation/test GT、teacher、dense reference、raw prediction cache、counterfactual ledger
  和 evaluation oracle 不得进入推理决策。
- cache 每个 768-frame window 重置；禁止跨窗口复用。
- seeds 只允许 `3407 / 3408 / 3409`。
- risk quantile 固定 `tau=0.9`；scheduler tolerance 固定 `epsilon=1.0`。

固定上游 provenance：

- OpenTAD：`1aa8ca4ac5e846b1e8ff69298dd6607121a01589`；
- AdaTAD：`25e06c720e450298ca5267fda6927f3591dcdfef`；
- VideoMAE：`14ef8d856287c94ef1f985fe30f958eb4ec2c55d`。

## 3. 精确 block、cache 与 adapter 语义

### 3.1 动作只控制 heavy 子路径

对 batch `b`、clip `c`、block `l`，令 `z[b,c,l]` 为该 block 的输入，`H_l` 为原始
attention+MLP heavy subpath。每个 block 独立维护 pre-adapter rolling cache：

- `anchor`：最近一次 RECOMPUTE 的 pre-adapter heavy 输出；
- `latest`：最近可复用的 pre-adapter heavy 输出；
- `actual_age`：自最近 RECOMPUTE 后经过的非重算 clip 数；
- `valid`、source clip、group id 与 action history。

动作输出 `u[b,c,l]` 定义为：

- RECOMPUTE：`u = H_l(z)`；随后 `anchor = latest = u`，`actual_age = 0`；
- HOLD：`u = latest`；不更新 `anchor/latest`，`actual_age += 1`；
- TRANSPORT：`u = T_g(latest, z, min(actual_age+1, 8))`；随后
  `latest = u`，`anchor` 不变，`actual_age += 1`。

TRANSPORT 始终从 latest 链式递推，不得偷偷改回 anchor transport。

### 3.2 全 rows AdaTAD adapter

对一个 block 的全部 48 个 clips 先完成动作执行，按原始 clip/tubelet lexicographic order
恢复完整 dense heavy-surrogate tensor `U_l`。若该 block 启用 AdaTAD adapter，则必须执行：

`Z_{l+1} = Adapter_l(U_l)`，并把 adapter 输出应用到所有 rows。

不得再使用“只把 adapted 输出写回 RECOMPUTE rows”的 masked scatter。adapter 输出作为
下一 block 的输入，但不得回灌到同一 block 的 rolling cache，因为 AdaTAD TIA 在完整窗口
上做非因果 dense temporal convolution；同一 block 的 cache 必须停留在 pre-adapter heavy
边界。

因此：HOLD 的 bitwise identity 只指 pre-adapter heavy cache，不能声称完整 block 输出
bitwise 不变。forced-dense 路径必须继续直接调用原始 block forward，并与官方 dense path
做数值 parity。

### 3.3 截断梯度与当前 row 的 live-tensor 合同

`cache_detach=True` 只截断跨 clip 的 recurrent gradient，不得截断当前
RECOMPUTE/TRANSPORT row 到本 block 输出的梯度。

对每个当前 row，先产生 `u_live` 并把该 live tensor 写入完整 heavy-surrogate tensor
`U_l`：

- RECOMPUTE：`u_live = H_l(z)`；
- HOLD：`u_live = latest_detached`；
- TRANSPORT：`u_live = T_g(latest_detached, z, min(actual_age + 1, 8))`。

随后才更新 recurrent cache alias：

- RECOMPUTE：`anchor_detached = u_live.detach()`，
  `latest_detached = u_live.detach()`；
- TRANSPORT：`latest_detached = u_live.detach()`；
- HOLD：cache 不变。

因此，当前 RECOMPUTE/TRANSPORT row 仍可通过
`U_l -> Adapter_l -> downstream loss` 获得本 row 梯度；后续 clip 不得通过
`latest_detached/anchor_detached` 向此前 clip 回传 recurrent gradient。本协议不允许
full BPTT。

### 3.4 两种 age 不再混用

- `hard_cache_validity_age=47`：一个 48-clip window 在首 clip RECOMPUTE 后最多 age=47；
  因此冻结库中的 `hold_only/transport_only` 可按名称原样执行。
- `transport_age_embedding_cap=8`：保持现有 transport age embedding 结构；实际 age>8
  时仅 embedding index clamp 到 8。
- risk head 的 deterministic age feature 使用真实 `actual_age`，归一化为
  `actual_age / (1 + actual_age)`，不 clamp 到 8。

旧的单一 `max_cache_age` 字段禁止继续承担上述两个含义。formal run 中任何 schedule
repair 都使该样本与 cost lookup 失效；不得把 repaired schedule 冒充原 candidate。

### 3.5 forced-dense parity acceptance

forced-dense 必须直接调用原始 AdaTAD/VideoMAE block forward，不得通过 RECOMPUTE
gather/scatter 重构 dense 路径。正式实现必须同时满足：

1. CPU deterministic tiny-block：输出与输入梯度 bitwise equal；
2. remote CUDA deterministic FP32：backbone 输出、detector loss、输入梯度和 adapter
   parameter gradients bitwise equal；
3. remote CUDA AMP FP16：上述量满足 `atol=1e-6, rtol=1e-5`；
4. 分别覆盖 adapter-enabled block、adapter-disabled block、activation checkpoint on/off；
5. forced-dense 不得产生 schedule repair、fallback 或 requested/executed action mismatch；
6. legacy dense checkpoint strict load 后必须继续满足上述 parity。

任一失败属于 `INVALID_IMPLEMENTATION`，不是 science FAIL。

## 4. 唯一窗口级 quantile head

旧 144-cell scalar sum 必须被唯一的 schedule-conditioned window head 替换，不允许并行
尝试第二种 pooling、normalization 或 head。

每个 cell 的输入维数固定为：deploy-visible signal 6、action embedding 8、group embedding
8、normalized true age 1，因此 `D=23`。action embedding 与 group embedding 是两个独立的
8-dimensional embedding tables，不得共享参数或在实现后改变维数。架构固定：

1. cell encoder：`Linear(D,64) → GELU → Linear(64,64) → GELU`；
2. 对全部 48×3 cell hidden 分别做 mean pooling 与 max pooling；
3. 拼接得到 128-dimensional window representation；
4. scalar head：`LayerNorm(128) → Linear(128,64) → GELU → Linear(64,1) → Softplus`；
5. 输出一个非负窗口级预测 `q_hat_tau(x,s)`。

训练 target 固定：

`r(x,s) = max(L_detector(x,s) - L_detector(x,dense), 0)`。

使用 `tau=0.9` pinball loss。dense 不进入 head fit、calibration 或 ranking；dense risk 与
upper risk 在 scheduler 外部精确设为 0。

部署信号白名单固定为 energy、delta L2、pooled L2、cosine change、normalized chunk
position、finite flag，以及候选 action、group identity 和 deterministic actual age。

## 5. 固定数据 split、checkpoint 与随机性

### 5.1 共享 video split 与 one-window-per-video manifest

Gate 1--3 的 population 固定为 THUMOS14 train split 中 200 个 unique videos。所有 video ID
先规范化为 UTF-8 NFC 字节。对 video `v` 定义：

`split_digest(v) = SHA256(b"CT-P3R-3S-r2-split-v1\0" +
                          b"3407\0" + video_id_utf8)`。

按原始 32-byte digest lexicographic ascending 排序；digest 相同时按 `video_id_utf8`
ascending 决定顺序。一次冻结：fit 140、calibration 30、evaluation 30。run seed 不参与
split 构造，seeds 3408/3409 必须加载同一 manifest。

然后为每个 video 恰好生成一个 label-free 768-point temporal window。不得调用 official
`random_trunc` 决定 window start，因为该 transform 使用 GT intersection 选择含动作 crop。

对 video `v`：

1. 按固定 dataset config 的 `snippet_stride`、`scale_factor`、rounding 与 clipping 规则构造
   source sampled-index vector `I_v`；空 `I_v` 为 `INVALID_DATA`；
2. 令 `n_v=len(I_v)`、`W=768`，将 media SHA-256 表示为 lowercase 64-byte ASCII hex，
   `n_v` 表示为无前导零 decimal ASCII；
3. 计算
   `d_v=SHA256(b"CT-P3R-3S-r2-window-v1\0" + video_id_utf8 + b"\0" +
               media_sha256_ascii + b"\0" + n_v_ascii)`；
4. 若 `n_v<=W`，`start_v=0`；否则
   `start_v=uint64_big_endian(d_v[0:8]) mod (n_v-W+1)`；
5. 取 `I_v[start_v:start_v+W]`；不足 W 时按 official `numpy.pad(mode="edge")` 语义重复
   最后一个有效 sampled index；原始 positions 的 valid mask 为 true，padding positions 为
   false。

annotation bytes、GT count、class、segment duration 与 detector output 均不得进入 split/window
digest。annotation SHA-256 只作为 data identity 记录。

one-window manifest 对每个 video 必须记录：video ID；media path/registry ID 与 media SHA-256；
source total frames、fps、snippet stride、scale factor；source sampled-index vector length；window
start；全部 768 sampled frame indices；padding positions 与 valid mask；data/config/annotation
SHA-256；canonical per-window payload SHA-256。

必须断言：恰有 200 个 unique video IDs 与 200 个 unique window IDs；二者一一对应；三个 split
分别恰有 140/30/30 windows；三个训练 seeds、全部 schedules 与 paired branches 使用相同
temporal window。manifest 记录 exact IDs、cardinality、各 split hash 与整体 manifest hash。

Stage B 固定 `batch_size=1`、`world_size=1`、`shuffle=false`。fit 可由 run seed 控制空间/颜色
augmentation，但同一 paired dense/counterfactual forward 必须共享 materialized pixel tensor。
calibration/evaluation 使用 deterministic spatial transform。

Gate 1--3 的 conformal、Spearman、coverage 与 bootstrap outer unit 均为 unique manifested
window。Gate 4 是不同的 official full-video/sliding-window population；Gate 3 的 conformal
guarantee、coverage 或 safety statement 不得自动转移到 Gate 4。

### 5.2 checkpoint

三个 seeds 使用同一已登记 dense AdaTAD/VideoMAE-S checkpoint 起点。manifest 必须记录
本地路径、远端路径、文件 SHA-256、模型 config hash 和 upstream provenance。legacy dense
checkpoint 只可初始化 Stage B，不可直接解锁 learned scheduler。

### 5.3 paired replay

同一 `sample_id × seed × schedule` 的 dense/counterfactual replay 必须使用相同 batch、
augmentation 和 Python/NumPy/Torch/CUDA RNG state。detector/head 保持 eval，仅 transport/
risk 为 train。必须加入 candidate-order permutation regression；若 regret 随顺序改变，
formal run 直接 invalid，不得先假设是 `loss_normalizer` 再静默修补。

## 6. 冻结 candidate library

所有 schedule 的 clip 0、三个层组都为 RECOMPUTE。canonical non-dense 顺序固定为：

1. `periodic2_transport`；
2. `periodic2_hold`；
3. `periodic4_transport`；
4. `periodic4_hold`；
5. `periodic8_transport`；
6. `periodic8_hold`；
7. `transport_only`；
8. `hold_only`；
9. `layer_only_early_recompute`；
10. `layer_only_early_recompute_hold`；
11. `layer_only_late_recompute`；
12. `layer_only_late_recompute_hold`；
13. `joint_progressive_transport`；
14. `joint_progressive_hold`；
15. `joint_reverse_transport`；
16. `joint_reverse_hold`。

定义：

- periodic P2/P4/P8：全部层组每 2/4/8 clips RECOMPUTE，其余分别 TRANSPORT/HOLD；
- only：除 clip 0 外全部分别 TRANSPORT/HOLD；
- early：早层组每 clip RECOMPUTE，中/晚层组除 clip 0 外分别 TRANSPORT/HOLD；
- late：晚层组每 clip RECOMPUTE，早/中层组除 clip 0 外分别 TRANSPORT/HOLD；
- progressive：早/中/晚 RECOMPUTE period 分别为 8/4/2；其余分别 TRANSPORT/HOLD；
- reverse：早/中/晚 period 分别为 2/4/8；其余分别 TRANSPORT/HOLD。

`dense` 是第 17 个 safety candidate，但不属于 non-dense fit/calibration vectors。

canonical library JSON 必须包含名字、顺序、48×3 action matrix、hard validity、action
hash 与整体 library SHA-256。训练、profile、checkpoint 和 evaluation 都必须绑定该 hash。

非学习 controls 固定为：

- `motion_topk_p{2,4,8}`：只使用 deploy-visible cosine-change signal。对每个 window、每个
  layer group，从对应 frozen periodic action matrix 读取该 group 的 exact RECOMPUTE count
  `K_p`。clip 0 必须 RECOMPUTE；在 clips 1--47 中按 cosine change descending 选择
  `K_p-1` 个 positions RECOMPUTE，其余 HOLD。finite ties 按较小 clip index 优先；finite
  constant-motion window 合法并使用同一 tie rule。该 control 不拟合 calibration threshold；
- `random_p{2,4,8}`：所有字段使用无前导零 decimal ASCII，window ID 使用 UTF-8 NFC。
  对 clip `c` 定义
  `SHA256(b"CT-P3R-3S-r2-random-v1\0" + window_id_utf8 + b"\0" + seed_ascii +
          b"\0" + group_ascii + b"\0" + period_ascii + b"\0" + clip_ascii)`。
  clip 0 强制 RECOMPUTE；clips 1--47 按 raw 32-byte digest ascending 排序，选择 `K_p-1`
  个 positions RECOMPUTE，其余 HOLD；
- `uncalibrated_risk`：同一 trained head，offset 强制 0；
- `oracle`：只存在于 evaluation-only adjudicator，不进入 checkpoint 或 scheduler。

motion/random requested action matrix 必须与 periodic comparator 在每个 `window × group` 上
具有完全相同的 RECOMPUTE count。成本使用各自 requested/executed action hash 对应的实测
分布，不得借用 periodic cost。任何 non-finite motion signal 必须触发 dense safety fallback，
并使该 formal comparator sample/run 标记 `INVALID_IMPLEMENTATION`；不得以 repaired schedule
进入 Gate 1。`motion_topk` 与 `random` 均为 Gate 1 hard comparators。

## 7. Stage B 的 140 次成功更新

Stage B 固定为 FP32，不使用 autocast/GradScaler。optimizer 与 loss 固定：

- transport+risk：AdamW，LR `1e-4`，weight decay `0`；
- `lambda_transport=0.1`、`lambda_risk=0.1`；
- EMA `0.999`；gradient clip `1.0`；
- detector、heavy VideoMAE、AdaTAD adapters、projection/head 全部冻结；
- fit split 恰好一次 canonical pass，共 140 次 successful optimizer updates。

Stage-B objective 固定为：

`L_B = L_detector_counterfactual + 0.1 × MSE(F_counterfactual,F_dense.detach())
       + 0.1 × Pinball_tau(q_hat,r.detach())`。

MSE 对 runtime 输出的全部元素取 mean；dense reference 无梯度。不得改变三项定义、增加
target normalization 或 schedule-dependent loss weight。

### 7.1 Candidate × window exposure

每个 update 只训练一个 non-dense schedule。令 `j` 为 canonical fit-window index，
`j in [0,139]`：

- `b=floor(j/16)`；
- `p=j mod 16`；
- `candidate=(p+5*b+seed_offset) mod 16`。

seed offsets 固定：3407 为 0、3408 为 4、3409 为 8。该公式不可替换为 hash permutation、
Latin-square 搜索或结果后重新排布。augmentation RNG 由 run seed 控制。

formal validator 必须断言：

1. 每个完整 16-window block 内 candidates 0--15 各出现一次；
2. 每 seed 为 12 candidates 暴露 9 次、4 candidates 暴露 8 次；
3. seeds 3407/3408/3409 的 8-exposure candidates 分别为 4--7、8--11、12--15；
4. 三 seed 汇总 candidate 0--3 各 27 次，4--15 各 26 次；
5. 前 128 windows 中，每 candidate、每 seed 在四种 `p mod 4` 上各出现 2 次，三 seed
   汇总各 6 次；
6. 每个 fit window 跨三 seed 获得三个不同 candidates；
7. tail 顺序精确为：3407 `[8,9,10,11,12,13,14,15,0,1,2,3]`；3408
   `[12,13,14,15,0,1,2,3,4,5,6,7]`；3409 `[0,1,2,3,4,5,6,7,8,9,10,11]`。

artifact 必须登记 canonical fit-window order hash、每个
`seed × update × window_id × candidate`、三个完整 140-row exposure matrices、per-seed hashes
与 combined matrix SHA-256。任一 mismatch 为 `INVALID_IMPLEMENTATION`。

固定 140-update protocol 的负结果最多否定本 head/library/exposure/budget 组合，不得写成
“所有 transport 或 dynamic refresh ideas 均失败”。

formal completion 必须满足：attempted optimizer updates=140、successful optimizer
updates=140、non-finite/skipped updates=0、AMP skips=0、EMA updates=140、LR scheduler
updates=140。基础设施中断可从原子 checkpoint 恢复，但不得额外增加 successful update；
重复读取但未到 optimizer attempt 边界的 batch 只记 infrastructure resume event。

## 8. 成本测量、B* 与 requested/executed 账本

### 8.1 full-stack total samples

成本 profile 在物理 GPU1、batch size 1、AMP FP16 下执行。每个 schedule warm-up 50 次，
随后至少记录 200 个完整窗口。每次 sample 的 `total_ms` 从 data/decode 开始前计时，到
postprocess 完成后结束；计时边界前后 CUDA synchronize。

full-stack p50/p95 必须直接从 `total_ms` samples 计算，禁止相加 stage p50/p95。每次
invocation 同时记录以下 diagnostic durations：decode、preprocess、H2D、innovation、
scheduler、heavy recompute、transport、cache movement、dense AdaTAD adapter、neck/head、
postprocess，以及 peak memory。stage durations 不替代 total distribution。

### 8.2 cost lookup provenance

lookup key 至少包含：GPU model/UUID、driver、CUDA、PyTorch、precision、batch size、source
commit、spec file SHA-256、config/checkpoint hash、library hash、candidate name、requested
action hash、executed action hash、selected rows per group。环境 fingerprint 或任一 hash
不匹配时 learned scheduling fail closed。

### 8.3 requested 与 executed 分离

pre-selection 可用 requested candidate 的 frozen measured p50。执行后必须重新登记：

- requested schedule/name/action hash/cost；
- executed schedule/name/action hash；
- repair count、NaN fallback、whole-window dense fallback；
- executed lookup cost（若 exact key 存在）与本次 actual `total_ms`。

只要 runtime repair 或 fallback 改变 action hash，requested cost 就不得用于 gate 或成本成功
claim。formal Gate 1–4 中任何 action-hash 改变都直接使样本/run invalid；executed lookup 仅
用于错误诊断。部署时允许按 exact executed key 记账；无 key 时必须 dense fallback，并记录
`safety_override_budget_violation=true`（当 dense p50>B*）。

### 8.4 固定预算

`B* = measured full-stack p50(periodic4_transport)`。

每个 candidate 仅以自身完整实测 p50 判断 `cost<=B*`；禁止线性 action-count 估计、借用
P4 cost 或 stage-percentile sum。B* 相对 dense 的 full-stack p50 saving 必须至少 20%
才能通过 Gate 1。

## 9. Gate 1：冻结库的 equal-cost oracle headroom

Gate 1 在任何新 Stage-B seed 训练前运行，只支持 `oracle_headroom=true/false`，不支持
deploy-visible input dependence。

首先以每个 candidate 自身 registered full-stack p50 判断 `p50(candidate)<=B*`，冻结
cost-feasible HOLD schedule set。不得借用 P4 cost、action-count proxy 或 stage-percentile
sum。

Comparator 分为：

1. `calibration-frozen global static`：在 calibration windows 上选择 mean regret 最低且
   cost-feasible 的单一 HOLD schedule；identity 在 evaluation 前冻结；
2. `motion_topk` 与 `random`：使用第 6 节 exact-count controls；
3. `time-only oracle`：每 evaluation window 在 P2/P4/P8 HOLD 中取 minimum；
4. `layer-only oracle`：每 evaluation window 在 early/late HOLD 中取 minimum；
5. `evaluation-best global static`：在 evaluation windows 上选择 mean regret 最低的单一
   cost-feasible HOLD schedule；它是 evaluation-only adjudication comparator；
6. `joint oracle`：每 evaluation window 在全部 cost-feasible HOLD schedules 中取 minimum。

`evaluation-best global static` 可以参与 Gate 1 pass/fail，但不得进入 deployment、checkpoint、
threshold、candidate library、Stage B/C training、calibration 或后续配置。删除全部
oracle-assignment shuffle hard conditions。schedule diversity、selection entropy 与 selected
schedule count 只作 diagnostics。

`strongest_comparator` 是 1--5 中 evaluation mean regret 最低者。若其 mean regret
`<=1e-12`，relative-reduction criterion 未定义，Gate 1 直接 FAIL，不得增加 epsilon。

Gate 1 同时要求：

1. joint oracle 相对 strongest comparator 的 full-sample mean detector-regret relative
   reduction `>=10%`；
2. 5000 次 paired unique-window bootstrap 中，
   `regret_strongest_comparator-regret_joint_oracle` 的 percentile 95% CI lower `>0`；
3. `B*` 相对 dense measured full-stack p50 saving `>=20%`。

每个 bootstrap replicate 必须 resample complete evaluation windows；在 replicate 内重新选择
evaluation-best global static 与 strongest comparator；重新计算 per-window joint/time/layer
oracles；保持每个 window 的完整 candidate vector。

candidate-set size、feasible names 与各 oracle set size 必须报告。任一失败即 H1=`no`，路线
永久冻结，不训练新 transport/risk seeds。Gate 1 PASS 只允许
`oracle_headroom=true`；input dependence 只能由 Gate 3 裁决。

## 10. Gate 2：matched TRANSPORT vs HOLD

Gate 1 PASS 后训练三个 Stage-B seeds。Gate 2 只比较相同 RECOMPUTE mask：

- P2 TRANSPORT vs P2 HOLD；
- P4 TRANSPORT vs P4 HOLD；
- P8 TRANSPORT vs P8 HOLD。

对每个 `seed × sample_id × period`：

- detector improvement=`regret_hold - regret_transport`；
- feature improvement=`mse_hold - mse_transport`。

Gate 2 同时要求：

1. 三 periods、三 seeds pooled detector regret relative reduction ≥5%；
2. detector absolute improvement paired hierarchical-bootstrap 95% CI lower>0；
3. feature-MSE absolute improvement paired hierarchical-bootstrap 95% CI lower>0；
4. 每 seed 的 detector 与 feature mean improvement 均≥0。

`hold_only/transport_only` 必须按 exact actions 报告，但不能替代主 gate。任一失败即
H2=`no`；不得改变 transport loss、结构、age cap 或训练长度再试。

## 11. Gate 3：窗口排序与 simultaneous marginal calibration

### 11.1 校准

每 seed 在 fit split 训练同一窗口 head。对 calibration window `i`，先在全部 16 个
non-dense candidates 上计算：

`score_i = max_s max(r_i,s - q_hat_i,s, 0)`。

唯一 conformal offset 是 30 个 window scores 的 finite-sample tau=0.9 quantile；rank 固定
为 `ceil((30+1)×0.9)=28`。`upper_i,s=q_hat_i,s+q_conf`。

该构造只提供窗口级 simultaneous marginal coverage，不是“被 scheduler 选择为 non-dense”
条件下的理论 coverage 保证。actual-selected coverage 是必须独立报告和裁决的 evaluation
统计量。

### 11.2 Fit-only schedule-conditioned constant baseline

每个 Stage-B seed 完成 140 successful updates 后，在打开 calibration/evaluation 前，对全部
140 fit manifested windows no-grad replay 全部 16 non-dense schedules。

对每个 schedule `s`，使用其 140 个 fit regret targets 的 finite-sample `tau=.9` order
statistic：`rank=ceil((140+1)*0.9)=127`，得到 schedule-conditioned constant prediction
`q_const_s`。baseline payload、完整 fit replay key set 与 SHA-256 必须在 calibration 前冻结；
不得只使用该 schedule 的 8--9 个 optimizer-exposure targets。

### 11.3 Ranking 与 bootstrap unit

对每个 `seed × evaluation window`，在完整 16-candidate vector 内计算 Spearman rho。
predicted vector 或 regret vector 少于 3 个 distinct ranks 时，该 seed-window fail closed，
不得填 0 或静默删除。

每 seed score 是 30 个 window rhos 的 arithmetic mean；pooled score 对全部
`seed × window` 等权。所有 Gate 2/3 bootstrap：outer resample unique manifested windows；
inner resample 三 seeds；candidate/period vector 随 window 整体移动；candidate rows 不得成为
bootstrap samples；replicates=5000，seed=`20260711`。

### 11.4 Scheduler

scheduler 仅在 non-dense candidates 中筛选：

`requested_p50<=B* AND upper<=epsilon AND finite AND metadata/hash valid`。

在可行项中选 requested measured p50 最低者；cost tie 按 canonical library order。无可行
项时选择 dense safety fallback。dense risk=0，但不因此成为 budget-feasible success。

### 11.5 Selected support 与 coverage

evaluation 同时要求：

1. 每 seed 至少 `6/30` windows 选择 non-dense；
2. pooled selected `seed × window` count 至少 18；
3. 至少 10 个 distinct evaluation windows 被任一 seed 选择；
4. pooled selected non-dense point coverage `>=0.85`；
5. 每 seed mean rho `>=0`，三个 seed means 的 median `>=0.2`；
6. pooled rho hierarchical-bootstrap 95% CI lower `>0`；
7. evaluation pinball loss 相对第 11.2 节 constant baseline 至少降低 10%。

coverage denominator 只包含实际 selected non-dense seed-window rows；dense fallback 不得进入
coverage numerator 或 denominator。对 selected row 定义
`coverage_margin=upper_selected-regret_selected`，covered 当且仅当 margin `>=0`。

同时报告：per-seed selected coverage、pooled selected coverage、selected unique-window count、
window-clustered one-sided 95% coverage LCB。对每个 seed `s`、window `w` 定义
`all_candidate_covered(w,s)=1[min_c(upper(w,s,c)-regret(w,s,c))>=0]`；all-window
simultaneous coverage 是该 indicator 在全部 90 个 seed-window 上的比例，并同时报告 per-seed
比例。对每个至少有一个 non-dense selection 的 window `w`，令 `S_w` 为选择 non-dense 的
seeds，定义
`window_all_selected_covered(w)=1[min_{s in S_w} coverage_margin(w,s)>=0]`。没有 non-dense
selection 的 windows 不进入该比例的分母，但必须单独计数并进入 support gate。

coverage CI lower 不作为 `>=0.85` hard gate；coverage `>0.95` 只标记 `OVERCOVERED`，不
直接失败。selected coverage 是 empirical diagnostic/gate，不得描述为 split-conformal
selected-conditional theoretical guarantee。

同时并列报告 calibrated、offset=0 uncalibrated 与 dense fallback 的 selection rate、regret、
upper sharpness、pinball loss 与成本，以识别“总 dense + 极保守 upper”伪成功。

所有 Gate 3 hard conditions 构成 intersection-union gate，不作额外 multiplicity correction。
baseline pinball mean `<=1e-12` 时，10% relative improvement 未定义，Gate 3 FAIL。任一
hard 条件失败即 H3=`no`；不得换 pooling、normalization、quantile、epsilon、loss、head
width 或额外训练。

## 12. 统计、诊断 proxy 与泄漏边界

### 12.1 bootstrap

- Gate 1：5000 次 paired unique-window bootstrap；每 replicate 重新选择 evaluation-best
  static 与 strongest comparator。
- Gate 2--3：5000 次 paired hierarchical bootstrap，outer resample unique manifested
  windows，inner resample seeds；完整 candidate/period vectors 随 window 移动。
- Gate 4 使用第 13 节独立定义的 official-video clustered latency/mAP bootstrap，不继承
  Gate 1--3 的 frozen-window population。
- CI 使用 percentile 95%，bootstrap seed=`20260711`。
- relative criterion 只有 denominator `>1e-12` 时定义；不得以 epsilon 替换零 denominator。
  所有 CI gate 使用 absolute difference。
- 缺记录、非有限值、重复 sample key、vector 不完整直接 FAIL，不得静默过滤。

### 12.2 endpoint/high-IoU/short-action diagnostics

这些 proxy 只作辅助诊断，不替代 detector regret 或官方 mAP。

在官方 postprocess 后、统一以秒为坐标，对每个 GT segment `g`：

- 在同类 predictions 中取 tIoU 最大者 `p*`，tie 取 score 更高者；无同类 prediction 时
  视为 unmatched；
- high-IoU hit=`1[tIoU(p*,g)>=0.7]`，unmatched 为 0；
- endpoint error=`min(1,(|start(p*)-start(g)|+|end(p*)-end(g)|)/(2×max(duration(g),1e-6)))`，
  unmatched 为 1；
- candidate high-IoU regret=`dense hit - candidate hit`；endpoint regret=`candidate error -
  dense error`。

short-action threshold 是 fit annotations 中 GT duration（秒）的 lower quartile，使用固定
线性插值 quantile，一次冻结到 manifest；calibration/evaluation durations 不参与阈值。
short-action proxy 只在 `duration<=fit_Q1` 的 GT 上聚合。prediction artifacts 标记
`evaluation_only=true`，与 scheduler 输入物理隔离。

## 13. Stage C、matched dense control 与 Gate 4

Gate 1--3 全 PASS 后才生成 hash-bound Stage-C unlock artifact。

### 13.1 Stage-C fixed execution semantics

Stage C 固定：`world_size=1`；plain single-process module；禁止 DDP、FSDP、`no_sync`；
global batch size 2；no gradient accumulation；`drop_last=false`；140 fit windows；70 successful
optimizer updates/epoch；60 epochs；4200 successful updates；AMP FP16；EMA decay 0.999；global
clip grad norm 1.0。

显式 optimizer groups 为：AdaTAD adapters LR `2e-4`、weight decay `0.05`；CT transport 与
risk LR `1e-4`、weight decay `0`；heavy VideoMAE、projection/head 与其他参数 frozen/excluded。

LR scheduler 固定为 OpenTAD `LinearWarmupCosineAnnealingLR`：warmup steps `5*70=350`；max
scheduler steps `100*70=7000`；training 在 successful update 4200 后停止；scheduler 与 EMA
只在 successful optimizer update 后各推进一次。

matched-dense control 使用相同 dense checkpoint 起点、seed、fit-window order、materialized
augmentation、60-epoch workflow、common adapter LR/WD、detector task loss 与 successful common
updates；forced dense，仅训练 A，不训练 T/R。matched dense 保存相同 shadow candidate ledger。

### 13.2 Stage-C per-window candidate exposure

令 successful update `u in [0,4199]`，batch canonical position `r in {0,1}`，window-exposure
ordinal `e=2*u+r`。令 `b=floor(e/16)`、`p=e mod 16`、
`candidate=(p+5*b+seed_offset) mod 16`，offsets 仍为 0/4/8。

每 seed 恰有 8400 window exposures，每 candidate 恰好 525 次。CT arm 可在同一 batch 对两个
examples 执行不同 action matrices。两个 arms 必须共享 ordered materialized
batch/augmentation hashes；retry 时同一 successful index 的 candidate 不变。

### 13.3 Object-identity parameter ownership

以 Python object identity 构造、排序并冻结三个 parameter tuples：A 为全部 AdaTAD adapter
parameters；T 为全部 ChronoTransport transport parameters；R 为 canonical risk predictor
parameters。

必须断言：A/T/R 两两不交；union 恰好等于全部 `requires_grad=True` parameters；optimizer
中每个 parameter object 恰好出现一次；heavy VideoMAE、projection/head 与其他参数均 frozen
且不在 optimizer；`risk_predictor` 与 `scheduler.predictor` alias 只以 canonical object 进入
R 一次。禁止 generic name-substring optimizer grouping。

### 13.4 Loss-specific AMP gradient algorithm

同一 counterfactual forward 产生 LD detector task loss、LF feature consistency loss 与 LR
pinball risk loss。LR 的 deploy-visible signals 与 regret target 对 A/T detach；scheduler argmin
不反传。

每个 attempt 必须：

1. `optimizer.zero_grad(set_to_none=True)`；
2. 读取当前 GradScaler scale `S`；
3. 计算：
   - `gD=autograd.grad(scaler.scale(LD), A+T, retain_graph=True, allow_unused=True)`；
   - `gF=autograd.grad(scaler.scale(0.1*LF), T, retain_graph=True, allow_unused=True)`；
   - `gR=autograd.grad(scaler.scale(0.1*LR), R, retain_graph=False, allow_unused=True)`；
4. 在 scaler update 前断言三次 `scale()` 使用同一 S；
5. 写入 scaled gradients：`A.grad=gD[A]`；`T.grad=gD[T]+gF[T]`，None 按零；
   `R.grad=gR[R]`；
6. 恰好一次 `scaler.unscale_(optimizer)`；
7. finite 与 expected-unused audit；
8. finite 时对 A+T+R 执行 global clip norm 1.0；
9. 恰好一次 `scaler.step(optimizer)` 与一次 `scaler.update()`。

Expected-unused：A aggregate detector gradient 必须 finite/nonzero；R aggregate pinball gradient
必须 finite/nonzero；executed batch 无 TRANSPORT cell 时 T 可全 None/0；存在 TRANSPORT 时该
successful exposure 的 T aggregate gradient 必须 finite，全部 TRANSPORT exposures 汇总 T norm
必须大于 0。LF forward 使用当前 trainable adapter 计算 input Jacobian，但 A 不在该次
`autograd.grad` inputs 中，因此 LF 不写 A.grad。

### 13.5 AMP overflow retry

每个 batch 首次 forward 前 materialize/hash batch 与 augmentation，并 snapshot
Python/NumPy/Torch/CUDA RNG、全部 forward-mutated model buffers/Python state、optimizer、EMA、
scheduler、CT diagnostics 与 profiler state。

overflow 时：调用 `scaler.step(optimizer)` 并验证 optimizer 未改变；调用 `scaler.update()`
保留 scaler backoff；不推进 sampler、batch cursor、successful index、candidate exposure、LR
scheduler 或 EMA；清空 gradients；恢复 pre-forward RNG；bitwise 恢复除 GradScaler 外全部
snapshot state；retry 同一 materialized batch。

必须恢复 BN running stats/counters、`AnchorFreeHead.loss_normalizer`、checkpoint/dropout RNG、
CT `latest_*`、cache/action history、profiler buffers、Python lists/counters、optimizer、scheduler
与 EMA。只有模型外 append-only retry audit log 可保留。初始 attempt 后最多 3 次 retry；第 4
个 overflow attempt 仍失败时标记 `INVALID_IMPLEMENTATION`，不是 science FAIL。

两个 arms 可有不同 overflow histories，不采用 lockstep mutual skip。matched exposure 必须满足：
successful batch hashes 完全相同且顺序相同；各 4200 common-adapter updates；common-A LR trace
与 EMA update count 相同；各自 attempted/retry/scaler traces 完整报告。

Stage C 后必须在 calibration split 重新计算唯一 q_conf，并重新通过 Gate 3，才可执行 learned
scheduler detector evaluation。

### 13.6 Gate-4 population and matched timing

Gate 4 使用 official full-video/sliding-window evaluation population；全部 overlapping windows
归属其 official video ID。每 arm/seed timed invocation set 必须包含全部 official invocation
IDs、至少 200 invocations、总数为 6 的整数倍。附加 repetitions 只按预注册 invocation-hash
顺序补齐；exact invocation list、repetition IDs 与 order SHA-256 在 timing 前冻结。

primary arms：D matched dense、C calibrated ChronoTransport、S calibration-frozen global
static。使用六序列循环：D-C-S、C-S-D、S-D-C、S-C-D、C-D-S、D-S-C。

每 arm 独立执行 decode、preprocess、H2D、model 与 postprocess，不得共享 decoded tensor 或
模型中间 cache。每 invocation 在 timing 边界前后 CUDA synchronize，保存完整 `total_ms`；
stage durations 只作 diagnostics。

### 13.7 Gate-4 bootstrap

latency bootstrap：outer resample official video IDs；在 sampled video 内 resample complete
matched invocation blocks；inner resample 三 seeds；将 resampled seeds 的 raw total samples
合并后为每 arm 重新计算 p50；不得 bootstrap stage percentiles 后相加。

mAP bootstrap：outer resample official video IDs，inner resample 三 seeds。对每个被抽中的
seed，在相同 outer-resampled video multiset 上用 synthetic IDs
`boot/<replicate>/<seed-position>/<video-position>` 重建完整 predictions/GT 与该 seed 的 mAP；
replicate statistic 是三个 resampled-seed mAP 的 arithmetic mean。不得把不同 seed 的
predictions 合并后做跨 seed NMS，也不得把重复 timing invocations 当额外 mAP samples。

bootstrap replicates=5000，seed=`20260711`。

### 13.8 Gate-4 hard conditions

定义 `latency_saving=(p50_dense-p50_CT)/p50_dense`。Gate 4 同时要求：

1. latency_saving one-sided 95% LCB `>=0.15`；
2. `mAP@0.7_dense-mAP@0.7_CT` one-sided 95% UCB `<=1.5`；
3. fit-Q1 shortest-duration subset 上同一 mAP drop UCB `<=1.5`；
4. 每 matched invocation 定义
   `heavy_saving_i=dense_heavy_i-selected_heavy_i`、
   `overhead_i=innovation_i+scheduler_i+transport_i+cache_movement_i`、
   `margin_i=0.40*heavy_saving_i-overhead_i`；要求 full-sample median heavy saving `>0`，
   且 median margin bootstrap one-sided 95% LCB `>0`；
5. `p50_CT-p50_static` one-sided 95% UCB `<=0`；
6. CT 相对 calibration-frozen static 的 evaluation detector-regret absolute improvement
   hierarchical-bootstrap 95% CI lower `>0`；
7. 每 seed point latency saving、mAP@0.7 drop、shortest-Q1 drop、median margin 与 CT-static
   latency difference 均不得越过对应失败阈值。

calibration-frozen static identity 在 evaluation/bootstrap 中固定。evaluation-selected
diagnostic comparator 必须在每 replicate 重选，但不得替换 hard static comparator。hard
conditions 构成 intersection-union gate，不作额外 multiplicity correction。

p95、Avg-mAP、mAP@0.3--0.7、duration quartiles、throughput、peak memory、stage breakdown、
raw total distributions 与 block-level NVML energy 同时报告，但不替代 primary gates。energy
只用 10 Hz power 对长 timed block 梯形积分，不声称 single-inference energy。任一 primary
条件失败即 H4=`no`，路线永久冻结。

## 14. Provenance、registration 与 claims

### 14.1 Immutable pre-Gate1 registration

选择独立 immutable registration artifact；不得仅把 identity 写入后续 checkpoint。

流程固定：

1. 先产生 clean implementation commit `I`，包含最终 spec、实现、tests、configs、launchers
   与 adjudicators，但不含任何 profile/replay/evaluation result；
2. 在 detached clean worktree at I 生成 canonical
   `chronotransport_pre_gate1_registration.json`；
3. generator 不得接收或读取 profile、replay、calibration、evaluation、Gate report 或实验
   output path；
4. 将该 JSON 作为唯一内容变化提交为 registration commit `R`；
5. formal launcher Git HEAD 必须精确等于 R；
6. unlock chain 同时记录 I、R 与 registration exact-byte SHA-256。

registration 至少包含：protocol ID；spec commit/hash；implementation I；registration
parent/tree identity；全部 source/test/config/launcher file hashes；固定 upstream commits；dense
checkpoint registry ID/authenticated URI/content-addressed path/byte size/SHA-256；data-root
identity、annotation 与每个 media SHA-256；200-video split 与 one-window manifest hashes；16
candidate library/action hashes；Stage-B exposure matrices；Stage-C exposure formula；motion/random
algorithm hashes；bootstrap seeds；profiler invocation/order hashes；全部 Gate thresholds；expected
GPU model/UUID、driver、CUDA、PyTorch、cuDNN、precision 与 environment/container hashes；output
root；以及未读取 profile/replay/evaluation result 的 attestation。

若 checkpoint 只存在远端，registration 前必须从 authenticated registry 完整读取并计算
SHA-256，复制到 content-addressed path；remote path/mtime 不得替代 content hash。

formal launch 要求：`git status --porcelain` 为空；HEAD=R；重新验证 source/spec/config/
checkpoint/data/window/library/exposure hashes；symlink-resolved output root 位于允许根目录。

不强制 GPG/signature。公开 immutable commit R、registration SHA-256 与 content-addressed
inputs 是必要证据；签名只可作为预先存在的附加项。profile 完成后生成独立 profile artifact
hash 并链接 registration R，不得修改原 registration。任何 code/spec/data/checkpoint 变化要求
新的 I/R，旧结果不得迁移。

resume 必须重新验证 registration、profile input chain、checkpoint、optimizer、GradScaler、
EMA、scheduler、successful exposure ledger prefix 与 next cursor；任一 mismatch 为
`INVALID_IMPLEMENTATION`。

### 14.2 Checkpoint、EMA、manifest 与 claim provenance

每个正式 checkpoint/sidecar 必须绑定：source commit、spec file SHA-256、config hash、
upstream commits、dense checkpoint hash、shared split hash、library/action hashes、cost lookup
hash/environment fingerprint、seed、Stage-B/Stage-C successful updates、candidate exposure、
calibration ledger hash、q_conf与当前 gate/unlock status。

`risk_predictor` 与 `scheduler.predictor` 是同一逻辑模块的注册 alias。保存、EMA、校准与
strict load 前后必须断言 canonical/alias tensors 完全相等；logical predictor 只计算一个
canonical hash。alias 不一致直接 invalid，不依赖 state_dict traversal order。

claim flags 初始全部 false：

- `oracle_headroom`；
- `mechanism`；
- `calibrated_risk_on_frozen_window_protocol`；
- `metric_adatad_thumos14_official_full_video`；
- `latency_gpu1_fixed_stack`；
- `deploy`；
- `paper`。

Gate 1 PASS 只允许 `oracle_headroom=true`；Gate 2 PASS 只允许 `mechanism=true`；Gate 3
PASS 只允许 `calibrated_risk_on_frozen_window_protocol=true`；Gate 4 全 PASS 后只允许
`metric_adatad_thumos14_official_full_video=true` 与 `latency_gpu1_fixed_stack=true`。

即使 Gate 1--4 全 PASS，`deploy=false`、`paper=false`。deploy 至少还需要 official
full-video population 上独立 calibration/safety protocol、与 bounded appeal 不共享
evaluation selection 的新验证及 deployment hardware/traffic/failure-mode 审查。paper 至少
还需要外部 novelty/collision review、paper-level claims-evidence review；若主张
detector-agnostic plugin 泛化，还需要第二 detector 独立验证。

不得把 Gate 4 PASS 扩张为首次提出 time×depth routing、三动作 cache、MoD、conformal
compute control、通用 TAD deployment 或 detector-agnostic generalization。

## 15. 远端执行、产物和停止纪律

- 本地只做静态源码/规格检查、hash 与 launcher 文本审计；所有 unit/focused integration、
  CUDA 行为、replay、training、calibration、profiling 与 detector evaluation 全部在远端。
- 写入根限定 `/data/run01/sczc063/yuzibo`；环境固定 CUDA 11.8、miniforge 24.11、
  `/data/run01/sczc063/yuzibo/conda_envs/opentad`。
- 只允许物理 GPU1；`CUDA_VISIBLE_DEVICES` 必须精确为 `1`，且位于 Slurm allocation/step
  或已授权 protected allocation，禁止登录节点直接训练。
- 每个 launcher 必须先通过 `PRECHECK_ONLY=1`，验证 source/spec/config/checkpoint/split/
  library/cost hashes、clean implementation state、单可见 GPU、输出根与 unlock chain。
- run root 固定为
  `/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2/<registration_commit>/`。
- 共享 Gate-1 profile/replay/report 放 `shared/gate1/`；三 seed 产物放 `<seed>/`。
- 必须原子写 checkpoint/report，并以 `SUCCESS/FAIL/STOPPED/INVALID_IMPLEMENTATION` marker
  区分科学完成、科学失败、基础设施中断与实现错误。实现错误修复后从受影响 Gate 完整
  重跑，旧 run 永久保留为 invalid。

compact ledger 只允许 sample ID、split、seed、requested/executed schedule/action hash、
deploy-visible pooled signals、cost、pooled targets、regret、feature MSE、repair/fallback flags。
不得持久化 full-token state或把 raw detector predictions作为 scheduler 输入。

## 16. 完成定义

当前规格不得进入 writing-plans。唯一下一步是：

1. 将本文件作为新的 spec-only commit；
2. 独立复算 exact-byte SHA-256；
3. 执行一次只比较 r1→r2 规格文本、不读取任何实验结果的 review；
4. 仅在该 review 输出 `APPROVE_SPEC_FOR_PLAN` 后，状态改为 `spec_approved` 并调用
   writing-plans。

后续状态定义：

- `implemented`：focused tests、validators、Gate 1--4 adjudicators、Stage B/C、matched
  control、profiler 与 GPU1 launchers 全部落地；不代表科学 PASS；
- `tested`：远端 precheck 与受控 CUDA integration 通过；不代表主实验结论；
- `experiment_running`：formal hash-bound GPU1 job 已登记 Job ID/run root；
- `empirically_supported`：Gate 1--4 全 PASS，原始 reports 与 claims 已写回 wiki；
- 任一 Gate FAIL：`negative_gate/frozen_baseline`；禁止删结果、改 Gate、补 seed、换预算或
  再次上诉。

spec-only approval 前，implementation、profiling、Gate 1、新 seed 与 Stage C/P5 全部锁定。
