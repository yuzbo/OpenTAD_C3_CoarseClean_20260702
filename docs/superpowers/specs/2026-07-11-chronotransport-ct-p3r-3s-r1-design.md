# ChronoTransport CT-P3R-3S-r1 本地校正版有界上诉规格

日期：2026-07-11

协议编号：`CT-P3R-3S-r1`

决策状态：设计已获用户批准；本书面规格等待最终复核

实现状态：`designed`，不是 `implemented`、`tested`、`experiment_running` 或
`empirically_supported`

替代关系：本文件 supersede commit `b74101d` 中的
`2026-07-11-chronotransport-bounded-rescue-validation-design.md`；旧文件仅保留为历史记录，
不可执行

## 1. 裁决、证据边界与目标

ChronoTransport seed-3407 formal Stage-B/P3 已正式失败：旧 risk head 把 144 个
`chunk × layer-group` 非负 cell scalar 相加，却用窗口级 detector regret 监督；旧校准与
Spearman 又把同一窗口的 candidate rows 当独立样本。该 checkpoint、cell-sum head 和旧
P3 裁决协议已经死亡，Stage C/P5 从未解锁。

Pro review 的统计修订总体成立，但 reviewer 没看到本地源码。本地审计与空白上下文
independent agent 进一步确认：

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

### 3.3 截断梯度合同

`cache_detach=True` 保持不变：每次更新 `latest/anchor` 后 detach，截断跨 clip recurrent
gradient。forward 链式 transport 不受影响；当前 TRANSPORT 输出仍通过 frozen 或可训练的
dense adapter Jacobian 接收本 clip 梯度。不得在本次上诉中改成 full BPTT。

### 3.4 两种 age 不再混用

- `hard_cache_validity_age=47`：一个 48-clip window 在首 clip RECOMPUTE 后最多 age=47；
  因此冻结库中的 `hold_only/transport_only` 可按名称原样执行。
- `transport_age_embedding_cap=8`：保持现有 transport age embedding 结构；实际 age>8
  时仅 embedding index clamp 到 8。
- risk head 的 deterministic age feature 使用真实 `actual_age`，归一化为
  `actual_age / (1 + actual_age)`，不 clamp 到 8。

旧的单一 `max_cache_age` 字段禁止继续承担上述两个含义。formal run 中任何 schedule
repair 都使该样本与 cost lookup 失效；不得把 repaired schedule 冒充原 candidate。

## 4. 唯一窗口级 quantile head

旧 144-cell scalar sum 必须被唯一的 schedule-conditioned window head 替换，不允许并行
尝试第二种 pooling、normalization 或 head。

每个 cell 的输入固定为：deploy-visible 6-dimensional signal、action embedding、group
embedding、真实 normalized age。架构固定：

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

### 5.1 共享精确 split

THUMOS14 train-video manifest 必须断言恰有 200 个 unique video IDs。使用 split seed
`3407` 对 ID 做 canonical SHA-256 排序，一次性切成：

- fit：140；
- calibration：30；
- evaluation：30。

manifest 记录 exact IDs、cardinality、各 split hash、整体 manifest hash 与 split seed。
训练 seeds 3408/3409 必须加载 seed-3407 的同一 manifest；run seed 不得参与 split 构造，
任何 hash 或 cardinality 不一致都使 run invalid。

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

- `motion_p{2,4,8}`：仅使用 deploy-visible cosine change；在 calibration split 以无标签
  quantile 匹配对应 periodic RECOMPUTE count，然后冻结 threshold；其余为 HOLD；
- `random_p{2,4,8}`：以 `SHA256(sample_id, seed, group, period)` 选择与 periodic 完全
  相同数量的 RECOMPUTE positions，其余 HOLD；
- `uncalibrated_risk`：同一 trained head，offset 强制 0；
- `oracle`：只存在于 evaluation-only adjudicator，不进入 checkpoint 或 scheduler。

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

每个 update 只训练一个 non-dense schedule。令 canonical candidate index 为 0–15，update
index 为 0–139，schedule 为 `(update + seed_offset) mod 16`，其中：

- seed 3407 offset=0；
- seed 3408 offset=4；
- seed 3409 offset=8。

每 seed 有 12 个 candidates 暴露 9 次、4 个暴露 8 次；汇总三 seed 后，candidate exposure
只能是 26 或 27 次。不得称每 seed 完全均衡。fit video order 使用 manifest canonical order；
augmentation RNG 由 run seed 控制。artifact 必须登记每个 `seed × update × sample_id ×
candidate`，以及 candidate×video exposure matrix。

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

## 9. Gate 1：冻结库 equal-cost oracle headroom

Gate 1 在任何新 Stage-B seed 训练前运行，只评估 HOLD schedules 与 HOLD-based controls，
避免把 H1 与 learned TRANSPORT 混淆。它只支持“冻结库存在预算可行的 oracle headroom”，
不能支持 deploy-visible input dependence。

对 evaluation 窗口定义：

- joint oracle：在全部自身 p50<=B* 的 HOLD schedules 中逐窗口选最低 detector regret；
- time-only oracle：仅在 P2/P4/P8 HOLD 中逐窗口选择；
- layer-only oracle：仅在 early/late HOLD 中逐窗口选择；
- calibration-frozen global static：在 calibration split 选择平均 regret 最低且 p50<=B*
  的单一 HOLD schedule，然后冻结到 evaluation；
- motion/random：使用 calibration 冻结的 controls；
- evaluation-best global static：可在 evaluation 上选择，但只报告 diagnostic upper
  comparator；不得进入任何 pass/fail、checkpoint、threshold 或后续部署选择。

primary comparator 是 time-only oracle、layer-only oracle、calibration-frozen static、
motion 与 random 中 evaluation mean regret 最低者。Gate 1 同时要求：

1. joint oracle 相对 primary comparator 的 mean detector-regret relative reduction ≥10%；
2. paired unique-window bootstrap absolute improvement 95% CI lower>0；
3. 对 exact feasible-HOLD-name tuple 相同的窗口分层，在层内置乱 joint-oracle schedule
   assignment 5000 次；`regret_shuffled - regret_true` 的 95% CI lower>0；少于两个可置乱
   窗口时直接 FAIL；
4. B* 相对 dense 的 measured full-stack p50 saving ≥20%。

任一失败即 H1=`no`，路线永久冻结，不训练新 transport/risk seeds。

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

### 11.2 ranking sample unit

对每个 `seed × evaluation window`，在完整 16-candidate vector 内计算 Spearman
`rho_seed,window`。predicted vector 或 regret vector 少于 3 个 distinct ranks 时，该窗口
fail closed；不得填 0 或静默删除。

每 seed score 是其 30 个 window rho 的 arithmetic mean；pooled score 对 seed/window
等权。bootstrap 外层以 unique window ID resample，内层 resample 三 seeds；每个窗口的
完整 candidate vectors 整体移动。

### 11.3 scheduler

scheduler 仅在 non-dense candidates 中筛选：

`requested_p50<=B* AND upper<=epsilon AND finite AND metadata/hash valid`。

在可行项中选 requested measured p50 最低者；cost tie 按 canonical library order。无可行
项时选择 dense safety fallback。dense risk=0，但不因此成为 budget-feasible success。

### 11.4 Gate 条件

evaluation 上同时要求：

1. 每 seed mean rho≥0，三个 seed means 的 median≥0.2；
2. pooled rho hierarchical-bootstrap 95% CI lower>0；
3. 实际选择为 non-dense 的 rows 上 pooled `upper>=regret` coverage≥0.85；逐 seed coverage、
   unique selected windows、cluster-aware CI 与 all-window coverage同时报告；coverage>0.95
   标记 `OVERCOVERED`，不作为 hard FAIL；
4. evaluation pinball loss 比 schedule-conditioned fit quantile constant baseline 至少低10%；
5. 至少 20% 的 `seed × evaluation window` 选择 non-dense；
6. calibrated、offset=0 uncalibrated 与 dense fallback 的 selection rate、regret、upper
   sharpness和成本并列报告。

任一 hard 条件失败即 H3=`no`；不得换 pooling、normalization、quantile、epsilon、loss、
head width 或额外训练。

## 12. 统计、诊断 proxy 与泄漏边界

### 12.1 bootstrap

- Gate 1：5000 次 paired unique-window bootstrap。
- Gate 2–4：5000 次 paired hierarchical bootstrap，外层 resample unique sample IDs，
  内层 resample seeds；同一 sample 的 candidate/period vectors 必须整体移动。
- CI：percentile 95%，bootstrap seed=`20260711`。
- relative denominator 固定为 comparator mean，以 `1e-12` 防零；同时报告 absolute
  difference，所有 CI gate 使用 absolute difference。
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

Gate 1–3 全 PASS 后才生成 hash-bound Stage-C unlock artifact。Stage C 固定 60 epochs，
沿用官方 batch order、scheduler、AMP 与三个 seeds，但显式 optimizer groups 为：

- AdaTAD adapters：LR `2e-4`，weight decay `0.05`；
- CT transport+risk：LR `1e-4`，weight decay `0`；
- heavy VideoMAE、projection/head 与其他参数：frozen/excluded。

必须按 object identity 审计每个 trainable parameter 恰好属于一个非零 LR group；
`risk_predictor` 与 `scheduler.predictor` alias 不得重复进入 optimizer。

matched dense Stage-C control 使用相同起点、seed、fit window order、augmentation seed、
60-epoch workflow、common AdaTAD parameter LR/WD、detector task loss与成功 common-parameter
updates；forced dense，不训练 transport/risk。两个 arms 都记录 attempted updates、AMP skip
vector、successful common updates、LR trace 与 EMA trace；任一 exposure 不一致使比较 invalid。

Stage C 后必须在 calibration split 重新计算唯一 q_conf，并重新通过 Gate 3，才可执行
learned scheduler detector evaluation。

Stage-C CT arm 不对 scheduler argmin 反传。训练 schedule 继续按 global successful update
使用第 7 节 canonical 16-candidate cycle 与相同 seed offsets，循环 60 epochs。objective
仍为第 7 节三项，但梯度所有权固定：

- `L_detector_counterfactual` 更新 AdaTAD adapters 与 transport；
- feature MSE 只更新 transport，AdaTAD adapters 在该辅助项中视为 stop-gradient operator；
- pinball loss 只更新 risk head，signals 与 regret target 对其他模块 detach。

matched dense control 的 AdaTAD adapters 只接受同一定义的 detector task loss。这样 common
adapter parameters 的 loss exposure 可匹配，不允许 auxiliary loss 偷偷改变 adapter objective。

Gate 4/P5 固定比较：原始 dense checkpoint、matched dense Stage C、calibration-frozen
global static、P2/P4/P8 T/H、early/late/progressive/reverse、motion/random、uncalibrated risk、
calibrated risk与 diagnostic-only oracle。

calibrated scheduler 同时要求：

1. 相对 matched dense control full-stack p50 saving≥15%；
2. mAP@0.7 absolute drop≤1.5；fit-Q1 shortest-duration subset mAP@0.7 drop≤1.5；完整报告
   Avg-mAP、mAP@0.3–0.7和所有 duration quartiles；
3. 对每个 matched invocation，`overhead_i=innovation+scheduler+transport+cache_movement`，
   `heavy_saving_i=dense_heavy_i-selected_heavy_i`；要求 p50(overhead)≤0.40×
   p50(heavy_saving)，且 p50(heavy_saving)>0；
4. 同一 B* 下，相对 calibration-frozen static 的 evaluation detector regret absolute
   improvement hierarchical-bootstrap 95% CI lower>0，且 full-stack p50 不高于该 static；
5. 三个 seeds 的 latency、mAP@0.7 和 shortest-Q1 mAP@0.7 均不得单 seed 越过上述失败阈值。

p95、peak memory、throughput、stage breakdown 与 GPU energy 同时报告。energy 只作 secondary
metric：NVML 10 Hz power samples 对长 timed block 做梯形积分，不声称精确 single-inference
energy。任一 primary 条件失败即 H4=`no`，路线永久冻结。

## 14. Checkpoint、EMA、manifest 与 claim provenance

每个正式 checkpoint/sidecar 必须绑定：source commit、spec file SHA-256、config hash、
upstream commits、dense checkpoint hash、shared split hash、library/action hashes、cost lookup
hash/environment fingerprint、seed、Stage-B/Stage-C successful updates、candidate exposure、
calibration ledger hash、q_conf与当前 gate/unlock status。

`risk_predictor` 与 `scheduler.predictor` 是同一逻辑模块的注册 alias。保存、EMA、校准与
strict load 前后必须断言 canonical/alias tensors 完全相等；logical predictor 只计算一个
canonical hash。alias 不一致直接 invalid，不依赖 state_dict traversal order。

claim flags 初始全部 false：`oracle_headroom / mechanism / calibrated_risk / metric / latency /
deploy / paper`。Gate 1 只允许 oracle_headroom；Gate 2 只允许 mechanism；Gate 3 只允许
calibrated_risk；Gate 4 全 PASS 后才允许在固定 AdaTAD/THUMOS14/GPU1 范围打开 metric 与
latency。即便全 PASS，也不得声称首次提出 time×depth routing、三动作 cache、MoD 或
conformal compute control。

## 15. 远端执行、产物和停止纪律

- 本地只做静态、unit、focused integration 和 launcher precheck；真实 CUDA 行为、replay、
  training、calibration、profiling 与 detector evaluation 全部在远端。
- 写入根限定 `/data/run01/sczc063/yuzibo`；环境固定 CUDA 11.8、miniforge 24.11、
  `/data/run01/sczc063/yuzibo/conda_envs/opentad`。
- 只允许物理 GPU1；`CUDA_VISIBLE_DEVICES` 必须精确为 `1`，且位于 Slurm allocation/step
  或已授权 protected allocation，禁止登录节点直接训练。
- 每个 launcher 必须先通过 `PRECHECK_ONLY=1`，验证 source/spec/config/checkpoint/split/
  library/cost hashes、clean implementation state、单可见 GPU、输出根与 unlock chain。
- run root 固定为
  `/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r1/<implementation_commit>/`。
- 共享 Gate-1 profile/replay/report 放 `shared/gate1/`；三 seed 产物放 `<seed>/`。
- 必须原子写 checkpoint/report，并以 `SUCCESS/FAIL/STOPPED/INVALID_IMPLEMENTATION` marker
  区分科学完成、科学失败、基础设施中断与实现错误。实现错误修复后从受影响 Gate 完整
  重跑，旧 run 永久保留为 invalid。

compact ledger 只允许 sample ID、split、seed、requested/executed schedule/action hash、
deploy-visible pooled signals、cost、pooled targets、regret、feature MSE、repair/fallback flags。
不得持久化 full-token state或把 raw detector predictions作为 scheduler 输入。

## 16. 完成定义

- `spec_approved`：本文件通过用户书面复核，Git commit 与 detached file SHA-256 冻结。
- `implemented`：focused tests、validators、Gate 1–4 adjudicators、Stage B/C、matched control、
  profiler与 GPU1 launchers 全部落地；不代表任何科学 PASS。
- `tested`：远端 precheck 与受控 CUDA integration 通过；不代表主实验结论。
- `experiment_running`：正式 hash-bound GPU1 job 已登记 Job ID/run root。
- `empirically_supported`：Gate 1–4 全部 PASS，原始 reports 与 claims 已写回 wiki。
- 任一 Gate FAIL：状态为 `negative_gate/frozen_baseline`，也是完整科学结果；禁止删除、
  改 gate、补 seed、换预算或再次上诉。

本规格没有授权立即运行实验。下一步只有：书面规格复核通过后编写 implementation plan，
按 TDD 实现，再依顺序部署到远端 Gate 1。
