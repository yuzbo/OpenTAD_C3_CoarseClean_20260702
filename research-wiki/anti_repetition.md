---
type: anti_repetition
updated: 2026-07-29
---

## GeoRoute deployment anti-repetition

0. Do not treat a Pro proposal as an implementation receipt. The 2026-07-29
   CER-TAD review correctly diagnoses Free v1 and motivates complementary
   evidence routing, but its dynamic role-count likelihood, critic, boundary
   head, coverage/stability losses, and numerical weights are underspecified.
   They remain `discussed` until the estimator/representation preexperiment
   passes.
0. Do not reuse the old seven-arm Free-first selector or its failed namespace
   for a new CER or estimator study. A changed arm set requires a new study ID,
   contract, selector, source commit, and namespace.
0. Do not claim a support-selection gain while absolute coordinates,
   ROI-relative coordinates, or the geometry projection differ between arms.
   These three representation paths must be independently switchable and
   matched.
0. Do not adopt the review-proposed `+0.50 pp` / `+0.30 pp` accuracy margins as
   confirmatory gates. They were proposed after the old development results and
   lack independent variance/power justification. Pilot estimates are
   exploratory and must be separated from confirmatory seeds.
0. Instrumentation replay is valid only in a new diagnostic namespace with
   exact prediction-SHA and population parity. It cannot repair the old ROI
   decode failure, complete the old selector, or create paper evidence.
0. D/K/M finalization
   `78b0598c70c9966dfd4e7bfa0cce35cfe3ec7d00ed016d0c3268a214e36e86fc`
   authorizes only the independent six-arm exploratory pilot. Never reinterpret
   `GO_PILOT_DESIGN_ONLY` as CER implementation, P2/P3 authorization, official
   test permission, or paper evidence.
0. In `georoute_estimator_representation_pilot_v1`, keep
   `absolute_position_enabled=true` in all arms. Representation-off means only
   the three new detector-visible paths are off: absolute source coordinates,
   ROI-relative coordinates, and geometry projection.
0. The exploratory pilot has no automatic winner. Do not use one seed or
   post-result margins to promote an arm. First report the four frozen
   contrasts; only then may its variance inform a new protocol with disjoint
   confirmatory seeds.
0. Never resume or interpret
   `georoute_estimator_representation_pilot_02b6efe7_20260729_1805`.
   P0 Jobs `1203380`--`1203385` failed mechanically before any model result;
   no training leaf ran and no performance conclusion exists. A repair requires
   a new commit and namespace.
0. JSON object key order is not experimental arm order. Deployment validators
   must compare the exact arm-key set, normalize it back to the frozen arm
   order, require unique numeric Slurm IDs, and then bind by arm. Never reject
   a valid receipt merely because atomic `sort_keys=True` serialization changed
   insertion order.
0. Never put an `afterany` closeout behind descendants that can remain
   `DependencyNeverSatisfied`. For the estimator pilot, P0 finalization runs
   `afterany` over all P0 leaves; every training wrapper runs `afterany` over
   that finalizer but must verify the sealed PASS P0 suite before creating its
   cell or launching training; the final closeout then runs `afterany` over all
   terminal leaves. A failed P0 must end in an INCOMPLETE receipt without
   partial-performance inference. P0-finalizer and final-closeout
   prevalidation/sealing exceptions must write hashed fail-safe receipts before
   re-raising.
0. Never launch a multi-cell 60-epoch GeoRoute matrix with per-epoch full
   model/optimizer/EMA checkpoint retention and no aggregate storage
   preflight. Jobs `1196071`--`1196077` accumulated 63 GB, filled `/data`, and
   all failed during checkpoint publication before any result JSON. A
   replacement must use a new namespace, prove aggregate headroom, and retain
   final EMA only or an explicitly bounded result-blind checkpoint set. Partial
   checkpoints and epoch logs are not P1 evidence and the failed namespace must
   not be resumed.
0. Never create synthetic native support by replicating a spatial remainder
   and then route it without a validity invariant. Match the pretrained
   floor-Conv3d support, propagate a boolean validity mask, and fail closed when
   the valid count is below K.
0. Packed attention/MLP is not a packed backbone claim while the original
   Adapter still executes on a dense carrier or lets unselected positions mix
   into selected lineages. P1R requires the coordinate-lineage packed Adapter
   and a full-K numerical parity gate.
0. A `free` NativeTokenSelect control must use fixed full-frame geometry and a
   frozen geometry head. Never call a learned-geometry route ROI-free.
0. Membership comparisons must use identical uniform-selected pooling.
   Route-logit pooling is a separately named ablation, not a hidden learned-
   route advantage.
0. Deterministic context may not receive hybrid route gradients. The
   straight-through or score-function surrogate must match the hard staged
   branches and temporal route likelihood.
0. Parallel Slurm scheduling is not parallel causal interpretation. All seven
   P1R arms may run concurrently after P0R, but NativeTokenSelect must pass its
   fixed/random/geometry-side-channel controls before geometry is interpreted.
0. Never revive the failed `1196071`--`1196078` namespace, infer utility from
   its partial checkpoints, or call the current native-token router a
   source-pixel crop. Do not use “Geometry Zoom” unless the native base and the
   strict conditional geometry add-on both pass, followed by multi-seed,
   cost, diagnostic, and generalization closure.
0. Every N16R4 GitHub clone/fetch/pull/ls-remote must use the academic
   acceleration proxy frozen in `RTK.md` from the first network attempt. After
   syncing, bind full HEAD and remote-tracking SHA and require a clean tree.
   Direct GitHub attempts, uncommitted source copies, or rsync-overwritten
   snapshots are not valid experiment provenance.
0. Never launch concurrent single-node GeoRoute leaves with an implicit
   `torch.distributed.run --standalone` rendezvous endpoint. Slurm may place
   independent one-GPU jobs on the same node, where the default localhost port
   aliases their TCPStore lifetimes. Job `1199869` attached to fixed lattice's
   port `29400` and died when that store closed; hybrid `1199871` repeated the
   failure on `g0048`, terminating nine seconds after random logged
   `Training Over`. Every leaf must use a unique or kernel-assigned endpoint
   such as the already audited `127.0.0.1:0` pattern, bind a unique rendezvous
   ID, and pass an intentional same-node concurrent isolation gate. A bind
   collision invalidates the cell; never resume or infer model utility from it.
0. On N16R4, do not rely on `srun --resv-ports`: co-located diagnostic Jobs
   `1203460/1203461` both failed immediately with `Requires more ports than can
   be reserved`. The repaired GeoRoute contract instead derives a distinct
   `127/8` loopback address from the decimal Slurm job ID, retains a
   kernel-assigned port and cell/phase-bound rendezvous ID, and records the
   actual runtime port. Keep the audited 120-second readiness bound and hashed
   per-probe failure sidecar; terminate the whole torchrun process group on
   failure so no worker or TCPStore survives its parent. A bare timeout string
   is not sufficient evidence.

## Continuous-RoI S2 deployment anti-repetition

0. Never pass a Windows/CRLF-derived final CLI value into a Slurm
   `--export` string without fail-closed validation. Every export key/value
   must reject ASCII controls, leading/trailing whitespace and commas before
   `sbatch`, and the launcher must repeat the check before nested `srun`.
0. A deployment launcher is experiment code. It must resolve to the tracked
   canonical path, match the expected Git blob, and be rehashed immediately
   before every submission. A caller-selected launcher is not an auditable
   alternative.
0. A Gate-passing campaign whose training launcher fails is immutable
   deployment-failure evidence. Never edit, resume or reinterpret it as a
   model result; use a new commit, Gate authorization and campaign namespace.
0. `SUBMITTED`, `PENDING`, `RUNNING`, `Training Starts`, and epoch-0 logs are
   distinct states. None is crop-sufficiency evidence; only complete registered
   development results can advance the scientific claim.

# 禁止重走清单

## Native-Crop S2 协议反例

0. 最终方法不得是固定分辨率、固定窗口大小或从 21 个固定 `128x128`
   位置中离散选一个。目标是连续回归 source-coordinate
   `(cx,cy,w,h)`；中心、宽、高、尺度和纵横比均可变化。
0. 固定的是 local backbone 的批处理 tensor shape 时，必须明确它只是对可变
   source ROI 的重采样规格，不得把它误写为固定 source crop，也不得继续声称
   strict native-pixel-density/no-resize 是最终方法特征。
0. 21-candidate fixed library 只能作为 D0 sanity/baseline。它的通过或失败均
   不能代替连续 variable-RoI sufficiency，更不能 KILL 连续回归路线。
0. 连续宽高学习必须防止 `w,h -> 0` 的退化，并对 in-bounds、面积/尺度、纵横比和
   时间平滑给出可微参数化及测试；不得依赖推理后硬裁剪掩盖训练退化。
0. 不得把 GT 可见、逐窗口词典序选择的 reference 称为 21-candidate library
   的上界或 global-mAP oracle。它通过可作为充分证据；它失败只能否定该规则，
   不能据此 `KILL_THIS_LIBRARY`。
0. 不得把 crop sufficiency、adaptive-selection headroom 和 deployable cost
   viability 合并成一个二元 GO/KILL。固定 crop 足够但无选择 headroom 时，应记录
   `SUFFICIENT_FIXED_CROP_ONLY`，而不是宣判 crop library 失败。
0. gate raw predictions 封存前不得创建可被训练或推理命名空间访问的 gate GT
   target cache。顺序必须是 no-GT raw sweep、不可变 receipt、特权 GT join。
0. 不得用 video-cluster bootstrap 同时代表检测 mAP 与 ABBA latency/energy
   不确定性。检测和成本必须按各自采样单位校正，再做交集裁决。
0. 不得把确定性的 candidate geometry coverage 与训练模型产生的
   `CandidateUnionRecall` 混为一谈；后者应称 model-conditioned reachability。
0. 不得声称 selector-free 的 policy-shaped path 已证明 learned selector 的部署成本。
   必须预留 selector 成本预算，或只主张 representation-path headroom。
0. 不得在看过 S2 结果后调整等效、headroom 或成本 margin。冻结前只能用 synthetic
   或历史方差做 result-blind power/Monte-Carlo feasibility audit。

## Native-Crop S1 新增反例

0. 不得把 `320x180` 称为原始摄像机采集分辨率；它只是当前数据副本中
   ffprobe/Decord 可见的解码源分辨率。
0. 不得沿用会漏掉 development 身份的滑窗设置。旧 `0.25` overlap
   会遗漏 `video_validation_0000054` 的末尾短动作；Native-Crop 的
   population audit 必须覆盖冻结 manifest 的 fit 160 / gate 40，
   当前隔离配置使用 `0.5` overlap。
0. 不得把共享 VideoMAE 权重写成一次 backbone 计算。global/local
   两个视图复用同一参数实例，但仍产生两次前向计算；成本证据必须分开记录
   global backbone 与 local backbone。
0. 不得把 source-pixel equality、no-padding census、nonzero gradient
   或 `[B,384,768]` shape parity 当成 crop sufficiency。它们只授权进入
   development crop-sufficiency 协议讨论。

## Spatial Zoom 当前边界

0. 不得把整图 `Resize + CenterCrop/RandomResizedCrop` 实验称为空间选择、原生分辨率
   crop 或 Zoom。旧 `Dense-160/224/256` 仅是 R0 分辨率控制，不再是 Native-Crop S1
   的逻辑必要前置门槛。
0. Native crop 必须先在源帧坐标中选择区域，再以明确记录的局部像素密度进入重分支。
   不得把低分辨率 crop 放大回完整重模型输入后仍声称节省了像素、FLOPs 或端到端成本。
0. 在 oracle/teacher-reference crop sufficiency 通过预注册 GO 条件前，不得实现 learned
   ROI policy；也不得继续为旧 dense-resize recovery matrix 消耗 GPU 来替代 crop 验证。
0. 不得把有限的八候选或其他固定 candidate library 称为整个连续空间的 oracle。没有
   coverage certificate 时，候选库失败只能否定该库，不能直接 KILL continuous crop。
0. 不得声称 final masked mean 已消除 padded token 污染。ViT self-attention 会在 pooling
   前混合 token；优先将固定 crop 平移回图内，仅在源帧小于 crop 时 padding。
0. 不得在 source-geometry 与统计审计前冻结 `96/128` crop、48 knots、速度/尺度约束或
   `+1 pp/30%` 等 GO/KILL 门槛。
0. 不得先物化或 H2D 整段 768 帧 native-resolution float tensor 再裁剪。必须尽可能在
   decoded uint8 source frames 上完成 global/local crop 后再 format 与传输。
0. 下一垂直切片不得使用 teacher、GT、oracle 或 official-test evidence。teacher split、
   cache、候选训练分布与 formal test exception 必须在 oracle 实验前另行冻结。

1. 不得把旧 R0 称为 Zoom/crop 模型。R0 只有 matched dense spatial-resolution matrix。
2. 不得在 Native-Crop S1 GO 前实现 learned ROI policy，或用 oracle ROI 结果倒推修改
   已冻结的预注册门槛。
3. 不得把 DUCA、时序选帧、dynamic budget、max-gap 或 X3D/SlowFast prior 混入当前任务。
4. 不得恢复 `35204f5` 的 warning-bearing partial checkpoints 作为正式结果；替换矩阵必须从
   新 exact commit、新 precheck 和新 canonical experiment namespace 全量重跑。
5. 不得把 precheck、pilot、checkpoint 数量或中间 epoch 当作 S1 性能结果。
6. S1 的正式统计不得拒绝缺少稀有类的 bootstrap replicate；使用正权重 paired Bayesian
   video-cluster bootstrap，并保持 baseline/candidate 同 replicate 配对。
7. 成本只允许表述为同节点同 GPU 的 warm serial per-window latency 与 gross GPU energy，
   不得冒充 cold-start、whole-video p95、incremental energy 或完整系统能耗。
8. VideoMAE `return_feat_map=True` 会绕过分类出口 `fc_norm`；formal gradient gate 只能
   精确允许 `backbone.model.backbone.fc_norm.{weight,bias}` 两个参数无梯度。不得用前缀、
   正则或宽泛白名单掩盖新的断图。
9. S1 只持久化预注册的 gate-eligible checkpoints；不得保存不会参与选择的 pre-gate
   周期权重耗尽共享存储。任何存储故障后的矩阵不得 resume，必须新 commit、门禁和 namespace。
10. S1 selector must follow the official evaluator's prediction-domain policy:
    finite zero-length proposals remain zero-IoU false positives. Do not reject
    or delete them, because either action diverges from or inflates official AP.
    The in-training evaluator log is not a gate score when its GT population is
    broader than the frozen gate prediction population.
11. Post-processing repair code must not reconstruct a historical bound config
    against its own current `ROOT` or commit. It must derive the original clean
    repository from the recorded audited config path, verify its exact Git HEAD
    and config matrix, and validate the original precheck there. Never copy or
    rewrite bound configs merely to make a repair snapshot accept them.
12. A clean repair clone does not own the training snapshot's ignored `data/`
    mount. Repair entrypoints that instantiate the official dataset must run
    with the historical clean training snapshot as the working directory while
    importing the certificate-bound repair code explicitly. Do not add hidden
    symlinks or Git excludes to make relative dataset paths appear available.
13. Do not certify a long formal power profile from a short synthetic cadence
    test while the sampler remains a Python thread inside the detector process.
    Job `1167538` proved that native NVML can still suffer a `2413.519` ms
    observed gap under the full memory-heavy inference/NMS path despite passing
    a ten-second Gate. Keep the 20 ms target and 100 ms limit unchanged; require
    an independently scheduled UUID-bound sampler process, preserve the raw
    failure trace, and pass a representative long-duration no-open stress Gate
    before any replacement matrix.
14. A locally passing sidecar implementation is not a passed Gate. Do not
    submit a replacement matrix until a clean remote snapshot completes the
    full 792-exposure dense256/seed3408 path with the frozen 20/100 ms cadence,
    4+1 CPU isolation, UUID parity, unchanged test-evidence hash, and no formal
    profile publication. Submit exactly one serial matrix only after that Gate.
15. Do not require a separately scheduled Gate and matrix to receive the same
    physical GPU UUID. The Gate must bind its own actual UUID; the matrix must
    match the Gate's stable hardware/software class, bind its own actual UUID,
    and keep all nine cells in one allocation on one physical GPU.
16. Do not pair a sidecar report with an independently selected trace. Every
    consumer must use the shared attempt validator and recompute trace hash and
    cadence. Partial salvage may publish a missing hash-matching counterpart
    but must never overwrite an existing report or trace.
17. A matrix namespace is single-use. The persistent atomic matrix lock and
    start/completion receipts are evidence, not temporary scheduling files.

## Native-Crop provenance anti-repetition

1. Do not accept `expected_commit` while ignoring untracked files or merely
   recording working-tree hashes. Every audited executable/configuration file
   must be tracked and byte-equal to its `HEAD` blob before the gate runs.
2. Do not treat a self-hashed geometry census as source-bound evidence. The
   gate must re-probe every frozen development video and match containment,
   path, file size, dimensions, rotation, frame count, and frame rate.
    Never remove a failed lock to resume or duplicate the same campaign.
18. Do not lower the formal 90,000 MiB memory floor to fit N16R4's 55 GB
    one-GPU outer-job default, and do not override `CUDA_VISIBLE_DEVICES`.
    Reserve the site's two-GPU outer resource only when required to obtain
    sufficient memory, then run the entire Gate or frozen matrix in one exact
    Slurm step with one GPU, five CPUs, and 96,000 MiB. Record the step-scoped
    GPU and finite cgroup limit. The idle outer GPU is scheduling overhead,
    not model compute or measured cost, and must be disclosed.
19. Do not acquire or consume a formal matrix namespace before all no-write
    preflights pass. This includes source/artifact checks, the representative
    cell, finite cgroup v2 memory, step-GPU membership, logical-CUDA/NVML UUID,
    Gate hardware/software class, and in-memory matrix-start receipt
    validation. Never hand-write an alternative receipt in a launcher.
20. Do not combine profiles from different Slurm jobs, steps, or GPUs. Every
    profile, marker, and descriptor must bind the same canonical start receipt;
    the analyzer requires one completion receipt that seals exactly nine
    frozen-order descriptors. A directory count is not equivalent evidence.
21. Do not open any new sealed-test cell before all nine frozen cells pass the
    matrix no-write dry-run and the current start receipt passes runtime
    hardware/software validation. A per-cell preflight after another cell has
    opened the test is not a substitute for the all-cell dry-run.
22. Do not accept unbound official-test evidence from a prior matrix. The only
    exception is the historical dense256/seed3408 evidence whose canonical
    path, file hash, internal hash, and cell identity are frozen by the active
    recovery certificate. Every newly opened cell must publish and validate a
    canonical test-to-matrix binding before profiling.
23. Do not treat marker, stdout, or stderr files as proof that the NVML sidecar
    started. Salvage is authorized only by sidecar PID/ready/raw-power/result
    evidence, and a failed salvage must remain a hard failure.
24. Do not run the historical training snapshot's `tools/test.py` unchanged
    inside the high-memory two-level Slurm allocation. Job `1170468` proved
    that its old guard rejects the valid exact one-GPU step because it inspects
    the two-GPU outer `SLURM_JOB_GPUS`. Never falsify Slurm variables or edit
    that snapshot in place. A replacement must use a recovery-certificate-bound
    runtime entrypoint while proving the model/config/evaluator tree is unchanged.
25. One completed descriptor in a failed matrix is diagnostic cell evidence,
    not a partial 3x3 result. Without the exact-nine completion receipt it
    cannot select a resolution, drive GO/KILL, or be pooled with another
    campaign.
26. Do not recompute S1 evidence with ambient Python user-site packages.
    Runtime commit `6524e1b` proved NumPy `2.2.6` changes tied-score AP ordering
    relative to the formal Conda NumPy `1.23.5`. This is not grounds for metric
    tolerance: formal Gate/matrix/cell launchers must set
    `PYTHONNOUSERSITE=1`, and exact stored metrics must reproduce.
27. A failed post-`sbatch` receipt writer does not authorize another
    submission. Reconcile the existing Job by name/accounting first, then
    atomically bind that Job ID with the certificate-bound Conda Python.
    Job `1170765` was recovered this way after the login-node bare `python`
    rejected an f-string; no duplicate Gate was submitted.
28. Do not use a recovery schema name as a proxy for inherited runtime
    capability. Job `1170765` failed before sidecar startup because v5 carried
    the exact buffered-sidecar contract while the profiler accepted only the
    literal v4 reason. Validate backend, atomic publication, no loop I/O,
    20/100 ms cadence, 4+1 CPUs, and long-Gate fields together. A descendant
    recovery must also bind the failed parent inventory and prove no sidecar or
    new test evidence appeared; the failed campaign remains immutable.
29. Do not let recovery evidence roles alias or mix. Parent certificates and
    Gate stdout/stderr must use their canonical campaign paths; stdout/stderr
    cannot share a path or inode. Legacy, matrix, Gate, and power-diagnostic
    evidence are mutually exclusive for a schema transition, and incomplete
    role sets must fail closed rather than be ignored.
30. Do not merge the pre-policy S2 representation gate with S3 learned-policy
    training while retaining the old stage names. A learned ROI head inside S2
    changes the scientific question and cost semantics.
31. Do not derive fixed-center, random, discrete, or fixed-size controls by
    overriding a variable-box-trained checkpoint only at inference. A
    decision-critical comparison requires matched training distributions.
32. Do not compare a per-window GT-privileged continuous reference with
    unprivileged fixed or location-only controls. Privilege and search budgets
    must match before attributing gains to variable width/height.
33. Do not certify continuous spatial-reference adequacy merely because a
    detector-confidence objective converges. Confidence optimization can select
    false positives or action interiors and is only a no-GT policy diagnostic.
34. Do not double-count both a measured ROI policy head and a future-selector
    reserve in the same cost path.
35. Do not delete a failed or contaminated formal namespace. Preserve it
    immutably and create a new recursively bound campaign.
36. Do not freeze one-GPU high-memory Slurm requests, storage floors, or NVML
    gap thresholds that contradict the audited N16R4 allocation and validated
    20/100 ms sampler contract.
37. Do not call equal `sx,sy` an equal physical center trajectory when the
    decoder also conditions `cx,cy` on `w,h`. A fixed-size versus variable-size
    contrast must pair physical centers explicitly or state the center change
    as part of the intervention.
38. Do not launch the Continuous-RoI S2 reference sweep until the Sobol engine,
    dtype, transform serialization, stable-hash bytes and known-answer hash are
    frozen. A seed and draw shape alone are not an auditable generator identity.
39. Do not interpret the raw no-leak ban on a preferred/GT-selected reference
    ID as a ban on a result-blind enumerated candidate ID. Freeze the typed raw
    schema and object-graph audit before inference; never let the privileged
    join run in the raw GPU process.
40. Exact-nine training completion proves only training/exposure integrity.
    It is not development mAP, reference adequacy, crop sufficiency, cost
    viability, official-test evidence, or authorization for S3.

## 任务与叙事

1. **不要再称 Online TAD。** 当前方法观察完整离线窗口；`online` 仅表示 forward
   内生成且不查 ledger/cache。
2. **不要把 THUMOS14 解释成 key-event timestamp spotting。** 它监督动作区间；项目
   应表述为边界敏感的稀疏 interval detection。
3. **不要把插件泛化当作已证明。** 当前只有 AdaTAD-derived 主路径，第二 detector
   仍缺正式结果。

## 模型

4. 不再回到“粗分类器独立训练 → selector 独立训练 → detector 独立训练”作为最终
   方法；它只能是归因 baseline。
5. 不允许 `asformer_lite` 冒充官方 ASFormer。
6. actionness 必须由二分类 GT 校准，但 selector 必须以 transition/boundary/utility
   为首要目标；不能再次退化为 actionness top-k。
7. 不允许用硬膨胀、uniform scaffold、max-gap repair 把坏分数修成看似合理的网格而
   不披露 repair 数量和影响。
8. `detector_utility_target` 若来自 GT 边界，只能叫 boundary-utility proxy。
9. 不得声称“完全未修改官方 AdaTAD”；源码 wrapper、selected-axis 和 GT remap 已变。

## 训练与梯度

10. nonzero grad 只证明连通，不证明梯度方向等价于 hard frame utility。
11. loss schedule 必须按 optimizer step 推进，不能按 raw forward 次数。
12. detector backend loss 与 selector gradient bridge 必须分开：关闭 bridge 不得关闭
    detector 学习。
13. dynamic budget 不得只优化 expected K；必须记录真实执行 K 与实测成本。

## 实验

14. 不再重复排同一 X3D dense export/grid；它计算过慢且可能吞掉节省。
15. 不再用旧 commit、失败 suite、重复 job 或缺失 checkpoint 的运行填论文表格。
16. 不再把 smoke、precheck、toy wrapper、geometry-only 指标称为主实验。
17. 不再跳过 exact-uniform/random/dense 等同提交基线后继续扩新方法。
18. 不再只看 Avg-mAP；必须看 mAP@0.6/0.7、短动作和边界误差。
19. 不再只报模型 FLOPs；必须报告完整数据和系统通路的 p50/p95、显存、energy。
20. 不允许 validation/test GT、teacher、oracle、raw prediction cache 或外部隐式 JSONL
    参与主方法选择。

## 决策纪律

21. 讨论提出的 CVCR/BCFT/CoDeTAD/physical-grid/CFPA 不等于已经实现或更优。
22. 决定性实验未完成前，不宣布 DUCA 成功；同样也不宣布其必然失败。
23. 每次部署前必须记录 commit、配置、checkpoint、数据、Job ID 和 run root。
24. 新结果必须先更新 experiment/claim 节点，再改论文叙事。
