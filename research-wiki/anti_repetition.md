---
type: anti_repetition
updated: 2026-07-16
---

# 禁止重走清单

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

## ChronoTransport bounded appeal

25. 不得执行 `02199f8`；它已被 GitHub-visible Pro 复核裁决为
    `REVISE_SPEC_BEFORE_PLAN`。
26. 不得把 200 个 video IDs 直接当 200 个 windows；Gate 1--3 必须使用一视频一
    label-free hash-frozen window，Gate 4 才使用 official full-video/sliding-window
    population。
27. 不得使用 `(update+offset)%16` 与 offsets 0/4/8；必须使用冻结的 `+5*b`
    block rotation 和完整 exposure-matrix hash。
28. 不得保留 oracle-minimum 对 shuffled assignment 的 hard Gate；它近乎由定义保证。
29. Stage-C loss ownership 必须由 object-identity sets 与独立 `autograd.grad` 实现，
    不能用 total-loss backward、name substring 或 detach adapter 输出来冒充。
30. AMP retry 必须恢复除 GradScaler backoff 外的全部 forward-mutated state；匹配的是
    successful batch/LR/EMA exposure，不是两个 arms 的 overflow vector。
31. Gate 3 的 selected coverage 不是 selected-conditional conformal guarantee；不得把
    frozen-window guarantee 转移到 Gate 4 official population。
32. formal profile/replay 前必须完成 immutable registration commit；任何 repair、fallback、
    identity mismatch 或 retry violation 都是 `INVALID_IMPLEMENTATION`，不是 science FAIL。
33. 即使 Gate 1--4 全 PASS，`deploy=false`、`paper=false` 仍保持冻结。
34. r2 resolved config 必须只在 inner
    `model.backbone.backbone.chronotransport` 生效；wrapper-level overlay 或 inner legacy age
    残留都不得生成 profile/Gate artifact。
35. Gate-3 simultaneous conformal 不得展平 `window×candidate`；必须先对每个窗口的 16 个
    candidates 取 residual 最大值，再对 30 个 window maxima 取 rank 28。
36. legacy six-schedule/old-split runner、pooled Spearman、row bootstrap 和 GT-aware
    `random_trunc` 不得从 r2 formal launcher 可达。
37. 不得把 `sandbox:/mnt/data/...` 的外部 patch proposal 当作本地可用、已应用或已测试；
    其明确状态是 unavailable + `NOT_EXECUTED_BY_REVIEWER`。
38. profiler 不得用 `count=0,p50=0,p95=0` 占位通过完整性检查；formal cost 必须来自直接
    测量的 invocation-level `total_ms`，且 exact requested cost 与 executed diagnostic cost
    分开记录。
39. 项目报告的 110 tests 只证明已覆盖的 primitive/subset；不得据此声称 formal manifest、
    Stage B/C、Gate 3/4、transactional retry、registration 或 science 已闭环。
40. 第二次独立审查返回 `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` 前，不得创建 I/R 或打开
    formal profile/replay/evaluation 数据。
41. 不得把 payload/schema 校验冒充 formal repository-context 校验；正式 CLI 必须验证 clean
    detached `R`、`R^=I`、R-only registration diff 与当前 source bytes/hash。
42. Stage-B training checkpoint 写成不等于整个 Stage-B phase 完成；没有同时绑定 checkpoint、
    ledger、140x16 rank-127 baseline 与 predictor identity 的原子 completion marker 时，
    calibration/evaluation 必须拒绝。
43. AMP overflow 回滚不得对未改变的 Parameters 无条件 `load_state_dict`；即使 tensor bytes
    相同也会改变 version counter，并可能使恢复的 non-leaf graph 无法再次求导。
44. 不得信任 caller 自报的 base LR、`transport_executed`、action identity 或 success callback；
    必须从真实 scheduler、actual action tensor 与被注册状态对象的 pre/post delta 推导。
45. Stage-C production ownership 不得保留隐式 legacy topology fallback；旧测试 fixture 必须升级
    或使用与正式入口隔离的 test-only helper。
46. `persistent=False` buffers 仍是 forward-mutated state；不能因不在 `state_dict()` 中而漏掉
    retry snapshot/restore。
47. Gate-1 control 不得只登记 action hash；motion/random 的 actual action bytes 必须由冻结算法、
    deploy-visible signal/window identity/seed 重新生成并逐窗口复核。
48. Slurm 单 GPU cgroup 会把物理 GPU 映射为 local ordinal 0；不得再用
    `CUDA_VISIBLE_DEVICES=1` 证明物理 GPU1，必须同时验证 Slurm physical GPU ID 与 local visibility。
49. formal profile 的生产 API 不得接收 caller-supplied backend；toy backend 只能存在于 test-only
    helper，正式 artifact 必须绑定固定仓库 backend 的源码身份。
50. Gate-1 regret 不得作为 caller 字段进入 record builder；必须由同一 materialized batch/RNG 的
    dense 与 candidate detector loss 在固定 paired runner 内部计算。
51. 200 个媒体文件的 registry/hash 校验必须在所有 candidate warmup/timer 之前完成；不得让第一
    candidate 独自承担校验 I/O 并污染 B* 成本比较。
52. Stage-B 不得接受简化的 PASS JSON；unlock 必须是 Gate-1 adjudicator 生成并由共享 exact
    schema 复验的完整证据 artifact。
53. registration R 必须是 I 的唯一单亲后继，且 I..R 精确只新增 registration 文件；检查第一父
    和 path name 不足以证明 R-only。
54. Stage-C success 不仅要检查各对象局部 `+1/+2`，还必须检查 scheduler/EMA/sampler/cursor/
    exposure/ledger 的共同起点与完整 trace coherence。
55. Formal profile/replay validator 不得从 raw test rows 重建并接受正式 schema；test-only artifact
    必须使用不相交 schema，不能生成 Gate evidence。
56. 即使返回 typed runner result，只要 detector、data batches、motion source 或 runtime action
    identity 仍由 caller 提供，就不能称 repository-owned evidence。
57. 每个可生成 formal Gate report/unlock 的 API 都必须独立验证 clean detached `R` 和全部未决
    protocol locks；仅在 launcher 检查不够。
58. Stage-C loss 必须绑定真实 forward 后从唯一 canonical runtime 读回的 executed action tensor；
    caller 提供 expected action payload 不构成执行证据。
63. Gate-4 纯函数能从 raw dict 重算统计并不等于 formal evidence；正式入口还必须绑定 frozen
    invocation/order、calibration-frozen static、checkpoint/cache/metric provenance、clean detached
    `R` 与独立 full-stack profiler。不得把 synthetic 12/12 称为 Gate-4 实验通过。
64. Stage-C 的“真实 forward 发生过”不足以证明 loss 来自它；LD/LF/LR 必须以可审计方式绑定
    同一个 runtime forward graph/outputs，不能用 dummy forward 加外部直连 loss 通过。
65. frozen parameter 不能只检查 object/metadata/version；`.data` 写入可能不增加 version，必须
    用 bitwise bytes/hash 或同等强度的不可变性证据覆盖 heavy/head/其他 frozen 参数。
66. runtime summary 的安全字段缺失不是 `false`；正式 Stage-C evidence 必须 exact-key/type/value
    fail closed，并绑定 production `ChronoTransportRuntime`/source identity，duck typing 不足。
67. 测试通过后只要 production 或 test 文件 SHA 又变化，旧 GREEN 立即失效；本地 py_compile
    不能替代最终 exact bytes 的远端行为复跑。
68. Python 的下划线、module-private token 或 issuer 不是安全边界；只要外部可 import/call 并注入
    detector、batch、raw rows，就不能用于铸造 formal evidence。
69. 每个 public formal report/unlock/mint API 必须自行验证 clean detached `R`、random lock，并把
    in-memory registration 绑定到 `R:path` 的 exact canonical bytes 与 regular Git blob mode；
    不能依赖某个 launcher 先检查。
70. 三 seed hierarchical bootstrap 必须按规格把 seed 作为全局 cluster draw；不能在每个 sampled
    window/video 内重新抽 seed，否则会制造伪独立、收窄 CI，甚至翻转 PASS/FAIL。
71. formal result/terminal 的“不存在检查 + `os.replace`/`mv`”不是并发安全；必须有独占 run lock
    与原子 no-clobber，两个同时通过 precheck 的进程不能互相覆盖正式证据。
72. Gate-4 bootstrap mAP 必须调用仓库官方 OpenTAD evaluator 重建；自写 AP 即使公式近似相同，
    equal-score 排序等细节仍会改变结果，不能作为 official metric evidence。
73. Stage-C detector forward 后必须冻结 deploy-visible `latest_signals` 的对象与逻辑字节直到
    risk forward消费；仅验证risk读取“当前属性”会允许中途替换或`.data`改写伪signals。
74. success path与overflow path都必须检查所有registered buffers的identity/metadata/逻辑字节；
    只在overflow比较buffer会让success中的`.data`静默污染永久保留。
75. Stage-C loss provenance不能只保存forward output的Tensor引用；引用本身可被`.data`改值而继续
    通过VJP。必须同时冻结forward-boundary的metadata与逻辑值，并在LD/LF/LR审计前重新核对。
76. test-only 与 formal 不得共享可重标记的 candidate/row representation 和通用 artifact rebuild；
    只把顶层 schema 改名不足以隔离。正式 builder 必须从固定执行会话直接产生不可由 fixture
    builder 复用的证据结构，validator 不能充当 raw-row-to-formal 转换器。
77. formal source validator 自己要求的文件必须全部进入 immutable registration required-source
    set；运行时再声明额外路径而 registration exact-set 又拒绝 extra，会形成永远不可执行的协议。
78. 只拒绝 leaf symlink 不够；formal input/output 的每个已有父路径组件都必须拒绝 symlink，且
    result/terminal 必须独占锁 + 原子 no-clobber，不能用 existence-check 后 `os.replace`。
79. Stage-C 的普通 Python Tensor 属性也属于可变模型状态；仅保存对象引用、version、shape、
    stride和值无法发现等值 storage rebind。success audit 与 overflow rollback 都必须绑定并恢复
    storage identity/offset/size/layout，否则 view/alias 关系可被静默拆开。
80. registration 的“全部正式 source/test/config/launcher”包含最终 hardening tests；不能只登记
    production 与功能测试而漏掉独立冻结所依据的边界测试，否则 exact source set 无法代表被批准
    的完整实现向量。
81. formal factory 不得读取已被 canonical registration 替代的 flat digest alias；manifest、
    exposure、checkpoint 等输入必须从 validated nested artifact 取 identity，并有真实入口回归。
82. Stage-C success buffer audit 不能一边要求 train-mode detector 语义、一边禁止
    `loss_normalizer` 的规范成功更新；module mode、允许的成功态 buffer 变化及 matched-dense 对齐
    必须先在规格中唯一化，不能由 runner 静默选择。
83. 当前 governing Slurm 规则禁止固定物理 GPU index/覆盖 scheduler visibility；不得继续执行旧
    `CUDA_VISIBLE_DEVICES=1` launcher 或把 local ordinal 当 physical identity。协议文字与 claims
    必须先修订、独立复核，再生成 I/R。
84. formal output/lock path 不得先 `.resolve()` 再检查 symlink；这会把指向合法 canonical root 的
    alias 洗白。必须对 lexical components 逐级 `lstat`，final lock 使用 `O_NOFOLLOW`，cleanup 只可
    删除与已打开 descriptor 的 device/inode 相同的 lock，不能删除同名替换文件。
59. Transactional topology 必须绑定有序 `path -> object/type` 与完整 alias graph；无序 Parameter
    identity set 无法发现同构 ViT blocks 被交换。
60. core primitives 通过不等于完整实现；formal Gates-2/3、Stage-C/matched-dense runners、Gate-4、
    launchers 及 validators/tests 全部存在前，不得标记 `implemented`。
61. Gate-4 mAP bootstrap 必须对每个 resampled seed 在同一 official-video multiset 上从 raw
    predictions/GT 重建；不得跨 seed 合并 predictions 或 NMS，也不得接受预汇总 mAP 代替。
62. timing repetitions 只能进入 latency block bootstrap；metric 与 detector-regret 每个 unique
    official invocation 只计一次，不能把重复计时伪装成额外统计样本。
85. A formal precheck must not canonicalize caller paths with `resolve()` before rejecting aliases.
    Validate every existing lexical component with `lstat`, require mandatory files without a
    missing-tail exception, and reconstruct the registered R-derived output independently before
    comparing any convenience resolver result.
86. No-clobber alone is not interruption recovery. A multi-artifact formal phase may reuse an existing
    object only after exact-byte recomputation and regular-inode verification; checkpoint bytes used for
    hashing and deserialization must come from the same `O_NOFOLLOW` descriptor. Validate state again
    inside the writer lock, and reject publication states that cannot arise from the registered order.
87. A pure Gate-4 statistic function over caller-supplied timing/metric/regret mappings is test-only,
    even when it recomputes official AP semantics. It must emit an explicitly test-only schema and must
    refuse `formal=True` until a repository-owned producer binds official population/order, checkpoints,
    post-Stage-C Gate-3 unlock, frozen static identity, live profiling and clean detached registration R.
88. `cost_is_measured=True` is a necessary runtime invariant, not cost provenance. Formal Stage C must
    reject proxy-cost summaries, but registration/runner still has to bind the exact profile artifact,
    environment, producer identity and requested/executed cost bytes; a test-only cost table cannot satisfy
    that obligation.
89. `GITHUB_SNAPSHOT_INCOMPLETE` only proves that the external reviewer obeyed the snapshot gate. When
    the branch is still the forbidden old SHA, missing-file/spec-hash/code checks are `NOT_EVALUATED`;
    never cite that response as an implementation verdict or silently audit the old snapshot instead.
90. An external review verdict and its proposed patch are different evidence classes. Accepting
    `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` does not authorize copying an interface, launcher name or
    discovery rule that is not uniquely implied by the approved specification.
91. Do not define the formal registration test vector with a raw filename glob. Every ChronoTransport test
    must be explicitly classified; newly discovered unclassified tests fail closed, while only approved
    `formal_r2_source` tests enter the immutable exact-source vector.
92. A proposed Stage-C evidence dataclass is not an executable contract. Official loss-dictionary
    reduction, per-window regret, dense/counterfactual forward count and order, and `loss_normalizer`
    transitions must follow approved A3/A4 before any production runner is implemented.
93. Do not preserve `_gpu1.sh`, `CUDA_VISIBLE_DEVICES=1` or physical-index semantics merely because old
    spec text or an audit names that path. Future launchers must use the approved Slurm-assigned-device
    contract and logical `cuda:0` without overriding scheduler visibility.
94. Before a valid I/R, GPU checks must remain synthetic/test-only. Do not touch the official evaluation
    population or create reusable Gate-4 evidence under the label of a “small precheck.”
95. A spec-only descendant is not an implementation repair. After `537f692`, reviewers must first judge
    the exact A1--A4 spec diff, then independently prove production compliance; neither the historical
    `b854adb` verdict nor the new specification may be auto-transferred into a current code verdict.
96. Do not execute the historical ChronoTransport implementation plan unchanged. It still binds
    `e4422f5`, physical GPU1, `CUDA_VISIBLE_DEVICES=1` and `_gpu1.sh`; replace it with an
    `APPROVE_SPEC_FOR_PLAN`-aware plan before implementation resumes.
97. Do not treat copied `CHRONOTRANSPORT_OBSERVED_*`, caller JSON or inherited environment variables as
    Slurm/GPU evidence. Every formal producer must re-observe scheduler, OS, torch/NVML device UUID and
    software state, then bind those bytes to the artifact under the writer lock.
98. An unauthorized formal job or broken provenance makes that run `INVALID_IMPLEMENTATION`; stop,
    quarantine, repair, re-review and rerun. Do not automatically rewrite it as a scientific Gate FAIL or
    permanent route kill unless the frozen protocol was intentionally/post-hoc changed, contamination is
    unrecoverable, or a formal science Gate actually fails.
99. Do not freeze a reviewer-proposed Stage-C dataclass, exact loss-key set or detector feature field as
    authority. Derive the per-window vector, aggregate identity equation and legal auxiliary-loss namespace
    from the registered ActionFormer config/head while preserving A3/A4 forward and normalizer semantics.
100. Same-descriptor hashing/loading of data is necessary but does not by itself prove Python executed the
    same source bytes. Formal source integrity must also bind detached worktree, import resolution, loaded
    module origins/bytes and entrypoint identity to the registered source vector.
101. When a registration context builder replaces the manifest invocation order, it must regenerate every
    seed-3407 random requested-action hash and its order digest from those exact window IDs. Reusing template
    hashes from another manifest is an invalid implementation even if candidate names and counts match.
102. Registration must not freeze a future GPU UUID or confuse a Slurm physical identifier with local
    ordinal zero. Freeze required model/software only; each producer must bind the current process PID to a
    full observed UUID plus raw allocation fields, while `cuda:0` remains only the process-local execution
    address. A hash-only artifact without recoverable allocation bytes is insufficient.
103. A batch-one ActionFormer action tensor may be normalized from `[1, T, L]` to `[T, L]` only after the
    producer proves the leading dimension is exactly one. Generic squeezing can silently alter action
    identity and invalidate the registered requested/executed-action hashes.
104. Deferred CUDA-event stage timing must be flushed only after the registered outer invocation boundary
    synchronization. A helper that synchronizes mid-forward changes the measured schedule and can manufacture
    a false transport overhead or speedup even when its arithmetic is correct.
105. Gate-4 energy evidence is a 10-Hz NVML trace over a long, complete official-population timed block.
    Record requested/observed cadence and median/max sampling gaps; never relabel a sparse or
    single-inference estimate as block energy, and never use energy to replace a primary Gate condition.
106. The official AP mirror must preserve the repository evaluator's exact equal-score ordering, including
    NumPy `argsort()[::-1]` quicksort behavior. Stable sorting, pre-aggregated per-video AP, or an approximate
    evaluator can flip a bootstrap replicate and is not official metric evidence.
107. Gate-4 margin is defined per matched invocation as
    `margin_i = 0.40 * (dense_heavy_i - selected_heavy_i) - overhead_i`. Do not replace it with a difference
    of arm-level p50s. The registered per-seed margin and CT-versus-static conditions remain hard conditions;
    pooled success cannot hide a reversing seed.
108. Formal precheck and producer must call the same config-override lock and compare the same canonical
    bytes. Duplicated parsers or one-sided validation permit a checked configuration to differ from the one
    actually executed.
109. Media provenance must be checked through the retained no-follow descriptor both before decode and
    immediately after decode. A path-only recheck can miss a swap during decoding and cannot prove that the
    measured invocation consumed the registered bytes.
110. Remote regression evidence is transferable to a local review candidate only after every changed/new
    implementation file is compared by exact SHA-256. A passing test in a merely similar remote checkout is
    not evidence for the candidate later pushed to GitHub.
