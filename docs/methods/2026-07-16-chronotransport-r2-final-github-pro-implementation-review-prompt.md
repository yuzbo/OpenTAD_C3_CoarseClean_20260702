# ChronoTransport CT-P3R-3S-r2 最终实现 GitHub Pro 严审 Prompt

请以零信任、fail-closed、只读方式，对 GitHub 上的 ChronoTransport CT-P3R-3S-r2 最新实现做
完整逐行审计。你的任务不是认可项目叙述，也不是设计另一套方法，而是判断这一个不可变代码
快照是否已经具备创建正式 registration 的实现完整性。不得修改仓库、创建 commit/PR、启动
训练、CUDA/Slurm 作业、Gate 或生成任何可被误认为实验结果的 artifact。

## 1. 角色与唯一问题

你同时扮演以下角色：

- 时序动作定位、VideoMAE、ActionFormer/AdaTAD 实现专家；
- PyTorch 训练事务、CUDA profiling、统计 bootstrap 专家；
- 文件系统/TOCTOU、Git provenance、Slurm 运行完整性红队；
- 严厉的科研完整性审稿人。

唯一实现裁决问题是：

> 这个精确 GitHub 快照能否被指定为 implementation commit `I`，从而允许随后只新增一份
> canonical registration artifact、形成 single-parent registration commit `R`？

它不是在问方法是否有效。当前不存在正式 Gate、训练或论文数字；禁止把单元测试、合成数据或
项目自报现象解释为科学结论。

## 2. 快照门：完整对象优先，严格等价证书作为受限 fallback

仓库与分支：

- repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- branch: `codex/chronotransport-r2-implementation`
- approved-spec ancestor: `537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`
- previous reviewed snapshot: `1b6366d0acb712e8096c2cceb0f05e66b16d30d4`
- implementation floor: `6c3606cc5161d415909a42741b3bc402278bf332`
- previous metadata-blocked review snapshot: `92a18bec2f5f247446083a8eb50fe889f367c23e`

必须 fresh-resolve 分支 HEAD 为 `REVIEW_SHA`，随后所有文件、diff、blame、tree、raw URL 和引用
只能固定到该 SHA；不得继续读取可移动 branch URL。`REVIEW_SHA` 必须是上述四个 anchors 的严格
后代且不等于任一个。

### Route A：完整 Git Data commit object（优先）

如果接口暴露完整 Git Data commit object，必须独立取得并报告：

- `REVIEW_SHA`；
- commit message、author/committer timestamp；
- 完整 `parents[]` vector；
- exact tree SHA；
- GitHub compare 证据，证明 `REVIEW_SHA` 是上述四个 commit 的严格后代；
- `git diff 6c3606c...REVIEW_SHA` 的 exact path list，确认 implementation floor 后只包含审计/研究
  记忆文档，不偷偷改动 production、test、config、launcher 或 registration 逻辑。

Route A 中任何字段实际存在但值不匹配时，不能转入 fallback，必须 fail closed。

### Route B：接口不暴露完整对象时的严格等价证书

仅当 reviewer 明确报告其可用 GitHub 接口不暴露 `tree.sha`、完整 `parents[]` 或分离的 Git-object
timestamps 时，才允许 Route B。不得因为字段值不一致、请求失败、权限不明或偷懒而使用 fallback。
Route B 必须独立完成全部条件：

1. fresh-resolve 移动分支一次并冻结 exact `REVIEW_SHA`；此后禁止读取 branch ref；
2. 分别 compare `537f692`、`1b6366d`、`6c3606c`、`92a18be` 到 `REVIEW_SHA`，每项都必须是
   `status=ahead`、`behind_by=0`、`ahead_by>0`，且 merge base 等于对应 anchor；
3. revision probes 必须证明 `REVIEW_SHA^1 = 92a18bec2f5f247446083a8eb50fe889f367c23e`，而
   `REVIEW_SHA^2` 不存在；不得由 `ahead_by` 推断 parent；
4. 独立枚举 `6c3606c...REVIEW_SHA` 的完整 changed-path list。每个 post-floor path 必须只属于
   `docs/methods/` 或 `research-wiki/` 的审计/研究记忆文档；任一 production、test、config、launcher、
   registration、Git submodule/LFS pointer 或其他路径变化都 fail closed；
5. 后续每个 mandatory file 必须通过显式传入 `ref=REVIEW_SHA` 或等价 SHA-pinned endpoint 读取。
   在 line-coverage ledger 中记录 path、固定 ref、读取是否成功和内容校验方式；所有本文给出 expected
   SHA-256 的文件仍必须计算并精确匹配。若 endpoint 暴露 blob/object SHA，一并报告；若不暴露，
   只要内容确由 SHA-pinned endpoint 返回且被完整审计，该字段缺失本身不阻断；
6. 明确把不可得的 tree SHA/双时间戳标为
   `UNAVAILABLE_NONBLOCKING_AFTER_EQUIVALENT_CERTIFICATE`，不得抄录项目文档中的值或自行推导。

Route B 是内容寻址的独立证书，不是信任项目自报 metadata。缺少上述任一条件、任一后续读取退回
移动 branch、或任一 exact content/hash 不一致时，唯一允许的输出是：

```text
GITHUB_SNAPSHOT_INCOMPLETE
```

并列出已解析 SHA、尝试的 route、已验证项与第一项失败条件；此时不得进入代码裁决。Route A 或
Route B 完整通过后必须继续规范与代码审计，不得仅因 tree SHA/双时间戳接口不可得再次停止。

## 3. 规范与历史证据门

以 exact `REVIEW_SHA` 读取并校验：

- `docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md`
- expected SHA-256:
  `E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`
- `research-wiki/sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-verbatim.txt`
- expected SHA-256:
  `C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`
- `research-wiki/sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-absorption.md`
- `research-wiki/sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be.md`
- expected SHA-256:
  `990E84F1D09116257D684090163BACB3F579ACA7290BADCB4D9FC6CFDA151FD1`
- `research-wiki/sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be-absorption.md`
- expected SHA-256:
  `EB7A5767C7274C5F22F3625FC993853C44D39E0F20B9E12C71A4C9222EE078B1`

上一轮 Pro 已对 exact A1--A4 返回 `APPROVE_SPEC_FOR_PLAN`，但对旧实现返回
`REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。你必须确认规范字节未变，并把上一轮未闭环 finding
逐项映射到当前代码。不得把规范批准自动转移为实现批准，也不得把本轮用户再次上传的同字节
附件计作第二份独立审计。

如果规范文件缺失或 hash 不符，也只输出 `GITHUB_SNAPSHOT_INCOMPLETE`。

## 4. 零假设证据纪律

对每一项事实明确标为以下之一：

1. `REPOSITORY_FACT`：可从 exact `REVIEW_SHA` 逐字节证明；
2. `REVIEWER_EXECUTED`：你实际执行并给出完整命令、环境和输出摘要；
3. `PROJECT_REPORTED_NOT_INDEPENDENTLY_VERIFIED`：仅项目记录声称；
4. `REVIEWER_INFERENCE`：从代码推断，并写清前提和不确定性。

不得声称执行你没有执行的测试。不得假设数据集、checkpoint、CUDA、Slurm、环境变量、隐藏
artifact、Git LFS object 或本地附件存在。不得用 tests 中的 fixture 弥补 production 缺失，也不得
因为测试数量多而降低审计强度。

项目当前报告但必须重新验证、不能直接采信的现象包括：

- 非 I/R implementation floor 是 `6c3606cc5161d415909a42741b3bc402278bf332`；
- source classification 共 65 paths：49 `REQUIRED`、13 `OUT_OF_SCOPE`、3
  `TEST_ONLY_NON_FORMAL`；25 个 matching tests 中 22 个进入 required vector；
- 18 个 changed/new implementation files 在本地与远端 CPU checkout 的 SHA-256 一致；
- targeted Gate 4 为 32/32；全 ChronoTransport CPU matrix 为
  `441 passed, 1 skipped, 2 warnings in 968.62s`；C3 compatibility 为 20/20；
- protected skip 是尚未执行的 CUDA-only surface；
- 没有 I、R、PRECHECK、Job ID、Gate report、训练结果、mAP、latency 或论文数字。

## 5. 必读范围：不能抽样

先读取：

- `AGENTS.md`、`RTK.md`；
- `research-wiki/query_pack.md`、`research-wiki/anti_repetition.md`；
- 上述规范、历史 review 与 absorption；
- `research-wiki/ideas/chronotransport.md`；
- `research-wiki/experiments/chronotransport-r2-execution-tracker.md`；
- `research-wiki/experiments/chronotransport-r2-implementation-verification.md`；
- `opentad/models/chronotransport/source_classification.json`；
- `opentad/models/chronotransport/source_inventory.py`；
- `opentad/models/chronotransport/registration.py`。

然后从 exact tree 枚举并逐行读取：

- `opentad/models/chronotransport/**` 全部 tracked source；
- 所有 ChronoTransport r2 config/base config 及其继承链；
- classification 中每个 `REQUIRED` production/tool/script/test；
- classification 中每个 `OUT_OF_SCOPE` 与 `TEST_ONLY_NON_FORMAL` 文件，确认它们无法被正式入口、
  import resolution、config、registration 或 launcher 可达；
- 所有文件名、import、entrypoint、registration source vector 或 Git diff 表明与本路线相关、但未被
  classification 枚举的 tracked file。

至少显式逐行覆盖以下新增/关键表面：

- `opentad/models/chronotransport/formal_stage_c.py`
- `opentad/models/chronotransport/post_stage_c.py`
- `opentad/models/chronotransport/formal_gate4.py`
- `opentad/models/chronotransport/gate4.py`
- `opentad/models/chronotransport/gates23.py`
- `opentad/models/chronotransport/profiler.py`
- `opentad/models/chronotransport/runtime.py`
- `tools/bata/chronotransport_r2_stage_c_factory.py`
- `tools/bata/train_chronotransport_r2_stage_c.py`
- `tools/bata/train_chronotransport_r2_matched_dense.py`
- `tools/bata/validate_chronotransport_r2_stage_c.py`
- `tools/bata/chronotransport_r2_post_stage_c_factory.py`
- `tools/bata/run_chronotransport_r2_post_stage_c_gate3.py`
- `tools/bata/validate_chronotransport_r2_post_stage_c_gate3.py`
- `tools/bata/build_chronotransport_r2_gate4_population.py`
- `tools/bata/chronotransport_r2_gate4_factory.py`
- `tools/bata/run_chronotransport_r2_gate4.py`
- `tools/bata/validate_chronotransport_r2_gate4.py`
- 四个 `scripts/run_chronotransport_r2_*_slurm_single_gpu.sh`
- `tests/test_chronotransport_r2_actionformer_per_window.py`
- `tests/test_chronotransport_r2_stage_c.py`
- `tests/test_chronotransport_r2_gates23.py`
- `tests/test_chronotransport_r2_gate4.py`
- `tests/test_chronotransport_r2_formal_gate4.py`
- `tests/test_chronotransport_r2_filesystem.py`
- `tests/test_chronotransport_r2_registration.py`
- `tests/test_chronotransport_r2_environment.py`

输出逐行覆盖清单：file、关键 line ranges/functions、审计主题；任何必读文件未读都不得 APPROVE。

## 6. 审计 A1/A2、registration 与源完整性

逐项证明或否定：

1. A1 唯一 unsuffixed `random_p2/p4/p8` seed 是整数 3407；manifest invocation order 被替换时，
   requested-action hashes 与 order digest 从 exact window IDs 重新生成，不能复用模板 hash。
2. registration generator 不读取任何 result/profile/replay/evaluation 路径；I 必须 clean；R 必须是 I
   的唯一单亲后继；`I..R` 只能新增一份 canonical registration artifact；in-memory object 与 R 中
   regular Git blob mode 的 exact bytes 相同。
3. required source set 与显式 classification exact closure；新增未分类 source/test 必须 fail closed；
   `OUT_OF_SCOPE`/test-only 表面无法 mint、rebuild 或 validate formal evidence。
4. 每个 public formal mint/report/unlock API 都在自身边界重新验证 clean detached R、source/import
   bytes、registration bytes、random locks、canonical output root，而不是依赖 launcher 先检查。
5. lexical path component 使用 no-follow/lstat，descriptor hash/read/decode 同源；Gate-4 media 在 decode
   前与 decode 后立即对同一个 retained descriptor 复核，拒绝 parent/leaf symlink、inode swap、
   storage replacement、concurrent no-clobber race 与中断后的不可能发布状态。
6. imported modules 的 origin 与 bytes 被绑定到 registered source；无法通过 `PYTHONPATH`、editable
   install、namespace package 或同名 module 执行未登记代码。
7. A2 registration 只冻结 required model/software，不预先冻结未来 UUID；每个 producer 从当前 PID
   重新观测 Slurm allocation、raw env、CUDA runtime 与完整 GPU UUID；仅允许一个可见 device 和
   process-local `cuda:0`；launcher 不覆盖 `CUDA_VISIBLE_DEVICES`，不固定物理 GPU index。
8. precheck 与实际 producer 使用同一个 config-override lock；被检查的 config、checkpoint、manifest、
   profile/replay、output roots 和实际执行对象逐字节一致。

给出针对 symlink parent、path swap during decode、descriptor/file mismatch、import shadowing、R dirty
tree、wrong parent vector、registration omission、unclassified addition、UUID spoof、visibility override、
precheck/producer config divergence 的 adversarial trace。

## 7. Gate 1、Stage B 与 Gates 2/3

### Gate 1

- 23 schedules、200 registered invocations/order 是否 exact；warmup 不能进入统计；D/C/S timing order 与
  raw observations 不得事后筛选。
- full-stack cost 是否真正包括注册定义的所有成本；profile、replay、result 是否从一次 repository-owned
  session 直接产生，而不是 caller raw rows 重建。
- oracle/evaluation-best static 的选择与 bootstrap 每 replicate 重选是否严格符合规范；Gate 1 FAIL
  是否原子锁死所有后续阶段。

### Stage B

- exactly 140 successful updates，而不是 attempts；overflow/retry 不推进 scheduler、optimizer、EMA、
  normalizer、RNG、ledger 或 exposure matrix。
- 三 seed、same batch/augmentation/LR/EMA 的 identity 是否可恢复；风险 predictor 只使用部署白名单；
  calibration/evaluation GT、teacher、raw-prediction cache、counterfactual ledger 不进入推理决策。

### Gates 2/3

- fit/calibration/evaluation split 与所有 artifact identity 不可混用；outer bootstrap unit 是 unique
  manifested window，seed 是全局 cluster draw；不能在每个 sampled window 内重新抽 seed。
- Gate-3 simultaneous marginal calibration、coverage/ranking/regret 的 exact formulas、strict inequalities、
  empty-selection handling、5000 replicates 与 seed 20260711 均与规范一致。
- post-Stage-C Gate 3 replay 只从合法 Stage-C checkpoints/artifacts 产生，unlock 无法由 caller payload、
  test-only schema 或旧 pre-Stage-C report 铸造。

## 8. Stage C 与 matched-dense：真实梯度和事务原子性

对 real registered ActionFormer/AnchorFreeHead 路径做数据流重建：

1. batch-two aggregate loss 与两个 per-window LD/LF/LR 来自同一 forward 的同一 logits、targets、
   reduction 与 `loss_normalizer` 语义；没有 dummy forward、外部 loss、第二次 hidden forward 或
   detach 后伪造 provenance。
2. runtime executed action 在唯一 canonical forward 后读回；仅在证明 batch size=1 后把 `[1,T,L]`
   规范化为 `[T,L]`；requested action、executed action、cost bytes 和 hashes 全部一致。
3. CT trainable ownership、risk predictor ownership、frozen VideoMAE/head/其他参数、optimizer param groups、
   gradients 与 EMA 与规范完全一致；`.data`、storage rebind、view/alias、module swap、ordinary Tensor
   attribute、persistent/nonpersistent buffer 都能被 success audit 与 overflow rollback 检测。
4. 每个 seed exactly 4200 successful updates；CT 与 matched dense 严格共享 batch、augmentation、LR、
   EMA、attempt order 与成功 update identity；overflow/retry 完整回滚 model/optimizer/scaler/RNG/sampler/
   normalizer/ledger，重试仍消费同一 batch。
5. checkpoint/resume 在任意 interruption point 产生与 uninterrupted run exact equivalent 的状态、
   ledger 和下一 batch；中断恢复不覆盖已有证据，不接受不可能的部分发布组合。
6. measured cost 不只是布尔 flag：exact registered profile/environment/producer/requested/executed bytes
   进入训练和证据；proxy/test-only cost 无法进入 formal runner。

必须检查真实 4200-success CLI、matched-dense CLI、validator、launcher 与 registration coverage，而不只
检查 dataclass/helper。列出会产生假梯度、假 shared-batch、假 update count、假 rollback 或假 checkpoint
的最短攻击路径，并确认相应 RED tests 是否真的击中 production boundary。

## 9. Gate 4：端到端速度、官方指标与 hard conditions

从 official full-video population builder 到 seed shard、finalizer、validator、terminal marker 全链路审计：

### Population 与 inference 决策

- official population 是规范定义的完整 full-video/sliding-window evaluation population；video/order、
  media/checkpoint/config bytes 固定；无缺失、重复、post-hoc filtering 或 test fixture 代替。
- dense、calibration-frozen best static、learned CT 使用冻结 identities；learned scheduler 只能消费部署
  白名单信号，不能访问 evaluation GT、teacher、replay ledger、raw-prediction cache 或未来窗口。
- overlapping-window predictions 按官方 pipeline 聚合并执行 full-video NMS；metric/regret 每个 unique
  official invocation 只计一次，timing repetitions 只能进入 latency block。

### Timing 与 heavy computation

- total latency 必须实际包含 decode、preprocess、H2D、model、postprocess 和 full-video NMS；六序列
  D/C/S counterbalanced order 正确执行；warmup 与 measured repetitions 分离。
- CUDA events 可延迟读取，但只在注册的 outer invocation synchronization 后 flush；任何 helper 中间
  `synchronize()`、隐式 CPU conversion 或改变 schedule 的 sync 都是 blocker。
- heavy computation 以真实 executed heavy path 计量；transport/hold 不能只改标签而仍执行 dense；
  cache hit、repair/fallback、action tensor、requested/executed cost 全部可审计。

### Official AP、regret 与 bootstrap

- AP mirror 必须与仓库官方 OpenTAD evaluator exact；特别检查 NumPy `argsort()[::-1]` quicksort 的
  equal-score ordering，禁止 stable-sort 替代、预汇总 per-video AP 或近似 evaluator。
- latency bootstrap：outer resample official video IDs，video 内 resample complete invocation blocks，
  每 arm 从 raw totals 重算 p50；不得 bootstrap stage percentiles 后相加。
- mAP bootstrap：outer resample official video IDs、inner resample three seeds；每个 sampled seed 在同一
  official-video multiset 上从 raw predictions/GT 重建，不能跨 seed 合并 predictions/NMS。
- detector-regret bootstrap 只使用 unique official invocations，按规范 official-video/seed hierarchy
  重建；5000 replicates、seed 20260711、percentile one-sided 95% bounds exact。

### 必须逐式核对的七项 hard conditions

1. latency saving 的 one-sided 95% LCB `>= 0.15`；
2. mAP@0.7 drop 的 one-sided 95% UCB `<= 1.5`；
3. shortest-Q1 drop `<= 1.5`；
4. 对每个 matched invocation：
   `heavy_saving_i = dense_heavy_i - selected_heavy_i`，
   `overhead_i = total_CT_i - selected_heavy_i`，
   `margin_i = 0.40 * heavy_saving_i - overhead_i`；full-sample median heavy saving `> 0` 且 median
   margin bootstrap one-sided 95% LCB `> 0`。禁止替换成 arm-level p50 差；
5. `p50_CT - p50_static` one-sided 95% UCB `<= 0`；
6. CT 相对 calibration-frozen static 的 detector-regret absolute improvement bootstrap 95% CI lower
   `> 0`；
7. 每个 seed 都必须分别通过 latency saving、mAP@0.7 drop、shortest-Q1 drop、median margin、
   CT-static 五个 point conditions；pooled PASS 不得隐藏 seed reversal。

确认 calibration-frozen static 在 evaluation/bootstrap 中固定；evaluation-selected diagnostic comparator
每 replicate 重选但不能替换 hard static comparator。

### Energy

energy 只能是 10-Hz NVML power 对长、完整 official-population timed block 的梯形积分；报告 requested
与 observed cadence、sample count、median/max gap、block boundary。禁止把单次 inference、稀疏采样或
估算值称为 energy，也不能用 energy 替代任一 primary condition。

## 10. Slurm、stop-chain 与无结果边界

- 所有 GPU 工作只能由 Slurm allocation/step 执行；登录节点不得训练；launcher 不覆盖 scheduler
  visibility；进程只使用 logical `cuda:0`。
- stop-chain 必须严格是 Gate1 → StageB → Gates2/3 → StageC+matched → post-Stage-C Gate3 → Gate4；
  上游 FAIL/invalid/missing artifact 原子锁死下游。
- `INVALID_IMPLEMENTATION`/provenance failure 与科学 Gate FAIL 必须区分；不可把未经授权或污染的 run
  解释成路线失败，也不可把 test PASS 解释成 Gate PASS。
- 当前审计不能创建 I/R 或运行 PRECHECK；即使 implementation APPROVE，也只允许下一步指定 exact
  REVIEW_SHA 为 I，再在全新 clean detached checkout 中创建唯一 registration-only child R。

## 11. 测试审计要求

不要只看 test 名称。对每个 blocking invariant 建立矩阵：规范条款 → production boundary → positive
test → adversarial RED test → residual untested risk。重点检查 mock/monkeypatch 是否绕过真实入口、
assert 是否只验证 self-consistent hash、fixture builder 是否与 formal builder 共用可重标记结构、以及
测试是否会在删除关键 production check 后仍通过。

如果可执行 CPU tests，先验证 exact source bytes/environment，再给完整命令与结果；CUDA/Slurm tests
只有在当前审计明确授权时才可执行，而本 prompt 没有该授权。不得为了填补 protected CUDA skip 而
启动 GPU 作业。

## 12. Finding 与具体实现合同

所有 finding 使用 P0/P1/P2/P3：

- 标题与 severity；
- exact SHA、file、line range、function/class；
- 规范条款；
- 最短可复现 attack/failure trace；
- 为什么现有 tests 未阻止；
- 必须新增的 RED test；
- 最小、可编译、可运行的 unified diff 或完整 replacement function；
- 影响哪些 artifact/Gate/claim，修复后必须重跑哪些命令。

对所有 registration-blocking P0/P1 必须给具体实现，不接受“加强校验”“增加测试”这类泛化建议。
建议不得引入 GT/teacher/replay leakage、改变已批准阈值/统计单位、降低 fail-closed 强度或扩大方法 claim。
如果建议本身超出已批准规范，明确标为“需要新 spec amendment，当前不得实施”。

## 13. 强制输出顺序

1. `Snapshot Certificate`：使用 Route A 或 B、SHA、可得的 parents/tree、全部 ancestry、
   implementation-floor diff path list、SHA-pinned content ledger；
2. `Evidence Classification`：repository/executed/reported/inference；
3. `Previous Pro Finding Closure Matrix`；
4. `Line-Coverage Ledger`；
5. `Executive Verdict`；
6. P0/P1 findings；
7. P2/P3 findings；
8. A1/A2/registration/source-integrity audit；
9. Gate1、StageB、Gates2/3 audit；
10. StageC/matched-dense audit；
11. Gate4/statistics/energy audit；
12. Slurm/stop-chain audit；
13. Test adequacy matrix；
14. Concrete implementation patches；
15. `Next Plan`，按依赖列出 exact commands、expected artifacts、stop conditions；
16. `Residual Unknowns`，尤其是只有合法 R-bound CUDA/Slurm PRECHECK 才能回答的事项。

## 14. 唯一允许的整体实现裁决

快照门通过后，整体实现裁决必须且只能逐字选择一个：

```text
APPROVE_IMPLEMENTATION_FOR_REGISTRATION
```

或

```text
REVISE_IMPLEMENTATION_BEFORE_REGISTRATION
```

只要存在任一 P0/P1、未读 mandatory file、source/classification 缺口、formal workflow 不可达、真实
ActionFormer loss/transaction 未闭环、Gate-4 official metric/timing/bootstrap 不等价、或无法证明代码
不会铸造假证据，就必须选择 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。

`APPROVE_IMPLEMENTATION_FOR_REGISTRATION` 的含义严格限于：exact `REVIEW_SHA` 可以被指定为 I。
它不创建 R，不批准 PRECHECK，不解锁任何正式实验，不证明 ChronoTransport 有效，也不授权论文 claim。
不得输出“条件批准”；有条件就 REVISE，并把条件写成 blocking findings。
