---
type: query_pack
updated: 2026-07-16
max_chars: 8000
---

# DUCA Query Pack

## 2026-07-16 ChronoTransport integrity delta

- Superseding every older ChronoTransport implementation-status bullet below: non-I/R candidate
  `6c3606cc5161d415909a42741b3bc402278bf332` implements W0--W6, including A1/A2, descriptor/import
  integrity, real ActionFormer per-window loss, paired 4,200-update CT/matched-dense Stage C,
  post-Stage-C Gate 3 and repository-owned official-population Gate 4. The explicit classification covers
  65 tracked test/tool/script paths: 49 `REQUIRED`, 13 `OUT_OF_SCOPE`, three
  `TEST_ONLY_NON_FORMAL`; all 25 matching tests are classified and 22 enter the required vector. Eighteen
  changed/new files were SHA-256-identical between local and remote verification trees. The final remote
  CPU matrix passed `441 passed, 1 skipped, 2 warnings in 968.62s`; r2 launcher syntax, remote py-compile
  and the required C3 compatibility pair (`20 passed`) also passed. This is implementation evidence only.
  W7 exact-byte independent Pro review and R-bound CUDA/Slurm precheck remain open; registration is
  `NOT_READY`, E0--E5 remain `LOCKED`, and no ChronoTransport Job, Gate result, training result or
  paper-usable number exists. Local pytest is not evidence because the Windows PyTorch import fails with
  `WinError 1114`.
- The first Pro call against review snapshot `92a18bec2f5f247446083a8eb50fe889f367c23e` returned only
  `GITHUB_SNAPSHOT_INCOMPLETE`. It independently established `^1=6c3606c`, no `^2`, ahead 1/behind 0 and
  the exact eight-path docs-only post-floor diff, but its normalized GitHub interface did not expose tree SHA
  or separate Git-object timestamps required by the prompt. No spec hash, code, test or registration verdict
  was evaluated. This is a snapshot-contract/tool mismatch, not implementation approval or rejection; W7
  remains open and all experiment stages remain locked.
- The user approved a replacement snapshot gate that keeps the full Git Data object preferred but permits a
  strict fallback only after frozen SHA, exact `^1/^2`, four-anchor ancestry, complete post-floor docs-only diff
  and SHA-pinned mandatory-file reads all pass. This is a reviewer-tool compatibility repair, not a reduction
  in code/content integrity and not an implementation verdict.
- The resulting unified prompt exceeded the Pro model's one-turn thinking duration. It is superseded as a
  direct input by two same-SHA prompts: Part 1 covers snapshot/A1/A2/registration/Gate1/StageB/Gates2-3 and
  emits a complete handoff packet; Part 2 requires that verbatim output, covers StageC/post-Gate3/Gate4/Slurm,
  closes the union coverage ledger and alone issues the overall verdict. Missing packet, changed SHA or dropped
  Part-1 blocker fails closed. No implementation or experiment status changes.
- Gate-1 precheck path hardening is `tested_and_bounded_code_approved`: focused 1/1,
  registration/precheck 38 passed/1 protected xfail, Gate-1 hardening/cost 25/25, and independent
  verdict `APPROVE_GATE1_PRECHECK_PATH_HARDENING` for exact SHAs `0BE0EA8B...F76808` and
  `55916FBD...C10BA`.
- This did not run Gate 1, create I/R, approve A1--A4, launch training, or generate scientific
  evidence. Registration remains `NOT_READY`.
- Stage-B partial-publication recovery is also `tested_and_bounded_code_approved` after one independently
  rejected candidate: final exact SHAs `50F4469D...F4F84`, `47342FFE...A7670`, `9BB46DE2...E378D`;
  final remote matrices 98 passed/1 protected xfail and 44/44; verdict
  `APPROVE_STAGEB_PARTIAL_PUBLICATION_RECOVERY`. This is implementation evidence only.
- Latest read-only Slurm refresh found only unrelated DUCA/P0/S1 jobs; no ChronoTransport job exists
  to monitor or reuse, and none was launched.
- 首次 GitHub-only Pro 调用正确返回 `GITHUB_SNAPSHOT_INCOMPLETE`：公开分支解析为
  `797a2df8d00560c8f7a7f66c13e95bb5b0d836ee`，与被禁止的旧快照 identical。reviewer 按第一门
  停止，未读取代码、未检查文件/spec hash，也未给实现裁决。下一次 Pro 审查必须等用户明确授权
  发布新的 immutable review snapshot；该 snapshot 不能冒充 I 或 R。
- 用户授权的 review-only snapshot `b854adb4f4c9235580b5e58c3f3255db6e9adbc0` 随后通过 GitHub
  快照门，但 Pro 完整审计仍裁决 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。总体裁决与停止
  条件已吸收：Stage-C/matched-dense/Gate-4 正式链缺失，真实 ActionFormer loss/normalizer 合同未
  接通，A1--A4 未批准，registration 漏掉两份本次修改的 integration tests，measured-cost provenance
  与 Slurm 合同仍未闭环。没有测试复跑、I/R、formal Gate、训练或论文数字。
- Pro 的具体补丁不全部照搬：不能用裸 `tests/test_chronotransport*.py` glob 定义 formal vector；
  当前 glob 命中 21 个文件、registration 仅含 14 个，除两份已确认遗漏外还有五份旧/通用表面。
  应采用显式 formal/legacy/nonformal 分类并拒绝未分类新文件。其 Stage-C evidence dataclass 只是
  `b854adb` 时 A3/A4 尚未唯一化的设计草案；未来 launcher 也不得沿用固定 physical-GPU1 语义。
- 第二次 GitHub-only Pro 已完整审核 review-only SHA
  `1b6366d0acb712e8096c2cceb0f05e66b16d30d4`，确认 tree `3fc64c7...5462`、唯一 parent
  `537f692...1d37` 与当前 spec SHA
  `E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`。A1--A4 得到
  `APPROVE_SPEC_FOR_PLAN`，故规范状态迁移为 `spec_approved`；整体实现仍为
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`，registration `NOT_READY`。reviewer 未执行 tests、
  CUDA、Slurm、训练、profiling 或 evaluation；没有正式实验事实。
- 当前 P0 均经源码复核：registration 仍绑定旧 `e4422f5`/旧 hash 且 random lock 使 A1 不可达；
  Gate-1 launcher/backend 仍固定 GPU1，A2 不可达；Stage-C 的单 Tensor/单 forward/成功 buffer
  不变合同与真实 ActionFormer dict、`loss_normalizer` 及 A3/A4 冲突；Stage-C、matched-dense、
  Gate-4 formal workflows 缺失。禁止 I/R、PRECHECK 和正式 Job。
- 不完全照搬 reviewer 补丁：A2 还缺现场重观测的 Slurm/GPU/software artifact schema，不能信任
  caller 环境变量；Stage-C dataclass/loss 字段集不是规范冻结接口；21 个 matching tests 必须逐一
  formal/legacy/nonformal 分类；无合法 I/R 或 provenance 缺陷先使 run
  `INVALID_IMPLEMENTATION`，不自动等同 science route kill。
- reviewer bounded patch 还未覆盖 full-stack profiler CLI、Stage-B CLI、Gate-1 profile backend、
  Gates-2/3 `latency_gpu1_fixed_stack` claim 与旧 implementation plan。该计划仍绑定旧 spec/GPU1，
  已标记 stale，禁止原样执行。先替换计划，再依次闭合 A1/source classification、A2、filesystem/
  runtime identity、真实 per-window ActionFormer API、Stage-C transaction 与完整 workflows，最后重新
  exact-byte implementation review。

## 项目方向

任务是离线 TAD 的高效时序计算，不是流式/因果 Online TAD。最终目标是任务感知
动态计算分配：在昂贵 backbone 前或内部去除时序冗余，同时保护 mAP@0.6/0.7，
并以 decode、预处理、H2D、probe、selector、backbone、head、后处理、显存和能耗
构成的真实总成本证明收益。

## 当前方法候选

DUCA 当前形态：全窗口低成本 trainable C3/official-ASFormer coarse probe 产生
`p_action` 与隐藏特征；selector 读取隐藏特征、`delta_p_action`、绝对变化、不确定性
和学习特征，以 transition/boundary/utility-first 评分；fixed-K structured policy 在
预算内产生 original-time selected positions；官方 OpenTAD/AdaTAD-derived detector
在 selected axis 上运行；训练期通过 structured zero-forward bridge 接受 detector
梯度。actionness 只是二分类校准和小权重辅助，不负责最终覆盖决策。

准确协议是 `offline_full_window + runtime_generated + cache_free + jointly_trained`。
类名中的 Online 是历史命名，不能用于声称 Online TAD。

## 当前裁决

- `70aa069`：冻结为待裁决的完整 DUCA baseline，不能直接作为论文最终方法。
- `a5e1774`：最新审计提交，加入 full-stack cost profiler 与 AdaTAD 诚实契约；没有
  对应 full-train，只有成本 smoke。
- 正式 fixed-384 Job `1154971` 使用 `70aa069`，不是 `a5e1774`。
- dynamic MUST 暂不作为主贡献；X3D/SlowFast 只作为 frozen-prior diagnostic。
- ChronoTransport `92029ea` formal P3 science gate 保持失败，Stage C/P5 未解锁。一次
  `CT-P3R-3S` bounded appeal 已冻结并通过 spec-only review：commit `e4422f5`、SHA-256
  `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`，状态仅为
  `spec_approved`。
- r2 protocol slice 已独立批准并提交为 `33378af`，但完整实现仍是
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。formal Stage B 获 `APPROVE_STAGEB_FROZEN`；Gate 1
  对上一冻结版返回 `REVISE_GATE1_FROZEN`，正在修复 test/formal 隔离、R/random lock、Git blob
  与 marker 合同。Stage C 第二轮修复虽远端通过 88 tests/1 protected-CUDA skip，但新独立
  审计又复现 detached dummy-forward loss 与 `.data` frozen-heavy mutation 两个 P1，状态退回
  `tested_then_rejected_under_repair`；第三轮候选曾远端 56 focused +134 compatibility +34 protocol
  全绿（另1 protected-CUDA skip）；risk-forward autocast 加固后的最终SHA虽重跑134/1skip，独立
  审计仍发现 latest_signals 可被替换/`.data`改写及success buffer字节未约束两个P1，退回
  `tested_then_rejected_under_repair`；三个远端真实成功步复现均错误返回 `SUCCESS`，逻辑值差异
  max=7.0。Green4 虽远端12/12、兼容143/1skip，独立审核仍发现普通Python Tensor属性未绑定
  storage identity，返回 `REVISE_STAGEC_GREEN4`。Green5 已RED复现并远端targeted14/14、12-file
  superset 198/1skip，正在独立复核。
  Gate-4 纯裁决器切换官方 evaluator
  后远端 13/13，
  但 formal profiler/CLI/launcher、Stage-C/matched-dense runners 与完整 provenance 仍未闭环。
  Gate-1 Green3 已隔离fixture/formal结构并移除caller backend/raw replay路径，远端focused 25、广域
  169/1xfail、Gates23兼容30/30，状态仅 `tested_under_independent_review`。Stage-B context repair
  远端47/47并获独立批准。Gates2/3 round3 已修复terminal、no-clobber和逐级symlink，远端21/21
  与Gate1兼容30/30并获独立 code approval。首轮registration加入4个Gates23路径后仍漏掉Gate1
  hardening test，被独立退回；最终source-vector修复中，没有 Gate report/unlock。
  Gate-4 的equal-score RED为自写AP 1.0 vs 官方0.5；当前仍只属pure adjudicator，formal
  profiler/provenance未完成。
  2026-07-15 pre-deployment 独立审计进一步确认：Stage-B factory/CLI 读取过期 flat
  registration 字段，真实 train-mode ActionFormer `loss_normalizer` 与当前 Stage-C success
  buffer 不变式冲突，顶层 Tensor/per-window regret 合同未接入真实 detector，matched-dense 与
  Gate4 formal runners 缺失，registration source vector 不完整。Stage-B nested-schema、
  no-clobber/writer-lock 与当前 Gate source-vector 修复已在远端隔离 worktree 通过
  `89 passed, 1 xfailed` 完整受影响套件，并通过 `43/43` Gate 兼容性矩阵；这些只属于
  implementation regression evidence。
  Gates2/3 partial-publication recovery 另通过 `59 passed, 1 xfailed`，并获独立
  `APPROVE_GATES23_RECOVERY`；该批准只覆盖 exact-byte resume/no-clobber 小切片。
  Stage-B formal path/lock 又 RED 复现 symlink-parent 与 pre-validation `.resolve()` alias
  laundering；修复后 targeted 5/5、Stage-B+registration `91 passed, 1 xfailed`，并获独立
  `APPROVE_STAGEB_PATH_LOCK_HARDENING`；批准仅覆盖该 bounded integrity slice。
  registration 为 `NOT_READY`；禁止 I/R、formal Gate 1、新 Stage-B seeds、Stage C/Gate 4；
  没有 r2 实验事实。
- 当前 A1 已将 Gate-1 unsuffixed `random_p{2,4,8}` 的唯一 control seed 固定为 3407，并要求绑定
  candidate/registration/replay/recomputation identity；当前代码尚未因该 spec-only commit 自动更新。
- 2026-07-15 后续远端刷新发现当前用户有多项与 ChronoTransport 无关的 DUCA Slurm jobs；
  不得复用或干扰。没有已登记的 ChronoTransport allocation/job，也未启动 ChronoTransport
  训练。
- 当前 A2 已把 remote contract 改为 Slurm 分配单 GPU、launcher 不覆盖 scheduler visibility、
  进程只用逻辑 `cuda:0` 并登记 allocation/GPU UUID；旧 physical-GPU1/CVD=1 实现路径若仍可达即
  不合规。A3/A4 同时固定 successful `loss_normalizer` trace 与一次 dense-reference 加一次
  differentiable counterfactual model forward 的 official per-window regret 合同。历史提案文件
  仍保留 `proposed_unapproved` frontmatter 作为来源记录，当前权威文本是 `537f692` 规范；所有
  execution lock、I/R、PRECHECK 和 formal Slurm job 仍保持关闭，因为 Pro 已批准规范但拒绝当前
  实现进入 registration；须完成修复并通过新的 exact-byte implementation review。
- Gate-4 caller-raw-dict formal minting 已 RED-first 封死：当前 `formal=True` 在统计前拒绝，
  `formal=False` 只生成 `chronotransport-r2-gate4-test-only-v1`。远端 focused 为 `1 passed`，
  forged-payload-with-recomputed-hash targeted 为 `1 passed in 105.28s`，完整 Gate-4 synthetic
  回归为 `13 passed in 242.40s`，registration source-vector focused 为 `1 passed`。独立 exact-byte
  review 返回 `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`；该批准只覆盖 test-only/formal 边界，
  不是 Gate-4 producer、Gate 结果或正式证据，Stage C/matched-dense/Gate-4 workflow 仍缺失。
- Stage-C formal summary 的 `cost_is_measured` 已从“只要求 bool 类型”收紧为必须逐值 `True`。
  RED 先复现 `False` 未被拒绝；首次全回归因旧夹具仍用 proxy cost 出现 `39 failed, 32 passed,
  1 skipped`，随后夹具显式绑定 test-only measured-cost table。最终远端 focused 为 `1 passed,
  71 deselected`，全文件为 `71 passed, 1 skipped in 76.60s`；exact SHA 为 stage-c
  `5BDC1862...5577C4` / test `C92FED39...3A262D7`。独立复核返回
  `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`。该批准只锁定布尔证据，不等于 immutable registered
  cost-profile provenance；后者仍须由缺失的正式 runner/registration 绑定。
- 暂不实现 physical-grid；selected-axis 几何风险保持公开，等待决定性对照。
- 不再增加 selector head/loss，先完成强基线、成本和 hard/soft 对齐。

## 已吸收的关键经验

1. C3 粗动作性有价值，但 actionness top-k 容易选动作内部，边界覆盖不足。
2. PAction learned 曾优于复杂 GAS-VT，说明复杂约束可能过拟合覆盖并趋向均匀。
3. GAS-VT 存在 train/apply 特征不一致、非真实 sequential value transport 和硬 gap
   repair 掩盖学习的问题，因此降级为 Stage1/工程 baseline。
4. move25/move50 显示选择会聚集，但聚集中心可偏离 GT 边界；粗分类误差与 selector
   目标不匹配必须分别诊断，不能继续靠膨胀补救。
5. detector teacher/GT 只可用于 train supervision，val/test 必须递归拒绝泄漏字段。
6. smoke、gradient nonzero、wrapper precheck 只证明接口，不证明 full detector utility。
7. requested K、effective K、unique K、padded detector K 和实际 backbone 输入必须分别
   记录；fixed-384 日志出现 effective budget 低于 384，尚待解释。
8. GT boundary target 只能称 `boundary_utility_proxy`，不能称 true detector utility。
9. 官方 base/head 配置一致不代表官方源码完全一致；DUCA 改变输入长度、GT 坐标和
   post-hoc true-time remap，必须表述为 official-derived detector components。

## 不得重走

- 不再把三阶段独立训练包装成论文最终联合模型。
- 不再用 RGB 均值或密集 X3D 代替主方法的低成本可学习 coarse probe。
- 不再让 actionness coverage 主导 selector；边界/转换优先。
- 不再用硬膨胀、后处理 repair 或 uniform scaffold 隐藏 selector 学习失败。
- 不再用旧 commit mAP 证明最新实现。
- 不再把 PENDING/smoke/precheck 写成正式实验。
- 不再只报 FLOPs；random-init 成本 smoke 不能成为论文数据。
- 不再在 matched uniform/random/dense 基线缺失时扩新模块。

## 决定性门槛

1. 同 commit exact-uniform、periodic、random、dense 与 DUCA fixed-K 对照。
2. one-swap finite-difference：ST gradient 与 hard replacement utility 正相关并优于
   actionness、transition 和 random。
3. same-selected-frames 几何对照，判断 selected-axis 风险是否实质伤害 high tIoU。
4. trained checkpoint 下的 full-stack p50/p95 latency、memory、energy；probe+selector
   不能吞掉 heavy-backbone 节省。
5. mAP@0.6/0.7、短动作、边界距离、max gap、聚集偏移不退化。
6. 至少第二 detector，或诚实收缩为 AdaTAD-specific。

任何核心门槛失败，不再继续调 loss 权重：按原因降级 DUCA、移除 ST bridge、改用
simpler residual selection，或转向 ChronoTransport/PhysTime/CVCR 等新假设。
