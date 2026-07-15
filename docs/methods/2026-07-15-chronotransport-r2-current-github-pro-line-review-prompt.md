# ChronoTransport r2 最新 GitHub 快照 Pro 全量逐行审计 Prompt

用途：在最新 ChronoTransport r2 实现已经发布到 GitHub 分支后，原样提交给
GPT-5 Pro / GPT-5.5 Pro / Oracle Pro。

重要前置事实：GitHub-visible review-only snapshot `b854adb4f4c9235580b5e58c3f3255db6e9adbc0`
已经接受过一次 Pro 全仓审计并得到 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。此后新增：

- `60ee691d148d90ebc5b06ff854fa7a5f4aaf5fec`：逐字归档并吸收该 Pro 审计；
- `537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37`：只修改规范的 A1--A4 spec-only commit；
- 新规范 SHA-256：`E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6`。

本 prompt 将随新的 review-only descendant snapshot 一起发布。下面正文自带 fail-closed
快照门：若 Pro 解析到的远端 HEAD 不是 `537f692` 的严格后代、仍停在 `b854adb`/更旧提交、
规范 hash 不符或缺少必读文件，必须停止并返回 `GITHUB_SNAPSHOT_INCOMPLETE`。不得把上一次
审计对 `b854adb` 的结论自动迁移到新 SHA。

以下分隔线后的正文可原样复制。

---

# 角色

你是 ChronoTransport CT-P3R-3S-r2 有界上诉实现的最终对抗性审查者。请同时以以下身份工作：

- 顶级 TAD/CV 会议的严苛审稿人；
- PyTorch、OpenTAD、AdaTAD、VideoMAE、AMP/GradScaler、缓存与条件计算专家；
- conformal calibration、cluster/hierarchical bootstrap 与实验设计统计专家；
- GPU/Slurm、端到端 profiling、原子产物与研究完整性专家；
- 可以否决整条路线的独立代码审计人。

使用最高推理强度。不要奖励工作量，不要默认同意项目叙述，不要为保住路线而降低标准。
你的目标不是“找几个可能问题”，而是从 GitHub 固定快照逐文件、逐函数、逐执行链重建当前
实现，判断它是否会生成假 Gate、假成本、假梯度、假重试、假 registration 或假科学结论。

# 0. 只读边界与零假设原则

本任务严格限定为：

READ_ONLY_GITHUB_CODE_REVIEW_DISCUSSION_AND_IMPLEMENTATION_PROPOSAL

允许：

- 浏览固定 GitHub fork、Git 历史和官方上游 primary sources；
- 读取代码、测试、配置、启动器、wiki、规格和计划；
- 在回答中给出完整 unified diff、完整替换函数/类、测试和命令；
- 给出 REVISE 或 REJECT_AND_FREEZE；
- 若环境允许，执行只读 clone、哈希、静态分析或 CPU 单测，但必须逐项报告真实执行事实。

禁止：

- 修改、commit、push、merge、tag、开 PR、SSH、Slurm、GPU、训练、校准、profiling 或正式评估；
- 把项目报告的测试次数当成你已经复现；
- 发明 checkpoint、manifest、registration I/R、Gate report、unlock、Job ID、成本或 mAP；
- 从文件名、测试名、注释、wiki 结论推断实现正确；
- 把缺字段解释成 false，把异常 fallback 解释成正常 dense safety；
- 静默改变冻结的模型、head、loss、candidate library、seed、quantile、训练步数、预算、阈值、
  bootstrap 单位、population 或 stop-chain；
- 称 ChronoTransport 为 Online TAD；
- 引入 DUCA/C3 action-frame selection 或 pre-backbone 删帧。

所有结论必须属于以下四类之一：

1. REPOSITORY_FACT：固定 GitHub SHA 的源码直接证明；
2. REVIEWER_EXECUTED_FACT：你实际执行并给出命令、环境、输出和 SHA；
3. PROJECT_REPORTED_FACT：项目 wiki 声称，但你没有独立复现；
4. INFERENCE_OR_PROPOSAL：你的推理、风险判断或建议。

无法从固定快照证明时，只能写 CANNOT_VERIFY。不得补全缺失事实。

# 1. GitHub 快照强绑定

仓库：

https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702

审查分支：

codex/chronotransport-r2-implementation

在开始阅读前必须执行以下逻辑：

1. 从 GitHub 解析该分支当前完整 40 位 HEAD；
2. 将其记为 REVIEW_SHA，之后全程只使用 REVIEW_SHA permalink；
3. 确认 REVIEW_SHA 是
   537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37 的后代且不等于该 spec-only commit；
4. 确认下面列出的最新 production/test/tool 文件全部存在于 REVIEW_SHA；
5. 确认规范文件在 REVIEW_SHA 的 SHA-256 精确等于
   E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6；
6. 报告分支解析时间、REVIEW_SHA、父提交、tree SHA 和可见性。

至少必须存在：

- opentad/models/chronotransport/full_stack_profiler.py
- opentad/models/chronotransport/gate1_unlock.py
- opentad/models/chronotransport/formal_stage_b.py
- opentad/models/chronotransport/gates23.py
- opentad/models/chronotransport/stage_c.py
- opentad/models/chronotransport/gate4.py
- tools/bata/chronotransport_r2_profile_factory.py
- tools/bata/chronotransport_r2_opentad_profile_backend.py
- tools/bata/chronotransport_r2_gate1_replay_factory.py
- tools/bata/chronotransport_r2_stage_b_factory.py
- tools/bata/chronotransport_r2_gates23_replay_factory.py
- tools/bata/profile_chronotransport_r2_full_stack.py
- tools/bata/train_chronotransport_r2_stage_b.py
- tools/bata/run_chronotransport_r2_gates23.py
- tools/bata/validate_chronotransport_r2_precheck.py
- tests/test_chronotransport_r2_gate1_cost_profile.py
- tests/test_chronotransport_r2_gate1_hardening.py
- tests/test_chronotransport_r2_gates23.py
- tests/test_chronotransport_r2_gate4.py
- tests/test_chronotransport_r2_registration.py
- tests/test_chronotransport_r2_stage_b.py
- tests/test_chronotransport_r2_stage_c.py
- tests/test_chronotransport_pipeline.py
- tests/test_chronotransport_vit_adapter_integration.py
- EXPERIMENT_AUDIT.md
- EXPERIMENT_AUDIT.json
- research-wiki/sources/2026-07-15-chronotransport-r2-predeployment-integrity-audit.md
- research-wiki/sources/2026-07-15-chronotransport-r2-github-pro-snapshot-gate-response.md
- research-wiki/sources/2026-07-15-chronotransport-r2-pro-review-b854adb-verbatim.txt
- research-wiki/sources/2026-07-15-chronotransport-r2-pro-review-b854adb-absorption.md
- docs/methods/2026-07-15-chronotransport-r2-minimal-protocol-amendment-proposal.md

任一条件失败，立即只返回：

GITHUB_SNAPSHOT_INCOMPLETE

并列出解析到的 SHA、缺失文件或 hash mismatch。不要继续审查，不要用 797a2df 或本 prompt
中的状态叙述代替最新源代码。

历史第一次调用因远端仍是 `797a2df` 而返回 `GITHUB_SNAPSHOT_INCOMPLETE`；第二次调用审计了
`b854adb` 并返回 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。你必须在本次调用中 fresh
resolve 分支；不得复用 `797a2df`、`b854adb`、旧 compare 结果或旧 implementation verdict。

# 2. 权威规格与证据边界

规范文件：

docs/superpowers/specs/2026-07-12-chronotransport-ct-p3r-3s-r2-design.md

当前 A1--A4 spec-only commit：

537f692189cf0c5a6ee7d40ad8c4ed1032bf1d37

当前规范 SHA-256：

E79DFAAB8F9B0093E96CBD6B46BEF4ECF8D6433009E2DCB922AD0F4C473B27A6

前置冻结规范 commit/hash：

- `e4422f5`；
- `87FA305CCAFC3A29176C3971F593489F86EDD23A4C02C1BFBDAE4144FCF34CF8`。

实现计划：

docs/superpowers/plans/2026-07-12-chronotransport-ct-p3r-3s-r2-implementation.md

A1--A4 最小修订的历史提案：

docs/methods/2026-07-15-chronotransport-r2-minimal-protocol-amendment-proposal.md

提案 SHA-256：

30371FFC17B02DF615FF0D772B93BADF30CF0A3AB84E36325CBF5A71EFD8469F

用户已授权按该提案的 exact A1--A4 decisions 推进，`537f692` 将它们合并进当前规范；提案
frontmatter 保留历史 `proposed_unapproved`，不能取代当前规范。你必须先做一次纯文本
`e4422f5 → 537f692` spec-diff review：确认 A1--A4 被精确合并、没有内部矛盾、没有改变其他
seed/split/threshold/candidate/budget/update/bootstrap/population/stop-chain。分别输出
`APPROVE_SPEC_FOR_PLAN` 或 `REVISE_SPEC_BEFORE_PLAN`。即使 spec 通过，也只解锁实现审查，不批准
当前代码、I/R、PRECHECK 或实验。

历史正式负结果：

- commit 92029ea 的旧 formal Stage-B/P3 为 negative gate；
- 它只否定旧 cell-sum risk / 旧统计协议；
- 它不是 r2 结果，也不允许 r2 变成无限调参；
- r2 仍只有一次有界上诉，任一科学 Gate FAIL 后永久冻结。

即使 Gate 1--4 将来全部通过，deploy=false 与 paper=false 仍保持冻结，等待外部验证和
新颖性复核。

# 3. 必读顺序

请完整读取，不得只搜索关键词：

## 3.1 仓库规则和当前记忆

1. AGENTS.md
2. RTK.md
3. research-wiki/query_pack.md
4. research-wiki/anti_repetition.md
5. research-wiki/ideas/chronotransport.md
6. research-wiki/experiments/chronotransport-formal-stage-b.md
7. research-wiki/experiments/chronotransport-r2-implementation-verification.md
8. research-wiki/sources/2026-07-13-chronotransport-r2-independent-frozen-audit-cycle.md
9. research-wiki/sources/2026-07-15-chronotransport-r2-predeployment-integrity-audit.md
10. research-wiki/source_registry.md
11. research-wiki/log.md 中所有 ChronoTransport r2 条目

## 3.2 规格、计划和历史审查

1. r2 design spec 与 implementation plan；
2. research-wiki/sources/2026-07-12-chronotransport-r2-spec-only-independent-agent-review.md
3. research-wiki/sources/2026-07-12-chronotransport-r2-independent-implementation-audit.md
4. research-wiki/sources/2026-07-13-chronotransport-r2-pro-github-code-audit-response.md
5. research-wiki/sources/2026-07-13-chronotransport-r2-pro-github-code-audit-absorption.md
6. docs/methods/2026-07-15-chronotransport-r2-minimal-protocol-amendment-proposal.md
7. research-wiki/sources/2026-07-15-chronotransport-r2-pro-review-b854adb-verbatim.txt
8. research-wiki/sources/2026-07-15-chronotransport-r2-pro-review-b854adb-absorption.md
9. `e4422f5 → 537f692` 的单文件 spec diff；
10. OSS_AUDIT.md，仅当它真实存在于 REVIEW_SHA 时读取；否则标记 NOT_IN_SNAPSHOT。

旧审查只能作为待复核的问题列表，不能作为当前代码缺陷或当前修复成功的事实。

## 3.3 全部实现表面

必须读取 opentad/models/chronotransport/ 下所有 Python 文件，并特别覆盖：

- actions.py
- cache.py
- controls.py
- protocol.py
- transport.py
- risk.py
- scheduler.py
- runtime.py
- replay.py
- losses.py
- training.py
- cost_lookup.py
- full_stack_profiler.py
- adjudication.py
- gate1_unlock.py
- formal_stage_b.py
- gates23.py
- stage_c.py
- gate4.py
- registration.py
- __init__.py

同时读取：

- opentad/models/backbones/vit_adapter.py
- 所有 ChronoTransport r2 configs；
- tools/bata 下所有 chronotransport_r2 文件；
- scripts 下所有 ChronoTransport launcher；
- 所有 tests/test_chronotransport*.py；
- 任何 registration required-source set 列出的文件；
- 任何可以生成、验证或发布 formal artifact/report/unlock/terminal/checkpoint 的间接依赖。

不能因为 registration 列表没写某文件就不读它；“遗漏 source”本身就是需要发现的问题。

# 4. 当前项目报告的实现现象：全部必须重新验证

以下只是 PROJECT_REPORTED_FACT，不是对当前快照的预判。逐项给出 VERIFIED、PARTLY_VERIFIED、
REFUTED 或 CANNOT_VERIFY，并附 REVIEW_SHA permalink 与最小复现。

## 4.0 新规范与 `b854adb` Pro 审计

- 上一次 Pro 对 `b854adb` 的 snapshot gate 为 PASS，整体 verdict 为
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`；reviewer 没有运行 tests/CUDA/Slurm；
- 它报告 Stage-C/matched-dense/Gate-4 formal workflow 缺失，当前 Stage-C canonical Tensor hook
  与 ActionFormer loss dict/`loss_normalizer` 合同不相容，registration 漏掉两份本次修改的
  integration tests，A1--A4 未批准，measured-cost provenance 与 Slurm identity 未闭环；
- `60ee691` 只归档/吸收审计，`537f692` 只修改规范；这不等于上述代码问题已经修复。你必须从
  当前 REVIEW_SHA 重新验证，不能因旧 finding 存在而直接复述，也不能因新 spec 存在而假定实现
  已符合；
- 当前规范 A1 固定 unsuffixed `random_p2/p4/p8` 的唯一 control seed 为 3407；
- A2 改为 Slurm 分配 single GPU、不改 visibility、进程只用 `cuda:0` 并绑定 allocation/GPU UUID；
- A3 允许且只允许 train-mode `rpn_head.loss_normalizer` 每 successful arm update 规范推进一次，
  dense-reference 临时 mutation 必须先恢复，CT/matched trace 必须相同；
- A4 要求每 attempt 恰好一次 no-grad dense model forward、恢复 paired state/RNG、恰好一次
  differentiable counterfactual model forward，从同一 official head logits/targets 暴露 length-two
  per-window task-loss vector，并生成 detached regret targets 与恰好一次 risk forward。

先审查新规范自身是否一致，再审查实现是否逐项满足。若 spec 有歧义，标 `SPEC_BLOCKER`；若
spec 清楚而代码未实现，标 `MISSING_OR_NONCOMPLIANT_IMPLEMENTATION`，不得混为一类。

## 4.1 Gate 1

- Green2 曾因 arbitrary backend/callable、fixture/formal 可重标记、caller detector/batch/raw
  replay 可进入 formal schema 而被拒绝；
- Green3 声称已移除上述路径，formal execution 改为 repository-owned session，fixture/formal
  schema 结构隔离；
- 项目报告 Green3 远端 25 focused、169 passed/1 expected xfail、30 Gates2/3 compatibility；
- Green3 仍处于 independent review，不是 Gate-1 PASS；
- predecessor spec 中 unsuffixed random_p2/p4/p8 的 control seed 曾有歧义；当前 A1 规范已固定
  为 3407。当前代码必须精确生成、注册和复算该 identity；若仍保持“缺 seed 即永久 lock”或接受
  caller seed，均是不符合当前规范，而不是继续存在的 spec ambiguity；
- 后续 precheck RED 证明 registration、output root 和 Gate1 fixed-artifact 的父目录 symlink 可被
  pre-validation `resolve()` 洗白；项目声称当前已改为逐级 lexical `lstat`，并独立重建
  registered `R/shared/gate1` 后再接受 resolver 输出；
- 项目报告 focused 1/1、registration/precheck `38 passed, 1 xfailed`、Gate1 hardening/cost 25/25；
  独立 reviewer 返回 `APPROVE_GATE1_PRECHECK_PATH_HARDENING`，匹配 precheck SHA
  `0BE0EA8BA3FCAD46387611E3140E116381FD7EE50344291F02E6D724FCF76808` 与 registration-test SHA
  `55916FBD5182EB2D6024BA5EA9A16B66117F51E545BEF704C3584B862B1C10BA`；
- 没有正式 profile、B*、Gate-1 report 或 unlock。

## 4.2 Stage B

- Stage-B core 与 Gate1-context repair 分别获得独立 code approval；
- 项目报告 context repair 47/47；
- 2026-07-15 pre-deployment audit 又发现 Stage-B factory/CLI 读取过期 flat registration 字段；
  当前项目声称已改为嵌套 artifact/split identity，并将 registration、phase、checkpoint/ledger
  publication 改为 no-clobber 加 exclusive writer lock；
- 项目报告新 targeted 7/7，完整 Stage-B+registration `89 passed, 1 xfailed`；
- 后续 RED 又证明 output argument 在检查前 `.resolve()` 会洗白指向 canonical R/seed 的 symlink
  alias，writer-lock 接受 symlink parent 且 cleanup 会删除同名替换文件；当前项目声称已改为
  lexical component `lstat`、O_EXCL/O_NOFOLLOW 与 device/inode-safe cleanup；
- 项目报告该修复 targeted 5/5、最终 Stage-B+registration `91 passed, 1 xfailed`；独立 reviewer
  返回 `APPROVE_STAGEB_PATH_LOCK_HARDENING`，匹配 runner SHA
  `64B4A5AAE70FEE358DEE3F639B8E9063E72DB6B8813509D9FB9BB16423053B3D` 与 test SHA
  `1E7BB88394521E8441FF92FC30AE8CC99A9F1099B531F85A9A77E7CB137094CA`；
- 上述 path/lock SHA 是历史候选，不是当前文件 SHA；当前 runner/core/test 已因 partial-publication
  recovery 再次变化，必须审查下面的最终 SHA，不得把旧 approval 自动转移；
- 后续 RED 又证明 exact-existing ledger/baseline/marker 被 no-clobber 永久拒绝，导致 periodic/final
  ledger 先落盘或 final checkpoint+ledger 已完成后的中断不可恢复；首个 recovery 候选虽然
  `98 passed, 1 xfailed`，仍被独立 `REJECT_STAGEB_PARTIAL_PUBLICATION_RECOVERY`，因为 phase dense
  checkpoint 父 alias 仍可被跟随，CLI 在 pre-lock path check 后又使用 `Path.exists()`/pathname
  `torch.load()`；该被拒绝候选不可复用；
- 当前替换候选声称：只允许 regular same-inode exact bytes 复用；periodic/final dangling ledger
  可继续；已有 final checkpoint+ledger 时不重训，而是先用临时 exact baseline 调完整 phase
  validator；marker-without-baseline 等不可能状态拒绝；dense/trained checkpoint 使用
  lexical `lstat`、`O_NOFOLLOW`、`fstat` device/inode 绑定，同一字节同时用于 SHA 与 torch.load；
  final-pair 状态在 seed lock 内重验；
- 项目报告最终 targeted 5/5、Stage-B+registration `98 passed, 1 xfailed`、Gate compatibility
  44/44；独立 reviewer 返回 `APPROVE_STAGEB_PARTIAL_PUBLICATION_RECOVERY`，匹配 current runner SHA
  `50F4469D82C4F2530741DB7E3D7B88C5517B73A91D6F087AF6342E04146F4F84`、core SHA
  `47342FFE2BC83481F76D004840C76D9FF72F79BC8BF8D7DAEF4B5ABA818A7670`、test SHA
  `9BB46DE26A8C9A38F7AA97F31F3D5F1546F189B1F8AD64872340F992C03E378D`；
- 它仍必须被 Gate-1 正式 unlock 阻塞；
- 没有任何新 seed 的 140 successful updates、fit replay、calibration 或 evaluation。

## 4.3 Gates 2/3

- round2 曾因 seed 在每个 window 内重抽、context bypass、伪 checkpoint/ledger、并发覆盖和
  symlink parent 等问题被拒绝；
- round3 声称修复 report-derived terminal、exclusive lock、atomic no-clobber 和逐级 symlink；
- 项目报告 21/21 focused、30/30 Gate1 compatibility，并有独立
  APPROVE_GATES23_ROUND3_CODE；
- 首次 registration integration 仍因漏登记
  tests/test_chronotransport_r2_gate1_hardening.py 被拒绝；
- 当前声称正在做最终 exact source-vector repair；
- 后续 audit 发现 replay 或 report 单独落盘后会使同一 immutable R 永久不可恢复；当前项目
  声称已改为重新计算后仅复用 byte-identical partial artifact，已有 terminal、mismatch、
  non-regular、symlink 和并发 writer 仍 fail closed；
- 项目报告该修复 targeted 1/1、Gates2/3+registration `59 passed, 1 xfailed`，独立 reviewer
  返回 `APPROVE_GATES23_RECOVERY`，并匹配 runner SHA
  `4CED5459B1785855F46FE0A22748229D77D885245D44B8A84160C3B814616885` 与 test SHA
  `10D134573FEB8029DCE02F4A01E0CE2D40006DAD199FA0E7AAA32129BF65AFB9`；
- 没有正式 Gate-2/3 report 或 unlock。

## 4.4 Stage C

- 早期实现多次错误返回 SUCCESS，已实际复现 detached losses、错误 action、拓扑交换、
  latest_signals 替换和 .data mutation、frozen parameter/buffer mutation、ordinary Python
  Tensor storage/alias rebind 等路径；
- Green4 仍漏 ordinary Python Tensor storage identity；
- Green5 声称加入 layout/storage cdata/data pointer/nbytes/offset 与 alias identity；
- 项目报告 Green5 2/2 targeted、14/14 aggregate、198 passed/1 protected-CUDA skip；
- Green5 仍在 independent review；
- 后续 reachability RED 证明 formal summary 只检查 `cost_is_measured` 的 bool 类型，允许
  `False`。该字段现被要求精确为 `True`；首次全回归因旧 fixture 仍用 proxy cost 出现
  `39 failed, 32 passed, 1 skipped`，修正为显式 test-only measured-cost fixture 后，远端 focused
  1/1、Stage-C 全文件 `71 passed, 1 skipped in 76.60s`。最终 SHA 为 stage-c
  `5BDC1862AD90F1D0A6134ADD778D5978A536848EEAD63EDE973444CBCA5577C4` 与 test
  `C92FED397F69E03BE6F0189483250F8132579DD844C521CE3E17BEF0B3A262D7`；独立逐行复核仅返回
  `APPROVE_STAGEC_MEASURED_COST_FLAG_LOCK`，明确不批准 immutable profile provenance 或 Stage C；
- 最新 pre-deployment audit 指出：真实 ActionFormer training forward 返回 loss dict，而当前
  Stage-C hook 要求顶层 differentiable Tensor；真实 train-mode head 会更新
  `loss_normalizer`，而当前 success audit 要求全部 registered buffers 不变；toy loss container
  也未证明同一 batch-two forward 产生两个逐窗口 regret targets；
- 当前 A3/A4 已把允许的 normalizer 成功态变化、dense-reference state restore、一次 dense + 一次
  counterfactual model forward、length-two official per-window loss 与 regret target 唯一化；必须
  审查生产代码是否真实接通，而不是让 toy fixture 自报；
- Slurm-assigned single-device CUDA GradScaler/autocast 行为与每次全模型 hash/clone 的真实开销未测；
- 4200-update Stage-C CT runner、matched-dense runner、完整 resume/EMA/LR/exposure artifact 和
  post-Stage-C recalibration 是否存在，必须从 REVIEW_SHA 重新判断。

## 4.5 Gate 4

- pure adjudicator 曾被 equal-score AP 反例击穿：自写 AP=1.0，官方 OpenTAD evaluator=0.5；
- 当前声称已切换官方 evaluator；后续 integrity RED 又证明 caller-owned raw mappings 只要写
  `formal=True` 就可绕过 provenance 铸造同一 Gate-4 report schema；
- 项目报告现候选在任何 raw evidence parsing 前拒绝 `formal=True`，并把非正式结果改为
  `chronotransport-r2-gate4-test-only-v1` / `formal_evidence=false`；远端 focused 1/1、完整 pure
  synthetic 13/13（242.40s）、forged payload + recomputed hash targeted 1/1、registration source
  focused 1/1；最终 SHA 为 gate4
  `A581D71338B130C2FF0ECB2B833B29F1B7B1FD5A8F5C36E7A24BC7B954B1A75F` 与 test
  `5C0FFAF398EC45958045C46CE714BE391E987197532F60409D840F6AAAB4506E`；独立逐行复核返回
  `APPROVE_GATE4_CALLER_EVIDENCE_LOCK_FINAL`，只批准这条 bounded evidence-boundary lock；
- 这不等于 formal Gate 4；
- formal CLI/launcher、registered Stage-C completion、matched full-stack profiler、frozen
  official population/order、raw predictions/provenance 与 terminal chain 是否存在，必须重查。

## 4.6 Registration 与部署

- registration 当前状态是 NOT_READY；
- 不存在 clean implementation commit I、single-parent registration-only commit R 或正式 I/R；
- 最新本地 `EXPERIMENT_AUDIT.md/.json` 的 readiness verdict 为 `FAIL`，理由是正式执行链缺失，
  不是发现了伪造论文数字；其 reviewer 是独立 Codex agents，无法证明不同模型家族，因此你必须
  从 REVIEW_SHA 独立重做全部判断，不能继承其结论；
- 任何 profile/replay 前必须先冻结 identity；不能从看过结果后再补 registration；
- 后续远端刷新发现同一账号有多项与 ChronoTransport 无关的 DUCA Slurm jobs；不得复用、修改
  或干扰。没有已登记的 ChronoTransport allocation/job；
- 当前 A2 要求 single-GPU Slurm allocation/step、launcher 不修改 scheduler visibility、进程只用
  local `cuda:0`，同时记录 Slurm physical identity/GPU UUID；旧 physical-GPU1/CVD=1 launcher、
  validator、registration field 或 claim flag 若仍可达，均为 noncompliant implementation；
- A1--A4 已进入 `537f692` 当前规范，但本次 reviewer 仍须先独立给 spec-diff verdict；规范通过
  不会自动修复 registration、launcher、Stage-C 或 provenance；
- 没有正式 ChronoTransport job、Gate 结果或论文数字。

# 5. 第一优先级：逐行审计 registration 是否可能铸造假证据

请从每个 public/importable API 出发做调用图，而不是只看 CLI happy path。

必须验证：

1. REQUIRED_REGISTRATION_SOURCE_PATHS 是否精确覆盖所有 production、test、config、launcher、
   validator 和 formal source，包括最终 hardening tests；
2. source validator 自己要求的每个文件是否也在 exact set；
3. 是否漏掉 Gate4、Stage-C/matched-dense runner、formal launchers，或者登记了并不存在的表面；
4. I 必须 clean；R 必须是 I 的唯一单亲后继；I..R 只能新增唯一 registration 文件；
5. in-memory registration 是否逐字节绑定 R:path 的 canonical bytes、regular Git blob mode 与
   当前 worktree bytes；
6. spec、config、checkpoint、annotation、200-video registry、140/30/30 frozen-window manifest、
   candidate/action library、exposure matrix、cost environment、source vector 是否全部深验证；
7. 任何 caller-authored identity、extra field、missing field、coercive bool/int/float、NaN/Inf、
   symlink、path traversal、parent symlink、hardlink/overwrite、TOCTOU 是否 fail closed；
8. registration generator 是否绝对无法读取 result/profile/replay/evaluation 路径；
9. 每个能生成 report/unlock/terminal 的 API 是否独立验证 clean detached R、random lock、
   registration blob 与 source bytes，而不是依赖 launcher 先检查；
10. test-only schema 是否无法被重标记、重哈希或经 validator 转换成 formal schema；
11. 两个并发进程是否绝不可能同时通过 precheck 并覆盖正式产物；
12. registration source-vector 的最终修复是否真的包含
    tests/test_chronotransport_r2_gate1_hardening.py、Gates2/3 的四个最终文件、
    tests/test_chronotransport_pipeline.py 与 tests/test_chronotransport_vit_adapter_integration.py；
13. 是否存在一份逐文件、显式、可审计的 source-classification manifest，能解释每个
    tests/test_chronotransport*.py 为什么属于 REQUIRED、TEST_ONLY_NON_FORMAL 或 OUT_OF_SCOPE；不得用
    宽泛文件名前缀自动扩大 registration，也不得让未分类测试静默逃逸。

对每个 fail-open 路径给出最小攻击脚本和 RED test。

# 6. Gate 1：真实成本、oracle headroom 与 evidence-origin 审计

逐行证明或否定：

- exact 23-candidate frozen order；
- 200 个 manifested windows 的固定 invocation order；
- 每 candidate 的 media/checkpoint/manifest verification 在 warmup/timer 前完成；
- 50 warmup、200 direct full-invocation total_ms samples；
- 每个 total sample 覆盖 decode、preprocess、H2D、patch embed、scheduler、cache/transport、
  heavy backbone、adapter、projection/head、postprocess/result accumulation；
- p50/p95 从完整 total distribution 计算，不能相加 stage percentiles；
- profile backend 固定为 repository-owned OpenTAD backend，不能注入 callable/backend；
- detector、batch、motion source、regret、action 或 success callback 不能由 caller 自报；
- dense/candidate replay 使用同一 materialized batch、RNG 和增广；
- requested 与 executed action/hash/cost 完全分离；
- repair、NaN fallback、dense fallback、safety_override_budget_violation 不能继续沿用 requested
  cost 作为成功证据；
- B* 直接来自真实 periodic4_transport p50，并用规范要求的精确 20% arithmetic；
- Gate-1 oracle 只证明 frozen-library oracle headroom，不证明 deploy-visible input dependence；
- bootstrap 每 replicate 重新选择 strongest/evaluation-best comparator；
- random/motion action bytes 从冻结算法、window identity、deploy-visible signals 和批准 seed
  重建；
- unsuffixed random_p2/p4/p8 必须只接受当前规范固定的 control_seed=3407，并把该值绑定到
  candidate bytes、registration、replay 与 recomputation identity；缺失、不同值、caller override、
  隐式 RNG 或继续永久 lock 都是实现不合规，不再是可猜测的 spec ambiguity；
- formal result/terminal 是 exclusive-lock + fsync + atomic no-clobber；
- 每个 public adjudication/unlock API 都无法从 caller rows 铸造 formal PASS。

给出 Gate-1 端到端数据流图与所有信任边界。

# 7. Stage B：140 successful updates 与 phase 原子性

验证：

- seeds 固定为 3407/3408/3409；
- one-window-per-video shared split；
- batch size 1、world size 1、shuffle false；
- candidate=(p+5*b+seed_offset)%16 的 block-rotation、tails、matrix hash 与跨 seed 暴露；
- 每 seed 精确 140 successful FP32 updates，无 AMP、无额外 epoch；
- attempted/successful/skipped、LR、EMA、schedule exposure、candidate×video exposure 全部记账；
- dense/counterfactual paired RNG、pixels、augmentation、head state 与 candidate-order permutation；
- strict load approved dense checkpoint，而不是只 hash 文件；
- transport/risk 独立 AdamW 非零 LR，其他参数确实 frozen；
- 16-candidate fit replay 和每 window 16-vector rank-127 baseline；
- predictor canonical/alias、EMA、state_dict、checkpoint hash 全等；
- training checkpoint、ledger、baseline、predictor identity 与 completion marker 原子完成；
- crash/resume 不接受 prefix corruption、重复 row、跳过 exposure 或半完成 COMPLETE；
- periodic/final dangling ledger 只能在 fresh recomputation exact-byte 相同时复用，不能覆盖；
- 已有 final checkpoint+ledger 时必须在锁内验证 regular inode，以同一 descriptor bytes 完成
  hash/deserialization，并在任何 canonical baseline/marker 发布前通过完整 phase preflight；
- baseline/marker 的发布顺序、允许恢复状态和不可能状态必须穷举；marker-without-baseline、
  checkpoint-without-ledger、phase artifact without training pair 必须 fail closed；
- public phase builder 对 dense/trained checkpoint、ledger、baseline 的 parent/leaf symlink、
  nonregular file 和 pathname replacement 必须独立拒绝，不能依赖 CLI 先检查；
- legacy six-schedule/old split/candidate-row pooled runner 从 formal launcher 不可达；
- Gate1 unlock 使用最终 mandatory context validator，不能接受简化 PASS JSON。

# 8. Gates 2/3：统计单位、全局 seed cluster 与正式产物

验证：

- Gate 2 在完全相同 RECOMPUTE mask 下比较 TRANSPORT 与 HOLD；
- feature 和 detector-regret 各自的 point/CI 与 stop rule；
- 30 calibration + 30 evaluation frozen windows，不把 candidate rows 当独立样本；
- Gate 3 先在每个 calibration window 的 16 candidates 取 residual max，再对 30 maxima 取
  rank 28；
- constant baseline 为 fit-only 每 schedule tau=0.9 empirical quantile；
- 每 seed/window 在完整 16-candidate vector 内计算 Spearman；
- hierarchical bootstrap 先全局抽 seed cluster，再抽 unique windows，绝不能在每个 sampled
  window 内重抽 seed；
- selected coverage、all-candidate-covered、window-all-selected-covered、non-dense support、
  distinct selected windows、pinball、sharpness 与 overcoverage 的定义和分母正确；
- evaluation-best static 只读 diagnostic，deployment comparator 在 fit/calibration 冻结；
- no-GT/no-teacher/no-oracle/no-raw-prediction/no-replay-ledger flags 由真实执行上下文证明；
- formal replay 只能由 repository-owned fixed session 产生；
- report 先完整发布并重验证，terminal 后生成；
- lock、no-clobber、symlink-parent、R/context/source-vector 在所有 public paths 独立执行；
- Gates2/3 最终 source files 和 tests 已真正进入 registration exact set。

必须给一个最小数值反例，展示错误 seed 重抽如何把 FAIL CI 变成 PASS。

# 9. Stage C：真实 forward、梯度所有权与事务重试的对抗审计

不要因测试很多就放松。逐行重建一次 CT update 和 matched-dense update。

必须验证：

## 9.1 真实 forward 与 loss provenance

- 唯一 production ChronoTransportRuntime/source identity；
- 每个 attempted paired update 恰好执行一次 no-grad dense **model** forward；随后恢复 dense-reference
  临时改变的 module/buffer/Python tensor/RNG 状态，再恰好执行一次 differentiable counterfactual
  **model** forward；只调用 head、复用旧 logits、额外第三次 model forward 或把 toy hook 当生产桥接均不合规；
- dense 与 counterfactual 必须从同一 official ActionFormer head logits/targets 暴露精确 shape `[2]`
  的 per-window task-loss vector；不得只用 batch-aggregate scalar、loss dict 求和后复制、外部伪造
  Tensor 或 detached proxy；
- 两个 detached regret targets 必须只由上述 paired official task losses 派生；risk predictor 在每个
  attempt 恰好一次 differentiable forward，且不得读取 GT、teacher、raw-prediction cache 或 replay ledger；
- LD、LF、LR 来自同一个 materialized batch、同一个真实 runtime forward graph/outputs；
- runtime executed action tensor 由真实 scheduler/forced control 产生，caller expected payload
  无权证明执行；
- detector output、feature output、latest_signals 从 forward boundary 到 loss/risk 消费期间，
  reference、storage、metadata、version 与 logical bytes 全部绑定；
- dummy forward + detached external losses、.data mutation、equal-value clone/rebind、view/alias
  split 都必须失败；
- runtime summary exact-key/type/value fail closed；forced dense、fallback、repair 或 invalid
  evidence 不得训练成正常 CT sample。

## 9.2 A/T/R ownership

- A=all AdaTAD adapters，T=transport，R=risk，object identities 两两不交；
- predictor alias 只进入一个 logical optimizer group；
- detector loss 只给 A∪T；
- 0.1 feature loss 只写 T.grad，但允许穿过 adapter Jacobian；
- 0.1 risk pinball 只写 R.grad，signals/target 对其他模块 detach；
- 使用三次 scaled autograd.grad、一次 unscale、finite audit、global clip=1、一次 step/update；
- generic name-substring optimizer 和 total-loss backward 不可达；
- heavy VideoMAE、projection/head 与其他参数 bytewise frozen。

## 9.3 Success/overflow state

- snapshot 覆盖 Parameter、persistent/nonpersistent buffer、ordinary Python Tensor 属性、
  shared mutable object graph、RNG、optimizer、scheduler、EMA、sampler、cursor、exposure、
  ledger、profiler 和 head loss_normalizer；
- ordered path→object/type、module/parameter/buffer alias graph 与 parent registration order；
- .data 写入、equal-value storage rebind、isomorphic block swap、alias split 都能检测；
- frozen head parameters 必须保持冻结，但 rpn_head 保持规范要求的 train mode；
- `rpn_head.loss_normalizer` 是唯一被规范允许成功持久化的 detector buffer：dense-reference forward 的
  临时推进必须在 counterfactual 前恢复；每个 successful CT/matched arm update 恰好规范推进一次，
  overflow/INVALID 不得推进，CT 与 matched-dense 的 successful trace 必须逐步相等；其他 detector
  parameter/buffer/Python tensor 仍须 bytewise/identity-safe frozen；
- overflow 只保留 GradScaler backoff，其他状态（包括 dense-reference normalizer 临时 mutation）恢复；
- 不对未改变 Parameters 无条件 load_state_dict，从而破坏 version counter/non-leaf graph；
- success path 也审计 buffer/Python/frozen bytes，不能只审 overflow；
- 同一 materialized batch 最多初次加三次 retry，第四次 INVALID；
- successful batch/order/augmentation hashes、4200 common A updates、LR/EMA trace 在 CT 与 dense
  匹配，不要求 overflow vector 相同。

## 9.4 完整 runner

检查当前快照是否真的存在：

- 4200-success CT runner；
- matched-dense runner；
- atomic checkpoint/resume/completion artifact；
- canonical 8400 exposure/seed、525/candidate；
- post-Stage-C calibration 与 Gate-3 rerun；
- Stage-C launcher、validator、registration source coverage；
- Slurm 分配的 single-device 上真正 CUDA AMP/autocast/GradScaler precheck：launcher 不得覆盖
  CUDA_VISIBLE_DEVICES，进程只使用 logical `cuda:0`，artifact 同时绑定 Slurm allocation/step、
  scheduler-visible device identity 与 GPU UUID；
- O(frozen-model-bytes) 每 attempt hash/clone 的实测开销与是否会吞掉节省。

若只存在 primitive 或 synthetic tests，必须判 MISSING_EXECUTABLE_WORKFLOW。

# 10. Runtime/VideoMAE/AdaTAD 语义逐层重建

跟踪一个 768-point input：

dataset/window → decode/preprocess/H2D → dense patch embedding → 48×16-frame chunks →
384 tubelet points → 每个 VideoMAE block/layer group → RECOMPUTE/TRANSPORT/HOLD →
cache → full-row AdaTAD temporal adapter → projection/head → 768 detector grid →
official postprocess/NMS。

对每个 action 写出：

- 输入 tensor、source、shape、dtype/device；
- cache anchor/latest/actual age/embedding age；
- current row 是否 live gradient、历史 cache 是否 detached；
- heavy attention/MLP 是否是唯一被跳过的重路径；
- full-row adapter 是否所有 rows 都执行并写回；
- requested/executed identity、fallback/repair 和下一 block 可见状态；
- actual validity age 47 与 embedding cap 8 是否分离；
- forced dense 是否与原始 dense block 数值/梯度等价；
- external 768 detector geometry 是否保持。

对官方 OpenTAD/AdaTAD/VideoMAE upstream 使用永久 commit 链接，标记 SAME、WRAPPED、
STRUCTURALLY_MODIFIED、CUSTOM、MISSING 或 CANNOT_VERIFY。不得称该 fork 为“官方未修改”。

# 11. Gate 4：official population、真实端到端成本与 mAP bootstrap

先区分 pure adjudicator 与 formal evidence producer。验证：

- official full-video/sliding-window population 与 Gate1--3 frozen-window population严格分离；
- matched invocation set、checkpoint/cache/config/metric provenance 全冻结；
- CT/dense/static 六序 balanced crossover；
- latency 每次完整 forward 的 total sample，不用 stage percentile 相加；
- timing repetitions 只进入 latency bootstrap；
- metric/regret 每个 unique official invocation 只计一次；
- 每个 resampled seed 在同一 official-video multiset 上，用 raw predictions/GT 调用仓库官方
  OpenTAD evaluator 重建 mAP，不跨 seed merge predictions/NMS；
- equal-score AP 语义与官方 evaluator 一致；
- shortest-action threshold 只来自 fit GT Q1；
- latency saving one-sided 95% LCB≥15%；
- mAP@0.7 drop one-sided 95% UCB≤1.5；
- shortest-Q1 mAP@0.7 drop UCB≤1.5；
- heavy_saving 确实 >0；
- overhead margin 与 static-regret improvement 的 CI 正确；
- learned CT full-stack p50 不慢于 calibration-frozen best static；
- 三个 seed 均不越过失败阈值；
- p95、memory、throughput、stage breakdown、10-Hz long-block energy 只作规定层级的报告；
- formal CLI、launcher、registration source set、Stage-C completion chain、exclusive output 与
  terminal 都实际存在。

当前若只有 gate4.py 纯函数和 synthetic tests，整体 Gate4 readiness 必须是 MISSING。

# 12. 统计红队

逐条数学复核并给最小数值 counterexample：

- rank 28/30 与 rank 127/140；
- per-window maximum 后 simultaneous marginal calibration；
- selected empirical coverage 不等于 selected-conditional guarantee；
- Gate3 guarantee 不转移到 Gate4 official population；
- per-window candidate-vector Spearman 与 degenerate ties；
- global seed cluster bootstrap；
- Gate1 replicate 内 comparator re-selection；
- Gate4 official-video → matched block → seed latency hierarchy；
- Gate4 official-video → seed metric/regret hierarchy；
- one-sided和two-sided CI的使用位置；
- denominator≤1e-12 的 relative improvement 未定义；
- candidate row、timing repeat、overlapping window、seed 不能伪装成独立 video。

# 13. 安全性与研究完整性攻击清单

至少主动构造以下攻击，而不是等待代码显式出现：

1. fixture row 改 schema/hash 后进入 formal validator；
2. import private token/issuer 后用 caller detector/batch 铸造 PASS；
3. caller 伪造 regret、action hash、transport_executed 或 callback；
4. in-memory registration 与 R blob 内容不同；
5. Git blob 为 symlink/非 regular mode；
6. R 为 merge commit或 I..R 含额外文件；
7. parent-directory symlink；
8. 两个并发 writer 覆盖 report/terminal；
9. result 与 marker 指向同一路径；
10. first candidate 独自承担 media verification I/O；
11. random seed 缺失却默认 3407；
12. requested schedule repair 后仍用 requested cost；
13. dense safety fallback 超预算却记为成功；
14. Stage-B checkpoint COMPLETE 但 baseline/ledger 未完成；
15. Gate2/3 用伪 checkpoint、任意 140-row ledger 或 caller no-leak booleans；
16. Stage-C dummy forward + detached LD/LF/LR；
17. runtime action 与 caller expected payload 不同；
18. .data 改 frozen parameter/buffer/signals/output；
19. equal-value storage rebind 或 base/view alias split；
20. depth-12 block swap或共享 container alias split；
21. success path 污染 Python/buffer state；
22. overflow restore 破坏 Parameter version/non-leaf graph；
23. Gate4 自写 AP 与官方 equal-score 语义不同；
24. timing repetitions 混入 metric bootstrap；
25. registration 漏登记最终 hardening test却声称 exact implementation vector。
26. Stage-B phase builder 的 dense checkpoint 经 parent symlink 指向注册文件；
27. Stage-B 在 pre-lock 检查后把 final checkpoint 替换成 symlink、目录或另一 inode；
28. ledger/baseline/marker 已存在但 bytes 不同，恢复逻辑静默接受或覆盖；
29. marker 存在但 baseline 缺失时自动重建，从而掩盖删除/篡改。

每个可行攻击必须给：

- exact permalink；
- 最短 failure trace；
- 为什么现有测试没有阻止它，或哪一个测试已经阻止；
- RED test 名和精确断言；
- 最小 production patch。

# 14. 逐行覆盖证明

不要只说“已全面审查”。输出 coverage table：

- file path；
- REVIEW_SHA blob SHA；
- line count；
- 实际读取范围；
- 关键 public APIs；
- callers/callees；
- verdict；
- 未读或不可见原因。

对所有 ChronoTransport production、tool、config、launcher、test 和 required wiki/spec 文件给出
coverage。任何相关文件未读，整体 verdict 不能是 APPROVE。

# 15. Finding 格式

按 P0→P3 排列。每个 finding 必须包含：

- 标题；
- REVIEW_SHA 永久链接和最小行范围；
- REPOSITORY_FACT；
- 违反的规范条款；
- 可执行 failure trace 或数值反例；
- 对科学结论、成本、梯度、数据隔离或可复现性的影响；
- 最小 patch 边界；
- RED test 与精确 assertions；
- 是否改变科学规范，还是只完成既定实现。

明确区分：

- confirmed bug；
- missing executable surface；
- weak/false-positive test；
- unverified risk；
- spec ambiguity（只能指 `e4422f5 → 537f692` diff 审完后仍真实存在的歧义，不能拿旧 A1--A4
  状态替代当前代码不合规）；
- project report contradicted by code。

# 16. 具体实现输出合同

审查后，对所有 registration-blocking P0/P1 给完整 implementation-grade unified diff 或完整
replacement code。

要求：

- 不得有 TODO、pass、伪代码、省略核心逻辑或“类似地处理”；
- 不得弱化 fail-closed、source binding、stop-chain 或 frozen protocol；
- 不得为了通过 Gate 改 head/loss/candidate/seed/budget/threshold；
- 不得把 result-derived data 放入 registration 或 inference；
- formal API 不接受 caller-owned evidence；
- canonical serialization、strict exact-key/type、finite checks；
- 每个 patch 先给 RED tests，再给 production change，再给 regression tests；
- 保护 C3/DUCA 和无关路径；
- 说明与现有 schema/checkpoint 的兼容或明确 invalidation；
- 给出 CPU/CUDA/Slurm 验证命令，但未执行必须标 NOT_EXECUTED_BY_REVIEWER。

如果缺少事实，写：

PATCH_BLOCKED_BY_MISSING_FACT

并列出缺失事实、完整接口合同和必须新增的测试；绝不发明实现。

# 17. 下一步计划要求

基于实际 findings 给出最小、按依赖排序的计划。每一步必须有：

- 输入 commit/artifact；
- 修改文件；
- RED test；
- GREEN command；
- independent review boundary；
- 输出 artifact；
- stop condition；
- wiki 状态迁移。

必须明确回答：

1. 当前 snapshot 能否成为 implementation commit I？
2. 若不能，最小 mandatory patch set 是什么？
3. 哪些 slice 可以保持冻结，哪些必须重开审查？
4. 何时才允许创建 I？
5. 何时才允许创建 single-parent registration-only R？
6. 何时才允许 PRECHECK_ONLY？
7. 何时才允许 formal Gate 1？
8. Gate 1 FAIL 后应如何永久停止？
9. Gate 1 PASS 后 Stage B、Gate2/3、Stage C、Gate4 的严格解锁顺序是什么？
10. 哪些 CUDA/性能检查必须在真实实验前完成，但不属于科学 Gate？

不接受“先跑起来再看”。

# 18. 强制裁决

先单独给出 `e4422f5 → 537f692` 的 A1--A4 spec-diff verdict，只能选择：

- APPROVE_SPEC_FOR_PLAN；
- REVISE_SPEC_BEFORE_PLAN。

`APPROVE_SPEC_FOR_PLAN` 只说明当前规范可作为实现合同；它不批准 implementation、registration、
PRECHECK 或实验。若 spec verdict 为 REVISE，仍须尽可能完成只读代码审计，但整体不得 APPROVE。

先分别给出以下 slice verdict：

- runtime/cache/adapter；
- registration/source vector；
- Gate1 profile/replay/adjudication/unlock；
- formal Stage B；
- Gates2/3；
- Stage-C primitive；
- Stage-C/matched-dense workflow；
- Gate4 pure adjudicator；
- Gate4 formal workflow；
- launchers/Slurm/GPU identity；
- end-to-end stop chain。

每个只能选择：

- APPROVE_FROZEN_SLICE；
- REVISE_SLICE；
- REJECT_SLICE；
- CANNOT_VERIFY。

整体只能选择一个：

- APPROVE_IMPLEMENTATION_FOR_REGISTRATION；
- REVISE_IMPLEMENTATION_BEFORE_REGISTRATION；
- REJECT_AND_FREEZE。

APPROVE 的必要条件包括：spec-diff verdict 为 `APPROVE_SPEC_FOR_PLAN`、相关文件全读、无 P0/P1、
所有 mandatory executable surfaces 存在、registration exact set 与显式 source classification 闭合、
A1--A4 在生产实现与 RED tests 中逐项满足、CPU/必要 CUDA 行为证据边界清楚。测试多、代码长或
“看起来防御性强”都不是批准理由。

# 19. 强制输出顺序

请按以下顺序用中文回答：

1. Executive Verdict，不超过 300 字；
2. GitHub Snapshot/Visibility Certificate；
3. A1--A4 Spec-Diff Verdict；
4. Evidence Taxonomy Table；
5. Line-by-Line Coverage Certificate；
6. Current Phenomena Recheck；
7. End-to-End Implementation and Trust-Boundary Map；
8. Spec Section-by-Section Compliance Matrix；
9. Registration/Evidence-Minting Audit；
10. Gate1 Audit；
11. Stage-B Audit；
12. Gates2/3 Statistical and Formal Audit；
13. Stage-C Forward/Gradient/Transaction Audit；
14. Runtime/AdaTAD/VideoMAE Upstream Matrix；
15. Gate4 Metric/Cost/Bootstrap Audit；
16. P0→P3 Findings；
17. Adversarial Counterexamples and RED Tests；
18. Slice Verdicts；
19. Overall Route Verdict；
20. Minimal Mandatory Patch Architecture；
21. Complete Unified Diffs or PATCH_BLOCKED_BY_MISSING_FACT；
22. TDD/CPU/CUDA/Slurm Verification Matrix；
23. Registration Readiness Checklist；
24. Dependency-Ordered Next-Step Plan；
25. Result-to-Claim Matrix；
26. Permanent Kill Criteria。

# 20. 最终纪律

- 代码是实现事实，spec 是目标合同，测试不是科学结果；
- precheck/smoke/synthetic adjudication 不是 Gate；
- 没有 r2 正式实验数字；
- 不得把 Gate1 oracle headroom 写成 input dependence；
- 不得把 Gate3 frozen-window coverage 转移到 Gate4 official population；
- 不得把 dense fallback 当作 budget success；
- 不得把 requested cost 当作 repaired/executed cost；
- 不得创建或建议创建 I/R，除非当前固定快照真的满足全部前置条件；
- 发现一个科学 Gate FAIL 时必须建议永久冻结，不得换 head/loss/seed/library 延长路线；
- 严厉、精确、逐行、可复现、给出具体实现；不要照顾作者情绪。
