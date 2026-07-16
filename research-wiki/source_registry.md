---
type: source_registry
updated: 2026-07-16
---

# 来源注册表

## Codex Tasks

| 来源 | 覆盖范围 | 本地归档 |
|---|---|---|
| `019f49d2-a7ef-7273-b420-8732fae46bf8` | DUCA 主讨论，191 轮，158 条用户消息 | [完整用户记录](sources/thread-019f49d2-user-record.md) |
| `019f20d8-5e8d-72d3-a2dc-898b75ce03ea` | 目标、实现、部署代理 | [近期记录](sources/delegated-thread-recent-record.md) |
| `019f3cd2-30cd-7452-a210-1ef9fd53fd14` | 论文写作代理 | [近期记录](sources/delegated-thread-recent-record.md) |
| `019f4066-8bd9-73f0-9af5-30dc9da45cce` | 早期目标梳理 | [近期记录](sources/delegated-thread-recent-record.md) |

## 代码来源

- C3 clean repo：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- DUCA/GASVT 审计 worktree：`E:/DeskTop/TAD/OpenTAD_GASVT_CostAudit_20260710`
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- DUCA branch：`codex/gas-vt-stage23-detector-aware-20260706`
- 当前审计 commit：`a5e1774b9941312569ca645341da1abad339db61`
- 当前正式训练 commit：`70aa069b895322c2307ffbb13dfdef9fac0d1305`

## 关键评审与吸收记录

以下记录是主要二级来源；原始 raw review 仍保存在各 repo 的
`docs/methods/reviews/`：

- `gas-vt-stage01-53124a2-review-absorption.md`
- `2026-07-07-c69c1a0-paction-gasvt-hold-review-absorption.md`
- `2026-07-07-46cacc1-pro-final-route-review-absorption.md`
- `2026-07-08-fbea37b-learned-context-radius-hold-review-absorption.md`
- `duca-plugin-final-method-review-absorption.md`
- `duca-online-plugin-final-design-review-absorption.md`
- `duca-online-plugin-603ed02-real-detector-review-absorption.md`
- `2026-07-09-544eca6-duca-transition-first-critical-review-absorption.md`
- `2026-07-09-7bea4fc-duca-hold-paper-claim-review-absorption.md`
- `2026-07-10-88e50b1-duca-final-method-audit-review-absorption.md`
- `2026-07-10-70aa069-researchclaw-duca-divergent-audit-absorption.md`
- `2026-07-10-duca-official-adatad-structural-audit.md`
- `sources/2026-07-12-chronotransport-r1-pro-github-review-response.md`：基于公开固定
  commit `1f5f7254a390f183121e6c4b7cebcebd2f2954d1` 的完整 1,924 行 Pro 规格终审；
  外部附件 SHA-256
  `07A5B4B519E64A39D7F84CE862F0E56117BFF2DB62206B6AE24BDD66768B19FE`。
- `sources/2026-07-12-chronotransport-r1-pro-github-review-absorption.md`：本地复算、
  上游核验、非完全同意项与 r2 决策记录。
- `sources/2026-07-12-chronotransport-r2-spec-only-independent-agent-review.md`：空白上下文
  独立 agent 的两轮 spec-only 复核；最终批准 commit `e4422f5`。
- `docs/methods/2026-07-12-chronotransport-r2-pro-github-code-audit-prompt.md`：面向无法读取
  本地文件的 Pro reviewer 的 GitHub-only 完整代码审核 prompt；固定审核代码快照
  `4b07020acb2611c3f085488d2f678f3be037f1be`；prompt 首次发布 commit `6079135`，draft PR
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/pull/1`；禁止执行实验或生成
  registration。
- `sources/2026-07-13-chronotransport-r2-pro-github-code-audit-response.md`：外部 Pro 对固定
  GitHub 快照 `4b07020acb2611c3f085488d2f678f3be037f1be` 的 1,429 行完整源码审计原文；原附件
  SHA-256 `1B3A02373366A95654C00A5FE76F451F800D16A877B2688BB460674B25849142`。
- `sources/2026-07-13-chronotransport-r2-pro-github-code-audit-absorption.md`：接受
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`、七项既有 blocker、config nesting 与
  conformal statistical unit 两项新增 P0，以及分级 P1--P3 和 Route-B 修复顺序；明确该审计
  不是实验、110 tests 未独立复跑、sandbox patch proposal 不可访问且未执行。

评审建议不是实验事实。wiki 中只有带 run root、Job ID、日志或 result artifact 的内容
才可标记为实验事实。

## 远端实验来源

- DUCA 70aa fixed-384 Job：`1154971`
- Run root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_70aa069_final_20260710_1544`
- a5e cost smoke Job：`1156079`
- Cost smoke root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cost_profile_smoke_20260710_1652`

## 外部附件

主任务显式引用的关键附件包括：`5f9a0d62...`、`86b473c6...`、
`d0087ae1...`、`1705e957...`、`60cb3e7a...`、`0ce290f9...`、
`a885a659...`、`391f061f...`、`c2008dfb...`、`c8a36eba...`、
`1fc36774...`、`d8b9f9fc...`。其中最后一份 ResearchClaw 审查已原样归档，
SHA256 为 `E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`。
本轮新增附件 `bf3c8b10-951f-4765-87d5-53c6ba02b7dd` 已完整归档；其原文件身份见上方
ChronoTransport r2 Pro 审计条目。

- `sources/2026-07-13-chronotransport-r2-independent-frozen-audit-cycle.md`：当前 exact-SHA
  Gate-1/Stage-B/Stage-C 独立复审、修复证据、仍缺实施表面与两个规格 blocker 的完整记录。
  这是代码审计来源，不是 Gate 或实验结果。
- docs/methods/2026-07-15-chronotransport-r2-current-github-pro-line-review-prompt.md：
  面向下一次 GitHub-only Pro 全量逐行审计的 fail-closed prompt。它要求先把远端分支解析为
  `537f692` 的严格后代；若仍为 `b854adb`/更旧提交、缺最新文件或当前 spec hash 不符则停止。
  它强制先裁 A1--A4 spec diff，再逐行审 production、registration、Stage C、Gate 4 与 Slurm
  identity，并要求 implementation-grade RED tests/diffs。该文件只是审查输入，不是 reviewer
  response、代码批准、Gate 结果或实验事实。2026-07-15 latest-current SHA-256 为
  `9DDCABC19E6B38874EA97F5E4702C247D2DF8F485CE273E08E4A6515EBFEC3D0`；旧
  `1D0E7FC1...3BF5` 已被当前 A1--A4 规范和 `b854adb` 审计吸收要求取代。
- `sources/2026-07-15-chronotransport-r2-predeployment-integrity-audit.md`：当前 dirty bytes 的
  独立只读 pre-deployment 审计；裁决为 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`，记录
  registration/reachability、Stage-C/matched-dense、Gate4 与 Slurm 规格冲突。不是 test rerun、
  Gate 或实验结果。
- `../EXPERIMENT_AUDIT.md` / `../EXPERIMENT_AUDIT.json`：2026-07-15 experiment-audit 技能的
  人工/机器可读 readiness 报告，SHA-256 分别为
  `EC576F6EA370C3DBF3A44B284D435FD286559F20909E6A754A660CA3FE79D742` 与
  `8C96408455D75924F057BA97B3A11D6A6BC1DB14B13E8BE2BC2A9B497F46B40D`。总体 `FAIL` 表示正式
  证据链不可执行，不表示发现了伪造结果；当前确实没有正式结果。
- `../.aris/traces/experiment-audit/2026-07-15_run01/`：上述独立 reviewer request/response 与
  metadata 的 forensic trace。reviewers 为分离的只读 Codex agents，未重跑测试、未编辑、未启动
  Job；无法证明不同模型家族，仍须使用 GitHub-only Pro prompt 做外部复核。
- `sources/2026-07-15-chronotransport-r2-github-pro-snapshot-gate-response.md`：用户提供的首次
  GitHub-only Pro 返回，SHA-256
  `AF4E4FA612671F426D7E9316434F60B48A14A26EE034D5DDDB65FF489D691DDE`。裁决仅为
  `GITHUB_SNAPSHOT_INCOMPLETE`：远端分支仍等于禁止使用的 `797a2df`，因此 reviewer 在第一门
  停止，未检查缺失文件、spec hash 或任何代码。它证明 prompt fail-closed，不是实现审查结果。
- `sources/2026-07-15-chronotransport-r2-pro-review-b854adb-verbatim.txt`：用户提供的 GitHub-only
  Pro 完整审计逐字归档，共 73,605 bytes / 1,430 lines；原附件与归档 SHA-256 均为
  `1A7B9D5AEA47302AC7BCB29DB9EF54DAD97CF3D45DF1536691CB9B536EC4C376`。审计对象是 review-only
  SHA `b854adb4f4c9235580b5e58c3f3255db6e9adbc0`，总体裁决
  `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`。reviewer 未复跑 tests/CUDA/Slurm，覆盖证书也不
  声称逐行读完全部仓库，因此它不是实验事实或 implementation approval。
- `sources/2026-07-15-chronotransport-r2-pro-review-b854adb-absorption.md`：对上述 Pro 审计的逐项
  事实核对、采纳矩阵与保留意见。总体裁决、主要 P1/P2 blocker 和修复依赖顺序被接受；裸 test
  glob、未冻结 Stage-C dataclass、`_gpu1.sh` 命名及 pre-I/R official-population precheck 不原样采用。
- `sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-verbatim.txt`：用户提供的第二次
  GitHub-only Pro 完整审计逐字归档，共 86,871 bytes / 2,019 lines；原附件与归档 SHA-256 均为
  `C61F93531885040A3593DB7552E23B67B34DEC3D55095D71FCE5B6D2A1F1BC08`。审计对象是 review-only
  SHA `1b6366d0acb712e8096c2cceb0f05e66b16d30d4`，tree
  `3fc64c72cf26b77f041d059f51385f29e5e85462`，唯一 parent `537f692...1d37`。它返回
  `APPROVE_SPEC_FOR_PLAN` 与 `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION`，未执行 tests、CUDA、
  Slurm、训练、profiling 或 evaluation。
- `sources/2026-07-16-chronotransport-r2-pro-review-1b6366d-absorption.md`：对该审计的完整采纳与
  保留记录。接受规范批准、总体实现裁决、四项核心 P0、source/path/workflow blocker；不原样采用
  不完整 A2 patch、未冻结 Stage-C dataclass/loss namespace、自动测试分类及过宽的永久 kill 语义，
  并补记 profiler/Stage-B/Gates23/stale-plan 等未覆盖实施面。
- `sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be.md`：用户提供的第三次
  GitHub-only Pro snapshot-gate 返回，75 lines / 2,194 bytes，SHA-256
  `990E84F1D09116257D684090163BACB3F579ACA7290BADCB4D9FC6CFDA151FD1`。它 fresh-resolved
  `92a18be`，独立验证了 `^1=6c3606c`、无 `^2`、ahead 1/behind 0 与 exact eight-path docs-only
  diff，但因 reviewer 接口不暴露 tree SHA 和分离的 author/committer timestamps 而按 prompt 返回
  `GITHUB_SNAPSHOT_INCOMPLETE`。未进入代码或 registration 裁决。
- `sources/2026-07-16-chronotransport-r2-github-pro-snapshot-gate-92a18be-absorption.md`：对上述
  snapshot-only 返回的证据边界与工具能力缺口记录；用户已批准以 SHA/parents/ancestry/exact
  diff/逐文件 content-addressed checks 构成严格等价 fallback，并要求发布新 docs-only snapshot。
- `../docs/methods/2026-07-16-chronotransport-r2-final-github-pro-implementation-review-prompt.md`：
  两次 Pro 讨论的 40-line 使用索引，1,631 bytes，SHA-256
  `1BBD516A1F758031635933B403677FD67E9C8456671760BCCCE1C7F381C104DD`。原 406-line 单体 prompt
  因超过 Pro one-turn thinking duration 而停止作为直接输入；索引本身不是第三份审计 prompt。
- `../docs/methods/2026-07-16-chronotransport-r2-github-pro-audit-part-1-foundation.md`：第一轮
  GitHub-only Pro prompt，267 lines / 13,141 bytes，SHA-256
  `D468586282C59AA182E09955C82456EC9E54ED3DF2974A851A298AFB4917972C`。覆盖 snapshot、规范、
  A1/A2、registration/source/filesystem、Gate 1、Stage B 与 Gates 2/3；只输出 Part-1 verdict 和
  `PART1_AUDIT_PACKET`，不得给整体 registration verdict。
- `../docs/methods/2026-07-16-chronotransport-r2-github-pro-audit-part-2-final.md`：第二轮
  GitHub-only Pro prompt，272 lines / 14,084 bytes，SHA-256
  `52A9C3A2422F0E186485F9C932821D37205F36EA45352083245A81887D519835`。必须绑定同一 SHA 并附上
  Part-1 完整原文，覆盖 Stage C/matched、post-Stage-C Gate 3、Gate 4、Slurm 与联合覆盖，且独占
  `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` / `REVISE_IMPLEMENTATION_BEFORE_REGISTRATION` 总裁决。
