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
- Spatial Zoom S1 worktree：
  `E:/DeskTop/TAD/OpenTAD_SpatialZoom_S1_AuditFix_20260715`
- Spatial Zoom S1 branch：`codex/spatial-zoom-s1-audit-fix-20260715`
- 首轮审计提交：`64e71ddc633f9c63f9dea1c5c60c49dc00441ebf`
- 当前 Spatial Zoom S1 formal code commit：
  `0421a8d9f6982a6d4ec1fb590cd108581fa2bb83`
- Spatial Zoom S1 official-evaluator policy fix:
  `cbc63d07` (`DetectionCorpus` retains finite zero-length predictions and
  rejects reversed/non-finite rows; focused tests `41 passed, 1 skipped`).

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

评审建议不是实验事实。wiki 中只有带 run root、Job ID、日志或 result artifact 的内容
才可标记为实验事实。

- 2026-07-15 Spatial Zoom S1/DUCA exact-commit Pro audit：本地附件
  `C:/Users/skywalker/.codex/attachments/69a2a56a-019c-43d1-9063-a2333ce34faa/pasted-text.txt`。
  当前路线只吸收其中 S1 的 strict determinism、Bayesian cluster bootstrap、事务证据和
  cost claim boundary；DUCA findings 不属于 Spatial Zoom 实验事实。

## 远端实验来源

- DUCA 70aa fixed-384 Job：`1154971`
- Run root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_70aa069_final_20260710_1544`
- a5e cost smoke Job：`1156079`
- Cost smoke root：`/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_cost_profile_smoke_20260710_1652`
- Spatial Zoom S1 exact snapshot：
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_64e71dd_20260715_ghfast`
- Spatial Zoom S1 suite：
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_64e71dd_20260715_221121`
- Spatial Zoom S1 packaging-only failure：Job `1165647`；不属于模型证据。
- Spatial Zoom S1 first full CUDA gate：Job `1165648`；Linux tests `41 passed`，全模型
  backward 到达后因预期未用的分类 `fc_norm` 两个参数触发过严门禁。无训练结果。
- Spatial Zoom S1 replacement exact snapshot：
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_4784242_20260715_ghfast`
- Spatial Zoom S1 replacement suite：
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_4784242_20260715_2245`
- Spatial Zoom S1 CUDA gate：Job `1165667`，COMPLETED 0:0，precheck v6 PASS。
- Spatial Zoom S1 formal 3x3 jobs：`1165669-1165677`；远端 `jobs.tsv` 位于 replacement
  suite 根目录。canonical namespace：
  `695803b687bf52197847e8b7fbf3d802c968d13070c660138f524ed31548f3a7`。
- Spatial Zoom S1 storage-invalid matrix：Jobs `1165669-1165677` 全部因共享存储耗尽
  fail-closed；失败 canonical root 保留日志、配置与 151 个 sidecar。无效权重回收收据：
  `invalid_storage_failure_purge_receipt.json`，文件 SHA-256
  `b5237253eaa8d196957da47d5ebd2c07ae6537596b6e53e1e4348286c88d58d9`，
  内部 receipt SHA-256
  `8c9eb6dbbfaec12a38eb6444a9594eb8286e4256cb01496ee6413510a6bed017`。
- Spatial Zoom S1 storage-safe snapshot：
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_0421a8d_20260716_ghfast`。
- Spatial Zoom S1 storage-safe suite：
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_0421a8d_20260716_0324`；
  deployment summary 文件 SHA-256
  `2219a4a52611d0940ee58210e98510d720720a83baf9f477e9d15a72d6a8255e`。
- Spatial Zoom S1 replacement CUDA gate：Job `1165774`，COMPLETED 0:0，precheck
  internal SHA `3d30ea5489b2ac7f07785dff94ed057ac420aebdd8762ab6df6c76a2ffb003ea`。
- Spatial Zoom S1 fresh formal 3x3 jobs：`1165775-1165783`；canonical namespace
  `bf71376e2d57946a3f898d25b7dcc88cfc002549a9ed78656293f1a95316a8f7`。
- Spatial Zoom S1 selector-failure evidence: Jobs `1165775-1165780` reached
  epoch 59 and complete gate artifacts, then exited `1:0` because the old
  analyzer rejected official finite `[-0.0, 0.0]` proposals. The raw suite and
  namespace above are the source; no sealed test was opened.

## 外部附件

主任务显式引用的关键附件包括：`5f9a0d62...`、`86b473c6...`、
`d0087ae1...`、`1705e957...`、`60cb3e7a...`、`0ce290f9...`、
`a885a659...`、`391f061f...`、`c2008dfb...`、`c8a36eba...`、
`1fc36774...`、`d8b9f9fc...`。其中最后一份 ResearchClaw 审查已原样归档，
SHA256 为 `E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`。
