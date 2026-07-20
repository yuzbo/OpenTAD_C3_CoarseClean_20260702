---
type: source_registry
updated: 2026-07-20
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
  `18139b930bef6ee234f6220a6adc898eb9c23c0c`
- Spatial Zoom S1 official-evaluator policy fix:
  `cbc63d07` (`DetectionCorpus` retains finite zero-length predictions and
  rejects reversed/non-finite rows; focused tests `41 passed, 1 skipped`).

## 关键评审与吸收记录

- Native-Crop S2 Crop-Sufficiency Preregistration v1, 2026-07-20. The Pro
  protocol audited commit
  `f9eca5ab81ee0429469789cfa697e7851dce4bd4` and proposed a 21-candidate,
  three-seed, source-native crop-sufficiency matrix with fit/gate isolation,
  detached GT-visible reference analysis, official evaluator parity,
  full-stack cost, immutable receipts, and mechanical decisions. Exact raw
  archive:
  `docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-raw.txt`,
  SHA-256
  `E14ABFAB41FAFA3C3F411DF87D3148170872A190C274ED9B7EB2DD44C520C7D5`.
  Project absorption:
  `docs/methods/reviews/2026-07-20-native-crop-s2-crop-sufficiency-preregistration-pro-absorption.md`.
  Project verdict is `ACCEPT_WITH_MAJOR_REVISION / HOLD_IMPLEMENTATION`:
  the GT-visible rule is not a library upper bound; crop sufficiency,
  selection headroom, and cost need separate outcomes; gate GT ordering,
  statistical sampling units, coverage terminology, selector-cost reserve,
  and result-blind power feasibility must be corrected in v1.1 before formal
  implementation or queueing.
- Native-Crop S1 Pro review, 2026-07-20. The reviewer audited immutable commits
  `d76ba1b`, `cef9548`, and `18139b9`, verified that the current path contains
  full-frame resize rather than source-native crop, and returned
  `PROCEED_NATIVE_CROP_S1`. Exact raw archive:
  `docs/methods/reviews/2026-07-20-native-crop-s1-pro-review-raw.txt`,
  SHA-256
  `7AB0E10624A14FDF2FCABCBEF5EF435EB4994B83BC3E32F3240A3E0143CD44D5`.
  Project absorption:
  `docs/methods/reviews/2026-07-20-native-crop-s1-pro-review-absorption.md`.
  The verdict is accepted only as authorization for a development-only,
  no-training vertical slice. The project rejects route-level interpretation
  of an eight-candidate oracle, final-mask-only padding claims, and uncalibrated
  crop/GO thresholds.
- Native-Crop primary-literature verification, 2026-07-20:
  Glance and Focus `https://arxiv.org/abs/2010.05300`;
  Uni-AdaFocus `https://arxiv.org/abs/2412.11228` and
  `https://github.com/LeapLabTHU/Uni-AdaFocus`;
  AdaSpot CVPR 2026
  `https://openaccess.thecvf.com/content/CVPR2026/html/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.html`
  and `https://github.com/arturxe2/AdaSpot`;
  EVAD ICCV 2023
  `https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Efficient_Video_Action_Detection_with_Token_Dropout_and_Context_Refinement_ICCV_2023_paper.html`;
  AdaTAD `https://arxiv.org/abs/2311.17241`. These primary sources support
  strong prior-art collision around global/local spatial focus. AdaSpot's
  official implementation also supports the narrower observation that its
  variable ROI is resized to a fixed local input, while this project's proposed
  source-native no-resize contract remains only a candidate differentiator.
- Native-Crop S1 implementation audit, 2026-07-20. Independent max agent
  `019f7b48-3fad-75e2-a848-0606e7bd8887` reviewed the complete working-tree
  implementation in three passes. Pass one found and verified repairs for two
  P0 runtime defects plus P1 contract/test gaps. Pass two found two remaining
  P1 provenance issues (expected-commit/source-blob binding and stale/self-
  signed geometry) plus one P2 single-axis crop-area error. After exact Git
  blob checks, in-gate re-probing of all 200 development videos, corrected
  intersection-area formulas, and forged/replaced-source negative tests, pass
  three returned `DEPLOY` with `P0/P1/P2/P3 = 0`. Remote WIP focused tests:
  `17 passed`. This is implementation-audit evidence only; it is not a CUDA
  gate, crop-accuracy result, cost result, or paper claim.
- Native-Crop S1 formal implementation Gate, 2026-07-20. Git commit
  `0bf59be877eeb6879166893641c12bc4e60a2b53`; clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_native_crop_s1_0bf59be_20260720_clean`;
  clean replay `173 passed`; Slurm Job `1174671`, `COMPLETED 0:0` on `g0059`;
  run root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/native_crop_s1_0bf59be_20260720_0225`.
  Full-model precheck internal/file SHA-256:
  `ba278a191905b492d78b07ec253857774a0311c363b70ceac8921159a855b0fc`
  /
  `b0cfe61261f39ef801be6b5800510d9feff54b0f0b73babfcc00d091a33bccde`.
  Geometry census internal/file SHA-256:
  `9a688297fec2ac8537a44e46254df2e277ec69785cfa179aac658ad0b0dc39d8`
  /
  `99089f8260a7a8c7ec3610e521e638f5c18e297c11217597e5b7b139407b013b`.
  Scope is implementation/provenance/gradient closure, not crop accuracy,
  efficiency, GO/KILL, or paper evidence.

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
- Spatial Zoom S1 evaluator-policy replacement snapshot:
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_18139b9_20260716_ghfast`.
- Replacement suite:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_18139b9_20260716_1120`;
  deployment summary file SHA-256
  `32f693fb391e2fa9777d6263e683210cac28a58ba30688009b60525e398529b0`.
- Full CUDA gate Job `1166358`: `COMPLETED 0:0`; precheck internal SHA
  `4275cadaf28cc78d548fe220dcfc3496cd3150b668074c560da791958e0838f1`.
- Fresh formal 3x3 Jobs `1166361-1166369`; canonical namespace
  `d95a36db4bc70aa2ac9d15e5fb5be82174a8a3488c5150c71d2ad4c10c7234a7`.
- Jobs `1166361-1166369` all completed `0:0`; each canonical cell contains ten
  frozen checkpoint/sidecar/gate-evidence/prediction sets and one validated
  gate-only selection. Raw selected metrics are recorded in
  `experiments/spatial-zoom-s1-infrastructure.md` and remain non-test evidence.
- Single S1 test-open certificate:
  `d95a36db4bc70aa2ac9d15e5fb5be82174a8a3488c5150c71d2ad4c10c7234a7/test_open/test_open_certificate.json`;
  file SHA-256 `a6d1bf973e3b55c20e30c8e99521a0317219e579df90c5bf61564d3f436a3c57`,
  internal SHA-256 `8627866a3dfed48a7ddab8df9cb6276d5710e4530c7d8089f929470b0f42f040`.
  Global marker file SHA-256:
  `9cf603afa1f2794e2f3b84958eb6a23e9acffaf73d4b89767e8caafbae9bb646`.
- Post-processing Job `1167230` failed before test read/profile start because
  host `SLURM_JOB_GPUS` was not a valid cgroup-local `nvidia-smi` index.
  Root-cause/adapter/preflight Jobs are `1167232/1167238/1167239`; the strict
  adapter SHA-256 is
  `2693cac2aaa7572045f9c69321e57944f3a09d8e5bb68227724cb83f2047888e`.
  Remediation Job `1167257` runs the frozen 3x3 test/profile order in one
  allocation. Source receipt:
  `spatial_zoom_s1_18139b9_20260716_1120/post_test_profile_resubmission_r1.json`.
- First sealed-test evidence from Job `1167257`: dense256/seed3408,
  `dense256/seed3408/gpu1_id0/test_evidence/test.evidence.json`, file SHA-256
  `10c0182d6fae42f37dec108988f22fbfd732725fc270426121ff2608837261e9`;
  prediction SHA-256
  `d4b6df44b0be9c9f735ef233dd39a9b28ad487ebfd6d530383285d9de7269194`.
  Raw official-test Avg-mAP is `67.09`; this is one incomplete matrix cell and
  not a resolution decision or GO/KILL result.
- Job `1167257` terminal profile-failure evidence:
  `spatial_zoom_s1_18139b9_20260716_1120/logs/test_profile_matrix_r1-1167257.{out,err}`.
  The canonical failed-attempt marker is
  `dense256/seed3408/profile/dense256_seed3408.started.json`; no corresponding
  summary/samples/power/descriptor was published. A read-only reconstruction
  from the frozen manifest and THUMOS annotation produced 792 official loader
  exposures, 791 physical identities, and one exact duplicate,
  `video_test_0001431:7680`. This evidence is registered as an infrastructure
  failure and does not supersede the sealed-test result.
- Exact immutable recovery inputs rechecked at `2026-07-17T03:34+08:00`:
  `test_profile_matrix_r1-1167257.out` SHA-256
  `6f8c0b6bb61cd0dbcf9abde51fdda0b43d79755771de99ba2f6cbe9d3bb37ec9`;
  `.err` SHA-256
  `d8b631c4feef73829184a5fb175cb7a8689d76683d348de09247d5dece33cf70`;
  failed v4 marker file SHA-256
  `093e58c015f93d62512851d9aec36d95c8da5b86283328c1df5eb7af3beac32a`
  and internal marker SHA-256
  `ef3ba7f6443b9e54a20c4eaef798caf0b7a7b9e72c97f6e263d421104db17ab4`.
  The validated dense256/seed3408 test-evidence file remains SHA-256
  `10c0182d6fae42f37dec108988f22fbfd732725fc270426121ff2608837261e9`.
  These inputs authorize only certificate construction after a clean audited
  repair commit; they do not themselves authorize another Job.
- S1 profile-recovery implementation commits: `20b84d210a994e08960bd7f7d542474cf2432603`
  adds exposure/physical-window identity separation and immutable campaign
  infrastructure; `341cf979458d5ea7d0e1c951f0e40cf7a36f738a` adds exact clean
  historical training-repository and
  precheck reconstruction after the first clean replay failed before issuing a
  certificate. Local verification at `2026-07-17T04:05+08:00` is `66 passed,
  1 skipped`; no recovery Job or result is attributed to these commits yet.
- Failed recovery campaign `bb56f9d0283b12c0`: certificate path under canonical
  `d95a36db.../profile_campaigns/bb56f9d0283b12c0`, internal SHA
  `1a0bc133d5006f31409ce9ea86a8ee70cc1e275ceef205d21aa7c9cb3334004f`.
  Gate Job `1167497` preserved script SHA `1d437d42...` and error-log SHA
  `3453a029...`; failure preceded Python/CUDA. Gate Job `1167500` preserved
  script SHA `9ff999a0...`, temporary failure JSON reporting missing relative
  `data/thumos-14/annotations/thumos_14_anno.json`, and Slurm accounting of
  about 249.5 GB disk read. These are infrastructure-failure sources only and
  do not authorize a test/profile result or replacement matrix.
- Failed recovery campaign `10105b8b590cd7fc`: certificate internal SHA
  `0f02a64b3150e97a8a172de75af756b677922085ae21ca8b4f48a2e654b7bdf0`.
  Gate Job `1167504` failed `127:0` before Python because the launcher used
  `python` before activating the OpenTAD environment. Submission receipt SHA
  is `80c230fec76d57e35b2e7f0cc89461729dc1646dd65c10f71af92a3ceca6c057`.
  The empty stdout and 78-byte stderr are infrastructure-failure evidence only.
- Failed recovery campaign `e647d6feff89cfd7`: certificate internal SHA
  `b76fa4afb9917452928612a9eeba38daa7152212eaf5efc63c9cd0a53fb766fc`.
  Gate Job `1167507` failed `2:0` before preflight because direct `sbatch`
  relocated `BASH_SOURCE` to `/var/spool/slurmd`. Submission receipt SHA is
  `8960f38ed31dfa3c8796824d4b48c9d805ec60a9a587579e340fe907c3660804`.
  No test/profile evidence was opened.
- Failed recovery campaign `bc9bacf31bae3749`: profile commit
  `04f8c28c85f333ea9b992c1e5bc4fade06f2fe06`, certificate internal SHA
  `77caf621f2c453fc90a627189727dde590a586134d1279be2f95b8b836e7d093`.
  No-open Gate Job `1167512` completed `0:0`; gate receipt SHA is
  `820a9721688f5de68fed9c48b6275058967f7eef93d1a9b7f4a883150ef345fa`.
  The one authorized serial matrix was Job `1167516`; submission receipt SHA is
  `273bb5269d68a0e8f90b936e61a4a4d6a176af89882f2fa7e7b0117cb5c3883e`.
  It failed in the first profile with a sparse-power-trace validator error.
  Matrix stdout/stderr/audit SHAs are
  `5d777a117d987eabf58aaa382c01a766008ceaabf6e0717ced3cb7622821f907`,
  `22661dd3357514ec7772ce03d9d1e9ebb2000a27278050eb00e316f1660c4ee7`,
  and `41079027f71bd5cfff9eca90a02331703ea6fc2a3ade2c81efa4abae1f9fb852`.
  The failed-attempt marker file SHA is
  `eeac13f9e4fa18c7d59b26972b5188dd50f3da8df6466a06968c776baf5c5edc`.
  No valid profile artifact or descriptor exists; these sources establish an
  infrastructure failure only.
- S1 power-cadence diagnostic Job `1167536`: clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_7e75b43_20260717_powerdiag`;
  output
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_power_diag_7e75b43/power_sampler_diagnostic.json`.
  Output file SHA-256 is
  `14c12730d488fefe6e95b8dc004667e271c7d86213045fca3f3ce28606cf8c45`;
  internal diagnostic SHA-256 is
  `596568ed0044ba35835e416650644c74a612529a7ac3a5feccf12085bd188ae1`;
  submission receipt SHA-256 is
  `e32675c3d50099074f58117eebf4466a18185189c8c448beee869a6e464cb6c4`.
  The artifact is test-blind and cannot itself support a paper cost claim; it
  establishes only the formal sampler backend choice.
- Formal S1 NVML recovery commit
  `2f8eb06f98ce35b61ce78b2b0cffa3eeb27a1b22`; clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_2f8eb06_20260717_nvml`
  passed 72 exact Linux tests. Chained campaign
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/spatial_zoom_s1_canonical/d95a36db4bc70aa2ac9d15e5fb5be82174a8a3488c5150c71d2ad4c10c7234a7/profile_campaigns/02f8e8bf7c2d6d25`
  has certificate internal SHA
  `e70cccc34a725beb2b899ae3f498bed5a03a430b7eed0f17a09fb06510381b3b`
  and file SHA
  `74ba2f5546f1c05142bc7aabbea964a0add26927dedc216b26a7c5fcb0d174b2`.
  No-open Gate `1167537` completed `0:0`; its power diagnostic internal/file
  SHAs are
  `037992cae297a1fe03356e6f461dad9cd5b5af681535e2525d2cd4b1956a4cdd`
  and `e271056e75d0f11bcadbcebbdcc1e4876e0d3e5b3b95b360b264143fb05cfdab`.
  Gate receipt internal/file SHAs are
  `a20341be651d1810b57428e62f603e3b16970dbbc6932390113bf56cee6f8a99`
  and `8f6f54c83515dcfdde549787f5e2a91b4c825663e9b40c646a9ad985e53dc581`.
  Serial matrix Job `1167538` has submission receipt internal/file SHAs
  `a20768d599c9eed4019dcaebd220e7eb4eb805e6b6043646dd8c82e19a45b448`
  and `bacf8b0f38efde7bcb0a5659978d8675c3004bf53151eb2a8ed2228840c2aa4f`;
  it failed in the first cell with `107147` NVML samples and a
  `2413.519286` ms maximum gap. Matrix stdout/stderr/audit SHAs are
  `f82d2a9d6f63969cce18687946a49a9219556dbdd45ed102e5abbcf3d5748f12`,
  `bce5003229c7f9ac14cc146ffb069bd9e0ddb6174487eb1be38351ab5ec2dae4`,
  and `300c4a3be1087e283470344497fd291b490bb565a8bc4b10e4981b737bd1162c`.
  The failed started-marker file/internal SHAs are
  `c9692531c3b12ec49fa4c72756d18fb1a39ebe860ce3fca3658442b10ea4ee65`
  and `1851fe1d8ead5225b2d37c6e9e7938cde74c29346795f6e1b997fcc38b8fb448`.
  No valid profile artifact or descriptor exists; these sources establish an
  infrastructure failure only.
- Out-of-process sidecar repair commit
  `35c7c5f6fdf2a85c7ecbadc2249f83476b7cdc3e`; clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_35c7c5f_20260717_sidecar`
  passed the exact combined S1+C3 Linux suite with `85 passed`. The earlier
  `c1253e6` snapshot's single failure was a test-only 150 ms process-start race
  and created no GPU Gate or evidence campaign.
- N16R4 resource-only diagnostics `1168504`, `1168506`, `1168509`, and
  `1168510` read no model, annotation, checkpoint, test evidence, or profile
  data. Their Slurm accounting establishes an outer `2 GPU / 8 CPU / 124400M`
  allocation and an inner exact `1 GPU / 5 CPU / 96000M` step. Job `1168510`
  measured the nearest finite cgroup v2 `memory.max` as `100663296000` bytes.
  Diagnostic `1168508` failed because `python` was unavailable and is retained
  only as an infrastructure error. A provisional campaign
  `f2b616846cad6bc2` was generated before the resource-scope code change and
  was never used for a Gate; it is superseded and cannot authorize execution.
- Independent follow-up implementation audit, agent
  `019f6ee1-aa1a-7912-a33b-904bec6f883a`, 2026-07-17. The reviewer read the
  current uncommitted matrix-binding diff, independently reran
  `99 passed, 5 skipped`, Python AST, Bash syntax, and diff whitespace checks,
  found no P0/P1, and returned `DEPLOY`. This is code-review evidence only; it
  does not replace the clean remote Linux/CUDA replay, recovery certificate,
  no-open Gate, matrix completion receipt, or S1 result analysis.
- Final S1 profile/evidence commit
  `5bfdc368304ec13a53b4571cd87c7d1f043c821a`; clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_5bfdc36_20260717_matrix_binding`
  passed `104` exact Linux tests. Campaign `e3fccb9b12a5d24d` certificate
  internal/file SHAs are
  `0a3203ddd57f79c4438194b5e85cd6ae6eeab4b567489c787fff1467675020c6`
  and
  `9953a73b7e2fdb4b61d89e4e91fc8e5566d0eaaf1de18e5e233c8d95aca40792`.
  No-open Gate Job `1168608` completed `0:0`; Gate internal/file SHAs are
  `1fb199b29def8d3e5c6feb6a74b10ff5fdd335cbbc5405c931f8dac2eca8557f`
  and
  `57e10f86d32ef4699ae1f891e87d343c4b1ad60db7083f89ec988604267d3493`.
  Gate submission receipt internal/file SHAs are
  `0e4bc5e05486f934923fa1f481fc9008a942ebbf2cf9acfb21ba6b5942c91c7e`
  and
  `23e3012ccb7cc35c4a3face248527c6d6203ca0c4fe2c00209035ad2facbabc8`.
  The sole serial matrix is Job `1168823`; its submission receipt internal/file
  SHAs are
  `7cfb3205252746fbb6fbdf5e6dd7d2b74d68d9338399c2b05124228f5752d985`
  and
  `7b5683e059f5680ae76fadf210fb384d5132cfd98b3361951723f89f704db694`.
  These establish deployment and Gate validity only; final matrix evidence is
  pending.
- Formal v3 matrix failure, Job `1168823`, 2026-07-17. Slurm accounting:
  `FAILED 2:0`, elapsed `02:33:27`, inner-step MaxRSS `72084384K`.
  Immutable first-cell attempt:
  `profile_campaigns/e3fccb9b12a5d24d/dense256/seed3408/`
  `dense256_seed3408.power_attempt.json`, internal SHA
  `6e2a9004f07ce30d04612692a20e22ece548195b8c6e1520c45655b71a9ebc4a`;
  raw trace SHA
  `ca60203051d5a44cae957cf77e8620f1d34124a5d7cef7750c9292caf6c033c0`.
  It contains `112107` samples and median/P95/max gaps
  `20.000005/20.023177/146.048168` ms; three gaps exceed `100` ms. The
  campaign contains no descriptor or completion receipt.
- Independent v4 implementation audit, agent
  `019f6ee1-aa1a-7912-a33b-904bec6f883a`, 2026-07-17. Pass one returned HOLD
  with three P1 findings: official matrix-start validation, healthy-child
  cadence isolation, and parent/current code plus trace-mode binding. Pass two
  returned HOLD for one remaining raw-trace lifecycle P1. After repair, pass
  three independently ran the exact combined suite (`102 passed, 5 skipped`),
  rehashed negative cases,
  `py_compile`, and `git diff --check`; it found no P0/P1 and returned
  `DEPLOY`. The reviewer made no file changes. This is code-review evidence
  only; clean commit, remote Linux/CUDA replay, v4 campaign, and no-open Gate
  remain pending.
- V4 buffered-sidecar deployment, 2026-07-17. GitHub commit `bc9350e` clean
  remote snapshot passed `107` exact Linux tests. Campaign
  `6021eaba62337726` certificate internal/file SHAs are
  `60c4a55813ce29bdabcfa378db2f91a70e01c45db366927fe2b7c86d95590dc6`
  and
  `10479cc65fbb291ceb726a1bb1463b96665f0547e893c99e345f4d0faf2f49c1`.
  Gate Job `1170341` failed `6:0` in six seconds before profiling; submission
  receipt SHA is
  `4fd692eb836637a72e15e2a220641d31d281477cf71d7ae01e08c0328dc4b9dc`.
  Resource-only Job `1170342` recorded `SLURM_STEP_GPUS=1`,
  `CUDA_VISIBLE_DEVICES=0`, `nvidia-smi -i 1` return code 6, and
  `nvidia-smi -i 0` UUID
  `GPU-2a3d0dd8-f85f-7e9e-30fa-d01c759694c8`. These are infrastructure
  diagnostics, not model, accuracy, or cost evidence.
- Physical-slot/cgroup-selector independent review, 2026-07-17. Read-only
  agent `019f70bc-5b43-71c0-9ca6-1122cb880eaf` audited the four changed
  code/test files and the corresponding contract/wiki records. It returned
  `DEPLOY` with no P0/P1, confirmed physical Slurm provenance remains distinct
  from the cgroup-visible NVML selector, and made no file changes. This
  authorizes only commit, remote replay, and a fresh no-open Gate.
- Cgroup-selector deployment evidence, 2026-07-18. Profile/runtime commit
  `43ac70bea6720f0c882ed3208bccfc89b089b5d4` passed `107` exact remote
  tests. Campaign `20fe22c380fd38bd` certificate internal/file SHAs are
  `455a0bbf799a091a6d17edc084930a71aa243a4ad34a5e909b3d8e1a286e7d7d`
  and
  `c984dba860ab9a8b6a0dbfb097435f0de486de5dd67d8fc04e8d7670cc96e1f1`.
  Gate Job `1170433` completed `0:0`; Gate internal/file SHAs are
  `38d07cdaf84969cb3a63862bd9a20b94a085bc2b4a05db0a373c1ceeca3ae46d`
  and
  `84499633d389e3dfb0a8402a63c4dc66ee6e22da0ecd3c20f88926170a28fe4d`.
  Matrix submission receipt internal/file SHAs are
  `5d09a3768fd985a24cbf05f95691a9f79c8622fc7a42d82664e680504c85a6c4`
  and
  `6501cc586aab889b4ef7173c21474afa76dbe07efc621c4a87063d3c021aead5`.
- Formal matrix runtime failure, Job `1170468`, 2026-07-18. Slurm accounting:
  `FAILED 1:0`, elapsed `03:05:47`, inner-step MaxRSS `77577404K`.
  Matrix-start internal/file SHAs are
  `e6199926c0364871fbad887c60ea05f7feb12d5571eac75b0233b35dca517968`
  and
  `b88827b236528f8f06c1a5e312ed316f6f88995c39f6548873bf435542642ba4`.
  The only descriptor is
  `descriptors/dense256_seed3408.run.json`, internal SHA
  `7b029470a52218f8f75bfbc499545c44ddf11785c2deca6c9394b1374a39e2ea`.
  The next cell failed in training snapshot `tools/test.py` with
  `formal S1 execution requires one auditable SLURM_JOB_GPUS identity`.
  No completion receipt or ordinal-1 test evidence exists. The one descriptor
  and profile remain diagnostic evidence within an immutable failed campaign.
- Step-scoped test-runtime recovery design,
  `docs/superpowers/specs/2026-07-18-spatial-zoom-s1-step-scoped-test-runtime-recovery.md`.
  It freezes model/test/cost/statistical invariants, rejects Slurm-variable
  rewriting and in-place historical-source edits, and requires a recursive
  recovery certificate plus independent P0/P1 audit before any new Gate.
- Step-scoped runtime independent read-only audit, 2026-07-18. The reviewer
  personally checked the complete working-tree diff and remote Job `1170468`
  evidence, ran the focused suite, and returned
  `DEPLOY_READY_WITH_GATES` with no P0/P1. Its P2 broad parent-delta allowlist
  and P3 split-field stdout concerns were fixed before commit; focused
  verification then passed `107` tests with `5` environment skips, and the
  reviewer reconfirmed no P0/P1.
- Step-runtime clean replay and environment diagnosis, 2026-07-18. Runtime
  commit `6524e1b698fd3e90f2621a8afd6a59b74395f972` passed `112` exact tests
  in clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_6524e1b_20260718_step_runtime`.
  The first v5 build failed before campaign publication because the login
  shell resolved user-site NumPy `2.2.6`; the formal selection environment is
  Conda NumPy `1.23.5`. Recomputing with `PYTHONNOUSERSITE=1` exactly matched
  every stored candidate metric, proving environment contamination rather
  than evidence drift.
- Step-runtime environment-binding audit, 2026-07-18. The v5 certificate and
  all three formal launchers now bind/verify `PYTHONNOUSERSITE=1`,
  `site.ENABLE_USER_SITE=False`, NumPy `1.23.5`, and the fixed Conda NumPy
  path. Focused verification passed `108` tests with `5` environment skips;
  independent review returned `DEPLOY_READY_WITH_GATES` with no P0/P1 and
  confirmed v1-v4 plus `verify_checkout=False` offline validation remain
  compatible.
- V5 real-evidence deployment, 2026-07-18. Environment-pinned runtime commit
  `3d01d3b7fc956ae17568ac3c8c04f9d6f36c42c5` passed `113` exact tests in
  clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_spatial_zoom_s1_3d01d3b_20260718_env_pinned`.
  Campaign `3180634880aa8de0` certificate internal/file SHAs are
  `a6d48b7c2fc0b162750ad5b215fcd1e46bd8fa3fb98b078beb7a2cf5aceaee20`
  and
  `352bd5cceb0659769c96f53c8b1c9b9329c1d43c6f42c429eb5bde794c0a0137`.
  Its sole no-open Gate is Job `1170765`; submission receipt internal/file
  SHAs are
  `d2d48a84180c2c75014c50614ba578c0d20efd1b0e58a87d263fa7eb5f712563`
  and
  `0e1968ee1f431649675db81a4e0afa7feffb88bd6d3e00f5267089b01bc33853`.
  The Gate entered the exact inner `1 GPU / 5 CPU / 96000 MiB` step on `g0024`
  but failed `2:0` before sidecar startup. Stdout/stderr file SHAs are
  `8345aff79068bf9d5382a7256c35366711e4fc9fbbc2c7e1fc35bf4af4520282`
  and
  `c510b037dfd59bb67c0c2312689a6fd143011390b64eac1f9c43f10832e78154`.
  The campaign published no sidecar/profile/descriptor/matrix artifact, and
  reused official-test evidence remained at SHA
  `10c0182d6fae42f37dec108988f22fbfd732725fc270426121ff2608837261e9`.
  This is immutable failed infrastructure, not an S1 result.
- V6 schema-family compatibility implementation, 2026-07-18. The local diff
  replaces literal v4 checks with an exact buffered-sidecar capability
  predicate and adds a recursive certificate binding Job `1170765` receipt,
  failure logs, exact four-file parent inventory, sidecar absence, and
  unchanged official-test evidence. Focused schema-compat verification passed
  `52` tests; the combined S1/train-engine/required-C3 suite passed `157` with
  `5` environment skips. Read-only reviewer
  `019f737e-a382-7901-a1a1-1673a98193eb` first returned HOLD on two P2
  test-sufficiency gaps. After exact double-stderr, hard-link, validator alias,
  legacy/power mixing, and incomplete-role regressions were added, its final
  verdict was `DEPLOY_READY_WITH_GATES` with no P0/P1/P2. Clean commit, remote
  runtime commit `cef95485d1bfebccddb1055f30800ab081decaf7` was pushed.
  Remote replay and a real-evidence v6 certificate remain pending; no new Gate
  or matrix is authorized.

## 外部附件

主任务显式引用的关键附件包括：`5f9a0d62...`、`86b473c6...`、
`d0087ae1...`、`1705e957...`、`60cb3e7a...`、`0ce290f9...`、
`a885a659...`、`391f061f...`、`c2008dfb...`、`c8a36eba...`、
`1fc36774...`、`d8b9f9fc...`。其中最后一份 ResearchClaw 审查已原样归档，
SHA256 为 `E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`。
