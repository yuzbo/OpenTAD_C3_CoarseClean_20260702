# 证据来源与覆盖图

## 1. 仓库级权威来源

| 来源 | 用途 |
| --- | --- |
| `docs/superpowers/specs/2026-07-10-phystime-tal-design.md` | PhysTime-TAL 1.0 原始设计与停止条件 |
| `docs/superpowers/specs/2026-07-10-phystime-tad-2-design.md` | PhysTime-TAD 2.0 support-integrated 最终算子合同 |
| `docs/superpowers/specs/2026-07-11-phystime-adatad-1-design.md` | 当前 raw-video AdaTAD 三头比较合同 |
| `docs/superpowers/plans/2026-07-10-phystime-tad-2.md` | feature-geometry 实现任务与 gate |
| `docs/superpowers/plans/2026-07-11-phystime-adatad-1.md` | raw-video 实现计划与已执行任务锚点 |
| `docs/methods/phystime_tad_contract.md` | 秒坐标、support provenance、no-leak runtime contract |
| `docs/evaluation/results.md` | 实验数字与 gate 状态唯一来源 |
| `docs/evaluation/phystime-performance-drop-diagnosis.md` | PhysTime 1.0 负结果的因果诊断与下一版 gate |
| `docs/evaluation/EXPERIMENT_AUDIT.md` | 独立只读实验完整性审计与整改状态 |
| `docs/evaluation/phystime-tad-track.md` | 已取消 feature-token track 的历史部署协议 |

## 2. DUCA/C3 评审与吸收记录

以下文档已全部纳入 timeline、decision register、lessons 和 idea pages：

- `docs/methods/2026-07-06-detector-aware-truetime-cvpr-route.md`
- `docs/methods/2026-07-06-detector-utility-route-review.md`
- `docs/methods/2026-07-06-duca-09b0d31-pro-review-absorption.md`
- `docs/methods/2026-07-07-46cacc1-cvpr-hold-review-absorption.md`
- `docs/methods/2026-07-07-46cacc1-pro-final-route-review-absorption.md`
- `docs/methods/2026-07-07-46cacc1-selector-geometry-hold-review-absorption.md`
- `docs/methods/2026-07-07-c69c1a0-paction-gasvt-hold-review-absorption.md`
- `docs/methods/2026-07-07-cvpr-intelligent-acquisition-no-leak-review-absorption.md`
- `docs/methods/2026-07-07-gasvt-diagnosis-shift-detector-utility-absorption.md`
- `docs/methods/2026-07-07-gasvt-paction-diagnosis-evidence.md`
- `docs/methods/2026-07-07-gasvt-plateau-paction-advantage-review-absorption.md`
- `docs/methods/2026-07-07-observation-grid-vs-raw-frame-pro-review-absorption.md`
- `docs/methods/reviews/2026-07-07-duca-acc6960-hold-review-absorption.md`
- `docs/methods/2026-07-08-46cacc1-visualization-hold-review-absorption.md`
- `docs/methods/2026-07-08-52ab63b-visualization-hold-review-absorption.md`
- `docs/methods/2026-07-08-fbea37b-learned-context-radius-hold-review-absorption.md`
- `docs/methods/2026-07-09-544eca6-final-duca-complete-model-review-absorption.md`
- `docs/methods/2026-07-09-duca-jct-progressive-deployment.md`
- `docs/methods/2026-07-10-88e50b1-duca-final-method-audit-review-absorption.md`
- `docs/methods/2026-07-10-duca-full-window-final-repair-status.md`
- `docs/methods/duca_online_plugin_contract.md`

对应 `docs/methods/reviews/*.raw.txt` 保留逐字原始评审，吸收记录不替代原文。

## 3. 本轮回顾的原始附件

附件没有复制进仓库；使用绝对路径和 SHA256 固定来源。若附件目录未来被清理，应先把该原文迁入受控归档并保持 hash。

| 附件 ID | SHA256 | 讨论主题 / 已吸收内容 |
| --- | --- | --- |
| `5f9a0d62-0671-4af6-9595-6c8b2f2cfca9` | `922CE63663302204C369FBA9D83468BE2A91F5886F0FF1F059F96A19AC0A00B5` | 最终目标不再调 PAction/GAS-VT；DUCA center-radius、主实验与 joint 条件 |
| `86b473c6-1aa7-453e-b79d-b8c124cc91b6` | `CF8183CEAADE93F9D92570DDD2ED2205E38907A80F7D4D57C0535DB3B0226924` | frozen/task-adapted/end-to-end/offline ledger 四路线比较与 task-adapted plugin |
| `d0087ae1-175a-4840-8fd1-ee8c8d41ceab` | `A61226E4D33399240C3F52745E8081110E24760260E66E5DACC650EFC3C6EA37` | DUCA final plugin、zero-shot source、ST joint、method contract 与 reviewer attacks |
| `1705e957-72c2-4261-a10c-83445bf8e238` | `7F5551348235D1CDF2A854510F8C5413E41BB618B51D3CD997931AAE1EFF15BA` | online plugin/X3D/selection eval/code/experiment 的严厉 HOLD |
| `60cb3e7a-a327-4a48-b1b6-165d897e204b` | `64794A57C54FF390AD679C6208BD929E7939A229B771A252111E8B7FC8B2B227` | continuous budget curve 不是动态预算；DUCA-MUST 设计与 Lagrangian 约束 |
| `0ce290f9-4dae-4382-b200-dd1f660311a0` | `B3A6A241D851566DE9AA3555A4111988E0DCC3DD1955794D92AB9F97C3FA37DD` | 当前实验不足、MUST/X3D/lattice 严查及最小论文矩阵 |
| `a885a659-8728-48c6-98f4-6241edd808f0` | `47247641F9ADBF7978FA4FFDB22BE57599F767C6BDFFEA42B1F6584ABF2F9508` | coarse probe + selector + official AdaTAD、hidden feature、joint gradient 与 optimizer |
| `f114558f-f390-4a6e-ba7b-7527d56734b7` | 同上 | 与 a885 内容重复，已去重，不重复计为独立建议 |
| `a1d1ebb7-6413-438d-973b-1046ad8016d0` | `BF8B586C4C49493615B740FA43AD43859F96160540D700E4BD9D5DB7C4E1C172` | hidden feature、proxy 命名、ST/official proof/max-gap、move 偏移诊断 |
| `391f061f-5fc1-43e2-ae74-504d145d90e0` | `2F73414564E83803FE3FFF6A9D44A6E60F60618EBE64FBF94921665E39356801` | 最终 DUCA 逐代码 patch：hidden fusion、soft/hard gap、official gradient gate |
| `c2008dfb-a265-468f-a84c-b66037805dc5` | `7C4207B30186179986F37CEF92F92059E6FDA0BB91C416560D58D7139D5555B4` | 7bea4fc HOLD、旧 run 低性能、dynamic collapse、主表/消融要求 |
| `c8a36eba-32f9-4524-9214-2c19bd3a7c0c` | `00107AC0E451A60DD66BCC623E04C4DF879554F24AFA53B99160FCE42E516966` | loss 重复、boundary proxy 非 boundary-first、hard/soft 错配、CFPA 修复 |
| `1fc36774-3c6e-461c-a5a0-d719ac99ecb5` | `9F87B12F4DE594C2E542973F20D22DF6F7D95AE3F5E7776BB38D8A8565D390B7` | PIVOT、23 个候选、ChronoTransport/CoDeR/ACTAL/PhysTime/No-Free-Frames |
| `e8d42e0b-ab00-43d4-8cbe-7d57f323f95a` | `A30B6D5547CD6891A4CA3A36C19651E0172ACF40B8D28F6904BD5EDE9E1CF302` | PhysTime 1.0 新颖性批判、support measure 2.0 与三类实验协议 |

附件根路径格式：

`C:\Users\skywalker\.codex\attachments\<附件 ID>\pasted-text.txt`

## 4. 代码提交锚点

| Commit | 含义 |
| --- | --- |
| `7e3a508` | 修复 DUCA max-gap scaffold 的旧主实验版本，结果仅诊断 |
| `88e50b1` | SlowFast Fast frozen-prior 诊断，不是主方法 |
| `70aa069` | DUCA structured curriculum DDP-static 修复末端 |
| `5a46ea6` | PhysTime-TAD 2.0 feature detector 核心实现 |
| `696f77d` 至 `1893004` | feature-track 部署、数据恢复与最终取消记录 |
| `9266ebc` | PhysTime-AdaTAD raw-video 设计规格 |
| `517785d` | PhysTime-AdaTAD implementation plan 锚点 |
| `2b7f83f` | 当前 PhysTime 分支首版 persistent Wiki |
| `92029ea` | ChronoTransport formal Stage-B 闭环与 P3 负 gate |
| `3ac93a1` | PhysTime-AdaTAD 最终稳定 matched full-run 实现 |

### 额外分支来源

- ChronoTransport 完整实现：本地 `codex/c3-coarse-clean-20260702`，commits `6e4bc54..92029ea`；对应 origin 仍在 `3554b6f`，本地 ahead 15。
- ChronoTransport final spec：`git show 92029ea:docs/superpowers/specs/2026-07-10-chronotransport-design.md`。
- ChronoTransport implementation/formal P3 record：`git show 92029ea:docs/methods/2026-07-10-chronotransport-implementation-plan.md`。
- ChronoTransport/C3 原始任务归档：`git show 92029ea:research-wiki/sources/thread-019f49d2-user-record.md` 与 `.../delegated-thread-recent-record.md`；现已复制到本 Wiki `sources/`。
- DUCA full-stack/structural audit：commit `a5e1774`，不在当前 HEAD ancestry。
- DUCA ResearchClaw raw audit SHA256：`E4344DAED297F02E23CE355A4B0BBA1845F2C05393820CE04843374AAB6A59AC`。

完整本地 worktree/branch/HEAD 快照见 `worktree_inventory.md`。原始讨论归档哈希见 `sources/README.md`。

## 5. 外部文献锚点

- mTAN: `https://arxiv.org/abs/2101.10318`
- ActionFormer: `https://arxiv.org/abs/2202.07925`
- Temporal Robustness Benchmark: `https://arxiv.org/abs/2403.20254`
- TE-TAD: `https://openaccess.thecvf.com/content/CVPR2024/html/Kim_TE-TAD_Towards_Full_End-to-End_Temporal_Action_Detection_via_Time-Aligned_Coordinate_CVPR_2024_paper.html`
- AdaTAD: `https://openaccess.thecvf.com/content/CVPR2024/html/Liu_End-to-End_Temporal_Action_Detection_with_1B_Parameters_Across_1000_Frames_CVPR_2024_paper.html`
- LiquidTAD: `https://arxiv.org/abs/2604.18274`

## 6. 覆盖声明

本 Wiki 已覆盖本线程中出现的目标争论、训练方式、zero-shot prior、固定/动态预算、max-gap、detector gradient、X3D/SlowFast、PIVOT/ChronoTransport/DCRT、两轮共 23+24 个发散候选、PhysTime 1.0/2.0、feature-track 取消、raw-video AdaTAD 头隔离和秒/帧坐标讨论。三份 `routes/*-complete-record.md` 进一步记录了跨分支实现与实验谱系。原始附件中的长代码建议没有逐行复制，但其方法决策、风险、停止条件、代码落点和可复用机制均已纳入。
