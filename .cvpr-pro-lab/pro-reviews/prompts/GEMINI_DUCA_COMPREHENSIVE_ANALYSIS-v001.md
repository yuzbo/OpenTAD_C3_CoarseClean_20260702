# Gemini 独立咨询：DUCA 全历史、代码与论文路径分析

你是 Pro 科学裁决之前的独立技术咨询者，不是本项目的最终科学负责人。请在只读模式下完整检查当前工作区、指定
Git 提交和研究 Wiki，给出简洁、证据化、可供 Pro 批判吸收的分析。不得修改任何文件、执行训练、访问 held-out
指标、启动浏览器或远端作业。

## 项目与代码身份

- Repository：`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`
- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- H65 clean base：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- Whole-video diagnostic：`33e4ed137c33eef07f0452b44506a6993bdf7535`
- Native tubelet：`b33391126eac05e3353d322b973dda91741f0732`
- TrueTime：`11126684af779aa2916a68ecf617c4f14c805478`
- Sparse reconstruction：`dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`
- Coverage：`048143124e2a36a76575200ae17d6f42ec79ea3a`
- Marginal：`f67d96fdf68a295eaa7f678f3dfc125530828889`

## 必须完整阅读的研究材料

1. `.cvpr-pro-lab/pro-reviews/materials/DUCA_COMPREHENSIVE_ROUTE_EVIDENCE-v001.md`
2. `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_COMPREHENSIVE_ROUTE_INTEGRATION-v001.md`
3. `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`
4. `PAPER_PROGRESS.md`
5. `research-wiki/decision_history.md`
6. `research-wiki/anti_repetition.md`
7. `research-wiki/query_pack.md`
8. `research-wiki/duca_final_model_contract.md`
9. `research-wiki/experiments/duca-multi-budget-detector-adaptation.md`
10. `research-wiki/experiments/duca-native-tubelet-coreset-fixed384.md`
11. `research-wiki/experiments/phystime-g1-matched-full60.md`
12. `research-wiki/experiments/duca-sparse-probe-and-coarse-backend-ablation.md`
13. `research-wiki/sources/2026-08-31-duca-irregular-temporal-sampling-external-proposal.md`
14. `research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md`

## 必须完整检查的代码

使用 Git 或指定 worktree 读取精确提交，不得把协调根的未提交代码当作身份：

- H65 worktree：
  `C:/Users/skywalker/.codex/worktrees/duca-full-data-identity-audit-v1-20260831/OpenTAD_C3_CoarseClean_20260702`
  - `opentad/models/duca/acquisition.py`
  - `configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py`
  - `configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py`
- Whole-video diagnostic worktree：
  `C:/Users/skywalker/.codex/worktrees/duca-whole-video-consistent-budget-falsifier-v1-20260831/OpenTAD_C3_CoarseClean_20260702`
  - `opentad/models/duca/dynamic_budget.py`
  - `tools/bata/run_duca_whole_video_consistent_budget_falsifier.py`
- TrueTime worktree：
  `E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculumV2_20260822`
  - `opentad/models/duca/true_time_residual.py`

如果需要核验其他路线，使用 `git show <commit>:<path>` 或只读搜索；不要切换或修改工作树。

## 分析问题

1. 以事实表核验所有主要路线的实现身份、实验预算、最终性能、证据等级和停止原因；纠正 Wiki、提示词或总结中的
   任何事实冲突。
2. 对 H65、TrueTime、native tubelet、sparse reconstruction、Coverage、Marginal/whole-video dynamic budget、curriculum/
   distillation 和 proposed multi-budget adaptation 做代码到机制的忠实度检查，给出 `file:line` 与符号。
3. 区分：已证实根因、最强但未证实假说、工程失败、协议混杂、真实科学负结果和完全未测试的想法。
4. 判断哪些路线已无继续价值，哪些只是未闭环，哪些有真正论文级信息增益；不得建议原样重复已完成实验。
5. 在完整 200-video training、完整 held-out evaluation、无泄漏、公平预算、sealed predictions、paired uncertainty 和
   真实成本约束下，提出你认为最合理的唯一下一机制实验。
6. 给出从当前状态到可投稿 DUCA 的最短路线：核心论文问题、最小创新、必要 baseline/ablation、三种子、成本、第二
   detector/数据集的解锁条件，以及失败时何时停止。
7. 当前 211/212 identity audit 是模型工作前置事实任务；不要绕过或声称已解决。

## 输出要求

保持简洁，但不能省略关键证据。输出以下部分：

1. `VERIFIED_FACTS_AND_CORRECTIONS`
2. `CODE_MECHANISM_AUDIT`
3. `ROUTE_DISPOSITION`
4. `ROOT_CAUSE_ANALYSIS`
5. `GEMINI_RECOMMENDED_SINGLE_ROUTE`
6. `MINIMAL_DECISIVE_EXPERIMENT`
7. `PUBLICATION_PATH_AND_STOP_RULES`
8. `QUESTIONS_PRO_MUST_DECIDE`

每个结论附 `file:line`、commit 或 job；无法核验写 `UNKNOWN`。最后一行只写：
`GEMINI_DUCA_ADVISORY_READY`

