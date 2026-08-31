---
type: review_index
date: 2026-08-31
scope: complete DUCA research history and public implementation lineage
---

# DUCA GitHub 深度审查入口

本页供独立科研审查使用。研究判断应从完整 Wiki 历史、精确提交代码与正式结果记录
共同得出，不能只依赖当前摘要，也不能把本页的链接顺序理解成预先选定的路线。

## 完整研究记忆

- 仓库：<https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702>
- 完整 Wiki 同步分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831/research-wiki>
- Wiki 总入口：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/index.md>
- 当前状态压缩包：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/query_pack.md>
- 防重复记录：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/anti_repetition.md>
- 决策历史：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/decision_history.md>
- 实验目录：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831/research-wiki/experiments>
- 思路目录：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831/research-wiki/ideas>
- 来源登记与原始材料：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-wiki-complete-sync-20260831/research-wiki/sources>
- 全部远端分支：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/branches/all>

## 本轮 Gemini 全量预审

- Gemini 3.7 Flash high-effort 完整只读报告：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md>
- 该报告只是独立咨询。其未隔离的根因判断、数值阈值、里程碑和路线偏好均须由 Pro
  结合原始代码与正式证据重新裁决。

## 本轮 Pro 全历史裁决

- Pro 完整报告：
  <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/codex/duca-wiki-complete-sync-20260831/research-wiki/sources/2026-08-31-pro-github-wiki-comprehensive-review-v002.md>
- 裁决为 `REVISE`。当前唯一任务是完整数据身份审计；模型实现尚未授权。数据通过并返回 Pro 后，才条件解锁固定
  K384 与 K256/K384/K512 多预算训练暴露的两臂、三种子、完整训练和一次性完整 held-out 比较。

## 关键实现谱系

下列链接用于逐版本核验，不代表未列出的提交可以忽略。完整细节见
`duca_model_version_registry.md`、`worktree_inventory.md`、`decision_history.md` 与实验页。

| 阶段 | 精确公开代码 | 审查重点 |
| --- | --- | --- |
| Transition / CellCF 证据树 | [`4ce69c852bdbd902046b47bc6019ae11e850dbe4`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4ce69c852bdbd902046b47bc6019ae11e850dbe4) | 早期全局选择、CellCF 与正式负结果 |
| Protected end-to-end | [`b3222af0895e23eca83113977c1bcfad75258c9e`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b3222af0895e23eca83113977c1bcfad75258c9e) | 受保护梯度、物理选择与同伦失败 |
| Global curriculum | [`63e25eb17e523d369f73434ed4d9b6446608861a`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/63e25eb17e523d369f73434ed4d9b6446608861a) | 全局 exact-K、课程训练与 P0 修复 |
| H65 clean fixed-K384 | [`04c35a3b76897e6c1569eeede41ed3aecaf7f854`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/04c35a3b76897e6c1569eeede41ed3aecaf7f854) | 当前可靠稀疏基线和两阶段训练基座 |
| TrueTime | [`11126684af779aa2916a68ecf617c4f14c805478`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11126684af779aa2916a68ecf617c4f14c805478) | 显式物理时间残差的匹配诊断 |
| Semantic dynamic cycle | [`d80022e963a8ad21d390c785cbd8a4c23f41484a`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d80022e963a8ad21d390c785cbd8a4c23f41484a) | 语义动态预算历史实现与恢复边界 |
| Query-bridge / full-official lineage | [`46c714249ff444fcc6428dbe95c52aefe55c488f`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46c714249ff444fcc6428dbe95c52aefe55c488f) | 查询桥和后续外部审查记录 |
| Native tubelet coreset | [`b33391126eac05e3353d322b973dda91741f0732`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/b33391126eac05e3353d322b973dda91741f0732) | 连续 tubelet 输入、uniform 与 coreset 结果 |
| Dynamic native-tubelet budget | [`d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610) | 真正变长 heavy execution 的窗口级尝试 |
| Temporal Coverage | [`048143124e2a36a76575200ae17d6f42ec79ea3a`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/048143124e2a36a76575200ae17d6f42ec79ea3a) | 设施位置覆盖与预运行机制门负结果 |
| Marginal budget | [`f67d96fdf68a295eaa7f678f3dfc125530828889`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889) | 加性边际效用与 training-side oracle |
| Cap-release neighborhood | [`46812facc8773d9b4a9c21833cbe397c8aaa5a2d`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d) | 解除饱和预算上限后的联合门诊断 |
| Whole-video 704-state falsifier | [`33e4ed137c33eef07f0452b44506a6993bdf7535`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/33e4ed137c33eef07f0452b44506a6993bdf7535) | 冻结检测器三档预算动作空间的终态负结果 |

## 审查边界

1. 先读完整 Wiki，再逐项打开精确提交和相关实验页；不得只读当前状态摘要。
2. 对每条路线分别区分：设计、代码存在、局部测试、工程中断、正式性能证据和论文可用主张。
3. 历史训练侧 holdout、内部 oracle、局部 smoke 和中间验证均不能替代完整训练集与一次性完整留出评估。
4. OpenTAD 的 211-video `validation` 与 ActionFormer 的 212-video `test` 身份仍未准入；在解释所有差集前，不得启动模型实现或正式训练。
5. Pro 应独立整合路线并下达唯一下一项可证伪任务；Gemini 与 Codex 只提供证据和实现核验，不预先替 Pro 选线。
