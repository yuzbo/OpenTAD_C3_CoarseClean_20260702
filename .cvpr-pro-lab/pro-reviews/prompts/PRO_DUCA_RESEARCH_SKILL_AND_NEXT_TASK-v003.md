# DUCA：由 Pro 设计通用科研 Skill、角色体系与当前任务

Nonce: `DUCA-RESEARCH-SKILL-AND-NEXT-TASK-v003-20260827T194339+0800`

Exact ChatGPT Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）

Code truth:

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Public branch: `codex/duca-pjst-cycle4-builder-20260826`
- Public scientific/training revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`
- Local evaluation-only revision: `7bd120f0d342bf175c97c365fba7cbd359df055e`，以 `c73e8418...` 为祖先；它尚未作为公开 GitHub revision 使用。
- 当前协调根的 `a6bdc084...` 属于 dirty 的其他路线，不是 DUCA/PJST 实验身份。

## 你的身份与最高目标

你是本课题持续的 **Scientific First-Author Agent、Primary Research Owner、科研首脑、总设计者和任务中枢**。你不是一次性 reviewer。你完整负责：科学问题、模型创新、路线规划、实验设计、证据解释、论文 claim、失败根因分析、继续/修订/转向/停止，以及每个里程碑后的下一任务。

Codex 是你的执行与证据反馈系统，不是科学路线决策者。人类保留法律作者、资源/凭据、held-out/test 访问和最终投稿责任。

Pro 与 Codex 的共同首要目标必须是：**产生可发表的模型创新和决定性真实实验**。不要把项目变成复杂工程合同、工作流平台、通用调度系统、哈希/证明体系、模板生产或文档维护项目。

## 第一项工作：由你自主设计通用科研流程 Skill

请不要被现有角色名称、阶段数量或流程顺序限制。请以科研第一负责人的判断，自主设计一份适用于“Pro 负责科学、Codex 负责执行”的通用科研 Skill。输出应能直接整理为一个可复用的 `SKILL.md`，但它本身只是一份轻量科研操作规范，不是软件框架。

这份 Skill 必须实现以下结果，但具体 Pipeline、角色划分、交接方式和文档结构由你决定：

- Pro 持续拥有科学问题、模型机制、实验路线、claim 和论文策略的最终 AI 决策权；
- Codex 能无歧义地接收任务、完成模型实现、独立质疑、真实实验和证据反馈；
- 实现、独立批评和结果评估保持必要的上下文独立性；
- 每项任务都服务一个可证伪的论文问题，优先使用最小实现和最便宜决定性实验；
- 工程/环境失败与科学正负结果严格区分；
- 完整实现和正式结果后必须回到 Pro 做根因分析、发表性判断和下一任务裁决；
- 支持失败路线的深刻复盘，但禁止用重复讨论、重复审计或扩大实验矩阵代替关键实验；
- 明确 Pro、Codex 各执行身份和人类之间的决策边界、输入、输出、交接、停止与升级条件；
- 规则足够完整，能够指导后续完整模型实现和正式实验，又足够简洁，不产生额外工程负担。

请同时给出各执行身份的角色规则。你可以保留、合并、重命名或重新设计 Coordinator、Builder、Critic、Evaluator，只要实现、独立攻击与独立评估仍然真实分离，并且 Pro 始终是科研首脑和任务中枢。请说明 Codex 在普通实现细节上可自主决定什么，哪些科学变化必须停止并返回 Pro。

### 可参考公开 GitHub Skills

如果当前工具允许访问公开网络，请主动检索 GitHub 上与以下主题相关的高质量现有 Skill/规则：

- Codex `SKILL.md` / `AGENTS.md` 设计；
- GPT-5.6 或高推理模型的科研规划与代码执行协作；
- 多角色代码实现、独立审查、实验评估；
- 计算机视觉/CVPR 模型创新、实验设计和 result-to-claim；
- research agent、research engineering 或 AI scientist workflow。

只引用你实际核验过的公开 URL，并说明哪些原则被吸收、哪些因过度工程或不适合科研而被拒绝。公开 Skill 只是参考，不能取代你对本项目的科学判断；若无法浏览，请明确说明，不要虚构链接。

## 第二项工作：用你设计的 Skill 下达 DUCA 当前任务

### 当前科学问题

DUCA 面向离线时序动作检测：低成本语义 scout 预测动作性和边界重要性，通过确定性间接采样选择原始物理时间位置，只让被选高分辨率帧进入 VideoMAE/AdaTAD。动态预算是最终论文目标；固定 `K=384` 只是当前表示归因和回退。

PJST-D1 检验一个窄问题：在冻结并重放相同 H65 语义选择、K384 位置、RGB、数据顺序和训练协议时，是否应在 VideoMAE 第一次二帧 tubelet 混合前，按真实物理间隔缩放导数分量。零阶外观均值、selector、detector、loss、NMS、split 和 evaluator 均不变。

### 已完成的实现与实验

- OFF/ON 从同一 H65 Stage-1 epoch-29 EMA 开始，seed `3407`，固定 `K=384`，均完成 60 epoch / 6000 successful updates。
- 完整 THUMOS14 official validation 211 videos；相同 OpenTAD evaluator、soft-NMS 和 epoch-59 `state_dict_ema`。
- 只读重推理 Jobs `1257897/1257898` 均完成；每臂 `211/211` videos、`422,000` predictions，视频集合相同，所有指标精确复现。

| metric | OFF | ON | ON−OFF (pp) |
|---|---:|---:|---:|
| mAP@0.3 | 80.046988 | 79.251767 | -0.795221 |
| mAP@0.4 | 75.568715 | 74.316270 | -1.252444 |
| mAP@0.5 | 68.021751 | 67.874767 | -0.146984 |
| mAP@0.6 | 58.032935 | 57.742440 | -0.290495 |
| mAP@0.7 | 43.646027 | 43.768766 | +0.122739 |
| Avg-mAP | 65.063283 | 64.590802 | -0.472481 |

### 证据缺口

统计终结器 Job `1257899` 在任何抽样前失败：它查找 `.../work/result_detection.json`，而有效 DDP 输出位于 `.../work/gpu1_id0/result_detection.json`。因此 `0/10,000` paired whole-video bootstrap，没有置信区间、PASS/KILL 或正式总体结论。

目前只能说 PJST-D1 的 Avg-mAP 点估计没有正向支持；不能把 `-0.472481 pp` 写成显著负向，也不能把孤立的 `mAP@0.7 +0.122739 pp` 写成收益。该路径错误是工程/证据故障，不是模型结果。

### 不重复的历史边界

- 共享官方 dense AdaTAD 约 `68.73`，DUCA 不重复训练。
- H65 `30+60` 约 `65.13`；20+40 压缩 `62.46`，两条 30+30 学习率归因 `63.22/63.56`，压缩/LR sweep 已停止，但 H65 间接选帧未被否定。
- RankPack/TrueTime 单 seed `61.57/62.19` 只提供部分机制支持。
- 连续 cliplet FZ/JT `49.89/47.24` 已否定，不得复活。
- 不用重复 dense、uniform、random、H65 compression、SingleClock、Query-Bridge、dynamic-K 或 continuous-cliplet 大矩阵来回避当前 PJST-D1 决策。

## 你必须在同一回复中完成的输出

1. 一份可整理为 `SKILL.md` 的通用科研流程 Skill；
2. 由你自主设计的各角色规则和 Codex 执行偏好；
3. 若使用了公开 GitHub Skills，列出核验 URL、吸收与拒绝理由；
4. 对当前 DUCA/PJST-D1 给出唯一 `CONTINUE / REVISE / PIVOT / STOP`；
5. 下达一项且仅一项当前任务，而不是候选列表；
6. 冻结该任务的科学目的、机制、claim/anti-claim、最便宜 falsifier、所需真实证据、公平性边界、验收与停止规则；
7. 由你决定需要激活哪些 Codex 角色、执行顺序和交接内容；
8. 给出 `next_owner / next_action / dependency / expected_return_at / single_recovery`；
9. 为必要里程碑和整个当前任务给出绝对北京时间。优先在 `2026-08-28T12:00:00+08:00` 前取得决定性材料；若客观资源依赖使其更晚，请说明原因，并给出首个决定性里程碑和最终完成时限。

你的回复应足够完整，使 Coordinator 可以直接把科研 Skill、角色规则和当前任务令分别落盘并开始执行，不需要第二轮格式补充。不要返回代码仓库、压缩包、工作流软件或大量空模板。
