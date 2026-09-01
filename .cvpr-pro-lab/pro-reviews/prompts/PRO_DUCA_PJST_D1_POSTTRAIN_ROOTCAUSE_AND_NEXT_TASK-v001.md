# DUCA PJST-D1 完整训练后根因分析与下一任务冻结

Nonce: `DUCA-PJST-D1-POSTTRAIN-ROOTCAUSE-NEXT-v001-20260827`

Exact Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）  
GitHub repository: `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`  
GitHub branch: `codex/duca-pjst-cycle4-builder-20260826`  
公开模型/训练 revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`  
本地 clean 终结器 revision: `4204937a933c7a48854b623efefc7fd662e98805`（在上述分支前进 2 个仅终结器提交，尚未同步 GitHub；不改变模型）

你是本轮 Scientific First Author、Primary Research Owner 和最严厉的结果审稿人。请直接阅读公开 revision 的
PJST-D1 模型、配置、训练路径和已有 Wiki；根据下面的原始结果与故障边界，给出唯一
`CONTINUE / REVISE / PIVOT / STOP`。不能把科学选择退回给人类或 Coordinator。

## 项目级科研规则（本轮必须采用）

1. 工作必须面向论文问题、模型创新、完整真实训练、官方评测和决定性结果；不得把复杂合同、通用框架、
   防御性工具、版图整理或重复审计当成研究目标。
2. 只有会改变模型行为、数据合法性、公平比较、指标真实性，或确实阻止训练/评测的问题可阻塞实验；
   其余工程问题不得终止科学路线。
3. 实现、环境、启动器、收据或封存失败不是科学失败，只能触发最短必要修复。路线转换前必须完成旧路线
   的失败根因、混杂因素和可证伪结论闭环。
4. 你可以直接设计并分配本轮 Builder、Critic、Evaluator 的任务边界、先后关系和验收条件；分工必须服务
   最短科学闭环，不得增加与模型和实验无关的角色、审批或文档层。
5. 正式实现和完整训练后必须由你判断：实现是否忠实、成功或失败原因、是否可发表、是否达到最终实验、
   下一项针对性优化或路线转换。请给每项任务明确截止时间。

## 冻结科学问题与方法

H65 使用低成本 ASFormer scout 学习动作性与边界语义，经确定性 transport 间接选择有序、非均匀的真实
RGB 帧。PJST-D1（Derivative-Only Physical-Jacobian Scaled Tubelet）只在首次 VideoMAE tubelet 混合前保留
普通 pair mean，并按 `canonical_gap / physical_gap` 重标定相邻帧差分；canonical uniform 在任何浮点时间计算
前直接旁路原 PatchEmbed。首个 estimand 是固定/重放 selector 下的表示效应：OFF/ON 使用同一 Stage-1、
positions、RGB、mask、K=384、seed、训练更新、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split 和 evaluator。

本轮不得把 dynamic K、support-weighted mean、continuous cliplet、Query/Bridge、UVT、SingleClock 或新的 selector
并入 PJST-D1。它们可作为背景，但不能制造无法归因的新实现。

## 已完成的真实完整训练

- clean model/training revision: `c73e8418...`
- THUMOS14 全量 official validation: 211 videos
- matched Stage-2: seed 3407, K=384, 60 epochs, 6000 successful updates
- 两臂从同一 Stage-1 epoch-29 EMA 起点严格加载；每 5 epoch 保存可恢复 checkpoint；正式读取 epoch-59 EMA

终端官方点估计：

| 指标 | OFF | PJST-D1 ON | ON-OFF（百分点） |
|---|---:|---:|---:|
| Avg-mAP | 65.06328323 | 64.59080197 | -0.47248126 |
| mAP@0.3 | 80.04698811 | 79.25176714 | -0.79522097 |
| mAP@0.4 | 75.56871469 | 74.31627020 | -1.25244449 |
| mAP@0.5 | 68.02175108 | 67.87476664 | -0.14698444 |
| mAP@0.6 | 58.03293531 | 57.74244005 | -0.29049525 |
| mAP@0.7 | 43.64602698 | 43.76876583 | +0.12273884 |

该点估计不满足正向支持门；但没有合法成对置信区间，因此尚不能签发冻结的 PASS/KILL，也不能声称 PJST-D1
已被科学否定。单 seed 整视频 bootstrap 只刻画视频抽样不确定性，不等于训练 seed 稳健性。

## 当前唯一运行阻塞与证据边界

训练评估只在内存中计算标量，没有保存逐视频 predictions。为执行预注册的 10,000 次整视频成对 bootstrap，
本地 clean 终结器提交 `45496b8a` 与 `4204937a` 只增加 OFF/ON 只读重推理、分片 bootstrap 和 merge，不改变
模型权重、数据、NMS 或 evaluator。

replacement eval jobs `1257283/1257284` 均在正式推理前失败，远端日志共同报错：

```text
FileNotFoundError: pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth can not be found.
```

根因是终结评估 launcher 未把 canonical absolute VideoMAE-S pretrain 路径覆盖进运行配置。没有生成合法 sealed
predictions、paired CI 或 gate；dependency job `1257285` 因前项失败不能形成统计终稿。它是确定性的评估绑定
故障，不是新模型结果。最小候选动作仅是在同一终结器中绑定已冻结的绝对 pretrain path，然后重跑同一
OFF/ON 只读评估和同一 bootstrap；不得重训或改模型。

## 你必须完成的结果后复盘

请以严厉审稿人和第一作者双重视角回答：

1. 当前 matched 实现是否足以忠实估计 derivative-only 首次混合效应；公开代码还需核验哪些关键事实？
2. 在没有 CI 前，负向点估计允许什么、不允许什么结论？完成 paired bootstrap 是否仍会改变下一步决策；
   若会，是否授权上述唯一 launcher-binding 修复；若不会，给出停止它的统计理由。
3. 对 `低 IoU 下降、高 IoU 近持平/轻微上升` 给出有证据优先级的多假设根因树，至少区分：
   表示尺度、梯度/优化、gap 分布和短动作、selector 固定所限制的中介、checkpoint/EMA、统计波动及实现错误。
   明确哪些可由现有 artifact 只读分析，哪些必须通过新的决定性实验区分，禁止把相关性写成因果。
4. 判断本轮是否已经构成论文可发表结果、诊断性负结果，或最终实验；明确最窄 claim、anti-claim 和 invalidator。
5. 在切换路线前，冻结最少的失败根因分析。若仍需实验，只允许一个最便宜、最能区分根因且不重复历史矩阵
   的下一实验；若无需实验，明确为什么已有证据足以 PIVOT/STOP。
6. 评审并直接指定 Builder、Critic、Evaluator 的职责和验收顺序。工程只允许最短路径；不得设计新的合同
   框架、审计体系或文档链。

## 必须返回的终稿

第一行仅为 `CONTINUE / REVISE / PIVOT / STOP`，随后给出：

1. 对本轮实现忠实度与因果识别的裁决；
2. 对当前点估计和缺失 CI 的证据分级；
3. 成功/失败根因树及可排除项；
4. 是否授权唯一 absolute-pretrain-path 终结器修复与 exact rerun；
5. 完成 CI 后的结果条件分支，但最终只指定一个当前动作；
6. 是否达到论文主结果/可发表负结果/最终实验，以及缺失证据；
7. 若继续，唯一下一模型或实验、最小改动面、正式数据/seed/update/evaluator/cost/stop/claim 边界；
8. 你定义的 Builder -> Critic -> Evaluator 分工和每一棒的验收条件；
9. `current_scientific_question / next_owner / next_action / dependency / expected_return_at / single_recovery`；
10. 明确研究里程碑截止时间，默认不得晚于 `2026-08-28T12:00:00+08:00`。

不得声称 PJST-D1 已提高性能、已降低成本、已 paper-ready，或把上述评估绑定失败写成科学否定。
