# DUCA Query-Bridge Pro 审查吸收与本地核验（2026-08-20）

## 审查身份与证据边界

- 外部审查：Project 内独立 Pro 会话 `duca-project-query-review`；请求/解析模型均为 Pro；状态 `completed`。
- 原始逐字转录：`C:/Users/skywalker/.oracle/sessions/duca-project-query-review/artifacts/transcript.md`，SHA-256 `e293399465904e2b9151ace704c6286f41bcde17e29d2d884ccd4245ba807d04`。
- 用户提供副本：`C:/Users/skywalker/.codex/attachments/69e5ae9f-e9a2-4afb-afd2-7c7e9eb9bb63/pasted-text.txt`，SHA-256 `f1de2f65a6b5f8bb1c2ba70bb462f1e1787c5eda22afb8567495b6816d8b40df`。
- 本文只吸收并核验审查意见；不产生新的实现、数据访问、训练、成本或论文效能结论。

## 外部建议的可接受内核

审查建议为 `REVISE`，而不是停止 DUCA：Query-Bridge 只能增强低成本 scout 的动作性、起点和终点边界预测；确定性规则由这些语义预测导出位置价值，动态预算 K 与位置价值分开计算；固定 K 是第一阶段公平归因，动态 K 只能在该归因成立、实际重骨干计算随 K 变化并且成本可核验之后启用。

这与项目的人类科学合同一致：主方法不是小模型直接学习帧索引；固定 K 只作对照/回退；被选观测必须保留物理时间并在 NMS 前使用正确时间坐标。该原则接受为当前路线的解释与实现约束，而非新论文结果。

## 已由本地代码或已封存结果核验的事实

1. 历史 `65.385724` 不是公平的 60-epoch 比较。项目防重复记录明确它是 30 epoch uniform 全检测器加 60 epoch learned sampler/full detector 的 90-epoch 课程，并混入蒸馏、transport 梯度和 ASFormer 适配；因此只能保留为“语义非均匀采样值得重建验证”的历史信号，不能作为 selector 增益证据（`research-wiki/anti_repetition.md:2503-2525`）。
2. UVT 的混杂成立。`duca_uvt_value_portal_n16r4.py` 同时启用 `dynamic_B`、straight-through detector loss、256--512 的动态预算；outer-K 只用 actionness（权重 1.0，boundary 0.0），而 `value_mode` 又改动 value alpha、geometry loss 和 EMA loss（`C:/Users/skywalker/.codex/worktrees/duca-uvt-official-v2/configs/adatad/thumos/duca_uvt_value_portal_n16r4.py:36-86`）。因此 `off/geo/geo_ema` 只能检验组合，不能把差值归因给 V(t) 的任一单独作用。
3. UVT 的骨干时间语义风险成立。当前实现把按物理位置排序、但可彼此不连续的 selected frames 直接 reshape 成 16-frame VideoMAE clips，再在骨干之后才按物理位置插值回 768 点轴（`C:/Users/skywalker/.codex/worktrees/duca-uvt-official-v2/opentad/models/backbones/backbone_wrapper.py:177-212`）。这确实会让 VideoMAE 的 clip 内时序操作看见非连续观测；“会导致多少 mAP 损失”仍是假设，必须用受控实验验证。
4. Fovea 的所谓单变量链不成立。配置中 `query_only` 换 score source，`query_gt_mask` 加 GT mask，`query_cycle` 加 post-heavy feedback，`query_fovea` 同时启用 boundary quota 和 MMR；同时所有臂都写为 `dynamic_budget=True`（`C:/Users/skywalker/.codex/worktrees/duca-full-official/configs/adatad/thumos/duca_fovea_qb_thumos.py:22-52, 71-91`）。
5. Fovea 的四项损失问题成立：端点值 2 与动作内部值 1 最终都经 `gt_mask > 0.5` 变为同一二值目标；center 被拉向零；mean width 被正向最小化；所谓 diversity 只有正号 entropy、没有 query orthogonality（`C:/Users/skywalker/.codex/worktrees/duca-full-official/opentad/models/losses/fovea_losses.py:43-47, 87-112`）。此外 budget loss 对 batch 内 selected count 求和；当前配置 batch size 为 1，故它不是已观测的 batch-size 故障，但该公式不具备 batch-size 不变性，且会把每个样本拉回固定目标 K（同文件:106-108；配置:105-108）。
6. Pro 的“当前无新训练”表述不再是当前事实。UVT Job `1244840` 的三个 60-epoch 训练与 test 已完成：off/geo/geo_ema Avg-mAP 为 `57.35/55.93/55.92`（`C:/Users/skywalker/.codex/worktrees/duca-uvt-official-v2/research-wiki/experiments/duca-uvt-value-portal.md:31-42`）。Fovea 第一波 Job `1244851` 的五个 60-epoch 单 seed 臂也完成，`query_cycle=54.67` 为该开发矩阵中最高，但缺同提交 dense/exact-uniform/random 和余下两臂（`C:/Users/skywalker/.codex/worktrees/duca-full-official/research-wiki/experiments/duca-fovea-qb-development.md:21-29`）。这些是真实的诊断性结果，不是可升级论文主张的 matched evidence。

## 吸收结论：不是“完全认可”

**实质接受**：停止将现有 UVT/Fovea 直接选择实现作为主方法；保留它们和结果作为可复核的负/诊断证据。下一候选必须让 Query 只调制 actionness/start/end 语义，且由确定性、可重复的 acquisition 选择物理位置；先在固定 K 下比较 dense、exact-uniform、random、actionness-only 与 actionness+boundary，再检验动态 K。

**条件接受**：连续 16-frame 物理块是修复 VideoMAE 内部伪连续时间的强候选，而非已经验证的唯一解。其 block 边界、cell 划分、前后端时间接口和“所有 block 都保留原生连续性”的实现必须先通过 regular-grid parity、timestamp metamorphic、pre-NMS coordinate trace 与实际 token/clip 计数；不能只由审查文字冻结。

**不接受为既定事实或参数**：`DUCA-SQB-Block-DK-v1` 名称、4 Query/2 层、16-frame block、边界半径 8、特定损失权重、1000/500 update 课程、K 集合、量化阈值以及 `+0.50 pp` 等门槛都是审查者的可检验提案。它们需要在训练侧数据、正式预算、方差和成本测量可用后预注册；不能以 Pro 文本代替科学裁决。

**明确纠正**：不应删除历史实现或把 64.352/65.696/65.385724 改写为无价值。前两者是不同 uniform/grid 参考，65.385724 是多因素、90-epoch 课程信号；均不能进入 matched 主表。UVT/Fovea 已得到真实负/开发性观察，但不是“未训练的 PRE_RUN 包”。

## 当前唯一下一科研动作

在新的干净实现周期中，先落地并审查同一 runtime 的固定-K 五臂控制：dense、exact-uniform、seeded random、actionness-only、actionness+boundary。所有 sparse 臂复用同一冻结语义 scout，使用同一 detector 初始化、更新数、NMS、评估器和终止 checkpoint 规则；只有 actionness+boundary 同时超过 uniform 与 random、且高 IoU/短动作不退化时，才开放动态 K。动态 K 随后必须与同均值 K、同 K 序列 uniform 和 K-shuffle 比较，并报告实际 `executed_k`、VideoMAE token/clip、padding、全流程成本和恢复闭环。

状态：`REVISE / designed_for_clean_reimplementation`。没有新的经验支持或论文就绪结论。
