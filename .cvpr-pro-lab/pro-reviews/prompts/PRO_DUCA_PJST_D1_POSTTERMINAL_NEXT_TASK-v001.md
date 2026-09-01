# DUCA PJST-D1 终态后的下一项科研任务下达

Nonce: `DUCA-PJST-D1-POSTTERMINAL-NEXT-TASK-v001-20260827T191100+0800`

Exact ChatGPT Project: `g-p-6a796fef9a00819194024cf1de3bd697`（DUCA）

GitHub code truth:

- Repository: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- Public branch: `codex/duca-pjst-cycle4-builder-20260826`
- Public scientific/training revision: `c73e8418de31cdcb2a445ff58a1e33ab9ab6a508`
- Local evaluation/finalizer revision: `7bd120f0d342bf175c97c365fba7cbd359df055e`，它以 `c73e8418...` 为祖先并领先 3 个仅与终态评估/统计打包有关的提交；该 SHA 尚未在公开远端分支出现，因此不得把它写成可由网页直接核验的 GitHub revision。
- 协调根 `a6bdc084...` 属于分叉且 dirty 的 SparseHead 路线，不是 DUCA/PJST 代码或实验身份。

请作为本项目的 **Scientific First-Author Agent and Primary Research Owner** 工作。你负责论文级科学问题、机制、下一实验、证据解释与发表策略；Codex 只作为有界执行者：Builder 实现，Critic 独立攻击，Evaluator 执行和测量，Coordinator 传递原始证据。不要把路线选择退回给 Codex 或人类。本会话必须独立重建项目，没有任何未说明的旧聊天依赖。

## 1. 论文问题与冻结主线

DUCA 研究离线时序动作检测中的任务感知稀疏视频计算：低成本语义 scout 预测动作性和边界重要性，确定性间接采样选择原始物理时间位置，随后只让被选高分辨率帧进入重型 VideoMAE/AdaTAD 路径。动态预算是最终论文目标；固定 `K=384` 只用于当前表示归因与回退，不是最终主张。

当前 PJST-D1 只回答一个窄问题：在冻结并重放同一 H65 语义间接选择、相同 K384 位置、相同 RGB、相同数据顺序和相同 60 epoch/6000 成功更新协议时，是否应在 VideoMAE 第一次二帧 tubelet 混合前按真实物理间隔缩放导数分量。零阶外观均值、selector、detector、loss、NMS、split 和 evaluator 均不变。它不是 dynamic-K、Query-Bridge 或端到端 selector 总效应实验。

相关代码路径：

- `opentad/models/backbones/backbone_wrapper.py`
- `opentad/models/backbones/vit_adapter.py`
- `opentad/models/utils/temporal_grid.py`
- `opentad/models/detectors/single_stage.py`
- `configs/adatad/thumos/duca_pjst_d1_stage2_off.py`
- `configs/adatad/thumos/duca_pjst_d1_stage2_on.py`
- `tests/test_duca_pjst_d1_derivative_only.py`
- `tools/bata/validate_duca_pjst_d1_derivative_only.py`

## 2. 已完成的真实实现与实验

- 训练 revision：`c73e8418...`；clean cycle-4 实现只关闭测试替身、不变性、canonical THUMOS14 路径与 Stage-1 零标量身份缺陷，没有改变 PJST 公式、选择器、损失、优化器、训练日程或 evaluator。
- OFF/ON 均从相同 H65 Stage-1 epoch-29 EMA 开始，seed `3407`，固定 `K=384`，完成 60 epoch / 6000 成功更新。
- 数据与评估：完整 THUMOS14 official validation 211 videos；官方 OpenTAD evaluator，tIoU `0.3/0.4/0.5/0.6/0.7`；相同 soft-NMS；terminal checkpoint 均为 epoch-59 `state_dict_ema`。
- 评估 revision `7bd120f0...` 的只读重推理成功：OFF Job `1257897`、ON Job `1257898` 均 `COMPLETED 0:0`；每臂 `211/211` videos、`422,000` predictions，视频 ID 集合完全一致；全部 mAP 与原终态记录逐位复现，误差为 `0 pp`。

点估计（百分数）：

| metric | OFF | ON | ON−OFF (pp) |
|---|---:|---:|---:|
| mAP@0.3 | 80.046988 | 79.251767 | -0.795221 |
| mAP@0.4 | 75.568715 | 74.316270 | -1.252444 |
| mAP@0.5 | 68.021751 | 67.874767 | -0.146984 |
| mAP@0.6 | 58.032935 | 57.742440 | -0.290495 |
| mAP@0.7 | 43.646027 | 43.768766 | +0.122739 |
| Avg-mAP | 65.063283 | 64.590802 | -0.472481 |

## 3. 当前不可准入的证据与客观故障

统计终结器 Job `1257899` 在任何 bootstrap 抽样前失败。冻结 argv 把 OFF/ON prediction 指向 `.../work/result_detection.json`，而 `tools/test.py` 的真实 DDP 输出位于 `.../work/gpu1_id0/result_detection.json`。因此：

- `0/16` shards；
- `0/10,000` paired whole-video bootstrap replicates；
- 无 95% paired interval；
- 无 PASS/KILL；
- 不能把 `-0.472481 pp` 写成总体显著负向，也不能把单独的 `mAP@0.7 +0.122739 pp` 写成收益。

该故障是终结器路径绑定错误，不是训练失败，也不是 PJST 科学结果。此前 DAG 的 `single_recovery` 已关闭；任何新统计执行、实现或路线动作必须是你此轮明确指定的新任务，不能伪装成旧作业自动重试。

## 4. 必须保留的历史边界

- 官方共享 dense AdaTAD 完整验证约 `68.73`；DUCA 不重复官方基线训练。
- 30+60 H65 参考约 `65.13`；20+40 压缩为 `62.46`，两条 30+30 学习率归因只到 `63.22/63.56`，所以 60-epoch compression/LR sweep 已停止；这不否定 H65 间接非均匀选帧。
- RankPack/TrueTime 单 seed 配对为 `61.57/62.19`，只提供物理时间解释的部分机制支持。
- 连续 cliplet FZ/JT 完整训练为 `49.89/47.24`，该实现路线已否定，不得复活。
- 不重复 dense、uniform、random、H65 compression、SingleClock、Query-Bridge、dynamic-K 或 continuous-cliplet 矩阵来回答 PJST-D1 首次混合问题。
- 不增加通用哈希、证明系统、复杂合同框架、迁移层或重复审计。以最短的论文证据闭环为先。

## 5. 你必须下达的唯一任务

请基于上述完整状态返回且只返回一个科学裁决：`CONTINUE / REVISE / PIVOT / STOP`（可加简短后缀，但不能同时给多个候选让 Codex选择）。

裁决必须完成以下内容：

1. 明确当前 active mechanism、claim、anti-claim 和最便宜 falsifier；说明点估计已经回答了什么、尚未回答什么。
2. 判断下一项工作究竟应是：
   - 新的、统计意义独立的既有预测只读配对区间任务；
   - 一个最小且 claim-preserving 的模型/表示修订；
   - 结束 PJST-D1 并进入另一条已被证据支持的 DUCA 路线；
   - 或停止当前论文主张。
   你必须自行选择唯一一项，不得把选择交回 Coordinator。
3. 下达一个可以立即执行的、最小、论文导向的任务，写明：任务名称、Builder/Critic/Evaluator 谁 active/idle、允许改动的模型/配置/统计表面、明确禁止项、所需数据/split/checkpoint/evaluator、验收标准、成功/失败停止规则和必须返回的原始证据。
4. 如果授权利用现有 OFF/ON predictions 完成配对统计，明确统计单位、重采样方式、区间定义、门槛与结果到 claim 的边界；不得事后修改冻结统计来挽救结论。
5. 如果授权模型修订，必须说明它为何不是继续堆叠时间编码，并给出能区分至少两个竞争根因的单个决定性实验；禁止重新训练大矩阵。
6. 给出 `next_owner / next_action / dependency / expected_return_at / single_recovery`。
7. 指定一个绝对的北京时间完成时限，不得写“尽快”。优先选择最早能得到决定性证据的时间；如晚于 `2026-08-28T12:00:00+08:00`，必须说明客观资源依赖并给出最迟不晚于 `2026-08-29T12:00:00+08:00` 的首个材料或实验里程碑。

Codex 在收到并核验你的任务与截止时间前不会开始新的实现、统计或实验。
