---
type: pro_scientific_decision
status: active
date: 2026-08-28
decision: REVISE
github_repo: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
code_inventory_revision: 5136011ed57df8a639427a633a488a592ba95924
implementation_base_revision: 04c35a3b76897e6c1569eeede41ed3aecaf7f854
latest_contract_date: 2026-08-29
latest_contract_nonce: DUCA-DYNAMIC-BUDGET-WINDOW-CONTRACT-CORRECTION-v005-20260828
---

# DUCA 语义动态预算匹配实验科学裁决

## 裁决来源

- ChatGPT Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- conversation：`6a9109a3-c5c8-83ea-af7e-9e996850187d`
- nonce：`DUCA-RESEARCH-CONTINUITY-NEXT-TASK-v001-20260828`
- Pro 路线：ChatGPT Pro，界面最高 5/5
- 完整可见终稿：`.cvpr-pro-lab/pro-reviews/runs/duca-research-continuity-next-task-v001/visible-report.md`

终稿的唯一科学裁决为 **REVISE**。DUCA 的长期问题继续成立，但当前主线停止继续修补 PJST-D1 物理时间表示，转而直接检验动态预算的核心因果问题。

> 本页第 23–100 行记录 2026-08-28 的初始合同。实现前核验发现训练窗口总体不成立后，Pro 已在同一科学问题上给出下文“修订后的权威实验合同”。两处冲突时以下文为准；初始合同保留用于说明修订原因，不能再作为实现规格。

## 当前论文问题

在每个离线视频的 VideoMAE 高分辨率总帧预算与固定 `K=384` 完全相同时，低成本动作性与边界语义能否把计算分配给更需要它的窗口，从而优于内容无关的预算置换，并保护高时间交并比下的边界定位？

关键预测是：即使部分窗口从 `K=384` 降到 `K=256`，只要同量预算转给语义上更困难的窗口，整体平均检测精度和高时间交并比检测精度仍应提高。最强反解释是收益仅来自存在 `K=512` 的窗口、总计算增加、最大 K padding、训练批组成或可变长度正则化，因此控制臂必须保持同视频完全相同的 K 多重集和总重型工作量。

## 冻结机制

对同一视频中的每个窗口计算：

`D_w = 0.5 * rank(mean(boundary_importance)) + 0.5 * rank(mean(4*p*(1-p)))`

其中 `p` 是既有二元动作性概率，排名只在同一视频的窗口之间进行，不使用训练、验证或测试 GT。若视频有 `n` 个窗口：

- `n=1` 时使用 `K=384`；
- `n>=2` 时令 `q=max(1,floor(n/4))`；
- 最高难度的 q 个窗口使用 `K=512`；
- 最低难度的 q 个窗口使用 `K=256`；
- 其余窗口使用 `K=384`；
- 同分按窗口原始起始时间排序。

因此每个视频总 K 严格等于 `384*n`。

内容无关控制臂复用同一视频的相同 K 多重集，但由 `seed=3407` 与视频 ID 决定一个不读取 RGB、语义分数、标签或预测的固定置换。两臂的实际帧仍由同一 H65 窗口内语义排序选取。

## 代码主线

- 唯一干净基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- 新分支：`codex/duca-semantic-budget-matched-20260828`
- 禁止从混合库存提交 `5136011...` 直接开实验。

允许修改：

- `opentad/models/duca/acquisition.py` 中动态预算、真实 K 分桶、顺序恢复、损失聚合与实际工作量记录；
- `opentad/models/duca/dynamic_budget.py` 中既有预算控制器和本轮无学习语义难度分配；
- 两个匹配配置；
- 一个聚焦测试文件；
- 一个最小 Slurm 启动器和结果分析入口。

禁止改变 H65 侦察学习目标、窗口内排序、detector/head/loss/NMS、数据划分、评估器、VideoMAE 预训练、Stage-1 epoch-29 检查点、Stage-2 更新和学习率日程、terminal EMA、物理时间解码、增强、seed、bootstrap 统计量或阈值。

## 唯一当前任务

实现真实的 `K=256/384/512` VideoMAE 分桶计算，不得补齐到 `K=512`；恢复原批次与视频顺序；在一次逻辑 batch 中按样本数加权聚合损失并只执行一次 optimizer update；记录实际输入帧数、tubelet 数和 VideoMAE 调用形状。

聚焦测试必须区分：

- 固定 K384 与 H65 原路径数值兼容；
- 三种 K 实际进入不同 VideoMAE 形状；
- 两臂每视频 K 多重集与总 K 完全一致；
- 内容无关置换不读取内容；
- 分桶重组不改变原始时间坐标；
- detector 梯度有效而冻结侦察模型不更新；
- 分桶损失聚合保持样本权重；
- validation/test GT 不进入预算选择。

## 正式实验

- 完整 THUMOS14 官方训练和 validation；
- seed `3407`；
- Stage-2 60 epoch、最多 6,000 次成功 detector update；
- 每 5 epoch 至少保存一次完整可恢复检查点；
- 固定使用 epoch-59 terminal `state_dict_ema`；
- 两臂各一张 GPU，可并行；
- 不重复 dense、uniform、random frame 或官方 AdaTAD 训练；
- 主要指标为 Avg-mAP、mAP@0.3–0.7 和 10,000 次整视频配对 bootstrap；
- 同时记录实际 VideoMAE 工作量、端到端推理时间、VideoMAE 时间和峰值显存。

第一种子只有同时满足下列条件才支持机制：语义臂相对置换 Avg-mAP 至少 `+0.50` 点，95% 配对区间下界大于 0，高 tIoU 平均差不为负，实际重型工作完全匹配，且语义臂相对 H65 `65.13` 下降不超过 `0.30` 点。它仍只属于开发筛选，不足以支持稳定性结论。

## 截止时间与证据边界

- Builder exact commit：2026-08-29 18:00 前；
- 独立 Critic 与必要的最小修复：2026-08-30 12:00 前；
- 两臂正式提交：2026-08-30 18:00 前；
- 完整结果、配对统计和成本：2026-09-04 23:59 前。

当前没有新的动态预算实现、PRE_RUN、训练、mAP、置信区间或成本结果。PJST-D1 的配对区间不再是当前任务；其平均点估计没有正向支持，但总体效应仍未形成配对统计结论。

## 实现前核验发现的合同冲突

对干净基座的实际数据链核验表明，当前 H65 训练不是把一个视频的全部固定窗口同时送入模型。训练集以视频为样本，每个训练轮次通过 `random_trunc` 只抽取一个 768 帧窗口；普通分布式采样器随后按样本打乱，批大小为 2。验证阶段才使用固定滑动窗口。因此，现有一次模型前向中没有同一视频的完整窗口集合，也没有可直接用于视频内排序的窗口总数。

这意味着直接在当前 `DucaAcquisitionAdapter` 中实现上述公式会令训练时的 `n` 退化为 1，所有窗口都得到 `K=384`，无法检验动态预算。若改为固定滑动窗口训练、预先生成冻结侦察器的逐窗口预算清单，或引入按视频组织的两遍数据流程，都会改变当前冻结的数据/训练合同，不能由执行者自行选择。另一个已确认事实是，现有动态预算路径仍补齐到最大 K，尚未减少 VideoMAE 的实际输入。

因此当前状态为：科学问题仍有效，但原冻结任务在进入代码实现前需要 Pro 明确修订训练期窗口定义、冻结侦察器的时点，以及如何在不破坏 H65 对照身份的前提下形成每视频 `384*n` 的预算。该问题属于实验设计与公平比较冲突，不是代码故障，也不是动态预算的负结果。

## 修订后的权威实验合同（2026-08-29）

### 裁决身份与唯一问题

- Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- conversation：`6a91a32f-ea68-83e9-965e-a58c3060c365`
- nonce：`DUCA-DYNAMIC-BUDGET-WINDOW-CONTRACT-CORRECTION-v005-20260828`
- 原始会话元数据：`.cvpr-pro-lab/oracle/duca-dynamic-budget-window-v005-transport2/sessions/duca-dynamic-window-correction-v2/meta.json`
- 用户提供的完整可见回复：`C:/Users/skywalker/.codex/attachments/313a3fc2-107c-4dcd-a2f6-ed0b2900e562/pasted-text.txt`

唯一裁决仍为 **REVISE**，但实现阻塞已经解除。修订后的科学问题是：在冻结的全视频滑动窗口总体、冻结的 H65 低成本语义侦察器和严格相同的每视频平均高分辨率预算下，把计算从低语义价值窗口转移到边界重要且动作性不确定的窗口，能否优于内容无关的匹配预算置换，同时不劣于同一新合同下的固定 `K=384`？

### 全视频窗口总体与训练单位

- THUMOS14 training 200 个视频、official validation 211 个视频；训练、验证和测试使用同一个确定性窗口生成器。
- 窗口长度 `L=768`，步长 `D=384`。短视频只产生一个补齐窗口；长视频使用升序固定起点，并补充唯一的右对齐末窗。训练不再使用 `random_trunc`，背景窗口也必须保留。
- 一个训练样本是一个视频；该视频的全部窗口在一个训练轮次中各出现一次。每个训练轮次按 seed `3407` 确定性打乱视频，逻辑批次为两个视频。
- 每个训练轮次恰有 100 次 optimizer update；60 个训练轮次恰有 6,000 次成功更新。窗口或 K 分桶不是独立 optimizer update。

### 冻结侦察器与逐窗口表

- 冻结 checkpoint：`/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823/gpu1_id0/checkpoint/epoch_29.pth`。
- 预期状态键为 `state_dict_ema`，预期 SHA256 为 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`。运行前必须对真实远端文件核验路径、状态键、训练轮次与摘要；相似 checkpoint 不得替代。
- 冻结低成本侦察器、ASFormer 时序编码、动作性输出、边界评分、窗口内 `global_structured_topk` 排序和所有计数器。侦察器始终处于评估模式，不进入 optimizer，不接收梯度。
- 使用确定性 resize/center crop，一次性生成 training 和 validation 的逐窗口表。键至少包含 split、video、window start/end、window index/count；值包含 `K=256/384/512` 的选帧位置，以及边界强度 `B`、动作性不确定度 `U` 和综合语义分数 `S`。正式训练不得重新计算或覆盖该表。

### 预算分配与三个正式实验臂

- `U` 为冻结二元动作性概率的平均归一化熵，`B` 为冻结边界 logit 经 sigmoid 后的窗口均值。`B` 和 `U` 只在同一视频的窗口间计算中位并列排名，`S=(R_B+R_U)/2`；同分依次按 `S/B/U/窗口起点` 决定。
- 对含 `n` 个窗口的视频，`q(n)=0`（`n=1`）、`q(n)=1`（`n=2`）、`q(n)=floor(n/3)`（`n>=3`）。语义臂将最高的 q 个窗口设为 `K=512`、最低的 q 个窗口设为 `K=256`，其余为 `K=384`。每视频 K 之和严格等于 `384n`。
- 内容无关控制臂用冻结 nonce、split、video 和窗口起点生成稳定伪随机顺序，把完全相同的 K 多重集分配给相同窗口总体；它不能读取 RGB、侦察语义、GT 或 detector 预测。
- 固定伴随臂在同一新滑窗合同下对所有窗口使用 `K=384`。历史 H65 `65.13` 只作为历史参考，不能代替这一伴随臂。

### 真实可变 K 与物理时间

- 将两个视频的全部窗口展平后，按 `K=256/384/512` 分桶；VideoMAE 必须分别接收确切长度，不得补齐到 `K=512`。对应的 16 帧 clip 数为 16、24、32。
- 各桶可以微批次运行，但同一逻辑批次只能执行一次 backward、optimizer update、学习率更新和指数移动平均更新。损失按窗口数加权，不能把视频或 K 桶重新等权。
- 分桶后必须恢复原视频和窗口顺序。选中帧的原始物理位置插值到 384 点 detector 网格；原始 proposal 在 NMS 前逆映射回物理时间并加回窗口偏移，官方检测头、损失、soft-NMS 和 evaluator 不变。
- 运行账本必须记录 requested/effective/unique/actual backbone K、tubelet 数、detector 长度、耗时、显存和真实执行标志。只有 VideoMAE 实际收到不同形状时才能写“真实可变计算”。

### 允许表面、运行前核验与正式判据

实现基座和分支保持 `04c35a3b76897e6c1569eeede41ed3aecaf7f854` / `codex/duca-semantic-budget-matched-20260828`。允许修改范围仅限数据窗口/组织、采集表、ActionFormer 真实分桶、必要的训练批逻辑、三个配置、一个聚焦测试文件和一个最小 N16R4 启动器；旧 `dynamic_budget.py` 控制器、PJST、UVT、Fovea、连续片段和其他历史路线都不是本轮依赖。

运行前核验必须先完成：200/211 视频窗口表；一个同时含三种 K 的真实 CUDA 逻辑更新；侦察器参数不变；每臂一个完整 validation 视频；proposal 物理范围；实际 VideoMAE 输入 hook 与成本账本；checkpoint 保存、恢复及下一更新计数。运行前核验不产生论文 mAP。

正式实验只运行 fixed384、semantic 和 content-independent 三臂，使用 THUMOS14、seed `3407`、60 个训练轮次、6,000 次成功更新、每 5 个训练轮次可恢复 checkpoint 和 epoch-59 `state_dict_ema`。主要比较为 semantic−control Avg-mAP；安全比较为 semantic−fixed 的 Avg-mAP 和 mAP@0.7。三项都使用 10,000 次整视频配对 bootstrap。

开发种子仅在协议有效、semantic−control 至少 `+0.30` 个百分点且区间下界大于 0、semantic−fixed 的 Avg-mAP 和 mAP@0.7 区间下界均不低于 `−0.30` 时支持机制；semantic−control 区间上界不大于 0，或任一安全比较区间上界低于 `−0.30` 时反驳当前机制。其他情况为统计未决或协议无效。无论结果如何，本轮不能声称降低每视频总帧预算或优于 dense；它只检验同平均预算下的跨窗口语义分配。
