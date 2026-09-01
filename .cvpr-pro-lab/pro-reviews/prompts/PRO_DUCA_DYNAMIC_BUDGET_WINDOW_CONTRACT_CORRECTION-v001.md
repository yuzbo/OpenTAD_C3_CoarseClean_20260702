# DUCA 动态预算训练窗口合同修订问询

Nonce: `DUCA-DYNAMIC-BUDGET-WINDOW-CONTRACT-CORRECTION-v001-20260828`

你是 DUCA 项目的科学负责人、整体研究流程维护者和实验设计者。Codex 只负责忠实执行你冻结的科学任务。请基于下列新发现，独立给出唯一 `CONTINUE / REVISE / PIVOT / STOP` 裁决；不要把路线选择交回人类或 Codex，也不要用新增流程文档代替一个可运行、可证伪的实验任务。

## 代码与证据身份

- GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 原冻结实现基座: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- 新实现分支已从该提交建立，但尚未写入模型改动: `codex/duca-semantic-budget-matched-20260828`
- 先前 Pro 终稿: conversation `6a9109a3-c5c8-83ea-af7e-9e996850187d`，决策 `REVISE`
- 先前终稿冻结的问题：同一离线视频内，以动作边界重要性和动作性不确定度对窗口排名，分配 `K=256/384/512`；每视频总预算严格等于 `384*n`；内容无关控制臂拥有相同 K 多重集，只置换窗口预算；两臂都必须真实减少/增加 VideoMAE 工作量，不能补齐到最大 K。

现有正式结果边界不变：共享 official dense AdaTAD Avg-mAP `68.73`；H65 30+60、seed 3407、terminal EMA Avg-mAP `65.1257`；H65 20+40 为 `62.4648`；PJST-D1 ON−OFF Avg-mAP 点差 `-0.472481` 但没有完成配对区间。当前没有动态预算 mAP 或成本证据。

## 实现前核验发现的真实冲突

### 1. H65 的训练输入不是固定滑动窗口集合

H65 Stage-2 继承：

- `configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py`
- `configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py`
- `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py`

实际训练 pipeline 在 `duca_transition_only_fixed384_official_adatad_backend_full_train.py:139-173` 和 `e2e_thumos_videomae_s_768x1_160_adapter.py:9-33` 使用 `LoadFrames(method="random_trunc", trunc_len=768)`。训练数据以视频为样本，每个视频在一个 epoch 只随机截取一个 768 帧窗口。

训练 dataloader 在 `e2e_thumos_videomae_s_768x1_160_adapter.py:113-122` 使用 batch size 2；`opentad/datasets/builder.py:26-59` 使用普通 `DistributedSampler` 并将样本打乱。一次 forward 不包含同一视频的全部窗口，也没有稳定的 `window_count`。

固定滑动窗口只存在于 validation/test：`e2e_thumos_videomae_s_768x1_160_adapter.py:35-61`。`ThumosSlidingDataset` 才提供 `video_name` 与 `window_start_frame`，见 `opentad/datasets/thumos.py:31-60`。

因此，若 Codex 直接把先前公式写入现有训练 forward，训练时每个视频的 `n` 只能视为 1，所有样本都会得到 `K=384`，动态预算实验退化为空操作。把训练改成固定滑动窗口、预先生成冻结侦察器的窗口预算清单、或引入按视频组织的 sampler/两遍数据流程，都会改变现有 H65 数据与训练合同，不能由实现者静默选择。

### 2. H65 侦察器的训练身份也需要明确

历史 H65 30+60 Stage-2 配置继续训练 ASFormer 侦察器及其 actionness/transition 分支；先前 Pro 终稿同时写了“冻结 H65 低成本侦察路径”和“不得改变 H65 侦察学习目标”。如果改用 Stage-1 epoch-29 的冻结侦察器生成全程预算，候选和控制臂内部仍可公平，但它们不再完全复现取得 `65.1257` 的 H65 Stage-2 训练身份。若侦察器继续更新，预先生成一次的预算清单又会随训练失效。

### 3. 真实 variable-K 是独立但可实现的工程问题

`opentad/models/duca/acquisition.py:2913-2924` 当前明确把 detector/backbone 输入补齐到最大预算，并记录 `dynamic_compute_realized=False`。按真实 `K=256/384/512` 分桶执行 VideoMAE、恢复样本顺序、按样本数加权损失并保持一次 optimizer update，在工程上可以实现；但只有先解决训练期“哪些窗口一起决定预算”的科学合同，才不会得到伪动态实现。

## 你必须独立冻结的内容

请给出一个唯一、完整、可直接交给 Builder 的修订任务，并明确：

1. **训练期的窗口总体是什么。** 定义每个视频的窗口集合、窗口起点、重叠、短视频和最后窗口处理；说明是否继续保留 H65 的 `random_trunc`，若不保留，为什么这仍是对 H65 的公平归因实验。
2. **侦察器在何时冻结。** 精确指定 checkpoint/state key；说明预算证据是否每个 epoch 重算、只预计算一次，或随训练在线变化；不能同时要求“固定预算清单”和“侦察器继续改变”。
3. **训练与验证的一致性。** 说明语义臂和内容无关控制臂在 training/validation 各自怎样获得相同的每视频 K 多重集和完全匹配的实际 VideoMAE 工作量。
4. **H65 参考身份。** 说明支持门中的“相对 H65 下降不超过 0.30 点”应绑定现有 `65.1257`，还是必须添加一个同训练数据合同的 H65 fixed-K companion；不得重复 official dense、旧 uniform 或 random-frame baseline。
5. **最小代码表面。** 列出允许修改的具体文件/符号。若必须修改 dataset、sampler、pipeline 或增加预算清单生成工具，请明确授权；若不允许，则给出不依赖这些表面的可运行机制。
6. **真实分桶合同。** 冻结 K 分桶、batch/order 恢复、梯度与损失权重、一次逻辑 optimizer update、physical-coordinate remap 和真实工作量记录的精确要求。
7. **最小可区分测试和 PRE_RUN。** 测试必须能杀死 `n=1` 退化、跨视频误排名、最大 K padding、内容泄漏、预算不匹配、训练/验证窗口身份漂移和侦察器状态漂移。
8. **唯一正式实验。** 给出两臂或必要的最小 companion、完整 THUMOS14、seed、60-epoch Stage-2/6000 updates、terminal EMA、每 5 epoch checkpoint、官方 evaluator、配对统计、成本指标、支持/反驳/协议无效门。
9. **连续执行要求。** 给出 Builder → 独立 Critic → Evaluator 的唯一当前任务、依赖、返回物和绝对截止时间。确定性的 claim-preserving 缺陷应在同一任务内最小修复并继续，不能再次把工程故障升级为路线中断。

## 禁止事项

- 不得假定尚不存在的动态预算效能或成本结果；
- 不得回到 PJST、UVT、Fovea、连续 cliplet、direct selector 或 fixed-K 论文主线；
- 不得通过修改 validation 结果、阈值、预算比例或挑选中间 checkpoint 调参；
- 不得让 Codex 或人类在多个未冻结方案之间自行选择；
- 不得以新增复杂合同、审计框架或工作流平台代替一个可运行科学任务。

请先给出唯一裁决和一句话修订后的科学问题，再给出冻结的数据/模型/实现/实验合同，最后给出唯一 `CURRENT_TASK_ORDER`。使用清楚的科研语言，不创造内部缩写。
