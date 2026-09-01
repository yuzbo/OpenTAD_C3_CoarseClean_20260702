# DUCA H65：60 轮压缩性能下降与学习率日程复核

Nonce: `DUCA-H65-60-LR-SCHEDULE-REASSESS-v002-20260824`

你是 DUCA 项目的独立 Scientific First-Author Agent、训练动力学专家和最严厉的实验审稿人。请在同一个回答中完成因果诊断、现有实验设计复核和终态结果后的决策树。不得把路线选择交回人类或 Codex；必须给出唯一的 `CONTINUE`、`REVISE`、`PIVOT` 或 `STOP`。

## 1. 项目与代码身份

- Exact ChatGPT Project: `DUCA`, `g-p-6a796fef9a00819194024cf1de3bd697`
- GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
- 历史 H65 evidence commit: `42dba3f90b37243e7965d18b6707e88e81bf7109`
- 课程压缩代码 revision: `04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- 当前 schedule-only attribution clean revision: `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`
- 当前实验只讨论 H65 训练日程；First-Mixing SingleClock 是独立的时间表示实验，禁止混为因果解释。

## 2. 冻结 H65 模型与公平性合同

H65 使用低成本 ASFormer 语义侦察器学习动作性、状态转移和边界证据，经确定性 sampling-rate/density transport 形成非均匀逐帧位置；不是小模型直接预测 frame index。固定 `K=384`，选中的 RGB 帧按原时间排序，送入相同 VideoMAE-S、Adapter、ActionFormer、loss、NMS、THUMOS14 training/validation split 与 evaluator。

当前 schedule-only attribution 中，下列内容全部冻结不变：模型结构、K384 非均匀逐帧选择、selected RGB、数据、seed `3407`、损失终值、检测器、评估器、AdamW、weight decay、backbone freeze、参数组成员和各组基础 LR。任何新建议不得借机改变 selector、Query、Bridge、dynamic K、连续 cliplet、loss 权重终值或基础 LR。

## 3. 真实终态证据

1. 历史 `30+60`：Stage-1 exact-uniform K384 训练 30 epoch/3000 successful updates；Stage-2 learned H65 训练 60 epoch/6000 successful updates。终态 epoch-59 EMA：Avg-mAP `65.1257`，mAP@0.7 `43.3137`。
2. 失败的压缩版 `20+40`：Stage-1 20 epoch/2000 updates；Stage-2 40 epoch/4000 updates。终态 epoch-39 EMA：Avg-mAP `62.4648`，mAP@0.7 `39.9434`。
3. 差值：Avg-mAP `-2.6609 pp`，mAP@0.7 `-3.3703 pp`。
4. Stage-1 终点也明显不同：30 epoch `59.4231`，20 epoch `49.5389`。因此失败的 `20+40` 不是“只少训 30 epoch”：它从不成熟的 epoch-19 EMA handoff，且同时压缩 LR、semantic/policy transition、feedback clock 与 full-joint tail。
5. 历史 `65.696` 是改造 physical-grid、协议未完全匹配的探索锚点，不得替代上述 matched H65 终态。
6. 所有数字均为单 seed，且历史 checkpoint 缺完整 RNG/DataLoader 恢复状态；它们可用于日程归因，但不是论文级多 seed 主张。

## 4. 已接受的上一轮 Pro 方案与当前正在运行的严格归因

上一轮独立 Pro 已给出 `REVISE`，要求保留成熟的 30-epoch Stage-1 epoch-29 EMA，不提高任何基础 LR，只运行 30-epoch/3000-update Stage-2 的两个 schedule-only 臂：

### A. `AM-RPCH25`

- successful update 1–500：历史绝对 warmup 到各组 base LR；
- 501–1500：保持 `1.0× base LR`；
- 1501–2500：相对 cosine 从 `1.0×` 降到 `0.25×`；
- 2501–3000：保持 `0.25× base LR`；
- 参数组 LR 比例始终不变；禁止统一绝对 `eta_min`。

### B. `LongCosine-H6000`

- 保持历史 6000-update cosine horizon；
- 只运行前 3000 successful updates，因此终点保留较高非零 LR；
- 其目的只是区分“相对 plateau/cosine/hold”与“历史长 horizon 截断”的 LR 形状，不是新模型。

两臂共同保留成熟 Stage-1 epoch-29 EMA、2000-step semantic/policy transition、1000-step detector-feedback warmup + 1000-step transition、1000-step full-joint tail；每 5 epoch 保存可恢复 checkpoint；固定 epoch-29 final 与 final-EMA，不按中间 validation 选择。

远端 N16R4 Jobs `1252979`、`1252980` 正在完整训练，PRE_RUN、恢复检查和路径 smoke 已通过，但尚无终态 validation、mAP 或成本结论。禁止读取中途 loss 后猜测优胜者，也禁止为了本次讨论追加第三个训练臂。

## 5. 需要你严厉回答的问题

1. 重新核验性能下降的证据排序：Stage-1 handoff 不成熟、Stage-2 更新量减半、LR 提前衰减、semantic supervision 退火、feedback timing、joint tail、EMA lag。哪些是确定事实，哪些只能由终态/已有曲线区分？不得编造数值归因。
2. 判断当前 A/B 两臂是否真正做到“模型不变、基础 LR 不变、只做 Stage-2 schedule attribution”；指出任何隐藏混杂项或无效比较。
3. 判断 30+30（总 60 epoch）是否存在理论或经验上的无损恢复保证。明确 LR 形状能修复什么，缺失 3000 次 Stage-2 minibatch/optimizer/EMA exposure 又不能修复什么。
4. 给出终态结果后的唯一决策树：
   - 若 A 或 B 恢复到历史锚点的可接受邻域，下一步是否只补 seeds，还是还需检查 schedule 稳定性？
   - 若两者都明显低于历史锚点，但梯度稳定、终点曲线仍上升，应优先增加 Stage-2 exposure、单独微调某参数组 LR，还是承认 60-epoch 预算不可无损压缩？
   - 若只有高 IoU 恶化、Avg-mAP 接近，应检查哪些边界/时间/selector 梯度现象？
5. 冻结可执行成功/失败阈值。请优先给出基于 matched historical anchor 的 Avg-mAP、mAP@0.7、训练稳定性与终端斜率判据；不得用中途最好 checkpoint。
6. 指定最少的只读诊断：历史 Stage-2 checkpoint 轨迹、各参数组 LR 曲线、未加权/加权语义 loss、梯度范数、selector 位移/熵、online-EMA gap。每项必须说明观察到什么会改变下一步。
7. 判断在 A/B 终态出来前是否应继续调参。若不应，明确 `HOLD_NEW_TUNING_UNTIL_TERMINAL`。
8. 训练日程恢复不是科学创新。请明确其结果到论文 claim 的边界。

## 6. 必须返回的终稿

- 唯一科学裁决及一句话原因；
- 性能下降的证据分层诊断；
- 对 A/B 两臂的严格正确性/混杂审查；
- 终态后的 if/then 决策树与停止规则；
- 不超过一个后续动作；若当前证据不足，应明确等待哪些终态，而不是设计新实验；
- Critic 与 Evaluator 必须验证的事实；
- `next_owner / next_action / dependency / expected_return_at`。

请用严肃、直接、外部评审可理解的中文作答。不要引入新的模型路线，不要重复 dense/uniform/random 对照，不要把正在运行的中间日志当作结果。
