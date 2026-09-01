# DUCA H65 90→60 轮压缩：终态结果裁决与训练动力学解释

Nonce: `DUCA-H65-60-COMPRESSION-TERMINAL-v001-20260824`

你是 DUCA 项目的独立 Scientific First-Author Agent、训练动力学专家和最严厉的审稿人。请直接阅读附加的历史比较收据、上一轮冻结裁决、两条真实配置与本次终态结果，给出唯一、可执行的中文终稿。不要把选择交还给人类，不要假定未提供的诊断数据存在，也不要为了“继续实验”而发明无界调参网格。

## 唯一任务

解释为什么 H65 从历史 `30+60` 压缩后显著下降，并用刚完成的两个 `30+30` LR 归因实验判断：

1. 性能下降的主因是否仍可归为学习率衰减过快，还是证据已经更支持 Stage-2 更新数、课程/反馈时钟、full-joint 暴露或其他训练动力学不足；
2. `LongCosine-H6000` 比 `AM-RPCH25` 多保留 LR 面积，却只获得有限 Avg-mAP 改善且高 IoU 不占优，这一模式能排除什么、不能排除什么；
3. 是否应严格执行上一轮预注册的 `STOP_60_EPOCH_COMPRESSION`，保留历史 `30+60` 作为 H65 训练参考；
4. 后续调整方法应是什么。这里必须区分：
   - 当前材料已经支持、无需再训练即可做的诊断；
   - 只有获得新证据后才值得开启的新实验；
   - 明确不应继续的 scheduler/LR 搜索。

## 不可改变的事实边界

- H65 的模型结构、语义间接非均匀逐帧选择、K=384、选中 RGB、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split 和官方 evaluator 在本次归因中不变。
- `20+40` 压缩同时改变了 Stage-1 成熟度、Stage-2 更新数、课程/反馈时钟、full-joint tail、cosine horizon 与 EMA 暴露，不能把其下降全部归因于一个因素。
- 两个新 arm 均复用成熟 30-epoch Stage-1 epoch-29 EMA，仅改变 Stage-2 LR 曲线；每个 arm 只有 3000 Stage-2 successful updates，semantic/policy transition 为 2000 steps，feedback 为 1000 warmup + 1000 transition，full-joint tail 约 1000 updates。
- 两个新 arm 的 terminal EMA 都明确失败，且 epoch 29 Avg-mAP 均比 epoch 24 略低；因此不满足上一轮唯一 `+1000` 延长所需的 rising-tail 条件。
- 不允许使用 intermediate checkpoint 挑最好，不允许把 terminal online 替代 EMA，不允许重新解释已冻结阈值。

## 你必须给出的终稿

首先给唯一裁决：`STOP_60_EPOCH_COMPRESSION / REVISE_WITH_ONE_NEW_MECHANISM_TEST / CONTINUE` 三选一。若不选择 STOP，必须明确指出哪条新证据足以推翻上一轮冻结的 flat/falling stop branch；仅凭“也许更多训练有用”不成立。

随后逐项给出：

1. 按证据强弱排序的因果解释，严禁把相互耦合因素机械分配为若干百分点；
2. 对两个 arm 的 terminal 与 epoch19/24/29 曲线作联合解释，特别分析 Avg-mAP 与 mAP@0.7 的不同走势；
3. 说明为何只调整 LR 曲线未恢复 H65，以及这是否意味着“60 轮原则上不可能”等价 90 轮；
4. 给出最小后续动作：优先列出无需训练的日志/梯度/selector/online-vs-EMA/更新轨迹诊断。若仍建议一个新训练，必须是单一、H65-compatible、能区分机制且不构成第三个 scheduler 搜索的实验，并给出严格触发条件、唯一终点和停止规则；
5. 明确禁止项：峰值 LR 全局抬高、参数组无依据微调、第三个 30+30 scheduler、20+40 重跑、intermediate-best 挽救、模型/selector/dynamic-K/TrueTime 混入本归因问题；
6. 输出 `next_owner / next_action / dependency / expected_return_at`。

请用论文级、严肃但易懂的中文写作。区分“观察事实”“合理推断”“尚未测量”。不要声称单 seed 结果具有统计稳定性或论文级训练效率结论。
