# DUCA H65 First-Mixing SingleClock：Unit-1 终态门操作化裁决

Nonce: `DUCA-H65-SINGLECLOCK-UNIT1-GATE-OP-v001-20260824`

你是 DUCA 项目的独立 Scientific First-Author Agent 与统计审稿人。本轮只解决已经接受的 `DUCA-H65C-SINGLECLOCK-DYNAMIC-v002` Unit-1 终态门如何被代码无歧义执行；不得重开路线、提高门槛、要求新训练、引入 Query/dynamic-K，或把成本变成未写入原合同的前置 kill gate。

## 已接受、不得修改的 Unit-1 合同

- H65 的固定 K384、语义间接非均匀逐帧选择、选中 RGB、VideoMAE-S/Adapter/ActionFormer、loss、NMS、split 和官方 evaluator 保持不变。
- 唯一表示变化是在第一个 temporal attention mixing 中加入零初始化 SingleClock relative physical-time residual。
- H65 replay 身份先验证 selected indices、gathered RGB、VideoMAE input tensor、raw proposals/scores 和 official evaluator JSON；uniform positions 必须 bit-identical。
- 相对 H65 replay 的主门原文为：`ΔAvg >= -0.20 pp`、`Δ@0.6 >= -0.20 pp`、`Δ@0.7 >= -0.20 pp`；高 gap-CV / 高 boundary-density 窗口中 paired boundary error 不恶化。任一主指标低于 margin 或 uniform identity 失败即 `KILL_SINGLECLOCK_REPRESENTATION`。
- 当前训练和推理已完成/正在封存；不允许为补诊断重训。

## 当前实现问题

现有 `finalize_duca_h65_singleclock_terminal.py` 使用了旧的正增益门：ON-vs-gate-zero Avg-mAP 至少 `+0.50 pp`，并把 ON-vs-H65 OFF `+0.50 pp`、coadaptation CI 和 cost 硬门混入 `main_pass`。这与已接受 Unit-1 的 H65 replay `-0.20 pp` 非劣门不一致。

现有 `analyze_duca_h65_singleclock_strata.py` 已能：

- 只用 training population 冻结 per-video distortion q25/q50/q75 与 short-action duration q25；
- 在 validation 上对 low/high distortion 和 short actions 做 10,000 次 whole-video cluster paired official-mAP bootstrap；
- 计算 high-low distortion interaction。

但它没有 boundary-density 定义，也没有 proposal-vs-GT boundary error。仓库中存在两个候选 boundary estimator：

1. 每个 GT 取同类最高 IoU proposal，报告 start/end MAE seconds 与归一化误差；
2. 按 score 排序做同类 IoU>=0.5 的一对一匹配，报告 matched recall 和 start/end MAE seconds。

二者均不是官方 THUMOS evaluator 的原生指标，也没有既有 Unit-1 freeze。已有 10,000 次 PCG64 whole-video cluster paired bootstrap 基础设施可复用。

## 你必须冻结的唯一可执行规格

请给出唯一 `REVISE_GATE_IMPLEMENTATION / STOP_UNDERDEFINED_GATE`，并明确：

1. Unit-1 主对照究竟是 SingleClock ON EMA vs H65 OFF/replay EMA，还是 ON vs same-checkpoint gate-zero；final 与 gate-zero 应各自承担 identity、机制诊断还是主门角色。
2. 三个 `-0.20 pp` 主指标使用 point estimate 还是 paired-bootstrap CI；若用 CI，请给 exact bound。不得事后提高原合同。
3. `boundary error` 的精确定义：proposal-GT matching、类别约束、IoU cutoff、score/top-k/NMS 状态、seconds 还是 duration-normalized、unmatched GT 如何计入、每视频如何聚合。
4. `gap-CV` 与 `boundary-density` 的精确定义；threshold 必须来自 training population。请给 exact q25/q75 或其他 cutpoint、window/video 粒度、短视频/padding处理。
5. “不恶化”的精确定义：point delta `<=0`、绝对容忍 margin、还是10k paired video-cluster bootstrap CI；给 exact sign、bound 与 resampling unit。说明高 gap-CV 和高 boundary-density 是 AND、OR 还是分别过门。
6. 现有训练已经结束，若当前 prediction/identity/GT annotations 足以离线计算，请要求只做离线统计；若无法合法计算，请说明 Unit-1 是否应只按主指标+identity裁决，而把未操作化条款降为诊断，禁止要求重训。
7. cost 在 Unit-1 中是报告项、后置准入项还是硬 kill gate。原文没有 Unit-1 cost margin，不得凭空发明。
8. `paper_claim_admissible` 在单 seed Unit-1 是否必须保持 false；Unit-2 Query residual 何时才允许进入 Builder。
9. 给出终结器应输出的最小字段、唯一 PASS/KILL token、以及可由 focused unit tests 覆盖的例子。
10. 输出 `next_owner / next_action / dependency / expected_return_at`。

要求：保留原 Unit-1 科学意图，选择最小、可复现、不会利用 validation 调阈值的定义。不要把本轮变成新方法或新实验矩阵。
