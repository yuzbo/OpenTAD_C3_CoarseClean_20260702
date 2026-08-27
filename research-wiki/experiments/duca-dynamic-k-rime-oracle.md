---
type: experiment_plan
node_id: exp:duca-dynamic-k-rime-oracle
status: discussed_proposal
updated: 2026-07-27
idea: idea:duca-rime
implementation: not_started
training_authorized: false
paper_claim_allowed: false
---

# DUCA dynamic-K / RIME Oracle 与因果门

## Question

在严格相同的 realized mean cost、物理时间坐标和后续 detector 下：

1. 视频间是否存在足够的预算异质性，使 per-video dynamic K 的 Oracle 超过 best fixed K；
2. strict nested physical-frame ladder 是否接近 independent-per-K optimum；
3. 低成本证据能否预测 hard budget-policy/group-add utility；
4. paired-boundary risk 是否独立保护 mAP@0.7、短动作和双端覆盖；
5. 动态增益是否独立于 mixed-K exposure、K 直方图和 learned positions。

## Evidence boundary

本节点仅登记实验设计。当前没有：

- clean native K=384/K=192 uniform；
- dynamic Oracle；
- nested regret；
- RIME implementation；
- fair total-60 development seed；
- multi-seed、second-detector 或 cost result。

旧 K=384 `65.385724%` 和 K=192 `57.967272%` 是 90 轮超预算诊断，不进入本实验
主比较。

## Phase 0: split and measurement freeze

在运行任何 Oracle 前创建完整视频 ID 级 manifest：

- detector/selector train；
- hard-label generation；
- utility/risk fitting；
- dual/risk calibration；
- certification/development；
- official final evaluation。

同一视频的窗口不得跨 split。validation/test 配置如果复用 THUMOS validation videos，
不能视为独立总体。所有 seed、6,000-update、EMA/checkpoint、评估次数、成本口径和
统计方案在主结果前冻结。

## Phase 1: clean controls

1. released dense re-evaluation；
2. local dense；
3. clean native uniform K=384/K=192；
4. dynamic panel 每个候选 K 的 uniform，或与 full 完全相同逐视频 K 序列的
   uniform-position control；
5. wrapper tensor/raw-proposal/coordinate/mAP parity；
6. raw `q -> t -> official NMS` contract；
7. exact-K/identity/round-trip/zero-violation property tests。

任何 parity 失败都阻塞后续 learned/dynamic claim。

## Phase 2: Oracle and hard evidence

### O1: Dynamic budget headroom

在同一 frozen detector、同一 inner family 和相同 realized mean cost 下比较：

- each fixed K；
- best global fixed K；
- per-video independent budget-policy Oracle；
- histogram-shuffled K assignment；
- simple complexity/motion budget baseline。

报告 Avg-mAP、mAP@0.7、short-action、pair support、K distribution、realized cost 和
video-cluster paired confidence intervals。

### O2: Decoder-family regret

逐 K 比较：

- independent exact-K Oracle；
- strict nested insertion Oracle；
- one predeclared weak-overlap Oracle。

报告每视频和总体 regret、预算间 overlap、endpoint error、pair support、max-gap 与
high-IoU。只在 train-only certification split 上冻结最终唯一 decoder family。

### O3: `G_rank`

hard perturbation families：

- 1 frame；
- 2/4/8/16 frames；
- 1%/5%/10% dispersed swaps；
- contiguous blocks；
- start-only/end-only/start+end pair；
- full hard re-decode after density steps。

targets：

- cls/reg detector loss；
- proposal recall；
- high-IoU success；
- start/end/pair miss。

controls：

- matched random；
- score shuffle；
- score reverse；
- actionness；
- transition；
- motion proxy。

统计单位是完整视频；使用 cross-fit 和 cluster bootstrap。阈值由预先功效分析冻结。

### O4: Pair-risk calibration

检验 risk prediction 的 Brier/ECE/coverage/sharpness，以及低风险集合的实际
start/end pair failure。`risk_infeasible -> Kmax` 必须进入 realized-budget ledger。

## Phase 3: one development seed

所有 trainable arms：

- exactly 6,000 successful detector updates；
- same initialization, effective batch, mixed-K exposure, warmup/ramp；
- same cumulative heavy frames during training；
- same EMA/checkpoint and evaluation count；
- development seed excluded from final statistics；
- utility-label generation cost reported separately。

| Arm | Policy |
|---|---|
| U-fixed | clean uniform K=384 |
| U-same-K | full model 的逐视频 K 序列，但位置 uniform |
| F-bound | fixed-K bounded exact-K + admitted hard utility/risk |
| D-shuffle | full K histogram，随机打乱 K-video pairing |
| D-no-risk | dynamic policy without paired/high-IoU risk |
| AdapTok-TAD | total-loss curve + test-batch ILP direct-transfer baseline |
| RIME-full | frozen decoder family + hard utility/risk + frozen per-video dual |

MGSampler-like/AdaFocus-like 作为强外部机制对照，可复用同一训练/评估表面；不允许
通过改变 detector schedule 或平均 K 获得优势。

## Phase 4: formal evidence

仅在 Phase 3 明确正向后：

1. 冻结结构、K 集合、thresholds、dual 和 statistics；
2. 三枚未参与开发的新种子；
3. 第二 detector；
4. mean-K≈192 或第二预算面板；
5. batch=1 latency、bucket throughput、end-to-end throughput、energy、memory；
6. strict frozen-detector/train-free mode 单列。

## Stop rules

- O1 无可利用 headroom：不实现 dynamic scorer/solver；
- O2 strict nested regret 过大：删除 strict nested，不能降低门槛；
- O3 `G_rank` 失败：删除 utility head 和对应长训练；
- O4 pair risk 不优于简单信号：删除 pair-risk contribution；
- U-same-K 与 full 无差：learned positions 无效；
- D-shuffle 与 full 无差：content-conditioned budget assignment 无效；
- development full 不超过 best fixed at matched cost，或 high-IoU/short-action 退化：
  不补多种子、第二 detector；
- actual execution pads to Kmax 或完整成本无净省：删除 efficiency claim。

## Current status

`discussed_proposal / no_training_authorized / no_result`。

## Dual-response resolution

两份接管回复没有冻结同一个实验合同：

- 回复 A 使用每 update 一个主 K forward，并以约 25% cadence 增加 alternative
  hard forward；
- 回复 B 使用半批视频双分支，以保持每 update 的累计 heavy frames；
- 两者的 K grid、阶段边界、loss 拆分和 `G_rank/G_direct` 数值门槛不同。

因此本节点只吸收共同的因果问题，不吸收任一现成日程。Phase 0 新增
video-cluster variance/ICC/MDE 与功效分析；所有正式阈值随后一次性预注册。
Phase 2 先决定 decoder、utility 和 risk 是否存在，Phase 3 运行前再冻结唯一
single-branch 或 matched paired-branch 训练合同。若选择 paired branch，所有
trainable controls 必须共享相同 unique-video exposure、effective batch 和 cumulative
heavy frames；hard-label/refresh GPU-hours 始终单列。

两份原文与项目裁决：
`docs/methods/2026-07-27-duca-dynamic-k-adaptok-dual-response-comparison.md`。
