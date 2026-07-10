# ChronoTransport / DCRT 完整讨论、实现与裁决档案

更新时间：2026-07-11

## 0. 重要状态纠正

ChronoTransport 不仅是一个被讨论后否定的 idea。它已经在独立本地 worktree/branch 中完成相当深的工程实现：

- Worktree：`E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702`
- Branch：`codex/c3-coarse-clean-20260702`
- 最新本地方法 commit：`92029ea feat: run formal ChronoTransport Stage B loop`
- 对应远端 branch 仍停在 `3554b6f`
- 本地 branch 比远端 ahead 15 commits（其中 `026f127` 是该分支自己的 Wiki 归档）
- ChronoTransport 源码不在当前 `codex/phystime-adatad-1` 工作树中

因此准确状态是：**工程实现与正式单种子 Stage-B 闭环均存在，但预注册 P3 科学 gate 已失败；对应提交未推送到远端，P5 未执行，并且该路线已被用户否决为当前论文主线。**

## 1. 研究动机

DUCA 通过删除/不处理部分输入来节省 heavy backbone，但会破坏 dense temporal geometry，并把计算转移到 scout、decode 或 selector。ChronoTransport 改问：

> 在保持完整 dense physical-time detector grid 的条件下，哪些 VideoMAE 重型中间计算可以被重算、传输或复用，同时把高-IoU 定位风险控制在校准上界内？

这把决策从“选哪一帧”改为：

```text
time block x layer group x {RECOMPUTE, TRANSPORT, HOLD}
```

## 2. 它是不是插件

它不是 detector 前的通用黑盒插件，也不是新 detector head。

更准确地说，它是 AdaTAD/VideoMAE backbone 内部的 conditional-compute runtime：

- 修改 heavy attention/MLP 的执行；
- 维护窗口内 cache；
- 保留 patch embedding、AdaTAD temporal adapters、detector grid、head、NMS；
- 关闭时或 forced-dense 时回到原 block loop。

因此它比 DUCA 更深地侵入 backbone，也更接近 feature reuse、Mixture-of-Depths 或 conditional execution。

## 3. 离线而非在线

ChronoTransport v1 是离线、全窗口 AdaTAD 条件计算：

- scheduler 可观察完整测试窗口中的 deploy-visible input；
- cache 只在当前窗口内使用；
- 每个 sliding window 重置，不跨窗口复用；
- 不声称 streaming、causal 或 prefix-invariant。

允许当前 patch/group input、cache age、proxy drift 和 chunk/layer identity；禁止 test GT、teacher、dense heavy current feature、raw detector prediction cache 或 counterfactual ledger。

## 4. 决策粒度：不是逐帧，也不是简单 16 帧固定单位

实际 AdaTAD/VideoMAE-S 几何：

```text
B x 768 input frames
  -> 48 clips x 16 frames
  -> tubelet_size=2
  -> each clip produces 8 temporal tubelets
  -> 48 x 8 = 384 internal temporal points
  -> existing postprocess interpolates to 768 detector positions
```

ChronoTransport v1 的 schedule 是 `[B, 48, G]`，默认 `G=3`：

- group 0：layers 0..3
- group 1：layers 4..7
- group 2：layers 8..11

因此 v1 的动作单位是**16-frame clip x 4-layer group**，不是逐 raw frame，也不是逐 tubelet。选择这一粒度是为了可实现和降低 scheduler 开销，不是理论上的唯一合理粒度。更细 per-tubelet/per-block routing 被留作未来消融。

这也正是用户质疑“层级优化过于死板”的来源：粒度是工程折中，可能掩盖短动作和局部边界需要。

## 5. 三种动作与 cache

### RECOMPUTE

- 当前 clip 的真实 group input 执行该组 heavy attention/MLP；
- 每个 block 后仍执行 dense AdaTAD temporal adapter；
- `anchor=latest=current`，`age=0`。

### TRANSPORT

- 跳过 heavy attention/MLP；
- 从 `latest` cache 与当前 deploy-visible input 生成 low-rank correction；
- 更新 `latest`，保留 `anchor`，`age+=1`。

### HOLD

- 输出逐位等于 `latest`；
- 不改变 anchor/latest；
- `age+=1`。

`anchor` 是最近真实 RECOMPUTE，`latest` 是最近 RECOMPUTE 或 TRANSPORT。连续 TRANSPORT 必须从 latest 链式递推，不能每次从旧 anchor 生成。

## 6. 哪些计算仍然 dense

v1 始终执行：

- decode；
- preprocess/H2D；
- patch embedding；
- deploy-visible innovation signal；
- 每个 block 后的 AdaTAD temporal adapter；
- neck/head/postprocess。

动态执行：

- VideoMAE heavy attention/MLP；
- transport correction；
- scheduler/cache movement。

因此它不节省 decode，也不跳过 AdaTAD dense temporal convolution。科学主张必须基于全栈实测，不是“跳过层数”或理论 FLOPs。

## 7. 风险、成本与 fail-closed

### Counterfactual regret

同一 batch、增广和 RNG：

1. dense reference no-grad；
2. counterfactual schedule；
3. 单侧 regret `max(L_cf - L_dense, 0)`。

风险 predictor 是 schedule-conditioned quantile model，fit/calibration/evaluation split 必须隔离。

### Measured cost

正式 lookup 至少以 hardware、precision、batch、candidate schedule、selected rows、p50/p95 为键。线性 group-cost 只可用于 TDD/debug，因为 GPU gather 与 occupancy 非线性。

### Dense fallback

以下任一条件 fail closed 到 RECOMPUTE/dense：

- 首 clip；
- cache invalid/age 超限；
- action 非法；
- signal/transport/output 非有限；
- OOD；
- 无 measured cost；
- 无 calibrated feasible schedule；
- risk/checkpoint 未 ready。

## 8. 训练阶段

### Stage A：runtime/smoke

- forced dense、periodic、HOLD、TRANSPORT、layer/joint schedules；
- 验证 forced-dense 与原 block loop 等价；
- mixed schedule 真正减少 heavy rows；
- 旧 dense checkpoint 只允许 forced baseline。

### Paired replay

- 同 batch/RNG 的 dense 与 counterfactual forward；
- compact ledger 只存 sample/split/schedule/signal/cost/regret；
- 不保存 raw predictions 或 full-token teacher state；
- ledger 不参与 inference。

### Stage B：transport + risk

- 冻结 VideoMAE、projection/head；
- detector 参数冻结但保留 input gradient；
- 训练 transport consistency、counterfactual task loss 和 pinball risk；
- predictor fit/calibration/eval split 隔离；
- scheduler 暂不通过不可微 argmin 反传。

在 `92029ea` 中，seed 3407 的正式 fit/calibration/evaluation 闭环已完成。工程链路、真实 EMA、断点恢复、split 隔离和 checkpoint 重载成立，但科学 gate 为 FAIL：risk 对窗口 regret 的排序相关性为负；cell-risk 求和与窗口 target 的尺度严重错配，导致 coverage 由系统性过预测获得；transport 相对 HOLD 的 detector-regret 改善为正，但 feature-MSE 改善置信区间跨零。按预注册规则没有启动 Stage C。

### Stage C：adapter 联调

- 只解冻 official AdaTAD adapters、transport、risk；
- dense reference branch no-grad；
- 禁止 detector loss 重复聚合；
- 三种子。

### Stage P5：科学 kill gate

需要完整 latency、high-IoU、short-action 和 periodic baseline 结果。由于 P3 已失败，该阶段未解锁。

## 9. 本地实现库存

`92029ea` tree 在 `78d4c00` 基础上包含：

```text
opentad/models/chronotransport/
  actions.py
  cache.py
  transport.py
  risk.py
  scheduler.py
  runtime.py
  replay.py
  training.py
  losses.py
  profiler.py
  cost_lookup.py
  formal_stage_b.py
```

配置：Stage-A、Stage-B、Stage-C。

工具/脚本：validator、checkpoint checker、schedule profiler、paired replay、Stage-B trainer、formal Stage-B runner、GPU1 launchers、N16R4 verifier。

测试覆盖：action schema、layer groups、cache、HOLD invariance、latest-based transport、risk/scheduler、fail-closed、forced-dense equivalence、mixed heavy-row reduction、compact replay、split isolation、Stage-B/C trainable sets、cost lookup、real AdaTAD smoke contracts。

## 10. 实现提交谱系

| Commit | 内容 |
| --- | --- |
| `6e4bc54` | 初始 ChronoTransport 设计 |
| `2bb3456` | 明确 48 clips、384 tubelets 与执行粒度 |
| `627c5ab` | 吸收评审后的实施计划 |
| `6979fed` | Stage-A runtime、cache、scheduler、ViT adapter integration |
| `9ed5ab7` | paired replay、Stage-B/C training、cost lookup |
| `3905cde` | N16R4 verification |
| `3e99056` / `b76c373` / `0500e51` | remote compatibility、cost debug、smoke gate 修复 |
| `cfaa88e` | real OpenTAD paired replay deployment |
| `78d4c00` | Stage-B gate 与 deployment |
| `026f127` | 该分支自己的 C3/DUCA Research Wiki 与原始任务归档 |
| `92029ea` | 正式 Stage-B fit/calibration/evaluation；P3 science gate FAIL |

这些 15 commits 当前只存在本地 branch，尚未推至 `origin/codex/c3-coarse-clean-20260702`。

## 11. 已得到的工程证据

根据 commit 内方法记录：

- focused core/adapter/replay/training tests 通过；
- real AdaTAD/VideoMAE-S Stage-A 单 batch smoke 完成；
- paired dense replay 达到近零 regret 与 deterministic ledger gate；
- Stage-B 单步更新了 ChronoTransport 动态参数且冻结 detector 状态未变化；
- 正式 Stage-B 的 manifest、训练、EMA、恢复、校准、独立评估和统计 gate 可运行；
- checkpoint 与 gate 仍禁止 deployment/metric/latency/paper claims。

正式 P3 还给出了可复核的负结论：risk 排序失败，feature transport 优势不稳定。它没有证明 mAP@0.7 被保护或 latency 达到论文门槛。

## 12. Kill criteria

设计预注册了以下停止线：

- full-stack p50 latency saving 不足 15%；
- periodic baseline 在三 seed 置信区间内持平；
- mAP@0.7 或 shortest-duration quartile 下降超过 1.5 absolute；
- scheduler+transport+cache overhead 超过重算收益 40%；
- calibrated risk 与 counterfactual regret 相关接近零。

当前 P3 已失败且 P5 未执行，因此不能称为科学完成，也不能继续解锁 Stage C。

## 13. 与 MoD/已有工作的关系

接近点：

- 按 layer/time 改变执行深度；
- feature cache/reuse/transport；
- scheduler 选择条件计算；
- dense grid 下跳过部分 heavy compute。

声称的差异：

- 高-IoU endpoint/short-action counterfactual regret；
- time x layer 三动作而非只 skip layer；
- dense TAD grid 和 AdaTAD adapter 保留；
- calibrated risk + measured cost fail-closed。

但这些差异尚未通过科学实验，因此用户认为“更复杂的 MoD，但有效性和创新性未必足够”是合理批评。

## 14. 为什么用户最终否决为当前主线

1. 模型单位受 16-frame clip、tubelet 和 layer grouping 限制，显得僵硬；
2. 与 MoD、DFF、AdaFuse、feature transport/caching 的 collision 强；
3. 需要同时证明 runtime、risk、transport、cost、high-IoU 和跨任务，实验面过大；
4. 它是 backbone conditional-compute 系统，不是用户希望的独立新 TAD detector；
5. P5 关键科学证据尚未完成；
6. 正式 P3 已出现 risk 尺度错配和负排序相关；
7. 本地实现还没有推到远端 branch，复现入口不完整。

## 15. 与 PhysTime 的关系

ChronoTransport 保持 dense detector grid，但不重新定义不规则观测的 TAD 几何；PhysTime 直接把 detector 定义在真实 timestamp/support measure 上。

未来两者可以组合：PhysTime 负责时间几何，ChronoTransport 负责 feature compute reuse。但当前严禁合并，因为会失去归因并扩大方法范围。

## 16. 恢复条件

只有满足以下条件才允许重新评估：

- 将 15 个本地 commits 推送并固定公开 SHA；
- 重新定义 cell/window risk 聚合与 target 尺度，并在新预注册 seed 上通过 P3；
- 完成 measured cost 与有效 calibration；
- periodic/hold baselines 下仍有显著收益；
- high-IoU 和短动作通过 kill gate；
- novelty audit 能清楚区分 MoD/feature reuse；
- 用户重新选择 conditional-compute 系统而非独立 TAD detector 为主线。

## 17. 当前裁决

ChronoTransport 是**已实现正式单种子 Stage-B、但该科学 gate 已失败且当前暂停的独立路线**。不能把它简单写成“只有 idea”，也不能写成“已经有效或可部署”。
