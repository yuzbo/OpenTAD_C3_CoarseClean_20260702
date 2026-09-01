# DUCA `0ea4e15` Pro 审计吸收记录

## 来源与完整性

- 原附件：`6065c548-e502-42e2-9b2c-d6bd4a7e2807/pasted-text.txt`
- 原附件 SHA-256：`60D4D9414F3F2D90EC9A0CE0F2D704D2184D8EEED9CE2FBB5315932997CEE957`
- 原文归档：`docs/methods/reviews/2026-07-13-0ea4e15-duca-fsu-pro-audit-review-raw.txt`
- 固定审计对象：分支 `codex/duca-transition-only-20260711`，代码提交
  `0ea4e15d08f2c4f92e4b927ea356f24f0a0b477d`
- reviewer 明确没有访问集群作业、checkpoint 或原始日志，也没有运行 corrected CUDA
  gate/full train。因此，代码判断与实验数字的证据等级必须分开。

## 总裁决

吸收 reviewer 的 `HOLD` 裁决：`0ea4e15` 是经过 focused tests 的 transition-only
候选，不是论文主方法成品。研究问题仍值得一次决定性验证，但 corrected matched
baseline、hard utility 对齐、时间几何和真实总成本均未闭环。

本轮不把 reviewer 推荐的 `DUCA-FSU-384` 升级为已实现路线。其状态仅为
`discussed/design proposed`，更不能改写 C1/C3/C4 的实验状态。

## 接受为代码事实的内容

1. 当前 coarse 模块应诚实命名为“自定义低分辨率 spatial stem + 官方 ASFormer
   temporal module 的二分类动作性适配”，不能称整个 probe 为 official ASFormer。
2. 当前 detector route 只把 detector signal 送到 utility scorer，不更新 spatial stem、
   ASFormer encoder 或 action head。它是 protected multi-loss training，不是 detector 与
   coarse representation 的完全端到端共适应。
3. `0ea4e15` 修复了 transition-only uniform reference，但 `acquisition.py` 的 legacy/direct
   `stable_selection` 仍存在同类 midpoint reference。故 `1159415` 也只能保留为旧行为诊断。
4. 当前 hard Viterbi 与 soft forward-backward 属于同一个 exact-K/max-gap 可行族；真正未证
   明的是 raw-pixel zero-forward surrogate 是否与 hard feasible replacement 同向。
5. selected-axis 的后处理逆映射不能消除 ActionFormer 内部按 selected rank 等间距建模的
   几何偏差。是否构成主要瓶颈仍需 same-selected-frames 实验，而不是靠推断定案。
6. 当前数据通路先 dense decode/resize，并通常把 768 帧送到 GPU 后才执行 selector；因此
   现阶段最多声称减少 heavy-backbone compute，不能声称输入、I/O 或全栈成本减半。
7. 短窗口 `effectiveK<K` 的零帧 padding、custom stem 的较高 MAC、解析 profiler 漏算
   ASFormer decoder route，都是成本和稳定性审计项。
8. formal gate 只证明 backward connectivity，且旧 gate `1159395` 已过期；它不能替代新
   commit 的真实 optimizer step、utility alignment、mAP 或成本证据。

## 对旧实验的合法解释

- `1159414` 55.67：degenerate DP tie-break diagnostic，不是 exact-uniform。
- `1159415` 57.71：direct-a5 旧代码诊断，stable warmup 仍含 midpoint 残留。
- `1159416` 64.34：错误 homotopy 起点下的 learned-policy 诊断；不能证明超过 matched uniform。
- `1159417` 63.55：当前 bridge 的负面诊断；不能证明 detector-gradient benefit。
- 历史 64.352/65.696：真实但跨协议的 uniform anchors，不能填入当前 matched 表。

## 三条路线的吸收

### 路线 A：最小修复 current bridge

只允许作为诊断路线。先统一 repo-wide uniform helper、审计 hard path exposure，并通过
one-swap finite-difference correlation；否则不再投入 beta 调参。reviewer 对 score-level
z-normalized homotopy 的质疑成立，但 batch/sample-level hard exposure 仍只是待测设计。

### 路线 B：DUCA-FSU

核心思想是用 train-only 的真实 hard feasible swap detector-loss gain，蒸馏
`u_t-u_s`，替代 raw-pixel soft bridge。它把“detector 梯度必须直接反传”改成“detector
counterfactual utility 监督 scorer”。这是科学假设与 C4 口径的实质变化，不能悄悄当作
原 bridge 的实现细节。

只有以下条件成立时才值得实施为候选：

- one-swap gain 在固定 detector/checkpoint 下稳定且可重复；
- transition-derived utility difference 能预测 gain 的方向与排序；
- counterfactual branch 严格 no-grad，不污染 detector/normalizer/RNG；
- feasible swap 生成保持 exact-K/max-gap；
- 推理不执行 counterfactual，不读取 GT/teacher/cache；
- 增加的训练成本与主方法收益可审计。

### 路线 C：Residual-Innovation Multigrid

若 action-state transition 或 additive one-swap utility 失败，可转向非语义的多尺度重建
残差路线。它目前只是 fallback hypothesis，并与 MGSampler/motion sampling 有明显近邻，
不能因概念新鲜就自动晋升。

## 不完全接受或必须保留条件的建议

1. **不接受“DUCA-FSU 是唯一最终模型”作为事实。** 它是 reviewer 的优先建议，尚无代码、
   gate、one-swap 统计或 mAP。
2. **不预先接受 physical-time RGB interpolation 必然更好。** PhysTime v1 已有负结果，
   reviewer 的 reconstruction 与旧 PhysTime 不完全相同，但必须通过 same-selected-frames
   selected-axis/physical-grid/reconstruction 对照后才能选择。
3. **不把 27%/24% 等根因概率或 AUROC/Spearman/latency 阈值当自然常数。** 它们是审计
   prior 和 proposed stopping rules，需要在 protocol 中预注册，不能写成实验事实。
4. **不自动采用 reviewer 给出的代码片段。** common uniform helper、swap loss、physical
   reconstruction 和 hard exposure 代码都需本地单测、数值边界与真实 detector 状态审计。
5. **不因删除 bridge 就宣称联合协同训练更完整。** FSU 是单 checkpoint、多损失、保护性
   路由，但 detector 不再直接更新 coarse trunk；论文必须如实称为 counterfactual utility
   distillation，而非 full detector-gradient co-adaptation。

## 吸收后的允许顺序

1. 先把所有 uniform/stable/control route 收敛到唯一 rounded-endpoint helper，并增加
   repo-wide midpoint ban 与 direct warmup exact-position test。
2. 在不跑 full train 的条件下先做 coarse AUROC/AUPRC/ECE、learned-vs-shuffle/noise、
   one-swap alignment、same-selected-frames geometry 和 trained-checkpoint full-stack cost。
3. 只有 transition signal、counterfactual utility 和成本三者通过，才实现并训练 FSU。
4. corrected fixed-384 先于 K=256/128、dynamic MUST、第二 detector 和复杂 probe。

## 当前状态

- `0ea4e15`：`tested`，未运行新 formal CUDA gate，未运行 corrected replacement full train。
- current raw-pixel bridge：实现存在，utility benefit `unproven`，旧 beta 对照不支持 C4。
- DUCA-FSU：`discussed/design proposed`，未实现、未测试、未部署。
- 论文主张：维持 `HOLD`。

