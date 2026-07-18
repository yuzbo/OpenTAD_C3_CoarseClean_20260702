# Research Wiki Query Pack

## 2026-07-18 Active Full60 Override

The user explicitly authorized the 60-epoch survivor validation, superseding
older text below that says it must not be started automatically. Commit
`0dc5851`, tree `bddc9b9`, and clean snapshot
`opentad_phystime_g1_full60_0dc5851_20260718` are frozen. Real gate `1170945`
passed; matched selected-axis `1170946` and physical-metric `1170947` are
running with K384/J192, seed 42, no interpolation, and a true 60-epoch cosine
schedule. Matched validations at epochs 41/43/45/47/49 consistently preserve the
physical-metric lead; the latest is selected-axis/physical-metric
`41.16/57.54%` Avg-mAP (`+16.37`), with physical-metric ahead at every IoU
threshold. This is interim evidence only: epoch 59, independent
recomputation, final online/EMA checkpoint validation, and both completion
artifacts remain pending. G1b is not part of this survivor run.

更新时间：2026-07-18。本文只保留当前决策所需的压缩记忆；完整历史见 `research-wiki/routes/`、各实验页与 `docs/evaluation/results.md`。

## 当前方向与状态

唯一执行主线是 **离线 PhysTime-TAD 稀疏检测头**，不是 DUCA 选帧插件，也不是 Online TAD。问题定义是：给定同一无学习、无 GT 的不规则原始帧采样，检测器能否直接使用真实秒时间戳和稀疏观测 support，在物理时间上完成分类、边界回归、NMS 与评价，并保护高 IoU 定位。

长期目标是独立 TAD 方法；AdaTAD/VideoMAE 只是当前 raw-video 端到端载体。GT 与预测始终使用原视频秒坐标，可按 `round(t*fps)` 导出原视频帧号，但不得映射到 selected-rank。

当前 matched 实现固定 K=384 原始观测，经 VideoMAE 得到 J=192 原生 tubelet token，不做 J192→384 feature interpolation。它比较三种头：selected-axis ActionFormer、使用真实秒度量的 physical-metric ActionFormer，以及把稀疏 support 与物理 query 解耦的 G1b SDPQ。

最新完成证据：commit `5e8a821` 的同 commit、同数据、同 K384/J192、同 seed=42、同 20 epochs 三臂实验已全部通过 gate、训练、独立评价和 online/EMA checkpoint validator。原始结果见 `docs/evaluation/results.md`：selected-axis `30.42%`、physical-metric `44.88%`、G1b SDPQ `30.88%` Avg-mAP。

当前裁决：

1. physical-metric 相对 selected-axis 提升 `+14.46` Avg-mAP 和 `+9.82` mAP@0.7，是明确的 matched-medium survivor；
2. G1b SDPQ 相对 selected-axis 仅 `+0.46` Avg-mAP，虽在 mAP@0.6/0.7 分别提高 `+2.11/+2.42`，但 mAP@0.3 降低 `-3.83`；
3. 因此证据支持真实物理时间度量，不支持当前 SDPQ 结构优于 physical-metric control。

下一项决定性任务不是继续放大 G1b，而是验证 physical-metric survivor：先做同配置多 seed/复现实验和 proposal/边界/短动作/gap 分解，再由用户决定是否进入 60-epoch full schedule。当前结果是 `matched-medium-supported`，不是 `paper_ready`。

## 当前核心科学问题

1. physical-metric 的大幅收益能否在多 seed、完整 schedule 和第二数据集复现？
2. 收益来自秒域 assignment、秒域回归、候选有效性还是后处理中的哪一环？
3. G1b 为何提高高 IoU 却损失低 IoU：是 support observability、分类召回、query coverage 还是 NMS 排序？
4. support pooling、query scale、positive assignment、分类排序和 NMS 中，哪一项限制 mAP@0.7？
5. K384 raw-video 的计算节省能否覆盖额外物理时间头开销？
6. 单 THUMOS、单 seed 的信号能否在多种子、第二数据集和不同固定采样族上复现？

## 最高优先级缺口

- `G4` 公平隔离：same-commit/schedule/seed 三臂 medium comparison 已完成；下一步需要多 seed 和完整 schedule 复现，旧的不同轮数结果仍禁止横比。
- `G2` provenance：K384 raw observations、J192 tubelet tokens、Q0 与多尺度候选必须分开审计；一个 tubelet 融合两帧，只能先称 multi-atom support anchor。
- `G5` 高 IoU：需要 proposal recall、class-aware recall、边界 MAE、短动作与 gap 条件分解，不能只看 Avg-mAP。
- `G7` 成本与泛化：缺完整 decode/VideoMAE/head latency、FLOPs、显存和第二数据集证据。
- `G3` 新颖性：mTAN、TE-TAD、RCL、FrameDrop/TRC、LiquidTAD 已占据连续时间或鲁棒 TAD 邻域；论文只能主张“不规则观测 support 与物理 query 解耦、无 dense imputation 的稀疏检测头”，不能宽泛声称首个 continuous-time TAD。

## 已否定或降级的路线

- C3/PAction/GAS-VT/lattice/move：是选帧与几何归因工具，不是当前最终检测头。
- DUCA：虽完成 official head gradient、no-leak、exact-K/max-gap 等工程闭环，但仍接近复杂 score+top-k+scaffold；旧结果不支持最终主张，且项目当前不做在线选帧插件。
- MUST dynamic budget：expected/hard/unique/padded/backbone K 不统一，动态预算降级 appendix。
- X3D/SlowFast Fast：dense frozen prior 过慢且带 Kinetics 先验，只能作 appendix；不得恢复为当前主 probe。
- ChronoTransport/DCRT：接近 MoD/feature reuse，系统归因风险高；工程 Stage-B 存在，但 science gate 为负，当前暂停。
- PhysTime feature-token pilot：已取消；不能替代 raw-video 端到端证据。
- PhysTime-AdaTAD 1.0：三头 full run 已完成并冻结为负基线。它同时改变坐标、投影、上下文、候选和容量，不能裁决 physical-time 假设。
- G1a 六轮 pilot：只能作早期学习诊断，不能与 G1b 20轮结果比较。
- 仅延长训练、调 endpoint loss 或 NMS：不能修复候选/assignment/上下文混杂，禁止作为主要解释。

## 关键实现与实验教训

- `K` 是原始观测槽位上限，`J` 是原生 tubelet token 数，`Q` 是候选数；三者不可混写。
- 短窗口可有少于 384 个有效观测，不能把 K384 误写为每个样本必须有 384 个有效索引。
- GT/window 使用 end-exclusive 物理域。相交 segment 可审计 clamp，空 segment filter，标签同步；必须记录视频、窗口、越界量与受影响索引。
- masked softmax 必须 mask-before-exp，避免 AMP 下 `inf*0 -> NaN`。
- gate 必须使用真实 DataLoader、优化器、scheduler、EMA、evaluator 与多步参数更新；one-step 或进程存活不等于方法有效。
- 结果必须绑定 commit/tree/config/data/checkpoint 哈希，并由独立 evaluator 重算；checkpoint 必须反序列化和有限性检查。
- `/data` 曾因逐 epoch 大 checkpoint 写满。medium suite 固定 final-only lightweight checkpoint，并用原子完成 artifact。
- 评价若使用 EMA，final-only lightweight checkpoint 仍必须保存 `state_dict_ema`；省略 optimizer/scheduler，不省略被评价权重。
- selected-axis、physical-metric 与 G1b 必须共享数据、采样、seed、训练轮数和评价协议；否则差值不可归因。
- 程序跑完只代表 `runnable`；gate 通过、medium evidence、full evidence、`paper_ready` 必须分级。

## 活跃机制链

不规则原始帧与原帧号/FPS → 非扩张物理 support 与 native tubelet provenance → support-decoupled physical query → measure projection / matched ActionFormer control → 秒坐标 assignment、回归、decode、NMS → mAP、高 IoU、短动作、gap robustness 与总成本审计。

## 状态边界

- G1b 20-epoch trainability：`empirically_supported`。
- physical-metric 相对 selected-axis：`matched-medium-supported`。
- G1b 相对 selected-axis：Avg-mAP 优势不成立，高 IoU 有弱正信号；相对 physical-metric 明显落后。
- 三臂 medium suite：`completed`，全部 validator 通过。
- 60-epoch full train：不再由 matched-medium 阻塞，但必须先由用户裁决 survivor、seed 和诊断矩阵，不能自动启动。
- PhysTime 论文主张：尚无 `paper_ready` claim。

## 必读入口

- 当前禁区：`research-wiki/anti_repetition.md`
- 方向：`research-wiki/current_direction.md`
- 完整路线：`research-wiki/routes/phystime-complete-record.md`
- G1b medium：`research-wiki/experiments/phystime-g1b-sdpq-medium20.md`
- 原始结果：`docs/evaluation/results.md`
- 关系图：`research-wiki/graph/edges.jsonl`
