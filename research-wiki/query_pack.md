# Research Wiki Query Pack

更新时间：2026-07-17。本文只保留当前决策所需的压缩记忆；完整历史见 `research-wiki/routes/`、各实验页与 `docs/evaluation/results.md`。

## 当前方向与状态

唯一执行主线是 **离线 PhysTime-TAD 稀疏检测头**，不是 DUCA 选帧插件，也不是 Online TAD。问题定义是：给定同一无学习、无 GT 的不规则原始帧采样，检测器能否直接使用真实秒时间戳和稀疏观测 support，在物理时间上完成分类、边界回归、NMS 与评价，并保护高 IoU 定位。

长期目标是独立 TAD 方法；AdaTAD/VideoMAE 只是当前 raw-video 端到端载体。GT 与预测始终使用原视频秒坐标，可按 `round(t*fps)` 导出原视频帧号，但不得映射到 selected-rank。

当前模型为 G1b SDPQ：K=384 原始观测，经 VideoMAE 得到 J=192 原生 tubelet token；稀疏 support 与物理 query 解耦，`PhysTimeMeasureProjection` 从不规则 support 向物理 query 聚合，`SupportDecoupledPhysicalQueryHead` 在秒坐标上 assignment、回归与解码。不做 J192→384 feature interpolation，不把 gap 填成已观测特征。

最新完成证据：commit `4a57577` 的 G1b 20-epoch medium run 已通过真实 gate、训练和独立评价。它证明 G1b 能稳定训练并持续学习，原始结果见 `docs/evaluation/results.md`；它**没有证明优于基线**，因为现有 G1a selected-axis / physical-metric 只有六轮且 commit 不同。该 run 的 checkpoint 未保存评价所用 EMA 权重，指标可由保存预测重算，但 evaluated-weight 不能精确重放；新三臂 suite 已把 EMA 纳入轻量 checkpoint 合同。

当前决定性任务：在同一新 commit、同 THUMOS 数据、K384/J192、无 GT sampler、seed=42、20 epochs、优化器、scheduler、评价器和 checkpoint 合同下，运行：

1. selected-axis：官方 ActionFormer，均匀 rank-derived 秒轴；
2. physical-metric：官方 ActionFormer，真实物理秒轴；
3. G1b SDPQ：物理 query 与稀疏 support 解耦。

三臂统一 runner、validator、shared G1a+G1b gate 与 Slurm DAG 已实现；本地静态测试与远端 Linux/PyTorch focused suite 已通过。正式 clean snapshot 与队列是当前下一步。只有 matched 20-epoch 结果显示 G1b 有稳定且有意义的优势，才解锁 60-epoch full train。

## 当前核心科学问题

1. 物理时间收益是否真实，还是训练时长、容量、候选数或 assignment 差异造成？
2. support-decoupled query 是否提高高 IoU、短动作和 contiguous-gap 条件下的定位？
3. G1b 的优势若只出现在低 IoU，是否只是候选覆盖增益而非边界定位增益？
4. support pooling、query scale、positive assignment、分类排序和 NMS 中，哪一项限制 mAP@0.7？
5. K384 raw-video 的计算节省能否覆盖额外物理时间头开销？
6. 单 THUMOS、单 seed 的信号能否在多种子、第二数据集和不同固定采样族上复现？

## 最高优先级缺口

- `G4` 公平隔离：必须完成 same-commit/schedule/seed 三臂 medium comparison；不同轮数结果禁止横比。
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
- G1b 相对 matched controls 的优越性：`unknown`。
- 三臂 medium suite 代码：`tested`，正式部署待完成。
- 60-epoch full train：`blocked_by_matched_medium`。
- PhysTime 论文主张：尚无 `paper_ready` claim。

## 必读入口

- 当前禁区：`research-wiki/anti_repetition.md`
- 方向：`research-wiki/current_direction.md`
- 完整路线：`research-wiki/routes/phystime-complete-record.md`
- G1b medium：`research-wiki/experiments/phystime-g1b-sdpq-medium20.md`
- 原始结果：`docs/evaluation/results.md`
- 关系图：`research-wiki/graph/edges.jsonl`
