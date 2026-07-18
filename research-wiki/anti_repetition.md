# Anti-Repetition Contract

## 2026-07-18 Full60 Guardrail

- The user has explicitly authorized the matched 60-epoch survivor run; do not
  cancel it because older wiki text says full60 was awaiting authorization.
- Only `selected_axis` and `physical_metric` belong to this run. Do not add G1b,
  DUCA, a learned selector, interpolation, a new sampler, or a dynamic budget.
- `scheduler.max_epoch` and `workflow.end_epoch` must both be 60. Stopping a
  100-epoch cosine schedule at epoch 60 is not an admissible full60 result.
- A Slurm completion code is insufficient. Both arms require independently
  recomputed epoch-59 mAP and replayable finite online/EMA final checkpoints.
- Epoch-41/43/45/47/49/51/53/55/57 interim validations consistently favor physical-metric;
  the latest is `41.37/57.65` selected-axis/physical-metric. This is a strong
  running signal, not a terminal full60 result. Do not promote it to
  `full60-single-seed-supported` before both completion artifacts pass.
- A single-seed THUMOS result may be called `full60-single-seed-supported`, not
  `paper_ready`.

## 2026-07-16 PhysTime GT/window boundary guardrail

- 不再允许 `BuildPhysTimeRawFrameGeometry` 因 GT segment 落在 end-exclusive window 外一点点而在训练中随机抛错。必须先定位 `video_name`、dense crop window、selected raw frame window、秒坐标 domain，再执行可审计的 clamp/filter。
- clamp/filter 只能发生在 train-time GT 转秒坐标之后：segment 与 `[domain_start_sec, domain_end_sec]` 相交则 clamp 到窗口内；clamp 后长度不大于 eps 的 segment 必须过滤；`gt_labels` 必须同步过滤。
- 每次修复都必须写入 `phystime_gt_boundary_audit`，至少包含原始/保留/过滤/裁剪数量、越界幅度、filtered/clamped indices、视频名和窗口元数据。不能把这类问题伪装成数据加载偶发错误，也不能静默吞掉。

- 不得把 focused tests 或独立审查前两轮修复称为部署完成。G1a 必须先取得第三轮零 P0/P1，再绑定 clean commit/tree；真实 gate 未通过时不得提交 pilot，pilot 未产生原始 mAP 时不得写方法 claim。
- gate 的 optimizer 证据必须绑定固定参数名称集合，并逐步满足 state count 完整、min=max=当前 step；不得用最大 step、动态 `requires_grad` hash 或 buffer 变化替代真实参数更新。
- production `drop_last` 必须来自实际 DataLoader 属性；CPU batch 不得被 gate 原地搬到 GPU 并跨步骤持有；正式训练与 gate 必须显式绑定同源 sampler/generator seed。

开始方法修改、实验部署、论文改写或外部讨论前，必须先读本文件与 `query_pack.md`。

## 禁止回退

- 不把 DUCA、X3D、SlowFast、ChronoTransport 或 feature-token pilot 恢复为当前论文主线，除非有新的 superseding decision 和新证据。
- 不把 selected-rank 当物理时间，不把 GT 或预测边界映射到 selected-rank；只允许从秒坐标导出原视频帧号。
- 不用 Voronoi/support 扩张填满真实缺失区，不用 learned selector、actionness、teacher、oracle、ledger 或动态 K 污染 K384 三头主比较。
- 不混用不同 commit、采样、增强、checkpoint、schedule、seed、NMS 或 selected indices 的结果。
- 不把 smoke、one-step、gradient proof、进程存活或 epoch 0 loss 当成 mAP 与论文 claim。

## PhysTime 数值教训

- masked softmax 必须先把未覆盖 logits 置为 `-inf` 再求指数；禁止先 `exp` 后乘零，否则 AMP 下会出现 `inf * 0 -> NaN`。
- 单视频 one-step gate 只能证明局部合同，不能覆盖批间时长、support、mask 与 logit 极值；正式训练至少要越过首个 logging window，并扫描每个 leaf loss 的非有限值。
- gate 通过后 formal 仍可能揭示实现错误；此时必须将 gate 与 full-run 证据分级记录，旧作业降为 diagnostic，并以同一修复 commit 重跑全部 matched heads。
- 只越过 epoch 0 或首个 logging window 仍不足以证明稳定；`0bbf0e9` 的 PhysTime 在 epoch 1 end 才首次记录全 NaN，后续 gate 必须执行多 optimizer step 并 fail-closed。
- 不要把 AMP 缩放后的纯 Inf 与模型 NaN 混为一谈。先在 `unscale_` 后按参数记录 NaN/Inf，再决定：纯 Inf 且 scale 正常下降可在严格次数预算内恢复；任何 NaN、参数污染或跳步超限立即失败。绝不能对 Inf 梯度继续做 `clip_grad_norm_`。
- 单 GPU 不启用 FP16 DDP bucket compression；它没有通信收益，还会放大 scaled-gradient 溢出风险。PhysTime matched 协议固定 `amp_init_scale=1024`。
- 正式 gate 必须实际构建 evaluator 并验证 annotation/class-map 解析；训练配置能读数据不等于 evaluator 的独立相对路径可用。

## PhysTime 性能诊断教训

- 不把当前三头称为“仅检测头/仅坐标表示隔离”：PhysTime 同时删除了 ActionFormer temporal projection、跨 query 上下文并显著缩小可训练容量。
- 不把 raw absolute seconds 直接当 content embedding；秒坐标用于几何、assignment、decode 和 evaluation，表征输入必须做窗口/域归一化并单独审计尺度贡献。
- 不以“query 覆盖若干观测”证明 support integration 有效；必须同时报告 effective observation count、content/relative logit span 和层级坍缩。
- 不用延长训练、调 NMS 或增大 endpoint loss 处理短动作崩溃；先匹配候选密度、target assignment、容量和时序上下文。
- 不把 PhysTime 1.0 的负结果外推为 physical-time TAD 无效；当前实验首先证明的是实现与对照存在架构混杂。
- 不在 PhysTime 1.0 上继续调 endpoint、NMS、训练长度或单独 attention weight；该实现已经冻结为负基线。
- 不再把原生 192 tubelet feature 插值为 384 后与 384 raw supports 一一绑定，并把长度相等称为 feature provenance。
- 不把一个已融合两帧的 tubelet token 当成两个可独立相加的 feature values；multi-atom 首先只是 set-valued anchor provenance。
- 不在同一个“coordinate-only”实验里同时引入 `J192 -> Q384` lift、候选恢复和 support-mass operator；`K`、`J`、`Q` 必须分别审计。
- 不把 zero-coverage query 直接删除，也不把 gap token/跨 query 推断描述成已观测 feature 或 dense imputation。
- 不把 `SM-PTAF` 的外部公式、伪代码或 patch map 写成 `implemented`、`tested` 或已有 mAP。
- 不用参数总数接近替代容量公平；projection 深度、跨 query context、candidate topology、assignment 与训练更新必须同时对齐。
- 不把“原始 AdaTAD 使用 interpolation”误解为 G1a 也必须立即恢复 interpolation。插值可作为两臂共享的中性 query-grid lift，但必须单独归因，且永远不能把插值位置计作新增原始观测。
- 物理点写入和候选 mask 必须使用不同张量：rank/slot center 在写入物理中心前必须 clone。禁止把已被物理秒坐标原地改写的 view 与 selected count 比较，否则会静默删除合法候选并伪造性能下降。
- static precheck、真实 CUDA gate 与 pilot 完成是三种不同证据；只有 commit/tree/config/data/checkpoint 哈希一致、真实三步更新和正式 evaluator 通过，才允许启动依赖 pilot。
- 不能只在 VideoMAE 输入前把重复 padding 像素置零：无效 token 还会经 attention K/V、残差、MLP、TIA 卷积和 norm 回流污染有效 token；严格隔离必须逐层传播 mask，并以 padding 反事实和无效输入零梯度验证。
- gate 的推理尾样本和 evaluator 必须来自真实 test split；使用 validation/train 样本即使能跑 NMS 也不能证明测试闭环。
- 数据集 provenance 不能只哈希文件名与大小；必须使用完整文件内容摘要。checkpoint 不能只检查文件存在或字节非空；必须真实反序列化并从 manifest 独立重算 evaluator。
- FPS 容差不能凭经验拍定或默认为零；先全量审计 decoder FPS、annotation duration 与 frame count，再把保守阈值写入 train/test 同一合同，并由正式 gate 重算。
- 全量 timebase 审计的范围必须来自正式 dataset `data_list`，不能直接把数据根目录每个 MP4 都假设为 evaluator 样本；目录中的未引用文件必须显式披露并纳入完整 inventory 哈希，被 dataset 消费但缺失的文件则必须 fail-closed。
- 模型 state-dict 摘要必须覆盖 0 维标量 buffer；不同元素大小的 dtype byte-view 前先 reshape 为一维，不能假设所有参数/缓冲区至少一维。
- 多步真实 gate 不能伪装成“每个单样本所有参数族梯度都必须非零”。ActionFormer 回归头末端 ReLU 可在某个有效样本上让参数梯度为零，即使该步有正 assignment 和正回归损失；正确合同是每步正 assignment、正回归监督与全部有限，关键通路逐步非零，而回归参数族必须在固定三步聚合中至少一次非零。不能删除回归梯度证明，也不能仅凭正 loss 判定梯度已连通。

## 当前唯一主线

PhysTime 1.0 的 THUMOS14 raw-RGB/K384 三头实验已经完成并冻结。当前唯一执行阶段是 P0 rebuild：native tubelet provenance + capacity/context/candidate/assignment-matched coordinate-only control；`SM-PTAF` 仍为 designed candidate。当前状态以 `query_pack.md`、`current_direction.md`、`experiments/phystime-adatad-k384.md` 和 `docs/evaluation/results.md` 为准。

## 2026-07-17 G1b Medium-Run Guardrail

- G1b SDPQ 的 20-epoch medium run 只能证明该实现稳定训练且持续学习，不能与旧 G1a 六轮 pilot 直接比较。
- 禁止把 `30.88%` 与旧 selected-axis `10.26%`、physical-metric `10.56%` 的差值解释为结构收益；commit、训练轮数和训练阶段均不匹配。
- 下一项决定性实验固定为同一 commit、同 K384/J192、同采样、同 seed、同 20 epochs 的 selected-axis / physical-metric / G1b SDPQ 三臂比较。
- 三臂训练必须共享 gate、dataset manifest、预训练权重、优化器、scheduler、验证周期与 evaluator；任一差异必须在 manifest 中显式列出并解释。
- 若正式评价使用 EMA，最终轻量 checkpoint 必须保留 `state_dict_ema`；只保存 online 权重再声称 checkpoint 可复现评价属于证据断裂。
- 三臂结果完成前，不启动 60-epoch full train，不创建 paper claim，也不把 `medium_run_supported` 写成 `paper_ready`。

## 2026-07-17 Matched-Medium Result Guardrail

- 三臂结果已经完成；禁止继续写成 `experiment_running` 或 mAP=NA。
- matched 结果支持的是 physical-time metric：physical-metric `44.88%` 对 selected-axis `30.42%` Avg-mAP。
- 当前结果不支持 G1b SDPQ 结构优势：G1b `30.88%`，只在高 IoU 有小幅改善，同时低 IoU 覆盖下降。
- 禁止把 physical-metric 的收益归到 SDPQ、support-measure operator 或“continuous-time TAD”整体概念。
- 单 seed、20 epochs、单 THUMOS 只能写 `matched-medium-supported`；多 seed、完整 schedule、成本和跨数据集证据前不得写 `paper_ready`。
- 不因 medium survivor 自动提交 60-epoch full train；先明确复现矩阵、机制拆分和停止条件。
