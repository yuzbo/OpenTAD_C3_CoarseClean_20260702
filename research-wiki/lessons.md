# 经验、教训与禁止回退清单

## A. 科学问题

1. **动作覆盖不是检测效用。** 选到更多动作内部帧，不代表起止边界更准，也不保证 mAP@0.6/0.7。
2. **边界 recall 不是边界聚集。** `boundary_support@r` 只说明边界附近至少有点，不说明选点分布集中，更不说明因果提升 detector proposal quality。
3. **GT boundary proxy 不是 detector utility。** 除非目标来自 train-only detector assignment/loss/gradient/counterfactual sensitivity，否则必须叫 `boundary_utility_proxy`。
4. **不规则观测的核心问题是时间度量。** 输出坐标 remap 正确，不代表 projection、attention、Conv1d、pyramid 和 assignment 内部理解真实时间间隔。
5. **新颖性不能只靠“continuous time”。** mTAN 已做不规则连续时间注意力，TE-TAD 已做实际时间线坐标，Temporal Robustness 已研究缺帧定位退化，LiquidTAD 已引入连续动力学启发的 TAD。

## B. 模型与训练

1. **leaf loss 只能聚合一次。** 不得同时返回 leaf、alias 和 `total_loss` 再被 detector 重复求和。
2. **hard forward 与 soft backward 必须来自同一策略。** 两套不同 top-k/resample 只会产生“可导但方向错误”的梯度。
3. **one-step gradient proof 不是效果证据。** 还要证明 hard one-swap utility、参数实际更新、selected positions 移动和 full-run 高-IoU mAP。
4. **稳定 warmup 不能改变 DDP 参数使用集合。** 后期开启新 head 会触发 static-graph mismatch；可用零值依赖保持图结构不变。
5. **uint8 原始窗口不能直接进入 soft einsum。** 桥接前必须安全转浮点，同时保持 hard forward 数值一致。
6. **optimizer coverage 必须逐参数审计。** Conv2d、BatchNorm、Embedding、probe、selector、budget controller 或 adapter 不能因旧 `get_optim_groups()` 漏掉。
7. **多阶段不是天然错误，但不能伪装成最终单模型。** teacher/ledger 可作 train-only 辅助与诊断，最终推理不得依赖独立导出或缓存。

## C. 选帧路线的具体教训

1. GAS-VT 的高覆盖和早期 mAP 不证明 value transport；fixed384 本身已暴露一半时间网格。
2. GAS-VT 训练/应用时 target budget 条件不一致会改变 `budget_pressure`、`gap_urgency` 等特征语义。
3. score-all + top-k + repair 不是 sequential value transport。
4. hard max-gap repair 能在元数据声称“无 uniform scaffold”时仍生成接近 uniform 的格点。
5. move25/move50 的聚集偏移可能来自二分类标签过粗、probe stride/smoothing 延迟、selector loss 与 hard decode 不同构，而不应简单归因于“粗分类没学好”。
6. actionness 必须以二分类 GT 监督粗分类器，但间接选择若存在，必须以 transition/start/end/utility 优先，并接收 detector 梯度。
7. X3D/SlowFast 这类 frozen prior 不是 train-free 就等于低成本；一次 dense inference 仍可能比省下的 heavy backbone 成本更贵。

## D. 动态预算

1. fixed budget curve 是 sweep，不是 dynamic policy。
2. expected soft K、hard requested K、unique K、padded K、实际 backbone K 必须分别记录。
3. budget controller 在 64/384 间跳通常是离散 bucket、温度、dual update 与 padded execution 不一致的结果，不是合理自适应。
4. 若 detector 总是 pad 到同一 cap，dynamic MUST 不能声称 variable compute。
5. 动态方法必须和 matched-average-K fixed baseline 比较 accuracy-latency Pareto。

## E. 实验与证据

1. 不同 commit 的结果不可混表，旧 run 只能作为诊断。
2. smoke、precheck、contract test、one-step gradient、full mAP 是五种不同证据，不能互相替代。
3. 缺失 mAP 必须记 `NA`，不能画成零。
4. geometry 要按 action instance 连接 detector best proposal tIoU、TP/FN 和边界误差，才能讨论因果链。
5. 主表必须共享数据、采样、backbone、预训练、schedule、seed、NMS、eval cadence 和坐标契约。
6. val/test 不得使用 GT、teacher、oracle boundary、raw prediction cache 或 ledger 决策。
7. 下载失败、队列等待、旧逻辑取消和方法失败必须分类，不得统称“实验失败”。

## F. PhysTime 专属禁区

1. support interval 必须来自可审计原始单元，不能跨缺失区扩张。
2. 重叠 support 只能按 midpoint clip 消除双计数，不能扩展填洞。
3. query 数由物理时长与 spacing 决定，不能由 K 或 selected-rank stride 决定。
4. GT、预测、NMS 统一用秒；允许导出原视频帧号，不允许导出 selected-rank 边界。
5. primary head comparison 使用同一无学习采样，避免把 selector 收益混进 PhysTime head 收益。
6. feature-token 代码是算子资产，不是 raw-video 论文证据。

## G. 启动新工作前的五问

1. 这个改动直接解决 gap map 中哪个 gap？
2. 它是否重走了 idea catalog 中已否定的路线？
3. 它改变了几何、监督、计算还是仅增加一个权重？
4. 最小反例和 kill criterion 是什么？
5. 证据将写到哪个 experiment ID，能否公平回答一个主张？
