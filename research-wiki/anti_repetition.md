# Anti-Repetition Contract

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
- 正式 gate 必须实际构建 evaluator 并验证 annotation/class-map 解析；训练配置能读数据不等于 evaluator 的独立相对路径可用。

## 当前唯一主线

`PhysTime-AdaTAD 1.0`：THUMOS14 raw RGB，逻辑 768，确定性、无学习、无 GT 的相同 K=384 不规则采样，比较 selected-axis、physical-grid 与 PhysTime 三头。当前实验 commit 与作业以 `query_pack.md`、`experiments/phystime-adatad-k384.md` 和 `docs/evaluation/results.md` 为准。
