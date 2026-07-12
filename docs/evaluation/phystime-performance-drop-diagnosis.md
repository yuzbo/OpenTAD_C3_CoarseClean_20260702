# PhysTime-AdaTAD 1.0 性能下降诊断

日期：2026-07-12

原始数字唯一来源：`docs/evaluation/results.md`

## 总裁决

性能下降是真实且可复现的。它不是由训练崩溃、错误 checkpoint、秒坐标重复转换、测试窗口缺失、分数自归一化或模型输出生成 GT 导致的。

当前结果否定的是 **PhysTime-AdaTAD 1.0 这一版实现**，而不是 physical-time TAD 研究方向。首轮三头实验同时改变了坐标语义、时序投影架构、跨 query 上下文和可训练容量，因此不能单独裁决物理时间坐标的科学价值。

## 证据链

1. 三个最终作业均完成，无 NaN、OOM、Traceback 或 AMP 跳步崩溃。
2. 最佳 checkpoint 的只读复算逐项复现正式 mAP。
3. 官方评估读取 THUMOS annotation 并计算原始 AP；evaluator 和分数归一化不能解释差距。
4. PhysTime 用更小的 measure projection 和浅层 head 替换了 ActionFormer 的丰富时序投影，也没有同等级跨 query 时序编码器。
5. PhysTime 暴露的有效候选约为 ActionFormer 对照的一半，短动作监督也更稀薄。
6. 学到的 query embedding 被未归一化的绝对秒数主导。
7. 粗尺度 attention 虽覆盖很多观测，实际却坍缩到约两个有效观测；content logits 压过 relative-time logits。
8. 完整预测分解显示：PhysTime 在正确高 IoU 匹配后可以回归较准边界，但候选覆盖、排序质量，尤其亚秒动作明显退化。

## 根因优先级

### P0：架构与容量混杂

当前比较不是只改变坐标的 head isolation。selected-axis 和 physical-grid 保留 ActionFormer projection stack，PhysTime 则删除它。这是最大的解释风险，在任何物理时间主张之前都必须消除。

### P0：Query 尺度与 attention 坍缩

原始 `center_sec` 未归一化就进入 content query，其学得贡献压过归一化坐标与 Fourier 时间特征。在粗尺度，内容相似度压倒相对时间几何，使 support integration 实际坍缩。

### P1：候选密度与短动作监督不匹配

PhysTime 构造的候选和正位置显著更少，严格 IoU 下的损失集中在短动作。这是分辨率设计错误，不能作为 support integration 天生无效的证据。

### P1：Target assignment 不同构

ActionFormer 在歧义位置保留同长度动作标签，PhysTime 只选择一个 `min_index`。这影响了不可忽略的正位置，并给三头比较增加了额外混杂。

### P2：Endpoint loss 的取舍

Endpoint 监督似乎改善了成功匹配样本相对 physical-grid 的边界精度，但没有恢复广泛候选覆盖和排序。只有清除 P0 问题后，才值得单独消融其权重与共享回归塔。

## 已排除因素

- 不存在秒坐标重复转换；
- PhysTime 评估没有映射到 selected-rank；
- 测试窗口协议没有遗漏；
- evaluator GT 来源正确；
- 官方 mAP 没有分数自归一化；
- 最终差距不是训练崩溃造成的；
- 不是简单的 post-NMS 序列化条目不足。

## 最小因果修复顺序

1. 构建等容量、同上下文的 physical-time control：保留 ActionFormer 时序栈，只改变坐标语义。
2. 从 content query 删除原始绝对秒数；表征使用窗口归一化坐标，秒只用于几何、assignment、decode 和 evaluation。
3. 对 content query/key 做 L2 归一化，限制温度，并加入显式 mass-pooling 残差和零初始化 content correction。
4. 将 level-0 候选数与 selected-axis 对齐；优先由有效 K 推导物理 query 数，而不是固定数据集特定秒间隔。
5. 与 ActionFormer 的多标签 target assignment 完全同构。
6. 最后才逐项消融 endpoint loss 与 support integration。

在前五项通过合成、真实 one-step、候选数同构和 attention 行为 gate 之前，不应启动新的 full training。

## 主张边界

当前证据只支持：在最终 THUMOS K384 单种子协议下，PhysTime-AdaTAD 1.0 的当前实现弱于两个 sparse controls；成功匹配后，它相对朴素 physical-grid 存在有限的边界质量收益。

当前证据不支持泛化性、SOTA、完整 accuracy-cost 优势，也不能断言 physical-time modeling 天生更优或更差。
