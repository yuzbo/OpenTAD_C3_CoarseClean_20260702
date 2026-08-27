---
type: claim_index
updated: 2026-08-28
---

# Current Project Claims

## 当前可说

- 本任务是离线全窗口 TAD，不是 Online TAD。方法不得使用 validation/test GT、teacher、
  oracle 或 raw-prediction cache 做推理选择。
- 在同源 seed-42 实验中，K100 与 BPNS-R1 的 final-EMA
  Avg-mAP/mAP@0.6/mAP@0.7 分别为 `68.51/61.19/46.27` 与
  `69.07/61.14/46.57`。这支持“严格连续 K64 原生支持在该单次运行中保持准确率”的有限主张。
- BPNS-R1 在 VideoMAE 前把原生空间输入从 100 降为 64；36% 是结构性 token 减少，不是
  实测加速、能耗或显存结论。
- RC32 temporal carry 在与 MOD32-KV 相同理论重块成本下三项准确率更低；该具体 carry
  机制有有效的单种子负证据。`R1-APM-C32/FULL64` 不减少主干计算且准确率下降；该输入载体
  也有有效负证据。两者都不能外推为所有时序复用无效。

## 当前不能说

- 不能声称 BPNS-R1 降低真实全栈延迟、峰值显存或能耗。旧回放 job `1257281` 在 R1 执行前
  因数值绑定问题停止；最小修正后的唯一替代 job `1258299` 仍在运行，尚无完整
  `profile.json`、`terminal_receipt.json` 或可解释成本结果。
- 不能声称多种子稳定、跨检测器/数据集泛化、论文最终方法或优于所有相关 token
  pruning/caching 方法。
- 不能把 FLOPs、原生 token 数、训练时长、smoke test 或单次实现审查当成模型效果证据。

## 历史主张索引

旧的 C1–C10 节点继续作为 C3/DUCA 路线的来源和负证据索引，不代表当前论文主张：
[C1](C1.md)、[C2](C2.md)、[C3](C3.md)、[C4](C4.md)、[C5](C5.md)、
[C6](C6.md)、[C7](C7.md)、[C8](C8.md)、[C9](C9.md)、[C10](C10.md)。
