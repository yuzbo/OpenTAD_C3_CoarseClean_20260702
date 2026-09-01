# DUCA Uni-Companion

## 状态

`implemented_local_static`，尚未通过远端 PyTorch/CUDA 门禁，尚无 mAP。

## 为什么提出

当前 Protected-E2E 让学习选帧从训练第一步起就决定官方 AdaTAD
看到的全部输入。若早期 selector 仍不稳定，detector 会同时适应一组不断变化且
质量较低的帧，检测梯度又可能压过动作二分类和边界监督。Uni-AdaFocus
官方实现使用学习分支与随机分支增加训练输入多样性，并为 policy 设置独立
学习强度；但它的硬时序索引是 detach 的，不能直接满足 DUCA 的
detector-to-selector 梯度要求。

## 当前有界实现

在独立分支 `codex/duca-uni-companion-20260721` 中保留相同的：

- 离线完整窗口、`T=768`、固定 `K=384`；
- 官方 ASFormer 二分类动作性与 hidden features；
- 状态转变/边界优先 scorer；
- 同一物理 exact-K DAG、硬 Viterbi 前向与软 Gibbs 反向；
- 官方 VideoMAE-S AdaTAD 与 ActionFormerHead；
- 推理期只使用学习 selector。

只新增两个可归因变化：

1. `protected_e2e_bridge025`：保持硬前向不变，把 detector-to-selector
   的软反向梯度缩放为 `0.25`。
2. `protected_e2e_uni_companion`：在 batch size 2 的训练批次内随机选择
   一行使用 exact-uniform hard path，另一行使用 learned hard/soft path；
   只做一次 detector forward，学习行接收 `0.25` 检测梯度，均匀行稳定
   detector 输入。推理期不使用 companion，不增加推理成本。

匹配实验的三个学习版本是：

- `protected_e2e`：直接梯度桥，scale `1.0`；
- `protected_e2e_bridge025`：缩放梯度桥，scale `0.25`；
- `protected_e2e_uni_companion`：缩放梯度桥加训练期均匀伴随。

`exact_uniform` 仍是必须的同协议控制。超过约 `65` Avg-mAP 仅是 GO
阈值，不是当前事实。

## 不能误写

- 不能声称 Uni-AdaFocus 的 hard temporal sampling 接收了 heavy branch
  的直接梯度；官方代码对 hard temporal indices 使用 detach。
- 不能把训练期均匀伴随写成推理 ensemble 或第二次 detector forward。
- 不能在远端门禁与 terminal official mAP 前称其有效或最终方法。
