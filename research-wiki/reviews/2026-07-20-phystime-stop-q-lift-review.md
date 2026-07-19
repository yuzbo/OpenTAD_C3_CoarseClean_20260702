# 2026-07-20 PhysTime STOP-Q-LIFT Pro 审查

## 来源

- 原文：
  `docs/methods/reviews/2026-07-20-phystime-stop-q-lift-pro-review-raw.md`
- SHA256：
  `F08AF135EAC342960929031FE84400144F0ADA55720F9A744203CFF2943A5057`
- 完整吸收：
  `docs/methods/2026-07-20-phystime-stop-q-lift-pro-review-absorption.md`
- 审查代码锚点：
  commit `0dc5851`，tree `bddc9b9`。

## 独立裁决

**高度认可核心顺序，不完全认可全部技术表述。**

认可：

- 立即停止把训练型 Q-lift 当作下一主任务；
- 保留 physical-metric Q192 和现有 `57.57%` 单种子完整训练证据；
- 先修跨窗口 NMS 的提前舍入，并用冻结 online/EMA 权重重放；
- 先在 Q192 下分解现有物理时间干预，再判断 Q 是否构成瓶颈；
- FPN tail remask、proposal 有效性、per-GT assignment、K/J/Q 命名和参数
  artifact 都需要补齐；
- 当前不是 `paper_ready`。

修正：

- `Q_LIFT_NEEDED=false` 是 fail-closed 授权状态，不是科学证伪；
- `CODE_CORRECT=false` 应收窄为发布级 pipeline 未闭环；
- 原报告把 decode 与 assignment 两条主效应公式的标签写反；
- 当前 strict inside-GT 仍使用 decode center，因此四臂只分解两个代码
  开关，不是完全纯净的抽象因子分解；
- P0 replay 应分别记录舍入与 proposal validity filter 的影响，避免把两项
  修复混成一个因果结果。

## 正确的四臂公式

首字母为 decode/回归轴，次字母为 assignment 轴：

```text
Δ_decode = 1/2 * [(PU - UU) + (PP - UP)]
Δ_assignment = 1/2 * [(UP - UU) + (PP - PU)]
Δ_interaction = PP - PU - UP + UU
```

## 当前唯一下一步

执行 `P0-FULLPRECISION-NMS-REPLAY`，禁止并行实现或训练 Q-lift。P0 通过
后依次做冻结 decode cross-replay、Q192 UU/UP/PU/PP、无训练 Q-density
replay；只有后者证明 oracle/pre-NMS 高 IoU coverage 受 Q 限制，才恢复
训练型 Q-lift 讨论。

## 状态

- Full60：`full60-single-seed-supported`。
- Q192 轴因子化：`designed`。
- SM-PTAF/Q-lift：仍为 `designed`，但暂停作为立即下一步。
- P0 replay：`approved_next_task`，尚未实现。
- Paper：`not_paper_ready`。
