# PhysTime STOP-Q-LIFT Pro 审查吸收记录

日期：2026-07-20

## 1. 来源与冻结范围

- 外部审查原文：
  `docs/methods/reviews/2026-07-20-phystime-stop-q-lift-pro-review-raw.md`
- 原文 SHA256：
  `F08AF135EAC342960929031FE84400144F0ADA55720F9A744203CFF2943A5057`
- 原文规模：1053 行，49884 字节。
- 文档快照：
  commit `21c264b85690c05ee7fe27e054d3b84eda1fc02c`
- 可执行代码快照：
  commit `0dc5851a8feb12b97d16bdb5ea8fc60e9273d132`
- 可执行代码 tree：
  `bddc9b9386604d00d213275a47ce7997b35d3f4c`
- 当前分支：
  `codex/phystime-performance-diagnosis-20260712`

本地核验确认：当前文档 HEAD 相对 `0dc5851` 没有
`opentad/`、`configs/`、`tools/`、`tests/` 或 `scripts/` 的可执行差异。
因此外部审查读取的代码仍是当前有效实验实现。

## 2. 独立总裁决

**高度认可核心研究顺序，但不完全认可报告中的全部技术表述。**

认可 `STOP-Q-LIFT` 作为当前操作性裁决：

> 暂停把 Q384、cross-attention 或其他学习式 Q-lift 当作下一项训练主线；
> 保留已经获得完整训练支持的 physical-metric Q192，先修评估链，并在
> 固定 K384/J192/QΣ378 下分解现有物理时间干预。

这不是“Q-lift 已被科学证伪”。当前科学状态仍是 **Q 是否构成瓶颈未知**；
`Q_LIFT_NEEDED=false` 只是未获实验授权的 fail-closed 决策。

现有结果继续保持：

- uniform-rank-seconds：`41.28%` Avg-mAP，`14.86%` mAP@0.7；
- physical-seconds：`57.57%` Avg-mAP，`28.64%` mAP@0.7；
- 差值：`+16.29` Avg-mAP；
- 状态：`full60-single-seed-supported`，不是 `paper_ready`。

## 3. 已由本地代码确认的问题

### 3.1 P0：跨窗口 NMS 前提前舍入

`SingleStageDetector.post_processing` 在每个滑窗输出时把 segment 舍入到
`0.01s`、score 舍入到 `1e-4`。`apply_sliding_window_nms` 随后从这些
已舍入的 Python 数值重建 tensor，再执行跨窗口 NMS，并在 NMS 后再次
舍入。

这会改变 Soft-NMS 的 IoU、排序、投票边界和高 IoU 结果。它是两臂共享
的评估缺陷，因此不自动抹掉 `+16.29`，但绝对 mAP 和差值必须使用冻结
checkpoint 全精度重放后才可成为发布级锚点。

### 3.2 P1：proposal 有效性合同缺失

`_clamp_physical_proposals_to_domain` 只独立 clamp 左右端点，没有过滤
非有限或 `end <= start + eps` 的 proposal；生产 NMS 也没有统一过滤。
该问题是前瞻修复项，当前没有证据证明它已经实质改变 `57.57`。

### 3.3 P1：FPN tail 可能污染最后一个有效位置

`FPNIdentity.forward` 在 LayerNorm 后没有重新乘 mask。训练后的 LN bias
可使 invalid tail 非零；检测头第一层 kernel-3 卷积先读取邻域，之后才
执行 mask，因此最后一个有效位置可能读取 invalid tail。

该风险不等于原始 RGB padding 泄漏，也不破坏两臂 matched 身份，但会
阻止“整个 detector 严格 padding-isolated”的发布级主张。应补 norm 后
remask、tail 内容反事实和 invalid-gradient-zero 测试。

### 3.4 P1：当前 physical intervention 是捆绑干预

`positions_key` 同时控制：

- point center；
- local stride；
- regression range scaling；
- 严格 inside-GT center 条件；
- regression target normalization；
- inference decode 与 domain clamp。

`assignment_positions_key` 只额外控制：

- center-sampling center/stride；
- regression-range eligibility。

因此 `57.57` 不能直接归因于 Q，也还不能精确归因于单一物理时间机制。

### 3.5 其他合同缺口

本地代码同时确认：

- assignment diagnostics 只有 batch/sample 聚合，没有 per-GT
  eligible、positive 和 zero-positive；
- `ActionFormer._update_native_temporal_query_audit` 把同一个容量值写成
  `query_tensor_count` 和 `query_count`，容易混淆容量、有效位置和
  class-location score 数；
- `random_trunc` 尝试耗尽后静默使用最后一次 crop；
- 跨窗口 NMS 的 class ID 使用 `float32`，应为 `torch.long`；
- 当前所谓 `selected-axis` 实际是 `uniform-rank-seconds`，不是旧式
  selected-rank GT remap；
- 当前 artifact 没有完整登记 total/trainable/optimizer-covered numel。

## 4. 对外部报告的两项实质修正

### 4.1 四臂主效应公式的标签写反了

报告明确规定 arm 的首字母是 regression/decode 轴，次字母是 assignment
轴：

| Arm | decode/回归轴 | assignment 轴 |
| --- | --- | --- |
| UU | uniform | uniform |
| UP | uniform | physical |
| PU | physical | uniform |
| PP | physical | physical |

因此正确主效应应为：

```text
Δ_decode = 1/2 * [(PU - UU) + (PP - UP)]
Δ_assignment = 1/2 * [(UP - UU) + (PP - PU)]
Δ_interaction = PP - PU - UP + UU
```

原文把前两条的名称互换了。后续预注册、artifact 和结果表必须使用上述
修正版，不能原样复制原报告公式。

### 4.2 当前四臂不是完全纯净的抽象因子分解

严格 inside-GT 条件仍使用 decode point center。因此首因子更准确的名称
是：

> regression/decode/strict-inside-GT axis

次因子更准确的名称是：

> center-sampling/range-eligibility axis

四臂能够分解现有两个代码开关，足以作为下一项最小机制实验；但不能把
结果直接包装成“纯 assignment 与纯 decode 已完全解耦”。若四臂显示
local scale 仍是关键，再进入 center/scale 的第二阶段分解。

## 5. 需要降温的表述

1. `CODE_CORRECT=false` 过于宽泛。更准确的是：
   core physical geometry 与当前训练结果有效，但
   `PUBLICATION_PIPELINE_READY=false`。
2. `STOP-Q-LIFT` 只否定“立即训练 Q-lift”的研究顺序，不证明 Q 增密永远
   无用。
3. 轴因子化四臂是当前最强、最低成本的下一机制实验，不是已经证明的
   最终模型或唯一理论路线。
4. tail mask、零时长 proposal 和 `random_trunc` fallback 是真实代码风险，
   但尚无反事实证明它们解释了当前性能差值，不能倒推成既成性能根因。
5. 在 P0 replay 中，同时改变舍入和 proposal filtering 会混合两个评估
   修复。artifact 应分别记录：
   rounded/full-precision 切换、validity filter 切换及各自决策差异；若
   原始非有限/非正时长计数为零，才可声明 filtering 没有形成混杂。

## 6. 吸收后的执行顺序

### 当前唯一任务：P0-FULLPRECISION-NMS-REPLAY

只允许：

1. 删除模型输出与跨窗口 NMS 前的 segment/score 舍入；
2. 使用 `torch.long` class ID；
3. 在 NMS 前过滤并计数非有限、非正时长 proposal；
4. 保存 scientific full-precision artifact，展示副本才允许格式化；
5. 添加 rounding 改变 suppression/ranking 的对抗性测试；
6. 用 uniform/physical 的 epoch-59 online 与 EMA 冻结权重重放；
7. 分开报告 rounding 与 validity-filter 的影响；
8. 输出五个 IoU mAP、Avg-mAP、proposal/NMS 决策差异、边界位移、
   短动作和高 IoU 分层，以及 physical-minus-uniform 差值。

该任务不得加入 Q384、interpolation、copy、cross-attention、gap
projection、新 loss 或新训练。

### P0 通过后的顺序

1. 冻结 checkpoint 做 train-axis × decode-axis cross-replay；
2. 在同一新 commit 下训练 Q192 UU/UP/PU/PP；
3. 对最佳 Q192 checkpoint 做无训练 QΣ378→756 subcell replay；
4. 只有 oracle/pre-NMS 高 IoU recall 显示 Q-density 明确受限，才重新讨论
   训练型 Q-lift；
5. 再进入多 seed、完整训练、成本和第二数据集。

## 7. 状态更新

- `exp:phystime-g1-matched-full60`：
  保持 `full60-single-seed-supported`。
- `idea:phystime-tad-2`：
  physical metric 继续获得支持，机制与独立方法主张尚未闭环。
- `idea:sm-ptaf`：
  保持 `designed`，但 Q-lift 作为立即下一步被暂停；未被永久证伪。
- Q192 轴因子化：
  `designed`，未实现、未测试、未部署。
- P0 full-precision replay：
  `approved_next_task`，尚未实现。
- 论文状态：
  `PAPER_READY=false`，不创建新 claim。
