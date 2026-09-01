# DUCA physical-cliplet dynamic-budget design

本设计的项目级冻结源为
[`research-wiki/DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FROZEN_CONTRACT-2026-08-21.md`](../../../research-wiki/DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FROZEN_CONTRACT-2026-08-21.md)。

它冻结新 clean implementation cycle 的四项可归因机制：端点覆盖语义 scout、每个 16 帧的
物理连续 VideoMAE cliplet、S0/SQ/SQC/SQD 四臂语义门，以及 fixed/dynamic/K-shuffle/
actionness-only 强控制。GAPPACK 仅以相同源帧集合解释历史 selected-rank 输入。它不授权代码、
训练或性能结论；实现必须先完成 Builder→Critic/DSH→Evaluator PRE_RUN。

