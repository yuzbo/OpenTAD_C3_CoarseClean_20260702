@RTK.md

# Repository Instructions

这是 OpenTAD/C3/DUCA 演化到 PhysTime-TAD 的当前研究仓库。当前主线已经从 learned frame selection 转向不规则原始视频观测下的物理时间 TAD；C3、PAction、GAS-VT、DUCA、X3D 和 SlowFast 代码保留为 baseline、诊断和可复用合同资产，不得自动恢复为论文中心。

## Mandatory Research Memory

开始任何方法设计、代码实现、实验部署或论文写作前，必须按顺序阅读：

1. `research-wiki/query_pack.md`
2. `research-wiki/current_direction.md`
3. `research-wiki/decision_register.md`
4. `research-wiki/lessons.md`
5. 与任务相关的 `research-wiki/ideas/` 和 `research-wiki/experiments/` 页面
6. 涉及历史路线时读取 `research-wiki/routes/` 对应完整档案

新的 idea、否定理由、实验状态、评审结论或方向变化必须按 `research-wiki/memory_protocol.md` 更新，不得只留在聊天记录中。

## Current Objective

当前唯一执行目标是 **`P0-FULLPRECISION-NMS-REPLAY`**：保留已经完成的 native-J192 physical-metric survivor，先修发布级评估链，再决定机制实验；当前禁止实现或训练 Q-lift：

- 从 THUMOS14 原始 RGB 视频出发；
- 训练时先使用标准 GT-aware `random_trunc` 接受窗口，再在已接受窗口内采用相同、确定性、无学习、无 GT 的 K=384 不规则子采样；val/test 全流程不读 GT；
- `DecordDecode` 和 official VideoMAE-S adapter 只处理选中的 K 帧；
- 当前 matched full60 已固定 K384/J192/QΣ378、同观测、backbone、checkpoint、schedule、seed 和 evaluator，并得到 uniform/physical `41.28/57.57%`；
- PhysTime 的 GT、query、预测、NMS 和评测使用 absolute seconds；
- 可以将秒坐标转换为原视频帧号，不得映射到 selected-rank 轴。

P0 只允许删除跨窗口 NMS 前舍入、过滤并审计无效 proposal、修正 label dtype、补对抗性测试，并重放冻结 epoch-59 uniform/physical online/EMA 权重。不得同时加入 Q384、interpolation、copy、cross-attention、gap projection、新 loss 或训练。P0 通过后顺序为冻结 decode cross-replay、Q192 UU/UP/PU/PP、无训练 Q-density replay；只有 oracle/pre-NMS 高 IoU coverage 明确受 Q 限制，才恢复训练型 Q-lift 讨论。

PhysTime-AdaTAD 1.0 已冻结为 negative baseline；native-J192 physical-metric 已达到 `full60-single-seed-supported`，不是 `paper_ready`。`SM-PTAF` 仍为 `designed`，但已暂停作为立即下一步。

## Scope

允许维护：

- `opentad/` 中 PhysTime geometry/projection/head/detector 与必要的 AdaTAD integration；
- `configs/adatad/thumos/` 中 matched PhysTime/physical-grid/selected-axis 配置；
- `tools/bata/`、`scripts/` 中 focused gate、validator、launcher 和 monitor；
- focused `tests/`；
- `docs/` 的方法合同、实施计划与唯一结果记录；
- `research-wiki/` 的长期研究记忆。

禁止把 checkpoint、数据集、服务器日志、生成图、压缩包或临时 run root 提交到仓库。

## Evidence Rules

- validation/test sampling 不得读取 GT、teacher、oracle、prediction cache 或 ledger。
- smoke、contract、one-step gradient、diagnostic full run 和 paper evidence 必须严格分级。
- 不同 commit/config 的结果不得混作 matched evidence。
- 实验数字只写 `docs/evaluation/results.md` 或正式权威 result artifact；Wiki 只记录状态与裁决。
- P0 replay 与 Q192 机制分解完成前，不扩展 learned selector、dynamic budget、paired consistency、训练型 Q-lift 或新的 frozen prior。

## Remote Rules

远端写入边界：`~/run/yuzibo` / `/data/run01/sczc063/yuzibo`。

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
```

正式训练必须使用 Slurm 或已授权计算节点，不在登录节点直接训练。远端实验必须从固定 commit 的 clean snapshot 运行。

## Verification

PhysTime 改动至少运行相关 focused tests、`py_compile`、`git diff --check` 和配置 validator。正式 full train 前必须通过一个真实 THUMOS raw-video CUDA decode/forward/backward/inference gate，并验证 same-index checksum、optimizer coverage、秒坐标和无 feature archive。
