---
type: source
status: discussed
updated: 2026-08-31
project: DUCA
---

# Pro 完整训练与完整官方留出评测协议裁决 v001

## 会话与代码身份

- Nonce：`DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`
- Project：`g-p-6a91061f789881918ccd8357ca3d6c92`
- Conversation：`6a952a19-9294-83ea-b09f-5524e7825316`
- 模型选择器：Pro；浏览器终态记录验证了 Project、conversation、nonce 和模型选择。
- H65 模型与训练基座：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- whole-video 诊断功能来源：`33e4ed137c33eef07f0452b44506a6993bdf7535`，只允许移植已核验的变长执行、
  packet 对齐、实际 observation 计数、K384 parity、whole-video 输出组织和 proposal 原始顺序保持。
- 完整原始回答：
  `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`
- 终态 manifest：
  `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/manifest.json`

## 科学裁决

Pro 选择 `REVISE`，保留上一轮单变量问题：在 Scout、嵌套 K256/K384/K512 位置、检测器结构、损失、物理
时间映射、Soft-NMS 和评价器不变时，只改变 Stage-2 训练预算分布，检验多预算训练是否改善跨预算检测兼容性。
冻结 K384 检测器上的 Marginal-v1、cap-release、96-state 和 704-state 路线继续永久只读，不因本裁决恢复。

上一轮的 160-train/40-development 正式协议、旧 40-video holdout、有标签训练侧 mAP 门和 whole-video oracle 均被
撤销。它们只能作为历史诊断，不能进入当前正式比较。

## 完整数据协议

- 完整训练集采用本项目 H65/OpenTAD 的 `training` subset，来源为规范
  `thumos_14_anno.json`；annotation、loader 和物理视频必须严格覆盖同一 200 个视频，无排除、无静默丢弃。
- 完整官方留出评测采用同一 annotation 下的 `validation` subset 和现有 THUMOS14 evaluator。正式评测前不得读取
  held-out 动作类别、时间边界、proposal、预测或 mAP。
- 211/212 不能由 Codex 选择。必须只读比较 annotation、loader、物理文件、历史 211 prediction IDs 与可核验的
  ActionFormer 212 来源。若 annotation/loader/物理/evaluator 集合不一致，或找不到来源支持的差异解释，则阻断
  模型实现与训练并把未知事实返回 Pro。
- 两臂正式训练都使用完整 200-video 训练集，从同一 H65 Stage-1 `epoch_29/state_dict_ema` 开始，各完成 6,000
  次成功 optimizer update；update 6,000 的 terminal `state_dict_ema` 是唯一结果模型。
- 正式 held-out 评测是一次预注册事件：先密封两臂在 K256/K384/K512 及相同无标签 fixed mixed-budget manifest
  上的预测和真实成本，再统一计算指标及 10,000 次整视频配对 bootstrap。held-out 结果不得回流到当前方法开发。

## 当前唯一任务与边界

当前只执行完整 train/held-out 身份核验。Pro 指定基于 `04c35a3b...` 的分支
`feature/duca-full-data-identity-audit-v1-20260831`，只允许新增或修改：

- `tools/bata/audit_duca_thumos14_split_identity.py`
- `tests/test_audit_duca_thumos14_split_identity.py`

身份核验通过前，不建立多预算模型分支、不加载 checkpoint、不提交 GPU、不训练、不生成 held-out 预测、不计算
held-out mAP。Builder 的精确提交必须经过独立 Critic；Critic 通过后，独立 Evaluator 只在 N16R4 CPU 上运行一次
无标签身份核验。无论结论为通过或阻断，都把 literal manifests 和 211/212 事实返回 Pro，不自动进入模型实现。

## 证据与主张边界

本裁决只冻结科学问题、数据语义、训练和评测协议以及当前任务，不产生新模型代码、训练、性能或成本证据。即使
后续单种子实验通过，也只能支持当前 H65 初始化和嵌套位置构造下的跨预算兼容性机制证据；不能据此声称 learned
controller、动态预算性能—成本优势、优于 dense AdaTAD、训练稳定性或跨数据集泛化。

同轮用户还提供了一份未在仓库中找到的 `research_project_analysis.md` 摘要，建议渐进解冻、五档预算曲线和
ActivityNet-1.3。该摘要的本地文件、生成会话和证据链当前不可核验；这些建议也不属于本 Pro 裁决的唯一当前任务，
因此只保留为未准入候选意见，不改变执行路线。
