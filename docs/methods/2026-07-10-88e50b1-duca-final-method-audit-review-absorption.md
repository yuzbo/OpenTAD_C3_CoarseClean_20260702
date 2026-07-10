---
updated: 2026-07-10
status: active
scope: 吸收 88e50b1/7e3a508 DUCA 最终方法严厉审查，并约束后续代码修复顺序
out-of-scope: 作为实验数字的唯一来源；替代原始 Pro 回复；直接宣称 CFPA 已实现
---

# DUCA Final Method Audit Absorption

原始回复完整保存在：

`docs/methods/reviews/2026-07-10-88e50b1-duca-final-method-audit-review-raw.txt`

原始附件 SHA256：

`00107AC0E451A60DD66BCC623E04C4DF879554F24AFA53B99160FCE42E516966`

审查对象：

- 主实验实现：`7e3a5081f58958fc924accf43088b24e2bf3093a`
- 当前分支最新提交：`88e50b17241cf4ef7d891e87b5455f45d72e345d`
- `88e50b1` 的 SlowFast Fast-side 仅作为 frozen-prior 诊断，不替代 DUCA 主方法。

## 本轮必须吸收的裁决

1. 当前 `ActionFormer.forward_train()` 会把 selector 叶子损失、聚合后的 `total_loss` 和重复 utility alias 一起求和。修复必须建立单一聚合点，不能用简单除权重掩盖重复。
2. 当前 action-body `1.0`、boundary `1.5` 的 proxy 不满足按实例边界优先。动作性校准、start、end、上下文和 detector utility 必须拆分。
3. 现有 soft-resample backward 与 center-radius/top-k/max-gap-repair hard policy 不同构。非零梯度 proof 只证明可达，不能证明 hard utility 改善。
4. 原审查把 `online` 理解成流式因果采集；项目目标实际是同一次 forward 内即时生成、无缓存的全窗口选择，因此 prefix-decision invariance 不是主方法要求。
5. MUST 的 expected soft K、hard K、unique K、padded K 和实际 backbone K 尚未统一，暂不作为论文主贡献。
6. `official_asformer` 当前表示官方代码结构来源，不表示加载了官方预训练权重。
7. 使用 THUMOS action target 的联合训练必须如实记录 `uses_labels=True` / `thumos_trained=True`；不能写成 no-label 或 train-free。

## 修复顺序

P0 正确性：

- 单一叶子损失聚合与梯度 multiplicity 测试。
- synthetic coordinate round-trip 与 half-open boundary cases。
- provenance 真实语义。

P1 当前模型可审计性：

- 按实例归一化的 start/end target 与双端点覆盖监督。
- radius/max-gap/soft bridge 行为审计。
- unselected-frame sentinel non-interference。
- detector/aux 分模块梯度范数与余弦记录。

P2 架构裁决：

- 用 hard one-swap finite difference 检验当前 surrogate。
- 若当前 center-radius policy 在正确性修复后仍达不到既定门槛，则停止继续调权，转向全窗口结构化 exact-budget 选择算子。
- CFPA 只适用于未来流式版本，不接入当前论文主方法。

## 论文表述边界

在上述验证完成前，只能称当前系统为接入 official AdaTAD 后端的 hard selected-frame 联合训练原型。不得声称 detector-utility-calibrated、boundary-first、streaming/strict online、train/inference consistent 或 dynamic compute。

## 2026-07-10 范围纠偏：即时生成不等于流式在线

项目负责人明确：此前所说的 online probe/actionness，是指粗分类信号在同一个模型 forward 中即时生成，不依赖离线 JSONL、ledger、预测缓存或独立 selector checkpoint；它不表示只能观察历史前缀，也不要求逐时刻不可撤销的 select/skip。

最终论文方法应定义为 **cache-free、full-window、jointly-trained pre-backbone frame-selection plugin**。低成本粗分类器和 selector 可以观察整个输入窗口，再决定昂贵 VideoMAE/AdaTAD backbone 实际消费哪些帧。代码与论文应优先使用 `in_forward`、`runtime_generated`、`cache_free`、`full_window`、`jointly_trained`，不得把它包装为 `streaming`、`causal acquisition` 或 `strict online`。

因此：损失重复、端点监督、head 直接监督、坐标闭环、成本账本、provenance 和 hard/soft 梯度一致性问题仍然成立；全窗口 ASFormer 特征与全局结构化选帧是允许且符合目标的；CFPA 不再是当前最终方法要求。
