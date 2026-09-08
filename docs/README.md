---
updated: 2026-09-08
status: active
scope: GeoSparse 研究修订与审查入口
out_of_scope: 远端实时状态、原始结果、官方 OpenTAD 文档替换
---

# GeoSparse 研究文档

现行训练模型 M 为 b70ae056c495b43ca3f305fe438926b97b2723b5。本审查分支新增的研究文档不改变 M、已提交作业或它们的结果身份。

- [论文定位与本次修订](project/paper-plan.md)：承认 MoD＋TIA 强对照，明确贡献假设和无结果摘要。
- [方法定义](methods/localization-value-routing.md)：当前实现、拟新增风险与 swap、操作范围和成本含义的唯一方法说明。
- [比较计划](evaluation/mod-tia-comparison-plan.md)：M0–M3/A0–A2、公平比较、复用边界和待冻结配置。
- [机器可读设计清单](evaluation/mod-tia-proposals.json)：7 个设计标签，不是可执行 job manifest，不改变原 1545 项任务计数。
- [四张证据图规格](evaluation/localization-value-figures.md)：需要什么数据、画什么、能够支持什么结论。
- [Pro 严格复审 Prompt](project/pro-review-localization-value.md)：供后续授权讨论使用；生成 Prompt 不代表已经咨询。
- [训练协议与 I01–I15 修订](GEOSPARSE_AUDIT_REPAIR.zh.md)及[原审计复核 Prompt](PRO_REVIEW_AUDIT_REPAIR_PROMPT.zh.md)。
- [外部执行代码说明](../review/audit_repair_execution/README.zh.md)。

实时状态及真实收据仍由操作员执行包的看板、observed_metrics 和远端运行目录提供。本目录不复制随时变化的中期分数，不把计划图、预设工作点或无结果表当作实验结果。
