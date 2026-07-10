---
type: gap_map
updated: 2026-07-11
---

# Gap Map

| ID | 缺口 | 当前状态 | 关闭条件 |
|---|---|---|---|
| G1 | 同提交强基线缺失 | open | dense/exact-uniform/periodic/random/actionness 与 DUCA matched matrix |
| G2 | ST surrogate 与 hard utility 未对齐 | open | one-swap finite-difference 正相关且优于简单信号 |
| G3 | selected-axis 不等物理间隔 | open | same-selected-frames 几何对照；或新 head 明确改善 |
| G4 | 真实总成本未知 | partial | trained checkpoint 下完整 p50/p95/memory/energy 矩阵 |
| G5 | coarse probe 与 selector 错误难分 | open | probe actionness/transition 质量与 selector 条件性能分解 |
| G6 | requested/effective/unique K 不一致 | open | 每样本严格 ledger 与 backbone 实际消费核验 |
| G7 | dynamic MUST 不稳定 | open | 多 target、matched mean-K、真实 K/latency 与稳定 mAP |
| G8 | 高 tIoU 与短动作保护 | open | mAP@0.6/0.7、边界距离、短动作分层不退化 |
| G9 | 插件泛化未证明 | open | 第二 detector；最好第二数据集 |
| G10 | no-leak/provenance | partial | 递归 fail-closed + artifact hashes + clean manifests |
| G11 | 官方 AdaTAD 语义边界 | partial | 论文和 validator 使用 official-derived 诚实口径 |
| G12 | 论文创新性与停止条件 | open | 结果达到 claim threshold；否则降级/转向 |

## 优先级

P0：G1、G2、G4、G6、G8。
P1：G3、G5、G9、G10。
P2：G7、G11、G12。
