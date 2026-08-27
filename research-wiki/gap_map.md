---
type: gap_map
updated: 2026-08-28
---

# Current Evidence Gaps

| 科学问题 | 已有证据 | 仍缺什么 | 哪项决定会改变 |
|---|---|---|---|
| BPNS-R1 是否降低真实总成本？ | K64 相对 K100 减少 36% 原生空间输入；旧 job `1257281` 因数值绑定失败而无效，唯一替代 job `1258299` 正在运行 | 同一 GPU 上完整 K100/R1 counterbalanced decode-to-NMS p50/p95、吞吐率、峰值显存与 gross energy，以及完整 profile/terminal receipt | 是否保留效率主张，或只保留准确率/支持拓扑结果 |
| 准确率保持是否稳定？ | seed 42 的 K100/R1 为 `68.51/61.19/46.27` 与 `69.07/61.14/46.57` | 在成本证据成立后再判断是否值得增加独立种子；当前不应先扩种子 | 单次可行性是否可升级为稳定方法证据 |
| 连续支持为何保护高 tIoU？ | R1 优于不规则 C；R3 优于时间错位对照；R2 优于乱序与全局 Top-48 | 统一的短动作、起止边界误差和错误分解；排除仅由该 seed 或训练波动造成 | 主张“边界保护机制”还是仅报告经验结果 |
| 实际成本收益来自哪里？ | 只有原生 token 和重块 FLOPs 代理 | 分阶段耗时与全链路耗时，区分 decode/H2D、VideoMAE、Adapter、detector、postprocess/NMS | 是否确有主干节省，还是被固定开销或稀疏执行开销抵消 |
| 与最近 token pruning/caching 工作的差异是否足够？ | 当前方法是 current-only pre-backbone contiguous native support，无 cache/carry/depth skip | 在最终主张前完成针对最近图像/视频 token reduction 与动态执行工作的精确比较 | 能否形成论文级新颖性主张，以及主张应缩小到何种范围 |
| 泛化性如何？ | 仅 AdaTAD/THUMOS14 有当前证据 | 只有主机制与效率在主要实验成立后，才考虑第二检测器或数据集 | 是论文必要验证，还是当前机制应先停止 |

## 下一项决定性工作

等待唯一既有 job `1258299` 终态，并核验八个完整 pass、预测、evaluator vectors、功耗轨迹、
显存、延迟、短动作、边界诊断、`profile.json` 与 `terminal_receipt.json`。不得重复提交或解释
live/partial 数值。完整结果会返回 Pro 独立判断效率主张与下一任务；在此之前，不以多种子、
第二检测器或新增路由结构替代该问题。
