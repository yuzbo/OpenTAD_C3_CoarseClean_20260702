# DUCA H65 SingleClock Unit-1 终结器实现复核

## 状态

`implemented / independently_reviewed`。这是一份离线证据终结器与边界统计实现收据，不是性能结果。

## 冻结科学判定

- 主比较固定为 `EMA CLOCK_ON - H65 OFF EMA`。
- Avg-mAP、mAP@0.6、mAP@0.7 的 point delta 均须包含等号地不低于 `-0.20 pp`；比较使用 JSON 十进制文本构造的 `Decimal`，不在判定前四舍五入。
- 10,000 次整视频配对 bootstrap 置信区间只报告，不参与 PASS/KILL。
- H65 replay 五边界或同 checkpoint gate-zero 执行身份失败时，证据状态为 `INVALID`，decision token 为 `null`。
- canonical-uniform 第一处时序混合与 VideoMAE backbone 不能保持 bit identity 时，输出 `KILL_SINGLECLOCK_REPRESENTATION`。
- 若既有训练/验证物理窗口 ledger 不足以合法计算高 gap-CV 与高 boundary-density 统计，则显式输出 `NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP`；不得把缺失材料写成边界机制通过。
- cost、ON-vs-gate-zero、旧 RankPack/TrueTime、Stage-1 maturity 与恢复状态只作诊断。
- 合法科学 decision token 只有 `PASS_UNIT1_SINGLECLOCK_GATE` 和 `KILL_SINGLECLOCK_REPRESENTATION`；无论结果如何，`paper_claim_admissible=false`，`dynamic_k_authorized=false`。

## 实现范围

- `tools/bata/finalize_duca_h65_singleclock_terminal.py`
- `tools/bata/analyze_duca_h65_singleclock_strata.py`
- `opentad/models/detectors/actionformer.py` 中默认关闭的终态身份审计字段；训练和普通推理数值路径未改变。
- 对应 focused tests。

边界统计复用官方 evaluator 的重复 GT 去除规则；预测在官方输出物理秒坐标上按 score 降序、稳定 tie-break、同类 IoU≥0.5 做一对一匹配。未匹配 GT 的起止误差均为 1；匹配误差按 GT 时长归一化并截断到 1，先逐 GT 平均、再逐视频平均、最后视频等权汇总。训练集 q75 冻结与 validation 计算严格分离。

## 验证

- Python 编译：通过。
- 无数据 focused tests：`33 passed`。
- 第一轮独立 Critic：`UNIT1_GATE_IMPLEMENTATION_BLOCKED`，指出官方 duplicate-removal 与完整 VideoMAE 输入 identity 两项确定性缺陷。
- focused correction：两项均已修复并增加可区分测试。
- 独立 focused recheck：`UNIT1_GATE_IMPLEMENTATION_PASS`。
- 本地 PyTorch 合同测试：未完成，原因是 Windows 环境加载 `torch/lib/c10.dll` 时发生 `WinError 1114`；需在 N16R4 环境复核，不能将该环境故障解释为代码失败或科学结果。

## 未闭合证据

当前既有终态作业使用旧 identity schema，未封存完整 VideoMAE 输入哈希、H65 replay 五边界身份和 canonical-uniform bit-identity 正式收据。因此终态指标即使产生，也必须先补齐硬身份收据；边界窗口 ledger 若不存在则按冻结合同降级为不可评估，不允许重跑推理只为补边界诊断。
