# 实验完整性审计报告

日期：2026-07-12

审计者：GPT-5.5 xhigh 独立只读 agent

项目：PhysTime-AdaTAD raw-video K384 matched comparison

## 审计时原始裁决：FAIL

远端实验和官方指标真实存在，但审计时本地结果台账与 Research Wiki 仍把已完成作业写为 `running/NA`。这是报告完整性阻断项，不是指标造假证据。

## 检查项

### A. GT 来源：PASS

官方评估通过 `opentad/evaluations/mAP.py` 读取 THUMOS annotation，并由 `opentad/cores/test_engine.py` 调用。GT 不是从模型预测生成的。

### B. 分数归一化：PASS/WARN

官方 mAP 使用原始预测分数和 AP。另一个边界 utility 使用相对最大值阈值，但不在官方 mAP 路径中，且必须继续标记为 proxy utility。

### C. 结果存在性与台账一致性：审计时 FAIL，已整改

远端最终日志和 checkpoint 存在，但本地台账滞后。`docs/evaluation/results.md` 和相关 Wiki 节点现已依据最佳 checkpoint 复算更新。

### D. 诊断执行：审计时 WARN，已整改

独立审计起初没有看到带 schema 的完整预测与 attention artifact。最终诊断随后产出 `phystime_prediction_diagnostic_v1`，真实 checkpoint attention 诊断也成功完成；两者均已登记到 `docs/evaluation/results.md`。

### E. 证据范围：WARN

当前证据只有一个数据集协议和一个 seed，不足以支撑广泛鲁棒性、泛化或 paper-ready 主张。

### F. 评估类型：REAL_GT

官方 mAP 与预测分解使用 THUMOS 数据集 GT。几何和 attention 诊断只是解释性分析，不能替代精度评估。

### G. Matched contract：PASS/WARN

Raw-video 输入、K384 采样、backbone 配置、evaluator 和 selected indices 相同；但科学比较仍改变 detector 架构和容量，因此不是纯坐标表示消融。

## 整改后完整性状态：WARN

结果台账滞后和诊断 artifact 缺失已经关闭。剩余警告来自科学范围与比较设计，而不是结果来源。

## 对主张的影响

- 单次 THUMOS matched comparison 有效。
- PhysTime-AdaTAD 1.0 优越性不成立。
- 由于架构混杂，physical-time modeling 的一般价值仍未裁决。
- 泛化、鲁棒、SOTA 与 paper-ready 主张均不成立。
