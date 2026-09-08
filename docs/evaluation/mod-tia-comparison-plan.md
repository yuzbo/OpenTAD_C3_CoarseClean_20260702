---
updated: 2026-09-08
status: proposed-not-dispatchable
scope: 强 MoD＋TIA 比较、因子控制、现有实验复用
out_of_scope: 已冻结配置清单、Slurm 提交、训练结果
---

# 强 MoD＋TIA 与定位价值路由比较计划

本计划采用用户提出的 7 个设计标签。它们不是 7 个已经实现或提交的任务；固定/动态、层布局及缩放控制展开后也不止 7 个训练。机器可读版本见 [mod-tia-proposals.json](mod-tia-proposals.json)，现行 1545 项 manifest 不因本文件改变。

## 设计标签及当前状态

|标签|路由|任务目标与校准|现状|
|---|---|---|---|
|M0|原风格逐层 hidden-state MoD＋TIA|标准 TAD loss；保留路由分数缩放梯度|待实现；包括全层路由与交替 dense/routed 布局。|
|M1|较强上下文 Router＋同一 TIA|标准 TAD loss|设计待冻结；输入、容量和开销必须明确。|
|M2|与 M1 相同|共享 interval risk，无反事实价值校准|新风险和对应训练待实现。|
|M3|与 M1 相同|共享 interval risk＋成本一致的价值校准|逐层操作、下游重路由规则及标签路径待实现。|
|A0|现行前置 Scout、一次计划|标准 TAD loss＋PG，无 acquisition 校准|PG-only 分支存在；须核对完整配置是否已有任务，不能重复创建等价训练。|
|A1|现行前置 Scout、一次计划|标准 TAD loss＋PG＋signed acquisition|现行 A 主方法属于这一机制，原身份复用。不是 A2。|
|A2|前置 Scout、一次计划|共享 interval risk＋固定预算 swap / 动态预算 acquisition 与 ΔC|新风险、donor 条件和校准均待实现。|

这里“完整主方法”仍指各次协议中约定的完整方法；不能因提出 A2 就把正在运行的 M 主方法称为遗漏了原来必需模块。A1 动态与固定 .5 的现行任务分别为 tr-0afe8e4fc09b、tr-624db235c913；结果引用原任务 ID、原目标和原成本。将其用于新增严格控制时先核对配置，不能宣称已匹配未运行 MoD 的实际 MAC 或延迟。

## 公共协议

沿用[已授权训练协议](../GEOSPARSE_AUDIT_REPAIR.zh.md)：THUMOS 全部 200 训练视频、211 测试视频/792 窗口；768 输入位置、160px、VideoMAE-B 同一识别预训练、原生 tubelet/PE/parent、全窗口 TIA、同一检测器；global batch2、warm-up5、cosine100、训练60轮、seed0。测试 GT 不进入梯度、收益标签或超参数拟合。

Geo 保留每5轮全测试 EMA best；另外报告固定60轮 EMA、共同50/60轮 best。官方原版按自身节奏且独立命名；统一 dense control 才是 Geo 训练器的直接控制。发布权重复测不冒充从识别权重训练的结果，单种子不报告种子标准差。

## 必须控制的因素

- **时间建模**：每个对照都有相同 TIA。Heavy 不跨 parent；TIA 不按 selected rank 或 parent 重置。保留 mask、尾部处理和原生位置。
- **预算**：同时记录每层每 parent 的有效/执行/填充 token、绝对 Heavy MAC 和全路径延迟。同 token 比例只是一个控制；跨分布比较按实际成本展开，不能标称 .5 就判成本一致。
- **MoD 布局**：保留原风格 routed/dense 交替及全层路由配置；算入 dense 层不可省的成本。在可达到的重叠成本区间比较，不外推不可达工作点。
- **学习信号**：保留原 MoD 风格的缩放/梯度作为迁移基线；它是 MoD-style TAD adaptation，不是原语言模型论文完整复现。受控组统一选择原子、执行器、缩放与训练估计器。若 D=I 且硬 Top-K 不可微，须提供两边一致的 PG 等梯度路径；不能去掉唯一梯度路径再比较“路由能力”。
- **目标**：M2/M3/A2 共享同一风险和权重。新增风险若只给 A，不能归因于 Scout。A0→A1 测 acquisition；M1→M2 测目标；M2→M3 测校准。A1→A2 同时改变目标和标签，单独这一对不能分离两者。
- **信息与容量**：说明 M1 的上下文来自哪里、宽度、参数和额外 FLOPs。原风格单 token MoD 和统一原生原子的受控 MoD 分列；选择粒度也不能藏进 Router 差异。
- **训练成本**：报告额外 probe forward、GPU小时与更新次数。相同60轮回答相同数据预算下的问题，不自动等于相同训练算力；不偷偷给任一方增加训练轮数。

机制定义和未冻结项以[方法规格](../methods/localization-value-routing.md)为准。M3 是否与 A2 同样有效，是可迁移性问题，不要求 M3 失败。

## 实现前还须写入配置的具体内容

1. M0 层布局和各层/parent 容量；M1 的上下文 Router 结构。
2. 原风格及受控组的原子粒度、residual scaling、梯度估计器及推理选择规则。
3. interval risk 的实例分配/归约、β/γ、边界尺度、空 GT 与漏检处理。
4. swap donor/candidate/集合条件、动作层范围、有效 token 成本匹配与 MoD 下游计划处理。
5. 预算点、匹配规则和 probe 数量；成本匹配只依据可审计成本，不按测试 mAP 挑点。
6. 新旧 task/config 的等价核对与复用映射；变化才创建新的协议任务身份。

这些是规格缺项，不是等待成绩。当前设计清单不生成 job ID 或伪造资源/产物依赖。冻结后接入已有 compiler/entry，做对应正确性检查，再并行入队；不另建与真实调度无关的假执行入口。

## 执行优先级与复用

正在推进的 A/B/C 动态和固定 .5 继续优先，两个服务器均继续执行已部署任务。已提交的 B-full、dense 和 uniform 保留。新增强 MoD 比较属于必要对照；实施和登记不等待 A 的最终 mAP，也不停止当前主方法。

先复用可解释的 A1 产物，并落实强 M0＋TIA 与 A2 的具体实现；M1–M3/A0 依据实现就绪和资源排队。完整依赖只能是资产、实现正确性、实际资源和本任务产物，不能使用“MoD 输了才继续”等门槛。当前没有新增 MoD Slurm 提交。

同 GPU、同测量边界、同 batch 和相同窗口集合形成 latency 比较。4090 与 A100 分面报告；准确率、选择/cost 与 latency 必须绑定同一个实际 checkpoint。修补测量优先复用已保存权重，不能用重新训练替代导出或计时。
