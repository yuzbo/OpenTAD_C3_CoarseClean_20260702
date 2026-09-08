---
updated: 2026-09-08
status: measured-cost-audit
scope: 冻结 M 的动态预算目标、实际执行与结果解释
out_of_scope: 更换训练实现、宣称新控制器已经完成、原始视频与逐窗结果入库
---

# 动态预算目标与实际执行

M 固定为 `b70ae056c495b43ca3f305fe438926b97b2723b5`。用户要求核查动态方法的较高精度是否来自接近全量的计算，并指出这种结果不能充当预定预算下的效率收益。本次核查使用真实 EMA 检查点与完整官方测试管线；源码与原训练身份保持不变。

## 已经存在的控制

`geosparse_ext/routing.py::policy_losses` 使用

\[
R=L_{\rm TAD}+\lambda(c-0.5),\qquad
L_{\rm actor}=\operatorname{stopgrad}(R-b)\log\pi(S,K).
\]

critic 拟合该成本，acquisition 另外监督当前检测损失的有符号下降。实际成本 `c` 从 Heavy trace 计算，并除以同分辨率、完整 padded 窗口的稠密 Heavy MAC；它不是总模型 FLOPs 或真实延迟。

`geosparse_ext/detector.py::after_optimizer_step` 在成功更新后执行：

\[
\bar c\leftarrow0.9\bar c+0.1c,\qquad
\lambda\leftarrow\operatorname{clip}\big(\lambda+0.001(\bar c-0.5),0,10\big).
\]

因此不能说预算损失未实现、成本没有进入学习，或仅因 `detach` 就声称 PG 没有梯度。这个目标是训练随机策略上的软平均约束，并不保证一个未收敛或转到不同输入分布上的确定性策略满足预算。

## 训练与验证之间的缺口

`routing.py::make_plan` 在训练时采样 categorical 预算；验证时直接 `logits.argmax()`。验证路径没有对最终实际成本执行可行性约束。最高概率的档位可以在每个输入中相同，即使其概率只有约四分之一至三分之一。概率均值、训练 `cost_ema`、验证实际预算因此是不同的量。

不能仅把 `budget=.5` 标签显示在较高 mAP 旁边，就把该点解释为50%推理计算。也不能只把 argmax 改成抽样后即宣布修复：抽样本身仍不保证约定成本，且要考虑 EMA、训练/测试有效长度分布及执行策略的差异。

原训练 trace 仅写每轮首批和发生 acquisition probe 的批次；全选时可能没有可追加候选而不写该类 probe 记录。它不是全训练预算分布，更不能替代对应 best EMA 的全量验证分布。

## 本次测量

执行代码审查副本位于 `review/audit_repair_execution/`：

- `replay_validation_budgets.py` 读取指定不可变 epoch 的 EMA Scout，用原 `video_batch`、原 Scout 与 `make_plan` 执行完整211视频/792窗口。Scout 在 Heavy 前决定计划，因此预算核查无需再执行 Heavy 或训练。
- A/B 按各 parent 被选 native token 计数；C 按有效 native数/4加3倍fine成员数/4计 mixed token。按冻结执行规则计算12层 QKV/attention/MLP MAC，并分别报告 padded-full 与同一有效输入的 full 参考。
- `collect_budget_replay.py` 将成本与同一训练ID、epoch、EMA、源码/配置、split的完整验证收据配对；best仍对应该epoch时额外比对已有 checkpoint 身份。未完成验证记 NA。
- `build_dynamic_budget_report.py` 保留实际预算直方图、全选比例、训练成本状态、概率均值与实际 MAC。完整导出检查792窗口、211视频和各方法的窗口/有效长度一致；不将部分结果写成全量完成。
- 只读复放在已有合法 allocation 内运行，使用有限 CUDA 内存与独立进程。解码从1个worker增至2个worker时只续接已保存窗口；原脚本、停止/续跑收据和前缀计数留在外部目录。没有取消 parent 训练、改变模型或重跑完成窗口。

原始逐窗 JSONL、检查点 SHA、验证收据、PNG/SVG/PDF 和执行记录保留在操作员执行包的 `official_adatad_audit/dynamic_budget_analysis_20260908/`，不放入模型仓库。是否已全量完成，以其 `summary.json` 的状态与 receipt 为准。A核查完成30/40轮，B/C核查完成10轮；这些是指定检查点，不自动代表随后更新的 best。

## 允许的结论和修订边界

当一个检查点全部选择 q=1 且有效 Heavy 比例=1，其精度可作为该实际成本的结果保留；不能用于“50%推理成本的动态收益”“内容自适应预算有效”或“无损稀疏加速”结论。省去无效 padding 不等于学会省去冗余内容。B若固定取较小档位，仍不能单凭这一行为证明内容自适应预算有效；其位置选择可以继续是内容相关的。

动态与固定方法的比较还须匹配完成轮数和实际成本。C的 q=.5 表示 mixed-token 数量比例，并非一半 fine 或一半 Heavy MAC。mAP差值不能全部归因于计算差异，因为两种训练得到不同权重。

最快的后续定位是复用已有动态 EMA，在明确区分身份的推理诊断中执行固定 q=.5，并同时记录实际 MAC，比较同一权重增加验证计算的作用。这项后续 Heavy 评价尚未执行，不能写成已有结果，也不需要再训练60轮。

方法修订须把预算作用范围明确下来，让训练/推理使用的分配规则真正满足既定成本。逐窗口上限与每视频平均预算不是同一方法；不能为方便实现而静默删除高预算档位，或按测试 mAP 调惩罚系数。新控制器尚未部署；保留当前 M 训练及检查点，不把旧结果改名为修订后的方法。预算审查不增加其他路线的科研结果准入门槛。
