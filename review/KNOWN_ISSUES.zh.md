# 已知审查线索与证据边界

本文件是主代理整理源码时发现的线索，不是 Pro 的最终判决。审查者必须独立复核并继续检查未列出的问题。没有为了让审查通过而修改被审模型902fa05。

## 已由静态代码证实的跨模块不一致

`geosparse_ext/records.py:44–53` 的新 `prepare_split` 只生成 `training`、`internal_diagnostic`（全部训练数据的诊断别名）和 `validation`。但 `geosparse_ext/evaluation.py:47–57` 的独立评估仍先导出 `internal_dev`，`prediction_export.py:24` 接受该旧名，`:55–60` 直接访问 `split[args.subset]` 并设置dataset subset。

因此在902fa05的完整新split上，独立评估的这条调用链会在 `internal_dev` 查找处不匹配。尚未通过一个真实60epoch已完成run执行到该步骤；这是明确的静态接口冲突，不伪称已测得运行时堆栈。训练中每5epochs全量验证走 `training_validation.py`，不能从它可运行推断独立评估也正确。修复不能恢复20视频holdout；风险阈值拟合用什么数据、导出哪个subset必须符合最新协议并明确偏差。请检查所有split消费者，不能只改这一个字符串。

## 协议字段与执行语义需要逐项核查

- `matrix.py:81` 仍写 `checkpoints=[5,20,40,59]`，`:247` 的evaluate写 `checkpoint_epoch=59`；focus manifest同一训练ID则写4/9/…/59及best验证协议。实际runner采用每5epochs与best。请判定哪些字段真正驱动执行、哪些已成为误导元数据，以及“同ID不同任务语义”的复现风险。
- 外围 `launchers/deploy_corrected.py:101` 把 `compile_all()` 的第二返回值命名为summary并在`:115`写入 `matrix.corrected.summary.json`，但 `matrix.py:263` 返回的是按实验族组织的ID引用refs，不是完整计数summary。附带文件保留真实内容；准确计数来自manifest与本包的build_catalog，不能依赖这个文件名推断统计含义。
- 官方原版40/2验证与GeoSparse每5epochs验证的选best机会不同。用户已明确要求best；请提出基于现有检查点的可比报告，而不是静默换回旧方案。
- `protocol.implementation_blockers` 静态允许不等于所有参数在代码中生效。ROI、第二轮、若干估计器等被明确拒绝，但对已接受字段仍需逐个追踪。
- 当前seed0阶段与 `figures.py` 中三种子聚合要求、最终导出必须完整60epochs的要求，可能影响“尽早观察选择和预算分布”。请判定具体受影响功能，不把所有图一概称为已就绪或全坏。
- 当前160px改变了原生grid，空间7×7等已登记组合是否可执行应检查真实分组逻辑与拒绝规则。

## 已修复事项，不应重复当成当前错误

902fa05修复了已实际发生的多进程共享临时JSON写入竞争、按配置独立GPU能力凭证，以及输出专用额外对照保留整张梯度图导致的预检OOM。真实原版/扩展输出及梯度对照仍覆盖768帧/48parents/384-TIA。以前失败attempt仍在外部运行档案保留。

当前A与C的实际完整形状source-limit输出/梯度最大绝对差均记录为0；这只证明该测试设定，不能替代整套训练、随机性、优化器、EMA、各种预算/尾部的等价性证明。B没有声称架构与原版等价。

## 不应误读的结果

官方发布权重完整复测71.1387948970%是实测。独立seed0训练和A/B/C最终成绩尚无完成凭证。20项原始工具级测试、迁移后的focused tests以及GPU precheck均不是TAD训练结果；不得相加重复测试数来制造更高覆盖。

审查包没有包含所有服务器训练日志和weights。小型状态凭证已去除机器路径；要重算正式mAP，原始211视频预测保存在本地执行包，可按需提供，不应从摘要反推逐实例指标。
