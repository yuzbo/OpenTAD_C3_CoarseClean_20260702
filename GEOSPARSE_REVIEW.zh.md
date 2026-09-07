# GeoSparse-TAD 完整源码审查入口

这份快照汇集当前负责的 GeoSparse 实验源码、撤销版本源码、原始实验方案、修订矩阵和真实执行证据，用于独立审查。**审查 prompt 已准备，不代表 Pro 已完成审查。**

**最新用户修订**：[主方法优先，部署后推进补充实验](review/amendments/20260908-primary-priority/README.zh.md)。六项seed0主实验保持首要地位；部署后已登记五项较低优先级补充配置，两端协调器并行。后续队列使用主作业开始依赖，不等训练完成或mAP。模型代码仍为902fa05；早先“B-full与消融全部延期”的阶段记录保留为历史。

## 从这里开始

1. [可直接复制的 Pro 严格审查 prompt](review/PRO_REVIEW_PROMPT.zh.md)
2. [当前有效协议与版本优先级](review/PROTOCOL.zh.md)
3. [源码地图](review/CODE_MAP.zh.md)与[需要优先复核的问题](review/KNOWN_ISSUES.zh.md)
4. [实际运行状态](review/STATUS.zh.md)与[仓库/提交清单](review/REPOSITORIES.json)
5. [全部配置的静态实现登记](review/IMPLEMENTATION_CATALOG.zh.md)（[JSON](review/IMPLEMENTATION_CATALOG.json)）
6. [修订完整矩阵](review/manifests/experiments.corrected.full.jsonl)与[当前聚焦矩阵](review/manifests/experiments.corrected.focus.jsonl)

## 代码范围

|目录|来源和用途|
|---|---|
|根目录 `geosparse_ext/`、`tests/test_geosparse*`|两台服务器当前模型版本 `902fa05b5c64452ff1c94b82cabce801943d3484` 的完整实现与测试|
|根目录 `opentad/`、官方 configs、`tools/train.py`、`tools/test.py`|官方基座 `346d09d19e2091372cec48172dbe40f7b28bdee6`，与该版本的对应代码完全一致|
|`review/launchers/`|官方原版执行/完整验证/best记录器、当前双服务器部署器和监控器；资产绑定的机器路径已脱去，不能作为生产配置运行|
|`review/historical/withdrawn_340a3541/`|旧 GeoSparse 已撤销协议的模型、基础配置、入口和 focused tests，用于追溯；不能用于当前正式实验|
|`review/historical/original_package/`|原始方案、12 份 agent 指令、1538 项原始矩阵、参考实现和调度工具；是历史设计，不是实现完成证明|
|`review/evidence/`|发布权重复测指标、脱去机器路径的当前运行凭证、真实预检与解析协议；没有训练数据或权重|

当前纯官方基座上的扩展是唯一生效实现。原 C3 工作区存在其他工作且有未提交改动，未混入本次 GeoSparse 快照，边界详见版本清单。审查分支只增加材料和入口说明，不把这些材料伪装成服务器正在训练的模型 commit。

## 结果边界

官方发布权重的完整 211 视频 / 792 窗口复测实测 Avg-mAP **71.1387948970%**，两位小数为 **71.14%**；这是推理复测，独立 seed0 训练尚未完成。A/B/C 不能借用这个数值作自身成绩。训练状态见带时间戳的状态快照，不保证此静态页面实时更新。

旧 180-video 训练结果已撤销。新矩阵有1545项注册任务，大量变体仍缺实现或资产。当前部署六项seed0主实验和五项补充配置，另有官方原版seed0训练和已完成的发布权重复测；部署、预检、Slurm排队均不等于正式训练完成。实时进度需要查询两服务器，不能从此静态快照推断。

## 阅读或复算

静态清单可用 `python review/build_catalog.py` 从固定源码和附带 manifest 重新生成；该命令不训练、不访问 GPU、不判断资产可用性。完整训练环境和资产路径仍需真实绑定，仓库内的示例路径不是可直接运行的生产配置。源码审查不需要下载数据或权重。

如无法直接读取 GitHub，可将本分支的源码 ZIP 与 prompt 一并交给 Pro。审查时固定实际 checkout commit，并引用 `file:line`；不得只读摘要后报告已审完全部实现。
