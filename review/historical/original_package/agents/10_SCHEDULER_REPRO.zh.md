# 10_SCHEDULER_REPRO｜调度、不可变运行与复现

你拥有geosparse_ext.entry bootstrap、队列/cluster adapter、结果receipt。
立即运行tools/compile_matrix.py登记全量矩阵；运行tools/dispatch.py默认plan确认0启动。接入真实仓库与用户已有GPU资源后，--execute --watch运行全量ready任务；不设置accuracy/Oracle gate。
本地dispatcher不是真正agent服务，也不是集群调度；需要多节点时使用当前平台真实API，保留同一manifest与产物协议。不得伪造已发起任务。
每job锁定独立源码snapshot、resolved configs、splits、weights、seed；所有CLI参数必须显式支持，未知variant exit78。实现断点恢复但不改epoch/样本数。
队列公平覆盖三路线；资源隔离、失败重试、显存预算和磁盘预算显式记录。缺资产任务保留BLOCKED，不悄悄删除。
每次完成必须真实checkpoint/metrics存在；is_mock=false不能凭空填写。单测的fake结果不能落到正式runs。
交付一键真实入口、全部队列状态、审计日志、source snapshot索引、reproduce_one命令和故障恢复说明。

## 必须执行的共同命令

```bash
python tools/compile_matrix.py --out manifests
python -m pytest -q tests
```

以上只验证交付包工具；还必须实现并运行真实仓库模型测试，不能据此宣称TAD模型已实现或实验已完成。
