# ZoomToken 长任务事件驱动等待修正

## 结论

ZoomToken 不再用周期性 Codex/LLM heartbeat 监控长时实验。正式任务被后端接受后，Codex 记录身份与恢复条件，退出主动推理阶段；等待由计算系统或单个机器侧后台进程承担。

## 当前 v003 落地

- 唯一正式实验仍是 Slurm job `1258526`，candidate `8a59d655005b9030d8ea5dc17ee2620844cb587b`；没有提交新的 Slurm 作业，也没有修改、取消、恢复或重启该实验。
- 本机 FastCtx background job `j-bkeyzz` 每 300 秒只读取一次 `sacct` allocation 状态，运行中不输出内容、不读取日志或任何 accuracy/cost/power/boundary/prediction 数值；只有识别到 Slurm 终态才输出一行终态记录并结束。
- 原每 30 分钟触发的 Codex 自动化已改为北京时间 `2026-08-28 20:00` 的单次预计终态恢复。此前不因普通 goal continuation 查询或输出等待文字。
- 未接触、刷新、重启或关闭 iXBrowser profile 61/CDP，也未创建或重提 Pro 对话。

## 恢复边界

Codex 只在机器终态信号、用户重新进入、客观预计终态时间到达或已有异常证据时恢复。恢复后只做一次终态核验；若终态成立，摄取冻结证据并完成既定 fresh post-result Pro 复盘；若没有终态，则保存精确 blocker，不建立新的周期轮询。

这项修正只改变等待方式，不改变 v003 的科学问题、代码、协议、阈值、资源、结果解释或 Pro 冻结任务。
