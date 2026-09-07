# 当前外围执行代码

`official_job.py`、`deploy_corrected.py`、`refresh_corrected_status.py`是整理时外部执行目录的逐字节副本。它们不是另一个模型实现，但真实影响原版复现、focus任务字段和资源调度，故纳入审查。

`bindings.*.public.json`保留实际运行参数，机器绝对路径替换为 `<MACHINE_PATH>/文件名`。它们不是生产bindings。部署/监控还依赖工作区的SSH传输helper `transfer_assets.command`；该helper和SSH连接资料未放入公共仓库，它只提供已经授权的两台服务器连接。不得在审查中直接执行部署器，或把缺少私有绑定说成模型计算错误。

当前运行模型仍是根目录902fa05源码。外围脚本没有独立Git版本，故以审查分支此次提交保存其精确副本；这个限制应在复现审查中披露。当前完成的官方公开权重复测保存了运行当时的脚本，另见 `../evidence/official_released_checkpoint/launcher_executed.py`，不要用当前脚本替换历史执行证据。
