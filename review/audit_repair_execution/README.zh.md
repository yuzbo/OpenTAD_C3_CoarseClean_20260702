# 审计修订版的执行代码复审

实际模型/训练源码 M 固定为 `b70ae056c495b43ca3f305fe438926b97b2723b5`。本审查分支在 M 上添加执行审查与研究文档；`geosparse_ext/`、`tests/`、`opentad/`、`configs/` 与 M 相同。不要把仅添加审查材料的提交误当作新的模型实验版本，或要求已正确启动的 M 训练重跑。

先读 `docs/PRO_REVIEW_AUDIT_REPAIR_PROMPT.zh.md` 和 `docs/GEOSPARSE_AUDIT_REPAIR.zh.md`。这里的脚本记录冻结模型仓库以外的执行行为：

- `audit_queue.py`：主方法与补充控制的真实Slurm调度。只有资源、实现、正确性和自身产物依赖；主方法没有mAP门槛。补充控制的开始依赖随主方法续训产生的新Slurm ID更新。
- `full_method_witness.py`：动态预算、预算头与acquisition梯度、dual/cost EMA更新的实际batch正确性检查，模型在检查后丢弃，不生成科学结果。
- `activate_dependency_refresh.py`：只更新补充coordinator；按实际登录host、进程owner和命令核验并重启，不停止训练进程。
- `held_controls.py`：按用户要求先用真实 `sbatch --hold` 提交已部署的五项seed0补充训练，记为 `HELD_SUBMITTED`；只有本任务真实GPU能力凭证、资源与主方法开始依赖满足时才更新节点/依赖并放行。N16补充节点列表为空时保持挂起。既有checkpoint、attempt和任务ID保留，正式产物不使用mock能力凭证。
- `activate_submitted_controls.py`：当前补充coordinator的v3激活入口，只上传外部调度脚本并重启两个补充coordinator。主coordinator和Slurm训练进程不改动；不要再用历史v1/v2激活入口覆盖v3。
- `test_held_controls.py`：八项外部调度检查，覆盖一次提交、真实能力准入、并发/节点保留、放行失败、Slurm对账、模糊提交响应与磁盘不足。本地和两台Linux服务器均通过；mock只在独立单测目录，不生成科学结果。
- `consolidate_control_prechecks.py`：一次性处理A100现场 `MaxSubmitJobs=10` 的提交上限。核对四项原预检均未开始及其owner/脚本/命令后，用一个两小时上限的Slurm作业顺序执行四个完全相同的独立GPU预检命令，逐项保留日志与能力凭证；一项失败仍继续其余检查。原pending提交记录保留，正式训练仍须自己的凭证通过。只停止/恢复A100补充coordinator，不触碰主方法或官方训练；已有amendment时拒绝重复操作。
- `reserve_primary_gpu1.py`：针对实际发生的N16主方法续训资源冲突，暂停补充队列对两个GPU1节点的使用。在1278034的第1轮全状态保存后缩短其本次allocation，之后恢复仍使用同一任务checkpoint；它不是改变60轮训练预算的工具。
- `collect_audit_validation_receipts.py`：读取指定主方法已完成的全量验证，核对源版本、配置/数据/初始化权重、211视频/792窗口和best身份后同步小型收据。原始预测保留在远端，不运行推理或训练；规范视频列表来自此前已核验的动态A第10轮收据。本次已用于动态A第25轮、固定A第15轮及固定C第5轮的真实产物，模型M不变。
- `refresh_audit_status.py` 与 `export_audit_disposition.py`：更新看板与204配置/1545任务处置表。只读SSH查询设120秒进程超时；某侧失败时保留并标注最后成功记录及查询错误，另一侧独立更新。`OBSERVATION_UNAVAILABLE`不代表训练失败，不触发训练或coordinator重启。
- `test_monitor_observation_replay.py`：在独立、明确标记为mock的监控目录回放部分网关失败、连续超时及恢复，检查历史时间戳、看板状态和CSV状态。它不访问SSH、不执行训练，不生成科学结果。四类检查在操作员执行包内通过。

`audit_queue.py`、`held_controls.py` 和 `full_method_witness.py` 放在操作员配置的 `control/` 目录，激活脚本在本地执行包目录运行并复用该包的SSH辅助函数。manifest和bindings由冻结M矩阵生成并绑定真实数据、初始化权重、环境和工作目录；本目录不携带凭据或数据。主方法配置 `primary_binding_paths=[]`，补充队列指向同服务器主队列bindings；`slurm_nice`分别0/10000。N16的 `control/gpu1_nodes.json` 包含已经实际验证能分配物理GPU1的节点，不凭CUDA逻辑索引猜测物理编号。

监控脚本同样在操作员执行包运行，使用其`deploy_corrected.command`、`refresh_corrected_status.PROBE`和实际部署bindings；导出器读取本机冻结M仓库。这里提供审查副本，不是无资产即可启动的独立分发包。不要运行历史部署脚本的main。2026-09-08 08:24后N16网关两次通道建立错误促成此次修复；错误日志保留在执行包，模型及远端任务未改动。

2026-09-08现场发现：平台把主方法与补充作业的实际priority均截到1，Nice不能单独证明排序。主方法TIMEOUT后换了Slurm ID，旧`after`依赖已满足。因此补充coordinator先升级为`backfill-1h-primary-resume-deps-v2`，仅对本队列尚未运行的作业同步当前主方法开始依赖，已运行任务和其他任务不改动；相同依赖不重复写入。随后用户要求补充正式训练直接提交等待，两台补充coordinator已升级为`backfill-1h-held-controls-v3`，保留v2依赖同步并增加上面的提交/放行链。主coordinator仍用父审查提交`a23f58eee757d976ca3bc6fc339fb8d0dadf91eb`记录的v1；新增函数在空`primary_binding_paths`下不执行，主方法模型、训练器和配置均保持M不变。

v3的 `HELD_SUBMITTED` 是已有实际Slurm ID的正式训练，不是仅登记或GPU预检。挂起作业不计入正在占用的并发名额；原队列先处理已经就绪的评价/计时，再放行补充训练。能力通过前保持hold；放行前把真实能力凭证写入该次 `job.json`。提交响应不明确时保留 `SUBMITTING` 供核对，不盲目重提。Slurm记录消失后交给原有对账链；TIMEOUT仍恢复本任务自己的完整状态。看板将挂起作业显示为 `PENDING_HELD`，并显示实际Slurm ID和等待原因。具体现场ID及时间以操作员执行包的状态与激活凭证为准。

A100四项独立预检在一个allocation中执行，是提交名额的合并，不是减少检查内容。原245019–245022均在尚未运行时退役，合并作业245161已提交；逐项命令与原提交逐token比较一致，并在变更Slurm前通过实际 `bash -n`。对应 `precheck_submissions.json` 的四个条目共享此Slurm ID，但各自仍写入原来的独立预检目录。不得把共享ID算作少做三项检查，或把旧CANCELLED记录算作三项模型训练失败；也不要再次执行此一次性合并脚本。

06:17现场又确认A-uniform先于动态A的续训取得g0056 GPU1。轮询依赖不是抢占保证，因此N16补充队列现将允许节点列表暂置空，保留g0056/g0087的验证证据；这两个GPU1先供两项A主方法。已核对1278034的实际owner、脚本与第1轮完整checkpoint，将其当前分配结束时间改为06:19:29，由原TIMEOUT恢复链处理。待主方法完成释放一个节点或发现另一处已核验的空闲GPU1，再恢复补充训练的节点允许列表；不以精度作为资源释放条件。A100队列没有实施此节点保留操作。

审计后已经得到124项CPU测试和六个主方法的生产GPU预检通过。A/C指定内核下的源模型输出/相关梯度误差为0；实际检测头mask/loss/proposals/特征梯度也做了比较。B规则TIA内部长度384。此处不包含完整60轮结果，不宣称方法有效或加速。

资源调度额外使用 `--time-min=01:00:00`、最大24小时，让Slurm在较短空档先运行可恢复训练；完整训练仍为60轮。每epoch保存完整状态；TIMEOUT恢复同一任务已有checkpoint。独立evaluate/benchmark及官方原版训练没有应用该选项。入口的GPU1保护和UUID检查仍保留；N16获得其他物理卡时会在模型执行前退出78。

复审时继续严格检查：

1. 完整状态恢复后是否有重复优化、遗漏验证或错误准入；短时allocation不得被描述为缩短科研训练。
2. 补充任务的 `after` 依赖是否跟随当前续训ID、仅控制开始顺序；动态witness失败是否能错误发布能力凭证。依赖同步为coordinator轮询，不能包装为Slurm原生抢占优先级。
3. GPU节点查询与实际分配可能变化，入口是否会拒绝非授权GPU；不能用容器内NVML index0声称物理GPU0。
4. 冻结M模型源、外部执行脚本和实际checkpoint身份是否区分；代码中存在一种机制不能作为已完成实验的证据。

源报告的I01–I15逐项修复/阻塞说明见docs。117诊断和其他显式未实现扩展仍未完成。所有评论必须给出可达触发、具体源码位置和最小修复，不能用重新训练整个矩阵替代补测或重算。

2026-09-08研究定位修订见[文档入口](../../docs/README.md)：MoD＋同一TIA作为强对照，贡献假设转向定位价值学习。该文档修订不改变M、1545任务登记和远端队列；M0–M3/A2的新设计尚未实现或提交，当前A主方法仅按原目标归入A1机制。
