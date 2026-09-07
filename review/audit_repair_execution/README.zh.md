# 审计修订版的执行代码复审

实际模型/训练源码 M 固定为 `b70ae056c495b43ca3f305fe438926b97b2723b5`。本审查分支在 M 上只添加这个执行审查目录；`geosparse_ext/`、`tests/`、`opentad/`、`configs/` 与 M 相同。不要把仅添加审查材料的提交误当作新的模型实验版本，或要求已正确启动的 M 训练重跑。

先读 `docs/PRO_REVIEW_AUDIT_REPAIR_PROMPT.zh.md` 和 `docs/GEOSPARSE_AUDIT_REPAIR.zh.md`。这里两份脚本与实际部署的外部执行脚本对应，补足冻结模型仓库以外的调度行为：

- `audit_queue.py`：主方法与补充控制的真实Slurm调度。只有资源、实现、正确性和自身产物依赖；主方法没有mAP门槛。补充控制使用较低priority，并只等待主方法开始。
- `full_method_witness.py`：动态预算、预算头与acquisition梯度、dual/cost EMA更新的实际batch正确性检查，模型在检查后丢弃，不生成科学结果。

运行时把这两份脚本放在操作员配置的 `control/` 目录。manifest和bindings由冻结M矩阵生成并绑定真实数据、初始化权重、环境和工作目录；本目录不携带凭据或数据。主方法N16配置 `primary_binding_paths=[]`，补充队列指向同服务器主队列bindings；`slurm_nice`分别0/10000。N16的 `control/gpu1_nodes.json` 包含已经实际验证能分配物理GPU1的节点，不凭CUDA逻辑索引猜测物理编号。

审计后已经得到124项CPU测试和六个主方法的生产GPU预检通过。A/C指定内核下的源模型输出/相关梯度误差为0；实际检测头mask/loss/proposals/特征梯度也做了比较。B规则TIA内部长度384。此处不包含完整60轮结果，不宣称方法有效或加速。

资源调度额外使用 `--time-min=01:00:00`、最大24小时，让Slurm在较短空档先运行可恢复训练；完整训练仍为60轮。每epoch保存完整状态；TIMEOUT恢复同一任务已有checkpoint。独立evaluate/benchmark及官方原版训练没有应用该选项。入口的GPU1保护和UUID检查仍保留；N16获得其他物理卡时会在模型执行前退出78。

复审时继续严格检查：

1. 完整状态恢复后是否有重复优化、遗漏验证或错误准入；短时allocation不得被描述为缩短科研训练。
2. 补充任务的 `after` 依赖是否只控制开始顺序；动态witness失败是否能错误发布能力凭证。
3. GPU节点查询与实际分配可能变化，入口是否会拒绝非授权GPU；不能用容器内NVML index0声称物理GPU0。
4. 冻结M模型源、外部执行脚本和实际checkpoint身份是否区分；代码中存在一种机制不能作为已完成实验的证据。

源报告的I01–I15逐项修复/阻塞说明见docs。117诊断和其他显式未实现扩展仍未完成。所有评论必须给出可达触发、具体源码位置和最小修复，不能用重新训练整个矩阵替代补测或重算。
