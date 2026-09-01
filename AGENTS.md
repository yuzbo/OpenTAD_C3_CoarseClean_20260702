@RTK.md

# Repository Instructions

这是当前 C3/DUCA 高效时序计算路线的 OpenTAD 仓库；C3 与 DUCA 是项目内沿用的方法路线名称。保持仓库可运行：不要加入旧 tracker、服务器日志、生成图、检查点、数据集、压缩包或与当前决策无关的旧路线报告。当前 `research-wiki/` 是必须维护的研究记忆，不属于应删除的历史负担。

## Objective

最终目标是面向离线时序动作检测（Temporal Action Detection, TAD）的任务感知时序去冗余：使用低成本的动作与状态证据分配帧、片段或层级计算，在降低真实端到端成本时保护高时间交并比（temporal Intersection over Union, tIoU）下的定位性能。C3、PAction、GAS-VT 和 lattice 是历史基线与归因工具；当前 DUCA 是位于重型视频骨干网络之前、仍需决定性实验检验的候选，不能称为在线时序动作检测，也不能在证据闭环前称为论文最终方法。

## Scope

允许维护的主要表面：

- `opentad/`
- `configs/adatad/thumos/*c3*` 及其最小基础配置
- `tools/bata/` 中的 C3 探针、选择记录和验证工具
- `scripts/` 中的 C3/N16R4 启动器
- 针对关键模型行为的聚焦 `tests/`
- `research-wiki/` 中当前研究决策、想法、实验、论文主张、证据缺口与来源记录

新增内容应服务当前 C3/DUCA/AdaTAD 路线或已声明的并行路线。不要把协调根里的旧 worktree、server log 或临时产物搬进来。

## Research Wiki Memory Contract

- 开始任何方法修改、实验部署、论文改写或 Pro 讨论前，必须先读 `research-wiki/query_pack.md` 和 `research-wiki/anti_repetition.md`。
- 新 idea、否定理由、路线选择、实验状态、原始结果或 claim 变化必须在同轮更新对应 wiki 节点与 `research-wiki/log.md`。
- Wiki 元数据可以使用 `discussed`、`designed`、`implemented`、`tested`、`experiment_running`、`empirically_supported` 和 `paper_ready`，但这些只是内部检索字段。面向研究者的正文必须用自然语言说明“提出、完成设计、已有代码、通过局部测试、实验运行中、获得实验证据、证据足以支撑论文主张”，不得把内部字段当作科学结论。
- 关系只写入 `research-wiki/graph/edges.jsonl`；原始来源登记在 `research-wiki/source_registry.md`。
- 失败或负结果是高价值研究记忆，不能从 `query_pack.md` 与 `anti_repetition.md` 中删除。

## Scientific Language and Context Fidelity

- 使用领域论文和科研社区广泛接受的术语。首次出现的非公认缩写应给出中文含义和英文全称；除原始代码标识、配置名和历史实验臂名称外，不创造私人缩写、状态语言或自造术语。
- 科研正文应让熟悉机器学习论文、但未参与本项目的人第一次阅读即可理解。内部任务编号、角色队列、浏览器调度和运行状态码不得进入论文进展或方法叙述。
- 文档必须忠实于当前权威科学状态，不得在转述中改变科学问题、机制、预测、公平比较、数据划分、指标或论文主张。缺失信息保持未知，冲突信息明确列出并交由科学判断解决。
- 新文档只表达已有设计与证据，不产生新的科学事实。局部测试、静态审查、运行成功和基础设施结果不得写成模型有效性证据；点估计、置信区间和因果解释也必须分别陈述。

## Formal Comparable Experiments

- 论文主比较及其结论必须来自完整、冻结的官方训练集和完整、独立的官方留出测试/评估集。所有 matched arms 都必须在完整训练集上重新训练；训练子集、160/40 或其他内部留出、smoke、pilot 和缩短训练只能用于机制诊断，不能替代正式比较或支撑论文主结果。
- 官方留出测试/评估集只允许用于预注册后的最终比较，不得参与训练、checkpoint/epoch 选择、超参数与阈值选择、规则或路线选择、反复调试及结果驱动的修改。正式评估前必须冻结完整视频 ID、annotation 语义、class mapping、evaluator、Soft-NMS 和一次性结果读取边界。
- OpenTAD/DUCA 的 211-video `validation` 与 ActionFormer 的 212-video `test` 是当前未解决的协议差异，不得按名称或近似数量静默视为同一集合。若训练/评估 subset 名称、完整视频集合或映射仍有冲突，必须在 Builder 或正式提交前停止并交由 Pro 裁决。
- 除预注册的唯一实验变量外，matched arms 必须共享起始 checkpoint/初始化、可训练参数集合、优化器与学习率日程、成功更新预算、seed 或预注册 seed 集、terminal checkpoint/EMA 规则、数据与后处理、评价器、真实计算量口径、prediction 保存和不确定性分析协议。缺少其中会改变比较解释的身份或产物时，不得声称结果正式可比。

## Remote Rules

远端写入边界是 `~/run/yuzibo` / `/data/run01/sczc063/yuzibo`。默认环境：

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
```

远端 GPU 任务必须由 Slurm 分配。不得固定物理 GPU 索引，也不得覆盖 Slurm 提供的 `CUDA_VISIBLE_DEVICES`；单卡任务在进程内使用 `cuda:0`。不要在登录节点直接训练。

## Silent Wall-clock Waiting

- 当用户明确要求等待，或一个尚未完成的 Goal 已无其他安全且有意义的工作、唯一正确动作是等待同一个已登记的 Pro 会话、下载、构建、训练或评测任务时，必须保持该 Goal，使用当前终端的阻塞计时命令真实等待（Bash/WSL/Linux/macOS 使用 `sleep <seconds>`，PowerShell 使用 `Start-Sleep -Seconds <seconds>`）；不得用模型推理、时间估算、快速轮询或提前结束 Goal 代替。
- 用户指定的等待间隔优先。未指定时，Pro、下载、构建等任务采用至多 10 分钟的有限间隔，完整训练或正式评测采用至多 30 分钟的有限间隔；每段结束后根据同一任务的真实状态决定是否进入下一段，不建立无限单次等待。
- 从阻塞计时命令开始到返回，计时是唯一允许的活动：不得发送 commentary、进度文字或倒计时输出，不得读取文件、日志、进程、作业或会话状态，不得调用其他工具，不得编辑、提交、部署、启动新任务或执行任何并行工作。
- 计时返回后至多对同一个权威作业或会话句柄做一次只读状态检查。若仍未终态且没有实质变化，立即开始下一段静默阻塞等待，不向用户重复报告“仍在运行”；只有终态、会改变下一动作的实质变化、真实阻塞或用户主动询问才允许输出。
- 本节是本仓库对长任务等待的具体规则；在 Goal 仍未完成且等待是唯一正确动作时，下述“退出主动执行阶段”表示停止科学推理和其他工作，不表示结束 Goal 或跳过终端静默等待。

## Long-running experiment waiting

下载、传输、Slurm 实验、完整训练或 Pro 长时间生成一旦成功提交，并记录不可变的作业或会话身份、代码与配置、预期终态产物和恢复规则，就不再属于 Codex 的主动推理工作。

- 具体实验确有需要时，使用现有服务或最小的任务专用 Slurm `afterany` 收尾作业记录终态；不得为等待建设通用调度系统。
- Codex 不得在静默等待间隔内或间隔之外仅为确认“仍在运行”而额外调用 `squeue`、`sacct`、`tail`、`ps`、`stat`、会话状态或等价工具。提交确认后应停止科学推理和其他主动工作；若当前 Goal 尚未完成，则按上节执行“静默计时 → 同一句柄一次只读检查”的有限循环。
- 计算系统可以定期写进度或 heartbeat；Codex 不高频消费，也不为无变化状态发消息。除上节每个静默间隔结束后的单次只读检查外，只在终态信号、外部通知、用户重新进入任务，或已有明确故障诊断需求时读取。
- Pro 生成同样适用：保留原会话与对话身份，不重复提交；后台生成期间不得让模型轮询等待。
- 运行进度和存活状态属于基础设施事实，不是科学结果，也不触发研究 Wiki 更新。

## Verification

改动后至少跑 focused checks：

```bash
python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

远端训练前先跑对应 `PRECHECK_ONLY=1` 启动器或 validator。

## ChronoTransport Parallel Route

ChronoTransport 是与 C3/DUCA 并行的动态特征刷新路线，不删除或重写现有路线。允许维护：

- `opentad/models/chronotransport/`；
- `configs/adatad/thumos/*chronotransport*`；
- `tools/bata/` 与 `scripts/` 中对应 validator/launcher；
- focused ChronoTransport tests 与方法文档。

v1 保持外部 768 点 detector 网格、384 点内部 tubelet 网格，以及 dense patch embedding、AdaTAD temporal adapter、head 和 NMS。无实测成本、校准风险、有效 cache 或专用 checkpoint 时必须回退 dense。validation/test GT、teacher、raw-prediction cache 和 counterfactual ledger 不得参与推理决策。
