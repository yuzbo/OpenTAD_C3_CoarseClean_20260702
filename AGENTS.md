@RTK.md

# Repository Instructions

这是 ZoomToken 面向离线时序动作检测（TAD）的论文实验仓库。保持仓库可运行：不要加入旧 tracker、服务器日志、生成图、检查点、数据集、压缩包或与当前科学决策无关的旧路线报告。当前 `research-wiki/` 是必须维护的研究记忆，不属于应删除的历史负担。

## Objective

最终目标是面向离线 TAD 的任务感知冗余计算削减：在真实端到端成本下降时保护动作边界和高 tIoU 定位。当前主问题由 `ZoomToken-BPNS-R1` 检验：只依赖当前观测，在 VideoMAE 前保留连续无孔洞的 `8×8/K64` 原生空间支持，所有保留 token 仍完整通过 12 层主干和既有 Adapter。C3、PAction、GAS-VT、lattice 与 DUCA 是历史基线、归因工具或已停止候选；它们不是当前论文方法。BPNS-R1 已有单种子准确率证据，但真实效率尚未建立，因此也不能称为论文最终方法或 Online TAD。

## Scope

允许维护的主要表面：

- `opentad/`
- `configs/adatad/thumos/*c3*` 及其最小 base config
- `tools/bata/` 中的 C3 probe、ledger、validator 工具
- `scripts/` 中的 C3/N16R4 启动器
- focused `tests/`
- `research-wiki/` 中当前研究决策、idea、实验、claim、gap 与来源记录

新增内容应直接服务当前 BPNS-R1/AdaTAD 科学问题、必要对照或已明确声明的并行假设。不要把协调根里的旧 worktree、server log 或临时产物搬进来。

## Research Wiki Memory

- 开始方法修改、实验部署、论文改写或新的科学讨论前，先读 `research-wiki/query_pack.md` 和 `research-wiki/anti_repetition.md`。
- 当前记忆只需清楚保存：研究问题、最近相关工作、候选主张、机制、可证伪预测、最强替代解释、当前实验、决定性证据、已停止方向、未知项和下一项 Pro 科学决策。
- 写作必须区分可核验事实、来源中的陈述、项目解释和待检验提案；实现完成、实验运行和论文证据是不同事实，不用私有状态码代替这些普通表述。
- 只有科学问题、证据、路线或主张发生实质变化时才更新相应节点。原始结果和重要负证据保留在历史记录中；浏览器调度、队列、重复审查和一般协调信息不进入论文叙事。
- failed/negative ideas 是高价值记忆，不能从 `query_pack.md` 与 `anti_repetition.md` 中删除或改写成成功叙事。

## Formal Dataset Completeness

- 任何用于模型准确率、泛化结论或论文表格的正式训练，必须覆盖冻结协议指定的完整官方训练集、完整 epoch/update 计划和全部训练样本。subset、short-run、截断 epoch、缩短 loader、smoke 或 `PRECHECK_ONLY` 只能形成工程证据，不能冒充科学结果。
- 任何可比较的正式评测必须让所有 arm 覆盖同一个完整官方评测/测试 population，并匹配 annotation、媒体 inventory、样本顺序与数量、evaluator、postprocess 和 NMS。不得用 validation 子集、抽样窗口或 development replay 替代要求的 official test。
- 只要对比包含训练，baseline 与每个 candidate arm 都必须使用同一完整官方训练 population、同一完整训练计划，并在同一冻结的完整评测 population 上比较；任何 arm 缺样本、缩短训练、跨 validation/test split 或只有部分终态时，该对比都只能分类为协议 blocker。论文结论若要求 official test，则 baseline 与所有候选必须在同一个一次性、无泄漏 test-opening 协议下完整评测。
- official test 只能在独立冻结、无 validation/test GT 泄漏的 test-opening 协议下完整运行。无法由终态 receipt 证明训练集或评测集完整性时，结果分类为协议 blocker，不从子集外推科学结论。
- 只读成本实验不重新训练模型，但仍必须在所有 arm 上使用相同的完整冻结评测 population 和完整端到端通路。当前四臂 job `1262120` 固定为完整 THUMOS14 validation population（211 videos / 792 ordered windows）的 matched full-stack 成本回放；它不是 official-test 准确率实验，不得升级为 official-test 证据。

## Silent Terminal Waiting

- 当没有进一步立即可执行的科学/工程安排，或完整正式实验/外部进程正在运行且唯一正确动作是等待时，必须启动或沿用唯一的真实终端倒计时进程静默等待；Bash 使用 `sleep 600`，PowerShell 使用 `Start-Sleep -Seconds 600`（冻结任务另有间隔时按冻结值）。不得用模型推理、字符输出、预计完成时间或连续快速查询模拟墙钟等待。
- 默认等待间隔为 10 分钟，除非冻结任务另有规定。只允许一个计时器/终态 waiter；已有后端 waiter 时不得再建立第二套前台或后台轮询。
- 从计时命令开始到返回的整个窗口内不得打印倒计时、进度点、“仍在等待”等文字，不得查询任务状态、读取日志或 partial 指标，不得修改文件、操作浏览器、提交新任务或执行任何与计时无关的命令。
- 普通 heartbeat、goal continuation、界面刷新或状态问候都不得打断已经开始的 10 分钟静默窗口，也不得触发字符输出或额外工具调用；只有该计时器自然返回、后端主动给出终态，或用户用新的明确请求替换当前等待任务时，才能恢复动作。
- 计时返回后只做一次权威终态检查。若仍未终态且没有硬故障或预注册恢复条件，立即进入下一轮同样的静默终端等待；只有终态、失败/停止条件或真正需要人类输入的 blocker 才离开等待循环。

## Remote Rules

远端写入边界是 `~/run/yuzibo` / `/data/run01/sczc063/yuzibo`。默认环境：

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
```

远端 GPU 任务必须由 Slurm 分配。不得固定物理 GPU 索引，也不得覆盖 Slurm 提供的 `CUDA_VISIBLE_DEVICES`；单卡任务在进程内使用 `cuda:0`。不要在登录节点直接训练。

## Verification

改动后至少跑 focused checks：

```bash
python -m py_compile tools/train.py tools/bata/train_lowres_action_probe.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

远端训练前先跑对应 `PRECHECK_ONLY=1` 启动器或 validator。

## ChronoTransport Parallel Route

ChronoTransport 是与 BPNS-R1 相互独立的动态特征刷新假设，不删除或重写现有路线。允许维护：

- `opentad/models/chronotransport/`；
- `configs/adatad/thumos/*chronotransport*`；
- `tools/bata/` 与 `scripts/` 中对应 validator/launcher；
- focused ChronoTransport tests 与方法文档。

v1 保持外部 768 点 detector 网格、384 点内部 tubelet 网格，以及 dense patch embedding、AdaTAD temporal adapter、head 和 NMS。无实测成本、校准风险、有效 cache 或专用 checkpoint 时必须回退 dense。validation/test GT、teacher、raw-prediction cache 和 counterfactual ledger 不得参与推理决策。
