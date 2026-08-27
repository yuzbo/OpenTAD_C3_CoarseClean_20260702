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
