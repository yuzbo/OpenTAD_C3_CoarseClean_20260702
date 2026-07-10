@RTK.md

# Repository Instructions

这是当前 C3/DUCA 高效时序计算路线的 OpenTAD 仓库。保持仓库可运行：不要加入旧 tracker、服务器日志、生成图、检查点、数据集、压缩包或与当前决策无关的旧路线报告。当前 `research-wiki/` 是必须维护的研究记忆，不属于应删除的历史负担。

## Objective

最终目标是面向离线 TAD 的任务感知时序去冗余：使用低成本粗粒度动作/状态证据分配帧、片段或层级计算，在真实总成本下降时保护高 IoU 定位。C3/PAction/GAS-VT/lattice 是历史基线和归因工具；当前 DUCA 是待决定性实验裁决的 pre-backbone 候选，不能称 Online TAD，也不能在证据闭环前称论文最终方法。

## Scope

允许维护的主要表面：

- `opentad/`
- `configs/adatad/thumos/*c3*` 及其最小 base config
- `tools/bata/` 中的 C3 probe、ledger、validator 工具
- `scripts/` 中的 C3/N16R4 启动器
- focused `tests/`
- `research-wiki/` 中当前研究决策、idea、实验、claim、gap 与来源记录

新增内容应服务当前 C3/DUCA/AdaTAD 路线或已声明的并行路线。不要把协调根里的旧 worktree、server log 或临时产物搬进来。

## Research Wiki Memory Contract

- 开始任何方法修改、实验部署、论文改写或 Pro 讨论前，必须先读 `research-wiki/query_pack.md` 和 `research-wiki/anti_repetition.md`。
- 新 idea、否定理由、路线选择、实验状态、原始结果或 claim 变化必须在同轮更新对应 wiki 节点与 `research-wiki/log.md`。
- 必须区分 `discussed`、`designed`、`implemented`、`tested`、`experiment_running`、`empirically_supported` 和 `paper_ready`；不得跨级声称完成。
- 关系只写入 `research-wiki/graph/edges.jsonl`；原始来源登记在 `research-wiki/source_registry.md`。
- failed/negative ideas 是高价值记忆，不能从 `query_pack.md` 与 `anti_repetition.md` 中删除。

## Remote Rules

远端写入边界是 `~/run/yuzibo` / `/data/run01/sczc063/yuzibo`。默认环境：

```bash
module load cuda/11.8
module load miniforge3/24.11
source /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate
```

C3 主线优化默认使用物理 GPU1；GPU1 子启动器必须在 `CUDA_VISIBLE_DEVICES=1` 时才继续。不要在登录节点直接训练。

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
