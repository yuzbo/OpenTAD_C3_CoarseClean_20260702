@RTK.md

# Repository Instructions

这是当前 C3 粗分类路线的纯净 OpenTAD 仓库。保持仓库小而可运行：不要加入历史 `research-wiki/`、旧 tracker、服务器日志、生成图、检查点、数据集、压缩包或旧路线报告。

## Objective

当前目标是用低成本粗分类模型产生可部署 `p_action` 信号，构造严格的 value-transport 帧选择 ledger，并把 384/768 选择输入接到 OpenTAD/AdaTAD 检测器。固定预算阶段是安全与归因锚点；最终目标仍是动态、任务感知的时序采集，并保护高 IoU TAD 定位。

## Scope

允许维护的主要表面：

- `opentad/`
- `configs/adatad/thumos/*c3*` 及其最小 base config
- `tools/bata/` 中的 C3 probe、ledger、validator 工具
- `scripts/` 中的 C3/N16R4 启动器
- focused `tests/`

新增内容应服务当前粗分类/ledger/AdaTAD 接入路线。不要把协调根里的历史 worktree、wiki、log 或临时产物搬进来。

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
