# C3 粗分类模型 + OpenTAD 纯净代码库

这是当前 C3 粗分类路线的纯净工作仓库，只保留 OpenTAD 代码库、当前粗分类探针、value-transport ledger 转换、C3 配置、N16R4 启动器和 focused tests。历史 `research-wiki/`、`logs/`、图表、检查点、压缩包和旧路线报告不属于本仓库。

来源：2026-07-02 从 `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3TCNCoarseProbe_Worktree_20260701` 创建；源分支为 `codex/c3-tcn-coarse-probe-20260701`，源 HEAD 为 `e9eee3b`，并包含当时本地未提交的粗分类训练器、GPU1 启动脚本和测试改动。

## 当前目标

最终目标是可部署的任务感知动态时序采集系统：根据视频、窗口、动作区域和难度动态分配帧/片段/Token，减少持续时间冗余，把更关键的信息送给 TAD 检测器，并尽量保持或提升高 IoU 定位性能。

当前 C3 阶段是固定预算控制锚点：用低成本粗分类模型估计动作/背景概率 `p_action`，把它转换成严格的帧选择 ledger，再将 384/768 的选择输入送入 OpenTAD/AdaTAD 检测器。该阶段用于归因、安全门和失败诊断，不是最终论文贡献的全部形态。

## 目录

- `opentad/`：OpenTAD 主库，以及当前 C3 selector、ledger、temporal grid、ActionFormer 接入代码。
- `configs/`：最小 OpenTAD base config 与当前 THUMOS14 C3 路线配置。
- `tools/bata/`：粗分类训练、模型矩阵、ledger 转换和启动门验证工具。
- `scripts/`：N16R4 GPU1 粗分类探针、ledger 导出、AdaTAD full-train 启动器。
- `docs/`：方法规划、审计吸收、规格说明及全量实验记录汇总文档（见 `docs/CONSOLIDATED_EXPERIMENTS_RECORD.md`）。
- `tests/`：当前 C3 路线的 focused tests。

## 本地使用

```powershell
cd E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702
pip install -r requirements.txt
```

轻量检查：

```powershell
python -m py_compile tools/train.py tools/test.py tools/bata/train_lowres_action_probe.py tools/bata/c3_coarse_classifier_model_matrix.py
python -m pytest tests/test_c3_coarse_classifier_model_matrix.py tests/test_c3_asformer_delta_ledger_full_train.py -q
```

完整 Torch 相关测试应在 N16R4 的 OpenTAD 环境中跑；当前本机 Windows Python 的 user-site `torch` 会在加载 `c10.dll` 时失败。

## N16R4 远端环境

本地登录用 Windows PowerShell 和原生 OpenSSH：

```powershell
ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i C:\Users\skywalker\.ssh\id_rsa -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com
```

远端只在 `~/run/yuzibo` / `/data/run01/sczc063/yuzibo` 下放代码、数据链接、缓存、日志、检查点和实验输出。不要在登录节点直接训练，正式训练使用 Slurm 或已授权的保护分配。

```bash
BASE=/data/run01/sczc063/yuzibo
module load cuda/11.8
module load miniforge3/24.11
source "$BASE/conda_envs/opentad/bin/activate"

export HOME="$BASE/tmp/home"
export XDG_CACHE_HOME="$BASE/tmp/xdg_cache"
export XDG_CONFIG_HOME="$BASE/tmp/xdg_config"
export HF_HOME="$BASE/hf_cache"
```

需要下载外部学术/模型资源时，在登录节点设置代理：

```bash
export http_proxy='http://u-MtfrT7:vH5orjDV@10.244.6.36:3128'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/raw/Validation Data/validation
$BASE/raw/Test Data/TH14_test_set_mp4
```

## 常用命令

GPU1 跑 temporal-TCN 粗分类探针：

```bash
cd /data/run01/sczc063/yuzibo/OpenTAD_C3_CoarseClean_20260702
CUDA_VISIBLE_DEVICES=1 bash scripts/run_c3_tcn_coarse_probe_gpu1_20260701.sh
```

GPU1 跑官方动作分割模型粗分类探针：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run_c3_official_action_seg_probe_gpu1_20260702.sh
```

把 probe checkpoint 导出为 deployable value-transport ledger：

```bash
CUDA_VISIBLE_DEVICES=1 PROBE_CHECKPOINT=/path/to/probe_reader.pth \
  EXPORT_SPLIT=val bash scripts/run_c3_lowres_probe_ledger_export_gpu1_20260702.sh
```

AdaTAD full-train 前置检查：

```bash
PRECHECK_ONLY=1 bash scripts/run_c3_asformer_delta_ledger_adatad_full_train_gpu1.sh
```

C3 主线优化默认使用物理 GPU1。GPU0 保留给发散创新实验，除非用户在同一轮明确覆盖。

