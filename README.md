# C3 / DUCA 高效时序计算研究仓库

C3 与 DUCA 是本项目沿用的两条方法路线名称；本仓库是它们的 OpenTAD 工作仓库。仓库保留当前模型代码、实验配置、必要的训练与评测工具、聚焦测试，以及 `research-wiki/` 中不可替代的研究记忆。服务器日志、检查点、数据集、压缩包和与当前判断无关的临时产物不进入版本库。

来源：2026-07-02 从 `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3TCNCoarseProbe_Worktree_20260701` 创建；源分支为 `codex/c3-tcn-coarse-probe-20260701`，源 HEAD 为 `e9eee3b`，并包含当时本地未提交的粗分类训练器、GPU1 启动脚本和测试改动。

## 当前目标

最终目标是面向离线时序动作检测（Temporal Action Detection, TAD）的任务感知动态时序采集：根据视频、窗口、动作区域和难度分配帧、片段或时序表示的计算量，在降低真实端到端成本的同时，尽量保持或提升高时间交并比（temporal Intersection over Union, tIoU）下的边界定位性能。

当前 C3 阶段是固定预算的归因基线：低成本粗分类模型估计逐时刻动作概率 `p_action`，确定性规则据此生成帧选择记录，再把 384/768 的稀疏输入送入 OpenTAD/AdaTAD 检测器。该阶段用于机制归因和失败诊断，不代表最终论文方法已经确定。

## 目录

- `opentad/`：OpenTAD 主库，以及当前 C3 选择器、选择记录、时间坐标和 ActionFormer 接入代码。
- `configs/`：最小 OpenTAD 基础配置与当前 THUMOS14 C3 路线配置。
- `tools/bata/`：粗分类训练、模型矩阵、选择记录转换和运行前验证工具。
- `scripts/`：N16R4 粗分类探针、选择记录导出和 AdaTAD 完整训练启动器。
- `tests/`：针对当前方法关键行为的聚焦测试。

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

需要下载已获准的外部学术或模型资源时，从个人环境加载代理配置。代理地址和凭据不得写入仓库：

```bash
export http_proxy='<authorized-proxy-url>'
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"
```

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/thumos14/raw_data/video
$BASE/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth
```

## GPU 任务

正式 GPU 任务统一交给 Slurm 分配，不固定物理索引，也不覆盖 Slurm 提供的
`CUDA_VISIBLE_DEVICES`。单卡作业只在进程内使用逻辑设备 `cuda:0`，例如：

```bash
sbatch --partition=gpu --gres=gpu:1 \
  --wrap='bash scripts/run_spatial_zoom_s1_precheck_slurm.sh'
```

仓库中名称带 `gpu0`/`gpu1` 的脚本与相关命令属于历史实验协议，仅保留用于审计；
再次运行前必须迁移为正常 Slurm 映射并重新通过对应门禁。
