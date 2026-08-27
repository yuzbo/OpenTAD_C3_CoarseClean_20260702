# ZoomToken：离线 TAD 冗余计算研究

这是 ZoomToken 的论文实验仓库，研究如何在离线时序动作检测中减少 VideoMAE 的冗余计算，同时保护动作边界和高 tIoU 定位。仓库保留 OpenTAD 主代码、当前方法与必要对照、最小实验工具、focused tests，以及作为长期科学记忆的 `research-wiki/`。服务器日志、生成图、检查点、数据集、压缩包和与当前决策无关的旧路线报告不进入 Git 仓库。

来源：2026-07-02 从 `E:\DeskTop\TAD\temrefuse-tad\OpenTAD_C3TCNCoarseProbe_Worktree_20260701` 创建；源分支为 `codex/c3-tcn-coarse-probe-20260701`，源 HEAD 为 `e9eee3b`，并包含当时本地未提交的粗分类训练器、GPU1 启动脚本和测试改动。

## 当前目标

最终目标是可部署的任务感知动态时序采集系统：根据视频、窗口、动作区域和难度动态分配帧/片段/Token，减少持续时间冗余，把更关键的信息送给 TAD 检测器，并尽量保持或提升高 IoU 定位性能。

当前主问题由 `ZoomToken-BPNS-R1` 检验：仅根据当前观测，在 VideoMAE 前保留一个连续无孔洞的 `8×8/K64` 原生空间支持，所有保留 token 仍完整通过 12 层 VideoMAE-S 和既有 Adapter。已有单种子结果支持准确率可行性，但同硬件完整成本尚未测成，因此“36% 原生 token 减少”只能称结构性计算代理，不能称实际加速、节能或显存收益。C3 与 DUCA 继续作为历史基线和归因材料，不是当前论文方法。

## 目录

- `opentad/`：OpenTAD 主库，以及当前 C3 selector、ledger、temporal grid、ActionFormer 接入代码。
- `configs/`：最小 OpenTAD base config 与当前 THUMOS14 C3 路线配置。
- `tools/bata/`：粗分类训练、模型矩阵、ledger 转换和启动门验证工具。
- `scripts/`：N16R4 GPU1 粗分类探针、ledger 导出、AdaTAD full-train 启动器。
- `tests/`：当前 C3 路线的 focused tests。
- `research-wiki/`：当前问题、证据、负结果、主张边界与下一科学决策。

## 本地使用

```powershell
cd E:\DeskTop\TAD\OpenTAD_ZoomToken_CVPR2027
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

## GPU 任务

正式 GPU 任务统一交给 Slurm 分配，不固定物理索引，也不覆盖 Slurm 提供的
`CUDA_VISIBLE_DEVICES`。单卡作业只在进程内使用逻辑设备 `cuda:0`，例如：

```bash
sbatch --partition=gpu --gres=gpu:1 \
  --wrap='bash scripts/run_spatial_zoom_s1_precheck_slurm.sh'
```

仓库中名称带 `gpu0`/`gpu1` 的脚本与相关命令属于历史实验协议，仅保留用于审计；
再次运行前必须迁移为正常 Slurm 映射并重新通过对应门禁。
