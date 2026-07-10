# RTK Project Rules

## North Star

最终目标是一个独立的离线 PhysTime-TAD 检测器：接收任意不规则视频观测、真实时间戳和可审计支持区间，在物理时间轴上直接完成动作分类与起止边界定位，并在缺帧、非均匀密度和不同 FPS 下保护高-IoU 定位。

当前阶段只实现 PhysTime-AdaTAD 1.0 的 matched raw-video K384 head comparison。完整定义见：

- `research-wiki/current_direction.md`
- `docs/superpowers/specs/2026-07-11-phystime-adatad-1-design.md`
- `docs/superpowers/plans/2026-07-11-phystime-adatad-1.md`

## Historical Route Boundary

C3/PAction/GAS-VT/DUCA/MUST/CFPA/X3D/SlowFast 是历史 baseline、诊断或工程资产。除非 `research-wiki/decision_register.md` 新增 superseding decision，否则不得把这些路线重新设为论文主线，也不得继续通过新增 prior、loss weight、top-k 或 gap/radius 修补来扩展。

## Physical-Time Contract

- canonical coordinate 是 absolute video seconds；
- GT 与预测不得映射到 selected-rank；
- `round(time_sec * fps)` 可用于原视频帧号导出；
- support interval 不得跨 sparse gap 扩张；
- query grid 由物理时长/spacing 定义，不由 K 或 rank stride 定义；
- predictions in seconds 只允许 duration clamp，不经过 snippet-axis inverse remap。

## Primary Comparison Contract

- logical dense window: 768；
- selected raw observations: K=384；
- sampling: deterministic, no learning, no GT；
- same selected indices across three heads；
- same official VideoMAE-S/AdaTAD backbone and optimizer；
- selected-axis、physical-grid、PhysTime 三头只改变时间几何与检测 head；
- no selector, actionness, teacher, ledger, dynamic budget or paired consistency in Phase 1。

## Evidence and Storage

- 实验数字唯一写入 `docs/evaluation/results.md`；
- 研究方向、失败理由与裁决写入 `research-wiki/`；
- 不提交数据集、checkpoint、server logs、run directories、生成图或压缩包；
- 所有正式实验绑定 commit、config hash、checkpoint hash、dataset manifest 和 Slurm job。

## Remote Environment

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

THUMOS14 默认路径：

```bash
$BASE/thumos14/annotations/thumos_14_anno.json
$BASE/thumos14/annotations/category_idx.txt
$BASE/raw/Validation Data/validation
$BASE/raw/Test Data/TH14_test_set_mp4
```

不在登录节点直接训练。正式任务使用 Slurm，训练前先跑对应 validator 与 real-data gate。
