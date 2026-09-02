# CT-DP-BAMoD 4 臂全因子消融实验全量追踪与结果回收文档
# (CT-DP-BAMoD 4-Arm Matrix Experiment Recovery & Provenance Ledger)

> **更新时间**：2026-09-02  
> **归属主线**：DUCA / C3 / Continuous-Time Dual-Phase TAD 主线（本纯净仓库原生）  
> **远端部署根目录**：`/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901`  
> **状态**：**4 臂全部 RUNNING 正常训练中 (当前进度 Epoch 34~36 / 60)**  
> **目的**：锁定 4 臂消融实验的完整配置、调度节点、Slurm Job ID、日志路径、权重保存路径与评估产物回收指令，防止实验运行完毕后无法精确归因与回收。

---

## 1. 实验定位与路线边界裁决

### 1.1 实验体系归属
- **CT-DP-BAMoD 4 臂实验**：**严格基于当前 `OpenTAD_C3_CoarseClean` 纯净代码库架构**。它通过前置双相选帧（`DualPhaseFrameSelector`，全局 128 骨架 + 局部 256 边界微簇 = 384 稀疏帧）、骨干网 3D 卷积物理速度归一化（CT-Tubelet）、边界偏置混合专家稀疏路由（B-AMoD）与连续物理坐标自适应检测头（CT-Conv1d），构成完整的 pre-backbone 稀疏时序采集与物理检测闭环。
- **与 Zoom TAD / ZoomToken 的边界隔离**：Zoom TAD（如 BA-FDR K16）属于外部多分辨率空间裁剪刷新分支（$96\times 96$ 全局 + $128\times 128$ 局部 ROI Token），不属于本纯净仓库的前置时序稀疏主线，本实验记录严格不包含任何 ZoomToken / Zoom TAD 实验。

---

## 3. 4 臂机制正交定义与最终完赛指标 (Completed Full Results)

| 臂编号 | 架构与机制定义 | 选帧预算 | 配置文件 | 运行节点 / Slurm Job | 状态 | **最终 Avg-mAP** | **mAP@0.3** | **mAP@0.4** | **mAP@0.5** | **mAP@0.6** | **mAP@0.7** |
|:---:|---|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm 1 (主方法)** | **Full CT-DP-BAMoD**<br>(CT-Conv + B-AMoD) | 384/768 | `configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py` | `g0048` / `1264438` | **COMPLETED** | **51.63%** | 70.26% | 63.87% | 54.82% | 42.71% | **26.51%** |
| **Arm 2 (消融 CT-Conv)** | **DP-BAMoD w/ Std Conv**<br>(Std Conv1d + B-AMoD) | 384/768 | `configs/adatad/thumos/duca_dual_phase_bamod_thumos.py` | `g0063` / `1264439` | **RUNNING** (Ep 59 终评) | **52.68%** (Ep 57) | 71.00% | 64.65% | 55.98% | 44.26% | **27.52%** |
| **Arm 3 (消融 B-AMoD)** | **CT-DP w/ Dense ViT**<br>(CT-Conv + Dense ViT) | 384/768 | `configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py` | `g0056` / `1264440` | **COMPLETED** | **57.61%** | 76.20% | 70.11% | 61.95% | 48.95% | **30.84%** |
| **Arm 4 (双消融对照)** | **DP-DenseViT w/ Std Conv**<br>(Std Conv1d + Dense ViT) | 384/768 | `configs/adatad/thumos/duca_dual_phase_densevit_stdconv_thumos.py` | `g0056` / `1264441` | **COMPLETED** | **57.99%** | 76.20% | 70.79% | 61.43% | 49.38% | **32.14%** |
远端部署调度与实时运行清单

### 3.1 调度与运行节点映射

| 实验臂 | Slurm Job ID | 调度计算节点 | GPU 分配 | 运行状态 | 当前训练轮次 | 当前训练损失 | 显存占用 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Arm 2** | `1264439` | `g0063` | 1 × GPU | **RUNNING** (04:10+) | **Epoch 33** | $\text{Loss}=0.576, \text{cls}=0.315, \text{reg}=0.261$ | $4118\text{MB}$ |
| **Arm 3** | `1264440` | `g0056` | 1 × GPU | **RUNNING** (04:10+) | **Epoch 36** | $\text{Loss}=0.560, \text{cls}=0.293, \text{reg}=0.267$ | $3597\text{MB}$ |
| **Arm 4** | `1264441` | `g0056` | 1 × GPU | **RUNNING** (04:10+) | **Epoch 36** | $\text{Loss}=0.550, \text{cls}=0.287, \text{reg}=0.263$ | $3598\text{MB}$ |

### 3.2 训练参数与评测计划 (Evaluation Schedule)
- **数据集**：THUMOS14 官方划分（200 训练 / 211 验证）
- **优化器与学习率**：AdamW，`lr_backbone=1.8e-4`，`lr_det=9.0e-5`，Cosine Annealing
- **总轮次**：60 轮（每轮 100 次有效更新，共计 6,000 次优化器更新）
- **在线评测机制**：`val_start_epoch=40, val_eval_interval=2`
  - **第 0~39 轮**：专注快速特征与检测头收敛，不进行验证评估；
  - **第 40~60 轮**：每 2 轮在验证集上执行全量评估（Epoch 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60），自动输出 Avg-mAP 及 mAP@0.3~0.7。

---

## 4. 产物路径与结果回收协议 (Artifact Recovery Protocol)

为确保实验完成后数据可 100% 完整回收，以下路径与产物已实现严格绑定：

### 4.1 原始日志文件 (Slurm Logs)
- **Arm 1**：`/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_bamod_s3407_1264438.out` (及 `.err`)
- **Arm 2**：`/data/run01/sczc063/yuzibo/slurm_logs/dp_bamod_stdconv_s3407_1264439.out` (及 `.err`)
- **Arm 3**：`/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_densevit_s3407_1264440.out` (及 `.err`)
- **Arm 4**：`/data/run01/sczc063/yuzibo/slurm_logs/dp_densevit_stdconv_s3407_1264441.out` (及 `.err`)

### 4.2 实验目录与检查点位置 (Checkpoint Directories)
- **Arm 1**：`/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/exps/thumos/adatad/duca_ct_dual_phase_bamod_seed3407/gpu1_id0/`
  - 检查点：`checkpoint/epoch_*.pth`（每 2 轮保存）
  - 结构化指标日志：`log.json`（实时记录每步损失与验证 mAP）
- **Arm 2**：`/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/exps/thumos/adatad/duca_dual_phase_bamod_stdconv_seed3407/gpu1_id0/`
- **Arm 3**：`/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/exps/thumos/adatad/duca_ct_dual_phase_densevit_seed3407/gpu1_id0/`
- **Arm 4**：`/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/exps/thumos/adatad/duca_dual_phase_densevit_stdconv_seed3407/gpu1_id0/`

---

## 5. 一键结果回收与监控命令 (Recovery Tooling)

### 5.1 实时查看 4 臂训练最新进展
```bash
ssh -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com \
  "echo '=== Arm 1 ==='; tail -n 10 /data/run01/sczc063/yuzibo/slurm_logs/ct_dp_bamod_s3407_1264438.out; \
   echo '=== Arm 2 ==='; tail -n 10 /data/run01/sczc063/yuzibo/slurm_logs/dp_bamod_stdconv_s3407_1264439.out; \
   echo '=== Arm 3 ==='; tail -n 10 /data/run01/sczc063/yuzibo/slurm_logs/ct_dp_densevit_s3407_1264440.out; \
   echo '=== Arm 4 ==='; tail -n 10 /data/run01/sczc063/yuzibo/slurm_logs/dp_densevit_stdconv_s3407_1264441.out"
```

### 5.2 提取 Epoch 40~60 验证集 mAP 对比表
```bash
ssh -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com \
  "python -c '
import json, glob
for path in sorted(glob.glob(\"/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/exps/thumos/adatad/*/gpu1_id0/log.json\")):
    exp_name = path.split(\"/\")[-3]
    print(f\"\\n>>> {exp_name}\")
    with open(path) as f:
        lines = [line.strip() for line in f if \"Average-mAP\" in line or \"mAP at tIoU 0.70\" in line]
        for l in lines[-10:]: print(l)
'"
```

---
*文档已密封生效，可作为 4 臂实验结束后的统一验收与指标提取依据。*
