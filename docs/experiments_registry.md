# 实验全景追踪登记册 (Experiments Registry & Live Tracker)

本文档记录 **连续物理时空自适应与双相动作检测体系（Continuous-Time Scale-Adaptive & Dual-Phase Action Detection, CT-DP-TAD / DUCA-BAMoD）** 的所有实验矩阵、本地与远端部署状态、Slurm 作业句柄以及结果更新协议。

---

## 1. 代码库与部署环境元数据

| 项目属性 | 配置与路径信息 |
| :--- | :--- |
| **本地代码仓库根目录** | `E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702` |
| **GitHub 远程仓库 URL** | [OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git) |
| **当前开发分支** | `codex/duca-total60-plugin-cvpr-20260727` |
| **当前核心提交 Commit** | `c26883e30fdfefbcdd23f7ce2e49c7bcfa93b2ef` |
| **GitHub 提交树链接** | [Commit c26883e3 Tree](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/c26883e30fdfefbcdd23f7ce2e49c7bcfa93b2ef) |
| **远端计算集群** | `N16R4` (`sczc063@...`) |
| **远端独立工作区** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901` |
| **远端 Python/PyTorch 环境** | `/data/run01/sczc063/yuzibo/conda_envs/opentad` (Python 3.10, PyTorch 2.0.1 + CUDA 11.8) |
| **THUMOS-14 数据集软链** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/data/thumos-14` |
| **预训练 VideoMAE 权重软链** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/pretrained` |

---

## 2. 单种子三实验正式消融矩阵 (Single-Seed 3407 Factorial Matrix)

为了严谨正交归因 **连续物理时间调制卷积 (Continuous-Time Conv1d)** 与 **边界调制自适应 Token 稀疏化 (B-AMoD)** 的独立与协同效能，建立以下 3 臂全训练消融对照：

| 实验编号 (Arm) | 实验名称与科学机制 | 本地/远端配置文件路径 | Slurm Job ID | 部署/运行状态 | 日志与产物路径 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arm 1 (完整主方法)** | **Full CT-DP-BAMoD**<br>• 双相选帧 ($K=384, K_s=128, K_b=256$)<br>• B-AMoD ViT-Adapter (6 Dense + 6 调制层)<br>• CT-Tubelet 3D 速度归一化 Patch 嵌入<br>• CT-Conv1d 自适应连续物理卷积<br>• ActionFormer 头 | [`configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py) | **`1263598`** | `PENDING (Priority)` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_bamod_s3407_1263598.out`<br>**输出**: `exps/thumos/adatad/duca_ct_dual_phase_bamod_thumos/` |
| **Arm 2 (消融 CT-Conv)** | **DP-BAMoD (Standard Conv1d)**<br>• 双相选帧 ($K=384$)<br>• B-AMoD ViT-Adapter (6 Dense + 6 调制层)<br>• **消融项**：回退为标准均匀 Conv1d (无物理时空采样偏移修正) | [`configs/adatad/thumos/duca_dual_phase_bamod_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_dual_phase_bamod_thumos.py) | **`1263599`** | `PENDING (Priority)` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/dp_bamod_stdconv_s3407_1263599.out`<br>**输出**: `exps/thumos/adatad/duca_dual_phase_bamod_thumos/` |
| **Arm 3 (消融 B-AMoD)** | **CT-DP (Dense ViT-Adapter)**<br>• 双相选帧 ($K=384$)<br>• **消融项**：ViT-Adapter 全 12 层稠密计算 (无 Token 剪枝与稀疏路由)<br>• CT-Conv1d 自适应连续物理卷积 | [`configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py) | **`1263600`** | `PENDING (Priority)` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_densevit_s3407_1263600.out`<br>**输出**: `exps/thumos/adatad/duca_ct_dual_phase_densevit_thumos/` |
| **Baseline (历史基准)** | **Dense 768 / Uniform 384 Baseline**<br>• 稠密 768 帧输入 / 均匀 384 帧降采样基线<br>• 仅作指标基准对比，**严禁重新提交** | 历史已冻结检查点与测试记录 | N/A (历史已就绪) | `COMPLETED` | 历史评测基准数据已锁定 |

---

## 3. 实验评测指标登记表 (待训练完成后更新)

所有实验均在 THUMOS-14 官方 211 验证集上使用官方评估协议进行评测（mAP@0.3 ~ 0.7 及 Avg-mAP）：

| 实验组别 | 选帧预算 $K$ | 骨干等效层数 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP (0.3:0.7) | 相对基线收益 (Avg-mAP) | GFLOPs / 相对算力 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense 768 Baseline** | 768 | 12 层 | - | - | - | - | - | 历史基准 | 基准 (0.0) | 100% |
| **Uniform 384 Baseline** | 384 | 12 层 | - | - | - | - | - | 历史基准 | - | ~50% |
| **Arm 1: Full CT-DP-BAMoD** (`1263598`) | 384 (128+256) | 9 层 (6D+6M) | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | **~35%** |
| **Arm 2: DP-BAMoD (StdConv)** (`1263599`) | 384 (128+256) | 9 层 (6D+6M) | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | **~35%** |
| **Arm 3: CT-DP (Dense ViT)** (`1263600`) | 384 (128+256) | 12 层稠密 | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | ~50% |

---

## 4. 实验监控与结果自动更新协议

1. **静默阻塞监控规则**：
   - 遵循 `AGENTS.md` 规范，Slurm 训练任务提交后，通过静默阻塞（每 30 分钟检查一次）轮询 Slurm 队列状态，不产生冗余的高频日志轮询。
2. **终态检测与日志抓取**：
   - 当作业状态变为 `COMPLETED` 时，自动读取评测输出日志 `srun.out` 中的最终 mAP 指标；
   - 提取 `test_mAP@0.30`, `test_mAP@0.40`, `test_mAP@0.50`, `test_mAP@0.60`, `test_mAP@0.70` 及 `test_average_mAP`。
3. **结果回填与 Wiki 同步**：
   - 同步更新本文件 (`docs/experiments_registry.md`) 第 3 节的指标表格；
   - 在 `research-wiki/log.md` 中记录最终收敛结论与科学解释。
