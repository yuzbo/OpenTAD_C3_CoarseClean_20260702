# 实验全景追踪登记册 (Experiments Registry & Live Tracker)

本文档记录 **连续物理时空自适应与双相动作检测体系（Continuous-Time Scale-Adaptive & Dual-Phase Action Detection, CT-DP-TAD / DUCA-BAMoD）** 的所有实验矩阵、本地与远端部署状态、Slurm 作业句柄以及结果更新协议。

---

## 1. 代码库与部署环境元数据

| 项目属性 | 配置与路径信息 |
| :--- | :--- |
| **本地代码仓库根目录** | `E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702` |
| **GitHub 远程仓库 URL** | [OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git) |
| **当前开发分支** | `codex/duca-total60-plugin-cvpr-20260727` |
| **远端计算集群** | `N16R4` (`sczc063@...`) |
| **远端独立工作区** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901` |
| **远端 Python/PyTorch 环境** | `/data/run01/sczc063/yuzibo/conda_envs/opentad` (Python 3.10, PyTorch 2.0.1 + CUDA 11.8) |
| **THUMOS-14 数据集软链** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/data/thumos-14` |
| **预训练 VideoMAE 权重软链** | `/data/run01/sczc063/yuzibo/projects/opentad_duca_ct_dp_bamod_d9bdb3f_20260901/pretrained` |

---

## 2. 单种子完备 2×2 正交消融矩阵 (Single-Seed 3407 Complete Factorial Matrix)

为了严谨正交归因 **连续物理时间调制卷积 (Continuous-Time Conv1d)** 与 **边界调制自适应 Token 稀疏化 (B-AMoD)** 的独立主效应与二阶交互项（$\Delta_{\text{interaction}}$），建立以下 4 臂全训练消融对照：

| 实验编号 (Arm) | 实验名称与科学机制 | 本地/远端配置文件路径 | CT-Conv1d | B-AMoD | CT-Tubelet | Physical-Grid | Slurm Job ID | 部署状态 | 日志与产物路径 |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Arm 1 (完整主方法)** | **Full CT-DP-BAMoD**<br>• 双相选帧 ($K=384, K_s=128, K_b=256$)<br>• B-AMoD 稀疏化 (6D+6M@0.5)<br>• CT-Tubelet 3D 速度归一化<br>• CT-Conv1d 连续时间卷积 | [`configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py) | ✅ | ✅ | ✅ | ✅ | **`1264438`** | `PENDING/RUNNING` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_bamod_s3407_1264438.out`<br>**输出**: `exps/thumos/adatad/duca_ct_dual_phase_bamod_seed3407/gpu1_id0/` |
| **Arm 2 (消融 CT-Conv)** | **DP-BAMoD (Standard Conv1d)**<br>• 双相选帧 + B-AMoD 稀疏化<br>• **消融项**：回退为标准均匀 Conv1d | [`configs/adatad/thumos/duca_dual_phase_bamod_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_dual_phase_bamod_thumos.py) | ❌ | ✅ | ✅ | ✅ | **`1264439`** | `PENDING/RUNNING` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/dp_bamod_stdconv_s3407_1264439.out`<br>**输出**: `exps/thumos/adatad/duca_dual_phase_bamod_stdconv_seed3407/gpu1_id0/` |
| **Arm 3 (消融 B-AMoD)** | **CT-DP (Dense ViT-Adapter)**<br>• 双相选帧 + CT-Conv1d<br>• **消融项**：ViT-Adapter 全 12 层稠密计算 | [`configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py) | ✅ | ❌ | ✅ | ✅ | **`1264440`** | `PENDING/RUNNING` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/ct_dp_densevit_s3407_1264440.out`<br>**输出**: `exps/thumos/adatad/duca_ct_dual_phase_densevit_seed3407/gpu1_id0/` |
| **Arm 4 (双消融基线对照)** | **DP-DenseViT (Standard Conv1d)**<br>• 双相选帧 + Dense ViT + Standard Conv1d<br>• **消融项**：同时关闭 CT-Conv 与 B-AMoD | [`configs/adatad/thumos/duca_dual_phase_densevit_stdconv_thumos.py`](file:///E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/configs/adatad/thumos/duca_dual_phase_densevit_stdconv_thumos.py) | ❌ | ❌ | ✅ | ✅ | **`1264441`** | `PENDING/RUNNING` | **日志**: `/data/run01/sczc063/yuzibo/slurm_logs/dp_densevit_stdconv_s3407_1264441.out`<br>**输出**: `exps/thumos/adatad/duca_dual_phase_densevit_stdconv_seed3407/gpu1_id0/` |
| **Baseline (历史基准)** | **Dense 768 / Uniform 384 Baseline**<br>• 稠密 768 帧输入 / 均匀 384 帧降采样基线<br>• 仅作指标基准对比，**严禁重新提交** | 历史已冻结检查点与测试记录 | N/A | N/A | N/A | N/A | N/A | `COMPLETED` | 历史评测基准数据已锁定 |

---

## 3. 实验评测指标登记表 (待训练完成后更新)

所有实验均在 THUMOS-14 官方 211 验证集上使用官方评估协议进行评测（mAP@0.3 ~ 0.7 及 Avg-mAP）：

| 实验组别 | 选帧预算 $K$ | 骨干等效结构 | mAP@0.3 | mAP@0.4 | mAP@0.5 | mAP@0.6 | mAP@0.7 | Avg-mAP (0.3:0.7) | 相对基线收益 (Avg-mAP) | 相对算力 / 显存 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dense 768 Baseline** | 768 | 12 层稠密 | - | - | - | - | - | 历史基准 | 基准 (0.0) | 100% |
| **Uniform 384 Baseline** | 384 | 12 层稠密 | - | - | - | - | - | 历史基准 | - | ~50% |
| **Arm 1: Full CT-DP-BAMoD** | 384 (128+256) | 6 Dense + 6 Sparse@0.5 | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | **~35% / 3.78GB** |
| **Arm 2: DP-BAMoD (StdConv)** | 384 (128+256) | 6 Dense + 6 Sparse@0.5 | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | **~35% / 3.78GB** |
| **Arm 3: CT-DP (Dense ViT)** | 384 (128+256) | 12 层稠密 | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | ~50% / 2.52GB |
| **Arm 4: DP-DenseViT (StdConv)** | 384 (128+256) | 12 层稠密 | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | *待回填* | ~50% / 2.52GB |

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
