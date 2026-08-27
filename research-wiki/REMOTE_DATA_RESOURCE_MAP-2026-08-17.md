---
type: resource_map
scope: DUCA official baseline and prospective dynamic-budget experiments
status: read_only_inventory
updated: 2026-08-17
---

# DUCA 远端数据资源地图（2026-08-17）

## 1. 适用范围与证据边界

本页只记录本项目会用到的远端基础设施与数据资源事实；来源是中央在 2026-08-17 完成的只读核验，不是本项目新运行、性能、成本或论文证据。当前科学路线仍为：训练 scout 的 0/1 actionness 与 boundary-importance 预测，经确定性 acquisition 导出物理选帧与 dynamic outer-K；固定 K 只作公平对照、归因或回退。原始 official AdaTAD-S 的 THUMOS14 released-checkpoint evaluation 是 ZoomToken 负责人唯一一次共享基线工作：DUCA 不得重复 evaluation 或原始训练，只在 receipt 返回后只读绑定 official dense 数字；等待它不阻塞 DUCA 的方法实现、审查与 PRE_RUN 准备。

禁止复制共享数据、把非规范视频混入 split、把 feature identity 替换为另一项目资产，或以本页为由下载、解压、建链接、提交 Slurm、训练或评估。

## 2. 正确 N16R4 访问与共享根

- 共享根：`/data/run01/sczc063/yuzibo`；存储：`/data` 5.3T total、3.1T free。
- PowerShell 登录：`ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i C:\Users\skywalker\.ssh\id_rsa -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com`
- Git Bash 登录：`ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i /c/Users/skywalker/.ssh/id_rsa -p 22 -l 'sczc063@BSCC-N16R4' ssh.cn-zhongwei-1.paracloud.com`
- `yuzibo` 是共享目录名，不是可用 SSH host alias。此前的 alias 解析失败不能再被记作数据缺失。

## 3. DUCA 官方 AdaTAD-S / THUMOS14 规范绑定

| 资源 | 精确远端路径与状态 | 规范使用 |
|---|---|---|
| Canonical videos | `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`：**411** 个有效 MP4 symlink，0 broken，target 约 33G；`training=200`、`validation=211` | 这是 DUCA official AdaTAD baseline 及后续冻结六臂唯一视频入口；只允许与 annotation video-ID mapping 的 canonical 411 一起使用。 |
| Physical video store | `/data/run01/sczc063/yuzibo/raw/Validation Data/validation` 有 200；`/data/run01/sczc063/yuzibo/raw/Test Data/TH14_test_set_mp4` 有 213；合计 413 | 物理源只解释 canonical symlink 的来源。`video_test_0000270.mp4`、`video_test_0001292.mp4` 不在 canonical 411 mapping，永不擅自加入。 |
| Annotation | `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`：**COMPLETE** | 固定 official split、视频 ID 与 evaluator ground truth；不得改写或从 validation/test 取 selector 决策输入。 |
| Category map | `/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt`：**COMPLETE** | 与 annotation、官方 AdaTAD config 共同绑定；不得用其他项目的 class ordering 替换。 |
| VideoMAE-S pretrain | `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`：**COMPLETE** | 仅用于 official AdaTAD-S config 的既定 backbone pretrain binding；不是 DUCA 方法的额外证据。 |
| Official evaluator | OpenTAD `mAP`，同一 annotation，tIoU `0.3/0.4/0.5/0.6/0.7`；train=`training`，evaluation=`validation` | 所有 official dense / fixed-K / indirect / dynamic-K 比较必须共享此 evaluator、split、detector、loss、NMS、updates 与 seed contract。 |

## 4. checkpoint、特征与缺失资源

| 资源 | 精确路径/观察 | 状态与阻塞 |
|---|---|---|
| Released AdaTAD-S THUMOS checkpoint | 官方 release link 已知；本次只读 inventory **未给出已存在、完整、可读且 config-compatible 的远端文件路径** | **UNVERIFIED（共享基线输入）**。由 ZoomToken 负责人在唯一共享 official-baseline 流程中定位、评测并写 receipt；DUCA 不得自行探测、评测或训练。它只使 DUCA official-dense 数字暂缺，不阻塞方法实现、static review 或 PRE_RUN 准备。 |
| OpenTAD I3D | `$BASE/thumos14/features/i3d_actionformer_stride4_thumos` absent；`$BASE/datasets/phystime_thumos_i3d/features/i3d_actionformer_stride4_thumos` 为空，仅约 746MB 未完成 `.part` | **MISSING**，阻塞任何声称使用 OpenTAD I3D feature 的完整正式实验；不阻塞 raw-video AdaTAD-S baseline。 |
| InternVideo2 | `$BASE/thumos14/features/thumos14_6b` absent；仅 HuggingFace README directory | **MISSING**，阻塞任何 InternVideo2 feature config；不得代入 AdaTAD-S raw-video recipe。 |
| Native MATR features/checkpoint | `$BASE/external_ontal_baselines/MATR_codebase/data/thumos_all_feature_val_V3.pickle`（约 3.33G）和 `...test_V3.pickle`（约 3.69G）存在；native checkpoint dir 为空，仅 `$BASE/eventmatr_data/official_google_drive_folder/best_epoch.pth9fa8xl1m.part`（约 9MB incomplete） | feature 为 **PRESENT but incompatible**；checkpoint **MISSING**。两者只属于 native MATR identity，不能作为 OpenTAD/AdaTAD/DUCA checkpoint 或 feature substitute。 |
| SigLIP2 | `$BASE/thumos14/features/pes_siglip2_stride8`：823 files、约 477MB | **PRESENT but incompatible**；不得替换 I3D、InternVideo2 或 native MATR identity。 |
| Derived DUCA cache/manifest | 当前没有经 PRE_RUN 承认的 DUCA-specific derived cache、track、proposal 或 manifest | **NOT YET AUTHORED**；不阻塞 official released-checkpoint evaluation，阻塞未来 dynamic DUCA full experiment，直到 Builder 在冻结协议下创建最小项目局部 binding 并经 Critic/Evaluator PRE_RUN。 |

## 5. 其他可见视频数据的隔离边界

| 数据集 | 只读状态 | 对 DUCA 当前实验的结论 |
|---|---|---|
| MultiSports | `projects/stad-paper/data/r0b02/archives/{aerobic_gymnastics,basketball,football,volleyball}.tar` 合计 43,820,810,240 B、内含 2129 videos；仅 18 MP4 extracted；annotation/proposal archive 可见 | archived / partial，不能替代完整 DUCA 数据集；不解压、不复制。 |
| TOC-Bench | `tstep_v0_phase0/datasets/toc_bench_full/videos`：1951 MP4、15,235,916,868 B、无 zero-size | raw tree complete，但无 DUCA frozen split/evaluator，不进入当前实验。 |
| Charades | `datasets/charades/raw_data/Charades_v1_480`：9848 MP4、16,588,858,990 B；原 zip 也存在 | raw tree complete，但无当前 TAD protocol，不进入当前实验。 |
| ActivityNet | 仅 3 readable MP4（约 71.7MB）；v1-2/v1-3 archives/missing-files zips 未组装验证 | partial / unusable，不得视为可训练 video tree。 |
| FineAction、HACS、EPIC-Kitchens、Ego4D | 无实际 video dataset；仅 scripts/readmes/small source archives，FineAction probe 空 | missing，不构成当前或后续自动实验资源。 |

## 6. 周期性 checkpoint 与恢复合同

- 每个 future full training 至少每 **5 epochs** 写一次可恢复 `.pth`；若完全未改的 official recipe 更频繁，保留更频繁的官方间隔。AdaTAD-S official recipe 的 `checkpoint_interval=2` 已满足并优先于 5-epoch 下限。
- recovery package 至少恢复：model、optimizer、scheduler、mixed-precision scaler state、epoch/update counter、Python/NumPy/PyTorch/CUDA random states。它只支持中断恢复与诊断，**不改变**预登记的 final/final-EMA 选择规则，也不得事后挑选中间 checkpoint。
- 保留最近至少 3 个有效 recovery checkpoints 及预定义 milestone/final；在 future PRE_RUN packet 写明 interval、retention、resume argv、output root 与存储预算。
- 官方现有 checkpoint writer 未证实包含 scaler/RNG state；任何 DUCA full-run 若要声称可恢复，Builder 必须先完成最小 project-local recovery binding，随后由 Critic/Evaluator 核验。此要求当前不授权修改。

## 7. 给注册角色的职责限定通知

- **Builder（writable）**：在 clean DUCA worktree 实现 scout 的 0/1 actionness 与 boundary-importance 输出、确定性 indirect acquisition、dynamic outer-K，以及六臂 config/launcher/recovery contract；绑定 §3 canonical root/annotation/map/pretrain，排除 2 个 extra videos 与所有异构 feature。不得运行或复制共享 official baseline，也不得自行定位其 released checkpoint。
- **Critic（read-only）**：审查 semantic scout 是否没有退化为 direct index policy，dynamic K 是否由语义证据确定性导出；并检查 411 canonical videos、official split/evaluator、recovery fields、无异构 feature/extra-video/validation leakage。共享 dense 仅可引用 ZoomToken receipt。
- **Evaluator（evaluation-only）**：在 Builder 与 Critic 都通过后进行 PRE_RUN，验证 canonical path/count/split/map/evaluator、future DUCA checkpoint interval/retention/resume fields 与 no-leak boundary；不得执行共享 checkpoint evaluation 或放行 held-out 评测。

三份同内容、按职责裁剪的 durable mailbox 通知位于 `.cvpr-pro-lab/messages/{builder,critic,evaluator}/`；本轮没有唤醒角色、提交代码或发起实验。

## 8. 当前状态与下一棒

- **current_scientific_question**：0/1 actionness + boundary-importance 的 semantic scout，能否经确定性 indirect acquisition 与 dynamic outer-K，在 matched full-stack cost 下保护高-IoU；fixed K 只作 baseline/control/fallback。
- **next_owner**：Coordinator terminal hold；如要继续，只能请求 fresh exact-Project 的 STOP vs genuinely simplified replacement 裁决。
- **next_action**：不再修改当前 semantic package。结构 intake 对 GT 路径的初报已由冻结代码直接复读更正：`forward_train` 已把 `gt_segments` 交给 selector 与 semantic loss；仍未闭合的是动态 K 把整批字典当作可逐样本索引的回执，无法得到每视频 requested/effective/executed K。一次 correction/recheck 已耗尽，当前包不可 PRE_RUN。
- **dependency**：若继续方法路线，需要一项新的外部科学处置；共享 official baseline receipt 仍只在正式矩阵填入 dense official 数字时绑定。
- **expected_return_at**：外部处置或明确 STOP；无新代码、训练或数据操作。
- **single_recovery**：none；共享基线不在 DUCA 重跑。
