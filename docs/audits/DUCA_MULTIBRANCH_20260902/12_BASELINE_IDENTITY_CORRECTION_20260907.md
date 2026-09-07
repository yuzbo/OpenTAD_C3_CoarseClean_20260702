# AdaTAD 全帧与均匀半帧基线口径纠正

核验日期：2026-09-07。此次核验涉及公开结果、配置及既有本地实验记录；没有重跑训练，也没有改写任何 checkpoint 或原始指标收据。

2026-09-08 补证：已找到并读取 job1245842 的原始 stdout/有效配置/epoch59 EMA，日志终轮为68.73%，当前官方01c58b9f源码与记录配置只差work_dir；详见 [15号修复与基线报告](15_PRO_AUDIT_REPAIR_STATUS_20260908.md) 和 [14号原始产物核验](14_OFFICIAL_BASELINE_RAW_IDENTITY_20260908.json)。下表保留9月7日当时的证据范围，不代表最新仍未找到原始产物。另已确认REF-D768的TIA时间聚合边界不同于官方，不能只称训练配方变化。历史clean、预训练/数据来源及独立评测认证仍须补齐，不能从当前clean状态推定全部历史事实。

## 结论

用户指出“全768帧约69，均匀384帧约65”有来源依据。之前只回答本轮的 67.58%/63.89%，没有区分公开基线、历史基线与改动训练协议后的参考臂，表述不完整。

收据完整只能支持“这次运行产生了该数值且可追溯”，不能自动证明“官方 AdaTAD 已复现”“代码无误”或“方法改善”。也不能反过来把 69/65 作为必须通过调整结果达到的验收分数。

## 数值身份

| 名称 | THUMOS14 Avg-mAP | 来源与可使用范围 |
|---|---:|---|
| 上游 AdaTAD 全768帧，VideoMAE-S，160分辨率 | 69.03% | [上游结果表](https://github.com/sming256/OpenTAD/blob/main/configs/adatad/README.md#thumos-14-results)，表中为2 GPU；是公开性能锚点 |
| 项目共享的未修改 AdaTAD 复现 | 68.73% | 项目文档登记 job 1245842、revision 01c58b9f2370e914150cf94d392208a4e211c053、seed42、60轮；本次未重新读取该旧作业的原始 checkpoint/日志，作为有出处的历史共享参考，不冒称本轮新核验终态结果 |
| 历史均匀384帧，原生 stride-2 | 64.352% | 项目 source registry 登记 job 1150701 的 best Avg-mAP；非本轮 terminal-EMA 配对结果 |
| 历史均匀384帧，physical-grid 检测坐标 | 65.696% | 项目 source registry 登记 job 1150842 的 best Avg-mAP；改变了时间坐标处理，不得冒充原生均匀对照 |
| 本次修改协议的全768帧参考臂 | 67.58% | REF-D768，训练 e553a5a4，seed3407，严格6000 successful updates，epoch59 EMA |
| 本次修改协议的均匀384帧控制臂 | 63.89% | REF-U384，同一训练提交和种子，但继承 DUCA selected-axis 后端，不是原生 AdaTAD 直接 stride-2 的同义配置 |

历史两种均匀384的记录明确标注 protocol-unmatched。不能择取其中较高或较低的数字，直接充当新方法的匹配对照。历史 H65 30+60 的 65.13% 是选帧方法结果，也不是均匀384。

## 已确认的配置差异

1. 学习率曲线改变。上游配置 `scheduler.max_epoch=100`，同时 `workflow.end_epoch=60`；本次 REF-D768 覆盖为 `scheduler.max_epoch=60`。两者都训练60轮，但余弦退火轨迹不同。不能误写成“上游训练100轮，本轮训练60轮”。
2. batch 与训练曝光口径改变。上游命令使用2个进程，配置每卡 batch=2；本次 H65 启动脚本明确 `WORLD_SIZE=1`、1 GPU，本次配置 batch=2。固定6000次更新不自动等于相同样本曝光或上游按完整 DataLoader 的60轮训练。
3. checkpoint 规则不能默认相同。上游每2轮验证，从第40轮开始；本次禁用中间验证，固定 epoch59 EMA。仅凭上游配置不能认定其公布分数选用了哪个 checkpoint，需看原始日志/模型。
4. 均匀384不是只改帧数。REF-U384 继承 `base_h65_pro_strict60.py` 及 DUCA sampling 配置，含 selected-axis、全窗口 exact-uniform 选择合同和 relative physical-time 路径。配置存在不代表每条路径在均匀输入下都有数值影响，仍需实际索引、位置与特征 parity 证明。

以上是确定的配方差异，不是已经隔离出的性能下降原因。随机种子、训练曝光、学习率与坐标因素尚不能仅凭两组分数分摊因果贡献。

## 科学结论与后续动作

- 保留 67.58%/63.89% 原始结果，只将其标为修改协议下的实测参考；不能替代69.03%/68.73%的公开/共享全帧锚点。
- H65-Pro F02 的64.2265%相对本轮63.89%只有约0.34个百分点的算术差，不能据此声称超过约65%的既有均匀基线，或据单种子确认有效提升。
- 先只读核对共享68.73%运行的原配置、权重、实际batch/曝光、数据划分、终态模型规则和后处理；不重复训练既有共享未修改模型。
- 对均匀384复核实际索引和时间坐标，并比较各参考臂与方法臂的完整 resolved config。只修复已定位的问题，不能为追到69/65而看测试集改超参数或挑checkpoint。
- CT-DP 的成功更新计数缺陷及 DUCA-Unified 的缺失机制是独立的未完成整改项，不因本次基线表述纠正而变成已修复。
- BAFDR 的完整收据也不能排除模型实现错误；它是多分辨率分块路线，不能直接用 H65 的帧预算和上述数值判定其正确性。

## 本地核验出处

- 共享复现记录：`E:/DeskTop/TAD/OpenTAD_DUCA_Unified_FormalGates_20260903/docs/pro-packets/DUCA_ROUTE_CODE_ORGANIZATION_UPLOAD-v001/04_PROJECT_RULES.md:56`。
- 历史均匀记录：`E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903/research-wiki/source_registry.md:485`。
- 本次全帧配置：`E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission/configs/adatad/thumos/h65_pro/h65_pro_ref_d768.py:54`。
- 本次均匀配置：`E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission/configs/adatad/thumos/h65_pro/h65_pro_ref_u384.py:1`。
- 本次单GPU启动：`E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission/tools/experiments/run_h65_pro_train.sbatch:41`。
- 对比源码：上游 [VideoMAE-S AdaTAD配置](https://github.com/sming256/OpenTAD/blob/main/configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py)；本地通过 `git show e553a5a4a1063a755900d3dfa4bf8909bf97d466:configs/adatad/thumos/h65_pro/base_h65_pro_strict60.py` 回看参考臂的冻结配置，避免把后续 F 臂 optimizer 修复误认为原始参考臂已有。
