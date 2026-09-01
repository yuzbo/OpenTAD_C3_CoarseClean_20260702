# DUCA V8 精确提交逐行审查与下一步优化裁决 Prompt

你现在不是普通顾问，而是同时具备以下身份的严厉审查者：CCF-A/CVPR 级
TAD 方法审稿人、PyTorch autograd/DDP/AMP 专家、OpenTAD/AdaTAD/ActionFormer
代码维护者、离散结构化选择与高效视频计算研究者。你的任务不是鼓励项目，
而是通过精确代码证据判断方法是否正确、为何仍未超过均匀采样，以及下一次
唯一值得实施的有界改进是什么。

## 1. 固定审计对象与可见性要求

- 仓库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 精确提交：`63e25eb17e523d369f73434ed4d9b6446608861a`
- 精确代码树：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/63e25eb17e523d369f73434ed4d9b6446608861a`
- 活动分支：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/duca-global-curriculum-20260721`

该分支已执行 fetch/push，本地与 GitHub ahead/behind 为 `0/0`。必须先实际读取
精确提交中的代码再下结论；如果无法读取，输出 `VISIBILITY_BLOCKED` 并停止，
不得凭本 Prompt 猜测实现。所有重要结论必须标记为 `[CODE_FACT]`、
`[EXPERIMENT_FACT]`、`[INFERENCE]` 或 `[PROPOSAL]`，并尽量给出 `文件:行号`。

## 2. 研究目标与不可误读边界

这是离线 TAD，不是 Online TAD。目标是在昂贵视频 backbone 之前放置低成本、
任务感知的时序采样插件：低分辨率粗分类器学习动作/背景证据，间接选择器利用
状态转变、边界和不确定性，在固定预算下选择稀疏帧/片段，再交给完整官方
AdaTAD/ActionFormerHead，以真实总成本下降为前提保护或提高高 IoU mAP。

主方法不是 actionness top-k，不是每个均匀 cell 固定一帧，不是新 TAD detector，
也不能宣称检测损失天然穿过离散 hard index。最终裁决指标是同协议 terminal-EMA
mAP，而不是训练损失、边界代理分数或代码门禁。

## 3. 当前 V8 合同

1. 使用现有 `global_structured_topk`，全时间轴 exact K=384，
   `max_unselected_hole=2`，允许跨区域预算转移和边界附近聚集。
2. 粗分类前端为低成本 RGB spatial stem + 官方 ASFormer temporal trunk；
   二分类 actionness 负责语义证据，transition scorer 负责选帧排序。
3. P0 为 20 epoch 的 train-only frontend 训练，完整 AdaTAD 对象可构建但 detector、
   backbone、neck、head 必须零调用。当前三项权重为 actionness/transition/
   transition-boundary=`1.0/0.10/16.0`，其余 16 项显式 graph-free zero。
4. P0 transition/boundary 输入从 coarse evidence `detach`，动作损失只更新 coarse
   参数，transition/boundary 只更新六参数 scorer。
5. 三组 P0 学习率只比较组件学习速度：control=`2.5e-5/5e-5/1e-4`，
   moderate coarse-first=`5e-5/1e-4/2.5e-5`，strong coarse-first=
   `1e-4/2e-4/5e-5`，顺序为 coarse trunk/action head/scorer。
6. P0 后冻结 coarse 分支。official-60 前 1000 个成功 optimizer updates 用
   exact-uniform 预热 detector；随后 1500 updates 把 policy alpha 从 uniform
   过渡到 learned；update 2500 后再用 1500 updates 把 protected detector
   feedback 提升到 0.25，且只更新 transition scorer。
7. 四个 matched 臂：U=exact-uniform；G0=学习全局策略但无 detector feedback；
   G1=G0+scorer-only protected feedback；G2=G1+仅训练期 50% uniform companion，
   并按 learned rows 归一化检测梯度；推理没有 companion。

## 4. 已知实验事实

V5 同提交四臂 epoch-59 EMA：U `64.4580`，direct-0.25 `63.7102`，
homotopy-0.25 `63.0601`，homotopy+companion `63.6931`。全部 learned 臂低于 U；
companion 相对 homotopy 恢复 `0.6330`，但仍低于 U `0.7649`。禁止把这些旧合同
换名重跑。

早期诊断中 coarse AUROC 约 `0.463--0.469`、AUPRC 约 `0.255`；后续 P0 诊断
coarse AUROC 提高到 `0.6161`、AUPRC `0.3750`。learned scorer 的 transition
AUROC 约 `0.527--0.530`，低于简单 `abs(delta p)` 的 `0.611--0.618`。学习策略
把相邻选择比例从均匀采样约 `4.4%` 提高到约 `35%`，说明它能聚集，但聚集位置
错误，radius-1 边界召回下降约 9--10 点。均匀采样 radius-1 recall 已约
`0.9998`，该代理指标饱和，不能代替 mAP。

当前 V8 Job `1178989` 的 real-CUDA P0 gate 已 `ok=true`：真实 THUMOS 两行 batch，
detector 零调用，冻结参数字节不变，68 个前端参数均有梯度，optimizer/scheduler/
EMA 更新一致，hard K=384、max-hole=2。2026-07-21 22:22 +08，第一组 control
候选已完成 epoch 8 并进入 epoch 9/20；最新总损失 `1.4665`、actionness BCE
`0.6855`、加权 transition loss `0.6396`、加权 transition-boundary loss
`0.1415`，无 Traceback/OOM/non-finite。尚无 P0 winner 或 V8 mAP。

历史强 uniform `65.696` 仅是协议不匹配背景锚点，不能填入当前 matched 主表；
当前目标仍是用同协议证据超过 65。physical-grid 因与 selected-axis detector
目标约 `24.1%` 差异而 HOLD；CellCF/local-cell、X3D 主线和 dynamic budget 均冻结。

## 5. 必须逐行审查的代码

- `opentad/models/duca/acquisition.py`
- `opentad/models/duca/transition_only.py`
- `opentad/models/duca/structured_selection.py`
- `opentad/models/duca/hard_soft_alignment.py`
- `configs/adatad/thumos/duca_frontend_pretrain_fixed384_base.py`
- 三个 `duca_frontend_pretrain_lr_*.py`
- `configs/adatad/thumos/duca_two_stage_joint_fixed384_official60_base.py`
- `duca_exact_uniform_fixed384_official60.py` 与三个
  `duca_global_curriculum_g{0,1,2}_*.py`
- `tools/bata/run_duca_frontend_p0_real_gate.py`
- `tools/bata/aggregate_duca_frontend_candidates.py`
- `tools/bata/select_duca_frontend_checkpoint.py`
- `tools/bata/duca_frontend_initialization.py`
- `scripts/run_duca_two_stage_curriculum_serial_gpu1.sh`
- `scripts/submit_duca_two_stage_curriculum_serial.sh`
- `tests/test_duca_frontend_p0_contract.py`
- `tests/test_duca_frontend_checkpoint_selection.py`
- `tests/test_duca_global_curriculum.py`
- `tests/test_duca_structured_selection.py`
- `tests/test_duca_detector_gradient_bridge.py`

历史配置很多，但不能把目录数量当成模型数量。审查必须确认当前 U/G0/G1/G2
究竟解析到哪些 base config、参数和运行路径，并指出任何被历史默认值静默覆盖的字段。

## 6. 必答审查问题

1. 画出实际 tensor/梯度图，列出每项 loss 最终能更新的精确参数集合；核验 P0
   是否真正隔离 detector，P1 coarse 是否始终冻结，G1/G2 detector feedback 是否
   只到 scorer，是否存在 detach、ST、mask、gather 或 companion 分支造成的断梯度、
   错梯度、重复缩放或梯度曝光不匹配。
2. 核验 hard path 与 surrogate 是否共享同一个 exact-K/max-hole 可行集；hard 选择、
   soft coverage、训练选点和推理选点是否同构；不得用 post-hoc repair 掩盖不一致。
3. 核验 selected-axis 输入 AdaTAD 后的 GT remap、mask、时间坐标、预测 inverse-map、
   高 IoU 边界含义是否正确。U 是否真的是官方 AdaTAD 的可信同协议均匀基线，
   64.458 与历史 65.696 的差异最可能来自哪里。
4. 审查粗分类器到底看到了什么输入与 hidden feature，transition scorer 是否真正
   使用 coarse hidden/delta/uncertainty，而非退化成概率曲线或 actionness top-k。
5. 按 raw loss、权重、梯度范数和组件 LR 审查 `1.0/0.10/16.0` 是否合理；解释
   actionness BCE 长期约 0.68 是欠拟合、类别平衡后的正常尺度、标签/掩码错误，
   还是优化器与特征容量问题。不能只凭 loss 大小下结论。
6. 审查 transition target、边界覆盖 target、padding/valid mask 和 train-only GT
   生成是否无泄漏且与研究初心一致。P0 winner 指标是否可能选择出代理分数好、
   终点 mAP 差的 checkpoint；提出更可信但不使用验证/test mAP 选模的规则。
7. 审查 1000/1500/1500 successful-update 日程、scheduler、EMA、AMP skip 恢复、
   optimizer state 和冻结/解冻是否严格匹配；判断 uniform warmup 是否足够，
   policy 切换是否仍造成 detector 输入分布突变。
8. 判断 protected structured transport 的优化目标是否真的与最终 mAP 相关，还是
   只优化一个不可靠 surrogate。明确它能证明什么、不能证明什么。
9. 计算并审查端到端成本口径：coarse probe、selector、稀疏 backbone、AdaTAD、
   数据解码和 companion 的训练/推理成本。不得只报 selector FLOPs。
10. 基于代码和现有负结果，给出“粗证据不足、scorer 学错、结构化 surrogate
    不对齐、坐标/remap 错误、curriculum 不稳、detector 本身未匹配”等原因的排序，
    每项必须附可证伪检查。

## 7. 下一步设计与训练优化要求

请大胆质疑 V8，但不要默认另建 selector、decoder、detector wrapper 或 worktree。
优先在现有 V8 中寻找最小而有理论意义的改动：粗证据训练、transition 表示、
scorer 校准、损失尺度、冻结策略、课程切换或 protected feedback。每个建议必须说明：
为什么能修复已观察到的错误聚集、改哪一文件/函数、核心代码或精确伪代码、梯度
归属、额外成本、所需测试、唯一变量实验和 KILL 条件。

最多推荐一个首选改进和两个备选。若你判断全局学习选帧在 K=384、G=2 下没有
足够 headroom，必须给出代码/实验依据并建议 KILL，而不是继续调参。只有明确证明
V8 的科学假设或可行集已失败后，才允许提出一个替代路线；必须先解释它与 V0--V7
有何本质不同，不能把 local-cell、physical-grid、X3D、dynamic budget 或旧 direct/
homotopy/companion 换名重做。

## 8. 强制输出格式

1. `VISIBILITY_CERTIFICATE`：实际读取的提交、文件与不可见范围。
2. `最终裁决`：`GO / HOLD / KILL`，先给结论，不说客套话。
3. `当前真实模型图`：前向数据流、hard/soft 路径、监督和反向梯度。
4. `逐行问题表`：严重级别、`文件:行号`、代码事实、影响、修复。
5. `梯度归属矩阵`：loss × coarse trunk/action head/scorer/AdaTAD。
6. `实验与数值解释`：先列原始结果，再解释为何 learned < uniform。
7. `根因排序与证伪检查`：至少覆盖上述六类原因。
8. `首选优化方案`：数学定义、核心实现代码、配置、测试和训练日程。
9. `两个备选及拒绝理由`。
10. `最小 matched 实验矩阵`：必须回答 U→G0→G1→G2，每个实验的停止条件、
    terminal-EMA 指标和超过 65 的判据；禁止无界网格搜索。
11. `论文主张边界`：现在能说什么、不能说什么，什么证据后才可 paper-ready。

不要输出泛泛的“调学习率、增加数据、尝试更多模型”。不要因为测试通过就声称
方法有效，也不要用代理指标代替最终 mAP。你的回复必须足以让工程人员直接修改
现有文件，并让审稿人能够复核每个科学主张。
