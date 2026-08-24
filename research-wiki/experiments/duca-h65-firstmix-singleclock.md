# DUCA H65 First-Mixing SingleClock

## 状态

`experiment_running`。尚无正式 mAP、效率或论文结论。

## 科学问题

历史 H65 以低成本语义模型进行逐帧、非均匀的间接选帧，并将选中的 384 帧送入 VideoMAE-S 与 ActionFormer。该输入仍按选中次序组成 tubelet，第一层时序混合并不知道帧在原视频中的真实间距。本实验只检验一个问题：在不改变 H65 选帧集合、训练课程、检测器和评估器的条件下，向 VideoMAE 第 0 个注意力块加入有界的真实物理时间间隔，是否能改善稀疏输入的时序表示。

## 冻结方法

- H65 的语义间接非均匀逐帧选择保持不变，固定 `K=384`。
- 选中 RGB、24 个 16 帧 clip、VideoMAE-S、Adapter、ActionFormer、损失、Soft-NMS、THUMOS14 split 与官方评估器保持不变。
- 仅 VideoMAE 第 0 个自注意力块接收相对于规范均匀位置的成对物理时间残差。
- 残差尺度为共享标量 `tanh(theta)`；`theta` 以 FP32 的 0 初始化，学习率 `2e-4`，权重衰减 0。
- exact-uniform 输入严格退化为原路径；gate-zero 使用同一计算路径并把有效尺度置零。
- 反事实 teacher 保持 Clock OFF，不让新时间信号污染监督目标。

## 实现与验证

- 部署代码 revision：`08a817e91867839abf3a81e24f8469512b26a6ea`。
- 分支：`codex/duca-h65-firstmix-singleclock-20260824`。
- 独立 Critic：实现与可恢复训练合同均通过。
- GPU PRE_RUN：Jobs `1252471`、`1252480` 完成；非均匀时间产生有限非零梯度，uniform/gate-zero 恒等，输入 RGB 未改变。
- 恢复审计：epoch 0 检查点保存 model、EMA、optimizer、scheduler、AMP scaler、全局 RNG、DataLoader epoch-boundary contract 与 2 个成功更新；恢复后的 epoch 1 检查点累计为 4 个成功更新。

## 正式实验

- Slurm Job：`1252482`，N16R4，1 GPU / 8 CPU，seed `3407`。
- 输出：`/data/run01/sczc063/yuzibo/duca_h65_firstmix_singleclock_stage2_on_08a817e9_20260824`。
- Stage-1 初始化：历史 H65 uniform-384 epoch-29 EMA，SHA256 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`。
- 训练：完整 THUMOS14，60 epoch / 6000 次成功更新；每 5 epoch 保存可恢复 checkpoint；主结果预先固定为 epoch-59 final 与 final-EMA。
- 首个周期恢复点 `epoch_4.pth` 已在 Job `1252482` 运行中生成：`epoch=4`、累计 `500` 次成功更新，并同时保存 model、EMA、optimizer、scheduler、AMP scaler、Python/NumPy/Torch CPU/Torch CUDA 随机状态，以及指向 epoch 5 的 DataLoader/DistributedSampler 状态。该恢复点验证了正式训练的五轮保存合同，但不构成效能证据。
- 归因：不重训 OFF、dense、uniform 或 random。训练终态后，同一 ON checkpoint 运行 gate-zero twin，并只读引用同起点既有 OFF 对照。

## 终态证据实现

- 新增默认关闭的 selected-input identity 审计：逐窗口封存选中 RGB、完整 VideoMAE 输入张量、原始物理位置与有效掩码的 SHA256，并保留原始位置序列；只在终态评估显式启用，不改变训练或普通推理数值路径。
- 终态启动器固定执行同一个 epoch-59 checkpoint 的 final/final-EMA `CLOCK_ON` 与 `CLOCK_GATE_ZERO` 四次官方评估，并在同一评估代码 revision 上仅重推理既有 H65 OFF final/final-EMA，不重训任何基线。
- 配对统计实现固定为 10,000 次整视频 cluster bootstrap；每次以相同视频 multiplicity 重算官方 pooled AP，使用 `SHA256(nonce + "\n" + namespace)` 前 8 字节大端无符号种子、NumPy PCG64，以及排序后 1-based ranks 250/9750。
- 最新冻结门只以 `EMA CLOCK_ON - H65 OFF EMA` 的 Avg-mAP、mAP@0.6、mAP@0.7 三项 point delta 判定非劣，三项均须包含等号地不低于 `-0.20 pp`；配对 bootstrap 置信区间、ON-vs-gate-zero、旧 RankPack/TrueTime、训练恢复完整性和 Stage-1 成熟度均只作诊断。
- 边界风险分析使用 H65 OFF training population 的完整 tubelet 物理中心 gap-CV 与原始 GT boundary-density 冻结 q75；validation 采用官方重复 GT 去除、物理秒坐标、score-ranked IoU≥0.5 一对一匹配、漏检惩罚、先逐 GT 后逐视频等权汇总，并分别检查 high-gap-CV 与 high-boundary-density 的 `EMA ON - H65 OFF` error delta 是否 `<=0`。当前既有运行未预先封存 H65 OFF 窗口秒级 ledger，缺失时只能报告 `NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP`，不得伪称边界机制通过。
- 完整成本比较仍可固定为同一 epoch-59 EMA checkpoint 的 `CLOCK_ON/CLOCK_GATE_ZERO` 三次同节点顺序配对，但最新 Unit-1 合同将成本明确降为报告项，不得改变 Unit-1 PASS/KILL。
- 终态回执不仅校验同一 checkpoint、final/EMA state key 和选中 RGB/位置/掩码一致，还封存并重算 ON 与 gate-zero 的配置路径、配置哈希和门状态，防止把 ON 输出误标为 gate-zero twin。
- 本地 Python 编译与 33 项无数据 focused tests 通过；独立 Critic 对修正后的终结器、物理时间边界分析和身份审计返回 `UNIT1_GATE_IMPLEMENTATION_PASS`。本地含 PyTorch 的合同测试因 Windows `c10.dll` 初始化失败无法收集，必须在 N16R4 目标环境复核；该环境故障不是性能证据。
- 最新 Pro 终稿删除了旧正增益、旧配对证据、cost 与 coadaptation 的硬门。合法科学判决只有 `PASS_UNIT1_SINGLECLOCK_GATE` 与 `KILL_SINGLECLOCK_REPRESENTATION`；H65 replay 或 same-checkpoint 身份失败必须输出 `INVALID` 且无科学判决。

## 证据边界

Job `1252482` 的训练完成不等于 Unit-1 结论。六族终态评估与整视频 bootstrap 仍在运行；H65 replay 五边界身份和 canonical-uniform bit identity 尚未形成正式收据。现阶段不得声称 SingleClock 改善或损害 H65，也不得把 PRE_RUN、训练完成、恢复检查或证据工具实现解释为效能证据。
