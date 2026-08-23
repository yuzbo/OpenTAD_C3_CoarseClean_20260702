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

- 新增默认关闭的 selected-input identity 审计：逐窗口封存选中 RGB、原始物理位置与有效掩码的 SHA256，并保留原始位置序列；只在终态评估显式启用，不改变训练或普通推理数值路径。
- 终态启动器固定执行同一个 epoch-59 checkpoint 的 final/final-EMA `CLOCK_ON` 与 `CLOCK_GATE_ZERO` 四次官方评估，并在同一评估代码 revision 上仅重推理既有 H65 OFF final/final-EMA，不重训任何基线。
- 配对统计实现固定为 10,000 次整视频 cluster bootstrap；每次以相同视频 multiplicity 重算官方 pooled AP，使用 `SHA256(nonce + "\n" + namespace)` 前 8 字节大端无符号种子、NumPy PCG64，以及排序后 1-based ranks 250/9750。
- 分层分析只用 training population 冻结短动作阈值与时间畸变量分位点；validation 只按冻结阈值报告短动作差值和高/低畸变交互，不使用 validation 标签调整阈值。
- 完整成本比较固定为同一 epoch-59 EMA checkpoint 的 `CLOCK_ON/CLOCK_GATE_ZERO` 三次同节点顺序配对，顺序为 ON-ZERO、ZERO-ON、ON-ZERO；每次先完成 50 个预热样本，再覆盖完整官方 validation workload。主统计为三次完整运行中位延迟比、三次逐窗口 p90 的中位比和全窗口峰值显存比。
- 终态回执不仅校验同一 checkpoint、final/EMA state key 和选中 RGB/位置/掩码一致，还封存并重算 ON 与 gate-zero 的配置路径、配置哈希和门状态，防止把 ON 输出误标为 gate-zero twin。
- 本地 shell 语法、Python 编译与 30 项纯合同测试通过；这些仍是尚待 N16R4 目标环境复核和终态执行的证据工具，不是实验结果。涉及真实模型加载、GPU 评估和完整 workload 的接纳必须以 N16R4 为准。
- 最新 Pro 终稿在 Job 已启动后补充了“旧 RankPack/TrueTime bootstrap 与 H65 OFF baseline maturity 必须先通过”的时序条件。当前不取消已授权且正常运行的训练，但其未来指标只能在这些前置门事后完整通过时接纳；否则整次运行保持条件性不可采纳，不能用于论文结论。

## 证据边界

Job `1252482` 当前只证明正式训练已经越过预检、完成前五个 epoch，并形成合格恢复点。旧证据 bootstrap、H65 OFF baseline eligibility、终态双读出、短动作/畸变分层与完整成本仍未闭合。训练完成前不得声称 SingleClock 改善或损害 H65，也不得把 PRE_RUN、早期 loss、恢复检查或证据工具实现解释为效能证据。
