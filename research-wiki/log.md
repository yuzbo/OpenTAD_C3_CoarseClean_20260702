# Research Wiki Log

本文件只追加，不回写历史。

- 2026-07-11：初始化 research-wiki。
- 2026-07-11：清点 C3/PAction/GAS-VT、DUCA、MUST、X3D/SlowFast、PIVOT/ChronoTransport、PhysTime 的仓库文档、原始附件、提交历史和实验记录。
- 2026-07-11：建立当前方向、决策台账、时间线、经验禁区、gap map、idea catalog、experiment register 和 query pack。
- 2026-07-11：将 feature-token PhysTime 轨道标记为取消/诊断，将 PhysTime-AdaTAD K384 三头比较标记为当前唯一执行主线。
- 2026-07-11：明确秒坐标可转换回原视频帧号，但禁止 selected-rank GT/预测坐标。
- 2026-07-11：声明当前无 claim 实体；任何论文主张必须等待 matched full run 与 result-to-claim 审计。
- 2026-07-11：完成首轮 lint：31 个实体、10 个 gaps、48 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3348 字符。
- 2026-07-11：第二轮完整性审计发现并修正遗漏：ChronoTransport 在本地分支已实现到 `92029ea`，formal P3 science gate 为负且 15 commits 未推远端；DUCA 另有 `a5e1774` full-stack/structural audit 分支。
- 2026-07-11：新增 DUCA、ChronoTransport、PhysTime 三份完整路线档案、逐主题覆盖矩阵，以及 ResearchClaw 第二组 24 个候选 idea。
- 2026-07-11：迁入主任务用户侧完整导出与跨代理近期记录，固定 SHA256；新增 11-worktree 审计库存，防止单一 checkout 遗忘历史实现。
- 2026-07-11：第二轮 lint：36 个实体、10 个 gaps、55 条关系、0 孤立实体、0 失效关系、0 断链，query pack 3351 字符。
- 2026-07-11：PhysTime-AdaTAD 1.0 在 `549bb81` 完成 raw-video K384 三头 matched pipeline、原帧 same-index 审计、one-step 梯度证明、真实 CUDA gate 工具及 gate-dependent 启动器；远端 focused suite `45 passed`。状态为 `tested`，真实 THUMOS gate、正式训练与 mAP 仍 pending。
- 2026-07-11：首次 raw-video gate `1158528` 在 Python/模型执行前因非登录 shell 无 `module` 命令以 127 退出；依赖训练 `1158529/1158530/1158531` 未启动并取消。分类为 infrastructure failure；GPU launchers 改为可选 module 初始化并新增回归测试，等待新 commit 重跑。
- 2026-07-11：第二次 gate `1158546` 的 matched validator 通过，但 submission 覆盖 Slurm GPU mask，导致模型构建前 `CUDA is not available`；依赖训练 `1158547/1158548/1158549` 未启动并取消。launcher 已改为 Slurm 内保留调度器 mask，专项测试通过，等待新 commit 重跑。
- 2026-07-11：第三次 gate `1158556` 通过 CUDA、真实 THUMOS decode 与 same-frame checksum，但 imgaug 独立 RNG 导致增强后像素不一致；模型未构建，依赖训练 `1158557/1158558/1158559` 未启动并取消。gate 已统一 Python/NumPy/Torch/imgaug/OpenCV seed 并新增确定性测试。
- 2026-07-11：第四次 gate `1158576` 与逐 transform 诊断 `1158591` 将剩余分叉定位到首次 ImgAug 构造改变 ColorJitter 的 NumPy 状态；加入增强库预热后，真实诊断 `1158614` 证明三头 decode、crop、ImgAug、ColorJitter、FormatShape 像素 hash 全部一致。仍需重跑完整 detector gate。
- 2026-07-11：FP32 real gate `1158636` 完成三头真实 raw-video forward/backward/inference 并通过全部梯度/optimizer contract；formal selected-axis `1158637` 启动，physical-grid `1158638` 因 torchrun rendezvous broken pipe 失败，PhysTime `1158639` 因 endpoint probability BCE 不兼容 AMP 失败。已作 event-logit BCE 等价修复并将 gate 升级为 AMP，等待同 commit 重排完整三头。
- 2026-07-11：最终实验 commit `bd27544` 的真实 AMP gate `1158668` 通过；三头 same raw-frame/input、optimizer 与梯度合同全部满足。formal jobs `1158669/1158670/1158671` 已解除依赖并进入 epoch 0，状态提升为 `experiment_running`，mAP 与 claim 仍为 pending。
- 2026-07-11：formal PhysTime `1158671` 在 epoch 0 第 50 步出现全 NaN，定位为未覆盖 logits 在 support-measure masked attention 中先 `exp` 后乘零。`0bbf0e9` 改为 mask-before-exp，新增极值回归测试，远端 `68 passed`；新 AMP gate `1158718` 通过，matched jobs `1158719/1158720/1158721` 已越过第 50 步且 loss 有限，mAP 仍 pending。
- 2026-07-12：`0bbf0e9` matched jobs `1158719/1158720/1158721` 全部 FAILED。selected-axis/physical-grid 训练至 epoch 41 后首次验证因 evaluator GT annotation 相对路径不存在而退出；PhysTime 从 epoch 1 step 99 起持续全 NaN，并叠加相同验证路径错误。三头均无有效 mAP，实验状态改为 `experiment_failed`，禁止使用 checkpoint 填表。
- 2026-07-12：`52b5756` 修复 evaluator 路由与物理时间 FP32 数值路径；gate `1159481` 通过，但 stability gate `1159482` 在正式作业启动前 fail-closed。诊断作业 `1159489` 将问题定位到 epoch 0 iter 47 的 `rpn_head.cls_head.weight`：forward loss 有限，11 个 scaled gradient 为 Inf，无 NaN。
- 2026-07-12：最终 commit `3ac93a1` 将 AMP 初始 scale 设为 1024，限制可恢复 Inf 跳步，关闭单 GPU FP16 DDP compression，并保留 NaN、参数污染与跳步超限硬失败。远端 `102 passed`；gate `1159491` 与两 epoch stability gate `1159492` 均通过且零跳步。formal jobs `1159493/1159494/1159495` 正在运行，mAP pending。
- 2026-07-12：formal jobs `1159493/1159494/1159495` 均越过 epoch 1 step 50，loss 分别为 0.9929、1.0115、1.1880，全部有限；这只提升训练稳定性证据，mAP 与方法 claim 仍 pending。
- 2026-07-12：`3ac93a1` 三头正式训练全部完成；最佳 checkpoint 复算 `1159819/1159820/1159821` 逐项复现官方结果。PhysTime 1.0 未胜两个 sparse controls，状态改为“负结果已验证”，不是 paper-ready。
- 2026-07-12：完成性能下降诊断：排除训练崩溃、evaluator、重复坐标换算与缺失 test window；确认比较存在容量/上下文混杂，并发现 absolute-second query 主导、粗层 attention 坍缩、候选密度和短动作监督不足、单标签 assignment 差异。完整数字只写入 `docs/evaluation/results.md`。
- 2026-07-12：独立 GPT-5.5 xhigh 完整性审计确认 real GT 与 raw mAP 路径有效；发现本地 registry 曾滞后于远端完成状态。结果表与 Wiki 已整改，剩余风险为单数据集单种子和非等容量比较。
