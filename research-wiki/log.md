# 2026-07-13 PhysTime G1a d1747d6 deployment gate

- Independent Max reviewer returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE` after the two P1 fixes: per-step optimizer-state parameter-name hash validation, and cross-arm recomputation of parameter/initial-state/optimizer-schema matches from `variants`.
- Commit `d1747d6657e185495b4db9eb491fd135d4b90360` was pushed to `codex/phystime-performance-diagnosis-20260712`; clean remote snapshot is `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_d1747d6_20260713_gate`, tree `2651bd30eda5b0e0960518da4060ccfc628b7a58`.
- Formal Slurm queue was submitted: `1161476` real gate, `1161477` selected-axis 6-epoch pilot afterok, `1161478` physical-metric 6-epoch pilot afterok. Current status at submission check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. This is `queued_for_gate`, not yet `experiment_running`, and there is still no new mAP.
- Gate `1161476` then failed before pilot start because the gate incorrectly required every production train sample to have `mask.sum()==384`. Static contract/G0 passed, but real AdaTAD training windows can be shorter/padded; K=384 is the decoded slot count, not a guarantee of 384 valid raw observations per sample. The fix keeps `decoded_frame_count==384`, records `production_train_raw_valid_counts`, and requires each valid count to satisfy `0 < count <= 384` with min/max consistency. Focused remote regression after the fix: gate contract `30 passed`, PhysTime/C3 physical-grid `243 passed`. The fix is under renewed Max review before any requeue.
- Renewed Max review found two more P1 issues before requeue: the gate checked `inputs.shape[2]` instead of six-dimensional time axis `shape[-3]`, and assignment valid-point validation still assumed `batch_size*378`. Both are fixed: `_decoded_temporal_length` now validates `[B,N,C,T,H,W]`, `assignment_valid_point_per_sample` is recorded by the head, and validator checks per-sample valid candidate ranges/sums. Remote regression after the final fix: gate contract `34 passed`, PhysTime/C3 physical-grid `247 passed`; Max returned `GREEN_FOR_CLEAN_SNAPSHOT_AND_REAL_GATE`, P0/P1 none.
- Commit `56c7e98e54ba83eb32b84dbdbeb74c3b5698eca2` was pushed and deployed from clean snapshot `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1a_56c7e98_20260713_gate`, tree `d698d451edc165ff4ac6179181157646262002a9`. New run root: `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1a_56c7e98_gatepilot_20260713_194728_+0800`. Jobs: `1161486` real gate, `1161487` selected-axis pilot afterok, `1161488` physical-metric pilot afterok. First queue check: gate `PENDING (Priority)`, pilots `PENDING (Dependency)`. Status remains `queued_for_gate`; no gate pass and no mAP yet.

# Research Wiki Log

- 2026-07-13：第三轮 Max 审查等待期间自查发现 manifest 中的 VideoMAE pretrained checkpoint 与 pilot `epoch_5.pth` 被测试夹具错误合并；已拆分二者并删除错误路径相等约束，completion 仍独立验证 epoch checkpoint 的 EMA/optimizer/scheduler。focused tests 保持 `65 passed`，必须以最新 diff 重新复审。
- 2026-07-13：独立 Max code review 第二轮发现 assignment 伪计数、optimizer state 覆盖、DataLoader `drop_last`、pilot artifact 传递信任等 4 个 P1，并补充固定参数集合、GPU batch 生命周期和显式 seed 风险。现已按测试先行修复；远端 gate/artifact `65 passed`，PhysTime/shared physical-grid `240 passed`。第三轮复审前禁止部署，状态保持 `tested`。

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
- 2026-07-12：形成基于 GitHub 分支 `codex/phystime-performance-diagnosis-20260712`、正式实现 `3ac93a1` 与诊断提交 `d900c7c` 的 Pro 严厉审核 prompt；要求逐文件裁决根因，并交付等容量/同上下文/同候选数的最终模型、核心代码和因果实验 gate，禁止回退 selector 或用调参掩盖结构混杂。
- 2026-07-13：逐字归档 1539 行 PhysTime 性能 Pro 回复，附件与仓库归档 SHA256 均为 `651C4CA673073D7E4C05746138C82EBBE2E6174C459516FB40B3EFDCA47305AB`；审查裁决为 `HOLD AND REBUILD`。
- 2026-07-13：吸收 native tubelet feature-support provenance、capacity/context/candidate/assignment parity、gap-query 与训练态 mass-path 等新约束；新增 `idea:sm-ptaf`，严格标记为 `designed`。PhysTime 1.0 继续冻结为负基线，下一步先做 provenance 与 coordinate-only P0 gates，不创建虚假的实验或结果状态。
- 2026-07-13：重建 Wiki index/query pack/lint；共 38 个实体、10 个 gaps、66 条关系，0 孤立、0 失效引用、0 重复节点/边，query pack 4430 字符。
- 2026-07-13：独立复核远端最终作业与最佳 checkpoint 复算，`1159491..1159495`、`1159819..1159821` 均为 `COMPLETED 0:0`；正式快照关键合同测试 `69 passed`。确认结果可信、PhysTime 1.0 失败、physical-time 假设未被裁决。
- 2026-07-13：分级接受 Pro 审查而非照单全收：锁定 `HOLD AND REBUILD`，但 SM-PTAF 保持 `designed`。新增 tubelet 跨 gap 非线性融合风险，并把下一步拆成 G0 provenance、G1a `Q=J` temporal-metric、G1b 双侧共享 Q384 中性 lift、G2 mass residual。
- 2026-07-13：登记 RCL 为连续锚 TAD 近邻，进一步限制新颖性主张；建立 `source_registry.md` 记录本轮原始审查、正式结果、远端作业、代码和文献来源。
- 2026-07-13：独立核验后的 Wiki 完整性检查：39 个实体、10 个 gaps、67 条关系，0 孤立实体、0 失效引用、0 重复关系，query pack 4684 字符。
- 2026-07-13：实现 PhysTime G1a native-J192 matched control：分离 K384/J192/Q0=192/QΣ=378，补齐全部 patch 输入槽与 padding-repeat provenance、显式秒域起止边界、官方 ActionFormer 梯度/完整后处理 gate，以及 static-contract→G0→real-gate→pilot 哈希链。当前状态仅 `implemented`；远端 PyTorch、真实 THUMOS gate 与 pilot 尚未完成。
- 2026-07-13：G1a 扩展诊断与回归完成。远端 Linux/Torch 新旧相关 suite `100 passed`；修复 `AnchorFreeHead` 的 `dense_valid_len` 残留 NameError，以及 `selected_center` view 被物理中心原地写入污染、从而错误裁剪合法候选的关键 bug。部署合同升级为 commit/tree/config/data/checkpoint 全链绑定、双臂三步 AMP、正式单视频滑窗 NMS/evaluator 和严格 6 epoch artifact 验收。G1a 状态提升为 `tested`，正式 fixed-snapshot gate/pilot 与 mAP 仍 pending。
- 2026-07-13：G1a 预部署收口完成。真实 gate 改用 test split 尾样本和 test evaluator；数据指纹升级为逐文件完整 SHA256/Merkle；checkpoint/metrics 验收升级为真实反序列化与 evaluator 独立重算；VideoMAE/TIA 在 patch、attention、残差、MLP、卷积和 norm 全路径实施严格 padding isolation。全量 411 个 THUMOS14 MP4 的 decoder/annotation timebase 审计确认最大相对 FPS 偏差约 1.12%、帧数偏差为 0，配置容差固定为 1.25%/0.01%。远端新旧回归 `116 passed`；证据仍仅为 `tested`，正式 clean snapshot gate、pilot 和 mAP pending。
- 2026-07-13：首个 G1a clean snapshot `8e2b832` 已部署；gate `1161304` 在模型执行前正确失败，依赖 pilot `1161305/1161306` 未启动并取消。根因是 test 根目录 213 个 MP4 中有 2 个不在 annotation/正式 data_list，旧全量审计错误把目录集合等同于 evaluator 集合。修复后审计范围严格来自 `build_dataset(...).data_list`，消费 200 train+211 test；两个未引用文件显式登记并由目录 Merkle 绑定，被消费文件缺失仍硬失败。真实目录范围 precheck 与远端 `116 passed` 完成，等待新 commit/snapshot 重排。
- 2026-07-13：范围修复 commit `e598bd7` 的 gate `1161353` 越过 timebase 审计后，在模型初始 state 摘要处因 0 维 LongTensor 直接 byte-view 失败；pilot `1161354/1161355` 未启动并取消。已改为 `reshape(-1).view(torch.uint8)` 并加入标量 buffer 摘要回归；证据仍为工程修复，尚无 pilot mAP。
- 2026-07-13：标量摘要修复 commit `d193417` 的 gate `1161378` 越过全量数据、checkpoint、evaluator 与模型构建，在 selected-axis 首个真实样本因旧逐样本 `regression_gradient` 非零合同 fail-closed；pilot `1161379/1161380` 未启动并取消。根因是三步 gate 错把 ActionFormer ReLU 回归头的单样本零参数梯度当成断路。补丁现要求每步正 assignment、正 `reg_loss`、全部有限，adapter/projection/classification 每步非零，regression 三步内至少一次非零，并保存逐步 assignment/梯度证据；远端完整回归 `118 passed`，正在接受独立最高强度逐行审查，状态保持 `tested`。
- 2026-07-13：对上一条“ReLU 根因”作证据纠正：旧 `1161378` artifact 没有 assignment、`reg_loss` 或 pre-ReLU 激活，故只能说现象与 dead zone 一致，不能说已证明。独立 max 审查发现 gate validator 可被顶层字段伪造、buffer 可冒充参数更新、batch/scheduler 不是生产轨迹、`scale` 漏检及 schema/artifact 防御缺口。v3 修复改用正式 batch=2 DataLoader、warmup scheduler、EMA 和生产更新顺序，记录并重算逐步 assignment/pre-ReLU/梯度/LR/optimizer state，以 trainable-only hash+delta 证明更新，并重算 pilot 全部绑定；远端回归 `142 passed`，同一代理第二轮复审中，状态仍为 `tested`。
