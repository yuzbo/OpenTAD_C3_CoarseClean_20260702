---
type: wiki_log
append_only: true
---

# Research Wiki Log

- 2026-08-26：PJST-D1 Cycle-3 在独立 clean revision
  `a16a67c4f74ce19de640704c357850c0e7b85ba3` 完成实现与独立静态复核。候选保持冻结的
  H65 选择器、同一 K384 RGB 集合和 matched OFF/ON 表示归因，只在首次 VideoMAE
  二帧 tubelet 混合前加入导数型物理时间变换，并补齐物理时间回映、Stage-1 checkpoint
  身份校验和单一启动器。N16R4 最终聚焦门为 `27 passed, 3 failed`：一项测试把七个非恒定
  帧对错误地纳入全张量不变断言，另外两项测试替身没有实现生产骨干的 `masks/metas` 接口。
  这些失败没有构成模型效能负证据，也没有证明生产公式错误，但零失败准入门尚未满足；因此
  Cycle-3 实现包关闭，未进入 PRE_RUN、训练或评估，当前无 PJST-D1 mAP/成本结论。科学假设
  保持未裁决；后续只能在另行授权的干净后继实现中修正测试合同并重新执行完整准入。

- 2026-08-25：SingleClock 统一终结作业在 24 小时时限后 `TIMEOUT`。四个
  SingleClock-on/gate-zero family 已完整生成，但 H65 OFF 配对终结器、整视频 bootstrap、
  H65 五边界回放身份和 nominal-uniform 首次混合/骨干逐位身份没有共同闭合，因此无合法
  PASS/KILL、无新 mAP 结论、无成本结论。这是终态证据不完整，不是科学失败。用户提供的 PJST
  独立审查已被核验为合理设计候选，但“允许 selector 随表示联合变化的系统总效应”与“冻结/重放
  selector 的纯表示效应”会产生不同训练、对照和 claim，不能混称。项目已准备唯一 fresh exact-DUCA
  Pro 请求 `PRO_REQUEST_DUCA_PJST_CAUSAL_CONTRACT-v001`；裁决前不实现、不 PRE_RUN、不训练。

- 2026-08-24：DUCA-TAS 固定机制五折正式矩阵已终态。Jobs `1252219/1252220/1252221`
  分别完成 Uniform-2x(K384)、Uniform-4x(K192) 与 H65-Fixed384，均为 `COMPLETED 0:0`，
  使用 clean revision `b0103cea4f3b4aff68463c3f28a1b9f4213c2df6`、seed 42、200 epoch、
  epoch-199 EMA 和官方 50Salads 五折 evaluator。五折 Avg-mAP 均值为
  `82.138/78.434/83.192`，mAP@0.7 为 `74.070/68.744/74.678`；H65 相对同预算
  Uniform-2x 为 `+1.054/+0.608` 点。五折 Avg 差值四正一负，且尚无独立训练 seed、
  配对区间或完整成本，因此只支持固定预算语义间接采样的初步定位保护，不支持动态预算、
  统计稳健性、端到端成本或跨任务主张；动态两臂仍未启动。

- 2026-08-24：H65 60轮学习率归因已经终态。复用成熟 Stage-1 的 AM-RPCH25 与 LongCosine-H6000 均完成30轮 Stage-2/3000次成功更新，终态 EMA Avg-mAP/mAP@0.7 分别为 `63.22/41.25` 与 `63.56/41.01`，均未进入相对历史30+60参考的预注册恢复邻域。LongCosine 保留更多学习率面积但没有恢复高 IoU；两臂末段 Avg-mAP 均平台或回落。项目据此停止全部60轮压缩调参，保留30+60参考；下一动作仅为既有 checkpoint、日志和预测的只读训练动力学尸检，不新增训练。该负结果否定已测试的日程压缩，不否定H65间接选帧。

- 2026-08-23：DUCA-TAS H65 True-Time 固定机制矩阵现以最终 clean revision
  `b0103cea4f3b4aff68463c3f28a1b9f4213c2df6` 运行。早期正式作业先后暴露 evaluator 类别路径、
  checkpoint 随机状态设备、指标账本路径和 PyTorch 2.0.1 确定性 CUDA 累计限制；修正均只闭合
  执行合同，没有改变 scout、采样、预算、EAST head、损失或 evaluator，也没有产生可引用的效能
  证据。最终同提交 N16R4 focused suite 为 `21 passed`，远端 checkout 为 clean。
- 2026-08-23：最终五折正式 Jobs `1252219/1252220/1252221` 分别执行
  `uniform_2x(K=384)`、`uniform_4x(K=192)` 与 `h65_fixed384`，并从已封存的 split-1
  epoch `13/9/3` checkpoint 恢复。三者均已进入 `RUNNING`；H65 已完成 epoch-5 的真实验证并
  继续到后续 epoch，证明确定性逆累计采样、官方 evaluator 和恢复链能够贯通。epoch-5 的早期
  数值不是终态性能证据。协议仍为五折、每折 200 epoch、seed 42、每 2 epoch checkpoint、
  terminal epoch-199 EMA。动态预算两臂维持未提交，等待固定 K 机制门的完整五折终态裁决。

- 2026-08-23：DUCA-TAS H65 True-Time 的真实数据准入和固定机制正式矩阵已启动。执行源码来自独立
  clean worktree `E:/DeskTop/TAD/OpenTAD_DUCA_TAS_H65_TrueTime_20260823`，分支
  `codex/duca-tas-h65-truetime-vits-20260823`，冻结训练 revision
  `3eb2fbfd37c03984a12e8e86517c91bb3675b80e`。N16R4 同提交 focused suite 为 `18 passed`；
  fold-1 真实训练准入 Job `1252185` 完成 1 epoch、19 次优化步骤并生成 `624,566,148` 字节的
  epoch-0 checkpoint，恢复包含 model、EMA、optimizer、scheduler、AMP scaler 和两 rank RNG，状态为
  `PRE_RUN_READY`。首个 attempt `1252184` 在训练前被 clean-snapshot 门阻断，原因仅是生成的 Python
  bytecode 改写了执行快照；没有产生模型、优化或效能证据。
- 2026-08-23：固定机制矩阵 Jobs `1252193/1252194/1252195` 已分别开始 `uniform_2x(K=384)`、
  `uniform_4x(K=192)` 与 `h65_fixed384` 的完整五折训练。每折使用真实 50Salads 官方 split、seed 42、
  200 epoch、每 2 epoch 可恢复 checkpoint、latest-3 加 milestone `49/99/149/199`，终态固定读取
  epoch-199 EMA 并运行官方 evaluator。动态 `dynamic_uniform/h65_dynamic` 尚未提交：只有固定
  `h65_fixed384` 相对 `uniform_2x` 的完整五折机制门通过后才可进入动态预算实验。当前状态为
  `experiment_running`，尚无新的 TAS mAP、稳定性或端到端成本结论。

- 2026-08-23：DUCA-TAS H65 True-Time 最小实现已在独立 clean 工作树
  `E:/DeskTop/TAD/OpenTAD_DUCA_TAS_H65_TrueTime_20260823` 完成，分支
  `codex/duca-tas-h65-truetime-vits-20260823`，候选提交
  `42e1b639d08481b9042f5c4d5ec0544955795b01`。实现包含低分辨率 19 类语义/类别转换 scout、
  确定性 exact-K 逆累计采样、patch embedding 前真实 RGB gather、按实际 K 分桶的 VideoMAE-S
  重计算、零初始化物理时间偏置、规范 2 fps 轴重建、五折配置及单一启动器；EAST head、损失、
  NMS 和 evaluator 未改。N16R4 focused tests 为 `18 passed`，绑定真实 50Salads/预训练路径的
  launcher `PRECHECK_ONLY` 为 `17 passed`，完整配置模型实例化通过。该证据仅支持
  `implemented_static_verified`；尚未执行 PRE_RUN、训练或评估，因此没有新的 TAS 性能或成本结论。

- 2026-08-23：用户批准 DUCA 向 50Salads/EAST 迁移的新语义合同。由于 19 类 EAST 标注完整
  覆盖时间轴、没有背景类，原前景/背景 actionness 二分类退化为全正标签；Scout 改为低分辨率
  19 类粗动作识别与类别转换二分类，确定性地形成 H65 正密度和 exact-K 非均匀原始时间位置，
  不直接学习帧索引。选中 RGB 必须在 VideoMAE-S patch embedding 前真实 gather，并在第一次
  时序混合前注入零初始化 True-Time 相对物理时间偏置。固定 K=384 只作机制兼容门，动态 K
  `{192,256,320,384}` 为主线；已有 dense 锚点不重训。独立 clean EAST/DUCA 分支
  `codex/duca-tas-h65-truetime-vits-20260823` 已在基座 `37c0d080...` 冻结设计合同，提交
  `abfea355ec0361444cb71ea374a96f65403dcd5d`。状态仅 `designed`；未实现、未 PRE_RUN、未训练，
  尚无 TAS 性能或成本结论。

- 2026-08-23：EAST 官方 VideoMAE-S 五折原始 RGB 复现已完成。Slurm 数组
  `1249797_1…5` 均以 `COMPLETED 0:0` 结束，固定加载 `epoch_199.pth` 的 EMA 权重进行
  官方 validation evaluator 测试；五折 Avg-mAP 为 `81.68/83.59/86.08/84.62/82.97`，
  算术均值为 `83.79`。五折 tIoU 0.1/0.25/0.5/0.6/0.7 的算术均值分别为
  `89.50/88.21/84.42/80.78/76.03`。五个 epoch-199 checkpoint 均存在，Slurm exit code
  全为 `0:0`，Traceback、CUDA OOM 与非有限 loss 扫描均为 0。该结果是 24 GiB GPU 可运行的
  ViT-S 次级锚点；EAST 官方论文只发布 ViT-G 结果，故不得把 `83.79` 描述为 ViT-G 复现或
  与其 `88.86` 锚点作同模型等价复现结论。

- 2026-08-23：用户提供的 `D:/chrome_download/50salads.tar.gz` 已作只读身份核验。该
  `56,052,488`-byte 归档包含五折 `l_5uniform` 的 `best_eva_acc/best_eva_FEA.model`、optimizer
  与 100-epoch 日志；单个模型约 2.06 MB，保存名称和代码路径与 EAST 官方
  `ms-tcn-master2` Stage-2 高帧率聚合/细化网络吻合。它不含 ViT-G/adapter/ActionFormer 或
  `state_dict_ema`，不是 Stage-1 ViT-G detector checkpoint，故没有解压或启动错误评估。
  官方 ViT-G 发布权重仍是唯一未满足输入。

- 2026-08-23：EAST ViT-G 官方发布 checkpoint 原配置评估已完成执行前身份核验，但被作者资源
  访问门阻断。官方 `tqosu/EAST@a3233c2e...` 的 50Salads 配置、`tools/test.py`、EMA 加载和
  tIoU `0.3--0.7` evaluator 已逐项固定；规范 50 个 RGB 视频与五折标注均已就绪。远端定向清点
  只找到 2,025,314,665-byte VideoMAEv2 ViT-G backbone 预训练权重，没有 EAST detector
  checkpoint；本机下载目录也没有该文件。作者 README 的官方 Box model link 在匿名 HTTP、应用内
  浏览器和 Edge 中都落到 Oregon State University 登录页，无法取得文件清单或下载地址。故本轮
  没有提交 Slurm、没有产生指标，也没有用预训练权重冒充发布模型。完整启动合同记录于
  `experiments/east-vitg-released-checkpoint-evaluation.md`；唯一恢复条件是取得作者原始发布包并确认
  checkpoint 与 fold/EMA 的绑定。

- 2026-08-23：EAST ViT-S 五折正式复现的四折中间状态。数据入口仍为 50 个原始 RGB 视频，
  `121,662,019` bytes；未发现下载材料丢失或运行错误。`1249797` 中 split2、split3、split4、
  split5 已完成 200 epoch、固定 epoch-199 EMA test 并 `COMPLETED 0:0`，对应 Avg-mAP 为
  `83.59/86.08/84.62/82.97`；各自 tIoU 0.1/0.25/0.5/0.6/0.7 分别为
  `89.33/88.49/85.50/79.58/75.05`、`90.83/90.33/86.64/83.91/78.70`、
  `90.48/88.94/84.98/82.56/76.16` 和 `87.16/85.99/83.99/81.04/76.66`。split1 仍在训练，
  当前约为 epoch 175，
  Traceback、OOM、非有限 loss 计数均为 0。因此暂不形成五折均值，也不将中间 validation 峰值
  作为最终结果。

- 2026-08-23：EAST ViT-S 的训练—评估联合准入已闭合。早期完整数组在 epoch-1 暴露两个官方
  路径假设：evaluator 通过 annotation 文件名推导类别表，且 ViT-S 配置遗漏 ViT-G 已定义的
  `metr_path`。最终启动器只在结果目录建立指向规范 JSON/类别表的只读协议链接，并补入同仓库
  ViT-G 的 `epoch_metrics_mAP.csv` 字段；不改视频、标注内容、模型、损失或 evaluator。联合准入
  `1249796` 完成 2 个 full-data epoch、40 次优化、epoch-1 checkpoint 和官方 evaluator，峰值显存
  2531 MiB，`COMPLETED 0:0`。epoch-1 的 0 mAP 仅是未收敛准入输出，不是效能结论。五折
  200-epoch 完整训练 `1249797_1…5` 已全部进入运行；结果根为
  `/data/run01/sczc063/yuzibo/duca_tas_east_official_vits_37c0d08_v6_20260823T011000Z`。

- 2026-08-23：EAST 50Salads 原始 RGB 下载与协议门闭合后，正式复现已进入可运行阶段。官方
  ViT-G 双卡五折 `1249455_1…5` 均在第一次真实前向因 24 GiB 单卡显存不足而终止；分配器
  碎片控制和 CUDA 异步分配器复核仍在同一 546 MiB 申请处失败，因此没有 epoch、预测或 TAS
  指标，不能把它解释为模型负结果。clean 执行候选已推进到
  `37c0d080a2bce948dc73643578f05b2229934d2c`：在不改数据、网络、损失和评估器的前提下，补齐
  AMP/RNG 恢复并将每 2 epoch 的官方 checkpoint 保留策略限制为最新 3 个加 50/100/150/200
  或 60/120/180/240 里程碑。当前硬件可承载的官方 ViT-S 配方通过完整 split1 一轮准入
  `1249643`：40 个训练视频、20 次优化、峰值显存 2528 MiB、exit 0；其第一版五折作业随后由
  evaluator 路径合同阻断并已被上方联合准入和新运行链替代。ViT-S 结果将独立报告，不得冒充
  ViT-G 论文锚点。

- 2026-08-22：EAST 正式五折训练已实际启动。第二次准入 `1249389` 发现官方
  `opentad.models.backbones.__all__` 导出了两个已注释且不存在的 InternVideo2 类；clean 候选
  `94b24753588ff60be986b35fefcca3f43d9c3fe6` 只移除这两个无效导出并增加回归测试，恢复合同
  不变，目标环境共 `3 passed`。一次后续空日志失败由官方仓库跟踪的 `.pyc` 被测试重写导致
  clean 门关闭；恢复冻结缓存并设置 `PYTHONDONTWRITEBYTECODE=1` 后，双卡 PRE_RUN `1249454`
  于 71 秒内 `COMPLETED 0:0`，实际构建 40/10 split、1.011B ViT-G、44.7M adapter、双卡 DDP、
  AMP、EMA 与 optimizer。依赖释放后，五折完整训练 `1249455_1…5` 已分别在
  `g0030/g0056/g0056/g0059/g0063` 运行，首轮日志均已进入官方 240-epoch 配置和权重加载，错误
  扫描为空。当前无 epoch、TAS 指标或成本结果。

- 2026-08-22：EAST 双卡结构准入 `1249294` 在进入模型前因启动器顺序失败：脚本先启用
  `set -u`、后加载集群 `/etc/profile`，从而由站点脚本的未定义 `LC_BYOBU` 触发退出。该终态
  不涉及数据、模型、显存或科学假设。其永不满足依赖的完整数组 `1249295` 已取消；启动器仅将
  `/etc/profile` 前移，Bash 语法通过。同一冻结代码/数据/配置的替代 PRE_RUN `1249389` 和
  afterok 五折完整数组 `1249390_[1-5]` 曾提交；其后由上方记录的官方导出缺陷阻断并被新的
  clean 候选与运行链替代。该阶段没有训练或 TAS 指标。

- 2026-08-22：EAST 50Salads 原始 RGB 数据门已经闭合。官方 Box `video_fps2` 的 50 个
  `160x160 @ 2 FPS` MP4 已逐文件下载并绑定到
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/data/50salads/raw_data/video_fps2`；
  总字节数 `121,662,019`，50/50 文件尺寸匹配，ffprobe 元数据与完整 ffmpeg 解码均通过。
  五折 JSON、类别表和视频 UID 全量对齐；每折 training/validation 为 40/10，50 个视频各作为
  validation 恰好一次。官方 EAST `a3233c2e...` 的 ViT-G 配方已建立 clean 候选
  `57282b4d...`，唯一代码差异是补齐 epoch-boundary checkpoint 的 AMP scaler 与各 rank
  Python/NumPy/Torch/CUDA 随机状态，focused tests 为 `2 passed`。官方更频繁的每 2 epoch
  checkpoint、EMA 与 terminal epoch-239 选择保持不变。双卡结构 PRE_RUN `1249294` 与其
  afterok 五折完整训练数组 `1249295_[1-5]` 已提交；PRE_RUN 当前因共享账户 GPU 配额
  `AssocGrpGRES` 排队，完整训练保持 dependency。该初次启动链随后由上方记录的纯启动器问题
  终止并已替换。结果根：
  `/data/run01/sczc063/yuzibo/duca_tas_east_official_57282b4_20260822T220000Z`。当前没有 TAS
  性能结果；released checkpoint 的作者 Box 链接要求 Oregon State 登录，未被伪装成已获得。

- 2026-08-22：EAST 官方 Oregon State Box 共享页已核验并接入 N16R4 学术代理。50Salads
  `video_fps2` 明确为 50 个 `160x160 @ 2 FPS` RGB MP4（约 116 MB），与 Dundee 原始
  `640x480 @ 30 Hz` AVI、MS-TCN++ Zenodo 预提取特征严格分开。视频归档正在远端
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/video_fps2.zip.part`
  下载；Box 服务器不提供 Range，故 aria2 实际单连接。EAST 标注归档已完成（1,417,273
  bytes，ZIP 完整性通过），包含类别表、五折 JSON 与五个 `.swp.json`。当前证据状态为
  `RGB_DOWNLOAD_RUNNING / EAST_PROTOCOL_ANNOTATIONS_READY`；尚未解压、绑定或启动训练。

- 2026-08-14（local material epoch v013）: 项目所有 ARIS DeepSeek V4 Pro Executor 在**不裁决路线**的前提下，将最后一轮 fresh Project Pro 的科学问题从“修复 F1 单调解码器 identity”重构为**内层机制二选一**：(A) outer dynamic K + 任意非连续选帧 + 逐帧精确原始时间戳/physical-coordinate 逆映射；(B) outer dynamic K + monotone/local physical exact-K transport。两条都必须保留真实时间戳、禁止“把 selected ordinal 当作 uniform time”（该行为是历史 selected-rank/selected-axis 扭曲，属结构性禁止而非阈值）。A 的趣味是最大化分配自由度（最接近历史 GT-oracle boundary burst）并以“诚实时间戳”而非几何约束保证正确性，但代价是时间戳必须进入 detector 时间语义（true-time positional/adapter），属 enhanced integration、非纯插件；B 的趣味是纯插件兼容（有界单调保持 detector ordinal-grid 假设）、有硬覆盖保证、常量密度严格退化为 canonical uniform，但单调约束限制了聚集自由。video 复杂度→不同 K 的候选驱动：动作实例数/边界密度、短动作占比、时长分布、背景占比、边界时间歧义（运动模糊/遮挡/镜头切）、相邻动作转变锐度。受益边界/相位：start/end 边界、短动作、重叠/相邻动作、边界附近高运动转变、paired start+end（tIoU 0.7 跨界）。novelty 无效化：AdaFrame/MGSampler/AdapTok/AdaFocusV3/SMART/TAPS/Progressive Block Drop/keyframe/semantic-boundary wavelet；A 另有 TE-TAD/PhysTime/TrueTime 时间对齐先验；B 另有 Hartley systematic sampling 与 Uni-AdaFocus inverse-CDF 近邻。六臂合同：dense / uniform_k384 / dynamic_A / dynamic_B / k_shuffle(F2) / no_risk，全臂共享 detector/loss/NMS/evaluator/6k updates/terminal EMA，official val/test 不可达。证伪：F-O1 动态 headroom、F-O2 内层几何+decoder identity（F1 在此决定性）、F-O3 G_rank、F-O4 pair-risk、F-INV 时间戳不变量。F1/F2 保持为**未决 Pro 问题**。v008 source batch = UNKNOWN_REMOTE_STATE/quarantined，本轮新建 distinct local epoch，绝不 retransmit 或依赖 v008。交付：`.cvpr-pro-lab/role-returns/BUILDER_DUCA_DYNAMIC_INNER_MECHANISM_MINIMAL_CHANGE_PLAN-v001.md` + 5 个命名 wiki/ARIS 文档 + 新终态 receipt。无实现、无 PRE_RUN、无数据、无 GPU/Slurm、无训练/推理/评估/指标/成本/claim。停在 MATERIAL_READY。

- 2026-08-14: ARIS transport-only correction（唯一替换 Executor）完成 C/P/R 材料准备，非科学重试。已固化环境事实：cwd=`E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702`、git common-dir=`.git`、HEAD=`a6bdc084cc145c80b6b2c68d0a38f0deea3e8518`（SparseHead 提交，**非 DUCA revision**）、branch=`codex/duca-total60-plugin-cvpr-20260727`、工作树脏约 275 条——本工作树绝不可称 clean frozen DUCA revision；DUCA 证据=命名 untracked working-tree 表面+`research-wiki/`。parent-fidelity 分类：`PrefixMarginalUtilityBudgetController`+center-radius=`真实已有生产 dynamic-budget 工作`（`paused/negative`）；RIME/outer-budget-inner-transport=`仅讨论/规划，无实现`；`density_decode.py`=frozen fixed-K 有界密度分位解码器（`PROTOTYPE_ONLY`，只作 baseline/attribution/fallback 与内层复用原语）；`pc_ot_mras_*` selectors/configs=SparseHead 基础（tracked，非 DUCA）。C 阶段三条非等价路线：A 原始动态选帧恢复（REJECTED as final，`lambda_dual` 留作原语）；B 层级 Dynamic DUCA（outer per-video K + inner physical exact-K transport + hard utility + paired-boundary/high-IoU risk，**RECOMMENDED default，仅条件默认**）；C realized-cost Lagrangian density acquisition（复用 `lambda_dual`+frozen `density_decode`，strongest deliverable fallback）。sealed U/O/R execution-surface 终态失败仅作负证据，不提议第三次修正。交付：`docs/aris/ARIS_CPR_PLAN-2026-08-14.md`、`ARIS_DECISION_LOG-2026-08-14.md`、`DUCA_ARIS_SOURCES_TO_PRO_REQUEST-2026-08-14.md`、终态 receipt。无实现、无 PRE_RUN、无 pilot、无训练、无性能/成本/claim。

- 2026-08-14: Fresh exact-Project Pro terminal adjudication (`DUCA-UOR-TERMINAL-v001`) selected `REVISE-AUTHORIZE`, not a third repair of the failed U/O/R generic launch surface. The old package at `657678946` remains immutable terminal failure evidence after its second equivalent `--cfg-options` gate-disable bypass. The sole next route is the additive `DUCA_UOR_SEALED_EXECUTION_SURFACE-v001`: dedicated no-override runner, closed manifest, sealed evaluator, and fixed future launcher, followed by one independent static Critic and (only on PASS) one existing-Evaluator structural intake. Any block stops the replacement. FIT/CAL and official-validation boundaries stay unchanged. This decision authorizes only tracked local static/synthetic no-data work; no data, runtime, GPU, Slurm, metrics, efficacy, cost, or paper claim exists. The later CAL U/O/R comparison remains the cheapest potential falsifier, but is not authorized.

- 2026-08-14: ARIS Executor 将 DUCA 主线从 P0 投影 identity/optimality gate（纯数学正确性，
  连续数日阻塞、无科学进展）抽回可证伪科学。锁定主路线为 Pro 冻结的
  `DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002` 的科学校准：在正确 physical-time
  decode 下，检验边界集中有界密度相对 exact-uniform 的 mAP headroom。最便宜 falsifier =
  无训练 frozen-detector 的 GT-boundary oracle 密度 vs exact-uniform（逐视频 paired CI）；
  CI 下界 ≤0 即 KILL 密度路线并转 fallback（坐标正确性独立贡献 / CVCR-BCFT）。本轮落地并
  双端验证缺失的冻结解码器：`opentad/models/duca/density_decode.py`
  （`canonical_uniform_positions`/`decode_duca_density_positions_v001`/
  `project_duca_density_positions`/`DUCAProjectionError`），全部冻结 fixture（768/384 止于
  767、G16-U/G17-E2/EINF/E1/U1/PLEX/G31-U/G32-U/F768-U/G767-U/G385-X）与独立 brute-force
  reference 一致，负例 code 全对。环境探针确认 SSH N16R4 + sbatch + OpenTAD env + THUMOS
  数据齐备。CPR 包见 `docs/aris/ARIS_CPR_PLAN-2026-08-14.md`，决策日志
  `docs/aris/ARIS_DECISION_LOG-2026-08-14.md`，独立 Pro 批评包
  `docs/aris/ARIS_PRO_HANDOFF_PACKET-2026-08-14.md`。证据状态仍 `BLOCKED_PRE_RESULT`，无
  性能/成本/claim 结论；oracle headroom falsifier（GPU eval-only）为 launch-ready，尚未投递。

- 2026-07-23: 新增 `duca_paper_story_theory_figure_contract.md`，将当前论文裁决冻结为
  `HOLD / designed_waiting_terminal_evidence`。复核发现现有 `paper/` 仍在讲旧
  zero-shot、teacher utility、generic Top-K 和 `window-online`，与当前 offline TAD 的
  transition/boundary-burst 主线不一致。新合同固定三项贡献、五类理论命题、主张-证据
  矩阵、五幅主图和四张主表，并明确内部 R0 93--94 mAP、粗分类指标和 selection proxy
  不得冒充官方 TAD 结果。同步增加防止“实验编号=创新”和提前主打 TTDI/feedback 的
  anti-repetition 规则；未改变模型、实验或 claim 状态。

- 2026-07-22 16:30 +08:00: e49 unrestricted GT Oracle completed at exact Avg-mAP
  `93.970057`, `+0.382987 pp` over uniform and `-0.220440 pp` below R2Q3. Per-tIoU
  mAP is `98.1881/97.3696/96.3783/93.0427/84.8715`. All four e49 R0 point
  estimates now reproduce d9 exactly. Job `1179795` entered the frozen 1000-sample paired-video
  bootstrap; summary/CI/family seal and P0 start remain pending, with no error signature.

- 2026-07-22 16:23 +08:00: e49 R0 projected R4Q5 returned exact raw Avg-mAP
  `93.999241`, `+0.412170 pp` over uniform but `-0.191256 pp` below R2Q3.
  Per-tIoU mAP is `98.0039/97.3369/95.8351/92.8220/85.9983`. The result is
  reproducible with prior d9 and suggests wider/denser boundary bursts are not monotonically
  better. Unrestricted Oracle and paired bootstrap remain pending; no P0 unlock or paper claim.

- 2026-07-22 16:17 +08:00: e49 R0 projected R2Q3 raw Avg-mAP is `94.1905`,
  `+0.6034 pp` over exact-uniform `93.5871`. Per-tIoU mAP is
  `98.15/97.52/96.63/93.04/85.61`; the largest gains are at 0.5 and 0.6. This is positive
  Oracle reachability information, not a learned-selector or paper-test result. R4Q5,
  unrestricted and paired bootstrap remain required before P0 unlock.

- 2026-07-22 16:13 +08:00: e49 R0 produced its first raw official-evaluator result.
  Exact-uniform Avg-mAP is `93.5871`; tIoU `0.3/0.4/0.5/0.6/0.7` is
  `98.05/97.45/95.44/91.85/85.15`. It matches the prior d9 point estimate, closing baseline
  replay identity. R2Q3/R4Q5/unrestricted and bootstrap are still running, so no family or P0
  decision is authorized yet.

- 2026-07-22 16:11 +08:00: e49 R0 Job `1179795` completed 124-window construction for
  exact-uniform, R2Q3, R4Q5 and unrestricted Oracle; each family has 124 rows and the promoted
  family JSONL SHA-256 is `f2cbcd27...e6eafa4d`. The job entered frozen official-AdaTAD mAP
  replay, beginning with exact-uniform. stderr and explicit error scan remain clean; no raw mAP,
  bootstrap decision or P0 start exists yet.

- 2026-07-22 16:12 +08:00: 基于 GitHub 精确提交 `e49ef696`、R0--R5 完整代码与
  当前正式 Slurm DAG，生成可直接交给 Pro 的逐行方法审查 Prompt：
  `research-wiki/duca_r0_r5_pro_review_prompt_20260722.md`。它强制审查间接边界初心、
  Oracle 式 burst、exact-K/G、hard RGB/selected-axis、梯度所有权、R0--R5 因果链、
  双后端、完整成本和论文改进空间，并禁止用站点调度洁癖替代方法裁决。

- 2026-07-22 16:05 +08:00: 精确提交 `e49ef69605e1f98a7217957483f93a8a64bfc348`
  已推送 GitHub 并全面部署 R0--R5。远端 clean snapshot 通过 focused `192`、强制 C3
  `23`，部署前独立 MAX `019f88bf-272f-7373-b702-5b66b142cbdc` 给出
  `GO_TO_SLURM`。R0--R4 依赖链为 `1179795 -> 1179796 -> 1179797 ->
  {1179798,1179799} -> 1179825 -> 1179826`。R5 真 TemporalMaxer 门禁为
  `1179827`。因站点 `AssocMaxSubmitJobLimit`，六个未运行逐 cell 重复 Job
  `1179828--1179833` 取消；完整 24 个 cell 改由四个等价 GPU 批次
  `1179861--1179864` 执行，9 个 cost 加 aggregate 由 `1179865` 执行。部署收据、
  bundle manifest SHA 分别为 `ed217ee2...aa1bd`、`e6ca0acf...5656d8`。16:05
  `1179795` RUNNING，其余 dependency pending；错误扫描干净，尚无 terminal e49 mAP/cost。

- 2026-07-22 13:05 +08:00: 部署前真实模型门禁 Job `1179602` 已提交，依赖 `afterok:1179533`，当前 `PENDING (Dependency)`。它在 `44c7227` 干净快照上执行 selected-family G1/G2 official AdaTAD exact full-model gate 与 live DUCA→VideoMAE→TemporalMaxer one-step gate；产物根 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_44c7227_model_gate_20260722_1305`。这是门禁，不是正式论文实验；独立 MAX 与正式 DAG 仍在其后。

- 2026-07-22 12:59 +08:00: 最终合并候选 `44c7227b575b22c666b2f309c69b1dcfdc4102c8` 已推送。干净 N16R4 快照 `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_44c7227_20260722` 通过 R0-R5 focused `90 passed`、C3/ASFormer `23 passed`、py_compile/bash。此前唯一失败是测试 fake selector 漏坐标字段，生产 fail-closed 检查未放宽。真实 CUDA 门禁和部署前独立 MAX 仍待完成，正式 DAG 仍禁止提交。

- 2026-07-22 12:49 +08:00: R5 第二后端提交为 `163be81fb640376f0d6e1b09c86eb011f3402242`。真实 DUCA→VideoMAE→TemporalMaxer、GT/推理轴映射、optimizer 覆盖与 detector→selector 一步反传门禁已实现；24 个 ActionFormer/TemporalMaxer × U/learned × K384/K256 × 3 seeds 配置可生成。复杂 orchestrator 已删，Windows `6 passed, 2 skipped`；真实 CUDA 门禁待补，未提交 Slurm。

- 2026-07-22 12:47 +08:00: R4 legal hard-swap 与 G1/G2 模型实现提交为 `e253bba52dce1814f4ef356adcad286bc0884457`。真实 RGB remove/add、GT 重映射、冻结 official AdaTAD utility、梯度/utility alignment、G1/G2 均已落地；focused `47 passed`、基础 `23 passed`。未提交 Slurm，等待最终合并、真实 CUDA gate 与部署前独立 MAX。

- 2026-07-22 12:33 +08:00: 用户要求停止复杂启动器和工程框架，重心回到模型。R4/R5 并行实现智能体已被立即改向：保留最小 runner/sbatch，删除或停止扩张通用编排层；优先完成 hard swap、检测梯度、TemporalMaxer 真后端和机制测试。该规则已同步两份 RTK。

- 2026-07-22 12:31 +08:00: R1/R3 生产执行合同提交为 `523b45f62eb3a3d0c2856f33161dc541932c564a`。DAG 仅保留 R0、P0、gate、matched-U、运行时唯一 selected-G0 与 aggregate 六个角色；远端 focused `74 passed`，真实资产 precheck 成功，未提交 Slurm，等待 R4/R5 合并、真实门禁与部署前独立 MAX。

- 2026-07-22 12:25 +08:00: 用户纠正最终独立 MAX 的时点。唯一正确顺序为：完成 R0-R5 生产实现与真实后端门禁，随后在任何新的正式 Slurm DAG 部署前执行一次无实现上下文的独立 MAX 机理审阅；GO 后才允许部署，再收割实验结果。根 RTK、canonical RTK 与执行计划已同步；已运行 R0/P0 只保留为先前证据。

- 2026-07-22 11:31 +08:00: Corrected R0 Job `1179517` completed input export
  for all `124/124` windows in about four minutes and entered the four-family
  Oracle construction stage (`holdout_families.jsonl.partial` growing).
  Export SHA-256 is `5e65ac3d...23749`; exact commit/checkpoint/dataset seals
  reopened successfully. The job remains RUNNING on `g0048`, stderr is empty,
  and no hard error exists. Final frozen-detector mAP is still pending.

- 2026-07-22 11:29 +08:00: Live R0 check found Job `1179517` still RUNNING
  on `g0048`; export reached log batch 80 (`81/124` samples). The partial
  JSONL was actively growing, stderr remained empty, and the explicit
  Traceback/OOM/non-finite/FAIL scan found nothing. No R0 mAP or bootstrap
  decision artifact exists yet.

- 2026-07-22 11:27 +08:00: Corrected R0 Job `1179517` entered RUNNING on
  `g0048`. The exact d9 snapshot remained clean; the runtime binding and
  blocked-video artifacts were created, holdout export began at batch 0,
  stderr was empty, and no Traceback/OOM/non-finite/FAIL was found.

- 2026-07-22 11:24 +08:00: After explicit independent R0-only GO, submitted
  corrected exact-d9 R0 as Job `1179517` from run root
  `duca_boundary_d9fb398_r0_formal_20260722_112357`. It initially queued on
  Priority. Manifest, r0 sbatch, jobs journal and split hashes are
  `66bb5f5d...74e15`, `4a6ace6c...ab8c`, `25d90d86...07f4a`, and
  `88309edf...708f`. The atomic journal contains one no-dependency R0 row;
  no P0, gate, official arm or aggregate job was submitted.

- 2026-07-22 11:24 +08:00: Fresh independent MAX
  `019f87bb-d767-7713-825e-92b893e49a98` completed a read-only audit of exact
  d9 and granted R0-only GO. P0/full-model/official-60 remain separately HOLD.
  It verified solver uniqueness, family propagation, ASFormer rehash, U plus
  selected-G0 gate consumption and no-leak. Two future official-only defects
  were recorded: four sentinel GPU jobs can block aggregate, and aggregate
  does not rebind terminal pretrain path identity to the sealed P0/full gate.
  Neither changes or blocks R0 math. No job had yet been submitted at this
  record point.

- 2026-07-22 11:18 +08:00: Read-only remote reconciliation confirmed the d9
  snapshot is exact and clean, both replay JSONL hashes are
  `b49e03c...edd6d7`, old Job `1179392` remains immutable `FAILED/1:0`, and
  the current DUCA/boundary/R0 queue is empty. No corrected experiment was
  accidentally submitted while the independent MAX audit was running.

- 2026-07-22 11:06 +08:00: Unified and pushed the bounded R0/evidence repairs
  as exact commit `d9fb398578716d278e818745677a92976bcedf2c`. Clean Linux
  snapshot passed focused DUCA `88` and mandatory C3 `23` tests plus compile
  and shell checks. Two complete solves of the real 124-window R0 input were
  byte-identical at JSONL SHA-256 `b49e03c...edd6d7`; summary hashes differ
  only because they contain distinct output paths. Fresh independent MAX
  `019f87bb-d767-7713-825e-92b893e49a98` remains running. No corrected R0,
  P0 or official-60 job was submitted and no new mAP exists.

- 2026-07-22 10:08 +08:00: Independent solver worker produced
  `e267e1f9562c91fc0ad9a60382eb829d82d41acd`; it was cherry-picked to the
  canonical boundary-burst branch as
  `c418a951a9b9b7f7f19df785ead8642a4205c804`. The exact-quota MILP now pins
  overlap, position sum and a final block-wise lexicographic decision. Evidence:
  solver/GT `24 passed`, mandatory C3 `23 passed`, canonical tied/repeat
  `7 passed`, pycompile pass. No model surface changed and no job was submitted.

- 2026-07-22 09:37 +08:00: Independent replacement MAX
  `019f875f-1668-7e51-bf97-1f565b25e106` returned HOLD for R0-only, P0,
  full-model gate and official-60 on exact commit `22555a4`. It verified the
  exact-quota feasible-space constraints and zero-gap optimality; R0 is held
  only because equal optima lack a strict unique lexicographic finalization.
  Solver/test worker `019f877c-4595-7431-96a0-edff1f7b8251` now owns that
  minimal fix. Family/evidence worker `019f8766-c4db-7e30-8fc8-265d85d83b07`
  continues selected-family propagation, official-ASFormer consumer hash
  revalidation and full-model gate artifact reopening. No job was submitted.

- 2026-07-22 09:25 +08:00: Read-only remote reconciliation confirmed Job
  `1179392` remains `FAILED/1:0`; no job name matching corrected `22555a4`,
  boundary R0 or burst R0 exists in the current Slurm queue. Remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_22555a4_20260722`
  is exact HEAD `22555a4e830ce24f9bb516897b1bb7f44b70c188` and clean. The no-submit
  precheck root `duca_boundary_22555a4_precheck_20260722_0845` still contains
  its split, manifest and generated R0/P0/gate/arm/aggregate sbatches; none has
  been submitted.

- 2026-07-22 09:12 +08:00: Beijing 09:00 passed without corrected R0 mAP.
  MAX `019f8743-aed1-7a80-a7d6-552b08491019` returned no verdict and was shut
  down; it grants no experiment permission. Replacement no-context MAX
  `019f875f-1668-7e51-bf97-1f565b25e106` now audits exact commit `22555a4` with
  separate R0/P0/full-model/official-60 decisions. In parallel, bounded worker
  `019f8766-c4db-7e30-8fc8-265d85d83b07` implements only the frozen rule-300
  propagation from R0's selected family to P0/gate/R3. The hourly thread
  automation was updated to this exact state. No corrected R0, P0 or
  official-60 job is queued and no new detector mAP exists.

- 2026-07-20: Archived and independently absorbed the exact-commit DUCA
  physical allocation-family Pro review. Raw SHA-256 is
  `E40A69BD2DA9EBE32B41B45A136C2AA1A9FB8109A4875A16E2E3ABB7AF8FCC14`.
  The central `HOLD_AND_REVISE_FAMILY` verdict is accepted: fixed
  `G=3,192+192` CARA is withdrawn, and global exact-K under a frozen physical
  interval becomes the primary ceiling family. For an explicitly
  original-frame cap of 15 on the stride-4 grid, independent DP confirms
  effective cap 12 and minimum fixed scaffolds 255/382. The project does not
  yet freeze the meaning of "frame", accept the proposed code/statistical
  gates, or authorize training. Idea/experiment/decision/anti-repetition
  nodes were revised without changing any empirical claim.
- 2026-07-19: Added a bounded exact-commit Pro prompt for the immediate
  DUCA-CARA feasible-set ceiling decision. It fixes the GitHub review tree at
  `4ce69c8`, requires physical-coordinate reconstruction and family inclusion
  proofs, and explicitly forbids model training or broad method expansion.
  This is a discussion artifact only.
- 2026-07-19: Froze the immediate post-CellCF action. New bounded development
  starts from `4ce69c8` (verified descendant of immutable model commit
  `1642f26`) and implements only the allocation-family ceiling first. The
  candidate family must contain exact uniform, permit cross-region residual
  quota transfer, and report both dense-index and original-frame gap units.
  No CARA training change or long run is authorized before this gate.
- 2026-07-19: Archived and independently absorbed the exact-commit CellCF
  KILL/CARA redesign review. Raw SHA-256 is
  `3FB06655193E7CF665BB37CF0701C2708139B15DF40AC2114742C23B19E292E7`.
  Source/math checks confirmed one-per-cell quota rigidity, at-most-one-index
  displacement for `T=768,K=384`, actual acquisition versus uniform detector
  coordinate substitution, detached counterfactual supervision and the
  existing physical-grid ActionFormer path. CellCF is killed only as an
  adaptive-allocation main method and retained as `tested_diagnostic`.
  `DUCA-CARA` and its allocation-family ceiling were recorded as
  `discussed`/`designed`; no code, experiment or claim was promoted.
- 2026-07-18: Cost recovery Job `1170366` failed after 1,357 seconds before
  publishing evidence. The runtime profiler emitted seven component
  `*_cpu_enqueue_ms` fields, while `duca_full_stack_cost._derived_sample`
  rejected them as unsupported. Dependent completion Job `1170367` was
  cancelled with zero runtime. No resubmission was made; the next action is an
  exact producer-consumer schema repair and gate, not model retraining.
- 2026-07-18: Preserved the original CellCF terminal failure
  (`1167485 FAILED/1:0`, `1167486 CANCELLED`) and two fail-closed recovery
  diagnostics. `cost_recovery_5ab3042_v1` exposed the N16R4 policy requiring a
  GPU request even for the completion validator. `cost_recovery_67a8a0a_v1`
  exposed delayed `sacct SubmitLine` visibility; held Job `1170354` was
  cancelled before any release. Evidence commit
  `e153c96bfa0f37b9d4b82046e05b1bbce70dfe50` reuses the existing bounded
  strict-accounting retry, passed 230 exact Linux tests and independent
  P0/P1 review, and submitted the isolated recovery DAG: cost Job `1170366`
  plus completion Job `1170367` under `afterok:1170366`. No model training was
  repeated. C3/C4/C7 and cost claims remain unproven.
- 2026-07-17: Superseded post-run evidence commit `787569e` after the exact
  Linux suite reproduced a transient swap/restore gap that final hash, inode,
  mtime, ctime and parent-directory checks could miss. Replacement commit
  `9e96967a158534b014aacde57c1b78bd1591e71a` starts Linux inotify monitoring
  before every evidence read/hash and deterministically closes the monitor.
  Independent max review returned GO. On the target `/data/run01` `fuseblk`
  mount, the clean exact snapshot passed 14 finalizer tests and 253 broad
  evidence tests, including an independent-process replacement/restore
  attack. Formal cost Job `1167485` continues normally; repaired post-run jobs
  remain intentionally unsubmitted until `1167485/1167486` finish and the
  real-artifact precheck passes.
- 2026-07-17: All three immutable `1642f26` formal training arms completed
  `0:0`. Terminal-EMA Avg-mAP is exact-uniform `63.8594`,
  transition-beta0 `64.2755`, and CellCF `64.0610`; transition is `+0.4161`
  points over uniform while CellCF is `-0.2145` below transition. Aggregate
  Job `1167484` completed, cost Job `1167485` is running and completion Job
  `1167486` remains pending. These are raw one-seed results; C3/C4/C7 and
  paper readiness remain unproven until external sealing.
- 2026-07-17: Finalized the separate CellCF evidence-tooling branch at exact
  commit `2a0f848f7dbf17b7bcb40aa7a996954e8f87c4de`. Remote Linux/Torch
  verification passed 303 tests with three skips, including both training
  profiles and scheduler-free prepared-suite reopening. The immutable
  `1642f26` formal arms remained running at epochs 125/123/114
  (uniform/transition-beta0/CellCF), with checkpoints through 124/119/109,
  five-epoch checkpoint cadence unchanged, finite losses and no fatal log hit.
  No terminal mAP, allocation cost, official-60 result or paper claim exists.
- 2026-07-16: Pushed DUCA-CellCF evidence-DAG replacement commit
  `3a0f5ae54d1dbd23ff170cda8a4706f5ed0d38d3`. It hash-binds the six-job
  Slurm DAG, makes trained terminal-EMA cost mandatory, reopens terminal
  artifacts, distinguishes cost-pending aggregate from final completion, and
  validates intent/receipt identity against live/accounting Slurm state. Local
  CellCF/cost/submission checks are `84 passed, 1 skipped`; required C3
  regressions are `23 passed`. Independent max reviewer
  `019f6af9-7f66-7ea2-9bd8-38cfb75b92c8` returned
  `GO_TO_EXACT_COMMIT_GATE` after one receipt-fallback P1 was reproduced and
  repaired. No new gate, pilot, full train, mAP or cost result exists yet.

- 2026-07-16: Advanced DUCA-CellCF commit
  `475634e1be4a77ad1d9bc6bcf5f4bed04c3d6f31` through the first remote evidence
  layers. The clean Linux snapshot passed 62 focused tests. The exact-commit
  CPU synthetic gate passed exhaustive L=1..768 geometry, step-zero exact
  uniform and signed distinct-cell local-flip contracts; artifact SHA-256 is
  `ada3a32faaa496924a867ee616309ef06c5c3b653135b828f03107ac9ec7519c`.
  Initial Job `1167135` failed before Python because the batch bootstrap
  sourced the cluster profile under nounset. Job `1167140` then failed closed
  on a stale fallback THUMOS path because canonical dataset variables were not
  exported. Environment-corrected Job `1167145` passed real full/mixed/
  all-short THUMOS loading, informative signed utility, complete gradient
  groups and forced-overflow same-batch replay; its revalidated artifact hash
  is `e0f762fb1387fc823ca1b8ab5b2c291052897b24a75f712d1b6ba9e810b6d7f3`.
  Three-arm DDP pilot Job `1167146` then passed with ten successful updates,
  one forced replay and all K-patterns per arm; artifact SHA-256 is
  `1c180572683e5dafea00cea7364253b1a5fcc7a24b1916d34642c831de7929c0`.
  A fresh deployment audit returned HOLD on five final handoff P1s: mutable
  seed/job bindings, stale terminal-summary trust, unbound cost checkpoint,
  cost outside completion and weak receipt idempotence. No full job was
  submitted. Parallel bounded repairs are in progress; their replacement
  commit must rerun every gate before training.

- 2026-07-16: Implemented the bounded DUCA-CellCF route locally on
  `codex/duca-cellcf-20260716`. Added exact-cell acquisition, transition-first
  scoring, fixed detector anchors, detached distinct-cell hard-flip utility,
  matched uniform/beta0/CellCF configs, successful-update evidence,
  real-loader and forced-overflow gates, terminal metric recomputation, and a
  frontend cost pair. Local evidence: 27 CellCF contracts passed, three
  Windows Torch tests deferred to Linux, 23 required C3 tests passed, compile
  and shell syntax passed. A read-only audit found semantic cfg drift and
  shallow artifact-trust gaps; strict cfg allowlists, train/eval runtime hashes,
  raw artifact revalidation and finalizer binding now close them. Status is
  `tested` at clean commit `475634e1be4a77ad1d9bc6bcf5f4bed04c3d6f31`;
  remote gates, mAP and cost remain pending.

- 2026-07-16: Archived and partially absorbed the `7525efb` Round-2 method/
  paper Pro verdict. The 11,877-byte source and archive are byte-identical under
  SHA-256 `B4415ABA4B7B779257DF0F0D4E107586181C4DCBCBFF9F8B38BADB156A191E0B`.
  Reviewer verdict is `REDESIGN` to exact-uniform-anchored Local-cell DUCA.
  The project accepts coverage-preserving cells, detached hard local-flip
  utility and the bounded execution DAG, advancing CellCF to `designed` only.
  It rejects the TAPS/TAPOS substitution, direct-gradient C4 reuse, universal
  single-seed kill inference, evidence-free hyperparameter/threshold status,
  and unconditional deletion of direct-boundary attribution. No code, gate,
  experiment or claim becomes supported.

- 2026-07-16: Archived and absorbed the DUCA `7525efb` Round-1 code/math Pro
  review. The 16,034-byte attachment and raw archive are byte-identical under
  SHA-256 `DA4201C2D947C81EE6A799EF8B4572AD3D9C11DF047E29F8B70A9462B475F4C1`.
  Reviewer verdict is `GO_TO_REAL_GATE`, with no static P0 model blocker. The
  project accepts this only within scope: detector supervision is detached
  selection-policy utility, signed proximal correctness is a local score-space
  identity, utility includes selected-axis geometry, and the current gate is
  synthetic. A real THUMOS loader/DDP/train-engine/AMP-replay/EMA gate plus an
  informative-utility check is mandatory before any pilot or full train.
  Status remains `tested`; C3/C4/C7 remain unproven.

- 2026-07-16: Split the oversized DUCA `7525efb` Pro review prompt into two
  dependency-ordered discussions after the combined version exceeded the Pro
  model's reasoning window. Round 1 is a bounded code, mathematics, gradient,
  coordinate, AMP/DDP, and gate audit and must emit a structured
  `HANDOFF_PACKET`; Round 2 accepts only that exact-commit packet and decides
  the final method, supervision, matched experiment DAG, cost claim, and
  publication route without repeating the full line-by-line audit. Artifacts:
  `docs/methods/prompts/2026-07-16-7525efb-duca-pro-round1-code-math-audit.md`
  and
  `docs/methods/prompts/2026-07-16-7525efb-duca-pro-round2-method-paper-verdict.md`.
  This workflow change does not alter implementation, experiment, or claim
  status.

- 2026-07-15: Generated the exact-commit Pro review artifact
  `docs/methods/prompts/2026-07-15-7525efb-duca-signed-utility-exact-commit-pro-audit-prompt.md`.
  It binds the sole audit target to GitHub commit `7525efb`, separates code
  facts from unverified remote reports, demands mathematical and line-level
  review of signed proximal utility, and forbids new routes before the
  fixed-384 mechanism has real-loader/CUDA/matched evidence. Prompt generation
  changes no experiment or claim status.

- 2026-07-15: Pushed DUCA signed score-space counterfactual repair commit
  `7525efb2e07214615a59c482443246174a6adaf1`. Gate `1165646` had invalidated
  the prior candidate-relative surrogate. The new objective uses swap-incidence
  Gram whitening to align actual selector score descent with detached signed
  detector loss reduction. Follow-up fixes force FP32 under AMP, reject invalid
  swap indices, make mixed/all-short audits finite and per-sample, restore all
  relevant RNG families, and bind a clean gate to core file hashes. Clean
  remote focused verification is `160 passed, 7 skipped`. Exact-commit CUDA,
  real-loader, forced-overflow pilot, matched mAP and C3/C4 evidence remain
  pending; status is `tested` only.

- 2026-07-15: Archived and absorbed the 86,173-byte S1/DUCA exact-commit Pro
  audit with byte-identical SHA-256
  `AC54C5B633DC9FD0CD801B2B12B2C4E44114E16B7569C220B77528674E2D04E2`.
  Independent code inspection confirmed the review's core `STOP_AND_FIX`
  findings. Live DUCA logs then made its P0 concern definitive: Jobs
  `1164700-1164703` were already 3-4 successful updates behind their exposed
  batch counts around epochs 24-25, so the fixed 132-epoch suite cannot produce
  the declared 13,200 matched updates and is diagnostic-only. All nine S1 jobs
  were still healthy and 31 AMP retries had recovered, but every cell emitted
  nondeterministic CUDA linear-upsampling warnings (221 total), invalidating
  strict deterministic-formal status while keeping the sealed test closed. The
  absorption accepts the core findings but does not blindly adopt fail-on-skip,
  Bayesian bootstrap, terminal-EMA, or numerical LCB thresholds as unique or
  already validated fixes.

- 2026-07-13: Completed the authorized Spatial Zoom S1 infrastructure and its
  independent Max audit loop. Added repository-frozen checkpoint identity,
  real-writer sidecar validation, deterministic 9-descriptor GO/KILL rebuild,
  a preregistered study-level once-only test lock, frozen 3x3 profile order,
  and identical hardware/software preflight before test. Local verification is
  `46 passed` (`26` S1 tests), with compile, config validator, static precheck,
  launcher syntax, and diff checks passing. Reviewer
  `019f5b3c-9b7f-73f2-8eea-00157a60a119` (`gpt-5.6-sol`, max) ended at
  `PASS_BEFORE_REMOTE_TRAINING`, no P0/P1/P2. Formal CUDA/full train/test/cost
  remain unrun; status is infrastructure `tested`, not S1 GO, and S2 stays
  locked.

- 2026-07-13: Implemented and locally verified the authorized S1
  infrastructure: matched configs, resolved-config drift validation, frozen
  manifest/seeds, static/clip/full prechecks, trained-checkpoint profiling,
  immutable run descriptors, and paired video-cluster AP/bootstrap gating.
  Required regression is `28 passed`; S1 tests are `8 passed`; static geometry
  passed. Local real-clip execution is blocked by the known Windows `c10.dll`
  failure, so formal CUDA full-window validation remains pending. This is
  infrastructure-level `tested`, not S1 GO or permission to implement S2/ROI.

- 2026-07-13: Archived the complete 84,533-byte Dense-Time Spatial Zoom Pro
  review with byte-identical SHA-256
  `667A319CA2ABB0601EE0D6A76DF9D8D139D1F116A7BD93D55B48CFA2DC655650`
  and wrote a structured absorption. Accepted `HOLD`, S1/S2-first, AdaSpot as
  the closest baseline, regular detector time, native small crops, and strict
  full-stack cost. Corrected the branch-visibility claim and reclassified the
  S2 dense-teacher oracle as privileged gate-only evidence. The route is now
  `designed` only at gate level: S1 infrastructure is authorized; S2 and the
  over-specified DART-Zoom model remain locked. No experiment code or job was
  started in this turn.

- 2026-07-13: Created the read-only Pro discussion prompt for Dense-Time
  Spatial Zoom. It requires explicit skill and repository visibility
  certificates, primary-source comparison with Uni-AdaFocus/AdaSpot, strict
  dense-resolution and equal-total-cost oracle-ROI kill gates, four mutually
  exclusive routes, one conditional final architecture, implementation
  mapping, full-stack profiling, result-to-claim outcomes, and an adversarial
  CVPR/ICCV review. No method code, experiment, deployment, or route-status
  promotion was performed.

- 2026-07-13: Discussed a route-level pivot from temporal frame deletion to
  dense-time spatial zoom. Registered Uni-AdaFocus and AdaSpot as primary
  neighbors and created `idea:dense-time-spatial-zoom-tad` with status
  `discussed/oracle_gate_required`. The route keeps the full temporal lattice,
  uses low-resolution global context plus temporally coherent high-resolution
  ROI tubes, and may avoid DUCA's coverage/max-gap/selected-axis failures.
  Direct Uni-AdaFocus/AdaSpot transplantation is rejected as a main novelty.
  No implementation or experiment was started; dense-resolution headroom,
  oracle-ROI sufficiency, context/concurrency, temporal stability, and strict
  total-cost gates must pass first.

- 2026-07-13: Archived and absorbed the `1fc7037` DUCA Pro review (SHA-256
  `DDBC15BC20BFDD503FAA2DA4832093325EB2D8997E4A685638DFF46F90CC780D`).
  Local source verification accepts REDESIGN for the current global Top-K/G15/
  soft-RGB-bridge route and records DUCA-CellCF as a discussed bounded appeal,
  not an implemented final model. Corrected the epoch-89 diagnostic wording:
  learned scores trail the `abs_delta + uncertainty_peak` compound proxy; pure
  delta remains unmeasured. Recorded reservations about fixed-grid time
  mismatch, non-additive per-cell utility, proposed thresholds, and the
  fixed-384 pilot kill rule. Paper status remains HOLD.

- 2026-07-13: Completed a parallel purpose, code, protocol, and literature
  audit of the DUCA-to-Oracle gap. Recorded that Oracle Job `1001959` uses
  test-time GT boundary neighborhoods and a label-dependent selected-axis
  remap, so 76.67 is privileged evidence rather than a learnable ceiling.
  Static review found an unnormalized coverage loss that can reward local
  clustering, a remaining direct-arm midpoint-uniform bug, and an unvalidated
  soft detector-gradient surrogate. The deployable learned route remains
  `unproven`; matching a practical dense model is plausible only with
  coverage-preserving selection and matched full-stack evidence. Added
  `experiments/duca-oracle-gap-reachability-audit.md`; paper status stays HOLD.

- 2026-07-13: Completed selector-only quality export and analysis for legacy
  DUCA beta=0 EMA epoch 89 (Job `1161079`, 211 videos, 487 windows). Coarse
  AUROC/AUPRC is 0.621/0.411; learned transition ranking trails raw delta-p;
  learned selection improves exact-r0 recall by 1.53 points but loses 15.55
  points at r1 versus exact uniform and is worse in 308/487 windows. Recorded as
  `tested/diagnostic`, not empirical support or paper evidence. Added
  `experiments/duca-selection-quality-epoch89.md` and anti-repetition rules.

- 2026-07-13: Updated the GitHub Pro audit prompt at documentation-only commit
  `b38080d` with the completed legacy diagnostic table. Exact method code remains
  `0ea4e15`; no model or experiment protocol changed.

- 2026-07-13: Strict result-to-claim gate for legacy DUCA P0
  `exp:duca-transition-only-fixed384`: `C3=no`, `C4=no`, confidence `high`.
  Runtime execution passed, but protocol integrity failed because Job `1159414`
  was not exact-uniform and Jobs `1159416/1159417` did not execute the intended
  homotopy. This is failure to support the claims, not refutation. Reviewer
  delegation was unavailable, so the local verdict is pending external review.

- 2026-07-13: Legacy invalidated DUCA P0 jobs `1159414-1159417` all completed
  with exit 0, 17 evaluations each, empty stderr, and no Traceback/OOM/runtime
  non-finite skip. Best diagnostic Avg-mAP was 55.67/57.71/64.34/63.55;
  beta0.25 trailed beta0 by 0.79 Avg-mAP and showed no stable high-IoU gain.
  The runs support numerical stability only. C3/C4 remain unproven, and the
  corrected `0ea4e15` formal gate/matched runs are still missing.

- 2026-07-12: Invalidated DUCA P0 monitoring reached epochs 126/126/131/131
  after 24h24m. Best diagnostic Avg-mAP was 55.67/57.71/64.34/63.55; all
  stderr files remained empty. Jobs had not exited, and C3/C4 remain unproven.

- 2026-07-12: Legacy invalidated DUCA P0 jobs remained healthy after 22h23m at
  epochs 118/116/121/121. Best diagnostic Avg-mAP became
  55.67/57.60/64.34/63.55 for invalid-alpha0/direct/beta0/beta0.25; stderr
  remained empty. Beta0 numerically reaches the historical 64.352 anchor but
  does not establish superiority because the matched uniform and homotopy are
  invalid. C3/C4 remain unproven.

- 2026-07-12: Published the complete DUCA transition-only Pro audit prompt at
  branch head `855949f`; the exact method-code target remains `0ea4e15` and the
  new commit changes documentation only. The prompt forces file/line visibility,
  resets invalid uniform/homotopy evidence, audits protected gradients,
  structured hard/soft alignment, selected-axis AdaTAD geometry and total cost,
  and requires a GO/HOLD/KILL verdict plus one implementation-ready final route.

- 2026-07-12: Legacy invalidated DUCA P0 jobs `1159414-1159417` remained
  healthy after 14h20m at epochs 82/81/86/86. Best diagnostic Avg-mAP was
  55.50/56.34/63.98/63.55 for invalid-alpha0/direct/beta0/beta0.25; all
  Slurm stderr files were empty and no hard failure was found. C3/C4 remain
  unproven because the uniform control and homotopy start are invalid.

- 2026-07-12: Invalidated the `8bfc0e5` DUCA P0 exact-uniform control. At
  T=768/K=384 its midpoint-distance reference logits collapse to one value,
  and Viterbi emits a tie-break path with only 47.135% overlap with rounded
  endpoint linspace and 179.695-frame mean rank error. Job `1159414` best
  Avg-mAP 55.46 is therefore not a uniform baseline; Jobs `1159416/1159417`
  also do not implement the intended uniform-to-learned homotopy. Historical
  true-uniform anchors were located at 64.352 (`1150701`) and 65.696
  (`1150842`), but use unmatched protocols. Commit `0ea4e15` fixes the
  reference, adds exact decoded-position gates, passed 26 remote focused tests
  with 2 skips, and is not yet formally gated or full-trained.

- 2026-07-12: Completed the PhysTime performance-acceptability audit. The
  deficit separates into a 4.68 Avg-mAP sparse-observation gap from the local
  dense AdaTAD anchor (68.29 to selected-axis 63.61), plus a further 6.40-point
  PhysTime geometry/head gap (63.61 to 57.21). Published AdaTAD VideoMAE-S at
  the matched 768-frame scale reports 68.8, while larger backbone/data settings
  reach 76.9; absolute SOTA is therefore neither matched nor achieved. The
  selected-axis result is acceptable as a sparse control, but PhysTime v1 is
  only a clean negative mechanism result, not a paper-ready method.

- 2026-07-12: PhysTime-AdaTAD raw-video K=384 matched jobs `1159493-1159495`
  completed 60 epochs with exit 0 and no AMP skips, non-finite values, OOMs,
  or tracebacks. Best selected-axis/physical-grid/PhysTime Avg-mAP was
  63.61/59.14/57.21; tIoU-0.7 mAP was 41.87/32.34/34.96. Independent
  result-to-claim verdict is `no` with high confidence: PhysTime's narrow
  +2.62 tIoU-0.7 gain over physical-grid does not offset its -6.40 Avg-mAP and
- 2026-07-13: Released the long-running Zhongwei debug GPU hold Job `1118197`
  (`pcot_dbg2g`, `gres/gpu=2`, node `g0030`, runtime 18-05:47:37) by
  `scancel`. Slurm records it as `CANCELLED by 1258`; transient `COMPLETING`
  cleanup may remain until the scheduler finishes node cleanup. Left Job
  `1137541` untouched because its WorkDir is `/data/run01/sczc063/wangruofan`,
  not the current C3/DUCA/yuzibo debug allocation.

  -6.91 tIoU-0.7 deficit versus selected-axis. Phase 2 is not unlocked and the
  automation was retired.

- 2026-07-12: DUCA transition-only P0 reached five scheduled evaluations
  through logged epoch 71. Best-to-date Avg-mAP is 55.46/56.15/63.93/63.55
  for uniform/direct/beta0/beta0.25. At the latest evaluation beta0.25 trails
  beta0 by 0.38 Avg-mAP but leads at tIoU 0.7 by 42.28 vs 41.33; its bridge is
  now near the 0.25 target. Jobs remain healthy and `experiment_running`.

- 2026-07-12: Updated the running PhysTime-AdaTAD K=384 comparison at matched
  epoch 53. Selected-axis/physical-grid/PhysTime Avg-mAP was
  63.16/58.92/56.84 and tIoU-0.7 mAP was 41.18/32.12/34.18. By 09:34 CST the
  jobs had reached epochs 55-56 with zero executed AMP skips, non-finite
  values, OOMs, or tracebacks. Selected-axis additionally reached 63.27 at
  epoch 55; status remains `experiment_running`, with no PhysTime advantage
  claim.

- 2026-07-12: DUCA transition-only P0 completed three scheduled evaluations
  through logged epoch 61. Latest Avg-mAP is 55.11 exact-uniform, 55.82
  direct-a5, 63.74 transition beta=0, and 63.21 beta=0.25; latest tIoU-0.7 is
  33.48/31.97/41.85/40.86. All jobs remain healthy around epoch 65-66 and the
  beta=0.25 bridge is still ramping, so status remains `experiment_running`.

- 2026-07-12: Recorded the first matched PhysTime-AdaTAD raw-video K=384
  evaluation for commit `3ac93a1`, jobs `1159493-1159495`. At the epoch-45
  evaluation, selected-axis/physical-grid/PhysTime Avg-mAP was
  62.90/58.21/56.03 and tIoU-0.7 mAP was 40.87/31.01/33.45. All three jobs
  remained healthy at epoch 47 with zero executed AMP skips, NaNs, OOMs, or
  tracebacks. Status remains `experiment_running`; the current checkpoint does
  not support a PhysTime superiority claim.

- 2026-07-12: Recorded the first scheduled seed-0 evaluation for DUCA
  transition-only P0 jobs `1159414-1159417` at commit `8bfc0e5`. Interim
  Avg-mAP was 53.68 exact-uniform, 55.77 direct-a5, 63.07 transition beta=0,
  and 62.02 transition beta=0.25; corresponding tIoU 0.7 mAP was
  31.35/32.19/41.28/39.98. All jobs remained healthy around epoch 56, so the
  experiment stays `experiment_running` and no paper claim is upgraded.

- 2026-07-11: DUCA transition-only current HEAD advanced to `8bfc0e5` after
  diagnosing real-data AMP padded-window NaNs. Gate `1159395` completed with 37
  passed/1 skipped and real T768/K384 GradScaler detector-only proof. Submitted
  hash-bound P0 jobs `1159414-1159417`; status is `experiment_running`, with no
  mAP result yet.

- 2026-07-11: Superseded DUCA transition-only `fc98eca` with hardening commit
  `8e38cca`. Fixed hidden-semantics baseline drift, real detector-loss gradient
  attribution, delayed learned-policy validation, honest train-vs-inference
  provenance, pre-DDP freezing, matched component LRs, and hash-bound launchers.
  Submitted only formal CUDA gate `1159383`; no P0 full train is queued yet.

- 2026-07-11: Implemented and tested the isolated Shared-ASFormer
  Transition-Only fixed-384 candidate. Slurm gate `1159350` completed with 26
  focused tests and official ActionFormerHead gradient/train/test proof. Status
  is `tested`; full-train evidence remains pending.

- 2026-07-11：初始化 C3/DUCA research-wiki。
- 2026-07-11：逐轮读取主任务 191 轮，归档 158 条用户侧原始消息。
- 2026-07-11：登记实现代理、论文代理和早期目标任务的近期记录。
- 2026-07-11：登记 C3、PAction、GAS-VT、lattice、detector-aware、TrueTime、
  DUCA、MUST、X3D/SlowFast、physical-grid、CFPA、CVCR、ChronoTransport、
  PhysTime 路线。
- 2026-07-11：冻结当前裁决：70aa069 是待裁决 DUCA baseline，a5e1774 是最新
  审计代码；正式论文 claim 尚未闭环。
- 2026-07-11：wiki lint 通过：16 ideas、7 experiments、10 claims、47 edges、
  0 orphan nodes、0 curated broken links；query pack 2825 chars。
- 2026-07-11：纠正 ChronoTransport 过期状态：`92029ea` formal Stage-B P3 science gate 为负，Stage C/P5 未解锁；新增独立 negative experiment 节点，路线暂停。
- 2026-07-11：完成 DUCA 新颖性复核。裁决为“具体组合新，但核心构件已有密集近邻，当前呈现偏组合式创新”；若决定性门槛失败，优先转向以执行单元反事实 detector regret / measured cost 为目标的 CVCR+BCFT，而非继续增加 selector loss。登记 AdaFrame、ASL、AdaTAD、TE-TAD、TAPS 和 Progressive Block Drop 六项近邻来源。
- 2026-07-11：wiki 增量 lint 通过：16 ideas、8 experiments、10 claims、51 条有效 JSONL edges；query pack 3256 字符，未引入新孤立节点。
- 2026-07-11：DUCA `70aa069` fixed-384 Job `1154971` 正常完成 60 epoch，最终/最佳 Avg-mAP 58.39，IoU-wise 76.26/71.06/61.20/48.90/34.53；无训练硬错误。结果仍为 unmatched：同提交 baseline 与 full-stack cost 缺失。记录 120 个预算摘要中 35 个 effective K<384，最低 214，需审计后再解释 fixed-384。
- 2026-07-11：登记当前 ChronoTransport task `019f4ae5-93dd-7381-8203-42360125b41b`；
  吸收 MoD 差异、重算语义、16-frame/tubelet/token 粒度、旧 p_action 污染与在线误判纠正。
- 2026-07-11：完成 ChronoTransport 最近文献查新，新增 7 个核心 paper 节点；路线
  novelty 暂评 `4.5/10`，维持暂停并只允许一次有界 P3 修复。
- 2026-07-11：登记三个 proposed 候选：Boundary-Adaptive Temporal Multigrid、
  Counterfactual Value-of-Computation、Spectral Innovation Operator；均未实现或测试。
- 2026-07-11：完成 ChronoTransport Result-to-Claim：H1 unsupported、H2 partial、H3 no、
  H4 unverified，完整主张 verdict=`no`。冻结唯一上诉协议 `CT-P3R-3S`；第一道失败
  gate 即永久降级 baseline，禁止第二轮 head/loss/权重搜索。
- 2026-07-11：完成 DUCA 58.39 与历史分离训练/均匀采样差距的根因审计。发现训练
  exposure 不匹配（5940 vs 13080 optimizer steps）、selector raw losses 接近 chance、
  coarse probe 无已验证 checkpoint、hard-forward/soft-backward utility 未对齐，以及
  irregular selected-axis geometry 风险。历史 lattice 可审计最佳 63.18、PAction 最佳
  61.02；uniform 约 65 尚缺同协议成功 provenance。冻结选择重训、matched uniform、
  one-swap gradient audit 和 geometry control 被列为决定性诊断，禁止先调 loss 权重。
- 2026-07-11：登记 DUCA 联合训练优化提案为 `designed`、未实现：单模型连续同伦、
  exact-K/max-gap 同构 hard/soft structured policy、模块化梯度路由、detector-priority
  conflict projection，以及通用预训练 MobileNetV3 + 官方 ASFormer 的单次联合微调。
  fixed-384 未通过 matched uniform、one-swap 和 geometry gate 前不开放 dynamic MUST。
- 2026-07-11：纠正 coarse 组合表述：MobileNetV3 + 官方 ASFormer 不是默认必选的两个
  模型。ASFormer 需要 spatial features，且与现有 selector temporal encoder 可能重复；
  默认先验证 MobileNet-only coarse，ASFormer 仅可替换 selector temporal encoder 成为
  共享 trunk，并以边界收益和 full-stack 成本决定是否保留。
- 2026-07-11：澄清 DUCA 成本 claim 仍为 unproven。MobileNetV3 相对当前随机两层 stem
  必然增加 dense probe 成本；768→384 提供约 50% heavy-backbone 理论空间，但是否净省
  必须由 trained-checkpoint full-stack p50/p95/energy 和 accuracy-cost Pareto 裁决。
- 2026-07-11：用户批准将 ChronoTransport 唯一上诉 `CT-P3R-3S` 固化为书面验证规格；
  当前仍为 `designed`、等待书面规格复核，尚未实施或启动实验。协议冻结三 seed、共享
  split、唯一窗口级 quantile head、selection-safe simultaneous calibration、equal-cost
  oracle/mechanism/risk/full-stack 四道顺序 gate，首个失败即永久降级 baseline。
- 2026-07-11：审计 standalone vs joint official ASFormer。二者可使用同一 probe class，
  主要差异是纯 BCE/best checkpoint/稳定输入/13080 steps 对 actionness=0.05/动态选择/
  surrogate gradient/5940 steps。当前裁决为先修 joint optimization，MobileNet 非必需且
  暂不加入；仅在同架构修复后 coarse 指标仍不足时做 cost-aware spatial front-end 消融。
- 2026-07-11：用户指出 shared ASFormer 直接 boundary head 会违背间接定位初心。已纠正
  设计：coarse 只学 action/background state；transition selector 只从 deploy-visible
  状态变化证据间接评分；GT boundary 仅作 train supervision，禁止直接边界粗模型与
  absolute-hidden bypass。旧图对应实现暂停。
- 2026-07-11：生成面向 Pro 的 DUCA joint-ASFormer/间接边界全面审查 prompt，锁定公开
  branch HEAD `a5e1774` 与结果 commit `70aa069`，要求逐行代码证据、根因排序、最终架构、
  数学训练目标、patch 级实现、one-swap/geometry/cost gates 与结果到 claim/kill matrix。
- 2026-07-11：完整归档并吸收两份字节相同的 Pro 回复，SHA256
  `011EBB67CC52D943248D18E4638E2220763DED44329BEF8EB78DBD77973BE863`。裁决 `HOLD`；
  本地复核 hidden=pre-ASFormer stem、absolute/direct-head bypass、same-family exact DP、
  duty-cycle curriculum 与 full optimizer coverage 缺口。冻结 a5e，唯一下一候选为
  Shared-ASFormer Transition-Only fixed-384；MobileNet/MUST/新增 heads 继续冻结。
- 2026-07-11：固化 CT-P3R-3S Pro 路线核验与代码生成主控 Prompt。Prompt 只授权严厉
  路线/统计/代码审查、官方 GitHub 对照、优化 patch、本地验证证据和下一步计划；禁止
  当前工作区写入、远端连接、GPU 训练、部署、push 或 PR，不改变方法/实验状态。
- 2026-07-11：完整归档并吸收 ChronoTransport CT-P3R-3S Pro review；三个重复附件
  SHA256 均为 `E7971A22044B384092B833A1137F8EC0B543B504D271078CBCB4198F96D35CAF`。
  裁决 `REVISE_SPEC_BEFORE_CODE`：`b74101d` 不可原样执行，下一合法协议为
  `CT-P3R-3S-r1`。吸收 window-vector rank、unique-window bootstrap、coverage 单边下界、
  oracle-headroom claim、完整 total latency 与 successful-update 合同；r1 冻结前不写
  争议代码、不跑 profiler/Gate 1、不训练新 seed、不解锁 Stage C。
- 2026-07-11：完成 CT-P3R-3S-r1 本地源码复核。接受 Pro 的 window-vector rank、
  simultaneous window calibration、单边 coverage、full-stack total samples 与
  successful-update 合同；否定当前 Stage-B `lr=0`、head EMA 顺序污染、CT 与 packed/
  DUCA 同时激活三个推测。确认 cell-sum/row-statistics、per-seed split、残缺 library、
  provenance、Stage C/P5 等缺口，并新增发现：dense TIA 只写回 RECOMPUTE rows，
  skipped rows 在有效语义上绕过 adapter。当前状态仍为 spec revision，未写争议代码。
- 2026-07-11：按用户要求完成空白上下文独立 agent 两轮复核，最终 verdict 同为
  `LOCAL_CORRECTED_R1`。独立确认 pre-adapter heavy cache + all-row TIA，并新增 P0：
  `max_cache_age=8` 与 hold/transport-only 47-clip 连续复用冲突；runtime repair 后 nominal
  requested cost 不能代表 executed cost。另冻结 Stage-B FP32 事实、140/16 跨 seed exposure
  ledger 与 EMA alias equality 审计。仍未进入实现或实验。
- 2026-07-11：用户批准独立复核后的 ChronoTransport r1 设计；完成 536 行正式书面规格
  `docs/superpowers/specs/2026-07-11-chronotransport-ct-p3r-3s-r1-design.md`，单文件提交
  `02199f8`，SHA-256
  `871420261BD1C19CC515218A6016A91ED7D553B73740AB41C2E02AA7F96609F9`。状态升级为
  `written_spec_pending_user_review`，尚未进入 implementation plan、代码或实验。
- 2026-07-12：按用户要求把 commit `02199f8` 交给新的空白上下文 agent 两轮核验。
  最终 verdict=`REVISE_SPEC_BEFORE_PLAN`：确认 all-row TIA、cache-age 与 cost 修复，但发现
  simple offsets 的 mod-4 candidate/video confounding、video/window conformal unit 错位、
  Stage-C loss-specific autograd 与 AMP retry state 未闭合、Gate-1 shuffle tautology和
  checkpoint/data identity 未预冻结。本地复算验证 block-rotated assignment 为可行最小修复；
  当前不调用 writing-plans，不修改模型或运行实验。
- 2026-07-12：为无法读取本地工作区的 Pro reviewer 建立 GitHub 固定提交审查入口；仅同步
  ChronoTransport r1 规格、实现表面、原 Pro 记录、两轮独立复核与本地源码审计，明确排除
  数据、checkpoint、GPU 日志和新行为结果。审查仍止于 `REVISE_SPEC_BEFORE_PLAN`，不得借
  GitHub 同步越过到实现、profiling、Gate 1、新 seed 或 Stage C。
- 2026-07-13：完整归档并吸收附件 `6065c548...` 的 `0ea4e15` DUCA exact-commit Pro
  审计，原附件 SHA256 为 `60D4D9414F3F2D90EC9A0CE0F2D704D2184D8EEED9CE2FBB5315932997CEE957`。
  裁决维持 `HOLD`；新增确认 direct/legacy midpoint 残留、raw-pixel bridge utility 未对齐、
  selected-axis/short-window/全栈成本风险。登记 DUCA-FSU 为 `discussed` 条件候选，明确
  counterfactual utility distillation 不等于 detector-gradient backprop，未改变 C1/C3/C4
  的 unproven 状态，也未授权 full train。
- 2026-07-13: Strengthened the authorized Spatial Zoom S1 infrastructure
  without implementing S2/ROI/scout/teacher/policy. Dense160 is now bound back
  to the official-derived local baseline; checkpoint choice requires a hashed
  frozen-gate prediction proof; statistical pooling resamples both training
  seeds and paired video clusters. Local verification is `29 passed` (`9` S1
  tests), validator/static precheck PASS. CUDA full-window and all empirical S1
  runs remain pending, so the route is still `designed` and not S1 GO.

- 2026-07-13: Finalized and independently audited DUCA transition-only commit
  `1dae0d7`. Formal CUDA gate `1161481`, focused 106-pass suite, and GPU FP32
  auxiliary-loss Job `1161480` passed. Submitted corrected matched P0 Jobs
  `1161482-1161485`; all four entered RUNNING. State is `experiment_running`;
  C3/C4 remain unproven.
- 2026-07-13: Real full-train startup invalidated the `1dae0d7` suite. Jobs
  `1161482-1161485` failed because generated sbatch files overwrote Slurm's
  remapped single GPU. Gate `1161489` exposed official-ASFormer FP16 backward
  NaNs; Jobs `1161492` and `1161494` exposed an unbound direct calibration
  declaration and invalid short-window one-swap handling. None are results.
- 2026-07-13: Commit `e8f4460bd9dc2419f3df7838d7406249a8fb8961`
  repairs those paths. Focused GPU Job `1161498` passed 42 tests; formal gate
  `1161499` passed; independent GPT-5.5 xhigh audit found no P0/P1. Matched
  seed-0 Jobs `1161505-1161508` entered RUNNING. Status remains
  `experiment_running`; C3/C4 and performance claims remain unproven.
- 2026-07-13: Completed the second hardening pass for authorized Spatial Zoom
  S1 infrastructure after independent Max round 2 remained
  `FAIL_BEFORE_REMOTE_TRAINING`. Corrected the one-sided max-T pivot, removed
  cost vetoes from the accuracy GO/KILL decision, added preregistered-SHA and
  exact VideoMAE core-load proof, made positional interpolation calls exact,
  reused the official world-size-one DDP gather/NMS/output path in profiling,
  persisted bounded-gap raw power traces, and enforced global 3x3 profile
  comparability. Local verification is `38 passed` (`18` S1 tests), validator,
  static precheck v3, shell syntax, compilation, and diff checks PASS. A third
  read-only Max review is pending; formal CUDA full precheck and all S1 runs
  remain unrun, so this is infrastructure `tested`, not S1 GO or S2 permission.
- 2026-07-13: Invalidated the `e8f4460` four-arm run after Job `1161508`
  exposed a batch-varying DDP static graph. Diagnostics `1161536-1161539`,
  `1161545`, and `1161548` ruled out find-unused/reentrant,
  non-reentrant-on-PyTorch-2.0.1, and static-graph-disabled workarounds. Commit
  `40eb86ee69e19b3105f9ddd6a977fb7693f724ad` now keeps the all-short
  counterfactual scorer path with a finite FP32 connected zero; formal CUDA
  gate `1161590` passed. Exact-commit independent audit and a DDP startup pilot
  remain pending, so no replacement full train is yet deployed.
- 2026-07-15: Hardened the corrected DUCA P0 deployment through commits
  `f048c31`, `34b9106`, `77fc7a4`, `a5b3c67`, `51330c8`, and current
  `cff479e`. Gates `1162051`, `1162124`, `1163435`, and `1163439` passed;
  `1163434` exposed a test-path mismatch and pilot `1163437` fail-closed on a
  preparation/runtime config-hash drift before training. Current `cff479e`
  binds a canonical environment TSV/SHA and byte-compares it at runtime.
  Local P0 tests are 27 passed and required C3 checks are 23 passed. A fresh
  exact-commit audit, CUDA gate, and counterfactual-only DDP pilot are pending;
  no corrected full train is running and C3/C4 remain unproven.
- 2026-07-13: Completed the third provenance hardening pass for authorized
  Spatial Zoom S1 after independent Max round 3 returned
  `FAIL_BEFORE_REMOTE_TRAINING`. Added annotation-deterministic manifest
  reconstruction, a repository-frozen VideoMAE-S filename/SHA across the full
  evidence chain, per-cell atomic profile-start markers, node/GPU/CPU/software/
  decode-stack identities, and a shared-precheck identity requirement before
  complete 3x3 test opening. Local verification is `41 passed` (`21` S1
  tests), with compilation, validator, static precheck, shell syntax, and diff
  checks PASS. Post-remediation Max review is pending; formal CUDA full
  precheck and every S1 train/test/profile run remain unrun, so status is only
  infrastructure `tested` and S2/ROI/policy remain locked.
- 2026-07-15: Closed the PhysTime G1a native-J192 heartbeat as completed and
  recorded pilot results. Jobs `1162048-1162050` completed with exit code 0.
  Selected-axis six-epoch pilot reached Avg-mAP 10.26 / mAP@0.7 1.09;
  physical-metric reached Avg-mAP 10.56 / mAP@0.7 1.04. This is only a weak
  low-IoU signal and not paper evidence. The active continuation is sparse
  downstream detector-head diagnosis, not DUCA selector work.
- 2026-07-15: Real-batch DUCA DDP diagnostics invalidated the remaining static
  protocol. Jobs `1163456`, `1163460`, and `1163471` failed when parameter use
  changed across batches; Job `1163472` completed ten real optimizer updates
  under `with_cp=False/static_graph=False/find_unused_parameters=True` and
  failed only at checkpoint save because `/data` was full. Current pushed
  commit `28908e2de974ff90fe1e16e8f12a02085742f9f7` applies that protocol to
  all four matched arms, adds a four-arm ten-step machine-readable DDP pilot,
  and blocks formal suite deployment without commit/core-gate/protocol-bound
  pilot evidence. Local and clean-remote focused verification is 62 passed,
  1 skipped. Storage now has about 277 GB free and the five-epoch checkpoint
  interval remains unchanged. Independent review and fresh CUDA evidence are
  pending; no formal four-arm job has been submitted.
- 2026-07-15: Removed S1's physical-GPU-1 binding and deployed the formal S1
  matrix through normal Slurm. The earlier `911448a`
  matrix exposed fail-fast AMP skips and static-port collisions; `7d1e9cc`
  still missed forward-mutated buffer rollback. Commit `9298c0e` restored RNG
  and model buffers while preserving scaler backoff and uses kernel-assigned
  c10d ports, but pilot `1164261` then failed its first checkpoint because
  runtime config mutation was misclassified as protocol drift; Jobs
  `1164267-1164274` were cancelled. Current commit `35204f5` isolates all four
  legal runtime config mutations in deep copies. Full precheck Job `1164289`
  passed on Slurm-selected physical GPU 4; pilot `1164291` completed two epochs
  and produced a validated checkpoint/sidecar with 160 successful updates and
  two AMP skips. Jobs `1164307-1164314` complete the replacement 3x3 matrix;
  all nine current cells entered RUNNING without a fatal startup pattern.
  Status is `experiment_running`, not S1 GO or empirical support; S2 remains
  locked.
- 2026-07-15: Tested a PhysTime G1a rank-assignment diagnostic in the sparse
  detector-head worktree. Remote focused tests passed (`20 passed in 44.77s`),
  but real THUMOS geometry was negative: validation
  `physical_time_rank_assignment` produced 7745 positives / 18.96% GT
  no-eligible / 65.16% `<1s` no-eligible, worse than physical-time seconds and
  uniform-rank seconds. Train-s5 likewise produced 14201 positives / 25.13%
  no-eligible / 72.02% `<1s` no-eligible. Conclusion: swapping assignment
  reference axes alone cannot fix G1a because physical anchor centers can be
  absent from short GT; next sparse-head work must separate query anchors from
  sparse observation support instead of long-training this variant.
- 2026-07-15: DUCA commit `18dc1cd` fixed no-grad counterfactual teacher
  contamination of CUDA autocast's cast cache; exact gate `1164279` passed.
  Four-arm pilot `1164286` then completed uniform but failed direct-a5 on a
  real mixed-length structured-slot mass invariant. Commit `043be401` aligns
  hard/soft DP to each sample's valid prefix/effective K/max-hole and adds
  direct/transition mixed-length regression tests. Clean Linux verification is
  `122 passed, 5 skipped`; independent max review is GO with no P0/P1. Exact
  gate `1164318` and four-arm pilot `1164319` completed with exit code 0. The
  hash-bound 132-epoch suite is now running as Jobs `1164700-1164703` for
  uniform/direct/transition-beta0/transition-counterfactual. This upgrades only
  deployment status to `experiment_running`; no mAP exists and C3/C4 remain
  unproven.
- 2026-07-15: Generated the exact-commit, evidence-first Pro audit request
  `docs/methods/prompts/2026-07-15-s1-duca-exact-commit-pro-audit-prompt.md`.
  It treats Spatial Zoom S1 commit `35204f5` and corrected DUCA P0 commit
  `043be401` as independent review objects, labels live Slurm status as
  externally supplied and unverified by GitHub, forbids reuse of invalid or
  protocol-unmatched historical results, and requires file-line findings,
  concrete patches/tests, a minimal execution DAG, and result-to-claim gates.
  No method, experiment, or claim status changed in this prompt-generation turn.
- 2026-07-15: Initial formal DUCA startup check found all Jobs `1164700-1164703`
  at real epoch-0 step `50/99`, finite losses, about 8.68 GB logged memory, and
  no fatal scan hit. Uniform and beta0 are intentionally identical during
  policy warmup; counterfactual adds a finite distillation term, while direct's
  larger total includes extra endpoint/context/boundary losses. These totals
  are not cross-arm performance metrics. Although the config fields use
  val-start/anchor 47, the exact mixed zero/one-based code first evaluates
  after one-based epoch 52.
- 2026-07-15: Archived and absorbed attachment `48c9c615...`, the exact-commit
  Pro audit of DUCA `043be401`, under SHA-256
  `1D395844396D644295BF83BF08753C14B2E638295B8C37D15048924B0F415FC9`.
  Local recheck accepts GO-to-finish/HOLD-claims, the unsealed test-checkpoint
  risk, relative-only counterfactual semantics, first evaluation at one-based
  epoch 52, and selected-axis geometry/cost gaps. Absorption is
  `PARTIAL_ACCEPT`: no-op softmax is not uniquely privileged, world-size-two is
  not required for the current one-GPU protocol, and physical-time head work
  remains gated. Live monitoring at 18:55 CST found uniform/direct/beta0 at
  epoch 16 and counterfactual at epoch 15, checkpoint 14 present, no mAP and no
  fatal scan hit. C3/C4 remain unproven.
- 2026-07-15: Before any DUCA `043be401` test mAP existed, sealed the
  study-level primary-result protocol at 2026-07-15T12:16:50Z. Artifact
  `docs/methods/2026-07-15-duca-043be401-primary-result-protocol.json` and all
  five remote copies have SHA-256
  `AAC0FCA8671AE6F58CF4C9B5D4D40282BE714AA354028246E86504FD39C89B48`.
  Final one-based epoch 132 `state_dict_ema` is the sole primary result;
  intermediate THUMOS test mAP is diagnostic and cannot select a checkpoint.
  At declaration, uniform/direct/beta0 were at epoch 27 and counterfactual at
  epoch 25 with zero evaluation hits.
- 2026-07-15: Absorbed the Pro audit for PhysTime sparse-head commit
  `b7a37f584ba7477159dd90ba08c14728c65fb19e`. The review KILLs the current
  observation-timestamp-coupled physical-anchor ActionFormer route and the
  `physical_time + rank_assignment` variant as method candidates, while keeping
  physical time as a possible research direction. The accepted replacement
  direction is a support-decoupled physical query sparse head with complete
  physical query anchors, sparse observation support, signed center/width
  regression, parity diagnostics, support observability, C0-C4 controls, and
  staged gates before any paper claim. The wiki was corrected to state that the
  active continuation is sparse downstream detector-head adaptation, not DUCA
  selector work.
- 2026-07-15: Corrected the active task boundary to Dense-Time Spatial Zoom
  only. The route keeps the dense temporal axis and advances S1 spatial
  headroom, S2 oracle ROI/crop sufficiency, and only then a learnable zoom/crop
  policy before official AdaTAD. DUCA remains isolated in separate worktrees.
  The accidentally created clean DUCA fix worktree/branch was removed without
  code changes. The agent also recorded that it mistakenly cancelled S1 Jobs
  `1164291/1164307-1164314`; 222 partial checkpoints and 222 metadata sidecars
  remain, with last train
  epochs 56/55/55 (160), 47/47/47 (224), and 44/43/43 (256). Because the suite
  was already strict-determinism-invalidated, these are diagnostic artifacts,
  not a result. The clean continuation is
  `OpenTAD_SpatialZoom_S1_AuditFix_20260715` at `35204f5`; no final S1 mAP,
  cost verdict, ROI module, or learned crop policy exists yet.
- 2026-07-15: Current-turn task is again PhysTime sparse downstream detection
  head refactoring. Implemented support-decoupled physical query sparse head at
  commit `d72948d580e2101967fc32413b58b9a901d4ff2a`: complete physical query
  grid, sparse observation support, learned null evidence, signed center/width
  regression, G1b native-J192 raw-video config, real THUMOS gate, and 6-epoch
  pilot scripts. Clean remote snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_d72948d_20260715`
  passed focused verification (`28 passed in 48.96s`). Queued real gate Job
  `1165248` and dependent pilot Job `1165249` under run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/phystime_g1b_sdpq_d72948d_20260715_204208_+0800`.
  Status is `implemented` + `tested` + `experiment_running`; full train and
  paper claim remain held.
- 2026-07-15: Repaired the SDPQ G1b real gate after two deployment failures.
  First failure exposed a real K/J metadata bug: projection received J=192
  features but K=384 raw support metadata. Commit `996f928` rewrote native
  alignment to supply J-axis patch-input envelope support while preserving raw
  audit metadata. Second failure was a gate-script bookkeeping bug reading a
  missing query count field; commit `372fcbf` reads SDPQ head
  `valid_query_count`. Clean snapshot
  `/data/run01/sczc063/yuzibo/projects/opentad_phystime_g1b_sdpq_372fcbf_20260715`
  passed `41 passed in 49.38s`. Real gate Job `1165340` completed with
  `gate_pass=true`: K=384, J=192, no interpolation, 412 queries/proposals,
  zero missing GT assignment, finite predictions, optimizer coverage true, and
  nonzero finite gradients for adapter, projection, null evidence,
  classification, regression, and endpoint. Pilot Job `1165341` is queued by
  Slurm priority. Status remains `experiment_running`; no mAP yet.
- 2026-07-15: Finalized DUCA successful-update P0 infrastructure at commit
  `a6903ae036d7b4bfd0c25752c51f020b20427fff`: exact AMP replay/state rollback,
  successful-update-only schedules, scaler/RNG/audit checkpoints, terminal
  epoch-131 EMA-only evaluation, official mAP recomputation from prediction,
  full config/data/evaluator binding, and idempotent cluster-bound Slurm
  receipts. Local result is `80 passed, 3 skipped`; final independent blocking
  review is GO. Status is `implemented + tested`; fresh CUDA gate, pilot, and
  formal mAP remain pending.
- 2026-07-15: PhysTime G1b SDPQ pilot Job `1165341` completed on commit
  `372fcbf58d1b2eb895b724f6f040458bde4d636e` after gate Job `1165340`
  passed. Six-epoch raw result from `train.out`: Avg-mAP 10.17, mAP@0.3/0.4/
  0.5/0.6/0.7 = 23.72/15.22/7.65/3.26/1.01. Loss decreased from epoch-0
  1.7659 to epoch-5 0.9905; scan found no Traceback, OOM, NaN gradient, AMP
  skipped optimizer step, FileNotFound, FAILED, or ERROR. `epoch_5.pth` exists
  and `PILOT_COMPLETE.json` exists with `validation_pass=true`, but the pilot
  completion parser failed to record the observed mAP. Compared with G1a
  selected-axis 10.26/1.09 and physical-metric 10.56/1.04, SDPQ is runnable and
  weakly pilot-supported as a principled decoupled geometry, but not superior
  and not paper-ready. Next steps are same-commit matched controls, result
  parser fix, assignment/support/query-scale/NMS diagnostics, and only then a
  full-train decision.
- 2026-07-16: Archived and absorbed the external `REVISE-BEFORE-FULL-TRAIN`
  review of PhysTime G1b SDPQ commit `372fcbf` at
  `docs/methods/reviews/2026-07-16-372fcbf-phystime-g1b-sdpq-revise-before-full-train-raw.txt`
  with SHA-256
  `E3389D57F179BB4FFD6C1F25AC24FF1321C7865E1EBEF80BC02EF2A4E59368AF`.
  Accepted core verdict: SDPQ fixes anchor representability, but not complete
  support-query evidence identifiability. Current pilot support is
  engineering-only; full train is blocked until same-commit matched controls
  and P0 fixes/diagnostics cover evidence masks, uncovered-positive assignment,
  support context/coverage features, query residual geometry, result parsing,
  and high-IoU proposal/ranking/NMS decomposition.
- 2026-07-16: Replacement CellCF commit `3a0f5ae` passed clean Linux tests
  (87 focused plus 23 C3 regressions), synthetic gate SHA-256
  `1d8234e9...9adad`, and exact real-THUMOS CUDA gate Job `1167222` with
  artifact SHA-256 `b128f587...c4334`. Jobs `1167220/1167221` are retained as
  scheduler/environment diagnostics only. Gate-bound three-arm forced-overflow
  DDP pilot Job `1167227` is running; no full train, mAP, or cost claim exists.
- 2026-07-16: CellCF DDP pilot `1167227` completed successfully (10/10
  successful updates per arm, one forced overflow/replay, complete gradients;
  pilot SHA-256 `f199f4dc...d8085`). The first formal submission exposed a P0
  Bash transaction defect: invalid nested Slurm bindings still produced null
  receipts for aggregate/cost/completion. Only arm Jobs `1167234-1167236`
  existed; all were cancelled after 95 seconds and are invalid evidence. A
  working-tree repair now explicitly guards every substitution and requires
  exact `jobid;cluster`; local result is 110 passed/5 skipped plus 23 C3
  regressions. Status returned to `tested`; fresh exact-commit gates are
  mandatory before any replacement full train.
- 2026-07-16: Pushed exact CellCF transaction replacement
  `b8cd29f621d410b720f12380b3095dd39574e01f`. It closes null receipt,
  canonical job/cluster, intent durability, scheduler dependency mutation,
  predecessor-success and target-start ordering gaps. Independent max review
  converged to GO with P0/P1=0. Local: 127 passed/5 skipped plus 23 C3;
  accelerated Linux snapshot: 155 passed/3 skipped plus 23 C3. Exact synthetic
  gate is running; no new full train is submitted.
- 2026-07-16: Exact transaction commit `b8cd29f` passed its CPU synthetic gate
  (SHA-256 `9606f6325e05767e7b748b85e73352cdc52a439b382541a4dd5ef66ca855a76f`)
  and environment-corrected real-loader CUDA gate Job `1167345` (`0:0`, 62 s;
  artifact SHA-256
  `c4f6b5ce7d2bb830236ee51cef6d2b5ac5965bd4b84811a12cb2e86eb039b673`).
  Job `1167338` failed before Python because a non-login shebang lacked the
  cluster `module` function and is diagnostic only. Gate-bound DDP pilot
  `1167348` is pending; no mAP/cost claim changes.
- 2026-07-16: Gate-bound CellCF DDP pilot `1167348` completed `0:0` in 4:38;
  all three arms reached 10/10 successful updates, restored one forced AMP
  replay, covered full/mixed/all-short batches and complete gradient groups.
  CellCF utility was nonzero on 9/10 steps; artifact SHA-256 is
  `572e47440c54da558f6320148549de8fd62204d0f524b410f53400fe02249270`.
  Formal submit then fail-closed because pending `sacct Comment` was blank even
  though live comment and SubmitLine token were exact. Jobs `1167359/1167360`
  never ran and were cancelled. A strict SubmitLine fallback is being repaired;
  no mAP/cost claim changes.
- 2026-07-17: CellCF evidence deployment converged to exact commit `1642f26`.
  Two prior exact roots remain fail-closed diagnostics: `4bf6485` arm Jobs
  `1167469-1167471` were zero-runtime cancelled after n16r4 rejected a
  GPU-less aggregate; `522925e` Jobs `1167475-1167478` were zero-runtime
  cancelled after live Slurm exposed strict annotated `afterok` tokens not
  accepted by the old validator. The replacement passed clean Linux 212/3
  skipped plus 23 C3, synthetic SHA `3dd4750c...e5cd`, real-loader gate Job
  `1167479` SHA `3d630a32...7cde`, and DDP pilot Job `1167480` SHA
  `8e6a59e9...48cd3`. Formal Jobs `1167481-1167486` are now bound by exact
  receipts and scheduler-script hashes. At the 12:45 CST audit, exact-uniform,
  transition-beta0 and CellCF had completed 91/90/83 epochs with
  9,100/9,000/8,300 matched successful updates. Checkpoints reached epochs
  89/89/79. AMP replay counts were 5/6/5, all
  isolated replay 1/8 events without exhaustion; logged losses remained finite,
  stderr empty, and the script hashes/dependency DAG exact.
  Requested K stayed 384, legal effective-K means ranged 190--384, and CellCF's
  latest counterfactual distillation loss was finite at 0.1664 but does not
  establish benefit.
  Status is `experiment_running`; C3/C4/C7, terminal mAP, cost savings and
  paper readiness remain unproven.
- 2026-07-20: Implemented the bounded DUCA Allocation-Ceiling diagnostic and
  pushed branch `codex/duca-allocation-ceiling-20260720`. The first exact
  implementation reached `b18dd8f`, passed clean Linux `104` focused tests and
  a scheduler-transaction precheck, but formal gate Job `1174706` failed
  before producing scientific output. Root cause was an invalid timeline
  assumption: it required decoder FPS and THUMOS annotation FPS to accumulate
  less than one frame of drift. Descendants `1174707-1174710` were cancelled
  without runtime and remain invalid evidence.
- 2026-07-20: Audited metadata for all 200 THUMOS training videos. Decoded and
  annotation frame counts agree exactly for 200/200 videos; full-video
  FPS-clock drift has median about 3.00 frames and maximum 3.69 frames. Fixed
  the contract so decoded frame index is canonical, frame-count mismatch is
  fail-closed, and FPS-clock drift is reported rather than used as a false
  rejection criterion. Exact commit
  `1d51379d5feb32c8dfb11ec9a2ef238f4c3f7bbe` passed local `49` pure tests,
  clean Linux `105` focused tests, py_compile, Bash syntax and precheck.
- 2026-07-20: Submitted the corrected five-job training-side formal DAG under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_1d51379_training_20260720_041247`.
  Jobs are `1174711` gate, `1174712` export, `1174713` exact diagnostics,
  `1174714` frozen physical-grid detector candidate loss and `1174715`
  completion. At submission the gate was priority-pending and descendants
  dependency-pending. Status is `experiment_running`; no validation/test,
  selector training, mAP or paper claim is authorized.
- 2026-07-20: Corrected Allocation-Ceiling gate Job `1174711` completed
  `0:0` in 2:18. Gate artifact SHA-256 is
  `34246ef45d6e4835e32e0e720dfec0017743928b4aaf098eb2ed6d3bd0e482d0`;
  exact solver replay, candidate-loss, solver-cost and scheduler validations
  all passed. Full export Job `1174712` then started on `g0048`; the remaining
  DAG stays dependency-gated. This advances the code/runtime gate only, not
  the empirical headroom claim.
- 2026-07-20: Allocation export Job `1174712` completed `0:0` in 25:55 and
  produced a strictly validated 670-window recoverability artifact
  (`e23fdf...5968`, summary `999b38...848f`). GT diagnostic Job `1174713`
  then failed `1:0` on first GT32 sample at `lex_block_0210_0240`: the old
  solver multiplied tiny floating binary residuals by `2^29` before deciding
  an integer lexicographic objective. Candidate/completion Jobs
  `1174714/1174715` were cancelled at zero runtime. This invalidates the chain
  but is not negative method evidence.
- 2026-07-20: Repaired and pushed the exact solver at
  `8ebdd2a11ea5cc0644979324872a3b1cae5a2170`. The solver now validates every
  OPTIMAL/zero-gap primal and dual certificate, canonicalizes integer
  variables, derives distance and coverage values from actual selected
  positions, and replays every pinned objective at termination. Adversarial
  tests include lex-size 30 residual amplification, multiple lex blocks,
  upper envelopes, nonzero/boolean gap, material nonintegrality and a terminal
  position swap. Local focused contracts are `55 passed`; independent final
  review is `P0=0/P1=0/GO`. Clean Linux verification and a replacement formal
  DAG are pending; validation/test remains untouched.
- 2026-07-20: Exact `8ebdd2a` clean snapshot passed `111` relevant Linux
  tests and submitter precheck. Valid Slurm replay Job `1175393` then solved
  the exact old failure sample in 1:48 and passed independent artifact replay;
  output/summary/validation hashes are `b877bf...4b97`,
  `224256...7294`, and `19aeae...b0eb`. Temporary Jobs `1175380/1175392`
  did not enter Python and are deployment diagnostics only.
- 2026-07-20: Submitted transactional replacement root
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_allocation_8ebdd2a_training_20260720_1320`.
  Jobs are `1175395` gate, `1175396` export, `1175397` diagnostics,
  `1175398` candidate detector loss and `1175399` completion. Scheduler
  pre/post-release validation passed; gate is priority-pending and descendants
  are strict `afterok`. Status is `experiment_running`; no validation/test or
  selector training was authorized.
- 2026-07-20: Replacement exact gate `1175395` completed `0:0` in 4:02.
  Gate artifact SHA-256 is
  `6030d9fb7110aa7c73b2df244eff50136d1342c5e2e90bd86db485d38faafc61`;
  exact solver replay, candidate-loss, solver-cost, submission and scheduler
  validations all passed. Export `1175396` is released and priority-pending;
  later jobs remain dependency-gated. This closes the numerical/runtime gate
  only. Formal headroom, recoverability and frozen detector-loss evidence
  remain `experiment_running`, and validation/test remains untouched.
- 2026-07-20: Replacement Jobs `1175395-1175399` all completed `0:0` with
  empty stderr. Sealed evidence SHA-256 is
  `8232f2f0889bc5e0579abcf82d42ab4009397366c5c4b0e6bfd71d0c658ad6d6`.
  The 670-window deploy score is worse than exact uniform in endpoint distance
  (`0.508843` vs `0.464399`) and radius-1 both-endpoint recall
  (`-0.151970`). Privileged GT on 32 windows shows modest geometry headroom
  (`0.446381 -> 0.238764` distance), but frozen detector loss is worse for
  deploy, constrained-GT and unrestricted-GT selections
  (`0.275252/0.318532/0.487739` vs uniform `0.224627`). The registered current
  transition-score/global-allocation/frozen-detector route is therefore
  killed before selector training. Validation/test and paper claims remain
  unauthorized.
- 2026-07-20: Published the bounded negative-result detector-utility Pro
  audit prompt at documentation commit `706e23b`:
  `docs/methods/prompts/2026-07-20-duca-allocation-negative-result-detector-utility-pro-audit-prompt.md`.
  The exact implementation audit target remains `8ebdd2a`; `706e23b` adds
  documentation only. The prompt forbids validation/test and long training,
  requires line-referenced physical-grid detector-path review, and permits
  only a bounded detector-utility diagnostic if the evaluator is credible.
- 2026-07-20: Corrected the route verdict after the user rejected frozen
  detector loss as a final TAD metric. Documentation commit `db11aee` revises
  the Pro prompt: the `8ebdd2a` suite is now `HOLD_FOR_MATCHED_MAP`, not a
  route KILL. The next decision experiment is one single-use hash-bound
  official mAP replay of exact uniform versus deploy-visible transition
  selection under the same checkpoint and post-processing. Privileged GT
  selection remains excluded from deployable mAP.
- 2026-07-20: Froze the next DUCA development sequence to one protected
  end-to-end route. A clean branch
  `codex/duca-protected-e2e-20260720` starts from `db11aee`. The first artifact
  is a strict Pro design-adjudication prompt; no model code has changed. Only
  after Pro GO may P0 protocol, P1 implementation, P2 separate-loss gradient
  ownership and P3 hard-soft alignment proceed. Only after all four gates pass
  may exact-uniform, transition-no-bridge, protected-E2E and protected-E2E-rho
  run under one official-60 protocol. Status is `designed`, not implemented or
  experiment-running.
- 2026-07-17: 冻结 CellCF 训练预算解释。当前 `epoch_131.pth` 对应 132
  epochs/13,200 successful updates，其用途是匹配历史约 13,080-step
  分离训练、消除旧 DUCA 5,940-step 欠训练混杂，不自动成为论文主训练方案。
  当前三臂继续完成，不中断。若终点通过，必须补 same-commit 官方
  60-epoch 三臂匹配；epoch 59/89/131 只能作为固定收敛轨迹，epoch 59
  不能冒充 60-epoch 独立训练或用于挑选 checkpoint。最终必须报告训练
  GPU-hours、峰值显存、反事实训练开销、完整推理 p50/p95 与 break-even。
  若收益只在 132 epoch 出现，则训练效率主张不成立。
- 2026-07-18: 修复并验证 CellCF 成本 profiler 的 CPU-enqueue 数据契约。
  精确提交 `4ce69c852bdbd902046b47bc6019ae11e850dbe4` 共享七字段严格白名单，
  保持字段 raw-only，并把验证提前到每个 sample 写入前。验证为本地
  259 passed/10 skipped、clean Linux 279 passed。Job `1170932` 的 profiler
  产物有效，但作业因临时 heredoc 校验器 SyntaxError 仍是失败诊断；替代
  GPU 门禁 Job `1170940` 使用正式 CellCF 配置与终态 EMA，在两样本上
  `COMPLETED/0:0`，严格重建、JSONL/summary 一致、字段集合与哈希均通过。
  这只把数据契约推进到 `tested`；没有提交新的 500-sample 正式成本恢复，
  C7 与论文成本主张不变。
- 2026-07-19: 复核当前目的、路线和失败链时修正陈旧 wiki 状态。CellCF
  idea 从 `experiment_running` 改为 `tested_diagnostic`；C3 记录 matched
  seed-0 的 uniform/transition/CellCF 63.8594/64.2755/64.0610，仍为
  `unproven`；C4 明确 CellCF 是 detached utility、没有测试 direct detector
  gradient；C7 记录 schema gate 已通过但正式 500-sample cost pair 仍缺。
  没有新实验、模型或论文 claim 被批准。
- 2026-07-20: Archived and independently rechecked the Protected-E2E Pro
  adjudication (`f91db53a...97ccb0`, verdict `REVISE`). The reviewer saw only
  prompt commit `280631a`, so its no-implementation/no-gradient statements are
  stale: later commits through `b3222af` and Job `1176948` passed real
  full-model main/rho P1/P2 connectivity and ownership. The structural HOLD
  remains: that candidate uses a local slope surrogate, candidate-hole
  constraint, selected-axis GT, rho 0.05, bridge 0.25 ramp and a 4x8
  checkpoint-based P3 rather than one physical exact-K Gibbs feasible family.
  P3 stopped before statistics on an invalid old-manifest field expectation.
  The route is reset to `designed`; no official-60 is authorized. P0 must
  derive loader length rather than blindly accepting the review's unverified
  5940-update count.
- 2026-07-20: Completed a pre-edit inventory of all 37 local `OpenTAD_*`
  directories and the 18 primary-registered worktrees. Recorded exact heads,
  branches, dirty counts, reusable DUCA/PhysTime/TrueTime assets and explicit
  no-edit boundaries in `worktree_inventory.md`. The only construction tree is
  the isolated clone
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` on
  `codex/duca-physical-protected-e2e-20260720` from `b3222af`; it has one
  untested physical-DAG draft. SparseHead, Spatial-Zoom, ChronoTransport and
  historical route trees remain untouched. P0-P3 and official-60 remain
  unauthorized.
- 2026-07-20: Refreshed the worktree snapshot before further DUCA edits.
  There are still 37 top-level `E:\DeskTop\TAD\OpenTAD_*` directories, while
  the registered count is now 19 because Codex created one detached inspection
  tree under `C:\Users\skywalker\.codex\worktrees`. Standardized dirty counts
  to recursive per-file porcelain: the Spatial-Zoom primary has 164 entries,
  ChronoTransport 13 and historical OnlineTAD 47. Corrected
  `OpenTAD_SpatialZoom_S1_AuditFix_20260715` to HEAD `28b0a67`. The only DUCA
  construction surface remains
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720` at base `b3222af`
  with one untested `structured_selection.py` draft. No SparseHead,
  Spatial-Zoom, ChronoTransport or other route code was modified.
- 2026-07-20: Closed two independent read-only cross-tree audits. The exact
  count is 37 top-level `E:\DeskTop\TAD\OpenTAD_*` directories plus the nested
  isolated DUCA clone, for 38 relevant Git trees; 19 are registered worktrees.
  Neither audit found a hidden complete frozen implementation. Allocation-
  Ceiling has the audited physical hard-solver semantics, Protected-E2E has
  ASFormer routing and real-model gate machinery, and PhysTime/TrueTime provide
  only selected physical-coordinate and gradient-test ideas. No existing tree
  combines the physical exact-K hard/soft graph, coverage floor, explicit
  selector adapter/head, native-time detector contract and P0-P3 evidence.
  This closes inventory only; official-60 remains unauthorized.
- 2026-07-20: Retained the isolated physical exact-K DAG draft after its first
  remote mathematical gate. Exact construction tree is
  `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720`, branch
  `codex/duca-physical-protected-e2e-20260720`, base `b3222af`, with exactly
  two dirty implementation entries. Disposable remote copy
  `opentad_duca_physical_dag_draft_20260720_02` passed `py_compile` and
  `9` focused tests covering exhaustive irregular-axis hard/soft parity,
  gradients, ties, short rows and fail-closed axes. The test bypassed an
  unrelated NumPy-2/old-OpenCV package-import conflict by directly loading the
  standalone module. No SparseHead, Spatial-Zoom, ChronoTransport, spatial
  crop or other route file was touched. This is component-level `tested`
  evidence only; P0-P3 and official-60 remain unauthorized.
- 2026-07-20: Added an uncommitted protected-selector integration draft only
  inside `.codex_tmp/OpenTAD_DUCA_ProtectedE2E_Final_20260720`. The draft
  introduces an explicit 197-to-64 selector adapter/head, the fixed coverage
  floor, and four physical exact-K arms with exact-hard forward/soft-Gibbs
  backward. Dense GT is unchanged and selected-axis remap is disabled. Local
  compile passed; Windows pytest still fails at the pre-existing PyTorch
  `c10.dll` load, and the prepared remote integrated test was interrupted
  before execution. ActionFormer, strict physical-head validation, configs,
  P0-P3 and official-60 remain incomplete. Status is `implemented_draft`;
  no training or claim is authorized.
- 2026-07-20: Resumed the protected-selector remote focused gate. A first
  PowerShell upload-map failure occurred before Python; after fixing the
  wrapper, pytest exposed one synthetic fake-ASFormer indexing typo. The
  corrected identical suite passed `14/14` in remote copy
  `opentad_duca_protected_selector_draft_20260720_01`. This tests the physical
  DAG, exact-uniform bypass/native metadata, protected/rho gradient ownership
  and hard-only inference. Independent P0-P3 audit still returns `HOLD` due to
  selected-axis formal configs/validator, unsealed loader exposure,
  train-time val/test construction, implicit protected LR ownership and stale
  P3. Selector status advances only to `tested_focused`; no full-model gate or
  training is authorized.
## 2026-07-20 - Protected DUCA detector-contract focused gate passed

- Added route-scoped official ActionFormer integration for detector RNG
  isolation and explicit protected-selector/ASFormer optimizer LR ownership.
- Added strict native dense-physical metadata validation in AnchorFreeHead;
  float positions, mismatched duplicate fields, selected-axis declarations,
  invalid ordering/range and detector-mask/count mismatch now fail closed.
- Disposable N16R4 real-OpenTAD focused run passed `24/24`.
- Evidence status is only `tested_focused`; full-model P0/P1/P2/P3 and
  official-60 terminal EMA mAP remain unproven and undeployed.
- The bounded GO target is greater than the strictly matched approximately
  65 Avg-mAP uniform baseline. It is an experiment criterion, not a promised
  result.
- 2026-07-20: Expanded the isolated Protected-E2E implementation from the
  detector-contract gate to a focused P0-P3 and terminal-evidence chain.
  Added exact loader/config/data/pretrain hashing, real main/rho full-model
  gate bindings, a 48-window/576-swap physical P3 with padded-window support,
  complete buffer/module/custom-state/RNG restoration, and a 60-epoch
  terminal-EMA finalizer that recomputes official mAP. The disposable N16R4
  copy passed `37/37` focused tests in `53.62s`. This remains uncommitted
  `tested_focused` evidence only: real Slurm P1/P2/P3, authorization,
  official-60 and any `>65` result are still absent.
- 2026-07-20: Completed a repository-grounded design and paper-readiness
  review of the current DUCA Protected-E2E pre-backbone route. Verdict is
  `HOLD_AND_REVISE`: physical exact-K, same-graph hard/soft, exact-hard
  forward, protected gradient ownership and native detector coordinates form
  a coherent hypothesis, but no claim is promoted beyond the existing
  statuses. Registered the new primary structural risk that nonuniform
  selected frames are packed by rank into nominal 16-frame VideoMAE
  clips/tubelets; a physical ActionFormer head cannot by itself repair
  temporal semantics already consumed by the backbone. P1 now requires a
  complete 384-frame chunk/feature/mask build, uniform parity, short-action
  support, timestamp-spacing counterfactual and raw-gather-to-head roundtrip.
  Also froze wording that fixed K is adaptive placement rather than dynamic
  budget, binary-only supervision applies to the action head rather than the
  whole shared trunk, and efficiency needs complete decode-to-output
  accounting. The current zero-shot/teacher/HardTopK LaTeX draft is not
  Protected-E2E paper evidence and all result tables remain empty. No model,
  config, experiment or remote state was changed.
- 2026-07-20: Closed the focused P0-P3-to-official60 evidence implementation
  to `50/50` passing tests on the disposable N16R4 copy. Added transactional
  held submission with rollback for the two real-model gates, three P3
  shards and completion authorization, plus a separate four-arm official-60
  transaction and terminal-EMA aggregate. The evaluator now fail-closes on
  dirty/wrong commits, non-3407 seed, non-terminal checkpoint or non-EMA
  state. No real P0-P3 CUDA artifact or mAP exists yet, so status remains
  `tested_focused`; `>65` is still only the GO threshold.
- 2026-07-21: Re-audited the isolated Protected-E2E draft before submission.
  Fixed random-crop pseudo-boundaries by carrying per-endpoint
  `gt_boundary_validity`; rebuilt P3 around original uncropped annotations,
  seconds-based boundary strata, retained near-zero rows and at least four
  padded windows per duration stratum; added fail-closed full/padded
  real-loader coverage, two real AMP optimizer updates with scheduler/EMA,
  and exact-uniform physical-versus-selected-axis target/loss/decode parity
  to the CUDA gate and authorization contract. The disposable N16R4 focused
  suite now passes `55/55`; required legacy C3 checks pass `23/23`. This is
  still uncommitted `tested_focused` evidence. At `2026-07-21 00:52 +0800`
  the remote DUCA queue was empty: no CUDA gate, P3 shard, official-60 run or
  mAP result had been submitted.
- 2026-07-21: Published the complete protected physical implementation at
  `ce5d03ebf5fd51634adcd2c7fbca2542a399d532`, then repaired only the N16R4
  submission contract at `8f852ce59347886b4dee717fc8208613af36cf35`.
  The exact clean `8f852ce` snapshot passed 83 matched focused-plus-legacy
  tests and froze P0 successfully. Two formal attempts created no jobs:
  `BSCC-N16R4` is an SSH label rather than a Slurm cluster name, and N16R4
  requires `--gpus=1` instead of `--gres=gpu:1`.
- 2026-07-21: The corrected six-job transaction then failed cleanly on
  `AssocMaxSubmitJobLimit`; no `jobs.tsv` or residual DUCA job exists. The
  user association already had 12 jobs. To avoid cancelling unrelated work,
  added the scheduling-only single-allocation launcher and pushed exact commit
  `ee05f610133fc37f8f1ee67b7225bb38ae917cc5` (tree
  `a190e399bb1fdfdac230c0a4305c4b08946a8ec1`). Its clean snapshot passes
  84 tests. Exact P0 manifest SHA-256 is
  `a02b6e690804d574d7929a408c17b396cc3cca4887a352be6c55270846e46a7e`
  with loader 100, epochs 60, 6000 updates/arm, 48 P3 windows and 576 swaps.
  Single-job precheck passed; a bounded watcher is waiting for one free
  association slot. Status remains `tested_focused`, not
  `experiment_running`, until a real Job ID exists.
- 2026-07-21: Association count fell from 12 to 11 after unrelated screening
  Job `1177653` terminated. The bounded watcher transactionally submitted and
  released protected physical gate Job `1177681` (`dp_all_ee05f61`).
  Formal root:
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_protected_physical_ee05f61_gate_single_20260721_021705`.
  `jobs.tsv` SHA-256 is
  `29c69d51c2bc99199e8ddfa4dafa1f56fac66c856452a5a9854d3d5405f8f8b7`;
  generated sbatch SHA-256 is
  `cff10183d2e7d04f7d55def0f031d1d42265bcfc6fc1097bfa8245bdc0e2aff6`.
  Initial state is `PENDING (AssocGrpGRES)`. Gate-stage status advances to
  `experiment_running`; CUDA/P3 authorization, official-60, mAP and all paper
  claims remain unproven.
- 2026-07-21: Audited official
  `LeapLabTHU/Uni-AdaFocus@8846488310fdd4a18412608006030643e794c36e`.
  Its useful transferable ideas are learned/random input diversity and
  policy-specific training strength; its detached hard temporal indices do
  not satisfy DUCA's direct detector-gradient requirement. Created isolated
  branch `codex/duca-uni-companion-20260721` from immutable `ee05f61` and
  implemented three matched learned variants: bridge `1.0`, bridge `0.25`,
  and bridge `0.25` plus a one-pass 50% exact-uniform training companion.
  Static compile and diff checks pass; local PyTorch remains unavailable due
  to the known Windows `c10.dll` failure. Status is
  `implemented_local_static`, not tested or empirically supported.
- 2026-07-21: Closed and deployed the bounded Uni-companion optimization at
  exact commit `d748684bc6a3da5b5cbbb0b78a64b71ef1cdd1dc` (tree
  `50e43e7a91dc529b11d660f21e6fef46e4340601`). A clean remote Linux snapshot
  passed `66/66` focused tests plus `23/23` required legacy C3/ASFormer
  regressions. P0 froze successfully at file SHA-256
  `e4fc629305fbb41ca5915ad71f866041340cf7e5b1c61d960ea647c74f6d2538`,
  with batch size 2, 60 epochs and 6000 updates per arm. Current formal gate
  Job is `1177687`, pending on `AssocGrpGRES`. Three learned official-60 Jobs
  are directly queued behind it: direct `1177690`, bridge-0.25 `1177691`,
  and Uni-companion `1177692`. Superseded gate `1177681`, dispatcher
  `1177688`, and uniform attempt `1177689` were cancelled before runtime.
  Exact-uniform remains fully implemented/P0-frozen but awaits a freed
  submission slot. Status is `experiment_running`; no CUDA authorization,
  terminal mAP or greater-than-65 evidence exists.
- 2026-07-21: Started bounded fail-closed watcher PID `485235` for the missing
  exact-uniform control and final aggregate. Its script SHA-256 is
  `f69c8387bda43a06e9373d8bcd05d5a98d9c5992ede1883b9bb965fb5faa338a`.
  It submits nothing unless gate `1177687` is `COMPLETED` and its authorization
  JSON exists with `ok=true`; any gate failure stops the watcher.
- 2026-07-21: Rechecked the live queue: gate `1177687` remains
  `PENDING (AssocGrpGRES)` and learned Jobs `1177690-1177692` remain
  dependency-pending; watcher PID `485235` is alive. Registered, without
  changing the sealed commit, that learned arms use a randomly initialized
  transition-score output head and that tied global physical exact-K paths
  resolve lexicographically rather than to exact-uniform positions. This is
  an early-training risk to measure, not a proven cause or a new result.
- 2026-07-21: Quantified the initialization risk with solver-only diagnostics
  on the clean `d748684` snapshot. The true `T=768,K=384` exact-uniform cap is
  3 frames. Zero-score physical Viterbi gives `0.5000` overlap and `96.9974`
  frame rank-MAE versus exact uniform; one synthetic seed-3407 random-scoring
  example gives `0.5208` overlap, `7.4974` rank-MAE and 13-frame maximum rank
  error. These are not real-data results. A fixed exact-uniform reference
  score passed through the current coverage-floor and physical solver returns
  the exact-uniform path bit-for-bit, motivating a bounded residual-policy
  successor if the real-loader audit confirms the risk.
- 2026-07-21: The real THUMOS full-train loader audit exposed a blocking
  contract missed by the synthetic gates: RGB windows are `torch.uint8`, while
  `d748684` rejected non-floating selector input before soft resampling. Jobs
  `1177687/1177690/1177691/1177692` were therefore cancelled with zero runtime
  and watcher PID `485235` was stopped. They produced no model evidence.
- 2026-07-21: Implemented the narrow real-loader repair on branch
  `codex/duca-uni-companion-inputfix-20260721`, exact commit
  `4d84acda4d073fb6aac956c21386df8ed5d4d2f5`, tree
  `b15a064784f25d888cc66df01c39781422403195`. Hard gathering preserves exact
  raw values; soft/straight-through detector input is promoted to FP32 before
  differentiable arithmetic. No official detector or unrelated route changed.
  The clean exact snapshot passed `67/67` focused and `23/23` required legacy
  tests. P0 froze at file SHA-256
  `eabc6da8c3cc4308b70a8c8d6bbecc6c6e4b408cb17d2ee6041ed83f24a4eb3f`
  with 60 epochs, 100 steps/epoch and 6000 updates/arm.
- 2026-07-21: Submitted the replacement formal queue. Gate Job `1177696` is
  pending on `AssocGrpGRES`; direct bridge Job `1177697`, bridge-0.25 Job
  `1177698` and Uni-companion Job `1177699` are dependency-pending behind
  `afterok:1177696`. Fail-closed watcher PID `808310` will submit exact-uniform
  only after successful gate authorization and a freed association slot.
  Deployment manifest SHA-256 is
  `13f2b13c906f6605b8bbca6d06ad24201bcae83a027477dd42385b242807f6f4`.
  No CUDA/P3 authorization, optimizer update, checkpoint or mAP exists yet.
- 2026-07-21: Audited initial learned selection on eight real full training
  windows after the input repair. Mean exact-uniform overlap is `0.502604`,
  mean rank error `4.038737` frames and maximum rank error 18. Mean nearest
  true-boundary distance is `0.528646`, versus `0.555556` for exact uniform.
  This confirms a nonuniform random start but does not establish worse
  boundary geometry, so no policy homotopy was added to the sealed run.
- 2026-07-21: Added a fail-closed sequential deployment for the already
  implemented and P0-frozen `protected_e2e_rho001` arm. Watcher PID `883230`
  waits for successful gate `1177696` and direct arm `1177697`, then uses the
  freed association slot. It tests detector-gradient scale `0.01` into only
  the final official ASFormer encoder layer; it is not yet a Slurm Job and has
  no result. Watcher SHA-256 is
  `c9b15b5f3a0cf369349a66548686db509a4dd90bf521ae4d418c5557860b1902`.
- 2026-07-21: Added a second sequential evidence arm for
  `transition_no_bridge`. Watcher PID `933605` waits for the gate-bound rho
  receipt and successful rho completion before it can submit. The watcher
  revalidates commit `4d84acd`, the P0 arm source hash and original-four-arm
  authorization; watcher SHA-256 is
  `940ddb5797d998850be2477ab47cc0a2fbaf840e76da12b62109c1ba4eaed136`.
  No Slurm Job exists yet. Also registered that the frozen batch-two Uni
  companion gives detector gradient to only one learned row, making its
  aggregate bridge exposure approximately half the plain bridge-0.25 arm.
  A homotopy or normalized-companion successor must be a new commit/P0.
- 2026-07-21: Gate `1177696` failed at zero runtime with exit `127` because
  the generated non-login sbatch called `module load` before sourcing
  `/etc/profile`. Dependent Jobs `1177697-1177699` were dependency-never-
  satisfied and cancelled; watcher PIDs `808310/883230/933605` exited without
  submitting jobs. This is a launcher failure, not model evidence.
- 2026-07-21: Began an isolated exact-uniform-to-learned physical-policy
  homotopy successor. Current draft adds successful-optimizer-step schedule
  accounting, hard-forward exact-uniform alpha-zero checks, learned alpha-one
  inference, AMP replay/EMA persistence, P0-bound authorization, a matched
  four-arm official-60 aggregator, and `/etc/profile` initialization in every
  generated Slurm job. Status remains `implemented_draft/tested_static` until
  an exact clean commit passes remote tests and a real CUDA gate.
- 2026-07-21: Closed the homotopy implementation audit at exact commit
  `be18ba53fb34c6d68d60b7b63edf1a7380d55c93` (tree
  `3050759426596db68e2d1bf247ac63150c1861ac`). The clean Linux snapshot
  passed 152 focused/legacy tests and an independent read-only review returned
  PASS. P0 file SHA-256 is
  `8385b9fb74c90c0faf7fad4761d85864450460bc43c69e6a606a5a0f1dfb8414`,
  freezing 100 batches/epoch and 6000 successful updates/arm. Formal serial
  gate Job `1177713` is RUNNING on `g0003`; logs show real VideoMAE/AdaTAD
  construction. This is `experiment_running` at gate scope only. No training
  authorization, terminal mAP or greater-than-65 evidence exists yet.
- 2026-07-21: Gate `1177713` failed after 2m33s in the first real full-model
  arm because the float64 physical max-gap cap used by the exact graph was
  narrowed to AMP policy-score precision before metadata validation. This was
  a control-plane precision defect, not a measured mAP result. Commit
  `bc503fc3aa5c21487ca0c3679648f3c3085af82d` preserves physical caps in
  float64 for learned, exact-uniform and fixed-path routes and adds uint8/FP16
  regressions. Its clean Linux suite passed 154 tests. A fresh P0 with file
  SHA-256 `7b5820fea25ae7866952341b9983c23f6d3a4891d4cf2aaf047175cb6ad96483`
  was frozen, and replacement gate Job `1177714` is RUNNING on `g0003`.
- 2026-07-21: Gate `1177714` also failed after 2m33s, after crossing the
  physical-cap check. The next fail-closed check exposed a validation-tool
  assumption: `_perturb_unselected` used `torch.randn_like` on real-loader
  uint8 RGB and raised `normal_kernel_cuda not implemented for Byte`. This is
  not a model loss, optimizer, checkpoint or mAP result. Commit
  `b987c8c6bd2b9f83027354adaaf6f338a205798a` uses deterministic uint8 XOR
  perturbations only for this gate check, preserves selected observations
  exactly, and retains Gaussian perturbation for floating inputs. Clean Linux
  verification passed 155 tests. Fresh P0 SHA-256 is
  `a246dc8c3fbc6f6e4a65a3a706a1259e54421f93a4707a922c567db1c92f9b99`;
  gate Job `1177715` was submitted and entered RUNNING on `g0003`. Four-arm
  official-60 training remains unsubmitted pending hash-bound authorization.
- 2026-07-21: Gate `1177715` failed closed after 2m40s at the exact-uniform
  physical-vs-selected-axis detector-loss parity check, so no authorization
  and no training job were created. Read-only diagnostic Job `1177719` on the
  identical commit/P0 measured physical cls/reg/objective
  `0.040603362/0.031404633/0.072007999` and selected-axis
  `0.054389104/0.040477306/0.094866410`; objective relative difference is
  24.10%. The endpoint-uniform positions contain 382 gaps of 2 and one gap of
  3, confirming a non-affine geometry rather than small numeric drift. Per the
  frozen P1 decision this route is now `stop_and_revise_representation`; do
  not relax parity or submit the four official-60 arms from `b987c8c`.
- 2026-07-21: Implemented the bounded selected-axis optimization successor in
  the isolated DUCA tree. It reuses the official ActionFormerHead and existing
  GT/proposal selected-axis maps, and adds direct-0.25, homotopy-0.25, and
  homotopy plus 50% one-pass uniform-companion configs. Local syntax and all
  three static official-60 config contracts pass; Windows Torch remains
  unavailable. Exact remote CUDA gate and formal jobs are pending.
- 2026-07-21: Pushed selected-axis implementation `1678d13` passed 34 clean
  Linux focused tests, but real CUDA gate `1177721` stopped before training on
  a non-finite transition-scorer gradient in its fresh-GradScaler ownership
  check. No optimizer update, checkpoint or mAP was produced. Temporary Jobs
  `1177722/1177723` exposed only diagnostic launcher/PYTHONPATH defects.
- 2026-07-21: Read-only diagnostic `1177724` completed on the same real
  T=768/K=384 two-row THUMOS contract. Transition-distribution,
  boundary-mass and combined objectives all had finite gradients at scales 1
  and 65536; soft occupancy summed exactly to 384 per row. A local successor
  now freezes seed 3407, gives all learned arms the same detector-gradient
  schedule, includes exact-uniform in the real full-model gate, and reuses the
  production AMP replay path with forced-overflow optimizer/scheduler/EMA/
  selector-schedule checks. State is `implemented_local_static`; replacement
  commit, CUDA gate and all four official-60 training Jobs remain pending.
- 2026-07-21: The corrected selected-axis successor was committed and pushed
  as `c2de186f8edae3b3d19e799cff4792b44b827159` (tree
  `f09c90edb79d51554bfc70a701f01f7f24381a9f`). Its clean remote snapshot
  passed 35 focused Linux tests. Exact real CUDA gate Job `1177732` was
  submitted and is initially pending on Slurm priority. This promotes only to
  `experiment_running` at gate scope; the four official-60 training Jobs still
  have no IDs and no mAP exists.
- 2026-07-21: Gate `1177732` reached the production one-step audit but checked
  selector-parameter movement at the formal warmup learning rate of zero. It
  therefore failed an audit-positioning contract, not a model numeric test;
  no checkpoint or mAP was produced. Exact pushed replacement
  `1af6ff84f2cc5c4348710807bd960cea5d1741c0` (tree
  `95043c2eb7aed0247ed6eb53c7c72a4f61406047`) positions only the gate
  proof at the first nonzero successful step and leaves formal training from
  step zero unchanged. Its clean remote snapshot passed 35 focused Linux
  tests. CUDA gate Job `1177733` was submitted and entered RUNNING; all four
  static config contracts passed. Official-60 training remains unsubmitted
  pending the hash-bound gate suite.
- 2026-07-21: Exact gate `1177733` completed successfully in 4m23s and sealed
  `gate_suite.json` with SHA-256
  `38d5e185b36dd1ffc0adba979ce00623ed202b42d604eee811cf8f9c35d80c09`.
  All four real-model variants passed gradient ownership, actual hard
  selection, K=384/max-hole=2, forced AMP overflow replay, and exactly-one
  optimizer/scheduler/EMA/selector-schedule update. Formal same-commit seed
  3407 official-60 Jobs were then submitted and entered RUNNING: `1177734`
  exact-uniform, `1177735` direct-0.25, `1177736` homotopy-0.25 and `1177737`
  homotopy+50% one-pass uniform companion. Each launch manifest binds the gate
  hash and terminal epoch-59 EMA. No checkpoint or mAP exists yet; status is
  `experiment_running`, and greater-than-65 remains unproven.
- 2026-07-21: Formal Jobs `1177734-1177737` all stopped in 33--41 seconds
  before model construction and before any optimizer update. `tools/train.py`
  routed their 60-epoch configs through the legacy epoch-131 checkpoint
  criterion, yielding `formal DUCA checkpoint criterion is not frozen`.
  Exact correction `cb89586a92b8b0a8349ecc9551bc50aa97982360` adds the
  selected-axis official-60 protocol, exact gate/config/pretrain runtime
  bindings and semantic CLI override allowlist without changing model, loss or
  training length. Its clean Linux snapshot passed 38 focused and 23 required
  C3/ASFormer tests. Replacement CUDA gate `1177776` was submitted and entered
  RUNNING; no replacement formal job or mAP exists yet.
- 2026-07-21: Replacement CUDA gate `1177776` completed successfully in 4m26s
  and sealed suite SHA-256 `76628abd...0a27`. A read-only production
  runtime-binding preflight then reopened all four exact config/full-model
  gate/pretrain/data bindings and passed, artifact SHA-256 `0844c030...29504`.
  Replacement formal Jobs were submitted and entered epoch 0: `1177779`
  exact-uniform, `1177780` direct-0.25, `1177781` homotopy-0.25 and `1177782`
  homotopy+50% one-pass uniform companion. All are commit `cb89586`, seed 3407,
  60 epochs/6000 successful updates, terminal epoch-59 EMA. Two isolated
  batch-17 AMP skips were replaying within the 8-retry contract; no hard error,
  checkpoint or mAP exists yet.
- 2026-07-21: All four replacement Jobs reached epoch-0
  `duca_schedule_step=50` with finite total loss 5.5197--5.5264, exact K=384
  and 8596--8597 MB memory. Each had isolated AMP replay events at batches 17
  and 47, both replay 1/8 after scale reduction; no replay exhaustion,
  Traceback, OOM or non-finite loss was observed. This is verified real
  optimizer/schedule progress, but not terminal stability or mAP evidence.
- 2026-07-21: All four hash-sealed epoch-0 training audits were written and
  each records 100/100 attempted/successful batches, 100 scheduler, EMA and
  selector-schedule updates, 102 optimizer attempts, two AMP replays,
  `max_amp_retries_observed=1` and zero exhaustion. All four Jobs entered epoch
  1. This closes the one-epoch execution contract only; no checkpoint or mAP
  exists and the greater-than-65 criterion remains unproven.
- 2026-07-21 09:16 +08:00: Exact running evidence remains commit `cb89586`,
  gate Job `1177776`, and formal Jobs `1177779-1177782`. All four arms closed
  three epochs / 300 successful updates without fatal anomalies. Homotopy
  step 350 entered `continuous_policy_homotopy` under the frozen 300+1800
  cosine schedule; expected policy alpha is about 0.0019. Registered that
  `duca_schedule_progress` is not policy alpha. No checkpoint geometry, mAP,
  or >65 evidence was claimed.
- 2026-07-21 09:28 +08:00: All four epoch-4 EMA checkpoints existed. Submitted
  read-only selector-quality Job `1177987`, which entered RUNNING on `g0048`.
  It uses the first 32 validation batches per arm and exports coarse,
  transition and hard-selection geometry without executing the AdaTAD heavy
  backbone, changing training, evaluating intermediate mAP or selecting a
  checkpoint.
- 2026-07-21 09:29 +08:00: Job `1177987` failed in 23 seconds before model
  construction with `ModuleNotFoundError: tools`; preserved it as launcher
  history. Corrected the diagnostic only by using module entry points and
  submitted fresh v2 Job `1178004`, which entered RUNNING on `g0030`. The four
  formal training Jobs and checkpoints were not changed.
- 2026-07-21 09:40 +08:00: Formal Jobs `1177779-1177782` remained RUNNING and
  each had completed at least seven epochs / 700 successful updates; exact-
  uniform had entered epoch 8. No Traceback, OOM, ValueError, non-finite loss
  or replay exhaustion was found. Corrected selector diagnostic Job `1178004`
  completed in 8m23s and sealed all four epoch-4 summaries. Coarse AUROC was
  only `0.4625--0.4689`; learned radius-1 transition AUROC `0.5267--0.5300`
  trailed raw `abs(delta p_action)` `0.6110--0.6175`. Learned policies gained
  `0.0091--0.0139` exact boundary recall but lost `0.0924--0.0984` radius-1
  recall versus uniform. This is diagnostic-only early evidence: the protected
  detector-to-selector bridge remains zero until step 2100, and no terminal
  epoch-59 EMA or mAP exists.
- 2026-07-21 09:44 +08:00: Independent read-only audit of exact commit
  `cb89586` found no P0 defect requiring the valid four-arm suite to stop. It
  issued a method-language HOLD: the homotopy is continuous in scores/soft
  occupancy but the hard Viterbi path is piecewise constant; T=768/K=384/G=2
  makes broad-radius boundary recall weakly discriminative; and the one-pass
  batchwise uniform companion is only AdaFocusV2-inspired, not an exact
  implementation of its same-video second random-crop forward. Registered
  alpha-sweep hard-path, r0/r1, gap and short-window audits as read-only future
  diagnostics, not grounds to alter the running protocol.
- 2026-07-21 09:50 +08:00: Added a pure read-only hard-set audit over the
  epoch-4 `1178004` records. Three of 64 windows had `valid_len<=384` and were
  necessarily all-frame. On the remaining 61 windows, direct/homotopy/
  companion retained only 51.4%/51.7%/51.8% of exact-uniform positions and
  swapped about 186.5/185.6/185.0 frames. Adjacent-selection rate rose from
  uniform 4.4% to 35.2%--35.5%. Since homotopy alpha was only about 0.03, this
  proves score interpolation does not constrain hard-path movement. The route
  can cluster frames; early failure is aggressive, misplaced clustering rather
  than insufficient allocation freedom. No running protocol was changed.
- 2026-07-21 10:04 +08:00: Ran the existing pure-JSON selection decomposition
  on all four epoch-4 exports and sealed manifest SHA-256 `dd8a8603...3827`.
  The evaluation-only GT-informed heuristic improved r0 recall from 0.1342 to
  0.2472 and endpoint distance from 0.4834 to 0.2416, but violated the current
  hard contract with mean max hole 11.36. Registered it as an infeasible
  diagnostic, not a privileged oracle or deployable method result.
- 2026-07-21 10:08 +08:00: Independent coordinate audit returned PASS for the
  epoch-4 selector-quality records. All 64 GT windows and valid lengths matched
  original THUMOS metadata; formal action targets and diagnostic labels agreed
  at all 46,527 positions. Recomputed exact-uniform AUROC/AUPRC was
  0.463353/0.255005, action/background mean p_action was 0.458222/0.459672,
  and one-candidate shifts changed AUROC by under 0.002. The weak coarse score
  is a real early optimization issue, not a coordinate bug.
- 2026-07-21 10:18 +08:00: Added and pushed a read-only hard-homotopy
  trajectory audit at commit `87cfd20` on branch
  `codex/duca-selected-axis-diagnostics-20260721`. It supports both frozen RGB
  batches and existing hash-bound score records, accepts real-loader uint8,
  and reports soft/hard alpha trajectories plus selection freedom without
  constructing AdaTAD. Local isolated tests were 6 passed; clean remote
  focused regressions were 47 passed/2 skipped. The first sbatch request was
  rejected before a Job ID due an explicit memory field; corrected v2 script
  SHA-256 `1ac8f120...de138` submitted diagnostic Job `1178357`. Formal Jobs
  `1177779-1177782` remain untouched.
- 2026-07-21 10:23 +08:00: Re-audited the four formal Jobs and trajectory Job.
  Jobs `1177779-1177782` were all RUNNING; exact-uniform and companion had
  completed epoch index 14 / update 1499 and entered epoch 15, while direct
  and homotopy were in epoch index 14. Reported losses were finite, K remained
  384 and memory was 8596--8598 MB. Each arm had exactly three isolated AMP
  skips recovered at replay 1/8; grep `FAIL` hits were only fail-closed config
  field names. No Traceback, OOM, NaN/Inf or replay exhaustion existed.
  Detector-to-selector weight remained exactly zero before update 2100, so no
  current diagnostic was relabelled as end-to-end feedback evidence. Read-only
  Job `1178357` was RUNNING and consuming resources normally without errors.
- 2026-07-21 10:40 +08:00: Implemented and pushed the bounded normalized
  raw-delta residual audit on diagnostic commit `7f9ad10` (tree `c397d073`).
  It preserves the gamma-zero endpoint hard path, uses only deploy-visible
  transition scores and `abs_delta_p_action`, decodes every gamma with the same
  exact-K=384/G=2 solver and evaluates GT only after selection. Clean Linux
  PyTorch tests were 9 passed. Bundle SHA-256 is `651bdd63...f8383`; clean
  snapshot is `opentad_duca_diag_7f9ad10_20260721`. Submitted Job `1178384`
  with `afterok:1178357`; script SHA-256 is `f797af69...d997b`. It is a
  diagnostic-only successor test and changes no formal arm or checkpoint.
- 2026-07-21 10:45 +08:00: Corrected an important interpretation error in the
  epoch-4 hard-set audit. All learned `1178004` records contain
  `policy_mix_alpha=1.0` because selector-only export uses `eval()` and
  `forward_test`; they are inference-endpoint probes, not the alpha about 0.03
  hard paths consumed by epoch-4 training. Exact `1178357` replay for the first
  two completed arms keeps every audited path uniform through alpha 0.1; first
  changes start at alpha 0.3 and mean first-change alpha is about 0.34. The
  alpha=1 endpoint still replaces about half the uniform budget and is poorly
  boundary-aligned, but the old claim of a global hard-path jump at alpha 0.03
  is invalidated. Coarse-grid interval swaps are cumulative and are not
  evidence of one instantaneous threshold jump.
- 2026-07-21 12:12 +08:00: Implemented the bounded DUCA two-stage curriculum
  on isolated branch `codex/duca-two-stage-curriculum-20260721`. P0 is a true
  detector-skipping 20-epoch frontend stage with three preregistered loss
  scales and a deterministic training-only 80/20 split. Official-60 retains
  exactly 6000 successful updates and uses its first 1000 for exact-uniform
  AdaTAD warmup with all frontend/bridge weights zero, followed by policy and
  detector-gradient ramps. Four matched terminal-EMA arms are frozen.
- 2026-07-21 12:12 +08:00: Initial serial Job `1178480` failed before any
  optimizer update because P0 inherited a frozen-backbone optimizer field that
  PyTorch AdamW does not accept. Exact commit
  `6f2ed48d8cb31fe984b8a20223a0624fcf95d4b9` deletes that field, adds a
  fail-closed submission preflight and passed 83 clean remote tests.
  Replacement serial Job `1178487` entered epoch 0 and started real optimizer
  updates. Status is `experiment_running`; no P0 winner or mAP exists.
- 2026-07-21 12:12 +08:00: Job `1178487` reached P0 step 20 with finite total
  loss `0.8680`, raw action/transition/boundary losses
  `0.7413/6.3456/0.0073`, exact K=384, detector loss and bridge weight both
  zero, and `duca_detector_path=skipped`. This verifies the intended
  frontend-only execution path for an initial optimizer interval; it does not
  establish P0 quality, convergence or detector mAP.
- 2026-07-21 12:31 +08:00: Slurm accounting showed selected-axis Jobs
  `1177779-1177782` all `FAILED/1:0`; all four logs contain `Disk quota
  exceeded` around epoch 26--27. The common storage failure invalidates no
  model numerics but leaves the matrix without terminal EMA or mAP. The shared
  `/data` filesystem reported 100% use while two-stage Job `1178487` remained
  RUNNING with a stale P0 log near epoch 1. Registered both as infrastructure
  evidence and froze scientific interpretation pending storage recovery.
- 2026-07-21 13:22 +08:00: Completed a fail-closed checkpoint retention audit
  over `/data/run01/sczc063/yuzibo`. Reduced 909 numeric training records to
  220 CRC-validated records, exactly one per independent directory; deleted
  689 files/334,791,638,367 bytes and restored about 310 GiB free space. Data,
  pretrained weights, environments, logs/configs/results and the external best
  symlink were preserved. Consolidated manifest SHA-256 is
  `a06d3062a1fc2f8ec9d1ef336271f688368dfe2c788fea6933d8cc9e1a04b60a`.
- 2026-07-21 13:22 +08:00: Verified all four interrupted selected-axis arms can
  resume from retained epoch-24 checkpoints. CPU load exposed model, EMA,
  optimizer, scheduler, GradScaler and RNG state in every arm. This is only
  recoverability evidence; no terminal epoch-59 EMA or mAP exists.
- 2026-07-21 13:22 +08:00: Reconciled two-stage Job `1178487` as immutable
  `FAILED/1:0` at 12:33:31 after 24m02s. It stopped in P0 epoch 1 under the
  storage outage and wrote no checkpoint; early finite losses remain
  execution evidence only.
- 2026-07-21 13:34 +08:00: Deployed the exact selected-axis continuation from
  all four hash-bound epoch-24 states. Resume gate `1178581` entered RUNNING;
  exact-uniform/direct/homotopy/companion Jobs `1178582-1178585` wait on its
  `afterok` dependency. Commit `cb89586`, frozen config hashes, seed 3407,
  five-epoch checkpoint interval and terminal epoch-59 EMA remain unchanged.
- 2026-07-21 13:34 +08:00: Two-stage parallel-DAG precheck passed on exact
  commit `6f2ed48`, but Slurm rejected atomic submission with
  `AssocMaxSubmitJobLimit`; transactional rollback prevented a partial suite.
  Submitted protocol-equivalent serial Job `1178591` from fresh root
  `duca_twostage_6f2ed48_serial_20260721_133422`. It is pending on priority and
  restarts P0 because failed Job `1178487` produced no checkpoint.
- 2026-07-21 13:44 +08:00: Resume gate `1178581` completed `0:0` and sealed
  all four full-model plus checkpoint-load contracts. First arm Jobs
  `1178582-1178585` then failed before model construction because the generated
  wrapper verified but did not export `DUCA_EXPECTED_COMMIT`. Registered this
  as one launcher defect with zero optimizer updates, stopped the obsolete
  instances, and submitted corrected v2 Jobs `1178614-1178617` from fresh root
  `duca_selected_axis_cb89586_resume_e24_v2_20260721_134433`.
- 2026-07-21 13:44 +08:00: Two-stage Job `1178591` is RUNNING. Its first P0
  candidate reached step 40 with finite losses, exact K=384, zero detector
  loss/bridge and an explicitly skipped detector path. No frontend selection
  or mAP claim was promoted.
- 2026-07-21 14:12 +08:00: Registered selected-axis continuation launcher
  failures `1178614-1178617` (missing canonical `BASE`) and
  `1178633-1178636` (missing gate-suite environment). Both stopped before
  checkpoint restoration and have zero optimizer updates. v4 recomputed every
  runtime binding against all four epoch-24 checkpoints under the original
  official-60 environment; all matched, with preflight SHA-256
  `dae2775878465da16417faf33e20236bf8658f5d2df317b1ec9e5dda72d009d1`.
- 2026-07-21 14:12 +08:00: Submitted two-GPU/two-wave continuation Job
  `1178642` from root
  `duca_selected_axis_cb89586_resume_e24_v4_20260721_135701`. Exact-uniform
  and direct-0.25 both restored epoch 24 and entered epoch 25; homotopy-0.25
  and its uniform companion remain the ordered second wave. No mAP exists.
- 2026-07-21 14:12 +08:00: Two-stage serial Job `1178591` remained healthy at
  P0 candidate `a1_t005_b8`, epoch 7/step 580. Losses were finite, requested K
  was 384, detector loss/bridge stayed zero and the detector path was skipped.
  No holdout winner or terminal detector result exists.
- 2026-07-21 14:17 +08:00: Selected-axis exact-uniform and direct-0.25 each
  completed the first resumed epoch and entered epoch 26. Both advanced the
  selector schedule from 2500 to 2599 with finite losses and exact effective
  K=384. Exact-uniform kept detector bridge zero; direct-0.25 reached bridge
  weight `0.0623`. Its one AMP skip replayed successfully. Full log scan found
  no Traceback, OOM, disk-quota, non-finite or fail-closed error.
  Cross-route runtime-validation SHA-256 is
  `3af133daa84e8d31de2c8cb5b08ca30b440a0e381461030e4007e82c9466c0b5`.
- 2026-07-21 14:17 +08:00: Two-stage Job `1178591` reached P0 epoch 8/step
  700 with finite losses, requested K=384, zero detector loss/bridge and the
  detector path skipped. No P0 winner or mAP exists.
- 2026-07-21 15:30 +08:00: Ingested the exact-commit DUCA two-stage Pro audit.
  Raw SHA-256 is
  `0b265d08b811b821b1014cf7c52b579a759ee79e637710260a48cfc284367379`.
  Independent verification on the clean remote `6f2ed48` snapshot confirmed
  hidden selector-loss defaults in P0 cost, transition gradients into ASFormer,
  shared-AdamW warmup without byte-invariant freezing and inclusive schedule
  boundaries. Registered the implementation as HOLD and Job `1178591` as
  protocol-invalidated diagnostic evidence only.
- 2026-07-21 15:30 +08:00: Accepted strict loss/gradient ownership, isolated
  optimizer transactions, bounded residual selection and legal hard-swap
  utility as the leading repair principles. Did not accept the proposed local
  radius, detached detector-loss teacher, architecture uniqueness or numeric
  GO/KILL thresholds as proven facts; terminal matched mAP and total cost
  remain decisive.
- 2026-07-21 16:10 +08:00: Ingested the second exact-commit DUCA two-stage
  route audit. Raw SHA-256 is
  `bca69084bfb1c09f5fe92d49aa10362b18fecf69ff8d2fa754c1d53335734703`.
  Independent remote inspection confirmed its additional entropy, BCE,
  radius, descriptor and padding-BatchNorm findings. The implementation
  remains HOLD and Job `1178591` remains diagnostic-only.
- 2026-07-21 16:10 +08:00: Recorded that the two audits are not fully
  identical. V1 proposes detached legal hard-swap utility distillation; V2
  proposes a local-cell hard-forward/soft-RGB-backward bridge. Froze the next
  decision sequence as local-family reachability, P0 contract repair and one
  shared real hard-swap alignment gate before matched `U/D/R0/R1` mAP runs.
- 2026-07-21 16:06 +08:00: Holdout export Job `1178738` completed with 120
  selector-only training-holdout records. Five/eight-row exact pilots found
  the local GT oracle matched the global GT oracle on all reported boundary
  metrics, while the current checkpoint lagged. Full evidence is still
  running and no mAP claim was made.
- 2026-07-21 16:06 +08:00: The full oracle run failed closed on real record 7
  when a 20-bit lexicographic HiGHS solve returned values -1 and 2. The same
  row passed exactly with 8-bit blocks; restarted the 120-row audit with the
  stable protocol.
- 2026-07-21 16:06 +08:00: Implemented the bounded P0 contract repair in the
  isolated `codex/duca-local-residual-20260721` worktree: complete loss
  inventory, graph-free inactive losses, class-balanced BCE, detached
  transition evidence, GroupNorm, frontend-only optimizer evidence and no
  global clipping. Local contract/solver suites are `23 passed`; commit and
  Linux/CUDA evidence remain pending.
- 2026-07-21: Completed the exact 120-record local-reachability audit over 40
  training-holdout videos. Local and global privileged GT oracles match on all
  reported boundary and both-endpoint coverage radii; mean endpoint distance
  is `0.2484` versus `0.2462`. Coarse AUROC/AUPRC are only `0.6161/0.3750`,
  and the invalidated checkpoint trails uniform and pure delta. The bottleneck
  is learning/supervision rather than local-family reachability; no detector-
  mAP claim is made.
- 2026-07-21: Stabilized the oracle solver by retaining exact zero-gap pins for
  every semantic objective and disabling only the final position tie-break.
  Full records SHA-256 is
  `362aa4b22a5fa56e4a393bdbdba025f2ea47afa094d34ec43bda92a7e459b2e2`.
- 2026-07-21: Simplified P0 deployment to one real one-step CUDA gate followed
  by three sequential frontend candidates. Added a frontend-only stop before
  unrepaired old official-60 arms. Local pure verification is now `25 passed`;
  Windows Torch-dependent tests remain skipped pending Linux/CUDA.
- 2026-07-21: Committed and pushed the repair as exact commit `5d17dcb` on
  `codex/duca-local-residual-20260721`. The clean remote snapshot passed
  `96` focused Linux tests with `2` skips. Submitted only fail-closed
  frontend Job `1178774`; its real CUDA gate must pass before any candidate
  update, and `DUCA_FRONTEND_ONLY=1` prevents old official-60 execution.
- 2026-07-21 16:57 +08:00: Job `1178774` failed before candidate training on
  gate evidence classification: real parameter names contain `spatial_stem`,
  while the gate searched for obsolete `spatial_encoder`. Corrected only this
  classifier and added a regression test in exact commit
  `9442b9487f871efd02c85dceeed26574c641369d`.
- 2026-07-21 16:57 +08:00: Clean remote verification at `9442b94` passed
  `74` focused tests with `3` skips. Submitted replacement frontend-only Job
  `1178809`; it retains the same split, model, losses and one-gate/three-candidate
  sequence and cannot enter the old official-60 route.
- 2026-07-21 17:55 +08:00: DUCA local residual implementation pushed as commit 6c56e11 on branch codex/duca-local-residual-20260721. Remote academic-accelerated shallow clone succeeded and exact snapshot is /data/run01/sczc063/yuzibo/projects/opentad_duca_local_6c56e11_20260721_accel. Focused Linux evidence: 87 passed/2 skipped at 56c2683 and 19 passed at 6c56e11 plus serial submit precheck. Submitted frontend-only P0 Job 1178863 under /data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_local_residual_6c56e11_p0_20260721_175500. Status is experiment_running only; no P0 winner, official60 mAP, or greater-than-65 claim yet.
- 2026-07-21 18:15 +08:00: User route裁决：one-frame-per-local-cell / 每两帧必须选一帧不符合最初 DUCA 设计初心。它可以作为 exact-uniform 附近的诊断、保底或局部微调对照，但不能再包装为最终主方法，因为它不能把背景区域预算释放给边界/状态转变密集区域，选帧自由度本质上接近均匀采样局部抖动。当前 running 的 local-residual P0 Job 1178863 只保留为机制诊断，不应自动解锁 paper-grade official60 主矩阵。下一步主线应转向覆盖受保护但允许跨区域预算转移的选择族：粗动作二分类负责产生 p_action/状态变化证据，selector 以边界/状态转变覆盖为首要目标，并在严格 max-gap/coverage 约束下学习把更多帧分配到高转变区域。
- 2026-07-21 18:15 +08:00: Completed a fresh inventory of all 41 top-level
  `OpenTAD*` directories, 20 registered worktrees, the nested protected-
  physical tree, and seven key DUCA source lineages. File-hash and call-site
  checks confirm that full-window `global_structured_topk` already provides
  exact-K/max-gap selection with cross-region quota transfer; the formal
  selected-axis source is `cb89586`. The apparent proliferation of trees is
  mostly shared ancestry, not independent models. Added
  `duca_model_version_registry.md`, refreshed `worktree_inventory.md`, and
  froze anti-repetition rules 209--212. The next admissible implementation is
  a narrow reuse of the existing global selector plus P0/curriculum/protected-
  gradient repairs; local-cell remains diagnostic only.
## 2026-07-21 DUCA 全局课程主线冻结

- 完成 41 个顶层 OpenTAD 目录、20 个登记 worktree 与嵌套 protected-physical
  树的代码/版本核对；确认 `global_structured_topk` 已经实现全局 exact-K、
  max-gap 约束和跨区域预算转移，禁止再次实现同类 selector/decoder。
- 创建隔离树 `.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721`，从 `9442b94`
  派生，保留 P0 修复与两阶段课程，明确排除 `56c2683` 之后的 local-cell 路线。
- 冻结 matched U/G0/G1/G2 设计。当前新增代码工作的唯一核心缺口是 G2
  uniform companion 造成 learned-row detector-gradient exposure 约减半，必须按
  实际 learned-row 比例归一化；不得借此新建模型家族。

## 2026-07-21 18:51 DUCA 全局课程精确提交与排队

- 在隔离树 `.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721` 完成唯一允许的
  受控合并，分支 `codex/duca-global-curriculum-20260721`，精确提交
  `4c777a691d65fe484dfe537ac3e33f82b5bbe5a8`，已推送 GitHub。
- 没有新建 selector、decoder、detector wrapper 或 profiler；复用
  `global_structured_topk`、P0 修复、两阶段课程和 protected structured
  transport。新增核心合同仅为 G2 learned-row 梯度曝光归一化，并冻结 matched
  U/G0/G1/G2 配置与 gate。
- 首次远端 focused run 在前一提交发现 `bridge_row_scale` 被误传入
  `structured_zero_forward`；修复后的精确提交在干净 Linux 快照通过
  `74 passed, 2 skipped`，该测试证明代码合同，不证明 mAP。
- 正式串行 Slurm Job `1178911` 已从
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_4c777a6_20260721`
  提交，run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_4c777a6_serial_20260721_1849`；
  receipt 将 Job/commit 绑定到 split-manifest SHA-256
  `3e98c3fff0e24fe50003e6af3cad7f88e02b32fed8161dfb470a445cb875059a`；
  18:54 +08:00 仍为 `PENDING (Priority)`。它将顺序执行 P0、精确全模型门禁和
  U/G0/G1/G2 official-60；当前没有 terminal mAP，状态为
  `experiment_running`。
- 旧 Job `1178642` 继续作为 selected-axis 前驱诊断，Job `1178863` 继续作为
  local-cell/P0 诊断；二者均不得替代 `1178911` 的 matched 主线证据。

## 2026-07-21 19:00 全树核对与 selected-axis 部分终局结果

- 复核 41 个顶层 OpenTAD 目录、20 个登记 worktree 和两个嵌套 DUCA 克隆；
  核心文件哈希再次确认多数目录是同一继承链快照，不是独立模型。当前唯一主线仍是
  `4c777a6` 的 U/G0/G1/G2；共享源码中保留 local-cell 函数仅为历史可复现性，
  主线配置和测试均强制 `global_structured_topk`。
- Job `1178642` 的 exact-uniform 与 direct-0.25 已产生 terminal epoch-59 EMA：
  Avg-mAP 分别为 `64.4580` 和 `63.7102`，direct 低 `0.7478`，且五个 IoU
  阈值均更低。该负结果禁止再次以新名字重跑旧 direct 联合训练，但不替代
  `4c777a6` 对 P0 预训练、冻结粗分类与 scorer-only 受保护梯度的检验。
- homotopy/companion 仍在运行，Job `1178911` 仍为 `PENDING`；超过 65、C3、
  C4 和论文主方法身份均未获证明。

## 2026-07-21 19:15 DUCA 版本收口锁

- 应用户要求再次检查全部 43 个相关 OpenTAD Git 树，并对照源码调用点、核心文件
  哈希、配置和测试确认：需要的“覆盖受保护但允许跨区域预算转移”模型早已存在于
  `global_structured_topk`，无需也禁止再次实现 selector/decoder。
- 反复回跳的原因被固定为历史教训：早期全局 scorer 聚集位置不准后，路线用
  one-frame-per-cell 收缩可行集换取稳定性，却改变了原始科学问题。local-cell 现在只
  是诊断/消融，不能再升格为最终方法。
- 唯一活动主线锁定为 `4c777a6` 的 U/G0/G1/G2。新增 anti-repetition 规则 219：需求
  已能映射到版本注册表时只能复用原实现；未先登记可验证缺口，不得新建同义模型、
  worktree、配置族或启动器。
- 19:12 +08:00 远端核对：Job `1178911` 仍为 `PENDING`；前驱 Job `1178642` 的第二波
  homotopy/companion 正在运行；local-cell/P0 诊断 Job `1178863` 正在运行。它们的
  证据角色保持分离，不得互相替代。

## 2026-07-21 19:20 P0 学习速度与检查点选择缺口

- 在不改 selector 架构的前提下逐行复核 P0，确认 `4c777a6` 的检查点选择仍把已
  饱和的 radius-one boundary recall 作为硬门和第一排序键，直接违反规则 203。
- 当前 P0 的 coarse trunk/action head/scorer 学习率为
  `2.5e-5/5e-5/1e-4`；随机初始化 coarse trunk 的学习速度只有 scorer 的四分之一。
  Job `1178863` 到 epoch 16 的 actionness BCE 仍约 `0.68`，与历史“过早且错位聚集”
  现象一致，但最终 AUROC 尚待 holdout 导出，不能只凭 BCE 下定论。
- 逐行核对 Uni-AdaFocus 官方提交 `8846488`：hard temporal indices 来自
  `weights_T.detach()`，策略靠 Monte-Carlo auxiliary task loss 学习；其 temporal
  policy/global CNN LR multiplier 是 `0.2/0.5`。决定只迁移分组学习率与辅助任务原则，
  不声称 hard-index 直通，也不立即引入 MobileNet。
- 冻结一个有界 P0 successor：保留同一 `global_structured_topk` 和 U/G0/G1/G2，固定
  三项 loss 比例，比较旧 LR 控制与两个 coarse-first LR 组；检查点改由 r0、短动作
  双端 r0、端点距离、transition AUROC 和 coarse AUROC/AUPRC 决策。`1178911` 零运行时
  pending 作业在替代提交通过测试前标记为 protocol HOLD。

## 2026-07-21 19:42 P0 合同修订与主线替换

- 在原 V8 全局模型上完成有界修订，精确提交
  `e0397ec0bcb917593664ce36efd8105e31d0a302` 并推送同一分支。没有新增
  selector、decoder、detector wrapper，也没有修改 local-cell、SparseHead、
  Spatial-Zoom 或 ChronoTransport。
- P0 三臂固定同一损失 `1.0/0.10/16.0`，只比较
  `2.5e-5/5e-5/1e-4`、`5e-5/1e-4/2.5e-5`、
  `1e-4/2e-4/5e-5` 三组 coarse/action/scorer 学习率。真实 CUDA 门禁新增逐参数
  学习率落地核验，避免“配置声明正确、优化器实际分组错误”。
- 检查点选择删除饱和 radius-one 硬门和第一排序键，改用 r0、短动作双端 r0、
  端点距离、transition AUROC 与 coarse AUROC/AUPRC；r1 只作诊断。远端精确
  快照通过 `158 passed, 3 skipped`，直接受影响集合为 `27 passed`。
- 旧 Job `1178911` 在替代验证期间开始运行，进入首个旧 P0 臂后于运行
  `00:09:01` 被取消，日志保留但不得作为证据。新 Job `1178927` 从精确快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_e0397ec_20260721_v2`
  提交；run root 为 `duca_global_e0397ec_serial_20260721_1939`，初始状态
  `PENDING (Priority)`。超过 65 和论文主方法身份仍未证明。

## 2026-07-21 19:51 质量导出路径修复与最终重排

- 已失败 local-cell 诊断 Job `1178863` 实际完成了首个 20-epoch P0 训练，最终
  `actionness_bce_loss` 仍约 `0.684`；失败发生在训练后质量导出：用文件路径启动
  脚本后，Python 无法解析顶层 `tools` 包。该作业不是 local-cell 数值失败证据。
- 因 `e0397ec` 启动器继承同一调用方式，零运行时 Job `1178927` 被主动取消。
  原脚本两处调用改为 `python -m tools.bata...`，并加入防回归测试；没有修改模型。
- 最终活动精确提交为 `2c403a853d55057ae772e1b8dcc0c4ebb8cbc0f5`。
  远端精确快照受影响集合 `28 passed`，两个模块的 `--help` 实际解析成功，
  PRECHECK 通过。
- 新 Job `1178933` 已提交，run root 为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_global_2c403a8_serial_20260721_1949`，
  初始状态 `PENDING (Priority)`；回执和 split SHA-256 已登记。当前仍无 P0
  holdout winner 或 terminal mAP。

## 2026-07-21 20:00 全树模型身份再次冻结

- 再次核对 41 个顶层 OpenTAD Git 目录、20 个注册 worktree 与 2 个嵌套 DUCA
  clone，共 43 个相关 tree；没有发现比 V8 更新且同时满足全局 exact-K、
  max-hole=2、跨区域预算转移、P0 修复和受保护 detector gradient 的另一实现。
- 当前唯一主线明确为
  `codex/duca-global-curriculum-20260721@2c403a853d55057ae772e1b8dcc0c4ebb8cbc0f5`。
  U/G0/G1/G2 是同一个 `global_structured_topk` 模型的四个配置，不是四个模型；
  local-cell、CellCF 和 local-residual 仅为历史诊断。
- 修正模型注册表里仍指向 `4c777a6/1178911` 的旧主线指针，增加配置级冻结证书与
  anti-repetition rule 227。此记录轮没有修改 DUCA 模型代码、SparseHead、
  SpatialZoom 或 ChronoTransport，也没有再创建 selector、decoder 或 worktree。
- 已建立一次性 21:00 当前任务复核 `duca-21-00-full-progress-report`；它只监控
  当前唯一精确提交/Job 与前驱诊断 `1178642`，并要求把结果写回同一 wiki，不允许
  借监控结果另起 selector/decoder/worktree。

## 2026-07-21 20:20 完整串行入口修复与重新部署

- 继续逐行审计发现两个聚合器仍以文件路径执行，且本地均可复现
  `ModuleNotFoundError: No module named 'tools'`。在同一 V8 分支只改两处为
  `python -m tools.bata...`，增加防回归测试并推送精确提交 `6b6363e`；没有修改
  selector、decoder、P0 三组参数、U/G0/G1/G2、AdaTAD 或 ActionFormerHead。
- 本地 module 入口通过；相关测试 `14 passed, 1 skipped`，skip 为本机 Torch DLL
  限制。远端精确快照通过 `18 passed`、`py_compile`、`bash -n`、HEAD 与 clean-
  tree 检查。
- 旧 Job `1178933` 恰在兼容性审计临时文件位于旧 snapshot 时启动，clean-tree
  门禁在 `00:00:03` 正确失败，没有模型构建或 optimizer update。临时文件已删除，
  旧 snapshot 重新核验 clean；anti-repetition 新增“精确快照内禁止临时文件”。
- 新唯一活动 Job `1178947` 已从
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_global_6b6363e_20260721`
  提交，run root 为 `duca_global_6b6363e_serial_20260721_2022`，初始状态
  `PENDING (Priority)`。receipt/split SHA-256 分别为 `af788b...afd5` 与
  `c0fb81...f3fc`。
- 现有 standalone official-ASFormer checkpoint SHA-256 `34e4d510...5f3bbba`
  已做严格兼容性核验：BatchNorm 可加载，当前 GroupNorm 因六个 running-stat/
  counter 键拒绝加载；禁止静默 `strict=False`。只有当前三组 P0 全部失败后，才
  允许讨论逐键登记的 BN-to-GN warm-start，不另建 coarse 模型。

## 2026-07-21 20:42 全树去重冻结与 P0 门禁修复

- 再次扫描 41 个顶层 OpenTAD Git tree、20 个登记 worktree 和两个嵌套 DUCA
  clone，共 43 个相关 tree。十二个 tree 含 DUCA 源码，但哈希和配置审计仍只归并为
  V0-V8 模型族；没有发现缺失的全局 exact-K/max-gap/跨区域预算转移 selector。
- 明确重复回跳根因：早期全局策略出现错位聚集后，路线用 local-cell 收缩可行集来
  换稳定性，实际上改变了研究问题。以后不得以训练不稳为由重新实现 one-frame-per-
  cell 或同义全局 decoder，只能在 V8 原实现上诊断粗证据、scorer 和训练合同。
- Job `1178947` 于 20:24 开始，运行一分钟后在 P0 真实门禁失败，没有 optimizer
  update。失败不是模型分组错误，而是门禁把官方 ASFormer 注意力层内部的
  `conv_out` 误判为二分类 action head。真实 `ActionFormer.get_optim_groups()` 已按
  拓扑正确分组。
- 在同一 V8 分支只修正门禁拓扑判定并加入四个参数名回归案例，精确提交
  `91381568637f6358bdec67e3d8400d70869f1dd6` 已推送。新远端干净快照的针对性回归
  `15 passed`，`py_compile`、`bash -n`、HEAD 与 clean-tree 均通过。
- 新唯一作业 `1178975` 已从 run root
  `duca_global_9138156_serial_20260721_2042` 提交。它仍按原顺序执行真实 P0 gate、
  三组 P0、winner、U/G0/G1/G2；没有创建新模型、selector、decoder 或 worktree，
  也没有触碰 SparseHead、Spatial-Zoom 或 ChronoTransport。

## 2026-07-21 20:53 全量目录复扫与唯一版本锁

- 以展开全部 untracked 文件的同一口径重新读取 41 个顶层 OpenTAD Git tree：HEAD、
  branch 与 dirty count 均写入 `worktree_inventory.md`。协调仓库当前为 189 个 dirty
  path，UniCompanion 诊断树为 15 个；这里只记录，不清理、不覆盖用户改动。
- 20 个登记 worktree 与两个嵌套 DUCA clone 也完成核对。当前全局课程 clone 精确为
  `9138156` 且 clean；physical clone `ee05f61` 只保留为历史组件源。十二个含 DUCA
  源码的 tree 仍归并为 V0-V8，没有发现第九套需要采用的模型实现。
- 活动配置逐文件复核：G0/G1/G2 全部声明 `global_structured_topk`，测试显式拒绝
  `local_cell_deformation`；U 只是同协议 exact-uniform 控制。以后 local-cell 仅可
  作为 V2/V7 历史诊断，不能因训练稳定或命名相近再次替换 V8。
- `anti_repetition.md` 中三个残留旧指针已统一到 `9138156 / 1178975`；旧
  `1178947` 明确标为门禁分类器失败而非模型版本。20:53 远端查询显示 `1178975`
  仍为 `PENDING (Priority)`，前驱诊断 `1178642` 仍运行；超过 65 尚无证据。

## 2026-07-21 21:00 目标时点完整状态

- 已完成并部署三个独立 P0 学习率版本：control、coarse-first moderate、
  coarse-first strong。三者共享相同 ASFormer coarse probe、transition scorer、
  `1.0/0.10/16.0` 三项监督和 global exact-K/max-hole 策略，只比较组件学习速度。
- 同一串行 DAG 已包含后续 U/G0/G1/G2 四个 matched official-60 配置；它们是同一
  V8 的控制/消融，不是新模型。精确提交仍为 `9138156`，Job `1178975` 因
  `Priority` 保持零运行时，尚未执行 real-CUDA P0 gate。
- 复核 Uni-AdaFocus 官方源码：其 temporal hard indices 明确使用
  `weights_T.detach()`，同时通过 temporal/global/local/auxiliary task losses 和
  分组件 `lr_mult` 学习政策。V8 合理复用的是分组件学习率、辅助监督和训练期稳定
  companion，而不是虚构“下游损失穿过离散索引”的梯度。
- V8 official-60 继承链复核通过：前 1000 updates 为 exact-uniform detector
  warmup，随后 1500 updates 将全局策略从 uniform 平滑至 learned，检测反馈从
  update 2500 后再用 1500 updates 增至 0.25；P0 后 coarse probe 冻结，检测反馈
  只更新 transition scorer。没有日程被错误覆盖为全程均匀。
- 前驱 `1178642` 的 homotopy/companion 运行到约 epoch 52/53，loss 有限、K 请求为
  384，孤立 AMP skip 均可重放；尚未生成 terminal mAP。当前可引用的终局结果仍仅
  为 exact-uniform `64.4580` 和 direct-0.25 `63.7102`，超过 65 未证明。

## 2026-07-21 21:20 全树复核、EMA 门禁修复与同一 V8 重排

- 实际重扫 `43` 个相关 Git tree（41 顶层 + 2 嵌套），其中 12 个含 DUCA 模块；
  核心源码哈希已写入 `duca_model_version_registry.md`。仍只归并为 V0-V8，没有
  发现需要另写的全局 selector、decoder 或第九套模型。
- V8 活动配置再次核对为 `global_structured_topk`、K=384、max-hole=2；共享源码中的
  `local_cell_deformation` 只用于 V2/V7 历史复现。未修改 SparseHead、SpatialZoom、
  ChronoTransport 或协调仓库中的模型代码。
- Job `1178975` 真实完成一个有限 P0 optimizer step，coarse/scorer 均有梯度并更新，
  detector 明确 skipped；失败来自 EMA 门禁只看单个代表参数时的 FP32 舍入，不是
  模型、损失或优化器失败。
- 同一 V8 分支仅修改门禁和测试，推送精确提交
  `63e25eb17e523d369f73434ed4d9b6446608861a`。远端受影响合同 `21 passed`，
  `py_compile` 和 diff check 通过。
- 通过学术代理创建干净快照 `opentad_duca_global_63e25eb_20260721`；同一串行
  P0/U/G0/G1/G2 预检通过并提交 Job `1178989`，run root 为
  `duca_global_63e25eb_serial_20260721_2120`。初始状态 `PENDING`，无 terminal mAP，
  超过 65 仍未证明。

## 2026-07-21 21:27 唯一主线无重写复核

- 隔离 DUCA 树仍为干净的 `63e25eb` V8；`1178989` 仍因 `Priority`
  保持零运行时。没有新的 gate 失败或模型证据，因此不新建 selector、
  decoder、model class、config family 或 worktree。
- 前驱 `1178642` 的 homotopy/companion 分别进入 epoch 58/57 of 60，
  loss 有限，但均未生成新的 `terminal_evaluation.json`。已封存终点仍仅为
  uniform `64.4580` 与 direct `63.7102`。
- 本轮只更新同一 V8 实验记录，没有修改任何模型、local-cell、
  SparseHead、Spatial-Zoom 或 ChronoTransport 代码。

## 2026-07-21 21:41 V8 真实 P0 门禁通过

- Job `1178989` 已获得 Slurm GPU 并生成 `p0_real_gate.json`，`ok=true`，
  精确绑定干净提交 `63e25eb`。
- 门禁使用真实 THUMOS batch；完整 AdaTAD 对象被构建，但 P0 中 detector/
  backbone/head 调用数均为零，冻结 detector 字节不变。
- actionness 只更新 coarse 组，transition/boundary 只更新 scorer；68 个
  可训练参数均获得梯度。一次真实 AMP/AdamW/scheduler/EMA 更新通过，
  group-wide EMA 证据分别覆盖 60/62 与 5/6 个参数。
- 硬选择仍为 exact K=384、max-hole=2；仅三个显式 P0 损失活跃。
  第一个 control-LR 候选已进入 epoch 0 并保持有限损失；尚无 holdout
  winner 或 terminal mAP。

## 2026-07-21 22:07 V5 四臂终局与 V8 P0 进度

- 前驱 Job `1178642` 已 `COMPLETED/0:0`。epoch-59 EMA Avg-mAP 为：
  exact-uniform `64.4580`，direct-0.25 `63.7102`，homotopy-0.25 `63.0601`，
  homotopy+uniform-companion `63.6931`。三个 learned V5 臂均低于 uniform。
- uniform companion 相对 homotopy 恢复约 `0.6330` Avg-mAP，但仍低于 uniform
  `0.7649`。这些臂已写入防重复清单，禁止换名重跑。
- V8 Job `1178989` 仍健康运行，第一 control-LR P0 候选进入 epoch
  6/20；actionness BCE 约 `0.68--0.69`，detector 仍明确 skipped，无 Traceback、OOM
  或 non-finite collapse。尚无 holdout winner 或 V8 terminal mAP。

## 2026-07-21 22:14 V8 P0 实时检查

- Job `1178989` 仍在 `g0006` 健康运行，第一组 control-LR 候选已完成
  epoch 7 并进入 epoch 8/20。最新总损失 `1.4643`，actionness BCE
  `0.6840`，损失均有限。
- detector 路径仍明确 `skipped`，请求和实际预算均为 `384`；未发现
  Traceback、OOM、ValueError 或 non-finite。尚无留出集优胜者或 V8 mAP，
  状态仍为 `experiment_running`。

## 2026-07-21 22:22 V8 GitHub 同步与 Pro 审查 Prompt

- 活动隔离树 `codex/duca-global-curriculum-20260721@63e25eb` 经 fetch 后与
  GitHub ahead/behind 为 `0/0`，显式 push 返回 `Everything up-to-date`；未制造
  新提交，也未混入协调根的 SpatialZoom、历史文档或其他路线改动。
- Job `1178989` 第一组 control-LR P0 候选已完成 epoch 8 并进入 epoch 9/20；
  最新总损失 `1.4665`、actionness BCE `0.6855`，detector 仍明确 skipped，
  无 Traceback、OOM 或 non-finite。仍无 holdout winner 或 V8 mAP。
- 已生成精确提交 Pro 逐行审查 Prompt：
  `docs/methods/prompts/2026-07-21-duca-v8-pro-code-route-review-prompt.md`。
  Prompt 冻结 V8/U/G0/G1/G2 范围，并要求文件行号、梯度矩阵、根因证伪、
  一个首选有界改进及终点 mAP 裁决；禁止换名重建历史路线。

## 2026-07-21 max-hole=2 自由度审计

- 对 T=768、K=384、最大连续未选位置 G=2，覆盖约束最低需要约 256 个
  分布式骨架位置，只剩约 128 个位置可额外集中；因此它不等于均匀采样，
  但约三分之二预算受到强覆盖先验约束。
- G=2 对应最大选中中心间隔三个 dense candidates；THUMOS 候选步长为四个
  source frames，因此上限约 12 source frames。该值来自此前“不得超过15帧”
  的保守合同，而不是由 mAP 证明的最优值。
- 新增 gap `G22`，状态仅为 `discussed`。不得修改正在运行的 V8 精确提交；
  先使用同一保存分数做 G=2/3/5/7 的无训练 reachability/oracle 诊断，再决定
  是否值得在同一 V8 中运行一个单变量 max-hole mAP 消融。

## 2026-07-21 Uni-AdaFocus 深度方法与覆盖审计

- 对论文和官方 commit `8846488310fdd4a18412608006030643e794c36e`
  进行了逐实现复核。时间策略采用概率质量上的逆 CDF 分位采样与碰撞修复，只保证
  K 个有序唯一索引，不保证物理时间 max-gap、均匀覆盖或 TAD 边界召回。
- 硬时间索引来自 `weights_T.detach()`；策略通过 Monte-Carlo 期望分类代理和分组
  学习率训练。所谓联合训练不等于重分支损失穿过离散选帧索引。
- 其稳健性的关键还包括均匀低成本全局观察，以及在最终分类中复用全局特征。对 DUCA
  最重要的后继假设是“dense cheap coarse context + sparse heavy refinement”，但必须
  等 V8 Job `1178989` 终局后再做有界裁决，当前不修改模型或运行协议。
- ActivityNet 数值是视频分类 mAP，不是 TAD mAP；已写入来源边界和防重复规则。

## 2026-07-21 coarse hidden 与 VideoMAE 融合边界

- 逐行确认当前 V8 的 coarse hidden 仅供 transition scorer/选帧；选中原始帧再进入
  VideoMAE 和 AdaTAD，当前没有 detector feature fusion。
- 条件后继冻结为“VideoMAE 主表征 + timestamp-aware coarse context”：完整 coarse
  sequence 经独立 adapter，作为 selected VideoMAE token 的上下文，以零初始化门控残差
  融合。TAD loss 默认只能更新 fusion/VideoMAE/AdaTAD，不能重写 coarse action head。
- context fusion 与 canonical physical-grid gap filling 是两个不同假设；前者可先做且保持
  官方 detector 路径，后者才可能支持放宽 max-hole。二者必须分开验证。
- 状态仅为 `discussed_conditional_post_v8`；不修改正在运行的 V8 Job `1178989`。

## 2026-07-21 Uni-AdaFocus 融合方式的 DUCA 发散设计

- 官方 global/local raw features 并未直接相加：独立 MLP、各自时序 max pooling，
  再在最终分类器拼接；各支继续保留独立辅助损失。
- 登记三个互斥的 post-V8 假设：低风险 prediction fusion、首选 timestamp-aware
  zero-gated context、以及可支持放宽 max-hole 的 canonical-grid coarse fallback。
- 首选实验必须保持 gate=0 时逐值等于当前 VideoMAE/AdaTAD；coarse-to-detector
  adapter 可接受 stop-gradient heavy feature distillation，但 TAD loss 默认不得改写
  coarse action head/trunk。
- 任何使用 post-backbone 融合的版本都必须诚实称 acquisition-and-fusion adapter，
  不能继续声称严格的 pre-backbone-only plugin。当前仅讨论，不实施。

## 2026-07-21 DUCA × Uni-AdaFocus 独立 Pro 裁决 Prompt

- 新 Prompt 固定当前 DUCA `63e25eb`、Uni-AdaFocus 原文及官方 `8846488`
  为强制阅读对象，要求先给可见性证书并逐行重建两套计算图。
- Prompt 只提供代码与实验事实入口，不预设是否融合，也不把 late fusion、attention、
  gate、canonical grid、distillation 或纯选帧列为正确方向。
- Pro 必须先裁决是否应融合，再独立生成至少四个机制不同候选，淘汰至少一半，最后
  给出唯一首选、一个失败 fallback、梯度所有权、核心代码和最小 matched mAP 闭环。
- 文件：
  `docs/methods/prompts/2026-07-21-duca-uni-adafocus-coarse-videomae-fusion-pro-prompt.md`。

## 2026-07-21 `63e25eb` V8 Pro 审查吸收

- 将附件按字节归档到
  `docs/methods/reviews/2026-07-21-63e25eb-duca-v8-unique-endpoint-pro-review-raw.txt`；
  SHA-256 为
  `DF19960D0B3158CE7F31E0FE4A92F8CD22C7B2AAFD5FB78D13E91DDACEA8EC70`。
- 在本地精确干净 clone `63e25eb` 逐项复核，确认 Gaussian transition target
  同时进入 distribution 与 mass coverage、`exp(-mass)` 重复奖励、exact event
  primitive 未接入、bridge 是 surrogate、真实 hard-swap alignment 未进入当前
  formal gate、direct-gradient metadata 错误、schedule 一拍偏移及 G2 companion
  的精确作用域。
- 不完全接受 reviewer 的唯一首选：在 `max-hole=2` 下，radius-one 内部端点事件
  含三个连续位置，所有 feasible path 都必然命中，因此 exact coverage 概率恒为一、
  梯度为零。项目裁决为
  `SUBSTANTIAL_ACCEPT_DIAGNOSIS / REVISE_OBJECTIVE_BEFORE_IMPLEMENTATION`。
- 修订后的有界候选是先证明 nontrivial headroom，再测试 rounded radius-zero
  unique-endpoint objective；G1/G2 必须补真实 legal hard-swap alignment。
  当前 Job `1178989` 不改代码、不改状态，仍仅为 `experiment_running` 诊断证据。

## 2026-07-21 23:40 V8 第一组 P0 候选完成

- Job `1178989` 保持健康。`lr_control_c25_a50_s100` 完成 20 个 P0 epoch，
  `completion.json` 为 `ok=true`；这只证明训练和证据协议跑通，不代表候选胜出。
- 粗分类分支确实在学习：epoch 20 的视频宏平均 AUROC 为 `0.624512`，但选帧
  评分器的 radius-zero transition AUROC 只有 `0.521321`，低于直接使用
  `abs(delta p_action)` 的 `0.553237`。
- 学习选帧的 radius-one 端点召回为 `0.883237`，明显低于均匀采样
  `0.999775`；平均端点距离为 `0.538120`，也差于均匀采样 `0.477457`。
  预算仍为 K=384、mean max-hole=`1.916667`，因此问题是可行域内的位置分配，
  不是预算或硬间隔失效。
- 第一候选不满足既定质量门禁，不能被称为 winner。第二候选已进入 epoch 3，
  第三候选、holdout winner、U/G0/G1/G2 和 terminal mAP 均仍待完成；状态继续为
  `experiment_running`。

## 2026-07-21 Oracle 边界聚集目标纠偏

- 更正此前过强表述：目标不是“每个边界只选一帧、命中后不再聚集”。历史 GT
  Oracle 会对每个起点和终点选择中心及前后半径 2 的多个位置，再均匀补足剩余预算。
- 真正需要修复的是宽高斯带内没有中心、左右结构和配额上限的无约束 mass 堆积。
  后续有界目标应为“精确端点锚定 + 左右有支撑的限额边界微簇 + 剩余全局覆盖”，
  并对短动作的重叠端点去重。
- `radius=0` unique event 只能作为 anchor 项，不能单独代表最终目标。该纠偏仅更新
  设计语义，不修改冻结的 V8 `63e25eb` 或正在运行的 Job `1178989`。

## 2026-07-21 全版本边界中心/左右数量审计

- 复核 V0--V8 版本注册表及 PAction/GAS-VT、move25/move50、learned radius、
  Oracle 和 bracket 分析代码。结论是已有零件但没有完整模型。
- Oracle 唯一明确执行“每个 GT 端点中心及 `±2` 多帧微簇 + 剩余均匀填充”，但它在
  评估时使用 GT，不能部署。旧 `legacy_center_radius` 只有中心分数和一个对称半径；
  learned radius/dilation 没有每端点左右数量；GAS-VT 只有左右各命中一次的二值损失；
  lattice 会聚集但中心可偏移；V8 允许全局聚集却没有 center/左右配额语义。
- 登记 G23：唯一目标是复用 V8 transition scorer、global exact-K/max-hole DP 和官方
  AdaTAD 图，补充 deploy-visible transition center、限额左右多帧微簇、重叠端点去重
  与剩余全局预算。状态为 `designed_not_implemented`，不新建模型、不修改 Job
  `1178989`。

## 2026-07-22 DUCA 最终产物与完整改进计划冻结

- 将最终交付固定为一个 offline-TAD pre-backbone acquisition plugin，而不是新 TAD
  detector、Online TAD 或三个独立部署模型。首个正式 backend 继续使用
  official-derived AdaTAD/ActionFormer。
- 固定 canonical forward：dense cheap coarse/official-ASFormer evidence -> indirect
  transition center -> capped bilateral boundary burst -> overlap-aware saturation and
  residual global utility -> existing exact-K/max-hole DP -> hard original-time observations
  -> official detector。
- 固定训练归属：P0 中 action BCE 训练 coarse，anchor/bilateral/quota/context 训练
  scorer/burst；official stage 先 exact-uniform detector warmup，再冻结 coarse，仅在真实
  hard-swap alignment 通过后让 TAD loss 通过 protected bridge 更新 scorer/burst。
- 新建 preregistered 实验节点
  `exp:duca-oracle-calibrated-boundary-burst`，顺序为 R0 Oracle/KG 可达性、R1 数学代码
  门禁、R2 P0 单变量 objective、R3 U/G0 terminal mAP、R4 alignment/G1/G2、R5
  三种子/预算曲线/第二 detector/完整成本。
- 明确停止条件：constrained Oracle 无 mAP headroom、corrected G0 不超过 U、bridge
  alignment 失败或完整成本无净节省时，停止相应论文主张，不再以反馈、训练轮数或
  新 selector 掩盖失败。
- 当前状态仍为 `designed_not_implemented` / `designed_not_authorized`；没有创建 V9，
  没有修改冻结的 V8 Job `1178989`。
- 补充主锚点 GO 条件：terminal-EMA Avg-mAP `>=65.00`、相对 matched U
  `>=+0.20`、mAP@0.6/0.7 退化均不超过 `0.20`，且完整端到端实测成本低于 dense。
  固定 K=384 为主方法、K=256 为首个效率扩展；dynamic budget 暂不进入主张。

## 2026-07-22 Uni-AdaFocus / EU-CRR Pro 审查归档与独立裁决

- 将 65,069-byte / 1,426-line 原文按字节归档到
  `docs/methods/reviews/2026-07-22-63e25eb-duca-uni-adafocus-eucrr-pro-review-raw.txt`；
  SHA-256 为
  `0678A31C17D3FCD983726CE9056E463CF09A0325DAF69C7C41947EEB57602DAA`。
- 在本地干净 `63e25eb` 副本核验：coarse hidden 当前只供 selector；hard selected RGB
  直接进入 VideoMAE；detector 无 fusion；protected bridge 是 surrogate；VideoMAE
  post-processing 真正输出 `[B,C,K]`，且 K 轴经历 tubelet/chunk/interpolation。
- 接受 reviewer 对 Uni-AdaFocus 迁移边界、selected-rank/physical-time 风险、完整成本
  缺口及 U0/U1 单变量设计的主要诊断；不接受把 EU-CRR 作为唯一下一步或最终模型。
- 新增 G24 和 `exp:duca-eucrr-fusion-diagnostic`，状态为
  `discussed_conditional_not_authorized`。固定四个 contrast：`U1-U0`、`L1-L0`、
  `L0-U0`、`L1-U1`。fusion 失败只 KILL fusion。
- G23/R0--R5、strict pre-backbone 最终合同和 Job `1178989` 均不改变；没有创建 V9、
  没有修改模型代码或实验队列。

## 2026-07-22 DUCA 最终模型单一合同建立

- 新建 `research-wiki/duca_final_model_contract.md`，作为最终交付、结构、训练、推理、
  成本、R0--R5 实验与 GO/KILL 的唯一权威说明。
- 最终产物固定为 fixed-budget offline-TAD pre-backbone acquisition plugin；K384 为主，
  K256 为首个效率扩展。EU-CRR、dynamic MUST、local-cell 与新 detector 均不进入主线。
- 固定结构为 binary coarse action state -> indirect transition center -> bilateral capped
  boundary burst -> overlap saturation/residual context -> existing exact-K/max-hole DP ->
  hard RGB -> official-derived AdaTAD/ActionFormer。
- 固定梯度所有权：action loss 只训练 coarse；endpoint/bilateral/quota/fairness 只训练
  scorer/burst；official stage 冻结 coarse，TAD surrogate 通过 hard-swap alignment 后才
  允许更新 scorer/burst。
- 固定两种成本口径：dense-materialization 复现模式只能证明 heavy-backbone 降帧；
  low-resolution proxy + selected-high-resolution materialization 部署模式才可支撑总成本。
- 修正 Wiki 首页中过时的 `4c777a6/1178911` 主线描述，改为 canonical contract 与
  `63e25eb` 旧目标证据锚点。状态仍为 `designed_not_implemented`，没有创建 V9、
  没有修改模型代码或实验队列。
- 2026-07-22 00:50 +08:00：把 DUCA 从当前 V8 诊断到论文主表的完整计划写入
  `duca_final_model_contract.md`。当前精确作业 `1178989` 仍运行：第一 P0 候选完成但
  selector-versus-delta、endpoint distance 和 radius-one coverage 门槛失败，第二候选训练中。
  冻结 R0--R5 的进入/退出证据、五张论文结果表和 Pro 审查边界；不再进行开放式方向发散，
  只在 R0 参数冻结、R1 exact-commit 长训前和 R4 hard-swap alignment 做定向审查。
- 2026-07-22 02:00 +08:00：完成 canonical boundary-burst 首个实现候选并推送
  `codex/duca-boundary-burst-20260722@4a07a2af72e68f1330467161cbcac2ffba53d367`。
  复用 V8 transition scorer 与 global exact-K/max-hole DP，新增 exact endpoint、双侧、
  配额、fairness 和 saturation 目标；冻结 Gaussian/R2Q3/R4Q5 P0 与 U/G0 四臂。
  本地静态证据为 `24 passed`、三套 P0 validator 通过、compile/shell 通过。
  R0 回放审查发现并修复 evaluator subset 与 blocked-video 真值范围两处会使 mAP 失真的
  契约错误；R0 绝对 training-holdout mAP 明确禁止作泛化主张。独立 MAX agent
  `019f85d3-38b6-7a90-8d20-1d7c8b88fe8e` 与远端 Linux tests 正在运行；尚未部署 GPU DAG。
- 2026-07-22 02:33 +08:00：数值复核发现 P0 候选门禁错误地要求跨样本平均
  `selected_count==384`，会把逐样本正确的短视频/尾窗
  `selected_count=min(384,valid_len)` 误判失败。已在同一分支修复并增加回归测试，
  新精确提交为 `fdf25f5d08bc0bf9b550e059228ce1d6ac587499`，GitHub 已同步；本地新增
  测试 `3 passed`。此前远端 Linux focused `83 passed, 2 skipped`、必要回归
  `23 passed`，提交 DAG 预检通过。两位独立 MAX 正审核新提交；GPU DAG 尚未提交。
- 2026-07-22 02:44 +08:00：独立 MAX 对 `fdf25f5` 给出 `HOLD_FIX_REQUIRED`。
  审核确认 offset head optimizer/gradient、同一 global exact-K/G hard-soft DP、selected-axis
  inverse mapping、推理 no-leak 与 official AdaTAD/ActionFormer 均无阻断；阻断集中于 P0
  逐样本 K/G 循环自证、机制 gate/earliest-pass、endpoint 离散化、R0 非阻断依赖、Q 仅
  作为 loss 参数、crop endpoint validity、R0 Oracle 可行族、U arm artifact 语义和 split
  hashes。正式 GPU DAG 未提交。三组合同修复已交给并行 agent；主线程已开始把 offset
  profile 改为每个预测中心 forward-exact quota support，继续复用现有 global decoder。
- 2026-07-22 03:27 +08:00：上一轮九个 blocker 已在同一 branch 就地整合并推送为
  `899630a5ef4927e78ef4ca6b8cc51fdf754056da`。最终干净 Linux 快照
  `opentad_duca_boundary_899630a_20260722` 通过 `136 passed, 3 skipped` DUCA 回归、
  `23 passed` 强制 C3 回归以及 compile/bash/clean-tree。新独立 MAX
  `019f8614-53e8-79e2-8daa-d52f7be04623` 审计中；CUDA gate 和正式 DAG 仍阻断。
  旧 V8 Job `1178989` 已 `FAILED/2:0`，只作为待封存负诊断，不是主实验结果。
- 2026-07-22 03:31 +08:00：解析 `1178989/frontend_decision.json`，确认三个 P0 LR
  profile 的 12 个 checkpoint 全部完成，但 `eligible_count=0`、`winner=null`、状态为
  `HOLD_FRONTEND_MECHANISM_FAILED`。coarse AUROC 最高 `0.619653`；所有 learned scorer
  transition AUROC 均低于 pure delta；所有 endpoint distance 均差于 uniform；r0 gain 仅
  `[-0.025613,+0.010858]`。旧 V8 因机制门禁主动停止，未产生 official-60 mAP。
- 2026-07-22 04:00 +08:00：独立 MAX 对 boundary-burst exact candidate `899630a`
  给出 `HOLD_FIX_REQUIRED`。核心 burst/DP/selected-axis/no-leak/official detector 未被
  否定；正式 blocker 是四臂 runtime binder/gate mapping、pooled crop validity 与
  R0/P0/aggregate provenance/hash 链。pooled validity 已完成最小修复并通过 focused
  `20 passed`；其余修复继续在同一分支，CUDA gate 与正式 DAG 仍阻断。同步把论文当前
  阶段修正为“V8 负向封存后、R0 运行前的 R1 部署证据合同修复”，并重申只需 exact-commit
  有界复审，不再进行开放式 Pro 发散。
- 2026-07-22 04:05 +08:00：完成上述 blocker 的同分支修复并推送精确提交
  `aa3352ecf803c81d007a62ed5398667d9551684b`。新增真实四臂 runtime binding 回归，关闭
  boundary gate schema/config-stem、pooled crop validity 与 R0/P0/aggregate provenance/
  upstream seal。干净远端快照通过 DUCA `139 passed, 3 skipped`、C3 `23 passed`、
  compile/bash/HEAD/clean。启动全新无上下文独立 MAX
  `019f8647-ad93-70f3-a763-218f7552ac95`；CUDA gate 与正式 DAG 继续阻断。
- 2026-07-22 04:26 +08:00：独立 MAX 对 `aa3352e` 返回 `HOLD_FIX_REQUIRED`。模型主体未被
  否定，剩余 P0 阻断仅为 submit-frozen AdaTAD pretrain path/SHA 的全链消费，以及 real
  legal hard-swap alignment 前 G1/G2 的 production binder 锁死。远端确认旧 V8 Job
  `1178989` 为 `FAILED/2:0`，当前无 boundary-burst CUDA/R0/P0/official-60 作业；论文阶段
  仍为 R0 前 R1 合同修复，不是主实验 running。
- 2026-07-22 04:45 +08:00：最小关闭 aa3352e 的两个剩余 blocker 并推送
  `f629ad79461941f405bc2028f087034abd17a840`。P0/frontend/gate 现在消费提交时冻结的
  AdaTAD pretrain path/SHA；production binder 在 hard-swap alignment 前拒绝 G1/G2。
  干净远端快照通过受影响 DUCA `63 passed`、C3 `23 passed`、pycompile/bash/HEAD/clean。
  启动全新无上下文 MAX `019f866a-6879-75a0-99f4-3c9524ebd076`；没有提前提交 GPU DAG。
- 2026-07-22 05:45 +08:00：`7b9ad0b` 已封闭 selected-axis terminal checkpoint、官方评测、
  prediction 与 aggregate 的证据链，但真实历史 sidecar 复核发现 production
  `build_training_audit()` 未写 `formal_protocol/training_profile`，而 terminal validator
  强制要求这两个字段；旧单元夹具曾手工补键掩盖错误。已在同一分支最小修复并推送
  `86f7663a94d628eace316d17e31db7043f731f75`，测试改为调用真实 builder，aggregate 也独立
  核验协议字段。远端干净快照通过 DUCA `64 passed`、C3/update evidence `29 passed`、
  compile/bash/HEAD/clean。启动全新无上下文 MAX
  `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac`；当前仍为 R0 前 R1 复审，无 CUDA/R0/P0/mAP。
- 2026-07-22 06:21 +08:00：无上下文 MAX `019f86a6-fe1b-7921-b576-8d9cd3d4c8ac`
  对 `86f7663` 返回 `HOLD_FIX_REQUIRED`。模型主体与 official AdaTAD 未被否定；阻断项为
  R0 unrestricted/projected/uniform Oracle 的 official-evaluator bootstrap 与完整身份重算、
  simple delta 进入同一 exact-K/max-hole DP 并执行停止规则，以及 crop-valid、逐作业原子
  提交日志和 no-mock 生产链测试。当前无 CUDA/R0/P0/official-60，论文阶段仍是 R0 前 R1
  合同修复；禁止开放式重设计，只在新 exact commit 后做有界复审。
- 2026-07-22 06:54 +08:00：完成 `86f7663` 独立审计列出的有界修复并推送
  `codex/duca-boundary-burst-20260722@4ec3e078a3aad834ffe504d74d414bf7e2b6fad3`。
  R0 现包含 U/R2Q3/R4Q5/unrestricted exact-K Oracle、逐视频 official-evaluator bootstrap、
  全证据链重开重算与唯一最弱可行族 CI 决策；simple delta 进入同一 global exact-K/G DP，
  并写入 learned selector 的严格 Pareto 停止门禁。同步关闭 crop-valid、no-mock evaluator
  集成测试和逐作业原子 Slurm journal。复核启动脚本时另发现并修正 blocked-video 曾误取
  holdout list 的真实生产错误，现严格取 train block list。纯协议测试 `22 passed`、journal
  `8 passed`、pycompile/bash/diff 通过；Linux/PyTorch/CUDA 与全新独立 MAX 待完成。当前无
  CUDA/R0/P0/official-60 作业，无 headroom/mAP/V9/paper-ready 结论。
- 2026-07-22 07:02 +08:00：远端学术加速克隆精确提交到
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_4ec3e07_20260722`，核验真实 HEAD
  `4ec3e078a3aad834ffe504d74d414bf7e2b6fad3` 与 clean tree。受影响 DUCA 回归
  `109 passed`，强制 C3/official-ASFormer 回归 `23 passed`，pycompile/bash/HEAD/clean 全过。
  状态仅提升为 `linux_tested_pending_independent_max`；未提交 CUDA/R0/P0/official-60。
- 2026-07-22 07:07 +08:00：在同一远端 clean snapshot 运行 `PRECHECK_ONLY=1`，未调用
  `sbatch`。生成的 submission manifest SHA 为 `b068843...a78b4b3`，split manifest SHA 为
  `dc4ca5b...b0493c`；精确绑定 commit、transition-beta0 epoch-131 checkpoint、AdaTAD
  pretrain、annotation/train/holdout split 及 R0→P0→gate→四臂→aggregate 依赖。等待独立
  MAX 裁决，仍未提交 GPU 作业。
- 2026-07-22 07:15 +08:00：计划-代码一致性复核发现新的有界风险：R0 已通过
  `selected_weakest_projected_family` 冻结唯一最弱可行 burst family，但 P0 candidate selector、
  full-model gate 与提交 DAG 仍强制 Gaussian/R2Q3/R4Q5 全部通过并运行四臂。这样诊断臂失败
  会错误阻断 R0 选中的主候选，也与 canonical R3“先跑 matched U/G0”不一致。该问题登记为
  `audit_pending`，等待独立 MAX `019f86e9-8aa0-75e1-8373-686265ac8b61` 裁决；当前不修改模型，
  不提交作业，不创建新版本。
- 2026-07-22 07:30 +08:00：独立 MAX `019f86e9-8aa0-75e1-8373-686265ac8b61` 对
  `4ec3e07` 返回 `HOLD_FIX_REQUIRED`。审计确认模型、R0 Oracle/official evaluator 证据链、
  simple-delta 同可行域、no-leak、selected-axis 与 official-derived AdaTAD 无方法级 P0 blocker；
  唯一 P1 blocker 是 split block list 按 consumer 命名，launcher 却把包含 holdout videos 的
  `train_block_list` 用作 holdout evaluator blocked set。旧记录 06:54 对该变量语义的描述写反，
  本条为权威更正。
- 2026-07-22 07:34 +08:00：在同一分支最小修复并推送精确提交
  `f90595d8620e42e8e3d74722f2ab48126c6b65f2`：R0 改从包含 train videos 的
  `holdout_block_list` 生成 evaluator blocked JSON；翻转错误脚本断言并增加真实 split 语义测试。
  学术加速新克隆通过 affected DUCA `168 passed, 2 skipped`、强制 C3 `23 passed`、compile/bash/
  HEAD/clean 与 no-submit precheck；manifest SHA 为 `14f345dc...7c8a2c`。已启动全新无上下文 MAX
  `019f8701-edaa-7e83-a572-49024b524098`，仍未提交 CUDA/R0/P0/official-60。
- 2026-07-22 07:52 +08:00：第二次独立 MAX 对 `f90595d` 返回分阶段
  `HOLD_FIX_REQUIRED`：R0 明确获准，P0/CUDA downstream unlock/official-60 不获准。审计未发现
  R0 方法、GT 泄漏、可行域、official evaluator、simple-delta、selected-axis 或 AdaTAD 主体错误；
  P1 缺口限定为 P0 records→summary 重算、real-gate 绑定、跨臂 matched identity 与 official
  ASFormer 源码哈希，另有 `gt_boundary_validity` 和原子 aggregate 两项 P2。
- 2026-07-22 07:54 +08:00：严格按阶段许可只提交 R0 Job `1179392`，未提交 P0/gate/
  official-60。run root 为 `duca_boundary_f90595d_r0_formal_20260722_0753`；manifest/r0 sbatch/
  journal SHA 分别为 `d22bd987...dd7b`、`837d4612...ec0f`、`34696160...c6f5`。作业在 `g0006`
  RUNNING，stderr 为空。P0 证据修复并行推进，不修改运行中的精确 R0。
- 2026-07-22 08:22 +08:00：R0 Job `1179392` 在成功导出 40 个 holdout videos、124 个窗口后
  `FAILED`；失败发生在 Oracle 构造、detector evaluation 之前，因此无 mAP。首个失败样本为
  `video_validation_0000206|0`，K384/G2，33 个 segment、65 个有效端点。旧 R4Q5 把每端点
  nearest-Q 固定集合并集作为硬 required positions，物理路径至少需 429 点，超过 K384。
  独立数学复核证明真正 center+within-radius quota+bilateral+K384/G2 存在零 gap witness，故
  这是 Oracle 假不可行，不是否定 G2/R4Q5，也不是 coarse/selector 训练失败。
- 2026-07-22 08:34 +08:00：完成 bounded exact-quota 与 P0 证据链修复，提交并推送
  `codex/duca-boundary-burst-20260722@22555a4e830ce24f9bb516897b1bb7f44b70c188`。
  privileged R0 Oracle 改为联合二进制约束；部署 selector、K/G、loss 和 official AdaTAD 均未改。
  同提交关闭 P0 records→summary 生产重算、real-gate 绑定、official-ASFormer hash、
  `gt_boundary_validity` 传递、跨臂 identity 与原子 self-sealed aggregate。
- 2026-07-22 08:40 +08:00：学术代理干净快照
  `/data/run01/sczc063/yuzibo/projects/opentad_duca_boundary_22555a4_20260722` 通过 solver/Oracle
  `22 passed`、P0 summary `9 passed`、runtime/gate/aggregate `54 passed`、强制 C3 `23 passed`
  与 pycompile/bash/HEAD/clean。旧第 25 条真实失败样本重放后 U/R2Q3/R4Q5/unrestricted 均
  `ok=true`，输出 SHA-256 为
  `168c6f21f869d802e8e3a11fdfcedc2ddc7968fe6fb5b6909776fdb8f84e76ce`。启动独立 MAX
  `019f8743-aed1-7a80-a7d6-552b08491019`；corrected R0、P0 与 official-60 均尚未提交。
- 2026-07-22 11:44 +08:00：将本轮延误根因与强制纠偏规则写入根目录 `RTK.md`，并同步为
  anti-repetition 规则 313-318。性能/mAP 成为执行关键路径；R 系列无冲突实现提前并行，运行依赖
  fail-closed；独立 MAX 只用于三类真正关键版本，小补丁不再触发完整审计；Wiki 只在证据变化时更新。
  显式四小时截止采用 15/30 分钟盘点与提交时间盒。该变更是执行合同，不改变 DUCA 模型、数据协议、
  当前精确提交或运行中实验身份，也不触发新的部署审计。
- 2026-07-22 11:46 +08:00：corrected R0 Job `1179517` 已完成 124-window 输入和四族构造，
  正在执行冻结 official-AdaTAD 的 `R2Q3` replay，stderr/错误扫描为空，尚无 sealed mAP。为消除
  串行排队延误，使用现有原子 journal 提交 P0 Job `1179533`，依赖为 `afterok:1179517`，更新后
  jobs SHA-256 为 `cbd7f59a...48a750`。R0 若按预注册规则 KILL/非零退出，P0 不会启动；未提交
  gate 或 official-60。
- 2026-07-22 12:01 +08:00：用户将四小时验收明确为 R0-R5 全部生产代码、真实后端门禁、正式配置
  和实际 Slurm 部署；Job/依赖/exact commit/run root/manifest/hash/产物必须完整记录。该标准已写入
  `RTK.md` 与 anti-repetition 规则 319-322，并以 interrupt 指令下达到 R1/R3、R4、R5 三个并行
  实现智能体。mock、sentinel、占位 backend、仅 precheck 和未集成的文字报告均不再计为完成。
- 2026-07-22 12:13 +08:00：corrected R0 Job `1179517` 已产生四族冻结 official-AdaTAD
  原始 Avg-mAP：U `93.587070`、R2Q3 `94.190497`、R4Q5 `93.999241`、unrestricted
  `93.970057`。R2Q3 原始 headroom 为 `+0.603427` pp，但 1000 次逐视频 paired bootstrap
  尚未结束，故唯一 family 与 GO/KILL 仍未裁决；Job `1179533` 继续以 `afterok:1179517`
  等待，以上只属于 training-internal holdout reachability，不得写成测试集论文结果。
- 2026-07-22 13:16 +08:00：定位到与模型无关的关键延误：R0 producer 已执行 1000 次逐视频
  bootstrap、每次四族 official evaluator，而旧 P0/family/decision consumer 会多次完整重跑同一
  统计过程。已在 canonical branch 将 consumer 改为重开并哈希核验全部源文件、重算四族原始
  official mAP，并从封存样本重算差值、均值和置信区间，不再重新执行 4000 次 evaluator。
  统计定义、R0 producer 与模型均未改变；目标测试 `11 passed`。旧 P0 Job `1179533` 已在启动前
  hold，等待 corrected consumer 的 Linux 验证和替代作业，避免浪费数小时 GPU。
- 2026-07-22 16:55 +08:00：R0--R5 生产代码已在精确 `e49ef696` 推送并完整部署。有效 DAG 为
  `1179795 -> 1179796 -> 1179797 -> {1179798,1179799} -> 1179825 -> 1179826 -> 1179827 ->
  {1179861..1179864} -> 1179865`，覆盖 24 个 terminal cell 与 9 个成本配置；六个站点上限导致的
  未运行重复项已取消，不代表缺失实验。R0 正常运行、四族点估计已复现、错误扫描为空，下游均按
  `afterok` 等待。
- 2026-07-22 16:55 +08:00：将 R0 串行 1000 x 4 官方 evaluator 的已实测多小时瓶颈最小修为
  `9ed10139317c4196072d471ced883eb1dfc31703` 并推送。相同父进程 RNG 样本与有序输出的串并行
  结果逐项相等；远端 R0 `35 passed`、强制 C3 `23 passed`、compile/bash/HEAD/clean。真实预测
  benchmark Job `1179956` 正在运行；在其通过且旧 R0 未完成前，不取消或重复正式 e49 DAG。
- 2026-07-22 17:20 +08:00：更正 R0 证据口径。40-video split 只对 frontend family selection
  留出；复用的 `transition_beta0/epoch_131.pth` detector 先前按完整 THUMOS `training` subset 训练，
  未排除该 holdout。因此 U/R2Q3/R4Q5/unrestricted 的 93--94 mAP 是 detector-seen 的训练内部回放，
  不可与完整 `validation` 协议下 64--65 mAP 比较，也不可写入论文主表。点估计虽调用 OpenTAD 官方
  mAP 实现，但 40-video split 与 1000-resample paired bootstrap 均为自定义内部诊断；正式论文结果
  必须回到完整 validation/test terminal-checkpoint 评测与多种子统计。R0 状态降级为
  `diagnostic_protocol_contaminated_for_absolute_map`，其相对 headroom 只可用于提出干净复验。
- 2026-07-22 17:48 +08:00：提交执行修正版
  `2bc6ca6fcf34f3e980437b5b830cabeef0de63c0`。新增可执行协议审计：R0/P0/gate/R4/cost
  均不属于论文 mAP，正式行必须使用完整 THUMOS validation、OpenTAD mAP、tIoU 0.3--0.7 与
  terminal epoch-59 EMA。focused 本地/远端 `49 passed`；独立 MAX 在修正 R5“未完成不得预认证”
  后给出 GO。取消旧串行 Jobs `1179795--1179865`，并提交、启动互不依赖的 U/Gaussian/R2Q3/
  R4Q5 Jobs `1180075--1180078`，Slurm 均为 `Dependency=(null)`。GitHub push 被目的地隐私安全
  审查阻止；远端干净快照通过已验证 Git bundle 导入同一精确 commit。
- 2026-07-22 18:20 +08:00：完成全部实验指标口径审计并形成可执行
  `map_protocol_audit.json`。仅 R3 official60 terminal 行满足完整 THUMOS validation、OpenTAD
  `mAP`、tIoU 0.3--0.7 与 epoch-59 EMA，可进入官方对比；R0 的 93--94、P0、一步门禁、R4
  alignment、bootstrap 和成本统计均明确降级为诊断/机制/成本证据。
- 2026-07-22 18:20 +08:00：前三个 uniform 尝试 `1180075/1180097/1180106` 依次暴露生产
  ASFormer provenance 路径、容器身份误判和 CPU/CUDA 值比较三个门禁合同缺陷，全部发生在
  official-60 训练前且无 mAP。最小修复最终封存在
  `8d85929ea04dc40f1eb0c3cc806061ce3b071d3f`，本地/远端 focused 均为 `28 passed`；未改变
  selector、decoder、detector、loss、训练日程或评价协议。
- 2026-07-22 18:20 +08:00：取消所有已知会命中旧门禁的学习臂，提交最终独立队列
  `1180111/1180112/1180113/1180114`（U/Gaussian/R2Q3/R4Q5）。正式根为
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_independent_8d85929_formal_20260722_1820`；
  四个 Slurm 记录均为 `Dependency=(null)`，初始仅因 Priority 排队，不再存在跨实验等待。
- 2026-07-22 18:23 +08:00：Jobs `1180111--1180114` 全部获得 GPU 并同时进入 RUNNING。
  Uniform `1180111` 的真实 AMP/DDP/full-model gate 已返回
  `p1_p2_exact_full_model_amp_ddp_gate_passed`，随后启动 official-60；三个 learned 臂各自在 P0
  epoch 0 产生有限损失并保持 K384。当前 Traceback/OOM/non-finite/ValueError/FAIL 扫描为空；尚无
  terminal epoch-59 EMA mAP，因此状态仅为 `experiment_running`。
- 2026-07-22 18:24 +08:00：逐字归档并逐项核验 e49 R0--R5 Pro 审查。原文存档为
  `docs/methods/reviews/2026-07-22-e49ef696-duca-r0-r5-pro-audit-raw.txt`，共 `53,578`
  字节、`1,404` 行，源/存档 SHA-256 同为
  `1D0F9909D2C3DF3966DED0B9F71BFA0A73F9CA2B8D7C68DF15F64265EC8AD636`；结构化吸收写入
  `docs/methods/2026-07-22-e49ef696-duca-r0-r5-pro-audit-absorption.md`。代码复核确认 H1--H5：
  当前 burst 为 soft bilateral/quota、密集 160x160 decode/H2D 先于选择、R5 未从 raw prediction
  独立重评、cost pair 未 fail-closed、dense source 未固定到封存身份。当前精确 8d 快照相关远端回归
  `96 passed, 1 warning`，表明旧合同稳定而非缺失合同已经实现。项目裁决为
  `SUBSTANTIAL_ACCEPT_CODE_DIAGNOSIS / PARTIAL_ACCEPT_PLAN`；拒绝继续已取消的串行 e49 DAG、让污染 R0
  决定 family 或在当前四臂 terminal mAP 前恢复 24-cell 矩阵。当前 `1180111--1180114` 继续作为唯一
official-mAP 关键路径；无模型代码变更，状态仍为 `experiment_running`。

## 2026-07-22 R5 paired-cost backend correction

`designed` and `implemented`: the single sealed dense AdaTAD receipt is explicitly ActionFormer-only. R5 retains all 24 ActionFormer/TemporalMaxer mAP cells and their raw-prediction OpenTAD evaluator reopen, but formal paired cost is restricted to the four seed-3407 ActionFormer cells. TemporalMaxer is mAP generalization evidence only until it has a sealed same-backend dense receipt. Aggregation now rejects cross-backend pairing and detector-stack fingerprint drift. Focused generator/aggregator tests: 28 passed, 3 skipped; no new experimental result or paper-ready cost claim.

- 2026-07-22 20:57 +08:00：R0--R5 最终候选已整合为 `codex/duca-boundary-burst-20260722@cd68d89dcc0854baa3c0107607086e801509b552`。首次 `2645e68` 全量投递在真实 soft-bilateral P0 首 batch 暴露 PyTorch 2.0 CUDA 二维行切片标量赋值错误（`5920 vs 1184`），无 mAP；已用等价 `index_fill_` 修复并取消 `1180326--1180331`。修复版无依赖 bundles 为 `1180336--1180340`，aggregate `1180341` 仅依赖 `1180340`；run root 为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_cd68d89_parallel_20260722_205506`。当前均为 `PENDING(Priority)`，状态 `experiment_running`，尚无 terminal mAP 或论文 claim。
- 2026-07-22 21:04 +08:00：`cd68d89` 五个独立模型 bundle 已全部并行进入 `RUNNING`：`1180336/g0043`、`1180337/g0006`、`1180338/g0045`、`1180339/g0005`、`1180340/g0067`；R5 aggregate `1180341` 仅为合法 dependency pending。限定到 `.out/.err/train.out` 的扫描未发现 Traceback、OOM、non-finite loss、FAIL 或旧 `5920 vs 1184`。R2/R3 core exact-uniform official-60 与 adapted R4Q5-G0 P0 已进入 epoch 0；尚无成功 optimizer-step 记录、terminal artifact、mAP 或成本结果，状态仍严格为 `experiment_running`。
- 2026-07-22 21:08 +08:00：建立并补全专用账本 `experiments/duca-r0-r5-cd68d89-parallel.md`，逐项登记 R0 四族、R2/R3 六臂、R4 R2Q3 G1/G2、R5 真实第二后端门禁、24 个 backend/policy/budget/seed mAP 单元、四个同后端成对成本 profile、aggregate、精确提交/快照/Job/产物路径及验收标准。`anti_repetition.md` 同步声明 cd68 取代旧 8d 队列规则，并禁止复用 `2645e68` 启动故障；避免后续把诊断 mAP、旧作业或工程失败混入论文结果。
- 2026-07-22 21:10 +08:00：确认 heartbeat automation `duca-21-00-full-progress-report` 已更新为 `DUCA R0-R5 cd68d89 progress`，状态 `ACTIVE`、每小时一次。其提示词固定读取 cd68 专页/反重复规则/模型合同/版本表/日志，并绑定 Jobs `1180336--1180341`；新作业状态、失败、terminal mAP、成本或 claim 裁决必须同轮回写 Wiki，无变化不重复制造长报告。
- 2026-07-22 21:45 +08:00：完成 DUCA 五点预算扩展。代码 `a00498e15d69294f78d0abeadfb47bc456db0b0e` 支持 K384/320/256/192/128 与动态矩阵轴；远端门禁 `71 passed`。旧 K384/K256 不重训，新 K320/K192/K128 共 36 个 ActionFormer/TemporalMaxer x uniform/learned x 3 seeds 完整 TAD cells 已提交为 Job `1180356`，聚合 `1180357`，五点官方 mAP 与选帧分布 `1180358`。增量 root 为 `duca_budget_a00498e_extension_20260722_2145`，jobs ledger SHA-256 为 `d1c32352940ff2c47926a9c95e6e571924ff0623069604d185b7c8e7e52f7bf0`。当前均无 terminal mAP，状态仅为 `experiment_running`。
- 2026-07-22 21:55 +08:00：将现行 DUCA Pro 审查 Prompt 更新并固定到 GitHub 精确提交 `a00498e15d69294f78d0abeadfb47bc456db0b0e`。新版 `duca_r0_r5_pro_review_prompt_20260722.md` 强制重建 merged runtime 模型图、selected-axis/true-time 语义、loss/梯度所有权、Oracle 式边界微簇、五点预算与两个真实 backend，并要求在当前 official mAP 结果出来后只选择一个主要结构修正；纯启动器或审计工程不得挤占性能分析。

- 2026-07-22 22:20 +08:00：完成当前 `a00498e` 与官方
  `sming256/OpenTAD@1aa8ca4` 的 AdaTAD 源码/配置/前向语义对照。官方 base config、
  `ActionFormerHead`、projection、focal/IoU loss 和 NMS 源码保持一致；完整
  `ActionFormer` wrapper 、VideoMAE 输入时序与训练协议不是源码同一。DUCA 将
  768 帧窗口选成 384 个观测，把 GT 映射到 selected axis，再将预测逆映射回
  true time 后执行未改的 NMS。当前口径固定为“官方派生 AdaTAD/ActionFormer
  backend，head/loss/NMS 不变”，禁止声称“完整官方 AdaTAD 源码不变”。当前
  config/full-model focused tests `24 passed`；非均匀 selected axis 没有显式 true-time
  interval encoding 仍是待由 mAP 裁决的方法风险。

- 2026-07-22 22:13 +08:00：远端正式训练健康推进。Jobs `1180336--1180340`
  仍并行 `RUNNING`，`1180341` 合法 dependency pending；R2/R3 core 的 exact-uniform
  official-60 与 adapted bundle 的 R4Q5-G0 P0 均进入 epoch 9，日志损失有限并已有连续
  optimizer updates。isolated AMP replay 未耗尽，完整 scoped scan无 Traceback、OOM、
  non-finite loss、ValueError 或 FAIL。当前仍无 terminal epoch-59 EMA mAP。
  新三档 Job `1180356` 继续 `PENDING (AssocGrpGRES)`，Slurm 当时估计开始时间为
  `2026-07-23 21:55:29 +08:00`；`1180357/1180358` 依赖等待，禁止重复提交。
- 2026-07-22：冻结“模型算法优先”仓库规则。研究目标是性能更好、创新性更强的模型算法，而非复杂工程平台；默认先做最小端到端模型、matched baseline、官方 mAP/高 tIoU、真实成本和机制消融。仅模型行为、数据/GT 泄漏、指标/成本真实性及可运行性问题可以阻塞实验。当前空间路线直接优先验证由真实 AdaTAD 检测损失学习的连续可变区域策略；手工或 GT 特权裁剪仅保留为诊断，不再作为进入 learned ROI 的前置裁决。
- 2026-07-22：完成 Uni-AdaFocus 与当前 DUCA 粗扫描粒度复核。Uni-AdaFocus 官方流程是从全视频形成 48 个均匀候选、16 个轻量 glance、策略插值后再选 16 个重 focus，并非 native-FPS 逐帧重推理。当前 DUCA 的 768 点 probe 约每 4 个源帧取一点且批量运行，但在选择前仍解码/变换/搬运全部低分辨率候选，因此目前仅能证明重 backbone 处理帧数下降。登记 V8/R 终局后的有界消融：密集候选粗扫、低频粗扫、低频加状态转变峰值局部补扫，在固定 K/selector/detector 下以官方 mAP 和完整端到端成本裁决。
- 2026-07-22 23:15 +08:00：远端复核确认 `1180336--1180340` 仍为 RUNNING、`1180341` 依赖等待；exact-uniform official-60 与 adapted R4Q5-G0 P0 均到 epoch 17，损失有限，已有 epoch 4/9/14 检查点。五个顶层 stderr 均为 0 行，未发现 Traceback、OOM、non-finite loss、ValueError 或 FAIL。新发现的执行瓶颈是 R0/R1 bootstrap 仅到 400/1000，R4 与 R5 bundle 各自仍在重复 R0 bootstrap 且仅到 200/1000，尚未进入对应后续模型阶段；这不是方法失败，也不能把占用 GPU 误报成 R4/R5 已训练。新三档 `1180356` 仍为 `PENDING (AssocGrpGRES)`，Slurm 估计开始时间更新为 `2026-07-23 22:47:00 +08:00`；当前仍无 terminal official mAP。
- 2026-07-22 23:38 +08:00：确认旧并行入口同时存在重复 trainable bootstrap 与 Slurm step 继承全部 GPU 的执行错误。提交并推送 `9f97f2c7f081b10fbf1f63d0602a621c6b43a780`，将 R0/P0/U/G0/alignment 收敛为一次共享前置，并以 `srun --exact --gpus=1 --gpus-per-task=1` 约束每个子臂。Local focused `22 passed, 4 skipped`；远端 clean snapshot、Linux focused 与 bash 语法均通过。取消旧 `1180336--1180341,1180356--1180358`，其日志/checkpoint 保留但无 terminal mAP。新正式根为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_boundary_9f97f2c_formal_20260722_2343`，Jobs `1180490--1180493` 已 RUNNING，`1180494/1180495/1180496` 按共享证据依赖排队。真实 `squeue --steps` 显示 R2/R3 四臂同时运行，四个 step 均为 `TresPerStep=gres/gpu:1`。状态仍为 `experiment_running`。
- 2026-07-22 23:45 +08:00：登记 `experiments/duca-sparse-probe-and-coarse-backend-ablation.md` 为 `designed`。粗扫轴固定源帧间隔 4/8/12/16，hidden-linear 重建必须携带真实 anchor mask/距离；粗分类器轴只使用官方 MS-TCN2、ASFormer、Video-Mamba-ASFormer 和 FACT。边界分布轴固定比较 uniform、动作内部、单转变峰、R2Q3、R4Q5 与边界预算比例。所有结论以完整 official terminal mAP 和真实总成本为主，粗分类/选帧质量只做解释；尚未实现或提交这些后续消融。
- 2026-07-22 23:46 +08:00：新正式 Job `1180490` 已 `COMPLETED/0:0`，focused `96 passed in 26.93s`。`1180491/1180492` 四个并行 P0 子臂均已进入 epoch 0，有限损失、K384、约 3719 MB，孤立 AMP replay 仅 1/8；`1180493` 执行唯一共享 frontend/R0。四个顶层 stderr 均为 0 行，仍无 terminal mAP。
- 2026-07-22 23:50 +08:00：复核 `9f97f2c` 新队列：`1180491/1180492` 四个 P0 子臂继续真实并行，三个到 epoch 0 batch 40、一个到 batch 20，损失有限且每臂至多一次 AMP replay；错误扫描无 Traceback/OOM/non-finite loss/ValueError/FAIL。`1180494/1180495` 只等待共享 U/G0 与 hard-swap alignment，不再重复 R0；仍无 terminal official mAP。
- 2026-07-23 00:16 +08:00：提交并推送 `4f81299`，新增极简四官方粗分类后端实验入口；Linux focused `11 passed in 8.23s`。远端官方源码固定为 MS-TCN2 `f423a9e`、ASFormer `e1bbe4f`、FACT `7bd81bd`、Video-Mamba suite `ec9108b`。独立无依赖 Jobs `1180502/1180503/1180504/1180505` 已分别在四节点 RUNNING，四者真实 CUDA 小前向均通过，当前无 Traceback/OOM/non-finite/FAIL。Wiki 同轮修正稀疏粗扫合同：插值 temporal hidden 直接作为证据，不再提供 anchor mask/距离；P0 结果仅为粗分类诊断，完整 TAD 前仍须统一四后端 temporal-hidden 接口。
- 2026-07-23 00:25 +08:00：四个官方粗分类 P0 作业均进入 epoch 4，累计训练 loss 约为 ASFormer/FACT/Video-Mamba `0.61`、MS-TCN2 `0.62`，错误扫描为空。该现象只证明四个官方适配均可稳定训练；终轮 actionness/边界指标和完整 TAD mAP 均尚未产生，禁止提前排名。
- 2026-07-23 00:40 +08:00：完整归档并复核 `a00498e` selected-axis/TTDI Pro 回复。原文
  `76,688` bytes、`1,705` 行，源/归档 SHA-256 同为
  `36523b2f1a7456f8d4a4314ea445971f8066eec59611f9632d7bc1d33e31a884`。远端确认
  `a00498e -> 9f97f2c` 只改两个 bundle 启动器和测试，`9f97f2c -> 4f81299` 只新增
  粗分类后端 P0 启动入口，因此 selected-rank 时间扭曲、mandatory union 无接纳前
  completeability、local-RGB-slope surrogate 和五预算成本 parser 缺口仍存在。项目裁决为
  `SUBSTANTIAL_ACCEPT_MODEL_DIAGNOSIS / PARTIAL_ACCEPT_TTDI_REMEDY`：TTDI 只作为 terminal
  mAP 后的候选，先单独验证 zero-init true-time feature residual，physical-coordinate head
  作为第二变量；不取消或修改当前 Jobs `1180490--1180496`。
- 2026-07-23 01:10 +08:00：四种官方粗分类后端 P0 Jobs `1180502--1180505` 全部 `COMPLETED/0:0`。终轮 Action AP 分别为 MS-TCN2 `0.4078`、ASFormer `0.4087`、FACT `0.3945`、Video-Mamba-ASFormer `0.4161`；四者最佳间接边界策略均为 `delta_p_action`，boundary support@1 分别为 `0.7225/0.8184/0.7956/0.8302`。这些是无 detector 的诊断，不是 TAD mAP。
- 2026-07-23 01:16 +08:00：实现并推送 `codex/duca-sparse-probe-interpolation-20260723@dd3c97cf5ee628c2b0b6f26ce976618e36b7cd45`。只在 d=1/2/3/4 anchor 计算低分辨率空间 stem 与官方 ASFormer，把 action logits、temporal encoder hidden 和 policy hidden 线性重建到完整网格，不提供 anchor mask/距离。新 focused `4 passed`；Gate Job `1180556 COMPLETED/0:0`，真实 CUDA 下四档梯度非零且估算 MACs 单调下降。四卡 Suite Job `1180557` 已 RUNNING，四个独立一 GPU step 均进入 P0 epoch 0；root 为 `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_sparse_probe_dd3c97c_20260723_011329`，尚无 terminal mAP。
- 2026-07-23 01:17 +08:00：首次 suite 预检查在生成回执时调用登录环境旧 Python，因无 `pathlib` 在 suite 提交前退出；Gate 已正常完成。随后只提交唯一 Suite `1180557`，并用 shell 补写 jobs.tsv、deployment receipt 与哈希，没有重复实验。当前日志无 Traceback/OOM/non-finite/FAIL。
- 2026-07-23 01:24 +08:00：稀疏四臂均到 P0 epoch 1 batch 20，loss `3.7978/3.6695/3.3860/3.5707`、K384，显存随 d=1/2/3/4 为 `3719/2108/1568/1315 MB`。d=1 两次 AMP replay 后已成功有限更新，其他臂直接有限；无 Traceback/OOM/non-finite/FAIL。当前 detector path 在 P0 阶段按设计 skipped，端到端梯度与 mAP 仍待各臂 full-model gate/official-60。
- 2026-07-23：复核 TTDI 与五项开放问题的性能优先级。最高风险是 learned 非均匀帧在 selected-axis detector 中被当作等间隔；mandatory bool union/无接纳前 completeability 与 local-RGB-slope detector bridge 分别是 selector 和 G1/G2 的中高风险。五预算 cost parser、G2 paired-counterfactual 表述及 official-derived 命名主要影响论文证据。冻结触发式顺序：selection 未优先修 scorer/decoder；selection 已优但 mAP 未升才测 T1；G1/G2 低于 G0 则修/关 bridge；T1 后仍有尺度错误才做 T2。
- 2026-07-23：新增免目标域训练候选 `ideas/duca-target-train-free-transition-prior.md`，状态仅为
  `designed_only`。冻结外部预训练 encoder + 无参数转变证据 + 现有 R2Q3 exact-K/G 被定义为严格
  无目标训练/无测试时梯度模式；T3AL 式测试时适配、目标 mAP 调 prompt/权重以及密集高成本视频
  prior 被排除。同步补全 PhysTime T1/T2 分层：v1 负结果是多变量失败，下一步仅允许相同硬选帧和
  selected-axis detector 下的零初始化 true-time residual 三臂。五类论文图按实际证据映射，运行中
  suite 与 coarse-only P0 不预填 TAD mAP 或 Pareto。无模型代码、作业或论文 claim 状态变化。
- 2026-07-23 11:16 +08:00：T1 与 target-train-free 前端升级为
  `implemented/tested/experiment_running`。精确分支
  `codex/duca-t1-trainfree-20260723@4c5604b4a0abde9e59f625d519934e855bfe1519`
  修复 train-free 配置误走 132-epoch 合同的问题；远端 `29 passed in 43.19s`，pycompile、bash
  syntax、clean-tree 通过。Fast-only 真实 CUDA preflight 明确 Slow/lateral 均未执行，Job
  `1180653` 已完成 epoch 0 并进入 epoch 1；batch 99 有限损失 `1.4430`，K384。MobileNet
  官方权重已按 SHA-256 `047dcff4...f4e1f`
  放入共享缓存，最终三臂 Job `1180654` 已无依赖提交，当前等待账户 GPU 配额。旧
  `1180639/1180644/1180652` 均为 optimizer 前下载/合同基础设施退出，不是 mAP 结果。新增总账
  `experiments/duca-t1-and-target-trainfree-official60.md`；当前仍无 terminal mAP/成本结论。
- 2026-07-23 03:42 +08:00：自动巡检发现 R2/R3 Jobs `1180491/1180492` 失败。四个 P0
  均已完成 20 epoch 和每臂 6000 次有效更新；失败只发生在后置整模门禁，根因是 checkpoint
  未保存六个 BatchNorm 非参数 buffer，而旧初始化器把它们误判为学习参数缺失，无 official mAP。
  最小修复 `codex/duca-boundary-burst-gatefix-20260723@487a1784554b8c07cbaf8e3948c5aea785a2d8e1`
  保持可训练参数严格加载，只显式保留并记录缺失 buffer 的新阶段初值；Linux focused
  `64 passed`。四个封存 P0 的恢复 Jobs `1180671--1180674` 已独立提交，前两臂已越过旧错误点，
  后两臂等待 GRES；manifest SHA-256 为
  `ca3e42e1bc624faaa592f632a77c8452143b444a19e1ec03486cfa9bf288cc25`。
- 2026-07-23 03:58 +08:00：继续自动根因收敛。`1180671--1180673` 已通过修复后的 BatchNorm
  buffer 门禁，但在 optimizer 前暴露“实验标签被误作 runtime variant”的第二个合同错误；修复并推送
  `codex/duca-boundary-burst-gatefix-20260723@ca40c9c5a097e8ab083ba3ffd2ff7f5709841010`，
  远端 focused `84 passed in 69.38s`。首次 ca40 提交 `1180682--1180684` 又因临时提交器预建
  `ARM_ROOT` 被 fail-closed 拒绝，未进入模型；去除预建后，唯一有效重试为 `1180685--1180687`，
  root `duca_boundary_ca40c9c_retry_20260723_035756`，manifest SHA-256
  `f6c053a9452c26eefc427f375318fcec03689041b8829fbe1d6fc11d9e88268f`，当前
  `PENDING(Priority)`。R4Q5 `1180674` 已完成 official-60 epoch 0 的 99 batch、损失有限并进入
  epoch 1；当前仍无 terminal mAP，状态保持 `experiment_running`。
- 2026-07-23 04:21 +08:00：全矩阵巡检发现 sparse Suite `1180557` 四个 P0 均完成后命中同一
  BatchNorm buffer 门禁，及 MobileNet `1180654` 在 optimizer 前因冻结未初始化 `LazyLinear`
  失败。稀疏修复 `cee4ccd33fb20e11978e4a2a6eaa3f5845b51489` 只复用封存 P0，远端
  `10 passed`，恢复 Job `1180696`；MobileNet 修复
  `e30db0f3987128798da6bc8ff446065b818b1a7f` 同时纠正“随机 1 维头破坏预训练多类语义”的
  模型错误，远端 `8 passed`，三臂 Job `1180697`。两者当前均为 `PENDING(AssocGrpGRES)`。
  ca40 R2Q3 Jobs `1180685--1180687` 已到 epoch 2，R4Q5 `1180674` 到 epoch 4，损失有限；仍无
  terminal mAP。
- 2026-07-23 04:25 +08:00：30 分钟自动巡检合同已启用并完成首次全矩阵复核。有效 R2Q3
  恢复 Jobs `1180685--1180687` 均进入 official-60 epoch 3，R4Q5 `1180674` 进入 epoch 5；
  T1 core/controls `1180637/1180638` 进入 P0/official 阶段约 epoch 15，Fast-only `1180653`
  进入 official-60 epoch 13。上述活动日志未发现新的 Traceback、OOM、non-finite collapse、
  ValueError 或 FAIL。稀疏粗扫恢复 `1180696` 与修复后的 MobileNet 三臂 `1180697` 仍为
  `PENDING(AssocGrpGRES)`，只是账户 GPU 配额等待，禁止重复提交。扫描到的异常均来自已经封存并
  修复的旧 Jobs `1180491/1180492/1180557/1180654/1180671--1180673`，不登记为新故障。
  当前所有主臂仍无 terminal epoch-59 EMA OpenTAD mAP，状态保持 `experiment_running`。
- 2026-07-23 05:05 +08:00：定时巡检发现 T1 三个学习臂完成 P0 后命中已知 BatchNorm buffer
  门禁；`1180638` 顶层失败，`1180637` 因健康 exact-uniform 子臂仍运行。三个 P0 checkpoint
  均已哈希封存且没有 official-60 更新。修复并推送
  `codex/duca-t1-gatefix-20260723@26ce86d7810e8f7c0568dc045bb1db7240c66de2`，远端
  `15 passed`；恢复 Jobs `1180717/1180718/1180719` 已独立提交，其中前两项已获 GPU，第三项
  等待 GRES。MobileNet 修复 Job `1180697` 三臂已越过原 LazyLinear 故障并进入 epoch 1。
  R2Q3 恢复臂到 epoch 7、R4Q5 到 epoch 9，Fast-only 到 epoch 20；稀疏恢复 `1180696` 继续
  等待四卡配额。所有实验仍无 terminal official mAP。
- 2026-07-23 05:07 +08:00：T1 恢复 `1180717` 已通过原 BatchNorm buffer 失败位置并进入
  official-60 epoch 0；`1180718` 执行门禁，`1180719` 等待 GRES。新恢复根暂无异常。
- 2026-07-23 05:35 +08:00：巡检发现 `1180718/1180719` 通过整模门禁后，在 official-60
  optimizer 前因 T1 实验标签未登记到 selected-axis runtime config 映射而 fail-closed；没有
  mAP，不能作为性能负证据。最小修复并推送
  `codex/duca-t1-gatefix-20260723@919aa555d1aa36191ee318477409dfbfdfb0e807`，远端干净快照
  `16 passed in 49.55s`。只恢复受影响的 actual/reversed 两臂为 `1180731/1180732`，root
  `duca_t1_919aa55_recovery_20260723_053508`；manifest/jobs SHA-256 为
  `8fc94b4eb02994c42d137fd65ee6286b574a875fd488e7d7cb4a152c0432df26` /
  `623b575ad7e8f93c45692c45f4e7e21264753048edb0e98e9c450680d6389dd2`。健康 R2Q3
  `1180717` 不重跑，`1180718/1180719` 归档为零更新合同失败。05:37 复核确认 `1180731`
  已越过失败点并进入 official-60 epoch 0，`1180732` 等待账户 GRES。
- 2026-07-23 05:58 +08:00：全矩阵无新错误或 terminal mAP。T1 actual `1180731` 已在
  `919aa55` 上形成有限 official-60 更新并到 epoch 2；R2Q3 `1180717` 到 epoch 6，R4Q5
  `1180674` 到 epoch 17，soft/hard/adapted `1180685--1180687` 到 epoch 15，MobileNet 三臂约
  epoch 8，exact-uniform 与 Fast-only 到 epoch 30。共享 `1180493` bootstrap 到 `800/1000`
  且 05:46 仍写入，不是卡死；`1180732/1180696` 继续等待账户 GRES。
- 2026-07-23 06:54 +08:00：共享 R0 `1180493` 完成 1000 次 paired bootstrap 后按预注册规则
  以 `KILL_PROJECTED_FEASIBLE_SET`/`2:0` 终止，不是工程异常。内部 holdout replay 的
  uniform/R2Q3/R4Q5/Oracle Avg-mAP 为 `93.5871/94.1905/93.9992/93.9701`；三项 headroom
  CI 下界均为负，未通过 `+0.20 pp` 门槛，因此 selected family 为空，`1180494/1180495`
  进入 `DependencyNeverSatisfied`。禁止重跑或事后强选；90+ 指标不是 official validation mAP。
  同时 T1 reversed `1180732` 已获得 GPU，其他正式训练臂继续运行，仍无 terminal mAP。
- 2026-07-23 07:27 +08:00：T1 reversed `1180732@919aa55` 已通过真实整模门禁并到
  official-60 epoch 3 batch 50，有限损失 `0.8838`；启动时一次 AMP replay `1/8` 已恢复，
  未出现 replay 耗尽或数值崩溃。T1 actual/R2Q3 分别到 epoch 14/18，exact-uniform 与
  Fast-only 到 epoch 44/46，MobileNet 三臂约 epoch 20--21。sparse recovery `1180696`
  仍等待账户 GRES；所有活动臂尚无 terminal mAP。
- 2026-07-23 09:02 +08:00：Fast-only `1180653@4c5604b` 已 `COMPLETED/0:0`，完成
  6000 次 optimizer/scheduler/EMA 更新并以 `epoch_59.pth/state_dict_ema` 完整评估 211 个
  validation videos。官方 Avg-mAP `63.5297%`，tIoU 0.3--0.7 为
  `79.9106/74.5241/66.3665/54.7535/42.0937%`。结果未达到约 65% 目标；matched exact-uniform
  尚未结束，暂不计算精确差值。当前优先诊断 Fast motion evidence、R2Q3 边界位置和 selected-axis
  时间扭曲，不重跑相同 Fast-only 配置。
- 2026-07-23 09:11 +08:00：matched exact-uniform `1180637` 已正常完成 60 epoch 与
  `6000/6000` 次训练更新，随后严格加载 `epoch_59.pth/state_dict_ema` 开始 211-video 官方
  full-validation；当前约 `134/396` batches。尚无终点 mAP，不使用历史基线代替本次 paired control。
- 2026-07-23 09:18 +08:00：复算 R0 的 1000 次配对 bootstrap。R2Q3 差值为正/超过
  `+0.2 pp` 的比例为 `85.2%/72.3%`，中位数 `+0.4553 pp`，但 2.5% 分位为
  `-0.5300 pp`；R4Q5 对应为 `73.3%/59.8%` 与 `-1.3154 pp`。因此 exit 2 代表小样本冻结
  internal replay 无法稳定选 family，不是平均负收益，也不能替代独立 official-60 终点裁决。
- 2026-07-23 09:26 +08:00：matched exact-uniform `1180637` 已完成 `epoch_59.pth/state_dict_ema`
  完整 THUMOS validation，官方 Avg-mAP `64.49%`，tIoU 0.3--0.7 为
  `79.59/75.42/67.71/57.27/42.45%`。Fast-only `63.5297%` 的 matched delta 为
  `-0.9603 pp`，且在 tIoU 0.4--0.7 全部退化；其定位为 train-free 强先验负结果，
  不升级为主方法。
- 2026-07-23 09:28 +08:00：稀疏粗扫恢复 `1180696@cee4ccd` 在四个已封存 P0
  臂的 official 首更新前，因 `sparse_probe_hidden_linear_d1...d4` 未登记到
  selected-axis runtime 映射而失败。这是零 official-update 配置合同失败，无 mAP，
  不作为稀疏插值方法负证据。
- 2026-07-23 09:38 +08:00：确认 R2Q3/R4Q5/soft-detached/hard-detached/soft-adapted
  五个独立 official-60 臂全部 `RUNNING`，约为 epoch `34/46/45/42/44`，无新数值或
  代码错误。这些臂不受 R0 科学停止结论阻断，必须继续到 epoch-59 EMA 完整
  validation，再以官方 Avg-mAP 及 tIoU 0.3--0.7 相对 matched uniform `64.49%` 裁决。
- 2026-07-23 09:44 +08:00：复核旧日志和配置：`1150842` 的终点 `65.69%` 使用
  physical-grid/grid-aware ActionFormer，不是标准 AdaTAD 均匀基线；可比较的 native
  stride-2 selected-axis `1150701` 终点 `64.31%`，与当前 `64.49%` 一致。五个 R
  臂均未运行中间 validation，当前无中间 mAP；最新损失只能证明训练健康。
- 2026-07-23 11:42 +08:00：R 系列终点开始收割。R4Q5 `1180674`、
  soft-detached `1180685` 与 soft-adapted `1180687` 已完成 epoch 59 的
  `6000/6000` 次训练更新并进入终端 EMA 完整验证；hard-detached `1180686`
  到 epoch 58。日志损失有限，未出现新的 Traceback、OOM 或 non-finite collapse，
  当前仍未输出 official validation mAP，禁止据训练损失提前排序。T1 R2Q3
  `1180717` 到 epoch 50、true-time residual `1180731` 到 epoch 48、reversed
  control `1180732` 到 epoch 36；MobileNet 三臂继续运行。连续密度新候选的
  `1181031` 只在 optimizer 前暴露旧 full-model gate 错用硬时序合同 validator；
  独立 MAX 同时发现投影前连续位置与投影后硬位置的梯度锚点错位，正式长训保持
  HOLD，修复方向固定为实际硬帧局部斜率桥，不把该门禁退出当作性能证据。
## 2026-07-23 13:00 - Budget-calibrated sampling-rate unified candidate deployed to gates

`codex/duca-density-transport-20260723@685ebe106302e20bed9e933fa6a01945b0b72cc4`
implements a single rate-based pre-backbone acquisition model with exact-K
calibration, contribution distillation, and nested ASFormer adaptation
variants. The `asformer_full_encoder` variant permits detector/utility
adaptation across all ASFormer encoder layers while the coarse action BCE
continues to train the entire coarse classifier. First wrapper `1181228` had a
zero-update `/bin/sh` launch failure; corrected focused gate `1181234` and
full-ASFormer real-model gate `1181235` are the active evidence. See
`experiments/duca-budget-calibrated-sampling-rate.md`. No mAP claim is allowed.

## 2026-07-23 13:55 - Full-ASFormer rate gate repair and focused pass

Focused CUDA Job `1181249` on
`268c357a749e6e6278ee57676e782a82ec7b7b81` passed all 10 targeted tests. The
two prior gate-only failures were resolved before a valid optimizer update:
small nonzero contribution-head initialization preserves first-step encoder
gradients without changing the zero-initial policy fusion, and the exact-K
sampling-rate bridge now legally supports uniform teacher rows. Job `1181250`
is the active real AdaTAD full-ASFormer gradient-ownership gate. No mAP claim.

## 2026-07-23 14:10 - Rate full-ASFormer gate reached model path, then exposed one missing import

The first two real-gate retries (`1181250`, `1181254`) made zero official
updates because their manual wrappers did not source the canonical THUMOS
environment. The canonical-environment retry `1181256` reached real data,
VideoMAE, AdaTAD and the contribution-distillation loss, then failed before an
optimizer update with `NameError: F is not defined`. Exact commit
`e8a2fea3a6034ae51960aca230ec0fb6efd1aff` adds the missing functional import;
the sampling policy, targets and loss formula are unchanged. This is a bounded
implementation repair, not a model-performance result. The replacement gate
must use the canonical environment and this exact commit.

## 2026-07-23 14:18 - Exact functional-import gate chain queued

Remote snapshot `opentad_duca_rate_685ebe1_20260723` is detached at exact
commit `e8a2fea3a6034ae51960aca230ec0fb6efd1aff3`. Focused GPU Job `1181288`
and dependent full-ASFormer real AdaTAD gate `1181289` were submitted under
`duca_rate_e8a2fea_gate_20260723_1418`. The site rejects explicit memory
requests even below its nominal single-GPU default, so the resubmission uses
Slurm's default allocation; this changes no model setting. At submission the
focused job waits on priority and the gate waits on `afterok`. No mAP claim.

## 2026-07-23 14:32 - Gate wrapper ordering corrected

The first e8a2fea resubmission `1181288` exited before Python because `set -u`
preceded the site `/etc/profile`, whose own script reads an unset
`XDG_DATA_DIRS`; dependent `1181289` therefore never started. This was another
zero-update wrapper error. The corrected two-job chain is `1181295` (focused)
and `1181296` (dependent full-ASFormer real-model gate), with the profile
loaded before strict unset-variable checking. The code commit and all model
arguments are unchanged.

## 2026-07-23 14:34 - Full gate resubmitted with exact 40-character commit

The source checkout and GitHub commit are
`e8a2fea3a6034ae51960aca230ec0fb6efd1aff3`; earlier gate arguments omitted its
last character, so Job `1181296` fail-closed before a model update. Focused Job
`1181295` already passed `7` tests and is reused. Job `1181304` is the only
admissible replacement full-ASFormer gradient gate, bound to the full commit.
# 2026-07-23 Train-only selection / gradient attribution visualization implemented
- Committed and pushed at `codex/duca-density-transport-20260723@5357ff1b06d661ebfa276b91d60fac769b35d4f9`: `tools/bata/export_duca_training_attribution.py`, `tools/bata/plot_duca_training_attribution.py`, and a focused structural test.
- The exporter replays one deterministic real THUMOS training batch from an existing official60 checkpoint without an optimizer step. It records coarse evidence, sampling rate/density, hard selected positions, train-only detector cls/reg `|input * gradient|`, contribution logits/distributions, and `|d detector loss / d sampling-rate logit|`.
- GT is explicitly labelled as training-loss / visualization-only context and is never an inference-time decision input. The plotting tool renders one aligned four-lane PNG/PDF per checkpoint for a fixed video window, suitable for epoch 0/10/.../59 comparison.
- Evidence state: `implemented/tested` for syntax, schema, and synthetic PNG/PDF rendering (`4 passed`); the real-CUDA gate is `1181409` under `duca_rate_5357ff1_gate_20260723_1347`. No selector, decoder, training algorithm, SparseHead, spatial crop, or ChronoTransport code was altered.

# 2026-07-23 Fixed-small-sample visualization gate wrapper repair
- Job `1181409` failed before Python because Slurm's non-interactive shell had no `module` command. It is a zero-update environment wrapper failure, not a model or visualization result.
- Job `1181418` replaces it with `/etc/profile` loaded before strict shell mode. It is bound to `5357ff1b06d661ebfa276b91d60fac769b35d4f9` and runs focused tests plus the real full-ASFormer/AdaTAD gradient gate. Only after that gate can an actual checkpoint produce the fixed `4 train + 4 validation` visualization evidence.

# 2026-07-23 Sampling-rate matrix now evaluates every five epochs
- Commit `aa439fb2e2f3d29e3225c73b49f733ad93a40906` adds a strict
  `sampling_rate_exact_uniform` control and changes the sampling-rate
  official-60 workflow to validate at epochs 5, 10, ..., 60 while retaining
  terminal `epoch_59` EMA as the only primary checkpoint.
- Job `1181418` is classified as a zero-update configuration-contract failure:
  it used the old 132-epoch config against the official-60 validator. It is not
  model evidence.
- Initial no-dependency submissions `1181482--1181494` are all zero-update
  Slurm-wrapper failures (shell source, nested variant quoting, then a fresh
  `ARM_ROOT` precondition). The first valid concurrent submissions are
  `1181495` uniform, `1181496` rate-only, `1181497` rate plus cls/reg
  contribution distillation, and `1181498` the full-ASFormer dual-supervision
  arm. They all use the same `aa439fb` source snapshot and canonical VideoMAE
  hash. Their intermediate mAP values are diagnostic trajectories only;
  terminal epoch-59 EMA remains the comparable result.

# 2026-07-23 Every-five-epoch learning-curve evidence and gate repair
- The sampling-rate official-60 matrix now persists evaluation results at
  epochs `5/10/.../60`; future result tables must show the complete curve,
  peak epoch and terminal EMA, rather than treating only the final number as
  evidence. Evaluation metrics are observational and do not feed training.
- `1181495--1181498@ff8dfb3` all stopped in the real model gate before the
  first official optimizer update: rate-only still executed a contribution
  head; the transition and full-ASFormer gradient ownership assertions did
  not pass. No checkpoint or mAP exists for these jobs.
- `3d133a08fe6a90fd2cd1426a78b0a17a87d2b348` removes the rate-only head path
  and adds permanent action/transition/policy-hidden gate diagnostics. Local
  focused check: `5 passed`. Real CUDA diagnostic `1181556` is active under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_3d133a0_gate_debug_20260723_1425`.

- 2026-07-23 14:06 +08:00: Read-only N16R4 audit recovered four single-seed DUCA
  boundary-burst terminal artifacts: R4Q5 63.4211, R2Q3 soft 63.9794, hard 64.0002,
  and soft-adapted 64.1539 Avg-mAP, all below matched exact-uniform 64.49. T1 actual
  completed at 64.0200 and MobileNet fusion at 64.33, also below uniform. The four
  sampling-rate final-matrix jobs did not remain valid concurrently: 1181495/1497/1498
  failed before a comparable result while 1181496 rate-only remained running. C3 remains
  unproven; no cost claim is added.

- 2026-07-23 14:12 +08:00: Follow-up N16R4 `sacct` check showed 1181496 also failed after
  1 m 55 s. Thus all four aa439fb sampling-rate matrix arms 1181495--1181498 are terminal
  failures with no terminal evaluation artifact or mAP; the matrix is not running.

- 2026-07-23 14:13 +08:00: Read-only audit of the two live checkpoint-curve diagnostics:
  `1180868` completed exact-uniform at one-based epochs 10/20/30/40/50 =
  30.86/48.56/60.16/63.13/64.30 Avg-mAP and R2Q3 at 10/20/30/40 =
  30.67/48.81/59.70/62.64 (epoch 50 evaluating). `1180869` completed soft-detached at
  30.34/47.67/59.99/62.64/63.65 and hard-detached at 29.83/47.51/59.58/62.88
  (epochs 10/20/30/40; epoch 50 evaluating). R4Q5 and soft-adapted have not started.
  Status remains `experiment_running`, diagnostic-only; no periodic value changes the sealed
  epoch-59 EMA terminal comparison or supports a cost claim.

- 2026-07-23 14:45 +08:00: Submitted `1181557` (one GPU, Slurm) to draw a matched
  selector-decision diagnostic for terminal R2Q3 hard-detached and MobileNet fixed fusion.
  The job replays one deterministic four-sample validation batch using the current exporter,
  geometry analyzer, timeline/dashboard, and geometry-suite plotters; it performs zero optimizer
  updates and writes only under `duca_selection_visualization_20260723_1445`. Status:
  `experiment_running`, visualization-only. GT is an evaluation overlay, never selector input.

- 2026-07-23 14:48 +08:00: `1181557` was a zero-runtime wrapper failure: strict-unset mode
  was active while `/etc/profile` read an unset `XDG_DATA_DIRS`. No Python/GPU/optimizer action
  occurred. `1181559` is its only valid replacement; it temporarily disables `set -u` while
  sourcing the profile and restores it before the same one-GPU selector-only replay.

- 2026-07-23 14:50 +08:00: `1181559` reached the current exporter but stopped before model
  construction because the R2Q3 config requires its recorded epoch-19 frontend-init path/SHA/epoch
  environment variables. It emitted no selector records and made zero optimizer updates. `1181560`
  is the sole replacement, bound to those values from the terminal artifact; status remains
  `experiment_running`, visualization-only.

- 2026-07-23 14:54 +08:00: `1181560` loaded the original validation videos and reached the
  R2Q3 selector exporter, but its official validation pipeline did not collect
  `gt_boundary_validity` for a post-hoc overlay. No complete record or optimizer update occurred.
  Derived visualization-only configs add that existing annotation field while retaining the same
  validation video/window pipeline and passing no GT to selector `forward_test`. Malformed-path
  `1181563` was cancelled immediately; `1181564` is the sole corrected one-GPU replay.

- 2026-07-23 14:58 +08:00: `1181564` completed R2Q3 selector-only export but MobileNet tried
  to download torchvision initialization weights and the GPU node had no DNS route; no MobileNet
  selector record or optimizer update resulted. `1181572` replaces it using a derived
  selector-only config with constructor download disabled, followed by strict terminal checkpoint
  load. This preserves the terminal MobileNet weights and decision rule.

- 2026-07-23 15:01 +08:00: Static inspection found `1181572`'s constructor override at the
  selector root instead of nested `actionness_source_cfg`; it was cancelled before generating
  a valid MobileNet record. CPU config precheck for sole replacement `1181575` passes with
  `actionness_source_cfg.mobilenet_pretrained=False`, while the terminal state still strict-loads.

- 2026-07-23 15:03 +08:00: `1181575` completed selector-only replay and rendering. Four
  validation windows per arm were exported under K=384/G2; selected-set Jaccard
  R2Q3-versus-MobileNet is 1.0000 (253 valid frames; saturated), 0.6879 (503), and
  0.3813/0.3497 on the two full 768-frame windows of `video_test_0000007`. Timeline and dashboard
  PNGs are `tested` mechanism diagnostics; GT remains overlay-only and no mAP/cost claim changes.

- 2026-07-23 15:00 +08:00: Exact commit `d2fd58d` adds official full-validation
  mAP at epochs 5/10/.../60 for the sampling-rate official-60 route, records
  a per-epoch EMA metric JSON, and retains the best validation checkpoint by a
  pointer without changing training gradients or stopping. Real CUDA gate
  `1181576` passed with a true first-step transition-to-ASFormer gradient
  path after removing rate-route-only zero output initialization. This is
  `implemented/tested`, not terminal mAP evidence; rate official-60 arms must
  be re-submitted from the exact commit.

- 2026-07-23 15:05 +08:00: Submitted the independent seven-arm sampling-rate
  official-60 matrix at `d2fd58d`: Jobs `1181580--1181586`, with all receipts
  recorded in `duca_rate_d2fd58d_every5_20260723_144503/jobs.tsv`. Each run
  evaluates official validation mAP at epochs 5/10/.../60 and preserves its
  best-validation EMA pointer. Initial `PENDING(Priority)` is scheduler wait,
  not a numerical or model failure.

- 2026-07-23 15:18 +08:00: Corrected the seven-arm sampling-rate matrix
  routing. `1181580--1181586` all exited before an optimizer update because
  their configs lacked the already-existing selected-axis formal-protocol
  declaration and were rejected by the legacy P0 validator; they contain no
  mAP/checkpoint/model evidence. Commit `9491ba1` declares the existing
  protocol and maps all seven variants to the existing binder (Linux focused
  tests: `7 passed`). Fresh independent official-60 jobs `1181593--1181599`
  were submitted without dependencies at
  `duca_rate_9491ba1_every5_20260723_151800`; each runs the official THUMOS
  evaluator every five epochs and writes a best-EMA pointer plus terminal EMA.

- 2026-07-23 15:24 +08:00: Repaired the R2Q3 selection diagnostic figure from
  the original `r2q3_hard_detached.jsonl` record for
  `video_test_0000007|6912`. The new plot uses raw `gt_segments` for four
  post-hoc GT intervals/boundaries and plots `transition_policy_scores`; the
  misleading flat blue `p_action` line is removed. It leaves the R2Q3 and
  MobileNet fixed-fusion K=384 selected-position rugs comparable and changes
  neither inference inputs nor experimental results (`tested` visualization
  only).

- 2026-07-23 15:25 +08:00: Quantified the corrected full-window diagnostic:
  of 94 raw-GT positions, hard-detached R2Q3 selected 53 (56.4% coverage;
  13.8% of K=384) while MobileNet fixed fusion selected 79 (84.0%; 20.6%).
  This explains the apparent MobileNet advantage in this one post-hoc selector
  plot, but is not an mAP or cost conclusion. Its single-seed terminal mAP is
  64.33% versus R2Q3's 64.00%, and fixed fusion remains about 0.16 pp below
  matched exact-uniform.

- 2026-07-23 16:30 +08:00: Added and pushed `2a86d15` trained small-sample
  visualization diagnostics. The unified sampling-rate model receives 40 real
  optimizer updates (10 epochs x 4 batches, batch size 1); checkpoints after
  one-based epochs 1/5/10 export the same fixed training window's detector
  contribution and selector gradients, plus two fixed validation windows'
  teacher-free sampling-rate/selected-position evidence. This is deliberately
  not a separate training framework and never reports mAP. A strict DDP
  checkpoint-prefix normalization fixes the training-attribution exporter.
  Local and remote focused checks pass (`15 passed`). Job `1181615` is the
  only submitted small-sample diagnostic, currently `PENDING(Priority)`, with
  evidence root `duca_rate_2a86d15_mini_visual_20260723_1630`.

- 2026-07-23 16:41 +08:00: The first real-update mini-visual Job `1181615`
  and formal rate jobs `1181593--1181599` exposed a shared variable-window
  decoder error before usable updates: hard occupancy used the padded temporal
  length, while soft occupancy retained a truncated valid length, making the
  straight-through occupancy reject 768 versus 417 tensors. Commit `dea59d2`
  pads soft occupancy identically and adds a focused mixed-length gradient
  test that requires padded positions to remain zero-gradient. The affected
  jobs are implementation failures, not mAP/model-performance evidence; an
  exact remote Torch check and fresh mini-visual run are pending.

- 2026-07-23 16:05 +08:00: Exact Linux tests for `dea59d2` passed (`14 passed
  in 42.78s`). Fresh Job `1181648` was submitted from clean snapshot
  `opentad_duca_rate_dea59d2_20260723` for the same 40-update trained
  mini-visual diagnostic. It emits fixed-window evidence at epochs 1/5/10 and
  is not an official mAP run.

- 2026-07-23 16:47 +08:00: The first real batch of mini-visual Job `1181648`
  exposed an Autograd dtype defect: contribution distillation asked for
  `grad(loss, selected_input)` on `uint8` RGB. It exited before any optimizer
  update, so it is not performance evidence. Commit `596d982` makes only the
  temporary contribution-teacher observation float-valued without changing its
  numeric pixels, the hard sampled observation, the straight-through sampling
  route, or inference. A remote focused regression passed `3 in 37.31s`.
  Fresh trained diagnostic Job `1181671` was submitted from the exact commit;
  its 40-update visual evidence root is
  `duca_rate_596d982_mini_visual_20260723_154700`.

- 2026-07-23 16:52 +08:00: Job `1181671` completed all 40 real updates and
  retained epoch 1/5/10 checkpoints, then its attribution-only postprocessing
  failed to reload the absolute VideoMAE checkpoint. Commit `5029691` adds the
  smallest explicit pretrain-path argument only to that full-detector exporter.
  Read-only Slurm Job `1181683` now recovers contribution, sampling-rate and
  selected-frame figures from those existing checkpoints; it does not repeat
  training and is not an official mAP experiment.

- 2026-07-23 16:56 +08:00: Recovery `1181683` failed before checkpoint access
  because Slurm `--wrap` used `/bin/sh` rather than Bash. Corrected Job
  `1181684` runs the identical read-only commands under `bash -lc` into a new
  recovery directory. Neither event retrains the model or changes its evidence
  status.

- 2026-07-23 16:58 +08:00: `1181684` also failed before model work because
  multi-line shell quoting was malformed. Active `1181685` replaces it with a
  one-line Bash command, using the same completed checkpoints and exact
  `5029691` source. Only `1181685` is eligible to produce the requested
  diagnostic artifacts.

- 2026-07-23 16:25 +08:00: Implemented lightweight training-attribution v2
  at `0b7e075` on `codex/duca-density-transport-20260723`. It keeps one
  existing sampling-rate training path, expands the small diagnostic to 30
  epochs / 120 real updates, fixes two training and two validation windows,
  and emits figures at epochs 10/20/30. Primary temporal evidence is now
  AdaTAD head-input feature `|F*dL/dF|` and `|dL/dF|`; selected-RGB
  input-times-gradient is an auxiliary pixel sensitivity only. Validation
  figures are teacher-free selector inference with GT used only as a post-hoc
  overlay. This is `implemented`, not official mAP evidence; remote focused
  verification and a distinct small Slurm diagnostic remain pending.

- 2026-07-23 16:35 +08:00: Exact remote focused suite for attribution v2
  passed `12` tests in `104.84s`; py_compile and Bash syntax also passed. The
  unique mini diagnostic Slurm Job `1181894` (`duca-ratevis30`) was submitted
  independently at commit `0b7e075`, snapshot
  `opentad_duca_attribution_d844c18_20260723`, evidence root
  `duca_rate_0b7e075_mini_visual_20260723_1635`. It runs 30 epochs/120 real
  updates and produces fixed-window plots at epochs 10/20/30, not official
  mAP.

- 2026-07-23 16:38 +08:00: Mini visualization Job `1181894` entered real
  training and reached epoch 7. `cls_loss/reg_loss` are finite and nonzero,
  memory is about 5.7GB, and no terminal error marker exists. One early batch
  used both allowed AMP replays and then completed; it is not a collapse.
  Sampling-rate alpha is rising while detector-gradient weight has only just
  left zero, so no mechanism conclusion is admissible before the epoch-10
  exported figures.

- 2026-07-23 16:43 +08:00: Job `1181894` passed the epoch-10 and epoch-20
  diagnostic boundaries, writing `epoch_9.pth` and `epoch_19.pth`; it is
  continuing through epoch 26. The joint phase has sampling alpha `1.0` and
  detector-gradient bridge weight `0.25`. Finite nonzero classification,
  regression, actionness, and transition losses plus stable 5.8GB memory show
  the diagnostic is healthy. A single epoch-23 AMP replay recovered normally.
  Final postprocessing will render fixed training/validation samples at
  epochs 10/20/30; it is mechanism evidence only, never official mAP.

- 2026-07-23 16:48 +08:00: `1181894` completed all 120 training updates and
  retained epoch 10/20/30 checkpoints, but stopped during an unused legacy
  quality-summary call on a short boundary-free validation window. Direct
  rendering verified the training attribution plot works; the selected test
  windows were nevertheless invalid for sampling evidence because
  `valid_len=256 < K=384` forces full selection. Commit `5459275` removes the
  redundant call and requires two full-budget windows with valid GT
  boundaries. Read-only recovery Job `1182026` is queued into a new output
  root and will not repeat training.

- 2026-07-23 17:10 +08:00: The first recovery `1182026` exited before model
  work because `set -u` and the site `/etc/profile` disagreed on an unset
  `XDG_DATA_DIRS`; it is zero-work infrastructure history. Commit `4253639`
  corrects a more important instrumentation error: the density-transport
  bridge is measured through density and soft slot-assignment gradients, not
  just hard-decoder rate logits. Remote PyTorch focused tests passed. Job
  `1182079` is a checkpoint-only recovery from `1181894`, outputs into
  `postprocess_4253639`, uses full-budget boundary-bearing validation windows,
  and never repeats any optimizer update.

- 2026-07-23 17:18 +08:00: The epoch-10 coarse `p_action` is genuinely weak,
  not a scale-only artifact: on the fixed 768-point window it spans only
  `0.4117--0.6670` (std `0.0433`) and the mini-run actionness BCE remains near
  `0.69`. This 120-update joint diagnostic is therefore not valid evidence
  that the coarse binary probe learned actionness. It motivates a matched
  stage-one coarse-probe convergence check and action-target overlay before
  interpreting density-selection performance; detector contribution
  distillation losses are zero in this mini run.

- 2026-07-23 17:xx +08:00: Replaced the ambiguous sampling-rate warmup story
  with one canonical curriculum implementation. Stage 1 is exact-uniform
  K=384 full-model training on the complete training subset, with binary
  actionness, transition-distribution, and boundary supervision monitored at
  every 5 epochs through AP/AUC/Brier/ECE and boundary support. Stage 2 loads
  the terminal Stage-1 EMA strictly for the whole model, resets optimizer /
  scheduler / AMP and the selector schedule, then ramps policy, contribution
  distillation, density-transport detector gradient, and full-ASFormer policy
  adaptation over 3,000 updates. The final 3,000 updates remain TAD-led but
  retain actionness/transition/boundary weights 0.25/0.10/0.25. This is
  `implemented_local_pending_remote_gate`, not a new performance result.

- 2026-07-23 18:xx +08:00: The canonical rate-curriculum implementation was
  pushed as `codex/duca-density-transport-20260723@8ae6371`. On the remote
  CUDA environment, `tests/test_duca_sampling_rate_official60.py` and
  `tests/test_duca_two_stage_curriculum.py` passed together (`15 passed in
  46.78s`); the train entry point, both configs, initializer and runner also
  passed syntax checks. The status advances to
  `implemented_tested_pending_experiment`. This verifies strict full-model
  EMA initialization and phase-state reset, not coarse convergence or final
  TAD mAP.

- 2026-07-23 18:xx +08:00: Submitted the canonical full-data two-phase
  curriculum as single-GPU Slurm Job `1182195` at exact commit `8ae6371`.
  It first trains uniform full-model K=384 for 30 epochs, exports coarse
  convergence diagnostics from epochs 5/10/15/20/25/30, then strictly loads
  its terminal EMA into the 60-epoch low-LR joint phase. The job records
  terminal OpenTAD validation mAP and selector-quality diagnostics. The site
  rejects explicit per-job memory requests under its per-GPU policy, so the
  submission intentionally uses the scheduler's standard one-GPU allocation;
  this changes no model or experiment setting.

- 2026-07-23 18:xx +08:00: The initial submission `1182195` failed before
  model construction or any optimizer update because its launch command used
  an incorrectly transcribed 40-character commit SHA. The fail-closed commit
  check worked as designed. It was replaced without changing code, data or
  hyperparameters by `1182341`, whose `DUCA_EXPECTED_COMMIT` is read directly
  from the clean remote checkout (`8ae637116602a0a8c0841d7b8a4045e613f5934a`).

- 2026-07-23 18:xx +08:00: Replacement `1182341` reached Stage-1 model
  construction and fail-closed before any update: the stage-one override set
  `training_uniform_companion_fraction=0` but inherited learned-row gradient
  normalization from the joint base. Commit `04b7df4` explicitly disables
  that inapplicable normalizer for uniform warmup and adds a regression
  assertion. The remote focused suite again passed (`15 passed`); its
  standalone bare builder check requires a transform registry that
  `tools/train.py` normally initializes, so the prior real `tools/train.py`
  construction is the relevant integration path. Job `1182391` is the sole
  active replacement at exact `04b7df4`; no prior curriculum job executed an
  optimizer update.

- 2026-07-26 03:xx +08:00: Job `1182391` completed Stage 1 (30 uniform-K=384
  epochs) and sealed `epoch_29.pth` with SHA-256
  `7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e`.
  Its scheduled diagnostics reached `60.39` Avg-mAP at epoch 30, which is a
  warmup learning-curve point rather than a 60-epoch main result. Stage 2
  failed before model construction and any optimizer update: it had inherited
  `formal_successful_update_contract=True` from the old P0 base while its
  formal protocol was empty, causing the P0 binder to reject an empty variant.

- 2026-07-26 03:xx +08:00: Commit `5a87529` marks the Stage-2 curriculum as
  non-P0 and verifies that configuration. Commit `b554f04` adds a strictly
  SHA-bound Stage-2-only launcher path so Stage 1 is not repeated. Both
  commits passed the remote focused suite (`15 passed`); Job `1190439` is
  queued with one GPU to reuse the sealed Stage-1 EMA under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_b554f04_stage2_recovery_20260726_034417`.

- 2026-07-26 04:xx +08:00: Job `1190439` failed at zero runtime on its assigned
  node, before Python, model construction, or an optimizer update. The Slurm
  stderr is `/etc/profile.d/apps-bin-path.sh: line 11: XDG_DATA_DIRS: unbound
  variable`: its outer submission wrapper had sourced the site profile under
  `set -u`. This is an environment-launch failure, not curriculum evidence.
  The sole replacement `1190528` retains exact commit `b554f04` and the sealed
  Stage-1 SHA, safely relaxes nounset only while loading the site environment,
  then restores strict mode before invoking the existing Stage-2-only runner.
  It writes to
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_b554f04_stage2_recovery_20260726_040504`;
  Stage 1 is not rerun.

- 2026-07-26 04:05 +08:00: `1190528` cleared the repaired environment launch
  and is running on `g0053`. Its `stage2/train.out` records strict full-model
  initialization from the sealed Stage-1 `state_dict_ema` at epoch 29, including
  reset of `frame_selector._loss_weight_schedule_step`, then `Epoch 0 started`.
  The scoped scan contains no Traceback, OOM, non-finite loss, ValueError, or
  AMP-replay exhaustion. It has not yet emitted a finite successful-update
  receipt, diagnostic, or performance result.

- 2026-07-26 04:31 +08:00: `1190528` reached schedule step `350` under its
  successful-optimizer-step schedule, with finite losses and approximately
  `9874 MB` memory. AMP skipped two batches but each recovered at replay `1/8`;
  there is no non-finite collapse or replay exhaustion. The density policy is
  leaving uniform selection (`policy_alpha_w=0.0332`), whereas the detector
  gradient is correctly still zero during the configured first-1,000-update
  warmup. Actionness/transition/boundary supervision remains active. No
  five-epoch quality diagnostic or mAP has been emitted, so this is only a
  healthy-training receipt, not performance evidence.

- 2026-07-26 04:42 +08:00: Stage 2 sealed `epoch_4.pth` (one-based epoch 5,
  SHA-256 `4016ca083dd56286b6a32edce92f085945ce415c0b237c1fc43b0cb7ad1cc2a3`).
  The live runner only checkpoints at this cadence, so its first five-epoch
  test is Job `1190606`: a read-only full official-validation evaluation and
  fixed 64-window actionness/calibration/boundary/selection-quality export.
  The manifest marks `training_mutation=false` and `selection_rule=none`; the
  result is a learning-curve diagnostic and cannot select a checkpoint or
  alter the terminal epoch-59 rule.

- 2026-07-26 05:05 +08:00: The first epoch-5 diagnostic Job `1190606` exited
  in one second before Python or model construction because its standalone
  wrapper omitted `scripts/duca_cellcf_canonical_env.sh`, leaving `PYTHON`
  unset under strict mode. This is a zero-runtime evaluator-launch failure, not
  a test or model result. Job `1190626` is the only replacement; it reuses the
  same epoch-5 checkpoint and SHA and changes only the wrapper to source the
  canonical environment before its read-only evaluation and fixed-window
  quality export. Main Stage-2 training remains healthy through epoch 8.

- 2026-07-26 05:09 +08:00: The live Stage-2 runner completed its scheduled
  full THUMOS validation after one-based epoch 5 and reported diagnostic
  Avg-mAP `60.52%` (`76.92/71.98/63.70/52.93/37.08%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). This is a fixed learning-curve observation only: it
  cannot select a checkpoint, substitute for terminal epoch-59 EMA evaluation,
  or be compared with the matched 60-epoch uniform result. Training continued
  with finite updates, stable approximately 9.9 GB memory, and no replay
  exhaustion.

- 2026-07-26 05:35 +08:00: The e5 read-only replacement `1190626` sourced the
  canonical environment correctly but failed before DDP initialization, model
  construction, inference, or a metric result: `tools/test.py` could not JSON
  serialize the frozen `CellCFTrainingProtocol` while hashing the resolved
  configuration. Commit `a1bf61d` adds deterministic dataclass conversion to
  the shared evaluator hash utility and makes the test entry point consume it;
  its focused regression suite passed locally and on the clean remote exact
  checkout (`9 passed`). Job `1190633` reused `epoch_4.pth` and SHA
  `4016ca083dd56286b6a32edce92f085945ce415c0b237c1fc43b0cb7ad1cc2a3`, requests
  no training mutation, and has `selection_rule=none`, but it failed before
  the Stage-2 checkpoint load because its wrapper omitted the absolute
  `model.backbone.custom.pretrain` override used by the training launcher. The
  relative default was not present in the remote repository. Job `1190637`
  preserved every evidence binding, added the already-frozen pretrain override,
  strictly loaded the EMA checkpoint, and began official testing, but it was
  cancelled after `10/396` windows because it omitted the required fixed
  64-window actionness/calibration/boundary export. It yielded neither final
  mAP nor quality records and is read-only workflow evidence only. The unique
  replacement Job `1190643` adds that export while preserving the same
  bindings: full official validation followed by exactly 64 fixed validation
  quality windows, with `selection_rule=none` and no training mutation. Two
  explicit-memory submission attempts were rejected by Slurm before a job
  existed and are not experiment or model
  evidence.

- 2026-07-26 06:29 +08:00: Read-only Job `1190643` completed `0:0` on the
  sealed e5 EMA `4016ca083dd56286b6a32edce92f085945ce415c0b237c1fc43b0cb7ad1cc2a3`.
  Its full official 211-video/396-window THUMOS evaluation reports Avg-mAP
  `60.521318%` (`76.917970/71.979260/63.698867/52.930814/37.079679%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), independently matching the live e5 curve. Its
  manifest is read-only with `selection_rule=none`; this remains a curve
  diagnostic, never a checkpoint-selection or final-comparison result. The
  required fixed 64-window/40-video selector-only export completed with no
  detector, GT, raw-prediction, or teacher input to the selector. Coarse
  actionness AUPRC/AUROC/Brier/ECE are
  `0.343840/0.577626/0.202376/0.012158`; learned action enrichment is
  `1.003583` (CI `0.998122--1.008924`), r0 boundary recall is `0.537405`
  (CI `0.482378--0.610760`), and the learned-minus-uniform r0 recall delta is
  `-0.017789` (CI `-0.075717--0.046480`). Pure-delta same-feasible-DP has
  enrichment `1.070008` and R2Q3/R4Q5 both-endpoint coverage
  `0.506823/0.509369`, versus learned `0.331581/0.356900`; this is
  selector-mechanism evidence only and shows the learned policy has not yet
  exploited the available boundary microclusters.

- 2026-07-26 06:14 +08:00: Stage-2 Job `1190528` ended `FAILED 1:0` after
  1,000 finite updates, stable approximately 9.9 GB memory, and a sealed
  `epoch_9.pth`. Its scheduled e10 official validation immediately beforehand
  is diagnostic Avg-mAP `61.62%` (`77.94/72.84/64.61/54.34/38.39%`), not a
  terminal result. Directly after `Epoch 10 started`, the first following
  train forward raised the formal pre-AMP `FloatingPointError` for non-finite
  loss. This is a fail-closed numerical failure of the affected Stage-2 arm,
  not offline TAD model evidence and not a terminal EMA evaluation. The event
  is adjacent to the detector-gradient/contribution warmup boundary, but the
  causative component is unknown. Before any recovery, run one read-only,
  hash-bound e10 batch from `epoch_9.pth` and record per-component finite
  status and schedule weights. Do not rerun Stage 1, select e5/e10, relax
  strict loading, bypass dependencies, or resubmit healthy arms.

- 2026-07-26 06:50 +08:00: Submitted unique read-only numerical diagnostic
  Job `1190683` (`duca-rate-e10-diag`; pending Slurm priority) for Stage-2
  recovery prerequisite only. It binds commit
  `3a87132d60b0a328ccbe9d153e795a7ce3987911`, strict Stage-1 EMA
  initialization, `epoch_9.pth/state_dict` SHA
  `3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`, epoch
  10/batch 0, seed 3407, and the training's absolute pretrain path. The
  separate manifest prohibits backward, optimizer/scheduler/EMA construction,
  checkpoint mutation, and checkpoint selection. Its sole output is each loss
  component's finite status plus schedule weights; it is not offline TAD model
  evidence and cannot authorize a recovery until terminal inspection.

- 2026-07-26 06:51 +08:00: `1190683` completed `0:0`; its input training
  checkpoint SHA is unchanged before/after. Under exact commit
  `3a87132d60b0a328ccbe9d153e795a7ce3987911`, strict sealed Stage-1 transfer,
  and `epoch_9.pth/state_dict`, the read-only epoch-10 batch-0 forward at
  selector step 1000 has finite cost `3.402128` and every recorded loss is
  finite. Detector-gradient, detector-contribution, and detector-utility
  weights are exactly zero. The original runner emitted a backward-kernel
  warning after Epoch 10 began and before its later pre-AMP exception; batch 0
  therefore reached a successful post-optimizer schedule advance. This rules
  out a batch-0/step-1000 zero-weight explanation but not the next forward at
  step 1001. Commit `45198e45af141605db3bda31ccc54a7ac58e4c8c` adds the only
  admissible follow-up mode: batch 1 with an in-memory selector-step override
  to its immediate post-optimizer value, still with no backward, optimizer,
  scheduler, EMA, checkpoint mutation, or selection.

- 2026-07-26 06:56 +08:00: `1190699` completed `0:0` under exact commit
  `45198e45af141605db3bda31ccc54a7ac58e4c8c`. It reads epoch-10 batch 1 with
  only an in-memory selector-clock override to step 1001 and preserves all
  parameters and checkpoint bytes. No backward, optimizer, scheduler, or EMA
  is constructed. Cost `4.324651` and all losses are finite; the first nonzero
  detector-gradient/contribution weights (`1.542125e-7`/`6.168501e-7`) and
  cls/reg contribution losses (`4.095707e-6`/`4.101499e-6`) are finite. The
  schedule opening alone does not reproduce the Stage-2 failure, so recovery
  remains blocked pending a post-batch-0-state or input/kernel explanation.

- 2026-07-26 15:xx +08:00: The user clarified that a bounded number of
  non-finite events can be acceptable when AMP recovers them. Inspection of
  the resolved Stage-2 config established `max_amp_retries_per_batch=8`, but
  the observed event was rejected by `require_finite_train_loss=True` before
  AMP scaling, so the existing replay path never ran. Commit
  `49caf7bb7627147bd2a4b37378606480816ae05c` adds an isolated in-memory
  diagnostic rather than changing the training policy: it restores
  `epoch_9.pth` model/AdamW/scheduler/EMA/GradScaler, executes batch 0 through
  the exact AMP update and selector-step advance, then inspects batch 1 across
  eight controlled RNG trials. The sealed checkpoint has no serialized RNG
  state, so the report explicitly records that it is a controlled
  post-update-state probe, not an exact stochastic replay. Remote focused
  tests passed (`5 passed`); Job `1191745` is the sole submitted diagnostic,
  has no terminal-evaluation path, and writes no training mutation.

- 2026-07-26 16:xx +08:00: The post-update isolation prerequisite completed.
  Diagnostic Job `1191745` is invalid workflow evidence only: its file-mode
  invocation failed before model construction with `ModuleNotFoundError`.
  Corrected module-mode Job `1191754` completed `0:0` under
  `65a4cfb31716f84c153af881a71fe05069637848`. It strictly restored sealed
  Stage-2 `epoch_9.pth` SHA
  `3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc` in
  memory, ran the real batch-0 AMP update, then tested batch 1 after the
  1000-to-1001 selector/scheduler transition under controlled seeds 3407--3414.
  All eight outcomes were finite. Every trial had finite batch-0 gradients,
  finite 49,914,588 post-update parameters, finite 56,069,713 optimizer-state
  values, unchanged GradScaler scale 8192, and finite backbone/projection/neck
  outputs plus all batch-1 loss components. Checkpoint bytes and persistent
  optimizer/scheduler/EMA state were unchanged; no terminal evaluation ran.
  This rules out persistent finite-to-nonfinite contamination and an
  immediately deterministic contribution-schedule boundary. Because epoch-9
  stores no RNG state, the original draw cannot be exactly replayed; the
  remaining justified diagnosis is a one-shot pre-AMP stochastic or
  nondeterministic forward transient, not an offline TAD performance result.

- 2026-07-26 16:xx +08:00: Commit
  `9519760a26cd7fda08c3e648b1e7d7f459b3b6b` adds the smallest recovery policy:
  Stage-2 alone sets `max_nonfinite_loss_retries=8`; a non-finite pre-AMP cost
  restores RNG, buffers, and custom state, records atomic
  `stage2/update_audit.json` counters, and replays the same batch. It does not
  execute optimizer, scheduler, selector, or EMA updates until a finite
  forward succeeds; exhaustion restores state and raises. Focused code precheck
  Job `1191787` completed `0:0` with 15 tests passing. It is a code-validation
  job only, with no model update, checkpoint, or mAP. This permits exactly one
  Stage-2 recovery from the sealed epoch-9 source; no Stage-1 rerun, no
  `strict=False`, no loss/schedule retuning, and no intermediate-mAP checkpoint
  selection are authorized.

- 2026-07-26 16:12 +08:00: The isolated recovery launcher at commit
  `adc6fb13114584188da4ac17eeeab6d89d69d04f` passed Slurm precheck Job
  `1191796` (`COMPLETED 0:0`). It binds the sealed Stage-1 epoch-29 SHA
  `7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e` and
  Stage-2 epoch-9 SHA
  `3d1444da7fbae2566ab379501db353900219d2bc23c918654db26e13833016fc`, emits a
  recovery manifest declaring the eight-replay fail-closed policy, and exits
  before `tools/train.py`. An initial `--mem=62200M` submission was rejected
  by the current single-GPU 55GB scheduler limit before a Slurm job existed;
  removing only that obsolete resource request produced the valid precheck.
  This is launcher evidence only, with no update, checkpoint, or offline TAD
  mAP. It authorizes one and only one Stage-2 continuation from epoch 9.

- 2026-07-26 16:15 +08:00: Submitted the one authorized Stage-2 continuation
  Job `1191806` (`duca-rate-stage2-recovery`, `experiment_running`) at commit
  `adc6fb13114584188da4ac17eeeab6d89d69d04f`. Its sole run root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_adc6fb1_stage2_recovery_20260726_161500`.
  It resumes only sealed Stage-2 epoch 9 under the prechecked strict bindings,
  uses the bounded same-batch replay audit, and retains terminal epoch-59 EMA
  OpenTAD official mAP as the only performance endpoint. No terminal result is
  available yet; e5/e10 values remain non-selecting curve diagnostics.

- 2026-07-26 16:16 +08:00: Job `1191806` failed closed (`FAILED 1:0`) at
  epoch 10/batch index 2. Batches 0 and 1 completed exactly two finite
  optimizer, selector, scheduler, and EMA updates. The pre-AMP cost at batch
  2 was non-finite on the initial forward and every one of eight same-batch
  replays. Its atomic audit records 9 attempts, 8 replays, 9 state
  restorations, 1 replay exhaustion, and no AMP-skip path; no failed-batch
  optimizer/scheduler/selector/EMA update occurred. This is the first
  reproducible affected state, but the loss component is not yet isolated.
  The job has no terminal checkpoint or offline TAD mAP and must not be
  restarted. The sole admissible next action is a read-only, hash-bound
  prefix-state diagnosis that reproduces the two finite updates in memory and
  reports batch-2 component and module finite status.

- 2026-07-26 16:45 +08:00: Read-only prefix diagnoses `1191823`, `1191833`,
  and `1191840` reproduced the sealed `epoch_9.pth` prefix through two finite
  AMP updates and then the same batch-2 pre-AMP failure. Backbone, projection,
  neck, detector cls/reg objectives, selected RGB inputs, their first-order
  gradients, and selected-frame contributions were all finite. The exact
  failure was downstream: FP16 contribution logits were masked with `-65504`
  before division by temperature `0.7`, producing 816 masked `-inf` entries
  per cls/reg call. Multiplying zero target mass by those `-inf`
  log-probabilities yielded NaN. This is a deterministic masked-distribution
  arithmetic defect, not persistent optimizer/EMA corruption or an offline
  TAD result. The diagnostics persisted no checkpoint, optimizer, scheduler,
  or EMA state and did not run terminal evaluation.

- 2026-07-26 16:53 +08:00: Minimal affected-arm repair commit
  `4c1f5384ae693c74a141619ded03196a72c594ed` applies temperature scaling
  before the finite invalid-position mask. It preserves valid logits and does
  not alter the sampler, schedule, strict loading, or replay bound. Remote
  verification Job `1191853` completed `0:0` with 32 tests. Hash-bound
  read-only replay `1191854` made the former batch finite: total cost
  `3.2601470947265625`, cls contribution loss `1.4956006452848669e-05`, and
  reg contribution loss `1.2520967175078113e-05`. This is numerical repair
  evidence only, not training, a terminal checkpoint, or offline TAD mAP.

- 2026-07-26 17:00 +08:00: Recovery launcher precheck Job `1191874`
  completed `0:0` at commit `4c1f5384ae693c74a141619ded03196a72c594ed` and
  wrote only its fresh recovery manifest. The single repaired affected-arm
  continuation Job `1191880` is `RUNNING` under
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate_curriculum_4c1f538_stage2_maskfix_20260726_165800`.
  Its train receipt confirms strict full-model Stage-1 `state_dict_ema`
  initialization from epoch 29 with only
  `frame_selector._loss_weight_schedule_step` reset, then strict resume of
  sealed Stage-2 epoch 9 and start of epoch 10. No optimizer update, quality
  diagnostic, checkpoint, or offline TAD mAP is claimed yet.

- 2026-07-26 17:03 +08:00: `1191880` remains healthy through epoch 10 update
  50/99: total loss `3.5796`, cls/reg contribution-distillation losses
  `0.0027/0.0026`, detector-gradient weight `0.0004`, and memory `10176MB`
  are finite. No Traceback, OOM, non-finite event, AMP replay exhaustion, or
  failed update is present. This is runtime health evidence only; no checkpoint
  is selected and no offline TAD mAP has been produced.

- 2026-07-26 18:07 +08:00: Controller cancelled `1191880` (`CANCELLED 0:0`)
  after detecting a course-protocol defect: the inherited Stage-2 workflow
  had `intermediate_validation_selects_checkpoint=True` and wrote
  `best_validation_ema.json`. It completed 700 finite updates first; its
  audit has zero non-finite attempts, replays, state restorations, AMP skips,
  or exhausted retries, with 700 matching optimizer/scheduler/selector/EMA
  updates. The pointer did not feed back into training, but intermediate mAP
  selection is forbidden, so the run is not offline TAD model evidence and
  none of its checkpoints may be reused. Its epoch-15 EMA curve audit is
  `62.403751%` Avg-mAP (`79.036658/73.799613/64.984514/54.566999/39.630969%`)
  at tIoU `0.3/0.4/0.5/0.6/0.7`; it is not a terminal result or a comparator.
  The expected AP/AUC/Brier/ECE/state-transition-boundary quality export was
  absent. The smallest repair now disables intermediate selection in the
  Stage-2 config and adds a launcher precheck; no model/schedule/loss change
  is authorized.

- 2026-07-26 18:16 +08:00: Contract-repair precheck `1191956` completed
  `0:0` at commit `42dba3f90b37243e7965d18b6707e88e81bf7109`. Its manifest
  binds the original Stage-1/e9 hashes, `learning_curve_only`, no intermediate
  selection, and bounded fail-closed replay; it runs no training or mAP.
  Job `1191957` is the only new strict Stage-2 continuation. Its launcher
  emits read-only quality exports for epochs 5 through 60 after training, and
  accepts only terminal epoch-59 EMA OpenTAD official mAP as performance
  evidence.

- 2026-07-26 18:36 +08:00: Corrected sole continuation `1191957` remains
  `RUNNING` on `g0024` at exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`. Its epoch-12 audit proves
  300 attempted and 300 successful optimizer/scheduler/selector/EMA updates
  after strict e9 resume, with zero non-finite attempts, replay/state
  restorations, AMP skips, or exhaustion. Epoch 13 started; no selection
  pointer and no Traceback/OOM/FAIL were found. This is runtime-health
  evidence only, not an offline TAD checkpoint or mAP result.

- 2026-07-26 19:35 +08:00: `1191957` reached 1,000 successful post-e9
  updates through epoch 19 on exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`. There are zero non-finite-loss
  attempts/replays/exhaustions. Two AMP-overflow attempts each restored state
  and succeeded on their first same-batch replay, leaving exactly 1,000
  optimizer/scheduler/selector/EMA updates; this bounded transient is
  acceptable and not a failure. Its one-based epoch-15 diagnostic curve is
  `62.40%` Average-mAP, with no selection pointer. That value cannot select a
  checkpoint, substitute for a matched-uniform comparison, or be reported as
  an offline TAD result. No Traceback/OOM/fail-closed event is present.

- 2026-07-26 20:05 +08:00: `1191957` reached 1,200 successful post-e9
  updates and began epoch 22. The audit remains zero for non-finite loss
  attempts/replays/exhaustions; the earlier two AMP-overflow restores remain
  the only bounded transient events. The one-based epoch-20 diagnostic EMA
  curve is `63.15%` Average-mAP (`79.19/74.28/65.93/55.74/40.60%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), with no selection pointer. It is learning-curve
  audit only, not matched-uniform evidence, a checkpoint selector, or an
  offline TAD result.

- 2026-07-26 20:35 +08:00: `1191957` completed 1,500 successful post-e9
  updates and began its diagnostic-only epoch-25 evaluation. Loss non-finites
  remain zero. The third AMP-overflow attempt (epoch 24, batch 27) restored
  state and completed on first same-batch replay; the audit preserves matched
  1,500 optimizer/scheduler/selector/EMA updates and zero exhaustion. No new
  mAP is yet available.

- 2026-07-26 21:05 +08:00: `1191957` reached 1,900 successful post-e9
  updates and began epoch 29 with no added AMP or non-finite-loss event. Its
  one-based epoch-25 diagnostic EMA curve is `63.98%` Average-mAP
  (`79.88/75.62/67.19/56.08/41.15%` at tIoU `0.3/0.4/0.5/0.6/0.7`), no
  selection pointer. The curve is audit-only, not a matched-uniform
  comparison, checkpoint selector, or offline TAD result.

- 2026-07-26 21:38 +08:00: `1191957` completed 2,100 successful post-e9
  updates through epoch 30 and remains running. Its one-based epoch-30
  diagnostic EMA curve is `64.40%` Average-mAP
  (`80.37/75.07/67.28/57.11/42.18%` at tIoU `0.3/0.4/0.5/0.6/0.7`), with no
  selection pointer. This curve is audit-only, not a matched-uniform
  comparison, checkpoint choice, or terminal offline TAD result. The fourth
  AMP-overflow event (epoch 30, batch 12) restored state and succeeded on first
  same-batch replay; the audit retains matched 2,100
  optimizer/scheduler/selector/EMA updates, four restores/replayed batches,
  zero replay exhaustion, and zero non-finite-loss attempts/replays/exhaustions.

- 2026-07-26 22:05 +08:00: sole active Stage-2 continuation `1191957` remains
  on exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109` and has completed
  2,500 successful post-e9 updates through epoch 34. No additional AMP event
  occurred; the four bounded replays retain one retry each, zero loss
  non-finites, zero replay exhaustion, and exactly matched 2,500
  optimizer/scheduler/selector/EMA updates. The strict e9 dependency remains
  present; no selection pointer, Traceback, OOM, fail-closed receipt, or
  terminal epoch-59 EMA official offline TAD mAP exists.

- 2026-07-26 22:35 +08:00: `1191957` completed 2,600 successful post-e9
  updates through epoch 35. Its one-based epoch-35 diagnostic EMA curve is
  `65.20%` Average-mAP (`80.56/75.94/67.87/58.24/43.36%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), with no selection pointer. This is learning-curve
  audit only, not checkpoint selection, a matched-uniform comparison, or a
  terminal offline TAD result. Four accepted bounded AMP replays remain the
  only transient events; loss non-finites/exhaustions remain zero and the
  2,600 optimizer/scheduler/selector/EMA updates are exactly matched.

- 2026-07-26 23:37 +08:00: `1191957` completed 3,300 successful post-e9
  updates through epoch 42. Its one-based epoch-40 diagnostic EMA curve is
  `65.13%` Average-mAP (`80.57/75.73/67.82/58.22/43.31%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), `-0.07pp` versus the epoch-35 diagnostic. It has no
  selection pointer and remains audit-only, not checkpoint selection, a
  matched-uniform comparison, or terminal offline TAD performance. The audit
  retains exactly matched 3,300 optimizer/scheduler/selector/EMA updates, four
  bounded AMP restores/replays, zero loss non-finites/exhaustions, and no
  Traceback/OOM/fail-closed receipt.

- 2026-07-27 00:08 +08:00: `1191957` completed 3,500 successful post-e9
  updates through epoch 44 and wrote the immutable one-based epoch-45
  diagnostic EMA JSON. Average-mAP is `64.94%`, with
  `80.31/75.58/67.73/57.84/43.23%` at tIoU `0.3/0.4/0.5/0.6/0.7`, `-0.19pp`
  versus e40 and `-0.26pp` versus e35. It is a read-only learning-curve point,
  not checkpoint selection, early stopping, a matched-uniform comparison, or
  terminal offline TAD performance. No selection pointer, new AMP event,
  non-finite loss, replay exhaustion, Traceback, OOM, or fail-closed receipt
  exists; optimizer/scheduler/selector/EMA updates remain exactly matched.

- 2026-07-27 00:59 +08:00: `1191957` completed 4,000 successful post-e9
  updates through epoch 49 and wrote the immutable one-based epoch-50
  diagnostic EMA JSON. Average-mAP is `65.650497%`, with
  `80.433202/76.607056/68.955569/58.776518/43.480139%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`. It remains a read-only learning-curve point, not
  checkpoint selection, early stopping, a formal matched-uniform comparison,
  or terminal offline TAD performance. The sole continuation is still
  `RUNNING` at exact commit
  `42dba3f90b37243e7965d18b6707e88e81bf7109`; four bounded one-retry AMP
  restores remain the only transient events. Loss non-finites and replay
  exhaustions remain zero, optimizer/scheduler/selector/EMA updates remain
  exactly matched, and no selection pointer, Traceback, OOM, or fail-closed
  receipt exists.

- 2026-07-27 01:05 +08:00: Archived and absorbed the 2026-07-26 DUCA
  multi-round joint review. The 18,959-byte/141-line raw archive is
  byte-identical to the attachment, SHA-256
  `67409BC9B140275BFC6804DD65FACBBEB568719304768A322FCF3A3F54576484`.
  Project verdict is
  `SUBSTANTIAL_ACCEPT_GOVERNANCE_AND_EXPERIMENT_DESIGN /
  ACCEPT_WITH_CURRENT_FACT_CORRECTIONS /
  REJECT_STALE_STATUS_AS_CURRENT_CONTRACT /
  HOLD_REVIEWER_PROPOSED_THRESHOLDS_UNTIL_RATIFIED`.
  Draft PR #2 is now open at exact head
  `42dba3f90b37243e7965d18b6707e88e81bf7109`, closing G0's read-surface
  portion but not the independent source adjudication. Public base and DUCA
  branches still contain plaintext proxy authentication in `README.md`,
  traceable to initial commit `7eb8a413`; redaction, rotation and history
  treatment are recorded as separate urgent actions without reproducing the
  secret. Stage-2 Job `1191957` remains running with no terminal epoch-59 EMA
  result, so paper claims remain HOLD. Reviewer-proposed thresholds/matrices
  and ChronoTransport parking were recorded as proposals rather than silently
  frozen.

- 2026-07-27 01:31 +08:00: Implemented and deployed the matched DUCA 25%
  sampling-rate curriculum. Branch
  `codex/duca-rate25-curriculum-20260727`, exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, changes the complete course from
  `K=384/768` to `K=192/768` while retaining Stage-2 `cls+reg` contribution
  distillation, `density_transport_st` detector gradient, full ASFormer
  adaptation, seed/schedules/evaluator and terminal-only checkpoint contract.
  Linux verification passed 18 curriculum/config tests and 23 C3 regressions.
  Precheck `1193418` exposed a zero-update launcher-order defect and is not
  model evidence; the smallest strict-order repair was committed, and
  corrected GPU precheck `1193433` completed `0:0`. Sole formal Job `1193437`
  is `RUNNING` on `g0015` from the clean exact snapshot and entered Stage-1
  epoch 0. Its root is
  `/data/run01/sczc063/yuzibo/projects/c3_lowres_action_probe/duca_rate25_ed0d490_formal_20260727_013015`.
  No Traceback/OOM/non-finite/fail-closed event was present at launch. Only
  terminal Stage-2 epoch-59 EMA OpenTAD official mAP is performance evidence;
  intermediate curves remain diagnostic and no duplicate job is permitted.

- 2026-07-27 01:37 +08:00: K=384 Stage-2 Job `1191957` completed epoch 54
  with exactly 4,500 successful optimizer/scheduler/selector/EMA updates.
  Its audit retains four accepted one-retry AMP restores, zero non-finite-loss
  attempts/replays/exhaustions and no selection pointer, Traceback, OOM or
  fail-closed receipt. K=192 Job `1193437` completed Stage-1 epoch 0 and
  entered epoch 1. Batch 17 required two bounded AMP overflow restores before
  succeeding; the epoch-0 terminal receipt is finite (`Loss=5.4105`,
  `cls/reg=0.8852/0.4453`, `5994MB`) with exact requested/effective
  `K=192/192` and all Stage-2-only policy/contribution/detector-gradient
  weights still zero. This is runtime-health evidence only. Neither job has
  produced terminal epoch-59 EMA official offline TAD mAP.

- 2026-07-27 02:08 +08:00: K=384 Stage-2 Job `1191957` completed epoch 56
  with exactly 4,700 successful optimizer/scheduler/selector/EMA updates and
  entered epoch 57. The one-based epoch-55 diagnostic EMA curve is `65.11%`
  Average-mAP (`79.99/75.71/68.05/57.90/43.88%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`), `-0.54pp` versus epoch 50; it remains
  learning-curve evidence only. Epoch 55 batch 37 added a fifth accepted
  one-retry AMP restore at scale `512`, while loss non-finites and replay
  exhaustions remain zero. K=192 Job `1193437` completed Stage-1 epoch 4 with
  finite `Loss=4.7960`, `cls/reg=0.4619/0.3513`, exact requested/effective
  `K=192/192`, and sealed `epoch_4.pth`; the scheduled one-based epoch-5
  diagnostic is in progress. It has three accepted AMP replay attempts across
  two batches and no Traceback/OOM/non-finite/fail-closed receipt. No
  selection pointer, Stage-2 K=192 evidence, or terminal epoch-59 EMA official
  offline TAD result exists in either arm.

- 2026-07-27 02:44 +08:00: K=384 Stage-2 Job `1191957` completed epoch 59,
  sealed `epoch_59.pth` with SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`,
  and closed the continuation audit at 5,000 successful post-e9 updates.
  Optimizer/scheduler/selector/EMA counters match exactly; there were five
  accepted one-retry AMP restores and zero loss non-finites or replay
  exhaustions. The training-loop final EMA diagnostic is `65.385724%`
  Average-mAP (`80.193191/75.662461/68.607247/58.581766/43.883956%` at
  tIoU `0.3/0.4/0.5/0.6/0.7`), provisionally about `+0.896pp` over matched
  exact-uniform `64.49%`, with `+1.312/+1.434pp` at tIoU `0.6/0.7`.
  The explicit OpenTAD terminal evaluator independently loaded
  `epoch_59.pth/state_dict_ema` and was still running, so this value is not
  yet promoted to the sealed terminal result. K=192 Job `1193437` completed
  its one-based epoch-5 EMA diagnostic at `8.730398%` Average-mAP
  (`19.988334/12.780593/6.844830/3.013880/1.024351%`) and entered Stage-1
  epoch 8. That early uniform-warmup curve cannot select a checkpoint; its
  AP/AUC/Brier/ECE and transition-boundary quality stage is deferred until the
  Stage-1 checkpoint set is sealed. Both exact source snapshots are clean,
  dependencies are null, selection pointers are absent, and error scans remain
  clear.

- 2026-07-27 03:14 +08:00: K=384 Job `1191957`'s explicit terminal OpenTAD
  evaluator completed all 211 videos and 422,000 predictions and independently
  reproduced `65.385724%` Average-mAP
  (`80.193191/75.662461/68.607247/58.581766/43.883956%`). The job then ended
  `FAILED/1:0` after metric computation because its config retained
  `post_processing.save_dict=False` while the requested structured
  `--metrics-json` receipt requires a saved final prediction file. This is a
  post-evaluation packaging failure, not a model, checkpoint, inference or
  metric failure. Submitted evaluation-only repair Job `1193610` at the same
  clean exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109`; it reads the
  sealed epoch-59 EMA checkpoint, enables only the allowed final prediction
  save, and must seal prediction/evaluation hashes plus
  `terminal_evaluation.json`. It does not rerun training, change the old exit
  code, select a checkpoint, or allow raw-prediction replay. K=192 formal Job
  `1193437` remains `RUNNING` and completed its one-based epoch-10 EMA
  exact-uniform warmup diagnostic at `27.82%` Average-mAP
  (`50.03/39.49/28.41/15.44/5.75%`), then entered Stage-1 epoch 10. That point
  remains diagnostic-only and cannot select a checkpoint or establish terminal
  K=192 performance.

- 2026-07-27 03:37 +08:00: Evaluation-only K=384 receipt Job `1193610`
  completed `0:0` and sealed the terminal epoch-59 EMA OpenTAD evidence at
  clean exact commit `42dba3f90b37243e7965d18b6707e88e81bf7109`. The receipt
  binds checkpoint SHA-256
  `848abe3deace90e03b7fb3bca993a223d6155c037d7fed11c7c8b1e80eac9a87`,
  211 videos, 422,000 predictions, prediction SHA-256
  `b7a26f270d0ed4e3f7036793dd4c48fe6011e7b15f2570525843ab0cfb7497f1`
  and evaluation SHA-256
  `d239a1be1f2eaff15d310a6ee8cceaa36b5d8f70ee3b3516d6cb44cd7e049b74`.
  Sealed official Average-mAP is `65.385724%`, with
  `80.193191/75.662461/68.607247/58.581766/43.883956%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, approximately `+0.896pp` over matched exact-uniform
  `64.49%` and `+1.312/+1.434pp` at tIoU `0.6/0.7`. Original Job `1191957`
  remains `FAILED/1:0` as the immutable post-metric packaging-failure record;
  training and checkpoint selection were not repeated. K=192 Job `1193437`
  remains healthy at exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2` and entered Stage-1 epoch 14.
  Its epoch-15 EMA diagnostic and post-Stage-1 AP/AUC/Brier/ECE/boundary
  exports are pending, with no Stage-2 initialization, selection pointer,
  Traceback, OOM, non-finite-loss failure or fail-closed receipt.

- 2026-07-27 04:07 +08:00: K=192 formal Job `1193437` remains healthy at
  clean exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, sealed its
  one-based epoch-15 EMA diagnostic from `epoch_14.pth/state_dict_ema`, and
  entered Stage-1 epoch 16. The exact-uniform warmup curve is `35.616501%`
  Average-mAP
  (`59.176347/48.839082/36.417352/22.762136/10.887588%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. This is
  diagnostic learning-curve evidence only and cannot select a checkpoint or
  establish terminal/learned-selector K=192 performance. The dependency
  remains null; no `best_validation_ema.json`, Stage-2 initialization,
  Traceback, OOM, non-finite-loss failure or fail-closed receipt exists.
  AP/AUC/Brier/ECE and transition-boundary quality exports remain deferred
  until Stage 1 seals all scheduled checkpoints.

- 2026-07-27 05:07 +08:00: K=192 formal Job `1193437` sealed its one-based
  epoch-20 EMA diagnostic from `epoch_19.pth/state_dict_ema` and entered
  Stage-1 epoch 23. The exact-uniform warmup curve is `43.132056%`
  Average-mAP
  (`65.227910/55.799303/44.794554/32.149065/17.689446%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. This is
  learning-curve evidence only and cannot select a checkpoint or establish
  learned-selector/terminal K=192 performance. The source remains clean at
  exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, the dependency is
  null, and there is no `best_validation_ema.json`, Stage-2 initialization,
  Traceback, OOM, non-finite-loss failure or fail-closed receipt.

- 2026-07-27 06:07 +08:00: K=192 formal Job `1193437` sealed its one-based
  epoch-25 EMA diagnostic from `epoch_24.pth/state_dict_ema` at `49.091036%`
  Average-mAP
  (`69.158801/61.778057/51.280304/38.465483/24.772537%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. It remains
  exact-uniform warmup learning-curve evidence only. All 30 Stage-1 training
  epochs then completed and sealed `epoch_29.pth` with SHA-256
  `141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`.
  The epoch-30 EMA diagnostic and AP/AUC/Brier/ECE/transition-boundary quality
  exports remain pending; strict Stage-2 initialization has not begun. The
  exact source is clean, dependency null, and no selection pointer,
  Traceback, OOM, non-finite-loss failure or fail-closed receipt exists.

- 2026-07-27 06:37 +08:00: K=192 Stage-1 one-based epoch-30 EMA diagnostic
  from sealed `epoch_29.pth/state_dict_ema` completed at `51.954148%`
  Average-mAP
  (`70.747919/64.400867/54.530413/41.835973/28.255569%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. This
  closes the exact-uniform warmup learning curve but is not the terminal
  full-course K=192 offline TAD result and cannot select a checkpoint. The
  launcher began the one-based epoch-5 selection-quality export over 211
  validation videos; no AP/AUC/Brier/ECE/boundary-support analyzer summary
  exists yet. Strict Stage-2 initialization remains pending, with no selection
  pointer, Traceback, OOM, non-finite-loss failure or fail-closed receipt.

- 2026-07-27 07:10 +08:00: K=192 formal Job `1193437` remains `RUNNING` at
  clean exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`.
  Stage-1 quality summaries for one-based epochs 5/10/15 are sealed over 211
  videos, 487 windows and 355,592 frame observations. Coarse macro AP is
  `0.332623/0.378387/0.406062`, macro AUC is
  `0.485973/0.542316/0.575196`, pooled AP is
  `0.302061/0.347483/0.363583`, pooled AUC is
  `0.488838/0.546702/0.567331`, Brier is
  `0.233573/0.222322/0.217861`, and ECE is
  `0.118097/0.060039/0.024695`. At epoch 15, transition macro policy AUPRC
  at `r0/r1/r2/r4/r8` is
  `0.031783/0.071428/0.110247/0.177410/0.287755`. The exact-uniform selected
  geometry is invariant and every paired learned-minus-uniform delta is zero
  by construction, so this is a matched-control sanity check rather than
  learned-selector evidence. GT is evaluation-only and there are zero budget
  or max-hole violations. Epoch-20 export is running; Stage-2 has not
  initialized, and no `best_validation_ema.json`, Traceback, OOM, non-finite
  event or fail-closed receipt exists.

- 2026-07-27 07:37 +08:00: K=192 Stage-1 quality summaries for one-based
  epochs 20 and 25 completed. Epoch-20 macro AP/AUC, pooled AP/AUC,
  Brier/ECE are
  `0.410136/0.578777`, `0.364094/0.570474`, and
  `0.216209/0.012080`; epoch-25 values are
  `0.410307/0.579620`, `0.367067/0.571312`, and
  `0.215715/0.021642`. Epoch-25 transition macro policy AUPRC at `r0/r8` is
  `0.031584/0.292148`. The small epoch-15-to-25 macro AP/AUC gain
  (`+0.004245/+0.004424`) shows a Stage-1 coarse-evidence plateau, not a
  checkpoint-selection or terminal offline TAD result. Exact-uniform
  geometry and all paired learned-minus-uniform deltas remain exactly
  matched. Epoch-30 export is processing at least 242/487 windows. Job
  `1193437` remains healthy with null dependency; repo HEAD is clean and
  matches manifest commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`. Stage-2 has not initialized,
  with no Traceback, OOM, runtime error, non-finite event,
  `best_validation_ema.json` or fail-closed receipt.

- 2026-07-27 08:07 +08:00: K=192 Stage-1 epoch-30 quality completed and
  sealed all six scheduled summaries. Epoch-30 macro AP/AUC is
  `0.418318/0.584329`, pooled AP/AUC is `0.372005/0.574455`, Brier/ECE is
  `0.215742/0.021818`, and transition macro policy AUPRC at `r0/r8` is
  `0.032568/0.294064`. Exact-uniform geometry and all paired deltas remain
  invariant, as required. Stage-2 started at `07:46:38` from strict full-model
  initialization of `epoch_29.pth/state_dict_ema`, epoch 29, SHA-256
  `141e4c1f3ce7b1b11a477fecf59478694055b8897102180137f007a825fe2595`,
  resetting only `frame_selector._loss_weight_schedule_step`. The artifact
  hash matches the configured and sealed Stage-1 binding. Through epoch 2,
  the audit records 300 successful optimizer, EMA, scheduler and DUCA
  schedule updates. One AMP attempt was restored and replayed successfully on
  retry 1/8; non-finite-loss attempts and replay exhaustions are zero. Epoch 3
  is running with finite losses and exact budget 192. Detector-gradient and
  teacher-utility weights remain zero only during the registered 2,100-step
  bridge delay before the 1,500-step ramp. Job `1193437`, its null dependency
  and clean exact source commit remain healthy; no Traceback, OOM, runtime
  error, non-finite loss, `best_validation_ema.json` or fail-closed receipt
  exists.

- 2026-07-27 08:37 +08:00: K=192 Stage-2 one-based epoch-5 intermediate EMA
  evaluation from `epoch_4.pth/state_dict_ema` completed at `53.426779%`
  Average-mAP
  (`71.631253/64.833503/56.501860/44.442697/29.724581%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. Checkpoint
  SHA-256 is
  `d611a51f50a889cf08048a303206e1a06db37403de94f0910350158588b09fd3`,
  and the structured intermediate receipt SHA-256 is
  `a56f83cc09bd5ac3f58c723a8439d91ea347d024b74e3ac7e734a0b60ac908e9`.
  This is `+1.472631pp` over the Stage-1 epoch-30 diagnostic, with
  `+2.606724pp` at tIoU 0.6, but remains non-selecting intermediate evidence.
  The epoch-4 audit records 500 successful optimizer, EMA, scheduler and DUCA
  schedule updates. Two AMP attempts recovered on retry 1/8; non-finite-loss
  attempts and replay exhaustions are zero. Epoch 5 is running. The direct
  detector bridge and contribution teacher remain in their registered delay,
  so this point does not isolate those terms or learned selection versus
  uniform. Job state, null dependency, clean exact source, budget and no-best
  contract remain healthy, with no hard failure signature.

- 2026-07-27 09:37 +08:00: K=192 Stage-2 one-based epoch-10 intermediate EMA
  evaluation from `epoch_9.pth/state_dict_ema` completed at `54.515667%`
  Average-mAP
  (`72.941354/65.978947/57.882904/45.013669/30.761460%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`). Checkpoint SHA-256 is
  `b5169d2949cc9a89ab0a286c1e61a6e567465be4b36cd4ca32e1534ac91ef63d`
  and receipt SHA-256 is
  `e91049677bca9f8685520c487f04872dd4bf9f708f4834ba6f2a5218c5fcf6bd`.
  This is `+1.088889pp` over epoch 5 and `+2.561519pp` over the Stage-1
  endpoint, but remains non-selecting intermediate evidence. At successful
  step 1,099, finite nonzero cls/reg contribution-distillation losses
  (`0.0115/0.0113`) and `duca_detector_grad_w=0.0015` confirm activation of
  contribution distillation and `density_transport_st`; the separate
  detector-utility weight remains zero. The audit reached 1,100 successful
  optimizer/EMA/scheduler/schedule updates, with the same two one-retry AMP
  recoveries and zero non-finite-loss attempts or replay exhaustion. Epoch 11
  is running; job, source, budget, no-best contract and hard-error scans remain
  healthy.

- 2026-07-27 10:16 +08:00: K=192 Stage-2 one-based epoch-15 intermediate EMA
  evaluation from `epoch_14.pth/state_dict_ema` completed at `55.403415%`
  Average-mAP
  (`73.319852/67.646858/58.673394/45.968481/31.408491%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. Checkpoint
  SHA-256 is
  `c56240b3181b5907555f07ff16c838b9a005e0ab2fe8169306dae009a269e94e`
  and receipt SHA-256 is
  `0465943b24adac7647e6a8232d40b7a275e176a91e16cb691b118b72d07873b6`.
  This is `+0.887748pp` over epoch 10 and `+3.449267pp` over the Stage-1
  endpoint, but remains non-selecting intermediate evidence. The epoch-14
  audit reached 1,500 successful optimizer/EMA/scheduler/schedule updates.
  A third isolated AMP skip recovered on retry 1/8, while non-finite-loss
  attempts and replay exhaustions remained zero. At step 1,499, cls/reg
  contribution-distillation losses were finite and nonzero
  (`0.7043/0.6853`), `duca_detector_grad_w=0.0365`, and requested/effective
  budget remained `192/192`; the detector-utility weight was still zero.
  Epoch 15 started with the job, clean exact source, null dependency,
  no-best-checkpoint contract and hard-error scans healthy.

- 2026-07-27 11:07 +08:00: K=192 Stage-2 one-based epoch-20 intermediate EMA
  evaluation from `epoch_19.pth/state_dict_ema` completed at `56.050489%`
  Average-mAP
  (`73.483220/67.033731/58.709834/47.921247/33.104414%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`) over 211 videos and 422,000 predictions. Checkpoint
  SHA-256 is
  `5e213343bd5f2f4994d47ca7b042c7c41f9e40a4705d096b133333ee454ae276`
  and receipt SHA-256 is
  `39e743dcc3585b0cd2fa05e97dc0a6e109bca490d110a777a17706403f909e01`.
  This is `+0.647074pp` over epoch 15 and `+4.096341pp` over the Stage-1
  endpoint. The gain is concentrated at tIoU 0.6/0.7
  (`+1.952766/+1.695923pp`), while tIoU 0.4 decreased by `0.613127pp`.
  The result remains non-selecting intermediate evidence. The epoch-19 audit
  reached 2,000 successful optimizer/EMA/scheduler/schedule updates. Four
  isolated AMP skips each recovered on retry 1/8, with zero non-finite-loss
  attempts or replay exhaustion. At step 1,999, cls/reg contribution losses
  were finite (`2.5477/2.5470`), `duca_detector_grad_w=0.1248`, and the
  requested/effective budget was `192/192`; detector utility remained zero.
  Total loss was finite at `7.2839` and remains under trend monitoring.
  Epoch 20 started with job, source, dependency, no-best contract and
  hard-error scans healthy.

- 2026-07-27 12:19 +08:00: K=192 formal Job `1193437` ended `FAILED/1:0`
  at `11:43:54`, after Stage-2 epoch 24 completed and
  `checkpoint/epoch_24.pth` was saved. Its one-based epoch-25 non-selecting
  EMA evaluation stopped at 147/396 batches when a Decord worker exhausted
  10,240 EOF retries while retrieving final video frames. No epoch-25 mAP
  receipt exists and no update occurred after the checkpoint. The sealed
  epoch-24 checkpoint SHA-256 is
  `d37cad6e1fcbf9078f9e186c0735f291461332572df67ef6df16ab05db3c00f6`;
  it contains model/EMA, optimizer, scheduler, GradScaler and epoch state at
  scheduler and selector step 2,500. Source audit SHA-256
  `ec1536dfd68d16144e242c8ca7ee10828b5de05d5fea4f26b968e21b7a1dcf9d`
  records 2,500 successful updates, 2,506 optimizer attempts, six bounded AMP
  skips, five replayed batches, maximum retry 2/8, and zero non-finite-loss
  attempts or replay exhaustion. Thus the terminal state is a decoder failure
  during diagnostic evaluation, not OOM, non-finite training, model failure
  or terminal K=192 performance.

- 2026-07-27 12:19 +08:00: minimal K=192 epoch-24 recovery precheck Job
  `1194469` completed `0:0`. Recovery manifest SHA-256 is
  `44a545805e11051a80c91f834292915efee2c870a9d0c54265b24d097c9a8d75`;
  launcher SHA-256 is
  `66353a213053bc0981bf349734ee1e699c9e312722e2c8a36f5f620e481229bf`.
  Formal recovery Job `1194471` started at `12:19:03` with null dependency,
  the same clean exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, strict state loading and a
  fresh run root. The only runtime repair is
  `DECORD_EOF_RETRY_MAX=20480`. It first retries the missing epoch-25 EMA
  evaluation before any new update; only success unlocks the separately
  audited 3,500-update continuation through epoch 59. Source plus continuation
  must equal 6,000 successful updates, and terminal evidence remains
  epoch-59 EMA official OpenTAD mAP. The source checkpoint has no global RNG
  state: model/EMA/optimizer/scheduler/GradScaler/DUCA schedule restoration is
  exact, while random-stream continuity is not bit-exact. This limitation is
  sealed as protocol provenance and is not model evidence.

- 2026-07-27 12:26:54 +08:00: K=192 recovery Job `1194471` reached 169/396
  batches in the missing epoch-25 EMA evaluation, crossing the original
  147/396 Decord failure position with no Decord error, Traceback, OOM or
  non-finite evidence. No continuation work directory or new training audit
  exists yet, so the pre-update evaluation gate is still enforced. This is
  decoder-repair evidence only, not an mAP result or completed evaluation.

- 2026-07-27 12:41 +08:00: K=192 recovery Job `1194471` completed all 396
  batches of the missing one-based epoch-25 EMA evaluation from sealed
  `epoch_24.pth/state_dict_ema`. Official OpenTAD Average-mAP is
  `56.646995%`, with
  `73.566280/67.812407/59.759908/48.284381/33.811996%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions.
  Prediction SHA-256 is
  `17c9fce0f909eed7d08e82ba3cd133c68bf681c75635b3e2edeb946c1674d422`;
  structured receipt SHA-256 is
  `e9069dc8d621268014f32761759f3a590f2ff9d85ac822a9ec09425423894638`.
  This is `+0.596506pp` over the epoch-20 intermediate point and
  `+4.692847pp` over the Stage-1 endpoint. Per-threshold epoch-20 deltas are
  `+0.083060/+0.778676/+1.050074/+0.363134/+0.707582pp`. It remains a
  non-selecting learning-curve point, not terminal K=192 offline TAD
  performance or learned-selector evidence against a matched uniform
  endpoint.

- 2026-07-27 12:41 +08:00: the evaluate-before-update gate for Job `1194471`
  passed at `12:37:51`. Only then did strict epoch-24 recovery start epoch 25
  training at `12:38:28`, restoring model/EMA, optimizer, scheduler,
  GradScaler and DUCA schedule while retaining the sealed global-RNG
  continuity limitation. The job remains `RUNNING` with null dependency and
  no Decord error, Traceback, OOM, non-finite evidence or
  `best_validation_ema.json`.

- 2026-07-27 12:45 +08:00: Job `1194471` completed continuation epoch 25 and
  entered epoch 26. The first 100 resumed updates are finite and move the
  restored selector schedule through step 2,599. Final epoch-25 total loss is
  `11.6996`; cls/reg losses are `0.2304/0.2485`; active cls/reg contribution
  losses are `5.1453/5.0894`; `duca_detector_grad_w=0.2260`; and the
  requested/effective budget is exactly `192/192`. There is no hard error or
  replay exhaustion. The large but finite contribution terms remain a trend
  diagnostic, not a non-finite failure.

- 2026-07-27 13:07 +08:00: K=192 recovery Job `1194471` completed
  continuation epochs 25--29 and sealed `epoch_29.pth`, SHA-256
  `4ec40e031af6087ff4db509df333e62da3514440d55d3615907cdd8ec2acd2dc`.
  Checkpoint state is epoch 29 with scheduler and selector step 3,000 and
  finite GradScaler scale 1,024. Continuation audit SHA-256
  `fc9860d56fc80980f1bdedd050ebf390f15644daf5f16f7b4fbb29576f1f81f4`
  records 500 successful optimizer/EMA/scheduler/DUCA-schedule updates, 500
  optimizer attempts, zero AMP skips, zero non-finite losses and zero replay
  exhaustion. At the final epoch-29 batch, finite total, cls/reg and cls/reg
  contribution losses are `12.4112`, `0.2235/0.2378` and
  `5.6252/5.5110`; schedule progress is `1.0000`,
  `duca_detector_grad_w=0.2500`, and budget is `192/192`.

- 2026-07-27 13:07 +08:00: recovery logs contain isolated effective-budget
  means `128.5` and `153.0` while requested budget remains 192. Exact code
  uses `effective_k=min(K, valid_len)`, and the immutable source run contains
  the same pattern, so these are short-valid-sequence caps rather than global
  budget drift. The one-based epoch-30 EMA diagnostic is running from the
  sealed epoch-29 checkpoint. No new mAP, `best_validation_ema.json`, Decord
  error, Traceback, OOM, non-finite event or fail-closed receipt exists.

- 2026-07-27 13:38 +08:00: K=192 recovery Job `1194471` completed the
  non-selecting one-based epoch-30 EMA evaluation from sealed
  `epoch_29.pth/state_dict_ema`. Official OpenTAD Average-mAP is
  `57.464558%`, with
  `74.527791/69.177787/60.361168/49.020347/34.235695%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions.
  Intermediate-evaluation JSON SHA-256 is
  `30fdb7579505e04619f932a0673701ed6733152540ad7092733b4c225750ea39`.
  This is `+0.817563pp` over epoch 25 and `+5.510410pp` over the Stage-1
  endpoint. The apparent `-7.921166pp` gap to sealed K=384 terminal
  performance is intermediate-to-terminal only. This point cannot select a
  checkpoint, establish terminal K=192 offline TAD performance, or isolate
  learned selection without a matched K=192 uniform endpoint.

- 2026-07-27 13:38 +08:00: after the epoch-30 diagnostic, Job `1194471`
  completed continuation epochs 30--31 and entered epoch 32. Audit snapshot
  SHA-256
  `8f33d67200fcacb8bd0a91c53d0cd4b479a8d7e35b7e7449403c69f70fb15f80`
  records 700 successful optimizer/EMA/scheduler/DUCA-schedule updates from
  700 attempts, with zero AMP skips, non-finite losses, replay restorations
  or replay exhaustion. Final epoch-31 total, cls/reg and cls/reg
  contribution losses are finite at `12.2577`, `0.2188/0.2354` and
  `5.4511/5.5401`; detector-gradient weight is `0.2500`, schedule progress is
  `1.0000`, and budget is `192/192`. The job remains `RUNNING` with null
  dependency and no Decord error, Traceback, OOM, hard non-finite event,
  `FAIL` receipt or `best_validation_ema.json`; the global-RNG continuity
  limitation remains sealed.

- 2026-07-27 14:12 +08:00: K=192 recovery Job `1194471` completed
  continuation epochs 32--34, sealed `epoch_34.pth` with SHA-256
  `2da7de83c0e9feb1f3b267deed7a41593680305b2491d47d4352a81254ac4a02`,
  and completed the non-selecting one-based epoch-35 EMA evaluation.
  Official OpenTAD Average-mAP is `57.921948%`, with
  `74.668088/69.304388/61.149872/49.416606/35.070788%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions.
  Intermediate-evaluation JSON SHA-256 is
  `00d22c18c188738d63b9f065a912ba08dfb74cf5d78df7e24d2fabe48d1784cc`.
  This is `+0.457390pp` over epoch 30 and `+5.967800pp` over the Stage-1
  endpoint; the tIoU 0.5/0.7 gains are `+0.788703/+0.835092pp`. The apparent
  `-7.463776pp` gap to sealed K=384 terminal performance remains
  intermediate-to-terminal. This point cannot select a checkpoint,
  establish terminal K=192 offline TAD performance, or isolate learned
  selection without a matched K=192 uniform endpoint.

- 2026-07-27 14:12 +08:00: the K=192 epoch-34 continuation audit records
  exactly 1,000 successful optimizer/EMA/scheduler/DUCA-schedule updates and
  1,000 attempted batches. One epoch-33 batch incurred three bounded
  AMP-overflow skips at replay attempts 1--3/8, with exact state restoration
  and scale reduction from 512 to 128 before successful completion.
  Non-finite-loss attempts, loss replays and replay exhaustion remain zero;
  this is an accepted finite transient under the bounded AMP contract, not
  model-failure evidence. Job `1194471` entered epoch 35 at `14:11:11`, is
  running from a clean detached snapshot at exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2` with null dependency, and has no
  Decord error, Traceback, OOM, hard non-finite event, `FAIL` receipt or
  `best_validation_ema.json`. The sealed global-RNG continuity limitation
  still applies.

- 2026-07-27 15:08 +08:00: K=192 recovery Job `1194471` completed
  continuation epochs 35--39, sealed `epoch_39.pth` with SHA-256
  `c1a9c6393920f1189000800362564239d5b9ee38ef94c00373f6fe6551b1f445`,
  and completed the non-selecting one-based epoch-40 EMA evaluation.
  Official OpenTAD Average-mAP is `58.116412%`, with
  `73.838536/69.371290/61.536349/50.339880/35.496002%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions.
  Intermediate-evaluation JSON SHA-256 is
  `5ef548dd4ab4cb8f7bdadc5938ce3670d1c075620a35279df39b99cc953f547e`.
  This is `+0.194464pp` over epoch 35 and `+6.162264pp` over the Stage-1
  endpoint. Relative to epoch 35, tIoU 0.3 falls `0.829552pp`, while tIoU
  0.4--0.7 gain `0.066902/0.386477/0.923274/0.425214pp`. The apparent
  `-7.269312pp` gap to sealed K=384 terminal performance remains
  intermediate-to-terminal. This point cannot select a checkpoint,
  establish terminal K=192 offline TAD performance, or isolate learned
  selection without a matched K=192 uniform endpoint.

- 2026-07-27 15:08 +08:00: epoch-40 continuation audit SHA-256
  `8bb3290894c0d3642a5326149313272c1357e862445468dcce3dd344a2cd21ee`
  records exactly 1,600 successful optimizer/EMA/scheduler/DUCA-schedule
  updates from 1,600 attempted batches and 1,603 optimizer attempts. The only
  three AMP skips/restorations remain the previously sealed single epoch-33
  replayed batch; non-finite-loss attempts and all replay exhaustions remain
  zero. Final epoch-40 total, cls/reg and cls/reg contribution losses are
  finite at `12.3632`, `0.1950/0.2267` and `5.5061/5.6228`; detector-gradient
  weight and schedule progress are `0.2500/1.0000`. The `172/192`
  effective/requested budget is a known short-valid-sequence cap. By
  `15:11 +08:00`, Job `1194471` completed epoch 41 and entered epoch 42 from
  the clean exact commit with null dependency, no hard failure or
  `best_validation_ema.json`; the global-RNG continuity limitation remains
  sealed.

- 2026-07-27 15:37 +08:00: K=192 recovery Job `1194471` completed
  continuation epochs 40--44 and sealed `epoch_44.pth`, SHA-256
  `fabe373abd4e7f2f982bf6a6fad26e7d022930ea4da8789d400f613200c8c9ea`.
  Continuation-audit SHA-256
  `6991f412001928853a3976baa07dc5c2b4bb46c288cc9788e3ad2c4bb2219413`
  records exactly 2,000 successful optimizer/EMA/scheduler/DUCA-schedule
  updates from 2,000 attempted batches and 2,003 optimizer attempts. The only
  three AMP skips/restorations remain the previously sealed single epoch-33
  replayed batch; non-finite-loss attempts and all replay exhaustions remain
  zero. The non-selecting epoch-45 EMA evaluation is in progress and has no
  metric JSON yet. The job remains running from a clean exact commit with null
  dependency, no hard failure or `best_validation_ema.json`; epoch 40 remains
  the latest completed mAP point.

- 2026-07-27 15:44 +08:00: K=192 recovery Job `1194471` completed the
  non-selecting one-based epoch-45 EMA evaluation from sealed
  `epoch_44.pth/state_dict_ema`. Official OpenTAD Average-mAP is
  `57.877041%`, with
  `74.416763/68.892371/61.103833/49.558953/35.413287%` at tIoU
  `0.3/0.4/0.5/0.6/0.7`, over 211 videos and 422,000 predictions.
  Checkpoint SHA-256 is
  `fabe373abd4e7f2f982bf6a6fad26e7d022930ea4da8789d400f613200c8c9ea`;
  intermediate-evaluation JSON SHA-256 is
  `4e13c794f1cf5ad709081e227833abf422be245f102ce97feb7a9f2f2b902670`.
  The point is `-0.239371pp` versus epoch 40 and `+5.922893pp` over the
  Stage-1 endpoint. Per-threshold epoch-40 changes are
  `+0.578227/-0.478919/-0.432516/-0.780927/-0.082715pp`. It remains
  non-selecting intermediate evidence, not terminal K=192 offline TAD
  performance or learned-selector evidence against a missing matched K=192
  uniform endpoint.

- 2026-07-27 15:44 +08:00: the epoch-45 evaluation completed without the
  prior Decord failure and Job `1194471` entered continuation epoch 45 at
  `15:43:49`. The sealed epoch-44 audit remains 2,000 successful updates with
  three bounded AMP restores from one epoch-33 batch, zero non-finite-loss
  attempts and zero replay exhaustion. The job is running with null
  dependency from exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`; no intermediate checkpoint
  selection is permitted, and the sealed global-RNG continuity limitation
  remains in force.

- 2026-07-27 16:00 +08:00: re-read the locally preserved multi-round and Pro
  review originals, current exact K=384/K=192 configs, official-derived model
  surfaces and sealed receipts. Corrected the K=384 claim: Stage 1 is 30
  epochs of full-model training and Stage 2 is 60 more, so `65.385724%` is a
  valid 90-epoch over-budget curriculum candidate rather than a fair
  official-60 final result. Stage-2 epoch 50 `65.650497%` is the best observed
  diagnostic but still costs 80 total epochs. Re-prioritized the CVPR route
  around the source reviews' shared bottleneck: nonuniform observations are
  still processed under an equal selected-rank time metric. Designed, but did
  not implement, a physical-time budget-transport candidate with exact-K
  inverse-CDF sampling, timestamp/gap conditioning and physical-coordinate
  assignment/regression. Restricted immediate experiments to fair total-60
  schedule tests, a 90-epoch uniform control, and selected-axis versus
  physical-time uniform/DUCA comparisons at K=384 and K=192.

- 2026-07-27 16:25 +08:00: completed a second source-level synthesis of the
  locally preserved DUCA multi-round/Pro originals and corrected the prior
  physical-head priority. The original curriculum is recovered as a
  detector-free frontend P0 followed by one 6,000-update official-60 detector
  course with uniform warmup and staged mechanism release, not the current
  30+60 full-model course. Audited the upstream AdaTAD table: VideoMAE-S
  `768/160` reports `69.03%`; current local dense `~68.29%` is not yet an
  equivalent reproduction, and K=384 exact-uniform `64.49%` is a DUCA
  selected-axis wrapper control rather than a clean native official
  half-rate baseline. Restored the main method identity to a detector-agnostic
  pre-backbone exact-K monotone temporal-acquisition plugin; detector timestamp
  injection and physical-coordinate head changes are now diagnostic/enhanced
  variants only. Preserved target-domain train-free negative evidence and
  explicitly separated it from the still-untested frozen-detector
  plug-and-play contract. Added
  `research-wiki/duca_prebackbone_plugin_and_baseline_recovery_contract.md`,
  Decision 26, query-pack pointers and anti-repetition guards. No model code or
  new experiment was launched in this design-recovery pass.

- 2026-07-27 16:40 +08:00: closed the remaining memory conflict. Marked the
  old `duca_final_model_contract.md` as a superseded historical R0--R5 contract
  and pointed it to the pure-pre-backbone/baseline-recovery contract. Corrected
  current-course parity wording: at exact commit `42dba3f9`, nominal detector
  settings remain official-derived, but active `ActionFormer` and
  `AnchorFreeHead` source files are extended and are not byte-identical to
  upstream. This strengthens the requirement for clean native official dense,
  K=384 and K=192 baselines before any selector-gain claim.

- 2026-07-27 17:20 +08:00: user approved the total-60 pure pre-backbone DUCA
  experiment. Wrote the canonical design and experiment node. Clarified that
  a one-frame remove/add swap is necessary for local finite-difference
  fidelity but insufficient for policy-level evidence. The detector-gradient
  gate now also requires dispersed 1%/5%/10% multi-frame swaps, contiguous
  block swaps and full hard decoding after 0.25/0.5/1.0 density steps. Direct
  detector gradient is excluded from the final arm if either local or
  multi-frame alignment fails. No model code or new long training was started
  in this specification pass.

- 2026-07-27 17:21 +08:00: K=192 recovery Job `1194471` completed the
  non-selecting one-based epoch-50 EMA diagnostic at `58.383005%`
  Average-mAP
  (`74.477950/69.245107/61.193389/50.712331/36.286249%` at tIoU
  `0.3--0.7`) and the epoch-55 diagnostic at `58.082562%`
  (`73.934313/69.065870/61.031079/49.908786/36.472761%`). Epoch 55 is
  `-0.300443pp` versus epoch 50 and `+6.128414pp` over Stage-1. Sealed
  checkpoint SHA-256 values are
  `90d6c4cb791a1b908c5c4cfcf2123a1c3aac3a721498ce26b2b18a819d708161`
  for epoch 49 and
  `59d24803e63efd2d0177e5d5baf106fc4dd0c01a737c2053740b1c3148177fcc`
  for epoch 54; receipt SHA-256 values are
  `6ca0f449398bf75686921b38e7767a697b6cacc97b23b83e41e57ab1554f1a22`
  and
  `77cdc89b79d4a460a7024f71c914739b153e33c927c588ccc19f76ec3b8867cb`.

- 2026-07-27 17:21 +08:00: the epoch-54 continuation audit SHA-256
  `a42d6e162b96b5fa6cb96c56eff8144be3911db3a56292106f65a2d8054365be`
  records exactly 3,000 successful continuation updates, hence 5,500/6,000
  combined Stage-2 updates. The only three continuation AMP
  skips/restorations remain the single epoch-33 batch; non-finite-loss
  attempts and all replay exhaustions remain zero. Job `1194471` entered
  epoch 55 at `17:16:00`, remains running with null dependency from clean
  exact commit `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, and has no Decord
  error, Traceback, OOM, hard non-finite event, `FAIL` receipt or
  `best_validation_ema.json`. Epoch 50/55 remain non-selecting diagnostics;
  the registered terminal evidence is epoch-59 EMA only.

- 2026-07-27 17:49 +08:00: K=192 recovery Job `1194471` completed Stage-2
  epochs 55--59 and exactly 6,000/6,000 combined successful updates. Sealed
  `epoch_59.pth` SHA-256 is
  `4a5389506263b8fd76ca3de6ce3475dee64cc0d9ed1ca73c896692c8db288455`;
  continuation audit SHA-256 is
  `36fc64d4542ce671b4c891f8b8270a51b629ad4b514165f80fd66b439a7451f0`
  and records 3,500/3,500 successful continuation updates, three bounded AMP
  restores on one epoch-33 batch, zero non-finite-loss attempts and zero replay
  exhaustion. Final epoch-59 total, cls/reg and cls/reg contribution losses are
  finite at `12.4513`, `0.1843/0.2128` and `5.5674/5.6760`. The predeclared
  terminal EMA evaluation is running and had processed 126/396 batches at
  `17:49:28`; no terminal mAP receipt exists yet. Job `1194471` remains
  `RUNNING` with null dependency from exact commit
  `ed0d4900bffe3546997ea1f00ae806d82cad55f2`, with no Decord error,
  Traceback, OOM, hard non-finite event, `FAIL` receipt or
  `best_validation_ema.json`.

- 2026-07-27 18:20 +08:00: the explicit terminal OpenTAD evaluator loaded
  K=192 `epoch_59.pth/state_dict_ema` and exactly reproduced the automatic
  one-based epoch-60 EMA diagnostic. Sealed official Average-mAP is
  `57.967272%`, with
  `73.907179/68.926135/61.194230/49.841145/35.967670%` at tIoU
  `0.3--0.7`, over 211 validation videos and 422,000 predictions. Terminal
  receipt SHA-256 is
  `febc59d463476bcf6a1a0d77f237a54f12c59ce3028ce623fba9844c07fada04`;
  prediction SHA-256 is
  `719ba43b0f76f5647b2394a23b622aff0bac0c17a54d15613c4d9dbdb57d02d0`;
  combined 6,000-update audit SHA-256 is
  `f13ef3f8650c4fe75f795ddf255b32a377ea4b7c31d4bdc007f0121114ba97a1`.
  The result is `+6.013124pp` over the Stage-1 endpoint and
  `-7.418452pp` versus the sealed K=384 30+60 endpoint, but neither
  comparison identifies learned-selection gain. This course consumed 90
  full-model epochs, lacks a matched native K=192 uniform terminal control,
  and retains the documented non-bit-exact global-RNG recovery limitation.
  It is terminal only for this historical over-budget course, not a fair
  total-60 paper result or evidence that learned selection beats uniform at
  25%. At `18:24`, Job `1194471` remained running only for post-terminal
  selector-quality export; formal stderr was empty, with no Decord error,
  Traceback, OOM, hard non-finite event, replay exhaustion, `FAIL` receipt or
  `best_validation_ema.json`.

- 2026-07-27 18:46 +08:00: K=192 post-terminal Stage-2 epoch-5 and epoch-10
  selection-quality summaries completed over 487 windows and 211 videos.
  Summary SHA-256 values are
  `fcbeb7b77f3dc574639261399848baaf6e0172809a3d924d443d390529c01864`
  and
  `c6677d757340c8fc721fa289743f638087e2291ad3b812eca5c9c92e6c7d910d`;
  both record zero budget or max-hole contract violations. Coarse macro
  AUPRC/AUROC moves only from `0.421082/0.585920` to
  `0.425132/0.590137`, while pooled Brier/ECE moves from
  `0.215769/0.027737` to `0.216125/0.035281`. At epoch 10, learned minus
  matched uniform action enrichment is `+0.014851`, boundary recall at radii
  0/1 is `-0.006059/+0.013539`, R2Q3 bilateral endpoint recall is
  `+0.014461`, and R4Q5 bilateral endpoint recall is `-0.026231`; mean
  endpoint distance improves only `0.007479` and maximum hole worsens
  `0.030801`. The quality evidence is therefore mixed and does not support a
  strong learned-selector geometry advantage. It remains explanatory only and
  cannot replace the missing full-model matched-uniform K=192 terminal
  control. At `18:47`, Job `1194471` remained running for epoch-15 quality
  export, with empty formal stderr, no hard failure and no
  `best_validation_ema.json`.

- 2026-07-27 19:10 +08:00: K=192 post-terminal Stage-2 epoch-15/20
  selection-quality summaries completed with SHA-256
  `a8b0917ad46cac2921b2534318cec1eb89a2d62a1f9ce4d5bd299a44d1fb824e`
  and
  `3de85f7579650c007d23af1d22b73e461ab03226478fdd4c315a10c45eb7f757`.
  Both cover 487 windows/211 videos with zero budget or max-hole contract
  violations. At epoch 20, coarse macro AUPRC/AUROC reaches
  `0.437216/0.599782`, while pooled Brier/ECE is
  `0.216509/0.050933`: discrimination rises slowly as calibration worsens.
  Learned minus matched uniform radius-1 boundary recall is
  `+0.017566` (95% CI `[0.001394, 0.034104]`), but exact-radius recall,
  R2Q3 bilateral support and endpoint distance have zero-crossing intervals.
  R4Q5 bilateral support is significantly worse by `-0.026594`
  (`[-0.033483, -0.021020]`), and maximum hole worsens `0.121150`.
  This isolates a narrow local-support gain coupled to degraded wider paired
  boundary support; it is explanatory evidence for weak high-tIoU behavior,
  not a causal plugin attribution. At `19:17`, Job `1194471` remained running
  for epoch-25 quality export with empty formal stderr and no new hard failure.

- 2026-07-27 19:46 +08:00: K=192 post-terminal Stage-2 epoch-25/30/35
  quality summaries completed with SHA-256
  `cfab3759813577fe1187d9f2a6c8340642c991f92577523eff8409d3ac5e8af6`,
  `719da69f91bcab1decfac2dfe47600fa7d5f014151f4d51a505e4f59e0589f0a`,
  and
  `284f91c9dde2c01b8782a9ea6d3581ea07aa61eed10fc34556bc7ce0b0934ef6`.
  At epoch 35, coarse macro AUPRC/AUROC reaches
  `0.441664/0.605025`; learned-minus-uniform action enrichment is
  `+0.024621`. Radius-1 boundary recall improves `+0.031290`
  (`[0.013792, 0.053063]`) and endpoint distance improves `0.047062`
  (`[0.013641, 0.078358]`), while exact-radius recall and R2Q3 bilateral
  support remain inconclusive. R4Q5 bilateral support worsens
  `-0.045869` (`[-0.053979, -0.037741]`) and maximum hole worsens
  `0.501027`. Training is therefore producing stronger action enrichment and
  narrow local boundary proximity at the cost of progressively weaker paired
  wide-boundary protection. This is model-diagnostic evidence consistent with
  weak high-tIoU localization, not a causal attribution. At `19:47`, Job
  `1194471` remained running after the epoch-35 export with empty stderr, no
  hard error and no best-checkpoint pointer.

- 2026-07-27 20:14 +08:00: fully read and byte-identically archived the
  886-line total-60 pre-backbone Pro review at
  `docs/methods/reviews/2026-07-27-duca-total60-prebackbone-pro-review-raw.txt`
  (52,824 bytes, SHA-256
  `D493FD3497D412B3B873940447F1C743F3A1A50418EBCFC20B9FCE16945A4E11`).
  Issued `major_revision_accepted_with_corrections`, not blanket acceptance,
  in
  `docs/methods/2026-07-27-duca-total60-prebackbone-pro-review-absorption.md`.
  Accepted clean official/native uniform and wrapper parity, one unique
  exact-K bounded `e -> p -> F -> y -> S` contract, symmetric warped-time I/O,
  separate `G_rank/G_direct`, video-level statistics, development-seed
  exclusion and separate task-adapted/train-free claims. Clarified that raw
  proposals must be inverse-mapped to physical time before unchanged official
  NMS because nonlinear warps do not preserve IoU.

- 2026-07-27 20:14 +08:00: held reviewer-proposed density/DP constants, the
  exact RDD formula, publication deltas and cost ratios as
  `designed_reviewer_proposal`; retained conditional checkpoint selection
  when a strict training-side held-out set exists, otherwise terminal EMA.
  A4 short forks are development-only and cannot exceed the common 6,000
  updates in the final table. PR #3 was independently rechecked as
  `OPEN/DRAFT`, 133 changed files, 27,627 additions, 116 deletions and 28
  commits; cleanup is required for truthful review but is not a model-science
  blocker. Corrected the review's stale K192 evidence grade using the sealed
  `57.967272%` terminal result, while preserving its 90-epoch/no-clean-uniform
  limitation. Updated the canonical total-60 design, experiment node,
  pre-backbone contract, Decision 27, query pack, anti-repetition guards and
  source registry. No model code, checkpoint or new long training was created
  in this review-absorption pass.

- 2026-07-27 20:21 +08:00: K=192 post-terminal Stage-2 epoch-40/45
  selection-quality summaries completed with SHA-256
  `a9d361a4f0c7d9f56095bfc19210ce979e3652db9262a23c2a3ebd26619dfa50`
  and
  `224f87458dceab37276142d08c3f92dae36674bb2f466f7e1d0810af86c17994`.
  Coarse macro AUPRC/AUROC reaches `0.443028/0.605677` and
  `0.443816/0.606281`; action enrichment versus matched uniform remains
  `+0.025338/+0.025265`. Radius-1 boundary recall and endpoint distance
  retain small significant improvements, but exact-radius and R2Q3
  bilateral support remain inconclusive. R4Q5 bilateral support remains
  significantly worse at `-0.046030/-0.046575`, and maximum hole worsens
  `0.542094/0.570842`. This confirms the late-epoch mechanism pattern of
  narrow local support coupled to degraded broad paired-boundary protection.
  Job `1194471` remained `RUNNING` solely for epoch-50 quality export at the
  exact clean commit, with null dependency, empty stderr, no hard failure and
  no best-checkpoint pointer.

- 2026-07-27 20:24 +08:00: K=192 post-terminal Stage-2 epoch-50
  selection-quality summary completed with SHA-256
  `cdb63e8babcd239967b68dd95818c2c4e3fc0d4d865d340c0bc25afcfcc1c2a4`.
  Coarse macro AUPRC/AUROC is `0.444491/0.606776`, pooled Brier/ECE is
  `0.215020/0.047492`, and action enrichment is `+0.024000`. Radius-1
  boundary recall remains significantly positive at `+0.040996`, while
  exact-radius and R2Q3 bilateral changes remain inconclusive. R4Q5
  bilateral support remains significantly worse by `-0.043515`, and
  maximum hole worsens `0.579055`. The terminal diagnosis is unchanged:
  modest local boundary proximity is coupled to degraded wide paired
  boundary support.

- 2026-07-27 20:48 +08:00: recorded the reviewer-defense theory-closure
  discussion as `discussed / not_frozen`. The scientific center is narrowed
  to localization-preserving bounded temporal transport, hard
  counterfactual-benefit alignment and paired-boundary protection, rather
  than generic inverse-CDF sampling. Audited a material implementation gap:
  the current generic AdaTAD path performs NMS before selected-axis inverse
  mapping, whereas the canonical pure-plugin contract requires inverse
  mapping raw proposals to physical time before unchanged official NMS.
  Classified GT remapping as an external invertible coordinate adapter but
  still a disclosed training-path change. Kept `G_rank` and `G_direct`
  separate, left classification/regression weighting, geometry constants,
  dynamic-K risk control and the train-free evidence model unresolved, and
  recommended one focused Pro theory-closure review after a written memo.
  No model code or long experiment was started.

- 2026-07-27 20:54 +08:00: K=192 recovery Job `1194471` reached final
  Slurm state `COMPLETED`, exit `0:0`, at `20:47:16` after `08:28:13`.
  Epoch-55/60 quality summaries completed with SHA-256
  `f74cb521dd3c16af3bb6fc42a476f9ead8e69deee9fcac31999149ce5877e19f`
  and
  `fb53b13243235be945e30fd9b2b9bede7cfb2f3558b80c763d83f84c574fc2e3`.
  Terminal epoch-60 coarse macro AUPRC/AUROC is
  `0.445240/0.607663`, pooled Brier/ECE is `0.214704/0.044528`, and
  action enrichment versus matched uniform is `+0.024107`. Radius-1
  boundary recall remains significantly positive at `+0.036524` and
  endpoint distance improves `0.040021`; exact-radius and R2Q3 bilateral
  intervals still cross zero. R4Q5 bilateral support remains significantly
  worse by `-0.041925`, while maximum hole worsens `0.587269`. All 12
  scheduled quality checkpoints now exist. Exact commit and sealed terminal
  `57.967272%` mAP/hashes remain unchanged; combined updates remain exactly
  6,000 with zero non-finite-loss attempts or replay exhaustion. Final scans
  found no Traceback, OOM, Decord failure, hard `FAIL` or
  `best_validation_ema.json`. The recovery's non-bit-exact global-RNG
  limitation and over-budget/no-clean-uniform evidence boundary remain.

- 2026-07-27 21:01 +08:00: recorded the user's rejection of fixed-K-first
  as the default paper structure. Dynamic K is now a required candidate main
  innovation pending binding Pro adjudication, not an automatically accepted
  claim. Created
  `docs/methods/prompts/2026-07-27-duca-dynamic-k-adaptok-pro-adjudication-prompt.md`,
  pinned to public DUCA commit
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`. The prompt requires a full
  reading of AdapTok's paper, supplement and code; compares fixed-K,
  dynamic-K and hierarchical outer-budget/inner-transport paper structures;
  requests an equation-level solver, total-60 causal matrix, novelty
  boundary, train-free unification and reviewer-attack defenses. No model
  implementation or experiment was started before the adjudication.

- 2026-07-27 21:16 +08:00: superseded the initial compact dynamic-K prompt
  with a first-author-level research-takeover mandate after the user judged
  it insufficiently innovative and deep. Independently inspected the
  AdapTok official repository at exact commit
  `a72076cf6474f930a181aa78971de70d65289b49`, including the concrete
  block-prefix masking, multi-budget label export, scorer training,
  `TransformeScorer`, `AdapTok.encode_eval` and `solve_ilp_min` paths. The
  revised prompt now requires full paper/supplement/code read receipts,
  file/function evidence ledgers, at least ten non-equivalent model
  families, a three-round kill tournament, an equation-level dynamic-budget
  model and theorem candidates, multi-scale hard-counterfactual utility,
  batch-invariant budget calibration, a fair 60-epoch mixed-budget training
  plan, train-free unification, a CVPR paper package and adversarial reviewer
  simulation. This is a discussion artifact only:
  `pro_adjudication_prompt_v2_ready / model_not_frozen /
  no_training_authorized`. The final 707-line prompt SHA-256 is
  `01c496a94bd9f349d5b0d4ca9ca073568805b890b63ea73cf98379c83e788548`.

- 2026-07-27 22:15 +08:00: fully read and byte-identically archived the
  4,589-line dynamic-K / AdapTok research-takeover response at
  `docs/methods/reviews/2026-07-27-duca-dynamic-k-adaptok-research-takeover-raw.txt`
  (152,867 bytes, SHA-256
  `5ae7850662d726d91c4b3dc7f362ad223d33c35e3cbad9bb87771e939e07e031`).
  Issued
  `substantial_accept_research_direction /
  major_correction_before_design_freeze`, not blanket acceptance, in
  `docs/methods/2026-07-27-duca-dynamic-k-adaptok-research-takeover-absorption.md`.
  The response covers the major scientific questions, but its compact and
  expanded halves conflict on independent-per-K versus strict nested frame
  sets; thresholds, split design, matched dynamic controls, formula validity
  and novelty coverage also require correction.

- 2026-07-27 22:15 +08:00: registered `idea:duca-rime` as
  `discussed/pending` and `exp:duca-dynamic-k-rime-oracle` as
  `discussed_proposal/no_training_authorized`. The recommended current decision
  protocol is train-only regret-gated selection among independent, strict
  nested and one weak-overlap decoder before freezing a single model. Clean
  dense/native uniform/wrapper parity, raw `q -> t -> official NMS`, dynamic
  Oracle headroom, decoder-family regret, video-cluster `G_rank` and pair-risk
  are required before a 6,000-update development seed. No DUCA model code,
  checkpoint, Slurm training or paper-ready claim was created in this
  absorption pass; the old negative MUST record remains intact.

- 2026-07-27 23:38 +08:00: fully read, byte-identically archived and
  independently compared two additional dynamic-K/AdapTok takeover replies.
  Source A is `96,650` bytes / `2,694` physical lines / SHA-256
  `2032fcaeddbd4f758ac1be024dd3f867e8dbc6baacd9955de40241ce35595127`;
  source B is `122,113` bytes / `2,667` physical lines / SHA-256
  `e2231c0928c7dd345a4c7a0cf8b55afe4de95270b710b95602ddd6b5c3fb4bf5`.
  Their scientific center is highly aligned but their executable contracts are
  not: DUCA-METER/METER-TAD versus MERTAD, different K grids and total-60
  forwards, hard versus soft risk treatment, different fallbacks and
  incompatible numeric gates. Project decision is
  `accept_scientific_core_with_major_corrections`; keep `DUCA-RIME` as an
  unfrozen internal candidate, require train-only regret among independent,
  strict nested and one weak-overlap decoder, and derive all thresholds from
  clean video-cluster power. Updated the idea, experiment, source registry,
  query pack, anti-repetition rules and graph. No RIME model implementation,
  checkpoint, Slurm job, long training or paper-ready claim was created.

- 2026-07-28: restored and consolidated the SparseHead task into the current
  repository as its only writable implementation route. The unrelated-root
  `OpenTAD_SparseHeadClean_20260702@dce2c66` is now a read-only archive:
  irregular bridge/point generator/native-axis assignment audit and its dirty
  hard-GT balance repair were selectively absorbed, while 16 old repair/retrain
  configs and 6 remote launchers were not copied. The absorbed bridge remains
  fail-closed diagnostic-only; it is not a dense-equivalent or paper method.

- 2026-07-28: restored the complete runnable PhysTime/SDPQ surface from
  `codex/phystime-performance-diagnosis-20260712@e05f6231`: feature/raw/native
  geometry, PhysTime detector/projection/heads, SDPQ, matched controls,
  gate/launchers, focused tests and seven primary experiment records. Merged
  strict temporal padding isolation into the existing VideoMAE adapter while
  preserving ChronoTransport and explicitly rejecting strict masks combined
  with ChronoTransport or packed routing. Registered all restored transforms,
  detectors, projections, heads and utils; added seconds post-processing and
  metadata lineage needed to avoid double conversion and geometry loss.

- 2026-07-28: froze the scientific evidence boundary. Native-J192 matched
  20-epoch selected/physical/SDPQ Avg-mAP is `30.42/44.88/30.88`; current SDPQ
  is trainable but not superior. The full60 and full-precision P0 evidence is
  selected `41.283021` versus physical `57.608685` (`+16.325664 pp`) and does
  not contain SDPQ. Decode-cross dtype repair still has no fresh result.
  Local py_compile, shell syntax, five resolved-config loads, two fail-closed
  legacy config checks, three pure-Python SparseHead contract tests and the
  repository's 20 C3 focused tests passed. Torch tensor/runtime tests
  remain pending Linux/N16R4 because the current Windows installation fails to
  load `c10.dll`; no Slurm run, checkpoint, commit or paper-ready claim was
  created.

- 2026-07-28: implemented approved SparseHead Approach A as an evidence-first
  frozen decode chain. Source ranking-score dtype is preserved; capture is
  opt-in; the suite consumes explicit artifacts without owner/jobs/scheduler
  state; exact P0 full-precision NMS controls and fail-closed proposal audit
  were restored. Remote focused collection found and fixed a real contract
  break where the physical config declared seconds-axis metadata that the
  current head ignored. The explicit path now uses the historical verified
  seconds mapping/domain clamp while the unconfigured C3 path is unchanged.
  Final isolated N16R4 package SHA-256 is
  `e4814e3544784b3608c007a11946464b4f597e0fbf9a23a5910e3b0171bef388`;
  3 configs, import closure, compileall, 3 launcher syntax checks and
  `59 passed in 64.69s` succeeded. No GPU, Slurm training, four-condition CUDA
  gate, formal replay or new mAP was run; status is `tested`, not
  `empirically_supported`.

- 2026-07-28: deployed the complete SparseHead Approach A evidence chain to
  N16R4 from a separate clean runtime snapshot, commit/tree
  `8e31b9e3c08b0a8d320e031b04dfd63e19eb08df` /
  `aae5503424aa3925ef99bba851d600a03e3c3377`. Full-content preflight passed
  with manifest SHA-256
  `3551816b8e056b9afea4fc9ee8575f525e78ffba64ff087915130b2e10e54712`.
  Slurm Job `1201048` serializes the four-condition CUDA gate, four formal
  direct/cross-decode replays and explicit suite in one fail-closed allocation.
  Latest state is `PENDING (AssocGrpGRES)`, so there is no CUDA result, replay
  completion, suite verdict or new mAP yet. Three earlier deployment attempts
  produced zero jobs and are retained only as environment/path diagnostics.

- 2026-07-29: Job `1201048` left the queue, ran on `g0043` for `1m54s`, and
  failed `1:0` after its `39` gate pre-tests passed. The first real model build
  raised `ActionFormer.__init__() got an unexpected keyword argument
  'native_temporal_geometry'`; no gate JSON, replay, suite or metric was
  produced. The signature was traced to consolidation omitting the historical
  native-J192 ActionFormer alignment contract from `8e2b8322`. Switching to
  `PhysTimeTAD` was rejected because it changes constructor/projection/head and
  checkpoint APIs.

- 2026-07-29: performed the single allowed protocol-preserving recovery for
  that signature. Restored explicit ActionFormer native-geometry normalization,
  strict-padding-aware backbone invocation, K384/J192 alignment and query
  audit; added config/constructor and runtime-consumption regressions. A clean
  v5 runtime at commit/tree `0338f4777bd02fb327573ef716f54fec76d4af0e` /
  `cb98c64c17d2983c22181d4908c4f31024a82a2f` passed `74` Linux focused tests
  and full-content preflight SHA-256
  `77b9918aa3173b73fc71d821defa8c14b3165de1b35f0ae4c0382eeb5d21b43d`.
  Replacement Job `1201317` is running; its `41` gate pre-tests passed, but no
  gate/replay/suite verdict or new mAP exists yet.

- 2026-07-29: replacement Job `1201317` passed the complete four-condition
  real-CUDA gate on an RTX 4090. `gate_pass=true`,
  `all_native_direct_exact_equivalence=true`, and selected/physical ×
  online/EMA all report immutable raw tensors under clean runtime
  `0338f477/cb98c64c`. The serial job advanced to `selected_online` formal
  replay. No formal completion, suite verdict or new mAP exists yet.

- 2026-07-29: user superseded the earlier one-repair-per-signature monitor
  limit. Confirmed non-model engineering failures must now be diagnosed,
  regression-tested, preflighted and redeployed through new immutable roots
  until complete final performance is obtained; unchanged blind resubmission
  remains forbidden. A legitimate negative model result must trigger an
  immediate Pro-level multi-factor attribution with competing explanations,
  counterevidence, falsifiable predictions and minimal decisive experiments.
  It must not be relabeled as infrastructure failure or silently tuned away.

- 2026-07-29: Job `1201317` terminated `FAILED 1:0` after `25m29s`.
  Its four-condition real-CUDA gate remains valid and selected-online direct
  inference completed with displayed Avg-mAP `41.26`, mAP@0.3--0.7
  `64.50/56.39/42.66/27.82/14.90`, `3325` GT and `422000` predictions.
  The launcher then failed because
  `direct_work/gpu1_id0/pre_cross_window_detections.json.gz` did not exist.
  Consumer/validator/launcher expected the artifact while
  `eval_one_epoch()` no longer produced it. Registered signature:
  `direct_postprocessing_artifact_producer_contract_missing_v1`. This is an
  engineering contract failure, not a model result; the other three replays
  and suite did not start, and the partial direct metrics are diagnostic only.

- 2026-07-29: restored the historical direct post-processing artifact
  producer for pre-cross full-precision detections, audit and evaluation
  metrics, using atomic writes, and added an end-to-end producer-contract
  regression. Local compile and diff checks passed; local pytest remains
  unavailable because of the registered Windows Torch `c10.dll` WinError
  1114. The clean v6 Linux suite passed `75 tests in 76.43s`.
  Full-content preflight reproduced the frozen inputs and passed with SHA-256
  `97fe5af28b2647396c052c9bdf956997d98e264af74432b57e0fc983b071fb91`.
  New runtime commit/tree is `ac326ffdc97652433b55ccc596e734b112f51806` /
  `0c58027756997995bda0de6fdd8ec0deb49966d3`; unique successor Job
  `1201469` (`ptdc-a1-r2`) is `RUNNING` on `g0030`. Its gate focused suite has
  `42 passed`; v6 CUDA gate/replays/suite/final metrics remain pending, so
  status is `experiment_running`.

- 2026-07-29: v6 Job `1201469` produced a clean four-condition real-CUDA gate
  artifact, SHA-256
  `775e1f2dae70b7863324fd9d235712195dca4d0846968b3bd5e55b754e7b3ea4`.
  It reports `gate_pass=true`, `all_native_direct_exact_equivalence=true`,
  and immutable raw tensors for selected/physical × online/EMA, all under
  runtime `ac326ffd/0c580277` on RTX 4090. The job advanced to
  `selected_online` full direct inference and remains healthy/running; no v6
  completion, suite verdict or final metric exists yet.

- 2026-07-29: v6 `selected_online` direct inference completed and crossed the
  exact v5 producer-contract failure boundary. Exact metrics JSON reports
  Avg-mAP `0.4125660433077075`; mAP@0.3--0.7 is
  `0.6450446628552113 / 0.5638932489689005 / 0.4266348135535575 /
  0.27820781407261164 / 0.14904967708825695`. The direct workdir contains
  `opentad_pre_cross_window_detections_v1` for `211` videos,
  `opentad_post_processing_audit_v1`, evaluation metrics and epoch-59 result,
  all bound to `ac326ffd/0c580277`; audit-recorded pre-cross SHA-256 is
  `31e70dc728aff9061f2c56266e3e6d32ef892b227a5c16b15da85e81f731b50e`.
  Job `1201469` remains `RUNNING` and has begun selected-online dual-axis
  replay. Uniform-rank intermediate artifacts exist, but no formal completion
  or suite verdict exists; status remains `experiment_running`.

- 2026-07-29: Job `1201469` terminated `FAILED 1:0` after `32m32s`.
  Selected-online replay producer completed and validated the shared frozen
  tensors/native-direct equivalence. Uniform/native Avg-mAP was
  `0.4125660433077075`; physical-time cross-decode Avg-mAP was
  `0.5015355102106833`. The validator then referenced unbound
  `numeric_precision` while assembling completion. Registered signature:
  `decode_cross_validator_numeric_precision_scope_v1`. No formal completion,
  other three replays or suite exists; metrics remain diagnostic-only.

- 2026-07-29: bound, validated and propagated a copied numeric-precision
  contract and added a focused non-mutation regression. Clean runtime
  commit/tree is `1631d0b60f6552a6f5eb0378d74e766850f34ffd` /
  `f485c8708e22bbbf9a73063d5293a20bc4aa658f`; the exact v6 recovery surface
  plus the new regression passed `76 tests`. v7 preflight passed, but its
  deployment metadata truncated the final digit of a test-log SHA-256 and was
  rejected before `sbatch --test-only`. Signature
  `deployment_expected_sha256_truncation_v1`; zero Slurm jobs were created and
  the v7 root is preserved.

- 2026-07-29: prepared a fresh v8 runtime/run root at the same clean
  commit/tree, reran `76 passed`, and passed full-content preflight SHA-256
  `e9f36c221156e5411dad5e3bfe43508b4aa59310539fdbe24da985fc99a27d53`.
  `1201494` is test-only. Unique formal Job `1201495` (`ptdc-a1-r4`) is
  `RUNNING` on `g0024`/RTX4090; deployment identity SHA-256 is
  `abb8aefc41c24a7d94de5ec0938c42f4ebd17b84f3eff368cd7badbd61d87f22`
  and submission receipt SHA-256 is
  `ae43e2744ece2898ca46dba2ad26d943a7524f932df95c71725f380a0b59cac4`.
  Status remains `experiment_running`.

- 2026-07-29: v8 Job `1201495` passed `43` gate focused tests and the complete
  four-condition real-CUDA gate. Artifact SHA-256 is
  `5e323e5ccdedd7dd39d70148aed7108beca94bb5952125a124ad20accfd634f6`;
  `gate_pass=true`, all native/direct comparisons are exact and all four raw
  tensor sets are immutable. It advanced to selected-online full direct
  inference and remains healthy/running; no v8 completion or suite exists yet.

- 2026-07-29: v8 selected-online direct inference completed with exact
  Avg-mAP `0.4125660433077075` and mAP@0.3--0.7
  `0.6450446628552113/0.5638932489689005/0.4266348135535575/
  0.27820781407261164/0.14904967708825695`. Its pre-cross artifact covers
  `211` videos, is bound to `1631d0b6/f485c870`, and has SHA-256
  `b4adcf545655424d2b2dfdfce0d107109c5010850143fadf925706fb3de60322`.
  Uniform-rank replay artifacts are complete and physical-time replay is
  running. No formal completion, suite verdict or hard failure exists yet.

- 2026-07-29: Job `1201495` terminated `FAILED 1:0` after `02:00:24`.
  All four replay completions and producer completions exist and pass their
  frozen/native-direct/P0 parity checks. Their uniform -> physical Avg-mAP pairs
  are selected-online `0.4125660433 -> 0.5015355102`, selected-EMA
  `0.4128302079 -> 0.5009785403`, physical-online
  `0.4010767719 -> 0.5755558109`, and physical-EMA
  `0.4029649803 -> 0.5760868491`. The explicit suite failed before completion
  because `fatal_log_findings` was JSON object `{}` while its consumer requires
  array `[]`. Registered signature:
  `decode_cross_completion_fatal_log_findings_container_type_v1`; v8 failure
  receipt SHA-256
  `22739defebe8261f61e1fff9910d6d74592d6de4621f7147b07138154ae94d13`.
  The metrics remain diagnostic-only, not model evidence.

- 2026-07-29: the first fresh v9 root stopped before commit/preflight/Slurm
  because repository-local Git author identity was missing. Signature
  `runtime_git_author_identity_missing_v1`; zero jobs; failure receipt SHA-256
  `ca7f75bc72e85fd466331012775cff72ca14fd685b1db4cc52c8212450c994d2`.
  The root is preserved and not reused.

- 2026-07-29: froze v10 branch
  `codex/sparsehead-evidence-recovery-20260729-v10` at commit/tree
  `c878fbe3a5e960671f03d93fff8367ed3414f5c5` /
  `8d3e73bb26544d1bcf7bfb61154d0b003f2658e0`. The protocol-preserving fix
  serializes clean fatal findings as `[]`, detects explicit PhysTime error
  markers, and adds a cross-module type regression. Linux exact suite passed
  `77 tests`; test-log SHA-256 is
  `7f1787308250a6c9bd62e452f6e16357f5d6bf44cdbcfc6fedd61b7cc63c6936`;
  preflight SHA-256 is
  `f46f6299f7fccc899140ad8fdf001052772ef550dd34cdb68c17d5ba5fc59a8f`.

- 2026-07-29: `1203046` is test-only. Unique formal v10 Job `1203047`
  (`ptdc-a1-r5`) is `RUNNING` on `g0050`/RTX4090. Deployment identity and
  submission receipt SHA-256 are
  `1ece7c71b3fc9c396f49401460e5474e3dcaa7ba7f6cf009b987c2b3909a2246`
  and
  `9ec33e550d72f69847bcb2a5b2457fad03aa15df54d843e233b2020b5ef5724f`.
  Gate focused tests passed `44 tests`; CUDA gate/four fresh replays/suite are
  still pending. Status remains `experiment_running`.

- 2026-07-29: v10 Job `1203047` passed the four-condition real-CUDA gate.
  Artifact SHA-256 is
  `e5516af02289d15dd1465f5387471bb1a3c357873980d22645c08acbf6aa141c`;
  `gate_pass=true`, all native/direct checks are exact, and all four raw-tensor
  sets are immutable. The job advanced to selected-online full direct
  inference. No v10 replay completion or suite result exists yet.

- 2026-07-29: v10 selected-online completed direct inference, both frozen
  decode replays and validation. Completion / producer-completion SHA-256 are
  `a4e727cf094127be7b91a4a13b140463ad9dc3e0c8c1bcfa3acb9887b5ff6dda` /
  `8a2d38db8a2130a8b617940361a8637dfdc0bff3b6947b0f35d75167a809bfa6`;
  `validation_pass=true`, clean fatal findings are `[]`, and all
  frozen-raw/native-direct/reviewed-P0 checks pass. Uniform / physical decode
  Avg-mAP is `0.4125660433077075 / 0.5015355102106833`. This is one `tested`
  component only. Job `1203047` remains healthy and advanced to selected-EMA
  direct inference; overall status remains `experiment_running`.

- 2026-07-29: v10 selected-EMA also completed direct inference, both frozen
  decode replays and validation. Completion / producer-completion SHA-256 are
  `0c6f87617b1cbd6a5bc4a6be6e9a5a2174f8a5a568c2f24db7253c15a315b8dc` /
  `ddddd42174eb987cdeb723ae4422df8105e773bd7af74d31e67760dba20d74ff`;
  validation, clean fatal-finding array and all frozen-raw/native-direct/P0
  checks pass. Uniform / physical decode Avg-mAP is
  `0.41283020792762315 / 0.5009785403306161`. Two `tested` components now
  exist. Job `1203047` advanced to physical-online direct inference; overall
  status remains `experiment_running`.

- 2026-07-29: v10 physical-online completed native physical-time direct
  inference. Exact Avg-mAP/mAP@0.3–0.7 is
  `0.5755558109390063 /
  0.7704022473065874/0.7055742485050899/0.6207490393477052/
  0.48593657950784275/0.29511694002780653`; direct-metrics SHA-256 is
  `b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009`.
  Dual-axis replay is running; no producer or validator completion exists.
  This is `diagnostic_only`, not the third tested component. No hard failure
  is present and overall status remains `experiment_running`.

- 2026-07-29: physical-online then completed both frozen decode replays and
  validation. Completion / producer-completion SHA-256 are
  `02384da2c71c93bdcd6ce003cd59451510c9d095e222653202f09f38b73b153f` /
  `b9ba401a92e0d828aeabe48cb8972df74a64720a12f160d939daa355856aaf58`;
  all contract checks pass with `fatal_log_findings=[]`. Uniform / physical
  decode Avg-mAP is `0.40107677185286417 / 0.5755558109390063`, a
  `+17.447903908614215` pp decode-axis difference on frozen raw tensors.
  Three components are now `tested`; Job `1203047` advanced to physical-EMA
  direct inference and remains `experiment_running`.

- 2026-07-29: physical-EMA completed native physical-time direct inference.
  Exact Avg-mAP/mAP@0.3–0.7 is
  `0.5760868491267752 /
  0.7721224901972557/0.7045574192938243/0.6257613932435541/
  0.4900660583199814/0.28792688457926047`; direct-metrics SHA-256 is
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`.
  Final dual-axis replay is running; no producer/validator completion exists.
  This remains `diagnostic_only`; no hard failure is present and overall
  status remains `experiment_running`.

- 2026-07-29: v10 physical-EMA completed both replays and validation.
  Completion / producer SHA-256 are
  `a5c0c5248bf196d17f1cbf4f11a61d01459cb2ff3cfbf37541046fdb508b7ad1` /
  `8433bd22b620cd60300d94289cf991b69c1f64bcd5eacea557fbc463d7981086`;
  uniform / physical Avg-mAP is
  `0.40296498031949024 / 0.5760868491267752`. This is the fourth `tested`
  component.

- 2026-07-29: Job `1203047` then ended `FAILED 1:0` in the explicit suite.
  Preflight and gate bind the same checkpoint path/SHA, but the consumer
  compared differently enriched records as raw dictionaries. Failure signature
  `decode_cross_suite_checkpoint_binding_schema_shape_mismatch_v1`; suite log /
  failure receipt SHA-256
  `68b7b3d34e587392bdac2df1eb2a36d971009d4c07165ef2a18157449ccb931f` /
  `42c394f11153a862819876b3915c34ca2ef0a68b6b62ed78a121d65db4269cec`.
  No model verdict is allowed.

- 2026-07-29: v11–v15 were preserved as separate zero-job pre-submission roots:
  profile/nounset+mode, recovery-test scope, preflight import path, finalizer
  `$BASE` token, and SSH transport interruption respectively. Receipt SHA-256:
  `2a95ca48464564d4979754525129414124c769a8a97852a9fad404087bc08545`,
  `387d61f33eb3dc055c182a8df23c721378ac4191ad170646311df021fc67e259`,
  `a6d0ccf593e5cb01b9f6a90dee1a47d0042f8ab8a4201be68033b6868fb19858`,
  `8a85f361fdfa90a6a753c5c3446a43617359cd18ee2a5ea541eac4f6ac00d387`,
  `f7b7402cc1c565a69a01d057c0e50d7ea63632c3a6a8613be30adc866401630e`.

- 2026-07-29: v16 froze the canonical checkpoint-identity fix at commit/tree
  `54e7f9abeaabf710a505f0a0f595a4eb3bb47f98` /
  `f8490f9c25c2e0e6958c406e19c83cc3d5a40535`. Linux exact recovery passed
  `78 tests`; test/preflight SHA-256 are
  `d81ca79bd9af216c106fb9718e7b171dd47c9aff3ddecb9787d8e0203c88d0fc` /
  `ccc7a83e27b8d18ad0892b644e7338667b72d8eba3e3feedbc387dc4ac1d9a0d`.
  Deployment identity / submission receipt SHA-256 are
  `6f22152938b2ad3949a19672e622e97d861a7604f8ff9b5408d59e21bcfcf6d4` /
  `65c325fbd53b3c8386ce459e557f7d8e09f768eb38d77057d8e442b680393ad7`.
  `1203916` is test-only; unique Job `1203917` (`ptdc-a1-r11`) is `RUNNING`
  on `g0045`/RTX4090.

- 2026-07-29: v16 gate focused tests passed (`45 passed`). Four-condition
  real-CUDA gate artifact SHA-256 is
  `0d2153effee84a0e1aa6410125bb291eb4ef4d41e4b40604f49d9e5868e0ada9`;
  `gate_pass=true`, all native/direct exact equivalence, and all four
  `raw_tensors_immutable=true`. Job `1203917` has entered selected-online direct
  inference. Hard-failure scan is empty; no completion or explicit suite exists.
  State remains `experiment_running`.

- 2026-07-29: v16 selected-online completed full direct inference and both
  replay modes. Uniform/native Avg-mAP is `0.4125660433077075`; physical-time
  Avg-mAP is `0.5015355102106833` (`+8.89694669029758 pp`). Direct/uniform and
  physical metrics SHA-256 are
  `8860bdcaf3b998e6cddb1187c564d0bb0693496552439b104efad7145a6bd34c` /
  `7a032eaf8e4fc776ae0d670d572e02f74c23b82ef55bc29185e796e5be2f0f8b`.
  Producer completion SHA-256
  `97410d9855a3f6db859e36213bf6b201e10c96941a164b5588af02cdfba4ee20`
  reports `validation_pass=true`, frozen raw tensors and native/direct exact
  equivalence. Formal component validation and the other three conditions are
  still pending; state remains `experiment_running`.

- 2026-07-29: v16 Job `1203917` wrote the first formal component receipt:
  `selected_online/DECODE_CROSS_COMPLETE.json` SHA-256
  `6937fc6b7b050fd7009ee967ceef446aebaa8b3daa695c7959106ff87048c038`.
  It records `status=tested`, `validation_pass=true`,
  `fatal_log_findings=[]`, frozen raw tensors, native/direct exact equivalence,
  reviewed-P0 parity and `new_training=false`; all invalid/filter/rounding
  counters are zero. The job has entered `selected_ema`. Three formal
  components and the explicit suite remain pending, so the route stays
  `experiment_running` and no final model attribution is authorized.

- 2026-07-29: v16 `selected_ema` became the second formal `tested` component.
  Uniform/native Avg-mAP is `0.41283020792762315`; physical-time Avg-mAP is
  `0.5009785403306161` (`+8.814833240299292 pp`). Direct/uniform and physical
  metrics SHA-256 are
  `ed3750a61a27dc70ac570f29ccefff8eef8d4dc10ea29802743b403807b82a34` /
  `742b9a810f52dfe9bd12c29987148bf3c95e99c58aefb5774f2f8b3d18d30c1`;
  producer/formal completion SHA-256 are
  `43c737fe3c5a9a534c565bf63e419fa152ee35b3be796ddf3f601c954fa52877` /
  `4a1b405b7849f396e1b649da8895070e6176023c4a959c6d7fd9148f2bd8afe0`.
  All validator contracts pass and hard-failure scan is empty. Job `1203917`
  entered `physical_online`; two components and explicit suite remain pending,
  so status stays `experiment_running`.

- 2026-07-29: v16 `physical_online` completed direct inference and both replay
  modes, but its formal validator receipt is still pending. Physical/native
  Avg-mAP is `0.5755558109390063`; uniform-rank cross-decode Avg-mAP is
  `0.40107677185286417` (`+17.447903908614215 pp`). Direct/physical, uniform and
  producer SHA-256 are
  `b68f2ad1393b59c40d58f7cfa1e450a52f84d8acbc80ad785a2d3a31352d6009` /
  `0c258e563fe7b9886e6d56c9c3370b6536e187b521526318622b07ffcf1e4a4b` /
  `d61d8fbf8b977b59b65eb87d55227904b2a5a2e6994e584226bda19a265b26eb`.
  Producer validation passes and hard-failure scan is empty. This remains
  diagnostic producer evidence, not a third formal component; route status is
  still `experiment_running`.

- 2026-07-29: v16 `physical_online` formal validator completed. Receipt
  SHA-256 is
  `fd18348e6ae6ecf4bdc4390ca4620a109616582f7f77138ed137085e0df6c260`;
  `status=tested`, `validation_pass=true`, `fatal_log_findings=[]`,
  frozen-raw/native-direct/reviewed-P0 parity and `new_training=false` all pass.
  This is now the third formal component. Job `1203917` entered
  `physical_ema`; the fourth completion and explicit suite remain pending, so
  route status stays `experiment_running`.

- 2026-07-29: v16 `physical_ema` formal validator completed. Uniform/physical
  Avg-mAP are `0.40296498031949024 / 0.5760868491267752`
  (`+17.312186880728497 pp`); mAP@0.3--0.7 are
  `0.622154649489393 / 0.5316588686305871 / 0.4113769771975965 /
  0.2880843206041682 / 0.16155008567570622` and
  `0.7721224901972557 / 0.7045574192938243 / 0.6257613932435541 /
  0.4900660583199814 / 0.28792688457926047`. Uniform/physical metrics SHA-256
  are `5058f789de9fd74544427fd8201d7b32cc83f18524409ee9e8f3b96fe32292dc` /
  `43c33d551c19f4f3ab90108af30b13c103aa3a875fd87f00a4f50c7e5a83ecac`;
  producer/formal receipt SHA-256 are
  `aa6356a509898b94a38f2b9e0548c5f647cc6498655697b37fd39ea8982fc733` /
  `cd6da2f827524e0b9eb2b46c6cbbcc5b6e89243aa9cd8d7e45efafcb4cb6b565`.
  All contracts pass, making this the fourth formal `tested` component.
  Explicit suite and terminal Job `1203917` state remain pending; route status
  stays `experiment_running` and final model attribution has not begun.

- 2026-07-29: v16 closed the complete evidence chain. Slurm Job `1203917`
  ended `COMPLETED 0:0` after `02:34:30`; commit/tree/cleanliness match and
  hard-failure scan is empty. Explicit suite completion / validation-marker
  SHA-256 are
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3`.
  Both pass; completion is `status=tested`, `new_training=false`,
  `fatal_findings=[]` and binds preflight/gate/P0/four completions/checkpoints.

- 2026-07-29: Pro multi-agent attribution completed. Four frozen P-U Avg-mAP
  gains are `+8.8969/+8.8148/+17.4479/+17.3122 pp`; all agree and online/EMA
  variation is negligible. Duration/proposal diagnostics localize the largest
  damage to short actions and high tIoU. Ranked explanation is decode/native
  temporal geometry mismatch first, assignment/support representation coupling
  second, ranking/NMS amplification third, and single-seed chance last as a
  sole cause. There is no four-condition contradiction.

- 2026-07-29: Route verdict is `tested`, not `empirically_supported` or
  `paper_ready`. Approach A is positive for physical-time-before-NMS decode and
  negative for harmless selected-rank decode; it does not rescue current SDPQ
  or prove training causality/cost/robustness. Next designed sequence is
  independent sealed-artifact evaluation, 64-window assignment/support audit,
  class/calibration/failure decomposition, native parity, then only conditional
  SDPQ micro-overfit, multi-seed and full cost. No retraining was launched.

- 2026-07-29: implemented the approved no-training diagnostic closure and
  official-comparability hard gate in isolated branch
  `codex/sparsehead-diagnostic-closure-20260729`, commit/tree
  `57917e7bf2b991478b4f6fc4ce1db5ca5878b68d` /
  `aaf7c82bd837078bb7276baf6c0a504da0684194`. The independent v16 evaluator,
  sealed 64-window SDPQ support audit, pinned official ActionFormer record
  builder, strict single-intervention comparator and focused tests compile and
  pass (`35 passed`). No remote run or model metric was produced.

- 2026-07-29: official ActionFormer comparability is now fail-closed. The gate
  pins upstream commit/tree `61ea7eb9/7b06c526`, config/README hashes and
  official THUMOS archive MD5; verifies live receipts; hashes every I3D feature;
  validates raw `eval_results.pkl`; and requires official-log metrics to agree
  with an independent invocation of the pinned official evaluator. The
  skip-hash and arbitrary-difference-prefix paths were removed. v16
  VideoMAE/K384 remains diagnostic-only and `63.61` remains external-reference
  only. Remote independent closure and the official I3D anchor are the next
  execution steps; structural retraining remains unauthorized.

- 2026-07-29: repaired two independent-recompute engineering contracts without
  changing v16 evidence. The first failure rejected 1,443 legal NaN padding
  entries per axis despite zero invalid valid-prefix entries; the second used
  logical `test` against OpenTAD annotation subset `validation`. Commit/tree
  advanced to `6d74ad7b7c7736bbff48976a626b951512a54e96` /
  `80cd2431ebf9809f03ab1216b84b45380d51f33b`; local/Linux focused results are
  `46 passed, 1 skipped` / `58 passed, 1 skipped`. Fresh independent v3 is
  running; older failed roots remain intact.

- 2026-07-29: official ActionFormer resources were downloaded from the pinned
  release rather than reconstructed from legacy pickles. Download v1 failed
  because proxy variables were not exported; v2 failed because `gdown` wrote
  cookies into a quota-exhausted home cache. Fresh v3 completed with official
  THUMOS MD5 `375f76ffbf7447af1035e694971ec9b2`; released checkpoint/log ZIP
  SHA-256 is
  `e028f7e487713d0c68f0515ba9bdafda0ed05fc1271b9999ea995652b034c929`.
  Exact upstream extraction and dataset/checkpoint preflight are running.

- 2026-07-29: the SDPQ audit now binds the expected checkpoint epoch explicitly.
  Exact diagnostic input is clean config repo `4a57577`, epoch-19 online
  checkpoint SHA-256 `40fccfd...b2c3fc7`, VideoMAE SHA-256
  `4b96b7f4...e0de251`, seed 42 and 64 sealed train windows. Resource test-only
  rejected explicit `--mem=48G`; omitting it passed. Job `1204961` then failed
  in one second because Slurm `--wrap` used `/bin/sh` with `pipefail`.
  Test-only IDs `1204959/1204960/1204980` are not jobs. The unique Bash-wrapped
  successor Job `1204981` is pending. No model training or performance metric
  was produced.

- 2026-07-29: sealed the official-comparability and support-observability
  contracts at commit/tree
  `2b074845497f6ada3314cb895f0d4ab2f4ce3eca` /
  `7779862c5422dc8e527b304bf881a760b0c90625`. The exact Linux runtime passed
  `95 passed, 1 skipped` (log SHA-256
  `265046cd7fc3b1e847e87880e061a5a76092c4b194d1d4e727ca706f5b8884b6`).
  The official audit now distinguishes nominal THUMOS 200/213 from the pinned
  annotation DB's 200 `Validation`/212 `Test`, binds the canonical 20 classes,
  validates all 413 finite nonempty `T×2048` feature arrays, identifies
  feature-only `video_test_0001292`, reparses raw predictions and requires exact
  evaluated-video-set equality. Matched method rows now fail closed without a
  base-anchored live source-diff attestation.

- 2026-07-29: official ActionFormer Job `1205131` produced diagnostic
  mAP@0.3–0.7 `82.13/77.81/70.95/59.40/43.87` (Avg `66.83`) and 42,400
  predictions over the exact 212-video official evaluated set, then failed the
  obsolete split schema contract
  `official_annotation_split_schema_contract_v1`. The raw prediction SHA-256 is
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
  Unique clean rerun Job `1205178` was submitted; `66.83` remains a candidate,
  not a paper-main-table result, until its strict verdict passes.

- 2026-07-29: SDPQ support Job `1205132` exposed
  `sdpq_support_overlap_query_padding_mask_omission_v1`. Epsilon-clamped padded
  widths created tiny first-interval support mass because the support-overlap
  path omitted the final query mask. The production fix applies the mask after
  either geometry branch and leaves every valid query unchanged; a
  variable-duration batch regression is included in the 95/1 Linux suite.
  Unique diagnostic successor Job `1205179` was submitted. Independent replay
  Job `1205133` remains the sole recomputation. Test-only IDs `1205176/1205177`
  are not jobs. No training or method metric was created by either repair.

- 2026-07-29: official released-checkpoint evaluation Job `1205206` completed
  with mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118` and Avg
  `66.833392`; raw prediction SHA-256 is
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`.
  The result remains unsealed because the old record asserted seed `0` while
  official config/log evidence uses `1234567891`. It is not yet a paper row.

- 2026-07-29: repaired SDPQ support Job `1205240` passed its diagnostic
  completion (SHA-256
  `abf28cf420f0e2e06b3d727e9da92c98f55fba626f334cd73c6b4c4cb3ee1167`):
  647/647 GT assignments had support evidence, with zero missing domains,
  collisions or uncovered positives and maximum offset error
  `3.0517578125e-05`. This is observability evidence, not a performance result.

- 2026-07-29: source/effective-config comparability was hardened, including
  live Git base ancestry, exact allowed A/M paths, no rename/copy/delete,
  protected loader/config expansion and equality of all non-method protocol
  fields. Linux v14 passed `125 passed, 2 skipped` after preserving the v11-v13
  fixture/transport failure roots.

- 2026-07-29: independent recomputation Job `1205243` failed after completing
  all reports with `independent_recompute_semantic_match_drift_v1`. Exact raw
  scores/masks and proposal geometry matched and all delta signs were stable;
  up to `0.00185658` aggregate drift came from NumPy stable/float64 semantics
  differing from production PyTorch `2.0.1` unstable CPU sort and scalar
  float32 C++ Soft-NMS. This is an engineering closure failure.

- 2026-07-29: commit/tree
  `e2a0d74f561b158c531d4909e72ecee69b153c16` /
  `0b6cb7996ee90f3209a78b78bbf7a55525e3badd` independently ported the
  pinned production sort and float32 Soft-NMS/`expf` semantics without
  importing production decode/NMS/evaluator code. Exact Linux v16 passed
  `127 passed, 2 skipped`, log SHA-256
  `115dd497a3a662b3fc0f19ae9104257d245cbadbb7fd4001f3eb3ea71432534c`.

- 2026-07-29: unique formal successor Job `1205388` was submitted for exact
  eight-report independent closure under run root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/sparsehead_diagnostic_closure_20260730_v9`.
  Submission receipt SHA-256 is
  `bf27a72af865c7db4148912df9b3fbdc75530fba01b0aae006953f844054fbcb`;
  test-only ID `1205384` is not a job. Status: `experiment_running`.

- 2026-07-29: Job `1205388` failed in `00:00:01` with exit `127:0` before
  model or validator startup. Signature
  `slurm_module_function_unavailable_v1` records that the non-login Slurm
  allocation lacked the shell `module` function. The v9 sbatch/stdout/stderr
  SHA-256 values are
  `1478424538a1cad2c2fed81385f12e64165d2b6370352deacfa35f9a4e39cfcb`,
  `f8a9fe85b5fe43c9f604211489b7791e57ac2916688095d3c88b94ac2724861b`
  and `261f606d5d9466acb58ea0402363fd98e985d733bf2f438700085e7dbe84017f`.
  Failure receipt SHA-256 is
  `f4f2b305be639575310dc290accbc88d381812902b3edd090a9137438f7a0359`.

- 2026-07-29: an empty-environment probe verified that directly sourcing
  `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/activate` resolves the
  pinned Python and PyTorch `2.0.1`. Unique successor Job `1205400` was
  submitted with v10 sbatch SHA-256
  `21372695291fdf8089f93920665a9ac844f4ee21ca9da07dcb5a6c95df9dd506`;
  submission receipt SHA-256 is
  `da679424ad5a3dbfd3cc0b6e28fd74b638d2bc7873098a4d1c46a2e80c14bea2`.
  Test-only ID `1205398` is not a job. Status: `experiment_running`.

- 2026-07-29: unique official dense-anchor reseal Job `1205409` was submitted
  under fresh root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_official_anchor_reseal_20260730_v1`.
  It locks official commit/tree `61ea7eb...`/`7b06c...`, audit commit/tree
  `e2a0d74...`/`0b6cb...`, seed `1234567891`, effective-config SHA
  `835cf30f...`, 15 receipts and `--require-main-table`. Sbatch/submission
  receipt SHA-256 values are
  `4a377d0d580c6baae2b50a277a2c9f04ce0d5470719bcf32354005775e14cfa0` /
  `87f901b944fe5a1054cd9ae168336c024d7d8142407fc322b25956163ec1b68d`.
  Test-only ID `1205408` is not a job. Status: `experiment_running`.

- 2026-07-29: Job `1205409` failed in `00:00:36`, exit `1:0`, before official
  inference. NMS build and ABI smoke passed, then a new environment probe
  imported `nms_1d_cpu` before `torch`, so `libc10.so` was not loaded.
  Signature: `official_environment_probe_nms_import_order_v1`. The official
  source and audit runtime remained clean. Failure receipt SHA-256:
  `2d4df6637af61f39f0d516eeba519da9fcaff73289e0b73ff55f2bbf2c841af6`.

- 2026-07-29: a clean-shell regression verified the official order
  `import torch; import nms_1d_cpu`. Unique anchor successor Job `1205419`
  was submitted under fresh v2 root with unchanged model/protocol gates.
  Sbatch/submission receipt SHA-256 values are
  `bced7838ee244613222c236fd9393baf40eb09aa8e6a0d7021042ebb95597777` /
  `3a70fd84de376f377ce34d0085ea11772ca0fefaa8223b4d2d495edf0a69f03a`.
  Test-only ID `1205418` is not a job. Status: `experiment_running`.

- 2026-07-29: Job `1205419` completed both official evaluation passes at
  mAP@0.3–0.7 `82.13/77.81/70.95/59.40/43.87`, Avg `66.83`, then failed
  closed in the record builder. Signature
  `official_released_train_log_default_serialization_omission_v1` records the
  only config difference: the released log omits `model.fpn_start_level`,
  while pinned `libs/core/config.py` injects exact integer `0`. Raw predictions
  retain SHA-256
  `1333df9202eec7ae217542b6bd2b15b597c1a004ebb3634de54a7a37adb6d7fe`;
  failure receipt SHA-256 is
  `079818253bc87a78ed67ce41dbd092aa64f0e54b5a61972f2313adeb7d10fa4a`.
  This is an engineering provenance failure, not a paper result.

- 2026-07-29: commit/tree
  `8b80c98ee2af65561bf305b4fdc2ef16e460da73` /
  `148a93eac4ff1b6a3be46fdca72c705aa17294a6` added a fail-closed
  official-default normalization attestation. It pins raw released-log config
  SHA-256 `ad426e1a...3d3a7`, permits only missing
  `model.fpn_start_level=0`, and requires exact normalized equality with the
  source-expanded config. Local focused result: `44 passed, 1 skipped`.
  A GitHub TLS termination left immutable v17 incomplete; SHA-verified bundle
  deployment produced clean exact v18. Linux full suite passed
  `131 passed, 2 skipped`, log SHA-256
  `6899bf6126d1ce9b3d880d348cdf5c1f152235d3b2e6f6de028b5fc807fb34fb`.

- 2026-07-29: unique official-anchor successor Job `1205455` was submitted
  under fresh v3 root; `1205454` is test-only. It keeps official
  commit/tree, checkpoint, seed `1234567891`, I3D features, evaluator and all
  model/protocol settings unchanged and adds only the audited normalization
  receipt. Sbatch/submission receipt SHA-256 values are
  `76fc3df0c1faadfc9f62fb2982a8aee6013fc4a6a502cd32dbdf3e27fd7ec0a7` /
  `8f26fbed8284f83d6d099779f88c76d61ee0181323d208f355aff10dbb426744`.
  Status at submission: `experiment_running`; K384 remained blocked.

- 2026-07-29: official anchor Job `1205455` completed `0:0` in `00:03:26`.
  The independent evaluator reproduced mAP@0.3–0.7
  `82.133988/77.805571/70.953608/59.401673/43.872118`, Avg
  `66.833392`, from 42,400 predictions. All 15 live receipts passed;
  `official_actionformer_protocol_match=true` and
  `main_table_eligible=true`. Completion/protocol/verdict SHA-256 values are
  `90c8bae14fcb20cc2434cea37f47065704766e38ff9663eac6e70c0d338b9e94` /
  `808199b54b0ebcfebda403419873cc5fd46c36a4d404d3d8ce31838ce3b5bd95` /
  `0706247ef978bf339f9a9cb4adaef07500e8d991129c6d0862118088b917a2ec`.
  Status: official dense comparator `empirically_supported`; it is the sole
  paper baseline for the next matched K384 intervention.

- 2026-07-29: independent design reviews preregistered the official K384
  control. The primary mask is deterministic stratified-uniform support on the
  original full-video/FPN physical grid; fixed video-hash random support is a
  secondary robustness control. Only the head-query computation may change.
  Three kernel-3 head layers require a radius-3 physical halo per selected
  center, real skipped computation, dense-index scatter and unchanged
  loss/decoder/NMS. Status: `designed`; implementation and testing begin only
  on the exact official base.

- 2026-07-29: Job `1203917` reached `COMPLETED 0:0`; four formal decode-cross
  completions and the explicit suite all validate. Suite
  completion/validation/deployment SHA-256 values are
  `ed2770c35cf9a3acd5fa80465eda1c34b3541ba3dea404c75388aaeffefbdc31` /
  `f2da143127b3a01aef7bda451e2351c494f72552f3810f604f895f4c0a7767d3` /
  `bc825f08445e4c8fe8f3ab5dd768b6f9cdf3ec7fdd40dc02438428237c004b2e`.
  Status remains `tested`: frozen inference/decoder-axis evidence only, not an
  official ActionFormer paper comparison.

- 2026-07-29: official-native SparseHead candidate was implemented and pushed
  at commit/tree `55763a9ef7ce18a51827fe48040081c4fe2b84d4` /
  `c489a54aa501b39421cddb5df98385b3889ed479`; clean Linux focused tests passed
  `11`. An adversarial comparability review corrected the earlier
  execution-only description: the exact training contract is
  `training.loss_support=selected_native_grid_queries`, so unselected
  positives/negatives and the loss normalizer differ. Released-checkpoint
  inference is diagnostic only; official dense/sparse matched retraining is
  mandatory for a paper row.

- 2026-07-29: audit/launcher commit/tree
  `aab72e484538931a565930b99d1beb71f47b9ceb` /
  `25e7e0eb3b8cd5edfb48eac594eda6b89edffa36` passed local focused tests
  `40` and launcher tests `5`; remote source-diff receipt SHA-256 is
  `409ffd3035a0c957d3b250db24fe017c5c09efda526d746ace0d54f00c695abc`.
  Preserved non-model failure signatures are
  `audit_preflight_test_path_drift_v1`,
  `slurm_module_function_unavailable_v1`,
  `runtime_profile_source_under_nounset_v1`,
  `github_http2_remote_ref_transport_v1`,
  `github_remote_ref_live_check_transport_hang_v1` and
  `readonly_bundle_push_target_v1`.

- 2026-07-29: unique real-CUDA gate Job `1205541` was submitted under
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v1`;
  `1205539` is test-only. Deployment/submission receipt SHA-256 values are
  `b37a08c2031bb7b043038ea6baf022830bda4ca1203abbff41619401537e8a8e` /
  `471022b2e726cf923e5a445aef8c21ca5f17c9e59b7e586ed8fb3ed4bbc49665`.
  Last submission status was `PENDING (Priority)`. Status: engineering gate
  `experiment_running`; no metric claim.

- 2026-07-29: Job `1205541` reached `FAILED 2:0` in `00:00:30`.
  Correctness contracts passed (maximum selected error `4.0531e-6`, immutable
  raw/masks, exact zero unselected outputs), while cost failed: dense
  `6.1918 ms`, sparse preselected `19.6508 ms`, sparse with selector
  `20.5046 ms`, speedup `0.3009x`. Failure signature:
  `native_grid_sparse_head_microkernel_launch_and_scatter_slowdown_v1`.
  CUDA gate/failure-analysis SHA-256:
  `8aeb2cdbf02da0f8ad675b2f5a33d3ef6d89198ac7e216511ffde45d66f505a3` /
  `ef6b462d79316e2c3f80bf125eb8704b30c0c3e229568048b67095a172152b7d`.
  Status: engineering failure; no model metric.

- 2026-07-29: packed-kernel recovery commit/tree
  `d64e66dfd7fc9881552b342f5523926cc78c0848` /
  `16265c70b235034acb52521b00c259ec6d8b59e1` was pushed and frozen at
  `/data/run01/sczc063/yuzibo/projects/actionformer_sparsehead_official_matched_20260730_v3`.
  It batches all samples/FPN levels into one convolution per head layer and
  adds a launch-count regression without changing the method contract. Linux
  candidate/audit/launcher tests passed `12/40/1`; source-diff SHA-256
  `5aea817bf1fd1b2c0e36193b9d99ee71bde3dfd00c05673ece5dc4f6da9304d4`.

- 2026-07-29: a first SCP of the successor pre-submission hash file closed
  before transfer; remote state was confirmed `MISSING`, then an exact retry
  passed. Signature:
  `ssh_transport_interruption_during_pre_submission_receipt_copy_v1`.
  Slurm test-only `1205566` is not a job. Unique Job `1205567` was submitted
  under fresh root
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_native_grid_k384_cuda_gate_20260730_v2`
  and is `PENDING (Priority)`. Deployment/submission SHA-256:
  `c2890c1b37e22810fdc8284b80ca6292e7bf5cc1c38820fb74e8d68d96647b52` /
  `f71c394c09f5d5a65bdf37036739294d553098ea3ecfa79b8ebf10c8486b3798`.
  Status: `experiment_running`; training remains blocked.

- 2026-07-29: Job `1205567` reached `FAILED 2:0` in `00:00:28`.
  Correctness passed (maximum error `2.8610e-6`, immutable raw/masks, zero
  unselected), but dense/sparse-preselected/sparse-with-selector means were
  `6.2409/12.7657/13.6182 ms`, speedup `0.4590x`. Signature:
  `native_grid_sparse_head_packed_patch_materialization_and_microconv_slowdown_v1`.
  CUDA-gate/failure-analysis SHA-256:
  `f4a0479b48c434832c45d84e9eccc6ebc9e56be88a03d8e8eff4fca525981113` /
  `fe2f6d62272ad558be18e068ca1796808d516b105b2ed41202eb5a7e0e1fb6d6`.

- 2026-07-29: candidate commit/tree
  `31e6112ea28747098cfe5412c097d737731bfaa1` /
  `d2619cd075c4e7192ca060f34d811ac3fe5768f8` replaced packed Conv1d with an
  algebraically equivalent flattened GEMM and passed `12` Linux tests. A
  GitHub clone failed as `github_https_clone_tls_termination_v1`; bundle clone
  v5 failed closed as `bundle_clone_remote_head_unset_v1`; SHA-verified bundle
  `c50bea0b79e242bb4c96cf11fb35a3ef095a8b9c3bc4a13fc56abca02be4ec49`
  produced exact clean remote v6.

- 2026-07-29: remote live source-diff lookup failed as
  `github_remote_ref_dns_timeout_during_source_diff_v1`. Clean local live-ref
  attestation/provenance SHA-256 values are
  `3ef485f82678453538aef6f58ba81d548149394ef93356a811593e67cdf22e9d` /
  `780c0aa5a8a00ba9180974d4bee001782e83d747d492589a2a2da4b5bc40e2d6`;
  paper sealing is explicitly forbidden until a live remote recomputation.

- 2026-07-29: unique Job `1205569` reached `FAILED 2:0` in `00:00:28`.
  Correctness passed at maximum error `4.5300e-6`; dense/sparse-preselected/
  sparse-with-selector means were `6.1934/12.7646/13.5469 ms`, speedup
  `0.4577x`, with all rounds near `0.458x`. Signature:
  `native_grid_sparse_head_packed_gather_scatter_overhead_v1`.
  CUDA-gate/failure-analysis SHA-256:
  `7e91345babcce40bb9a157d2b29fbc718fe7f0e2a059bdc02e2edff386709197` /
  `8b49859031a48ef2a4367a156f452761c66a1e75c1a5e6a87a8fb242766f3a50`.
  Status: engineering failure; no model metric.

- 2026-07-29: a final global packed-state recovery is `designed`: share one
  cls/reg physical plan, keep raw-hole first-layer semantics, keep later hidden
  states sparse and scatter only final outputs. Below `1.0x` stops this
  implementation; the unchanged formal gate is `>=1.05x`. Official matched
  training and every paper claim remain blocked until the gate passes.

- 2026-07-29: global packed-state candidate/audit commits
  `d86a4acda21e35a1609f19f1a46bc470ee18b7e1` /
  `14bd14f9b6a087dc2ec623fc4238c89e0cb86960` were pushed and frozen as exact
  clean remote v7/v22. The implementation shares one zero-guarded global plan,
  gather indices and first-layer patches across cls/reg and scatters only final
  outputs. Linux candidate/audit suites passed `14/18`; full-content preflight
  SHA-256 is
  `08b05123edbaccd10d5b43031a43ebac11a3616ceb454bfbd588d4d7395a6a95`.

- 2026-07-29: Slurm test-only ID `1205570` is not a job. Unique formal Job
  `1205571` was submitted under immutable v4 run root. Deployment/submission
  SHA-256:
  `f070f46f023be6152faf1818342633a8d6f713fb55e37fa5c79fc2a43434f140` /
  `04d206c3ad220155f8f63a1b6a086c6c3c6c5beaeac13a7a001334f2d0fef4c7`.
  It reached `COMPLETED 0:0` in `00:00:30`. Dense / sparse-preselected /
  sparse-with-selector medians were `6.240573 / 3.129646 / 3.970906 ms`;
  selector-inclusive median speedup was `1.571574x`, with all three rounds
  above `1.5686x`. Numerical/mask/zero-scatter contracts all passed.
  Gate/completion/runtime SHA-256:
  `cddfb80af237a41d3c3e1121e39cbc5114ad8abc472c56f6daf519a50cf95988` /
  `ceec00f799eb40a1dd56c1949576783e06599205d63f1d1909a598787d99fd85` /
  `f3f4b13be3433d2307ce10a8370ab168d8af00368060e61229441e27131cb0f5`.
  Status: `tested` isolated-head engineering gate; no model or end-to-end
  paper metric. Official same-commit matched dense/sparse training is now the
  next authorized stage.

- 2026-07-29: remote live-ref source-diff recomputation passed at SHA-256
  `a07d038d87632d1f8cc984ba24af44ca7ce9a9902e30e501f5de80a32265d46b`.
  A live full rehash of 413 official I3D files exactly reproduced feature
  manifest `cda269dace393b9af1f6fcb87a9a531beed69e3c71279ba3ca2cee76e198d59a`.
  This upgrades source/data provenance for the matched training stage.

- 2026-07-29: an ad hoc repository-external data-manifest import failed before
  Slurm/model execution; its first receipt command also invoked the unloaded
  system Python. Signatures:
  `official_data_live_revalidation_import_scope_v1` and
  `preflight_failure_receipt_python_environment_unloaded_v1`. Failure receipt
  SHA-256:
  `2cd20095d49566761ed8feb16af7989d96cbe57d2b5441f10e12fa2504ababde`.
  The preserved v1 root has no model result; fresh v2 passed.

- 2026-07-29: audit commit/tree
  `643c42e8cfe4018fb891202f7ffdae554acc2e4a` /
  `25fa3eda9fc62960c69c2952c957ebab39e71c27` was pushed and frozen as exact
  clean v23. It adds an official same-commit dense/K384 paired launcher and
  independent pinned evaluator. N16R4 focused tests pass `18/18`, log SHA-256
  `f15e5d2c6b8cfeba5a31489b318f3e784233ddc8880fd09966e98e6ff63fcded`.

- 2026-07-29: test-only ID `1205572` is not a job. Unique Job `1205573` was
  submitted under fresh matched-pair root with preflight/deployment/submission
  SHA-256
  `3b827cfe10b3267d013373f89a9c3b90b2eb6f450b0aa4b7d1e5082615a0ac4e` /
  `65cb544960c619f4243c7829a41950719d2591493c05fbad70a07f1b9a037da2` /
  `ead6f35af71e2de9308d6ed0aad642dc27845e68169f1cee8ca32e3d157a3e77`.
  State: `PENDING`, route `experiment_running`. The dense and sparse arms use
  seed `1234567891`, official data/schedule, terminal epoch-35 EMA and
  independent raw-prediction recomputation. The pair is screening and
  explicitly not main-table eligible.

- 2026-07-29: Job `1205573` reached `FAILED 1:0` in `00:00:31`, before tests or
  model training. The g0024 compute node could not resolve GitHub during a
  redundant runtime `ls-remote`. Signature:
  `compute_node_github_dns_during_redundant_live_source_diff_revalidation_v1`.
  Failure/runtime/stderr SHA-256:
  `f0bf8fe6258260d55fffe88d35dfb75d647340adccb06dc2efae1c5e419c64d9` /
  `8bc85a66f37bc98eec780ec76ef5fab1978bd45195917780c269267dc5b2a057` /
  `fef8ce4b812cf04882328f4f12a5ddcac8c61077a1f8107c19b63e142808d74b`.

- 2026-07-29: audit recovery commit/tree
  `debbde469f938e09e4debfe7831e64755ae665f5` /
  `3721612aae55eecb07e9f4183a53e1d8156e143b` was pushed. It preserves the
  pre-submission live remote proof and adds network-independent, content-exact
  runtime validation. Remote tests passed `19/19`; offline validation passed.
  Their SHA-256 values are
  `d3d76af3095d792b6af0a8709a7e83addca17aa8e1d5e4d36a13b9cc8d9856f7` /
  `f409abc67b630fbc6c1b30db7ba5e614ecb8925a2d5c7aa6b0e9d7746581067b`.
  A GitHub TLS failure preserved partial v24; exact clean v25 was frozen from
  bundle SHA-256
  `6c59f1d568017d8ee82e32d3132b595b73c3d469a2cb91976968d330cd789104`.

- 2026-07-29: test-only ID `1205579` is not a job. Unique successor
  `1205580` was submitted under fresh v2 root and is `PENDING (Priority)`.
  Preflight/deployment/submission SHA-256:
  `45e60ba0f68132b8cfa11ec036ed71789e83d718dc300df62f0cdf19f1375e8a` /
  `a151cf03c67395771eb386c6fe48687e867b40df1d8f7a562be6d1df459125a0` /
  `fca38a1cad01222ef8bda967116993742319bdc94b2d8e9582a783abe21c479f`.
  Status remains `experiment_running`; no model/config/protocol change.

- 2026-07-29: Job `1205580` reached `FAILED 1:0` after 26 seconds. Offline
  source validation and focused tests passed, but official `train.py` import
  failed before any optimizer step because TensorBoard was absent. Signature:
  `official_declared_tensorboard_dependency_missing_v1`; failure receipt
  SHA-256
  `a959ef415f383d5368edf806b1166cca9cd25e91e49ea4398853775059e35385`.
  This is an engineering dependency failure with no model result.

- 2026-07-29: created isolated TensorBoard `2.20.0` venv
  `/data/run01/sczc063/yuzibo/projects/python_envs/actionformer_tensorboard_2_20_0_20260730_v1`.
  Python `3.10.20`, torch `2.0.1`, CUDA `11.8` and NumPy `1.23.5` are
  unchanged; environment receipt SHA-256 is
  `acc5909360970cfad1f390a4f5ab046a3876ac9378448b2f94da26ffb312ece2`.
  Audit commit/tree
  `a3d987961c0e6ac0166194cfc30ca0d375765ef1` /
  `51c53773d266e614d6c1054a1e6127fe73c69f38` was pushed and frozen as exact
  clean v26 from bundle SHA-256
  `e8812a84489bb55aea419b1b637778574539a44b0c7399b18a04d346430ce419`.
  Remote focused tests pass `19/19`, log SHA-256
  `f18a52300731975c81d0fffa1cd4c8e5787ccc83b07abba212d4d2a1f6fcbb7c`.

- 2026-07-29: fresh preflight/deployment/submission SHA-256 values are
  `9ff27367e10717b012d0f06a85b980f54c9b91a6fe45be9e8f87c00cac90d47b` /
  `00736c6b07fff77e0a6ca92ad24744eab0e2c089a22b350f9f2537054891b4f4` /
  `f4512010b2d675611f97e61a929ee4edda421b7f29506969d49028b3a7ac041a`.
  Test-only ID `1205583` is not a job. Unique Job `1205584` was observed
  `RUNNING` on g0024. Environment/source probes, candidate `14/14` tests and
  audit `4/4` tests passed before dense training started. Status:
  `experiment_running`, single-seed screening,
  `paper_main_table_eligible=false`; scientific conditions are unchanged.

- 2026-07-29: before Job `1205584` produced a metric, five independent reviews
  froze
  `experiments/actionformer-sparsehead-official-main-table-prereg-20260729.md`
  at status `designed`. The five paired seeds are
  `1234567891/1423812477/737690612/1788897292/1322022747`; canonical seed-set
  SHA-256 is
  `a4038a752aa46b97e5854c20574d65ece078bad6124e4778cc4269e75747c7c6`.
  S0 GO bounds are `Delta Avg>=-1.00 pp` and both high-IoU deltas
  `>=-1.50 pp`.

- 2026-07-29: paper accuracy-preserving efficiency was preregistered as all
  five paired terminal-EMA results, seed-level Avg CI lower bound
  `>=-0.20 pp`, @0.6/@0.7 lower bounds `>=-0.50 pp`, and synchronized official
  precomputed-feature detector-pipeline median speedup `>=1.05x` with lower CI
  `>1.00x` and no duration-stratum regression. A 2x2 training-support x
  evaluation-query attribution and stricter live feature/effective-config
  receipts are mandatory. No new training was submitted by this design update.

- 2026-07-29: Job `1205584` reached `FAILED 1:0` after `00:09:48`. Dense
  completed all 35 epochs and wrote checkpoint SHA-256
  `ea3c16fcf17fd6fb8cec57829804e96736a8ab231b07d820e5939fd5db3cba00`,
  but save-only EMA evaluation failed before a metric; sparse never started.
  Signature:
  `official_actionformer_softnms_extension_abi_shadowed_by_opentad_v9arg_v1`.
  Candidate official code calls seven-argument Soft-NMS, but the absent local
  extension caused Python to resolve OpenTAD's nine-argument site-packages
  module. Failure-analysis/saveonly/runtime SHA-256:
  `99f83a03715fa935a422451f9fe842aeaae867546d37c9af39cda8869958f852` /
  `496468bf5c327ae0a31a3a581cc086fd7cfb69dd5d2b249b088acc6e8aee7338` /
  `b3f8cca479ad22a674a433badcabb9d928b012af7b60221ec11f8e54e5bf6cc5`.
  Status: engineering failure; no model result.

- 2026-07-30: NMS provenance recovery was implemented at audit commit/tree
  `71f955a7301f07875a35e0be366241e548e5c775` /
  `d328093644e040741e16dbdd8bc93b6b0d608a10` and frozen as clean v27 from
  bundle SHA-256
  `a9ee267333c9371d087e806fe61cef19c14122b18fee1a4e6c75fa4c58846ad6`.
  The isolated runtime receipt/NMS extension SHA-256 values are
  `13d57c1161905f059204f7101f26029503a03da7f5eb44b81c418a0b97999f24` /
  `b67e0e41f9f55cd69e8b90cfc75a1947214365857d851a510047838ad49ed98d`.
  Candidate/audit remote focused tests passed `14+5`.

- 2026-07-30: official comparability preflight confirmed byte-identical dense
  configs (`c0ac0df...`) and the pinned upstream `validation`-train /
  `test`-eval THUMOS split. All 413 I3D files were live-rehashed with exact
  ID/content/shape/dtype equality; receipt SHA-256
  `73a2f714c100f541306d7d7f9c32e36481574d2ac6c5e78925ee4ee1dcca96b3`.
  Preflight/deployment/submission SHA-256 values are
  `d9e1f897de51e46aac52cb450f72daa8bc19a64bf999b01112013489038d4a55` /
  `b8d4079c9ddc8faa7a0a575dbe63f700c2448409df5dbccf972101cc0e4a282b` /
  `2a31a1d01056f39159d17d99fb9047f5bd6946b68475c1eae31008659df07a08`.
  `1205593` is test-only; unique Job `1205594` is `PENDING (Priority)` under
  fresh v4 root. Status is `experiment_running`; it is official-comparable
  single-seed screening and not paper-main-table evidence.

- 2026-07-30: Job `1205594` reached `FAILED 1:0` after four seconds, before
  tests/training. Signature:
  `official_environment_probe_nms_import_order_v1`. The probe imported
  `nms_1d_cpu` before `torch` loaded `libc10.so`; failure-analysis SHA-256 is
  `06bbc29e5f57b3b9a12f421f5ddd814487bf01733d0f0e5bbcc4c0551c877a41`.
  No metric exists.

- 2026-07-30: audit commit/tree
  `98f5b875315b4a2b5c6829f5d74ccce68f478e47` /
  `2e6b4bba6868c323d70c97140f7cbed044eb1a7b` adds a focused
  torch-before-NMS import-order regression. Local `5/5` passed; clean v28 was
  frozen from bundle SHA-256
  `713a1d839e8e8ea50f141df9dba1feb44dc43c91dffbd4dd85bf8910bbdf9e24`.
  Remote exact recovery is in progress; scientific conditions are unchanged.

- 2026-07-30: the remote torch-before-NMS/seven-argument probe passed; its log
  SHA-256 is
  `7d79381ed64b27059aa6f4204bbfce3f606fc1e81e0a7962e4e1d1c7413a0488`.
  Candidate/audit tests pass `14+5`. Preflight/deployment/submission SHA-256:
  `19230f06e0eda57c34607db250dba9ebc1f0d6365e5ab33c339dffe0468ddd86` /
  `250068a1de36c00fabe37596e302dc9e3fd22249be09b267fc4e9762e6f4ce46` /
  `0549ff04a30bb4efea176a484a6f51d652b8bdd023227564b0fc2fdfe492cabf`.
  `1205598` is test-only; unique Job `1205599` is `PENDING (Priority)` under
  fresh v5 root. Status is `experiment_running`, single-seed S0 only.

- 2026-07-30: Job `1205599` entered `RUNNING` on g0030. Its in-allocation
  environment/source and `14+5` focused gates passed; dense epoch 24 showed
  finite loss. No arm completion or metric exists yet.

- 2026-07-30: Job `1205599` dense arm completed from scratch and independently
  validated exact 212-video/42,400-prediction epoch-35 EMA output. Avg-mAP is
  `0.6658301251307708`; mAP@0.3–0.7 is
  `0.8190849486121916/0.7795203466370499/0.7128549836803181/0.5825550463357125/0.43513530038858167`.
  ARM completion SHA-256 is
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a`.
  Sparse training has started; no S0 verdict yet.

- 2026-07-30: Job `1205599` completed `0:0` on g0030 in `00:19:21`.
  Pair/dense-ARM/sparse-ARM SHA-256 values are
  `545e420aa1d437aedeffd15cb30390ceb0cfe4d6565d7eb35c53a8bf17ac76fd` /
  `a15b0526ef9a75a0fe32c0798b609c738781ab5c063c53df165ace6cbcdf138a` /
  `fc682cfb01b9ed6639f821938922051edc2afa55490f504170eb7e3a6fd49037`.
  Both independent official recomputations validate 212 videos and 42,400
  predictions. Dense/sparse Avg-mAP is `66.583013/43.919699`; sparse-minus-dense
  Avg/@0.6/@0.7 is `-22.663313/-25.472328/-24.218944 pp`.
  This is a legal model negative, not an engineering failure. The frozen K384
  + selected-loss intervention fails S0 and the five-seed/cost continuation is
  stopped. Status: experiment `tested`, intervention rejection
  `empirically_supported`, `paper_main_table_eligible=false`.

- 2026-07-30: six independent reviews completed the initial Pro-level negative
  analysis. They rank combined native-query coverage loss and selected-loss
  supervision/EMA-normalizer drift above calibration/Soft-NMS interaction and
  implementation/evaluator defects. The next authorized experiment is the
  preregistered no-retraining 2x2 checkpoint cross-eval, followed by
  per-class/duration/boundary/retained-recall and assignment/support
  diagnostics. No method rescue or retraining was authorized.

- 2026-07-30: Slurm Job `1205701` completed `0:0` and closed the frozen
  no-retraining 2x2 attribution. Full×dense/full×K384/selected×dense/
  selected×K384 Avg-mAP is
  `66.583013/45.784332/64.537343/43.919699`. The K384 execution main effect is
  `-20.7082 pp`, selected-loss training `-1.9552 pp`, interaction
  `+0.1810 pp`. Attribution/diagnostics/suite SHA-256 values are
  `d0bffe87cfb582b1b0649da3833e9fe0147db5a0a78500b6700fb78019323afb` /
  `a6b7fa0c4a41aac75ae2fb4cb4fcfbe68cf48bc7d2c813b37485b35998838791` /
  `e71721cb07334f1b6abb09347a7b609e51d6da1ed4be864c190ed60433a197d6`.
  Status: `tested`, diagnostic-only.

- 2026-07-30: assignment/support implementation commit/tree
  `465b2bc284d5c3b62ec9e21023052b5eabddf260` /
  `da1e515398017345deb4c39d98751ade0a8aa8db` passed remote focused gates.
  Formal Slurm Job `1205799` completed `0:0` in `00:00:41`; suite/producer
  SHA-256 values are
  `475b61ddad4b0b56a86b2e2616ef2584b252c3169b4ad1268223f21d6e118567` /
  `ca7e97a4124e49eb2ac30e949bcd50d4407998e8518eb72c8c6c8c8bb3f86e8b`.
  Across 64 deterministic official-training windows, K384 retains
  `461/2721` positives, leaves `395/804` GT without candidates and `427/804`
  without assignments. No test GT or training is used. Status: `tested`,
  diagnostic-only.

- 2026-07-30: final Pro synthesis ranks structural proposal/query/support
  deletion first, selected-loss/high-IoU optimization second,
  calibration/NMS third and implementation/evaluator defect fourth. Evidence
  streams are mutually consistent. The exact hard K384 formulation remains an
  `empirically_supported` rejection.

- 2026-07-30: the only continuing SparseHead design is frozen as DCSR
  (dense cheap scaffold + sparse expensive residual refinement). It preserves
  dense proposal and supervision support, keeps unselected scaffold outputs,
  and requires per-FPN floors. Status: `designed`; no implementation or result
  exists. The official preregistration requires internal validation-only
  selection, five fixed paired seeds, same-run dense controls and complete
  feature-to-final-detection cost before any main-table efficiency claim.

- 2026-07-30: source/receipt integrity audit found real dataset GT, official
  ActionFormer THUMOS evaluation, no prediction self-normalization and no
  phantom metric. Overall audit remains `WARN` because evidence scope is one
  official paired seed plus diagnostics and all configured external
  cross-model file-review routes were unavailable. This limitation and all
  diagnostic/main-table boundaries were recorded explicitly.

- 2026-07-30: after Jobs `1205599`, `1205701` and `1205799`, all final receipt
  checks, negative analysis and DCSR preregistration completed, heartbeat
  `sparsehead-official-matched-monitor` was set to `PAUSED`. No SparseHead GPU
  job remains active or authorized by the closed hard-K384 protocol.

- 2026-07-30: implemented DCSR G0/G1 on
  `codex/actionformer-dcsr-g0-g1-20260730`. Final exact commit/tree is
  `bf0df83d7400c89fc61f38d169d68085420a2263` /
  `2f9346fcfd2bfb7fc5a76a86ef65545030a67469`; clean N16R4 focused and module
  entry suite passed `31 passed`. G0 is full-official-head identity only; G1
  is one-layer dense scaffold plus uniform K384 signed residual refinement
  with full-grid supervision. Status advanced from `designed` to
  `implemented/tested`.

- 2026-07-30: preserved two distinct zero-metric engineering failures.
  Job `1206160` failed before model execution because ad-hoc Slurm `--wrap`
  used a non-Bash shell and unexported local paths
  (`slurm_wrap_shell_and_unexported_variable_scope_v1`). Job `1206166`
  entered the checked-in launcher and passed `27` focused tests but direct
  `python tools/...` invocation lacked repository import scope
  (`python_script_repository_import_scope_v1`). Old v3/v4 roots remain
  immutable; neither is a model result.

- 2026-07-30: launcher-only recovery Job `1206168` completed real-CUDA G0
  `0:0` on RTX 4090. Receipt SHA-256
  `b87fc59ec6529e83e99f7bf5fbfb7f3bff5ec637060c62057da07a669a8c1ff4`
  confirms exact state keys, points, full masks, pre-decode logits/offsets and
  final official Soft-NMS/timestamps, with no test GT/predictions and no
  metric/efficiency claim.

- 2026-07-30: submitted formal validation-only DCSR G1 Slurm array
  `1206273_[0-2]`; `1206266` is test-only. Run root is
  `/data/run01/sczc063/yuzibo/projects/phystime_tad/runs/actionformer_dcsr_g0_g1_internal_20260730_v5`.
  The frozen manifest SHA-256 is
  `ba683bc5ddbb1fe219fab0545e9d808808d9b25fc9b32e7c5c0b6339b68b9bbb`
  over 160/40 official-validation videos. Status is `experiment_running`.
  Internal results cannot be paper rows; official five-seed and complete-cost
  work remains gated on G0--G4 freeze.

- 2026-07-30: DCSR G1 array `1206273_[0-2]` completed for all three frozen
  development seeds. Aggregate SHA-256
  `b98d59468ef39aa6fe6de387adfd6f872c848ab8f63b26c3bf1bf6161f5f7939`.
  Mean DCSR-minus-dense is `-7.556202 pp` Avg and
  `-11.043134/-11.019821 pp` at 0.6/0.7; every seed/threshold is negative.
  The preregistered G1 gate is false. Status moved from `experiment_running` to
  `tested`, and the exact rejection is `empirically_supported`.

- 2026-07-30: implemented and pushed no-training DCSR negative diagnostics at
  commit/tree `8d6f6e5e7fcf8c27b6aa46870bc4c0b242f6314b` /
  `1ac5a68c6b8d0b1c9028ea3154765ae20e87622a`; clean Linux suite passed
  `38 passed`. Three pre-Slurm deployment roots were preserved:
  `diagnostic_deployment_inline_ssh_quoting_v1`,
  `diagnostic_deployment_nonlogin_module_function_v1`, and
  `diagnostic_deployment_profile_under_nounset_v1`. None is a model result.

- 2026-07-30: counterfactual tasks and aggregate Job `1207441` completed
  `0:0`. Completion/prediction/checkpoint SHA-256:
  `954d7944428fcf0d26dd917ff9562a9c3e7a53de71c09e9a382aaf49f5bd4a53` /
  `47dcca7e179544e348966bf92cf92cddeff19a1fdc8cfea100150dc1bc580a36` /
  `c596bc942d2617e3824d21c96d0289316be4ee1ad465f23dc507b2d90466e006`.
  Scaffold-only is `-7.418076 pp` versus dense; all-query residual remains
  `-6.316665 pp`; K384 adds a `-1.239537 pp` support penalty. The leading
  observed cause is the weak scaffold/decomposition, not random seed or
  selected-only supervision.

- 2026-07-30: completed Pro-level competing-explanation analysis and integrity
  audit. Scientific integrity is `PASS`, official paper comparability is
  `FAIL`, overall audit is `WARN`; external cross-model reviewer routes were
  unavailable due credentials. The current DCSR/SparseHead route terminates at
  G1. G2--G4, official five seeds and cost expansion are not authorized. A
  possible official-quality dense proposal floor with conditional residual
  compute remains only `discussed` and needs a new preregistration.

- 2026-07-30: after all formal G1 results, no-training diagnostics, checkpoint
  analysis, audit and claim-boundary records completed, heartbeat
  `sparsehead-official-matched-monitor` was set to `PAUSED`. No SparseHead/DCSR
  GPU job remains active or authorized under the terminated protocol.

- 2026-07-31: froze a separately named ODF-CR successor at design commit
  `codex/actionformer-densefloor-factorial-20260731@77244d5`. The internal
  decision is depth `1/3` × residual `off/all_valid`, followed only by
  frozen-checkpoint K384 replay. Two independent specification reviews added a
  holdout-v2 set-membership contract, seed/generalization boundary, exact CUDA
  identity details, paired-delta definitions and deterministic replay
  allocation. Status is `designed`; no implementation, Slurm job, metric or
  paper claim exists yet.

- 2026-07-31: implemented and deployed ODF-CR at exact commit/tree
  `01cdb78d2b7668098b6b13a1e49433d48fbc1a8d` /
  `e70d2956a197b1204e721239178e76152efe282b`. Linux focused suite passed
  `71`; holdout-v2 is validation-only 160/40 with SHA-256
  `b8cac555f3d31e02468dbca3b3b0ada2d30b05bf046c10eb16304abb92499d1a`.
  Formal array `1209259_[0-2]` is running and all three real-CUDA G0 receipts
  pass all 14 exact identity/initialization checks. G2 Job `1209267` is the
  unique `afterok:1209259` successor. Pre-Slurm/profile and YAML coercion
  failures plus the recovered missing-GPU G2 submission are preserved as
  engineering-only signatures; no main job was duplicated. Status is
  `experiment_running`; no arm metric or paper claim exists.

- 2026-08-01: ODF-CR array tasks `1209259/1209260/1209261` and G2 Job
  `1209267` completed `0:0`. Three matrix receipts and aggregate
  `9172eddcbf5f9a4943b303e20b57f4492f0a44b18c39f892d5829b1f0a79ddec`
  validate clean frozen identity. `d3_all-d3_off=-0.1806 pp` Avg, only `1/3`
  seeds positive and @0.6 `-2.7468 pp`; G2 is false. No K384/G3 was submitted.
  Status moved from `experiment_running` to `tested`, and the exact
  residual-utility rejection is `empirically_supported`.

- 2026-08-01: completed multi-agent and raw-prediction terminal attribution.
  `d3_off-d1_off=+7.5600 pp` Avg with high-IoU recall/boundary gains;
  `d1_all-d1_off=+3.8689 pp`; interaction is `-4.0496 pp`. `d3_all` lowers late
  training loss without holdout gain and has mixed class/duration/video effects.
  Saturation/overfit and ranking interference remain competing explanations;
  calibration/NMS and gradient conflict are not identified. The all-valid
  residual route stops, the official dense floor remains a future prerequisite,
  and sparse conditional execution is not universally rejected.

- 2026-08-01: retired heartbeat monitor
  `sparsehead-official-matched-monitor` after terminal attribution and claim
  tracing. The Codex app self-delete RPC timed out repeatedly, so the exact
  configuration was recoverably moved out of the active automation directory.
  No Slurm job was inspected, submitted or cancelled during this fallback.

- 2026-08-11: fresh serial DUCA Project Pro review `PRO_INITIAL_REVIEW-v002`
  was route-audited and confirmed in the Project Sources with
  `CURRENT_RESEARCH_STATE-v002` and `MODEL_EXPERIMENT_HISTORY-v002`. Its sole
  scientific decision is `REVISE`: replace the confounded detector-changing
  primary route with a minimal fixed-K bounded monotone physical-density
  acquisition candidate, external pre-NMS coordinate transport, and a clean
  unchanged-detector uniform control. Status remains `BLOCKED_PRE_RESULT`;
  no GPU/Slurm task, metric, cost, claim, or result was generated. Builder,
  Critic, and Evaluator return work is pending under the accepted decision.
- 2026-08-11: the accepted `PRO_INITIAL_REVIEW-v002` P0/P1 work has been
  dispatched through the three independent CVPR Pro Lab queues: Builder
  `msg-20260811T050534Z-931161b9db78`, Critic
  `msg-20260811T050549Z-b54624740c6c`, and Evaluator
  `msg-20260811T050602Z-5ea4bb06fd29`. These are implementation-fidelity,
  frozen-route attack, and pre-registration tasks only; no local or remote
  model execution, GPU/Slurm job, metric, or claim has begun.
- 2026-08-11: Builder briefly created three exploratory subagents, which
  violated this project's fixed three-role process boundary. They were
  immediately interrupted (`registry_probe`, `coordinate_probe`, and
  `config_test_probe`); their outputs are quarantined and will not inform the
  implementation. Builder was instructed to continue personally. No code,
  experiment, metric, or scientific decision resulted from this deviation.
- 2026-08-11: Evaluator return `EVALUATOR_DUCA_DENSITY_P0P1-v001` was
  received as preparatory preregistration only: P0 is blocked, P1 was not run,
  and P2 is not PRE_RUN_READY. Builder and Critic P0/P1 turns exceeded their
  bounded duration and were interrupted without durable returns; their
  unsealed partial work is non-evidence. No experiment, GPU/Slurm job, metric,
  or claim was generated.
- 2026-08-11: the fresh P0 Pro request was fail-closed before submission:
  Oracle could not attach to the verified iXBrowser CDP endpoint and resolved
  the requested model to a non-verified Sol tier. The shared browser profile
  also contained other-project tabs/turn locks, so no direct-CDP UI fallback
  was attempted. No Pro response or scientific decision was created; status is
  `BLOCKED_PRO_MODEL_ROUTE` pending a verified serial Pro transport.
- 2026-08-11: a later fresh direct-CDP DUCA Project turn
  `duca-p0-blocker-51c88fd75537120ce96a417beb7e81dd` was recovered, identity
  audited, preserved verbatim, and accepted as `PRO_P0_BLOCKER_DECISION-v001`.
  It matched Project `g-p-6a796fef9a00819194024cf1de3bd697`, nonce, and fixed
  commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, with no other-project
  material. The Pro decision is `REVISE`: P0 is a no-execution correctness
  repair for the 766/767 uniform endpoint disagreement and pre-NMS coordinate
  transport. Builder and Evaluator receive bounded static/protocol work;
  Critic waits for the Builder diff. P1/P2, data, CPU/GPU, Slurm, metrics,
  Git push, result promotion, and claims remain blocked pending fresh Pro
  admission. Evidence remains `BLOCKED_PRE_RESULT`.

- 2026-08-11: Builder plan `BUILDER_DUCA_P0_MINIMAL_CHANGE_PLAN-v002` and
  independent Critic return `CRITIC_DUCA_P0_PLAN_AMBIGUITY_CLASSIFICATION-v001`
  found that canonical-uniform and pre-NMS coordinate transport are deterministic
  P0 corrections, but no frozen positive density input or inverse-CDF hard
  decoder exists at commit `63a726a4`. The missing mechanism is classified
  `SCIENTIFIC_AMBIGUITY`, not a patchable interface defect. No code, test,
  remote job, dataset access, metric, cost, or claim was produced. A fresh
  serial Pro decision is required before P0 implementation resumes.

- 2026-08-11: Fresh Project Pro response was received through the verified
  standalone remote-CDP Oracle route and accepted locally as
  `PRO_P0_ROUTE_ADJUDICATION-v002` (`REVISE`). The prompt explicitly appointed
  Pro as the acting Scientific First-Author Agent, Primary Research Owner and
  scientific publication planner; it owns scientific route, experiment design,
  stop/revise decisions, claim scope and paper narrative, while legal authorship
  and submission remain human responsibilities. Pro resolved the P0 density
  ambiguity with `DUCA_FIXEDK_BOUNDED_DENSITY_QUANTILE_ACQUISITION-v002`, a named
  density-only reader plus constrained inverse-CDF decoder, and retained the
  canonical-uniform and pre-NMS coordinate changes as claim-neutral corrections.
  This is a designed, no-execution route: no patch, test, data access, CPU/GPU,
  Slurm job, metric, cost, result or claim was generated. The decision and
  updated current-state/history files are prepared locally for centrally leased
  Project-Source synchronization; they are not yet remotely confirmed.

- 2026-08-12: Fresh Project Pro turn `duca-projection-policy-20260812-48a111ed75674967` was submitted in the exact DUCA Project and completed through the verified Oracle remote-CDP route. The Project ID, fixed commit `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, required v005 Sources, fresh nonce and conversation binding were matched; no other-project material was present. Its verbatim response is preserved at `.cvpr-pro-lab/pro-reviews/runs/duca-projection-policy-v001/raw-response.md` and accepted as `PRO_P0_PROJECTION_POLICY-v001` with `CONTINUE`. Pro froze the exact nonconstant constrained-integer projection and requires cross-implementation identity for equal serialized projection inputs. This is a scientific-definition intake only: no tests, data access, CPU/GPU/Slurm job, metric, cost, result, Git push or claim was produced. Builder may only return the next authored-not-run file/symbol plan; Critic and Evaluator have no execution authority.

- 2026-08-13: Fresh Project Pro decision `PRO_P0_IDENTITY_GATE_AUTHORIZATION-v001` accepted `CONTINUE` for exactly one finite cross-implementation identity/optimality gate over the frozen integer projector. It freezes canonical integer `(T,K,u,a,Q)` inputs, a closed positive/negative/certificate-mutation matrix, independent Evaluator reference optimality witnesses, and first-discrepancy fail-closed behavior. It changes no model, detector, loss, split, metric, NMS, requested budget, evaluator or paper claim. The authorized evidence chain is Evaluator normative package → Builder production interface/receipt → Evaluator comparison receipt → Critic closure. No TAD training, data access, metric, cost, checkpoint or performance result has been produced; status remains `BLOCKED_PRE_RESULT`.

- 2026-08-13: The bounded Builder return `BUILDER_DUCA_P0_TYPED_FAILURE_INTERFACE_CORRECTION-v001` authored, without execution, the production `DUCAProjectionError.code` mapping and one focused negative-code regression. It preserves the frozen projector mathematics, Q, feasible set, candidate ordering, objective, tie rule, detector/data/metric contract and paper claim. This is an `IMPLEMENTATION_CORRECTION`, not a P0 identity or performance result; the next dependency is a single independent Critic recheck of the exact diff.

- 2026-08-13: The independent focused Critic recheck
  `CRITIC_DUCA_P0_TYPED_FAILURE_INTERFACE_FOCUSED_RECHECK-v001` closed the
  typed-failure correction for the next P0 dependency: it found no remaining
  implementation correction and no scientific ambiguity, and confirmed that
  C-PROJ-001 preserves the frozen projection mathematics. This is still static,
  unexecuted evidence (`BLOCKED_PRE_RESULT`), not an identity, optimality,
  performance, cost or paper result. The sole next dependency is a bounded plan
  for a clean revision-bound execution snapshot and production receipt interface
  for the already authorized P0 gate.

- 2026-08-13: `BUILDER_DUCA_P0_IDENTITY_EXECUTION_SNAPSHOT_PLAN-v001` was
  accepted as a bounded, no-execution preparation plan. It preserves the exact
  P0 projector contract and restricts the next operation to a clean execution
  commit containing only the two Critic-reviewed production/test paths. No
  production invocation, reference execution, comparison, data/model access,
  metric, cost, GPU/Slurm work or claim was performed. The gate remains
  `BLOCKED_PRE_RESULT` pending that clean snapshot and the separately frozen
  comparison sequence.

- 2026-08-13: Builder completed the bounded clean snapshot preparation as
  `BUILDER_DUCA_P0_IDENTITY_CLEAN_SNAPSHOT_PREPARATION-v001`. Commit
  `df8228072b871adbd8dedb480e80f1f7daaca69e` has pinned parent
  `63a726a4aaf48ecbf6780bb196de43a890c6b4df`, exactly one reviewed production
  modification and one reviewed focused test addition, and a clean worktree.
  This establishes only a potential formal execution snapshot. Fixture bytes,
  independent reference expectations and production output remain uncreated;
  no projector/reference/comparison, data/model, metric, GPU/Slurm or performance
  evidence was executed. P0 therefore remains `BLOCKED_PRE_RESULT`.

- 2026-08-14: A fresh exact-Project GPT-5.6 Sol Pro review independently
  returned `REVISE` and froze only
  `DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION-v001` (Candidate B) at
  **plan-only** status. The potential contribution is not generic dynamic
  sampling: an outer discrete heavy-frame budget must react to paired
  boundary/high-IoU risk, while a K-independent physical exact-K transport
  controls placement before unchanged official NMS. Candidate A remains a
  paused/negative historical control and Candidate C is not an automatic
  fallback. The result is designed-only: O1 dynamic-oracle headroom, O2
  physical exact-K transport, O3 hard-utility predictability, and O4 risk
  superiority over actionness/transition/no-risk/K-shuffle are hard kill gates.
  The only potentially next sequence is no-code Builder plan authoring,
  independent Critic static review, then Evaluator structural intake with
  `PRE_RUN_NOT_READY`; data, official validation/test, execution, N16R4,
  metrics, cost and claims remain forbidden. The full independent response is
  preserved at `C:/Users/skywalker/.fastctx/jobs/j-n1xm7e/output.log`.

- 2026-08-14: The one authorized independent static review of the Route-B
  plan returned `DYNAMIC_ROUTE_B_PLAN_BLOCKED`, with no correction or recheck
  authorized. The general midpoint-CDF-plus-lexicographic decoder selects
  `(0,2,...,766)` at `T_v=768,K=384`, whereas its canonical-uniform identity
  separately requires `(1,3,...,767)`; hence the future fixed-K and
  constant-density paths lack a unique common contract. In addition, the
  K-shuffle control did not freeze its nonce, canonical row order, or
  permutation algorithm, so its content-to-K allocation cannot be reproduced.
  These are deterministic plan-level negative findings, not performance or
  cost results. Route B is terminally held; A/C/old U/O/R remain unavailable as
  automatic substitutes, and the Evaluator stays dormant. Receipt:
  `.cvpr-pro-lab/critic-returns/CRITIC_DUCA_DYNAMIC_ROUTE_B_PLAN_STATIC_REVIEW-v001.md`.

- 2026-08-14: The Route-B Critic terminal closes the role correction loop, but
  not the user-required dynamic-budget scientific question. Its two deterministic
  findings are now preserved as the final Pro adjudication boundary: one common
  constant-density/canonical-uniform identity must resolve the midpoint-CDF
  lexicographic tie at `T_v=768,K=384`, and K-shuffle must freeze nonce/seed,
  canonical row order, permutation algorithm, stratum, and invalid-input
  behavior. The final material packet asks fresh exact-Project Pro to freeze both
  definitions or REVISE/PIVOT/STOP the route. Fixed K remains a baseline/control/
  fallback only. The official THUMOS14/OpenTAD-AdaTAD six-arm FIT/CAL/HOLD,
  O1--O4, and N16R4 protocol remain designed-only. No code, role action,
  PRE_RUN, data, pilot, training, metric, cost, or claim was produced.

- 2026-08-15: Final fresh exact-Project Pro (`DUCA-DYNAMIC-INNER-FINAL-v004-20260814T203230Z`)
  returned `REVISE` on the authoritative v013 pair only; the prior v008 epoch stays quarantined.
  It freezes dynamic outer K plus B=`DUCA_HIERARCHICAL_DYNAMIC_PHYSICAL_ACQUISITION_B-v001` as
  the sole claim-bearing candidate: a positive-density bounded monotone/local exact-K physical
  transport with physical-time mapping before every suppression/output operation. A remains only
  the timestamp-aware enhanced-integration freedom ceiling under the same outer K; it cannot be
  promoted if B is falsified. F1 is the endpoint-inclusive integer-half-up uniform generator; F2
  is nonce-derived seed + canonical row order + Fisher--Yates K-shuffle. The prospective
  FIT/CAL/HOLD six-arm/O1--O4/cost/pilot-stop protocol is frozen, but all evidence remains
  `STATIC_PLAN_ONLY / NOT_EXECUTED`: no new DUCA identity, implementation, PRE_RUN, data,
  remote/GPU/Slurm, pilot, training, evaluation, metric, cost, novelty, or efficacy evidence.
  Full original response: `C:/Users/skywalker/.codex/oracle/duca-dynamic-inner-final-v004/final.md`.

- 2026-08-15: Under the later user-authorized downstream continuation, a new clean DUCA worktree
  was created from `63a726a4...`, explicitly excluding dirty SparseHead `a6bdc084...`. Builder's
  clean snapshot `9eb328f9...` implements only the frozen B route's static surface: dynamic outer-K
  selection, bounded monotone/local physical exact-K transport, F1 canonical-uniform and F2 shuffle
  metadata, a six-arm FIT/CAL/HOLD declaration, and a fail-closed future N16R4 PRE_RUN launcher.
  Static syntax/contract checks are `infrastructure_evidence` only; no data, held-out, inference,
  training, evaluation, metric, cost, GPU, Slurm, remote action, or claim occurred. One independent
  read-only Critic review of that exact clean snapshot is the active dependency; Evaluator is dormant.

- 2026-08-15: The independent Critic reviewed clean `9eb328f9...` and returned `DYNAMIC_ROUTE_B_STATIC_BLOCKED / IMPLEMENTATION_CORRECTION`, not a scientific ambiguity. The dynamic-B helper can fall back to `candidates[:K]` after a locality failure and treats zero local radius as unbounded, contradicting the frozen bounded monotone/local physical exact-K mechanism; its dynamic-B metadata also claims F2 shuffle without executing it. This is a deterministic implementation defect, not efficacy evidence. One focused correction is authorized: contract-preserving or fail-closed locality behavior, F2 execution/metadata reconciliation, and targeted static tests, followed by one focused recheck by the same Critic. Evaluator remains dormant; no data/held-out/PRE_RUN/remote/GPU/Slurm/pilot/training/evaluation/metric/claim occurred.

- 2026-08-15: The single focused Builder correction at `3e551595...` removed the prefix fallback, made zero radius fail closed, and marked dynamic-B F2 as not executed; nine static tests passed. The same Critic's sole recheck nevertheless observed `bounded_monotone_local_exact_k([0,1,0,1], K=2, radius=1) -> [0,3]`. Anchor choices bypass nonzero locality validation, so the B claim path still breaches its bounded/local physical exact-K definition. This is the second equivalent deterministic implementation defect, so the one-correction loop is terminally exhausted: `DYNAMIC_ROUTE_B_FOCUSED_STATIC_BLOCKED / PRE_RUN_NOT_READY`. The frozen scientific candidate has no efficacy evidence; Evaluator remains dormant and no data/held-out/GPU/Slurm/remote/pilot/training/evaluation/metric/cost/claim action is permitted. A scientifically distinct replacement would require new explicit authority.

- 2026-08-16: Experiment-first continuation replaced the subset/pilot-first plan with the user-required full official THUMOS14 matrix. Dynamic outer K remains the paper core; fixed K remains baseline only. The new clean implementation makes K change the real VideoMAE frame count, carries arbitrary selected frames in original physical-time order, and reconstructs the unchanged 768-step detector axis after the backbone. The full matrix is official-dense / uniform-k384 / learned-k384 / dynamic-k / dynamic-k-no-risk × seeds 3407/3408/3409 × 60 epochs. All 15 first-epoch cells reached real model/data loading and then stopped on one shared path-with-space parsing failure; the same official videos were rebound through the established no-space video root. The first corrected dynamic-k submissions then exposed a pre-optimizer 16-frame clip-alignment defect when valid-count clamping produced a non-multiple-of-16 K. Commit `06b02be1…` floors the valid count to the largest complete clip before clamping and passed the remote regression. N16R4 job `1239627` (seed 3407) is now in epoch 0 with at least 50 finite-loss optimizer updates; `1239628` (seed 3408) is queued for GPU. No official mAP/cost conclusion exists until the complete matrix finishes.

- 2026-08-16 14:03: corrected dynamic-budget job `1239627` reached epoch 4 with finite detector and selector losses. Submitted queue now also includes dynamic seeds 3408/3409, official dense seed3407 and uniform-k384 seed3407 (`1239638`). The shared N16R4 account cap is the only current scheduling constraint. No official validation or cost result exists yet.

- 2026-08-16 14:20: dynamic-budget seed-3407 job `1239627` reached epoch 6 and wrote an epoch-5 checkpoint with finite losses. The account slot released by a completed ZoomToken baseline was immediately used for `dynamic_k_no_risk` seed 3407 as job `1239646`; this is the direct control for the boundary-risk term. A subsequent learned-k384 submission was rejected by `AssocMaxSubmitJobLimit` before job creation and remains next in queue. No mAP or cost conclusion exists before official validation.

- 2026-08-17: completed the project-owned full DUCA memory and read-only resource audit in
  `research-wiki/DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`. The restored scientific contract is
  binary action/background scout plus boundary-importance prediction followed by deterministic
  acquisition; it is not a direct learned-index policy. Dynamic outer K remains the required
  candidate core, while fixed K is baseline/control/fallback. The audit preserves the old MUST,
  CellCF, selected-axis/direct-controller, physical-grid and 80/90-epoch negative or
  protocol-mismatched records instead of promoting them to current evidence. Official AdaTAD S
  baseline source/config/evaluator bindings are identified, but the local official reference has
  no data/pretrained/exps and a read-only N16R4 attempt cannot resolve `yuzibo`; resources are
  therefore not independently ready. No model code, data, remote, GPU, Slurm, evaluation,
  metric or claim action occurred. Future full-run recovery policy is 5 epochs by default,
  retaining a more frequent untouched official interval (AdaTAD S remains 2 epochs), with final/
  final-EMA selection fixed and recovery-only checkpoints.

- 2026-08-17: corrected the audit's N16R4 resource classification from Central's verified
  read-only inventory. The earlier `ssh ... yuzibo` failure was an undefined local hostname alias,
  not evidence that remote data were absent. Under shared root
  `/data/run01/sczc063/yuzibo`, THUMOS14 raw video is COMPLETE (411 valid MP4 symlinks, 200
  training + 211 validation, zero broken, targets about 33G); its annotation/category map and the
  VideoMAE-S pretrain are also COMPLETE. `/data` has 3.1T free of 5.3T. I3D and InternVideo2
  OpenTAD feature paths remain absent; the empty alternate I3D download, incompatible SigLIP2,
  and native MATR pickles/checkpoint fragment must not be substituted into the raw-video AdaTAD
  baseline. The data/storage blocker is therefore removed. Remaining first-experiment blockers
  are strictly the released AdaTAD-S checkpoint's readable, complete, config-compatible binding
  and a clean DUCA-owned official code/config/evaluator/runtime/Slurm PRE_RUN binding. No data,
  code, config, remote, GPU, Slurm, evaluation, metric, or claim action occurred.

- 2026-08-17: added the central read-only remote video resource map to
  `DUCA_WIKI_MEMORY_AUDIT-2026-08-17.md`. It freezes THUMOS14 canonical use to the 411
  annotation-mapped symlinks (200 training + 211 validation), explicitly excludes two extra
  physical test files, and records availability boundaries for MultiSports, TOC-Bench, Charades,
  ActivityNet, FineAction/HACS/EPIC-Kitchens/Ego4D, and EventMATR feature assets. Only THUMOS14
  is bound to the current official AdaTAD baseline. No remote mutation, download, extraction,
  symlink, config/manifest change, training, evaluation, metric, or claim action occurred.

- 2026-08-17: persisted project-owned remote bindings in
  `research-wiki/REMOTE_DATA_RESOURCE_MAP-2026-08-17.md`. It records the correct N16R4 FQDN and
  shared root; THUMOS14 canonical 411-video binding, physical-store exclusions, annotation/map and
  VideoMAE-S paths; exact feature/checkpoint absences; dataset isolation; and the 5-epoch minimum
  resumable-checkpoint contract. It separately records that the AdaTAD-S released checkpoint path
  remains unverified and therefore blocks the first checkpoint evaluation, while THUMOS data itself
  does not. Role-specific durable notices were written for Builder, Critic, and Evaluator. No role
  was woken; no remote mutation, download, extraction, link, config/manifest edit, training,
  evaluation, metric, or claim action occurred.

- 2026-08-17（用户共享基线更正）: 原始、未修改 AdaTAD 的 THUMOS14 checkpoint evaluation
  与必要时的 source training 被定义为 ZoomToken 负责人唯一一次共享 official baseline；DUCA 不得
  重复这项 evaluation 或 training。共享 receipt 返回前 official dense 数字留空，历史 66.xx、
  matched-source 数字和此前任务状态均不得替代。这个外部数字依赖不阻塞 DUCA 非重复轨道：在 clean
  DUCA worktree 中恢复“0/1 actionness + boundary-importance scout → deterministic indirect
  acquisition → dynamic outer-K”，direct index policy 仅作 ablation，fixed K 仅作 control/fallback。
  Builder 接下来只做最小实现、六臂 config/launcher 和五 epoch recovery contract；随后独立 Critic
  与 Evaluator PRE_RUN。无 checkpoint evaluation、官方训练、数据访问、GPU/Slurm、指标、成本或
  efficacy claim 在本记录中发生。

- 2026-08-17（DSH external review）: 按 frozen anchored-standard / deepseek-official /
  deepseek-v4-pro / max 的三次 DUCA DSH 会话均在 completed 前因 `402 Insufficient Balance`
  终止；receipt 为 `C:/Users/skywalker/.codex/central/cvpr-pro-lab/dsh-reviews/DSH_DUCA_ZOOM_REVIEW_RECEIPT-2026-08-17.md`。
  因无 completed turn，DSH 没有产生项目报告、code-review verdict、实现输入或科学裁决，不能称
  DSH PASS。旧 dynamic-B focused recheck 的 physical-locality defect 仍是该旧实现的
  `PRE_RUN_NOT_READY` 负证据，不否定 semantic-indirect dynamic-K 科学方向；当前独立 Critic
  正在审查新的 clean semantic package。无 DSH 替代、重试、实验或性能结论发生。

- 2026-08-17（semantic-indirect structural intake）: clean semantic package 的一次 focused
  correction/recheck 后，结构 PRE_RUN intake 仍发现两项同类可达性问题：`forward_train` 未把
  training GT 送入 semantic selector/auxiliary-loss 路径；dynamic-K metadata 以整包 dict 写入，却
  被下游按 batch index 读取，不能形成逐视频 requested/effective/executed-K receipt。物理坐标、
  PRECHECK-only fail-closed launcher 和 recovery 字段虽存在，但这两项使 package 保持
  `PRE_RUN_NOT_READY`。按一轮 correction/recheck 边界，不再开始第三次 Builder 修复或 Critic 重审。
  这不是 dynamic-budget 科学方向的效能否定，也没有数据、GPU/Slurm、训练、推理、评估、指标、成本或
  claim；若继续，必须先取得 STOP 与真正简化替代之间的外部科学处置。

- 2026-08-17（semantic-indirect intake 更正）: 对冻结 clean commit `6125654...` 直接复读
  `PCOTMRASPreBackboneFrameSelector.forward_train` 确认：它把 `gt_segments` 传给 `_select`，并传给
  `_losses` 以构造 training-only actionness/boundary targets。因此上一条关于“GT 未接入”的角色报告
  不保留为事实 blocker，也不据此声称 package 已通过。仍可重现的结构 blocker 是
  `_semantic_budget_from_predictions` 返回整批 `requested_k/effective_k/executed_k` 字典，而 metadata
  writer 以 `dynamic_budget_meta[idx]` 当作逐视频字典读取；这不能封存每视频动态预算实际值。该缺口
  属于同一 dynamic-budget execution-contract 类别；一次 correction/recheck 已耗尽，故保持
  `PRE_RUN_NOT_READY`、不启动第三次修正、PRE_RUN、数据或远端实验。DSH 三次会话均因 402 在 completed
  前终止，亦不构成 PASS、审查裁决或替代此处置的科学输入。

- 2026-08-17（完成 DSH 外部只读审查）: 新空白 session
  `session-70ec494d-3bf8-46a1-b45b-0162827e5e00` 以 anchored-standard / deepseek-official /
  deepseek-v4-pro / max / 256000、Minimal 首工具 `bash`+`str_replace_editor` 完成。唯一验收
  指纹 `/^We need\b/` 在第一条非空 reasoning 自然命中，`turn/end.kind=completed`；原始 session、
  任务、stdout 和项目回执在 `docs/dsh/DUCA_DSH_OWNER_REVIEW_RECEIPT-2026-08-17.md` 登记。外审认可
  “0/1 动作性与边界重要性预测 -> 确定性间接 acquisition -> dynamic outer-K”的科学方向，但以
  `FAIL / PRE_RUN_NOT_READY` 否决当前 `6125654...` 实现进入实验：batch-level dynamic-budget dict 被
  逐样本索引会导致前向失败；heavy backbone 仍消费 384 行 padding；physical-time inverse mapping 在
  NMS 后；六臂仅为部分占位/同构/不可运行声明。再次确认 `forward_train` 的 GT 传递不是 blocker。
  这是外部代码审查输入，不是 efficacy、数据、GPU、训练、评估、成本或论文结论；下一步必须先由
  Builder 提交最小修正包，再经独立 Critic 与 Evaluator PRE_RUN，不得跳过这些门。

- 2026-08-17（round-2 frozen snapshot 的 DSH+Critic 归一化）: clean candidate
  `90748a1e46efbed760401edf80cec7c5816af0b4` 在同一科学路线下完成最小实现尝试。独立 Critic 和
  新鲜 DSH session `session-ae79794b-5e51-4a69-b065-6d09d9ec4cb5` 均给出 implementation `BLOCKED/FAIL`。
  DSH 的 header、自然首行 `/^We need\b/` 与 completed 终态已在
  `docs/dsh/DUCA_DSH_ROUND2_CODE_REVIEW_RECEIPT-2026-08-17.md` 记录；原始 session 与可见报告同目录保存，
  未将隐藏 reasoning 写入 Wiki。共同事实是：per-video K 只能写出计划 receipt，尚未驱动 ActionFormer/
  VideoMAE 的真实变长或分桶重计算；六臂仍不是可实例化且语义隔离的官方配置；CAL/HOLD 没有可验证的不相交
  绑定；checkpoint/resume/retention/final 选择和 launcher 没有可执行闭环。`single_stage` 的 physical-time
  mapping 已在 threshold/top-k/NMS 前执行，作为本 snapshot 唯一结构通过项。该结论是静态审查证据，不是性能、
  成本或科学路线否定；允许且仅允许一次 claim-preserving focused Builder correction，之后由 Critic 与 fresh DSH
  focused recheck 裁决。没有数据、GPU、Slurm、远端训练、PRE_RUN 或官方结果发生。

- 2026-08-17（semantic-indirect implementation package terminal close）: 唯一允许的 focused Builder
  correction `4b78b7d12dd5e19194c5661ab46678afda7ec1ae` 只增加了部分 arm/HOLD 声明，未能把动态 K 接入真实
  ActionFormer/VideoMAE 变长或分桶计算，也没有实现 observed executed-K、可实例化六臂、实际 FIT/CAL/HOLD
  dataset binding 或可恢复 checkpoint/launcher。same Critic 的 focused recheck 判为
  `DYNAMIC_SEMANTIC_INDIRECT_FOCUSED_BLOCKED / BLOCKED_PRE_RUN`。fresh DSH session
  `session-432cdf69-3e91-4bde-972d-c08fc9e2b463` 的 header、自然首行 `/^We need\b/` 与 completed 终态合格，
  可见结论同为 `FAIL`；其 receipt、raw session、可见报告在
  `docs/dsh/DUCA_DSH_ROUND3_FOCUSED_RECHECK_RECEIPT-2026-08-17.md` 及同目录文件，未记录隐藏 reasoning。
  `cfg.selector`/`SELECTOR` mismatch 令 validator 本身崩溃，actionness-only 仍受 boundary 排序，fixed-K 为死声明，
  进一步确认 PRE_RUN 不可准入。物理时间在 threshold/top-k/NMS 前回映仍是唯一通过的静态子合同。
  因 one-correction budget 已用尽，当前状态归一化为 `BLOCKED_PRE_RUN / IMPLEMENTATION_PACKAGE_CLOSED`：禁止
  第三次 Builder、任何再审、Evaluator、PRE_RUN、数据/GPU/Slurm/远端实验或效能 claim。该终态没有真实视频结果，
  因此不否定“0/1 动作性+边界预测→确定性间接选帧→dynamic outer-K”的科学假设，只关闭当前实现包。

- 2026-08-17（新的 semantic-indirect 动态预算实现周期）: 用户授权保留既有科学主线、关闭旧包
  `4b78b7d1`，并从 clean DUCA 基线 `6125654b946cc30c614428ce1141f1903b015867` 建立独立 worktree
  `C:/Users/skywalker/.codex/worktrees/duca-semantic-cycle2-20260817`。首个候选为
  `5863b2f1fa1812c6c39ed275e1639c3dd78d4468`，状态仅为 `implemented/static-tested`，无数据、GPU、
  远端、训练、验证、mAP 或成本结果。独立静态审查确认它仍把不等长选择 padding 为固定目标长度，未把
  dynamic K 接入 ActionFormer/VideoMAE 的实际变长或分桶重计算；六臂实例化、FIT/CAL/HOLD 不相交
  loader 绑定、物理时间全链路测试与完整 recovery checkpoint/launcher 也未闭合。后续仅允许对这些
  冻结、claim-preserving 实现缺陷做最小修正和可区分测试；科学路线与 split、metric、成本合同不变。

- 2026-08-18（semantic-indirect cycle2 implementation terminal）: clean snapshot
  `d80022e963a8ad21d390c785cbd8a4c23f41484a` 完成三次允许的、未改变科学路线的实现修正。静态证据显示
  ActionFormer 已可按 K 分桶调用重型骨干，六臂 policy 与 FIT/CAL/HOLD 静态合同已写入，checkpoint state
  也扩展到 scaler/RNG/DataLoader 等恢复字段；但最终独立审查发现 dense arm 仍是 placeholder，shared
  detector/loss/NMS/evaluator/update/seed 没有形成真实训练对象绑定，且 checkpoint 未生成 latest-3、
  milestone、final 与 final-EMA 的完整生命周期。本机 `c10.dll` 同时阻断 runtime fixture。归一化结果为
  `BLOCKED_PRE_RUN / CYCLE2_IMPLEMENTATION_PACKAGE_CLOSED`；禁止本周期第四次 Builder、Evaluator、PRE_RUN
  或正式训练。没有任何数据、GPU、远端、mAP、成本或论文效能结论发生，因而不否定“0/1 actionness+boundary
  →确定性 acquisition→dynamic outer-K”的科学路线。
- 2026-08-19（DUCA full-official matrix relaunch / weight correction）: user confirmed remote
  checkpoint write reliability is restored; a read-only 640 MB torch.save probe on the shared
  root completed and was removed, so the previous JuiceFS stream-writer failure is treated as
  an environmental incident rather than a model failure. A new clean DUCA runtime commit
  `7529fba6...` on `codex/duca-full-official-rerun` changes the frame-score fusion preset to
  boundary-first (`action/boundary/uncertainty/redundancy = 0.20/0.65/0.15/0.10`, no-risk arm
  `0.70/0/0/0.15`) and makes dynamic outer-K consume only the already-fused
  `frame_selection_logits` (`actionness_weight=1.0`, secondary weights 0), fixing neutral-score
  budget bias and boundary double-counting. Remote CPU focused tests `8 passed`; config parsed
  for all five arms; `bash -n`, path checks, and `sbatch --test-only` passed. Slurm array
  `1244133` (15 cells, arms × seeds 3407/3408/3409, fresh run root
  `/data/run01/sczc063/yuzibo/duca_full_official_7529fba6_wv1_20260819T132000Z`) was submitted.
  Eight cells entered epoch 0 with finite losses; no checkpoint/mAP yet. Status is
  `experiment_running`; no efficacy or cost claim is licensed.

- 2026-08-20（DUCA Query-Bridge 独立 Pro 审查吸收）: Project 内 Pro 会话
  `duca-project-query-review` 给出 `REVISE`：Query 只增强 scout 的 actionness/start/end
  语义，确定性 acquisition 负责物理位置，固定 K 的 dense/uniform/random/actionness/
  actionness+boundary 归因先于 dynamic K。项目逐项复核并记录于
  `DUCA_PRO_REVIEW_ABSORPTION-2026-08-20.md`。代码事实确认 UVT 将 V(t) 的位置、几何/EMA
  训练和 K evidence 混合，Fovea 五臂不是单变量链且其 geometry/coarse/diversity/budget
  损失有明确语义问题；按非连续 selected frame 直接拼 VideoMAE clip 是高优先级、待实证的
  时间语义风险。审查中“没有新训练”的说法已过时：UVT `1244840` 与 Fovea `1244851` 均有
  60-epoch 真实开发性训练，但都缺同提交 matched dense/uniform/random（Fovea 还缺两臂），
  不能升级为效率、mAP 优势或论文 claim。接受其诊断内核，不把其 block/损失/阈值数值当作事实；
  当前科学状态为 `REVISE / designed_for_clean_reimplementation`，无新增主张。

- 2026-08-20（DUCA 第二份外部审查对照与核验）: 用户提供的第二份 `REVISE` 审查文本已以
  SHA-256 `fe19d0e...d475b` 登记；由于没有 session/model/raw transcript，它只作为外部建议而非
  已认证裁决。两份审查在 semantic-indirect、fixed-K-first、真实 variable VideoMAE compute 和
  physical-time-before-NMS 上一致，相关 UVT/Fovea 代码事实已复核。第二份的“没有新训练”表述被
  更正为：old cycle2 包仍为 `BLOCKED_PRE_RUN`，但 UVT `1244840`、Fovea `1244851` 已完成真实
  60-epoch 单 seed 开发训练，仍缺同提交 matched control/成本而不能升格。连续 cliplet、边界标签、
  warmup/release、K 集合与动态阶段门槛在两份审查之间并不相同，均未冻结；完整对照见
  `DUCA_SECOND_REVIEW_COMPARISON_AND_ABSORPTION-2026-08-20.md`。无代码、数据、训练或 claim 变化。

- 2026-08-20（历史 65 输入链与协同机制处置）: 冻结提交 `42dba3f...` 复核确认，`65.385724` 的
  384 个 RGB 输入来自非均匀 selected positions，经 gather 后直接替代 dense 输入并进入重型 backbone；
  selected-axis proposal 在 NMS 前可回映原时间，但 VideoMAE forward 合同没有表达相邻 selected frame
  的真实间隔。它是间接非均匀输入的历史信号，不是 uniform 结果，也不能因 90 epoch/蒸馏/transport/
  adaptation 混杂升级为公平增益。按用户要求，Query context 与知识传递不删除：后续仅以 semantic-only
  Query residual 和 detached teacher-to-semantic-logit 两个单变量扩展保存；二者均不允许直接索引、K、
  在线 detector feedback 或额外 detector updates。既有 dense/uniform/random/VC 记录只读绑定，不重跑。
  详细决定见 `decision_history.md:716`；无新训练或 claim。

- 2026-08-20（DUCA 对抗性 Pro 审查材料）: 用户授权一场新的 Oracle Project 内 Pro 讨论，用于严厉反驳
  历史输入恢复、Query 协同和 teacher 知识传递的设计，并独立提出更优的唯一路线。材料包
  `DUCA_ADVERSARIAL_PRO_PACKET-2026-08-20.md` 固定了历史 `42dba3f` 的非均匀输入、UVT/Fovea
  开发结果、既有 VC dense/uniform/random 不得重跑、S0/SQ/SQD 单变量机制和 H/S/P 候选。该材料不是
  科学决定或实现授权；等待 Pro 原文后逐项核验。

- 2026-08-21（DUCA 对抗性 RiskClip 审查的记录与核验）: 用户提供 `PIVOT`/RiskClip 审查副本。
  本地确认其关于历史 `42dba3f` 的 pre-backbone 非均匀 RGB gather、NMS 前物理时间回映和
  90-epoch 多因素混杂的关键事实；同时确认 `4ae50671`/`df544c78` 本地提交存在但相应分支未见于
  origin，故该审查的 GitHub 逐行核验边界有效。审查的路线内核仅条件接受：Query/KD 限于语义
  scout，确定性规则产生采样与 dynamic K，fixed-K 是控制；连续物理 cliplet 是待验证运行时合同。
  RiskClip 的具体 16-frame 单元、风险公式、阈值、冻结 detector 与数值门槛均未冻结。无代码、数据、
  实验、成本或论文 claim 变化；详见 `DUCA_ADVERSARIAL_RISKCLIP_REVIEW_VERIFICATION-2026-08-21.md`。

- 2026-08-21（DUCA BSC-DK 审查的对照与核验）: 第二份外部审查同样建议 `PIVOT`。其关于历史
  `42dba3f`、历史 65 混杂、UVT/Fovea bundle 不可归因与 semantic-only Query/KD 的诊断得到复核；
  但 BSC-DK 和前一份 RiskClip 在预算目标（阈值节省 vs 固定总预算重分配）、detector 训练合同、
  physical reconstruction、P0 门槛和动态控制上不一致，不能拼成同一实现。明确不冻结 48×16 cliplet、
  certificate/rho、160/40 split、数值门槛或 SQD 默认主臂；BSC 未来若继续必须补强内容—预算的
   K-shuffle 控制。无代码、数据、训练、性能、成本或 claim 变化；详见
   `DUCA_BSC_DK_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md`。

- 2026-08-21（DUCA SCOPE-DK 第三份外部审查的对照与核验）: 用户提供第三份 `PIVOT`/SCOPE-DK
  建议。核验支持其最强事实边界：历史 `42dba3f` 是 pre-backbone 非均匀 RGB 输入但将 selected-rank
  帧伪连续地送入 VideoMAE，物理坐标只在输出端回映；selector 位于 dense decode/resize 之后；实际
  backbone 调用没有已运行的 `executed_k` 收据；历史恢复合同没有完整 RNG/DataLoader replay。三份审查
  共同条件接受“语义 scout → 确定性物理采样 → dynamic K”，fixed-K 只作控制。SCOPE 的连续 cliplet、
  `GAPPACK`/`CONTIG` 和 K-shuffle 是值得保留的未实现合同候选；但其最小覆盖预算与 BSC 固定总预算、
  RiskClip 阈值预算不同，且其 splat/新增通道和 detector 训练合同会引入新的混杂，均未冻结。无代码、
  数据、训练、性能、成本或论文 claim 变化；详见
  `DUCA_SCOPE_DK_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md`。

- 2026-08-21（DUCA IPEC-K 第四份外部审查的对照与核验）: 用户提供第四份 `PIVOT`/IPEC-K
  审查。复核支持其关于历史 `42dba3f`、历史 65 的联合课程混杂、UVT/Fovea bundle 不可归因、
  selected-rank 伪连续、未证明 actual executed-K 的事实；并吸收 S0/SQ/SQC/SQD 对 Query、cycle、
  teacher 的独立语义门建议。IPEC 的端点覆盖公式、48×16 cliplet、预算范围、阈值、模块与训练包仍是
  proposal。它与前三份在预算命题、detector 训练合同和重建器上不一致；其 P1 缺少同一 RGB 帧集的
  `GAPPACK`/`CONTIG` 连续性对照，P2 也未明列内容—预算 K-shuffle，故均不冻结。无代码、数据、
  训练、性能、成本或论文 claim 变化；详见
  `DUCA_IPEC_K_FOURTH_REVIEW_COMPARISON_AND_VERIFICATION-2026-08-21.md`。
## 2026-08-21 — DUCA 端点覆盖、物理连续 cliplet 与 dynamic-K 新主线冻结

- 用户确认将新 clean implementation cycle 冻结为：0/1 action/start/end scout → 确定性端点覆盖 → 每个 16 帧的物理连续 VideoMAE cliplet → physical-time pre-NMS → 不变 detector；dynamic outer budget 是论文候选核心，fixed K 仅是公平对照/回退。
- S0/SQ/SQC/SQD 只隔离 Query-Bridge、cycle 和 detached 语义蒸馏对 scout 表征的帮助；它们不得直接输出 frame index、K、proposal 或 test-time teacher/cache 信号。
- 历史 selected-rank 非连续拼接被保留为 GAPPACK 强归因控制，必须与 CONTIG 使用完全相同的原始 RGB 帧集合和 K。该控制用于解释预训练时间语义，不能重新包装为主方法。
- dynamic-K 后续只能与 fixed M=24、K-shuffle 和 actionness-only dynamic 比较，并记录 actual executed K 与完整成本。没有新增代码、PRE_RUN、远端作业、性能或成本结果；状态为 `designed_not_implemented_not_tested`。

## 2026-08-21 — 最终 Pro 裁决与可执行合同

- exact DUCA Project 的第二轮且最终 Pro 裁决为 `CONTINUE`。保留“action/start/end 语义
  scout → 确定性物理连续 cliplet → dynamic outer M → 真实 sparse VideoMAE → physical-time
  pre-NMS”的科学内核；fixed M24/K384 只作控制和回退，dense/uniform/random 不重复训练。
- 最终合同删除 SQD/语义蒸馏，只允许 S0 通过后加入 SQ、SQ 通过后加入 detached cycle；
  GAPPACK 只有在实例化后的 VideoMAE temporal atom 与输出 slot 一一对应时存在，否则失败关闭，
  不阻断 CONTIG。动态预算的 certificate deficit 已统一为 int64 定点定义，避免 float/int 混合歧义。
- 三方只读核验确认：唯一忠实 base 是官方 AdaTAD `01c58b9`；THUMOS14 canonical 411、注释、
  类别图和 VideoMAE-S 权重齐全。共享官方 AdaTAD receipt 与完整 Slurm 资源元组仍未绑定，故
  当前只能进入 clean 实现和静态审查，不能 PRE_RUN 或提交训练。没有新增性能或成本结论。
- 唯一规范文件为 `DUCA_PHYSICAL_CLIPLET_DYNAMIC_BUDGET_FINAL_CONTRACT-2026-08-21.md`；
  第一版冻结合同标记为 superseded，保留作决策追溯。

## 2026-08-21 — 物理连续 cliplet 候选实现、目标环境准入与 S0 启动

- 从官方 AdaTAD `01c58b9f2370e914150cf94d392208a4e211c053` 建立独立 clean 候选，冻结为
  `8be817bba7a906c9b58446749d58c1752e1d5b6c`。实现把 action/start/end 语义监督、定点端点覆盖、
  非重叠物理连续 16 帧 cliplet、selected-only VideoMAE、dense 物理时间重建和 NMS 前秒级回映接入
  同一执行链；checkpoint 每 5 epoch 保存并覆盖模型、EMA、优化器、调度器、混合精度、epoch/update、
  各 rank 随机数、DataLoader、选择状态和成本 ledger 游标。GAPPACK 保持失败关闭，dynamic-M 尚未启用。
- 独立静态审查判为 PASS。N16R4 上的只检查作业 `1245927` 在同一冻结 revision 完成：canonical
  THUMOS14 411 视频与训练期 S0 ledger 绑定通过，目标环境编译通过，3 个聚焦测试文件共 11 项测试通过；
  状态由 `S0_PRECHECK_READY` 晋级为 S0 运行准入。两次更早提交仅暴露 Slurm 内存声明和非交互
  `/etc/profile` 的启动器问题，均在 Python、数据训练和模型前停止，不构成科学或模型负证据。
- 完整 S0 第一种子训练已经提交并开始运行。该阶段只训练低成本 scout 的动作性与起止边界语义，
  不执行 AdaTAD 检测 mAP 比较，不能形成动态预算、定位性能或计算节省结论。后续 FZ/JT 仍必须等待
  S0 预注册 `final_ema.pth` 和跨项目共享 official AdaTAD receipt；不重复官方 dense 训练。
- S0 已完成首个 epoch 的 200 个 batch，末批有限损失 `0.536369`，累计 197 次成功优化更新。
  这只证明训练链已进入真实反向传播并跨过首 epoch；尚未检查收敛、语义校准、FIT/CAL/HOLD 或
  最终 checkpoint，因而不升级为方法支持证据。

## 2026-08-21 — S0 终态语义测量与固定片段完整检测实验启动

- S0 完整训练 Job `1245928` 在 clean 候选上完成 60 epoch、11990 次更新并正常退出，预注册
  `final_ema.pth` 与五 epoch 恢复链齐全。训练集内 FIT/CAL/HOLD 评估 Job `1246187` 完成；HOLD
  action AUROC/AP/Brier 为 `0.832567/0.711655/0.159139`，常数先验 Brier 为 `0.223236`，说明
  动作性排序与概率具有可用信号。start/end AUROC 为 `0.815920/0.869611`，但 AP 仅
  `0.056408/0.065448`，Brier `0.095374/0.098901` 明显差于常数先验 `0.008585`，说明边界绝对
  校准仍弱。该证据仅为训练总体语义诊断，不是 TAD mAP、效率或论文支持。
- S0 评估过程中两个启动失败均被归为可区分的实现合同缺陷而非科学负结果：第一次修正 stride-4
  时间中心之间的完整物理 source-grid bin 覆盖；第二次按冻结合同恢复短窗口
  `M=min(24,floor(T/16))`。最终代码 revision 为
  `ad8d01c31adac685e1bf4d96e82a62c730b1e536`，目标环境 17 项 focused tests 通过。
- 跨项目共享官方 AdaTAD Job `1245842` 已只读绑定：官方 revision `01c58b9f...`、seed 42、
  60 epoch、最终 official validation Avg-mAP `68.73`，相对公开 `69.03` 锚点为 `-0.30`。
  DUCA 不重复训练该 dense 基线。绑定文档加入后的 clean HEAD 为
  `3cc7fcdf46bd6e47b36cdcdbf5150428708cd440`，不改变已审查模型代码。
- FZ_CONTIG/JT_CONTIG 的 PRE_RUN Jobs `1246253/1246254` 均 `COMPLETED 0:0`，资源、S0 final-EMA、
  共享基线、恢复合同与 11 项 focused tests 通过。完整 60-epoch seed `3203700` 实验已提交为
  Jobs `1246260/1246261`，结果根分别为 `runs/fz_m24` 与 `runs/jt_m24`。状态升级为
  `experiment_running`；尚无终态检测 mAP 或完整成本，不得提前解释为方法支持。

## 2026-08-21 — FZ/JT 首批启动缺陷与 claim-preserving 修复

- Jobs `1246260/1246261` 在首 batch 反向传播前后同时终止，原始错误为 DDP 对共享 VideoMAE
  adapter 参数重复 `mark ready`。根因是每样本重复调用重型 backbone 时，默认重入 activation
  checkpoint 为同一 adapter 参数注册了多个 reducer hook；这不涉及数据、损失、采样机制或性能。
- clean revision `8a6e7ea23b5389fbbd071820f43eb194b39ae5cd` 只将 ViT checkpoint 改为 PyTorch 2.0.1
  支持的非重入形式并增加可区分测试。模型数学、adapter 冻结、优化器、EMA、采样和评估合同不变。
  独立 focused 审查为 PASS；N16R4 编译与 12 项 focused tests 通过，精确 revision PRE_RUN
  `1246282/1246283` 均 `COMPLETED 0:0`。
- 新结果目录下的完整 FZ_CONTIG/JT_CONTIG Jobs `1246284/1246285` 已启动，并跨过原始失败位置
  进入真实优化更新。旧失败目录保留为实现诊断且不参与任何结果汇总；当前仍无终态 mAP 或成本。

## 2026-08-21 — 首个五 epoch 恢复点与终态评估入口

- FZ_CONTIG/JT_CONTIG 均已在 Job `1246284/1246285` 生成首个 `epoch_005.pth`。两份 checkpoint
  均为 `DUCA_FULL_RESUME-v001`，在 epoch-end 边界记录 986 次成功更新，包含模型、EMA、优化器、
  调度器、AMP scaler、每 rank Python/NumPy/Torch/CUDA/DataLoader 随机状态和选择状态；最近三份恢复点、
  milestone、final 与 final-EMA 的保留清单已落盘。该材料证明恢复状态被真实写出，不是性能证据。
- 实测发现训练继续后成本/执行账本会超过最近 checkpoint 的游标，原恢复函数因要求严格相等而会拒绝
  合法中断恢复。该确定性实现缺陷已在独立 recovery revision
  `22a2985bd3f97b39ec70ff54282442186e62763b` 修复：恢复前只原子裁剪 checkpoint 之后的诊断尾行，
  缺行仍失败关闭；远端同环境 17 项 focused tests 通过。正在运行的模型 revision、权重和结果未改变。
- 终态评估入口冻结为 `b91f3e11c18ec365c5e532b01015bcaeee4d5ce1`，明确绑定两臂各自
  `gpu1_id1/checkpoint/final_ema.pth`、seed `3203700`、完整 THUMOS14 validation、官方 mAP evaluator
  与 Soft-NMS。该入口已通过独立静态复核，仍须等待两项 60-epoch 训练正常完成后执行；当前无新增 mAP。
- 终态评估 Jobs `1246337/1246338` 已分别以 Slurm `afterok` 依赖绑定到训练 Jobs
  `1246284/1246285`。它们不会读取中间 checkpoint，只有对应训练成功并生成预注册 final-EMA 后才会
  获得 GPU、执行完整 validation；当前为 dependency pending，不是评估结果。
- 对两臂各自首 986 次成功更新的执行账本进行只读核验：`executed_k` 与
  `patch_embed_input_k` 在每个样本上完全一致，范围 48–384、均值 359.043；对应连续 cliplet 数范围
  3–24、均值 22.440，FZ/JT 分布相同。低于 384 的条目来自短有效窗口的
  `min(24,floor(T/16))`，不是 dynamic budget。该证据证明实际送入 patch embedding 的长度随有效输入
  缩短，没有 384/768 padding 假稀疏；它仍不是终态效率或性能结论。

## 2026-08-21 — DUCA 向 TAS 的迁移可行性研究

- 新增 `ideas/duca-tas-migration.md`，状态为 `discussed / migration_design`。结论是 DUCA 的
  低成本全局侦察、物理连续 cliplet、真实重主干减算和物理时间重建可迁移到时序动作分割，
  但 ActionFormer 区间检测头、Soft-NMS 和 mAP 合同不可直接迁移。
- 首个唯一迁移宿主建议为 MS-TCN++，首个数据集为 50Salads 五折。证据分为官方 I3D 协议锚点、
  同 I3D 特征空间机制诊断，以及同一 VideoMAE 主干下 dense/连续均匀 cliplet/DUCA cliplet 的
  端到端因果比较；禁止跨表示计算采样增益。主看 F1@50 与 Edit，并核验短动作、类别转换边界和覆盖缺口。
- TAS 的特有风险已写明：相邻动作可以都保持 actionness 为正，因此迁移后的边界头必须学习
  动作类别转换，而非只学习前景开始/结束；物理时间插值能修正坐标但不能恢复未观察到的短动作。
  当前未核验 TAS 数据/特征/许可证，未写代码、未启动实验，也未改变正在运行的 DUCA-TAD 路线。

## 2026-08-21 — MS-TCN++ 官方锚点与 FineGym/FineDiving 取舍

- 已从官方 `https://github.com/sj-li/MS-TCN2.git` 建立 DUCA 独立克隆
  `E:/DeskTop/TAD/external/MS-TCN2_DUCA_20260821`。官方当前 `master=f423a9e...` 的
  `model.py:14` 含可复现语法错误，不能作为可执行锚点；固定最后一个可解析的官方历史提交
  `9d31fb3c23467b9ce3030d43b6d33a96869b6422`，其 MS-TCN++ 主路径通过 `py_compile`。
- FineGym 被归类为第二阶段短动作/层级边界压力测试：官方具有细粒度时间区间，但没有
  MS-TCN++ 逐帧 TAS 的直接 loader/split/evaluator。FineDiving 的主任务是动作质量评价，只适合在
  另行冻结 step-segmentation 协议后作辅助诊断。二者均不替代首个 50Salads 官方五折复现。
- N16R4 共享根只读检查未在常见路径确认 FineGym、FineDiving、50Salads、GTEA 或 Breakfast
  的合法完整绑定；大范围检索超时，因此状态为 `UNVERIFIED/MISSING_AT_COMMON_PATHS`，不是
  “已证明不存在”。在数据/许可/官方 split 绑定完成前不提交训练，也不声称 TAS 性能。

- 执行准备继续完成：远端兼容 venv 已通过 CPU 前向；GPU Slurm 环境见证 Job `1247227`
  在 RTX 4090 上 `COMPLETED/0:0`，输出形状 `[2,1,4,9]` 且数值全有限。官方 Zenodo
  `data.zip` 为 30,210,005,282 bytes，远端直连约 15 KB/s，已停止不可行下载，仅存 `.part`
  且不作为数据。五折启动器已绑定通过门禁的 venv，本地/远端 Bash 语法通过；因 50Salads
  官方包尚无合法完整绑定，正式复现未提交，当前没有 TAS 性能或效率结果。

## 2026-08-21 — MS-TCN++ 官方数据学术代理续传

- 经用户明确授权，使用 N16R4 既有学术 HTTP(S) 代理和远端 `aria2c 1.36.0`，从官方 Zenodo
  `10.5281/zenodo.3625992` 对 `data.zip` 发起唯一八连接断点续传。公开元数据固定文件大小
  `30,210,005,282` bytes、MD5 `078aa08875747e6264b892ae6e0ac7be`；代理凭据没有写入命令行、
  日志或项目文件。下载 PID `3586871`，日志与目标位于
  `/data/run01/sczc063/yuzibo/duca_tas_mstcn2_9d31fb3/`。初始观测吞吐约 85--103 KiB/s，较直连
  约 15 KB/s 提升但仍需约 80--96 小时。任务保持运行；完成前不解压、不提交官方五折训练，
  当前仍无 TAS 性能或效率结果。

- 终态更新（2026-08-22 03:29:17 +08:00）：下载完成，文件精确大小
  `30,210,005,282` bytes；aria2 使用预注册官方 MD5
  `078aa08875747e6264b892ae6e0ac7be` 完成校验并报告 `Verification finished successfully`、
  `Download complete` 和 `stat OK`。目标仍为 `data.zip.part`，未解压、未改名、未绑定 loader，
  未启动 MS-TCN++ 训练。小时 watchdog 已在完成后暂停，避免重复通知。

## 2026-08-22 — 历史逐帧 True-Time 课程实现止于 PRE_RUN 前

- 冻结了历史 ASFormer 间接逐帧选取的配对实验：`RANKPACK_K384` 与
  `TRUETIME_K384` 共用固定 `K=384`、同一检测器和总计 60 epoch/6,000 次成功更新；训练分为
  20 epoch 语义预热、20 epoch 余弦过渡和 20 epoch 联合训练。设计提交为 `d712df7f`，独立
  clean worktree 为 `E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculum_20260821`。
- 配置/启动候选 `e6708ef8` 没有把真实物理时间接入模型运行；最终候选 `60816c9b` 新增了
  `PhysicalTubeletPatchEmbed` 和 metadata bridge，但实际 `VisionTransformerAdapter.forward`
  仍无条件调用原 Conv3D `PatchEmbed`，新算子没有被实例化或执行，长度 384 的位置 metadata
  也没有按 16 帧 clip 分组。
- 独立 focused recheck 因此给出 `DYNAMIC_TRUETIME_FOCUSED_STATIC_BLOCKED / BLOCKED_PRE_RUN`。
  当前实现周期的有界修正机会已耗尽；Evaluator 未启动，未访问数据、未提交 GPU/Slurm、未产生
  optimizer update、mAP 或成本结果。该终态只否定当前实现的可运行性，不否定“物理时间一致性
  可能修复历史非均匀逐帧输入”的科学假设。

## 2026-08-22 — True-Time 新干净周期通过全模型门并启动正式配对实验

- 用户授权后，新周期没有复活旧 `60816c9b` 候选，而是在独立 clean worktree
  `E:/DeskTop/TAD/OpenTAD_DUCA_TrueTimeCurriculumV2_20260822` 建立
  `codex/duca-truetime-curriculum-v3-20260822`。冻结 revision
  `11126684af779aa2916a68ecf617c4f14c805478` 保留历史 train-only ASFormer 间接选帧、固定
  `K=384`、同一 RGB 帧集合、检测器、损失、NMS、官方 evaluator、seed `3407` 和 20/20/20
  课程；唯一因果变量是 VideoMAE 在第一次时间混合时使用 selected-rank 还是原物理位置。
- 新实现把 TrueTime 物理位置实际送入生产重型前向，并让两臂共享物理坐标 ActionFormer 头和
  pre-NMS 解码。两次门禁失败均发生在性能测量前并已作 claim-preserving 修复：短视频 metadata
  只含有效位置而执行张量固定为 K384；全无效 padding tubelet 的注意力 NaN 不能通过乘零消除，
  因而改为显式 `masked_fill`。独立审查为 PASS；N16R4 同环境 26 项 focused tests 通过。
- 新证据根 `/data/run01/sczc063/yuzibo/evidence/duca_truetime_v3_pre_11126684` 下，RankPack
  Job `1248812` 与 TrueTime Job `1248813` 均完成 full/padded/short-padded 三步全模型门，短样本
  `effective_k=62` 且损失有限；两臂 protocol/gate/authorization 均冻结为可执行。该证据证明运行合同，
  不证明 mAP 或效率。
- 正式 60-epoch 配对已提交并开始运行：RankPack Job `1248822`、TrueTime Job `1248823`，共同结果根
  `/data/run01/sczc063/yuzibo/duca_truetime_curriculum_official60_11126684_20260821T200057Z`。
  两臂各一张 N16R4 GPU，6000 次成功更新，前 2000 次语义预热、中间 2000 次余弦过渡、后 2000 次
  联合训练；每 5 epoch 保存完整恢复点，终态固定 epoch-59 final-EMA。当前状态仅为
  `experiment_running`，尚无 mAP、方差、完整成本或论文支持结论。
- 两项正式作业均已在节点 `g0022` 进入 epoch 0 并跨过 50 个真实训练 batch。RankPack/TrueTime
  在该点的总损失分别为 `5.5309/5.5392`，课程均为 `exact_uniform_warmup`、`alpha=0`；数值有限且
  没有复现门禁阶段的 padding NaN。该观察只证明首次真实优化链成立，不能用于比较两种时间解释。

## 2026-08-22 — 固定连续片段终态为负，True-Time 配对获得单种子机制增量

- 固定 `M=24,K=384` 的 FZ_CONTIG/JT_CONTIG 训练 Jobs `1246284/1246285` 与终态官方 validation
  Jobs `1246337/1246338` 均 `COMPLETED 0:0`。FZ 的 Avg-mAP 为 `49.89`，tIoU
  `0.3/0.4/0.5/0.6/0.7` 为 `65.04/59.98/52.51/42.22/29.68`；JT 为 `47.24` 和
  `63.51/57.89/49.69/38.57/26.52`。执行账本确认 `executed_k=patch_embed_input_k=384`、M=24，
  这是真稀疏负结果，不是 padding 假动态。联合训练低于冻结 scout，基础连续片段门失败；按冻结合同
  停止 Query/cycle 与 dynamic-M，不重复 dense/uniform/random。
- RankPack/TrueTime Jobs `1248822/1248823` 均完成 60 epoch、6000 次成功 optimizer/scheduler/EMA/
  课程更新和 epoch-59 EMA 官方 validation。RankPack Avg-mAP 为 `61.5722`，tIoU 五点为
  `78.6567/73.8490/65.3328/52.9221/37.1003`；TrueTime 为 `62.1930` 和
  `78.7428/74.2565/65.4630/54.6107/37.8918`。TrueTime 的增量为 Avg `+0.6208` 点、
  tIoU 0.6 `+1.6885`、0.7 `+0.7915`，说明同一 seed 下物理时间解释小幅优于 selected-rank，
  主要改善高 IoU；单 seed 不能支持稳定论文主张。
- 两臂 epoch-59 checkpoint、prediction 与 `terminal_evaluation.json` 均存在。冻结官方 mAP evaluator
  对两份 prediction 的只读重算与记录指标逐项误差不超过 `1e-12`。Slurm 最终为 `FAILED 1:0` 只因
  post-run 封存器比较了 raw evaluation config 哈希 `927e…` 与补齐默认字段后的 normalized 哈希
  `2965…`；该错误发生在训练和评估之后，不是科学失败，但 `post_run_evidence.ok=false` 必须保留。
- 当前证据裁决：连续片段主张为 `no`；TrueTime 机制为 `partial`。下一步不重训：先统一封存哈希口径并
  对既有预测执行预注册的 10,000 次逐视频配对 bootstrap；只有该统计门支持后才考虑最小多 seed。

## 2026-08-22 — TAS 数据接口更正为 EAST 原始 RGB

- Zenodo `data.zip` 已确认是 MS-TCN++ 的预提取 2048 维特征、逐帧标注和官方划分，不是原始视频。
  为避免错误启动，远端特征解压在完成标记生成前停止；完整压缩包和约 23GB 部分目录均保留，未绑定
  EAST、未启动训练。
- 用户要求验证 backbone 前选帧，首个端到端宿主因而改为 EAST；MS-TCN++ 仅保留为特征级 TAS
  锚点。EAST 的共享协议视频是 `160x160 @ 2 FPS` RGB，Dundee 原始采集是 `640x480 @ 30 Hz`
  AVI，两者在证据中严格区分。
- 远端常见路径没有现成 50Salads AVI。Dundee 官方下载端点经学术代理持续 HTTP 503，EAST README
  的 Box 共享链接当前也不可达；唯一官方源监控保持运行，当前 `BLOCKED_RAW_RGB`。
- Facebook Research AVT 公开的 50Salads 标注包已保存到
  `/data/run01/sczc063/yuzibo/datasets/TAS/annotations/50Salads/avt_50salads_annotations.zip`，大小
  688,521 bytes，ZIP 完整性检查通过并含 50 个逐帧标注及类别映射。未完成 EAST JSON/split
  对齐前不启动端到端训练，也没有 TAS 性能或成本结果。

## 2026-08-22 — H65C SingleClock Pro 核验与实现周期终态

- 独立 Pro 原文与用户粘贴件统一换行后逐字一致，裁决 `REVISE`：恢复 H65 的 Stage1/Stage2、
  sampling-rate exact-K、50% uniform companion、cls/reg contribution、完整 ASFormer adaptation 和
  selected-axis 合同；Unit-1 只加入首个 ViT 时空注意力的 SingleClock 相对时间偏置。代码核验认为
  主判断合理，但不接受把 score-only threshold/top-k 的顺序改变混入同一 Unit-1。
- 新 clean 分支 `codex/duca-h65c-singleclock-unit1-20260822` 从 H65 `42dba3f9` 派生。前三个候选闭合了
  嵌套配置、24×16 位置分段、activation checkpoint、共享偏置与 fail-closed validator；终态
  `87d9a1aef355a508b5324b0469f5a68d0f967cfe` 又加入零初始化 scalar 和 uniform 无 mask 快路径。
- 终态复核发现该提交仍在每个 16-frame clip 内独立构造全 768 点 canonical uniform，而不是先生成
  H65 全局 K384 canonical 再按全局 rank 切片。该缺陷改变 SingleClock 定义，现有测试也没有覆盖它。
  实现包因此关闭为 `BLOCKED_PRE_RUN`；没有数据、GPU、Slurm、训练、mAP 或成本证据。
- N16R4 只读核验确认原 Stage-1 epoch-29 EMA 记录目录为空，声明 SHA `7233fa...b39e` 无实体；Stage-2
  回放 checkpoint 同样未定位。只有权威备份恢复 exact 资产并建立新的正确 clean candidate，才可重新
  进入 Critic/Evaluator。重新训练 Stage-1 只能算新实验，不能冒充历史 H65 identity replay。

## 2026-08-22 — H65-first SingleClock 第二 clean 实现周期终态

- 用户再次明确：历史 H65 的选择单位是整个 dense window 上的逐帧 sampling-rate/CDF-systematic
  exact-K；`K=384` 的全局物理帧位置生成后才按 `24×16` 送入 VideoMAE，16 帧不是采样决策单元。
- 新 clean 分支 `codex/duca-h65-first-singleclock-cycle2-20260822` 从 H65 `42dba3f9` 派生，终态
  `e12257d77547efb82a36f01844ee2c9d6289ac32`。代码已将全局 actual/canonical `[B,384]` 按同一
  global rank 切成 `24×16`，再形成每包 8 个 tubelet 中心；没有在每个包内重新选择帧或重算 canonical。
- 终局独立审查仍判定 `BLOCKED_PRE_RUN`：`TwoStageDetector` 在位置 metadata 缺失时返回空值，
  `BackboneWrapper` 随后静默运行无 SingleClock 的原路径；准入器也没有以生产 forward 测试证明所有
  runtime 不变量。本周期已用完三次聚焦修正并关闭，不启动 Evaluator、Slurm 或训练。
- 该负证据只说明本实现包没有达到准入要求，不是否定 H65 的逐帧间接采样或 SingleClock 科学假设。
  原 Stage-1 epoch-29 EMA 仍无实体；若以后用新重训 Stage-1，只能成对运行 matched H65-off 与
  SingleClock-on，不能把新资产称为历史 65.385724 的 identity replay。

## 2026-08-22 — H65-first matched Cycle3 终态

- 用户接受 matched fallback 后，从 H65 `42dba3f9` 建立 clean 分支
  `codex/duca-h65-first-matched-cycle3-20260822`。终态 `5e4b7d3790803b5b7adcf7ee182d8e9d3b747ce5`
  保留原 Stage-1 uniform-K384 与 Stage-2 learned sampling-rate curriculum；H65-off 是原 Stage-2，
  SingleClock-on 只增加第一 ViT block 的全局物理时间残差。
- 终局审查将 exact-uniform 误判为 H65-off；该判断不被采纳，因为历史 `65.3857%` 来自 learned
  sampling-rate Stage-2，exact-uniform 只是 Stage-1/归因控制。改变 OFF 会偏离用户要求的 H65 基线。
- 但负责人在部署前核验发现真正阻塞：Cycle3 launcher 绑定
  `/data/run01/sczc063/yuzibo/data/thumos14`、`annotations/thumos14.json` 和
  `/checkpoints/adatad_pretrained.pth`，而权威只读资源路径是 `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`、
  `thumos14/annotations/thumos_14_anno.json`、`thumos14/annotations/category_idx.txt` 与
  `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`。
- 本周期三次聚焦修正已用完，focused tests 也未在 N16R4 得到实际 PASS，故实现包关闭为
  `BLOCKED_PRE_RUN`。没有提交 Stage-1、H65-off 或 SingleClock-on Slurm 作业，没有新的 mAP/成本证据。

## 2026-08-22 — H65-first matched Cycle4 终态

- 用户授权的新 clean 周期严格限定为修正 N16R4 资源路径、完成目标环境测试与 PRE_RUN；只有准入通过
  才可提交新 Stage-1，并让原始 learned H65-off 与 SingleClock-on 共享同一 epoch-29 EMA 起点。
- clean 分支 `codex/duca-h65-first-matched-cycle4-20260822` 的终态 revision 为
  `2edd62713199531ca7c165b843feb3bd7c364e0e`。它没有改变全窗口逐帧 sampling-rate exact-K384、
  H65 课程、检测器或损失；SingleClock-on 仍只增加第一 VideoMAE block 的相对物理时间残差。
- 目标环境的 shell/Python 静态检查和 23 项 focused tests 通过。首次真实 smoke 还发现整数物理位置直接
  求均值的 dtype 缺陷，修正仅将 tubelet 中心计算转换为 float32，原始位置 metadata 仍保持 int64。
- 真实 Slurm smoke 先证实 Stage-1 数据、模型、EMA 与混合精度路径能够构建并进入 CUDA 训练；随后
  发现错误的一批式命令使用了训练循环不读取的顶层停止键，作业已停止且原始日志移出源码树保存。
  仓库 launcher 改为真实 `workflow.end_epoch=1`、`workflow.max_train_iters=1`、关闭 validation 并在
  Stage-2 config 加载前校验同一 checkpoint/path/SHA/epoch 三元组。
- 最终 PRE_RUN 仍在训练入口因 `LOCAL_RANK` 未由当前 Slurm 命令绑定而退出；没有生成 epoch-0
  checkpoint，因而不能审计 model/optimizer/scheduler/scaler/RNG/DataLoader/selector 恢复状态，也不能
  构造同起点 Stage-2 OFF/ON smoke。Cycle4 已达到三次聚焦修正上限并关闭为 `BLOCKED_PRE_RUN`。
  没有提交正式 Stage-1、H65-off 或 SingleClock-on，没有新的 mAP、成本或效能结论；该终态不否定
  H65 逐帧间接采样或 SingleClock 假设。

## 2026-08-22 — H65-first matched Cycle5 终态

- 用户将新周期严格限制为 Slurm 运行环境绑定。clean 分支
  `codex/duca-h65-first-matched-cycle5-20260822` 的冻结 revision 为
  `800e8b70193d907c3554ceb60f7ee1ea7eca6c1f`；唯一生产改动是让 launcher 在 spool cwd 中显式解析
  exact source root，并在所有 PRE_RUN/正式模式前设置单进程 `LOCAL_RANK/RANK/WORLD_SIZE` 与
  rendezvous 地址/端口，不覆盖 Slurm 提供的 `CUDA_VISIBLE_DEVICES`。模型、采样、数据、seed、训练长度和
  checkpoint 选择规则均未改变。
- N16R4 使用 `/data/run01/sczc063/yuzibo/conda_envs/opentad/bin/python`（Python 3.10.20）完成编译和
  23 项 focused tests。有界 Stage-1 PRE_RUN 完成一批真实 CUDA 前后向与 optimizer 更新并正常退出；
  这是基础设施证据，不是效能结果。
- 生成的 epoch-0 checkpoint 包含 `epoch/state_dict/state_dict_ema/optimizer/scheduler/grad_scaler`，但不含
  Python/NumPy/Torch/CUDA RNG、DataLoader/selector 状态和显式 update 计数，故未通过用户要求的恢复审计。
  Evaluator 按 fail-closed 合同没有构造 epoch-29 fixture，也没有提交同起点 H65-off/SingleClock-on smoke
  或正式 Stage-1。当前终态为 `BLOCKED_PRE_RUN`；没有 validation、mAP、成本或科学结论，H65/SingleClock
  科学假设未被否定。

## 2026-08-23 — H65-first matched Cycle6 终态

- 用户授权从新的 Stage-1 uniform-K384 开始复现。clean 分支
  `codex/duca-h65-first-matched-cycle6-20260822` 从 Cycle5 `800e8b70...` 派生，最终冻结 revision 为
  `61397c0ed8beda8b0b87e1f6ca3be02046614e02`。该周期没有改变 H65 的全窗口逐帧采样、检测器、损失、
  数据、seed、训练长度或 SingleClock 定义。
- 实现补充了 checkpoint 顶层成功 optimizer update 计数，并在 formal resume 时与恢复后的 training audit
  严格比对；同时保留 model/EMA/optimizer/scheduler/GradScaler、Python/NumPy/Torch CPU/CUDA RNG 与
  epoch-boundary loader/selector 合同。PRE_RUN 与正式模式在同一 launcher 中显式隔离，源码根、rank 与
  rendezvous 绑定均位于任何训练入口之前。
- 独立静态审查通过。N16R4 exact clean source `61397c0e...` 的 checkpoint/RNG 测试为 `2 passed`，
  Python 编译通过。最终 H65 focused suite 尚未开始执行测试体：pytest 收集阶段分别因
  `_duca_homotopy_under_test.models.duca._fixed_budget_autograd` 缺失和一个相对导入没有 package 上下文而
  退出。该错误没有形成 GPU、checkpoint 恢复、H65-off/SingleClock-on 或效能证据。
- 本周期三次 claim-preserving 修正已用完，终态为 `BLOCKED_PRE_RUN`。没有提交 Stage-1 Slurm 作业，
  没有新的 validation、mAP 或成本结果；科学假设未被效能实验否定。

## 2026-08-23 — H65 Stage-1 直接复现启动

- 对 Cycle6 frozen revision `61397c0ed8beda8b0b87e1f6ca3be02046614e02` 重新按真实运行面执行准入。
  先前阻断的是 broad pytest 收集到两个与 H65 无关的隔离导入错误，不是 H65 模型、配置或训练入口失败；
  精确运行 checkpoint round-trip、Cycle3 与 Cycle4 H65 合同测试共 `19 passed`，launcher shell 语法和训练/
  validator Python 编译均通过。
- 一批真实 THUMOS14 GPU 训练 Job `1249963` 在 N16R4 完成（`COMPLETED 0:0`）并写出
  `/data/run01/sczc063/yuzibo/projects/duca_h65_matched_cycle6_direct_pre_run_61397c0e_20260823/gpu1_id0/checkpoint/epoch_0.pth`。
  checkpoint 包含模型、EMA、优化器、调度器、GradScaler 与成功 optimizer update 计数；未观察到顶层 RNG
  字段，因此该产物只证明真实训练和主要状态写入，不支持位级随机流恢复结论，也不包含 mAP 或成本证据。
- 正式 Stage-1 uniform-K384 Job `1249971` 已由同一 clean revision 启动并进入 `RUNNING`，seed 为 `3407`，
  训练长度为 30 epoch，预注册唯一终点为 epoch-29 `state_dict_ema`，结果根为
  `/data/run01/sczc063/yuzibo/duca_h65_stage1_uniform384_cycle6_61397c0e_20260823`。本阶段不访问结果选择用
  held-out，不启动 Stage-2，也不产生 H65 效能结论。历史 `65.385724%` 仍是旧 30+60 course 的诊断性终点；
  新复现必须等待 Stage-1 终点后，才可从同一 EMA 起点配对启动原始 H65-off 与 SingleClock-on。

## 2026-08-23 — H65 模型不变的总 60 轮课程候选实现并启动

- 历史 H65 的 `65.385724%` 来自 30 轮 exact-uniform K384 预热与 60 轮 learned H65 联合阶段。此前
  TrueTime 的 20+20+20 结果不能单独回答课程压缩问题，因为它同时改变了重型编码器的时间解释和若干
  训练合同。本轮因此建立独立 schedule-only 候选：不修改 H65 模型、输入方式、损失或评估器，只把总
  训练长度压缩为 Stage-1 20 轮，以及单次连续 Stage-2 40 轮（前 20 轮余弦过渡、后 20 轮完整联合）。
- clean worktree 为 `E:/DeskTop/TAD/OpenTAD_DUCA_H65First_MatchedCycle6_20260822`，分支
  `codex/duca-h65-60-curriculum-20260823`，冻结 revision
  `84acc15e948c213db48d1bc74a23d66ac868f7ca`。生产改动仅包括两份课程配置、一个阶段交接 validator 和
  一个 N16R4 launcher；没有改动 `opentad/models/` 或训练循环。Stage-2 handoff 还要求 Stage-1 输出目录中
  的已复制来源配置与仓库冻结配置一致，防止任意 epoch-19 checkpoint 被误接入。
- 目标环境相关测试共 `20 passed`，两阶段 PRECHECK 通过。真实 GPU 两步 PRE_RUN Job `1250163`
  `COMPLETED 0:0`，写出 epoch-0 checkpoint，成功 optimizer update 数为 2，405 个 optimizer state 的 step
  均为 2，第二步学习率非零；这只证明可训练性和非零参数更新，不是性能证据。
- 正式 Stage-1 Job `1250200` 已从 exact clean revision 启动并进入 epoch 0，结果根为
  `/data/run01/sczc063/yuzibo/duca_h65_60_stage1_uniform20_84acc15e_20260823`。Stage-2 在 epoch-19 EMA 及
  provenance/checkpoint 检查通过前不得提交。当前没有新的 validation、mAP 或成本结论。

## 2026-08-23 — H65 两种 Stage-1 终点性能

- 原始 30+60 H65 复现的 30轮 uniform-K384 Stage-1 Job `1249971` 与总60轮候选的 20轮 Stage-1
  Job `1250200` 均 `COMPLETED 0:0`，分别耗时 `04:41:47/03:13:58`。两者都使用 seed `3407`、
  `state_dict_ema`、211个官方 validation 视频和同 SHA 的官方 mAP evaluator。
- 30轮 epoch-29 EMA 的 Avg-mAP 为 `59.4231`，tIoU 0.3–0.7 为
  `77.0357/71.0208/62.2334/51.0882/35.7376`；20轮 epoch-19 EMA 为 `49.5389`，对应
  `70.6127/62.4926/51.8561/39.2215/23.5116`。压缩预热终点低 `9.8842` Avg-mAP，tIoU 0.7
  低 `12.2259` 点。
- 为区分训练长度和学习率日程，同为 epoch 20 的检查为：原30轮日程 `50.8707`，压缩日程
  `49.5389`，差 `1.3318` 点；其余约 `8.55` 点终点差距来自原日程继续训练到30轮时获得的成熟度。
  这说明20轮预热明显欠成熟，但不是总60轮候选的终局结论。两条 Stage-2 均尚未启动，当前不能报告
  learned H65 最终 mAP、课程压缩成功或失败。

## 2026-08-23 — H65 40轮/60轮 Stage-2 正式启动

- Stage-2 准入只修复运行合同，不改变模型：在原始 H65 Stage-2 config 显式登记已继承的 seed `3407`、
  60 epoch 和6000次更新；旧 launcher 增加 exact source `PYTHONPATH`，并把 smoke 从首个零学习率 step 扩为
  两步，以证明非零参数更新。最终本地/远端源码 revision 为 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`；
  远端相关测试 `19 passed`，H65-60/H65-90 Stage-2 PRECHECK 分别 `7/12 passed`。
- H65-60 Stage-2 smoke Job `1251561` `COMPLETED 0:0`，第二步学习率非零、课程时钟为1；正式40轮
  Job `1251622` 已在 exact `87ff0883651a631d48468ab4f9d6392f587c15e4` source 上运行，结果根为
  `/data/run01/sczc063/yuzibo/duca_h65_60_stage2_transition20_joint20_87ff0883_20260823`。
- 原始H65 Stage-2 第一次 smoke `1251562` 仅因同节点并发作业共享端口29500而在分布式初始化前退出；
  独立端口的一步 smoke `1251586` 完成但第一步学习率为零，故未用于正式准入。两步 smoke Job `1251740`
  在新 exact source 上 `COMPLETED 0:0`，第二步学习率非零、课程时钟为1。正式60轮 H65-off Job
  `1251782` 已运行，结果根为 `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823`。
- 两条正式作业都从各自唯一预注册 Stage-1 EMA 与 SHA 启动。当前仍无 learned H65 终态 mAP；smoke 和
  运行状态仅是准入/执行证据，不是课程压缩有效性证据。
## 2026-08-23 — H65-60 TrueTime-Aware Bridge 科学讨论请求

用户要求在总 60 轮 H65 候选上继续研究 TrueTime。当前把问题限定为表示层而不是重新设计 selector：保持 H65 的语义间接非均匀 K384 逐帧选择、课程、检测器和官方评估协议不变，比较第一次重型时序混合是否使用真实物理时间，以及 detector 前是否使用有 support/max-gap 约束的确定性物理时间重建。连续 cliplet 已有 49.89/47.24 的完整负结果，不作为候选；dense/uniform/random 历史控制不重复。

截至 22:38 +08:00，H65-60 Stage-2 Job `1251622` 与原始日程 Stage-2 Job `1251782` 均仍在运行，尚无终态 learned H65 mAP。已准备 `CURRENT_RESEARCH_STATE-v014.md` 和 `DUCA_H65_60_TRUETIME_BRIDGE_PRO_ADJUDICATION-v001.md`，请求 fresh exact-Project Pro 在 First-Mixing SingleClock、Deterministic Support-Aware Bridge 与分阶段 Clock×Bridge 因果分解之间冻结唯一方案。当前没有新实现、PRE_RUN、训练、指标或效能主张。

## 2026-08-24 — H65 First-Mixing SingleClock 正式运行与终态证据冻结

- 最终科学裁决冻结 `H65_FIRST_CROSS_TUBELET_BOUNDED_SINGLECLOCK_WITH_GATE_ZERO_TWIN-v001`：保持历史 H65 的语义间接非均匀逐帧选择、固定 `K=384`、RGB 集合、训练课程、VideoMAE-S/Adapter/ActionFormer、损失、NMS、split 和 evaluator，只在 VideoMAE 第 0 个注意力块加入共享、零初始化、有界的物理时间残差。Bridge 与 dynamic-K 在本表示门中关闭。
- clean worktree `E:/DeskTop/TAD/OpenTAD_DUCA_H65_FirstMixSingleClock_20260824` 的训练 revision 为 `08a817e91867839abf3a81e24f8469512b26a6ea`。GPU PRE_RUN 已证明非均匀时间产生有限非零梯度，uniform/gate-zero 恒等且 RGB 不变；正式 Job `1252482` 在 N16R4 运行 60 epoch/6000 次成功更新，首个 `epoch_4.pth` 已封存 500 次成功更新与完整恢复状态。该事实只证明运行和恢复合同，不是效能证据。
- 终态证据基础代码在 `668cfa9b3de86f6b86b89d84c6f565fab85d9091`，恢复状态与 OFF 身份的证据分级修正冻结在 `3089300f`；后者在 N16R4 通过 20 项聚焦合同测试。证据计划固定同 checkpoint final/EMA ON 与 gate-zero twin、同评估代码 H65 OFF 只读重推理、整视频 10,000 次配对 bootstrap、training-only 短动作/畸变阈值和同节点三次完整 validation workload 成本配对。终态回执还必须重算 ON/gate-zero 配置哈希并验证门状态，防止同一 ON 输出被误标为 gate-zero。
- 在终态证据代码冻结时，旧 RankPack/TrueTime 配对 bootstrap Job `1252515`、SingleClock 正式 Job `1252482` 与 H65 OFF Job `1251782` 尚未终态；当时没有从运行状态推断任何 mAP、置信区间或成本结论。H65 OFF 的后续终态审计记录见下一节。

## 2026-08-24 — H65 OFF 训练终态与恢复合同缺口

- H65 OFF Job `1251782` 已 `COMPLETED 0:0`。epoch-59 checkpoint SHA-256 为 `dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`；训练子配置与冻结证据 checkout 的对应配置字节一致，SHA-256 为 `73a5b75bce219b3df725a8e6f97a273a7ee6dd1e67c661fd33f261a681563867`。Stage-1 epoch-29 EMA SHA-256 仍为 `bcbc877c204a1ce7778f559be0b218295223367983450274671b17356e5be4e3`。
- 冻结审计确认 seed `3407`、60 epoch、6000 次成功更新、scheduler step 6000，以及 final、EMA、optimizer、scheduler、AMP scaler 均存在。checkpoint 不含 `rng_state` 与 `data_loader_state`，所以完整恢复合同失败；训练完成与可恢复性必须分开陈述。
- 证据层在 clean revision `3089300f` 上完成边界修正并经 N16R4 `20/20` 聚焦测试：Clock ON 仍严格要求完整 RNG/DataLoader 恢复状态；旧 H65 OFF 允许在只读终态推理中显式记录恢复缺口。H65 OFF final/EMA 各含一个严格为 `0.0` 的注册物理时间标量，resolved detector 配置的 SingleClock 准入关闭，因此属于结构注册但功能恒等的 OFF 身份。
- 该 checkpoint 可进行固定 final/EMA 只读重推理，且不阻止同 checkpoint ON/gate-zero 主因果评估；但缺失的恢复状态不能伪造。若后续性能门通过而该偏差仍存在，终态最多为修订并不得升级为论文级可复现实验。没有读取训练期中间 validation 作为终态结果，也尚未启动 ON/gate-zero/OFF 终评。

## 2026-08-24 — H65 总60轮课程压缩终态

- H65 总60轮候选 Job `1251622` 已 `COMPLETED 0:0`。冻结 epoch-39 EMA 在 211 个官方 validation 视频上的 Avg-mAP 为 `62.4648`，tIoU 0.3–0.7 为 `78.0914/73.4479/65.0772/55.7639/39.9434`；结果文件为 `/data/run01/sczc063/yuzibo/duca_h65_60_stage2_transition20_joint20_87ff0883_20260823/gpu1_id0/intermediate_validation/epoch_040_ema.json`。
- 原 30+60 H65 OFF 的冻结 epoch-59 EMA 为 `65.1257`，对应 `80.2808/75.7109/68.5475/57.7757/43.3137`；结果文件为 `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/intermediate_validation/epoch_060_ema.json`。两者 evaluator 源码 SHA 相同，均使用 seed `3407` 和完整 211 视频 validation。
- 压缩候选相对原日程的 Avg-mAP 低 `2.6609` 点，tIoU 0.7 低 `3.3703` 点。因此 20+40 课程没有保住 30+60 H65 的终点性能；这是训练日程压缩负证据，不是否定 H65 的语义间接非均匀选帧。两份终态 checkpoint 均缺少 RNG/DataLoader 恢复状态，结果可作冻结 EMA 诊断比较，但不升级为论文级复现结论。

## 2026-08-24 — H65 60-epoch 学习率与课程速率归因冻结为 designed

- exact DUCA Project 的独立 Pro 优化审查完成 `REVISE`。代码复核确认 `30+60` 与 `20+40` 的 H65
  模型、K384 非均匀逐帧选择、重型骨干、检测器和 evaluator 身份相同；性能下降同时伴随 Stage-1
  handoff 不成熟、Stage-2 更新从 6000 降到 4000、cosine 更早接近绝对 floor、语义退火和 feedback
  加速，以及 full-joint 尾段缩短，不能把 `-2.6609` 点全部归于某一项。
- 下一门复用原 Stage-1 epoch-29 EMA，保持参数组基准 LR、模型和损失终值不变，只运行两个
  30-epoch/3000-successful-update Stage-2 臂。主臂 `AM-RPCH25` 使用 500-step warmup、1000-step
  peak plateau、1000-step relative cosine 到 `0.25×`、500-step `0.25×` hold；归因臂
  `LongCosine-H6000` 使用 500-step warmup 和历史 6000-step horizon，在 3000 updates 处停止。
  两臂共同使用 2000-step semantic/policy transition、1000-step feedback warmup 加 1000-step cosine
  开启，并保留 1000-step full-joint tail。
- 两臂之后停止 scheduler 搜索；只有单种子终态 Avg-mAP 与 mAP@0.7 均进入原 `65.1257/43.3137`
  的 0.50 点工程保真带，才增加两个预注册种子。若最佳臂任一关键指标恢复不足 50%，承认固定 60-epoch
  预算无法无损压缩；若允许放宽计算，只运行一个 45-epoch Stage-2 exposure-dose 臂。当前仅完成 Pro
  裁决和冻结设计，尚未实现、PRE_RUN 或提交新训练。

## 2026-08-24 — RankPack/TrueTime bootstrap 执行诊断更正

- 先前“8 个外层 worker 再各启 16 个 evaluator 子进程”的诊断经直接调用链核验被撤回。
  `tools/bata/duca_p0_evaluation.py::_metrics_from_evaluator` 直接逐类调用官方
  `compute_average_precision_detection`；bootstrap 路径没有调用 evaluator 的多进程 `evaluate()`，
  因而 `thread=16` 只是未被该路径使用的元数据，不存在 8×16 的嵌套进程。
- revision `618181727fc49ffff3e92276453ce063235dfa58` 的 thread/chunksize 改动通过合同测试，但没有消除
  真正的耗时来源。原 Job `1252515` 的取消和替代 Job `1252592` 只属于执行核算尝试，不能表述为已修复
  的并行度问题，也没有产生新的统计或科学结果。
- 当前确认的瓶颈是每个 resample、每个 family 重建 JSON、重新读取、构造 pandas 表并重复执行官方逐类
  AP 循环。下一执行改动只能在保持 PCG64 抽样矩阵、重复视频的独立 synthetic id、动态类别顺序、
  score-tie 顺序、未知类别处理、ranks `250/9750` 和官方 AP 核心完全一致时，改为内存数据路径；必须先
  通过 legacy JSON 路径逐项等价测试和目标环境小规模基准，再决定是否替换正在运行的 Job `1252592`。

## 2026-08-24 — First-Mixing SingleClock 训练终态与统一终评恢复

- 正式训练 Job `1252482` 已 `COMPLETED 0:0`。训练 revision 为 `08a817e91867839abf3a81e24f8469512b26a6ea`；epoch-59 checkpoint SHA-256 为 `720f41912eca47ccc9c0413b711d7fbc75f3a4175d983125d1a2df6015d3ae23`，包含 final、EMA、optimizer、scheduler、AMP scaler、随机状态、DataLoader 下一 epoch=60 和 6000 次成功更新。训练完成不等于机制有效，尚未从中间 validation 推导论文结论。
- 统一终评工具在 clean `12b96132a88b338dbf6f14997ee4e23b5e91733b` 上通过独立静态复核和 N16R4 29 项聚焦测试。首次终评 Job `1252897` 在第一个 validation batch 的选中输入身份封存处失败：数据集给出的 `window_start_frame` 是 NumPy `int64`，不能被标准 JSON 编码器直接序列化。该错误发生在证据哈希记录，不在模型前向、checkpoint、数据或官方 evaluator 中，因而不是性能结果或科学失败。
- 最小修正 `e866a9ae52dd64b775854029d09ce72a6c86ad01` 只把 NumPy 标量归一化为 Python 标量并加入对应回归测试；独立复核为 `PASS_TO_EVALUATOR`，N16R4 正式环境按启动器冻结的 cycle4/bootstrap/strata 测试集合得到 `30 passed`。模型结构、权重、选中 RGB、物理位置、阈值和评价协议均未改变。
- 统一终评恢复 Job `1252954` 已从相同 SingleClock epoch-59 checkpoint、相同 H65 OFF checkpoint 和相同 Stage-1 checkpoint 启动，结果根为 `/data/run01/sczc063/yuzibo/duca_h65_singleclock_terminal_eval_e866a9ae_20260824`。预期回执包含 final/EMA ON、gate-zero twin、H65 OFF final/EMA、10,000 次整视频配对 bootstrap 和 training-only 分层证据；在终态完成前仍无可报告的 SingleClock mAP、置信区间、分层收益或成本结论。

## 2026-08-24 — H65 60轮学习率归因实现与正式训练启动

- clean worktree `E:/DeskTop/TAD/OpenTAD_DUCA_H65_LRSchedule_20260824` 冻结 revision `ae3642a138c5b2e1ac2daad75a6d43d17cdb6c2f`。两臂都复用 Stage-1 epoch-29 EMA `bcbc877c…`，保持 H65 模型、固定 K384 非均匀逐帧选择、数据、seed `3407`、损失、检测器、评估器和参数组基础 LR 不变；只比较 `AM-RPCH25` 与 `LongCosine-H6000` 的成功更新调度。
- 实现补齐 checkpoint 中的 model/EMA/optimizer/scheduler/AMP、Python/NumPy/Torch/CUDA RNG、epoch-boundary DataLoader 状态与完整累计 update audit。N16R4 17 项聚焦测试通过；fresh PRE_RUN `1252972/1252973` 各完成 2 次真实更新，恢复 PRE_RUN `1252975` 从累计 2 次继续到 4 次，scheduler/EMA/selector 时钟均为 4。规范 THUMOS14 根路径绑定后的复核 Job `1252977` 也为 `COMPLETED 0:0`。
- 完整 Stage-2 Jobs `1252979/1252980` 已从同一 Stage-1 checkpoint 提交，各为 30 epoch/3000 次成功更新、每5 epoch 可恢复 checkpoint、固定 epoch-29 final-EMA。当前只有执行准入证据，没有新的 validation、mAP、成本或课程有效性结论；中间验证不用于挑选 checkpoint。

## 2026-08-24 — SingleClock 统一终评的第二个证据实现阻断

- Job `1252954` 在 `final_on` 的完整 validation 后段遇到重复 `video_test_0001431|window_start_frame=7680`。官方滑窗数据可以出现相同 `(video, start)` 的多次观测，而证据封存器当前把该二元组当作全局唯一键并 fail closed；作业在 360/396 batch 处停止，其余 family、bootstrap、分层和成本均未执行。
- 该错误只影响样本身份封存，不改变 SingleClock checkpoint、选中 RGB、物理时间、模型前向或官方 evaluator；但没有 family 形成终态指标，故禁止从部分日志推断 mAP。下一修正只能为合法重复窗口增加确定性 occurrence 身份并保持 ON/gate-zero 一一配对，不能改变数据集或科学阈值。

## 2026-08-24 — SingleClock 重复尾窗身份闭合、PRE_RUN 通过与终评重启

- 对 `video_test_0001431|window_start_frame=7680` 的进一步代码与真实数据核验纠正了上一条中的 occurrence 方案：该现象是同一物理尾窗被数据加载器完全相同地暴露两次，不是两个应当由 batch 序号区分的窗口。最终实现保持物理身份 `(video_name, window_start_frame)` 不变；只有 RGB、选中位置、掩码和有效长度完全一致的重复暴露才合并为一个物理记录，并显式保存总暴露数、唯一窗口数和重复计数；同键内容不一致继续 fail closed。这样既验证了两次暴露的一致性，也不把同一物理窗口伪装成两个独立样本。
- 证据修正从 `e866a9ae...` 经独立实现与三次有界聚焦复核冻结在 clean revision `b2ccfccab5b4912b59954afcc9b0364955327f7c`。最终合同同时要求记录数量、字段类型、身份唯一性和排序、重复成员关系，以及所有物理位置严格满足 `0 <= p < dense_valid_len` 且递增。改动只涉及 ActionFormer 的输入合同、终态身份 finalizer 及两份聚焦测试；没有改变有效输入上的模型计算、选择器、checkpoint、数据、评估器、配置、启动器或科学阈值。终态独立复核为 `PASS_TO_EVALUATOR_PRE_RUN`。
- N16R4 PRE_RUN Job `1253016` 已 `COMPLETED 0:0`：远端 checkout 精确为 `b2ccfcca...` 且 clean，规范 THUMOS14 入口解析为 411 个 MP4；三份冻结 checkpoint 的 SHA-256、分布式环境、Python 编译、两份 checkpoint 审计均通过，目标环境聚焦测试为 `49 passed`。该证据只说明运行准入，不说明机制有效。
- 冻结的六族终态评估 Job `1253017` 已提交，结果根为 `/data/run01/sczc063/yuzibo/duca_h65_singleclock_terminal_eval_b2ccfcca_20260824`。它将顺序执行 final/EMA ON、同 checkpoint gate-zero twin、H65 OFF final/EMA、10,000 次整视频配对 bootstrap 和仅由训练 population 冻结的分层分析；终态回执形成前仍禁止解释部分 mAP、置信区间、分层或成本结果。

## 2026-08-24 — SingleClock 正式终评启动路径恢复

- Job `1253017` 在 10 秒内由启动器防覆盖检查主动停止：提交命令把 Slurm stdout/stderr 放入 `DUCA_EVAL_OUTPUT_ROOT`，从而在启动器运行前创建了本应不存在的结果根。它没有进入测试、checkpoint 加载、数据迭代、推理或指标计算，因此不构成实验失败。
- PRE_RUN `1253016` 仍有效；在不改源码、checkpoint、配置、seed、评估器、nonce 或统计协议的条件下，只把 Slurm 日志移到独立目录，并为启动器提供新的空结果根。恢复 Job `1253023` 已在 `g0041` 进入 `RUNNING`，结果根为 `/data/run01/sczc063/yuzibo/duca_h65_singleclock_terminal_eval_b2ccfcca_r1_20260824`。在完整 terminal receipt 形成前仍无新的 SingleClock 性能或成本结论。

## 2026-08-24 — SingleClock 四族完成与 H65 OFF 只读兼容恢复

- Job `1253023` 已完整写出 `final_on`、`final_gate_zero`、`ema_on`、`ema_gate_zero` 四个 family 及逐窗口身份材料；独立 Evaluator 对这四族的结构准入结论为可接受。作业随后在 H65 OFF checkpoint 严格加载处停止，未执行 OFF 推理：checkpoint 含 `module.backbone.model.backbone.blocks.0.relative_physical_time_scale`，而原 OFF config 没有注册该参数。
- 训练 checkpoint 的 final/EMA 中该标量均严格为零，H65 OFF 的 `single_clock_admission=False`。代码核验确认：只在运行时启用 `relative_physical_time_residual=True` 以注册现有零标量，同时保持 admission 关闭，ActionFormer 不传物理坐标，骨干计算与原 H65 OFF 完全一致。这是 evaluator/config 身份兼容，不是模型、checkpoint 或科学路线修正；禁止使用 `strict=False`、改写 checkpoint 或更改 OFF 计算。
- 两个自定义预检包装分别因相对预训练路径和未导入数据变换注册表在推理前停止，没有产生性能证据。依据第二次等价确定性包装缺陷，已删除额外预检层，改由权威 `tools/test.py` 直接严格加载并完成推理。
- 恢复 Job `1253090` 已通过 H65 OFF final checkpoint 的严格加载并进入完整 validation；随后将执行 EMA OFF、训练集冻结的 SingleClock EMA 身份、六族合并、10,000 次整视频配对 bootstrap 与分层统计。所有终态材料形成前禁止披露部分 mAP 或推导机制结论；同节点完整 validation 成本测量仍待性能与分层证据准入后执行。
- Job `1253090` 随后完成 H65 OFF final/EMA 两次完整 validation、两套 state_dict 的严格加载兼容回执，以及训练 population 的 SingleClock EMA 身份与分层冻结；当前进入 final/EMA 的 10,000 次整视频配对 bootstrap。旧 RankPack/TrueTime bootstrap 由既有 Job `1252592` 独立执行，禁止重复提交。两套 bootstrap、分层和终态 finalizer 完成前仍不报告部分指标。

## 2026-08-24 — H65 60轮学习率日程独立复核与终态判据冻结

- 在精确 DUCA Project 的全新 Pro 会话中，对历史 `30+60`、失败的 `20+40` 以及当前 `30+30` 的 `AM-RPCH25/LongCosine-H6000` 两臂完成独立训练动力学复核。裁决为 `CONTINUE / HOLD_NEW_TUNING_UNTIL_TERMINAL`：只完成已经运行的 Jobs `1252979/1252980`，终态前不新增第三日程或参数组学习率微调。
- 复核确认性能下降不能简化为“只少训练30轮”：失败 `20+40` 同时使用未成熟的 epoch-19 Stage-1 EMA，并减少 Stage-2 minibatch/optimizer/EMA 暴露、压缩 semantic/policy 与 detector-feedback 时钟和 full-joint 尾段。当前 A/B 都复用成熟 epoch-29 EMA，且模型、K384 非均匀逐帧输入、损失、数据、seed、参数组和基础学习率不变；但 A/B 相对历史锚仍同时缩短更新和课程时钟，所以只能回答某个完整 3000-update 日程包能否恢复性能，不能把差异全部归因于 LR 曲线几何。
- `LongCosine-H6000` 的3000步累计相对学习率剂量约比 `AM-RPCH25` 高 `18.33%`，且它是保持参数组比例的历史长 horizon 类比，不是 legacy 绝对 `eta_min` scheduler 的 bit-exact 截断。两臂之间的比较仍有效，但 winner 只能解释为完整 schedule package。
- 单 seed 恢复邻域冻结为 epoch-29 EMA Avg-mAP `>=64.6257` 且 mAP@0.7 `>=42.8137`，并通过精确3000 successful-update、同一 Stage-1 SHA、final/final-EMA、官方评估器和稳定性门；Avg-mAP `<64.1257` 或 mAP@0.7 `<42.3137` 为明确失败。中间 checkpoint 不能选择 winner。若两臂都低且终端仍上升，优先承认缺少 Stage-2 exposure，而不是提高基础 LR；若低且平台/下降，则停止“60轮无损压缩”子问题，保留历史 `30+60` recipe。
- 该讨论只冻结训练协议和结果解释边界，不构成 H65、SingleClock、动态预算、成本或论文创新证据。终态后先做独立身份审计和 result-blind 评价，不为补齐缺失诊断重训。

## 2026-08-24 — H65 90→60轮压缩因果诊断与终态分支冻结

- 精确 DUCA Project 的全新 Pro 会话再次基于真实 30+60、失败 20+40 和正在运行的 30+30 A/B 配置完成训练动力学裁决，结论为 `CONTINUE`，但仅授权 Jobs `1252979/1252980` 运行到唯一 terminal EMA；两臂结束前仍禁止第三日程、峰值或参数组基础学习率调整以及中间 checkpoint 选择。
- 证据排序被冻结为：20轮 Stage-1 交接不成熟是最强直接信号；Stage-2 successful updates 与 full-joint 尾段不足是次强解释；LR horizon、累计剂量和非零尾段由当前 A/B 直接检验；semantic/policy clock、feedback clock 和 EMA lag 仍是未隔离假设。旧 20+40 的 `-2.6609` Avg-mAP 与 `-3.3703` mAP@0.7 不能被单独归因于学习率，也不能证明60轮原则上不可能恢复。
- 终态分支保持既有恢复邻域与明确失败双门。至少一臂进入恢复邻域时停止 scheduler 搜索；没有通过但有灰区，或两臂明确失败而 epoch `19/24/29` 的 Avg-mAP 与 mAP@0.7 同时满足预冻结上升条件时，只允许从预注册 parent checkpoint 无重置连续增加 `1000` 次 full-joint update，形成最多70轮的暴露诊断；两臂明确失败且平台/下降则终止60轮压缩子问题并保留30+60参考。
- 该延长若被触发，必须恢复 model、EMA、optimizer、scheduler、AMP scaler、successful-update/curriculum 时钟、随机数和 DataLoader/sampler 状态，基础学习率与模型身份不变；若恢复状态不完整则 fail closed。该讨论没有产生新 mAP、没有授权当前作业之外的训练，也没有改变 H65 科学路线。

## 2026-08-25 — DUCA 全矩阵同资源独立审查终稿

- 对官方 dense、H65、训练压缩、学习率归因、TrueTime、UVT、Fovea/Query-Bridge、连续 cliplet 与
  SingleClock 完成同一份独立审查。裁决为窄义 `CONTINUE`：只完成 First-Transformer-Mixing
  SingleClock Gate-v2 的既有 artifact 准入；不授权重训、Query、Bridge、dynamic-K 或新性能实验。
- H65 `65.1257/65.3857` 被明确归类为复合配方结果，不能把收益单独归因给语义 selector。当前最可信的
  性能下降因素依次是 Stage-2 训练成熟度/暴露不足、selected-rank 伪连续时间语义、未成熟且混杂的
  dynamic-K、Fovea 训练/推理 policy 不一致，以及尚未隔离的多目标梯度冲突。
- SingleClock 的现有 terminal-EMA ON 相对 matched H65 的跨运行根算术差为 Avg-mAP `−0.6596`、
  mAP@0.6 `−0.4037`、mAP@0.7 `−0.1720` 个百分点，只能作为负向预警。合法终结必须先恢复 H65 五边界
  replay、ON/gate-zero 输入身份和 canonical-uniform 骨干逐位身份，再按三项 point delta `>=−0.20`
  判断；置信区间仅报告。任一既有边界无法确定性恢复时返回 `EVIDENCE_INVALID`，不补跑或重训。
- 独立终稿保存在 `.cvpr-pro-lab/pro-reviews/runs/duca-all-matrix-same-budget-v001/PRO_DUCA_ALL_MATRIX_SAME_BUDGET_ADJUDICATION-v001.md`。

## 2026-08-25 — 稀疏 Token 物理时间表示跨领域 Pro 裁决材料就绪

- 新科学问题不是再次调参 SingleClock，而是处理其尚未覆盖的表示缺口：H65 的 384 个非均匀选中 RGB 帧在 Conv3D PatchEmbed 中仍按 selected rank 两两混合，真实物理时间只在之后的第 0 个 Transformer attention 生效；后置 proposal 回映不能恢复已发生的时序混合。
- 已将 video token pruning/merging、视觉原始坐标保持、LLM token/KV 删除后的 position-ID/RoPE/relative-bias 处理，以及不规则连续时间建模组织成一次对抗性裁决。候选只允许作用在 PatchEmbed 之前或内部、首个 token interaction，必须说明 VideoMAE-S 预训练兼容、timestamp/support/padding、TAD 高 IoU 边界和真实 K384 计算合同。
- 本轮保持 H65 同一 selected RGB、固定 K=384、30+60/6000 successful updates、detector/loss/NMS/evaluator/seed 不变；不重复 dense/uniform/random，不恢复连续 cliplet，不加入 dynamic-K/Query/UVT，也不提前解释仍在运行的 Job `1253090`。
- 完整内联请求为 `.cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_SPARSE_TOKEN_PHYSICAL_TIME_TRANSFER-v001.md`；等待 exact DUCA Project 的一次 fresh Pro 裁决后，才决定唯一最小实现与真实 falsifier。

## 2026-08-25 — PJST Pro 用户提供终稿的核验与吸收

- 用户提供的 `DUCA 稀疏 Token 物理时间表示终态裁决` 已逐字归档到
  `docs/methods/reviews/2026-08-25-b2ccfcca-duca-pjst-pro-response-user-supplied-raw.md`，SHA-256 为
  `d1dce144eeff2b2bc474154df948b20b82536252df1f24cc40e4d84b62a02160`；完整项目核验写入
  `docs/methods/2026-08-25-b2ccfcca-duca-pjst-pro-review-absorption.md`。
- 核验接受“物理时间必须进入 VideoMAE 首次 temporal kernel=2 重型混合”这一主要诊断，并保留零参数、
  canonical-uniform 直接旁路的 PJST 为 `designed_candidate`。它比继续在 PatchEmbed 后加入时间 bias、
  Query 或 Bridge 更直接地针对当前表示缺口。
- 项目不接受把一次完整 30+60 联合训练同时描述为 selected RGB 不变的纯表示实验：表示变化会经梯度
  改变 selector。执行前必须冻结为系统级总效应或固定/重放 selector 的表示归因，不能混称。
- 新增静态问题包括：全局 K384 support 必须先于 24×16 切片构造；有效位置必须严格递增且无 duplicate；
  native physical head 与后处理不得二次映射；support 是归一化插值而非完整时间积分；PJST 只修复 pair
  内首次混合，不代表后续 rank-time 语义已经完备。
- 正式浏览器调用的本地 receipt 未被验收为 exact-DUCA completed turn，因此该终稿只登记为用户提供的
  独立审查，未伪造或改写旧路由证据。本轮无代码、PRE_RUN、训练或 mAP 变化，`PAPER_PROGRESS.md`
  保持不变。

## 2026-08-25 — PJST 终稿二次核验：统计修正与最小机制收缩

- 在干净 DUCA `b2ccfccab5b4912b59954afcc9b0364955327f7c` worktree 上复核真实调用顺序：
  `vit_adapter.py:889` 先执行 Conv3D PatchEmbed，物理时间残差随后才进入第 0 个 Transformer block；
  因而“first heavy mixing 尚未看到物理时间”的诊断成立。当前 dirty `a6bdc084...` 根目录不是代码证据。
- 原终稿把 10,000 个排序 bootstrap 样本的第 500/9500 个值称为双侧 95% 区间，这是明确错误；它约为
  90% 中心区间。未来合同必须预先冻结 2.5%/97.5% quantiles、索引基准和插值规则。
- 为避免同时测试 support quadrature 与物理差分两个假设，首门推荐收缩为 derivative-only PJST：保留
  `(x_i+x_{i+1})/2`，只用 `bar_delta/delta` 重标定差分；support 仅作审计元数据。首个正式归因实验
  优先固定/重放 selector，端到端 selector mediation 后置。该方向仍为 `designed_candidate`，没有新代码、
  PRE_RUN、作业或结果。

## 2026-08-25 — fresh exact-DUCA 裁决冻结 PJST-D1，进入干净实现周期

- fresh exact-DUCA completed Pro 终态裁决为 `REVISE`，冻结唯一机制
  `PJST-D1: Derivative-Only Physical-Jacobian Scaled Tubelet`：普通 pair mean 保持不变，只按
  canonical-gap/physical-gap 重标定 frame difference；exact-uniform 输入在任何 cast/division 前直接旁路，
  support/Voronoi 仅用于审计。
- 首个因果问题固定为 selector 表示归因，而不是端到端系统总效应。matched OFF/ON 必须从同一 Stage-1
  terminal checkpoint 出发，冻结/重放 selector，并逐 exposure 证明 positions、RGB、valid mask、RNG 和
  executed K 相同；唯一介入是 pre-PatchEmbed PJST-D1。
- 实现只允许落在 temporal-grid、backbone/PatchEmbed、必要的共享 pre-NMS physical remap、一个配置和
  focused tests；禁止改变 H65 selector、ASFormer、K384、VideoMAE/Adapter/ActionFormer、loss、optimizer、
  schedule、split、NMS 或 evaluator。
- 当前为 `designed_frozen / implementation_starting`。尚无 PRE_RUN、训练、mAP、置信区间或成本结果；
  通过独立代码审查和 Evaluator PRE_RUN 之前不启动 matched OFF/ON 正式实验。

## 2026-08-25 — PJST-D1 首个实现包在 focused recheck 后关闭

- Builder 从 clean `b2ccfcc...` 形成 `877d893f...`，独立审查发现 forward 接口、metadata reachability、
  canonical generator 和测试覆盖阻塞；一次 claim-preserving focused correction 形成 `84325205...`。
- focused recheck 仍发现可执行路径把 `[B,192]` pair metadata 重复为 `[B*24,192]`，而每个 16-frame
  clip 只接受 8 个 pair；temporal checkpointing 未同步切分 pair metadata；matched OFF 配置缺失。
  角色收据也被误提交进生产 snapshot。
- 这是第二个等价确定性实现缺陷，当前实现包终止，不再做第三次 Builder/recheck，也不交 Evaluator、
  PRE_RUN 或 Slurm。Windows Torch DLL 阻断测试收集是环境事实，但不是上述静态阻塞的原因。
- 当前证据仍为 `designed_frozen / implementation_package_closed / no PRE_RUN / no result`。没有效能实验，
  因而 PJST-D1 科学路线未被否定。

## 2026-08-26 — PJST-D1 第二个实现包关闭，新 clean 周期获授权

- 从 clean `b2ccfcc...` 形成的 `987f4811...` 及 focused correction `c8faf96b...` 未通过独立复核：
  OFF 仍构造 PJST metadata，selected-to-physical remap 仍晚于 confidence/top-k，候选测试没有执行
  production transform/gradient/checkpoint/remap 合同，新增 shell 入口也不能真正表达 matched OFF/ON 训练。
- 两个提交均未进入 Evaluator PRE_RUN、数据访问、Slurm 或效能评估；这是第二个实现包的确定性负证据，
  不是 PJST-D1 假设的实验否定。
- 用户已明确要求继续落实正确代码。新的 clean implementation cycle 从 `b2ccfcc...` 独立建立，禁止
  cherry-pick 两个失败提交；冻结科学合同、H65 30+60 训练合同与 matched OFF/ON 因果问题保持不变。

## 2026-08-26 — PJST-D1 clean 后继实现通过 PRE_RUN，matched OFF/ON 开始正式训练

- 新周期从 clean `a16a67c4...` 建立；`c195b97c...` 只修正不变性测试和生产接口替身，独立静态审查
  通过。第一次正式提交在数据集构建前暴露启动器仍写入未被训练程序读取的 `data.*` 配置树；第二次
  提交在完整模型加载前暴露冻结 Stage-1 的 block-0 零标量没有在 Stage-2 目标结构中注册。两次均未
  发生 optimizer update、checkpoint 或 metric，属于运行链负证据而非模型性能结果。
- focused correction `dc260fad...` 将 canonical THUMOS14 视频、annotation 和 class map 绑定到配置真实消费的
  `THUMOS14_* -> cfg.dataset.*` 接口；`c73e8418...` 在两臂共同注册冻结 H65-OFF state 中已存在且严格为
  零的 `relative_physical_time_scale`，同时显式保持 `single_clock_admission=False`。没有放宽严格加载、
  更换 checkpoint、启用 SingleClock 或改变 PJST/selector/optimizer/schedule/evaluator。
- N16R4 fresh checkout 对 `c73e8418...` 完成 31 项 focused tests、validator、canonical path 解析、两个
  config 的生产模型构建和 epoch-29 EMA 严格加载；两臂 state 均为 579 项，零标量逐值等于 0，OFF/ON
  仍仅在 `pjst_derivative_only` 和输出根上不同。
- matched OFF/ON 已从同一 Stage-1 EMA、seed 3407、固定 K=384 和全量 THUMOS14 协议进入 60 epoch 正式
  训练。两臂均已完成 epoch 0 第 50 次真实更新；当时 loss 分别为 `4.2918/4.2920`，预算均为 384，未见
  Traceback、非有限 loss 或显存错误。该早期数值只证明执行链运行，不是效能证据；终态 mAP、配对统计和
  成本结论仍为尚无。

## 2026-08-27 — PJST-D1 matched 训练完成，负向点估计与成对统计终结启动

- `c73e8418...` 的 matched OFF/ON 均完成 60 epoch、6000 次成功更新和全量 211 视频官方 validation。
  epoch-59 EMA 的 OFF/ON Avg-mAP 为 `65.063/64.591`；ON−OFF 为 `-0.472` 点。mAP@0.6 为
  `58.033/57.742`（差 `-0.290` 点），mAP@0.7 为 `43.646/43.769`（差 `+0.123` 点）。该点估计
  不满足预注册的正向支持门，不能声称 PJST-D1 有效；它也不足以单独签发否定门。
- 原训练流程只在内存中完成官方评估，没有保存逐视频 prediction，因而不能从标量 mAP 合法重建成对区间。
  当前只读冻结 epoch-59 EMA，使用同一数据、配置、评估器和 NMS 重新导出两臂逐视频 prediction，再执行
  固定 10,000 次整视频成对 bootstrap。终结实现只增加结果封存与统计调度，不改变模型或科学合同。
- 首次终结重推理在配置加载前因缺少 Stage-1 checkpoint 环境绑定退出，未生成任何部分预测；该确定性运行
  缺陷已用原 epoch-29 路径、SHA256 和轮次作一次精确恢复。新的 OFF/ON 重推理及其依赖的单作业统计终结
  已启动。最终状态仍是 `tested / paired_CI_pending`，尚无 PJST-D1 置信区间、成本或论文 claim。

## 2026-08-27 — 论文实验优先与 Pro 前后闭环成为项目规则

- 后续工作以论文问题、模型实现、完整真实训练、官方评测和决定性结果为主线。复杂合同、通用框架、重复审计
  或版图整理不得成为研究目标；只有会改变模型、破坏数据/公平性/指标真实性或阻止执行的问题可阻塞实验。
- 实现、环境、启动器或证据封存故障不构成科学路线失败，只允许最小修复后继续。任何路线转换前必须先闭合
  失败根因、混杂因素与可证伪结论，不能用工程失败替代科学否定。
- Pro 在新执行前负责冻结唯一任务、最小实现、决定性实验和截止时间；正式实现及完整训练/评测后必须再次
  审阅实现忠实度、成功或失败原因、论文可发表性、是否达到终局以及下一路线。分角色规则固定在
  `research-wiki/PRO_RESEARCH_ROLE_RULES.md`，其用途是减少空转，而不是新增工程门禁。

## 2026-08-27 — PJST-D1 结果审阅冻结一次只读统计终结

- 结果审阅确认 `-0.472` 个百分点 Avg-mAP 是完整 60-epoch、单 seed、211-video validation 点估计：当前
  没有正向支持，但没有成对区间时不能称为正式负结果，也不能把 mAP@0.7 的 `+0.123` 点解释为收益。
- jobs `1257283/1257284` 在正式推理前因基础配置中的相对 VideoMAE-S 预训练路径退出；这是确定性评估绑定
  故障，不是模型证据。训练 launcher 已使用 canonical absolute path，剩余修复仅限终结 eval launcher 的
  同一路径覆盖，不得改模型、checkpoint、selector、数据、NMS、evaluator 或统计量。
- 修复经一次独立静态审查通过后，只复用既有 OFF/ON epoch-59 EMA 做 211-video 只读官方推理、封存预测和
  10,000 次整视频成对 bootstrap；禁止重训。网页会话存在同 profile 并发与终态捕获失败警告，但 exact
  DUCA Project、nonce、材料和完整可见回复一致，未发现跨项目污染；该警告只影响传输记录，不是科学结果。

## 2026-08-27 — PJST-D1 完整点复现通过，bootstrap 在抽样前阻断

- clean `7bd120f0...` 的 OFF/ON 只读重推理均完成：两臂各覆盖 211/211 视频、产生 422,000 条预测，视频 ID
  集合完全相同；epoch-59 `state_dict_ema`、官方 evaluator、数据、NMS 和类别映射保持冻结。
- 两臂所有 mAP 与原 epoch-60 EMA 记录逐项精确复现。Avg-mAP 为 `65.063283/64.590802`，ON−OFF 仍为
  `-0.472481` 个百分点；mAP@0.7 的 `+0.122739` 点不能被单独解释为收益。
- 唯一统计终结器把 prediction 绑定为 `work/result_detection.json`，而单卡官方评估实际输出在
  `work/gpu1_id0/result_detection.json`。它在任何 bootstrap shard 或 draw 前退出：`0/16` shards、
  `0/10,000` replicates，无 paired CI、无 PASS/KILL。
- 因此当前只接纳“完整点估计身份复现且没有正向支持”；正式负向总体结论仍不可签发。该路径错误不是科学
  失败，本轮不修复、不重提、不重训，PJST-D1 科学假设保持未决。

## 2026-08-28 — 完整代码库存同步，首轮科学问题转为路线与代码主线裁决

- 当前模型、配置、启动器、测试、Wiki 和论文进展已形成 GitHub 可访问的库存分支
  `codex/duca-research-sync-20260828`。库存代码固定于 `5136011...`，路线与代码整理提示固定于
  `29eb7798...`。该分支用于检查重叠实现和历史边界，不是经过运行前检查的实验候选；H65、PJST-D1、
  UVT、Fovea/Query-Bridge 等结果仍分别以各自 clean revision 为实验身份。
- 当前没有运行中的 DUCA 正式训练。H65 30+60 仍是最强干净语义间接选帧参考；PJST-D1 只有匹配
  OFF/ON 点估计与未完成的配对区间；UVT、Fovea/Query-Bridge、连续片段和 60 轮压缩保留为负结果或
  诊断证据，不机械重复。
- 下一项科学问题不是让 Codex 预选新结构，而是由 Pro 直接核对固定提交与原始证据，独立决定值得继续的
  论文问题，指定一个最小权威代码主线，并明确保留、历史封存和移出当前候选的实现表面。随后只下达一个
  能改变论文判断的实现或实验任务；在该裁决前不合并路线、不启动新训练。

## 2026-08-28 — Pro 路线裁决提交在认证前阻断，原始请求已恢复

- 向 exact DUCA Project `g-p-6a9061a41bbc819190f4cde94a6c733c` 发起的路线与代码整理请求在正文提交前
  因 profile 61 未认证退出：`0` 次实质提交、未创建 conversation、没有 Pro 科学裁决，也没有 Source、代码或
  实验变更。该事实不能解释为 Pro 拒绝路线或科学结果。
- 原始完整 prompt 已从同步 worktree 恢复；其证据 commit `5136011...`、提示 commit `29eb7798...` 和远端分支
  `codex/duca-research-sync-20260828` 均可访问，不需要重写或预选科学方向。
- 03:03（北京时间）的只读 CDP 复核仍显示 exact Project 重定向到 `/auth/login`。当前唯一 blocker 是该专用
  profile 的人工 ChatGPT 登录/2FA。认证完成后应申请新的独占 browser lease，在同一 exact Project 中用原
  prompt 开启全新 Pro 对话；在此之前禁止重复提交、回退旧 Project 或启动未经裁决的新实验。
- Oracle 0.17.1 离线 bundle 预检确认完整 prompt 约 5,479 tokens，可作为单个 inline file 装载。原失败命令
  的 `gpt-5.6-sol` 只固定 base Sol，恢复时必须改用 dry-run 已确认的 `gpt-5-pro -> target=Pro` picker；若
  无法验证 Pro 选择则失败闭锁，不能把普通 Sol 回复登记为科学负责人裁决。

## 2026-08-28 — Pro 冻结 H65 语义动态预算的同预算因果实验

- 在新的权威 DUCA Project 中完成 fresh Pro 科学裁决，终稿为 `REVISE`。当前不补齐 PJST-D1 配对区间，
  不恢复 UVT、Fovea/Query-Bridge 或连续片段路线；从 clean H65 `04c35a3...` 建立唯一新候选。
- 科学问题固定为：每视频总高分辨率帧数严格等于 `384 × 窗口数` 时，按冻结动作性不确定度和边界重要性
  为窗口分配 `K=256/384/512`，是否优于拥有完全相同 K 多重集和实际重型工作量的内容无关预算置换。
  两臂的窗口内帧仍由同一 H65 语义排序选择。
- 当前实现任务必须消除最大 K padding，按真实 K 分桶执行 VideoMAE，保持原始时间坐标、一次逻辑 batch
  的样本加权损失与单次 optimizer update。detector、loss、NMS、split、evaluator、Stage-1 checkpoint、
  Stage-2 6,000 次成功更新、seed 3407 和 terminal EMA 均冻结。
- 正式开发种子将在完整 THUMOS14 上比较语义与置换两臂，报告官方 mAP、10,000 次整视频配对区间和真实
  VideoMAE/端到端成本。当前尚无新代码、PRE_RUN、训练、动态预算 mAP 或成本结果。

## 2026-08-28 — 动态预算实现前核验发现训练窗口合同冲突

- 已从 clean H65 `04c35a3...` 建立独立分支 `codex/duca-semantic-budget-matched-20260828`，尚未写入模型改动。
- 代码核验确认，H65 训练集以视频为样本，每个训练轮次通过 `random_trunc` 只抽取一个 768 帧窗口；批大小为 2
  且由普通分布式采样器打乱。同一视频的固定滑动窗口集合只在 validation/test 存在。
- 因而原冻结的“同视频窗口内排名并保持总预算 `384*n`”若直接写入当前模型前向，训练时会退化为
  `n=1, K=384`，不能检验动态预算。固定滑动窗口训练、冻结侦察器离线生成预算清单或按视频组织两遍数据
  流程都会改变训练合同，必须由 Pro 明确裁决，不能由代码实现默默选择。
- 这是一项实验设计与公平比较冲突，不是代码失败或动态预算负结果。现有动态预算仍补齐到最大 K、未兑现
  VideoMAE 工作量的事实保持不变；在合同修订前不启动训练。

## 2026-08-29 — Pro 修订并冻结动态预算的全视频滑窗实验合同

- Pro 在同一动态预算科学问题上给出 `REVISE`：训练和评估统一改用长度 768、步长 384 的完整视频滑窗总体，背景窗口不删除；一个训练样本包含一个视频的全部窗口，两个视频构成一个逻辑批次，60 个训练轮次仍严格对应 6,000 次成功更新。
- H65 Stage-1 epoch-29 指数移动平均侦察器完全冻结，只用确定性图像处理生成 training/validation 逐窗口表。语义臂按同视频边界强度和动作性不确定度排名分配 `K=256/384/512`；内容无关臂在相同窗口上置换完全相同的 K 多重集；固定 K384 伴随臂使用同一新数据合同。
- 真实计算要求按 K 分桶，并让 VideoMAE 接收确切长度 256、384 或 512；分桶后恢复视频/窗口顺序，在 NMS 前回到物理时间。一个逻辑批次只能有一次反向传播、optimizer、学习率和指数移动平均更新，损失按窗口数加权。
- 开发种子的主要检验是 semantic−content-independent Avg-mAP 及整视频配对区间，fixed384 只承担新合同下的安全比较。该裁决不声称降低每视频平均预算、优于 dense 或已经获得性能证据。
- 干净分支 `codex/duca-semantic-budget-matched-20260828@04c35a3b...` 已核验无改动。当前进入最小实现；尚无代码完成、独立审查、运行前核验、正式训练、mAP、区间或成本结果。

## 2026-08-29 — 动态预算候选实现冻结并进入真实 PRE_RUN

- 在独立干净分支完成冻结合同的最小实现，精确提交为 `36d75c146492a38eb8966c66ff6b2881938cf3c6`，已同步 GitHub。实现使用完整视频滑窗、按视频组成逻辑批次、冻结 Stage-1 侦察器生成一次性窗口表，并让 `K=256/384/512` 窗口分别以 16/24/32 个真实 16-frame clip 进入 VideoMAE；分桶后恢复窗口顺序，统一到长度 384 的检测轴，并在 NMS 前回映物理时间。
- 内容无关控制由冻结 nonce、split、视频名和窗口起点产生稳定置换；语义臂与控制臂逐视频共享相同 K 多重集和总预算。数据加载阶段不再预先宣称动态计算已经发生，只有 backbone 实际消费请求 K 后才记录该事实。
- 本机完成 Python 编译、Slurm 脚本语法和 Git 差异检查；本机 PyTorch 因 `c10.dll` 初始化错误不能收集模型测试，因此没有把本机状态写成测试通过。独立静态 Critic 对精确提交给出 PASS，真实数据身份、CUDA 输入长度、一次更新、侦察器不变和 checkpoint 恢复仍由 N16R4 PRE_RUN 决定。
- N16R4 独立干净部署位于 `/data/run01/sczc063/yuzibo/duca_semantic_budget_evaluator_20260829`。窗口表生成作业 `1260126` 已提交；只在其成功后运行的 semantic 单更新 PRE_RUN 作业为 `1260127`。该链只验证实验资格，不产生 mAP 或成本收益结论；三个 60 轮正式实验臂尚未提交。

## 2026-08-29 — 当前路线转为固定预算原生 tubelet 时序 coreset 归因实验

- 后续 Pro 裁决将当前实验收缩为固定 `K=384` 归因：768 帧形成 384 个原生两帧 tubelet，选择 192 个进入重型 VideoMAE 路径；比较确定性均匀选择与冻结动作性、边界强度、时序新颖性驱动的时序 coreset。低分辨率上下文回收、物理时间残差、384 点 tubelet 网格重建、检测器、训练和评估合同在两臂间保持一致。
- 独立干净分支 `codex/duca-native-tubelet-coreset-20260828` 的冻结提交为 `1957bdee1abdedcad2a509c7292bb05b07ba6548`。N16R4 正式环境的 8 项聚焦测试通过；独立只读审查核对真实 NCTHW 输入为选择后的 384 帧、固定 192 tubelet 预算、冻结侦察器、上下文回收和物理时间重建后给出通过结论。
- Slurm 运行前检查为 uniform `1260158` 与 coreset `1260159`。完整 60 轮训练 uniform `1260160` 与 coreset `1260161` 已设为仅在两个运行前检查共同成功时开始。提交、等待和通过局部测试均不是性能证据；当前无本轮 mAP、区间或成本结果。
- 固定预算只用于判断“选哪里”的贡献。若本轮不否定低成本侦察与稀疏重型计算假设，下一阶段必须转向真实动态预算：根据动作状态、边界密度、新颖性和冗余度改变实际 heavy clip 数，并使平均真实 VideoMAE 计算量与 fixed384 对照相同或更低。

## 2026-08-29 — 首次运行前检查发现并修复中间验证选模冲突

- uniform `1260158` 与 coreset `1260159` 均在任何数据迭代前退出。原始配置启用每 5 轮中间验证但声明只用于学习曲线，而正式训练入口把中间验证定义为最佳验证检查点选择；这与预注册 epoch-59 EMA 唯一选模冲突。依赖它们的正式作业从未开始并已撤销，因此没有训练或效能证据。
- focused correction `2e6b31c6b14c03015147cba093c45f42fc31ca12` 只关闭训练期中间验证，保留每 5 轮可恢复检查点、60 轮训练和最终 epoch-59 EMA 官方评估。N16R4 的 8 项聚焦测试与新的独立只读审查均通过；模型、选择器、数据、优化器、学习率、NMS 和评估器未改变。
- 修复后的运行前检查为 uniform `1260172`、coreset `1260173`；受共同成功条件保护的正式训练为 uniform `1260174`、coreset `1260175`。当前仍无 mAP、区间或成本结果。

## 2026-08-29 — 第二组运行前检查暴露旧 P0 绑定，最终轻量恢复合同冻结

- uniform `1260172` 与 coreset `1260173` 在训练开始前通过聚焦测试，但随后被 `DUCA_EXPECTED_COMMIT` 门拒绝。代码核验表明，继续满足该门还会要求历史 P0 专属的旧 variant、core gate、DDP pilot 和多组哈希材料；这些材料不属于当前原生 tubelet 归因实验。依赖它们的正式作业从未启动并已撤销，因此仍无模型效能证据。
- 最终 correction `b33391126eac05e3353d322b973dda91741f0732` 关闭遗留 `formal_successful_update_contract`，不构造新的证明框架。训练仍保留有限 AMP/非有限损失重试、成功更新计数、每 5 轮检查点和最终 epoch-59 EMA；新增的轻量恢复开关使 checkpoint 保存并恢复模型、EMA、优化器、调度器、AMP、epoch、成功更新数和全局随机状态。
- PRE_RUN 被扩展为真实执行两次更新、保存 epoch 0、从该 checkpoint 恢复并再执行两次更新，然后核对 epoch 1、4 次成功更新与全部恢复状态键。N16R4 的 20 项相关测试和最终独立只读审查均通过。
- 最终运行前检查为 uniform `1260182`、coreset `1260183`；受共同成功条件保护的正式训练为 uniform `1260184`、coreset `1260185`。这是本实现周期最后一次执行链；若仍出现等价确定性实现缺陷，将关闭实现周期并返回 Pro。

## 2026-08-29 — 最终运行前检查通过，固定预算归因训练开始

- uniform `1260182` 与 coreset `1260183` 均完成运行前检查。两臂实际执行了保存—恢复—继续更新，并通过冻结的状态与更新计数核验；这关闭了本实现周期的工程准入问题，但不构成性能或成本证据。
- 受共同成功条件保护的完整 60 轮训练 uniform `1260184` 与 coreset `1260185` 已自动开始，代码身份保持 `b33391126eac05e3353d322b973dda91741f0732`。在终态 epoch-59 指数移动平均模型完成官方评估前，不读取中间 mAP、不选择中间检查点，也不推断时序 coreset 是否有效。
- 固定 `K=384` 仍只承担“选哪里”与拼接系统生命力的归因检验。若真实 THUMOS14 结果不否定低成本侦察与稀疏重型计算假设，下一阶段必须转向真正改变 heavy clip 执行数量、且平均实际 VideoMAE 计算量不高于固定预算对照的动态预算。

## 2026-08-29 — 固定预算原生 tubelet 归因训练终态

- uniform `1260184` 与 coreset `1260185` 都完成 60 轮训练，写出各自 epoch-59 checkpoint，并以指数移动平均模型完成 211 个 validation 视频、422,000 条预测的官方评估。日志点估计分别为 Avg-mAP `64.13%` 与 `62.81%`，mAP@0.7 `42.45%` 与 `40.56%`；coreset 相对 uniform 为 `-1.32/-1.89` 个百分点。
- 两个作业均在指标计算之后因同一配置缺陷退出：`post_processing.save_dict=False` 没有保存预测，而结构化指标入口要求保存预测。冻结合同要求的 `metrics_epoch59_ema.json` 未生成，因此没有配对区间或实测成本；当前点估计只能作为对任务状态驱动 coreset 不利的诊断性结果。
- 该失败不是训练中断，也不等于没有模型证据；它是终态证据封存失败。当前实现周期按既有终止规则关闭，不自动修改、重训或重评。下一步把负向点估计、共同封存缺陷和缺失证据中立返回 Pro，由 Pro 独立决定是否进行纯评估封存、修改选择机制，或停止这一候选。

## 2026-08-29 — Pro 终止细粒度 coreset，并冻结窗口级动态计算任务

- fresh exact-DUCA Project Pro 在对话 `6a92b5a5-fb8c-83ea-9cb6-0f13520b1050` 给出 `PIVOT`。当前 H65 任务状态细粒度 tubelet coreset 作为候选终止；不得调整分数、端点、最大空洞、打包或重新训练，也不单独补跑旧 uniform-versus-coreset 封存。
- 裁决只否定该完整实现包优于确定性均匀选择的操作性预测，不是否定所有低成本侦察、稀疏重型计算、均匀可变预算或动态预算。选择分数不匹配、非连续 tubelet 的人工邻接、边界局部密度不足及重建交互仍只是按证据排序的假设，不是已证实因果。
- 唯一后继任务使用冻结 H65 Stage-1 epoch-29 指数移动平均侦察器，对同一视频的窗口按平均动作性、边界重要性第 90 百分位和新颖性第 90 百分位形成需求排序。低需求一半实际执行 16 个 VideoMAE clip，高需求一半执行 24 个，奇数视频的中位窗口执行 20 个；各预算内部只做确定性均匀原生 tubelet 选择。
- 新臂的平均实际重型工作量严格为每窗口 20 个 clip，相比 fixed24 控制少 16.67%。Padding 到 24 再记录名义预算属于实现失败。只允许一个 seed 3407、60 轮动态训练臂；对照复用 Job `1260184` 的 epoch-59 EMA checkpoint，不重训旧对照。
- 下一流程为最小 Builder 实现、一次独立 Critic 审查和 Evaluator 运行前检查；通过后直接执行完整 THUMOS14 动态臂并生成官方 211 视频预测、真实 clip 计数、匹配成本、短动作/边界诊断与整视频配对区间。绝对截止为 `2026-09-04T23:59:00+08:00`。

## 2026-08-29 — 窗口级动态原生 tubelet 候选实现完成并通过静态审查

- 从固定归因基座 `b3339112...` 建立独立干净分支 `codex/duca-dynamic-native-tubelet-budget-20260829`。最小实现冻结于 `d127c2b2ceea7ff8a6932aa4a1925e1ff86cf610`：冻结 Stage-1 侦察器生成逐视频窗口需求表，各预算内确定性均匀选择 128/160/192 个原生 tubelet，并让 VideoMAE 按窗口真实执行 16/20/24 个 clip；重型骨干之前不补齐到 24，分组后恢复样本顺序并沿用骨干后的物理时间残差与 384 点重建。
- 一次独立审查错误地要求把不规则物理坐标直接注入 VideoMAE。对照 Pro 冻结的“沿用现有 packing”以及固定对照基座后，新的独立聚焦审查确认该要求会改变匹配对照的重型表示，因而不适用于本实验。物理时间仍由既有骨干后残差和重建维护。
- 聚焦审查发现短窗口可能使声明预算被静默缩小。候选现已在预算表生成和运行时选择两处明确拒绝有效 tubelet 少于分配预算的窗口，并增加可区分测试；修复后的精确提交由新的独立 Critic 给出静态通过。
- 修改文件的 Python 编译、Slurm 语法、Git 差异检查和不依赖本机 PyTorch 的启动器测试通过。本机张量测试仍受 Windows `c10.dll` 初始化错误阻断；这不是模型证据。当前没有 PRE_RUN、Slurm 正式训练、动态预算 mAP、配对区间或成本结果。下一步仅由 Evaluator 在 N16R4 核验真实 split 覆盖、短窗口可行性、三种实际重型输入长度和 checkpoint 恢复；通过后才启动唯一动态训练臂。

## 2026-08-29 — 窗口级动态预算运行前检查已提交

- 精确提交 `d127c2b2...` 已同步到 GitHub 分支 `codex/duca-dynamic-native-tubelet-budget-20260829`，并部署到 N16R4 独立干净目录 `/data/run01/sczc063/yuzibo/duca_dynamic_native_tubelet_d127c2b2_20260829`。远端 HEAD 与本地冻结提交一致。
- Evaluator 提交了唯一运行前检查 Job `1261074`，结果根为 `/data/run01/sczc063/yuzibo/duca_dynamic_native_tubelet_prerun_d127c2b2_20260829`。它负责生成 train/validation 窗口预算表，执行 N16R4 聚焦测试、真实短程训练、checkpoint 保存—恢复—继续更新核验，并验证动态重型执行入口。
- 提交回执显示作业为 `PENDING (Priority)`。该调度状态不是实现通过、训练结果或性能证据；不读取中间输出，也不重复提交。只有作业成功终止后，才允许提交唯一 60 轮动态预算训练臂。

## 2026-08-29 — Coverage-v1 实现、独立审查与远端部署完成

- 最新路线把当前实验收窄为固定 `K=384` 的单变量分配归因：冻结 H65 Stage-1 语义优先级与全部后端，只把逐帧 Top-K 替换为 96 个物理时间锚点上的确定性设施位置覆盖。边界梯度、特征相似度、特征合并、注意力偏置和动态 K 均不进入本轮。
- 从 clean H65 `04c35a3b...` 建立分支 `feature/duca-coverage-only-v1-20260829`。最终提交为 `a8a0514b00c3528fcf201e6a042b6056429346e1`；它包含张量化 `TemporalCoverageSelector`、matched H65/Coverage 配置、真实训练样本无标签重放门、聚焦测试和 N16R4 启动器。代码已同步 GitHub。
- 远端聚焦检查共 27 项通过，严格加载 Stage-1 `state_dict_ema` 成功。主实现、smoke 日志目录修正、Slurm account/QOS 绑定修正及 `/etc/profile` 初始化均经过独立只读复核；后面三项只影响运行入口，不改变模型、数据、训练或评估语义。
- 最终干净快照部署在 `/data/run01/sczc063/yuzibo/duca_coverage_v1_a8a0514b_20260829`，HEAD 与本地一致，启动器 Bash 语法通过。
- 第一次 PRE_RUN 提交在作业创建前被 Slurm `AssocMaxSubmitJobLimit` 拒绝，没有作业编号或输出目录。当前只有实现与审查证据，没有训练、mAP、配对区间或成本结果。项目未取消其他共享作业，也未修改实验以绕过额度；额度可用后应原样提交同一 PRE_RUN。

## 2026-08-30 — Coverage-v1 中间验证用途修正并原样重提

- PRE_RUN Job `1261660` 最终获得调度，但在训练开始前的静态合同检查终止。Coverage-v1 继承的 Stage-2 规则明确规定中间验证每 5 轮只记录学习曲线，正式模型固定为 epoch-59 `state_dict_ema`；通用训练合同当时只接受“中间验证选择最佳 checkpoint”，因此错误报告用途未显式声明。该终态是配置验证失败，不是模型或数据结果。
- 最小修正冻结于 `feature/duca-coverage-only-v1-20260829@048143124e2a36a76575200ae17d6f42ec79ea3a`：Coverage 配置直接声明 `learning_curve_only` 与不选择中间 checkpoint；通用合同接受该一致组合，同时保留既有 official60 的 curve-best 模式并拒绝角色与选择布尔值不一致。模型、选择器、数据、60 轮/6000 更新、终态 EMA、NMS 和官方评估均未改变。
- 本地 Python 编译、非 Torch 合同测试和 9 项既有 official60 测试通过；本机 Coverage 张量测试仍受 Windows PyTorch `c10.dll` 初始化失败影响，将由相同 N16R4 PRE_RUN 执行。独立 Critic 对精确 clean commit 返回通过。GitHub 已同步，远端干净部署为 `/data/run01/sczc063/yuzibo/duca_coverage_v1_04814312_20260830`。
- 修正后的首次 `sbatch` 仍在作业创建前被 `AssocMaxSubmitJobLimit` 拒绝。按用户要求，本地守护已停止；N16R4 上唯一的自终止进程 PID `427132` 只对同一 commit、同一 launcher、同一 PRE_RUN 参数每 60 秒重试，日志为 `/data/run01/sczc063/yuzibo/duca_coverage_v1_submit_04814312_20260830.log`。容量释放后它提交一次、写出 receipt 并退出；其他错误会直接停止。当前没有新的 PRE_RUN 作业号，也没有 Coverage 训练、mAP、配对区间或成本结果。

## 2026-08-30 — Coverage-v1 真实训练数据干预门未通过

- N16R4 自终止提交进程成功创建修正后 PRE_RUN Job `1261679`，绑定 clean commit `048143124e2a36a76575200ae17d6f42ec79ea3a` 与结果根 `/data/run01/sczc063/yuzibo/duca_coverage_v1_prerun_04814312_20260830`，随后退出。Job 运行 6 分 24 秒并以退出码 `2:0` 终止。
- 27 项聚焦测试通过；Stage-1 epoch-29 `state_dict_ema` 正确加载。200 个无标签 training 样本重放中，有效且唯一选择比例为 `1.0`，两臂优先级张量相同，且选择不使用标签、教师或预测缓存。集合变化中位数为 `0.4805`，未达 `0.80`；锚点覆盖中位数由 `0.9365` 增至 `0.9676`，相对增益 `0.0332`，未达 `0.10`；最大未选空洞第 95 百分位由 `2` 增至 `8`，相对下降统计为 `-3.0`，方向与 `0.20` 门槛相反；保留优先级比率 `1.2097` 通过 `0.90` 下限。
- 作业在两臂 smoke 和训练前按预注册规则停止。该结果说明当前实现没有产生预期的覆盖/最大空洞几何干预，不是 mAP、显著性或成本结果。禁止绕过门槛、调低阈值或直接启动 60 轮训练。
- 同轮代码核验发现，Wiki 冻结设计曾把 control 概括为 H65 Top-K，但实际 control 是 `budget_calibrated_sampling_rate` 系统采样；代码冻结的是侦察器和优先级适配器，后端结构/训练协议保持匹配但训练参数未冻结；设施位置锚点使用归一化候选帧索引而非秒级时间。上述差异会改变因果归因，必须返回 Pro 后再决定基线定义、失败机制和后续任务。

## 2026-08-30 — Pro 在 Coverage 中间机制失败后转向 DUCA-Marginal-v1

- Pro 摄取 PRE_RUN `1261679` 与真实 H65 系统采样身份后给出 `PIVOT`。停止 Coverage-v1 的 matched
  60-epoch 训练；当前 96-anchor facility-location 只被判定为未形成预注册的覆盖干预，不能扩展为
  coverage 家族或动态预算无效。
- 唯一当前任务是冻结 H65 的 K256/K384/K512 反事实边际预算实验：先用训练侧 detached detector loss
  测量 equal-budget reallocation 的 oracle headroom，再训练轻量 utility head 预测升/降预算收益；每个
  视频总 observation 数严格等于 `384 × 窗口数`，三种 K 必须真实进入不同长度的 VideoMAE 执行。
- clean base 固定为 `04c35a3b76897e6c1569eeede41ed3aecaf7f854`；H65 terminal checkpoint 为
  `/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/gpu1_id0/checkpoint/epoch_59.pth`，
  SHA-256 `dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c`，正式使用 `state_dict_ema`。
- 当前只有科学合同和既有 artifact 身份，没有 DUCA-Marginal 实现、运行前结果、mAP、配对区间或成本证据。

## 2026-08-31 — DUCA-Marginal 短窗口合同实现、独立审查与 PRE_RUN 提交

- Pro 对公开候选 `e45dda787a6880da4cbde0b6436ffd2a2b9df218` 返回 `REVISE`，冻结
  `c_i(K)=min(V_i,K)`、相同实际成本折叠到 K384、历史 K384 固定 384 slots、非基线按 16-observation
  packet 真实执行、每视频 `sum_i min(V_i,384)` 精确预算和 eligible-only utility targets。完整回复保存在
  `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-short-window-contract-v002/visible-report.md`。
- Codex 在同一最小代码面完成修订；最终 clean commit
  `be5bb8033c0b11c628394d268c1923ab398c04ed` 已推送至
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-budget-v1-20260830`。
  独立 Critic 对 accounting、折叠别名、真实执行、masked targets、K384 parity 和 PRE_RUN 证据绑定给出通过。
- 同一提交已部署至 N16R4 clean snapshot
  `/data/run01/sczc063/yuzibo/duca_marginal_be5bb803_20260831`，Evaluator PRE_RUN Job `1262073` 已提交，
  输出根为 `/data/run01/sczc063/yuzibo/duca_marginal_prerun_be5bb803_20260831`。当前只有提交事实，没有
  PRE_RUN 终态、反事实结果、headroom、可预测性、mAP 或成本证据。

## 2026-08-31 — PRE_RUN 启动 shell 最小恢复

- Job `1262073` 在 `00:00:00` 内以 `127:0` 退出；stderr 的首个且唯一错误为
  `/var/spool/slurmd/job1262073/slurm_script: 4: source: not found`。`sbatch --wrap` 生成 `/bin/sh` 脚本，
  因此没有进入 Python runner、加载 checkpoint、读取数据或运行任何模型检查，也没有产生
  `failure_pre-run.json`。这是一项确定性的启动环境错误，不是模型或科学结果。
- 未修改 Git commit、配置、数据、checkpoint、参数、输出根或预登记门槛，仅将 Slurm 输入改为带
  `#!/usr/bin/env bash` 的批处理脚本，并先执行 `source /etc/profile`。同一 PRE_RUN 已唯一重提为 Job
  `1262075`；在它写出终态 `PRE_RUN_PASS` 前，后续四阶段 probe 仍未获准。

## 2026-08-31 — PRE_RUN 聚焦测试合同修复

- Job `1262075` 使用 Bash 正常进入 Linux 测试，运行 `24.57` 秒后得到 `32 passed, 1 failed`。唯一失败
  位于短窗口分配测试：测试显式传入 `max_changed_fraction=1.0`，允许四个窗口全部改变，于是总效用更高
  的 `[256,512,512,256]` 合法胜出；测试却预期只改变两个窗口的 `[256,512,384,384]`。
- Pro 冻结合同明确要求改变窗口数不超过 `floor(0.5N)`，并按“总效用最大、改变窗口更少、确定性字典序”
  排序；配置和真实 runner 也都使用 `0.5`。因此实现没有错误，最小修复只把该测试参数从 `1.0` 恢复为
  `0.5`。公开提交为
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f87555f7da362fe1a20d4ca08f7a68c975ed8280`。
- 新提交已部署到 clean snapshot `/data/run01/sczc063/yuzibo/duca_marginal_f87555f7_20260831`；唯一后继
  PRE_RUN 为 Job `1262076`，输出根 `/data/run01/sczc063/yuzibo/duca_marginal_prerun_f87555f7_20260831`。
  分配器、模型、checkpoint、数据、阈值和科学问题均未改变。
- 为避免 Codex 主动轮询，已创建单一 Slurm `afterany` 后继 Job `1262077`。它绑定同一 clean snapshot、
  checkpoint、数据、参数与输出根；前置 Job `1262076` 终态后先验证 `pre_run_receipt.json` 的状态必须为
  `PRE_RUN_PASS`，然后才运行既有 runner 的 `--stage all`，即 `select-k384`、`counterfactual-k256`、
  `counterfactual-k512` 和 `summarize`。若 PRE_RUN 未通过，它不执行 probe。该依赖提交不是科学结果。

## 2026-08-31 — DUCA-Marginal PRE_RUN 通过

- Job `1262076` 于 `2026-08-31T04:10:25` 完成，状态 `COMPLETED 0:0`，运行 11 分 16 秒；同一输出根写出
  `pre_run_receipt.json`，状态为 `PRE_RUN_PASS`，绑定 clean commit
  `f87555f7da362fe1a20d4ca08f7a68c975ed8280` 与冻结 epoch-59 `state_dict_ema`。
- 准入覆盖 200 个训练侧视频、720 个窗口、160/40 utility fit/holdout 划分和所有短窗口；47 个 collapsed
  arm 别名 K384。完整 K384 384-slot tensor 逐窗口一致，每视频实际 K384 成本与冻结目标完全相等。
  冻结 forward 覆盖 historical short K384、explicit full K384、K256-256 slots 和 K512 的
  400/448/464/480/496/512 slots。没有 detector/Scout 梯度、utility-head fitting、detector training、
  official evaluator 或 official-test 访问。
- 该结果只证明实现与运行准入，不是动态预算 headroom、可预测性、mAP、显著性或真实成本证据。唯一
  probe Job `1262077` 负责随后四阶段，不得重复提交。

## 2026-08-31 — DUCA-Marginal producer 完成但首次汇总失败

- Job `1262077` 在同一 `f87555f7...` clean snapshot 下完成 `select-k384`、`counterfactual-k256` 和
  `counterfactual-k512`，封存了三个 JSONL 产物及其 receipts。随后 `summarize` 在第一次训练侧 holdout
  evaluator 调用前退出，首个真实错误为 `JSONDecodeError: Expecting value: line 1 column 1`。
- 根因是 `create_duca_frontend_split.py` 写出的 `frontend_holdout_block_list.txt` 为换行文本，而 OpenTAD
  mAP evaluator 把 `blocked_videos` 路径按 JSON 读取。该作业没有生成 `probe_result.json`，所以只有三段
  producer 产物存在性，没有 headroom、predictability 或机制门结论。
- 最小修复提交 `f67d96fdf68a295eaa7f678f3dfc125530828889` 只把该文本列表确定性转换为 evaluator 接受的相邻 JSON
  文件，并加入回归测试。公开链接为
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f67d96fdf68a295eaa7f678f3dfc125530828889`。
  独立 Critic 判定 summary-only fix 不改变模型、选择、预测、NMS、指标或门槛；N16R4 clean snapshot
  `/data/run01/sczc063/yuzibo/duca_marginal_f67d96fd_20260831` 的 11 项 focused tests 通过。

## 2026-08-31 — DUCA-Marginal summary 恢复完成并落入 Pro 灰区

- 唯一恢复 Job `1262098` 于 `2026-08-31T04:58:08+08:00` 启动，`05:09:34+08:00` 以
  `COMPLETED 0:0` 结束。它只在 `f67d96fd...` 下运行 PRE_RUN 和 `summarize`；没有重跑三个 producer、
  训练 detector/Scout/utility head 或访问 official test。输出根为
  `/data/run01/sczc063/yuzibo/duca_marginal_summary_f67d96fd_20260831`。
- 当前 PRE_RUN 再次覆盖 200 个训练侧视频、720 个窗口、160/40 视频划分、短窗口、K384 完整张量身份、
  真实 packetized 执行、冻结梯度与零训练/零 official test。producer receipts 仍绑定 `f87555f7...`；
  summary 与当前 PRE_RUN 绑定 `f67d96fd...`。两者 config SHA、checkpoint SHA、annotation、类别映射和
  VideoMAE 预训练哈希一致；该双提交来源被保留，不表述成单提交重算。
- 40 个 utility holdout 视频、124 个窗口的 Fixed-H65-384 为 Avg-mAP `88.131197%`、mAP@0.7
  `76.270583%`；真实效用 oracle 等预算重分配为 `88.856786%`、`76.999587%`，增益
  `+0.725589/+0.729004` 个百分点。oracle 分配为 K384 102 个、K256 11 个、K512 11 个窗口，实际
  observation 总成本 `47110`，预算误差为零。
- 强 headroom 门是 `+0.8/+1.0`，无 headroom 边界是 `<+0.3/<+0.5`；当前介于两者之间。终态按冻结
  runner 写为 gray zone 并返回 Pro。utility head、predictability、learned allocation、K320、official
  test、统计区间和端到端成本均未运行；Codex 不自行放宽门槛或选择后继路线。

## 2026-08-31 — Pro 修订灰区并启动唯一 cap-release 只读诊断

- Pro 终态为 `REVISE`：接受 Job `1262098` 的训练侧机制诊断和显式双提交来源，不要求重跑 producer；当前
  结果仍不准入为论文主结果、official validation/test 结论或显著性结论。
- 唯一冻结任务只把 oracle 的 `max_changed_fraction` 从 `0.5` 改为 `1.0`，其余三档产物、真实 utility、
  每视频实际 observation 总预算、tie-break、NMS 与 evaluator 均不变。点门未同时达到 `+0.8/+1.0`
  时停止当前机制；同时达到后才运行 10,000 次整视频配对 bootstrap，本任务仍禁止 utility-head 训练。
- Builder 从 `f67d96fd...` 产生单一 clean 提交
  `d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`，公开链接为
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/d2fad7c0dfc4a5efe98b10b9eee4723c6805699f`。
  差异仅为 runner 与聚焦测试；N16R4 `14 passed`，独立 Critic 为 PASS。
- 唯一 Evaluator Job `1262117` 已提交。集群只有强制申请 GPU 的 Slurm 分区，因此作业申请最小 1 卡作为
  调度占位，但 runner 固定 `--device cpu`，不执行 detector/Scout forward 或模型训练。终态前不产生新的
  性能或统计解释。

## 2026-08-31 — cap-release 只读诊断终态触发 Marginal-v1 停止条件

- 唯一 Evaluator Job `1262117` 于 `2026-08-31T05:53:33+08:00` 启动，`05:54:25+08:00` 以
  `COMPLETED 0:0` 结束。它只在 CPU 上读取原密封产物；未执行 detector/Scout forward、训练、utility-head
  拟合或 official test。原 probe 与三个 producer 的 SHA-256 均保持不变。
- fixed K384 与 50% capped oracle 的复现误差均为 `0.0` 个百分点。解除上限后的 K256/K384/K512 分配为
  `17/90/17`，改变 11 个视频和 34 个窗口，实际 observation 总成本仍为 `47110`，误差为零。
- released oracle 的 Avg-mAP/mAP@0.7 为 `88.558507%/76.720863%`，相对 fixed K384 为
  `+0.427310/+0.450280` 个百分点，相对 capped oracle 为 `-0.298279/-0.278724` 个百分点。两项强门均
  未通过，冻结 runner 写出 `CAP_RELEASE_POINT_GATE_FAILED_STOP_CURRENT_MECHANISM`，没有运行 bootstrap。
- 本终态停止当前 Marginal-v1 机制，但不外推为所有动态预算方法无效。结果文件保存在
  `.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-d2fad7c0-job1262117/oracle_cap_release_result.json`；下一项
  科学任务必须连同最新 GitHub 实现 `d2fad7c0...` 返回 Pro 独立裁决。

## 2026-08-31 — Pro PIVOT 到唯一联合 mAP 邻域证伪

- Fresh exact-DUCA 对话 `6a94abb7-bd48-83e9-9516-c650c982dd57` 完整结束，nonce、Project、Pro 模型和最新
  GitHub `feature/duca-marginal-cap-release-falsifier-v1-20260831@d2fad7c0...` 及三个关键文件永久链接均通过
  终态绑定。完整报告保存于
  `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-terminal-v001/visible-report.md`。
- Pro 裁决为 `PIVOT`：停止当前三档预算的独立窗口加性损失分配，但不停止 DUCA 动态计算。主要证据是
  cap release 提高内部加性目标却降低最终 mAP，说明窗口损失不能充分表示重叠预测、NMS、排序和 AP 的
  视频级联合效用；粗档位和等成本组合约束是可能放大器，固有 headroom 小仍未排除。
- 唯一冻结任务只在 runner/test 内枚举 capped→released 差分邻域。由数据导出 12 个差分窗口、所有每视频
  等成本平衡子集和当前应有的 96 个联合状态；复现 fixed/capped/released，保持逐视频预算与全局 `47110`，
  零 forward、零训练、零 official test、零 bootstrap。
- 至少一个状态同时达到 `+0.8 pp` Avg 与 `+1.0 pp` @0.7 才获得继续研究资格，且仍须返回 Pro；否则停止
  用视频级联合效用修复本次差分。Builder、独立 Critic、唯一 CPU Evaluator 依次执行，不扩展工程面。

## 2026-08-31 — 联合 mAP 邻域 96-state 诊断完成并触发停止条件

- Builder 在公开分支 `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831` 形成 clean commit
  `46812facc8773d9b4a9c21833cbe397c8aaa5a2d`：
  `https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/46812facc8773d9b4a9c21833cbe397c8aaa5a2d`。
  只修改 runner 与聚焦测试，allocator 相对父提交逐字不变。实现由真实成本与 sealed allocation 推导
  5 个差分视频、12 个窗口、6 个净转移组、8 个最小合法转移和 96 个唯一状态；没有为多解视频硬编码配对。
- N16R4 上 16 项聚焦测试、23 项既有回归测试通过，独立 Critic 返回 `TERMINATOR_STATIC_PASS`。唯一
  Evaluator Job `1262121` 以 `COMPLETED 0:0` 结束，完成 96 次 CPU evaluator 调用。集群分区强制申请
  1 张 GPU，但作业清空 CUDA 可见性并固定 CPU；没有模型前向、训练、梯度、utility-head fitting、
  official test 或 bootstrap。stderr 只有无关的 `requests` 依赖版本警告。
- fixed/capped/released 复现误差均为 `0.0 pp`，全部状态保持逐视频预算和全局成本 `47110`。没有状态同时
  达到 `+0.8/+1.0`：联合门最优 `state_014` 为 `+0.553972/+0.933234`，Avg 最优 `state_020` 为
  `+0.732990/+0.479291`，@0.7 最优 `state_001` 为 `+0.548669/+0.933539`。
- 8 个最小合法转移均未同时改善 Avg 与 @0.7，冻结分类为以单项误排序为主，未发现窗口交互反转见证。
  按预注册规则，本次差分邻域的联合效用修复停止，且不补做选择后 bootstrap。原始终态位于
  `.cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-neighborhood-46812fac-job1262121/oracle_cap_release_neighborhood_result.json`，
  SHA-256 为 `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`。下一项路线只能由 Pro 在摄取最新
  GitHub 永久链接和完整证据后独立裁决。

## 2026-08-31 — Pro STOP 并完成 Marginal-v1 终态归档

- Fresh exact-DUCA 对话 `6a94bbaf-b2f8-83ea-81bc-5c0b6b23bdb5` 完整结束；Project
  `g-p-6a91061f789881918ccd8357ca3d6c92`、nonce、浏览器 `Pro` 选择和最新公开 GitHub
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`
  均通过终态绑定。完整报告保存于
  `.cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-neighborhood-terminal-compact-v002/visible-report.md`。
- 第一次传输因内联材料过大在提交前停止，元数据明确 `promptSubmitted=false`；只关闭了该次新建的空白专用
  标签页，未关闭或重启 profile61、端口或用户现有页面。精简后的同一科学请求只产生上述一次远端提交。
- Pro 裁决为 `STOP`：窗口级加性反事实 detector loss 不是该邻域中视频级联合检测效用的充分排序统计量；
  八个最小等成本转移均未同时改善 Avg 与 @0.7，因此不存在“各自联合有益、只在组合后被交互破坏”的转移
  集来修复本次差分。内部分类不直接进入论文，`interaction_witness_count=0` 也不外推为所有交互不存在。
- 停止范围包括现有 H65 priority sequence + K256/K384/K512 + 逐窗口反事实损失 + 每视频等成本加性分配，
  以及 cap、配对、tie-break 和同一 96-state 邻域内的修复。没有后继代码、审查、PRE_RUN、Evaluator、
  bootstrap、utility-head 或 official-test 任务。最终分支只作为负证据读取。
- JSON SHA-256 已在 Pro 终态后独立重算并确认：
  `a80208921cbb907e522f56dae885b9786395ccabe14026e5f551e92e46e61a4b`。本停止不否定三档预算本身、H65
  priority sequence 或任务感知动态计算的一般问题；未来后继必须由 Pro 以新机制假设独立启动。

## 2026-08-31 — 项目级 Pro PIVOT 到整视频一致预算的跨视频转移

- Fresh exact DUCA Project turn 使用 nonce `DUCA-PROJECT-LEVEL-AFTER-MARGINAL-STOP-v001-20260831`、浏览器
  `Pro` 和最新公开
  `feature/duca-marginal-cap-release-neighborhood-falsifier-v1-20260831@46812facc8773d9b4a9c21833cbe397c8aaa5a2d`。
  对话为 `https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a94c0ae-5388-83e9-afd3-6b8f1e596e1e`。
- Pro 裁决 `PIVOT`：不恢复 Marginal-v1，也不停止整个 DUCA；只执行一次整视频一致预算、跨视频 donor–recipient
  转移的开发集 privileged oracle。一个 changed video 的所有窗口必须使用同一 requested tier。
- 只使用 40 个 controller holdout 视频、124 个窗口、密封三档预测和真实 observation 成本。候选在读取标签
  或指标前完整生成，成本不得超过 `47110`；正式运行只调用相同 CPU evaluator，不执行 forward、训练、
  bootstrap 或 official test。
- 联合门仍为 `+0.8/+1.0` 个百分点。若通过候选为零，按 Pro 冻结规则在当前数据、H65 priority sequence、
  三档动作空间和资源边界内项目级停止 DUCA 方法创新，不再扩大搜索。
- 完整报告：`.cvpr-pro-lab/pro-reviews/runs/duca-project-level-after-marginal-stop-v001/visible-report.md`；Prompt
  SHA-256 `2d9a87a2c425d920ca1f9c8cb3dc4469ce7a4aa5f4a76b3bf2f3e4397a9527d1`；报告 SHA-256
  `f2fcef731e3e6545ae06759bc6eae9f7f900f9807cf97c2583aa4c4d65f350ce`。

## 2026-08-31 — 整视频一致预算 falsifier 完成实现并通过独立审查

- 新分支已推送：`feature/duca-whole-video-consistent-budget-falsifier-v1-20260831`；精确提交
  `c27d77aafd4aa514def033b03f2dfc2d6c24771e`，远端与本地一致且 worktree 干净。
- 提交只新增 `tools/bata/run_duca_whole_video_consistent_budget_falsifier.py` 与
  `tests/test_duca_whole_video_consistent_budget_falsifier.py`。候选在读取终态 mAP、annotation、GT 或评估器前
  完整生成；视频内 requested tier 一致；成本只读密封 `budget_accounting.actual_cost`；不调用旧 allocator。
- 本地 4 项新测试通过；旧 marginal 测试在 Windows 按既有保护跳过。全新独立 Critic 核对 exact clean
  commit、两文件 diff、候选顺序、成本、密封预测与相同 Soft-NMS/评估器路径后返回 `PASS`。
- 当前只有实现与审查证据，没有 PRE_RUN、候选 mAP 或项目级终态。下一步是部署 exact commit 到 N16R4，
  运行冻结 PRE_RUN；仅通过后提交一次 CPU evaluator。

## 2026-08-31 — 整视频 falsifier PRE_RUN 已提交

- GitHub 精确提交 `c27d77aafd4aa514def033b03f2dfc2d6c24771e` 已部署到 N16R4 干净快照
  `/data/run01/sczc063/yuzibo/duca_whole_video_c27d77aa_20260831`；远端 `HEAD` 与 GitHub 一致且 worktree 干净。
- 输入目录中的 `probe_result.json`、cap-release、neighborhood 及三档密封 artifact SHA 已在提交前逐项复核，均与
  冻结值一致。PRE_RUN Slurm Job 为 `1262147`，提交脚本 SHA-256 为
  `ba0540563af8b4e876945befac109954791bbb8b51a73eb1c76646fd882f62e9`。
- PRE_RUN 将执行 Linux 两文件聚焦测试、完整候选清单生成、split/producer/终态 JSON 身份核验以及
  fixed/capped/released 锚点复现。当前仅有作业创建事实；不读取中间状态，不推断通过，不提交正式 evaluator。

## 2026-08-31 — 整视频 falsifier 修正顺序重放、通过 PRE_RUN 并启动正式评估

- 原 PRE_RUN Job `1262147` 在候选性能计算前停止。原因是 runner 对密封 proposal 行做了字典序排序，改变了
  分数并列时不变 Soft-NMS 所接收的输入顺序；这是确定性的证据重放错误，不是机制性能结果。
- 已推送精确提交 `33e4ed137c33eef07f0452b44506a6993bdf7535`。它只恢复 producer 的密封行顺序并增加
  对应回归测试，不修改候选空间、成本口径、Soft-NMS、评估器或旧三档 allocator。28 项聚焦测试通过，
  全新独立 Critic 对 exact clean commit 返回 PASS。
- 修正后的 PRE_RUN Job `1262161` 通过：40 个视频、124 个窗口、固定成本 `47110`、1560 个有序视频对、
  704 个合法候选、1330 个实际干预候选，fixed/capped/released 三个锚点的复现误差均为 `0.0 pp`。
  PRE_RUN receipt SHA-256 为 `734b178bfb7bdaa05879edfeb8e129263c9e2c4cf80867415eec6d41df3c12a3`；
  candidate manifest SHA-256 为 `c4a02c47be1ab7e73dc81c18b32635d3347ece2f0d26b0d96de3ec4af053f69a`。
- 正式 Evaluator Job `1262162` 已从 clean snapshot
  `/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831` 提交；当前未读取或宣称终态性能。
- 后续返回 Pro 的材料必须同时给出最新已推送仓库、实际远端分支、精确提交、runner、聚焦测试与未改动
  allocator 的永久 GitHub 链接，禁止引用本地、未推送或过时实现。

## 2026-08-31 — 整视频正式评估因节点下线中断并作同任务恢复

- Job `1262162` 在完成 `500/704` 个候选后由 Slurm 以 `NODE_FAIL` 终止；`scontrol` 明确记录
  `Reason=NodeDown`、`FailedNode=g0022`。stderr 只有环境加载信息，运行器没有写出 `failure_evaluate.json`，
  终态 `whole_video_consistent_budget_result.json` 也不存在，因此没有候选通过数或科学裁决。
- 该故障没有改变代码、候选、数据、成本、Soft-NMS、评估器或门槛。完全相同的同任务恢复 Job `1262190`
  已从 clean snapshot `/data/run01/sczc063/yuzibo/duca_whole_video_33e4ed13_20260831` 与原提交脚本启动，继续以
  `33e4ed137c33eef07f0452b44506a6993bdf7535` 为唯一实现身份。它是唯一恢复作业，不再创建第三份作业。

## 2026-08-31 — 整视频一致预算 falsifier 完成并触发冻结停止条件

- 唯一基础设施恢复 Job `1262190` 于 `2026-08-31T11:59:29+08:00` 完成，Slurm 为 `COMPLETED 0:0`，
  在未修改的 `33e4ed137c33eef07f0452b44506a6993bdf7535` 干净快照上评估完 `704/704` 个合法候选。
  终态结果位于
  `/data/run01/sczc063/yuzibo/duca_whole_video_result_33e4ed13_20260831/whole_video_consistent_budget_result.json`，
  SHA-256 为 `40686fa73114eedfa14b3d34a01717aacb0b93f629f5a1e7f2ee27de300ad19c`；没有 failure receipt。
- 固定 K384 在 40 个训练侧 controller holdout 视频上为 `88.1312%` Avg-mAP、`76.2706%` mAP@0.7，实际
  observation 成本 `47110`；全部指标复现误差均为 `0.0` 个百分点。固定臂与 704 个候选共调用同一评估器
  705 次，没有 detector/Scout forward、训练、梯度、bootstrap、official validation 或 official test。
- Avg-mAP 最优候选为 `+0.6942/-0.0436` 个百分点（Avg-mAP/mAP@0.7，成本 `46982`）；mAP@0.7 最优
  候选为 `-0.2359/+0.4970`（成本 `46854`）；联合门余量最优候选为 `+0.1474/+0.4898`（成本 `45830`），
  距联合门仍为 `-0.6526` 个百分点。
- 704 个合法候选中没有一个同时达到预登记的 `+0.8/+1.0` 门。该结果触发此前 Pro 冻结的当前
  THUMOS14 controller holdout、H65 priority sequence、三档真实 observation 动作空间与资源范围内的停止条件。
  不扩大搜索、不改预算档、不降门、不训练控制器、不做选择后 bootstrap 或 official test。
- 这是训练侧开发集 privileged oracle 的机制负结果，没有不确定性区间、可部署策略或论文正式性能含义。
  当时唯一下一步是把完整中立证据与最新 GitHub 仓库、实际分支、精确提交和关键文件链接返回 Pro；该动作已在
  下文完成，Codex 没有自行选择新机制或恢复旧路线。

## 2026-08-31 — Pro 完成项目级 STOP 并关闭当前三档预算转移路线

- Fresh exact DUCA Project 对话
  <https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9501ec-3cc4-83ea-ba60-b8302e6e2632>
  完整结束。终态元数据核验了 exact Project、nonce、浏览器 `Pro`，以及最新公开仓库、实际远端分支、精确
  `33e4ed137c33eef07f0452b44506a6993bdf7535` 提交、runner、测试和未修改 allocator 的永久 GitHub 链接。
- 完整报告保存于
  `.cvpr-pro-lab/pro-reviews/runs/duca-whole-video-terminal-adjudication-v001/visible-report.md`；Prompt SHA-256 为
  `128f49e6dd43e3835057e9d8cc2379cba92b319425aff0683d1069babbff2f46`，报告 SHA-256 为
  `4ed9e00834d9980bf44fc703d559de50abdd8f9b9e48d1764679f7c9e007359c`。
- Pro 裁决 `STOP`：在当前 THUMOS14 训练侧 controller holdout、冻结 H65 detector 与 priority sequence、
  K256/K384/K512 密封预测、真实 observation 成本和资源范围内，停止基于在窗口或视频间转移三档预算的 DUCA
  方法。Marginal-v1、cap-release、96-state 与 whole-video 分支均只读保留。
- 支持的机制判断是：视频内混合预算不是失败的充分解释，Avg-mAP 与高 tIoU 最优状态分离，当前动作空间没有
  预登记联合效用。最强但未验证的解释是 H65 priority sequence 与仅在 K384 下训练的 detector 缺少跨预算
  兼容、单调且边界敏感的表示；该解释不能被写成已证实原因。
- 当前没有新的 Builder、Critic、Evaluator 或 Slurm 任务。不开新分支、不恢复旧路线、不改门或扩大搜索。
  只有边界之外的新机制先在独立训练侧开发划分上、匹配真实计算且不事后改门的条件下展示预登记联合 oracle
  headroom，才可由新的项目级科学问询重新开放。

## 2026-08-31 — 审计并吸收用户提供的 Gemini 动态预算分析

- 用户提供了一份围绕 `33e4ed137c33eef07f0452b44506a6993bdf7535` 的外部动态预算分析。Codex 已将其与
  当前代码、整视频终态结果和 Pro `STOP` 边界逐项对齐，规范化记录于
  `research-wiki/sources/2026-08-31-duca-gemini-dynamic-budget-review-audit.md`。
- 可吸收内容包括：重型 observation 减少不等于端到端等比例加速；当前动作空间的 Avg-mAP 与高 tIoU
  最优状态分离；跨预算兼容表示是值得新科学问询检验的候选假设。
- 需要纠正的关键点包括：H65 实际为优先级调制的预算校准系统采样而非普通全局 Top-K；704 个候选来自
  密封预测上的 privileged oracle 枚举而非学习控制器；跨预算分布偏移仍是未验证解释，不是已确定根因。
- 本次吸收没有新实验事实或新 Pro 裁决，因此不改 `PAPER_PROGRESS.md`，不建立 Builder/Critic/Evaluator/Slurm
  任务，也不把 Gumbel-Softmax、Mamba、Block Drop、CUDA/TensorRT 等多变量组合直接转成实现计划。

## 2026-08-31 — 摄取 REVISE，冻结多预算检测器适应的单变量科学问题

- 用户提供的最新裁决保留旧动作空间的 `STOP`，但在其边界之外提出一项新实验：保持现有嵌套
  K256/K384/K512 位置构造，只比较固定 K384 训练和三档多预算检测器适应。跨预算表示不匹配是待检验假说，
  不是已证实根因。
- 当前正文与同消息较早附件在一处冲突：附件同时改为预算原生 H65 选点，正文要求第一轮冻结现有嵌套位置。
  已按较新的正文处理，不把两个变量合并。
- 新路线以 `04c35a3b...` 为 H65 模型基座；`33e4ed...` 只提供真实变长执行、packet 对齐、actual-observation
  accounting、K384 parity、whole-video 评价和原始顺序保持，不承担新模型科学身份。
- 规范化裁决和实验设计分别写入
  `research-wiki/sources/2026-08-31-duca-multi-budget-detector-adaptation-revise.md` 与
  `research-wiki/experiments/duca-multi-budget-detector-adaptation.md`，并同步更新 `PAPER_PROGRESS.md`、
  `query_pack.md`、`anti_repetition.md`、`decision_history.md` 和 `source_registry.md`。
- 当前没有实现或运行授权：裁决尚未给出唯一的匹配成功更新数/训练轮数，也没有给出未参与学习或规则选择的
  训练侧开发视频 ID。两项冻结前不建立 Builder、Critic、Evaluator、PRE_RUN 或 Slurm 作业。

## 2026-08-31 — 登记完整训练集与完整官方留出评估的正式比较约束

- 人类要求后续可进入论文比较的固定 K384 控制臂与多预算适应臂都使用完整训练集；设计冻结后，两臂只在完整
  官方 held-out evaluation split 上作最终可比评估。训练子集、40-video holdout、pilot 或 smoke 不能替代正式
  主结果。
- 官方留出评估仅用于冻结方案的评价，不参与训练、checkpoint 选择、阈值/规则选择、路线选择或反复调试。
  若训练侧开发划分仍用于机制门，它必须与最终完整训练和最终留出评估明确分层。
- 当前资料中存在真实协议冲突：OpenTAD/DUCA 记录常用 `training/validation` 并报告 211 个评估视频；
  ActionFormer 官方记录使用 `validation/test`，历史正式运行报告 212 个评估视频。未静默选择其中一个；Pro 必须
  冻结精确 subset 名称、完整视频 ID、annotation、类别映射和 evaluator。
- 约束已记录于
  `research-wiki/sources/2026-08-31-duca-full-train-official-test-human-constraint.md`，并同步到实验页、
  `PAPER_PROGRESS.md`、`query_pack.md`、`anti_repetition.md`、`decision_history.md` 和 `source_registry.md`。
- 当前正在生成的 Pro 对话早于本约束，保持原会话不追问、不打断、不重提。终态后先审计其训练/评估划分；
  如有冲突，另行发起 fresh Pro 裁决。在此之前不建立 Builder 或训练任务。

## 2026-08-31 — Pro v001 完成；160/40 数据协议与正式全量要求冲突

- 同一精确 DUCA Project 的 Pro turn 已完成并通过 Project、conversation、nonce 与模型选择核验。完整回答保存在
  `.cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md`。
- Pro 选择 `CONTINUE`：保持现有嵌套 K256/K384/K512，只改变训练预算分布；两臂从 H65 Stage-1 terminal EMA
  开始，各完成 6,000 次成功 update，匹配优化、随机性、可训练参数与 terminal EMA，并按实际 observation 成本
  校准候选概率。旧冻结检测器的 704-state 路线继续 `STOP`。
- 该 prompt 早于人类的完整数据约束。回答把 200 个训练侧视频切成 160 train / 40 development，并禁止访问
  official test，因此不能作为当前正式可比实验的 Builder 授权。
- 已将终态与冲突记录于
  `research-wiki/sources/2026-08-31-duca-multi-budget-pro-freeze-v001.md`，并构建新的独立 Pro 请求，专门冻结完整
  训练、完整官方留出评测、211/212 身份差异及诊断到正式实验的隔离关系。当前没有新代码或算力任务。
- 新请求已在精确 DUCA Project 中作为一个新 conversation 提交：nonce
  `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`，conversation
  `6a952a19-9294-83ea-b09f-5524e7825316`，模型选择器为 Pro。当前仅等待同一会话终态，不追加 follow-up、
  不重提，也不在裁决前建立 Builder。

## 2026-08-31 — 正式比较与显式墙钟等待进入项目级规则

- 根目录 `AGENTS.md` 新增正式可比实验约束：论文主比较的 matched arms 必须在完整冻结训练集上重新训练，
  并仅在完整独立官方留出测试/评估集上作预注册后的最终比较；训练子集、160/40、pilot、smoke 和缩短训练
  只能用于机制诊断，不能替代论文主结果。
- 留出测试/评估集不得用于训练、checkpoint/epoch、超参数、阈值、规则或路线选择。OpenTAD/DUCA 211-video
  `validation` 与 ActionFormer 212-video `test` 的真实协议冲突必须在 Builder 前由 Pro 冻结，不允许 Codex
  静默合并。除预注册唯一变量外，两臂还必须匹配起点、训练预算、随机性、后处理、评价器、真实成本、
  prediction 保存和不确定性协议。
- 同一文件新增显式墙钟等待规则。此前真实 `sleep 600` 已经过完整 600 秒，但由后台作业句柄分段等待终态，
  因而只证明墙钟时间真实经过，不作为新规则所要求的单次前台阻塞范例。今后从显式计时命令开始到返回，计时
  必须是唯一活动，不发送文字、不检查状态、不读写文件、不调用其他工具或并行工作。
- 这是实验协议和执行纪律的项目级同步，不是模型实现、实验结果或新科学裁决；没有因此修改模型、提交作业、
  访问留出评估集或触碰当前正在生成的同一 Pro 会话。

## 2026-08-31 — 无实质工作期间改为终端静默等待

- 根目录 `AGENTS.md` 将墙钟等待从“仅响应用户显式要求”扩展为项目默认执行纪律：只要 Goal 尚未完成、且
  唯一正确动作是等待同一个已登记的 Pro、下载、构建、训练或评测任务，Codex 就保持该 Goal，并使用终端
  阻塞计时命令做有限时长的真实等待。
- 每段计时期间不得产生 commentary、进度或倒计时输出，不得检查状态、读写文件、调用其他工具、启动任务或
  并行工作。计时结束后至多只读核验一次同一个权威句柄；若仍未终态且无实质变化，则立即进入下一段静默
  等待，不重复报告“仍在运行”。
- 用户指定的间隔优先；未指定时，Pro/下载/构建等任务使用至多 10 分钟的有限间隔，完整训练或正式评测使用
  至多 30 分钟的有限间隔。只有终态、会改变下一动作的实质变化、真实阻塞或用户主动询问才允许输出。
- 这是执行等待方式的规则修订，不是模型、数据协议、实验状态或科学结论变化；本轮没有因此读取正在生成的
  Pro 内容、提交训练或访问留出评估集。

## 2026-08-31 — Pro 完整数据协议终态；当前只执行 211/212 身份核验

- 同一精确 DUCA Project 的 Pro job `j-czfmae` 在一次 10 分钟终端静默等待后权威终态 `exited 0`。终态 manifest
  验证 Project `g-p-6a91061f789881918ccd8357ca3d6c92`、conversation
  `6a952a19-9294-83ea-b09f-5524e7825316`、nonce `DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831`
  和 Pro 模型选择一致；完整回答保存于
  `.cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md`。
- Pro 选择 `REVISE`：保留固定 K384 对照与嵌套 K256/K384/K512 多预算适应的单变量问题，撤销上一轮 160/40
  正式协议、旧 40-video holdout、有标签训练侧 mAP 门和 whole-video oracle。未来正式两臂都使用完整 200-video
  `training` 集合、同一 Stage-1 EMA 和 6,000 次成功 update；正式预测先密封，再在完整 `validation` held-out
  集合上执行一次统一评测和 10,000 次整视频配对 bootstrap。
- 当前唯一任务不是模型训练，而是基于 `04c35a3b...` 的最小无标签 split identity audit：物化 annotation、loader、
  物理视频和 evaluator 集合，找到历史 211 IDs 与 source-backed ActionFormer 212 IDs，解释所有差集。身份 Builder
  只允许修改审计脚本和聚焦测试；独立 Critic 与 N16R4 CPU Evaluator 后必须先把 PASS/BLOCK 返回 Pro。
- 身份准入前禁止模型实现、checkpoint、PRE_RUN、GPU、训练、held-out temporal labels、预测和 mAP。用户同轮提供的
  `research_project_analysis.md` 摘要未在仓库找到；其中渐进解冻、STE 退火、五档预算和 ActivityNet 建议不属于
  本轮 Pro 授权，不改变当前路线。

## 2026-08-31 — 完整吸收非等间隔时序采样外部提案

- 用户提供的 266 行外部研究提案已原文归档到
  `docs/methods/reviews/2026-08-31-duca-irregular-temporal-sampling-external-proposal-raw.txt`，并在
  `research-wiki/sources/2026-08-31-duca-irregular-temporal-sampling-external-proposal.md` 完成结构化吸收。
- 提案被拆成四组可分别证伪的机制问题：原生连续 tubelet、显式物理时间编码、稀疏到密集时序重构和端到端优化
  稳定性。144/48 双流分配、连续时间旋转位置编码、高斯时序散射、Gumbel 退火、H65 蒸馏和五档预算曲线均保留为
  未验证设计，不作为根因或性能证据。
- 记录明确保留了提案中的 H65 65.13、端到端 58.39、连续块 49.89、PJST 64.59、预期 +0.8--1.5 个百分点和
  >=64.5 等数值，同时区分项目已登记结果、需要精确实验身份解释的历史数字和纯目标值。
- 本次摄取没有建立模型分支、Builder、PRE_RUN、GPU 或训练任务，也没有修改论文进展或科学决策。当前唯一授权
  任务仍是 211/212 数据身份审计；未来任何 selector、位置编码、重构核或训练课程变更均须在当前 Pro 序列之外
  重新裁决。

## 2026-08-31 — 四项非等间隔机制完成历史防重复核验

- 原生连续 tubelet 并非未尝试：连续 cliplet、固定 K384 native-tubelet uniform/task-state coreset 与 CONTIG bundle
  都已有实现或完整训练；但没有实验固定同一 RGB 集合并只改变 tubelet 内物理连续性，因此配对失真仍未被因果回答。
- 显式物理时间已有多轮完整训练。早期 PhysTime 为负，特定 physical-metric 架构为正，最接近单变量的
  RankPack/TrueTime 为 61.5722 对 62.1930 Avg-mAP；其单种子和缺少配对区间限制了主张，连续时间旋转位置编码未做。
- 稀疏到密集的 hidden-linear 桥已有聚焦测试和 CUDA Gate，但正式训练在结果前因工程问题终止；没有 nearest、linear、
  Gaussian 重构核的匹配 TAD 边界质量对照。该核比较是真正未执行的方向，不应重复搭建线性桥。
- 分阶段训练、homotopy 与 CellCF 蒸馏均有历史尝试。30+60 课程的 65.385724 Avg-mAP 使用 90 总轮次，不能与
  60 轮控制作公平课程归因；homotopy 与 CellCF 未超过相关控制。等成功 update 的 scratch/warmup 或蒸馏/no-distill
  单变量比较仍未闭环。
- 上述核验只收紧未来防重复边界，没有形成新科学裁决或模型授权。当前唯一任务仍是 211/212 数据身份审计；任何
  新机制实验仍需在数据准入后由 Pro 单独冻结。

## 2026-08-31 — Gemini 全量预审完成，完整 Wiki 发布到 GitHub

- 在下一次 Pro 科学裁决前，使用 `agy` CLI 的 `gemini-3.7-flash-high`、`effort=high` 对当前完整 Wiki 历史与
  H65、TrueTime、whole-video 三个精确实现快照完成一次只读预审。终态标记为
  `GEMINI_DUCA_ADVISORY_READY`；完整原文保存于
  `research-wiki/sources/2026-08-31-agy-gemini-comprehensive-wiki-code-review-v001.md`。
- Gemini 建议把多预算检测器适应作为下一候选，但其把跨预算表示不匹配、旧端到端失败和非连续 tubelet 影响写成
  已证实根因的部分没有单变量证据。该报告只作为独立咨询与审查问题清单，不替 Pro 选线，也不授权实现或实验。
- 为避免把数十个 Wiki/代码文件塞入 Pro Prompt，建立独立公开分支
  `codex/duca-wiki-complete-sync-20260831`，初始完整快照提交为
  `9624d220d67d3947bfe7e49379e8bf3168e52b8e`。该提交以 whole-video 终态代码
  `33e4ed137c33eef07f0452b44506a6993bdf7535` 为基座，只发布完整 `research-wiki/`、
  `PAPER_PROGRESS.md` 和 Wiki 跟踪规则，没有修改模型、配置、训练、数据或评价代码。
- GitHub 深度审查入口为
  `research-wiki/GITHUB_REVIEW_INDEX-2026-08-31.md`。它链接完整 Wiki 树、Gemini 原文、全部远端分支入口，
  以及 H65、TrueTime、native tubelet、dynamic budget、Coverage、Marginal、cap-release 和 whole-video 等关键
  精确提交。下一次 Pro Prompt 只提供该公开索引、完整仓库与精确版本链接，并要求 Pro 自主逐版本阅读和裁决。
- 此前两次 Oracle 本地传输均在远端提交前终止：第一次卡在超长前端编辑器写入，第二次只遇到已终止会话的本地
  重复检测，后续压缩尝试在 `RUNNING_PREFLIGHT` 被用户的新材料要求取代。均未生成新 conversation、未提交 Prompt、
  未开始 Pro 推理；现有 profile61 浏览器、端口、登录和页面均未关闭或重启。
