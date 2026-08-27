---
type: idea
status: designed
updated: 2026-08-23
project: DUCA
task: temporal_action_segmentation
evidence_level: official_videomae_s_baseline_plus_human_approved_design
---

# DUCA 向时序动作分割迁移的可行性与最小研究合同

## 结论与证据边界

DUCA 可以迁移到时序动作分割（Temporal Action Segmentation, TAS），但不能通过简单替换
检测头完成。TAS 要求为每个评估时间点输出动作类别；DUCA 当前只对少量物理连续片段执行昂贵
视频主干，并在检测前恢复到稠密物理时间轴。可迁移的核心是“低成本全局侦察、物理连续片段
采集、真实重计算下降和原始时间坐标”；不可直接继承的是 ActionFormer 检测头、区间回归、
Soft-NMS 和 mAP 评价协议。

本节点只达到 `discussed / migration_design`。没有 TAS 代码、数据核验、预训练权重、训练、
逐帧准确率、Edit 或 segmental F1 结果；当前 DUCA-TAD 正式实验不因本讨论改变。

## 代表性模型谱系

1. **MS-TCN / MS-TCN++**：多阶段空洞时序卷积，先作逐帧预测，再逐阶段纠正；输入输出合同
   简洁稳定，适合作为 DUCA 的首个迁移宿主。MS-TCN++ 的官方实现还给出 Breakfast/GTEA
   四折和 50Salads 五折协议。
2. **ASRF**：并行预测逐帧动作与边界，用边界分支抑制过分割。它是检验 DUCA 边界证据
   是否真正有用的直接基线。
3. **C2F-TCN**：粗到细编码解码与多尺度预测融合，代表卷积式多分辨率分割。
4. **ASFormer**：局部/层次注意力编码器与迭代解码器，代表强 Transformer 基线；其规则
   稠密时间窗假设使它不适合成为第一个稀疏迁移宿主。
5. **UVAST**：从视频到动作转录序列，再通过对齐恢复逐帧标签；科学问题和直接逐帧分类不同。
6. **DiffAct**：以扩散去噪迭代修正动作序列，性能强但多步推理成本与 DUCA 的效率目标冲突，
   适合作为精度上界参考而非首个迁移底座。
7. **LTContext**：结合局部窗口和稀疏长程注意力，代表长视频上下文建模。
8. **FACT**：帧分支与动作 token 分支双向交叉注意，兼顾效率和长程语义；与 Query-Bridge
   的协同建模思路接近，可作为第二阶段结构参考。
9. **BaFormer**：以边界感知 query 和连续动作片段提案直接形成分割；它与 DUCA 的边界/片段
   视角最接近，但迁移改动显著大于 MS-TCN++。

MS-TCN++ 只接收预提取的 2048 维时序特征，适合作为官方特征级 TAS 评价锚点，但不能验证
backbone 前选帧是否减少真实视觉计算。用户已明确要求原始 RGB 输入，因此首个端到端迁移宿主
改为 **EAST（End-to-End Action Segmentation Transformer）**：保持其 VideoMAEv2、动作片段检测、
高帧率聚合和 MS-TCN 细化路径，在同一像素输入合同下插入 DUCA。MS-TCN++ 不删除，但降为
特征级诊断和细化头参考；ASFormer/FACT/BaFormer 仍不与首个端到端迁移同时改变。

## 数据集与优先级

1. **50Salads**：50 个长视频、25 名参与者、约 4.5 小时、17 个动作类，标准五折交叉验证。
   数据量小、边界密集，最适合首个机制证伪。必须报告逐帧准确率、Edit 及 F1@10/25/50。
2. **Breakfast**：52 人、18 个厨房、约 77 小时、48 个动作单元，标准四个主体划分。
   它检验长视频、多场景和跨主体泛化，作为第二个正式数据集。
3. **GTEA**：4 名参与者、28 个第一视角视频、11 个动作，标准四折。规模很小，适合边界
   诊断，不能单独承担主要结论。
4. **Assembly101**：4321 个视频、513 小时、53 人、12 视角，提供粗/细粒度密集标注；
   适合在前两项通过后检验多视角、长尾和规模扩展，不应作为第一个调试数据集。

50Salads、Breakfast、GTEA 和 Assembly101 的本地/远端视频、特征、标注与许可证状态当前
均未形成可启动的权威绑定，因而本节点不授权实验启动。2026-08-21 对共享根的只读检查未定位到
这些数据集的确定路径；大范围递归检索超时，所以状态是 `UNVERIFIED/MISSING_AT_COMMON_PATHS`，
不能写成绝对不存在。

### FineGym 与 FineDiving 的适用边界

- **FineGym** 是有价值的第二阶段压力测试，但不是首个标准 TAS 复现集。官方数据包含
  event/set/element 的层级语义以及 action/sub-action 的时间区间，适合检验 DUCA 是否漏掉快速、
  短时、边界敏感的体操子动作；但是官方 Gym99/Gym288 协议以 action instance 的识别/定位为主，
  没有与 MS-TCN++ 的逐帧标签、四/五折和 Acc/Edit/F1@10/25/50 直接等价的 loader 与 evaluator。
  将其改成 TAS 必须先冻结层级展开、背景、重叠区间、采样率和评价协议，结果只能称自定义迁移协议。
- **FineDiving** 的官方主任务是 procedure-aware action quality assessment（程序感知动作质量评价），
  不是标准 TAS。它有 step 转换帧和逐帧/步骤标注，可作动作阶段与边界的辅助迁移诊断；但样本是
  经过筛选的完整跳水过程，官方评价是质量分数/程序建模，不能直接承担 DUCA-TAS 主结论。
- 两者均不能直接使用 MS-TCN++ 官方 loader。FineGym 还依赖 YouTube 视频可得性；FineDiving
  需签署数据发布协议并向作者申请。当前共享 N16R4 根未核验到两者的合法完整数据副本。

因此数据顺序冻结为：先用 **50Salads 原始 RGB** 完成 EAST 端到端官方锚点和 DUCA 接口证伪；
MS-TCN++ 五折只作特征级评价锚点，不能产生端到端减算主张。通过后再将 **FineGym** 作为短动作/
层级边界外部压力测试。FineDiving 仅在明确建立派生 step-segmentation 协议后作为辅助集，不替换
标准 TAS 主基准。

## TAS 特有的科学问题

TAD 中，actionness 可以区分动作与背景；TAS 中，两个相邻动作常常都处于前景，例如
“切菜”紧接“搅拌”。因此，只有 actionness 的侦察器可能把整个区间看成一个连续动作。TAS
迁移必须把边界头训练成**动作类别转换证据**，而不只是前景的开始/结束证据。部署时仍只消费
侦察器预测，不能消费验证/测试标签、教师预测或缓存结果。

第二个风险是稀疏片段之间存在不可观测区间。物理时间插值能修正坐标，不能凭空恢复未观察到
的短动作。大缺口必须显式保留 coverage/uncertainty，而不能把插值特征当成同等可信的真实帧。
MS-TCN++ 的平滑损失还可能在插值之后再次过度平滑，尤其伤害短动作和 F1@50。

## 最小模型合同

```text
完整低分辨率视频
  -> DUCA scout：actionness + action-transition boundary
  -> 确定性连续 cliplet 采集（固定 M 首证伪，随后才允许动态 M）
  -> 仅选中高分辨率片段进入 VideoMAE
  -> 按 int64 原始帧号/时间戳重建到数据集官方评估网格
  -> MS-TCN++：逐帧动作类别预测与多阶段细化
  -> 官方逐帧标签展开和 Acc/Edit/F1@10/25/50
```

冻结张量合同：

- `selected_t: int64[B,K]`：被重计算帧在原视频中的物理位置；禁止用选中序号替代。
- `selected_feat: float[B,C,K]`：只来自真实执行的重主干输入。
- `dense_feat: float[B,C,L]`：按物理位置重建到官方 TAS 时间网格。
- `valid_mask: bool[B,1,L]`：合法时间点；padding 不参与损失和评价。
- `coverage/uncertainty: float[B,1,L]`：区分真实支持与大间隔插值。
- `stage_logits: float[S,B,A,L]`：MS-TCN++ 每阶段逐帧类别输出。

重建必须使用物理时间 `searchsorted`/邻域插值；区间端点使用最近合法观测。最终预测必须按照
官方映射回原始帧率或官方特征率。不能先把稀疏观测按 rank 当成等间隔序列，再事后修正标签。

训练损失采用 MS-TCN++ 官方逐帧交叉熵与平滑项；侦察器 actionness/transition 损失单独记录。
当前 DUCA 硬采集对索引是 detached 的，因此分割损失不会自动训练 selector；任何可微桥或
Query-Bridge 都是后续独立假设，首个迁移不得把它写成已有能力。

## 最便宜的真实证伪实验

数据集固定 50Salads，并把证据分成两层，禁止跨层计算采样增益：

1. **官方协议锚点**：完全复现官方五折、I3D 特征、MS-TCN++、训练日程和评价脚本。它回答
   TAS 代码与指标是否正确，但预提取 I3D 已经支付视觉特征成本，不能支持 DUCA 端到端节省。
2. **特征空间机制诊断**：在同一份 I3D 特征上比较稠密、连续均匀 support、DUCA 固定 M
   support 和 scout-only 下界。它隔离稀疏支持与物理时间重建是否保留分割信息，但仍不是重主干
   计算节省证据。
3. **同主干端到端因果比较**：所有臂使用同一视觉主干、分辨率、MS-TCN++ head、训练日程、
   seed 和 checkpoint 规则；比较 dense VideoMAE、连续均匀 cliplet VideoMAE、DUCA 固定 M
   cliplet VideoMAE。只有这一层可以报告真实视觉重计算、端到端延时和显存差异。

不再增加已经被反复运行的无关随机采样矩阵。连续均匀 cliplet 必须保留，因为它与 DUCA 使用
同一种输入单元，能够隔离语义选片是否优于位置均匀选片。主停止指标是 F1@50 与 Edit；同时报告
Acc、F1@10/25、短动作、动作转换边界距离和覆盖缺口。

若同主干固定 M 的 DUCA 臂不能在相同计算量下优于连续均匀 cliplet，停止动态预算扩展，先判断是
侦察器不能识别类别转换，还是稀疏重建不可辨识。若通过，再冻结 M 曲线，并以相同 realized
mean heavy compute 比较动态 M；成本必须包含侦察器、采集、VideoMAE、重建、MS-TCN++、显存和
端到端延时，不能只报所选帧数或理论 FLOPs。

## 代码迁移边界

建议在独立 TAS worktree 中增加任务适配面，不修改当前 TAD 主实验：

- 复用 DUCA scout、连续 cliplet acquisition、`selected_t` 和真实 heavy-path 账本；
- 增加 `DUCATASPhysicalReconstructor`，输出稠密 TAS 特征、mask 和 coverage；
- 以官方 MS-TCN++ `PredictionGeneration/Refinement` 为首个 head，保持其折数、类别映射、
  loss 和 evaluator；
- 配置分别绑定 50Salads 五折与 Breakfast 四折，不共享标签空间；
- 禁止把 THUMOS14 mAP、ActionFormer NMS 或现有 DUCA-TAD checkpoint 当成 TAS 结果。

## 官方代码身份与执行准备（2026-08-21）

- 官方仓库：`https://github.com/sj-li/MS-TCN2.git`。
- DUCA 独立只读克隆：`E:/DeskTop/TAD/external/MS-TCN2_DUCA_20260821`。
- 官方当前 `master=f423a9e65f4ccb1cd7322eb9f94946a19e787993` 在
  `model.py:14` 含多余的 `MS_TCB` 文本，`MS_TCN2` 无法被 Python 解析。该缺陷来自官方
  2020-11-22 提交，不能静默在复现实验中修补。
- 首个可执行官方历史身份固定为
  `9d31fb3c23467b9ce3030d43b6d33a96869b6422`：它的 `MS_TCN2` 主路径可通过
  `py_compile`；文件中旧 `MS_TCN` 类的 `super(MS_TCB, ...)` 错误不在官方 `main.py` 使用的
  `MS_TCN2` 路径上。若后续需要修正非主路径，必须另建 project-local candidate 并保留未改官方锚点。
- 官方代码只原生绑定 `breakfast`、`50salads`、`gtea`，读取
  `features/*.npy + groundTruth/* + splits/*.bundle + mapping.txt`。FineGym/FineDiving 的适配
  必须在官方锚点复现通过后另建，不能修改锚点后再称“未改官方复现”。
- 当前尚未取得 50Salads 官方约 30 GB 数据包/特征/标签在 N16R4 的可验证路径，因此尚不能提交
  官方五折复现；该阻塞不妨碍准备独立环境与 DUCA 适配设计，但禁止产生性能声称。

### 环境、数据与启动器状态（2026-08-21）

- 官方历史提交已部署到远端只读锚点
  `/data/run01/sczc063/yuzibo/external_official_action_segmentation_repos/DUCA_MS-TCN2_9d31fb3`，
  `HEAD=9d31fb3c23467b9ce3030d43b6d33a96869b6422` 且工作树干净。
- 官方仓库没有发布精确 Python/PyTorch 锁。项目先声明了独立兼容环境规格
  `research-wiki/experiments/ms-tcn2-official-env-20260821.yml`；其 conda 安装先受 home quota、
  后受软件源大包下载失败阻断。现有可执行兼容环境为
  `/data/run01/sczc063/yuzibo/venvs/duca_mstcn2_official_9d31fb3`，基于已验证 OpenTAD 环境建立，
  版本为 Python 3.10.20、NumPy 1.23.5、PyTorch 2.0.1、CUDA runtime 11.8、loguru 0.7.3。
  登录节点 CPU 前向见证通过：输入 `[1,16,9]`，输出 `[2,1,4,9]` 且全有限；这只是运行环境证据，
  不是数据、训练或效果证据。GPU Slurm 环境见证 Job `1247227` 在 RTX 4090 上
  `COMPLETED/0:0`：同一模型输出 `[2,1,4,9]` 且全有限；stderr 仅有 PyTorch 的 cuDNN
  workaround warning，无执行错误。
- 官方 Zenodo 数据记录 `10.5281/zenodo.3625992` 的 `data.zip` 大小为 30,210,005,282 bytes，
  许可证为 CC BY 4.0，官方 MD5 为 `078aa08875747e6264b892ae6e0ac7be`。远端直连下载实测约
  15 KB/s 后已停止。经用户授权，当前改用 N16R4 既有学术代理与 `aria2c` 八连接断点续传；唯一
  进程 PID `3586871`，日志
  `/data/run01/sczc063/yuzibo/duca_tas_mstcn2_9d31fb3/download-aria2.log`，目标
  `data.zip.part`。该任务于 2026-08-22 03:29:17 +08:00 完成，实际文件大小
  `30,210,005,282` bytes；aria2 对预注册 MD5 完成 100% 校验并报告
  `Verification finished successfully / Download complete / stat OK`。`.aria2` 控制文件已由下载器正常移除。
  文件仍保留 `.part` 名称，尚未解压、改名或绑定到官方 loader；这些属于后续独立 PRE_RUN 工作。
- 五折启动器为 `scripts/run_duca_tas_mstcn2_official_50salads_n16r4.sbatch`，远端副本为
  `/data/run01/sczc063/yuzibo/duca_tas_mstcn2_9d31fb3/run_official_50salads.sbatch`。它固定官方
  100 epoch、11 层 prediction generation、10 层 refinement、3 个 refinement stages，并在数据路径
  不完整时失败关闭；本地与远端 Bash 语法检查通过。官方数据未绑定前不提交正式训练。

## EAST 原始 RGB 数据获取状态（2026-08-22）

- EAST 官方代码：`https://github.com/tqosu/EAST`。其 50Salads 数据说明将 EAST 协议视频放在
  `data/50salads/raw_data/video_fps2/`；作者共享版本为 `160x160 @ 2 FPS` RGB 视频。Dundee
  原始采集为 `640x480 @ 30 Hz` DivX AVI，二者不得混称。
- Dundee 官方数据许可为 CC BY-NC-SA 4.0。Dundee 原始 30 Hz AVI 源经学术代理仍返回 HTTP
  503；这不再与 EAST 作者发布的 2 FPS 输入混称。Oregon State Box `video_fps2` 的 50 个
  `rgb-*.mp4` 已通过逐文件下载完成，规范入口为
  `/data/run01/sczc063/yuzibo/datasets/TAS/east_50salads_160x160_2fps/data/50salads/raw_data/video_fps2`。
  50/50 文件大小与 Box 元数据一致，总计 `121,662,019` bytes；全部为 MPEG-4、160×160、2 fps，
  ffprobe 与完整 ffmpeg 解码均通过。早先失败的单归档 `.zip.part` 保留为损坏运输证据，不参与数据绑定。
- 可权威获取的逐帧标注已保存到
  `/data/run01/sczc063/yuzibo/datasets/TAS/annotations/50Salads/avt_50salads_annotations.zip`，大小
  688,521 bytes，ZIP 完整性检查通过，内含 50Salads 的 50 个逐帧 ground-truth 文件和
  `mapping.txt`。该包来自 Facebook Research AVT 的公开复现资源；在 EAST JSON 转换与官方
  split 对齐完成前只记为 `ANNOTATION_ARCHIVE_READY`。
- EAST Box 官方标注归档亦已下载到同一远端目录的
  `east_annotations_bundle.zip.part`，大小 1,417,273 bytes，`unzip -tq` 通过；其中精确包含
  `50salads_category_idx.txt`、五折 `50salads.fps2.split{1..5}.json` 及对应五个 `.swp.json`。
  当前绑定使用普通五折 JSON：每折 training/validation 为 40/10，50 个视频各作为 validation
  恰好一次，视频名、2 fps 帧数和 19 类类别表均闭合。状态为 `RGB_AND_PROTOCOL_READY`；Zenodo
  `.npy` 特征仍只属于 MS-TCN++ 特征级协议，不能充当 EAST 原始视频。
- 官方 EAST 当前 revision 为 `a3233c2e6a6e3bbe36f9663e18180bdc5c126556`；远端 clean 候选
  `37c0d080a2bce948dc73643578f05b2229934d2c` 补齐完整恢复状态、有限 checkpoint 保留，并删除官方 `__all__` 中两个
  实际不存在、会阻断全部模型导入的 InternVideo2 名称；不改变 ViT-G、数据、训练长度、EMA 或
  评估器。目标环境恢复、保留与导出测试 `5 passed`。VideoMAEv2-G K710 预训练权重已绑定。官方 released EAST checkpoint 的 Box
  页面要求 Oregon State 登录，故未取得 released-checkpoint evaluation；不以其他权重替代。
- 2026-08-23 再次按用户要求准备官方 ViT-G 发布权重评估：官方配置、EMA state、fold 参数、
  tIoU `0.3--0.7` evaluator、数据和远端候选均已冻结；远端与本机仍无 detector checkpoint。
  官方 Box link 在两个浏览器会话及匿名 HTTP 中均要求 Oregon State 登录，故当前唯一 blocker
  是发布 checkpoint 的合法访问。没有提交替代权重评估或新的训练。完整 launch-ready 合同见
  `../experiments/east-vitg-released-checkpoint-evaluation.md`。
- ViT-G 的双卡五折 `1249455_1…5` 均在第一次真实前向因 24 GiB GPU 显存不足而终止；默认、
  分块限制和 CUDA 异步分配器三种运行时分配方式得到同一结论，均未完成一个优化步，也没有
  TAS 指标。该配方需要更大显存节点，不能用较小模型结果冒充。官方 ViT-S 配方的完整 split1
  训练—评估联合准入 `1249796` 已完成：2 个 full-data epoch、40 次优化、epoch-1 checkpoint、
  官方 evaluator、峰值显存 2531 MiB、exit 0；五折 200-epoch 正式训练 `1249797_1…5` 均已
  `COMPLETED 0:0`。固定 epoch-199 EMA test 的 Avg-mAP 为
  `81.68/83.59/86.08/84.62/82.97`，五折算术均值为 `83.79`；tIoU
  0.1/0.25/0.5/0.6/0.7 的五折均值为 `89.50/88.21/84.42/80.78/76.03`。checkpoint 每 2 epoch 保存，保留最新 3 个和
  50/100/150/200 epoch 里程碑；恢复包含 model、EMA、optimizer、scheduler、AMP scaler、epoch
  与各 rank 随机状态，终态固定测试 epoch-199 EMA，不按中间 validation 挑选。

## H65 True-Time 迁移设计冻结（2026-08-23）

连续 cliplet 方案不再是首个迁移合同。既有 TAD 证据表明，将计算限制为少量连续 16-frame
片段会丢失长程动作阶段信息；新的已批准合同回到历史 H65 的**非均匀逐帧间接采样**，并在
EAST/VideoMAE-S 中显式保留原始物理时间。

50Salads 的 EAST 五折标注含 19 个动作类别，标注时间轴由动作段完整覆盖，没有背景类；因此
前景/背景 actionness 二分类在这里退化为全正标签。Scout 的任务冻结为：

1. 低分辨率 19 类粗动作识别；
2. 类别转换二分类 `1[label[t] != label[t-1]]`；
3. 由相邻类别分布的 Jensen-Shannon divergence、转换概率、隐藏变化和熵形成确定性正密度；
4. 通过累计密度的逆变换得到有序、唯一、exact-K 的原始时间位置；小模型不直接预测帧索引。

选中的高分辨率 RGB 帧必须在 VideoMAE-S patch embedding 前真实 gather，使实际重计算长度等于
`executed_k`。非均匀 token 在第一次时序混合前接收零初始化的相对物理时间偏置，后端输出按
原始时间坐标重建到规范 2 fps 网格，再进入未修改的 EAST head 与 evaluator。

实验顺序冻结为已有 dense 只读锚点、Uniform-2x、Uniform-4x、H65-Fixed384-TrueTime、
Dynamic-Uniform 和 H65-Dynamic-TrueTime。固定 K 只作机制兼容门与对照；动态 K 才是方法主线。
若 fixed384 在同协议下不能稳定优于 Uniform-2x，则停止动态升级。完整合同位于独立 clean
EAST/DUCA 工作树
`E:/DeskTop/TAD/OpenTAD_DUCA_TAS_H65_TrueTime_20260823/docs/superpowers/specs/2026-08-23-duca-tas-h65-truetime-videomae-s-design.md`，
design commit 为 `abfea355ec0361444cb71ea374a96f65403dcd5d`；设计冻结时状态为 `designed`。

最小代码包已在同一 clean 工作树提交为
`42e1b639d08481b9042f5c4d5ec0544955795b01`。其实现已进入
`implemented_static_verified`：N16R4 上方法、checkpoint 恢复和 backbone 导出 focused tests
共 `18 passed`，绑定规范视频、五折标注与 VideoMAE-S 预训练路径的 launcher 静态预检查为
`17 passed`，完整配置模型实例化通过。以上均为代码/配置证据，不是 PRE_RUN 或效能证据；六臂
训练尚未启动，仍无新增 TAS 性能与端到端成本结果。

最终执行代码随后冻结为 `b0103cea4f3b4aff68463c3f28a1b9f4213c2df6`。正式 Jobs
`1252219/1252220/1252221` 在真实 50Salads、官方五折、seed 42、200 epoch、epoch-199 EMA
协议下均 `COMPLETED 0:0`。Uniform-2x、Uniform-4x 与 H65-Fixed384 的五折 Avg-mAP 均值
分别为 `82.138/78.434/83.192`，mAP@0.7 均值为 `74.070/68.744/74.678`。因此固定
`K=384` 的 H65 相对同预算 Uniform-2x 为 `+1.054` Avg-mAP、`+0.608` mAP@0.7；五折
Avg 差值为 `[+1.07,-0.87,+0.46,+1.68,+2.93]`。该结果状态为 `tested / partially_supported`：
它支持固定预算下语义间接采样具有初步定位保护迹象，但单训练 seed、缺少配对不确定性和完整
成本，且既有 dense VideoMAE-S 五折锚点 Avg-mAP 为 `83.79`，故不能声称统计稳健、动态预算
有效、端到端 Pareto 更优或跨任务泛化。Dynamic-Uniform 与 H65-Dynamic-TrueTime 尚未运行。

## 主要来源

- MS-TCN: https://openaccess.thecvf.com/content_CVPR_2019/html/Abu_Farha_MS-TCN_Multi-Stage_Temporal_Convolutional_Network_for_Action_Segmentation_CVPR_2019_paper.html
- MS-TCN++: https://github.com/sj-li/MS-TCN2
- ASRF: https://github.com/yiskw713/asrf
- C2F-TCN: https://github.com/dipika-singhania/C2F-TCN
- ASFormer: https://github.com/ChinaYi/ASFormer
- UVAST: https://github.com/boschresearch/UVAST
- DiffAct: https://github.com/Finspire13/DiffAct
- LTContext: https://openaccess.thecvf.com/content/ICCV2023/papers/Bahrami_How_Much_Temporal_Long-Term_Context_is_Needed_for_Action_Segmentation_ICCV_2023_paper.pdf
- FACT: https://openaccess.thecvf.com/content/CVPR2024/papers/Lu_FACT_Frame-Action_Cross-Attention_Temporal_Modeling_for_Efficient_Action_Segmentation_CVPR_2024_paper.pdf
- BaFormer: https://proceedings.neurips.cc/paper_files/paper/2024/hash/42770daf4a3384b712ea9c36e9279998-Abstract-Conference.html
- Breakfast: https://serre.lab.brown.edu/breakfast-actions-dataset.html
- 50Salads: https://discovery.dundee.ac.uk/en/datasets/50-salads/
- EAST: https://github.com/tqosu/EAST ; https://openaccess.thecvf.com/content/ICCV2025W/SVU/papers/Wang_End-to-End_Action_Segmentation_Transformer_ICCVW_2025_paper.pdf
- AVT 50Salads annotations: https://dl.fbaipublicfiles.com/avt/datasets/50salads/annotations.zip
- GTEA: https://cbs.ic.gatech.edu/fpv/
- Assembly101: https://assembly-101.github.io/
- FineGym: https://sdolivia.github.io/FineGym/ ; https://github.com/SDOlivia/FineGym
- FineDiving: https://github.com/xujinglin/FineDiving ; https://openaccess.thecvf.com/content/CVPR2022/html/Xu_FineDiving_A_Fine-Grained_Dataset_for_Procedure-Aware_Action_Quality_Assessment_CVPR_2022_paper.html
