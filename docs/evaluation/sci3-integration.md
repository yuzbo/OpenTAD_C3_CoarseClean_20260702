# SCI3 v3：研究规格接入与当前执行边界

2026-09-08。来源是用户提供的 `GeoSparse_TAD_Research_and_Agent_Commands_v3.md`，不是当前模型已经实现的声明。运行中的模型 M 仍为 b70ae056c495b43ca3f305fe438926b97b2723b5。本分支保留 `geosparse_ext/`、官方 `opentad/`、训练配置及入口原样，新增 `geosparse_research/` 作为研究数据生产器。

## 接受的研究命题

- A：同一执行器、TIA、风险与成本下，等成本的有符号计算价值监督是否有独立收益，并能否迁移到 MoD？
- B：在相同 evidence、完整 metadata 和合理容量下，新增来源条件机制是否减少错误延伸与同类粘连，同时保持召回？
- C：同成本计划下，任务细化收益和 coarse 校准是否比相似性及成熟合并方法更好；模式切换是否实际引入有害的时间变化？

不预设这些命题成立。强近邻拥有相同的公共能力。现有 C 的 signed coarse-to-fine probe 已实现，等成本 swap 与新风险不能借用它的名称充当完成。

## 与当前协议的差异

| v3 材料 | 当前事实与处置 |
| --- | --- |
| 340a3541、224px、14×14、7×7组 | 当前 M 使用768×160、10×10 native patches、5×5个2×2组、native384 TIA、48个Heavy parents；旧版本只保留为历史。 |
| 用户消息中的负 attention 成本项 | 以附件研究规格的正号为准：每层每parent为 `12*k*d*d + 2*k*k*d`。不能按总token数替代二次项。 |
| 三种子、Discovery fit/probe/select、主结果第60轮 | 与此前明确的seed0优先、全200训练、211视频全量EMA best不同；已提出协议澄清。答复前沿用现行运行协议，不建立新holdout，不提交seed1/2。已训练视频的诊断标IN_SAMPLE，不称held-out。保留固定60轮及共同50/60节点的补充比较。 |
| 核心805请求、37方法、旧609训练索引 | 当前只收到v3 Markdown，本机两份ZIP均为旧1538包；无法核验805个原始ID。当前正式旧矩阵实际为1545项/612训练。缺失的机器清单登记MISSING_SOURCE_PACKAGE，不猜造ID。 |
| 0.30/0.40/0.60 Heavy目标 | 是新增成本工作点，与旧fixed-q=.5、dynamic软目标=.5分开。新增分配器应报告实际MAC及可行性；不能沿用无约束argmax的高分当等成本结果。 |
| OpenTAD_Strict_Audit_20260907 | 是另外六路线的VideoMAE-S、REF/H65/CT-DP等固定版本审计。69.03/68.73/67.58不能替换当前GeoSparse VideoMAE-B的官方身份或结果。当前未核验其历史产物，不操作该项目作业。 |

## 第一批公共实现的精确设计

采用 v3 已给出的同模型成对干预与视频聚类统计设计。先连接真实已有接口，避免创建第二套模型或通用调度器。

1. `equal_cost_swap` 接受真实 `RoutePlan` 与 native validity，要求 donor 已选、candidate 未选，位于同一 parent 且有效成员数相同。A/B交换独立Heavy原子，C交换fine/coarse组。修改后逐parent实际token数量必须相同。反事实计划不伪造采样概率，也不直接用于PG actor更新。
2. `CounterfactualRunner.evaluate_pair` 使用同一个已经加载权重的 GeoSparseDetector、同一个 VideoBatch、GT和epoch，在确定性eval状态下分别执行真实 `forced_plan` 前向与源检测损失。每条路径重跑Heavy及其后续TIA/head；不使用跨计划feature缓存。保存实际执行trace、cls/reg/total有符号差和检测有效mask；恢复RNG、buffers、module modes及被前向改动的临时状态。第一版只输出源R0的总项分解，不冒称已实现R1、逐query风险或训练swap校准。
3. `paired_video_map_bootstrap` 直接调用固定官方AP函数，按video成对抽样；重复抽到的视频获得独立ID，空预测视频与背景视频保留。每次在完整重采样GT/预测上重算类别AP，再按官方规则平均，禁止平均每视频AP。原始类别集合中有类别在某draw无GT时，该draw的固定类别mAP为未定义，记录missing classes并不重抽；CI注明条件及有效draw数，不能把缺失类记0或悄悄删类。seed不确定性另行报告。

这三个接口不会改变现有动态预算分配，也不会自动生成新的模型精度。模型诊断与正式训练分开；没有真实视频和checkpoint输出时只能报告代码验证，不报告机制成立。

## 尚需完成的接口和方法

| 表面 | 可复用 | 尚缺 |
| --- | --- | --- |
| 计划执行 | RoutePlan、GeoSparseModel.forward(forced_plan=...)、真实稀疏Heavy | 持久计划bank、逐层MoD计划、成本可行的动态allocator |
| 风险 | 源ActionFormer losses、GT-only assigner、已有miss-aware metrics | query分解、实例平衡R1、重叠分类/回归归属规则及真实训练CF loss |
| B | 五receiver、union几何、native384 query/TIA | 完整三层support、固定producer的EvidenceBank、mTAN/metadata-rich gate及训练 |
| C | 逐层mixed回写、signed refinement probe | 等成本swap训练、coarse affine、相似性/ToMeSD/MSViT等强适配 |
| 统计 | 召回/边界paired bootstrap | 新官方mAP接口的真实成对结果、多种子波动、预注册primary contrast |
| 图件 | 原有训练/预算/风险图与实际预算专项图 | v3的22图数据生产与正式receipt；任务注册不等于图完成 |

## 验证与部署边界

计划交换要在真实A/B/C执行器上核对QKV/MLP计数和有效mask，覆盖尾部、空GT、K0/Kall拒绝交换、异常退出状态恢复。同计划pair应给出零差；反向交换应反转signed差。统计测试用官方evaluator交叉核对，包含重复视频、漏检、背景FP及缺失类别draw。

本机默认Torch的c10.dll加载失败，使用既有OpenTAD环境中的独立CPU测试checkout，不修改全局环境。生产GPU准入与真实科学结果另计；不运行原124项测试代替新功能验证。现有六项主方法及已提交控制保持其身份继续训练，新增研究请求不能重复它们，也不以其mAP作为启动门槛。

30个显式工单、22图和6表按附件的可读ID记录于 `sci3-intake.json`；它是本地接入台账，不是缺失的原始805任务manifest。每项待完成证据保持UNTESTED/PENDING。无真实结果不生成装饰性曲线。

## 已完成的首批验证

代码提交 `b85eac309bd9adb7f44f06fc7732750517e4a613` 在独立远端目录 `/data/run01/sczc063/yuzibo/geosparse_official_20260908/sci3_evidence_b85eac30/repo` 以干净提交运行，17项新增测试全部通过（pytest 64.45秒）。实际小规模 VideoMAE、原TIA、ActionFormer参与前向；输入为合成数据，CPU-only。原始日志在操作员执行包 `official_adatad_audit/sci3_verification_b85eac30/remote.stdout.txt`，没有启动训练。独立只读子代理复核未发现这批接口的阻断错误。

复现代码检查：

```bash
python -m pytest tests/test_sci3_interventions.py tests/test_sci3_bootstrap.py -q
```

测量接口的调用方必须提供已经正确加载权重的 `GeoSparseDetector`、来自真实视频的 `VideoBatch` 与元数据、同一窗口的GT，以及与该模型原生网格一致的 `RoutePlan`。`CounterfactualRunner` 不加载或选择checkpoint；输出须由实际实验记录关联到run、epoch、权重及split。此接口的源R0采用eval正样本归一化，不能当作训练时带历史normalizer的损失或实例平衡R1。

`paired_video_map_bootstrap` 接受官方 `mAP` loader已经去重和映射类别后的GT与两份预测DataFrame，并显式传入完整split的video IDs（包括无GT或空预测视频）。输出以0–1比例为单位，差值方向是right-minus-left，按视频抽样的区间不替代seed方差，也不纠正test-best选点偏差。两个结果的权重、协议及成本可比性仍由具体研究工单控制。

下一批依赖是：真实视频/已锁定checkpoint的成对干预产物与GPU验证；R1/query风险归属和训练swap校准；强MoD/LITE、receiver和merge适配；原始805请求包解析。以上均未被这17项测试标为完成。
