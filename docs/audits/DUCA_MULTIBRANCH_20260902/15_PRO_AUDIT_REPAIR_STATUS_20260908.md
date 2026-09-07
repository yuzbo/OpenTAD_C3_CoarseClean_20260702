# Pro 严格审查核对与修复进度

更新：2026-09-08 02:57 CST。范围：核对用户提供的完整审查报告及 CPU 反例，修复已明确且不改变研究定义的局部错误，补查官方基线原始产物。不是六路线全部代码验收，也不是一次所有终态结果重新认证。

## 结论

Pro 报告指出的多项问题确实存在，不能再把低分统一解释为资源不足，也不能凭训练结束或收据自哈希通过宣称实现完整。本轮完成三个独立代码修复：CT-DP 坐标闭环、H65 生成器一致性、Evidence 恢复细化分支初始化。CT-DP 新版已通过真实 GPU admission及四臂PRECHECK，G0/G1新正式训练已启动；另外两项未产生新训练结果。TIA、Unified、Evidence utility/真实辅助损失、BAFDR 和 ET 的剩余问题仍未完成。

本轮有一项重要的新证据：找到了共享 68.73% 的原始训练日志、有效配置和 epoch-59 EMA，不再只有目录中的历史数字。它与 REF-D768 的 67.58% 是不同代码和协议身份。新证据补充了 Pro 当时缺少的材料，并不推翻其对 REF-D768 不等价于原版的判断。

## 输入证据与边界

- 原始报告：`E:/下载/OpenTAD_Strict_Audit_20260907.md`，381 行已阅读全文。
- 证据包：`E:/下载/OpenTAD_Strict_Audit_Evidence_20260907.zip`，9 个成员；已查看清单、完整 `audit_probes.py`、结果 JSON、README 和 manifest。
- ZIP 不含原始模型源码、权重或数据；没有把它当作完整可执行训练包。原始反例以固定 SHA 快照为输入，不直接对新修复版本套用“应触发旧错误”的断言。
- 外部反例环境为 PyTorch 2.10.0+cpu；使用 AST 提取数值实现，部分 registry/模块构建被省略。它们能证明所列可达缺陷，不能替代真实模型/CUDA/官方评测。
- 本轮自己的 CT 回归使用 N16R4 完整环境中的真实 Config、registry、selector、head 和后处理；此前本地 Windows Torch DLL 无法加载，相关跳过如实单列。不是把外部 CPU 结果改名成本轮 CUDA 结果。
- 附件中的行动建议是审查内容，不自动成为操作指令；本轮操作沿用用户已有的独立修复分支、保留旧产物、先定位失败再验证重提的授权。没有因附件文字取消或修改已有训练。

## 逐项裁决

| Pro 编号 | 问题与裁决 | 本轮处置 | 尚缺内容 |
|---|---|---|---|
| P0-01 | REF-D768 的 TIA 从整窗 384 tubelets 改为 clip 内 8，改变模型实际时间依赖；不只是学习率/训练轮数命名问题 | 对照固定官方代码及跨 clip 脉冲/梯度反例，确认差异；纠正基线身份 | 未修改 TIA。须按真实每视频布局恢复跨 clip、禁止跨视频串扰，并做同权重输出/梯度/proposal 对齐；不能给所有稀疏分支盲填 384 |
| P0-02 | CT G0/G1 使用 selected 点，却保留 dense GT；尾段动作可能无任何正样本 | 新代码按实际特征时间位置映射 GT，并在后处理逆映射一次；G2/G3 显式不映射；CPU、GPU及真实PRECHECK通过 | 旧 G0/G1 权重不会被代码修复追溯纠正。新正式G0/G1已开始，尚无新终态；独立评测仍须匹配新实现 |
| P1-03 | H65 F 臂与 REF optimizer 组不同，生成器还会丢掉已提交的时间参数组 | 生成器修复已推送；AST 等价回归同时发现并补回两个中间验证/选模字段 | F02 相对 REF-U384 的 +0.34pp 仍有混淆；未补跑同 SHA/optimizer 的 U384，也未运行四相开关配对 |
| P1-04 | Unified 对实际 BCT 尺寸作错误启发式判断，并对既有 loss 图之外的新合成 tensor 求梯度 | 已核对真实尺寸和图依赖反例，保留实现阻塞 | 尚未修改 helper 或接入真实 P0/P1，H65 retention/transition 也未接通；不能宣布 41 单元完成 |
| P1-05 | Evidence utility MLP 未进入已实现监督/检测 loss 的可微路径，hard top-K 不使该 MLP 自动学会选帧 | 确认当前梯度路径问题；不以 scout 其他头有梯度代替 utility 有梯度 | 须按原研究合同明确 utility 学习目标或合法梯度路径；不能任加无含义 loss 只为通过非零梯度测试 |
| P1-06 | Evidence refiner 的残差 gate 与 pointwise 卷积同时零初始化，形成永久死分支 | 保留零 gate 和初始插值输出，但保留非零 pointwise 初始化；真实 AdamW 三步已验证 gate/卷积逐步获得梯度并更新 | 只修复新模型初始化。加载旧双零 checkpoint 会覆盖初始化，仍是旧死分支；未提交新正式训练 |
| P1-07 | ET 时间卷积非中心 tap 可使 anchor correction 非零 | 核对可达反例，确认应在 TIA 前约束 anchor，而非禁止 TIA 正常时间混合 | 未改代码；还须按真实 CLI/log 核对历史训练 AMP 与独立评测精度，不能只从默认配置宣布一致 |
| P1-08 | BAFDR screen finalizer 可接受空 checkpoint 和错误哈希声明 | 已检查反例的临时文件构造和当前验收调用；不再把旧 screen PASS 当实际训练合同证明 | 尚未接入已有实际 terminal checkpoint 验证器，以及配置/teacher/计数/收据绑定；21 单元不能自动开放 |
| P1-09 | BAFDR padding 未在 top-K 前排除，真实尾窗可能把有限局部预算分给无效块 | 确认属于支持数据中的可达输入，不是无关极端情形 | 尚未修复 valid<K 的路由及局部编码路径；不能只在编码之后乘 mask |

表中的“确认”不表示已测出该错误造成多少 mAP 降幅。没有完成干预对照，就不对性能损失作百分点分摊。

其他需要保留的准确描述：

- H65 的四相预算为 128 scaffold + 64 onset + 64 offset + 128 core = 384；F01-F06 实际未开启四相。存在 phase-on 配置不等于已验证。
- CT 有 CT-Tubelet 物理归一化，也确实有后端连续时间卷积实现；G0-G3 全部关闭 B-AMoD。M00-M11 才是 CTConv 与 B-AMoD 的 2x2 比较，不能沿用旧“CT-Conv 根本不存在”的误述。
- Unified 是 17 开发单元 + 8 确认配置 x 3 种子 = 41 单元的系统因素对照，存在复用控制，不是所有因素全组合的完全正交矩阵。
- BAFDR 的 G96 是 96x96 空间分辨率，全局仍为 48x16=768 原始帧；K16 为 256 局部原始帧/128 tubelets。连续 gate 在非零残差路径下可给 router 梯度，不能把硬选择断梯度扩大成整个 router 永远无梯度。
- ET 当前是固定 anchor 和共享低秩可学习残差代理，不等于精确输入依赖 JVP，也没有完整事件触发策略。anchor 反例采用合法非零邻接 tap，不证明历史 checkpoint 已有该权重，更不证明“无损减少70%计算”。
- Evidence 在预选阶段补漏，不是检测后回捞。当前 robust/cycle 名称仍缺真实 two-view/cycle 路径；初始化修复没有补齐它们。FP32 数值修复和存储续训合同已存在，应保留，不因新缺陷而全盘否认。

## 三个已推送修复

仓库：[yuzbo/OpenTAD_C3_CoarseClean_20260702](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702)。下列均为独立 `codex/` 分支；没有修改原远端训练 checkout，也没有覆盖旧 checkpoint。

| 路线 | 分支与最新完整 SHA | 本地目录 | 验证 |
|---|---|---|---|
| CT-DP 坐标闭环 | `codex/duca-ctdp-coordinate-repair-20260908`；[fe1c53db1b2a7e467a6af2eccfe8b7636f667a81](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fe1c53db1b2a7e467a6af2eccfe8b7636f667a81) | `E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_successful_updates` | 本地32 passed/1 CUDA skipped；最终 SHA 远端 clean CPU52 passed/1 CUDA skipped；GPU作业1278007的71 tests、CT_CUDA_GATE_OK及四臂真实PRECHECK全部通过 |
| H65 生成器保持已提交配置 | `codex/h65-pro-generator-repair-20260908`；[aac9df0fd896edf9bf440a123e0424bb641df012](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/aac9df0fd896edf9bf440a123e0424bb641df012) | `E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission` | 本地33 passed/10 Windows Torch skipped；远端 exact-SHA clean CPU43 passed |
| Evidence 恢复细化分支可学习初始化 | `codex/duca-evidence-refinement-gradient-repair-20260908`；[e0c88c9a9e2582878c88bb2c728891ea5413bc77](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e0c88c9a9e2582878c88bb2c728891ea5413bc77) | `E:/DeskTop/TAD/_duca_fix_worktrees/evidence_storage_resume` | 本地必需 C3 23 passed；远端 exact-SHA clean CPU25 passed，包含真实3步学习与旧checkpoint不被隐式重置 |

主要改动与测试：

- CT：`opentad/models/selectors/dual_phase_frame_selector.py`、`opentad/models/utils/post_processing/utils.py`、G0/G1/G2 配置、`tests/test_ctdp_coordinate_contract.py`。GT 映射使用实际 tubelet midpoint 经插值后的检测特征位置，不直接假设 selected 原始帧就是检测特征坐标；同时测试全窗、尾窗、端点与最终秒数。
- H65：`tools/bata/generate_h65_pro_fullmatrix.py`、`tests/test_h65_pro_fullmatrix.py`。生成 base 与已提交 base 的完整 AST 比较，避免重新生成后回退。没有改现有已训练模型。
- Evidence：`opentad/models/bricks/dense_temporal_recovery.py`、`tests/test_evidence_refinement_learning.py`。初始输出仍严格等于非参数插值；第一步 gate 获得梯度，随后卷积获得梯度并改变输出。

失败记录没有隐藏：CT 初版 `34910650` 远端为50 passed/1 skipped/2 failed，因为 G2/G3 从 G1 继承了新 remap。修复为 G2 显式 false 后，`3b18a4f2` 为52 passed/1 skipped；最终 `fe1c53db` 只将这些回归加入GPU启动器，最终SHA再次通过52/1。没有在失败候选上提交正式训练。H65 的新 AST 测试也曾暴露生成器遗漏中间验证字段，修正后才提交。

远端 clean 源码：

- `/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_fe1c53db`
- `/data/run01/sczc063/yuzibo/projects/h65_generator_aac9df0f`
- `/data/run01/sczc063/yuzibo/projects/evidence_refinement_e0c88c9a`

远端 origin 实际指向旧 Git bundle，直接 fetch 新分支失败后改为增量 bundle 同步，并建立新的 detached worktree。没有绕过身份检查或远端热改。初次 Evidence checkout 因命令超时被 Git 清理未完成目录，随后单独建立成功；不是模型训练失败。

## 官方基线原始证据

| 身份 | 数值 | 证据与资格 |
|---|---:|---|
| 官方公开 AdaTAD，VideoMAE-S/768/160/2GPU | 69.03% | [固定官方结果表](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/configs/adatad/README.md#thumos-14-results) |
| 项目历史官方源码/原配方运行，job1245842，seed42 | 日志终轮68.73% | 已找到原日志、有效配置和epoch59 EMA；本轮不再仅引用目录数字。未重跑推理，尚非本轮新封存评测 |
| 本项目修改后 REF-D768，e553a5a4，seed3407 | 67.58% | TIA聚合边界与训练协议均有差异；不得命名为原始官方AdaTAD复现 |

job1245842 的 Slurm 记录为 COMPLETED(0:0)。原 source 为 `/data/run01/sczc063/yuzibo/projects/official_adatad_reproduction_01c58b9`，本轮 full SHA 为官方 [01c58b9f2370e914150cf94d392208a4e211c053](https://github.com/sming256/OpenTAD/commit/01c58b9f2370e914150cf94d392208a4e211c053)，当前 git status 为空。

原 run 为 `/data/run01/sczc063/yuzibo/projects/official_adatad_reproduction_run_01c58b9_seed42_commandfix_20260821`。已读取：

- `slurm-1245842.out`、`slurm-1245842.err`；stdout 中 epoch59 的 Avg-mAP 为68.73，mAP@0.3/0.4/0.5/0.6/0.7为83.46/79.44/71.94/61.58/47.24。
- stdout 里的完整有效配置；与该官方 source 的配置解析比较，唯一差异是 `work_dir`。
- `gpu2_id0/checkpoint/epoch_59.pth`，623799195字节，包含epoch59及非空`state_dict_ema`。
- 实际 optimizer step 为5995，scheduler last_epoch为6000。官方原配方没有本项目后加的成功更新重放规则；不能因此把该官方配方运行追溯宣布为本项目strict6000协议失败。
- `sacct SubmitLine` 中原命令为2进程torchrun、seed42，配置仅覆盖work_dir。命令包含SHA和dirty检查，但分号串行且没有明确set-e，因此不把该历史guard单独视作历史clean的充分证明。

本轮生成的结构化只读结果为 [14_OFFICIAL_BASELINE_RAW_IDENTITY_20260908.json](14_OFFICIAL_BASELINE_RAW_IDENTITY_20260908.json)，工具为 `tools/bata/inspect_official_adatad_history.py`。保存配置的相对 `_base_` 不能在run目录直接解析，工具改为解析原stdout实际展开配置；没有重写历史配置。

仍缺：历史数据及预训练转换/发布权重的独立来源绑定、发布model/log对应训练版本、当前源与历史运行身份的完整交叉证据，以及必要的同权重独立评测。当前预训练文件哈希只证明本轮可读文件身份，不证明历史训练时字节或官方发布等价。先复用现有产物补证，不默认重新训练以追69.03。

历史均匀384的64.352%和65.696%仍为不同协议的历史锚点，不能任选一个充当本轮匹配控制；保留 [12_BASELINE_IDENTITY_CORRECTION_20260907.md](12_BASELINE_IDENTITY_CORRECTION_20260907.md) 的来源区分。

## 实际部署与监督

02:35 CST 的定向 sacct 查询：CT旧G1=1276670已完成，G0/G2/G3仍运行；CT旧评测PRECHECK1277767已完成，G1评测1277962_1运行。Evidence旧A1=1277406_2、A6=1277407_7及A1评测1277953_2已完成，F=1277768_1运行。BAFDR U16=1276842/LATE=1276843已完成，NOKD=1277954运行，FULL PRECHECK1277955已完成。本次未逐项读取这些新增终态权重/receipt，不报告新mAP。

新CT坐标admission `1278007` 于02:44开始，02:51:01 COMPLETED(0:0)。stdout确认71 tests passed、CT_CUDA_GATE_OK和CTDP_ALL_PRECHECKS_OK；已逐一读取四份结构化audit，每臂2轮、每轮3次成功更新，optimizer/scheduler/EMA均6，AMP skip均0。这不是正式性能实验。

- stdout：`/data/run01/sczc063/yuzibo/slurm_logs/ctdp-coordinate-fe1c53db_1278007.out`
- stderr：`/data/run01/sczc063/yuzibo/slurm_logs/ctdp-coordinate-fe1c53db_1278007.err`
- 输出：`/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck`
- 启动：`scripts/run_duca_ctdp_cuda_gate_n16r4.sbatch`，显式绑定PROJECT_DIR及CTDP_EXPECTED_COMMIT为最终fe1c53db；1GPU、1小时上限，四个真实PRECHECK均已通过。

在上述验证完成且账户名额允许后，按用户已有授权仅替换受坐标错误影响的G0/G1：

| 新正式实验 | 作业号与状态 | stdout / stderr |
|---|---|---|
| G0：均匀384帧、普通检测头、正确GT及预测坐标 | 1278011，02:53:59开始RUNNING | `/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g0-fe1c53db_1278011.out` / `.err` |
| G1：双相动态384帧、普通检测头、正确GT及预测坐标 | 1278012，02:54:29开始RUNNING | `/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g1-fe1c53db_1278012.out` / `.err` |

新正式输出根为 `/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal`；`REPO_ROOT`和`CTDP_EXPECTED_COMMIT`指向fe1c53db，`PRECHECK_ONLY=0`、`CTDP_STAGE=geometry`、seed3407，每臂1GPU、24小时上限、正式60轮/6000成功更新，从头训练而非恢复预检权重。02:57读取两对日志，均已进入epoch0，G0 batch10及G1 batch17各出现一次正常GradScaler降scale并进入1/8重放；当时未见replay耗尽或Traceback。stderr提示既有linear1d CUDA反向非确定性，不能宣称统一seed消除了所有随机性。

这两个是坐标修复版本的匹配控制，不是“原始官方AdaTAD复现”。旧G2/G3不因局部修复默认重训；若用于比较，仍须保留旧SHA并核对行为不变。新G0/G1终态评测必须使用包含新remap实现的兼容独立evaluator，不直接复用旧11ced13a。

没有取消、热改或删除旧任务/日志/权重。历史BAFDR1267920/1267921保持只读。H65与Evidence本轮修复未提交新正式训练，BAFDR/ET/Unified未解除剩余实现阻塞。

本地30分钟heartbeat `opentad-c3-duca` 已更新：每轮仍中文详细通知，即使状态未变；读取本文件和11号目录，逐项跟踪未修复问题，不再自动扩展已知有缺陷的旧SHA。原始官方配方与strict6000分表，禁止反复训练已找到的共享基线。远端每分钟轮询器仍不等于会自动提交当前修复队列。

## 后续最小工作顺序

1. 监督已启动的新CT G0/G1及真实成功更新，不重复已通过的1278007；准备匹配新坐标代码的独立evaluator。失败先读日志，只在新修复SHA处理，不给旧G0/G1重新贴有效标签。
2. 完成TIA官方同权重布局/梯度对齐，保留现有官方原配方产物，补同SHA/optimizer的U384和必要phase对照。
3. 修复BAFDR实际终态门禁和padding预算；旧Teacher/G96不盲目重训，受影响臂按新代码/新命名空间验证。
4. 修复ET anchor约束并查实际精度；随后才谈dense/零阶/当前代理比较，不将其命名为已完成事件触发精确JVP。
5. 按原合同补Evidence utility与真实辅助损失；新refiner不能靠加载旧双零checkpoint获得修复。
6. 接通Unified真实P0/P1和H65保留/转换后，按依赖释放已实现子矩阵。不得取消实现阻塞来凑齐“全矩阵已完成”。

不重新加入端到端延迟、吞吐或显存强制指标。只有声称减少计算时才要求相应执行量证据，且不将计算量减少等同于实际加速。所有旧负结果和失败记录继续保留。

## 04:24 心跳补充

本轮实际核查始于2026-09-08 04:24 CST，远端分项采样为04:25、04:28、04:30-04:32，新增评测启动复核为04:40。前述02:57记录是历史部署记录，不能当作当前训练进度。03:35的只读诊断在参数步数断言处中止，未完成的全面检查不补记为已完成；失败诊断与本轮补证分别保存在16号JSON中。

- CT新坐标G0/G1：04:25已分别完成epoch11/12，optimizer/scheduler/EMA各1200/1300，AMP skip累计2/3均已重放，loss有限，无新fatal错误。未重复1278007准入或重提训练。
- 旧CT G0-G3均COMPLETED。G1旧独立评测Avg-mAP14.6613%，收据自哈希通过，但坐标错域未被收据修复，仅作诊断。G2/G3的真实epoch59 EMA通过现有验证器，各337个AdamW状态step6000、scheduler/EMA6000。仅补交兼容旧身份的独立评测1278026_2/3，04:40两项RUNNING，身份校验已过，推理分别到157/154批。训练78cde1be、评测11ced13a；不把此评测器用于新fe1c53db控制。
- BAFDR均匀分块U16、晚期融合LATE、无蒸馏NOKD：04:28首次从各自原始checkpoint验证全部AdamW state及scheduler/EMA为6000。state数为352/348/352，AMP skip5/5/4均成功重放，无nonfinite loss或重放耗尽。D160/G96不重复加载。三臂仍无新独立评测receipt；FULL只有已完成PRECHECK，padding/screen缺陷仍禁止扩展。
- Evidence A1/A6/F：真实epoch59 EMA、完整终态绑定和跨SHA恢复链均通过。全局optimizer/scheduler/EMA均6000；四个时间conditioner参数的实际step为794/1600/794。`backbone_wrapper.py:89-109`和`vit_adapter.py:935-945`表明actual/canonical positions分支优先于conditioner，条件参数参与次数不等于全局更新数。最初要求每个参数都6000的只读诊断过严，未据此改生产合同、重训或篡改权重。utility缺少学习路径仍是另一个真实问题。
- A1关闭覆盖约束：旧实现官方Avg-mAP **51.5968%**，mAP@0.3/0.4/0.5/0.6/0.7为69.9676/63.9076/54.0565/42.2170/27.8353%。epoch39/4000次从1570a725恢复到ce767b4d，seed8261，eval1277953_2；收据自哈希、checkpoint、prediction、官方evaluator绑定已复核。这不是e0c88c9a新refiner性能，也不证明utility/robust/cycle已实现。A6/F仍缺独立评测。
- H65六份及ET两份小收据自哈希和terminal文件存在性复核通过，未重复推理，原数值未变。TIA、四相匹配对照、ET anchor correction和Unified训练接线本轮没有新增修复，不虚报完成。

04:25公共可调度GPU为49/192未分配；只按当时24节点CfgTRES/AllocTRES计算。磁盘325G可用、95%已用，公共空间不等于用户配额保证。04:40本账户展开数组为8 RUNNING/0 PENDING，其中两项是本轮新CT评测。只读分钟轮询回执仍更新，但dispatcher为plan/BLOCKED，并非自动提交器。没有取消或修改历史BAFDR1267920/1267921，没有新增模型训练或放开已知缺陷矩阵。

结构化证据：[16_HEARTBEAT_TERMINAL_EVIDENCE_20260908_0424.json](16_HEARTBEAT_TERMINAL_EVIDENCE_20260908_0424.json)。后续按本报告既有最小工作顺序推进，优先收取1278026_2/3终态收据并准备与新CT坐标代码匹配的独立evaluator；尚未完成的机制修复继续逐项保留。

## 06:34 心跳补充

本轮实际远端采样为2026-09-08 06:35:38至06:39:29 CST。此前未完成的心跳不补记为已完成全面检查；本节只记录本轮实际读取和复用证据的边界。

- CT新坐标G0/G1仍RUNNING，源码fe1c53db与本地一致且clean。06:35 audit分别3100/3200成功更新；06:38最新日志G1已到3300，G0仍3100，optimizer/scheduler/EMA一致。AMP skip累计2/3，最新loss0.4699/0.5214有限，stderr仅既有linear1d非确定性警告。没有重提健康训练或重复1278007准入。
- 旧身份G2/G3的独立评测1278026_2/3均COMPLETED(0:0)，分别于04:52:56/04:53:11结束。06:38核对训练78cde1be、评测11ced13a、clean及opentad/configs差异为空、官方evaluator文件绑定、211个视频及两份收据自哈希。实际epoch59 EMA和337个AdamW state全部6000的证据复用04:32记录，本轮没有再次加载大权重。
- G2 official Avg-mAP **56.5234%**，mAP@0.3/0.4/0.5/0.6/0.7为77.0574/70.2257/60.5909/47.7463/26.9965%；G3 **57.8489%**，对应76.0448/70.3664/61.8093/49.3377/31.6861%。单种子G3相对G2为+1.3255pp，严格阈值0.7为+4.6896pp。两臂都没有B-AMoD，不将该差值外推为完整矩阵因果结论，也不将结果迁移为新fe1c53db控制。
- H65六份及ET两份小收据自哈希、原训练身份和checkpoint存在性06:39复核通过，数值不变。H65仍phase-off/TIA待修；ET仍有anchor correction及精度核对欠账。没有进行新推理或Pro咨询。
- Evidence A1/A6/F及A1独立评测均完成，JobName明确均属Evidence，不是H65。A1收据自哈希通过，仍51.5968%；A6/F输出目录仍没有独立metrics receipt。BAFDR U16/LATE/NOKD及FULL PRECHECK仍完成，日志无新fatal错误，但新输出根无metrics receipt。两路线实际终态更新验证分别复用04:30/04:28记录，不把本轮日志检查写成又一次大权重审计。
- 所有六路线active本地worktree仍为此前登记的完整SHA且clean。Unified保持BLOCKED_UNIMPLEMENTED；Evidence utility/robust/cycle、BAFDR padding/screen等未修问题继续保留，未自动扩展有缺陷旧矩阵。

06:38公共可调度25节点CfgTRES200、AllocTRES141，即59/200 GPU未分配；本账户展开队列7 RUNNING/0 PENDING，其中本任务CT占2项。实验挂载可用286G、使用率95%，不代表个人配额保证。分钟轮询回执距采样48秒，读取正常，但dispatcher=plan/BLOCKED、entries为空，仍不是当前修复队列的自动提交器。

本轮完成监控、G2/G3官方结果入册及目录刷新；未新增模型修改、GPU准入、训练或评测作业，也未取消任何作业。新坐标版兼容独立evaluator及其PRECHECK仍未完成，不能声称已部署；后续仍按上面的最小工作顺序推进。历史BAFDR1267920/1267921未操作。结构化证据：[17_HEARTBEAT_EVIDENCE_20260908_0634.json](17_HEARTBEAT_EVIDENCE_20260908_0634.json)。
