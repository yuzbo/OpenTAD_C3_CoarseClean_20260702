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

## 07:57 心跳补充

本轮触发为2026-09-08 07:57:27 CST，六路线实际监控采样为07:59至08:01。此前07:20只完成部分只读检查，未完成目录推送；本节不将其补记为已完成心跳。

- CT新G0/G1仍RUNNING，08:01落盘audit均为epoch40、optimizer/scheduler/EMA各4100，AMP skip均3。两个作业的epoch41最后训练批次已结束，loss分别0.4476/0.4852；stderr评测分别到161/374批且持续增长。训练器在评测后刷新audit，因此不是训练挂起，也不能把尚未落盘的更新数当已验证计数。没有epoch59或新最终性能。
- H65六份、ET两份及Evidence A1小收据的mtime/数值未变，复用此前身份与自哈希验证，不重复加载大权重。Evidence A1/A6/F及A1评测、BAFDR U16/LATE/NOKD及FULL PRECHECK均COMPLETED(0:0)，没有新fatal。A6/F和BAFDR仍缺新的独立性能收据；剩余机制问题保持原裁决，Unified仍BLOCKED_UNIMPLEMENTED。
- 已完成新坐标兼容评测入口，分支`codex/duca-ctdp-coordinate-terminal-eval-20260908`，完整SHA为[073832744dd171dee4119dc29d1bb12a057e84f4](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/073832744dd171dee4119dc29d1bb12a057e84f4)。基于fe1c53db，`opentad/`与`configs/`差异为空，拒绝旧78cde1be checkpoint；旧G2/G3及11ced13a身份不改写。没有修改正在训练的源码。
- 本地轻量测试45 passed/1 CUDA skipped；坐标测试收集因已知Windows Torch c10.dll错误失败，已保留记录。远端独立clean checkout `/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_eval_07383274` 于08:09完成50 passed/1 CUDA skipped，包含真实坐标回归，编译通过。首个同步命令在checkout完成后120秒超时，未冒称测试通过；后续复用该clean checkout单独验证成功，没有重建或热改源码。
- 08:11:23提交唯一单GPU推理PRECHECK `1278042`，复用1278007已有四臂epoch1/6-update checkpoint，每臂只推理一批、不开指标、不重复训练准入。08:13:01确认RUNNING，31个GPU测试及G0 checkpoint/EMA/真实6更新绑定、单批推理均已通过。之后终态查询120秒无输出，只读重连也未返回；截至本节不能确认其余三臂完成，不能称全PRECHECK通过。旧作业、日志、checkpoint均保留，不重提该作业。
- 新评测日志：`/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/slurm_logs/precheck_1278042.out`及`.err`。后续先收取同job终态；成功后等待新G0/G1 epoch59/6000真实更新，再按07383274和新命名空间提交独立正式评测，不使用旧11ced13a。

08:01公共25个可调度节点CfgTRES200/AllocTRES132，未分配68GPU；本账户7 RUNNING/0 PENDING，此样本早于新增1278042。08:11磁盘263G可用、96%使用，公共空间与GPU余量不保证本账户配额。07:59分钟轮询仍plan/BLOCKED且entries为空，只能称只读轮询；后续SSH不可达不推导监督器或Slurm已退出。目录查询已加45秒总超时，将真实超时标UNAVAILABLE，避免心跳无限等待。

本轮没有新正式训练、正式性能评测、作业取消或Pro咨询；仅新增兼容评测代码与一次推理PRECHECK。六路线未修复机制不因这项接线获得验收，历史BAFDR1267920/1267921未操作。证据见[18_HEARTBEAT_EVIDENCE_20260908_0757.json](18_HEARTBEAT_EVIDENCE_20260908_0757.json)。


## 09:31 心跳补充

本轮触发为2026-09-08 09:31:29 CST，实际远端采样09:32至09:37。09:32:19 SSH读取恢复，既有任务未因先前查询超时而失败；没有把其他未执行触发补记为已检查。

- CT新坐标G0/G1仍RUNNING，09:35:43落盘audit分别为epoch44/46，optimizer/scheduler/EMA各4500/4700，AMP skip均3并已重放。G0已结束epoch45训练并在评测，G1到epoch47 batch50，最新loss分别0.4519/0.4641有限。两臂均无epoch59，不报告新正式性能；audit在评测后落盘，不能将间隔误判为挂起。
- 兼容评测器07383274的既有PRECHECK1278042已于08:14:51 COMPLETED(0:0)。本轮收取31个GPU测试通过、G0-G3四份epoch1 EMA/真实6更新/337 optimizer states绑定、四次单批推理PASS及CTDP_ALL_EVAL_PRECHECKS_OK。没有重跑任何预检。远端训练fe1c53db及评测07383274仍exact/clean，分别对opentad与configs执行git diff为空。
- 07383274现已具备评测启动证据，但正式评测仍等待新G0/G1终态epoch59及实际6000更新。旧G2/G3继续保留78cde1be/11ced13a及56.5234%/57.8489%原身份；不将它们重新命名为新坐标版结果。
- H65六份小收据及ET两份小收据mtime/数值未变；H65最高64.2265%，仍phase-off且TIA未修。ET关闭/开启为62.0768%/54.8096%，anchor correction与精度审计未完成。复用此前自哈希及大checkpoint验证，没有新推理。
- Evidence A1/A6/F训练与A1评测均COMPLETED(0:0)。初次路径查询未命中后已补读真实slurm_logs，Training/Testing Over及stderr未见新fatal；配置里的fail_on_amp_replay_exhaustion=True不是失败事件。A1仍51.5968%，A6/F仍缺独立metrics。utility/robust/cycle问题仍在，不能把e0c88c9a新初始化赋给旧checkpoint。
- BAFDR U16/LATE/NOKD及FULL PRECHECK均COMPLETED(0:0)，前三者终轮日志为epoch59/update6000，实际权重验证复用04:28记录。当前710ce8a6输出仍无metrics.json，padding/screen问题不因预检通过获得豁免。FULL正式与21单元未开放。历史1267920/1267921仅只读查询，未取消、修改或重提。
- 六条active本地worktree身份均与目录一致且clean；Unified保持BLOCKED_UNIMPLEMENTED，真实P0/P1与H65保留/转换本轮没有新增实现。官方原配方1245842的68.73%仍为此前核查的日志值，预训练/数据来源与独立复评未完成，不用修改版REF67.58%替代。

09:33公共可调度25节点CfgTRES200/AllocTRES139，即61/200 GPU未分配；本用户6 RUNNING/0 PENDING，其中本任务CT占2项。磁盘245G可用、96%已用，公共余量不等于账户配额保证。分钟回执在09:33:35读取时距生成60秒，只读轮询正常；dispatcher=plan/BLOCKED、entries为空，不是自动提交正常。

本轮完成远端状态恢复读取、既有CT预检终态收取及目录更新；没有新的模型修复、训练、正式评测、作业取消或Pro咨询。剩余工作继续按前述最小顺序执行，不能把本轮监控当作六路线验收。证据见[19_HEARTBEAT_EVIDENCE_20260908_0931.json](19_HEARTBEAT_EVIDENCE_20260908_0931.json)。


## 10:15 心跳补充

触发时间为2026-09-08 10:14:59.973 CST，按10:15心跳记录；实际远端采样10:16:40至10:18:59。其他未执行的触发不补记。本轮没有新运行失败或新最终mAP。

- CT新G0/G1仍RUNNING，10:18落盘audit分别epoch46/48，optimizer/scheduler/EMA各4700/4900，AMP skip均3并已重放，max retry仍1。两臂最新训练分别为epoch47/49 batch99，loss0.4236/0.4694有限；stderr显示训练内评测进度仍增长。两者均无epoch59，评测后才写audit，不把暂未刷新计数当训练挂起。
- 远端fe1c53db训练与07383274评测源码仍exact/clean。复用已经通过的1278007和1278042，不重复测试或预检；07383274正式评测仍等待新终态及实际6000更新。旧G2/G3保持原SHA与56.5234%/57.8489%结果，不迁移为新坐标版。
- H65六份和ET两份小收据mtime/数值未变；H65最高64.2265%，F臂仍phase-off/TIA待修。ET关闭/开启62.0768%/54.8096%，anchor correction与精度问题未解决。没有重复加载大checkpoint、复算已认证hash或推理。
- Evidence A1/A6/F训练及A1评测仍COMPLETED(0:0)，stdout/stderr无新fatal；A1仍51.5968%，A6/F独立metrics继续缺失。BAFDR U16/LATE/NOKD及FULL PRECHECK仍COMPLETED(0:0)，前三终轮日志6000，当前输出仍无metrics.json。复用此前实际权重验证，utility/真实辅助损失和padding/screen缺陷仍在；不扩展已知错误的旧矩阵。
- 各active本地源码身份由目录生成器刷新；H65、Evidence、BAFDR、ET与Unified独立读取均为既有SHA且clean。Unified仍BLOCKED_UNIMPLEMENTED。本轮无新机制实现或Pro咨询；官方原配方68.73%的来源与独立复评待补，不用REF67.58%替代。

10:16公共可调度25节点CfgTRES200/AllocTRES159，未分配41/200 GPU；本用户7 RUNNING/0 PENDING，本任务CT占2项。磁盘238G可用、96%已用。10:18:59分钟回执年龄47秒，只读轮询正常，但dispatcher=plan/BLOCKED、entries为空，不是自动提交正常。

本轮只执行监控及目录刷新，没有新训练、评测、取消或重复准入。下一步仍是等待CT终态后提交兼容独立评测，并按原设计逐项补齐剩余实现，而不是降低验收口径。结构化证据：[20_HEARTBEAT_EVIDENCE_20260908_1015.json](20_HEARTBEAT_EVIDENCE_20260908_1015.json)。

## 11:43 心跳补充

本轮触发为2026-09-08 11:43:31.103 CST，实际远端采样11:45:07至11:48:56。11:04曾进行部分只读检查，11:12答复了实际运行时间，但未完成目录推送；本节不将其补记为已完成的完整心跳。没有新最终性能或新的运行失败。

- CT新G0/G1均RUNNING，11:47的运行时长分别08:53:51/08:53:21。落盘audit为epoch51/52，optimizer/scheduler/EMA分别5200/5300，AMP skip均3、max retry均1。G0到epoch52 batch50、loss0.3984；G1结束epoch53 batch99、loss0.4362，stderr评测到224批。均无epoch59；不把中期分数或评测后的audit落盘间隔当作终态/挂起。
- 近期每轮训练约6-7分钟、每两轮评测约26分钟，合计每两轮39-40分钟；不能把含评测的33分钟错误套到每一轮。条件估计G0于14:10-14:45、G1于13:50-14:20结束训练，独立评测另需30-60分钟及排队，不承诺固定截止时间。
- fe1c53db训练与07383274评测远端HEAD仍exact/clean。既有1278007和1278042通过证据复用，不重复准入；按臂等待epoch59 EMA及真实6000后，再提交07383274独立终态评测。旧G2/G3沿用78cde1be/11ced13a的56.5234%/57.8489%结果，本轮未重新推理。
- H65六份与ET两份小收据mtime、数值未变。H65最高64.2265%，F01-F06仍phase-off、TIA及匹配控制未修；ET关闭/开启62.0768%/54.8096%，anchor correction与实际精度核对未完成。复用已有自哈希与大checkpoint认证。
- Evidence A1/A6/F及A1评测均COMPLETED(0:0)，本轮读取真实stdout/stderr终段；A1直接读取metrics.average_mAP为0.5159681679375281，A6/F仍无独立metrics。BAFDR U16/LATE/NOKD及FULL PRECHECK均COMPLETED(0:0)，前三终轮日志确为epoch59/update6000，新根仍无metrics。实际权重验证复用04:30/04:28；utility/真实robust-cycle和padding/screen缺陷继续阻止有缺陷矩阵扩展。
- 六条active源码身份由目录生成器刷新；其余五路线只读HEAD核查均为登记版本且clean。Unified仍BLOCKED_UNIMPLEMENTED，无新P0/P1或H65保留/转换实现。官方原配方68.73%仍为此前已读原始日志，来源绑定及独立复评待补，不用修改REF67.58%补位。

11:45公共可调度25节点CfgTRES200、AllocTRES173，未分配27/200 GPU；磁盘217G可用、97%已用。11:47展开账户队列为11 RUNNING/1 PENDING，其中本任务CT占2项；1278774为其他未登记任务的JobHeldUser，未解除其hold，不当成本路线的提交失败。公共余量不保证个人配额。11:48:56分钟回执年龄33秒，dispatcher嵌套字段为plan/BLOCKED、entries为空，只读轮询正常不等于自动提交正常。

本轮完成状态/日志/小收据核查和目录更新，没有新模型修复、Slurm提交、取消、重复准入、大权重加载或Pro咨询。历史BAFDR1267920/1267921未操作；所有剩余实现问题继续逐项跟踪。证据：[21_HEARTBEAT_EVIDENCE_20260908_1143.json](21_HEARTBEAT_EVIDENCE_20260908_1143.json)。

## 12:25 心跳补充

本轮触发为2026-09-08 12:25:32.886 CST，实时远端采样为12:27。只记录实际执行的检查；未执行或仅部分完成的历史触发不补记为完整心跳。

- CT新G0/G1仍RUNNING，运行时长09:33:19/09:32:49。落盘audit为epoch53/54，optimizer/scheduler/EMA各5400/5500，完成90.0%/91.7%；AMP skip均3、max retry均1，未新增。G0到epoch54 batch50、loss0.4269；G1已结束epoch55 batch99、loss0.4402，训练内评测到226批。均无epoch59，不报告中期分数为最终性能。
- 远端训练fe1c53db与评测07383274仍exact/clean。最近仍为每两轮约39-40分钟，训练结束估计保持G0 14:10-14:45、G1 13:50-14:20；独立评测另需30-60分钟和排队，不作截止承诺。未重复1278007或1278042，终态实际6000确认前不提前认证结果。
- H65六份及ET两份小收据mtime和数值未变，分别最高64.2265%及关闭/开启62.0768%/54.8096%。H65四相仍未开启，TIA与匹配控制未修；ET anchor correction及实际精度核对仍未完成。旧CT G2/G3沿用原SHA结果，本轮未重新推理。
- Evidence A1/A6/F及A1评测均COMPLETED(0:0)，stdout的Training/Testing Over与stderr已读；A1直接读取metrics.average_mAP仍为0.5159681679375281，A6/F仍无独立metrics。BAFDR U16/LATE/NOKD及FULL PRECHECK仍COMPLETED(0:0)，前三终轮日志为epoch59/update6000，新root仍无metrics。复用04:30/04:28的大权重认证，没有重复加载。
- 六条active本地代码身份由生成器刷新。Unified仍BLOCKED_UNIMPLEMENTED；Evidence utility/真实robust-cycle、BAFDR padding/screen及其他既有问题未修，不扩展已知错误的旧矩阵。原始官方配方68.73%仍是此前核查的日志值，来源绑定与独立复评待补；修改REF67.58%不能替代。

12:27公共可调度25节点CfgTRES200、AllocTRES173，未分配27/200 GPU；账户展开队列7 RUNNING/1 PENDING，本任务CT占2项。其他未登记任务1278774仍JobHeldUser，未操作。磁盘209G可用、97%已用，公共余量不保证个人配额。分钟回执距生成31秒，dispatcher仍plan/BLOCKED、entries为空，只读轮询正常而非自动提交正常。

本轮无新增模型运行失败、正式性能、模型修改、Slurm提交或取消；仅完成监控和目录刷新，没有Pro咨询。历史BAFDR1267920/1267921保持未操作，所有未完成事项继续保留。证据：[22_HEARTBEAT_EVIDENCE_20260908_1225.json](22_HEARTBEAT_EVIDENCE_20260908_1225.json)。

## 13:04 心跳补充

本轮触发为2026-09-08 13:04:03.484 CST，实时远端采样13:05:55至13:06:01。其他未执行或部分完成的触发不补记；本轮没有新增最终性能或运行失败。

- CT新G0/G1仍RUNNING，运行时长10:11:56/10:11:26。落盘audit为epoch55/56，optimizer/scheduler/EMA各5600/5700，完成93.3%/95.0%；AMP skip均3、max retry均1，未增加。G0到epoch56 batch50、loss0.3547；G1结束epoch57 batch99、loss0.4175，训练内评测到214批。均无epoch59，不能据中期日志提前认证终态。
- 训练fe1c53db及独立评测07383274远端HEAD仍exact/clean，复用既有1278007/1278042准入。近两轮周期仍约39-40分钟；训练完成估计维持G1 13:50-14:20、G0 14:10-14:45，独立评测另需30-60分钟与排队。先收取真实epoch59/6000再按臂提交正式评测，不重复训练或预检。
- H65六份与ET两份小收据mtime/数值不变，分别最高64.2265%与关闭/开启62.0768%/54.8096%。H65仍phase-off、TIA/匹配控制待修；ET anchor correction及实际精度核对未完成。旧CT G2/G3仍保留原身份结果，不用新坐标源码重命名旧成绩。
- Evidence A1/A6/F及A1评测均COMPLETED(0:0)，已读真实stdout/stderr终段；A1直接读取metrics.average_mAP仍0.5159681679375281，A6/F缺独立metrics。BAFDR U16/LATE/NOKD及FULL PRECHECK均完成，前三终轮日志6000，新根仍无metrics。既有大checkpoint认证复用，不重复加载；FULL正式训练及有缺陷旧矩阵不扩展。
- 六个active本地身份由生成器刷新；Unified仍BLOCKED_UNIMPLEMENTED。Evidence utility/真实robust-cycle、BAFDR padding/screen及其他已登记缺陷未修复。原版AdaTAD的68.73%仍为已核查历史日志、待来源补证及独立复评；修改版REF不能替代官方锚点。

13:06公共可调度25节点CfgTRES200、AllocTRES169，未分配31/200 GPU；账户展开队列8 RUNNING/1 PENDING，本任务CT占2项。其他任务1278774仍JobHeldUser，未操作。实验盘201G可用、97%已用，公共余量不保证个人配额。分钟回执年龄34秒，dispatcher仍plan/BLOCKED且entries为空，只读轮询正常不等于自动提交正常。

本轮仅完成监控与目录刷新，没有新模型修改、训练/评测提交、取消、重复准入或Pro咨询。历史BAFDR1267920/1267921未操作；全部未完成事项继续保留。证据：[23_HEARTBEAT_EVIDENCE_20260908_1304.json](23_HEARTBEAT_EVIDENCE_20260908_1304.json)。

## 13:41 心跳补充

本轮触发为2026-09-08 13:41:34.049 CST，实时远端采样13:43:19至13:43:24。未执行或部分完成的历史触发不补记为完整检查。

- CT新G0/G1均RUNNING，运行时长10:49:20/10:48:50。落盘audit为epoch57/58，optimizer/scheduler/EMA各5800/5900，完成96.7%/98.3%；AMP skip均3、max retry均1，未增加。
- G0于13:40:27进入epoch58，上个完整epoch57末批loss0.3784。G1已于13:32:49结束最后epoch59的batch99，loss0.4201，正在末轮评测，stderr到186批。两臂epoch_59.pth均尚不存在，因此没有加载终态权重或提前认证6000，也未提交独立终态评测。最后训练批次结束与终态验证通过是不同状态。
- 训练fe1c53db和兼容评测07383274远端仍exact/clean。复用1278007/1278042，不重复准入。按当前速度预计G1约14:00、G0约14:20完成训练；独立评测另需30-60分钟与排队。终态权重及实际计数齐备后按臂提交，不重跑健康训练。
- H65六份和ET两份小收据mtime与数值未变：H65最高64.2265%，ET关闭/开启62.0768%/54.8096%。既有自哈希与大checkpoint认证复用；H65四相未开启、TIA/匹配控制问题及ET anchor correction/精度问题均未修。
- Evidence A1/A6/F与A1评测均COMPLETED(0:0)，真实日志已读，A1 metrics.average_mAP仍0.5159681679375281，A6/F无独立metrics。BAFDR U16/LATE/NOKD和FULL PRECHECK仍完成，前三终轮日志6000，但新root无metrics，FULL正式未运行。utility/真实辅助损失和padding/screen问题不因旧训练完成而获得验收。
- 本地六个active代码身份由生成器刷新；Unified仍BLOCKED_UNIMPLEMENTED，没有新增Taylor/H65运行时接线。原版AdaTAD历史68.73%仍待来源补证及独立复评，不以修改REF67.58%代替。

13:43公共可调度25节点CfgTRES200、AllocTRES174，未分配26/200 GPU；账户展开队列8 RUNNING/1 PENDING，本任务CT占2项。1278774是其他未登记任务的JobHeldUser，未操作。实验盘195G可用、97%已用，公共余量不保证个人配额。分钟回执年龄28秒，dispatcher仍plan/BLOCKED、entries为空，只读轮询正常，不是自动提交正常。

本轮没有新增运行失败、最终性能、模型修改、Slurm提交、取消、重复准入、大权重加载或Pro咨询；仅更新监控证据与目录。历史BAFDR1267920/1267921未操作，所有未完成事项继续保留。证据：[24_HEARTBEAT_EVIDENCE_20260908_1341.json](24_HEARTBEAT_EVIDENCE_20260908_1341.json)。

## 14:28 用户进度查询与终态评测部署

本次为用户询问运行时长后的定向查询与已有部署计划衔接，不补记为完整六路线心跳。远端采样14:28至14:32，其他路线科学状态仍沿用各行原时间戳。

- CT坐标修复fe1c53db的G0训练1278011从02:53:59至14:20:05，耗时11:26:06；G1训练1278012从02:54:29至13:58:46，耗时11:04:17，均COMPLETED(0:0)。终轮stdout/stderr已读，Training Over已出现。
- 14:29分别读取两份新epoch59 EMA，07383274现有checkpoint_counts确认各337个Adam参数状态均step6000，optimizer/scheduler/EMA各6000，AMP skip均3且已重放。训练与评测checkout均exact/clean，模型及配置diff为空。Slurm完成、真实更新验证与独立最终性能明确分开。
- 复用已通过的1278042，没有重跑任何训练准入或PRECHECK。14:30:07在全新formal输出提交独立正式评测1279211_0/1，两项14:30:35均获得GPU；14:32观察到两臂CTDP_EVAL_CHECKPOINT_OK和EMA加载。G0已28批，G1进度刚初始化；无独立metrics，不把训练日志64.27/61.67%作为本次最终结果。
- 预计15:00-15:30取得这两臂独立评测结果，前提是推理和收据生成正常。这不是六路线全部完成的时间：其他待修机制、未部署训练和缺少独立评测的实验尚无可靠总工期。
- 14:31可调度25节点CfgTRES200/AllocTRES183，未分配17；账户7 RUNNING/1 PENDING。实验盘186G可用、97%已用。分钟dispatcher仍plan/BLOCKED；本次两项评测由监督任务提交，不是分钟脚本自动提交。没有取消、远端热改、模型修改或Pro咨询；历史BAFDR1267920/1267921未操作。

证据：[25_RUNTIME_AND_CT_EVAL_DEPLOYMENT_20260908_1430.json](25_RUNTIME_AND_CT_EVAL_DEPLOYMENT_20260908_1430.json)。

## 15:08 心跳补充

本轮真实触发2026-09-08 15:08:05.401 CST，实时远端采样15:10:16至15:12:20。六个active本地HEAD与clean状态已检查，均无新模型提交；不补记未执行的触发。

- 新CT独立评测1279211_0/1均COMPLETED(0:0)，分别14:49:56/14:50:58结束，实际用时19:21/20:23。比此前15:00-15:30估计提前完成；这里只以本轮真实核查时间15:10报告，不冒称14:50已经核查。
- 两份新receipt自哈希重算一致，绑定fe1c53db训练与07383274评测、seed3407、epoch59 EMA、optimizer/scheduler/EMA各6000及211个唯一视频。官方metrics与evaluation_metrics.json逐项相同。337个真实Adam状态的14:29验证复用，当前checkpoint大小/mtime未变，本轮没有重复加载大权重或准入。
- 均匀G0官方Avg-mAP64.272003%，mAP@0.3/0.4/0.5/0.6/0.7为80.078855/74.608671/67.412856/56.579434/42.680198%；动态G1为61.673970%，对应78.616332/72.857573/63.476768/53.503821/39.915357%。G1相对G0 Avg-mAP -2.598032pp、mAP@0.7 -2.764841pp。该单种子结果不支持当前动态选择带来提升，但不证明所有动态路线失败。
- G2/G3的56.5234%/57.8489%保留原78cde1be/11ced13a身份，不能迁移为新SHA结果。geometry四臂均B-AMoD关闭；新G0不是原始官方AdaTAD的复现替代品。官方原配方68.73仍待来源补证和独立复评。
- H65六份旧收据mtime/指标未变，最高64.2265%；F01-F06仍phase-off，TIA/匹配控制未修。ET关闭/开启仍62.0768%/54.8096%，anchor correction及实际精度核对未完成。
- Evidence A1/A6/F训练及A1评测已完成，A1的metrics.average_mAP仍0.5159681679375281，A6/F没有独立收据。utility/robust/cycle缺失仍未修；BAFDR U16/LATE/NOKD实际终态验证复用，stdout末行6000，新metrics仍0份，FULL只有PRECHECK，padding/screen尚未修。Unified793c4f9c仍缺运行时P0/P1、合法one-swap及H65保留/转换。

15:10可见可调度25节点CfgTRES200/AllocTRES200，未分配GPU0/200；账户9 RUNNING/2 PENDING，其他任务1279307为Priority、1278774为JobHeldUser，未操作或纳入六路线。实验盘182G可用、97%已用。分钟回执17秒新鲜，dispatcher仍plan/BLOCKED，只能称只读轮询正常。

本轮完成两份新独立结果认证、六路线监控与目录刷新；没有新模型修复、Slurm提交、取消、重复PRECHECK或Pro咨询。已知实现欠账并未因评测成功而获得验收，历史BAFDR1267920/1267921仍不操作。证据：[26_HEARTBEAT_EVIDENCE_20260908_1508.json](26_HEARTBEAT_EVIDENCE_20260908_1508.json)。

## 15:51 心跳补充

本轮触发2026-09-08 15:51:06.015 CST，实时远端采样15:53:03至15:53:09。此前已认证结果的指标与mtime未变，没有新增模型提交、登记作业运行失败或独立最终性能。

- CT训练1278011/1278012和独立评测1279211_0/1仍COMPLETED(0:0)，真实audit均epoch59/optimizer6000/scheduler6000/EMA6000，checkpoint大小与mtime未变。复用14:29实际337个Adam状态认证与15:10 receipt自哈希认证，不重复大权重、准入或推理。G0/G1仍64.2720%/61.6740%，mAP@0.7为42.6802%/39.9154%。
- H65最高64.2265%，六臂仍phase-off；ET关闭/开启62.0768%/54.8096%。H65时间聚合/匹配控制、ET anchor修正/实际精度核对仍未完成。原始官方AdaTAD68.73仍仅保留原日志身份，不能用修改REF或新CT G0替代。
- Evidence A1/A6/F与A1评测均完成，A1仍51.5968%，A6/F缺独立收据；BAFDR U16/LATE/NOKD终态日志仍6000，输出新metrics为0，FULL只有预检。utility/真实辅助损失、padding/真实终态screen及Unified真实P0/P1/H65接线缺项继续保留，不扩已知缺陷旧SHA矩阵。

本轮为落实已有Pro授权，检查指定Computer Use入口：读取已安装Computer Use技能与运行说明后，node_repl初始化@oai/sky失败，重置并重试仍返回 `failed to write kernel assets: 系统找不到指定的路径。 (os error 3)`；另一Computer Use会话的浏览器库存读取也同样失败。阻塞发生在运行环境初始化，尚未读取ixBrowser窗口、登录状态或所选模型。未发送材料、未完成Pro咨询、未改用API/其他浏览器或其他模型；不能把此错误说成账号未登录或Pro服务不可用。

15:53可见可调度25节点CfgTRES200/AllocTRES199，剩余1/200 GPU；账户5 RUNNING/1 PENDING共6项，其他任务1278774仍JobHeldUser且未操作。实验盘179G可用、97%已用。分钟回执年龄32秒，ACTIVE，但dispatcher仍plan/BLOCKED，不是自动提交成功。

本轮只更新监控证据并定位Pro入口运行环境阻塞，没有训练/评测重提、模型修改、远端热改、取消或重复PRECHECK。历史BAFDR1267920/1267921未操作。证据：[27_HEARTBEAT_EVIDENCE_20260908_1551.json](27_HEARTBEAT_EVIDENCE_20260908_1551.json)。

## 16:31 心跳补充

本轮真实触发2026-09-08 16:31:36.597 CST，实时远端采样16:32:59至16:33:06。登记的12个作业均保持COMPLETED(0:0)，其中1277955仍只是FULL PRECHECK；没有新增运行失败或最终性能。没有将其他任务的运行/排队状态并入六路线。

- CT G0/G1仍64.2720%/61.6740%，mAP@0.7为42.6802%/39.9154%；训练与评测SHA均exact/clean，epoch59 audit均6000，checkpoint stat及收据mtime/指标未变。复用14:29实际权重与15:10自哈希认证，不重复大checkpoint、准入或推理。G2/G3继续保留78cde1be训练身份。
- H65六份小收据未变，最高64.2265%，已出分配置仍phase-off；ET关闭/开启仍62.0768%/54.8096%。时间聚合/匹配控制、anchor修正/精度核对仍未解决，原始官方基线来源补证和独立复评仍待完成。
- Evidence A1/A6/F训练及A1评测日志无新fatal，A1仍51.5968%，A6/F没有独立metrics；BAFDR U16/LATE/NOKD仍保存epoch59/update6000，新metrics为0，FULL无正式训练。utility/真实辅助损失、padding/终态screen，以及Unified真实Taylor/H65运行时机制缺项继续保留。

16:33可见可调度25节点CfgTRES200/AllocTRES197，剩余3/200 GPU；账户9 RUNNING/2 PENDING，共11项。其他任务1279871为Priority、1278774为JobHeldUser，均未操作。实验盘175G可用、97%已用；分钟回执56秒，ACTIVE但dispatcher仍plan/BLOCKED，只读轮询不等于自动提交正常。

上一轮Computer Use初始化的kernel assets路径错误仍记为未解决，本轮没有重复重置/探测，也没有访问ixBrowser、发送材料或进行Pro咨询。没有新模型修改、提交、取消、重提、远端热改或重复PRECHECK；历史BAFDR1267920/1267921未操作。证据：[28_HEARTBEAT_EVIDENCE_20260908_1631.json](28_HEARTBEAT_EVIDENCE_20260908_1631.json)。

## 17:33 用户协议修订与正式部署

用户明确要求每5 epoch全测试集评测，并明确允许选择最佳checkpoint、按测试集调参。
新实验标记TEST_GUIDED_EXPLORATORY_EVAL5，选择未四舍五入Avg-mAP最高的EMA、同分取最早epoch；
保留完整曲线、全部调参尝试和单独的epoch59终态EMA。披露测试集重复使用，不能称未见测试泛化
或与官方论文协议公平可比；推理仍无测试GT。旧实验不追溯改写，本节覆盖此前针对新实验的禁止选模规则。

- H65新629162cd恢复整窗TIA，选后384帧使用192tubelets，dense768使用384tubelets。
  跨clip、不跨视频的输出和梯度测试通过；相位开关控制只改allocation。CUDA51tests与三个PRECHECK由1280125通过。
  新均匀/相位关闭/相位开启正式1280127/1280128/1280129均RUNNING，已进入epoch1。
- ET新9a346f0d修复TIA前anchor correction污染，仍是fixed-stride共享低秩代理而非精确JVP/事件门。
  远端37tests和真实2GPU训练/推理PRECHECK1280077/1280078通过。ON1280117、OFF1280118均RUNNING，FP32/global batch2。
- 17:33:38核对五份实际protocol.json，自哈希、精确SHA、种子、211视频/792窗口和五轮选模日程一致。
  新全测试集分数尚未产生，不能把初期loss当性能；当前是实现修复和新协议启动，尚未完成结果驱动调参。
- CT后续矩阵的eval5、BAFDR padding/screen及专用训练接线、Evidence utility/robust/cycle和Unified Taylor/H65机制仍未完成。
  此次5项启动不是六路线整体验收。其他任务与历史BAFDR1267920/1267921未操作，未进行Pro咨询。

详见[29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.md](29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.md)及同名JSON。

## 18:25 新协议真实中期成绩与最佳 EMA

五项新作业仍 RUNNING，尚无 epoch59 终态。ET-TRC ON/OFF 第5轮 Avg-mAP 为
5.7073%/4.2952%，第10轮升至32.0496%/33.7828%，最佳EMA已自动更新到第10轮。
H65 均匀/相位关/四相开第5轮分别为5.2404%/5.4013%/4.9665%。
五份真实最佳EMA权重有限，内部SHA/seed/epoch/指标与对应轮次收据一致，现有收据自哈希通过。
H65三组第5轮optimizer/scheduler/EMA/DUCA schedule均500，初始AMP skip已replay、没有耗尽。

这些是测试集参与选模的中期结果，不是终态性能。没有因初期低分重启健康训练，没有新超参数尝试。
其他四路线的既有结果和缺失机制未变化；CT G0/G1为64.2720%/61.6740%，BAFDR新结果、
Evidence F/A6独立收据以及Unified缺失机制仍未补齐。历史BAFDR1267920/1267921未操作。

详见[30_TEST_GUIDED_EVAL5_PROGRESS_20260908_1825.md](30_TEST_GUIDED_EVAL5_PROGRESS_20260908_1825.md)及同名JSON。
