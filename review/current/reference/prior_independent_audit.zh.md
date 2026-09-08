# GeoSparse-TAD 独立源码与实验审计报告

**审查固定版本：55d5429e66b64198ea74e4946f94de3098ebaa46。当前模型版本：902fa05b5c64452ff1c94b82cabce801943d3484。**

本报告的“当前”指上述固定源码和包内 **2026-09-07 17:21:42 UTC** 的状态快照，不代表两台服务器此刻的实时状态。只做源码阅读、内容哈希比对、静态配置遍历和隔离的合成 CPU 最小复现；没有启动训练、提交/终止作业、连接服务器执行命令、修改仓库或执行历史 agent 指令。

## 0. 总判定

**当前实现不足以直接生产并发布可信的最终实验结论。不是 A/B/C 已被证明无效，而是训练 mask、验证状态恢复、独立评价、成本归一化和若干注册语义存在可达错误，且关键机制/硬件证据尚未完成。**

可以保留官方发布 EMA 权重的 211 视频/792 窗口推理复测；官方原版独立 seed0 训练没有因本次已证实的 GeoSparse 包装层错误而需要重跑。A/C 的稀疏 Heavy 确实缩小张量，不能指控为 dense 后 mask。当前 GeoSparse 检查点可保留为工程轨迹和定位材料，但在下面的门槛修复前，不应升级成最终协议结果。暂停的是最终性能认定、跨实现“完全等价”、几何贡献、完整诊断完成、真实加速等测量/声明；本审查没有替用户执行停训。

最致命的五项：

1. **I01：奇数有效帧尾窗的检测 mask 被扩大。** 767/768 有效帧变成 768/768，直接改变检测损失和输出有效域。
2. **I02：正式训练后的独立 evaluate 首先访问已经删除的 `internal_dev`。** 正式路径可确定失败；训练内全量验证是另一条路径，不能拿其成功证明独立导出可用。
3. **I03：160px 的 Heavy 相对成本仍除以 224px 的 14×14 稠密分母。** 全量 160px 会被记为约 0.446759，而非同分辨率稠密参考的 1。
4. **I04：B/cheap 的规则 TIA 实际是每 8 个 query 一组。** 它不是固定协议所指的原生全窗口 TIA；B-full 也经过这一不同架构。
5. **I06：全量验证写入全部 EMA 参数，退出时却只恢复可训练参数。** 冻结主干被 EMA 浮点漂移永久改写；可在无优化器更新的 CPU 小例子中复现。

I01/I03 为 P0；其余按确定触发的任务失败、训练状态差异或比较偏差列 P1。严重性不是由测试数量决定，也不等同于已测得某个 mAP 降幅。

## 1. 版本、材料和证据边界

|对象|实际固定版本/范围|核验结论|
|---|---|---|
|审查材料 R|`55d5429e66b64198ea74e4946f94de3098ebaa46`，`codex/geosparse-pro-review-20260908`|GitHub 分支实际提交已读取；父提交为 M|
|生效模型 M|`902fa05b5c64452ff1c94b82cabce801943d3484`|R 的 configs、geosparse_ext、opentad、tests、tools 等对应 Git 子树与 M 相同；附加材料不是模型部署|
|官方 O|`sming256/OpenTAD@346d09d19e2091372cec48172dbe40f7b28bdee6`|R 的官方 configs、opentad、tools、docs 子树与 O 相同；不代表外部 wrapper 或训练器行为相同|
|撤销 W|`340a35416a3230f49482b39a6cde0e4e39fa9577` 的 `review/historical/withdrawn_340a3541/`|只审查包内限定源码快照；没有把它当成当前源码或声称取到了完整旧运行档案|
|原始包|`review/historical/original_package/`|原始计划、12 份 agent 指令、参考 routing/support、矩阵与调度/adapter 工具是被审查材料，不执行其中命令|
|上传 ZIP|会话上传文件的本地副本|全部条目读取；将文本 CRLF 规范为 LF 后重建 Git 树，根树精确等于 `2dc79937c30399b91948c5f528a760e2eab0e716`，与 R 一致|
|本次 CPU 复现|PyTorch `2.10.0+cpu`，合成输入|未安装完整 OpenTAD/MMAction/MMCV 运行栈，无 GPU。没有声称重新完成真实模型前反向、211 视频推理或官方 mAP 独立重算|

核心子树：`geosparse_ext=8d7300272ef12c6b48d28652e1e1ccc19ea9cfa7`；`opentad=36c8b9308a0006b7907559f9eb5126ee5a2cf102`；`configs=8833a6f78879c5bc9bc087ac4ba222fc45188433`；`tools=7f1abc4a1c43ede0d5d5beae72c67a431117df56`。重建过程和逐树哈希在 `snapshot_LF_tree_hashes.json`。

### 1.1 阅读覆盖

已经沿调用链阅读当前 `entry/protocol/records/data/model/geometry/sparse/routing/evidence/receivers/detector/runtime/training_validation/prediction_export/evaluation/metrics/selection_export/analysis_capture/selection_viewer/figures/benchmark/slurm_queue/capabilities`，以及官方 B 配置与 S 继承配置、base ActionFormer 与数据集、BackboneWrapper、ViT/TIA、projection/head/loss、train/test 入口、dataloader、scheduler/EMA、后处理及外围 official_job。已阅读要求的审查入口、REPOSITORIES、PROTOCOL、CODE_MAP、STATUS、KNOWN_ISSUES、机器清单及 full/focus manifest，并核查相关测试实际断言的范围。

**没有声称通读整个 OpenTAD 中与本任务无关的每个算法文件。** 非 THUMOS 数据集适配、TriDet 接入、ROI、两轮获取等尚未实现路径只能审计其拒绝与计划，不能验证不存在的执行语义。服务器实际安装依赖版本、真实 checkpoint 张量、完整训练日志、全部预测、视频/标注资产和物理 GPU UUID 没有随包提供，无法现场核实。这些限制不妨碍上述源码错误的证明，但妨碍计算它们的实际 mAP 影响或认定正在运行的作业是否已经触发。

### 1.2 全部注册配置与任务覆盖

独立解析结果：**204 个 dataset/model 组合；612 train、612 evaluate、204 benchmark、117 diagnostic，共 1545。** focus 为 5/5/5/15，共 30；没有新增 seed1/2 的授权。

对 204 组合逐项输出 `configuration_coverage_204.csv` 和同名中文 Markdown 附表，含完整 model JSON、配置差异字段、调用链依据、拒绝原因、关联问题和运行证据。`job_coverage_1545.csv` 对每个 ID 给出 kind、seed、父任务、静态准入和当前阶段状态。

**119 个“实现存在且字段生效”**仅表示注册差异字段已经追到实际算子/损失/路由操作，不表示共同错误不存在、完整运行通过或性能成立。**79 个显式阻塞。6 个放在“静态入口允许但语义未核验”审计类**：4 个 VideoMAE-L 组合缺实际运行证据；另 2 个并非没读，而是已发现语义异常——160px 的 spatial_group=7 可确定运行失败、feature_l2 不是登记的特征一致性损失，附表已明确标注。没有把这两项伪装成待查。

125 个组合被静态入口接受，对应 375 train、375 evaluate、125 benchmark；这不是可成功完成数。117 个 diagnostic 全部被入口显式阻塞。“只有计划”体现在这些诊断机制和未落地扩展内部；不把它们重复算成另一批模型配置。

|族|组合数|静态接受 / 显式阻塞|已经核验的范围及例外|
|---|---:|---:|---|
|F00|10|5 / 5|THUMOS dense、A-full、B-full、C-all-fine、cheap；ActivityNet 阻塞|
|F01|46|23 / 23|A/B T/ST 固定和动态预算、C 粗细预算；非 THUMOS 阻塞|
|F02|28|22 / 6|uniform/random/motion/actionness/uncertainty/CDF/hybrid/PG 等实际分支；DUCA 阻塞|
|F03|13|7 / 6|DEPTH 4/6/8/10、BCR prefix 0/2/4；PBD-inspired 与非 THUMOS 阻塞|
|F04|11|11 / 0|temporal atom 1/2/4/8，spatial 1/2/7；7 在 160px 下是静态漏拒绝，见 I09|
|F05|12|7 / 5|五种 B receiver 有不同算子；A/C native/no-position；其余错误几何控制阻塞|
|F06|14|10 / 4|全局/parent 配额、是否允许零、categorical 动态预算；阈值/LUT 变体阻塞|
|F07|18|16 / 2|Scout 分辨率80/112/160、宽度64/128/256、stride1/2/4、actionness BCE；detail probe 阻塞|
|F08|22|16 / 6|PG/acquisition、探索概率、probe周期、warmup；retention/梯度代理/ST 等未实现分支阻塞|
|F09|13|4 / 9|B 的时间与空间 fidelity 四组合；ROI 九组合阻塞|
|F10|12|12 / 0|slots1/4/8、receiver层1/2/4、融合、C四变体；feature_l2 有登记语义冲突|
|F11|7|1 / 6|默认 parent16/单轮/一次路由可执行；两轮、local4、cross-parent、逐层重路由阻塞|
|F12|8|4 / 4|VideoMAE-L 4 组合仅静态/构造路径，无实际形状与训练证明；FineAction 4 阻塞|
|F13|8|5 / 3|ActionFormer 和 query384/768；TriDet 接入阻塞|

族之间复用配置，表中行数不能相加当作 204。

## 2. 五个聚焦训练的完整调用链

状态来自 [`review/STATUS.zh.md:3–19`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/STATUS.zh.md#L3-L19) 与所附凭证，不是实时查询：

|train ID|路线|快照状态|专属执行路径|最终评价共同阻断|
|---|---|---|---|---|
|tr-c43d3e9cad58|A_ST_fixed_0.5|N16R4，1277994，RUNNING，完成5轮|Scout/Router→原生 atom→selected Heavy→全窗口 TIA→ActionFormer|I01、I06、I02、I03；快照 best=null 不能单独证明验证失败|
|tr-be89cd6cb2ca|B_ST_fixed_0.5|A100，244973，PENDING|dense cheap query + 无TIA的稀疏 Heavy→evidence slots→support receiver→8-query TIA|共同问题，另 I04/I05|
|tr-a45686a40afa|C_fixed_0.5|A100，244975，PENDING|粗细互斥 pack→Heavy→原坐标 delta 回写→全窗口 TIA|共同问题；q=.5 是 mixed-token 比，不是 fine 比|
|tr-c2d93c4c3979|B_full|A100，244976，PENDING|同一个新 receiver/query/TIA，全量 evidence|可估 B 架构损失；不是官方 dense-limit 等价证明|
|tr-bacdbb058eca|geosparse_dense_control|A100，244977，PENDING|统一训练器中的全量原生 ViT/TIA→ActionFormer|不是官方原版独立复现；I07|

五者均经 job 绑定、未知字段/能力检查、识别预训练来源检查、200 视频训练 split、原版 LoadFrames 与附加支持观测、相同 detector head 目标；统一 trainer 的 global batch2/microbatch2；autocast/GradScaler、clip_grad_norm、scheduler/EMA、完整状态 checkpoint、每5轮全测试集验证、严格改进 best；再进入独立 export/evaluate、诊断、硬件与绘图。链路并非只看三个模型类。

A 的实际凭证 [`review/evidence/live_protocols/N16R4/1_resolved_protocol.json:1–71`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/evidence/live_protocols/N16R4/1_resolved_protocol.json#L1-L71) 记录输入 `2×1×3×768×160×160`、12 个 TIA 的时间尺寸384、可训练参数 **31,388,046**；快照 500 次尝试、498 次成功更新。其他四配置有对应 GPU 预检，不据此外推60轮成功。没有实际运行时参数清单的路线，不编造其精确 trainable 参数数；静态可确认其 trainable 组件和 optimizer 分组代码。

## 3. 官方基线与统一稠密对照

### 3.1 recipe 和实现对齐到哪里

官方依据为 [`configs/adatad/README.md:63–80`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/configs/adatad/README.md#L63-L80)、[`configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py:1–13`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py#L1-L13) 及其 S 继承配置，而非论文中其他 VideoMAE 尺度的表格。统一配置从 B 构造，保留官方像素 recipe：RGB mean=(123.675,116.28,103.53)，std=(58.395,57.12,57.375)；不是把正确的 57.12 指成错误。LoadFrames 实际 temporal stride4，文件名768x1不能当成stride1。随机时间裁剪/GT截断、padding、训练空间裁剪/Flip/ImgAug/ColorJitter，测试160中心裁剪均沿官方管线，新增 PTS 记录不重采样原版 frame indices。见 [`geosparse_ext/protocol.py:76–115`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/protocol.py#L76-L115)、[`geosparse_ext/data.py:18–34`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/data.py#L18-L34)、[`geosparse_ext/data.py:54–69`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/data.py#L54-L69)、[`opentad/datasets/transforms/end_to_end.py:199–215`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/opentad/datasets/transforms/end_to_end.py#L199-L215)。

|项目|官方原版 O|统一稠密对照 / A、C|B / cheap|
|---|---|---|---|
|768输入、160px、2-frame tubelet|是|是；空间网格10×10|cheap保留规则序列；B Heavy fidelity 受独立注册参数影响|
|attention parent|48个16-frame parent|保持，不跨parent做Heavy attention|Heavy保持parent16|
|TIA|原生全窗口384tubelets，时间depthwise卷积；不是全窗口self-attention|A/C/DENSE保持该布局|regular_tia=8，见I04|
|PE、LN/MHSA/MLP、残差|固定源码|原生PE插值、原顺序；全选编码器极限成立|新query/receiver，不主张官方输出一致|
|检测mask|原始frame mask|被pair-any扩张，I01|同样受I01影响|
|projection/head/loss/NMS|ActionFormer原实现|复用原实现，输入通道按路线改变|projection输入256；receiver是新增容量|
|识别预训练与冻结|识别权重，冻结ViT，训练TIA/head|同来源；额外Scout/Router等可训练；没有独立Dense TAD Teacher|Heavy禁用TIA，训练query/receiver/regularTIA/head等|
|optimizer/LR/WD|B的adapter LR=1e-4，WD=.05；检测器按官方分组|B继承与分组在代码中保持，不误用S的2e-4；新模块是额外组|新模块成本/容量必须另报|
|global batch|DDP2卡，每卡1|单卡2，没有再除一次world size|同统一训练器|
|loss normalizer|每卡局部positive数更新，随后DDP平均梯度|单卡全batch positive数更新，数学上不等价，I07|同左|
|AMP、梯度裁剪|solver.amp=True，clip_grad_norm=1；训练/测试均按cfg启用|autocast/GradScaler/clip保持大框架；overflow步进不同|同左|
|scheduler/EMA步进|原版每iteration前进，包括GradScaler跳过更新的iteration|仅successful update推进，I07|同左|
|EMA|全部state，.999/.001|算术相同，但验证restore不完整，I06|同左|
|seed/初始化|原构造顺序、DDP、原deterministic策略|构造顺序/loader RNG/分布式归一化不同|同左|
|warmup/cosine/结束|5/100/60|5/100/60，不是cosine60|同左|
|验证与best|完成42、44…60，10次，EMA full-test best|完成5、10…60，12次，EMA full-test best|同左|

因此，**forward/相关梯度一致只证明指定输入与随机设置下的编码器极限，不证明完整训练轨迹、detector mask、初始化、loss归一化、overflow或选点机会等价。** 统一稠密对照必须保留 `geosparse_dense_control` 的名字。

### 3.2 official_job 外围独立核验

[`review/launchers/official_job.py:34–82`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/launchers/official_job.py#L34-L82) 加载固定 B 配置，只覆盖真实资产路径/work_dir/保存预测；断言200/211和792，官方train使用torchrun world2，test world1，调用未改动的 `tools/train.py`/`tools/test.py`。[`tools/train.py:187–224`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/tools/train.py#L187-L224) 先保存当轮checkpoint，再验证；wrapper从训练日志识别零基epoch，从完整预测文件重新调用官方 evaluator，并保存同epoch的EMA/best，见 [`review/launchers/official_job.py:94–134`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/launchers/official_job.py#L94-L134)。完成42轮对应epoch41，完成60轮对应epoch59，未发现此处off-by-one。

日志中的“Average-mAP”只是触发读取完整预测，不是把打印的四舍五入值当最终指标。完整视频集合检查存在。包中已执行的 released-checkpoint launcher 与当前 wrapper 的 train 条件分支不同：旧分支有5轮验证覆盖，但该次 `mode=evaluate`，没有触发；不能据此指控当前官方训练仍每5轮验证。

当前 wrapper 的资产检查和代码边界正确，不构成其训练已经完成的证据；也没有现场验证长时间中断后的官方原入口恢复操作。

### 3.3 发布权重复测：保留什么，不能推出什么

来源 [`review/evidence/official_released_checkpoint/metrics.json:1–18`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/evidence/official_released_checkpoint/metrics.json#L1-L18)，与官方表 [`configs/adatad/README.md:67–71`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/configs/adatad/README.md#L67-L71) 对比：

|tIoU|作者表 %|复测 %|差异：百分点|
|---|---:|---:|---:|
|0.3|85.95|85.96337198|+0.01337198|
|0.4|81.86|81.85797683|-0.00202317|
|0.5|75.02|75.00131299|-0.01868701|
|0.6|63.29|63.29255169|+0.00255169|
|0.7|49.56|49.57876099|+0.01876099|
|平均|71.14|71.1387948970|-0.0012051030|

逐tIoU最大绝对差为 **0.01876099个百分点，发生在0.7**。可以称发布权重真实推理复测接近官方表；不能称逐位一致，不能未经验证归因于单一舍入/硬件因素。本审查没有完整预测或权重，所以是核验凭证与执行链，不是本地再次独立重算71.1388%。更不能据此宣称官方从识别预训练独立训练已复现，或 GeoSparse 已达到该精度。

## 4. 问题清单

以下所有代码链接固定到 R；核心模型文件同时已证明与 M 内容一致。P0=可能改变结果或支持错误核心结论；P1=合法任务失败、协议语义冲突或比较偏差；P2=非阻断缺陷/有条件风险；P3=文档或维护。每条的“影响”不外推未提供的真实mAP。

### I01 · P0 · 检测有效域被 tubelet.any 扩大

**确定程度：代码已证实，合成CPU已复现。**

**证据。** [`geosparse_ext/geometry.py:18–25`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/geometry.py#L18-L25) 用 `pair_valid.any(-1)` 构造 native eligibility；[`geosparse_ext/model.py:126–129`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/model.py#L126-L129) 将这个tubelet mask最近邻放大回query网格并返回 `DetectionState.valid`；[`geosparse_ext/detector.py:69–73`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/detector.py#L69-L73) 将其交给检测目标/损失。官方最终使用输入frame mask，[`opentad/models/backbones/backbone_wrapper.py:118–120`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/opentad/models/backbones/backbone_wrapper.py#L118-L120)。

**触发配置与错误链。** 五个focus及其他768-query配置，只需窗口含奇数个有效帧。例如 `[True]*767+[False]` → 最后tubelet两帧any=True → 回插值得到768个True → 首个padding位置参与projection/head的有效域、loss或候选预测。原版随机裁剪和尾窗允许奇数有效长度，并没有使该条件不可达。

**影响。** “A-full/Call-fine与官方训练、目标和完整输出等价”目前不成立；即便EMA验证遍历全部211视频，它仍是在错误mask上测的该实现。实际出现频率和mAP影响没有资产不能量化。既有测试的偶数尾部/编码器输出比较不能证伪此例。

**最小修复。** 分离Heavy的tubelet eligibility与detector的输出mask；query768保持原始frame mask；query384采用显式登记的对应规则，不把pair-any逆插值当无损逆变换。

**证伪/回归测试。** 真实768输入、valid=767、带GT与无GT，各比较官方和dense/A-full/C-all-fine的最终mask、正负样本、loss、raw proposals及相关梯度；再覆盖双样本不同尾长、AMP、train态DropPath/checkpoint。只检查encoder特征为零不够。

### I03 · P0 · 160px成本输出继续使用224px分母

**确定程度：代码已证实，数值CPU重算。**

**证据。** [`geosparse_ext/analysis_capture.py:133–149`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/analysis_capture.py#L133-L149) 的分子来自真实trace，但分母硬编码 `native_n = 8 * 14 * 14`，注释明确仍用224参考；[`geosparse_ext/figures.py:187–198`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/figures.py#L187-L198) 将该比例用于full-reference成本展示。训练时 [`geosparse_ext/model.py:105–108`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/model.py#L105-L108) 的hn/wn分母已经正确，因此这不是“Router成本监督也必然错”的证据。

**触发与错误链。** 任一当前160px正式selection/cost导出，包括DENSE全量：每parent真实N=800，分母N=1568。以B的D=768、12层、48parent计，真实Heavy公式得3,827,721,830,400 MAC；旧分母8,567,755,112,448 MAC；全量被标 **0.4467590145**。这不是稀疏带来的55.3%节省。

**影响。** 当前相对Heavy成本、跨分辨率图和由其支撑的“近半成本”结论须暂停。绝对trace MAC、另一套实际算子counter或真实latency不能因此一概作废。

**最小修复。** 将参考分辨率、parent形状、backbone深度/宽度作为显式导出字段；当前主协议用160稠密参考。224参照只能用于清楚命名的跨fidelity消融，不能冒充同配置full limit。

**证伪测试。** 同分辨率160-full=1、224-full=1、C-all-coarse>0；重算旧JSON/trace即可，不需要重训。B112对何种参考比较必须在字段和轴标签明示。

### I02 · P1 · 独立evaluate仍消费已删除的internal_dev

**确定程度：代码已证实，路径确定失败；无需等60轮验证才能定位。**

**证据。** [`geosparse_ext/records.py:44–65`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/records.py#L44-L65) 生产 `training/internal_diagnostic/validation`；[`geosparse_ext/evaluation.py:47–59`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/evaluation.py#L47-L59) 首先循环 `internal_dev`；[`geosparse_ext/prediction_export.py:24–60`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/prediction_export.py#L24-L60) 子进程接受旧名称并取 `split[subset]`；[`geosparse_ext/metrics.py:87–98`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/metrics.py#L87-L98) 校准继续消费旧键。[`geosparse_ext/training_validation.py:75–79`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/training_validation.py#L75-L79) 也有旧别名分支。

**触发与链。** 任一静态接受的正式evaluate，包括五个focus的训练后任务：训练完成 → 首先导出internal_dev → split无此键 → KeyError/子进程失败 → 风险校准、正式指标导出无完成结果。周期性训练内validation直接走validation，并不触发这个首分支。

**影响。** 375个静态接受evaluate并不等于可执行；已存在内部全量验证checkpoint仍可保留。没有证据说此bug把测试GT用于梯度；当前首先是失败，不是隐蔽泄漏。

**最小修复。** 完整迁移生产者与消费者；`internal_diagnostic`用全部200训练视频、物理annotation subset=training、训练视频root和诊断用确定性test-style管线；不能只改字面键后仍筛选不存在的annotation subset，不能恢复20视频heldout。短动作阈值和风险阈值只在训练别名上拟合。

**证伪测试。** 从split生成开始，完整跑诊断别名200视频→冻结阈值→正式211视频/792窗口；断言真实annotation未重标、GT没有进test梯度/utility/阈值拟合；空预测也保留视频。补做的是测量，不是另一轮60epoch训练。

### I04 · P1 · B/cheap regular TIA 的实际时间作用域不符

**确定程度：代码已证实；不声称已经测出精度损失。**

**证据。** [`geosparse_ext/model.py:33–41`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/model.py#L33-L41) 创建 `Adapter(256, temporal_size=8)`；[`geosparse_ext/model.py:117–121`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/model.py#L117-L121) 用 `query.reshape(-1,8,256)`，并错误注释为source clip-local scope；[`geosparse_ext/protocol.py:45–45`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/protocol.py#L45-L45) 只接受 `tia_scope=source_native`。协议 [`review/PROTOCOL.zh.md:12–12`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/PROTOCOL.zh.md#L12-L12) 明确不能把TIA缩成单parent。

**触发与链。** B、B-full和cheap的query768 → 96组独立8-query TIA；query7与8不能在这层通过时间卷积交互。官方是沿384tubelets的规则窗口执行TIA，parent是Heavy attention的分组，不是该TIA的时间边界。B禁用Heavy内部TIA本身是明确的路线设计，错误在新regularTIA的声明与实际范围。

**影响。** B的几何/原生时间保持主张不能按现实现写；B-full通过相同新receiver是正确控制，但它的架构税包含query/receiver/TIA，不能当成官方等价full极限。对C/A的384TIA不能连带定错。

**最小修复。** 在B规则状态定义处实现约定的全窗口时间作用域，并固定native384→query768的先后；简单把8改成384却继续在768 query后reshape会切成两组，仍不等价。若作者选择另一种全query定义，必须显式版本化，不能称用户已同意新的协议。

**证伪测试。** 在原生parent边界及query7/8边界做脉冲扰动，记录每个TIA实际内部时间尺寸；同时验证Heavy attention仍不跨parent、无证据null行为不变。快照中的B/B-full尚PENDING，修此处不必扔掉不存在的完整训练结果。

### I05 · P1 · 官方随机几何增强没有进入source ROI记录

**确定程度：当前记录代码缺项已证实；部署环境具体触发依赖实际MMAction版本和随机增强，未取得其现场版本凭证。**

**证据。** [`geosparse_ext/protocol.py:97–109`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/protocol.py#L97-L109) 保留官方ImgAug；[`geosparse_ext/data.py:43–49`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/data.py#L43-L49) 只记录crop_quadruple与显式flip。核读MMAction2 v1.1.0 `mmaction/datasets/transforms/wrappers.py` 的ImgAug：default包含Shear/Translate/Rotate，transform变更图片而不为这些操作更新crop_quadruple/flip等自定义字段；其文档也明确这一点。[`geosparse_ext/evidence.py:34–39`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/evidence.py#L34-L39) 根据不完整矩阵反投影source ROI。

**触发与链。** B使用原版default ImgAug且抽到空间变换 → 实际采样内容已经平移/旋转/剪切 → GeoSparse仍只保存crop+flip → support-attention的source ROI/原图解释与实际内容不对应。测试CenterCrop更新crop_quadruple是正确的，不能当成另一个bug。

**影响。** 主要伤及B的几何归因与原图可视化可靠性；A/C在增强后view上的native index并不因此自动失效。没有现场依赖版本就不能断言每个训练batch已发生该错。

**最小修复。** 不移除官方ImgAug；在保持同样像素、随机调用顺序的前提下记录已实现的空间变换及填充/擦除区域。旋转/剪切的ROI回投影需四角或多边形，不能继续只投影左上/右下两点；或者显式改称增强view坐标、不虚报source ROI。

**证伪测试。** 安装版本锁定后，以坐标棋盘/已知点执行固定平移、旋转、剪切和crop+flip，选中像素与导出的source support逐项一致；记录开启/关闭时像素张量逐位相同。

### I06 · P1 · EMA验证后冻结参数没有恢复

**确定程度：源码及实际CPU最小复现已证实；实际GPU精度影响未知。**

**证据。** [`geosparse_ext/training_validation.py:18–55`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/training_validation.py#L18-L55)：`parameters`仅存requires_grad；进入时对 `ema.shadow` 的全部state执行copy；退出时只copy保留过的参数。[`geosparse_ext/runtime.py:59–73`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/runtime.py#L59-L73) 的EMA对全部参数做 `.999 * shadow + .001 * value`，浮点数即便输入不变也可有累计漂移。

**触发与链。** 全部统一训练器每5轮验证：冻结param与其EMA浮点值略异 → 验证覆盖它 → finally不恢复 → 后续训练的识别主干已经改变。checkpoint在验证前写入，则连续训练与从同checkpoint恢复也可能不同。buffer、模块train/eval mode和RNG的恢复代码确实存在，不能把它说成全面没恢复。

**CPU证据。** 10000个冻结随机数、500次EMA更新、无优化器更新：验证后1547个冻结元素变化，最大绝对差 `4.076957702636719e-05`；可训练参数正常恢复。这不是构造一个不可能出现的任意EMA差值。

**影响。** “冻结主干不变”“验证不改变训练状态”“原版训练行为等价”不成立。已记录best仍可对应当次实际EMA；不能据此声称best文件必定错配或mAP必定大幅失效。

**最小修复。** 保存和恢复所有被EMA覆盖的参数，或采用独立EMA模型；不要通过去掉冻结参数EMA来悄悄改官方算术。

**证伪测试。** 自然EMA更新后，对进入/正常退出/异常退出比较完整state、RNG、mode；所有参数严格恢复。保留现有checkpoint用于量化，不能仅凭roundoff就要求所有训练重跑。

### I07 · P1 · 统一dense与官方训练不等价，不能改名

**确定程度：代码差异已证实；比较偏差大小没有真实训练对照不能估计。**

**证据。** [`geosparse_ext/runtime.py:34–38`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/runtime.py#L34-L38) 单卡batch2；[`opentad/models/dense_heads/anchor_free_head.py:173–208`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/opentad/models/dense_heads/anchor_free_head.py#L173-L208) 用本次local positives更新loss_normalizer并除loss。官方双卡每卡1随后DDP平均，与单卡2的全batch正样本数归一化不同。[`opentad/cores/train_engine.py:38–72`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/opentad/cores/train_engine.py#L38-L72) 原版每iteration更新scheduler/EMA；[`geosparse_ext/runtime.py:207–218`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/runtime.py#L207-L218) 只在GradScaler成功更新时推进。

**触发与链。** geosparse_dense_control和所有Geo路线：相同global batch并不消除local loss归一化和不同初始化RNG顺序；AMP overflow时scheduler实际进度也分岔。A快照已有500尝试/498成功，因此overflow步进区别不是永远不可达的理论分支。没有再次除world size的错误。

**影响。** 同一recipe字段、同一pretrain、某一次full forward相同，仍不能称“官方原版复现”。这是归因边界，不等于统一训练器本身不合法。

**最小处理。** 保留两条基线名称：官方原版独立训练；统一trainer dense control。A/B/C的增量优先相对后者，官方作外部协议参照。不得为追求叙述等价擅改已授权官方训练或恢复holdout。

**证伪对照。** 固定相同权重和两个样本，记录两卡local positive数/normalizer/梯度均值，对比单卡2，加入一次受控overflow。只有完整比较能够推翻“不等价”，encoder梯度测试不能。

### I08 · P1 · 部分验证失败仍可被队列视为训练完成并启动benchmark

**确定程度：代码已证实；快照未提供实际部分验证失败证据。**

**证据。** [`geosparse_ext/runtime.py:245–253`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/runtime.py#L245-L253) 返回60轮完成，同时可能 `best_checkpoint_selection_complete=False`；[`geosparse_ext/entry.py:121–136`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/entry.py#L121-L136) 写completed；[`geosparse_ext/slurm_queue.py:126–131`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/slurm_queue.py#L126-L131) 不要求该完整性字段。[`geosparse_ext/runtime.py:256–270`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/runtime.py#L256-L270) 的benchmark模型加载也不检查它；相对地，prediction_export的training_source会拒绝不完整选点。

**触发与链。** 至少一个验证成功产生best、另一次失败、训练跑到60 → result/status仍completed →队列DONE/benchmark可进入 → 只基于部分选点机会的best可能被计时。若全部验证失败，best不存在会失败；失败epoch列表和中间异常也有记录，不能称“完全漏报”。

**影响。** 完成60轮优化与完成12次正式选点被混用，影响流程和可比性，且给I13错配创造合法触发路径。

**最小修复。** 分开training_complete与selection_complete，下游正式评价/计时统一检查后者；用已经保存的epoch检查点补齐失败验证。不得为补齐一次评价重跑60轮训练。

**证伪测试。** 人为让第5轮验证失败、其余成功：优化可完成，但最终eligible状态必须拒绝；补测并重算best后才能通过。保留失败凭证，不覆盖成从未失败。

### I09 · P1 · 160px下spatial_group=7静态允许、运行必败

**确定程度：代码和CPU复现已证实。**

**证据。** [`geosparse_ext/protocol.py:51–51`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/protocol.py#L51-L51) 允许1/2/7且未与实际网格联检；[`geosparse_ext/model.py:58–60`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/model.py#L58-L60) 以10×10调用atoms；[`geosparse_ext/geometry.py:35–36`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/geometry.py#L35-L36) 要求整除。

**触发。** F04 THUMOS A_spatial_group7、source_resolution160，静态blockers为空，canonical_atoms立即抛出 `spatial groups must partition the native patch grid`；不属于当前五focus。

**影响/最小修复。** 登记的合法任务会预检/运行失败，而不是“尾部已经正确处理”。加入resolution×axis×group交叉校验，显式BLOCKED直到定义完尾部分组。不能静默改224、丢边缘token或减少数据来通过。

**证伪测试。** 160下g1/g2通过、g7明确拒绝；224下g7可整除；112仅允许其登记的B时间选择；C严格要求2×2互斥。

### I10 · P1 · 同ID的full/focus保存契约不同，诊断仍指向旧epoch

**确定程度：两个manifest逐ID比较已证实。**

**证据。** [`geosparse_ext/matrix.py:64–81`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/matrix.py#L64-L81) 的ID签名不包括checkpoint列表，原列表为零基 `[5,20,40,59]`；focus部署器将五train改为 `[4,9,14,19,24,29,34,39,44,49,54,59]`。诊断仍请求旧索引，见 [`geosparse_ext/matrix.py:251–260`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/matrix.py#L251-L260)。实际完整差异见 `full_focus_differences.json`。

**触发与链。** 同train ID在full/focus模型JSON相同，但不可变产物集合不同；focus训练完成后，诊断请求epoch5/20/40时没有对应文件，只有59交集。当前D任务显式BLOCKED，所以不是已经观察到的诊断运行失败。

**影响。** 同ID不再唯一表示产物契约；不能把两个manifest当完全同义。evaluate登记最终epoch59、实际选EMA best的说明也应同步，不能把best当固定末轮。

**最小修复。** 给产物契约/协议修订单独digest并保存原绑定，不改写正在运行任务的旧SHA；诊断显式改为已有完成5/20/40/60轮，即索引4/19/39/59，作为可追溯修订，不额外训练。

**证伪测试。** 对每个same-ID比较模型、训练定义、checkpoint集合、选择规则和输出契约；每条诊断的实际epoch文件必须可解析。附表对所有30个共有ID已比较，不只抽样。

### I11 · P1 · feature_l2不是登记的特征L2对齐

**确定程度：代码与原始登记语义冲突已证实。**

**证据。** [`geosparse_ext/receivers.py:90–91`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/receivers.py#L90-L91) 只执行 `F.normalize(features, dim=-1)`；检测loss未添加特征一致性目标。原始方案 [`review/historical/original_package/docs/EXPERIMENT_PROTOCOL.zh.md:32–32`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/historical/original_package/docs/EXPERIMENT_PROTOCOL.zh.md#L32-L32) 的F10称“L2对齐”，[`review/historical/original_package/docs/MODEL_SPEC.zh.md:83–83`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/historical/original_package/docs/MODEL_SPEC.zh.md#L83-L83) 讨论feature L2一致性；没有更新协议把它改成unit-norm控制。

**触发与链。** F10 B fusion_variant=feature_l2 → 单位范数化evidence → 不是最小化两个表示间距离 → 用该结果说明“L2对齐有害/无益”不成立。它确实改变forward，不是字段完全没使用。

**最小修复。** 把现有控制如实改名为feature normalization并留原登记轨迹，或实现注册的当前模型内一致性loss、系数及detach目标；不能借此引入被禁止的独立Dense TAD Teacher。

**证伪测试。** 检查loss字典和目标侧梯度；缩放evidence对unit-norm控制与L2一致性损失的作用不同，能直接区分两者。

### I13 · P1 · latency没有绑定具体best权重，Pareto只按训练ID拼接

**确定程度：缺失检查已由代码证实；未提供实际错配图。**

**证据。** [`geosparse_ext/benchmark.py:67–82`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/benchmark.py#L67-L82) 加载当前best但hardware.json不写selected_checkpoint_epoch/权重SHA；[`geosparse_ext/figures.py:405–410`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/figures.py#L405-L410) 对cost检查了epoch和model source，[`geosparse_ext/figures.py:418–425`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/figures.py#L418-L425) 对hardware仅按source_train_id和MEASURED/optimized/batch1选择。

**触发与链。** I08允许在不完整选点best上先benchmark；之后利用已有epoch补测，best改变，沿用同训练ID → 新accuracy和新cost能通过校验，但旧latency仍被拼入 → 准确率—延迟点不是同一组权重。即使没有I08，更新评价best后保留旧计时文件也有相同风险。

**影响。** 未来Pareto与速度归因可能错配。不能因ID相同或source_commit相同推断参数相同。包内没有真实完成硬件结果，因此不能声称现有速度已被污染。

**最小修复/测试。** hardware receipt记录checkpoint epoch、权重SHA、模型source、resolved配置和窗口IDs；绘图全部join，错配显示WAITING_DATA或拒绝。同ID两个best的合成receipt必须拒绝。只需重测无法证明配对的硬件结果，不需重训检测器。

### I12 · P2 · CUDA逻辑索引不保证等于nvidia-smi物理索引

**确定程度：合理疑点，有代码与部署映射证据；没有现场UUID核验，不能升级为已污染计时。**

**证据。** [`geosparse_ext/benchmark.py:31–37`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/benchmark.py#L31-L37) 用 `nvidia-smi -i CUDA_VISIBLE_DEVICES` 查询隔离；N16凭证为Slurm物理GPU1、容器CUDA0，A100也有分配GPU号与可见0不同的情形。NVML可见序号如何映射取决于实际容器/Slurm配置。

**触发/影响。** 若nvidia-smi仍使用主机索引而torch使用容器逻辑索引，会检查另一张卡的进程、误报隔离或误拒绝。没有实际UUID证据就不能保证这条检查实现了独占。

**最小修复/测试。** 用torch设备PCI/UUID映射到NVML UUID，并对照SLURM allocation记录物理UUID与PID。测试物理1→逻辑0映射，分别在物理0和1放置其他进程，只根据目标UUID判断。不要以修复为由终止非本任务作业。

### I14 · P2 · B的support是锚点支持，不能自动当作完整感受野或独立缓存键

**确定程度：跨token依赖由代码已证实；当前没有已实现缓存可供认定stale-cache bug。**

**证据。** Heavy在同parent的selected tokens间联合attention；[`geosparse_ext/evidence.py:21–39`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/evidence.py#L21-L39) 随后按空间slot池化，support只重复本tubelet两帧、ROI是slot范围。这是“这个slot锚在哪里”，不是“它的feature只依赖这两帧和这个矩形”。

**触发/影响。** 当前集合S变更时，即使packet自己的像素没变，其他selected packet可改变其feature。D06若用packet ID复用缓存、或论文把support说成精确独立信息依赖，就会错。现在反事实实际重跑受影响编码路径，D06又BLOCKED，不能虚构已经使用了错误缓存。

**最小修复/测试。** 分开anchor support、直接观测范围和编码依赖；不提供精确依赖图时明确无独立packet缓存保证。缓存至少绑定当前模型与上下文选择，或不缓存。固定anchor、扰动同parent其他selected token，feature应可变化；这个预期变化本身就是反驳独立缓存假设的最小对照。

### I15 · P3 · 部署器把family引用表写成matrix summary

**确定程度：代码已证实；未发现其改变训练队列的证据。**

**证据。** [`geosparse_ext/matrix.py:263–263`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/matrix.py#L263-L263) 返回 `jobs, refs`；[`review/launchers/deploy_corrected.py:101–115`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/review/launchers/deploy_corrected.py#L101-L115) 却用 `jobs, summary = matrix.compile_all()` 接收，然后把第二项写进 `matrix.corrected.summary.json`。该值是family→任务ID引用表，不是含counts/total_jobs等字段的真正summary。

**触发/影响。** 该部署器生成审查材料或再次生成部署文件时，summary文件名下出现另一种schema；消费计数/完成范围的工具可能失败或误读。训练job本身来自第一返回项，不能因此宣称已有训练配置被改变。历史1538计数与当前1545的差异也不能靠这个文件名自行解释。

**最小修复/证伪测试。** 分别命名refs和真正summary，复用矩阵主程序的计数逻辑并给schema断言；不执行部署命令，只对编译返回值做纯数据测试，要求total_jobs=1545、counts为612/612/204/117。历史方案应保持原文并标明过时，而不是把保存历史本身说成当前bug。

## 5. 逐路线技术结论：哪些正确，哪些证据还不够

### 5.1 A：真实稀疏重更新，编码核心成立，完整等价尚不成立

[`geosparse_ext/sparse.py:45–79`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/sparse.py#L45-L79) 在Heavy之前收集selected状态，QKV/attention/MLP收到实际K；回写native位置，reference/bucket均不是先做全量Heavy再mask。相同K可按parent分桶形成batch，但每个parent仍是独立attention样本，不构成跨parent注意力。原tubelet由原生相邻两帧构造，路由不按selected rank重新配帧；PE也不是按selected rank重新编号。

原先后顺序LN→MHSA→residual→LN→MLP→residual→TIA保持。A/C将规则state交回原版Adapter，以384tubelets恢复整窗口；TIA虽为局部时间卷积，但不会在每个16-frame parent边界重置。K=0跳过Heavy，不代表Scout/patch embedding/规则state/TIA或总成本为零；局部空parent和batch内不同K有实际分支。DEPTH是真实选择分布于层深的4/6/8/10层，BCR是真实dense前缀0/2/4，不是PBD完整算法。

DropPath按原parent顺序抽样，空组不任意改变随机调用；冻结Heavy的dropout/eval策略与source保持；checkpoint在输入需要梯度时执行，不因Heavy参数冻结就把上游TIA梯度切断。K=all有原block快捷路径。已有small-shape和生产shape source-limit证据是积极证据，但生产预检使用固定数学attention/随机条件，且仅比编码器。**没有覆盖I01的奇数frame级最终mask，也不能代替完整train态、AMP、随机DropPath、ragged、最终head梯度一致性。**

### 5.2 B：五种receiver确实不同，几何与架构归因仍未打通

Scout先形成每个原生tubelet对应的规则cheap序列，默认不是把query只留在selected时刻；native384到query768是插值，不能称创造了768个独立原生观测。stride2/4是注册的信息压缩消融，不应写成与stride1同等信息保留。

Heavy evidence来自当前selected native tokens，在parent内联合编码，然后空间slot1/4/8池化。空slot标invalid，全batch空slot被移除，不会凭空制造evidence。B-full仍经过同一query、receiver、regularTIA，因而确实可测这套架构相对统一dense的总损失；不能拿它冒充官方结构。

五种receiver不是仅换标签：rank插值用排序索引，physical插值用物理时间，concat-scatter把evidence归入最近query并MLP融合，timestamp attention加入时间偏差，support attention使用区间并集距离、覆盖时长和空间信息。`[10,11]∪[30,31]` 对20的距离为9、并集时长2，不把中间20秒当可见；该CPU检查通过。null/no-evidence路径实际存在；无有效证据时保持null update，no_null是取消局部无证据限制的有意对照，仍需保证全空输入数值安全。

slots和receiver layers改变容量、attention大小、训练/推理成本；no_null/coarse_overwrite改变信息注入机制；feature_l2实际上是单位范数化，见I11。F05分别训练不同receiver，不等于“固定同一evidence计划”的因果实验；所需D02尚未实现。I04/I05修复前，不能将B的好坏归因于正确的全窗口时间支持或源图空间几何。

### 5.3 C：互斥、回写和非零coarse成本成立；预算词义必须保留

C按2×2 native区域在每个slice选择1个coarse或4个fine，当前10×10网格恰有25组；selection控制互斥，不把同一区域的coarse和fine同时当额外证据。coarse由对应state聚合，位置/可选scale与projection进入Heavy；回写的是coarse更新delta，分发给其覆盖native位置，fine直接写updated状态，之后规则TIA。all-fine有原生全量极限，all-coarse仍有25 token/slice的Heavy，绝不是零计算。

注册固定预算q表示mixed-token比例：p=(4q−1)/3，因此q=.5对应p=1/3，计数 `25(1-p)+100p=50`。这是与原始VARIANT_RULES一致的转换，不应编造成错误；但把图中的q=.5写作“50% fine”或“50% Heavy FLOPs”是错的。token数和含二次attention的MAC不是同一个预算量；动态目标又按traced Heavy MAC约束，必须分别报告请求值与实测值。224px的49/196只限显式224消融；I03不能通过引用历史224登记掩盖。

C的source-limit不涵盖I01导致的detector mask不一致；group7不属于C合法2×2注册，F04的A配置则必须按I09拒绝。

### 5.4 Router、critic、反事实

[`geosparse_ext/routing.py:9–31`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/routing.py#L9-L31) 的有序Plackett–Luce log-prob沿采样顺序计算；执行采用native位置顺序，两者分开。合成n=4,k=2所有有序选择概率总和为1。局部配额在可选域内计算条件概率，动态K把categorical预算概率和选位置概率相加，不是把动态预算静默改成固定。warmup、探索与无学习选择有实际分支；独立探索分支不训练actor，等价于只对learned分支求梯度，不能仅凭没有额外branch log-prob就指为遗漏。

成本最小化actor是正号 `(cost-baseline).detach()*log_prob`，梯度下降降低高成本动作概率；合成检查把概率.5降至约.450166。baseline来自行动前Scout context；actor中的baseline和reward detach，critic有单独回归损失；任务loss没有被统一detach切断。上下游该有的梯度链存在。critic依赖当前cheap输入不是行动泄漏。

当前acquisition probe取未选候选，signed gain为 `L(S)-L(S∪{a})`，没有把retention/swap当同一标签。[`geosparse_ext/detector.py:97–130`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/detector.py#L97-L130) 反事实重新运行真正受影响的模型路径，使用同一Heavy RNG、保存/恢复buffer（含head loss_normalizer）并无梯度执行，保留当前模型、当前集合的条件依赖；不会每个probe额外更新EMA。基础路径/反事实的状态还原与I06的独立验证context不是同一个问题。

额外probe有trace/开销记录，但不能据此声称已计量全部训练FLOPs、解码、能耗；完整训练成本比较还缺测量。retention、梯度代理、ST、有硬件LUT的预算、ROI、两轮获取、逐层重路由仍显式BLOCKED，不能因为矩阵有名字就声称实现了这些估计器。

## 6. 评价、诊断、统计、图形与硬件

### 6.1 全量验证与best

训练内validation直接用test管线的完整窗口集合，检查视频集合，保留空预测视频，官方NMS与官方evaluator计算完整tIoU。保存checkpoint→测EMA→严格大于更新best的顺序正确；tie保留先到的较早checkpoint，不能指为off-by-one。实际验证失败会写失败凭证；I08的问题是部分失败后的状态准入，不是所有失败都隐瞒。

独立prediction/export用训练源码snapshot构造冻结模型，measurement代码与model source分开并校验；这是正确边界，不能把测量修补说成重新部署模型。正式源模型/权重一致性检查存在，但I02阻断split消费者，I13缺hardware的具体参数身份。

边界指标不仅是匹配成功样本的MAE：实现有漏检惩罚/召回类指标，同时保留conditional matched误差；空GT按NA处理。short-action阈值从训练数据分位数冻结。风险阈值拟合的旧split接口必须修复，不能把测试GT拿来补救。不能只展示conditional边界MAE而省略miss与recall。

### 6.2 117项diagnostic：没有执行实现

|诊断|当前状态|不能写成什么|
|---|---|---|
|D01|显式BLOCKED；收益校准/有限菜单loss-best数据与执行缺失|不能称全局mAP Oracle，也不能由Oracle存在推断cheap可预测|
|D02|显式BLOCKED；同证据计划receiver对照未完成|不能把分别训练的receiver差异全部归因于support geometry|
|D03|显式BLOCKED；同K直方图shuffle未完成|不能通过同时打乱位置、模型或总成本来声称“预算异质性贡献”|
|D04|显式BLOCKED；连续loss/边界误差交互未完成|不能用AP阈值跳变证明时间×空间交互|
|D05|显式BLOCKED；mask模式与真实Heavy内容干预未完成|不能把mask可预测性当重型内容有用|
|D06|显式BLOCKED；编码依赖/缓存机制未落地|不能声称缓存数值等价或已获得缓存加速|

这里“未完成”不是“已证明无效”。现有报告明确WAITING_IMPLEMENTATION是正确做法；不把它当已完成诊断的凭证。

### 6.3 可视化：并非只有计划，也并非全部完成

实现已有逐视频/窗口selection导出、native token/原图帧可视化、实际支持与ROI字段、预算分布、训练和验证曲线、边界/短动作切片、latencyCDF、p50/p95、throughput、显存及Pareto，并有缺数据展示。selection_export支持显式中间checkpoint；`--videos`是部分定性导出，不能冒充正式full split。

[`geosparse_ext/figures.py:390–414`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/figures.py#L390-L414) 和 CLI 有 **`--single-seed`**：seed0模式required=(0,)，SD=None并标feasibility；不能批评“绘图只能等三种子”。默认三种子主图不应直接用于当前阶段。200个计时重复也不是200个训练seed。

收益校准图、上述D01–D06机制图仍缺实现/数据。[`geosparse_ext/figures.py:471–497`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/figures.py#L471-L497) 明确WAITING_IMPLEMENTATION。空结果/缺数据应继续如实显示，不用占位随机点。当前主要绘图错误是I03的成本分母和I13的checkpoint配对，不是所有图都伪造。

### 6.4 成本、延迟、单位、失败处理

真实稀疏trace可计QKV/attention/MLP；实际算子counter统计conv/linear/mm等，并列出未支持算子。LN/softmax/激活/插值/部分融合算子等遗漏时是计数下界，而不是完整模型FLOPs；MAC与FLOPs约2倍的换算也只适用于对应乘加范围，不能对任意遗漏总量乘2冒充完整成本。

benchmark有CUDA同步，隔离进程检查在开始、每25次及结束执行；50 warmup、200 repeats，batch1/8/32，reference/optimized配对。测量前在同模型/确定evidence下做数值比较；小型CPU梯度测试不能替代生产AMP/随机训练的全部梯度等价。

|边界|包含|不包含/不可外推|
|---|---|---|
|device_model|已经归一化并驻GPU的VideoBatch→模型proposals，含实际Scout/TIA/packing|CPU解码、H2D、CPU NMS|
|decoded_tensor_to_output|CPU RGB batch→传输、归一化、模型、window NMS|视频打开和解码|
|encoded_video_to_output|视频open/解码/PTS/transform/collate→传输/模型/window NMS|整个长视频所有窗口及最终跨窗口merge的总墙钟；计时为warm OS cache|

ms是**每batch的768位置窗口延迟**；throughput是窗口/秒，`B*1000/mean_ms`；峰值显存原始bytes、图中GiB需除2^30。p50/p95来自计时样本，不能当训练随机性置信区间。源视频子集和window索引有记录；100视频采样不是整个211视频推理墙钟，也不是由视频时长严格分层抽样。

[`geosparse_ext/benchmark.py:142–156`](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/55d5429e66b64198ea74e4946f94de3098ebaa46/geosparse_ext/benchmark.py#L142-L156) 明确保留每case OOM/FAILED并在任何case失败时使整个benchmark失败。**不存在“benchmark有OOM也总是completed”的代码错误。** 当前没有正式完成的硬件结果，不能给出真实加速倍数。A训练在4090、其他路线在A100，也不能把这两个GPU上的绝对延迟拼成方法排名；可把已有checkpoint在同一类GPU上单独计时，不需重训。

## 7. 协议、实现与凭证的矛盾表

|主题|用户最新要求/PROTOCOL|历史方案或manifest|实际代码|运行凭证/图表边界|裁定|
|---|---|---|---|---|---|
|主路线|A预指定，B/C平行|原论文框架同样承诺平行|focus五train无mAP/Oracle依赖门槛|A已运行、B/C排队|未发现按成绩门控；完整挑战路线仍未实现，不能称承诺已兑现|
|训练视频|全200，diagnostic别名|旧180/20 holdout已撤销|生产split已修，消费者internal_dev遗留|训练内全量验证≠独立evaluate|生产者正确，I02当前仍错|
|输入协议|768/160/grid768/stride4原管线|旧224/grid384等过时|实际像素主协议正确；mask逆变换不正确|A真实shape凭证匹配|I01不是旧bug复述|
|TIA|Heavy48parents、TIA384|部分旧注释误称clip-local|A/C正确，B新regularTIA8|A/Csource-limit通过不覆盖B|I04当前仍错|
|成本|160 native100、coarse25|历史49/196仅224适用|训练cost已改实际shape；analysis分母仍14×14|全量160会显示0.446759|I03当前仍错|
|验证节奏|O42..60；Geo每5|历史final59/best-dev过时|当前O wrapper正确；Geo12次|发布权重推理不是train分支|10vs12有偏差，不能称同选点机会|
|保存epoch|60轮完成、best导出|full[5,20,40,59]；focus[4,9..59]|训练执行保存集合不同，诊断仍旧索引|诊断现BLOCKED|I10当前契约矛盾|
|seed|只推进0|full保留1/2，历史三seed主图|focus0；图有single-seed|无1/2静默提交证据|不增加seed；seed0不报mean±SD|
|注册数量|1545任务/204组合|原1538矩阵过时|125组合静态接受，79阻塞；117诊断全阻塞|5个预检≠125/204已跑|不准写全矩阵完成|
|L2融合|必须与登记机制一致|原F10特征L2一致性|只有normalize|暂无其正式结果|I11，不能据此讨论L2对齐优劣|
|官方精度|发布权重只复测|官方表71.14|wrapper正确调用独立入口|71.1387948970真实推理receipt|保留；不可借作Geo或独立训练成绩|
|最终完成|完整12次选点与模型/测量配对|ID与completed字段易被当充分条件|部分选点失败可DONE；hardware缺checkpoint身份|没有实际污染图证据|I08/I13需补门槛|

迁移中确实修复了旧180划分、以PTS重采样替换官方frame indices、旧cosine/batch/grid、checkpoint只保存可训练状态、部分反事实buffer问题和官方核心改动等。不能把这些已修复问题继续列成当前缺陷。反之，消费者别名、分析分母和验证restore等外围遗漏不能因为核心恢复官方就忽略。

## 8. 论文贡献与H1–H6：能写到哪里

### 8.1 原始工作核对，不按名字否定

CoDA原论文第3节已经有dense cheap adapters处理全部token、冻结重型attention/FFN只处理selected token、收集/回写的框架；A不能宣称这些原语首次提出。GeoSparse可能新增的是TAD原生时间/支持约束、当前模型signed acquisition与受控系统验证，但这些组合是否有用仍待证据。[文献1]

Coarse-Fine原文是Grid Pool动态时间下采样和Fine全时间分支的多阶段融合，不应简单等同Geo B，也不能把“粗细双分支+时间重对齐”本身称新。[文献2]

mTAN已经使用连续时间表示与attention把不规则观测映射到参考时间表示；timestamp embedding/cross-attention不是本工作首创。[文献3]

MSViT已有按区域选择mixed-scale tokenization；C的1 coarse/4 fine这一基本想法不能脱离先例声称首创。原生重用预训练块、互斥回写到规则TIA的细节和TAD收益才是应验证的增量。[文献4]

PBD原文是逐步选择/移除block，随后进行跨深度feature和prediction对齐的压缩过程；当前均匀DEPTH和BCR不等于PBD，teacher-free inspired变体也不能叫原算法复现。用户禁止独立Dense TAD Teacher，应该如实报告这一对照边界，不偷偷引入原论文未获授权的训练阶段。[文献5]

AdaTAD原文TIA是带时间depthwise卷积的adapter、插入backbone并利用frame representation；具体768/160/B、48parent、384TIA、训练节奏以固定官方源码为准，而不是论文其他尺度的最高mAP。[文献6]

### 8.2 四项必要证据必须分开

|命题|现有证据|当前裁定|
|---|---|---|
|好稀疏配置存在|A/C机制和若干precheck；无当前协议完整Geo成绩|未证明；不能从官方发布权重或有限菜单loss-best推出|
|cheap能预测任务收益|PG/acquisition代码存在；D01校准/符号准确率/regret未执行|未证明；有好的选择集合不代表cheap可辨识|
|receiver能利用证据与几何|五机制存在，B-full可测架构税；D02缺失，I04/I05未修|未证明；分别训练mAP差不等于几何因果贡献|
|硬件真正加速|真稀疏Heavy和同步计时实现；无正式配对hardware结果|未证明；MAC节省不能替代端到端p50/p95/吞吐|

H1需要F01/F02/F07/F08+D01；H2需要匹配成本/容量的规则状态比较；H3需要同evidence计划、gap变化和几何无误；H4需要D03只shuffle预算K、同模型重新选位置且总成本一致；H5需要连续loss/边界支持的时间×空间交互；H6需要同GPU、同窗口、同权重、包含解码的配对时延。**当前H1–H6均没有完成支持链，不等于都被证伪。**

### 8.3 选择偏差与可发表措辞

全测试集EMA best是用户明确授权的选择策略，不擅改为heldout；但被选择的数据不能再称独立未触及测试。O有10次、Geo有12次且时间范围不同，不能用一个统一减数“校正”机会偏差，因为各epoch指标高度相关。seed0只能称单种子结果，不能称均值±标准差、稳定显著优于或跨随机性稳健。

不改已授权训练、不重复已有实验的补充报告：保留原规则best为登记结果；增加固定**完成60轮EMA**；再增加只在共同已验证节点 **完成50与60轮** 中选出的best，两边均2次机会。这些节点均在各自既定保存计划中，不需要新增训练；当前快照尚未到达50/60轮，并不已有这些checkpoint。对实际已经存在的完整预测直接复用，仅补缺失推理。明确这是本次审查提出的补充分析，不伪装成原先预注册。

还可公开所有逐tIoU和完整验证曲线；按视频配对bootstrap只描述给定权重下的数据采样不确定性，不能当seed方差，也不能消除test-best选择偏差。不同GPU训练不自动使mAP不可比较，但延迟必须同硬件、同边界、同窗口。

可支持的最强措辞：**“在固定官方基座上已实现A/B/C核心执行与部分注册变体，A/C编码器稠密极限在指定条件有数值/相关梯度证据，正在进行seed0可行性实验；已完成官方发布权重全量推理复测。”** 当前不能写“全矩阵完成”“三条路线已证明有效”“无损加速”“官方独立训练已复现”“首次提出dense-light/sparse-heavy”“支持信息完全恢复”。

## 9. 最小修复顺序、保存与重跑边界

### 第一层：会改变训练目标或状态的修复

先处理I01；用现有训练数据长度/裁剪日志确认实际是否出现奇数有效尾部，不能用输入总长768为偶数排除。再修I06完整状态恢复。B/B-full/cheap进入正式新版本前解决I04的规则时间域，并锁定MMAction增强与source几何记录I05。对正在运行的A，本审查没有停作业；保留其全部产物和原M代码身份，由作者依据实际触发决定是否继续作为旧实现工程轨迹。

I01若已进入训练loss，不能只改评估mask后把旧权重叫成“在修正目标下训练60轮”；要得到这一更强声明，受影响的那一次训练确需从同一识别预训练重新训练60轮。快照只显示A已经训练，不应据此重跑五条乃至全部612条。B/B-full/C/dense在快照PENDING，模型修正并不意味着丢弃已完成训练。若继续混合版本恢复，必须称协议修订轨迹，不能伪称整程单一版本。

I06的漂移本身尚未证明大精度损失，先用保存state量化，不能因微小roundoff要求无差别重跑。任何修复后的模型都应单独有源码commit；测量代码更新不应改写原模型来源。

### 第二层：只影响评价与产物契约，不重训

修I02所有别名消费者与物理subset/root；修I08下游完整选点门槛；用既有epoch补失败验证并重算best；修I13具体checkpoint配对。I03从已有trace重算参考成本/图表。保留正式发布权重复测，不因测量层修复再跑一次已成功的同推理。

### 第三层：解除明确误准入、完成最小机制证据

修I09交叉validator、I11控制命名/目标、I10诊断epoch与manifest契约。继续将真正未实现变体标BLOCKED；不要把所有79个阻塞立即解锁以制造“覆盖率”。优先用已经训练的checkpoint完成登记的D01/D02/D03等离线/诊断机制，使A/B/C判定可解释；这不是增加seed或另训一遍所有路线。

### 第四层：测量与调度

修I12 UUID映射，并验证checkpoint哈希配对和reference/bucket的数值/梯度条件；将已有各路线checkpoint安排到同类GPU上计时，模型/decoded/encoded边界分开，50/200、batch1/8/32不减。任何OOM和失败继续保留，不能通过删合法case美化表。使用已存在的队列唯一归属、source快照、capability和恢复记录；A仍属N16，B/C/Bfull/dense属A100，seed1/2保持注册但不提交。

### 产物保留/撤销判定

|产物|处理|
|---|---|
|官方released EMA推理71.1387948970%|保留为独立推理复测，不挪作训练/新方法结果|
|官方原版seed0独立训练|继续按既定入口/节奏；没有证据要求重训|
|W旧180/224/grid384/cos60/batch8|维持撤销，不能混入新主表|
|当前M的A中间checkpoint|保留原始源码/优化器/EMA/RNG与触发证据；尚不能认定为修正后最终协议结果|
|受I01影响训练的checkpoint|可诊断/展示旧实现；修正目标下的正式60轮结论须重训实际受影响的训练，而非只重测|
|仅I02/I03/I08/I10/I13影响的结果|重做缺失/错配测量、重算图表，不重训|
|当前不存在的B/C最终精度、硬件加速、D01–D06结果|不编造、不撤销“从未存在”的结果，继续标未完成|

## 10. 本次已执行的CPU最小复现

`reproduce_audit.py`用AST只提取必要类/函数，在CPU合成数据上运行；不import仓库训练入口，不读取真实资产，不训练。命令：

```bash
python reproduce_audit.py --source /path/to/GeoSparse_Pro_Review_20260908 --output cpu_probe_results.json
```

|检查|实际结果|解释|
|---|---|---|
|767帧有效mask回插值|错误新增有效index767，输出有效数768|I01复现|
|自然EMA后进入/退出验证|1547/10000冻结值未恢复，max abs 4.07696e-5|I06复现；不是mAP测量|
|160px、group7|静态接受，atoms抛ValueError|I09复现|
|160 full Heavy参考比例|0.44675901448662364|I03公式复现；不是实测硬件成本|
|区间并集gap|distance=9，duration=2|此处正确|
|n4,k2有序PL|全部有序概率和=1|此处正确|
|高成本动作actor符号|概率.5→.450165987|成本最小化方向正确|
|局部/全batch normalizer代数例|.1098901099 vs .2173913043|说明I07不等价，不代表真实模型loss数值|

可复现脚本、输出、204组合和1545任务CSV及manifest差异一并交付。并未把这些合成检查数量加到作者GPU测试数上宣传覆盖。

## 11. 行动结论（300字以内）

先修最终检测mask、EMA验证恢复和B的规则TIA，再决定实际受影响的A训练是否须重启；不要重跑全部矩阵。并行修完internal_dev消费者、完整best门槛和checkpoint—latency配对，利用已有epoch补测而非重训。重算160px成本分母，保留官方发布权重复测。继续seed0的A/B/C平行推进，先获得同训练器dense对照、固定60轮与共同50/60轮选点报告，再做同GPU配对计时和登记诊断；此前不宣称方法有效、无损或加速。

## 文献与外部实现

只用一手论文/官方源码限定已有技术，不将作者精度或速度搬入当前主表。

1. [CoDA，Conditional Adapters，原论文第3节](https://arxiv.org/html/2304.04947v2)。
2. [Coarse-Fine Networks，原论文第3节](https://arxiv.org/html/2103.01302v2)。
3. [mTAN，Multi-Time Attention Networks](https://arxiv.org/abs/2101.10318)。
4. [MSViT，Dynamic Mixed-Scale Tokenization](https://arxiv.org/abs/2307.02321)。
5. [Progressive Block Drop，原论文](https://arxiv.org/html/2503.16916v1)。
6. [AdaTAD，原论文第3节](https://arxiv.org/html/2311.17241v2)。
7. [MMAction2 v1.1.0 ImgAug官方实现](https://github.com/open-mmlab/mmaction2/blob/v1.1.0/mmaction/datasets/transforms/wrappers.py#L245-L425)。此版本用于验证标准ImgAug行为，不冒充服务器依赖锁。
8. [官方OpenTAD固定基座](https://github.com/sming256/OpenTAD/tree/346d09d19e2091372cec48172dbe40f7b28bdee6)。
