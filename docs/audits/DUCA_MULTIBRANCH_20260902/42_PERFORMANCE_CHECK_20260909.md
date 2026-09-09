# 六路线性能复核与监督记录

远端检查：2026-09-09 08:57:18及09:08:25 CST。响应用户“检查实验性能”，继续维护完整实验目录。
本轮没有模型修改、调参、训练或独立评测重提。已完成作业不重复运行，历史失败和缺项不删除。

## 最新实测性能

单位为百分比。Avg-mAP为官方THUMOS14 tIoU 0.3/0.4/0.5/0.6/0.7的平均值。
最佳成绩来自每5轮完整测试的EMA选择；终轮为完成60轮、6000次成功更新的epoch59 EMA。

| 路线和具体实验 | 测试集最佳Avg-mAP（完成轮数） | 独立终轮Avg-mAP | 终轮mAP@0.7 |
|---|---:|---:|---:|
| H65：均匀选择384帧 | 63.9646（50） | 63.8596 | 42.0630 |
| H65：动态选帧，四相分配关闭 | 64.2632（60） | 64.2632 | 43.0885 |
| H65：128+64+64+128四相分配开启 | 57.5081（55） | 57.4857 | 34.2285 |
| ET：完整Transformer，近似关闭 | 62.5093（35） | 61.7587 | 42.7085 |
| ET：共享低秩残差近似开启 | 55.3617（40） | 54.5976 | 33.4394 |
| CT-DP：坐标修复版均匀对照G0 | 本版未采用eval5 | 64.2720 | 42.6802 |
| CT-DP：坐标修复版双相选帧G1 | 本版未采用eval5 | 61.6740 | 39.9154 |
| Evidence：历史A1证据恢复消融 | 历史终轮协议 | 51.5968 | 27.8353 |

当前修复版对比最高64.2720%，没有超过65.6%。不混入原版AdaTAD历史参考、旧修改版REF-D768或不合格训练成绩。
H65四相开启相对关闭：终轮低6.7774个百分点、各自最佳低6.7550点。ET开启近似：终轮低7.1612点、各自最佳低7.1476点。
这些都是已完成训练的负结果，不是“未分配GPU”“没训练完”或“还没有做评测”。

## 成绩真实性与边界

- H65训练1280127/1280128/1280129，ET训练1280118/1280117全部COMPLETED(0:0)。每项60轮、12次完整测试，共60份周期记录，最佳与协议记录未变。
- H65独立1281175_0/1/2及ET独立1281049_0/1均COMPLETED(0:0)。五项独立终轮的Avg及五个tIoU指标，与训练内终轮逐项完全相同。
- 09:08读取八份现有独立metrics原文，复算各自既有evaluation_sha256/receipt_sha256均通过，身份和数值未变。F/A6缺失结果保持缺失，不伪造空收据。
- H65训练629162cd、评测4af4ad96；ET训练9a346f0d、评测edfbd7e5。远端源仍exact/clean；本地身份随目录生成器刷新。
- 08:57核对checkpoint路径、尺寸和mtime均未变，沿用已有实际加载和更新数认证，没有为轮询重复读取大型权重。H65全局成功更新6000与条件参数参与计数分开记录；ET无独立保存EMA计数，保持原FP32控制流证据说明。
- 五项新训练采用TEST_GUIDED_EXPLORATORY_EVAL5。测试集被用于最佳选择，并允许指导后续调参；不能称未见测试泛化、跨种子统计显著或与官方论文公平可比。测试GT不进入模型推理。
- 独立复评一致排除了这两条评分路径之间的差异，不排除共有系统偏差、模型实现缺陷或配方不适配，也不证明整个研究方向无效。

## H65降分的源码线索

本轮在629162cd进行有限的源码核查，不把线索冒称因果定位完成。

1. 开启与关闭配置在模型内仅切换acquisition_policy，训练协议保持匹配。但一个配置开关可以切换多个实际行为，不等于只改变四相配额。
2. budget_calibrated_sampling_rate使用center_scores，包含transition policy和detector utility fusion。semantic_phase_sampling入口却传入actionness_logits/p_action，不传center_scores。
3. 四相路径使用平滑动作性logits及其导数构建core/onset/offset、选择分数和soft slot assignment；真实训练的density_transport_st桥使用这份assignment。
4. 因此，当前开关同时改变用于采样的评分来源和ST梯度路径。不能把全部降分单独归因于“128+64+64+128预算不好”，也不能因为utility在上游算过就声称它已参与四相采样。

源码抓手（相对E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission）：
- configs/adatad/thumos/h65_pro/base_h65_pro_strict60.py:31，h65_pro_eval5_phaseoff.py:1，tests/test_h65_tia_eval5.py:12。
- opentad/models/duca/acquisition.py:2142、2202、2530、2700、3344。
- opentad/models/selectors/duca_online_frame_selector.py:2481。

尚未确定：单一路径导致的分数变化幅度、各少步Adam状态对应的参数名，以及原设计是否要求四相直接融合transition/utility。
下一步应先对照原设计和真实梯度归属，选择最小修复或另立命名的匹配控制；本轮没有盲目把fusion塞入四相、热改远端或重训。
零值梯度与grad为None不同，不用“梯度值为零”直接解释Adam参与步数减少。

## 其余实验与官方基线

- CT-DP：旧G2/G3的56.5234%/57.8489%保留78cde1be身份，不能当作fe1c53db整矩阵新结果。G0-G3几何组均未开启B-AMoD，单独机制矩阵和eval5接入仍未完成；不能继续沿用“没有CTConv”的旧描述。
- BAFDR：710ce8a6的五个终态checkpoint均保留，独立metrics.json仍为0，FULL仅PRECHECK。旧50.93/48.17/53.11/49.44/52.38%更新数不合格，只作诊断。b142有效chunk/ragged修复不等于terminal screen、eval5和生产验证完成。
- Evidence：C0=59.2292%、A2=54.2756%是历史记录，本轮未重新认证；F/A6仍缺独立终态结果。79aea已有基础监督、时间梯度和support接线，但utility、真实two-view、cycle仍未完成，旧权重不代表新修复性能。
- DUCA-Unified：0fce时间轴接线修复已登记；Taylor P0/P1、合法one-swap及H65 retention/transition仍未完成，没有完整正式性能。
- 原始官方AdaTAD：历史原始日志68.73%不等于修改REF-D768的67.58%。原配方5995更新不追溯套用新6000合同；来源、预训练和数据身份补证及独立评测仍未闭环。H65均匀384和ET关闭对照不能冒充原版AdaTAD。

## 失败、资源和监督

历史30+60匹配H65作业1280314/1280316仍FAILED(1:0)。08:57再次完整读取stdout/stderr，仍是epoch16 AMP replay之后p_action非有限；首个污染源尚未定位。
原“查找90轮H65实验配置”任务负责这两项，保留旧job/log/checkpoint，未接管、删除检查或盲目重提。四个相关修复分支git ls-remote成功但仍无对应GitHub ref，不冒称已发布。

09:08公共可调度节点未分配GPU为72/200，账户队列2 RUNNING、1 PENDING，实验盘120.45GiB可用。
队列1278774/1281300/1281271不是本目录五项已完成训练或五项已完成独立评测；本轮未核实新1281300的WorkDir，不擅自归属或操作。
这些资源余量不保证个人即时配额；当前完成实验的低分及其他路线的机制缺项不能解释成资源排队。

历史BAFDR1267920/1267921继续只读。本轮没有Pro咨询，没有声称既有ixBrowser Computer Use阻塞已解除。
automation_update工具搜索仍无结果，未修改自动化配置；分钟服务由目录生成器读取，其plan/BLOCKED状态不能称自动正式提交。

目录freshness与仓库C3 focused tests合计28 passed；三个相关工具py_compile通过。22项唯一身份、60次周期记录、五项独立终态VERIFIED，以及五项新训练和四个相关修复本地exact/clean检查通过。
09:16刷新时分钟服务回执年龄50秒、ACTIVE，但dispatcher仍plan/BLOCKED、entries=0，不是自动正式提交。完整结构化记录见[42号JSON](42_PERFORMANCE_CHECK_20260909.json)。
