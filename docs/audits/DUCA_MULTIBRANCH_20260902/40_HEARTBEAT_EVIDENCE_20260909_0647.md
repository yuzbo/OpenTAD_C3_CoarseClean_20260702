# 06:47 心跳：检查性能并启动 H65 独立终态评测

触发：2026-09-09 06:47:34.967 CST。实际检查：06:49:27（六路线）、07:20:05（修复预检终态）、07:26:41（正式评测与资源）。没有将触发时间当作完成时间。

## 当前性能

五项新训练均已 COMPLETED(0:0)，每项完成 60 轮、12 次完整 THUMOS14 测试（211 视频、792 窗口）。
本轮重读 60 份周期、5 份最佳和5份协议收据，原始自哈希及最佳选取规则有效；与39号指标完全相同。
终态及最佳权重文件未变化，复用此前实际权重检查，不重复加载大文件做轮询。

| 实验（当前修复版本） | 测试集选出的最佳 Avg-mAP（完成轮数） | 第60轮 Avg-mAP | 第60轮 mAP@0.7 | 证据状态 |
|---|---:|---:|---:|---|
| H65 均匀选择384帧对照 | 63.9646%（50） | 63.8596% | 42.0630% | 训练内官方全测试；独立终态运行中 |
| H65 动态选帧，四相预算关闭 | 64.2632%（60） | 64.2632% | 43.0885% | 训练内官方全测试；独立终态运行中 |
| H65 四相预算128+64+64+128 | 57.5081%（55） | 57.4857% | 34.2285% | 训练内官方全测试；独立终态运行中 |
| ET 完整Transformer对照，近似关闭 | 62.5093%（35） | 61.7587% | 42.7085% | 终态独立复评与训练内六项mAP完全一致 |
| ET 共享低秩残差近似开启 | 55.3617%（40） | 54.5976% | 33.4394% | 终态独立复评与训练内六项mAP完全一致 |

H65 四相开启相对关闭：最佳低6.7550点、终轮低6.7774点。
ET 近似开启相对关闭：最佳低7.1476点、终轮低7.1612点。
这是当前实现与配方完整训练后的负结果，不能再解释成GPU未分配或未训完；但没有据此证明整个研究方向无效。
ET 当前为固定anchor/共享可学习低秩代理，不能宣称已实现真正事件触发、精确JVP或无损减少70%计算。

## H65 评测修复与部署

新独立分支：codex/h65-pro-eval5-terminal-20260909。
最终评测提交：[4af4ad96e0a9688488246b2632fad55926ffc4c8](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/4af4ad96e0a9688488246b2632fad55926ffc4c8)。
本地目录：E:/DeskTop/TAD/_duca_fix_worktrees/h65_eval5_terminal_20260909。
远端源：/data/run01/sczc063/yuzibo/projects/h65_eval5_terminal_4af4ad96。
输出根：/data/run01/sczc063/yuzibo/experiments/h65_eval5_terminal_4af4ad96。

原训练仍为629162cdddbe490d896899a067e9803e237df467。opentad/和configs/相对训练提交无差异，不改权重、不重新训练。
评测绑定原训练绝对配置、原epoch59 EMA、训练审计、原数据和预训练身份，保留6000成功更新及官方evaluator要求。
恢复H65专用终态绑定分支，不再索引通用路线才有的gate字段。额外PRECHECK仅使用一个真实推理batch，不生成性能收据。

首个b4776cbd版本的1281146_0/1/2均FAILED(1:0)，07:06在推理前被resolved_config_sha256严格拒绝。
完整三对stdout/stderr及原训练配置日志已读，定位到两个原因：
- 训练用default=str序列化CellCFTrainingProtocol，旧评测器却将dataclass转成dict。
- 启动器使用thumos14/train/test字符串，没有恢复训练时raw Validation/Test路径。
仅修序列化仍不匹配；两项一起恢复后，三项源配置哈希均与原训练审计精确相同。没有关闭绑定来通过预检。
旧源码、job、output、log与checkpoint全部保留，无远端热改。

最终4af4ad96本地43项、远端exact/clean 43项测试通过，py_compile、bash语法和diff检查通过。
真实终态EMA加载和一个推理batch的PRECHECK1281162_0/1/2均COMPLETED(0:0)，stdout均有成功标记。

| 独立正式全测试 | Slurm | 07:26状态 | 收据预期路径（相对输出根） |
|---|---|---|---|
| 均匀384帧 | 1281175_0 | RUNNING，已进入完整推理 | formal/uniform/gpu1_id0/official_eval/metrics.json |
| 动态选帧四相关闭 | 1281175_1 | RUNNING，已进入完整推理 | formal/phaseoff/gpu1_id0/official_eval/metrics.json |
| 四相预算开启 | 1281175_2 | RUNNING，已进入完整推理 | formal/phaseon/gpu1_id0/official_eval/metrics.json |

三项从07:25:29开始，占用3张GPU。日志为logs/formal_1281175_0/1/2.out及对应.err。
截至07:26尚无独立收据，不能将训练内分数预填为独立结果。后续须核验原始收据并比较全部未四舍五入mAP。
07:33:46再次读取三项stdout/stderr和sacct，全部仍RUNNING，已运行8分17秒，推理约完成一半；无Traceback/RuntimeError/OOM，尚无metrics.json。
此次是评测接线修复，不是性能修复或新调参。

## 其他路线与基线

- 连续时间几何与双相采样（CT-DP）：fe1c53db训练、07383274独立评测的均匀对照64.2720%、双相61.6740%收据仍有效。当前修复对比最高64.2720%，没有超过65.6%。G2/G3保持旧源码身份；G0-G3不是B-AMoD完整矩阵完成的证据。
- 分块多分辨率路由与蒸馏（BAFDR）：710ce8a6五个终态权重在，独立metrics.json仍为0，FULL仅PRECHECK。旧50.93/48.17/53.11/49.44/52.38不满足严格实际更新数，不能列为修复版终态性能。b142有效chunk修复仍缺真实screen/eval5生产结果。
- 预选阶段证据补漏（Evidence-Recovery）：旧A1为51.5968%，evaluation_sha256通过；F/A6独立终态缺失。C0=59.2292%、A2=54.2756%为历史记录，本轮未重新认证。79aea接线修复未补齐utility、真实two-view和cycle机制。
- 系统因素消融（DUCA-Unified）：缺真实Taylor P0/P1、合法one-swap及H65 retention/transition，仍无完整正式结果。0fce时间轴修正不解除机制阻塞。
- 原始官方AdaTAD：历史原始日志68.73%与修改REF-D768的67.58%分开；原版来源和独立评测的完整证据仍待补。当前H65均匀384和ET关闭对照不能替代原版AdaTAD，不将不同协议数字直接作公平提升比较。

上述五项新实验使用TEST_GUIDED_EXPLORATORY_EVAL5；用户授权测试集选best和调参，必须披露测试复用。
最佳EMA可能早于6000更新，终态EMA单列。此次没有新超参数尝试，不声称未见测试泛化或原论文公平复现。

## 失败、资源与监督

历史30+60匹配控制1280314/1280316仍FAILED(1:0)，本轮再次读取stdout/stderr：
epoch16 AMP replay后p_action非有限，首个数值污染源未定位。这与五项已完成新训练分开。
原“查找90轮H65实验配置”任务负责修复，本监督不接管或盲目重提，失败和全部产物继续登记。
本轮四个相关修复分支git ls-remote均exit0但无对应ref，仍不构造可用commit链接；线程工具不可用，不声称已向原任务发消息。

07:26可调度节点未分配GPU75/200；账户3项RUNNING均为本次H65独立评测，另1项其他任务JobHeldUser未操作。
实验盘120.65GiB可用，公共余量不保证个人即时配额。未实现机制不能解释成资源排队。
分钟服务由目录生成器读取最新回执；其dispatcher仍是plan/BLOCKED时，不声称已自动提交修复队列。
本轮新增的是监督任务手动提交的独立评测；没有训练、调参、取消或Pro咨询。历史BAFDR1267920/1267921完全只读。
automation_update工具未暴露，不能声称修改了已有自动化提示；旧提示仍须以当前目录与本记录校正。

同时纠正39号JSON中36号证书文件名0319误写，实际为36_HEARTBEAT_EVIDENCE_20260909_0323.json，不改变历史指标。
目录渲染器移除过时的“ET仍等待独立收据”固定段落，改为引用当前实验条目和本轮记录。

目录focused tests为5 passed；22项身份、60份周期指标未变、5项完整训练、2份ET独立收据、3项H65新评测绑定及失败保留检查均通过。
07:31目录刷新时分钟服务回执年龄11秒、ACTIVE；dispatcher为plan/BLOCKED、entries=0，不等于自动正式提交已启用。

结构化证据及原路径见[40号JSON](40_HEARTBEAT_EVIDENCE_20260909_0647.json)。
