# 04:56 心跳：ET独立终态评测已启动

触发：2026-09-09 04:56:02.214 CST。六路线实际采样05:04:40；新预检终态05:11:09；
正式评测05:14:19及05:21:36复核。本轮使用37号最新记录，不复述触发提示中的旧RUNNING状态。
没有把未执行的历史触发补记为检查。

## 本轮实际推进

ET-TRC原独立评测入口仍绑定74473c27及旧权重目录，不能用于新9a346f0d训练。
本轮在独立分支[codex/zoomtoken-ettrc-eval5-terminal-20260909](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-ettrc-eval5-terminal-20260909)
修复为[edfbd7e580bbd903db543e1be67169eff6d9125f](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/edfbd7e580bbd903db543e1be67169eff6d9125f)，已推送。

- 本地工作区：E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_eval5_terminal_20260909。
- 新远端clean源码：/data/run01/sczc063/yuzibo/projects/ettrc_eval5_terminal_edfbd7e5。
- 新输出：/data/run01/sczc063/yuzibo/experiments/ettrc_eval5_terminal_edfbd7e5。
- 仅修改4个评测工具/脚本/测试文件，opentad/和configs/与9a346f0d差异为空；未改训练代码、配方或旧checkpoint。
- 绑定实际OFF/ON权重路径、原始protocol.json的SHA/seed/测试集使用声明，并校验实际label-free dataset的211视频/792窗口。
- 收据明确INDEPENDENT_TERMINAL_EMA和测试集参与选模，拒绝旧目录、错误臂、错误协议、部分或有标签推理。
- 不新增独立EMA计数伪证；明确ET没有存储该计数，6000次调用依据FP32成功更新控制流。

本地34 tests、入口py_compile、bash语法和git diff --check通过。
已推送增量bundle同步到新detached worktree，远端exact/clean的34 tests通过，没有远端热改。
新双GPU终态推理PRECHECK 1281042_0/1分别于05:11:07/05:10:46 COMPLETED(0:0)，
均有ETTRC_EVAL5_TERMINAL_PRECHECK_OK，实际加载原epoch59 EMA并完成推理；非性能结果。
此前训练准入1280077/1280078没有重跑，也没有重新训练。

## 新独立评测作业

在两项PRECHECK实际通过后提交完整评测，各2GPU/global batch2/FP32/seed4407：
没有未满足的Slurm依赖，dependency为空，前置通过证据是1281042两项终态与原日志。

| 实验 | Job | 05:21实际状态 | 预期收据（相对新输出根） |
|---|---|---|---|
| 完整Transformer对照（OFF） | 1281049_0 | RUNNING，已加载epoch59 EMA，推理226/396批 | formal/off/gpu2_id0/official_eval/metrics.json |
| 低秩残差近似开启（ON） | 1281049_1 | RUNNING，已加载epoch59 EMA，推理184/396批 | formal/on/gpu2_id0/official_eval/metrics.json |

两项05:13:04开始，无新fatal，尚无独立metrics。不从中途推理推导mAP或加速收益。
stdout/stderr为新输出根logs/formal_1281049_0.out/.err和formal_1281049_1.out/.err；
预检日志同目录precheck_1281042_0.out/.err及precheck_1281042_1.out/.err。
结束后核验原始收据、官方evaluator、源/权重/完整population与训练内终轮差异，再升级结果资格。

## 五项完整训练成绩

05:04复核五项均COMPLETED，60份周期、5份best及5份协议收据有效，best与未四舍五入最大值及最早同分规则一致。
每项12次完整测试，211视频/792窗口。终态/best文件stat未变；训练源629162cd和9a346f0d仍exact/clean。

| 实验 | 最佳Avg-mAP（完成轮数） | 第60轮Avg-mAP | 第60轮mAP@0.7 |
|---|---:|---:|---:|
| H65均匀384帧匹配对照 | 63.9646%（50） | 63.8596% | 42.0630% |
| H65动态选帧、四相关闭 | 64.2632%（60） | 64.2632% | 43.0885% |
| H65四相128+64+64+128 | 57.5081%（55） | 57.4857% | 34.2285% |
| ET完整Transformer对照 | 62.5093%（35） | 61.7587% | 42.7085% |
| ET低秩残差近似开启 | 55.3617%（40） | 54.5976% | 33.4394% |

最佳差：H65四相开比关低6.7550点，ET近似开比关低7.1476点；完整训练后的负结果不归因于资源未分配或没跑完。
这些分数尚不是本轮新独立终态评测的结果；H65的独立终态评测仍待补。
H65全局成功更新6000与参数参与次数分布分开，少步参数归属未因此获得认证；复用36号实际权重核验。
新协议TEST_GUIDED_EXPLORATORY_EVAL5使用测试集选模/调参，不能称未见测试泛化或原论文公平复现。
本轮没有超参数调整。原始官方AdaTAD仍须单列来源与独立复评，不用修改REF-D768或新对照替代。

## 其他路线与未解决失败

- CT连续时间几何：fe1c53db/07383274两份独立收据仍有效，均匀64.2720%、双相61.6740%。G2/G3保留旧身份，后续B-AMoD机制矩阵/eval5未完成。
- BAFDR分块多分辨率路由与蒸馏：710ce8a6五份terminal在，新metrics仍0，FULL仅PRECHECK。b142bffa有效chunk接线已修，但真实terminal screen/eval5及生产验证待补；不扩有缺陷旧矩阵。
- Evidence预选阶段证据补漏：旧A1 51.5968%，evaluation_sha256有效；F/A6独立metrics仍缺。79aea08a接线不代表utility/真实two-view/cycle完成。
- Unified系统因素消融：0fcef0bf时间轴接线不等于Taylor P0/P1、合法one-swap与H65 retention/transition，仍BLOCKED_UNIMPLEMENTED。
- 历史90轮1280314/1280316仍FAILED(1:0)，已再次读取stdout/stderr，仍是epoch16 AMP replay后的p_action非有限。最早污染源未定位，原任务负责修复；旧job/log/checkpoint保留，本监督未接管或盲目重提。
- 相关四个修复分支GitHub ls-remote仍成功但未返回ref，不能构造已发布commit链接。本轮未取得新增生产验收或原任务消息工具。
- 历史BAFDR1267920/1267921保持只读，没有取消、修改或重提。

## 资源和后续

05:14公共未分配GPU52/200，账户3 RUNNING/1 PENDING；其中2项RUNNING是本轮ET独立评测，
其余项目/held任务未操作。实验盘121.33GiB可用，公共资源余量不保证个人配额。
分钟监督器由目录生成器实时读取；plan/BLOCKED仍只代表只读轮询，新增评测由本监督任务提交。
没有新训练、模型修改、调参或取消；唯一代码变更为独立评测入口。本轮未咨询Pro，旧Computer Use初始化阻塞不冒称已解决。
当前工具未暴露automation_update/任务消息接口；本轮以最新目录覆盖旧提示观察，没有声称已更新自动化提示或通知原任务。

结构化证据：[38_HEARTBEAT_EVIDENCE_20260909_0456.json](38_HEARTBEAT_EVIDENCE_20260909_0456.json)。
