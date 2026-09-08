# 22:15 心跳：最佳权重保持正确，历史匹配实验开始运行

真实触发：2026-09-08 22:15:36.280 CST。首次SSH连接被关闭，原配置重试于22:20:10恢复；完整远端采样22:21:55-22:22:24，补充日志核查22:23:56。未把连接失败当模型失败。

## 五项训练的最新完整测试

均RUNNING，ET运行约5小时、H65约4小时56分；尚无epoch59终态。测试集为211视频、792窗口。

| 实验 | Job | 最新测试轮数 | 最新Avg-mAP | 当前最佳Avg-mAP（轮数） |
|---|---|---:|---:|---:|
| Transformer低秩残差近似开启 | 1280117 | 45 | 55.2871% | 55.3617%（40） |
| 完整Transformer匹配对照 | 1280118 | 40 | 62.4262% | 62.5093%（35） |
| 均匀抽取384帧，整窗时间适配器 | 1280127 | 30 | 60.7207% | 60.7207%（30） |
| 动态选帧，关闭四相分配 | 1280128 | 30 | 60.2283% | 60.2283%（30） |
| 动态选帧，开启四相预算 | 1280129 | 30 | 52.6773% | 52.6773%（30） |

ET新增两次测试都略低于此前最佳，所以best仍保持ON第40轮、OFF第35轮；不是按最后一轮覆盖best，也不改用mAP@0.7选模。两份未变实际EMA复用21:31认证。H65三份新第30轮best实际EMA有限，source/seed/epoch/指标匹配。总计35份周期收据及best/protocol既有自哈希核验通过。

同第40轮，ET ON55.3617%对OFF62.4262%，差-7.0645点；同第30轮，H65四相比相位关低7.5511点、比均匀低8.0434点。仍为中期、单种子测试指导结果，不作终态裁决。

ON已完成50轮训练批次，正在第50轮完整测试，stderr推进到362批；OFF日志进入epoch44。H65审计完成30/31/32轮，optimizer/scheduler/EMA/DUCA schedule一致为3000/3100/3200。

## 数值与失败

H65累计AMP skip为3/9/6。相位关在epoch29 batch74出现连续4次重试，scale降至256后恢复，随后完成该epoch并产生第30轮正式测试。该组最新审计3100成功更新，replay_exhaustions=0；这是需保留的数值波动，不是已确认训练失败，也不据此热改或重启。

本轮未发现新增Traceback、非有限cost、OOM或磁盘写入失败。首次SSH退出255的完整日志保留在C:/Users/skywalker/.fastctx/jobs/j-plba8b/output.log，重试未改变密钥、网络设置或模型。

## 历史H65匹配实验

- 另一任务“查找90轮H65实验配置”的PRECHECK1280306为COMPLETED(0:0)。已读原始stdout/stderr：28项focused tests通过，control/signed/fulltia均完成真实GPU三batch训练。没有重复跑预检。
- signed1280314与control/fulltia链1280316于22:01:16获得资源，当前均RUNNING，日志到epoch3。链内目前仅观察到control，尚未看到fulltia正式开始，不能称三臂已同时开训。
- scontrol已经查不到完成的1280306；sacct及原始日志仍证明完成，不能把控制器清理视作作业失败。
- 96e5b3a5远端exact/clean；共享30轮Stage-1 epoch29 EMA再做60轮Stage-2，当前协议仍以终态为主，不与本任务strict60测试选best混为同一比较。本监督未接管或重提。

## 其余路线与代码身份

CT G0/G1为64.2720%/61.6740%，Evidence旧A1为51.5968%，与上次一致；BAFDR新独立metrics仍0。CT、BAFDR、Evidence已认证终态文件stat未变，不重复加载旧大checkpoint。Evidence F/A6独立receipt仍缺，CT后续eval5仍待接入。

四个相关本地修复分支仍clean、SHA未变，GitHub查询成功但没有对应分支。BAFDR padding、Evidence基础监督/时间条件/support、Unified时间轴已经有代码修改；不能写成完全未动工，也不能说生产验证或完整机制已完成。剩余terminal screen/eval5、utility/two-view/cycle、Taylor P0/P1/合法one-swap/H65 retention-transition继续保留。

原始官方AdaTAD来源补证和独立复评仍待完成；不以修改版REF-D768或本轮均匀384替代。旧分数不迁移为修复版成绩。

## 资源与本轮动作

22:22公共可见未分配GPU7/200，账户11 RUNNING、4 PENDING；实验盘143.1GiB。公共余量不保证个人配额，未取消其他项目腾资源。分钟回执年龄约27秒，ACTIVE但dispatcher仍plan/BLOCKED，只读轮询不等于自动提交成功。

本轮仅监控、保留SSH短暂失败及数值重试记录、核验新增曲线/相关任务准入，并更新目录。没有新模型修改、超参数尝试、Slurm提交、取消、重复准入或Pro咨询。历史BAFDR1267920/1267921不操作。

新实验继续标TEST_GUIDED_EXPLORATORY_EVAL5；全部曲线与尝试保留，最佳EMA和终态EMA分开，测试集复用如实披露，不称独立泛化或与官方论文公平可比。

结构化证据：[33_HEARTBEAT_EVIDENCE_20260908_2215.json](33_HEARTBEAT_EVIDENCE_20260908_2215.json)。

