# 实验性能检查：最佳 checkpoint 与终轮分开报告

本次由用户“检查实验性能”触发，不补记为00:39或01:49心跳。主采样2026-09-09 02:19-02:24 CST，失败日志02:28，H65成绩与更新数补收02:38。

## 新协议实验

均为完整THUMOS14测试集211视频、792窗口；采用官方evaluator。
下表是测试集指导的探索成绩，不是未见测试集泛化或公平官方论文复现成绩。

| 实验 | Job | 当前状态 | 最新完整测试Avg-mAP | 最佳Avg-mAP（轮数） | 最佳对应mAP@0.7 | 第60轮Avg-mAP |
|---|---|---|---:|---:|---:|---:|
| 完整Transformer匹配对照（ET-TRC OFF） | 1280118 | COMPLETED，12次全测试完成 | 61.7587%（60） | 62.5093%（35） | 41.1250% | 61.7587% |
| Transformer低秩残差近似开启（ET-TRC ON） | 1280117 | COMPLETED，12次全测试完成 | 54.5976%（60） | 55.3617%（40） | 33.6320% | 54.5976% |
| 均匀抽取384帧匹配对照 | 1280127 | RUNNING，审计5700更新 | 63.9520%（55） | 63.9646%（50） | 41.7472% | 未返回 |
| 动态选帧、关闭四相分配 | 1280128 | RUNNING，审计5800更新 | 63.7444%（55） | 63.7444%（55） | 41.7527% | 未返回 |
| 四相128+64+64+128选帧 | 1280129 | RUNNING，审计6000更新，epoch59文件已出现 | 57.5081%（55） | 57.5081%（55） | 34.0924% | 未返回 |

两条代码均remote exact-SHA/clean：ET为9a346f0d71e8870ad499bb0a58b9f1824b8904c0，H65为629162cdddbe490d896899a067e9803e237df467。
57份周期收据及五份best的自哈希有效；best与未四舍五入曲线最大值及较早轮次规则匹配。
02:24实际读取三份H65最佳EMA，数值有限、SHA/seed/epoch/指标匹配；ET最佳未变，复用已有实际权重认证。

## 已能确认的性能差异

- ET开启对关闭：最佳低7.1476个百分点，终轮低7.1612个百分点。两项均完成训练，这个退化不能归因于未分配GPU或训练尚未结束；仍不能单凭分差判定整条理论路线无效。
- H65同第55轮：四相比相位关闭低6.2362点、比均匀低6.4439点。存在持续的中期退化，完整终轮成绩仍待返回。
- 本次列出的新修正对比结果尚未超过65.6%。不能用被修改的REF-D768 67.58%或尚待独立复评的原始官方配方日志68.73%来冒充本轮方法突破。

## 终态有效性

ET OFF于01:17:13结束，耗时7:55:42。实际epoch59的337个optimizer state均6000，scheduler last_epoch6000，571项EMA及模型状态有限。
ON于09-08 23:23:38结束、耗时6:02:07；复用此前实际373个optimizer state均6000的认证。
两者checkpoint未存独立EMA更新计数，EMA调用次数来自FP32成功更新控制流，不能写成“读到了独立EMA计数6000”。
新SHA独立terminal evaluator receipt仍缺；当前分数是训练内完整测试EMA评测，不能称为独立终态封存完成。

H65的optimizer/scheduler/EMA/DUCA schedule落盘计数一致；累计AMP skip为3/9/9，replay耗尽均0。
02:19日志检查未发现新三项的fatal；02:38状态仍在推进。四相的终态文件存在不等于已完成独立终态权重认证。

## 其余四条路线

| 路线 | 本次核查结果 | 边界 |
|---|---|---|
| 物理时间嵌入与稀疏路由（CT-DP） | 修正几何G0 64.2720%、G1 61.6740% | 独立收据自哈希有效；两组不是完整B-AMoD机制矩阵 |
| 预选阶段证据补漏（Evidence-Recovery） | 旧A1评测文件51.5968%；F/A6终态存在但独立receipt缺失 | 不迁移到新接线修复SHA，utility/真实two-view/cycle仍未完整落地 |
| 分块多分辨率路由与蒸馏（BAFDR） | 五份6000终态保留，新独立metrics仍0，FULL仅PRECHECK | padding后继修复仍待terminal screen/eval5与生产验证，旧低分不是新修复版最终成绩 |
| manifest驱动的系统消融（DUCA-Unified） | 无正式性能 | Taylor P0/P1、合法one-swap和H65 retention/transition仍缺失 |

## 新发现的相关历史作业失败

另一任务的历史30+60轮H65配对：
- signed1280314：FAILED(1:0)，耗时2:29:52，stage2 epoch16/batch57附近。
- control链1280316：FAILED(1:0)，耗时2:27:54，stage2 epoch16/batch6附近；未见fulltia启动。
- 两者stdout均先发生AMP replay；stderr共同终止于 `opentad/models/duca/acquisition.py:2042`：`p_action calibration check received non-finite values`。
- 本地源码确认是动作性概率或由logits重算的期望值在有效位置出现非有限值，不是普通容差超限。具体数值污染源仍待定位，不擅自归因或删除校验。
- 最后落盘审计1600成功更新，只是epoch15快照，不冒称包含失败前的所有batch；没有终态性能。
- 原任务 `01a08058-1402-7b21-8c17-ec6c1ad522f7` 负责修复和重提。本监督已保留失败记录、读取两对日志，不接管、热改或重复提交。当前工具未暴露任务消息接口，不声称已向该任务发送通知。

日志根：`/data/run01/sczc063/yuzibo/experiments/h65_matched_96e5b3a5_seed3407/logs/`，
文件为 `signed_1280314.out/.err` 和 `control_fulltia_1280316.out/.err`。
下一步必须从保留checkpoint/RNG追首个非有限值及AMP重放状态；独立修复SHA通过focused/remote exact/CUDA/PRECHECK后才新命名空间重提。

## 本次动作

02:28可见公共GPU未分配47/200；02:22本用户5 RUNNING/5 PENDING，实验盘123.97GiB可用。公共余量不保证个人即时配额。
本轮仅检查和更新记录，未改模型、调参、提交/取消Slurm、重复准入或咨询Pro；历史BAFDR1267920/1267921未操作。
终轮与最佳分开保留，继续收齐曲线；原始官方AdaTAD来源/配方对齐及独立复评仍待完成。

结构化证据：[35_PERFORMANCE_CHECK_20260909_0218.json](35_PERFORMANCE_CHECK_20260909_0218.json)。

