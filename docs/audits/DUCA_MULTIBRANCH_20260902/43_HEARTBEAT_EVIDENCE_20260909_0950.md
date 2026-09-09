# 09:50 心跳：成绩未变，补充四相设计依据

触发：2026-09-09 09:50:07.650 CST。实际远端采集09:52:58；H65更新审计补读09:57:10。
优先读取42号最新记录和总目录。提示中的34号、五项训练仍RUNNING等为历史状态，不当实时结果。

## 六路线最新成绩

| 实验 | 测试集最佳Avg-mAP（完成轮数） | 独立终轮Avg-mAP |
|---|---:|---:|
| H65均匀384帧 | 63.9646%（50） | 63.8596% |
| H65动态选帧、四相关闭 | 64.2632%（60） | 64.2632% |
| H65四相128+64+64+128 | 57.5081%（55） | 57.4857% |
| ET完整Transformer、近似关闭 | 62.5093%（35） | 61.7587% |
| ET共享低秩残差近似开启 | 55.3617%（40） | 54.5976% |
| CT-DP坐标修复均匀G0 | 未采用eval5 | 64.2720% |
| CT-DP坐标修复双相G1 | 未采用eval5 | 61.6740% |
| Evidence历史A1 | 历史终轮协议 | 51.5968% |

上述结果均未变化。当前修复版最高64.2720%，仍未超过65.6%；不把原版历史参考或不合格旧训练混入。
五项新训练1280117/1280118/1280127/1280128/1280129全部COMPLETED(0:0)，五项独立1281049_0/1及1281175_0/1/2也均COMPLETED(0:0)。
H65四相开启终轮比关闭低6.7774个百分点，ET近似开启低7.1612点。不能再以资源未分配或训练未完成解释。

## 本轮核验

- 60份周期、5份best、5份protocol原始自哈希有效，身份与完整12次曲线未变；未四舍五入最优值及最早同分epoch选择正确。
- 八份现有独立终态原始收据自哈希有效、身份和所有指标未变。H65/ET五项独立评分与训练内终轮逐项相同的认证继续有效，不重复推理。
- 五项训练和五项评测stdout/stderr完整读取，未发现新致命错误。训练及评测源均exact/clean。
- 五项终态及最佳checkpoint路径、尺寸、mtime与既有实际加载认证相同，未重复torch.load大型权重；BAFDR五个终态也未变。
- H65三项update_audit的optimizer/scheduler/EMA/DUCA schedule均6000；AMP skip4/9/9，replay耗尽均0。全局成功更新与条件参数参与次数不混淆。
- ET实际optimizer6000沿用原终態认证；没有独立保存EMA调用计数，继续明确其FP32成功更新代码路径证据，不虚构独立计数器。
- 临时采集器最初取错JSON键training_updates得到null，按源码改读update_audit后取得原值；只是只读采集字段纠正，不是训练失败或远端热改。

## H65设计解释的补充

主代理读取了初版实现说明全文、身份文件及对应测试，并用git show核查初版内容。
[初版bd862375的A Phase说明](https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/bd8623754a4375c39eb5c941893c606cffbcd6de/docs/experiments/h65_pro_fullmatrix_20260902/01_IMPLEMENTATION.md#L3)明确采用ASFormer logits或logit(p_action)、Gaussian平滑、导数及四相配额。
该文件首次提交于2026-09-02 16:26 CST，不是本轮为解释结果新增的说法。tests/test_h65_pro_fullmatrix.py:324也以显式logits测试此路径。

因此，42号观察到的两条采样评分路径确实不同，但“semantic phase没有直接消费center_scores的transition/utility fusion”本身不能直接认定为接线遗漏。
这与随初版实现提交的A Phase说明一致；它不是对全部原始用户方案的独立验收，也不能证明模型实现整体正确。

29号“只改变allocation”现补充限定：只在模型配置中切换一个acquisition_policy，但实际同时改变评分构造与ST回传路径，不是只改四个配额的隔离消融。
仍不能把6.78点降分全部归因配额；各条件参数的真实梯度/Adam参与归属和降分因果未完成定位。
本轮没有强塞utility fusion、热改模型或盲目重训。

## 未完成实验与失败

- CT-DP：G2/G3旧56.5234%/57.8489%仍属78cde1be，不冒充fe1整矩阵；几何四臂未开B-AMoD，单独机制矩阵/eval5仍未完成。
- BAFDR：710ce8a6五个6000更新终态保留，独立metrics仍0，FULL只有PRECHECK。b142有效chunk/ragged修复不等于terminal screen/eval5与生产验收完成；不放行旧缺陷版或用旧低更新数成绩替代。
- Evidence：A1上述收据有效；C0=59.2292%、A2=54.2756%为历史记录、本轮未重新认证；F/A6仍缺独立终态metrics。新接线修复不能替代utility、真实two-view和cycle缺项。
- Unified：已有时间轴修复，Taylor P0/P1、合法one-swap、H65 retention/transition仍未完成，无完整正式性能。
- 历史30+60 H65的1280314/1280316仍FAILED(1:0)。重新读取完整stdout/stderr，仍为epoch16 AMP replay后p_action非有限，首个污染源未定位。原任务负责修复，旧job/log/checkpoint保留，不接管或盲目重提。
- 四个相关修复本地身份随目录刷新；本轮git ls-remote成功但仍无对应GitHub ref，不称已发布。历史BAFDR1267920/1267921只读，不取消、不修改、不重提。

## 协议、基线与资源

新训练按TEST_GUIDED_EXPLORATORY_EVAL5，测试集参与最佳EMA选择并允许指导调参，最佳可能早于6000更新，终态单列。
不称未见测试泛化、跨种子显著或官方论文公平复现；推理仍不输入测试GT。本轮无新调参尝试。
原始AdaTAD历史68.73%来源补证和独立复评仍待闭环；不以修改REF-D76867.58%、H65均匀或ET关闭替代。

09:52公共可调度节点未分配GPU53/200；账户1 RUNNING、1 PENDING；实验盘120.05GiB可用。
WorkDir/日志路径确认队列1278774属于GeoSparse、1281343属于FineDiving，未并入六路线或操作其作业。
分钟服务状态由目录生成器刷新；plan/BLOCKED只能称只读轮询。automation_update工具仍不可用，未谎称改写陈旧提示或补执行错过的心跳。
本轮没有Pro咨询，没有声称ixBrowser Computer Use初始化阻塞解除，也没有切换API或其他浏览器。

本轮目录freshness及仓库C3 focused tests为28 passed；三个工具py_compile通过。22项唯一身份、60份周期结果、5项独立终态VERIFIED均通过检查；五项新训练和四个相关修复本地exact/clean，原指标/曲线/调参记录未变，29号原部署对象仅增加显式解释备注。
10:02目录刷新时分钟服务ACTIVE、回执年龄22秒，dispatcher仍plan/BLOCKED、entries=0，不是自动正式提交。完整结构化证据见[43号JSON](43_HEARTBEAT_EVIDENCE_20260909_0950.json)。
