# 待通过ixBrowser ChatGPT Pro发送的诊断问题（本轮尚未发送）

请严厉审查以下GeoSparse-TAD性能退化的因果解释。不要将未测假设写成已证实bug，也不要建议同时增加多个模块。请返回：最可能的3个原因、每个原因的反证、无需重新训练的最小实测、确有依据后才做的新训练改动。

固定官方OpenTAD O=346d09d19e2091372cec48172dbe40f7b28bdee6，当前Geo模型M=b70ae056c495b43ca3f305fe438926b97b2723b5。全部200 THUMOS训练、211测试/792窗口，VideoMAE-B、768输入、160px、tubelet2、48个parent attention、全native384 TIA、检测768、warmup5、cosine100、训练60。cosine100/end60是官方原版，不要当bug。官方原版独立训练双卡每卡1，已完成60轮；统一Geo trainer单卡effective2，不能声称两者数学完全等价。统一dense控制已开始训练，尚无结果。

官方独立训练best E52=70.1806%，E60=69.8879%；发布EMA独立复测71.1388%是另一项。Geo每5轮全测试EMA best：
- A动态已完成60，best E40=68.0196%，E60=63.7006%；独立重导出同best68.0156%。E40全792窗口全部预算q1，eligible-full Heavy为100%，padded-full为96.97%。不能称50%效率方法。
- A固定q.5 best E50=61.2806%，完成55轮待续训。固定q.5 E15：A38.09%、B24.23%、C40.35%；对应mAP@.7为12.46%、5.78%、14.51%。这不是总延迟已匹配。
- B动态E20=30.2768%；其E10全窗口q.25、约22.43%eligible-full Heavy，E20预算未测。
- C动态E20=53.4452%、mAP@.7=29.4846%，高于A同期52.07%/24.89%；不能认定C最终失败。C E10曾全q1，E20预算未测。

A/C在原生位置真实gather-selected→Heavy QKV/attention/MLP→scatter→原全时间TIA；C每层coarse mean4、共享delta回原state。当前 source-limit 已有数值/梯度证据。B则禁用源12层TIA，冻结识别Heavy，按tubelet压成4slot、投影768→256，128维随机Scout投影到256的cheap query经2层support receiver，最后仅一次native384的256维TIA→插值768。receiver硬支持半径=相邻tubelet时间差中位数×8，带null。B-full同结构全Heavy的控制已经开始，但尚无性能；不能预先归因哪个模块导致损失。

硬策略：训练budget multinomial，测试budget argmax；位置训练Gumbel top-k并记录完整有序PL logprob，推理按分数排序。actor=(task+dual*(actualMAC-.5)-critic).detach()*joint_logprob，critic平方误差；随机探索分支不做actor/critic，前6轮全随机，6–20轮探索从.5降到.1，之后.1。cost_ema以.1更新，dual每成功step加.001*(cost_ema-.5)，clamp0..10。A E40普通checkpoint dual=.18907/cost_ema=.45991，EMA=.14215/.54694；E60普通=.05944/.37548，EMA=.11609/.44330。训练成本期望控制不能保证argmax部署成本。

在线acquisition每32minibatch抽一个未选原子（完整窗口9600候选2×2组），signed L(S)-L(S+a)，同模型完整重算、RNG/buffer恢复，.1 smoothL1训练同一个gain head。A60轮去除重放后157probe，abs gain中位3.73e-5，72%小于1e-4。不是swap/R1，不把gain接近零直接当噪声，不把它未加进actor reward解释为无梯度。

实际训练log：A动态E40→E60任务loss=.50850→.42866而测试回退4.319pp；同阶段global preclip梯度中位3.39→2.41。actor标量约−4000但不能据此推断梯度支配。B固定E15任务loss=.72631，A=.58341/C=.57404。B动态E20 preclip梯度p50=13.09、p95=111.71，而A/C同期p50=3.80/3.48。loss合并后全部参数global clip1；尚未做分量梯度测量。A所有6000步骤实际5994成功，6次AMP overflow，不能将日志重放重复行算额外更新。

用户授权用全测试集挑best并做超参数开发，未来结果将标成test-tuned development，不能宣称独立泛化；测试GT仍不加入optimizer或utility标签。现有训练不改源码、不重复训练。已提交无训练诊断1280030：固定动态A best E40，同一次全211/792 forward固定q.5输出精度与实际成本；尚未运行完，不能冒充训练好的预算约束模型。

请重点判断：(1)哪项训练/推理策略不一致最值得先修；(2)B缺逐层TIA与receiver压缩如何最小拆解；(3)联合PG/critic/CF梯度诊断应该怎么测，哪些未经证明的“按K归一化”会改变目标；(4)同成本swap与几乎零gain如何合理校准；(5)先测什么足以决定是否需要新训练。不要推荐凭测试分数删最大预算档来冒充动态预算算法。
