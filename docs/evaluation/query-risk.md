# 检测位置上的R0风险与成对收益

本工具用于SCI3 A01/A03：将真实源检测头的eval R0分类、回归损失分解至每个FPN输出点。
它不是新的训练loss、实例平衡R1、未实现的swap监督或已经得到的科学结论。生产M未改动。

`collect_query_risk`复用原头的prepare_targets、Focal、IoU及proposal算子，保留原label
smoothing、回归权重和eval正样本数分母。背景仍有分类风险，padding贡献为0，回归只在
原positive mask上计算；GT长度相近导致的多标签不压成单一类别。空GT不变成零总风险。

`evaluate_query_pair`调用现有完整重算CounterfactualRunner，两次仅改变强制计算计划，
在源head真实loss调用处保存分解，返回原标量loss而不替换。默认匹配逐层parent Heavy容量，
正常及异常退出恢复原method与既有runner负责的state/RNG/mode。没有缓存末端特征删减。

返回`points`四列为检测网格中的center/reg_min/reg_max/stride，并保留`level`与
`level_index`。FPN不同层同坐标的点仍分别存在，不能当作新增视频帧，也不能按拼接rank
画物理时间。detector_to_input_position_scale只转换为输入采样位置；真实秒数还需原PTS。

`signed_query_gain[key] = left[key] - right[key]`表示当前实际计划改变带来的逐输出点风险差。
多次合法干预可堆叠成操作→受益位置图。分类/回归分量均保留符号，attention不是这张图。
每个点的风险不是边界误差或AP，也不能根据其高低宣布某个真实实例被检出/漏检。

源输出限定FP32、eval模式；不使用训练EMA normalizer，不承诺AMP或生产GPU已核验。
逐点归一化与源整体求和存在浮点归约次序差，记录`reduction_residual`及成对gain的加总残差。
源标量独立保留；不能把接近数值误差的微小gain解释为可靠机制效应。

用法：`evaluate_query_pair(model, video_batch, metas, gt_segments, gt_labels, plan_a, plan_b)`。
返回CPU张量适合保存为原始.pt产物；正式诊断由调用者绑定视频、checkpoint、源码与计划身份。
新增检查：`python -m pytest tests/test_sci3_query_risk.py -q`。合成输入上的源算子正确性不能
代替实际视频、独立checkpoint诊断或论文结果。GPU入口与完整A03数据仍待实现/执行。
