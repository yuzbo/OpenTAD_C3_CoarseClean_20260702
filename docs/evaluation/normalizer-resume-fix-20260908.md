# 普通检查点恢复的 loss normalizer 截断

2026-09-08 21:12（北京时间）补充核验。生产 M 为 b70ae056c495b43ca3f305fe438926b97b2723b5；本修复只进入独立 review 分支，不热改 M 或自动重跑现有训练。

源 ActionFormerHead 将配置整数100注册为整数buffer，训练时的移动平均更新将其替换成浮点buffer。普通checkpoint保存了完整浮点值，但原 restore_mutable_state 在新建整数buffer上执行load_state_dict，会将小数截断。这是恢复时的dtype转换问题，不是未保存normalizer。

真实普通checkpoint只读核查：A固定E50保存58.29771423339844，原恢复得到58；B动态E20保存42.68238067626953，原恢复得到42。相对状态差分别约−0.51%和−1.60%。实际小型GeoSparse B训练forward也复现10（int64）→9.300000190734863（float32）→9（int64）。这些数值是状态误差，不是mAP变化或整段训练损失变化。

独立修复在load_state_dict前使持久buffer采用对应checkpoint dtype，保持当前设备；不替换parameters，因此已构造optimizer的参数引用不变。官方opentad代码、EMA算法、数据和训练预算均不改。回归覆盖真实A/B/C小模型经过一次训练forward后的分数normalizer恢复，并逐项比较相同下一批的loss和梯度；另跑运行时、验证、原生执行和官方协议检查。测试结果保存在外部执行包，不将CPU合成输入当作生产GPU验证。

EMA normalizer的整数算术与官方ModelEma一致，不因本问题重新指控整个EMA。正式预测不使用训练loss normalizer，不能直接撤销已有推理mAP。中断续训存在确定的状态不等价，但其真实精度影响尚未量化；没有证据据此重跑官方原版或全部矩阵。

独立梯度诊断仍显式导入该实验绑定的M模型，记录checkpoint与实际恢复normalizer的值、dtype及是否精确；不在诊断中暗中修复M。它测量M真实恢复语义下下一epoch首批的梯度，不能称连续不中断训练的精确重放。

外部证据：official_adatad_audit/normalizer_restore_reproduction.stdout.txt、actual_normalizer_restore_20260908.json。此前20:33审查关于“buffer已保存”的结论保留，但不能据此推出恢复无损；本条作为新增发现保留时间顺序。
