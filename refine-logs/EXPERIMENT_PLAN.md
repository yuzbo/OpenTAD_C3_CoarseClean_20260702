# 实验计划

最新冻结版本：`EXPERIMENT_PLAN_20260806_053037.md`。

当前 must-run 是同一 G1 锚点上的 fresh `none_control` 与
`residual_window_center` 双臂：seed 3407、60 epochs、9,600 successful
updates、exact B=24576、strict duplicate Gate replay。只有 center 在
mAP@0.6 与 mAP@0.7 均严格更高且 Avg-mAP 不降时，才授权另行冻结
ABBA+BAAB paired full-stack cost；多种子、M3、official test 与 paper claim
均保持关闭。

完整 claim map、实验 blocks、run order、成本和失败解释见上述时间戳版本。
