# DUCA H65 总 60 轮课程压缩设计

## 科学问题

历史 H65 在完全相同的 K=384 稀疏输入规模下，以 30 轮 uniform 语义预训练加 60 轮 learned-selection 联合训练取得 65.385724% Avg-mAP。该结果包含 90 轮训练，不能直接作为公平 60 轮主结果。本设计只检验：保持 H65 模型、输入、损失、检测器与评估器不变时，能否把训练时间轴压缩为总计 60 轮而不损害定位性能。

## 唯一改动

模型代码保持冻结。训练课程改为：

1. epoch 0–19：exact-uniform K=384，训练完整检测器与 H65 的动作性、转变和边界预测；learned sampling、detector-to-selector 梯度、贡献蒸馏和 ASFormer adaptation 均关闭。
2. epoch 20–39：从 Stage-1 的 epoch-19 EMA 初始化并重建 optimizer/scheduler/AMP；在 2,000 次成功更新内，以余弦函数把语义权重、policy alpha、detector feedback、贡献蒸馏与 ASFormer adaptation过渡到 H65 的联合训练终值。
3. epoch 40–59：继续同一个 Stage-2 optimizer/scheduler，不进行第二次重置；保持完整 H65 learned-selection 联合训练 2,000 次成功更新。

实现由一个 20 轮 Stage-1 和一个连续 40 轮 Stage-2 构成。阶段边界只有 epoch 20 的一次初始化与优化器重建，禁止在 epoch 40 再次加载 checkpoint 或重启优化器。

## 冻结不变量

- H65 ASFormer scout、exact-K decoder、`density_transport_st` 梯度桥和 contribution distillation；
- 全局非均匀 K=384 帧集合及 selected-rank VideoMAE 输入；
- detector/backbone/head/loss/NMS/official evaluator；
- NMS 前物理时间回映；
- THUMOS14 split、数据增强、seed 3407、batch 与每轮 100 次成功更新；
- selector 参数组学习率和各损失起止权重；
- 每 5 epoch 可恢复 checkpoint、terminal EMA 唯一选择规则；
- 不启用 SingleClock、TrueTime、UVT、Fovea、Query-Bridge 或 dynamic outer-K。

## 实现边界

只新增两份配置、一份 N16R4 launcher、一份只读 validator 和 focused contract tests。不修改 `opentad/models/`、`tools/train.py` 或历史 30+60 配置。

## 证据合同

- 总训练为 60 epochs / 6,000 successful optimizer updates；
- Stage-1 terminal 为 epoch-19 EMA，Stage-2 terminal 为 epoch-39 EMA；论文时间轴映射为总 epoch 59；
- Stage-2 必须从 Stage-1 EMA 初始化，但不得恢复 Stage-1 optimizer/scheduler/AMP；
- 中间评估只画学习曲线，不选择 checkpoint；
- 主比较是当前原样 H65 30+60 复现与本 20+20+20 候选；不得重复 dense、random 或 uniform 基线训练；
- 成功门：Avg-mAP 相对 65.385724% 不低于 0.20 个百分点，且 tIoU 0.6/0.7 无显著退化；最终结论需逐视频配对统计支持。

## 失败解释

若 H65-60 低于非劣门，说明历史 H65 的前置语义收敛、后半程联合优化或额外 detector updates 不能被该课程无损重叠；不能据此否定 H65 选择机制本身。
