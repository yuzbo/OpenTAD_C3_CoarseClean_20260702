# VideoMAE A-MoD 最小实施计划

日期：2026-08-24

基线：`c6327a891809aa30370b3b2d9bedab0dcfe0d326`

设计：`docs/superpowers/specs/2026-08-24-videomae-amod-paper-exact-design.md`

## 1. 改动表面

1. `opentad/models/backbones/vit_adapter.py`
   - 为现有 Attention 增加一次前向同时产出正常输出与注意力列均值的精确分块实现；
   - 为 Block 增加 Dense-with-score 与 A-MoD exact-top-K 两条无新增参数路径；
   - 为 VisionTransformerAdapter 增加互斥的 A-MoD 配置和 Dense/A-MoD 交替调度；
   - 保存不参与梯度和模型选择的执行摘要。
2. `configs/adatad/thumos/georoute_official_amod50_prebackbone_seed42_v001.py`
   - 直接继承 untouched AdaTAD 配方；
   - 只增加 A-MoD-50、五轮恢复和 result-blind 合同；
   - 不启用 GeoRoute/ROI/native-packed/temporal-carry。
3. `scripts/run_zoomtoken_official_prebackbone_bc_n16r4.sh`
   - 复用现有两卡启动器，增加唯一 `AMOD50` arm 映射。
4. `tools/train.py`
   - 仅将 `AMOD50` 纳入现有五轮完整状态恢复的冻结 arm 集合。
5. `tests/test_zoomtoken_amod_paper_exact.py`
   - 覆盖论文分数、稳定 exact-top-K、交替调度、未选 token 恒等旁路、capacity=1 dense parity、无新增参数、配置与 launcher 绑定。

## 2. 实施顺序

1. 抽取 Attention 的共享 QKV 投影，并实现 query-chunked 精确 softmax；以同一次概率矩阵分块累计列和。
2. 在 Block 中实现：
   - `forward_dense_with_amod_score`：全 token Attention+MLP 后执行现有 dense Adapter；
   - `forward_amod`：按上一 Dense 分数 gather K 个 token，只对其运行本层 Attention+MLP，scatter 回完整 carrier，再运行 dense Adapter。
3. 在 VisionTransformerAdapter 中验证 12 层、零 attention dropout、奇数 A-MoD 层和 capacity；与 packed runtime route、ChronoTransport 互斥。
4. 添加 seed-42 配置、launcher arm 和恢复 allowlist。
5. 先运行 focused 数值测试，再运行语法、配置加载和改动范围检查。

## 3. 验收条件

- 论文列均值与直接构造的完整 attention 一致；
- 每个样本 exact K，分数并列时较小原 token 索引优先；
- 六个 A-MoD 层分别消费紧邻 Dense 层产生的分数；
- 未选 token 在 VideoMAE Attention+MLP 子块严格保持输入值；
- Adapter 仍在完整 token 网格运行；
- `capacity=1.0` 与原 dense 路径在 eval 模式数值一致；
- A-MoD 开关不改变 `named_parameters()`；
- 官方数据、split、优化器、调度器、EMA、评测器和 NMS 不改变。

## 4. 非目标

- 不加入 ROI、K64 预筛选、DSR6 short-Q/full-KV、RC32 carry、KV cache 或时序状态复用；
- 不加入可训练 router、辅助损失或 score scaling；
- 不在本实现轮提交 GPU/Slurm 训练；
- 不用理论 FLOPs 代替 selector-inclusive 延迟、能耗或显存结果。
