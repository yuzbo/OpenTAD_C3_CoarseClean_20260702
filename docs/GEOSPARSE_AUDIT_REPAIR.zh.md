# 独立审计修订：2026-09-08

审计对象是模型提交 `902fa05b5c64452ff1c94b82cabce801943d3484`。报告按证据逐项核验后，本修订仅处理影响正式训练、独立评价、几何解释和测量配对的问题。原始官方 OpenTAD 提交 `346d09d19e2091372cec48172dbe40f7b28bdee6` 的核心、配置与 train/test 入口保持不变。没有新的 GeoSparse 最终精度或加速结论。

## 保留的用户协议

- 全部 200 个官方训练视频；全部 211 个官方测试视频、792 个测试窗口；不恢复 180/20 留出。
- 768 输入位置、160px、2-frame tubelet、48 个 Heavy attention parent、原生 384 时间位置；检测网格默认 768。
- global batch 2、warm-up 5、cosine horizon 100、实际训练 60 epochs；当前只推进 seed0。
- A/B/C 完整创新方法优先，固定 50% 和动态预算均执行；各路线不等待其他路线的 mAP 或 Oracle。
- GeoSparse 每完成 5 轮全量 EMA 验证，主结果选本配置 best；官方原版保留完成 42、44、…、60 轮的原验证节奏。
- 两者选点机会不同，不能称训练等价。统一训练器的稠密控制明确命名 `geosparse_dense_control`；官方原版独立训练和 published-checkpoint 推理复测另列。

## 修复与边界

|审计项|处置|验证与限制|
|---|---|---|
|I01|Heavy tubelet eligibility 与 detector mask 分开。query768 原样使用 frame mask；query384 每对任一有效即该 cell 有效。|真实配置回放 seed0/epoch0，第二个 batch 的 125 帧在旧实现变126，确认为实际可达。CPU测试覆盖767/763/0；GPU预检增加完整检测头的loss、raw proposals和特征梯度比较。|
|I02|独立导出、校准、评价统一 `internal_diagnostic`，映射 annotation 的 `training` 和训练视频 root，采用完整 deterministic test pipeline。|诊断为全部200训练视频；阈值在生成正式测试预测前冻结；测量修复无需重训。|
|I03|Heavy MAC 分母使用当前实际 spatial grid，记录参考 resolution/backbone/width/depth/parent shape。|160-full=1、224-full=1、C-all-coarse>0；跨分辨率使用绝对成本。训练内成本分母原来就是正确的。|
|I04|B/cheap 在完整 native384 query 接收 evidence 后执行全窗口规则 TIA，再插值到 query768。|跨 native7/8 的脉冲有传播；不同 batch 不串扰；Heavy attention 仍限制在原 parent。|
|I05|沿官方 ImgAug 相同的一次 deterministic draw 记录 keypoints 的仿射变换，原像素增强不移除；source ROI 使用四角，保留未裁剪 polygon。|比较多次增强的逐像素结果及 Python/NumPy/imgaug 后续 RNG。记录是锚点几何，不声称颜色增强/Cutout 后仍观测到全部原像素。|
|I06|EMA 验证备份/恢复所有被覆盖参数，包含冻结权重；保留 buffers、mode、RNG 恢复。|自然500次EMA造成的冻结舍入差在正常/异常退出后均逐位恢复；不由舍入差推导mAP下降或全矩阵重训。|
|I07|保留两种基线名称和训练行为差异。|单卡batch2的local positive normalization、AMP overflow时推进规则仍不同于官方双卡每卡1，不能宣称完整训练等价。|
|I08|分离 training_complete/selection_complete；缺失验证从既有 epoch 权重补测。持续失败为 selection_pending，队列和下游不准入。|重新进入完成60轮的同一任务只补缺失推理，无优化器更新。永久失败须人工定位后补测，不自动重复训练。|
|I09|按 source_resolution × spatial_group 联合验证。|160/g7 显式阻塞，224/g7仍合法；不改分辨率或丢边缘。|
|I10|新协议及新任务ID；所有训练保存零基4/9/…/59，诊断请求4/19/39/59。|保存点对应完成5/20/40/60轮；full/focus使用同一产物契约；旧ID与新ID映射留在外部运行记录。|
|I11|登记的 feature_l2 显式阻塞，直接构造也拒绝；去除冒充对齐的 normalize 分支。|未新增 Teacher 或未获定义的对齐目标；不扩展当前消融。|
|I12|CUDA driver 读取当前逻辑设备UUID，与 NVML UUID 匹配；隔离检查以UUID查询。|GPU预检记录Slurm分配、CUDA逻辑索引、NVML索引、PCI与UUID；无现场结果前不声称已完成硬件计时。|
|I13|best 文件、epoch、模型源码、resolved config、split和初始化共同形成配对身份，评价/成本/硬件/绘图检查同一身份。|保存实际计时window IDs及GPU UUID；同train ID下更换best将拒绝旧latency；不配对的历史计时仅需重测。|
|I14|明确 evidence support 是 anchor union，feature 依赖同 parent 全部 selected tokens；保留parent字段和ROI polygon。|未实现独立packet缓存；D06保持未完成，不虚构已有stale-cache错误。|
|I15|矩阵compiler提供明确 `matrix_summary(jobs, refs)`，refs与计数摘要分离。|总计1545：612 train、612 evaluate、204 benchmark、117 diagnostic；登记不代表实现或完成。|

## 旧实验处理

旧 A 固定50%已保存完成10轮的状态和5/10轮验证，动态A已保存完成4轮状态；停止旧训练，保留日志及checkpoint。原固定A完成8轮的全状态另在A100备份。旧权重可用于原实现轨迹和影响诊断，不转成“修正目标完整60轮”的结果。

B/C固定与动态、B-full、统一dense和uniform此前尚未训练，旧待运行作业hold，修订源码预检通过后从同一识别预训练独立初始化。官方原版独立训练不因外部wrapper缺陷重训；published EMA全量推理复测71.1387948970%继续保留，不能挪作GeoSparse或独立训练结果。

## 补充比较与尚缺证据

保留各自既定best规则，另报告固定完成60轮EMA，以及共同完成50/60轮之间的best。GeoSparse独立评价直接复用这些既定节点的全量预测；这项补充明确记为本次修订分析。官方对照待相应节点产物到达后按同一含义提取，不改官方训练入口。

seed0不报告跨训练种子的均值/标准差或稳健显著提升。117项D01–D06诊断仍未完成；ROI、其他估计器和未接入变体保留实现阻塞。FineAction原视频到位不等于其dataset adapter或训练已完成。同GPU准确率—延迟配对结果尚需真实训练和计时。

CPU回归结果与GPU能力凭证、训练状态保存在仓库外的执行包；CPU小模型测试不替代生产768×160和真实权重的GPU预检。
