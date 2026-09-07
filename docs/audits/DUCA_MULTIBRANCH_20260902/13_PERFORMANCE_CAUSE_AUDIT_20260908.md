# 六路线低性能原因核验

日期：2026-09-08 00:37 CST。范围：只读配置、代码、既有结果和 CT-DP 日志；在 N16R4 精确干净源码上执行一个 CPU 标签分配复现。未修改训练代码、checkpoint 或分数，未提交或取消训练。本记录是原因分析，不是一次完整队列心跳，也不是模型修复交付。

## 总结

不能统一归因于“路线错误”或“资源不足”。已经证实的情况同时包括：当前 CT-DP 坐标接线错误、部分关键机制根本尚未实验、基线配方改变，以及 ET-TRC 当前近似版本的负结果。此前把六条路线概括为“完整实现且完整验证”不成立。

## 1. CT-DP：当前实现仍有坐标错误

远端源码：`/data/run01/sczc063/yuzibo/projects/ctdp_successful_updates_78cde1be`，精确提交 `78cde1be1cb8b3acc7d750afc92ea740f2a03d06`。诊断前后 tracked tree 均干净。

G0/G1 将 768 帧压为 384 个检测位置，关闭 physical-grid head，却没有把 dense-axis GT 转换到 selected axis。selector 原样返回 GT，并设置 `irregular_native_axis=True`；普通 head 的点生成器仍按序号生成 0 到 383 的位置。后处理还会因 native-axis 标记跳过逆映射。

真实配置构建的 selector/head，在 CPU 上使用人工动作区间 `[500, 600]`，调用真实 `prepare_targets`，得到：

| 配置 | 传给 head 的 GT | 检测点范围 | 分配到的正样本 |
|---|---|---|---:|
| G0：均匀选择、普通检测头 | [500, 600] | 0 到 383 | 0 |
| G1：动态选择、普通检测头 | [500, 600] | 0 到 383 | 0 |
| G2：动态选择、物理网格检测头 | [500, 600] | 3 到 766.5 | 4 |
| G3：G2 加 CT-Tubelet | [500, 600] | 3 到 766.5 | 4 |

这是标签分配的确定性诊断，不是合成数据上的性能实验。它证实窗口后半段动作可在 G0/G1 完全没有正样本；没有量化它解释了多少真实 mAP 差值。G1 当前日志仍有 epoch51/5200 successful updates、训练期 Average-mAP 14.60%，说明补齐更新计数没有解决该坐标问题。

源码证据（以下相对路径均以该 CT-DP worktree 为根）：

- `configs/adatad/thumos/duca_ctdp_geometry_g0.py:2`、`duca_ctdp_geometry_g1.py:2`：禁用物理网格。
- `opentad/models/selectors/dual_phase_frame_selector.py:264`、`:303`：native-axis metadata 与未转换 GT。
- `opentad/models/dense_heads/anchor_free_head.py:330`、`:737`、`:916`：关闭物理网格后沿用序号 points，直接与 GT 做距离和正样本分配。
- `opentad/models/dense_heads/prior_generator/point_generator.py:25`：按特征序号乘 FPN stride 生成 points。
- `opentad/models/utils/post_processing/utils.py:187`：native-axis 标记跳过 selected-to-dense 转换。

已读取日志：`/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/slurm_logs/formal_g0_1276669.out/.err` 与 `formal_g1_1276670.out/.err`。存在有限训练 loss 和正常更新记录并不能排除该错误。

处置：G0/G1 不能作为合法均匀/动态选帧比较基线；先在独立修复分支统一 GT、proposal 和 postprocessing 坐标，做真实正样本分配与回映射验证，再按既有 focused/CUDA/PRECHECK 规则决定新训练。禁止只把 head 打开而使消融变量失去定义。本轮尚未修复或改动现有 Slurm 作业。

另外，当前 G0-G3 全部 `amod_config.enabled=False`，G2 只增加 physical-grid，G3 再增加 CT-Tubelet。这四臂不是“无模块/CT/B-AMoD/二者组合”，不能声称已完成 B-AMoD 消融。

## 2. H65-Pro：六个已完成配置并未开启四相选择

`E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission/tools/bata/generate_h65_pro_fullmatrix.py:15` 定义的 F01-F06 全部 `phase=0`；真实配置中的 `acquisition_policy` 为 `budget_calibrated_sampling_rate`。F02 的 64.2265% 是 `phase=False, taylor=True`，不是四相完整模型的分数。F09 才是一个明确开启 `semantic_phase_sampling` 的配置，不能把磁盘上存在它当成已经完成训练。

已完成六臂的数值仍真实，不撤回原始收据；撤回的是“四相机制已经由这六臂验证”的表述。

F 臂训练提交 `f2068e18` 比参考臂 `e553a5a4` 多了 `relative_physical_time_scale` optimizer group。两组都声明 6000 更新，不等于完整 optimizer 配置相同。+0.34 只能先作为数值差，尚未隔离该配置变化的影响。既有时间参数 optimizer 故障已修复，不能反复作为当前低分的已证实原因。

## 3. ET-TRC：当前版本有真实负结果，也未兑现完整事件触发设计

训练 `74473c2775caebf0da9d368ce8009d78e2942098` 与独立 evaluator `67d7079d` 之间，相关 backbone 与配置文件无差异。OFF 62.0768%，ON 54.8096%，本种子开启近似下降 7.2672 个百分点，不支持该版本保持精度的主张。

`E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval/opentad/models/backbones/et_trc_videomae.py:21` 明确说模块不是 attention/MLP 的精确 Jacobian：它只接收 `delta_h`，不接收 anchor state。代码用 rank64 的 down/temporal-conv/up 可学习线性近似，没有显式真实 JVP 或 dense residual 重建监督。两个独立正交初始化矩阵也不使 rank64 通道变换成为 384 维恒等映射，不能把初始化注释当作 dense parity 证明。

`:319` 使用固定间隔 anchors；配置 `et_trc_videomae_s_768x1_160_adapter_seed4407.py:35` 也明确记录 fixed-stride，而非事件门控。维持 dense tensor 的形状不代表保存了 dense Transformer 的真实特征。

“低秩误差在多层累积，且仅靠检测 loss 难以补偿”是有代码基础的待验证解释，不是已量化原因。下一步应在训练侧固定视频上测 dense/零阶/当前一阶残差误差、anchor 保真与梯度，先明确固定间隔对照和事件策略的边界，不从测试分数调参。不能据此否定所有 Taylor/JVP 路线。

## 4. BAFDR：空间分辨率、帧数和 tubelet 曾被混淆

`E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_successful_updates/opentad/models/backbones/bafdr_wrapper.py:64` 和实现说明共同确认：G96 是 96x96 空间分辨率；全局流仍处理 48 个 16 帧 chunks，即 768 帧。K16 局部流处理 16x16=256 个原始帧，经过 tubelet-size2 才得到 128 个时序 tubelet，不能称为 128 个输入帧。

它的输入几何、全局低清、中心局部裁剪、硬 Top-K 以及非对称特征注入都不同于“均匀384帧、160分辨率的 AdaTAD”。因此 50%左右不能仅用帧数比较解释。硬 Top-K 的 detector-to-router 选择梯度为零是显式设计，不是此次已发现的漏接；仍需用匹配的 uniform-K16 对照判断辅助监督是否有效。

旧训练少 3-6 次成功更新是真实协议缺陷；没有证据证明补这几次就能弥补十余个百分点。修复版新五臂尚无本轮已核验的正式 mAP，不能先许诺修复后回到65或69。

## 5. Evidence 与 Unified：已有数字不代表完整机制

Evidence C0 是 `MATCHED_H65_60` 选择控制；A2 是关闭 time conditioning、仍保留 coverage/merge/recovery/robust 的反事实配置。59.23/54.28 都不是完整 F 的终态效果，C0 也不是均匀384。完整方案及配对消融未齐前，不能以这两个分数判定证据补漏路线失败。

Unified 的生成器仍在 `tools/bata/generate_duca_unified_fullmatrix.py:22` 明确阻断真实 Taylor P0/P1 接线和原始 H65 retention/transition。它是实现欠账，不是“训练过以后效果低”。

## 6. 共同的基线配方差异

本地已核对：官方 AdaTAD 使用随机空间裁剪、翻转、颜色增强；ZoomToken 的 D160 base 改为固定全画面 letterbox。多条路线还改变了学习率周期、batch/样本曝光和终态选择。H65 reference 的 cosine max_epoch=60，而原始配置 max_epoch=100、end_epoch=60，都是60轮但学习率曲线不同。ET 的 OFF 已只有62.08，因此其与69.03的差距不能归因于尚未开启的 Taylor。

这证明比较配方发生变化，不证明每项变化分别造成了多少下降。69/65是有来源的参考，不是算法理论保证，也不能通过看测试集追分来替代复现。

## 下一步优先级

1. 优先修 CT-DP 的坐标错误，保留现有证据，不再将其 G0/G1 作为有效公平对照。
2. 校正已跑配置的实际因子：H65 phase-off 六臂、CT-DP geometry 四臂和 BAFDR 空间/时间单位分别记录。
3. 只读核对已存在的共享 AdaTAD 原始配置和产物，不重复训练共享未修改基线；先解释 OFF/对照自身的差距。
4. 以最小诊断验证 ET 近似误差、BAFDR 输入/教师映射，再决定修复或收缩方法；不能盲目重跑全矩阵。
5. 完成 Unified 缺失机制、H65 phase-on 与 Evidence FULL 的对应实验后，才做路线取舍。

本次没有证明“所有路线正确”，也没有证明“所有路线错误”；已经证明先前的完整性和部分机制标签报告过度，必须纠正。
