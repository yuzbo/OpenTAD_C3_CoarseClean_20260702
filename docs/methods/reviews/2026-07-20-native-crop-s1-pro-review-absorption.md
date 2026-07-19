# Native-Crop S1 Pro 审查吸收裁决

## 来源与完整性

- 原始回复：
  `docs/methods/reviews/2026-07-20-native-crop-s1-pro-review-raw.txt`
- 原始附件：
  `C:/Users/skywalker/.codex/attachments/d5efbafb-a0d0-4e2e-b601-e377de269c68/pasted-text.txt`
- 原文 SHA-256：
  `7AB0E10624A14FDF2FCABCBEF5EF435EB4994B83BC3E32F3240A3E0143CD44D5`
- 审查对象：
  - 路线纠偏 commit `d76ba1b82fbd43d278700e81ae1c688439db76b3`
  - R0/runtime commit `cef95485d1bfebccddb1055f30800ab081decaf7`
  - 训练/模型 commit `18139b930bef6ee234f6220a6adc898eb9c23c0c`
- Pro 原始裁决：`PROCEED_NATIVE_CROP_S1`
- 本地复核日期：2026-07-20

原始回复原样保存。本文件只记录本项目对其事实、设计建议和裁决的独立吸收，
不能替代原文。

## 总体结论

**不完全同意，但同意其主方向和受限裁决。**

我们接受 `PROCEED_NATIVE_CROP_S1`，但它只授权一个不训练、不打开 official test 的
Native-Crop 垂直切片，不能被解释为：

- Native-Crop 已实现；
- crop sufficiency 已验证；
- 八候选 teacher oracle 足以裁决整个空间裁剪路线；
- 数值 GO/KILL 门槛已经合理；
- learned ROI policy 已获准；
- Spatial Zoom 已具备论文贡献或可发表性。

当前研究状态仍是 `designed`。旧 R0 是整图分辨率控制，不是 crop 证据；当前没有任何
Native-Crop 实验结果。

## 完全采纳的事实判断

1. **当前仓库没有 Native-Crop。**
   现有数据通路是全图 `Resize/RandomResizedCrop/CenterCrop`，随后经 VideoMAE、
   空间均值、384 到 768 插值和 ActionFormer 检测。不存在 source-coordinate ROI、
   local native-density branch、ROI tube 或 global/local fusion。

2. **旧 R0 合约和最新研究目标存在 split-brain。**
   旧 `spatial_zoom_s1_contract.md` 仍把 `dense160/224/256 x 3 seeds` 当作正式 S1，
   而最新路线已经把它降级为 `R0 dense-resize headroom control`。旧文件必须冻结，
   Native-Crop 必须使用新的方法身份、配置、证据 schema 和实验 namespace。

3. **R0 不能否证空间裁剪。**
   三个 resolution 的 gate mean 约为 `64.220/64.228/64.252`，近乎平坦，但它只说明
   当前全图 resize 区间内分辨率敏感性很弱。唯一 official-test cell
   `dense256/seed3408=67.09` 也不能选择分辨率，更不能证明 crop 有效或无效。

4. **必须先做 crop sufficiency，再做 learned policy。**
   在不知道“保留原生局部像素是否足够保护 TAD”的情况下直接训练 scout/policy，
   会把路线可行性、候选覆盖、策略学习和 detector 适配混在一起。

5. **Native crop 必须发生在任何全图空间 resize 之前。**
   crop 坐标必须属于源视频帧，并记录源尺寸、crop box、padding、有效像素比例和
   token grid。local 分支若被 resize 回固定大尺寸，就不能声称保持 source-native
   pixel density，也不能按 crop 面积直接声称成本下降。

6. **时间轴与检测后端应保持正交。**
   Native-Crop S1 保留完整 768 点时间轴、现有 VideoMAE/AdaTAD adapter 的时间合同、
   `[B,384,768]` detector feature contract，以及现有 ActionFormer
   projection/head/loss/NMS。空间假设不应通过改 detector 获益。

7. **成本必须按完整通路报告。**
   decode、global/scout、ROI 决策、crop、H2D、global/local backbone、fusion、
   detector、NMS、显存、延时和能耗必须分项及汇总。teacher/oracle 搜索成本必须单独
   报告，不能算进可部署推理成本，也不能隐去。

8. **当前 novelty collision 很强。**
   Glance and Focus、Uni-AdaFocus、AdaSpot 和 EVAD 已覆盖 global/local、
   spatial focus、动态分辨率或 token pruning 的大部分上位概念。不能声称首次提出
   video spatial focus 或 global/local video model。

9. **当前最可能成立的创新只能是组合，而非单点。**
   候选组合是：dense-time continuous ROI tube、source-native local compute、
   detector-regret/high-tIoU/boundary-aware supervision、面向 TAD 的 384 点
   global/local fusion，以及完整端到端成本闭环。即便全部实现，也仍需实验支持。

10. **当前不应打开新的 official test。**
    结构与门槛仍在设计期，official test 已存在历史暴露。下一步只允许 development
    fit/gate 上的实现验证和方法充分性实验。

## 部分采纳或不同意的判断

### 1. 八候选库不是路线级 oracle

Pro 提议的 center/corner/motion/person/saliency 等八候选，只能给出
**library-conditional upper bound**。如果该库失败，只能说明这个候选库覆盖不足，
不能直接 `KILL_SPATIAL_CROP`。

在用其裁决整条路线前，必须至少满足其一：

- 给出候选覆盖率证书；
- 使用更密集的 source-coordinate 多尺度网格作为上界；
- 明确把 KILL 结论限定为“固定候选库路线失败”，而不是“连续 learned crop 不可行”。

### 2. 最终 masked pooling 不能消除 padding 污染

VideoMAE/ViT 的 self-attention 会在最终 pooling 前让 padded patch 与有效 patch
相互影响。只在输出端做 masked mean，不能撤销这种污染。

优先实现应是：

- 当源帧尺寸足够时，将固定大小 crop box 平移回图内，保持 1:1 像素且不 padding；
- 仅当源帧本身小于 crop 尺寸时才允许 padding，并显式记录；
- 若要在一般情形支持 padding，需要 patch/attention-level mask；这会改变 backbone
  合同，应单独审计，不能用输出 masked pooling 冒充已经解决。

### 3. `global96/local128/48 knots` 等数值尚未校准

`96/128` 空间尺寸、每 16 帧一个 knot、速度 `0.20`、尺度 `1.5x` 和两 knot hold
均是可实现建议，不是已证实参数。不同源分辨率下，固定 128 source pixels 对应的视野
比例差异很大。

在冻结这些值前，必须先在 development fit/gate 做不使用 test GT 的 geometry census：

- 源视频 H/W 与宽高比分布；
- 96/112/128 crop 的无 padding 覆盖率；
- crop 面积占源帧比例；
- 不同 source resolution 下的相对视野；
- knot 频率对 box 速度/平滑约束的可实现性。

### 4. teacher/test 冲突不是当前垂直切片的 P0

Pro 指出的 teacher exception 与 test leakage 冲突，确实是 formal oracle/test 的
阻断问题，也必须在论文中披露历史 official-test 暴露。但下一步垂直切片不使用
teacher、不打开 test，因此它不是当前实现阻塞项。

当前真正 P0 是：

- 旧 R0 可执行合约仍与新方法身份混杂；
- 仓库没有 source-native crop 数据和模型通路。

teacher 的 split、cache 和 test exception 应在进入 oracle experiment 前单独冻结。

### 5. 数值 GO/KILL 门槛只能视为草案

Pro 给出的 `+1 pp`、`30%` 成本下降、`0.5 pp` 等门槛没有由当前 seed 方差、测量误差、
实际等价界或 power analysis 推导。它们不能直接写成 canonical contract。

正式预注册前必须使用：

- 历史 seed variance；
- paired video-level uncertainty；
- cost profiler measurement error；
- practical-equivalence margin；
- 独立统计审计

来校准门槛。

### 6. teacher-reference 候选模型的训练合同尚不完整

如果 detector 只用 center crop 训练，再在 gate 上枚举 motion/person/corner 候选，
候选间性能差可能来自 train/test crop-distribution shift，而非 crop sufficiency。

在 oracle 实验前必须明确：

- 一个共享模型是否用预注册的候选增强训练；
- 还是每个候选拥有 matched 训练；
- teacher 如何避免用同一标签同时做模型选择与性能报告；
- oracle regret 如何与 candidate coverage 分离。

### 7. 不得物化整段 native float video

伪代码中的 `[B,1,3,768,H,W]` 若在 float/GPU 上物化，会吞掉预期成本收益。正确实现应在
decoded uint8/CPU source frames 上先生成 global/local crop，再做 tensor formatting
与 H2D；不能把完整 768 帧 native float tensor 送入 GPU 后再裁剪。

### 8. 两次共享 VideoMAE 的真实成本不能按像素比例推断

共享权重不等于共享计算。global 和 local 两次 forward 的 kernel launch、位置编码、
attention、decode、crop 和 H2D 开销都可能使实际收益偏离 token/pixel 预算。像素量只能
作为预算变量，最终结论必须来自相同硬件上的 full-stack profile。

### 9. `no-resize` 是必要差异，但不足以独立构成创新

AdaSpot 的官方实现会把可变 ROI 插值到固定 `roi_size`，因此 source-native
no-resize 是真实可区分点；但仅此一点不足以支撑论文。需要和 detector-regret、
boundary/high-tIoU 目标、continuous tube 及成本闭环共同成立。

## 独立文献核验

- Glance and Focus:
  https://arxiv.org/abs/2010.05300
- Uni-AdaFocus:
  https://arxiv.org/abs/2412.11228
- Uni-AdaFocus official repository:
  https://github.com/LeapLabTHU/Uni-AdaFocus
- AdaSpot, CVPR 2026:
  https://openaccess.thecvf.com/content/CVPR2026/html/Xarles_AdaSpot_Spend_Resolution_Where_It_Matters_for_Precise_Event_Spotting_CVPR_2026_paper.html
- AdaSpot official repository:
  https://github.com/arturxe2/AdaSpot
- EVAD, ICCV 2023:
  https://openaccess.thecvf.com/content/ICCV2023/html/Chen_Efficient_Video_Action_Detection_with_Token_Dropout_and_Context_Refinement_ICCV_2023_paper.html
- AdaTAD:
  https://arxiv.org/abs/2311.17241

核验结果支持 Pro 关于 novelty collision 的主判断，也支持 AdaSpot 将 ROI 插值到固定
尺寸这一实现差异。但文献存在并不自动证明本项目提出的全部组合新颖，仍需后续系统查新。

## 冻结后的唯一下一步

1. **先做 development-only geometry census/precheck。**
   不读取 official test GT，统计 source H/W、crop 可行率、padding rate 和相对视野，
   决定 96/112/128 是否可作为 honest source-native 候选。
2. **实现不训练的 center-crop 垂直切片。**
   建立独立 Native-Crop namespace，实现
   `global96 + fixed center/local128 + shared VideoMAE-S + 384-point fusion
   + [B,384,768] output`。
3. **垂直切片暂不加入 teacher 和八候选枚举。**
   先验证 source-coordinate、1:1 像素、无隐式 local resize、可变 grid、backward、
   detector parity、no-leak 和分阶段 cost schema。
4. **旧 R0 全部冻结。**
   不修改旧 checkpoint/test/profile 证据，不继续 recovery matrix，不将其写成 crop
   baseline。
5. **通过独立代码审计后才设计 oracle experiment。**
   届时再冻结候选覆盖、teacher split/cache、训练分布、matched baseline 和统计门槛。

## 必须通过的垂直切片测试

- crop 坐标在 source frame 中定义并可逆映射；
- local crop 与源像素逐值一致，禁止 local interpolation/upsample；
- crop 发生在任何整图 spatial resize 前；
- box 尽可能平移回图内，padding 只在源帧小于 crop 时发生；
- 96/112/128 与 rectangular source grid 均能运行；
- runtime H/W token grid、position interpolation 和 adapter reshape 正确；
- shared VideoMAE 两分支 backward 有限且必要参数有梯度；
- 输出严格为 detector 所需 `[B,384,768]`；
- 禁止 GT、teacher、oracle、test certificate 或历史 test evidence 进入 forward 决策；
- cost schema 分离 decode/crop/H2D/global/local/fusion/head/NMS；
- 不在 CPU/GPU 上物化完整 native-resolution float video；
- 旧 R0 配置、证据与 namespace 未被修改。

## 当前允许的最小陈述

> 当前代码已正确识别旧 R0 与真实 Native-Crop 研究对象之间的差异；Pro 审查支持继续实现
> 一个 development-only、source-native、dense-time 的无训练垂直切片。该裁决尚不支持
> crop 有效性、效率、路线 GO、learned policy 或论文创新主张。
