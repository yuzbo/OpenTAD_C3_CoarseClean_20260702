# 实现冻结规范 v1.0

这份规范决定算法含义。agents 不得自行把概念换成方便实现但不等价的操作。默认值为预注册设计选择，不是已经验证的最优设置。

## 0. 基座审计和共同配置

优先适配用户现有 AdaTAD/OpenTAD/DUCA 仓库；没有可用仓库时使用官方 OpenTAD，但必须 pin commit。不得覆盖用户原实现。

共同对比配置：raw RGB end-to-end；VideoMAE-B 识别预训练初始化；Backbone 原有大权重冻结，训练 TIA/检测头/新增小模块。`requires_grad=False` 不等于可以对整个 Heavy forward 使用 `torch.no_grad()`；上游 adapter 仍需梯度穿过冻结 block。

共同窗口默认 768 个输入采样位置、224×224、native tubelet=2、parent clip=16；这些默认值必须经 source config 审计形成 resolved_protocol.json。若现有基座不兼容，以一次公开 protocol amendment 统一改变全部实验，不能逐方法改变。48个parent clips、384个tubelet slices是该配置下的推导，不是所有数据集永恒常量。

GPU1为默认资源单位，effective batch=8（microbatch和accumulation可随显存调整，但优化步数与样本量相同）。epoch=60，seed={0,1,2}。所有主比较同数据增强、窗口和 detector；LR/WD继承同一已审计 base recipe；新增 router 使用 AdamW 1e-4、WD0.05、clip_grad_norm1.0（预注册起点）。原 optim 参数及新增组必须完整写日志。

检测网格默认使用基座原 native length→detector length 映射；另有384/768消融。时间标签为秒，不对 selected rank 做回归。

## 1. 接口（实现时使用 typed dataclass / TypedDict）

VideoBatch:
- frames_hi: [B,C,T,H,W]，包含经批准的规则采样输入；原图ROI分支另存源帧访问器。
- pts_s: [B,T] 实际源PTS；target_time_s: [B,T] 规则采样参考点。
- valid_frames: [B,T]；source_frame_id、video_id、window_id、window_start_s、spatial_transform。
- targets: 类别与源秒坐标区间；valid time span、官方ignore区域。

NativeLayout:
- token_id；parent_clip_id；tubelet_id；patch_yx。
- source_support: 每一tubelet实际两个样本的PTS或有效子区间列表。
- nominal_support: 原输入采样计划的支持；不得替代source_support。
- roi_xyxy_orig_norm、scale_tag、valid。

RoutePlan:
- atomic_action_ids、selected_native_indices、packed_offsets、parent_clip_id。
- requested_budget、realized_token_count、per_clip_counts。
- log_prob（训练期）、sampling_order（概率用途）、execution_order（原生坐标用途）。
- predicted_gain、routing_logit、budget_id、probe_ids、cost_estimate。

EvidenceBatch（B/C）：
- features、source_support_ids/interval unions、physical time、source ROI、parent partition、valid、slots、fidelity。
- 不把[t_min,t_max]误当所有内部时刻均被观察。

DetectionState:
- dense features、regular_time_s、valid mask、grid_stride_s。

关键分离：Plackett–Luce采样顺序用于log_probability；送入ViT时按原生索引排序并gather原PE。不能把采样概率顺序当时间序列。

## 2. 主路线 A：Geometry-preserving sparse refinement

### 2.1 Scout
默认：112×112 输入；逐帧轻量2D depthwise-separable Conv，空间总stride16得到7×7 map，宽度128；按native两帧配对融合；再使用两层kernel3 temporal depthwise Conv+pointwise Conv。输出[B,384,7,7,128]及全局pool。

默认spatial group=2×2 native patches，对应32×32名义像素区域；7×7 coarse map每个位置对应4个native patch token。时间原子=1个native tubelet。ST候选数=384×49；T-only将49个空间分数聚合成384个候选，选择时取整幅空间。

scout不使用已完成的heavy features、测试GT或预训练dense TAD teacher。可有actionness辅助头；必须计算其开销。主模型score head为两层MLP，输出有符号gain；budget head从global/context features输出离散预算概率。不得使用每视频z-score掩盖绝对价值。

### 2.2 Heavy结构
X0 = 原 PatchEmbedding + 原 PE。
对于原模型每个 block 的 MHSA、MLP 及 TIA 实际边界，保留原顺序、residual、LN、drop path和valid mask，仅将原本的重型算子改成 selected execution。
抽象写法：Delta = F_heavy(G_S X)-G_S X；X <- X + Scatter(Delta)；但它不能替代源码中的逐子层hook。

**严禁把48个原生clip的所有选中tokens拼成一个全局self-attention。** Heavy attention仍在同一个原parent_clip内运行，除非注册为不同上下文实验。TIA原先如果全时间作用，仍按全时间规则布局作用；如果用户仓库实际上clip-local，则单独标记基座并建立对应controls，不默默统一成另一种实现。

同一输入窗口默认一次route、全部heavy层共享S。局部K_j=0时，跳过该parent的重型更新，状态仍走合法轻路径。S为空时不能构造空softmax。S全量必须等价原模型。

实际执行selected QKV/attention/MLP；dense state只用于identity/轻路径。`dense heavy * mask`只准作为数学参考，不准进入speedup主表。

### 2.3 Packing
分组键至少包含parent clip、valid token count bucket和fidelity；绝不跨视频attention。PyTorch reference用逐parent循环，optimized使用分组/bucket/真实varlen算子。重复索引须去重；尾部padding不占语义预算但实际计算成本仍计量。

### 2.4 成本声明
A减少selected heavy MHSA/MLP更新，不声称节省dense decode、全部PE、全部TIA或全部激活内存。token ratio不是FLOPs ratio，attention按各parent的N_j分别核算。

## 3. 挑战路线 B：Dense Query + Sparse Evidence

与A共享同一Scout接口；C dense cheap features投影到256维query。默认384个query，原native→检测grid映射保留。
Heavy保留原native配对和parent分组；选中tubelets在同parent中共同编码，明确heavy TIA为packet-local或none，跨时间TIA在规则receiver后执行。此结构不是原AdaTAD等价改写；B-full是必做架构税control。
空间证据压到S=4 slots（2×2 adaptive pool）并保存源ROI/支持；消融S={1,4,8}，不能假定learned slots分别对应actor/object。

Receiver baseline顺序并行实现而非实验先后：rank_interp、physical_interp+linear residual、scatter+concat+MLP、timestamp-only attention、support-aware attention。
默认support-aware receiver为2层、width256、heads4、local radius=8 native tubelet步对应的秒数；加少量global pooled semantic slots。距离由物理时间及支持集合确定，采用相对秒差、duration、尺度和ROI编码；可学bias仅为参数化，不宣称保证识别。

必须提供zero/null update：没有相关evidence时输出coarse state的旁路，不强制远处attention。精确删除证据必须从key/value/softmax分母及任何影响其他证据的前序路径移除。

Dense query加入时空内容，纯learned位置query仅为消融。规则grid后使用原检测FPN/头。不同fidelity独立projection+LN；不得在未验证情况下强制feature L2一致。

ROI分支默认tubelet内共享ROI，跨tubelet可固定/平滑移动；crop来自source high-resolution pixels；coords经过同一空间增强的可逆映射。ROI224与fullframe224的token数一样，不能称为ROI本身减少编码FLOPs。多ROI计总cost，重叠区域去重策略须日志化。

## 4. 挑战路线 C：Mixed-scale state refinement

默认每native时间slice保持时间覆盖；空间2×2 fine组用一个coarse或四个fine表示，二者互斥。coarse由同native底座组平均+共享/轻量projection产生。首版PE仍密集；raw coarse kernel可优化但必须另测等价和实际成本。

每个block：P_Omega从dense state取mixed-scale tokens；heavy在原parent内处理；U_Omega将更新按支持映射回原native空间；然后按原序运行dense TIA。fine=identity，coarse更新分发到该组四个状态。全fine需建立dense limit。该路线不是“完全不维护dense状态”的轻内存模型。

mixed token ratio q>=0.25，p_fine=(4q-1)/3。q=0.25全部coarse，仍执行49个token的heavy；不得称K=0。C-fixed的q=0.25与C-coarse-only是同一个配置，manifest去重。

## 5. Utility与联合训练（主默认可执行，不依赖teacher）

### 5.1 默认硬策略
固定预算：gain/temp作为logits，Gumbel Top-K采样有序Plackett–Luce选择；硬执行。动态预算：先从{0,.25,.5,.75,1}采样预算，再条件采样位置；C的菜单限制为{.25,.5,.75,1}。
离散预算目标{.25,.5,.75}是期望heavy-cost设计点；实际匹配按measure，不按名字假定相等。C动态最低目标为.5，避免退化的全coarse预算控制。

策略目标：R=L_TAD+lambda*(c-target)，critic b(C)仅依赖采样动作之前可知信息（不能看到本次采样K后再作为整个joint policy的baseline）。
L_actor=stopgrad(R-b(C))*log pi(S,K|C)；最小化。L_critic=0.5*(b(C)-stopgrad(R))^2。
Detector照常最小化L_TAD；actor reward必须detach，critic/Router和detector梯度作用域有单测。不能错误负号优化成偏爱高损失。
默认lambda初值0.1，用训练EMA成本残差做投影dual update，eta=1e-3，[0,10]；固定K不靠lambda选数量。
默认cost为实际shape推导的normalized heavy MAC（以全量heavy子图为分母；总模型MAC另报），并记录counter口径；hardware-LUT是独立注册变体，不能在同一训练中无版本记录地更换cost目标。

### 5.2 在线acquisition校准
从epoch6开始，每32个minibatch选1个原计划未选action，额外运行当前模型L(S+a)，target=L(S)-L(S+a)。共享同视频、增强、GT normalizer；以共同随机数控制dropout差异。target stopgrad；Huber损失权重0.1监督gain head。主gain是对当前集合分布边缘化的预期，不是精确集合条件效用。
A/C有跨token/跨层依赖时必须重跑受影响路径，不能使用dense cache假装反事实；B只有编码独立且receiver关系正确时允许缓存。全量预算无未选候选时跳过acquisition label，不创造假label。
主模型不混用retention标签；删去证据的价值、swap收益与新增价值分开诊断。

### 5.3 一次60epoch课程
epoch0–5：随机预算/随机位置训练receiver和cheap state，Router只运行不做actor/gain更新。
epoch6–20：random exploration比例从0.5线性降至0.1。
epoch21–59：比例0.1。
随机分支不谎称on-policy learned sample：actor只用真实从learned policy采样的样本；比例为动作无关常量时可按1/(1-eps)校正缩放。detector训练所有分支。inference为确定性budget选择+Top-K，validation评估stochastic→deterministic差异。

save checkpoints epoch5/20/40/59用于已注册训练动态诊断。最终主表用59；不需等待其他模型的checkpoint才能训练。

### 5.4 比较估计器
PG-only；Hybrid(PG+acquisition)；gate-gradient retention；zero-gated acquisition probe；ST-hard-forward（有偏）；直接actionness路由。
标记哪些训练时执行了额外候选或dense supernet。所有训练GPU-hours、forward次数、峰值显存单独报告。

## 6. 固定协议与风险评价

以官方split为准；从官方train按video_id确定性hash留出10%作为internal_dev/calibration，两个子任务再用hash分隔。主training使用余下90%，所有方法相同；主official held-out不用于选超参、lambda、budget阈值或ROI。若用户必须使用全官方train，则统一以单独internal seed train/subset协议开展调参，锁参后全量重训全部主方法；不能只重训自己方法。

THUMOS14主mAP按官方常用0.3:0.1:0.7，另报AP0.8/0.9诊断；ActivityNet按官方0.5:0.05:0.95及0.5/0.75/0.95。必须与所用官方评估器匹配并记录commit。
短/中/长阈值来自训练集duration分位数，另给absolute-duration bins作为补充。用固定per-video top-N proposal cap=100与score阈值仅internal_dev校准。class-agnostic boundary error和class-aware recall均报，漏检不能从边界统计中消失；同时报告matched fraction和capped全GT误差。

native metrics + gap到边界距离、max gap/action duration、overlap、背景比例、跨parent clip、tail padding、真实PTS可用性、低运动代理。小物体和ROI语义相关性没有标注时标NA，不能靠运动阈值冒充小物体标签；可创建盲审subset并保存标注协议和一致性。

## 7. 正确性硬门槛（不是科研结果门槛）

- 全选择输出及adapter梯度匹配原模型：FP32 atol1e-5 rtol1e-4；AMP单独记录容差。
- 原生pair不跨错帧，selected rank不重编号PE。
- parent attention隔离，无跨视频污染。
- K=0、K=all、重复indices、空GT、tail padding、VFR、ROI越界可运行。
- physical support gaps不被hull填平。
- 修正坐标后的存储排列不变性。
- frozen weights梯度None、adapter梯度nonzero。
- packed与reference数值/梯度一致；不要拿不同数学模型比较速度。
- no-heavy-mask诊断不属于label泄漏自动证明；selection是来自输入的合法信号。

未通过本路线的正确性测试不能进入该路线长训练；其他已通过路线立即运行，不等待它。


完整baseline/ROI/预算参数解释还须读取VARIANT_RULES.zh.md；两文冲突时以其显式变体解释为准，任何进一步改变需amendment。
