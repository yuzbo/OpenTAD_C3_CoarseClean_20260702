# 注册变体的补充实现规则

用途：禁止agent对未明确的baseline或参数自行做有利于自身路线的解释。下面是设计默认，不是论文先验结论。

## A. F02选择器

- uniform：在native候选时间上分位点均匀选择，按video/seed固定tie-break；ST时使用规则空间覆盖pattern，不读GT。
- random：候选无放回均匀采样，训练按(seed,epoch,video,window)确定随机性；评价按(seed,video,window)固定，不多次抽样挑最好。
- motion：同一Scout分辨率下相邻样本灰度绝对差，按native组平均；摄像机运动不另引入昂贵flow模型。
- actionness：共享廉价特征预测“候选名义中心是否在任一GT动作区间”；只training用GT二元BCE；deployment用预测值排序。标签重叠区域仍为positive。
- uncertainty：同一个cheap actionness head的Bernoulli entropy；不声称它等同可改善任务风险。
- cdf_native：上述cheap actionness logit经softplus+1e-4形成密度；inverse-CDF分位点投到native atom，去重。为固定K公平性，剩余名额按未覆盖累计质量分位补齐并记录；动态去重数量另记录诊断，不能混成同K。hard离散采样不谎称存在TAD的路径梯度；该baseline由aux BCE训练scorer，不称原DUCA完整复现。
- pg_only：相同廉价上下文+标量排序头，以TAD cost的policy gradient训练；没有额外acquisition Huber。它是强标量router控制，避免把提升归因给“有一个小网络”。
- legacy：必须读取用户实际DUCA配置与代码，所有boundary/core/coverage分数、采样、插值、接收器和超参存档；缺任一关键内容为BLOCKED，而不是猜测。

所有选择器使用相同canonical候选与预算算子。额外cheap head算入训练/推理成本。none/static选择器不运行无用Router。

## B. 子集与预算

T-only候选1/2/4/8 tubelets须在同parent内组成不重叠native组；跨parent候选只在显式context变体出现。预算按有效native tokens计，group取整会偏离request，记录realized。

固定预算r是token更新占比，不直接称FLOP占比。dynamic r是期望归一化heavy cost目标；主默认heavy成本只计被路由的重型子图，fixed TIA/PE/Scout另报，总模型成本另报。因此r=0.5的fixed和dynamic在名字上相同，不保证真实cost匹配；主比较在internal_dev对实际cost校准/插值后的Pareto进行。

C的fine比例p与total mixed-token ratio不同；最小mixed r=.25。若任何硬件cost约束低于实现的最小开销，报告INFEASIBLE而不是让controller无限提高lambda或伪造满足。

## C. ROI注册变体

ROI变体采用B的时间packet选择+一个独立廉价ROI head；不是先按完整ST native mask删patch又裁图造成重复选择。时间预算按请求的native时间组分配，ROI选择从9个固定source-normalized anchor boxes中取1或2个，box大小为原图宽高的0.5，中心为{.25,.5,.75}×{.25,.5,.75}，越界clamp。

ROI head汇聚同一cheap spatial map的anchor区域并预测gain。2ROI使用IoU<0.5的去重选择，无法找到第二个有效ROI时明确缺省，不用重复ROI填满计算。ROI112/160/224仅表示编码输入尺寸，源crop保留原像素细节。时间预算、ROI数、输入尺寸一起决定actual cost。

tubelet内同ROI；smooth_trajectory对anchor中心做三tubelet时间平滑，保持native pairs，裁剪支持随实际中心更新。fullframe_fallback允许整帧作为额外互斥候选，cost正确计费。resize_lowres_negative_control只改变像素来源，不用作主方法。

## D. Sequential、depth和context控制

B两轮将同总heavy预算分为各半。第一轮沿默认策略；第二轮random或cheap/first-round detector的start/end邻域选择。候选仍投影到native组，pair共享包去重；缓存按照D06的依赖校验。初始proposal错误和遗漏必须计入结果。

static depth默认从原12层近似均匀保留{4,6,8,10}层，index选择由round(linspace(0,11,d))冻结。同配置保持width。PBD-inspired_tf只用当前模型training/internal_dev的删除影响在单run内渐进删至8层，不用dense teacher；它不等价原PBD，需要记录内部搜索成本。

BCR的prefix_depth为0/2/4层全量执行，suffix在clip/native时间组上route；当前A的prefix0结构不能因为名字不同重复宣称创新。共享context/TIA机制不同的部分必须记录。

cross_parent_selected显式允许跨parent heavy attention；它是context变化控制，必须计算更大attention代价，不能放入声称原dense-limit等价的结果。

## E. 尺度、位置和Scout输出对齐

Scout80/112/160输出统一到7×7候选坐标网格，用固定bilinear采样对齐原图支持。80变7×7不代表恢复了丢失细节。source_resolution112的B全帧fidelity使用实际7×7native patch tokens；nominal/source支持重新计算，不能沿用224时的14×14硬索引。

B在稀疏空间选择时，2×2slot池化只聚合对应support内实际存在的tokens；空slot带invalid mask，不填造语义。额外learned slots属于注册变体，不假定自动分成actor/object。

C fine分支的额外scale embedding为恒零或明确旁路，从而allfine输出测试不被新增参数破坏；coarse才使用scale tag。A额外physical bias属于独立变体，dense-limit control不偷偷包含未在原模型出现的偏置。
