# 本次审查的有效协议

更新时间：2026-09-08。用户最新要求覆盖原始计划中的不一致内容；原始文件保留用于对照，不是当前执行命令。

新增执行优先级：[完整动态方法与固定50%先行](amendments/20260908-full-methods/FULL_METHODS_FIRST.zh.md)。当前是六项seed0主实验，最初五项focus中的B-full、额外稠密对照延期。以下数据、模型与训练协议不变；旧五项队列描述只保留作历史记录。

|项目|当前有效要求|被撤销或必须区分的内容|
|---|---|---|
|模型基座|官方 OpenTAD `346d09d19e2091372cec48172dbe40f7b28bdee6`|C3 fork 的模型不能冒充原版 AdaTAD|
|训练集|全部 200 个官方训练视频|180 训练＋20 留出被用户否决|
|正式验证/测试|全部 211 个官方测试视频；官方 test pipeline 792 窗口|GPU 预检中的 1 窗口只是正确性检查，不得出正式指标|
|内部诊断|可观察全部训练视频，不从训练集扣除；明确不是 held-out|`internal_dev` 留出语义已废弃，消费者遗留待修复|
|默认输入/输出|768 帧、160px、768 检测网格|旧 224px/grid384 为已撤销默认；只能在新协议中明确注册为独立消融|
|原生结构|tubelet=2；48 个 parent，各 16 帧；TIA 原始时间长度384|不能全局拼接 Heavy attention，也不能把 TIA 缩成单 parent|
|优化|全局 batch2；head/adapter LR1e-4；WD0.05；warm-up5；cosine horizon100；实际60epochs|不能把 cosine 周期改为60，或把官方 batch2 理解成每卡2|
|官方独立训练验证|原版配置 start40/interval2/checkpoint2；完成42/44/…/60epochs后全量验证|保留官方周期，不能宣称每5epochs是其默认|
|GeoSparse与其稠密控制验证|每5epochs全量验证，尽早观察|与官方验证次数不同，比较时必须披露|
|选点|各配置完整测试集 EMA 精确 Avg-mAP best；保留完整预测；并列取较早点|原始“主表 final epoch59”已被用户 best 要求替代；选 test best 的偏差需如实说明|
|种子|当前只推进seed0；seed1/2保留注册但延后|不能以3种子均值/方差展示单种子数据|
|初始化|识别预训练；无独立 Dense TAD Teacher|公开 TAD checkpoint 只作单独复测，不作训练初始化|
|路线|A主线，B/C平行挑战；正确性/资产/物理资源依赖允许|不允许性能或Oracle晋级门槛|
|旧实验|保留记录并撤销主表资格；不续训成新协议|不得混淆旧 `official_dense` 标签与真正官方模型|

160px、patch16 下每个 temporal slice 为100个原生空间 token；2×2分组为25个。C混合粒度计数为 `25(1-p)+100p`。计数、实际 Heavy MAC 和完整模型成本分别记录。A仍支付解码、patch embed、规则状态、dense TIA以及Scout等成本。

修订后的完整矩阵为204个配置与数据集组合、1545项任务；当前focus是5个seed0训练＋相应子任务。官方原版训练/发布权重复测独立管理，没有把其成绩伪装成focus中的 `geosparse_dense_control`。

以下是当前实现边界，不是对原始目标的完成声明：ROI、二轮获取、若干梯度估计器、TriDet、ActivityNet/FineAction接入、D01–D06套件等仍有显式阻塞。全部参数状态见静态登记，但静态接受仍须审查实际语义。

当前模型快照902fa05保留；审查材料分支未修改模型。发现的迁移遗留应由后续独立修复提交解决，不靠改文档把错误说成已修复。

官方来源：[AdaTAD](https://github.com/sming256/AdaTAD)、[固定版本结果表](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/configs/adatad/README.md)、[B配置](https://github.com/sming256/OpenTAD/blob/346d09d19e2091372cec48172dbe40f7b28bdee6/configs/adatad/thumos/e2e_thumos_videomae_b_768x1_160_adapter.py)。本地发布权重复测凭证在 `evidence/official_released_checkpoint/`。
