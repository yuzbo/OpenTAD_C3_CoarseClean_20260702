# 已核验的一手资料（访问日期2026-09-06）

这些资料用于背景与实现对照。本文提出的模型规格与预注册阈值不是来源已经证明的结果。

|ID|资料|一手入口|本项目用途|
|---|---|---|---|
|R1|AdaTAD, CVPR2024|https://arxiv.org/html/2311.17241v2|TIA、原生frame representation、PEFT；实际顺序仍以pin源码为准|
|R2|Official OpenTAD|https://github.com/sming256/OpenTAD|官方config、source、dataset和evaluation adapter|
|R3|CoDA|https://arxiv.org/html/2304.04947v2|dense-light/sparse-heavy已有先例，不能重复宣称首次|
|R4|Coarse-Fine Networks, CVPR2021|https://arxiv.org/html/2103.01302v2|Grid Pool/Unpool、非均匀时间重对齐先例|
|R5|ActionFormer, ECCV2022|https://arxiv.org/abs/2202.07925|多尺度时间检测器与固定stride语义|
|R6|mTAN, ICLR2021|https://arxiv.org/abs/2101.10318|不规则观测到固定参考时间点|
|R7|MSViT, ICCVW2023|https://arxiv.org/abs/2307.02321|输入混合尺度token；覆盖不代表信息无损|
|R8|PBD, CVPR2025|https://arxiv.org/abs/2503.16916|静态depth压缩强比较；modified teacher-free不冒充原算法|
|R9|Dynamic Feature Selection, ICML2023|https://arxiv.org/abs/2301.00557|获取信息价值与任务预测的区别|
|R10|AdaSpot, 2026|https://arxiv.org/abs/2602.22073|密集时间+ROI对照；事件点检测不等于interval TAD|
|R11|PyTorch logcumsumexp官方文档|https://docs.pytorch.org/docs/stable/generated/torch.logcumsumexp.html|稳定PL有序logprob参考实现|

论文与最终报告必须使用自己的同协议实验，不能直接将以上作者的mAP或speedup放进同硬件/同训练表。公开网站会变化，代码与软件必须pin commit/version，而非无限跟随main或stable。
