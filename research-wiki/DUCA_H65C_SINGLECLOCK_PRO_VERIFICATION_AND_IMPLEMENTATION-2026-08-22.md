# DUCA H65C SingleClock：Pro 裁决核验与实现终态

- 日期：2026-08-22
- 科学状态：`REVISE_ACCEPTED / IMPLEMENTATION_PACKAGE_CLOSED_BLOCKED`
- 证据等级：代码核验、独立静态审查与只读资源核验；没有 PRE_RUN、训练、推理、mAP 或成本结果

## 1. Pro 原文身份

独立 Pro 原文位于
`C:/Users/skywalker/.codex/oracle/duca-h65-truetime-pro-20260822-v002/final.md`，
原始 Oracle 日志位于
`C:/Users/skywalker/.fastctx/jobs/j-y27w4i/output.log`。原文 SHA-256 为
`55954295c8224a476c1119bd6509b4c9d9b1b938bfc31a9d41dc8b26b60f3740`；
用户粘贴件只比原文多 Windows CRLF 换行，统一为 LF 后 SHA-256 完全相同。

本轮是 exact DUCA Project 中一次完成的 fresh Pro 审查。模型路由证据为 requested
`gpt-5.5-pro`、resolved `Pro`、picker verified。唯一裁决为 `REVISE`，建议冻结
`DUCA-H65C-SINGLECLOCK-DYNAMIC-v002`。

## 2. 逐项核验与吸收

### 2.1 接受的判断

1. 历史 H65 的 `65.385724` 不是单一“非均匀输入格式”的结果，而是 30-epoch
   exact-uniform Stage-1 EMA 与 60-epoch Stage-2 的组合：sampling-rate systematic exact-K、
   50% uniform companion、分类/回归 contribution distillation、完整 ASFormer action/transition
   adaptation、selected-axis detector 和 NMS 前物理坐标回映共同存在。当前 UVT、Fovea、
   RankPack/TrueTime 均替换了其中多项，因此它们的下降不能单独归因于 Query 或物理时间。
2. 先恢复 H65 兼容合同，再只加入一个可归因的时间机制，是比继续叠加 UVT/Fovea 组件更可靠的
   falsifier。Query 只能在后续作为 scout 语义残差；dynamic outer-K 只有在固定 K 表示门通过后才进入。
3. fixed K=384 只用于机制归因，不构成最终动态预算主张。已有 dense、uniform、random 等历史控制
   不再重复训练。

### 2.2 经代码核验后的限定

1. 现有 VideoMAE 没有独立的“时间注意力层”。可实现的最小定义是：在首个 ViT 时空自注意力中，
   对所有空间 token 对共享同一个 tubelet 时间相对偏置；不得将其表述为独立 temporal attention。
2. H65 的 score threshold/top-k 只读取分数，随后进行 q→physical 映射，再执行 NMS。为保持单变量
   归因，Unit-1 保留这个历史顺序；关键不变量是任何 IoU、NMS、投票和序列化前都处于物理时间。
   不接受在同一实验中同时移动 score-only 筛选顺序并加入 SingleClock。
3. Pro 给出的单 seed `-0.20` 点非劣阈值只可作为开发门，不能作为论文级非劣结论。高 gap-CV 与
   boundary-density 风险分层必须在运行前固定定义，并以逐视频配对不确定性报告。
4. H65 历史 checkpoint 未被 Pro 独立重放，因此只有 checkpoint、配置、参数组、预处理、选帧、
   gathered RGB、VideoMAE 输入、raw proposal、NMS/evaluator JSON 全部匹配，才允许把回放称为 H65 身份。

## 3. 实现周期与终态

独立 clean worktree 为
`E:/DeskTop/TAD/OpenTAD_DUCA_H65C_SingleClock_20260822`，分支
`codex/duca-h65c-singleclock-unit1-20260822`，冻结父提交为
`42dba3f90b37243e7965d18b6707e88e81bf7109`。候选提交依次为：

- `bb7fc4237ebd2ebe20cfe89c75a870f963a851f3`：首个最小包；
- `45c92dc1f0ffffe7178ed48ab3f17eabd77ce109`：配置、位置分段和 fail-closed validator；
- `0b81bb11216b6ac8eca664a69eb17b504c1e5917`：24×16 分段与 activation-checkpoint 位置切分；
- `87d9a1aef355a508b5324b0469f5a68d0f967cfe`：零初始化 scalar 与 uniform 无 mask 快路径。

已正确实现的局部合同包括：实际嵌套 VideoMAE 配置接线、物理位置按 24×16 分段、tubelet-center
位置、activation checkpoint 同步切分、共享 `[B,1,N,N]` 偏置、不支持的 packed runtime fail-closed、
历史 threshold/top-k→物理映射→NMS 顺序、缺失 Stage-1 身份时的 fail-closed validator。

终态独立复核仍发现一个决定性错误：`87d9a1ae` 在每个 flattened 16-frame clip 内重新执行
`exact_uniform_positions(768,16)`（tubelet 前实际为 16 帧），而正确 canonical 必须先按 H65 生成一个
全局 `exact_uniform_positions(L,K=384)`，再按全局 rank 切成 24×16。当前实现丢失了 clip 的全局 rank
offset，因此比较的不是 H65 canonical clock。现有 focused tests 也没有覆盖全局 K384 canonical 切片、
uniform attention 无 mask spy 或 nonuniform scale gradient。该错误改变被测试的时间机制，不能带入实验。

终态为 `IMPLEMENTATION_PACKAGE_CLOSED_BLOCKED`。本周期没有第三次之后的继续修补、Evaluator 准入或
Slurm 提交；科学路线没有被效能实验否定。

## 4. 资源核验与客观阻塞

N16R4 上的 THUMOS14 canonical 原始视频、注释、类别映射、VideoMAE-S 预训练、OpenTAD Python 环境、
Slurm 与存储资源可用。真正阻塞 H65 身份的是：

- Stage-1 epoch-29 EMA 记录路径的 checkpoint 目录为空；声明 SHA-256
  `7233fa6944659f432f8deaf22448b4a25cf8794b1e912f59a4d5b3715d54b39e` 无实体可核验；
- Stage-2 e9 回放 checkpoint 与 epoch-59 H65 checkpoint/prediction/evaluator 实体没有在旧运行根定位；
- 当前启动器只是 allocation 内执行入口，缺少绑定具体 Slurm partition/account 的已验证提交收据。

因此，即使代码错误被新的 clean 实现周期修复，仍必须先从权威备份恢复 exact Stage-1 EMA 并核验 SHA。
重新训练 Stage-1 可以构成一个新实验，但不能冒充历史 H65 identity-equivalent 回放。

## 5. 唯一后续动作

当前不运行训练。恢复条件同时包含：

1. 从权威备份恢复并核验 H65 Stage-1 epoch-29 EMA；
2. 新的 clean 实现周期把全局 H65 canonical K384 序列及 clip rank offset 贯穿到首层注意力；
3. 独立 Critic 与 Evaluator 在 N16R4 通过 actual-config、uniform byte identity、非均匀 scale gradient、
   replay identity 和 Slurm PRE_RUN。

只有三项同时满足，才允许提交一个 seed 3407、60 epoch/6000 successful updates、每 5 epoch 可恢复
checkpoint、终态 final-EMA 的 Unit-1 完整实验。通过只产生单种子机制证据；Query 与 dynamic-K 仍不启动。
