# GeoSparse-TAD 当前实现与实验审查入口

这是用户要求发布的当前代码审查快照，整理于 **2026-09-08**。仓库默认分支属于另一个 C3 项目；请以打开本文件的 **GeoSparse 提交** 为审查对象，不切回默认分支，不混用 C3 模型。

**完整审查请求：[PRO_REVIEW_PROMPT.zh.md](review/current/PRO_REVIEW_PROMPT.zh.md)。**

## 固定版本与用途

| 对象 | 精确身份 | 含义 |
| --- | --- | --- |
| 当前运行模型 M | `b70ae056c495b43ca3f305fe438926b97b2723b5` | 两台服务器正在使用的冻结实现；本审查分支的 `geosparse_ext/` 与其完全相同 |
| 官方 OpenTAD O | `346d09d19e2091372cec48172dbe40f7b28bdee6` | `opentad/`、源配置、原 train/test 入口的固定官方基座 |
| 最新研究输入 | `290e51b4c35238f93b1274edb6b512f03ded1d66` | SCI3 成对干预、官方 AP bootstrap、固定预算诊断及当前说明 |
| 最新执行工具输入 | `c33c7b1662340f8066e84f0b77b1940a265a4aef` | 审计修复分支的执行、能力和资源核验工具 |
| 本分支 | `codex/geosparse-pro-current-20260908` | 合并上述输入和可公开的真实实验材料；最终提交 SHA 以本文件所在 permalink 为准 |

[整理检查](review/current/verification.json)确认模型子树与 M 相同、官方核心与 O 相同、1545 个实际任务 ID 唯一、依赖可解析、full 与两台服务器 focus 的共有任务记录一致。**这些检查不是“代码没有错误”的结论，也不证明完整训练数学等价于官方。**

## 推荐阅读顺序

1. [当前修复与重训边界](docs/GEOSPARSE_AUDIT_REPAIR.zh.md)、[用户最新选点与开发协议](docs/evaluation/test-development-amendment-20260908.md)。
2. `geosparse_ext/model.py`、`sparse.py`、`routing.py`、`detector.py`、`receivers.py`、`evidence.py`；随后沿真实数据/训练/验证/导出/计时调用链阅读。
3. [动态预算审计](docs/evaluation/dynamic-budget-audit.md)、[性能诊断](docs/evaluation/performance-diagnosis-20260908.md)、[实际学习曲线](review/current/results/learning_curves.csv)、[七项完整预算回放](review/current/results/budget_replay_seven_cases.json)。
4. [完整机器矩阵](review/current/manifests/experiments.full.jsonl)、[逐任务状态](review/current/status/job_disposition_1545.csv)、[逐配置状态](review/current/status/configuration_disposition_204.csv)。
5. [SCI3 接入边界](docs/evaluation/sci3-integration.md)、`geosparse_research/` 与对应测试；[v3 原始材料](review/current/reference/research_spec_v3.zh.md)仅作为需求资料，不作为已实现证明。
6. [旧独立审计](review/current/reference/prior_independent_audit.zh.md)固定的是旧 `55d5429e/902fa05`，用于逐项复核修复，不把旧缺陷自动指认为当前缺陷。

## 当前结果的关键边界

| 结果 | 截止本材料采集时的事实 | 不能推导的结论 |
| --- | --- | --- |
| 官方独立原版训练 | 已完成60轮，best E52 **70.1806%**，E60 **69.8879%** | 不等于发布权重复测71.1388%；不等于统一Geo dense训练 |
| A dynamic | 已完成60轮；训练内best E40 **68.0196%**，独立同best导出 **68.0156%** | E40全部792窗口选择q1，不能作为50%成本成绩 |
| A fixed q=.5 | best E50 **61.2806%**；完成55轮后等待资源续训 | 尚不是完成60轮的最终结论 |
| B dynamic | E20 **30.2768%**；E10全部选择q=.25 | E10预算不能替代E20预算，更不能作为最终60轮结论 |
| C dynamic | E20 **53.4452%**；E10全部选择q1且all-fine | E20缺预算账本，不能从精度或cost EMA推断同成本改善 |
| B/C fixed q=.5 | 已有中期曲线 | 不能将早期低分写为完整路线失败 |
| B-full / unified dense | 原登记任务已开始训练，暂无可用结果 | 不能预先量化B架构损失或统一训练器偏差 |

全测试预算回放是**实际视频、实际EMA Scout、源路由计划**产生的选择，加上源Heavy形状公式的MAC计算；它不运行Heavy，不重新测精度，也不是实测延迟。每个预算点必须绑定同一epoch/权重的已有精度。

C 动态 E20 的新回放仅补此前不存在的预算证据：Slurm **245503**，检查全部211视频/792窗口，采集时 `PENDING (Priority)`。见 [请求状态](review/current/results/c_dynamic_epoch020_budget_review.json)。不重复此前七项回放，不修改源模型或重训。

## 矩阵与尚未实现的研究

实际正式矩阵为 **204配置、612 train、612 evaluate、204 benchmark、117 diagnostic，共1545任务**。612项训练均登记60轮、seed0/1/2；当前只推进seed0。最新状态表为123配置静态准入、81配置显式阻塞；117个登记诊断仍阻塞。静态准入、排队、运行和科学结果完成是不同状态。

SCI3材料声称的805个请求未取得原始机器清单，不能用自行编造ID补齐。30个研究工单、22图和6表已按原文登记；强MoD/CoDA/LITE、R1、swap训练、mTAN/provenance gate、新merge基线及严格成本分配器等尚未完成。现有公共成对干预和bootstrap接口不能替代它们。

## 数据与验证材料

- `review/current/results/`：完整验证receipt的必要字段、真实学习曲线、训练动态、有限probe统计、预算记录和控制器状态。时间/epoch以每个文件内容为准，均是有限时点快照。
- `review/current/verification.json`：Git树和矩阵检查；17项SCI3及6项固定预算诊断测试的既有收据。测试使用合成输入/CPU，不称真实TAD科学验证。
- `review/current/operator/`：产生这些汇总和C20请求的操作员脚本；站点私有连接/资产绑定不发布，脚本不是无需绑定就可独立运行的部署包。
- 未包含数据集、视频、checkpoint张量、凭证或全部原始长日志。审查者无法据此独立重算mAP时应明确说出。

## Pro 咨询状态

[consultation.json](review/current/consultation.json) 当前是 **NOT_SENT_TOOL_BLOCKED**。Computer Use 在执行任何页面操作前报 `failed to write kernel assets ... os error 3`；尚未观察到ixBrowser页面或确认Pro模型，不能声称已经咨询。完整提示已准备。若恢复指定工具，通过ixBrowser的ChatGPT Pro发送本提交链接与提示；不改用API、其他浏览器或模型。
