# DUCA 路线决策与代码整理：Pro 材料包

本目录用于一次新的、独立的 Pro 科学问询。目标是让 Pro 在完整但不过载的上下文中，独立决定当前值得继续的论文问题、权威代码主线和唯一下一项可证伪任务。材料不预设路线，也不把代码存在、局部测试或运行成功写成方法有效性证据。

## 使用方式

1. 将 `01` 至 `07` 作为 Project Sources 上传，保持文件名不变。
2. `00_PROMPT.md` 不作为背景 Source 重复上传；在新建的空白 Pro 对话中完整粘贴为唯一问题。
3. 在提交前核对目标为当前 DUCA Project，并使用最高可验证的 Pro 模型与最大可用推理强度。
4. 不引用旧对话中未包含在本材料包或固定 GitHub revision 中的隐含上下文。

## 上传清单与证据地位

| 顺序 | 文件 | 用途 | 权威边界 |
|---|---|---|---|
| 1 | `01_CURRENT_RESEARCH_STATE.md` | 当前论文问题、方法边界、完整结果表、未完成证据和唯一待裁决问题 | 当前科学状态摘要；如与原始证据冲突，以原始证据为准 |
| 2 | `02_CODE_INVENTORY_BOUNDARY.md` | GitHub 同步提交、实验 clean revision 与 dirty inventory 的区别 | 代码组织边界；inventory commit 不是实验身份 |
| 3 | `03_EVALUATOR_RAW_EVIDENCE.md` | PJST-D1 OFF/ON 推理、点估计和统计程序失败的原始证据摘要 | 只有点估计；0/10,000 bootstrap，不能形成总体效应结论 |
| 4 | `04_PROJECT_RULES.md` | 数据、基线、公平性、远端运行和论文证据规则 | 项目规则；历史小节中的旧任务名不代表当前路线 |
| 5 | `05_RESEARCH_ROLE_RULES.md` | Pro、Coordinator、Builder、Critic、Evaluator 的论文优先职责 | Pro 独立作科学决定，Codex 只执行冻结任务 |
| 6 | `06_PAPER_FIRST_RESEARCH_SKILL.md` | 通用的论文优先科研流程与中立 Pro 咨询规则 | 流程原则，不提供 DUCA 科学结论 |
| 7 | `07_HISTORICAL_MODEL_VERSION_REGISTRY.md` | 查找历史分支、提交和实现谱系，避免重复造轮子 | 历史索引，更新时间较早；不得覆盖 `01` 和 `02` 中的当前 clean revision |

## 代码真值

- GitHub：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`
- 完整代码库存提交：`5136011ed57df8a639427a633a488a592ba95924`
- 本轮提示词提交：`29eb779824d4f8146d557d1225c7634a0acba85c`

代码库存提交只用于让 Pro 检查重叠实现和历史表面。任何实验结论必须绑定 `01_CURRENT_RESEARCH_STATE.md` 与 `02_CODE_INVENTORY_BOUNDARY.md` 中列出的独立 clean revision、配置和原始结果。

## 明确不上传

- 不上传整个 `research-wiki/`：其中包含大量已停止路线、历史运行状态和重复叙述，会掩盖当前科学问题。
- 不上传 `query_pack.md` 与 `anti_repetition.md` 全文：它们保留在 GitHub 作为可追溯研究记忆；当前决定所需的有效边界已经在本包中汇总。
- 不上传数据、检查点、预测文件、服务器日志、仓库压缩包或 dirty 项目根快照。
- 不上传旧 Pro 对话或其摘要作为权威事实；Pro 应从固定代码和本包证据独立判断。

## 期望 Pro 输出

Pro 应独立给出继续、修订、转向或停止中的一个判断，并明确：当前论文问题与可证伪预测、唯一权威 clean 主线、保留/复用/仅历史保留/移出候选的代码表面、最小代码整理、唯一下一项决定性实验、成功与失败条件、负责人和绝对截止时间。Pro 可以拒绝提示词中隐含的候选划分并提出更好的单一路线。
