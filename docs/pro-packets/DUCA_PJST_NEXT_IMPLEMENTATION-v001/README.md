# DUCA PJST-D1 下一步实施问询材料包

目标 Project：`g-p-6a9061a41bbc819190f4cde94a6c733c`

代码库：`https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702`

## 使用顺序

先向同一个全新 DUCA Project 会话附加以下材料：

1. `01_CURRENT_RESEARCH_STATE.md`
2. `02_EVALUATOR_RAW_EVIDENCE.md`
3. `03_PROJECT_RULES.md`
4. `04_RESEARCH_ROLE_RULES.md`
5. `05_PAPER_FIRST_RESEARCH_SKILL.md`

随后把 `00_PROMPT.md` 的完整正文作为该会话唯一一轮用户问题提交。

不要附加整个代码库、历史聊天、旧 Prompt、服务器日志、检查点或数据文件。代码通过 GitHub 地址和 Prompt 中的精确 revision 定位。

## 文件职责

- `00_PROMPT.md`：要求 Pro 作出唯一科学裁决并下达一个当前任务。
- `01_CURRENT_RESEARCH_STATE.md`：论文问题、当前方法、代码身份、结果和证据边界。
- `02_EVALUATOR_RAW_EVIDENCE.md`：PJST-D1 OFF/ON 推理与统计失败的原始实验回执。
- `03_PROJECT_RULES.md`：项目科学目标、数据、训练、公平比较和证据规则。
- `04_RESEARCH_ROLE_RULES.md`：Pro、实现者、独立审查者和实验评估者的职责。
- `05_PAPER_FIRST_RESEARCH_SKILL.md`：通用论文优先科研流程。

## 提交边界

- 必须创建全新会话，不复用旧对话或追加第二轮问题。
- 必须在提交前后确认实际 Project 是 DUCA。
- 本轮只请求科学裁决和一个下一任务，不授权训练、代码修改或 held-out 评估。
- 若浏览器未登录、Project 身份不明确或文件未完整附加，应在提交前停止。
