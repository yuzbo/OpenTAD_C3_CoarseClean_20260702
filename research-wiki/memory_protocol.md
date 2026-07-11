# Research Wiki 维护协议

## 每个 agent 开始工作前

必须依次读取：

1. `research-wiki/query_pack.md`
2. `research-wiki/anti_repetition.md`
3. `research-wiki/current_direction.md`
4. `research-wiki/decision_register.md`
5. `research-wiki/lessons.md`
6. 与任务相关的 `research-wiki/routes/*-complete-record.md`
7. 与任务相关的 idea/experiment 页面

若用户最新明确决策与 Wiki 冲突，以用户最新决策为准，但必须同步新增决策记录，不能静默覆盖历史。

## 何时必须更新

- 新 idea 被提出、合并、否定或恢复；
- 新评审指出新的风险或推翻旧判断；
- 实验提交、取消、失败、完成或产生 result-to-claim 裁决；
- 当前主线、主张门槛、数据契约或坐标契约改变；
- 论文使用了新的方法名或范围声明。

## 更新动作

1. 更新对应实体页；
2. 只在 `graph/edges.jsonl` 增加关系；
3. 在 `log.md` 追加不可变记录；
4. 重建 `index.md` 与 `query_pack.md`；
5. 检查 `query_pack.md` 小于 8000 字符；
6. 不在 Wiki 复制未经核验的实验数字，数字只链接 `docs/evaluation/results.md` 或对应权威结果文件。

## 全量回顾的强制发现步骤

在使用“所有、完整、全面、最终历史”等表述前，必须：

1. 扫描当前仓库的 `git worktree list`、local/remote branches 和关键 commit ancestry；
2. 对每个历史 worktree 查找 specs、plans、reviews、results、核心源码与测试，不得只读当前 HEAD；
3. 将未推送分支、未合入 commit、外部附件和远端实验记录写入 `source_map.md`；
4. 在 `discussion_coverage.md` 中标明已覆盖、刻意去重和无法访问的缺口；
5. 对“已实现”“已验证”“已推送”“论文可用”四种状态分别给证据，禁止互相代替。

扫描结果同步刷新 `worktree_inventory.md`；不能只在终端中检查后遗忘。

## 防止原地打转

- 任何“再加一个 prior/loss/top-k/gap weight”的建议，先查 failed ideas。
- 任何“已经端到端/动态/物理时间”的表述，先查 claim boundary。
- 任何新实验，必须先指定它淘汰哪个不确定性；不能只因为脚本可运行就排队。
- 任何旧 run 都必须标注 commit 与 evidence category。
- 任何路线恢复必须写 superseding decision 和新证据。
- 任何“某路线没有实现”的判断必须先检查其他 worktree/branch；任何本地未推送实现必须同时记录可复现风险。
