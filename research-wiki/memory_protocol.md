# Research Wiki 维护协议

## 每个 agent 开始工作前

必须依次读取：

1. `research-wiki/query_pack.md`
2. `research-wiki/current_direction.md`
3. `research-wiki/decision_register.md`
4. `research-wiki/lessons.md`
5. 与任务相关的 idea/experiment 页面

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

## 防止原地打转

- 任何“再加一个 prior/loss/top-k/gap weight”的建议，先查 failed ideas。
- 任何“已经端到端/动态/物理时间”的表述，先查 claim boundary。
- 任何新实验，必须先指定它淘汰哪个不确定性；不能只因为脚本可运行就排队。
- 任何旧 run 都必须标注 commit 与 evidence category。
- 任何路线恢复必须写 superseding decision 和新证据。
