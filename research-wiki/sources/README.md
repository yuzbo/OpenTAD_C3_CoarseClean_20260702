# 原始讨论归档

本目录用于保留可审计的任务原文，不作为实验事实或当前结论来源。

## 已归档

- `thread-019f49d2-user-record.md`：DUCA 主任务按时间顺序导出的用户侧完整记录，覆盖 191 轮、158 条用户消息，包含目标、纠偏、附件引用、实验追问、PIVOT、ChronoTransport 与 PhysTime 转向。
- `delegated-thread-recent-record.md`：目标/实现/部署、论文写作和早期目标梳理等代理任务的近期导出，保留代理反馈与跨任务衔接。

固定哈希：

- `thread-019f49d2-user-record.md`：`56DA6D521871DC99F9CD4EEAEE8F474D2D21DCDE327D7A61C8FAEE5C5BDCFD65`
- `delegated-thread-recent-record.md`：`9D02A2B893FCAD1326F56615813B1D3922187DCE3DC187ADF5BDFCF8432600C9`

两份文件最初固定于 local commit `026f127`，并随 ChronoTransport branch 的 `92029ea` 保留。本次迁入当前 PhysTime Wiki，避免该未推送分支丢失后无法追溯。

## 使用边界

1. 原始聊天可以证明“讨论过什么”，不能证明“代码已实现”或“实验有效”。
2. 方法状态以 `routes/`、`decision_register.md` 和源码 commit 为准。
3. 实验状态与数字以 `experiment_register.md`、`docs/evaluation/results.md` 或正式 artifact 为准。
4. 附件正文仍通过 `source_map.md` 的绝对路径和 SHA256 固定。
5. 导出发生后的新讨论通过 `log.md`、路线档案和覆盖矩阵追加；不能改写原始导出。
