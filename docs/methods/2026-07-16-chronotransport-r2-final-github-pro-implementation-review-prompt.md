# ChronoTransport r2 GitHub Pro 审计入口（两次讨论）

原单次全量 prompt 已停止作为直接审计输入：用户报告 Pro 模型超过思考时长。当前审计必须依次
使用两份较短、依赖封闭的 prompt：

1. [Part 1：快照、注册与 Gates 1–3](./2026-07-16-chronotransport-r2-github-pro-audit-part-1-foundation.md)
2. [Part 2：Stage C、Gate 4 与最终裁决](./2026-07-16-chronotransport-r2-github-pro-audit-part-2-final.md)

## 使用顺序

### 第一次讨论

只提交 Part 1。建议在调用消息中写入：

```text
EXPECTED_REVIEW_SHA=<fresh GitHub snapshot SHA>
```

若不提供该字段，Part 1 会 fresh-resolve 一次并冻结 SHA，而不是仅因缺少 caller 字段停止。保存 Pro
的完整原文输出，尤其是 `PART1_AUDIT_PACKET`。Part 1 不允许给整体 registration verdict。

### 第二次讨论

提交 Part 2，同时附上第一次讨论的**完整原文输出**。建议再次写入相同的：

```text
EXPECTED_REVIEW_SHA=<the same SHA used in Part 1>
```

若不提供，Part 2 必须使用 packet 的 `review_sha`。它会独立复核同一 SHA，继承 Part-1 blockers，
完成 Stage C/Gate 4/Slurm 审计并给唯一整体裁决。

## 不变边界

- 两次讨论必须绑定同一个 immutable GitHub SHA；
- 第一次若为 `GITHUB_SNAPSHOT_INCOMPLETE`，不得启动第二次；
- 第二次缺少完整 Part-1 输出时只能返回 `PART2_INPUT_INCOMPLETE`；
- 两份 prompt 均只读，不授权代码修改、I/R、PRECHECK、CUDA/Slurm 或实验；
- 只有 Part 2 的 exact `APPROVE_IMPLEMENTATION_FOR_REGISTRATION` 才允许把该 SHA 指定为 I；
- 即使批准，也不自动创建 R 或解锁实验。

不要再把本索引或旧单体正文直接提交给 Pro；它们不是第三份审计 prompt。
