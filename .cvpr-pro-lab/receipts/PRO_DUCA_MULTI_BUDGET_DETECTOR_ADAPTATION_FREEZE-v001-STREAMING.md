# STREAMING RECEIPT — DUCA 多预算检测器适应冻结

- request: `PRO_DUCA_MULTI_BUDGET_DETECTOR_ADAPTATION_FREEZE-v001`
- nonce: `DUCA-MULTI-BUDGET-DETECTOR-ADAPTATION-FREEZE-v001-20260831`
- exact Project: `g-p-6a91061f789881918ccd8357ca3d6c92`
- conversation ID: `6a9521de-d020-83e9-a0b9-19045c8d5390`
- conversation URL:
  <https://chatgpt.com/g/g-p-6a91061f789881918ccd8357ca3d6c92-duca/c/6a9521de-d020-83e9-a0b9-19045c8d5390>
- Oracle session: `duca-multi-budget-adaptation-freeze-2`
- profile/CDP: `61` / `127.0.0.1:15359`
- browser target: `D38CF9543E1A12D32EB3F0C03CF12EA1`
- model picker: `Pro`, `verified=true`
- effort: `MAX_EFFORT_NOT_SEPARATELY_EXPOSED`
- promptSubmitted: `true`
- state: response generation running

第一次控制平面尝试在提交前的新标签页导航阶段进入 `chrome-error://chromewebdata/`，Oracle 将其归类为未登录。
该次没有模型选择、对话 URL 或 prompt submission，属于 attempt0 传输失败。只读 CDP 检查随后仍显示已登录的精确
DUCA Project 页面。一次同 nonce 传输恢复重新获取相同共享锁并成功提交；它不是第二个科研回合。

当前绑定来自 Oracle session `meta.json`：状态 `running`，exact Project conversation URL、conversation ID、
`promptSubmitted=true` 和经 ChatGPT model picker 核验的 `Pro` 全部一致。Prompt 和五份最小充分材料以内联文本提交，
没有修改 Project Source。

原 Oracle invocation 与 profile/project/turn 锁继续持有。不得新开 Pro 对话、follow-up、重提、切换 Project、关闭
或重启 profile61。按 RTK，长时间思考不是失败；只在终态或外部信号出现时回取同一 session。

