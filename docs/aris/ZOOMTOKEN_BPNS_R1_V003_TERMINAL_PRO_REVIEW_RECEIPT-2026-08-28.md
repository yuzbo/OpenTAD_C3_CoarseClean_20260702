# ZoomToken BPNS-R1 v003 终态 Pro 复盘收据

- request：`PRO_BPNS_R1_V003_TERMINAL_PROTOCOL_FAILURE_REVIEW_REQUEST-v001`
- nonce：`ZOOMTOKEN-BPNS-R1-V003-TERMINAL-PRO-v001-20260828T200000+0800`
- Project：`g-p-6a79701398bc8191a9ef61db6302b24b`
- conversation：`6a919f06-bc94-83ea-b3e6-dd07f22375ee`
- URL：`https://chatgpt.com/g/g-p-6a79701398bc8191a9ef61db6302b24b-zoomtoken/c/6a919f06-bc94-83ea-b3e6-dd07f22375ee`
- 界面模型：`GPT-5.6 Pro`
- 传输：attachment-only，`browserInlineFiles=false`，七个附件
- 实际提交：1；follow-up：0
- 完成：`2026-08-28T22:59:47+08:00`
- transcript SHA-256：`b300f7c19dcf7fa8187865a5489cf95245124e61248290c55e1bca6bd286fcb2`

## 独立裁决

Pro 给出 `CONTINUE_ONCE_WITH_DECOUPLED_COST_CLOSURE`，角色合同 `KEEP`。v003 是工程缺陷触发、协议无效、科学无方向的终态：它既不支持也不否定 BPNS-R1 的效率主张。八个未舍入 evaluator vector 只支持单种子、固定 checkpoint 下“没有广泛准确率崩塌”的窄诊断；R1 相对 K100 的 Avg-mAP/mAP@0.3–0.7 差值依次为 `+0.5353/+0.7520/+0.1518/+1.6238/-0.1042/+0.2528 pp`。速度、能耗、显存、短动作与边界仍未知。

## 唯一任务

`ZOOMTOKEN-BPNS-R1-DECOUPLED-DIAGNOSTIC-AND-COST-CLOSURE-v004` 是当前唯一 successor。它从 `8a59d655005b9030d8ea5dc17ee2620844cb587b` 建立最小 clean descendant，只允许修改：

- `tools/bata/profile_zoomtoken_bpns_r1_cost.py`
- `tests/test_zoomtoken_bpns_r1_cost.py`
- `scripts/run_zoomtoken_bpns_r1_cost_n16r4.sh`

正式流程必须把成本采集、prediction identity 与离线 short-action/boundary diagnostics 解耦。每个 pass 结束后，先原子保存 raw cost、power coverage、prediction SHA 与 pass receipt；完成八个计时 pass 后才运行非计时诊断。成功路径上的 registry/factory 组件必须用真实 production config merge 构造，并在正式提交前执行一次 result-free known-answer call。

## 冻结判据

- pass 顺序：`K100,R1,R1,K100,R1,K100,K100,R1`。
- K100 SHA：`008daf5a55af90318506e913c13a4bd2d6ce8ff17a45cc8e856f5619eaa45eb7`。
- R1 SHA：`ffc78393e4097a578def8fdd62ffe4f36dd87c2dddd52de9b3ae248cb108c734`。
- 主延迟：`median4(R1 pass p50) / median4(K100 pass p50) <= 0.95`。
- 主能耗：`median4(R1 complete-pass joules) / median4(K100 complete-pass joules) <= 0.95`。
- pooled-window 比较不能替代上述四-pass 中位数主估计。
- 若任一主比值失败，停止 BPNS-R1 作为当前效率 headline 候选。
- 若在完整 raw cost acquisition 前再次协议失败，不授权 v005，不再重放；效率保持未知，转交 fresh Pro 作论文/资源层裁决。
- 若 raw cost 已完整而离线诊断失败，不得重放 GPU；保留原始证据并交 fresh Pro。

## 时间边界（北京时间）

Builder plan `2026-08-28 23:30`；clean candidate 与 local/N16R4 tests `2026-08-29 02:00`；Critic `03:00`；Evaluator `04:00`；正式提交 `04:30`；终态证据目标 `13:30`；fresh post-result Pro `14:00`；完整科学返回 `16:00`。无论实验是否终态，`2026-08-29 12:00` 向用户发送一次证据化进度报告。

## 原始证据

- Pro 可见回复：`.cvpr-pro-lab/reviews/PRO_BPNS_R1_V003_TERMINAL_PROTOCOL_FAILURE_REVIEW_RESPONSE-v001.md`
- streaming receipt：`.cvpr-pro-lab/reviews/PRO_BPNS_R1_V003_TERMINAL_PROTOCOL_FAILURE_REVIEW_STREAMING_RECEIPT-ATTEMPT2-v001.json`
- terminal receipt：`.cvpr-pro-lab/reviews/PRO_BPNS_R1_V003_TERMINAL_PROTOCOL_FAILURE_REVIEW_TERMINAL_RECEIPT-ATTEMPT2-v001.json`
- Oracle transcript/meta：`.cvpr-pro-lab/reviews/runs/zoomtoken-bpns-r1-v003-terminal-pro-attempt2-20260828t224500/oracle-home/sessions/zoomtoken-process-continuity-v003-pro/`
