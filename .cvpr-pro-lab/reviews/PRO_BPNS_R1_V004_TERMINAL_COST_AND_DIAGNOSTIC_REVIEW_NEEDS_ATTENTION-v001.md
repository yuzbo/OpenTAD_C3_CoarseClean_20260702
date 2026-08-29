# BPNS-R1 v004 post-result Pro：提交前模型路由阻塞

- request：`PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_REQUEST-v001`
- nonce：`ZOOMTOKEN-BPNS-R1-V004-TERMINAL-PRO-v001-20260829T130000+0800`
- exact Project：`g-p-6a79701398bc8191a9ef61db6302b24b`
- profile/CDP：`61 / 127.0.0.1:39567`
- 登录探针：`sessionStatus=200`，`sessionAuthenticated=true`，`domLoginCta=false`
- exact Project URL：已到达并核验
- 状态：`PRE_SUBMISSION_MODEL_TIER_UNAVAILABLE`
- 实际科研提交数：`0`
- follow-up：`0`
- Project Sources mutation：`false`
- response：不存在

Oracle 在附件上传和 prompt 发送前尝试选择冻结要求的最高浏览器可验证 `Pro` 路由。当前模型选择器只暴露：`Extra High`、`GPT-5.6 Sol`、`GPT-5.5`；不存在可验证的 `Pro` 选项。三次只读选择匹配均失败，随后 invocation 终态 `FAILED`。没有上传附件、没有创建科研 conversation、没有提交 prompt，也没有 Pro 科学内容可摄取。

不得把 `GPT-5.6 Sol` 或 `GPT-5.5` 擅自冒充 Pro，不得启动第二个科学请求、follow-up 或复用旧 conversation。原 request、nonce 与七个附件保持冻结；只有 profile 61 的 exact Project 模型选择器重新提供可验证 Pro 路由，或人类明确修订模型路由要求后，才可恢复同一未提交请求。

原始证据：

- streaming receipt：`.cvpr-pro-lab/reviews/PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_STREAMING_RECEIPT-v001.json`
- terminal receipt：`.cvpr-pro-lab/reviews/PRO_BPNS_R1_V004_TERMINAL_COST_AND_DIAGNOSTIC_REVIEW_TERMINAL_RECEIPT-v001.json`
- Oracle log：`.cvpr-pro-lab/reviews/runs/zoomtoken-bpns-r1-v004-terminal-pro-20260829t130000/oracle.log`
