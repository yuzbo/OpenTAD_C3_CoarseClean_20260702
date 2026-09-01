ZOOMTOKEN-CONTINUOUS-ROI-S2-V2.2-EXACT-BYTE-TERMINAL-PRO-v001-20260901T075800+0800::g-p-6a79701398bc8191a9ef61db6302b24b

```text
request_id=PRO_CONTINUOUS_ROI_S2_V2_2_EXACT_BYTE_RECOVERY_TERMINAL_ADJUDICATION-v001
exact_project_id=g-p-6a79701398bc8191a9ef61db6302b24b
profile=61

actual_attachment_count=13
actual_attachments_read=
1. ../../../ztv22recovery/docs/methods/continuous_roi_s2_v2_2_exact_byte_search_manifest.json
2. ../../../ztv22recovery/tools/validate_continuous_roi_s2_v2_2_exact_byte_recovery.py
3. ../../../ztv22recovery/tests/test_continuous_roi_s2_v2_2_exact_byte_recovery.py
4. .cvpr-pro-lab/reviews/PRO_CONTINUOUS_ROI_S2_V2_2_EXACT_BYTE_RECOVERY_TERMINAL_ADJUDICATION_REQUEST-v001.md
5. PAPER_PROGRESS.md
6. research-wiki/query_pack.md
7. research-wiki/anti_repetition.md
8. docs/aris/ZOOMTOKEN_PRO_CODEX_RESEARCH_ROLES.md
9. docs/aris/ZOOMTOKEN_CONTINUOUS_ROI_S2_V2_2_EXACT_BYTE_RECOVERY_TERMINAL_RECEIPT-2026-09-01.md
10. .cvpr-pro-lab/evidence/continuous-roi-v22-exact-byte-058095cc/recovery_inventory.json
11. .cvpr-pro-lab/evidence/continuous-roi-v22-exact-byte-058095cc/terminal_receipt.json
12. .cvpr-pro-lab/project-files/prepared/CURRENT_RESEARCH_STATE-v019.md
13. .cvpr-pro-lab/project-files/prepared/MODEL_EXPERIMENT_HISTORY-v014.md

repository_url=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702
implementation_branch_url=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-continuous-roi-s2-v2-2-exact-byte-recovery-v001
implementation_commit_url=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/058095cc763756dd941f6f113fca90f4fd54123c
evidence_branch_url=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-cvpr2027
terminal_evidence_commit_url=https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/078633518439a8c0fcbfbf7ed4791d30feaad8f2

browser_visible_model=GPT-5.6 Pro
browser_visible_effort_route=Pro / highest browser-verifiable Pro tier / maximum exposed effort
```

上述请求身份、附件传输方式、单次科研提交要求和五个 GitHub 身份与主请求一致。实现分支当前精确指向 `058095cc…`；证据分支当前已经前进到 `a51522ab…`，其直接父提交才是本轮冻结的 `07863351…`，因此证据身份必须以精确提交而不是可移动分支头为准。

## 一、Executive verdict

```text
overall_decision=PIVOT
terminal_classification_decision=ACCEPT_AS_WRITTEN
historical_exact_nine_route=PERMANENTLY_CLOSED
engineering_classification=PASS_STRONG_WITH_NONMATERIAL_DISCLOSURES
protocol_classification=VALID_AND_COMPLETE_FOR_THE_ONE_BOUNDED_EXACT_BYTE_CENSUS
scientific_classification=NO_MODEL_SCIENTIFIC_RESULT
paper_claim_classification=NO_METHOD_OR_PERFORMANCE_CLAIM
role_contract_decision=KEEP
```

我接受：

`STOP_CONTINUOUS_ROI_S2_HISTORICAL_REFERENCE_ROUTE_ARTIFACTS_UNRECOVERABLE`

但必须把它解释为**冻结项目协议内的永久关闭**：

> 由 D160/G96/U128 × seeds 3407/3408/3409、原 campaign、原 checkpoint SHA、原 sidecar SHA、原 v2.2 reference protocol 共同定义的历史 exact-nine 路线，已经永久失去形成合法 reference execution 的条件。

这不等于“这些字节在宇宙任何位置都不可能存在”，也不等于 retention 操作是每个文件缺失的唯一可证明因果路径。它表示：一次且仅一次、事先冻结范围的合法搜寻已经完成，结果为 `0/18`；项目不得再扩大根目录、进行第二次扫描、按日志重建文件、用哈希替代文件、从部分工件拼装 reference，或在旧身份下重训后声称“恢复”。原始终态同时证明没有训练、模型 forward、推理、预测、metric、cost 或 official-test 行为，因此它不是 Continuous-RoI 的科学正结果或负结果。 

## 二、GitHub 实现与证据审计

### 2.1 工程证据

`058095cc…` 相对于冻结 protocol base 增加的科学执行面恰好是：

* 一个自哈希、有限源集合的 search manifest；
* 一个只读、exact-byte、all-or-none validator；
* 一个 focused test 文件。

实现分支当前仍以该提交为头。终态证据收据同时记录 local `23 passed, 3 skipped`、N16R4/Linux `26 passed`、fresh Critic `PASS`、result-blind Evaluator `PRE_RUN_READY`、manifest SHA `3754d24f…` 和 protocol SHA `644f0c56…`。这些只证明该 census 具备执行资格，不是模型效果证据。 

实现的关键正确性如下：

1. **查询对象不可漂移。** Validator 从冻结 protocol 重建精确的 9 个 checkpoint 和 9 个 sidecar，并要求与 manifest 中的 18 行逐项相同。
2. **搜索范围有限。** Payload tree 只扫描冻结根、深度至多 4、只接受两个冻结 basename，不跟随 symlink，不允许根据结果扩大搜索根。
3. **catalog 永远不是 payload。** Retention manifest、journal 和 applied record 只能证明历史记录，不能成为 checkpoint 或 sidecar 字节。
4. **来源时间受约束。** 每个候选都必须早于预冻结 cutoff；后创建的副本不能伪装成历史工件。
5. **恢复是 all-or-none。** 只有 18 项全部各自拥有唯一、provenance-valid、SHA 精确匹配来源时，才允许在独立 quarantine 中复制；任何部分匹配都不能发布。
6. **原 campaign 只读。** 扫描前后对目录、文件 mode、mtime、size 做 metadata snapshot；变化即形成 blocker。
7. **终态原子发布。** Inventory 与 terminal receipt 先在临时根写完，再整体 rename；测试覆盖了中途失败不得产生半个正式结果根。
8. **测试覆盖了决定性失败面。** 包括 hash 漂移、越界 basename、symlink、部分恢复、重复来源、晚于 cutoff、嵌套目录 mutation、覆盖写、半发布、formal exception、precheck exception 和 catalog-as-payload。

我同时记录三个**不改变本次结论**的工程披露：

* `load_and_validate_contract()` 位于 formal `try` 之外，因此如果 manifest/protocol 在装载阶段损坏，程序会直接退出，而不是发布 `formal_action_incomplete` 收据。实际正式调用已经通过合同装载并完整生成 18 行结果，所以该缺陷不影响本次终态。
* `all_sources_preexisting_and_provenance_valid` 在 `run_formal()` 内主要由候选文件集合计算；候选数为零时会出现逻辑真空。实际 precheck 已独立核验 source roots 和三个 catalog 文件身份，所以本次“来源已冻结”的判断仍有直接证据，但以后不应把该字段单独当作充分证明。
* Manifest 的 `documented_absence_cause` 文字把 retention 操作写成了删除原因；catalog 能证明对应删除记录存在，却不足以排除所有其他因果链。故最终论文或研究记录只能写 `cause_of_absence=UNKNOWN_WITH_RETENTION_RECORDS_PRESENT`，不能写成逐文件唯一因果归因。

因此工程分类不是无保留的“claim-grade perfect”，而是：

`PASS_STRONG_WITH_NONMATERIAL_DISCLOSURES`。

### 2.2 协议证据

协议在它真正承担的问题上是完整的：只问“冻结的 18 个历史 payload 是否能从冻结来源中以原始字节形式全部找回”。它没有把部分恢复、日志、哈希文本或重新生成文件当成成功，也没有在看到结果后增加搜索表面。

正式结果为：

* scan ordinal `1/1`；
* formal invocation ordinal `1/1`；
* candidate payload files `0`；
* checkpoint matches `0/9`；
* sidecar matches `0/9`；
* 18 行全部 `MISSING_EXACT_BYTES`；
* 原 campaign root 前后均为 117 entries；
* metadata SHA 前后均为 `19524244…`；
* reconstruction `false`；
* quarantine `false`；
* GPU、training、forward、raw inference、prediction、metric、cost、performance access、official test 全部 `false`。 

因此协议分类是：

`VALID_AND_COMPLETE_FOR_THE_ONE_BOUNDED_EXACT_BYTE_CENSUS`

但不是：

`VALID_REFERENCE_EXECUTION_PROTOCOL`

因为 reference execution 所必需的 checkpoint 与 sidecar 本身不存在。

### 2.3 科学证据

本轮没有运行模型，没有读取性能，也没有形成 prediction。因此：

* 不支持 Continuous-RoI 成功；
* 不支持 Continuous-RoI 失败；
* 不支持 fixed-size crop 优于 variable-size crop；
* 不支持 variable-size crop 优于 fixed-size crop；
* 不支持 crop sufficiency；
* 不支持 accuracy、短动作、边界、latency、energy、memory 或 Pareto 结论；
* 不支持把旧九个训练作业写入性能表。

最新 `CURRENT_RESEARCH_STATE-v019` 和 `MODEL_EXPERIMENT_HISTORY-v014` 对此边界是正确的：exact-byte terminal 永久关闭历史 exact-nine reference route，但未来合法 crop 实验必须具有新的训练身份、协议身份和主张边界。 

### 2.4 论文主张证据

可进入论文或补充材料的内容仅限：

> 历史 Continuous-RoI S2 training-only artifacts 因 final checkpoint 与 sidecar 未被保留，无法执行预注册的 reference evaluation；因此它们没有被用于任何性能结论。

不得把 `0/18` 放进方法性能表，不得把它标成 accuracy/cost negative，不得把 retention 清理记录包装成方法失败，也不得用旧 training loss、GPU memory log 或 training completion receipt 推断模型质量。

## 三、历史 exact-nine 路线的最终处置

历史路线现在永久关闭，具体含义是：

1. 旧 jobs `1177668–1177676` 只保留为 `PASS_TRAINING_ONLY` 历史事实。
2. 旧 D160/G96/U128 × `3407/3408/3409` checkpoint SHA 和 sidecar SHA 不再作为未来执行输入。
3. 禁止第二次 census、全盘 root scan、对象存储搜索、S3 搜索、日志重建、checkpoint synthesis、partial checkpoint、相近 checkpoint、同路径覆盖或在旧 namespace 中重新生成文件。
4. 禁止把任何 fresh training 称为“恢复”“复现旧字节”或“历史 reference replacement”。
5. 概念性的 Continuous-RoI、D160/G96/U128 表征族和 fixed/variable geometry 问题并未被关闭；它们只能在全新的科学身份下重新接受检验。

这一边界与 v014 的现时研究记忆一致。

## 四、独立下一科学决定

我不下达 documentation-only 任务，也不继续 artifact recovery。

前面的固定 token 削减、固定 depth sparsity 和 ordered-decode reuse 已经多次显示：结构 FLOPs 或 token 减少不自动转化为足够的 full-stack p50 收益。相反，Continuous-RoI S2 至今只有训练完整性，没有一次 development prediction、mAP、reference headroom 或成本结果。

因此下一项最高信息增益任务应当是：

```text
ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FRESH-3X3-MATCHED-TRAIN-REFERENCE-AND-COST-FALSIFIER-v001
```

它不是恢复历史 exact-nine，而是一个**全新训练身份的三族、三种子、完整模型 falsifier**。

科学问题是：

> 在相同训练配方与相同总表征预算下，`G96 + source-coordinate local U128` 的 Continuous-RoI 表征，是否能够保持 D160 的 TAD 定位质量；在 shared physical center、equal search privilege 下，variable-size crop 是否真的比 fixed-size crop提供增量 headroom；并且这一表征路径是否存在真实 full-stack cost 余量？

现有 S2 设计明确区分 D160、G96 和带两个真实分支的 U128，并把 `96+128 versus 160` 作为预注册表征假设；v2.2 又冻结了 FS/VS shared physical center、17 个 Sobol candidates、raw/privileged separation、fit160/gate40 和 129-window population。它们足以支持一次新的实证检验，但旧 checkpoint 不再可用。

## 五、唯一任务的冻结科学合同

### 5.1 执行身份

```text
execution_base=10aed28659a08fa703def278fc0f5f1422dcad89
new_branch=codex/zoomtoken-continuous-roi-s2-v3-fresh-3x3-v001
new_campaign_identity=required
historical_campaign_reuse=false
historical_checkpoint_or_sidecar_reuse=false
```

`058095cc…` 与 `07863351…` 只作为历史 stop 证据，不合并为模型恢复输入。

### 5.2 冻结三族与种子

新矩阵必须是：

* `D160-V3`；
* `G96-V3`；
* `U128-V3`；
* seeds `4407/4408/4409`。

采用 disjoint seeds 是为了明确区分 fresh experiment 与旧 exact-nine；不得回到 `3407/3408/3409` 并声称恢复。

三族必须继承已经审过的 S2 科学语义：

* D160 是同一 S2 runtime 中的 dense-160 comparator；
* G96 是 matched global-96 表征控制；
* U128 使用 matched G96 global 分支与 source-coordinate local-128 分支、一个共享 VideoMAE 参数实例、两次真实分支执行和既有 AdaTAD-derived detector 后端；
* S2 不训练或嵌入 deployable learned ROI policy；
* 不新增 head、teacher、distillation、loss、selector 或 cache。

### 5.3 数据人口与训练完整性

* development fit/gate 固定为 `160/40`；
* sanitized raw gate population 固定为 `129` ordered windows；
* official test 始终关闭；
* 每个 cell 完整训练 60 epochs、80 successful updates/epoch、4,800 successful updates；
* 同一 pretrained initialization family、optimizer、loss、augmentation、AMP、EMA、batch semantics 和 detector；
* final `state_dict_ema` 是唯一主 checkpoint；
* 不允许 early stopping、best-checkpoint selection 或中间 metric 选模；
* 每个 cell 必须原子保存 `epoch_59.pth`、metadata sidecar、rendered config 和 completion receipt；
* final artifacts 必须保留在新的 immutable campaign root，不执行全局 retention 清理。

历史 S2 的训练完成只能证明 optimization/exposure integrity；本任务必须完整进入 prediction 和 reference evaluation，才形成模型证据。

### 5.4 Reference evaluation

只有九个 fresh cell 全部通过 strict finalizer 后，才允许 reference evaluation。

必须逐字继承 v2.2 protocol core：

* Torch `2.0.1` Sobol；
* `dimension=48`；
* scramble `true`；
* seed `20260720`；
* 1 anchor + 16 non-anchor candidates；
* 48 tubelets；
* FS 与 VS 每个 tubelet 使用相同 decoded physical center；
* FS 固定 anchor area/aspect；
* VS 使用冻结的 transformed area/aspect；
* result-blind enumerated IDs；
* raw GPU 过程无 GT、annotation、target cache、teacher 和 preferred ID；
* raw payload 先 canonical SHA seal；
* 独立 CPU privileged join 后才可读取 development GT；
* FS-PREF、VS-PREF 和 D0-PREF 使用相同 candidate privilege 和 join；
* D0、Short-Q1、boundary、bootstrap、max-T、missing-evidence 规则不变；
* 所有既有数值阈值和 outcome state machine 不得重写。

v2.2 明确说明它不改变模型、metric 或 threshold；本任务只把旧 `frozen_training_identities` 替换为事先冻结的新 v3 identities。

### 5.5 完整成本

若且仅若 reference finalizer 得到有效的：

```text
S_CR=true
H=true
```

才允许一次成本作业。

其中：

* `S_CR`：按冻结 v2.1/v2.2 state machine，Continuous-RoI representation sufficiency 成立；
* `H`：在 shared physical centers 和 equal privilege 下，VS 相对 FS 的 adaptive headroom 成立。

成本作业必须：

* 在同一张物理 GPU 上；
* 使用三个 fresh seed 的 final EMA；
* 测完整 decode → crop/resize → H2D → one/two real VideoMAE evaluations → detector → postprocess → full-video Soft-NMS；
* 报告完整 pass 的 p50、p95、throughput、peak allocated/reserved memory 和 gross GPU energy；
* 把 17-candidate exhaustive reference search cost单独报告，绝不能伪装为 deployable-policy cost；
* 使用冻结 cost-family 阈值形成 `F`，不得从结果修改阈值。

### 5.6 唯一接受规则

```text
PASS_TO_FRESH_PRO_FOR_S3_CONSIDERATION
iff
S_CR=true AND H=true AND F=true
AND all three seeds complete
AND all evidence/provenance/no-leak gates pass
```

即使 PASS，也只表示：

* source-coordinate crop representation 在该 development protocol 下有充分性证据；
* variable size 在 equal-privilege fixed-size comparator 之上有 headroom；
* 当前 representation path 有成本可行性。

它仍不证明 learned S3 policy、official-test、跨数据集、跨 detector 或最终论文优越性。

### 5.7 停止规则

* `S_CR=false`：永久停止这个精确定义的 S2-v3 representation family，不开 S3。
* `S_CR=true, H=false`：记录“continuous representation sufficient but no variable-size headroom”，停止 learned variable-size geometry 路线。
* `S_CR=true, H=true, F=false`：允许保留 crop-headroom 科学结果，但停止其当前效率路线，不开 S3。
* 任一 cell、raw inference、seal、privileged join、population、metric 或 cost evidence 不完整：`NO_DECISION_INVALID_EVIDENCE`，不得解释部分结果。
* 任一 valid negative 不准增加 seed、candidate count、crop size、loss、head 或替代阈值进行 rescue。
* 任一 objective blocker 都直接返回 fresh Pro；没有自动修复或 replacement。

## 六、允许路径

只允许以下新执行面：

```text
docs/methods/continuous_roi_s2_v3_fresh_3x3_protocol.json

configs/adatad/thumos/continuous_roi_s2_v3_fresh/
  d160_seed4407.py
  d160_seed4408.py
  d160_seed4409.py
  g96_seed4407.py
  g96_seed4408.py
  g96_seed4409.py
  u128_seed4407.py
  u128_seed4408.py
  u128_seed4409.py

tools/bata/run_continuous_roi_s2_v3_fresh_3x3.py
tools/bata/evaluate_continuous_roi_s2_v3_reference.py
tools/bata/profile_continuous_roi_s2_v3_cost.py
scripts/run_continuous_roi_s2_v3_fresh_3x3_n16r4.sh
tests/test_continuous_roi_s2_v3_fresh_3x3.py
```

既有模型、dataset、transform、VideoMAE、Adapter、ActionFormer、evaluator 和 NMS 源码全部只读。若现有 S2 runtime 不能在这一边界内执行，Codex 必须返回客观 blocker，不得自行扩大模型修改面。

## 七、正式提交和 replacement 上限

```text
formal_campaign_count=1
training_cell_submissions=9
training_finalizer_submissions=1
raw_reference_submissions=3
privileged_join_and_reference_finalizer_submissions=1
conditional_fullstack_cost_submissions_max=1
maximum_scheduler_submissions=15
replacement_count=0
second_campaign=false
```

九个 training cells 和 training finalizer 必须由一个原子 deployment manifest 绑定。Raw reference 每个 seed 一项，只有 training finalizer 全 PASS 才能释放。成本作业只有 `S_CR=true && H=true` 才能释放。

## 八、禁止工作

本任务禁止：

* 继续搜寻旧 checkpoint；
* 旧 SHA-byte reconstruction；
* 历史 namespace 写入；
* resume 或重提旧 jobs；
* 使用部分历史 cell；
* official test；
* S3 learned policy；
* validation/test GT 进入 raw GPU 图；
* teacher、distillation、new loss、new head；
* candidate count、Sobol seed、crop-size、area/aspect、seed 或 threshold sweep；
* 按中间结果选 checkpoint；
* 把 exhaustive reference search 当部署成本；
* 把 tests、training completion、FLOPs 或 token/pixel count当性能证据；
* 第二次 formal campaign、replacement、rescue 或 post-hoc threshold revision。

## 九、角色合同裁决

```text
role_contract_decision=KEEP
```

现有合同已经正确规定：

* Pro 独立拥有科学问题、机制、实验、证据解释和路线选择；
* Codex 只实现冻结的唯一任务；
* Critic 检查会改变准入或解释的决定性问题；
* Evaluator 不选路线；
* smoke、tests 和工程成功不是效果证据；
* 有效负结果必须保存；
* 终态返回 Pro 后才能改变路线。

本次失败来自历史 artifact retention，而不是角色职责冲突，因此没有理由修改角色文件或 RTK 规则。

```text
role_contract_replacement_text=NOT_APPLICABLE
rtk_role_rule_update=RECORD_KEEP_ONLY; NO_RULE_TEXT_CHANGE
```

## 十、北京时间期限

本裁决制定时，北京时间为 `2026-09-01T08:32:58+08:00`。

```text
builder_plan_due_at=2026-09-01T12:00:00+08:00
role_rules_sync_due_at=2026-09-01T12:00:00+08:00
builder_candidate_due_at=2026-09-02T06:00:00+08:00
critic_due_at=2026-09-02T09:00:00+08:00
evaluator_due_at=2026-09-02T12:00:00+08:00
formal_action_due_at=2026-09-02T14:00:00+08:00
terminal_evidence_return_due_at=2026-09-06T12:00:00+08:00
mandatory_fresh_post_result_pro_review_due_at=2026-09-06T16:00:00+08:00
```

终态证据应在最后一个已授权作业 terminal 后 4 小时内返回；`2026-09-06T12:00:00+08:00` 是硬截止。排队未开始、资源不足或任何 formal blocker 也必须在该时点前以不含部分性能解读的客观证据返回。

## 十一、可直接摄取的执行块

```text
decision=PIVOT
terminal_classification_decision=ACCEPT_AS_WRITTEN
accepted_terminal_classification=STOP_CONTINUOUS_ROI_S2_HISTORICAL_REFERENCE_ROUTE_ARTIFACTS_UNRECOVERABLE
historical_exact_nine_route_status=PERMANENTLY_CLOSED
cause_of_absence=UNKNOWN_WITH_RETENTION_RECORDS_PRESENT

engineering_classification=PASS_STRONG_WITH_NONMATERIAL_DISCLOSURES
protocol_classification=VALID_AND_COMPLETE_FOR_THE_ONE_BOUNDED_EXACT_BYTE_CENSUS
scientific_classification=NO_MODEL_SCIENTIFIC_RESULT
paper_claim_classification=NO_METHOD_OR_PERFORMANCE_CLAIM

selected_task=ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FRESH-3X3-MATCHED-TRAIN-REFERENCE-AND-COST-FALSIFIER-v001
scientific_purpose=Freshly test D160/G96/U128 representation sufficiency, equal-privilege variable-size headroom, and full-stack cost viability without recovering or inheriting historical payload artifacts.

role_contract_decision=KEEP
role_contract_replacement_text=NOT_APPLICABLE
rtk_role_rule_update=RECORD_KEEP_ONLY; NO_RULE_TEXT_CHANGE

execution_base=10aed28659a08fa703def278fc0f5f1422dcad89
candidate_branch=codex/zoomtoken-continuous-roi-s2-v3-fresh-3x3-v001
historical_evidence_only_commits=058095cc763756dd941f6f113fca90f4fd54123c,078633518439a8c0fcbfbf7ed4791d30feaad8f2

allowed_paths=docs/methods/continuous_roi_s2_v3_fresh_3x3_protocol.json; configs/adatad/thumos/continuous_roi_s2_v3_fresh/d160_seed4407.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/d160_seed4408.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/d160_seed4409.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/g96_seed4407.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/g96_seed4408.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/g96_seed4409.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/u128_seed4407.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/u128_seed4408.py; configs/adatad/thumos/continuous_roi_s2_v3_fresh/u128_seed4409.py; tools/bata/run_continuous_roi_s2_v3_fresh_3x3.py; tools/bata/evaluate_continuous_roi_s2_v3_reference.py; tools/bata/profile_continuous_roi_s2_v3_cost.py; scripts/run_continuous_roi_s2_v3_fresh_3x3_n16r4.sh; tests/test_continuous_roi_s2_v3_fresh_3x3.py

model_dataset_transform_source_changes_allowed=false
data_population=fit160/gate40; sanitized 129 ordered gate windows; official test sealed
training_matrix=D160/G96/U128 x seeds 4407/4408/4409
training_completeness=60 epochs; 80 successful updates per epoch; 4800 successful updates; final state_dict_ema only
reference_protocol_core=inherit v2.2 shared physical centers, 17 candidates, 48 tubelets, no-GT raw seal, equal-privilege CPU join, unchanged metrics/thresholds/state machine
cost_scope=conditional full decode-to-Soft-NMS p50/p95/throughput/allocated/reserved/gross-energy; exhaustive reference-search cost reported separately

formal_campaign_count=1
maximum_scheduler_submissions=15
replacement_count=0
second_campaign_allowed=false

acceptance_rule=S_CR=true AND H=true AND F=true AND all three seeds complete AND all identity/no-leak/population/evidence gates pass
stop_rule=S_CR=false stops exact S2-v3 representation; H=false stops variable-size/S3; F=false stops current efficiency continuation; incomplete evidence yields NO_DECISION_INVALID_EVIDENCE; every terminal returns to fresh Pro
claim_boundary=Development-level three-seed S2 reference and systems falsifier only; no learned-policy, official-test, cross-dataset, cross-detector, universal crop-family, or final-paper claim.

forbidden_work=second exact-byte scan; old artifact reconstruction; old namespace reuse; partial historical recovery; official test; S3 policy; teacher; distillation; new loss/head; model-source changes; candidate/seed/crop/threshold sweep; intermediate checkpoint selection; replacement; rescue; post-hoc threshold revision

required_tests=protocol-core identity; nine-config matchedness; D160/G96/U128 real-shape forward/backward; shared-weight two-branch U128; exact update/EMA completion; raw object-graph no-GT; candidate known answer; 129-window population identity; raw seal/privileged join separation; atomic final artifacts; cost full-stack accounting
critic_scope=mechanism/config/runtime fidelity, matchedness, no-leak, checkpoint retention, state-machine identity, evidence completeness
evaluator_scope=result-blind exact revision/data/pretrain/config/job/root verification and PRE_RUN readiness; no partial metrics
formal_action=one atomic fresh 3x3 campaign, then gated three-seed raw reference sweep and at most one conditional same-GPU cost job

builder_plan_due_at=2026-09-01T12:00:00+08:00
role_rules_sync_due_at=2026-09-01T12:00:00+08:00
builder_candidate_due_at=2026-09-02T06:00:00+08:00
critic_due_at=2026-09-02T09:00:00+08:00
evaluator_due_at=2026-09-02T12:00:00+08:00
formal_action_due_at=2026-09-02T14:00:00+08:00
terminal_evidence_return_due_at=2026-09-06T12:00:00+08:00
mandatory_fresh_post_result_pro_review_due_at=2026-09-06T16:00:00+08:00
expected_return_bound=within 4 hours of the last authorized terminal, hard no later than 2026-09-06T12:00:00+08:00

post_result_pro_trigger=any complete terminal or objective blocker
post_result_pro_required_evidence=exact clean/pushed revision and URLs; frozen protocol/config hashes; Critic/Evaluator reports; all scheduler identities; nine final checkpoint and sidecar hashes; completion receipts; raw sealed predictions; privileged-join outputs; full metric vectors; Short-Q1 and boundary diagnostics; full cost/power rows if authorized; deviations; missing evidence; anomalies; exact claim boundary
post_result_pro_decision_scope=independently adjudicate S_CR/H/F, alternative explanations, evidence grade, paper usefulness, exact-route stop or S3 consideration; no automatic successor
next_owner=Codex Builder
```
