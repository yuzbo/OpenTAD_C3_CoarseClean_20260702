---
type: wiki_memory_audit
date: 2026-08-17
scope: ZoomToken / SCNR / GeoRoute / AdaTAD baseline and data-resource audit
status: complete_read_only_audit
evidence_boundary: historical-and-resource-accounting; no new experiment or method result
---

# ZoomToken 全量研究记忆与数据资源审计（2026-08-17）

## 审计结论先行

本轮逐页读完了本项目的 `RTK.md`、`PAPER_PROGRESS.md`、`research-wiki/` 的索引、
`query_pack`、`anti_repetition`、决策史、日志、来源登记、图关系、全部 ZoomToken /
SCNR / GeoRoute / ROI / residual idea 与 experiment 节点，以及其引用的官方 AdaTAD
配置、路由实现、P1/F0 作业回执和本地/远端资源。最重要的纠偏是：**69.03 / 48.27
是 AdaTAD 发布表的 upstream anchor；66.xx / 45.xx 是本项目 matched-source dense
full-compute 的官方 validation 输出，不是一次“精确、未修改 AdaTAD 官方复现”。**
两者不能互换，任何 ZoomToken 质量主张必须先经过精确官方 checkpoint 评测。

当前没有可投稿的 ZoomToken 效能或方法结果。现有可复核成果是路线设计、路由层静态
接线、少量开发诊断、完整的基础设施负证据，以及 matched-source 五臂训练输出；它们
不能被写成 exact-official baseline 或最终方法品质。

## 1. 原始目标、时间线与未被遗忘的教训

| 时期 | 已实现/观察到的事实 | 仍然有效的记忆与边界 |
| --- | --- | --- |
| C3/DUCA 早期 | 尝试低成本粗分类、transition/boundary-first、max-gap、下游 detector utility 与多种选帧路线。 | 任务是**离线** TAD，不是 Online TAD；选择机制要服务边界与高 tIoU，不能以昂贵的 X3D/SlowFast 密集先验掩盖总成本。旧 DUCA/X3D 数字仅历史诊断，不能嫁接给 ZoomToken。 |
| Native/SCNR 阶段 | 形成 native 物理 token、全局 unique budget、动态 `K_t`、`K_t=0` masked-zero 与 no-leak 约束；通过过生产选择器静态/局部机械检查。 | 不能把 `[B,T,K]` padding、固定每 tubelet top-k、role-copy 重复 token 或 content carrier 伪装成 ragged exact-B。 |
| ROI + residual 机制 | seed-3407 的 matched 开发诊断：uncentered `10.52/8.90/6.98`，centered `12.57/11.04/8.14`（Avg/@.6/@.7），即 `+2.05/+2.14/+1.16` pp。 | 这是单种子、development、机制诊断；residual centering 只允许作为 anti-collapse diagnosis，ROI/residual 不得抢占论文 headline。 |
| 成本诊断 | 冷态 ABBA/BAAB 诊断 p50 比 `0.95763`（95% CI `[0.78773,1.10761]`），能耗比 `0.94417`（`[0.79042,1.05603]`）。 | 两个上界都超过 1.05，故为 HOLD；不可写成节省成本。 |
| 路由接线与失败夹具 | `cd6463df…` routing-layer parity 通过；`b157433d…` detector/FPN fixture 确定性失败且已关闭。 | 前者只证明 config→wrapper→selector 接线；后者绝不可第三次修复或作为 detector-output/性能论据。 |
| P1/F0 基础设施 | 多个 F0/P1 epoch 因 config-role、work_dir、container git、MIG、attestor、telemetry、config-path、population 等确定性准入问题无效。`5491` 权威终结器为 `0/5` accuracy、`0/8` cost。 | 这些都是实施/协议负证据，**不是** Q-core 效能/成本反证，也绝不能从单臂、checkpoint 或半成品指标推出结论。 |
| 当前 matched-source 矩阵 | dense: seed3407/3408/3409 = `66.42/67.14/65.99` Avg，`45.19/45.84/45.02` @.7；Q: seed3408/3409=`57.84/53.81` Avg、`36.93/33.40` @.7；U/R 同源对照也已有部分输出。 | 这些是**修改后同源**的 full-compute / sparse matrix 输出：可用于该 source-family 内部的负向诊断，但不得称 exact AdaTAD 官方复现、不得证明最终方法质量，也不得越过缺失的 full-stack cost 与 exact official baseline。 |

## 2. 当前最终路线及严格证据边界

论文终点仍是：在离线 TAD 中减少 VideoMAE 的冗余**空间重计算**，同时保护短动作、
边界和高 IoU。被保留的机制候选是 task-aware dynamic spatial routing：scout 的
`q_base(t,n)` 经全局、物理唯一的 exact `B=24576` 选择，诱导动态 `K_t`（可为零），
仅对选中的 native token 做无 padding ragged heavy forward，其他位置为 masked-zero。

`DO/DN/U/R/Q` 是 P1 的公平控制；`G/N/F` 的 ROI/residual 是仅在 Q-core 通过后才可进入的
机制/因果控制。当前 Q 的两次 matched-source 输出是需要正视的负向诊断，但在 exact
official baseline 未锚定、训练源与发布路径未逐项对齐、成本终结器缺失时，它们不能升级为
论文的最终比较或科学 STOP。

## 3. 遗忘、陈旧或误标问题

1. `query_pack.md`、`anti_repetition.md`、`index.md` 和大量 July idea/experiment 节点仍以
   Hybrid-centered/DUCA 作为“current”；这是历史记忆，不是今天的 route pointer。它们必须
   保留来防止重走错误，但使用者需先读本审计和 `PAPER_PROGRESS.md` 的 baseline-first 更正。
2. `PAPER_PROGRESS.md` 与 `decision_history.md` 的 2026-08-16 条目曾把 66.xx 标为
   “official validation baseline/result”。其 evaluator 确实用了 official validation split，
   但代码/训练源不是未经修改的 AdaTAD 发布复现；本审计将其改标为 matched-source。
3. P1/F0 链中的“PASS”多数是 protocol/deployment PASS，不是训练/效能 PASS。尤其
   `5491` P1 的 epoch-59 checkpoint 不能作为可用模型、指标或选择诊断的证据。
4. Uni-AdaFocus 文献页仍有未填的 TODO；历史讨论已保存于日志和 source archive，但该页
   不应再被当作完整 prior-art card。需要后续文献整理，而不是替换当前 baseline-first 门。
5. `LINT_REPORT.md` 与 Wiki index 的日期/计数停留在旧代际；结构索引仍有价值，但不能单独
   用来判断本轮 ZoomToken 证据状态。

## 4. 官方 AdaTAD baseline-first 审计

| 项目 | 观察到的绑定 | 状态 | 对下一步的意义 |
| --- | --- | --- | --- |
| 官方 OpenTAD 参考 | `E:/DeskTop/TAD/OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817`，clean `346d09d19e2091372cec48172dbe40f7b28bdee6` | COMPLETE（本地 read-only ref） | 用于逐项 diff，不用当前 dirty ZoomToken worktree 代替。 |
| AdaTAD release | `E:/DeskTop/TAD/OpenTAD_OFFICIAL_ADATAD_01c58b9`，clean `01c58b9f2370e914150cf94d392208a4e211c053` | COMPLETE（本地 read-only ref） | `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` 的 published table 是 Avg `69.03`、@.7 `48.27`。 |
| 发布 checkpoint | 官方 README 给出 release checkpoint Google Drive ID `1HGUBroK90KBAkFqQreAVtHCIclJh7DmM`；已审计的远端 known paths 中未发现可绑定副本。 | UNVERIFIED / known-location MISSING | **首先阻塞** exact released-checkpoint evaluation；不得把 66.xx 当作替代。 |
| 官方训练配方 checkpoint | 官方 config 的 `workflow.checkpoint_interval=2`、`max_epoch=100`、end epoch 60。 | COMPLETE（配方事实） | 为 untouched official reproduction 保留 2 epoch，因其比项目默认更频繁。 |
| matched-source dense | 当前矩阵是 GeoRoute/matched source family，不等于 release `01c58b9`。 | PARTIAL provenance | 在官方 anchor 前只能称“matched-source dense”。必须 diff preprocessing、继承、pretrain、optimizer/schedule、EMA/checkpoint、evaluator/NMS、runtime。 |

## 4A. 共享官方 AdaTAD 基线的唯一执行与并行边界（2026-08-17）

用户已指定 ZoomToken 为所有相关 TAD 项目的唯一 shared official AdaTAD baseline executor。
唯一 packet 为 `docs/aris/ADATAD_SHARED_OFFICIAL_BASELINE_PACKET-2026-08-17.md`：它预先绑定
clean release `01c58b9…`、unmodified official THUMOS14 config、canonical 411、release checkpoint
或 training pretrain、固定 seed/evaluator/NMS/EMA-final、immutable runtime 和一个结果根。先且仅先
运行 released-checkpoint evaluation；只有 checkpoint 确实不可得且必须复现时，才由同一负责人
执行一次 clean official training。其他项目只能读取最终 receipt，禁止重复训练或把 66.xx
matched-source 输出补写成 shared baseline。

这项共享数值门只冻结“official dense 数字及由它承载的方法质量 claim”，不冻结 ZoomToken 的
方法准备：Q 动态空间路由的正式矩阵入口、ROI/残差的条件对照、5-epoch recovery policy、独立
审查和 PRE_RUN packet 都可继续，但不得启动新实验或把未绑定 dense 数字用于结论。

## 5. 数据与资源 read-only inventory

下表只报告本轮实际读取到的路径/计数；`MISSING` 是“已检查的规范路径中无”，不是对整个
远端的全盘否定。

**N16R4 read-only 登录与共享根**（不含凭据）：PowerShell 使用
`ssh -o IdentitiesOnly=yes -o PubkeyAcceptedAlgorithms=+ssh-rsa -o HostkeyAlgorithms=+ssh-rsa -i C:\Users\skywalker\.ssh\id_rsa -p 22 -l "sczc063@BSCC-N16R4" ssh.cn-zhongwei-1.paracloud.com`；
Git Bash 使用同一 host、`-i /c/Users/skywalker/.ssh/id_rsa`。共享根固定为
`/data/run01/sczc063/yuzibo`；不使用不存在的 `yuzibo` host alias。

| 必需项 | 可核验路径与观察 | 状态 | 阻塞的精确实验 | 最小合法动作 / 权限 |
| --- | --- | --- | --- | --- |
| THUMOS14 raw video + split | 远端共享只读 `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`：411 个 MP4 **symlink**，0 broken，目标总量约 33G；注释 UID 对齐为 training 200、validation 211、缺失 0、额外 0。物理 store 有 413 个视频；非 canonical 的额外文件是 `video_test_0000270.mp4` 与 `video_test_0001292.mp4`，不参与映射。 | COMPLETE | 不阻塞 clean official training 或 validation evaluation 的原始视频输入。 | 绑定该 shared root；不得复制数据，计数须跟随 symlink target 而非仅 `find -type f`；只用 canonical 411/annotation mapping。 |
| train/validation/test 语义 | 官方 config 使用 `training` 作 train、`validation` 作 val/test evaluator；当前项目没有另一个独立“official test” split，且 official test 始终 closed。 | COMPLETE（官方两 split 语义） / UNVERIFIED（额外 held-out test 不存在） | 任何把 validation 结果写成 external held-out test 的实验。 | 明确报告为 THUMOS14 validation；若要第三 split，需新数据协议和用户授权。 |
| annotations + categories + UID mapping | `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json` 1,312,424 bytes；`category_idx.txt` 258 bytes / 20 类。注释数：training 3,003，validation 3,325。 | COMPLETE | 不阻塞官方 loader/evaluator。 | 无需 acquisition。 |
| official evaluator/config | exact release checkout 含 official THUMOS config；本项目也有相同官方 config text，但远端 exact-evaluation launch packet 未绑定到 release checkout。 | PARTIAL | 发布 checkpoint 的 exact official evaluation；随后 clean reproduction。 | 在干净 `01c58b9` checkout 中固定 test command、config 和 evaluator digest；不需凭据，需项目方写 launch receipt。 |
| VideoMAE-S pretrain | `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth`，90,605,819 bytes。 | COMPLETE | clean official training reproduction 的 backbone pretrain 不被此项阻塞。 | 提交前绑定 file checksum/manifest；无需下载。 |
| AdaTAD released checkpoint | 规范的 release ID 已知；已检查 `/data/.../pretrained/AdaTAD`、`/data/.../checkpoints/AdaTAD` 与当前 known runtime locations，未见可验证 release artifact。 | UNVERIFIED / known-location MISSING | **首个 published-anchor checkpoint evaluation**。 | 以有权访问者的 Google Drive/发布镜像凭据合法定位或下载 release checkpoint，再记录来源、文件身份和 license；本轮未下载。 |
| E2E derived feature/cache/track/proposal | official VideoMAE E2E raw-video config 不需要预抽 feature/track/proposal；既有 136-window/40-cluster manifest 是 ZoomToken P1/G5 资产，不是官方 baseline evaluator 的必要输入，具体可复现路径本轮未找到。 | COMPLETE（official E2E 不需） / UNVERIFIED（P1 manifest） | exact P1 full-stack cost or any claim引用 136/40 manifest。 | 先从已登记的 manifest SHA/receipt 找到规范文件并做 read-only identity binding；无需新 route。 |
| OpenTAD I3D feature asset | expected `/data/run01/sczc063/yuzibo/thumos14/features/i3d_actionformer_stride4_thumos` 不存在；alternate `datasets/phystime_thumos_i3d/features/i3d_actionformer_stride4_thumos` 为空，只有约 746MB 未完成 `.part`。 | MISSING / incomplete alternate | 任何 I3D/ActionFormer stride-4 feature config 的完整训练或评测。 | 由拥有合法来源/凭据者完成官方 feature acquisition；不能借用 MATR/SigLIP2。 |
| InternVideo2 feature asset | expected `/data/run01/sczc063/yuzibo/thumos14/features/thumos14_6b` 不存在；仅 HuggingFace README 目录，无 tensor。 | MISSING | 任何 InternVideo2 feature config。 | 合法取得该特征或不选择这一路线；需访问凭据/下载授权。 |
| native MATR feature assets | `external_ontal_baselines/MATR_codebase/data/thumos_all_feature_val_V3.pickle` 与 `...test_V3.pickle` 存在，解析至官方 extracted assets，约 3.33GB + 3.69GB；MATR checkpoint 目录为空，只有 9MB incomplete `.part`。 | PARTIAL（features COMPLETE；checkpoint MISSING） | native MATR 训练/复现实验；不阻塞 VideoMAE E2E AdaTAD baseline。 | 仅为 MATR 路线合法补齐官方 checkpoint；不得把 pickle 接到 OpenTAD I3D/InternVideo2 `.npy` config。 |
| SigLIP2 feature assets | `/data/run01/sczc063/yuzibo/thumos14/features/pes_siglip2_stride8`：823 files / 约 477MB。 | COMPLETE（仅 SigLIP2 identity） | 任何错误把它当 I3D、InternVideo2 或 MATR source 的实验。 | 保持身份隔离；不作替代绑定。 |
| 已运行的 seed-3407 full-matrix config snapshot | root `/data/run01/sczc063/yuzibo/zoomtoken_full_official_b88a11ba_seed3407_20260815_070546`。DO snapshot `DO/gpu2_id0/zoomtoken_full_official_do_seed3407_v001.py` 继承 `e2e_thumos_videomae_s_768x1_160_adapter.py`；DN snapshot 继承 `georoute_adatad_development_base.py`，Q 继承 `georoute_dynamic_scnr_stage1_base.py`，U/R 继承 Q；都绑定 canonical annotation/video/pretrain path。DN/Q（及其 U/R inheritance）显式 `checkpoint_interval=60`。 | PARTIAL / historical matched-source provenance | 该历史矩阵不能以 60-epoch recovery cadence 作为未来完整实验包；也不是 exact released-checkpoint AdaTAD reproduction。 | 未来任何 full run 先由 Builder 在 MCL 中把非 untouched-official arm 改为每 5 epoch、保留至少 3 resume checkpoints；DO 若保持 untouched official recipe 则可保留 2 epoch。 |
| local raw source | `E:/下载/THUMOS14_video/OpenDataLab___THUMOS14_video/raw/Test Data/TH14_test_set_mp4` 存在，1574 顶层文件、106,266,888,470 bytes；未逐 UID/许可/版本验证，且不作为当前远端训练绑定。当前 repo 与 official checkout 内 `data/thumos-14`/`pretrained` 不存在。 | UNVERIFIED（local candidate） | 任何试图把该本地目录直接当 official train/val source 的实验。 | 仅在合法许可确认后以 annotation UID 对齐；不应复制或自动使用。 |
| runtime/carrier | remote SIF `/data/run01/sczc063/yuzibo/containers/zoomtoken_p1_ubi9_20260815.sif` 37,453,824 bytes；lock `.../env-locks/zoomtoken_p1_0a2e004e.txt` 20,101 bytes；filesystem 可用约 3.1T。 | PARTIAL | future full official run runtime identity/容量 gate。 | 准备 exact release runtime receipt（driver, SIF digest, Python/Torch/CUDA, storage budget）；无下载，但需要 Slurm allocation later。 |
| license/access receipt | official README 指向 THUMOS official site/Google Drive；本项目未找到 data-license or checkpoint-access receipt。 | UNVERIFIED | 对外宣称可再分发数据/weights，或实际下载 released checkpoint。 | 获取/登记数据与 checkpoint 的合法访问证据；如需要受限账号，需用户/持证者凭据，不能由本审计假定。 |

### 非 ZoomToken 当前路线的视频资产图（只作可用性边界）

| 数据集 | remote inventory | 可直接用于当前 ZoomToken？ |
| --- | --- | --- |
| MultiSports | 四个 sport `.tar` archive 合计 43,820,810,240 B（2,129 videos）；只有 18 selected MP4 解压，annotation/proposal archive 存在。 | 否。完整树未解压，且不是 THUMOS/AdaTAD contract。 |
| TOC-Bench | `tstep_v0_phase0/datasets/toc_bench_full/videos` 有 1,951 MP4、15,235,916,868 B；另有 75 个小逻辑/诊断视频。 | 否。需独立路线、split 与 evaluator。 |
| Charades | `datasets/charades/raw_data/Charades_v1_480` 有 9,848 MP4、16,588,858,990 B，原始 zip 同时存在。 | 否。需独立路线与协议。 |
| ActivityNet | 只有 3 个可直接读取 MP4（约 71.7MB）；其余 split/missing-file archive 未组装成可验证 tree。 | 否，PARTIAL / unusable tree。 |
| FineAction / HACS / EPIC-Kitchens / Ego4D | 无实际视频数据集；只有脚本、README 或小源 archive，FineAction probe 为空。 | 否，MISSING。 |
| EventMATR | 不重复保存 raw video；仅有 native MATR feature pickles，checkpoint 仍不完整。 | 否；不得将特征当作 raw-video 或其他 feature identity。 |

这些资源不改变 ZoomToken 的 THUMOS14 baseline-first 优先级。任何将来引入它们的 config、
manifest 或 symlink 必须由 Builder 在独立科学协议下实施；本审计没有创建链接、解压 archive
或改动配置。

### 资源可否立即启动下一项完整官方实验？

**否。** 对 clean official *training* 而言，远端 canonical raw video、annotation/class map 和
VideoMAE pretrain 在技术上齐备；但用户指定的下一优先级是先做 published AdaTAD released-
checkpoint anchor evaluation，而该 checkpoint 的可验证位置/访问回执尚缺。此外，必须先用
clean `01c58b9` checkout 完成 exact config/evaluator/EMA/checkpoint selection 与运行时绑定，
不能复用 matched-source full matrix 的 launcher 或 66.xx 数字。因而当前为
`NEEDS_ATTENTION — BASELINE_CHECKPOINT_AND_PROVENANCE`，不是数据下载授权请求。

## 6. 后续唯一优先动作

1. 合法取得或定位 released checkpoint `1HGUBroK90KBAkFqQreAVtHCIclJh7DmM` 的可验证副本；
   用 clean `01c58b9` 的 unmodified config/evaluator 在 THUMOS14 validation 上先检验
   `Avg=69.03, mAP@0.7=48.27`。不通过时先做 exact official reproduction。
2. 对比该 exact path 与 matched-source dense 的 data preprocessing、config inheritance、
   pretrain、optimizer/schedule、EMA/final selection、evaluator/NMS、runtime；未经此 diff
   不解释 69.03 与 66.xx 的 gap，更不归因于 seed 或 routing。
3. 只有 baseline identity 已锁定后，才可重新审查 Q、U、R 与 ROI/residual 的比较语义；
   当前 P1 失败矩阵和 matched-source Q 数字不升级为论文结论。

## 7. 对未来完整官方训练的 checkpoint 政策（用户更正）

- 项目默认保存**每 5 epoch**一个可恢复 `.pth`；若 untouched official recipe 更频繁，保留
  官方频率。故 exact AdaTAD `01c58b9` reproduction 维持其既有 **每 2 epoch**。
- checkpoint 只用于中断恢复/诊断；模型选择固定为预注册的 final/final-EMA，禁止看 validation
  后挑最佳中间 checkpoint。
- 保留至少最近 3 个有效 recovery checkpoints，以及预定义 milestone/final；不删除唯一 resume
  点。每次提交前须在 config/launch packet/Wiki 记录 interval、保留规则、输出目录、预估空间和
  resume command。
- PRE_RUN 必须验证 resume 恢复 model、optimizer、scheduler、scaler、epoch/update 计数及框架
  所需 RNG 状态。若现有 full-run config 没有该能力，Builder 只能做最小项目内配置更正后再提交。

本审计没有改代码、下载数据、创建 Slurm 任务或读取新实验指标。
