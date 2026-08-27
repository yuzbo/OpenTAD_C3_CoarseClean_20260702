---
doc_id: ADATAD_SHARED_OFFICIAL_BASELINE_PACKET
version: v001
date: 2026-08-17
owner_project: ZoomToken
owner_role: Coordinator
status: PREPARED_BLOCKED_RELEASED_CHECKPOINT
execution: NOT_STARTED
consumer_policy: READ_ONLY_FINAL_RECEIPT
---

# AdaTAD 共享官方基线：唯一 evaluation / reproduction packet

## 1. 范围、唯一性与证据边界

ZoomToken 是所有相关 TAD 项目的唯一执行负责人。该 packet 只允许两条互斥的、串行的
official baseline 路径：先做**一次** released-checkpoint evaluation；仅在 released artifact
确实无法合法取得、且该 evaluation 已被明确记录为 blocked 后，由同一负责人做**一次** clean
untouched official reproduction。其他项目只能只读引用最终 durable receipt，不能复制 checkpoint
evaluation、训练或结果根。

`66.42/67.14/65.99` 是 matched-source dense 的 THUMOS14 validation 输出，不是本 packet 的
official result；不得以其替代 published anchor `Avg-mAP=69.03, mAP@0.7=48.27`，也不得用它
推出 ZoomToken 质量结论。

## 2. 固定输入（提交前逐项读回并写入 receipt）

| 项 | 固定身份 | 当前状态 |
| --- | --- | --- |
| exact AdaTAD release | `E:/DeskTop/TAD/OpenTAD_OFFICIAL_ADATAD_01c58b9` @ `01c58b9f2370e914150cf94d392208a4e211c053` | 本地 read-only clean ref |
| independent upstream diff ref | `E:/DeskTop/TAD/OpenTAD_OFFICIAL_BASELINE_AUDIT_20260817` @ `346d09d19e2091372cec48172dbe40f7b28bdee6` | 只用于 exact-path diff，不能替代 release execution |
| untouched official config | `configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py` | 原始 release config |
| canonical THUMOS14 | `/data/run01/sczc063/yuzibo/thumos14/raw_data/video`，411 valid MP4 symlinks（training 200，validation 211，0 broken） | shared read-only；不得复制 |
| annotation / categories | `/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json`；`.../category_idx.txt` | available |
| VideoMAE-S pretrain | `/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth` | available，供仅有的一次 training fallback 使用 |
| released checkpoint | official Drive ID `1HGUBroK90KBAkFqQreAVtHCIclJh7DmM` | `UNVERIFIED / MISSING_KNOWN_PATHS`；先合法定位/验证，不得下载猜测副本 |
| evaluator / NMS | exact release evaluator；validation split；soft-NMS `sigma=0.7`、`max_seg_num=2000`、`multiclass=True`、`voting_thresh=0.7` | 冻结，须以 resolved config 回执复核 |
| seed | official `tools/test.py` / `tools/train.py` default `42` | 固定为 `42`；receipt 写入实际 CLI 值，不得暗中沿用 ZoomToken `3407` |
| model selection | released checkpoint 原样评测；training fallback 按 official final / final-EMA pre-registered rule | 禁止 validation 后挑 best intermediate |
| runtime / output root | single immutable container + dependency lock + Slurm allocation；共享结果根提交前指定且必须为空 | `UNBOUND_PRE_RUN`；本 packet 不创建目录或作业 |

Official config 的训练 fallback 保留其更频繁的 `checkpoint_interval=2`（高于本项目 5-epoch
默认）；每个 resume point 仍须可恢复 model、optimizer、scheduler、scaler、epoch/update 与 RNG。

## 3. 唯一执行序列

1. **Checkpoint evaluation（优先且最多一次）**：在 clean `01c58b9` checkout，用上表原始 config、
   canonical validation、exact evaluator/NMS 和可验证 release checkpoint 做一次评测；目标仅是
   检验 published anchor `69.03 / 48.27`，不是打开 official held-out test。
2. **Fail-closed 判定**：若 checkpoint identity、license/access、runtime、data binding 或 command
   任一缺失，记录 `BLOCKED_RELEASED_CHECKPOINT`，不以 matched-source 结果补位。
3. **Training fallback（至多一次）**：仅在 checkpoint 被确认不可得且 shared owner 作出
   `REPRODUCTION_REQUIRED` receipt 后，运行一次 untouched official recipe。它与 evaluation 共用
   同一个 shared result root family；任何相关项目不得平行或重跑。
4. **Final receipt**：唯一 durable receipt 须记录 full SHA/ref/clean state、config digest/resolved
   config、checkpoint/pretrain identity、seed=42、canonical 411/annotation/category map、evaluator/NMS、
   final/final-EMA rule、container/driver/Python/Torch/CUDA/lock、Slurm job、output root、raw result 和
   published-anchor comparison。无这一整组绑定，数字只能标记 `UNBOUND_SHARED_INPUT`。

## 4. 与 ZoomToken 方法并行的轨道 B

共享 baseline 的数值未绑定时，ZoomToken 不停工，但只做不产生效能 claim 的准备：

- 以独立 Builder→Critic→Evaluator 链修正/审查 Q 动态空间路由的正式 `DO/DN/U/R/Q` 矩阵入口；
- 保持 Q 的 global exact `B=24576`、dynamic `K_t`、ragged、masked-zero 与 no-leak 边界；
- 将 ROI/残差限定为 conditional `G/N/F` 因果对照，不能借共享 baseline 缺口改写成论文主线；
- 对非 untouched-official full run，配置每 5 epoch 可恢复 `.pth`、至少最近 3 个 recovery
  checkpoints，加上 milestone/final；验证完整 resume state；
- 准备 data/runtime/empty-root/launcher/PRE_RUN receipt。任何新的 remote matrix 仍需独立明确授权，
  不因本 packet 自动启动。

## 5. 当前缺口、角色交接与下一动作

| 轨道 | current_scientific_question | next owner / action | dependency | expected return | single recovery |
| --- | --- | --- | --- | --- | --- |
| A｜共享基线 | 能否在 exact release path 上复核 published `69.03/48.27`？ | Coordinator：合法定位 release checkpoint 后，准备一次 exact evaluation PRE_RUN receipt | checkpoint identity + access/license receipt + immutable runtime + empty shared result root | checkpoint location / PRE_RUN event | checkpoint 仍缺则记录唯一 `BLOCKED_RELEASED_CHECKPOINT`，不训练或重复试跑 |
| B｜方法准备 | Q matrix 是否可在未绑定 dense 数字下保持可审计并可恢复？ | Builder：仅最小入口/恢复实现；Critic：独立审查；Evaluator：result-blind PRE_RUN | accepted Q-core contracts；shared dense remains unbound | each durable role receipt | deterministic defect 仅一次 focused correction；第二次同类缺陷停止该面 |

**角色通知内容**：Builder 只处理 Track B 的最小配置/launcher/recovery change，不能复制 official
baseline；Critic 只审 Track B 的公平性、入口与恢复边界；Evaluator 只准备 Track B 的 result-blind
PRE_RUN，同时把 shared dense 视为待绑定输入。没有一项角色任务由本 packet 授权启动 data、GPU、
Slurm 或 remote execution。

## 6. 本次材料状态

本 packet 是设计/治理材料，不是 evaluation、training、PRE_RUN 通过或性能证据。未创建、下载、
复制 checkpoint；未启动远端作业；未改变任何 Q、ROI 或 residual 的科学结论。
