---
type: experiment
status: tested
updated: 2026-08-31
project: DUCA
---

# DUCA THUMOS14 完整数据身份核验

## 目的与边界

本核验只回答正式比较所使用的完整训练视频与完整留出评估视频是否具有一致、可追溯的字面身份。它不读取留出集动作类别或时间边界，不加载模型或检查点，不生成预测，不计算 mAP，也不申请 GPU。

科学任务来自 Pro 裁决 `DUCA-COMPREHENSIVE-ROUTE-INTEGRATION-v001-20260831`。当前结果必须先返回 Pro 作数据准入裁决；`PASS` 不是模型实现或训练授权。

## 代码与独立审查

- 分支：`feature/duca-full-data-identity-audit-v1-20260831`
- 提交：`fdd2bcdddf3f23f3546244adf90c4427ed022837`
- 父提交：`04c35a3b76897e6c1569eeede41ed3aecaf7f854`
- 修改文件仅为 `tools/bata/audit_duca_thumos14_split_identity.py` 与 `tests/test_audit_duca_thumos14_split_identity.py`。
- 本地聚焦与既有回归测试：`29 passed`；N16R4 聚焦测试：`6 passed`。
- 独立 Critic 只读核验 exact commit、parent、两文件差异、身份层访问边界和测试后返回 `PASS`。

GitHub：

- <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdd2bcdddf3f23f3546244adf90c4427ed022837>
- <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/fdd2bcdddf3f23f3546244adf90c4427ed022837/tools/bata/audit_duca_thumos14_split_identity.py>
- <https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/fdd2bcdddf3f23f3546244adf90c4427ed022837/tests/test_audit_duca_thumos14_split_identity.py>

## N16R4 CPU Evaluator

远端 clean root：

```text
/data/run01/sczc063/yuzibo/duca_full_data_identity_audit_fdd2bcdd_20260831
```

有效命令仅把 ActionFormer 原始 subset 字面值绑定为区分大小写的 `Test`：

```bash
python -m tools.bata.audit_duca_thumos14_split_identity \
  --repo-root /data/run01/sczc063/yuzibo/duca_full_data_identity_audit_fdd2bcdd_20260831/repo \
  --annotation /data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json \
  --class-map /data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt \
  --media-root /data/run01/sczc063/yuzibo/thumos14/raw_data/video \
  --historical-211 /data/run01/sczc063/yuzibo/duca_pjst_d1_c73e8418_20260826/off/terminal_eval_v3/work/gpu1_id0/result_detection.json \
  --actionformer-annotation /data/run01/sczc063/yuzibo/duca_full_data_identity_audit_fdd2bcdd_20260831/sources/thumos/annotations/thumos14.json \
  --actionformer-subset Test \
  --exclusion-source /data/run01/sczc063/yuzibo/duca_full_data_identity_audit_fdd2bcdd_20260831/repo/tools/prepare_data/thumos/README.md \
  --output-dir /data/run01/sczc063/yuzibo/duca_full_data_identity_audit_fdd2bcdd_20260831/result_v2 \
  --ffprobe /data/run01/sczc063/yuzibo/conda_envs/opentad/bin/ffprobe \
  --ffprobe-timeout 30
```

首次命令误把该原始字面值写成小写 `test`，因此 ActionFormer 集合为空并得到一个调用层 `BLOCK`。代码、数据与来源均未改变；保留首次输出后，仅修正这一参数并写入独立 `result_v2`。该操作性修正不得写成科学重试。

## 完整训练集身份

| 身份层 | 数量 | manifest SHA-256 |
|---|---:|---|
| annotation `training` | 200 | `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0` |
| formal loader replay | 200 | `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0` |
| canonical physical media | 200 | `5b11e290eb24c93c79f23cb1aecc8b85be4c13b47d7cf3b35e30601c1663f4c0` |

三套字面 ID 完全相同；训练侧排除为空，训练与留出交集为空。

## 完整留出集身份与 211/212 解释

| 身份层 | 数量 | manifest SHA-256 |
|---|---:|---|
| OpenTAD annotation `validation` | 211 | `5f9adf639fbcff869075ac78f6aa26d9da14986199a7d5b2181127769600746e` |
| OpenTAD loader | 211 | 同上 |
| canonical physical media | 211 | 同上 |
| OpenTAD evaluator | 211 | 同上 |
| 历史 PJST 预测 ID | 211 | 同上 |
| ActionFormer annotation `Test` | 212 | `a1507abc217baed5eb0f341391ab7e73ae44edbf06c08c08f681b983e11db2af` |

OpenTAD 的 annotation、loader、physical、evaluator 与历史 211 预测集合逐字相同。ActionFormer 212 相对 OpenTAD 211 唯一多出：

```text
video_test_0000270
```

OpenTAD 源文件 `tools/prepare_data/thumos/README.md:11` 明确说明：`video_test_0000270` 因错误标注、`video_test_0001292` 因空标注从其 THUMOS14 test 集删除，所以 OpenTAD 使用 211 个视频。本次字面 annotation 对比只出现 `video_test_0000270`，因为 `video_test_0001292` 不在 ActionFormer 的 212 条 annotation 中；它属于额外物理/特征文件，不是评估视频。

411 个期望视频全部通过基本 `ffprobe` 解码；无缺文件、坏链接、重复 ID、未分配 canonical ID 或解码失败。集合差分文件除上述一条 ActionFormer 右侧 ID 外为空。

## 配置、代码来源与哈希

- Stage-1：`configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py`
- Stage-2：`configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py`
- 两条继承链和逐文件 SHA 见完整报告；解析后 train=`training`、held-out/evaluator=`validation`。
- annotation SHA-256：`ee526d55aa4315a8adc68c501d0331f96a56ce16fa960f1d2ea182b9381ab9ad`
- class map SHA-256：`a158b7c4c130ce74375a9b114160e2faae7a0221e605a0464a556fe082644f31`
- evaluator `opentad/evaluations/mAP.py` SHA-256：`e855e70d41d087d039a90ecdb8f3cc3efece209130417320edf35062b8503fd4`；ground-truth import 位于第 46 行，subset filter 位于第 72 行。
- 完整报告 SHA-256：`d7251c11935644cf8661e6bfdcfb857e29d2357cb894b7de9d8b2bd7eaf6f1ab`
- 完整报告与 literal manifests：`research-wiki/sources/2026-08-31-duca-full-data-identity-audit-fdd2bcdd/`。

## 隔离证明与结论

报告记录：held-out annotation values 未解码，held-out label/segment access 为 `false`，checkpoint/model/GPU/mAP 均未使用，prediction payload 未读取。历史 prediction 只读取顶层视频 ID 键。

唯一执行结论：

```text
DATA_IDENTITY_PASS_211
```

下一动作是把本页、完整报告、literal manifests、Critic `PASS` 与首次参数大小写修正事实中立返回 Pro。Pro 数据准入前继续禁止多预算模型代码、checkpoint、PRE_RUN、GPU、训练、held-out 预测与 mAP。

两份最新 Pro 报告对数据任务一致，但对数据准入后的种子顺序存在待裁决差异：较早完整 Wiki 复核写为直接执行 3407/3408/3409；新综合裁决写为先执行 3407，只有全部门通过才复制 3408/3409。Codex 不选择该后续分歧。
