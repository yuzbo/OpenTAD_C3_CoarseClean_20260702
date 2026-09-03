#!/usr/bin/env python3
"""Refresh the human-readable catalog for every DUCA/ZoomToken experiment."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "docs" / "audits" / "DUCA_MULTIBRANCH_20260902"
DEFAULT_JSON = AUDIT / "11_EXPERIMENT_CATALOG.json"
DEFAULT_MD = AUDIT / "11_EXPERIMENT_CATALOG.md"
REMOTE_ROOT = "/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902"
SSH_ARGS = [
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=15",
    "-o", "IdentitiesOnly=yes",
    "-o", "PubkeyAcceptedAlgorithms=+ssh-rsa",
    "-o", "HostkeyAlgorithms=+ssh-rsa",
    "-i", "C:/Users/skywalker/.ssh/id_rsa",
    "-p", "22",
    "-l", "sczc063@BSCC-N16R4",
    "ssh.cn-zhongwei-1.paracloud.com",
]


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def git_head(path: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    except OSError:
        return None
    return proc.stdout.strip() if proc.returncode == 0 else None


def clean_tree(path: str) -> bool | None:
    try:
        proc = subprocess.run(["git", "-C", path, "status", "--porcelain"], text=True, capture_output=True, check=False)
    except OSError:
        return None
    return proc.returncode == 0 and not proc.stdout.strip()


def remote_receipt() -> dict[str, Any]:
    if os.environ.get("DUCA_CATALOG_DISABLE_REMOTE") == "1":
        return {"status": "NOT_QUERIED", "reason": "DUCA_CATALOG_DISABLE_REMOTE=1"}
    try:
        proc = subprocess.run(
            ["ssh", *SSH_ARGS, "cat", f"{REMOTE_ROOT}/latest_receipt.json"],
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        return {"status": "UNAVAILABLE", "reason": f"ssh unavailable: {exc}"}
    if proc.returncode != 0:
        return {"status": "UNAVAILABLE", "reason": (proc.stderr or proc.stdout).strip()[-500:]}
    try:
        receipt = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "INVALID_RECEIPT", "reason": proc.stdout[-500:]}
    return {
        "status": "ACTIVE",
        "checked_at": receipt.get("checked_at"),
        "dispatcher_status": receipt.get("dispatcher", {}).get("status"),
        "entries": receipt.get("entries", []),
    }


def route_entries() -> list[dict[str, Any]]:
    return [
        {
            "category": "frozen_route",
            "name": "H65-Pro 严格 60 轮全矩阵：物理时间坐标与高质量动作定位",
            "internal_id": "H65_PRO",
            "branch": "codex/h65-pro-fullmatrix-strict60-20260902",
            "sha": "cfb7041d876f6e38e9ef6ce77cef7cee04b79659",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/cfb7041d876f6e38e9ef6ce77cef7cee04b79659",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/h65_pro",
            "deployment_status": "已完成精确 SHA CUDA focused admission；P0 admission 失败，正式矩阵未提交",
            "result_status": "无最终结果",
            "final_result": "15 个 focused CUDA 测试通过；更深 P0 检查 14 通过、1 失败，暴露 x-only backbone 收到 masks 的签名错误",
            "next_action": "在独立修正 SHA 完成签名路由复验，再重新冻结 H65 SHA",
        },
        {
            "category": "frozen_route",
            "name": "DUCA 统一全矩阵：Taylor 归因、H65 保留机制与真实成本",
            "internal_id": "DUCA_UNIFIED",
            "branch": "codex/duca-unified-fullmatrix-20260902",
            "sha": "89b9ea3e8e018b41034917ee14de7f409354a7e9",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/89b9ea3e8e018b41034917ee14de7f409354a7e9",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/duca_unified",
            "deployment_status": "生成器 fail-closed；Taylor P0/P1、原始 H65 retention/transition、真实 cost 未实现，未提交训练",
            "result_status": "无最终结果",
            "final_result": "无合法 mAP、速度或成本结果；41 个 cell 保持关闭",
            "next_action": "完成三个真实机制后重新运行 generator、preflight 和 exact-head admission",
        },
        {
            "category": "frozen_route",
            "name": "DUCA 证据恢复：历史 H65 证据链与 8261 单种子数值复现",
            "internal_id": "EVIDENCE",
            "branch": "codex/duca-evidence-recovery-numerical-correction-20260902",
            "sha": "08d425a259fc468dde7c496e77b4c43e953d8d0c",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/08d425a259fc468dde7c496e77b4c43e953d8d0c",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/evidence",
            "deployment_status": "精确 SHA CUDA focused admission 和 seed 8261 precheck 已通过；C0 parity 尚未完成，正式训练未提交",
            "result_status": "无最终结果",
            "final_result": "35 个 focused CUDA/证据测试通过；尚无 terminal EMA、官方评测或 mAP",
            "next_action": "完成 indices、physical positions、features、logits、loss、decode、predictions 的 C0 精确 parity",
        },
        {
            "category": "frozen_route",
            "name": "DUCA CT-DP-BAMoD：CT-Tubelet 物理时间差归一化与 B-AMoD 稀疏层路由",
            "internal_id": "CT_DP_BAMOD",
            "branch": "codex/duca-ctdp-geometry-mechanism-correction-20260902",
            "sha": "2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/2b7f81808006c6cb09a4d21a7f6fdc8ed3f6babc",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/ct_dp_bamod",
            "deployment_status": "精确 SHA geometry focused admission 已通过；冻结 SHA 的 G0/G1 因子化与声明冲突，正式矩阵未提交",
            "result_status": "无最终结果",
            "final_result": "7 个 focused CUDA/几何测试通过；不能据此宣称 CT-DP 机制有效",
            "next_action": "采用独立修正分支完成 geometry、有限差分 gradient、batch/DDP 后重新冻结 SHA",
        },
        {
            "category": "frozen_route",
            "name": "ZoomToken BAFDR：48 分块全局低清、K16 局部高清路由与 D160 教师蒸馏",
            "internal_id": "BAFDR",
            "branch": "codex/zoomtoken-bafdr-gradient-correction-20260902",
            "sha": "fdeaeb98340bf7070201a02feb8093f50486aeaa",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/fdeaeb98340bf7070201a02feb8093f50486aeaa",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/bafdr",
            "deployment_status": "静态协议 admission 已通过；精确 SHA 五臂 screen 尚未通过，21-cell 矩阵关闭",
            "result_status": "无最终结果",
            "final_result": "11 个静态协议测试通过；缺少同种子 D160 epoch 59 EMA Teacher 和 selection-screen PASS",
            "next_action": "提供并核验 terminal Teacher，再运行不依赖 held-out 的五臂 screen",
        },
        {
            "category": "frozen_route",
            "name": "ZoomToken ET-TRC：Transformer 内部 Anchor 全计算与非 Anchor 局部 Taylor/JVP 修正",
            "internal_id": "ET_TRC",
            "branch": "codex/zoomtoken-et-trc-correction-20260902",
            "sha": "59eab0c6aaacf5039d2ae20969a6dd5772bcb80f",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/59eab0c6aaacf5039d2ae20969a6dd5772bcb80f",
            "local_directory": "E:/DeskTop/TAD/_duca_audit_worktrees/et_trc",
            "deployment_status": "静态 launcher/pretrain 协议测试已通过；真实 checkpoint coverage、单卡加载和双 GPU DDP 尚未完成",
            "result_status": "无最终结果",
            "final_result": "10 个协议测试通过；无合法 OFF/ON terminal EMA 或评测结果",
            "next_action": "核验 VideoMAE checkpoint 覆盖，再执行真实 global-batch=2 双 GPU OFF/ON DDP 和 resume",
        },
        {
            "category": "correction_route",
            "name": "H65-Pro 当前正式单种子矩阵：384 帧四相预算与物理时间定位",
            "internal_id": "H65_PRO_ACTIVE",
            "branch": "codex/h65-pro-admission-fix-20260902",
            "sha": "e553a5a4a1063a755900d3dfa4bf8909bf97d466",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/e553a5a4a1063a755900d3dfa4bf8909bf97d466",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/h65_eval_repair"],
            "deployment_status": "远端 exact-SHA admission 1267684 已通过；REF-D768、REF-U384、REF-MNV3FC384 正式训练 1267709/1267711/1267737 均 COMPLETED(0:0)，官方评测 1269303/1269360/1269373 均 COMPLETED(0:0) 并写出独立 metrics receipt。旧 F01 1269287、F02 1269291、F03 1269341 在课程切换附近因相同 p_action calibration 异常失败；确认共享缺陷后已停止旧 SHA 的 F04-F06 及依赖评测，日志和 checkpoint 保留。修复 codex/h65-pro-paction-tolerance-repair-20260904@30514803 已推送并通过远端 23 tests；F01-F06 PRECHECK 1269372/1269376/1269378/1269380/1269384/1269386 全部 PASS，正式重训 1269375/1269377/1269379/1269381/1269385/1269387 全部 RUNNING。2026-09-04 06:34 CST 进度分别约为 epoch 16/16/15/15/12/11，错误日志为空",
            "result_status": "D768、U384、MNV 三个参考臂均有有效官方终态性能；F01-F06 在修复 SHA 上运行中，尚无最终性能",
            "final_result": "D768 Avg-mAP=67.58%，mAP@0.3/0.4/0.5/0.6/0.7=82.56/78.12/69.98/60.37/46.88%。U384 Avg-mAP=63.89%，对应 79.37/74.83/67.15/55.94/42.18%，比 D768 低 3.69 个 Avg-mAP 百分点。MNV3FC384 Avg-mAP=57.01%，对应 73.80/68.49/60.54/48.06/34.14%，比 D768 低 10.57 个 Avg-mAP 百分点；MNV receipt 绑定 evaluator 67c8f39f、训练 e553a5a4、seed 3407、epoch 59 EMA、211 videos/422000 predictions。F 臂尚未跨越旧故障集中出现的 epoch 20-23 区间",
            "next_action": "持续监控修复版 F01-F06，全部跨越原故障区间并完成 terminal EMA 后，逐臂运行独立官方评测并生成身份回执",
        },
        {
            "category": "correction_route",
            "name": "CT-DP 当前正式四臂：基础嵌入、CT-Tubelet、B-AMoD 及二者组合",
            "internal_id": "CT_DP_BAMOD_ACTIVE",
            "branch": "codex/duca-ctdp-formal-repair-20260903",
            "sha": "c0fae67a1236f2c47e6c2935d217659cd1f8fb9d",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/c0fae67a1236f2c47e6c2935d217659cd1f8fb9d",
            "local_directory": "E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903",
            "deployment_status": "远端 admission 1267221 已通过；G0/G1/G2/G3 正式训练 1267229/1267230/1267231/1267232 均 COMPLETED(0:0)，四个 epoch_59.pth 均存在",
            "result_status": "四臂 60 轮训练全部完成；只有训练内 terminal validation telemetry，尚无独立终态 evaluator receipt",
            "final_result": "训练日志末次 Avg-mAP：G0=14.71%、G1=14.84%、G2=56.15%、G3=57.84%；mAP@0.7 分别为 2.84/3.26/26.24/31.53%。这些数值显示 G2/G3 明显优于 G0/G1，但在独立 evaluator 与身份回执生成前仍只作为暂定终轮 telemetry",
            "next_action": "对四个 epoch 59 EMA checkpoint 独立运行 tools/test.py，保存 result_detection.json、evaluation_metrics.json 与身份回执后再裁决 CT-Tubelet 和 B-AMoD 的独立贡献",
        },
        {
            "category": "correction_route",
            "name": "DUCA-Unified 当前 41 单元正交消融控制台",
            "internal_id": "DUCA_UNIFIED_ACTIVE",
            "branch": "codex/duca-unified-formal-gates-20260903",
            "sha": "793c4f9cdf7dac4f224bc73012aff8bc93949f87",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/793c4f9cdf7dac4f224bc73012aff8bc93949f87",
            "local_directory": "E:/DeskTop/TAD/OpenTAD_DUCA_Unified_FormalGates_20260903",
            "deployment_status": "提交器 fail-closed，Taylor P0/P1、H65 retention/transition 与真实 cost 未落地前禁止提交相关单元",
            "result_status": "BLOCKED_UNIMPLEMENTED，无正式性能",
            "final_result": "manifest/生成器/准入规则可验证，但不能把缺失机制的占位配置当实验",
            "next_action": "逐项实现并测试缺失机制后重新生成 41-cell manifest，再分阶段释放正式矩阵",
        },
        {
            "category": "correction_route",
            "name": "BAFDR 当前 seed 4407 正式流水线：D160 教师与五臂 K16 筛选",
            "internal_id": "BAFDR_ACTIVE",
            "branch": "codex/zoomtoken-bafdr-admission-fix-20260903（训练） + codex/zoomtoken-bafdr-eval-metadata-repair-20260904（评测）",
            "sha": "539287fa8a035765afd7e79863ce77278bef83f2（训练） / 29b5a7a2b291203ea7b697cfe416b64f0d365d02（评测）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/29b5a7a2b291203ea7b697cfe416b64f0d365d02",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_admission",
            "deployment_status": "真实 CUDA/focused 门禁 1267855 已通过；D160 教师 1267884、G96 1268698、U16 1269124、LATE 1269129、NOKD 1269137、FULL 重试 1269297 均 COMPLETED(0:0)，全部写出 6000-update epoch-59 receipt；screen_receipt_r2_1269297.json 精确绑定成功 FULL 并为 PASS。评测 1269382 因 PROJECT_DIR 解析错误失败；1269383 因 U16 validation 请求不存在的训练标签失败；02d34e6b 修正标签后，1269389 完成 G96 prediction receipt，但 U16 后处理又因自定义 meta_keys 漏掉 snippet_stride 失败。独立修复 codex/zoomtoken-bafdr-eval-metadata-repair-20260904@29b5a7a2 补齐标准时序后处理元数据并强化 validator；本地和远端静态测试均 11 passed、21-cell validator PASS、双 GPU U16 PRECHECK 1269540 COMPLETED(0:0)，五臂正式评测 1269541 已自动启动",
            "result_status": "D160 与五个 screen 臂训练终态有效，screen gate PASS；seed 4407 五臂官方评测在修复 SHA 上运行中，尚无开放性能",
            "final_result": "FULL receipt 绑定训练 539287fa、seed 4407、world_size 2、6000 successful updates、D160 teacher 和 checkpoint SHA256=c0eb8677…；评测器固定为 29b5a7a2，screen receipt 仍严格验证训练身份 539287fa。1269389 已留下 G96 prediction receipt，但未完成全部五臂且未开放指标，因此不得报告 mAP；1269541 必须完成 prediction 与 metric-opening receipts 才构成最终结果",
            "next_action": "监控 1269541 依次完成 G96、U16、LATE、NOKD、FULL 的 prediction-only 与 official metric opening；若再次失败，先读当前臂日志，再按独立修复 SHA 和新作业命名空间处理",
        },
        {
            "category": "correction_route",
            "name": "ET-TRC 当前双臂：完整 Transformer 与局部 Taylor/JVP 近似",
            "internal_id": "ET_TRC_ACTIVE",
            "branch": "codex/zoomtoken-et-trc-formal-repair-20260903",
            "sha": "74473c2775caebf0da9d368ce8009d78e2942098",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/74473c2775caebf0da9d368ce8009d78e2942098",
            "local_directory": "E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902",
            "deployment_status": "真实 VideoMAE 覆盖与 2-GPU/global-batch=2 admission 已通过；OFF 1267218 于 01:04、ON 1267219 于 01:10 均 COMPLETED(0:0)，两者均写出 epoch_59.pth",
            "result_status": "双臂 60 轮训练已完成；日志含终轮 validation 指标，但当前路线没有独立终态 evaluator/receipt，因此仍无可裁决最终性能",
            "final_result": "终轮日志 OFF Avg-mAP 62.08%、mAP@0.3/0.4/0.5/0.6/0.7=77.06/71.91/65.34/54.51/41.55%；ON 为 54.81%、70.38/64.89/57.49/46.65/34.63%。这些数值是完整 validation telemetry，不是具备独立终态 receipt 的最终 scientific result",
            "next_action": "验证 epoch_59 EMA/6000 updates，并分别用 OFF/ON config 独立运行 tools/test.py；保存 evaluator 日志、result_detection.json 和 metrics artifact 后再裁决 Taylor/JVP 机制",
        },
        {
            "category": "historical_exact_route",
            "name": "BAFDR 历史五臂三种子终态 checkpoint 独立评测",
            "internal_id": "BAFDR_EFE69D2E_EVAL",
            "branch": "历史远端 exact-SHA 结果身份",
            "sha": "efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "local_directory": "E:/DeskTop/TAD/NO_LOCAL_WORKTREE_FOR_EFE69D2E",
            "deployment_status": "1266410-1266414 已产生 15 个 epoch-59 checkpoint；独立任务的 prediction eval 1267920 于 2026-09-04 05:47 CST FAILED(1:0)，postprocess 1267921 为 DependencyNeverSatisfied。该链由其他任务所有，本监督任务仅记录，不取消、不修改、不重提",
            "result_status": "历史 exact-SHA 评测失败，尚无可报告终态指标",
            "final_result": "只按 efe69d2e 的真实训练/teacher/数据/evaluator receipts 报告，绝不迁移为当前 539287fa 结果",
            "next_action": "由历史评测任务读取 1267920 日志并决定修复；当前监督任务保持只读边界",
        },
        {
            "category": "correction_route",
            "name": "Evidence-Recovery 当前 C0：轻量侦察器不确定性与最大空洞补漏",
            "internal_id": "EVIDENCE_ACTIVE",
            "branch": "codex/duca-evidence-fullgrid-repair-20260904",
            "sha": "246058f2c24edc78818ada60eec26249bbf7d5d2",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/246058f2c24edc78818ada60eec26249bbf7d5d2",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_eval_repair",
            "supporting_local_directories": ["E:/DeskTop/TAD/OpenTAD_Evidence_FormalRepair_20260903"],
            "deployment_status": "旧 77c8d173 C0 训练 1267979 虽完成 6000 updates，但旧 test ledger 不覆盖 792 次真实 loader 暴露，因此只保留为诊断。最终 codex/duca-evidence-fullgrid-repair-20260904@246058f2 使用 train/val 精确旧网格与 test full-grid，强制 allow_missing=False 并按唯一物理窗口校验；admission 1269357 COMPLETED(0:0)，验证 train 438/438、val 487/487、test 792 exposures/791 unique/791 ledger，34 tests 与 CUDA FP32 gate 均通过。新 C0 seed 8261 作业 1269374_0 于 2026-09-04 06:34 CST 运行至 epoch 14；一次 AMP skip 已按 20-replay 合同恢复，当前错误日志为空",
            "result_status": "最终 ledger 合同已通过准入；新 C0 seed 8261 正式重训运行中；无最终性能",
            "final_result": "test 的 792 次暴露对应 791 个唯一物理窗口，因为 video_test_0001431|7680 是 OpenTAD 尾窗逻辑产生的同一窗口重复暴露；246058f2 按唯一物理窗口严格覆盖，不是放宽或漏掉样本。新命名空间 /data/run01/sczc063/yuzibo/experiments/duca_evidence_normalized_246058f2 尚无 terminal EMA 或官方 evaluator receipt",
            "next_action": "持续监控 1269374_0 的 6000 successful updates 与 terminal EMA；完成后只在同一 ledger 身份下运行官方评测并生成 metrics receipt",
        },
    ]


def catalog() -> dict[str, Any]:
    entries = route_entries()
    for entry in entries:
        entry["local_head"] = git_head(entry["local_directory"])
        entry["local_clean_tree"] = clean_tree(entry["local_directory"])
    return {
        "schema_version": "DUCA-EXPERIMENT-CATALOG-v001",
        "last_updated_utc": utc_now(),
        "scope": "所有当前 DUCA/ZoomToken 代码实验及其独立修正路线；旧远端作业另列为不纳入结果",
        "repository": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702",
        "entries": entries,
        "remote_supervisor": remote_receipt(),
        "cluster_observation": {
            "checked_at_cst": "2026-09-04 06:32:04 CST",
            "public_gpu_available": 124,
            "public_gpu_total": 240,
            "user_jobs_in_queue": 11,
            "account_constraint": "AssocGrpGRES/AssocMaxSubmitJobLimit；公共 GPU 空闲不等于本账户仍有可用配额",
        },
        "result_policy": "没有 exact SHA、clean-tree、terminal EMA、官方 evaluator 和合法 aggregation receipt，不得报告为最终科学结果",
        "excluded_remote_jobs": [
            {"job_ids": "1266325-1266330", "remote_directory": "/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_6ae16954", "source_head": "6ae16954d875ce310cb0fc514ad54663be626db6", "reason": "旧 BAFDR checkout，不属于当前冻结 SHA；1266328-1266330 stderr 已诊断为 LoadFrames.__init__ 不接受 window_size，修复已移植到 BAFDR_ADMISSION_FIX，旧作业不重标结果"},
            {"job_ids": "1266185-1266186", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902_59eab0c6", "source_head": "be330c071638249e7c5268a5464e454c0f2a5621", "reason": "晚于冻结 ET-TRC SHA；1266185 在 S1 batch 17 出现 cls_loss/reg_loss/cost 非有限，1266186 随后取消，未产生合法 checkpoint"},
            {"job_ids": "1265704-1265705", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902", "source_head": "679b71214d05a21cd08ae1f5e5c3879e2df8fb83", "reason": "旧 CT-DP checkout；stderr 显示启动器引用不存在的 opentad_ct_dp_revised_20260902 路径，属于提交协议错误"},
            {"job_ids": "1266218-1266219", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_et_trc_correction_20260902_59eab0c6", "source_head": "be330c071638249e7c5268a5464e454c0f2a5621", "reason": "晚于冻结 ET-TRC SHA；Slurm COMPLETED 但仅有 log/config，没有 terminal checkpoint 或 receipt，不纳入当前结果"},
            {"job_ids": "1266401-1266420", "remote_directory": "/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_5dba75c7", "source_head": "efe69d2ea10accd01d0129dfe99cba4d1d5773cb", "reason": "不是当前 539287fa BAFDR 身份；1266401/1266402 的 LoadFrames 失败保留，1266410-1266414 的 15 个终态 checkpoint 由独立任务按 efe69d2e 身份评测，满足 receipt 后可单列历史结果，但不得迁移到当前提交"},
            {"job_ids": "1266475,1266479,1266480,1267819,1267820,1267822", "remote_directory": "/data/run01/sczc063/yuzibo/projects/bafdr_k16_fullmatrix_5dba75c7", "source_head": "efe69d2ea10accd01d0129dfe99cba4d1d5773cb", "reason": "efe69d2e 的旧评测/cexec/summary 链；与当前 539287fa 分栏，后续评测由任务 01a0660c-d75e-7f92-8921-d902ce792561 独立负责，本监督任务不得再取消或重提"},
            {"job_ids": "1267747,1267748", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_formal_21d1d229", "source_head": "21d1d22975686852d0c1dc31a0f62419252f17d4", "reason": "Evidence C0 首次正式启动因 DataLoader 219 batches 与 100-update 合同实现冲突而失败；0d1abf6d 修复了更新暴露合同"},
            {"job_ids": "1267857,1267858", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_formal_0d1abf6d", "source_head": "0d1abf6dc0b3b5f13c1f18118e0689af32d84229", "reason": "Evidence C0 第二次启动读取真实 legacy ledger 时发现 policy_source/config-hash 契约与行 schema 错配；77c8d173 已改为绑定 policy=c3_lowres_probe_delta_p_action 并在 admission 扫描三份 ledger"},
            {"job_ids": "1267818", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_formal_52c940f2", "source_head": "52c940f20d099a53c954c5533a68018294665e8f", "reason": "BAFDR D160 教师提交脚本由 /bin/sh 执行 source/pipefail，启动即失败；539287fa 已改为 bash -lc 并重新运行 CUDA/focused 门禁"},
            {"job_ids": "1268680", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_formal_77c8d173", "source_head": "77c8d173c95aef153c04fd1355a0e75a63ff22c9", "reason": "Evidence 训练完成后的首轮评测把 checkpoint 定位到 seed_8261/checkpoint，漏掉训练器自动添加的 gpu1_id0；7934e0c9 修复路径、训练提交身份与独立评测输出命名空间，1269270 正在等待资源复验"},
            {"job_ids": "1269230,1269231,1269233", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_admission_e553a5a4", "source_head": "e553a5a4a1063a755900d3dfa4bf8909bf97d466", "reason": "H65 三臂训练已成功，但首轮 evaluator 使用 canonical_jsonable，而训练器对 dataclass 使用 default=str，导致 resolved_config_sha256 口径不一致；a88388d9 统一训练哈希并隔离评测输出，1269269/1269271 已排队"},
            {"job_ids": "1269271,1269283,1269285", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_eval_a88388d9", "source_head": "a88388d9dd4815de7664bae782aca11d4e89b1f4", "reason": "H65 evaluator 在不同 clean checkout 上把绝对 source_config_path 当成必须相同的身份字段；ca8337e7 改为 basename 加严格 source_config_sha256 验证，未放宽配置内容绑定"},
            {"job_ids": "1269294", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_eval_ca8337e7", "source_head": "ca8337e7b293c36b265471fdc12667a985aadae4", "reason": "H65 D768 首次 ca8337e7 评测未恢复训练时 raw Validation/Test 路径，resolved_config_sha256 正确拒绝；1269303 已用原始路径重提"},
            {"job_ids": "1269265,1269266", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_eval_120df5a4 和 /data/run01/sczc063/yuzibo/projects/duca_evidence_eval_30fee9f1", "source_head": "120df5a46fa51dee30ba41c70bdcc32ec53b6b60 / 30fee9f1b3c75714c7f907625c3e01f94f5c8af7", "reason": "中间修复 PRECHECK 在等待资源时被最终的独立评测输出命名空间修复 a88388d9/7934e0c9 取代，已主动取消；属于 superseded，不是模型失败"},
            {"job_ids": "无 Slurm job id", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2", "reason": "BAFDR screen 提交器在 FULL 前用系统旧 Python 执行含 f-string 的教师检查，SyntaxError 后退出；5a199a49 已绑定项目 Python 并通过本地/远端 11 tests，FULL 的 539287fa 精确训练由独立每分钟 watcher 续提"},
            {"job_ids": "1269284", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_formal_539287fa", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2", "reason": "BAFDR FULL 构造 D160 teacher 时相对 pretrain 未获得 YUZIBO_ROOT，启动 40 秒后失败；54bd3cf2 只修复 launcher 环境导出，PRECHECK 1269296 PASS，FULL 以原 539287fa 身份重提为 1269297"},
            {"job_ids": "1269286,1269298", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_eval_7934e0c9 和 /data/run01/sczc063/yuzibo/projects/duca_evidence_eval_0e11fba1", "source_head": "7934e0c9ce8d003fdaed99e433bfc6b7edc3b988 / 0e11fba15b1e539b84788417ea6e8c78cdaee893", "reason": "Evidence C0 评测依次暴露跨 checkout source_config_path 绑定错误和未恢复训练时 raw data/三份 ledger 环境；d6f1cc28 已同时保持严格 config hash并恢复完整训练环境"},
            {"job_ids": "1269323,1269324", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_eval_96aba608", "source_head": "96aba608（由 67c8f39f 取代）", "reason": "H65 U384/MNV evaluator 对 H65 专用 runtime binding 错误索引 generic gate_suite_sha256，触发 KeyError；67c8f39f 保持 H65 专用绑定语义并通过远端 21 tests、admission 1269353，原失败日志保留"},
            {"job_ids": "1269325,1269354,1269356", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_eval_eea0eea4 及后续 Evidence 修复 checkout", "source_head": "eea0eea4 / 1b905ac7 / 032c555c（由 246058f2 取代）", "reason": "依次暴露 test ledger 不覆盖 792 loader 暴露、full-grid train ledger 与 438-window 训练拓扑不符、覆盖检查器未接受归一化 NumPy ndarray；246058f2 修复后 admission 1269357 PASS，失败均保留且未作为结果"},
            {"job_ids": "1269287,1269288,1269291,1269292", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_formal_e553a5a4", "source_head": "e553a5a4a1063a755900d3dfa4bf8909bf97d466", "reason": "H65 F01 与 F02 均在课程切换附近遇到 p_action calibration 一致性异常并失败，依赖评测 1269288/1269292 已取消释放额度。30514803 增加 dtype-aware 有界容差、非有限值专门失败和回归测试，远端 23 tests PASS；仍须逐臂重跑 admission 并跨过原故障点后才能确认修复"},
            {"job_ids": "1269341,1269342,1269358,1269359,1269361,1269362,1269367,1269368", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_formal_e553a5a4", "source_head": "e553a5a4a1063a755900d3dfa4bf8909bf97d466", "reason": "H65 F03 在 epoch 23 复现与 F01/F02 相同的 p_action calibration 错误，依赖评测 1269342 取消；确认共享缺陷后，旧 SHA 上仍运行的 F04-F06 及其依赖评测被主动停止以避免继续浪费 GPU，日志和 checkpoint 保留。30514803 的 F01-F04 逐臂 PRECHECK 已通过并重训"},
            {"job_ids": "1269382", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_formal_539287fa", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2", "reason": "BAFDR seed 4407 五臂评测外层已进入 exact checkout，但嵌套脚本优先采用 SLURM_SUBMIT_DIR=/data/home/sczc063，触发 checkout HEAD mismatch；显式导出 PROJECT_DIR 后以 1269383 重提，未改模型、数据或 checkpoint"},
            {"job_ids": "1269383", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_formal_539287fa", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2（训练）", "reason": "完成 G96 prediction-only 后，U16 validation 的生成配置错误请求 gt_segments/gt_labels，触发 KeyError；02d34e6b 将 val/test 管道修正为 masks-only，远端 11 tests、21-cell validator 和双 GPU PRECHECK 1269388 均 PASS，正式评测以 1269389 重提"},
            {"job_ids": "1269389", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_eval_02d34e6b", "source_head": "02d34e6b146c62df0300007d75019a6c665ef2cf（评测） / 539287fa8a035765afd7e79863ce77278bef83f2（训练）", "reason": "成功完成 G96 prediction receipt 后，U16 后处理读取 meta['snippet_stride'] 时失败；原因是 BAFDR 自定义 Collect.meta_keys 覆盖默认字段却漏掉 snippet_stride 和 offset_frames。29b5a7a2 补齐标准元数据并强化 21-cell validator，双 GPU PRECHECK 1269540 PASS，正式评测以 1269541 重提"},
            {"job_ids": "1265777-1265778", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902", "source_head": "679b71214d05a21cd08ae1f5e5c3879e2df8fb83", "reason": "旧 CT-DP checkout；G0/G1 已完成并留下 epoch_59.pth 与 Average-mAP 63.95% 日志，但无当前 exact SHA 或 audit-owned terminal receipt，mAP 不纳入结果"},
            {"job_ids": "1265779-1265780", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_ctdp_revised_20260902", "source_head": "679b71214d05a21cd08ae1f5e5c3879e2df8fb83", "reason": "旧 CT-DP checkout；G2/G3 仍在运行，不能迁移为当前冻结或修正路线结果"},
            {"job_ids": "1265077_[0-2,3-7]", "remote_directory": "/data/run01/sczc063/yuzibo/projects/opentad_duca_evidence_recovery", "source_head": "647151facd36d4df3f21de6865bcb225c8ba91fc", "reason": "dirty 且旧 Evidence checkout；失败/完成作业均不纳入当前结果，缺少 exact SHA 与终态 receipt"},
        ],
    }


def md_text(payload: dict[str, Any]) -> str:
    lines = [
        "# DUCA/ZoomToken 全部代码实验目录",
        "",
        f"最后更新时间（UTC）：`{payload['last_updated_utc']}`",
        "",
        "本表用完整中文描述实验目的；括号中的内部 ID 仅用于与 Slurm/manifest 对照。每一行都是独立代码身份，结果不能跨 SHA 转移。",
        "",
        "## 当前实验与修正路线",
        "",
        "| 实验名称（面向外部读者） | 本地目录 | GitHub 提交 | 部署状态 | 结果状态与最终结果 | 下一步 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in payload["entries"]:
        name = f"{entry['name']}（`{entry['internal_id']}`）"
        commit = f"[`{entry['sha'][:8]}`]({entry['github_commit']})"
        result = f"{entry['result_status']}：{entry['final_result']}"
        local_paths = [entry["local_directory"], *entry.get("supporting_local_directories", [])]
        local_directory = "<br>".join(f"`{path}`" for path in local_paths)
        lines.append(f"| {name} | {local_directory} | {commit} | {entry['deployment_status']} | {result} | {entry['next_action']} |")
    lines += [
        "",
        "## 监督器与动态状态",
        "",
        f"远端 N16R4 监督器：`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902`，每 60 秒轮询；本地 heartbeat 每 30 分钟刷新本表。当前远端监督器状态：`{payload['remote_supervisor'].get('status')}`，dispatcher：`{payload['remote_supervisor'].get('dispatcher_status', '未知')}`。",
        "",
        f"集群观测（{payload['cluster_observation']['checked_at_cst']}）：公共分区可用 GPU {payload['cluster_observation']['public_gpu_available']}/{payload['cluster_observation']['public_gpu_total']}，本用户队列 {payload['cluster_observation']['user_jobs_in_queue']} 项；当前约束为 {payload['cluster_observation']['account_constraint']}。",
        "",
        "## 明确排除的旧远端作业",
        "",
        "这些作业可以继续作为诊断材料，但不属于当前冻结实验，不能写入最终结果：",
        "",
        "| 作业号 | 远端目录 | source HEAD | 排除原因 |",
        "|---|---|---|---|",
    ]
    for item in payload["excluded_remote_jobs"]:
        lines.append(f"| `{item['job_ids']}` | `{item['remote_directory']}` | `{item['source_head'][:8]}` | {item['reason']} |")
    lines += [
        "",
        f"结果规则：{payload['result_policy']}。当前只有 H65 D768 与 U384 两臂具有合法终态 evaluator metrics；其余路线或比较矩阵仍为 `NO_FINAL_PERFORMANCE_RESULTS`。终态训练 receipt 只能证明训练合同完成，不得从 admission 或中期验证推导最终 mAP、speedup、bootstrap 或 cost。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default=str(DEFAULT_JSON))
    parser.add_argument("--markdown", default=str(DEFAULT_MD))
    args = parser.parse_args()
    payload = catalog()
    json_path = Path(args.json)
    md_path = Path(args.markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(md_text(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "entries": len(payload["entries"]), "remote_supervisor": payload["remote_supervisor"].get("status")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
