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
            "deployment_status": "远端 exact-SHA admission 1267684 已通过；REF-D768、REF-U384、REF-MNV3FC384 正式训练 1267709/1267711/1267737 均 COMPLETED(0:0)。首轮评测 1269230/1269231/1269233 因训练/评测配置哈希序列化口径不一致失败；评测修复分支 codex/h65-pro-eval-binding-repair-20260903@a88388d9 已推送并部署到 /data/run01/sczc063/yuzibo/projects/h65_pro_eval_a88388d9。修复后 PRECHECK 1269269 因账户 AssocGrpGRES 上限排队，D768 评测 1269271 已按 afterok 入队，U384/MNV 由每分钟 watcher 续提",
            "result_status": "三臂训练终态有效，官方性能评测待资源；尚无最终性能",
            "final_result": "三个训练审计均绑定 e553a5a4、seed 3407、epoch 59 state_dict_ema、60 epochs、6000 successful/EMA/scheduler updates；U384/MNV selector schedule 也为 6000。评测错误未污染 checkpoint，但在新 evaluator 成功前不得报告最终 mAP",
            "next_action": "等待 1269269 获得 GPU 并核验 PASS；随后完成 1269271 及 watcher 提交的 U384/MNV 官方 THUMOS14 评测",
        },
        {
            "category": "correction_route",
            "name": "CT-DP 当前正式四臂：基础嵌入、CT-Tubelet、B-AMoD 及二者组合",
            "internal_id": "CT_DP_BAMOD_ACTIVE",
            "branch": "codex/duca-ctdp-formal-repair-20260903",
            "sha": "c0fae67a1236f2c47e6c2935d217659cd1f8fb9d",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/c0fae67a1236f2c47e6c2935d217659cd1f8fb9d",
            "local_directory": "E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903",
            "deployment_status": "远端 admission 1267221 已通过；G0、G1、G2、G3 正式训练 1267229-1267232 均持续 RUNNING",
            "result_status": "G0/G1 已进入 epoch 51，G2/G3 已进入 epoch 53；尚无终态性能",
            "final_result": "最新中期验证 G0/G1/G2/G3 的 Avg-mAP 分别为 14.53/14.69/56.43/57.51%，mAP@0.7 为 2.91/3.28/26.13/30.92%；这些是训练中周期验证，只能用于健康监控，不能作为最终比较",
            "next_action": "等待四臂终态 checkpoint，随后运行同一 evaluator 并做配对比较",
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
            "branch": "codex/zoomtoken-bafdr-admission-fix-20260903",
            "sha": "539287fa8a035765afd7e79863ce77278bef83f2",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/539287fa8a035765afd7e79863ce77278bef83f2",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_admission",
            "deployment_status": "真实 CUDA/focused 门禁 1267855 已通过；D160 教师 1267884 与 G96 1268698 均 COMPLETED(0:0)。U16 1269124、LATE 1269129、NOKD 1269137 正在运行。FULL 提交前检查曾因系统旧 Python 执行 f-string 而中止；提交器修复 codex/zoomtoken-bafdr-admission-fix-20260903@5a199a49 已推送、远端 11 tests PASS，539287fa 训练身份的 FULL 每分钟 watcher PID 3911350 正等待提交名额",
            "result_status": "D160 与 G96 训练终态有效；U16 已到 update 3500，LATE 到 2000，NOKD 到 500；尚无五臂最终性能",
            "final_result": "D160 receipt 绑定 539287fa、seed 4407、6000 updates 与 terminal checkpoint；G96 已完成但尚待统一 screen finalizer。FULL 仍使用 539287fa 训练代码与同一 D160 教师，不把提交器 5a199a49 冒充为训练身份；历史 efe69d2e 继续由独立任务评测",
            "next_action": "等待账户提交槽释放并自动提交 FULL；五臂均完成后运行 screen finalizer，再决定是否开放后续矩阵",
        },
        {
            "category": "correction_route",
            "name": "ET-TRC 当前双臂：完整 Transformer 与局部 Taylor/JVP 近似",
            "internal_id": "ET_TRC_ACTIVE",
            "branch": "codex/zoomtoken-et-trc-formal-repair-20260903",
            "sha": "74473c2775caebf0da9d368ce8009d78e2942098",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/74473c2775caebf0da9d368ce8009d78e2942098",
            "local_directory": "E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902",
            "deployment_status": "真实 VideoMAE 覆盖与 2-GPU/global-batch=2 admission 已通过；OFF 1267218 与 ON 1267219 正在运行",
            "result_status": "OFF 已进入 epoch 53，ON 已进入 epoch 52；最新中期验证 OFF Avg-mAP 62.52%、ON 55.22%，仍不是终态性能",
            "final_result": "最新中期验证 OFF/ON 的 tIoU 0.3/0.4/0.5/0.6/0.7 分别为 77.51/72.63/65.49/55.00/42.00% 与 71.06/65.40/57.63/47.08/34.94%；当前作业持续运行，需等待 epoch-59 EMA 才能裁决",
            "next_action": "等待 OFF/ON epoch-59 EMA 与官方评估，比较定位性能和执行算子计数",
        },
        {
            "category": "historical_exact_route",
            "name": "BAFDR 历史五臂三种子终态 checkpoint 独立评测",
            "internal_id": "BAFDR_EFE69D2E_EVAL",
            "branch": "历史远端 exact-SHA 结果身份",
            "sha": "efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "local_directory": "E:/DeskTop/TAD/NO_LOCAL_WORKTREE_FOR_EFE69D2E",
            "deployment_status": "1266410-1266414 已产生 15 个 epoch-59 checkpoint；独立任务正在运行 prediction eval 1267920，postprocess 1267921 afterok 等待",
            "result_status": "历史 exact-SHA 评测进行中，尚无可报告终态指标",
            "final_result": "只按 efe69d2e 的真实训练/teacher/数据/evaluator receipts 报告，绝不迁移为当前 539287fa 结果",
            "next_action": "等待 792-window prediction seal、C_exec、官方 mAP opening 与 strict completeness receipt",
        },
        {
            "category": "correction_route",
            "name": "Evidence-Recovery 当前 C0：轻量侦察器不确定性与最大空洞补漏",
            "internal_id": "EVIDENCE_ACTIVE",
            "branch": "codex/duca-evidence-formal-repair-20260903",
            "sha": "77c8d173c95aef153c04fd1355a0e75a63ff22c9",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/77c8d173c95aef153c04fd1355a0e75a63ff22c9",
            "local_directory": "E:/DeskTop/TAD/OpenTAD_Evidence_FormalRepair_20260903",
            "deployment_status": "actual-ledger admission 1267969 已通过；C0 seed 8261 正式训练 1267979 COMPLETED(0:0)。首轮评测 1268680 因 checkpoint 路径漏写 gpu1_id0 失败；评测修复分支 codex/duca-evidence-eval-path-repair-20260903@7934e0c9 已推送并部署到 /data/run01/sczc063/yuzibo/projects/duca_evidence_eval_7934e0c9。修复后 PRECHECK 1269270 因账户 AssocGrpGRES 上限排队，C0 评测由每分钟 watcher 等待提交名额",
            "result_status": "C0 训练终态有效，官方性能评测待资源；尚无最终性能",
            "final_result": "终态审计绑定 77c8d173、seed 8261、epoch 59 state_dict_ema、60 epochs、6000 successful/EMA/scheduler updates、nonfinite=0；评测路径错误未污染 checkpoint，新 evaluator 成功前不得报告 official mAP",
            "next_action": "等待 1269270 获得 GPU并核验 PASS；随后由 watcher 提交 C0 官方评测，再按结果决定是否释放其余七臂",
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
            {"job_ids": "1269265,1269266", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_eval_120df5a4 和 /data/run01/sczc063/yuzibo/projects/duca_evidence_eval_30fee9f1", "source_head": "120df5a46fa51dee30ba41c70bdcc32ec53b6b60 / 30fee9f1b3c75714c7f907625c3e01f94f5c8af7", "reason": "中间修复 PRECHECK 在等待资源时被最终的独立评测输出命名空间修复 a88388d9/7934e0c9 取代，已主动取消；属于 superseded，不是模型失败"},
            {"job_ids": "无 Slurm job id", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2", "reason": "BAFDR screen 提交器在 FULL 前用系统旧 Python 执行含 f-string 的教师检查，SyntaxError 后退出；5a199a49 已绑定项目 Python 并通过本地/远端 11 tests，FULL 的 539287fa 精确训练由独立每分钟 watcher 续提"},
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
        lines.append(f"| {name} | `{entry['local_directory']}` | {commit} | {entry['deployment_status']} | {result} | {entry['next_action']} |")
    lines += [
        "",
        "## 监督器与动态状态",
        "",
        f"远端 N16R4 监督器：`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902`，每 60 秒轮询；本地 heartbeat 每 30 分钟刷新本表。当前远端监督器状态：`{payload['remote_supervisor'].get('status')}`，dispatcher：`{payload['remote_supervisor'].get('dispatcher_status', '未知')}`。",
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
        f"结果规则：{payload['result_policy']}。当前性能结果账本仍为 `NO_FINAL_PERFORMANCE_RESULTS`；已有终态训练 receipt 只能证明训练合同完成，不得从 admission 或中期验证推导最终 mAP、speedup、bootstrap 或 cost。",
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
