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
    if proc.returncode != 0:
        return None
    return not proc.stdout.strip()


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
        "dispatcher_mode": receipt.get("dispatcher", {}).get("mode"),
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
            "receipt_recheck": "2026-09-07 19:23 CST：六份F01-F06 metrics JSON可读，evaluation_sha256逐份复算一致，引用的epoch59 checkpoint均存在；未重复推理或改写指标",
            "current_cycle": "2026-09-07 基线口径纠正：上游 AdaTAD VideoMAE-S/768/160 公开结果为 69.03%；项目登记的未修改共享复现为 68.73%（本轮未重新读取原始运行产物）。历史 uniform384 的 native stride-2 为 64.352%，physical-grid 版本为 65.696%，均与本轮协议不匹配。67.58%/63.89% 只保留为 strict6000 修改协议的实测参考，不能称为已复现的官方 AdaTAD 基线。已确认学习率余弦周期、world size、终态模型选择和 U384 后端有协议差异；差异对分数的因果贡献尚未隔离。F02 的 +0.34 仅是相对本轮 U384 的数值差，不能据此宣称超过约 65% 的历史均匀基线",
            "branch": "codex/h65-pro-admission-fix-20260902（参考臂） + codex/h65-pro-physical-time-optimizer-repair-20260904（F 臂）",
            "sha": "e553a5a4a1063a755900d3dfa4bf8909bf97d466（参考臂） / f2068e18e2c68bdbdd7a607b47f32d05ac3beed7（F 臂）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/h65_eval_repair"],
            "deployment_status": "远端 exact-SHA admission 1267684 已通过；REF-D768、REF-U384、REF-MNV3FC384 正式训练 1267709/1267711/1267737 与官方评测 1269303/1269360/1269373 均 COMPLETED(0:0)。30514803 的 F01/F02/F04 在 AMP replay 后独立出现 probe 全 NaN，F03/F05/F06 随即安全停止并保留 checkpoint。精确 RNG 复现 1269647、梯度审计 1269698/1269699 定位到 relative_physical_time_scale；独立修复 f2068e18 将该标量纳入零 weight-decay optimizer group，validator 强制该合同；本地 H65 tests 9 passed/11 skipped、C3 tests 23 passed，远端 exact-SHA clean checkout 59 passed/1 skipped。F01-F06 PRECHECK 全部通过，正式重训 1269705/1269707/1269759/1269761/1269764/1269769 均 COMPLETED(0:0)；六臂均有 epoch_59.pth/state_dict_ema，training audit 均确认 optimizer/scheduler/EMA/DUCA schedule 各 6000 successful updates，日志未见 non-finite/NaN/traceback/OOM。首轮评测 1270970-1270975 因错误使用训练 checkout 的旧 evaluator，全部在推理前以 resolved_config_sha256 mismatch 退出；日志已读，checkpoint 未损坏。独立 evaluator 67c8f39f 的 PRECHECK 1270985 COMPLETED(0:0)，修正后的官方评测 1270987-1270992 均 COMPLETED(0:0)；六份结构化 metrics receipt 均绑定训练 SHA f2068e18、evaluator SHA 67c8f39f、seed 3407、epoch 59 state_dict_ema、6000 successful updates，且 evaluation_sha256 自哈希复算一致",
            "result_status": "D768、U384、MNV 及 F01-F06 有本协议下官方评估器产生的终态数值与收据；这不等于未修改 AdaTAD 的复现成功或机制有效。仅 seed 3407，非完整 24 臂或多种子结论",
            "final_result": "参考臂：D768 Avg-mAP=67.58%，U384=63.89%，MNV3FC384=57.01%。H65 F01-F06 official Avg-mAP 分别为 63.3854%、64.2265%、60.3648%、60.7013%、64.0886%、63.8181%；对应 mAP@0.3/0.4/0.5/0.6/0.7：F01=79.1680/74.3484/66.5598/56.0579/40.7929，F02=79.9377/74.5653/67.3877/56.1257/43.1163，F03=75.9121/70.5691/63.0241/52.9793/39.3393，F04=75.8391/71.1599/63.8342/53.5592/39.1141，F05=79.2504/74.3064/67.1331/56.9056/42.8476，F06=79.2828/74.6584/66.6302/56.2470/42.2723。相对 U384，F02 最好为 +0.34 个百分点，F05 为 +0.20，F06 为 -0.07，F01 为 -0.50，F03 为 -3.53，F04 为 -3.19；所有 F 臂仍低于 D768",
            "next_action": "保留原始数值和收据；先核对共享 68.73% 的原配置/产物、均匀384的真实索引/坐标与各臂完整训练配方，厘清基线退化和配对差异后再决定重训范围。不得因新数值较低而替换历史基线，也不得按目标分数调参或重复训练共享未修改 AdaTAD",
        },
        {
            "category": "correction_route",
            "name": "CT-DP 当前正式四臂：基础嵌入、CT-Tubelet、B-AMoD 及二者组合",
            "internal_id": "CT_DP_BAMOD_ACTIVE",
            "current_cycle": "2026-09-07 19:45 CST：修复版1276669/1276670/1276671/1276672四臂全部RUNNING。审计实际optimizer/scheduler/EMA分别为G0=1800/1800/1800、G1=1800/1800/1800、G2=1700/1700/1700、G3=1600/1600/1600；AMP skips分别3/3/2/3，均通过同批重放补齐。四对stdout/stderr未发现新Traceback、RuntimeError或replay耗尽，保留插值算子的nondeterministic warning，不把warning当训练失败。最终GPU admission1276643已通过49tests及G0-G3两轮三批真实PRECHECK；不重复已通过验证或重提现有训练",
            "branch": "codex/duca-ctdp-successful-updates-20260907",
            "sha": "78cde1be1cb8b3acc7d750afc92ea740f2a03d06（更新合同及测试修复） / c0fae67a1236f2c47e6c2935d217659cd1f8fb9d（旧训练）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/78cde1be1cb8b3acc7d750afc92ea740f2a03d06",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_successful_updates",
            "supporting_local_directories": ["E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903"],
            "remote_source": "/data/run01/sczc063/yuzibo/projects/ctdp_successful_updates_78cde1be",
            "repair_run_root": "/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be",
            "admission_job": "1276643",
            "admission_progress": "1276643 COMPLETED(0:0): 49 CUDA tests passed; CT_CUDA_GATE_OK; all G0-G3 real prechecks passed; CTDP_ALL_PRECHECKS_OK",
            "formal_jobs": {"G0": "1276669", "G1": "1276670", "G2": "1276671", "G3": "1276672"},
            "formal_job_states": {"G0": "RUNNING; 1800 successful updates", "G1": "RUNNING; 1800 successful updates", "G2": "RUNNING; 1700 successful updates", "G3": "RUNNING; 1600 successful updates"},
            "observed_successful_updates": {"G0": 1800, "G1": 1800, "G2": 1700, "G3": 1600},
            "deployment_status": "G0/G1/G2/G3 训练 1267229/1267230/1267231/1267232 均 COMPLETED(0:0)，四个 epoch_59.pth/EMA 均存在。但 2026-09-07 直接读取 checkpoint，实际 optimizer step 分别只有 5997/5996/5996/5997，scheduler 均为 6000。源码确认该路线未启用正式 AMP replay，跳过更新仍推进 scheduler",
            "result_status": "旧四臂仍为PROTOCOL_ERROR；78cde1be已通过GPU/逐臂PRECHECK，四个修复版正式训练已提交，尚无新最终性能",
            "final_result": "仅保留旧训练 telemetry：Avg-mAP G0=14.71%、G1=14.84%、G2=56.15%、G3=57.84%；mAP@0.7=2.84/3.26/26.24/31.53%。旧 checkpoint 和日志不删除，不把 6000 scheduler steps 改称 6000 successful updates，也不据此裁决机制贡献",
            "next_action": "监督1276669-1276672，日志repair_run_root/slurm_logs/formal_g{0..3}_JOB.out/.err；checkpoint预期repair_run_root/formal/duca_ctdp_geometry_g{0..3}_seed3407/gpu1_id0/checkpoint/epoch_59.pth。不得重复已通过admission或重提现有四臂。训练后仍需独立官方evaluator与自哈希receipt，不能把训练telemetry当最终性能；该独立评测闭环尚待接入",
        },
        {
            "category": "correction_route",
            "name": "DUCA-Unified 当前 41 单元正交消融控制台",
            "internal_id": "DUCA_UNIFIED_ACTIVE",
            "current_cycle": "2026-09-07 19:23本地HEAD仍为793c4f9c且clean。此前定位的缺失机制未修复：feature_attribution.py的Taylor函数无运行时调用，duca_online_frame_selector.py:449附近仍为GT边界utility代理，generator:60附近把H65 original retention/transition映射成legacy_dual_phase。本轮优先处理已失败Evidence的续训及补部署BAFDR，没有把Unified阻塞改称实现完成",
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
            "current_cycle": "2026-09-07 15:54 CST 实质纠正：先读六个原训练stdout/stderr，再直接加载六个epoch59 checkpoint的AdamW state。D160/G96/U16/LATE/NOKD/FULL实际更新为5996/5997/5996/5996/5994/5995，所有已初始化参数的step一致，但scheduler及total_successful_updates都写6000。确认train_epoch无条件推进scheduler/EMA/计数，遗漏GradScaler skip。此前依据自报计数判定严格6000有效的结论撤回，完整自哈希不能证明计数语义正确。独立修复710ce8a6保持模型/损失/超参数不变，加入同批AMP重放、学生和教师buffer/RNG恢复、真实计数、逐轮AdamW核对、教师与evaluator拒绝不足6000的checkpoint；PRECHECK改为独立目录两轮各三批。本地41 passed/1 Windows Torch skipped，独立有限审阅已处理checkpoint只能评测的歧义。远端Linux/CUDA和新训练状态以随后证据为准，当前不称修复完成部署。教师加载/归一化/KD目标detach静态检查未发现足以解释大幅低分的缺陷；hard排序不可导，但残差gate仍在梯度图中，不能说整个router只受BCE训练",
            "branch": "codex/zoomtoken-bafdr-successful-updates-20260907（修复） + codex/zoomtoken-bafdr-eval-metadata-repair-20260904（旧评测）",
            "sha": "710ce8a6246c471742c83bf7c180d9ab87c36fac（更新合同修复） / 539287fa8a035765afd7e79863ce77278bef83f2（旧训练） / 29b5a7a2b291203ea7b697cfe416b64f0d365d02（旧评测）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/710ce8a6246c471742c83bf7c180d9ab87c36fac",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_successful_updates",
            "deployment_status": "真实 CUDA/focused 门禁 1267855 已通过；D160 教师 1267884、G96 1268698、U16 1269124、LATE 1269129、NOKD 1269137、FULL 1269297 均 COMPLETED(0:0)，全部写出 6000-update epoch-59 receipt，screen_receipt_r2_1269297.json 为 PASS。评测器 29b5a7a2 已通过本地/远端 11 tests、21-cell validator 和双 GPU U16 PRECHECK 1269540。五臂评测 1269541 已生成 G96/U16/LATE/NOKD 四份 prediction receipt，FULL 因 DECORD EOF 失败；1269763 的首次定向重试又因 wrapper 未显式传入外部 D160 Teacher 路径而 fail-closed。读取两次日志后，1269771 显式绑定训练 checkout 的 Teacher config/checkpoint 及既有 SHA256、通过 PRECHECK，完成 FULL prediction 和五臂 metric opening，并于 10:16 CST COMPLETED(0:0)",
            "result_status": "PROTOCOL_ERROR：五臂和D160教师均不足6000次真实optimizer更新，无本合同下有效最终性能。84f1f035补封存保留了真实评测身份和数值，却沿用了错误的total_successful_updates字段；1274829完成及自哈希通过不能修复训练预算违规。原始评测和权重保留，以下数值仅供诊断",
            "final_result": "保留的协议不合格诊断数值：G96官方评估器Avg-mAP=50.93%，各阈值64.67/59.83/53.47/44.68/32.00%；U16均匀分块48.17%，63.50/57.95/50.12/40.46/28.84%；晚期融合LATE53.11%，68.84/63.39/56.05/45.40/31.88%；无蒸馏NOKD49.44%，64.86/58.81/51.60/41.85/30.09%；完整FULL52.38%，67.63/62.71/54.60/45.14/31.83%。五份旧receipt绑定seed4407/evaluator29b5a7a2，但其中自报6000 successful updates已被真实AdamW state证伪。FULL绑定539287fa的D160教师也只有5996更新。不得据此宣称满足严格预算、模型优劣或多种子显著性；原始分数和checkpoint不改写",
            "next_action": "监督D1601276650、G961276675与新补交U16均匀分块1276842、LATE晚融合1276843；后两项19:45因AssocGrpGRES排队。NOKD无蒸馏臂的PRECHECK也已通过，尚未提交，额度释放后补交且不得重复前四作业。终态预期repair_run_root/work_dirs/bafdr_k16_{d160,g96,u16_uniform_a0,late,nokd}_seed4407/checkpoint/epoch_59.pth。FULL仍须等待新D160真实6000更新终态，再做自身带明确Teacher绑定的PRECHECK与训练；旧五臂分数仅诊断",
            "receipt_sealing_branch": "codex/zoomtoken-bafdr-receipt-seal-20260907",
            "receipt_sealing_github": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-receipt-seal-20260907",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_admission", "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_receipt_seal"],
            "remote_repair_source": "/data/run01/sczc063/yuzibo/projects/bafdr_successful_updates_710ce8a6",
            "repair_run_root": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6",
            "observed_optimizer_steps": {"D160": 5996, "G96": 5997, "U16": 5996, "LATE": 5996, "NOKD": 5994, "FULL": 5995},
            "repair_validation": {"commit": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "local": "41 passed, 1 Windows Torch skipped", "remote_exact_clean_linux": "44 passed, 1 CUDA skipped", "cuda_and_two_gpu_precheck": "1276618 COMPLETED(0:0): 45 tests passed; D160/G96/U16/LATE/NOKD各两轮三批，BAFDR_SUCCESSFUL_UPDATE_PRECHECKS_OK", "formal_retraining": "19:45 CST: D1601276650已完成epoch33，optimizer/scheduler/EMA各3400；G961276675完成epoch31，各3200。两臂各4次AMP skip均已replay，非有限loss与replay耗尽均0。核验既有PRECHECK成功后补交U161276842、LATE1276843，均PENDING AssocGrpGRES；NOKD仍待额度，FULL等待新终态教师"},
            "admission_job": "1276618",
            "teacher_training_job": "1276650",
            "g96_training_job": "1276675",
            "u16_training_job": "1276842",
            "late_training_job": "1276843",
            "observed_successful_updates": {"D160": 3400, "G96": 3200},
        },
        {
            "category": "correction_route",
            "name": "ET-TRC 当前双臂：完整 Transformer 与局部 Taylor/JVP 近似",
            "internal_id": "ET_TRC_ACTIVE",
            "receipt_recheck": "2026-09-07 19:23 CST：OFF/ON metrics JSON可读，receipt_sha256均复算一致；已有独立官方数值不变，未重跑评测",
            "branch": "codex/zoomtoken-et-trc-formal-repair-20260903（训练） + codex/zoomtoken-ettrc-terminal-eval-20260907（独立评测）",
            "sha": "74473c2775caebf0da9d368ce8009d78e2942098（训练） / 67d7079d0d1c33e129d31cd3a45aaf93a67db252（评测）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/67d7079d0d1c33e129d31cd3a45aaf93a67db252",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval",
            "supporting_local_directories": ["E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902"],
            "current_cycle": "独立官方评测1275624_0 OFF、1275624_1 ON均COMPLETED(0:0)，用时15:52/15:42。两份metrics.json已读取并复算receipt_sha256通过：OFF=5d0fcf57bb74f5acce2a9ddcd0ba597a4579398a109532f90a011104159cb3d0，ON=06c254b83d4fed0608d243afbd2d257d1bbfd2c67c2973efe9903b78b3e0d445。绑定原训练74473c27/evaluator67d7079d、clean tree、epoch59 EMA、optimizer/scheduler6000、2GPU/global batch2；日志未发现新异常。不是把旧telemetry改名为新结果，独立推理及官方evaluator已实际完成。validation对应211个video_test_*，不更改划分",
            "deployment_status": "OFF1267218/ON1267219正式训练及epoch59 EMA均完成；独立评测代码67d7079d、本地/远端28 tests、2GPU PRECHECK1275569_0/1通过，完整官方评测1275624_0/1和结构化自哈希receipt现已完成",
            "result_status": "双臂具有可追溯的官方终态结果；本种子开启Taylor/JVP近似明显退化，不支持保持准确率的主张；算子实现忠实度和训练配方的因果排查尚未完成",
            "final_result": "OFF official Avg-mAP=62.076775%，mAP@0.3/0.4/0.5/0.6/0.7=77.063201/71.910513/65.342222/54.513056/41.554882%；ON official Avg-mAP=54.809580%，对应70.378016/64.889287/57.494778/46.651646/34.634174%。ON比OFF低7.267195个百分点，mAP@0.7低6.920708个百分点。单种子4407，不外推多种子显著性，也不把低分本身当作已证明实现错误",
            "next_action": "保留新独立收据、原训练权重和所有日志，不重复已完成的相同评测。围绕Taylor/JVP实际执行、初始化、训练梯度与OFF基线配方做有界原因分析，区分实现错误与方法负结果；发现确定缺陷才按修复/PRECHECK/新SHA重提，不按测试集分数调参",
            "remote_eval_root": "/data/run01/sczc063/yuzibo/experiments/ettrc_terminal_eval_67d7079d/formal",
            "remote_log_root": "/data/run01/sczc063/yuzibo/experiments/ettrc_terminal_eval_67d7079d/slurm_logs",
        },
        {
            "category": "historical_exact_route",
            "name": "BAFDR 历史五臂三种子终态 checkpoint 独立评测",
            "internal_id": "BAFDR_EFE69D2E_EVAL",
            "branch": "历史远端 exact-SHA 结果身份",
            "sha": "efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/efe69d2ea10accd01d0129dfe99cba4d1d5773cb",
            "local_directory": "E:/DeskTop/TAD/NO_LOCAL_WORKTREE_FOR_EFE69D2E",
            "deployment_status": "1266410-1266414已产生15个epoch59 checkpoint；独立prediction eval1267920 FAILED(1:0)。2026-09-07 17:57只读sacct显示1267921已由UID1258取消，本监督任务未执行该取消。该链仍归其他任务，本任务不修改、不重提",
            "result_status": "历史 exact-SHA 评测失败，尚无可报告终态指标",
            "final_result": "只按 efe69d2e 的真实训练/teacher/数据/evaluator receipts 报告，绝不迁移为当前 539287fa 结果",
            "next_action": "由历史评测任务读取 1267920 日志并决定修复；当前监督任务保持只读边界",
        },
        {
            "category": "correction_route",
            "name": "Evidence-Recovery 当前八臂：预选阶段轻量侦察器、不确定性与最大空洞补漏",
            "internal_id": "EVIDENCE_ACTIVE",
            "branch": "codex/duca-evidence-storage-resume-20260907（续训修复） + codex/duca-evidence-fullgrid-repair-20260904（C0/A2） + codex/duca-evidence-optimizer-repair-20260907与codex/duca-evidence-remaining-arms-20260907（原训练）",
            "sha": "ce767b4df49a82b42f4fab4897ac4ea562e4b548（续训接线） / 246058f2c24edc78818ada60eec26249bbf7d5d2（C0/A2） / 1570a72507491899a50767700a35a04eee3f5fe9（A1/A6恢复源） / 73bdd34ae21c6c675a00d1927b8af7c12f9edc05（F/A3-A5恢复源）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ce767b4df49a82b42f4fab4897ac4ea562e4b548",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_storage_resume",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/evidence_remaining_arms", "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_optimizer_repair", "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_eval_repair", "E:/DeskTop/TAD/OpenTAD_Evidence_FormalRepair_20260903"],
            "current_cycle": "2026-09-07 19:45：六个原正式训练仍为Disk quota exceeded失败，旧job/log/checkpoint全保留；恢复源A1/A6 epoch39为4000更新，F/A3/A4/A5 epoch29为3000更新，不能把崩溃前但未保存的更新计入续训。六项存储恢复PRECHECK均COMPLETED(0:0)，确认是两轮各100批共200批。独立续训修复分支ce767b4d已推送：保持opentad模型/更新源与科学配置不变，仅支持新输出目录、显式原SHA、旧计数/RNG恢复及resume_lineage。首个83ceb5f4真实checkpoint验证因漏保留work_dir末尾分隔符被runtime_config_sha256拒绝，ce767b4d修正并增加回归；没有远端热改或改写旧收据。本地49passed/5WindowsTorchskipped，远端最终精确clean SHA53passed/1CUDA skipped；实际六checkpoint接线及GPU验证仍按后续准入记录，不提前宣称正式续训已开始",
            "receipt_recheck": "2026-09-07 19:23 CST：C0/A2两份metrics JSON可读且evaluation_sha256复算通过，仍分别59.2292%/54.2756%；不属于完整Recovery结果",
            "deployment_status": "C0 训练 1269374_0 / 官方评测 1270672_0 完成；A2 训练 1270870_3 / 独立官方评测 1274711_3 也完成，均为原始 246058f2 训练及评测语义。A2 epoch59 EMA/6000 成功更新训练审计和 evaluation_sha256 自哈希复算通过。旧 A1 1270870_2 在 epoch54 batch11 非有限 utility 失败；A6 1270870_7 在 epoch0 batch2 非有限 cost 失败。FP32 修复 9739811e 的 A6 正式重试 1274671_7 又在 epoch1 batch2 非有限 cost 失败，旧日志/checkpoint 全保留。真实 ViT optimizer 回归定位：forward 重新启用时间参数，但 custom optimizer 只含 adapter，时间参数未被 optimizer/GradScaler 管理，可能经全局 clipping 扩散 NaN。1570a725 补齐 continuous_timestamp_conditioner/relative_physical_time_scale 分组，使用 dynamic DDP 和 non-reentrant checkpoint 支持随真实窗口变化的时间分支；不关闭 time/merge/recovery。远端 clean exact-SHA admission 1274755 全部 49 tests PASS；真实视频 A1/A6 两轮各 100 batch（共 200 batch）PRECHECK 1274759_2/7 均 COMPLETED(0:0)。正式 A1/A6 1274924_2/7 已 RUNNING；独立官方 evaluator 1274926_2 / 1274927_7 分别 afterok 依赖对应训练任务，尚未运行",
            "result_status": "C0/A2官方终态结果不变；其余六臂因存储失败尚无epoch59。目录迁移续训修复已提交，GPU恢复验证及正式续训状态单列；不把PRECHECK或原有中途checkpoint称为最终性能",
            "final_result": "C0 MATCHED_H65_60 Avg-mAP=59.23%，各 tIoU=75.22/70.01/61.54/50.90/38.48%。A2 NO_TIME official Avg-mAP=54.2756%，各 tIoU=72.2334/67.0169/57.6467/44.8214/29.6595%。C0 是 matched-H65 基线，A2 是关闭物理时间的消融，二者都不是完整 recovery 方法。test 792 次暴露对应 791 个唯一物理窗口（video_test_0001431|7680 重复），246058f2 按唯一物理窗口严格覆盖",
            "next_action": "收取ce767b4d精确SHA的CUDA及六臂真实恢复PRECHECK。通过后仅从原A1/A6 epoch39、F/A3/A4/A5 epoch29 checkpoint在新formal命名空间续训至epoch59/6000更新；不得从PRECHECK checkpoint续训，不复活旧死依赖eval。按六个新训练job分别登记独立官方eval。明确保留旧SHA与新SHA的更新边界，不把续训包装成新SHA从零完成6000更新",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/formal",
            "remote_log_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/slurm_logs",
            "remote_eval_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/evaluation",
            "remaining_arms_remote_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a",
            "storage_recovery_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_recovery_20260907",
            "storage_recovery_precheck_jobs": {"A1": "1276644_2", "A6": "1276644_7", "F": "1276645_1", "A3": "1276645_4", "A4": "1276645_5", "A5": "1276645_6"},
            "storage_recovery_precheck_status": "1276644_2/7及1276645_1/4/5/6均于18:10-18:13 COMPLETED(0:0)，两轮各100批共200批；只验证从头运行与存储写入，不是旧checkpoint恢复测试",
            "resume_source": "/data/run01/sczc063/yuzibo/projects/duca_evidence_storage_resume_ce767b4d",
            "resume_run_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d",
            "admission_job": "1276876",
            "admission_progress": "19:49 CST：1276876 PENDING(AssocGrpGRES)，包含最终ce767b4d CUDA admission及六臂各一轮真实checkpoint恢复PRECHECK，尚未开始。此前最终SHA clean Linux53passed/1CUDA skipped，六个真实checkpoint的metadata、原runtime摘要、科学绑定、4000/3000更新及EMA/optimizer/scaler/RNG均已通过CPU接线验证；该验证不代替实际GPU恢复",
            "resume_validation": {"commit": "ce767b4df49a82b42f4fab4897ac4ea562e4b548", "local": "49 passed, 5 Windows Torch skipped", "remote_exact_clean_linux": "53 passed, 1 CUDA skipped", "real_checkpoint_bindings": "ALL_SIX_REAL_CHECKPOINT_BINDINGS_OK", "cuda_and_real_resume_precheck_job": "1276876", "formal_resume_jobs": {}},
            "resume_precheck_stdout": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/resume_admission_1276876.out",
            "resume_precheck_stderr": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/resume_admission_1276876.err",
            "last_persisted_updates": {"A1": 4200, "A6": 4200, "F": 3100, "A3": 3100, "A4": 3200, "A5": 3200},
            "recoverable_checkpoint_epochs": {"A1": 39, "A6": 39, "F": 29, "A3": 29, "A4": 29, "A5": 29},
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
            "checked_at_cst": "2026-09-07 19:49 CST",
            "public_gpu_available": 35,
            "public_gpu_total": 200,
            "gpu_count_basis": "scontrol show nodes可见gpu分区，排除DOWN/DRAIN/FAIL/MAINT/RESERVED节点后CfgTRES为200、AllocTRES为165；全部配置含不可用节点为424，不沿用旧快照240作为本轮分母",
            "user_jobs_in_queue": 16,
            "account_constraint": "本用户12 RUNNING/4 PENDING，共16提交项。新BAFDR U161276842、LATE1276843与Evidence恢复准入1276876均因AssocGrpGRES排队，NOKD仍未提交；不取消其他任务腾额度。可调度节点35块未分配GPU不等于本账户可立即获得资源。本轮GPU分母200来自scontrol可见节点排除DOWN/DRAIN等状态后的统计，全部配置424含224不可用，不机械沿用旧240。实验挂载实际df为5.3T总量、465G可用、92%已用，不使用/data父挂载798T作为本用户空间；没有删除旧实验，未确认配额解除原因。历史1267920 FAILED、1267921 CANCELLED by1258均只读观察",
        },
        "measurement_scope": "当前比较以官方 TAD mAP、固定输入/更新预算和机制消融为主；不把端到端延迟、吞吐量或显存作为强制验收指标",
        "baseline_reference_note": "12_BASELINE_IDENTITY_CORRECTION_20260907.md",
        "baseline_comparison_policy": "区分上游公开结果、历史共享复现和本次修改协议实测；官方 evaluator 与完整 receipt 不代表官方训练 recipe 已复现。比较提升前须对齐预训练、数据、batch/曝光、学习率、坐标、模型选择和后处理，不能用较低参考数值替代约69/65的既有锚点，也不能把约69/65规定为每次运行必须达到的分数",
        "result_policy": "没有 exact SHA、clean-tree、epoch-59 EMA、6000 successful updates、官方 evaluator 和结构化自哈希 receipt，不得报告为最终科学结果；不得用 scheduler steps 替代 optimizer successful updates。跨SHA续训必须同时记录原SHA/恢复点更新数与新SHA，不得包装成单一新SHA从零训练的结果",
        "excluded_remote_jobs": [
            {"job_ids": "83ceb5f4真实checkpoint接线诊断，无Slurm job", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_storage_resume_83ceb5f4", "source_head": "83ceb5f4317225f5b9c2acc1bb967e76c48bdd4e", "reason": "CPU读取原A1 checkpoint后，重建runtime配置时遗漏update_workdir保留的末尾分隔符，被runtime_config_sha256严格拒绝。未训练、未改原checkpoint；ce767b4d修正并通过最终Linux53tests及全部六个真实checkpoint绑定，GPU恢复验证新job1276876排队"},
            {"job_ids": "BAFDR G96第一次提交失败，无job id", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6", "source_head": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "reason": "sbatch --dependency=afterok:1276618返回Job dependency problem；scontrol证实该完成job已不在控制器，sacct仍为COMPLETED(0:0)，日志有BAFDR_SUCCESSFUL_UPDATE_PRECHECKS_OK。核验已满足前置验证后删除过期dependency参数，未修改模型或放宽门禁，成功提交新job1276675"},
            {"job_ids": "1274924_2/1274924_7/1275675_1/1275675_4/1275675_5/1275675_6", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725 + duca_evidence_remaining_73bdd34a", "source_head": "1570a725 / 73bdd34a", "reason": "六个正式训练在2026-09-07 16:49-16:56因Disk quota exceeded无法写training_audit临时文件而FAILED；六对stdout/stderr已读，无terminal epoch59。旧checkpoint可加载，不能称最终性能。其死依赖评测均已取消；1276644/1276645存储恢复PRECHECK已通过；续训接线ce767b4d已推送，GPU恢复准入1276876排队，尚无正式续训job"},
            {"job_ids": "1276617", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_0aa72a60", "source_head": "0aa72a60c74309f1fff8142d2ab5fb71763622fd", "reason": "GPU admission的CUDA witness未初始化分布式进程组而调用日志reduce_loss，48 passed/1 failed；未执行后续G0-G3真实PRECHECK。已读stdout/stderr，78cde1be只修复测试夹具，保留生产机制并以1276643重跑"},
            {"job_ids": "1267884/1268698/1269124/1269129/1269137/1269297", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/zoomtoken_bafdr_539287fa_seed4407", "source_head": "539287fa8a035765afd7e79863ce77278bef83f2", "reason": "2026-09-07读取六对stdout/stderr及terminal AdamW state：D160/G96/U16/LATE/NOKD/FULL只有5996/5997/5996/5996/5994/5995次真实更新，而scheduler与receipt均6000。自报计数及补封存未识别GradScaler skip；此前严格6000有效结论撤回。新710ce8a6修复重放及consumer校验，旧产物只读保留，不迁移为新结果"},
            {"job_ids": "2026-09-07 本轮CT-DP admission提交失败，无job id", "remote_directory": "/data/run01/sczc063/yuzibo/projects/ctdp_successful_updates_3a9f3dfe", "source_head": "3a9f3dfe753a55a7aa09119d771e5808f14e0b23", "reason": "exact-SHA CPU40tests通过后，单GPU admission sbatch返回AssocMaxSubmitJobLimit / Job violates accounting/QOS policy；stdout/stderr已完整读取。旧作业/模型不动。后继0aa72a60仅将相同验证流程保存到已有sbatch脚本，远端clean checkout与CPU40tests已准备；额度释放后提交最终SHA的CUDA+四臂真实PRECHECK，不得把当前状态写成CUDA通过"},
            {"job_ids": "2026-09-07 13:18 CST 提交失败，无 job id", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_remaining_73bdd34a", "source_head": "73bdd34ae21c6c675a00d1927b8af7c12f9edc05", "reason": "Evidence A3 官方评测 --array=4 --dependency=afterok:1275675_4 的 sbatch 返回 AssocMaxSubmitJobLimit / Job violates accounting/QOS policy；stdout/stderr 已完整读取。F 评测 1275691_1 已成功，A4/A5 未继续尝试。资源类提交失败不改模型代码，额度释放后仅补任务 4/5/6"},
            {"job_ids": "1267229-1267232", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_ctdp_c0fae67a_seed3407", "source_head": "c0fae67a1236f2c47e6c2935d217659cd1f8fb9d", "reason": "CT-DP 四臂虽 COMPLETED，但 checkpoint optimizer steps=5997/5996/5996/5997，scheduler=6000；AMP skip 未 replay 造成更新预算不合格，必须修复重训，旧数值仅为诊断 telemetry"},
            {"job_ids": "1274671_7", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_a6_repair_9739811e/slurm_logs", "source_head": "9739811e89acef380806a271bc224c61c66bce0f", "reason": "A6 FP32 修复后正式训练在 epoch1 batch2 再次 non-finite cost、50 replay 耗尽。a6_formal_1274671_7.out/.err 已读取；时间参数 optimizer coverage 回归复现共享缺陷，1570a725 补齐分组并处理动态 DDP，不删除旧日志或 checkpoint"},
            {"job_ids": "1274694", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_e24dd5d9/slurm_logs", "source_head": "e24dd5d9", "reason": "修复 admission 的 3 个新测试使用了缺少 ActionFormer.get_optim_groups 的最小 fixture，测试构造失败；c68eb840 改为真实 ActionFormer fixture，1274704 的 48 tests 通过"},
            {"job_ids": "1274708_2,1274708_7", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_c68eb840/slurm_logs", "source_head": "c68eb840dfb3535437c6ec63b7693e2a084c6497", "reason": "真实窗口改变时间参数使用分支，static_graph=True/reentrant checkpoint 触发 DDP training graph changed；日志已读。6d36a0d9 使用 dynamic DDP/non-reentrant checkpoint，并新增实际交替时间分支 DDP 回归"},
            {"job_ids": "1274740", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_6d36a0d9/slurm_logs", "source_head": "6d36a0d9f8392e1074b475356482cc05f693a8d4", "reason": "49 tests 中 48 PASS，新 AMP 恢复测试在随机高初始 scale 下先发生正常 overflow，未能隔离故意注入的 Inf；1570a725 固定测试 seed 和低初始 scale，仅修正测试，不改生产 AMP。最终 admission 1274755 全部 49 tests PASS"},
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
            {"job_ids": "1269375,1269377,1269379,1269381,1269385,1269387", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/h65_pro_paction_30514803", "source_head": "30514803d00bde579c9872cbcb77141aba8ecb3e", "reason": "F01、F02、F04 在 AMP skip/replay 后独立复现真实 p_action 非有限值，F03、F05、F06 随即主动停止以避免继续消耗 GPU；全部日志和 checkpoint 保留，不纳入最终性能。epoch-19 checkpoint 静态扫描有限，1269595 正在从该点隔离复现首个坏输出"},
            {"job_ids": "1269543", "remote_directory": "/data/run01/sczc063/yuzibo/diagnostics/h65_f01_epoch19_30514803", "source_head": "30514803d00bde579c9872cbcb77141aba8ecb3e", "reason": "首次只读恢复诊断错误使用 selected-axis 合同禁止的 workflow cfg override，启动器按规范拒绝；已删除禁用 override 并以新诊断任务 1269595 重提，不属于模型训练结果"},
            {"job_ids": "1269595", "remote_directory": "/data/run01/sczc063/yuzibo/diagnostics/h65_f01_epoch19_30514803", "source_head": "30514803d00bde579c9872cbcb77141aba8ecb3e", "reason": "第二次诊断关闭 formal contract 后也跳过了 checkpoint 全局 RNG 恢复，epoch-20 batch 序列与原运行不同；在发现该差异后主动停止。其有限 probe 记录只证明该不同序列未立即失败，不能用于否定原故障；1269646 已改为显式恢复 RNG"},
            {"job_ids": "1269646,1269647,1269698,1269699", "remote_directory": "/data/run01/sczc063/yuzibo/diagnostics/h65_f01_epoch19_30514803", "source_head": "30514803d00bde579c9872cbcb77141aba8ecb3e", "reason": "1269646 暴露诊断 wrapper 符号引用错误；修正后的 1269647 精确恢复 RNG 并在 probe call 69 复现全 NaN。1269698/1269699 进一步定位首个异常为 relative_physical_time_scale 的单个非有限梯度：该参数被 train() 重新启用却不在 optimizer/GradScaler 内，随后全局 clip 把 NaN 扩散到所有优化参数。诊断结果导向 f2068e18 的最小 optimizer-group 修复，诊断作业本身不属于性能结果"},
            {"job_ids": "1269541", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_eval_29b5a7a2", "source_head": "29b5a7a2b291203ea7b697cfe416b64f0d365d02（评测） / 539287fa8a035765afd7e79863ce77278bef83f2（训练）", "reason": "BAFDR 五臂评测已生成 G96/U16/LATE/NOKD 四份 prediction receipt，FULL 在读取视频尾部时因 Decord 达到 DECORD_EOF_RETRY_MAX=10240 失败；四份收据保留，未打开指标。定向 FULL 重试改用 20480 并保留同一模型、数据、checkpoint 和评测 SHA"},
            {"job_ids": "1269763", "remote_directory": "/data/run01/sczc063/yuzibo/projects/zoomtoken_bafdr_eval_29b5a7a2", "source_head": "29b5a7a2b291203ea7b697cfe416b64f0d365d02（评测） / 539287fa8a035765afd7e79863ce77278bef83f2（训练）", "reason": "第一次 FULL 定向重试的 wrapper 未把训练收据中的外部 D160 Teacher config/checkpoint 显式传给 fail-closed 评测器，错误地在评测 checkout 内寻找不存在的 checkpoint，35 秒后按合同失败；1269771 已绑定既有 Teacher 路径与 SHA256 并通过对应 PRECHECK 后重提"},
            {"job_ids": "1270970-1270975", "remote_directory": "/data/run01/sczc063/yuzibo/projects/h65_pro_physical_optimizer_f2068e18", "source_head": "f2068e18e2c68bdbdd7a607b47f32d05ac3beed7", "reason": "H65 F01-F06 首轮终态评测错误复用了训练 checkout 中不含 67c8f39f 跨-checkout identity 修复的 evaluator，六作业均在推理前由 resolved_config_sha256 mismatch 严格拒绝；训练 checkpoint 完整保留。67c8f39f PRECHECK 1270985 通过后，已用训练 checkout 绝对 config 与独立 evaluator checkout 重提为 1270987-1270992"},
            {"job_ids": "1270870_7", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_normalized_246058f2", "source_head": "246058f2c24edc78818ada60eec26249bbf7d5d2", "reason": "Evidence A6 seed8261 在正式 batch 2 连续 50 次 deterministic replay 都产生 pre-AMP non-finite cost，13:51 CST FAILED(1:0)；非 OOM。9739811e 保持 H65 selection/time/merge/recovery 设计，仅将几何 decomposition 和 recovery accumulation 固定为 FP32，并增加真实 loader 三 batch PRECHECK"},
            {"job_ids": "1270870_2", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_normalized_246058f2", "source_head": "246058f2c24edc78818ada60eec26249bbf7d5d2", "reason": "Evidence A1 seed8261 在 epoch 54 batch 11 的 AMP replay 后，selector 检测到 768 个 active utility 值全部非有限并 FAILED(1:0)；完整日志和 epoch-49 checkpoint 保留。该故障独立于 A6 的 batch-2 cost 故障，必须先做 exact-RNG 恢复诊断和对应 PRECHECK，不能直接把既有 A6 修复迁移为 A1 结论"},
            {"job_ids": "1270986", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_a6_35340cad", "source_head": "35340cadff3950e8b1155e72ca987337bc3466c2", "reason": "A6 首次三 batch PRECHECK 仍继承 formal_successful_update_contract=True，与缩短的三更新预检冲突，在模型构建前退出；未训练。9739811e 仅关闭预检自身的正式结果身份，仍保留非有限 loss fail-closed 和正式 A6 完整合同"},
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
        f"基线口径纠正：[{payload['baseline_reference_note']}]({payload['baseline_reference_note']})。{payload['baseline_comparison_policy']}。",
        "",
        "## 当前实验与修正路线",
        "",
        "| 实验名称（面向外部读者） | 本地目录 | GitHub 分支与提交 | 部署状态 | 结果状态与最终结果 | 下一步 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in payload["entries"]:
        name = f"{entry['name']}（`{entry['internal_id']}`）"
        linked_sha = entry["github_commit"].rstrip("/").rsplit("/", 1)[-1]
        commit = f"{entry['branch']}<br>[`{linked_sha[:8]}`]({entry['github_commit']})<br>代码身份：`{entry['sha']}`"
        if entry.get("receipt_sealing_github"):
            commit += f"<br>补封存：[分支]({entry['receipt_sealing_github']})"
        result = f"{entry['result_status']}：{entry['final_result']}"
        deployment = entry["deployment_status"]
        if entry.get("current_cycle"):
            deployment += f"<br>本轮更新：{entry['current_cycle']}"
        if entry.get("receipt_recheck"):
            deployment += f"<br>收据复核：{entry['receipt_recheck']}"
        if entry.get("admission_progress"):
            deployment += f"<br>GPU准入：{entry['admission_progress']}"
        if entry.get("formal_job_states"):
            deployment += "<br>正式作业：" + "; ".join(f"{arm}={state}" for arm, state in entry["formal_job_states"].items())
        if entry.get("storage_recovery_precheck_status"):
            deployment += f"<br>恢复预检：{entry['storage_recovery_precheck_status']}"
        if entry.get("repair_validation"):
            validation = entry["repair_validation"]
            deployment += f"<br>修复验证：本地 {validation['local']}；远端精确干净 SHA {validation['remote_exact_clean_linux']}；CUDA/双GPU预检 {validation['cuda_and_two_gpu_precheck']}；正式重训 {validation['formal_retraining']}"
        local_paths = [entry["local_directory"], *entry.get("supporting_local_directories", [])]
        local_directory = "<br>".join(f"`{path}`" for path in local_paths)
        lines.append(f"| {name} | {local_directory} | {commit} | {deployment} | {result} | {entry['next_action']} |")
    lines += [
        "",
        "## 监督器与动态状态",
        "",
        f"远端 N16R4 监督器：`/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902`，每 60 秒轮询；本地 heartbeat 每 30 分钟刷新本表并用中文详细通知用户，即使无变化。当前远端轮询状态：`{payload['remote_supervisor'].get('status')}`，dispatcher：`{payload['remote_supervisor'].get('dispatcher_status', '未知')}`，mode：`{payload['remote_supervisor'].get('dispatcher_mode', '未知')}`。旧 dispatcher 仍是 plan/BLOCKED，不能称为正在自动续提本轮修复作业；本轮新增作业由监督任务显式提交。",
        "",
        f"集群观测（{payload['cluster_observation']['checked_at_cst']}）：本次可见可调度节点未分配 GPU {payload['cluster_observation']['public_gpu_available']}/{payload['cluster_observation']['public_gpu_total']}，本用户队列 {payload['cluster_observation']['user_jobs_in_queue']} 项；当前约束为 {payload['cluster_observation']['account_constraint']}。",
        "",
        f"指标范围：{payload['measurement_scope']}。",
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
        f"结果规则：{payload['result_policy']}。H65本协议参考臂/F01-F06、Evidence C0/A2及ET-TRC OFF/ON已有可追溯的官方终态数值，本轮收据再次复核，H65基线仍按单独纠正说明区分。BAFDR旧五臂及教师真实optimizer不足6000，旧分数仅诊断；修复版D160/G96运行，U16/LATE已补交排队，NOKD仍待额度，FULL仍待新教师。CT-DP修复版四臂运行且真实计数同步，尚无最终性能或独立评测收据。Evidence六个原正式训练因磁盘配额失败，ce767b4d续训接线已完成本地/远端CPU与真实checkpoint身份验证，GPU恢复准入1276876排队，正式续训未开始。DUCA-Unified缺失机制尚未实现。完整receipt不等于官方配方复现、模型无错或机制有效；不得从admission或中期验证推导最终mAP。",
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
