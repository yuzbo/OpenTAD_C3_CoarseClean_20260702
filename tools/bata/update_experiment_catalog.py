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
REMOTE_ROOT = "/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_recovery_20260907"
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
            timeout=45,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "UNAVAILABLE", "reason": f"ssh unavailable: {exc}"}
    if proc.returncode != 0:
        return {"status": "UNAVAILABLE", "reason": (proc.stderr or proc.stdout).strip()[-500:]}
    try:
        receipt = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"status": "INVALID_RECEIPT", "reason": proc.stdout[-500:]}
    try:
        checked_at = dt.datetime.fromisoformat(receipt["checked_at"])
        age = (dt.datetime.now(dt.timezone.utc) - checked_at).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        return {"status": "INVALID_RECEIPT", "reason": f"invalid checked_at: {exc}"}
    return {
        "status": "ACTIVE" if age <= 180 else "STALE",
        "root": REMOTE_ROOT,
        "receipt_age_seconds": int(age),
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
            "next_action": "这是冻结历史快照；后续按当前修正路线补Taylor/H65真实接线。延迟、吞吐及显存不作为强制门禁，不因历史cost占位恢复已取消的要求",
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
            "name": "H65-Pro 历史已完成六臂：旧分配策略下的归因、课程与骨干消融（四相未开启）",
            "internal_id": "H65_PRO_ACTIVE",
            "scientific_audit": "旧F01-F06全部phase=False；F02与参考臂optimizer组不同，不能作纯Taylor收益；旧REF-D768按每clip8 tubelets聚合，不是原始官方整窗TIA。新629162cd已恢复按每视频实际选后帧数聚合，384帧对应192tubelets，并新增匹配U384/phase-off/phase-on。修复不迁移旧性能，也不等于官方原始模型复现完成",
            "receipt_recheck": "2026-09-08 06:39 CST：六份F01-F06 metrics JSON的evaluation_sha256逐份复算一致，f2068e18训练/67c8f39f评测身份及六个epoch59 EMA文件存在性已复核；六份Avg-mAP不变。未重复加载已认证的大权重或重新推理，四相仍未开启，TIA及匹配控制仍待修复",
            "current_cycle": "2026-09-08 17:27:59 CST：新629162cd的CUDA51tests和三组PRECHECK由1280125通过；均匀/相位关闭/相位开启正式1280127/1280128/1280129均RUNNING。新协议每五轮全测试集选最佳EMA，允许测试集调参并披露；三组新实验单独列示，旧六臂结果保留原身份",
            "branch": "codex/h65-pro-admission-fix-20260902（参考臂） + codex/h65-pro-physical-time-optimizer-repair-20260904（F 臂）",
            "sha": "e553a5a4a1063a755900d3dfa4bf8909bf97d466（参考臂） / f2068e18e2c68bdbdd7a607b47f32d05ac3beed7（F 臂）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/f2068e18e2c68bdbdd7a607b47f32d05ac3beed7",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/h65_eval_repair"],
            "latest_repair": {"branch":"codex/h65-pro-tia-eval5-20260908","sha":"629162cdddbe490d896899a067e9803e237df467","local_directory":"E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission","remote_source":"/data/run01/sczc063/yuzibo/projects/h65_tia_eval5_629162cd","validation":"本地36 passed/12 Windows Torch skipped；远端exact-SHA clean CUDA51 passed；跨clip、不跨视频输出及梯度验证通过，三组PRECHECK通过。正式1280127/1280128/1280129运行中","status":"TIA_FIXED; MATCHED_TEST_GUIDED_CONTROLS_RUNNING"},
            "deployment_status": "远端 exact-SHA admission 1267684 已通过；REF-D768、REF-U384、REF-MNV3FC384 正式训练 1267709/1267711/1267737 与官方评测 1269303/1269360/1269373 均 COMPLETED(0:0)。30514803 的 F01/F02/F04 在 AMP replay 后独立出现 probe 全 NaN，F03/F05/F06 随即安全停止并保留 checkpoint。精确 RNG 复现 1269647、梯度审计 1269698/1269699 定位到 relative_physical_time_scale；独立修复 f2068e18 将该标量纳入零 weight-decay optimizer group，validator 强制该合同；本地 H65 tests 9 passed/11 skipped、C3 tests 23 passed，远端 exact-SHA clean checkout 59 passed/1 skipped。F01-F06 PRECHECK 全部通过，正式重训 1269705/1269707/1269759/1269761/1269764/1269769 均 COMPLETED(0:0)；六臂均有 epoch_59.pth/state_dict_ema，training audit 均确认 optimizer/scheduler/EMA/DUCA schedule 各 6000 successful updates，日志未见 non-finite/NaN/traceback/OOM。首轮评测 1270970-1270975 因错误使用训练 checkout 的旧 evaluator，全部在推理前以 resolved_config_sha256 mismatch 退出；日志已读，checkpoint 未损坏。独立 evaluator 67c8f39f 的 PRECHECK 1270985 COMPLETED(0:0)，修正后的官方评测 1270987-1270992 均 COMPLETED(0:0)；六份结构化 metrics receipt 均绑定训练 SHA f2068e18、evaluator SHA 67c8f39f、seed 3407、epoch 59 state_dict_ema、6000 successful updates，且 evaluation_sha256 自哈希复算一致",
            "result_status": "D768、U384、MNV 及 F01-F06 有本协议下官方评估器产生的终态数值与收据；这不等于未修改 AdaTAD 的复现成功或机制有效。仅 seed 3407，非完整 24 臂或多种子结论",
            "final_result": "参考臂：D768 Avg-mAP=67.58%，U384=63.89%，MNV3FC384=57.01%。H65 F01-F06 official Avg-mAP 分别为 63.3854%、64.2265%、60.3648%、60.7013%、64.0886%、63.8181%；对应 mAP@0.3/0.4/0.5/0.6/0.7：F01=79.1680/74.3484/66.5598/56.0579/40.7929，F02=79.9377/74.5653/67.3877/56.1257/43.1163，F03=75.9121/70.5691/63.0241/52.9793/39.3393，F04=75.8391/71.1599/63.8342/53.5592/39.1141，F05=79.2504/74.3064/67.1331/56.9056/42.8476，F06=79.2828/74.6584/66.6302/56.2470/42.2723。相对 U384，F02 最好为 +0.34 个百分点，F05 为 +0.20，F06 为 -0.07，F01 为 -0.50，F03 为 -3.53，F04 为 -3.19；所有 F 臂仍低于 D768",
            "next_action": "监督新匹配三组五轮全测试曲线，按用户修订保存最佳EMA、终态EMA和调参记录。原始官方1245842另补来源及独立复评，不用新U384替代官方原版；旧结果不重跑或重贴新SHA",
        },
        {
            "category": "correction_route",
            "name": "CT-DP 当前几何四臂：均匀选择、动态选择、物理网格及 CT-Tubelet（均未启用 B-AMoD）",
            "internal_id": "CT_DP_BAMOD_ACTIVE",
            "scientific_audit": "78cde1be的G0/G1确有dense GT与selected检测点错域。fe1c53db已按实际tubelet中心及插值后特征位置将GT映射到selected轴，并使proposal只逆映射一次；G2/G3显式保留dense GT，未通过开启physical head混淆消融。真实registry/head CPU回归通过，但旧训练权重不因此变正确。G0-G3全部B-AMoD关闭；M00-M11才是CTConv乘B-AMoD的机制矩阵",
            "current_cycle": "2026-09-08 16:33 CST：1278011/1278012及1279211_0/1保持COMPLETED(0:0)，epoch59 audit仍optimizer/scheduler/EMA各6000，checkpoint stat和receipt数值未变；G0/G1 Avg-mAP64.2720%/61.6740%，mAP@0.7为42.6802%/39.9154%。复用14:29真实337个Adam状态和15:10 receipt认证，不重训、不重复准入或评测。上一轮Computer Use运行环境阻塞仍保留，本轮未重复重置/探测、未咨询Pro",
            "branch": "codex/duca-ctdp-successful-updates-20260907（训练） + codex/duca-ctdp-terminal-eval-20260907（独立评测）",
            "sha": "78cde1be1cb8b3acc7d750afc92ea740f2a03d06（更新合同修复训练） / 11ced13ac6b72091d26405c5d6f152f09914c22d（独立评测） / c0fae67a1236f2c47e6c2935d217659cd1f8fb9d（旧训练）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/78cde1be1cb8b3acc7d750afc92ea740f2a03d06",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_successful_updates",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_coordinate_eval", "E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_terminal_eval", "E:/DeskTop/TAD/OpenTAD_CTDP_FormalRepair_20260903"],
            "latest_repair": {"branch":"codex/duca-ctdp-coordinate-repair-20260908","sha":"fe1c53db1b2a7e467a6af2eccfe8b7636f667a81","local_directory":"E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_successful_updates","remote_source":"/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_fe1c53db","validation":"本地32 passed/1 CUDA skipped；最终SHA远端clean CPU52 passed/1 CUDA skipped。1278007的71 tests/CUDA及四臂真实PRECHECK已通过，未重复。2026-09-08 14:29两份正式epoch59 EMA通过现有checkpoint_counts，各337个Adam参数状态step均6000，scheduler/EMA亦6000；fe1c53db远端exact/clean","status":"G0_G1_FORMAL_TRAINING_COMPLETED; TERMINAL_6000_VERIFIED","admission_job":"1278007","admission_state":"COMPLETED(0:0), ended 2026-09-08 02:51:01 CST","admission_stdout":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-coordinate-fe1c53db_1278007.out","admission_stderr":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-coordinate-fe1c53db_1278007.err","precheck_run_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck","formal_jobs":{"G0":"1278011","G1":"1278012"},"formal_job_states":{"G0":"COMPLETED(0:0); start2026-09-08 02:53:59; end14:20:05 CST; elapsed11:26:06; actual epoch59 EMA/337 Adam states6000 verified14:29","G1":"COMPLETED(0:0); start2026-09-08 02:54:29; end13:58:46 CST; elapsed11:04:17; actual epoch59 EMA/337 Adam states6000 verified14:29"},"formal_run_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal","formal_stdout":{"G0":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g0-fe1c53db_1278011.out","G1":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g1-fe1c53db_1278012.out"},"formal_stderr":{"G0":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g0-fe1c53db_1278011.err","G1":"/data/run01/sczc063/yuzibo/slurm_logs/ctdp-g1-fe1c53db_1278012.err"},"launch_contract":"PRECHECK_ONLY=0; CTDP_STAGE=geometry; seed3407; one GPU per arm; exact new SHA; fresh output; 60 epochs/6000 successful updates; no resume from PRECHECK","first_log_observation":"08:01累计AMP skip均3，无replay耗尽或Traceback；epoch41最终batch loss G0=0.4476/G1=0.4852有限。stderr评测分别到161/374批，仍有既有linear1d非确定性警告，不能声称消除全部随机性","formal_update_observation":{"checked_at_cst":"2026-09-08 14:29 CST","verification":"Actual CPU checkpoint_counts, not log-only inference","G0":{"epoch":59,"successful_optimizer_updates":6000,"scheduler_updates":6000,"ema_updates":6000,"optimizer_attempts":6003,"amp_skipped_attempts":3,"optimizer_state_count":337},"G1":{"epoch":59,"successful_optimizer_updates":6000,"scheduler_updates":6000,"ema_updates":6000,"optimizer_attempts":6003,"amp_skipped_attempts":3,"optimizer_state_count":337}}},
            "coordinate_evaluator": {"branch":"codex/duca-ctdp-coordinate-terminal-eval-20260908","sha":"073832744dd171dee4119dc29d1bb12a057e84f4","training_sha":"fe1c53db1b2a7e467a6af2eccfe8b7636f667a81","github_commit":"https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/073832744dd171dee4119dc29d1bb12a057e84f4","local_directory":"E:/DeskTop/TAD/_duca_fix_worktrees/ctdp_coordinate_eval","remote_source":"/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_eval_07383274","local_validation":"45 passed, 1 CUDA skipped; coordinate collection blocked by Windows Torch c10.dll; py_compile and bash -n PASS","remote_validation":"07383274 exact-SHA clean Linux 50 passed/1 CUDA skipped, including real coordinate regression; py_compile PASS; opentad/configs diff from fe1c53db empty; completed 2026-09-08 08:09 CST","status":"G0_G1_FORMAL_EVALUATION_COMPLETED; TERMINAL_RECEIPTS_VERIFIED","precheck_train_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck","precheck_eval_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/precheck","formal_train_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal","formal_eval_root":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/formal","scope":"No model/config changes from fe1c53db; rejects old78 checkpoints. Existing1278042 admission and14:29 actual terminal6000 validation reused. New G0/G1 independent official results only; no B-AMoD or original AdaTAD reproduction claim.","precheck_job":"1278042","precheck_state":"COMPLETED(0:0), started 2026-09-08 08:11:29, ended 08:14:51 CST; sacct and all four arm markers observed 09:33 CST","precheck_stdout":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/slurm_logs/precheck_1278042.out","precheck_stderr":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/slurm_logs/precheck_1278042.err","precheck_launch":{"script":"scripts/run_ctdp_terminal_eval_n16r4.sbatch","PRECHECK_ONLY":"1","CTDP_EVAL_REPO":"/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_eval_07383274","CTDP_EVAL_COMMIT":"073832744dd171dee4119dc29d1bb12a057e84f4","CTDP_TRAIN_ROOT":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/precheck","CTDP_EVAL_ROOT":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/precheck","gres":"gpu:1"},"formal_evaluation_jobs":["1279211_0","1279211_1"],"setup_issue":"Initial synchronous SSH command timed out after checkout completed; no CPU PASS assumed. Reused completed clean checkout; separate bounded validation finished successfully, FastCtx job j-1ywacm.","precheck_observed_validation":"31 GPU tests passed; G0/G1/G2/G3 each epoch1 state_dict_ema, actual optimizer/scheduler/EMA6 and 337 optimizer states verified, followed by one-batch inference PASS; CTDP_ALL_EVAL_PRECHECKS_OK. Existing job only, not rerun this cycle.","precheck_verified_arms":["G0","G1","G2","G3"],"terminal_query_issue":"Historical 08:16-08:22 SSH queries timed out; recovered read-only access at 2026-09-08 09:32:19 CST. Existing1278042 completed normally; no Slurm resubmission or cancellation.","remote_identity_recheck":"2026-09-08 15:10 CST: training fe1c53db and evaluator07383274 exact/clean; checkpoint size/mtime unchanged. Reused14:29 actual state and model/config identity proof, no repeated admission or large checkpoint load.","formal_evaluation_states":{"G0":"COMPLETED(0:0); started2026-09-08 14:30:35; ended14:49:56 CST; elapsed00:19:21; receipt verified15:10","G1":"COMPLETED(0:0); started2026-09-08 14:30:35; ended14:50:58 CST; elapsed00:20:23; receipt verified15:10"},"formal_launch":{"submitted_at_cst":"2026-09-08 14:30:07","script":"/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_eval_07383274/scripts/run_ctdp_terminal_eval_n16r4.sbatch","array":"0,1%2","PRECHECK_ONLY":"0","CTDP_EVAL_REPO":"/data/run01/sczc063/yuzibo/projects/ctdp_coordinate_eval_07383274","CTDP_EVAL_COMMIT":"073832744dd171dee4119dc29d1bb12a057e84f4","CTDP_TRAIN_ROOT":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_fe1c53db/formal","CTDP_EVAL_ROOT":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/formal","gres":"gpu:1 per array task","dependency":"none; both completed training and existing admission validated before submission","stdout":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/slurm_logs/formal_%A_%a.out","stderr":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/slurm_logs/formal_%A_%a.err"},"expected_receipts":{"G0":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/formal/g0/gpu1_id0/official_eval/metrics.json","G1":"/data/run01/sczc063/yuzibo/experiments/ctdp_coordinate_eval_07383274/formal/g1/gpu1_id0/official_eval/metrics.json"},"result_status":"Two complete211-video official terminal receipts verified at2026-09-08 15:10 CST; G0/G1 Avg-mAP64.2720%/61.6740%.","official_terminal_results":{"training_sha":"fe1c53db1b2a7e467a6af2eccfe8b7636f667a81","evaluator_sha":"073832744dd171dee4119dc29d1bb12a057e84f4","checked_at_cst":"2026-09-08 15:10 CST","metric_scale":"fraction_0_to_1","seed":3407,"world_size":1,"global_batch":2,"checkpoint_epoch":59,"checkpoint_state_key":"state_dict_ema","successful_optimizer_updates":6000,"scheduler_updates":6000,"ema_updates":6000,"optimizer_state_count":337,"evaluated_video_count":211,"G0":{"average_mAP":0.6427200272571093,"mAP@0.3":0.8007885484368389,"mAP@0.4":0.7460867056195133,"mAP@0.5":0.6741285579333751,"mAP@0.6":0.5657943413687253,"mAP@0.7":0.42680198292709387,"receipt_sha256":"140e78dd2f0f23a977c5f55051e4c1510d04f576966ffcc9951edbc64be2d962"},"G1":{"average_mAP":0.6167397028285398,"mAP@0.3":0.7861633235376628,"mAP@0.4":0.7285757322889715,"mAP@0.5":0.6347676761532852,"mAP@0.6":0.5350382081934079,"mAP@0.7":0.39915357396937123,"receipt_sha256":"15782f03c350358f32f20639952b6676892047d2213388a43a2d00535783b7bc"},"paired_delta_G1_minus_G0_pp":{"average_mAP":-2.5980324428569457,"mAP@0.7":-2.764840895772264},"verification":"Both receipt self hashes recomputed; exact source/clean tree/arm/seed/epoch/EMA/update fields and211 unique videos checked; metrics equal production evaluation_metrics.json. Actual337 Adam states6000 verified14:29 and unchanged checkpoint stat reused.","qualification":"One seed and project-modified training protocol; dynamic selection underperforms matched uniform control here. Not proof that all dynamic methods fail, not original AdaTAD reproduction, not B-AMoD evidence; G2/G3 retained separately under78cde1be."}},
            "evaluator_github": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/11ced13ac6b72091d26405c5d6f152f09914c22d",
            "evaluator_remote_source": "/data/run01/sczc063/yuzibo/projects/ctdp_terminal_eval_11ced13a",
            "evaluator_validation": "11ced13a本地及远端clean CPU均43 passed/1 CUDA skipped。2026-09-08 02:35 sacct确认GPU单批评测PRECHECK1277767 COMPLETED(0:0)；该评测器针对旧训练身份，不能将旧G0/G1错误坐标结果认证为合法控制。当前修复需要新身份的对应验证",
            "evaluator_precheck_job": "1277767",
            "evaluator_precheck_state": "COMPLETED(0:0), sacct 2026-09-08 02:35 CST; ended 00:08:51",
            "evaluator_precheck_stdout": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/slurm_logs/precheck_1277767.out",
            "evaluator_precheck_stderr": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/slurm_logs/precheck_1277767.err",
            "evaluator_precheck_launch": {"script": "scripts/run_ctdp_terminal_eval_n16r4.sbatch", "PRECHECK_ONLY": "1", "CTDP_EVAL_REPO": "/data/run01/sczc063/yuzibo/projects/ctdp_terminal_eval_11ced13a", "CTDP_EVAL_COMMIT": "11ced13ac6b72091d26405c5d6f152f09914c22d", "CTDP_TRAIN_ROOT": "/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/precheck", "CTDP_EVAL_ROOT": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/precheck"},
            "remote_source": "/data/run01/sczc063/yuzibo/projects/ctdp_successful_updates_78cde1be",
            "repair_run_root": "/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be",
            "admission_job": "1276643",
            "admission_progress": "1276643 COMPLETED(0:0): 49 CUDA tests passed; CT_CUDA_GATE_OK; all G0-G3 real prechecks passed; CTDP_ALL_PRECHECKS_OK",
            "formal_jobs": {"G0": "1276669", "G1": "1276670", "G2": "1276671", "G3": "1276672"},
            "formal_job_states": {"G0": "COMPLETED(0:0); coordinate defect remains", "G1": "COMPLETED(0:0); coordinate defect remains", "G2": "COMPLETED(0:0); terminal update validator PASS 2026-09-08 04:32 CST", "G3": "COMPLETED(0:0); terminal update validator PASS 2026-09-08 04:32 CST"},
            "observed_successful_updates": {"G1": 6000, "G2": 6000, "G3": 6000},
            "successful_update_observation_at_cst": "2026-09-08 04:31-04:32; G1 receipt, G2/G3 actual checkpoint; old G0 terminal state not loaded this cycle",
            "new_evaluation_jobs": {"G2": "1278026_2", "G3": "1278026_3"},
            "new_evaluation_states": {"G2": "COMPLETED(0:0), ended 04:52:56 CST; receipt verified 2026-09-08 06:38 CST", "G3": "COMPLETED(0:0), ended 04:53:11 CST; receipt verified 2026-09-08 06:38 CST"},
            "official_terminal_results": {
                "training_sha": "78cde1be1cb8b3acc7d750afc92ea740f2a03d06",
                "evaluator_sha": "11ced13ac6b72091d26405c5d6f152f09914c22d",
                "checked_at_cst": "2026-09-08 06:38 CST",
                "metric_scale": "fraction_0_to_1",
                "seed": 3407,
                "checkpoint_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "successful_optimizer_updates": 6000,
                "scheduler_updates": 6000,
                "ema_updates": 6000,
                "evaluated_video_count": 211,
                "G2": {"average_mAP": 0.5652337312036566, "mAP@0.7": 0.26996504437553115, "receipt_sha256": "93b509829c11cb11073ae6390ed0be108c951fa264f02d8fb0c61f8f562028fe"},
                "G3": {"average_mAP": 0.5784886648457458, "mAP@0.7": 0.31686146269381815, "receipt_sha256": "ffcdb8a8693c655ff4643d8b247d91ba7e980eb6365b0772905cc91461be5a53"},
                "qualification": "As-implemented G2/G3 terminal results only; not new fe1c53db G0/G1 results or full mechanism acceptance",
            },
            "new_evaluation_launch": {"evaluator_sha": "11ced13ac6b72091d26405c5d6f152f09914c22d", "training_sha": "78cde1be1cb8b3acc7d750afc92ea740f2a03d06", "script": "scripts/run_ctdp_terminal_eval_n16r4.sbatch", "array": "2,3%2", "PRECHECK_ONLY": "0", "admission_satisfied_by": "1277767 COMPLETED(0:0), unchanged SHA; not repeated", "train_root": "/data/run01/sczc063/yuzibo/experiments/ctdp_successful_updates_78cde1be/formal", "eval_root": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/formal", "stdout_pattern": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/slurm_logs/formal_1278026_%a.out", "stderr_pattern": "/data/run01/sczc063/yuzibo/experiments/ctdp_terminal_eval_11ced13a/slurm_logs/formal_1278026_%a.err", "result_pattern": "g2 or g3/gpu1_id0/official_eval/metrics.json", "scope": "old G2/G3 only; not compatible evaluator admission for new fe1c53db G0/G1"},
            "deployment_status": "G0/G1/G2/G3 训练 1267229/1267230/1267231/1267232 均 COMPLETED(0:0)，四个 epoch_59.pth/EMA 均存在。但 2026-09-07 直接读取 checkpoint，实际 optimizer step 分别只有 5997/5996/5996/5997，scheduler 均为 6000。源码确认该路线未启用正式 AMP replay，跳过更新仍推进 scheduler",
            "result_status": "新fe1c53db G0/G1的6000更新训练、07383274独立官方评测及自哈希收据均完成；78cde1be G2/G3沿用原身份结果。最早更新不合格四臂及旧坐标错误G0/G1仍仅诊断，不能将新收据等同完整CT/B-AMoD机制验收",
            "final_result": "新fe1c53db/07383274：G0均匀选帧Avg-mAP=64.2720%，mAP@0.3/0.4/0.5/0.6/0.7=80.0789/74.6087/67.4129/56.5794/42.6802%；G1动态选帧=61.6740%，对应78.6163/72.8576/63.4768/53.5038/39.9154%。G1相对G0 Avg-mAP -2.5980pp、mAP@0.7 -2.7648pp。旧78cde1be/11ced13a的G2/G3仍56.5234%/57.8489%，mAP@0.7为26.9965%/31.6861%，不迁移训练SHA。单种子3407，geometry四臂均未启用B-AMoD，不外推多种子显著性、完整机制有效或原始AdaTAD性能；旧c0fae67a预算不合格指标及78cde1be坐标错误G0/G1只保留诊断",
            "next_action": "新G0/G1训练和独立评测已闭环，不重复提交。保留配对单种子负结果，下一步核对动态选帧的原设计与实际梯度/边界行为；需要Pro讨论时使用已授权ixBrowser Computer Use入口。与原始官方AdaTAD来源补证/独立复评区分，不按测试集追分改预算或挑checkpoint。G2/G3保留原训练身份，未实现机制不得扩矩阵",
        },
        {
            "category": "correction_route",
            "name": "DUCA-Unified 当前 41 单元系统因素对照：17 开发单元与 24 多种子确认单元",
            "internal_id": "DUCA_UNIFIED_ACTIVE",
            "scientific_audit": "Taylor helper同时存在真实BCT尺寸误判，以及对loss图之外的新合成tensor求梯度的问题；运行时P0/P1与H65 retention/transition仍未接通。41单元是复用控制的系统对照，不是所有因素完全正交的全因子矩阵。已实现子问题可在依赖明确后单独验证，不能给D/F/H缺失机制签发通过",
            "current_cycle": "2026-09-08 16:31心跳：本地身份由生成器实时刷新；真实Taylor P0/P1训练接线、合法one-swap及H65 retention/transition仍缺失，保持BLOCKED_UNIMPLEMENTED。无新实现或正式结果，不将实现欠账解释为资源不足，也不重复全源码审查",
            "branch": "codex/duca-unified-formal-gates-20260903",
            "sha": "793c4f9cdf7dac4f224bc73012aff8bc93949f87",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/793c4f9cdf7dac4f224bc73012aff8bc93949f87",
            "local_directory": "E:/DeskTop/TAD/OpenTAD_DUCA_Unified_FormalGates_20260903",
            "deployment_status": "提交器保持fail-closed；Taylor P0/P1与H65 retention/transition未落地的相关单元禁止提交。不重新加入已取消的延迟、吞吐、显存门槛；如声称节省计算，另需对应执行量证据",
            "result_status": "BLOCKED_UNIMPLEMENTED，无正式性能",
            "final_result": "manifest/生成器/准入规则可验证，但不能把缺失机制的占位配置当实验",
            "next_action": "先明确P0/P1的BCT布局并从已有loss祖先tensor取梯度，再按原合同接入运行时训练和H65保留/转换。各子问题按真实依赖分阶段释放，不把helper单测或配置存在当机制完成",
        },
        {
            "category": "correction_route",
            "name": "BAFDR 当前 seed 4407 正式流水线：D160 教师与五臂 K16 筛选",
            "internal_id": "BAFDR_ACTIVE",
            "scientific_audit": "G96是96x96空间分辨率，全局48x16=768原始帧；K16局部256原始帧/128 tubelets。新增确认：padding未在top-K前屏蔽，可能浪费真实尾窗预算；screen finalizer仅凭声明及文件存在，0字节checkpoint和错误hash也能PASS，须复用实际终态验证。连续gate在非零残差路径下可回传router，不能笼统说整个router断梯度。少3-6更新不能单独解释十余个百分点差距",
            "current_cycle": "2026-09-08 16:33 CST：U16/LATE/NOKD及FULL PRECHECK仍COMPLETED(0:0)，前三终轮stdout仍保存epoch59/update6000，stderr未见新fatal；复用04:28真实权重验证。710ce8a6新metrics仍0份，FULL无正式训练。padding/screen仍未修复，不扩已知缺陷旧SHA矩阵；历史1267920/1267921未操作",
            "branch": "codex/zoomtoken-bafdr-successful-updates-20260907（修复） + codex/zoomtoken-bafdr-eval-metadata-repair-20260904（旧评测）",
            "sha": "710ce8a6246c471742c83bf7c180d9ab87c36fac（更新合同修复） / 539287fa8a035765afd7e79863ce77278bef83f2（旧训练） / 29b5a7a2b291203ea7b697cfe416b64f0d365d02（旧评测）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/710ce8a6246c471742c83bf7c180d9ab87c36fac",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_successful_updates",
            "deployment_status": "历史539287fa更新不足仍保留。710ce8a6新Teacher/G96此前已核验，U16/LATE/NOKD于04:28实际终态6000验证通过；本轮复查作业与日志，无新独立性能。FULL只有PRECHECK，无正式训练。padding/screen缺陷尚未修复，不继续自动扩展有缺陷代码",
            "result_status": "旧五臂及旧D160仍PROTOCOL_ERROR；新710ce8a6教师/G96/U16/LATE/NOKD实际6000已验证，但没有新的独立性能。padding与screen缺陷未修复，训练合同通过不能覆盖机制问题。以下旧分数仅供诊断",
            "final_result": "保留的协议不合格诊断数值：G96官方评估器Avg-mAP=50.93%，各阈值64.67/59.83/53.47/44.68/32.00%；U16均匀分块48.17%，63.50/57.95/50.12/40.46/28.84%；晚期融合LATE53.11%，68.84/63.39/56.05/45.40/31.88%；无蒸馏NOKD49.44%，64.86/58.81/51.60/41.85/30.09%；完整FULL52.38%，67.63/62.71/54.60/45.14/31.83%。五份旧receipt绑定seed4407/evaluator29b5a7a2，但其中自报6000 successful updates已被真实AdamW state证伪。FULL绑定539287fa的D160教师也只有5996更新。不得据此宣称满足严格预算、模型优劣或多种子显著性；原始分数和checkpoint不改写",
            "next_action": "先修screen对真实terminal checkpoint/配置/teacher/更新状态的验证，以及top-K前padding掩码与valid<K的局部编码。保留已完成及运行作业，不重复Teacher/G96；旧710ce8a6的FULL PRECHECK不自动授权含缺陷版本继续扩展。修复后新SHA做focused/CUDA/PRECHECK，再按受影响范围安排配对训练和五臂独立评测，不直接放开21单元",
            "evaluation_preparation": {"checked_at_cst": "2026-09-07 22:55心跳", "already_correct": "710ce8a6五臂val/test masks-only和Collect snippet_stride/offset_frames已包含29b5a7a2修复，无需再次移植；eval-only不构造teacher，从checkpoint保留teacher_identity", "remaining": "scripts/run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh的metrics模式循环21单元；tools/bata/bafdr_k16_fullmatrix.py:371的seal_prediction_receipts也固定全部21单元。当前仅五臂，需最小五臂封存/开启入口、独立输出目录和训练/评测双SHA收据接线；本轮只读定位，尚未实施或做新GPU验证", "existing_entry_points": ["tools/bata/finalize_bafdr_screen_gate.py:41", "tools/bata/bafdr_k16_fullmatrix_train.py:894", "tools/bata/bafdr_k16_fullmatrix.py:371"]},
            "receipt_sealing_branch": "codex/zoomtoken-bafdr-receipt-seal-20260907",
            "receipt_sealing_github": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/codex/zoomtoken-bafdr-receipt-seal-20260907",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_admission", "E:/DeskTop/TAD/_duca_fix_worktrees/bafdr_receipt_seal"],
            "remote_repair_source": "/data/run01/sczc063/yuzibo/projects/bafdr_successful_updates_710ce8a6",
            "repair_run_root": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6",
            "observed_optimizer_steps": {"D160": 5996, "G96": 5997, "U16": 5996, "LATE": 5996, "NOKD": 5994, "FULL": 5995},
            "repair_validation": {"commit": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "local": "41 passed, 1 Windows Torch skipped", "remote_exact_clean_linux": "44 passed, 1 CUDA skipped", "cuda_and_two_gpu_precheck": "1276618 COMPLETED(0:0): 45 tests passed; D160/G96/U16/LATE/NOKD各两轮三批，BAFDR_SUCCESSFUL_UPDATE_PRECHECKS_OK。该历史验证不覆盖本轮padding/screen发现", "formal_retraining": "2026-09-08 04:28：D160/G96不重复加载；U16/LATE/NOKD三份真实epoch59 EMA通过现有terminal validator，optimizer/scheduler/EMA各6000。FULL PRECHECK1277955通过，不代替padding/screen修复，无新正式FULL"},
            "full_precheck_launch": {"script": "scripts/run_bafdr_successful_update_admission_n16r4.sbatch", "PROJECT_DIR": "/data/run01/sczc063/yuzibo/projects/bafdr_successful_updates_710ce8a6", "BAFDR_EXPECTED_COMMIT": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "ZOOMTOKEN_RUN_ROOT": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6", "BAFDR_ADMISSION_FULL_ONLY": "1", "BAFDR_TEACHER_COMMIT": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "BAFDR_TEACHER_CONFIG": "/data/run01/sczc063/yuzibo/projects/bafdr_successful_updates_710ce8a6/configs/adatad/thumos/bafdr_k16_d160_seed4407.py", "BAFDR_TEACHER_CHECKPOINT": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6/work_dirs/bafdr_k16_d160_seed4407/checkpoint/epoch_59.pth", "required_binding": "提交前以现有teacher identity合同计算并显式导出BAFDR_TEACHER_CONFIG_SHA256与BAFDR_TEACHER_CHECKPOINT_SHA256；该检查会拒绝错误Teacher，不生成无人消费的hash。2GPU，仅FULL两轮三批；通过后才提交FULL正式训练。不能直接运行全7臂DAG或重新提交已完成Teacher/G96"},
            "admission_job": "1276618",
            "teacher_training_job": "1276650",
            "g96_training_job": "1276675",
            "u16_training_job": "1276842",
            "late_training_job": "1276843",
            "nokd_training_job": "1277954",
            "observed_successful_updates": {"D160": 6000, "G96": 6000, "U16": 6000, "LATE": 6000, "NOKD": 6000},
            "successful_update_observation_at_cst": "2026-09-08 04:28; U16/LATE/NOKD actual terminal state; D160/G96 reuse prior verified terminal evidence",
        },
        {
            "category": "correction_route",
            "name": "ET-TRC 当前双臂：完整 Transformer 与局部 Taylor/JVP 近似",
            "internal_id": "ET_TRC_ACTIVE",
            "scientific_audit": "仍为fixed-stride anchors加共享rank64可学习代理，不是精确anchor-dependent Jacobian或真实event gate。旧版本anchor correction污染已由9a346f0d屏蔽修复，非中心tap、跨视频和梯度回归通过；TIA可合法跨时混合。新OFF/ON实际2GPU、global batch2、FP32，已真实预检和运行；当前没有新性能，不声称70%节省或无损",
            "receipt_recheck": "2026-09-08 06:39 CST：OFF/ON metrics JSON可读且receipt_sha256复算一致，原74473c27训练身份和epoch59 EMA文件存在性复核通过，数值仍62.0768%/54.8096%。未重新加载已认证大权重或重跑推理；anchor correction及实际精度问题仍待修复/核对",
            "branch": "codex/zoomtoken-et-trc-formal-repair-20260903（训练） + codex/zoomtoken-ettrc-terminal-eval-20260907（独立评测）",
            "sha": "74473c2775caebf0da9d368ce8009d78e2942098（训练） / 67d7079d0d1c33e129d31cd3a45aaf93a67db252（评测）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/67d7079d0d1c33e129d31cd3a45aaf93a67db252",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval",
            "supporting_local_directories": ["E:/DeskTop/TAD/zoomtoken_ettrc_correction_20260902"],
            "latest_repair": {"branch":"codex/zoomtoken-ettrc-anchor-eval5-20260908","sha":"9a346f0d71e8870ad499bb0a58b9f1824b8904c0","local_directory":"E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval","remote_source":"/data/run01/sczc063/yuzibo/projects/ettrc_anchor_eval5_9a346f0d","validation":"本地31 passed/1 Windows Torch skipped；远端exact-SHA clean37 passed；真实双GPU训练/推理PRECHECK1280077/1280078通过。正式ON1280117/OFF1280118运行中","status":"ANCHOR_CORRECTION_FIXED; TEST_GUIDED_OFF_ON_RUNNING"},
            "current_cycle": "2026-09-08 17:27:59 CST：新9a346f0d远端37tests、真实2GPU预检1280077/1280078均通过；ON1280117、OFF1280118已RUNNING至epoch1 batch50。测试加载器为211视频/792窗口，protocol.json已写入测试集选模/调参标记，首轮全测试集分数尚未生成",
            "deployment_status": "OFF1267218/ON1267219正式训练及epoch59 EMA均完成；独立评测代码67d7079d、本地/远端28 tests、2GPU PRECHECK1275569_0/1通过，完整官方评测1275624_0/1和结构化自哈希receipt现已完成",
            "result_status": "双臂具有可追溯的官方终态结果；本种子开启Taylor/JVP近似明显退化，不支持保持准确率的主张；算子实现忠实度和训练配方的因果排查尚未完成",
            "final_result": "OFF official Avg-mAP=62.076775%，mAP@0.3/0.4/0.5/0.6/0.7=77.063201/71.910513/65.342222/54.513056/41.554882%；ON official Avg-mAP=54.809580%，对应70.378016/64.889287/57.494778/46.651646/34.634174%。ON比OFF低7.267195个百分点，mAP@0.7低6.920708个百分点。单种子4407，不外推多种子显著性，也不把低分本身当作已证明实现错误",
            "next_action": "监督9a346f0d新OFF/ON训练，每五轮读取完整测试分数、最佳EMA与单独终态EMA，记录后续结果驱动调参。保留旧负结果；fixed-stride代理仍不等同事件触发或精确JVP，不重复已通过37tests和双GPU预检",
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
            "scientific_audit": "C0是历史H65选择控制而非均匀384，A2是NO_TIME而非FULL。utility MLP通过hard top-K/.item选择且未进入已实现loss，3步无梯度；robust/cycle名称没有对应真实two-view/cycle路径。DenseTemporalRecovery的gate与pwconv双零导致永久死分支，e0c88c9a已修复新模型初始化并通过真实3步学习测试；加载旧双零checkpoint仍会覆盖新初始化，不能当修复版续训",
            "branch": "codex/duca-evidence-storage-resume-20260907（续训修复） + codex/duca-evidence-fullgrid-repair-20260904（C0/A2） + codex/duca-evidence-optimizer-repair-20260907与codex/duca-evidence-remaining-arms-20260907（原训练）",
            "sha": "ce767b4df49a82b42f4fab4897ac4ea562e4b548（续训接线） / 246058f2c24edc78818ada60eec26249bbf7d5d2（C0/A2） / 1570a72507491899a50767700a35a04eee3f5fe9（A1/A6恢复源） / 73bdd34ae21c6c675a00d1927b8af7c12f9edc05（F/A3-A5恢复源）",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/ce767b4df49a82b42f4fab4897ac4ea562e4b548",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_storage_resume",
            "supporting_local_directories": ["E:/DeskTop/TAD/_duca_fix_worktrees/evidence_remaining_arms", "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_optimizer_repair", "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_eval_repair", "E:/DeskTop/TAD/OpenTAD_Evidence_FormalRepair_20260903"],
            "latest_repair": {"branch": "codex/duca-evidence-refinement-gradient-repair-20260908", "sha": "e0c88c9a9e2582878c88bb2c728891ea5413bc77", "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/evidence_storage_resume", "remote_source": "/data/run01/sczc063/yuzibo/projects/evidence_refinement_e0c88c9a", "validation": "本地C3 23 passed；远端exact-SHA clean CPU25 passed，含初始输出等价、真实AdamW三步门控/卷积更新、旧checkpoint加载不被隐式重置。未做新模型CUDA或正式训练", "status": "REFINER_INIT_FIXED; UTILITY_AND_ROBUST_CYCLE_PENDING"},
            "current_cycle": "2026-09-08 16:33 CST：A1/A6/F训练及A1评测仍COMPLETED(0:0)，日志终段无新增fatal；ce767b4d仍仅有A1一份metrics，metrics.average_mAP=0.5159681679375281且mtime未变，A6/F独立评测仍缺失。复用04:30终态/恢复认证，utility/robust/cycle未接通，不以初始化局部修复或旧权重充当完整机制",
            "receipt_recheck": "2026-09-07 22:55心跳：C0/A2两份metrics JSON可读且evaluation_sha256复算通过，仍分别59.2292%/54.2756%；seed8261、246058f2和epoch59 EMA身份未变，不属于完整Recovery结果",
            "deployment_status": "C0训练1269374_0/评测1270672_0、A2训练1270870_3/评测1274711_3均已完成，保留246058f2原身份。旧A1/A6非有限cost修复经过1570a725真实optimizer覆盖、动态DDP、non-reentrant checkpoint及49tests/真实PRECHECK验证，未关闭time/merge/recovery；相关失败日志和权重均保留。其后A1/A6的1274924_2/7与F/A3/A4/A5的1275675_1/4/5/6又因磁盘配额失败，不再运行；旧死依赖eval已取消。ce767b4d增加保持原模型/配置/计数/RNG的跨目录续训，六臂恢复PRECHECK已完成，当前新提交与待办见本轮更新",
            "result_status": "C0/A2原登记值不变；新增A1旧实现的可追溯终态评测51.5968%，不作完整机制验收。A6/F训练终态已绑定但独立评测未完成。e0c88c9a仅修refiner初始化且无新正式性能；utility/robust/cycle仍未实现",
            "final_result": "C0 MATCHED_H65_60 Avg-mAP=59.23%，各tIoU=75.22/70.01/61.54/50.90/38.48%。A2 NO_TIME=54.2756%，各tIoU=72.2334/67.0169/57.6467/44.8214/29.6595%。新增旧实现A1 NO_COVERAGE=51.5968%，各tIoU=69.9676/63.9076/54.0565/42.2170/27.8353%；seed8261，1570a725 epoch39/4000次恢复到ce767b4d终态6000，独立eval1277953_2。C0是历史H65选择控制，A2关闭物理时间，A1关闭覆盖约束，均不代表缺失机制已实现。test 792次暴露对应791个唯一物理窗口（video_test_0001431|7680重复），246058f2按唯一物理窗口严格覆盖",
            "next_action": "先按原设计明确utility可学习目标/梯度路径并实现真实two-view与cycle，禁止只加无科学含义loss以制造非零梯度。refiner旧checkpoint会覆盖新初始化，后续使用新模型/新命名空间或明确登记的转换实验，不冒充无改动续训。保留旧A1/A6/F及原始receipt，核验已完成产物但不继续自动提交有缺陷的A3/A4/A5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/formal",
            "remote_log_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs",
            "remote_eval_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/evaluation",
            "original_a1_a6_run_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725/formal",
            "remaining_arms_remote_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_remaining_73bdd34a",
            "storage_recovery_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_recovery_20260907",
            "storage_recovery_precheck_jobs": {"A1": "1276644_2", "A6": "1276644_7", "F": "1276645_1", "A3": "1276645_4", "A4": "1276645_5", "A5": "1276645_6"},
            "storage_recovery_precheck_status": "1276644_2/7及1276645_1/4/5/6均于18:10-18:13 COMPLETED(0:0)，两轮各100批共200批；只验证从头运行与存储写入，不是旧checkpoint恢复测试",
            "resume_source": "/data/run01/sczc063/yuzibo/projects/duca_evidence_storage_resume_ce767b4d",
            "resume_run_root": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d",
            "admission_job": "1276876",
            "admission_progress": "1276876 COMPLETED(0:0)，80 passed；DUCA_EVIDENCE_CUDA_FP32_GATE_OK；六个DUCA_EVIDENCE_RESUME_PRECHECK_OK和ALL_RESUME_PRECHECKS_OK。恢复验证已完成，不能再写GPU未验证或重复提交该预检",
            "resume_validation": {"commit": "ce767b4df49a82b42f4fab4897ac4ea562e4b548", "local": "49 passed, 5 Windows Torch skipped", "remote_exact_clean_linux": "53 passed, 1 CUDA skipped", "real_checkpoint_bindings": "ALL_SIX_REAL_CHECKPOINT_BINDINGS_OK", "cuda_and_real_resume_precheck_job": "1276876", "cuda_and_real_resume_status": "COMPLETED(0:0): 80 tests, CUDA FP32, all six real resume prechecks PASS; historical resume validation, not utility/refiner acceptance", "formal_resume_jobs": {"A1": "1277406_2", "A6": "1277407_7", "F": "1277768_1"}, "formal_resume_job_states": {"A1": "COMPLETED(0:0); terminal binding PASS 2026-09-08 04:30 CST", "A6": "COMPLETED(0:0); terminal binding PASS 2026-09-08 04:30 CST", "F": "COMPLETED(0:0); terminal binding PASS 2026-09-08 04:30 CST"}, "admission_satisfied_by": "1276876 already COMPLETED before submission", "pending_formal_arms": ["A3", "A4", "A5"], "new_eval_jobs": {"A1": "1277953_2 COMPLETED(0:0); receipt/checkpoint/prediction/evaluator binding PASS 2026-09-08 04:30 CST"}, "new_eval_dependencies": {"A1": "afterok:1277406_2", "A6": "afterok:1277407_7", "F": "afterok:1277768_1"}, "new_eval_blocker": "A6/F independent evaluation pending; A1 is the old implemented arm, not corrected full mechanism. A3/A4/A5 blocked by known mechanism defects, not only resources"},
            "observed_successful_updates": {"A1": 6000, "A6": 6000, "F": 6000},
            "successful_update_observation_at_cst": "2026-09-08 04:30-04:31; actual nonconditional optimizer states and scheduler/EMA; conditional conditioner exposure separately recorded",
            "conditional_time_parameter_updates": {"A1": 794, "A6": 1600, "F": 794},
            "resume_start_successful_updates": {"A1": 4000, "A6": 4000, "F": 3000, "A3": 3000, "A4": 3000, "A5": 3000},
            "resume_formal_stdout_pattern": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/formal_%A_%a.out",
            "resume_formal_stderr_pattern": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/formal_%A_%a.err",
            "resume_precheck_stdout": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/resume_admission_1276876.out",
            "resume_precheck_stderr": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_storage_resume_ce767b4d/slurm_logs/resume_admission_1276876.err",
            "last_persisted_updates": {"A1": 4200, "A6": 4200, "F": 3100, "A3": 3100, "A4": 3200, "A5": 3200},
            "recoverable_checkpoint_epochs": {"A1": 39, "A6": 39, "F": 29, "A3": 29, "A4": 29, "A5": 29},
        },
        {
            "category": "test_guided_exploratory",
            "name": "Transformer 低秩残差近似开启：修复锚点后重训",
            "internal_id": "ETTRC_EVAL5_ON",
            "branch": "codex/zoomtoken-ettrc-anchor-eval5-20260908",
            "sha": "9a346f0d71e8870ad499bb0a58b9f1824b8904c0",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9a346f0d71e8870ad499bb0a58b9f1824b8904c0",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval",
            "config": "configs/adatad/thumos/ettrc_test_guided_on_seed4407.py",
            "seed": 4407,
            "job_id": "1280117",
            "protocol": "TEST_GUIDED_EXPLORATORY_EVAL5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/on/gpu2_id0",
            "remote_stdout": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/on_1280117.out",
            "remote_stderr": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/on_1280117.err",
            "deployment_status": "2026-09-08 17:27:59 CST：1280117 RUNNING；2 GPU，完整60轮训练。37项远端tests及1280077/1280078真实2GPU PRECHECK通过",
            "result_status": "新实验尚无全测试集结果",
            "final_result": "未到首次第5轮全测试评测；不把初期loss或旧SHA成绩作为新性能。选择最佳EMA并允许测试集调参，终态epoch59 EMA/6000实际更新另列，不称未见测试集泛化",
            "next_action": "每5轮保存完整官方测试曲线与当前最佳EMA，后续按测试结果调整超参数并留存全部尝试；每30分钟汇总。失败先读日志、独立修复并新身份重提"
        },
        {
            "category": "test_guided_exploratory",
            "name": "完整 Transformer 对照：相同双 GPU 与 FP32 训练",
            "internal_id": "ETTRC_EVAL5_OFF",
            "branch": "codex/zoomtoken-ettrc-anchor-eval5-20260908",
            "sha": "9a346f0d71e8870ad499bb0a58b9f1824b8904c0",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/9a346f0d71e8870ad499bb0a58b9f1824b8904c0",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/ettrc_terminal_eval",
            "config": "configs/adatad/thumos/ettrc_test_guided_off_seed4407.py",
            "seed": 4407,
            "job_id": "1280118",
            "protocol": "TEST_GUIDED_EXPLORATORY_EVAL5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/off/gpu2_id0",
            "remote_stdout": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/off_1280118.out",
            "remote_stderr": "/data/run01/sczc063/yuzibo/experiments/ettrc_anchor_eval5_9a346f0d_seed4407/formal/logs/off_1280118.err",
            "deployment_status": "2026-09-08 17:27:59 CST：1280118 RUNNING；2 GPU，完整60轮训练。37项远端tests及1280077/1280078真实2GPU PRECHECK通过",
            "result_status": "新实验尚无全测试集结果",
            "final_result": "未到首次第5轮全测试评测；不把初期loss或旧SHA成绩作为新性能。选择最佳EMA并允许测试集调参，终态epoch59 EMA/6000实际更新另列，不称未见测试集泛化",
            "next_action": "每5轮保存完整官方测试曲线与当前最佳EMA，后续按测试结果调整超参数并留存全部尝试；每30分钟汇总。失败先读日志、独立修复并新身份重提"
        },
        {
            "category": "test_guided_exploratory",
            "name": "均匀抽取 384 帧：整窗时间适配器匹配对照",
            "internal_id": "H65_EVAL5_UNIFORM",
            "branch": "codex/h65-pro-tia-eval5-20260908",
            "sha": "629162cdddbe490d896899a067e9803e237df467",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "config": "configs/adatad/thumos/h65_pro/h65_pro_eval5_uniform.py",
            "seed": 3407,
            "job_id": "1280127",
            "protocol": "TEST_GUIDED_EXPLORATORY_EVAL5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-UNIFORM_seed3407/gpu1_id0",
            "remote_stdout": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/uniform_1280127.out",
            "remote_stderr": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/uniform_1280127.err",
            "deployment_status": "2026-09-08 17:27:59 CST：1280127 RUNNING；1 GPU，完整60轮训练。1280125的CUDA51tests和三组PRECHECK通过",
            "result_status": "新实验尚无全测试集结果",
            "final_result": "未到首次第5轮全测试评测；不把初期loss或旧SHA成绩作为新性能。选择最佳EMA并允许测试集调参，终态epoch59 EMA/6000实际更新另列，不称未见测试集泛化",
            "next_action": "每5轮保存完整官方测试曲线与当前最佳EMA，后续按测试结果调整超参数并留存全部尝试；每30分钟汇总。失败先读日志、独立修复并新身份重提"
        },
        {
            "category": "test_guided_exploratory",
            "name": "动态选帧但关闭四相分配：整窗时间适配器对照",
            "internal_id": "H65_EVAL5_PHASEOFF",
            "branch": "codex/h65-pro-tia-eval5-20260908",
            "sha": "629162cdddbe490d896899a067e9803e237df467",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "config": "configs/adatad/thumos/h65_pro/h65_pro_eval5_phaseoff.py",
            "seed": 3407,
            "job_id": "1280128",
            "protocol": "TEST_GUIDED_EXPLORATORY_EVAL5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEOFF_seed3407/gpu1_id0",
            "remote_stdout": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseoff_1280128.out",
            "remote_stderr": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseoff_1280128.err",
            "deployment_status": "2026-09-08 17:27:59 CST：1280128 RUNNING；1 GPU，完整60轮训练。1280125的CUDA51tests和三组PRECHECK通过",
            "result_status": "新实验尚无全测试集结果",
            "final_result": "未到首次第5轮全测试评测；不把初期loss或旧SHA成绩作为新性能。选择最佳EMA并允许测试集调参，终态epoch59 EMA/6000实际更新另列，不称未见测试集泛化",
            "next_action": "每5轮保存完整官方测试曲线与当前最佳EMA，后续按测试结果调整超参数并留存全部尝试；每30分钟汇总。失败先读日志、独立修复并新身份重提"
        },
        {
            "category": "test_guided_exploratory",
            "name": "四相预算 128+64+64+128：整窗时间适配器实验",
            "internal_id": "H65_EVAL5_PHASEON",
            "branch": "codex/h65-pro-tia-eval5-20260908",
            "sha": "629162cdddbe490d896899a067e9803e237df467",
            "github_commit": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/629162cdddbe490d896899a067e9803e237df467",
            "local_directory": "E:/DeskTop/TAD/_duca_fix_worktrees/h65_admission",
            "config": "configs/adatad/thumos/h65_pro/h65_pro_eval5_phaseon.py",
            "seed": 3407,
            "job_id": "1280129",
            "protocol": "TEST_GUIDED_EXPLORATORY_EVAL5",
            "remote_run_root": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/TEST-PHASEON_seed3407/gpu1_id0",
            "remote_stdout": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseon_1280129.out",
            "remote_stderr": "/data/run01/sczc063/yuzibo/experiments/h65_tia_eval5_629162cd_seed3407/formal/logs/phaseon_1280129.err",
            "deployment_status": "2026-09-08 17:27:59 CST：1280129 RUNNING；1 GPU，完整60轮训练。1280125的CUDA51tests和三组PRECHECK通过",
            "result_status": "新实验尚无全测试集结果",
            "final_result": "未到首次第5轮全测试评测；不把初期loss或旧SHA成绩作为新性能。选择最佳EMA并允许测试集调参，终态epoch59 EMA/6000实际更新另列，不称未见测试集泛化",
            "next_action": "每5轮保存完整官方测试曲线与当前最佳EMA，后续按测试结果调整超参数并留存全部尝试；每30分钟汇总。失败先读日志、独立修复并新身份重提"
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
        "remote_supervisor_recovery": {
            "old_root": "/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902",
            "old_last_checked_at_utc": "2026-09-07T08:48:01+00:00",
            "failure_log": "/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902/logs/supervisor.log",
            "failure": "OSError [Errno 122] Disk quota exceeded while writing supervisor_state.json.tmp; old process exited",
            "root": REMOTE_ROOT,
            "pid": 1914589,
            "restart_at_utc": "2026-09-07T13:32:31+00:00",
            "precheck": "unchanged supervisor --once passed; squeue/sacct returncode0; dispatcher BLOCKED; entries0",
            "scope": "每60秒恢复只读队列轮询，旧source/queue/manifest/admission不变；旧日志和状态保留。不是当前修复队列的自动提交器。目录生成器现以180秒识别过期回执，不能仅因JSON可读就称ACTIVE",
        },
        "cluster_observation": {
            "checked_at_cst": "2026-09-08 17:33:38 CST",
            "public_gpu_available": 1,
            "public_gpu_total": 200,
            "gpu_count_basis": "排除DOWN/DRAIN/FAIL/MAINT/RESERVED/NOT_RESPONDING节点后，CfgTRES200、AllocTRES199；公共未分配1不等于本账户新增配额",
            "user_jobs_in_queue": 11,
            "account_constraint": "9 RUNNING/2 PENDING；本任务新5项正式训练均RUNNING，其余任务不纳入六路线或取消。磁盘剩余168.4GiB。H65已进入epoch1；五份真实protocol.json自哈希、精确SHA、seed、211视频/792窗口及五轮评测标记均核对通过",
        },
        "measurement_scope": "当前比较以官方 TAD mAP、固定输入/更新预算和机制消融为主；不把端到端延迟、吞吐量或显存作为强制验收指标",
        "baseline_reference_note": "12_BASELINE_IDENTITY_CORRECTION_20260907.md",
        "pro_audit_followup": "15_PRO_AUDIT_REPAIR_STATUS_20260908.md",
        "official_baseline_raw_identity": "14_OFFICIAL_BASELINE_RAW_IDENTITY_20260908.json",
        "performance_cause_audit": "13_PERFORMANCE_CAUSE_AUDIT_20260908.md",
        "heartbeat_evidence": "29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.json",
        "performance_cause_audit_scope": "2026-09-08用户交互任务：落实每5轮全测试集评测、最佳EMA选择和测试集调参授权；完成H65整窗TIA与ET anchor修复、两份推送SHA、真实远端验证及五项正式部署。17:27:59实时sacct全部RUNNING，未到新完整测试分数。其他四路线未实施新修复，不将旧结果重验或虚报为六路线全完成",
        "baseline_comparison_policy": "区分上游公开结果、历史共享复现和本次修改协议实测；官方 evaluator 与完整 receipt 不代表官方训练 recipe 已复现。比较提升前须对齐预训练、数据、batch/曝光、学习率、坐标、模型选择和后处理，不能用较低参考数值替代约69/65的既有锚点，也不能把约69/65规定为每次运行必须达到的分数",
        "result_policy": "历史strict6000结果仍需exact SHA、clean-tree、epoch59 EMA、实际6000次optimizer/scheduler/EMA更新、官方evaluator及自哈希receipt。新TEST_GUIDED_EXPLORATORY_EVAL5每5轮完整测试、按未四舍五入Avg-mAP选最佳EMA且允许测试集调参；所有尝试和曲线保留，终态EMA另列。必须披露测试集重复使用，不称独立泛化或与官方论文公平可比。推理不输入测试GT，原始官方配方与历史封闭实验不追溯改写，收据不替代真实机制验证",
        "test_guided_policy": "2026-09-08用户授权每5 epoch完整测试、最佳EMA和测试集调参；新实验单列，终态另报，不声称未见测试泛化。详见29_TEST_GUIDED_EVAL5_DEPLOYMENT_20260908.md",
        "excluded_remote_jobs": [
            {"job_ids": "远端监督器PID2272497退出，无Slurm job", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902", "source_head": "0ef4d81f（原安装记录）", "reason": "latest_receipt停在2026-09-07 16:48 CST；读取logs/supervisor.log确认Errno122 Disk quota exceeded导致退出。此前目录只因JSON存在标ACTIVE的判断错误已修复。保持原脚本和全部旧产物，新root duca_multibranch_supervisor_recovery_20260907通过--once后以PID1914589恢复60秒轮询；dispatcher仍plan/BLOCKED，不宣称自动提交当前修复队列"},
            {"job_ids": "CT-DP evaluator同步两次失败，无Slurm job", "remote_directory": "/data/run01/sczc063/yuzibo/projects/ctdp_successful_updates_78cde1be", "source_head": "11ced13ac6b72091d26405c5d6f152f09914c22d", "reason": "第一次fetch origin失败：origin实际为旧single_seed_unified.bundle，无新分支；改用GitHub地址后代理Empty reply。两次完整日志保留在本地FastCtx j-c7aun1/j-pi13p4。改用增量Git bundle传输，同一11ced13a已建立独立clean checkout并通过Linux43tests/1CUDA skipped，未热改训练目录或修改模型"},
            {"job_ids": "83ceb5f4真实checkpoint接线诊断，无Slurm job", "remote_directory": "/data/run01/sczc063/yuzibo/projects/duca_evidence_storage_resume_83ceb5f4", "source_head": "83ceb5f4317225f5b9c2acc1bb967e76c48bdd4e", "reason": "CPU读取原A1 checkpoint后，重建runtime配置时遗漏update_workdir保留的末尾分隔符，被runtime_config_sha256严格拒绝。未训练、未改原checkpoint；ce767b4d修正并通过最终Linux53tests、六个真实checkpoint绑定及GPU恢复验证1276876，A1/A6正式续训1277406_2/1277407_7已提交"},
            {"job_ids": "BAFDR G96第一次提交失败，无job id", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/bafdr_successful_updates_710ce8a6", "source_head": "710ce8a6246c471742c83bf7c180d9ab87c36fac", "reason": "sbatch --dependency=afterok:1276618返回Job dependency problem；scontrol证实该完成job已不在控制器，sacct仍为COMPLETED(0:0)，日志有BAFDR_SUCCESSFUL_UPDATE_PRECHECKS_OK。核验已满足前置验证后删除过期dependency参数，未修改模型或放宽门禁，成功提交新job1276675"},
            {"job_ids": "1274924_2/1274924_7/1275675_1/1275675_4/1275675_5/1275675_6", "remote_directory": "/data/run01/sczc063/yuzibo/experiments/duca_evidence_optimizer_1570a725 + duca_evidence_remaining_73bdd34a", "source_head": "1570a725 / 73bdd34a", "reason": "六个正式训练在2026-09-07 16:49-16:56因Disk quota exceeded无法写training_audit临时文件而FAILED；六对stdout/stderr已读，无terminal epoch59。旧checkpoint可加载并保留，死依赖评测已取消。存储PRECHECK1276644/1276645及真实恢复PRECHECK1276876全部通过；A1/A6/F以ce767b4d和新目录提交1277406_2/1277407_7/1277768_1，22:17 A1已从原checkpoint恢复训练，A3/A4/A5仍待额度"},
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
        "本表用完整中文描述实验目的；括号中的内部 ID 仅用于与 Slurm/manifest 对照。前六行为9月2日冻结历史快照，当前后继修复另行列示，不把历史准入状态当最新状态。每一行都是独立代码身份，结果不能跨 SHA 转移。",
        "",
        f"基线口径纠正：[{payload['baseline_reference_note']}]({payload['baseline_reference_note']})。{payload['baseline_comparison_policy']}。",
        "",
        f"性能原因核验：[{payload['performance_cause_audit']}]({payload['performance_cause_audit']})。{payload['performance_cause_audit_scope']}。",
        "",
        f"Pro 审查与修复跟踪：[{payload['pro_audit_followup']}]({payload['pro_audit_followup']})；官方原始产物核验：[{payload['official_baseline_raw_identity']}]({payload['official_baseline_raw_identity']})。下表的原训练 SHA 与最新修复 SHA 分开列示，修复不改写历史性能。",
        "",
        f"本轮产物核验：[{payload['heartbeat_evidence']}]({payload['heartbeat_evidence']})。全局成功更新与条件模块参数的实际参与次数分别记录，不能互相替代。",
        "",
        f"新实验协议：{payload['test_guided_policy']}。",
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
        if entry.get("evaluator_github"):
            commit += f"<br>独立评测：[提交]({entry['evaluator_github']})"
        if entry.get("latest_repair"):
            repair = entry["latest_repair"]
            repair_url = f"{payload['repository']}/commit/{repair['sha']}"
            commit += f"<br>最新修复：{repair['branch']}<br>[`{repair['sha'][:8]}`]({repair_url})"
        if entry.get("coordinate_evaluator"):
            evaluator = entry["coordinate_evaluator"]
            commit += f"<br>新坐标兼容评测：{evaluator['branch']}<br>[`{evaluator['sha'][:8]}`]({evaluator['github_commit']})"
        result = f"{entry['result_status']}：{entry['final_result']}"
        if entry.get("scientific_audit"):
            result += f"<br>原因核验：{entry['scientific_audit']}"
        deployment = entry["deployment_status"]
        if entry.get("current_cycle"):
            deployment += f"<br>本轮更新：{entry['current_cycle']}"
        if entry.get("receipt_recheck"):
            deployment += f"<br>收据复核：{entry['receipt_recheck']}"
        if entry.get("admission_progress"):
            deployment += f"<br>GPU准入：{entry['admission_progress']}"
        if entry.get("evaluator_validation"):
            deployment += f"<br>评测接线：{entry['evaluator_validation']}"
        if entry.get("formal_job_states"):
            deployment += "<br>正式作业：" + "; ".join(f"{arm}={state}" for arm, state in entry["formal_job_states"].items())
        if entry.get("new_evaluation_states"):
            deployment += "<br>新增独立评测：" + "; ".join(f"{arm}={entry['new_evaluation_jobs'][arm]} {state}" for arm, state in entry["new_evaluation_states"].items())
        if entry.get("storage_recovery_precheck_status"):
            deployment += f"<br>恢复预检：{entry['storage_recovery_precheck_status']}"
        if entry.get("repair_validation"):
            validation = entry["repair_validation"]
            deployment += f"<br>修复验证：本地 {validation['local']}；远端精确干净 SHA {validation['remote_exact_clean_linux']}；CUDA/双GPU预检 {validation['cuda_and_two_gpu_precheck']}；正式重训 {validation['formal_retraining']}"
        if entry.get("latest_repair"):
            repair = entry["latest_repair"]
            deployment += f"<br>最新修复状态：{repair['status']}；{repair['validation']}"
        if entry.get("coordinate_evaluator"):
            evaluator = entry["coordinate_evaluator"]
            deployment += f"<br>新坐标评测状态：{evaluator['status']}；{evaluator['remote_validation']}"
        local_paths = [entry["local_directory"], *entry.get("supporting_local_directories", [])]
        local_directory = "<br>".join(f"`{path}`" for path in local_paths)
        lines.append(f"| {name} | {local_directory} | {commit} | {deployment} | {result} | {entry['next_action']} |")
    lines += [
        "",
        "## 监督器与动态状态",
        "",
        f"远端 N16R4 监督器恢复目录：`{REMOTE_ROOT}`，每60秒轮询；本地heartbeat每30分钟刷新本表并用中文详细通知用户，即使无变化。当前回执状态：`{payload['remote_supervisor'].get('status')}`，距生成`{payload['remote_supervisor'].get('receipt_age_seconds', '未知')}`秒；dispatcher：`{payload['remote_supervisor'].get('dispatcher_status', '未知')}`，mode：`{payload['remote_supervisor'].get('dispatcher_mode', '未知')}`。旧进程因磁盘配额退出的日志保留；新进程恢复只读轮询，超过180秒标STALE。dispatcher不自动续提当前修复队列。本轮通过监督任务提交5项新协议正式训练，见独立行及29号记录；不是分钟dispatcher自动提交。已有Computer Use初始化阻塞仍保留，本轮未咨询Pro，也未取消旧任务。",
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
        f"结果规则：{payload['result_policy']}。历史性能保留原身份；CT G2/G3的56.5234%/57.8489%为78cde1be终态结果，A1的51.5968%仍是旧实现终态评测，不是新refiner修复后性能。官方原配方68.73仍为此前核查的原始日志值，不能替代尚待补证的独立复评。各行区分训练完成、收据认证和机制验收，不从CPU反例、PRECHECK或中期训练推导最终性能。",
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
