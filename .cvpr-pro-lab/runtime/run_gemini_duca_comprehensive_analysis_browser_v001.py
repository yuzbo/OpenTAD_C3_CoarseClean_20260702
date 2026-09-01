#!/usr/bin/env python3
"""Run one read-only Gemini web consultation over the frozen DUCA evidence pack."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702")
H65_ROOT = Path(
    r"C:\Users\skywalker\.codex\worktrees\duca-full-data-identity-audit-v1-20260831"
    r"\OpenTAD_C3_CoarseClean_20260702"
)
DIAG_ROOT = Path(
    r"C:\Users\skywalker\.codex\worktrees\duca-whole-video-consistent-budget-falsifier-v1-20260831"
    r"\OpenTAD_C3_CoarseClean_20260702"
)
TRUETIME_ROOT = Path(r"E:\DeskTop\TAD\OpenTAD_DUCA_TrueTimeCurriculumV2_20260822")
PROMPT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/GEMINI_DUCA_COMPREHENSIVE_ANALYSIS-v001.md"
RUN_DIR = ROOT / ".cvpr-pro-lab/pro-reviews/runs/gemini-duca-comprehensive-analysis-browser-v001"
ORACLE_HOME = ROOT / ".cvpr-pro-lab/oracle/gemini-duca-comprehensive-analysis-browser-v001"
REPORT = RUN_DIR / "gemini-advisory.md"
RAW_LOG = RUN_DIR / "oracle-gemini.log"
MANIFEST = RUN_DIR / "manifest.json"
MODEL = "gemini-3.5-flash"
NONCE = "GEMINI-DUCA-COMPREHENSIVE-ADVISORY-v001-20260831"
COOKIE_DB = Path(
    r"C:\Users\skywalker\AppData\Roaming\ixBrowser\Browser Data"
    r"\8dd9dfe42a24409ac475ccfa90cd7654\Default\Network\Cookies"
)
LOCK_ROOT = Path.home() / ".codex/browser-broker-locks"
LOCK_IMPL = Path(r"C:\Users\skywalker\.codex\skills\cvpr-pro-lab\scripts\ix_project_sources.py")
MATERIALS = (
    PROMPT,
    ROOT / ".cvpr-pro-lab/pro-reviews/materials/DUCA_COMPREHENSIVE_ROUTE_EVIDENCE-v001.md",
    ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_COMPREHENSIVE_ROUTE_INTEGRATION-v001.md",
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001/visible-report.md",
    ROOT / "PAPER_PROGRESS.md",
    ROOT / "research-wiki/decision_history.md",
    ROOT / "research-wiki/anti_repetition.md",
    ROOT / "research-wiki/query_pack.md",
    ROOT / "research-wiki/duca_final_model_contract.md",
    ROOT / "research-wiki/experiments/duca-multi-budget-detector-adaptation.md",
    ROOT / "research-wiki/experiments/duca-native-tubelet-coreset-fixed384.md",
    ROOT / "research-wiki/experiments/phystime-g1-matched-full60.md",
    ROOT / "research-wiki/experiments/duca-sparse-probe-and-coarse-backend-ablation.md",
    ROOT / "research-wiki/sources/2026-08-31-duca-irregular-temporal-sampling-external-proposal.md",
    ROOT / "research-wiki/sources/2026-08-31-duca-full-data-comparable-protocol-v001.md",
    H65_ROOT / "opentad/models/duca/acquisition.py",
    H65_ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage1_uniform384.py",
    H65_ROOT / "configs/adatad/thumos/duca_sampling_rate_curriculum_stage2_joint384.py",
    TRUETIME_ROOT / "opentad/models/duca/true_time_residual.py",
    DIAG_ROOT / "opentad/models/duca/dynamic_budget.py",
    DIAG_ROOT / "tools/bata/run_duca_whole_video_consistent_budget_falsifier.py",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_manifest(payload: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_lock_impl():
    spec = importlib.util.spec_from_file_location("ix_project_sources", LOCK_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lock implementation: {LOCK_IMPL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("Gemini advisory already exists; refusing duplicate consultation")
    required = (*MATERIALS, COOKIE_DB, LOCK_IMPL)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Gemini inputs: {missing}")
    oracle = shutil.which("oracle")
    if not oracle:
        raise RuntimeError("oracle executable not found")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_HOME.mkdir(parents=True, exist_ok=True)
    lock_path = LOCK_ROOT / "browser-61.lock"
    manifest = {
        "status": "WAITING_FOR_EXCLUSIVE_PROFILE_LOCK",
        "model": MODEL,
        "nonce": NONCE,
        "transport": "oracle-gemini-web-cookie-read-only",
        "prompt_path": str(PROMPT),
        "materials": [str(path) for path in MATERIALS],
        "report_path": str(REPORT),
        "raw_log": str(RAW_LOG),
        "lock": str(lock_path),
        "started_at": now(),
    }
    write_manifest(manifest)

    env = os.environ.copy()
    env["ORACLE_HOME_DIR"] = str(ORACLE_HOME)
    command = [
        oracle,
        "--engine", "browser",
        "--model", MODEL,
        "--browser-cookie-path", str(COOKIE_DB),
        "--browser-attachments", "always",
        "--browser-bundle-files",
        "--browser-bundle-format", "text",
        "--timeout", "10m",
        "--heartbeat", "30",
        "--wait",
        "--verbose",
        "--slug", "gemini-duca-comprehensive-analysis-browser-v001",
        "--write-output", str(REPORT),
        "--prompt",
        (
            "Read the complete bundled evidence and code before answering. Produce one concise, evidence-grounded, "
            "read-only independent DUCA advisory for the subsequent Pro scientific review. Do not modify files or "
            "invent missing facts. Preserve nonce " + NONCE + " verbatim and end with GEMINI_DUCA_ADVISORY_READY."
        ),
    ]
    for path in MATERIALS:
        command.extend(("--file", str(path)))

    lock_impl = load_lock_impl()
    with contextlib.ExitStack() as stack:
        stack.enter_context(lock_impl.PortableFileLock(lock_path, 10800.0))
        manifest["status"] = "RUNNING"
        manifest["command_shape"] = command
        write_manifest(manifest)
        with RAW_LOG.open("w", encoding="utf-8", errors="replace") as output:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )

    report_text = REPORT.read_text(encoding="utf-8", errors="replace") if REPORT.is_file() else ""
    log_text = RAW_LOG.read_text(encoding="utf-8", errors="replace") if RAW_LOG.is_file() else ""
    terminator_bound = report_text.rstrip().endswith("GEMINI_DUCA_ADVISORY_READY")
    nonce_bound = NONCE in report_text
    model_bound = "gemini-3.5-flash" in log_text.lower()
    complete = bool(process.returncode == 0 and report_text and terminator_bound and nonce_bound and model_bound)
    manifest.update(
        status="COMPLETED" if complete else "NEEDS_ATTENTION",
        return_code=process.returncode,
        completed_at=now(),
        report_exists=REPORT.is_file(),
        report_bytes=REPORT.stat().st_size if REPORT.is_file() else 0,
        terminator_bound=terminator_bound,
        nonce_bound=nonce_bound,
        model_bound=model_bound,
    )
    write_manifest(manifest)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
