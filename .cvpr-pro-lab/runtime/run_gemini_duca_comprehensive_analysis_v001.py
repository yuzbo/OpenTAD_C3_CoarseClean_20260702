#!/usr/bin/env python3
"""Run one read-only Gemini CLI consultation and save its advisory report."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702")
PROMPT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/GEMINI_DUCA_COMPREHENSIVE_ANALYSIS-v001.md"
RUN_DIR = ROOT / ".cvpr-pro-lab/pro-reviews/runs/gemini-duca-comprehensive-analysis-v001"
REPORT = RUN_DIR / "gemini-advisory.md"
RAW_LOG = RUN_DIR / "gemini-cli.log"
MANIFEST = RUN_DIR / "manifest.json"
MODEL = "gemini-3.7-flash"
INCLUDE_DIRS = (
    Path(
        r"C:\Users\skywalker\.codex\worktrees\duca-full-data-identity-audit-v1-20260831"
        r"\OpenTAD_C3_CoarseClean_20260702"
    ),
    Path(
        r"C:\Users\skywalker\.codex\worktrees\duca-whole-video-consistent-budget-falsifier-v1-20260831"
        r"\OpenTAD_C3_CoarseClean_20260702"
    ),
    Path(r"E:\DeskTop\TAD\OpenTAD_DUCA_TrueTimeCurriculumV2_20260822"),
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def save_manifest(payload: dict) -> None:
    MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    if REPORT.exists():
        raise RuntimeError("Gemini advisory already exists; refusing duplicate consultation")
    if not PROMPT.is_file():
        raise FileNotFoundError(PROMPT)
    missing_dirs = [str(path) for path in INCLUDE_DIRS if not path.is_dir()]
    if missing_dirs:
        raise FileNotFoundError(f"missing Gemini include directories: {missing_dirs}")
    gemini = shutil.which("gemini")
    if not gemini:
        raise RuntimeError("gemini executable not found")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    prompt_text = PROMPT.read_text(encoding="utf-8")
    command = [
        gemini,
        "--model", MODEL,
        "--approval-mode", "plan",
        "--skip-trust",
        "--output-format", "text",
    ]
    for path in INCLUDE_DIRS:
        command.extend(("--include-directories", str(path)))
    command.extend(("--prompt", prompt_text))

    manifest = {
        "status": "RUNNING",
        "model": MODEL,
        "approval_mode": "plan",
        "prompt_path": str(PROMPT),
        "report_path": str(REPORT),
        "raw_log": str(RAW_LOG),
        "include_directories": [str(path) for path in INCLUDE_DIRS],
        "started_at": now(),
    }
    save_manifest(manifest)
    process = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    RAW_LOG.write_text(
        process.stdout + ("\n" if process.stdout and process.stderr else "") + process.stderr,
        encoding="utf-8",
    )
    if process.returncode == 0 and process.stdout.strip():
        REPORT.write_text(process.stdout.rstrip() + "\n", encoding="utf-8")
    report_text = REPORT.read_text(encoding="utf-8", errors="replace") if REPORT.is_file() else ""
    ready = report_text.rstrip().endswith("GEMINI_DUCA_ADVISORY_READY")
    manifest.update(
        status="COMPLETED" if process.returncode == 0 and ready else "NEEDS_ATTENTION",
        return_code=process.returncode,
        completed_at=now(),
        report_exists=REPORT.is_file(),
        report_bytes=REPORT.stat().st_size if REPORT.is_file() else 0,
        terminator_bound=ready,
    )
    save_manifest(manifest)
    return 0 if manifest["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
