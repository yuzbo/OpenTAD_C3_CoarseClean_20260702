#!/usr/bin/env python3
"""Launch one exact-DUCA Pro turn to freeze the formal full-data protocol."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request


ROOT = Path(r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702")
PROMPT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_FULL_DATA_COMPARABLE_PROTOCOL-v001.md"
PROJECT_ID = "g-p-6a91061f789881918ccd8357ca3d6c92"
PROJECT_URL = f"https://chatgpt.com/g/{PROJECT_ID}/project?tab=chats"
PROFILE_ID = 61
NONCE = "DUCA-FULL-DATA-COMPARABLE-PROTOCOL-v001-20260831"
RUN_DIR = ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-full-data-comparable-protocol-v001"
ORACLE_HOME = ROOT / ".cvpr-pro-lab/oracle/duca-full-data-comparable-protocol-v001"
LOCK_ROOT = Path.home() / ".codex/browser-broker-locks"
LOCK_IMPL = Path(r"C:\Users\skywalker\.codex\skills\cvpr-pro-lab\scripts\ix_project_sources.py")
OPENED_LIST_URL = "http://127.0.0.1:53200/api/v2/native-client-profile-opened-list"
MATERIALS = (
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-multi-budget-detector-adaptation-freeze-v001/visible-report.md",
    ROOT / "research-wiki/sources/2026-08-31-duca-full-train-official-test-human-constraint.md",
    ROOT / "research-wiki/sources/2026-08-31-duca-multi-budget-pro-freeze-v001.md",
    ROOT / "research-wiki/experiments/duca-multi-budget-detector-adaptation.md",
    ROOT / "PAPER_PROGRESS.md",
)


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_lock_impl():
    spec = importlib.util.spec_from_file_location("ix_project_sources", LOCK_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load lock implementation: {LOCK_IMPL}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def discover_profile_cdp() -> str:
    request = urllib.request.Request(
        OPENED_LIST_URL,
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        opened = json.load(response)
    matches = [
        row
        for row in opened.get("data", [])
        if int(row.get("profile_id", -1)) == PROFILE_ID and row.get("debugging_address")
    ]
    if len(matches) != 1:
        raise RuntimeError("profile 61 does not expose one unique current debugging address")
    cdp = str(matches[0]["debugging_address"])
    with urllib.request.urlopen(f"http://{cdp}/json/version", timeout=10) as response:
        version = json.load(response)
    if not version.get("webSocketDebuggerUrl"):
        raise RuntimeError("current profile 61 endpoint has no debugger websocket")
    return cdp


def enter_shared_lock(stack: contextlib.ExitStack, lock_impl, path: Path) -> None:
    deadline = time.monotonic() + 7200.0
    while True:
        try:
            stack.enter_context(lock_impl.PortableFileLock(path, 7200.0))
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for shared browser lock access: {path}")
            time.sleep(15.0)


def main() -> int:
    required = (PROMPT, *MATERIALS)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing Pro inputs: {missing}")
    oracle = shutil.which("oracle")
    if not oracle:
        raise RuntimeError("oracle executable not found")

    visible_report = RUN_DIR / "visible-report.md"
    if visible_report.exists():
        raise RuntimeError("terminal report already exists; refusing a duplicate Pro turn")
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_HOME.mkdir(parents=True, exist_ok=True)
    raw_log = RUN_DIR / "oracle.log"
    manifest_path = RUN_DIR / "manifest.json"
    lock_impl = load_lock_impl()
    locks = [
        LOCK_ROOT / "browser-61.lock",
        LOCK_ROOT / f"project-{PROJECT_ID}.lock",
        LOCK_ROOT / f"pro-turn-{NONCE}.lock",
    ]
    manifest = {
        "status": "WAITING_FOR_EXCLUSIVE_PROFILE_LOCK",
        "project_id": PROJECT_ID,
        "project_url": PROJECT_URL,
        "nonce": NONCE,
        "profile": PROFILE_ID,
        "requested_model": "gpt-5-pro",
        "effort": "MAX_EFFORT_NOT_SEPARATELY_EXPOSED",
        "prompt_path": str(PROMPT),
        "materials": [str(path) for path in MATERIALS],
        "oracle_home": str(ORACLE_HOME),
        "raw_log": str(raw_log),
        "visible_report": str(visible_report),
        "locks": [str(path) for path in locks],
        "started_at": now(),
    }
    write_json(manifest_path, manifest)

    with contextlib.ExitStack() as stack:
        for path in locks:
            enter_shared_lock(stack, lock_impl, path)
        cdp = discover_profile_cdp()
        manifest["runtime_cdp"] = cdp
        manifest["status"] = "RUNNING"
        write_json(manifest_path, manifest)

        env = os.environ.copy()
        env["ORACLE_HOME_DIR"] = str(ORACLE_HOME)
        command = [
            oracle,
            "--engine", "browser",
            "--remote-chrome", cdp,
            "--chatgpt-url", PROJECT_URL,
            "--browser-model-strategy", "select",
            "--model", "gpt-5-pro",
            "--browser-attachments", "never",
            "--browser-archive", "never",
            "--timeout", "120m",
            "--heartbeat", "30",
            "--wait",
            "--verbose",
            "--slug", "duca-full-data-comparable-protocol-v001",
            "--write-output", str(visible_report),
            "--prompt",
            (
                "Read every provided file completely. Independently reconcile the verified prior Pro freeze with the later "
                "human requirement for complete matched training and complete official held-out evaluation. Freeze exact data "
                "identities, the no-leakage sequence, and one executable current task without delegating scientific choices to "
                f"Codex. Preserve nonce {NONCE} verbatim in the response."
            ),
            "--file", str(PROMPT),
        ]
        for path in MATERIALS:
            command.extend(("--file", str(path)))
        manifest["command_shape"] = command
        write_json(manifest_path, manifest)
        with raw_log.open("w", encoding="utf-8", errors="replace") as output:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            manifest["oracle_pid"] = process.pid
            write_json(manifest_path, manifest)
            return_code = process.wait()

    report_text = visible_report.read_text(encoding="utf-8", errors="replace") if visible_report.is_file() else ""
    log_text = raw_log.read_text(encoding="utf-8", errors="replace")
    nonce_bound = NONCE in report_text
    project_bound = f"https://chatgpt.com/g/{PROJECT_ID}-duca/c/" in log_text
    model_verified = "Model selection evidence:" in log_text and "verified=yes" in log_text
    complete = bool(return_code == 0 and visible_report.is_file() and nonce_bound and project_bound and model_verified)
    manifest.update(
        status="COMPLETED" if complete else "NEEDS_ATTENTION",
        return_code=return_code,
        completed_at=now(),
        visible_report_exists=visible_report.is_file(),
        visible_report_bytes=visible_report.stat().st_size if visible_report.is_file() else 0,
        nonce_bound=nonce_bound,
        project_bound=project_bound,
        model_verified=model_verified,
    )
    write_json(manifest_path, manifest)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
