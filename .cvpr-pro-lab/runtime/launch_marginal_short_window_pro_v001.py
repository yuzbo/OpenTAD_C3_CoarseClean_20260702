#!/usr/bin/env python3
"""Launch one exact-Project Pro adjudication without restarting profile 61."""

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
PROMPT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_SHORT_WINDOW_CONTRACT-v002.md"
PROJECT_ID = "g-p-6a91061f789881918ccd8357ca3d6c92"
PROJECT_URL = f"https://chatgpt.com/g/{PROJECT_ID}/project?tab=chats"
PROFILE_ID = 61
NONCE = "DUCA-MARGINAL-SHORT-WINDOW-CONTRACT-v002-20260831"
RUN_DIR = ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-short-window-contract-v002"
ORACLE_HOME = ROOT / ".cvpr-pro-lab/oracle/duca-marginal-short-window-contract-v002"
LOCK_ROOT = Path.home() / ".codex/browser-broker-locks"
LOCK_IMPL = Path(r"C:\Users\skywalker\.codex\skills\cvpr-pro-lab\scripts\ix_project_sources.py")
OPENED_LIST_URL = "http://127.0.0.1:53200/api/v2/native-client-profile-opened-list"


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


def main() -> int:
    if not PROMPT.is_file():
        raise FileNotFoundError(PROMPT)
    oracle = shutil.which("oracle")
    if not oracle:
        raise RuntimeError("oracle executable not found")
    if (RUN_DIR / "visible-report.md").exists():
        raise RuntimeError("terminal report already exists; refusing a duplicate Pro turn")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ORACLE_HOME.mkdir(parents=True, exist_ok=True)
    raw_log = RUN_DIR / "oracle.log"
    visible_report = RUN_DIR / "visible-report.md"
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
        "oracle_home": str(ORACLE_HOME),
        "raw_log": str(raw_log),
        "visible_report": str(visible_report),
        "locks": [str(path) for path in locks],
        "started_at": now(),
    }
    write_json(manifest_path, manifest)

    with contextlib.ExitStack() as stack:
        for path in locks:
            stack.enter_context(lock_impl.PortableFileLock(path, 30.0))
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
            "--slug", "duca-marginal-short-window-v002",
            "--write-output", str(visible_report),
            "--prompt",
            (
                "Read the attached authoritative prompt completely and answer it as one independent "
                f"scientific adjudication. Preserve nonce {NONCE} verbatim in the response."
            ),
            "--file", str(PROMPT),
        ]
        manifest["command_shape"] = command[:-2] + ["--file", str(PROMPT)]
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

    manifest.update(
        status="COMPLETED" if return_code == 0 and visible_report.is_file() else "NEEDS_ATTENTION",
        return_code=return_code,
        completed_at=now(),
        visible_report_exists=visible_report.is_file(),
        visible_report_bytes=visible_report.stat().st_size if visible_report.is_file() else 0,
    )
    write_json(manifest_path, manifest)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
