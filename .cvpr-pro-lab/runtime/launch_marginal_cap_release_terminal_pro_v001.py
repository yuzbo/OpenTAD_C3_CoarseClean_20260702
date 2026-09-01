#!/usr/bin/env python3
"""Launch one exact-DUCA Pro terminal adjudication on the current profile 61 session."""

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
PROMPT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_DUCA_MARGINAL_CAP_RELEASE_TERMINAL_ADJUDICATION-v001.md"
PROJECT_ID = "g-p-6a91061f789881918ccd8357ca3d6c92"
PROJECT_URL = f"https://chatgpt.com/g/{PROJECT_ID}/project?tab=chats"
PROFILE_ID = 61
NONCE = "DUCA-MARGINAL-CAP-RELEASE-TERMINAL-ADJUDICATION-v001-20260831"
LATEST_COMMIT = "d2fad7c0dfc4a5efe98b10b9eee4723c6805699f"
RUN_DIR = ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-cap-release-terminal-v001"
ORACLE_HOME = ROOT / ".cvpr-pro-lab/oracle/duca-marginal-cap-release-terminal-v001"
LOCK_ROOT = Path.home() / ".codex/browser-broker-locks"
LOCK_IMPL = Path(r"C:\Users\skywalker\.codex\skills\cvpr-pro-lab\scripts\ix_project_sources.py")
OPENED_LIST_URL = "http://127.0.0.1:53200/api/v2/native-client-profile-opened-list"
MATERIALS = (
    ROOT / ".cvpr-pro-lab/evaluator-runs/duca-marginal-cap-release-d2fad7c0-job1262117/oracle_cap_release_result.json",
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/probe_result.json",
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/selection_k384_receipt.json",
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/counterfactual_k256_receipt.json",
    ROOT / ".cvpr-pro-lab/pro-reviews/runs/duca-marginal-oracle-gray-zone-v001/materials/counterfactual_k512_receipt.json",
)
GITHUB = {
    "repository": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702",
    "branch": "https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/tree/feature/duca-marginal-cap-release-falsifier-v1-20260831",
    "commit": f"https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/commit/{LATEST_COMMIT}",
    "runner": f"https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/{LATEST_COMMIT}/tools/bata/run_duca_marginal_frozen_h65_probe.py",
    "allocator": f"https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/{LATEST_COMMIT}/opentad/models/duca/dynamic_budget.py",
    "test": f"https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702/blob/{LATEST_COMMIT}/tests/test_duca_marginal_budget.py",
}


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
    """Wait for a Windows-exclusive lock handle before acquiring the same shared lock."""
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
    missing = [str(path) for path in (PROMPT, *MATERIALS) if not path.is_file()]
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
        "github": GITHUB,
        "oracle_home": str(ORACLE_HOME),
        "raw_log": str(raw_log),
        "visible_report": str(visible_report),
        "locks": [str(path) for path in locks],
        "started_at": now(),
    }
    write_json(manifest_path, manifest)

    with contextlib.ExitStack() as stack:
        for path in locks:
            # Shared profile 61 is intentionally serialized across projects. Waiting here
            # touches no browser state and starts no conversation until the current owner exits.
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
            "--slug", "duca-marginal-cap-release-terminal-v001",
            "--write-output", str(visible_report),
            "--prompt",
            (
                "Read every attached file completely and answer the authoritative prompt as one independent "
                "scientific adjudication. Treat the GitHub repository, branch, exact commit "
                f"{LATEST_COMMIT}, runner, allocator and test permalinks in the prompt as the latest code truth. "
                f"Preserve nonce {NONCE} verbatim in the response."
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
