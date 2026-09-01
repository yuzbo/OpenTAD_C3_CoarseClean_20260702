#!/usr/bin/env python3
"""Launch exactly one fresh DUCA Project Pro turn under shared browser locks."""

from __future__ import annotations

import contextlib
import importlib.util
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import time
import urllib.request


ROOT = Path(r"E:\DeskTop\TAD\OpenTAD_C3_CoarseClean_20260702")
SKILL_SCRIPT = Path(r"C:\Users\skywalker\.codex\skills\cvpr-pro-lab\scripts\ix_project_sources.py")
PROJECT_ID = "g-p-6a9061a41bbc819190f4cde94a6c733c"
PROJECT_URL = f"https://chatgpt.com/g/{PROJECT_ID}/project"
COORDINATOR_ID = "019fa3db-42bf-7f30-a0ab-2b8171ab33ed"
COMMIT = "63a726a4aaf48ecbf6780bb196de43a890c6b4df"
CDP = "127.0.0.1:59064"
MODEL = "gpt-5-pro"
PARENT = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_SCIENTIFIC_TAKEOVER-v003-serial-pending.md"
DELTA = ROOT / ".cvpr-pro-lab/pro-reviews/prompts/PRO_SERIAL_DECISION_DELTA-v001.md"
RUNS = ROOT / ".cvpr-pro-lab/pro-reviews/runs"
ORACLE_HOME = Path(r"C:\Users\skywalker\.codex\private\cvpr-pro-lab") / PROJECT_ID / "oracle"
LOCK_ROOT = Path.home() / ".codex/browser-broker-locks"


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=10) as response:
        return json.load(response)


def targets():
    result = []
    for item in get_json(f"http://{CDP}/json/list"):
        result.append({
            "id": item.get("id"),
            "type": item.get("type"),
            "title": item.get("title"),
            "url": item.get("url"),
        })
    return result


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_lock_module():
    spec = importlib.util.spec_from_file_location("cvpr_ix_project_sources", SKILL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load lock implementation from {SKILL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_prompt(marker: str, nonce: str) -> str:
    return f"""{marker}

You are the acting Scientific First-Author Agent and Primary Research Owner for this specific model, experiment program, and paper. Take first responsibility for the paper-level innovation, implementation-aware method design, fair experiment route, interpretation, claim scope, and publication plan. Think and decide as the researcher who must make this work rigorous and publishable, not as a detached reviewer.

Codex is only your implementation and evidence-feedback system. Builder implements your frozen decision; Critic attacks it; Evaluator measures it; the coordinator records and routes artifacts. Codex may expose blockers, alternatives, failures, or falsification evidence, but it cannot select, continue, pivot, stop, or expand the scientific route. You must adjudicate that evidence and issue the next scientific decision.

This is a newly created conversation. Do not assume access to an earlier chat. Reconstruct the project from all twelve named Project Sources, the pinned canonical GitHub revision, CURRENT_RESEARCH_STATE-v001.md, MODEL_EXPERIMENT_HISTORY-v001.md, and the two supplied control-plane files. Identify anything missing or stale before deciding. Exclude the dirty local coordinator checkout, the quarantined ten-Source review, and every 2026-08-11 parallel-routing/stress-test response from scientific evidence.

The human remains the legal/accountable author, PI, spending and test-access authority, and final submission approver. Your direction cannot override human approval, venue rules, research integrity, or evidence-admission gates. This turn is not authorization for GPU/paid or remote execution, held-out/test access, Git push, route freeze, formal result promotion, claim expansion, submission, or public release.

Routing identity (echo exactly in SESSION_ASSERTION and CONTEXT_USED):
- fresh marker: {marker}
- coordinator task ID: {COORDINATOR_ID}
- Project ID: {PROJECT_ID}
- Project URL: {PROJECT_URL}
- fresh nonce: {nonce}
- canonical GitHub: https://github.com/yuzbo/OpenTAD_C3_CoarseClean_20260702.git
- exact review commit: {COMMIT}
- CURRENT_RESEARCH_STATE version: v001
- MODEL_EXPERIMENT_HISTORY version: v001
- expected ordered Project Sources: PROJECT_CHARTER-v001.md; LITERATURE_AND_GAP-v001.md; ROUTE_DECISION-v001.md; EXPERIMENT_PLAN-v001.md; IMPLEMENTATION_STATUS-v001.md; RESULTS-v001.md; RESULT_ANALYSIS-v001.md; FAILURES_AND_PIVOTS-v001.md; CLAIM_MAP-v001.md; PAPER_DRAFT-v001.md; CURRENT_RESEARCH_STATE-v001.md; MODEL_EXPERIMENT_HISTORY-v001.md

First perform HISTORY_SYNTHESIS and CONTEXT_USED verification. Inspect the canonical GitHub repository at exactly {COMMIT}; distinguish verified repository facts, confirmed Source facts, historical/advisory proposals, and missing information. Do not invent metrics, literature, implementation facts, access, or permissions. The formal evidence state is BLOCKED_PRE_RESULT: no partial, proxy, subset, single-seed, failed-root, upstream, synthetic, infrastructure, or unsealed value may be promoted or compared against the historical AdaTAD/uniform-0.5 reference near mAP 65.

Then act as scientific first author: decide the smallest defensible TAD-specific problem and contribution; judge whether the pinned implementation faithfully tests it; compare the existing candidate routes; identify novelty/fairness/leakage/overengineering risks; select exactly one route status; design the cheapest decisive falsification and strongest fair official baseline; freeze success/failure thresholds, seeds, stop/checkpoint/evaluator/split/budget/cost rules; and issue only bounded no-GPU/no-held-out/no-push tasks for the three independent Codex roles.

Your response must contain all of these top-level sections and be complete enough to save verbatim as PRO_INITIAL_REVIEW-v002:
1. SESSION_ASSERTION — say this is a wholly new conversation and echo marker, coordinator ID, Project ID, nonce, commit, and state/history versions.
2. MODEL_EFFORT_ASSERTION — report the model/effort route observed in the UI; coordinator browser evidence remains authoritative.
3. ROLE_ACKNOWLEDGMENT — repeat the acting first-author/primary-owner responsibility in one sentence.
4. CONTEXT_USED — Project ID, exact commit, all twelve ordered Source versions actually used, GitHub access result, and every missing/stale input.
5. HISTORY_SYNTHESIS — reconstruct model, route, experiment, failure, interpretation, and paper-narrative evolution.
6. PAPER_OBJECTIVE — offline TAD task boundary, unresolved scientific problem, intended contribution, falsifiable hypothesis, and CVPR publication bar.
7. CURRENT_JUDGMENT — implementation fidelity plus novelty, fairness, leakage, reproducibility, and overengineering verdicts.
8. SCIENTIFIC_DECISION — exactly one of CONTINUE, REVISE, PIVOT, STOP, ESCALATE_HUMAN, with rationale. Do not delegate the choice to Codex.
9. ROUTE_AND_CLAIMS — selected mechanism; at most two claims; anti-claims; falsifier; explicit non-claims; precise freeze/revision requirements.
10. CODEX_DISPATCH — three separate queue-ready briefs for Builder, Critic, and Evaluator. Each must cite this decision, one claim/falsification question, frozen inputs, allowed reads/writes, expected artifact, tests/audit, stop/block conditions, and current human-permission boundary.
11. EXPERIMENT_PLAN — ordered ladder, strongest baseline, isolation ablations, complete official split/evaluator, seeds, compute/cost boundary, success and failure thresholds, checkpoint/stop rules, resource estimate, and required future human gates. Clearly distinguish preparatory/pilot/formal evidence.
12. PUBLICATION_PLAN — paper thesis, closest novelty risk, intended sections/tables/figures if evidence is admitted, required missing evidence, limitations, and work that must not be done now.
13. DRIFT_CHECKLIST — answer all ten items with artifact citations: (1) is implementation testing the hypothesis; (2) contribution vs infrastructure; (3) unnecessary/overengineered parts; (4) baseline/compute/data/tuning/stopping/evaluator fairness; (5) leakage/cherry-picking/post-hoc drift/relabeling; (6) formal/reproducible/serious/publishable; (7) falsifying evidence and preservation; (8) continue/simplify/pivot/stop; (9) publishable model idea vs engineering system; (10) exact decision, three-role work, and required return evidence.
14. NEXT_RETURN_CONTRACT — exact versioned artifacts/evidence Codex must return before a later fresh Pro decision.

If GitHub or any essential Source is inaccessible, do not guess: use ESCALATE_HUMAN or make the missing context an explicit blocking precondition. Do not treat this prompt or the attached control-plane files as scientific decisions.

{marker}"""


def main() -> int:
    for path in (PARENT, DELTA):
        if not path.is_file():
            raise FileNotFoundError(path)
    version = get_json(f"http://{CDP}/json/version")
    if not version.get("webSocketDebuggerUrl"):
        raise RuntimeError("CDP version probe lacks webSocketDebuggerUrl")

    lock_module = load_lock_module()
    browser_lock = LOCK_ROOT / f"browser-{lock_module.safe_lock_name(CDP)}.lock"
    project_lock = LOCK_ROOT / f"project-{lock_module.safe_lock_name(PROJECT_ID)}.lock"

    with contextlib.ExitStack() as stack:
        stack.enter_context(lock_module.PortableFileLock(browser_lock, 7200))
        stack.enter_context(lock_module.PortableFileLock(project_lock, 7200))

        nonce = secrets.token_hex(16)
        turn_id = f"duca-serial-pro-{nonce}"
        pro_lock = LOCK_ROOT / f"pro-turn-{lock_module.safe_lock_name(turn_id)}.lock"
        stack.enter_context(lock_module.PortableFileLock(pro_lock, 7200))

        marker = f"DUCA-FRESH-SERIAL-PRO-20260811::{COORDINATOR_ID}::{PROJECT_ID}::{nonce}"
        run_dir = RUNS / turn_id
        run_dir.mkdir(parents=True, exist_ok=False)
        prompt_path = run_dir / "prompt.md"
        response_path = run_dir / "raw-response.md"
        log_path = run_dir / "oracle.log"
        receipt_path = run_dir / "route-receipt.json"
        prompt = build_prompt(marker, nonce)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")

        before = targets()
        receipt = {
            "schema_version": "1.0",
            "status": "RUNNING",
            "queue_message_id": "msg-20260810T185405Z-ad987978f297",
            "human_gate": "HUMAN_AUTHORIZATION_FRESH_SERIAL_PRO_AFTER_FAIL_CLOSE",
            "turn_id": turn_id,
            "marker": marker,
            "nonce": nonce,
            "expected_project_id": PROJECT_ID,
            "expected_project_url": PROJECT_URL,
            "expected_commit": COMMIT,
            "expected_sources": [
                "PROJECT_CHARTER-v001.md", "LITERATURE_AND_GAP-v001.md",
                "ROUTE_DECISION-v001.md", "EXPERIMENT_PLAN-v001.md",
                "IMPLEMENTATION_STATUS-v001.md", "RESULTS-v001.md",
                "RESULT_ANALYSIS-v001.md", "FAILURES_AND_PIVOTS-v001.md",
                "CLAIM_MAP-v001.md", "PAPER_DRAFT-v001.md",
                "CURRENT_RESEARCH_STATE-v001.md", "MODEL_EXPERIMENT_HISTORY-v001.md",
            ],
            "requested_model": MODEL,
            "requested_route": "ChatGPT Project browser target=Pro (highest verified Pro picker; no separate effort control on this route)",
            "cdp": CDP,
            "cdp_browser": version.get("Browser"),
            "cdp_protocol": version.get("Protocol-Version"),
            "locks": [str(browser_lock), str(project_lock), str(pro_lock)],
            "oracle_home_dir": str(ORACLE_HOME),
            "prompt_path": str(prompt_path),
            "response_path": str(response_path),
            "started_at": utc_now(),
            "targets_before": before,
        }
        write_json(receipt_path, receipt)

        ORACLE_HOME.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ORACLE_HOME_DIR"] = str(ORACLE_HOME)
        oracle = shutil.which("oracle")
        if not oracle:
            raise RuntimeError("oracle executable not found")
        command = [
            oracle,
            "--engine", "browser",
            "--remote-chrome", CDP,
            "--chatgpt-url", PROJECT_URL,
            "--browser-model-strategy", "select",
            "--model", MODEL,
            "--browser-attachments", "never",
            "--browser-archive", "never",
            "--timeout", "120m",
            "--heartbeat", "30",
            "--wait",
            "--verbose",
            "--slug", f"duca-serial-{nonce[:8]}-pro",
            "--write-output", str(response_path),
            "--prompt", f"{marker} Read every supplied Markdown file in full and follow prompt.md as the authoritative response contract. {marker}",
            "--file", str(prompt_path), str(PARENT), str(DELTA),
        ]

        with log_path.open("w", encoding="utf-8", errors="replace") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()

        after = targets()
        before_ids = {item.get("id") for item in before}
        new_targets = [item for item in after if item.get("id") not in before_ids]
        response_exists = response_path.is_file() and response_path.stat().st_size > 0
        receipt.update({
            "status": "COMPLETED_PENDING_AUDIT" if return_code == 0 and response_exists else "FAILED",
            "return_code": return_code,
            "response_exists": response_exists,
            "response_size": response_path.stat().st_size if response_exists else 0,
            "ended_at": utc_now(),
            "targets_after": after,
            "new_targets": new_targets,
            "actual_project_candidates": [
                item for item in after
                if PROJECT_ID in (item.get("url") or "") or "DUCA" in (item.get("title") or "")
            ],
        })
        write_json(receipt_path, receipt)
        print(json.dumps({
            "turn_id": turn_id,
            "marker": marker,
            "status": receipt["status"],
            "return_code": return_code,
            "run_dir": str(run_dir),
            "receipt": str(receipt_path),
            "response": str(response_path),
        }, ensure_ascii=False, indent=2), flush=True)
        return return_code if return_code != 0 else (0 if response_exists else 2)


if __name__ == "__main__":
    raise SystemExit(main())
