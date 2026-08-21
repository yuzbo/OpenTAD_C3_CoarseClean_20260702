from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.duca_protected_physical_training import canonical_sha256, sha256_file


SCHEMA = "duca_protected_physical_authorization_v1"
GATE_SCHEMA = "duca_protected_physical_full_model_gate_v1"
PROTOCOL_SCHEMA = "duca_protected_physical_protocol_manifest_v1"
def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"TrueTime PRE_RUN authorization failed: {message}")


def _load(path: str, schema: str) -> tuple[Path, dict]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"missing evidence {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(payload.get("schema") == schema, f"schema mismatch for {resolved.name}")
    return resolved, payload


def authorize(*, expected_commit: str, protocol_json: str, gate_json: str, output_json: str) -> dict:
    _require(re.fullmatch(r"[0-9a-f]{40}", expected_commit) is not None, "bad commit")
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    _require(actual_commit == expected_commit, "commit drift")
    _require(
        not subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        ).strip(),
        "dirty tree",
    )
    protocol_path, protocol = _load(protocol_json, PROTOCOL_SCHEMA)
    gate_path, gate = _load(gate_json, GATE_SCHEMA)
    _require(protocol.get("ok") is True, "protocol did not pass")
    _require(protocol.get("git_commit") == expected_commit, "protocol commit drift")
    _require(gate.get("status") == "p1_p2_full_model_gate_passed", "full-model gate did not pass")
    _require(gate.get("runtime", {}).get("git_commit") == expected_commit, "gate commit drift")
    protocol_sha = sha256_file(protocol_path)
    _require(gate.get("protocol_manifest", {}).get("sha256") == protocol_sha, "gate/P0 mismatch")
    route_arm = str(protocol["route_arm"])
    arm_record = protocol["configs"]["arms"][route_arm]
    _require(gate.get("config", {}).get("route_arm") == route_arm, "gate route-arm drift")
    _require(gate.get("config", {}).get("sha256") == arm_record["source_sha256"], "gate config drift")

    output = Path(output_json).expanduser().resolve()
    _require(not output.exists(), "refusing to overwrite authorization")
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "git_commit": expected_commit,
        "route": protocol["route"],
        "route_arm": protocol["route_arm"],
        "protocol_manifest_sha256": protocol_sha,
        "config_hashes": {route_arm: arm_record["source_sha256"]},
        "authorized_scope": {
            "official60_four_arm_training": True,
            "official60_homotopy_training": True,
            "official60_truetime_paired_arm_training": True,
        },
        "evaluator_pre_run": {
            "path": str(gate_path),
            "sha256": sha256_file(gate_path),
            "status": gate["status"],
        },
        "paper_claim_allowed": False,
    }
    payload["authorization_content_sha256"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        authorize(
            expected_commit=args.expected_commit,
            protocol_json=args.protocol_json,
            gate_json=args.gate_json,
            output_json=args.output_json,
        )
    except Exception as exc:
        print(json.dumps({"schema": SCHEMA, "ok": False, "error": str(exc)}, indent=2))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
