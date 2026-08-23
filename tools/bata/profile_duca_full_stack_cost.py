from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata.duca_full_stack_cost import (
    OFFLINE_FULL_WINDOW_PROTOCOL,
    StageRecorder,
    build_profile_summary,
    compare_profile_summaries,
    cpu_enqueue_diagnostic_key,
    integrate_power_samples,
    validate_profile_sample,
    write_profile_artifacts,
)
from tools.bata.duca_cellcf_protocol import (
    LEGACY_EXPOSURE132_COMMITS,
    protocol_for_name,
)
from tools.bata.duca_trained_checkpoint_binding import (
    load_trained_checkpoint_binding,
)


CELLCF_COST_METHODS = frozenset({"cellcf-fixed384", "bare-uniform384"})
DENSE_COST_METHODS = frozenset({"dense-adatad"})
SINGLECLOCK_COST_METHODS = frozenset({"h65-singleclock-on", "h65-singleclock-gate_zero"})
CELLCF_POST_RUN_SCHEMA = "duca_cellcf_post_run_evidence_v1"
CELLCF_COST_BINDING_SCHEMA = "duca_cellcf_cost_binding_v1"
R5_COST_BINDING_SCHEMA = "duca_r5_terminal_cost_binding_v1"
R5_FORMAL_PROTOCOL = "duca_r5_mechanism_matrix_v1"
R5_MAX_UNSELECTED_HOLES = {384: 2, 256: 3}
R5_METHOD_PATTERN = re.compile(
    r"^(?P<backend>actionformer|temporalmaxer)_"
    r"(?P<arm>uniform|learned)_k(?P<budget>384|256)_"
    r"s(?P<seed>3407|5801|8123)$"
)


def parse_r5_method_name(method_name: str) -> dict[str, Any] | None:
    match = R5_METHOD_PATTERN.fullmatch(str(method_name))
    if match is None:
        return None
    return {
        "backend": match.group("backend"),
        "arm": match.group("arm"),
        "budget": int(match.group("budget")),
        "seed": int(match.group("seed")),
    }


def is_r5_cost_method(method_name: str) -> bool:
    return parse_r5_method_name(method_name) is not None


def _validate_r5_cell_payload(
    payload: Any,
    *,
    core: Mapping[str, Any],
    require_sampling_contract: bool,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("R5 cell payload is missing")
    for key, expected in core.items():
        if payload.get(key) != expected:
            raise ValueError(f"R5 cell identity drift: {key}")
    budget = int(core["budget"])
    max_hole = R5_MAX_UNSELECTED_HOLES[budget]
    if int(payload.get("max_unselected_hole", -1)) != max_hole:
        raise ValueError("R5 cell max-gap contract drift")
    expected_interval = (max_hole + 1) * 4
    interval = payload.get("max_selected_interval_source_frames")
    regime = payload.get("sampling_regime")
    if require_sampling_contract and (interval is None or regime is None):
        raise ValueError("R5 cell sampling contract is incomplete")
    if interval is not None and int(interval) != expected_interval:
        raise ValueError("R5 cell source-frame interval drift")
    if regime is not None and regime != "boundary_burst_with_global_coverage":
        raise ValueError("R5 cell sampling regime drift")
    return {**dict(core), "max_unselected_hole": max_hole}


class ProfileArgs(argparse.Namespace):
    def validate(self) -> None:
        r5_method = is_r5_cost_method(self.method_name)
        if str(self.method_name).startswith(("actionformer_", "temporalmaxer_")) and not r5_method:
            raise ValueError("R5 method name is outside the frozen cell matrix")
        if not self.allow_random_init and not self.checkpoint:
            raise ValueError("a checkpoint is required unless --allow-random-init is explicit")
        if self.samples <= 0:
            raise ValueError("--samples must be positive")
        if self.warmup_samples < 0:
            raise ValueError("--warmup-samples must be non-negative")
        if self.loader_workers != 0:
            raise ValueError("paper full-stack serial profiling requires --loader-workers 0")
        if self.batch_size <= 0:
            raise ValueError("--batch-size must be positive")
        if self.power_interval_ms <= 0:
            raise ValueError("--power-interval-ms must be positive")
        if self.trained_commit and re.fullmatch(
            r"[0-9a-f]{40}", str(self.trained_commit)
        ) is None:
            raise ValueError("--trained-commit must be a full lowercase Git commit")
        if self.evidence_commit and re.fullmatch(
            r"[0-9a-f]{40}", str(self.evidence_commit)
        ) is None:
            raise ValueError("--evidence-commit must be a full lowercase Git commit")
        has_post_run_path = bool(str(self.post_run_evidence or "").strip())
        has_post_run_sha = bool(str(self.post_run_evidence_sha256 or "").strip())
        has_checkpoint_evidence = bool(str(self.checkpoint_evidence or "").strip())
        has_checkpoint_evidence_sha = bool(
            str(self.checkpoint_evidence_sha256 or "").strip()
        )
        if has_post_run_path != has_post_run_sha:
            raise ValueError("--post-run-evidence and --post-run-evidence-sha256 are required together")
        if has_checkpoint_evidence != has_checkpoint_evidence_sha:
            raise ValueError(
                "--checkpoint-evidence and --checkpoint-evidence-sha256 are required together"
            )
        if has_post_run_path and has_checkpoint_evidence:
            raise ValueError("CellCF and generic checkpoint evidence are mutually exclusive")
        if self.method_name in CELLCF_COST_METHODS:
            if not has_post_run_path:
                raise ValueError("formal CellCF cost profiles require hash-bound post-run evidence")
            if self.allow_random_init or not self.use_ema:
                raise ValueError("formal CellCF cost profiles require CellCF-trained EMA weights")
        if self.method_name in DENSE_COST_METHODS:
            if not has_checkpoint_evidence:
                raise ValueError("formal dense cost profiles require hash-bound checkpoint evidence")
            if self.allow_random_init or not self.use_ema:
                raise ValueError("formal dense cost profiles require trained EMA weights")
        if r5_method:
            if self.allow_random_init or not self.use_ema or not self.checkpoint:
                raise ValueError("formal R5 cost profiles require an epoch-59 EMA checkpoint")
            if re.fullmatch(r"[0-9a-f]{40}", str(self.config_commit or "")) is None:
                raise ValueError("formal R5 cost profiles require a full --config-commit")
        if self.method_name in CELLCF_COST_METHODS | DENSE_COST_METHODS:
            if not self.trained_commit:
                raise ValueError(
                    "formal cost profiles require --trained-commit"
                )
            if re.fullmatch(
                r"[0-9a-f]{40}", str(self.config_commit or "")
            ) is None:
                raise ValueError(
                    "formal cost profiles require a full --config-commit"
                )
            if not self.evidence_commit:
                raise ValueError(
                    "formal cost profiles require --evidence-commit"
                )
            if self.config_commit != self.trained_commit:
                raise ValueError(
                    "--config-commit must identify the trained model/config commit"
                )
            if self.evidence_commit == self.trained_commit:
                raise ValueError(
                    "trained and evidence commits must be distinct"
                )
            if not str(self.profile_session_id or "").strip():
                raise ValueError("formal cost profiles require --profile-session-id")
            if not str(self.profile_pair_id or "").strip():
                raise ValueError("formal cost profiles require --profile-pair-id")
            if self.profile_repeat_index <= 0:
                raise ValueError(
                    "formal cost profiles require a positive --profile-repeat-index"
                )
            if self.profile_order_position not in (1, 2):
                raise ValueError(
                    "formal cost profiles require --profile-order-position 1 or 2"
                )
        if self.method_name in SINGLECLOCK_COST_METHODS:
            if self.allow_random_init or not self.use_ema or not self.checkpoint:
                raise ValueError("formal SingleClock cost profiles require epoch-59 EMA weights")
            if not self.complete_official_workload or self.warmup_samples != 50:
                raise ValueError("formal SingleClock cost profiles require a complete workload after 50 warmups")
            if re.fullmatch(r"[0-9a-f]{40}", str(self.trained_commit or "")) is None:
                raise ValueError("formal SingleClock cost profiles require --trained-commit")
            if re.fullmatch(r"[0-9a-f]{40}", str(self.evidence_commit or "")) is None:
                raise ValueError("formal SingleClock cost profiles require --evidence-commit")
            if str(self.config_commit or "") != str(self.trained_commit):
                raise ValueError("SingleClock --config-commit must identify the trained commit")
            if not str(self.profile_session_id or "").strip() or not str(self.profile_pair_id or "").strip():
                raise ValueError("formal SingleClock cost profiles require session and pair IDs")
            if self.profile_repeat_index not in (1, 2, 3) or self.profile_order_position not in (1, 2):
                raise ValueError("formal SingleClock cost repeat/order is outside the frozen 3x2 design")


class ProfileArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        return super().parse_args(args=args, namespace=ProfileArgs() if namespace is None else namespace)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = ProfileArgumentParser(description="Measure offline DUCA full-stack latency and energy")
    parser.add_argument("config", help="OpenTAD config")
    parser.add_argument("--checkpoint", default="", help="trained detector checkpoint")
    parser.add_argument("--backbone-pretrain", default="", help="override the config backbone initialization path")
    parser.add_argument("--video-root", default="", help="override the test video root")
    parser.add_argument("--annotation", default="", help="override test/evaluator annotation")
    parser.add_argument("--class-map", default="", help="override the test category map")
    parser.add_argument("--output-prefix", required=True, help="output path without extension")
    parser.add_argument("--method-name", default="duca-fixed384")
    parser.add_argument("--config-commit", default="")
    parser.add_argument(
        "--trained-commit",
        default="",
        help="commit that produced the bound trained checkpoint; defaults to profiler HEAD",
    )
    parser.add_argument(
        "--evidence-commit",
        default="",
        help="exact clean profiler/evidence code commit",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument(
        "--complete-official-workload",
        action="store_true",
        help="measure exactly one complete deterministic test-loader traversal after warmup",
    )
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--sample-power", action="store_true")
    parser.add_argument("--power-interval-ms", type=int, default=20)
    parser.add_argument("--power-gpu-id", default=None, help="physical nvidia-smi GPU index or UUID")
    parser.add_argument("--gpu-index", dest="power_gpu_id", help=argparse.SUPPRESS)
    parser.add_argument("--compare-baseline-summary", default="")
    parser.add_argument("--post-run-evidence", default="", help="CellCF post_run_evidence.json")
    parser.add_argument("--post-run-evidence-sha256", default="", help="frozen SHA256 of post-run evidence")
    parser.add_argument("--checkpoint-evidence", default="", help="generic trained-checkpoint binding JSON")
    parser.add_argument("--checkpoint-evidence-sha256", default="", help="generic binding SHA256")
    parser.add_argument("--profile-session-id", default="")
    parser.add_argument("--profile-pair-id", default="")
    parser.add_argument("--profile-repeat-index", type=int, default=0)
    parser.add_argument("--profile-order-position", type=int, default=0)
    return parser


def resolve_profile_commit_identities(
    args: ProfileArgs, *, actual_commit: str
) -> tuple[str, str]:
    trained_commit = str(args.trained_commit or actual_commit)
    if re.fullmatch(r"[0-9a-f]{40}", trained_commit) is None:
        raise ValueError("--trained-commit must be a full lowercase Git commit")
    evidence_commit = str(args.evidence_commit or "")
    if is_r5_cost_method(args.method_name):
        if trained_commit != actual_commit:
            raise ValueError("R5 trained checkpoint and profiler must use the same exact commit")
        if str(args.config_commit or "") != trained_commit:
            raise ValueError("R5 --config-commit must identify the exact trained commit")
        if evidence_commit and evidence_commit != actual_commit:
            raise ValueError("R5 --evidence-commit must equal the profiler repository HEAD")
        evidence_commit = actual_commit
    elif args.method_name in CELLCF_COST_METHODS | DENSE_COST_METHODS:
        if evidence_commit != actual_commit:
            raise ValueError(
                "--evidence-commit must equal the exact profiler repository HEAD"
            )
        if str(args.config_commit or "") != trained_commit:
            raise ValueError(
                "--config-commit must identify the trained model/config commit"
            )
        if trained_commit == evidence_commit:
            raise ValueError("trained and evidence commits must be distinct")
    elif args.method_name in SINGLECLOCK_COST_METHODS:
        if evidence_commit != actual_commit:
            raise ValueError("SingleClock --evidence-commit must equal profiler HEAD")
        if str(args.config_commit or "") != trained_commit:
            raise ValueError("SingleClock --config-commit must identify the trained commit")
    return trained_commit, evidence_commit


def strip_ddp_prefix(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    if not state_dict:
        return {}
    has_prefix = all(str(key).startswith("module.") for key in state_dict)
    if not has_prefix:
        return dict(state_dict)
    return {str(key)[len("module.") :]: value for key, value in state_dict.items()}


def parse_nvidia_smi_power_lines(lines: Sequence[str]) -> list[float]:
    values = []
    for line in lines:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", str(line))
        if match is None:
            continue
        try:
            value = float(match.group(0))
        except ValueError:
            continue
        if value >= 0.0:
            values.append(value)
    return values


def component_elapsed_ms(*, cuda_elapsed_ms: float, cpu_enqueue_ms: float) -> float:
    cuda_ms = float(cuda_elapsed_ms)
    cpu_ms = float(cpu_enqueue_ms)
    if not (cuda_ms >= 0.0 and cuda_ms < float("inf")):
        raise ValueError("CUDA component elapsed time must be finite and non-negative")
    if not (cpu_ms >= 0.0 and cpu_ms < float("inf")):
        raise ValueError("CPU enqueue time must be finite and non-negative")
    return cuda_ms


def discover_profile_modules(model: Any) -> tuple[dict[str, Any], set[str]]:
    modules: dict[str, Any] = {}
    zero_stages = set()

    selector = getattr(model, "frame_selector", None)
    if selector is None:
        zero_stages.update(("frame_selector_total_ms", "coarse_probe_ms"))
    else:
        modules["frame_selector_total_ms"] = selector
        probe = getattr(selector, "raw_actionness_source", None)
        if probe is None:
            zero_stages.add("coarse_probe_ms")
        else:
            modules["coarse_probe_ms"] = probe

    backbone = getattr(model, "backbone", None)
    if backbone is None:
        zero_stages.update(("backbone_wrapper_total_ms", "heavy_backbone_ms"))
    else:
        modules["backbone_wrapper_total_ms"] = backbone
        wrapped = getattr(backbone, "model", None)
        heavy = getattr(wrapped, "backbone", None)
        if heavy is None:
            zero_stages.add("heavy_backbone_ms")
        else:
            modules["heavy_backbone_ms"] = heavy

    for stage, attribute in (
        ("projection_ms", "projection"),
        ("neck_ms", "neck"),
        ("head_ms", "rpn_head"),
    ):
        module = getattr(model, attribute, None)
        if module is None:
            zero_stages.add(stage)
        else:
            modules[stage] = module
    return modules, zero_stages


class CudaModuleEventHooks:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.handles = []
        self.pending: dict[str, list[tuple[Any, Any, float]]] = {}
        self.starts: dict[tuple[int, str], list[tuple[Any, float]]] = {}

    def register(self, name: str, module: Any) -> None:
        key = (id(module), name)

        def before(_module, _inputs):
            start_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.starts.setdefault(key, []).append((start_event, time.perf_counter()))

        def after(_module, _inputs, _output):
            starts = self.starts.get(key)
            if not starts:
                raise RuntimeError(f"missing CUDA event start for {name}")
            start_event, cpu_start = starts.pop()
            end_event = self.torch.cuda.Event(enable_timing=True)
            end_event.record()
            cpu_ms = max(0.0, (time.perf_counter() - cpu_start) * 1000.0)
            self.pending.setdefault(name, []).append((start_event, end_event, cpu_ms))

        self.handles.append(module.register_forward_pre_hook(before))
        self.handles.append(module.register_forward_hook(after))

    def flush_into(self, recorder: StageRecorder) -> None:
        self.torch.cuda.synchronize()
        for name, events in self.pending.items():
            for start_event, end_event, cpu_ms in events:
                cuda_ms = max(0.0, float(start_event.elapsed_time(end_event)))
                recorder.record_value(
                    name,
                    component_elapsed_ms(cuda_elapsed_ms=cuda_ms, cpu_enqueue_ms=cpu_ms),
                    accumulate=True,
                )
                recorder.record_value(
                    cpu_enqueue_diagnostic_key(name),
                    cpu_ms,
                    accumulate=True,
                )
        self.pending.clear()

    def close(self) -> None:
        for handle in reversed(self.handles):
            handle.remove()
        self.handles.clear()
        self.pending.clear()
        self.starts.clear()


class CudaMethodEventHooks:
    def __init__(self, torch_module: Any) -> None:
        self.torch = torch_module
        self.originals: list[tuple[Any, str, Any]] = []
        self.pending: dict[str, list[tuple[Any, Any, float]]] = {}

    def register(self, name: str, target: Any, method_name: str) -> None:
        original = getattr(target, method_name, None)
        if original is None or not callable(original):
            raise ValueError(f"cannot profile missing method {method_name} for {name}")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            start_event = self.torch.cuda.Event(enable_timing=True)
            end_event = self.torch.cuda.Event(enable_timing=True)
            start_event.record()
            cpu_start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                end_event.record()
                cpu_ms = max(0.0, (time.perf_counter() - cpu_start) * 1000.0)
                self.pending.setdefault(name, []).append((start_event, end_event, cpu_ms))

        self.originals.append((target, method_name, original))
        setattr(target, method_name, wrapped)

    def flush_into(self, recorder: StageRecorder) -> None:
        self.torch.cuda.synchronize()
        for name, events in self.pending.items():
            for start_event, end_event, cpu_ms in events:
                cuda_ms = max(0.0, float(start_event.elapsed_time(end_event)))
                recorder.record_value(
                    name,
                    component_elapsed_ms(cuda_elapsed_ms=cuda_ms, cpu_enqueue_ms=cpu_ms),
                    accumulate=True,
                )
                recorder.record_value(
                    cpu_enqueue_diagnostic_key(name),
                    cpu_ms,
                    accumulate=True,
                )
        self.pending.clear()

    def close(self) -> None:
        for target, method_name, original in reversed(self.originals):
            setattr(target, method_name, original)
        self.originals.clear()
        self.pending.clear()


class ContinuousPowerSampler:
    def __init__(self, *, gpu_id: str, interval_ms: int) -> None:
        self.gpu_id = str(gpu_id)
        self.interval_ms = int(interval_ms)
        self.samples: list[tuple[float, float]] = []
        self.process = None
        self.thread = None

    def start(self) -> None:
        command = [
            "nvidia-smi",
            f"--id={self.gpu_id}",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={self.interval_ms}",
        ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )

        def read_output() -> None:
            assert self.process is not None and self.process.stdout is not None
            for line in self.process.stdout:
                values = parse_nvidia_smi_power_lines([line])
                if values:
                    self.samples.append((time.perf_counter(), values[0]))

        self.thread = threading.Thread(target=read_output, daemon=True)
        self.thread.start()

    def wait_until_ready(self, *, timeout_s: float = 3.0) -> None:
        deadline = time.perf_counter() + float(timeout_s)
        while not self.samples and time.perf_counter() < deadline:
            time.sleep(min(0.01, self.interval_ms / 1000.0))
        if not self.samples:
            raise RuntimeError("nvidia-smi did not produce a power sample before profiling")

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=3.0)
        if self.thread is not None:
            self.thread.join(timeout=3.0)


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _git_repo_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=True,
    )
    return Path(result.stdout.strip()).resolve()


def _git_tree_oid(repo_root: Path, commit: str, relative_path: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", f"{commit}:{relative_path}"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    oid = result.stdout.strip()
    if re.fullmatch(r"[0-9a-f]{40}", oid) is None:
        raise ValueError(
            f"invalid Git tree OID for {commit}:{relative_path}"
        )
    return oid


def _tracked_tree_is_clean(repo_root: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return not bool(result.stdout.strip())


def _ignored_python_sources(repo_root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            "*.py",
            "*.pth",
            "sitecustomize.py",
            "usercustomize.py",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if re.fullmatch(r"[0-9a-f]{64}", text) is None:
        raise ValueError(f"{label} must be a lowercase SHA256")
    return text


def _r5_load_json(path: str | Path, digest: str, label: str) -> tuple[dict[str, Any], Path]:
    resolved = Path(path).expanduser().resolve()
    expected = _require_sha256(digest, f"{label} SHA256")
    if not resolved.is_file():
        raise ValueError(f"R5 {label} is missing: {resolved}")
    if _sha256_file(resolved) != expected:
        raise ValueError(f"R5 {label} SHA256 mismatch")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"R5 {label} must be a JSON object")
    return payload, resolved


def _validate_r5_self_hash(
    payload: Mapping[str, Any], *, hash_key: str, label: str
) -> str:
    expected = _require_sha256(payload.get(hash_key), f"R5 {label} self hash")
    unsigned = dict(payload)
    unsigned.pop(hash_key, None)
    if _canonical_sha256(unsigned) != expected:
        raise ValueError(f"R5 {label} self-hash mismatch")
    return expected


def _r5_bound_file(path: Any, digest: Any, label: str) -> dict[str, str]:
    resolved = Path(str(path or "")).expanduser().resolve()
    expected = _require_sha256(digest, f"R5 {label} SHA256")
    if not resolved.is_file():
        raise ValueError(f"R5 {label} is missing: {resolved}")
    if _sha256_file(resolved) != expected:
        raise ValueError(f"R5 {label} content drift")
    return {"path": str(resolved), "sha256": expected}


def load_r5_terminal_cost_binding(
    *,
    method_name: str,
    config_path: str | Path,
    checkpoint_path: str | Path,
    expected_commit: str,
    matrix_summary_path: str | Path,
    matrix_summary_sha256: str,
    mechanism_gate_path: str | Path,
    mechanism_gate_sha256: str,
    expected_resolved_config_sha256: str | None = None,
    expected_training_identity: Mapping[str, Any] | None = None,
    expected_evaluation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reopen the complete terminal R5 provenance chain for cost and aggregation."""

    cell = parse_r5_method_name(method_name)
    if cell is None:
        raise ValueError(f"unsupported R5 method/cell: {method_name}")
    if re.fullmatch(r"[0-9a-f]{40}", str(expected_commit)) is None:
        raise ValueError("R5 binding requires an exact Git commit")

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise ValueError(f"R5 config is missing: {config}")
    config_sha256 = _sha256_file(config)
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if checkpoint.name != "epoch_59.pth" or not checkpoint.is_file():
        raise ValueError("R5 cost binding requires the terminal epoch_59 checkpoint")
    checkpoint_sha256 = _sha256_file(checkpoint)

    matrix, matrix_path = _r5_load_json(
        matrix_summary_path, matrix_summary_sha256, "matrix summary"
    )
    if (
        matrix.get("schema") != "duca_r5_paper_matrix_v1"
        or matrix.get("task") != "offline_temporal_action_detection"
        or matrix.get("git_commit") != expected_commit
    ):
        raise ValueError("R5 matrix summary protocol/commit drift")
    matrix_cells = matrix.get("cells")
    matrix_cell = next(
        (
            item
            for item in matrix_cells
            if isinstance(item, Mapping) and item.get("id") == method_name
        ),
        None,
    ) if isinstance(matrix_cells, list) else None
    try:
        bound_cell = _validate_r5_cell_payload(
            matrix_cell,
            core=cell,
            require_sampling_contract=False,
        )
    except ValueError as exc:
        raise ValueError("R5 cell/config differs from the sealed matrix") from exc
    if (
        Path(str(matrix_cell.get("config", ""))).expanduser().resolve() != config
        or matrix_cell.get("config_sha256") != config_sha256
    ):
        raise ValueError("R5 cell/config differs from the sealed matrix")

    gate, gate_path = _r5_load_json(
        mechanism_gate_path, mechanism_gate_sha256, "mechanism gate"
    )
    if (
        Path(str(matrix.get("mechanism_gate_output", ""))).expanduser().resolve()
        != gate_path
        or gate.get("ok") is not True
        or gate.get("task") != "offline_temporal_action_detection"
        or gate.get("git_commit") != expected_commit
        or gate.get("forward_backward_optimizer_step_completed") is not True
    ):
        raise ValueError("R5 mechanism gate did not authorize this matrix")

    sidecar_path = Path(f"{checkpoint}.metadata.json").resolve()
    if not sidecar_path.is_file():
        raise ValueError("R5 terminal checkpoint sidecar is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if (
        not isinstance(sidecar, Mapping)
        or sidecar.get("schema_version") != "duca_p0_checkpoint_sidecar_v2"
    ):
        raise ValueError("R5 terminal checkpoint sidecar schema mismatch")
    sidecar_self_sha256 = _validate_r5_self_hash(
        sidecar, hash_key="sidecar_sha256", label="checkpoint sidecar"
    )
    if (
        Path(str(sidecar.get("checkpoint_path", ""))).expanduser().resolve()
        != checkpoint
        or sidecar.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ValueError("R5 checkpoint/sidecar identity drift")

    metadata = sidecar.get("experiment_metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("schema_version") != "duca_p0_checkpoint_metadata_v2"
    ):
        raise ValueError("R5 checkpoint metadata is missing")
    metadata_self_sha256 = _validate_r5_self_hash(
        metadata, hash_key="metadata_sha256", label="checkpoint metadata"
    )
    audit = metadata.get("training_audit")
    if (
        not isinstance(audit, Mapping)
        or audit.get("schema_version") != "duca_p0_training_audit_v2"
    ):
        raise ValueError("R5 terminal training audit is missing")
    audit_self_sha256 = _validate_r5_self_hash(
        audit, hash_key="audit_sha256", label="training audit"
    )
    audit_path = checkpoint.parent.parent / "duca_selected_axis_training_audit.json"
    if not audit_path.is_file():
        raise ValueError("R5 persisted training audit is missing")
    persisted_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if persisted_audit != audit:
        raise ValueError("R5 persisted and checkpoint-embedded audits differ")

    required = {
        "status": "complete",
        "git_commit": expected_commit,
        "variant": method_name,
        "seed": cell["seed"],
        "formal_protocol": R5_FORMAL_PROTOCOL,
        "training_profile": "official60",
        "checkpoint_criterion": "terminal_epoch_59_state_dict_ema",
        "primary_checkpoint_epoch": 59,
        "primary_checkpoint_state_key": "state_dict_ema",
        "expected_train_batches_per_epoch": 100,
        "expected_successful_optimizer_updates": 6000,
        "last_completed_epoch": 59,
        "epochs_completed": 60,
        "train_batches_per_epoch": 100,
        "scheduler_last_epoch": 6000,
        "selector_schedule_step": 6000,
    }
    for key, expected in required.items():
        if audit.get(key) != expected:
            raise ValueError(f"R5 terminal training identity mismatch: {key}")
    _validate_r5_cell_payload(
        audit.get("r5_cell"),
        core=cell,
        require_sampling_contract=True,
    )
    if (
        Path(str(audit.get("source_config_path", ""))).expanduser().resolve()
        != config
        or audit.get("source_config_sha256") != config_sha256
    ):
        raise ValueError("R5 audit config identity drift")
    resolved_config_sha256 = _require_sha256(
        audit.get("resolved_config_sha256"), "R5 resolved config SHA256"
    )
    runtime_config_sha256 = _require_sha256(
        audit.get("runtime_config_sha256"), "R5 runtime config SHA256"
    )
    if (
        expected_resolved_config_sha256 is not None
        and resolved_config_sha256 != expected_resolved_config_sha256
    ):
        raise ValueError("R5 resolved config differs from the terminal evidence")
    if (
        Path(str(audit.get("matrix_summary_path", ""))).expanduser().resolve()
        != matrix_path
        or audit.get("matrix_summary_sha256") != matrix_summary_sha256
        or Path(str(audit.get("mechanism_gate_path", ""))).expanduser().resolve()
        != gate_path
        or audit.get("mechanism_gate_sha256") != mechanism_gate_sha256
    ):
        raise ValueError("R5 audit matrix/gate binding drift")

    counters = audit.get("update_audit")
    if not isinstance(counters, Mapping):
        raise ValueError("R5 update audit is missing")
    for key in (
        "attempted_batches",
        "successful_optimizer_updates",
        "scheduler_updates",
        "ema_updates",
        "duca_schedule_updates",
    ):
        if int(counters.get(key, -1)) != 6000:
            raise ValueError(f"R5 terminal update accounting mismatch: {key}")
    if (
        int(counters.get("optimizer_attempts", -1))
        != 6000 + int(counters.get("amp_skipped_attempts", -1))
        or int(counters.get("replay_exhaustions", -1)) != 0
        or int(counters.get("forced_amp_overflow_attempts", -1)) != 0
    ):
        raise ValueError("R5 optimizer/AMP accounting drift")
    records = audit.get("epoch_records")
    if (
        not isinstance(records, list)
        or len(records) != 60
        or [int(record.get("epoch", -1)) for record in records] != list(range(60))
    ):
        raise ValueError("R5 epoch records are incomplete")
    for epoch, record in enumerate(records):
        delta = record.get("counter_delta")
        if not isinstance(delta, Mapping):
            raise ValueError(f"R5 epoch {epoch} counter delta is missing")
        for key in (
            "attempted_batches",
            "successful_optimizer_updates",
            "scheduler_updates",
            "ema_updates",
            "duca_schedule_updates",
        ):
            if int(delta.get(key, -1)) != 100:
                raise ValueError(f"R5 epoch {epoch} accounting mismatch: {key}")
        if (
            int(record.get("scheduler_last_epoch", -1)) != (epoch + 1) * 100
            or int(record.get("selector_schedule_step", -1)) != (epoch + 1) * 100
        ):
            raise ValueError(f"R5 epoch {epoch} scheduler/selector state drift")

    data_bindings = {}
    for prefix in ("pretrain", "evaluation_annotation", "evaluation_class_map"):
        data_bindings[prefix] = _r5_bound_file(
            audit.get(f"{prefix}_path"), audit.get(f"{prefix}_sha256"), prefix
        )
    evaluation_config_sha256 = _require_sha256(
        audit.get("evaluation_config_sha256"), "R5 evaluation config SHA256"
    )
    if gate.get("pretrain_path") != data_bindings["pretrain"]["path"] or (
        gate.get("pretrain_sha256") != data_bindings["pretrain"]["sha256"]
    ):
        raise ValueError("R5 mechanism gate pretrain drift")

    frontend = None
    alignment = None
    if cell["arm"] == "learned":
        initialization = audit.get("selector_initialization_contract")
        if not isinstance(initialization, Mapping):
            raise ValueError("R5 learned cell lacks frontend initialization")
        frontend = {
            **_r5_bound_file(
                initialization.get("checkpoint_path"),
                initialization.get("checkpoint_sha256"),
                "learned frontend checkpoint",
            ),
            "checkpoint_epoch": int(initialization.get("checkpoint_epoch", -1)),
            "checkpoint_state_key": str(initialization.get("checkpoint_state_key", "")),
            "selected_p0_variant": str(initialization.get("selected_p0_variant", "")),
            "learned_variant": str(initialization.get("learned_variant", "")),
        }
        if frontend["checkpoint_epoch"] < 0 or frontend["checkpoint_state_key"] != "state_dict_ema":
            raise ValueError("R5 learned frontend is not a frozen EMA checkpoint")
        alignment_binding = audit.get("hard_swap_alignment")
        if not isinstance(alignment_binding, Mapping):
            raise ValueError("R5 learned cell lacks hard-swap alignment")
        alignment_payload, alignment_path = _r5_load_json(
            alignment_binding.get("path", ""),
            str(alignment_binding.get("sha256", "")),
            "hard-swap alignment",
        )
        alignment_self_sha256 = _validate_r5_self_hash(
            alignment_payload,
            hash_key="alignment_sha256",
            label="hard-swap alignment",
        )
        if alignment_binding.get("self_sha256") != alignment_self_sha256:
            raise ValueError("R5 hard-swap alignment self-hash drift")
        alignment = {
            "path": str(alignment_path),
            "sha256": str(alignment_binding["sha256"]),
            "self_sha256": alignment_self_sha256,
            "context_sha256": str(alignment_binding.get("context_sha256", "")),
            "terminal_suite_sha256": str(
                alignment_binding.get("terminal_suite_sha256", "")
            ),
        }
        _require_sha256(alignment["context_sha256"], "R5 alignment context SHA256")
        _require_sha256(
            alignment["terminal_suite_sha256"], "R5 terminal suite SHA256"
        )
    elif audit.get("selector_initialization_contract") is not None or audit.get(
        "hard_swap_alignment"
    ) is not None:
        raise ValueError("R5 uniform cell unexpectedly carries learned evidence")

    if expected_evaluation is not None:
        expected_pairs = {
            "resolved_config_sha256": resolved_config_sha256,
            "runtime_config_sha256": runtime_config_sha256,
            "evaluation_annotation_path": data_bindings["evaluation_annotation"]["path"],
            "evaluation_annotation_sha256": data_bindings["evaluation_annotation"]["sha256"],
            "evaluation_class_map_path": data_bindings["evaluation_class_map"]["path"],
            "evaluation_class_map_sha256": data_bindings["evaluation_class_map"]["sha256"],
            "evaluation_config_sha256": evaluation_config_sha256,
        }
        for key, expected in expected_pairs.items():
            observed = expected_evaluation.get(key)
            if key.endswith("_path"):
                observed = str(Path(str(observed or "")).expanduser().resolve())
            if observed != expected:
                raise ValueError(f"R5 evaluation/audit binding mismatch: {key}")

    binding = {
        "schema": R5_COST_BINDING_SCHEMA,
        "git_commit": expected_commit,
        "method": method_name,
        "r5_cell": bound_cell,
        "config_path": str(config),
        "config_sha256": config_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "runtime_config_sha256": runtime_config_sha256,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "checkpoint_sidecar_path": str(sidecar_path),
        "checkpoint_sidecar_sha256": _sha256_file(sidecar_path),
        "checkpoint_sidecar_self_sha256": sidecar_self_sha256,
        "checkpoint_metadata_self_sha256": metadata_self_sha256,
        "checkpoint_experiment_metadata_sha256": _canonical_sha256(metadata),
        "training_audit_path": str(audit_path.resolve()),
        "training_audit_sha256": _sha256_file(audit_path),
        "training_audit_self_sha256": audit_self_sha256,
        "successful_optimizer_updates": 6000,
        "scheduler_updates": 6000,
        "ema_updates": 6000,
        "epoch_record_count": 60,
        "matrix_summary_path": str(matrix_path),
        "matrix_summary_sha256": matrix_summary_sha256,
        "mechanism_gate_path": str(gate_path),
        "mechanism_gate_sha256": mechanism_gate_sha256,
        "pretrain": data_bindings["pretrain"],
        "evaluation_annotation": data_bindings["evaluation_annotation"],
        "evaluation_class_map": data_bindings["evaluation_class_map"],
        "evaluation_config_sha256": evaluation_config_sha256,
        "frontend_initialization": frontend,
        "hard_swap_alignment": alignment,
    }
    binding["binding_sha256"] = _canonical_sha256(binding)

    if expected_training_identity is not None:
        identity_pairs = {
            "variant": method_name,
            "seed": cell["seed"],
            "successful_optimizer_updates": 6000,
            "checkpoint_sidecar_path": binding["checkpoint_sidecar_path"],
            "checkpoint_sidecar_sha256": binding["checkpoint_sidecar_sha256"],
            "training_audit_path": binding["training_audit_path"],
            "training_audit_sha256": binding["training_audit_sha256"],
            "training_audit_self_sha256": binding["training_audit_self_sha256"],
            "matrix_summary_sha256": matrix_summary_sha256,
            "mechanism_gate_sha256": mechanism_gate_sha256,
            "pretrain_path": binding["pretrain"]["path"],
            "pretrain_sha256": binding["pretrain"]["sha256"],
            "frontend_initialization": frontend,
        }
        for key, expected in identity_pairs.items():
            if expected_training_identity.get(key) != expected:
                raise ValueError(f"R5 terminal identity mismatch: {key}")
    return binding


def load_cellcf_cost_binding(
    post_run_evidence_path: str | Path,
    post_run_evidence_sha256: str,
    *,
    expected_checkpoint_path: str | Path | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    evidence_path = Path(post_run_evidence_path).expanduser().resolve()
    if not evidence_path.is_file():
        raise ValueError(f"CellCF post-run evidence is missing: {evidence_path}")
    expected_evidence_sha = _require_sha256(
        post_run_evidence_sha256,
        "CellCF post-run evidence SHA256",
    )
    observed_evidence_sha = _sha256_file(evidence_path)
    if observed_evidence_sha != expected_evidence_sha:
        raise ValueError("CellCF post-run evidence SHA256 mismatch")

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CellCF post-run evidence must be a JSON object")
    if payload.get("schema") != CELLCF_POST_RUN_SCHEMA or payload.get("ok") is not True:
        raise ValueError("CellCF post-run evidence has an incompatible schema/status")
    artifact_chain_sha = _require_sha256(
        payload.get("artifact_chain_sha256"),
        "CellCF post-run artifact-chain SHA256",
    )
    unsigned = dict(payload)
    unsigned.pop("artifact_chain_sha256", None)
    if artifact_chain_sha != _canonical_sha256(unsigned):
        raise ValueError("CellCF post-run artifact-chain SHA256 mismatch")

    commit = str(payload.get("git_commit") or "")
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("CellCF post-run evidence has an invalid git commit")
    if expected_commit is not None and commit != str(expected_commit):
        raise ValueError("CellCF post-run evidence is bound to another commit")
    training_profile = payload.get("training_profile")
    if training_profile is None:
        if commit not in LEGACY_EXPOSURE132_COMMITS:
            raise ValueError(
                "CellCF post-run training profile may be absent only for "
                "the audited legacy exposure132 commit"
            )
        training_profile = "exposure132"
    protocol = protocol_for_name(str(training_profile))
    if payload.get("variant") != "cellcf":
        raise ValueError("cost evidence must use the trained CellCF variant")
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("CellCF post-run evidence has an invalid seed")

    config_hashes = {}
    for key in (
        "config_sha256",
        "resolved_config_sha256",
        "runtime_config_sha256",
        "evaluation_runtime_config_sha256",
    ):
        config_hashes[key] = _require_sha256(payload.get(key), f"CellCF post-run {key}")

    if (
        int(payload.get("successful_optimizer_updates", -1))
        != protocol.expected_successful_optimizer_updates
    ):
        raise ValueError("CellCF post-run evidence is not at the frozen terminal update")
    if int(payload.get("checkpoint_epoch", -1)) != protocol.terminal_epoch:
        raise ValueError("CellCF post-run evidence does not bind the terminal epoch")
    if payload.get("checkpoint_state_key") != protocol.terminal_state_key:
        raise ValueError("CellCF post-run evidence does not bind state_dict_ema")

    checkpoint_path = Path(str(payload.get("checkpoint_path") or "")).expanduser().resolve()
    if checkpoint_path.name != f"epoch_{protocol.terminal_epoch}.pth":
        raise ValueError(
            "CellCF post-run evidence does not name the exact "
            f"epoch_{protocol.terminal_epoch} checkpoint"
        )
    if not checkpoint_path.is_file():
        raise ValueError(f"CellCF terminal checkpoint is missing: {checkpoint_path}")
    if expected_checkpoint_path is not None:
        requested_checkpoint = Path(expected_checkpoint_path).expanduser().resolve()
        if requested_checkpoint != checkpoint_path:
            raise ValueError("profile checkpoint path differs from CellCF post-run evidence")
    checkpoint_sha = _require_sha256(
        payload.get("checkpoint_sha256"),
        "CellCF terminal checkpoint SHA256",
    )
    if _sha256_file(checkpoint_path) != checkpoint_sha:
        raise ValueError("CellCF terminal checkpoint SHA256 differs from post-run evidence")
    checkpoint_contract = payload.get("checkpoint_payload_contract")
    if not isinstance(checkpoint_contract, Mapping):
        raise ValueError("CellCF post-run evidence is missing the reopened checkpoint contract")
    if checkpoint_contract.get("payload_reopened") is not True:
        raise ValueError("CellCF terminal checkpoint was not reopened during finalization")
    if int(checkpoint_contract.get("epoch", -1)) != protocol.terminal_epoch:
        raise ValueError("CellCF reopened checkpoint contract is not terminal")

    return {
        "schema": CELLCF_COST_BINDING_SCHEMA,
        "post_run_evidence_path": str(evidence_path),
        "post_run_evidence_sha256": observed_evidence_sha,
        "post_run_artifact_chain_sha256": artifact_chain_sha,
        "git_commit": commit,
        "seed": int(seed),
        "variant": "cellcf",
        "training_profile": protocol.name,
        "training_protocol": protocol.to_dict(),
        **config_hashes,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": protocol.terminal_epoch,
        "checkpoint_state_key": protocol.terminal_state_key,
    }


def validate_loaded_checkpoint_binding(
    checkpoint_metadata: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> None:
    loaded_path = Path(str(checkpoint_metadata.get("checkpoint_path") or "")).resolve()
    bound_path = Path(str(binding.get("checkpoint_path") or "")).resolve()
    if loaded_path != bound_path:
        raise ValueError("loaded checkpoint path differs from the CellCF cost binding")
    for key in ("checkpoint_sha256", "checkpoint_epoch", "checkpoint_state_key"):
        if checkpoint_metadata.get(key) != binding.get(key):
            raise ValueError(f"loaded checkpoint differs from the CellCF cost binding on {key}")


def _stable_payload(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _stable_payload(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_stable_payload(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _payload_fingerprint(value: Any) -> str:
    encoded = json.dumps(_stable_payload(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _host_fingerprint() -> str:
    processor = platform.processor().strip()
    generic_processor = processor.lower() in {"", "x86_64", "amd64", "aarch64", "arm64"}
    if generic_processor and Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name") and ":" in line:
                processor = line.split(":", 1)[1].strip()
                break
    return f"{platform.system()}|{platform.release()}|{platform.machine()}|cpu={processor}|count={os.cpu_count()}"


def _software_fingerprint(torch_module: Any) -> str:
    versions = {}
    for package in ("mmengine", "numpy", "decord", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return _payload_fingerprint(
        {
            "python": platform.python_version(),
            "torch": torch_module.__version__,
            "cuda": torch_module.version.cuda,
            "packages": versions,
        }
    )


def _cuda_nvml_uuid(torch_module: Any, device: Any) -> str:
    cuda = torch_module.cuda
    try:
        uuids = cuda._raw_device_uuid_nvml()
        if uuids is None:
            raise RuntimeError("NVML did not return GPU UUIDs")
        if hasattr(cuda, "_get_nvml_device_index"):
            nvml_index = int(cuda._get_nvml_device_index(device))
            value = uuids[nvml_index]
        else:
            logical_index = int(cuda._get_device_index(device, optional=True))
            visible_devices = list(cuda._parse_visible_devices())
            if logical_index < 0 or logical_index >= len(visible_devices):
                raise RuntimeError("Torch logical CUDA index is outside CUDA_VISIBLE_DEVICES")
            visible = visible_devices[logical_index]
            if isinstance(visible, str):
                matches = [uuid for uuid in uuids if uuid.startswith(visible)]
                if len(matches) != 1:
                    raise RuntimeError("CUDA_VISIBLE_DEVICES UUID is missing or ambiguous in NVML")
                value = matches[0]
            else:
                value = uuids[int(visible)]
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot map the Torch CUDA device to an NVML UUID") from exc
    if isinstance(value, bytes):
        value = value.decode("ascii")
    uuid = str(value or "").strip()
    if not uuid:
        raise RuntimeError("Torch NVML mapping returned an empty GPU UUID")
    return uuid


def _query_nvidia_smi_uuid(gpu_id: str) -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"nvidia-smi GPU id {gpu_id!r} did not resolve to exactly one UUID")
    return lines[0]


def _resolve_power_gpu_id(explicit: str | None, torch_module: Any, device: Any) -> str:
    actual_uuid = _cuda_nvml_uuid(torch_module, device)
    requested = actual_uuid if explicit is None or not str(explicit).strip() else str(explicit).strip()
    sampled_uuid = _query_nvidia_smi_uuid(requested)
    if sampled_uuid.lower() != actual_uuid.lower():
        raise RuntimeError(
            "power sampler GPU does not match the Torch compute device: "
            f"requested={requested!r}, sampled_uuid={sampled_uuid!r}, torch_uuid={actual_uuid!r}"
        )
    return actual_uuid


def _detector_stack_fingerprint(model: Any) -> str:
    payload = {}
    for attribute in ("backbone", "projection", "neck", "rpn_head", "token_compressor"):
        module = getattr(model, attribute, None)
        if module is None:
            payload[attribute] = None
            continue
        submodules = [
            (name, f"{type(child).__module__}.{type(child).__qualname__}")
            for name, child in module.named_modules()
        ]
        parameters = [
            {
                "name": name,
                "shape": [int(value) for value in parameter.shape],
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
            for name, parameter in module.named_parameters()
        ]
        payload[attribute] = {
            "class": f"{type(module).__module__}.{type(module).__qualname__}",
            "submodules": submodules,
            "parameters": parameters,
        }
    return _payload_fingerprint(payload)


def _move_to_device(value: Any, device: Any, torch_module: Any) -> Any:
    if torch_module.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device, torch_module) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device, torch_module) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device, torch_module) for item in value)
    return value


def _next_batch(iterator, loader):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _selected_count(model: Any, inputs: Any) -> float:
    selector = getattr(model, "frame_selector", None)
    summary = getattr(selector, "last_forward_summary", {}) if selector is not None else {}
    values = summary.get("effective_budget") if isinstance(summary, Mapping) else None
    if isinstance(values, (list, tuple)) and values:
        return sum(float(value) for value in values) / float(len(values))
    if inputs.ndim == 6:
        return float(inputs.shape[3])
    return float(inputs.shape[2])


def _load_checkpoint(
    model: Any,
    path: str,
    *,
    use_ema: bool,
    torch_module: Any,
    drop_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    checkpoint = torch_module.load(path, map_location="cpu")
    key = "state_dict_ema" if use_ema else "state_dict"
    if key not in checkpoint:
        raise ValueError(f"checkpoint {path} is missing {key}")
    state = strip_ddp_prefix(checkpoint[key])
    dropped = sorted(
        name
        for name in state
        if any(str(name).startswith(prefix) for prefix in drop_prefixes)
    )
    if drop_prefixes:
        state = {
            name: value
            for name, value in state.items()
            if not any(str(name).startswith(prefix) for prefix in drop_prefixes)
        }
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    return {
        "checkpoint_path": str(Path(path).resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_state_key": key,
        "checkpoint_sha256": _sha256_file(path),
        "checkpoint_dropped_prefixes": list(drop_prefixes),
        "checkpoint_dropped_key_count": len(dropped),
        "checkpoint_experiment_metadata_sha256": (
            None
            if not isinstance(checkpoint.get("experiment_metadata"), Mapping)
            else _canonical_sha256(checkpoint["experiment_metadata"])
        ),
    }


def _hardware_fingerprint(torch_module: Any, device: Any, *, gpu_uuid: str | None = None) -> str:
    if device.type != "cuda":
        return f"cpu|torch={torch_module.__version__}"
    props = torch_module.cuda.get_device_properties(device)
    uuid = str(gpu_uuid or _cuda_nvml_uuid(torch_module, device))
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={uuid}",
            "--query-gpu=driver_version,power.limit,clocks.max.sm,clocks.max.memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    runtime = " ".join(query.stdout.split())
    return (
        f"{props.name}|uuid={uuid}|cc={props.major}.{props.minor}|mem={props.total_memory}|{runtime}|"
        f"torch={torch_module.__version__}|cuda={torch_module.version.cuda}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.validate()
    output_prefix = Path(args.output_prefix).expanduser().resolve()
    output_targets = [
        output_prefix.with_suffix(".summary.json"),
        output_prefix.with_suffix(".summary.tsv"),
        output_prefix.with_suffix(".samples.jsonl"),
    ]
    if args.compare_baseline_summary:
        output_targets.append(output_prefix.with_suffix(".comparison.json"))
    existing_outputs = [str(path) for path in output_targets if path.exists()]
    if existing_outputs:
        raise FileExistsError(
            "refusing to overwrite profile evidence: "
            + ", ".join(existing_outputs)
        )

    import torch
    from mmengine.config import Config

    repo_root = REPO_ROOT
    tracked_tree_clean = _tracked_tree_is_clean(repo_root)
    if not args.allow_random_init and not tracked_tree_clean:
        raise RuntimeError("paper cost profiling requires a clean tracked git tree")
    actual_commit = _git_commit(repo_root)
    trained_commit, evidence_commit = resolve_profile_commit_identities(
        args,
        actual_commit=actual_commit,
    )
    config_path = Path(args.config).expanduser().resolve()
    r5_cell = parse_r5_method_name(args.method_name)
    git_blob_bound_profile = args.method_name in (
        CELLCF_COST_METHODS | DENSE_COST_METHODS
    )
    code_tree_binding: dict[str, Any] | None = None
    profile_config_git_binding: dict[str, Any] | None = None
    execution_model_repo = repo_root
    if git_blob_bound_profile:
        trained_repo_root = _git_repo_root(config_path.parent)
        try:
            config_relative_path = config_path.relative_to(
                trained_repo_root
            ).as_posix()
        except ValueError as exc:
            raise ValueError(
                "formal profile config escaped its trained repository"
            ) from exc
        if _git_commit(trained_repo_root) != trained_commit:
            raise ValueError(
                "formal profile config repository is not at the trained commit"
            )
        if not _tracked_tree_is_clean(trained_repo_root):
            raise ValueError(
                "formal profile config repository is not clean"
            )
        if _ignored_python_sources(trained_repo_root):
            raise ValueError(
                "ignored Python source could shadow the trained config repository"
            )
        trained_model_tree = _git_tree_oid(
            trained_repo_root, trained_commit, "opentad"
        )
        evidence_model_tree = _git_tree_oid(
            repo_root, evidence_commit, "opentad"
        )
        trained_config_tree = _git_tree_oid(
            trained_repo_root,
            trained_commit,
            "configs/adatad/thumos",
        )
        model_trees_equal = trained_model_tree == evidence_model_tree
        if args.method_name in CELLCF_COST_METHODS and not model_trees_equal:
            raise ValueError(
                "trained and evidence commits differ on the inference model tree"
            )
        execution_model_repo = trained_repo_root
        profile_config_sha256 = _sha256_file(config_path)
        profile_config_blob_oid = _git_tree_oid(
            trained_repo_root,
            trained_commit,
            config_relative_path,
        )
        code_tree_binding = {
            "trained_opentad_tree_oid": trained_model_tree,
            "evidence_opentad_tree_oid": evidence_model_tree,
            "trained_adatad_thumos_config_tree_oid": trained_config_tree,
            "model_trees_equal": model_trees_equal,
            "profile_model_loaded_from_trained_repository": True,
            "profile_configs_loaded_from_trained_repository": True,
        }
        profile_config_git_binding = {
            "trained_repository": str(trained_repo_root),
            "trained_commit": trained_commit,
            "relative_path": config_relative_path,
            "git_blob_oid": profile_config_blob_oid,
            "sha256": profile_config_sha256,
            "trained_adatad_thumos_config_tree_oid": trained_config_tree,
        }
    if "opentad" in sys.modules:
        raise RuntimeError(
            "opentad was imported before the formal execution repository was bound"
        )
    sys.path.insert(0, str(execution_model_repo))
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.datasets.base import SlidingWindowDataset
    from opentad.models import build_detector
    import opentad as opentad_package

    loaded_opentad_root = Path(opentad_package.__file__).resolve().parent
    expected_opentad_root = execution_model_repo / "opentad"
    if loaded_opentad_root != expected_opentad_root:
        raise RuntimeError(
            "formal profile imported opentad from a repository other than the bound execution tree"
        )
    if code_tree_binding is not None:
        code_tree_binding["loaded_opentad_root"] = str(loaded_opentad_root)
        code_tree_binding["execution_repository"] = str(execution_model_repo)
    cellcf_cost_binding = None
    trained_checkpoint_binding = None
    r5_cost_binding = None
    if args.post_run_evidence:
        cellcf_cost_binding = load_cellcf_cost_binding(
            args.post_run_evidence,
            args.post_run_evidence_sha256,
            expected_checkpoint_path=args.checkpoint,
            expected_commit=trained_commit,
        )
    cfg = Config.fromfile(str(config_path))
    profile_config_sha256 = _sha256_file(config_path)
    profile_resolved_config_sha256 = _payload_fingerprint(cfg)
    if args.method_name in SINGLECLOCK_COST_METHODS:
        expected_gate_zero = args.method_name.endswith("gate_zero")
        if cfg.model.get("single_clock_admission", False) is not True:
            raise ValueError("formal SingleClock cost config does not admit SingleClock")
        if bool(cfg.model.get("single_clock_gate_zero", False)) != expected_gate_zero:
            raise ValueError("SingleClock cost method/config gate mode mismatch")
    if r5_cell is not None:
        configured_cell = _stable_payload(cfg.get("r5_cell", None))
        _validate_r5_cell_payload(
            configured_cell,
            core=r5_cell,
            require_sampling_contract=True,
        )
        r5_cost_binding = load_r5_terminal_cost_binding(
            method_name=args.method_name,
            config_path=config_path,
            checkpoint_path=args.checkpoint,
            expected_commit=trained_commit,
            matrix_summary_path=os.environ.get("R5_MATRIX_SUMMARY", ""),
            matrix_summary_sha256=os.environ.get("R5_MATRIX_SUMMARY_SHA256", ""),
            mechanism_gate_path=os.environ.get("R5_MECHANISM_GATE_JSON", ""),
            mechanism_gate_sha256=os.environ.get(
                "R5_MECHANISM_GATE_SHA256", ""
            ),
            expected_resolved_config_sha256=profile_resolved_config_sha256,
        )
    if args.checkpoint_evidence:
        trained_checkpoint_binding = load_trained_checkpoint_binding(
            args.checkpoint_evidence,
            args.checkpoint_evidence_sha256,
            expected_role="dense_adatad_baseline",
            expected_commit=trained_commit,
            expected_config_path=config_path,
            expected_config_sha256=profile_config_sha256,
            expected_resolved_config_sha256=profile_resolved_config_sha256,
            expected_checkpoint_path=args.checkpoint,
        )
    if cellcf_cost_binding is not None and args.method_name == "cellcf-fixed384":
        if profile_config_sha256 != cellcf_cost_binding["config_sha256"]:
            raise ValueError("CellCF profile source config differs from post-run evidence")
        if profile_resolved_config_sha256 != cellcf_cost_binding["resolved_config_sha256"]:
            raise ValueError("CellCF profile resolved config differs from post-run evidence")
    if args.backbone_pretrain:
        cfg.model.backbone.custom.pretrain = str(Path(args.backbone_pretrain).expanduser().resolve())
    resource_overrides = (args.video_root, args.annotation, args.class_map)
    if any(resource_overrides) and not all(resource_overrides):
        raise ValueError("--video-root, --annotation and --class-map must be supplied together")
    if all(resource_overrides):
        cfg.dataset.test.data_path = str(Path(args.video_root).expanduser().resolve())
        cfg.dataset.test.ann_file = str(Path(args.annotation).expanduser().resolve())
        cfg.dataset.test.class_map = str(Path(args.class_map).expanduser().resolve())
        cfg.evaluation.ground_truth_filename = str(Path(args.annotation).expanduser().resolve())
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dataset = build_dataset(cfg.dataset.test)
    loader_cfg = dict(cfg.solver.test)
    loader_cfg["batch_size"] = int(args.batch_size)
    loader_cfg["num_workers"] = int(args.loader_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    full_workload_batch_count = int(len(loader))
    if full_workload_batch_count <= 0:
        raise ValueError("official test loader is empty")
    if args.complete_official_workload:
        args.samples = full_workload_batch_count
    cfg.post_processing.sliding_window = isinstance(dataset, SlidingWindowDataset)

    model = build_detector(cfg.model).to(device).eval()
    checkpoint_meta: dict[str, Any] = {
        "checkpoint_path": None,
        "checkpoint_epoch": None,
        "checkpoint_state_key": None,
        "checkpoint_sha256": None,
    }
    if args.checkpoint:
        cost_contract = cfg.get("duca_cellcf_cost_contract", {})
        drop_prefixes = (
            ("frame_selector.",)
            if cost_contract and cost_contract.get("builds_selector") is False
            else ()
        )
        checkpoint_meta = _load_checkpoint(
            model,
            args.checkpoint,
            use_ema=args.use_ema,
            torch_module=torch,
            drop_prefixes=drop_prefixes,
        )
        if cellcf_cost_binding is not None:
            validate_loaded_checkpoint_binding(checkpoint_meta, cellcf_cost_binding)
        if trained_checkpoint_binding is not None:
            for key in (
                "checkpoint_path",
                "checkpoint_sha256",
                "checkpoint_epoch",
                "checkpoint_state_key",
            ):
                if checkpoint_meta.get(key) != trained_checkpoint_binding.get(key):
                    raise ValueError(f"loaded checkpoint differs from generic binding: {key}")
        if r5_cost_binding is not None:
            validate_loaded_checkpoint_binding(checkpoint_meta, r5_cost_binding)
            if (
                checkpoint_meta.get("checkpoint_experiment_metadata_sha256")
                != r5_cost_binding["checkpoint_experiment_metadata_sha256"]
            ):
                raise ValueError(
                    "loaded checkpoint metadata differs from the R5 terminal binding"
                )
        if args.method_name in SINGLECLOCK_COST_METHODS:
            if checkpoint_meta.get("checkpoint_epoch") != 59 or checkpoint_meta.get("checkpoint_state_key") != "state_dict_ema":
                raise ValueError("formal SingleClock cost profile requires epoch-59 state_dict_ema")

    modules, zero_stages = discover_profile_modules(model)
    external_cls = getattr(dataset, "class_map", None)
    use_amp = bool(args.amp)
    iterator = iter(loader)

    def forward_once(batch: Mapping[str, Any], *, record_components: CudaModuleEventHooks | None = None):
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_amp):
            predictions = model.forward_test(
                batch["inputs"],
                batch["masks"],
                batch.get("metas"),
                cfg.inference,
            )
        metas = getattr(model, "_last_forward_test_metas", batch.get("metas"))
        return predictions, metas

    for _ in range(args.warmup_samples):
        cpu_batch, iterator = _next_batch(iterator, loader)
        gpu_batch = _move_to_device(cpu_batch, device, torch)
        predictions, metas = forward_once(gpu_batch)
        model.post_processing(predictions, metas, cfg.post_processing, ext_cls=external_cls)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if args.warmup_samples:
        del cpu_batch, gpu_batch, predictions, metas

    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)
    recorder = StageRecorder(synchronize=synchronize)
    hooks = CudaModuleEventHooks(torch) if device.type == "cuda" else None
    method_hooks = CudaMethodEventHooks(torch) if device.type == "cuda" else None
    if hooks is not None and method_hooks is not None:
        selector = modules.pop("frame_selector_total_ms", None)
        head = modules.pop("head_ms", None)
        if selector is not None:
            method_hooks.register("frame_selector_total_ms", selector, "forward_test")
        if head is not None:
            method_hooks.register("head_ms", head, "forward_test")
        for name, module in modules.items():
            hooks.register(name, module)

    power_gpu_id = _resolve_power_gpu_id(args.power_gpu_id, torch, device) if args.sample_power else None
    power_sampler = (
        ContinuousPowerSampler(gpu_id=power_gpu_id, interval_ms=args.power_interval_ms)
        if args.sample_power
        else None
    )
    if power_sampler is not None:
        power_sampler.start()
        power_sampler.wait_until_ready()

    samples = []
    power_windows = []
    try:
        for _ in range(args.samples):
            recorder.begin_sample()
            with recorder.measure("input_pipeline_serial_ms"):
                cpu_batch, iterator = _next_batch(iterator, loader)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            power_start = time.perf_counter()
            with recorder.measure("h2d_ms"):
                gpu_batch = _move_to_device(cpu_batch, device, torch)
            with recorder.measure("model_forward_ms"):
                predictions, metas = forward_once(gpu_batch, record_components=hooks)
            if hooks is not None:
                hooks.flush_into(recorder)
            if method_hooks is not None:
                method_hooks.flush_into(recorder)
            for stage in zero_stages:
                recorder.record_value(stage, 0.0)
            with recorder.measure("postprocess_ms"):
                model.post_processing(predictions, metas, cfg.post_processing, ext_cls=external_cls)
            power_end = time.perf_counter()
            if device.type == "cuda":
                recorder.record_value(
                    "peak_gpu_memory_mb",
                    torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0),
                )
            else:
                recorder.record_value("peak_gpu_memory_mb", 0.0)
            recorder.record_value("selected_count", _selected_count(model, gpu_batch["inputs"]))
            sample = recorder.end_sample()
            validate_profile_sample(sample, index=len(samples))
            samples.append(sample)
            power_windows.append((power_start, power_end))
            del cpu_batch, gpu_batch, predictions, metas
    finally:
        if power_sampler is not None:
            time.sleep(max(0.01, power_sampler.interval_ms / 1000.0 * 1.5))
            power_sampler.stop()
        if hooks is not None:
            hooks.close()
        if method_hooks is not None:
            method_hooks.close()

    if power_sampler is not None:
        for sample, (start, end) in zip(samples, power_windows):
            try:
                energy = integrate_power_samples(power_sampler.samples, start_time_s=start, end_time_s=end)
            except ValueError:
                sample["gpu_energy_j"] = None
            else:
                sample["gpu_energy_j"] = float(energy["energy_j"])
                sample["average_gpu_power_w"] = float(energy["average_power_w"])

    source_dataset = _stable_payload(cfg.dataset.test)
    if isinstance(source_dataset, dict):
        source_dataset.pop("pipeline", None)
    normalized_cfg = _stable_payload(cfg)
    if args.method_name in SINGLECLOCK_COST_METHODS:
        normalized_cfg["model"]["single_clock_gate_zero"] = False
    metadata = {
        "method": args.method_name,
        "protocol": OFFLINE_FULL_WINDOW_PROTOCOL,
        "hardware_fingerprint": _hardware_fingerprint(torch, device, gpu_uuid=power_gpu_id),
        "host_fingerprint": _host_fingerprint(),
        "software_fingerprint": _software_fingerprint(torch),
        "config_commit": args.config_commit or trained_commit,
        "trained_commit": trained_commit,
        "evidence_git_commit": evidence_commit or actual_commit,
        "inference_code_tree_binding": code_tree_binding,
        "profile_config_git_binding": profile_config_git_binding,
        "profile_session_id": str(args.profile_session_id),
        "profile_pair_id": str(args.profile_pair_id),
        "profile_repeat_index": int(args.profile_repeat_index),
        "profile_order_position": int(args.profile_order_position),
        "profile_config_sha256": profile_config_sha256,
        "profile_resolved_config_sha256": profile_resolved_config_sha256,
        "config_fingerprint": _payload_fingerprint(cfg),
        "gate_zero_normalized_config_fingerprint": _payload_fingerprint(normalized_cfg),
        "single_clock_gate_zero": bool(cfg.model.get("single_clock_gate_zero", False)),
        "dataset_fingerprint": _payload_fingerprint(cfg.dataset.test),
        "source_dataset_fingerprint": _payload_fingerprint(source_dataset),
        "inference_fingerprint": _payload_fingerprint(
            {"inference": cfg.inference, "post_processing": cfg.post_processing}
        ),
        "detector_stack_fingerprint": _detector_stack_fingerprint(model),
        "tracked_tree_clean": bool(tracked_tree_clean),
        "config_path": str(config_path),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "loader_workers": int(args.loader_workers),
        "warmup_samples": int(args.warmup_samples),
        "complete_official_workload": bool(args.complete_official_workload),
        "full_workload_batch_count": full_workload_batch_count,
        "amp": use_amp,
        "uses_ema": bool(args.use_ema),
        "random_init": bool(args.allow_random_init),
        "power_sampling_enabled": bool(args.sample_power),
        "power_interval_ms": int(args.power_interval_ms),
        "power_gpu_id": power_gpu_id if args.sample_power else None,
        **checkpoint_meta,
    }
    if cellcf_cost_binding is not None:
        metadata.update(
            {
                "cellcf_cost_binding": cellcf_cost_binding,
                "cellcf_cost_binding_sha256": _canonical_sha256(cellcf_cost_binding),
                "weight_source": "cellcf_trained_terminal_state_dict_ema",
                "frontend_variant": (
                    "cellcf" if args.method_name == "cellcf-fixed384" else "bare_exact_uniform_lower_bound"
                ),
                "dense_full_stack_savings_claimed": False,
            }
        )
    if trained_checkpoint_binding is not None:
        metadata.update(
            {
                "trained_checkpoint_binding": trained_checkpoint_binding,
                "trained_checkpoint_binding_sha256": _canonical_sha256(
                    trained_checkpoint_binding
                ),
                "weight_source": "dense_adatad_trained_state_dict_ema",
                "frontend_variant": "dense_no_selector",
                "dense_full_stack_savings_claimed": False,
            }
        )
    if r5_cost_binding is not None:
        metadata.update(
            {
                "r5_cost_binding": r5_cost_binding,
                "r5_cost_binding_sha256": _canonical_sha256(r5_cost_binding),
                "weight_source": "r5_terminal_epoch_59_state_dict_ema",
                "frontend_variant": str(r5_cell["arm"]),
                "dense_full_stack_savings_claimed": False,
            }
        )
    report = build_profile_summary(samples, metadata=metadata)
    raw_path = output_prefix.with_suffix(".samples.jsonl")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("x", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    comparison_path = None
    if args.compare_baseline_summary:
        baseline = json.loads(Path(args.compare_baseline_summary).read_text(encoding="utf-8"))
        comparison = compare_profile_summaries(baseline, report)
        comparison_path = output_prefix.with_suffix(".comparison.json")
        with comparison_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    paths = write_profile_artifacts(report, output_prefix)
    outputs: dict[str, Any] = {
        "summary_json": str(paths["json"]),
        "summary_tsv": str(paths["tsv"]),
        "samples_jsonl": str(raw_path),
    }
    if comparison_path is not None:
        outputs["comparison_json"] = str(comparison_path)
    print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
