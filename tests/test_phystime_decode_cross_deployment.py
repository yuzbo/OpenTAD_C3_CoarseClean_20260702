import ast
import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _read(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


def _load_module(relative_path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _submit_function_source():
    source = _read("scripts/submit_phystime_decode_cross_replay.sh")
    start = source.index("record_ambiguous_submission() {")
    end = source.index("\nwrite_header() {", start)
    return source[start:end]


def _run_submit_harness(tmp_path, mode, initial_jobs=""):
    state_path = tmp_path / "jobs.txt"
    calls_path = tmp_path / "sbatch_calls.txt"
    queries_path = tmp_path / "query_calls.txt"
    state_path.write_text(initial_jobs, encoding="utf-8")
    calls_path.unlink(missing_ok=True)
    queries_path.unlink(missing_ok=True)
    script = f"""
set -eu
fail() {{
  echo "FAIL:$*" >&2
  return 1
}}
lookup_jobs_by_comment() {{
  local query_count=0
  if [[ -s "${{QUERIES_FILE}}" ]]; then
    query_count="$(cat "${{QUERIES_FILE}}")"
  fi
  query_count=$((query_count + 1))
  printf '%s\\n' "${{query_count}}" > "${{QUERIES_FILE}}"
  if [[ "${{HARNESS_MODE}}" == "delayed_lost_response" \
        && -s "${{STATE_FILE}}" && "${{query_count}}" -lt 5 ]]; then
    return 0
  fi
  if [[ "${{HARNESS_MODE}}" == "invisible_lost_response" \
        && -s "${{STATE_FILE}}" ]]; then
    return 0
  fi
  if [[ -s "${{STATE_FILE}}" ]]; then
    cat "${{STATE_FILE}}"
  fi
}}
sbatch() {{
  echo called >> "${{CALLS_FILE}}"
  case "${{HARNESS_MODE}}" in
    success)
      printf '4321\\n' > "${{STATE_FILE}}"
      printf '4321\\n'
      return 0
      ;;
    lost_response)
      printf '4321\\n' > "${{STATE_FILE}}"
      echo "transport lost" >&2
      return 1
      ;;
    delayed_lost_response)
      printf '4321\\n' > "${{STATE_FILE}}"
      echo "transport lost before scheduler visibility" >&2
      return 1
      ;;
    invisible_lost_response)
      printf '4321\\n' > "${{STATE_FILE}}"
      echo "transport lost beyond visibility budget" >&2
      return 1
      ;;
    duplicate)
      printf '4321\\n4322\\n' > "${{STATE_FILE}}"
      printf '4321\\n'
      return 0
      ;;
    *)
      return 99
      ;;
  esac
}}
sleep() {{ :; }}
{_submit_function_source()}
PHYSTIME_SUBMIT_VISIBILITY_POLLS=6
PHYSTIME_SUBMIT_VISIBILITY_DELAY_SEC=0
DAG_TOKEN=test-token
RUN_ROOT="${{TEST_RUN_ROOT}}"
AMBIGUOUS_SUBMISSION_ROOT="${{RUN_ROOT}}/submission_attempts"
COMMIT=test-commit
TREE=test-tree
PYTHON=python
mkdir -p "${{AMBIGUOUS_SUBMISSION_ROOT}}"
submit selected_online fake.sbatch
"""
    env = os.environ.copy()
    env.update(
        {
            "STATE_FILE": state_path.as_posix(),
            "CALLS_FILE": calls_path.as_posix(),
            "QUERIES_FILE": queries_path.as_posix(),
            "HARNESS_MODE": mode,
            "TEST_RUN_ROOT": tmp_path.as_posix(),
        }
    )
    completed = subprocess.run(
        ["bash", "-c", script],
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
        check=False,
    )
    calls = (
        calls_path.read_text(encoding="utf-8").splitlines()
        if calls_path.is_file()
        else []
    )
    return completed, calls


def test_decode_cross_configs_are_capture_only_overlays():
    cases = (
        (
            "phystime_g1a_selected_axis_native_j192_decode_replay.py",
            "phystime_g1a_selected_axis_native_j192_p0_replay.py",
            "uniform_rank_seconds",
        ),
        (
            "phystime_g1a_physical_metric_native_j192_decode_replay.py",
            "phystime_g1a_physical_metric_native_j192_p0_replay.py",
            "physical_time_seconds",
        ),
    )
    for filename, base, axis in cases:
        source = _read(f"configs/adatad/thumos/{filename}")
        assert ast.parse(source) is not None
        assert base in source
        assert "phystime_decode_replay_capture" in source
        assert f'train_axis="{axis}"' in source
        assert f'expected_native_coordinate_mode="{axis}"' in source
        assert 'weights_source="must_be_overridden"' in source
        assert "max_in_memory_bytes=8589934592" in source
        assert "scheduler" not in source
        assert "workflow" not in source


def test_submitter_is_one_gate_four_frozen_replays_one_suite():
    source = _read("scripts/submit_phystime_decode_cross_replay.sh")
    assert "SOURCE_COMMIT=\"${PHYSTIME_SOURCE_COMMIT:-0dc5851" in source
    assert "SOURCE_TREE=\"${PHYSTIME_SOURCE_TREE:-bddc9b9" in source
    assert "P0_SUITE_COMPLETE.json" in source
    assert 'gate_job="$(submit decode_cross_gate' in source
    assert (
        'submit "${variant}" --dependency="afterok:${gate_job}"'
        in source
    )
    for variant in (
        "selected_online",
        "selected_ema",
        "physical_online",
        "physical_ema",
    ):
        assert variant in source
    assert (
        'afterok:${jobs[selected_online]}:${jobs[selected_ema]}:'
        '${jobs[physical_online]}:${jobs[physical_ema]}'
    ) in source
    assert '"new_training": False' in source
    assert '"frozen_epoch": 59' in source
    assert '"native_exact_equivalence_required": True' in source
    assert "tools/train.py" not in source
    assert "echo '#SBATCH --gpus=1'" in source
    assert "echo '#SBATCH --mem=32G'" in source
    assert "--gres=gpu:1" not in source
    assert 'gate["dataset_manifest"]' in source
    assert (
        'ANNOTATION="${SOURCE_ANNOTATION}"' in source
        and 'CLASS_MAP="${SOURCE_CLASS_MAP}"' in source
        and 'VIDEOMAE_CHECKPOINT="${SOURCE_VIDEOMAE}"' in source
    )
    assert "OPENTAD_THUMOS14_ANNOTATION:-" not in source
    assert "PHYSTIME_VIDEOMAE_CHECKPOINT:-" not in source
    assert "${BASE}/raw/Validation Data/validation" not in source
    assert "${BASE}/raw/Test Data/TH14_test_set_mp4" not in source
    assert "P0_SUITE_SHA256" in source
    assert "jobs_tsv_sha256" in source
    assert "sbatch_sha256" in source
    assert "cancel_partial_submission" in source
    assert "preflight_phystime_decode_cross.py" in source
    assert "PREFLIGHT_SHA256" in source
    assert "--comment=\"${comment}\"" in source
    assert "lookup_jobs_by_comment" in source
    assert "uuid.uuid4().hex" in source
    assert "submission_owner.json" in source
    assert "claim_phystime_decode_cross_owner.py" in source
    assert "GLOBAL_OWNER_MANIFEST=" in source
    assert "global_submission_owner_manifest_sha256" in source
    assert "flock -n 9" in source
    assert "RECOVERY_MODE=" in source
    assert "PHYSTIME_SUBMIT_VISIBILITY_POLLS" in source
    assert "manage_phystime_decode_cross_submission_state.py" in source
    assert '"before_sbatch"' in source
    assert "refusing automatic resubmission" in source
    assert "PHYSTIME_SUBMIT_RETRIES" not in source
    assert "capture_phystime_decode_cross_scheduler.py" in source
    assert "scheduler_submission_sha256" in source


def test_replay_runner_captures_once_then_replays_both_axes():
    source = _read("scripts/run_phystime_decode_cross_replay_slurm.sh")
    assert source.count("tools/test.py") == 1
    assert "tools/train.py" not in source
    assert "replay_phystime_decode_cross.py" in source
    assert "validate_phystime_decode_cross_replay.py" in source
    assert "DECODE_CROSS_COMPLETE.json" in source
    assert "DIRECT_INFERENCE_COMPLETE" in source
    assert '"solver.ema=${USE_EMA}"' in source
    assert '"solver_ema": use_ema == "True"' in source
    assert '"new_training": False' in source
    assert '"shared_raw_tensor_dual_decode": True' in source
    producer = _read("tools/bata/replay_phystime_decode_cross.py")
    assert '"uses_captured_production_proposals": False' in producer
    assert "decoded = native_proposals" not in producer
    assert source.index("preflight_phystime_decode_cross.py") < source.index(
        "manifest = {"
    )
    assert "runtime_preflight_manifest_sha256" in source


@pytest.mark.skipif(os.name == "nt", reason="requires native Linux bash")
def test_submit_adopts_a_job_when_sbatch_response_is_lost(tmp_path):
    completed, calls = _run_submit_harness(
        tmp_path,
        "lost_response",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "4321"
    assert calls == ["called"]


@pytest.mark.skipif(os.name == "nt", reason="requires native Linux bash")
def test_submit_waits_for_delayed_visibility_without_duplicate_sbatch(tmp_path):
    completed, calls = _run_submit_harness(
        tmp_path,
        "delayed_lost_response",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "4321"
    assert calls == ["called"]


@pytest.mark.skipif(os.name == "nt", reason="requires native Linux bash")
def test_submit_never_retries_when_visibility_budget_is_exhausted(tmp_path):
    first, first_calls = _run_submit_harness(
        tmp_path,
        "invisible_lost_response",
    )
    assert first.returncode != 0
    assert "refusing automatic resubmission" in first.stderr
    assert first_calls == ["called"]
    ambiguous = (
        tmp_path
        / "submission_attempts"
        / "selected_online.ambiguous.json"
    )
    assert ambiguous.is_file()

    blocked, blocked_calls = _run_submit_harness(tmp_path, "success")
    assert blocked.returncode != 0
    assert "recovery is query-only" in blocked.stderr
    assert blocked_calls == []

    recovered, recovered_calls = _run_submit_harness(
        tmp_path,
        "success",
        initial_jobs="4321\n",
    )
    assert recovered.returncode == 0, recovered.stderr
    assert recovered.stdout.strip() == "4321"
    assert recovered_calls == []
    assert not ambiguous.exists()
    assert (
        tmp_path
        / "submission_attempts"
        / "selected_online.resolved.json"
    ).is_file()
    invisible, invisible_calls = _run_submit_harness(
        tmp_path,
        "success",
        initial_jobs="",
    )
    assert invisible.returncode != 0
    assert "temporarily invisible" in invisible.stderr
    assert invisible_calls == []


@pytest.mark.skipif(os.name == "nt", reason="requires native Linux bash")
def test_submit_rejects_duplicate_exact_comments_after_sbatch(tmp_path):
    completed, calls = _run_submit_harness(tmp_path, "duplicate")
    assert completed.returncode != 0
    assert "ambiguous accepted jobs" in completed.stderr
    assert calls == ["called"]
    assert not (
        tmp_path
        / "submission_attempts"
        / "selected_online.ambiguous.json"
    ).exists()
    assert (
        tmp_path
        / "submission_attempts"
        / "selected_online.fatal.json"
    ).is_file()
    blocked, blocked_calls = _run_submit_harness(
        tmp_path,
        "success",
        initial_jobs="",
    )
    assert blocked.returncode != 0
    assert "persistent fatal submission state" in blocked.stderr
    assert blocked_calls == []


@pytest.mark.skipif(os.name == "nt", reason="requires native Linux bash")
def test_submit_recovery_adopts_existing_unique_job_without_sbatch(tmp_path):
    completed, calls = _run_submit_harness(
        tmp_path,
        "success",
        initial_jobs="5555\n",
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "5555"
    assert calls == []


def test_global_owner_permanently_binds_token_to_one_run_root(tmp_path):
    owner_tool = _load_module(
        "tools/bata/claim_phystime_decode_cross_owner.py",
        "decode_cross_owner_test",
    )
    global_owner = tmp_path / "locks" / "token.owner.json"
    first_local = tmp_path / "run-a" / "submission_owner.json"
    first = owner_tool.claim_submission_ownership(
        global_owner_path=global_owner,
        local_owner_path=first_local,
        run_root=tmp_path / "run-a",
        dag_token="token",
        runtime_commit="commit",
        runtime_tree="tree",
        run_uuid="a" * 32,
    )
    assert first["recovery_mode"] is False
    original_global = global_owner.read_bytes()

    recovered = owner_tool.claim_submission_ownership(
        global_owner_path=global_owner,
        local_owner_path=first_local,
        run_root=tmp_path / "run-a",
        dag_token="token",
        runtime_commit="commit",
        runtime_tree="tree",
        run_uuid="b" * 32,
    )
    assert recovered["recovery_mode"] is True
    assert recovered["run_uuid"] == "a" * 32

    with pytest.raises(RuntimeError, match="run_root"):
        owner_tool.claim_submission_ownership(
            global_owner_path=global_owner,
            local_owner_path=tmp_path / "run-b" / "submission_owner.json",
            run_root=tmp_path / "run-b",
            dag_token="token",
            runtime_commit="commit",
            runtime_tree="tree",
            run_uuid="c" * 32,
        )
    assert global_owner.read_bytes() == original_global
    assert not (tmp_path / "run-b" / "submission_owner.json").exists()


def test_submission_attempt_requires_visible_matching_job_to_resolve(tmp_path):
    state_tool = _load_module(
        "tools/bata/manage_phystime_decode_cross_submission_state.py",
        "decode_cross_submission_state_test",
    )
    ambiguous = tmp_path / "selected_online.ambiguous.json"
    resolved = tmp_path / "selected_online.resolved.json"
    kwargs = {
        "output_path": ambiguous,
        "run_root": tmp_path / "run",
        "dag_token": "token",
        "variant": "selected_online",
        "comment": "token:selected_online",
        "runtime_commit": "commit",
        "runtime_tree": "tree",
    }
    state_tool.record_attempt(
        **kwargs,
        phase="before_sbatch",
    )
    state_tool.record_attempt(
        **kwargs,
        phase="numeric_response",
        expected_job_id="4321",
        sbatch_output="4321",
    )
    with pytest.raises(RuntimeError, match="differs"):
        state_tool.resolve_attempt(
            ambiguous_path=ambiguous,
            resolved_path=resolved,
            job_id="9999",
        )
    assert ambiguous.is_file()
    assert not resolved.exists()

    payload = state_tool.resolve_attempt(
        ambiguous_path=ambiguous,
        resolved_path=resolved,
        job_id="4321",
    )
    assert payload["state"] == "resolved"
    assert payload["resolved_job_id"] == "4321"
    assert not ambiguous.exists()
    assert resolved.is_file()
    assert (
        state_tool.inspect_resolved_attempt(
            resolved_path=resolved,
            run_root=tmp_path / "run",
            dag_token="token",
            variant="selected_online",
            comment="token:selected_online",
            runtime_commit="commit",
            runtime_tree="tree",
        )
        == "4321"
    )


def test_gate_and_suite_are_fail_closed():
    gate = _read("scripts/run_phystime_decode_cross_gate_slurm.sh")
    suite = _read("scripts/run_phystime_decode_cross_suite_slurm.sh")
    gate_python = _read("tools/bata/run_phystime_decode_cross_gate.py")
    suite_python = _read(
        "tools/bata/validate_phystime_decode_cross_suite.py"
    )
    assert "test_phystime_decode_cross_replay.py" in gate
    assert "test_phystime_decode_cross_deployment.py" in gate
    assert "run_phystime_decode_cross_gate.py" in gate
    assert "native_direct_exact_equivalence" in gate
    assert "all_native_direct_exact_equivalence" in gate
    assert "raw_tensors_immutable" in gate
    assert "build_dataset_manifest" in gate_python
    assert "P0_DATASET_MANIFEST_SHA256" in gate_python
    assert "P0_VIDEOMAE_SHA256" in gate_python
    assert "state_dict_sha256" in gate_python
    assert "tensor.reshape(-1)" in gate_python
    for condition in (
        "selected_online",
        "selected_ema",
        "physical_online",
        "physical_ema",
    ):
        assert condition in gate_python
    assert "DECODE_CROSS_COMPLETE.json" in suite
    assert "validate_phystime_decode_cross_suite.py" in suite
    assert "DECODE_CROSS_SUITE_COMPLETE.json" in suite
    assert "--preflight-manifest" in gate
    assert "scheduler_terminal.json" in suite
    assert '"*.ambiguous.json", "*.fatal.json"' in suite_python
    assert "resolved submission marker set differs from jobs TSV" in suite_python
    assert 'marker.get("resolved_job_id", "")' in suite_python


def test_source_validation_returns_a_real_provenance_report(tmp_path):
    source = _read("tools/bata/run_phystime_decode_cross_gate.py")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_source_dir"
    )
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace = {
        "Path": Path,
        "SOURCE_COMMIT": "source-commit",
        "SOURCE_TREE": "source-tree",
    }
    source_dir = tmp_path / "selected"
    source_dir.mkdir()
    gate_path = tmp_path / "gate.json"
    gate_path.write_text("{}", encoding="utf-8")
    payloads = {
        "FULL_COMPLETE.json": {
            "validation_pass": True,
            "artifacts": {"checkpoint": {"sha256": "checkpoint-sha"}},
        },
        "run_manifest.json": {
            "commit": "source-commit",
            "git_tree": "source-tree",
            "variant": "selected_axis",
            "dataset_manifest_sha256": "dataset-sha",
            "pretrained_checkpoint_sha256": "videomae-sha",
            "g1a_gate": str(gate_path),
        },
        "gate.json": {
            "gate_pass": True,
            "dataset_manifest_sha256": "dataset-sha",
        },
    }

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def read_json(path, _description):
        return payloads[Path(path).name]

    namespace.update(
        {
            "require": require,
            "read_json": read_json,
            "sha256_file": lambda path: f"sha:{Path(path).name}",
        }
    )
    exec(compile(isolated, "<validate_source_dir>", "exec"), namespace)
    report = namespace["validate_source_dir"](
        source_dir,
        "selected_axis",
        "checkpoint-sha",
        "dataset-sha",
        "videomae-sha",
    )
    assert isinstance(report, dict)
    assert report["path"] == str(source_dir.resolve())
    assert report["source_gate"]["path"] == str(gate_path.resolve())
    assert report["source_gate"]["sha256"] == "sha:gate.json"


def test_scheduler_submission_snapshot_checks_real_identity(monkeypatch):
    scheduler = _load_module(
        "tools/bata/capture_phystime_decode_cross_scheduler.py",
        "decode_cross_scheduler_test",
    )
    stdout = "/tmp/phystime/job.out"
    stderr = "/tmp/phystime/job.err"
    record = {
        "variant": "selected_online",
        "job_id": "12345",
        "job_name": "pt_dc_selected_online",
        "comment": "token:selected_online",
        "dependency": "afterok:12344",
        "stdout": stdout,
        "stderr": stderr,
    }
    line = (
        "JobId=12345 JobName=pt_dc_selected_online "
        "Comment=token:selected_online "
        "Dependency=afterok:12344(unfulfilled) "
        f"StdOut={stdout} StdErr={stderr}"
    )
    monkeypatch.setattr(
        scheduler.subprocess,
        "check_output",
        lambda *args, **kwargs: line,
    )
    report = scheduler.capture_submission([record], "token")
    assert report["selected_online"]["job_id"] == "12345"
    assert report["selected_online"]["dependency"].startswith("afterok:12344")


def test_preflight_recomputes_content_before_submission():
    preflight = _read("tools/bata/preflight_phystime_decode_cross.py")
    submitter = _read("scripts/submit_phystime_decode_cross_replay.sh")
    for term in (
        "build_dataset_manifest",
        "checkpoint_file_report",
        "validate_source_dir",
        "validate_p0",
        "P0_DATASET_MANIFEST_SHA256",
        "P0_VIDEOMAE_SHA256",
    ):
        assert term in preflight
    assert submitter.index("preflight_phystime_decode_cross.py") < (
        submitter.index('gate_job="$(submit decode_cross_gate')
    )


def test_real_gate_rejects_swapped_states_and_axis_contracts():
    source = _read("tools/bata/run_phystime_decode_cross_gate.py")
    module = ast.parse(source)
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "validate_real_window_contracts"
    )
    isolated = ast.Module(body=[function], type_ignores=[])

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    expected = {
        "selected_online": ("selected_axis", "online"),
        "selected_ema": ("selected_axis", "ema"),
        "physical_online": ("physical_metric", "online"),
        "physical_ema": ("physical_metric", "ema"),
    }
    namespace = {"require": require, "EXPECTED_CONDITIONS": expected}
    exec(compile(isolated, "<real_window_contracts>", "exec"), namespace)
    validate = namespace["validate_real_window_contracts"]
    checkpoint_reports = {
        "selected_axis": {
            "online_state_dict_sha256": "s-online",
            "ema_state_dict_sha256": "s-ema",
        },
        "physical_metric": {
            "online_state_dict_sha256": "p-online",
            "ema_state_dict_sha256": "p-ema",
        },
    }
    contract = {
        "observation_sequence_sha256": "obs",
        "uniform_axis_sha256": "u",
        "physical_axis_sha256": "p",
    }
    selected_axis_contract = {
        "native_axis": "uniform_rank_seconds",
        "window_sequence_sha256": "selected-window",
    }
    physical_axis_contract = {
        "native_axis": "physical_time_seconds",
        "window_sequence_sha256": "physical-window",
    }
    windows = {
        "selected_online": {
            "checkpoint_state_dict_sha256": "s-online",
            "observation_contract": dict(contract),
            "axis_window_contract": dict(selected_axis_contract),
        },
        "selected_ema": {
            "checkpoint_state_dict_sha256": "s-ema",
            "observation_contract": dict(contract),
            "axis_window_contract": dict(selected_axis_contract),
        },
        "physical_online": {
            "checkpoint_state_dict_sha256": "p-online",
            "observation_contract": dict(contract),
            "axis_window_contract": dict(physical_axis_contract),
        },
        "physical_ema": {
            "checkpoint_state_dict_sha256": "p-ema",
            "observation_contract": dict(contract),
            "axis_window_contract": dict(physical_axis_contract),
        },
    }
    assert validate(windows, checkpoint_reports) == contract

    swapped = {key: dict(value) for key, value in windows.items()}
    swapped["selected_online"]["checkpoint_state_dict_sha256"] = "s-ema"
    try:
        validate(swapped, checkpoint_reports)
    except ValueError as error:
        assert "wrong checkpoint state" in str(error)
    else:
        raise AssertionError("swapped online/EMA state was accepted")

    changed_axis = {key: dict(value) for key, value in windows.items()}
    changed_axis["physical_ema"]["observation_contract"] = {
        **contract,
        "physical_axis_sha256": "changed",
    }
    try:
        validate(changed_axis, checkpoint_reports)
    except ValueError as error:
        assert "observation contract differs" in str(error)
    else:
        raise AssertionError("changed physical axis contract was accepted")

    changed_window = {
        key: {
            **value,
            "axis_window_contract": dict(value["axis_window_contract"]),
        }
        for key, value in windows.items()
    }
    changed_window["physical_ema"]["axis_window_contract"][
        "window_sequence_sha256"
    ] = "changed-window"
    try:
        validate(changed_window, checkpoint_reports)
    except ValueError as error:
        assert "online/EMA axis window contract differs" in str(error)
    else:
        raise AssertionError("changed axis-specific window hash was accepted")


def test_recompute_validator_does_not_import_replay_producer():
    source = _read(
        "tools/bata/validate_phystime_decode_cross_replay.py"
    )
    assert "from tools.bata.replay_phystime_decode_cross" not in source
    assert "native_direct_exact_equivalence" in source
    assert "production_semantic_recompute_evaluate" in source
    assert "recomputed_metrics" in source


def test_suite_exposes_frozen_intervention_and_descriptive_differences():
    source = _read("tools/bata/validate_phystime_decode_cross_suite.py")
    for term in (
        "within_checkpoint_physical_decode_minus_uniform_decode",
        "fixed_decode_cross_checkpoint_descriptive_difference",
        "descriptive_difference_in_differences",
        "weight_source_ema_minus_online",
    ):
        assert term in source
    assert "DECODE_CROSS_COMPLETE.json" in source
    assert "p0_suite_completion" in source
    assert "shared_observation_contract" in source
    assert "observation_sequence_sha256" in source
    assert "uniform_axis_sha256" in source
    assert "physical_axis_sha256" in source
