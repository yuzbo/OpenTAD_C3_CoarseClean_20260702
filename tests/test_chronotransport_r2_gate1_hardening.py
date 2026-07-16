import copy
import inspect
from pathlib import Path
import subprocess
import sys

import pytest


def _retag_and_rehash(
    artifact: dict, *, schema: str, digest_field: str
) -> dict:
    from opentad.models.chronotransport.protocol import canonical_sha256

    forged = copy.deepcopy(artifact)
    forged["schema"] = schema
    forged.pop(digest_field, None)
    forged[digest_field] = canonical_sha256(forged)
    return forged


def _git(repo: Path, *args: str, input_text: str | None = None) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=input_text,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _init_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "ct@test.invalid")
    _git(repo, "config", "user.name", "CT Test")
    (repo / "base.txt").write_text("I\n", encoding="utf-8")
    _git(repo, "add", "base.txt")
    _git(repo, "commit", "-qm", "I")
    return repo, _git(repo, "rev-parse", "HEAD")


def _commit_regular_registration(
    repo: Path, relative: str, payload: bytes
) -> tuple[Path, str]:
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    _git(repo, "add", relative)
    _git(repo, "commit", "-qm", "R")
    return path, _git(repo, "rev-parse", "HEAD")


def test_every_formal_gate1_public_api_requires_repository_context():
    from opentad.models.chronotransport.adjudication import (
        build_gate1_paired_replay_artifact,
        build_gate1_record_artifact,
        gate1_oracle_headroom_from_profile,
        validate_gate1_paired_replay_artifact,
        validate_gate1_record_artifact,
    )
    from opentad.models.chronotransport.full_stack_profiler import (
        build_full_stack_profile_artifact,
        validate_full_stack_profile_artifact,
    )
    from opentad.models.chronotransport.gate1_unlock import (
        build_gate1_unlock_artifact,
        validate_gate1_unlock_artifact,
    )
    from opentad.models.chronotransport.replay import (
        run_registered_gate1_paired_replay,
    )

    callbacks = (
        build_full_stack_profile_artifact,
        validate_full_stack_profile_artifact,
        run_registered_gate1_paired_replay,
        build_gate1_paired_replay_artifact,
        validate_gate1_paired_replay_artifact,
        build_gate1_record_artifact,
        validate_gate1_record_artifact,
        gate1_oracle_headroom_from_profile,
        build_gate1_unlock_artifact,
        validate_gate1_unlock_artifact,
    )
    required = ("repository_root", "registration_commit", "registration_relpath")
    for callback in callbacks:
        parameters = inspect.signature(callback).parameters
        assert all(name in parameters for name in required), callback.__name__
        assert all(parameters[name].default is inspect.Parameter.empty for name in required)

    with pytest.raises((TypeError, ValueError), match="repository_root"):
        build_gate1_unlock_artifact(
            {},
            repository_root=None,
            registration_commit="a" * 40,
            registration_relpath="artifacts/registration.json",
        )


def test_formal_profile_and_replay_have_no_importable_caller_input_issuer():
    import opentad.models.chronotransport.full_stack_profiler as profile
    import opentad.models.chronotransport.adjudication as adjudication
    import opentad.models.chronotransport.replay as replay
    from opentad.models.chronotransport.adjudication import (
        build_gate1_paired_replay_artifact,
    )

    for name in (
        "_FIXED_SESSION_TOKEN",
        "_ATTESTED_PROFILE_TOKEN",
        "FixedProfileSessionResult",
        "AttestedFullStackProfileArtifact",
        "_issue_fixed_profile_session_result",
        "_attest_serialized_full_stack_profile_artifact",
        "_build_full_stack_profile_artifact_from_validated",
    ):
        assert not hasattr(profile, name), name
    for name in (
        "_GATE1_RUNNER_TOKEN",
        "Gate1PairedRunnerResult",
        "consume_gate1_paired_runner_result",
        "_run_registered_gate1_paired_replay_from_materialized",
    ):
        assert not hasattr(replay, name), name
    assert "runner_output" not in inspect.signature(
        build_gate1_paired_replay_artifact
    ).parameters
    assert not hasattr(
        adjudication, "_build_gate1_paired_replay_artifact_from_rows"
    )


def test_formal_profile_and_replay_expose_no_arbitrary_execution_boundary():
    import opentad.models.chronotransport as chronotransport
    import opentad.models.chronotransport.full_stack_profiler as profile
    import opentad.models.chronotransport.replay as replay
    import tools.bata.chronotransport_r2_profile_factory as profile_factory
    from tools.bata.chronotransport_r2_gate1_replay_factory import (
        RegisteredGate1ReplaySession,
    )

    assert "profile_full_stack_callable" not in chronotransport.__all__
    assert not hasattr(profile, "profile_full_stack_callable")
    assert not hasattr(profile_factory, "RegisteredProfileInvocation")
    assert not hasattr(
        profile_factory.RegisteredOpenTADProfileSession, "build_invocation"
    )
    assert not hasattr(replay, "_execute_gate1_paired_replay_rows")

    fake_profile = object.__new__(
        profile_factory.RegisteredOpenTADProfileSession
    )
    fake_profile._registration = {"profiler": {"candidate_plan": []}}
    fake_profile._backend = object()
    with pytest.raises(TypeError, match="fixed.*backend|repository-owned"):
        fake_profile.run_fixed_profile()

    fake_replay = object.__new__(RegisteredGate1ReplaySession)
    fake_replay._registration = {
        "window_manifest": {
            "artifact": {"splits": {"calibration": [], "evaluation": []}}
        }
    }
    fake_replay._backend = object()
    with pytest.raises(TypeError, match="fixed.*backend|repository-owned"):
        fake_replay.run_split("calibration")


def test_profile_fixture_cannot_be_retagged_and_rehashed_as_formal(monkeypatch):
    import opentad.models.chronotransport.full_stack_profiler as profile
    from test_chronotransport_r2_gate1_cost_profile import (
        _profile_artifact,
        _registration,
    )

    registration = _registration()
    fixture = _profile_artifact(registration)
    assert "candidates" not in fixture and "profile_sha256" not in fixture
    candidate = fixture["fixture_candidates"][0]
    assert "provenance" not in candidate and "invocations" not in candidate
    invocation = candidate["fixture_invocations"][0]
    assert not {
        "invocation_id",
        "total_ms",
        "cost_ledger",
        "execution_provenance",
    } & set(invocation)
    forged = _retag_and_rehash(
        fixture,
        schema=profile.PROFILE_ARTIFACT_SCHEMA,
        digest_field="profile_sha256",
    )
    monkeypatch.setattr(
        profile,
        "validate_formal_gate1_context",
        lambda registration, **kwargs: registration,
    )
    with pytest.raises((TypeError, ValueError), match="formal|fields|candidate|artifact"):
        profile.validate_full_stack_profile_artifact(
            forged,
            registration=registration,
            repository_root="/repo",
            registration_commit="a" * 40,
            registration_relpath="artifacts/registration.json",
        )


def test_replay_and_record_fixtures_cannot_be_retagged_as_formal(monkeypatch):
    import opentad.models.chronotransport.adjudication as adjudication
    from test_chronotransport_r2_gate1_cost_profile import _records, _registration

    registration = _registration()
    record_fixture = _records(registration, "calibration", 0.2)
    replay_fixture = (
        record_fixture["paired_replay"]
        if "paired_replay" in record_fixture
        else record_fixture["fixture_paired_replay"]
    )
    assert "rows" not in replay_fixture and "artifact_sha256" not in replay_fixture
    assert set(replay_fixture["fixture_rows"][0]) == {
        "fixture_window_id",
        "fixture_source",
        "fixture_derived_detector_regret",
    }
    assert not hasattr(adjudication, "_validate_replay_row")
    fixture_replay_row = replay_fixture["fixture_rows"][0]
    raw_replay_row = {
        "window_id": fixture_replay_row["fixture_window_id"],
        **fixture_replay_row["fixture_source"],
    }
    invocation_index = registration["profiler"]["invocation_ids"].index(
        raw_replay_row["window_id"]
    )
    with pytest.raises(ValueError, match="serialized regret"):
        adjudication._validate_serialized_formal_replay_row(
            raw_replay_row,
            registration=registration,
            window_id=raw_replay_row["window_id"],
            invocation_index=invocation_index,
        )
    serialized_replay_row = {
        **raw_replay_row,
        "detector_regret": fixture_replay_row[
            "fixture_derived_detector_regret"
        ],
    }
    assert adjudication._validate_serialized_formal_replay_row(
        serialized_replay_row,
        registration=registration,
        window_id=raw_replay_row["window_id"],
        invocation_index=invocation_index,
    ) == serialized_replay_row
    assert "rows" not in record_fixture and "artifact_sha256" not in record_fixture
    assert set(record_fixture["fixture_rows"][0]) == {
        "fixture_window_id",
        "fixture_candidate_names",
        "fixture_detector_regret",
    }
    forged_replay = _retag_and_rehash(
        replay_fixture,
        schema=adjudication.GATE1_PAIRED_REPLAY_SCHEMA,
        digest_field="artifact_sha256",
    )
    monkeypatch.setattr(
        adjudication,
        "validate_formal_gate1_context",
        lambda registration, **kwargs: registration,
    )
    context = {
        "repository_root": "/repo",
        "registration_commit": "a" * 40,
        "registration_relpath": "artifacts/registration.json",
    }
    with pytest.raises((TypeError, ValueError), match="formal|fields|rows|artifact"):
        adjudication.validate_gate1_paired_replay_artifact(
            forged_replay,
            registration=registration,
            expected_split="calibration",
            **context,
        )


def test_formal_replay_binds_exact_observed_allocation_identity(monkeypatch):
    import opentad.models.chronotransport.adjudication as adjudication
    from opentad.models.chronotransport.environment import (
        build_test_only_observed_environment,
    )
    from opentad.models.chronotransport.protocol import canonical_sha256
    from test_chronotransport_r2_gate1_cost_profile import _records, _registration

    registration = _registration()
    record_fixture = _records(registration, "calibration", 0.2)
    replay_fixture = record_fixture["fixture_paired_replay"]
    rows = [
        {
            "window_id": row["fixture_window_id"],
            **row["fixture_source"],
            "detector_regret": row["fixture_derived_detector_regret"],
        }
        for row in replay_fixture["fixture_rows"]
    ]
    expected_ids = registration["window_manifest"]["artifact"]["splits"][
        "calibration"
    ]
    artifact = {
        "schema": adjudication.GATE1_PAIRED_REPLAY_SCHEMA,
        "registration_sha256": registration["registration_sha256"],
        "observed_environment": build_test_only_observed_environment(
            registration["environment"]
        ),
        "split": "calibration",
        "window_ids": list(expected_ids),
        "window_order_sha256": canonical_sha256(expected_ids),
        "candidate_names": list(adjudication.GATE1_PAIRED_CANDIDATE_ORDER),
        "candidate_order_sha256": canonical_sha256(
            adjudication.GATE1_PAIRED_CANDIDATE_ORDER
        ),
        "order_probe_candidate_names": list(
            reversed(adjudication.GATE1_PAIRED_CANDIDATE_ORDER)
        ),
        "order_probe_candidate_order_sha256": canonical_sha256(
            tuple(reversed(adjudication.GATE1_PAIRED_CANDIDATE_ORDER))
        ),
        "rows": rows,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    monkeypatch.setattr(
        adjudication,
        "validate_formal_gate1_context",
        lambda registration, **kwargs: registration,
    )
    context = {
        "repository_root": "/repo",
        "registration_commit": "a" * 40,
        "registration_relpath": "artifacts/registration.json",
    }
    assert adjudication.validate_gate1_paired_replay_artifact(
        artifact,
        registration=registration,
        expected_split="calibration",
        **context,
    ) == artifact

    damaged = copy.deepcopy(artifact)
    damaged["observed_environment"]["slurm_step_id"] = "different"
    damaged["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in damaged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(RuntimeError, match="allocation identity"):
        adjudication.validate_gate1_paired_replay_artifact(
            damaged,
            registration=registration,
            expected_split="calibration",
            **context,
        )

    forged_record = _retag_and_rehash(
        record_fixture,
        schema=adjudication.GATE1_RECORD_SCHEMA,
        digest_field="artifact_sha256",
    )
    with pytest.raises((TypeError, ValueError), match="formal|fields|record|artifact"):
        adjudication.validate_gate1_record_artifact(
            forged_record,
            registration=registration,
            expected_split="calibration",
            **context,
        )


def test_registration_commit_shape_binds_current_regular_file_and_expected_bytes(
    tmp_path,
):
    from opentad.models.chronotransport.registration import (
        validate_registration_commit_shape,
    )

    payload = b'{"immutable":true}\n'
    relative = "artifacts/registration.json"
    repo, implementation = _init_repo(tmp_path)
    path, registration_commit = _commit_regular_registration(repo, relative, payload)
    validate_registration_commit_shape(
        repository_root=repo,
        registration_commit=registration_commit,
        implementation_commit=implementation,
        registration_relpath=relative,
        registration_bytes=payload,
    )

    path.write_bytes(b'{"different":true}\n')
    with pytest.raises(ValueError, match="current registration|exact bytes"):
        validate_registration_commit_shape(
            repository_root=repo,
            registration_commit=registration_commit,
            implementation_commit=implementation,
            registration_relpath=relative,
            registration_bytes=payload,
        )

    path.write_bytes(payload)
    target = repo / "registration-target.json"
    target.write_bytes(payload)
    path.unlink()
    path.symlink_to(target.name)
    with pytest.raises(ValueError, match="regular|symlink"):
        validate_registration_commit_shape(
            repository_root=repo,
            registration_commit=registration_commit,
            implementation_commit=implementation,
            registration_relpath=relative,
            registration_bytes=payload,
        )


def test_registration_commit_shape_rejects_non_regular_git_mode(tmp_path):
    from opentad.models.chronotransport.registration import (
        validate_registration_commit_shape,
    )

    payload = b"registration-target.json"
    relative = "artifacts/registration.json"
    repo, implementation = _init_repo(tmp_path)
    blob = _git(repo, "hash-object", "-w", "--stdin", input_text=payload.decode())
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},{relative}")
    _git(repo, "commit", "-qm", "symlink R")
    registration_commit = _git(repo, "rev-parse", "HEAD")
    path = repo / relative
    if path.is_symlink() or path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="regular blob|Git mode"):
        validate_registration_commit_shape(
            repository_root=repo,
            registration_commit=registration_commit,
            implementation_commit=implementation,
            registration_relpath=relative,
            registration_bytes=payload,
        )


def test_gate1_output_create_is_atomic_no_clobber_across_processes(tmp_path):
    output = tmp_path / "gate1_result.json"
    source = (
        "from pathlib import Path; import sys; "
        "from tools.bata.run_chronotransport_r2_gate1 import _atomic_write; "
        "_atomic_write(Path(sys.argv[1]), sys.argv[2].encode())"
    )
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", source, str(output), payload],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for payload in ("first\n", "second\n")
    ]
    results = [process.communicate(timeout=180) for process in processes]
    returncodes = sorted(process.returncode for process in processes)
    assert returncodes == [0, 1], results
    assert output.read_bytes() in (b"first\n", b"second\n")


def test_launcher_uses_exclusive_lock_and_no_clobber_terminal_link():
    text = Path(
        "scripts/run_chronotransport_r2_gate1_slurm_single_gpu.sh"
    ).read_text(
        encoding="utf-8"
    )
    assert "RUN_LOCK" in text
    assert 'mkdir "$RUN_LOCK"' in text
    assert 'ln -- "$TEMP_MARKER" "$TERMINAL_MARKER"' in text
    assert 'mv "$TEMP_MARKER" "$TERMINAL_MARKER"' not in text
