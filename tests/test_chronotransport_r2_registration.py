import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from opentad.models.chronotransport.actions import LayerGroup
from opentad.models.chronotransport.controls import (
    r2_control_algorithm_identity,
    random_exact_count_actions,
)
from opentad.models.chronotransport.environment import REQUIRED_ENVIRONMENT_SCHEMA
from opentad.models.chronotransport.protocol import (
    build_r2_manifest,
    build_stage_b_exposure_artifact,
    canonical_json_bytes,
    canonical_sha256,
    manifest_exact_bytes,
)
from opentad.models.chronotransport.registration import (
    APPROVED_SPEC_COMMIT,
    APPROVED_SPEC_SHA256,
    CHECKPOINT_RECEIPT_SCHEMA,
    EXPECTED_PROFILE_CANDIDATE_ORDER,
    REGISTERED_PROFILE_FACTORY_IDENTITY,
    REGISTERED_PROFILE_BACKEND_IDENTITY,
    REGISTERED_PROFILE_BACKEND_SOURCE,
    REQUIRED_REGISTRATION_SOURCE_PATHS,
    SOURCE_CLASSIFICATION_PATH,
    build_pre_gate1_registration,
    build_pre_gate1_registration_from_context,
    claim_flags,
    validate_source_classification_manifest,
    validate_pre_gate1_registration,
)
from opentad.models.chronotransport.scheduler import ScheduleLibrary
from test_chronotransport_r2_manifest_protocol import _config_identity, _registry
import tools.bata.register_chronotransport_r2 as registration_cli
from tools.bata.validate_chronotransport_r2_precheck import (
    validate_precheck_for_test_only as validate_precheck,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def test_registration_publish_is_atomic_no_clobber(tmp_path: Path) -> None:
    output = tmp_path / "chronotransport_pre_gate1_registration.json"
    registration_cli._atomic_write(output, b"first\n")
    with pytest.raises(FileExistsError):
        registration_cli._atomic_write(output, b"second\n")
    assert output.read_bytes() == b"first\n"


def _checkpoint_receipt_identity(
    *,
    checkpoint_sha256: str,
    checkpoint_bytes: int,
    source_path: str = "/registry/receipt.json",
    provider_receipt_path: str = "/registry/provider.receipt",
):
    artifact = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "provider_identity": "paracloud-registry",
        "registry_id": "opentad/dense/epoch_57.pth",
        "authenticated_uri": "registry://opentad/dense/epoch_57.pth",
        "retrieval_tool_identity": "paracloud-registry-client/v1",
        "authenticated_principal": "sczc063@BSCC-N16R4",
        "registry_request_id": "request-registered-0001",
        "retrieved_at_utc": "2026-07-13T12:00:00Z",
        "content_sha256": checkpoint_sha256,
        "content_bytes": checkpoint_bytes,
        "provider_receipt_sha256": _sha("provider-receipt"),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    exact_bytes = canonical_json_bytes(artifact) + b"\n"
    return {
        "artifact": artifact,
        "exact_bytes_sha256": hashlib.sha256(exact_bytes).hexdigest(),
        "source_path": source_path,
        "provider_receipt_path": provider_receipt_path,
    }


def _control_actions(
    name: str,
    invocation_index: int,
    *,
    invocation_ids: list[str] | None = None,
) -> list[list[int]]:
    period = int(name.rsplit("p", 1)[1])
    if name.startswith("random_p"):
        if invocation_ids is None:
            raise ValueError("random control helper requires invocation IDs")
        return random_exact_count_actions(
            invocation_ids[invocation_index],
            seed=3407,
            num_groups=3,
            period=period,
        ).tolist()
    actions = [[2, 2, 2] for _ in range(48)]
    for clip in range(0, 48, period):
        actions[clip] = [0, 0, 0]
    return actions


def _requested_actions(
    identity: dict,
    name: str,
    invocation_index: int,
    *,
    invocation_ids: list[str],
) -> list[list[int]]:
    for candidate in identity["candidate_library"]["candidates"]:
        if candidate["name"] == name:
            return candidate["actions"]
    return _control_actions(name, invocation_index, invocation_ids=invocation_ids)


def _identity():
    registry = _registry()
    manifest = build_r2_manifest(registry, _config_identity())
    library = ScheduleLibrary.r2(
        layer_groups=(LayerGroup(0, 4), LayerGroup(4, 8), LayerGroup(8, 12))
    ).canonical_payload()
    controls = r2_control_algorithm_identity()
    action_hashes = {
        row["name"]: row["action_sha256"] for row in library["candidates"]
    }
    splits = manifest["splits"]
    invocation_ids = [*splits["fit"], *splits["calibration"], *splits["evaluation"]]
    environment = {
        "schema": REQUIRED_ENVIRONMENT_SCHEMA,
        "gpu_model": "NVIDIA A100-SXM4-80GB",
        "driver": "535.54",
        "cuda": "11.8",
        "pytorch": "2.1.0",
        "cudnn": "8902",
        "precision": "amp_fp16",
        "batch_size": 1,
    }
    environment["environment_sha256"] = canonical_sha256(environment)
    candidate_plan = []
    for name in EXPECTED_PROFILE_CANDIDATE_ORDER:
        if name in action_hashes:
            identity_sha = action_hashes[name]
            hashes = [identity_sha] * 200
        else:
            algorithm = "motion_topk" if name.startswith("motion_topk") else "random"
            identity_sha = controls[algorithm]["sha256"]
            hashes = [
                canonical_sha256(
                    _control_actions(name, index, invocation_ids=invocation_ids)
                )
                for index in range(200)
            ]
        factory_config = {
            "candidate_name": name,
            "mode": "registered_full_stack",
            "backend_identity": REGISTERED_PROFILE_BACKEND_IDENTITY,
            "backend_source_sha256": hashlib.sha256(
                (ROOT / REGISTERED_PROFILE_BACKEND_SOURCE).read_bytes()
            ).hexdigest(),
        }
        if name.startswith("random_p"):
            factory_config["control_seed"] = 3407
        first_actions = _requested_actions(
            {"candidate_library": library},
            name,
            0,
            invocation_ids=invocation_ids,
        )
        candidate_plan.append(
            {
                "candidate_name": name,
                "candidate_identity_sha256": identity_sha,
                "factory_identity": REGISTERED_PROFILE_FACTORY_IDENTITY,
                "factory_config": factory_config,
                "factory_config_sha256": canonical_sha256(factory_config),
                "requested_action_sha256_by_invocation": hashes,
                "requested_action_order_sha256": canonical_sha256(hashes),
                "selected_rows_per_group": [
                    sum(row[group] == 0 for row in first_actions) for group in range(3)
                ],
            }
        )
    profiler = {
        "schema": "chronotransport-r2-profiler-plan-v1",
        "candidate_order": list(EXPECTED_PROFILE_CANDIDATE_ORDER),
        "candidate_order_sha256": canonical_sha256(list(EXPECTED_PROFILE_CANDIDATE_ORDER)),
        "invocation_ids": invocation_ids,
        "invocation_order_sha256": canonical_sha256(invocation_ids),
        "warmup_count": 50,
        "sample_count": 200,
        "candidate_plan": candidate_plan,
        "expected_environment": environment,
        "model_config_sha256": manifest["config_identity"]["config_sha256"],
    }
    return {
        "protocol_id": "CT-P3R-3S-r2",
        "spec": {"commit": APPROVED_SPEC_COMMIT, "sha256": APPROVED_SPEC_SHA256},
        "implementation_commit": _sha("implementation")[:40],
        "registration_parent": {
            "commit": _sha("implementation")[:40],
            "tree": _sha("tree")[:40],
        },
        "source_files": {
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in REQUIRED_REGISTRATION_SOURCE_PATHS
        },
        "upstream_commits": {"opentad": _sha("upstream")[:40]},
        "dense_checkpoint": {
            "sha256": _sha("checkpoint"),
            "bytes": 1,
            "registry_id": "opentad/dense/epoch_57.pth",
            "authenticated_uri": "registry://opentad/dense/epoch_57.pth",
            "content_addressed_path": (
                "/data/run01/sczc063/yuzibo/checkpoints/sha256/"
                + _sha("checkpoint")
            ),
            "registry_receipt": _checkpoint_receipt_identity(
                checkpoint_sha256=_sha("checkpoint"),
                checkpoint_bytes=1,
            ),
        },
        "data": {
            "root_identity": registry["data_sha256"],
            "root_path": "/data/run01/sczc063/yuzibo/datasets/thumos14",
            "annotation_sha256": registry["annotation_sha256"],
            "media_sha256": {
                row["video_id"]: row["media_sha256"] for row in registry["records"]
            },
        },
        "window_manifest": {
            "artifact": manifest,
            "exact_bytes_sha256": hashlib.sha256(manifest_exact_bytes(manifest)).hexdigest(),
            "source_path": "/data/run01/sczc063/yuzibo/chronotransport_inputs/r2_manifest.json",
            "registry_path": "/data/run01/sczc063/yuzibo/chronotransport_inputs/r2_registry.json",
            "config_identity_path": "/data/run01/sczc063/yuzibo/chronotransport_inputs/r2_config.json",
        },
        "candidate_library": library,
        "exposures": {
            "stage_b": build_stage_b_exposure_artifact(manifest["splits"]["fit"]),
            "stage_c_formula": "candidate=(p+5*b+seed_offset)%16",
        },
        "controls": controls,
        "bootstrap": {"gate1_samples": 5000, "seed": 20260711},
        "profiler": profiler,
        "gates": {"gate1_relative": 0.1, "budget_saving": 0.2},
        "environment": environment,
        "output_root": {
            "base": "/data/run01/sczc063/yuzibo/chronotransport_runs/ct_p3r_3s_r2",
            "template": "{base}/{registration_commit}/shared/gate1",
        },
        "attestation": {"result_data_unread": True},
    }


def test_registration_is_deeply_canonical_hash_bound_and_validatable():
    registration = build_pre_gate1_registration(_identity())
    assert len(registration["registration_sha256"]) == 64
    assert validate_pre_gate1_registration(registration) == registration
    assert registration["profiler"]["candidate_order"] == list(EXPECTED_PROFILE_CANDIDATE_ORDER)
    assert len(registration["profiler"]["invocation_ids"]) == 200


def test_all_current_gate_sources_are_inside_the_exact_registration_surface():
    gate_required_sources = {
        "opentad/evaluations/mAP.py",
        "opentad/models/chronotransport/gates23.py",
        "opentad/models/chronotransport/gate4.py",
        "tools/bata/run_chronotransport_r2_gates23.py",
        "tools/bata/chronotransport_r2_gates23_replay_factory.py",
        "tests/test_chronotransport_r2_gate1_hardening.py",
        "tests/test_chronotransport_r2_gate4.py",
        "tests/test_chronotransport_r2_gates23.py",
    }
    required_sources = set(REQUIRED_REGISTRATION_SOURCE_PATHS)
    assert gate_required_sources <= required_sources

    valid_identity = _identity()
    assert set(valid_identity["source_files"]) == required_sources

    missing = copy.deepcopy(valid_identity)
    missing["source_files"].pop("opentad/models/chronotransport/gates23.py")
    with pytest.raises(ValueError, match="complete required surface"):
        build_pre_gate1_registration(missing)

    extra = copy.deepcopy(valid_identity)
    extra["source_files"]["tools/bata/unregistered_formal_minter.py"] = _sha(
        "future-unregistered-source"
    )
    with pytest.raises(ValueError, match="complete required surface"):
        build_pre_gate1_registration(extra)


def test_registration_accepts_only_the_approved_spec_commit_and_exact_bytes():
    from opentad.models.chronotransport.registration import (
        APPROVED_SPEC_COMMIT,
        APPROVED_SPEC_SHA256,
    )

    identity = _identity()
    identity["spec"] = {
        "commit": APPROVED_SPEC_COMMIT,
        "sha256": APPROVED_SPEC_SHA256,
    }
    build_pre_gate1_registration(identity)
    for field, value in (
        ("commit", _sha("unapproved-spec")[:40]),
        ("sha256", _sha("unapproved-spec-bytes")),
    ):
        damaged = copy.deepcopy(identity)
        damaged["spec"][field] = value
        with pytest.raises(ValueError, match="approved spec"):
            build_pre_gate1_registration(damaged)


def test_registration_commit_shape_requires_one_parent_unique_add_and_exact_blob(
    tmp_path,
):
    from opentad.models.chronotransport.registration import (
        validate_registration_commit_shape,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "ct@test.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "CT Test"], check=True)
    (repo / "base.txt").write_text("I", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "base.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "I"], check=True)
    implementation = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    registration_relpath = "artifacts/registration.json"
    registration_bytes = b'{"immutable":true}\n'
    registration_path = repo / registration_relpath
    registration_path.parent.mkdir()
    registration_path.write_bytes(registration_bytes)
    subprocess.run(["git", "-C", str(repo), "add", registration_relpath], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "R"], check=True)
    registration_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    validate_registration_commit_shape(
        repository_root=repo,
        registration_commit=registration_commit,
        implementation_commit=implementation,
        registration_relpath=registration_relpath,
        registration_bytes=registration_bytes,
    )

    merge_commit = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "commit-tree",
            f"{registration_commit}^{{tree}}",
            "-p",
            implementation,
            "-p",
            registration_commit,
        ],
        input="invalid merge R\n",
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    with pytest.raises(ValueError, match="exactly one parent"):
        validate_registration_commit_shape(
            repository_root=repo,
            registration_commit=merge_commit,
            implementation_commit=implementation,
            registration_relpath=registration_relpath,
            registration_bytes=registration_bytes,
        )
    with pytest.raises(ValueError, match="blob bytes"):
        validate_registration_commit_shape(
            repository_root=repo,
            registration_commit=registration_commit,
            implementation_commit=implementation,
            registration_relpath=registration_relpath,
            registration_bytes=b'{"immutable":false}\n',
        )


def test_formal_random_control_plan_accepts_only_exact_integer_seed_3407():
    from opentad.models.chronotransport.registration import (
        validate_formal_random_control_lock,
    )

    registration = build_pre_gate1_registration(_identity())
    validate_formal_random_control_lock(registration)

    for seed in (None, "3407", 3408, 3409):
        damaged = copy.deepcopy(registration)
        random_plan = next(
            plan
            for plan in damaged["profiler"]["candidate_plan"]
            if plan["candidate_name"] == "random_p4"
        )
        if seed is None:
            random_plan["factory_config"].pop("control_seed")
        else:
            random_plan["factory_config"]["control_seed"] = seed
        with pytest.raises(ValueError, match="control_seed.*3407"):
            validate_formal_random_control_lock(damaged)


@pytest.mark.parametrize("seed", [None, "3407", 3408, 3409])
def test_registration_identity_rejects_missing_or_alternate_random_seed(seed):
    identity = _identity()
    random_plan = next(
        plan
        for plan in identity["profiler"]["candidate_plan"]
        if plan["candidate_name"] == "random_p4"
    )
    if seed is None:
        random_plan["factory_config"].pop("control_seed")
    else:
        random_plan["factory_config"]["control_seed"] = seed
    random_plan["factory_config_sha256"] = canonical_sha256(
        random_plan["factory_config"]
    )
    with pytest.raises(ValueError, match="factory config"):
        build_pre_gate1_registration(identity)


def test_registration_identity_recomputes_and_rejects_random_action_hashes():
    identity = _identity()
    random_plan = next(
        plan
        for plan in identity["profiler"]["candidate_plan"]
        if plan["candidate_name"] == "random_p4"
    )
    random_plan["requested_action_sha256_by_invocation"][17] = _sha(
        "caller-substituted-random-action"
    )
    random_plan["requested_action_order_sha256"] = canonical_sha256(
        random_plan["requested_action_sha256_by_invocation"]
    )
    with pytest.raises(ValueError, match="generated action hashes"):
        build_pre_gate1_registration(identity)


def test_source_classification_is_exhaustive_and_exactly_matches_required_vector():
    manifest = json.loads((ROOT / SOURCE_CLASSIFICATION_PATH).read_text(encoding="utf-8"))
    tracked = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "ls-files",
            "tests/test_chronotransport*.py",
            "tools/bata/*chronotransport*.py",
            "scripts/*chronotransport*.sh",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    validated = validate_source_classification_manifest(
        manifest,
        tracked_paths=tracked,
        required_source_paths=REQUIRED_REGISTRATION_SOURCE_PATHS,
    )
    tests = [path for path in validated["files"] if path.startswith("tests/")]
    assert len(tests) == 22


def test_source_classification_rejects_omission_addition_and_vector_drift():
    manifest = json.loads((ROOT / SOURCE_CLASSIFICATION_PATH).read_text(encoding="utf-8"))
    tracked = list(manifest["files"])

    omitted = copy.deepcopy(manifest)
    omitted["files"].pop("tests/test_chronotransport_pipeline.py")
    with pytest.raises(ValueError, match="exact tracked inventory"):
        validate_source_classification_manifest(
            omitted,
            tracked_paths=tracked,
            required_source_paths=REQUIRED_REGISTRATION_SOURCE_PATHS,
        )

    with pytest.raises(ValueError, match="exact tracked inventory"):
        validate_source_classification_manifest(
            manifest,
            tracked_paths=[*tracked, "tests/test_chronotransport_new_escape.py"],
            required_source_paths=REQUIRED_REGISTRATION_SOURCE_PATHS,
        )

    missing_required = tuple(
        path
        for path in REQUIRED_REGISTRATION_SOURCE_PATHS
        if path != "tests/test_chronotransport_pipeline.py"
    )
    with pytest.raises(ValueError, match="REQUIRED classification.*source vector"):
        validate_source_classification_manifest(
            manifest,
            tracked_paths=tracked,
            required_source_paths=missing_required,
        )


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda x: x["profiler"]["candidate_order"].reverse(), "candidate order"),
        (lambda x: x["profiler"].update(warmup_count=True), "warmup_count"),
        (lambda x: x["profiler"]["invocation_ids"].__setitem__(1, x["profiler"]["invocation_ids"][0]), "invocation"),
        (lambda x: x["profiler"]["candidate_plan"][0].update(extra="cheat"), "fields mismatch"),
        (
            lambda x: x["window_manifest"]["artifact"]["splits"]["evaluation"].__setitem__(
                0, x["window_manifest"]["artifact"]["splits"]["calibration"][0]
            ),
            "split|window",
        ),
        (lambda x: x["bootstrap"].update(gate1_samples=5000.0), "gate1_samples"),
    ],
)
def test_registration_rejects_profiler_manifest_and_scalar_counterexamples(mutator, match):
    identity = _identity()
    mutator(identity)
    with pytest.raises((TypeError, ValueError), match=match):
        build_pre_gate1_registration(identity)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda x: x["profiler"]["candidate_plan"].pop(), "candidate plan"),
        (
            lambda x: x["profiler"]["candidate_plan"][0].update(
                candidate_identity_sha256=_sha("fake-action")
            ),
            "candidate identity",
        ),
        (
            lambda x: x["profiler"]["candidate_plan"][-1][
                "requested_action_sha256_by_invocation"
            ].pop(),
            "200 requested action",
        ),
        (
            lambda x: x["profiler"]["candidate_plan"][0].update(
                factory_config_sha256=_sha("fake-factory-config")
            ),
            "factory config",
        ),
        (
            lambda x: x["profiler"]["candidate_plan"][0].update(
                factory_identity="tests.fake_profile_factory:build"
            ),
            "repo-owned profile factory",
        ),
        (
            lambda x: x["profiler"].update(
                expected_environment={**x["environment"], "driver": "535.fake"}
            ),
            "environment",
        ),
        (
            lambda x: x["controls"]["motion_topk"].update(sha256=_sha("fake-control")),
            "control algorithm",
        ),
        (lambda x: x.update(extra="cheat"), "fields mismatch"),
        (lambda x: x["source_files"].pop(REQUIRED_REGISTRATION_SOURCE_PATHS[0]), "source files"),
        (
            lambda x: x["window_manifest"]["artifact"]["windows"][0][
                "sampled_frame_indices"
            ].__setitem__(0, 999999),
            "fixed config window|window hash|source bounds",
        ),
        (
            lambda x: x["window_manifest"].update(exact_bytes_sha256=_sha("fake-manifest-bytes")),
            "exact bytes",
        ),
        (
            lambda x: x["exposures"]["stage_b"]["matrices"]["3407"][0].update(candidate=15),
            "candidate formula",
        ),
        (
            lambda x: x["dense_checkpoint"].update(
                authenticated_uri="file:///tmp/epoch_57.pth"
            ),
            "authenticated registry URI",
        ),
        (
            lambda x: x["dense_checkpoint"].update(
                content_addressed_path="/tmp/unbound-checkpoint.pth"
            ),
            "content-addressed",
        ),
    ],
)
def test_registration_rejects_missing_fake_or_unbound_profile_identity(mutator, match):
    identity = _identity()
    mutator(identity)
    with pytest.raises((TypeError, ValueError), match=match):
        build_pre_gate1_registration(identity)


def test_registration_rejects_result_derived_fields():
    identity = _identity()
    identity["gate_report_path"] = "/tmp/result.json"
    with pytest.raises(ValueError, match="forbidden result-derived"):
        build_pre_gate1_registration(identity)


def test_claims_follow_gate_chain_and_never_unlock_deploy_or_paper():
    flags = claim_flags(gate1=True, gate2=True, gate3=True, gate4=True)
    assert flags["oracle_headroom"] and flags["mechanism"]
    assert flags["metric_adatad_thumos14_official_full_video"]
    assert flags["deploy"] is False and flags["paper"] is False
    with pytest.raises(ValueError, match="Gate 2"):
        claim_flags(gate2=True)


def test_context_registration_derives_clean_detached_git_manifest_checkpoint_and_data(
    tmp_path, monkeypatch
):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "ct@test.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "CT Test"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "fetch",
            "-q",
            os.environ.get("CHRONOTRANSPORT_TEST_APPROVED_BUNDLE", str(ROOT)),
            APPROVED_SPEC_COMMIT,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )
    classification = json.loads(
        (ROOT / SOURCE_CLASSIFICATION_PATH).read_text(encoding="utf-8")
    )["files"]
    stale_classified = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-files",
            "tests/test_chronotransport*.py",
            "tools/bata/*chronotransport*.py",
            "scripts/*chronotransport*.sh",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative in stale_classified:
        if relative not in classification:
            (repo / relative).unlink()
    for relative in classification:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    for relative in REQUIRED_REGISTRATION_SOURCE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "implementation I"], check=True)

    data_root = tmp_path / "data"
    registry = _registry()
    for index, record in enumerate(registry["records"]):
        media = data_root / record["media_path"]
        media.parent.mkdir(parents=True, exist_ok=True)
        media.write_bytes(f"registered-media-{index}".encode())
        record["media_sha256"] = hashlib.sha256(media.read_bytes()).hexdigest()
    registry_path = tmp_path / "registry.json"
    config_path = tmp_path / "config.json"
    manifest_path = tmp_path / "manifest.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    config = _config_identity()
    config_path.write_text(json.dumps(config), encoding="utf-8")
    manifest = build_r2_manifest(registry, config)
    manifest_path.write_bytes(manifest_exact_bytes(manifest))
    checkpoint = tmp_path / "epoch_57.pth"
    checkpoint.write_bytes(b"registered-dense-checkpoint")
    provider_receipt = tmp_path / "provider.receipt"
    provider_receipt.write_bytes(b"provider-receipt")
    receipt_identity = _checkpoint_receipt_identity(
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        checkpoint_bytes=len(checkpoint.read_bytes()),
        source_path=str(tmp_path / "receipt.json"),
        provider_receipt_path=str(provider_receipt),
    )
    receipt_identity["artifact"]["provider_receipt_sha256"] = hashlib.sha256(
        provider_receipt.read_bytes()
    ).hexdigest()
    receipt_unsigned = dict(receipt_identity["artifact"])
    receipt_unsigned.pop("artifact_sha256")
    receipt_identity["artifact"]["artifact_sha256"] = canonical_sha256(
        receipt_unsigned
    )
    receipt_path = Path(receipt_identity["source_path"])
    receipt_path.write_bytes(canonical_json_bytes(receipt_identity["artifact"]) + b"\n")

    identity = _identity()
    registered_output = (tmp_path / "allowed" / "runs").resolve()
    import opentad.models.chronotransport.registration as registration_module

    # The production constant remains immutable; this isolated filesystem
    # harness substitutes its temporary allowed root without weakening runtime
    # validation.
    monkeypatch.setattr(
        registration_module, "FORMAL_OUTPUT_BASE", str(registered_output)
    )
    identity["output_root"] = {
        "base": str(registered_output),
        "template": "{base}/{registration_commit}/shared/gate1",
    }
    registration = build_pre_gate1_registration_from_context(
        identity,
        repository_root=repo,
        manifest_path=manifest_path,
        registry_path=registry_path,
        config_identity_path=config_path,
        checkpoint_source=checkpoint,
        checkpoint_registry_id="opentad/dense/epoch_57.pth",
        checkpoint_authenticated_uri="registry://opentad/dense/epoch_57.pth",
        checkpoint_receipt_path=receipt_path,
        checkpoint_provider_receipt_path=provider_receipt,
        content_store_root=tmp_path / "content-store",
        data_root=data_root,
    )
    assert validate_pre_gate1_registration(
        registration,
        repository_root=repo,
        context_mode="generation",
    ) == registration
    assert Path(registration["dense_checkpoint"]["content_addressed_path"]).read_bytes() == checkpoint.read_bytes()
    assert registration["dense_checkpoint"]["registry_receipt"]["artifact"] == receipt_identity["artifact"]

    registration_file = repo / "artifacts" / "chronotransport_pre_gate1_registration.json"
    registration_file.parent.mkdir(parents=True)
    registration_file.write_bytes(canonical_json_bytes(registration) + b"\n")
    subprocess.run(["git", "-C", str(repo), "add", registration_file.relative_to(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "registration R"], check=True)
    registration_commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    derived_output = (
        registered_output / registration_commit / "shared" / "gate1"
    ).resolve()
    report = validate_precheck(
        registration_path=registration_file,
        repository_root=repo,
        registration_commit=registration_commit,
        output_root=derived_output,
        allowed_output_root=tmp_path / "allowed",
    )
    assert report["status"] == "PRECHECK_OK"

    import tools.bata.train_chronotransport_r2_stage_b as stage_b_cli

    monkeypatch.setattr(stage_b_cli, "ROOT", repo)
    loaded, derived_r, registration_relpath = stage_b_cli._load_formal_registration(
        registration_file
    )
    assert loaded == registration
    assert derived_r == registration_commit
    assert registration_relpath == registration_file.relative_to(repo).as_posix()

    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "commit",
            "--allow-empty",
            "-qm",
            "invalid R successor",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="exactly one parent"):
        stage_b_cli._load_formal_registration(registration_file)
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "-q", "--detach", registration_commit],
        check=True,
    )

    provider_receipt.write_bytes(b"tampered-provider-receipt")
    with pytest.raises(ValueError, match="provider receipt hash"):
        validate_precheck(
            registration_path=registration_file,
            repository_root=repo,
            registration_commit=registration_commit,
            output_root=derived_output,
            allowed_output_root=tmp_path / "allowed",
        )
    provider_receipt.write_bytes(b"provider-receipt")

    (repo / REQUIRED_REGISTRATION_SOURCE_PATHS[1]).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        stage_b_cli._load_formal_registration(registration_file)
    with pytest.raises(ValueError, match="clean"):
        validate_pre_gate1_registration(
            registration,
            repository_root=repo,
            context_mode="formal",
            registration_commit=registration_commit,
            registration_relpath=registration_file.relative_to(repo).as_posix(),
        )


def test_slurm_single_gpu_launcher_contains_registration_and_environment_guards():
    text = open(
        "scripts/run_chronotransport_r2_gate1_slurm_single_gpu.sh", encoding="utf-8"
    ).read()
    assert "CUDA_VISIBLE_DEVICES=" not in text
    assert "export CUDA_VISIBLE_DEVICES" not in text
    assert "CHRONOTRANSPORT_REGISTRATION_COMMIT" in text
    assert "SLURM_JOB_ID" in text and "SLURM_STEP_ID" in text
    assert "PRECHECK_ONLY" in text
    assert "validate_chronotransport_r2_precheck.py" in text
    assert "--gate1-input" in text
    assert "module load cuda/11.8" in text
    assert "conda_envs/opentad/bin/activate" in text
    assert "gate1_terminal.json" in text and "GATE1_${state}" in text
    assert "environment.py" in text
    assert text.index("SLURM_JOB_ID") < text.index('if [[ "${PRECHECK_ONLY:-0}"')


def test_launcher_has_atomic_exact_terminal_state_traps():
    text = open(
        "scripts/run_chronotransport_r2_gate1_slurm_single_gpu.sh", encoding="utf-8"
    ).read()
    for state in (
        "SUCCESS",
        "FAIL",
        "STOPPED",
        "INVALID_ENVIRONMENT",
        "INVALID_IMPLEMENTATION",
    ):
        assert state in text
    assert "trap" in text and "EXIT" in text and "INT" in text and "TERM" in text
    assert "chronotransport-r2-gate1-terminal-v1" in text
    assert 'ln -- "$TEMP_MARKER" "$TERMINAL_MARKER"' in text
    assert 'mv "$TEMP_MARKER" "$TERMINAL_MARKER"' not in text
    assert 'mkdir "$RUN_LOCK"' in text


def test_precheck_requires_fixed_distinct_fresh_gate1_filenames(tmp_path, monkeypatch):
    from tools.bata import validate_chronotransport_r2_precheck as precheck

    output = tmp_path / "R" / "shared" / "gate1"
    output.mkdir(parents=True)
    registration = tmp_path / "registration.json"
    registration.write_text("{}", encoding="utf-8")
    gate1_input = output / "gate1_input.json"
    gate1_input.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(precheck, "load_exact_registration", lambda path: {"registered": True})
    monkeypatch.setattr(
        precheck,
        "validate_pre_gate1_registration",
        lambda registration, **kwargs: {
            "registration_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
        },
    )
    monkeypatch.setattr(precheck, "resolve_gate1_output_root", lambda registration, commit: output)
    monkeypatch.setattr(precheck, "_validate_gate1_input_payload", lambda *args, **kwargs: {})

    report = precheck.validate_precheck_for_test_only(
        registration_path=registration,
        repository_root=tmp_path,
        registration_commit="c" * 40,
        output_root=output,
        gate1_input_path=gate1_input,
        gate1_output_path=output / "gate1_result.json",
        terminal_marker_path=output / "gate1_terminal.json",
        allowed_output_root=tmp_path,
    )
    assert report["status"] == "PRECHECK_OK"

    with pytest.raises(ValueError, match="canonical"):
        precheck.validate_precheck_for_test_only(
            registration_path=registration,
            repository_root=tmp_path,
            registration_commit="c" * 40,
            output_root=output,
            gate1_input_path=gate1_input,
            gate1_output_path=output / "renamed.json",
            terminal_marker_path=output / "gate1_terminal.json",
            allowed_output_root=tmp_path,
        )
    (output / "gate1_result.json").write_text("stale", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        precheck.validate_precheck_for_test_only(
            registration_path=registration,
            repository_root=tmp_path,
            registration_commit="c" * 40,
            output_root=output,
            gate1_input_path=gate1_input,
            gate1_output_path=output / "gate1_result.json",
            terminal_marker_path=output / "gate1_terminal.json",
            allowed_output_root=tmp_path,
        )


def test_precheck_rejects_parent_symlink_aliases_for_registration_output_and_artifacts(
    tmp_path, monkeypatch
):
    from tools.bata import validate_chronotransport_r2_precheck as precheck

    real_tree = tmp_path / "real-tree"
    registration_commit = "c" * 40
    output = real_tree / registration_commit / "shared" / "gate1"
    output.mkdir(parents=True)
    alias_tree = tmp_path / "alias-tree"
    registration_parent = tmp_path / "real-registration"
    registration_parent.mkdir()
    registration = registration_parent / "registration.json"
    registration.write_text("{}", encoding="utf-8")
    registration_alias_parent = tmp_path / "registration-alias"
    try:
        alias_tree.symlink_to(real_tree, target_is_directory=True)
        registration_alias_parent.symlink_to(
            registration_parent, target_is_directory=True
        )
    except OSError:
        pytest.skip("symlinks are unavailable in this test environment")

    monkeypatch.setattr(
        precheck, "load_exact_registration", lambda path: {"registered": True}
    )
    monkeypatch.setattr(
        precheck,
        "validate_pre_gate1_registration",
        lambda registration, **kwargs: {
            "registration_sha256": "a" * 64,
            "implementation_commit": "b" * 40,
            "output_root": {
                "base": str(real_tree),
                "template": "{base}/{registration_commit}/shared/gate1",
            },
        },
    )
    monkeypatch.setattr(
        precheck, "resolve_gate1_output_root", lambda registration, commit: output
    )
    monkeypatch.setattr(
        precheck, "_validate_gate1_input_payload", lambda *args, **kwargs: {}
    )

    with pytest.raises(ValueError, match="symlink"):
        precheck.validate_precheck_for_test_only(
            registration_path=registration_alias_parent / "registration.json",
            repository_root=tmp_path,
            registration_commit=registration_commit,
            output_root=output,
            allowed_output_root=tmp_path,
        )

    with pytest.raises(ValueError, match="symlink"):
        precheck.validate_precheck_for_test_only(
            registration_path=registration,
            repository_root=tmp_path,
            registration_commit=registration_commit,
            output_root=alias_tree / registration_commit / "shared" / "gate1",
            allowed_output_root=tmp_path,
        )

    gate1_input = output / "gate1_input.json"
    gate1_input.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="symlink"):
        precheck.validate_precheck_for_test_only(
            registration_path=registration,
            repository_root=tmp_path,
            registration_commit=registration_commit,
            output_root=output,
            gate1_input_path=alias_tree
            / registration_commit
            / "shared"
            / "gate1"
            / "gate1_input.json",
            gate1_output_path=output / "gate1_result.json",
            terminal_marker_path=output / "gate1_terminal.json",
            allowed_output_root=tmp_path,
        )


def test_registered_source_helper_rejects_symlink(tmp_path):
    from opentad.models.chronotransport.registration import _validate_registered_source_file

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "ct@test.invalid"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "CT Test"], check=True)
    source = repo / "source.py"
    source.write_bytes(b"print('fixed')\n")
    subprocess.run(["git", "-C", str(repo), "add", "source.py"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "I"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    _validate_registered_source_file(
        root=repo,
        revision=revision,
        relative="source.py",
        registered_sha256=digest,
    )

    target = repo / "target.py"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target.name)
    with pytest.raises(ValueError, match="regular|symlink"):
        _validate_registered_source_file(
            root=repo,
            revision=revision,
            relative="source.py",
            registered_sha256=digest,
        )


def test_launcher_never_pins_a_physical_gpu_index():
    text = open(
        "scripts/run_chronotransport_r2_gate1_slurm_single_gpu.sh", encoding="utf-8"
    ).read()
    assert "physical GPU1" not in text
    assert not any(
        "CUDA_VISIBLE_DEVICES" in line and '== "1"' in line
        for line in text.splitlines()
    )
    assert "CUDA_VISIBLE_DEVICES=" not in text


def test_checkpoint_registry_receipt_is_external_exact_and_content_bound(tmp_path):
    from opentad.models.chronotransport.registration import (
        CHECKPOINT_RECEIPT_SCHEMA,
        validate_checkpoint_registry_receipt,
    )

    provider_receipt = tmp_path / "provider.receipt"
    provider_receipt.write_bytes(b"provider-issued-authenticated-download-receipt")
    checkpoint_bytes = b"registered-dense-checkpoint"
    artifact = {
        "schema": CHECKPOINT_RECEIPT_SCHEMA,
        "provider_identity": "paracloud-registry",
        "registry_id": "opentad/dense/epoch_57.pth",
        "authenticated_uri": "registry://opentad/dense/epoch_57.pth",
        "retrieval_tool_identity": "paracloud-registry-client/v1",
        "authenticated_principal": "sczc063@BSCC-N16R4",
        "registry_request_id": "request-registered-0001",
        "retrieved_at_utc": "2026-07-13T12:00:00Z",
        "content_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
        "content_bytes": len(checkpoint_bytes),
        "provider_receipt_sha256": hashlib.sha256(
            provider_receipt.read_bytes()
        ).hexdigest(),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    assert validate_checkpoint_registry_receipt(
        artifact,
        provider_receipt_path=provider_receipt,
        registry_id="opentad/dense/epoch_57.pth",
        authenticated_uri="registry://opentad/dense/epoch_57.pth",
        content_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        content_bytes=len(checkpoint_bytes),
    ) == artifact

    unapproved_tool = copy.deepcopy(artifact)
    unapproved_tool["retrieval_tool_identity"] = "caller-tool/v9"
    unapproved_unsigned = dict(unapproved_tool)
    unapproved_unsigned.pop("artifact_sha256")
    unapproved_tool["artifact_sha256"] = canonical_sha256(unapproved_unsigned)
    with pytest.raises(ValueError, match="not approved"):
        validate_checkpoint_registry_receipt(
            unapproved_tool,
            provider_receipt_path=provider_receipt,
            registry_id="opentad/dense/epoch_57.pth",
            authenticated_uri="registry://opentad/dense/epoch_57.pth",
            content_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            content_bytes=len(checkpoint_bytes),
        )

    tampered = copy.deepcopy(artifact)
    tampered["authenticated_principal"] = "anonymous"
    with pytest.raises(ValueError, match="hash|receipt"):
        validate_checkpoint_registry_receipt(
            tampered,
            provider_receipt_path=provider_receipt,
            registry_id="opentad/dense/epoch_57.pth",
            authenticated_uri="registry://opentad/dense/epoch_57.pth",
            content_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            content_bytes=len(checkpoint_bytes),
        )

    impossible_date = copy.deepcopy(artifact)
    impossible_date["retrieved_at_utc"] = "2026-02-31T12:00:00Z"
    impossible_unsigned = dict(impossible_date)
    impossible_unsigned.pop("artifact_sha256")
    impossible_date["artifact_sha256"] = canonical_sha256(impossible_unsigned)
    with pytest.raises(ValueError, match="timestamp"):
        validate_checkpoint_registry_receipt(
            impossible_date,
            provider_receipt_path=provider_receipt,
            registry_id="opentad/dense/epoch_57.pth",
            authenticated_uri="registry://opentad/dense/epoch_57.pth",
            content_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
            content_bytes=len(checkpoint_bytes),
        )
