from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.name != "posix",
    reason="the formal Slurm deployer imports POSIX fcntl",
)


def _head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def test_deployer_accepts_only_the_commit_bound_canonical_launcher(tmp_path):
    from tools.bata.deploy_continuous_roi_s2_training_matrix import (
        _resolve_canonical_launcher,
    )

    root = Path(__file__).resolve().parents[1]
    launcher, launcher_sha256 = _resolve_canonical_launcher(
        source_root=root,
        requested_launcher=Path(
            "scripts/run_continuous_roi_s2_train_slurm.sh"
        ),
        expected_commit=_head(root),
    )
    assert launcher == (
        root / "scripts/run_continuous_roi_s2_train_slurm.sh"
    ).resolve()
    assert len(launcher_sha256) == 64

    outside = tmp_path / "launcher.sh"
    outside.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="canonical tracked launcher"):
        _resolve_canonical_launcher(
            source_root=root,
            requested_launcher=outside,
            expected_commit=_head(root),
        )


def test_submit_rejects_dirty_export_before_sbatch(monkeypatch, tmp_path):
    import tools.bata.deploy_continuous_roi_s2_training_matrix as deployer
    from tools.bata.spatial_zoom_s1_contract import sha256_file

    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")

    def unexpected_run(*_arguments):
        raise AssertionError("sbatch must not run for a dirty export")

    monkeypatch.setattr(deployer, "_run", unexpected_run)
    with pytest.raises(ValueError, match="control character"):
        deployer._submit_job(
            launcher=launcher,
            job_name="clean_name",
            job_token="a" * 64,
            log_dir=tmp_path,
            exports={"YUZIBO_ROOT": "/data/run01/sczc063/yuzibo\r"},
            expected_launcher_sha256=sha256_file(launcher),
        )


def test_submit_revalidates_launcher_immediately_before_sbatch(
    monkeypatch,
    tmp_path,
):
    import tools.bata.deploy_continuous_roi_s2_training_matrix as deployer
    from tools.bata.spatial_zoom_s1_contract import sha256_file

    launcher = tmp_path / "launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    frozen_sha256 = sha256_file(launcher)
    launcher.write_text("#!/usr/bin/env bash\nexit 9\n", encoding="utf-8")

    def unexpected_run(*_arguments):
        raise AssertionError("sbatch must not run after launcher mutation")

    monkeypatch.setattr(deployer, "_run", unexpected_run)
    with pytest.raises(ValueError, match="changed after deployment intent"):
        deployer._submit_job(
            launcher=launcher,
            job_name="clean_name",
            job_token="a" * 64,
            log_dir=tmp_path,
            exports={"YUZIBO_ROOT": "/data/run01/sczc063/yuzibo"},
            expected_launcher_sha256=frozen_sha256,
        )


def test_existing_summary_revalidates_intent_and_environment(tmp_path):
    from tools.bata.continuous_roi_s2_contract import canonical_sha256
    from tools.bata.deploy_continuous_roi_s2_training_matrix import (
        INTENT_SCHEMA,
        _validate_existing_deployment_request,
    )

    environment = {
        "yuzibo_root": "/data/run01/sczc063/yuzibo",
        "launcher_path": "/repo/scripts/run_continuous_roi_s2_train_slurm.sh",
        "launcher_sha256": "1" * 64,
        "slurm_export_encoding": "comma_delimited_control_free_v1",
    }
    intent_core = {
        "schema_version": INTENT_SCHEMA,
        "submission_environment": environment,
    }
    intent = {
        **intent_core,
        "intent_sha256": canonical_sha256(intent_core),
    }
    intent_path = tmp_path / "deployment_intent.json"
    intent_path.write_text(
        json.dumps(intent, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    root = tmp_path / "campaign"
    summary = {
        "code_commit": "2" * 40,
        "base_experiment_namespace": "3" * 64,
        "campaign_namespace": "4" * 64,
        "canonical_experiment_root": str(root),
        "full_model_gate_sha256": "5" * 64,
        "training_runtime_precheck_sha256": "6" * 64,
        "runtime_authorization_sha256": "7" * 64,
        "submission_environment": environment,
        "intent_path": str(intent_path),
        "intent_sha256": intent["intent_sha256"],
    }
    arguments = {
        "summary": summary,
        "intent_path": intent_path,
        "expected_commit": "2" * 40,
        "base_experiment_namespace": "3" * 64,
        "campaign_namespace": "4" * 64,
        "canonical_root": root,
        "full_model_gate_sha256": "5" * 64,
        "precheck_sha256": "6" * 64,
        "authorization_sha256": "7" * 64,
        "submission_environment": environment,
    }
    _validate_existing_deployment_request(**arguments)
    with pytest.raises(ValueError, match="summary differs"):
        _validate_existing_deployment_request(
            **{
                **arguments,
                "submission_environment": {
                    **environment,
                    "launcher_sha256": "8" * 64,
                },
            }
        )


def test_v1_deployment_evidence_is_rejected(tmp_path):
    from tools.bata.continuous_roi_s2_contract import canonical_sha256
    from tools.bata.deploy_continuous_roi_s2_training_matrix import (
        DEPLOYMENT_SCHEMA,
        _load_self_hashed,
    )

    core = {
        "schema_version": "continuous_roi_s2_training_deployment_v1",
        "status": "SUBMITTED",
    }
    payload = {**core, "deployment_sha256": canonical_sha256(core)}
    path = tmp_path / "deployment_summary.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported evidence schema"):
        _load_self_hashed(
            path,
            hash_key="deployment_sha256",
            schema=DEPLOYMENT_SCHEMA,
        )


def test_shell_launcher_rejects_control_character_before_slurm():
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "CONTINUOUS_ROI_S2_SOURCE_ROOT": str(root),
        "YUZIBO_ROOT": "/data/run01/sczc063/yuzibo\r",
        "CONTINUOUS_ROI_S2_RUN_ROOT": "/data/run01/sczc063/yuzibo/run",
        "CONTINUOUS_ROI_S2_MANIFEST": "/tmp/manifest.json",
        "CONTINUOUS_ROI_S2_DEVELOPMENT_ANNOTATION": "/tmp/annotation.json",
        "CONTINUOUS_ROI_S2_CLASS_MAP": "/tmp/classes.txt",
        "CONTINUOUS_ROI_S2_DEVELOPMENT_VIDEO_ROOT": "/tmp/videos",
        "CONTINUOUS_ROI_S2_PRETRAINED": "/tmp/pretrained.pth",
        "CONTINUOUS_ROI_S2_FULL_MODEL_GATE": "/tmp/gate.json",
        "CONTINUOUS_ROI_S2_TRAINING_RUNTIME_PRECHECK": "/tmp/precheck.json",
        "CONTINUOUS_ROI_S2_RUNTIME_AUTHORIZATION": "/tmp/auth.json",
        "CONTINUOUS_ROI_S2_EXPECTED_COMMIT": "1" * 40,
        "CONTINUOUS_ROI_S2_FAMILY": "D160",
        "CONTINUOUS_ROI_S2_SEED": "3407",
    }
    completed = subprocess.run(
        ["bash", str(root / "scripts/run_continuous_roi_s2_train_slurm.sh")],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 2
    assert "BASE contains an ASCII control character" in completed.stderr
    assert "formal training requires Slurm" not in completed.stderr
