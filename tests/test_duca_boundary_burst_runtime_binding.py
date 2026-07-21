from __future__ import annotations

import json
from pathlib import Path

from mmengine.config import Config

from tools.bata import duca_selected_axis_training as training


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"
BOUNDARY_VARIANTS = (
    "two_stage_exact_uniform",
    "gaussian_matched_g0",
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_boundary_burst_suite_binds_all_four_production_variants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    commit = "a" * 40
    pretrain = tmp_path / "pretrain.pth"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.txt"
    checkpoint = tmp_path / "frontend.pth"
    pretrain.write_bytes(b"pretrain")
    annotation.write_text("{}\n", encoding="utf-8")
    class_map.write_text("action\n", encoding="utf-8")
    checkpoint.write_bytes(b"frontend")
    checkpoint_sha256 = training.sha256_file(checkpoint)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT", str(checkpoint))
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_SHA256", checkpoint_sha256)
    monkeypatch.setenv("DUCA_FRONTEND_CHECKPOINT_EPOCH", "19")

    artifacts = []
    for variant in BOUNDARY_VARIANTS:
        config_path = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
        initialized = variant != "two_stage_exact_uniform"
        gate = {
            "schema": "duca_protected_e2e_exact_full_model_gradient_gate_v1",
            "ok": True,
            "config_sha256": training.sha256_file(config_path),
            "runtime": {"git_commit": commit},
            "adatad_pretrain": {"sha256": training.sha256_file(pretrain)},
            "selector_initialization": (
                {
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_epoch": 19,
                    "checkpoint_state_key": "state_dict_ema",
                    "detector_state_loaded": False,
                    "optimizer_state_loaded": False,
                    "scheduler_state_loaded": False,
                    "receipt_sha256": "b" * 64,
                }
                if initialized
                else None
            ),
        }
        gate_path = tmp_path / "full_model" / f"{config_path.stem}.json"
        _write_json(gate_path, gate)
        artifacts.append(
            {"path": str(gate_path), "sha256": training.sha256_file(gate_path)}
        )

    suite_path = tmp_path / "gate_suite.json"
    _write_json(
        suite_path,
        {
            "schema": training.BOUNDARY_BURST_GATE_SCHEMA,
            "ok": True,
            "formal_training_unlocked": True,
            "git_commit": commit,
            "artifacts": artifacts,
        },
    )
    monkeypatch.setenv("DUCA_SELECTED_OPT_GATE_SUITE", str(suite_path))
    monkeypatch.setenv(
        "DUCA_SELECTED_OPT_GATE_SUITE_SHA256",
        training.sha256_file(suite_path),
    )

    for variant in BOUNDARY_VARIANTS:
        config_path = CONFIG_DIR / training.VARIANT_CONFIGS[variant]
        cfg = Config.fromfile(str(config_path))
        evaluation = cfg.evaluation.to_dict()
        evaluation["ground_truth_filename"] = str(annotation)
        bindings = training.build_runtime_bindings(
            git_commit=commit,
            variant=variant,
            seed=3407,
            slurm_job_id="1",
            source_config_path=config_path,
            source_config_sha256=training.sha256_file(config_path),
            resolved_config_sha256="c" * 64,
            runtime_config_sha256="d" * 64,
            evaluation_annotation_path=annotation,
            evaluation_class_map_path=class_map,
            evaluation_config=evaluation,
            runtime_pretrain_path=pretrain,
            selector_initialization=cfg.workflow.get("selector_initialization", None),
        )
        assert bindings["variant"] == variant
        expected_gate = tmp_path / "full_model" / f"{config_path.stem}.json"
        assert bindings["full_model_gate_sha256"] == training.sha256_file(
            expected_gate
        )


def test_boundary_gate_launcher_names_artifacts_by_config_stem_and_unlocks_training() -> None:
    launcher = (
        ROOT / "scripts" / "run_duca_boundary_burst_gate_gpu1.sh"
    ).read_text(encoding="utf-8")

    assert 'config_stem="$(basename "${config}" .py)"' in launcher
    assert 'full_model/${config_stem}.json' in launcher
    assert '"schema": "duca_boundary_burst_full_model_gate_v1"' in launcher
    assert '"formal_training_unlocked": True' in launcher
