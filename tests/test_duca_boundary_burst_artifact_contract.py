from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.create_duca_frontend_split import (
    create_split,
    validate_split_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _split(tmp_path: Path) -> tuple[dict, Path]:
    annotation = tmp_path / "annotation.json"
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    **{
                        f"video_train_{index:03d}": {"subset": "training"}
                        for index in range(10)
                    },
                    "video_test_000": {"subset": "validation"},
                }
            }
        ),
        encoding="utf-8",
    )
    payload = create_split(annotation, tmp_path / "split", seed=3407)
    return payload, tmp_path / "split" / "frontend_split_manifest.json"


def test_split_manifest_reopens_every_sealed_artifact(tmp_path: Path) -> None:
    payload, manifest = _split(tmp_path)
    binding = validate_split_manifest(
        manifest,
        annotation_path=payload["annotation_path"],
        train_block_list=payload["train_block_list"],
        holdout_block_list=payload["holdout_block_list"],
    )

    assert binding["ok"] is True
    assert binding["schema"] == "duca_frontend_train_holdout_split_v2"
    assert binding["annotation_sha256"] == payload["annotation_sha256"]
    assert binding["train_block_list_sha256"] == payload["train_block_list_sha256"]
    assert binding["holdout_block_list_sha256"] == payload["holdout_block_list_sha256"]


@pytest.mark.parametrize(
    "field",
    ("annotation_path", "train_block_list", "holdout_block_list"),
)
def test_split_manifest_fails_closed_on_artifact_content_drift(
    tmp_path: Path,
    field: str,
) -> None:
    payload, manifest = _split(tmp_path)
    Path(payload[field]).write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash drift"):
        validate_split_manifest(manifest)


def test_split_manifest_fails_closed_on_runtime_path_substitution(tmp_path: Path) -> None:
    payload, manifest = _split(tmp_path)
    replacement = tmp_path / "replacement.txt"
    replacement.write_text(Path(payload["train_block_list"]).read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(ValueError, match="runtime path"):
        validate_split_manifest(manifest, train_block_list=replacement)


def test_submission_dag_requires_r0_before_every_learned_stage() -> None:
    source = (ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh").read_text(
        encoding="utf-8"
    )

    assert '--dependency="afterok:${r0}" "${P0_SBATCH}"' in source
    assert 'printf \'p0\\t%s\\tafterok:%s\\n\' "${p0}" "${r0}"' in source
    assert '"p0": "afterok:r0_holdout_map"' in source
    assert '"gate": "afterok:p0"' in source
    assert '"official60_arms": "afterok:gate"' in source
    assert '"r0_positive_headroom_required": True' in source


def test_p0_blocks_nonpositive_r0_headroom_before_training() -> None:
    source = (ROOT / "scripts" / "run_duca_boundary_burst_p0_gpu1.sh").read_text(
        encoding="utf-8"
    )

    headroom_gate = source.index("R0 nonpositive Oracle-U headroom blocks learned training")
    real_gate = source.index("run_duca_frontend_p0_real_gate.py")
    first_variant = source.index("run_duca_frontend_pretrain_variant_gpu1.sh")
    assert headroom_gate < real_gate < first_variant
    assert 'digest != row.get("metrics_sha256")' in source
    assert 'if not headroom > 0.0:' in source


def test_uniform_arm_never_claims_a_gaussian_frontend_checkpoint() -> None:
    source = (ROOT / "scripts" / "run_duca_two_stage_curriculum_variant_gpu1.sh").read_text(
        encoding="utf-8"
    )

    assert 'if [[ "${VARIANT}" == "two_stage_exact_uniform" ]]' in source
    assert "unset DUCA_FRONTEND_CHECKPOINT" in source
    assert 'FRONTEND_BINDING="not_applicable_exact_uniform"' in source
    assert "FRONTEND_CHECKPOINT_SHA256_JSON=null" in source
    assert '"two_stage_exact_uniform": "gaussian_matched"' not in source
    assert '"frontend_checkpoint_sha256": ${FRONTEND_CHECKPOINT_SHA256_JSON}' in source


def test_all_split_consumers_reopen_the_shared_hash_contract() -> None:
    paths = (
        "scripts/run_duca_boundary_burst_p0_gpu1.sh",
        "scripts/run_duca_frontend_pretrain_variant_gpu1.sh",
        "tools/bata/run_duca_frontend_p0_real_gate.py",
    )
    for relative in paths:
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "validate_split_manifest" in source or "--validate-manifest" in source
