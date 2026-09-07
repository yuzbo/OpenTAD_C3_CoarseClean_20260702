import copy

import pytest

from tools.bata.seal_bafdr_receipt import build_seal, validate_seal, verify_file


def receipts():
    common = dict(protocol_id="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001", expected_epochs=60,
                  expected_total_updates=6000, total_successful_updates=6000,
                  arm="BAFDR-K16-FULL", seed=4407, checkpoint="epoch_59.pth",
                  checkpoint_sha256="original-checkpoint", teacher_identity={"teacher_model": "D160"})
    metrics = {key: 0.5 for key in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")}
    return (dict(common, phase="metric_opening", metric_opened=True, commit_sha="original-evaluator", eval_results=metrics),
            dict(common, phase="training", commit_sha="original-training"))


def test_seal_preserves_original_receipts_and_commit():
    evaluation, training = receipts()
    before = copy.deepcopy((evaluation, training))
    sealed = build_seal(evaluation, training, {"sealing_commit": "new-sealer", "inference_repeated": False})
    assert (evaluation, training) == before
    assert sealed["commit_sha"] == "original-evaluator"
    assert sealed["retrospective_sealing"]["sealing_commit"] == "new-sealer"
    assert sealed["retrospective_sealing"]["training_receipt"] == training
    validate_seal(sealed)
    sealed["eval_results"] = {"average_mAP": 0.9}
    with pytest.raises(ValueError, match="self-hash"):
        validate_seal(sealed)


@pytest.mark.parametrize("field,value", [("total_successful_updates", 5999), ("phase", "prediction_seal"),
                                          ("metric_opened", False), ("seed", 3407)])
def test_seal_rejects_incomplete_or_mismatched_receipt(field, value):
    evaluation, training = receipts()
    evaluation[field] = value
    with pytest.raises(ValueError):
        build_seal(evaluation, training, {})


def test_seal_rejects_missing_metrics():
    evaluation, training = receipts()
    evaluation["eval_results"].pop("mAP@0.7")
    with pytest.raises(ValueError, match="metric"):
        build_seal(evaluation, training, {})


def test_artifact_binding_is_verified(tmp_path):
    path = tmp_path / "config.py"
    path.write_bytes(b"original")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        verify_file(path, "wrong")
    assert path.read_bytes() == b"original"
