"""CPU-only checks of capability rejection; no fixture is written as a receipt."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from benchmark_capability_precheck import IMPLEMENTATIONS, MODES, load_source_config, validate_witness


class WitnessValidationTests(unittest.TestCase):
    def setUp(self):
        self.witness = dict(status="PASS", is_mock=False, is_scientific_result=False,
            source_commit="fixture-commit", source_train_id="fixture-train", selection_complete=True,
            completed_epochs=60, slurm_job_id="fixture-allocation", gpu_identity={"cuda_uuid": "fixture-gpu"},
            checkpoint_identity=dict(source_train_id="fixture-train", source_commit="fixture-commit",
                                     weights="ema", checkpoint_sha256="fixture-digest"),
            same_plan_output_check="PASS", cases=[dict(implementation=i, mode=m, status="PASS", batch=1,
                warmup=1, sanity_durations_ms=[1.0, 2.0], gpu_start="fixture-gpu", gpu_end="fixture-gpu",
                window_ids=["fixture-video:0"]) for i in IMPLEMENTATIONS for m in MODES])

    def check(self, witness):
        validate_witness(witness, "fixture-commit", "fixture-train")

    def test_complete_contract(self):
        self.check(self.witness)

    def test_reject_incomplete_selection_and_wrong_identity(self):
        changes = [("selection_complete", False), ("completed_epochs", 59), ("source_commit", "other"),
                   ("source_train_id", "other"), ("is_mock", True), ("status", "FAILED"),
                   ("is_scientific_result", True), ("same_plan_output_check", "FAILED"),
                   ("slurm_job_id", ""), ("checkpoint_identity", {})]
        for field, value in changes:
            with self.subTest(field=field):
                witness = copy.deepcopy(self.witness)
                witness[field] = value
                with self.assertRaises(ValueError):
                    self.check(witness)

    def test_reject_missing_duplicate_and_failed_paths(self):
        for kind in ("missing", "duplicate", "failed", "no_isolation", "wrong_batch"):
            with self.subTest(kind=kind):
                witness = copy.deepcopy(self.witness)
                if kind == "missing":
                    witness["cases"].pop()
                elif kind == "duplicate":
                    witness["cases"][-1] = copy.deepcopy(witness["cases"][0])
                else:
                    key, value = {"failed": ("status", "FAILED"), "no_isolation": ("gpu_end", ""),
                                  "wrong_batch": ("batch", 8)}[kind]
                    witness["cases"][0][key] = value
                with self.assertRaises(ValueError):
                    self.check(witness)

    def test_reject_unfinished_or_invalid_durations(self):
        for values in ([], [1.0], [0, 1], [-1, 1], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(values=values):
                witness = copy.deepcopy(self.witness)
                witness["cases"][0]["sanity_durations_ms"] = values
                with self.assertRaises(ValueError):
                    self.check(witness)

    @unittest.skipUnless(importlib.util.find_spec("mmengine"), "requires the production mmengine environment")
    def test_saved_python_config_preserves_resize_tuple(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "resolved_opentad.py"
            path.write_text("dataset = dict(test=dict(pipeline=[dict(type='Resize', scale=(160, 160))]))\n")
            resolved = dict(dataset=dict(test=dict(pipeline=[dict(type="Resize", scale=[160, 160])])))
            cfg = load_source_config(path, resolved)
            self.assertIsInstance(cfg.dataset.test.pipeline[0].scale, tuple)
            wrong = json.loads(json.dumps(resolved))
            wrong["dataset"]["test"]["pipeline"][0]["scale"] = [224, 224]
            with self.assertRaises(ValueError):
                load_source_config(path, wrong)


if __name__ == "__main__":
    unittest.main()
