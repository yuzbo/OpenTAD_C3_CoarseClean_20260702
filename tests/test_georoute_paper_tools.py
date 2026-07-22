import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "bata"))

from analyze_georoute_results import analyze  # noqa: E402
from georoute_result_schema import (  # noqa: E402
    GeoRouteResultSchemaError,
    SCHEMA_VERSION,
    VALID_VARIANTS,
    validate_records,
)
from plot_georoute_paper import COLORS, MARKERS  # noqa: E402


def make_record(variant="free_token_select", seed=3407, budget=64):
    selected = 240 if variant == "dense_native" else budget
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": "georoute-dev-unit-test",
        "stage": "P1",
        "split_role": "development",
        "dataset": "THUMOS14-dev",
        "detector": "AdaTAD-derived",
        "variant": variant,
        "seed": seed,
        "budget": {
            "tokens_per_tubelet": budget,
            "source_tokens_per_tubelet": 240,
            "selected_tokens_per_tubelet": selected,
        },
        "metrics": {
            "average_map": 60.0 + (1.0 if variant == "roi_residual" else 0.0),
            "map_by_tiou": {"0.3": 75.0, "0.4": 70.0, "0.5": 65.0, "0.6": 55.0, "0.7": 43.0},
        },
        "cost": {
            "end_to_end_p50_ms": 20.0 + (1.0 if variant == "roi_residual" else 0.0),
            "end_to_end_p95_ms": 25.0,
            "peak_memory_mb": 2048.0,
            "gross_gpu_energy_j": 12.0,
            "scope": {
                "decode": True,
                "preprocess": True,
                "host_to_device": True,
                "scout": True,
                "route": True,
                "patch_embed": True,
                "backbone": True,
                "adapter": True,
                "detector": True,
                "nms": True,
            },
        },
        "evidence": {
            "runtime_commit": "a" * 40,
            "config_sha256": "b" * 64,
            "checkpoint_sha256": "c" * 64,
            "prediction_sha256": "d" * 64,
            "run_receipt_sha256": "e" * 64,
        },
        "diagnostics": {
            "one_heavy_backbone_forward": True,
            "uses_grid_sample": False,
            "uses_resized_local_crop": False,
            "packed_attention_tokens": float(selected),
            "packed_mlp_tokens": float(selected),
            "roi_area_fraction": 0.3,
            "roi_center_velocity": 0.04,
            "residual_token_fraction": 0.25,
            "training_curve": [{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.8}],
        },
        "policy_estimator": "score_function",
        "score_function_kat_passed": True,
    }


class GeoRoutePaperToolsTest(unittest.TestCase):
    def test_development_only_rejects_official_test(self):
        record = make_record()
        record["split_role"] = "official_test"
        with self.assertRaisesRegex(GeoRouteResultSchemaError, "official_test"):
            validate_records([record], development_only=True)

    def test_st_requires_explicit_bias_label(self):
        record = make_record()
        record["policy_estimator"] = "straight_through"
        record.pop("score_function_kat_passed")
        with self.assertRaisesRegex(GeoRouteResultSchemaError, "biased_surrogate"):
            validate_records([record])
        record["estimator_bias_label"] = "biased_surrogate"
        validate_records([record])

    def test_analysis_pairs_structured_and_free_without_claim(self):
        records = [
            make_record("free_token_select", 3407),
            make_record("roi_residual", 3407),
            make_record("free_token_select", 3408),
            make_record("roi_residual", 3408),
        ]
        output = analyze(records, development_only=True, structured_variant="roi_residual")
        self.assertEqual(output["raw_record_count"], 4)
        self.assertEqual(len(output["paired_structured_vs_free"]["raw_pairs"]), 2)
        self.assertIn("No mAP theorem", output["non_claims"][0])
        self.assertTrue(output["analysis_sha256"])

    def test_plotter_writes_only_to_requested_external_directory(self):
        records = [
            make_record("free_token_select", 3407),
            make_record("roi_residual", 3407),
        ]
        analysis = analyze(records, development_only=True, structured_variant="roi_residual")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            records_path = temp_dir / "records.json"
            analysis_path = temp_dir / "analysis.json"
            output_dir = temp_dir / "plots"
            records_path.write_text(json.dumps(records), encoding="utf-8")
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "bata" / "plot_georoute_paper.py"),
                    "--records",
                    str(records_path),
                    "--analysis",
                    str(analysis_path),
                    "--output-dir",
                    str(output_dir),
                    "--format",
                    "png",
                    "--development-only",
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertIn("georoute_accuracy_cost_pareto.png", completed.stdout)
            self.assertTrue((output_dir / "georoute_accuracy_cost_pareto.png").is_file())

    def test_plotter_has_a_distinct_style_for_every_valid_variant(self):
        self.assertTrue(VALID_VARIANTS.issubset(COLORS))
        self.assertTrue(VALID_VARIANTS.issubset(MARKERS))

    def test_table_renderer_preserves_raw_rows_and_writes_external_outputs(self):
        records = [
            make_record("fixed_lattice_geometry", 3407),
            make_record("free_token_select", 3407),
            make_record("roi_residual", 3407),
        ]
        analysis = analyze(records, development_only=True, structured_variant="roi_residual")
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            records_path = temp_dir / "records.json"
            analysis_path = temp_dir / "analysis.json"
            output_dir = temp_dir / "tables"
            records_path.write_text(json.dumps(records), encoding="utf-8")
            analysis_path.write_text(json.dumps(analysis), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "bata" / "render_georoute_paper_tables.py"),
                    "--records",
                    str(records_path),
                    "--analysis",
                    str(analysis_path),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertIn("georoute_raw_seed_table.csv", completed.stdout)
            raw_table = output_dir / "georoute_raw_seed_table.csv"
            inventory = output_dir / "georoute_evidence_inventory.md"
            self.assertTrue(raw_table.is_file())
            self.assertTrue((output_dir / "georoute_summary_table.tex").is_file())
            self.assertIn("fixed_lattice_geometry", raw_table.read_text(encoding="utf-8"))
            self.assertIn("incomplete", inventory.read_text(encoding="utf-8"))

    def test_architecture_plot_is_generated_from_the_checked_in_figurespec(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "georoute_architecture.pdf"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "bata" / "plot_georoute_architecture.py"),
                    "--spec",
                    str(REPO_ROOT / "docs" / "methods" / "georoute_adatad_architecture_spec.json"),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
            self.assertIn("georoute_architecture.pdf", completed.stdout)
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
