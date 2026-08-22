from pathlib import Path
import ast
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "visualize_zoomtoken_token_selection.py"


class TokenSelectionVisualizationContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SCRIPT.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_all_frozen_rows_are_present_once(self):
        for label in (
            '"B"',
            '"C"',
            '"R1"',
            '"R2"',
            '"R2-SHUF48"',
            '"Q48-GLOBAL"',
            '"R3"',
            '"R3-AREA-SHIFT"',
            '"R4"',
            '"R4-SHUF15"',
            '"Q64-GLOBAL"',
        ):
            self.assertIn(f"ArmDefinition({label},", self.source)

    def test_production_route_is_called_and_heavy_forward_is_not(self):
        calls = [
            node.func.attr
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("_official_fixed_support_route", calls)
        self.assertNotIn("forward_native_ragged", calls)
        self.assertNotIn("forward", calls)

    def test_caption_preserves_evidence_boundary(self):
        self.assertIn("on-policy", self.source)
        self.assertIn("current recovery checkpoint; qualitative observation only", self.source)
        self.assertIn("不提供精度或成本结论", self.source)

    def test_mask_semantics_are_color_selected_and_gray_unselected(self):
        self.assertIn("np.where(pixel_mask[..., None], frame, gray_rgb)", self.source)
        self.assertIn("selected_spatial_indices", self.source)
        self.assertIn("selected_physical_indices", self.source)

    def test_single_sample_is_reused_for_every_arm(self):
        sample_calls = [
            node
            for node in ast.walk(self.tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_select_sample"
        ]
        self.assertEqual(len(sample_calls), 1)
        self.assertIn("for arm in ARMS", self.source)
        self.assertIn("routing_gt_used", self.source)


if __name__ == "__main__":
    unittest.main()
