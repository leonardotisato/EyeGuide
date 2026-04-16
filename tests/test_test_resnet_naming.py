import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"


class TestTestResnetNaming(unittest.TestCase):
    def test_canonical_fp32_checkpoint_uses_standard_kd_name(self):
        self.assertTrue(
            (MODELS_DIR / "test_resnet_fp32_kd.pth").exists(),
            "The canonical test_resnet FP32 checkpoint should use the standard KD name.",
        )
        self.assertFalse(
            (MODELS_DIR / "test_resnet_fp32_kd_uw_strong.pth").exists(),
            "The old experiment-specific FP32 checkpoint name should be retired once the canonical name is chosen.",
        )

    def test_active_qat_path_uses_standard_kd_name(self):
        qat_script = (ROOT / "src" / "qat_test_resnet.py").read_text()
        self.assertIn('"test_resnet_fp32_kd.pth"', qat_script)
        self.assertNotIn('"test_resnet_fp32_kd_uw_strong.pth"', qat_script)

    def test_active_project_docs_use_standard_kd_name(self):
        for rel_path in ("AGENTS.md", "CLAUDE.md"):
            content = (ROOT / rel_path).read_text()
            self.assertIn("models/test_resnet_fp32_kd.pth", content)
            self.assertNotIn("models/test_resnet_fp32_kd_uw_strong.pth", content)


if __name__ == "__main__":
    unittest.main()
