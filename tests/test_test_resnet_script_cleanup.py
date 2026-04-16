import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


class TestTestResnetScriptCleanup(unittest.TestCase):
    def test_only_one_canonical_train_script_remains(self):
        self.assertTrue((SRC_DIR / "train_test_resnet.py").exists())
        self.assertFalse(
            (SRC_DIR / "train_test_resnet_exp.py").exists(),
            "The experiment-matrix script should be retired once train_test_resnet.py becomes canonical.",
        )
        self.assertFalse(
            (SRC_DIR / "train_test_resnet_kd.py").exists(),
            "The separate KD-only script should be retired once train_test_resnet.py becomes canonical.",
        )

    def test_canonical_train_script_is_fixed_to_final_configuration(self):
        content = (SRC_DIR / "train_test_resnet.py").read_text()
        self.assertIn('"test_resnet_fp32_kd.pth"', content)
        self.assertIn('"resnet18_fp32_kd.pth"', content)
        self.assertIn("from utils.transforms_224_strong import", content)
        self.assertIn("from utils.transforms_512_strong import", content)
        self.assertNotIn("USE_KD", content)
        self.assertNotIn("USE_WEIGHTED", content)
        self.assertNotIn("USE_STRONG_AUG", content)
        self.assertNotIn("exp_tag()", content)

    def test_run_sh_uses_only_canonical_test_resnet_command(self):
        content = (ROOT / "run.sh").read_text()
        self.assertIn('train_test_resnet)      SCRIPT="src/train_test_resnet.py"', content)
        self.assertNotIn("train_test_resnet_kd", content)
        self.assertNotIn("exp_test_resnet", content)
        self.assertNotIn("USE_KD", content)
        self.assertNotIn("USE_WEIGHTED", content)
        self.assertNotIn("USE_STRONG_AUG", content)


if __name__ == "__main__":
    unittest.main()
