import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


class TestCustomNetScriptCleanup(unittest.TestCase):
    def test_custom_net_utils_use_the_simplified_names(self):
        utils_dir = SRC_DIR / "utils"
        self.assertTrue((utils_dir / "custom_net.py").exists())
        self.assertTrue((utils_dir / "quant_custom_net.py").exists())
        self.assertFalse((utils_dir / "custom_small_net.py").exists())
        self.assertFalse((utils_dir / "quant_custom_small_net.py").exists())

    def test_only_one_canonical_train_script_remains(self):
        self.assertTrue((SRC_DIR / "train_custom_net.py").exists())
        self.assertFalse(
            (SRC_DIR / "train_custom_net_kd.py").exists(),
            "The separate KD trainer should be retired once train_custom_net.py becomes canonical.",
        )

    def test_canonical_train_script_matches_best_archived_branch(self):
        content = (SRC_DIR / "train_custom_net.py").read_text()
        self.assertIn("from utils.custom_net import CustomSmallNet", content)
        self.assertIn("MULTIPLIER = 3", content)
        self.assertIn("from utils.transforms_512_strong import", content)
        self.assertIn("class_weights = 1.0 / label_counts", content)
        self.assertIn("criterion = nn.CrossEntropyLoss(weight=class_weights)", content)
        self.assertIn('"custom_net_m{MULTIPLIER}_fp32.pth"', content)
        self.assertNotIn("from utils.transforms_512_light import", content)
        self.assertNotIn("ResNet18Classifier", content)
        self.assertNotIn("kd_loss(", content)

    def test_qat_export_and_build_use_plain_custom_net_lineage(self):
        qat_content = (SRC_DIR / "qat_custom_net.py").read_text()
        export_content = (SRC_DIR / "export_custom_net.py").read_text()
        build_content = (SRC_DIR / "finn_build" / "build_custom_net.py").read_text()

        self.assertIn("from utils.quant_custom_net import", qat_content)
        self.assertIn("from utils.quant_custom_net import", export_content)
        self.assertIn('"custom_net_m{MULTIPLIER}_fp32.pth"', qat_content)
        self.assertNotIn("custom_small_net_m{MULTIPLIER}_fp32_kd.pth", qat_content)
        self.assertNotIn("train_custom_net_kd.py", qat_content)
        self.assertIn('"custom_net_m{MULTIPLIER}_{tag}_qat.pth"', qat_content)
        self.assertIn('"custom_net_m{MULTIPLIER}_{tag}.onnx"', export_content)
        self.assertIn('--onnx", default="models/custom_net_m3_8w8a.onnx"', build_content)

    def test_run_sh_exposes_only_the_canonical_custom_net_command(self):
        content = (ROOT / "run.sh").read_text()
        self.assertIn('train_custom_net)       SCRIPT="src/train_custom_net.py"', content)
        self.assertNotIn("train_custom_net_kd", content)


if __name__ == "__main__":
    unittest.main()
