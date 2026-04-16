import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"


class TestMobilenetScriptCleanup(unittest.TestCase):
    def test_only_one_canonical_mobilenetv1_script_family_remains(self):
        finn_dir = SRC_DIR / "finn_build"
        self.assertTrue((SRC_DIR / "train_mobilenetv1.py").exists())
        self.assertTrue((SRC_DIR / "qat_mobilenetv1.py").exists())
        self.assertTrue((SRC_DIR / "export_mobilenetv1.py").exists())
        self.assertTrue((finn_dir / "build_mobilenetv1.py").exists())
        self.assertTrue((finn_dir / "custom_steps_mobilenetv1.py").exists())
        self.assertFalse(
            (SRC_DIR / "train_mobilenet.py").exists(),
            "The intermediate mobilenet trainer name should be retired once train_mobilenetv1.py becomes canonical.",
        )
        self.assertFalse((SRC_DIR / "qat_mobilenet.py").exists())
        self.assertFalse((SRC_DIR / "export_mobilenet.py").exists())
        self.assertFalse((finn_dir / "build_mobilenet.py").exists())
        self.assertFalse((finn_dir / "custom_steps_mobilenet.py").exists())

    def test_canonical_train_script_preserves_archived_kd_lineage(self):
        content = (SRC_DIR / "train_mobilenetv1.py").read_text()
        self.assertIn('"mobilenetv1_fp32_kd.pth"', content)
        self.assertIn('KD_ALPHA = 0.5', content)
        self.assertIn('timm.create_model("mobilenetv1_100", pretrained=True', content)

    def test_qat_export_and_build_paths_stay_aligned_to_mobilenetv1(self):
        qat_content = (SRC_DIR / "qat_mobilenetv1.py").read_text()
        export_content = (SRC_DIR / "export_mobilenetv1.py").read_text()
        build_content = (SRC_DIR / "finn_build" / "build_mobilenetv1.py").read_text()

        self.assertIn('"mobilenetv1_fp32_kd.pth"', qat_content)
        self.assertIn('f"mobilenet_{tag}_qat.pth"', qat_content)
        self.assertIn('f"mobilenet_{tag}.onnx"', export_content)
        self.assertIn("from custom_steps_mobilenetv1 import", build_content)
        self.assertIn('--onnx", default="models/mobilenet_8w8a.onnx"', build_content)

    def test_run_sh_uses_only_canonical_mobilenetv1_commands(self):
        content = (ROOT / "run.sh").read_text()
        self.assertIn('train_mobilenetv1)      SCRIPT="src/train_mobilenetv1.py"', content)
        self.assertIn('qat_mobilenetv1)        SCRIPT="src/qat_mobilenetv1.py"', content)
        self.assertIn('export_mobilenetv1)     SCRIPT="src/export_mobilenetv1.py"', content)
        self.assertNotIn("train_mobilenet)", content)
        self.assertNotIn("qat_mobilenet)", content)
        self.assertNotIn("export_mobilenet)", content)


if __name__ == "__main__":
    unittest.main()
