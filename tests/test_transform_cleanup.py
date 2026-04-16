import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UTILS_DIR = ROOT / "src" / "utils"
SRC_DIR = ROOT / "src"


class TestTransformCleanup(unittest.TestCase):
    def test_transform_modules_follow_the_uniform_naming_scheme(self):
        self.assertTrue((UTILS_DIR / "transforms_224_strong.py").exists())
        self.assertTrue((UTILS_DIR / "transforms_512_light.py").exists())
        self.assertTrue((UTILS_DIR / "transforms_512_strong.py").exists())
        self.assertFalse((UTILS_DIR / "transforms.py").exists())
        self.assertFalse((UTILS_DIR / "transforms_224.py").exists())
        self.assertFalse((UTILS_DIR / "transforms_strong.py").exists())
        self.assertFalse(
            (UTILS_DIR / "transforms_224_light.py").exists(),
            "The unused 224-light transform file should stay retired.",
        )

    def test_active_test_resnet_path_keeps_using_transforms_224(self):
        for rel_path in ("train_test_resnet.py", "qat_test_resnet.py", "export_test_resnet.py"):
            content = (SRC_DIR / rel_path).read_text()
            self.assertIn("transforms_224_strong", content)
            self.assertNotIn("transforms_224_light", content)

    def test_active_512_paths_use_the_new_uniform_module_names(self):
        for rel_path in ("train_custom_net.py", "qat_custom_net.py"):
            content = (SRC_DIR / rel_path).read_text()
            self.assertIn("transforms_512_strong", content)

        for rel_path in (
            "train_mobilenetv1.py",
            "qat_mobilenetv1.py",
            "export_custom_net.py",
            "export_mobilenetv1.py",
            "export_resnet18.py",
            "main.py",
            "zoom_experiment.py",
        ):
            content = (SRC_DIR / rel_path).read_text()
            self.assertIn("transforms_512_light", content)


if __name__ == "__main__":
    unittest.main()
