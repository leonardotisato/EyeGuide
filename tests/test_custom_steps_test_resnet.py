import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CUSTOM_STEPS = ROOT / "src" / "finn_build" / "custom_steps_test_resnet.py"


def get_lower_transformations():
    module = ast.parse(CUSTOM_STEPS.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "step_test_resnet_lower":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "lower_transformations"
                    for target in stmt.targets
                ):
                    names = []
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Call):
                            fn = elt.func
                            if isinstance(fn, ast.Name):
                                names.append(fn.id)
                            elif isinstance(fn, ast.Attribute):
                                names.append(fn.attr)
                    return names
    raise AssertionError("Could not locate lower_transformations in step_test_resnet_lower")


def get_to_hw_transformations():
    module = ast.parse(CUSTOM_STEPS.read_text())
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == "step_test_resnet_to_hw":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "to_hw_transformations"
                    for target in stmt.targets
                ):
                    names = []
                    for elt in stmt.value.elts:
                        if isinstance(elt, ast.Call):
                            fn = elt.func
                            if isinstance(fn, ast.Name):
                                names.append(fn.id)
                            elif isinstance(fn, ast.Attribute):
                                names.append(fn.attr)
                    return names
    raise AssertionError("Could not locate to_hw_transformations in step_test_resnet_to_hw")


class TestTestResnetLowerPipeline(unittest.TestCase):
    def test_lower_step_does_not_rewrite_stem_maxpool_to_nhwc(self):
        transforms = get_lower_transformations()
        self.assertNotIn(
            "MakeMaxPoolNHWC",
            transforms,
            "MakeMaxPoolNHWC in the lower step moves the first residual fork before stem MaxPool.",
        )

    def test_to_hw_reorders_transposes_after_pool_inference(self):
        transforms = get_to_hw_transformations()
        infer_pool_idx = transforms.index("InferPool")
        infer_add_idx = transforms.index("InferAddStreamsLayer")
        between = transforms[infer_pool_idx + 1 : infer_add_idx]
        self.assertIn(
            "MoveTransposePastFork",
            between,
            "InferPool introduces stem-pool transposes late; they must be pushed past the residual fork before AddStreams inference.",
        )
        self.assertIn(
            "AbsorbConsecutiveTransposes",
            between,
            "After moving late transposes past the fork, inverse transpose pairs should collapse before partitioning.",
        )


if __name__ == "__main__":
    unittest.main()
