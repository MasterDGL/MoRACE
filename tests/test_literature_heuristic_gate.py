import importlib.util
import sys
import types
import unittest
from pathlib import Path


RULE_VERIFIER_PATH = Path(__file__).resolve().parents[1] / "morace/rule_verifier.py"


def load_rule_verifier_module():
    previous_modules = {
        name: sys.modules.get(name)
        for name in ("morace.rule_compiler", "scipy", "scipy.stats")
    }
    try:
        fake_rule_compiler = types.ModuleType("morace.rule_compiler")
        fake_rule_compiler.build_feature_matrix = lambda compiled_rules, smiles_train: [[0.0] for _ in smiles_train]
        fake_rule_compiler.compile_rule = lambda rule: object()
        sys.modules["morace.rule_compiler"] = fake_rule_compiler

        fake_scipy = types.ModuleType("scipy")
        fake_scipy_stats = types.ModuleType("scipy.stats")
        fake_scipy_stats.fisher_exact = lambda table: (0.0, 1.0)
        fake_scipy_stats.mannwhitneyu = lambda pos, neg, alternative="two-sided": (0.0, 1.0)
        fake_scipy_stats.spearmanr = lambda values, y: (0.0, 1.0)
        fake_scipy.stats = fake_scipy_stats
        sys.modules["scipy"] = fake_scipy
        sys.modules["scipy.stats"] = fake_scipy_stats

        spec = importlib.util.spec_from_file_location("d2_rule_verifier_under_test", RULE_VERIFIER_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class LiteratureHeuristicGateTests(unittest.TestCase):
    def setUp(self):
        self.rule_verifier = load_rule_verifier_module()

    def test_literature_only_heuristic_rule_uses_relaxed_gate(self):
        rule = {
            "compile_valid": True,
            "support_sources": [" literature "],
            "extraction_source": "heuristic",
            "projected_support_count": 0,
            "heuristic_support_count": 1,
            "support_count": 1,
            "p_value": 0.024,
            "effect_size": 0.061,
            "feature_entropy": 0.94,
            "non_nan_coverage": 1.0,
        }
        reasons = self.rule_verifier.gate_failure_reasons(rule)
        self.assertEqual(reasons, [])

    def test_non_literature_heuristic_rule_keeps_strict_gate(self):
        rule = {
            "compile_valid": True,
            "support_sources": ["data"],
            "extraction_source": "heuristic",
            "projected_support_count": 0,
            "heuristic_support_count": 1,
            "support_count": 1,
            "p_value": 0.024,
            "effect_size": 0.061,
            "feature_entropy": 0.94,
            "non_nan_coverage": 1.0,
        }
        reasons = self.rule_verifier.gate_failure_reasons(rule)
        self.assertIn("support<3", reasons)
        self.assertIn("effect_size<0.08", reasons)

    def test_support_source_normalization_handles_none_and_string(self):
        normalized_none = self.rule_verifier._normalized_support_sources({"support_sources": None})
        normalized_string = self.rule_verifier._normalized_support_sources({"support_sources": " Literature "})
        self.assertEqual(normalized_none, set())
        self.assertEqual(normalized_string, {"literature"})

    def test_source_bucket_uses_normalized_support_sources(self):
        self.assertEqual(
            self.rule_verifier.source_bucket({"support_sources": " Literature "}),
            "literature",
        )
        self.assertEqual(
            self.rule_verifier.source_bucket({"support_sources": None, "primary_source": "data"}),
            "data",
        )


if __name__ == "__main__":
    unittest.main()
