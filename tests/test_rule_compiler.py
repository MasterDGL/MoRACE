import unittest

from morace.rule_compiler import compile_rule, feature_value


class RuleCompilerTests(unittest.TestCase):
    def test_compile_aromatic_ring_presence(self):
        rule = {
            "source": "data",
            "feature_family": "pattern_presence",
            "feature_name": "aromatic_ring",
            "feature_mode": "binary_presence",
            "relation_to_target": "positive",
            "threshold": None,
            "rationale": "aromaticity matters",
        }
        compiled = compile_rule(rule)
        self.assertIsNotNone(compiled)
        self.assertEqual(feature_value(compiled, "c1ccccc1"), 1.0)


if __name__ == "__main__":
    unittest.main()
