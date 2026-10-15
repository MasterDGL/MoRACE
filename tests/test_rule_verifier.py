import unittest

from morace.rule_verifier import passes_rule_gate


class RuleVerifierTests(unittest.TestCase):
    def test_data_rule_requires_support(self):
        rule = {
            "compile_valid": True,
            "support_count": 1,
            "support_sources": ["data"],
            "p_value": 0.01,
            "effect_size": 0.5,
            "feature_entropy": 0.3,
            "non_nan_coverage": 0.9,
        }
        self.assertFalse(passes_rule_gate(rule))


if __name__ == "__main__":
    unittest.main()
