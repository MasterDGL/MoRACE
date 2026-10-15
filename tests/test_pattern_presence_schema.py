import importlib.util
import sys
import types
import unittest
from pathlib import Path


SCHEMAS_PATH = Path(__file__).resolve().parents[1] / "morace/schemas.py"


def load_schemas_module():
    previous_rule_registry = sys.modules.get("morace.rule_registry")
    try:
        fake_rule_registry = types.ModuleType("morace.rule_registry")
        fake_rule_registry.COUNT_ALIAS = {}
        fake_rule_registry.DESCRIPTOR_REGISTRY = {"MolLogP": object()}
        fake_rule_registry.PATTERN_ALIAS = {
            "amine": "[NX3;H2,H1;!$(NC=O)]",
            "fused_ring": "fused_ring_pattern",
            "nitro_group": "nitro_pattern",
        }

        def canonicalize_feature_name(feature_family, feature_name):
            raw = str(feature_name or "").strip()
            lowered = raw.lower()
            if feature_family == "descriptor":
                return "MolLogP" if lowered == "mollogp" else None
            if feature_family == "pattern_presence":
                alias_map = {
                    "amine": "amine",
                    "fused_ring": "fused_ring",
                    "nitro_group": "nitro_group",
                }
                return alias_map.get(lowered)
            if feature_family == "count":
                return None
            return None

        fake_rule_registry.canonicalize_feature_name = canonicalize_feature_name
        sys.modules["morace.rule_registry"] = fake_rule_registry

        spec = importlib.util.spec_from_file_location("d2_schemas_under_test", SCHEMAS_PATH)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_rule_registry is None:
            sys.modules.pop("morace.rule_registry", None)
        else:
            sys.modules["morace.rule_registry"] = previous_rule_registry


class PatternPresenceSchemaTests(unittest.TestCase):
    def setUp(self):
        self.schemas = load_schemas_module()

    def test_normalize_rule_accepts_explicit_pattern_presence(self):
        normalized = self.schemas.normalize_rule(
            {
                "feature_family": "pattern_presence",
                "feature_name": "amine",
                "feature_mode": "binary_presence",
                "relation_to_target": "positive",
                "threshold": None,
                "rationale": "amine is relevant",
            },
            source="literature",
            provenance={},
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["feature_family"], "pattern_presence")
        self.assertEqual(normalized["feature_name"], "amine")
        self.assertEqual(normalized["feature_mode"], "binary_presence")

    def test_normalize_rule_keeps_legacy_pattern_alias(self):
        normalized = self.schemas.normalize_rule(
            {
                "feature_family": "pattern",
                "feature_name": "amine",
                "feature_mode": "binary_presence",
                "relation_to_target": "positive",
                "threshold": None,
                "rationale": "amine is relevant",
            },
            source="literature",
            provenance={},
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["feature_family"], "pattern_presence")
        self.assertEqual(normalized["feature_name"], "amine")

    def test_normalize_rule_accepts_real_pattern_registry_name(self):
        normalized = self.schemas.normalize_rule(
            {
                "feature_family": "pattern_presence",
                "feature_name": "fused_ring",
                "feature_mode": "binary_presence",
                "relation_to_target": "negative",
                "threshold": None,
                "rationale": "fused ring is relevant",
            },
            source="literature",
            provenance={},
        )
        self.assertIsNotNone(normalized)
        self.assertEqual(normalized["feature_family"], "pattern_presence")
        self.assertEqual(normalized["feature_name"], "fused_ring")


if __name__ == "__main__":
    unittest.main()
