import importlib.util
from pathlib import Path
import sys
import types
import unittest

rule_registry_stub = types.ModuleType("morace.rule_registry")
rule_registry_stub.DESCRIPTOR_REGISTRY = {
    "MolWt": object(),
    "MolLogP": object(),
    "NumHBD": object(),
}
rule_registry_stub.PATTERN_ALIAS = {
    "amine": "amine",
    "amide": "amide",
}
rule_registry_stub.COUNT_ALIAS = {
    "NumHBD": "NumHBD",
    "NumHBA": "NumHBA",
}


def _canonicalize_feature_name(feature_family, feature_name):
    cleaned = str(feature_name or "").strip()
    descriptor_aliases = {"molecular weight": "MolWt", "molwt": "MolWt", "logp": "MolLogP"}
    if feature_family == "descriptor":
        return descriptor_aliases.get(cleaned.lower(), cleaned if cleaned in rule_registry_stub.DESCRIPTOR_REGISTRY else "")
    if feature_family == "pattern_presence":
        return cleaned if cleaned in {"amine", "amide"} else ""
    if feature_family == "count":
        return cleaned if cleaned in {"NumHBD", "NumHBA"} else ""
    return ""


rule_registry_stub.canonicalize_feature_name = _canonicalize_feature_name

_previous_rule_registry = sys.modules.get("morace.rule_registry")
sys.modules["morace.rule_registry"] = rule_registry_stub
try:
    _schemas_path = Path(__file__).resolve().parents[1] / "morace/schemas.py"
    _schemas_spec = importlib.util.spec_from_file_location("schemas_under_test", _schemas_path)
    schemas_under_test = importlib.util.module_from_spec(_schemas_spec)
    assert _schemas_spec.loader is not None
    _schemas_spec.loader.exec_module(schemas_under_test)
finally:
    if _previous_rule_registry is None:
        sys.modules.pop("morace.rule_registry", None)
    else:
        sys.modules["morace.rule_registry"] = _previous_rule_registry

heuristic_extract_rules_from_draft = schemas_under_test.heuristic_extract_rules_from_draft
parse_projected_rules = schemas_under_test.parse_projected_rules
strict_parse_projected_rules = schemas_under_test.strict_parse_projected_rules


class SchemaParseTests(unittest.TestCase):
    def test_parse_projected_rules_with_code_fence(self):
        raw = """```json
        {"rules":[{"source":"data","feature_family":"descriptor","feature_name":"MolLogP","relation_to_target":"positive","threshold":{"op":">=","value":2.0,"unit":null},"rationale":"higher logP may help"}]}
        ```"""
        rules = parse_projected_rules(raw, source="data", provenance={"draft_model": "qwen", "draft_id": 0, "batch_id": 1})
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["feature_name"], "MolLogP")

    def test_heuristic_sentence_scoped_threshold(self):
        draft = (
            "Determinant A: hydrogen bond donor count may reduce permeability. "
            "Determinant B: molecular weight below 500 may still be favorable."
        )
        rules = heuristic_extract_rules_from_draft(draft, source="data", provenance={"draft_model": "g", "draft_id": 0, "batch_id": 0})
        hbd = next(rule for rule in rules if rule["feature_name"] == "NumHBD")
        mw = next(rule for rule in rules if rule["feature_name"] == "MolWt")
        self.assertIsNone(hbd["threshold"])
        self.assertIsNotNone(mw["threshold"])

    def test_strict_parse_reports_trailing_text_or_prefix(self):
        raw = 'prefix {"rules":[{"source":"data","feature_family":"descriptor","feature_name":"MolLogP"}]}'
        rules, meta = strict_parse_projected_rules(
            raw,
            source="data",
            provenance={"draft_model": "chemdfm", "draft_id": 0, "batch_id": 1},
        )
        self.assertEqual(rules, [])
        self.assertEqual(meta["failure_stage"], "json_extraction")
        self.assertEqual(meta["failure_reason"], "trailing_text_or_prefix")

    def test_strict_parse_reports_source_pollution_but_normalizes_rule(self):
        raw = """{
          "rules": [
            {
              "source": "Prior chemistry draft Determinant A: MolWt positive",
              "feature_family": "descriptor",
              "feature_name": "MolWt",
              "feature_mode": "continuous",
              "relation_to_target": "positive"
            }
          ]
        }"""
        rules, meta = strict_parse_projected_rules(
            raw,
            source="literature",
            provenance={"draft_model": "chemdfm", "draft_id": 0, "batch_id": 1},
        )
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["source"], "literature")
        self.assertEqual(rules[0]["primary_source"], "literature")
        self.assertEqual(meta["failure_reason"], "source_polluted")

    def test_strict_parse_reports_rules_empty_after_normalization(self):
        raw = """{
          "rules": [
            {
              "source": "data",
              "feature_family": "descriptor",
              "feature_name": "NotARealFeature"
            }
          ]
        }"""
        rules, meta = strict_parse_projected_rules(
            raw,
            source="data",
            provenance={"draft_model": "chemdfm", "draft_id": 0, "batch_id": 1},
        )
        self.assertEqual(rules, [])
        self.assertEqual(meta["failure_stage"], "strict_semantic_validation")
        self.assertEqual(meta["failure_reason"], "unknown_or_unmapped_field_value")

    def test_llama_native_strict_parse_strips_known_assistant_wrapper(self):
        raw = 'Assistant: {"rules":[{"source":"data","feature_family":"descriptor","feature_name":"MolLogP"}]}'
        rules, meta = strict_parse_projected_rules(
            raw,
            source="data",
            provenance={
                "draft_model": "llama-3.1-8b-instruct",
                "draft_id": 0,
                "batch_id": 1,
                "stage2_protocol_profile": "llama_native",
            },
        )
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0]["feature_name"], "MolLogP")
        self.assertEqual(meta["failure_reason"], "")

    def test_non_llama_strict_parse_does_not_strip_wrapper(self):
        raw = 'Assistant: {"rules":[{"source":"data","feature_family":"descriptor","feature_name":"MolLogP"}]}'
        rules, meta = strict_parse_projected_rules(
            raw,
            source="data",
            provenance={"draft_model": "chemdfm", "draft_id": 0, "batch_id": 1},
        )
        self.assertEqual(rules, [])
        self.assertEqual(meta["failure_stage"], "json_extraction")
        self.assertEqual(meta["failure_reason"], "trailing_text_or_prefix")


if __name__ == "__main__":
    unittest.main()
