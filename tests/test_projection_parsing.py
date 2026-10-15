import json
import unittest

from morace.projection_parsing import (
    clean_projected_rule,
    extract_balanced_objects_from_array_prefix,
    extract_cleaned_rule_json,
    extract_first_balanced_json_object,
    find_rules_array_start,
    new_extraction_diagnostics,
    salvage_rules_array_prefix,
)


VALID_COUNT_RULE = {
    "feature_family": "count",
    "feature_name": "NumHBA",
    "feature_mode": "count_value",
    "relation_to_target": "negative",
    "threshold": None,
}


class ProjectionParsingTests(unittest.TestCase):
    def test_extracts_first_balanced_object_without_losing_braces_in_strings(self):
        raw = 'prefix {"rules":[],"note":"literal } brace"} trailing {"rules":[1]}'
        payload, audit = extract_first_balanced_json_object(raw)

        self.assertEqual(json.loads(payload), {"rules": [], "note": "literal } brace"})
        self.assertTrue(audit["prefix_removed"])
        self.assertTrue(audit["suffix_removed"])
        self.assertTrue(audit["multiple_json_objects_detected"])

    def test_cleans_supported_rule_and_overwrites_conflicting_source(self):
        diagnostics = new_extraction_diagnostics("tolerant_first_json_object")
        rule = {**VALID_COUNT_RULE, "source": "literature", "extra": "drop me"}

        cleaned = clean_projected_rule(rule, "data", diagnostics)

        self.assertEqual(cleaned["source"], "data")
        self.assertNotIn("extra", cleaned)
        self.assertEqual(diagnostics["invalid_field_dropped_count"], 1)
        self.assertEqual(diagnostics["source_overwritten_count"], 1)

    def test_safe_family_policy_corrects_registry_family_and_mode(self):
        diagnostics = new_extraction_diagnostics(
            "tolerant_first_json_object",
            "safe_enum_feature_canonicalization",
            "safe_family_feature_consistency",
        )
        rule = {
            "feature_family": "descriptor",
            "feature_name": "NumHBA",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
            "threshold": {"op": ">=", "value": 2},
        }

        cleaned = clean_projected_rule(rule, "data", diagnostics)

        self.assertEqual(cleaned["feature_family"], "count")
        self.assertEqual(cleaned["feature_mode"], "count_value")
        self.assertIsNone(cleaned["threshold"])
        self.assertEqual(diagnostics["family_feature_corrected_count"], 1)

    def test_salvages_complete_rules_from_truncated_array(self):
        first = json.dumps(VALID_COUNT_RULE)
        raw = '{"rules":[' + first + ',{"feature_family":"descriptor"'
        start, audit = find_rules_array_start(raw)
        objects, object_audit = extract_balanced_objects_from_array_prefix(raw, start)

        self.assertTrue(audit["rules_array_prefix_found"])
        self.assertEqual(len(objects), 1)
        self.assertTrue(object_audit["stopped_on_unclosed_object"])

        cleaned_json, diagnostics = salvage_rules_array_prefix(raw, "data", "tolerant_first_json_object")
        self.assertEqual(len(json.loads(cleaned_json)["rules"]), 1)
        self.assertEqual(diagnostics["d4_rules_array_cleaned_rule_count"], 1)

    def test_extract_cleaned_rule_json_reports_missing_object(self):
        cleaned_json, diagnostics = extract_cleaned_rule_json(
            "not json",
            "data",
            "tolerant_first_json_object",
        )

        self.assertEqual(cleaned_json, "")
        self.assertEqual(diagnostics["d4_extraction_reason"], "no_balanced_json_object")


if __name__ == "__main__":
    unittest.main()
