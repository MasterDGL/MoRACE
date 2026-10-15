import unittest

import morace.rule_canonicalizer as rule_canonicalizer
from morace.rule_canonicalizer import detect_cross_source_consensus_groups, merge_rules


class RuleCanonicalizerTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_support_inheritance_is_preserved_for_preaggregated_rule(self):
        rules = [
            {
                "source": "data",
                "primary_source": "data",
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "threshold": None,
                "rationale": "data",
                "support_count": 7,
                "source_votes": {"data": 7},
                "extraction_votes": {"projected": 7},
                "projected_support_count": 7,
                "heuristic_support_count": 0,
                "support_sources": ["data"],
                "support_models": ["qwen"],
                "provenance_list": [{"draft_model": "qwen", "draft_id": i, "batch_id": i} for i in range(7)],
                "provenance": {"draft_model": "qwen", "draft_id": 0, "batch_id": 0},
            }
        ]
        merged = merge_rules(rules)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["support_count"], 7)
        self.assertEqual(merged[0]["source_votes"], {"data": 7})
        self.assertEqual(merged[0]["projected_support_count"], 7)

    def test_merge_preserves_multi_source_metadata(self):
        rules = [
            {
                "source": "literature",
                "primary_source": "literature",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "numeric_value",
                "relation_to_target": "positive",
                "threshold": None,
                "rationale": "lit",
                "support_count": 1,
                "source_votes": {"literature": 1},
                "extraction_votes": {"projected": 1},
                "projected_support_count": 1,
                "heuristic_support_count": 0,
                "support_sources": ["literature"],
                "support_models": ["a"],
                "provenance_list": [{"draft_model": "a", "draft_id": 0, "batch_id": None}],
                "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": None},
            },
            {
                "source": "data",
                "primary_source": "data",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "numeric_value",
                "relation_to_target": "positive",
                "threshold": None,
                "rationale": "data",
                "support_count": 5,
                "source_votes": {"data": 5},
                "extraction_votes": {"projected": 4, "heuristic": 1},
                "projected_support_count": 4,
                "heuristic_support_count": 1,
                "support_sources": ["data"],
                "support_models": ["b"],
                "provenance_list": [{"draft_model": "b", "draft_id": i, "batch_id": i} for i in range(5)],
                "provenance": {"draft_model": "b", "draft_id": 0, "batch_id": 1},
            },
        ]
        merged = merge_rules(rules)
        self.assertEqual(len(merged), 1)
        self.assertEqual(set(merged[0]["support_sources"]), {"literature", "data"})
        self.assertEqual(merged[0]["source"], "multi")
        self.assertEqual(merged[0]["support_count"], 6)
        self.assertEqual(merged[0]["source_votes"], {"literature": 1, "data": 5})
        self.assertEqual(merged[0]["extraction_votes"], {"projected": 5, "heuristic": 1})
        self.assertEqual(merged[0]["projected_support_count"], 5)
        self.assertEqual(merged[0]["heuristic_support_count"], 1)

    def test_nullish_feature_mode_values_share_same_canonical_key(self):
        rules = [
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": None,
                "relation_to_target": "negative",
                "threshold": None,
                "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
            },
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "None",
                "relation_to_target": "negative",
                "threshold": None,
                "provenance": {"draft_model": "a", "draft_id": 1, "batch_id": 1},
            },
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "null",
                "relation_to_target": "negative",
                "threshold": None,
                "provenance": {"draft_model": "a", "draft_id": 2, "batch_id": 2},
            },
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "MolWt",
                "feature_mode": "",
                "relation_to_target": "negative",
                "threshold": None,
                "provenance": {"draft_model": "a", "draft_id": 3, "batch_id": 3},
            },
        ]
        merged = merge_rules(rules)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["canonical_key"], "descriptor|MolWt|null|negative|null")
        self.assertEqual(merged[0]["support_count"], 4)

    def test_provenance_list_is_extended_instead_of_reset(self):
        rules = [
            {
                "source": "data",
                "primary_source": "data",
                "feature_family": "count",
                "feature_name": "NumHBA",
                "feature_mode": "count_value",
                "relation_to_target": "negative",
                "threshold": None,
                "support_count": 3,
                "source_votes": {"data": 3},
                "support_sources": ["data"],
                "support_models": ["qwen"],
                "provenance_list": [
                    {"draft_model": "qwen", "draft_id": 0, "batch_id": 0},
                    {"draft_model": "qwen", "draft_id": 1, "batch_id": 1},
                    {"draft_model": "qwen", "draft_id": 2, "batch_id": 2},
                ],
                "provenance": {"draft_model": "qwen", "draft_id": 0, "batch_id": 0},
            },
            {
                "source": "data",
                "primary_source": "data",
                "feature_family": "count",
                "feature_name": "NumHBA",
                "feature_mode": "count_value",
                "relation_to_target": "negative",
                "threshold": None,
                "support_count": 2,
                "source_votes": {"data": 2},
                "support_sources": ["data"],
                "support_models": ["qwen"],
                "provenance_list": [
                    {"draft_model": "qwen", "draft_id": 3, "batch_id": 3},
                    {"draft_model": "qwen", "draft_id": 4, "batch_id": 4},
                ],
                "provenance": {"draft_model": "qwen", "draft_id": 3, "batch_id": 3},
            },
        ]
        merged = merge_rules(rules)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["support_count"], 5)
        self.assertEqual(len(merged[0]["provenance_list"]), 5)

    def test_rationale_none_does_not_break_merge_and_prefers_longer_text(self):
        rules = [
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "threshold": None,
                "rationale": None,
                "provenance": {"draft_model": "qwen", "draft_id": 0, "batch_id": 0},
            },
            {
                "source": "data",
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "threshold": None,
                "rationale": "lower TPSA is associated with better permeability",
                "provenance": {"draft_model": "qwen", "draft_id": 1, "batch_id": 1},
            },
        ]
        merged = merge_rules(rules)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["rationale"], "lower TPSA is associated with better permeability")

    def test_detect_cross_source_consensus_groups_reports_shared_root_buckets(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "compile_valid": True,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "compile_valid": True,
                    "support_sources": ["data"],
                    "source_votes": {"data": 3},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolWt",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "compile_valid": True,
                    "support_sources": ["data"],
                    "source_votes": {"data": 2},
                    "provenance": {"draft_model": "b", "draft_id": 2, "batch_id": 2},
                },
            ]
        )
        groups = detect_cross_source_consensus_groups(merged)
        tpsa_group = next(item for item in groups if item["root_feature_key"] == "descriptor::TPSA")
        self.assertTrue(tpsa_group["has_cross_source_overlap"])
        self.assertGreaterEqual(tpsa_group["compile_valid_true_count"], 1)
        self.assertIn("negative", tpsa_group["shared_direction_buckets"])
        self.assertIn("continuous", tpsa_group["shared_mode_buckets"])

    def test_merge_rules_promotes_compatible_cross_source_consensus_to_multi(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "support_count": 1,
                    "source_votes": {"literature": 1},
                    "support_sources": ["literature"],
                    "compile_valid": True,
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "threshold_binary",
                    "relation_to_target": "negative",
                    "threshold": {"op": "<=", "value": 90.0},
                    "support_count": 3,
                    "source_votes": {"data": 3},
                    "support_sources": ["data"],
                    "compile_valid": True,
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        , multi_consensus_promotion_enable=True)
        promoted = [item for item in merged if item.get("consensus_promoted")]
        self.assertEqual(len(promoted), 1)
        self.assertEqual(promoted[0]["source"], "multi")
        self.assertEqual(promoted[0]["promotion_type"], "cross_source_consensus")
        self.assertEqual(promoted[0]["promoted_root_feature_key"], "descriptor::TPSA")
        self.assertEqual(len(promoted[0]["promoted_from_canonical_keys"]), 2)
        self.assertEqual(set(promoted[0]["support_sources"]), {"literature", "data"})

    def test_merge_rules_does_not_promote_conflicting_cross_source_rules(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_count": 1,
                    "source_votes": {"literature": 1},
                    "support_sources": ["literature"],
                    "compile_valid": True,
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "support_count": 3,
                    "source_votes": {"data": 3},
                    "support_sources": ["data"],
                    "compile_valid": True,
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        , multi_consensus_promotion_enable=True)
        self.assertFalse(any(item.get("consensus_promoted") for item in merged))

    def test_merge_rules_preserves_dual_source_evidence_for_mollogp_alias_group(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": None,
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["data"],
                    "source_votes": {"data": 3},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        )
        mollogp_rules = [item for item in merged if item.get("feature_name") == "MolLogP"]
        self.assertEqual(len(mollogp_rules), 2)
        self.assertTrue(
            any(set(item.get("support_sources", [])) == {"literature", "data"} and item.get("source") == "multi" for item in mollogp_rules)
        )

    def test_merge_rules_mollogp_alias_group_does_not_end_literature_only_when_data_side_exists(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["data"],
                    "source_votes": {"data": 2},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        )
        exact = next(item for item in merged if item.get("canonical_key") == "descriptor|MolLogP|continuous|positive|null")
        self.assertEqual(set(exact.get("support_sources", [])), {"literature", "data"})
        self.assertEqual(exact.get("source"), "multi")

    def test_merge_rules_non_mollogp_root_is_not_affected_by_dual_source_retention_pass(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": "continuous",
                    "relation_to_target": "negative",
                    "threshold": None,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "feature_mode": None,
                    "relation_to_target": "negative",
                    "threshold": None,
                    "support_sources": ["data"],
                    "source_votes": {"data": 3},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        )
        tpsa_rules = [item for item in merged if item.get("feature_name") == "TPSA"]
        self.assertEqual(len(tpsa_rules), 2)
        self.assertFalse(any(item.get("source") == "multi" for item in tpsa_rules))

    def test_merge_rules_existing_mollogp_multi_object_is_not_regressed(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": None,
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["data"],
                    "source_votes": {"data": 3},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
                {
                    "source": "multi",
                    "primary_source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": None,
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["data", "literature"],
                    "source_votes": {"data": 3, "literature": 1},
                    "provenance": {"draft_model": "c", "draft_id": 2, "batch_id": 2},
                },
            ]
        )
        mollogp_rules = [item for item in merged if item.get("feature_name") == "MolLogP"]
        self.assertTrue(any(item.get("source") == "multi" for item in mollogp_rules))
        self.assertTrue(any(set(item.get("support_sources", [])) == {"literature", "data"} for item in mollogp_rules))

    def test_merge_rules_mollogp_multi_source_survives_final_source_recalculation_with_single_sided_votes(self):
        merged = merge_rules(
            [
                {
                    "source": "literature",
                    "primary_source": "literature",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": "continuous",
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["literature"],
                    "source_votes": {"literature": 1},
                    "provenance": {"draft_model": "a", "draft_id": 0, "batch_id": 0},
                },
                {
                    "source": "data",
                    "primary_source": "data",
                    "feature_family": "descriptor",
                    "feature_name": "MolLogP",
                    "feature_mode": None,
                    "relation_to_target": "positive",
                    "threshold": None,
                    "support_sources": ["data"],
                    "source_votes": {"data": 1},
                    "provenance": {"draft_model": "b", "draft_id": 1, "batch_id": 1},
                },
            ]
        )
        representative = next(
            item
            for item in merged
            if item.get("feature_name") == "MolLogP"
            and item.get("canonical_key") == "descriptor|MolLogP|continuous|positive|null"
        )
        self.assertEqual(set(representative.get("support_sources", [])), {"literature", "data"})
        self.assertEqual(representative.get("source"), "multi")
        self.assertEqual(representative.get("primary_source"), "literature")


if __name__ == "__main__":
    unittest.main()
