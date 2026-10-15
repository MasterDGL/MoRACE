import types
import unittest


from dataclasses import replace
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline


def _rule(source: str, family: str, name: str, canonical_key: str) -> dict:
    return {
        "source": source,
        "primary_source": source,
        "feature_family": family,
        "feature_name": name,
        "feature_mode": "continuous",
        "relation_to_target": "negative",
        "canonical_key": canonical_key,
        "support_count": 3,
        "compile_valid": True,
        "p_value": 0.01,
        "effect_size": 1.0,
        "support_sources": [source],
    }


class _Compiled:
    def __init__(self, name: str, rule: dict):
        self.name = name
        self.rule = rule


class P18aObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = EvaluationPipeline()

    def test_winner_metric_extract_prefers_named_winner_candidate_summary(self):
        branch_metrics = {
            "combined": {
                "candidate_set_winner": "M3_literature_anchor_data_consensus",
                "candidate_sets": [
                    {"name": "M1_top_global", "second_variant_quality_score": 0.0, "template_type": "baseline_global"},
                    {
                        "name": "M3_literature_anchor_data_consensus",
                        "second_variant_quality_score": 1.75,
                        "template_type": "literature_anchor_data_consensus",
                        "selected_rule_count": 5,
                    },
                ],
            }
        }
        winner = self.pipeline.winner_metric_extract(branch_metrics)
        self.assertEqual(winner["template_type"], "literature_anchor_data_consensus")
        self.assertEqual(winner["second_variant_quality_score"], 1.75)
        self.assertEqual(winner["selected_rule_count"], 5)

    def test_build_shrinkage_audit_contains_stage_metrics(self):
        selected = [
            _rule("literature", "descriptor", "TPSA", "descriptor|TPSA|null|negative|null"),
            _rule("data", "count", "NumHBA", "count|NumHBA|count_value|negative|null"),
        ]
        prepared = list(selected)
        compiled = [_Compiled("feat_tpsa", selected[0]), _Compiled("feat_numhba", selected[1])]
        branch_metrics = {
            "combined": {
                "candidate_set_winner": "M3_literature_anchor_data_consensus",
                "candidate_sets": [
                    {
                        "name": "M3_literature_anchor_data_consensus",
                        "template_type": "literature_anchor_data_consensus",
                        "selected_rule_count": 2,
                        "second_variant_quality_score": 0.0,
                    }
                ],
                "selection_trace": {
                    "raw_rules": list(selected),
                    "canonical_rules": list(selected),
                    "gated_rules": list(selected),
                    "selected_rules": list(selected),
                    "selection_audit": {"dropped": []},
                },
            }
        }
        feature_metrics = {
            "feature_count_after_filter": 2,
            "root_feature_count": 2,
            "family_breakdown": {"descriptor": 1, "count": 1, "pattern_presence": 0},
            "family_diversity": 2,
            "source_breakdown": {"literature": 1, "data": 1, "multi": 0},
            "effective_literature_coverage": 1,
            "effective_data_coverage": 1,
            "kept_feature_names": ["feat_tpsa", "feat_numhba"],
        }
        payload = self.pipeline.build_shrinkage_audit(
            branch_metrics=branch_metrics,
            selected_rules=selected,
            prepare_audit={"threshold_learned_rules": list(prepared), "dropped": []},
            prepared_rules=prepared,
            compiled_rules=compiled,
            invalid_rules=[],
            feature_metrics=feature_metrics,
        )
        self.assertIn("stage_metrics", payload)
        self.assertEqual(payload["stage_metrics"]["selected"]["rule_count"], 2)
        self.assertEqual(payload["stage_metrics"]["prepared"]["root_feature_count"], 2)
        self.assertEqual(payload["stage_metrics"]["compiled"]["rule_count"], 2)
        self.assertEqual(payload["winner_metrics"]["template_type"], "literature_anchor_data_consensus")

    def test_literature_underdiverse_uses_profile_fields(self):
        projection_metrics = {
            "canonical_rule_count": 2,
            "compile_safe_count": 2,
            "root_feature_count": 2,
            "family_diversity": 1,
            "descriptor_only": True,
            "dominant_root_share": 0.5,
        }
        self.assertTrue(self.pipeline.literature_underdiverse(projection_metrics))

    def test_promoted_multi_rule_metrics_tracks_surviving_roots(self):
        rules = [
            {
                **_rule("multi", "descriptor", "TPSA", "descriptor|TPSA|continuous|negative|null"),
                "consensus_promoted": True,
                "promoted_root_feature_key": "descriptor::TPSA",
                "support_sources": ["literature", "data"],
            },
            _rule("data", "count", "NumHBA", "count|NumHBA|count_value|negative|null"),
        ]
        metrics = self.pipeline._promoted_multi_rule_metrics(rules)
        self.assertEqual(metrics["promoted_multi_surviving_rule_count"], 1)
        self.assertEqual(metrics["promoted_multi_surviving_root_count"], 1)
        self.assertEqual(metrics["surviving_multi_root_count"], 1)
        self.assertIn("descriptor::TPSA", metrics["promoted_multi_surviving_root_keys"])


if __name__ == "__main__":
    unittest.main()
