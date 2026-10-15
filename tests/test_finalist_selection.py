import types
import unittest


from dataclasses import replace
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline


def _candidate(
    name: str,
    alias_signature,
    effective_lit: int,
    roots: int,
    borda: float,
    mrr: float,
) -> dict:
    return {
        "name": name,
        "template_type": name,
        "alias_group_signature": list(alias_signature),
        "effective_literature_coverage": effective_lit,
        "effective_data_coverage": 6,
        "selected_root_feature_count": roots,
        "compiled_rule_count": roots,
        "selected_rule_count": roots,
        "borda_score": borda,
        "mrr_score": mrr,
        "rules": [],
    }


def _rule(
    feature_name: str,
    source: str,
    canonical_key: str,
    *,
    family: str = "descriptor",
    feature_mode: str = "None",
    relation_to_target: str = "negative",
    support_count: int = 1,
    p_value: float = 0.01,
    effect_size: float = 1.0,
):
    support_sources = ["data", "literature"] if source == "multi" else [source]
    return {
        "feature_family": family,
        "feature_name": feature_name,
        "feature_mode": feature_mode if family == "descriptor" else "count_value",
        "relation_to_target": relation_to_target,
        "threshold": None,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources,
        "canonical_key": canonical_key,
        "support_count": support_count,
        "p_value": p_value,
        "effect_size": effect_size,
    }


class P18dFinalistChooserTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = EvaluationPipeline()

    def test_shortlist_keeps_m1_literature_improved_and_distinct_third(self):
        candidate_sets = [
            _candidate("M1_top_global", ("a", "b", "c"), 1, 6, 10.0, 3.0),
            _candidate("M2_branch_preserving", ("a", "b", "d"), 2, 6, 9.0, 2.0),
            _candidate("M3_literature_anchor_data_consensus", ("a", "b", "c"), 1, 7, 8.0, 1.0),
            _candidate("M4_data_heavy_alias", ("x", "y", "z"), 1, 6, 7.0, 1.0),
        ]
        finalists = self.pipeline.shortlist_all_finalists(candidate_sets)
        self.assertEqual([item["name"] for item in finalists], ["M1_top_global", "M2_branch_preserving", "M4_data_heavy_alias"])
        self.assertEqual(finalists[0]["finalist_shortlist_reason"], "static_baseline")
        self.assertEqual(finalists[1]["finalist_shortlist_reason"], "literature_improved")
        self.assertEqual(finalists[2]["finalist_shortlist_reason"], "static_distinct")

    def test_choose_finalist_prefers_coverage_candidate_within_delta(self):
        finalist_reports = [
            {
                "name": "M1_top_global",
                "valid_mean": 0.9350,
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 6,
            },
            {
                "name": "M2_branch_preserving",
                "valid_mean": 0.9325,
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 2,
                "effective_data_coverage": 5,
            },
        ]
        decision = self.pipeline.choose_finalist_with_coverage_guard(finalist_reports)
        self.assertEqual(decision["winner"]["name"], "M2_branch_preserving")
        self.assertEqual(decision["decision"], "coverage_guard_kept_within_delta")
        self.assertTrue(decision["coverage_guard_applied"])

    def test_choose_finalist_reverts_to_best_valid_when_coverage_gap_too_large(self):
        finalist_reports = [
            {
                "name": "M1_top_global",
                "valid_mean": 0.9350,
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 6,
            },
            {
                "name": "M2_branch_preserving",
                "valid_mean": 0.9200,
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 2,
                "effective_data_coverage": 5,
            },
        ]
        decision = self.pipeline.choose_finalist_with_coverage_guard(finalist_reports)
        self.assertEqual(decision["winner"]["name"], "M1_top_global")
        self.assertEqual(decision["decision"], "best_valid")
        self.assertFalse(decision["coverage_guard_applied"])
        self.assertGreater(decision["valid_gap_to_best"], 0.0)

    def test_choose_finalist_uses_lower_is_better_for_regression(self):
        finalist_reports = [
            {
                "name": "M1_top_global",
                "valid_mean": 0.8200,
                "higher_is_better": False,
                "selection_metric": "rmse",
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 6,
            },
            {
                "name": "M2_branch_preserving",
                "valid_mean": 0.8230,
                "higher_is_better": False,
                "selection_metric": "rmse",
                "compiled_rule_count": 6,
                "root_feature_count": 6,
                "effective_literature_coverage": 2,
                "effective_data_coverage": 5,
            },
        ]
        decision = self.pipeline.choose_finalist_with_coverage_guard(finalist_reports)
        self.assertEqual(decision["winner"]["name"], "M2_branch_preserving")
        self.assertEqual(decision["decision"], "coverage_guard_kept_within_delta")

    def test_update_combined_metrics_prefers_surviving_feature_coverage(self):
        branch_metrics = {
            "combined": {
                "candidate_set_winner": "M1_top_global",
                "selection_trace": {
                    "selected_rules": [],
                    "selection_audit": {"dropped": []},
                },
            }
        }
        chosen_candidate = {
            "name": "M2_branch_preserving",
            "template_type": "branch_preserving",
            "selected_rule_count": 6,
            "selected_root_feature_count": 6,
            "selected_source_breakdown": {"literature": 2, "data": 4, "multi": 0},
            "selected_family_breakdown": {"descriptor": 4, "count": 2, "pattern_presence": 0},
            "effective_literature_coverage": 2,
            "effective_data_coverage": 4,
            "family_diversity": 2,
            "rules": [{"feature_name": "TPSA"}],
            "selection_audit": {"dropped": []},
        }
        chosen_report = {
            "name": "M2_branch_preserving",
            "selection_metric": "auc",
            "higher_is_better": True,
            "root_feature_count": 5,
            "effective_literature_coverage": 1,
            "effective_data_coverage": 5,
            "source_breakdown": {"literature": 1, "data": 4, "multi": 0},
            "family_breakdown": {"descriptor": 3, "count": 2, "pattern_presence": 0},
            "family_diversity": 2,
        }
        decision = {
            "decision": "coverage_guard_kept_within_delta",
            "coverage_guard_applied": True,
            "best_valid_name": "M1_top_global",
            "coverage_candidate_name": "M2_branch_preserving",
            "valid_gap_to_best": 0.003,
        }
        updated = self.pipeline._update_combined_metrics_with_finalist(
            branch_metrics,
            chosen_candidate,
            chosen_report,
            decision,
        )
        combined = updated["combined"]
        self.assertEqual(combined["candidate_set_winner"], "M2_branch_preserving")
        self.assertEqual(combined["effective_literature_coverage"], 1)
        self.assertEqual(combined["effective_data_coverage"], 5)
        self.assertEqual(combined["selected_root_feature_count"], 5)
        self.assertEqual(combined["selected_source_breakdown"]["literature"], 1)
        self.assertEqual(combined["candidate_set_winner_explanation"]["selection_metric"], "auc")

    def test_plus_one_alias_candidate_prefers_same_root_alias_over_new_root(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null"),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
        ]
        candidate_pool = [
            _rule(
                "TPSA",
                "data",
                "descriptor|TPSA|threshold_binary|negative|null",
                feature_mode="threshold_binary",
                support_count=3,
            ),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count", support_count=20),
        ]
        choice = self.pipeline.choose_plus_one_alias_candidate(selected_rules, candidate_pool)
        self.assertEqual(choice["status"], "ok")
        self.assertEqual(choice["slot_type"], "same_root_alias")
        self.assertEqual(choice["candidate_root_feature_key"], "descriptor::TPSA")

    def test_plus_one_alias_acceptance_requires_surviving_candidate_and_valid_drop_bound(self):
        base_report = {
            "valid_mean": 0.9329,
            "compiled_rule_count": 6,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 5,
        }
        extended_report = {
            "valid_mean": 0.9312,
            "compiled_rule_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "plus_one_candidate_survived": True,
        }
        self.assertTrue(self.pipeline.accept_plus_one_alias_extension(base_report, extended_report))
        extended_report["plus_one_candidate_survived"] = False
        self.assertFalse(self.pipeline.accept_plus_one_alias_extension(base_report, extended_report))

    def test_plus_one_alias_acceptance_rejects_when_candidate_does_not_survive(self):
        base_report = {
            "valid_mean": 0.9310,
            "compiled_rule_count": 6,
            "effective_literature_coverage": 2,
            "effective_data_coverage": 5,
        }
        extended_report = {
            "valid_mean": 0.9320,
            "compiled_rule_count": 7,
            "effective_literature_coverage": 2,
            "effective_data_coverage": 6,
            "plus_one_candidate_survived": False,
        }
        self.assertFalse(self.pipeline.accept_plus_one_alias_extension(base_report, extended_report))

    def test_plus_one_alias_candidate_falls_back_to_new_root_when_no_same_root_alias(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null"),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
        ]
        candidate_pool = [
            _rule("TPSA", "data", "descriptor|TPSA|null|negative|null", support_count=30),
            _rule(
                "NumRotatableBonds",
                "data",
                "count|NumRotatableBonds|count_value|negative|null",
                family="count",
                support_count=5,
            ),
        ]
        choice = self.pipeline.choose_plus_one_alias_candidate(selected_rules, candidate_pool)
        self.assertEqual(choice["status"], "ok")
        self.assertEqual(choice["slot_type"], "new_root_fallback")
        self.assertEqual(choice["candidate_root_feature_key"], "count::NumRotatableBonds")

    def test_plus_one_alias_extension_skips_when_base_rule_count_not_6(self):
        self.pipeline = EvaluationPipeline(replace(self.pipeline.config, all_plus_one_alias_enable=True))
        selected_rules = [_rule(f"R{i}", "data", f"descriptor|R{i}|null|negative|null") for i in range(7)]
        result = self.pipeline.maybe_apply_plus_one_alias_extension(
            dataset="bbbp",
            subtask="",
            spec=None,
            branch_metrics={},
            selected_rules=selected_rules,
            prepared_rules=list(selected_rules),
            compiled_rules=[object()] * 7,
            invalid_rules=[],
            feature_metrics={},
            valid_scores=[],
            test_scores=[],
            ablation_summary=[],
            shrinkage_audit_payload={},
            second_variant_rule_ids=set(),
            smiles_train=[],
            y_train=[],
            smiles_valid=[],
            y_valid=[],
            smiles_test=[],
            y_test=[],
        )
        report = result[-1]
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "base_rule_count_not_6")
        self.assertEqual(report["base_selected_rule_count"], 7)
        self.assertEqual(report["base_compiled_rule_count"], 7)


if __name__ == "__main__":
    unittest.main()
