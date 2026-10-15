import types
import unittest


from dataclasses import replace
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline


def _rule(
    feature_name: str,
    source: str,
    canonical_key: str,
    *,
    family: str = "descriptor",
    support_sources=None,
    consensus_promoted: bool = False,
):
    return {
        "feature_family": family,
        "feature_name": feature_name,
        "feature_mode": "None" if family == "descriptor" else "count_value",
        "relation_to_target": "negative",
        "threshold": None,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources or [source],
        "canonical_key": canonical_key,
        "consensus_promoted": consensus_promoted,
    }


class LitOnlySlotControlTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = EvaluationPipeline(EvaluationConfig(lit_only_slot_control_enable=True))
        self._orig_eval = self.pipeline.evaluate_rule_set_end_to_end

    def tearDown(self):
        self.pipeline.evaluate_rule_set_end_to_end = self._orig_eval

    def test_skips_when_literature_only_rule_count_not_one(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolLogP", "multi", "descriptor|MolLogP|null|positive|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
            _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count"),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count"),
            _rule("AromaticRings", "data", "count|NumAromaticRings|count_value|positive|null", family="count"),
            _rule("HeavyAtomCount", "data", "count|HeavyAtomCount|count_value|positive|null", family="count"),
        ]
        compiled_rules = [
            types.SimpleNamespace(name=f"f{i}", rule=rule) for i, rule in enumerate(selected_rules)
        ]
        feature_metrics = {
            "kept_feature_names": [f"f{i}" for i in range(7)],
            "root_feature_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "promoted_multi_surviving_root_count": 2,
        }
        result = self.pipeline.maybe_apply_lit_only_slot_control(
            "bbbp", "", None, {}, selected_rules, selected_rules, compiled_rules, [], feature_metrics,
            [0.93], [0.71], [], {}, set(), [], [], [], [], [], []
        )
        report = result[-1]
        self.assertFalse(report["trigger_qualified"])
        self.assertEqual(report["reason"], "literature_only_rule_count_not_1")

    def test_accepts_when_drop_keeps_valid_and_coverage(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolLogP", "multi", "descriptor|MolLogP|null|positive|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
            _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count"),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count"),
            _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
            _rule("AromaticRings", "data", "count|NumAromaticRings|count_value|positive|null", family="count"),
        ]
        compiled_rules = [types.SimpleNamespace(name=f"f{i}", rule=rule) for i, rule in enumerate(selected_rules)]
        feature_metrics = {
            "kept_feature_names": [f"f{i}" for i in range(7)],
            "root_feature_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "promoted_multi_surviving_root_count": 2,
        }

        def fake_eval(*args, **kwargs):
            return {
                "status": "ok",
                "selected_rules": args[3],
                "prepared_rules": args[3],
                "compiled_rules": [types.SimpleNamespace(name=f"c{i}", rule=rule) for i, rule in enumerate(args[3])],
                "invalid_rules": [],
                "feature_metrics": {
                    "root_feature_count": 6,
                    "effective_literature_coverage": 2,
                    "effective_data_coverage": 5,
                    "kept_feature_names": [f"c{i}" for i in range(6)],
                },
                "valid_scores": [0.9330],
                "test_scores": [0.7200],
                "ablation_summary": [],
                "shrinkage_audit": {},
                "rendered_code": "",
                "function_names": [],
                "surviving_rules_by_key": {
                    "descriptor|TPSA|null|negative|null",
                    "descriptor|MolLogP|null|positive|null",
                    "descriptor|MolWt|null|negative|null",
                    "count|NumHBA|count_value|negative|null",
                    "count|NumRotatableBonds|count_value|negative|null",
                    "count|NumAromaticRings|count_value|positive|null",
                },
            }

        self.pipeline.evaluate_rule_set_end_to_end = fake_eval
        result = self.pipeline.maybe_apply_lit_only_slot_control(
            "bbbp", "", None, {}, selected_rules, selected_rules, compiled_rules, [], feature_metrics,
            [0.9335], [0.7137], [], {}, set(), [], [], [], [], [], []
        )
        report = result[-1]
        self.assertTrue(report["accepted"])
        self.assertEqual(report["status"], "accepted")
        self.assertEqual(report["reason"], "accepted")

    def test_rejects_when_drop_breaks_coverage_floor(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolLogP", "multi", "descriptor|MolLogP|null|positive|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
            _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count"),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count"),
            _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
            _rule("AromaticRings", "data", "count|NumAromaticRings|count_value|positive|null", family="count"),
        ]
        compiled_rules = [types.SimpleNamespace(name=f"f{i}", rule=rule) for i, rule in enumerate(selected_rules)]
        feature_metrics = {
            "kept_feature_names": [f"f{i}" for i in range(7)],
            "root_feature_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "promoted_multi_surviving_root_count": 2,
        }

        def fake_eval(*args, **kwargs):
            return {
                "status": "ok",
                "selected_rules": args[3],
                "prepared_rules": args[3],
                "compiled_rules": [types.SimpleNamespace(name=f"c{i}", rule=rule) for i, rule in enumerate(args[3])],
                "invalid_rules": [],
                "feature_metrics": {
                    "root_feature_count": 6,
                    "effective_literature_coverage": 1,
                    "effective_data_coverage": 5,
                    "kept_feature_names": [f"c{i}" for i in range(6)],
                },
                "valid_scores": [0.9340],
                "test_scores": [0.7200],
                "ablation_summary": [],
                "shrinkage_audit": {},
                "rendered_code": "",
                "function_names": [],
                "surviving_rules_by_key": {
                    "descriptor|TPSA|null|negative|null",
                    "descriptor|MolLogP|null|positive|null",
                    "descriptor|MolWt|null|negative|null",
                    "count|NumHBA|count_value|negative|null",
                    "count|NumRotatableBonds|count_value|negative|null",
                    "count|NumAromaticRings|count_value|positive|null",
                },
            }

        self.pipeline.evaluate_rule_set_end_to_end = fake_eval
        result = self.pipeline.maybe_apply_lit_only_slot_control(
            "bbbp", "", None, {}, selected_rules, selected_rules, compiled_rules, [], feature_metrics,
            [0.9335], [0.7137], [], {}, set(), [], [], [], [], [], []
        )
        report = result[-1]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["reason"], "drop_breaks_coverage_floor")

    def test_rejects_when_candidate_not_removed_from_surviving(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolLogP", "multi", "descriptor|MolLogP|null|positive|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
            _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count"),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count"),
            _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
            _rule("AromaticRings", "data", "count|NumAromaticRings|count_value|positive|null", family="count"),
        ]
        compiled_rules = [types.SimpleNamespace(name=f"f{i}", rule=rule) for i, rule in enumerate(selected_rules)]
        feature_metrics = {
            "kept_feature_names": [f"f{i}" for i in range(7)],
            "root_feature_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "promoted_multi_surviving_root_count": 2,
        }

        def fake_eval(*args, **kwargs):
            return {
                "status": "ok",
                "selected_rules": args[3],
                "prepared_rules": args[3],
                "compiled_rules": [types.SimpleNamespace(name=f"c{i}", rule=rule) for i, rule in enumerate(args[3])],
                "invalid_rules": [],
                "feature_metrics": {
                    "root_feature_count": 6,
                    "effective_literature_coverage": 2,
                    "effective_data_coverage": 5,
                    "kept_feature_names": [f"c{i}" for i in range(6)],
                },
                "valid_scores": [0.9340],
                "test_scores": [0.7200],
                "ablation_summary": [],
                "shrinkage_audit": {},
                "rendered_code": "",
                "function_names": [],
                "surviving_rules_by_key": {
                    "descriptor|TPSA|null|negative|null",
                    "descriptor|MolLogP|null|positive|null",
                    "descriptor|MolWt|null|negative|null",
                    "count|NumHBA|count_value|negative|null",
                    "count|NumRotatableBonds|count_value|negative|null",
                    "count|NumAromaticRings|count_value|positive|null",
                    "count|NumHBD|count_value|negative|null",
                },
            }

        self.pipeline.evaluate_rule_set_end_to_end = fake_eval
        result = self.pipeline.maybe_apply_lit_only_slot_control(
            "bbbp", "", None, {}, selected_rules, selected_rules, compiled_rules, [], feature_metrics,
            [0.9335], [0.7137], [], {}, set(), [], [], [], [], [], []
        )
        report = result[-1]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["reason"], "candidate_not_removed_from_surviving")

    def test_rejects_when_valid_drop_too_large(self):
        selected_rules = [
            _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolLogP", "multi", "descriptor|MolLogP|null|positive|null", support_sources=["data", "literature"], consensus_promoted=True),
            _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
            _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count"),
            _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count"),
            _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
            _rule("AromaticRings", "data", "count|NumAromaticRings|count_value|positive|null", family="count"),
        ]
        compiled_rules = [types.SimpleNamespace(name=f"f{i}", rule=rule) for i, rule in enumerate(selected_rules)]
        feature_metrics = {
            "kept_feature_names": [f"f{i}" for i in range(7)],
            "root_feature_count": 7,
            "effective_literature_coverage": 3,
            "effective_data_coverage": 6,
            "promoted_multi_surviving_root_count": 2,
        }

        def fake_eval(*args, **kwargs):
            return {
                "status": "ok",
                "selected_rules": args[3],
                "prepared_rules": args[3],
                "compiled_rules": [types.SimpleNamespace(name=f"c{i}", rule=rule) for i, rule in enumerate(args[3])],
                "invalid_rules": [],
                "feature_metrics": {
                    "root_feature_count": 6,
                    "effective_literature_coverage": 2,
                    "effective_data_coverage": 5,
                    "kept_feature_names": [f"c{i}" for i in range(6)],
                },
                "valid_scores": [0.9310],
                "test_scores": [0.7200],
                "ablation_summary": [],
                "shrinkage_audit": {},
                "rendered_code": "",
                "function_names": [],
                "surviving_rules_by_key": {
                    "descriptor|TPSA|null|negative|null",
                    "descriptor|MolLogP|null|positive|null",
                    "descriptor|MolWt|null|negative|null",
                    "count|NumHBA|count_value|negative|null",
                    "count|NumRotatableBonds|count_value|negative|null",
                    "count|NumAromaticRings|count_value|positive|null",
                },
            }

        self.pipeline.evaluate_rule_set_end_to_end = fake_eval
        result = self.pipeline.maybe_apply_lit_only_slot_control(
            "bbbp", "", None, {}, selected_rules, selected_rules, compiled_rules, [], feature_metrics,
            [0.9335], [0.7137], [], {}, set(), [], [], [], [], [], []
        )
        report = result[-1]
        self.assertFalse(report["accepted"])
        self.assertEqual(report["status"], "rejected")
        self.assertEqual(report["reason"], "valid_drop_too_large")


if __name__ == "__main__":
    unittest.main()
