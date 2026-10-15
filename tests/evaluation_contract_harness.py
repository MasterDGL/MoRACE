import copy
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch



PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from morace import evaluation_pipeline as evaluation_module
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline



def healthy_projection_metrics():
    return {
        "strict_projector_valid_rate": 0.8,
        "projector_valid_rate": 0.8,
        "heuristic_rule_rate": 0.1,
        "schema_valid_rate": 1.0,
        "repair_success_rate": 1.0,
        "raw_rule_count": 3,
        "canonical_rule_count": 3,
        "compile_safe_count": 3,
        "root_feature_count": 3,
        "family_diversity": 2,
        "mollogp_collapse_rate": 0.0,
        "dominant_root_share": 0.34,
        "descriptor_only": False,
    }


def branch_metrics(synthesize=None, inference=None):
    return {
        "synthesize_projection": copy.deepcopy(synthesize or healthy_projection_metrics()),
        "inference_projection": copy.deepcopy(inference or healthy_projection_metrics()),
        "combined": {
            "selected_rule_count": 4,
            "compile_valid_rate": 0.75,
        },
    }


class EvaluationGateContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from morace import evaluation_pipeline

        cls.pipeline = EvaluationPipeline()

    def evaluate(self, knowledge_type, synthesize=None, inference=None):
        self.pipeline = EvaluationPipeline(EvaluationConfig(knowledge_type=knowledge_type))
        return self.pipeline.evaluate_quality_gate(branch_metrics(synthesize, inference))

    def test_quality_gate_branch_matrix(self):
        healthy = healthy_projection_metrics()
        underdiverse = {**healthy, "underdiverse": True}
        low_soft = {**healthy, "strict_projector_valid_rate": 0.1}
        low_not_soft = {
            **low_soft,
            "schema_valid_rate": 0.5,
        }
        low_inference = {**healthy, "strict_projector_valid_rate": 0.1}
        heuristic_heavy = {**healthy, "heuristic_rule_rate": 0.6}
        empty_synthesis = {**underdiverse, "raw_rule_count": 0}

        cases = [
            ("synthesize", healthy, healthy, ("success", "", "strict_pass")),
            (
                "synthesize",
                underdiverse,
                healthy,
                ("degraded", "strict_valid_but_underdiverse", "semantic_gate"),
            ),
            (
                "synthesize",
                low_soft,
                healthy,
                ("success", "literature_soft_pass(strict=0.100)", "literature_soft_pass"),
            ),
            (
                "synthesize",
                low_not_soft,
                healthy,
                ("degraded", "low_projector_valid_rate(strict=0.100)", "strict_gate"),
            ),
            ("inference", healthy, healthy, ("success", "", "strict_pass")),
            (
                "inference",
                healthy,
                low_inference,
                ("degraded", "low_projector_valid_rate(strict=0.100)", "strict_gate"),
            ),
            (
                "inference",
                healthy,
                heuristic_heavy,
                ("degraded", "heuristic_domination(rate=0.600)", "strict_gate"),
            ),
            ("all", healthy, healthy, ("success", "", "full_all")),
            (
                "all",
                underdiverse,
                healthy,
                ("success", "inference_only_fallback", "inference_only_fallback"),
            ),
            (
                "all",
                empty_synthesis,
                low_inference,
                ("skipped", "literature_branch_empty", "strict_gate"),
            ),
            (
                "all",
                low_soft,
                healthy,
                ("success", "inference_dominant_all(strict=0.100)", "inference_dominant_all"),
            ),
        ]
        actual = [
            self.evaluate(knowledge_type, synthesize, inference)
            for knowledge_type, synthesize, inference, _ in cases
        ]
        self.assertEqual(actual, [expected for *_, expected in cases])

    def test_branch_failure_diagnostics_contract(self):
        metrics = branch_metrics()
        metrics["synthesize_projection"]["strict_projector_valid_rate"] = 0.1
        metrics["synthesize_projection"]["strict_failure_diagnostics"] = {
            "total_failures": 1,
            "stage_counts": {"parse": 1},
            "reason_counts": {"invalid_json": 1},
            "label_counts": {"parse.invalid_json": 1},
            "by_stage": {"parse": {"invalid_json": 1}},
        }
        metrics["inference_projection"]["strict_failure_diagnostics"] = {
            "total_failures": 2,
            "stage_counts": {"parse": 1, "schema": 1},
            "reason_counts": {"invalid_json": 1, "bad_enum": 1},
            "label_counts": {"parse.invalid_json": 1, "schema.bad_enum": 1},
            "by_stage": {"parse": {"invalid_json": 1}, "schema": {"bad_enum": 1}},
        }
        self.pipeline = EvaluationPipeline(EvaluationConfig(knowledge_type="all"))
        diagnostics = self.pipeline.build_branch_failure_diagnostics(
            metrics,
            "strict_gate",
            "low_projector_valid_rate(strict=0.100)",
        )

        self.assertEqual(
            diagnostics["overall"],
            {
                "total_failures": 3,
                "stage_counts": {"parse": 2, "schema": 1},
                "reason_counts": {"invalid_json": 2, "bad_enum": 1},
                "label_counts": {"parse.invalid_json": 2, "schema.bad_enum": 1},
            },
        )
        self.assertEqual(
            diagnostics["all"],
            {
                "quality_gate_mode": "strict_gate",
                "quality_gate_reason": "low_projector_valid_rate(strict=0.100)",
                "limiting_branch": "synthesize",
                "selected_rule_count": 4,
                "compile_valid_rate": 0.75,
            },
        )
        self.assertEqual(diagnostics["synthesize"]["total_failures"], 1)
        self.assertEqual(diagnostics["inference"]["total_failures"], 2)

    def test_outcome_records_generation_seed(self):
        self.pipeline = EvaluationPipeline(EvaluationConfig(
            knowledge_type="all",
            run_tag="seeded-run",
            generation_seed=42,
        ))
        metadata = {
            "status": "success",
            "quality_gate": {},
            "null_model_scores": {},
        }

        with (
            patch.object(self.pipeline, "build_outcome_payload", return_value={}) as build_outcome,
            patch.object(self.pipeline, "outcome_output_path_for", return_value="outcome.json"),
            patch.object(self.pipeline, "write_json"),
            patch.object(evaluation_module.os.path, "exists", return_value=True),
        ):
            self.pipeline.write_unified_outcome_artifact(
                "bbbp",
                "",
                "rules.json",
                "rules.txt",
                metadata,
            )

        self.assertEqual(build_outcome.call_args.kwargs["generation_seed"], 42)


class RuleDiversityContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from morace import evaluation_pipeline

        cls.pipeline = EvaluationPipeline()

    def test_empty_rule_diversity_contract(self):
        self.assertEqual(
            self.pipeline.compute_rule_diversity_metrics([]),
            {
                "family_breakdown": {},
                "family_diversity": 0,
                "source_breakdown": {},
                "effective_literature_coverage": 0,
                "effective_data_coverage": 0,
                "root_feature_count": 0,
                "dominant_feature_share": 0.0,
            },
        )

    def test_mixed_rule_diversity_and_promoted_multi_contract(self):
        rules = [
            {
                "feature_family": "descriptor",
                "feature_name": "MolLogP",
                "support_sources": ["literature"],
                "consensus_promoted": True,
            },
            {
                "feature_family": "descriptor",
                "feature_name": "MolLogP",
                "support_sources": ["data"],
            },
            {
                "feature_family": "count",
                "feature_name": "MolLogP",
                "support_sources": ["literature", "data"],
                "consensus_promoted": True,
                "promoted_root_feature_key": "shared::MolLogP",
            },
            {
                "feature_family": "pattern_presence",
                "feature_name": "AromaticRing",
                "source": "data",
            },
        ]

        self.assertEqual(
            self.pipeline.compute_rule_diversity_metrics(rules),
            {
                "family_breakdown": {
                    "descriptor": 2,
                    "pattern_presence": 1,
                    "count": 1,
                },
                "family_diversity": 3,
                "source_breakdown": {"literature": 1, "data": 2, "multi": 1},
                "effective_literature_coverage": 2,
                "effective_data_coverage": 3,
                "root_feature_count": 3,
                "dominant_feature_share": 0.75,
            },
        )
        self.assertEqual(
            self.pipeline._promoted_multi_rule_metrics(rules),
            {
                "promoted_multi_surviving_rule_count": 2,
                "promoted_multi_surviving_root_count": 2,
                "promoted_multi_surviving_root_keys": [
                    "descriptor::MolLogP",
                    "shared::MolLogP",
                ],
                "surviving_multi_root_count": 1,
                "surviving_multi_root_keys": ["count::MolLogP"],
            },
        )


class FinalistSelectionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from morace import evaluation_pipeline

        cls.pipeline = EvaluationPipeline()

    def setUp(self):
        self.pipeline = EvaluationPipeline()

    def test_shortlist_uses_root_improvement_and_skips_duplicate_aliases(self):
        candidate_sets = [
            {
                "name": "M1_top_global",
                "alias_group_signature": ["a"],
                "effective_literature_coverage": 2,
                "selected_root_feature_count": 5,
            },
            {
                "name": "M2_root_improved",
                "alias_group_signature": ["b"],
                "effective_literature_coverage": 2,
                "selected_root_feature_count": 6,
                "borda_score": 2.0,
                "mrr_score": 1.0,
            },
            {
                "name": "M3_duplicate_baseline",
                "alias_group_signature": ["a"],
                "effective_literature_coverage": 2,
                "selected_root_feature_count": 5,
            },
            {
                "name": "M4_distinct",
                "alias_group_signature": ["c"],
                "effective_literature_coverage": 1,
                "selected_root_feature_count": 4,
            },
        ]

        finalists = self.pipeline.shortlist_all_finalists(candidate_sets)
        self.assertEqual(
            [(item["name"], item["finalist_shortlist_reason"]) for item in finalists],
            [
                ("M1_top_global", "static_baseline"),
                ("M2_root_improved", "root_improved_fallback"),
                ("M4_distinct", "static_distinct"),
            ],
        )

    def test_choose_finalist_without_valid_scores_uses_stable_fallback(self):
        reports = [
            {"name": "first", "valid_mean": None},
            {"name": "second", "valid_mean": None},
        ]
        self.assertEqual(
            self.pipeline.choose_finalist_with_coverage_guard(reports),
            {
                "winner": reports[0],
                "decision": "no_valid_scores_fallback",
                "best_valid_name": "first",
                "coverage_guard_applied": False,
                "valid_gap_to_best": 0.0,
            },
        )


class RulePreparationContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from morace import evaluation_pipeline

        cls.pipeline = EvaluationPipeline()

    def test_semantic_deduplication_keeps_highest_support_in_stable_order(self):
        first = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "relation_to_target": "positive",
            "threshold": {"op": ">=", "value": 2.00004},
            "support_count": 1,
            "marker": "first",
        }
        replacement = {
            **first,
            "threshold": {"op": ">=", "value": 2.00003},
            "support_count": 4,
            "marker": "replacement",
        }
        distinct = {
            "feature_family": "count",
            "feature_name": "NumHBA",
            "relation_to_target": "negative",
            "threshold": None,
            "support_count": 2,
            "marker": "distinct",
        }

        self.assertEqual(
            self.pipeline.semantic_rule_signature(first),
            ("descriptor", "MolLogP", "positive", ">=", "2.0000"),
        )
        deduplicated = self.pipeline.dedupe_prepared_rules([first, distinct, replacement])
        self.assertEqual([rule["marker"] for rule in deduplicated], ["replacement", "distinct"])

    def test_direction_normalization_and_threshold_correction_contract(self):
        inferred = self.pipeline.normalize_directional_relation(
            {
                "feature_family": "descriptor",
                "feature_name": "MolLogP",
                "relation_to_target": "unknown",
                "rationale": "Higher values increase the target response.",
                "threshold": None,
            }
        )
        self.assertEqual(inferred["relation_to_target"], "monotonic_up")
        self.assertTrue(inferred["direction_consistency_relation_inferred"])

        corrected = self.pipeline.enforce_direction_consistency(
            {
                "feature_family": "descriptor",
                "feature_name": "MolLogP",
                "relation_to_target": "negative",
                "rationale": "Lower values are preferred.",
                "threshold": {"op": ">=", "value": 2.0},
            }
        )
        self.assertEqual(corrected["threshold"], {"op": "<=", "value": 2.0})
        self.assertTrue(corrected["direction_consistency_adjusted"])
        self.assertEqual(corrected["direction_consistency_original_op"], ">=")
        self.assertEqual(corrected["threshold_origin"], "direction_consistency_corrected")

    def test_non_descriptor_threshold_is_not_rewritten(self):
        original = {
            "feature_family": "count",
            "feature_name": "NumHBA",
            "relation_to_target": "negative",
            "threshold": {"op": ">=", "value": 2.0},
        }
        updated = self.pipeline.enforce_direction_consistency(original)
        self.assertEqual(updated["threshold"], original["threshold"])
        self.assertNotIn("direction_consistency_adjusted", updated)
        self.assertIsNot(updated, original)

    def test_threshold_learning_keeps_validation_gain(self):
        rule = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "threshold": None,
            "rationale": "Higher lipophilicity increases the target.",
        }
        train_smiles = [f"train-{index}" for index in range(10)]
        train_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        def validation_score(candidate, *_args, **_kwargs):
            return 0.8 if candidate.get("threshold") else 0.6

        with (
            patch.object(self.pipeline, "compile_rule", return_value=object()),
            patch.object(
                self.pipeline,
                "build_feature_matrix",
                return_value=[[float(index)] for index in range(10)],
            ),
            patch.object(self.pipeline, "_single_feature_validation_score", side_effect=validation_score),
        ):
            prepared = self.pipeline.learn_threshold_if_needed(
                rule,
                train_smiles,
                train_y,
                ["valid-a", "valid-b"],
                [0, 1],
                "classification",
            )

        self.assertEqual(prepared["threshold"], {"op": ">=", "value": 4.0, "unit": None})
        self.assertEqual(prepared["feature_mode"], "threshold_binary")
        self.assertEqual(prepared["threshold_origin"], "train_learned_snapped")
        self.assertEqual(prepared["threshold_kept_reason"], "validation_gain")
        self.assertAlmostEqual(prepared["threshold_score_delta"], 0.2)

    def test_threshold_learning_prefers_continuous_without_gain(self):
        rule = {
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "threshold": None,
            "rationale": "Lipophilicity may matter.",
        }
        train_smiles = [f"train-{index}" for index in range(10)]
        train_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

        def validation_score(candidate, *_args, **_kwargs):
            return 0.7 if candidate.get("threshold") else 0.8

        with (
            patch.object(self.pipeline, "compile_rule", return_value=object()),
            patch.object(
                self.pipeline,
                "build_feature_matrix",
                return_value=[[float(index)] for index in range(10)],
            ),
            patch.object(self.pipeline, "_single_feature_validation_score", side_effect=validation_score),
        ):
            prepared = self.pipeline.learn_threshold_if_needed(
                rule,
                train_smiles,
                train_y,
                ["valid-a", "valid-b"],
                [0, 1],
                "classification",
            )

        self.assertIsNone(prepared["threshold"])
        self.assertEqual(prepared["feature_mode"], "numeric_value")
        self.assertEqual(prepared["threshold_kept_reason"], "prefer_continuous_numeric")
        self.assertAlmostEqual(prepared["threshold_score_delta"], -0.1)

    def test_preparation_audit_preserves_failures_duplicates_and_variant_retention(self):
        low_support = {
            "canonical_key": "rule-low",
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "threshold": None,
            "support_count": 1,
            "compile_valid": True,
        }
        failing = {
            "canonical_key": "rule-fail",
            "feature_family": "descriptor",
            "feature_name": "TPSA",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
            "threshold": None,
            "support_count": 2,
            "compile_valid": True,
        }
        high_support = {
            **low_support,
            "canonical_key": "rule-high",
            "support_count": 4,
        }

        def fake_load_split(_dataset, *, subtask, split):
            if split == "train":
                return ["train"], [0], SimpleNamespace(task_type="classification")
            return ["valid"], [1], None

        def fake_learn(rule, *_args, **_kwargs):
            if rule["canonical_key"] == "rule-fail":
                raise RuntimeError("contract failure")
            return copy.deepcopy(rule)

        with (
            patch.object(self.pipeline, "load_split_xy", side_effect=fake_load_split),
            patch.object(self.pipeline, "learn_threshold_if_needed", side_effect=fake_learn),
        ):
            prepared, audit = self.pipeline.prepare_rules_for_compilation_with_audit(
                [low_support, failing, high_support],
                "bbbp",
                "",
                second_variant_rule_ids={"rule-low", "rule-high"},
            )

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["canonical_key"], "rule-high")
        self.assertEqual(prepared[0]["audit_origin_rule_id"], "rule-high")
        self.assertTrue(prepared[0]["_audit_is_second_variant_participant"])
        self.assertEqual(len(audit["threshold_learned_rules"]), 2)
        self.assertEqual(
            [(item["reason"], item["rule"]["rule_id"]) for item in audit["dropped"]],
            [
                ("other_preparation_failure", "rule-fail"),
                ("preparation_duplicate_feature", "rule-low"),
            ],
        )
        self.assertEqual(audit["dropped"][0]["error"], "RuntimeError: contract failure")
        self.assertEqual(audit["dropped"][1]["replaced_by_rule_id"], "rule-high")

        retention = self.pipeline._prepared_retention_metrics(
            [low_support, failing, high_support],
            prepared,
            {"rule-low", "rule-high"},
        )
        self.assertAlmostEqual(retention["prepared_retention_score"], 1.0 / 3.0)
        self.assertEqual(retention["selected_second_variant_rule_count"], 2)
        self.assertEqual(retention["prepared_second_variant_rule_count"], 1)
        self.assertEqual(retention["prepared_second_variant_retention_score"], 0.5)

    def test_main_reraises_unexpected_preparation_failure(self):
        self.pipeline = EvaluationPipeline(EvaluationConfig(
            dataset="bbbp",
            subtask="",
            knowledge_type="all",
        ))
        with (
            patch.object(self.pipeline, "load_rules", side_effect=ValueError("broken rule artifact")),
            patch.object(self.pipeline, "save_skip_outputs") as save_failed,
        ):
            with self.assertRaisesRegex(ValueError, "broken rule artifact"):
                self.pipeline.main()

        self.assertEqual(save_failed.call_args.kwargs["status"], "failed")
        self.assertEqual(save_failed.call_args.kwargs["reason"], "ValueError: broken rule artifact")

    def test_main_reraises_unexpected_evaluation_failure(self):
        self.pipeline = EvaluationPipeline(EvaluationConfig(
            dataset="bbbp",
            subtask="",
            knowledge_type="all",
        ))
        rule = {
            "source": "literature",
            "feature_family": "descriptor",
            "feature_name": "MolWt",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "threshold": None,
        }
        compiled = SimpleNamespace(name="mol_wt", rule=rule)
        spec = SimpleNamespace(task_type="classification", dataset="bbbp")
        feature_metrics = {
            "feature_count_before_filter": 1,
            "feature_count_after_filter": 1,
            "non_nan_coverage": 1.0,
        }
        split_data = (["C", "CC"], [0, 1], spec)
        with (
            patch.object(self.pipeline, "load_rules", return_value=([rule], branch_metrics())),
            patch.object(self.pipeline, "evaluate_quality_gate", return_value=("success", "", "full_all")),
            patch.object(self.pipeline, "_collect_second_variant_rule_ids", return_value=set()),
            patch.object(self.pipeline, "build_branch_failure_diagnostics", return_value={}),
            patch.object(
                self.pipeline,
                "prepare_rules_for_compilation_with_audit",
                return_value=([rule], {"threshold_learned_rules": [], "dropped": []}),
            ),
            patch.object(self.pipeline, "compile_rules", return_value=([compiled], [])),
            patch.object(self.pipeline, "load_split_xy", return_value=split_data),
            patch.object(
                self.pipeline,
                "compute_null_model_scores",
                return_value={"valid_mean": 0.5, "test_mean": 0.5},
            ),
            patch.object(self.pipeline, "build_feature_matrix", return_value=[[1.0], [2.0]]),
            patch.object(
                self.pipeline,
                "sanitize_feature_matrices",
                return_value=([[1.0], [2.0]], [[1.0], [2.0]], [[1.0], [2.0]], ["mol_wt"], feature_metrics),
            ),
            patch.object(self.pipeline, "compute_rule_diversity_metrics", return_value={}),
            patch.object(self.pipeline, "_prepared_retention_metrics", return_value={}),
            patch.object(self.pipeline, "_promoted_multi_rule_metrics", return_value={}),
            patch.object(self.pipeline, "winner_metric_extract", return_value={}),
            patch.object(self.pipeline, "build_shrinkage_audit", return_value={}),
            patch.object(self.pipeline, "load_runtime_snapshot", side_effect=OSError("runtime snapshot failed")),
            patch.object(self.pipeline, "save_skip_outputs") as save_failed,
        ):
            with self.assertRaisesRegex(OSError, "runtime snapshot failed"):
                self.pipeline.main()

        self.assertEqual(save_failed.call_args.kwargs["status"], "failed")
        self.assertEqual(save_failed.call_args.kwargs["reason"], "OSError: runtime snapshot failed")

    def test_feature_matrix_sanitization_contract(self):
        nan = float("nan")
        X_train, X_valid, X_test, kept, metrics = self.pipeline.sanitize_feature_matrices(
            [
                [1.0, nan, 5.0],
                [2.0, nan, 5.0],
                [3.0, nan, 5.0],
            ],
            [[nan, 10.0, 5.0]],
            [[4.0, nan, 5.0]],
            ["varying", "all_nan", "constant"],
        )

        self.assertEqual(kept, ["varying"])
        self.assertEqual(X_train.tolist(), [[1.0], [2.0], [3.0]])
        self.assertEqual(X_valid.tolist(), [[2.0]])
        self.assertEqual(X_test.tolist(), [[4.0]])
        self.assertEqual(metrics["feature_count_before_filter"], 3)
        self.assertEqual(metrics["feature_count_after_filter"], 1)
        self.assertAlmostEqual(metrics["non_nan_coverage"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["feature_entropy"], 1.584962500721156)
        self.assertEqual(metrics["pairwise_correlation"], 0.0)
        self.assertEqual(
            metrics["dropped_features"],
            [
                {"feature_name": "all_nan", "reason": "all_nan"},
                {"feature_name": "constant", "reason": "constant_or_single_value"},
            ],
        )

    def test_feature_matrix_sanitization_rejects_duplicate_names(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate feature names before sanitization"):
            self.pipeline.sanitize_feature_matrices(
                [[1.0, 2.0], [2.0, 3.0]],
                [[1.0, 2.0]],
                [[1.0, 2.0]],
                ["duplicate", "duplicate"],
            )

    def test_feature_matrix_sanitization_marks_no_valid_features_as_expected(self):
        with self.assertRaisesRegex(
            evaluation_module.NoValidFeaturesError,
            "No valid compiled features remained after filtering",
        ):
            self.pipeline.sanitize_feature_matrices(
                [[1.0], [1.0]],
                [[1.0]],
                [[1.0]],
                ["constant"],
            )


if __name__ == "__main__":
    unittest.main()
