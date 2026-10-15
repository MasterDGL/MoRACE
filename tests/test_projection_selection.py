import unittest
from unittest.mock import patch


def load_projection_selection():
    try:
        from morace import projection_selection
    except ModuleNotFoundError as exc:
        if exc.name == "rdkit":
            raise unittest.SkipTest("projection selection requires the core RDKit dependency") from exc
        raise
    return projection_selection


class ProjectionSelectionTests(unittest.TestCase):
    def test_projection_status_distinguishes_success_rescue_and_parse_error(self):
        selection = load_projection_selection()

        self.assertEqual(selection.classify_projection_status("", [{}], []), "projected")
        self.assertEqual(selection.classify_projection_status("", [], [{}]), "heuristic_rescued")
        self.assertEqual(selection.classify_projection_status('{"rules":', [], []), "parse_error")
        self.assertEqual(selection.classify_projection_status("", [], []), "empty")

    def test_feature_family_helpers_preserve_first_seen_order(self):
        selection = load_projection_selection()
        rules = [
            {"feature_family": "count", "feature_name": "NumHBA"},
            {"feature_family": "descriptor", "feature_name": "MolLogP"},
            {"feature_family": "count", "feature_name": "NumHBD"},
        ]

        self.assertEqual(selection.feature_families(rules), ["count", "descriptor"])
        self.assertEqual(selection.feature_family_diversity_count(rules), 2)
        self.assertEqual(
            selection.late_selection_root_keys(rules),
            {"count::NumHBA", "descriptor::MolLogP", "count::NumHBD"},
        )

    def test_selection_score_rewards_strict_compile_safe_candidate(self):
        selection = load_projection_selection()
        rule = {
            "source": "literature",
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
            "threshold": None,
            "rationale": "lipophilicity",
        }
        base = {
            "draft_text": "MolLogP and lipophilicity may matter.",
            "parsed_rules": [rule],
            "strict_projector_valid": True,
            "stage2_logprob_norm": 0.5,
        }

        self.assertGreater(selection.compile_valid_ratio([rule]), 0.0)
        self.assertGreater(selection.score_projected_record(base), selection.score_projected_record({}))

    def test_pipeline_runner_reexports_selection_helpers(self):
        selection = load_projection_selection()
        from morace import pipeline_runner

        self.assertIs(pipeline_runner._compile_valid_ratio, selection.compile_valid_ratio)
        self.assertIs(pipeline_runner._feature_families, selection.feature_families)
        self.assertIs(pipeline_runner.literature_projection_profile, selection.literature_projection_profile)
        self.assertIs(pipeline_runner._score_projected_record, selection.score_projected_record)

    def test_pipeline_wrapper_passes_dynamic_thresholds_without_changing_output(self):
        selection = load_projection_selection()
        from morace import pipeline_runner

        def record(draft_id, score, feature_name):
            return {
                "prompt_id": "p1",
                "draft_model": "llama",
                "draft_id": draft_id,
                "batch_id": None,
                "stage2_logprob_norm": score,
                "strict_projector_valid": True,
                "parsed_rules": [
                    {
                        "source": "literature",
                        "feature_family": "descriptor",
                        "feature_name": feature_name,
                        "feature_mode": "numeric_value",
                        "relation_to_target": "positive",
                        "threshold": None,
                        "rationale": "signal",
                    }
                ],
            }

        records = [record(0, 1.0, "MolLogP"), record(1, 0.9, "TPSA")]
        settings = {
            "keep_topk": 2,
            "max_score_delta": 1.0,
            "max_survivors_per_prompt": 2,
            "require_novelty": True,
        }
        expected = selection.select_best_projected_records(records, **settings)
        with (
            patch.object(pipeline_runner, "LATE_SELECTION_KEEP_TOPK", settings["keep_topk"]),
            patch.object(pipeline_runner, "LATE_SELECTION_MAX_SCORE_DELTA", settings["max_score_delta"]),
            patch.object(
                pipeline_runner,
                "LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT",
                settings["max_survivors_per_prompt"],
            ),
            patch.object(pipeline_runner, "LATE_SELECTION_REQUIRE_NOVELTY", settings["require_novelty"]),
        ):
            actual = pipeline_runner.select_best_projected_records(records)

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
