import importlib
import unittest


aggregate_rules = importlib.import_module("morace.rule_aggregation")
rule_verifier = importlib.import_module("morace.rule_verifier")


def _make_rule(
    source: str,
    feature_family: str,
    feature_name: str,
    feature_mode,
    relation_to_target,
    support_count: int,
    canonical_key: str,
    p_value: float = 0.01,
    effect_size: float = 1.0,
    threshold=None,
):
    return {
        "source": source,
        "primary_source": source,
        "feature_family": feature_family,
        "feature_name": feature_name,
        "feature_mode": feature_mode,
        "relation_to_target": relation_to_target,
        "canonical_key": canonical_key,
        "support_count": support_count,
        "support_sources": [source],
        "source_votes": {source: support_count},
        "compile_valid": True,
        "p_value": p_value,
        "effect_size": effect_size,
        "non_nan_coverage": 1.0,
        "feature_entropy": 0.8,
        "extraction_source": "projected",
        "threshold": threshold,
    }


class CandidateTemplateTests(unittest.TestCase):
    def setUp(self):
        pass

    def test_priority_sorted_candidates_is_source_neutral_without_explicit_priority(self):
        literature_rule = _make_rule(
            "literature",
            "descriptor",
            "TPSA",
            "continuous",
            "negative",
            2,
            "descriptor|TPSA|continuous|negative|null",
            p_value=0.02,
            effect_size=1.0,
        )
        data_rule = _make_rule(
            "data",
            "descriptor",
            "MolWt",
            "continuous",
            "negative",
            7,
            "descriptor|MolWt|continuous|negative|null",
            p_value=0.001,
            effect_size=3.0,
        )
        ranked = rule_verifier._priority_sorted_candidates([literature_rule, data_rule], source_priority=None)
        self.assertEqual(ranked[0]["source"], "data")
        self.assertEqual(ranked[0]["feature_name"], "MolWt")

    def test_alias_group_key_collapses_soft_mode_and_direction_aliases(self):
        lhs = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            None,
            "negative",
            8,
            "descriptor|TPSA|null|negative|null",
        )
        rhs = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "continuous",
            "monotonic_down",
            7,
            "descriptor|TPSA|continuous|monotonic_down|null",
        )
        self.assertEqual(aggregate_rules.alias_group_key(lhs), aggregate_rules.alias_group_key(rhs))

    def test_alias_group_summary_and_compaction_contract(self):
        data_rule = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            None,
            "negative",
            8,
            "descriptor|TPSA|null|negative|null",
        )
        literature_rule = _make_rule(
            "literature",
            "descriptor",
            "TPSA",
            "continuous",
            "monotonic_down",
            2,
            "descriptor|TPSA|continuous|monotonic_down|null",
        )

        self.assertEqual(
            aggregate_rules._alias_group_summary([data_rule, literature_rule]),
            [
                {
                    "alias_group_key": "descriptor::TPSA::down::continuousish",
                    "root_feature_key": "descriptor::TPSA",
                    "feature_family": "descriptor",
                    "feature_name": "TPSA",
                    "direction_bucket": "down",
                    "mode_bucket": "continuousish",
                    "rule_count": 2,
                    "total_support": 10,
                    "sources": {"data": 1, "literature": 1},
                    "canonical_keys": [
                        "descriptor|TPSA|null|negative|null",
                        "descriptor|TPSA|continuous|monotonic_down|null",
                    ],
                }
            ],
        )
        self.assertEqual(aggregate_rules._alias_group_count([data_rule, literature_rule]), 1)
        self.assertEqual(aggregate_rules._alias_compaction_score([data_rule, literature_rule]), 0.5)
        self.assertEqual(aggregate_rules._alias_compaction_score([]), 0.0)

    def test_semantic_variant_signature_and_equivalence_contract(self):
        continuous = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "continuous",
            "negative",
            4,
            "continuous",
        )
        thresholded_same_signal = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "threshold_binary",
            "negative",
            3,
            "thresholded-low",
            threshold={"op": "<=", "value": 90.0},
        )
        thresholded_opposite_signal = {
            **thresholded_same_signal,
            "canonical_key": "thresholded-high",
            "threshold": {"op": ">=", "value": 90.0},
        }

        self.assertEqual(
            aggregate_rules._semantic_variant_signature(continuous),
            ("descriptor::TPSA", "value_like", "low_value_signal"),
        )
        self.assertEqual(
            aggregate_rules._semantic_variant_signature(thresholded_same_signal),
            ("descriptor::TPSA", "presence_like", "low_value_signal"),
        )
        self.assertTrue(
            aggregate_rules._is_semantically_equivalent_second_variant(
                continuous,
                thresholded_same_signal,
            )
        )
        self.assertFalse(
            aggregate_rules._is_semantically_equivalent_second_variant(
                continuous,
                thresholded_opposite_signal,
            )
        )

    def test_second_variant_complement_and_quality_contract(self):
        existing = _make_rule(
            "literature",
            "descriptor",
            "TPSA",
            "continuous",
            "negative",
            4,
            "existing",
        )
        candidate = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "threshold_binary",
            "positive",
            2,
            "candidate",
            threshold={"op": ">=", "value": 90.0},
        )

        self.assertEqual(
            aggregate_rules._second_variant_complement_dimensions(existing, candidate),
            ["source", "direction", "mode"],
        )
        self.assertAlmostEqual(
            aggregate_rules._second_variant_pair_quality([existing], candidate),
            1.8875,
        )
        self.assertEqual(aggregate_rules._second_variant_pair_quality([], candidate), 0.0)

    def test_alias_representatives_choose_strongest_variant_per_group(self):
        stronger = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            None,
            "negative",
            8,
            "descriptor|TPSA|null|negative|null",
            p_value=0.001,
            effect_size=4.0,
        )
        weaker = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "continuous",
            "monotonic_down",
            7,
            "descriptor|TPSA|continuous|monotonic_down|null",
            p_value=0.002,
            effect_size=3.5,
        )
        reps = aggregate_rules._select_alias_representatives(
            [weaker, stronger],
            source_priority=("data", "multi", "literature"),
        )
        self.assertEqual(len(reps), 1)
        self.assertEqual(reps[0]["canonical_key"], "descriptor|TPSA|null|negative|null")

    def test_alias_representative_tie_breaks_and_quality_key_contract(self):
        data_alias = _make_rule(
            "data",
            "descriptor",
            "TPSA",
            "continuous",
            "negative",
            4,
            "z-data",
        )
        literature_alias = _make_rule(
            "literature",
            "descriptor",
            "TPSA",
            "continuous",
            "negative",
            4,
            "a-literature",
        )
        count_rule = _make_rule(
            "data",
            "count",
            "NumHBA",
            "count_value",
            "negative",
            4,
            "count-rule",
        )

        representatives = aggregate_rules._select_alias_representatives(
            [literature_alias, data_alias, count_rule],
            source_priority=("data", "multi", "literature"),
            prefer_non_descriptor=True,
        )
        self.assertEqual(
            [item["canonical_key"] for item in representatives],
            ["count-rule", "z-data"],
        )

        quality_rule = {
            **data_alias,
            "canonical_key": "quality-rule",
            "p_value": 0.01,
            "effect_size": -2.0,
            "non_nan_coverage": 0.9,
            "support_count": 4,
            "extraction_source": "projected",
        }
        self.assertEqual(
            aggregate_rules._candidate_quality_key(quality_rule),
            (0, 0.01, -2.0, -0.9, -4, "quality-rule"),
        )
        self.assertEqual(
            aggregate_rules._template_type_from_name("M3_literature_anchor_data_consensus"),
            "literature_anchor_data_consensus",
        )
        self.assertEqual(aggregate_rules._template_type_from_name("Custom"), "custom")

    def test_assign_rank_votes_is_tie_aware_for_equal_metric_values(self):
        candidates = [
            {
                "name": "M1_top_global",
                "selected_rule_count": 3,
                "compiled_rule_count": 3,
                "selected_root_feature_count": 3,
                "family_diversity": 2,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 1,
                "selected_non_descriptor_count": 1,
                "selected_avg_support": 4.0,
                "selected_avg_effect_size": 1.5,
                "template_novelty_score": 0.4,
                "selected_avg_p_value": 0.02,
            },
            {
                "name": "M2_branch_preserving",
                "selected_rule_count": 3,
                "compiled_rule_count": 3,
                "selected_root_feature_count": 3,
                "family_diversity": 2,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 1,
                "selected_non_descriptor_count": 1,
                "selected_avg_support": 4.0,
                "selected_avg_effect_size": 1.5,
                "template_novelty_score": 0.4,
                "selected_avg_p_value": 0.02,
            },
            {
                "name": "M3_literature_anchor_data_consensus",
                "selected_rule_count": 2,
                "compiled_rule_count": 2,
                "selected_root_feature_count": 2,
                "family_diversity": 1,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 0,
                "selected_non_descriptor_count": 0,
                "selected_avg_support": 2.0,
                "selected_avg_effect_size": 0.8,
                "template_novelty_score": 0.1,
                "selected_avg_p_value": 0.05,
            },
        ]
        votes = aggregate_rules._assign_rank_votes(candidates)
        self.assertEqual(votes["M1_top_global"]["metric_ranks"]["selected_rule_count"], 1.5)
        self.assertEqual(votes["M2_branch_preserving"]["metric_ranks"]["selected_rule_count"], 1.5)
        self.assertEqual(votes["M1_top_global"]["borda_score"], votes["M2_branch_preserving"]["borda_score"])
        self.assertEqual(votes["M1_top_global"]["mrr_score"], votes["M2_branch_preserving"]["mrr_score"])

    def test_assign_rank_votes_treats_nearly_equal_float_metrics_as_ties(self):
        candidates = [
            {
                "name": "M1_top_global",
                "selected_rule_count": 3,
                "compiled_rule_count": 3,
                "selected_root_feature_count": 3,
                "family_diversity": 2,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 1,
                "selected_non_descriptor_count": 1,
                "selected_avg_support": 4.0000000001,
                "selected_avg_effect_size": 1.5000000001,
                "template_novelty_score": 0.4000000001,
                "selected_avg_p_value": 0.0200000001,
            },
            {
                "name": "M2_branch_preserving",
                "selected_rule_count": 3,
                "compiled_rule_count": 3,
                "selected_root_feature_count": 3,
                "family_diversity": 2,
                "effective_literature_coverage": 1,
                "effective_data_coverage": 1,
                "selected_non_descriptor_count": 1,
                "selected_avg_support": 4.0000000002,
                "selected_avg_effect_size": 1.5000000002,
                "template_novelty_score": 0.4000000002,
                "selected_avg_p_value": 0.0200000002,
            },
        ]
        votes = aggregate_rules._assign_rank_votes(candidates)
        self.assertEqual(votes["M1_top_global"]["metric_ranks"]["selected_avg_support"], 1.5)
        self.assertEqual(votes["M2_branch_preserving"]["metric_ranks"]["selected_avg_support"], 1.5)
        self.assertEqual(votes["M1_top_global"]["metric_ranks"]["template_novelty_score"], 1.5)
        self.assertEqual(votes["M2_branch_preserving"]["metric_ranks"]["template_novelty_score"], 1.5)

    def test_assign_rank_votes_preserves_complete_metric_bundle_totals(self):
        stronger = {
            "name": "stronger",
            "selected_rule_count": 2,
            "compiled_rule_count": 2,
            "selected_root_feature_count": 2,
            "family_diversity": 2,
            "effective_literature_coverage": 2,
            "effective_data_coverage": 2,
            "selected_non_descriptor_count": 2,
            "selected_avg_support": 2.0,
            "selected_avg_effect_size": 2.0,
            "template_novelty_score": 2.0,
            "second_variant_quality_score": 2.0,
            "selected_avg_p_value": 0.01,
        }
        weaker = {
            "name": "weaker",
            "selected_rule_count": 1,
            "compiled_rule_count": 1,
            "selected_root_feature_count": 1,
            "family_diversity": 1,
            "effective_literature_coverage": 1,
            "effective_data_coverage": 1,
            "selected_non_descriptor_count": 1,
            "selected_avg_support": 1.0,
            "selected_avg_effect_size": 1.0,
            "template_novelty_score": 1.0,
            "second_variant_quality_score": 1.0,
            "selected_avg_p_value": 0.02,
        }

        votes = aggregate_rules._assign_rank_votes([stronger, weaker])
        self.assertEqual(set(votes["stronger"]["metric_ranks"].values()), {1.0})
        self.assertEqual(set(votes["weaker"]["metric_ranks"].values()), {2.0})
        self.assertEqual(votes["stronger"]["borda_score"], 24.0)
        self.assertEqual(votes["stronger"]["mrr_score"], 12.0)
        self.assertEqual(votes["weaker"]["borda_score"], 12.0)
        self.assertEqual(votes["weaker"]["mrr_score"], 6.0)
        self.assertEqual(aggregate_rules._assign_rank_votes([]), {})

    def test_template_novelty_jaccard_contract(self):
        rule_a = _make_rule("data", "descriptor", "A", "continuous", "negative", 3, "a")
        rule_b = _make_rule("data", "descriptor", "B", "continuous", "negative", 3, "b")
        rule_c = _make_rule("data", "descriptor", "C", "continuous", "negative", 3, "c")
        candidate_sets = [
            {"name": "M1", "rules": [rule_a, rule_b]},
            {"name": "M2", "rules": [rule_b, rule_c]},
            {"name": "M3", "rules": [rule_a, rule_b]},
        ]

        aggregate_rules._attach_template_novelty_scores(candidate_sets)

        self.assertAlmostEqual(candidate_sets[0]["template_novelty_score"], 1.0 / 3.0)
        self.assertAlmostEqual(candidate_sets[1]["template_novelty_score"], 2.0 / 3.0)
        self.assertAlmostEqual(candidate_sets[2]["template_novelty_score"], 1.0 / 3.0)
        self.assertAlmostEqual(
            aggregate_rules._single_candidate_template_novelty(
                {"name": "new", "rules": [rule_a]},
                candidate_sets[:2],
            ),
            0.75,
        )
        self.assertEqual(
            aggregate_rules._single_candidate_template_novelty(
                {"name": "new", "rules": [rule_a]},
                [],
            ),
            0.0,
        )

    def test_candidate_templates_produce_at_least_three_distinct_structures(self):
        enriched_rules = [
            _make_rule("literature", "count", "NumHBD", "count_value", "negative", 1, "count|NumHBD|count_value|negative|null"),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 1, "descriptor|MolLogP|range|positive|null"),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 8, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "TPSA", "continuous", "monotonic_down", 7, "descriptor|TPSA|continuous|monotonic_down|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", "continuous", "negative", 4, "descriptor|MolWt|continuous|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 3, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "descriptor", "MolLogP", None, "positive", 2, "descriptor|MolLogP|null|positive|null"),
            _make_rule(
                "data",
                "count",
                "NumRotatableBonds",
                "count_value",
                "negative",
                2,
                "count|NumRotatableBonds|count_value|negative|null",
            ),
            _make_rule(
                "data",
                "pattern_presence",
                "halogen",
                "binary_presence",
                "negative",
                2,
                "pattern_presence|halogen|binary_presence|negative|null",
                p_value=0.015,
                effect_size=1.4,
            ),
        ]
        candidate_sets = aggregate_rules._build_candidate_rule_sets(enriched_rules, target_rule_count=5)
        self.assertEqual(len(candidate_sets), 6)
        structures = {
            tuple(
                sorted(
                    f"{rule.get('source')}::{rule.get('feature_family')}::{rule.get('feature_name')}::{rule.get('relation_to_target')}::{rule.get('feature_mode')}"
                    for rule in item.get("rules", [])
                )
            )
            for item in candidate_sets
        }
        self.assertGreaterEqual(len(structures), 3)
        alias_signatures = {
            tuple(sorted(aggregate_rules.alias_group_key(rule) for rule in item.get("rules", [])))
            for item in candidate_sets
        }
        self.assertGreaterEqual(len(alias_signatures), 3)
        novelty_scores = [float(item.get("template_novelty_score", 0.0)) for item in candidate_sets]
        self.assertGreaterEqual(sum(1 for score in novelty_scores if score > 0.10), 3)
        self.assertTrue(all(item.get("template_type") for item in candidate_sets))
        self.assertEqual(aggregate_rules.CANDIDATE_SET_VOTE_METHOD, "borda_then_mrr_average_rank_tie_aware")
        self.assertEqual(
            [
                (
                    item["name"],
                    item["template_type"],
                    item["selected_rule_count"],
                    item["selected_root_feature_count"],
                    item["family_diversity"],
                    item["effective_literature_coverage"],
                    item["effective_data_coverage"],
                    item["borda_score"],
                    round(item["mrr_score"], 12),
                    round(item["template_novelty_score"], 12),
                )
                for item in candidate_sets
            ],
            [
                ("M1_top_global", "baseline_global", 5, 5, 3, 0, 5, 45.0, 4.539826839827, 0.409523809524),
                (
                    "M5_non_descriptor_structure",
                    "non_descriptor_structure",
                    5,
                    5,
                    3,
                    0,
                    5,
                    45.0,
                    4.539826839827,
                    0.409523809524,
                ),
                (
                    "M6_descriptor_capped_balanced",
                    "descriptor_capped_balanced",
                    5,
                    5,
                    2,
                    2,
                    3,
                    42.5,
                    4.502453102453,
                    0.52380952381,
                ),
                ("M4_data_heavy_alias", "data_heavy", 5, 5, 2, 0, 5, 40.5, 3.637301587302, 0.428571428571),
                (
                    "M2_branch_preserving",
                    "baseline_branch_preserving",
                    5,
                    5,
                    2,
                    1,
                    4,
                    40.0,
                    3.420634920635,
                    0.47619047619,
                ),
                (
                    "M3_literature_anchor_data_consensus",
                    "literature_anchor_data_consensus",
                    5,
                    5,
                    2,
                    2,
                    3,
                    39.0,
                    3.635786435786,
                    0.47619047619,
                ),
            ],
        )

    def test_baseline_templates_keep_strict_root_cap(self):
        enriched_rules = [
            _make_rule("literature", "descriptor", "TPSA", "threshold", "negative", 3, "descriptor|TPSA|threshold|negative|null"),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 2, "descriptor|MolLogP|range|positive|null"),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "count", "NumHBD", "count_value", "negative", 3, "count|NumHBD|count_value|negative|null"),
        ]
        candidate_sets = aggregate_rules._build_candidate_rule_sets(enriched_rules, target_rule_count=5)
        baseline = next(item for item in candidate_sets if item["name"] == "M1_top_global")
        self.assertEqual(baseline["capacity_release_mode"], "strict_root_cap_1")
        root_counts = {}
        for rule in baseline["rules"]:
            root = f"{rule['feature_family']}::{rule['feature_name']}"
            root_counts[root] = root_counts.get(root, 0) + 1
        self.assertTrue(all(count == 1 for count in root_counts.values()))
        self.assertEqual(baseline["selected_second_variant_count"], 0)

    def test_template_selection_cap_precedence_and_drop_order_contract(self):
        selected_first = _make_rule(
            "data", "descriptor", "DataA", "continuous", "negative", 5, "selected-data-a"
        )
        candidates = [
            selected_first,
            {**selected_first, "feature_name": "DuplicateCanonical"},
            {**selected_first, "canonical_key": "same-root-variant"},
            _make_rule("data", "descriptor", "DataB", "continuous", "negative", 4, "data-cap"),
            _make_rule("literature", "count", "LitA", "count_value", "negative", 4, "selected-lit"),
            _make_rule(
                "literature",
                "pattern_presence",
                "LitB",
                "binary_presence",
                "negative",
                3,
                "literature-cap",
            ),
            _make_rule("multi", "descriptor", "MultiA", "continuous", "negative", 4, "selected-multi"),
            _make_rule("multi", "descriptor", "MultiB", "continuous", "negative", 3, "descriptor-cap"),
            _make_rule("multi", "count", "MultiC", "count_value", "negative", 3, "selected-count"),
        ]

        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=4,
            template_type="cap_contract",
            source_priority=("data", "multi", "literature"),
            descriptor_cap_ratio=0.5,
            max_literature_rules=1,
            max_data_rules=1,
        )

        self.assertEqual(
            [rule["canonical_key"] for rule in selected],
            ["selected-data-a", "selected-lit", "selected-multi", "selected-count"],
        )
        self.assertEqual(
            [(item["rule"]["canonical_key"], item["reason"]) for item in audit["dropped"]],
            [
                ("selected-data-a", "canonical_duplicate"),
                ("same-root-variant", "root_feature_cap"),
                ("data-cap", "data_cap"),
                ("literature-cap", "literature_cap"),
                ("descriptor-cap", "descriptor_cap"),
            ],
        )
        self.assertEqual(audit["capacity_release_mode"], "strict_root_cap_1")
        self.assertEqual(audit["selected_second_variant_count"], 0)

    def test_balanced_coverage_compatibility_selector_contract(self):
        rules = [
            _make_rule("data", "descriptor", "DataDescriptor", "continuous", "negative", 4, "data-desc"),
            _make_rule("multi", "count", "MultiCount", "count_value", "negative", 4, "multi-count"),
            _make_rule(
                "literature",
                "descriptor",
                "LiteratureDescriptor",
                "continuous",
                "negative",
                4,
                "lit-desc",
            ),
            _make_rule("data", "count", "DataCount", "count_value", "negative", 4, "data-count"),
            _make_rule(
                "multi",
                "descriptor",
                "MultiDescriptor",
                "continuous",
                "negative",
                4,
                "multi-desc",
            ),
            _make_rule(
                "literature",
                "pattern_presence",
                "LiteraturePattern",
                "binary_presence",
                "negative",
                4,
                "lit-pattern",
            ),
        ]

        selected, audit = aggregate_rules._select_balanced_coverage_rules(rules, target_rule_count=6)

        self.assertEqual(
            [rule["canonical_key"] for rule in selected],
            ["lit-pattern", "lit-desc", "data-count", "data-desc", "multi-count", "multi-desc"],
        )
        self.assertEqual(audit, {"dropped": [], "strategy": "balanced_source_family_coverage"})

    def test_m3_can_keep_second_same_root_rule_when_alias_is_complementary(self):
        candidates = [
            _make_rule(
                "literature",
                "descriptor",
                "TPSA",
                "threshold",
                "positive",
                3,
                "descriptor|TPSA|threshold|positive|>=:90",
                threshold={"op": ">=", "value": 90.0},
            ),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 2, "descriptor|MolLogP|range|positive|null"),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
        ]
        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=5,
            template_type="literature_anchor_data_consensus",
            source_priority=("data", "multi", "literature"),
            literature_anchor_count=2,
            data_seed_count=2,
            require_non_descriptor_count=1,
            descriptor_cap_ratio=0.70,
            max_literature_rules=2,
            max_same_root_per_template=2,
            allow_second_variant_by_root=True,
        )
        tpsa_rules = [rule for rule in selected if rule["feature_name"] == "TPSA"]
        self.assertEqual(len(tpsa_rules), 2)
        self.assertEqual(audit["capacity_release_mode"], "local_root_cap_release_max2")
        self.assertEqual(audit["selected_second_variant_count"], 1)
        self.assertEqual(audit["selected_repeated_root_count"], 1)
        self.assertEqual(audit["selected_cross_source_second_variant_count"], 1)
        self.assertEqual(len(audit["selected_same_root_alias_pairs"]), 1)
        self.assertGreater(audit["second_variant_quality_score"], 0.0)

    def test_non_complementary_same_alias_second_variant_is_blocked(self):
        candidates = [
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "TPSA", "continuous", "monotonic_down", 5, "descriptor|TPSA|continuous|monotonic_down|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 4, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 2, "descriptor|MolLogP|range|positive|null"),
        ]
        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=5,
            template_type="data_heavy",
            source_priority=("data", "multi", "literature"),
            data_seed_count=3,
            require_non_descriptor_count=1,
            descriptor_cap_ratio=0.80,
            max_literature_rules=1,
            max_same_root_per_template=2,
            allow_second_variant_by_root=True,
        )
        tpsa_rules = [rule for rule in selected if rule["feature_name"] == "TPSA"]
        self.assertEqual(len(tpsa_rules), 1)
        self.assertEqual(audit["selected_second_variant_count"], 0)
        self.assertTrue(any(item["reason"] == "second_variant_same_alias" for item in audit["dropped"]))

    def test_m4_can_keep_second_same_root_rule_when_complementary(self):
        candidates = [
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule(
                "data",
                "descriptor",
                "TPSA",
                "threshold",
                "positive",
                4,
                "descriptor|TPSA|threshold|positive|>=:90",
                threshold={"op": ">=", "value": 90.0},
            ),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 2, "descriptor|MolLogP|range|positive|null"),
        ]
        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=5,
            template_type="data_heavy",
            source_priority=("data", "multi", "literature"),
            data_seed_count=3,
            require_non_descriptor_count=1,
            descriptor_cap_ratio=0.80,
            max_literature_rules=1,
            max_same_root_per_template=2,
            allow_second_variant_by_root=True,
        )
        tpsa_rules = [rule for rule in selected if rule["feature_name"] == "TPSA"]
        self.assertEqual(len(tpsa_rules), 2)
        self.assertEqual(audit["capacity_release_mode"], "local_root_cap_release_max2")
        self.assertEqual(audit["selected_second_variant_count"], 1)
        self.assertGreater(audit["second_variant_quality_score"], 0.0)

    def test_same_source_second_variant_can_pass_when_direction_or_mode_differs(self):
        candidates = [
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule(
                "data",
                "descriptor",
                "TPSA",
                "threshold",
                "positive",
                4,
                "descriptor|TPSA|threshold|positive|>=:90",
                threshold={"op": ">=", "value": 90.0},
            ),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
        ]
        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=4,
            template_type="descriptor_capped_balanced",
            source_priority=("multi", "data", "literature"),
            require_non_descriptor_count=1,
            descriptor_cap_ratio=0.75,
            max_same_root_per_template=2,
            allow_second_variant_by_root=True,
        )
        tpsa_rules = [rule for rule in selected if rule["feature_name"] == "TPSA"]
        self.assertEqual(len(tpsa_rules), 2)
        self.assertEqual(audit["selected_second_variant_count"], 1)
        self.assertEqual(audit["selected_cross_source_second_variant_count"], 0)
        self.assertIn(
            "direction",
            audit["selected_same_root_alias_pairs"][0]["complement_dimensions"],
        )

    def test_semantically_equivalent_second_variant_is_blocked_even_if_alias_differs(self):
        candidates = [
            _make_rule(
                "literature",
                "descriptor",
                "TPSA",
                "threshold",
                "positive",
                3,
                "descriptor|TPSA|threshold|positive|<=:90",
                threshold={"op": "<=", "value": 90.0},
            ),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 6, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 5, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("literature", "descriptor", "MolLogP", "range", "positive", 2, "descriptor|MolLogP|range|positive|null"),
        ]
        selected, audit = aggregate_rules._select_template_rules(
            candidates,
            target_rule_count=5,
            template_type="literature_anchor_data_consensus",
            source_priority=("data", "multi", "literature"),
            literature_anchor_count=2,
            data_seed_count=2,
            require_non_descriptor_count=1,
            descriptor_cap_ratio=0.70,
            max_literature_rules=2,
            max_same_root_per_template=2,
            allow_second_variant_by_root=True,
        )
        tpsa_rules = [rule for rule in selected if rule["feature_name"] == "TPSA"]
        self.assertEqual(len(tpsa_rules), 1)
        self.assertEqual(audit["selected_second_variant_count"], 0)
        self.assertTrue(any(item["reason"] == "second_variant_semantic_equivalent" for item in audit["dropped"]))

    def test_m7_lit_preserving_swap_creates_distinct_candidate_when_novel_literature_root_exists(self):
        enriched_rules = [
            _make_rule("literature", "count", "NumHBD", "count_value", "negative", 2, "count|NumHBD|count_value|negative|null"),
            {
                **_make_rule("multi", "descriptor", "PEOE_VSA1", "continuous", "negative", 3, "descriptor|PEOE_VSA1|continuous|negative|null"),
                "support_sources": ["literature", "data"],
                "source_votes": {"literature": 1, "data": 2},
            },
            _make_rule("data", "descriptor", "TPSA", None, "negative", 8, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 6, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "descriptor", "MolLogP", None, "positive", 5, "descriptor|MolLogP|null|positive|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "count", "NumRotatableBonds", "count_value", "negative", 3, "count|NumRotatableBonds|count_value|negative|null"),
        ]
        candidate_sets = aggregate_rules._build_candidate_rule_sets(enriched_rules, target_rule_count=6, m7_lit_preserving_swap_enable=True)
        names = [item["name"] for item in candidate_sets]
        self.assertIn("M7_lit_preserving_swap", names)
        m1 = next(item for item in candidate_sets if item["name"] == "M1_top_global")
        m7 = next(item for item in candidate_sets if item["name"] == "M7_lit_preserving_swap")
        self.assertNotEqual(tuple(m1["alias_group_signature"]), tuple(m7["alias_group_signature"]))
        self.assertEqual(m7["selection_audit"]["strategy"], "lit_preserving_swap")
        self.assertEqual(m7["selected_rule_count"], m1["selected_rule_count"])
        self.assertGreaterEqual(m7["effective_literature_coverage"], 2)
        self.assertGreaterEqual(m7["effective_data_coverage"], 5)
        self.assertIn("multi", m7["selected_source_breakdown"])

    def test_m7_swap_candidate_and_audit_contract(self):
        multi_rule = {
            **_make_rule("multi", "descriptor", "MultiRoot", "continuous", "negative", 1, "multi-root"),
            "support_sources": ["literature", "data"],
        }
        data_rules = [
            _make_rule(
                "data",
                "descriptor",
                f"DataRoot{index}",
                "continuous",
                "negative",
                index + 1,
                f"data-{index}",
            )
            for index in range(1, 6)
        ]
        reserve = _make_rule(
            "literature",
            "count",
            "ReserveLit",
            "count_value",
            "negative",
            4,
            "reserve-lit",
        )
        anchor = {"name": "M1_top_global", "rules": [multi_rule, *data_rules]}

        swapped, audit = aggregate_rules._build_lit_preserving_swap_candidate(
            anchor,
            [multi_rule, *data_rules, reserve],
            target_rule_count=6,
        )

        self.assertEqual(
            [rule["canonical_key"] for rule in swapped],
            ["multi-root", "data-2", "data-3", "data-4", "data-5", "reserve-lit"],
        )
        self.assertEqual(
            audit,
            {
                "dropped": [],
                "strategy": "lit_preserving_swap",
                "static_reason": "single_swap_preserve_m1_with_novel_literature_root",
                "anchor_candidate_name": "M1_top_global",
                "replaced_rule_canonical_key": "data-1",
                "reserve_rule_canonical_key": "reserve-lit",
                "reserve_root_feature_key": "count::ReserveLit",
                "reserve_alias_group_key": "count::ReserveLit::down::count_value",
                "distinct_signature_created": True,
            },
        )

    def test_winner_explanation_contract(self):
        winner = {
            "name": "Winner",
            "template_type": "winner_type",
            "borda_score": 10.0,
            "mrr_score": 2.0,
            "selected_rule_count": 6,
            "effective_data_coverage": 5,
            "effective_literature_coverage": 2,
            "selected_non_descriptor_count": 2,
            "selected_root_feature_count": 6,
            "mean_support": 4.0,
            "template_novelty_score": 0.2,
            "second_variant_quality_score": 1.0,
            "selected_alias_group_count": 6,
            "alias_group_signature": ["shared"],
        }
        runner_up = {
            **winner,
            "name": "Runner",
            "template_type": "runner_type",
            "selected_root_feature_count": 5,
        }

        explanation = aggregate_rules._winner_explanation([winner, runner_up])

        self.assertEqual(explanation["winner"], "Winner")
        self.assertEqual(explanation["runner_up"], "Runner")
        self.assertEqual(explanation["reason"], "higher_ranked_vote_bundle")
        self.assertEqual(explanation["winner_decided_by"], "selected_root_feature_count")
        self.assertEqual(explanation["top_driver"], "root_feature_delta")
        self.assertEqual(explanation["deltas_vs_runner_up"]["root_feature_delta"], 1)
        self.assertEqual(explanation["deltas_vs_runner_up"]["alias_group_jaccard_similarity"], 1.0)
        self.assertEqual(aggregate_rules._winner_explanation([]), {"reason": "no_candidates"})
        self.assertEqual(
            aggregate_rules._winner_explanation([winner]),
            {
                "winner": "Winner",
                "winner_template_type": "winner_type",
                "runner_up": "",
                "runner_up_template_type": "",
                "reason": "single_candidate_set",
            },
        )

    def test_m7_is_not_added_when_only_pure_literature_reserve_would_break_data_coverage(self):
        enriched_rules = [
            _make_rule("literature", "count", "NumHBD", "count_value", "negative", 2, "count|NumHBD|count_value|negative|null"),
            _make_rule("literature", "descriptor", "PEOE_VSA1", "continuous", "negative", 3, "descriptor|PEOE_VSA1|continuous|negative|null"),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 8, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 6, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "descriptor", "MolLogP", None, "positive", 5, "descriptor|MolLogP|null|positive|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "count", "NumRotatableBonds", "count_value", "negative", 3, "count|NumRotatableBonds|count_value|negative|null"),
        ]
        candidate_sets = aggregate_rules._build_candidate_rule_sets(enriched_rules, target_rule_count=6, m7_lit_preserving_swap_enable=True)
        self.assertNotIn("M7_lit_preserving_swap", [item["name"] for item in candidate_sets])

    def test_compile_safe_multi_feasibility_metrics_report_budget_feasible_multi_geometry(self):
        merged_rules = [
            {
                "canonical_key": "descriptor|PEOE_VSA1|continuous|negative|null",
                "feature_family": "descriptor",
                "feature_name": "PEOE_VSA1",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "support_sources": ["literature", "data"],
                "source_votes": {"literature": 1, "data": 2},
                "source": "multi",
                "compile_valid": True,
            }
        ]
        enriched_rules = [
            _make_rule("literature", "count", "NumHBD", "count_value", "negative", 2, "count|NumHBD|count_value|negative|null"),
            _make_rule("data", "descriptor", "TPSA", None, "negative", 8, "descriptor|TPSA|null|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 6, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "descriptor", "MolLogP", None, "positive", 5, "descriptor|MolLogP|null|positive|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "count", "NumRotatableBonds", "count_value", "negative", 3, "count|NumRotatableBonds|count_value|negative|null"),
            {
                **_make_rule("multi", "descriptor", "PEOE_VSA1", "continuous", "negative", 3, "descriptor|PEOE_VSA1|continuous|negative|null"),
                "support_sources": ["literature", "data"],
                "source_votes": {"literature": 1, "data": 2},
            },
        ]
        metrics = aggregate_rules._compile_safe_multi_feasibility_metrics(
            merged_rules,
            enriched_rules,
            target_rule_count=6,
        )
        self.assertEqual(metrics["compile_safe_multi_rule_count"], 1)
        self.assertEqual(metrics["compile_safe_multi_root_count"], 1)
        self.assertIn("descriptor::PEOE_VSA1", metrics["compile_safe_multi_root_keys"])
        self.assertTrue(metrics["coverage_feasible_under_budget"])

    def test_coverage_feasibility_counts_distinct_compile_safe_roots(self):
        rules = [
            _make_rule("literature", "count", "LitRoot", "count_value", "negative", 2, "lit"),
            _make_rule("multi", "descriptor", "MultiRoot", "continuous", "negative", 2, "multi"),
            _make_rule("data", "descriptor", "DataRoot1", "continuous", "negative", 2, "data-1"),
            _make_rule("data", "descriptor", "DataRoot2", "continuous", "negative", 2, "data-2"),
            _make_rule("data", "descriptor", "DataRoot3", "continuous", "negative", 2, "data-3"),
            _make_rule("data", "descriptor", "DataRoot4", "continuous", "negative", 2, "data-4"),
        ]

        self.assertTrue(aggregate_rules._coverage_feasible_under_budget(rules, target_rule_count=6))
        self.assertFalse(aggregate_rules._coverage_feasible_under_budget(rules, target_rule_count=5))

        same_root_variant = dict(rules[-1], feature_name="DataRoot3", canonical_key="data-3-variant")
        self.assertFalse(
            aggregate_rules._coverage_feasible_under_budget(
                [*rules[:-1], same_root_variant],
                target_rule_count=6,
            )
        )

        compile_invalid = dict(rules[-1], compile_valid=False)
        self.assertFalse(
            aggregate_rules._coverage_feasible_under_budget(
                [*rules[:-1], compile_invalid],
                target_rule_count=6,
            )
        )

    def test_compile_safe_cross_source_root_count_uses_enriched_compile_valid_instead_of_merged(self):
        merged_rules = [
            {
                "canonical_key": "descriptor|TPSA|continuous|negative|null",
                "feature_family": "descriptor",
                "feature_name": "TPSA",
                "feature_mode": "continuous",
                "relation_to_target": "negative",
                "support_sources": ["literature", "data"],
                "source_votes": {"literature": 1, "data": 2},
                "source": "multi",
            }
        ]
        enriched_rules = [
            {
                **_make_rule(
                    "multi",
                    "descriptor",
                    "TPSA",
                    "continuous",
                    "negative",
                    3,
                    "descriptor|TPSA|continuous|negative|null",
                ),
                "support_sources": ["literature", "data"],
                "source_votes": {"literature": 1, "data": 2},
                "compile_valid": True,
            },
            _make_rule("literature", "count", "NumHBD", "count_value", "negative", 2, "count|NumHBD|count_value|negative|null"),
            _make_rule("data", "descriptor", "MolWt", None, "negative", 6, "descriptor|MolWt|null|negative|null"),
            _make_rule("data", "descriptor", "MolLogP", None, "positive", 5, "descriptor|MolLogP|null|positive|null"),
            _make_rule("data", "count", "NumHBA", "count_value", "negative", 4, "count|NumHBA|count_value|negative|null"),
            _make_rule("data", "count", "NumRotatableBonds", "count_value", "negative", 3, "count|NumRotatableBonds|count_value|negative|null"),
            _make_rule("data", "descriptor", "PEOE_VSA1", None, "negative", 3, "descriptor|PEOE_VSA1|null|negative|null"),
        ]
        metrics = aggregate_rules._compile_safe_multi_feasibility_metrics(
            merged_rules,
            enriched_rules,
            target_rule_count=6,
        )
        self.assertEqual(metrics["compile_safe_cross_source_root_count"], 1)
        self.assertIn("descriptor::TPSA", metrics["compile_safe_cross_source_root_keys"])


if __name__ == "__main__":
    unittest.main()
