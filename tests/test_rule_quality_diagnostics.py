from morace.diagnostics.rule_quality import classify_rule_type, summarize_leave_one_out


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
        "feature_mode": "None",
        "relation_to_target": "negative",
        "threshold": None,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources or [source],
        "canonical_key": canonical_key,
        "consensus_promoted": consensus_promoted,
    }


def test_classify_rule_type_marks_promoted_multi_and_same_root_alias():
    winner_rules = [
        _rule(
            "TPSA",
            "multi",
            "descriptor|TPSA|null|negative|null",
            support_sources=["data", "literature"],
            consensus_promoted=True,
        ),
        _rule("TPSA", "data", "descriptor|TPSA|threshold_binary|negative|null"),
        _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
        _rule("MolWt", "data", "descriptor|MolWt|null|negative|null"),
    ]
    assert classify_rule_type(winner_rules[0], winner_rules) == "promoted_multi"
    assert classify_rule_type(winner_rules[1], winner_rules) == "same_root_alias"
    assert classify_rule_type(winner_rules[2], winner_rules) == "literature_only"
    assert classify_rule_type(winner_rules[3], winner_rules) == "data_only"


def test_summarize_leave_one_out_flags_test_harmful_rules():
    rows = [
        {
            "dropped_rule_type": "promoted_multi",
            "valid_delta_vs_full": -0.0020,
            "test_delta_vs_full": 0.0035,
            "dropped_feature_name": "TPSA",
            "dropped_rule_id": "descriptor|TPSA|null|negative|null",
            "possible_test_harmful_rule": True,
            "validation_friendly_test_harmful_rule": True,
        },
        {
            "dropped_rule_type": "literature_only",
            "valid_delta_vs_full": 0.0005,
            "test_delta_vs_full": -0.0010,
            "dropped_feature_name": "NumHBD",
            "dropped_rule_id": "count|NumHBD|count_value|negative|null",
            "possible_test_harmful_rule": False,
            "validation_friendly_test_harmful_rule": False,
        },
    ]
    summary = summarize_leave_one_out(rows)
    grouped = {item["rule_type"]: item for item in summary["grouped_summary"]}
    assert grouped["promoted_multi"]["rule_count"] == 1
    assert grouped["promoted_multi"]["avg_test_delta"] == 0.0035
    assert len(summary["suspect_rules"]["possible_test_harmful_rule"]) == 1
    assert len(summary["suspect_rules"]["validation_friendly_test_harmful_rule"]) == 1
