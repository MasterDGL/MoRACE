from morace.diagnostics.synthesis_feature_stability import (
    build_root_comparison_row,
    extract_root_state_from_all,
    extract_root_state_from_synth,
)


def _rule(feature_family, feature_name, source, canonical_key, *, feature_mode="None", relation="negative", compile_valid=True, consensus_promoted=False, support_sources=None):
    return {
        "feature_family": feature_family,
        "feature_name": feature_name,
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources or [source],
        "canonical_key": canonical_key,
        "compile_valid": compile_valid,
        "consensus_promoted": consensus_promoted,
    }


def test_extract_root_state_from_synth_counts_canonical_presence_and_alias_groups():
    synth_artifact = {
        "branch_metrics": {
            "synthesize": {
                "selection_trace": {
                    "raw_rules": [
                        _rule("descriptor", "TPSA", "literature", "descriptor|TPSA|continuous|negative|null", feature_mode="continuous"),
                    ],
                    "canonical_rules": [
                        _rule("descriptor", "TPSA", "literature", "descriptor|TPSA|continuous|negative|null", feature_mode="continuous"),
                    ],
                }
            }
        },
        "rules": [
            _rule("descriptor", "TPSA", "literature", "descriptor|TPSA|continuous|negative|null", feature_mode="continuous"),
        ],
    }
    state = extract_root_state_from_synth(synth_artifact, "descriptor::TPSA")
    assert state["root_present_in_synth_rules"] is True
    assert state["root_present_in_canonical_rules"] is True
    assert state["root_present_in_final_synth_rules"] is True
    assert state["canonical_rule_count"] == 1
    assert state["canonical_compile_valid_count"] == 1
    assert state["canonical_alias_group_count"] == 1


def test_extract_root_state_from_all_tracks_consensus_and_promoted_multi():
    all_artifact = {
        "branch_metrics": {
            "combined": {
                "cross_source_consensus_groups": [
                    {
                        "root_feature_key": "descriptor::TPSA",
                        "has_cross_source_overlap": True,
                        "compile_valid_true_count": 2,
                    }
                ],
                "selection_trace": {
                    "gated_rules": [
                        _rule(
                            "descriptor",
                            "TPSA",
                            "multi",
                            "descriptor|TPSA|null|negative|null",
                            support_sources=["data", "literature"],
                            consensus_promoted=True,
                        )
                    ]
                },
            }
        },
        "rules": [
            _rule(
                "descriptor",
                "TPSA",
                "multi",
                "descriptor|TPSA|null|negative|null",
                support_sources=["data", "literature"],
                consensus_promoted=True,
            )
        ],
    }
    state = extract_root_state_from_all(all_artifact, "descriptor::TPSA")
    assert state["present_in_cross_source_consensus"] is True
    assert state["cross_source_compile_valid_true_count"] == 2
    assert state["promoted_multi_candidate_present"] is True
    assert state["promoted_multi_compile_safe_present"] is True
    assert state["present_in_final_winner"] is True
    assert state["final_source_bucket"] == "multi"
    assert state["present_in_surviving_rules"] is True


def test_build_root_comparison_row_marks_first_missing_stage():
    row = build_root_comparison_row(
        "weak_run",
        "descriptor::TPSA",
        {
            "root_present_in_synth_rules": True,
            "root_present_in_canonical_rules": True,
            "root_present_in_final_synth_rules": True,
            "raw_rule_count": 1,
            "canonical_rule_count": 1,
            "canonical_source_breakdown": {"literature": 1},
            "canonical_compile_valid_count": 1,
            "canonical_compile_valid_known_count": 1,
            "canonical_alias_group_count": 1,
            "canonical_alias_groups": ["descriptor::TPSA::down::continuousish"],
            "canonical_keys": ["descriptor|TPSA|continuous|negative|null"],
        },
        {
            "present_in_cross_source_consensus": False,
            "cross_source_overlap_sources": [],
            "cross_source_compile_valid_true_count": 0,
            "promoted_multi_candidate_present": False,
            "promoted_multi_compile_safe_present": False,
            "present_in_final_winner": False,
            "final_source_bucket": "",
            "present_in_surviving_rules": False,
            "final_rule_count": 0,
            "final_source_breakdown": {},
        },
    )
    assert row["first_missing_stage"] == "cross_source_consensus"


def test_root_can_survive_as_data_only_while_first_missing_stage_is_cross_source_consensus():
    row = build_root_comparison_row(
        "weak_run",
        "descriptor::TPSA",
        {
            "root_present_in_synth_rules": True,
            "root_present_in_canonical_rules": True,
            "root_present_in_final_synth_rules": True,
            "raw_rule_count": 1,
            "canonical_rule_count": 1,
            "canonical_source_breakdown": {"literature": 1},
            "canonical_compile_valid_count": 1,
            "canonical_compile_valid_known_count": 1,
            "canonical_alias_group_count": 1,
            "canonical_alias_groups": ["descriptor::TPSA::down::continuousish"],
            "canonical_keys": ["descriptor|TPSA|continuous|negative|null"],
        },
        {
            "present_in_cross_source_consensus": False,
            "cross_source_overlap_sources": [],
            "cross_source_compile_valid_true_count": 0,
            "promoted_multi_candidate_present": False,
            "promoted_multi_compile_safe_present": False,
            "present_in_final_winner": True,
            "final_source_bucket": "data",
            "present_in_surviving_rules": True,
            "final_rule_count": 1,
            "final_source_breakdown": {"data": 1},
        },
    )
    assert row["present_in_final_winner"] is True
    assert row["final_source_bucket"] == "data"
    assert row["first_missing_stage"] == "cross_source_consensus"
