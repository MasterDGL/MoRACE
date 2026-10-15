from morace.diagnostics.logp_matching import (
    _build_empty_pair_row,
    audit_matching_diff,
    build_matching_pair_row,
    extract_mollogp_all_consensus_state,
    extract_mollogp_inference_selected_rules,
    extract_mollogp_synth_canonical_rules,
)


def _rule(feature_family, feature_name, source="data", canonical_key="", *, feature_mode="continuous", relation="positive", compile_valid=True, support_sources=None):
    return {
        "feature_family": feature_family,
        "feature_name": feature_name,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources or [source],
        "canonical_key": canonical_key,
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "compile_valid": compile_valid,
    }


def test_extractors_capture_synth_and_inference_mollogp_rules():
    synth = {
        "canonical_rules": [
            _rule("descriptor", "MolLogP", "literature", canonical_key="descriptor|MolLogP|continuous|positive|null"),
        ]
    }
    inference = {
        "stage2_projection": [
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "MolLogP", "data", feature_mode="None", relation="positive")]},
        ]
    }
    synth_rows = extract_mollogp_synth_canonical_rules(synth)
    inf_rows = extract_mollogp_inference_selected_rules(inference)
    assert synth_rows[0]["direction_bucket"] == "up"
    assert synth_rows[0]["mode_bucket"] == "continuousish"
    assert inf_rows[0]["direction_bucket"] == "up"
    assert inf_rows[0]["mode_bucket"] == "continuousish"


def test_build_matching_pair_row_detects_mode_gap_and_all_input_miss():
    synth_rule = {
        "canonical_key": "descriptor|MolLogP|continuous|positive|null",
        "feature_mode": "continuous",
        "relation_to_target": "positive",
        "direction_bucket": "up",
        "mode_bucket": "continuousish",
        "alias_group_key": "descriptor::MolLogP::up::continuousish",
        "compile_valid": True,
        "source_bucket": "literature",
    }
    inference_rule = {
        "canonical_key": "descriptor|MolLogP|monotonic_up|positive|null",
        "feature_mode": "monotonic_up",
        "relation_to_target": "positive",
        "direction_bucket": "up",
        "mode_bucket": "monotonic_up",
        "alias_group_key": "descriptor::MolLogP::up::monotonic_up",
        "compile_valid": False,
        "source_bucket": "data",
    }
    row = build_matching_pair_row(
        "gap_run",
        synth_rule,
        inference_rule,
        {
            "present_in_cross_source_consensus": False,
            "shared_direction_buckets": [],
            "shared_mode_buckets": [],
            "cross_source_compile_valid_true_count": 0,
            "consensus_canonical_keys": [],
            "all_input_canonical_keys": ["descriptor|MolLogP|continuous|positive|null"],
            "all_input_source_buckets": ["literature"],
            "all_input_alias_group_keys": ["descriptor::MolLogP::up::continuousish"],
        },
    )
    assert row["direction_match"] is True
    assert row["mode_match"] is False
    assert row["alias_match"] is False
    assert row["inference_selected_key_present_in_all_input"] is False


def test_extract_mollogp_all_consensus_state_tracks_consensus_fields():
    all_artifact = {
        "branch_metrics": {
            "combined": {
                "selection_trace": {
                    "gated_rules": [
                        _rule("descriptor", "MolLogP", "literature", canonical_key="descriptor|MolLogP|continuous|positive|null", support_sources=["literature"]),
                        _rule("descriptor", "MolLogP", "data", canonical_key="descriptor|MolLogP|null|positive|null", support_sources=["data"]),
                    ]
                },
                "cross_source_consensus_groups": [
                    {
                        "root_feature_key": "descriptor::MolLogP",
                        "has_cross_source_overlap": True,
                        "shared_direction_buckets": ["positive"],
                        "shared_mode_buckets": ["null"],
                        "compile_valid_true_count": 2,
                        "canonical_keys": [
                            "descriptor|MolLogP|continuous|positive|null",
                            "descriptor|MolLogP|null|positive|null",
                        ],
                    }
                ],
            }
        }
    }
    state = extract_mollogp_all_consensus_state(all_artifact)
    assert state["present_in_cross_source_consensus"] is True
    assert state["cross_source_compile_valid_true_count"] == 2
    assert len(state["all_input_canonical_keys"]) == 2


def test_build_empty_pair_row_marks_missing_side_reason():
    row = _build_empty_pair_row(
        "gap_run",
        {
            "present_in_cross_source_consensus": False,
            "shared_direction_buckets": [],
            "shared_mode_buckets": [],
            "cross_source_compile_valid_true_count": 0,
            "consensus_canonical_keys": [],
            "all_input_canonical_keys": [],
            "all_input_source_buckets": [],
            "all_input_alias_group_keys": [],
        },
        reason="no_synth_pair_candidate",
    )
    assert row["matching_gap_reason"] == "no_synth_pair_candidate"
    assert row["synth_canonical_key"] == ""
    assert row["inference_canonical_key"] == ""
