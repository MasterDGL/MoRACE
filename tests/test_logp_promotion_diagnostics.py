from morace.diagnostics.logp_promotion import (
    TARGET_ROOT,
    build_mollogp_gap_row,
    compute_first_missing_stage,
    extract_mollogp_state_from_all,
    extract_mollogp_state_from_inference,
)


def _rule(feature_family, feature_name, source="data", canonical_key="", *, feature_mode="continuous", relation="positive", compile_valid=True, consensus_promoted=False, support_sources=None):
    return {
        "feature_family": feature_family,
        "feature_name": feature_name,
        "source": source,
        "primary_source": source,
        "support_sources": support_sources or [source],
        "canonical_key": canonical_key or f"{feature_family}|{feature_name}|{feature_mode}|{relation}|null",
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "compile_valid": compile_valid,
        "consensus_promoted": consensus_promoted,
    }


def test_extract_mollogp_state_from_inference_tracks_selected_fallback():
    artifact = {
        "rules": [_rule("descriptor", "MolLogP", "data", compile_valid=False)],
        "canonical_rules": [_rule("descriptor", "MolLogP", "data", compile_valid=True)],
        "stage2_projection": [
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "MolLogP", "data", compile_valid=True)]},
        ],
    }
    state = extract_mollogp_state_from_inference(artifact)
    assert state["present_in_inference_raw_rules"] is True
    assert state["present_in_inference_canonical_rules"] is True
    assert state["present_in_inference_compile_safe_rules"] is True
    assert state["present_in_inference_selected_rules"] is True
    assert state["inference_source_bucket"] == "data"


def test_extract_mollogp_state_from_all_marks_cross_source_gap_before_final_data_only():
    all_artifact = {
        "branch_metrics": {
            "combined": {
                "selection_trace": {
                    "gated_rules": [
                        _rule("descriptor", "MolLogP", "data", support_sources=["data"]),
                    ]
                },
                "cross_source_consensus_groups": [
                    {
                        "root_feature_key": TARGET_ROOT,
                        "has_cross_source_overlap": False,
                        "compile_valid_true_count": 1,
                    }
                ],
            }
        },
        "rules": [
            _rule("descriptor", "MolLogP", "data", support_sources=["data"]),
        ],
    }
    state = extract_mollogp_state_from_all(all_artifact)
    row = build_mollogp_gap_row(
        "gap_run",
        {
            "present_in_inference_raw_rules": True,
            "present_in_inference_canonical_rules": True,
            "present_in_inference_compile_safe_rules": True,
            "present_in_inference_selected_rules": True,
            "inference_source_bucket": "data",
            "inference_canonical_keys": [],
            "inference_alias_group_keys": [],
        },
        {
            "present_in_synth_raw_rules": True,
            "present_in_synth_canonical_rules": True,
            "present_in_synth_compile_safe_rules": True,
            "synth_source_bucket": "literature",
            "synth_canonical_keys": [],
            "synth_alias_group_keys": [],
        },
        state,
    )
    assert row["present_in_final_winner"] is True
    assert row["final_source_bucket"] == "data"
    assert row["first_missing_stage"] == "cross_source_consensus"
    assert row["suggested_next_route"] == "cross_source_matching_refinement"


def test_compute_first_missing_stage_marks_all_input_gap():
    stage = compute_first_missing_stage(
        {
            "present_in_inference_selected_rules": True,
            "present_in_synth_raw_rules": True,
            "present_in_synth_canonical_rules": True,
            "present_in_all_input_rules": False,
            "present_in_cross_source_consensus": False,
            "promoted_multi_candidate_present": False,
            "promoted_multi_compile_safe_present": False,
            "present_in_final_winner": False,
            "present_in_surviving_rules": False,
        }
    )
    assert stage == "all_input"


def test_compute_first_missing_stage_marks_promotion_compile_safe_gap():
    stage = compute_first_missing_stage(
        {
            "present_in_inference_selected_rules": True,
            "present_in_synth_raw_rules": True,
            "present_in_synth_canonical_rules": True,
            "present_in_all_input_rules": True,
            "present_in_cross_source_consensus": True,
            "promoted_multi_candidate_present": True,
            "promoted_multi_compile_safe_present": False,
            "present_in_final_winner": False,
            "present_in_surviving_rules": False,
        }
    )
    assert stage == "promotion_compile_safe"
