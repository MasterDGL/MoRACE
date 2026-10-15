from morace.diagnostics.logp_ingestion import (
    _ingestion_gap_summary,
    audit_ingestion_gap,
    extract_all_gated_mollogp_objects,
    extract_consensus_group_state,
    extract_inference_selected_mollogp_objects,
    extract_synth_canonical_mollogp_objects,
)


def _rule(feature_family, feature_name, source="data", canonical_key="", *, feature_mode="continuous", relation="positive", compile_valid=True, support_sources=None):
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
    }


def test_extract_inference_selected_mollogp_objects_returns_object_rows():
    inference_artifact = {
        "stage2_projection": [
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "MolLogP", "data", feature_mode="None")]},
        ]
    }
    rows = extract_inference_selected_mollogp_objects(inference_artifact, "gap_run")
    assert len(rows) == 1
    assert rows[0]["layer"] == "inference_selected"
    assert rows[0]["source_bucket"] == "data"


def test_extract_all_gated_mollogp_objects_marks_exact_and_alias_membership():
    synth_objects = [
        {
            "run_tag": "gap_run",
            "layer": "synth_canonical",
            "canonical_key": "descriptor|MolLogP|continuous|positive|null",
            "feature_mode": "continuous",
            "relation_to_target": "positive",
            "direction_bucket": "up",
            "mode_bucket": "continuousish",
            "alias_group_key": "descriptor::MolLogP::up::continuousish",
            "source_bucket": "literature",
            "primary_source": "literature",
            "support_sources": ["literature"],
            "compile_valid": False,
        }
    ]
    inference_objects = [
        {
            "run_tag": "gap_run",
            "layer": "inference_selected",
            "canonical_key": "descriptor|MolLogP|none|positive|null",
            "feature_mode": "None",
            "relation_to_target": "positive",
            "direction_bucket": "up",
            "mode_bucket": "continuousish",
            "alias_group_key": "descriptor::MolLogP::up::continuousish",
            "source_bucket": "data",
            "primary_source": "data",
            "support_sources": ["data"],
            "compile_valid": False,
        }
    ]
    all_artifact = {
        "branch_metrics": {
            "combined": {
                "selection_trace": {
                    "gated_rules": [
                        _rule("descriptor", "MolLogP", "data", canonical_key="descriptor|MolLogP|none|positive|null", feature_mode="None"),
                    ]
                },
                "cross_source_consensus_groups": [
                    {
                        "root_feature_key": "descriptor::MolLogP",
                        "has_cross_source_overlap": False,
                        "canonical_keys": [],
                    }
                ],
            }
        }
    }
    rows = extract_all_gated_mollogp_objects(all_artifact, inference_objects, synth_objects, "gap_run")
    assert len(rows) == 1
    assert rows[0]["present_in_inference_selected_by_exact_key"] is True
    assert rows[0]["present_in_inference_selected_by_alias_group"] is True
    assert rows[0]["present_in_synth_canonical_by_exact_key"] is False
    assert rows[0]["present_in_synth_canonical_by_alias_group"] is True


def test_extract_consensus_group_state_tracks_overlap_flag():
    all_artifact = {
        "branch_metrics": {
            "combined": {
                "cross_source_consensus_groups": [
                    {
                        "root_feature_key": "descriptor::MolLogP",
                        "has_cross_source_overlap": True,
                        "canonical_keys": ["descriptor|MolLogP|none|positive|null"],
                        "shared_direction_buckets": ["positive"],
                        "shared_mode_buckets": ["null"],
                        "compile_valid_true_count": 2,
                    }
                ]
            }
        }
    }
    state = extract_consensus_group_state(all_artifact)
    assert state["consensus_group_has_cross_source_overlap"] is True
    assert state["cross_source_compile_valid_true_count"] == 2


def test_ingestion_gap_summary_counts_multi_as_data_and_literature():
    summary = _ingestion_gap_summary(
        inference_objects=[
            {
                "canonical_key": "descriptor|MolLogP|none|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::continuousish",
            }
        ],
        synth_objects=[
            {
                "canonical_key": "descriptor|MolLogP|continuous|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::continuousish",
            }
        ],
        all_gated_objects=[
            {
                "source_bucket": "multi",
                "support_sources": ["data", "literature"],
                "present_in_inference_selected_by_exact_key": True,
                "present_in_inference_selected_by_alias_group": True,
                "present_in_synth_canonical_by_exact_key": True,
                "present_in_synth_canonical_by_alias_group": True,
                "canonical_key": "descriptor|MolLogP|none|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::continuousish",
            }
        ],
        consensus_state={"consensus_group_has_cross_source_overlap": True},
    )
    assert summary["gated_has_data"] is True
    assert summary["gated_has_literature"] is True


def test_ingestion_gap_summary_marks_synth_input_ingestion_gap():
    summary = _ingestion_gap_summary(
        inference_objects=[
            {
                "canonical_key": "descriptor|MolLogP|none|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::continuousish",
            }
        ],
        synth_objects=[
            {
                "canonical_key": "descriptor|MolLogP|continuous|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::continuousish",
            }
        ],
        all_gated_objects=[
            {
                "source_bucket": "data",
                "support_sources": ["data"],
                "present_in_inference_selected_by_exact_key": True,
                "present_in_inference_selected_by_alias_group": True,
                "present_in_synth_canonical_by_exact_key": False,
                "present_in_synth_canonical_by_alias_group": False,
                "canonical_key": "descriptor|MolLogP|none|positive|null",
                "alias_group_key": "descriptor::MolLogP::up::null_bucket",
            }
        ],
        consensus_state={"consensus_group_has_cross_source_overlap": False},
    )
    assert summary["primary_gap_reason"] == "synth_input_ingestion_gap"
