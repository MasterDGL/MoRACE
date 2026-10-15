from morace.diagnostics.logp_merge import _gated_row_with_match_counts, _merge_bookkeeping_summary


def test_gated_row_detects_literature_overwrite_of_data_object():
    row = {
        "canonical_key": "descriptor|MolLogP|continuous|positive|null",
        "alias_group_key": "descriptor::MolLogP::up::continuousish",
        "source_bucket": "literature",
        "support_sources": ["literature"],
    }
    inference_objects = [
        {
            "canonical_key": "descriptor|MolLogP|continuous|positive|null",
            "alias_group_key": "descriptor::MolLogP::up::continuousish",
        }
    ]
    synth_objects = [
        {
            "canonical_key": "descriptor|MolLogP|continuous|positive|null",
            "alias_group_key": "descriptor::MolLogP::up::continuousish",
        }
    ]
    enriched = _gated_row_with_match_counts(
        row,
        inference_objects,
        synth_objects,
        {"consensus_group_has_cross_source_overlap": False},
    )
    assert enriched["matched_inference_objects_by_exact_key_count"] == 1
    assert enriched["matched_synth_objects_by_exact_key_count"] == 1
    assert enriched["looks_like_literature_overwrite_of_data_object"] is True


def test_merge_bookkeeping_summary_marks_merge_retention_gap_when_no_data_side_object_is_retained():
    summary = _merge_bookkeeping_summary(
        inference_objects=[
            {"canonical_key": "descriptor|MolLogP|none|positive|null", "alias_group_key": "descriptor::MolLogP::up::continuousish"}
        ],
        synth_objects=[
            {"canonical_key": "descriptor|MolLogP|continuous|positive|null", "alias_group_key": "descriptor::MolLogP::up::continuousish"}
        ],
        gated_rows=[
            {
                "source_bucket": "literature",
                "support_sources": ["literature"],
                "matched_inference_objects_by_exact_key_count": 1,
                "matched_inference_objects_by_alias_group_count": 1,
                "matched_synth_objects_by_exact_key_count": 1,
                "matched_synth_objects_by_alias_group_count": 1,
                "looks_like_merged_dual_source_object": False,
                "looks_like_literature_overwrite_of_data_object": True,
            }
        ],
        consensus_state={"consensus_group_has_cross_source_overlap": False},
    )
    assert summary["data_side_object_retained"] is False
    assert summary["primary_gap_reason"] == "merge_retention_gap"


def test_merge_bookkeeping_summary_marks_present_when_dual_source_object_is_retained():
    summary = _merge_bookkeeping_summary(
        inference_objects=[
            {"canonical_key": "descriptor|MolLogP|none|positive|null", "alias_group_key": "descriptor::MolLogP::up::continuousish"}
        ],
        synth_objects=[
            {"canonical_key": "descriptor|MolLogP|continuous|positive|null", "alias_group_key": "descriptor::MolLogP::up::continuousish"}
        ],
        gated_rows=[
            {
                "source_bucket": "multi",
                "support_sources": ["data", "literature"],
                "matched_inference_objects_by_exact_key_count": 0,
                "matched_inference_objects_by_alias_group_count": 1,
                "matched_synth_objects_by_exact_key_count": 1,
                "matched_synth_objects_by_alias_group_count": 1,
                "looks_like_merged_dual_source_object": True,
                "looks_like_literature_overwrite_of_data_object": False,
            }
        ],
        consensus_state={"consensus_group_has_cross_source_overlap": True},
    )
    assert summary["gated_has_multi"] is True
    assert summary["dual_source_object_retained"] is True
    assert summary["primary_gap_reason"] == "present"
