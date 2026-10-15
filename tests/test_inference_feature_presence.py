from morace.diagnostics.inference_feature_presence import (
    build_bbbp_inference_mollogp_stability_cue,
    build_root_presence_row,
    extract_root_presence_from_inference_artifact,
    summarize_root_presence_frequency,
)


def _rule(feature_family, feature_name, source="data", *, compile_valid=True, feature_mode="continuous", relation="positive"):
    return {
        "feature_family": feature_family,
        "feature_name": feature_name,
        "source": source,
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "compile_valid": compile_valid,
    }


def test_bbbp_inference_mollogp_stability_cue_only_enables_for_bbbp():
    enabled = build_bbbp_inference_mollogp_stability_cue(dataset="bbbp", enabled=True)
    disabled = build_bbbp_inference_mollogp_stability_cue(dataset="bace", enabled=True)
    disabled_flag = build_bbbp_inference_mollogp_stability_cue(dataset="bbbp", enabled=False)
    assert "MolLogP" in enabled
    assert disabled == ""
    assert disabled_flag == ""


def test_extract_root_presence_from_inference_artifact_tracks_raw_canonical_compile_safe_and_selected():
    artifact = {
        "stage1_drafts": [{"response_text": "MolLogP and TPSA are relevant here."}],
        "rules": [
            _rule("descriptor", "MolLogP", compile_valid=False),
            _rule("descriptor", "TPSA", compile_valid=False),
        ],
        "canonical_rules": [
            _rule("descriptor", "MolLogP", compile_valid=True),
            _rule("descriptor", "TPSA", compile_valid=False),
            _rule("count", "NumHBA", compile_valid=True, feature_mode="count_value"),
        ],
        "stage2_projection": [
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "MolLogP", compile_valid=True)]},
            {"selected_for_merge": False, "parsed_rules": [_rule("descriptor", "TPSA", compile_valid=False)]},
        ],
    }
    presence = extract_root_presence_from_inference_artifact(artifact)
    assert "descriptor::MolLogP" in presence["raw_root_keys"]
    assert "descriptor::MolLogP" in presence["compile_safe_root_keys"]
    assert "descriptor::MolLogP" in presence["selected_root_keys"]
    assert presence["root_presence_summary"]["descriptor::MolLogP"]["present_in_selected_rules"] is True
    assert presence["root_presence_summary"]["descriptor::TPSA"]["present_in_compile_safe_rules"] is False
    assert presence["raw_root_mentions"]["descriptor::MolLogP"]["mentioned"] is True


def test_raw_root_mentions_fallback_and_summary_rates():
    artifact = {
        "stage1_drafts": [
            {"text": "Lipophilicity via logP and rotatable bond count may matter."},
        ],
        "rules": [_rule("descriptor", "MolLogP", compile_valid=True)],
        "canonical_rules": [_rule("descriptor", "MolLogP", compile_valid=True)],
        "stage2_projection": [
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "MolLogP", compile_valid=True)]},
            {"selected_for_merge": True, "parsed_rules": [_rule("descriptor", "TPSA", compile_valid=True)]},
        ],
    }
    presence = extract_root_presence_from_inference_artifact(artifact)
    row = build_root_presence_row("run_01", presence)
    summary = summarize_root_presence_frequency(
        [
            dict(row),
            {
                "MolLogP_present_in_raw": False,
                "MolLogP_present_in_canonical": False,
                "MolLogP_present_in_compile_safe": False,
                "MolLogP_present_in_selected": False,
                "TPSA_present_in_selected": True,
            },
        ]
    )
    assert presence["raw_root_mentions"]["descriptor::MolLogP"]["mentioned"] is True
    assert row["TPSA_present_in_raw"] is False
    assert row["TPSA_present_in_selected"] is True
    assert summary["MolLogP_selected_presence_rate"] == 0.5
    assert summary["TPSA_raw_presence_rate"] == 0.0
    assert summary["TPSA_selected_presence_rate"] == 1.0
