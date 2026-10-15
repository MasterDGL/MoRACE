from morace.diagnostics.synthesis_feature_presence import (
    build_bbbp_literature_stability_cue,
    extract_root_presence_from_synth_artifact,
    summarize_root_presence_frequency,
)


def _rule(feature_family, feature_name, source="literature", *, compile_valid=True, feature_mode="continuous", relation="negative"):
    return {
        "feature_family": feature_family,
        "feature_name": feature_name,
        "source": source,
        "feature_mode": feature_mode,
        "relation_to_target": relation,
        "compile_valid": compile_valid,
    }


def test_bbbp_literature_stability_cue_only_enables_for_bbbp_literature():
    enabled = build_bbbp_literature_stability_cue(dataset="bbbp", source="literature", enabled=True)
    disabled_dataset = build_bbbp_literature_stability_cue(dataset="bace", source="literature", enabled=True)
    disabled_source = build_bbbp_literature_stability_cue(dataset="bbbp", source="data", enabled=True)
    assert "TPSA" in enabled
    assert disabled_dataset == ""
    assert disabled_source == ""


def test_extract_root_presence_from_synth_artifact_tracks_roots_and_mentions():
    artifact = {
        "stage1_drafts": [
            {"draft_text": "TPSA and lipophilicity both matter for BBBP."},
        ],
        "rules": [
            _rule("descriptor", "TPSA", compile_valid=False),
            _rule("descriptor", "MolLogP", compile_valid=True),
        ],
        "canonical_rules": [
            _rule("descriptor", "TPSA", compile_valid=False),
            _rule("descriptor", "MolLogP", compile_valid=True),
            _rule("count", "NumHBD", compile_valid=True, feature_mode="count_value"),
        ],
    }
    presence = extract_root_presence_from_synth_artifact(artifact)
    assert "descriptor::TPSA" in presence["raw_root_keys"]
    assert "descriptor::MolLogP" in presence["compile_safe_root_keys"]
    assert presence["root_presence_summary"]["descriptor::TPSA"]["present_in_raw_rules"] is True
    assert presence["root_presence_summary"]["descriptor::TPSA"]["present_in_compile_safe_rules"] is False
    assert presence["raw_root_mentions"]["descriptor::TPSA"]["mentioned"] is True
    assert presence["raw_root_mentions"]["descriptor::MolLogP"]["mentioned"] is True


def test_summarize_root_presence_frequency_computes_rates():
    summary = summarize_root_presence_frequency(
        [
            {
                "TPSA_present_in_raw": True,
                "TPSA_present_in_canonical": False,
                "MolLogP_present_in_raw": True,
                "MolLogP_present_in_canonical": True,
                "NumHBD_present_in_raw": False,
                "NumHBD_present_in_canonical": False,
            },
            {
                "TPSA_present_in_raw": False,
                "TPSA_present_in_canonical": False,
                "MolLogP_present_in_raw": True,
                "MolLogP_present_in_canonical": False,
                "NumHBD_present_in_raw": True,
                "NumHBD_present_in_canonical": True,
            },
        ]
    )
    assert summary["TPSA_raw_presence_rate"] == 0.5
    assert summary["MolLogP_raw_presence_rate"] == 1.0
    assert summary["NumHBD_canonical_presence_rate"] == 0.5
