from morace.ablation import build_ablation_a, build_ablation_b, build_ablation_c


def _rule(
    feature_name,
    source,
    canonical_key,
    *,
    family="descriptor",
    support_count=1,
    p_value=0.01,
    effect_size=1.0,
    support_sources=None,
    consensus_promoted=False,
    promotion_lineage=None,
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
        "support_count": support_count,
        "p_value": p_value,
        "effect_size": effect_size,
        "compile_valid": True,
        "consensus_promoted": consensus_promoted,
        "promotion_lineage": promotion_lineage,
    }


def test_ablation_a_replaces_promoted_multi_with_best_parent():
    promoted = _rule(
        "TPSA",
        "multi",
        "descriptor|TPSA|null|negative|null",
        support_count=10,
        support_sources=["data", "literature"],
        consensus_promoted=True,
        promotion_lineage={
            "literature_canonical_key": "descriptor|TPSA|continuous|negative|null",
            "data_canonical_key": "descriptor|TPSA|null|negative|null",
            "primary_canonical_key": "descriptor|TPSA|null|negative|null",
            "secondary_canonical_key": "descriptor|TPSA|continuous|negative|null",
        },
    )
    anchor = [
        promoted,
        _rule("MolWt", "data", "descriptor|MolWt|null|negative|null", support_count=5),
    ]
    pool = [
        _rule("TPSA", "literature", "descriptor|TPSA|continuous|negative|null", support_count=1, p_value=0.2),
        _rule("TPSA", "data", "descriptor|TPSA|null|negative|null", support_count=12, p_value=0.001),
        _rule("MolWt", "data", "descriptor|MolWt|null|negative|null", support_count=5),
    ]
    rules, audit = build_ablation_a(anchor, pool)
    kept_keys = {rule["canonical_key"] for rule in rules}
    assert "descriptor|TPSA|null|negative|null" in kept_keys
    assert all(rule.get("source") != "multi" for rule in rules)
    assert audit["replacement_pairs"][0]["replacement_canonical_key"] == "descriptor|TPSA|null|negative|null"
    assert audit["skipped_pairs"] == []


def test_ablation_b_replaces_literature_only_with_best_new_root_data():
    anchor = [
        _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"]),
        _rule("MolWt", "data", "descriptor|MolWt|null|negative|null", support_count=7),
        _rule("NumHBD", "literature", "count|NumHBD|count_value|negative|null", family="count"),
    ]
    pool = [
        _rule("NumHBA", "data", "count|NumHBA|count_value|negative|null", family="count", support_count=9),
        _rule("MolWt", "data", "descriptor|MolWt|continuous|negative|null", support_count=20),
    ]
    rules, audit = build_ablation_b(anchor, pool)
    kept_keys = {rule["canonical_key"] for rule in rules}
    assert "count|NumHBD|count_value|negative|null" not in kept_keys
    assert "count|NumHBA|count_value|negative|null" in kept_keys
    assert audit["replacement_status"] == "ok"


def test_ablation_c_prefers_new_root_for_plus_one_slot():
    anchor = [
        _rule("TPSA", "multi", "descriptor|TPSA|null|negative|null", support_sources=["data", "literature"]),
        _rule("MolWt", "data", "descriptor|MolWt|null|negative|null", support_count=7),
    ]
    pool = [
        _rule("MolWt", "data", "descriptor|MolWt|continuous|negative|null", support_count=20),
        _rule("NumRotatableBonds", "data", "count|NumRotatableBonds|count_value|negative|null", family="count", support_count=3),
    ]
    rules, audit = build_ablation_c(anchor, pool)
    assert len(rules) == 3
    assert audit["addition_status"] == "ok"
    assert audit["added_root_feature_key"] == "count::NumRotatableBonds"


def test_ablation_a_records_skipped_when_parent_missing():
    promoted = _rule(
        "MolLogP",
        "multi",
        "descriptor|MolLogP|null|positive|null",
        support_sources=["data", "literature"],
        consensus_promoted=True,
        promotion_lineage={
            "literature_canonical_key": "descriptor|MolLogP|continuous|positive|null",
            "data_canonical_key": "descriptor|MolLogP|null|positive|null",
        },
    )
    rules, audit = build_ablation_a([promoted], [])
    assert len(rules) == 1
    assert audit["replacement_pairs"] == []
    assert audit["skipped_pairs"][0]["reason"] == "no_parent_candidate"
