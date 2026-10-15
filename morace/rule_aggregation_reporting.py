"""Selection traces and metric reports for rule aggregation."""

from typing import Dict, List


def build_selection_trace(
    rule_list: List[Dict],
    enriched: List[Dict],
    selected: List[Dict],
    selection_audit: Dict | None = None,
    *,
    passes_rule_gate_fn,
    alias_group_summary_fn,
    root_feature_summary_fn,
) -> Dict:
    gated = [item for item in enriched if passes_rule_gate_fn(item)]
    return {
        "raw_rules": rule_list,
        "canonical_rules": enriched,
        "gated_rules": gated,
        "selected_rules": selected,
        "alias_group_summary": {
            "raw": alias_group_summary_fn(rule_list),
            "canonical": alias_group_summary_fn(enriched),
            "gated": alias_group_summary_fn(gated),
            "selected": alias_group_summary_fn(selected),
        },
        "root_feature_summary": {
            "raw": root_feature_summary_fn(rule_list),
            "canonical": root_feature_summary_fn(enriched),
            "gated": root_feature_summary_fn(gated),
            "selected": root_feature_summary_fn(selected),
        },
        "selection_audit": selection_audit or {"dropped": []},
    }


def build_base_metrics(
    rule_list: List[Dict],
    merged: List[Dict],
    enriched: List[Dict],
    selected: List[Dict],
    target_rule_count: int,
    selection_audit: Dict | None = None,
    *,
    passes_rule_gate_fn,
    alias_group_count_fn,
    alias_compaction_score_fn,
    family_breakdown_fn,
    source_breakdown_fn,
    dominant_feature_share_fn,
    selection_trace_fn,
    compile_safe_multi_feasibility_metrics_fn,
) -> Dict:
    return {
        "raw_rule_count": len(rule_list),
        "canonical_rule_count": len(merged),
        "gated_rule_count": len([item for item in enriched if passes_rule_gate_fn(item)]),
        "selected_rule_count": len(selected),
        "compile_valid_rate": (
            float(sum(1 for item in enriched if item.get("compile_valid")) / len(enriched))
            if enriched
            else 0.0
        ),
        "selected_avg_support": (
            float(sum(item.get("support_count", 0) for item in selected) / len(selected))
            if selected
            else 0.0
        ),
        "selected_avg_p_value": (
            float(sum(item.get("p_value", 1.0) for item in selected) / len(selected))
            if selected
            else 1.0
        ),
        "selected_avg_effect_size": (
            float(
                sum(abs(float(item.get("effect_size", 0.0))) for item in selected)
                / len(selected)
            )
            if selected
            else 0.0
        ),
        "alias_group_count": alias_group_count_fn(enriched),
        "alias_compaction_score": alias_compaction_score_fn(enriched),
        "family_breakdown": family_breakdown_fn(selected),
        "source_breakdown": source_breakdown_fn(selected),
        "dominant_feature_share": dominant_feature_share_fn(selected),
        "mollogp_collapse_rate": (
            float(
                sum(1 for item in selected if item.get("feature_name") == "MolLogP")
                / max(1, len(selected))
            )
            if selected
            else 0.0
        ),
        "selection_trace": selection_trace_fn(
            rule_list,
            enriched,
            selected,
            selection_audit=selection_audit,
        ),
        **compile_safe_multi_feasibility_metrics_fn(
            merged,
            enriched,
            target_rule_count=target_rule_count,
        ),
    }


def candidate_rule_set_summary(
    name: str,
    selected: List[Dict],
    selection_audit: Dict | None = None,
    *,
    template_type_from_name_fn,
    alias_group_key_fn,
    alias_group_count_fn,
    alias_compaction_score_fn,
    root_feature_key_fn,
    family_breakdown_fn,
    source_breakdown_fn,
    effective_source_coverage_fn,
    direction_bucket_fn,
    mode_bucket_fn,
) -> Dict:
    selection_audit = selection_audit or {"dropped": []}
    selected_source_breakdown = source_breakdown_fn(selected)
    selected_family_breakdown = family_breakdown_fn(selected)
    effective_source_coverage = effective_source_coverage_fn(selected_source_breakdown)
    root_count = len(
        {
            root_feature_key_fn(item)
            for item in selected
            if root_feature_key_fn(item)
        }
    )
    family_diversity = sum(1 for value in selected_family_breakdown.values() if value > 0)
    non_descriptor_count = sum(
        1
        for item in selected
        if str(item.get("feature_family", "")) in {"count", "pattern_presence"}
    )
    second_variant_quality_score = float(
        selection_audit.get("second_variant_quality_score", 0.0) or 0.0
    )
    return {
        "name": name,
        "template_type": template_type_from_name_fn(name),
        "rules": selected,
        "selection_audit": selection_audit,
        "alias_group_signature": sorted(
            {alias_group_key_fn(item) for item in selected if alias_group_key_fn(item)}
        ),
        "selected_rule_count": len(selected),
        "compiled_rule_count": sum(1 for item in selected if item.get("compile_valid")),
        "selected_root_feature_count": root_count,
        "selected_alias_group_count": alias_group_count_fn(selected),
        "alias_compaction_score": alias_compaction_score_fn(selected),
        "selected_family_breakdown": selected_family_breakdown,
        "selected_source_breakdown": selected_source_breakdown,
        "effective_literature_coverage": effective_source_coverage["literature"],
        "effective_data_coverage": effective_source_coverage["data"],
        "family_diversity": family_diversity,
        "selected_non_descriptor_count": non_descriptor_count,
        "non_descriptor_count": non_descriptor_count,
        "mean_support": (
            float(sum(item.get("support_count", 0) for item in selected) / len(selected))
            if selected
            else 0.0
        ),
        "selected_avg_support": (
            float(sum(item.get("support_count", 0) for item in selected) / len(selected))
            if selected
            else 0.0
        ),
        "selected_avg_effect_size": (
            float(
                sum(abs(float(item.get("effect_size", 0.0))) for item in selected)
                / len(selected)
            )
            if selected
            else 0.0
        ),
        "selected_avg_p_value": (
            float(sum(item.get("p_value", 1.0) for item in selected) / len(selected))
            if selected
            else 1.0
        ),
        "selected_second_variant_count": int(
            selection_audit.get("selected_second_variant_count", 0)
        ),
        "selected_same_root_alias_pairs": list(
            selection_audit.get("selected_same_root_alias_pairs", [])
        ),
        "selected_cross_source_second_variant_count": int(
            selection_audit.get("selected_cross_source_second_variant_count", 0)
        ),
        "selected_repeated_root_count": int(
            selection_audit.get("selected_repeated_root_count", 0)
        ),
        "second_variant_quality_score": second_variant_quality_score,
        "capacity_release_mode": str(
            selection_audit.get("capacity_release_mode", "strict_root_cap_1")
        ),
        "diagnostic_alias_summary": {
            "alias_group_count": alias_group_count_fn(selected),
            "direction_buckets": {
                bucket: sum(
                    1
                    for item in selected
                    if direction_bucket_fn(item.get("relation_to_target", "")) == bucket
                )
                for bucket in sorted(
                    {
                        direction_bucket_fn(item.get("relation_to_target", ""))
                        for item in selected
                    }
                )
            },
            "mode_buckets": {
                bucket: sum(
                    1
                    for item in selected
                    if mode_bucket_fn(item.get("feature_mode", "")) == bucket
                )
                for bucket in sorted(
                    {mode_bucket_fn(item.get("feature_mode", "")) for item in selected}
                )
            },
        },
    }
