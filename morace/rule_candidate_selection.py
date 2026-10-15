"""Deterministic candidate ordering helpers for rule aggregation."""

from typing import Dict, List, Tuple


def candidate_quality_key(rule: Dict) -> Tuple:
    return (
        0 if str(rule.get("extraction_source", "projected")) == "projected" else 1,
        float(rule.get("p_value", 1.0)),
        -abs(float(rule.get("effect_size", 0.0))),
        -float(rule.get("non_nan_coverage", 0.0)),
        -int(rule.get("support_count", 0) or 0),
        str(rule.get("canonical_key", "")),
    )


def template_type_from_name(name: str) -> str:
    mapping = {
        "M1_top_global": "baseline_global",
        "M2_branch_preserving": "baseline_branch_preserving",
        "M3_literature_anchor_data_consensus": "literature_anchor_data_consensus",
        "M4_data_heavy_alias": "data_heavy",
        "M5_non_descriptor_structure": "non_descriptor_structure",
        "M6_descriptor_capped_balanced": "descriptor_capped_balanced",
        "M7_lit_preserving_swap": "lit_preserving_swap",
    }
    return mapping.get(name, name.lower())


def source_priority_rank(rule: Dict, source_priority: Tuple[str, ...], *, source_bucket_fn) -> int:
    return (
        source_priority.index(source_bucket_fn(rule))
        if source_bucket_fn(rule) in source_priority
        else len(source_priority)
    )


def alias_representative_sort_key(
    rule: Dict,
    source_priority: Tuple[str, ...],
    prefer_non_descriptor: bool = False,
    *,
    source_priority_rank_fn,
) -> Tuple:
    family = str(rule.get("feature_family", ""))
    non_descriptor_rank = 0 if prefer_non_descriptor and family in {"count", "pattern_presence"} else 1
    return (
        -int(rule.get("support_count", 0) or 0),
        float(rule.get("p_value", 1.0)),
        -abs(float(rule.get("effect_size", 0.0))),
        0 if bool(rule.get("compile_valid", False)) else 1,
        source_priority_rank_fn(rule, source_priority),
        non_descriptor_rank,
        str(rule.get("canonical_key", "")),
    )


def select_alias_representatives(
    rules: List[Dict],
    source_priority: Tuple[str, ...],
    prefer_non_descriptor: bool = False,
    *,
    alias_group_key_fn,
    alias_representative_sort_key_fn,
) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for item in rules:
        grouped.setdefault(alias_group_key_fn(item), []).append(item)
    representatives: List[Dict] = []
    for items in grouped.values():
        winner = sorted(
            items,
            key=lambda item: alias_representative_sort_key_fn(
                item,
                source_priority=source_priority,
                prefer_non_descriptor=prefer_non_descriptor,
            ),
        )[0]
        representatives.append(winner)
    return sorted(
        representatives,
        key=lambda item: alias_representative_sort_key_fn(
            item,
            source_priority=source_priority,
            prefer_non_descriptor=prefer_non_descriptor,
        ),
    )


def family_matches(rule: Dict, family_group: str) -> bool:
    family = str(rule.get("feature_family", ""))
    if family_group == "descriptor":
        return family == "descriptor"
    if family_group == "non_descriptor":
        return family in {"count", "pattern_presence"}
    return family == family_group


def lit_reserve_sort_key(
    rule: Dict,
    anchor_family_breakdown: Dict[str, int],
    *,
    source_bucket_fn,
) -> Tuple:
    family = str(rule.get("feature_family", ""))
    family_penalty = int(anchor_family_breakdown.get(family, 0))
    source_rank = 0 if source_bucket_fn(rule) == "multi" else 1
    return (
        source_rank,
        family_penalty,
        0 if bool(rule.get("compile_valid", False)) else 1,
        -int(rule.get("support_count", 0) or 0),
        float(rule.get("p_value", 1.0) or 1.0),
        -abs(float(rule.get("effect_size", 0.0) or 0.0)),
        str(rule.get("canonical_key", "")),
    )


def replaceable_anchor_sort_key(
    rule: Dict,
    anchor_family_breakdown: Dict[str, int],
    *,
    source_bucket_fn,
) -> Tuple:
    family = str(rule.get("feature_family", ""))
    source = source_bucket_fn(rule)
    return (
        0 if source != "literature" else 1,
        -int(anchor_family_breakdown.get(family, 0)),
        int(rule.get("support_count", 0) or 0),
        abs(float(rule.get("effect_size", 0.0) or 0.0)),
        -float(rule.get("p_value", 1.0) or 1.0),
        str(rule.get("canonical_key", "")),
    )


def has_literature_evidence(rule: Dict, *, source_bucket_fn) -> bool:
    support_sources = {str(item) for item in (rule.get("support_sources") or []) if item}
    if "literature" in support_sources:
        return True
    return source_bucket_fn(rule) in {"literature", "multi"}


def build_lit_preserving_swap_candidate(
    anchor_candidate: Dict,
    gated_rules: List[Dict],
    target_rule_count: int,
    *,
    source_breakdown_fn,
    effective_source_coverage_fn,
    root_feature_key_fn,
    alias_group_key_fn,
    family_breakdown_fn,
    has_literature_evidence_fn,
    lit_reserve_sort_key_fn,
    source_bucket_fn,
    replaceable_anchor_sort_key_fn,
) -> Tuple[List[Dict], Dict] | None:
    anchor_rules = list(anchor_candidate.get("rules", []) or [])
    if not anchor_rules:
        return None
    anchor_source_breakdown = source_breakdown_fn(anchor_rules)
    anchor_effective = effective_source_coverage_fn(anchor_source_breakdown)
    if anchor_effective["literature"] >= 2:
        return None

    anchor_root_keys = {root_feature_key_fn(rule) for rule in anchor_rules if root_feature_key_fn(rule)}
    anchor_canonical_keys = {str(rule.get("canonical_key", "")) for rule in anchor_rules}
    anchor_signature = tuple(
        sorted(alias_group_key_fn(rule) for rule in anchor_rules if alias_group_key_fn(rule))
    )
    anchor_family_breakdown = family_breakdown_fn(anchor_rules)

    reserve_lit_pool = [
        rule
        for rule in gated_rules
        if has_literature_evidence_fn(rule)
        and bool(rule.get("compile_valid", False))
        and root_feature_key_fn(rule) not in anchor_root_keys
        and str(rule.get("canonical_key", "")) not in anchor_canonical_keys
    ]
    if not reserve_lit_pool:
        return None

    reserve_lit_pool = sorted(
        reserve_lit_pool,
        key=lambda item: lit_reserve_sort_key_fn(item, anchor_family_breakdown),
    )
    replaceable = sorted(
        [rule for rule in anchor_rules if source_bucket_fn(rule) != "literature"],
        key=lambda item: replaceable_anchor_sort_key_fn(item, anchor_family_breakdown),
    )
    if not replaceable:
        return None

    for reserve_rule in reserve_lit_pool:
        for replaced_rule in replaceable:
            swapped = [
                rule
                for rule in anchor_rules
                if str(rule.get("canonical_key", ""))
                != str(replaced_rule.get("canonical_key", ""))
            ]
            swapped.append(reserve_rule)
            if len(swapped) != len(anchor_rules):
                continue
            if len({root_feature_key_fn(rule) for rule in swapped if root_feature_key_fn(rule)}) < len(
                anchor_root_keys
            ):
                continue
            swapped_source_breakdown = source_breakdown_fn(swapped)
            swapped_effective_coverage = effective_source_coverage_fn(swapped_source_breakdown)
            if swapped_effective_coverage["literature"] < 2 or swapped_effective_coverage["data"] < 5:
                continue
            if sum(1 for rule in swapped if bool(rule.get("compile_valid", False))) < sum(
                1 for rule in anchor_rules if bool(rule.get("compile_valid", False))
            ):
                continue
            swapped_signature = tuple(
                sorted(alias_group_key_fn(rule) for rule in swapped if alias_group_key_fn(rule))
            )
            if swapped_signature == anchor_signature:
                continue
            audit = {
                "dropped": [],
                "strategy": "lit_preserving_swap",
                "static_reason": "single_swap_preserve_m1_with_novel_literature_root",
                "anchor_candidate_name": str(anchor_candidate.get("name", "") or "M1_top_global"),
                "replaced_rule_canonical_key": str(replaced_rule.get("canonical_key", "")),
                "reserve_rule_canonical_key": str(reserve_rule.get("canonical_key", "")),
                "reserve_root_feature_key": root_feature_key_fn(reserve_rule),
                "reserve_alias_group_key": alias_group_key_fn(reserve_rule),
                "distinct_signature_created": True,
            }
            return swapped[:target_rule_count], audit
    return None
