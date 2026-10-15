"""Metrics and coverage feasibility for rule aggregation."""

from typing import Dict, List, Sequence, Tuple


def family_breakdown(rules: List[Dict], *, family_order: Sequence[str]) -> Dict[str, int]:
    return {
        family: sum(1 for item in rules if item.get("feature_family") == family)
        for family in family_order
    }


def source_breakdown(
    rules: List[Dict],
    *,
    source_order: Sequence[str],
    source_bucket_fn,
) -> Dict[str, int]:
    return {
        source: sum(1 for item in rules if source_bucket_fn(item) == source)
        for source in source_order
    }


def effective_source_coverage(source_breakdown: Dict[str, int]) -> Dict[str, int]:
    multi_count = int(source_breakdown.get("multi", 0))
    return {
        "literature": int(source_breakdown.get("literature", 0)) + multi_count,
        "data": int(source_breakdown.get("data", 0)) + multi_count,
    }


def compile_safe_source_options(rule: Dict, *, source_bucket_fn) -> List[Tuple[int, int]]:
    if not bool(rule.get("compile_valid", False)):
        return []
    source_name = source_bucket_fn(rule)
    if source_name == "literature":
        return [(1, 0)]
    if source_name == "data":
        return [(0, 1)]
    if source_name == "multi":
        return [(1, 1)]
    return []


def coverage_feasible_under_budget(
    compile_safe_rules: List[Dict],
    *,
    target_rule_count: int,
    min_effective_literature: int = 2,
    min_effective_data: int = 5,
    min_roots: int = 6,
    root_feature_key_fn,
    compile_safe_source_options_fn,
) -> bool:
    if target_rule_count < min_roots:
        return False
    root_options: Dict[str, set[Tuple[int, int]]] = {}
    for rule in compile_safe_rules:
        root_key = root_feature_key_fn(rule)
        if not root_key:
            continue
        for option in compile_safe_source_options_fn(rule):
            root_options.setdefault(root_key, set()).add(option)
    if len(root_options) < min_roots:
        return False

    states = {(0, 0, 0)}
    for options in root_options.values():
        next_states = set(states)
        for selected_roots, literature_cov, data_cov in states:
            for add_literature, add_data in options:
                if selected_roots + 1 > target_rule_count:
                    continue
                next_states.add(
                    (
                        selected_roots + 1,
                        min(min_effective_literature, literature_cov + add_literature),
                        min(min_effective_data, data_cov + add_data),
                    )
                )
        states = next_states
    return any(
        selected_roots >= min_roots
        and literature_cov >= min_effective_literature
        and data_cov >= min_effective_data
        for selected_roots, literature_cov, data_cov in states
    )


def compile_safe_multi_feasibility_metrics(
    merged_rules: List[Dict],
    enriched_rules: List[Dict],
    *,
    target_rule_count: int,
    root_feature_key_fn,
    source_bucket_fn,
    detect_cross_source_consensus_groups_fn,
    coverage_feasible_under_budget_fn,
) -> Dict:
    promoted_multi_rules = [item for item in merged_rules if bool(item.get("consensus_promoted", False))]
    promoted_multi_root_keys = sorted(
        {
            str(item.get("promoted_root_feature_key", "") or root_feature_key_fn(item) or "").strip()
            for item in promoted_multi_rules
            if str(item.get("promoted_root_feature_key", "") or root_feature_key_fn(item) or "").strip()
        }
    )
    compile_safe_rules = [item for item in enriched_rules if bool(item.get("compile_valid", False))]
    compile_safe_multi_rules = [item for item in compile_safe_rules if source_bucket_fn(item) == "multi"]
    compile_safe_promoted_multi_rules = [
        item for item in compile_safe_rules if bool(item.get("consensus_promoted", False))
    ]
    compile_safe_multi_root_keys = sorted(
        {
            root_feature_key_fn(item)
            for item in compile_safe_multi_rules
            if root_feature_key_fn(item)
        }
    )
    cross_source_consensus_groups = detect_cross_source_consensus_groups_fn(enriched_rules)
    compile_safe_cross_source_root_keys = sorted(
        {
            str(item.get("root_feature_key", ""))
            for item in cross_source_consensus_groups
            if bool(item.get("has_cross_source_overlap"))
            and int(item.get("compile_valid_true_count", 0) or 0) > 0
            and str(item.get("root_feature_key", ""))
        }
    )
    return {
        "cross_source_consensus_groups": cross_source_consensus_groups,
        "promoted_multi_candidate_count": len(promoted_multi_rules),
        "promoted_multi_root_count": len(promoted_multi_root_keys),
        "promoted_multi_root_keys": promoted_multi_root_keys,
        "compile_safe_multi_rule_count": len(compile_safe_multi_rules),
        "compile_safe_multi_root_count": len(compile_safe_multi_root_keys),
        "compile_safe_multi_root_keys": compile_safe_multi_root_keys,
        "promoted_multi_compile_safe_count": len(compile_safe_promoted_multi_rules),
        "compile_safe_cross_source_root_count": len(compile_safe_cross_source_root_keys),
        "compile_safe_cross_source_root_keys": compile_safe_cross_source_root_keys,
        "coverage_feasible_under_budget": coverage_feasible_under_budget_fn(
            compile_safe_rules,
            target_rule_count=target_rule_count,
        ),
    }


def dominant_feature_share(rules: List[Dict]) -> float:
    if not rules:
        return 0.0
    feature_counts: Dict[str, int] = {}
    for item in rules:
        feature_name = str(item.get("feature_name", ""))
        feature_counts[feature_name] = feature_counts.get(feature_name, 0) + 1
    return float(max(feature_counts.values()) / max(1, len(rules)))
