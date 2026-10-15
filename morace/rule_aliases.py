"""Alias grouping and semantic comparison for rule variants."""

from typing import Dict, List, Tuple


def direction_bucket(relation: str) -> str:
    lowered = str(relation or "").strip().lower()
    if lowered in {"negative", "monotonic_down"}:
        return "down"
    if lowered in {"positive", "monotonic_up"}:
        return "up"
    return lowered or "unknown"


def mode_bucket(mode: str) -> str:
    lowered = str(mode or "").strip().lower()
    if lowered in {"", "none", "null", "continuous"}:
        return "continuousish"
    if lowered in {"threshold_binary", "binary"}:
        return "thresholded"
    return lowered or "unknown"


def alias_group_key(
    rule: Dict,
    *,
    root_feature_key_fn,
    direction_bucket_fn=direction_bucket,
    mode_bucket_fn=mode_bucket,
) -> str:
    root_key = root_feature_key_fn(rule) or str(rule.get("canonical_key", "") or "unknown")
    return "::".join(
        [
            root_key,
            direction_bucket_fn(rule.get("relation_to_target", "")),
            mode_bucket_fn(rule.get("feature_mode", "")),
        ]
    )


def alias_group_summary(
    rules: List[Dict],
    *,
    alias_group_key_fn,
    root_feature_key_fn,
    source_bucket_fn,
    direction_bucket_fn=direction_bucket,
    mode_bucket_fn=mode_bucket,
) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for item in rules:
        group_key = alias_group_key_fn(item)
        entry = grouped.setdefault(
            group_key,
            {
                "alias_group_key": group_key,
                "root_feature_key": root_feature_key_fn(item),
                "feature_family": str(item.get("feature_family", "")),
                "feature_name": str(item.get("feature_name", "")),
                "direction_bucket": direction_bucket_fn(item.get("relation_to_target", "")),
                "mode_bucket": mode_bucket_fn(item.get("feature_mode", "")),
                "rule_count": 0,
                "total_support": 0,
                "sources": {},
                "canonical_keys": [],
            },
        )
        entry["rule_count"] += 1
        entry["total_support"] += int(item.get("support_count", 0) or 0)
        source_name = source_bucket_fn(item)
        entry["sources"][source_name] = entry["sources"].get(source_name, 0) + 1
        canonical_key = str(item.get("canonical_key", ""))
        if canonical_key and canonical_key not in entry["canonical_keys"]:
            entry["canonical_keys"].append(canonical_key)
    return sorted(
        grouped.values(),
        key=lambda item: (-int(item["total_support"]), -int(item["rule_count"]), item["alias_group_key"]),
    )


def alias_group_count(rules: List[Dict], *, alias_group_key_fn) -> int:
    return len({alias_group_key_fn(item) for item in rules if alias_group_key_fn(item)})


def alias_compaction_score(rules: List[Dict], *, alias_group_count_fn) -> float:
    if not rules:
        return 0.0
    return float(1.0 - (alias_group_count_fn(rules) / max(1, len(rules))))


def semantic_value_polarity(rule: Dict, *, mode_bucket_fn=mode_bucket) -> str:
    family = str(rule.get("feature_family", "")).strip().lower()
    relation = str(rule.get("relation_to_target", "")).strip().lower()
    threshold = rule.get("threshold") or {}
    threshold_op = str(threshold.get("op", "")).strip().lower()
    mode = str(rule.get("feature_mode", "")).strip().lower()
    normalized_mode = mode_bucket_fn(mode)

    if family == "pattern_presence":
        if relation in {"positive", "monotonic_up"}:
            return "presence_positive_signal"
        if relation in {"negative", "monotonic_down"}:
            return "presence_negative_signal"
        return "presence_unknown_signal"

    if threshold_op in {"<", "<="}:
        return "low_value_signal"
    if threshold_op in {">", ">="}:
        return "high_value_signal"
    if mode == "range":
        if relation in {"positive", "monotonic_up"}:
            return "range_positive_signal"
        if relation in {"negative", "monotonic_down"}:
            return "range_negative_signal"
        return "range_unknown_signal"
    if normalized_mode == "thresholded":
        if relation in {"positive", "monotonic_up"}:
            return "presence_positive_signal"
        if relation in {"negative", "monotonic_down"}:
            return "presence_negative_signal"
        return "presence_unknown_signal"
    if relation in {"negative", "monotonic_down"}:
        return "low_value_signal"
    if relation in {"positive", "monotonic_up"}:
        return "high_value_signal"
    return "unknown_signal"


def semantic_variant_signature(
    rule: Dict,
    *,
    root_feature_key_fn,
    semantic_value_polarity_fn,
) -> Tuple[str, str, str]:
    family = str(rule.get("feature_family", "")).strip().lower()
    mode = str(rule.get("feature_mode", "")).strip().lower()
    if family == "pattern_presence" or mode in {"binary", "binary_presence", "threshold_binary"}:
        semantic_form = "presence_like"
    elif mode == "range":
        semantic_form = "range"
    else:
        semantic_form = "value_like"
    return (
        root_feature_key_fn(rule) or str(rule.get("canonical_key", "") or "unknown"),
        semantic_form,
        semantic_value_polarity_fn(rule),
    )


def is_semantically_equivalent_second_variant(
    existing_rule: Dict,
    candidate_rule: Dict,
    *,
    semantic_variant_signature_fn,
) -> bool:
    existing_root, existing_form, existing_polarity = semantic_variant_signature_fn(existing_rule)
    candidate_root, candidate_form, candidate_polarity = semantic_variant_signature_fn(candidate_rule)
    if not existing_root or existing_root != candidate_root:
        return False
    if existing_polarity == "unknown_signal" or candidate_polarity == "unknown_signal":
        return False
    if existing_polarity != candidate_polarity:
        return False
    if existing_form == candidate_form:
        return True
    return {existing_form, candidate_form} == {"value_like", "presence_like"}


def root_feature_summary(
    rules: List[Dict],
    *,
    root_feature_key_fn,
    source_bucket_fn,
    direction_bucket_fn=direction_bucket,
    mode_bucket_fn=mode_bucket,
) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for item in rules:
        root_key = root_feature_key_fn(item)
        if not root_key:
            continue
        entry = grouped.setdefault(
            root_key,
            {
                "root_feature_key": root_key,
                "feature_family": str(item.get("feature_family", "")),
                "feature_name": str(item.get("feature_name", "")),
                "rule_count": 0,
                "total_support": 0,
                "sources": {},
                "direction_buckets": {},
                "mode_buckets": {},
                "canonical_keys": [],
            },
        )
        entry["rule_count"] += 1
        entry["total_support"] += int(item.get("support_count", 0) or 0)
        source_name = source_bucket_fn(item)
        entry["sources"][source_name] = entry["sources"].get(source_name, 0) + 1
        normalized_direction = direction_bucket_fn(item.get("relation_to_target", ""))
        entry["direction_buckets"][normalized_direction] = (
            entry["direction_buckets"].get(normalized_direction, 0) + 1
        )
        normalized_mode = mode_bucket_fn(item.get("feature_mode", ""))
        entry["mode_buckets"][normalized_mode] = entry["mode_buckets"].get(normalized_mode, 0) + 1
        canonical_key = str(item.get("canonical_key", ""))
        if canonical_key and canonical_key not in entry["canonical_keys"]:
            entry["canonical_keys"].append(canonical_key)
    return sorted(
        grouped.values(),
        key=lambda item: (-int(item["total_support"]), -int(item["rule_count"]), item["root_feature_key"]),
    )


def second_variant_complement_dimensions(
    existing_rule: Dict,
    candidate_rule: Dict,
    *,
    source_bucket_fn,
    direction_bucket_fn=direction_bucket,
    mode_bucket_fn=mode_bucket,
) -> List[str]:
    dimensions: List[str] = []
    if source_bucket_fn(existing_rule) != source_bucket_fn(candidate_rule):
        dimensions.append("source")
    if direction_bucket_fn(existing_rule.get("relation_to_target", "")) != direction_bucket_fn(
        candidate_rule.get("relation_to_target", "")
    ):
        dimensions.append("direction")
    if mode_bucket_fn(existing_rule.get("feature_mode", "")) != mode_bucket_fn(
        candidate_rule.get("feature_mode", "")
    ):
        dimensions.append("mode")
    return dimensions


def second_variant_pair_quality(
    existing_rules: List[Dict],
    candidate_rule: Dict,
    *,
    source_bucket_fn,
    second_variant_complement_dimensions_fn,
) -> float:
    if not existing_rules:
        return 0.0
    complement_dimensions = sorted(
        {
            dim
            for existing_rule in existing_rules
            for dim in second_variant_complement_dimensions_fn(existing_rule, candidate_rule)
        }
    )
    cross_source = any(
        source_bucket_fn(existing_rule) != source_bucket_fn(candidate_rule)
        for existing_rule in existing_rules
    )
    support_floor = max(
        0.0,
        min(
            min(int(existing_rule.get("support_count", 0) or 0) for existing_rule in existing_rules),
            int(candidate_rule.get("support_count", 0) or 0),
        ),
    )
    quality = 1.0
    if cross_source:
        quality += 0.35
    if "direction" in complement_dimensions:
        quality += 0.20
    if "mode" in complement_dimensions:
        quality += 0.20
    if "source" in complement_dimensions:
        quality += 0.10
    quality += min(support_floor, 8.0) / 8.0 * 0.15
    return float(quality)
