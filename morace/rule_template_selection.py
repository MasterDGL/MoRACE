"""Stateful template selection for aggregated rule candidates."""

from typing import Dict, List, Sequence, Tuple


def select_template_rules(
    candidates: List[Dict],
    target_rule_count: int,
    template_type: str,
    source_priority: Tuple[str, ...],
    require_non_descriptor_count: int = 0,
    descriptor_cap_ratio: float = 1.0,
    literature_anchor_count: int = 0,
    data_seed_count: int = 0,
    max_literature_rules: int | None = None,
    max_data_rules: int | None = None,
    coverage_plan: List[Tuple[str, str]] | None = None,
    max_same_root_per_template: int = 1,
    allow_second_variant_by_root: bool = False,
    *,
    source_order: Sequence[str],
    root_feature_key_fn,
    source_bucket_fn,
    alias_group_key_fn,
    family_matches_fn,
    second_variant_complement_dimensions_fn,
    is_semantically_equivalent_second_variant_fn,
    second_variant_pair_quality_fn,
    semantic_variant_signature_fn,
) -> Tuple[List[Dict], Dict]:
    selected: List[Dict] = []
    selected_keys = set()
    root_counter: Dict[str, int] = {}
    selected_rules_by_root: Dict[str, List[Dict]] = {}
    source_counts = {source: 0 for source in source_order}
    descriptor_count = 0
    descriptor_cap = max(1, int(round(target_rule_count * descriptor_cap_ratio)))
    audit = {
        "dropped": [],
        "strategy": template_type,
        "capacity_release_mode": (
            f"local_root_cap_release_max{max_same_root_per_template}"
            if allow_second_variant_by_root and max_same_root_per_template > 1
            else "strict_root_cap_1"
        ),
        "selected_second_variant_count": 0,
        "selected_same_root_alias_pairs": [],
        "selected_cross_source_second_variant_count": 0,
        "selected_repeated_root_count": 0,
        "second_variant_quality_score": 0.0,
    }

    def can_add(rule: Dict) -> str | None:
        root_key = root_feature_key_fn(rule)
        existing_rules = selected_rules_by_root.get(root_key, [])
        if root_counter.get(root_key, 0) >= max_same_root_per_template:
            return "root_feature_cap"
        if existing_rules:
            if not allow_second_variant_by_root:
                return "root_feature_cap"
            if len(existing_rules) >= max_same_root_per_template:
                return "root_feature_cap"
            if int(rule.get("support_count", 0) or 0) < 2:
                return "second_variant_support_cap"
            candidate_alias = alias_group_key_fn(rule)
            existing_aliases = {alias_group_key_fn(item) for item in existing_rules}
            if candidate_alias in existing_aliases:
                return "second_variant_same_alias"
            complementary = False
            cross_source = False
            projected_present = str(rule.get("extraction_source", "projected")) == "projected"
            for existing_rule in existing_rules:
                dimensions = second_variant_complement_dimensions_fn(existing_rule, rule)
                if dimensions:
                    complementary = True
                if source_bucket_fn(existing_rule) != source_bucket_fn(rule):
                    cross_source = True
                if str(existing_rule.get("extraction_source", "projected")) == "projected":
                    projected_present = True
            if not complementary:
                return "second_variant_not_complementary"
            if not projected_present:
                return "second_variant_all_heuristic"
            for existing_rule in existing_rules:
                if is_semantically_equivalent_second_variant_fn(existing_rule, rule):
                    return "second_variant_semantic_equivalent"
        family = str(rule.get("feature_family", ""))
        if family == "descriptor" and descriptor_count >= descriptor_cap:
            return "descriptor_cap"
        bucket = source_bucket_fn(rule)
        if (
            max_literature_rules is not None
            and bucket == "literature"
            and source_counts["literature"] >= max_literature_rules
        ):
            return "literature_cap"
        if max_data_rules is not None and bucket == "data" and source_counts["data"] >= max_data_rules:
            return "data_cap"
        return None

    def add(rule: Dict, reason_prefix: str = "") -> bool:
        nonlocal descriptor_count
        canonical_key = str(rule.get("canonical_key", ""))
        if canonical_key in selected_keys:
            audit["dropped"].append(
                {"rule": rule, "reason": f"{reason_prefix}canonical_duplicate".strip("_")}
            )
            return False
        blocked = can_add(rule)
        if blocked is not None:
            audit["dropped"].append(
                {"rule": rule, "reason": f"{reason_prefix}{blocked}".strip("_")}
            )
            return False
        selected.append(rule)
        selected_keys.add(canonical_key)
        root_key = root_feature_key_fn(rule)
        existing_rules = list(selected_rules_by_root.get(root_key, []))
        root_counter[root_key] = root_counter.get(root_key, 0) + 1
        selected_rules_by_root.setdefault(root_key, []).append(rule)
        bucket = source_bucket_fn(rule)
        source_counts[bucket] = source_counts.get(bucket, 0) + 1
        if str(rule.get("feature_family", "")) == "descriptor":
            descriptor_count += 1
        if existing_rules:
            pair_quality_score = second_variant_pair_quality_fn(existing_rules, rule)
            audit["selected_second_variant_count"] += 1
            complement_dimensions = sorted(
                {
                    dim
                    for existing_rule in existing_rules
                    for dim in second_variant_complement_dimensions_fn(existing_rule, rule)
                }
            )
            cross_source = any(
                source_bucket_fn(existing_rule) != source_bucket_fn(rule)
                for existing_rule in existing_rules
            )
            if cross_source:
                audit["selected_cross_source_second_variant_count"] += 1
            audit["second_variant_quality_score"] += pair_quality_score
            audit["selected_same_root_alias_pairs"].append(
                {
                    "root_feature_key": root_key,
                    "existing_alias_group_keys": sorted(
                        {alias_group_key_fn(existing_rule) for existing_rule in existing_rules}
                    ),
                    "new_alias_group_key": alias_group_key_fn(rule),
                    "existing_canonical_keys": sorted(
                        {str(existing_rule.get("canonical_key", "")) for existing_rule in existing_rules}
                    ),
                    "new_canonical_key": canonical_key,
                    "existing_sources": sorted(
                        {source_bucket_fn(existing_rule) for existing_rule in existing_rules}
                    ),
                    "new_source": bucket,
                    "complement_dimensions": complement_dimensions,
                    "cross_source": cross_source,
                    "existing_semantic_variant_signatures": [
                        list(semantic_variant_signature_fn(existing_rule))
                        for existing_rule in existing_rules
                    ],
                    "new_semantic_variant_signature": list(semantic_variant_signature_fn(rule)),
                    "pair_quality_score": pair_quality_score,
                }
            )
            audit["selected_repeated_root_count"] = sum(
                1 for rules_for_root in selected_rules_by_root.values() if len(rules_for_root) > 1
            )
        return True

    if coverage_plan:
        for source_name, family_group in coverage_plan:
            for rule in candidates:
                if source_bucket_fn(rule) != source_name or not family_matches_fn(rule, family_group):
                    continue
                if add(rule, reason_prefix=f"coverage_seed_{source_name}_{family_group}_"):
                    break

    if literature_anchor_count > 0:
        anchors_added = 0
        for rule in candidates:
            if source_bucket_fn(rule) != "literature":
                continue
            if add(rule, reason_prefix="literature_anchor_"):
                anchors_added += 1
                if anchors_added >= literature_anchor_count:
                    break

    if data_seed_count > 0:
        data_added = 0
        for rule in candidates:
            if source_bucket_fn(rule) not in {"data", "multi"}:
                continue
            if add(rule, reason_prefix="data_seed_"):
                data_added += 1
                if data_added >= data_seed_count:
                    break

    if require_non_descriptor_count > 0:
        non_descriptor_added = sum(
            1 for item in selected if str(item.get("feature_family", "")) in {"count", "pattern_presence"}
        )
        for rule in candidates:
            if str(rule.get("feature_family", "")) not in {"count", "pattern_presence"}:
                continue
            if add(rule, reason_prefix="non_descriptor_seed_"):
                non_descriptor_added += 1
                if non_descriptor_added >= require_non_descriptor_count:
                    break

    for rule in candidates:
        if len(selected) >= target_rule_count:
            break
        add(rule)

    return selected[:target_rule_count], audit


def select_balanced_coverage_rules(
    enriched: List[Dict],
    target_rule_count: int,
    *,
    passes_rule_gate_fn,
    candidate_quality_key_fn,
    root_feature_key_fn,
    source_bucket_fn,
) -> Tuple[List[Dict], Dict]:
    candidates = sorted([item for item in enriched if passes_rule_gate_fn(item)], key=candidate_quality_key_fn)
    selected: List[Dict] = []
    selected_keys = set()
    root_counter: Dict[str, int] = {}
    descriptor_count = 0
    descriptor_cap = max(2, int(round(target_rule_count * 0.5)))
    audit = {"dropped": [], "strategy": "balanced_source_family_coverage"}
    coverage_order = [
        ("literature", "count"),
        ("literature", "descriptor"),
        ("data", "count"),
        ("data", "descriptor"),
        ("multi", "count"),
        ("multi", "descriptor"),
    ]

    def can_add(rule: Dict) -> str | None:
        root_key = root_feature_key_fn(rule)
        if root_counter.get(root_key, 0) >= 1:
            return "root_feature_cap"
        if str(rule.get("feature_family", "")) == "descriptor" and descriptor_count >= descriptor_cap:
            return "descriptor_cap"
        return None

    def add(rule: Dict, reason_prefix: str = "") -> bool:
        nonlocal descriptor_count
        canonical_key = str(rule.get("canonical_key", ""))
        if canonical_key in selected_keys:
            audit["dropped"].append(
                {"rule": rule, "reason": f"{reason_prefix}canonical_duplicate".strip("_")}
            )
            return False
        blocked = can_add(rule)
        if blocked is not None:
            audit["dropped"].append(
                {"rule": rule, "reason": f"{reason_prefix}{blocked}".strip("_")}
            )
            return False
        selected.append(rule)
        selected_keys.add(canonical_key)
        root_key = root_feature_key_fn(rule)
        root_counter[root_key] = root_counter.get(root_key, 0) + 1
        if str(rule.get("feature_family", "")) == "descriptor":
            descriptor_count += 1
        return True

    for source_name, family_name in coverage_order:
        for rule in candidates:
            family = str(rule.get("feature_family", ""))
            if source_bucket_fn(rule) != source_name:
                continue
            if family_name == "count" and family not in {"count", "pattern_presence"}:
                continue
            if family_name == "descriptor" and family != "descriptor":
                continue
            if add(rule, reason_prefix=f"coverage_seed_{source_name}_{family_name}_"):
                break

    for rule in candidates:
        if len(selected) >= target_rule_count:
            break
        add(rule)

    return selected[:target_rule_count], audit
