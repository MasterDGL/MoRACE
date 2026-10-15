import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from morace.rule_compiler import build_feature_matrix, compile_rule


DEFAULT_MAX_P_VALUE = 0.35
DEFAULT_MIN_EFFECT_SIZE = 0.05
DEFAULT_MIN_ENTROPY = 0.05
DEFAULT_MIN_COVERAGE = 0.55
DEFAULT_MIN_SUPPORT = 1
DEFAULT_DATA_MIN_SUPPORT = 2


def _normalized_support_sources(rule: Dict) -> set[str]:
    raw = rule.get("support_sources", [])
    if raw is None:
        return set()
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip().lower() for item in raw if str(item).strip()}


def _is_literature_only_heuristic(rule: Dict) -> bool:
    return (
        str(rule.get("extraction_source", "projected")).strip().lower() == "heuristic"
        and _normalized_support_sources(rule) == {"literature"}
    )


def compile_valid(rule: Dict) -> bool:
    return compile_rule(rule) is not None


def _safe_numeric(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def rule_statistics(rule: Dict, smiles_train: List[str], y_train: List[float], task_type: str) -> Dict:
    compiled = compile_rule(rule)
    if compiled is None:
        return {
            "compile_valid": False,
            "p_value": 1.0,
            "effect_size": 0.0,
            "non_nan_coverage": 0.0,
            "feature_entropy": 0.0,
        }

    values = np.asarray([row[0] for row in build_feature_matrix([compiled], smiles_train)], dtype=float)
    valid_mask = np.isfinite(values)
    coverage = float(valid_mask.mean()) if len(values) else 0.0
    values = values[valid_mask]
    y = np.asarray(y_train, dtype=float)[valid_mask]

    if len(values) == 0:
        return {
            "compile_valid": True,
            "p_value": 1.0,
            "effect_size": 0.0,
            "non_nan_coverage": 0.0,
            "feature_entropy": 0.0,
        }

    _, counts = np.unique(values.round(6), return_counts=True)
    probs = counts / counts.sum()
    entropy = float(-(probs * np.log2(np.maximum(probs, 1e-12))).sum())

    if task_type == "classification":
        y = y.astype(int)
        if len(np.unique(y)) < 2:
            p_value = 1.0
            effect_size = 0.0
        elif set(np.unique(values)).issubset({0.0, 1.0}):
            present = values >= 0.5
            table = [
                [int(((y == 1) & present).sum()), int(((y == 0) & present).sum())],
                [int(((y == 1) & ~present).sum()), int(((y == 0) & ~present).sum())],
            ]
            _, p_value = fisher_exact(table)
            effect_size = float(values[y == 1].mean() - values[y == 0].mean())
        else:
            pos = _safe_numeric(values[y == 1])
            neg = _safe_numeric(values[y == 0])
            if len(pos) and len(neg):
                _, p_value = mannwhitneyu(pos, neg, alternative="two-sided")
                effect_size = float(np.nanmean(pos) - np.nanmean(neg))
            else:
                p_value = 1.0
                effect_size = 0.0
    else:
        corr, p_value = spearmanr(values, y)
        if math.isnan(corr):
            corr = 0.0
        if math.isnan(p_value):
            p_value = 1.0
        effect_size = float(corr)

    return {
        "compile_valid": True,
        "p_value": float(p_value),
        "effect_size": float(effect_size),
        "non_nan_coverage": coverage,
        "feature_entropy": entropy,
    }


def enrich_rules_with_statistics(
    rules: Sequence[Dict],
    smiles_train: List[str],
    y_train: List[float],
    task_type: str,
) -> List[Dict]:
    enriched: List[Dict] = []
    for rule in rules:
        item = dict(rule)
        item.update(rule_statistics(item, smiles_train, y_train, task_type))
        enriched.append(item)
    return enriched


def gate_failure_reasons(
    rule: Dict,
    max_p_value: float = DEFAULT_MAX_P_VALUE,
    min_effect_size: float = DEFAULT_MIN_EFFECT_SIZE,
    min_entropy: float = DEFAULT_MIN_ENTROPY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    data_min_support: int = DEFAULT_DATA_MIN_SUPPORT,
) -> List[str]:
    reasons: List[str] = []
    if not rule.get("compile_valid", False):
        return ["compile_invalid"]
    sources = _normalized_support_sources(rule)
    required_support = data_min_support if sources == {"data"} else min_support
    literature_only_heuristic = _is_literature_only_heuristic(rule)
    if str(rule.get("extraction_source", "projected")).strip().lower() == "heuristic":
        if literature_only_heuristic:
            required_support = max(required_support, 1)
        else:
            required_support = max(required_support, 3)
    if int(rule.get("projected_support_count", 0)) == 0 and int(rule.get("heuristic_support_count", 0)) > 0:
        max_p_value = min(max_p_value, 0.2)
        if literature_only_heuristic:
            min_effect_size = max(min_effect_size, 0.05)
        else:
            min_effect_size = max(min_effect_size, 0.08)
    if int(rule.get("support_count", 0)) < required_support:
        reasons.append(f"support<{required_support}")
    if float(rule.get("p_value", 1.0)) > max_p_value:
        reasons.append(f"p_value>{max_p_value}")
    if abs(float(rule.get("effect_size", 0.0))) < min_effect_size:
        reasons.append(f"effect_size<{min_effect_size}")
    if float(rule.get("feature_entropy", 0.0)) < min_entropy:
        reasons.append(f"entropy<{min_entropy}")
    if float(rule.get("non_nan_coverage", 0.0)) < min_coverage:
        reasons.append(f"coverage<{min_coverage}")
    return reasons


def passes_rule_gate(
    rule: Dict,
    max_p_value: float = DEFAULT_MAX_P_VALUE,
    min_effect_size: float = DEFAULT_MIN_EFFECT_SIZE,
    min_entropy: float = DEFAULT_MIN_ENTROPY,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_support: int = DEFAULT_MIN_SUPPORT,
    data_min_support: int = DEFAULT_DATA_MIN_SUPPORT,
) -> bool:
    return not gate_failure_reasons(
        rule,
        max_p_value=max_p_value,
        min_effect_size=min_effect_size,
        min_entropy=min_entropy,
        min_coverage=min_coverage,
        min_support=min_support,
        data_min_support=data_min_support,
    )


def _sort_key(rule: Dict):
    heuristic_share = float(rule.get("heuristic_support_count", 0)) / max(1, int(rule.get("support_count", 1)))
    return (
        0 if str(rule.get("extraction_source", "projected")) == "projected" else 1,
        heuristic_share,
        -int(rule.get("support_count", 1)),
        float(rule.get("p_value", 1.0)),
        -abs(float(rule.get("effect_size", 0.0))),
        -float(rule.get("feature_entropy", 0.0)),
        -float(rule.get("non_nan_coverage", 0.0)),
    )


def source_bucket(rule: Dict) -> str:
    support_sources = _normalized_support_sources(rule)
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("primary_source") or rule.get("source") or "literature")


def root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"


def _bucket(rule: Dict) -> str:
    return source_bucket(rule)


def _priority_sorted_candidates(rules: Sequence[Dict], source_priority: Sequence[str] | None) -> List[Dict]:
    if source_priority is None:
        return sorted(rules, key=_sort_key)
    priority_map = {
        source_name: idx
        for idx, source_name in enumerate(source_priority)
    }
    return sorted(
        rules,
        key=lambda rule: (
            priority_map.get(source_bucket(rule), len(priority_map)),
            *_sort_key(rule),
        ),
    )


def select_top_rules(
    rules: Sequence[Dict],
    target_rule_count: int = 30,
    source_balance: bool = False,
    min_source_ratio: float = 0.4,
    heuristic_cap_ratio: float = 0.34,
    max_same_feature_name: int = 2,
    descriptor_cap_ratio: float = 0.4,
    max_same_root_feature: int | None = 1,
    require_non_descriptor: bool = False,
    non_descriptor_seed_count: int = 1,
    source_priority: Sequence[str] | None = None,
    return_audit: bool = False,
) -> List[Dict] | Tuple[List[Dict], Dict]:
    candidates = [rule for rule in rules if passes_rule_gate(rule)]
    candidates = _priority_sorted_candidates(candidates, source_priority)
    descriptor_cap = max(1, int(math.ceil(target_rule_count * descriptor_cap_ratio)))
    audit = {"dropped": [], "require_non_descriptor": require_non_descriptor}

    def can_add(
        rule: Dict,
        feature_name_counter: Dict[str, int],
        family_counter: Dict[str, int],
        root_counter: Dict[str, int],
    ) -> str | None:
        feature_name = str(rule.get("feature_name", ""))
        family = str(rule.get("feature_family", ""))
        root_key = root_feature_key(rule)
        if feature_name_counter.get(feature_name, 0) >= max_same_feature_name:
            return "feature_name_cap"
        if max_same_root_feature is not None and root_counter.get(root_key, 0) >= max_same_root_feature:
            return "root_feature_cap"
        if family == "descriptor" and family_counter.get("descriptor", 0) >= descriptor_cap:
            return "descriptor_cap"
        return None

    def register(
        rule: Dict,
        feature_name_counter: Dict[str, int],
        family_counter: Dict[str, int],
        root_counter: Dict[str, int],
    ) -> None:
        feature_name = str(rule.get("feature_name", ""))
        family = str(rule.get("feature_family", ""))
        root_key = root_feature_key(rule)
        feature_name_counter[feature_name] = feature_name_counter.get(feature_name, 0) + 1
        family_counter[family] = family_counter.get(family, 0) + 1
        root_counter[root_key] = root_counter.get(root_key, 0) + 1

    if not source_balance:
        selected: List[Dict] = []
        selected_keys = set()
        heuristic_count = 0
        heuristic_cap = max(1, int(target_rule_count * heuristic_cap_ratio))
        feature_name_counter: Dict[str, int] = {}
        family_counter: Dict[str, int] = {}
        root_counter: Dict[str, int] = {}

        def try_add(rule: Dict, reason_prefix: str = "") -> bool:
            nonlocal heuristic_count
            canonical_key = rule.get("canonical_key")
            if canonical_key in selected_keys:
                if return_audit:
                    audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}canonical_duplicate".strip("_")})
                return False
            is_heuristic = str(rule.get("extraction_source", "projected")) == "heuristic"
            if is_heuristic and heuristic_count >= heuristic_cap:
                if return_audit:
                    audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}heuristic_cap".strip("_")})
                return False
            blocked = can_add(rule, feature_name_counter, family_counter, root_counter)
            if blocked is not None:
                if return_audit:
                    audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}{blocked}".strip("_")})
                return False
            selected.append(rule)
            selected_keys.add(canonical_key)
            register(rule, feature_name_counter, family_counter, root_counter)
            if is_heuristic:
                heuristic_count += 1
            return True

        if require_non_descriptor:
            seeded_non_descriptors = 0
            for rule in candidates:
                if str(rule.get("feature_family", "")) in {"count", "pattern_presence"} and try_add(
                    rule,
                    reason_prefix="non_descriptor_seed_",
                ):
                    seeded_non_descriptors += 1
                    if seeded_non_descriptors >= max(1, non_descriptor_seed_count):
                        break

        for rule in candidates:
            if len(selected) >= target_rule_count:
                break
            try_add(rule)
        if return_audit:
            return selected, audit
        return selected

    grouped = {"literature": [], "data": [], "multi": []}
    for rule in candidates:
        grouped.setdefault(_bucket(rule), []).append(rule)

    min_per_source = max(1, int(target_rule_count * min_source_ratio))
    selected = []
    selected_keys = set()
    feature_name_counter: Dict[str, int] = {}
    family_counter: Dict[str, int] = {}
    root_counter: Dict[str, int] = {}

    def try_add_balanced(rule: Dict, reason_prefix: str = "") -> bool:
        canonical_key = rule.get("canonical_key")
        if canonical_key in selected_keys:
            if return_audit:
                audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}canonical_duplicate".strip("_")})
            return False
        blocked = can_add(rule, feature_name_counter, family_counter, root_counter)
        if blocked is not None:
            if return_audit:
                audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}{blocked}".strip("_")})
            return False
        selected.append(rule)
        selected_keys.add(canonical_key)
        register(rule, feature_name_counter, family_counter, root_counter)
        return True

    if require_non_descriptor:
        seeded_non_descriptors = 0
        for rule in candidates:
            if str(rule.get("feature_family", "")) in {"count", "pattern_presence"} and try_add_balanced(
                rule,
                reason_prefix="non_descriptor_seed_",
            ):
                seeded_non_descriptors += 1
                if seeded_non_descriptors >= max(1, non_descriptor_seed_count):
                    break

    for source_name in ("literature", "data"):
        source_rules = grouped.get(source_name, [])
        for rule in source_rules:
            if len([item for item in selected if _bucket(item) == source_name]) >= min_per_source:
                break
            try_add_balanced(rule)
        deficit = min_per_source - len([item for item in selected if _bucket(item) == source_name])
        if deficit > 0:
            for rule in grouped.get("multi", []):
                if deficit <= 0 or rule.get("canonical_key") in selected_keys:
                    continue
                if try_add_balanced(rule):
                    deficit -= 1

    for rule in candidates:
        if len(selected) >= target_rule_count:
            break
        if rule.get("canonical_key") in selected_keys:
            continue
        try_add_balanced(rule)
    if return_audit:
        return selected[:target_rule_count], audit
    return selected[:target_rule_count]


def branch_preserving_select_top_rules(
    rules: Sequence[Dict],
    target_rule_count: int = 30,
    heuristic_cap_ratio: float = 0.34,
    descriptor_cap_ratio: float = 0.75,
    max_same_root_feature: int = 1,
    return_audit: bool = False,
) -> List[Dict] | Tuple[List[Dict], Dict]:
    candidates = [rule for rule in rules if passes_rule_gate(rule)]
    candidates = sorted(candidates, key=_sort_key)
    if not candidates:
        return ([], {"dropped": []}) if return_audit else []

    descriptor_cap = max(2, int(math.ceil(target_rule_count * descriptor_cap_ratio)))
    heuristic_cap = max(1, int(math.ceil(target_rule_count * heuristic_cap_ratio)))

    selected: List[Dict] = []
    selected_canonical_keys = set()
    root_counter: Dict[str, int] = {}
    family_counter: Dict[str, int] = {}
    heuristic_count = 0
    audit = {"dropped": []}

    def can_add(rule: Dict) -> str | None:
        nonlocal heuristic_count
        root_key = root_feature_key(rule)
        if root_counter.get(root_key, 0) >= max_same_root_feature:
            return "root_feature_cap"
        family = str(rule.get("feature_family", ""))
        if family == "descriptor" and family_counter.get("descriptor", 0) >= descriptor_cap:
            return "descriptor_cap"
        is_heuristic = str(rule.get("extraction_source", "projected")) == "heuristic"
        if is_heuristic and heuristic_count >= heuristic_cap:
            return "heuristic_cap"
        return None

    def add(rule: Dict, reason_prefix: str = "") -> bool:
        nonlocal heuristic_count
        canonical_key = rule.get("canonical_key")
        if canonical_key in selected_canonical_keys:
            if return_audit:
                audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}canonical_duplicate".strip("_")})
            return False
        blocked = can_add(rule)
        if blocked is not None:
            if return_audit:
                audit["dropped"].append({"rule": rule, "reason": f"{reason_prefix}{blocked}".strip("_")})
            return False
        selected.append(rule)
        selected_canonical_keys.add(canonical_key)
        root_key = root_feature_key(rule)
        root_counter[root_key] = root_counter.get(root_key, 0) + 1
        family = str(rule.get("feature_family", ""))
        family_counter[family] = family_counter.get(family, 0) + 1
        if str(rule.get("extraction_source", "projected")) == "heuristic":
            heuristic_count += 1
        return True

    literature_candidates = [rule for rule in candidates if source_bucket(rule) == "literature"]
    inference_descriptor_candidates = [
        rule
        for rule in candidates
        if source_bucket(rule) == "data" and str(rule.get("feature_family", "")) == "descriptor"
    ]
    inference_non_descriptor_candidates = [
        rule
        for rule in candidates
        if source_bucket(rule) == "data" and str(rule.get("feature_family", "")) in {"count", "pattern_presence"}
    ]
    multi_descriptor_candidates = [
        rule
        for rule in candidates
        if source_bucket(rule) == "multi" and str(rule.get("feature_family", "")) == "descriptor"
    ]
    multi_non_descriptor_candidates = [
        rule
        for rule in candidates
        if source_bucket(rule) == "multi" and str(rule.get("feature_family", "")) in {"count", "pattern_presence"}
    ]

    def add_first(candidate_lists: List[List[Dict]]) -> None:
        for candidate_list in candidate_lists:
            for rule in candidate_list:
                if add(rule, reason_prefix="seed_"):
                    return

    add_first([literature_candidates])
    add_first([inference_descriptor_candidates, multi_descriptor_candidates])
    add_first([inference_non_descriptor_candidates, multi_non_descriptor_candidates])
    if not inference_non_descriptor_candidates and not multi_non_descriptor_candidates:
        add_first([multi_non_descriptor_candidates, inference_descriptor_candidates, multi_descriptor_candidates])

    for rule in candidates:
        if len(selected) >= target_rule_count:
            break
        add(rule)
    if return_audit:
        return selected[:target_rule_count], audit
    return selected[:target_rule_count]
