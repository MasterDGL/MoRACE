"""Shared rule metadata parsing for artifact diagnostics."""

from typing import Dict, List, Sequence


def _root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', '')).strip()}::{str(rule.get('feature_name', '')).strip()}"


def _source_bucket(rule: Dict) -> str:
    support_sources = {str(item).strip() for item in (rule.get("support_sources") or []) if str(item).strip()}
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("source", "") or rule.get("primary_source", "") or "").strip()


def _direction_bucket(relation: str) -> str:
    lowered = str(relation or "").strip().lower()
    if lowered in {"negative", "monotonic_down"}:
        return "down"
    if lowered in {"positive", "monotonic_up"}:
        return "up"
    return lowered or "unknown"


def _mode_bucket(mode: str) -> str:
    lowered = str(mode or "").strip().lower()
    if lowered in {"", "none", "null", "continuous"}:
        return "continuousish"
    if lowered in {"threshold_binary", "binary"}:
        return "thresholded"
    return lowered or "unknown"


def _alias_group_key(rule: Dict) -> str:
    return "::".join(
        [
            _root_feature_key(rule),
            _direction_bucket(rule.get("relation_to_target", "")),
            _mode_bucket(rule.get("feature_mode", "")),
        ]
    )


def _threshold_token(rule: Dict) -> str:
    threshold = rule.get("threshold")
    if not isinstance(threshold, dict) or not threshold:
        return "null"
    op = str(threshold.get("op", "")).strip()
    value = str(threshold.get("value", "")).strip()
    if not op and not value:
        return "null"
    return f"{op}{value}" or "null"


def _effective_canonical_key(rule: Dict) -> str:
    explicit = str(rule.get("canonical_key", "")).strip()
    if explicit:
        return explicit
    family = str(rule.get("feature_family", "")).strip() or "unknown"
    feature = str(rule.get("feature_name", "")).strip() or "unknown"
    mode = str(rule.get("feature_mode", "")).strip()
    relation = str(rule.get("relation_to_target", "")).strip()
    mode_token = mode.lower() if mode else "null"
    relation_token = relation.lower() if relation else "null"
    return f"{family}|{feature}|{mode_token}|{relation_token}|{_threshold_token(rule)}"


def _selected_rules_from_projection(projected_records: Sequence[Dict]) -> List[Dict]:
    selected_rules: List[Dict] = []
    for item in projected_records or []:
        if not bool(item.get("selected_for_merge", False)):
            continue
        selected_rules.extend(list(item.get("parsed_rules", []) or []))
    return selected_rules


def _as_list(value) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []
