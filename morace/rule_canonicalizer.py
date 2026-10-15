from copy import deepcopy
import os
from typing import Dict, Iterable, List


NULLISH = {"", "none", "null", "n/a", "na"}
MULTI_CONSENSUS_PROMOTION_ENABLE = False
MOLLOGP_MERGE_SOURCE_REFINEMENT_ROOT = "descriptor::MolLogP"


def canonical_threshold_key(threshold) -> str:
    if not threshold:
        return "null"
    return f"{threshold['op']}:{round(float(threshold['value']), 4)}"


def canonical_field_value(value) -> str:
    if value is None:
        return "null"
    text = str(value).strip()
    if text.lower() in NULLISH:
        return "null"
    return text if text else "null"


def canonical_rule_key(rule: Dict) -> str:
    return "|".join(
        [
            canonical_field_value(rule.get("feature_family")),
            canonical_field_value(rule.get("feature_name")),
            canonical_field_value(rule.get("feature_mode")),
            canonical_field_value(rule.get("relation_to_target")),
            canonical_threshold_key(rule.get("threshold")),
        ]
    )


def root_feature_key_for_canonical_rule(rule: Dict) -> str:
    family = canonical_field_value(rule.get("feature_family"))
    feature_name = canonical_field_value(rule.get("feature_name"))
    if family != "null" or feature_name != "null":
        return f"{family}::{feature_name}"
    canonical_key = str(rule.get("canonical_key", "")).strip()
    return canonical_key or "unknown"


def _as_nonneg_int(value, default: int = 0) -> int:
    try:
        parsed = int(value)
        return parsed if parsed >= 0 else default
    except Exception:
        return default


def _normalize_counter(raw, fallback_key=None, fallback_value: int = 1) -> Dict[str, int]:
    normalized: Dict[str, int] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            counter_key = str(key or "unknown")
            counter_value = _as_nonneg_int(value, 0)
            if counter_value > 0:
                normalized[counter_key] = normalized.get(counter_key, 0) + counter_value
    if not normalized and fallback_key is not None:
        normalized[str(fallback_key or "unknown")] = max(1, _as_nonneg_int(fallback_value, 1))
    return normalized


def _incoming_support(rule: Dict) -> int:
    source_votes = _normalize_counter(rule.get("source_votes"))
    if source_votes:
        return sum(source_votes.values())

    declared_support = _as_nonneg_int(rule.get("support_count"), 0)
    if declared_support > 0:
        return declared_support

    projected_support = _as_nonneg_int(rule.get("projected_support_count"), 0)
    heuristic_support = _as_nonneg_int(rule.get("heuristic_support_count"), 0)
    if projected_support + heuristic_support > 0:
        return projected_support + heuristic_support

    provenance_list = rule.get("provenance_list")
    if isinstance(provenance_list, list) and provenance_list:
        return len(provenance_list)

    return 1


def _incoming_source_votes(rule: Dict) -> Dict[str, int]:
    total_support = _incoming_support(rule)
    fallback_source = rule.get("primary_source") or rule.get("source") or "unknown"
    return _normalize_counter(rule.get("source_votes"), fallback_source, total_support)


def _incoming_extraction_votes(rule: Dict) -> Dict[str, int]:
    extraction_votes = _normalize_counter(rule.get("extraction_votes"))
    if extraction_votes:
        return extraction_votes

    projected_support = _as_nonneg_int(rule.get("projected_support_count"), 0)
    heuristic_support = _as_nonneg_int(rule.get("heuristic_support_count"), 0)
    if projected_support or heuristic_support:
        recovered_votes: Dict[str, int] = {}
        if projected_support:
            recovered_votes["projected"] = projected_support
        if heuristic_support:
            recovered_votes["heuristic"] = heuristic_support
        return recovered_votes

    total_support = _incoming_support(rule)
    extraction_source = str(rule.get("dominant_extraction_source") or rule.get("extraction_source") or "projected")
    return {extraction_source: total_support}


def _incoming_support_models(rule: Dict) -> List[str]:
    models = {str(item) for item in (rule.get("support_models") or []) if item}
    if not models:
        for provenance in rule.get("provenance_list") or []:
            model_name = str((provenance or {}).get("draft_model") or "").strip()
            if model_name:
                models.add(model_name)
    if not models:
        models.add(str(rule.get("provenance", {}).get("draft_model") or "unknown"))
    return sorted(models)


def _incoming_support_sources(rule: Dict) -> List[str]:
    sources = {str(item) for item in (rule.get("support_sources") or []) if item}
    if not sources:
        sources.update(_incoming_source_votes(rule).keys())
    if not sources:
        sources.add(str(rule.get("source") or "unknown"))
    return sorted(sources)


def _incoming_provenance_list(rule: Dict) -> List[Dict]:
    provenance_list = rule.get("provenance_list")
    if isinstance(provenance_list, list) and provenance_list:
        return [deepcopy(item) for item in provenance_list]
    return [deepcopy(rule.get("provenance", {}))]


def _rule_source_evidence(rule: Dict) -> List[str]:
    evidence = []
    support_sources = rule.get("support_sources")
    if isinstance(support_sources, list):
        for item in support_sources:
            source_name = str(item or "").strip()
            if source_name and source_name not in evidence:
                evidence.append(source_name)
    source_votes = rule.get("source_votes")
    if isinstance(source_votes, dict):
        for source_name in source_votes.keys():
            normalized = str(source_name or "").strip()
            if normalized and normalized not in evidence:
                evidence.append(normalized)
    for field_name in ("source", "primary_source"):
        normalized = str(rule.get(field_name, "") or "").strip()
        if normalized == "multi":
            for source_name in ("literature", "data"):
                if source_name not in evidence:
                    evidence.append(source_name)
            continue
        if normalized and normalized not in evidence:
            evidence.append(normalized)
    return evidence


def _is_targeted_mollogp_rule(rule: Dict) -> bool:
    return root_feature_key_for_canonical_rule(rule) == MOLLOGP_MERGE_SOURCE_REFINEMENT_ROOT


def _direction_bucket_for_rule(rule: Dict) -> str:
    relation = canonical_field_value(rule.get("relation_to_target")).lower()
    if relation in {"negative", "monotonic_down"}:
        return "down"
    if relation in {"positive", "monotonic_up"}:
        return "up"
    return relation or "unknown"


def _mode_bucket_for_rule(rule: Dict) -> str:
    mode = canonical_field_value(rule.get("feature_mode")).lower()
    if mode in {"continuous", "null"}:
        return "continuousish"
    if mode in {"binary", "threshold_binary"}:
        return "thresholded"
    return mode or "unknown"


def _mollogp_alias_group_key(rule: Dict) -> str:
    return "::".join(
        [
            root_feature_key_for_canonical_rule(rule),
            _direction_bucket_for_rule(rule),
            _mode_bucket_for_rule(rule),
        ]
    )


def _semantic_value_polarity(rule: Dict) -> str:
    family = canonical_field_value(rule.get("feature_family")).lower()
    relation = canonical_field_value(rule.get("relation_to_target")).lower()
    threshold = rule.get("threshold") or {}
    threshold_op = str(threshold.get("op", "")).strip().lower()
    mode_bucket = _mode_bucket_for_rule(rule)

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
    if mode_bucket == "range":
        if relation in {"positive", "monotonic_up"}:
            return "range_positive_signal"
        if relation in {"negative", "monotonic_down"}:
            return "range_negative_signal"
        return "range_unknown_signal"
    if relation in {"negative", "monotonic_down"}:
        return "low_value_signal"
    if relation in {"positive", "monotonic_up"}:
        return "high_value_signal"
    return "unknown_signal"


def _mode_buckets_compatible(lhs_rule: Dict, rhs_rule: Dict) -> bool:
    lhs_bucket = _mode_bucket_for_rule(lhs_rule)
    rhs_bucket = _mode_bucket_for_rule(rhs_rule)
    if lhs_bucket == rhs_bucket:
        return lhs_bucket != "unknown"
    return {lhs_bucket, rhs_bucket} == {"continuousish", "thresholded"}


def _rules_are_consensus_compatible(literature_rule: Dict, data_rule: Dict) -> bool:
    if root_feature_key_for_canonical_rule(literature_rule) != root_feature_key_for_canonical_rule(data_rule):
        return False
    literature_direction = _direction_bucket_for_rule(literature_rule)
    data_direction = _direction_bucket_for_rule(data_rule)
    if literature_direction == "unknown" or data_direction == "unknown":
        return False
    if literature_direction != data_direction:
        return False
    if not _mode_buckets_compatible(literature_rule, data_rule):
        return False
    literature_polarity = _semantic_value_polarity(literature_rule)
    data_polarity = _semantic_value_polarity(data_rule)
    if "unknown_signal" in {literature_polarity, data_polarity}:
        return False
    return literature_polarity == data_polarity


def _consensus_primary_sort_key(rule: Dict) -> tuple:
    source_name = str(rule.get("source", "") or rule.get("primary_source", "") or "").strip().lower()
    return (
        0 if bool(rule.get("compile_valid", False)) else 1,
        0 if source_name == "data" else 1,
        -_as_nonneg_int(rule.get("support_count"), 0),
        float(rule.get("p_value", 1.0) or 1.0),
        -abs(float(rule.get("effect_size", 0.0) or 0.0)),
        str(rule.get("canonical_key", "")),
    )


def _mollogp_dual_source_representative_sort_key(rule: Dict) -> tuple:
    sources = set(_rule_source_evidence(rule))
    return (
        0 if "literature" in sources and "data" not in sources else 1,
        *_consensus_primary_sort_key(rule),
    )


def _preserve_mollogp_dual_source_evidence(merged_rules: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for rule in merged_rules:
        if not _is_targeted_mollogp_rule(rule):
            continue
        grouped.setdefault(_mollogp_alias_group_key(rule), []).append(rule)

    for group_rules in grouped.values():
        if len(group_rules) < 2:
            continue
        group_sources = set()
        for rule in group_rules:
            group_sources.update(_rule_source_evidence(rule))
        if not {"literature", "data"}.issubset(group_sources):
            continue
        if any({"literature", "data"}.issubset(set(_rule_source_evidence(rule))) for rule in group_rules):
            continue
        representative = min(group_rules, key=_mollogp_dual_source_representative_sort_key)
        merged_support_sources = sorted(
            {
                source_name
                for rule in group_rules
                for source_name in (_incoming_support_sources(rule) or [])
                if source_name
            }
        )
        if not {"literature", "data"}.issubset(set(merged_support_sources)):
            merged_support_sources = sorted(set(merged_support_sources) | {"literature", "data"})
        representative["support_sources"] = merged_support_sources
        representative["source"] = "multi"
    return merged_rules


def build_multi_consensus_rule(literature_rule: Dict, data_rule: Dict) -> Dict | None:
    if not _rules_are_consensus_compatible(literature_rule, data_rule):
        return None
    primary_rule = min([literature_rule, data_rule], key=_consensus_primary_sort_key)
    secondary_rule = data_rule if primary_rule is literature_rule else literature_rule
    merged_source_votes = _normalize_counter(primary_rule.get("source_votes"))
    for source_name, votes in _normalize_counter(secondary_rule.get("source_votes")).items():
        merged_source_votes[source_name] = merged_source_votes.get(source_name, 0) + votes
    merged_extraction_votes = _normalize_counter(primary_rule.get("extraction_votes"))
    for source_name, votes in _normalize_counter(secondary_rule.get("extraction_votes")).items():
        merged_extraction_votes[source_name] = merged_extraction_votes.get(source_name, 0) + votes
    item = deepcopy(primary_rule)
    item["source"] = "multi"
    item["primary_source"] = "multi"
    item["support_sources"] = sorted(set(_incoming_support_sources(literature_rule)) | set(_incoming_support_sources(data_rule)))
    item["source_votes"] = merged_source_votes or {"literature": 1, "data": 1}
    item["support_count"] = sum(item["source_votes"].values())
    item["extraction_votes"] = merged_extraction_votes
    item["projected_support_count"] = (
        _as_nonneg_int(literature_rule.get("projected_support_count"), 0)
        + _as_nonneg_int(data_rule.get("projected_support_count"), 0)
    )
    item["heuristic_support_count"] = (
        _as_nonneg_int(literature_rule.get("heuristic_support_count"), 0)
        + _as_nonneg_int(data_rule.get("heuristic_support_count"), 0)
    )
    item["support_models"] = sorted(set(_incoming_support_models(literature_rule)) | set(_incoming_support_models(data_rule)))
    item["provenance_list"] = _incoming_provenance_list(literature_rule) + _incoming_provenance_list(data_rule)
    item["consensus_promoted"] = True
    item["promotion_type"] = "cross_source_consensus"
    item["promoted_root_feature_key"] = root_feature_key_for_canonical_rule(item)
    item["consensus_direction_bucket"] = _direction_bucket_for_rule(primary_rule)
    item["consensus_mode_bucket"] = _mode_bucket_for_rule(primary_rule)
    item["promoted_from_canonical_keys"] = sorted(
        {
            str(literature_rule.get("canonical_key", "")).strip(),
            str(data_rule.get("canonical_key", "")).strip(),
        }
    )
    item["promotion_lineage"] = {
        "literature_canonical_key": str(literature_rule.get("canonical_key", "")).strip(),
        "data_canonical_key": str(data_rule.get("canonical_key", "")).strip(),
        "primary_canonical_key": str(primary_rule.get("canonical_key", "")).strip(),
        "secondary_canonical_key": str(secondary_rule.get("canonical_key", "")).strip(),
    }
    if len(str(secondary_rule.get("rationale", ""))) > len(str(item.get("rationale", ""))):
        item["rationale"] = secondary_rule.get("rationale")
    return item


def attach_multi_promotion_audit(rules: List[Dict], promoted_rules: List[Dict]) -> List[Dict]:
    if not promoted_rules:
        return rules
    audited_rules = list(rules)
    promoted_root_keys = {
        str(rule.get("promoted_root_feature_key", "")).strip()
        for rule in promoted_rules
        if str(rule.get("promoted_root_feature_key", "")).strip()
    }
    for rule in audited_rules:
        root_key = root_feature_key_for_canonical_rule(rule)
        rule["cross_source_consensus_root"] = root_key in promoted_root_keys
    return audited_rules + promoted_rules


def _promote_cross_source_consensus_rules(rules: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for rule in rules:
        grouped.setdefault(root_feature_key_for_canonical_rule(rule), []).append(rule)

    promoted_rules: List[Dict] = []
    for root_key, group_rules in grouped.items():
        if any(bool(rule.get("consensus_promoted", False)) for rule in group_rules):
            continue
        literature_rules = [
            rule for rule in group_rules if "literature" in _rule_source_evidence(rule) and "data" not in _rule_source_evidence(rule)
        ]
        data_rules = [
            rule for rule in group_rules if "data" in _rule_source_evidence(rule) and "literature" not in _rule_source_evidence(rule)
        ]
        existing_multi = any("literature" in _rule_source_evidence(rule) and "data" in _rule_source_evidence(rule) for rule in group_rules)
        if not literature_rules or not data_rules or existing_multi:
            continue
        compatible_pairs = []
        for literature_rule in literature_rules:
            for data_rule in data_rules:
                promoted = build_multi_consensus_rule(literature_rule, data_rule)
                if promoted is not None:
                    compatible_pairs.append(promoted)
        if not compatible_pairs:
            continue
        promoted_rules.append(min(compatible_pairs, key=_consensus_primary_sort_key))
    return attach_multi_promotion_audit(rules, promoted_rules)


def detect_cross_source_consensus_groups(canonical_rules: Iterable[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for rule in canonical_rules:
        root_key = root_feature_key_for_canonical_rule(rule)
        sources = _rule_source_evidence(rule)
        direction_bucket = canonical_field_value(rule.get("relation_to_target")).lower()
        mode_bucket = canonical_field_value(rule.get("feature_mode")).lower()
        entry = grouped.setdefault(
            root_key,
            {
                "root_feature_key": root_key,
                "feature_family": canonical_field_value(rule.get("feature_family")),
                "feature_name": canonical_field_value(rule.get("feature_name")),
                "total_rule_count": 0,
                "literature_rule_count": 0,
                "data_rule_count": 0,
                "multi_rule_count": 0,
                "compile_valid_known_count": 0,
                "compile_valid_true_count": 0,
                "direction_buckets": {},
                "mode_buckets": {},
                "canonical_keys": [],
                "_direction_sources": {},
                "_mode_sources": {},
            },
        )
        entry["total_rule_count"] += 1
        if "literature" in sources:
            entry["literature_rule_count"] += 1
        if "data" in sources:
            entry["data_rule_count"] += 1
        if "literature" in sources and "data" in sources:
            entry["multi_rule_count"] += 1
        if "compile_valid" in rule:
            entry["compile_valid_known_count"] += 1
            if bool(rule.get("compile_valid", False)):
                entry["compile_valid_true_count"] += 1
        entry["direction_buckets"][direction_bucket] = entry["direction_buckets"].get(direction_bucket, 0) + 1
        entry["mode_buckets"][mode_bucket] = entry["mode_buckets"].get(mode_bucket, 0) + 1
        entry["_direction_sources"].setdefault(direction_bucket, set()).update(sources)
        entry["_mode_sources"].setdefault(mode_bucket, set()).update(sources)
        canonical_key = str(rule.get("canonical_key", "")).strip()
        if canonical_key and canonical_key not in entry["canonical_keys"]:
            entry["canonical_keys"].append(canonical_key)

    audited_groups: List[Dict] = []
    for entry in grouped.values():
        shared_direction_buckets = sorted(
            bucket
            for bucket, sources in entry["_direction_sources"].items()
            if "literature" in sources and "data" in sources
        )
        shared_mode_buckets = sorted(
            bucket
            for bucket, sources in entry["_mode_sources"].items()
            if "literature" in sources and "data" in sources
        )
        audited_groups.append(
            {
                key: value
                for key, value in entry.items()
                if not key.startswith("_")
            }
            | {
                "has_cross_source_overlap": bool(
                    entry["literature_rule_count"] > 0 and entry["data_rule_count"] > 0
                ),
                "shared_direction_buckets": shared_direction_buckets,
                "shared_mode_buckets": shared_mode_buckets,
                "shared_direction_bucket_count": len(shared_direction_buckets),
                "shared_mode_bucket_count": len(shared_mode_buckets),
            }
        )
    return sorted(
        audited_groups,
        key=lambda item: (
            -int(item.get("has_cross_source_overlap", False)),
            -int(item.get("compile_valid_true_count", 0)),
            -int(item.get("multi_rule_count", 0)),
            item.get("root_feature_key", ""),
        ),
    )


def merge_rules(rule_iterable: Iterable[Dict], *, multi_consensus_promotion_enable: bool = False) -> List[Dict]:
    if type(multi_consensus_promotion_enable) is not bool:
        raise ValueError("multi_consensus_promotion_enable must be a boolean")
    merged: Dict[str, Dict] = {}
    for rule in rule_iterable:
        key = canonical_rule_key(rule)
        incoming_support = _incoming_support(rule)
        incoming_source_votes = _incoming_source_votes(rule)
        incoming_extraction_votes = _incoming_extraction_votes(rule)
        incoming_projected_support = _as_nonneg_int(
            rule.get("projected_support_count"),
            incoming_extraction_votes.get("projected", 0),
        )
        incoming_heuristic_support = _as_nonneg_int(
            rule.get("heuristic_support_count"),
            incoming_extraction_votes.get("heuristic", 0),
        )
        incoming_support_models = _incoming_support_models(rule)
        incoming_support_sources = _incoming_support_sources(rule)
        incoming_provenance_list = _incoming_provenance_list(rule)
        if key not in merged:
            item = deepcopy(rule)
            item["canonical_key"] = key
            item["support_count"] = incoming_support
            item["source_votes"] = deepcopy(incoming_source_votes)
            item["extraction_votes"] = deepcopy(incoming_extraction_votes)
            item["projected_support_count"] = incoming_projected_support
            item["heuristic_support_count"] = incoming_heuristic_support
            item["support_models"] = incoming_support_models
            item["support_sources"] = incoming_support_sources
            item["provenance_list"] = incoming_provenance_list
            merged[key] = item
            continue
        item = merged[key]
        item["support_count"] = _as_nonneg_int(item.get("support_count"), 0) + incoming_support
        item.setdefault("source_votes", {})
        for source_name, votes in incoming_source_votes.items():
            item["source_votes"][source_name] = item["source_votes"].get(source_name, 0) + votes
        item.setdefault("extraction_votes", {})
        for extraction_source, votes in incoming_extraction_votes.items():
            item["extraction_votes"][extraction_source] = item["extraction_votes"].get(extraction_source, 0) + votes
        item["projected_support_count"] = (
            _as_nonneg_int(item.get("projected_support_count"), 0) + incoming_projected_support
        )
        item["heuristic_support_count"] = (
            _as_nonneg_int(item.get("heuristic_support_count"), 0) + incoming_heuristic_support
        )
        item["support_models"] = sorted(set(item.get("support_models", [])) | set(incoming_support_models))
        item["support_sources"] = sorted(set(item.get("support_sources", [])) | set(incoming_support_sources))
        item.setdefault("provenance_list", []).extend(incoming_provenance_list)
        if len(str(rule.get("rationale", ""))) > len(str(item.get("rationale", ""))):
            item["rationale"] = rule.get("rationale")
    merged_rules = list(merged.values())
    merged_rules = _preserve_mollogp_dual_source_evidence(merged_rules)
    if multi_consensus_promotion_enable:
        merged_rules = _promote_cross_source_consensus_rules(merged_rules)
    for item in merged_rules:
        source_votes = item.get("source_votes", {})
        if source_votes:
            item["primary_source"] = sorted(source_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        extraction_votes = item.get("extraction_votes", {})
        if extraction_votes:
            item["dominant_extraction_source"] = sorted(
                extraction_votes.items(), key=lambda kv: (-kv[1], kv[0])
            )[0][0]
            item["extraction_source"] = item["dominant_extraction_source"]
        if len(item.get("support_sources", [])) > 1 or len(item.get("source_votes", {})) > 1:
            item["source"] = "multi"
        elif item.get("support_sources"):
            item["source"] = item["support_sources"][0]
        elif item.get("primary_source"):
            item["source"] = item["primary_source"]
    return merged_rules
