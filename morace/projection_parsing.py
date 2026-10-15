"""Projection JSON extraction, canonicalization, and rule cleanup."""

import json
from typing import Dict, List

from morace.feature_vocabulary import d4_feature_name_allowlist, expected_family_for_feature


def _record_family_feature_sample(
    diagnostics: Dict,
    key: str,
    feature_name: str,
    original_family: str,
    original_mode: str,
    corrected_family: str,
    corrected_mode: str,
    reason: str,
) -> None:
    diagnostics[key] = int(diagnostics.get(key, 0) or 0) + 1
    sample_key = key.replace("_count", "_samples")
    samples = diagnostics.setdefault(sample_key, [])
    if len(samples) < 10:
        samples.append(
            {
                "feature_name": feature_name,
                "original_feature_family": original_family,
                "original_feature_mode": original_mode,
                "corrected_feature_family": corrected_family,
                "corrected_feature_mode": corrected_mode,
                "reason": reason,
            }
        )


def apply_family_feature_consistency(cleaned: Dict, diagnostics: Dict) -> Dict:
    if diagnostics.get("family_feature_policy") != "safe_family_feature_consistency":
        return cleaned
    feature_name = str(cleaned.get("feature_name", "") or "").strip()
    current_family = str(cleaned.get("feature_family", "") or "").strip()
    current_mode = str(cleaned.get("feature_mode", "") or "").strip()
    current_threshold = cleaned.get("threshold")
    expected_family = expected_family_for_feature(feature_name)
    if expected_family == "count":
        corrected_family = "count"
        corrected_mode = "count_value"
    elif expected_family == "pattern_presence":
        corrected_family = "pattern_presence"
        corrected_mode = "binary_presence"
    else:
        return cleaned
    if current_family == corrected_family and current_mode == corrected_mode:
        return cleaned

    reason = f"feature_name_in_{corrected_family}_registry"
    _record_family_feature_sample(
        diagnostics,
        "family_feature_corrected_count",
        feature_name,
        current_family,
        current_mode,
        corrected_family,
        corrected_mode,
        reason,
    )
    if current_family == "descriptor" and corrected_family == "count":
        diagnostics["descriptor_to_count_corrected_count"] = (
            int(diagnostics.get("descriptor_to_count_corrected_count", 0) or 0) + 1
        )
    if current_family == "descriptor" and corrected_family == "pattern_presence":
        diagnostics["descriptor_to_pattern_corrected_count"] = (
            int(diagnostics.get("descriptor_to_pattern_corrected_count", 0) or 0) + 1
        )
    if corrected_family == "count":
        diagnostics["count_feature_family_corrected_count"] = (
            int(diagnostics.get("count_feature_family_corrected_count", 0) or 0) + 1
        )
    if corrected_family == "pattern_presence":
        diagnostics["pattern_feature_family_corrected_count"] = (
            int(diagnostics.get("pattern_feature_family_corrected_count", 0) or 0) + 1
        )
    if current_family != corrected_family and corrected_family != "descriptor":
        diagnostics["non_descriptor_rescued_by_family_consistency_count"] = (
            int(diagnostics.get("non_descriptor_rescued_by_family_consistency_count", 0) or 0) + 1
        )
    if current_threshold is not None:
        diagnostics["family_feature_threshold_cleared_count"] = (
            int(diagnostics.get("family_feature_threshold_cleared_count", 0) or 0) + 1
        )
        samples = diagnostics.setdefault("family_feature_threshold_cleared_samples", [])
        if len(samples) < 10:
            samples.append(
                {
                    "feature_name": feature_name,
                    "original_feature_family": current_family,
                    "original_feature_mode": current_mode,
                    "original_threshold": current_threshold,
                    "corrected_feature_family": corrected_family,
                    "corrected_feature_mode": corrected_mode,
                    "corrected_threshold": None,
                    "reason": reason,
                }
            )
    cleaned["feature_family"] = corrected_family
    cleaned["feature_mode"] = corrected_mode
    cleaned["threshold"] = None
    return cleaned


SAFE_CANONICALIZATION_FORBIDDEN_FEATURE_ALIASES = {
    "phenolic_or_carboxylic group",
    "phenolic_or_carboxylic_group",
    "phenolic oh group",
    "phenolic_oh_group",
    "hydrogen-bonding groups",
    "hydrogen bonding groups",
    "hydrogen_bonding_groups",
    "hydrogenbondinggroups",
    "hydroxyl",
    "thiophene_ring",
    "thiophene ring",
}


def _record_canonicalization_sample(
    diagnostics: Dict,
    key: str,
    field: str,
    original: str,
    canonical: str,
    reason: str,
    family: str = "",
) -> None:
    diagnostics[key] = int(diagnostics.get(key, 0) or 0) + 1
    sample_key = key.replace("_count", "_samples")
    samples = diagnostics.setdefault(sample_key, [])
    if len(samples) < 10:
        samples.append(
            {
                "field": field,
                "original_value": original,
                "canonical_value": canonical,
                "reason": reason,
                "family": family,
            }
        )


def normalize_enum_candidate(value: str) -> str:
    raw = str(value or "").strip()
    chars = []
    previous_lower = False
    for char in raw:
        if char in {" ", "-", "/"}:
            chars.append("_")
            previous_lower = False
        elif char.isupper() and previous_lower:
            chars.append("_")
            chars.append(char.lower())
            previous_lower = False
        else:
            chars.append(char.lower())
            previous_lower = char.islower() or char.isdigit()
    return "_".join(part for part in "".join(chars).split("_") if part)


def canonicalize_enum_value(value: object, allowed: set[str], diagnostics: Dict, field: str) -> str | None:
    raw = str(value or "").strip()
    if raw in allowed:
        return raw
    if diagnostics.get("canonicalization_policy") != "safe_enum_feature_canonicalization":
        return None
    candidate = normalize_enum_candidate(raw)
    if candidate in allowed:
        family = candidate if field == "feature_family" else str(diagnostics.get("_active_feature_family", "") or "")
        _record_canonicalization_sample(
            diagnostics,
            "enum_canonicalized_count",
            field,
            raw,
            candidate,
            "enum_case_space_normalized",
            family,
        )
        return candidate
    return None


def feature_name_candidates(raw: str) -> list[tuple[str, str]]:
    stripped = str(raw or "").strip()
    compact = "".join(stripped.split())
    snake = "_".join(part for part in stripped.replace("-", " ").split() if part).lower()
    return [
        (stripped, "exact"),
        (compact, "remove_spaces_exact_match"),
        (snake, "space_to_snake_case_exact_match"),
    ]


def canonicalize_feature_name(value: object, family: str, diagnostics: Dict) -> str | None:
    raw = str(value or "").strip()
    allowed = d4_feature_name_allowlist(family)
    if raw in allowed:
        return raw
    if diagnostics.get("canonicalization_policy") != "safe_enum_feature_canonicalization":
        return None
    raw_norms = {raw.lower(), normalize_enum_candidate(raw), "".join(raw.split()).lower()}
    if raw_norms & SAFE_CANONICALIZATION_FORBIDDEN_FEATURE_ALIASES:
        diagnostics["canonicalization_failed_count"] = int(diagnostics.get("canonicalization_failed_count", 0) or 0) + 1
        samples = diagnostics.setdefault("canonicalization_failed_samples", [])
        if len(samples) < 10:
            samples.append(
                {
                    "field": "feature_name",
                    "original_value": raw,
                    "canonical_value": "",
                    "reason": "semantic_alias_mapping_forbidden",
                    "family": family,
                }
            )
        return None
    lower_map = {name.lower(): name for name in allowed}
    for candidate, reason in feature_name_candidates(raw):
        if candidate in allowed:
            _record_canonicalization_sample(
                diagnostics, "feature_name_canonicalized_count", "feature_name", raw, candidate, reason, family
            )
            return candidate
        lowered = candidate.lower()
        if lowered in lower_map:
            canonical = lower_map[lowered]
            _record_canonicalization_sample(
                diagnostics, "feature_name_canonicalized_count", "feature_name", raw, canonical, reason, family
            )
            return canonical
    diagnostics["canonicalization_failed_count"] = int(diagnostics.get("canonicalization_failed_count", 0) or 0) + 1
    samples = diagnostics.setdefault("canonicalization_failed_samples", [])
    if len(samples) < 10:
        samples.append(
            {
                "field": "feature_name",
                "original_value": raw,
                "canonical_value": "",
                "reason": "no_exact_canonical_feature_name_match",
                "family": family,
            }
        )
    return None


def extract_first_balanced_json_object(text: str) -> tuple[str | None, Dict]:
    raw = str(text or "")
    audit = {
        "json_start_index": -1,
        "json_end_index": -1,
        "prefix_removed": False,
        "suffix_removed": False,
        "balanced_object_found": False,
        "multiple_json_objects_detected": False,
    }
    start = raw.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for idx in range(start, len(raw)):
            char = raw[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    prefix = raw[:start].strip()
                    suffix = raw[end:].strip()
                    audit.update(
                        {
                            "json_start_index": start,
                            "json_end_index": end,
                            "prefix_removed": bool(prefix),
                            "suffix_removed": bool(suffix),
                            "balanced_object_found": True,
                            "multiple_json_objects_detected": "{" in suffix,
                        }
                    )
                    return raw[start:end], audit
                if depth < 0:
                    break
        start = raw.find("{", start + 1)
    return None, audit


def find_rules_array_start(text: str) -> tuple[int, Dict]:
    raw = str(text or "")
    audit = {
        "rules_array_prefix_found": False,
        "rules_array_start_index": -1,
        "rules_array_open_bracket_index": -1,
    }
    token = '"rules"'
    search_from = 0
    while True:
        key_index = raw.find(token, search_from)
        if key_index < 0:
            return -1, audit
        idx = key_index + len(token)
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw) or raw[idx] != ":":
            search_from = key_index + 1
            continue
        idx += 1
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx < len(raw) and raw[idx] == "[":
            audit.update(
                {
                    "rules_array_prefix_found": True,
                    "rules_array_start_index": key_index,
                    "rules_array_open_bracket_index": idx,
                }
            )
            return idx + 1, audit
        search_from = key_index + 1


def extract_balanced_objects_from_array_prefix(text: str, start_index: int) -> tuple[List[str], Dict]:
    raw = str(text or "")
    idx = max(0, int(start_index or 0))
    objects: List[str] = []
    audit = {
        "candidate_object_count": 0,
        "array_closed": False,
        "stopped_on_unclosed_object": False,
    }
    while idx < len(raw):
        next_object = raw.find("{", idx)
        next_array_close = raw.find("]", idx)
        if next_array_close >= 0 and (next_object < 0 or next_array_close < next_object):
            audit["array_closed"] = True
            break
        if next_object < 0:
            break
        depth = 0
        in_string = False
        escaped = False
        end = -1
        for cursor in range(next_object, len(raw)):
            char = raw[cursor]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = cursor + 1
                    break
                if depth < 0:
                    break
        if end < 0:
            audit["stopped_on_unclosed_object"] = True
            break
        objects.append(raw[next_object:end])
        audit["candidate_object_count"] = len(objects)
        idx = end
    return objects, audit


def new_extraction_diagnostics(
    policy: str,
    canonicalization_policy: str = "strict_canonical_only",
    family_feature_policy: str = "strict_family",
) -> Dict:
    return {
        "extraction_policy": policy,
        "canonicalization_policy": canonicalization_policy,
        "family_feature_policy": family_feature_policy,
        "d4_extraction_method": "none",
        "d4_extraction_used": False,
        "d4_extraction_reason": "",
        "original_strict_valid": False,
        "strict_after_extraction_valid": False,
        "raw_first_json_object_found": False,
        "raw_first_json_parseable": False,
        "d4_any_json_fragment_parseable": False,
        "json_object_closes_before_suffix": False,
        "prefix_removed": False,
        "suffix_removed": False,
        "multiple_json_objects_detected": False,
        "rules_before_extraction": 0,
        "rules_after_extraction": 0,
        "invalid_field_dropped_count": 0,
        "invalid_fields_dropped": [],
        "invalid_enum_dropped_count": 0,
        "unsupported_feature_name_dropped_count": 0,
        "missing_required_field_dropped_count": 0,
        "relation_defaulted_to_null_count": 0,
        "source_overwritten_count": 0,
        "threshold_invalid_dropped_count": 0,
        "enum_canonicalized_count": 0,
        "enum_canonicalized_samples": [],
        "feature_name_canonicalized_count": 0,
        "feature_name_canonicalized_samples": [],
        "canonicalization_failed_count": 0,
        "canonicalization_failed_samples": [],
        "non_descriptor_canonicalized_count": 0,
        "non_descriptor_rescued_by_canonicalization_count": 0,
        "family_feature_corrected_count": 0,
        "family_feature_corrected_samples": [],
        "family_feature_correction_failed_count": 0,
        "family_feature_correction_failed_samples": [],
        "family_feature_threshold_cleared_count": 0,
        "family_feature_threshold_cleared_samples": [],
        "count_feature_family_corrected_count": 0,
        "pattern_feature_family_corrected_count": 0,
        "descriptor_to_count_corrected_count": 0,
        "descriptor_to_pattern_corrected_count": 0,
        "non_descriptor_rescued_by_family_consistency_count": 0,
        "cleaned_json_preview": "",
        "d4_rules_array_salvage_used": False,
        "d4_rules_array_salvage_attempted": False,
        "d4_rules_array_prefix_found": False,
        "d4_rules_array_candidate_object_count": 0,
        "d4_rules_array_parseable_object_count": 0,
        "d4_rules_array_cleaned_rule_count": 0,
        "d4_rules_array_salvage_strict_valid": False,
        "d4_rules_array_salvage_reason": "",
        "d4_rules_array_cleaned_json_preview": "",
    }


def clean_projected_rule(rule: Dict, source: str, diagnostics: Dict) -> Dict | None:
    allowed_fields = {"feature_family", "feature_name", "feature_mode", "relation_to_target", "threshold", "rationale", "source"}
    cleaned = {key: value for key, value in (rule or {}).items() if key in allowed_fields}
    dropped = sorted(str(key) for key in (set((rule or {}).keys()) - allowed_fields))
    diagnostics["invalid_field_dropped_count"] += len(dropped)
    diagnostics.setdefault("invalid_fields_dropped", []).extend(dropped)

    raw_family = str(cleaned.get("feature_family", "") or "").strip()
    raw_feature_name = str(cleaned.get("feature_name", "") or "").strip()
    raw_mode = str(cleaned.get("feature_mode", "") or "").strip()
    pre_family_valid = raw_family in {"descriptor", "pattern_presence", "count"}
    pre_mode_valid = raw_mode in {"numeric_value", "threshold_binary", "binary_presence", "count_value"}
    pre_feature_valid = raw_feature_name in d4_feature_name_allowlist(raw_family) if pre_family_valid else False
    before_enum_canonicalized = int(diagnostics.get("enum_canonicalized_count", 0) or 0)
    before_feature_canonicalized = int(diagnostics.get("feature_name_canonicalized_count", 0) or 0)
    family = canonicalize_enum_value(cleaned.get("feature_family"), {"descriptor", "pattern_presence", "count"}, diagnostics, "feature_family")
    diagnostics["_active_feature_family"] = family or raw_family
    mode = canonicalize_enum_value(
        cleaned.get("feature_mode"),
        {"numeric_value", "threshold_binary", "binary_presence", "count_value"},
        diagnostics,
        "feature_mode",
    )
    if not raw_family or not raw_feature_name or not raw_mode:
        diagnostics["missing_required_field_dropped_count"] += 1
        return None
    if not family or not mode:
        diagnostics["invalid_enum_dropped_count"] += 1
        return None
    feature_name = canonicalize_feature_name(cleaned.get("feature_name"), family, diagnostics)
    if not feature_name and diagnostics.get("family_feature_policy") == "safe_family_feature_consistency":
        for alternate_family in ("count", "pattern_presence", "descriptor"):
            if alternate_family == family:
                continue
            alternate_feature_name = canonicalize_feature_name(
                cleaned.get("feature_name"), alternate_family, diagnostics
            )
            if alternate_feature_name:
                feature_name = alternate_feature_name
                break
    if not feature_name:
        diagnostics["unsupported_feature_name_dropped_count"] += 1
        return None
    canonicalized_this_rule = (
        int(diagnostics.get("enum_canonicalized_count", 0) or 0) > before_enum_canonicalized
        or int(diagnostics.get("feature_name_canonicalized_count", 0) or 0) > before_feature_canonicalized
    )
    would_drop_before = not (pre_family_valid and pre_mode_valid and pre_feature_valid)
    if canonicalized_this_rule and family != "descriptor":
        diagnostics["non_descriptor_canonicalized_count"] = (
            int(diagnostics.get("non_descriptor_canonicalized_count", 0) or 0) + 1
        )
        if would_drop_before:
            diagnostics["non_descriptor_rescued_by_canonicalization_count"] = (
                int(diagnostics.get("non_descriptor_rescued_by_canonicalization_count", 0) or 0) + 1
            )
    cleaned["feature_family"] = family
    cleaned["feature_mode"] = mode
    cleaned["feature_name"] = feature_name
    cleaned = apply_family_feature_consistency(cleaned, diagnostics)
    family = str(cleaned.get("feature_family", "") or "").strip()
    mode = str(cleaned.get("feature_mode", "") or "").strip()

    relation = cleaned.get("relation_to_target")
    if "relation_to_target" not in cleaned:
        cleaned["relation_to_target"] = None
        diagnostics["relation_defaulted_to_null_count"] += 1
    elif relation is not None:
        canonical_relation = canonicalize_enum_value(
            relation,
            {"positive", "negative", "monotonic_up", "monotonic_down"},
            diagnostics,
            "relation_to_target",
        )
        if not canonical_relation:
            diagnostics["invalid_enum_dropped_count"] += 1
            return None
        cleaned["relation_to_target"] = canonical_relation

    if mode == "threshold_binary":
        threshold = cleaned.get("threshold")
        if not isinstance(threshold, dict):
            diagnostics["threshold_invalid_dropped_count"] += 1
            return None
        op = threshold.get("op")
        value = threshold.get("value")
        if op not in {">", ">=", "<", "<="} or not isinstance(value, (int, float)):
            diagnostics["threshold_invalid_dropped_count"] += 1
            return None
    else:
        cleaned["threshold"] = None

    pipeline_source = str(source or "").strip()
    model_source = str(cleaned.get("source", "") or "").strip()
    if pipeline_source and model_source and model_source != pipeline_source:
        diagnostics["source_overwritten_count"] += 1
    cleaned["source"] = str(pipeline_source or model_source).strip()
    return cleaned


def extract_cleaned_rule_json(
    raw_output: str,
    source: str,
    policy: str,
    canonicalization_policy: str = "strict_canonical_only",
    family_feature_policy: str = "strict_family",
) -> tuple[str, Dict]:
    diagnostics = new_extraction_diagnostics(policy, canonicalization_policy, family_feature_policy)
    diagnostics["d4_extraction_used"] = True
    diagnostics["d4_extraction_method"] = "first_balanced_json_object"
    json_text, extract_audit = extract_first_balanced_json_object(raw_output)
    diagnostics["raw_first_json_object_found"] = bool(json_text)
    diagnostics["prefix_removed"] = bool(extract_audit.get("prefix_removed"))
    diagnostics["suffix_removed"] = bool(extract_audit.get("suffix_removed"))
    diagnostics["json_object_closes_before_suffix"] = bool(extract_audit.get("suffix_removed"))
    diagnostics["multiple_json_objects_detected"] = bool(extract_audit.get("multiple_json_objects_detected"))
    if not json_text:
        diagnostics["d4_extraction_reason"] = "no_balanced_json_object"
        return "", diagnostics
    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError:
        diagnostics["d4_extraction_reason"] = "first_json_not_parseable"
        return "", diagnostics
    diagnostics["raw_first_json_parseable"] = True
    diagnostics["d4_any_json_fragment_parseable"] = True
    if not isinstance(obj, dict):
        diagnostics["d4_extraction_reason"] = "top_level_not_object"
        return "", diagnostics
    rules = obj.get("rules")
    if not isinstance(rules, list):
        diagnostics["d4_extraction_reason"] = "rules_missing_or_not_array"
        return "", diagnostics
    diagnostics["rules_before_extraction"] = len(rules)
    cleaned_rules: List[Dict] = []
    for rule in rules:
        if not isinstance(rule, dict):
            diagnostics["missing_required_field_dropped_count"] += 1
            continue
        cleaned = clean_projected_rule(rule, source, diagnostics)
        if cleaned is not None:
            cleaned_rules.append(cleaned)
    diagnostics["rules_after_extraction"] = len(cleaned_rules)
    cleaned_json = json.dumps({"rules": cleaned_rules}, ensure_ascii=False, separators=(",", ":"))
    diagnostics["cleaned_json_preview"] = cleaned_json[:1200]
    diagnostics["d4_extraction_reason"] = "cleaned_rules_available" if cleaned_rules else "rules_empty_after_cleaning"
    return cleaned_json, diagnostics


def salvage_rules_array_prefix(
    raw_output: str,
    source: str,
    policy: str,
    canonicalization_policy: str = "strict_canonical_only",
    family_feature_policy: str = "strict_family",
) -> tuple[str, Dict]:
    diagnostics = new_extraction_diagnostics(policy, canonicalization_policy, family_feature_policy)
    diagnostics["d4_extraction_used"] = True
    diagnostics["d4_extraction_method"] = "rules_array_prefix_salvage"
    diagnostics["d4_rules_array_salvage_used"] = True
    diagnostics["d4_rules_array_salvage_attempted"] = True
    start_index, prefix_audit = find_rules_array_start(raw_output)
    diagnostics["d4_rules_array_prefix_found"] = bool(prefix_audit.get("rules_array_prefix_found"))
    if start_index < 0:
        diagnostics["d4_extraction_reason"] = "rules_array_prefix_not_found"
        diagnostics["d4_rules_array_salvage_reason"] = "rules_array_prefix_not_found"
        return "", diagnostics

    object_texts, object_audit = extract_balanced_objects_from_array_prefix(raw_output, start_index)
    diagnostics["d4_rules_array_candidate_object_count"] = int(object_audit.get("candidate_object_count", 0) or 0)
    if not object_texts:
        diagnostics["d4_extraction_reason"] = "rules_array_no_complete_rule_object"
        diagnostics["d4_rules_array_salvage_reason"] = "rules_array_no_complete_rule_object"
        return "", diagnostics

    parseable_rules: List[Dict] = []
    for object_text in object_texts:
        try:
            obj = json.loads(object_text)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            parseable_rules.append(obj)
    diagnostics["d4_rules_array_parseable_object_count"] = len(parseable_rules)
    diagnostics["d4_any_json_fragment_parseable"] = bool(parseable_rules)
    diagnostics["rules_before_extraction"] = len(parseable_rules)
    if not parseable_rules:
        diagnostics["d4_extraction_reason"] = "rules_array_objects_not_parseable"
        diagnostics["d4_rules_array_salvage_reason"] = "rules_array_objects_not_parseable"
        return "", diagnostics

    cleaned_rules: List[Dict] = []
    for rule in parseable_rules:
        cleaned = clean_projected_rule(rule, source, diagnostics)
        if cleaned is not None:
            cleaned_rules.append(cleaned)
    diagnostics["rules_after_extraction"] = len(cleaned_rules)
    diagnostics["d4_rules_array_cleaned_rule_count"] = len(cleaned_rules)
    cleaned_json = json.dumps({"rules": cleaned_rules}, ensure_ascii=False, separators=(",", ":"))
    diagnostics["cleaned_json_preview"] = cleaned_json[:1200]
    diagnostics["d4_rules_array_cleaned_json_preview"] = cleaned_json[:1200]
    if cleaned_rules:
        diagnostics["d4_extraction_reason"] = "rules_array_salvage_cleaned_rules_available"
        diagnostics["d4_rules_array_salvage_reason"] = "cleaned_rules_available"
    else:
        diagnostics["d4_extraction_reason"] = "rules_array_rules_empty_after_cleaning"
        diagnostics["d4_rules_array_salvage_reason"] = "rules_empty_after_cleaning"
    return cleaned_json, diagnostics
