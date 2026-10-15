"""Eligibility, assessment, and thinning helpers for Stage 2 variants."""

import json
from typing import Callable, Dict, Sequence

from morace.stage2_protocol import (
    DEFAULT_CHAT_NATIVE_CORE_TARGET_THINNING_POLICY,
    DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
)


def chat_native_core_independent_single_enabled(
    stage2_protocol_profile: str,
    source: str,
    *,
    termination_policy: str,
    rule_count_policy: str,
) -> bool:
    return (
        str(stage2_protocol_profile or "").strip() == "chat_native_core"
        and str(source or "").strip() == "literature"
        and termination_policy == "pure_json_single_rule"
        and rule_count_policy == "two_independent_single_rule_projections"
    )


def chat_native_core_family_target_enabled(
    stage2_protocol_profile: str,
    source: str,
    *,
    family_target_policy: str,
) -> bool:
    return (
        str(stage2_protocol_profile or "").strip() == "chat_native_core"
        and str(source or "").strip() == "literature"
        and family_target_policy == "family_targeted_single_rule_variants"
    )


def stage2_family_target_enabled(
    stage2_protocol_profile: str,
    source: str,
    *,
    chat_native_family_target_policy: str,
    native_family_target_policy: str,
) -> bool:
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "chat_native_core":
        return chat_native_core_family_target_enabled(
            profile,
            source,
            family_target_policy=chat_native_family_target_policy,
        )
    return (
        profile == "chemdfm_native"
        and str(source or "").strip() == "literature"
        and native_family_target_policy == "family_targeted_single_rule_variants"
    )


def family_target_candidate_assessment(
    rules: Sequence[Dict],
    target_family: str,
    existing_root_keys: Sequence[str],
    *,
    effective_strict_valid: bool,
    target_vocab_policy: str = DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
    target_vocab_options: Sequence[str] | None = None,
    rule_root_feature_key: Callable[[Dict], str],
    feature_families: Callable[[Sequence[Dict]], list[str]],
    compile_valid_ratio: Callable[[Sequence[Dict]], float],
) -> Dict:
    target_family = str(target_family or "").strip()
    target_vocab_policy = str(target_vocab_policy or DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY).strip()
    target_vocab_options = list(target_vocab_options or [])
    candidate_rules = list(rules or [])
    root_keys = sorted(
        {
            rule_root_feature_key(rule)
            for rule in candidate_rules
            if rule.get("feature_family") or rule.get("feature_name")
        }
    )
    target_feature_families = feature_families(candidate_rules)
    target_family_match = bool(candidate_rules) and all(
        str(rule.get("feature_family", "") or "").strip() == target_family for rule in candidate_rules
    )
    multi_rule_violation = len(candidate_rules) > 1
    off_target_family_drop = bool(candidate_rules) and not target_family_match
    compile_valid = compile_valid_ratio(candidate_rules) > 0.0
    duplicate_root = bool(set(root_keys) & set(existing_root_keys or []))
    target_feature_name = ""
    if len(candidate_rules) == 1:
        target_feature_name = str(candidate_rules[0].get("feature_name", "") or "").strip()
    target_vocab_match = True
    target_vocab_drop_reason = ""
    if target_vocab_policy == "micro_canonical_options":
        target_vocab_match = bool(len(candidate_rules) == 1 and target_feature_name in set(target_vocab_options))
        if not target_vocab_match:
            target_vocab_drop_reason = "feature_name_not_in_target_micro_vocab"
    kept = bool(
        effective_strict_valid
        and candidate_rules
        and not multi_rule_violation
        and target_family_match
        and target_vocab_match
        and compile_valid
        and not duplicate_root
    )
    if kept:
        drop_reason = ""
    elif not effective_strict_valid:
        drop_reason = "strict_invalid"
    elif not candidate_rules:
        drop_reason = "empty_rules"
    elif multi_rule_violation:
        drop_reason = "multi_rule_target_violation"
    elif off_target_family_drop:
        drop_reason = "off_target_family"
    elif target_vocab_policy == "micro_canonical_options" and not target_vocab_match:
        drop_reason = "feature_name_not_in_target_micro_vocab"
    elif not compile_valid:
        drop_reason = "compile_invalid"
    elif duplicate_root:
        drop_reason = "duplicate_root"
    else:
        drop_reason = "unknown"
    return {
        "family_target_variant_kept": kept,
        "family_target_drop_reason": drop_reason,
        "target_family_match": target_family_match,
        "off_target_family_drop": off_target_family_drop,
        "multi_rule_target_violation": multi_rule_violation,
        "target_root_keys": root_keys,
        "target_feature_families": target_feature_families,
        "target_compile_valid": compile_valid,
        "target_vocab_policy": target_vocab_policy,
        "target_vocab_options": list(target_vocab_options),
        "target_vocab_match": bool(target_vocab_match),
        "target_vocab_drop_reason": target_vocab_drop_reason if drop_reason == target_vocab_drop_reason else "",
        "target_feature_name": target_feature_name,
    }


def target_family_mode_matches(target_family: str, feature_mode: str) -> bool:
    target_family = str(target_family or "").strip()
    feature_mode = str(feature_mode or "").strip()
    if target_family == "pattern_presence":
        return feature_mode == "binary_presence"
    if target_family == "count":
        return feature_mode == "count_value"
    if target_family == "descriptor":
        return feature_mode in {"numeric_value", "threshold_binary"}
    return False


def target_thinning_empty_result(policy: str, trigger_reason: str = "") -> Dict:
    return {
        "target_thinning_policy": str(policy or DEFAULT_CHAT_NATIVE_CORE_TARGET_THINNING_POLICY),
        "target_thinning_attempted": False,
        "target_thinning_success": False,
        "target_thinning_trigger_reason": str(trigger_reason or ""),
        "target_thinning_candidate_count": 0,
        "target_thinning_valid_candidate_count": 0,
        "target_thinning_selected_root": "",
        "target_thinning_selected_family": "",
        "target_thinning_selected_feature_name": "",
        "target_thinning_drop_reason": "",
        "target_thinning_rule_preview": "",
        "target_thinning_rules": [],
    }


def target_rule_for_strict_reparse(rule: Dict, source: str) -> Dict:
    return {
        "source": str(rule.get("source") or source or "literature"),
        "feature_family": str(rule.get("feature_family", "") or "").strip(),
        "feature_name": str(rule.get("feature_name", "") or "").strip(),
        "feature_mode": str(rule.get("feature_mode", "") or "").strip(),
        "relation_to_target": rule.get("relation_to_target"),
        "threshold": rule.get("threshold"),
        "rationale": str(rule.get("rationale", "") or "target rule").strip() or "target rule",
    }


def thin_first_valid_target_rule(
    rules: Sequence[Dict],
    target_family: str,
    target_vocab_options: Sequence[str],
    existing_root_keys: Sequence[str],
    *,
    source: str,
    provenance: Dict,
    thinning_policy: str,
    trigger_reason: str,
    compile_rule: Callable[[Dict], object | None],
    rule_root_feature_key: Callable[[Dict], str],
    strict_parser: Callable,
    preview_rules: Callable[[Sequence[Dict], int], str],
    family_mode_matches: Callable[[str, str], bool],
    empty_result: Callable[[str, str], Dict],
    rule_for_strict_reparse: Callable[[Dict, str], Dict],
) -> Dict:
    result = empty_result(thinning_policy, trigger_reason)
    if thinning_policy != "first_valid_target_rule":
        return result
    result["target_thinning_attempted"] = True
    candidate_rules = list(rules or [])
    result["target_thinning_candidate_count"] = len(candidate_rules)
    if not candidate_rules:
        result["target_thinning_drop_reason"] = (
            "strict_invalid_no_candidates" if trigger_reason == "strict_invalid" else "no_candidate_rules"
        )
        return result

    target_family = str(target_family or "").strip()
    target_vocab_set = {str(option) for option in (target_vocab_options or [])}
    existing_roots = {str(root) for root in (existing_root_keys or [])}
    strict_parse_failed = False
    best_drop_reason = ""
    drop_priority = {
        "multi_rule_target_violation": 3,
        "off_target_family": 4,
        "feature_name_not_in_target_micro_vocab": 5,
        "compile_invalid": 6,
        "duplicate_root": 7,
    }

    def remember_drop(reason: str) -> None:
        nonlocal best_drop_reason
        if not best_drop_reason or drop_priority.get(reason, 99) < drop_priority.get(best_drop_reason, 99):
            best_drop_reason = reason

    for rule in candidate_rules:
        if not isinstance(rule, dict):
            continue
        family = str(rule.get("feature_family", "") or "").strip()
        feature_name = str(rule.get("feature_name", "") or "").strip()
        feature_mode = str(rule.get("feature_mode", "") or "").strip()
        if family != target_family:
            remember_drop("off_target_family")
            continue
        if feature_name not in target_vocab_set:
            remember_drop("feature_name_not_in_target_micro_vocab")
            continue
        if not family_mode_matches(target_family, feature_mode):
            remember_drop("off_target_family")
            continue
        if compile_rule(rule) is None:
            remember_drop("compile_invalid")
            continue
        root_key = rule_root_feature_key(rule)
        if root_key in existing_roots:
            remember_drop("duplicate_root")
            continue
        result["target_thinning_valid_candidate_count"] += 1
        cleaned_rule = rule_for_strict_reparse(rule, source)
        cleaned_json = json.dumps({"rules": [cleaned_rule]}, ensure_ascii=False, separators=(",", ":"))
        parsed_rules, _ = strict_parser(cleaned_json, source=source, provenance=provenance)
        if not parsed_rules or len(parsed_rules) != 1:
            strict_parse_failed = True
            continue
        parsed_rule = parsed_rules[0]
        parsed_root = rule_root_feature_key(parsed_rule)
        result.update(
            {
                "target_thinning_success": True,
                "target_thinning_selected_root": parsed_root,
                "target_thinning_selected_family": str(parsed_rule.get("feature_family", "") or ""),
                "target_thinning_selected_feature_name": str(parsed_rule.get("feature_name", "") or ""),
                "target_thinning_drop_reason": "",
                "target_thinning_rule_preview": preview_rules(parsed_rules, limit=1),
                "target_thinning_rules": parsed_rules,
            }
        )
        return result

    if result["target_thinning_valid_candidate_count"] > 0 and strict_parse_failed:
        result["target_thinning_drop_reason"] = "strict_parse_after_thinning_failed"
    else:
        result["target_thinning_drop_reason"] = best_drop_reason or "unknown"
    return result
