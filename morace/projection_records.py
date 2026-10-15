"""Record builders for Stage 2 projection variants."""

from typing import Callable, Dict, Sequence


def build_base_projection_record(
    source_item: Dict,
    attempt: Dict,
    *,
    projected_rules: Sequence[Dict],
    heuristic_rules: Sequence[Dict],
    projection_status: str,
    projector_valid: bool,
    parse_meta: Dict,
    strict_valid: bool,
    original_strict_valid: bool,
    effective_strict_valid: bool,
    qwen_legacy_d2_effective_strict: bool,
    strict_failure: Dict,
    original_strict_failure: Dict,
    original_profile: Dict,
    retry_assessment: Dict,
    projector_model: str,
    projector_model_path: str,
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
    stage2_protocol_profile: str,
    stage2_protocol_profile_source: str,
    stage2_protocol_profile_version: str,
    termination_policy: str,
    rule_count_policy: str,
    family_target_policy: str,
    target_vocab_policy: str,
    target_thinning_policy: str,
    family_target_projection_used: bool,
    underdiverse_retry_used: bool,
    underdiverse_retry_success: bool,
    base_stage2_call_count: int,
    base_stage2_token_count: int,
    retry_stage2_call_count: int,
    retry_stage2_token_count: int,
    independent_single_projection_used: bool,
    independent_single_projection_kept: bool,
    independent_novel_roots: Sequence[str],
    default_stage2_backend: str,
    is_underdiverse_literature_projection: Callable[[Sequence[Dict]], bool],
) -> Dict:
    projected_rules = list(projected_rules)
    heuristic_rules = list(heuristic_rules)
    parsed_rules = projected_rules + heuristic_rules
    raw_output = attempt["raw_output"]
    repair_output = attempt["repair_output"]
    repair_used = bool(attempt["repair_used"])
    return {
        **source_item,
        "projection_variant_id": 0,
        "projector_model": projector_model,
        "projector_model_path": projector_model_path,
        "draft_family": draft_family,
        "projector_family": projector_family,
        "use_cross_model_projector": use_cross_model_projector,
        "stage2_protocol_profile": stage2_protocol_profile,
        "stage2_protocol_profile_source": stage2_protocol_profile_source,
        "stage2_protocol_profile_version": stage2_protocol_profile_version,
        "termination_policy": termination_policy,
        "rule_count_policy": rule_count_policy,
        "family_target_policy": family_target_policy,
        "target_vocab_policy": target_vocab_policy,
        "target_thinning_policy": target_thinning_policy,
        "family_target_enabled": bool(family_target_projection_used),
        "family_target_family": "",
        "family_target_variant_role": "",
        "family_target_variant_id": 0,
        "family_target_variant_kept": False,
        "family_target_drop_reason": "",
        "target_family_match": False,
        "off_target_family_drop": False,
        "multi_rule_target_violation": False,
        "target_root_keys": [],
        "target_feature_families": [],
        "target_vocab_options": [],
        "target_vocab_match": False,
        "target_vocab_drop_reason": "",
        "target_feature_name": "",
        "target_thinning_attempted": False,
        "target_thinning_success": False,
        "target_thinning_trigger_reason": "",
        "target_thinning_candidate_count": 0,
        "target_thinning_valid_candidate_count": 0,
        "target_thinning_selected_root": "",
        "target_thinning_selected_family": "",
        "target_thinning_selected_feature_name": "",
        "target_thinning_drop_reason": "",
        "target_thinning_rule_preview": "",
        "projection_variant_role": "base_single_rule",
        "projector_raw_output": raw_output.strip(),
        "projector_repair_output": repair_output.strip(),
        "repair_used": repair_used,
        "repair_success": bool(repair_used and projected_rules),
        "projection_status": projection_status,
        "strict_projector_valid": strict_valid,
        "original_strict_projector_valid": original_strict_valid,
        "effective_strict_projector_valid": effective_strict_valid,
        "qwen_legacy_d2_effective_strict": qwen_legacy_d2_effective_strict,
        "strict_failure": strict_failure,
        "original_strict_failure": original_strict_failure,
        "projector_valid": projector_valid,
        "chat_native_extraction_diagnostics": dict(
            attempt.get("chat_native_extraction_diagnostics", {}) or {}
        ),
        "d4_cleaned_json": str(attempt.get("d4_cleaned_json", "") or ""),
        "heuristic_rescued": bool(heuristic_rules),
        "underdiverse_retry_used": underdiverse_retry_used,
        "underdiverse_retry_success": underdiverse_retry_success,
        "underdiverse_projection": (
            source_item["source"] == "literature"
            and bool(projected_rules)
            and is_underdiverse_literature_projection(projected_rules)
        ),
        "underdiverse_trigger_reasons": list(original_profile.get("underdiverse_trigger_reasons", [])),
        "literature_projection_profile": original_profile,
        "projected_rule_count": len(projected_rules),
        "heuristic_rule_count": len(heuristic_rules),
        "projector_parsed_rules": projected_rules,
        "parsed_rules": parsed_rules,
        "schema_valid": bool(parsed_rules),
        "parse_meta": parse_meta,
        "stage2_backend": attempt.get("stage2_backend", default_stage2_backend),
        "stage2_avg_logprob": attempt.get("stage2_avg_logprob"),
        "stage2_logprob_norm": float(attempt.get("stage2_logprob_norm", 0.0) or 0.0),
        "stage2_generated_token_count": base_stage2_token_count + retry_stage2_token_count,
        "stage2_base_generated_token_count": base_stage2_token_count,
        "stage2_retry_generated_token_count": retry_stage2_token_count,
        "stage2_total_call_count": base_stage2_call_count + retry_stage2_call_count,
        "stage2_base_call_count": base_stage2_call_count,
        "stage2_retry_call_count": retry_stage2_call_count,
        "literature_recall_bonus": 0.0,
        "literature_recall_retry_candidate": bool(underdiverse_retry_used),
        "literature_recall_retry_kept": bool(underdiverse_retry_success),
        "literature_recall_retry_reason": (
            "novel_root_quality_preserving_retry"
            if underdiverse_retry_success
            else ("retry_not_quality_preserving" if underdiverse_retry_used else "")
        ),
        "literature_recall_novel_root_count": int(retry_assessment.get("novel_root_count", 0) or 0),
        "independent_single_projection_used": bool(independent_single_projection_used),
        "independent_single_projection_kept": bool(independent_single_projection_kept),
        "independent_single_projection_novel_root_count": int(len(independent_novel_roots)),
    }


def build_independent_projection_record(
    source_item: Dict,
    attempt: Dict,
    *,
    projection_kept: bool,
    novel_roots: Sequence[str],
    projector_model: str,
    projector_model_path: str,
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
    stage2_protocol_profile: str,
    stage2_protocol_profile_source: str,
    stage2_protocol_profile_version: str,
    termination_policy: str,
    rule_count_policy: str,
    default_stage2_backend: str,
    classify_projection_status: Callable[[str, Sequence[Dict], Sequence[Dict]], str],
    literature_projection_profile: Callable[[Sequence[Dict]], Dict],
    compile_valid_ratio: Callable[[Sequence[Dict]], float],
) -> Dict:
    projected_rules = list(attempt.get("projected_rules", []) or [])
    original_strict_valid = bool(attempt.get("strict_rules") or attempt.get("repair_strict_rules"))
    effective_strict_valid = bool(original_strict_valid or attempt.get("d4_strict_rules"))
    parse_meta = dict(attempt.get("parse_meta", {}) or {})
    repair_used = bool(attempt.get("repair_used"))
    failure_meta = (
        dict(attempt.get("repair_strict_parse_meta", {}) or {})
        if repair_used and not original_strict_valid
        else dict(attempt.get("strict_parse_meta", {}) or {})
    )
    parsed_rules = list(projected_rules) if projection_kept else []
    projection_profile = literature_projection_profile(projected_rules)
    generated_token_count = int(attempt.get("stage2_generated_token_count", 0) or 0)
    total_call_count = int(attempt.get("stage2_total_call_count", 0) or 0)
    return {
        **source_item,
        "projection_variant_id": 1,
        "projection_variant_role": "independent_single_rule_2",
        "projector_model": projector_model,
        "projector_model_path": projector_model_path,
        "draft_family": draft_family,
        "projector_family": projector_family,
        "use_cross_model_projector": use_cross_model_projector,
        "stage2_protocol_profile": stage2_protocol_profile,
        "stage2_protocol_profile_source": stage2_protocol_profile_source,
        "stage2_protocol_profile_version": stage2_protocol_profile_version,
        "termination_policy": termination_policy,
        "rule_count_policy": rule_count_policy,
        "projector_raw_output": str(attempt.get("raw_output", "")).strip(),
        "projector_repair_output": str(attempt.get("repair_output", "")).strip(),
        "repair_used": repair_used,
        "repair_success": bool(repair_used and projected_rules),
        "projection_status": classify_projection_status(
            attempt.get("raw_output", ""),
            projected_rules,
            [],
        ),
        "strict_projector_valid": effective_strict_valid,
        "original_strict_projector_valid": original_strict_valid,
        "effective_strict_projector_valid": effective_strict_valid,
        "strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not effective_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not effective_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not effective_strict_valid else "",
        },
        "original_strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not original_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not original_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not original_strict_valid else "",
        },
        "projector_valid": bool(projected_rules),
        "chat_native_extraction_diagnostics": dict(
            attempt.get("chat_native_extraction_diagnostics", {}) or {}
        ),
        "d4_cleaned_json": str(attempt.get("d4_cleaned_json", "") or ""),
        "heuristic_rescued": False,
        "underdiverse_retry_used": False,
        "underdiverse_retry_success": False,
        "underdiverse_projection": bool(projection_profile.get("underdiverse", False)),
        "underdiverse_trigger_reasons": list(projection_profile.get("underdiverse_trigger_reasons", [])),
        "literature_projection_profile": projection_profile,
        "projected_rule_count": len(projected_rules),
        "heuristic_rule_count": 0,
        "projector_parsed_rules": projected_rules,
        "parsed_rules": parsed_rules,
        "schema_valid": bool(parsed_rules),
        "parse_meta": parse_meta,
        "stage2_backend": attempt.get("stage2_backend", default_stage2_backend),
        "stage2_avg_logprob": attempt.get("stage2_avg_logprob"),
        "stage2_logprob_norm": float(attempt.get("stage2_logprob_norm", 0.0) or 0.0),
        "stage2_generated_token_count": generated_token_count,
        "stage2_base_generated_token_count": generated_token_count,
        "stage2_retry_generated_token_count": 0,
        "stage2_total_call_count": total_call_count,
        "stage2_base_call_count": total_call_count,
        "stage2_retry_call_count": 0,
        "literature_recall_bonus": 0.0,
        "literature_recall_retry_candidate": False,
        "literature_recall_retry_kept": False,
        "literature_recall_retry_reason": "",
        "literature_recall_novel_root_count": 0,
        "independent_single_projection_used": True,
        "independent_single_projection_kept": bool(projection_kept),
        "independent_single_rule_strict_valid": bool(effective_strict_valid),
        "independent_single_rule_compile_valid": bool(compile_valid_ratio(projected_rules) > 0.0),
        "independent_single_rule_novel_root_count": int(len(novel_roots)),
        "independent_single_rule_novel_root_keys": list(novel_roots),
        "selected_for_merge": bool(projection_kept),
        "selected_for_merge_reason": (
            "independent_single_rule_kept" if projection_kept else "independent_single_rule_rejected"
        ),
    }


def build_family_target_projection_record(
    source_item: Dict,
    target_info: Dict,
    *,
    projector_model: str,
    projector_model_path: str,
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
    stage2_protocol_profile: str,
    stage2_protocol_profile_source: str,
    stage2_protocol_profile_version: str,
    termination_policy: str,
    rule_count_policy: str,
    family_target_policy: str,
    target_vocab_policy: str,
    target_thinning_policy: str,
    default_stage2_backend: str,
    classify_projection_status: Callable[[str, Sequence[Dict], Sequence[Dict]], str],
    literature_projection_profile: Callable[[Sequence[Dict]], Dict],
) -> Dict:
    attempt = target_info["attempt"]
    target_family = str(target_info["target_family"])
    assessment = dict(target_info["assessment"])
    projected_rules = list(
        target_info.get("projected_rules_override") or attempt.get("projected_rules", []) or []
    )
    original_strict_valid = bool(target_info["original_strict_valid"])
    effective_strict_valid = bool(target_info["effective_strict_valid"])
    parse_meta = dict(attempt.get("parse_meta", {}) or {})
    repair_used = bool(attempt.get("repair_used"))
    failure_meta = (
        dict(attempt.get("repair_strict_parse_meta", {}) or {})
        if repair_used and not original_strict_valid
        else dict(attempt.get("strict_parse_meta", {}) or {})
    )
    parsed_rules = list(projected_rules) if assessment.get("family_target_variant_kept") else []
    projection_profile = literature_projection_profile(projected_rules)
    generated_token_count = int(attempt.get("stage2_generated_token_count", 0) or 0)
    total_call_count = int(attempt.get("stage2_total_call_count", 0) or 0)
    return {
        **source_item,
        "projection_variant_id": int(target_info["variant_id"]),
        "projection_variant_role": f"family_target_{target_family}",
        "projector_model": projector_model,
        "projector_model_path": projector_model_path,
        "draft_family": draft_family,
        "projector_family": projector_family,
        "use_cross_model_projector": use_cross_model_projector,
        "stage2_protocol_profile": stage2_protocol_profile,
        "stage2_protocol_profile_source": stage2_protocol_profile_source,
        "stage2_protocol_profile_version": stage2_protocol_profile_version,
        "termination_policy": termination_policy,
        "rule_count_policy": rule_count_policy,
        "family_target_policy": family_target_policy,
        "target_vocab_policy": target_vocab_policy,
        "target_thinning_policy": target_thinning_policy,
        "family_target_enabled": True,
        "family_target_family": target_family,
        "family_target_variant_role": f"family_target_{target_family}",
        "family_target_variant_id": int(target_info["variant_id"]),
        "projector_raw_output": str(attempt.get("raw_output", "")).strip(),
        "projector_repair_output": str(attempt.get("repair_output", "")).strip(),
        "repair_used": repair_used,
        "repair_success": bool(repair_used and projected_rules),
        "projection_status": classify_projection_status(
            attempt.get("raw_output", ""),
            projected_rules,
            [],
        ),
        "strict_projector_valid": effective_strict_valid,
        "original_strict_projector_valid": original_strict_valid,
        "effective_strict_projector_valid": effective_strict_valid,
        "strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not effective_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not effective_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not effective_strict_valid else "",
        },
        "original_strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not original_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not original_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not original_strict_valid else "",
        },
        "projector_valid": bool(projected_rules),
        "chat_native_extraction_diagnostics": dict(
            attempt.get("chat_native_extraction_diagnostics", {}) or {}
        ),
        "d4_cleaned_json": str(attempt.get("d4_cleaned_json", "") or ""),
        "heuristic_rescued": False,
        "underdiverse_retry_used": False,
        "underdiverse_retry_success": False,
        "underdiverse_projection": bool(projection_profile.get("underdiverse", False)),
        "underdiverse_trigger_reasons": list(projection_profile.get("underdiverse_trigger_reasons", [])),
        "literature_projection_profile": projection_profile,
        "projected_rule_count": len(projected_rules),
        "heuristic_rule_count": 0,
        "projector_parsed_rules": projected_rules,
        "parsed_rules": parsed_rules,
        "schema_valid": bool(parsed_rules),
        "parse_meta": parse_meta,
        "stage2_backend": attempt.get("stage2_backend", default_stage2_backend),
        "stage2_avg_logprob": attempt.get("stage2_avg_logprob"),
        "stage2_logprob_norm": float(attempt.get("stage2_logprob_norm", 0.0) or 0.0),
        "stage2_generated_token_count": generated_token_count,
        "stage2_base_generated_token_count": generated_token_count,
        "stage2_retry_generated_token_count": 0,
        "stage2_total_call_count": total_call_count,
        "stage2_base_call_count": total_call_count,
        "stage2_retry_call_count": 0,
        "literature_recall_bonus": 0.0,
        "literature_recall_retry_candidate": False,
        "literature_recall_retry_kept": False,
        "literature_recall_retry_reason": "",
        "literature_recall_novel_root_count": 0,
        "independent_single_projection_used": False,
        "independent_single_projection_kept": False,
        "independent_single_projection_novel_root_count": 0,
        "family_target_variant_kept": bool(assessment.get("family_target_variant_kept")),
        "family_target_drop_reason": str(assessment.get("family_target_drop_reason", "")),
        "family_target_original_drop_reason": str(assessment.get("family_target_original_drop_reason", "")),
        "family_target_variant_kept_reason": str(assessment.get("family_target_variant_kept_reason", "")),
        "target_family_match": bool(assessment.get("target_family_match")),
        "off_target_family_drop": bool(assessment.get("off_target_family_drop")),
        "multi_rule_target_violation": bool(assessment.get("multi_rule_target_violation")),
        "target_root_keys": list(assessment.get("target_root_keys", []) or []),
        "target_feature_families": list(assessment.get("target_feature_families", []) or []),
        "target_vocab_options": list(assessment.get("target_vocab_options", []) or []),
        "target_vocab_match": bool(assessment.get("target_vocab_match")),
        "target_vocab_drop_reason": str(assessment.get("target_vocab_drop_reason", "")),
        "target_feature_name": str(assessment.get("target_feature_name", "")),
        "target_thinning_attempted": bool(assessment.get("target_thinning_attempted")),
        "target_thinning_success": bool(assessment.get("target_thinning_success")),
        "target_thinning_trigger_reason": str(assessment.get("target_thinning_trigger_reason", "")),
        "target_thinning_candidate_count": int(assessment.get("target_thinning_candidate_count", 0) or 0),
        "target_thinning_valid_candidate_count": int(
            assessment.get("target_thinning_valid_candidate_count", 0) or 0
        ),
        "target_thinning_selected_root": str(assessment.get("target_thinning_selected_root", "")),
        "target_thinning_selected_family": str(assessment.get("target_thinning_selected_family", "")),
        "target_thinning_selected_feature_name": str(
            assessment.get("target_thinning_selected_feature_name", "")
        ),
        "target_thinning_drop_reason": str(assessment.get("target_thinning_drop_reason", "")),
        "target_thinning_rule_preview": str(assessment.get("target_thinning_rule_preview", "")),
        "selected_for_merge": bool(assessment.get("family_target_variant_kept")),
        "selected_for_merge_reason": (
            str(assessment.get("family_target_variant_kept_reason", ""))
            if assessment.get("family_target_variant_kept")
            else "family_target_variant_rejected"
        ),
    }


def build_literature_retry_projection_record(
    source_item: Dict,
    attempt: Dict,
    assessment: Dict,
    *,
    projector_model: str,
    projector_model_path: str,
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
    stage2_protocol_profile: str,
    stage2_protocol_profile_source: str,
    stage2_protocol_profile_version: str,
    termination_policy: str,
    rule_count_policy: str,
    default_stage2_backend: str,
    classify_projection_status: Callable[[str, Sequence[Dict], Sequence[Dict]], str],
    literature_projection_profile: Callable[[Sequence[Dict]], Dict],
    effective_strict_validator: Callable[..., bool],
) -> Dict:
    projected_rules = list(attempt["projected_rules"])
    parsed_rules = list(projected_rules)
    projection_profile = assessment.get(
        "retry_profile",
        literature_projection_profile(projected_rules),
    )
    original_strict_valid = bool(attempt["strict_rules"] or attempt["repair_strict_rules"])
    qwen_legacy_d2_effective_strict = bool(
        str(stage2_protocol_profile or "") == "qwen_legacy_d2"
        and not original_strict_valid
        and not attempt.get("d4_strict_rules")
        and projected_rules
    )
    effective_strict_valid = effective_strict_validator(
        stage2_protocol_profile=stage2_protocol_profile,
        original_strict_valid=original_strict_valid,
        d4_strict_rules=attempt.get("d4_strict_rules"),
        projected_rules=projected_rules,
    )
    failure_meta = attempt.get("repair_strict_parse_meta", {}) or {}
    generated_token_count = int(attempt.get("stage2_generated_token_count", 0) or 0)
    total_call_count = int(attempt.get("stage2_total_call_count", 0) or 0)
    return {
        **source_item,
        "projection_variant_id": 1,
        "projector_model": projector_model,
        "projector_model_path": projector_model_path,
        "draft_family": draft_family,
        "projector_family": projector_family,
        "use_cross_model_projector": use_cross_model_projector,
        "stage2_protocol_profile": stage2_protocol_profile,
        "stage2_protocol_profile_source": stage2_protocol_profile_source,
        "stage2_protocol_profile_version": stage2_protocol_profile_version,
        "termination_policy": termination_policy,
        "rule_count_policy": rule_count_policy,
        "projection_variant_role": "literature_recall_retry",
        "projector_raw_output": str(attempt["raw_output"]).strip(),
        "projector_repair_output": str(attempt["repair_output"]).strip(),
        "repair_used": bool(attempt["repair_used"]),
        "repair_success": bool(attempt["repair_used"] and projected_rules),
        "projection_status": classify_projection_status(
            attempt["raw_output"],
            projected_rules,
            [],
        ),
        "strict_projector_valid": effective_strict_valid,
        "original_strict_projector_valid": original_strict_valid,
        "effective_strict_projector_valid": effective_strict_valid,
        "qwen_legacy_d2_effective_strict": qwen_legacy_d2_effective_strict,
        "strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not effective_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not effective_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not effective_strict_valid else "",
        },
        "original_strict_failure": {
            "stage": str(failure_meta.get("failure_stage", "") or "") if not original_strict_valid else "",
            "reason": str(failure_meta.get("failure_reason", "") or "") if not original_strict_valid else "",
            "label": str(failure_meta.get("failure_label", "") or "") if not original_strict_valid else "",
        },
        "projector_valid": bool(projected_rules),
        "chat_native_extraction_diagnostics": dict(
            attempt.get("chat_native_extraction_diagnostics", {}) or {}
        ),
        "d4_cleaned_json": str(attempt.get("d4_cleaned_json", "") or ""),
        "heuristic_rescued": False,
        "underdiverse_retry_used": False,
        "underdiverse_retry_success": False,
        "underdiverse_projection": bool(projection_profile.get("underdiverse", False)),
        "underdiverse_trigger_reasons": list(projection_profile.get("underdiverse_trigger_reasons", [])),
        "literature_projection_profile": projection_profile,
        "projected_rule_count": len(projected_rules),
        "heuristic_rule_count": 0,
        "projector_parsed_rules": projected_rules,
        "parsed_rules": parsed_rules,
        "schema_valid": bool(parsed_rules),
        "parse_meta": dict(attempt["parse_meta"]),
        "stage2_backend": attempt.get("stage2_backend", default_stage2_backend),
        "stage2_avg_logprob": attempt.get("stage2_avg_logprob"),
        "stage2_logprob_norm": float(attempt.get("stage2_logprob_norm", 0.0) or 0.0),
        "stage2_generated_token_count": generated_token_count,
        "stage2_base_generated_token_count": generated_token_count,
        "stage2_retry_generated_token_count": 0,
        "stage2_total_call_count": total_call_count,
        "stage2_base_call_count": total_call_count,
        "stage2_retry_call_count": 0,
        "literature_recall_bonus": float(assessment.get("recall_bonus", 0.0) or 0.0),
        "literature_recall_retry_candidate": True,
        "literature_recall_retry_kept": bool(assessment.get("accepted", False)),
        "literature_recall_retry_reason": (
            "novel_root_quality_preserving_retry"
            if assessment.get("accepted", False)
            else "retry_not_quality_preserving"
        ),
        "literature_recall_novel_root_count": int(assessment.get("novel_root_count", 0) or 0),
        "literature_recall_condition_pass_count": int(assessment.get("condition_pass_count", 0) or 0),
        "literature_recall_score_delta_to_base": float(assessment.get("score_delta_to_base", 0.0) or 0.0),
    }
