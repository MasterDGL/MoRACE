from collections import Counter, defaultdict
from morace.generation_runtime import (
    GenerationRuntime, generation_option, science_env_get, with_generation_runtime,
)
from typing import Dict, List, Sequence

from morace.model_backend import (
    LocalTextGenerator,
    ModelSpec,
    infer_model_family,
    is_non_qwen_family,
    normalize_model_name,
)
from morace.projection_diagnostics import (
    _sample_signature,
    audit_first_json_output as _first_json_output_audit,
    collect_llama_native_strict_failure_samples as _collect_llama_native_strict_fail_samples,
    new_strict_failure_diagnostics as _blank_strict_failure_diagnostics,
    normalize_failure_counter as _normalize_failure_counter_map,
    profile_matches as _profile_matches,
    record_strict_failure as _record_strict_failure,
    summarize_llama_native_output_discipline as _llama_native_output_discipline_summary,
    truncate_sample_preview as _truncate_sample_preview,
)
from morace.projection_attempts import run_projection_attempt as _run_projection_attempt_impl
from morace.projection_parsing import (
    SAFE_CANONICALIZATION_FORBIDDEN_FEATURE_ALIASES,
    _record_canonicalization_sample as _canonicalization_sample,
    _record_family_feature_sample as _family_feature_sample,
    apply_family_feature_consistency as _apply_family_feature_consistency,
    canonicalize_enum_value as _canonicalize_enum_value,
    canonicalize_feature_name as _canonicalize_feature_name,
    clean_projected_rule as _d4_clean_rule,
    extract_balanced_objects_from_array_prefix as _extract_balanced_objects_from_array_prefix,
    extract_cleaned_rule_json as _d4_extract_cleaned_json,
    extract_first_balanced_json_object as _extract_first_balanced_json_object,
    feature_name_candidates as _feature_name_candidates,
    find_rules_array_start as _find_rules_array_start,
    new_extraction_diagnostics as _blank_d4_extraction_diagnostics,
    normalize_enum_candidate as _normalize_enum_candidate,
    salvage_rules_array_prefix as _d4_salvage_rules_array_prefix,
)
from morace.projection_prompts import (
    STAGE1_SYSTEM_PROMPT,
    STAGE2_CHEMDFM_REPAIR_SYSTEM_PROMPT,
    STAGE2_CHEMDFM_SYSTEM_PROMPT,
    STAGE2_LITERATURE_SYSTEM_PROMPT,
    STAGE2_REPAIR_SYSTEM_PROMPT,
    STAGE2_SYSTEM_PROMPT,
    _chat_native_core_compact_vocabulary_block,
    _chat_native_core_vocab_rescue_constraints,
    _chemdfm_native_closure_baseline_constraints,
    build_stage1_user_prompt,
    build_stage2_family_target_single_rule_prompt,
    build_stage2_independent_single_rule_prompt,
    build_stage2_literature_retry_prompt,
    build_stage2_repair_prompt,
    build_stage2_user_prompt,
    stage2_max_new_tokens as _stage2_max_new_tokens,
    stage2_repair_system_prompt as _stage2_repair_system_prompt_for,
    stage2_stop_strings as _stage2_stop_strings,
    stage2_system_prompt as _stage2_system_prompt_for,
)
from morace.projection_records import (
    build_base_projection_record,
    build_family_target_projection_record,
    build_independent_projection_record,
    build_literature_retry_projection_record,
)
from morace.projection_selection import (
    DIRTY_DRAFT_TOKENS,
    build_scored_projection_record as _build_scored_projection_record,
    classify_projection_status as _classify_projection_status,
    compile_safe_rules as _compile_safe_rules,
    compile_valid_ratio as _compile_valid_ratio,
    direction_conflict_penalty as _direction_conflict_penalty,
    draft_alignment_score as _draft_alignment_score,
    example_leak_penalty as _example_leak_penalty,
    family_diversity_score as _family_diversity_score,
    feature_families as _feature_families,
    feature_family_diversity_count as _feature_family_diversity_count,
    is_underdiverse_literature_projection as _is_underdiverse_literature_projection,
    late_selection_novelty_pass as _late_selection_novelty_pass_impl,
    late_selection_root_keys as _late_selection_root_keys,
    literature_retry_assessment as _literature_retry_assessment_impl,
    literature_projection_profile,
    mollogp_collapse_rate as _mollogp_collapse_rate,
    novel_root_keys as _novel_root_keys,
    rule_family_list as _rule_family_list,
    rule_root_feature_key as _rule_root_feature_key,
    score_projected_record as _score_projected_record,
    select_best_projected_records as _select_best_projected_records_impl,
    selection_components as _selection_components,
)
from morace.projection_variants import (
    chat_native_core_family_target_enabled as _chat_native_core_family_target_enabled_impl,
    chat_native_core_independent_single_enabled as _chat_native_core_independent_single_enabled_impl,
    family_target_candidate_assessment as _family_target_candidate_assessment_impl,
    stage2_family_target_enabled as _stage2_family_target_enabled_impl,
    target_family_mode_matches as _target_family_mode_matches_impl,
    target_rule_for_strict_reparse as _target_rule_for_strict_reparse_impl,
    target_thinning_empty_result as _target_thinning_empty_result_impl,
    thin_first_valid_target_rule as _thin_first_valid_target_rule_impl,
)
from morace.rule_compiler import compile_rule
from morace.schemas import (
    END_JSON_MARKER,
    heuristic_extract_rules_from_draft,
    lenient_parse_projected_rules,
    strict_parse_projected_rules,
)
from morace.stage2_protocol import (
    STAGE2_PROTOCOL_PROFILE_VERSIONS,
    ALLOWED_STAGE2_PROTOCOL_PROFILES,
    CHAT_NATIVE_CORE_TERMINATION_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_TERMINATION_POLICY,
    CHAT_NATIVE_CORE_RULE_COUNT_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_RULE_COUNT_POLICY,
    CHAT_NATIVE_CORE_EXTRACTION_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_EXTRACTION_POLICY,
    CHAT_NATIVE_CORE_CANONICALIZATION_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_CANONICALIZATION_POLICY,
    CHAT_NATIVE_CORE_FAMILY_TARGET_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY,
    CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS,
    DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGETS,
    CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES,
    DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
    CHAT_NATIVE_CORE_TARGET_MICRO_VOCAB_PREFERRED,
    CHAT_NATIVE_CORE_TARGET_FORBIDDEN_ALIASES,
    CHAT_NATIVE_CORE_TARGET_THINNING_POLICIES,
    ALLOWED_STAGE2_EXTRACTION_POLICIES,
    ALLOWED_STAGE2_CANONICALIZATION_POLICIES,
    ALLOWED_STAGE2_FAMILY_FEATURE_POLICIES,
    ALLOWED_STAGE2_FAMILY_TARGET_POLICIES,
    DEFAULT_STAGE2_FAMILY_TARGET_POLICY,
    ALLOWED_STAGE2_FAMILY_TARGETS,
    DEFAULT_STAGE2_FAMILY_TARGETS,
    ALLOWED_STAGE2_TARGET_VOCAB_POLICIES,
    DEFAULT_STAGE2_TARGET_VOCAB_POLICY,
    STAGE2_TARGET_MICRO_VOCAB_PREFERRED,
    STAGE2_TARGET_FORBIDDEN_ALIASES,
    ALLOWED_STAGE2_TARGET_THINNING_POLICIES,
    DEFAULT_STAGE2_TARGET_THINNING_POLICY,
    chat_native_core_extraction_policy,
    chat_native_core_rule_count_policy,
    chat_native_core_termination_policy,
    chat_native_core_canonicalization_policy,
    chat_native_core_family_target_policy,
    chat_native_core_family_targets,
    chat_native_core_target_vocab_policy,
    chat_native_core_target_micro_vocab_options,
    _chat_native_core_target_vocab_options_for_policy,
    chat_native_core_target_thinning_policy,
    stage2_extraction_policy,
    stage2_canonicalization_policy,
    stage2_family_feature_policy,
    stage2_family_target_policy,
    stage2_family_targets,
    stage2_target_vocab_policy,
    stage2_target_micro_vocab_options,
    _stage2_target_vocab_options_for_policy,
    stage2_target_thinning_policy,
    infer_stage2_protocol_profile,
    resolve_stage2_protocol_context,
    stage2_protocol_profile_version,
)





def _preview_text(text: str, limit: int = 160) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _preview_rules(rules: Sequence[Dict], limit: int = 3) -> str:
    if not rules:
        return "no valid rules"
    parts = []
    for rule in list(rules)[:limit]:
        threshold = rule.get("threshold")
        threshold_text = ""
        if threshold:
            threshold_text = f" {threshold['op']} {threshold['value']}"
        parts.append(f"{rule['feature_family']}:{rule['feature_name']}{threshold_text}")
    return "; ".join(parts)


def _qwen_legacy_d2_effective_strict_valid(
    *,
    stage2_protocol_profile: str,
    original_strict_valid: bool,
    d4_strict_rules: Sequence[Dict] | None,
    projected_rules: Sequence[Dict] | None,
) -> bool:
    if original_strict_valid or d4_strict_rules:
        return True
    return str(stage2_protocol_profile or "") == "qwen_legacy_d2" and bool(projected_rules)














LATE_SELECTION_KEEP_TOPK = 1
LATE_SELECTION_MAX_SCORE_DELTA = 0.0
LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT = 1
LATE_SELECTION_REQUIRE_NOVELTY = True
LITERATURE_RECALL_RETRY_ENABLE = True
LITERATURE_RECALL_BONUS = 0.30
LITERATURE_RECALL_FAMILY_BONUS = 0.20
LITERATURE_RECALL_MAX_SCORE_DELTA = 0.30


def _literature_retry_assessment(
    base_item: Dict,
    original_rules: Sequence[Dict],
    retry_rules: Sequence[Dict],
    *,
    original_strict_valid: bool,
    retry_strict_valid: bool,
    original_repair_used: bool,
    retry_repair_used: bool,
    original_stage2_logprob_norm: float,
    retry_stage2_logprob_norm: float,
) -> Dict[str, int | float | bool]:
    return _literature_retry_assessment_impl(
        base_item,
        original_rules,
        retry_rules,
        original_strict_valid=original_strict_valid,
        retry_strict_valid=retry_strict_valid,
        original_repair_used=original_repair_used,
        retry_repair_used=retry_repair_used,
        original_stage2_logprob_norm=original_stage2_logprob_norm,
        retry_stage2_logprob_norm=retry_stage2_logprob_norm,
        recall_bonus_value=generation_option("LITERATURE_RECALL_BONUS", LITERATURE_RECALL_BONUS),
        family_bonus_value=generation_option("LITERATURE_RECALL_FAMILY_BONUS", LITERATURE_RECALL_FAMILY_BONUS),
        max_score_delta=generation_option("LITERATURE_RECALL_MAX_SCORE_DELTA", LITERATURE_RECALL_MAX_SCORE_DELTA),
    )


def _late_selection_novelty_pass(candidate_rules: Sequence[Dict], kept_rule_sets: Sequence[Sequence[Dict]]) -> bool:
    return _late_selection_novelty_pass_impl(candidate_rules, kept_rule_sets)


def select_best_projected_records(projected_records: Sequence[Dict]) -> List[Dict]:
    return _select_best_projected_records_impl(
        projected_records,
        keep_topk=generation_option("LATE_SELECTION_KEEP_TOPK", LATE_SELECTION_KEEP_TOPK),
        max_score_delta=generation_option("LATE_SELECTION_MAX_SCORE_DELTA", LATE_SELECTION_MAX_SCORE_DELTA),
        max_survivors_per_prompt=generation_option("LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT", LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT),
        require_novelty=generation_option("LATE_SELECTION_REQUIRE_NOVELTY", LATE_SELECTION_REQUIRE_NOVELTY),
    )



def _chat_native_core_independent_single_enabled(stage2_protocol_profile: str, source: str) -> bool:
    if (
        str(stage2_protocol_profile or "").strip() != "chat_native_core"
        or str(source or "").strip() != "literature"
    ):
        return False
    return _chat_native_core_independent_single_enabled_impl(
        stage2_protocol_profile,
        source,
        termination_policy=chat_native_core_termination_policy(),
        rule_count_policy=chat_native_core_rule_count_policy(),
    )


def _chat_native_core_family_target_enabled(stage2_protocol_profile: str, source: str) -> bool:
    if (
        str(stage2_protocol_profile or "").strip() != "chat_native_core"
        or str(source or "").strip() != "literature"
    ):
        return False
    return _chat_native_core_family_target_enabled_impl(
        stage2_protocol_profile,
        source,
        family_target_policy=chat_native_core_family_target_policy(),
    )


def _stage2_family_target_enabled(stage2_protocol_profile: str, source: str) -> bool:
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "chat_native_core":
        return _chat_native_core_family_target_enabled(profile, source)
    if profile != "chemdfm_native" or str(source or "").strip() != "literature":
        return False
    return _stage2_family_target_enabled_impl(
        profile,
        source,
        chat_native_family_target_policy="",
        native_family_target_policy=stage2_family_target_policy(),
    )




def _family_target_candidate_assessment(
    rules: Sequence[Dict],
    target_family: str,
    existing_root_keys: Sequence[str],
    *,
    effective_strict_valid: bool,
    target_vocab_policy: str = DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
    target_vocab_options: Sequence[str] | None = None,
) -> Dict:
    return _family_target_candidate_assessment_impl(
        rules,
        target_family,
        existing_root_keys,
        effective_strict_valid=effective_strict_valid,
        target_vocab_policy=target_vocab_policy,
        target_vocab_options=target_vocab_options,
        rule_root_feature_key=_rule_root_feature_key,
        feature_families=_feature_families,
        compile_valid_ratio=_compile_valid_ratio,
    )


def _target_family_mode_matches(target_family: str, feature_mode: str) -> bool:
    return _target_family_mode_matches_impl(target_family, feature_mode)


def _target_thinning_empty_result(policy: str, trigger_reason: str = "") -> Dict:
    return _target_thinning_empty_result_impl(policy, trigger_reason)


def _target_rule_for_strict_reparse(rule: Dict, source: str) -> Dict:
    return _target_rule_for_strict_reparse_impl(rule, source)


def _thin_first_valid_target_rule(
    rules: Sequence[Dict],
    target_family: str,
    target_vocab_options: Sequence[str],
    existing_root_keys: Sequence[str],
    *,
    source: str,
    provenance: Dict,
    thinning_policy: str,
    trigger_reason: str,
) -> Dict:
    return _thin_first_valid_target_rule_impl(
        rules,
        target_family,
        target_vocab_options,
        existing_root_keys,
        source=source,
        provenance=provenance,
        thinning_policy=thinning_policy,
        trigger_reason=trigger_reason,
        compile_rule=compile_rule,
        rule_root_feature_key=_rule_root_feature_key,
        strict_parser=strict_parse_projected_rules,
        preview_rules=_preview_rules,
        family_mode_matches=_target_family_mode_matches,
        empty_result=_target_thinning_empty_result,
        rule_for_strict_reparse=_target_rule_for_strict_reparse,
    )


def _run_projection_attempt(
    backend: LocalTextGenerator,
    *,
    system_prompt: str,
    user_prompt: str,
    source: str,
    provenance: Dict,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_input_tokens: int,
    allow_repair_pass: bool,
    projector_family: str = "generic",
    stage2_protocol_profile: str = "",
) -> Dict:
    return _run_projection_attempt_impl(
        backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        source=source,
        provenance=provenance,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_input_tokens=max_input_tokens,
        allow_repair_pass=allow_repair_pass,
        projector_family=projector_family,
        stage2_protocol_profile=stage2_protocol_profile,
        strict_parser=strict_parse_projected_rules,
        lenient_parser=lenient_parse_projected_rules,
    )


@with_generation_runtime
def generate_stage1_drafts(
    prompt_records: Sequence[Dict],
    draft_specs: Sequence[ModelSpec],
    drafts_per_prompt: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_input_tokens: int = 0,
    dirty_retry_tokens: Sequence[str] | None = None,
    seed: int = 42,
    *,
    runtime: GenerationRuntime | None = None,
) -> List[Dict]:
    draft_outputs: List[Dict] = []
    for spec_index, spec in enumerate(draft_specs):
        total_jobs = len(prompt_records) * drafts_per_prompt
        print(
            f"[Stage1] draft_model={normalize_model_name(spec.name)} "
            f"jobs={total_jobs} max_new_tokens={max_new_tokens} max_input_tokens={max_input_tokens or 'auto'}"
        )
        with LocalTextGenerator(spec.name, spec.path, seed=seed + spec_index * 100_000) as backend:
            job_idx = 0
            for record in prompt_records:
                for local_draft_id in range(drafts_per_prompt):
                    job_idx += 1
                    print(
                        f"[Stage1][{job_idx}/{total_jobs}] generating prompt={record['prompt_id']} "
                        f"source={record['source']} draft_id={local_draft_id}"
                    )
                    response = ""
                    dirty_retry_count = 0
                    for attempt_idx in range(2):
                        response = backend.generate_batch(
                            [(STAGE1_SYSTEM_PROMPT, record["stage1_user_prompt"])],
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            num_beams=num_beams,
                            max_input_tokens=max_input_tokens if max_input_tokens > 0 else None,
                            preserve_head_tokens=320,
                            preserve_tail_tokens=320,
                        )[0]
                        if not dirty_retry_tokens or not any(token.lower() in response.lower() for token in dirty_retry_tokens):
                            break
                        dirty_retry_count += 1
                        print(
                            f"[Stage1][{job_idx}/{total_jobs}] draft hygiene retry prompt={record['prompt_id']} "
                            f"attempt={attempt_idx + 1}"
                        )
                    print(
                        f"[Stage1][{job_idx}/{total_jobs}] finished prompt={record['prompt_id']} "
                        f"preview={_preview_text(response)}"
                    )
                    draft_outputs.append(
                        {
                            "prompt_id": record["prompt_id"],
                            "source": record["source"],
                            "draft_model": normalize_model_name(spec.name),
                            "draft_model_path": spec.path,
                            "draft_id": local_draft_id,
                            "batch_id": record.get("batch_id"),
                            "stage1_user_prompt": record["stage1_user_prompt"],
                            "draft_text": response.strip(),
                            "dirty_retry_count": dirty_retry_count,
                            "draft_hygiene_flagged": bool(
                                dirty_retry_tokens and any(token.lower() in response.lower() for token in dirty_retry_tokens)
                            ),
                        }
                    )
    return draft_outputs


@with_generation_runtime
def project_drafts_to_rules(
    prompt_records_by_id: Dict[str, Dict],
    draft_outputs: Sequence[Dict],
    projector_spec: ModelSpec,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_beams: int,
    max_input_tokens: int = 0,
    allow_heuristic_fallback: bool = True,
    heuristic_fallback_sources: Sequence[str] | None = None,
    heuristic_max_rules: int = 2,
    allow_repair_pass: bool = True,
    seed: int = 42,
    *,
    runtime: GenerationRuntime | None = None,
) -> List[Dict]:
    total_jobs = len(draft_outputs)
    print(
        f"[Stage2] projector_model={normalize_model_name(projector_spec.name)} "
        f"jobs={total_jobs} max_new_tokens={max_new_tokens} max_input_tokens={max_input_tokens or 'auto'}"
    )
    projected_records: List[Dict] = []
    with LocalTextGenerator(projector_spec.name, projector_spec.path, seed=seed) as backend:
        projector_family = getattr(backend, "model_family", None) or infer_model_family(projector_spec.name, projector_spec.path)
        compat_stage2 = is_non_qwen_family(projector_family)
        stage2_max_new_tokens = max(max_new_tokens, 384) if compat_stage2 else max_new_tokens
        use_cross_model_projector = bool(int(science_env_get("USE_CROSS_MODEL_PROJECTOR", "0") or 0))
        for idx, item in enumerate(draft_outputs, 1):
            base_record = prompt_records_by_id[item["prompt_id"]]
            draft_family = infer_model_family(item["draft_model"], item.get("draft_model_path", ""))
            protocol_context = resolve_stage2_protocol_context(
                draft_family=draft_family,
                projector_family=projector_family,
                use_cross_model_projector=use_cross_model_projector,
            )
            stage2_protocol_profile = str(protocol_context["profile"])
            stage2_protocol_profile_source = str(protocol_context["profile_source"])
            stage2_protocol_profile_version_value = str(protocol_context["profile_version"])
            termination_policy = str(protocol_context["termination_policy"])
            rule_count_policy = str(protocol_context["rule_count_policy"])
            family_target_policy = str(protocol_context["family_target_policy"])
            target_vocab_policy = str(protocol_context["target_vocab_policy"])
            target_thinning_policy = str(protocol_context["target_thinning_policy"])
            family_target_families = list(protocol_context["family_target_families"])
            print(
                f"[Stage2][{idx}/{total_jobs}] projecting prompt={item['prompt_id']} "
                f"draft_model={item['draft_model']} profile={stage2_protocol_profile} "
                f"profile_version={stage2_protocol_profile_version_value} profile_source={stage2_protocol_profile_source} "
                f"termination_policy={termination_policy or '-'} rule_count_policy={rule_count_policy or '-'} "
                f"family_target_policy={family_target_policy or '-'} target_vocab_policy={target_vocab_policy or '-'} "
                f"target_thinning_policy={target_thinning_policy or '-'}"
            )
            system_prompt = _stage2_system_prompt_for(item["source"], stage2_protocol_profile)
            provenance = {
                "draft_model": item["draft_model"],
                "draft_id": item["draft_id"],
                "batch_id": item.get("batch_id"),
                "stage2_protocol_profile": stage2_protocol_profile,
                "projector_family": projector_family,
                "termination_policy": termination_policy,
                "rule_count_policy": rule_count_policy,
                "family_target_policy": family_target_policy,
                "target_vocab_policy": target_vocab_policy,
                "target_thinning_policy": target_thinning_policy,
            }
            attempt_stage2_max_new_tokens = _stage2_max_new_tokens(stage2_protocol_profile, stage2_max_new_tokens)
            attempt = _run_projection_attempt(
                backend,
                system_prompt=system_prompt,
                user_prompt=build_stage2_user_prompt(
                    task_brief=base_record.get("task_brief", base_record["base_prompt"]),
                    draft_text=item["draft_text"],
                    source=item["source"],
                    short_schema=True,
                    projector_family=projector_family,
                    stage2_protocol_profile=stage2_protocol_profile,
                ),
                source=item["source"],
                provenance=provenance,
                max_new_tokens=attempt_stage2_max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                max_input_tokens=max_input_tokens,
                allow_repair_pass=allow_repair_pass,
                projector_family=projector_family,
                stage2_protocol_profile=stage2_protocol_profile,
            )
            underdiverse_retry_used = False
            underdiverse_retry_success = False
            retry_assessment = {}
            base_stage2_call_count = int(attempt.get("stage2_total_call_count", 0) or 0)
            base_stage2_token_count = int(attempt.get("stage2_generated_token_count", 0) or 0)
            retry_stage2_call_count = 0
            retry_stage2_token_count = 0
            independent_single_projection_used = False
            independent_single_projection_kept = False
            independent_attempt = None
            independent_novel_roots: List[str] = []
            family_target_projection_used = False
            family_target_attempt_records: List[Dict] = []
            family_target_kept_roots = set()
            original_profile = literature_projection_profile(attempt["projected_rules"])
            base_root_keys = sorted(
                {
                    _rule_root_feature_key(rule)
                    for rule in attempt["projected_rules"]
                    if rule.get("feature_family") or rule.get("feature_name")
                }
            )
            base_families = sorted(
                {str(rule.get("feature_family", "")) for rule in attempt["projected_rules"] if rule.get("feature_family")}
            )
            family_target_kept_roots.update(base_root_keys)
            if _chat_native_core_independent_single_enabled(stage2_protocol_profile, item["source"]):
                independent_single_projection_used = True
                independent_attempt = _run_projection_attempt(
                    backend,
                    system_prompt=_stage2_system_prompt_for(item["source"], stage2_protocol_profile),
                    user_prompt=build_stage2_independent_single_rule_prompt(
                        task_brief=base_record.get("task_brief", base_record["base_prompt"]),
                        draft_text=item["draft_text"],
                        previous_roots=base_root_keys,
                        previous_families=base_families,
                        source=item["source"],
                    ),
                    source=item["source"],
                    provenance={
                        **provenance,
                        "projection_variant_role": "independent_single_rule_2",
                    },
                    max_new_tokens=attempt_stage2_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=max(1, num_beams),
                    max_input_tokens=max_input_tokens,
                    allow_repair_pass=allow_repair_pass,
                    projector_family=projector_family,
                    stage2_protocol_profile=stage2_protocol_profile,
                )
                independent_rules = list(independent_attempt.get("projected_rules", []) or [])
                independent_root_keys = sorted(
                    {
                        _rule_root_feature_key(rule)
                        for rule in independent_rules
                        if rule.get("feature_family") or rule.get("feature_name")
                    }
                )
                independent_novel_roots = sorted(set(independent_root_keys) - set(base_root_keys))
                independent_strict_valid = bool(
                    independent_attempt.get("strict_rules") or independent_attempt.get("repair_strict_rules")
                )
                independent_single_projection_kept = bool(
                    independent_strict_valid
                    and _compile_valid_ratio(independent_rules) > 0.0
                    and independent_novel_roots
                )
            if _stage2_family_target_enabled(stage2_protocol_profile, item["source"]):
                family_target_projection_used = True
                for target_offset, target_family in enumerate(family_target_families, 1):
                    target_vocab_options = (
                        _chat_native_core_target_vocab_options_for_policy(target_family, target_vocab_policy)
                        if stage2_protocol_profile == "chat_native_core"
                        else _stage2_target_vocab_options_for_policy(target_family, target_vocab_policy)
                    )
                    target_attempt = _run_projection_attempt(
                        backend,
                        system_prompt=_stage2_system_prompt_for(item["source"], stage2_protocol_profile),
                        user_prompt=build_stage2_family_target_single_rule_prompt(
                            task_brief=base_record.get("task_brief", base_record["base_prompt"]),
                            draft_text=item["draft_text"],
                            target_family=target_family,
                            source=item["source"],
                            target_vocab_policy=target_vocab_policy,
                            target_vocab_options=target_vocab_options,
                        ),
                        source=item["source"],
                        provenance={
                            **provenance,
                            "projection_variant_role": f"family_target_{target_family}",
                            "family_target_family": target_family,
                            "target_vocab_policy": target_vocab_policy,
                            "target_vocab_options": list(target_vocab_options),
                            "target_thinning_policy": target_thinning_policy,
                        },
                        max_new_tokens=attempt_stage2_max_new_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        num_beams=max(1, num_beams),
                        max_input_tokens=max_input_tokens,
                        allow_repair_pass=allow_repair_pass,
                        projector_family=projector_family,
                        stage2_protocol_profile=stage2_protocol_profile,
                    )
                    target_rules = list(target_attempt.get("projected_rules", []) or [])
                    target_original_strict_valid = bool(
                        target_attempt.get("strict_rules") or target_attempt.get("repair_strict_rules")
                    )
                    target_effective_strict_valid = bool(
                        target_original_strict_valid or target_attempt.get("d4_strict_rules")
                    )
                    target_assessment = _family_target_candidate_assessment(
                        target_rules,
                        target_family,
                        sorted(family_target_kept_roots),
                        effective_strict_valid=target_effective_strict_valid,
                        target_vocab_policy=target_vocab_policy,
                        target_vocab_options=target_vocab_options,
                    )
                    target_thinning_result = _target_thinning_empty_result(target_thinning_policy)
                    target_original_drop_reason = str(target_assessment.get("family_target_drop_reason", "") or "")
                    if (
                        not target_assessment.get("family_target_variant_kept")
                        and target_thinning_policy == "first_valid_target_rule"
                        and target_vocab_policy == "micro_canonical_options"
                    ):
                        target_thinning_result = _thin_first_valid_target_rule(
                            target_rules,
                            target_family,
                            target_vocab_options,
                            sorted(family_target_kept_roots),
                            source=item["source"],
                            provenance={
                                **provenance,
                                "projection_variant_role": f"family_target_{target_family}",
                                "family_target_family": target_family,
                                "target_vocab_policy": target_vocab_policy,
                                "target_vocab_options": list(target_vocab_options),
                                "target_thinning_policy": target_thinning_policy,
                            },
                            thinning_policy=target_thinning_policy,
                            trigger_reason=target_original_drop_reason,
                        )
                        if target_thinning_result.get("target_thinning_success"):
                            target_rules = list(target_thinning_result.get("target_thinning_rules", []) or [])
                            target_effective_strict_valid = True
                            target_assessment = _family_target_candidate_assessment(
                                target_rules,
                                target_family,
                                sorted(family_target_kept_roots),
                                effective_strict_valid=True,
                                target_vocab_policy=target_vocab_policy,
                                target_vocab_options=target_vocab_options,
                            )
                            target_assessment["family_target_original_drop_reason"] = target_original_drop_reason
                            target_assessment["family_target_variant_kept_reason"] = "target_thinning_rescue"
                    target_assessment.update(
                        {
                            key: value
                            for key, value in target_thinning_result.items()
                            if key != "target_thinning_rules"
                        }
                    )
                    target_assessment.setdefault("family_target_original_drop_reason", target_original_drop_reason)
                    target_assessment.setdefault(
                        "family_target_variant_kept_reason",
                        "family_target_variant_kept"
                        if target_assessment.get("family_target_variant_kept")
                        else "family_target_variant_rejected",
                    )
                    if target_assessment["family_target_variant_kept"]:
                        family_target_kept_roots.update(target_assessment["target_root_keys"])
                    family_target_attempt_records.append(
                        {
                            "variant_id": target_offset,
                            "target_family": target_family,
                            "attempt": target_attempt,
                            "assessment": target_assessment,
                            "projected_rules_override": list(target_rules),
                            "original_strict_valid": target_original_strict_valid,
                            "effective_strict_valid": target_effective_strict_valid,
                            "target_vocab_options": list(target_vocab_options),
                        }
                    )
            if (
                item["source"] == "literature"
                and generation_option("LITERATURE_RECALL_RETRY_ENABLE", LITERATURE_RECALL_RETRY_ENABLE)
                and not independent_single_projection_used
                and not family_target_projection_used
                and attempt["projected_rules"]
                and bool(original_profile.get("underdiverse", False))
            ):
                underdiverse_retry_used = True
                retry_attempt = _run_projection_attempt(
                    backend,
                    system_prompt=_stage2_system_prompt_for(item["source"], stage2_protocol_profile),
                    user_prompt=build_stage2_literature_retry_prompt(
                        task_brief=base_record.get("task_brief", base_record["base_prompt"]),
                        draft_text=item["draft_text"],
                        previous_output=(attempt["repair_output"] or attempt["raw_output"]),
                        current_roots=base_root_keys,
                        current_families=base_families,
                        projector_family=projector_family,
                        stage2_protocol_profile=stage2_protocol_profile,
                    ),
                    source=item["source"],
                    provenance=provenance,
                    max_new_tokens=attempt_stage2_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=max(1, num_beams),
                    max_input_tokens=max_input_tokens,
                    allow_repair_pass=allow_repair_pass,
                    projector_family=projector_family,
                    stage2_protocol_profile=stage2_protocol_profile,
                )
                retry_stage2_call_count = int(retry_attempt.get("stage2_total_call_count", 0) or 0)
                retry_stage2_token_count = int(retry_attempt.get("stage2_generated_token_count", 0) or 0)
                if retry_attempt["projected_rules"]:
                    retry_assessment = _literature_retry_assessment(
                        item,
                        attempt["projected_rules"],
                        retry_attempt["projected_rules"],
                        original_strict_valid=bool(attempt["strict_rules"] or attempt["repair_strict_rules"]),
                        retry_strict_valid=bool(retry_attempt["strict_rules"] or retry_attempt["repair_strict_rules"]),
                        original_repair_used=bool(attempt["repair_used"]),
                        retry_repair_used=bool(retry_attempt["repair_used"]),
                        original_stage2_logprob_norm=float(attempt.get("stage2_logprob_norm", 0.0) or 0.0),
                        retry_stage2_logprob_norm=float(retry_attempt.get("stage2_logprob_norm", 0.0) or 0.0),
                    )
                    underdiverse_retry_success = bool(retry_assessment.get("accepted", False))
            raw_output = attempt["raw_output"]
            repair_output = attempt["repair_output"]
            projected_rules = list(attempt["projected_rules"])
            strict_rules = list(attempt["strict_rules"])
            d4_strict_rules = list(attempt.get("d4_strict_rules", []) or [])
            repair_strict_rules = list(attempt["repair_strict_rules"])
            repair_used = bool(attempt["repair_used"])
            parse_meta = dict(attempt["parse_meta"])
            heuristic_rules: List[Dict] = []
            heuristic_allowed = allow_heuristic_fallback and (
                heuristic_fallback_sources is None or item["source"] in set(heuristic_fallback_sources)
            )
            if not projected_rules and heuristic_allowed:
                heuristic_rules = heuristic_extract_rules_from_draft(
                    item["draft_text"],
                    source=item["source"],
                    provenance=provenance,
                    max_rules=heuristic_max_rules,
                )
                if item["source"] == "literature":
                    heuristic_rules = _compile_safe_rules(heuristic_rules)[:heuristic_max_rules]
                    if heuristic_rules and all(str(rule.get("feature_name", "")) == "MolLogP" for rule in heuristic_rules):
                        heuristic_rules = []
            parsed_rules = list(projected_rules) + list(heuristic_rules)
            projection_status = _classify_projection_status(raw_output, projected_rules, heuristic_rules)
            original_strict_valid = bool(strict_rules or repair_strict_rules)
            qwen_legacy_d2_effective_strict = (
                str(stage2_protocol_profile or "") == "qwen_legacy_d2"
                and not original_strict_valid
                and not d4_strict_rules
                and bool(projected_rules)
            )
            effective_strict_valid = _qwen_legacy_d2_effective_strict_valid(
                stage2_protocol_profile=stage2_protocol_profile,
                original_strict_valid=original_strict_valid,
                d4_strict_rules=d4_strict_rules,
                projected_rules=projected_rules,
            )
            strict_valid = effective_strict_valid
            lenient_valid = bool(projected_rules)
            strict_failure_meta = (
                dict(attempt.get("repair_strict_parse_meta", {}))
                if repair_used and not original_strict_valid
                else dict(attempt.get("strict_parse_meta", {}))
            )
            strict_failure = {
                "stage": "" if strict_valid else str(strict_failure_meta.get("failure_stage", "") or ""),
                "reason": "" if strict_valid else str(strict_failure_meta.get("failure_reason", "") or ""),
                "label": "" if strict_valid else str(strict_failure_meta.get("failure_label", "") or ""),
            }
            original_strict_failure = {
                "stage": "" if original_strict_valid else str(strict_failure_meta.get("failure_stage", "") or ""),
                "reason": "" if original_strict_valid else str(strict_failure_meta.get("failure_reason", "") or ""),
                "label": "" if original_strict_valid else str(strict_failure_meta.get("failure_label", "") or ""),
            }
            print(
                f"[Stage2][{idx}/{total_jobs}] status={projection_status} parsed_rules={len(parsed_rules)} "
                f"strict_valid={int(strict_valid)} strict_failure={strict_failure.get('label', '') or 'none'} "
                f"preview={_preview_rules(parsed_rules)} raw_preview={_preview_text(repair_output or raw_output)}"
            )
            base_projection_record = build_base_projection_record(
                item,
                attempt,
                projected_rules=projected_rules,
                heuristic_rules=heuristic_rules,
                projection_status=projection_status,
                projector_valid=lenient_valid,
                parse_meta=parse_meta,
                strict_valid=strict_valid,
                original_strict_valid=original_strict_valid,
                effective_strict_valid=effective_strict_valid,
                qwen_legacy_d2_effective_strict=qwen_legacy_d2_effective_strict,
                strict_failure=strict_failure,
                original_strict_failure=original_strict_failure,
                original_profile=original_profile,
                retry_assessment=retry_assessment,
                projector_model=normalize_model_name(projector_spec.name),
                projector_model_path=projector_spec.path,
                draft_family=draft_family,
                projector_family=projector_family,
                use_cross_model_projector=use_cross_model_projector,
                stage2_protocol_profile=stage2_protocol_profile,
                stage2_protocol_profile_source=stage2_protocol_profile_source,
                stage2_protocol_profile_version=stage2_protocol_profile_version_value,
                termination_policy=termination_policy,
                rule_count_policy=rule_count_policy,
                family_target_policy=family_target_policy,
                target_vocab_policy=target_vocab_policy,
                target_thinning_policy=target_thinning_policy,
                family_target_projection_used=family_target_projection_used,
                underdiverse_retry_used=underdiverse_retry_used,
                underdiverse_retry_success=underdiverse_retry_success,
                base_stage2_call_count=base_stage2_call_count,
                base_stage2_token_count=base_stage2_token_count,
                retry_stage2_call_count=retry_stage2_call_count,
                retry_stage2_token_count=retry_stage2_token_count,
                independent_single_projection_used=independent_single_projection_used,
                independent_single_projection_kept=independent_single_projection_kept,
                independent_novel_roots=independent_novel_roots,
                default_stage2_backend=science_env_get("STAGE2_BACKEND", "prompt_only"),
                is_underdiverse_literature_projection=_is_underdiverse_literature_projection,
            )
            projected_records.append(base_projection_record)
            if independent_single_projection_used and independent_attempt is not None:
                projected_records.append(
                    build_independent_projection_record(
                        item,
                        independent_attempt,
                        projection_kept=independent_single_projection_kept,
                        novel_roots=independent_novel_roots,
                        projector_model=normalize_model_name(projector_spec.name),
                        projector_model_path=projector_spec.path,
                        draft_family=draft_family,
                        projector_family=projector_family,
                        use_cross_model_projector=use_cross_model_projector,
                        stage2_protocol_profile=stage2_protocol_profile,
                        stage2_protocol_profile_source=stage2_protocol_profile_source,
                        stage2_protocol_profile_version=stage2_protocol_profile_version_value,
                        termination_policy=termination_policy,
                        rule_count_policy=rule_count_policy,
                        default_stage2_backend=science_env_get("STAGE2_BACKEND", "prompt_only"),
                        classify_projection_status=_classify_projection_status,
                        literature_projection_profile=literature_projection_profile,
                        compile_valid_ratio=_compile_valid_ratio,
                    )
                )
            for target_info in family_target_attempt_records:
                projected_records.append(
                    build_family_target_projection_record(
                        item,
                        target_info,
                        projector_model=normalize_model_name(projector_spec.name),
                        projector_model_path=projector_spec.path,
                        draft_family=draft_family,
                        projector_family=projector_family,
                        use_cross_model_projector=use_cross_model_projector,
                        stage2_protocol_profile=stage2_protocol_profile,
                        stage2_protocol_profile_source=stage2_protocol_profile_source,
                        stage2_protocol_profile_version=stage2_protocol_profile_version_value,
                        termination_policy=termination_policy,
                        rule_count_policy=rule_count_policy,
                        family_target_policy=family_target_policy,
                        target_vocab_policy=target_vocab_policy,
                        target_thinning_policy=target_thinning_policy,
                        default_stage2_backend=science_env_get("STAGE2_BACKEND", "prompt_only"),
                        classify_projection_status=_classify_projection_status,
                        literature_projection_profile=literature_projection_profile,
                    )
                )
            if underdiverse_retry_used and retry_assessment:
                retry_record = build_literature_retry_projection_record(
                    item,
                    retry_attempt,
                    retry_assessment,
                    projector_model=normalize_model_name(projector_spec.name),
                    projector_model_path=projector_spec.path,
                    draft_family=draft_family,
                    projector_family=projector_family,
                    use_cross_model_projector=use_cross_model_projector,
                    stage2_protocol_profile=stage2_protocol_profile,
                    stage2_protocol_profile_source=stage2_protocol_profile_source,
                    stage2_protocol_profile_version=stage2_protocol_profile_version_value,
                    termination_policy=termination_policy,
                    rule_count_policy=rule_count_policy,
                    default_stage2_backend=science_env_get("STAGE2_BACKEND", "prompt_only"),
                    classify_projection_status=_classify_projection_status,
                    literature_projection_profile=literature_projection_profile,
                    effective_strict_validator=_qwen_legacy_d2_effective_strict_valid,
                )
                if retry_record["literature_recall_retry_kept"]:
                    projected_records.append(retry_record)
    return select_best_projected_records(projected_records)


def flatten_projected_rules(projected_records: Sequence[Dict]) -> List[Dict]:
    rules: List[Dict] = []
    selected_items = [item for item in projected_records if item.get("selected_for_merge", True)]
    for item in selected_items:
        rules.extend(item.get("parsed_rules", []))
    return rules


def _strict_valid_projected_rules(projected_records: Sequence[Dict]) -> List[Dict]:
    rules: List[Dict] = []
    for item in projected_records:
        if not bool(item.get("strict_projector_valid")):
            continue
        item_rules = item.get("parsed_rules", []) or item.get("projector_parsed_rules", []) or []
        rules.extend(item_rules)
    return rules


def _synth_path_counts(projected_records: Sequence[Dict], profile: str, branch_name: str = "") -> Dict[str, object]:
    all_profile_records = [item for item in projected_records if _profile_matches(item, profile)]
    profile_records = [
        item for item in all_profile_records
        if int(item.get("projection_variant_id", 0) or 0) == 0
    ]
    if not profile_records:
        return {}
    base_variant_records = [item for item in all_profile_records if int(item.get("projection_variant_id", 0) or 0) == 0]
    retry_variant_records = [item for item in all_profile_records if int(item.get("projection_variant_id", 0) or 0) != 0]
    raw_rule_count = int(sum(int(item.get("projected_rule_count", 0) or 0) for item in profile_records))
    normalized_rule_count = int(sum(len(item.get("projector_parsed_rules", []) or []) for item in profile_records))
    missing_selected_for_merge_count = int(sum(1 for item in profile_records if "selected_for_merge" not in item))
    merge_survivor_count = int(sum(1 for item in profile_records if bool(item.get("selected_for_merge", True))))
    verifier_accept_count = int(sum(1 for item in profile_records if _compile_valid_ratio(item.get("parsed_rules", [])) > 0.0))
    selected_items = [item for item in profile_records if bool(item.get("selected_for_merge", True))]
    selected_rule_count = int(sum(len(item.get("parsed_rules", []) or []) for item in selected_items))
    selected_rules = flatten_projected_rules(selected_items)
    compile_candidate_count = int(sum(1 for item in selected_items if bool(item.get("parsed_rules"))))
    reject_histogram = Counter(
        str(item.get("selected_for_merge_reason", "") or "unknown") for item in profile_records
    )
    failure_stage = ""
    if raw_rule_count == 0:
        failure_stage = "raw_empty"
    elif normalized_rule_count == 0:
        failure_stage = "normalize_empty"
    elif verifier_accept_count == 0:
        failure_stage = "verifier_empty"
    elif merge_survivor_count == 0 or selected_rule_count == 0:
        failure_stage = "selection_empty"
    elif compile_candidate_count == 0:
        failure_stage = "compile_empty"
    return {
        "profile": profile,
        "profile_version": STAGE2_PROTOCOL_PROFILE_VERSIONS.get(profile, ""),
        "termination_policy": chat_native_core_termination_policy() if profile == "chat_native_core" else "",
        "rule_count_policy": chat_native_core_rule_count_policy() if profile == "chat_native_core" else "",
        "branch": str(branch_name or ""),
        "record_count": len(profile_records),
        "synth_record_count_base_variant": len(base_variant_records),
        "synth_record_count_retry_variant": len(retry_variant_records),
        "synth_raw_rule_count": raw_rule_count,
        "synth_normalized_rule_count": normalized_rule_count,
        "synth_merge_survivor_count": merge_survivor_count,
        "synth_missing_selected_for_merge_count": missing_selected_for_merge_count,
        "synth_verifier_accept_count": verifier_accept_count,
        "synth_selected_rule_count": selected_rule_count,
        "synth_selected_rule_count_base_variant": int(
            sum(len(item.get("parsed_rules", []) or []) for item in base_variant_records if bool(item.get("selected_for_merge", True)))
        ),
        "synth_selected_rule_count_retry_variant": int(
            sum(len(item.get("parsed_rules", []) or []) for item in retry_variant_records if bool(item.get("selected_for_merge", True)))
        ),
        "synth_selected_feature_families": _feature_families(selected_rules),
        "synth_reject_reason_histogram": dict(reject_histogram),
        "synth_compile_candidate_count": compile_candidate_count,
        "synth_empty_selection_stage": failure_stage if failure_stage.endswith("_empty") else "",
        "synth_failure_stage": failure_stage,
    }


def _strict_fail_output_summary(projected_records: Sequence[Dict], profile: str, branch_name: str) -> Dict:
    profile_records = [item for item in projected_records if _profile_matches(item, profile)]
    if not profile_records:
        return {}
    raw_outputs = [str(item.get("projector_raw_output", "") or "") for item in profile_records]
    strict_fail_items = [item for item in profile_records if not bool(item.get("strict_projector_valid"))]
    json_audits = [_first_json_output_audit(text) for text in raw_outputs]
    suffix_after_first_json_count = sum(
        1 for audit in json_audits if str(audit.get("suffix_after_first_json", "") or "")
    )
    suffix_previews = [
        _truncate_sample_preview(str(audit.get("suffix_after_first_json", "")), limit=300)
        for audit in json_audits
        if str(audit.get("suffix_after_first_json", "") or "")
    ][:5]
    return {
        "branch": str(branch_name or ""),
        "profile": profile,
        "profile_version": STAGE2_PROTOCOL_PROFILE_VERSIONS.get(profile, ""),
        "termination_policy": chat_native_core_termination_policy() if profile == "chat_native_core" else "",
        "record_count": len(profile_records),
        "strict_fail_count": len(strict_fail_items),
        "strict_fail_count_base_variant": sum(
            1 for item in strict_fail_items if int(item.get("projection_variant_id", 0) or 0) == 0
        ),
        "strict_fail_count_retry_variant": sum(
            1 for item in strict_fail_items if int(item.get("projection_variant_id", 0) or 0) != 0
        ),
        "starts_with_json_object_count": sum(1 for text in raw_outputs if text.lstrip().startswith("{")),
        "ends_with_closing_brace_count": sum(1 for text in raw_outputs if text.rstrip().endswith("}")),
        "contains_code_fence_count": sum(1 for text in raw_outputs if "```" in text),
        "contains_end_json_count": sum(1 for text in raw_outputs if END_JSON_MARKER in text),
        "contains_prefix_text_count": sum(1 for text in raw_outputs if text and not text.lstrip().startswith("{")),
        "contains_suffix_text_count": sum(1 for text in raw_outputs if "}" in text and not text.rstrip().endswith("}")),
        "raw_first_json_parseable_count": sum(1 for audit in json_audits if bool(audit.get("first_json_parseable"))),
        "json_object_closes_before_suffix_count": sum(
            1 for audit in json_audits if bool(audit.get("json_closes_before_suffix"))
        ),
        "suffix_after_first_json_count": suffix_after_first_json_count,
        "suffix_after_first_json_preview": suffix_previews,
    }


def _collect_strict_fail_samples_by_reason(
    projected_records: Sequence[Dict],
    profile: str,
    preview_limit: int = 500,
    per_reason_limit: int = 5,
) -> Dict[str, List[Dict]]:
    by_reason: Dict[str, List[Dict]] = {}
    seen_by_reason: Dict[str, set[str]] = defaultdict(set)
    reason_counts: Dict[str, int] = defaultdict(int)
    for item in reversed(list(projected_records)):
        if not _profile_matches(item, profile):
            continue
        if bool(item.get("strict_projector_valid")):
            continue
        failure = item.get("strict_failure", {}) or {}
        reason = str(failure.get("reason", "") or "unknown")
        if reason_counts[reason] >= per_reason_limit:
            continue
        raw_preview = _truncate_sample_preview(item.get("projector_raw_output", ""), limit=preview_limit)
        repair_preview = _truncate_sample_preview(item.get("projector_repair_output", ""), limit=preview_limit)
        signature = _sample_signature(item, raw_preview, repair_preview)
        if signature in seen_by_reason[reason]:
            continue
        seen_by_reason[reason].add(signature)
        reason_counts[reason] += 1
        by_reason.setdefault(reason, []).append(
            {
                "source": str(item.get("source", "") or "unknown"),
                "label": str(failure.get("label", "") or ""),
                "prompt_id": str(item.get("prompt_id", "") or ""),
                "draft_id": int(item.get("draft_id", 0) or 0),
                "projection_variant_id": int(item.get("projection_variant_id", 0) or 0),
                "raw_output_preview": raw_preview,
                "repair_output_preview": repair_preview,
            }
        )
    return {key: list(reversed(value)) for key, value in by_reason.items()}


@with_generation_runtime
def summarize_projection_metrics(
    projected_records: Sequence[Dict], branch_name: str = "", *,
    runtime: GenerationRuntime | None = None,
) -> Dict:
    flattened_rules = flatten_projected_rules(projected_records)
    flattened_root_counts = Counter(_rule_root_feature_key(rule) for rule in flattened_rules)
    strict_valid_rules = _strict_valid_projected_rules(projected_records)
    strict_valid_root_keys = sorted(
        {
            _rule_root_feature_key(rule)
            for rule in strict_valid_rules
            if rule.get("feature_family") or rule.get("feature_name")
        }
    )
    strict_valid_feature_families = _feature_families(strict_valid_rules)
    total = len(projected_records)
    base_items = [item for item in projected_records if int(item.get("projection_variant_id", 0) or 0) == 0]
    valid = sum(1 for item in projected_records if item.get("schema_valid"))
    projector_valid = sum(1 for item in projected_records if item.get("projector_valid"))
    strict_projector_valid = sum(1 for item in projected_records if item.get("strict_projector_valid"))
    heuristic_rescues = sum(1 for item in projected_records if item.get("heuristic_rescued"))
    by_source = defaultdict(int)
    by_model = defaultdict(int)
    projected_rules_per_source = defaultdict(int)
    heuristic_rules_per_source = defaultdict(int)
    status_counts = defaultdict(int)
    heuristic_rule_count = 0
    projected_rule_count = 0
    strict_scientific_rule_success_count = 0
    repair_used_count = 0
    repair_success_count = 0
    underdiverse_retry_count = 0
    underdiverse_retry_success_count = 0
    underdiverse_projection_count = 0
    selected_count = 0
    dirty_retry_total = 0
    draft_hygiene_flagged_count = 0
    max_nesting_depth = 0
    rules_field_inside_rule_count = 0
    mollogp_rule_count = 0
    selection_component_sums = defaultdict(float)
    selection_component_count = 0
    selection_score_total = 0.0
    stage2_generated_token_total = 0
    stage2_base_generated_token_total = 0
    stage2_retry_generated_token_total = 0
    stage2_total_call_count = 0
    stage2_retry_call_count = 0
    draft_rank_counts = defaultdict(int)
    selected_reason_counts = defaultdict(int)
    survivor_rank_counts = defaultdict(int)
    survivor_selected_count = 0
    survivor_novel_count = 0
    underdiverse_trigger_reason_counts = defaultdict(int)
    literature_recall_retry_candidate_count = 0
    literature_recall_retry_kept_count = 0
    literature_recall_novel_root_total = 0
    literature_recall_bonus_total = 0.0
    profile_counts = defaultdict(int)
    profile_source_counts = defaultdict(int)
    profile_version_counts = defaultdict(int)
    termination_policy_counts = defaultdict(int)
    rule_count_policy_counts = defaultdict(int)
    extraction_policy_counts = defaultdict(int)
    canonicalization_policy_counts = defaultdict(int)
    family_feature_policy_counts = defaultdict(int)
    family_target_policy_counts = defaultdict(int)
    extraction_method_counts = defaultdict(int)
    base_single_rule_strict_valid_count = 0
    independent_single_rule_strict_valid_count = 0
    independent_single_rule_compile_valid_count = 0
    independent_single_rule_novel_root_count = 0
    independent_single_rule_kept_count = 0
    independent_single_projection_used_count = 0
    independent_single_projection_kept_count = 0
    original_strict_projector_valid_count = 0
    effective_strict_projector_valid_count = 0
    d4_extraction_used_count = 0
    d4_extraction_success_count = 0
    d4_extraction_strict_valid_count = 0
    d4_any_json_fragment_parseable_count = 0
    d4_extraction_rule_count_before = 0
    d4_extraction_rule_count_after = 0
    d4_suffix_removed_count = 0
    d4_invalid_enum_dropped_count = 0
    d4_unsupported_feature_name_dropped_count = 0
    d4_missing_required_field_dropped_count = 0
    d4_threshold_invalid_dropped_count = 0
    d4_enum_canonicalized_count = 0
    d4_feature_name_canonicalized_count = 0
    d4_canonicalization_failed_count = 0
    d4_non_descriptor_canonicalized_count = 0
    d4_non_descriptor_rescued_by_canonicalization_count = 0
    d4_family_feature_corrected_count = 0
    d4_family_feature_correction_failed_count = 0
    d4_family_feature_threshold_cleared_count = 0
    d4_count_feature_family_corrected_count = 0
    d4_pattern_feature_family_corrected_count = 0
    d4_descriptor_to_count_corrected_count = 0
    d4_descriptor_to_pattern_corrected_count = 0
    d4_non_descriptor_rescued_by_family_consistency_count = 0
    d4_source_overwritten_count = 0
    d4_compile_safe_rule_count = 0
    d4_rules_array_salvage_attempted_count = 0
    d4_rules_array_prefix_found_count = 0
    d4_rules_array_candidate_object_count = 0
    d4_rules_array_parseable_object_count = 0
    d4_rules_array_cleaned_rule_count = 0
    d4_rules_array_salvage_strict_valid_count = 0
    qwen_legacy_d2_effective_strict_count = 0
    family_target_variant_used_count = 0
    family_target_variant_kept_count = 0
    family_target_variant_effective_strict_count = 0
    family_target_variant_compile_safe_count = 0
    family_target_variant_off_target_drop_count = 0
    family_target_variant_multi_rule_violation_count = 0
    family_target_non_descriptor_kept_count = 0
    family_target_non_descriptor_root_keys = set()
    family_target_non_descriptor_feature_families = set()
    target_vocab_policy_counts = Counter()
    target_vocab_variant_used_count = 0
    target_vocab_match_count = 0
    target_vocab_drop_count = 0
    target_vocab_drop_reason_histogram = Counter()
    target_vocab_non_descriptor_kept_count = 0
    target_vocab_non_descriptor_root_keys = set()
    target_vocab_non_descriptor_feature_families = set()
    target_thinning_policy_counts = Counter()
    target_thinning_attempted_count = 0
    target_thinning_success_count = 0
    target_thinning_candidate_count = 0
    target_thinning_valid_candidate_count = 0
    target_thinning_non_descriptor_kept_count = 0
    target_thinning_non_descriptor_root_keys = set()
    target_thinning_non_descriptor_feature_families = set()
    family_target_by_family = {
        family: {
            "used_count": 0,
            "effective_strict_count": 0,
            "compile_safe_count": 0,
            "kept_count": 0,
            "root_keys": set(),
        }
        for family in sorted(CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS)
    }
    target_vocab_by_family = {
        family: {
            "options": [],
            "used_count": 0,
            "match_count": 0,
            "drop_count": 0,
            "kept_count": 0,
            "root_keys": set(),
            "drop_reason_histogram": Counter(),
        }
        for family in sorted(CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS)
    }
    target_thinning_by_family = {
        family: {
            "attempted_count": 0,
            "success_count": 0,
            "selected_roots": set(),
            "selected_feature_names": set(),
            "drop_reason_histogram": Counter(),
        }
        for family in sorted(CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS)
    }
    enum_canonicalized_samples: List[Dict] = []
    feature_name_canonicalized_samples: List[Dict] = []
    canonicalization_failed_samples: List[Dict] = []
    family_feature_corrected_samples: List[Dict] = []
    family_feature_correction_failed_samples: List[Dict] = []
    family_feature_threshold_cleared_samples: List[Dict] = []
    strict_failure_diagnostics = _blank_strict_failure_diagnostics()
    strict_failure_diagnostics_by_source = defaultdict(_blank_strict_failure_diagnostics)
    for item in projected_records:
        status_counts[item.get("projection_status", "unknown")] += 1
        profile_counts[str(item.get("stage2_protocol_profile", "") or "unknown")] += 1
        profile_source_counts[str(item.get("stage2_protocol_profile_source", "") or "unknown")] += 1
        profile_version_counts[str(item.get("stage2_protocol_profile_version", "") or "unknown")] += 1
        if item.get("original_strict_projector_valid", item.get("strict_projector_valid")):
            original_strict_projector_valid_count += 1
        if item.get("effective_strict_projector_valid", item.get("strict_projector_valid")):
            effective_strict_projector_valid_count += 1
        if item.get("qwen_legacy_d2_effective_strict"):
            qwen_legacy_d2_effective_strict_count += 1
        if item.get("stage2_protocol_profile") in {"chat_native_core", "chemdfm_native"}:
            termination_policy_counts[str(item.get("termination_policy", "") or "unknown")] += 1
            rule_count_policy_counts[str(item.get("rule_count_policy", "") or "unknown")] += 1
            family_target_policy_counts[str(item.get("family_target_policy", "none") or "none")] += 1
            target_vocab_policy_counts[str(item.get("target_vocab_policy", "none") or "none")] += 1
            target_thinning_policy_counts[str(item.get("target_thinning_policy", "none") or "none")] += 1
            diagnostics = item.get("chat_native_extraction_diagnostics", {}) or {}
            extraction_policy_counts[str(diagnostics.get("extraction_policy", "") or "unknown")] += 1
            canonicalization_policy_counts[
                str(diagnostics.get("canonicalization_policy", "strict_canonical_only") or "strict_canonical_only")
            ] += 1
            family_feature_policy_counts[
                str(diagnostics.get("family_feature_policy", "strict_family") or "strict_family")
            ] += 1
            extraction_method_counts[str(diagnostics.get("d4_extraction_method", "") or "none")] += 1
            if diagnostics.get("d4_extraction_used"):
                d4_extraction_used_count += 1
                d4_extraction_rule_count_before += int(diagnostics.get("rules_before_extraction", 0) or 0)
                d4_extraction_rule_count_after += int(diagnostics.get("rules_after_extraction", 0) or 0)
                if diagnostics.get("raw_first_json_parseable"):
                    d4_extraction_success_count += 1
                if diagnostics.get("d4_any_json_fragment_parseable"):
                    d4_any_json_fragment_parseable_count += 1
                if diagnostics.get("strict_after_extraction_valid"):
                    d4_extraction_strict_valid_count += 1
                    d4_compile_safe_rule_count += len(_compile_safe_rules(item.get("projector_parsed_rules", []) or []))
                if diagnostics.get("suffix_removed"):
                    d4_suffix_removed_count += 1
                d4_invalid_enum_dropped_count += int(diagnostics.get("invalid_enum_dropped_count", 0) or 0)
                d4_unsupported_feature_name_dropped_count += int(
                    diagnostics.get("unsupported_feature_name_dropped_count", 0) or 0
                )
                d4_missing_required_field_dropped_count += int(
                    diagnostics.get("missing_required_field_dropped_count", 0) or 0
                )
                d4_threshold_invalid_dropped_count += int(diagnostics.get("threshold_invalid_dropped_count", 0) or 0)
                d4_enum_canonicalized_count += int(diagnostics.get("enum_canonicalized_count", 0) or 0)
                d4_feature_name_canonicalized_count += int(
                    diagnostics.get("feature_name_canonicalized_count", 0) or 0
                )
                d4_canonicalization_failed_count += int(diagnostics.get("canonicalization_failed_count", 0) or 0)
                d4_non_descriptor_canonicalized_count += int(
                    diagnostics.get("non_descriptor_canonicalized_count", 0) or 0
                )
                d4_non_descriptor_rescued_by_canonicalization_count += int(
                    diagnostics.get("non_descriptor_rescued_by_canonicalization_count", 0) or 0
                )
                d4_family_feature_corrected_count += int(
                    diagnostics.get("family_feature_corrected_count", 0) or 0
                )
                d4_family_feature_correction_failed_count += int(
                    diagnostics.get("family_feature_correction_failed_count", 0) or 0
                )
                d4_family_feature_threshold_cleared_count += int(
                    diagnostics.get("family_feature_threshold_cleared_count", 0) or 0
                )
                d4_count_feature_family_corrected_count += int(
                    diagnostics.get("count_feature_family_corrected_count", 0) or 0
                )
                d4_pattern_feature_family_corrected_count += int(
                    diagnostics.get("pattern_feature_family_corrected_count", 0) or 0
                )
                d4_descriptor_to_count_corrected_count += int(
                    diagnostics.get("descriptor_to_count_corrected_count", 0) or 0
                )
                d4_descriptor_to_pattern_corrected_count += int(
                    diagnostics.get("descriptor_to_pattern_corrected_count", 0) or 0
                )
                d4_non_descriptor_rescued_by_family_consistency_count += int(
                    diagnostics.get("non_descriptor_rescued_by_family_consistency_count", 0) or 0
                )
                d4_source_overwritten_count += int(diagnostics.get("source_overwritten_count", 0) or 0)
                for sample in diagnostics.get("enum_canonicalized_samples", []) or []:
                    if len(enum_canonicalized_samples) < 10:
                        enum_canonicalized_samples.append(sample)
                for sample in diagnostics.get("feature_name_canonicalized_samples", []) or []:
                    if len(feature_name_canonicalized_samples) < 10:
                        feature_name_canonicalized_samples.append(sample)
                for sample in diagnostics.get("canonicalization_failed_samples", []) or []:
                    if len(canonicalization_failed_samples) < 10:
                        canonicalization_failed_samples.append(sample)
                for sample in diagnostics.get("family_feature_corrected_samples", []) or []:
                    if len(family_feature_corrected_samples) < 10:
                        family_feature_corrected_samples.append(sample)
                for sample in diagnostics.get("family_feature_correction_failed_samples", []) or []:
                    if len(family_feature_correction_failed_samples) < 10:
                        family_feature_correction_failed_samples.append(sample)
                for sample in diagnostics.get("family_feature_threshold_cleared_samples", []) or []:
                    if len(family_feature_threshold_cleared_samples) < 10:
                        family_feature_threshold_cleared_samples.append(sample)
                if diagnostics.get("d4_rules_array_salvage_attempted"):
                    d4_rules_array_salvage_attempted_count += 1
                if diagnostics.get("d4_rules_array_prefix_found"):
                    d4_rules_array_prefix_found_count += 1
                d4_rules_array_candidate_object_count += int(
                    diagnostics.get("d4_rules_array_candidate_object_count", 0) or 0
                )
                d4_rules_array_parseable_object_count += int(
                    diagnostics.get("d4_rules_array_parseable_object_count", 0) or 0
                )
                d4_rules_array_cleaned_rule_count += int(
                    diagnostics.get("d4_rules_array_cleaned_rule_count", 0) or 0
                )
                if diagnostics.get("d4_rules_array_salvage_strict_valid"):
                    d4_rules_array_salvage_strict_valid_count += 1
            role = str(item.get("projection_variant_role", "") or "")
            if role == "base_single_rule" and item.get("strict_projector_valid"):
                base_single_rule_strict_valid_count += 1
            if role == "independent_single_rule_2":
                if item.get("strict_projector_valid"):
                    independent_single_rule_strict_valid_count += 1
                if _compile_valid_ratio(item.get("projector_parsed_rules", [])) > 0.0:
                    independent_single_rule_compile_valid_count += 1
                independent_single_rule_novel_root_count += int(
                    item.get("independent_single_rule_novel_root_count", 0) or 0
                )
                if item.get("independent_single_projection_kept"):
                    independent_single_rule_kept_count += 1
            if role.startswith("family_target_"):
                family_target_variant_used_count += 1
                target_family = str(item.get("family_target_family", "") or "")
                if target_family not in family_target_by_family:
                    family_target_by_family[target_family] = {
                        "used_count": 0,
                        "effective_strict_count": 0,
                        "compile_safe_count": 0,
                        "kept_count": 0,
                        "root_keys": set(),
                    }
                if target_family not in target_vocab_by_family:
                    target_vocab_by_family[target_family] = {
                        "options": [],
                        "used_count": 0,
                        "match_count": 0,
                        "drop_count": 0,
                        "kept_count": 0,
                        "root_keys": set(),
                        "drop_reason_histogram": Counter(),
                    }
                if target_family not in target_thinning_by_family:
                    target_thinning_by_family[target_family] = {
                        "attempted_count": 0,
                        "success_count": 0,
                        "selected_roots": set(),
                        "selected_feature_names": set(),
                        "drop_reason_histogram": Counter(),
                    }
                family_target_by_family[target_family]["used_count"] += 1
                target_vocab_policy = str(item.get("target_vocab_policy", "none") or "none")
                if target_vocab_policy == "micro_canonical_options":
                    target_vocab_variant_used_count += 1
                    target_vocab_by_family[target_family]["used_count"] += 1
                    for option in item.get("target_vocab_options", []) or []:
                        option = str(option)
                        if option not in target_vocab_by_family[target_family]["options"]:
                            target_vocab_by_family[target_family]["options"].append(option)
                    if item.get("target_vocab_match"):
                        target_vocab_match_count += 1
                        target_vocab_by_family[target_family]["match_count"] += 1
                    target_vocab_drop_reason = str(
                        item.get("target_vocab_drop_reason") or item.get("family_target_drop_reason") or ""
                    )
                    if target_vocab_drop_reason and not item.get("family_target_variant_kept"):
                        target_vocab_drop_count += 1
                        target_vocab_drop_reason_histogram[target_vocab_drop_reason] += 1
                        target_vocab_by_family[target_family]["drop_count"] += 1
                        target_vocab_by_family[target_family]["drop_reason_histogram"][target_vocab_drop_reason] += 1
                if item.get("target_thinning_attempted"):
                    target_thinning_attempted_count += 1
                    target_thinning_by_family[target_family]["attempted_count"] += 1
                    target_thinning_candidate_count += int(item.get("target_thinning_candidate_count", 0) or 0)
                    target_thinning_valid_candidate_count += int(
                        item.get("target_thinning_valid_candidate_count", 0) or 0
                    )
                    drop_reason = str(item.get("target_thinning_drop_reason", "") or "")
                    if drop_reason:
                        target_thinning_by_family[target_family]["drop_reason_histogram"][drop_reason] += 1
                    if item.get("target_thinning_success"):
                        target_thinning_success_count += 1
                        target_thinning_by_family[target_family]["success_count"] += 1
                        selected_root = str(item.get("target_thinning_selected_root", "") or "")
                        selected_feature = str(item.get("target_thinning_selected_feature_name", "") or "")
                        selected_family = str(item.get("target_thinning_selected_family", "") or target_family)
                        if selected_root:
                            target_thinning_by_family[target_family]["selected_roots"].add(selected_root)
                        if selected_feature:
                            target_thinning_by_family[target_family]["selected_feature_names"].add(selected_feature)
                        if selected_family != "descriptor":
                            target_thinning_non_descriptor_kept_count += 1
                            target_thinning_non_descriptor_feature_families.add(selected_family)
                            if selected_root:
                                target_thinning_non_descriptor_root_keys.add(selected_root)
                if item.get("effective_strict_projector_valid", item.get("strict_projector_valid")):
                    family_target_variant_effective_strict_count += 1
                    family_target_by_family[target_family]["effective_strict_count"] += 1
                if _compile_valid_ratio(item.get("projector_parsed_rules", [])) > 0.0:
                    family_target_variant_compile_safe_count += 1
                    family_target_by_family[target_family]["compile_safe_count"] += 1
                if item.get("off_target_family_drop"):
                    family_target_variant_off_target_drop_count += 1
                if item.get("multi_rule_target_violation"):
                    family_target_variant_multi_rule_violation_count += 1
                if item.get("family_target_variant_kept"):
                    family_target_variant_kept_count += 1
                    family_target_by_family[target_family]["kept_count"] += 1
                    if target_vocab_policy == "micro_canonical_options":
                        target_vocab_by_family[target_family]["kept_count"] += 1
                    for root in item.get("target_root_keys", []) or []:
                        family_target_by_family[target_family]["root_keys"].add(str(root))
                        if target_vocab_policy == "micro_canonical_options":
                            target_vocab_by_family[target_family]["root_keys"].add(str(root))
                        if target_family != "descriptor":
                            family_target_non_descriptor_root_keys.add(str(root))
                            if target_vocab_policy == "micro_canonical_options":
                                target_vocab_non_descriptor_root_keys.add(str(root))
                    if target_family != "descriptor":
                        family_target_non_descriptor_kept_count += 1
                        family_target_non_descriptor_feature_families.add(target_family)
                        if target_vocab_policy == "micro_canonical_options":
                            target_vocab_non_descriptor_kept_count += 1
                            target_vocab_non_descriptor_feature_families.add(target_family)
        if str(item.get("projection_variant_role", "") or "") == "independent_single_rule_2":
            independent_single_projection_used_count += 1
            if item.get("independent_single_projection_kept"):
                independent_single_projection_kept_count += 1
        if not item.get("strict_projector_valid"):
            _record_strict_failure(strict_failure_diagnostics, item.get("strict_failure"))
            _record_strict_failure(
                strict_failure_diagnostics_by_source[str(item.get("source", "") or "unknown")],
                item.get("strict_failure"),
            )
        if item.get("selected_for_merge", True):
            selected_count += 1
            if int(item.get("draft_rank", 1) or 1) > 1:
                survivor_selected_count += 1
                survivor_rank_counts[int(item.get("draft_rank", 1) or 1)] += 1
                if item.get("late_selection_novelty_pass"):
                    survivor_novel_count += 1
        selected_reason_counts[str(item.get("selected_for_merge_reason", "unknown"))] += 1
        if item.get("repair_used"):
            repair_used_count += 1
        if item.get("repair_success"):
            repair_success_count += 1
        if item.get("underdiverse_retry_used"):
            underdiverse_retry_count += 1
        if item.get("underdiverse_retry_success"):
            underdiverse_retry_success_count += 1
        if item.get("underdiverse_projection"):
            underdiverse_projection_count += 1
        for reason in item.get("underdiverse_trigger_reasons", []) or []:
            underdiverse_trigger_reason_counts[str(reason)] += 1
        if int(item.get("projection_variant_id", 0) or 0) == 0:
            dirty_retry_total += int(item.get("dirty_retry_count", 0))
        if item.get("draft_hygiene_flagged"):
            draft_hygiene_flagged_count += 1
        if int(item.get("projection_variant_id", 0) or 0) == 0:
            if item.get("literature_recall_retry_candidate"):
                literature_recall_retry_candidate_count += 1
            if item.get("literature_recall_retry_kept"):
                literature_recall_retry_kept_count += 1
            literature_recall_novel_root_total += int(item.get("literature_recall_novel_root_count", 0) or 0)
            literature_recall_bonus_total += float(item.get("literature_recall_bonus", 0.0) or 0.0)
        parse_meta = item.get("parse_meta", {})
        max_nesting_depth = max(max_nesting_depth, int(parse_meta.get("max_nesting_depth", 0)))
        rules_field_inside_rule_count += int(parse_meta.get("rules_field_inside_rule_count", 0))
        selection_score_total += float(item.get("selection_score", 0.0) or 0.0)
        draft_rank_counts[int(item.get("draft_rank", 1) or 1)] += 1
        for key, value in (item.get("selection_components", {}) or {}).items():
            selection_component_sums[key] += float(value or 0.0)
        if item.get("selection_components"):
            selection_component_count += 1
        if item.get("schema_valid"):
            parsed_rules = item.get("parsed_rules", [])
            by_source[item.get("source", "unknown")] += len(parsed_rules)
            by_model[item.get("draft_model", "unknown")] += len(parsed_rules)
            for rule in parsed_rules:
                if str(rule.get("feature_name", "")) == "MolLogP":
                    mollogp_rule_count += 1
                if str(rule.get("extraction_source", "projected")) == "heuristic":
                    heuristic_rule_count += 1
                    heuristic_rules_per_source[item.get("source", "unknown")] += 1
                else:
                    projected_rule_count += 1
                    projected_rules_per_source[item.get("source", "unknown")] += 1
            if (
                item.get("strict_projector_valid")
                and not item.get("repair_used")
                and not item.get("heuristic_rescued")
                and _compile_valid_ratio(item.get("projector_parsed_rules", [])) > 0.0
            ):
                strict_scientific_rule_success_count += 1
    for item in base_items:
        stage2_generated_token_total += int(item.get("stage2_generated_token_count", 0) or 0)
        stage2_base_generated_token_total += int(item.get("stage2_base_generated_token_count", 0) or 0)
        stage2_retry_generated_token_total += int(item.get("stage2_retry_generated_token_count", 0) or 0)
        stage2_total_call_count += int(item.get("stage2_total_call_count", 0) or 0)
        stage2_retry_call_count += int(item.get("stage2_retry_call_count", 0) or 0)
    chat_native_core_synth_observability = _synth_path_counts(
        projected_records,
        "chat_native_core",
        branch_name=branch_name,
    )
    chat_native_core_output_discipline_summary = _strict_fail_output_summary(
        projected_records,
        "chat_native_core",
        branch_name,
    )
    chat_native_core_strict_fail_samples = _collect_strict_fail_samples_by_reason(
        projected_records,
        "chat_native_core",
    )
    chat_native_core_base_selected_rules = [
        rule
        for item in projected_records
        if str(item.get("projection_variant_role", "") or "") == "base_single_rule"
        and bool(item.get("selected_for_merge", True))
        for rule in (item.get("parsed_rules", []) or [])
    ]
    chat_native_core_independent_selected_rules = [
        rule
        for item in projected_records
        if str(item.get("projection_variant_role", "") or "") == "independent_single_rule_2"
        and bool(item.get("selected_for_merge", True))
        for rule in (item.get("parsed_rules", []) or [])
    ]
    chat_native_core_family_target_selected_rules = [
        rule
        for item in projected_records
        if str(item.get("projection_variant_role", "") or "").startswith("family_target_")
        and bool(item.get("selected_for_merge", True))
        for rule in (item.get("parsed_rules", []) or [])
    ]
    base_selected_roots = sorted(_late_selection_root_keys(chat_native_core_base_selected_rules))
    family_target_selected_roots = sorted(_late_selection_root_keys(chat_native_core_family_target_selected_rules))
    family_target_non_descriptor_selected_roots = sorted(
        {
            _rule_root_feature_key(rule)
            for rule in chat_native_core_family_target_selected_rules
            if str(rule.get("feature_family", "") or "") != "descriptor"
        }
    )
    base_vs_family_target_comparison = {
        "base_feature_families": _feature_families(chat_native_core_base_selected_rules),
        "family_target_feature_families": _feature_families(chat_native_core_family_target_selected_rules),
        "combined_feature_families": _feature_families(
            list(chat_native_core_base_selected_rules) + list(chat_native_core_family_target_selected_rules)
        ),
        "new_non_descriptor_roots_from_family_targets": sorted(
            set(family_target_non_descriptor_selected_roots) - set(base_selected_roots)
        ),
    }
    family_target_by_family_serialized = {
        family: {
            "used_count": int(values["used_count"]),
            "effective_strict_count": int(values["effective_strict_count"]),
            "compile_safe_count": int(values["compile_safe_count"]),
            "kept_count": int(values["kept_count"]),
            "root_keys": sorted(values["root_keys"]),
        }
        for family, values in family_target_by_family.items()
    }
    target_vocab_by_family_serialized = {
        family: {
            "options": list(values["options"]),
            "used_count": int(values["used_count"]),
            "match_count": int(values["match_count"]),
            "drop_count": int(values["drop_count"]),
            "kept_count": int(values["kept_count"]),
            "root_keys": sorted(values["root_keys"]),
            "drop_reason_histogram": dict(values["drop_reason_histogram"]),
        }
        for family, values in target_vocab_by_family.items()
    }
    target_thinning_by_family_serialized = {
        family: {
            "attempted_count": int(values["attempted_count"]),
            "success_count": int(values["success_count"]),
            "selected_roots": sorted(values["selected_roots"]),
            "selected_feature_names": sorted(values["selected_feature_names"]),
            "drop_reason_histogram": dict(values["drop_reason_histogram"]),
        }
        for family, values in target_thinning_by_family.items()
    }
    llama_native_strict_fail_samples = _collect_llama_native_strict_fail_samples(projected_records, branch_name)
    llama_native_output_discipline_summary = _llama_native_output_discipline_summary(projected_records, branch_name)
    return {
        "num_projected_items": total,
        "schema_valid_count": valid,
        "schema_valid_rate": float(valid / total) if total else 0.0,
        "strict_projector_valid_count": strict_projector_valid,
        "strict_projector_valid_rate": float(strict_projector_valid / total) if total else 0.0,
        "strict_projector_valid_rate_semantics": "effective_when_extraction_policy_enabled",
        "raw_strict_projector_valid_count": int(original_strict_projector_valid_count),
        "raw_strict_projector_valid_rate": (
            float(original_strict_projector_valid_count / total) if total else 0.0
        ),
        "original_strict_projector_valid_count": int(original_strict_projector_valid_count),
        "original_strict_projector_valid_rate": (
            float(original_strict_projector_valid_count / total) if total else 0.0
        ),
        "effective_projector_valid_count": int(effective_strict_projector_valid_count),
        "effective_projector_valid_rate": (
            float(effective_strict_projector_valid_count / total) if total else 0.0
        ),
        "effective_strict_projector_valid_count": int(effective_strict_projector_valid_count),
        "effective_strict_projector_valid_rate": (
            float(effective_strict_projector_valid_count / total) if total else 0.0
        ),
        "qwen_legacy_d2_effective_strict_count": int(qwen_legacy_d2_effective_strict_count),
        "qwen_legacy_d2_effective_strict_rate": (
            float(qwen_legacy_d2_effective_strict_count / total) if total else 0.0
        ),
        "strict_after_extraction_rate": float(d4_extraction_strict_valid_count / total) if total else 0.0,
        "strict_valid_rule_count": len(strict_valid_rules),
        "strict_valid_root_keys": strict_valid_root_keys,
        "strict_valid_feature_families": strict_valid_feature_families,
        "strict_valid_family_count": len(strict_valid_feature_families),
        "projector_valid_count": projector_valid,
        "projector_valid_rate": float(projector_valid / total) if total else 0.0,
        "heuristic_rescue_count": heuristic_rescues,
        "heuristic_rescue_rate": float(heuristic_rescues / total) if total else 0.0,
        "projection_status_counts": dict(status_counts),
        "stage2_protocol_profile_counts": dict(profile_counts),
        "stage2_protocol_profile_source_counts": dict(profile_source_counts),
        "stage2_protocol_profile_version_counts": dict(profile_version_counts),
        "chat_native_core_termination_policy_counts": dict(termination_policy_counts),
        "chat_native_core_rule_count_policy_counts": dict(rule_count_policy_counts),
        "chat_native_core_extraction_policy_counts": dict(extraction_policy_counts),
        "chat_native_core_canonicalization_policy_counts": dict(canonicalization_policy_counts),
        "stage2_extraction_policy_counts": dict(extraction_policy_counts),
        "stage2_canonicalization_policy_counts": dict(canonicalization_policy_counts),
        "stage2_family_feature_policy_counts": dict(family_feature_policy_counts),
        "chemdfm_native_extraction_policy_counts": dict(extraction_policy_counts),
        "chemdfm_native_canonicalization_policy_counts": dict(canonicalization_policy_counts),
        "chemdfm_native_family_feature_policy_counts": dict(family_feature_policy_counts),
        "chat_native_core_family_target_policy_counts": dict(family_target_policy_counts),
        "stage2_family_target_policy_counts": dict(family_target_policy_counts),
        "chemdfm_native_family_target_policy_counts": dict(family_target_policy_counts),
        "target_vocab_policy_counts": dict(target_vocab_policy_counts),
        "target_thinning_policy_counts": dict(target_thinning_policy_counts),
        "d4_extraction_method_counts": dict(extraction_method_counts),
        "d4_extraction_used_count": int(d4_extraction_used_count),
        "d4_extraction_success_count": int(d4_extraction_success_count),
        "d4_extraction_parseable_count": int(d4_extraction_success_count),
        "d4_extraction_strict_valid_count": int(d4_extraction_strict_valid_count),
        "d4_any_json_fragment_parseable_count": int(d4_any_json_fragment_parseable_count),
        "d4_extraction_parseable_rate_on_attempted": (
            float(d4_extraction_success_count / d4_extraction_used_count) if d4_extraction_used_count else 0.0
        ),
        "d4_extraction_strict_valid_rate_on_attempted": (
            float(d4_extraction_strict_valid_count / d4_extraction_used_count) if d4_extraction_used_count else 0.0
        ),
        "d4_extraction_rule_count_before": int(d4_extraction_rule_count_before),
        "d4_extraction_rule_count_after": int(d4_extraction_rule_count_after),
        "d4_suffix_removed_count": int(d4_suffix_removed_count),
        "d4_invalid_enum_dropped_count": int(d4_invalid_enum_dropped_count),
        "d4_unsupported_feature_name_dropped_count": int(d4_unsupported_feature_name_dropped_count),
        "d4_missing_required_field_dropped_count": int(d4_missing_required_field_dropped_count),
        "d4_threshold_invalid_dropped_count": int(d4_threshold_invalid_dropped_count),
        "d4_enum_canonicalized_count": int(d4_enum_canonicalized_count),
        "d4_enum_canonicalized_samples": enum_canonicalized_samples,
        "d4_feature_name_canonicalized_count": int(d4_feature_name_canonicalized_count),
        "d4_feature_name_canonicalized_samples": feature_name_canonicalized_samples,
        "d4_canonicalization_failed_count": int(d4_canonicalization_failed_count),
        "d4_canonicalization_failed_samples": canonicalization_failed_samples,
        "d4_non_descriptor_canonicalized_count": int(d4_non_descriptor_canonicalized_count),
        "d4_non_descriptor_rescued_by_canonicalization_count": int(
            d4_non_descriptor_rescued_by_canonicalization_count
        ),
        "non_descriptor_canonicalized_count": int(d4_non_descriptor_canonicalized_count),
        "non_descriptor_rescued_by_canonicalization_count": int(
            d4_non_descriptor_rescued_by_canonicalization_count
        ),
        "d4_family_feature_corrected_count": int(d4_family_feature_corrected_count),
        "family_feature_corrected_count": int(d4_family_feature_corrected_count),
        "d4_family_feature_corrected_samples": family_feature_corrected_samples,
        "family_feature_corrected_samples": family_feature_corrected_samples,
        "d4_family_feature_correction_failed_count": int(d4_family_feature_correction_failed_count),
        "family_feature_correction_failed_count": int(d4_family_feature_correction_failed_count),
        "d4_family_feature_correction_failed_samples": family_feature_correction_failed_samples,
        "family_feature_correction_failed_samples": family_feature_correction_failed_samples,
        "d4_family_feature_threshold_cleared_count": int(d4_family_feature_threshold_cleared_count),
        "family_feature_threshold_cleared_count": int(d4_family_feature_threshold_cleared_count),
        "d4_family_feature_threshold_cleared_samples": family_feature_threshold_cleared_samples,
        "family_feature_threshold_cleared_samples": family_feature_threshold_cleared_samples,
        "d4_count_feature_family_corrected_count": int(d4_count_feature_family_corrected_count),
        "count_feature_family_corrected_count": int(d4_count_feature_family_corrected_count),
        "d4_pattern_feature_family_corrected_count": int(d4_pattern_feature_family_corrected_count),
        "pattern_feature_family_corrected_count": int(d4_pattern_feature_family_corrected_count),
        "d4_descriptor_to_count_corrected_count": int(d4_descriptor_to_count_corrected_count),
        "descriptor_to_count_corrected_count": int(d4_descriptor_to_count_corrected_count),
        "d4_descriptor_to_pattern_corrected_count": int(d4_descriptor_to_pattern_corrected_count),
        "descriptor_to_pattern_corrected_count": int(d4_descriptor_to_pattern_corrected_count),
        "d4_non_descriptor_rescued_by_family_consistency_count": int(
            d4_non_descriptor_rescued_by_family_consistency_count
        ),
        "non_descriptor_rescued_by_family_consistency_count": int(
            d4_non_descriptor_rescued_by_family_consistency_count
        ),
        "d4_source_overwritten_count": int(d4_source_overwritten_count),
        "d4_compile_safe_rule_count": int(d4_compile_safe_rule_count),
        "d4_rules_array_salvage_attempted_count": int(d4_rules_array_salvage_attempted_count),
        "d4_rules_array_prefix_found_count": int(d4_rules_array_prefix_found_count),
        "d4_rules_array_candidate_object_count": int(d4_rules_array_candidate_object_count),
        "d4_rules_array_parseable_object_count": int(d4_rules_array_parseable_object_count),
        "d4_rules_array_cleaned_rule_count": int(d4_rules_array_cleaned_rule_count),
        "d4_rules_array_salvage_strict_valid_count": int(d4_rules_array_salvage_strict_valid_count),
        "d4_rules_array_salvage_strict_valid_rate_on_attempted": (
            float(d4_rules_array_salvage_strict_valid_count / d4_rules_array_salvage_attempted_count)
            if d4_rules_array_salvage_attempted_count
            else 0.0
        ),
        "base_single_rule_strict_valid_count": int(base_single_rule_strict_valid_count),
        "independent_single_rule_strict_valid_count": int(independent_single_rule_strict_valid_count),
        "independent_single_rule_compile_valid_count": int(independent_single_rule_compile_valid_count),
        "independent_single_rule_novel_root_count": int(independent_single_rule_novel_root_count),
        "independent_single_rule_kept_count": int(independent_single_rule_kept_count),
        "independent_single_projection_used_count": int(independent_single_projection_used_count),
        "independent_single_projection_kept_count": int(independent_single_projection_kept_count),
        "family_target_variant_used_count": int(family_target_variant_used_count),
        "family_target_variant_kept_count": int(family_target_variant_kept_count),
        "family_target_variant_effective_strict_count": int(family_target_variant_effective_strict_count),
        "family_target_variant_compile_safe_count": int(family_target_variant_compile_safe_count),
        "family_target_variant_off_target_drop_count": int(family_target_variant_off_target_drop_count),
        "family_target_variant_multi_rule_violation_count": int(family_target_variant_multi_rule_violation_count),
        "family_target_non_descriptor_kept_count": int(family_target_non_descriptor_kept_count),
        "family_target_non_descriptor_root_keys": sorted(family_target_non_descriptor_root_keys),
        "family_target_non_descriptor_feature_families": sorted(family_target_non_descriptor_feature_families),
        "family_target_by_family": family_target_by_family_serialized,
        "target_vocab_variant_used_count": int(target_vocab_variant_used_count),
        "target_vocab_match_count": int(target_vocab_match_count),
        "target_vocab_drop_count": int(target_vocab_drop_count),
        "target_vocab_drop_reason_histogram": dict(target_vocab_drop_reason_histogram),
        "target_vocab_by_family": target_vocab_by_family_serialized,
        "target_vocab_non_descriptor_kept_count": int(target_vocab_non_descriptor_kept_count),
        "target_vocab_non_descriptor_root_keys": sorted(target_vocab_non_descriptor_root_keys),
        "target_vocab_non_descriptor_feature_families": sorted(target_vocab_non_descriptor_feature_families),
        "target_thinning_attempted_count": int(target_thinning_attempted_count),
        "target_thinning_success_count": int(target_thinning_success_count),
        "target_thinning_candidate_count": int(target_thinning_candidate_count),
        "target_thinning_valid_candidate_count": int(target_thinning_valid_candidate_count),
        "target_thinning_success_rate_on_attempted": (
            float(target_thinning_success_count / target_thinning_attempted_count)
            if target_thinning_attempted_count
            else 0.0
        ),
        "target_thinning_by_family": target_thinning_by_family_serialized,
        "target_thinning_non_descriptor_kept_count": int(target_thinning_non_descriptor_kept_count),
        "target_thinning_non_descriptor_root_keys": sorted(target_thinning_non_descriptor_root_keys),
        "target_thinning_non_descriptor_feature_families": sorted(target_thinning_non_descriptor_feature_families),
        "base_vs_family_target_comparison": base_vs_family_target_comparison,
        "base_single_rule_selected_root_keys": base_selected_roots,
        "independent_single_rule_selected_root_keys": sorted(
            _late_selection_root_keys(chat_native_core_independent_selected_rules)
        ),
        "family_target_selected_root_keys": family_target_selected_roots,
        "base_single_rule_selected_feature_families": _feature_families(chat_native_core_base_selected_rules),
        "independent_single_rule_selected_feature_families": _feature_families(chat_native_core_independent_selected_rules),
        "family_target_selected_feature_families": _feature_families(chat_native_core_family_target_selected_rules),
        "rules_per_source": dict(by_source),
        "rules_per_draft_model": dict(by_model),
        "projected_rules_per_source": dict(projected_rules_per_source),
        "heuristic_rules_per_source": dict(heuristic_rules_per_source),
        "heuristic_rule_count": heuristic_rule_count,
        "projected_rule_count": projected_rule_count,
        "strict_scientific_rule_success_count": strict_scientific_rule_success_count,
        "strict_scientific_rule_success_rate": float(strict_scientific_rule_success_count / total) if total else 0.0,
        "heuristic_rule_rate": float(heuristic_rule_count / max(1, heuristic_rule_count + projected_rule_count)),
        "repair_projector_valid_count": int(repair_success_count),
        "repair_projector_valid_rate": float(repair_success_count / total) if total else 0.0,
        "repair_used_count": repair_used_count,
        "repair_used_rate": float(repair_used_count / total) if total else 0.0,
        "repair_success_count": repair_success_count,
        "repair_success_rate": float(repair_success_count / max(1, repair_used_count)) if repair_used_count else 0.0,
        "underdiverse_retry_count": underdiverse_retry_count,
        "underdiverse_retry_rate": float(underdiverse_retry_count / total) if total else 0.0,
        "underdiverse_retry_success_count": underdiverse_retry_success_count,
        "underdiverse_retry_success_rate": (
            float(underdiverse_retry_success_count / max(1, underdiverse_retry_count)) if underdiverse_retry_count else 0.0
        ),
        "underdiverse_projection_count": underdiverse_projection_count,
        "underdiverse_projection_rate": float(underdiverse_projection_count / total) if total else 0.0,
        "underdiverse_trigger_reason_counts": dict(underdiverse_trigger_reason_counts),
        "late_selection_count": selected_count,
        "late_selection_survivor_count": survivor_selected_count,
        "late_selection_survivor_rank_distribution": dict(survivor_rank_counts),
        "late_selection_reason_counts": dict(selected_reason_counts),
        "late_selection_novel_selected_count": survivor_novel_count,
        "late_selection_config": {
            "keep_topk": int(generation_option("LATE_SELECTION_KEEP_TOPK", LATE_SELECTION_KEEP_TOPK)),
            "max_score_delta": float(generation_option("LATE_SELECTION_MAX_SCORE_DELTA", LATE_SELECTION_MAX_SCORE_DELTA)),
            "max_survivors_per_prompt": int(generation_option("LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT", LATE_SELECTION_MAX_SURVIVORS_PER_PROMPT)),
            "require_novelty": bool(generation_option("LATE_SELECTION_REQUIRE_NOVELTY", LATE_SELECTION_REQUIRE_NOVELTY)),
        },
        "literature_recall_retry_candidate_count": literature_recall_retry_candidate_count,
        "literature_recall_retry_kept_count": literature_recall_retry_kept_count,
        "literature_recall_novel_root_total": literature_recall_novel_root_total,
        "literature_recall_bonus_mean": float(literature_recall_bonus_total / max(1, total)),
        "literature_recall_config": {
            "retry_enable": bool(generation_option("LITERATURE_RECALL_RETRY_ENABLE", LITERATURE_RECALL_RETRY_ENABLE)),
            "bonus": float(generation_option("LITERATURE_RECALL_BONUS", LITERATURE_RECALL_BONUS)),
        },
        "dirty_retry_count": dirty_retry_total,
        "draft_hygiene_flagged_count": draft_hygiene_flagged_count,
        "max_nesting_depth": max_nesting_depth,
        "rules_field_inside_rule_count": rules_field_inside_rule_count,
        "mollogp_collapse_rate": float(mollogp_rule_count / max(1, heuristic_rule_count + projected_rule_count)),
        "feature_family_diversity": _feature_family_diversity_count(flattened_rules),
        "compile_safe_count": len(_compile_safe_rules(flattened_rules)),
        "root_feature_count": len(_late_selection_root_keys(flattened_rules)),
        "family_diversity": _feature_family_diversity_count(flattened_rules),
        "dominant_root_share": (
            float(max(flattened_root_counts.values()) / max(1, len(flattened_rules)))
            if flattened_rules and flattened_root_counts
            else 0.0
        ),
        "descriptor_only": bool(flattened_rules)
        and all(str(rule.get("feature_family", "")) == "descriptor" for rule in flattened_rules),
        "selection_score_mean": float(selection_score_total / total) if total else 0.0,
        "stage2_generated_token_total": int(stage2_generated_token_total),
        "stage2_generated_token_mean": float(stage2_generated_token_total / max(1, len(base_items))) if base_items else 0.0,
        "stage2_base_generated_token_total": int(stage2_base_generated_token_total),
        "stage2_retry_generated_token_total": int(stage2_retry_generated_token_total),
        "stage2_total_call_count": int(stage2_total_call_count),
        "stage2_retry_call_count": int(stage2_retry_call_count),
        "draft_rank_distribution": dict(draft_rank_counts),
        "selection_component_means": {
            key: float(value / max(1, selection_component_count))
            for key, value in selection_component_sums.items()
        },
        "strict_failure_diagnostics": {
            "total_failures": int(strict_failure_diagnostics.get("total_failures", 0) or 0),
            "stage_counts": _normalize_failure_counter_map(strict_failure_diagnostics.get("stage_counts", {})),
            "reason_counts": _normalize_failure_counter_map(strict_failure_diagnostics.get("reason_counts", {})),
            "label_counts": _normalize_failure_counter_map(strict_failure_diagnostics.get("label_counts", {})),
            "by_stage": {
                str(stage): _normalize_failure_counter_map(reason_map)
                for stage, reason_map in (strict_failure_diagnostics.get("by_stage", {}) or {}).items()
            },
        },
        "strict_failure_diagnostics_by_source": {
            str(source): {
                "total_failures": int(source_diag.get("total_failures", 0) or 0),
                "stage_counts": _normalize_failure_counter_map(source_diag.get("stage_counts", {})),
                "reason_counts": _normalize_failure_counter_map(source_diag.get("reason_counts", {})),
                "label_counts": _normalize_failure_counter_map(source_diag.get("label_counts", {})),
                "by_stage": {
                    str(stage): _normalize_failure_counter_map(reason_map)
                    for stage, reason_map in (source_diag.get("by_stage", {}) or {}).items()
                },
            }
            for source, source_diag in strict_failure_diagnostics_by_source.items()
        },
        "chat_native_core_synth_observability": chat_native_core_synth_observability,
        "chat_native_core_output_discipline_summary": chat_native_core_output_discipline_summary,
        "chat_native_core_strict_fail_samples": chat_native_core_strict_fail_samples,
        "llama_native_strict_fail_samples": llama_native_strict_fail_samples,
        "llama_native_output_discipline_summary": llama_native_output_discipline_summary,
    }
