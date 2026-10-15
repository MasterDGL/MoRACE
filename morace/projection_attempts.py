"""Execute one Stage-2 projection attempt, including its optional repair pass."""

from morace.generation_runtime import generation_overrides, science_env_get
from typing import Dict, List

from morace.model_backend import LocalTextGenerator
from morace.projection_parsing import (
    extract_cleaned_rule_json,
    find_rules_array_start,
    new_extraction_diagnostics,
    salvage_rules_array_prefix,
)
from morace.projection_prompts import (
    build_stage2_repair_prompt,
    stage2_repair_system_prompt,
    stage2_stop_strings,
)
from morace.schemas import lenient_parse_projected_rules, strict_parse_projected_rules
from morace.stage2_protocol import (
    chat_native_core_canonicalization_policy,
    chat_native_core_extraction_policy,
    stage2_canonicalization_policy,
    stage2_extraction_policy,
    stage2_family_feature_policy,
)


def run_projection_attempt(
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
    strict_parser=None,
    lenient_parser=None,
) -> Dict:
    strict_parser = strict_parser or strict_parse_projected_rules
    lenient_parser = lenient_parser or lenient_parse_projected_rules
    stop_strings = stage2_stop_strings(stage2_protocol_profile)
    raw_result = backend.generate_structured(
        [(system_prompt, user_prompt)],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        num_beams=num_beams,
        max_input_tokens=max_input_tokens if max_input_tokens > 0 else None,
        preserve_head_tokens=256,
        preserve_tail_tokens=512,
        stop_strings=stop_strings,
        stage2_protocol_profile=stage2_protocol_profile,
        return_metadata=True,
    )[0]
    raw_output = raw_result["text"]
    strict_rules, strict_parse_meta = strict_parser(
        raw_output,
        source=source,
        provenance=provenance,
    )
    lenient_rules, lenient_parse_meta = lenient_parser(
        raw_output,
        source=source,
        provenance=provenance,
    )
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "chat_native_core":
        extraction_policy = chat_native_core_extraction_policy()
        canonicalization_policy = chat_native_core_canonicalization_policy()
        family_feature_policy = "strict_family"
    elif profile == "chemdfm_native":
        extraction_policy = stage2_extraction_policy()
        canonicalization_policy = stage2_canonicalization_policy()
        family_feature_policy = stage2_family_feature_policy()
    else:
        extraction_policy = "strict_only"
        canonicalization_policy = "strict_canonical_only"
        family_feature_policy = "strict_family"
    d4_cleaned_json = ""
    d4_strict_rules: List[Dict] = []
    d4_strict_parse_meta = {
        "max_nesting_depth": 0,
        "rules_field_inside_rule_count": 0,
        "failure_stage": "",
        "failure_reason": "",
        "failure_label": "",
    }
    d4_diagnostics = new_extraction_diagnostics(
        extraction_policy,
        canonicalization_policy,
        family_feature_policy,
    )
    d4_diagnostics["original_strict_valid"] = bool(strict_rules)
    if extraction_policy == "tolerant_first_json_object" and not strict_rules:
        d4_cleaned_json, d4_diagnostics = extract_cleaned_rule_json(
            raw_output,
            source,
            extraction_policy,
            canonicalization_policy,
            family_feature_policy,
        )
        d4_diagnostics["original_strict_valid"] = False
        if d4_cleaned_json:
            d4_strict_rules, d4_strict_parse_meta = strict_parser(
                d4_cleaned_json,
                source=source,
                provenance=provenance,
            )
            d4_diagnostics["strict_after_extraction_valid"] = bool(d4_strict_rules)
        salvage_start_index, _ = find_rules_array_start(raw_output)
        if not d4_strict_rules and salvage_start_index >= 0:
            salvage_cleaned_json, salvage_diagnostics = salvage_rules_array_prefix(
                raw_output,
                source,
                extraction_policy,
                canonicalization_policy,
                family_feature_policy,
            )
            if salvage_diagnostics.get("d4_rules_array_salvage_attempted"):
                d4_cleaned_json = salvage_cleaned_json
                d4_diagnostics = salvage_diagnostics
                d4_diagnostics["original_strict_valid"] = False
                if d4_cleaned_json:
                    d4_strict_rules, d4_strict_parse_meta = strict_parser(
                        d4_cleaned_json,
                        source=source,
                        provenance=provenance,
                    )
                    d4_diagnostics["strict_after_extraction_valid"] = bool(d4_strict_rules)
                    d4_diagnostics["d4_rules_array_salvage_strict_valid"] = bool(d4_strict_rules)
    projected_rules = strict_rules or d4_strict_rules or lenient_rules
    repair_output = ""
    repair_used = False
    repair_generated_token_count = 0
    repair_strict_rules: List[Dict] = []
    repair_lenient_rules: List[Dict] = []
    repair_parse_meta = {
        "max_nesting_depth": 0,
        "rules_field_inside_rule_count": 0,
        "failure_stage": "",
        "failure_reason": "",
        "failure_label": "",
    }
    repair_lenient_parse_meta = {
        "max_nesting_depth": 0,
        "rules_field_inside_rule_count": 0,
        "failure_stage": "",
        "failure_reason": "",
        "failure_label": "",
    }
    if not projected_rules and allow_repair_pass:
        repair_used = True
        with generation_overrides(_MORACE_STAGE2_PROTOCOL_PROFILE=str(stage2_protocol_profile or "")):
            repair_result = backend.generate_structured(
                [
                    (
                        stage2_repair_system_prompt(stage2_protocol_profile),
                        build_stage2_repair_prompt(
                            raw_output=raw_output,
                            source=source,
                            projector_family=projector_family,
                        ),
                    )
                ],
                max_new_tokens=min(max_new_tokens, 220),
                temperature=0.0,
                top_p=1.0,
                num_beams=max(1, num_beams),
                max_input_tokens=max_input_tokens if max_input_tokens and max_input_tokens > 0 else None,
                preserve_head_tokens=64,
                preserve_tail_tokens=768,
                stop_strings=stop_strings,
                stage2_protocol_profile=stage2_protocol_profile,
                return_metadata=True,
            )[0]
        repair_output = repair_result["text"]
        repair_generated_token_count = int(repair_result.get("generated_token_count", 0) or 0)
        repair_strict_rules, repair_parse_meta = strict_parser(
            repair_output,
            source=source,
            provenance=provenance,
        )
        repair_lenient_rules, repair_lenient_parse_meta = lenient_parser(
            repair_output,
            source=source,
            provenance=provenance,
        )
        repair_parse_meta["max_nesting_depth"] = max(
            repair_parse_meta["max_nesting_depth"],
            repair_lenient_parse_meta["max_nesting_depth"],
        )
        repair_parse_meta["rules_field_inside_rule_count"] += repair_lenient_parse_meta[
            "rules_field_inside_rule_count"
        ]
        projected_rules = repair_strict_rules or repair_lenient_rules
    d4_diagnostics["original_strict_valid"] = bool(strict_rules or repair_strict_rules)
    parse_meta = {
        "max_nesting_depth": max(
            strict_parse_meta.get("max_nesting_depth", 0),
            d4_strict_parse_meta.get("max_nesting_depth", 0),
            lenient_parse_meta.get("max_nesting_depth", 0),
            repair_parse_meta.get("max_nesting_depth", 0),
        ),
        "rules_field_inside_rule_count": (
            strict_parse_meta.get("rules_field_inside_rule_count", 0)
            + d4_strict_parse_meta.get("rules_field_inside_rule_count", 0)
            + lenient_parse_meta.get("rules_field_inside_rule_count", 0)
            + repair_parse_meta.get("rules_field_inside_rule_count", 0)
        ),
    }
    return {
        "raw_output": raw_output,
        "repair_output": repair_output,
        "projected_rules": projected_rules,
        "strict_rules": strict_rules,
        "strict_parse_meta": strict_parse_meta,
        "d4_cleaned_json": d4_cleaned_json,
        "d4_strict_rules": d4_strict_rules,
        "d4_strict_parse_meta": d4_strict_parse_meta,
        "chat_native_extraction_diagnostics": d4_diagnostics,
        "lenient_rules": lenient_rules,
        "lenient_parse_meta": lenient_parse_meta,
        "repair_strict_rules": repair_strict_rules,
        "repair_strict_parse_meta": repair_parse_meta,
        "repair_lenient_rules": repair_lenient_rules,
        "repair_lenient_parse_meta": repair_lenient_parse_meta,
        "repair_used": repair_used,
        "parse_meta": parse_meta,
        "stage2_backend": raw_result.get("backend", science_env_get("STAGE2_BACKEND", "prompt_only")),
        "stage2_avg_logprob": raw_result.get("avg_logprob"),
        "stage2_logprob_norm": raw_result.get("logprob_norm", 0.0),
        "stage2_projection_generated_token_count": int(raw_result.get("generated_token_count", 0) or 0),
        "stage2_repair_generated_token_count": repair_generated_token_count,
        "stage2_generated_token_count": (
            int(raw_result.get("generated_token_count", 0) or 0) + repair_generated_token_count
        ),
        "stage2_projection_call_count": 1,
        "stage2_repair_call_count": int(repair_used),
        "stage2_total_call_count": 1 + int(repair_used),
    }
