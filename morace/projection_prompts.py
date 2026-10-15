"""Prompt templates and builders for the two-stage rule projection workflow."""

from morace.generation_runtime import science_env_get
from typing import List, Sequence

from morace.feature_vocabulary import registry_supported_names
from morace.model_backend import is_non_qwen_family
from morace.schemas import END_JSON_MARKER, build_rule_schema_text
from morace.stage2_protocol import (
    CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS,
    CHAT_NATIVE_CORE_TARGET_FORBIDDEN_ALIASES,
    DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
    chat_native_core_termination_policy,
)


STAGE1_SYSTEM_PROMPT = (
    "You are an experienced chemist working on molecular property prediction. "
    "Produce a chemistry draft, not a final rule set. Abstract from examples into descriptors, motifs, "
    "ring systems, ionization, polarity, lipophilicity, hydrogen-bonding, steric bulk, and countable motifs. "
    "Do not copy training examples verbatim. Do not output JSON."
)

STAGE2_SYSTEM_PROMPT = (
    "You are the Stage 2 projector for structured scientific rules. "
    "Only project determinants that already appear in the chemistry draft. "
    "Do not rediscover, paraphrase, or invent new chemistry. Return JSON only, with no explanation, no thinking process, "
    "and no text before or after the JSON. If information is unsupported, use null or omit the determinant."
)

STAGE2_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed scientific rule JSON. Return only one valid JSON object with a top-level 'rules' array, "
    f"then write {END_JSON_MARKER}. Never include nested 'rules' fields inside a rule."
)

STAGE2_LITERATURE_SYSTEM_PROMPT = (
    "You are the Stage 2 projector for literature-derived chemistry rules. "
    "Return exactly one JSON object and nothing else. No markdown, no prose, no prefixes, no suffixes. "
    "Each item in rules must describe exactly one rule family only. "
    "If the draft does not support a clean rule, return {\"rules\": []} then END_JSON."
)

STAGE2_CHEMDFM_SYSTEM_PROMPT = (
    "You are the Stage 2 projector for native ChemDFM structured chemistry rules. "
    "Return exactly one JSON object with a top-level 'rules' array and nothing else. "
    "Do not write labels, headings, markdown, code fences, explanations, or stop markers. "
    "The first character must be { and the final character must be }."
)

STAGE2_CHEMDFM_REPAIR_SYSTEM_PROMPT = (
    "You repair malformed native ChemDFM scientific rule JSON. "
    "Return only one valid JSON object with a top-level 'rules' array. "
    "Repair existing shape and enum/vocabulary issues only; do not invent rules or feature names. "
    "If repair is uncertain, return {\"rules\":[]}."
)



def stage2_stop_strings(stage2_protocol_profile: str) -> List[str]:
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "llama_native":
        return ["```"]
    if profile == "chat_native_core":
        return [END_JSON_MARKER] if chat_native_core_termination_policy() == "end_json_marker_single_rule" else []
    return [END_JSON_MARKER]


def stage2_max_new_tokens(stage2_protocol_profile: str, requested_max_new_tokens: int) -> int:
    profile = str(stage2_protocol_profile or "").strip()
    requested = int(requested_max_new_tokens or 0)
    if profile == "llama_native":
        return max(96, min(requested, 256)) if requested > 0 else 192
    if profile == "chat_native_core":
        return max(96, min(requested, 160)) if requested > 0 else 128
    return requested


def stage2_system_prompt(source: str, stage2_protocol_profile: str) -> str:
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "chemdfm_native":
        return STAGE2_CHEMDFM_SYSTEM_PROMPT
    if profile == "chat_native_core":
        if chat_native_core_termination_policy() == "end_json_marker_single_rule":
            return (
                "You are the Stage 2 projector for generic native chat-model ablation. "
                "Return one JSON object followed by END_JSON. No markdown, explanations, prefixes, or suffixes."
            )
        return (
            "You are the Stage 2 projector for generic native chat-model ablation. "
            "Return one JSON object only. No markdown, explanations, prefixes, suffixes, or stop markers."
        )
    return STAGE2_LITERATURE_SYSTEM_PROMPT if source == "literature" else STAGE2_SYSTEM_PROMPT


def stage2_repair_system_prompt(stage2_protocol_profile: str) -> str:
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "chemdfm_native":
        return STAGE2_CHEMDFM_REPAIR_SYSTEM_PROMPT
    if profile == "chat_native_core":
        if chat_native_core_termination_policy() == "end_json_marker_single_rule":
            return (
                "You repair malformed scientific rule JSON. Return only one valid JSON object "
                "with a top-level 'rules' array, then write END_JSON."
            )
        return (
            "You repair malformed scientific rule JSON. Return only one valid JSON object "
            "with a top-level 'rules' array. Do not write any stop marker."
        )
    return STAGE2_REPAIR_SYSTEM_PROMPT


def _chemdfm_native_closure_baseline_constraints() -> str:
    return (
        "Return exactly one JSON object.\n"
        "Top-level object must contain \"rules\".\n"
        "Prefer 1 short clean rule.\n"
        "Never force a second rule.\n"
        "A single closed rule is better than multiple incomplete rules.\n"
        "If no clean rule is supported, return exactly {\"rules\":[]}.\n"
        "Allowed feature_family: descriptor, pattern_presence, count.\n"
        "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
        "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
        "Use JSON null, not the string \"null\".\n"
        "Use threshold only when feature_mode is threshold_binary.\n"
        "For numeric_value, binary_presence, or count_value, set \"threshold\": null.\n"
        "If unsure about threshold, avoid threshold_binary.\n"
        "rationale must be 1-3 words. No sentence. No comma. No Chinese. No explanation."
    )




def _chat_native_core_compact_vocabulary_block() -> str:
    descriptors = registry_supported_names(
        "DESCRIPTOR_REGISTRY",
        ("MolWt", "MolLogP", "TPSA", "NumHBD", "NumHBA", "NumRotatableBonds", "RingCount", "NumAromaticRings"),
    )
    patterns = registry_supported_names(
        "PATTERN_ALIAS",
        ("aromatic_ring", "amide", "amine", "halogen", "ester", "ether", "carbonyl", "carboxyl"),
    )
    custom_patterns = registry_supported_names(
        "CUSTOM_PATTERN_FUNCTIONS",
        ("aromatic_ring", "fused_ring"),
    )
    for pattern_name in custom_patterns:
        if pattern_name not in patterns:
            patterns.append(pattern_name)
    counts = registry_supported_names(
        "COUNT_ALIAS",
        ("RingCount", "NumAromaticRings", "NumAliphaticRings", "NumRotatableBonds", "NumHBD", "NumHBA"),
    )
    return (
        "Allowed descriptor feature_name: " + ", ".join(descriptors) + ".\n"
        "Allowed pattern_presence feature_name: " + ", ".join(patterns) + ".\n"
        "Allowed count feature_name: " + ", ".join(counts) + "."
    )


def _chat_native_core_vocab_rescue_constraints() -> str:
    return (
        "Allowed feature_family: descriptor, pattern_presence, count.\n"
        "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
        '"count" is allowed only as feature_family; for feature_mode use "count_value", never "count".\n'
        "Never use feature_mode values: presence, binary, continuous, count, numeric, boolean.\n"
        "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
        'Use JSON null, not the string "null".\n'
        f"{_chat_native_core_compact_vocabulary_block()}\n"
        "Choose feature_name only from the allowed vocabulary above.\n"
        "If no allowed feature_name fits, return {\"rules\":[]}.\n"
        "Do not invent new feature_name values."
    )








def build_stage1_user_prompt(base_prompt: str, extra_context: str = "") -> str:
    if extra_context:
        return f"{base_prompt.strip()}\n\nContext for this draft:\n{extra_context.strip()}"
    return base_prompt.strip()


def build_stage2_user_prompt(
    task_brief: str,
    draft_text: str,
    source: str,
    short_schema: bool = True,
    projector_family: str = "generic",
    stage2_protocol_profile: str = "",
) -> str:
    schema_text = build_rule_schema_text(short_form=short_schema)
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "cross_qwen_for_non_qwen_draft":
        source_value = "literature" if source == "literature" else "data"
        return (
            "You are normalizing a non-Qwen chemistry draft into strict scientific rule JSON.\n"
            "The input draft may contain envelope text, explanation, field drift, and polluted source strings.\n"
            "Return JSON only. No markdown. No prose. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            "Allowed source values: literature, data.\n"
            "Allowed feature_family: descriptor, pattern_presence, count.\n"
            "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
            "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
            "Never copy prior draft sentences into source, primary_source, or provenance.\n"
            "If a field is unclear, use null instead of prose.\n"
            "Return 1 to 3 rules only.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            f"{schema_text}\n\n"
            "Output exactly this shape and stop:\n"
            "{\"rules\":[{\"source\":\""
            f"{source_value}"
            "\",\"feature_family\":\"descriptor\",\"feature_name\":\"MolWt\",\"feature_mode\":\"numeric_value\",\"relation_to_target\":\"monotonic_up\",\"threshold\":null,\"rationale\":\"short reason\"}]}\n"
        )
    if profile == "llama_native":
        source_value = "literature" if source == "literature" else "data"
        return (
            "Convert the draft into strict JSON rules for native Llama projection.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Do not output markdown, code fences, END_JSON, explanations, notes, prefixes, or suffixes.\n"
            "Do not stop until the final two characters are ]}.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            "Only return 2 rules when the draft clearly supports 2 clean rules; otherwise return 1 rule or {\"rules\":[]}.\n"
            "Prefer 2 different feature families when 2 clean rules are supported.\n"
            "Prefer 1 descriptor rule and 1 non-descriptor rule when the draft supports both.\n"
            "Do not return two descriptor rules if a clean pattern_presence or count rule is supported by the draft.\n"
            "Each rationale must be at most 8 words and plain.\n"
            "If no clean rule is supported, return exactly {\"rules\":[]}.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            f"{schema_text}\n\n"
            "Output exactly this shape and stop immediately after the closing brace:\n"
            "{\"rules\":[{\"source\":\""
            f"{source_value}"
            "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "chat_native_core":
        source_value = "literature" if source == "literature" else "data"
        policy = chat_native_core_termination_policy()
        if policy == "pure_json_single_rule":
            return (
                "Convert the draft into strict JSON rules for generic native chat-model ablation.\n"
                "Termination policy: pure_json_single_rule.\n"
                "Return exactly one JSON object only.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "Do not output markdown, code fences, explanations, notes, prefixes, suffixes, or stop markers.\n"
                f"Use source exactly \"{source_value}\" for every rule.\n"
                "If a clean rule is supported, return exactly 1 rule.\n"
                "If no clean rule is supported, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Prefer rules that are easy to compile and semantically clean.\n"
                "Allowed feature_family: descriptor, pattern_presence, count.\n"
                "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
                "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
                "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
                "If a rule needs an unsupported field or value, drop that rule instead of inventing a synonym or alias.\n"
                "Each rationale must be 1 to 3 words.\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                f"{schema_text}\n\n"
                "Output exactly this shape and stop immediately after the closing brace:\n"
                "{\"rules\":[{\"source\":\""
                f"{source_value}"
                "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        if policy == "pure_json_single_vocab_rescue":
            return (
                "Convert the draft into strict JSON rules for generic native chat-model ablation.\n"
                "Termination policy: pure_json_single_vocab_rescue.\n"
                "Return exactly one JSON object only.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "Do not output markdown, code fences, explanations, notes, prefixes, suffixes, or stop markers.\n"
                f"Use source exactly \"{source_value}\" for every rule.\n"
                "If a clean rule is supported, return exactly 1 rule.\n"
                "If no clean rule is supported, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Prefer the shortest compile-safe rule that stays inside the supported vocabulary.\n"
                f"{_chat_native_core_vocab_rescue_constraints()}\n"
                "Each rationale must be 1 to 3 words and a short noun phrase, not a sentence.\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                f"{schema_text}\n\n"
                "Output exactly this shape and stop immediately after the closing brace:\n"
                "{\"rules\":[{\"source\":\""
                f"{source_value}"
                "\",\"feature_family\":\"<family>\",\"feature_name\":\"<allowed_feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation_or_null>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        if policy == "pure_json_short_output":
            return (
                "Convert the draft into strict JSON rules for generic native chat-model ablation.\n"
                "Termination policy: pure_json_short_output.\n"
                "Return exactly one JSON object only.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "Do not output markdown, code fences, explanations, notes, prefixes, suffixes, or stop markers.\n"
                f"Use source exactly \"{source_value}\" for every rule.\n"
                "Valid single rule is better than two risky rules.\n"
                "If returning 2 rules may make the JSON longer or less certain, return 1 rule.\n"
                "Return 1 clean rule by default.\n"
                "Return 2 rules only if both are short, valid, and from different feature families.\n"
                "Never return more than 2 rules.\n"
                "If no clean rule is supported, return {\"rules\":[]}.\n"
                "Prefer rules that are easy to compile and semantically clean.\n"
                "Allowed feature_family: descriptor, pattern_presence, count.\n"
                "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
                "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
                "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
                "If a rule needs an unsupported field or value, drop that rule instead of inventing a synonym or alias.\n"
                "Each rationale must be 1 to 3 words and a short phrase, not a sentence.\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                f"{schema_text}\n\n"
                "Output exactly this shape and stop immediately after the closing brace:\n"
                "{\"rules\":[{\"source\":\""
                f"{source_value}"
                "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        return (
            "Convert the draft into strict JSON rules for generic native chat-model ablation.\n"
            "Termination policy: end_json_marker_single_rule.\n"
            "Return exactly one JSON object followed by END_JSON.\n"
            "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
            "Output must start with {.\n"
            "Output must end with }.\n"
            "Do not write anything before {.\n"
            "After the final }, write END_JSON and stop.\n"
            "Do not write anything after END_JSON.\n"
            "Do not output markdown, code fences, explanations, notes, prefixes, or suffixes.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            "If a clean rule is supported, return exactly 1 rule.\n"
            "If no clean rule is supported, return {\"rules\":[]} followed by END_JSON.\n"
            "Do not return 2 rules.\n"
            "Prefer rules that are easy to compile and semantically clean.\n"
            "Allowed feature_family: descriptor, pattern_presence, count.\n"
            "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
            "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
            "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
            "If a rule needs an unsupported field or value, drop that rule instead of inventing a synonym or alias.\n"
            "Each rationale must be 1 to 3 words.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            f"{schema_text}\n\n"
            "Output exactly this shape, then END_JSON:\n"
            "{\"rules\":[{\"source\":\""
                f"{source_value}"
                "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}END_JSON"
        )
    if profile == "gemma_native":
        source_value = "literature" if source == "literature" else "data"
        return (
            "Convert the draft into compact JSON rules for native Gemma projection.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "No markdown. No prose. No prefixes. No suffixes.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            "Return 1 or 2 clean rules when clearly supported.\n"
            "Do not output an empty rule list unless the draft clearly supports no valid rule.\n"
            "Prefer rules that are easy to compile and distinct in feature_family.\n"
            "Prefer 2 different feature families when 2 clean rules are supported.\n"
            "Keep rationale short and plain.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            f"{schema_text}\n\n"
            "Output exactly this shape and stop immediately after the closing brace:\n"
            "{\"rules\":[{\"source\":\""
            f"{source_value}"
            "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "chemdfm_native":
        source_value = "literature" if source == "literature" else "data"
        return (
            "Convert the draft into strict JSON rules for native ChemDFM projection.\n"
            "Return exactly one JSON object and nothing else.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Output must start with {.\n"
            "Output must end with }.\n"
            "Do not write anything before {.\n"
            "Do not write anything after }.\n"
            "Stop immediately after the final }.\n"
            "Forbidden prefixes/labels: Output:, Result:, JSON:, Here is, Here's, repaired JSON, markdown headings, code fences, ENDJSON, END_JSON.\n"
            "Do not output explanations, notes, prefixes, suffixes, headings, code fences, ENDJSON, or END_JSON.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            f"{_chemdfm_native_closure_baseline_constraints()}\n"
            "Do not force diversity or a second family.\n"
            "Keep the output short, compact, and semantically clean.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            f"{schema_text}\n\n"
            "Output exactly this shape and stop immediately after the closing brace:\n"
            "{\"rules\":[{\"source\":\""
            f"{source_value}"
            "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "non_qwen_native" or is_non_qwen_family(projector_family):
        source_value = "literature" if source == "literature" else "data"
        return (
            "Convert the draft into compact JSON rules.\n"
            "Return JSON only. No markdown. No prose. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            f"Use source exactly \"{source_value}\" for every rule.\n"
            "Return 1 to 2 rules only.\n"
            "Each rationale must be at most 12 words.\n"
            "Use threshold null unless numeric cutoff is explicit.\n"
            "Allowed feature_family: descriptor, pattern_presence, count.\n"
            "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
            "Allowed descriptor feature_name examples: MolWt, MolLogP, TPSA, NumHBD, NumHBA, NumRotatableBonds, NumAromaticRings, RingCount.\n"
            "Allowed pattern_presence feature_name examples: amide, amine, halogen, aromatic_ring, heteroaromatic_ring, fused_ring, nitro_group, ether, ester, carboxyl, carbonyl, sulfonamide.\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Output exactly this shape and stop:\n"
            "{\"rules\":[{\"source\":\""
            f"{source_value}"
            "\",\"feature_family\":\"descriptor\",\"feature_name\":\"MolWt\",\"feature_mode\":\"numeric_value\",\"relation_to_target\":\"monotonic_up\",\"threshold\":null,\"rationale\":\"short reason\"}]}\n"
        )
    if source == "literature":
        return (
            f"Task brief:\n{task_brief.strip()}\n\n"
            "Prior chemistry draft:\n"
            f"{draft_text.strip()}\n\n"
            "Project this literature draft into 0 to 3 clean rule objects only.\n"
            "Return exactly one JSON object with a top-level rules array.\n"
            "Do not output markdown fences, explanations, commentary, or extra text.\n"
            "Each rule must belong to exactly one feature family.\n"
            "If uncertain, return {\"rules\": []}.\n\n"
            f"{schema_text}\n\n"
            f"End with {END_JSON_MARKER}."
        )
    return (
        f"Task brief:\n{task_brief.strip()}\n\n"
        f"Rule source: {source}\n\n"
        "Prior chemistry draft:\n"
        f"{draft_text.strip()}\n\n"
        "Project only the determinants that are explicitly supported by the draft above. "
        "Do not rewrite the draft into a new discovery. If a field is unclear, use null.\n\n"
        f"{schema_text}\n\n"
        f"Output exactly one JSON object with a top-level 'rules' array. "
        f"Return the supported subset only. No markdown, no explanations. End with {END_JSON_MARKER}."
    )


def build_stage2_repair_prompt(raw_output: str, source: str, projector_family: str = "generic") -> str:
    schema_text = build_rule_schema_text(short_form=True)
    profile = str(
        science_env_get("_MORACE_STAGE2_PROTOCOL_PROFILE")
        or science_env_get("_LLM4SD_STAGE2_PROTOCOL_PROFILE", "")
        or ""
    ).strip()
    if profile == "cross_qwen_for_non_qwen_draft":
        source_value = "literature" if source == "literature" else "data"
        if source == "literature":
            return (
                "Repair the malformed literature projection into one strict JSON object only.\n"
                "Output must start with { and end with }.\n"
                "No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.\n"
                "Top-level object must be exactly {\"rules\":[...]}.\n"
                "Delete every non-JSON fragment, including explanations, notes, trailing text, and repeated draft text.\n"
                f"Use source exactly \"{source_value}\" for every kept rule.\n"
                "Allowed fields: source, feature_family, feature_name, feature_mode, relation_to_target, threshold, rationale.\n"
                "Normalize polluted source text back to the allowed source value.\n"
                "Drop incomplete or semantically invalid rules.\n"
                "If no clean rule survives, return exactly {\"rules\":[]}.\n\n"
                "Invalid output:\n"
                f"{raw_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Return only repaired JSON."
            )
        return (
            "Repair the malformed output into one strict JSON object only.\n"
            "No markdown. No prose. No extra text before or after JSON.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            f"Use source exactly \"{source_value}\".\n"
            "Allowed fields: source, feature_family, feature_name, feature_mode, relation_to_target, threshold, rationale.\n"
            "Normalize polluted source text back to the allowed source value.\n"
            "Drop incomplete or semantically invalid rules.\n\n"
            "Invalid output:\n"
            f"{raw_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Return only repaired JSON."
        )
    if profile == "llama_native":
        source_value = "literature" if source == "literature" else "data"
        return (
            "Repair the malformed native Llama projection into one strict JSON object only.\n"
            "Output must start with { and end with }.\n"
            "No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Delete every non-JSON fragment, including assistant wrappers and trailing notes.\n"
            f"Use source exactly \"{source_value}\" for every kept rule.\n"
            "Drop incomplete or semantically invalid rules.\n"
            "If no clean rule survives, return exactly {\"rules\":[]}.\n\n"
            "Invalid output:\n"
            f"{raw_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Return only repaired JSON."
        )
    if profile == "chemdfm_native":
        source_value = "literature" if source == "literature" else "data"
        return (
            "Repair the malformed native ChemDFM projection into one strict JSON object only.\n"
            "Output must start with { and end with }.\n"
            "Do not write anything before { or after }.\n"
            "Forbidden prefixes/labels: Output:, Result:, JSON:, Here is, Here's, repaired JSON, markdown headings, code fences, ENDJSON, END_JSON.\n"
            "Repair the existing JSON shape only. Do not add a new rule. Do not add a second rule.\n"
            "If input contains multiple rules, keep only the first clean canonical rule and drop the rest.\n"
            "Do not guess feature_name or chemistry terms. If feature_name is unsupported, drop the rule.\n"
            "Use only rules already clearly present in the invalid output.\n"
            "Drop incomplete, unsupported, or semantically invalid rules.\n"
            "If repair is uncertain or no clean rule survives, return exactly {\"rules\":[]}.\n"
            f"Use source exactly \"{source_value}\" for every kept rule.\n"
            "Allowed fields: source, feature_family, feature_name, feature_mode, relation_to_target, threshold, rationale.\n"
            f"{_chemdfm_native_closure_baseline_constraints()}\n"
            "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n\n"
            "Invalid output:\n"
            f"{raw_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Return only repaired JSON."
        )
    if profile == "chat_native_core":
        source_value = "literature" if source == "literature" else "data"
        policy = chat_native_core_termination_policy()
        if policy == "pure_json_single_rule":
            return (
                "Repair the malformed generic chat-model projection into one strict JSON object only.\n"
                "Termination policy: pure_json_single_rule.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                f"Use source exactly \"{source_value}\" for every kept rule.\n"
                "Allowed feature_family: descriptor, pattern_presence, count.\n"
                "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
                "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
                "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
                "Drop incomplete or semantically invalid rules.\n"
                "If a clean rule survives, return exactly 1 rule.\n"
                "If no clean rule survives, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Each rationale must be 1 to 3 words.\n\n"
                "Invalid output:\n"
                f"{raw_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Return only repaired JSON."
            )
        if policy == "pure_json_single_vocab_rescue":
            return (
                "Repair the malformed generic chat-model projection into one strict JSON object only.\n"
                "Termination policy: pure_json_single_vocab_rescue.\n"
                "Repair the existing JSON shape, enum values, and vocabulary only.\n"
                "Do not add a new rule during repair.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                f"Use source exactly \"{source_value}\" for every kept rule.\n"
                f"{_chat_native_core_vocab_rescue_constraints()}\n"
                "If feature_name is unsupported, drop the rule.\n"
                "Do not replace unsupported feature_name with a guessed chemistry term.\n"
                "If a clean rule survives, return exactly 1 rule.\n"
                "If no clean rule survives, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Each rationale must be 1 to 3 words and a short noun phrase, not a sentence.\n\n"
                "Invalid output:\n"
                f"{raw_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Return only repaired JSON."
            )
        if policy == "pure_json_short_output":
            return (
                "Repair the malformed generic chat-model projection into one strict JSON object only.\n"
                "Termination policy: pure_json_short_output.\n"
                "Repair the existing JSON shape only. Do not add a second rule unless it was already clearly present.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                f"Use source exactly \"{source_value}\" for every kept rule.\n"
                "Allowed feature_family: descriptor, pattern_presence, count.\n"
                "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
                "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
                "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
                "Drop incomplete or semantically invalid rules.\n"
                "Valid single rule is better than two risky rules.\n"
                "If returning 2 rules may make the JSON longer or less certain, return 1 rule.\n"
                "Return 1 clean rule by default.\n"
                "Return 2 rules only if both were already clearly present, short, valid, and from different feature families.\n"
                "Never return more than 2 rules.\n"
                "If no clean rule survives, return {\"rules\":[]}.\n"
                "Each rationale must be 1 to 3 words and a short phrase, not a sentence.\n\n"
                "Invalid output:\n"
                f"{raw_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Return only repaired JSON."
            )
        return (
            "Repair the malformed generic chat-model projection into one strict JSON object only.\n"
            "Termination policy: end_json_marker_single_rule.\n"
            "Output must start with {.\n"
            "Output must end with }.\n"
            "Do not write anything before {.\n"
            "After the final }, write END_JSON and stop.\n"
            "Do not write anything after END_JSON.\n"
            "No markdown. No prose. No code fence. No prefix. No suffix.\n"
            "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
            f"Use source exactly \"{source_value}\" for every kept rule.\n"
            "Allowed feature_family: descriptor, pattern_presence, count.\n"
            "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
            "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
            "Compact feature_name examples: MolLogP, aromatic_ring, ring_count.\n"
            "Drop incomplete or semantically invalid rules.\n"
            "If a clean rule survives, return exactly 1 rule.\n"
            "If no clean rule survives, return {\"rules\":[]} followed by END_JSON.\n"
            "Do not return 2 rules.\n"
            "Each rationale must be 1 to 3 words.\n\n"
            "Invalid output:\n"
            f"{raw_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Return only repaired JSON followed by END_JSON."
        )
    if profile == "non_qwen_native" or is_non_qwen_family(projector_family):
        source_value = "literature" if source == "literature" else "data"
        return (
            "Repair the text into one compact JSON object only.\n"
            "No markdown. No prose. No END marker inside JSON.\n"
            "Keep at most 2 complete rules. Drop incomplete rules.\n"
            f"Use source exactly \"{source_value}\".\n"
            "Each rule fields: source, feature_family, feature_name, feature_mode, relation_to_target, threshold, rationale.\n"
            "Set threshold to null unless it is an object with op and value.\n\n"
            "Invalid output:\n"
            f"{raw_output.strip()}\n\n"
            "Return only {\"rules\":[...]}."
        )
    return (
        f"Rule source: {source}\n\n"
        "The previous projector output was invalid or nested incorrectly.\n"
        "Repair it so that:\n"
        "- the top-level object is exactly {\"rules\": [...]}.\n"
        "- each rule is a flat object and must not contain a nested 'rules' field.\n"
        "- unsupported details become null instead of prose.\n"
        f"- end with {END_JSON_MARKER}.\n\n"
        f"{schema_text}\n\n"
        "Invalid output to repair:\n"
        f"{raw_output.strip()}\n\n"
        f"Return only repaired JSON, then {END_JSON_MARKER}."
    )


def build_stage2_literature_retry_prompt(
    task_brief: str,
    draft_text: str,
    previous_output: str,
    current_roots: Sequence[str],
    current_families: Sequence[str],
    projector_family: str = "generic",
    stage2_protocol_profile: str = "",
) -> str:
    schema_text = build_rule_schema_text(short_form=True)
    root_text = ", ".join(str(root) for root in current_roots) if current_roots else "none"
    family_text = ", ".join(str(family) for family in current_families) if current_families else "none"
    profile = str(stage2_protocol_profile or "").strip()
    if profile == "cross_qwen_for_non_qwen_draft":
        return (
            "Return one strict JSON object only.\n"
            "Output must start with { and end with }.\n"
            "No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "The prior draft is non-Qwen and may be noisy; preserve only explicit determinants.\n"
            "Add at most 1 new literature rule that is not already covered.\n"
            "Do not copy prior draft text into source or provenance fields.\n"
            "Delete any explanation-like fragment instead of keeping it near the JSON.\n"
            "If no clean new rule is supported, return exactly {\"rules\":[]}.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Previous malformed output to improve:\n"
            f"{previous_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "llama_native":
        return (
            "Return one strict JSON object only for native Llama literature retry.\n"
            "Output must start with { and end with }.\n"
            "No markdown. No prose. No code fence. No END_JSON. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Do not stop until the final two characters are ]}.\n"
            "Add at most 1 new literature rule that is not already covered.\n"
            "Prefer a new feature family not already covered.\n"
            "Prefer pattern_presence or count when the current families are descriptor-only.\n"
            "Each rationale must be at most 8 words and plain.\n"
            "Delete assistant wrappers or explanation-like fragments instead of preserving them.\n"
            "If no clean new rule is supported, return exactly {\"rules\":[]}.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Previous malformed output to improve:\n"
            f"{previous_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "gemma_native":
        return (
            "Return one compact JSON object only for native Gemma literature retry.\n"
            "No markdown. No prose. No code fence. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Add at most 1 new literature rule that is not already covered.\n"
            "Prefer a new feature family not already covered.\n"
            "Do not return an empty rule list unless the draft clearly supports no valid new rule.\n"
            "Keep rationale short and plain.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Previous malformed output to improve:\n"
            f"{previous_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "chemdfm_native":
        return (
            "Return one strict JSON object only for native ChemDFM literature retry.\n"
            "Output must start with {.\n"
            "Output must end with }.\n"
            "Do not write anything before {.\n"
            "Do not write anything after }.\n"
            "Stop immediately after the final }.\n"
            "Forbidden prefixes/labels: Output:, Result:, JSON:, Here is, Here's, repaired JSON, markdown headings, code fences, ENDJSON, END_JSON.\n"
            "No markdown. No prose. No code fence. No ENDJSON. No END_JSON. No prefix. No suffix.\n"
            "Top-level object must be exactly {\"rules\":[...]}.\n"
            "Prefer 1 short clean literature rule.\n"
            "Never force a second rule.\n"
            "A single closed rule is better than multiple incomplete rules.\n"
            "If no clean new rule is supported, return exactly {\"rules\":[]}.\n"
            "Keep rationale short and plain.\n"
            f"{_chemdfm_native_closure_baseline_constraints()}\n"
            "Keep feature_family, feature_mode, relation_to_target, and feature_name within the supported schema vocabulary only.\n"
            "Do not rescue unsupported field values by inventing aliases or synonyms.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Previous malformed output to improve:\n"
            f"{previous_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    if profile == "chat_native_core":
        policy = chat_native_core_termination_policy()
        if policy == "pure_json_single_rule":
            return (
                "Return one strict JSON object only for generic native chat-model literature retry.\n"
                "Termination policy: pure_json_single_rule.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "If a clean new literature rule is supported, return exactly 1 rule.\n"
                "If no clean new rule is supported, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Prefer the shortest valid rule that stays inside the supported schema vocabulary.\n"
                "Do not invent aliases or unsupported field values.\n"
                "Each rationale must be 1 to 3 words.\n"
                f"Already covered roots: {root_text}\n"
                f"Already covered families: {family_text}\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                "Previous malformed output to improve:\n"
                f"{previous_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        if policy == "pure_json_single_vocab_rescue":
            return (
                "Return one strict JSON object only for generic native chat-model literature retry.\n"
                "Termination policy: pure_json_single_vocab_rescue.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "If a clean new literature rule is supported, return exactly 1 rule.\n"
                "If no clean new rule is supported, return {\"rules\":[]}.\n"
                "Do not return 2 rules.\n"
                "Prefer the shortest valid rule that stays inside the supported vocabulary.\n"
                f"{_chat_native_core_vocab_rescue_constraints()}\n"
                "Each rationale must be 1 to 3 words and a short noun phrase, not a sentence.\n"
                f"Already covered roots: {root_text}\n"
                f"Already covered families: {family_text}\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                "Previous malformed output to improve:\n"
                f"{previous_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<allowed_feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation_or_null>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        if policy == "pure_json_short_output":
            return (
                "Return one strict JSON object only for generic native chat-model literature retry.\n"
                "Termination policy: pure_json_short_output.\n"
                "Output must start with {.\n"
                "Output must end with }.\n"
                "Do not write anything before {.\n"
                "Do not write anything after the final }.\n"
                "No markdown. No prose. No code fence. No prefix. No suffix. No stop marker.\n"
                "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
                "Valid single rule is better than two risky rules.\n"
                "If returning 2 rules may make the JSON longer or less certain, return 1 rule.\n"
                "Return 1 clean new literature rule by default.\n"
                "Return 2 rules only if both are short, valid, and from different feature families.\n"
                "Never return more than 2 rules.\n"
                "If no clean new rule is supported, return {\"rules\":[]}.\n"
                "Prefer the shortest valid rule that stays inside the supported schema vocabulary.\n"
                "Do not invent aliases or unsupported field values.\n"
                "Each rationale must be 1 to 3 words and a short phrase, not a sentence.\n"
                f"Already covered roots: {root_text}\n"
                f"Already covered families: {family_text}\n\n"
                f"Task:\n{task_brief.strip()}\n\n"
                f"Draft:\n{draft_text.strip()}\n\n"
                "Previous malformed output to improve:\n"
                f"{previous_output.strip()}\n\n"
                f"{schema_text}\n\n"
                "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
            )
        return (
            "Return one strict JSON object only for generic native chat-model literature retry.\n"
            "Termination policy: end_json_marker_single_rule.\n"
            "Output must start with {.\n"
            "Output must end with }.\n"
            "Do not write anything before {.\n"
            "After the final }, write END_JSON and stop.\n"
            "Do not write anything after END_JSON.\n"
            "No markdown. No prose. No code fence. No prefix. No suffix.\n"
            "Top-level object must be a JSON object with exactly one key: \"rules\", whose value is an array.\n"
            "If a clean new literature rule is supported, return exactly 1 rule.\n"
            "If no clean new rule is supported, return {\"rules\":[]} followed by END_JSON.\n"
            "Do not return 2 rules.\n"
            "Prefer the shortest valid rule that stays inside the supported schema vocabulary.\n"
            "Do not invent aliases or unsupported field values.\n"
            "Each rationale must be 1 to 3 words.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Previous malformed output to improve:\n"
            f"{previous_output.strip()}\n\n"
            f"{schema_text}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}END_JSON"
        )
    if profile == "non_qwen_native" or is_non_qwen_family(projector_family):
        return (
            "Return one compact JSON object only. No markdown or prose.\n"
            "Add at most 1 new literature rule that is not already covered.\n"
            f"Already covered roots: {root_text}\n"
            f"Already covered families: {family_text}\n\n"
            f"Task:\n{task_brief.strip()}\n\n"
            f"Draft:\n{draft_text.strip()}\n\n"
            "Shape: {\"rules\":[{\"source\":\"literature\",\"feature_family\":\"pattern_presence\",\"feature_name\":\"amine\",\"feature_mode\":\"binary_presence\",\"relation_to_target\":\"positive\",\"threshold\":null,\"rationale\":\"short reason\"}]}"
        )
    return (
        f"Task brief:\n{task_brief.strip()}\n\n"
        "Prior chemistry draft:\n"
        f"{draft_text.strip()}\n\n"
        "Previous projection underfit the literature draft.\n"
        "Retry once and preserve multiple determinants when the draft supports them.\n"
        "Do not repeat the same determinant root unless the draft clearly supports a complementary second rule.\n"
        "Prefer a literature-supported determinant that adds a different root and, if possible, a different family.\n"
        "Prioritize complementary classic determinants such as hydrogen-bond donors/acceptors, rotatable bonds, TPSA, "
        "molecular weight, or lipophilicity only when the draft explicitly supports them.\n"
        "Return only 1 to 2 candidate rules on this retry.\n\n"
        f"Already covered roots:\n{root_text}\n\n"
        f"Already covered families:\n{family_text}\n\n"
        "Return exactly one JSON object with a top-level rules array and nothing else.\n"
        "If the draft only supports one rule, return that one rule rather than prose.\n\n"
        f"{schema_text}\n\n"
        "Previous projection to improve:\n"
        f"{previous_output.strip()}\n\n"
        f"End with {END_JSON_MARKER}."
    )


def build_stage2_independent_single_rule_prompt(
    task_brief: str,
    draft_text: str,
    previous_roots: Sequence[str],
    previous_families: Sequence[str],
    source: str = "literature",
) -> str:
    schema_text = build_rule_schema_text(short_form=True)
    root_text = ", ".join(str(root) for root in previous_roots) if previous_roots else "none"
    family_text = ", ".join(str(family) for family in previous_families) if previous_families else "none"
    source_value = "literature" if source == "literature" else "data"
    return (
        "Return exactly one JSON object only.\n"
        "Projection variant role: independent_single_rule_2.\n"
        "Return exactly 1 clean JSON rule if the draft supports a clean rule not using the previous root.\n"
        "If no clean novel rule is supported, return {\"rules\":[]}.\n"
        "Do not return 2 rules.\n"
        "Do not use stop markers, markdown, prose, prefixes, suffixes, or explanations.\n"
        "Prefer a different root. Prefer a different feature_family only when clearly supported.\n"
        "Validity is more important than diversity.\n"
        "Output must start with { and end with }.\n"
        f"Use source exactly \"{source_value}\" for every rule.\n"
        "Allowed feature_family: descriptor, pattern_presence, count.\n"
        "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n"
        "Allowed feature_mode: numeric_value, threshold_binary, binary_presence, count_value.\n"
        "Each rationale must be 1 to 3 words.\n"
        f"Previous roots: {root_text}\n"
        f"Previous families: {family_text}\n\n"
        f"Task:\n{task_brief.strip()}\n\n"
        f"Draft:\n{draft_text.strip()}\n\n"
        f"{schema_text}\n\n"
        "Shape: {\"rules\":[{\"source\":\""
        f"{source_value}"
        "\",\"feature_family\":\"<family>\",\"feature_name\":\"<feature>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
    )


def build_stage2_family_target_single_rule_prompt(
    task_brief: str,
    draft_text: str,
    target_family: str,
    source: str = "literature",
    target_vocab_policy: str = DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY,
    target_vocab_options: Sequence[str] | None = None,
) -> str:
    target_family = str(target_family or "").strip()
    if target_family not in CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS:
        raise ValueError(f"Unsupported family target: {target_family!r}")
    target_vocab_policy = str(target_vocab_policy or DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY).strip()
    target_vocab_options = list(target_vocab_options or [])
    if target_vocab_policy == "micro_canonical_options" and not target_vocab_options:
        raise ValueError(f"Empty target micro-vocabulary for target family {target_family!r}.")
    mode_hint = {
        "pattern_presence": "Use feature_mode exactly binary_presence.",
        "count": "Use feature_mode exactly count_value.",
        "descriptor": "Use feature_mode numeric_value or threshold_binary.",
    }[target_family]
    vocab_hint = ""
    if target_vocab_policy == "micro_canonical_options":
        forbidden_aliases = ", ".join(CHAT_NATIVE_CORE_TARGET_FORBIDDEN_ALIASES.get(target_family, ()))
        vocab_hint = (
            "feature_name must be exactly one of: "
            + ", ".join(target_vocab_options)
            + ".\n"
            "Do not invent feature_name.\n"
            "If none of these options is supported, return {\"rules\":[]}.\n"
            + (f"Do not output these aliases: {forbidden_aliases}.\n" if forbidden_aliases else "")
        )
    source_value = "literature" if source == "literature" else "data"
    return (
        "Return exactly one JSON object only.\n"
        f"Projection variant role: family_target_{target_family}.\n"
        "Return at most 1 rule.\n"
        f"The rule's feature_family must be exactly {target_family}.\n"
        "Do not output any other feature_family.\n"
        f"If the draft does not clearly support one clean {target_family} rule, return {{\"rules\":[]}}.\n"
        "If only another family is supported, return {\"rules\":[]} instead.\n"
        f"{mode_hint}\n"
        f"{vocab_hint}"
        "No markdown. No prose. No prefix. No suffix. No stop marker.\n"
        "Rationale must be 1-3 words.\n"
        "Use only canonical names already supported by the schema. If uncertain, return {\"rules\":[]}.\n"
        f"Use source exactly \"{source_value}\" for every rule.\n"
        "Allowed relation_to_target: positive, negative, monotonic_up, monotonic_down, null.\n\n"
        f"Task:\n{task_brief.strip()}\n\n"
        f"Draft:\n{draft_text.strip()}\n\n"
        "Shape: {\"rules\":[{\"source\":\""
        f"{source_value}"
        "\",\"feature_family\":\""
        f"{target_family}"
        "\",\"feature_name\":\"<canonical>\",\"feature_mode\":\"<mode>\",\"relation_to_target\":\"<relation>\",\"threshold\":null,\"rationale\":\"short\"}]}"
    )
