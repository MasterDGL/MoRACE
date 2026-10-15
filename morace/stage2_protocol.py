"""Stage-2 protocol configuration and environment policy resolution."""

from morace.generation_runtime import science_env_get
from typing import List

from morace.feature_vocabulary import d4_feature_name_allowlist


STAGE2_PROTOCOL_PROFILE_VERSIONS = {
    "qwen_native": "qwen_native_v1",
    "qwen_legacy_d2": "qwen_legacy_d2_v1",
    "non_qwen_native": "non_qwen_native_v1",
    "llama_native": "llama_native_v1",
    "gemma_native": "gemma_native_v1",
    "chemdfm_native": "chemdfm_native_v2_9_b3_family_target_thinning_v1",
    "chat_native_core": "chat_native_core_d5_2_target_thinning_v1",
    "cross_qwen_for_non_qwen_draft": "cross_qwen_for_non_qwen_draft_v1",
}
ALLOWED_STAGE2_PROTOCOL_PROFILES = set(STAGE2_PROTOCOL_PROFILE_VERSIONS.keys())
CHAT_NATIVE_CORE_TERMINATION_POLICIES = {
    "pure_json_single_rule",
    "pure_json_single_vocab_rescue",
    "pure_json_short_output",
    "end_json_marker_single_rule",
}
DEFAULT_CHAT_NATIVE_CORE_TERMINATION_POLICY = "pure_json_single_rule"
CHAT_NATIVE_CORE_RULE_COUNT_POLICIES = {
    "single_rule",
    "two_independent_single_rule_projections",
}
DEFAULT_CHAT_NATIVE_CORE_RULE_COUNT_POLICY = "single_rule"
CHAT_NATIVE_CORE_EXTRACTION_POLICIES = {
    "strict_only",
    "tolerant_first_json_object",
}
DEFAULT_CHAT_NATIVE_CORE_EXTRACTION_POLICY = "strict_only"
CHAT_NATIVE_CORE_CANONICALIZATION_POLICIES = {
    "strict_canonical_only",
    "safe_enum_feature_canonicalization",
}
DEFAULT_CHAT_NATIVE_CORE_CANONICALIZATION_POLICY = "strict_canonical_only"
CHAT_NATIVE_CORE_FAMILY_TARGET_POLICIES = {
    "none",
    "family_targeted_single_rule_variants",
}
DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY = "none"
CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS = {
    "descriptor",
    "pattern_presence",
    "count",
}
DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGETS = ("pattern_presence", "count")
CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES = {
    "none",
    "micro_canonical_options",
}
DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY = "none"
CHAT_NATIVE_CORE_TARGET_MICRO_VOCAB_PREFERRED = {
    "pattern_presence": ("aromatic_ring", "amide", "amine", "halogen", "carbonyl", "carboxyl"),
    "count": ("NumHBA", "NumHBD", "NumRotatableBonds", "RingCount", "NumAromaticRings"),
}
CHAT_NATIVE_CORE_TARGET_FORBIDDEN_ALIASES = {
    "pattern_presence": ("aromaticity", "aromatic_rings", "benzene_system", "phenyl_group", "polar_groups"),
    "count": ("polar_groups", "hbond_groups", "aromatic_rings", "number", "present"),
}
CHAT_NATIVE_CORE_TARGET_THINNING_POLICIES = {
    "none",
    "first_valid_target_rule",
}
DEFAULT_CHAT_NATIVE_CORE_TARGET_THINNING_POLICY = "none"
ALLOWED_STAGE2_EXTRACTION_POLICIES = CHAT_NATIVE_CORE_EXTRACTION_POLICIES
ALLOWED_STAGE2_CANONICALIZATION_POLICIES = CHAT_NATIVE_CORE_CANONICALIZATION_POLICIES
ALLOWED_STAGE2_FAMILY_FEATURE_POLICIES = {"strict_family", "safe_family_feature_consistency"}
ALLOWED_STAGE2_FAMILY_TARGET_POLICIES = CHAT_NATIVE_CORE_FAMILY_TARGET_POLICIES
DEFAULT_STAGE2_FAMILY_TARGET_POLICY = DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY
ALLOWED_STAGE2_FAMILY_TARGETS = CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS
DEFAULT_STAGE2_FAMILY_TARGETS = DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGETS
ALLOWED_STAGE2_TARGET_VOCAB_POLICIES = CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES
DEFAULT_STAGE2_TARGET_VOCAB_POLICY = DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY
STAGE2_TARGET_MICRO_VOCAB_PREFERRED = CHAT_NATIVE_CORE_TARGET_MICRO_VOCAB_PREFERRED
STAGE2_TARGET_FORBIDDEN_ALIASES = CHAT_NATIVE_CORE_TARGET_FORBIDDEN_ALIASES
ALLOWED_STAGE2_TARGET_THINNING_POLICIES = CHAT_NATIVE_CORE_TARGET_THINNING_POLICIES
DEFAULT_STAGE2_TARGET_THINNING_POLICY = DEFAULT_CHAT_NATIVE_CORE_TARGET_THINNING_POLICY


def infer_stage2_protocol_profile(
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
) -> tuple[str, str]:
    override = str(science_env_get("STAGE2_PROTOCOL_PROFILE", "") or "").strip()
    if override:
        if override not in ALLOWED_STAGE2_PROTOCOL_PROFILES:
            raise ValueError(
                f"Unsupported STAGE2_PROTOCOL_PROFILE={override!r}; allowed="
                f"{sorted(ALLOWED_STAGE2_PROTOCOL_PROFILES)}"
            )
        return override, "env_override"
    draft_family = str(draft_family or "generic")
    projector_family = str(projector_family or "generic")
    if projector_family == "qwen" and draft_family == "qwen":
        profile = "qwen_legacy_d2"
    elif projector_family == "qwen" and (use_cross_model_projector or draft_family != "qwen"):
        profile = "cross_qwen_for_non_qwen_draft"
    elif projector_family == "llama" and draft_family == "llama" and not use_cross_model_projector:
        profile = "llama_native"
    elif projector_family == "gemma" and draft_family == "gemma" and not use_cross_model_projector:
        profile = "gemma_native"
    elif projector_family == "chemdfm" and draft_family == "chemdfm" and not use_cross_model_projector:
        profile = "chemdfm_native"
    else:
        profile = "non_qwen_native"
    if profile not in ALLOWED_STAGE2_PROTOCOL_PROFILES:
        raise ValueError(f"Auto-inferred unsupported stage2 protocol profile: {profile!r}")
    return profile, "auto_inferred"


def stage2_protocol_profile_version(stage2_protocol_profile: str) -> str:
    normalized = str(stage2_protocol_profile or "").strip()
    if normalized not in STAGE2_PROTOCOL_PROFILE_VERSIONS:
        raise ValueError(
            f"Unknown stage2_protocol_profile={normalized!r}; allowed="
            f"{sorted(STAGE2_PROTOCOL_PROFILE_VERSIONS)}"
        )
    return STAGE2_PROTOCOL_PROFILE_VERSIONS[normalized]


def chat_native_core_extraction_policy() -> str:
    policy = str(
        science_env_get("STAGE2_EXTRACTION_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_EXTRACTION_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_EXTRACTION_POLICY
    ).strip()
    if policy not in CHAT_NATIVE_CORE_EXTRACTION_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_EXTRACTION_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_EXTRACTION_POLICIES)}"
        )
    return policy


def chat_native_core_canonicalization_policy() -> str:
    policy = str(
        science_env_get("STAGE2_CANONICALIZATION_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_CANONICALIZATION_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_CANONICALIZATION_POLICY
    ).strip()
    if policy not in CHAT_NATIVE_CORE_CANONICALIZATION_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_CANONICALIZATION_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_CANONICALIZATION_POLICIES)}"
        )
    return policy


def chat_native_core_family_target_policy() -> str:
    policy = str(
        science_env_get("STAGE2_FAMILY_TARGET_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGET_POLICY
    ).strip()
    if policy not in CHAT_NATIVE_CORE_FAMILY_TARGET_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_FAMILY_TARGET_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_FAMILY_TARGET_POLICIES)}"
        )
    return policy


def chat_native_core_family_targets() -> List[str]:
    raw_targets = str(
        science_env_get("STAGE2_FAMILY_TARGETS")
        or science_env_get("CHAT_NATIVE_CORE_FAMILY_TARGETS")
        or ",".join(DEFAULT_CHAT_NATIVE_CORE_FAMILY_TARGETS)
    )
    targets: List[str] = []
    invalid: List[str] = []
    for part in raw_targets.split(","):
        target = part.strip()
        if not target:
            continue
        if target not in CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS:
            invalid.append(target)
            continue
        if target not in targets:
            targets.append(target)
    if invalid:
        raise ValueError(
            f"Unsupported STAGE2_FAMILY_TARGETS values={invalid!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_ALLOWED_FAMILY_TARGETS)}"
        )
    if not targets:
        raise ValueError("STAGE2_FAMILY_TARGETS must include at least one target family.")
    return targets


def chat_native_core_target_vocab_policy() -> str:
    policy = str(
        science_env_get("STAGE2_TARGET_VOCAB_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY
    ).strip()
    if policy not in CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_TARGET_VOCAB_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES)}"
        )
    return policy


def chat_native_core_target_micro_vocab_options(target_family: str) -> List[str]:
    target_family = str(target_family or "").strip()
    preferred = list(CHAT_NATIVE_CORE_TARGET_MICRO_VOCAB_PREFERRED.get(target_family, ()))
    allowlist = d4_feature_name_allowlist(target_family)
    options = [name for name in preferred if name in allowlist]
    if not options:
        raise ValueError(
            "STAGE2_TARGET_VOCAB_POLICY=micro_canonical_options requires non-empty "
            f"registry-backed options for enabled target family {target_family!r}."
        )
    return options


def _chat_native_core_target_vocab_options_for_policy(target_family: str, policy: str) -> List[str]:
    policy = str(policy or DEFAULT_CHAT_NATIVE_CORE_TARGET_VOCAB_POLICY).strip()
    if policy == "none":
        return []
    if policy == "micro_canonical_options":
        return chat_native_core_target_micro_vocab_options(target_family)
    raise ValueError(
        f"Unsupported STAGE2_TARGET_VOCAB_POLICY={policy!r}; allowed="
        f"{sorted(CHAT_NATIVE_CORE_TARGET_VOCAB_POLICIES)}"
    )


def chat_native_core_target_thinning_policy() -> str:
    policy = str(
        science_env_get("STAGE2_TARGET_THINNING_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_TARGET_THINNING_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_TARGET_THINNING_POLICY
    ).strip()
    if policy not in CHAT_NATIVE_CORE_TARGET_THINNING_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_TARGET_THINNING_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_TARGET_THINNING_POLICIES)}"
        )
    return policy


def stage2_extraction_policy() -> str:
    policy = str(science_env_get("STAGE2_EXTRACTION_POLICY") or DEFAULT_CHAT_NATIVE_CORE_EXTRACTION_POLICY).strip()
    policy = policy or DEFAULT_CHAT_NATIVE_CORE_EXTRACTION_POLICY
    if policy not in ALLOWED_STAGE2_EXTRACTION_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_EXTRACTION_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_EXTRACTION_POLICIES)}"
        )
    return policy


def stage2_canonicalization_policy() -> str:
    policy = str(science_env_get("STAGE2_CANONICALIZATION_POLICY") or DEFAULT_CHAT_NATIVE_CORE_CANONICALIZATION_POLICY).strip()
    policy = policy or DEFAULT_CHAT_NATIVE_CORE_CANONICALIZATION_POLICY
    if policy not in ALLOWED_STAGE2_CANONICALIZATION_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_CANONICALIZATION_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_CANONICALIZATION_POLICIES)}"
        )
    return policy


def stage2_family_feature_policy() -> str:
    policy = str(science_env_get("STAGE2_FAMILY_FEATURE_POLICY") or "strict_family").strip()
    policy = policy or "strict_family"
    if policy not in ALLOWED_STAGE2_FAMILY_FEATURE_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_FAMILY_FEATURE_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_FAMILY_FEATURE_POLICIES)}"
        )
    return policy


def stage2_family_target_policy() -> str:
    policy = str(science_env_get("STAGE2_FAMILY_TARGET_POLICY") or DEFAULT_STAGE2_FAMILY_TARGET_POLICY).strip()
    policy = policy or DEFAULT_STAGE2_FAMILY_TARGET_POLICY
    if policy not in ALLOWED_STAGE2_FAMILY_TARGET_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_FAMILY_TARGET_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_FAMILY_TARGET_POLICIES)}"
        )
    return policy


def stage2_family_targets() -> List[str]:
    raw_targets = str(science_env_get("STAGE2_FAMILY_TARGETS") or ",".join(DEFAULT_STAGE2_FAMILY_TARGETS))
    targets: List[str] = []
    invalid: List[str] = []
    for part in raw_targets.split(","):
        target = part.strip()
        if not target:
            continue
        if target not in ALLOWED_STAGE2_FAMILY_TARGETS:
            invalid.append(target)
            continue
        if target not in targets:
            targets.append(target)
    if invalid:
        raise ValueError(
            f"Unsupported STAGE2_FAMILY_TARGETS values={invalid!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_FAMILY_TARGETS)}"
        )
    if not targets:
        raise ValueError("STAGE2_FAMILY_TARGETS must include at least one target family.")
    return targets


def stage2_target_vocab_policy() -> str:
    policy = str(science_env_get("STAGE2_TARGET_VOCAB_POLICY") or DEFAULT_STAGE2_TARGET_VOCAB_POLICY).strip()
    policy = policy or DEFAULT_STAGE2_TARGET_VOCAB_POLICY
    if policy not in ALLOWED_STAGE2_TARGET_VOCAB_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_TARGET_VOCAB_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_TARGET_VOCAB_POLICIES)}"
        )
    return policy


def stage2_target_micro_vocab_options(target_family: str) -> List[str]:
    target_family = str(target_family or "").strip()
    preferred = list(STAGE2_TARGET_MICRO_VOCAB_PREFERRED.get(target_family, ()))
    allowlist = d4_feature_name_allowlist(target_family)
    options = [name for name in preferred if name in allowlist]
    if not options:
        raise ValueError(
            "STAGE2_TARGET_VOCAB_POLICY=micro_canonical_options requires non-empty "
            f"registry-backed options for enabled target family {target_family!r}."
        )
    return options


def _stage2_target_vocab_options_for_policy(target_family: str, policy: str) -> List[str]:
    policy = str(policy or DEFAULT_STAGE2_TARGET_VOCAB_POLICY).strip()
    if policy == "none":
        return []
    if policy == "micro_canonical_options":
        return stage2_target_micro_vocab_options(target_family)
    raise ValueError(
        f"Unsupported STAGE2_TARGET_VOCAB_POLICY={policy!r}; allowed="
        f"{sorted(ALLOWED_STAGE2_TARGET_VOCAB_POLICIES)}"
    )


def stage2_target_thinning_policy() -> str:
    policy = str(science_env_get("STAGE2_TARGET_THINNING_POLICY") or DEFAULT_STAGE2_TARGET_THINNING_POLICY).strip()
    policy = policy or DEFAULT_STAGE2_TARGET_THINNING_POLICY
    if policy not in ALLOWED_STAGE2_TARGET_THINNING_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_TARGET_THINNING_POLICY={policy!r}; allowed="
            f"{sorted(ALLOWED_STAGE2_TARGET_THINNING_POLICIES)}"
        )
    return policy


def chat_native_core_termination_policy() -> str:
    raw_policy = (
        science_env_get("STAGE2_TERMINATION_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_TERMINATION_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_TERMINATION_POLICY
    )
    policy = str(raw_policy or "").strip()
    if policy not in CHAT_NATIVE_CORE_TERMINATION_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_TERMINATION_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_TERMINATION_POLICIES)}"
        )
    return policy


def chat_native_core_rule_count_policy() -> str:
    raw_policy = (
        science_env_get("STAGE2_RULE_COUNT_POLICY")
        or science_env_get("CHAT_NATIVE_CORE_RULE_COUNT_POLICY")
        or DEFAULT_CHAT_NATIVE_CORE_RULE_COUNT_POLICY
    )
    policy = str(raw_policy or "").strip()
    if policy not in CHAT_NATIVE_CORE_RULE_COUNT_POLICIES:
        raise ValueError(
            f"Unsupported STAGE2_RULE_COUNT_POLICY={policy!r}; allowed="
            f"{sorted(CHAT_NATIVE_CORE_RULE_COUNT_POLICIES)}"
        )
    return policy


def resolve_stage2_protocol_context(
    draft_family: str,
    projector_family: str,
    use_cross_model_projector: bool,
) -> dict:
    profile, profile_source = infer_stage2_protocol_profile(
        draft_family=draft_family,
        projector_family=projector_family,
        use_cross_model_projector=use_cross_model_projector,
    )
    profile_version = stage2_protocol_profile_version(profile)
    termination_policy = chat_native_core_termination_policy() if profile == "chat_native_core" else ""
    rule_count_policy = chat_native_core_rule_count_policy() if profile == "chat_native_core" else ""
    family_target_policy = (
        chat_native_core_family_target_policy()
        if profile == "chat_native_core"
        else (stage2_family_target_policy() if profile == "chemdfm_native" else "none")
    )
    target_vocab_policy = (
        chat_native_core_target_vocab_policy()
        if profile == "chat_native_core"
        else (stage2_target_vocab_policy() if profile == "chemdfm_native" else "none")
    )
    target_thinning_policy = (
        chat_native_core_target_thinning_policy()
        if profile == "chat_native_core"
        else (stage2_target_thinning_policy() if profile == "chemdfm_native" else "none")
    )
    family_target_families = (
        chat_native_core_family_targets() if profile == "chat_native_core" else stage2_family_targets()
    ) if (
        profile in {"chat_native_core", "chemdfm_native"}
        and family_target_policy == "family_targeted_single_rule_variants"
    ) else []
    return {
        "profile": profile,
        "profile_source": profile_source,
        "profile_version": profile_version,
        "termination_policy": termination_policy,
        "rule_count_policy": rule_count_policy,
        "family_target_policy": family_target_policy,
        "target_vocab_policy": target_vocab_policy,
        "target_thinning_policy": target_thinning_policy,
        "family_target_families": family_target_families,
    }
