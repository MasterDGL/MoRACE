"""Canonical feature vocabulary helpers shared by projection stages."""

from typing import List, Sequence


STABLE_PATTERN_NAMES = {
    "aromatic_ring",
    "fused_ring",
    "amide",
    "amine",
    "halogen",
    "ester",
    "ether",
    "carbonyl",
    "carboxyl",
}


def registry_supported_names(attr_name: str, preferred: Sequence[str]) -> List[str]:
    import morace.rule_registry as rule_registry

    registry = getattr(rule_registry, attr_name) or {}
    return [name for name in preferred if name in registry]


def d4_feature_name_allowlist(feature_family: str) -> set[str]:
    import morace.rule_registry as rule_registry

    if feature_family == "descriptor":
        return set(rule_registry.DESCRIPTOR_REGISTRY.keys())
    if feature_family == "pattern_presence":
        custom_patterns = set(rule_registry.CUSTOM_PATTERN_FUNCTIONS.keys())
        # Do not admit every PATTERN_ALIAS key here: D4 v1 is boundary extraction, not alias rescue.
        return STABLE_PATTERN_NAMES | custom_patterns
    if feature_family == "count":
        return set(rule_registry.COUNT_ALIAS.keys())
    return set()


def expected_family_for_feature(feature_name: str) -> str:
    """Infer the safest rule family from canonical registry membership."""
    name = str(feature_name or "").strip()
    if not name:
        return ""
    # Count first: count-like features are useful diversity-bearing rules.
    if name in d4_feature_name_allowlist("count"):
        return "count"
    if name in d4_feature_name_allowlist("pattern_presence"):
        return "pattern_presence"
    if name in d4_feature_name_allowlist("descriptor"):
        return "descriptor"
    return ""
