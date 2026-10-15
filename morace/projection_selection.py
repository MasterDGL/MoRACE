"""Pure scoring and profiling helpers for projected rule candidates."""

from collections import Counter, defaultdict
from typing import Dict, List, Sequence

from morace.rule_canonicalizer import merge_rules
from morace.rule_compiler import compile_rule


DIRTY_DRAFT_TOKENS = (
    "### Instruction",
    "### Response",
    "### Example",
    "### Citation",
    "Question:",
    "Example:",
    "Citation:",
)


def classify_projection_status(
    raw_output: str,
    projected_rules: Sequence[Dict],
    heuristic_rules: Sequence[Dict],
) -> str:
    if projected_rules:
        return "projected"
    if heuristic_rules:
        return "heuristic_rescued"
    stripped = (raw_output or "").strip()
    if not stripped:
        return "empty"
    lowered = stripped.lower()
    if any(token in lowered for token in ('"rules"', "feature_family", "feature_name", "json")) or any(
        char in stripped for char in "{["
    ):
        return "parse_error"
    return "empty"


def draft_alignment_score(draft_text: str, rules: Sequence[Dict]) -> float:
    if not rules:
        return 0.0
    text = (draft_text or "").lower()
    hits = 0
    for rule in rules:
        feature_name = str(rule.get("feature_name", "")).lower()
        feature_tokens = feature_name.replace("_", " ").split()
        if feature_name and (feature_name in text or any(token and token in text for token in feature_tokens)):
            hits += 1
    return float(hits / max(1, len(rules)))


def compile_valid_ratio(rules: Sequence[Dict]) -> float:
    if not rules:
        return 0.0
    valid = sum(1 for rule in rules if compile_rule(rule) is not None)
    return float(valid / len(rules))


def compile_safe_rules(rules: Sequence[Dict]) -> List[Dict]:
    return [rule for rule in rules if compile_rule(rule) is not None]


def feature_family_diversity_count(rules: Sequence[Dict]) -> int:
    return len({str(rule.get("feature_family", "")) for rule in rules if rule.get("feature_family")})


def rule_family_list(rules: Sequence[Dict]) -> List[str]:
    families: List[str] = []
    for rule in rules or []:
        family = str(rule.get("feature_family", "") or "").strip()
        if family:
            families.append(family)
    return families


def feature_families(rules: Sequence[Dict]) -> List[str]:
    seen: List[str] = []
    for family in rule_family_list(rules):
        if family not in seen:
            seen.append(family)
    return seen


def mollogp_collapse_rate(rules: Sequence[Dict]) -> float:
    if not rules:
        return 0.0
    canonical_rules = merge_rules(list(rules))
    if not canonical_rules:
        return 0.0
    mollogp_count = sum(1 for rule in canonical_rules if str(rule.get("feature_name", "")) == "MolLogP")
    return float(mollogp_count / max(1, len(canonical_rules)))


def rule_root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"


def literature_projection_profile(rules: Sequence[Dict]) -> Dict[str, int | float | bool | List[str]]:
    canonical_rules = merge_rules(list(rules))
    safe_rules = compile_safe_rules(canonical_rules)
    root_counts = Counter(
        rule_root_feature_key(rule)
        for rule in canonical_rules
        if rule.get("feature_family") or rule.get("feature_name")
    )
    family_diversity = feature_family_diversity_count(canonical_rules)
    descriptor_only = bool(canonical_rules) and all(
        str(rule.get("feature_family", "")) == "descriptor" for rule in canonical_rules
    )
    dominant_root_share = (
        float(max(root_counts.values()) / len(canonical_rules))
        if canonical_rules and root_counts
        else 0.0
    )
    profile = {
        "canonical_rule_count": int(len(canonical_rules)),
        "compile_safe_count": int(len(safe_rules)),
        "root_feature_count": int(len(root_counts)),
        "family_diversity": int(family_diversity),
        "dominant_root_share": float(dominant_root_share),
        "descriptor_only": bool(descriptor_only),
        "mollogp_collapse_rate": float(mollogp_collapse_rate(canonical_rules)),
    }
    trigger_reasons: List[str] = []
    if profile["canonical_rule_count"] < 2:
        trigger_reasons.append("canonical_rule_count_lt_2")
    if profile["compile_safe_count"] < 2:
        trigger_reasons.append("compile_safe_count_lt_2")
    if profile["root_feature_count"] < 2:
        trigger_reasons.append("root_feature_count_lt_2")
    if profile["family_diversity"] < 2:
        trigger_reasons.append("family_diversity_lt_2")
    if profile["descriptor_only"] and profile["dominant_root_share"] >= 0.67:
        trigger_reasons.append("descriptor_only_dominant_root")
    profile["underdiverse_trigger_reasons"] = trigger_reasons
    profile["underdiverse"] = bool(trigger_reasons)
    return profile


def is_underdiverse_literature_projection(rules: Sequence[Dict]) -> bool:
    return bool(literature_projection_profile(rules).get("underdiverse", False))


def family_diversity_score(rules: Sequence[Dict]) -> float:
    families = {str(rule.get("feature_family", "")) for rule in rules if rule.get("feature_family")}
    return float(len(families) / 3.0)


def direction_conflict_penalty(rules: Sequence[Dict]) -> float:
    if not rules:
        return 0.0
    conflicts = 0
    for rule in rules:
        threshold = rule.get("threshold") or {}
        op = str(threshold.get("op", "")).strip()
        relation = str(rule.get("relation_to_target", "")).strip().lower()
        if not op or not relation:
            continue
        if relation in {"negative", "monotonic_down"} and op == ">=":
            conflicts += 1
        elif relation in {"positive", "monotonic_up"} and op == "<=":
            conflicts += 1
    return float(conflicts / max(1, len(rules)))


def example_leak_penalty(draft_text: str) -> float:
    text = draft_text or ""
    matches = sum(1 for token in DIRTY_DRAFT_TOKENS if token.lower() in text.lower())
    return min(1.0, 0.35 * matches)


def selection_components(item: Dict) -> Dict[str, float]:
    rules = item.get("parsed_rules", [])
    return {
        "strict_valid": float(bool(item.get("strict_projector_valid"))),
        "stage2_logprob_norm": float(item.get("stage2_logprob_norm", 0.0) or 0.0),
        "compile_valid_ratio": compile_valid_ratio(rules),
        "draft_alignment": draft_alignment_score(item.get("draft_text", ""), rules),
        "family_diversity": family_diversity_score(rules),
        "direction_conflict": direction_conflict_penalty(rules),
        "repair_used": float(bool(item.get("repair_used", False))),
        "example_leak": example_leak_penalty(item.get("draft_text", "")),
        "stage1_leak": float(item.get("stage1_leakage_score", 0.0) or 0.0),
        "stage2_leak": float(item.get("stage2_leakage_score", 0.0) or 0.0),
        "literature_recall_bonus": float(item.get("literature_recall_bonus", 0.0) or 0.0),
    }


def score_projected_record(item: Dict) -> float:
    parts = selection_components(item)
    return (
        2.5 * parts["strict_valid"]
        + 1.5 * parts["stage2_logprob_norm"]
        + 1.0 * parts["compile_valid_ratio"]
        + 0.8 * parts["draft_alignment"]
        + 0.5 * parts["family_diversity"]
        - 1.0 * parts["direction_conflict"]
        - 0.5 * parts["repair_used"]
        - 0.5 * parts["example_leak"]
        - 0.3 * parts["stage1_leak"]
        - 0.3 * parts["stage2_leak"]
        + parts["literature_recall_bonus"]
    )


def late_selection_root_keys(rules: Sequence[Dict]) -> set[str]:
    return {
        f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"
        for rule in rules
        if rule.get("feature_family") or rule.get("feature_name")
    }


def novel_root_keys(candidate_rules: Sequence[Dict], reference_rules: Sequence[Dict]) -> set[str]:
    return late_selection_root_keys(candidate_rules) - late_selection_root_keys(reference_rules)


def build_scored_projection_record(
    source_item: Dict,
    *,
    parsed_rules: Sequence[Dict],
    strict_valid: bool,
    repair_used: bool,
    stage2_logprob_norm: float,
    literature_recall_bonus: float = 0.0,
) -> Dict:
    candidate = dict(source_item)
    candidate["parsed_rules"] = list(parsed_rules)
    candidate["strict_projector_valid"] = bool(strict_valid)
    candidate["repair_used"] = bool(repair_used)
    candidate["heuristic_rescued"] = False
    candidate["stage2_logprob_norm"] = float(stage2_logprob_norm or 0.0)
    candidate["literature_recall_bonus"] = float(literature_recall_bonus or 0.0)
    return candidate


def literature_retry_assessment(
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
    recall_bonus_value: float,
    family_bonus_value: float,
    max_score_delta: float,
) -> Dict[str, int | float | bool]:
    original_profile = literature_projection_profile(original_rules)
    retry_profile = literature_projection_profile(retry_rules)
    new_root_keys = novel_root_keys(retry_rules, original_rules)
    original_scored = build_scored_projection_record(
        base_item,
        parsed_rules=original_rules,
        strict_valid=original_strict_valid,
        repair_used=original_repair_used,
        stage2_logprob_norm=original_stage2_logprob_norm,
        literature_recall_bonus=0.0,
    )
    retry_scored = build_scored_projection_record(
        base_item,
        parsed_rules=retry_rules,
        strict_valid=retry_strict_valid,
        repair_used=retry_repair_used,
        stage2_logprob_norm=retry_stage2_logprob_norm,
        literature_recall_bonus=0.0,
    )
    original_score = score_projected_record(original_scored)
    retry_score = score_projected_record(retry_scored)
    family_bonus = (
        float(family_bonus_value)
        if int(retry_profile.get("family_diversity", 0)) >= 2
        and int(original_profile.get("family_diversity", 0)) < 2
        else 0.0
    )
    recall_bonus = 0.0
    if new_root_keys and compile_valid_ratio(retry_rules) > 0.0:
        recall_bonus += float(recall_bonus_value)
    recall_bonus += family_bonus
    conditions = {
        "strict_valid_not_lower": int(bool(retry_strict_valid)) >= int(bool(original_strict_valid)),
        "compile_valid_positive": compile_valid_ratio(retry_rules) > 0.0,
        "novel_root_present": bool(new_root_keys),
        "score_within_delta": (original_score - retry_score) <= max_score_delta,
    }
    accepted = bool(conditions["novel_root_present"]) and sum(1 for value in conditions.values() if value) >= 3
    return {
        "accepted": accepted,
        "novel_root_count": int(len(new_root_keys)),
        "recall_bonus": float(recall_bonus),
        "base_selection_score": float(original_score),
        "retry_selection_score": float(retry_score),
        "score_delta_to_base": float(original_score - retry_score),
        "condition_pass_count": int(sum(1 for value in conditions.values() if value)),
        "original_profile": original_profile,
        "retry_profile": retry_profile,
        "family_bonus": float(family_bonus),
        "strict_valid_not_lower": bool(conditions["strict_valid_not_lower"]),
        "compile_valid_positive": bool(conditions["compile_valid_positive"]),
        "novel_root_present": bool(conditions["novel_root_present"]),
        "score_within_delta": bool(conditions["score_within_delta"]),
    }


def late_selection_novelty_pass(
    candidate_rules: Sequence[Dict],
    kept_rule_sets: Sequence[Sequence[Dict]],
) -> bool:
    candidate_roots = late_selection_root_keys(candidate_rules)
    if not candidate_roots:
        return False
    kept_roots = set()
    for rules in kept_rule_sets:
        kept_roots.update(late_selection_root_keys(rules))
    return bool(candidate_roots - kept_roots)


def select_best_projected_records(
    projected_records: Sequence[Dict],
    *,
    keep_topk: int,
    max_score_delta: float,
    max_survivors_per_prompt: int,
    require_novelty: bool,
) -> List[Dict]:
    by_prompt: Dict[str, List[Dict]] = defaultdict(list)
    for item in projected_records:
        by_prompt[item["prompt_id"]].append(item)
    rank_map: Dict[tuple, Dict[str, int | float | bool | str]] = {}
    for items in by_prompt.values():
        scored_items = []
        for item in items:
            updated = dict(item)
            updated["selection_components"] = selection_components(item)
            updated["selection_score"] = score_projected_record(item)
            updated["compile_valid_ratio"] = compile_valid_ratio(item.get("parsed_rules", []))
            scored_items.append(updated)
        scored_items = sorted(
            scored_items,
            key=lambda item: float(item.get("selection_score", 0.0) or 0.0),
            reverse=True,
        )
        total_candidates = len(scored_items)
        best_score = float(scored_items[0].get("selection_score", 0.0) or 0.0) if scored_items else 0.0
        kept_rule_sets: List[Sequence[Dict]] = []
        kept_count = 0
        for rank, ranked in enumerate(scored_items, 1):
            key = (
                ranked["prompt_id"],
                ranked["draft_model"],
                ranked["draft_id"],
                ranked.get("batch_id"),
                ranked.get("projection_variant_id", 0),
            )
            score_delta = float(best_score - float(ranked.get("selection_score", 0.0) or 0.0))
            novelty_pass = late_selection_novelty_pass(ranked.get("parsed_rules", []), kept_rule_sets)
            force_independent_single_survivor = bool(
                ranked.get("stage2_protocol_profile") == "chat_native_core"
                and ranked.get("rule_count_policy") == "two_independent_single_rule_projections"
                and ranked.get("projection_variant_role") == "independent_single_rule_2"
                and ranked.get("independent_single_projection_kept")
            )
            force_family_target_survivor = bool(
                ranked.get("stage2_protocol_profile") == "chat_native_core"
                and ranked.get("family_target_policy") == "family_targeted_single_rule_variants"
                and str(ranked.get("projection_variant_role", "") or "").startswith("family_target_")
                and ranked.get("family_target_variant_kept")
            )
            selected_for_merge = False
            selected_reason = ""
            if rank == 1:
                selected_for_merge = True
                selected_reason = "rank1_primary"
            elif force_independent_single_survivor:
                selected_for_merge = True
                selected_reason = "independent_single_rule_forced_survivor"
            elif force_family_target_survivor:
                selected_for_merge = True
                selected_reason = "family_target_forced_survivor"
            elif rank <= max(1, keep_topk):
                if kept_count >= max(1, max_survivors_per_prompt):
                    selected_reason = "survivor_cap_reached"
                elif score_delta > max_score_delta:
                    selected_reason = "score_delta_exceeded"
                elif float(ranked.get("compile_valid_ratio", 0.0) or 0.0) <= 0.0:
                    selected_reason = "compile_valid_ratio_zero"
                elif require_novelty and not novelty_pass:
                    selected_reason = "novelty_failed"
                else:
                    selected_for_merge = True
                    selected_reason = "runner_up_survivor"
            else:
                selected_reason = "rank_cutoff"
            if selected_for_merge:
                kept_count += 1
                kept_rule_sets.append(ranked.get("parsed_rules", []))
            rank_map[key] = {
                "rank": rank,
                "candidate_count": total_candidates,
                "best_score": best_score,
                "score_delta_to_best": score_delta,
                "selected_for_merge": selected_for_merge,
                "selected_for_merge_reason": selected_reason,
                "late_selection_novelty_pass": novelty_pass,
            }
    marked = []
    for item in projected_records:
        key = (
            item["prompt_id"],
            item["draft_model"],
            item["draft_id"],
            item.get("batch_id"),
            item.get("projection_variant_id", 0),
        )
        updated = dict(item)
        updated["selection_components"] = selection_components(item)
        updated["selection_score"] = score_projected_record(item)
        updated["compile_valid_ratio"] = compile_valid_ratio(item.get("parsed_rules", []))
        updated["draft_rank"] = int(rank_map.get(key, {}).get("rank", 1))
        updated["selected_from_k"] = int(rank_map.get(key, {}).get("candidate_count", 1))
        updated["late_selection_score_delta_to_best"] = float(
            rank_map.get(key, {}).get("score_delta_to_best", 0.0) or 0.0
        )
        updated["late_selection_novelty_pass"] = bool(
            rank_map.get(key, {}).get("late_selection_novelty_pass", False)
        )
        updated["selected_for_merge"] = bool(
            rank_map.get(key, {}).get("selected_for_merge", updated["draft_rank"] == 1)
        )
        updated["selected_for_merge_reason"] = str(
            rank_map.get(key, {}).get(
                "selected_for_merge_reason",
                "rank1_primary" if updated["draft_rank"] == 1 else "rank_cutoff",
            )
        )
        marked.append(updated)
    return marked
