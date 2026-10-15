import json
import os
from typing import Dict, Iterable, List, Tuple

from morace.candidate_ranking import (
    alias_group_jaccard as _alias_group_jaccard_impl,
    assign_rank_votes as _assign_rank_votes_impl,
    attach_template_novelty_scores as _attach_template_novelty_scores_impl,
    metric_rank_value as _metric_rank_value_impl,
    metric_tie_value as _metric_tie_value_impl,
    single_candidate_template_novelty as _single_candidate_template_novelty_impl,
    winner_explanation as _winner_explanation_impl,
)
from morace.datasets import load_split_xy
from morace.rule_canonicalizer import detect_cross_source_consensus_groups, merge_rules
from morace.rule_aggregation_metrics import (
    compile_safe_multi_feasibility_metrics as _compile_safe_multi_feasibility_metrics_impl,
    compile_safe_source_options as _compile_safe_source_options_impl,
    coverage_feasible_under_budget as _coverage_feasible_under_budget_impl,
    dominant_feature_share as _dominant_feature_share_impl,
    effective_source_coverage as _effective_source_coverage_impl,
    family_breakdown as _family_breakdown_impl,
    source_breakdown as _source_breakdown_impl,
)
from morace.rule_aggregation_reporting import (
    build_base_metrics as _build_base_metrics_impl,
    build_selection_trace as _build_selection_trace_impl,
    candidate_rule_set_summary as _candidate_rule_set_summary_impl,
)
from morace.rule_aliases import (
    alias_compaction_score as _alias_compaction_score_impl,
    alias_group_count as _alias_group_count_impl,
    alias_group_key as _alias_group_key_impl,
    alias_group_summary as _alias_group_summary_impl,
    direction_bucket as _direction_bucket_impl,
    is_semantically_equivalent_second_variant as _is_semantically_equivalent_second_variant_impl,
    mode_bucket as _mode_bucket_impl,
    root_feature_summary as _root_feature_summary_impl,
    second_variant_complement_dimensions as _second_variant_complement_dimensions_impl,
    second_variant_pair_quality as _second_variant_pair_quality_impl,
    semantic_value_polarity as _semantic_value_polarity_impl,
    semantic_variant_signature as _semantic_variant_signature_impl,
)
from morace.rule_candidate_selection import (
    alias_representative_sort_key as _alias_representative_sort_key_impl,
    candidate_quality_key as _candidate_quality_key_impl,
    family_matches as _family_matches_impl,
    build_lit_preserving_swap_candidate as _build_lit_preserving_swap_candidate_impl,
    has_literature_evidence as _has_literature_evidence_impl,
    lit_reserve_sort_key as _lit_reserve_sort_key_impl,
    replaceable_anchor_sort_key as _replaceable_anchor_sort_key_impl,
    select_alias_representatives as _select_alias_representatives_impl,
    source_priority_rank as _source_priority_rank_impl,
    template_type_from_name as _template_type_from_name_impl,
)
from morace.rule_candidate_sets import build_candidate_rule_sets as _build_candidate_rule_sets_impl
from morace.rule_template_selection import (
    select_balanced_coverage_rules as _select_balanced_coverage_rules_impl,
    select_template_rules as _select_template_rules_impl,
)
from morace.rule_verifier import (
    branch_preserving_select_top_rules,
    enrich_rules_with_statistics,
    passes_rule_gate,
    root_feature_key,
    select_top_rules,
    source_bucket,
)


FAMILY_ORDER = ("descriptor", "pattern_presence", "count")
SOURCE_ORDER = ("literature", "data", "multi")
CANDIDATE_SET_RANK_MODE = "average_rank_tie_aware"
CANDIDATE_SET_VOTE_METHOD = "borda_then_mrr_average_rank_tie_aware"
M7_LIT_PRESERVING_SWAP_ENABLE = False
FLOAT_TIE_METRICS = {
    "selected_avg_support",
    "selected_avg_effect_size",
    "template_novelty_score",
    "selected_avg_p_value",
}
SELECTOR_PROFILES = {
    "synthesize_baseline": {
        "max_same_root_feature": None,
        "require_non_descriptor": False,
        "source_priority": None,
    },
    "upgraded_single_selector": {
        "max_same_root_feature": 1,
        "require_non_descriptor": True,
        "source_priority": None,
    },
}


def _family_breakdown(rules: List[Dict]) -> Dict[str, int]:
    return _family_breakdown_impl(rules, family_order=FAMILY_ORDER)


def _source_breakdown(rules: List[Dict]) -> Dict[str, int]:
    return _source_breakdown_impl(
        rules,
        source_order=SOURCE_ORDER,
        source_bucket_fn=source_bucket,
    )


def _effective_source_coverage(source_breakdown: Dict[str, int]) -> Dict[str, int]:
    return _effective_source_coverage_impl(source_breakdown)


def _compile_safe_source_options(rule: Dict) -> List[Tuple[int, int]]:
    return _compile_safe_source_options_impl(rule, source_bucket_fn=source_bucket)


def _coverage_feasible_under_budget(
    compile_safe_rules: List[Dict],
    *,
    target_rule_count: int,
    min_effective_literature: int = 2,
    min_effective_data: int = 5,
    min_roots: int = 6,
) -> bool:
    return _coverage_feasible_under_budget_impl(
        compile_safe_rules,
        target_rule_count=target_rule_count,
        min_effective_literature=min_effective_literature,
        min_effective_data=min_effective_data,
        min_roots=min_roots,
        root_feature_key_fn=root_feature_key,
        compile_safe_source_options_fn=_compile_safe_source_options,
    )


def _compile_safe_multi_feasibility_metrics(
    merged_rules: List[Dict],
    enriched_rules: List[Dict],
    *,
    target_rule_count: int,
) -> Dict:
    return _compile_safe_multi_feasibility_metrics_impl(
        merged_rules,
        enriched_rules,
        target_rule_count=target_rule_count,
        root_feature_key_fn=root_feature_key,
        source_bucket_fn=source_bucket,
        detect_cross_source_consensus_groups_fn=detect_cross_source_consensus_groups,
        coverage_feasible_under_budget_fn=_coverage_feasible_under_budget,
    )


def _dominant_feature_share(rules: List[Dict]) -> float:
    return _dominant_feature_share_impl(rules)


def _direction_bucket(relation: str) -> str:
    return _direction_bucket_impl(relation)


def _mode_bucket(mode: str) -> str:
    return _mode_bucket_impl(mode)


def alias_group_key(rule: Dict) -> str:
    return _alias_group_key_impl(
        rule,
        root_feature_key_fn=root_feature_key,
        direction_bucket_fn=_direction_bucket,
        mode_bucket_fn=_mode_bucket,
    )


def _alias_group_summary(rules: List[Dict]) -> List[Dict]:
    return _alias_group_summary_impl(
        rules,
        alias_group_key_fn=alias_group_key,
        root_feature_key_fn=root_feature_key,
        source_bucket_fn=source_bucket,
        direction_bucket_fn=_direction_bucket,
        mode_bucket_fn=_mode_bucket,
    )


def _alias_group_count(rules: List[Dict]) -> int:
    return _alias_group_count_impl(rules, alias_group_key_fn=alias_group_key)


def _alias_compaction_score(rules: List[Dict]) -> float:
    return _alias_compaction_score_impl(rules, alias_group_count_fn=_alias_group_count)


def _semantic_value_polarity(rule: Dict) -> str:
    return _semantic_value_polarity_impl(rule, mode_bucket_fn=_mode_bucket)


def _semantic_variant_signature(rule: Dict) -> Tuple[str, str, str]:
    return _semantic_variant_signature_impl(
        rule,
        root_feature_key_fn=root_feature_key,
        semantic_value_polarity_fn=_semantic_value_polarity,
    )


def _is_semantically_equivalent_second_variant(existing_rule: Dict, candidate_rule: Dict) -> bool:
    return _is_semantically_equivalent_second_variant_impl(
        existing_rule,
        candidate_rule,
        semantic_variant_signature_fn=_semantic_variant_signature,
    )


def _root_feature_summary(rules: List[Dict]) -> List[Dict]:
    return _root_feature_summary_impl(
        rules,
        root_feature_key_fn=root_feature_key,
        source_bucket_fn=source_bucket,
        direction_bucket_fn=_direction_bucket,
        mode_bucket_fn=_mode_bucket,
    )


def _selection_trace(
    rule_list: List[Dict],
    enriched: List[Dict],
    selected: List[Dict],
    selection_audit: Dict | None = None,
) -> Dict:
    return _build_selection_trace_impl(
        rule_list,
        enriched,
        selected,
        selection_audit=selection_audit,
        passes_rule_gate_fn=passes_rule_gate,
        alias_group_summary_fn=_alias_group_summary,
        root_feature_summary_fn=_root_feature_summary,
    )


def _base_metrics(
    rule_list: List[Dict],
    merged: List[Dict],
    enriched: List[Dict],
    selected: List[Dict],
    target_rule_count: int,
    selection_audit: Dict | None = None,
) -> Dict:
    return _build_base_metrics_impl(
        rule_list,
        merged,
        enriched,
        selected,
        target_rule_count,
        selection_audit=selection_audit,
        passes_rule_gate_fn=passes_rule_gate,
        alias_group_count_fn=_alias_group_count,
        alias_compaction_score_fn=_alias_compaction_score,
        family_breakdown_fn=_family_breakdown,
        source_breakdown_fn=_source_breakdown,
        dominant_feature_share_fn=_dominant_feature_share,
        selection_trace_fn=_selection_trace,
        compile_safe_multi_feasibility_metrics_fn=_compile_safe_multi_feasibility_metrics,
    )


def _candidate_rule_set_summary(name: str, selected: List[Dict], selection_audit: Dict | None = None) -> Dict:
    return _candidate_rule_set_summary_impl(
        name,
        selected,
        selection_audit=selection_audit,
        template_type_from_name_fn=_template_type_from_name,
        alias_group_key_fn=alias_group_key,
        alias_group_count_fn=_alias_group_count,
        alias_compaction_score_fn=_alias_compaction_score,
        root_feature_key_fn=root_feature_key,
        family_breakdown_fn=_family_breakdown,
        source_breakdown_fn=_source_breakdown,
        effective_source_coverage_fn=_effective_source_coverage,
        direction_bucket_fn=_direction_bucket,
        mode_bucket_fn=_mode_bucket,
    )


def _metric_rank_value(item: Dict, metric_name: str):
    return _metric_rank_value_impl(item, metric_name)


def _metric_tie_value(item: Dict, metric_name: str):
    return _metric_tie_value_impl(
        item,
        metric_name,
        float_tie_metrics=FLOAT_TIE_METRICS,
        metric_rank_value_fn=_metric_rank_value,
    )


def _alias_group_jaccard(lhs: Dict, rhs: Dict) -> float:
    return _alias_group_jaccard_impl(lhs, rhs)


def _assign_rank_votes(candidate_summaries: List[Dict]) -> Dict[str, Dict]:
    return _assign_rank_votes_impl(
        candidate_summaries,
        metric_rank_value_fn=_metric_rank_value,
        metric_tie_value_fn=_metric_tie_value,
    )


def _candidate_quality_key(rule: Dict) -> Tuple:
    return _candidate_quality_key_impl(rule)


def _template_type_from_name(name: str) -> str:
    return _template_type_from_name_impl(name)


def _source_priority_rank(rule: Dict, source_priority: Tuple[str, ...]) -> int:
    return _source_priority_rank_impl(
        rule,
        source_priority,
        source_bucket_fn=source_bucket,
    )


def _alias_representative_sort_key(
    rule: Dict,
    source_priority: Tuple[str, ...],
    prefer_non_descriptor: bool = False,
) -> Tuple:
    return _alias_representative_sort_key_impl(
        rule,
        source_priority,
        prefer_non_descriptor,
        source_priority_rank_fn=_source_priority_rank,
    )


def _select_alias_representatives(
    rules: List[Dict],
    source_priority: Tuple[str, ...],
    prefer_non_descriptor: bool = False,
) -> List[Dict]:
    return _select_alias_representatives_impl(
        rules,
        source_priority,
        prefer_non_descriptor,
        alias_group_key_fn=alias_group_key,
        alias_representative_sort_key_fn=_alias_representative_sort_key,
    )


def _family_matches(rule: Dict, family_group: str) -> bool:
    return _family_matches_impl(rule, family_group)


def _second_variant_complement_dimensions(existing_rule: Dict, candidate_rule: Dict) -> List[str]:
    return _second_variant_complement_dimensions_impl(
        existing_rule,
        candidate_rule,
        source_bucket_fn=source_bucket,
        direction_bucket_fn=_direction_bucket,
        mode_bucket_fn=_mode_bucket,
    )


def _second_variant_pair_quality(existing_rules: List[Dict], candidate_rule: Dict) -> float:
    return _second_variant_pair_quality_impl(
        existing_rules,
        candidate_rule,
        source_bucket_fn=source_bucket,
        second_variant_complement_dimensions_fn=_second_variant_complement_dimensions,
    )


def _select_template_rules(
    candidates: List[Dict],
    target_rule_count: int,
    template_type: str,
    source_priority: Tuple[str, ...],
    require_non_descriptor_count: int = 0,
    descriptor_cap_ratio: float = 1.0,
    literature_anchor_count: int = 0,
    data_seed_count: int = 0,
    max_literature_rules: int | None = None,
    max_data_rules: int | None = None,
    coverage_plan: List[Tuple[str, str]] | None = None,
    max_same_root_per_template: int = 1,
    allow_second_variant_by_root: bool = False,
) -> Tuple[List[Dict], Dict]:
    return _select_template_rules_impl(
        candidates,
        target_rule_count,
        template_type,
        source_priority,
        require_non_descriptor_count,
        descriptor_cap_ratio,
        literature_anchor_count,
        data_seed_count,
        max_literature_rules,
        max_data_rules,
        coverage_plan,
        max_same_root_per_template,
        allow_second_variant_by_root,
        source_order=SOURCE_ORDER,
        root_feature_key_fn=root_feature_key,
        source_bucket_fn=source_bucket,
        alias_group_key_fn=alias_group_key,
        family_matches_fn=_family_matches,
        second_variant_complement_dimensions_fn=_second_variant_complement_dimensions,
        is_semantically_equivalent_second_variant_fn=_is_semantically_equivalent_second_variant,
        second_variant_pair_quality_fn=_second_variant_pair_quality,
        semantic_variant_signature_fn=_semantic_variant_signature,
    )


def _attach_template_novelty_scores(candidate_sets: List[Dict]) -> None:
    _attach_template_novelty_scores_impl(
        candidate_sets,
        alias_group_key_fn=alias_group_key,
    )


def _single_candidate_template_novelty(candidate: Dict, reference_candidates: List[Dict]) -> float:
    return _single_candidate_template_novelty_impl(
        candidate,
        reference_candidates,
        alias_group_key_fn=alias_group_key,
    )


def _select_balanced_coverage_rules(enriched: List[Dict], target_rule_count: int) -> Tuple[List[Dict], Dict]:
    return _select_balanced_coverage_rules_impl(
        enriched,
        target_rule_count,
        passes_rule_gate_fn=passes_rule_gate,
        candidate_quality_key_fn=_candidate_quality_key,
        root_feature_key_fn=root_feature_key,
        source_bucket_fn=source_bucket,
    )


def _lit_reserve_sort_key(rule: Dict, anchor_family_breakdown: Dict[str, int]) -> Tuple:
    return _lit_reserve_sort_key_impl(
        rule,
        anchor_family_breakdown,
        source_bucket_fn=source_bucket,
    )


def _replaceable_anchor_sort_key(rule: Dict, anchor_family_breakdown: Dict[str, int]) -> Tuple:
    return _replaceable_anchor_sort_key_impl(
        rule,
        anchor_family_breakdown,
        source_bucket_fn=source_bucket,
    )


def _has_literature_evidence(rule: Dict) -> bool:
    return _has_literature_evidence_impl(rule, source_bucket_fn=source_bucket)


def _build_lit_preserving_swap_candidate(
    anchor_candidate: Dict,
    gated_rules: List[Dict],
    target_rule_count: int,
) -> Tuple[List[Dict], Dict] | None:
    return _build_lit_preserving_swap_candidate_impl(
        anchor_candidate,
        gated_rules,
        target_rule_count,
        source_breakdown_fn=_source_breakdown,
        effective_source_coverage_fn=_effective_source_coverage,
        root_feature_key_fn=root_feature_key,
        alias_group_key_fn=alias_group_key,
        family_breakdown_fn=_family_breakdown,
        has_literature_evidence_fn=_has_literature_evidence,
        lit_reserve_sort_key_fn=_lit_reserve_sort_key,
        source_bucket_fn=source_bucket,
        replaceable_anchor_sort_key_fn=_replaceable_anchor_sort_key,
    )


def _winner_explanation(candidate_sets: List[Dict]) -> Dict:
    return _winner_explanation_impl(
        candidate_sets,
        alias_group_jaccard_fn=_alias_group_jaccard,
    )


def _build_candidate_rule_sets(enriched: List[Dict], target_rule_count: int, *, m7_lit_preserving_swap_enable: bool = False) -> List[Dict]:
    if type(m7_lit_preserving_swap_enable) is not bool:
        raise ValueError("m7_lit_preserving_swap_enable must be a boolean")
    return _build_candidate_rule_sets_impl(
        enriched,
        target_rule_count,
        m7_lit_preserving_swap_enable=m7_lit_preserving_swap_enable,
        passes_rule_gate_fn=passes_rule_gate,
        select_alias_representatives_fn=_select_alias_representatives,
        select_top_rules_fn=select_top_rules,
        branch_preserving_select_top_rules_fn=branch_preserving_select_top_rules,
        select_template_rules_fn=_select_template_rules,
        candidate_rule_set_summary_fn=_candidate_rule_set_summary,
        attach_template_novelty_scores_fn=_attach_template_novelty_scores,
        assign_rank_votes_fn=_assign_rank_votes,
        build_lit_preserving_swap_candidate_fn=_build_lit_preserving_swap_candidate,
        single_candidate_template_novelty_fn=_single_candidate_template_novelty,
    )


def aggregate_rule_candidates(
    rules: Iterable[Dict],
    dataset: str,
    subtask: str = "",
    input_folder: str = "data/splits",
    target_rule_count: int = 30,
    source_balance: bool = False,
    selector_profile: str = "synthesize_baseline",
    *,
    multi_consensus_promotion_enable: bool = False,
) -> Tuple[List[Dict], Dict]:
    rule_list = list(rules)
    merged = merge_rules(rule_list, multi_consensus_promotion_enable=multi_consensus_promotion_enable)
    smiles_train, y_train, spec = load_split_xy(dataset, subtask=subtask, split="train", input_folder=input_folder)
    enriched = enrich_rules_with_statistics(merged, smiles_train, y_train, spec.task_type)
    selector_settings = dict(SELECTOR_PROFILES.get(selector_profile, SELECTOR_PROFILES["synthesize_baseline"]))
    selected, selection_audit = select_top_rules(
        enriched,
        target_rule_count=target_rule_count,
        source_balance=source_balance,
        max_same_root_feature=selector_settings["max_same_root_feature"],
        require_non_descriptor=selector_settings["require_non_descriptor"],
        source_priority=selector_settings["source_priority"],
        return_audit=True,
    )
    metrics = _base_metrics(
        rule_list,
        merged,
        enriched,
        selected,
        target_rule_count=target_rule_count,
        selection_audit=selection_audit,
    )
    metrics["aggregate_mode"] = "single_selector"
    metrics["selector_profile"] = selector_profile
    return selected, metrics


def aggregate_rule_candidates_branch_preserving(
    rules: Iterable[Dict],
    dataset: str,
    subtask: str = "",
    input_folder: str = "data/splits",
    target_rule_count: int = 30,
    *,
    multi_consensus_promotion_enable: bool = False,
    m7_lit_preserving_swap_enable: bool = False,
) -> Tuple[List[Dict], Dict]:
    rule_list = list(rules)
    merged = merge_rules(rule_list, multi_consensus_promotion_enable=multi_consensus_promotion_enable)
    smiles_train, y_train, spec = load_split_xy(dataset, subtask=subtask, split="train", input_folder=input_folder)
    enriched = enrich_rules_with_statistics(merged, smiles_train, y_train, spec.task_type)
    candidate_sets = _build_candidate_rule_sets(enriched, target_rule_count=target_rule_count,
                                              m7_lit_preserving_swap_enable=m7_lit_preserving_swap_enable)
    winner = candidate_sets[0] if candidate_sets else {"rules": [], "selection_audit": {"dropped": []}, "name": "none"}
    selected = winner.get("rules", [])
    selected_source_breakdown = _source_breakdown(selected)
    selected_family_breakdown = _family_breakdown(selected)
    effective_source_coverage = _effective_source_coverage(selected_source_breakdown)
    selected_root_feature_count = len({root_feature_key(item) for item in selected if root_feature_key(item)})
    family_diversity = sum(1 for value in selected_family_breakdown.values() if value > 0)
    descriptor_only = bool(selected) and selected_family_breakdown.get("descriptor", 0) == len(selected)
    underdiverse_all = bool(
        len(selected) < 3
        or effective_source_coverage["literature"] < 1
        or effective_source_coverage["data"] < 1
        or family_diversity < 2
        or descriptor_only
    )
    metrics = {
        **_base_metrics(
            rule_list,
            merged,
            enriched,
            selected,
            target_rule_count=target_rule_count,
            selection_audit=winner.get("selection_audit", {"dropped": []}),
        ),
        "aggregate_mode": "candidate_set_ranked_voting",
        "candidate_set_count": len(candidate_sets),
        "candidate_set_rank_mode": CANDIDATE_SET_RANK_MODE,
        "template_type_count": len({item.get("template_type", "") for item in candidate_sets if item.get("template_type", "")}),
        "winner_template_type": winner.get("template_type", ""),
        "selected_literature_count": selected_source_breakdown.get("literature", 0),
        "selected_inference_count": selected_source_breakdown.get("data", 0),
        "selected_multi_count": selected_source_breakdown.get("multi", 0),
        "selected_root_feature_count": selected_root_feature_count,
        "selected_family_breakdown": selected_family_breakdown,
        "selected_source_breakdown": selected_source_breakdown,
        "effective_literature_coverage": effective_source_coverage["literature"],
        "effective_data_coverage": effective_source_coverage["data"],
        "family_diversity": family_diversity,
        "underdiverse_all": underdiverse_all,
        "candidate_set_vote_method": CANDIDATE_SET_VOTE_METHOD,
        "candidate_set_winner": winner.get("name", "none"),
        "candidate_set_winner_explanation": _winner_explanation(candidate_sets),
        "candidate_sets": candidate_sets,
    }
    return selected, metrics


def load_rule_artifact(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_rule_artifact(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
