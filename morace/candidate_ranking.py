"""Tie-aware metric voting for aggregated rule candidates."""

from typing import Dict, List, Set


def metric_rank_value(item: Dict, metric_name: str):
    return item.get(metric_name, 0.0)


def metric_tie_value(
    item: Dict,
    metric_name: str,
    *,
    float_tie_metrics: Set[str],
    metric_rank_value_fn=metric_rank_value,
):
    value = metric_rank_value_fn(item, metric_name)
    if metric_name in float_tie_metrics:
        return round(float(value), 8)
    return value


def assign_rank_votes(
    candidate_summaries: List[Dict],
    *,
    metric_rank_value_fn=metric_rank_value,
    metric_tie_value_fn,
) -> Dict[str, Dict]:
    if not candidate_summaries:
        return {}
    metrics = [
        ("selected_rule_count", True),
        ("compiled_rule_count", True),
        ("selected_root_feature_count", True),
        ("family_diversity", True),
        ("effective_literature_coverage", True),
        ("effective_data_coverage", True),
        ("selected_non_descriptor_count", True),
        ("selected_avg_support", True),
        ("selected_avg_effect_size", True),
        ("template_novelty_score", True),
        ("second_variant_quality_score", True),
        ("selected_avg_p_value", False),
    ]
    total = len(candidate_summaries)
    votes = {
        item["name"]: {
            "metric_ranks": {},
            "borda_score": 0.0,
            "mrr_score": 0.0,
        }
        for item in candidate_summaries
    }
    for metric_name, reverse in metrics:
        ranked = sorted(
            candidate_summaries,
            key=lambda item: (metric_rank_value_fn(item, metric_name), item["name"]),
            reverse=reverse,
        )
        idx = 0
        while idx < len(ranked):
            current_value = metric_tie_value_fn(ranked[idx], metric_name)
            end = idx
            while end + 1 < len(ranked) and metric_tie_value_fn(ranked[end + 1], metric_name) == current_value:
                end += 1
            start_rank = idx + 1
            end_rank = end + 1
            avg_rank = float(start_rank + end_rank) / 2.0
            borda_points = float(total - avg_rank + 1.0)
            mrr_points = float(1.0 / avg_rank)
            for group_idx in range(idx, end + 1):
                item = ranked[group_idx]
                votes[item["name"]]["metric_ranks"][metric_name] = avg_rank
                votes[item["name"]]["borda_score"] += borda_points
                votes[item["name"]]["mrr_score"] += mrr_points
            idx = end + 1
    return votes


def attach_template_novelty_scores(candidate_sets: List[Dict], *, alias_group_key_fn) -> None:
    if not candidate_sets:
        return
    alias_sets = {
        item["name"]: {alias_group_key_fn(rule) for rule in item.get("rules", [])}
        for item in candidate_sets
    }
    for item in candidate_sets:
        current = alias_sets[item["name"]]
        distances = []
        for other_name, other in alias_sets.items():
            if other_name == item["name"]:
                continue
            union = current | other
            if not union:
                distances.append(0.0)
            else:
                distances.append(1.0 - (len(current & other) / len(union)))
        item["template_novelty_score"] = float(sum(distances) / len(distances)) if distances else 0.0


def single_candidate_template_novelty(
    candidate: Dict,
    reference_candidates: List[Dict],
    *,
    alias_group_key_fn,
) -> float:
    current = {
        alias_group_key_fn(rule)
        for rule in candidate.get("rules", [])
        if alias_group_key_fn(rule)
    }
    distances = []
    for other in reference_candidates:
        other_aliases = {
            alias_group_key_fn(rule)
            for rule in other.get("rules", [])
            if alias_group_key_fn(rule)
        }
        union = current | other_aliases
        if not union:
            distances.append(0.0)
        else:
            distances.append(1.0 - (len(current & other_aliases) / len(union)))
    return float(sum(distances) / len(distances)) if distances else 0.0


def alias_group_jaccard(lhs: Dict, rhs: Dict) -> float:
    lhs_aliases = set(lhs.get("alias_group_signature", []))
    rhs_aliases = set(rhs.get("alias_group_signature", []))
    union = lhs_aliases | rhs_aliases
    if not union:
        return 1.0
    return float(len(lhs_aliases & rhs_aliases) / len(union))


def winner_explanation(candidate_sets: List[Dict], *, alias_group_jaccard_fn=alias_group_jaccard) -> Dict:
    if not candidate_sets:
        return {"reason": "no_candidates"}
    winner = candidate_sets[0]
    runner_up = candidate_sets[1] if len(candidate_sets) > 1 else None
    explanation = {
        "winner": winner.get("name", "none"),
        "winner_template_type": winner.get("template_type", ""),
        "runner_up": runner_up.get("name", "") if runner_up else "",
        "runner_up_template_type": runner_up.get("template_type", "") if runner_up else "",
        "reason": "single_candidate_set" if runner_up is None else "higher_ranked_vote_bundle",
    }
    if runner_up is None:
        return explanation

    deltas = {
        "borda_score_delta": float(winner.get("borda_score", 0.0) - runner_up.get("borda_score", 0.0)),
        "mrr_score_delta": float(winner.get("mrr_score", 0.0) - runner_up.get("mrr_score", 0.0)),
        "rule_count_delta": int(
            winner.get("selected_rule_count", 0) - runner_up.get("selected_rule_count", 0)
        ),
        "data_coverage_delta": int(
            winner.get("effective_data_coverage", 0) - runner_up.get("effective_data_coverage", 0)
        ),
        "literature_coverage_delta": int(
            winner.get("effective_literature_coverage", 0)
            - runner_up.get("effective_literature_coverage", 0)
        ),
        "non_descriptor_delta": int(
            winner.get("selected_non_descriptor_count", 0)
            - runner_up.get("selected_non_descriptor_count", 0)
        ),
        "root_feature_delta": int(
            winner.get("selected_root_feature_count", 0)
            - runner_up.get("selected_root_feature_count", 0)
        ),
        "mean_support_delta": float(
            winner.get("mean_support", 0.0) - runner_up.get("mean_support", 0.0)
        ),
        "template_novelty_score_delta": float(
            winner.get("template_novelty_score", 0.0)
            - runner_up.get("template_novelty_score", 0.0)
        ),
        "second_variant_quality_score_delta": float(
            winner.get("second_variant_quality_score", 0.0)
            - runner_up.get("second_variant_quality_score", 0.0)
        ),
        "alias_group_jaccard_similarity": alias_group_jaccard_fn(winner, runner_up),
    }
    explanation["deltas_vs_runner_up"] = deltas
    if deltas["borda_score_delta"] == 0.0 and deltas["mrr_score_delta"] == 0.0:
        if int(winner.get("selected_root_feature_count", 0)) != int(
            runner_up.get("selected_root_feature_count", 0)
        ):
            explanation["winner_decided_by"] = "selected_root_feature_count"
        elif int(winner.get("selected_rule_count", 0)) != int(runner_up.get("selected_rule_count", 0)):
            explanation["winner_decided_by"] = "selected_rule_count"
        elif float(winner.get("template_novelty_score", 0.0)) != float(
            runner_up.get("template_novelty_score", 0.0)
        ):
            explanation["winner_decided_by"] = "template_novelty_score"
        elif int(winner.get("selected_alias_group_count", 0)) != int(
            runner_up.get("selected_alias_group_count", 0)
        ):
            explanation["winner_decided_by"] = "selected_alias_group_count"
        else:
            explanation["winner_decided_by"] = "name_tie_breaker"
    ranked = sorted(deltas.items(), key=lambda item: abs(float(item[1])), reverse=True)
    explanation["top_driver"] = ranked[0][0] if ranked else ""
    return explanation
