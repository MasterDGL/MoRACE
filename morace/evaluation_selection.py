"""Deterministic finalist shortlisting and coverage-guard selection."""

from typing import Dict, List, Tuple


def candidate_alias_signature(candidate: Dict, *, alias_group_key_fn) -> Tuple[str, ...]:
    signature = candidate.get("alias_group_signature", [])
    if signature:
        return tuple(sorted(str(item) for item in signature if item))
    rules = candidate.get("rules", []) or []
    return tuple(sorted(alias_group_key_fn(rule) for rule in rules if alias_group_key_fn(rule)))


def shortlist_all_finalists(
    candidate_sets: List[Dict],
    *,
    max_candidates: int,
    candidate_alias_signature_fn,
) -> List[Dict]:
    if not candidate_sets:
        return []

    max_candidates = max(1, max_candidates)
    by_name = {str(item.get("name", "")): item for item in candidate_sets}
    static_winner = by_name.get("M1_top_global", candidate_sets[0])
    finalists: List[Dict] = [{**dict(static_winner), "finalist_shortlist_reason": "static_baseline"}]

    baseline_lit = int(static_winner.get("effective_literature_coverage", 0) or 0)
    baseline_roots = int(static_winner.get("selected_root_feature_count", 0) or 0)
    remaining = [item for item in candidate_sets if item.get("name") != static_winner.get("name")]

    literature_improved = [
        item for item in remaining if int(item.get("effective_literature_coverage", 0) or 0) > baseline_lit
    ]
    if literature_improved:
        literature_improved = sorted(
            literature_improved,
            key=lambda item: (
                -int(item.get("effective_literature_coverage", 0) or 0),
                -int(item.get("selected_root_feature_count", 0) or 0),
                -float(item.get("borda_score", 0.0) or 0.0),
                -float(item.get("mrr_score", 0.0) or 0.0),
                -int(item.get("compiled_rule_count", 0) or 0),
                str(item.get("name", "")),
            ),
        )
        finalists.append({**dict(literature_improved[0]), "finalist_shortlist_reason": "literature_improved"})
    else:
        root_improved = [
            item for item in remaining if int(item.get("selected_root_feature_count", 0) or 0) > baseline_roots
        ]
        if root_improved:
            root_improved = sorted(
                root_improved,
                key=lambda item: (
                    -int(item.get("selected_root_feature_count", 0) or 0),
                    -float(item.get("borda_score", 0.0) or 0.0),
                    -float(item.get("mrr_score", 0.0) or 0.0),
                    str(item.get("name", "")),
                ),
            )
            finalists.append({**dict(root_improved[0]), "finalist_shortlist_reason": "root_improved_fallback"})

    while len(finalists) < max_candidates:
        existing_names = {str(item.get("name", "")) for item in finalists}
        existing_signatures = {candidate_alias_signature_fn(item) for item in finalists}
        next_candidate = None
        for item in candidate_sets:
            name = str(item.get("name", ""))
            if name in existing_names:
                continue
            signature = candidate_alias_signature_fn(item)
            if signature and signature in existing_signatures:
                continue
            next_candidate = {**dict(item), "finalist_shortlist_reason": "static_distinct"}
            break
        if next_candidate is None:
            break
        finalists.append(next_candidate)

    return finalists[:max_candidates]


def selection_objective(report: Dict) -> float:
    valid_mean = report.get("valid_mean")
    if valid_mean is None:
        return float("-inf")
    higher_is_better = bool(report.get("higher_is_better", True))
    value = float(valid_mean)
    return value if higher_is_better else -value


def objective_gap(best_report: Dict, other_report: Dict, *, selection_objective_fn=selection_objective) -> float:
    return float(selection_objective_fn(best_report) - selection_objective_fn(other_report))


def coverage_qualified_finalist(
    report: Dict,
    *,
    min_roots: int,
    min_effective_data: int,
    min_effective_literature: int,
) -> bool:
    return bool(
        int(report.get("compiled_rule_count", 0) or 0) >= min_roots
        and int(report.get("root_feature_count", 0) or 0) >= min_roots
        and int(report.get("effective_data_coverage", 0) or 0) >= min_effective_data
        and int(report.get("effective_literature_coverage", 0) or 0) >= min_effective_literature
    )


def choose_finalist_with_coverage_guard(
    finalist_reports: List[Dict],
    *,
    keep_literature_delta: float,
    selection_objective_fn=selection_objective,
    coverage_qualified_finalist_fn,
    objective_gap_fn,
) -> Dict:
    valid_reports = [item for item in finalist_reports if item.get("valid_mean") is not None]
    if not valid_reports:
        fallback = finalist_reports[0] if finalist_reports else {}
        return {
            "winner": fallback,
            "decision": "no_valid_scores_fallback",
            "best_valid_name": str(fallback.get("name", "") or ""),
            "coverage_guard_applied": False,
            "valid_gap_to_best": 0.0,
        }

    best_valid = max(
        valid_reports,
        key=lambda item: (
            selection_objective_fn(item),
            int(item.get("compiled_rule_count", 0) or 0),
            int(item.get("root_feature_count", 0) or 0),
            str(item.get("name", "")),
        ),
    )
    coverage_candidates = [item for item in valid_reports if coverage_qualified_finalist_fn(item)]
    coverage_best = None
    coverage_gap = None
    if coverage_candidates:
        coverage_best = max(
            coverage_candidates,
            key=lambda item: (
                selection_objective_fn(item),
                int(item.get("effective_literature_coverage", 0) or 0),
                int(item.get("root_feature_count", 0) or 0),
                str(item.get("name", "")),
            ),
        )
        coverage_gap = objective_gap_fn(best_valid, coverage_best)
        if coverage_gap <= keep_literature_delta:
            return {
                "winner": coverage_best,
                "decision": "coverage_guard_kept_within_delta",
                "best_valid_name": str(best_valid.get("name", "") or ""),
                "coverage_candidate_name": str(coverage_best.get("name", "") or ""),
                "coverage_guard_applied": True,
                "valid_gap_to_best": float(coverage_gap),
            }

    return {
        "winner": best_valid,
        "decision": "best_valid",
        "best_valid_name": str(best_valid.get("name", "") or ""),
        "coverage_candidate_name": str((coverage_best or {}).get("name", "") or ""),
        "coverage_guard_applied": False,
        "valid_gap_to_best": float(coverage_gap if coverage_gap is not None else 0.0),
    }
