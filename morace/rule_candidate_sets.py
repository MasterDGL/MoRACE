"""Candidate-set orchestration for branch-preserving rule aggregation."""

from typing import Dict, List


def build_candidate_rule_sets(
    enriched: List[Dict],
    target_rule_count: int,
    *,
    m7_lit_preserving_swap_enable: bool,
    passes_rule_gate_fn,
    select_alias_representatives_fn,
    select_top_rules_fn,
    branch_preserving_select_top_rules_fn,
    select_template_rules_fn,
    candidate_rule_set_summary_fn,
    attach_template_novelty_scores_fn,
    assign_rank_votes_fn,
    build_lit_preserving_swap_candidate_fn,
    single_candidate_template_novelty_fn,
) -> List[Dict]:
    candidate_sets: List[Dict] = []
    gated = [item for item in enriched if passes_rule_gate_fn(item)]
    alias_reps_anchor = select_alias_representatives_fn(
        gated,
        source_priority=("data", "multi", "literature"),
        prefer_non_descriptor=False,
    )
    alias_reps_data = select_alias_representatives_fn(
        gated,
        source_priority=("data", "multi", "literature"),
        prefer_non_descriptor=False,
    )
    alias_reps_structure = select_alias_representatives_fn(
        gated,
        source_priority=("data", "literature", "multi"),
        prefer_non_descriptor=True,
    )
    alias_reps_balanced = select_alias_representatives_fn(
        gated,
        source_priority=("multi", "data", "literature"),
        prefer_non_descriptor=True,
    )
    strategies = [
        (
            "M1_top_global",
            lambda: select_top_rules_fn(
                enriched,
                target_rule_count=target_rule_count,
                source_balance=False,
                max_same_root_feature=1,
                require_non_descriptor=False,
                return_audit=True,
            ),
        ),
        (
            "M2_branch_preserving",
            lambda: branch_preserving_select_top_rules_fn(
                enriched,
                target_rule_count=target_rule_count,
                return_audit=True,
            ),
        ),
        (
            "M3_literature_anchor_data_consensus",
            lambda: select_template_rules_fn(
                alias_reps_anchor,
                target_rule_count=target_rule_count,
                source_priority=("data", "multi", "literature"),
                template_type="literature_anchor_data_consensus",
                literature_anchor_count=2,
                data_seed_count=2,
                require_non_descriptor_count=1,
                descriptor_cap_ratio=0.70,
                max_literature_rules=2,
                max_same_root_per_template=2,
                allow_second_variant_by_root=True,
            ),
        ),
        (
            "M4_data_heavy_alias",
            lambda: select_template_rules_fn(
                alias_reps_data,
                target_rule_count=target_rule_count,
                source_priority=("data", "multi", "literature"),
                template_type="data_heavy",
                data_seed_count=3,
                require_non_descriptor_count=1,
                descriptor_cap_ratio=0.80,
                max_literature_rules=1,
                max_same_root_per_template=2,
                allow_second_variant_by_root=True,
            ),
        ),
        (
            "M5_non_descriptor_structure",
            lambda: select_template_rules_fn(
                alias_reps_structure,
                target_rule_count=target_rule_count,
                source_priority=("data", "literature", "multi"),
                template_type="non_descriptor_structure",
                require_non_descriptor_count=2,
                descriptor_cap_ratio=0.45,
                max_literature_rules=2,
            ),
        ),
        (
            "M6_descriptor_capped_balanced",
            lambda: select_template_rules_fn(
                alias_reps_balanced,
                target_rule_count=target_rule_count,
                source_priority=("multi", "data", "literature"),
                template_type="descriptor_capped_balanced",
                require_non_descriptor_count=1,
                descriptor_cap_ratio=0.50,
                max_literature_rules=2,
                coverage_plan=[
                    ("literature", "non_descriptor"),
                    ("literature", "descriptor"),
                    ("data", "non_descriptor"),
                    ("data", "descriptor"),
                ],
                max_same_root_per_template=2,
                allow_second_variant_by_root=True,
            ),
        ),
    ]
    for name, builder in strategies:
        selected, selection_audit = builder()
        candidate_sets.append(
            candidate_rule_set_summary_fn(name, selected, selection_audit=selection_audit)
        )
    attach_template_novelty_scores_fn(candidate_sets)
    votes = assign_rank_votes_fn(candidate_sets)
    for item in candidate_sets:
        item.update(votes.get(item["name"], {}))
    candidate_sets = sorted(
        candidate_sets,
        key=lambda item: (
            -float(item.get("borda_score", 0.0)),
            -float(item.get("mrr_score", 0.0)),
            -int(item.get("selected_root_feature_count", 0)),
            -int(item.get("selected_rule_count", 0)),
            -float(item.get("template_novelty_score", 0.0)),
            -int(item.get("selected_alias_group_count", 0)),
            item["name"],
        ),
    )
    if m7_lit_preserving_swap_enable and candidate_sets:
        anchor_candidate = next(
            (item for item in candidate_sets if item.get("name") == "M1_top_global"),
            candidate_sets[0],
        )
        m7_payload = build_lit_preserving_swap_candidate_fn(
            anchor_candidate,
            gated,
            target_rule_count=target_rule_count,
        )
        if m7_payload is not None:
            selected, selection_audit = m7_payload
            m7_summary = candidate_rule_set_summary_fn(
                "M7_lit_preserving_swap",
                selected,
                selection_audit=selection_audit,
            )
            m7_summary["template_novelty_score"] = single_candidate_template_novelty_fn(
                m7_summary,
                candidate_sets,
            )
            m7_summary["borda_score"] = 0.0
            m7_summary["mrr_score"] = 0.0
            candidate_sets.append(m7_summary)
    return candidate_sets
