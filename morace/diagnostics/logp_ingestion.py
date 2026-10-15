from typing import Dict, List, Sequence

from morace.diagnostics.logp_promotion import TARGET_ROOT, load_run_artifacts
from morace.diagnostics.rule_metadata import (
    _alias_group_key,
    _direction_bucket,
    _effective_canonical_key,
    _mode_bucket,
    _root_feature_key,
    _selected_rules_from_projection,
    _source_bucket,
    _threshold_token,
)


def _subset_for_root(rules: Sequence[Dict], root_feature_key: str = TARGET_ROOT) -> List[Dict]:
    return [rule for rule in (rules or []) if _root_feature_key(rule) == root_feature_key]


def _base_object(rule: Dict, layer: str, run_tag: str) -> Dict:
    return {
        "run_tag": run_tag,
        "layer": layer,
        "canonical_key": _effective_canonical_key(rule),
        "feature_mode": str(rule.get("feature_mode", "")).strip(),
        "relation_to_target": str(rule.get("relation_to_target", "")).strip(),
        "direction_bucket": _direction_bucket(rule.get("relation_to_target", "")),
        "mode_bucket": _mode_bucket(rule.get("feature_mode", "")),
        "alias_group_key": _alias_group_key(rule),
        "source_bucket": _source_bucket(rule),
        "primary_source": str(rule.get("primary_source", "") or rule.get("source", "")).strip(),
        "support_sources": sorted({str(item).strip() for item in (rule.get("support_sources") or []) if str(item).strip()}),
        "compile_valid": bool(rule.get("compile_valid", False)),
    }


def extract_inference_selected_mollogp_objects(inference_artifact: Dict, run_tag: str) -> List[Dict]:
    selected_rules = _subset_for_root(_selected_rules_from_projection(inference_artifact.get("stage2_projection") or []))
    if not selected_rules:
        selected_rules = _subset_for_root(inference_artifact.get("selected_rules") or inference_artifact.get("canonical_rules") or [])
    return [_base_object(rule, "inference_selected", run_tag) for rule in selected_rules]


def extract_synth_canonical_mollogp_objects(synth_artifact: Dict, run_tag: str) -> List[Dict]:
    branch = ((synth_artifact.get("branch_metrics") or {}).get("synthesize") or {})
    trace = branch.get("selection_trace") or {}
    canonical_rules = _subset_for_root(
        synth_artifact.get("canonical_rules")
        or trace.get("canonical_rules")
        or trace.get("selected_rules")
        or synth_artifact.get("rules")
        or []
    )
    return [_base_object(rule, "synth_canonical", run_tag) for rule in canonical_rules]


def extract_consensus_group_state(all_artifact: Dict) -> Dict:
    combined = ((all_artifact.get("branch_metrics") or {}).get("combined") or {})
    group = next(
        (
            item
            for item in (combined.get("cross_source_consensus_groups") or [])
            if str(item.get("root_feature_key", "")).strip() == TARGET_ROOT
        ),
        None,
    )
    return {
        "consensus_group_has_cross_source_overlap": bool(group and group.get("has_cross_source_overlap")),
        "consensus_group_canonical_keys": list((group or {}).get("canonical_keys", []) or []),
        "shared_direction_buckets": list((group or {}).get("shared_direction_buckets", []) or []),
        "shared_mode_buckets": list((group or {}).get("shared_mode_buckets", []) or []),
        "cross_source_compile_valid_true_count": int((group or {}).get("compile_valid_true_count", 0) or 0),
    }


def extract_all_gated_mollogp_objects(
    all_artifact: Dict,
    inference_objects: Sequence[Dict],
    synth_objects: Sequence[Dict],
    run_tag: str,
) -> List[Dict]:
    combined = ((all_artifact.get("branch_metrics") or {}).get("combined") or {})
    gated_rules = _subset_for_root(((combined.get("selection_trace") or {}).get("gated_rules") or []))
    inference_keys = {item["canonical_key"] for item in inference_objects}
    inference_aliases = {item["alias_group_key"] for item in inference_objects}
    synth_keys = {item["canonical_key"] for item in synth_objects}
    synth_aliases = {item["alias_group_key"] for item in synth_objects}
    consensus_state = extract_consensus_group_state(all_artifact)
    rows: List[Dict] = []
    for rule in gated_rules:
        obj = _base_object(rule, "all_gated", run_tag)
        obj.update(
            {
                "present_in_inference_selected_by_exact_key": bool(obj["canonical_key"] in inference_keys),
                "present_in_inference_selected_by_alias_group": bool(obj["alias_group_key"] in inference_aliases),
                "present_in_synth_canonical_by_exact_key": bool(obj["canonical_key"] in synth_keys),
                "present_in_synth_canonical_by_alias_group": bool(obj["alias_group_key"] in synth_aliases),
                "included_in_consensus_group_keys": bool(
                    obj["canonical_key"] in set(consensus_state["consensus_group_canonical_keys"])
                ),
                "consensus_group_has_cross_source_overlap": bool(
                    consensus_state["consensus_group_has_cross_source_overlap"]
                ),
            }
        )
        rows.append(obj)
    return rows


def _ingestion_gap_summary(
    inference_objects: Sequence[Dict],
    synth_objects: Sequence[Dict],
    all_gated_objects: Sequence[Dict],
    consensus_state: Dict,
) -> Dict:
    gated_keys = {item["canonical_key"] for item in all_gated_objects}
    gated_aliases = {item["alias_group_key"] for item in all_gated_objects}
    inference_exact_in_gated = any(item["canonical_key"] in gated_keys for item in inference_objects)
    inference_alias_in_gated = any(item["alias_group_key"] in gated_aliases for item in inference_objects)
    synth_exact_in_gated = any(item["canonical_key"] in gated_keys for item in synth_objects)
    synth_alias_in_gated = any(item["alias_group_key"] in gated_aliases for item in synth_objects)
    gated_has_data = any(
        item["source_bucket"] in {"data", "multi"} or "data" in set(item["support_sources"])
        for item in all_gated_objects
    )
    gated_has_lit = any(
        item["source_bucket"] in {"literature", "multi"} or "literature" in set(item["support_sources"])
        for item in all_gated_objects
    )
    bookkeeping_issue = any(
        (item["present_in_inference_selected_by_exact_key"] or item["present_in_inference_selected_by_alias_group"])
        and item["source_bucket"] not in {"data", "multi"}
        for item in all_gated_objects
    ) or any(
        (item["present_in_inference_selected_by_exact_key"] or item["present_in_inference_selected_by_alias_group"])
        and "data" not in set(item["support_sources"])
        for item in all_gated_objects
    )
    if not inference_exact_in_gated and not inference_alias_in_gated:
        primary = "all_input_ingestion_gap"
        route = "branch_merge_input_retention_refinement"
    elif not synth_exact_in_gated and not synth_alias_in_gated:
        primary = "synth_input_ingestion_gap"
        route = "branch_merge_input_retention_refinement"
    elif bookkeeping_issue:
        primary = "source_bookkeeping_gap"
        route = "source_bookkeeping_refinement"
    elif not (gated_has_data and gated_has_lit):
        primary = "consensus_input_candidate_selection_gap"
        route = "consensus_input_candidate_refinement"
    elif not consensus_state["consensus_group_has_cross_source_overlap"]:
        primary = "consensus_grouping_gap"
        route = "consensus_grouping_refinement"
    else:
        primary = "present"
        route = "present_or_upstream_gap"
    return {
        "inference_exact_in_gated": bool(inference_exact_in_gated),
        "inference_alias_in_gated": bool(inference_alias_in_gated),
        "synth_exact_in_gated": bool(synth_exact_in_gated),
        "synth_alias_in_gated": bool(synth_alias_in_gated),
        "gated_has_data": bool(gated_has_data),
        "gated_has_literature": bool(gated_has_lit),
        "bookkeeping_issue_detected": bool(bookkeeping_issue),
        "primary_gap_reason": primary,
        "suggested_next_route": route,
    }


def audit_ingestion_gap(
    project_root: str,
    run_tags: Sequence[str],
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    rows: List[Dict] = []
    per_run: Dict[str, Dict] = {}
    for run_tag in run_tags:
        artifacts = load_run_artifacts(
            project_root,
            run_tag,
            model_name=model_name,
            dataset=dataset,
            sample_dir=sample_dir,
        )
        inference_objects = extract_inference_selected_mollogp_objects(artifacts["inference_artifact"], run_tag)
        synth_objects = extract_synth_canonical_mollogp_objects(artifacts["synth_artifact"], run_tag)
        consensus_state = extract_consensus_group_state(artifacts["all_artifact"])
        all_gated_objects = extract_all_gated_mollogp_objects(
            artifacts["all_artifact"],
            inference_objects,
            synth_objects,
            run_tag,
        )
        summary = _ingestion_gap_summary(inference_objects, synth_objects, all_gated_objects, consensus_state)
        rows.extend(inference_objects)
        rows.extend(synth_objects)
        rows.extend(all_gated_objects)
        per_run[run_tag] = {
            "inference_objects": inference_objects,
            "synth_objects": synth_objects,
            "all_gated_objects": all_gated_objects,
            "consensus_state": consensus_state,
            "summary": summary,
        }
    return {
        "project_root": project_root,
        "model_name": model_name,
        "dataset": dataset,
        "sample_dir": sample_dir,
        "run_tags": list(run_tags),
        "root_feature_key": TARGET_ROOT,
        "rows": rows,
        "per_run": per_run,
    }
