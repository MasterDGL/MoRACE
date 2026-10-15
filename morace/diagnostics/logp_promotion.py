import json
import os
from typing import Dict, List, Sequence

from morace.diagnostics.rule_metadata import (
    _alias_group_key,
    _as_list,
    _direction_bucket,
    _mode_bucket,
    _root_feature_key,
    _selected_rules_from_projection,
    _source_bucket,
)


TARGET_ROOT = "descriptor::MolLogP"


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_existing_path(path: str, label: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _subset_for_root(rules: Sequence[Dict], root_feature_key: str) -> List[Dict]:
    return [rule for rule in (rules or []) if _root_feature_key(rule) == root_feature_key]


def load_run_artifacts(
    project_root: str,
    run_tag: str,
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    inference_path = os.path.join(
        project_root,
        "inference_model_response",
        run_tag,
        model_name,
        dataset,
        f"{model_name}_{dataset}_dk_response_sample_30.json",
    )
    synth_path = os.path.join(
        project_root,
        "synthesize_model_response",
        run_tag,
        model_name,
        dataset,
        f"{model_name}_{dataset}_pk_response.json",
    )
    all_path = os.path.join(
        project_root,
        "eval_code_generation_repo",
        run_tag,
        model_name,
        dataset,
        "all",
        sample_dir,
        f"{model_name}_{dataset}_all_rules.json",
    )
    inference_path = _require_existing_path(inference_path, f"inference artifact for {run_tag}")
    synth_path = _require_existing_path(synth_path, f"synth artifact for {run_tag}")
    all_path = _require_existing_path(all_path, f"all artifact for {run_tag}")
    return {
        "run_tag": run_tag,
        "inference_path": inference_path,
        "synth_path": synth_path,
        "all_path": all_path,
        "inference_artifact": _load_json(inference_path),
        "synth_artifact": _load_json(synth_path),
        "all_artifact": _load_json(all_path),
    }


def extract_mollogp_state_from_inference(inference_artifact: Dict) -> Dict:
    summary = (inference_artifact.get("root_presence_summary") or {}).get(TARGET_ROOT)
    if summary is not None:
        canonical_rules = _subset_for_root(inference_artifact.get("canonical_rules") or [], TARGET_ROOT)
        selected_rules = _subset_for_root(
            _selected_rules_from_projection(inference_artifact.get("stage2_projection") or []), TARGET_ROOT
        )
        return {
            "present_in_inference_raw_rules": bool(summary.get("present_in_raw_rules", False)),
            "present_in_inference_canonical_rules": bool(summary.get("present_in_canonical_rules", False)),
            "present_in_inference_compile_safe_rules": bool(summary.get("present_in_compile_safe_rules", False)),
            "present_in_inference_selected_rules": bool(summary.get("present_in_selected_rules", False)),
            "inference_source_bucket": ",".join(sorted({str(item).strip() for item in summary.get("sources", []) if str(item).strip()})),
            "inference_canonical_keys": sorted(
                {str(rule.get("canonical_key", "")).strip() for rule in canonical_rules if str(rule.get("canonical_key", "")).strip()}
            ),
            "inference_alias_group_keys": sorted({_alias_group_key(rule) for rule in selected_rules or canonical_rules if _alias_group_key(rule)}),
        }

    raw_rules = _subset_for_root(inference_artifact.get("rules") or [], TARGET_ROOT)
    canonical_rules = _subset_for_root(inference_artifact.get("canonical_rules") or [], TARGET_ROOT)
    compile_safe_rules = [rule for rule in canonical_rules if bool(rule.get("compile_valid", False))]
    selected_rules = _subset_for_root(
        _selected_rules_from_projection(inference_artifact.get("stage2_projection") or []), TARGET_ROOT
    )
    return {
        "present_in_inference_raw_rules": bool(raw_rules),
        "present_in_inference_canonical_rules": bool(canonical_rules),
        "present_in_inference_compile_safe_rules": bool(compile_safe_rules),
        "present_in_inference_selected_rules": bool(selected_rules),
        "inference_source_bucket": ",".join(
            sorted({_source_bucket(rule) for rule in selected_rules or canonical_rules if _source_bucket(rule)})
        ),
        "inference_canonical_keys": sorted(
            {str(rule.get("canonical_key", "")).strip() for rule in canonical_rules if str(rule.get("canonical_key", "")).strip()}
        ),
        "inference_alias_group_keys": sorted({_alias_group_key(rule) for rule in selected_rules or canonical_rules if _alias_group_key(rule)}),
    }


def extract_mollogp_state_from_synth(synth_artifact: Dict) -> Dict:
    summary = (synth_artifact.get("root_presence_summary") or {}).get(TARGET_ROOT)
    if summary is not None:
        canonical_rules = _subset_for_root(synth_artifact.get("canonical_rules") or [], TARGET_ROOT)
        return {
            "present_in_synth_raw_rules": bool(summary.get("present_in_raw_rules", False)),
            "present_in_synth_canonical_rules": bool(summary.get("present_in_canonical_rules", False)),
            "present_in_synth_compile_safe_rules": bool(summary.get("present_in_compile_safe_rules", False)),
            "synth_source_bucket": ",".join(sorted({str(item).strip() for item in summary.get("sources", []) if str(item).strip()})),
            "synth_canonical_keys": sorted(
                {str(rule.get("canonical_key", "")).strip() for rule in canonical_rules if str(rule.get("canonical_key", "")).strip()}
            ),
            "synth_alias_group_keys": sorted({_alias_group_key(rule) for rule in canonical_rules if _alias_group_key(rule)}),
        }

    branch = ((synth_artifact.get("branch_metrics") or {}).get("synthesize") or {})
    trace = branch.get("selection_trace") or {}
    raw_rules = _subset_for_root(
        trace.get("raw_rules") or synth_artifact.get("raw_rules") or synth_artifact.get("rules") or [],
        TARGET_ROOT,
    )
    canonical_rules = _subset_for_root(
        trace.get("canonical_rules") or synth_artifact.get("canonical_rules") or synth_artifact.get("rules") or [],
        TARGET_ROOT,
    )
    compile_safe_rules = [rule for rule in canonical_rules if bool(rule.get("compile_valid", False))]
    return {
        "present_in_synth_raw_rules": bool(raw_rules),
        "present_in_synth_canonical_rules": bool(canonical_rules),
        "present_in_synth_compile_safe_rules": bool(compile_safe_rules),
        "synth_source_bucket": ",".join(
            sorted({_source_bucket(rule) for rule in canonical_rules if _source_bucket(rule)})
        ),
        "synth_canonical_keys": sorted(
            {str(rule.get("canonical_key", "")).strip() for rule in canonical_rules if str(rule.get("canonical_key", "")).strip()}
        ),
        "synth_alias_group_keys": sorted({_alias_group_key(rule) for rule in canonical_rules if _alias_group_key(rule)}),
    }


def extract_mollogp_state_from_all(all_artifact: Dict) -> Dict:
    combined = ((all_artifact.get("branch_metrics") or {}).get("combined") or {})
    trace = combined.get("selection_trace") or {}
    gated_rules = _subset_for_root(trace.get("gated_rules") or [], TARGET_ROOT)
    consensus_groups = combined.get("cross_source_consensus_groups") or []
    consensus_group = next(
        (item for item in consensus_groups if str(item.get("root_feature_key", "")).strip() == TARGET_ROOT),
        None,
    )
    promoted_candidates = [rule for rule in gated_rules if bool(rule.get("consensus_promoted", False))]
    promoted_compile_safe = [rule for rule in promoted_candidates if bool(rule.get("compile_valid", False))]
    final_rules = _subset_for_root(all_artifact.get("rules") or [], TARGET_ROOT)
    if not promoted_candidates:
        promoted_candidates = [rule for rule in final_rules if bool(rule.get("consensus_promoted", False))]
    if not promoted_compile_safe:
        promoted_compile_safe = [rule for rule in promoted_candidates if bool(rule.get("compile_valid", False))]
    final_sources = sorted({_source_bucket(rule) for rule in final_rules if _source_bucket(rule)})
    overlap_sources = []
    if consensus_group:
        overlap_sources = (
            _as_list(consensus_group.get("overlap_sources"))
            or _as_list(consensus_group.get("source_buckets"))
            or _as_list(consensus_group.get("support_sources"))
        )
        if not overlap_sources and bool(consensus_group.get("has_cross_source_overlap")):
            overlap_sources = ["literature", "data"]

    present_as_surviving_multi = any(_source_bucket(rule) == "multi" for rule in final_rules)
    if not gated_rules:
        failure_reason = "missing_all_input"
    elif not bool(consensus_group and consensus_group.get("has_cross_source_overlap")):
        failure_reason = "no_cross_source_consensus"
    elif not promoted_candidates:
        failure_reason = "not_promoted"
    elif not promoted_compile_safe:
        failure_reason = "promoted_not_compile_safe"
    elif not present_as_surviving_multi:
        failure_reason = "promoted_not_surviving_multi"
    else:
        failure_reason = ""

    return {
        "present_in_all_input_rules": bool(gated_rules),
        "present_in_cross_source_consensus": bool(consensus_group and consensus_group.get("has_cross_source_overlap")),
        "cross_source_overlap_sources": overlap_sources,
        "cross_source_compile_valid_true_count": int((consensus_group or {}).get("compile_valid_true_count", 0) or 0),
        "promoted_multi_candidate_present": bool(promoted_candidates),
        "promoted_multi_compile_safe_present": bool(promoted_compile_safe),
        "promoted_multi_root_feature_key": TARGET_ROOT if promoted_candidates else "",
        "promotion_failure_reason": failure_reason,
        "present_in_final_winner": bool(final_rules),
        "final_source_bucket": ",".join(final_sources),
        "present_in_surviving_rules": bool(final_rules),
        "present_as_surviving_multi": bool(present_as_surviving_multi),
    }


def compute_first_missing_stage(mollogp_row: Dict) -> str:
    if not bool(mollogp_row.get("present_in_inference_selected_rules", False)):
        return "inference_selected"
    if not bool(mollogp_row.get("present_in_synth_raw_rules", False)):
        return "synth_raw"
    if not bool(mollogp_row.get("present_in_synth_canonical_rules", False)):
        return "synth_canonical"
    if not bool(mollogp_row.get("present_in_all_input_rules", False)):
        return "all_input"
    if not bool(mollogp_row.get("present_in_cross_source_consensus", False)):
        return "cross_source_consensus"
    if not bool(mollogp_row.get("promoted_multi_candidate_present", False)):
        return "promotion_candidate"
    if not bool(mollogp_row.get("promoted_multi_compile_safe_present", False)):
        return "promotion_compile_safe"
    if not bool(mollogp_row.get("present_in_final_winner", False)):
        return "final_winner"
    if not bool(mollogp_row.get("present_in_surviving_rules", False)):
        return "surviving_rules"
    return "present"


def _route_from_first_missing_stage(first_missing_stage: str) -> str:
    if first_missing_stage == "all_input":
        return "merge_ingestion_gap"
    if first_missing_stage == "cross_source_consensus":
        return "cross_source_matching_refinement"
    if first_missing_stage == "promotion_candidate":
        return "promotion_eligibility_refinement"
    if first_missing_stage == "promotion_compile_safe":
        return "compile_safe_realization_refinement"
    if first_missing_stage in {"final_winner", "surviving_rules"}:
        return "survivability_refinement"
    return "present_or_upstream_gap"


def build_mollogp_gap_row(run_tag: str, inference_state: Dict, synth_state: Dict, all_state: Dict) -> Dict:
    row = {
        "run_tag": run_tag,
        "root_feature_key": TARGET_ROOT,
    }
    row.update(inference_state)
    row.update(synth_state)
    row.update(all_state)
    row["first_missing_stage"] = compute_first_missing_stage(row)
    row["suggested_next_route"] = _route_from_first_missing_stage(row["first_missing_stage"])
    return row


def audit_runs(
    project_root: str,
    run_tags: Sequence[str],
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    rows: List[Dict] = []
    details: Dict[str, Dict] = {}
    for run_tag in run_tags:
        artifacts = load_run_artifacts(
            project_root,
            run_tag,
            model_name=model_name,
            dataset=dataset,
            sample_dir=sample_dir,
        )
        inference_state = extract_mollogp_state_from_inference(artifacts["inference_artifact"])
        synth_state = extract_mollogp_state_from_synth(artifacts["synth_artifact"])
        all_state = extract_mollogp_state_from_all(artifacts["all_artifact"])
        row = build_mollogp_gap_row(run_tag, inference_state, synth_state, all_state)
        rows.append(row)
        details[run_tag] = {
            "inference_path": artifacts["inference_path"],
            "synth_path": artifacts["synth_path"],
            "all_path": artifacts["all_path"],
            "inference_state": inference_state,
            "synth_state": synth_state,
            "all_state": all_state,
        }
    return {
        "project_root": project_root,
        "model_name": model_name,
        "dataset": dataset,
        "sample_dir": sample_dir,
        "run_tags": list(run_tags),
        "root_feature_key": TARGET_ROOT,
        "rows": rows,
        "details": details,
    }
