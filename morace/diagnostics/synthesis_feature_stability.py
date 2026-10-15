import json
import os
from typing import Dict, List

from morace.diagnostics.rule_metadata import (
    _alias_group_key,
    _as_list,
    _direction_bucket,
    _mode_bucket,
    _root_feature_key,
)


DEFAULT_ROOTS = [
    "descriptor::TPSA",
    "descriptor::MolLogP",
    "count::NumHBD",
]


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_bucket(rule: Dict) -> str:
    support_sources = {str(item) for item in (rule.get("support_sources") or []) if item}
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("source", "") or rule.get("primary_source", "") or "")


def load_run_artifacts(
    project_root: str,
    run_tag: str,
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    synth_path = os.path.join(
        project_root,
        "eval_code_generation_repo",
        run_tag,
        model_name,
        dataset,
        "synthesize",
        f"{model_name}_{dataset}_synthesize_rules.json",
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
    log_path = os.path.join(project_root, "artifacts", run_tag, "logs", f"{dataset}.log")
    return {
        "run_tag": run_tag,
        "synth_path": synth_path,
        "all_path": all_path,
        "log_path": log_path,
        "synth_artifact": _load_json(synth_path),
        "all_artifact": _load_json(all_path),
    }


def extract_root_state_from_synth(synth_artifact: Dict, root_feature_key: str) -> Dict:
    branch = ((synth_artifact.get("branch_metrics") or {}).get("synthesize") or {})
    trace = branch.get("selection_trace") or {}
    raw_candidates = trace.get("raw_rules") or synth_artifact.get("raw_rules") or synth_artifact.get("rules") or []
    canonical_candidates = (
        trace.get("canonical_rules")
        or synth_artifact.get("canonical_rules")
        or (branch.get("selection_trace") or {}).get("selected_rules")
        or synth_artifact.get("rules")
        or []
    )
    raw_rules = [item for item in raw_candidates if _root_feature_key(item) == root_feature_key]
    canonical_rules = [item for item in canonical_candidates if _root_feature_key(item) == root_feature_key]
    final_rules = [item for item in (synth_artifact.get("rules") or []) if _root_feature_key(item) == root_feature_key]
    source_breakdown: Dict[str, int] = {}
    alias_groups = set()
    compile_valid_count = 0
    compile_valid_known_count = 0
    canonical_keys: List[str] = []
    for rule in canonical_rules:
        bucket = _source_bucket(rule)
        source_breakdown[bucket] = source_breakdown.get(bucket, 0) + 1
        alias_groups.add(_alias_group_key(rule))
        compile_valid = rule.get("compile_valid")
        if compile_valid is not None:
            compile_valid_known_count += 1
            if bool(compile_valid):
                compile_valid_count += 1
        canonical_key = str(rule.get("canonical_key", "")).strip()
        if canonical_key and canonical_key not in canonical_keys:
            canonical_keys.append(canonical_key)
    return {
        "root_present_in_synth_rules": bool(raw_rules),
        "root_present_in_canonical_rules": bool(canonical_rules),
        "root_present_in_final_synth_rules": bool(final_rules),
        "raw_rule_count": len(raw_rules),
        "canonical_rule_count": len(canonical_rules),
        "canonical_source_breakdown": source_breakdown,
        "canonical_compile_valid_count": compile_valid_count,
        "canonical_compile_valid_known_count": compile_valid_known_count,
        "canonical_alias_group_count": len(alias_groups),
        "canonical_alias_groups": sorted(alias_groups),
        "canonical_keys": canonical_keys,
    }


def extract_root_state_from_all(all_artifact: Dict, root_feature_key: str) -> Dict:
    combined = ((all_artifact.get("branch_metrics") or {}).get("combined") or {})
    consensus_groups = combined.get("cross_source_consensus_groups") or []
    consensus_group = next(
        (item for item in consensus_groups if str(item.get("root_feature_key", "")).strip() == root_feature_key),
        None,
    )
    final_rules = [item for item in (all_artifact.get("rules") or []) if _root_feature_key(item) == root_feature_key]
    source_breakdown: Dict[str, int] = {}
    final_sources = set()
    for rule in final_rules:
        bucket = _source_bucket(rule)
        source_breakdown[bucket] = source_breakdown.get(bucket, 0) + 1
        final_sources.add(bucket)
    final_source_bucket = ",".join(sorted(final_sources))
    promoted_multi_candidate_present = any(
        bool(rule.get("consensus_promoted", False)) and _root_feature_key(rule) == root_feature_key
        for rule in ((combined.get("selection_trace") or {}).get("gated_rules") or [])
    )
    promoted_multi_compile_safe_present = any(
        bool(rule.get("consensus_promoted", False))
        and bool(rule.get("compile_valid", False))
        and _root_feature_key(rule) == root_feature_key
        for rule in ((combined.get("selection_trace") or {}).get("gated_rules") or [])
    )
    overlap_sources = []
    if consensus_group:
        overlap_sources = (
            _as_list(consensus_group.get("overlap_sources"))
            or _as_list(consensus_group.get("source_buckets"))
            or _as_list(consensus_group.get("support_sources"))
        )
        if not overlap_sources and bool(consensus_group.get("has_cross_source_overlap")):
            overlap_sources = ["literature", "data"]
    return {
        "present_in_cross_source_consensus": bool(consensus_group and consensus_group.get("has_cross_source_overlap")),
        "cross_source_overlap_sources": overlap_sources,
        "cross_source_compile_valid_true_count": int(
            (consensus_group or {}).get("compile_valid_true_count", 0) or 0
        ),
        "promoted_multi_candidate_present": promoted_multi_candidate_present,
        "promoted_multi_compile_safe_present": promoted_multi_compile_safe_present,
        "present_in_final_winner": bool(final_rules),
        "final_source_bucket": final_source_bucket,
        "present_in_surviving_rules": bool(final_rules),
        "final_rule_count": len(final_rules),
        "final_source_breakdown": source_breakdown,
    }


def infer_first_drop_stage(root_row: Dict) -> str:
    if not bool(root_row.get("root_present_in_synth_rules", False)):
        return "synth_raw"
    if not bool(root_row.get("root_present_in_canonical_rules", False)):
        return "synth_canonical"
    if not bool(root_row.get("present_in_cross_source_consensus", False)):
        return "cross_source_consensus"
    if not bool(root_row.get("promoted_multi_candidate_present", False)):
        return "promotion_candidate"
    if not bool(root_row.get("present_in_final_winner", False)):
        return "final_winner"
    if not bool(root_row.get("present_in_surviving_rules", False)):
        return "surviving"
    return "present"


def build_root_comparison_row(run_tag: str, root_feature_key: str, synth_state: Dict, all_state: Dict) -> Dict:
    row = {
        "run_tag": run_tag,
        "root_feature_key": root_feature_key,
    }
    row.update(synth_state)
    row.update(all_state)
    row["first_missing_stage"] = infer_first_drop_stage(row)
    return row


def audit_two_runs(
    project_root: str,
    strong_run_tag: str,
    weak_run_tag: str,
    *,
    model_name: str,
    dataset: str,
    sample_dir: str,
    roots: List[str],
) -> Dict:
    strong = load_run_artifacts(project_root, strong_run_tag, model_name=model_name, dataset=dataset, sample_dir=sample_dir)
    weak = load_run_artifacts(project_root, weak_run_tag, model_name=model_name, dataset=dataset, sample_dir=sample_dir)

    comparison_rows: List[Dict] = []
    per_root: Dict[str, Dict] = {}
    for root in roots:
        strong_synth = extract_root_state_from_synth(strong["synth_artifact"], root)
        strong_all = extract_root_state_from_all(strong["all_artifact"], root)
        weak_synth = extract_root_state_from_synth(weak["synth_artifact"], root)
        weak_all = extract_root_state_from_all(weak["all_artifact"], root)
        strong_row = build_root_comparison_row(strong_run_tag, root, strong_synth, strong_all)
        weak_row = build_root_comparison_row(weak_run_tag, root, weak_synth, weak_all)
        comparison_rows.extend([strong_row, weak_row])
        per_root[root] = {
            "strong": strong_row,
            "weak": weak_row,
        }

    return {
        "project_root": project_root,
        "model_name": model_name,
        "dataset": dataset,
        "sample_dir": sample_dir,
        "strong_run_tag": strong_run_tag,
        "weak_run_tag": weak_run_tag,
        "roots": roots,
        "per_root": per_root,
        "comparison_rows": comparison_rows,
    }
