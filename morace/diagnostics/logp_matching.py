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


def _synth_canonical_candidates(synth_artifact: Dict) -> List[Dict]:
    branch = ((synth_artifact.get("branch_metrics") or {}).get("synthesize") or {})
    trace = branch.get("selection_trace") or {}
    return (
        synth_artifact.get("canonical_rules")
        or trace.get("canonical_rules")
        or trace.get("selected_rules")
        or synth_artifact.get("rules")
        or []
    )


def _inference_selected_candidates(inference_artifact: Dict) -> List[Dict]:
    projection_selected = _selected_rules_from_projection(inference_artifact.get("stage2_projection") or [])
    if projection_selected:
        return projection_selected
    return (
        inference_artifact.get("selected_rules")
        or ((inference_artifact.get("metrics") or {}).get("selected_rules") if isinstance(inference_artifact.get("metrics"), dict) else None)
        or inference_artifact.get("canonical_rules")
        or []
    )


def extract_mollogp_synth_canonical_rules(synth_artifact: Dict) -> List[Dict]:
    rules = _subset_for_root(_synth_canonical_candidates(synth_artifact))
    rows = []
    for rule in rules:
        rows.append(
            {
                "canonical_key": _effective_canonical_key(rule),
                "feature_mode": str(rule.get("feature_mode", "")).strip(),
                "relation_to_target": str(rule.get("relation_to_target", "")).strip(),
                "direction_bucket": _direction_bucket(rule.get("relation_to_target", "")),
                "mode_bucket": _mode_bucket(rule.get("feature_mode", "")),
                "alias_group_key": _alias_group_key(rule),
                "compile_valid": bool(rule.get("compile_valid", False)),
                "source_bucket": _source_bucket(rule),
            }
        )
    return rows


def extract_mollogp_inference_selected_rules(inference_artifact: Dict) -> List[Dict]:
    rules = _subset_for_root(_inference_selected_candidates(inference_artifact))
    rows = []
    for rule in rules:
        rows.append(
            {
                "canonical_key": _effective_canonical_key(rule),
                "feature_mode": str(rule.get("feature_mode", "")).strip(),
                "relation_to_target": str(rule.get("relation_to_target", "")).strip(),
                "direction_bucket": _direction_bucket(rule.get("relation_to_target", "")),
                "mode_bucket": _mode_bucket(rule.get("feature_mode", "")),
                "alias_group_key": _alias_group_key(rule),
                "compile_valid": bool(rule.get("compile_valid", False)),
                "source_bucket": _source_bucket(rule),
            }
        )
    return rows


def extract_mollogp_all_consensus_state(all_artifact: Dict) -> Dict:
    combined = ((all_artifact.get("branch_metrics") or {}).get("combined") or {})
    gated_rules = _subset_for_root(((combined.get("selection_trace") or {}).get("gated_rules") or []))
    consensus_group = next(
        (
            item
            for item in (combined.get("cross_source_consensus_groups") or [])
            if str(item.get("root_feature_key", "")).strip() == TARGET_ROOT
        ),
        None,
    )
    return {
        "present_in_cross_source_consensus": bool(consensus_group and consensus_group.get("has_cross_source_overlap")),
        "shared_direction_buckets": list((consensus_group or {}).get("shared_direction_buckets", []) or []),
        "shared_mode_buckets": list((consensus_group or {}).get("shared_mode_buckets", []) or []),
        "cross_source_compile_valid_true_count": int((consensus_group or {}).get("compile_valid_true_count", 0) or 0),
        "consensus_canonical_keys": list((consensus_group or {}).get("canonical_keys", []) or []),
        "all_input_canonical_keys": sorted({_effective_canonical_key(rule) for rule in gated_rules}),
        "all_input_source_buckets": sorted({_source_bucket(rule) for rule in gated_rules if _source_bucket(rule)}),
        "all_input_alias_group_keys": sorted({_alias_group_key(rule) for rule in gated_rules if _alias_group_key(rule)}),
    }


def build_matching_pair_row(run_tag: str, synth_rule: Dict, inference_rule: Dict, all_state: Dict) -> Dict:
    direction_match = synth_rule["direction_bucket"] == inference_rule["direction_bucket"]
    mode_match = synth_rule["mode_bucket"] == inference_rule["mode_bucket"]
    alias_match = synth_rule["alias_group_key"] == inference_rule["alias_group_key"]
    synth_key = synth_rule["canonical_key"]
    inf_key = inference_rule["canonical_key"]
    return {
        "run_tag": run_tag,
        "synth_canonical_key": synth_key,
        "inference_canonical_key": inf_key,
        "synth_feature_mode": synth_rule["feature_mode"],
        "inference_feature_mode": inference_rule["feature_mode"],
        "synth_relation": synth_rule["relation_to_target"],
        "inference_relation": inference_rule["relation_to_target"],
        "synth_direction_bucket": synth_rule["direction_bucket"],
        "inference_direction_bucket": inference_rule["direction_bucket"],
        "synth_mode_bucket": synth_rule["mode_bucket"],
        "inference_mode_bucket": inference_rule["mode_bucket"],
        "synth_alias_group_key": synth_rule["alias_group_key"],
        "inference_alias_group_key": inference_rule["alias_group_key"],
        "direction_match": bool(direction_match),
        "mode_match": bool(mode_match),
        "alias_match": bool(alias_match),
        "canonical_exact_match": bool(synth_key == inf_key),
        "would_match_under_root_only": True,
        "would_match_under_root_plus_direction": bool(direction_match),
        "would_match_under_root_plus_direction_plus_mode": bool(direction_match and mode_match),
        "present_in_cross_source_consensus": bool(all_state.get("present_in_cross_source_consensus", False)),
        "shared_direction_buckets": list(all_state.get("shared_direction_buckets", [])),
        "shared_mode_buckets": list(all_state.get("shared_mode_buckets", [])),
        "cross_source_compile_valid_true_count": int(all_state.get("cross_source_compile_valid_true_count", 0) or 0),
        "consensus_canonical_keys": list(all_state.get("consensus_canonical_keys", [])),
        "all_input_canonical_keys": list(all_state.get("all_input_canonical_keys", [])),
        "all_input_source_buckets": list(all_state.get("all_input_source_buckets", [])),
        "all_input_alias_group_keys": list(all_state.get("all_input_alias_group_keys", [])),
        "inference_selected_key_present_in_all_input": bool(inf_key in set(all_state.get("all_input_canonical_keys", []))),
        "synth_canonical_key_present_in_all_input": bool(synth_key in set(all_state.get("all_input_canonical_keys", []))),
    }


def _matching_gap_reason(row: Dict) -> str:
    if not row.get("inference_selected_key_present_in_all_input", False):
        return "inference_key_missing_in_all_input"
    if not row.get("synth_canonical_key_present_in_all_input", False):
        return "synth_key_missing_in_all_input"
    if not row.get("direction_match", False):
        return "direction_bucket_mismatch"
    if not row.get("mode_match", False):
        return "mode_bucket_mismatch"
    if not row.get("alias_match", False):
        return "alias_group_mismatch"
    if not row.get("canonical_exact_match", False):
        return "canonical_key_mismatch_but_bucket_aligned"
    return "pair_aligned"


def _build_empty_pair_row(run_tag: str, all_state: Dict, *, reason: str) -> Dict:
    return {
        "run_tag": run_tag,
        "synth_canonical_key": "",
        "inference_canonical_key": "",
        "synth_feature_mode": "",
        "inference_feature_mode": "",
        "synth_relation": "",
        "inference_relation": "",
        "synth_direction_bucket": "",
        "inference_direction_bucket": "",
        "synth_mode_bucket": "",
        "inference_mode_bucket": "",
        "synth_alias_group_key": "",
        "inference_alias_group_key": "",
        "direction_match": False,
        "mode_match": False,
        "alias_match": False,
        "canonical_exact_match": False,
        "would_match_under_root_only": False,
        "would_match_under_root_plus_direction": False,
        "would_match_under_root_plus_direction_plus_mode": False,
        "present_in_cross_source_consensus": bool(all_state.get("present_in_cross_source_consensus", False)),
        "shared_direction_buckets": list(all_state.get("shared_direction_buckets", [])),
        "shared_mode_buckets": list(all_state.get("shared_mode_buckets", [])),
        "cross_source_compile_valid_true_count": int(all_state.get("cross_source_compile_valid_true_count", 0) or 0),
        "consensus_canonical_keys": list(all_state.get("consensus_canonical_keys", [])),
        "all_input_canonical_keys": list(all_state.get("all_input_canonical_keys", [])),
        "all_input_source_buckets": list(all_state.get("all_input_source_buckets", [])),
        "all_input_alias_group_keys": list(all_state.get("all_input_alias_group_keys", [])),
        "inference_selected_key_present_in_all_input": False,
        "synth_canonical_key_present_in_all_input": False,
        "matching_gap_reason": reason,
    }


def audit_matching_diff(
    project_root: str,
    run_tags: Sequence[str],
    *,
    model_name: str,
    dataset: str,
    sample_dir: str = "sample_30",
) -> Dict:
    per_run: Dict[str, Dict] = {}
    pair_rows: List[Dict] = []
    for run_tag in run_tags:
        artifacts = load_run_artifacts(
            project_root,
            run_tag,
            model_name=model_name,
            dataset=dataset,
            sample_dir=sample_dir,
        )
        synth_rules = extract_mollogp_synth_canonical_rules(artifacts["synth_artifact"])
        inference_rules = extract_mollogp_inference_selected_rules(artifacts["inference_artifact"])
        all_state = extract_mollogp_all_consensus_state(artifacts["all_artifact"])
        run_pair_rows: List[Dict] = []
        if not synth_rules:
            row = _build_empty_pair_row(run_tag, all_state, reason="no_synth_pair_candidate")
            pair_rows.append(row)
            run_pair_rows.append(row)
        elif not inference_rules:
            row = _build_empty_pair_row(run_tag, all_state, reason="no_inference_pair_candidate")
            pair_rows.append(row)
            run_pair_rows.append(row)
        else:
            for synth_rule in synth_rules:
                for inference_rule in inference_rules:
                    row = build_matching_pair_row(run_tag, synth_rule, inference_rule, all_state)
                    row["matching_gap_reason"] = _matching_gap_reason(row)
                    pair_rows.append(row)
                    run_pair_rows.append(row)
        per_run[run_tag] = {
            "synth_rule_count": len(synth_rules),
            "inference_rule_count": len(inference_rules),
            "all_consensus_state": all_state,
            "synth_rules": synth_rules,
            "inference_rules": inference_rules,
            "pair_rows": run_pair_rows,
        }
    return {
        "project_root": project_root,
        "model_name": model_name,
        "dataset": dataset,
        "sample_dir": sample_dir,
        "run_tags": list(run_tags),
        "root_feature_key": TARGET_ROOT,
        "per_run": per_run,
        "pair_rows": pair_rows,
    }
