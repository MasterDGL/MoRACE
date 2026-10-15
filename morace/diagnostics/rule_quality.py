import copy
from typing import Dict, Iterable, List

from morace.ablation import canonical_key, evaluate_rule_variant, load_rule_artifact, source_bucket


def load_current_winner(all_rules_path: str) -> Dict:
    artifact = load_rule_artifact(all_rules_path)
    winner_rules = [copy.deepcopy(rule) for rule in artifact.get("rules", [])]
    return {
        "artifact": artifact,
        "dataset": str(artifact.get("dataset", "") or ""),
        "subtask": str(artifact.get("subtask", "") or ""),
        "winner_rules": winner_rules,
    }


def classify_rule_type(rule: Dict, winner_rules: Iterable[Dict]) -> str:
    root_key = f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"
    root_count = sum(
        1
        for item in winner_rules
        if f"{str(item.get('feature_family', ''))}::{str(item.get('feature_name', ''))}" == root_key
    )
    bucket = source_bucket(rule)
    if bool(rule.get("consensus_promoted", False)) and bucket == "multi":
        return "promoted_multi"
    if root_count > 1:
        return "same_root_alias"
    if bucket == "literature":
        return "literature_only"
    if bucket == "data":
        return "data_only"
    return "new_root_singleton"


def leave_one_out_variants(winner_rules: List[Dict]) -> List[Dict]:
    variants: List[Dict] = []
    for index, rule in enumerate(winner_rules):
        remaining = [copy.deepcopy(item) for pos, item in enumerate(winner_rules) if pos != index]
        variants.append(
            {
                "variant_name": f"drop_{index + 1}_{str(rule.get('feature_name', 'rule')).strip() or 'rule'}",
                "dropped_rule": copy.deepcopy(rule),
                "rules": remaining,
            }
        )
    return variants


def _suspect_flags(full_result: Dict, variant_result: Dict) -> Dict[str, bool]:
    valid_delta = float(variant_result.get("valid_mean", 0.0)) - float(full_result.get("valid_mean", 0.0))
    test_delta = float(variant_result.get("test_mean", 0.0)) - float(full_result.get("test_mean", 0.0))
    return {
        "possible_test_harmful_rule": bool(test_delta > 0.002 and valid_delta <= 0.0),
        "validation_friendly_test_harmful_rule": bool(test_delta > 0.002 and valid_delta < -0.001),
    }


def evaluate_leave_one_out_variants(
    winner_rules: List[Dict],
    dataset: str,
    subtask: str,
    full_result: Dict,
) -> List[Dict]:
    evaluated: List[Dict] = []
    for variant in leave_one_out_variants(winner_rules):
        dropped_rule = variant["dropped_rule"]
        result = evaluate_rule_variant(variant["variant_name"], variant["rules"], dataset, subtask)
        valid_delta = float(result.get("valid_mean", 0.0)) - float(full_result.get("valid_mean", 0.0))
        test_delta = float(result.get("test_mean", 0.0)) - float(full_result.get("test_mean", 0.0))
        audit_row = {
            "variant_name": variant["variant_name"],
            "dropped_rule_id": canonical_key(dropped_rule),
            "dropped_feature_name": str(dropped_rule.get("feature_name", "") or ""),
            "dropped_source_bucket": source_bucket(dropped_rule),
            "dropped_is_promoted_multi": bool(dropped_rule.get("consensus_promoted", False)),
            "dropped_root_feature_key": f"{str(dropped_rule.get('feature_family', ''))}::{str(dropped_rule.get('feature_name', ''))}",
            "dropped_rule_type": classify_rule_type(dropped_rule, winner_rules),
            "valid_mean": float(result.get("valid_mean", 0.0)),
            "test_mean": float(result.get("test_mean", 0.0)),
            "valid_delta_vs_full": valid_delta,
            "test_delta_vs_full": test_delta,
            "compiled_rule_count": int(result.get("compiled_rule_count", 0) or 0),
            "root_feature_count": int(result.get("root_feature_count", 0) or 0),
            "effective_literature_coverage": int(result.get("effective_literature_coverage", 0) or 0),
            "effective_data_coverage": int(result.get("effective_data_coverage", 0) or 0),
            "source_breakdown": dict(result.get("source_breakdown", {}) or {}),
            "feature_names": list(result.get("feature_names", []) or []),
        }
        audit_row.update(_suspect_flags(full_result, result))
        evaluated.append(audit_row)
    return evaluated


def summarize_leave_one_out(rows: List[Dict]) -> Dict:
    grouped: Dict[str, Dict] = {}
    for row in rows:
        group_key = str(row.get("dropped_rule_type", "") or "")
        entry = grouped.setdefault(
            group_key,
            {
                "rule_count": 0,
                "valid_deltas": [],
                "test_deltas": [],
            },
        )
        entry["rule_count"] += 1
        entry["valid_deltas"].append(float(row.get("valid_delta_vs_full", 0.0) or 0.0))
        entry["test_deltas"].append(float(row.get("test_delta_vs_full", 0.0) or 0.0))

    summary_rows = []
    for group_key, entry in grouped.items():
        valid_deltas = entry["valid_deltas"]
        test_deltas = entry["test_deltas"]
        summary_rows.append(
            {
                "rule_type": group_key,
                "rule_count": int(entry["rule_count"]),
                "avg_valid_delta": float(sum(valid_deltas) / len(valid_deltas)) if valid_deltas else 0.0,
                "avg_test_delta": float(sum(test_deltas) / len(test_deltas)) if test_deltas else 0.0,
                "max_test_delta": float(max(test_deltas)) if test_deltas else 0.0,
                "min_test_delta": float(min(test_deltas)) if test_deltas else 0.0,
            }
        )
    summary_rows.sort(key=lambda item: str(item.get("rule_type", "")))
    suspects = {
        "possible_test_harmful_rule": [row for row in rows if bool(row.get("possible_test_harmful_rule"))],
        "validation_friendly_test_harmful_rule": [
            row for row in rows if bool(row.get("validation_friendly_test_harmful_rule"))
        ],
    }
    return {
        "grouped_summary": summary_rows,
        "suspect_rules": suspects,
    }
