"""Result-row orchestration for contribution experiments."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def run_loro_smoke(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
    *,
    candidate_paths_fn,
    evaluate_compiled_loro_fn,
    fmt_float_fn,
    canonical_key_fn,
    source_type_fn,
    contribution_signal_fn,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    from morace.rule_compiler import compile_rules

    loro_rows: List[Dict[str, object]] = []
    base_rows: List[Dict[str, object]] = []
    issue_rows: List[Dict[str, object]] = []

    for row in inventory_rows:
        pilot_id = row.get("pilot_id", "")
        try:
            paths = candidate_paths_fn(project_root, row)
            artifact = json.loads(paths["rules_file"].read_text(encoding="utf-8"))
            outcome = json.loads(paths["outcome_file"].read_text(encoding="utf-8"))
            rules = artifact.get("rules") or []
            compiled_rules, invalid_rules = compile_rules(rules)
            if invalid_rules:
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "compile_base",
                        "severity": "warning",
                        "issue": f"invalid_rules={len(invalid_rules)}",
                        "recommended_action": "review_before_expanding_loro",
                    }
                )
            if len(compiled_rules) <= 1:
                raise RuntimeError("Need at least two compiled rules for LORO")

            rf_seeds = [int(seed) for seed in (outcome.get("rf_seeds") or [0, 1, 2, 3, 4])]
            metric = outcome.get("metric_name") or row.get("metric", "")
            outcome_valid = float(outcome.get("actual_valid_mean"))
            outcome_test = float(outcome.get("actual_test_mean"))
            variant_scores, sklearn_mode = evaluate_compiled_loro_fn(
                project_root,
                row["dataset"],
                compiled_rules,
                manifest,
                rf_seeds,
            )
            base_eval = variant_scores["base"]
            valid_abs_diff = abs(float(base_eval["valid_mean"]) - outcome_valid)
            test_abs_diff = abs(float(base_eval["test_mean"]) - outcome_test)
            base_status = (
                "reproduced"
                if max(valid_abs_diff, test_abs_diff) <= reproduction_warning_threshold
                else "base_reproduction_warning"
            )
            base_note = (
                f"CPU RF reproduction within threshold; sklearn_mode={sklearn_mode}"
                if base_status == "reproduced"
                else f"CPU RF reproduction differs from outcome.json beyond threshold={reproduction_warning_threshold}; sklearn_mode={sklearn_mode}"
            )
            base_rows.append(
                {
                    "pilot_id": pilot_id,
                    "route_or_model": row.get("route_or_model", ""),
                    "dataset": row.get("dataset", ""),
                    "run_tag": row.get("run_tag", ""),
                    "seed": row.get("seed", ""),
                    "metric": metric,
                    "metric_direction": row.get("metric_direction", ""),
                    "outcome_valid_mean": fmt_float_fn(outcome_valid),
                    "outcome_test_mean": fmt_float_fn(outcome_test),
                    "reproduced_valid_mean": fmt_float_fn(base_eval["valid_mean"]),
                    "reproduced_valid_std": fmt_float_fn(base_eval["valid_std"]),
                    "reproduced_test_mean": fmt_float_fn(base_eval["test_mean"]),
                    "reproduced_test_std": fmt_float_fn(base_eval["test_std"]),
                    "valid_abs_diff": fmt_float_fn(valid_abs_diff),
                    "test_abs_diff": fmt_float_fn(test_abs_diff),
                    "base_reproduction_status": base_status,
                    "notes": base_note,
                }
            )
            if base_status == "base_reproduction_warning":
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "base_reproduction",
                        "severity": "warning",
                        "issue": f"valid_abs_diff={valid_abs_diff}; test_abs_diff={test_abs_diff}; threshold={reproduction_warning_threshold}",
                        "recommended_action": "treat_loro_as_diagnostic_and_review_reproduction_gap_before_expansion",
                    }
                )

            for idx, compiled in enumerate(compiled_rules):
                variant_name = f"drop_{idx}"
                ablated_eval = variant_scores[variant_name]
                valid_delta = float(ablated_eval["valid_mean"]) - float(base_eval["valid_mean"])
                test_delta = float(ablated_eval["test_mean"]) - float(base_eval["test_mean"])
                rule = compiled.rule
                key = canonical_key_fn(rule)
                loro_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "route_or_model": row.get("route_or_model", ""),
                        "dataset": row.get("dataset", ""),
                        "run_tag": row.get("run_tag", ""),
                        "seed": row.get("seed", ""),
                        "metric": metric,
                        "metric_direction": row.get("metric_direction", ""),
                        "rule_id": key,
                        "canonical_rule": key,
                        "feature_family": rule.get("feature_family", ""),
                        "root_feature": rule.get("feature_name", ""),
                        "source_type": source_type_fn(rule),
                        "base_valid_mean": fmt_float_fn(base_eval["valid_mean"]),
                        "base_valid_std": fmt_float_fn(base_eval["valid_std"]),
                        "base_test_mean": fmt_float_fn(base_eval["test_mean"]),
                        "base_test_std": fmt_float_fn(base_eval["test_std"]),
                        "ablated_valid_mean": fmt_float_fn(ablated_eval["valid_mean"]),
                        "ablated_valid_std": fmt_float_fn(ablated_eval["valid_std"]),
                        "ablated_test_mean": fmt_float_fn(ablated_eval["test_mean"]),
                        "ablated_test_std": fmt_float_fn(ablated_eval["test_std"]),
                        "valid_delta": fmt_float_fn(valid_delta),
                        "test_delta": fmt_float_fn(test_delta),
                        "contribution_signal": contribution_signal_fn(metric, valid_delta),
                        "base_reproduction_status": base_status,
                        "notes": f"exact_loro_smoke_cpu_rf; valid_delta_primary; test_delta_supplemental; sklearn_mode={sklearn_mode}; {ablated_eval.get('note', '')}",
                    }
                )
        except Exception as exc:
            issue_rows.append(
                {
                    "pilot_id": pilot_id,
                    "stage": "run_loro",
                    "severity": "error",
                    "issue": f"{exc.__class__.__name__}: {exc}",
                    "recommended_action": "fix_before_expanding_loro",
                }
            )
    return loro_rows, base_rows, issue_rows


def run_family_ablation(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
    *,
    candidate_paths_fn,
    evaluate_compiled_family_fn,
    fmt_float_fn,
    contribution_signal_fn,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    from morace.rule_compiler import compile_rules

    family_rows: List[Dict[str, object]] = []
    base_rows: List[Dict[str, object]] = []
    issue_rows: List[Dict[str, object]] = []

    for row in inventory_rows:
        pilot_id = row.get("pilot_id", "")
        try:
            paths = candidate_paths_fn(project_root, row)
            artifact = json.loads(paths["rules_file"].read_text(encoding="utf-8"))
            outcome = json.loads(paths["outcome_file"].read_text(encoding="utf-8"))
            rules = artifact.get("rules") or []
            compiled_rules, invalid_rules = compile_rules(rules)
            if invalid_rules:
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "compile_base",
                        "severity": "warning",
                        "issue": f"invalid_rules={len(invalid_rules)}",
                        "recommended_action": "review_family_ablation_outputs_with_compile_warning",
                    }
                )
            if len(compiled_rules) <= 1:
                raise RuntimeError("Need at least two compiled rules for family ablation")

            rf_seeds = [int(seed) for seed in (outcome.get("rf_seeds") or [0, 1, 2, 3, 4])]
            metric = outcome.get("metric_name") or row.get("metric", "")
            outcome_valid = float(outcome.get("actual_valid_mean"))
            outcome_test = float(outcome.get("actual_test_mean"))
            variant_scores, skipped_variants, sklearn_mode = evaluate_compiled_family_fn(
                project_root,
                row["dataset"],
                compiled_rules,
                manifest,
                rf_seeds,
            )
            base_eval = variant_scores["base"]
            valid_abs_diff = abs(float(base_eval["valid_mean"]) - outcome_valid)
            test_abs_diff = abs(float(base_eval["test_mean"]) - outcome_test)
            base_status = (
                "reproduced"
                if max(valid_abs_diff, test_abs_diff) <= reproduction_warning_threshold
                else "base_reproduction_warning"
            )
            base_note = (
                f"CPU RF reproduction within threshold; sklearn_mode={sklearn_mode}"
                if base_status == "reproduced"
                else f"CPU RF reproduction differs from outcome.json beyond threshold={reproduction_warning_threshold}; sklearn_mode={sklearn_mode}"
            )
            base_rows.append(
                {
                    "pilot_id": pilot_id,
                    "route_or_model": row.get("route_or_model", ""),
                    "dataset": row.get("dataset", ""),
                    "run_tag": row.get("run_tag", ""),
                    "seed": row.get("seed", ""),
                    "metric": metric,
                    "metric_direction": row.get("metric_direction", ""),
                    "outcome_valid_mean": fmt_float_fn(outcome_valid),
                    "outcome_test_mean": fmt_float_fn(outcome_test),
                    "reproduced_valid_mean": fmt_float_fn(base_eval["valid_mean"]),
                    "reproduced_valid_std": fmt_float_fn(base_eval["valid_std"]),
                    "reproduced_test_mean": fmt_float_fn(base_eval["test_mean"]),
                    "reproduced_test_std": fmt_float_fn(base_eval["test_std"]),
                    "valid_abs_diff": fmt_float_fn(valid_abs_diff),
                    "test_abs_diff": fmt_float_fn(test_abs_diff),
                    "base_reproduction_status": base_status,
                    "notes": base_note,
                }
            )
            if base_status == "base_reproduction_warning":
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "base_reproduction",
                        "severity": "warning",
                        "issue": f"valid_abs_diff={valid_abs_diff}; test_abs_diff={test_abs_diff}; threshold={reproduction_warning_threshold}",
                        "recommended_action": "treat_family_ablation_as_diagnostic_and_review_reproduction_gap",
                    }
                )

            for family in ("descriptor", "count", "pattern_presence"):
                removed_rule_count = sum(1 for compiled in compiled_rules if compiled.rule.get("feature_family") == family)
                if removed_rule_count == 0:
                    issue_rows.append(
                        {
                            "pilot_id": pilot_id,
                            "stage": "family_ablation",
                            "severity": "info",
                            "issue": f"{family}: family_not_present",
                            "recommended_action": "skip_missing_family",
                        }
                    )
                    continue
                remaining_rule_count = len(compiled_rules) - removed_rule_count
                variant_name = f"drop_family_{family}"
                if variant_name not in variant_scores:
                    issue = skipped_variants.get(family, "family_variant_not_available")
                    issue_rows.append(
                        {
                            "pilot_id": pilot_id,
                            "stage": "family_ablation",
                            "severity": "warning",
                            "issue": f"{family}: {issue}",
                            "recommended_action": "review_before_using_family_ablation_row",
                        }
                    )
                    family_rows.append(
                        {
                            "pilot_id": pilot_id,
                            "route_or_model": row.get("route_or_model", ""),
                            "dataset": row.get("dataset", ""),
                            "run_tag": row.get("run_tag", ""),
                            "seed": row.get("seed", ""),
                            "metric": metric,
                            "metric_direction": row.get("metric_direction", ""),
                            "removed_family": family,
                            "removed_rule_count": removed_rule_count,
                            "remaining_rule_count": remaining_rule_count,
                            "base_valid_mean": fmt_float_fn(base_eval["valid_mean"]),
                            "base_valid_std": fmt_float_fn(base_eval["valid_std"]),
                            "base_test_mean": fmt_float_fn(base_eval["test_mean"]),
                            "base_test_std": fmt_float_fn(base_eval["test_std"]),
                            "family_contribution_signal": "unknown",
                            "base_reproduction_status": base_status,
                            "notes": f"family_ablation_variant_not_evaluated; {issue}; valid_delta_primary; test_delta_supplemental",
                        }
                    )
                    continue

                ablated_eval = variant_scores[variant_name]
                valid_delta = float(ablated_eval["valid_mean"]) - float(base_eval["valid_mean"])
                test_delta = float(ablated_eval["test_mean"]) - float(base_eval["test_mean"])
                family_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "route_or_model": row.get("route_or_model", ""),
                        "dataset": row.get("dataset", ""),
                        "run_tag": row.get("run_tag", ""),
                        "seed": row.get("seed", ""),
                        "metric": metric,
                        "metric_direction": row.get("metric_direction", ""),
                        "removed_family": family,
                        "removed_rule_count": removed_rule_count,
                        "remaining_rule_count": remaining_rule_count,
                        "base_valid_mean": fmt_float_fn(base_eval["valid_mean"]),
                        "base_valid_std": fmt_float_fn(base_eval["valid_std"]),
                        "base_test_mean": fmt_float_fn(base_eval["test_mean"]),
                        "base_test_std": fmt_float_fn(base_eval["test_std"]),
                        "ablated_valid_mean": fmt_float_fn(ablated_eval["valid_mean"]),
                        "ablated_valid_std": fmt_float_fn(ablated_eval["valid_std"]),
                        "ablated_test_mean": fmt_float_fn(ablated_eval["test_mean"]),
                        "ablated_test_std": fmt_float_fn(ablated_eval["test_std"]),
                        "valid_delta": fmt_float_fn(valid_delta),
                        "test_delta": fmt_float_fn(test_delta),
                        "family_contribution_signal": contribution_signal_fn(metric, valid_delta),
                        "base_reproduction_status": base_status,
                        "notes": f"family_ablation_cpu_rf; valid_delta_primary; test_delta_supplemental; sklearn_mode={sklearn_mode}; {ablated_eval.get('note', '')}",
                    }
                )
        except Exception as exc:
            issue_rows.append(
                {
                    "pilot_id": pilot_id,
                    "stage": "run_family_ablation",
                    "severity": "error",
                    "issue": f"{exc.__class__.__name__}: {exc}",
                    "recommended_action": "fix_before_using_family_ablation_outputs",
                }
            )
    return family_rows, base_rows, issue_rows


def run_feature_importance(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
    *,
    candidate_paths_fn,
    evaluate_compiled_feature_importance_fn,
    fmt_float_fn,
    load_loro_lookup_fn,
    load_family_lookup_fn,
    rank_vectors_fn,
    mean_std_fn,
    canonical_key_fn,
    source_type_fn,
    truth_fn,
    feature_alignment_label_fn,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    from morace.rule_compiler import compile_rules

    feature_rows: List[Dict[str, object]] = []
    base_rows: List[Dict[str, object]] = []
    issue_rows: List[Dict[str, object]] = []
    loro_lookup = load_loro_lookup_fn(project_root)
    family_lookup = load_family_lookup_fn(project_root)

    for row in inventory_rows:
        pilot_id = row.get("pilot_id", "")
        try:
            paths = candidate_paths_fn(project_root, row)
            artifact = json.loads(paths["rules_file"].read_text(encoding="utf-8"))
            outcome = json.loads(paths["outcome_file"].read_text(encoding="utf-8"))
            rules = artifact.get("rules") or []
            compiled_rules, invalid_rules = compile_rules(rules)
            if invalid_rules:
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "compile_base",
                        "severity": "warning",
                        "issue": f"invalid_rules={len(invalid_rules)}",
                        "recommended_action": "review_feature_importance_outputs_with_compile_warning",
                    }
                )
            if len(compiled_rules) <= 1:
                raise RuntimeError("Need at least two compiled rules for feature importance")

            rf_seeds = [int(seed) for seed in (outcome.get("rf_seeds") or [0, 1, 2, 3, 4])]
            metric = outcome.get("metric_name") or row.get("metric", "")
            outcome_valid = float(outcome.get("actual_valid_mean"))
            outcome_test = float(outcome.get("actual_test_mean"))
            eval_data = evaluate_compiled_feature_importance_fn(
                project_root,
                row["dataset"],
                compiled_rules,
                manifest,
                rf_seeds,
            )
            valid_abs_diff = abs(float(eval_data["valid_mean"]) - outcome_valid)
            test_abs_diff = abs(float(eval_data["test_mean"]) - outcome_test)
            base_status = (
                "reproduced"
                if max(valid_abs_diff, test_abs_diff) <= reproduction_warning_threshold
                else "base_reproduction_warning"
            )
            base_note = (
                f"CPU RF reproduction within threshold; sklearn_mode={eval_data['sklearn_mode']}"
                if base_status == "reproduced"
                else f"CPU RF reproduction differs from outcome.json beyond threshold={reproduction_warning_threshold}; sklearn_mode={eval_data['sklearn_mode']}"
            )
            base_rows.append(
                {
                    "pilot_id": pilot_id,
                    "route_or_model": row.get("route_or_model", ""),
                    "dataset": row.get("dataset", ""),
                    "run_tag": row.get("run_tag", ""),
                    "seed": row.get("seed", ""),
                    "metric": metric,
                    "metric_direction": row.get("metric_direction", ""),
                    "outcome_valid_mean": fmt_float_fn(outcome_valid),
                    "outcome_test_mean": fmt_float_fn(outcome_test),
                    "reproduced_valid_mean": fmt_float_fn(eval_data["valid_mean"]),
                    "reproduced_valid_std": fmt_float_fn(eval_data["valid_std"]),
                    "reproduced_test_mean": fmt_float_fn(eval_data["test_mean"]),
                    "reproduced_test_std": fmt_float_fn(eval_data["test_std"]),
                    "valid_abs_diff": fmt_float_fn(valid_abs_diff),
                    "test_abs_diff": fmt_float_fn(test_abs_diff),
                    "base_reproduction_status": base_status,
                    "notes": base_note,
                }
            )
            if base_status == "base_reproduction_warning":
                issue_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "stage": "base_reproduction",
                        "severity": "warning",
                        "issue": f"valid_abs_diff={valid_abs_diff}; test_abs_diff={test_abs_diff}; threshold={reproduction_warning_threshold}",
                        "recommended_action": "treat_feature_importance_as_diagnostic_and_review_reproduction_gap",
                    }
                )

            kept_features = list(eval_data["kept_features"])
            kept_index_by_name = {name: idx for idx, name in enumerate(kept_features)}
            importances = eval_data["importances"]
            ranks = rank_vectors_fn(importances)
            kept_feature_count = len(kept_features)

            for compiled in compiled_rules:
                rule = compiled.rule
                key = canonical_key_fn(rule)
                family = str(rule.get("feature_family", ""))
                loro_row = loro_lookup.get((pilot_id, key), {})
                family_row = family_lookup.get((pilot_id, family), {})
                filtered = compiled.name not in kept_index_by_name
                if filtered:
                    imp_mean = imp_std = rank_mean = rank_min = rank_max = ""
                else:
                    idx = kept_index_by_name[compiled.name]
                    imp_values = [float(seed_importances[idx]) for seed_importances in importances]
                    rank_values = [int(seed_ranks[idx]) for seed_ranks in ranks]
                    imp_mean_value, imp_std_value = mean_std_fn(imp_values)
                    rank_mean_value, _ = mean_std_fn([float(item) for item in rank_values])
                    imp_mean = fmt_float_fn(imp_mean_value)
                    imp_std = fmt_float_fn(imp_std_value)
                    rank_mean = fmt_float_fn(rank_mean_value)
                    rank_min = min(rank_values)
                    rank_max = max(rank_values)

                loro_signal = loro_row.get("contribution_signal", "")
                family_signal = family_row.get("family_contribution_signal", "")
                label = feature_alignment_label_fn(
                    loro_signal,
                    family_signal,
                    rank_mean,
                    kept_feature_count,
                    filtered,
                )
                feature_rows.append(
                    {
                        "pilot_id": pilot_id,
                        "route_or_model": row.get("route_or_model", ""),
                        "dataset": row.get("dataset", ""),
                        "run_tag": row.get("run_tag", ""),
                        "seed": row.get("seed", ""),
                        "metric": metric,
                        "metric_direction": row.get("metric_direction", ""),
                        "rule_id": key,
                        "canonical_rule": key,
                        "feature_family": family,
                        "root_feature": rule.get("feature_name", ""),
                        "source_type": source_type_fn(rule),
                        "importance_mean": imp_mean,
                        "importance_std": imp_std,
                        "rank_mean": rank_mean,
                        "rank_min": rank_min,
                        "rank_max": rank_max,
                        "filtered_by_sanitizer": truth_fn(filtered),
                        "loro_valid_delta": loro_row.get("valid_delta", ""),
                        "loro_contribution_signal": loro_signal,
                        "family_valid_delta": family_row.get("valid_delta", ""),
                        "family_contribution_signal": family_signal,
                        "alignment_label": label,
                        "base_reproduction_status": base_status,
                        "notes": f"cpu_rf_feature_importance; not_used_for_rule_selection; sklearn_mode={eval_data['sklearn_mode']}; kept_feature_count={kept_feature_count}",
                    }
                )
        except Exception as exc:
            issue_rows.append(
                {
                    "pilot_id": pilot_id,
                    "stage": "run_feature_importance",
                    "severity": "error",
                    "issue": f"{exc.__class__.__name__}: {exc}",
                    "recommended_action": "fix_before_using_feature_importance_outputs",
                }
            )
    return feature_rows, base_rows, issue_rows



