#!/usr/bin/env python
"""Rule contribution diagnostics for completed runs and explicit inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from morace.contribution_audit import (
    dry_run_candidate as _dry_run_candidate_impl,
    summarize_check_inputs as _summarize_check_inputs_impl,
)
from morace.contribution_compiled_evaluation import (
    evaluate_compiled_family_variants as _evaluate_compiled_family_variants_impl,
    evaluate_compiled_feature_importance as _evaluate_compiled_feature_importance_impl,
    evaluate_compiled_loro_variants as _evaluate_compiled_loro_variants_impl,
)
from morace.contribution_data import (
    INVENTORY_FIELDS,
    inventory_from_runs,
    filter_inventory,
    candidate_paths as _candidate_paths_impl,
    canonical_key as _canonical_key_impl,
    drop_rule_once as _drop_rule_once_impl,
    family_variants as _family_variants_impl,
    is_p0 as _is_p0_impl,
    load_smiles_sample as _load_smiles_sample_impl,
    load_split_manifest as _load_split_manifest_impl,
    load_split_xy_from_manifest as _load_split_xy_from_manifest_impl,
    read_csv as _read_csv_impl,
    sanitize_feature_matrices as _sanitize_feature_matrices_impl,
    select_pilot_ids as _select_pilot_ids_impl,
    select_scope as _select_scope_impl,
    source_type as _source_type_impl,
    split_paths as _split_paths_impl,
    write_csv as _write_csv_impl,
)
from morace.contribution_evaluation import (
    evaluate_feature_importance_in_process as _evaluate_feature_importance_in_process_impl,
    evaluate_feature_importance_with_external_sklearn as _evaluate_feature_importance_with_external_sklearn_impl,
    evaluate_scores_with_external_sklearn as _evaluate_scores_with_external_sklearn_impl,
    evaluate_variants_in_process as _evaluate_variants_in_process_impl,
    evaluate_variants_with_external_sklearn as _evaluate_variants_with_external_sklearn_impl,
    find_external_sklearn_python as _find_external_sklearn_python_impl,
)
from morace.contribution_metrics import (
    contribution_signal as _contribution_signal_impl,
    feature_alignment_label as _feature_alignment_label_impl,
    fmt_float as _fmt_float_impl,
    mean_std as _mean_std_impl,
    rank_vectors as _rank_vectors_impl,
    truth as _truth_impl,
)
from morace.contribution_orchestration import (
    run_family_ablation as _run_family_ablation_impl,
    run_feature_importance as _run_feature_importance_impl,
    run_loro_smoke as _run_loro_smoke_impl,
)
from morace.contribution_reporting import (
    write_dry_run_summary as _write_dry_run_summary_impl,
    write_family_ablation_summary as _write_family_ablation_summary_impl,
    write_feature_importance_summary as _write_feature_importance_summary_impl,
    write_loro_smoke_summary as _write_loro_smoke_summary_impl,
    write_protocol as _write_protocol_impl,
)

P0_PREFIXES = (
    "PILOT_A_QWEN_D2R_HIV",
    "PILOT_C_DEEPSEEK_D52_BACE",
    "PILOT_C_DEEPSEEK_D52_FREESOLV",
    "PILOT_D_CHEMDFM_B3_HIV",
)


DRY_RUN_FIELDS = [
    "pilot_id",
    "route_or_model",
    "dataset",
    "run_tag",
    "seed",
    "metric",
    "metric_direction",
    "selected_rule_count",
    "compiled_rule_count",
    "rules_file_found",
    "outcome_file_found",
    "split_train_found",
    "split_valid_found",
    "split_test_found",
    "can_build_base_rule_matrix",
    "can_prepare_loro",
    "can_prepare_family_ablation",
    "can_prepare_feature_importance",
    "requires_gpu",
    "requires_llm",
    "requires_new_rules",
    "dry_run_status",
    "issue",
    "recommended_next_action",
]

LORO_FIELDS = [
    "pilot_id",
    "route_or_model",
    "dataset",
    "run_tag",
    "seed",
    "metric",
    "metric_direction",
    "rule_id",
    "canonical_rule",
    "feature_family",
    "root_feature",
    "source_type",
    "base_valid_mean",
    "base_valid_std",
    "base_test_mean",
    "base_test_std",
    "ablated_valid_mean",
    "ablated_valid_std",
    "ablated_test_mean",
    "ablated_test_std",
    "valid_delta",
    "test_delta",
    "contribution_signal",
    "base_reproduction_status",
    "notes",
]

FAMILY_ABLATION_FIELDS = [
    "pilot_id",
    "route_or_model",
    "dataset",
    "run_tag",
    "seed",
    "metric",
    "metric_direction",
    "removed_family",
    "removed_rule_count",
    "remaining_rule_count",
    "base_valid_mean",
    "base_valid_std",
    "base_test_mean",
    "base_test_std",
    "ablated_valid_mean",
    "ablated_valid_std",
    "ablated_test_mean",
    "ablated_test_std",
    "valid_delta",
    "test_delta",
    "family_contribution_signal",
    "base_reproduction_status",
    "notes",
]

FEATURE_IMPORTANCE_FIELDS = [
    "pilot_id",
    "route_or_model",
    "dataset",
    "run_tag",
    "seed",
    "metric",
    "metric_direction",
    "rule_id",
    "canonical_rule",
    "feature_family",
    "root_feature",
    "source_type",
    "importance_mean",
    "importance_std",
    "rank_mean",
    "rank_min",
    "rank_max",
    "filtered_by_sanitizer",
    "loro_valid_delta",
    "loro_contribution_signal",
    "family_valid_delta",
    "family_contribution_signal",
    "alignment_label",
    "base_reproduction_status",
    "notes",
]

BASE_REPRO_FIELDS = [
    "pilot_id",
    "route_or_model",
    "dataset",
    "run_tag",
    "seed",
    "metric",
    "metric_direction",
    "outcome_valid_mean",
    "outcome_test_mean",
    "reproduced_valid_mean",
    "reproduced_valid_std",
    "reproduced_test_mean",
    "reproduced_test_std",
    "valid_abs_diff",
    "test_abs_diff",
    "base_reproduction_status",
    "notes",
]

ISSUE_FIELDS = ["pilot_id", "stage", "severity", "issue", "recommended_action"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    return _read_csv_impl(path)


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    _write_csv_impl(path, rows, fields)


def mean_std(values: List[float]) -> Tuple[float, float]:
    return _mean_std_impl(values)


def fmt_float(value: object) -> str:
    return _fmt_float_impl(value)


def truth(value: object) -> str:
    return _truth_impl(value)


def is_p0(row: Dict[str, str]) -> bool:
    return _is_p0_impl(row, p0_prefixes=P0_PREFIXES)


def select_scope(rows: List[Dict[str, str]], scope: str) -> List[Dict[str, str]]:
    return _select_scope_impl(rows, scope, is_p0_fn=is_p0)


def load_split_manifest(path: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    return _load_split_manifest_impl(path, read_csv_fn=read_csv)


def candidate_paths(project_root: Path, row: Dict[str, str]) -> Dict[str, Path]:
    return _candidate_paths_impl(project_root, row)


def split_paths(project_root: Path, manifest: Dict[Tuple[str, str], Dict[str, str]], dataset: str) -> Dict[str, Path]:
    return _split_paths_impl(project_root, manifest, dataset)


def load_smiles_sample(split_file: Path, smiles_column: str, sample_size: int) -> List[str]:
    return _load_smiles_sample_impl(split_file, smiles_column, sample_size)


def load_split_xy_from_manifest(
    split_file: Path,
    smiles_column: str,
    target_column: str,
) -> Tuple[List[str], List[float]]:
    return _load_split_xy_from_manifest_impl(
        split_file,
        smiles_column,
        target_column,
    )


def sanitize_feature_matrices(
    x_train: List[List[float]],
    x_valid: List[List[float]],
    x_test: List[List[float]],
    feature_names: List[str],
):
    return _sanitize_feature_matrices_impl(
        x_train,
        x_valid,
        x_test,
        feature_names,
    )


def canonical_key(rule: Dict) -> str:
    return _canonical_key_impl(rule)


def source_type(rule: Dict) -> str:
    return _source_type_impl(rule)


def drop_rule_once(rules: List[Dict], target_key: str) -> List[Dict]:
    return _drop_rule_once_impl(
        rules,
        target_key,
        canonical_key_fn=canonical_key,
    )


def family_variants(rules: List[Dict]) -> Dict[str, List[Dict]]:
    return _family_variants_impl(rules)


def select_pilot_ids(rows: List[Dict[str, str]], pilot_ids: str) -> List[Dict[str, str]]:
    return _select_pilot_ids_impl(rows, pilot_ids)


def evaluate_rule_set(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
):
    import numpy as np
    from morace.rule_compiler import build_feature_matrix

    with (project_root / "configs/model_registry.json").open("r", encoding="utf-8") as handle:
        hparams = json.load(handle)
    params_base = dict(hparams[dataset])

    split_data = {}
    for split in ("train", "valid", "test"):
        manifest_row = manifest[(dataset, split)]
        split_path = Path(manifest_row["split_file_path"])
        if not split_path.is_absolute():
            split_path = project_root / split_path
        split_data[split] = load_split_xy_from_manifest(
            split_path,
            manifest_row["smiles_column"],
            manifest_row["target_column"],
        )

    x_train_raw = build_feature_matrix(compiled_rules, split_data["train"][0])
    x_valid_raw = build_feature_matrix(compiled_rules, split_data["valid"][0])
    x_test_raw = build_feature_matrix(compiled_rules, split_data["test"][0])
    feature_names = [rule.name for rule in compiled_rules]
    x_train, x_valid, x_test, kept_features = sanitize_feature_matrices(
        x_train_raw,
        x_valid_raw,
        x_test_raw,
        feature_names,
    )
    y_train = split_data["train"][1]
    y_valid = split_data["valid"][1]
    y_test = split_data["test"][1]

    task_type = "classification" if dataset in {"bbbp", "bace", "clintox", "hiv", "tox21", "sider"} else "regression"
    valid_scores: List[float] = []
    test_scores: List[float] = []
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import mean_squared_error, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError:
        valid_scores, test_scores = evaluate_scores_with_external_sklearn(
            task_type,
            params_base,
            rf_seeds,
            x_train,
            y_train,
            x_valid,
            y_valid,
            x_test,
            y_test,
        )
    else:
        if task_type == "classification":
            scaler = StandardScaler()
            x_train_eval = scaler.fit_transform(x_train)
            x_valid_eval = scaler.transform(x_valid)
            x_test_eval = scaler.transform(x_test)
            for seed in rf_seeds:
                params = dict(params_base)
                params["random_state"] = seed
                clf = RandomForestClassifier(**params)
                clf.fit(x_train_eval, y_train)
                valid_scores.append(float(roc_auc_score(y_valid, clf.predict_proba(x_valid_eval)[:, 1])))
                test_scores.append(float(roc_auc_score(y_test, clf.predict_proba(x_test_eval)[:, 1])))
        else:
            x_train_eval = np.asarray(x_train)
            x_valid_eval = np.asarray(x_valid)
            x_test_eval = np.asarray(x_test)
            for seed in rf_seeds:
                params = dict(params_base)
                params["random_state"] = seed
                reg = RandomForestRegressor(**params)
                reg.fit(x_train_eval, y_train)
                valid_scores.append(float(math.sqrt(mean_squared_error(y_valid, reg.predict(x_valid_eval)))))
                test_scores.append(float(math.sqrt(mean_squared_error(y_test, reg.predict(x_test_eval)))))

    valid_mean, valid_std = mean_std(valid_scores)
    test_mean, test_std = mean_std(test_scores)
    return {
        "valid_scores": valid_scores,
        "test_scores": test_scores,
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "kept_features": kept_features,
    }


def find_external_sklearn_python(current_executable: str) -> str:
    return _find_external_sklearn_python_impl(current_executable)


def evaluate_scores_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Tuple[List[float], List[float]]:
    return _evaluate_scores_with_external_sklearn_impl(
        task_type,
        params_base,
        rf_seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )


def evaluate_variants_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
    variants: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    return _evaluate_variants_with_external_sklearn_impl(
        task_type,
        params_base,
        rf_seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        variants,
    )


def evaluate_variants_in_process(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
    variants: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    return _evaluate_variants_in_process_impl(
        task_type,
        params_base,
        rf_seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        variants,
    )


def evaluate_feature_importance_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Dict[str, object]:
    return _evaluate_feature_importance_with_external_sklearn_impl(
        task_type,
        params_base,
        rf_seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )


def evaluate_feature_importance_in_process(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Dict[str, object]:
    return _evaluate_feature_importance_in_process_impl(
        task_type,
        params_base,
        rf_seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )


def contribution_signal(metric: str, valid_delta: float) -> str:
    return _contribution_signal_impl(metric, valid_delta)


def feature_alignment_label(
    loro_signal: str,
    family_signal: str,
    rank_mean: object,
    kept_feature_count: int,
    filtered: bool,
) -> str:
    return _feature_alignment_label_impl(
        loro_signal,
        family_signal,
        rank_mean,
        kept_feature_count,
        filtered,
    )


def evaluate_compiled_loro_variants(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
):
    return _evaluate_compiled_loro_variants_impl(
        project_root,
        dataset,
        compiled_rules,
        manifest,
        rf_seeds,
        load_split_xy_fn=load_split_xy_from_manifest,
        sanitize_feature_matrices_fn=sanitize_feature_matrices,
        evaluate_variants_external_fn=evaluate_variants_with_external_sklearn,
        evaluate_variants_in_process_fn=evaluate_variants_in_process,
        mean_std_fn=mean_std,
    )


def evaluate_compiled_family_variants(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
):
    return _evaluate_compiled_family_variants_impl(
        project_root,
        dataset,
        compiled_rules,
        manifest,
        rf_seeds,
        load_split_xy_fn=load_split_xy_from_manifest,
        sanitize_feature_matrices_fn=sanitize_feature_matrices,
        evaluate_variants_external_fn=evaluate_variants_with_external_sklearn,
        evaluate_variants_in_process_fn=evaluate_variants_in_process,
        mean_std_fn=mean_std,
    )


def evaluate_compiled_feature_importance(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
):
    return _evaluate_compiled_feature_importance_impl(
        project_root,
        dataset,
        compiled_rules,
        manifest,
        rf_seeds,
        load_split_xy_fn=load_split_xy_from_manifest,
        sanitize_feature_matrices_fn=sanitize_feature_matrices,
        evaluate_feature_importance_external_fn=evaluate_feature_importance_with_external_sklearn,
        evaluate_feature_importance_in_process_fn=evaluate_feature_importance_in_process,
        mean_std_fn=mean_std,
    )


def run_loro_smoke(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    return _run_loro_smoke_impl(
        project_root,
        inventory_rows,
        manifest,
        reproduction_warning_threshold,
        candidate_paths_fn=candidate_paths,
        evaluate_compiled_loro_fn=evaluate_compiled_loro_variants,
        fmt_float_fn=fmt_float,
        canonical_key_fn=canonical_key,
        source_type_fn=source_type,
        contribution_signal_fn=contribution_signal,
    )


def run_family_ablation(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    return _run_family_ablation_impl(
        project_root,
        inventory_rows,
        manifest,
        reproduction_warning_threshold,
        candidate_paths_fn=candidate_paths,
        evaluate_compiled_family_fn=evaluate_compiled_family_variants,
        fmt_float_fn=fmt_float,
        contribution_signal_fn=contribution_signal,
    )


def load_loro_lookup(analysis_dir: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    path = analysis_dir / "loro.csv"
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return lookup
    for row in read_csv(path):
        lookup[(row.get("pilot_id", ""), row.get("canonical_rule", ""))] = row
    return lookup


def load_family_lookup(analysis_dir: Path) -> Dict[Tuple[str, str], Dict[str, str]]:
    path = analysis_dir / "family_ablation.csv"
    lookup: Dict[Tuple[str, str], Dict[str, str]] = {}
    if not path.exists():
        return lookup
    for row in read_csv(path):
        lookup[(row.get("pilot_id", ""), row.get("removed_family", ""))] = row
    return lookup


def rank_vectors(importances: List[List[float]]) -> List[List[int]]:
    return _rank_vectors_impl(importances)


def run_feature_importance(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    reproduction_warning_threshold: float,
    *,
    analysis_dir: Path | None = None,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    return _run_feature_importance_impl(
        project_root,
        inventory_rows,
        manifest,
        reproduction_warning_threshold,
        candidate_paths_fn=candidate_paths,
        evaluate_compiled_feature_importance_fn=evaluate_compiled_feature_importance,
        fmt_float_fn=fmt_float,
        load_loro_lookup_fn=(lambda _root: load_loro_lookup(analysis_dir)) if analysis_dir is not None else load_loro_lookup,
        load_family_lookup_fn=(lambda _root: load_family_lookup(analysis_dir)) if analysis_dir is not None else load_family_lookup,
        rank_vectors_fn=rank_vectors,
        mean_std_fn=mean_std,
        canonical_key_fn=canonical_key,
        source_type_fn=source_type,
        truth_fn=truth,
        feature_alignment_label_fn=feature_alignment_label,
    )


def summarize_check_inputs(
    project_root: Path,
    inventory_rows: List[Dict[str, str]],
    manifest_path: Path,
) -> Tuple[bool, List[str]]:
    return _summarize_check_inputs_impl(project_root, inventory_rows, manifest_path)


def dry_run_candidate(
    project_root: Path,
    row: Dict[str, str],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    matrix_sample_size: int,
) -> Dict[str, object]:
    return _dry_run_candidate_impl(
        project_root,
        row,
        manifest,
        matrix_sample_size,
    )


def write_dry_run_summary(path: Path, rows: List[Dict[str, object]], scope: str, matrix_sample_size: int) -> None:
    _write_dry_run_summary_impl(path, rows, scope, matrix_sample_size)


def write_protocol(path: Path) -> None:
    _write_protocol_impl(path)


def write_loro_smoke_summary(
    path: Path,
    loro_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    run_label: str = "smoke",
    candidate_count: int | None = None,
    project_root: Path | None = None,
) -> None:
    _write_loro_smoke_summary_impl(
        path,
        loro_rows,
        base_rows,
        issue_rows,
        run_label,
        candidate_count,
        project_root,
        read_csv_fn=read_csv,
    )


def write_family_ablation_summary(
    path: Path,
    family_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    candidate_count: int,
    project_root: Path,
) -> None:
    _write_family_ablation_summary_impl(
        path,
        family_rows,
        base_rows,
        issue_rows,
        candidate_count,
        project_root,
        read_csv_fn=read_csv,
    )


def write_feature_importance_summary(
    path: Path,
    feature_rows: List[Dict[str, object]],
    base_rows: List[Dict[str, object]],
    issue_rows: List[Dict[str, object]],
    candidate_count: int,
) -> None:
    _write_feature_importance_summary_impl(
        path,
        feature_rows,
        base_rows,
        issue_rows,
        candidate_count,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Rule contribution diagnostics for completed MoRACE runs.")
    parser.add_argument("--project-root", default=".")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--inventory", help="Explicit contribution inventory CSV.")
    inputs.add_argument("--run-dir", action="append", help="Completed artifact directory or summary.json; repeat for multiple runs.")
    parser.add_argument("--output-dir", help="Analysis directory; existing results for the requested mode are never overwritten.")
    parser.add_argument("--existing-ablation", default="", help="Deprecated compatibility option; ignored.")
    parser.add_argument("--split-manifest", default="data/manifests/split_manifest.csv")
    parser.add_argument("--scope", choices=["p0", "p1", "all"], default="all", help="p0/p1 are legacy inventory compatibility filters.")
    parser.add_argument("--routes", "--route", default="", help="Comma-separated exact route_or_model values.")
    parser.add_argument("--datasets", "--dataset", default="", help="Comma-separated dataset names.")
    parser.add_argument("--run-ids", "--run-id", default="", help="Comma-separated exact run_tag values.")
    parser.add_argument("--knowledge-types", default="", help="Comma-separated branch names when present in inventory.")
    parser.add_argument("--pilot-ids", default="", help="Optional comma-separated candidate IDs (legacy column name).")
    parser.add_argument("--matrix-sample-size", type=int, default=5)
    parser.add_argument("--check-inputs", action="store_true")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--run-loro", action="store_true")
    actions.add_argument("--run-family-ablation", action="store_true")
    actions.add_argument("--run-feature-importance", action="store_true")
    parser.add_argument("--loro-output-prefix", choices=["smoke", "p0"], help="Deprecated compatibility option; filenames now use --output-dir.")
    parser.add_argument("--reproduction-warning-threshold", type=float, default=0.02)
    args = parser.parse_args()
    mode = next((name for name, enabled in (
        ("dry_run", args.dry_run), ("loro", args.run_loro),
        ("family_ablation", args.run_family_ablation), ("feature_importance", args.run_feature_importance)
    ) if enabled), None)
    if not mode and not args.check_inputs:
        print("No action requested. Use --check-inputs, --dry-run, --run-loro, --run-family-ablation or --run-feature-importance.", file=sys.stderr)
        return 2
    if mode and not args.output_dir:
        print("--output-dir is required for contribution results.", file=sys.stderr)
        return 2
    if args.matrix_sample_size < 1 or args.reproduction_warning_threshold < 0:
        print("Matrix sample size must be positive and reproduction warning threshold nonnegative.", file=sys.stderr)
        return 2
    project_root = Path(args.project_root).resolve()
    manifest_path = project_root / args.split_manifest
    try:
        inventory_rows = (read_csv(project_root / args.inventory) if args.inventory else
                          inventory_from_runs(project_root, [Path(value) for value in args.run_dir]))
        selected_rows = filter_inventory(select_scope(inventory_rows, args.scope), routes=args.routes,
            datasets=args.datasets, run_ids=args.run_ids, knowledge_types=args.knowledge_types)
        if args.pilot_ids.strip():
            selected_rows = select_pilot_ids(selected_rows, args.pilot_ids)
        if not selected_rows:
            print("selection_status=empty_selection: no contribution candidates match the input and filters.", file=sys.stderr)
            return 1
        ids = [row.get("pilot_id", "") for row in selected_rows]
        if any(not value for value in ids) or len(ids) != len(set(ids)):
            raise ValueError("Selected inventory candidate IDs must be nonempty and unique.")
        if args.check_inputs:
            ok, issues = summarize_check_inputs(project_root, selected_rows, manifest_path)
            print(f"check_inputs_status={'passed' if ok else 'failed'}")
            print(f"candidate_rows={len(selected_rows)}")
            for issue in issues:
                print(f"- {issue}")
            if not ok or not mode:
                return 0 if ok else 1
        manifest = load_split_manifest(manifest_path)
        output_dir = (project_root / args.output_dir).resolve()
        targets = [output_dir / f"{mode}{suffix}" for suffix in
                   ((".csv", "_summary.md", "_config.json") if mode == "dry_run" else
                    (".csv", "_summary.md", "_config.json", "_base_reproduction.csv", "_issues.csv"))]
        if mode == "dry_run":
            targets.append(output_dir / "protocol.md")
        existing = [str(path) for path in targets if path.exists()]
        if existing:
            raise ValueError("Refusing to overwrite existing contribution results; choose a new --output-dir: " + ", ".join(existing))
        fields = list(dict.fromkeys(INVENTORY_FIELDS + [key for row in selected_rows for key in row]))
        snapshot_rows = [{field: str(row.get(field, "")) for field in fields} for row in selected_rows]
        inventory_path = output_dir / "inventory.csv"
        if inventory_path.exists() and read_csv(inventory_path) != snapshot_rows:
            raise ValueError("Output directory already indexes a different selection; choose a new --output-dir.")
        input_paths = {manifest_path.resolve(), (project_root / "configs/model_registry.json").resolve()}
        for row in selected_rows:
            input_paths.update(path.resolve() for path in candidate_paths(project_root, row).values())
            input_paths.update(path.resolve() for path in split_paths(project_root, manifest, row["dataset"]).values())
        input_files = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
                       for path in sorted(input_paths)}
        for previous_path in output_dir.glob("*_config.json"):
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            if previous.get("input_files") != input_files:
                raise ValueError("Output directory contains diagnostics for different input files; choose a new --output-dir.")
        output_dir.mkdir(parents=True, exist_ok=True)
        if not inventory_path.exists():
            write_csv(inventory_path, snapshot_rows, fields)
        (output_dir / f"{mode}_config.json").write_text(json.dumps({
            "schema_version": "contribution_analysis_v1", "analysis_id": output_dir.name,
            "mode": mode, "project_root": str(project_root), "output_dir": str(output_dir),
            "split_manifest": str(manifest_path.resolve()), "scope": args.scope,
            "input_files": input_files,
            "candidate_ids": ids, "matrix_sample_size": args.matrix_sample_size,
            "reproduction_warning_threshold": args.reproduction_warning_threshold,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"output_dir={output_dir}")
        print(f"candidate_rows={len(selected_rows)}")
        if mode == "dry_run":
            rows = [dry_run_candidate(project_root, row, manifest, args.matrix_sample_size) for row in selected_rows]
            write_csv(output_dir / "dry_run.csv", rows, DRY_RUN_FIELDS)
            write_dry_run_summary(output_dir / "dry_run_summary.md", rows, args.scope, args.matrix_sample_size)
            write_protocol(output_dir / "protocol.md")
            counts = Counter(str(row.get("dry_run_status", "")) for row in rows)
            print(f"passed={counts.get('passed', 0)}")
            print(f"failed={counts.get('failed', 0)}")
            return 0 if counts.get("passed", 0) == len(selected_rows) else 1
        if mode == "loro":
            rows, base_rows, issues = run_loro_smoke(project_root, selected_rows, manifest, args.reproduction_warning_threshold)
            fields = LORO_FIELDS
            write_loro_smoke_summary(output_dir / "loro_summary.md", rows, base_rows, issues,
                                    run_label=output_dir.name, candidate_count=len(selected_rows), project_root=project_root)
        elif mode == "family_ablation":
            rows, base_rows, issues = run_family_ablation(project_root, selected_rows, manifest, args.reproduction_warning_threshold)
            fields = FAMILY_ABLATION_FIELDS
            write_family_ablation_summary(output_dir / "family_ablation_summary.md", rows, base_rows, issues,
                                          candidate_count=len(selected_rows), project_root=project_root)
        else:
            rows, base_rows, issues = run_feature_importance(project_root, selected_rows, manifest,
                args.reproduction_warning_threshold, analysis_dir=output_dir)
            fields = FEATURE_IMPORTANCE_FIELDS
            write_feature_importance_summary(output_dir / "feature_importance_summary.md", rows, base_rows, issues,
                                             candidate_count=len(selected_rows))
        write_csv(output_dir / f"{mode}.csv", rows, fields)
        write_csv(output_dir / f"{mode}_base_reproduction.csv", base_rows, BASE_REPRO_FIELDS)
        write_csv(output_dir / f"{mode}_issues.csv", issues, ISSUE_FIELDS)
        print(f"{mode}_rows={len(rows)}")
        print(f"issues={len(issues)}")
        return 0 if rows and not any(row.get("severity") == "error" for row in issues) else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"contribution_status=failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
