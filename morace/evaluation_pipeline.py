import argparse
import copy
import json
import os
from math import sqrt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from morace.rule_aggregation import (
    aggregate_rule_candidates,
    aggregate_rule_candidates_branch_preserving,
    alias_group_key,
    load_rule_artifact,
    save_rule_artifact,
)
from morace.datasets import canonical_dataset_and_subtask, get_dataset_spec, load_model_hparams, load_split_xy
from morace.evaluation_gates import (
    build_branch_failure_diagnostics as _build_branch_failure_diagnostics_impl,
    evaluate_quality_gate as _evaluate_quality_gate_impl,
    infer_limiting_branch as _infer_limiting_branch_impl,
    literature_underdiverse as _literature_underdiverse_impl,
    merge_count_maps as _merge_count_maps_impl,
    projection_failure_diagnostics as _projection_failure_diagnostics_impl,
    synthesize_soft_pass_eligible as _synthesize_soft_pass_eligible_impl,
)
from morace.evaluation_features import (
    NoValidFeaturesError,
    compute_rule_diversity_metrics as _compute_rule_diversity_metrics_impl,
    effective_source_coverage as _effective_source_coverage_impl,
    promoted_multi_rule_metrics as _promoted_multi_rule_metrics_impl,
    rule_root_feature_key as _rule_root_feature_key_impl,
    rule_source_bucket as _rule_source_bucket_impl,
    sanitize_feature_matrices as _sanitize_feature_matrices_impl,
)
from morace.evaluation_selection import (
    candidate_alias_signature as _candidate_alias_signature_impl,
    choose_finalist_with_coverage_guard as _choose_finalist_with_coverage_guard_impl,
    coverage_qualified_finalist as _coverage_qualified_finalist_impl,
    objective_gap as _objective_gap_impl,
    selection_objective as _selection_objective_impl,
    shortlist_all_finalists as _shortlist_all_finalists_impl,
)
from morace.metric_policy import (
    is_better_by,
    is_within_delta,
    metric_direction,
    normalize_metric_name,
    score_improvement,
)
from morace.model_backend import normalize_model_name
from morace.paths import with_run_tag, resolve_artifact_path
from morace.evaluation_config import EvaluationConfig
from morace.rule_preparation import (
    annotate_second_variant_rules as _annotate_second_variant_rules_impl,
    audit_rule_id as _audit_rule_id_impl,
    audit_rule_summary as _audit_rule_summary_impl,
    dedupe_prepared_rules as _dedupe_prepared_rules_impl,
    direction_relation_from_text as _direction_relation_from_text_impl,
    enforce_direction_consistency as _enforce_direction_consistency_impl,
    filter_compile_safe_rules as _filter_compile_safe_rules_impl,
    learn_threshold_if_needed as _learn_threshold_if_needed_impl,
    normalize_directional_relation as _normalize_directional_relation_impl,
    prepare_rules_for_compilation_with_audit as _prepare_rules_for_compilation_with_audit_impl,
    prepared_retention_metrics as _prepared_retention_metrics_impl,
    semantic_rule_signature as _semantic_rule_signature_impl,
    single_feature_validation_score as _single_feature_validation_score_impl,
    snap_threshold as _snap_threshold_impl,
    threshold_hint_from_text as _threshold_hint_from_text_impl,
)
from morace.rule_compiler import build_feature_matrix, compile_rule, compile_rules, render_rule_module
from morace.rule_verifier import gate_failure_reasons
from morace.experiment_io import build_outcome_payload, compute_common_null_model_scores, write_json


MIN_PROJECTOR_VALID_RATE = 0.20
MAX_HEURISTIC_RULE_RATE = 0.50
LITERATURE_SOFT_PASS_MAX_RAW_RULES = 5
LITERATURE_UNDERDIVERSE_COLLAPSE_RATE = 0.80
SNAPPED_DESCRIPTOR_THRESHOLDS = {
    "MolWt": [150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 500.0],
    "MolLogP": [1.0, 1.5, 2.0, 3.0, 4.0, 5.0],
    "TPSA": [40.0, 60.0, 90.0, 120.0],
}


def _mean_score(values: List[float] | Tuple[float, ...] | None) -> float | None:
    if not values:
        return None
    return float(np.mean(values))

def literature_underdiverse(projection_metrics: Dict) -> bool:
    return _literature_underdiverse_impl(projection_metrics)

def synthesize_soft_pass_eligible(projection_metrics: Dict) -> bool:
    return _synthesize_soft_pass_eligible_impl(
        projection_metrics,
        max_raw_rules=LITERATURE_SOFT_PASS_MAX_RAW_RULES,
    )

def filter_compile_safe_rules(rules: List[Dict]) -> List[Dict]:
    return _filter_compile_safe_rules_impl(rules, compile_rule_fn=compile_rule)

def semantic_rule_signature(rule: Dict) -> Tuple[str, str, str, str, str]:
    return _semantic_rule_signature_impl(rule)

def dedupe_prepared_rules(rules: List[Dict]) -> List[Dict]:
    return _dedupe_prepared_rules_impl(
        rules,
        semantic_rule_signature_fn=semantic_rule_signature,
    )

def _audit_rule_id(rule: Dict) -> str:
    return _audit_rule_id_impl(
        rule,
        semantic_rule_signature_fn=semantic_rule_signature,
    )

def _audit_rule_summary(rule: Dict) -> Dict:
    return _audit_rule_summary_impl(
        rule,
        audit_rule_id_fn=_audit_rule_id,
        alias_group_key_fn=alias_group_key,
        root_feature_key_fn=_rule_root_feature_key,
        semantic_rule_signature_fn=semantic_rule_signature,
    )

def _direction_relation_from_text(text: str) -> str:
    return _direction_relation_from_text_impl(text)

def normalize_directional_relation(rule: Dict) -> Dict:
    return _normalize_directional_relation_impl(
        rule,
        direction_relation_from_text_fn=_direction_relation_from_text,
    )

def enforce_direction_consistency(rule: Dict) -> Dict:
    return _enforce_direction_consistency_impl(
        rule,
        normalize_directional_relation_fn=normalize_directional_relation,
        direction_relation_from_text_fn=_direction_relation_from_text,
    )

def _merge_count_maps(*maps: Dict) -> Dict[str, int]:
    return _merge_count_maps_impl(*maps)

def _projection_failure_diagnostics(metrics: Dict) -> Dict:
    return _projection_failure_diagnostics_impl(metrics)

def snap_threshold(rule: Dict, threshold: Dict | None) -> Dict | None:
    return _snap_threshold_impl(
        rule,
        threshold,
        snapped_descriptor_thresholds=SNAPPED_DESCRIPTOR_THRESHOLDS,
    )

def _single_feature_validation_score(rule: Dict, smiles: List[str], y: List[float], task_type: str) -> float | None:
    return _single_feature_validation_score_impl(
        rule,
        smiles,
        y,
        task_type,
        numpy_module=np,
        compile_rule_fn=compile_rule,
        build_feature_matrix_fn=build_feature_matrix,
        roc_auc_score_fn=roc_auc_score,
    )

def _threshold_hint_from_text(rule: Dict) -> bool:
    return _threshold_hint_from_text_impl(rule)

def _annotate_second_variant_rules(rules: List[Dict], second_variant_rule_ids: set[str]) -> List[Dict]:
    return _annotate_second_variant_rules_impl(
        rules,
        second_variant_rule_ids,
        audit_rule_id_fn=_audit_rule_id,
    )

def _prepared_retention_metrics(
    selected_rules: List[Dict],
    prepared_rules: List[Dict],
    second_variant_rule_ids: set[str],
) -> Dict:
    return _prepared_retention_metrics_impl(
        selected_rules,
        prepared_rules,
        second_variant_rule_ids,
        audit_rule_id_fn=_audit_rule_id,
    )

def sanitize_feature_matrices(
    X_train: List[List[float]],
    X_valid: List[List[float]],
    X_test: List[List[float]],
    feature_names: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Dict]:
    return _sanitize_feature_matrices_impl(
        X_train,
        X_valid,
        X_test,
        feature_names,
        numpy_module=np,
        pandas_module=pd,
    )

def _rule_source_bucket(rule: Dict) -> str:
    return _rule_source_bucket_impl(rule)

def _rule_root_feature_key(rule: Dict) -> str:
    return _rule_root_feature_key_impl(rule)

def _effective_source_coverage(source_breakdown: Dict[str, int]) -> Dict[str, int]:
    return _effective_source_coverage_impl(source_breakdown)

def compute_rule_diversity_metrics(rules: List[Dict]) -> Dict:
    return _compute_rule_diversity_metrics_impl(
        rules,
        rule_source_bucket_fn=_rule_source_bucket,
        rule_root_feature_key_fn=_rule_root_feature_key,
        effective_source_coverage_fn=_effective_source_coverage,
    )

def _promoted_multi_rule_metrics(rules: List[Dict]) -> Dict:
    return _promoted_multi_rule_metrics_impl(
        rules,
        rule_source_bucket_fn=_rule_source_bucket,
        rule_root_feature_key_fn=_rule_root_feature_key,
    )

def evaluate_classification(
    dataset: str,
    subtask: str,
    X_train: np.ndarray,
    y_train: List[float],
    X_valid: np.ndarray,
    y_valid: List[float],
    X_test: np.ndarray,
    y_test: List[float],
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    hparams = load_model_hparams()
    best_params = hparams[dataset][subtask] if dataset in {"tox21", "sider"} else hparams[dataset]

    test_scores = []
    valid_scores = []
    for seed in [0, 1, 2, 3, 4]:
        params = dict(best_params)
        params["random_state"] = seed
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        valid_scores.append(roc_auc_score(y_valid, clf.predict_proba(X_valid)[:, 1]))
        test_scores.append(roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))
    return valid_scores, test_scores

def evaluate_regression_rf(
    dataset: str,
    X_train: np.ndarray,
    y_train: List[float],
    X_valid: np.ndarray,
    y_valid: List[float],
    X_test: np.ndarray,
    y_test: List[float],
):
    hparams = load_model_hparams()
    best_params = hparams[dataset]
    valid_scores = []
    test_scores = []
    for seed in [0, 1, 2, 3, 4]:
        params = dict(best_params)
        params["random_state"] = seed
        clf = RandomForestRegressor(**params)
        clf.fit(X_train, y_train)
        valid_scores.append(sqrt(mean_squared_error(y_valid, clf.predict(X_valid))))
        test_scores.append(sqrt(mean_squared_error(y_test, clf.predict(X_test))))
    return valid_scores, test_scores

def evaluate_qm9(
    X_train: np.ndarray,
    y_train: List[float],
    X_valid: np.ndarray,
    y_valid: List[float],
    X_test: np.ndarray,
    y_test: List[float],
):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    valid_scores = []
    test_scores = []
    for seed in [0, 1, 2, 3, 4]:
        clf = RandomForestRegressor(random_state=seed)
        clf.fit(X_train, y_train)
        valid_scores.append(mean_absolute_error(y_valid, clf.predict(X_valid)))
        test_scores.append(mean_absolute_error(y_test, clf.predict(X_test)))
    return valid_scores, test_scores

def compute_null_model_scores(spec, y_train, y_valid, y_test):
    return compute_common_null_model_scores(spec.task_type, spec.dataset, y_train, y_valid, y_test)

def _candidate_alias_signature(candidate: Dict) -> Tuple[str, ...]:
    return _candidate_alias_signature_impl(candidate, alias_group_key_fn=alias_group_key)

def _evaluate_valid_only(
    dataset: str,
    subtask: str,
    spec,
    X_train: np.ndarray,
    y_train: List[float],
    X_valid: np.ndarray,
    y_valid: List[float],
) -> List[float]:
    if spec.task_type == "classification":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        hparams = load_model_hparams()
        best_params = hparams[dataset][subtask] if dataset in {"tox21", "sider"} else hparams[dataset]
        scores = []
        for seed in [0, 1, 2, 3, 4]:
            params = dict(best_params)
            params["random_state"] = seed
            clf = RandomForestClassifier(**params)
            clf.fit(X_train_scaled, y_train)
            scores.append(roc_auc_score(y_valid, clf.predict_proba(X_valid_scaled)[:, 1]))
        return scores

    if dataset == "qm9":
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        scores = []
        for seed in [0, 1, 2, 3, 4]:
            clf = RandomForestRegressor(random_state=seed)
            clf.fit(X_train_scaled, y_train)
            scores.append(mean_absolute_error(y_valid, clf.predict(X_valid_scaled)))
        return scores

    hparams = load_model_hparams()
    best_params = hparams[dataset]
    scores = []
    for seed in [0, 1, 2, 3, 4]:
        params = dict(best_params)
        params["random_state"] = seed
        clf = RandomForestRegressor(**params)
        clf.fit(X_train, y_train)
        scores.append(sqrt(mean_squared_error(y_valid, clf.predict(X_valid))))
    return scores

def _selection_metric_metadata(spec, dataset: str) -> Tuple[str, bool]:
    if spec.task_type == "classification":
        return "auc", True
    if dataset == "qm9":
        return "mae", False
    return "rmse", False

def _selection_objective(report: Dict) -> float:
    return _selection_objective_impl(report)

def _objective_gap(best_report: Dict, other_report: Dict) -> float:
    return _objective_gap_impl(
        best_report,
        other_report,
        selection_objective_fn=_selection_objective,
    )

def _collect_second_variant_rule_ids_for_rules(rules: List[Dict]) -> set[str]:
    root_counts: Dict[str, int] = {}
    for rule in rules:
        root_key = _rule_root_feature_key(rule)
        if root_key:
            root_counts[root_key] = root_counts.get(root_key, 0) + 1
    return {
        _audit_rule_id(rule)
        for rule in rules
        if root_counts.get(_rule_root_feature_key(rule), 0) > 1
    }

def _surviving_compiled_rules(compiled_rules, kept_features: List[str]):
    kept_set = set(kept_features)
    return [item for item in compiled_rules if item.name in kept_set]

def _winner_literature_only_rules(surviving_rules: List[Dict]) -> List[Dict]:
    return [copy.deepcopy(rule) for rule in surviving_rules if _rule_source_bucket(rule) == "literature"]

def _update_combined_metrics_with_finalist(
    branch_metrics: Dict,
    chosen_candidate: Dict,
    chosen_report: Dict,
    finalist_decision: Dict,
) -> Dict:
    updated = copy.deepcopy(branch_metrics)
    combined = dict(updated.get("combined", {}) or {})
    if not combined:
        return updated
    static_winner = str(combined.get("candidate_set_winner", "") or "")
    chosen_name = str(chosen_candidate.get("name", "") or "")
    combined["static_candidate_set_winner"] = static_winner
    combined["candidate_set_winner"] = chosen_name or static_winner
    combined["winner_template_type"] = str(chosen_candidate.get("template_type", "") or "")
    combined["selected_root_feature_count"] = int(chosen_report.get("root_feature_count", 0) or 0)
    combined["selected_source_breakdown"] = dict(chosen_report.get("source_breakdown", {}) or {})
    combined["selected_family_breakdown"] = dict(chosen_report.get("family_breakdown", {}) or {})
    combined["selected_literature_count"] = int(combined["selected_source_breakdown"].get("literature", 0))
    combined["selected_inference_count"] = int(combined["selected_source_breakdown"].get("data", 0))
    combined["selected_multi_count"] = int(combined["selected_source_breakdown"].get("multi", 0))
    combined["effective_literature_coverage"] = int(chosen_report.get("effective_literature_coverage", 0) or 0)
    combined["effective_data_coverage"] = int(chosen_report.get("effective_data_coverage", 0) or 0)
    combined["family_diversity"] = int(chosen_report.get("family_diversity", 0) or 0)
    combined["selected_rule_count"] = int(chosen_candidate.get("selected_rule_count", 0) or 0)
    combined["candidate_set_winner_explanation"] = {
        "winner_template_type": str(chosen_candidate.get("template_type", "") or ""),
        "winner_decided_by": str(finalist_decision.get("decision", "") or ""),
        "static_candidate_set_winner": static_winner,
        "coverage_guard_applied": bool(finalist_decision.get("coverage_guard_applied", False)),
        "best_valid_name": str(finalist_decision.get("best_valid_name", "") or ""),
        "coverage_candidate_name": str(finalist_decision.get("coverage_candidate_name", "") or ""),
        "valid_gap_to_best": float(finalist_decision.get("valid_gap_to_best", 0.0) or 0.0),
        "selection_metric": str(chosen_report.get("selection_metric", "") or ""),
        "higher_is_better": bool(chosen_report.get("higher_is_better", True)),
    }
    selection_trace = dict(combined.get("selection_trace", {}) or {})
    selection_trace["selected_rules"] = list(chosen_candidate.get("rules", []) or [])
    selection_trace["selection_audit"] = dict(chosen_candidate.get("selection_audit", {}) or {"dropped": []})
    combined["selection_trace"] = selection_trace
    updated["combined"] = combined
    return updated

def _plus_one_alias_candidate_sort_key(rule: Dict, selected_rules: List[Dict]) -> tuple:
    selected_roots = {_rule_root_feature_key(item) for item in selected_rules if _rule_root_feature_key(item)}
    selected_aliases = {alias_group_key(item) for item in selected_rules if alias_group_key(item)}
    root_key = _rule_root_feature_key(rule)
    alias_key = alias_group_key(rule)
    same_root_alias = root_key in selected_roots and alias_key not in selected_aliases
    new_root = root_key not in selected_roots
    if same_root_alias:
        category = 0
    elif new_root:
        category = 1
    else:
        category = 2
    source_bucket = _rule_source_bucket(rule)
    source_priority = {"data": 0, "multi": 1, "literature": 2}.get(source_bucket, 3)
    return (
        category,
        source_priority,
        -int(rule.get("support_count", 0) or 0),
        float(rule.get("p_value", 1.0) or 1.0),
        -abs(float(rule.get("effect_size", 0.0) or 0.0)),
        str(rule.get("canonical_key", "")),
    )

def choose_plus_one_alias_candidate(selected_rules: List[Dict], candidate_pool: List[Dict]) -> Dict:
    selected_keys = {str(rule.get("canonical_key", "")).strip() for rule in selected_rules if str(rule.get("canonical_key", "")).strip()}
    selected_roots = {_rule_root_feature_key(rule) for rule in selected_rules if _rule_root_feature_key(rule)}
    selected_aliases = {alias_group_key(rule) for rule in selected_rules if alias_group_key(rule)}
    compile_safe_candidates = []
    for rule in candidate_pool:
        canonical = str(rule.get("canonical_key", "")).strip()
        if canonical and canonical in selected_keys:
            continue
        if compile_rule(rule) is None:
            continue
        compile_safe_candidates.append(rule)
    same_root_alias_candidates = [
        rule
        for rule in compile_safe_candidates
        if _rule_root_feature_key(rule) in selected_roots
        and alias_group_key(rule)
        and alias_group_key(rule) not in selected_aliases
    ]
    new_root_candidates = [
        rule
        for rule in compile_safe_candidates
        if _rule_root_feature_key(rule) and _rule_root_feature_key(rule) not in selected_roots
    ]
    if same_root_alias_candidates:
        ranked = sorted(same_root_alias_candidates, key=lambda item: _plus_one_alias_candidate_sort_key(item, selected_rules))
        chosen = copy.deepcopy(ranked[0])
        return {
            "status": "ok",
            "slot_type": "same_root_alias",
            "candidate": chosen,
            "candidate_canonical_key": str(chosen.get("canonical_key", "")).strip(),
            "candidate_root_feature_key": _rule_root_feature_key(chosen),
            "candidate_alias_group_key": alias_group_key(chosen),
            "candidate_source": _rule_source_bucket(chosen),
        }
    if new_root_candidates:
        ranked = sorted(new_root_candidates, key=lambda item: _plus_one_alias_candidate_sort_key(item, selected_rules))
        chosen = copy.deepcopy(ranked[0])
        return {
            "status": "ok",
            "slot_type": "new_root_fallback",
            "candidate": chosen,
            "candidate_canonical_key": str(chosen.get("canonical_key", "")).strip(),
            "candidate_root_feature_key": _rule_root_feature_key(chosen),
            "candidate_alias_group_key": alias_group_key(chosen),
            "candidate_source": _rule_source_bucket(chosen),
        }
    return {
        "status": "skipped",
        "reason": "no_compile_safe_alias_or_new_root_candidate",
    }

def _remove_rule_once_by_canonical_key(rules: List[Dict], target_canonical_key: str) -> List[Dict]:
    updated: List[Dict] = []
    removed = False
    for rule in rules:
        if not removed and str(rule.get("canonical_key", "")).strip() == target_canonical_key:
            removed = True
            continue
        updated.append(copy.deepcopy(rule))
    return updated

def _stage_rule_metrics(rules: List[Dict]) -> Dict:
    metrics = compute_rule_diversity_metrics(rules)
    metrics["rule_count"] = len(rules)
    return metrics

def _compiled_stage_metrics(compiled_rules) -> Dict:
    compiled_rule_defs = [item.rule for item in compiled_rules]
    metrics = _stage_rule_metrics(compiled_rule_defs)
    metrics["compiled_feature_names"] = [item.name for item in compiled_rules]
    return metrics

def _outcome_task_and_metric(dataset: str, subtask: str) -> tuple[str | None, str | None]:
    spec = get_dataset_spec(dataset, subtask=subtask)
    task_type = str(getattr(spec, "task_type", "") or "") or None
    if task_type == "classification":
        metric_name = "roc_auc"
    elif str(dataset).lower() == "qm9":
        metric_name = "mae"
    elif task_type == "regression":
        metric_name = "rmse"
    else:
        metric_name = None
    return task_type, metric_name

def _rule_source_set(rules: List[Dict] | None) -> set[str]:
    return {str(rule.get("source", "")) for rule in (rules or []) if rule.get("source")}

def _all_candidate_diversity_bonus(candidate_metadata: Dict, inference_metadata: Dict) -> bool:
    candidate_features = set(candidate_metadata.get("compiled_feature_names", []) or [])
    inference_features = set(inference_metadata.get("compiled_feature_names", []) or [])
    candidate_rules = candidate_metadata.get("rules", []) or []
    inference_rules = inference_metadata.get("rules", []) or []
    candidate_sources = _rule_source_set(candidate_rules)
    inference_sources = _rule_source_set(inference_rules)
    return bool(
        len(candidate_features) > len(inference_features)
        or int(candidate_metadata.get("compiled_rule_count", 0)) > int(inference_metadata.get("compiled_rule_count", 0))
        or ("literature" in candidate_sources and "literature" not in inference_sources)
        or len(candidate_sources) > len(inference_sources)
    )

def _candidate_underdiverse_all(metadata: Dict) -> bool:
    combined_metrics = metadata.get("branch_metrics", {}).get("combined", {})
    if "underdiverse_all" in combined_metrics:
        return bool(combined_metrics.get("underdiverse_all"))

    feature_metrics = metadata.get("feature_metrics", {})
    family_breakdown = feature_metrics.get("family_breakdown", {}) or {}
    source_breakdown = feature_metrics.get("source_breakdown", {}) or {}
    effective_source_coverage = _effective_source_coverage(source_breakdown)
    effective_literature_coverage = int(
        feature_metrics.get("effective_literature_coverage", effective_source_coverage["literature"])
    )
    effective_data_coverage = int(
        feature_metrics.get("effective_data_coverage", effective_source_coverage["data"])
    )
    family_diversity = int(feature_metrics.get("family_diversity", 0))
    selected_rule_count = int(metadata.get("selected_rule_count", 0))
    descriptor_only = bool(selected_rule_count) and family_breakdown.get("descriptor", 0) == selected_rule_count
    return bool(
        selected_rule_count < 3
        or effective_literature_coverage < 1
        or effective_data_coverage < 1
        or family_diversity < 2
        or descriptor_only
    )

class EvaluationPipeline:
    """Evaluate one explicit protocol without changing process-global configuration."""

    def __init__(self, config: EvaluationConfig | None = None):
        self._config = config if config is not None else EvaluationConfig()
        if not isinstance(self._config, EvaluationConfig):
            raise TypeError("config must be an EvaluationConfig")

    @property
    def config(self) -> EvaluationConfig:
        return self._config

    def _scoped_path(self, base: str, run_tag: str) -> str:
        # An explicit empty scope stays empty even if another caller sets an env tag.
        return with_run_tag(base, run_tag, inherit_environment=False)

    # Stateless helpers retain the module-level public API.
    _all_candidate_diversity_bonus = staticmethod(_all_candidate_diversity_bonus)
    _annotate_second_variant_rules = staticmethod(_annotate_second_variant_rules)
    _audit_rule_id = staticmethod(_audit_rule_id)
    _audit_rule_summary = staticmethod(_audit_rule_summary)
    _candidate_alias_signature = staticmethod(_candidate_alias_signature)
    _candidate_underdiverse_all = staticmethod(_candidate_underdiverse_all)
    _collect_second_variant_rule_ids_for_rules = staticmethod(_collect_second_variant_rule_ids_for_rules)
    _compiled_stage_metrics = staticmethod(_compiled_stage_metrics)
    _direction_relation_from_text = staticmethod(_direction_relation_from_text)
    _effective_source_coverage = staticmethod(_effective_source_coverage)
    _evaluate_valid_only = staticmethod(_evaluate_valid_only)
    _mean_score = staticmethod(_mean_score)
    _merge_count_maps = staticmethod(_merge_count_maps)
    _objective_gap = staticmethod(_objective_gap)
    _outcome_task_and_metric = staticmethod(_outcome_task_and_metric)
    _plus_one_alias_candidate_sort_key = staticmethod(_plus_one_alias_candidate_sort_key)
    _prepared_retention_metrics = staticmethod(_prepared_retention_metrics)
    _projection_failure_diagnostics = staticmethod(_projection_failure_diagnostics)
    _promoted_multi_rule_metrics = staticmethod(_promoted_multi_rule_metrics)
    _remove_rule_once_by_canonical_key = staticmethod(_remove_rule_once_by_canonical_key)
    _rule_root_feature_key = staticmethod(_rule_root_feature_key)
    _rule_source_bucket = staticmethod(_rule_source_bucket)
    _rule_source_set = staticmethod(_rule_source_set)
    _selection_metric_metadata = staticmethod(_selection_metric_metadata)
    _selection_objective = staticmethod(_selection_objective)
    _single_feature_validation_score = staticmethod(_single_feature_validation_score)
    _stage_rule_metrics = staticmethod(_stage_rule_metrics)
    _surviving_compiled_rules = staticmethod(_surviving_compiled_rules)
    _threshold_hint_from_text = staticmethod(_threshold_hint_from_text)
    _update_combined_metrics_with_finalist = staticmethod(_update_combined_metrics_with_finalist)
    _winner_literature_only_rules = staticmethod(_winner_literature_only_rules)
    aggregate_rule_candidates = staticmethod(aggregate_rule_candidates)
    aggregate_rule_candidates_branch_preserving = staticmethod(aggregate_rule_candidates_branch_preserving)
    build_feature_matrix = staticmethod(build_feature_matrix)
    build_outcome_payload = staticmethod(build_outcome_payload)
    canonical_dataset_and_subtask = staticmethod(canonical_dataset_and_subtask)
    choose_plus_one_alias_candidate = staticmethod(choose_plus_one_alias_candidate)
    compile_rule = staticmethod(compile_rule)
    compile_rules = staticmethod(compile_rules)
    compute_null_model_scores = staticmethod(compute_null_model_scores)
    compute_rule_diversity_metrics = staticmethod(compute_rule_diversity_metrics)
    dedupe_prepared_rules = staticmethod(dedupe_prepared_rules)
    enforce_direction_consistency = staticmethod(enforce_direction_consistency)
    evaluate_classification = staticmethod(evaluate_classification)
    evaluate_qm9 = staticmethod(evaluate_qm9)
    evaluate_regression_rf = staticmethod(evaluate_regression_rf)
    filter_compile_safe_rules = staticmethod(filter_compile_safe_rules)
    literature_underdiverse = staticmethod(literature_underdiverse)
    load_model_hparams = staticmethod(load_model_hparams)
    load_rule_artifact = staticmethod(load_rule_artifact)
    normalize_directional_relation = staticmethod(normalize_directional_relation)
    render_rule_module = staticmethod(render_rule_module)
    sanitize_feature_matrices = staticmethod(sanitize_feature_matrices)
    save_rule_artifact = staticmethod(save_rule_artifact)
    semantic_rule_signature = staticmethod(semantic_rule_signature)
    snap_threshold = staticmethod(snap_threshold)
    synthesize_soft_pass_eligible = staticmethod(synthesize_soft_pass_eligible)
    write_json = staticmethod(write_json)

    def load_split_xy(self, dataset: str, subtask: str = "", split: str = "train"):
        """Load from this run's captured root, never from a later environment change."""
        return load_split_xy(dataset, subtask=subtask, split=split,
                             input_folder=self.config.dataset_root, inherit_environment=False)


    def model_tag(self) -> str:
        return normalize_model_name(self.config.model)


    def subfolder(self, knowledge_type: str | None = None) -> str:
        knowledge_type = knowledge_type or self.config.knowledge_type
        return f"sample_{self.config.list_num}" if knowledge_type in {"inference", "all"} else ""


    def resolve_existing_path(self, primary_path: str, legacy_path: str) -> str:
        return resolve_artifact_path(primary_path, legacy_path, run_tag=self.config.run_tag, inherit_environment=False)


    def load_runtime_snapshot(self) -> Dict:
        if not self.config.run_tag:
            return {}
        path = os.path.join("artifacts", self.config.run_tag, "run_config.json")
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


    def synthesize_artifact_path(self, dataset: str, subtask: str) -> str:
        base = self._scoped_path("synthesize_model_response", self.config.run_tag)
        legacy_base = "synthesize_model_response"
        base = os.path.join(base, self.model_tag(), dataset)
        legacy_base = os.path.join(legacy_base, self.model_tag(), dataset)
        if subtask:
            name = f"{self.model_tag()}_{dataset}_{subtask}_pk_response.json"
        else:
            name = f"{self.model_tag()}_{dataset}_pk_response.json"
        return self.resolve_existing_path(os.path.join(base, name), os.path.join(legacy_base, name))


    def inference_artifact_path(self, dataset: str, subtask: str) -> str:
        base = self._scoped_path("summarized_inference_rules", self.config.run_tag)
        legacy_base = "summarized_inference_rules"
        base = os.path.join(base, self.model_tag(), dataset)
        legacy_base = os.path.join(legacy_base, self.model_tag(), dataset)
        if subtask:
            name = f"{self.model_tag()}_{dataset}_{subtask}_dk_response_sample_{self.config.list_num}_summarized_rules.json"
        else:
            name = f"{self.model_tag()}_{dataset}_dk_response_sample_{self.config.list_num}_summarized_rules.json"
        return self.resolve_existing_path(os.path.join(base, name), os.path.join(legacy_base, name))


    def code_output_paths_for(self, dataset: str, subtask: str, knowledge_type: str) -> Tuple[str, str]:
        folder = self._scoped_path(self.config.code_gen_folder, self.config.run_tag)
        folder = os.path.join(folder, self.model_tag(), dataset, knowledge_type, self.subfolder(knowledge_type))
        os.makedirs(folder, exist_ok=True)
        suffix = f"_{subtask}" if subtask else ""
        filename = f"{self.model_tag()}_{dataset}{suffix}_{knowledge_type}_rules.txt"
        meta_name = f"{self.model_tag()}_{dataset}{suffix}_{knowledge_type}_rules.json"
        return os.path.join(folder, filename), os.path.join(folder, meta_name)


    def outcome_output_path_for(self, dataset: str, subtask: str, knowledge_type: str, metadata_path: str) -> str:
        if not subtask:
            return os.path.join(os.path.dirname(metadata_path), "outcome.json")
        filename = f"{self.model_tag()}_{dataset}_{subtask}_{knowledge_type}_outcome.json"
        return os.path.join(os.path.dirname(metadata_path), filename)


    def code_output_paths(self, dataset: str, subtask: str) -> Tuple[str, str]:
        return self.code_output_paths_for(dataset, subtask, self.config.knowledge_type)


    def shrinkage_audit_path(self, dataset: str, subtask: str) -> str:
        code_path, _ = self.code_output_paths(dataset, subtask)
        return code_path.replace("_rules.txt", "_shrinkage_audit.json")


    def result_output_dir(self, dataset: str) -> str:
        path = self._scoped_path(self.config.output_dir, self.config.run_tag)
        path = os.path.join(path, self.model_tag(), dataset, self.config.knowledge_type, self.subfolder(self.config.knowledge_type))
        os.makedirs(path, exist_ok=True)
        return path


    def load_rules(self, dataset: str, subtask: str) -> Tuple[List[Dict], Dict]:
        if self.config.knowledge_type == "synthesize":
            synth = self.load_rule_artifact(self.synthesize_artifact_path(dataset, subtask))
            selected, metrics = self.aggregate_rule_candidates(
                synth.get("canonical_rules") or synth.get("rules", []),
                dataset=dataset,
                subtask=subtask,
                input_folder=self.config.dataset_root,
                target_rule_count=self.config.target_rule_count,
                source_balance=False,
                selector_profile="synthesize_baseline",
                multi_consensus_promotion_enable=self.config.multi_consensus_promotion_enable,
            )
            return selected, {"synthesize": metrics, "synthesize_projection": synth.get("metrics", {})}

        if self.config.knowledge_type == "inference":
            inference = self.load_rule_artifact(self.inference_artifact_path(dataset, subtask))
            return inference.get("selected_rules", []), {
                "inference": inference.get("metrics", {}),
                "inference_projection": inference.get("projection_metrics", {}),
            }

        synth = self.load_rule_artifact(self.synthesize_artifact_path(dataset, subtask))
        inference = self.load_rule_artifact(self.inference_artifact_path(dataset, subtask))
        synth_projection = synth.get("metrics", {})
        synth_rules = synth.get("canonical_rules") or synth.get("rules", [])
        inference_rules = inference.get("selected_rules", [])
        combined_input_mode = "full_all"
        if int(synth_projection.get("raw_rule_count", 0)) == 0 or self.literature_underdiverse(synth_projection):
            combined_rules = inference_rules
            combined_input_mode = "inference_only_fallback"
        elif (
            float(synth_projection.get("strict_projector_valid_rate", synth_projection.get("projector_valid_rate", 0.0)))
            < MIN_PROJECTOR_VALID_RATE
            and self.synthesize_soft_pass_eligible(synth_projection)
        ):
            combined_rules = self.filter_compile_safe_rules(synth_rules) + inference_rules
            combined_input_mode = "inference_dominant_all"
        else:
            combined_rules = synth_rules + inference_rules
        selected, metrics = self.aggregate_rule_candidates_branch_preserving(
            combined_rules,
            dataset=dataset,
            subtask=subtask,
            input_folder=self.config.dataset_root,
            target_rule_count=self.config.target_rule_count,
            multi_consensus_promotion_enable=self.config.multi_consensus_promotion_enable,
            m7_lit_preserving_swap_enable=self.config.m7_lit_preserving_swap_enable,
        )
        metrics["combined_input_mode"] = combined_input_mode
        return selected, {
            "synthesize": synth.get("metrics", {}),
            "synthesize_projection": synth.get("metrics", {}),
            "inference": inference.get("metrics", {}),
            "inference_projection": inference.get("projection_metrics", {}),
            "combined": metrics,
        }


    def _infer_limiting_branch(self, branch_metrics: Dict, gate_reason: str) -> str:
        return _infer_limiting_branch_impl(
            branch_metrics,
            gate_reason,
            knowledge_type=self.config.knowledge_type,
        )


    def build_branch_failure_diagnostics(self, branch_metrics: Dict, gate_mode: str, gate_reason: str) -> Dict:
        return _build_branch_failure_diagnostics_impl(
            branch_metrics,
            gate_mode,
            gate_reason,
            projection_failure_diagnostics_fn=self._projection_failure_diagnostics,
            merge_count_maps_fn=self._merge_count_maps,
            infer_limiting_branch_fn=self._infer_limiting_branch,
        )


    def evaluate_quality_gate(self, branch_metrics: Dict) -> Tuple[str, str, str]:
        return _evaluate_quality_gate_impl(
            branch_metrics,
            knowledge_type=self.config.knowledge_type,
            min_projector_valid_rate=MIN_PROJECTOR_VALID_RATE,
            max_heuristic_rule_rate=MAX_HEURISTIC_RULE_RATE,
            synthesize_soft_pass_eligible_fn=self.synthesize_soft_pass_eligible,
            literature_underdiverse_fn=self.literature_underdiverse,
        )


    def learn_threshold_if_needed(self,
        rule: Dict,
        smiles_train: List[str],
        y_train: List[float],
        smiles_valid: List[str],
        y_valid: List[float],
        task_type: str,
    ) -> Dict:
        return _learn_threshold_if_needed_impl(
            rule,
            smiles_train,
            y_train,
            smiles_valid,
            y_valid,
            task_type,
            numpy_module=np,
            threshold_keep_min_gain=self.config.threshold_keep_min_gain,
            compile_rule_fn=self.compile_rule,
            build_feature_matrix_fn=self.build_feature_matrix,
            roc_auc_score_fn=roc_auc_score,
            normalize_directional_relation_fn=self.normalize_directional_relation,
            enforce_direction_consistency_fn=self.enforce_direction_consistency,
            snap_threshold_fn=self.snap_threshold,
            single_feature_validation_score_fn=self._single_feature_validation_score,
            threshold_hint_from_text_fn=self._threshold_hint_from_text,
        )


    def prepare_rules_for_compilation(self, rules: List[Dict], dataset: str, subtask: str) -> List[Dict]:
        prepared, _ = self.prepare_rules_for_compilation_with_audit(rules, dataset, subtask)
        return prepared


    def _collect_second_variant_rule_ids(self, branch_metrics: Dict) -> set[str]:
        active_metrics = self._active_branch_metrics(branch_metrics)
        selection_trace = active_metrics.get("selection_trace", {}) or {}
        selection_audit = selection_trace.get("selection_audit", {}) or {}
        second_variant_rule_ids: set[str] = set()
        for item in selection_audit.get("selected_same_root_alias_pairs", []) or []:
            new_key = str(item.get("new_canonical_key", "")).strip()
            if new_key:
                second_variant_rule_ids.add(new_key)
            for existing_key in item.get("existing_canonical_keys", []) or []:
                existing_key = str(existing_key).strip()
                if existing_key:
                    second_variant_rule_ids.add(existing_key)
        return second_variant_rule_ids


    def prepare_rules_for_compilation_with_audit(self,
        rules: List[Dict],
        dataset: str,
        subtask: str,
        second_variant_rule_ids: set[str] | None = None,
    ) -> Tuple[List[Dict], Dict]:
        return _prepare_rules_for_compilation_with_audit_impl(
            rules,
            dataset,
            subtask,
            second_variant_rule_ids,
            load_split_xy_fn=self.load_split_xy,
            learn_threshold_if_needed_fn=self.learn_threshold_if_needed,
            annotate_second_variant_rules_fn=self._annotate_second_variant_rules,
            semantic_rule_signature_fn=self.semantic_rule_signature,
            audit_rule_summary_fn=self._audit_rule_summary,
            audit_rule_id_fn=self._audit_rule_id,
        )


    def compute_rule_ablation_summary(self, dataset, subtask, spec, X_train, y_train, X_valid, y_valid, X_test, y_test, kept_features):
        if len(kept_features) <= 1:
            return []
        hparams = self.load_model_hparams()
        best_params = hparams[dataset][subtask] if dataset in {"tox21", "sider"} else hparams[dataset]
        max_features = min(len(kept_features), self.config.max_ablation_features)
        feature_order = list(range(max_features))
        summary = []

        if spec.task_type == "classification":
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_valid_s = scaler.transform(X_valid)
            X_test_s = scaler.transform(X_test)
            clf = RandomForestClassifier(**dict(best_params, random_state=0))
            clf.fit(X_train_s, y_train)
            full_valid = float(roc_auc_score(y_valid, clf.predict_proba(X_valid_s)[:, 1]))
            full_test = float(roc_auc_score(y_test, clf.predict_proba(X_test_s)[:, 1]))
            for idx in feature_order:
                keep = [col for col in range(X_train_s.shape[1]) if col != idx]
                if not keep:
                    continue
                ab_clf = RandomForestClassifier(**dict(best_params, random_state=0))
                ab_clf.fit(X_train_s[:, keep], y_train)
                ab_valid = float(roc_auc_score(y_valid, ab_clf.predict_proba(X_valid_s[:, keep])[:, 1]))
                ab_test = float(roc_auc_score(y_test, ab_clf.predict_proba(X_test_s[:, keep])[:, 1]))
                summary.append(
                    {
                        "feature": kept_features[idx],
                        "valid_delta": full_valid - ab_valid,
                        "test_delta": full_test - ab_test,
                    }
                )
            return summary

        if spec.dataset == "qm9":
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_valid_s = scaler.transform(X_valid)
            X_test_s = scaler.transform(X_test)
            clf = RandomForestRegressor(random_state=0)
            clf.fit(X_train_s, y_train)
            full_valid = float(mean_absolute_error(y_valid, clf.predict(X_valid_s)))
            full_test = float(mean_absolute_error(y_test, clf.predict(X_test_s)))
            for idx in feature_order:
                keep = [col for col in range(X_train_s.shape[1]) if col != idx]
                if not keep:
                    continue
                ab_clf = RandomForestRegressor(random_state=0)
                ab_clf.fit(X_train_s[:, keep], y_train)
                ab_valid = float(mean_absolute_error(y_valid, ab_clf.predict(X_valid_s[:, keep])))
                ab_test = float(mean_absolute_error(y_test, ab_clf.predict(X_test_s[:, keep])))
                summary.append(
                    {
                        "feature": kept_features[idx],
                        "valid_delta": ab_valid - full_valid,
                        "test_delta": ab_test - full_test,
                    }
                )
            return summary

        clf = RandomForestRegressor(**dict(best_params, random_state=0))
        clf.fit(X_train, y_train)
        full_valid = float(sqrt(mean_squared_error(y_valid, clf.predict(X_valid))))
        full_test = float(sqrt(mean_squared_error(y_test, clf.predict(X_test))))
        for idx in feature_order:
            keep = [col for col in range(X_train.shape[1]) if col != idx]
            if not keep:
                continue
            ab_clf = RandomForestRegressor(**dict(best_params, random_state=0))
            ab_clf.fit(X_train[:, keep], y_train)
            ab_valid = float(sqrt(mean_squared_error(y_valid, ab_clf.predict(X_valid[:, keep]))))
            ab_test = float(sqrt(mean_squared_error(y_test, ab_clf.predict(X_test[:, keep]))))
            summary.append(
                {
                    "feature": kept_features[idx],
                    "valid_delta": ab_valid - full_valid,
                    "test_delta": ab_test - full_test,
                }
            )
        return summary


    def _active_branch_metrics(self, branch_metrics: Dict) -> Dict:
        if self.config.knowledge_type == "synthesize":
            return branch_metrics.get("synthesize", {})
        if self.config.knowledge_type == "inference":
            return branch_metrics.get("inference", {})
        return branch_metrics.get("combined", {})


    def winner_metric_extract(self, branch_metrics: Dict) -> Dict:
        active_metrics = self._active_branch_metrics(branch_metrics)
        if self.config.knowledge_type != "all":
            return dict(active_metrics)

        candidate_sets = active_metrics.get("candidate_sets", []) or []
        winner_name = str(active_metrics.get("candidate_set_winner", "") or "")
        for item in candidate_sets:
            if str(item.get("name", "")) == winner_name:
                return dict(item)
        if candidate_sets:
            return dict(candidate_sets[0])
        return dict(active_metrics)


    def shortlist_all_finalists(self, candidate_sets: List[Dict]) -> List[Dict]:
        return _shortlist_all_finalists_impl(
            candidate_sets,
            max_candidates=self.config.all_finalist_max_candidates,
            candidate_alias_signature_fn=self._candidate_alias_signature,
        )


    def _build_feature_matrices(self, compiled_rules, smiles_train, smiles_valid, smiles_test):
        feature_names = [item.name for item in compiled_rules]
        matrices = [
            self.build_feature_matrix(compiled_rules, smiles)
            for smiles in (smiles_train, smiles_valid, smiles_test)
        ]
        return self.sanitize_feature_matrices(*matrices, feature_names)

    def _update_feature_metrics(
        self, feature_metrics, selected_rules, prepared_rules, compiled_rules,
        kept_features, branch_metrics, second_variant_rule_ids,
    ) -> List[Dict]:
        feature_metrics.update(self.compute_rule_diversity_metrics(prepared_rules))
        feature_metrics.update(self._prepared_retention_metrics(selected_rules, prepared_rules, second_variant_rule_ids))
        surviving_rules = [item.rule for item in self._surviving_compiled_rules(compiled_rules, kept_features)]
        feature_metrics.update(self._promoted_multi_rule_metrics(surviving_rules))
        active_metrics = self._active_branch_metrics(branch_metrics)
        for name in ("promoted_multi_candidate_count", "promoted_multi_root_count", "promoted_multi_compile_safe_count"):
            feature_metrics[name] = int(active_metrics.get(name, 0) or 0)
        winner_metrics = self.winner_metric_extract(branch_metrics)
        feature_metrics["second_variant_quality_score"] = float(
            winner_metrics.get("second_variant_quality_score", 0.0) or 0.0
        )
        feature_metrics["winner_template_type"] = str(winner_metrics.get("template_type", "") or "")
        for name in ("selected_rule_count", "selected_root_feature_count", "selected_alias_group_count"):
            feature_metrics[f"winner_{name}"] = int(winner_metrics.get(name, 0) or 0)
        feature_metrics["winner_template_novelty_score"] = float(winner_metrics.get("template_novelty_score", 0.0) or 0.0)
        feature_metrics["winner_selected_second_variant_count"] = int(
            winner_metrics.get("selected_second_variant_count", 0) or 0
        )
        feature_metrics["kept_feature_names"] = kept_features
        return surviving_rules


    def evaluate_candidate_set_valid_only(self,
        dataset: str,
        subtask: str,
        spec,
        smiles_train: List[str],
        y_train: List[float],
        smiles_valid: List[str],
        y_valid: List[float],
        candidate_set: Dict,
    ) -> Dict:
        finalist_name = str(candidate_set.get("name", ""))
        selected_rules = list(candidate_set.get("rules", []) or [])
        second_variant_rule_ids = self._collect_second_variant_rule_ids_for_rules(selected_rules)
        selection_metric, higher_is_better = self._selection_metric_metadata(spec, dataset)
        report = {
            "name": finalist_name,
            "template_type": str(candidate_set.get("template_type", "") or ""),
            "shortlist_reason": str(candidate_set.get("finalist_shortlist_reason", "") or ""),
            "selection_metric": selection_metric,
            "higher_is_better": higher_is_better,
            "selected_rule_count": len(selected_rules),
            "alias_group_signature": list(self._candidate_alias_signature(candidate_set)),
            "status": "ok",
            "valid_scores": [],
            "valid_mean": None,
            "prepared_rule_count": 0,
            "compiled_rule_count": 0,
            "invalid_rule_count": 0,
            "root_feature_count": 0,
            "effective_literature_coverage": 0,
            "effective_data_coverage": 0,
            "family_breakdown": {},
            "source_breakdown": {},
            "family_diversity": 0,
            "dominant_feature_share": 0.0,
            "kept_feature_names": [],
            "selected_feature_names": [str(rule.get("feature_name", "")) for rule in selected_rules],
        }
        if not selected_rules:
            report["status"] = "empty_selected_rules"
            return report

        try:
            prepared_rules, _ = self.prepare_rules_for_compilation_with_audit(
                selected_rules,
                dataset,
                subtask,
                second_variant_rule_ids=second_variant_rule_ids,
            )
            compiled_rules, invalid_rules = self.compile_rules(prepared_rules)
            report["prepared_rule_count"] = len(prepared_rules)
            report["compiled_rule_count"] = len(compiled_rules)
            report["invalid_rule_count"] = len(invalid_rules)
            if not compiled_rules:
                report["status"] = "no_compiled_rules"
                return report

            feature_names = [item.name for item in compiled_rules]
            X_train = self.build_feature_matrix(compiled_rules, smiles_train)
            X_valid = self.build_feature_matrix(compiled_rules, smiles_valid)
            X_train, X_valid, _, kept_features, feature_metrics = self.sanitize_feature_matrices(
                X_train,
                X_valid,
                X_valid,
                feature_names,
            )
            surviving_compiled = self._surviving_compiled_rules(compiled_rules, kept_features)
            surviving_rules = [item.rule for item in surviving_compiled]
            feature_metrics.update(self.compute_rule_diversity_metrics(surviving_rules))
            feature_metrics.update(self._promoted_multi_rule_metrics(surviving_rules))
            valid_scores = self._evaluate_valid_only(dataset, subtask, spec, X_train, y_train, X_valid, y_valid)
            report.update(
                {
                    "valid_scores": valid_scores,
                    "valid_mean": float(np.mean(valid_scores)) if valid_scores else None,
                    "root_feature_count": int(feature_metrics.get("root_feature_count", 0) or 0),
                    "effective_literature_coverage": int(feature_metrics.get("effective_literature_coverage", 0) or 0),
                    "effective_data_coverage": int(feature_metrics.get("effective_data_coverage", 0) or 0),
                    "family_breakdown": dict(feature_metrics.get("family_breakdown", {}) or {}),
                    "source_breakdown": dict(feature_metrics.get("source_breakdown", {}) or {}),
                    "family_diversity": int(feature_metrics.get("family_diversity", 0) or 0),
                    "dominant_feature_share": float(feature_metrics.get("dominant_feature_share", 0.0) or 0.0),
                    "kept_feature_names": list(kept_features),
                    "kept_feature_count": len(kept_features),
                    "promoted_multi_surviving_root_count": int(
                        feature_metrics.get("promoted_multi_surviving_root_count", 0) or 0
                    ),
                    "surviving_multi_root_count": int(feature_metrics.get("surviving_multi_root_count", 0) or 0),
                    "selection_objective": float(np.mean(valid_scores)) if higher_is_better else -float(np.mean(valid_scores)),
                }
            )
            return report
        except Exception as exc:
            report["status"] = f"{type(exc).__name__}: {exc}"
            return report


    def _coverage_qualified_finalist(self, report: Dict) -> bool:
        return _coverage_qualified_finalist_impl(
            report,
            min_roots=self.config.all_finalist_min_roots,
            min_effective_data=self.config.all_finalist_min_effective_data,
            min_effective_literature=self.config.all_finalist_min_effective_lit,
        )


    def choose_finalist_with_coverage_guard(self, finalist_reports: List[Dict]) -> Dict:
        return _choose_finalist_with_coverage_guard_impl(
            finalist_reports,
            keep_literature_delta=self.config.all_finalist_keep_lit_delta,
            selection_objective_fn=self._selection_objective,
            coverage_qualified_finalist_fn=self._coverage_qualified_finalist,
            objective_gap_fn=self._objective_gap,
        )


    def _coverage_qualified_final_winner(self, feature_metrics: Dict, compiled_rule_count: int) -> bool:
        return bool(
            int(compiled_rule_count or 0) >= self.config.all_finalist_min_roots
            and int(feature_metrics.get("root_feature_count", 0) or 0) >= self.config.all_finalist_min_roots
            and int(feature_metrics.get("effective_literature_coverage", 0) or 0) >= self.config.all_finalist_min_effective_lit
            and int(feature_metrics.get("effective_data_coverage", 0) or 0) >= self.config.all_finalist_min_effective_data
        )


    def _coverage_qualified_winner_for_plus_one(self, feature_metrics: Dict, compiled_rule_count: int) -> bool:
        return bool(
            self._coverage_qualified_final_winner(feature_metrics, compiled_rule_count)
            and int(feature_metrics.get("promoted_multi_surviving_root_count", 0) or 0) >= self.config.all_plus_one_alias_min_promoted_multi_roots
        )


    def accept_plus_one_alias_extension(self, base_report: Dict, extended_report: Dict) -> bool:
        if not extended_report:
            return False
        return bool(
            float(extended_report.get("valid_mean", float("-inf")) or float("-inf"))
            >= float(base_report.get("valid_mean", float("-inf")) or float("-inf")) - self.config.all_plus_one_alias_max_valid_drop
            and int(extended_report.get("compiled_rule_count", 0) or 0)
            >= int(base_report.get("compiled_rule_count", 0) or 0)
            and int(extended_report.get("effective_literature_coverage", 0) or 0)
            >= int(base_report.get("effective_literature_coverage", 0) or 0)
            and int(extended_report.get("effective_data_coverage", 0) or 0)
            >= int(base_report.get("effective_data_coverage", 0) or 0)
            and bool(extended_report.get("plus_one_candidate_survived", False))
        )


    def _lit_only_slot_control_qualified(self, current_report: Dict) -> Tuple[bool, str]:
        if not self.config.lit_only_slot_control_enable:
            return False, "feature_disabled"
        if not bool(current_report.get("winner_coverage_qualified", False)):
            return False, "winner_not_coverage_qualified"
        if int(current_report.get("base_promoted_multi_surviving_root_count", 0) or 0) < self.config.lit_only_slot_min_promoted_multi_roots:
            return False, "promoted_multi_roots_below_threshold"
        if int(current_report.get("literature_only_rule_count", 0) or 0) != 1:
            return False, "literature_only_rule_count_not_1"
        if int(current_report.get("base_compiled_rule_count", 0) or 0) < 7:
            return False, "base_compiled_rule_count_below_7"
        if int(current_report.get("base_root_feature_count", 0) or 0) < self.config.all_finalist_min_roots:
            return False, "base_root_feature_count_below_floor"
        if int(current_report.get("base_effective_literature_coverage", 0) or 0) < 3:
            return False, "base_literature_coverage_below_3"
        if int(current_report.get("base_effective_data_coverage", 0) or 0) < self.config.all_finalist_min_effective_data:
            return False, "base_data_coverage_below_floor"
        return True, "qualified"


    def maybe_apply_lit_only_slot_control(self,
        dataset: str,
        subtask: str,
        spec,
        branch_metrics: Dict,
        selected_rules: List[Dict],
        prepared_rules: List[Dict],
        compiled_rules,
        invalid_rules: List[Dict],
        feature_metrics: Dict,
        valid_scores: List[float],
        test_scores: List[float],
        ablation_summary,
        shrinkage_audit_payload: Dict | None,
        second_variant_rule_ids: set[str],
        smiles_train: List[str],
        y_train: List[float],
        smiles_valid: List[str],
        y_valid: List[float],
        smiles_test: List[str],
        y_test: List[float],
    ):
        surviving_rules = [copy.deepcopy(item.rule) for item in self._surviving_compiled_rules(compiled_rules, feature_metrics.get("kept_feature_names", []) or [])]
        literature_only_rules = self._winner_literature_only_rules(surviving_rules)
        report = {
            "trigger_qualified": False,
            "base_selected_rule_count": int(len(selected_rules)),
            "base_compiled_rule_count": int(len(compiled_rules)),
            "base_root_feature_count": int(feature_metrics.get("root_feature_count", 0) or 0),
            "base_effective_literature_coverage": int(feature_metrics.get("effective_literature_coverage", 0) or 0),
            "base_effective_data_coverage": int(feature_metrics.get("effective_data_coverage", 0) or 0),
            "base_promoted_multi_surviving_root_count": int(feature_metrics.get("promoted_multi_surviving_root_count", 0) or 0),
            "literature_only_rule_count": int(len(literature_only_rules)),
            "candidate_rule_id": "",
            "candidate_feature_name": "",
            "candidate_root_feature_key": "",
            "candidate_source_bucket": "literature",
            "status": "skipped",
            "reason": "",
            "base_valid_mean": float(np.mean(valid_scores)) if valid_scores else None,
            "drop_valid_mean": None,
            "valid_delta": None,
            "accepted": False,
            "post_selected_rule_count": None,
            "post_compiled_rule_count": None,
            "post_root_feature_count": None,
            "post_effective_literature_coverage": None,
            "post_effective_data_coverage": None,
            "candidate_removed_from_surviving": None,
            "max_valid_drop": self.config.lit_only_slot_max_valid_drop,
        }
        report["winner_coverage_qualified"] = self._coverage_qualified_final_winner(feature_metrics, len(compiled_rules))
        if literature_only_rules:
            candidate_rule = literature_only_rules[0]
            report["candidate_rule_id"] = str(candidate_rule.get("canonical_key", "")).strip()
            report["candidate_feature_name"] = str(candidate_rule.get("feature_name", "") or "")
            report["candidate_root_feature_key"] = self._rule_root_feature_key(candidate_rule)
        qualified, reason = self._lit_only_slot_control_qualified(report)
        report["trigger_qualified"] = bool(qualified)
        if not qualified:
            report["reason"] = reason
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )

        drop_variant_selected_rules = self._remove_rule_once_by_canonical_key(selected_rules, report["candidate_rule_id"])
        drop_eval = self.evaluate_rule_set_end_to_end(
            dataset,
            subtask,
            spec,
            drop_variant_selected_rules,
            branch_metrics,
            self._collect_second_variant_rule_ids_for_rules(drop_variant_selected_rules),
            smiles_train,
            y_train,
            smiles_valid,
            y_valid,
            smiles_test,
            y_test,
        )
        report["status"] = "evaluated"
        if drop_eval.get("status") != "ok":
            report["reason"] = str(drop_eval.get("status", "evaluation_failed"))
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )

        base_valid_mean = float(report["base_valid_mean"] or 0.0)
        drop_valid_mean = float(np.mean(drop_eval["valid_scores"])) if drop_eval.get("valid_scores") else None
        report["drop_valid_mean"] = drop_valid_mean
        report["valid_delta"] = (drop_valid_mean - base_valid_mean) if drop_valid_mean is not None else None
        post_metrics = drop_eval["feature_metrics"]
        report["post_selected_rule_count"] = int(len(drop_eval["selected_rules"]))
        report["post_compiled_rule_count"] = int(len(drop_eval["compiled_rules"]))
        report["post_root_feature_count"] = int(post_metrics.get("root_feature_count", 0) or 0)
        report["post_effective_literature_coverage"] = int(post_metrics.get("effective_literature_coverage", 0) or 0)
        report["post_effective_data_coverage"] = int(post_metrics.get("effective_data_coverage", 0) or 0)
        report["candidate_removed_from_surviving"] = bool(
            report["candidate_rule_id"] and report["candidate_rule_id"] not in drop_eval.get("surviving_rules_by_key", set())
        )

        if int(report["post_compiled_rule_count"] or 0) < 6:
            report["status"] = "rejected"
            report["reason"] = "drop_breaks_compiled_floor"
        elif int(report["post_root_feature_count"] or 0) < 6:
            report["status"] = "rejected"
            report["reason"] = "drop_breaks_root_floor"
        elif int(report["post_effective_literature_coverage"] or 0) < 2 or int(report["post_effective_data_coverage"] or 0) < 5:
            report["status"] = "rejected"
            report["reason"] = "drop_breaks_coverage_floor"
        elif not bool(report["candidate_removed_from_surviving"]):
            report["status"] = "rejected"
            report["reason"] = "candidate_not_removed_from_surviving"
        elif drop_valid_mean is None or drop_valid_mean < base_valid_mean - self.config.lit_only_slot_max_valid_drop:
            report["status"] = "rejected"
            report["reason"] = "valid_drop_too_large"
        else:
            report["status"] = "accepted"
            report["reason"] = "accepted"
            report["accepted"] = True
            return (
                drop_eval["selected_rules"],
                drop_eval["prepared_rules"],
                drop_eval["compiled_rules"],
                drop_eval["invalid_rules"],
                drop_eval["feature_metrics"],
                drop_eval["valid_scores"],
                drop_eval["test_scores"],
                drop_eval["ablation_summary"],
                drop_eval["shrinkage_audit"],
                self._collect_second_variant_rule_ids_for_rules(drop_eval["selected_rules"]),
                drop_eval["rendered_code"],
                drop_eval["function_names"],
                report,
            )

        return (
            selected_rules,
            prepared_rules,
            compiled_rules,
            invalid_rules,
            feature_metrics,
            valid_scores,
            test_scores,
            ablation_summary,
            shrinkage_audit_payload,
            second_variant_rule_ids,
            None,
            None,
            report,
        )


    def build_shrinkage_audit(self,
        branch_metrics: Dict,
        selected_rules: List[Dict],
        prepare_audit: Dict,
        prepared_rules: List[Dict],
        compiled_rules,
        invalid_rules: List[Dict],
        feature_metrics: Dict,
    ) -> Dict:
        active_metrics = self._active_branch_metrics(branch_metrics)
        winner_metrics = self.winner_metric_extract(branch_metrics)
        selection_trace = active_metrics.get("selection_trace", {}) or {}
        raw_rules = selection_trace.get("raw_rules", []) or []
        canonical_rules = selection_trace.get("canonical_rules", []) or []
        gated_rules = selection_trace.get("gated_rules", []) or []
        selected_from_trace = selection_trace.get("selected_rules", []) or selected_rules
        selection_audit = selection_trace.get("selection_audit", {}) or {}
        second_variant_rule_ids = self._collect_second_variant_rule_ids(branch_metrics)
        threshold_learned_rules = prepare_audit.get("threshold_learned_rules", []) or []
        compiled_rule_by_name = {item.name: item.rule for item in compiled_rules}
        compiled_rule_names = list(compiled_rule_by_name.keys())
        kept_feature_names = feature_metrics.get("kept_feature_names", []) or []

        raw_groups: Dict[str, List[Dict]] = {}
        for rule in raw_rules:
            raw_groups.setdefault(self._audit_rule_id(rule), []).append(rule)
        raw_to_canonical = []
        for grouped_rules in raw_groups.values():
            for rule in grouped_rules[1:]:
                raw_to_canonical.append(
                    {
                        "rule": self._audit_rule_summary(
                            dict(rule, _audit_is_second_variant_participant=self._audit_rule_id(rule) in second_variant_rule_ids)
                        ),
                        "reason": "merged_duplicate",
                        "replacement_rule_id": self._audit_rule_id(grouped_rules[0]),
                    }
                )

        gated_ids = {self._audit_rule_id(rule) for rule in gated_rules}
        canonical_to_gated = []
        for rule in canonical_rules:
            if self._audit_rule_id(rule) in gated_ids:
                continue
            canonical_to_gated.append(
                {
                    "rule": self._audit_rule_summary(
                        dict(rule, _audit_is_second_variant_participant=self._audit_rule_id(rule) in second_variant_rule_ids)
                    ),
                    "reason": "gate_filtered",
                    "gate_failure_reasons": gate_failure_reasons(rule),
                }
            )

        gated_to_selected = []
        for item in selection_audit.get("dropped", []):
            gated_to_selected.append(
                {
                    "rule": self._audit_rule_summary(
                        dict(
                            item.get("rule", {}),
                            _audit_is_second_variant_participant=(
                                self._audit_rule_id(item.get("rule", {})) in second_variant_rule_ids
                            ),
                        )
                    ),
                    "reason": item.get("reason", "selection_filtered"),
                }
            )

        prepared_to_compiled_invalid = [
            {
                "rule": self._audit_rule_summary(rule),
                "reason": "compile_invalid_after_prepare",
            }
            for rule in invalid_rules
        ]
        compiled_to_sanitized = []
        for item in feature_metrics.get("dropped_features", []) or []:
            feature_name = item.get("feature_name")
            compiled_rule = compiled_rule_by_name.get(feature_name)
            compiled_to_sanitized.append(
                {
                    "feature_name": feature_name,
                    "reason": item.get("reason", "sanitized"),
                    "rule": (
                        self._audit_rule_summary(
                            dict(
                                compiled_rule,
                                _audit_is_second_variant_participant=(
                                    self._audit_rule_id(compiled_rule) in second_variant_rule_ids
                                ),
                            )
                        )
                        if compiled_rule
                        else None
                    ),
                }
            )

        return {
            "knowledge_type": self.config.knowledge_type,
            "winner_metrics": winner_metrics,
            "stage_counts": {
                "raw_rule_count": len(raw_rules),
                "canonical_rule_count": len(canonical_rules),
                "gated_rule_count": len(gated_rules),
                "selected_rule_count": len(selected_from_trace),
                "threshold_learned_rule_count": len(threshold_learned_rules),
                "prepared_rule_count": len(prepared_rules),
                "compiled_rule_count": len(compiled_rules),
                "invalid_rule_count": len(invalid_rules),
                "sanitized_feature_count": len(kept_feature_names),
            },
            "stage_metrics": {
                "selected": self._stage_rule_metrics(selected_from_trace),
                "prepared": self._stage_rule_metrics(prepared_rules),
                "compiled": self._compiled_stage_metrics(compiled_rules),
                "sanitized": {
                    "rule_count": int(feature_metrics.get("feature_count_after_filter", 0)),
                    "root_feature_count": int(feature_metrics.get("root_feature_count", 0)),
                    "family_breakdown": feature_metrics.get("family_breakdown", {}) or {},
                    "family_diversity": int(feature_metrics.get("family_diversity", 0) or 0),
                    "source_breakdown": feature_metrics.get("source_breakdown", {}) or {},
                    "effective_literature_coverage": int(
                        feature_metrics.get("effective_literature_coverage", 0) or 0
                    ),
                    "effective_data_coverage": int(feature_metrics.get("effective_data_coverage", 0) or 0),
                    "promoted_multi_surviving_root_count": int(
                        feature_metrics.get("promoted_multi_surviving_root_count", 0) or 0
                    ),
                    "surviving_multi_root_count": int(feature_metrics.get("surviving_multi_root_count", 0) or 0),
                },
            },
            "drops": {
                "raw_to_canonical": raw_to_canonical,
                "canonical_to_gated": canonical_to_gated,
                "gated_to_selected": gated_to_selected,
                "selected_to_prepared": prepare_audit.get("dropped", []),
                "prepared_to_compiled_invalid": prepared_to_compiled_invalid,
                "compiled_to_sanitized": compiled_to_sanitized,
            },
            "stage_snapshots": {
                "selected_rules": [
                    self._audit_rule_summary(
                        dict(rule, _audit_is_second_variant_participant=self._audit_rule_id(rule) in second_variant_rule_ids)
                    )
                    for rule in selected_from_trace
                ],
                "prepared_rules": [
                    self._audit_rule_summary(
                        dict(rule, _audit_is_second_variant_participant=self._audit_rule_id(rule) in second_variant_rule_ids)
                    )
                    for rule in prepared_rules
                ],
                "compiled_rule_names": compiled_rule_names,
                "kept_feature_names": kept_feature_names,
            },
        }


    def evaluate_rule_set_end_to_end(self,
        dataset: str,
        subtask: str,
        spec,
        selected_rules: List[Dict],
        branch_metrics: Dict,
        second_variant_rule_ids: set[str],
        smiles_train: List[str],
        y_train: List[float],
        smiles_valid: List[str],
        y_valid: List[float],
        smiles_test: List[str],
        y_test: List[float],
    ) -> Dict:
        prepared_rules, prepare_audit = self.prepare_rules_for_compilation_with_audit(
            selected_rules,
            dataset,
            subtask,
            second_variant_rule_ids=second_variant_rule_ids,
        )
        compiled_rules, invalid_rules = self.compile_rules(prepared_rules)
        if not compiled_rules:
            return {
                "status": "no_compiled_rules",
                "selected_rules": selected_rules,
                "prepared_rules": prepared_rules,
                "compiled_rules": compiled_rules,
                "invalid_rules": invalid_rules,
                "prepare_audit": prepare_audit,
            }
        X_train, X_valid, X_test, kept_features, feature_metrics = self._build_feature_matrices(
            compiled_rules, smiles_train, smiles_valid, smiles_test,
        )
        surviving_rules = self._update_feature_metrics(
            feature_metrics, selected_rules, prepared_rules, compiled_rules,
            kept_features, branch_metrics, second_variant_rule_ids,
        )
        if spec.task_type == "classification":
            valid_scores, test_scores = self.evaluate_classification(dataset, subtask, X_train, y_train, X_valid, y_valid, X_test, y_test)
        elif dataset == "qm9":
            valid_scores, test_scores = self.evaluate_qm9(X_train, y_train, X_valid, y_valid, X_test, y_test)
        else:
            valid_scores, test_scores = self.evaluate_regression_rf(dataset, X_train, y_train, X_valid, y_valid, X_test, y_test)
        ablation_summary = self.compute_rule_ablation_summary(
            dataset,
            subtask,
            spec,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            kept_features,
        )
        shrinkage_audit_payload = self.build_shrinkage_audit(
            branch_metrics=branch_metrics,
            selected_rules=selected_rules,
            prepare_audit=prepare_audit,
            prepared_rules=prepared_rules,
            compiled_rules=compiled_rules,
            invalid_rules=invalid_rules,
            feature_metrics=feature_metrics,
        )
        rendered_code, function_names = self.render_rule_module(prepared_rules, compiled_rules)
        surviving_rules_by_key = {str(rule.get("canonical_key", "")).strip() for rule in surviving_rules}
        return {
            "status": "ok",
            "selected_rules": selected_rules,
            "prepared_rules": prepared_rules,
            "compiled_rules": compiled_rules,
            "invalid_rules": invalid_rules,
            "prepare_audit": prepare_audit,
            "kept_features": kept_features,
            "feature_metrics": feature_metrics,
            "valid_scores": valid_scores,
            "test_scores": test_scores,
            "ablation_summary": ablation_summary,
            "shrinkage_audit": shrinkage_audit_payload,
            "rendered_code": rendered_code,
            "function_names": function_names,
            "surviving_rules": surviving_rules,
            "surviving_rules_by_key": surviving_rules_by_key,
        }


    def maybe_apply_plus_one_alias_extension(self,
        dataset: str,
        subtask: str,
        spec,
        branch_metrics: Dict,
        selected_rules: List[Dict],
        prepared_rules: List[Dict],
        compiled_rules,
        invalid_rules: List[Dict],
        feature_metrics: Dict,
        valid_scores: List[float],
        test_scores: List[float],
        ablation_summary,
        shrinkage_audit_payload: Dict | None,
        second_variant_rule_ids: set[str],
        smiles_train: List[str],
        y_train: List[float],
        smiles_valid: List[str],
        y_valid: List[float],
        smiles_test: List[str],
        y_test: List[float],
    ):
        report = {
            "enabled": bool(self.config.all_plus_one_alias_enable),
            "trigger_qualified": False,
            "candidate_status": "disabled",
            "accepted": False,
            "slot_type": "",
            "selection_metric": "valid_mean",
            "max_valid_drop": self.config.all_plus_one_alias_max_valid_drop,
            "base_selected_rule_count": int(len(selected_rules)),
            "base_compiled_rule_count": int(len(compiled_rules)),
        }
        if not self.config.all_plus_one_alias_enable or self.config.knowledge_type != "all":
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        if len(selected_rules) != 6 or len(compiled_rules) != 6:
            report["status"] = "skipped"
            report["reason"] = "base_rule_count_not_6"
            report["candidate_status"] = "base_rule_count_not_6"
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        report["trigger_qualified"] = self._coverage_qualified_winner_for_plus_one(feature_metrics, len(compiled_rules))
        if not report["trigger_qualified"]:
            report["candidate_status"] = "trigger_not_qualified"
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        candidate_pool = list(
            (((branch_metrics.get("combined", {}) or {}).get("selection_trace", {}) or {}).get("gated_rules", []) or [])
        )
        candidate_choice = self.choose_plus_one_alias_candidate(selected_rules, candidate_pool)
        report.update({key: value for key, value in candidate_choice.items() if key != "candidate"})
        if candidate_choice.get("status") != "ok":
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        extended_selected_rules = list(selected_rules) + [copy.deepcopy(candidate_choice["candidate"])]
        extended_second_variant_rule_ids = self._collect_second_variant_rule_ids_for_rules(extended_selected_rules)
        extended_eval = self.evaluate_rule_set_end_to_end(
            dataset,
            subtask,
            spec,
            extended_selected_rules,
            branch_metrics,
            extended_second_variant_rule_ids,
            smiles_train,
            y_train,
            smiles_valid,
            y_valid,
            smiles_test,
            y_test,
        )
        if extended_eval.get("status") != "ok":
            report["candidate_status"] = str(extended_eval.get("status", "evaluation_failed"))
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        candidate_key = str(candidate_choice.get("candidate_canonical_key", "")).strip()
        base_report = {
            "valid_mean": float(np.mean(valid_scores)) if valid_scores else None,
            "compiled_rule_count": len(compiled_rules),
            "effective_literature_coverage": int(feature_metrics.get("effective_literature_coverage", 0) or 0),
            "effective_data_coverage": int(feature_metrics.get("effective_data_coverage", 0) or 0),
        }
        extended_report = {
            "valid_mean": float(np.mean(extended_eval["valid_scores"])) if extended_eval.get("valid_scores") else None,
            "compiled_rule_count": len(extended_eval["compiled_rules"]),
            "effective_literature_coverage": int(extended_eval["feature_metrics"].get("effective_literature_coverage", 0) or 0),
            "effective_data_coverage": int(extended_eval["feature_metrics"].get("effective_data_coverage", 0) or 0),
            "plus_one_candidate_survived": bool(candidate_key and candidate_key in extended_eval["surviving_rules_by_key"]),
        }
        report["extended_valid_mean"] = float(extended_report["valid_mean"] or 0.0)
        report["valid_delta"] = float((extended_report["valid_mean"] or 0.0) - (base_report["valid_mean"] or 0.0))
        report["extended_compiled_rule_count"] = int(extended_report["compiled_rule_count"])
        report["extended_effective_literature_coverage"] = int(extended_report["effective_literature_coverage"])
        report["extended_effective_data_coverage"] = int(extended_report["effective_data_coverage"])
        report["plus_one_candidate_survived"] = bool(extended_report["plus_one_candidate_survived"])
        if not self.accept_plus_one_alias_extension(base_report, extended_report):
            report["candidate_status"] = "rejected_by_acceptance_rule"
            return (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                None,
                None,
                report,
            )
        report["candidate_status"] = "accepted"
        report["accepted"] = True
        return (
            extended_selected_rules,
            extended_eval["prepared_rules"],
            extended_eval["compiled_rules"],
            extended_eval["invalid_rules"],
            extended_eval["feature_metrics"],
            extended_eval["valid_scores"],
            extended_eval["test_scores"],
            extended_eval["ablation_summary"],
            extended_eval["shrinkage_audit"],
            extended_second_variant_rule_ids,
            extended_eval["rendered_code"],
            extended_eval["function_names"],
            report,
        )


    def save_shrinkage_audit(self, dataset: str, subtask: str, payload: Dict) -> None:
        path = self.shrinkage_audit_path(dataset, subtask)
        self.save_rule_artifact(path, payload)


    def save_metric_report(self, dataset: str, subtask: str, valid_scores, test_scores):
        result_dir = self.result_output_dir(dataset)
        suffix = f"_{subtask}" if subtask else ""
        if dataset in {"bbbp", "bace", "clintox", "hiv", "tox21", "sider"}:
            filename = f"{self.model_tag()}_{dataset}{suffix}_{self.config.knowledge_type}_rules_test_roc_auc.txt"
            metric_name = "ROC-AUC"
        elif dataset in {"esol", "lipophilicity", "freesolv"}:
            filename = f"{self.model_tag()}_{dataset}_{self.config.knowledge_type}_rules_test_rmse.txt"
            metric_name = "RMSE"
        else:
            filename = f"{self.model_tag()}_{subtask}_{self.config.knowledge_type}_rules_test_mae.txt"
            metric_name = "MAE"
        with open(os.path.join(result_dir, filename), "w", encoding="utf-8") as handle:
            for value in test_scores:
                handle.write(f"{value}\n")
            handle.write(f"\nAverage test {metric_name}: {np.mean(test_scores)}\n")
            handle.write(f"Average valid {metric_name}: {np.mean(valid_scores)}\n")
            handle.write(f"Standard deviation of test {metric_name}: {np.std(test_scores)}\n")
            handle.write(f"Standard deviation of valid {metric_name}: {np.std(valid_scores)}\n")


    def save_skip_metric_report(self, dataset: str, subtask: str, reason: str, status: str = "skipped"):
        result_dir = self.result_output_dir(dataset)
        suffix = f"_{subtask}" if subtask else ""
        if dataset in {"bbbp", "bace", "clintox", "hiv", "tox21", "sider"}:
            filename = f"{self.model_tag()}_{dataset}{suffix}_{self.config.knowledge_type}_rules_test_roc_auc.txt"
        elif dataset in {"esol", "lipophilicity", "freesolv"}:
            filename = f"{self.model_tag()}_{dataset}_{self.config.knowledge_type}_rules_test_rmse.txt"
        else:
            filename = f"{self.model_tag()}_{subtask}_{self.config.knowledge_type}_rules_test_mae.txt"
        with open(os.path.join(result_dir, filename), "w", encoding="utf-8") as handle:
            handle.write(f"{status.upper()}\n")
            handle.write(f"Reason: {reason}\n")


    def save_skip_outputs(self,
        dataset: str,
        subtask: str,
        branch_metrics: Dict,
        selected_rules: List[Dict],
        prepared_rules: List[Dict],
        compiled_rules,
        invalid_rules: List[Dict],
        reason: str,
        status: str = "skipped",
        null_model_scores: Dict | None = None,
        shrinkage_audit: Dict | None = None,
    ):
        runtime_snapshot = self.load_runtime_snapshot()
        code_path, metadata_path = self.code_output_paths(dataset, subtask)
        with open(code_path, "w", encoding="utf-8") as handle:
            handle.write(f"# {status.upper()}\n# {reason}\n")

        metadata = {
            "dataset": dataset,
            "subtask": subtask,
            "knowledge_type": self.config.knowledge_type,
            "run_tag": self.config.run_tag,
            "status": status,
            "skip_reason": reason,
            "selected_rule_count": len(selected_rules),
            "compiled_rule_count": len(compiled_rules),
            "invalid_rule_count": len(invalid_rules),
            "compiled_feature_names": [item.name for item in compiled_rules],
            "kept_feature_names": [],
            "branch_metrics": branch_metrics,
            "strict_failure_diagnostics": branch_metrics.get("strict_failure_diagnostics", {}),
            "quality_gate": {
                "mode": "skip_or_degraded",
                "reason": reason,
            },
            "runtime_snapshot": runtime_snapshot,
            "feature_metrics": {
                "feature_count_before_filter": len(compiled_rules),
                "feature_count_after_filter": 0,
                "non_nan_coverage": 0.0,
                "feature_entropy": 0.0,
                "pairwise_correlation": 0.0,
                "family_breakdown": {},
                "family_diversity": 0,
                "source_breakdown": {},
                "effective_literature_coverage": 0,
                "effective_data_coverage": 0,
                "root_feature_count": 0,
                "dominant_feature_share": 0.0,
            },
            "null_model_scores": null_model_scores or {},
            "valid_scores": [],
            "test_scores": [],
            "rules": prepared_rules,
            "shrinkage_audit": shrinkage_audit,
        }
        self.save_rule_artifact(metadata_path, metadata)
        self.write_unified_outcome_artifact(dataset, subtask, metadata_path, code_path, metadata)
        if shrinkage_audit is not None:
            self.save_shrinkage_audit(dataset, subtask, shrinkage_audit)
        self.save_skip_metric_report(dataset, subtask, reason, status=status)
        print(f"[TrainEval] {status}: {reason}")
        print(f"[TrainEval] wrote {status} metadata to {metadata_path}")
        print(f"[TrainEval] wrote {status} code stub to {code_path}")


    def write_unified_outcome_artifact(self, dataset: str, subtask: str, metadata_path: str, code_path: str, metadata: Dict):
        quality_gate = metadata.get("quality_gate", {}) or {}
        status = str(metadata.get("status", "unknown") or "unknown")
        final_result_source = str(metadata.get("final_result_source", self.config.knowledge_type) or self.config.knowledge_type)
        null_scores = metadata.get("null_model_scores", {}) or {}
        task_type, metric_name = self._outcome_task_and_metric(dataset, subtask)
        outcome_payload = self.build_outcome_payload(
            system_id="D2",
            dataset=dataset,
            subtask=subtask,
            generation_seed=getattr(self.config, "generation_seed", None),
            rf_seeds=[0, 1, 2, 3, 4],
            status=status,
            final_result_source=final_result_source,
            quality_gate_mode=str(quality_gate.get("mode", "") or ""),
            quality_gate_reason=str(quality_gate.get("reason", metadata.get("skip_reason", "")) or ""),
            experiment_id=self.config.run_tag or None,
            run_tag=self.config.run_tag or None,
            task_type=task_type,
            metric_name=metric_name,
            actual_valid_mean=metadata.get("valid_mean") if status == "success" else None,
            actual_test_mean=metadata.get("test_mean") if status == "success" else None,
            null_model_valid_mean=null_scores.get("valid_mean"),
            null_model_test_mean=null_scores.get("test_mean"),
            selected_rule_count=int(metadata.get("selected_rule_count", 0) or 0),
            compiled_rule_count=int(metadata.get("compiled_rule_count", 0) or 0),
            invalid_rule_count=int(metadata.get("invalid_rule_count", 0) or 0),
            rendered_code_available=os.path.exists(code_path),
            artifact_paths={
                "code": code_path,
                "metadata": metadata_path,
            },
            extra={
                "knowledge_type": self.config.knowledge_type,
                "run_tag": self.config.run_tag,
                "strict_failure_diagnostics": metadata.get("strict_failure_diagnostics", {}),
                "aggregate_diagnostics": {
                    "selected_rule_count": int(((metadata.get("branch_metrics", {}) or {}).get("combined", {}) or {}).get("selected_rule_count", 0) or 0),
                    "compile_valid_rate": float(((metadata.get("branch_metrics", {}) or {}).get("combined", {}) or {}).get("compile_valid_rate", 0.0) or 0.0),
                },
            },
        )
        self.write_json(self.outcome_output_path_for(dataset, subtask, self.config.knowledge_type, metadata_path), outcome_payload)


    def maybe_apply_validation_promotion_guard(self,
        dataset: str,
        subtask: str,
        metadata: Dict,
        rendered_code: str,
    ) -> Tuple[Dict, str, str]:
        if self.config.knowledge_type != "all":
            metadata["final_result_source"] = self.config.knowledge_type
            metadata["promotion"] = {"decision": "not_applicable"}
            return metadata, rendered_code, ""

        inference_code_path, inference_metadata_path = self.code_output_paths_for(dataset, subtask, "inference")
        if not os.path.exists(inference_metadata_path):
            metadata["final_result_source"] = "all_promoted"
            metadata["promotion"] = {
                "decision": "all_promoted",
                "reason": "inference_reference_missing",
                "epsilon": self.config.validation_promotion_epsilon,
            }
            return metadata, rendered_code, ""

        inference_metadata = self.load_rule_artifact(inference_metadata_path)
        inference_valid_mean = self._mean_score(inference_metadata.get("valid_scores", []))
        all_valid_mean = self._mean_score(metadata.get("valid_scores", []))
        all_test_mean = self._mean_score(metadata.get("test_scores", []))
        inference_test_mean = self._mean_score(inference_metadata.get("test_scores", []))
        feature_metrics = metadata.get("feature_metrics", {}) or {}
        compiled_feature_count = int(feature_metrics.get("feature_count_after_filter", metadata.get("compiled_rule_count", 0)))
        root_feature_count = int(feature_metrics.get("root_feature_count", 0))
        capacity_underfilled = compiled_feature_count < self.config.min_all_compiled_features
        _, metric_name = self._outcome_task_and_metric(dataset, subtask)
        metric_info = metric_direction(metric_name)
        valid_gap_raw = None if inference_valid_mean is None or all_valid_mean is None else float(all_valid_mean - inference_valid_mean)
        valid_improvement = (
            None
            if inference_valid_mean is None or all_valid_mean is None
            else score_improvement(all_valid_mean, inference_valid_mean, metric_name)
        )
        diversity_bonus = self._all_candidate_diversity_bonus(metadata, inference_metadata)
        underdiverse_all = self._candidate_underdiverse_all(metadata)
        metadata["underdiverse_all"] = underdiverse_all
        metadata["capacity_underfilled_all"] = capacity_underfilled
        if inference_valid_mean is None or all_valid_mean is None:
            metadata["final_result_source"] = "all_promoted"
            metadata["promotion"] = {
                "decision": "all_promoted",
                "reason": "score_unavailable",
                "epsilon": self.config.validation_promotion_epsilon,
                "underdiverse_all": underdiverse_all,
            }
            return metadata, rendered_code, ""

        promotion_payload = {
            "valid_mean_all": all_valid_mean,
            "valid_mean_inference": inference_valid_mean,
            "test_mean_all": all_test_mean,
            "test_mean_inference": inference_test_mean,
            "epsilon": self.config.validation_promotion_epsilon,
            "keep_all_valid_delta": self.config.keep_all_valid_delta,
            "revert_all_valid_delta": self.config.revert_all_valid_delta,
            "metric_name": metric_name,
            "metric_higher_is_better": bool(metric_info["higher_is_better"]),
            "metric_direction_inferred": bool(metric_info["inferred"]),
            "metric_direction_defaulted": bool(metric_info["defaulted"]),
            "valid_improvement": valid_improvement,
            "valid_gap_raw": valid_gap_raw,
            "diversity_bonus": diversity_bonus,
            "underdiverse_all": underdiverse_all,
            "capacity_underfilled_all": capacity_underfilled,
            "compiled_feature_count": compiled_feature_count,
            "root_feature_count": root_feature_count,
            "selected_rule_count": int(metadata.get("selected_rule_count", 0)),
            "selected_source_breakdown": metadata.get("branch_metrics", {}).get("combined", {}).get("selected_source_breakdown", {}),
            "effective_literature_coverage": metadata.get("branch_metrics", {}).get("combined", {}).get(
                "effective_literature_coverage",
                feature_metrics.get("effective_literature_coverage", 0),
            ),
            "effective_data_coverage": metadata.get("branch_metrics", {}).get("combined", {}).get(
                "effective_data_coverage",
                feature_metrics.get("effective_data_coverage", 0),
            ),
            "selected_family_breakdown": metadata.get("branch_metrics", {}).get("combined", {}).get("selected_family_breakdown", {}),
            "candidate_set_winner": metadata.get("branch_metrics", {}).get("combined", {}).get("candidate_set_winner", ""),
            "candidate_set_vote_method": metadata.get("branch_metrics", {}).get("combined", {}).get("candidate_set_vote_method", ""),
        }

        if capacity_underfilled and not is_better_by(all_valid_mean, inference_valid_mean, self.config.capacity_keep_valid_delta, metric_name):
            fallback_metadata = copy.deepcopy(inference_metadata)
            fallback_metadata.update(
                {
                    "dataset": dataset,
                    "subtask": subtask,
                    "knowledge_type": "all",
                    "run_tag": self.config.run_tag,
                    "status": "success",
                    "branch_metrics": metadata.get("branch_metrics", {}),
                    "quality_gate": {
                        "mode": "all_capacity_blocked",
                        "reason": "all_capacity_blocked",
                    },
                    "runtime_snapshot": metadata.get("runtime_snapshot", {}),
                    "finalist_report": metadata.get("finalist_report", {}),
                    "promotion": {
                        "decision": "all_capacity_blocked",
                        **promotion_payload,
                    },
                    "final_result_source": "inference_only",
                    "candidate_all_result": {
                        "selected_rule_count": metadata.get("selected_rule_count", 0),
                        "compiled_rule_count": metadata.get("compiled_rule_count", 0),
                        "compiled_feature_names": metadata.get("compiled_feature_names", []),
                        "kept_feature_names": metadata.get("kept_feature_names", []),
                        "valid_mean": all_valid_mean,
                        "test_mean": all_test_mean,
                        "rules": metadata.get("rules", []),
                        "underdiverse_all": underdiverse_all,
                        "capacity_underfilled_all": capacity_underfilled,
                        "candidate_sets": metadata.get("branch_metrics", {}).get("combined", {}).get("candidate_sets", []),
                    },
                }
            )
            fallback_code = rendered_code
            if os.path.exists(inference_code_path):
                with open(inference_code_path, "r", encoding="utf-8") as handle:
                    fallback_code = handle.read()
            return (
                fallback_metadata,
                fallback_code,
                (
                    f"[TrainEval] all_capacity_blocked "
                    f"(compiled_features={compiled_feature_count} min_required={self.config.min_all_compiled_features} "
                    f"all_valid={all_valid_mean:.6f} inference_valid={inference_valid_mean:.6f})"
                ),
            )

        if underdiverse_all:
            if is_better_by(all_valid_mean, inference_valid_mean, self.config.validation_promotion_epsilon, metric_name):
                metadata["final_result_source"] = "all_underdiverse_kept"
                metadata["promotion"] = {"decision": "all_underdiverse_kept", **promotion_payload}
                return metadata, rendered_code, ""

            fallback_metadata = copy.deepcopy(inference_metadata)
            fallback_metadata.update(
                {
                    "dataset": dataset,
                    "subtask": subtask,
                    "knowledge_type": "all",
                    "run_tag": self.config.run_tag,
                    "status": "success",
                    "branch_metrics": metadata.get("branch_metrics", {}),
                    "quality_gate": {
                        "mode": "all_underdiverse_blocked",
                        "reason": "all_underdiverse_blocked",
                    },
                    "runtime_snapshot": metadata.get("runtime_snapshot", {}),
                    "finalist_report": metadata.get("finalist_report", {}),
                    "promotion": {
                        "decision": "all_underdiverse_blocked",
                        **promotion_payload,
                    },
                    "final_result_source": "inference_only",
                    "candidate_all_result": {
                        "selected_rule_count": metadata.get("selected_rule_count", 0),
                        "compiled_rule_count": metadata.get("compiled_rule_count", 0),
                        "compiled_feature_names": metadata.get("compiled_feature_names", []),
                        "kept_feature_names": metadata.get("kept_feature_names", []),
                        "valid_mean": all_valid_mean,
                        "test_mean": all_test_mean,
                        "rules": metadata.get("rules", []),
                        "underdiverse_all": underdiverse_all,
                        "capacity_underfilled_all": capacity_underfilled,
                        "candidate_sets": metadata.get("branch_metrics", {}).get("combined", {}).get("candidate_sets", []),
                    },
                }
            )
            fallback_code = rendered_code
            if os.path.exists(inference_code_path):
                with open(inference_code_path, "r", encoding="utf-8") as handle:
                    fallback_code = handle.read()
            return (
                fallback_metadata,
                fallback_code,
                (
                    f"[TrainEval] all_underdiverse_blocked "
                    f"(all_valid={all_valid_mean:.6f} inference_valid={inference_valid_mean:.6f} "
                    f"improvement={valid_improvement:.6f} raw_gap={valid_gap_raw:.6f} "
                    f"selected={int(metadata.get('selected_rule_count', 0))})"
                ),
            )

        if is_better_by(all_valid_mean, inference_valid_mean, self.config.validation_promotion_epsilon, metric_name):
            metadata["final_result_source"] = "all_promoted"
            metadata["promotion"] = {"decision": "all_promoted", **promotion_payload}
            return metadata, rendered_code, ""

        if is_within_delta(all_valid_mean, inference_valid_mean, self.config.keep_all_valid_delta, metric_name):
            metadata["final_result_source"] = "all_kept_within_delta"
            metadata["promotion"] = {"decision": "all_kept_within_delta", **promotion_payload}
            return metadata, rendered_code, ""

        if diversity_bonus and is_within_delta(all_valid_mean, inference_valid_mean, self.config.revert_all_valid_delta, metric_name):
            metadata["final_result_source"] = "all_kept_with_diversity_bonus"
            metadata["promotion"] = {"decision": "all_kept_with_diversity_bonus", **promotion_payload}
            return metadata, rendered_code, ""

        fallback_metadata = copy.deepcopy(inference_metadata)
        fallback_metadata.update(
            {
                "dataset": dataset,
                "subtask": subtask,
                "knowledge_type": "all",
                "run_tag": self.config.run_tag,
                "status": "success",
                "branch_metrics": metadata.get("branch_metrics", {}),
                "quality_gate": {
                    "mode": "validation_guard_reverted_to_inference",
                    "reason": "validation_guard_reverted_to_inference",
                },
                "runtime_snapshot": metadata.get("runtime_snapshot", {}),
                "finalist_report": metadata.get("finalist_report", {}),
                "promotion": {
                    "decision": "validation_guard_reverted_to_inference",
                    **promotion_payload,
                },
                "final_result_source": "inference_only",
                "candidate_all_result": {
                    "selected_rule_count": metadata.get("selected_rule_count", 0),
                    "compiled_rule_count": metadata.get("compiled_rule_count", 0),
                    "compiled_feature_names": metadata.get("compiled_feature_names", []),
                    "kept_feature_names": metadata.get("kept_feature_names", []),
                    "valid_mean": all_valid_mean,
                    "test_mean": all_test_mean,
                    "rules": metadata.get("rules", []),
                    "underdiverse_all": underdiverse_all,
                    "capacity_underfilled_all": capacity_underfilled,
                    "candidate_sets": metadata.get("branch_metrics", {}).get("combined", {}).get("candidate_sets", []),
                },
            }
        )
        fallback_code = rendered_code
        if os.path.exists(inference_code_path):
            with open(inference_code_path, "r", encoding="utf-8") as handle:
                fallback_code = handle.read()
        return (
            fallback_metadata,
            fallback_code,
            (
                f"[TrainEval] validation_guard_reverted_to_inference "
                f"(all_valid={all_valid_mean:.6f} inference_valid={inference_valid_mean:.6f} "
                f"improvement={valid_improvement:.6f} raw_gap={valid_gap_raw:.6f} "
                f"diversity_bonus={int(diversity_bonus)} "
                f"revert_delta={self.config.revert_all_valid_delta:.6f})"
            ),
        )


    def main(self):
        dataset, subtask = self.canonical_dataset_and_subtask(self.config.dataset, self.config.subtask)
        print(f"[TrainEval] dataset={dataset} subtask={subtask or '-'} knowledge_type={self.config.knowledge_type}")
        selected_rules: List[Dict] = []
        branch_metrics: Dict = {}
        prepare_audit: Dict = {"threshold_learned_rules": [], "dropped": []}
        prepared_rules: List[Dict] = []
        compiled_rules = []
        invalid_rules: List[Dict] = []
        feature_metrics: Dict = {}
        shrinkage_audit_payload: Dict | None = None
        second_variant_rule_ids: set[str] = set()
        finalist_report: Dict = {}
        plus_one_alias_report: Dict = {}
        lit_only_slot_control_report: Dict = {}
        spec = None
        y_train = y_valid = y_test = None
        null_model_scores = None
        try:
            selected_rules, branch_metrics = self.load_rules(dataset, subtask)
            gate_status, gate_reason, gate_mode = self.evaluate_quality_gate(branch_metrics)
            second_variant_rule_ids = self._collect_second_variant_rule_ids(branch_metrics)
            branch_metrics = dict(branch_metrics)
            branch_metrics["quality_gate"] = {
                "mode": gate_mode,
                "reason": gate_reason,
            }
            branch_metrics["strict_failure_diagnostics"] = self.build_branch_failure_diagnostics(
                branch_metrics,
                gate_mode=gate_mode,
                gate_reason=gate_reason,
            )
            print(f"[TrainEval] strict_failure_diagnostics={branch_metrics['strict_failure_diagnostics']}")
            if gate_status != "success":
                shrinkage_audit_payload = self.build_shrinkage_audit(
                    branch_metrics=branch_metrics,
                    selected_rules=selected_rules,
                    prepare_audit=prepare_audit,
                    prepared_rules=prepared_rules,
                    compiled_rules=compiled_rules,
                    invalid_rules=invalid_rules,
                    feature_metrics=feature_metrics,
                )
                self.save_skip_outputs(
                    dataset=dataset,
                    subtask=subtask,
                    branch_metrics=branch_metrics,
                    selected_rules=selected_rules,
                    prepared_rules=[],
                    compiled_rules=[],
                    invalid_rules=[],
                    reason=gate_reason,
                    status=gate_status,
                    shrinkage_audit=shrinkage_audit_payload,
                )
                return
            if self.config.knowledge_type == "all" and self.config.all_finalist_enable:
                combined_metrics = branch_metrics.get("combined", {}) or {}
                candidate_sets = list(combined_metrics.get("candidate_sets", []) or [])
                finalists = self.shortlist_all_finalists(candidate_sets)
                if finalists:
                    smiles_train, y_train, spec = self.load_split_xy(dataset, subtask=subtask, split="train")
                    smiles_valid, y_valid, _ = self.load_split_xy(dataset, subtask=subtask, split="valid")
                    finalist_reports = [
                        self.evaluate_candidate_set_valid_only(
                            dataset,
                            subtask,
                            spec,
                            smiles_train,
                            y_train,
                            smiles_valid,
                            y_valid,
                            finalist,
                        )
                        for finalist in finalists
                    ]
                    finalist_decision = self.choose_finalist_with_coverage_guard(finalist_reports)
                    chosen_finalist = finalist_decision.get("winner", {}) or {}
                    chosen_name = str(chosen_finalist.get("name", "") or "")
                    if chosen_name:
                        chosen_candidate = next(
                            (item for item in candidate_sets if str(item.get("name", "")) == chosen_name),
                            dict(chosen_finalist),
                        )
                        selected_rules = list(chosen_candidate.get("rules", []) or selected_rules)
                        chosen_report = next(
                            (item for item in finalist_reports if str(item.get("name", "")) == chosen_name),
                            dict(chosen_finalist),
                        )
                        branch_metrics = self._update_combined_metrics_with_finalist(
                            branch_metrics,
                            chosen_candidate,
                            chosen_report,
                            finalist_decision,
                        )
                        finalist_report = {
                            "enabled": True,
                            "config": {
                                "max_candidates": self.config.all_finalist_max_candidates,
                                "keep_lit_delta": self.config.all_finalist_keep_lit_delta,
                                "min_effective_literature_coverage": self.config.all_finalist_min_effective_lit,
                                "min_effective_data_coverage": self.config.all_finalist_min_effective_data,
                                "min_roots": self.config.all_finalist_min_roots,
                            },
                            "finalists": finalist_reports,
                            "selected_winner": chosen_name,
                            "selected_winner_report": chosen_report,
                            "decision": str(finalist_decision.get("decision", "") or ""),
                            "coverage_guard_applied": bool(finalist_decision.get("coverage_guard_applied", False)),
                            "best_valid_name": str(finalist_decision.get("best_valid_name", "") or ""),
                            "coverage_candidate_name": str(finalist_decision.get("coverage_candidate_name", "") or ""),
                            "valid_gap_to_best": float(finalist_decision.get("valid_gap_to_best", 0.0) or 0.0),
                        }
            print(f"[TrainEval] selected_rules_before_threshold_learning={len(selected_rules)}")
            prepared_rules, prepare_audit = self.prepare_rules_for_compilation_with_audit(
                selected_rules,
                dataset,
                subtask,
                second_variant_rule_ids=second_variant_rule_ids,
            )
            compiled_rules, invalid_rules = self.compile_rules(prepared_rules)
            print(f"[TrainEval] compiled_rules={len(compiled_rules)} invalid_rules={len(invalid_rules)}")

            if not selected_rules or not compiled_rules:
                shrinkage_audit_payload = self.build_shrinkage_audit(
                    branch_metrics=branch_metrics,
                    selected_rules=selected_rules,
                    prepare_audit=prepare_audit,
                    prepared_rules=prepared_rules,
                    compiled_rules=compiled_rules,
                    invalid_rules=invalid_rules,
                    feature_metrics=feature_metrics,
                )
                self.save_skip_outputs(
                    dataset=dataset,
                    subtask=subtask,
                    branch_metrics=branch_metrics,
                    selected_rules=selected_rules,
                    prepared_rules=prepared_rules,
                    compiled_rules=compiled_rules,
                    invalid_rules=invalid_rules,
                    reason=("No selected rules were available for compilation." if not selected_rules
                            else "No rules could be compiled into valid RDKit features."),
                    status="skipped",
                    shrinkage_audit=shrinkage_audit_payload,
                )
                return

            if spec is None or y_train is None:
                smiles_train, y_train, spec = self.load_split_xy(dataset, subtask=subtask, split="train")
            else:
                smiles_train, _, _ = self.load_split_xy(dataset, subtask=subtask, split="train")
            if y_valid is None:
                smiles_valid, y_valid, _ = self.load_split_xy(dataset, subtask=subtask, split="valid")
            else:
                smiles_valid, _, _ = self.load_split_xy(dataset, subtask=subtask, split="valid")
            smiles_test, y_test, _ = self.load_split_xy(dataset, subtask=subtask, split="test")
            null_model_scores = self.compute_null_model_scores(spec, y_train, y_valid, y_test)

            X_train, X_valid, X_test, kept_features, feature_metrics = self._build_feature_matrices(
                compiled_rules, smiles_train, smiles_valid, smiles_test,
            )
            self._update_feature_metrics(
                feature_metrics, selected_rules, prepared_rules, compiled_rules,
                kept_features, branch_metrics, second_variant_rule_ids,
            )
            shrinkage_audit_payload = self.build_shrinkage_audit(
                branch_metrics=branch_metrics,
                selected_rules=selected_rules,
                prepare_audit=prepare_audit,
                prepared_rules=prepared_rules,
                compiled_rules=compiled_rules,
                invalid_rules=invalid_rules,
                feature_metrics=feature_metrics,
            )
        except NoValidFeaturesError as exc:
            self.save_skip_outputs(
                dataset=dataset,
                subtask=subtask,
                branch_metrics=branch_metrics,
                selected_rules=selected_rules,
                prepared_rules=prepared_rules,
                compiled_rules=compiled_rules,
                invalid_rules=invalid_rules,
                reason=str(exc),
                status="skipped",
                null_model_scores=null_model_scores,
                shrinkage_audit=shrinkage_audit_payload,
            )
            return
        except Exception as exc:
            if null_model_scores is None and spec is not None and y_train is not None and y_valid is not None and y_test is not None:
                null_model_scores = self.compute_null_model_scores(spec, y_train, y_valid, y_test)
            self.save_skip_outputs(
                dataset=dataset,
                subtask=subtask,
                branch_metrics=branch_metrics,
                selected_rules=selected_rules,
                prepared_rules=prepared_rules,
                compiled_rules=compiled_rules,
                invalid_rules=invalid_rules,
                reason=f"{type(exc).__name__}: {exc}",
                status="failed",
                null_model_scores=null_model_scores,
                shrinkage_audit=shrinkage_audit_payload,
            )
            raise

        print(
            f"[TrainEval] feature_count_before={feature_metrics['feature_count_before_filter']} "
            f"feature_count_after={feature_metrics['feature_count_after_filter']} "
            f"non_nan_coverage={feature_metrics['non_nan_coverage']:.3f}"
        )
        if prepared_rules:
            print("[TrainEval] rule preview:")
            for idx, rule in enumerate(prepared_rules[:8], 1):
                threshold = rule.get("threshold")
                threshold_text = ""
                if threshold:
                    threshold_text = f" {threshold['op']} {threshold['value']}"
                print(
                    f"  - {idx}. {rule['feature_family']}::{rule['feature_name']}{threshold_text} "
                    f"| source={rule['source']}"
                )

        try:
            runtime_snapshot = self.load_runtime_snapshot()
            if spec.task_type == "classification":
                valid_scores, test_scores = self.evaluate_classification(dataset, subtask, X_train, y_train, X_valid, y_valid, X_test, y_test)
            elif dataset == "qm9":
                valid_scores, test_scores = self.evaluate_qm9(X_train, y_train, X_valid, y_valid, X_test, y_test)
            else:
                valid_scores, test_scores = self.evaluate_regression_rf(dataset, X_train, y_train, X_valid, y_valid, X_test, y_test)

            ablation_summary = self.compute_rule_ablation_summary(
                dataset,
                subtask,
                spec,
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
                kept_features,
            )

            (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                plus_one_rendered_code,
                plus_one_function_names,
                plus_one_alias_report,
            ) = self.maybe_apply_plus_one_alias_extension(
                dataset,
                subtask,
                spec,
                branch_metrics,
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                smiles_train,
                y_train,
                smiles_valid,
                y_valid,
                smiles_test,
                y_test,
            )

            (
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                lit_only_rendered_code,
                lit_only_function_names,
                lit_only_slot_control_report,
            ) = self.maybe_apply_lit_only_slot_control(
                dataset,
                subtask,
                spec,
                branch_metrics,
                selected_rules,
                prepared_rules,
                compiled_rules,
                invalid_rules,
                feature_metrics,
                valid_scores,
                test_scores,
                ablation_summary,
                shrinkage_audit_payload,
                second_variant_rule_ids,
                smiles_train,
                y_train,
                smiles_valid,
                y_valid,
                smiles_test,
                y_test,
            )

            code_path, metadata_path = self.code_output_paths(dataset, subtask)
            if lit_only_rendered_code is not None and lit_only_function_names is not None:
                rendered_code, function_names = lit_only_rendered_code, lit_only_function_names
            elif plus_one_rendered_code is not None and plus_one_function_names is not None:
                rendered_code, function_names = plus_one_rendered_code, plus_one_function_names
            else:
                rendered_code, function_names = self.render_rule_module(prepared_rules, compiled_rules)
            metadata = {
                "dataset": dataset,
                "subtask": subtask,
                "knowledge_type": self.config.knowledge_type,
                "run_tag": self.config.run_tag,
                "status": "success",
                "selected_rule_count": len(selected_rules),
                "compiled_rule_count": len(compiled_rules),
                "invalid_rule_count": len(invalid_rules),
                "compiled_feature_names": function_names,
                "kept_feature_names": kept_features,
                "branch_metrics": branch_metrics,
                "strict_failure_diagnostics": branch_metrics.get("strict_failure_diagnostics", {}),
                "quality_gate": {
                    "mode": branch_metrics.get("quality_gate", {}).get("mode", "strict_pass"),
                    "reason": branch_metrics.get("quality_gate", {}).get("reason", ""),
                },
                "runtime_snapshot": runtime_snapshot,
                "feature_metrics": feature_metrics,
                "null_model_scores": null_model_scores,
                "ablation_summary": ablation_summary,
                "valid_scores": valid_scores,
                "test_scores": test_scores,
                "valid_mean": float(np.mean(valid_scores)),
                "test_mean": float(np.mean(test_scores)),
                "rules": prepared_rules,
                "shrinkage_audit": shrinkage_audit_payload,
                "finalist_report": finalist_report,
                "plus_one_alias_report": plus_one_alias_report,
                "lit_only_slot_control_report": lit_only_slot_control_report,
                "lit_only_slot_control_applied": bool(lit_only_slot_control_report.get("accepted", False)),
                "lit_only_slot_control_dropped_rule_id": str(lit_only_slot_control_report.get("candidate_rule_id", "") or ""),
                "lit_only_slot_control_dropped_feature_name": str(
                    lit_only_slot_control_report.get("candidate_feature_name", "") or ""
                ),
            }
            metadata, rendered_code, promotion_message = self.maybe_apply_validation_promotion_guard(
                dataset,
                subtask,
                metadata,
                rendered_code,
            )
            with open(code_path, "w", encoding="utf-8") as handle:
                handle.write(rendered_code)
            self.save_rule_artifact(metadata_path, metadata)
            self.write_unified_outcome_artifact(dataset, subtask, metadata_path, code_path, metadata)
            if shrinkage_audit_payload is not None:
                self.save_shrinkage_audit(dataset, subtask, shrinkage_audit_payload)
            final_valid_scores = metadata.get("valid_scores", valid_scores)
            final_test_scores = metadata.get("test_scores", test_scores)
            self.save_metric_report(dataset, subtask, final_valid_scores, final_test_scores)
            if promotion_message:
                print(promotion_message)
        except Exception as exc:
            self.save_skip_outputs(
                dataset=dataset,
                subtask=subtask,
                branch_metrics=branch_metrics,
                selected_rules=selected_rules,
                prepared_rules=prepared_rules,
                compiled_rules=compiled_rules,
                invalid_rules=invalid_rules,
                reason=f"{type(exc).__name__}: {exc}",
                status="failed",
                null_model_scores=null_model_scores,
                shrinkage_audit=shrinkage_audit_payload,
            )
            raise
        print(
            f"[TrainEval] valid_mean={float(np.mean(final_valid_scores)):.6f} "
            f"test_mean={float(np.mean(final_test_scores)):.6f}"
        )
        print(
            f"[TrainEval] null_valid={null_model_scores['valid_mean']:.6f} "
            f"null_test={null_model_scores['test_mean']:.6f}"
        )
        if ablation_summary:
            print("[TrainEval] top ablation deltas:")
            ranked = sorted(ablation_summary, key=lambda item: abs(float(item["test_delta"])), reverse=True)
            for item in ranked[:5]:
                print(
                    f"  - {item['feature']} | valid_delta={float(item['valid_delta']):.6f} "
                    f"| test_delta={float(item['test_delta']):.6f}"
                )
        print(f"Compiled {len(compiled_rules)} rules and wrote code to {code_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bbbp", help="dataset/task name")
    parser.add_argument("--dataset_root", type=str, default=None, help="fixed split root; defaults to the CLI environment or data/splits")
    parser.add_argument("--subtask", type=str, default="", help="subtask of tox21/sider/qm9 dataset")
    parser.add_argument("--model", type=str, default="qwen3.5-9b", help="LLM run tag")
    parser.add_argument("--knowledge_type", type=str, default="all", help="synthesize/inference/all")
    parser.add_argument("--list_num", type=int, default=30, help="number of sample batches for inference")
    parser.add_argument("--output_dir", type=str, default="eval_result", help="evaluation folder")
    parser.add_argument("--code_gen_folder", type=str, default="eval_code_generation_repo", help="compiled code output folder")
    parser.add_argument("--target_rule_count", type=int, default=30, help="max rule budget")
    parser.add_argument("--run_tag", type=str, default="", help="optional run tag for segregated artifacts")
    parser.add_argument("--generation_seed", type=int, default=None, help="generation seed recorded in outcome metadata")
    parser.add_argument("--max_ablation_features", type=int, default=10, help="max number of kept features to run single-feature ablation on")
    return parser


def main(argv: List[str] | None = None) -> None:
    """CLI entry point; resolve the environment once, then run an isolated instance."""
    parser = build_parser()
    namespace = parser.parse_args(argv)
    try:
        config = EvaluationConfig.from_namespace(namespace, environment=os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    EvaluationPipeline(config).main()


if __name__ == "__main__":
    main()
