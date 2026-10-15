from __future__ import annotations

import json
import os
from math import sqrt
from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTCOME_SCHEMA_VERSION = "outcome_v1_1"
VALID_OUTCOME_CATEGORIES = frozenset(
    {"success_primary", "success_fallback", "degraded", "failed", "killed", "skipped", "partial"}
)

RESULTS_MASTER_FIELDS = [
    "system_id",
    "system_label",
    "experiment_id",
    "run_tag",
    "run_root",
    "outcome_path",
    "budget_report_path",
    "dataset",
    "subtask",
    "task_type",
    "metric_name",
    "generation_seed",
    "rf_seeds",
    "rf_seed_count",
    "outcome_category",
    "final_result_source",
    "fallback_mode",
    "actual_valid_mean",
    "actual_test_mean",
    "null_model_valid_mean",
    "null_model_test_mean",
    "itt_valid_mean",
    "itt_test_mean",
    "budget_available",
    "total_tokens",
    "token_count_source",
    "api_call_count",
    "stage1_total_calls",
    "stage2_total_calls",
    "stage1_retry_calls",
    "stage2_retry_calls",
    "selected_rule_count",
    "compiled_rule_count",
    "rendered_code_available",
    "protocol_version",
    "schema_version",
]


def _safe_float(value: Any):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def compute_common_null_model_scores(task_type: str, dataset: str, y_train, y_valid, y_test) -> Dict[str, Any]:
    kind = str(task_type or "").lower()
    if "class" in kind:
        return {
            "valid_mean": 0.5,
            "test_mean": 0.5,
            "strategy": "constant_auc_0.5",
        }
    baseline = float(np.nanmean(np.asarray(y_train, dtype=float)))
    if str(dataset or "").lower() == "qm9":
        valid_score = float(mean_absolute_error(y_valid, np.full(len(y_valid), baseline)))
        test_score = float(mean_absolute_error(y_test, np.full(len(y_test), baseline)))
        metric_name = "mae"
    else:
        valid_score = float(sqrt(mean_squared_error(y_valid, np.full(len(y_valid), baseline))))
        test_score = float(sqrt(mean_squared_error(y_test, np.full(len(y_test), baseline))))
        metric_name = "rmse"
    return {
        "valid_mean": valid_score,
        "test_mean": test_score,
        "baseline_value": baseline,
        "strategy": f"train_mean_{metric_name}",
    }


def infer_fallback_mode(final_result_source: str) -> str:
    source = str(final_result_source or "")
    if source in {"inference_only", "inference_only_fallback"}:
        return source
    return ""


def normalize_outcome_category(status: str, final_result_source: str = "", fallback_mode: str = "") -> str:
    normalized = str(status or "").lower()
    if normalized == "success":
        return "success_fallback" if (fallback_mode or infer_fallback_mode(final_result_source)) else "success_primary"
    if normalized in {"degraded", "failed", "killed", "skipped"}:
        return normalized
    if normalized == "success_full":
        return "success_primary"
    return normalized or "unknown"


def build_outcome_payload(
    *,
    system_id: str,
    dataset: str,
    subtask: str,
    generation_seed: int | None,
    rf_seeds,
    status: str,
    final_result_source: str,
    fallback_mode: str = "",
    quality_gate_mode: str = "",
    quality_gate_reason: str = "",
    experiment_id: str | None = None,
    run_tag: str | None = None,
    task_type: str | None = None,
    metric_name: str | None = None,
    actual_valid_mean=None,
    actual_test_mean=None,
    null_model_valid_mean=None,
    null_model_test_mean=None,
    selected_rule_count: int = 0,
    compiled_rule_count: int = 0,
    invalid_rule_count: int = 0,
    rendered_code_available: bool | None = None,
    artifact_paths: Dict[str, Any] | None = None,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    fallback = fallback_mode or infer_fallback_mode(final_result_source)
    outcome_category = normalize_outcome_category(status, final_result_source, fallback)
    actual_test = _safe_float(actual_test_mean)
    null_test = _safe_float(null_model_test_mean)
    actual_valid = _safe_float(actual_valid_mean)
    null_valid = _safe_float(null_model_valid_mean)
    payload = {
        "protocol_version": "experiment_protocol_v2_1",
        "schema_version": OUTCOME_SCHEMA_VERSION,
        "system_id": system_id,
        "dataset": dataset,
        "subtask": subtask,
        "experiment_id": experiment_id,
        "run_tag": run_tag,
        "task_type": task_type,
        "metric_name": metric_name,
        "generation_seed": generation_seed,
        "rf_seeds": list(rf_seeds or []),
        "outcome_category": outcome_category,
        "final_result_source": str(final_result_source or ""),
        "fallback_mode": fallback or None,
        "quality_gate_mode": str(quality_gate_mode or ""),
        "quality_gate_reason": str(quality_gate_reason or ""),
        "has_actual_metric": actual_test is not None,
        "actual_valid_mean": actual_valid,
        "actual_test_mean": actual_test,
        "null_model_valid_mean": null_valid,
        "null_model_test_mean": null_test,
        "itt_valid_mean": actual_valid if outcome_category in {"success_primary", "success_fallback"} and actual_valid is not None else null_valid,
        "itt_test_mean": actual_test if outcome_category in {"success_primary", "success_fallback"} and actual_test is not None else null_test,
        "selected_rule_count": int(selected_rule_count),
        "compiled_rule_count": int(compiled_rule_count),
        "invalid_rule_count": int(invalid_rule_count),
        "rendered_code_available": rendered_code_available,
        "artifact_paths": artifact_paths or {},
    }
    if extra:
        payload.update(extra)
    return payload


def build_budget_report_payload(
    *,
    system_id: str,
    branch_name: str,
    dataset: str,
    run_tag: str,
    experiment_id: str | None = None,
    model_id: str,
    model_path: str = "",
    projector_model_id: str = "",
    projector_model_path: str = "",
    budget_available: bool = True,
    total_tokens: int | None = None,
    token_count_source: str = "",
    api_call_count: int | None = None,
    stage1_total_calls: int | None = None,
    stage2_total_calls: int | None = None,
    stage1_retry_calls: int | None = None,
    stage2_retry_calls: int | None = None,
    call_counts: Dict[str, Any] | None = None,
    token_counts: Dict[str, Any] | None = None,
    prompt_lengths: Dict[str, Any] | None = None,
    generation_caps: Dict[str, Any] | None = None,
    notes: str = "",
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    payload = {
        "protocol_version": "experiment_protocol_v2_1",
        "schema_version": "budget_report_v1_1",
        "system_id": system_id,
        "branch_name": branch_name,
        "dataset": dataset,
        "run_tag": run_tag,
        "experiment_id": experiment_id,
        "model_id": model_id,
        "model_path": model_path,
        "projector_model_id": projector_model_id,
        "projector_model_path": projector_model_path,
        "budget_available": bool(budget_available),
        "total_tokens": total_tokens,
        "token_count_source": token_count_source or None,
        "api_call_count": api_call_count,
        "stage1_total_calls": stage1_total_calls,
        "stage2_total_calls": stage2_total_calls,
        "stage1_retry_calls": stage1_retry_calls,
        "stage2_retry_calls": stage2_retry_calls,
        "call_counts": call_counts or {},
        "token_counts": token_counts or {},
        "prompt_lengths": prompt_lengths or {},
        "generation_caps": generation_caps or {},
        "notes": notes,
    }
    if extra:
        payload.update(extra)
    return payload


def write_budget_report_json(path: str, payload: Dict[str, Any]) -> None:
    write_json(path, payload)
