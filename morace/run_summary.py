from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from morace.experiment_io import OUTCOME_SCHEMA_VERSION, VALID_OUTCOME_CATEGORIES
from morace.model_backend import normalize_model_name
from morace.paths import with_run_tag


KNOWLEDGE_TYPES = ("synthesize", "inference", "all")
SUCCESS_STATUSES = {"success", "success_primary", "success_fallback"}
REQUIRED_OUTCOME_FIELDS = {
    "schema_version",
    "outcome_category",
    "final_result_source",
    "metric_name",
    "generation_seed",
    "actual_valid_mean",
    "actual_test_mean",
    "selected_rule_count",
    "compiled_rule_count",
    "invalid_rule_count",
}
SUMMARY_SCHEMA_VERSION = "run_summary_v1_2"


def _relative_metadata_path(
    run_tag: str,
    model_tag: str,
    dataset: str,
    subtask: str,
    batch_num: int,
    knowledge_type: str,
) -> Path:
    folder = Path(with_run_tag("eval_code_generation_repo", run_tag)) / model_tag / dataset / knowledge_type
    if knowledge_type in {"inference", "all"}:
        folder /= f"sample_{batch_num}"
    suffix = f"_{subtask}" if subtask else ""
    if dataset == "qm9" and subtask:
        filename = f"{model_tag}_qm9_{subtask}_{knowledge_type}_rules.json"
    else:
        filename = f"{model_tag}_{dataset}{suffix}_{knowledge_type}_rules.json"
    return folder / filename


def _relative_outcome_path(
    metadata_path: Path,
    model_tag: str,
    dataset: str,
    subtask: str,
    knowledge_type: str,
) -> Path:
    if not subtask:
        return metadata_path.parent / "outcome.json"
    if dataset == "qm9":
        filename = f"{model_tag}_qm9_{subtask}_{knowledge_type}_outcome.json"
    else:
        filename = f"{model_tag}_{dataset}_{subtask}_{knowledge_type}_outcome.json"
    return metadata_path.parent / filename


def _read_object(path: Path) -> tuple[dict[str, Any] | None, str, str]:
    if not path.exists():
        return None, "missing", f"missing artifact: {path.as_posix()}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, "invalid", f"invalid artifact {path.as_posix()}: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, "invalid", f"invalid artifact {path.as_posix()}: expected a JSON object"
    return payload, "valid", ""


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _result_summary(
    relative_path: Path,
    outcome: dict[str, Any] | None,
    artifact_status: str,
    artifact_reason: str,
    fallback_seed: int | None,
) -> dict[str, Any]:
    outcome = outcome or {}
    generation_seed = _optional_int(outcome.get("generation_seed"))
    generation_seed_source = "outcome"
    if generation_seed is None:
        generation_seed = fallback_seed
        generation_seed_source = "run_config" if fallback_seed is not None else "missing"
    return {
        "artifact_status": artifact_status,
        "artifact_reason": artifact_reason,
        "outcome_path": relative_path.as_posix(),
        "schema_version": outcome.get("schema_version"),
        "outcome_category": outcome.get("outcome_category"),
        "final_result_source": outcome.get("final_result_source"),
        "metric_name": outcome.get("metric_name"),
        "generation_seed": generation_seed,
        "generation_seed_source": generation_seed_source,
        "actual_valid_mean": outcome.get("actual_valid_mean"),
        "actual_test_mean": outcome.get("actual_test_mean"),
        "selected_rule_count": outcome.get("selected_rule_count"),
        "compiled_rule_count": outcome.get("compiled_rule_count"),
        "invalid_rule_count": outcome.get("invalid_rule_count"),
    }


def _validate_outcome(
    relative_path: Path,
    outcome: dict[str, Any] | None,
    artifact_status: str,
    artifact_reason: str,
    fallback_seed: int | None,
) -> tuple[str, str]:
    if artifact_status != "valid" or outcome is None:
        return artifact_status, artifact_reason
    missing_fields = sorted(REQUIRED_OUTCOME_FIELDS.difference(outcome))
    if missing_fields:
        return "invalid", f"invalid artifact {relative_path.as_posix()}: missing fields {', '.join(missing_fields)}"
    if outcome.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        return (
            "invalid",
            f"invalid artifact {relative_path.as_posix()}: unsupported outcome schema "
            f"{outcome.get('schema_version')!r}",
        )
    outcome_category = str(outcome.get("outcome_category") or "")
    if outcome_category not in VALID_OUTCOME_CATEGORIES:
        return (
            "invalid",
            f"invalid artifact {relative_path.as_posix()}: unsupported outcome category "
            f"{outcome_category!r}",
        )
    raw_seed = outcome.get("generation_seed")
    outcome_seed = _optional_int(raw_seed)
    if raw_seed is not None and raw_seed != "" and outcome_seed is None:
        return "invalid", f"invalid artifact {relative_path.as_posix()}: generation seed is invalid"
    if outcome_seed is None and fallback_seed is None:
        return "invalid", f"invalid artifact {relative_path.as_posix()}: generation seed is missing"
    if outcome_seed is not None and fallback_seed is not None and outcome_seed != fallback_seed:
        return (
            "invalid",
            f"invalid artifact {relative_path.as_posix()}: generation seed differs from run config "
            f"({outcome_seed} != {fallback_seed})",
        )
    if outcome_category in {"success_primary", "success_fallback"}:
        if outcome.get("metric_name") in {None, ""} or outcome.get("actual_test_mean") is None:
            return "invalid", f"invalid artifact {relative_path.as_posix()}: successful outcome is missing its metric"
    return "valid", ""


def _task_status(
    exec_status: str,
    branch_statuses: dict[str, str],
    result_statuses: dict[str, str],
    provenance_status: str,
) -> str:
    if exec_status != "success":
        return exec_status
    if provenance_status != "valid":
        return "incomplete"
    statuses = list(branch_statuses.values())
    if any(status in {"missing", "unknown"} for status in statuses):
        return "incomplete"
    if any(status != "valid" for status in result_statuses.values()):
        return "incomplete"
    if all(status in SUCCESS_STATUSES for status in statuses):
        return "success"
    if any(status in {"failed", "degraded", "killed"} for status in statuses):
        return "degraded"
    if all(status == "skipped" for status in statuses):
        return "skipped"
    if any(status == "skipped" for status in statuses):
        return "partial"
    return "unknown"


def append_task_summary(
    *,
    summary_path: str,
    run_tag: str,
    model: str,
    dataset: str,
    subtask: str,
    batch_num: int,
    exec_status: str,
    exit_code: int,
    log_path: str,
    reason: str = "",
    project_root: Path | None = None,
) -> str:
    root = Path(project_root or Path.cwd())
    summary_file = root / summary_path
    model_tag = normalize_model_name(model)

    run_config_path = root / "artifacts" / run_tag / "run_config.json"
    run_config, run_config_status, run_config_reason = _read_object(run_config_path)
    run_config = run_config or {}
    fallback_seed = _optional_int(run_config.get("global_seed"))
    task_label = f"{dataset}:{subtask}" if subtask else dataset
    provenance = (run_config.get("dataset_provenance", {}) or {}).get(task_label)
    if not isinstance(provenance, dict):
        provenance = None
        provenance_status = "missing"
        provenance_reason = f"missing dataset provenance for {task_label}"
    elif provenance.get("status") != "valid":
        provenance_status = "invalid"
        provenance_reason = str(provenance.get("reason") or "dataset provenance is invalid")
    else:
        provenance_status = "valid"
        provenance_reason = ""

    branch_statuses: dict[str, str] = {}
    branch_details: dict[str, dict[str, Any]] = {}
    result_statuses: dict[str, str] = {}
    result_problems: list[str] = []
    for knowledge_type in KNOWLEDGE_TYPES:
        relative_metadata = _relative_metadata_path(
            run_tag,
            model_tag,
            dataset,
            subtask,
            int(batch_num),
            knowledge_type,
        )
        metadata, metadata_status, metadata_reason = _read_object(root / relative_metadata)
        metadata = metadata or {}
        branch_status = str(metadata.get("status", "") or "").strip()
        if metadata_status == "missing":
            branch_status = "missing"
        elif metadata_status == "invalid" or not branch_status:
            branch_status = "unknown"
        branch_statuses[knowledge_type] = branch_status

        branch_metrics = metadata.get("branch_metrics", {}) or {}
        projection_metrics = (
            branch_metrics.get("synthesize_projection")
            or branch_metrics.get("inference_projection")
            or metadata.get("projection_metrics", {})
            or {}
        )
        relative_outcome = _relative_outcome_path(
            relative_metadata,
            model_tag,
            dataset,
            subtask,
            knowledge_type,
        )
        outcome, outcome_status, outcome_reason = _read_object(root / relative_outcome)
        outcome_status, outcome_reason = _validate_outcome(
            relative_outcome,
            outcome,
            outcome_status,
            outcome_reason,
            fallback_seed,
        )
        result_statuses[knowledge_type] = outcome_status
        if outcome_status != "valid":
            result_problems.append(outcome_reason)
        branch_details[knowledge_type] = {
            "status": branch_status,
            "reason": metadata.get("skip_reason", "") or metadata_reason,
            "strict_projector_valid_rate": projection_metrics.get("strict_projector_valid_rate"),
            "repair_used_rate": projection_metrics.get("repair_used_rate"),
            "result": _result_summary(
                relative_outcome,
                outcome,
                outcome_status,
                outcome_reason,
                fallback_seed,
            ),
        }

    task_status = _task_status(exec_status, branch_statuses, result_statuses, provenance_status)
    summary_reason = reason
    if provenance_reason and not summary_reason:
        summary_reason = provenance_reason
    if result_problems and not summary_reason:
        summary_reason = "; ".join(result_problems)

    payload: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "run_tag": run_tag,
        "tasks": [],
        "counts": {},
    }
    if summary_file.exists():
        with summary_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    payload["schema_version"] = SUMMARY_SCHEMA_VERSION
    tasks = [
        item
        for item in payload.get("tasks", [])
        if not (item.get("dataset") == dataset and item.get("subtask", "") == subtask)
    ]
    tasks.append(
        {
            "dataset": dataset,
            "subtask": subtask,
            "batch_num": int(batch_num),
            "task_label": task_label,
            "status": task_status,
            "pipeline_exit_status": exec_status,
            "exit_code": int(exit_code),
            "log_path": log_path,
            "reason": summary_reason,
            "actual_projector_model": (run_config.get("resolved_config", {}) or {}).get("projector_model", ""),
            "actual_prompt_hash": (run_config.get("prompt_hashes", {}) or {}).get("inference_prompt.json", ""),
            "run_config_status": run_config_status,
            "run_config_reason": run_config_reason,
            "run_config_schema_version": run_config.get("schema_version"),
            "dataset_provenance_status": provenance_status,
            "dataset_provenance_reason": provenance_reason,
            "dataset_provenance": provenance,
            "branch_statuses": branch_statuses,
            "branch_details": branch_details,
            "result": branch_details["all"]["result"],
        }
    )
    payload["tasks"] = tasks
    counts: dict[str, int] = {}
    for item in tasks:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    payload["counts"] = counts

    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return task_status


def main() -> int:
    parser = argparse.ArgumentParser(description="Append one completed task to a MoRACE run summary.")
    parser.add_argument("--summary-path", required=True)
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subtask", default="")
    parser.add_argument("--batch-num", type=int, required=True)
    parser.add_argument("--exec-status", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--reason", default="")
    args = parser.parse_args()
    status = append_task_summary(
        summary_path=args.summary_path,
        run_tag=args.run_tag,
        model=args.model,
        dataset=args.dataset,
        subtask=args.subtask,
        batch_num=args.batch_num,
        exec_status=args.exec_status,
        exit_code=args.exit_code,
        log_path=args.log_path,
        reason=args.reason,
    )
    print(status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
