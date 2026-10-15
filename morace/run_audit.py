from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from morace.artifact_inputs import resolve_branch_artifacts
from morace.dataset_provenance import verify_recorded_provenance
from morace.experiment_io import OUTCOME_SCHEMA_VERSION, VALID_OUTCOME_CATEGORIES


AUDIT_SCHEMA_VERSION = "run_audit_v1"
CURRENT_SUMMARY_SCHEMA = "run_summary_v1_2"
LEGACY_SUMMARY_SCHEMAS = {"run_summary_v1_1"}
CURRENT_RUN_CONFIG_SCHEMA = "run_config_v1_1"
OUTCOME_COMPARE_FIELDS = (
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
)


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    if not path.is_file():
        return None, f"missing JSON artifact: {path}"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid JSON artifact {path}: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict):
        return None, f"invalid JSON artifact {path}: expected an object"
    return payload, None


def _resolve_artifact_dir(run_path: Path) -> Path:
    path = run_path.resolve()
    if path.is_file():
        if path.name != "summary.json":
            raise ValueError(f"run path must be an artifact directory or summary.json: {path}")
        return path.parent
    return path


def _resolve_project_root(artifact_dir: Path, project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.resolve()
    if artifact_dir.parent.name == "artifacts":
        return artifact_dir.parent.parent.resolve()
    return Path.cwd().resolve()


def _resolve_artifact_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _audit_outcome(
    result: dict[str, Any],
    *,
    label: str,
    project_root: Path,
    expected_seed: int | None,
    errors: list[str],
) -> Path | None:
    if result.get("artifact_status") != "valid":
        errors.append(
            f"{label}: outcome status is {result.get('artifact_status')!r}: "
            f"{result.get('artifact_reason', '')}"
        )
        return
    outcome_value = str(result.get("outcome_path") or "")
    if not outcome_value:
        errors.append(f"{label}: outcome path is missing")
        return
    outcome_path = _resolve_artifact_path(outcome_value, project_root)
    outcome, problem = _read_json(outcome_path)
    if problem:
        errors.append(f"{label}: {problem}")
        return
    assert outcome is not None
    missing_fields = [field for field in OUTCOME_COMPARE_FIELDS if field not in outcome]
    if missing_fields:
        errors.append(f"{label}: outcome is missing required fields: {', '.join(missing_fields)}")
    if outcome.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        errors.append(f"{label}: unsupported outcome schema: {outcome.get('schema_version')!r}")
    outcome_category = str(outcome.get("outcome_category") or "")
    if outcome_category not in VALID_OUTCOME_CATEGORIES:
        errors.append(f"{label}: unsupported outcome category: {outcome_category!r}")
    if outcome_category in {"success_primary", "success_fallback"} and (
        outcome.get("metric_name") in {None, ""} or outcome.get("actual_test_mean") is None
    ):
        errors.append(f"{label}: successful outcome is missing its metric")
    for field in ("actual_valid_mean", "actual_test_mean"):
        value = outcome.get(field)
        if value is not None:
            try:
                valid_number = not isinstance(value, bool) and math.isfinite(float(value))
            except (TypeError, ValueError, OverflowError):
                valid_number = False
            if not valid_number:
                errors.append(f"{label}: outcome {field} must be finite or null")
    outcome_seed = _optional_int(outcome.get("generation_seed"))
    if outcome_seed is not None and expected_seed is not None and outcome_seed != expected_seed:
        errors.append(
            f"{label}: generation seed differs from run config: {outcome_seed} != {expected_seed}"
        )
    for field in OUTCOME_COMPARE_FIELDS:
        if (
            field == "generation_seed"
            and outcome.get(field) is None
            and result.get("generation_seed_source") == "run_config"
        ):
            continue
        if outcome.get(field) != result.get(field):
            errors.append(
                f"{label}: outcome field {field} differs from summary: "
                f"{outcome.get(field)!r} != {result.get(field)!r}"
            )
    return outcome_path


def _artifact_snapshot(paths: set[Path]) -> dict[str, str]:
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths) if path.is_file()}


def audit_run(
    run_path: Path | str,
    *,
    project_root: Path | None = None,
    require_success: bool = False,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    artifact_dir = _resolve_artifact_dir(Path(run_path))
    root = _resolve_project_root(artifact_dir, project_root)
    errors: list[str] = []
    warnings: list[str] = []
    task_reports: list[dict[str, Any]] = []
    final_artifact_paths: set[Path] = set()

    summary, problem = _read_json(artifact_dir / "summary.json")
    if problem:
        errors.append(problem)
        summary = {}
    config, problem = _read_json(artifact_dir / "run_config.json")
    if problem:
        errors.append(problem)
        config = {}
    summary = summary or {}
    config = config or {}

    summary_schema = summary.get("schema_version")
    legacy = summary_schema in LEGACY_SUMMARY_SCHEMAS
    if summary_schema != CURRENT_SUMMARY_SCHEMA:
        message = f"unsupported summary schema: {summary_schema!r}"
        if legacy and allow_legacy:
            warnings.append(f"legacy summary accepted without provenance guarantees: {summary_schema}")
        else:
            errors.append(message)
    if not legacy and config.get("schema_version") != CURRENT_RUN_CONFIG_SCHEMA:
        errors.append(f"unsupported run config schema: {config.get('schema_version')!r}")
    if summary.get("run_tag") and config.get("run_tag"):
        if summary.get("run_tag") != config.get("run_tag"):
            errors.append(
                f"run tag mismatch: summary={summary.get('run_tag')!r} "
                f"config={config.get('run_tag')!r}"
            )

    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        errors.append("summary contains no tasks")
        tasks = []
    config_provenance = config.get("dataset_provenance") or {}
    expected_seed = _optional_int(config.get("global_seed"))
    computed_counts: dict[str, int] = {}
    for index, task in enumerate(tasks):
        if not isinstance(task, dict):
            errors.append(f"task {index}: expected an object")
            continue
        label = str(task.get("task_label") or f"task {index}")
        status = str(task.get("status") or "unknown")
        computed_counts[status] = computed_counts.get(status, 0) + 1
        task_errors_before = len(errors)
        task_warnings_before = len(warnings)
        if task.get("pipeline_exit_status") != "success":
            errors.append(f"{label}: pipeline exit status is {task.get('pipeline_exit_status')!r}")
        if status != "success":
            message = f"{label}: scientific status is {status!r}"
            if require_success:
                errors.append(message)
            else:
                warnings.append(message)

        log_value = str(task.get("log_path") or "")
        if not log_value:
            errors.append(f"{label}: log path is missing")
        elif not _resolve_artifact_path(log_value, root).is_file():
            errors.append(f"{label}: log artifact is missing: {log_value}")

        provenance = task.get("dataset_provenance")
        if not isinstance(provenance, dict):
            if legacy and allow_legacy:
                warnings.append(f"{label}: legacy task has no dataset provenance")
            else:
                errors.append(f"{label}: dataset provenance is missing")
        else:
            stored = config_provenance.get(label)
            if stored != provenance:
                errors.append(f"{label}: summary provenance differs from run config")
            errors.extend(verify_recorded_provenance(provenance, project_root=root))
            verification_status = (provenance.get("manifest") or {}).get("verification_status")
            if verification_status != "verified":
                warnings.append(
                    f"{label}: dataset manifest verification status is {verification_status!r}"
                )

        branch_details = task.get("branch_details")
        if not isinstance(branch_details, dict):
            errors.append(f"{label}: branch details are missing")
        else:
            for branch in ("synthesize", "inference", "all"):
                detail = branch_details.get(branch)
                if not isinstance(detail, dict):
                    errors.append(f"{label}/{branch}: branch detail is missing")
                    continue
                result = detail.get("result")
                if not isinstance(result, dict):
                    errors.append(f"{label}/{branch}: result summary is missing")
                    continue
                outcome_path = _audit_outcome(
                    result,
                    label=f"{label}/{branch}",
                    project_root=root,
                    expected_seed=expected_seed,
                    errors=errors,
                )
                if outcome_path is not None:
                    final_artifact_paths.add(outcome_path)
                if result.get("outcome_category") in {"success_primary", "success_fallback"}:
                    try:
                        final_artifacts = resolve_branch_artifacts(
                            task, branch, str(summary.get("run_tag") or config.get("run_tag") or ""),
                            root, require_code=True,
                        )
                        final_artifact_paths.update(final_artifacts[key] for key in ("rules_path", "code_path"))
                    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
                        errors.append(f"{label}/{branch}: final artifact chain: {exc}")
        if isinstance(task.get("result"), dict) and isinstance(branch_details, dict):
            all_detail = branch_details.get("all") or {}
            if task["result"] != all_detail.get("result"):
                errors.append(f"{label}: primary result differs from the all-branch result")

        task_reports.append(
            {
                "task_label": label,
                "status": status,
                "error_count": len(errors) - task_errors_before,
                "warning_count": len(warnings) - task_warnings_before,
            }
        )

    if summary.get("counts") != computed_counts:
        errors.append(
            f"summary counts mismatch: {summary.get('counts')!r} != {computed_counts!r}"
        )

    artifact_hashes = _artifact_snapshot(final_artifact_paths)
    # finish_run stores this first snapshot in audit.json before completion.json.
    # Later audits of completed runs must verify it rather than silently trust
    # a new snapshot of files that may have changed since completion.
    if (artifact_dir / "completion.json").is_file():
        previous, problem = _read_json(artifact_dir / "audit.json")
        recorded = (previous or {}).get("final_artifact_sha256")
        if problem or (previous or {}).get("status") != "pass" or not isinstance(recorded, dict) or not recorded:
            errors.append("completed run is missing its final artifact hash snapshot")
        else:
            if set(recorded) != set(artifact_hashes):
                errors.append("final artifact path set differs from completion audit snapshot")
            for path in set(recorded) & set(artifact_hashes):
                if recorded[path] != artifact_hashes[path]:
                    errors.append(f"final artifact SHA-256 mismatch: {path}")

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "status": "pass" if not errors else "fail",
        "run_tag": summary.get("run_tag") or config.get("run_tag"),
        "summary_schema_version": summary_schema,
        "run_config_schema_version": config.get("schema_version"),
        "artifact_dir": str(artifact_dir),
        "project_root": str(root),
        "tasks": task_reports,
        "errors": errors,
        "warnings": warnings,
        "final_artifact_sha256": artifact_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit a completed MoRACE run and its recorded input provenance."
    )
    parser.add_argument("run_path", type=Path, help="artifact directory or summary.json")
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--allow-legacy", action="store_true")
    parser.add_argument("--json", action="store_true", dest="json_output")
    args = parser.parse_args()
    try:
        report = audit_run(
            args.run_path,
            project_root=args.project_root,
            require_success=args.require_success,
            allow_legacy=args.allow_legacy,
        )
    except (OSError, ValueError) as exc:
        report = {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "status": "fail",
            "errors": [str(exc)],
            "warnings": [],
            "tasks": [],
        }
    if args.json_output:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(f"Run audit: {report['status']}")
        if report.get("run_tag"):
            print(f"- run_tag: {report['run_tag']}")
        print(f"- tasks: {len(report.get('tasks', []))}")
        print(f"- errors: {len(report.get('errors', []))}")
        print(f"- warnings: {len(report.get('warnings', []))}")
        for message in report.get("errors", []):
            print(f"ERROR: {message}")
        for message in report.get("warnings", []):
            print(f"WARNING: {message}")
    return 0 if report.get("status") == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
