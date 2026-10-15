"""Resolve final rule artifacts from completed MoRACE run summaries."""

import json
from pathlib import Path


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def resolve_branch_artifacts(task: dict, branch: str, run_tag: str, project_root: Path, *, require_code: bool = False) -> dict:
    """Resolve a successful branch independently of the aggregate task status.

    A partial/degraded task can still contain successful individual branches.
    Callers handling non-success outcomes must not use this success-only parser.
    """
    root = Path(project_root).resolve()
    task_dataset = str(task.get("dataset") or "")
    subtask = str(task.get("subtask") or "")
    detail = (task.get("branch_details") or {}).get(branch) or {}
    result = detail.get("result") or {}
    if result.get("artifact_status") != "valid" or not result.get("outcome_path"):
        raise ValueError(f"Missing valid outcome: {run_tag}/{task_dataset}/{branch}")
    outcome_path = (root / str(result["outcome_path"])).resolve()
    outcome = _read_object(outcome_path)
    if outcome.get("outcome_category") not in {"success_primary", "success_fallback"}:
        raise ValueError(f"Outcome is not successful: {outcome_path}")
    for field, expected in (("dataset", task_dataset), ("subtask", subtask),
                            ("knowledge_type", branch), ("run_tag", run_tag)):
        if field in outcome and outcome[field] != expected:
            raise ValueError(f"Outcome {field} differs from summary: {outcome_path}")
    linked_paths = [value for value in (
        (outcome.get("artifact_paths") or {}).get("metadata"),
        detail.get("metadata_path"), result.get("metadata_path"),
    ) if value]
    if linked_paths:
        resolved_paths = {(root / str(value)).resolve() for value in linked_paths}
        if len(resolved_paths) != 1:
            raise ValueError(f"Metadata paths differ between summary and outcome: {outcome_path}")
        rules_path = resolved_paths.pop()
    else:
        suffix = f"_{subtask}" if subtask else ""
        matches = list(outcome_path.parent.glob(f"*_{task_dataset}{suffix}_{branch}_rules.json"))
        if len(matches) != 1:
            raise ValueError(f"Expected one {task_dataset}/{branch} rules artifact beside {outcome_path}, found {len(matches)}")
        rules_path = matches[0]
    metadata = _read_object(rules_path)
    if not isinstance(metadata.get("rules"), list):
        raise ValueError(f"Missing rules array: {rules_path}")
    if metadata.get("status") not in {"success", "success_primary", "success_fallback"}:
        raise ValueError(f"Rule artifact is not successful: {rules_path}")
    if metadata.get("dataset") and metadata["dataset"] != task_dataset:
        raise ValueError(f"Rule artifact dataset differs from summary: {rules_path}")
    if metadata.get("subtask") and metadata["subtask"] != subtask:
        raise ValueError(f"Rule artifact subtask differs from summary: {rules_path}")
    for field, expected in (("knowledge_type", branch), ("run_tag", run_tag)):
        if field in metadata and metadata[field] != expected:
            raise ValueError(f"Rule artifact {field} differs from summary: {rules_path}")
    final_source = outcome.get("final_result_source", "")
    if result.get("final_result_source") and result["final_result_source"] != final_source:
        raise ValueError(f"Final result source differs between summary and outcome: {outcome_path}")
    if metadata.get("final_result_source") and metadata["final_result_source"] != final_source:
        raise ValueError(f"Final result source differs between rules and outcome: {rules_path}")
    code_path = None
    if require_code:
        rules = metadata["rules"]
        if not all(isinstance(rule, dict) for rule in rules):
            raise ValueError(f"Rules must be JSON objects: {rules_path}")
        if len(rules) != outcome.get("selected_rule_count"):
            raise ValueError(f"Rules array length differs from outcome selected_rule_count: {rules_path}")
        for field in ("selected_rule_count", "compiled_rule_count", "invalid_rule_count"):
            if field in metadata and metadata[field] != outcome.get(field):
                raise ValueError(f"Rule artifact {field} differs from outcome: {rules_path}")
        for field, outcome_field in (("valid_mean", "actual_valid_mean"), ("test_mean", "actual_test_mean")):
            if field in metadata and metadata[field] != outcome.get(outcome_field):
                raise ValueError(f"Rule artifact {field} differs from outcome: {rules_path}")
        expected_code_path = rules_path.with_suffix(".txt")
        code_value = (outcome.get("artifact_paths") or {}).get("code")
        code_path = (root / str(code_value)).resolve() if code_value else expected_code_path
        if code_path != expected_code_path:
            raise ValueError(f"Code path does not match final metadata artifact: {code_path} != {expected_code_path}")
        if not code_path.is_file():
            raise ValueError(f"Missing final rendered code artifact: {code_path}")
        if not code_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Empty final rendered code artifact: {code_path}")
        if outcome.get("rendered_code_available") is False:
            raise ValueError(f"Successful outcome reports unavailable rendered code: {outcome_path}")
    return {"dataset": task_dataset, "subtask": subtask, "run_tag": run_tag,
            "knowledge_type": branch, "rules_path": rules_path, "code_path": code_path,
            "outcome_path": outcome_path, "final_result_source": final_source,
            "metadata": metadata, "outcome": outcome}


def resolve_run_rules(run_dir: Path | str, branch: str = "all", dataset: str | None = None, *, project_root: Path | None = None) -> list[dict]:
    """Return final rules and outcomes for one branch of a completed run.

    Relative summary paths are relative to the project root, conventionally two
    levels above ``artifacts/<run_tag>``. Alternative layouts must supply it.
    Only the recorded outcome's directory is searched when older summaries omit
    a metadata path; missing or ambiguous artifacts are errors.
    """
    if branch not in {"synthesize", "inference", "all"}:
        raise ValueError(f"Unsupported knowledge branch: {branch}")
    directory = Path(run_dir).resolve()
    if directory.name == "summary.json":
        directory = directory.parent
    if project_root is None:
        if directory.parent.name != "artifacts":
            raise ValueError("Supply project_root for a run outside artifacts/<run_tag>")
        project_root = directory.parent.parent
    root = Path(project_root).resolve()
    summary = _read_object(directory / "summary.json")
    config = _read_object(directory / "run_config.json")
    run_tag = str(summary.get("run_tag") or config.get("run_tag") or "")
    if not run_tag or (config.get("run_tag") and config["run_tag"] != run_tag):
        raise ValueError(f"Missing or inconsistent run_tag: {directory}")
    tasks = summary.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"Run summary contains no completed tasks: {directory}")
    resolved_config = config.get("resolved_config") or {}
    model = str(resolved_config.get("model") or config.get("model") or resolved_config.get("projector_model") or "")
    rows = []
    for task in tasks:
        task_dataset = str(task.get("dataset") or "")
        if dataset is not None and task_dataset != dataset:
            continue
        if task.get("pipeline_exit_status") != "success" or task.get("status") != "success":
            raise ValueError(f"Run task is not complete and successful: {run_tag}/{task.get('task_label', '')}")
        if not task_dataset:
            raise ValueError(f"Missing dataset in run summary: {directory}")
        subtask = str(task.get("subtask") or "")
        item = resolve_branch_artifacts(task, branch, run_tag, root)
        rows.append({**item, "model": model, "run_config": config,
                     "summary_path": directory / "summary.json",
                     "recorded_dataset_provenance": task.get("dataset_provenance")})
    if not rows:
        raise ValueError(f"No completed candidates matched dataset {dataset!r}: {directory}")
    return rows
