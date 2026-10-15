import json
from pathlib import Path

from morace.run_summary import KNOWLEDGE_TYPES, append_task_summary


RUN_TAG = "paper_llama31_native_clean_bbbp_seed42"
MODEL_TAG = "llama-3.1-8b-instruct"


def _branch_folder(root: Path, knowledge_type: str) -> Path:
    folder = root / "eval_code_generation_repo" / RUN_TAG / MODEL_TAG / "bbbp" / knowledge_type
    if knowledge_type in {"inference", "all"}:
        folder /= "sample_30"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _write_run_config(root: Path) -> None:
    path = root / "artifacts" / RUN_TAG / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "run_config_v1_1",
                "global_seed": "42",
                "resolved_config": {"projector_model": MODEL_TAG},
                "prompt_hashes": {"inference_prompt.json": "prompt-hash"},
                "dataset_provenance": {
                    "bbbp": {
                        "schema_version": "dataset_provenance_v1",
                        "status": "valid",
                        "reason": "",
                        "dataset": "bbbp",
                        "subtask": "",
                        "task_label": "bbbp",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_branch(root: Path, knowledge_type: str, *, outcome_text: str | None = None) -> None:
    folder = _branch_folder(root, knowledge_type)
    metadata_path = folder / f"{MODEL_TAG}_bbbp_{knowledge_type}_rules.json"
    metadata_path.write_text(
        json.dumps(
            {
                "status": "success",
                "branch_metrics": {
                    "inference_projection": {
                        "strict_projector_valid_rate": 1.0,
                        "repair_used_rate": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    if outcome_text is None:
        outcome_text = json.dumps(
            {
                "schema_version": "outcome_v1_1",
                "outcome_category": "success_primary",
                "final_result_source": "all_promoted" if knowledge_type == "all" else knowledge_type,
                "metric_name": "roc_auc",
                "generation_seed": 42,
                "actual_valid_mean": 0.94,
                "actual_test_mean": 0.73,
                "selected_rule_count": 13,
                "compiled_rule_count": 13,
                "invalid_rule_count": 0,
            }
        )
    (folder / "outcome.json").write_text(outcome_text, encoding="utf-8")


def _update_outcome(root: Path, knowledge_type: str, **changes) -> None:
    path = _branch_folder(root, knowledge_type) / "outcome.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(changes)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _append(root: Path, *, exec_status: str = "success", exit_code: int = 0, reason: str = "") -> str:
    summary = root / "artifacts" / RUN_TAG / "summary.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps({"run_tag": RUN_TAG, "target": "bbbp", "tasks": [], "counts": {}}),
        encoding="utf-8",
    )
    return append_task_summary(
        summary_path=f"artifacts/{RUN_TAG}/summary.json",
        run_tag=RUN_TAG,
        model=MODEL_TAG,
        dataset="bbbp",
        subtask="",
        batch_num=30,
        exec_status=exec_status,
        exit_code=exit_code,
        log_path=f"artifacts/{RUN_TAG}/logs/bbbp.log",
        reason=reason,
        project_root=root,
    )


def test_summary_records_primary_result_and_branch_outcomes(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)

    assert _append(tmp_path) == "success"

    payload = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))
    task = payload["tasks"][0]
    assert payload["schema_version"] == "run_summary_v1_2"
    assert payload["target"] == "bbbp"
    assert payload["counts"] == {"success": 1}
    assert task["actual_projector_model"] == MODEL_TAG
    assert task["actual_prompt_hash"] == "prompt-hash"
    assert task["run_config_schema_version"] == "run_config_v1_1"
    assert task["dataset_provenance_status"] == "valid"
    assert task["dataset_provenance"]["task_label"] == "bbbp"
    assert task["result"] == task["branch_details"]["all"]["result"]
    assert task["result"]["artifact_status"] == "valid"
    assert task["result"]["generation_seed"] == 42
    assert task["result"]["generation_seed_source"] == "outcome"
    assert task["result"]["metric_name"] == "roc_auc"
    assert task["result"]["actual_test_mean"] == 0.73
    assert task["result"]["selected_rule_count"] == 13


def test_missing_outcome_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    (_branch_folder(tmp_path, "all") / "outcome.json").unlink()

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "missing"
    assert task["result"]["generation_seed"] == 42
    assert task["result"]["generation_seed_source"] == "run_config"
    assert "missing artifact" in task["reason"]


def test_invalid_outcome_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    (_branch_folder(tmp_path, "all") / "outcome.json").write_text("{invalid", encoding="utf-8")

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "invalid"
    assert "JSONDecodeError" in task["result"]["artifact_reason"]


def test_incomplete_outcome_schema_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    (_branch_folder(tmp_path, "all") / "outcome.json").write_text("{}", encoding="utf-8")

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "invalid"
    assert "missing fields" in task["result"]["artifact_reason"]


def test_unsupported_outcome_schema_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    _update_outcome(tmp_path, "all", schema_version="outcome_v999")

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "invalid"
    assert "unsupported outcome schema" in task["result"]["artifact_reason"]


def test_unknown_outcome_category_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    _update_outcome(tmp_path, "all", outcome_category="unknown_success")

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "invalid"
    assert "unsupported outcome category" in task["result"]["artifact_reason"]


def test_outcome_seed_mismatch_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)
    _update_outcome(tmp_path, "all", generation_seed=43)

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["result"]["artifact_status"] == "invalid"
    assert "generation seed differs from run config" in task["result"]["artifact_reason"]


def test_failed_pipeline_status_is_preserved_without_artifacts(tmp_path):
    assert _append(tmp_path, exec_status="failed", exit_code=2, reason="step=inference exit_code=2") == "failed"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["status"] == "failed"
    assert task["pipeline_exit_status"] == "failed"
    assert task["exit_code"] == 2
    assert task["reason"] == "step=inference exit_code=2"


def test_missing_provenance_marks_successful_pipeline_incomplete(tmp_path):
    _write_run_config(tmp_path)
    config_path = tmp_path / "artifacts" / RUN_TAG / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config.pop("dataset_provenance")
    config_path.write_text(json.dumps(config), encoding="utf-8")
    for knowledge_type in KNOWLEDGE_TYPES:
        _write_branch(tmp_path, knowledge_type)

    assert _append(tmp_path) == "incomplete"

    task = json.loads((tmp_path / "artifacts" / RUN_TAG / "summary.json").read_text(encoding="utf-8"))["tasks"][0]
    assert task["dataset_provenance_status"] == "missing"
    assert task["dataset_provenance"] is None
    assert "missing dataset provenance" in task["reason"]
