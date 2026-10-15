import json
import shutil
from pathlib import Path

import pytest

from morace.dataset_provenance import record_dataset_provenance
from morace.run_audit import audit_run
from morace.run_summary import KNOWLEDGE_TYPES, append_task_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv"
RUN_TAG = "paper_llama31_native_clean_bbbp_seed42"
MODEL_TAG = "llama-3.1-8b-instruct"


def _create_completed_run(root: Path, *, outcome_seed: int | None = 42) -> Path:
    input_root = root / "data" / "splits"
    shutil.copytree(PROJECT_ROOT / "data" / "splits" / "bbbp", input_root / "bbbp")
    artifact_dir = root / "artifacts" / RUN_TAG
    artifact_dir.mkdir(parents=True)
    config_path = artifact_dir / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "run_tag": RUN_TAG,
                "global_seed": "42",
                "resolved_config": {"projector_model": MODEL_TAG},
                "prompt_hashes": {"inference_prompt.json": "prompt-hash"},
            }
        ),
        encoding="utf-8",
    )
    record_dataset_provenance(
        config_path,
        "bbbp",
        project_root=root,
        input_folder=str(input_root),
        manifest_path=MANIFEST_PATH,
    )
    (artifact_dir / "logs").mkdir()
    (artifact_dir / "logs" / "bbbp.log").write_text("completed\n", encoding="utf-8")
    (artifact_dir / "summary.json").write_text(
        json.dumps({"run_tag": RUN_TAG, "target": "bbbp", "tasks": [], "counts": {}}),
        encoding="utf-8",
    )

    for knowledge_type in KNOWLEDGE_TYPES:
        folder = root / "eval_code_generation_repo" / RUN_TAG / MODEL_TAG / "bbbp" / knowledge_type
        if knowledge_type in {"inference", "all"}:
            folder /= "sample_30"
        folder.mkdir(parents=True)
        (folder / f"{MODEL_TAG}_bbbp_{knowledge_type}_rules.json").write_text(
            json.dumps(
                {
                    "status": "success", "dataset": "bbbp", "run_tag": RUN_TAG,
                    "knowledge_type": knowledge_type,
                    "rules": [{"fixture_rule": index} for index in range(13)],
                    "selected_rule_count": 13, "compiled_rule_count": 13, "invalid_rule_count": 0,
                    "valid_mean": 0.94, "test_mean": 0.73,
                    "final_result_source": "all_promoted" if knowledge_type == "all" else knowledge_type,
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
        (folder / f"{MODEL_TAG}_bbbp_{knowledge_type}_rules.txt").write_text("# rendered fixture\n", encoding="utf-8")
        (folder / "outcome.json").write_text(
            json.dumps(
                {
                    "schema_version": "outcome_v1_1",
                    "outcome_category": "success_primary",
                    "final_result_source": "all_promoted" if knowledge_type == "all" else knowledge_type,
                    "metric_name": "roc_auc",
                    "generation_seed": outcome_seed,
                    "actual_valid_mean": 0.94,
                    "actual_test_mean": 0.73,
                    "selected_rule_count": 13,
                    "compiled_rule_count": 13,
                    "invalid_rule_count": 0,
                }
            ),
            encoding="utf-8",
        )

    assert append_task_summary(
        summary_path=f"artifacts/{RUN_TAG}/summary.json",
        run_tag=RUN_TAG,
        model=MODEL_TAG,
        dataset="bbbp",
        subtask="",
        batch_num=30,
        exec_status="success",
        exit_code=0,
        log_path=f"artifacts/{RUN_TAG}/logs/bbbp.log",
        project_root=root,
    ) == "success"
    return artifact_dir


def test_run_audit_passes_for_complete_current_contract(tmp_path):
    artifact_dir = _create_completed_run(tmp_path)

    report = audit_run(artifact_dir, project_root=tmp_path, require_success=True)

    assert report["status"] == "pass"
    assert report["errors"] == []
    assert report["warnings"] == []
    assert report["tasks"] == [
        {"task_label": "bbbp", "status": "success", "error_count": 0, "warning_count": 0}
    ]


def test_run_audit_fails_after_recorded_split_is_modified(tmp_path):
    artifact_dir = _create_completed_run(tmp_path)
    test_path = tmp_path / "data" / "splits" / "bbbp" / "bbbp_test.csv"
    test_path.write_bytes(test_path.read_bytes() + b"\n")

    report = audit_run(artifact_dir, project_root=tmp_path)

    assert report["status"] == "fail"
    assert any("test file_sha256 mismatch" in error for error in report["errors"])


def test_run_audit_accepts_seed_recovered_from_run_config(tmp_path):
    artifact_dir = _create_completed_run(tmp_path, outcome_seed=None)

    report = audit_run(artifact_dir, project_root=tmp_path, require_success=True)

    assert report["status"] == "pass"
    assert report["errors"] == []


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("schema_version", "outcome_v999", "unsupported outcome schema"),
        ("outcome_category", "unknown_success", "unsupported outcome category"),
        ("generation_seed", 43, "generation seed differs from run config"),
        ("metric_name", None, "successful outcome is missing its metric"),
    ],
)
def test_run_audit_rejects_matching_but_invalid_outcome_semantics(
    tmp_path, field, value, expected_error
):
    artifact_dir = _create_completed_run(tmp_path)
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["tasks"][0]["branch_details"]["all"]["result"]
    result[field] = value
    summary["tasks"][0]["result"][field] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    outcome_path = tmp_path / result["outcome_path"]
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome[field] = value
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")

    report = audit_run(artifact_dir, project_root=tmp_path, require_success=True)

    assert report["status"] == "fail"
    assert any(expected_error in error for error in report["errors"])


def _final_paths(root):
    folder = root / "eval_code_generation_repo" / RUN_TAG / MODEL_TAG / "bbbp/all/sample_30"
    return folder / f"{MODEL_TAG}_bbbp_all_rules.json", folder / f"{MODEL_TAG}_bbbp_all_rules.txt"


@pytest.mark.parametrize("artifact", ["metadata", "code", "empty_code", "wrong_code_link"])
def test_run_audit_rejects_missing_or_inconsistent_successful_final_artifact(tmp_path, artifact):
    artifact_dir = _create_completed_run(tmp_path)
    rules_path, code_path = _final_paths(tmp_path)
    if artifact in {"metadata", "code"}:
        (rules_path if artifact == "metadata" else code_path).unlink()
    elif artifact == "empty_code":
        code_path.write_text("", encoding="utf-8")
    else:
        wrong_code = code_path.with_name("unrelated_rules.txt")
        wrong_code.write_text("# unrelated code\n", encoding="utf-8")
        outcome_path = rules_path.parent / "outcome.json"
        outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
        outcome["artifact_paths"] = {"metadata": str(rules_path), "code": str(wrong_code)}
        outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    report = audit_run(artifact_dir, project_root=tmp_path)
    assert report["status"] == "fail"
    assert any("final artifact chain" in error for error in report["errors"])


@pytest.mark.parametrize("artifact", ["metadata", "code", "snapshot", "failed_snapshot"])
def test_completed_run_detects_content_change_or_missing_hash_snapshot(tmp_path, artifact):
    artifact_dir = _create_completed_run(tmp_path)
    first_report = audit_run(artifact_dir, project_root=tmp_path)
    assert first_report["status"] == "pass"
    (artifact_dir / "audit.json").write_text(json.dumps(first_report), encoding="utf-8")
    (artifact_dir / "completion.json").write_text("{}", encoding="utf-8")
    assert audit_run(artifact_dir, project_root=tmp_path)["status"] == "pass"
    rules_path, code_path = _final_paths(tmp_path)
    if artifact == "metadata":
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        payload["rules"][0]["fixture_rule"] = "changed_without_changing_rule_count"
        rules_path.write_text(json.dumps(payload), encoding="utf-8")
    elif artifact == "code":
        code_path.write_text("# changed rendered code\n", encoding="utf-8")
    elif artifact == "snapshot":
        (artifact_dir / "audit.json").unlink()
    else:
        first_report["status"] = "fail"
        (artifact_dir / "audit.json").write_text(json.dumps(first_report), encoding="utf-8")
    report = audit_run(artifact_dir, project_root=tmp_path)
    assert report["status"] == "fail"
    expected = "missing its final artifact hash snapshot" if "snapshot" in artifact else "SHA-256 mismatch"
    assert any(expected in error for error in report["errors"])


@pytest.mark.parametrize("field,value", [("selected_rule_count", 12), ("valid_mean", 0.1), ("final_result_source", "wrong_source")])
def test_run_audit_rejects_metadata_outcome_mismatch(tmp_path, field, value):
    artifact_dir = _create_completed_run(tmp_path)
    rules_path, _ = _final_paths(tmp_path)
    payload = json.loads(rules_path.read_text(encoding="utf-8"))
    payload[field] = value
    rules_path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_run(artifact_dir, project_root=tmp_path)
    assert report["status"] == "fail"
    assert any("differs" in error for error in report["errors"])


def test_partial_task_keeps_finite_non_success_outcomes_and_audits_successful_branches(tmp_path):
    artifact_dir = _create_completed_run(tmp_path)
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task = summary["tasks"][0]
    task["status"] = "partial"
    summary["counts"] = {"partial": 1}
    detail = task["branch_details"]["synthesize"]
    detail["status"] = "skipped"
    detail["result"]["outcome_category"] = "skipped"
    outcome_path = tmp_path / detail["result"]["outcome_path"]
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["outcome_category"] = "skipped"
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    # A non-success branch need not have successful rules or rendered code.
    for path in outcome_path.parent.glob("*_rules.*"):
        path.unlink()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    report = audit_run(artifact_dir, project_root=tmp_path)
    assert report["status"] == "pass", report["errors"]
    assert report["warnings"] == ["bbbp: scientific status is 'partial'"]
    _final_paths(tmp_path)[0].unlink()
    assert audit_run(artifact_dir, project_root=tmp_path)["status"] == "fail"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_run_audit_rejects_nonfinite_actual_outcomes(tmp_path, value):
    artifact_dir = _create_completed_run(tmp_path)
    summary_path = artifact_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    result = summary["tasks"][0]["branch_details"]["all"]["result"]
    result["actual_test_mean"] = value
    summary["tasks"][0]["result"]["actual_test_mean"] = value
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    outcome_path = tmp_path / result["outcome_path"]
    outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
    outcome["actual_test_mean"] = value
    outcome_path.write_text(json.dumps(outcome), encoding="utf-8")
    report = audit_run(artifact_dir, project_root=tmp_path)
    assert report["status"] == "fail"
    assert any("must be finite or null" in error for error in report["errors"])
