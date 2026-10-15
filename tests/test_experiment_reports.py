import csv
import json
from pathlib import Path
import shutil

import pytest

from morace.dataset_provenance import record_dataset_provenance
from morace.experiment_reports import collect_run, pair_comparator, report_experiments
from morace.run_summary import append_task_summary

ROOT = Path(__file__).resolve().parents[1]
MODEL = "llama-3.1-8b-instruct"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def completed_run(root, tag="one", seed=42, score=0.7, status="success"):
    data = root / "data/splits/bbbp"
    if not data.exists():
        shutil.copytree(ROOT / "data/splits/bbbp", data)
    directory = root / "artifacts" / tag
    config = directory / "run_config.json"
    write_json(config, {"run_tag": tag, "global_seed": str(seed), "route_id": "route_a", "model": MODEL,
                        "resolved_config": {"projector_model": MODEL}, "expected_tasks": [{"dataset": "bbbp", "subtask": ""}]})
    record_dataset_provenance(config, "bbbp", project_root=root, input_folder=str(root / "data/splits"),
                              manifest_path=ROOT / "data/manifests/split_manifest.csv")
    log = directory / "run.log"
    log.write_text("completed\n", encoding="utf-8")
    for branch in ("synthesize", "inference", "all"):
        folder = root / "eval_code_generation_repo" / tag / MODEL / "bbbp" / branch
        if branch != "synthesize":
            folder /= "sample_30"
        metadata_path = folder / f"{MODEL}_bbbp_{branch}_rules.json"
        final_source = "all_promoted" if branch == "all" else branch
        branch_status = status if branch == "all" else "success"
        branch_score = score if branch == "all" else 0.7
        write_json(metadata_path, {"status": branch_status, "dataset": "bbbp", "subtask": "", "knowledge_type": branch,
                                   "run_tag": tag, "final_result_source": final_source, "rules": [{"feature_name": "MolWt"}]})
        metadata_path.with_suffix(".txt").write_text("# Stored rendered-code artifact for audit fixture.\n", encoding="utf-8")
        write_json(folder / "outcome.json", {"schema_version": "outcome_v1_1", "outcome_category": "success_primary" if branch_status == "success" else branch_status,
                   "dataset": "bbbp", "subtask": "", "knowledge_type": branch, "run_tag": tag,
                   "final_result_source": final_source, "metric_name": "roc_auc", "generation_seed": seed,
                   "rf_seeds": [0, 1, 2, 3, 4], "actual_valid_mean": 0.9 if branch_score is not None else None,
                   "actual_test_mean": branch_score, "null_model_test_mean": 0.5, "itt_test_mean": 0.5,
                   "selected_rule_count": 1, "compiled_rule_count": 1, "invalid_rule_count": 0,
                   "rendered_code_available": True,
                   "artifact_paths": {"metadata": str(metadata_path), "code": str(metadata_path.with_suffix(".txt"))}})
    append_task_summary(summary_path=str(directory / "summary.json"), run_tag=tag, model=MODEL, dataset="bbbp",
                         subtask="", batch_num=30, exec_status="success", exit_code=0, log_path=str(log), project_root=root)
    return directory


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def comparator(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "route", "seed", "metric", "score", "status"])
        writer.writeheader()
        writer.writerows(rows)


def test_complete_runs_are_audited_and_grouped_without_retraining(tmp_path):
    a = completed_run(tmp_path, "a", 42, 0.7)
    b = completed_run(tmp_path, "b", 43, 0.9)
    output = tmp_path / "report"
    totals = report_experiments(run_dirs=[a, b], output_dir=output, project_root=tmp_path)
    assert totals["n_audit_passed_runs"] == 2
    assert totals["n_main_scored_rows"] == 2
    summary = read_csv(output / "main_summary.csv")[0]
    assert float(summary["mean_score"]) == pytest.approx(0.8)
    assert float(summary["std_ddof0"]) == pytest.approx(0.1)
    assert summary["n_requested"] == "2"
    assert summary["n_distinct_generation_seeds"] == "2"
    row = read_csv(output / "main_runs.csv")[0]
    assert row["rf_seeds"] == "[0, 1, 2, 3, 4]"
    assert len(row["outcome_sha256"]) == 64


@pytest.mark.parametrize("score,expected_scored", [(0.61, True), (None, False)])
def test_degraded_actual_scores_are_preserved_but_null_scores_are_never_substituted(tmp_path, score, expected_scored):
    directory = completed_run(tmp_path, score=score, status="degraded")
    rows, audit = collect_run(directory, project_root=tmp_path)
    assert audit["status"] == "pass"
    assert rows[0]["status"] == "degraded"
    assert rows[0]["scored"] is expected_scored
    assert rows[0]["score"] == (score if score is not None else "")
    assert rows[0]["score"] != 0.5


def test_modified_data_fail_audit_and_preserve_requested_row(tmp_path):
    directory = completed_run(tmp_path)
    path = tmp_path / "data/splits/bbbp/bbbp_test.csv"
    path.write_bytes(path.read_bytes() + b"\n")
    rows, audit = collect_run(directory, project_root=tmp_path)
    assert audit["status"] == "fail"
    assert len(rows) == 1
    assert rows[0]["score"] == ""
    assert rows[0]["status"] == "audit_failed"


def test_matrix_missing_jobs_stay_in_same_group_denominator(tmp_path):
    directory = completed_run(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, {"jobs": [{"run_tag": directory.name, "route": "route_a", "dataset": "bbbp", "seed": 42},
                                    {"run_tag": "never_started", "route": "route_a", "dataset": "bbbp", "seed": 43}]})
    output = tmp_path / "report"
    totals = report_experiments(plan_path=plan_path, project_root=tmp_path, output_dir=output)
    assert totals["n_requested_run_dirs"] == 2
    assert totals["n_main_requested_rows"] == 2
    assert totals["n_main_scored_rows"] == 1
    summary = read_csv(output / "main_summary.csv")[0]
    assert summary["n_requested"] == "2"
    assert summary["n_not_scored"] == "1"
    assert float(summary["mean_score"]) == 0.7


def test_matched_baselines_keep_unpaired_rows_and_reject_duplicates(tmp_path):
    directory = completed_run(tmp_path)
    rows, _ = collect_run(directory, project_root=tmp_path)
    path = tmp_path / "external.csv"
    other = {"dataset": "bbbp", "route": "route_a", "seed": "42", "metric": "roc_auc", "score": "0.6", "status": "success"}
    comparator(path, [other, dict(other, seed="43", status="failed", score="0")])
    pairs, summary, count = pair_comparator(rows, path)
    assert count == 2
    assert len(pairs) == 2
    assert pairs[0]["directional_delta"] == pytest.approx(0.1)
    assert pairs[1]["comparator_score"] == ""
    assert summary[0]["n_requested_union"] == 2
    assert summary[0]["n_paired"] == 1
    comparator(path, [other, other])
    with pytest.raises(ValueError, match="Duplicate comparator"):
        pair_comparator(rows, path)


def test_regression_pair_direction_and_degraded_measured_status(tmp_path):
    path = tmp_path / "baseline.csv"
    row = {"dataset": "freesolv", "subtask": "", "route": "route_a", "seed": "42", "metric": "rmse",
           "score": 1.1, "status": "degraded", "scored": True, "run_tag": "measured"}
    comparator(path, [{"dataset": "freesolv", "route": "route_a", "seed": 42, "metric": "rmse", "score": 1.5, "status": "success"}])
    pairs, summary, _ = pair_comparator([row], path)
    assert pairs[0]["directional_delta"] == pytest.approx(0.4)
    assert pairs[0]["main_status"] == "degraded"
    assert summary[0]["n_paired"] == 1


def test_plan_identity_mismatch_is_not_scored(tmp_path):
    directory = completed_run(tmp_path)
    rows, _ = collect_run(directory, project_root=tmp_path, planned={"run_tag": directory.name,
                          "route": "route_a", "dataset": "bbbp", "seed": 42, "identity": {"fingerprint": "different"}})
    assert rows[0]["scored"] is False
    assert rows[0]["status"] == "artifact_failed"
    assert "effective identity" in rows[0]["reason"]


@pytest.mark.parametrize("planned_dataset,planned_seed,metric", [("bbbp", 43, "roc_auc"), ("freesolv", 42, "rmse")])
def test_conflicting_disk_identity_cannot_replace_or_duplicate_requested_pairing_key(tmp_path, planned_dataset, planned_seed, metric):
    directory = completed_run(tmp_path, seed=42)
    plan = tmp_path / "plan.json"
    write_json(plan, {"jobs": [{"run_tag": directory.name, "route": "route_a", "dataset": planned_dataset, "seed": planned_seed}]})
    baseline = tmp_path / "baseline.csv"
    comparator(baseline, [{"dataset": planned_dataset, "route": "route_a", "seed": planned_seed,
                           "metric": metric, "score": 0.6, "status": "success"}])
    output = tmp_path / "report"
    totals = report_experiments(plan_path=plan, comparator_path=baseline, project_root=tmp_path, output_dir=output)
    assert totals["n_requested_run_dirs"] == totals["n_main_requested_rows"] == 1
    assert totals["n_main_scored_rows"] == 0
    row = read_csv(output / "main_runs.csv")[0]
    assert row["seed"] == str(planned_seed)
    assert row["actual_seed"] == "42"
    assert row["dataset"] == planned_dataset and row["actual_dataset"] == "bbbp"
    assert row["status"] == "artifact_failed" and row["score"] == ""
    pairs = read_csv(output / "matched_pairs.csv")
    assert len(pairs) == 1
    assert pairs[0]["seed"] == str(planned_seed) and pairs[0]["pair_status"] == "unpaired"
    summary = read_csv(output / "main_summary.csv")[0]
    assert summary["n_requested"] == "1" and summary["n_scored"] == "0"


def test_report_refuses_overwrite_and_duplicate_requested_runs(tmp_path):
    directory = completed_run(tmp_path)
    output = tmp_path / "report"
    output.mkdir()
    with pytest.raises(FileExistsError):
        report_experiments(run_dirs=[directory], output_dir=output)
    with pytest.raises(ValueError, match="Duplicate requested run"):
        report_experiments(run_dirs=[directory, directory], output_dir=tmp_path / "new")


def test_feature_results_are_kept_in_a_separate_complete_table(tmp_path):
    directory = tmp_path / "feature"
    arms = ["base", "rules", "base_plus_rules", "base_plus_shuffled_rules", "base_plus_random_rules"]
    write_json(directory / "status.json", {"status": "success"})
    write_json(directory / "protocol.json", {"schema_version": "feature_experiments_v1", "dataset": "bbbp",
               "metric": "roc_auc", "rf_seeds": [0], "arms": arms, "source": {"final_result_source": "inference_only"}})
    with (directory / "per_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["rf_seed", "split", "arm", "score", "directional_delta_vs_base"])
        writer.writeheader()
        writer.writerows([{"rf_seed": 0, "split": split, "arm": arm, "score": 0.72, "directional_delta_vs_base": 0}
                         for split in ("valid", "test") for arm in arms])
    output = tmp_path / "report"
    totals = report_experiments(feature_dirs=[directory], output_dir=output)
    assert totals["n_main_requested_rows"] == 0
    assert len(read_csv(output / "feature_runs.csv")) == 10
    assert not (output / "main_summary.csv").exists()
