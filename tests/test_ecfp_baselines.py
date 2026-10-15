import importlib
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_runner():
    return importlib.import_module("morace.ecfp_baselines")


def result_row(dataset, task_type, fingerprint, valid_score, test_score, seed):
    return {
        "dataset": dataset,
        "task_type": task_type,
        "fingerprint": fingerprint,
        "model": "random_forest",
        "metric": "roc_auc" if task_type == "classification" else "rmse",
        "valid_score": str(valid_score),
        "test_score": str(test_score),
        "seed": str(seed),
    }


def test_fixed_split_manifest_contract():
    runner = load_runner()
    project_root = PROJECT_ROOT
    records = runner.load_manifest(
        project_root / "data" / "manifests" / "split_manifest.csv",
        project_root,
    )

    assert len(records) == 21
    assert records[("bace", "train")].split_file_path == (
        project_root / "data" / "splits" / "bace" / "bace_train.csv"
    )
    assert runner.task_type_for_dataset("bbbp") == "classification"
    assert runner.task_type_for_dataset("freesolv") == "regression"
    assert runner.task_type_for_dataset("unknown") == "unknown"


def test_validation_only_group_selection_contract():
    runner = load_runner()
    results = [
        result_row("bace", "classification", "ecfp4", 0.70, 0.99, 0),
        result_row("bace", "classification", "ecfp4", 0.80, 0.99, 1),
        result_row("bace", "classification", "ecfp6", 0.80, 0.10, 0),
        result_row("bace", "classification", "ecfp6", 0.90, 0.10, 1),
        result_row("freesolv", "regression", "ecfp4", 1.00, 0.01, 0),
        result_row("freesolv", "regression", "ecfp4", 1.20, 0.01, 1),
        result_row("freesolv", "regression", "ecfp6", 0.80, 9.00, 0),
        result_row("freesolv", "regression", "ecfp6", 1.00, 9.00, 1),
    ]

    summary = runner.summarize_results(results)
    selected = runner.valid_selected_rows(summary)

    assert [(row["dataset"], row["selected_fingerprint"]) for row in selected] == [
        ("bace", "ecfp6"),
        ("freesolv", "ecfp6"),
    ]
    assert [row["selection_rule"] for row in selected] == [
        "max_valid_mean_roc_auc_over_seeds",
        "min_valid_mean_rmse_over_seeds",
    ]
    assert [row["test_mean"] for row in selected] == ["0.1", "9.0"]
