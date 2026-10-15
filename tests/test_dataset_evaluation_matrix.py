import json
import math
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_CONTRACTS = {
    "bace": {
        "task_type": "classification",
        "smiles_column": "mol",
        "target_column": "Class",
        "rows": {"train": 1210, "valid": 151, "test": 152},
        "class_counts": {
            "train": {0: 695, 1: 515},
            "valid": {0: 67, 1: 84},
            "test": {0: 60, 1: 92},
        },
    },
    "bbbp": {
        "task_type": "classification",
        "smiles_column": "smiles",
        "target_column": "p_np",
        "rows": {"train": 1631, "valid": 204, "test": 204},
        "class_counts": {
            "train": {0: 290, 1: 1341},
            "valid": {0: 92, 1: 112},
            "test": {0: 97, 1: 107},
        },
    },
    "clintox": {
        "task_type": "classification",
        "smiles_column": "smiles",
        "target_column": "CT_TOX",
        "rows": {"train": 1182, "valid": 148, "test": 148},
        "class_counts": {
            "train": {0: 1087, 1: 95},
            "valid": {0: 141, 1: 7},
            "test": {0: 138, 1: 10},
        },
    },
    "esol": {
        "task_type": "regression",
        "smiles_column": "smiles",
        "target_column": "ESOL predicted log solubility in mols per litre",
        "rows": {"train": 902, "valid": 113, "test": 113},
        "null_rmse": {"valid": 1.761225822813107, "test": 1.9567495368902095},
    },
    "freesolv": {
        "task_type": "regression",
        "smiles_column": "smiles",
        "target_column": "calc",
        "rows": {"train": 513, "valid": 64, "test": 65},
        "null_rmse": {"valid": 6.846184553476518, "test": 5.893567317409183},
    },
    "hiv": {
        "task_type": "classification",
        "smiles_column": "smiles",
        "target_column": "HIV_active",
        "rows": {"train": 32901, "valid": 4113, "test": 4113},
        "class_counts": {
            "train": {0: 31669, 1: 1232},
            "valid": {0: 4032, 1: 81},
            "test": {0: 3983, 1: 130},
        },
    },
    "lipophilicity": {
        "task_type": "regression",
        "smiles_column": "smiles",
        "target_column": "exp",
        "rows": {"train": 3360, "valid": 420, "test": 420},
        "null_rmse": {"valid": 1.1548318631583938, "test": 1.1723065734598714},
    },
}


def _balanced_sample(smiles, targets, per_class):
    selected = []
    for label in (0, 1):
        candidates = [
            (smiles[index], targets[index])
            for index in range(len(targets))
            if int(targets[index]) == label
        ]
        selected.extend(candidates[: min(per_class, len(candidates))])
    return [item[0] for item in selected], [item[1] for item in selected]


def _even_sample(smiles, targets, size):
    if len(smiles) <= size:
        indices = list(range(len(smiles)))
    else:
        indices = [round(index * (len(smiles) - 1) / (size - 1)) for index in range(size)]
    return [smiles[index] for index in indices], [targets[index] for index in indices]


def _compact_hparams(full_hparams):
    compact = {}
    for dataset in DATASET_CONTRACTS:
        params = dict(full_hparams[dataset])
        assert int(params["n_estimators"]) > 0
        params.update(
            n_estimators=8,
            max_depth=6,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            warm_start=False,
            n_jobs=1,
        )
        compact[dataset] = params
    return compact


def _run_matrix():
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from rdkit import RDLogger

    from morace import evaluation_pipeline
    from morace.datasets import load_split_xy
    from morace.metric_policy import metric_direction
    from morace.rule_compiler import build_feature_matrix, compile_rules

    RDLogger.DisableLog("rdApp.warning")
    rules = [
        {
            "source": "literature",
            "feature_family": "descriptor",
            "feature_name": "MolWt",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
        },
        {
            "source": "data",
            "feature_family": "descriptor",
            "feature_name": "MolLogP",
            "feature_mode": "numeric_value",
            "relation_to_target": "positive",
        },
        {
            "source": "literature",
            "feature_family": "descriptor",
            "feature_name": "TPSA",
            "feature_mode": "numeric_value",
            "relation_to_target": "negative",
        },
    ]
    compiled_rules, invalid_rules = compile_rules(rules)
    assert len(compiled_rules) == len(rules)
    assert invalid_rules == []

    full_hparams = evaluation_pipeline.load_model_hparams()
    assert set(DATASET_CONTRACTS).issubset(full_hparams)
    compact_hparams = _compact_hparams(full_hparams)
    evaluation_pipeline.load_model_hparams = lambda: compact_hparams

    results = {}
    for dataset, contract in DATASET_CONTRACTS.items():
        loaded = {
            split: load_split_xy(dataset, split=split)
            for split in ("train", "valid", "test")
        }
        for split, (smiles, targets, spec) in loaded.items():
            assert len(smiles) == contract["rows"][split]
            assert len(targets) == contract["rows"][split]
            assert spec.task_type == contract["task_type"]
            assert spec.smiles_column == contract["smiles_column"]
            assert spec.target_column == contract["target_column"]

        y_train = loaded["train"][1]
        y_valid = loaded["valid"][1]
        y_test = loaded["test"][1]
        spec = loaded["train"][2]
        null_scores = evaluation_pipeline.compute_null_model_scores(
            spec, y_train, y_valid, y_test
        )

        if spec.task_type == "classification":
            for split, (_, targets, _) in loaded.items():
                actual_counts = {
                    label: sum(int(value) == label for value in targets)
                    for label in (0, 1)
                }
                assert actual_counts == contract["class_counts"][split]
            assert null_scores == {
                "valid_mean": 0.5,
                "test_mean": 0.5,
                "strategy": "constant_auc_0.5",
            }
            sampled = {
                "train": _balanced_sample(*loaded["train"][:2], per_class=32),
                "valid": _balanced_sample(*loaded["valid"][:2], per_class=16),
                "test": _balanced_sample(*loaded["test"][:2], per_class=16),
            }
            metric_name = "roc_auc"
        else:
            assert null_scores["strategy"] == "train_mean_rmse"
            for split in ("valid", "test"):
                assert math.isclose(
                    null_scores[f"{split}_mean"],
                    contract["null_rmse"][split],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            sampled = {
                "train": _even_sample(*loaded["train"][:2], size=96),
                "valid": _even_sample(*loaded["valid"][:2], size=48),
                "test": _even_sample(*loaded["test"][:2], size=48),
            }
            metric_name = "rmse"

        raw_matrices = [
            build_feature_matrix(compiled_rules, sampled[split][0])
            for split in ("train", "valid", "test")
        ]
        X_train, X_valid, X_test, kept_features, feature_metrics = (
            evaluation_pipeline.sanitize_feature_matrices(
                *raw_matrices,
                [rule.name for rule in compiled_rules],
            )
        )
        assert len(kept_features) == 3
        assert feature_metrics["feature_count_before_filter"] == 3
        assert feature_metrics["feature_count_after_filter"] == 3
        assert feature_metrics["non_nan_coverage"] == 1.0

        evaluator_args = (
            dataset,
            "",
            X_train,
            sampled["train"][1],
            X_valid,
            sampled["valid"][1],
            X_test,
            sampled["test"][1],
        )
        if spec.task_type == "classification":
            first_scores = evaluation_pipeline.evaluate_classification(*evaluator_args)
            second_scores = evaluation_pipeline.evaluate_classification(*evaluator_args)
        else:
            regression_args = evaluator_args[:1] + evaluator_args[2:]
            first_scores = evaluation_pipeline.evaluate_regression_rf(*regression_args)
            second_scores = evaluation_pipeline.evaluate_regression_rf(*regression_args)

        assert first_scores == second_scores
        valid_scores, test_scores = first_scores
        assert len(valid_scores) == len(test_scores) == 5
        assert all(math.isfinite(float(score)) for score in valid_scores + test_scores)
        if spec.task_type == "classification":
            assert all(0.0 <= float(score) <= 1.0 for score in valid_scores + test_scores)
        else:
            assert all(float(score) >= 0.0 for score in valid_scores + test_scores)

        direction = metric_direction(metric_name)
        assert direction["higher_is_better"] == (spec.task_type == "classification")
        results[dataset] = {
            "task_type": spec.task_type,
            "metric": metric_name,
            "feature_count": len(kept_features),
            "deterministic": first_scores == second_scores,
        }
    return results


def test_seven_dataset_fixed_rule_evaluation_matrix():
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--harness"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        "Seven-dataset evaluation contract failed in its isolated interpreter.\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert set(payload) == set(DATASET_CONTRACTS)
    assert all(item["feature_count"] == 3 for item in payload.values())
    assert all(item["deterministic"] for item in payload.values())


if __name__ == "__main__":
    print(json.dumps(_run_matrix(), sort_keys=True))
