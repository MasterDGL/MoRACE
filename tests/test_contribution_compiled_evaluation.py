import builtins
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from morace import contribution_analysis, rule_compiler


@pytest.fixture(params=("in_process_sklearn", "external_sklearn_worker"))
def sklearn_mode(request, monkeypatch):
    if request.param == "external_sklearn_worker":
        original_import = builtins.__import__

        def import_without_sklearn(name, *args, **kwargs):
            if name == "sklearn":
                raise ModuleNotFoundError("sklearn is unavailable in this worker")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", import_without_sklearn)
    return request.param


def make_compiled_evaluation_fixture(tmp_path, monkeypatch):
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "model_registry.json").write_text(
        json.dumps(
            {
                "bbbp": {"n_estimators": 11, "max_depth": 3},
                "esol": {"n_estimators": 13, "max_depth": 4},
            }
        ),
        encoding="utf-8",
    )

    manifest = {}
    for split in ("train", "valid", "test"):
        split_path = tmp_path / "data" / f"{split}.csv"
        split_path.parent.mkdir(exist_ok=True)
        split_path.write_text("smiles,target\nCCO,0\nCCN,1\nCCC,0\n", encoding="utf-8")
        for dataset in ("bbbp", "esol"):
            manifest[(dataset, split)] = {
                "split_file_path": str(split_path.relative_to(tmp_path)),
                "smiles_column": "smiles",
                "target_column": "target",
            }

    compiled_rules = [
        SimpleNamespace(name="a", rule={"feature_family": "descriptor"}),
        SimpleNamespace(name="b", rule={"feature_family": "count"}),
        SimpleNamespace(name="c", rule={"feature_family": "pattern_presence"}),
    ]

    monkeypatch.setattr(
        rule_compiler,
        "build_feature_matrix",
        lambda _rules, smiles: [
            [float(index), float(index + 1), float(index + 2)]
            for index, _smiles in enumerate(smiles)
        ],
    )

    def fake_sanitize(train, valid, test, feature_names):
        assert feature_names == ["a", "b", "c"]
        return (
            np.asarray(train, dtype=float)[:, :2],
            np.asarray(valid, dtype=float)[:, :2],
            np.asarray(test, dtype=float)[:, :2],
            ["a", "b"],
        )

    monkeypatch.setattr(contribution_analysis, "sanitize_feature_matrices", fake_sanitize)
    return compiled_rules, manifest


def variant_scores(variants):
    return {
        variant["name"]: {
            "valid_scores": [float(index + 1), float(index + 3)],
            "test_scores": [float(index + 2), float(index + 4)],
        }
        for index, variant in enumerate(variants)
    }


def test_compiled_loro_variant_contract(tmp_path, monkeypatch, sklearn_mode):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    calls = []

    def fake_evaluate(task_type, params, seeds, *args):
        variants = args[-1]
        calls.append((task_type, params, seeds, variants))
        return variant_scores(variants)

    monkeypatch.setattr(contribution_analysis, "evaluate_variants_in_process", fake_evaluate)
    monkeypatch.setattr(contribution_analysis, "evaluate_variants_with_external_sklearn", fake_evaluate)
    summarized, mode = contribution_analysis.evaluate_compiled_loro_variants(
        tmp_path,
        "bbbp",
        compiled_rules,
        manifest,
        [2, 5],
    )

    assert mode == sklearn_mode
    assert calls == [
        (
            "classification",
            {"n_estimators": 11, "max_depth": 3},
            [2, 5],
            [
                {"name": "base", "keep_indices": [0, 1]},
                {"name": "drop_0", "keep_indices": [1]},
                {"name": "drop_1", "keep_indices": [0]},
                {"name": "drop_2", "keep_indices": [0, 1]},
            ],
        )
    ]
    assert summarized["base"]["valid_mean"] == 2.0
    assert summarized["base"]["valid_std"] == math.sqrt(2.0)
    assert summarized["drop_0"]["note"] == "exact_loro_feature_removed_from_matrix"
    assert summarized["drop_2"]["note"] == (
        "compiled_rule_was_filtered_by_sanitizer; dropping_it_does_not_change_matrix"
    )


def test_compiled_family_variant_contract(tmp_path, monkeypatch, sklearn_mode):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    calls = []

    def fake_evaluate(task_type, params, seeds, *args):
        variants = args[-1]
        calls.append((task_type, params, seeds, variants))
        return variant_scores(variants)

    monkeypatch.setattr(contribution_analysis, "evaluate_variants_in_process", fake_evaluate)
    monkeypatch.setattr(contribution_analysis, "evaluate_variants_with_external_sklearn", fake_evaluate)
    summarized, skipped, mode = contribution_analysis.evaluate_compiled_family_variants(
        tmp_path,
        "bbbp",
        compiled_rules,
        manifest,
        [2, 5],
    )

    assert mode == sklearn_mode
    assert skipped == {}
    assert calls[0][3] == [
        {"name": "base", "keep_indices": [0, 1]},
        {"name": "drop_family_descriptor", "keep_indices": [1]},
        {"name": "drop_family_count", "keep_indices": [0]},
        {"name": "drop_family_pattern_presence", "keep_indices": [0, 1]},
    ]
    assert summarized["drop_family_descriptor"]["note"] == (
        "family_level_feature_columns_removed_from_matrix"
    )


def test_compiled_feature_importance_contract(tmp_path, monkeypatch, sklearn_mode):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    calls = []

    def fake_evaluate(task_type, params, seeds, *args):
        calls.append((task_type, params, seeds, args))
        return {
            "valid_scores": [1.0, 3.0],
            "test_scores": [2.0, 4.0],
            "importances": [[0.7, 0.3], [0.6, 0.4]],
        }

    monkeypatch.setattr(
        contribution_analysis,
        "evaluate_feature_importance_in_process",
        fake_evaluate,
    )
    monkeypatch.setattr(contribution_analysis, "evaluate_feature_importance_with_external_sklearn", fake_evaluate)
    result = contribution_analysis.evaluate_compiled_feature_importance(
        tmp_path,
        "esol",
        compiled_rules,
        manifest,
        [2, 5],
    )

    assert calls[0][:3] == (
        "regression",
        {"n_estimators": 13, "max_depth": 4},
        [2, 5],
    )
    assert result == {
        "valid_scores": [1.0, 3.0],
        "test_scores": [2.0, 4.0],
        "valid_mean": 2.0,
        "valid_std": math.sqrt(2.0),
        "test_mean": 3.0,
        "test_std": math.sqrt(2.0),
        "importances": [[0.7, 0.3], [0.6, 0.4]],
        "kept_features": ["a", "b"],
        "sklearn_mode": sklearn_mode,
    }


@pytest.mark.parametrize("method, message", [
    ("evaluate_compiled_loro_variants", "No kept features available for base model."),
    ("evaluate_compiled_family_variants", "No kept features available for base model."),
    ("evaluate_compiled_feature_importance", "No kept features available for feature importance."),
])
def test_compiled_evaluation_rejects_empty_features(tmp_path, monkeypatch, method, message):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contribution_analysis, "sanitize_feature_matrices",
        lambda *_args: ([], [], [], []),
    )
    with pytest.raises(RuntimeError) as failure:
        getattr(contribution_analysis, method)(tmp_path, "bbbp", compiled_rules, manifest, [2, 5])
    assert str(failure.value) == message


def test_compiled_loro_rejects_removing_last_feature(tmp_path, monkeypatch):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contribution_analysis, "sanitize_feature_matrices",
        lambda train, valid, test, _names: (train, valid, test, ["a"]),
    )
    with pytest.raises(RuntimeError, match="Dropping a leaves no kept matrix features"):
        contribution_analysis.evaluate_compiled_loro_variants(
            tmp_path, "bbbp", compiled_rules, manifest, [2, 5],
        )


def test_compiled_family_records_unusable_variants(tmp_path, monkeypatch, sklearn_mode):
    compiled_rules, manifest = make_compiled_evaluation_fixture(tmp_path, monkeypatch)
    for compiled in compiled_rules:
        compiled.rule["feature_family"] = "descriptor"

    def evaluate_base(_task, _params, _seeds, *args):
        assert args[-1] == [{"name": "base", "keep_indices": [0, 1]}]
        return variant_scores(args[-1])

    monkeypatch.setattr(contribution_analysis, "evaluate_variants_in_process", evaluate_base)
    monkeypatch.setattr(contribution_analysis, "evaluate_variants_with_external_sklearn", evaluate_base)
    summarized, skipped, mode = contribution_analysis.evaluate_compiled_family_variants(
        tmp_path, "bbbp", compiled_rules, manifest, [2, 5],
    )
    assert mode == sklearn_mode
    assert list(summarized) == ["base"]
    assert skipped == {
        "descriptor": "removing_family_leaves_no_kept_matrix_features",
        "count": "family_not_present",
        "pattern_presence": "family_not_present",
    }
