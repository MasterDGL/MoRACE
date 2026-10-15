"""Public evaluation settings are local to each invocation, including in one process."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, asdict, replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from morace.evaluation_config import EvaluationConfig
from morace.evaluation_pipeline import EvaluationPipeline, build_parser


def _reports():
    return [
        dict(name="best_value", valid_mean=0.935, compiled_rule_count=6,
             root_feature_count=6, effective_literature_coverage=1, effective_data_coverage=6),
        dict(name="coverage", valid_mean=0.9325, compiled_rule_count=6,
             root_feature_count=6, effective_literature_coverage=2, effective_data_coverage=5),
    ]


def test_two_evaluators_keep_distinct_selection_policies_in_one_process(monkeypatch):
    strict = EvaluationPipeline(EvaluationConfig(all_finalist_keep_lit_delta=0.001))
    tolerant = EvaluationPipeline(EvaluationConfig(all_finalist_keep_lit_delta=0.004))
    monkeypatch.setenv("ALL_FINALIST_KEEP_LIT_DELTA", "999")
    for _ in range(3):
        assert strict.choose_finalist_with_coverage_guard(_reports())["winner"]["name"] == "best_value"
        assert tolerant.choose_finalist_with_coverage_guard(_reports())["winner"]["name"] == "coverage"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda p: p.choose_finalist_with_coverage_guard(_reports()),
                                [strict, tolerant] * 4))
    assert [x["winner"]["name"] for x in results] == ["best_value", "coverage"] * 4


def test_config_is_immutable_and_namespace_environment_are_copied():
    namespace = SimpleNamespace(run_tag="a", list_num=5)
    environment = {"ALL_FINALIST_KEEP_LIT_DELTA": "0.002"}
    config = EvaluationConfig.from_namespace(namespace, environment=environment)
    namespace.run_tag = "b"
    environment["ALL_FINALIST_KEEP_LIT_DELTA"] = "0.9"
    assert config.run_tag == "a"
    assert config.all_finalist_keep_lit_delta == 0.002
    with pytest.raises(FrozenInstanceError):
        config.list_num = 6
    pipeline = EvaluationPipeline(config)
    with pytest.raises(AttributeError):
        pipeline.config = replace(config, run_tag="b")


def test_cli_defaults_and_python_defaults_are_identical_without_environment():
    assert EvaluationConfig.from_namespace(build_parser().parse_args([])) == EvaluationConfig()


def test_paper_evaluation_policy_keeps_recorded_defaults():
    defaults = asdict(EvaluationConfig())
    expected = {
        "validation_promotion_epsilon": 0.005, "keep_all_valid_delta": 0.01,
        "revert_all_valid_delta": 0.015, "min_all_compiled_features": 3,
        "capacity_keep_valid_delta": 0.01, "threshold_keep_min_gain": 0.01,
        "all_finalist_enable": False, "all_finalist_max_candidates": 3,
        "all_finalist_keep_lit_delta": 0.004, "all_finalist_min_effective_lit": 2,
        "all_finalist_min_effective_data": 5, "all_finalist_min_roots": 6,
        "all_plus_one_alias_enable": False, "all_plus_one_alias_max_valid_drop": 0.002,
        "all_plus_one_alias_min_promoted_multi_roots": 2,
        "lit_only_slot_control_enable": False, "lit_only_slot_max_valid_drop": 0.001,
        "lit_only_slot_min_promoted_multi_roots": 2,
        "multi_consensus_promotion_enable": False, "m7_lit_preserving_swap_enable": False,
    }
    assert {key: defaults[key] for key in expected} == expected


@pytest.mark.parametrize("environment", [
    {"ALL_FINALIST_ENABLE": "perhaps"}, {"ALL_FINALIST_ENABLE": "2"},
    {"THRESHOLD_KEEP_MIN_GAIN": "nan"}, {"KEEP_ALL_VALID_DELTA": "-0.1"},
    {"ALL_FINALIST_MAX_CANDIDATES": "0"},
])
def test_invalid_environment_rejected_only_when_explicitly_resolved(environment):
    with pytest.raises(ValueError):
        EvaluationConfig.from_namespace(SimpleNamespace(), environment=environment)


def test_import_and_explicit_api_ignore_malformed_ambient_environment(monkeypatch):
    monkeypatch.setenv("THRESHOLD_KEEP_MIN_GAIN", "broken")
    monkeypatch.setenv("ALL_FINALIST_ENABLE", "broken")
    monkeypatch.setenv("M7_LIT_PRESERVING_SWAP_ENABLE", "broken")
    monkeypatch.setenv("MULTI_CONSENSUS_PROMOTION_ENABLE", "broken")
    script = """
from morace.evaluation_pipeline import EvaluationConfig, EvaluationPipeline
assert EvaluationPipeline().config.threshold_keep_min_gain == 0.01
assert EvaluationConfig().all_finalist_enable is False
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            encoding="utf-8", cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 0, result.stderr


def test_cli_reports_invalid_protocol_before_reading_rules(monkeypatch):
    monkeypatch.setenv("ALL_FINALIST_KEEP_LIT_DELTA", "broken")
    result = subprocess.run([sys.executable, "-m", "morace.evaluation_pipeline"],
                            capture_output=True, encoding="utf-8",
                            cwd=Path(__file__).resolve().parents[1])
    assert result.returncode == 2
    assert "ALL_FINALIST_KEEP_LIT_DELTA" in result.stderr
    assert "Traceback" not in result.stderr


def test_two_evaluators_keep_output_scope_and_empty_scope_is_explicit(tmp_path, monkeypatch):
    common = EvaluationConfig(code_gen_folder=str(tmp_path / "code"), output_dir=str(tmp_path / "eval"))
    first = EvaluationPipeline(replace(common, run_tag="first", model="model-a", list_num=4))
    second = EvaluationPipeline(replace(common, run_tag="second", model="model-b", list_num=9))
    empty = EvaluationPipeline(common)
    monkeypatch.setenv("MORACE_RUN_TAG", "ambient-other")
    first_path, _ = first.code_output_paths("bbbp", "")
    second_path, _ = second.code_output_paths("bbbp", "")
    empty_path, _ = empty.code_output_paths("bbbp", "")
    assert "first" in Path(first_path).parts and "sample_4" in Path(first_path).parts
    assert "second" in Path(second_path).parts and "sample_9" in Path(second_path).parts
    assert "ambient-other" not in Path(empty_path).parts
    assert first.code_output_paths("bbbp", "")[0] == first_path
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}", encoding="utf-8")
    assert empty.resolve_existing_path(str(tmp_path / "missing"), str(legacy)) == str(legacy)


def test_dataset_roots_are_captured_and_stay_local_to_each_evaluator(tmp_path, monkeypatch):
    roots = [tmp_path / "one", tmp_path / "two"]
    for label, root in enumerate(roots):
        (root / "bbbp").mkdir(parents=True)
        (root / "bbbp" / "bbbp_train.csv").write_text(f"smiles,p_np\nCC,{label}\n", encoding="utf-8")
    environment = {"MORACE_DATASET_ROOT": str(roots[0]), "MORACE_RUN_TAG": "captured-run"}
    first_config = EvaluationConfig.from_namespace(SimpleNamespace(), environment=environment)
    first = EvaluationPipeline(first_config)
    second = EvaluationPipeline(EvaluationConfig(dataset_root=str(roots[1])))
    environment["MORACE_DATASET_ROOT"] = "missing"
    monkeypatch.setenv("MORACE_DATASET_ROOT", "also-missing")
    assert first.config.run_tag == "captured-run"
    assert second.config.run_tag == ""
    assert first.load_split_xy("bbbp")[1] == [0]
    assert second.load_split_xy("bbbp")[1] == [1]
    assert first.load_split_xy("bbbp")[1] == [0]


def test_posthoc_variant_keeps_explicit_evaluation_binding(tmp_path, monkeypatch):
    from morace import ablation, evaluation_pipeline

    root = tmp_path / "splits"
    (root / "bbbp").mkdir(parents=True)
    rows = "smiles,p_np\nCC,0\nCCO,1\nCCCC,0\nCCN,1\nCCCCC,0\nCCCO,1\n"
    for split in ("train", "valid", "test"):
        (root / "bbbp" / f"bbbp_{split}.csv").write_text(rows, encoding="utf-8")
    monkeypatch.setenv("MORACE_DATASET_ROOT", "missing")
    monkeypatch.setenv("THRESHOLD_KEEP_MIN_GAIN", "broken")
    monkeypatch.setattr(evaluation_pipeline, "load_model_hparams", lambda: {
        "bbbp": {"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
    })
    rules = [dict(source="literature", feature_family="descriptor", feature_name=name,
                  feature_mode="numeric_value", relation_to_target="context_dependent")
             for name in ("MolWt", "TPSA")]
    result = ablation.evaluate_rule_variant("fixed", rules, "bbbp", "",
                                           config=EvaluationConfig(dataset_root=str(root)))
    assert result["variant_name"] == "fixed"
    assert len(result["test_scores"]) == 5
    assert len(result["kept_feature_names"]) == 2
    assert all(np.isfinite(result["test_scores"]))


@pytest.mark.parametrize("has_selected, has_compiled, reason", [
    (False, False, "No selected rules were available for compilation."),
    (False, True, "No selected rules were available for compilation."),
    (True, False, "No rules could be compiled into valid RDKit features."),
])
def test_evaluation_preserves_empty_rule_skip_reason(monkeypatch, has_selected, has_compiled, reason):
    pipeline = EvaluationPipeline()
    selected = [{"canonical_key": "rule"}] if has_selected else []
    compiled = [SimpleNamespace(name="feature")] if has_compiled else []
    monkeypatch.setattr(pipeline, "load_rules", lambda *_args: (selected, {}))
    monkeypatch.setattr(pipeline, "evaluate_quality_gate", lambda *_args: ("success", "", "strict_pass"))
    monkeypatch.setattr(pipeline, "prepare_rules_for_compilation_with_audit", lambda *_args, **_kwargs: ([], {}))
    monkeypatch.setattr(pipeline, "compile_rules", lambda *_args: (compiled, []))
    monkeypatch.setattr(pipeline, "build_shrinkage_audit", lambda **_kwargs: {"audit": "unchanged"})
    saved = []
    monkeypatch.setattr(pipeline, "save_skip_outputs", lambda **kwargs: saved.append(kwargs))
    pipeline.main()
    assert len(saved) == 1
    assert saved[0]["status"] == "skipped"
    assert saved[0]["reason"] == reason
    assert saved[0]["selected_rules"] == selected
    assert saved[0]["compiled_rules"] == compiled
    assert saved[0]["shrinkage_audit"] == {"audit": "unchanged"}


@pytest.mark.parametrize("dataset,task_type", [("bbbp", "classification"), ("esol", "regression")])
def test_fixed_feature_predictions_match_pre_refactor_fixture(dataset, task_type, monkeypatch):
    """Expected scores were executed with the saved pre-refactor implementation."""
    from morace import evaluation_pipeline
    import sklearn
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    fixture = json.loads((Path(__file__).parent / "fixtures" / "evaluation_rf_reference.json").read_text())
    entry = fixture[dataset]
    monkeypatch.setattr(evaluation_pipeline, "load_model_hparams", lambda: fixture["hparams"])
    arrays = [np.asarray(fixture[key], dtype=float) for key in ("X_train", "X_valid", "X_test")]
    labels = [entry[key] for key in ("y_train", "y_valid", "y_test")]
    args = (arrays[0], labels[0], arrays[1], labels[1], arrays[2], labels[2])
    evaluator = EvaluationPipeline()
    actual = (evaluator.evaluate_classification(dataset, "", *args) if task_type == "classification"
              else evaluator.evaluate_regression_rf(dataset, *args))
    # Check the original estimator/seed/split contract even if a supported newer
    # sklearn version changes its numerical tree implementation.
    if task_type == "classification":
        scaler = StandardScaler().fit(arrays[0])
        transformed = [scaler.transform(array) for array in arrays]
        estimator = RandomForestClassifier
    else:
        transformed = arrays
        estimator = RandomForestRegressor
    reference = [[], []]
    for seed in range(5):
        model = estimator(**fixture["hparams"][dataset], random_state=seed)
        model.fit(transformed[0], labels[0])
        for index in (1, 2):
            prediction = (model.predict_proba(transformed[index])[:, 1] if task_type == "classification"
                          else model.predict(transformed[index]))
            score = (roc_auc_score(labels[index], prediction) if task_type == "classification"
                     else np.sqrt(mean_squared_error(labels[index], prediction)))
            reference[index - 1].append(score)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-14)
    if sklearn.__version__ == fixture["runtime"]["scikit_learn"]:
        np.testing.assert_allclose(actual, entry["scores"], rtol=0, atol=1e-14)
