import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from morace import contribution_analysis, contribution_evaluation


EXTERNAL_WORKERS = [
    ("evaluate_scores_with_external_sklearn", 600, "rule_contrib_loro_", "external sklearn worker",
     "f088dc6dffadc3bafb2bb357f39172c63d7503a4e7ae3e0e7bb2e66b7d908ec4"),
    ("evaluate_variants_with_external_sklearn", 1800, "rule_contrib_loro_variants_", "external sklearn variant worker",
     "33c638d3d565e0e079670bc4e3ad665824b04bc9a673549134170122dbc2a464"),
    ("evaluate_feature_importance_with_external_sklearn", 1800, "rule_contrib_feature_importance_",
     "external sklearn feature-importance worker",
     "6bf1a7f074f1a6fdc94aa29bf6c2009669cf8041f7cc09db672feae1667b1b58"),
]


def evaluation_data(task_type):
    x_train = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0],
            [2.0, 1.0],
        ]
    )
    x_valid = np.asarray([[0.1, 0.2], [0.8, 0.9], [1.5, 0.1], [2.0, 1.0]])
    x_test = np.asarray([[0.2, 0.8], [0.9, 0.2], [1.8, 0.9], [2.2, 0.1]])
    if task_type == "classification":
        return x_train, [0, 0, 0, 1, 1, 1], x_valid, [0, 0, 1, 1], x_test, [0, 0, 1, 1]
    return (
        x_train,
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        x_valid,
        [0.2, 1.0, 1.6, 2.7],
        x_test,
        [0.4, 0.9, 2.2, 2.4],
    )


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_external_variant_and_score_evaluation_matches_in_process(
    monkeypatch, task_type
):
    monkeypatch.setenv("RULE_CONTRIB_SKLEARN_PYTHON", sys.executable)
    x_train, y_train, x_valid, y_valid, x_test, y_test = evaluation_data(task_type)
    params = {"n_estimators": 5, "max_depth": 3, "n_jobs": 1}
    seeds = [3, 7]
    variants = [
        {"name": "base", "keep_indices": [0, 1]},
        {"name": "first_only", "keep_indices": [0]},
    ]

    expected = contribution_analysis.evaluate_variants_in_process(
        task_type,
        params,
        seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        variants,
    )
    actual = contribution_analysis.evaluate_variants_with_external_sklearn(
        task_type,
        params,
        seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        variants,
    )
    valid_scores, test_scores = contribution_analysis.evaluate_scores_with_external_sklearn(
        task_type,
        params,
        seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )

    for name in expected:
        np.testing.assert_allclose(actual[name]["valid_scores"], expected[name]["valid_scores"])
        np.testing.assert_allclose(actual[name]["test_scores"], expected[name]["test_scores"])
    np.testing.assert_allclose(valid_scores, expected["base"]["valid_scores"])
    np.testing.assert_allclose(test_scores, expected["base"]["test_scores"])


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_external_feature_importance_matches_in_process(monkeypatch, task_type):
    monkeypatch.setenv("RULE_CONTRIB_SKLEARN_PYTHON", sys.executable)
    x_train, y_train, x_valid, y_valid, x_test, y_test = evaluation_data(task_type)
    params = {"n_estimators": 5, "max_depth": 3, "n_jobs": 1}
    seeds = [3, 7]

    expected = contribution_analysis.evaluate_feature_importance_in_process(
        task_type,
        params,
        seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )
    actual = contribution_analysis.evaluate_feature_importance_with_external_sklearn(
        task_type,
        params,
        seeds,
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )

    np.testing.assert_allclose(actual["valid_scores"], expected["valid_scores"])
    np.testing.assert_allclose(actual["test_scores"], expected["test_scores"])
    np.testing.assert_allclose(actual["importances"], expected["importances"])


@pytest.mark.parametrize("method, timeout, prefix, label, worker_hash", EXTERNAL_WORKERS)
def test_external_worker_preserves_paper_code_and_input_protocol(
    monkeypatch, method, timeout, prefix, label, worker_hash,
):
    inputs = evaluation_data("classification")
    params = {"n_estimators": 5, "max_depth": 3, "n_jobs": 1}
    seeds = [3, 7]
    variants = [{"name": "base", "keep_indices": [0, 1]}]
    scores = {"valid_scores": ["0.5", "0.75"], "test_scores": ["0.6", "0.8"]}
    response = dict(scores, importances=[["0.4", "0.6"], ["0.3", "0.7"]])
    if "variants" in method:
        response = {"base": scores}
    temporary_paths = []
    monkeypatch.setattr(contribution_evaluation, "find_external_sklearn_python", lambda _path: "worker-python")

    def run(command, **kwargs):
        assert command[:2] == ["worker-python", "-c"]
        assert hashlib.sha256(command[2].encode("utf-8")).hexdigest() == worker_hash
        assert kwargs == {"check": False, "capture_output": True, "text": True, "timeout": timeout}
        data_path, payload_path = map(Path, command[3:])
        assert data_path.parent == payload_path.parent
        assert data_path.parent.name.startswith(prefix)
        temporary_paths.append(data_path.parent)
        with np.load(data_path, allow_pickle=False) as arrays:
            names = ("x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test")
            assert arrays.files == list(names)
            for name, index in zip(names, (0, 2, 4, 1, 3, 5)):
                assert arrays[name].dtype == np.dtype(float)
                np.testing.assert_array_equal(arrays[name], inputs[index])
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        expected_payload = {"task_type": "classification", "params_base": params, "rf_seeds": seeds}
        if "variants" in method:
            expected_payload["variants"] = variants
        assert payload == expected_payload
        return SimpleNamespace(returncode=0, stdout="worker log\n" + json.dumps(response), stderr="")

    monkeypatch.setattr(contribution_evaluation.subprocess, "run", run)
    arguments = ["classification", params, seeds, *inputs]
    if "variants" in method:
        arguments.append(variants)
    actual = getattr(contribution_evaluation, method)(*arguments)
    expected_scores = {"valid_scores": [0.5, 0.75], "test_scores": [0.6, 0.8]}
    if "variants" in method:
        assert actual == {"base": expected_scores}
    elif "importance" in method:
        assert actual == dict(expected_scores, importances=[[0.4, 0.6], [0.3, 0.7]])
    else:
        assert actual == ([0.5, 0.75], [0.6, 0.8])
    assert len(temporary_paths) == 1 and not temporary_paths[0].exists()
    assert params == {"n_estimators": 5, "max_depth": 3, "n_jobs": 1}
    assert seeds == [3, 7]


@pytest.mark.parametrize("method, timeout, prefix, label, worker_hash", EXTERNAL_WORKERS)
@pytest.mark.parametrize("stderr, detail", [(" problem \n", "problem"), (" \n", "stdout problem")])
def test_external_worker_preserves_failure_messages(
    monkeypatch, method, timeout, prefix, label, worker_hash, stderr, detail,
):
    monkeypatch.setattr(contribution_evaluation, "find_external_sklearn_python", lambda _path: "worker-python")
    monkeypatch.setattr(
        contribution_evaluation.subprocess, "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=7, stdout="stdout problem\n", stderr=stderr),
    )
    arguments = ["classification", {"n_estimators": 5}, [3, 7], *evaluation_data("classification")]
    if "variants" in method:
        arguments.append([{"name": "base", "keep_indices": [0, 1]}])
    with pytest.raises(RuntimeError) as failure:
        getattr(contribution_evaluation, method)(*arguments)
    assert str(failure.value) == f"{label} failed with worker-python: {detail}"
