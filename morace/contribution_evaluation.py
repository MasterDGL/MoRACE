"""Model evaluation backends for rule contribution analysis."""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


_EXTERNAL_WORKER_SETUP = r"""
import json
import math
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

data = np.load(sys.argv[1], allow_pickle=False)
with open(sys.argv[2], "r", encoding="utf-8") as handle:
    payload = json.load(handle)

task_type = payload["task_type"]
params_base = payload["params_base"]
rf_seeds = payload["rf_seeds"]
"""


def find_external_sklearn_python(current_executable: str) -> str:
    candidates = []
    env_path = os.environ.get("RULE_CONTRIB_SKLEARN_PYTHON")
    if env_path:
        candidates.append(env_path)
    candidates.append(current_executable)
    which_python = shutil.which("python")
    if which_python:
        candidates.append(which_python)

    probe = "import sklearn, numpy; print('ok')"
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if not path.exists():
            continue
        try:
            result = subprocess.run(
                [str(path), "-c", probe],
                check=False,
                capture_output=True,
                text=True,
                timeout=20,
            )
        except Exception:
            continue
        if result.returncode == 0:
            return str(path)
    raise ModuleNotFoundError(
        "No sklearn-capable Python found. Set RULE_CONTRIB_SKLEARN_PYTHON to a Python executable with scikit-learn."
    )


def _run_external_sklearn_worker(
    external_python, worker_code, payload, matrices, numpy_module,
    *, prefix, timeout, failure_label,
):
    with tempfile.TemporaryDirectory(prefix=prefix) as tmp:
        tmpdir = Path(tmp)
        data_path = tmpdir / "matrices.npz"
        payload_path = tmpdir / "payload.json"
        names = ("x_train", "x_valid", "x_test", "y_train", "y_valid", "y_test")
        numpy_module.savez(
            data_path,
            **{name: numpy_module.asarray(values, dtype=float) for name, values in zip(names, matrices)},
        )
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        result = subprocess.run(
            [external_python, "-c", worker_code, str(data_path), str(payload_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"{failure_label} failed with {external_python}: {result.stderr.strip() or result.stdout.strip()}"
            )
        return json.loads(result.stdout.strip().splitlines()[-1])


def evaluate_scores_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Tuple[List[float], List[float]]:
    import numpy as np

    external_python = find_external_sklearn_python(sys.executable)
    worker_code = _EXTERNAL_WORKER_SETUP + r"""x_train = data["x_train"]
x_valid = data["x_valid"]
x_test = data["x_test"]
y_train = data["y_train"]
y_valid = data["y_valid"]
y_test = data["y_test"]
valid_scores = []
test_scores = []

if task_type == "classification":
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    x_test = scaler.transform(x_test)
    for seed in rf_seeds:
        params = dict(params_base)
        params["random_state"] = int(seed)
        clf = RandomForestClassifier(**params)
        clf.fit(x_train, y_train)
        valid_scores.append(float(roc_auc_score(y_valid, clf.predict_proba(x_valid)[:, 1])))
        test_scores.append(float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])))
else:
    for seed in rf_seeds:
        params = dict(params_base)
        params["random_state"] = int(seed)
        reg = RandomForestRegressor(**params)
        reg.fit(x_train, y_train)
        valid_scores.append(float(math.sqrt(mean_squared_error(y_valid, reg.predict(x_valid)))))
        test_scores.append(float(math.sqrt(mean_squared_error(y_test, reg.predict(x_test)))))

print(json.dumps({"valid_scores": valid_scores, "test_scores": test_scores}))
"""
    parsed = _run_external_sklearn_worker(
        external_python, worker_code,
        {"task_type": task_type, "params_base": params_base, "rf_seeds": rf_seeds},
        (x_train, x_valid, x_test, y_train, y_valid, y_test), np,
        prefix="rule_contrib_loro_", timeout=600, failure_label="external sklearn worker",
    )
    return [float(x) for x in parsed["valid_scores"]], [float(x) for x in parsed["test_scores"]]


def evaluate_variants_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
    variants: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    import numpy as np

    external_python = find_external_sklearn_python(sys.executable)
    worker_code = _EXTERNAL_WORKER_SETUP + r"""variants = payload["variants"]
x_train_all = data["x_train"]
x_valid_all = data["x_valid"]
x_test_all = data["x_test"]
y_train = data["y_train"]
y_valid = data["y_valid"]
y_test = data["y_test"]
out = {}

for variant in variants:
    name = variant["name"]
    keep_indices = variant["keep_indices"]
    x_train = x_train_all[:, keep_indices]
    x_valid = x_valid_all[:, keep_indices]
    x_test = x_test_all[:, keep_indices]
    valid_scores = []
    test_scores = []
    if task_type == "classification":
        scaler = StandardScaler()
        x_train_eval = scaler.fit_transform(x_train)
        x_valid_eval = scaler.transform(x_valid)
        x_test_eval = scaler.transform(x_test)
        for seed in rf_seeds:
            params = dict(params_base)
            params["random_state"] = int(seed)
            clf = RandomForestClassifier(**params)
            clf.fit(x_train_eval, y_train)
            valid_scores.append(float(roc_auc_score(y_valid, clf.predict_proba(x_valid_eval)[:, 1])))
            test_scores.append(float(roc_auc_score(y_test, clf.predict_proba(x_test_eval)[:, 1])))
    else:
        for seed in rf_seeds:
            params = dict(params_base)
            params["random_state"] = int(seed)
            reg = RandomForestRegressor(**params)
            reg.fit(x_train, y_train)
            valid_scores.append(float(math.sqrt(mean_squared_error(y_valid, reg.predict(x_valid)))))
            test_scores.append(float(math.sqrt(mean_squared_error(y_test, reg.predict(x_test)))))
    out[name] = {"valid_scores": valid_scores, "test_scores": test_scores}
print(json.dumps(out))
"""
    parsed = _run_external_sklearn_worker(
        external_python, worker_code,
        {"task_type": task_type, "params_base": params_base, "rf_seeds": rf_seeds, "variants": variants},
        (x_train, x_valid, x_test, y_train, y_valid, y_test), np,
        prefix="rule_contrib_loro_variants_", timeout=1800, failure_label="external sklearn variant worker",
    )
    return {
        name: {
            "valid_scores": [float(x) for x in scores["valid_scores"]],
            "test_scores": [float(x) for x in scores["test_scores"]],
        }
        for name, scores in parsed.items()
    }


def evaluate_feature_importance_with_external_sklearn(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Dict[str, object]:
    import numpy as np

    external_python = find_external_sklearn_python(sys.executable)
    worker_code = _EXTERNAL_WORKER_SETUP + r"""x_train = data["x_train"]
x_valid = data["x_valid"]
x_test = data["x_test"]
y_train = data["y_train"]
y_valid = data["y_valid"]
y_test = data["y_test"]
valid_scores = []
test_scores = []
importances = []

if task_type == "classification":
    scaler = StandardScaler()
    x_train_eval = scaler.fit_transform(x_train)
    x_valid_eval = scaler.transform(x_valid)
    x_test_eval = scaler.transform(x_test)
    for seed in rf_seeds:
        params = dict(params_base)
        params["random_state"] = int(seed)
        clf = RandomForestClassifier(**params)
        clf.fit(x_train_eval, y_train)
        valid_scores.append(float(roc_auc_score(y_valid, clf.predict_proba(x_valid_eval)[:, 1])))
        test_scores.append(float(roc_auc_score(y_test, clf.predict_proba(x_test_eval)[:, 1])))
        importances.append([float(x) for x in clf.feature_importances_])
else:
    for seed in rf_seeds:
        params = dict(params_base)
        params["random_state"] = int(seed)
        reg = RandomForestRegressor(**params)
        reg.fit(x_train, y_train)
        valid_scores.append(float(math.sqrt(mean_squared_error(y_valid, reg.predict(x_valid)))))
        test_scores.append(float(math.sqrt(mean_squared_error(y_test, reg.predict(x_test)))))
        importances.append([float(x) for x in reg.feature_importances_])

print(json.dumps({"valid_scores": valid_scores, "test_scores": test_scores, "importances": importances}))
"""
    parsed = _run_external_sklearn_worker(
        external_python, worker_code,
        {"task_type": task_type, "params_base": params_base, "rf_seeds": rf_seeds},
        (x_train, x_valid, x_test, y_train, y_valid, y_test), np,
        prefix="rule_contrib_feature_importance_", timeout=1800,
        failure_label="external sklearn feature-importance worker",
    )
    return {
        "valid_scores": [float(x) for x in parsed["valid_scores"]],
        "test_scores": [float(x) for x in parsed["test_scores"]],
        "importances": [[float(x) for x in row] for row in parsed["importances"]],
    }


def evaluate_variants_in_process(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
    variants: List[Dict[str, object]],
) -> Dict[str, Dict[str, List[float]]]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    output: Dict[str, Dict[str, List[float]]] = {}
    for variant in variants:
        name = str(variant["name"])
        keep_indices = list(variant["keep_indices"])
        x_train_variant = x_train[:, keep_indices]
        x_valid_variant = x_valid[:, keep_indices]
        x_test_variant = x_test[:, keep_indices]
        valid_scores: List[float] = []
        test_scores: List[float] = []
        if task_type == "classification":
            scaler = StandardScaler()
            x_train_eval = scaler.fit_transform(x_train_variant)
            x_valid_eval = scaler.transform(x_valid_variant)
            x_test_eval = scaler.transform(x_test_variant)
            for seed in rf_seeds:
                params = dict(params_base)
                params["random_state"] = seed
                classifier = RandomForestClassifier(**params)
                classifier.fit(x_train_eval, y_train)
                valid_scores.append(
                    float(
                        roc_auc_score(
                            y_valid,
                            classifier.predict_proba(x_valid_eval)[:, 1],
                        )
                    )
                )
                test_scores.append(
                    float(
                        roc_auc_score(
                            y_test,
                            classifier.predict_proba(x_test_eval)[:, 1],
                        )
                    )
                )
        else:
            for seed in rf_seeds:
                params = dict(params_base)
                params["random_state"] = seed
                regressor = RandomForestRegressor(**params)
                regressor.fit(x_train_variant, y_train)
                valid_scores.append(
                    float(
                        math.sqrt(
                            mean_squared_error(
                                y_valid,
                                regressor.predict(x_valid_variant),
                            )
                        )
                    )
                )
                test_scores.append(
                    float(
                        math.sqrt(
                            mean_squared_error(
                                y_test,
                                regressor.predict(x_test_variant),
                            )
                        )
                    )
                )
        output[name] = {
            "valid_scores": valid_scores,
            "test_scores": test_scores,
        }
    return output


def evaluate_feature_importance_in_process(
    task_type: str,
    params_base: Dict,
    rf_seeds: List[int],
    x_train,
    y_train: List[float],
    x_valid,
    y_valid: List[float],
    x_test,
    y_test: List[float],
) -> Dict[str, object]:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import mean_squared_error, roc_auc_score
    from sklearn.preprocessing import StandardScaler

    valid_scores: List[float] = []
    test_scores: List[float] = []
    importances: List[List[float]] = []
    if task_type == "classification":
        scaler = StandardScaler()
        x_train_eval = scaler.fit_transform(x_train)
        x_valid_eval = scaler.transform(x_valid)
        x_test_eval = scaler.transform(x_test)
        for seed in rf_seeds:
            params = dict(params_base)
            params["random_state"] = seed
            classifier = RandomForestClassifier(**params)
            classifier.fit(x_train_eval, y_train)
            valid_scores.append(
                float(
                    roc_auc_score(
                        y_valid,
                        classifier.predict_proba(x_valid_eval)[:, 1],
                    )
                )
            )
            test_scores.append(
                float(
                    roc_auc_score(
                        y_test,
                        classifier.predict_proba(x_test_eval)[:, 1],
                    )
                )
            )
            importances.append(
                [float(value) for value in classifier.feature_importances_]
            )
    else:
        for seed in rf_seeds:
            params = dict(params_base)
            params["random_state"] = seed
            regressor = RandomForestRegressor(**params)
            regressor.fit(x_train, y_train)
            valid_scores.append(
                float(
                    math.sqrt(
                        mean_squared_error(y_valid, regressor.predict(x_valid))
                    )
                )
            )
            test_scores.append(
                float(
                    math.sqrt(
                        mean_squared_error(y_test, regressor.predict(x_test))
                    )
                )
            )
            importances.append(
                [float(value) for value in regressor.feature_importances_]
            )
    return {
        "valid_scores": valid_scores,
        "test_scores": test_scores,
        "importances": importances,
    }
