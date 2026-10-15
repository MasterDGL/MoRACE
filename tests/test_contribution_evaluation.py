import numpy as np

from morace import contribution_analysis


def install_fake_sklearn(monkeypatch):
    import sklearn.ensemble
    import sklearn.metrics
    import sklearn.preprocessing

    calls = {"scaled": [], "classifier_fits": [], "regressor_fits": []}

    class FakeScaler:
        def fit_transform(self, values):
            array = np.asarray(values, dtype=float) + 10.0
            calls["scaled"].append(("fit", array.copy()))
            return array

        def transform(self, values):
            array = np.asarray(values, dtype=float) + 10.0
            calls["scaled"].append(("transform", array.copy()))
            return array

    class FakeClassifier:
        def __init__(self, **params):
            self.params = params
            seed = int(params["random_state"])
            self.feature_importances_ = np.asarray([seed, seed + 1], dtype=float)

        def fit(self, values, targets):
            calls["classifier_fits"].append(
                (self.params.copy(), np.asarray(values).copy(), list(targets))
            )

        def predict_proba(self, values):
            probability = 0.5 + int(self.params["random_state"]) / 100.0
            return np.asarray([[1.0 - probability, probability] for _ in values])

    class FakeRegressor:
        def __init__(self, **params):
            self.params = params
            seed = int(params["random_state"])
            self.feature_importances_ = np.asarray([seed + 2, seed + 3], dtype=float)

        def fit(self, values, targets):
            calls["regressor_fits"].append(
                (self.params.copy(), np.asarray(values).copy(), list(targets))
            )

        def predict(self, values):
            return np.full(len(values), float(self.params["random_state"]))

    def fake_auc(_targets, probabilities):
        return float(np.asarray(probabilities)[0])

    def fake_mse(targets, predictions):
        targets_array = np.asarray(targets, dtype=float)
        predictions_array = np.asarray(predictions, dtype=float)
        return float(np.mean((targets_array - predictions_array) ** 2))

    monkeypatch.setattr(sklearn.ensemble, "RandomForestClassifier", FakeClassifier)
    monkeypatch.setattr(sklearn.ensemble, "RandomForestRegressor", FakeRegressor)
    monkeypatch.setattr(sklearn.metrics, "roc_auc_score", fake_auc)
    monkeypatch.setattr(sklearn.metrics, "mean_squared_error", fake_mse)
    monkeypatch.setattr(sklearn.preprocessing, "StandardScaler", FakeScaler)
    return calls


def matrices():
    return (
        np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        [0.0, 1.0],
        np.asarray([[5.0, 6.0], [7.0, 8.0]]),
        [0.0, 1.0],
        np.asarray([[9.0, 10.0], [11.0, 12.0]]),
        [1.0, 0.0],
    )


def test_in_process_variant_evaluation_contract(monkeypatch):
    calls = install_fake_sklearn(monkeypatch)
    x_train, y_train, x_valid, y_valid, x_test, y_test = matrices()
    params = {"n_estimators": 7}
    variants = [
        {"name": "base", "keep_indices": [0, 1]},
        {"name": "second_only", "keep_indices": [1]},
    ]

    classification = contribution_analysis.evaluate_variants_in_process(
        "classification",
        params,
        [1, 2],
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
        variants,
    )
    assert classification == {
        "base": {"valid_scores": [0.51, 0.52], "test_scores": [0.51, 0.52]},
        "second_only": {"valid_scores": [0.51, 0.52], "test_scores": [0.51, 0.52]},
    }
    assert params == {"n_estimators": 7}
    assert [call[0]["random_state"] for call in calls["classifier_fits"]] == [1, 2, 1, 2]
    assert [call[1].shape[1] for call in calls["classifier_fits"]] == [2, 2, 1, 1]
    assert len(calls["scaled"]) == 6
    assert all(np.min(call[1]) >= 11.0 for call in calls["scaled"])

    calls["scaled"].clear()
    regression = contribution_analysis.evaluate_variants_in_process(
        "regression",
        params,
        [1, 2],
        x_train,
        y_train,
        x_valid,
        [0.0, 0.0],
        x_test,
        [0.0, 0.0],
        variants[:1],
    )
    assert regression == {
        "base": {"valid_scores": [1.0, 2.0], "test_scores": [1.0, 2.0]}
    }
    assert calls["scaled"] == []


def test_in_process_feature_importance_contract(monkeypatch):
    calls = install_fake_sklearn(monkeypatch)
    x_train, y_train, x_valid, y_valid, x_test, y_test = matrices()
    params = {"n_estimators": 7}

    classification = contribution_analysis.evaluate_feature_importance_in_process(
        "classification",
        params,
        [1, 2],
        x_train,
        y_train,
        x_valid,
        y_valid,
        x_test,
        y_test,
    )
    assert classification == {
        "valid_scores": [0.51, 0.52],
        "test_scores": [0.51, 0.52],
        "importances": [[1.0, 2.0], [2.0, 3.0]],
    }
    assert len(calls["scaled"]) == 3

    calls["scaled"].clear()
    regression = contribution_analysis.evaluate_feature_importance_in_process(
        "regression",
        params,
        [1, 2],
        x_train,
        y_train,
        x_valid,
        [0.0, 0.0],
        x_test,
        [0.0, 0.0],
    )
    assert regression == {
        "valid_scores": [1.0, 2.0],
        "test_scores": [1.0, 2.0],
        "importances": [[3.0, 4.0], [4.0, 5.0]],
    }
    assert calls["scaled"] == []
