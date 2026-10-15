import json
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from morace.feature_experiments import (
    ARMS, ROOT, SPLITS, evaluate_one_seed, load_splits,
    make_gaussian_count_matched_controls,
    make_split_local_shuffled_controls, resolve_rule_source, run_experiment,
    sanitize_rule_matrices, stable_seed, P13_PHASE,
)


def rule(name="MolWt"):
    return {"source": "data", "feature_family": "descriptor", "feature_name": name,
            "feature_mode": "continuous", "relation_to_target": "positive", "threshold": None}


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def small_inputs(tmp_path, dataset="bbbp"):
    data_root = tmp_path / "data"
    folder = data_root / dataset
    folder.mkdir(parents=True)
    smiles = ["C", "CC", "CCC", "CCCC", "CCO", "CCN", "CO", "CN", "CCCO", "CCCN", "CCCl", "CCBr"]
    target = "p_np" if dataset == "bbbp" else "calc"
    for split in SPLITS:
        labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1] if dataset == "bbbp" else list(range(12))
        pd.DataFrame({"smiles": smiles, target: labels}).to_csv(folder / f"{dataset}_{split}.csv", index=False)
    rules = write_json(tmp_path / "final_rules.json", {
        "dataset": dataset, "status": "success", "final_result_source": "inference_only",
        "rules": [rule()], "candidate_all_result": {"rules": [rule("TPSA")]},
    })
    registry = write_json(tmp_path / "registry.json", {dataset: {"n_estimators": 7, "max_depth": 3, "n_jobs": 1}})
    return data_root, rules, registry


def test_fallback_top_level_rules_and_dataset_identity(tmp_path):
    _, source, _ = small_inputs(tmp_path)
    rules, info = resolve_rule_source(source, "bbbp")
    assert rules == [rule()]
    assert info["final_result_source"] == "inference_only"
    with pytest.raises(ValueError, match="does not match"):
        resolve_rule_source(source, "bace")


def test_outcome_source_disagreement_is_rejected(tmp_path):
    _, source, _ = small_inputs(tmp_path)
    outcome = write_json(tmp_path / "outcome.json", {
        "dataset": "bbbp", "outcome_category": "success_primary", "final_result_source": "all_promoted",
        "artifact_paths": {"metadata": source.name},
    })
    with pytest.raises(ValueError, match="disagree"):
        resolve_rule_source(outcome, "bbbp", tmp_path)


def test_run_directory_follows_final_outcome_and_keeps_hash_chain(tmp_path):
    _, source, _ = small_inputs(tmp_path)
    metadata = json.loads(source.read_text(encoding="utf-8"))
    rules_path = write_json(tmp_path / "eval/model_bbbp_all_rules.json", metadata)
    outcome_path = write_json(tmp_path / "eval/outcome.json", {
        "dataset": "bbbp", "outcome_category": "success_fallback", "final_result_source": "inference_only",
        "artifact_paths": {"metadata": str(rules_path)},
    })
    run_dir = tmp_path / "artifacts/one_run"
    result = {"artifact_status": "valid", "outcome_category": "success_fallback",
              "outcome_path": str(outcome_path), "final_result_source": "inference_only"}
    write_json(run_dir / "summary.json", {"run_tag": "one_run", "tasks": [{
        "dataset": "bbbp", "subtask": "", "pipeline_exit_status": "success", "status": "success",
        "branch_details": {"all": {"result": result}}, "result": result,
    }]})
    write_json(run_dir / "run_config.json", {"run_tag": "one_run", "model": "example"})
    rules, info = resolve_rule_source(run_dir, "bbbp")
    assert rules == [rule()]
    assert info["run_tag"] == "one_run"
    assert len(info["artifact_chain"]) == 4
    assert info["final_result_source"] == "inference_only"


def test_train_only_sanitization_and_p13_duplicate_policy():
    train = np.array([[1, 2, np.nan, 4, 9], [2, 2, np.nan, 8, 8], [3, 2, np.nan, np.nan, 7]])
    valid = np.array([[99, 99, 0, np.nan, 1]])
    matrices, kept, info = sanitize_rule_matrices({"train": train, "valid": valid, "test": valid}, ["a", "constant", "nan", "b", "a"])
    assert kept == ["a", "b"]
    np.testing.assert_array_equal(matrices["valid"], np.array([[99, 6]], dtype=np.float32))
    assert info["duplicate_feature_names"] == "a"


def test_controls_preserve_split_marginals_and_train_only_gaussian_moments():
    matrices = {split: np.arange(24, dtype=np.float32).reshape(8, 3) + i * 100 for i, split in enumerate(SPLITS)}
    shuffled = make_split_local_shuffled_controls(matrices, 872)
    random = make_gaussian_count_matched_controls(matrices, 873)
    for split in SPLITS:
        np.testing.assert_array_equal(np.sort(shuffled[split], axis=0), np.sort(matrices[split], axis=0))
        assert random[split].shape == matrices[split].shape
        assert np.isfinite(random[split]).all()
    changed = dict(matrices, valid=matrices["valid"] * 1000)
    gaussian_changed = make_gaussian_count_matched_controls(changed, 873)
    for split in SPLITS:
        np.testing.assert_array_equal(random[split], gaussian_changed[split])


def test_p13_golden_control_vectors_and_historical_seed_key():
    # Values obtained from the original P13 helper, not computed by this module.
    matrices = {s: np.arange(n * 2, dtype=np.float32).reshape(n, 2) + i * 100
                for i, (s, n) in enumerate(zip(SPLITS, (4, 3, 2)))}
    expected_shuffle = {
        "train": [[6, 7], [4, 5], [2, 1], [0, 3]],
        "valid": [[100, 103], [104, 101], [102, 105]], "test": [[202, 201], [200, 203]],
    }
    shuffled = make_split_local_shuffled_controls(matrices, 42)
    for split in SPLITS:
        np.testing.assert_array_equal(shuffled[split], np.array(expected_shuffle[split], dtype=np.float32))
    gaussian = make_gaussian_count_matched_controls(matrices, 42)
    expected_first_rows = {"train": [3.681368112564087, 1.6745247840881348],
                           "valid": [1.795894980430603, 2.4727325439453125],
                           "test": [4.254734992980957, 2.647022247314453]}
    for split in SPLITS:
        np.testing.assert_allclose(gaussian[split][0], expected_first_rows[split], rtol=1e-7)
    assert stable_seed(P13_PHASE, "llama", "bbbp", "r1", "shuffled_rules") == 701630874
    assert stable_seed(P13_PHASE, "llama", "bbbp", "r1", "random_features") == 2382249619


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_actual_rf_matches_direct_historical_preprocessing(task_type):
    rng = np.random.default_rng(194)
    matrices = {split: rng.normal(size=(20, 4)).astype(np.float32) for split in SPLITS}
    labels = {split: np.tile([0.0, 1.0], 10) if task_type == "classification" else rng.normal(size=20) for split in SPLITS}
    params = {"n_estimators": 7, "max_depth": 3, "n_jobs": 1}
    scores = evaluate_one_seed(task_type, matrices, labels, params, 17)
    if task_type == "classification":
        scaler = StandardScaler().fit(matrices["train"])
        clf = RandomForestClassifier(**params, random_state=17).fit(scaler.transform(matrices["train"]), labels["train"].astype(int))
        expected = {split: roc_auc_score(labels[split], clf.predict_proba(scaler.transform(matrices[split]))[:, 1]) for split in ("valid", "test")}
    else:
        reg = RandomForestRegressor(**params, random_state=17).fit(matrices["train"], labels["train"])
        expected = {split: np.sqrt(mean_squared_error(labels[split], reg.predict(matrices[split]))) for split in ("valid", "test")}
    assert scores == expected
    assert "random_state" not in params


@pytest.mark.parametrize("dataset", ["bbbp", "freesolv"])
def test_five_arms_small_end_to_end_and_no_overwrite(tmp_path, dataset):
    data_root, source, registry = small_inputs(tmp_path, dataset)
    output = run_experiment(dataset=dataset, source=source, output_dir=tmp_path / "result",
                            data_root=data_root, registry_path=registry, rf_seeds=[2, 11])
    rows = pd.read_csv(output / "per_seed.csv")
    assert len(rows) == 20
    assert set(rows.arm) == set(ARMS)
    assert set(rows.rf_seed) == {2, 11}
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["source"]["final_result_source"] == "inference_only"
    assert protocol["rf_seeds"] == [2, 11]
    assert protocol["test_used_for_selection"] is False
    assert protocol["features"]["compiled_rule_count"] == 1
    base = rows[rows.arm == "base"].set_index(["rf_seed", "split"])["score"]
    for row in rows.itertuples():
        sign = 1 if dataset == "bbbp" else -1
        assert row.directional_delta_vs_base == pytest.approx(sign * (row.score - base.loc[row.rf_seed, row.split]))
    prior = (output / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_experiment(dataset=dataset, source=source, output_dir=output)
    assert (output / "protocol.json").read_bytes() == prior


def test_data_hash_mismatch_and_invalid_smiles_never_drop_rows(tmp_path):
    data_root, _, _ = small_inputs(tmp_path)
    with pytest.raises(ValueError, match="recorded split hash"):
        load_splits("bbbp", data_root, {"splits": {"train": {"file_sha256": "bad"}}})
    path = data_root / "bbbp/bbbp_train.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "smiles"] = "invalid molecule"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="invalid SMILES"):
        load_splits("bbbp", data_root)


def test_rf_failure_is_recorded_without_fake_scores(tmp_path, monkeypatch):
    data_root, source, registry = small_inputs(tmp_path)
    def fail(*args, **kwargs):
        raise RuntimeError("simulated estimator failure")
    monkeypatch.setattr("morace.feature_experiments.evaluate_one_seed", fail)
    output = tmp_path / "failed_run"
    with pytest.raises(RuntimeError, match="simulated estimator failure"):
        run_experiment(dataset="bbbp", source=source, output_dir=output, data_root=data_root,
                       registry_path=registry, rf_seeds=[0])
    assert json.loads((output / "status.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (output / "protocol.json").is_file()
    assert not (output / "summary.csv").exists()


def test_50_seed_profile_and_no_generation_seed_conflation():
    config = json.loads((ROOT / "configs/feature_experiments.json").read_text(encoding="utf-8"))
    assert config["rf_seed_profiles"]["fifty"] == list(range(50))
    assert config["fingerprints"]["freesolv"]["radius"] == 3
    assert config["fingerprints"]["hiv"]["radius"] == 3
