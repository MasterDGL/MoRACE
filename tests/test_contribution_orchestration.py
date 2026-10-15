import json
from types import SimpleNamespace

from morace import contribution_analysis, rule_compiler


def make_orchestration_fixture(tmp_path, monkeypatch):
    rules = [
        {
            "canonical_key": "rule-a",
            "feature_family": "descriptor",
            "feature_name": "MolWt",
            "support_sources": ["data"],
        },
        {
            "canonical_key": "rule-b",
            "feature_family": "count",
            "feature_name": "NumHBD",
            "support_sources": ["literature"],
        },
    ]
    compiled = [
        SimpleNamespace(name="a", rule=rules[0]),
        SimpleNamespace(name="b", rule=rules[1]),
    ]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "rules.json").write_text(
        json.dumps({"rules": rules}),
        encoding="utf-8",
    )
    (artifact_dir / "outcome.json").write_text(
        json.dumps(
            {
                "rf_seeds": [2, 5],
                "metric_name": "roc_auc",
                "actual_valid_mean": 0.8,
                "actual_test_mean": 0.7,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(rule_compiler, "compile_rules", lambda _rules: (compiled, []))
    row = {
        "pilot_id": "PILOT_FIXTURE",
        "route_or_model": "llama",
        "dataset": "bbbp",
        "run_tag": "run",
        "seed": "r1",
        "metric": "fallback_metric",
        "metric_direction": "higher",
        "rules_file": "artifacts/rules.json",
        "compiled_rules_file": "artifacts/compiled.py",
        "outcome_file": "artifacts/outcome.json",
    }
    return row, compiled


def base_scores():
    return {
        "valid_scores": [0.8, 0.82],
        "test_scores": [0.67, 0.71],
        "valid_mean": 0.81,
        "valid_std": 0.01,
        "test_mean": 0.69,
        "test_std": 0.02,
        "note": "",
    }


def test_loro_orchestration_contract(tmp_path, monkeypatch):
    row, compiled = make_orchestration_fixture(tmp_path, monkeypatch)

    def fake_evaluate(project_root, dataset, compiled_rules, manifest, seeds):
        assert (project_root, dataset, compiled_rules, manifest, seeds) == (
            tmp_path,
            "bbbp",
            compiled,
            {},
            [2, 5],
        )
        return (
            {
                "base": base_scores(),
                "drop_0": {
                    **base_scores(),
                    "valid_mean": 0.78,
                    "test_mean": 0.68,
                    "note": "exact_loro_feature_removed_from_matrix",
                },
                "drop_1": {
                    **base_scores(),
                    "valid_mean": 0.82,
                    "test_mean": 0.72,
                    "note": "exact_loro_feature_removed_from_matrix",
                },
            },
            "in_process_sklearn",
        )

    monkeypatch.setattr(
        contribution_analysis,
        "evaluate_compiled_loro_variants",
        fake_evaluate,
    )
    bad_row = dict(row, pilot_id="PILOT_BAD", rules_file="missing.json")
    loro_rows, base_rows, issues = contribution_analysis.run_loro_smoke(
        tmp_path,
        [row, bad_row],
        {},
        0.02,
    )

    assert len(loro_rows) == 2
    assert len(base_rows) == 1
    assert base_rows[0]["base_reproduction_status"] == "reproduced"
    assert base_rows[0]["outcome_valid_mean"] == "0.8"
    assert loro_rows[0]["canonical_rule"] == "rule-a"
    assert loro_rows[0]["source_type"] == "data"
    assert loro_rows[0]["valid_delta"] == "-0.030000000000000027"
    assert loro_rows[0]["contribution_signal"] == "positive_signal"
    assert loro_rows[1]["valid_delta"] == "0.009999999999999898"
    assert loro_rows[1]["contribution_signal"] == "negative_or_harmful_signal"
    assert len(issues) == 1
    assert issues[0]["pilot_id"] == "PILOT_BAD"
    assert issues[0]["stage"] == "run_loro"
    assert issues[0]["severity"] == "error"
    assert issues[0]["issue"].startswith("FileNotFoundError: [Errno 2]")
    assert "missing.json" in issues[0]["issue"]
    assert issues[0]["recommended_action"] == "fix_before_expanding_loro"


def test_family_orchestration_contract(tmp_path, monkeypatch):
    row, _compiled = make_orchestration_fixture(tmp_path, monkeypatch)

    monkeypatch.setattr(
        contribution_analysis,
        "evaluate_compiled_family_variants",
        lambda *_args: (
            {
                "base": base_scores(),
                "drop_family_descriptor": {
                    **base_scores(),
                    "valid_mean": 0.79,
                    "test_mean": 0.68,
                    "note": "family_level_feature_columns_removed_from_matrix",
                },
            },
            {"count": "removing_family_leaves_no_kept_matrix_features"},
            "in_process_sklearn",
        ),
    )
    family_rows, base_rows, issues = contribution_analysis.run_family_ablation(
        tmp_path,
        [row],
        {},
        0.02,
    )

    assert len(base_rows) == 1
    assert len(family_rows) == 2
    assert family_rows[0]["removed_family"] == "descriptor"
    assert family_rows[0]["valid_delta"] == "-0.020000000000000018"
    assert family_rows[0]["family_contribution_signal"] == "positive_signal"
    assert family_rows[1]["removed_family"] == "count"
    assert family_rows[1]["family_contribution_signal"] == "unknown"
    assert "family_ablation_variant_not_evaluated" in family_rows[1]["notes"]
    assert [(issue["severity"], issue["issue"]) for issue in issues] == [
        ("warning", "count: removing_family_leaves_no_kept_matrix_features"),
        ("info", "pattern_presence: family_not_present"),
    ]


def test_feature_importance_orchestration_contract(tmp_path, monkeypatch):
    row, _compiled = make_orchestration_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        contribution_analysis,
        "evaluate_compiled_feature_importance",
        lambda *_args: {
            **base_scores(),
            "importances": [[0.8], [0.6]],
            "kept_features": ["a"],
            "sklearn_mode": "in_process_sklearn",
        },
    )
    monkeypatch.setattr(
        contribution_analysis,
        "load_loro_lookup",
        lambda _root: {
            ("PILOT_FIXTURE", "rule-a"): {
                "valid_delta": "-0.03",
                "contribution_signal": "positive_signal",
            },
            ("PILOT_FIXTURE", "rule-b"): {
                "valid_delta": "0.0",
                "contribution_signal": "near_zero",
            },
        },
    )
    monkeypatch.setattr(
        contribution_analysis,
        "load_family_lookup",
        lambda _root: {
            ("PILOT_FIXTURE", "descriptor"): {
                "valid_delta": "-0.02",
                "family_contribution_signal": "positive_signal",
            }
        },
    )

    feature_rows, base_rows, issues = contribution_analysis.run_feature_importance(
        tmp_path,
        [row],
        {},
        0.02,
    )

    assert issues == []
    assert len(base_rows) == 1
    assert len(feature_rows) == 2
    assert feature_rows[0]["canonical_rule"] == "rule-a"
    assert feature_rows[0]["importance_mean"] == "0.7"
    assert feature_rows[0]["importance_std"] == "0.14142135623730953"
    assert feature_rows[0]["rank_mean"] == "1.0"
    assert feature_rows[0]["filtered_by_sanitizer"] == "no"
    assert feature_rows[0]["alignment_label"] == "aligned_strong_diagnostic"
    assert feature_rows[1]["importance_mean"] == ""
    assert feature_rows[1]["filtered_by_sanitizer"] == "yes"
    assert feature_rows[1]["alignment_label"] == "low_priority"
