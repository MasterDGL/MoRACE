import hashlib
import importlib
from pathlib import Path

import pytest


def load_runner():
    return importlib.import_module("morace.contribution_analysis")


@pytest.mark.parametrize("report_name,case,expected_digest", [
    ("feature_importance", "empty", "d206ca4b16753eeeaa24d08bbc5434b4d2300e84a69df11d0712ad8b982ea5b4"),
    ("family_ablation", "empty", "c8c7c724d0c98de0a9954ad8b84e91045a871cb881bfef05ac372380dccee1f7"),
    ("loro_smoke", "empty", "a951b4c335749929f03f318703bdebd369e699e416d51d6f18e40d54048aa150"),
    ("feature_importance", "diagnostics", "ca6c4c1c85eaf6e2fdfe6e308d0a1a807efc42207673d29ca140734423560e9a"),
    ("family_ablation", "diagnostics", "2ed5f6ef5036540219b9c3ab594f6cd9e6cb936db1370581e431c8d61f7f9acf"),
    ("loro_smoke", "diagnostics", "1d2f9a7da841b4931fcf0a4f244b79666347117d9afd2584966811f2e6975fd6"),
])
def test_contribution_reports_keep_full_baseline_text(
    tmp_path, report_name, case, expected_digest,
):
    # Digests captured before refactoring, from commit 06c72f7.
    reporting = importlib.import_module("morace.contribution_reporting")
    base_rows, issues = [], []
    if case == "diagnostics":
        base_rows = [
            {
                "pilot_id": f"PILOT_{i}|row",
                "metric": "roc_auc" if i % 2 else "rmse",
                "outcome_valid_mean": 0.7,
                "reproduced_valid_mean": "0.69",
                "outcome_test_mean": None,
                "reproduced_test_mean": "",
                "base_reproduction_status": status,
            }
            for i, status in enumerate((
                "base_reproduction_warning", "ok", "base_reproduction_warning", "failed", None,
            ))
        ]
        issues = [
            {"pilot_id": "PILOT|0", "stage": "stage|one", "severity": "warn|ing", "issue": "a\\|b\nnext"},
            {"issue": None},
            {},
        ]
    output = tmp_path / "report-contract" / "summary.md"
    output.parent.mkdir()
    kwargs = {"candidate_count": len(base_rows)}
    if report_name != "feature_importance":
        kwargs["read_csv_fn"] = lambda _: []
    if report_name == "family_ablation":
        kwargs["project_root"] = tmp_path
    getattr(reporting, f"write_{report_name}_summary")(
        output, [], base_rows, issues, **kwargs,
    )
    text = output.read_text(encoding="utf-8")
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == expected_digest
    assert ("Base reproduction warnings are cautionary" in text) == (
        case == "diagnostics" and report_name != "loro_smoke"
    )


def test_rule_contribution_reports_preserve_diagnostics_without_historical_scope(tmp_path):
    runner = load_runner()
    base_rows = [
        {
            "pilot_id": "PILOT_A",
            "metric": "roc_auc",
            "outcome_valid_mean": "0.7",
            "reproduced_valid_mean": "0.69",
            "outcome_test_mean": "0.68",
            "reproduced_test_mean": "0.67",
            "base_reproduction_status": "base_reproduction_warning",
        }
    ]
    issue_rows = [
        {
            "pilot_id": "PILOT_A",
            "stage": "report|stage",
            "severity": "warning",
            "issue": "fixture|issue",
        }
    ]
    loro_rows = [
        {
            "pilot_id": "PILOT_A",
            "route_or_model": "route|one",
            "dataset": "bbbp",
            "feature_family": "descriptor",
            "canonical_rule": "descriptor|MolWt",
            "valid_delta": "-0.02",
            "test_delta": "-0.01",
            "contribution_signal": "positive_signal",
        }
    ]
    family_rows = [
        {
            "pilot_id": "PILOT_A",
            "route_or_model": "route|one",
            "dataset": "bbbp",
            "removed_family": "descriptor",
            "removed_rule_count": "2",
            "valid_delta": "-0.03",
            "test_delta": "-0.01",
            "family_contribution_signal": "positive_signal",
        }
    ]
    feature_rows = [
        {
            "pilot_id": "PILOT_A",
            "route_or_model": "route|one",
            "dataset": "bbbp",
            "feature_family": "descriptor",
            "canonical_rule": "descriptor|MolWt",
            "importance_mean": "0.4",
            "rank_mean": "1.0",
            "filtered_by_sanitizer": "no",
            "loro_contribution_signal": "positive_signal",
            "family_contribution_signal": "positive_signal",
            "alignment_label": "aligned_strong_diagnostic",
        }
    ]
    dry_rows = [
        {"dry_run_status": "passed", "dataset": "bbbp"},
        {"dry_run_status": "failed", "dataset": "hiv"},
    ]

    paths = {
        "dry_run": tmp_path / "dry_run.md",
        "protocol": tmp_path / "protocol.md",
        "loro": tmp_path / "loro.md",
        "family": tmp_path / "family.md",
        "feature": tmp_path / "feature.md",
    }
    runner.write_dry_run_summary(paths["dry_run"], dry_rows, "p0", 5)
    runner.write_protocol(paths["protocol"])
    runner.write_loro_smoke_summary(
        paths["loro"],
        loro_rows,
        base_rows,
        issue_rows,
        run_label="smoke",
        candidate_count=1,
    )
    runner.write_family_ablation_summary(
        paths["family"],
        family_rows,
        base_rows,
        issue_rows,
        candidate_count=1,
        project_root=tmp_path,
    )
    runner.write_feature_importance_summary(
        paths["feature"],
        feature_rows,
        base_rows,
        issue_rows,
        candidate_count=1,
    )

    reports = {name: path.read_text(encoding="utf-8") for name, path in paths.items()}
    for content in reports.values():
        assert "20260520" not in content
        assert "Phase 2" not in content
        assert "UCV1" not in content
    assert tmp_path.name in reports["dry_run"]
    assert "all inventory candidates" in reports["dry_run"]
    assert "ablated_valid_score - base_valid_score" in reports["protocol"]
    for name in ("loro", "family", "feature"):
        assert "PILOT_A" in reports[name]
        assert "fixture\\|issue" in reports[name]
        assert "positive_signal" in reports[name]
    assert "Signal Distribution By Route" in reports["loro"]
    assert "descriptor\\|MolWt" in reports["loro"]


def test_dry_run_uses_tracked_manifest_default_without_obsolete_ablation_file(
    tmp_path,
    monkeypatch,
):
    runner = load_runner()
    inventory = tmp_path / "inventory.csv"
    inventory.write_text(
        "pilot_id,route_or_model,dataset,run_tag,seed,metric,metric_direction,"
        "selected_rule_count,compiled_rule_count,rules_file,compiled_rules_file,outcome_file\n"
        "PILOT_A_QWEN_D2R_HIV_fixture,route,bbbp,run,r1,roc_auc,higher,2,2,"
        "missing-rules.json,missing-compiled.json,missing-outcome.json\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "data" / "manifests" / "split_manifest.csv"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        "dataset,split_name,split_file_path,smiles_column\n"
        "bbbp,train,missing-train.csv,smiles\n"
        "bbbp,valid,missing-valid.csv,smiles\n"
        "bbbp,test,missing-test.csv,smiles\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runner.sys,
        "argv",
        [
            "run_rule_contribution_analysis.py",
            "--project-root",
            str(tmp_path),
            "--inventory",
            inventory.name,
            "--dry-run",
            "--output-dir",
            "analysis",
        ],
    )

    assert runner.main() == 1
    assert (tmp_path / "analysis" / "dry_run.csv").exists()
