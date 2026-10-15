import json

import pytest

from morace import contribution_analysis as runner
from morace.artifact_inputs import resolve_run_rules


def completed_run(root):
    run_dir = root / "artifacts" / "new_run_42"
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(json.dumps({
        "run_tag": run_dir.name, "global_seed": 42, "route_id": "new_route",
        "resolved_config": {"model": "new_model"},
    }), encoding="utf-8")
    rules = [
        {"canonical_key": "descriptor|MolWt", "source": "data", "feature_family": "descriptor", "feature_name": "MolWt", "feature_mode": "continuous", "relation_to_target": "positive"},
        {"canonical_key": "count|NumHBD", "source": "literature", "feature_family": "count", "feature_name": "NumHBD", "feature_mode": "count_value", "relation_to_target": "negative"},
    ]
    details = {}
    for branch in ("synthesize", "inference", "all"):
        folder = root / "outputs" / run_dir.name / branch
        folder.mkdir(parents=True)
        outcome_path = folder / "outcome.json"
        source = "inference_only" if branch == "all" else branch
        outcome_path.write_text(json.dumps({
            "outcome_category": "success_fallback" if branch == "all" else "success_primary",
            "metric_name": "roc_auc", "actual_valid_mean": 0.7, "actual_test_mean": 0.6,
            "selected_rule_count": 2, "compiled_rule_count": 2, "generation_seed": 42,
            "final_result_source": source, "rf_seeds": [0, 1],
        }), encoding="utf-8")
        (folder / f"new_model_bbbp_{branch}_rules.json").write_text(json.dumps({
            "status": "success", "dataset": "bbbp", "rules": rules,
            "final_result_source": source,
            "candidate_all_result": {"rules": [{"must_not_use": True}]},
        }), encoding="utf-8")
        details[branch] = {"result": {
            "artifact_status": "valid", "outcome_path": str(outcome_path.relative_to(root)),
            "final_result_source": source,
        }}
    (run_dir / "summary.json").write_text(json.dumps({
        "run_tag": run_dir.name, "tasks": [{"dataset": "bbbp", "subtask": "", "status": "success",
        "pipeline_exit_status": "success", "branch_details": details,
        "dataset_provenance": {"fixture": True}}],
    }), encoding="utf-8")
    manifest_rows = []
    for split in ("train", "valid", "test"):
        path = root / f"{split}.csv"
        path.write_text("smiles,target\nCCO,0\nNCC,1\nc1ccccc1,1\n", encoding="utf-8")
        manifest_rows.append({"dataset": "bbbp", "split_name": split, "split_file_path": path.name,
                              "smiles_column": "smiles", "target_column": "target"})
    runner.write_csv(root / "manifest.csv", manifest_rows, list(manifest_rows[0]))
    return run_dir


def invoke(monkeypatch, root, *args):
    monkeypatch.setattr(runner.sys, "argv", ["contribution", "--project-root", str(root),
        "--split-manifest", "manifest.csv", *args])
    return runner.main()


def test_completed_run_resolves_final_rules_and_default_all_dry_run(tmp_path, monkeypatch):
    run_dir = completed_run(tmp_path)
    resolved = resolve_run_rules(run_dir)
    assert resolved[0]["final_result_source"] == "inference_only"
    assert len(resolved[0]["metadata"]["rules"]) == 2
    assert resolved[0]["recorded_dataset_provenance"] == {"fixture": True}
    assert invoke(monkeypatch, tmp_path, "--run-dir", str(run_dir), "--dry-run", "--output-dir", "analysis") == 0
    rows = runner.read_csv(tmp_path / "analysis" / "dry_run.csv")
    assert len(rows) == 3
    assert all(row["dry_run_status"] == "passed" for row in rows)
    inventory = runner.read_csv(tmp_path / "analysis" / "inventory.csv")
    assert {row["knowledge_type"] for row in inventory} == {"synthesize", "inference", "all"}
    assert all(row["route_or_model"] == "new_route" for row in inventory)
    assert invoke(monkeypatch, tmp_path, "--inventory", "analysis/inventory.csv", "--check-inputs") == 0


@pytest.mark.parametrize("filters", [("--datasets", "missing"), ("--routes", "missing"), ("--run-ids", "missing"), ("--scope", "p0")])
def test_empty_selection_fails_without_writing_results(tmp_path, monkeypatch, capsys, filters):
    run_dir = completed_run(tmp_path)
    assert invoke(monkeypatch, tmp_path, "--run-dir", str(run_dir), "--dry-run", "--output-dir", "empty", *filters) == 1
    assert "empty_selection" in capsys.readouterr().err
    assert not (tmp_path / "empty").exists()


def test_filters_and_no_implicit_overwrite(tmp_path, monkeypatch, capsys):
    run_dir = completed_run(tmp_path)
    args = ("--run-dir", str(run_dir), "--dry-run", "--output-dir", "analysis",
            "--routes", "new_route", "--datasets", "bbbp", "--run-ids", "new_run_42", "--knowledge-types", "all")
    assert invoke(monkeypatch, tmp_path, *args) == 0
    path = tmp_path / "analysis" / "dry_run.csv"
    before = path.read_bytes()
    assert len(runner.read_csv(path)) == 1
    assert invoke(monkeypatch, tmp_path, *args) == 1
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert path.read_bytes() == before


def test_shared_output_rejects_changed_data_before_alignment(tmp_path, monkeypatch, capsys):
    run_dir = completed_run(tmp_path)
    common = ("--run-dir", str(run_dir), "--output-dir", "analysis")
    assert invoke(monkeypatch, tmp_path, *common, "--dry-run") == 0
    (tmp_path / "train.csv").write_text("smiles,target\nCCO,1\nNCC,0\n", encoding="utf-8")
    assert invoke(monkeypatch, tmp_path, *common, "--run-family-ablation") == 1
    assert "different input files" in capsys.readouterr().err
    assert not (tmp_path / "analysis" / "family_ablation_config.json").exists()


@pytest.mark.parametrize("flag,method,writer", [
    ("--run-loro", "run_loro_smoke", "write_loro_smoke_summary"),
    ("--run-family-ablation", "run_family_ablation", "write_family_ablation_summary"),
    ("--run-feature-importance", "run_feature_importance", "write_feature_importance_summary"),
])
def test_formal_modes_accept_new_run_without_historical_ids(tmp_path, monkeypatch, flag, method, writer):
    run_dir = completed_run(tmp_path)
    calls = []
    def evaluate(_root, rows, _manifest, _threshold, **kwargs):
        calls.append(rows)
        return [{"pilot_id": rows[0]["pilot_id"]}], [], []
    monkeypatch.setattr(runner, method, evaluate)
    monkeypatch.setattr(runner, writer, lambda *args, **kwargs: None)
    assert invoke(monkeypatch, tmp_path, "--run-dir", str(run_dir), flag, "--output-dir", "analysis") == 0
    assert len(calls[0]) == 3
    assert all(row["pilot_id"].startswith("new_run_42:") for row in calls[0])


@pytest.mark.parametrize("problem", ["incomplete", "ambiguous", "source_mismatch", "dataset_mismatch"])
def test_run_resolver_rejects_inconsistent_artifacts(tmp_path, problem):
    run_dir = completed_run(tmp_path)
    rules_path = tmp_path / "outputs" / run_dir.name / "all" / "new_model_bbbp_all_rules.json"
    if problem == "incomplete":
        path = run_dir / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["tasks"][0]["pipeline_exit_status"] = "failed"
        path.write_text(json.dumps(payload), encoding="utf-8")
    elif problem == "ambiguous":
        rules_path.with_name("other_bbbp_all_rules.json").write_bytes(rules_path.read_bytes())
    else:
        payload = json.loads(rules_path.read_text(encoding="utf-8"))
        payload["final_result_source" if problem == "source_mismatch" else "dataset"] = "mismatch"
        rules_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        resolve_run_rules(run_dir)


def test_generic_alignment_lookups_use_explicit_analysis_directory(tmp_path):
    runner.write_csv(tmp_path / "loro.csv", [{"pilot_id": "new", "canonical_rule": "rule", "valid_delta": "-0.1"}],
                     ["pilot_id", "canonical_rule", "valid_delta"])
    runner.write_csv(tmp_path / "family_ablation.csv", [{"pilot_id": "new", "removed_family": "count"}],
                     ["pilot_id", "removed_family"])
    assert runner.load_loro_lookup(tmp_path)[("new", "rule")]["valid_delta"] == "-0.1"
    assert ("new", "count") in runner.load_family_lookup(tmp_path)


def test_run_resolver_follows_authoritative_metadata_link(tmp_path):
    run_dir = completed_run(tmp_path)
    folder = tmp_path / "outputs" / run_dir.name / "all"
    old = folder / "new_model_bbbp_all_rules.json"
    linked = folder / "final_rules.json"
    old.rename(linked)
    outcome_path = folder / "outcome.json"
    payload = json.loads(outcome_path.read_text(encoding="utf-8"))
    payload["artifact_paths"] = {"metadata": str(linked.relative_to(tmp_path))}
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    assert resolve_run_rules(run_dir)[0]["rules_path"] == linked
    payload["dataset"] = "hiv"
    outcome_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="dataset differs"):
        resolve_run_rules(run_dir)


def test_check_inputs_rejects_empty_inventory_and_directory_paths(tmp_path):
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("dataset,split_name,split_file_path\n", encoding="utf-8")
    ok, issues = runner.summarize_check_inputs(tmp_path, [], manifest)
    assert not ok
    assert "empty contribution inventory" in issues
    ok, issues = runner.summarize_check_inputs(tmp_path, [{"pilot_id": "new", "dataset": "bbbp"}], manifest)
    assert not ok
    assert any("missing rules_file" in issue for issue in issues)
