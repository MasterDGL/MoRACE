import json

from morace import contribution_analysis


def make_audit_fixture(tmp_path):
    rules = [
        {
            "canonical_key": "descriptor|MolWt|continuous|positive|null",
            "source": "data",
            "feature_family": "descriptor",
            "feature_name": "MolWt",
            "feature_mode": "continuous",
            "relation_to_target": "positive",
        },
        {
            "canonical_key": "count|NumHBD|count_value|negative|null",
            "source": "literature",
            "feature_family": "count",
            "feature_name": "NumHBD",
            "feature_mode": "count_value",
            "relation_to_target": "negative",
        },
        {
            "canonical_key": "pattern_presence|aromatic_ring|binary_presence|positive|null",
            "source": "data",
            "feature_family": "pattern_presence",
            "feature_name": "aromatic_ring",
            "feature_mode": "binary_presence",
            "relation_to_target": "positive",
        },
    ]
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "rules.json").write_text(
        json.dumps({"rules": rules}),
        encoding="utf-8",
    )
    (artifact_dir / "compiled.py").write_text("# fixture\n", encoding="utf-8")
    (artifact_dir / "outcome.json").write_text("{}\n", encoding="utf-8")

    manifest = {}
    for split in ("train", "valid", "test"):
        split_path = tmp_path / "data" / f"{split}.csv"
        split_path.parent.mkdir(exist_ok=True)
        split_path.write_text(
            "smiles,target\nCCO,0\nNCC,1\nc1ccccc1,1\n",
            encoding="utf-8",
        )
        manifest[("bbbp", split)] = {
            "dataset": "bbbp",
            "split_name": split,
            "split_file_path": str(split_path.relative_to(tmp_path)),
            "smiles_column": "smiles",
            "target_column": "target",
        }

    row = {
        "pilot_id": "PILOT_FIXTURE",
        "route_or_model": "fixture",
        "dataset": "bbbp",
        "run_tag": "run",
        "seed": "r1",
        "metric": "roc_auc",
        "metric_direction": "higher",
        "selected_rule_count": "3",
        "compiled_rule_count": "3",
        "rules_file": "artifacts/rules.json",
        "compiled_rules_file": "artifacts/compiled.py",
        "outcome_file": "artifacts/outcome.json",
    }
    return row, manifest


def write_manifest(path, manifest):
    path.write_text(
        "dataset,split_name,split_file_path,smiles_column,target_column\n"
        + "".join(
            f"{row['dataset']},{row['split_name']},{row['split_file_path']},"
            f"{row['smiles_column']},{row['target_column']}\n"
            for row in manifest.values()
        ),
        encoding="utf-8",
    )


def test_contribution_input_audit_success_contract(tmp_path):
    row, manifest = make_audit_fixture(tmp_path)
    manifest_path = tmp_path / "manifest.csv"
    write_manifest(manifest_path, manifest)

    ok, issues = contribution_analysis.summarize_check_inputs(
        tmp_path,
        [row],
        manifest_path,
    )
    result = contribution_analysis.dry_run_candidate(tmp_path, row, manifest, 3)

    assert ok is True
    assert issues == []
    assert result["dry_run_status"] == "passed"
    assert result["can_build_base_rule_matrix"] == "yes"
    assert result["can_prepare_loro"] == "yes"
    assert result["can_prepare_family_ablation"] == "yes"
    assert result["can_prepare_feature_importance"] == "yes"
    assert result["issue"] == ""
    assert result["recommended_next_action"] == "ready_for_contribution_analysis"


def test_contribution_input_audit_metadata_contract(tmp_path):
    row, manifest = make_audit_fixture(tmp_path)

    selected_mismatch = dict(row, selected_rule_count="4")
    selected_result = contribution_analysis.dry_run_candidate(
        tmp_path,
        selected_mismatch,
        manifest,
        2,
    )
    assert selected_result["dry_run_status"] == "passed"
    assert selected_result["issue"] == (
        "selected_rule_count_metadata_mismatch: rules_array=3 inventory=4"
    )

    compiled_mismatch = dict(row, compiled_rule_count="4")
    compiled_result = contribution_analysis.dry_run_candidate(
        tmp_path,
        compiled_mismatch,
        manifest,
        2,
    )
    assert compiled_result["dry_run_status"] == "failed"
    assert "compiled_rule_count_mismatch: compiled=3 inventory=4" in compiled_result["issue"]
    assert compiled_result["recommended_next_action"] == "fix_dry_run_issue_before_analysis"


def test_contribution_input_audit_reports_missing_files(tmp_path):
    row, _manifest = make_audit_fixture(tmp_path)
    missing_manifest = tmp_path / "missing-manifest.csv"

    ok, issues = contribution_analysis.summarize_check_inputs(
        tmp_path,
        [row],
        missing_manifest,
    )

    assert ok is False
    assert issues[0] == f"missing split manifest: {missing_manifest}"
    assert any("missing train split" in issue for issue in issues)
    assert any("missing valid split" in issue for issue in issues)
    assert any("missing test split" in issue for issue in issues)
