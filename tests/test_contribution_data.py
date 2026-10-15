from pathlib import Path

import numpy as np
import pytest

from morace import contribution_analysis


def test_csv_manifest_and_split_loading_contract(tmp_path):
    csv_path = tmp_path / "nested" / "rows.csv"
    contribution_analysis.write_csv(
        csv_path,
        [{"a": "1", "extra": "ignored"}, {"b": "2"}],
        ["a", "b"],
    )
    assert contribution_analysis.read_csv(csv_path) == [
        {"a": "1", "b": ""},
        {"a": "", "b": "2"},
    ]

    split_file = tmp_path / "split.csv"
    split_file.write_text(
        "smiles,target\n CCO ,1\n,2\nCCN,3.5\nCCC,4\n",
        encoding="utf-8",
    )
    assert contribution_analysis.load_smiles_sample(split_file, "smiles", 2) == [
        "CCO",
        "CCN",
    ]
    assert contribution_analysis.load_split_xy_from_manifest(
        split_file,
        "smiles",
        "target",
    ) == ([" CCO ", "", "CCN", "CCC"], [1.0, 2.0, 3.5, 4.0])

    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text(
        "dataset,split_name,split_file_path\n"
        "bbbp,train,relative/train.csv\n"
        "bbbp,valid,relative/valid.csv\n"
        "bbbp,test,relative/test.csv\n"
        ",test,ignored.csv\n",
        encoding="utf-8",
    )
    manifest = contribution_analysis.load_split_manifest(manifest_path)
    assert list(manifest) == [("bbbp", "train"), ("bbbp", "valid"), ("bbbp", "test")]
    assert contribution_analysis.split_paths(tmp_path, manifest, "bbbp") == {
        "train": tmp_path / "relative" / "train.csv",
        "valid": tmp_path / "relative" / "valid.csv",
        "test": tmp_path / "relative" / "test.csv",
    }


def test_candidate_path_resolution_contract(tmp_path):
    absolute_rules = (tmp_path / "absolute-rules.json").resolve()
    paths = contribution_analysis.candidate_paths(
        tmp_path,
        {
            "rules_file": str(absolute_rules),
            "compiled_rules_file": "artifacts/compiled.json",
            "outcome_file": "artifacts/outcome.json",
        },
    )
    assert paths == {
        "rules_file": absolute_rules,
        "compiled_rules_file": tmp_path / "artifacts" / "compiled.json",
        "outcome_file": tmp_path / "artifacts" / "outcome.json",
    }


def test_feature_matrix_sanitization_contract():
    train, valid, test, kept = contribution_analysis.sanitize_feature_matrices(
        [
            [np.nan, 1.0, 1.0, np.nan],
            [np.nan, 1.0, 2.0, 10.0],
            [np.nan, 1.0, 3.0, 20.0],
        ],
        [[5.0, 1.0, np.nan, np.nan]],
        [[0.0, 1.0, 4.0, 30.0]],
        ["all_nan", "constant", "signal", "filled"],
    )
    assert kept == ["signal", "filled"]
    np.testing.assert_allclose(train, [[1.0, 15.0], [2.0, 10.0], [3.0, 20.0]])
    np.testing.assert_allclose(valid, [[2.0, 15.0]])
    np.testing.assert_allclose(test, [[4.0, 30.0]])

    with pytest.raises(RuntimeError, match="Duplicate feature names"):
        contribution_analysis.sanitize_feature_matrices(
            [[1.0, 2.0]],
            [[1.0, 2.0]],
            [[1.0, 2.0]],
            ["same", "same"],
        )
    with pytest.raises(RuntimeError, match="No valid compiled features"):
        contribution_analysis.sanitize_feature_matrices(
            [[1.0], [1.0]],
            [[1.0]],
            [[1.0]],
            ["constant"],
        )
def test_rule_variant_and_source_contract():
    first = {
        "canonical_key": "same",
        "feature_family": "descriptor",
        "support_sources": ["literature", "data"],
    }
    second = {
        "canonical_key": "same",
        "feature_family": "count",
        "support_sources": ["data"],
    }
    third = {
        "audit_origin_canonical_key": "fallback",
        "feature_family": "pattern_presence",
        "support_sources": ["literature"],
    }
    rules = [first, second, third]

    assert contribution_analysis.canonical_key(third) == "fallback"
    assert contribution_analysis.source_type(first) == "multi"
    assert contribution_analysis.source_type(second) == "data"
    assert contribution_analysis.source_type(third) == "literature"
    assert contribution_analysis.source_type({"primary_source": "manual"}) == "manual"

    dropped = contribution_analysis.drop_rule_once(rules, "same")
    assert [contribution_analysis.canonical_key(rule) for rule in dropped] == [
        "same",
        "fallback",
    ]
    assert dropped[0] == second and dropped[0] is not second
    variants = contribution_analysis.family_variants(rules)
    assert list(variants) == ["descriptor", "count", "pattern_presence"]
    assert [rule["feature_family"] for rule in variants["count"]] == [
        "descriptor",
        "pattern_presence",
    ]


def test_scope_and_explicit_pilot_selection_contract():
    rows = [
        {"pilot_id": "PILOT_A_QWEN_D2R_HIV_r1"},
        {"pilot_id": "PILOT_A_QWEN_D2R_HIV_SUPP_r1"},
        {"pilot_id": "OTHER_r1"},
    ]
    assert contribution_analysis.select_scope(rows, "all") is rows
    assert contribution_analysis.select_scope(rows, "p0") == rows[:1]
    assert contribution_analysis.select_scope(rows, "p1") == rows[1:]
    with pytest.raises(ValueError, match="Unsupported scope"):
        contribution_analysis.select_scope(rows, "invalid")

    assert contribution_analysis.select_pilot_ids(rows, "OTHER_r1, PILOT_A_QWEN_D2R_HIV_r1") == [
        rows[2],
        rows[0],
    ]
    with pytest.raises(ValueError, match="required"):
        contribution_analysis.select_pilot_ids(rows, "")
    with pytest.raises(ValueError, match="Unknown pilot_id"):
        contribution_analysis.select_pilot_ids(rows, "missing")
