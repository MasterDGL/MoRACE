import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from morace import raw_closure as closure


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = json.loads(
    (ROOT / "tests/fixtures/raw_closure_v1.json").read_text(encoding="utf-8")
)


@pytest.mark.parametrize(
    "case", FIXTURE["cases"], ids=lambda case: case["candidate"]["raw_candidate_id"]
)
def test_frozen_projection_matches_synthetic_contract_case(case):
    assert closure.project_candidates([case["candidate"]]) == [case["expected"]]


def project(text, **extra):
    candidates = closure.adapt_candidates(
        [{"raw_candidate_id": "claim", "natural_language_rule_text": text, **extra}],
        input_mode="candidates",
    )
    return closure.project_candidates(candidates)[0]


def test_matching_priority_and_direction_are_frozen():
    row = project("molecular weight and amide increase")
    assert row["projected_feature_name"] == "MolWt"
    assert row["projected_relation_or_direction"] == "positive"
    assert (
        project("molecular weight", direction_hint="positive")[
            "projected_relation_or_direction"
        ]
        == "unspecified"
    )
    assert (
        project("molecular weight increases", direction_hint="negative")[
            "rejected_reason"
        ]
        == "direction_conflict"
    )
    assert (
        project("solubility")["unsupported_reason"]
        == "task_label_semantics_without_compiler_supported_molecular_feature"
    )
    assert (
        project("unmapped fragment")["unsupported_reason"]
        == "no_supported_projection_pattern"
    )


def test_nonconstant_boundary_retains_original_tolerance():
    assert not closure.nonconstant([])
    assert not closure.nonconstant([1.0])
    assert not closure.nonconstant([0.0, 1e-12])
    assert closure.nonconstant([0.0, 1.0001e-12])
    assert not closure.nonconstant([float("nan"), float("inf"), 1.0])


def test_compile_execution_retains_rejections_and_constant_features():
    records = [
        {"candidate_id": "variable", "natural_language_rule_text": "molecular weight"},
        {"candidate_id": "constant", "natural_language_rule_text": "amide"},
        {
            "candidate_id": "unsupported",
            "natural_language_rule_text": "unmapped fragment",
        },
        {"candidate_id": "extract-failed", "extraction_success": False},
    ]
    rows, summary = closure.analyze_candidates(
        closure.adapt_candidates(records, input_mode="candidates"), ["C", "CC", "CCC"]
    )
    assert [row["source_candidate_id"] for row in rows] == [
        "variable",
        "constant",
        "unsupported",
        "extract-failed",
    ]
    assert summary["input_records"] == 4
    assert summary["extracted_candidate_records"] == 3
    assert summary["extraction_failure_records"] == 1
    assert summary["projection_accepted"] == summary["compiler_accepted"] == 2
    assert summary["nonconstant_candidates"] == 1
    assert rows[1]["rejection_reason"] == "constant_or_degenerate_on_supplied_molecules"
    assert rows[2]["compiler_attempted"] == "false"
    assert rows[3]["rejection_reason"] == "candidate_extraction_failed"


def test_stage1_whole_records_are_not_silently_split(tmp_path):
    path = tmp_path / "stage1.json"
    path.write_text(
        json.dumps(
            {
                "dataset": "bace",
                "run_tag": "saved",
                "stage1_drafts": [
                    {
                        "draft_id": 0,
                        "draft_model": "qwen",
                        "draft_text": "molecular weight\namide",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    records = closure.read_records(path, input_mode="stage1")
    candidates = closure.adapt_candidates(records, input_mode="stage1")
    assert len(candidates) == 1
    assert candidates[0]["source_candidate_id"] == 0
    assert candidates[0]["dataset"] == "bace"
    assert candidates[0]["model_id"] == "qwen"
    assert (
        closure.project_candidates(candidates)[0]["projected_feature_name"] == "MolWt"
    )


def write_inputs(tmp_path):
    claims = tmp_path / "claims.jsonl"
    claims.write_text(
        json.dumps(
            {
                "raw_candidate_id": "claim-1",
                "natural_language_rule_text": "molecular weight",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    molecules = tmp_path / "molecules.csv"
    molecules.write_text(
        "mol,held_out_label\nC,0\nCC,100\nCCC,-999\n", encoding="utf-8"
    )
    return claims, molecules


def test_label_values_do_not_change_projection_or_closure(tmp_path):
    claims, molecules = write_inputs(tmp_path)
    first = closure.run_closure(
        claims, molecules, tmp_path / "first", smiles_column="mol"
    )
    molecules.write_text(
        "mol,held_out_label\nC,99999\nCC,wrong\nCCC,missing\n", encoding="utf-8"
    )
    second = closure.run_closure(
        claims, molecules, tmp_path / "second", smiles_column="mol"
    )
    assert {key: value for key, value in first.items() if key != "provenance"} == {
        key: value for key, value in second.items() if key != "provenance"
    }
    assert (
        first["provenance"]["ordered_smiles_sha256"]
        == second["provenance"]["ordered_smiles_sha256"]
    )
    assert (
        first["provenance"]["molecule_sha256"]
        != second["provenance"]["molecule_sha256"]
    )
    assert first["provenance"]["label_columns_used"] == []


def test_model_supplied_code_is_text_only(tmp_path):
    marker = tmp_path / "must_not_exist"
    malicious = f"__import__('pathlib').Path({str(marker)!r}).touch(); amide"
    candidates = closure.adapt_candidates(
        [{"natural_language_rule_text": malicious}], input_mode="candidates"
    )
    rows, _ = closure.analyze_candidates(candidates, ["C", "CC(=O)N"])
    assert rows[0]["projected_feature_name"] == "amide"
    assert rows[0]["execution_success"] == "true"
    assert not marker.exists()


def test_cli_outputs_accounting_and_refuses_overwrite(tmp_path):
    claims, molecules = write_inputs(tmp_path)
    output = tmp_path / "closure"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_raw_closure.py"),
        "--candidates",
        str(claims),
        "--molecules",
        str(molecules),
        "--smiles-column",
        "mol",
        "--output-dir",
        str(output),
    ]
    environment = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
    assert result.returncode == 0, result.stdout + result.stderr
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["nonconstant_candidates"] == 1
    before = (output / "candidates.jsonl").read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", env=environment)
    assert repeated.returncode == 1
    assert (output / "candidates.jsonl").read_bytes() == before


def test_malformed_jsonl_is_not_silently_dropped(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(
        '{"natural_language_rule_text": "amide"}\nnot json\n', encoding="utf-8"
    )
    with pytest.raises(ValueError):
        closure.read_records(path)
