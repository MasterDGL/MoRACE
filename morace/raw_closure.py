"""Frozen operational projection of raw molecular claims and CPU feature closure.

The matching table and acceptance/rejection logic are migrated from P7c3.
This is a separate deterministic diagnostic; it is not the main LLM projector.
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any
from morace import rule_compiler

PROJECTION_VERSION = "p7c3_operational_projection_v1"


SUPPORTED_TASK_LABEL_TERMS = {
    "toxicity",
    "toxic",
    "clinical toxicity",
    "hiv activity",
    "hiv-active",
    "bace inhibition",
    "bace-1 inhibition",
    "solubility",
    "hydration free energy",
    "blood-brain barrier",
    "permeability",
    "lipophilicity",
    "logd",
    "target label",
    "label",
}


PROJECTION_RULES = [
    (
        "descriptor_molwt",
        "molecular weight|molwt|molecular mass|heavy atom count|heavy atoms",
        "descriptor",
        "MolWt",
        "numeric_value",
        "high",
    ),
    (
        "descriptor_tpsa",
        "tpsa|topological polar surface area|polar surface area",
        "descriptor",
        "TPSA",
        "numeric_value",
        "high",
    ),
    (
        "descriptor_mollogp",
        "\\blogp\\b|clogp|mol logp|hydrophobicity|hydrophobic|lipophilic",
        "descriptor",
        "MolLogP",
        "numeric_value",
        "medium",
    ),
    (
        "count_hba",
        "hydrogen bond acceptor|\\bhba\\b|acceptor count|h-bond acceptor",
        "count",
        "NumHBA",
        "count_value",
        "high",
    ),
    (
        "count_hbd",
        "hydrogen bond donor|\\bhbd\\b|donor count|h-bond donor",
        "count",
        "NumHBD",
        "count_value",
        "high",
    ),
    (
        "count_rotatable_bonds",
        "rotatable bond|rotatable bonds|flexibility|flexible chain",
        "count",
        "NumRotatableBonds",
        "count_value",
        "medium",
    ),
    (
        "count_aromatic_rings",
        "aromatic ring count|number of aromatic rings|aromatic rings|aromatic ring system|aromatic ring",
        "count",
        "NumAromaticRings",
        "count_value",
        "high",
    ),
    (
        "pattern_amide",
        "amide|amide bond|amide group",
        "pattern_presence",
        "amide",
        "binary_presence",
        "high",
    ),
    (
        "pattern_amine",
        "amine|amino|basic nitrogen|tertiary amine|basic amine",
        "pattern_presence",
        "amine",
        "binary_presence",
        "medium",
    ),
    (
        "pattern_halogen",
        "halogen|halogenated|fluorine|chlorine|bromine",
        "pattern_presence",
        "halogen",
        "binary_presence",
        "high",
    ),
    (
        "pattern_carboxyl",
        "carboxylic acid|carboxyl|carboxylate",
        "pattern_presence",
        "carboxyl",
        "binary_presence",
        "high",
    ),
    (
        "pattern_carbonyl",
        "carbonyl|ketone|aldehyde",
        "pattern_presence",
        "carbonyl",
        "binary_presence",
        "medium",
    ),
    (
        "pattern_ester",
        "ester|ester group",
        "pattern_presence",
        "ester",
        "binary_presence",
        "high",
    ),
    (
        "pattern_ether",
        "ether|ether linkage",
        "pattern_presence",
        "ether",
        "binary_presence",
        "high",
    ),
    (
        "pattern_alcohol",
        "hydroxyl|alcohol|oh group",
        "pattern_presence",
        "alcohol",
        "binary_presence",
        "high",
    ),
    (
        "pattern_aromatic_ring_presence",
        "aromatic ring|aromatic scaffold|aromatic system",
        "pattern_presence",
        "aromatic_ring",
        "binary_presence",
        "medium",
    ),
    (
        "pattern_fused_ring",
        "fused ring|fused bicyclic|fused heterocycle|benzofuran|indole",
        "pattern_presence",
        "fused_ring",
        "binary_presence",
        "medium",
    ),
]


def nonconstant(values: list[float]) -> bool:
    clean = [
        v for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))
    ]
    if len(clean) < 2:
        return False
    first = float(clean[0])
    return any((abs(float(v) - first) > 1e-12 for v in clean[1:]))


def direction_conflict(text: str, hint: str) -> bool:
    combined = f"{text} {hint}".lower()
    pos_terms = any(
        (
            term in combined
            for term in ["increase", "higher", "positive", "more", "enhance"]
        )
    )
    neg_terms = any(
        (
            term in combined
            for term in ["decrease", "lower", "negative", "less", "reduce"]
        )
    )
    return pos_terms and neg_terms


def only_task_semantics(text: str, matched_rule_id: str | None) -> bool:
    if matched_rule_id:
        return False
    lower = text.lower()
    return any((term in lower for term in SUPPORTED_TASK_LABEL_TERMS))


def relation(text: str) -> str:
    lower = text.lower()
    if any(
        (term in lower for term in ["negative", "lower", "decrease", "reduce", "less"])
    ):
        return "negative"
    if any(
        (
            term in lower
            for term in ["positive", "higher", "increase", "more", "enhance"]
        )
    ):
        return "positive"
    return "unspecified"


def project_candidates(candidates):
    rows = []
    for row in candidates:
        text = " ".join(
            [
                row.get("natural_language_rule_text", ""),
                row.get("feature_hint", ""),
                row.get("rationale_short", ""),
            ]
        ).strip()
        base = {
            "raw_candidate_id": row.get("raw_candidate_id", ""),
            "prompt_id": row.get("prompt_id", ""),
            "model_id": row.get("model_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "natural_language_rule_text": row.get("natural_language_rule_text", ""),
            "feature_hint": row.get("feature_hint", ""),
            "direction_hint": row.get("direction_hint", ""),
            "strict_schema_parse_attempted": "false",
            "strict_schema_valid": "false",
            "strict_schema_failure_reason": "strict_parser_unavailable_for_raw_drafts",
            "operational_projection_attempted": "true",
        }
        if row.get("extraction_success") != "true":
            rows.append(
                {
                    **base,
                    "operational_projection_accepts": "false",
                    "unsupported_reason": "candidate_extraction_failed",
                    "rejected_reason": "",
                    "direction_conflict": "false",
                    "compiler_support_status": "not_applicable",
                    "projection_notes": "No extracted candidate.",
                }
            )
            continue
        matched = None
        for rule_id, pattern, family, name, mode, confidence in PROJECTION_RULES:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                matched = (rule_id, match.group(0), family, name, mode, confidence)
                break
        if only_task_semantics(text, matched[0] if matched else None):
            rows.append(
                {
                    **base,
                    "operational_projection_accepts": "false",
                    "projection_rule_id": "",
                    "projection_confidence": "",
                    "projection_source_terms": "",
                    "projected_feature_family": "",
                    "projected_feature_name": "",
                    "projected_feature_mode": "",
                    "projected_relation_or_direction": "",
                    "unsupported_reason": "task_label_semantics_without_compiler_supported_molecular_feature",
                    "rejected_reason": "",
                    "direction_conflict": "false",
                    "compiler_support_status": "unsupported",
                    "normalized_rule_id": "",
                    "normalized_rule_json": "",
                    "projection_notes": "Task/property term was not mapped to molecular feature.",
                }
            )
            continue
        if not matched:
            rows.append(
                {
                    **base,
                    "operational_projection_accepts": "false",
                    "projection_rule_id": "",
                    "projection_confidence": "",
                    "projection_source_terms": "",
                    "projected_feature_family": "",
                    "projected_feature_name": "",
                    "projected_feature_mode": "",
                    "projected_relation_or_direction": "",
                    "unsupported_reason": "no_supported_projection_pattern",
                    "rejected_reason": "",
                    "direction_conflict": "false",
                    "compiler_support_status": "unsupported",
                    "normalized_rule_id": "",
                    "normalized_rule_json": "",
                    "projection_notes": "No compiler-supported molecular feature term found.",
                }
            )
            continue
        conflict = direction_conflict(
            row.get("natural_language_rule_text", ""), row.get("direction_hint", "")
        )
        if conflict:
            rows.append(
                {
                    **base,
                    "operational_projection_accepts": "false",
                    "projection_rule_id": matched[0],
                    "projection_confidence": matched[5],
                    "projection_source_terms": matched[1],
                    "projected_feature_family": matched[2],
                    "projected_feature_name": matched[3],
                    "projected_feature_mode": matched[4],
                    "projected_relation_or_direction": relation(text),
                    "unsupported_reason": "",
                    "rejected_reason": "direction_conflict",
                    "direction_conflict": "true",
                    "compiler_support_status": "supported",
                    "normalized_rule_id": "",
                    "normalized_rule_json": "",
                    "projection_notes": "Rejected due direction conflict.",
                }
            )
            continue
        rule = {
            "feature_family": matched[2],
            "feature_name": matched[3],
            "feature_mode": matched[4],
            "relation": relation(text),
            "source": "p7c3_operational_projection",
            "raw_candidate_id": row.get("raw_candidate_id", ""),
        }
        norm_id = f"{row.get('raw_candidate_id')}__{matched[0]}"
        rows.append(
            {
                **base,
                "operational_projection_accepts": "true",
                "projection_rule_id": matched[0],
                "projection_confidence": matched[5],
                "projection_source_terms": matched[1],
                "projected_feature_family": matched[2],
                "projected_feature_name": matched[3],
                "projected_feature_mode": matched[4],
                "projected_relation_or_direction": relation(text),
                "unsupported_reason": "",
                "rejected_reason": "",
                "direction_conflict": "false",
                "compiler_support_status": "supported",
                "normalized_rule_id": norm_id,
                "normalized_rule_json": json.dumps(rule, sort_keys=True),
                "projection_notes": "Operational projection; not strict schema validation.",
            }
        )
    return rows


def evaluate_projections(projections, smiles, *, skip_execution=False):
    rows = []
    for row in projections:
        base = {
            "prompt_id": row.get("prompt_id", ""),
            "model_id": row.get("model_id", ""),
            "dataset": row.get("dataset", ""),
            "seed": row.get("seed", ""),
            "raw_candidate_id": row.get("raw_candidate_id", ""),
            "normalized_rule_id": row.get("normalized_rule_id", ""),
            "operational_projection_accepts": row.get(
                "operational_projection_accepts", "false"
            ),
            "projection_rule_id": row.get("projection_rule_id", ""),
            "downstream_rf_executed": "false",
            "table1_metrics_recomputed": "false",
        }
        if (
            row.get("operational_projection_accepts") != "true"
            or row.get("compiler_support_status") != "supported"
        ):
            reason = (
                row.get("unsupported_reason")
                or row.get("rejected_reason")
                or "operational_projection_not_accepted"
            )
            rows.append(
                {
                    **base,
                    "compiler_attempted": "false",
                    "compiler_accepted": "false",
                    "compile_failure_reason": reason,
                    "compiled_feature_id": "",
                    "execution_attempted": "false",
                    "execution_success": "false",
                    "execution_error": reason,
                    "molecules_total": "",
                    "valid_value_count": "",
                    "nan_or_failed_count": "",
                    "nonconstant_feature": "",
                    "constant_or_degenerate_feature": "",
                    "execution_seconds": "0",
                    "evidence_label": "not_applicable",
                    "notes": "Not sent to compiler.",
                }
            )
            continue
        try:
            rule = json.loads(row["normalized_rule_json"])
            (compiled, _invalid) = rule_compiler.compile_rules([rule])
            if not compiled:
                raise RuntimeError("compile_rules returned no compiled feature")
            compiled_feature_id = getattr(
                compiled[0], "name", row.get("normalized_rule_id", "compiled_feature")
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "compiler_attempted": "true",
                    "compiler_accepted": "false",
                    "compile_failure_reason": f"{type(exc).__name__}: {exc}",
                    "compiled_feature_id": "",
                    "execution_attempted": "false",
                    "execution_success": "false",
                    "execution_error": "compile_failed",
                    "molecules_total": "",
                    "valid_value_count": "",
                    "nan_or_failed_count": "",
                    "nonconstant_feature": "false",
                    "constant_or_degenerate_feature": "false",
                    "execution_seconds": "0",
                    "evidence_label": "deterministic_reconstructed_failure",
                    "notes": "Compiler rejection; no RF.",
                }
            )
            continue
        if skip_execution:
            rows.append(
                {
                    **base,
                    "compiler_attempted": "true",
                    "compiler_accepted": "true",
                    "compile_failure_reason": "",
                    "compiled_feature_id": compiled_feature_id,
                    "execution_attempted": "false",
                    "execution_success": "false",
                    "execution_error": "skipped_preflight",
                    "molecules_total": "",
                    "valid_value_count": "",
                    "nan_or_failed_count": "",
                    "nonconstant_feature": "",
                    "constant_or_degenerate_feature": "",
                    "execution_seconds": "0",
                    "evidence_label": "pending_server_execution",
                    "notes": "RDKit execution skipped by flag.",
                }
            )
            continue
        try:
            t0 = time.time()
            matrix = rule_compiler.build_feature_matrix(compiled, smiles)
            values = [line[0] for line in matrix]
            valid = [
                float(v)
                for v in values
                if isinstance(v, (int, float)) and math.isfinite(float(v))
            ]
            is_nonconstant = nonconstant(valid)
            rows.append(
                {
                    **base,
                    "compiler_attempted": "true",
                    "compiler_accepted": "true",
                    "compile_failure_reason": "",
                    "compiled_feature_id": compiled_feature_id,
                    "execution_attempted": "true",
                    "execution_success": "true",
                    "execution_error": "",
                    "molecules_total": str(len(smiles)),
                    "valid_value_count": str(len(valid)),
                    "nan_or_failed_count": str(len(values) - len(valid)),
                    "nonconstant_feature": str(is_nonconstant).lower(),
                    "constant_or_degenerate_feature": str(not is_nonconstant).lower(),
                    "execution_seconds": f"{time.time() - t0:.6f}",
                    "evidence_label": "deterministic_reconstructed",
                    "notes": "Operational projection route; feature-level execution only; no downstream RF.",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    **base,
                    "compiler_attempted": "true",
                    "compiler_accepted": "true",
                    "compile_failure_reason": "",
                    "compiled_feature_id": compiled_feature_id,
                    "execution_attempted": "true",
                    "execution_success": "false",
                    "execution_error": f"{type(exc).__name__}: {exc}",
                    "molecules_total": "",
                    "valid_value_count": "",
                    "nan_or_failed_count": "",
                    "nonconstant_feature": "false",
                    "constant_or_degenerate_feature": "false",
                    "execution_seconds": "0",
                    "evidence_label": "deterministic_reconstructed_failure",
                    "notes": "Execution failed; no RF.",
                }
            )
    return rows


FROZEN_SOURCE_SHA256 = {
    "p7c3_common.py": "eb7cb616cefd78f697fc73759dd9a23c20bb70314553973843c90a6cdf7ee4a5",
    "p7c3_run_schema_normalization.py": "8bcb8d2536cf1e1ae23376cb48bb7baaa6851a58ecb31336667429ee1a7860b4",
    "p7c3_run_compile_and_execution.py": "e83542d2fbbbde6d9aead798f146ea7c367b32f392e0c11ca7a13fe3b0361180",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_records(path: Path, *, input_mode: str = "candidates") -> list[dict[str, Any]]:
    """Read explicit candidate units; malformed JSON is an input error, not a dropped row."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        if path.suffix.lower() == ".csv":
            records = list(csv.DictReader(handle))
        elif path.suffix.lower() in {".jsonl", ".ndjson"}:
            records = [json.loads(line) for line in handle if line.strip()]
        elif path.suffix.lower() == ".json":
            records = json.load(handle)
        else:
            raise ValueError("candidate file must be CSV, JSONL or a JSON list")
    if input_mode == "stage1" and isinstance(records, dict):
        parent = records
        records = parent.get("stage1_drafts")
        if isinstance(records, list):
            records = [
                (
                    {
                        "dataset": parent.get("dataset", ""),
                        "subtask": parent.get("subtask", ""),
                        "run_tag": parent.get("run_tag", ""),
                        **row,
                    }
                    if isinstance(row, dict)
                    else row
                )
                for row in records
            ]
    if not isinstance(records, list) or not all(
        isinstance(row, dict) for row in records
    ):
        raise ValueError(
            "candidate input must contain a list of JSON objects or CSV records"
        )
    return records


def adapt_candidates(
    records: list[dict[str, Any]], *, input_mode: str
) -> list[dict[str, Any]]:
    """Keep one record as one unit; do not invent a sentence/bullet extraction rule."""
    if input_mode not in {"candidates", "stage1"}:
        raise ValueError(f"unsupported input_mode: {input_mode}")
    candidates = []
    for index, record in enumerate(records, 1):
        candidate = dict(record)
        source_id = next(
            (
                record[key]
                for key in (
                    "raw_candidate_id",
                    "source_candidate_id",
                    "candidate_id",
                    "draft_id",
                )
                if key in record and record[key] not in {None, ""}
            ),
            f"record_{index:06d}",
        )
        candidate.setdefault("raw_candidate_id", str(source_id))
        candidate["source_candidate_id"] = source_id
        candidate["source_record_index"] = index
        candidate["input_mode"] = input_mode
        if input_mode == "stage1":
            candidate["natural_language_rule_text"] = record.get(
                "draft_text", record.get("text", "")
            )
            candidate.setdefault("model_id", record.get("draft_model", ""))
        invalid_fields = []
        for field in (
            "natural_language_rule_text",
            "feature_hint",
            "direction_hint",
            "rationale_short",
        ):
            value = candidate.get(field, "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                invalid_fields.append(field)
                value = ""
            candidate[field] = value
        extraction = candidate.get("extraction_success", True)
        if isinstance(extraction, bool):
            extraction = str(extraction).lower()
        if extraction not in {"true", "false"}:
            invalid_fields.append("extraction_success")
        if input_mode == "stage1" and not candidate["natural_language_rule_text"]:
            invalid_fields.append("draft_text")
        candidate["extraction_success"] = "false" if invalid_fields else extraction
        candidate["input_validation_error"] = (
            "invalid or missing text fields: " + ", ".join(invalid_fields)
            if invalid_fields
            else ""
        )
        candidates.append(candidate)
    return candidates


def read_molecules(path: Path, smiles_column: str) -> list[str]:
    # Only the requested SMILES column is used. No target values enter projection,
    # compilation, variability checks or candidate selection.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if smiles_column not in (reader.fieldnames or []):
            raise ValueError(f"SMILES column {smiles_column!r} is missing from {path}")
        smiles = [row.get(smiles_column) or "" for row in reader]
    if not smiles:
        raise ValueError("molecule CSV contains no rows")
    return smiles


def closure_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    extracted = sum(row["extraction_success"] == "true" for row in rows)
    accepted = sum(row["operational_projection_accepts"] == "true" for row in rows)
    compiled = sum(row["compiler_accepted"] == "true" for row in rows)
    executed = sum(row["execution_success"] == "true" for row in rows)
    variable = sum(row["nonconstant_feature"] == "true" for row in rows)
    return {
        "input_records": len(rows),
        "extracted_candidate_records": extracted,
        "extraction_failure_records": len(rows) - extracted,
        "projection_accepted": accepted,
        "compiler_accepted": compiled,
        "execution_success": executed,
        "nonconstant_candidates": variable,
        "distinct_compiled_features": len(
            {
                row["compiled_feature_id"]
                for row in rows
                if row["compiler_accepted"] == "true"
            }
        ),
        "rejection_counts": dict(
            sorted(
                Counter(
                    row["rejection_reason"] for row in rows if row["rejection_reason"]
                ).items()
            )
        ),
        "rates_over_extracted_candidates": {
            "projection_accepted": accepted / extracted if extracted else None,
            "compiler_accepted": compiled / extracted if extracted else None,
            "execution_success": executed / extracted if extracted else None,
            "nonconstant": variable / extracted if extracted else None,
        },
    }


def analyze_candidates(
    candidates: list[dict[str, Any]], smiles: list[str]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    projections = project_candidates(candidates)
    executions = evaluate_projections(projections, smiles)
    rows = []
    for candidate, projection, execution in zip(candidates, projections, executions):
        # Preserve source IDs and all frozen stage fields alongside a single
        # explicit reason for candidate-level accounting.
        reason = (
            projection.get("unsupported_reason")
            or projection.get("rejected_reason")
            or execution.get("compile_failure_reason")
            or execution.get("execution_error")
            or ""
        )
        if not reason and execution.get("nonconstant_feature") == "false":
            reason = "constant_or_degenerate_on_supplied_molecules"
        rows.append(
            {
                **candidate,
                **projection,
                **execution,
                "rejection_reason": reason,
                "projection_version": PROJECTION_VERSION,
            }
        )
    summary = closure_summary(rows)
    groups: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        key = tuple(
            str(row.get(field, "")) for field in ("model_id", "dataset", "seed")
        )
        groups.setdefault(key, []).append(row)
    summary["groups"] = [
        {
            "model_id": key[0],
            "dataset": key[1],
            "seed": key[2],
            **closure_summary(group),
        }
        for key, group in sorted(groups.items())
    ]
    return rows, summary


def run_closure(
    candidate_path: Path,
    molecule_path: Path,
    output_dir: Path,
    *,
    smiles_column: str = "smiles",
    input_mode: str = "candidates",
) -> dict[str, Any]:
    input_hashes = {
        "candidate": sha256_file(candidate_path),
        "molecules": sha256_file(molecule_path),
    }
    candidates = adapt_candidates(
        read_records(candidate_path, input_mode=input_mode), input_mode=input_mode
    )
    if not candidates:
        raise ValueError("candidate file contains no records")
    smiles = read_molecules(molecule_path, smiles_column)
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite closure output directory: {output_dir}"
        )
    rows, summary = analyze_candidates(candidates, smiles)
    if input_hashes != {
        "candidate": sha256_file(candidate_path),
        "molecules": sha256_file(molecule_path),
    }:
        raise ValueError("input file changed during closure analysis")
    mapping = {
        "rules": PROJECTION_RULES,
        "task_label_terms": sorted(SUPPORTED_TASK_LABEL_TERMS),
    }
    provenance = {
        "projection_version": PROJECTION_VERSION,
        "input_mode": input_mode,
        "candidate_unit": "one supplied record; no automatic sentence splitting",
        "candidate_path": str(candidate_path.resolve()),
        "candidate_sha256": input_hashes["candidate"],
        "molecule_path": str(molecule_path.resolve()),
        "molecule_sha256": input_hashes["molecules"],
        "smiles_column": smiles_column,
        "molecule_rows": len(smiles),
        "ordered_smiles_sha256": hashlib.sha256(
            json.dumps(smiles, ensure_ascii=False).encode("utf-8")
        ).hexdigest(),
        "mapping_sha256": hashlib.sha256(
            json.dumps(mapping, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "historical_source_sha256": FROZEN_SOURCE_SHA256,
        "implementation_sha256": sha256_file(Path(__file__)),
        "compiler_sha256": sha256_file(Path(rule_compiler.__file__)),
        "label_columns_used": [],
        "llm_calls": 0,
        "downstream_rf_executed": False,
        "smiles_policy": "all supplied rows retained; frozen compiler semantics; valid_value_count counts finite feature values",
    }
    summary.update({"schema_version": "raw_closure_v1", "provenance": provenance})
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "candidates.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output_dir / "candidates.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Frozen P7c3 raw-claim projection and CPU feature closure on supplied molecules."
    )
    parser.add_argument(
        "--candidates", type=Path, required=True, help="CSV, JSONL, or JSON list"
    )
    parser.add_argument(
        "--molecules",
        type=Path,
        required=True,
        help="fixed molecule CSV; labels are unused",
    )
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument(
        "--input-mode", choices=["candidates", "stage1"], default="candidates"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="new output directory"
    )
    args = parser.parse_args()
    try:
        summary = run_closure(
            args.candidates,
            args.molecules,
            args.output_dir,
            smiles_column=args.smiles_column,
            input_mode=args.input_mode,
        )
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"Raw closure failed: {exc}", file=sys.stderr)
        return 1
    print(
        f"Raw closure: {summary['extracted_candidate_records']}/{summary['input_records']} extracted records, "
        f"{summary['projection_accepted']} projected, {summary['compiler_accepted']} compiled, "
        f"{summary['nonconstant_candidates']} nonconstant on supplied molecules"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
