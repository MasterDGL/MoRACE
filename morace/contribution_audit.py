"""Input validation and dry-run auditing for contribution analysis."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from morace.contribution_data import (
    candidate_paths,
    canonical_key,
    drop_rule_once,
    family_variants,
    load_smiles_sample,
    load_split_manifest,
    read_csv,
    split_paths,
)
from morace.contribution_metrics import truth


def summarize_check_inputs(project_root: Path, inventory_rows: List[Dict[str, str]], manifest_path: Path) -> Tuple[bool, List[str]]:
    issues: List[str] = []
    if not inventory_rows:
        issues.append("empty contribution inventory")
    manifest_found = manifest_path.is_file()
    if not manifest_found:
        issues.append(f"missing split manifest: {manifest_path}")
        manifest: Dict[Tuple[str, str], Dict[str, str]] = {}
    else:
        manifest = load_split_manifest(manifest_path, read_csv_fn=read_csv)

    for row in inventory_rows:
        paths = candidate_paths(project_root, row)
        for label, path in paths.items():
            if not path.is_file():
                issues.append(f"{row.get('pilot_id')}: missing {label}: {path}")
        dataset = row.get("dataset", "")
        for split, path in split_paths(project_root, manifest, dataset).items():
            manifest_row = manifest.get((dataset, split), {})
            if not manifest_row.get("split_file_path") or not path.is_file():
                issues.append(f"{row.get('pilot_id')}: missing {split} split: {path}")
    return not issues, issues


def dry_run_candidate(
    project_root: Path,
    row: Dict[str, str],
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    matrix_sample_size: int,
) -> Dict[str, object]:
    issues: List[str] = []
    notes: List[str] = []
    paths = candidate_paths(project_root, row)
    splits = split_paths(project_root, manifest, row.get("dataset", ""))

    rules_file_found = paths["rules_file"].is_file()
    outcome_file_found = paths["outcome_file"].is_file()
    split_found = {split: path.is_file() for split, path in splits.items()}

    can_build_base = False
    can_loro = False
    can_family = False
    can_importance = False

    if not rules_file_found:
        issues.append("rules_file_missing")
    if not outcome_file_found:
        issues.append("outcome_file_missing")
    for split, found in split_found.items():
        if not found:
            issues.append(f"split_{split}_missing")

    if not issues:
        try:
            from morace.rule_compiler import build_feature_matrix, compile_rules
        except Exception as exc:  # pragma: no cover - dependency/environment guard
            issues.append(f"rule_compiler_import_failed: {exc.__class__.__name__}: {exc}")
            compile_rules = None
            build_feature_matrix = None

        if not issues and compile_rules is not None and build_feature_matrix is not None:
            try:
                artifact = json.loads(paths["rules_file"].read_text(encoding="utf-8"))
                rules = artifact.get("rules") or []
                compiled_rules, invalid_rules = compile_rules(rules)
                expected_selected = int(row.get("selected_rule_count") or 0)
                expected_compiled = int(row.get("compiled_rule_count") or 0)

                if len(rules) != expected_selected:
                    notes.append(f"selected_rule_count_metadata_mismatch: rules_array={len(rules)} inventory={expected_selected}")
                if len(compiled_rules) != expected_compiled:
                    issues.append(f"compiled_rule_count_mismatch: compiled={len(compiled_rules)} inventory={expected_compiled}")
                if invalid_rules:
                    issues.append(f"unexpected_invalid_rules={len(invalid_rules)}")

                samples = []
                for split in ("train", "valid", "test"):
                    manifest_row = manifest.get((row.get("dataset", ""), split), {})
                    smiles_column = manifest_row.get("smiles_column", "")
                    if not smiles_column:
                        issues.append(f"missing_smiles_column_in_manifest_{split}")
                        continue
                    samples.extend(load_smiles_sample(splits[split], smiles_column, matrix_sample_size))
                if samples:
                    matrix = build_feature_matrix(compiled_rules, samples)
                    expected_cols = len(compiled_rules)
                    can_build_base = bool(matrix) and all(len(vec) == expected_cols for vec in matrix)
                    if not can_build_base:
                        issues.append("base_rule_matrix_shape_mismatch")
                else:
                    issues.append("no_smiles_samples_loaded")

                if len(rules) > 1:
                    loro_ok = True
                    seen_keys = [canonical_key(rule) for rule in rules]
                    for key in seen_keys:
                        variant_rules = drop_rule_once(
                            rules,
                            key,
                            canonical_key_fn=canonical_key,
                        )
                        variant_compiled, _ = compile_rules(variant_rules)
                        if len(variant_rules) != len(rules) - 1 or len(variant_compiled) < 1:
                            loro_ok = False
                            break
                    can_loro = loro_ok
                    if not can_loro:
                        issues.append("loro_variant_preparation_failed")
                else:
                    issues.append("not_enough_rules_for_loro")

                family_ok = False
                for variant_rules in family_variants(rules).values():
                    variant_compiled, _ = compile_rules(variant_rules)
                    if variant_rules and variant_compiled:
                        family_ok = True
                can_family = family_ok
                if not can_family:
                    issues.append("family_ablation_variant_preparation_failed")

                can_importance = can_build_base and len(compiled_rules) > 0
            except Exception as exc:
                issues.append(f"dry_run_exception: {exc.__class__.__name__}: {exc}")

    status = "passed" if not issues and can_build_base and can_loro and can_family and can_importance else "failed"
    return {
        "pilot_id": row.get("pilot_id", ""),
        "route_or_model": row.get("route_or_model", ""),
        "dataset": row.get("dataset", ""),
        "run_tag": row.get("run_tag", ""),
        "seed": row.get("seed", ""),
        "metric": row.get("metric", ""),
        "metric_direction": row.get("metric_direction", ""),
        "selected_rule_count": row.get("selected_rule_count", ""),
        "compiled_rule_count": row.get("compiled_rule_count", ""),
        "rules_file_found": truth(rules_file_found),
        "outcome_file_found": truth(outcome_file_found),
        "split_train_found": truth(split_found.get("train")),
        "split_valid_found": truth(split_found.get("valid")),
        "split_test_found": truth(split_found.get("test")),
        "can_build_base_rule_matrix": truth(can_build_base),
        "can_prepare_loro": truth(can_loro),
        "can_prepare_family_ablation": truth(can_family),
        "can_prepare_feature_importance": truth(can_importance),
        "requires_gpu": "no",
        "requires_llm": "no",
        "requires_new_rules": "no",
        "dry_run_status": status,
        "issue": "; ".join(issues + notes),
        "recommended_next_action": "ready_for_contribution_analysis" if status == "passed" else "fix_dry_run_issue_before_analysis",
    }


