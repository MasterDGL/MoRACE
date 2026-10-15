import copy
import json
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def source_bucket(rule: Dict) -> str:
    support_sources = {str(item) for item in (rule.get("support_sources") or []) if item}
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("source", "") or rule.get("primary_source", "") or "")


def root_feature_key(rule: Dict) -> str:
    return f"{str(rule.get('feature_family', ''))}::{str(rule.get('feature_name', ''))}"


def canonical_key(rule: Dict) -> str:
    return str(rule.get("canonical_key", "")).strip()


def quality_sort_key(rule: Dict) -> Tuple[int, int, float, float, str]:
    compile_valid = rule.get("compile_valid")
    if compile_valid is None:
        compile_rule = _compile_rule()
        compile_valid = compile_rule(rule) is not None
    return (
        0 if bool(compile_valid) else 1,
        -int(rule.get("support_count", 0) or 0),
        float(rule.get("p_value", 1.0) or 1.0),
        -abs(float(rule.get("effect_size", 0.0) or 0.0)),
        canonical_key(rule),
    )


def candidate_addition_sort_key(rule: Dict, existing_roots: set[str]) -> Tuple[int, int, int, float, float, str]:
    bucket = source_bucket(rule)
    bucket_priority = {"data": 0, "multi": 1, "literature": 2}.get(bucket, 3)
    has_new_root = 0 if root_feature_key(rule) not in existing_roots else 1
    return (
        has_new_root,
        bucket_priority,
        -int(rule.get("support_count", 0) or 0),
        float(rule.get("p_value", 1.0) or 1.0),
        -abs(float(rule.get("effect_size", 0.0) or 0.0)),
        canonical_key(rule),
    )


def load_rule_artifact(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def best_rule(rules: Iterable[Dict]) -> Optional[Dict]:
    items = [rule for rule in rules if rule]
    if not items:
        return None
    return copy.deepcopy(sorted(items, key=quality_sort_key)[0])


def compile_safe_rules(rules: Iterable[Dict]) -> List[Dict]:
    kept = []
    for rule in rules:
        compile_valid = rule.get("compile_valid")
        if compile_valid is None:
            compile_rule = _compile_rule()
            compile_valid = compile_rule(rule) is not None
        if compile_valid:
            kept.append(copy.deepcopy(rule))
    return kept


def winner_anchor_rules(all_artifact: Dict) -> List[Dict]:
    return [copy.deepcopy(rule) for rule in all_artifact.get("rules", [])]


def gated_candidate_pool(all_artifact: Dict) -> List[Dict]:
    selection_trace = ((all_artifact.get("branch_metrics") or {}).get("combined") or {}).get("selection_trace") or {}
    return [copy.deepcopy(rule) for rule in selection_trace.get("gated_rules", [])]


def canonical_rule_map(rules: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {}
    for rule in rules:
        key = canonical_key(rule)
        if not key:
            continue
        grouped.setdefault(key, []).append(copy.deepcopy(rule))
    return grouped


def _drop_rule_once(rules: List[Dict], target_rule: Dict) -> List[Dict]:
    target_key = canonical_key(target_rule)
    dropped = False
    updated: List[Dict] = []
    for rule in rules:
        if not dropped and canonical_key(rule) == target_key:
            dropped = True
            continue
        updated.append(copy.deepcopy(rule))
    return updated


def build_ablation_a(anchor_rules: List[Dict], candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict]:
    variants = [copy.deepcopy(rule) for rule in anchor_rules]
    pool_by_key = canonical_rule_map(candidate_pool)
    replaced_pairs = []
    skipped_pairs = []
    for rule in list(variants):
        if not bool(rule.get("consensus_promoted", False)):
            continue
        lineage = rule.get("promotion_lineage") or {}
        parent_candidates = []
        for field in ("literature_canonical_key", "data_canonical_key", "primary_canonical_key", "secondary_canonical_key"):
            parent_candidates.extend(pool_by_key.get(str(lineage.get(field, "")).strip(), []))
        replacement = best_rule(parent_candidates)
        if replacement is None:
            skipped_pairs.append(
                {
                    "dropped_canonical_key": canonical_key(rule),
                    "replacement_status": "skipped",
                    "reason": "no_parent_candidate",
                }
            )
            continue
        variants = _drop_rule_once(variants, rule)
        variants.append(replacement)
        replaced_pairs.append(
            {
                "dropped_canonical_key": canonical_key(rule),
                "replacement_canonical_key": canonical_key(replacement),
            }
        )
    return variants, {"replacement_pairs": replaced_pairs, "skipped_pairs": skipped_pairs}


def build_ablation_b(anchor_rules: List[Dict], candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict]:
    variants = [copy.deepcopy(rule) for rule in anchor_rules]
    literature_only_rules = [rule for rule in variants if source_bucket(rule) == "literature"]
    if len(literature_only_rules) != 1:
        return variants, {"replacement_status": "skipped", "reason": "literature_only_rule_count_not_1"}
    dropped_rule = literature_only_rules[0]
    survivors = _drop_rule_once(variants, dropped_rule)
    existing_roots = {root_feature_key(rule) for rule in survivors}
    existing_keys = {canonical_key(rule) for rule in survivors}
    candidates = []
    for rule in compile_safe_rules(candidate_pool):
        if source_bucket(rule) != "data":
            continue
        if canonical_key(rule) in existing_keys:
            continue
        if root_feature_key(rule) in existing_roots:
            continue
        candidates.append(rule)
    ranked = sorted(candidates, key=lambda item: candidate_addition_sort_key(item, existing_roots))
    if not ranked:
        return variants, {"replacement_status": "skipped", "reason": "no_new_root_data_candidate"}
    replacement = copy.deepcopy(ranked[0])
    survivors.append(replacement)
    return survivors, {
        "replacement_status": "ok",
        "dropped_canonical_key": canonical_key(dropped_rule),
        "replacement_canonical_key": canonical_key(replacement),
    }


def build_ablation_c(anchor_rules: List[Dict], candidate_pool: List[Dict]) -> Tuple[List[Dict], Dict]:
    variants = [copy.deepcopy(rule) for rule in anchor_rules]
    existing_roots = {root_feature_key(rule) for rule in variants}
    existing_keys = {canonical_key(rule) for rule in variants}
    candidates = []
    for rule in compile_safe_rules(candidate_pool):
        if canonical_key(rule) in existing_keys:
            continue
        candidates.append(rule)
    if not candidates:
        return variants, {"addition_status": "skipped", "reason": "no_compile_safe_candidate"}
    ranked = sorted(candidates, key=lambda item: candidate_addition_sort_key(item, existing_roots))
    addition = copy.deepcopy(ranked[0])
    variants.append(addition)
    return variants, {
        "addition_status": "ok",
        "added_canonical_key": canonical_key(addition),
        "added_root_feature_key": root_feature_key(addition),
    }


def _surviving_rules(compiled_rules, kept_feature_names: List[str]) -> List[Dict]:
    kept = set(kept_feature_names)
    surviving = []
    for compiled in compiled_rules:
        if compiled.name in kept:
            surviving.append(copy.deepcopy(compiled.rule))
    return surviving


def _promoted_multi_metrics(rules: List[Dict]) -> Dict:
    promoted_rules = [rule for rule in rules if bool(rule.get("consensus_promoted", False))]
    promoted_root_keys = {
        str(rule.get("promoted_root_feature_key", "") or root_feature_key(rule) or "").strip()
        for rule in promoted_rules
        if str(rule.get("promoted_root_feature_key", "") or root_feature_key(rule) or "").strip()
    }
    surviving_multi_rules = [rule for rule in rules if source_bucket(rule) == "multi"]
    return {
        "surviving_multi_rule_count": len(surviving_multi_rules),
        "promoted_multi_surviving_rule_count": len(promoted_rules),
        "promoted_multi_surviving_root_count": len(promoted_root_keys),
    }


def evaluate_rule_variant(variant_name: str, rules: List[Dict], dataset: str, subtask: str, *, config=None) -> Dict:
    (
        compute_rule_diversity_metrics,
        evaluate_classification,
        evaluate_qm9,
        evaluate_regression_rf,
        prepare_rules_for_compilation_with_audit,
        sanitize_feature_matrices,
        load_split_xy,
        build_feature_matrix,
        compile_rules,
    ) = _runtime_eval_bindings(config)
    smiles_train, y_train, spec = load_split_xy(dataset, subtask=subtask, split="train")
    smiles_valid, y_valid, _ = load_split_xy(dataset, subtask=subtask, split="valid")
    smiles_test, y_test, _ = load_split_xy(dataset, subtask=subtask, split="test")

    prepared_rules, prepare_audit = prepare_rules_for_compilation_with_audit(rules, dataset, subtask)
    compiled_rules, invalid_rules = compile_rules(prepared_rules)
    X_train = build_feature_matrix(compiled_rules, smiles_train)
    X_valid = build_feature_matrix(compiled_rules, smiles_valid)
    X_test = build_feature_matrix(compiled_rules, smiles_test)

    sanitized_train, sanitized_valid, sanitized_test, kept_feature_names, sanitize_metrics = sanitize_feature_matrices(
        X_train,
        X_valid,
        X_test,
        [compiled.name for compiled in compiled_rules],
    )

    if spec.task_type == "classification":
        valid_scores, test_scores = evaluate_classification(
            dataset,
            subtask,
            sanitized_train,
            y_train,
            sanitized_valid,
            y_valid,
            sanitized_test,
            y_test,
        )
    elif spec.dataset == "qm9":
        valid_scores, test_scores = evaluate_qm9(
            sanitized_train,
            y_train,
            sanitized_valid,
            y_valid,
            sanitized_test,
            y_test,
        )
    else:
        valid_scores, test_scores = evaluate_regression_rf(
            dataset,
            sanitized_train,
            y_train,
            sanitized_valid,
            y_valid,
            sanitized_test,
            y_test,
        )

    surviving_rules = _surviving_rules(compiled_rules, kept_feature_names)
    diversity_metrics = compute_rule_diversity_metrics(surviving_rules)
    promoted_metrics = _promoted_multi_metrics(surviving_rules)
    source_breakdown = diversity_metrics.get("source_breakdown", {})
    return {
        "variant_name": variant_name,
        "rule_count": len(rules),
        "valid_scores": [float(item) for item in valid_scores],
        "test_scores": [float(item) for item in test_scores],
        "valid_mean": float(np.mean(valid_scores)),
        "test_mean": float(np.mean(test_scores)),
        "prepared_rule_count": len(prepared_rules),
        "compiled_rule_count": len(compiled_rules),
        "invalid_rule_count": len(invalid_rules),
        "root_feature_count": int(diversity_metrics.get("root_feature_count", 0) or 0),
        "effective_literature_coverage": int(diversity_metrics.get("effective_literature_coverage", 0) or 0),
        "effective_data_coverage": int(diversity_metrics.get("effective_data_coverage", 0) or 0),
        "selected_multi_count": sum(1 for rule in rules if source_bucket(rule) == "multi"),
        "surviving_multi_rule_count": int(promoted_metrics.get("surviving_multi_rule_count", 0) or 0),
        "promoted_multi_surviving_rule_count": int(promoted_metrics.get("promoted_multi_surviving_rule_count", 0) or 0),
        "feature_names": [str(rule.get("feature_name", "")) for rule in surviving_rules],
        "source_breakdown": {
            "literature": int(source_breakdown.get("literature", 0) or 0),
            "data": int(source_breakdown.get("data", 0) or 0),
            "multi": int(source_breakdown.get("multi", 0) or 0),
        },
        "kept_feature_names": kept_feature_names,
        "rules": [copy.deepcopy(rule) for rule in rules],
        "surviving_rules": surviving_rules,
        "prepare_audit": prepare_audit,
        "sanitize_metrics": sanitize_metrics,
    }


def _compile_rule():
    from morace.rule_compiler import compile_rule

    return compile_rule


def _runtime_eval_bindings(config=None):
    from types import SimpleNamespace
    from morace.evaluation_pipeline import (
        EvaluationConfig,
        EvaluationPipeline,
        compute_rule_diversity_metrics,
        evaluate_classification,
        evaluate_qm9,
        evaluate_regression_rf,
        sanitize_feature_matrices,
    )
    from morace.rule_compiler import build_feature_matrix, compile_rules

    # Legacy diagnostic callers inherit the environment once. Python callers can
    # pass a frozen config to keep repeated variants on the same input/protocol.
    if config is None:
        config = EvaluationConfig.from_namespace(SimpleNamespace(), environment=os.environ)
    evaluator = EvaluationPipeline(config)
    return (
        compute_rule_diversity_metrics,
        evaluate_classification,
        evaluate_qm9,
        evaluate_regression_rf,
        evaluator.prepare_rules_for_compilation_with_audit,
        sanitize_feature_matrices,
        evaluator.load_split_xy,
        build_feature_matrix,
        compile_rules,
    )


def output_paths(all_rules_path: str, suffix: str = "_posthoc_ablation") -> Tuple[str, str]:
    folder = os.path.dirname(all_rules_path)
    stem = os.path.splitext(os.path.basename(all_rules_path))[0]
    return (
        os.path.join(folder, f"{stem}{suffix}.json"),
        os.path.join(folder, f"{stem}{suffix}.csv"),
    )
