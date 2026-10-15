"""Compiled-feature evaluation for contribution experiments."""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_compiled_inputs(
    project_root, dataset, compiled_rules, manifest,
    load_split_xy_fn, build_feature_matrix_fn, sanitize_feature_matrices_fn,
):
    with (project_root / "configs/model_registry.json").open("r", encoding="utf-8") as handle:
        hparams = json.load(handle)
    params_base = dict(hparams[dataset])
    split_data = {}
    for split in ("train", "valid", "test"):
        manifest_row = manifest[(dataset, split)]
        split_path = Path(manifest_row["split_file_path"])
        if not split_path.is_absolute():
            split_path = project_root / split_path
        split_data[split] = load_split_xy_fn(
            split_path, manifest_row["smiles_column"], manifest_row["target_column"],
        )
    matrices = [
        build_feature_matrix_fn(compiled_rules, split_data[split][0])
        for split in ("train", "valid", "test")
    ]
    feature_names = [rule.name for rule in compiled_rules]
    x_train, x_valid, x_test, kept_features = sanitize_feature_matrices_fn(*matrices, feature_names)
    return params_base, split_data, (x_train, x_valid, x_test), kept_features


def _evaluate_compiled_inputs(
    dataset, params_base, rf_seeds, split_data, matrices, numpy_module,
    evaluate_external_fn, evaluate_in_process_fn, variants=None,
):
    task_type = "classification" if dataset in {"bbbp", "bace", "clintox", "hiv", "tox21", "sider"} else "regression"
    try:
        import sklearn  # noqa: F401
    except ModuleNotFoundError:
        evaluate_fn = evaluate_external_fn
        sklearn_mode = "external_sklearn_worker"
    else:
        evaluate_fn = evaluate_in_process_fn
        matrices = tuple(numpy_module.asarray(matrix) for matrix in matrices)
        sklearn_mode = "in_process_sklearn"
    inputs = (
        matrices[0], split_data["train"][1],
        matrices[1], split_data["valid"][1],
        matrices[2], split_data["test"][1],
    )
    if variants is not None:
        inputs += (variants,)
    return evaluate_fn(task_type, params_base, rf_seeds, *inputs), sklearn_mode


def _summarize_scores(score_data, mean_std_fn):
    valid_mean, valid_std = mean_std_fn(score_data["valid_scores"])
    test_mean, test_std = mean_std_fn(score_data["test_scores"])
    return {
        "valid_scores": score_data["valid_scores"],
        "test_scores": score_data["test_scores"],
        "valid_mean": valid_mean,
        "valid_std": valid_std,
        "test_mean": test_mean,
        "test_std": test_std,
    }


def evaluate_compiled_loro_variants(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
    *,
    load_split_xy_fn,
    sanitize_feature_matrices_fn,
    evaluate_variants_external_fn,
    evaluate_variants_in_process_fn,
    mean_std_fn,
):
    from morace.rule_compiler import build_feature_matrix
    import numpy as np

    params_base, split_data, matrices, kept_features = _load_compiled_inputs(
        project_root, dataset, compiled_rules, manifest,
        load_split_xy_fn, build_feature_matrix, sanitize_feature_matrices_fn,
    )
    kept_index_by_name = {name: idx for idx, name in enumerate(kept_features)}
    all_indices = list(range(len(kept_features)))
    if not all_indices:
        raise RuntimeError("No kept features available for base model.")

    variants: List[Dict[str, object]] = [{"name": "base", "keep_indices": all_indices}]
    variant_notes: Dict[str, str] = {"base": ""}
    for idx, compiled in enumerate(compiled_rules):
        variant_name = f"drop_{idx}"
        if compiled.name in kept_index_by_name:
            drop_idx = kept_index_by_name[compiled.name]
            keep_indices = [item for item in all_indices if item != drop_idx]
            note = "exact_loro_feature_removed_from_matrix"
        else:
            keep_indices = all_indices
            note = "compiled_rule_was_filtered_by_sanitizer; dropping_it_does_not_change_matrix"
        if not keep_indices:
            raise RuntimeError(f"Dropping {compiled.name} leaves no kept matrix features.")
        variants.append({"name": variant_name, "keep_indices": keep_indices})
        variant_notes[variant_name] = note

    scores, sklearn_mode = _evaluate_compiled_inputs(
        dataset, params_base, rf_seeds, split_data, matrices, np,
        evaluate_variants_external_fn, evaluate_variants_in_process_fn, variants,
    )
    summarized = {
        name: {**_summarize_scores(score_data, mean_std_fn), "note": variant_notes.get(name, "")}
        for name, score_data in scores.items()
    }
    return summarized, sklearn_mode


def evaluate_compiled_family_variants(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
    *,
    load_split_xy_fn,
    sanitize_feature_matrices_fn,
    evaluate_variants_external_fn,
    evaluate_variants_in_process_fn,
    mean_std_fn,
):
    from morace.rule_compiler import build_feature_matrix
    import numpy as np

    params_base, split_data, matrices, kept_features = _load_compiled_inputs(
        project_root, dataset, compiled_rules, manifest,
        load_split_xy_fn, build_feature_matrix, sanitize_feature_matrices_fn,
    )
    if not kept_features:
        raise RuntimeError("No kept features available for base model.")

    family_by_name = {compiled.name: compiled.rule.get("feature_family", "") for compiled in compiled_rules}
    all_indices = list(range(len(kept_features)))
    variants: List[Dict[str, object]] = [{"name": "base", "keep_indices": all_indices}]
    variant_notes: Dict[str, str] = {"base": ""}
    skipped_variants: Dict[str, str] = {}

    for family in ("descriptor", "count", "pattern_presence"):
        if not any(compiled.rule.get("feature_family") == family for compiled in compiled_rules):
            skipped_variants[family] = "family_not_present"
            continue
        keep_indices = [idx for idx, name in enumerate(kept_features) if family_by_name.get(name) != family]
        variant_name = f"drop_family_{family}"
        if not keep_indices:
            skipped_variants[family] = "removing_family_leaves_no_kept_matrix_features"
            continue
        variants.append({"name": variant_name, "keep_indices": keep_indices})
        variant_notes[variant_name] = "family_level_feature_columns_removed_from_matrix"

    scores, sklearn_mode = _evaluate_compiled_inputs(
        dataset, params_base, rf_seeds, split_data, matrices, np,
        evaluate_variants_external_fn, evaluate_variants_in_process_fn, variants,
    )
    summarized = {
        name: {**_summarize_scores(score_data, mean_std_fn), "note": variant_notes.get(name, "")}
        for name, score_data in scores.items()
    }
    return summarized, skipped_variants, sklearn_mode


def evaluate_compiled_feature_importance(
    project_root: Path,
    dataset: str,
    compiled_rules,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    rf_seeds: List[int],
    *,
    load_split_xy_fn,
    sanitize_feature_matrices_fn,
    evaluate_feature_importance_external_fn,
    evaluate_feature_importance_in_process_fn,
    mean_std_fn,
):
    from morace.rule_compiler import build_feature_matrix
    import numpy as np

    params_base, split_data, matrices, kept_features = _load_compiled_inputs(
        project_root, dataset, compiled_rules, manifest,
        load_split_xy_fn, build_feature_matrix, sanitize_feature_matrices_fn,
    )
    if not kept_features:
        raise RuntimeError("No kept features available for feature importance.")

    eval_data, sklearn_mode = _evaluate_compiled_inputs(
        dataset, params_base, rf_seeds, split_data, matrices, np,
        evaluate_feature_importance_external_fn, evaluate_feature_importance_in_process_fn,
    )
    return {
        **_summarize_scores(eval_data, mean_std_fn),
        "importances": eval_data["importances"],
        "kept_features": kept_features,
        "sklearn_mode": sklearn_mode,
    }



