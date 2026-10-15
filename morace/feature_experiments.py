"""Paired, frozen-rule RF experiments; independent of the discovery pipeline.

The P7rf-e sanitizer and P13 controls are deliberately kept separate from
the main rule-only evaluation code: they describe a different experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from morace.artifact_inputs import resolve_run_rules
from morace.datasets import get_dataset_spec
from morace.rule_compiler import build_feature_matrix, compile_rules

ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
DATASETS = ("bace", "bbbp", "clintox", "hiv", "esol", "freesolv", "lipophilicity")
ARMS = ("base", "rules", "base_plus_rules", "base_plus_shuffled_rules", "base_plus_random_rules")
SCHEMA = "feature_experiments_v1"
P13_PHASE = "P13-shuffled-random-rule-feature-control"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: str) -> int:
    """Historical P13 seed derivation, including its phase/route/repeat labels."""
    return int(hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:8], 16)


def make_split_local_shuffled_controls(rule_matrices, control_seed):
    """P13: independent within-split permutation for every rule column."""
    out = {}
    for split_index, split in enumerate(SPLITS):
        matrix = np.asarray(rule_matrices[split], dtype=np.float32)
        rng = np.random.default_rng(control_seed + split_index * 1009)
        shuffled = np.empty_like(matrix)
        for col in range(matrix.shape[1]):
            shuffled[:, col] = matrix[rng.permutation(matrix.shape[0]), col]
        out[split] = shuffled.astype(np.float32, copy=False)
    return out


def make_gaussian_count_matched_controls(rule_matrices, control_seed):
    """P13: Gaussian columns with train-only moments and split-local RNGs."""
    train = np.asarray(rule_matrices["train"], dtype=np.float32)
    means = train.mean(axis=0)
    stds = train.std(axis=0)
    stds = np.where(stds > 1e-12, stds, 1.0).astype(np.float32)
    out = {}
    for split_index, split in enumerate(SPLITS):
        rng = np.random.default_rng(control_seed + split_index * 2003)
        out[split] = rng.normal(
            loc=means, scale=stds, size=(rule_matrices[split].shape[0], train.shape[1])
        ).astype(np.float32)
    return out


def append_control_columns(base_matrices, control_matrices):
    combined = {}
    for split in SPLITS:
        base = np.asarray(base_matrices[split], dtype=np.float32)
        control = np.asarray(control_matrices[split], dtype=np.float32)
        if base.shape[0] != control.shape[0]:
            raise ValueError(f"{split} row mismatch: base={base.shape}, control={control.shape}")
        if not np.isfinite(base).all() or not np.isfinite(control).all():
            raise ValueError(f"{split} contains NaN or infinite values")
        combined[split] = np.concatenate([base, control], axis=1).astype(np.float32, copy=False)
    return combined


def sanitize_rule_matrices(matrices, feature_names):
    """P7rf-e definition: deduplicate names, drop train constants, median fill.

    Preserve the historical nanmedian computation exactly. If infinities make
    its medians nonfinite, the caller rejects the run instead of fitting RF.
    """
    train = matrices["train"].astype(float)
    keep, seen, duplicates = [], set(), []
    dropped_constant = dropped_all_nan = 0
    for idx, name in enumerate(feature_names):
        if name in seen:
            duplicates.append(name)
            continue
        seen.add(name)
        finite = train[:, idx][np.isfinite(train[:, idx])]
        if finite.size == 0:
            dropped_all_nan += 1
        elif np.unique(finite).size <= 1:
            dropped_constant += 1
        else:
            keep.append(idx)
    medians = np.nanmedian(train[:, keep], axis=0) if keep else np.array([], dtype=float)
    out = {}
    for split, matrix in matrices.items():
        sub = matrix[:, keep].astype(float) if keep else np.zeros((matrix.shape[0], 0))
        indices = np.where(~np.isfinite(sub))
        if indices[0].size:
            sub[indices] = np.take(medians, indices[1])
        out[split] = sub.astype(np.float32)
    return out, [feature_names[i] for i in keep], {
        "input_rule_features": len(feature_names), "kept_rule_features": len(keep),
        "dropped_constant_rule_features": dropped_constant,
        "dropped_all_nan_rule_features": dropped_all_nan,
        "duplicate_feature_names": "|".join(duplicates),
    }


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def _resolve_reference(value, origin, project_root):
    path = Path(value)
    if path.is_absolute():
        if not path.is_file():
            raise FileNotFoundError(path)
        return path.resolve()
    candidates = {p.resolve() for p in (origin.parent / path, project_root / path) if p.is_file()}
    # Run artifacts record repository-relative paths, even when copied together.
    for ancestor in origin.parents:
        if ancestor.name == "artifacts":
            candidate = ancestor.parent / path
            if candidate.is_file():
                candidates.add(candidate.resolve())
    if len(candidates) != 1:
        raise ValueError(f"Artifact reference is missing or ambiguous: {value!r} from {origin}")
    return candidates.pop()


def resolve_rule_source(source: Path, dataset: str, project_root: Path = ROOT):
    """Resolve final top-level rules through summary -> outcome -> metadata.

    An explicit bare rule list is accepted, but cannot certify the historical
    final branch. Candidate-all rules are never substituted for top-level rules.
    """
    path = Path(source).resolve()
    if (path.is_dir() and (path / "summary.json").is_file()) or path.name == "summary.json":
        run_dir = path if path.is_dir() else path.parent
        inferred_root = run_dir.parent.parent if run_dir.parent.name == "artifacts" and Path(project_root) == ROOT else project_root
        resolved = resolve_run_rules(run_dir, dataset=dataset, project_root=inferred_root)
        matches = [item for item in resolved if not item["subtask"]]
        if len(matches) != 1:
            raise ValueError("Run must contain exactly one matching single-task final artifact")
        item = matches[0]
        rules, info = resolve_rule_source(item["rules_path"], dataset, project_root)
        paths = [item["summary_path"], run_dir / "run_config.json", item["outcome_path"]]
        info["artifact_chain"] = [{"path": str(p), "sha256": sha256_file(p)} for p in paths] + info["artifact_chain"]
        info.update(requested_source=str(source), final_result_source=item["final_result_source"],
                    recorded_dataset_provenance=item["recorded_dataset_provenance"],
                    run_tag=item["run_tag"], model=item["model"])
        return rules, info
    if path.is_dir():
        summaries = [path / name for name in ("summary.json", "run_summary.json") if (path / name).is_file()]
        if summaries:
            if len(summaries) != 1:
                raise ValueError("Run directory has multiple summaries; supply one explicitly")
            path = summaries[0]
        else:
            candidates = []
            for candidate in path.rglob("*_rules.json"):
                obj = _read_json(candidate)
                if isinstance(obj, dict) and obj.get("dataset") == dataset and obj.get("knowledge_type") == "all":
                    candidates.append(candidate)
            if len(candidates) != 1:
                raise ValueError("Run directory must identify exactly one final all metadata artifact for this dataset")
            path = candidates[0]
    chain, expected_source, recorded_data, visited = [], None, None, set()
    while True:
        path = path.resolve()
        if path in visited:
            raise ValueError("Cyclic rule artifact references")
        visited.add(path)
        obj = _read_json(path)
        chain.append({"path": str(path), "sha256": sha256_file(path)})
        if isinstance(obj, dict) and "tasks" in obj:
            matches = [task for task in obj["tasks"] if task.get("dataset") == dataset and not task.get("subtask")]
            if len(matches) != 1:
                raise ValueError("Summary must contain exactly one matching single-task result")
            task = matches[0]
            result = task.get("result") or task.get("branch_details", {}).get("all", {}).get("result", {})
            if result.get("outcome_category") not in {"success_primary", "success_fallback"}:
                raise ValueError("Final run result is not a successful measured result")
            expected_source = result.get("final_result_source")
            recorded_data = task.get("dataset_provenance")
            path = _resolve_reference(result["outcome_path"], path, project_root)
            continue
        if isinstance(obj, dict) and isinstance(obj.get("artifact_paths"), dict):
            if obj.get("outcome_category") not in {"success_primary", "success_fallback"}:
                raise ValueError("Cannot evaluate a failed/degraded outcome as final rules")
            if obj.get("dataset") not in (None, dataset):
                raise ValueError("Outcome dataset does not match requested dataset")
            source_name = obj.get("final_result_source")
            if expected_source and source_name != expected_source:
                raise ValueError("Summary and outcome disagree about final_result_source")
            expected_source = source_name
            path = _resolve_reference(obj["artifact_paths"]["metadata"], path, project_root)
            continue
        if isinstance(obj, dict):
            if obj.get("dataset") not in (None, dataset) or obj.get("subtask"):
                raise ValueError("Rule metadata does not match requested single-task dataset")
            if obj.get("status") not in (None, "success", "success_primary", "success_fallback"):
                raise ValueError("Rule metadata is not successful")
            source_name = obj.get("final_result_source") or obj.get("knowledge_type")
            if expected_source and source_name != expected_source:
                raise ValueError("Outcome and rule metadata disagree about final_result_source")
            rules = obj.get("rules")  # top-level rules include inference fallback; never candidate_all_result
        else:
            source_name, rules = None, obj
        if not isinstance(rules, list) or not rules or not all(isinstance(rule, dict) for rule in rules):
            raise ValueError("Rule artifact must contain a nonempty list of rule objects")
        return rules, {
            "requested_source": str(source), "artifact_chain": chain,
            "resolved_rules_path": str(path), "resolved_rules_sha256": sha256_file(path),
            "final_result_source": source_name or "explicit_rules_unverified_final_branch",
            "source_semantics": "top_level_rules", "recorded_dataset_provenance": recorded_data,
        }


def load_splits(dataset, data_root, recorded_data=None):
    spec = get_dataset_spec(dataset, input_folder=str(Path(data_root).resolve()))
    smiles, labels, records = {}, {}, {}
    for split in SPLITS:
        path = Path(spec.folder) / f"{spec.file_stem}_{split}.csv"
        frame = pd.read_csv(path)
        if frame.empty or frame[spec.smiles_column].isna().any():
            raise ValueError(f"{split}: empty split or missing SMILES")
        smiles[split] = frame[spec.smiles_column].astype(str).tolist()
        labels[split] = pd.to_numeric(frame[spec.target_column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(labels[split]).all():
            raise ValueError(f"{split}: nonfinite labels")
        if spec.task_type == "classification" and set(np.unique(labels[split])) != {0.0, 1.0}:
            raise ValueError(f"{split}: binary ROC-AUC requires both 0 and 1 labels")
        for idx, value in enumerate(smiles[split]):
            if not value.strip() or Chem.MolFromSmiles(value) is None:
                raise ValueError(f"{split}: invalid SMILES at zero-based row {idx}; rows are never dropped")
        digest = sha256_file(path)
        if recorded_data:
            expected = recorded_data.get("splits", {}).get(split, {}).get("file_sha256")
            if not expected or expected != digest:
                raise ValueError(f"{split}: data differ from final run's recorded split hash")
        records[split] = {
            "path": str(path), "file_sha256": digest, "row_count": len(frame),
            "ordered_smiles_sha256": hashlib.sha256(("\n".join(smiles[split]) + "\n").encode()).hexdigest(),
        }
    if recorded_data and (recorded_data.get("target_column") != spec.target_column or recorded_data.get("smiles_column") != spec.smiles_column):
        raise ValueError("Run provenance and current dataset label/SMILES columns disagree")
    return smiles, labels, spec, {
        "dataset": dataset, "target_column": spec.target_column, "smiles_column": spec.smiles_column,
        "splits": records, "run_split_hash_verified": bool(recorded_data),
    }


def make_feature_arms(smiles, rules, radius, n_bits, shuffled_seed, random_seed):
    compiled, invalid = compile_rules(rules)
    if not compiled or invalid:
        raise ValueError(f"Final rules must all compile: compiled={len(compiled)}, invalid={len(invalid)}")
    raw = {split: np.asarray(build_feature_matrix(compiled, smiles[split]), dtype=float) for split in SPLITS}
    rule_matrices, kept, sanitization = sanitize_rule_matrices(raw, [item.name for item in compiled])
    if not kept:
        raise ValueError("No variable rule features survive train-only sanitization")
    base = {}
    for split in SPLITS:
        base[split] = np.zeros((len(smiles[split]), n_bits), dtype=np.float32)
        for idx, smi in enumerate(smiles[split]):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"{split}: invalid SMILES at row {idx}")
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros(n_bits, dtype=np.int8)
            DataStructs.ConvertToNumpyArray(fp, arr)
            base[split][idx] = arr.astype(np.float32)
    arms = {
        "base": base, "rules": rule_matrices,
        "base_plus_rules": append_control_columns(base, rule_matrices),
        "base_plus_shuffled_rules": append_control_columns(base, make_split_local_shuffled_controls(rule_matrices, shuffled_seed)),
        "base_plus_random_rules": append_control_columns(base, make_gaussian_count_matched_controls(rule_matrices, random_seed)),
    }
    return arms, {"compiled_rule_count": len(compiled), "kept_feature_names": kept, "sanitization": sanitization}


def evaluate_one_seed(task_type, matrices, labels, params, seed):
    """Exact P13 estimator/preprocessing protocol; no tuning on test labels."""
    train = matrices["train"]
    evaluation = {split: matrices[split] for split in ("valid", "test")}
    rf_params = dict(params)
    rf_params["random_state"] = int(seed)
    if task_type == "classification":
        scaler = StandardScaler()
        train = scaler.fit_transform(train)
        evaluation = {split: scaler.transform(value) for split, value in evaluation.items()}
        model = RandomForestClassifier(**rf_params)
        model.fit(train, labels["train"].astype(int))
        return {split: float(roc_auc_score(labels[split].astype(int), model.predict_proba(value)[:, 1])) for split, value in evaluation.items()}
    if task_type != "regression":
        raise ValueError(f"Unsupported task type: {task_type}")
    model = RandomForestRegressor(**rf_params)
    model.fit(train, labels["train"])
    return {split: float(np.sqrt(mean_squared_error(labels[split], model.predict(value)))) for split, value in evaluation.items()}


def evaluate_paired(arms, labels, task_type, params, rf_seeds):
    if not rf_seeds or len(rf_seeds) != len(set(rf_seeds)):
        raise ValueError("RF seeds must be nonempty and unique")
    rows = []
    sign = 1 if task_type == "classification" else -1
    for seed in rf_seeds:
        scores = {arm: evaluate_one_seed(task_type, arms[arm], labels, params, seed) for arm in ARMS}
        for split in ("valid", "test"):
            base = scores["base"][split]
            for arm in ARMS:
                rows.append({"rf_seed": seed, "split": split, "arm": arm, "score": scores[arm][split],
                             "directional_delta_vs_base": sign * (scores[arm][split] - base)})
    return rows


def summarize(rows, task_type):
    frame = pd.DataFrame(rows)
    arm_rows, delta_rows = [], []
    sign = 1 if task_type == "classification" else -1
    for split in ("valid", "test"):
        pivot = frame[frame.split == split].pivot(index="rf_seed", columns="arm", values="score")
        for arm in ARMS:
            values = pivot[arm].to_numpy()
            arm_rows.append({"split": split, "arm": arm, "n_rf_seeds": len(values),
                             "mean": float(values.mean()), "std_ddof0": float(values.std(ddof=0))})
        comparisons = [(arm, "base") for arm in ARMS if arm != "base"]
        comparisons += [("base_plus_rules", arm) for arm in ("base_plus_shuffled_rules", "base_plus_random_rules")]
        for candidate, comparator in comparisons:
            deltas = sign * (pivot[candidate] - pivot[comparator]).to_numpy()
            delta_rows.append({"split": split, "candidate": candidate, "comparator": comparator,
                               "n_rf_seeds": len(deltas), "mean_directional_delta": float(deltas.mean()),
                               "std_ddof0": float(deltas.std(ddof=0)), "positive_seed_count": int((deltas > 0).sum())})
    return arm_rows, delta_rows


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(*, dataset, source, output_dir, config_path=None, registry_path=None,
                   data_root=None, project_root=ROOT, rf_seeds=None, seed_profile="five",
                   control_seeds=None, historical_control_key=None):
    if dataset not in DATASETS:
        raise ValueError(f"Only these single-task datasets are supported: {DATASETS}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    config_path = Path(config_path or ROOT / "configs/feature_experiments.json").resolve()
    registry_path = Path(registry_path or ROOT / "configs/model_registry.json").resolve()
    config, registry = _read_json(config_path), _read_json(registry_path)
    if config.get("schema_version") != SCHEMA:
        raise ValueError("Unsupported feature experiment configuration schema")
    seeds = list(rf_seeds) if rf_seeds is not None else config["rf_seed_profiles"][seed_profile]
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or s < 0 or s >= 2**32 for s in seeds):
        raise ValueError("RF seeds must be unique integers in [0, 2**32)")
    if control_seeds is not None and historical_control_key is not None:
        raise ValueError("Choose explicit control seeds or a historical control key")
    if historical_control_key is not None:
        route, repeat = historical_control_key
        controls = [stable_seed(P13_PHASE, route, dataset, repeat, name) for name in ("shuffled_rules", "random_features")]
    else:
        controls = list(control_seeds if control_seeds is not None else config["control_seeds"])
    if len(controls) != 2 or any(type(s) is not int or s < 0 for s in controls):
        raise ValueError("Exactly two nonnegative integer control seeds are required")
    fingerprint = config["fingerprints"][dataset]
    radius, n_bits = fingerprint["radius"], fingerprint["n_bits"]
    if type(radius) is not int or radius < 0 or type(n_bits) is not int or n_bits <= 0:
        raise ValueError("Invalid Morgan fingerprint configuration")
    rules, source_info = resolve_rule_source(Path(source), dataset, Path(project_root).resolve())
    smiles, labels, spec, data_info = load_splits(dataset, data_root or ROOT / "data/splits", source_info["recorded_dataset_provenance"])
    arms, feature_info = make_feature_arms(smiles, rules, radius, n_bits, *controls)
    matrix_info = {arm: {split: {"shape": list(matrix.shape), "dtype": str(matrix.dtype),
                               "c_order_bytes_sha256": hashlib.sha256(np.ascontiguousarray(matrix).tobytes()).hexdigest()}
                         for split, matrix in matrices.items()} for arm, matrices in arms.items()}
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "rdkit")}
    protocol = {"schema_version": SCHEMA, "dataset": dataset, "task_type": spec.task_type,
                "metric": "roc_auc" if spec.task_type == "classification" else "rmse",
                "metric_direction": "higher" if spec.task_type == "classification" else "lower",
                "rf_seeds": seeds, "rf_params": registry[dataset], "random_state_override": "each paired rf_seed",
                "arms": list(ARMS), "fingerprint": fingerprint,
                "classification_preprocessing": "train-fit StandardScaler per arm", "regression_preprocessing": "none",
                "control_seeds": {"shuffled_rules": controls[0], "random_rules": controls[1]},
                "historical_control_key": historical_control_key,
                "controls_frozen_across_rf_seeds": True, "controls_definition": "P13 split-local shuffle and train-moment Gaussian",
                "std_ddof": 0, "positive_delta_means": "candidate improves over comparator",
                "test_used_for_selection": False, "source": source_info, "data": data_info,
                "features": feature_info, "matrices": matrix_info, "python": platform.python_version(), "versions": versions,
                "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
                "registry": {"path": str(registry_path), "sha256": sha256_file(registry_path)},
                "implementation_sha256": sha256_file(Path(__file__)),
                "feature_dependency_sha256": {name: sha256_file(ROOT / "morace" / name)
                                               for name in ("datasets.py", "rule_compiler.py", "rule_registry.py", "artifact_inputs.py")},
                "scope": "frozen rules; RF-seed variation only, not generation- or split-seed uncertainty"}
    output.mkdir(parents=True, exist_ok=False)  # never mix with or overwrite an earlier experiment
    _write_json(output / "protocol.json", protocol)
    _write_json(output / "resolved_rules.json", {"dataset": dataset, "final_result_source": source_info["final_result_source"], "rules": rules})
    _write_json(output / "status.json", {"status": "running"})
    try:
        rows = evaluate_paired(arms, labels, spec.task_type, registry[dataset], seeds)
        arm_summary, delta_summary = summarize(rows, spec.task_type)
        _write_csv(output / "per_seed.csv", rows)
        _write_csv(output / "summary.csv", arm_summary)
        _write_csv(output / "paired_deltas.csv", delta_summary)
        _write_json(output / "status.json", {"status": "success", "fits": len(ARMS) * len(seeds), "score_rows": len(rows)})
    except Exception as exc:
        _write_json(output / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate five paired feature arms on one frozen final rule artifact (no LLM required).")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--source", required=True, type=Path, help="Final rule JSON, outcome JSON, summary JSON, or run directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory; existing paths are rejected")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/feature_experiments.json")
    parser.add_argument("--registry", type=Path, default=ROOT / "configs/model_registry.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/splits")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="Root used for repository-relative artifact references")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rf-seeds", type=int, nargs="+", help="Explicit paired estimator random_state values")
    group.add_argument("--seed-profile", choices=("five", "fifty"), default="five")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--control-seeds", type=int, nargs=2, metavar=("SHUFFLE", "GAUSSIAN"))
    controls.add_argument("--historical-control-key", nargs=2, metavar=("ROUTE", "REPEAT"), help="Exact P13 labels, e.g. llama r1; reconstructs historical noise seeds")
    args = parser.parse_args(argv)
    output = run_experiment(dataset=args.dataset, source=args.source, output_dir=args.output_dir,
                            config_path=args.config, registry_path=args.registry, data_root=args.data_root,
                            project_root=args.project_root, rf_seeds=args.rf_seeds, seed_profile=args.seed_profile,
                            control_seeds=args.control_seeds, historical_control_key=args.historical_control_key)
    print(f"Feature experiments completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
