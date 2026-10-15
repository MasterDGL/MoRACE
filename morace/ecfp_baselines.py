#!/usr/bin/env python
"""Morgan/ECFP baselines using the fixed split manifest.

This script intentionally never creates a new split. It reads
data/manifests/split_manifest.csv and reuses the referenced train/valid/test
CSV files. In --dry-run mode it performs only schema, count, RDKit parse, and
fingerprint smoke checks; it does not train models or write formal metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import platform
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem
    from rdkit import DataStructs
    from rdkit.Chem import AllChem
except Exception as exc:  # pragma: no cover - exercised by environment
    Chem = None
    DataStructs = None
    AllChem = None
    RDKIT_IMPORT_ERROR = exc
else:
    RDKIT_IMPORT_ERROR = None


CLASSIFICATION_DATASETS = {"bace", "bbbp", "clintox", "hiv"}
REGRESSION_DATASETS = {"esol", "freesolv", "lipophilicity"}
DEFAULT_DATASETS = [
    "bace",
    "bbbp",
    "clintox",
    "esol",
    "freesolv",
    "hiv",
    "lipophilicity",
]
DEFAULT_RF_SEEDS = [0, 1, 2, 3, 4]
FINGERPRINT_CONFIGS = [
    ("ecfp4_r2_2048", 2, 2048),
    ("ecfp6_r3_2048", 3, 2048),
]


@dataclass(frozen=True)
class SplitRecord:
    dataset: str
    split_name: str
    split_file_path: Path
    row_count: int
    smiles_column: str
    target_column: str
    file_sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def load_manifest(manifest_path: Path, project_root: Path) -> Dict[Tuple[str, str], SplitRecord]:
    _, rows = read_csv_rows(manifest_path)
    records: Dict[Tuple[str, str], SplitRecord] = {}
    for row in rows:
        dataset = row["dataset"].strip()
        split = row["split_name"].strip()
        raw_path = Path(row["split_file_path"].strip())
        split_path = raw_path if raw_path.is_absolute() else project_root / raw_path
        records[(dataset, split)] = SplitRecord(
            dataset=dataset,
            split_name=split,
            split_file_path=split_path,
            row_count=int(row["row_count"]),
            smiles_column=row["smiles_column"].strip(),
            target_column=row["target_column"].strip(),
            file_sha256=row["file_sha256"].strip(),
        )
    return records


def task_type_for_dataset(dataset: str) -> str:
    if dataset in CLASSIFICATION_DATASETS:
        return "classification"
    if dataset in REGRESSION_DATASETS:
        return "regression"
    return "unknown"


def parse_bool_issue(value: bool, message: str) -> str:
    return "" if value else message


def mol_from_smiles(smiles: str):
    if Chem is None:
        return None
    return Chem.MolFromSmiles(str(smiles))


def morgan_bits_for_mol(mol, radius: int, n_bits: int) -> List[int]:
    if AllChem is None:
        raise RuntimeError(f"RDKit unavailable: {RDKIT_IMPORT_ERROR}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return [int(bit) for bit in fp.ToBitString()]


def morgan_bits_for_smiles(smiles: str, radius: int, n_bits: int) -> Optional[List[int]]:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    return morgan_bits_for_mol(mol, radius=radius, n_bits=n_bits)


def sample_valid_smiles(smiles_values: Sequence[str], sample_size: int = 5) -> List[str]:
    return [str(value) for value in smiles_values[:sample_size]]


def dry_run_dataset(
    dataset: str,
    records: Dict[Tuple[str, str], SplitRecord],
    sample_size: int = 5,
) -> Dict[str, str]:
    issues: List[str] = []
    task_type = task_type_for_dataset(dataset)
    if task_type == "unknown":
        issues.append("unknown task type")

    split_rows: Dict[str, List[Dict[str, str]]] = {}
    split_headers: Dict[str, List[str]] = {}
    row_counts: Dict[str, int] = {}
    manifest_counts: Dict[str, int] = {}
    missing_smiles_total = 0
    missing_target_total = 0

    for split in ("train", "valid", "test"):
        record = records.get((dataset, split))
        if record is None:
            issues.append(f"missing manifest record for {split}")
            split_rows[split] = []
            split_headers[split] = []
            row_counts[split] = 0
            manifest_counts[split] = 0
            continue
        manifest_counts[split] = record.row_count
        if not record.split_file_path.exists():
            issues.append(f"missing split file for {split}: {record.split_file_path}")
            split_rows[split] = []
            split_headers[split] = []
            row_counts[split] = 0
            continue
        actual_hash = sha256_file(record.split_file_path)
        if actual_hash != record.file_sha256:
            issues.append(f"file hash mismatch for {split}")
        headers, rows = read_csv_rows(record.split_file_path)
        split_headers[split] = headers
        split_rows[split] = rows
        row_counts[split] = len(rows)
        if len(rows) != record.row_count:
            issues.append(f"row count mismatch for {split}: {len(rows)} != {record.row_count}")
        if record.smiles_column not in headers:
            issues.append(f"missing SMILES column {record.smiles_column} in {split}")
            continue
        if record.target_column not in headers:
            issues.append(f"missing target column {record.target_column} in {split}")
            continue
        missing_smiles_total += sum(1 for row in rows if not str(row.get(record.smiles_column, "")).strip())
        missing_target_total += sum(1 for row in rows if not str(row.get(record.target_column, "")).strip())

    if missing_smiles_total:
        issues.append(f"missing SMILES values: {missing_smiles_total}")
    if missing_target_total:
        issues.append(f"missing target values: {missing_target_total}")

    train_record = records.get((dataset, "train"))
    smiles_column = train_record.smiles_column if train_record else "unknown"
    target_column = train_record.target_column if train_record else "unknown"
    train_rows = split_rows.get("train", [])
    smiles_values = [row.get(smiles_column, "") for row in train_rows if smiles_column in row]
    sample_smiles = sample_valid_smiles(smiles_values, sample_size=sample_size)

    rdkit_parse_ok = False
    ecfp_ok = False
    if Chem is None or AllChem is None:
        issues.append(f"rdkit import failed: {RDKIT_IMPORT_ERROR}")
    elif sample_smiles:
        mols = [mol_from_smiles(smiles) for smiles in sample_smiles]
        rdkit_parse_ok = all(mol is not None for mol in mols)
        if not rdkit_parse_ok:
            issues.append("RDKit failed to parse one or more sample SMILES")
        else:
            try:
                for _, radius, n_bits in FINGERPRINT_CONFIGS:
                    for mol in mols:
                        bits = morgan_bits_for_mol(mol, radius=radius, n_bits=n_bits)
                        if len(bits) != n_bits:
                            raise RuntimeError(f"fingerprint length {len(bits)} != {n_bits}")
                ecfp_ok = True
            except Exception as exc:
                issues.append(f"ECFP generation failed: {type(exc).__name__}: {exc}")
    else:
        issues.append("no sample SMILES available")

    status = "passed" if not issues else "failed"
    return {
        "dataset": dataset,
        "task_type": task_type,
        "train_rows": str(row_counts.get("train", 0)),
        "valid_rows": str(row_counts.get("valid", 0)),
        "test_rows": str(row_counts.get("test", 0)),
        "smiles_column": smiles_column,
        "target_column": target_column,
        "manifest_train_rows": str(manifest_counts.get("train", 0)),
        "manifest_valid_rows": str(manifest_counts.get("valid", 0)),
        "manifest_test_rows": str(manifest_counts.get("test", 0)),
        "rdkit_parse_sample_ok": str(rdkit_parse_ok).lower(),
        "ecfp_sample_ok": str(ecfp_ok).lower(),
        "missing_smiles": str(missing_smiles_total),
        "missing_target": str(missing_target_total),
        "issue": "; ".join(issues),
        "status": status,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_dry_run_markdown(path: Path, dry_rows: Sequence[Dict[str, str]], lightgbm_available: bool) -> None:
    status_counts = {}
    for row in dry_rows:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    table_lines = [
        "| dataset | task_type | train | valid | test | RDKit parse | ECFP | missing SMILES | missing target | status |",
        "|---|---|---:|---:|---:|---|---|---:|---:|---|",
    ]
    for row in dry_rows:
        table_lines.append(
            f"| {row['dataset']} | {row['task_type']} | {row['train_rows']} | {row['valid_rows']} | {row['test_rows']} | "
            f"{row['rdkit_parse_sample_ok']} | {row['ecfp_sample_ok']} | {row['missing_smiles']} | {row['missing_target']} | {row['status']} |"
        )
    text = f"""# Morgan/ECFP Strong Baseline Dry Run 20260520

This is a dry-run validation only. It did not train RandomForest or LightGBM models and did not write formal metrics.

## Split Protocol

The dry run reused the fixed pre-materialized train/valid/test CSV splits recorded in `data/manifests/split_manifest.csv`. No new split was created, and no `train_test_split`, `RandomSplitter`, or `ScaffoldSplitter` is used by this baseline script.

## Dependency Status

- RDKit: required for dry-run parse/fingerprint checks.
- scikit-learn: required only for formal RandomForest training.
- LightGBM available in this environment: `{str(lightgbm_available).lower()}`.

## Dataset Checks

{chr(10).join(table_lines)}

## Status Counts

```json
{json.dumps(status_counts, indent=2, ensure_ascii=False)}
```

## Boundary

- No GPU was used.
- No LLM was called.
- No model training was run.
- No formal baseline metrics were generated.
- Existing split CSV files were not modified.
"""
    path.write_text(text, encoding="utf-8")


def import_status(module_name: str, version_attr: str = "__version__") -> Dict[str, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {
            "module": module_name,
            "available": "false",
            "version": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(module, version_attr, "")
    return {
        "module": module_name,
        "available": "true",
        "version": str(version),
        "error": "",
    }


def write_dependency_preflight(project_root: Path, json_path: Path, md_path: Path) -> int:
    checks = [
        import_status("numpy"),
        import_status("pandas"),
        import_status("rdkit"),
        import_status("sklearn"),
        import_status("lightgbm"),
    ]
    payload = {
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "project_root": str(project_root),
        "checks": checks,
        "all_required_for_formal_training_available": all(
            item["available"] == "true" for item in checks if item["module"] in {"numpy", "rdkit", "sklearn"}
        ),
        "lightgbm_available": next(item["available"] for item in checks if item["module"] == "lightgbm") == "true",
        "note": "LightGBM is optional; if unavailable, formal training skips LightGBM and records an issue.",
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    table_lines = [
        "| package | available | version | error |",
        "|---|---|---|---|",
    ]
    for item in checks:
        table_lines.append(
            f"| {item['module']} | {item['available']} | {item['version']} | {item['error']} |"
        )
    text = f"""# Morgan/ECFP Dependency Preflight 20260520

This preflight checks the Python dependencies needed for the Morgan/ECFP + RF/LightGBM baseline workflow. It does not train models, does not call an LLM, and does not use GPU.

## Runtime

- Python executable: `{payload['python_executable']}`
- Python version: `{payload['python_version']}`
- Platform: `{payload['platform']}`
- Project root: `{payload['project_root']}`

## Dependency Checks

{chr(10).join(table_lines)}

## Interpretation

- `numpy`, `rdkit`, and `sklearn` are required for formal RandomForest training.
- `lightgbm` is optional. If unavailable, the script skips LightGBM and writes an issue file during formal training.
- `pandas` is checked because it is common in the server environment, but the current script uses Python's standard CSV reader for manifest and split loading.
"""
    md_path.write_text(text, encoding="utf-8")
    return 0


def optional_import_sklearn():
    try:
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.metrics import mean_squared_error, roc_auc_score
    except Exception as exc:
        return None, exc
    return (RandomForestClassifier, RandomForestRegressor, mean_squared_error, roc_auc_score), None


def optional_import_lightgbm():
    try:
        from lightgbm import LGBMClassifier, LGBMRegressor
    except Exception as exc:
        return None, exc
    return (LGBMClassifier, LGBMRegressor), None


def load_xy_from_record(record: SplitRecord) -> Tuple[List[str], List[float]]:
    headers, rows = read_csv_rows(record.split_file_path)
    if record.smiles_column not in headers:
        raise ValueError(f"Missing SMILES column {record.smiles_column} in {record.split_file_path}")
    if record.target_column not in headers:
        raise ValueError(f"Missing target column {record.target_column} in {record.split_file_path}")
    smiles = [row[record.smiles_column] for row in rows]
    y = [float(row[record.target_column]) for row in rows]
    return smiles, y


def featurize_smiles(smiles: Sequence[str], radius: int, n_bits: int):
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy is required for formal training: {type(exc).__name__}: {exc}") from exc
    if Chem is None or AllChem is None or DataStructs is None:
        raise RuntimeError(f"RDKit unavailable: {RDKIT_IMPORT_ERROR}")

    matrix = np.zeros((len(smiles), n_bits), dtype=np.uint8)
    invalid_count = 0
    for row_idx, item in enumerate(smiles):
        mol = mol_from_smiles(item)
        if mol is None:
            invalid_count += 1
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        DataStructs.ConvertToNumpyArray(fp, matrix[row_idx])
    return matrix, invalid_count


def regression_rmse(mean_squared_error, y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def mean_or_blank(values: Sequence[float]) -> str:
    return "" if not values else str(float(statistics.mean(values)))


def std_or_blank(values: Sequence[float]) -> str:
    return "" if len(values) < 2 else str(float(statistics.stdev(values)))


def summarize_results(results: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = {}
    for row in results:
        key = (row["dataset"], row["task_type"], row["fingerprint"], row["model"])
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict[str, str]] = []
    for (dataset, task_type, fingerprint, model), rows in sorted(grouped.items()):
        valid_scores = [float(row["valid_score"]) for row in rows]
        test_scores = [float(row["test_score"]) for row in rows]
        summary_rows.append({
            "dataset": dataset,
            "task_type": task_type,
            "fingerprint": fingerprint,
            "model": model,
            "metric": rows[0]["metric"],
            "n_seeds": str(len(rows)),
            "valid_mean": mean_or_blank(valid_scores),
            "valid_std": std_or_blank(valid_scores),
            "test_mean": mean_or_blank(test_scores),
            "test_std": std_or_blank(test_scores),
            "selection_rule": "group-level selection: classification max valid_mean ROC-AUC; regression min valid_mean RMSE",
            "split_protocol": "fixed_pre_materialized_train_valid_test_csv",
        })
    return summary_rows


def valid_selected_rows(summary_rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["dataset"], []).append(row)

    selected: List[Dict[str, str]] = []
    for dataset, rows in sorted(grouped.items()):
        task_type = rows[0]["task_type"]
        if task_type == "classification":
            best = max(rows, key=lambda row: float(row["valid_mean"]))
            selection_rule = "max_valid_mean_roc_auc_over_seeds"
        else:
            best = min(rows, key=lambda row: float(row["valid_mean"]))
            selection_rule = "min_valid_mean_rmse_over_seeds"
        selected.append({
            "dataset": dataset,
            "task_type": task_type,
            "selected_model": best["model"],
            "selected_fingerprint": best["fingerprint"],
            "metric": best["metric"],
            "n_seeds": best["n_seeds"],
            "valid_mean": best["valid_mean"],
            "valid_std": best["valid_std"],
            "test_mean": best["test_mean"],
            "test_std": best["test_std"],
            "selection_rule": selection_rule,
            "test_score_reporting_note": "test mean±sd reported only after group-level validation selection; test is not used for selection",
            "split_protocol": best["split_protocol"],
        })
    return selected


def write_summary_markdown(path: Path, summary_rows: Sequence[Dict[str, str]], selected_rows: Sequence[Dict[str, str]], issues: Sequence[Dict[str, str]]) -> None:
    table_lines = [
        "| dataset | task_type | selected_model | selected_fingerprint | n_seeds | metric | valid_mean | valid_std | test_mean | test_std | selection_rule |",
        "|---|---|---|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in selected_rows:
        table_lines.append(
            f"| {row['dataset']} | {row['task_type']} | {row['selected_model']} | {row['selected_fingerprint']} | {row['n_seeds']} | "
            f"{row['metric']} | {row['valid_mean']} | {row['valid_std']} | {row['test_mean']} | {row['test_std']} | {row['selection_rule']} |"
        )
    text = f"""# Morgan/ECFP Strong Baseline Summary 20260520

This summary is generated only during explicit `--run-formal-training` execution. It uses fixed pre-materialized train/valid/test CSV splits from the split manifest and does not create a new split.

## Valid-Based Selection

Model/fingerprint selection is based on mean validation performance over seeds:

- classification: highest valid ROC-AUC mean across seeds;
- regression: lowest valid RMSE mean across seeds.

The script does not select the best individual seed. Seed-level results remain in `ecfp_baseline_metrics.csv`, while `ecfp_baseline_summary.csv` aggregates each dataset/model/fingerprint group as mean±sd.

Test mean±sd is reported only after validation-based model/fingerprint selection. No test score is used for model/fingerprint selection.

## Selected Configurations

{chr(10).join(table_lines)}

## Summary Rows

- Aggregated summary rows: {len(summary_rows)}
- Issue rows: {len(issues)}

## Boundary

- No LLM is called by this script.
- No GPU is required.
- Split CSV files are read only.
"""
    path.write_text(text, encoding="utf-8")


def run_training(
    datasets: Sequence[str],
    records: Dict[Tuple[str, str], SplitRecord],
    output_root: Path,
    rf_seeds: Sequence[int],
) -> int:
    """Run formal baselines.

    This path is intentionally not exercised by the current dry-run task. It is
    implemented so the script is ready for later explicit authorization while
    preserving the same manifest-based split loading.
    """
    if Chem is None or AllChem is None:
        raise RuntimeError(f"RDKit unavailable: {RDKIT_IMPORT_ERROR}")
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(f"numpy is required for formal training: {type(exc).__name__}: {exc}") from exc

    sklearn_mods, sklearn_error = optional_import_sklearn()
    lightgbm_mods, lightgbm_error = optional_import_lightgbm()
    output_root.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, str]] = []
    issues: List[Dict[str, str]] = []

    if sklearn_mods is None:
        issues.append({
            "model": "random_forest",
            "issue": f"sklearn unavailable: {type(sklearn_error).__name__}: {sklearn_error}",
        })
    if lightgbm_mods is None:
        issues.append({
            "model": "lightgbm",
            "issue": f"lightgbm unavailable: {type(lightgbm_error).__name__}: {lightgbm_error}",
        })

    for dataset in datasets:
        task_type = task_type_for_dataset(dataset)
        train_record = records[(dataset, "train")]
        valid_record = records[(dataset, "valid")]
        test_record = records[(dataset, "test")]
        train_smiles, y_train = load_xy_from_record(train_record)
        valid_smiles, y_valid = load_xy_from_record(valid_record)
        test_smiles, y_test = load_xy_from_record(test_record)
        y_train_arr = np.asarray(y_train)
        y_valid_arr = np.asarray(y_valid)
        y_test_arr = np.asarray(y_test)

        for fp_name, radius, n_bits in FINGERPRINT_CONFIGS:
            x_train, invalid_train = featurize_smiles(train_smiles, radius, n_bits)
            x_valid, invalid_valid = featurize_smiles(valid_smiles, radius, n_bits)
            x_test, invalid_test = featurize_smiles(test_smiles, radius, n_bits)
            invalid_total = invalid_train + invalid_valid + invalid_test

            if sklearn_mods is not None:
                RandomForestClassifier, RandomForestRegressor, mean_squared_error, roc_auc_score = sklearn_mods
                for seed in rf_seeds:
                    if task_type == "classification":
                        estimator = RandomForestClassifier(
                            n_estimators=500,
                            random_state=seed,
                            n_jobs=-1,
                            class_weight="balanced",
                        )
                        estimator.fit(x_train, y_train_arr.astype(int))
                        valid_score = float(roc_auc_score(y_valid_arr.astype(int), estimator.predict_proba(x_valid)[:, 1]))
                        test_score = float(roc_auc_score(y_test_arr.astype(int), estimator.predict_proba(x_test)[:, 1]))
                        metric = "roc_auc"
                    else:
                        estimator = RandomForestRegressor(
                            n_estimators=500,
                            random_state=seed,
                            n_jobs=-1,
                        )
                        estimator.fit(x_train, y_train_arr)
                        valid_score = regression_rmse(mean_squared_error, y_valid_arr, estimator.predict(x_valid))
                        test_score = regression_rmse(mean_squared_error, y_test_arr, estimator.predict(x_test))
                        metric = "rmse"
                    results.append({
                        "dataset": dataset,
                        "task_type": task_type,
                        "fingerprint": fp_name,
                        "radius": str(radius),
                        "n_bits": str(n_bits),
                        "model": "random_forest",
                        "seed": str(seed),
                        "metric": metric,
                        "valid_score": str(valid_score),
                        "test_score": str(test_score),
                        "invalid_smiles_count": str(invalid_total),
                        "split_protocol": "fixed_pre_materialized_train_valid_test_csv",
                    })

            if lightgbm_mods is not None:
                LGBMClassifier, LGBMRegressor = lightgbm_mods
                for seed in rf_seeds:
                    if task_type == "classification":
                        if sklearn_mods is None:
                            issues.append({"model": "lightgbm", "issue": "sklearn metrics unavailable for ROC-AUC"})
                            continue
                        _, _, _, roc_auc_score = sklearn_mods
                        estimator = LGBMClassifier(
                            n_estimators=500,
                            learning_rate=0.05,
                            num_leaves=31,
                            random_state=seed,
                            n_jobs=-1,
                            class_weight="balanced",
                        )
                        estimator.fit(x_train, y_train_arr.astype(int))
                        valid_score = float(roc_auc_score(y_valid_arr.astype(int), estimator.predict_proba(x_valid)[:, 1]))
                        test_score = float(roc_auc_score(y_test_arr.astype(int), estimator.predict_proba(x_test)[:, 1]))
                        metric = "roc_auc"
                    else:
                        if sklearn_mods is None:
                            issues.append({"model": "lightgbm", "issue": "sklearn metrics unavailable for RMSE"})
                            continue
                        _, _, mean_squared_error, _ = sklearn_mods
                        estimator = LGBMRegressor(
                            n_estimators=500,
                            learning_rate=0.05,
                            num_leaves=31,
                            random_state=seed,
                            n_jobs=-1,
                        )
                        estimator.fit(x_train, y_train_arr)
                        valid_score = regression_rmse(mean_squared_error, y_valid_arr, estimator.predict(x_valid))
                        test_score = regression_rmse(mean_squared_error, y_test_arr, estimator.predict(x_test))
                        metric = "rmse"
                    results.append({
                        "dataset": dataset,
                        "task_type": task_type,
                        "fingerprint": fp_name,
                        "radius": str(radius),
                        "n_bits": str(n_bits),
                        "model": "lightgbm",
                        "seed": str(seed),
                        "metric": metric,
                        "valid_score": str(valid_score),
                        "test_score": str(test_score),
                        "invalid_smiles_count": str(invalid_total),
                        "split_protocol": "fixed_pre_materialized_train_valid_test_csv",
                    })

    summary_rows = summarize_results(results)
    selected_rows = valid_selected_rows(summary_rows)

    if results:
        write_csv(
            output_root / "ecfp_baseline_metrics.csv",
            results,
            [
                "dataset", "task_type", "fingerprint", "radius", "n_bits", "model", "seed",
                "metric", "valid_score", "test_score", "invalid_smiles_count", "split_protocol",
            ],
        )
        write_csv(
            output_root / "ecfp_baseline_summary.csv",
            summary_rows,
            [
                "dataset", "task_type", "fingerprint", "model", "metric", "n_seeds",
                "valid_mean", "valid_std", "test_mean", "test_std", "selection_rule", "split_protocol",
            ],
        )
        write_csv(
            output_root / "ecfp_baseline_valid_selected_summary.csv",
            selected_rows,
            [
                "dataset", "task_type", "selected_model", "selected_fingerprint", "metric",
                "n_seeds", "valid_mean", "valid_std", "test_mean", "test_std",
                "selection_rule", "test_score_reporting_note", "split_protocol",
            ],
        )
        write_summary_markdown(
            output_root / "ecfp_baseline_summary.md",
            summary_rows,
            selected_rows,
            issues,
        )
    if issues:
        write_csv(output_root / "ecfp_baseline_issues.csv", issues, ["model", "issue"])
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, default=Path("data/manifests/split_manifest.csv"))
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--output-root", type=Path, default=Path("reports/ecfp_baselines_20260520"))
    parser.add_argument("--dry-run", action="store_true", help="Validate data/splits/fingerprints without training.")
    parser.add_argument("--dry-run-csv", type=Path, default=Path("reports/ecfp_baseline_dry_run_20260520.csv"))
    parser.add_argument("--dry-run-md", type=Path, default=Path("reports/ecfp_baseline_dry_run_20260520.md"))
    parser.add_argument("--check-deps", action="store_true", help="Write dependency preflight files without training.")
    parser.add_argument("--dependency-json", type=Path, default=Path("reports/ecfp_dependency_preflight_20260520.json"))
    parser.add_argument("--dependency-md", type=Path, default=Path("reports/ecfp_dependency_preflight_20260520.md"))
    parser.add_argument("--sample-size", type=int, default=5)
    parser.add_argument("--rf-seeds", type=str, default=",".join(str(s) for s in DEFAULT_RF_SEEDS))
    parser.add_argument("--run-formal-training", action="store_true", help="Actually train RF/LightGBM baselines. Not used in dry-run phase.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = args.project_root.resolve()
    if args.check_deps:
        dep_json = args.dependency_json if args.dependency_json.is_absolute() else project_root / args.dependency_json
        dep_md = args.dependency_md if args.dependency_md.is_absolute() else project_root / args.dependency_md
        rc = write_dependency_preflight(project_root, dep_json, dep_md)
        print(f"dependency_json={dep_json}")
        print(f"dependency_md={dep_md}")
        return rc

    manifest_path = args.manifest if args.manifest.is_absolute() else project_root / args.manifest
    datasets = [item.strip() for item in args.datasets.replace(",", " ").split() if item.strip()]

    records = load_manifest(manifest_path, project_root)
    _, lightgbm_error = optional_import_lightgbm()
    lightgbm_available = lightgbm_error is None

    if args.dry_run:
        rows = [dry_run_dataset(dataset, records, sample_size=args.sample_size) for dataset in datasets]
        fields = [
            "dataset",
            "task_type",
            "train_rows",
            "valid_rows",
            "test_rows",
            "smiles_column",
            "target_column",
            "manifest_train_rows",
            "manifest_valid_rows",
            "manifest_test_rows",
            "rdkit_parse_sample_ok",
            "ecfp_sample_ok",
            "missing_smiles",
            "missing_target",
            "issue",
            "status",
        ]
        dry_csv = args.dry_run_csv if args.dry_run_csv.is_absolute() else project_root / args.dry_run_csv
        dry_md = args.dry_run_md if args.dry_run_md.is_absolute() else project_root / args.dry_run_md
        write_csv(dry_csv, rows, fields)
        write_dry_run_markdown(dry_md, rows, lightgbm_available=lightgbm_available)
        failed = [row for row in rows if row["status"] != "passed"]
        print(f"dry_run_csv={dry_csv}")
        print(f"dry_run_md={dry_md}")
        print(f"lightgbm_available={str(lightgbm_available).lower()}")
        print(f"dry_run_status={'passed' if not failed else 'failed'}")
        return 0 if not failed else 2

    if not args.run_formal_training:
        raise SystemExit(
            "Refusing to train by default. Use --dry-run for validation or pass "
            "--run-formal-training after explicit authorization."
        )
    rf_seeds = [int(item.strip()) for item in args.rf_seeds.replace(",", " ").split() if item.strip()]
    output_root = args.output_root if args.output_root.is_absolute() else project_root / args.output_root
    return run_training(datasets, records, output_root, rf_seeds)


if __name__ == "__main__":
    raise SystemExit(main())
