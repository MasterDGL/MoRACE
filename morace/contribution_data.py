"""Input preparation and rule variants for contribution analysis."""

import csv
from pathlib import Path
from typing import Dict, List, Tuple


INVENTORY_FIELDS = [
    "pilot_id", "route_or_model", "dataset", "run_tag", "seed", "metric",
    "metric_direction", "selected_rule_count", "compiled_rule_count",
    "rules_file", "compiled_rules_file", "outcome_file", "knowledge_type",
    "source_run_dir",
]


def inventory_from_runs(project_root: Path, run_paths: List[Path]) -> List[Dict[str, str]]:
    """Build an inventory from final artifacts recorded by completed runs."""
    from morace.artifact_inputs import resolve_run_rules

    rows = []
    seen = set()
    for run_path in run_paths:
        resolved = (project_root / run_path).resolve()
        run_dir = resolved.parent if resolved.name == "summary.json" else resolved
        for branch in ("synthesize", "inference", "all"):
            for item in resolve_run_rules(run_dir, branch, project_root=project_root):
                dataset = item["dataset"]
                if item["subtask"]:
                    raise ValueError(f"Contribution manifests require one target per dataset; use an explicit inventory and matching manifest for {dataset}/{item['subtask']}")
                outcome = item["outcome"]
                metric = str(outcome.get("metric_name") or "")
                if metric not in {"roc_auc", "rmse"}:
                    raise ValueError(f"Unsupported contribution metric {metric!r}: {item['outcome_path']}")
                pilot_id = f"{item['run_tag']}:{dataset}:{branch}"
                if pilot_id in seen:
                    raise ValueError(f"Duplicate contribution candidate: {pilot_id}")
                seen.add(pilot_id)
                config = item["run_config"]
                seed = outcome.get("generation_seed")
                if seed is None:
                    seed = config.get("global_seed", "")
                rows.append({
                    "pilot_id": pilot_id,
                    "route_or_model": str(config.get("route_id") or config.get("route") or item["model"]),
                    "dataset": dataset, "run_tag": item["run_tag"], "seed": str(seed),
                    "metric": metric, "metric_direction": "higher" if metric == "roc_auc" else "lower",
                    "selected_rule_count": str(outcome.get("selected_rule_count", len(item["metadata"]["rules"]))),
                    "compiled_rule_count": str(outcome.get("compiled_rule_count", "")),
                    "rules_file": str(item["rules_path"]), "compiled_rules_file": str(item["rules_path"]),
                    "outcome_file": str(item["outcome_path"]), "knowledge_type": branch,
                    "source_run_dir": str(run_dir),
                })
    return rows


def filter_inventory(rows: List[Dict[str, str]], *, routes: str = "", datasets: str = "", run_ids: str = "", knowledge_types: str = "") -> List[Dict[str, str]]:
    """Apply exact, comma-separated filters without interpreting run prefixes."""
    filters = {
        "route_or_model": routes, "dataset": datasets, "run_tag": run_ids,
        "knowledge_type": knowledge_types,
    }
    selected = rows
    for field, value in filters.items():
        requested = {item.strip() for item in value.split(",") if item.strip()}
        if requested:
            selected = [row for row in selected if row.get(field, "") in requested]
    return selected


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def is_p0(row: Dict[str, str], *, p0_prefixes: Tuple[str, ...]) -> bool:
    pilot_id = row.get("pilot_id", "")
    if any(pilot_id.startswith(prefix) for prefix in p0_prefixes):
        return "SUPP" not in pilot_id
    return False


def select_scope(
    rows: List[Dict[str, str]],
    scope: str,
    *,
    is_p0_fn,
) -> List[Dict[str, str]]:
    if scope == "all":
        return rows
    if scope == "p0":
        return [row for row in rows if is_p0_fn(row)]
    if scope == "p1":
        return [row for row in rows if not is_p0_fn(row)]
    raise ValueError(f"Unsupported scope: {scope}")


def load_split_manifest(path: Path, *, read_csv_fn) -> Dict[Tuple[str, str], Dict[str, str]]:
    rows = read_csv_fn(path)
    manifest: Dict[Tuple[str, str], Dict[str, str]] = {}
    for row in rows:
        dataset = row.get("dataset", "")
        split = row.get("split_name", "")
        if dataset and split:
            manifest[(dataset, split)] = row
    return manifest


def candidate_paths(project_root: Path, row: Dict[str, str]) -> Dict[str, Path]:
    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else project_root / path

    return {
        "rules_file": resolve(row.get("rules_file", "")),
        "compiled_rules_file": resolve(row.get("compiled_rules_file", "")),
        "outcome_file": resolve(row.get("outcome_file", "")),
    }


def split_paths(
    project_root: Path,
    manifest: Dict[Tuple[str, str], Dict[str, str]],
    dataset: str,
) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for split in ("train", "valid", "test"):
        manifest_row = manifest.get((dataset, split), {})
        raw_path = manifest_row.get("split_file_path", "")
        path = Path(raw_path)
        paths[split] = path if path.is_absolute() else project_root / path
    return paths


def load_smiles_sample(
    split_file: Path,
    smiles_column: str,
    sample_size: int,
) -> List[str]:
    with split_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        sample: List[str] = []
        for row in reader:
            value = str(row.get(smiles_column, "")).strip()
            if value:
                sample.append(value)
            if len(sample) >= sample_size:
                break
    return sample


def load_split_xy_from_manifest(
    split_file: Path,
    smiles_column: str,
    target_column: str,
) -> Tuple[List[str], List[float]]:
    smiles: List[str] = []
    targets: List[float] = []
    with split_file.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            smiles.append(str(row[smiles_column]))
            targets.append(float(row[target_column]))
    return smiles, targets


def sanitize_feature_matrices(
    x_train: List[List[float]],
    x_valid: List[List[float]],
    x_test: List[List[float]],
    feature_names: List[str],
):
    import numpy as np

    if len(feature_names) != len(set(feature_names)):
        duplicates = [
            name
            for name in sorted(set(feature_names))
            if feature_names.count(name) > 1
        ]
        raise RuntimeError(f"Duplicate feature names before sanitization: {duplicates[:10]}")

    train = np.asarray(x_train, dtype=float)
    valid = np.asarray(x_valid, dtype=float)
    test = np.asarray(x_test, dtype=float)
    keep_indices: List[int] = []
    fill_values: List[float] = []
    keep_cols: List[str] = []
    for index, column in enumerate(feature_names):
        series = train[:, index]
        non_nan = series[~np.isnan(series)]
        if non_nan.size == 0:
            continue
        if int(np.unique(np.round(non_nan, 12)).size) <= 1:
            continue
        keep_indices.append(index)
        keep_cols.append(column)
        fill_values.append(float(np.median(non_nan)))
    if not keep_indices:
        raise RuntimeError("No valid compiled features remained after filtering.")

    def select_and_fill(matrix):
        selected = np.asarray(matrix[:, keep_indices], dtype=float)
        for column_index, fill in enumerate(fill_values):
            mask = np.isnan(selected[:, column_index])
            if mask.any():
                selected[mask, column_index] = fill
        return selected

    return (
        select_and_fill(train),
        select_and_fill(valid),
        select_and_fill(test),
        keep_cols,
    )


def canonical_key(rule: Dict) -> str:
    return str(
        rule.get("canonical_key") or rule.get("audit_origin_canonical_key") or ""
    ).strip()


def source_type(rule: Dict) -> str:
    support_sources = {
        str(item) for item in (rule.get("support_sources") or []) if item
    }
    if support_sources == {"literature"}:
        return "literature"
    if support_sources == {"data"}:
        return "data"
    if "literature" in support_sources and "data" in support_sources:
        return "multi"
    return str(rule.get("source", "") or rule.get("primary_source", "") or "")


def drop_rule_once(
    rules: List[Dict],
    target_key: str,
    *,
    canonical_key_fn,
) -> List[Dict]:
    dropped = False
    updated: List[Dict] = []
    for rule in rules:
        if not dropped and canonical_key_fn(rule) == target_key:
            dropped = True
            continue
        updated.append(dict(rule))
    return updated


def family_variants(rules: List[Dict]) -> Dict[str, List[Dict]]:
    variants: Dict[str, List[Dict]] = {}
    for family in ("descriptor", "count", "pattern_presence"):
        if any(rule.get("feature_family") == family for rule in rules):
            variants[family] = [
                dict(rule)
                for rule in rules
                if rule.get("feature_family") != family
            ]
    return variants


def select_pilot_ids(
    rows: List[Dict[str, str]],
    pilot_ids: str,
) -> List[Dict[str, str]]:
    requested = [item.strip() for item in pilot_ids.split(",") if item.strip()]
    if not requested:
        raise ValueError("--pilot-ids is required for --run-loro")
    by_id = {row.get("pilot_id", ""): row for row in rows}
    missing = [pilot_id for pilot_id in requested if pilot_id not in by_id]
    if missing:
        raise ValueError(f"Unknown pilot_id(s): {', '.join(missing)}")
    return [by_id[pilot_id] for pilot_id in requested]
