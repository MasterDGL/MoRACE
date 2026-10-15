from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from morace.datasets import canonical_dataset_and_subtask, get_dataset_spec


SPLIT_NAMES = ("train", "valid", "test")
PROVENANCE_SCHEMA_VERSION = "dataset_provenance_v1"
RUN_CONFIG_SCHEMA_VERSION = "run_config_v1_1"


class DatasetProvenanceError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def resolve_recorded_path(value: str, project_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.is_file():
        raise DatasetProvenanceError(f"dataset manifest does not exist: {manifest_path}")
    _, rows = _read_csv(manifest_path)
    return rows


def _manifest_int(row: dict[str, str], key: str, split_name: str) -> int:
    try:
        return int(row.get(key, ""))
    except (TypeError, ValueError) as exc:
        raise DatasetProvenanceError(
            f"invalid {key} for manifest split {split_name}: {row.get(key)!r}"
        ) from exc


def build_dataset_provenance(
    dataset: str,
    subtask: str = "",
    *,
    project_root: Path | None = None,
    input_folder: str = "data/splits",
    manifest_path: Path | str = Path("data/manifests/split_manifest.csv"),
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    dataset, subtask = canonical_dataset_and_subtask(dataset, subtask)
    spec = get_dataset_spec(dataset, subtask, input_folder=input_folder)
    manifest = Path(manifest_path)
    if not manifest.is_absolute():
        manifest = root / manifest
    manifest = manifest.resolve()
    rows = _manifest_rows(manifest)
    dataset_rows = [row for row in rows if row.get("dataset") == dataset]
    manifest_by_split = {row.get("split_name", ""): row for row in dataset_rows}
    if dataset_rows and set(manifest_by_split) != set(SPLIT_NAMES):
        raise DatasetProvenanceError(
            f"manifest rows for {dataset} must contain exactly {', '.join(SPLIT_NAMES)}"
        )

    folder = Path(spec.folder)
    if not folder.is_absolute():
        folder = root / folder
    split_records: dict[str, dict[str, Any]] = {}
    for split_name in SPLIT_NAMES:
        split_path = (folder / f"{spec.file_stem}_{split_name}.csv").resolve()
        if not split_path.is_file():
            raise DatasetProvenanceError(f"missing {dataset} {split_name} split: {split_path}")
        columns, split_rows = _read_csv(split_path)
        if spec.smiles_column not in columns:
            raise DatasetProvenanceError(
                f"missing SMILES column {spec.smiles_column!r} in {split_path}"
            )
        if spec.target_column not in columns:
            raise DatasetProvenanceError(
                f"missing target column {spec.target_column!r} in {split_path}"
            )
        smiles = [row.get(spec.smiles_column, "") for row in split_rows]
        targets = [row.get(spec.target_column, "") for row in split_rows]
        smiles_targets = [
            f"{smiles_value}\t{target_value}"
            for smiles_value, target_value in zip(smiles, targets)
        ]
        record = {
            "file_path": _portable_path(split_path, root),
            "row_count": len(split_rows),
            "columns": columns,
            "file_sha256": sha256_file(split_path),
            "ordered_smiles_sha256": sha256_lines(smiles),
            "ordered_smiles_target_sha256": sha256_lines(smiles_targets),
        }
        expected = manifest_by_split.get(split_name)
        if expected is not None:
            checks = {
                "row_count": (record["row_count"], _manifest_int(expected, "row_count", split_name)),
                "columns": ("|".join(columns), expected.get("columns", "")),
                "smiles_column": (spec.smiles_column, expected.get("smiles_column", "")),
                "target_column": (spec.target_column, expected.get("target_column", "")),
                "file_sha256": (record["file_sha256"], expected.get("file_sha256", "")),
                "ordered_smiles_sha256": (
                    record["ordered_smiles_sha256"],
                    expected.get("ordered_smiles_sha256", ""),
                ),
                "ordered_smiles_target_sha256": (
                    record["ordered_smiles_target_sha256"],
                    expected.get("ordered_smiles_target_sha256", ""),
                ),
            }
            mismatches = [
                f"{key} {actual!r} != {wanted!r}"
                for key, (actual, wanted) in checks.items()
                if actual != wanted
            ]
            if mismatches:
                raise DatasetProvenanceError(
                    f"manifest mismatch for {dataset} {split_name}: {'; '.join(mismatches)}"
                )
            record["manifest_file_path"] = expected.get("split_file_path", "")
            record["manifest_match"] = True
        else:
            record["manifest_file_path"] = None
            record["manifest_match"] = None
        split_records[split_name] = record

    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "status": "valid",
        "reason": "",
        "dataset": dataset,
        "subtask": subtask,
        "task_label": f"{dataset}:{subtask}" if subtask else dataset,
        "task_type": spec.task_type,
        "smiles_column": spec.smiles_column,
        "target_column": spec.target_column,
        "dataset_root": _portable_path(folder, root),
        "manifest": {
            "path": _portable_path(manifest, root),
            "sha256": sha256_file(manifest),
            "verification_status": "verified" if dataset_rows else "unlisted",
        },
        "splits": split_records,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def record_dataset_provenance(
    run_config_path: Path | str,
    dataset: str,
    subtask: str = "",
    *,
    project_root: Path | None = None,
    input_folder: str = "data/splits",
    manifest_path: Path | str = Path("data/manifests/split_manifest.csv"),
) -> dict[str, Any]:
    root = Path(project_root or Path.cwd()).resolve()
    config_path = Path(run_config_path)
    if not config_path.is_absolute():
        config_path = root / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise DatasetProvenanceError(f"run config must be a JSON object: {config_path}")

    canonical_dataset, canonical_subtask = canonical_dataset_and_subtask(dataset, subtask)
    task_label = (
        f"{canonical_dataset}:{canonical_subtask}"
        if canonical_subtask
        else canonical_dataset
    )
    try:
        provenance = build_dataset_provenance(
            canonical_dataset,
            canonical_subtask,
            project_root=root,
            input_folder=input_folder,
            manifest_path=manifest_path,
        )
    except (DatasetProvenanceError, OSError, csv.Error) as exc:
        provenance = {
            "schema_version": PROVENANCE_SCHEMA_VERSION,
            "status": "invalid",
            "reason": str(exc),
            "dataset": canonical_dataset,
            "subtask": canonical_subtask,
            "task_label": task_label,
        }
        config["schema_version"] = RUN_CONFIG_SCHEMA_VERSION
        config.setdefault("dataset_provenance", {})[task_label] = provenance
        _write_json_atomic(config_path, config)
        raise DatasetProvenanceError(str(exc)) from exc

    config["schema_version"] = RUN_CONFIG_SCHEMA_VERSION
    config["dataset_manifest"] = provenance["manifest"]
    config.setdefault("dataset_provenance", {})[task_label] = provenance
    _write_json_atomic(config_path, config)
    return provenance


def verify_recorded_provenance(
    provenance: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> list[str]:
    root = Path(project_root or Path.cwd()).resolve()
    errors: list[str] = []
    label = str(provenance.get("task_label") or provenance.get("dataset") or "task")
    if provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION:
        errors.append(f"{label}: unsupported provenance schema {provenance.get('schema_version')!r}")
    if provenance.get("status") != "valid":
        errors.append(f"{label}: provenance status is {provenance.get('status')!r}: {provenance.get('reason', '')}")
        return errors

    manifest = provenance.get("manifest") or {}
    manifest_value = str(manifest.get("path") or "")
    if not manifest_value:
        errors.append(f"{label}: manifest path is missing")
    else:
        manifest_path = resolve_recorded_path(manifest_value, root)
        if not manifest_path.is_file():
            errors.append(f"{label}: manifest is missing: {manifest_path}")
        elif sha256_file(manifest_path) != manifest.get("sha256"):
            errors.append(f"{label}: manifest SHA256 mismatch: {manifest_path}")

    smiles_column = str(provenance.get("smiles_column") or "")
    target_column = str(provenance.get("target_column") or "")
    splits = provenance.get("splits") or {}
    for split_name in SPLIT_NAMES:
        record = splits.get(split_name)
        if not isinstance(record, dict):
            errors.append(f"{label}: missing recorded {split_name} split")
            continue
        file_value = str(record.get("file_path") or "")
        split_path = resolve_recorded_path(file_value, root) if file_value else Path()
        if not file_value or not split_path.is_file():
            errors.append(f"{label}: {split_name} split is missing: {file_value!r}")
            continue
        try:
            columns, rows = _read_csv(split_path)
        except (OSError, csv.Error) as exc:
            errors.append(f"{label}: cannot read {split_name} split {split_path}: {exc}")
            continue
        smiles = [row.get(smiles_column, "") for row in rows]
        targets = [row.get(target_column, "") for row in rows]
        smiles_targets = [
            f"{smiles_value}\t{target_value}"
            for smiles_value, target_value in zip(smiles, targets)
        ]
        checks = {
            "row_count": (len(rows), record.get("row_count")),
            "columns": (columns, record.get("columns")),
            "file_sha256": (sha256_file(split_path), record.get("file_sha256")),
            "ordered_smiles_sha256": (
                sha256_lines(smiles),
                record.get("ordered_smiles_sha256"),
            ),
            "ordered_smiles_target_sha256": (
                sha256_lines(smiles_targets),
                record.get("ordered_smiles_target_sha256"),
            ),
        }
        for key, (actual, expected) in checks.items():
            if actual != expected:
                errors.append(
                    f"{label}: {split_name} {key} mismatch: {actual!r} != {expected!r}"
                )
        if smiles_column not in columns:
            errors.append(f"{label}: {split_name} missing SMILES column {smiles_column!r}")
        if target_column not in columns:
            errors.append(f"{label}: {split_name} missing target column {target_column!r}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and record the dataset inputs used by one MoRACE task."
    )
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--subtask", default="")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--input-folder", default="data/splits")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/split_manifest.csv"),
    )
    args = parser.parse_args()
    try:
        provenance = record_dataset_provenance(
            args.run_config,
            args.dataset,
            args.subtask,
            project_root=args.project_root,
            input_folder=args.input_folder,
            manifest_path=args.manifest,
        )
    except (DatasetProvenanceError, OSError, json.JSONDecodeError) as exc:
        print(f"Dataset provenance failed: {exc}")
        return 1
    print(
        "Dataset provenance verified: "
        f"task={provenance['task_label']} target={provenance['target_column']} "
        f"manifest={provenance['manifest']['verification_status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
