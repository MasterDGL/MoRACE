#!/usr/bin/env python
"""Repository integrity checks that do not require model weights or GPUs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import re
import sys
from pathlib import Path
from urllib.parse import unquote


EXPECTED_DATASETS = {
    "bace",
    "bbbp",
    "clintox",
    "esol",
    "freesolv",
    "hiv",
    "lipophilicity",
}
EXPECTED_SPLITS = {"train", "valid", "test"}
MANIFEST_PATH = Path("data/manifests/split_manifest.csv")
LOCAL_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def check_required_files(root: Path, failures: list[str]) -> None:
    required = [
        "README.md",
        "LICENSE",
        "CONTRIBUTING.md",
        "pyproject.toml",
        "requirements.txt",
        "requirements-dev.txt",
        "requirements-model.txt",
        "requirements-model-gemma.txt",
        ".github/workflows/ci.yml",
        "morace/__init__.py",
        "morace/contribution_analysis.py",
        "morace/contribution_audit.py",
        "morace/contribution_compiled_evaluation.py",
        "morace/contribution_data.py",
        "morace/contribution_evaluation.py",
        "morace/contribution_metrics.py",
        "morace/contribution_orchestration.py",
        "morace/contribution_reporting.py",
        "morace/dataset_provenance.py",
        "morace/model_canary.py",
        "morace/run_audit.py",
        "morace/run_summary.py",
        "morace/run_configuration.py",
        "morace/artifact_inputs.py",
        "morace/feature_experiments.py",
        "morace/experiment_reports.py",
        "morace/raw_closure.py",
        "scripts/run_pipeline.sh",
        "scripts/run_paper_experiments.py",
        "scripts/setup_environment.sh",
        "scripts/run_ecfp_baselines.py",
        "scripts/run_rule_contribution_analysis.py",
        "scripts/run_feature_experiments.py",
        "scripts/run_raw_closure.py",
        "scripts/summarize_experiments.py",
        "examples/compile_rules.py",
        "examples/rules.json",
        "docs/REPRODUCIBILITY.md",
        "docs/RELEASE_SCOPE.md",
        "configs/paper_experiment.json",
        "configs/feature_experiments.json",
        "configs/checkpoint_manifest.json",
        str(MANIFEST_PATH),
    ]
    for relative in required:
        if not (root / relative).exists():
            failures.append(f"missing required file: {relative}")


def check_split_manifest(root: Path, failures: list[str]) -> None:
    manifest = root / MANIFEST_PATH
    if not manifest.is_file():
        failures.append(f"missing split manifest: {MANIFEST_PATH}")
        return

    _, manifest_rows = read_csv(manifest)
    seen: set[tuple[str, str]] = set()
    manifest_files: set[Path] = set()

    for index, row in enumerate(manifest_rows, start=2):
        dataset = row.get("dataset", "")
        split_name = row.get("split_name", "")
        split_file_path = row.get("split_file_path", "")
        seen.add((dataset, split_name))

        if dataset not in EXPECTED_DATASETS:
            failures.append(f"{MANIFEST_PATH}:{index}: unexpected dataset {dataset!r}")
        if split_name not in EXPECTED_SPLITS:
            failures.append(f"{MANIFEST_PATH}:{index}: unexpected split {split_name!r}")
        if not split_file_path:
            failures.append(f"{MANIFEST_PATH}:{index}: missing split_file_path")
            continue

        split_path = root / split_file_path
        manifest_files.add(Path(split_file_path))
        if not split_path.is_file():
            failures.append(f"{MANIFEST_PATH}:{index}: missing split file {split_file_path}")
            continue

        if sha256_bytes(split_path) != row.get("file_sha256"):
            failures.append(f"{MANIFEST_PATH}:{index}: file_sha256 mismatch for {split_file_path}")

        fieldnames, split_rows = read_csv(split_path)
        expected_row_count = int(row.get("row_count") or -1)
        expected_column_count = int(row.get("column_count") or -1)
        if len(split_rows) != expected_row_count:
            failures.append(
                f"{MANIFEST_PATH}:{index}: row_count mismatch for {split_file_path} "
                f"({len(split_rows)} != {expected_row_count})"
            )
        if len(fieldnames) != expected_column_count:
            failures.append(
                f"{MANIFEST_PATH}:{index}: column_count mismatch for {split_file_path} "
                f"({len(fieldnames)} != {expected_column_count})"
            )
        if "|".join(fieldnames) != row.get("columns"):
            failures.append(f"{MANIFEST_PATH}:{index}: columns mismatch for {split_file_path}")

        smiles_column = row.get("smiles_column", "")
        target_column = row.get("target_column", "")
        if smiles_column not in fieldnames:
            failures.append(f"{MANIFEST_PATH}:{index}: missing SMILES column {smiles_column!r}")
            continue
        if target_column not in fieldnames:
            failures.append(f"{MANIFEST_PATH}:{index}: missing target column {target_column!r}")
            continue

        smiles = [item.get(smiles_column, "") for item in split_rows]
        targets = [item.get(target_column, "") for item in split_rows]
        smiles_targets = [f"{smiles_value}\t{target_value}" for smiles_value, target_value in zip(smiles, targets)]

        if sha256_lines(smiles) != row.get("ordered_smiles_sha256"):
            failures.append(f"{MANIFEST_PATH}:{index}: ordered_smiles_sha256 mismatch for {split_file_path}")
        if sha256_lines(smiles_targets) != row.get("ordered_smiles_target_sha256"):
            failures.append(f"{MANIFEST_PATH}:{index}: ordered_smiles_target_sha256 mismatch for {split_file_path}")

        has_missing_smiles = str(any(value == "" for value in smiles)).lower()
        has_missing_target = str(any(value == "" for value in targets)).lower()
        duplicate_smiles_count = str(len(smiles) - len(set(smiles)))
        if has_missing_smiles != row.get("has_missing_smiles"):
            failures.append(f"{MANIFEST_PATH}:{index}: has_missing_smiles mismatch for {split_file_path}")
        if has_missing_target != row.get("has_missing_target"):
            failures.append(f"{MANIFEST_PATH}:{index}: has_missing_target mismatch for {split_file_path}")
        if duplicate_smiles_count != row.get("duplicate_smiles_count"):
            failures.append(f"{MANIFEST_PATH}:{index}: duplicate_smiles_count mismatch for {split_file_path}")

    expected = {(dataset, split) for dataset in EXPECTED_DATASETS for split in EXPECTED_SPLITS}
    missing_manifest_rows = expected - seen
    extra_manifest_rows = seen - expected
    if missing_manifest_rows:
        failures.append(f"missing manifest rows: {sorted(missing_manifest_rows)}")
    if extra_manifest_rows:
        failures.append(f"extra manifest rows: {sorted(extra_manifest_rows)}")

    actual_split_files = {
        path.relative_to(root)
        for path in (root / "data/splits").glob("*/*.csv")
        if path.is_file()
    }
    extra_split_files = actual_split_files - manifest_files
    missing_split_files = manifest_files - actual_split_files
    if extra_split_files:
        failures.append(f"split CSV files not listed in manifest: {sorted(map(str, extra_split_files))}")
    if missing_split_files:
        failures.append(f"manifest files missing from data/splits: {sorted(map(str, missing_split_files))}")


def iter_markdown_files(root: Path) -> list[Path]:
    files = list(root.glob("*.md"))
    for folder in ["docs", "data"]:
        base = root / folder
        if base.exists():
            files.extend(base.rglob("*.md"))
    reports_readme = root / "reports" / "README.md"
    if reports_readme.exists():
        files.append(reports_readme)
    return sorted(set(files))


def check_markdown_links(root: Path, failures: list[str]) -> None:
    for markdown_path in iter_markdown_files(root):
        text = markdown_path.read_text(encoding="utf-8")
        for raw_target in LOCAL_LINK_RE.findall(text):
            target = raw_target.strip().strip("<>")
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            if "://" in target:
                continue
            path_part = unquote(target.split("#", 1)[0])
            if not path_part:
                continue
            resolved = (markdown_path.parent / path_part).resolve()
            try:
                resolved.relative_to(root)
            except ValueError:
                failures.append(f"{markdown_path.relative_to(root)}: link escapes repository: {target}")
                continue
            if not resolved.exists():
                failures.append(f"{markdown_path.relative_to(root)}: broken local link: {target}")


def check_runtime_imports(failures: list[str]) -> None:
    modules = (
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "rdkit",
        "pydantic",
    )
    for name in modules:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            failures.append(f"runtime import failed for {name}: {type(exc).__name__}: {exc}")
            continue
        version = getattr(module, "__version__", "unknown")
        print(f"- runtime {name}: {version}")

    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        print(f"- cuda available: {torch.cuda.is_available()}")
        print(f"- cuda device count: {torch.cuda.device_count()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="also import the selected model runtime stack and report CUDA visibility",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    check_required_files(root, failures)
    check_split_manifest(root, failures)
    check_markdown_links(root, failures)
    if args.runtime:
        check_runtime_imports(failures)

    if failures:
        print("Repository integrity check failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Repository integrity check passed.")
    print(f"- datasets: {len(EXPECTED_DATASETS)}")
    print(f"- split files: {len(EXPECTED_DATASETS) * len(EXPECTED_SPLITS)}")
    print(f"- markdown files checked: {len(iter_markdown_files(root))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
