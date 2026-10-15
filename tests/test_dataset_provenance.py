import json
import shutil
from pathlib import Path

import pytest

from morace.dataset_provenance import (
    DatasetProvenanceError,
    build_dataset_provenance,
    record_dataset_provenance,
    verify_recorded_provenance,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifests" / "split_manifest.csv"


def _copy_splits(tmp_path: Path, dataset: str) -> Path:
    input_root = tmp_path / "data" / "splits"
    shutil.copytree(PROJECT_ROOT / "data" / "splits" / dataset, input_root / dataset)
    return input_root


def test_freesolv_provenance_records_calc_target_and_verified_splits(tmp_path):
    input_root = _copy_splits(tmp_path, "freesolv")

    provenance = build_dataset_provenance(
        "freesolv",
        project_root=tmp_path,
        input_folder=str(input_root),
        manifest_path=MANIFEST_PATH,
    )

    assert provenance["status"] == "valid"
    assert provenance["task_type"] == "regression"
    assert provenance["smiles_column"] == "smiles"
    assert provenance["target_column"] == "calc"
    assert provenance["manifest"]["verification_status"] == "verified"
    assert {name: split["row_count"] for name, split in provenance["splits"].items()} == {
        "train": 513,
        "valid": 64,
        "test": 65,
    }
    assert all(split["manifest_match"] is True for split in provenance["splits"].values())
    assert verify_recorded_provenance(provenance, project_root=tmp_path) == []


def test_bace_provenance_records_mol_and_class_columns(tmp_path):
    input_root = _copy_splits(tmp_path, "bace")
    config_path = tmp_path / "artifacts" / "run" / "run_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"run_tag": "run"}), encoding="utf-8")

    provenance = record_dataset_provenance(
        config_path,
        "bace",
        project_root=tmp_path,
        input_folder=str(input_root),
        manifest_path=MANIFEST_PATH,
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert provenance["task_type"] == "classification"
    assert provenance["smiles_column"] == "mol"
    assert provenance["target_column"] == "Class"
    assert config["schema_version"] == "run_config_v1_1"
    assert config["dataset_provenance"]["bace"] == provenance


def test_manifest_hash_mismatch_fails_and_is_recorded(tmp_path):
    input_root = _copy_splits(tmp_path, "bace")
    train_path = input_root / "bace" / "bace_train.csv"
    train_path.write_bytes(train_path.read_bytes() + b"\n")
    config_path = tmp_path / "artifacts" / "run" / "run_config.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"run_tag": "run"}), encoding="utf-8")

    with pytest.raises(DatasetProvenanceError, match="manifest mismatch for bace train"):
        record_dataset_provenance(
            config_path,
            "bace",
            project_root=tmp_path,
            input_folder=str(input_root),
            manifest_path=MANIFEST_PATH,
        )

    recorded = json.loads(config_path.read_text(encoding="utf-8"))["dataset_provenance"]["bace"]
    assert recorded["status"] == "invalid"
    assert "file_sha256" in recorded["reason"]
