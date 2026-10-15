import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd


QM9_SUBTASKS = [
    "alpha",
    "c_v",
    "Delta_epsilon",
    "epsilon_HOMO",
    "epsilon_LUMO",
    "G",
    "H",
    "mu",
    "R^2",
    "U_0",
    "U",
    "ZPVE",
]

CLASSIFICATION_DATASETS = {"bbbp", "bace", "clintox", "hiv", "tox21", "sider"}
REGRESSION_DATASETS = {"esol", "freesolv", "lipophilicity", "qm9"}


def resolve_input_folder(input_folder: str = "data/splits", *, inherit_environment: bool = True) -> str:
    folder = (input_folder or "").strip() or "data/splits"
    if folder != "data/splits" or not inherit_environment:
        return folder
    env_folder = (
        os.environ.get("MORACE_DATASET_ROOT")
        or os.environ.get("LLM4SD_DATASET_ROOT")
        or os.environ.get("DATASET_ROOT")
    )
    if env_folder:
        return env_folder
    return folder


@dataclass(frozen=True)
class DatasetSpec:
    dataset: str
    subtask: str
    folder: str
    file_stem: str
    target_column: str
    smiles_column: str
    task_type: str


def canonical_dataset_and_subtask(dataset: str, subtask: str) -> Tuple[str, str]:
    dataset = dataset.strip()
    subtask = subtask.strip()
    if dataset in QM9_SUBTASKS:
        return "qm9", dataset
    return dataset, subtask


def get_dataset_spec(
    dataset: str, subtask: str = "", input_folder: str = "data/splits", *, inherit_environment: bool = True
) -> DatasetSpec:
    dataset, subtask = canonical_dataset_and_subtask(dataset, subtask)
    input_folder = resolve_input_folder(input_folder, inherit_environment=inherit_environment)
    if dataset == "qm9":
        if not subtask:
            raise ValueError("QM9 requires a subtask.")
        folder = os.path.join(input_folder, "qm9")
        file_stem = subtask
        target_column = subtask
        smiles_column = "smiles"
        task_type = "regression"
    elif dataset == "bbbp":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "p_np"
        smiles_column = "smiles"
        task_type = "classification"
    elif dataset == "clintox":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "CT_TOX"
        smiles_column = "smiles"
        task_type = "classification"
    elif dataset == "hiv":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "HIV_active"
        smiles_column = "smiles"
        task_type = "classification"
    elif dataset == "bace":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "Class"
        smiles_column = "mol"
        task_type = "classification"
    elif dataset == "esol":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "ESOL predicted log solubility in mols per litre"
        smiles_column = "smiles"
        task_type = "regression"
    elif dataset == "freesolv":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "calc"
        smiles_column = "smiles"
        task_type = "regression"
    elif dataset == "lipophilicity":
        folder = os.path.join(input_folder, dataset)
        file_stem = dataset
        target_column = "exp"
        smiles_column = "smiles"
        task_type = "regression"
    elif dataset in {"tox21", "sider"}:
        if not subtask:
            raise ValueError(f"{dataset} requires a subtask.")
        folder = os.path.join(input_folder, dataset)
        file_stem = subtask
        target_column = subtask
        smiles_column = "smiles"
        task_type = "classification"
    else:
        raise NotImplementedError(f"Unsupported dataset: {dataset}")

    return DatasetSpec(
        dataset=dataset,
        subtask=subtask,
        folder=folder,
        file_stem=file_stem,
        target_column=target_column,
        smiles_column=smiles_column,
        task_type=task_type,
    )


def load_split_dataframe(
    dataset: str,
    subtask: str = "",
    split: str = "train",
    input_folder: str = "data/splits",
    *,
    inherit_environment: bool = True,
) -> pd.DataFrame:
    spec = get_dataset_spec(dataset, subtask, input_folder=input_folder, inherit_environment=inherit_environment)
    filename = f"{spec.file_stem}_{split}.csv"
    path = os.path.join(spec.folder, filename)
    return pd.read_csv(path)


def load_split_xy(
    dataset: str,
    subtask: str = "",
    split: str = "train",
    input_folder: str = "data/splits",
    *,
    inherit_environment: bool = True,
) -> Tuple[List[str], List[float], DatasetSpec]:
    spec = get_dataset_spec(dataset, subtask, input_folder=input_folder, inherit_environment=inherit_environment)
    df = load_split_dataframe(dataset, subtask, split=split, input_folder=input_folder,
                              inherit_environment=inherit_environment)
    smiles = df[spec.smiles_column].astype(str).tolist()
    y = df[spec.target_column].tolist()
    return smiles, y, spec


def format_row_for_prompt(dataset: str, smiles: str, target_value) -> str:
    dataset, _ = canonical_dataset_and_subtask(dataset, "")
    if dataset in CLASSIFICATION_DATASETS:
        return f"SMILES: {smiles} | label: {target_value}"
    return f"SMILES: {smiles} | value: {target_value}"


def load_model_hparams(path: str = "configs/model_registry.json") -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
