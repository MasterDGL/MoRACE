# Dataset Inputs

The checkout includes fixed train/validation/test CSV files for **BACE, BBBP, ClinTox, HIV, ESOL, FreeSolv and Lipophilicity**. They are the stored paper inputs, not newly generated splits.

## Files and Endpoints

```text
data/splits/<dataset>/<dataset>_train.csv
data/splits/<dataset>/<dataset>_valid.csv
data/splits/<dataset>/<dataset>_test.csv
```

Endpoint definitions are fixed in [`morace.datasets`](../morace/datasets.py):

- BACE: `mol` SMILES and `Class` target.
- BBBP, ClinTox and HIV: `smiles` with `p_np`, `CT_TOX` and `HIV_active`, respectively.
- ESOL: `smiles` and `ESOL predicted log solubility in mols per litre`.
- FreeSolv: `smiles` and **`calc`**, not `expt`; both columns are present.
- Lipophilicity: `smiles` and `exp`.

The [split manifest](manifests/split_manifest.csv) records counts, columns, CSV hashes, ordered SMILES hashes and ordered SMILES-target hashes. [Manifest notes](manifests/split_manifest.md) retain the original capture and row counts; [compact hashes](manifests/split_hashes.csv) provide a smaller reference. Row order is part of the recorded input identity. The original split-generation algorithm is not established by these files, so they should not be relabeled as confirmed scaffold splits.

Verify all seven datasets and 21 files from the checkout:

```bash
python scripts/verify_repository.py
```

## Alternative Inputs

Data-root selection depends on the entry point:

- Generic launcher: `MORACE_DATASET_ROOT`; `LLM4SD_DATASET_ROOT` and `DATASET_ROOT` remain compatibility aliases.
- Paper runner: `common_environment.MORACE_DATASET_ROOT` in an explicit configuration, not an inherited shell override.
- Paired feature runner: `--data-root`; Python evaluator: `EvaluationConfig.dataset_root`.
- Manifest-based baseline/diagnostic tools: `--manifest` or `--split-manifest`, respectively.

Use the [manifest template](manifests/split_manifest.example.csv) for custom input metadata. The main pipeline also needs a matching `MORACE_DATASET_MANIFEST`; changing a root does not bypass listed-file verification. Manifest-driven tools read their supplied target columns, while the main generator and paired runner use the dataset specifications above.

The dataset loader additionally supports `qm9`, `sider` and `tox21`, but their multitask CSV files are not bundled. They require explicit subtasks and target-matched inputs; the paper matrix and paired feature runner cover only the seven bundled single-task datasets. New inputs or endpoints define a new experiment, not a replay of the fixed paper protocol.
