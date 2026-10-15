# Split Manifest 20260520

## Purpose

This manifest freezes the currently used train/valid/test split evidence for the seven core MoRACE release datasets. It is intended for later Morgan/ECFP + RF/LightGBM strong baselines so those baselines can reuse exactly the same split CSV files rather than creating fresh splits.

This step records existing files only. It does not create new train/valid/test splits.

## Split Type Statement

Current split type should be described conservatively as:

`fixed pre-materialized train/valid/test CSV splits`

Do not claim confirmed scaffold split from this manifest. The directory name is `data/splits`, but the prior split protocol audit did not find original split-generation provenance such as Bemis-Murcko code, RDKit/DeepChem `ScaffoldSplitter`, `RandomSplitter`, or a split seed.

## Manifest Files

- `data/manifests/split_manifest.csv`: per dataset/split file inventory with row counts, columns, file hashes, ordered SMILES hashes, missingness, duplicate SMILES counts, and first/last SMILES previews.
- `data/manifests/split_hashes.csv`: compact hash table for ordered SMILES and ordered SMILES+target checks.

Hash definitions:

- `file_sha256`: SHA256 over the full CSV bytes.
- `ordered_smiles_sha256`: SHA256 over SMILES values in original file order, one SMILES per line.
- `ordered_smiles_target_sha256`: SHA256 over `SMILES<TAB>target` pairs in original file order, one pair per line.

The SMILES are not sorted. Original CSV row order is preserved.

## Row Counts

| dataset | train | valid | test |
|---|---:|---:|---:|
| bace | 1210 | 151 | 152 |
| bbbp | 1631 | 204 | 204 |
| clintox | 1182 | 148 | 148 |
| esol | 902 | 113 | 113 |
| freesolv | 513 | 64 | 65 |
| hiv | 32901 | 4113 | 4113 |
| lipophilicity | 3360 | 420 | 420 |

These row counts match the completed split protocol audit.

## Schema Source

SMILES and target columns follow the staged dataset loading logic in `morace/datasets.py` and the prior split audit notes:

- bace: SMILES column `mol`, target column `Class`
- bbbp: SMILES column `smiles`, target column `p_np`
- clintox: SMILES column `smiles`, target column `CT_TOX`
- esol: SMILES column `smiles`, target column `ESOL predicted log solubility in mols per litre`
- freesolv: SMILES column `smiles`, target column `calc`
- hiv: SMILES column `smiles`, target column `HIV_active`
- lipophilicity: SMILES column `smiles`, target column `exp`

## Missingness And Duplicate Checks

No missing SMILES, missing target, or duplicate SMILES were detected in the 21 split files.

## Issues

No manifest-generation schema/file issues were detected.

## Usage Requirement For Future Baselines

Future Morgan/ECFP + RF/LightGBM baselines should read the split CSVs referenced in this manifest:

- `data/splits/<dataset>/<dataset>_train.csv`
- `data/splits/<dataset>/<dataset>_valid.csv`
- `data/splits/<dataset>/<dataset>_test.csv`

They should not call `train_test_split`, regenerate scaffold splits, shuffle and repartition molecules, or silently replace these files. Any baseline table should cite this manifest and record the `file_sha256` values used.

## Paper Wording

Recommended conservative wording:

"All rule-based and molecular-feature baselines were evaluated on fixed pre-materialized train/validation/test CSV splits. The split files are recorded by SHA256 hashes in the split manifest."

Avoid saying "confirmed scaffold split" unless the original split-generation code and seed are recovered and independently verified.
