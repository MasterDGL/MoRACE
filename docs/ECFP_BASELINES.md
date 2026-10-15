# Morgan/ECFP Baselines

`scripts/run_ecfp_baselines.py` evaluates conventional fingerprint baselines on manifest-listed fixed splits. It is separate from the [paired feature experiment](FEATURE_EXPERIMENTS.md), which uses frozen fingerprints and dataset-specific RF parameters.

## Check Without Training

Use explicit output paths for local dependency and input checks:

```bash
python scripts/run_ecfp_baselines.py --check-deps \
  --dependency-json reports/ecfp_check/dependencies.json \
  --dependency-md reports/ecfp_check/dependencies.md

python scripts/run_ecfp_baselines.py --dry-run \
  --manifest data/manifests/split_manifest.csv \
  --dry-run-csv reports/ecfp_check/splits.csv \
  --dry-run-md reports/ecfp_check/splits.md
```

Dependency checking writes availability/version records; its exit code indicates that the report was written, not that every package is installed. Dry run checks split hashes, row counts, columns and missing values, then tests RDKit parsing and fingerprints on a training sample (`--sample-size 5` by default). It does not validate every molecule or train a model.

## Fit Baselines

Training requires the explicit flag:

```bash
python scripts/run_ecfp_baselines.py --run-formal-training \
  --manifest data/manifests/split_manifest.csv \
  --datasets bbbp --rf-seeds 0,1,2,3,4 \
  --output-root reports/ecfp_bbbp
```

The runner compares radius-2 ECFP4 and radius-3 ECFP6 fingerprints, each with 2,048 bits. RF uses 500 trees, `n_jobs=-1`, and balanced class weights for classification. Optional LightGBM uses 500 estimators, learning rate 0.05 and 31 leaves. This runner does not read RF settings from `configs/model_registry.json`.

Scores are ROC-AUC for classification and RMSE for regression. Model/fingerprint selection uses mean validation scores across the requested estimator seeds, not the best individual seed or test score. The default seed list is 0-4; `--datasets` and `--rf-seeds` take comma-separated values. Inputs resolve against `--project-root`, defaulting to the current directory.

## Outputs and Scope

Training writes `ecfp_baseline_metrics.csv`, `ecfp_baseline_summary.csv`, `ecfp_baseline_valid_selected_summary.csv` and `ecfp_baseline_summary.md` when scores are available. Backend availability issues are written to `ecfp_baseline_issues.csv`; unavailable LightGBM is skipped. A zero exit code alone does not establish that every backend produced scores.

Unlike the newer analysis runners, this legacy entry point permits output overwrites and retains date-stamped defaults. Choose explicit fresh destinations, inspect backend issues, and use the dry run before fitting. Training retains its historical zero-vector treatment of invalid SMILES and records their count; it is not the strict invalid-SMILES policy of the paired feature runner.

These commands generate new baseline results. They do not automatically replay the manuscript's Table I RF+ECFP4 entry; historical comparison requires the recorded settings and completed artifacts. See [paper reproduction](PAPER_REPRODUCTION.md#summarize-and-compare-results) and [data provenance](../data/README.md).
