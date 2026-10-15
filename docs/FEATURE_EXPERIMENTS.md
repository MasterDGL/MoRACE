# Paired Feature Experiments

This runner measures the contribution of one frozen, final rule artifact to molecular prediction. It does not call an LLM, regenerate rules, change the main rule-only pipeline, or select a winning arm using test labels. It supports the seven bundled single-task datasets.

```bash
python scripts/run_feature_experiments.py --dataset bbbp --source artifacts/my_run --output-dir artifacts/features_bbbp_r1
python scripts/run_feature_experiments.py --dataset freesolv --source /path/to/final_rules.json --output-dir artifacts/features_freesolv_rf50 --seed-profile fifty
python scripts/run_feature_experiments.py --dataset bbbp --source /path/to/final_rules.json --output-dir artifacts/features_custom --rf-seeds 0 7 19 --control-seeds 42 43
```

The source may be a run directory, summary, outcome, or rule JSON. A current run-directory/`summary.json` input requires a successful task and resolves its final `all` outcome to metadata. For a usable final branch in a degraded run, supply its successful outcome or final rule JSON explicitly; retain the original run audit because an explicit file alone does not certify its training-data provenance.

For a validation fallback, `final_result_source=inference_only` means that the metadata's **top-level `rules`** contain the final inference rules. `candidate_all_result.rules` are never evaluated as final rules. Explicit bare rule lists are accepted and recorded as lacking final-branch verification. Failed or degraded outcome files are rejected. Supply `--project-root` if moved artifacts have repository-relative links in a nonstandard layout.

Five arms use exactly the same rows, targets, RF hyperparameters, and estimator seeds:

| Arm | Feature columns |
| --- | --- |
| `base` | Morgan bits |
| `rules` | Train-sanitized final rule columns |
| `base_plus_rules` | The unchanged Morgan prefix plus rule columns |
| `base_plus_shuffled_rules` | Same prefix; each rule column shuffled separately inside each split |
| `base_plus_random_rules` | Same prefix; Gaussian columns with matching count and train-derived column means/std |

## Protocol and Historical Compatibility

The sanitizer is migrated from `run_p7rf_e_three_route_full7.py`: duplicate feature names keep the first column; train-only nonfinite/constant columns are dropped; missing values are filled with training medians; matrices are float32. This is intentionally independent of the main pipeline's sanitizer. The split-local controls are migrated from P13's `p13_control_helpers.py`: shuffle RNG offsets are `split_index * 1009`, Gaussian offsets are `split_index * 2003`, and Gaussian train standard deviations use `ddof=0` with values at most `1e-12` replaced by 1.

`configs/feature_experiments.json` freezes the historical validation-selected Morgan settings: radius 3 for FreeSolv/HIV and radius 2 for the other five datasets, all 2048 bits, RDKit's historical default chirality setting. No fingerprint selection is rerun. RF parameters come from `configs/model_registry.json`, with only `random_state` overridden for each paired seed. Classification applies a separate train-fit `StandardScaler` to each arm and reports ROC-AUC; regression is unscaled and reports RMSE. The legacy targets are retained, including FreeSolv `calc` and ESOL `ESOL predicted log solubility in mols per litre`.

P13 originally copied the base and base+rules scores from P7rf-e and fitted only its two controls. This runner freshly fits all five arms in one environment and adds the rules-only arm. Its default control seeds `[42, 43]` are explicit, reproducible new noise realizations. To reproduce a particular historical P13 cell's noise, supply its exact route/repeat labels:

```bash
python scripts/run_feature_experiments.py --dataset bbbp --source /path/to/historical_final_rules.json --output-dir artifacts/llama_bbbp_r1_controls --historical-control-key llama r1
```

This uses the original SHA256-derived seed key `P13-shuffled-random-rule-feature-control|route|dataset|repeat|control_type`, where control types are `shuffled_rules` and `random_features`. Matching a historical score additionally requires identical source rules, splits, selected fingerprint, RF parameters, and dependency versions. A historical route label must match the archived table exactly (for example `qwen_legacy`, not an inferred model abbreviation).

The `five` profile uses RF seeds 0–4. `fifty` uses 0–49 with **unchanged rules, data and control matrices**; it measures estimator-seed sensitivity, not fifty independent rule generations or data splits. No mixed-unit aggregate across AUC and RMSE is produced, and RF seeds are not treated as independent dataset replications.

## Outputs and Validation

Each output directory must be new. `per_seed.csv` contains valid/test scores and directional deltas against base for every arm and paired RF seed. `summary.csv` contains mean and population standard deviation (`ddof=0`). `paired_deltas.csv` compares each nonbase arm with base and additionally compares base+rules with both nuisance controls. A positive delta always means improvement: candidate−comparator for AUC, comparator−candidate for RMSE. These are descriptive paired summaries, not a significance claim.

`protocol.json` records actual RF seeds/parameters, fingerprint settings, both control seeds, endpoint, ordered split row counts/SMILES hashes and CSV hashes, linked source artifact hashes, final branch, sanitization, matrix shapes/float32 byte hashes, configuration/registry/code hashes, and dependency versions. `resolved_rules.json` preserves the actual rules. `status.json` reports success or failure; a failed fit is never given zero scores or reported as successful.

When source-run dataset provenance is available, every current split hash and target/SMILES column must match it. Explicit rule files without recorded data provenance still get complete current data hashes, but do not certify which data generated those rules. Choose a source artifact whose training/selection history matches the intended experiment. Rows are never silently dropped: missing/nonfinite labels, invalid SMILES, single-class AUC splits, uncompiled rules and no surviving rule features fail before RF fitting. The historical P7rf-e zero-vector treatment of invalid SMILES is therefore **not** retained; bundled valid data retain the historical numerical path.

`--config`, `--registry` and `--data-root` can point at explicit alternative files for small tests or new protocols. Every override is recorded. Overrides are new experiments, not automatically historical-score reproductions. Paths to bundled defaults are resolved from the source checkout, so the script also runs from another working directory.
