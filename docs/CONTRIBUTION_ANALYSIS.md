# Rule Contribution Analysis

Contribution diagnostics recompile frozen final rules and reuse fixed split files and recorded RF seeds. They do not call an LLM or generate new rules. The entry points `scripts/run_rule_contribution_analysis.py` and `python -m morace.contribution_analysis` invoke the same implementation.

## Prepare an Inventory

Start with a completed successful run:

```bash
python scripts/run_rule_contribution_analysis.py \
  --run-dir artifacts/my_run \
  --output-dir reports/contribution/my_run --dry-run
```

`--run-dir` can be repeated and also accepts `summary.json`. Automatic inventory resolves all three `synthesize`, `inference` and `all` branches before filtering; the run task and branch outcomes must be successful and their artifact identities must agree. Filtering to `--knowledge-types all` does not bypass an invalid upstream branch. For a usable branch in a degraded run, provide an explicit inventory after checking its outcome and run evidence.

The exported candidate ID is `run_tag:dataset:branch`, stored in the legacy `pilot_id` column. Exact comma-separated filters are `--routes`, `--datasets`, `--run-ids`, `--knowledge-types` and `--pilot-ids`. `route_or_model` may hold the recorded model name when no explicit route ID is stored. An empty selection returns nonzero rather than creating a report.

Alternatively, use `--inventory path/to/inventory.csv`; it is mutually exclusive with `--run-dir`. Required inventory fields are:

```text
pilot_id,route_or_model,dataset,run_tag,seed,metric,metric_direction,selected_rule_count,compiled_rule_count,rules_file,compiled_rules_file,outcome_file
```

`rules_file` contains the final top-level `rules` array, including recorded inference fallback. `compiled_rules_file` is an input-presence reference and may point to the same JSON; analysis always recompiles the final rules. `metric_direction` is `higher` or `lower`; `knowledge_type` is optional for filtering. Relative inputs resolve against `--project-root`, which defaults to the current directory.

Automatic inventory supports single-target datasets. Multitask analysis requires an explicit inventory and a target-matched manifest. The default manifest is `data/manifests/split_manifest.csv`; override it with `--split-manifest`.

## Run Diagnostics

Reuse the dry run's inventory and output directory for different modes:

```bash
python scripts/run_rule_contribution_analysis.py \
  --inventory reports/contribution/my_run/inventory.csv \
  --output-dir reports/contribution/my_run --run-loro

python scripts/run_rule_contribution_analysis.py \
  --inventory reports/contribution/my_run/inventory.csv \
  --output-dir reports/contribution/my_run --run-family-ablation

python scripts/run_rule_contribution_analysis.py \
  --inventory reports/contribution/my_run/inventory.csv \
  --output-dir reports/contribution/my_run --run-feature-importance
```

LORO removes one rule at a time; family ablation removes a feature family; feature importance reports RF impurity importance and, when available, aligns it with this directory's deletion analyses. These diagnose the rule-only representation, not the paired Base+MoRACE arms.

Only one mode may be selected per invocation. `--check-inputs` checks required file presence and can be used alone; `--dry-run` additionally checks rule compilation and sample feature matrices, using five training molecules by default. Neither operation fits an RF or establishes predictive value.

RF seeds come from the outcome; missing legacy seed records fall back to 0-4. Hyperparameters come from `configs/model_registry.json`. Full analyses recompute a base rule-only score and record differences from the stored outcome; `--reproduction-warning-threshold` defaults to an absolute tolerance of 0.02.

## Outputs and Interpretation

Each mode writes `<mode>.csv`, `<mode>_summary.md` and `<mode>_config.json`, plus the shared `inventory.csv`. Prediction modes also write `<mode>_base_reproduction.csv` and `<mode>_issues.csv`; dry run writes `protocol.md`. Existing results for the same mode are not overwritten. Different modes may share a directory only with the same selection and input-file hashes.

The raw deletion delta remains **ablated score minus base score**: negative ROC-AUC or positive RMSE means deletion worsens prediction. This differs from the always-positive-for-improvement convention in [paired feature reports](FEATURE_EXPERIMENTS.md). Validation is the primary diagnostic; test results are supplementary and are not used to select rules.

Errors and empty analyses return nonzero; a reproduction warning can coexist with a completed diagnostic. Read the base-reproduction and issue records alongside the result. Historical `--scope p0/p1` filters remain available; `--existing-ablation` and `--loro-output-prefix` are ignored compatibility options, not new computation settings.
