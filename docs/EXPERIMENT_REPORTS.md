# Experiment Reports

Use this entry point after running the paper matrix or frozen-rule feature experiments. It reads actual local artifacts; it does not train models or supply manuscript-table scores as missing baselines.

```bash
python scripts/summarize_experiments.py --run-dir artifacts/run_a --run-dir artifacts/run_b --output-dir reports/main_results
python scripts/summarize_experiments.py --plan reports/public_matrix_plan.json --project-root /path/to/MoRACE --output-dir reports/all_requested_jobs
python scripts/summarize_experiments.py --feature-dir artifacts/features_bbbp_r1 --feature-dir artifacts/features_freesolv_rf50 --output-dir reports/feature_results
```

Generate the optional `--plan` with `scripts/run_paper_experiments.py --plan-output ...`. Its `jobs` array records `run_tag`, `route`, `dataset`, `seed` and `identity` (effective environment, task list and source/input hashes under `identity.payload`). The report checks recorded identity against a supplied planned identity; it does not recompute training or silently accept a different seed. Every planned job is represented, including absent directories. Without a plan, denominators cover the explicitly requested run directories, their summary tasks and any missing `run_config.expected_tasks`; they cannot infer directories the caller omitted.

Every run first goes through `audit_run(require_success=False, allow_legacy=False)`. This requires current `run_summary_v1_2` artifacts, matching outcomes and input provenance. A process failure or invalid audit leaves the score empty and retains the row and reason. Scientific status is independent of score availability: a completed `degraded` or `partial` result with a finite `actual_test_mean` retains that measurement and its status. `null_model_test_mean` and intention-to-treat (`itt_test_mean`) are never substituted for actual results. Because audit validation covers the whole run, a failed run audit excludes every task score in that directory; all rows remain visible.

For successful tasks, final rule resolution uses the shared artifact resolver, including validation-guard fallback to the final top-level inference rules. The main score is the final `all` outcome's actual test mean over its recorded RF seeds. This table reports the existing main rule-only predictor; Morgan-plus-rules feature experiments appear in a separate table.

## Outputs

- `main_runs.csv`: one row per requested/declared task, with route, generation seed (`seed`), RF seeds, actual score, `scored`, original scientific status, outcome category, rule count, final source and artifact hashes. With a matrix plan, `run_tag`, `dataset`, `route` and `seed` always identify the **requested job**; `actual_run_tag`, `actual_dataset`, `actual_subtask`, `actual_route`, `actual_seed` and `actual_metric` record the disk artifacts' identity. Without a plan, the primary identity fields reflect the recorded run. Unavailable RF seed records remain `[]`; they are not inferred from defaults.
- `main_summary.csv`: route/dataset/subtask/metric groups, requested/scored/not-scored counts, status counts, distinct generation seed counts, mean and population standard deviation (`ddof=0`) across available run-level scores. Missing values never enter arithmetic as zeros. A single observed run has std 0, which does not establish stability.
- `feature_runs.csv`: separate per-RF-seed valid/test scores and feature arms. A complete paired grid must agree with its protocol; failed/incomplete feature outputs receive a visible failure row. These results are not merged with main rule-only prediction.
- `totals.json` and `run_audits.json`: all requested denominators, audit reports, plan/comparator hashes and completion status.

Requested matrix routes come from the plan. Otherwise the report uses an explicit recorded route ID, or the recorded model plus Stage 2 profile. It does not derive route identities from a run-tag substring. A missing planned job's known dataset metric is used only to place that missing row in the correct denominator group, never to invent a score. AUC and RMSE are never averaged together.

Conflicting disk identities remain one unscored requested row. For example, if the plan requests seed 43 but the stored run used seed 42, `seed=43`, `actual_seed=42` and `status=artifact_failed`; pairing still uses seed 43. The stored seed-42 run cannot replace the requested key or add a second requested experiment. The same rule applies to a conflicting dataset or route. All audited disk-task details remain in `run_audits.json`.

## Matched External Comparator

The optional comparator must come from **real completed baseline runs under the intended matched protocol**. The tool checks keys, statuses and finite values; a CSV alone cannot prove matching data, compute budgets or training history. Preserve those baseline artifacts separately. A previously published full LLM4SD system is not automatically a same-model/current-protocol baseline.

```bash
python scripts/summarize_experiments.py --plan reports/public_matrix_plan.json --comparator reports/completed_matched_baseline.csv --output-dir reports/paired_results
```

Required comparator columns:

```csv
dataset,route,seed,metric,score,status
```

`route` identifies the main route being compared, not a baseline method name. `seed` is the matched **generation/run seed**, not one of the internal RF seeds. Each `(dataset, route, seed, metric)` must be unique on each side. The comparator accepts finite actual scores from completed success/degraded/partial rows and preserves those statuses. Failed, killed, skipped or missing rows are never given a usable score, even if their CSV contains zero. The caller must explicitly provide baseline results; no source is filled automatically.

`matched_pairs.csv` retains the union of requested main and comparator rows, including unmatched or failed sides. `matched_summary.csv` gives paired and unpaired counts and descriptive mean/std of paired deltas. Positive delta means better main prediction: main−baseline for AUC; baseline−main for RMSE. Unknown metric directions and duplicate pairing keys are rejected. This version supports single-task comparator datasets. Shared RF seeds or repeated generation seeds do not constitute independent datasets; no statistical significance is claimed by these summaries.

Output directories must be new. Reports are still written for incomplete or scientifically non-successful inputs, and the command returns exit code 1 to keep those statuses visible to automation. Exit code 0 means every supplied input has successful status; it does not certify a scientific improvement.
