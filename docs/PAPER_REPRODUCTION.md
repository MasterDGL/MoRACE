# Paper Reproduction

This guide maps the manuscript **MoRACE: Molecular Rule-to-Feature VAlidation and Compilation FramEwork for Molecular Property Prediction** to the available experiment entry points. Run commands from a source checkout; model paths and external artifact paths refer to files supplied by the user.

See the [results summary](PAPER_RESULTS.md) for a brief qualitative overview. The commands below operate on supplied inputs and actual completed outputs.

## Manuscript-to-Code Map

The manuscript separates feature construction, matched model comparisons and paired RF/control evidence. The implementation preserves this separation.

| Evidence | Implementation | Required input |
| --- | --- | --- |
| Draft generation (Fig. 2A) | `scripts/run_paper_experiments.py` | Local checkpoint, route-specific runtime and fixed splits |
| Raw-draft projection, rejection and executable closure (Fig. 2B-C; Fig. 3) | `scripts/run_raw_closure.py`, `morace/raw_closure.py`, `morace/rule_compiler.py` | Saved candidate records and the exact molecule set |
| Base/Base+MoRACE and count-matched controls (Fig. 2D; Fig. 6) | `scripts/run_feature_experiments.py` | Final selected rules, fixed splits and paired RF settings |
| Source-bound small-model comparison (Fig. 4) | `scripts/summarize_experiments.py --comparator ...` | Completed MoRACE and matched comparator records |
| Wider raw-input baseline context (Table I) | ECFP baseline entry point plus separately retained baseline implementations | Matched baseline source, inputs, settings and completed results |

Two distinctions are essential when selecting an entry point:

- The generation pipeline produces `synthesize`, `inference` and combined `all` **rule-only** predictions. The paired feature runner produces **Base** and **Base+MoRACE** predictions; it does not replace or alter the generation pipeline.
- The frozen raw-draft projector is an independent operational-closure analysis. Generation uses model-route-specific Stage-2 projection; the two mechanisms must not be described as the same end-to-end projector.

## Requirements

The core package and CPU analyses support Python **3.10/3.11**. The seven fixed split sets are bundled; confirm them with `python scripts/verify_repository.py`.

Model generation runs on Linux with Bash, a compatible GPU and local checkpoint files. Use [requirements-model.txt](../requirements-model.txt) in the model environment for Llama, Qwen, DeepSeek safetensors and ChemDFM; Gemma uses [requirements-model-gemma.txt](../requirements-model-gemma.txt). `--python-path ROUTE_ID=PATH` selects the interpreter for each route without changing the caller's environment.

The [checkpoint manifest](../configs/checkpoint_manifest.json) records the checkpoint identities used by the configured protocol. Execution preflight verifies the selected checkpoint and runtime. Weights are not downloaded by the experiment runner or redistributed in this repository.

| Model | Route ID |
| --- | --- |
| Qwen3.5-9B | `qwen_legacy` |
| Llama-3.1-8B-Instruct | `llama31_native_clean` |
| Gemma-4-E4B-it | `gemma_native_clean` |
| DeepSeek-LLM-7B-Chat | `deepseek_d52` |
| ChemDFM-v1.5-8B | `chemdfm_b3` |

## Generate and Audit Rules

The [paper configuration](../configs/paper_experiment.json) defines five routes, seven datasets and generation seeds **42/43/44**, giving **105 runs**. Preview this matrix without model loading or job execution:

```bash
python scripts/run_paper_experiments.py
```

Resolve and preflight a single Llama/BBBP run after supplying actual local paths:

```bash
python scripts/run_paper_experiments.py \
  --route llama31_native_clean --dataset bbbp --seed 42 \
  --model-path llama31_native_clean=/models/Meta-Llama-3.1-8B-Instruct \
  --python-path llama31_native_clean=/envs/llama/bin/python \
  --plan-output reports/llama_bbbp_plan.json --preflight-only
```

Replace `--preflight-only` with `--execute` to launch this run. Remove it to save the resolved plan without execution preflight. Saving a plan probes the selected interpreters and records runtime identities, but does not load model weights. The three filters (`--route`, `--dataset`, `--seed`) can each be repeated; omitting a filter selects all configured values. Supply checkpoint and interpreter paths for every selected route before full-matrix execution.

Each run uses a distinct tag, for example `paper_llama31_native_clean_bbbp_seed42`. The pipeline preserves validation-based rule selection and the `all` branch's inference-rule fallback. The paper orchestrator resolves scientific settings from its configuration and isolates them from ambient overrides; CUDA and cache settings remain machine-local.

Run configuration, prompt snapshots, provenance, logs and completion records are stored under `artifacts/<run_tag>/`. Rule and response files use the same tag in their stage-specific directories. Audit a completed run with:

```bash
python -m morace.run_audit artifacts/paper_llama31_native_clean_bbbp_seed42
```

Existing run namespaces are not overwritten. `--resume` reuses only identity-matched, completed and audited runs; use a fresh `--run-prefix` for a new experiment. Independent route/dataset subsets can be assigned to separate shells with `CUDA_VISIBLE_DEVICES`, provided their run tags do not overlap. Custom environment settings and the generic launcher are covered in [run configuration](RUN_CONFIGURATION.md); the generic launcher has separate defaults and is not a substitute for the paper configuration.

## Evaluate Paired Features

Use the final selected rules of a successful run, not an intermediate candidate rule set:

```bash
python scripts/run_feature_experiments.py \
  --dataset bbbp --source artifacts/paper_llama31_native_clean_bbbp_seed42 \
  --seed-profile fifty --output-dir artifacts/paired_bbbp_seed42
```

Run-directory inputs require successful task status. If a degraded run contains a successful final branch, supply that branch's outcome or final rule JSON explicitly and retain its run-audit evidence. The runner resolves top-level final rules, including recorded inference-only fallback, and freshly evaluates five feature arms on identical rows with matched RF parameters and seeds:

| Arm | Representation |
| --- | --- |
| `base` | Fixed Morgan features |
| `rules` | Final rule-derived features only |
| `base_plus_rules` | The same Morgan prefix plus final rule features |
| `base_plus_shuffled_rules` | The same prefix plus shuffled rule columns |
| `base_plus_random_rules` | The same prefix plus random count-matched columns |

The `fifty` profile uses RF seeds **0-49** with unchanged rules and control matrices; the default `five` profile uses seeds **0-4**. These are estimator-initialization repeats, not independent rule generations or new splits. The manuscript's 63-cell RF/control analysis covers Qwen/Llama/Gemma x seven datasets x three rule-generation repeats, not the entire five-route generation matrix.

ROC-AUC improvement is candidate minus comparator; RMSE improvement is comparator minus candidate. Both are positive for better candidate prediction. Classification and regression deltas retain their own units and are not averaged together.

Outputs include `per_seed.csv`, `summary.csv`, `paired_deltas.csv`, `resolved_rules.json`, `protocol.json` and `status.json`. Protocol records retain the final source, target, split hashes, actual seeds, RF parameters, fingerprints, control settings and feature matrices.

Fingerprint settings, train-derived sanitation and control construction are fixed in [the feature protocol](FEATURE_EXPERIMENTS.md). Default control seeds are explicit new noise realizations. To replay a historical control cell, use `--historical-control-key ROUTE REPEAT` with its recorded labels and exact final rules; a generation route ID is not automatically the historical control label.

## Check Raw-Draft Closure

Evaluate saved individual candidate records through the frozen operational projection table:

```bash
python scripts/run_raw_closure.py \
  --candidates /path/to/saved_candidates.jsonl \
  --molecules data/splits/bace/bace_train.csv --smiles-column mol \
  --output-dir reports/raw_closure_bace
```

This CPU analysis records accepted and rejected candidates, compiled rules, molecule-level execution and nonconstant status. It uses no target labels, LLM calls, generated Python code or RF fitting. The BACE training split above is an example molecule set; a historical closure result requires its recorded molecule pool.

`--input-mode stage1` accepts saved draft responses or stage artifacts. A whole response remains one candidate unit and is not automatically split into the individual claims used by the manuscript's raw-candidate analysis. Preserve the original extraction units and identifiers when comparing denominators.

See [raw closure](RAW_CLOSURE.md) for the candidate schema, frozen mapping, rejection categories and provenance fields. An executable nonconstant feature establishes operational closure, not a causal chemical mechanism or predictive gain.

## Summarize and Compare Results

Summarize completed generation runs and paired feature outputs in their separate tables:

```bash
python scripts/summarize_experiments.py \
  --plan reports/llama_bbbp_plan.json \
  --feature-dir artifacts/paired_bbbp_seed42 \
  --output-dir reports/llama_bbbp_results
```

The plan supplies the requested run denominator. Missing, failed and incomplete results remain visible rather than becoming zero scores. Main results first average recorded RF scores within each generation run, then summarize available runs; generation seeds and RF seeds are not pooled into one repetition count.

For a matched comparison, supply real completed comparator records explicitly:

```bash
python scripts/summarize_experiments.py \
  --plan reports/llama_bbbp_plan.json \
  --comparator /path/to/completed_matched_baseline.csv \
  --output-dir reports/llama_bbbp_matched
```

Comparator columns are `dataset,route,seed,metric,score,status`, where `seed` identifies the matched generation/run seed. The manuscript's matched LLM4SD-style comparison is limited to Qwen/Llama/Gemma (21 route-dataset cells). DeepSeek and ChemDFM belong to the wider generation analysis and must not be added to that matched denominator. The comparator is the backend-aligned, no-GPT-4-code setting described in the manuscript, not the unrestricted published LLM4SD system.

The conventional ECFP entry point can check its dependencies and input plan without fitting:

```bash
python scripts/run_ecfp_baselines.py --check-deps \
  --dependency-json reports/ecfp_check/dependencies.json \
  --dependency-md reports/ecfp_check/dependencies.md
python scripts/run_ecfp_baselines.py \
  --manifest data/manifests/split_manifest.csv --dry-run \
  --dry-run-csv reports/ecfp_check/splits.csv \
  --dry-run-md reports/ecfp_check/splits.md
```

These checks write local reports without fitting models. This baseline runner has its own estimator and fingerprint-selection settings; see [ECFP baselines](ECFP_BASELINES.md).

The historical LLM4SD-style launcher and the Chemprop, Mole-BERT, MolCLR and Uni-Mol implementations used for wider baseline context are external to this checkout. Retain their source, locked inputs, preprocessing policy, budgets and completed results alongside the comparison. The reporting tool does not fetch baseline scores or fill missing results from the manuscript. Full input/output contracts are in [experiment reports](EXPERIMENT_REPORTS.md).

## Additional Diagnostics

Rule-level deletion, family-level deletion and feature importance use frozen final rules without regenerating them. Check the input inventory first:

```bash
python scripts/run_rule_contribution_analysis.py \
  --run-dir artifacts/paper_llama31_native_clean_bbbp_seed42 \
  --output-dir reports/contribution_bbbp --dry-run
```

Use [contribution analysis](CONTRIBUTION_ANALYSIS.md) for the subsequent diagnostic commands and interpretation. These analyses supplement the manuscript's evidence modules; feature importance and test deltas are not rule-selection criteria.

## Artifact Scope

This checkout contains implementation source, experiment entry points, configurations, a compiler example, tests and all seven fixed dataset split sets. It also includes the original method figure and a manuscript results summary. Checkpoints, generated responses, historical final-rule artifacts, RF/control records, machine-readable publication result tables and figure-generation workspaces are not bundled.

Replaying a source-bound paper result requires the corresponding checkpoint identity, prompt/decoding records, source rules, ordered split files, endpoint, RF/control settings and completed baseline artifacts. Software verification does not establish reproduction of every manuscript result. See [reproducibility](REPRODUCIBILITY.md) for input and execution-record requirements.

Record an explicit source commit when documenting an execution. See [versioning](PAPER_VERSION.md) and [software verification](ENGINEERING_VALIDATION.md).
