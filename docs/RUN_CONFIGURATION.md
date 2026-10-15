# Run Configuration and Resume

Run settings are resolved by [`morace.run_configuration`](../morace/run_configuration.py). Each run captures its scientific parameters, input identities, prompt snapshots and completion evidence without changing the underlying generation, selection or evaluation method.

## Choose an Entry Point

Use `scripts/run_paper_experiments.py` for the configured paper matrix. It combines `DEFAULT_ENVIRONMENT`, the configuration's `common_environment`, route-specific settings, and selected checkpoint/interpreter paths and generation seed. Known ambient method controls are cleared; machine settings such as CUDA visibility, caches and proxies remain available.

Use `scripts/run_pipeline.sh DATASET` for custom single-run work. It resolves defaults plus explicit environment overrides. For example:

```bash
MODEL=llama-3.1-8b-instruct MODEL_PATH=/models/Meta-Llama-3.1-8B-Instruct \
PYTHON_BIN=/envs/llama/bin/python RUN_TAG=custom_llama_bbbp_seed42 \
GLOBAL_SEED=42 bash scripts/run_pipeline.sh bbbp
```

These entry points have different defaults. The paper matrix uses three synthesis and three inference drafts per prompt, keeps up to two late-selection survivors, and permits a score delta of 0.30. The generic launcher defaults to one draft per stage, one survivor and a score delta of 0.0. Route-specific projection settings also matter; a generic launch is not automatically the paper protocol.

For a new protocol, copy the paper configuration, pass `--config`, and choose a new `--run-prefix`. Checkpoint and interpreter overrides use `--model-path ROUTE=PATH` and `--python-path ROUTE=PATH`, or the route's declared path variables. Unknown method-variable names and unsupported values fail during resolution.

## Preview, Preflight and Execution

- With no execution flags or `--plan-output`, the paper runner only previews the matrix; it does not probe interpreters or load models.
- `--plan-output reports/my_plan.json` saves the complete requested matrix and probes selected interpreters for runtime identity, without loading model weights.
- `--preflight-only` verifies runtimes and checkpoint files without launching generation. It can be combined with `--plan-output`.
- `--execute` preflights and runs the matrix. Plans are saved before preflight or execution when requested.

Use identical configuration, paths and source for planning and execution. A preview with missing checkpoint paths is not an executable reproduction plan. Full commands are in [paper reproduction](PAPER_REPRODUCTION.md#generate-and-audit-rules).

## Run Records

```text
artifacts/<run_tag>/
  run_config.json
  prompts/
  logs/
  summary.json
  audit.json
  completion.json
```

`run_config.json` captures resolved parameters, expected tasks, data provenance and identity hashes. Identity includes the selected interpreter/version/platform, 12 package versions, scientific settings, and relevant data, prompt, checkpoint-manifest, source and RF-parameter hashes. The actual execution interpreter checks that the planned identity still matches.

Responses and rules remain in their run-tagged stage directories, such as `synthesize_model_response/<run_tag>/` and `summarized_inference_rules/<run_tag>/`. A named run never silently falls back to untagged historical inputs. Final artifact hashes are retained by the audit.

For custom launches, `SKIP_PROMPT_REGEN=1` copies existing prompts from `MORACE_PROMPT_SOURCE` into the new run; it does not share mutable prompt files. Run tags are 1-128 ASCII letters, digits, dots, underscores or hyphens, beginning with a letter or digit.

## Resume and Status

```bash
python scripts/run_paper_experiments.py \
  --route llama31_native_clean --dataset bbbp --seed 42 \
  --model-path llama31_native_clean=/models/Meta-Llama-3.1-8B-Instruct \
  --python-path llama31_native_clean=/envs/llama/bin/python \
  --execute --resume
```

Resume skips only an identity-matched completed run whose task set, prompt snapshots, completion marker and artifact audit agree. Audited `degraded`, `partial` or `skipped` scientific outcomes may be complete; a process failure or an incomplete directory is not reusable. To retry failed work or change settings, choose a new `--run-prefix` or `RUN_TAG`.

The generic launcher returns **0** for fully successful audited work and **20** for audited, completed non-success scientific outcomes. Other failures return nonzero. The paper runner continues past valid completed non-success outcomes and returns **0** if the matrix completes without process failures; inspect the recorded statuses rather than interpreting that exit code as universal scientific success.

Audit a run with `python -m morace.run_audit artifacts/<run_tag>`. Add `--require-success` to require successful scientific status as well as valid evidence. Report tools preserve measured scores and statuses separately; see [experiment reports](EXPERIMENT_REPORTS.md).

For Python calls, use immutable stage configurations and independent `EvaluationPipeline` instances instead of module-global arguments. See the [evaluation API](EVALUATION_API.md).
