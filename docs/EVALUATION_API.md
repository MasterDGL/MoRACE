# Evaluation API

`EvaluationPipeline` holds an immutable `EvaluationConfig` for one rule-only evaluation. It does not generate rules or run the separate Morgan+rules experiment. For the full paper workflow, use [paper reproduction](PAPER_REPRODUCTION.md).

## Evaluate a Completed Generation

Run from the source checkout after the named generation, projection and rule-summary artifacts exist:

```python
from dataclasses import replace

from morace.evaluation_config import EvaluationConfig
from morace.evaluation_pipeline import EvaluationPipeline

config = EvaluationConfig(
    dataset="bbbp",
    dataset_root="data/splits",
    model="qwen3.5-9b",
    knowledge_type="all",
    run_tag="bbbp_seed42",
    generation_seed=42,
)

EvaluationPipeline(config).main()

second = replace(config, run_tag="bbbp_seed43", generation_seed=43)
EvaluationPipeline(second).main()
```

Both run tags require their own upstream artifacts. Changing a tag or seed does not regenerate rules. Named inputs cannot silently resolve to another run's untagged files.

## Configuration Boundary

Direct construction uses explicit defaults, not the caller's environment. `dataset_root` becomes absolute at construction; an empty `run_tag` retains the ungrouped legacy path layout. The default evaluator uses RF seeds 0-4 and dataset hyperparameters from `configs/model_registry.json`. For 50-seed Morgan+rules evaluation, use the [paired feature runner](FEATURE_EXPERIMENTS.md), not this configuration object.

`EvaluationConfig.from_namespace(namespace, environment=mapping)` accepts an explicit environment mapping. Omitting that mapping does not read `os.environ`; CLI adapters capture their environment once. Later environment changes do not change an existing instance.

Validation thresholds, coverage guards, finalist policies and aggregation switches are configuration fields. Direct API defaults are not a substitute for the paper configuration or a recorded run's resolved settings. Invalid values are reported during configuration parsing, not merely on module import.

Pure computation helpers remain importable from `morace.evaluation_pipeline`; configuration-dependent methods belong to an evaluator instance. `morace.ablation.evaluate_rule_variant(..., config=config)` can use the same configuration. Legacy diagnostic calls without an explicit configuration capture their environment at entry.

## Verification Scope

CPU regression tests cover configuration isolation, independent data roots, interleaved instances, validation-direction decisions, and fixed-feature classification/regression equivalence to the earlier evaluator. These checks do not establish arbitrary concurrent GPU execution or replace model-generation experiments. See [software verification](ENGINEERING_VALIDATION.md) for the check workflow.
