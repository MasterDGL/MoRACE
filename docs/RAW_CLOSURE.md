# Raw-Draft Operational Closure

`scripts/run_raw_closure.py` maps saved molecular claims through the frozen P7c3
matching table, compiles accepted claims with the MoRACE rule compiler, and checks
feature execution and variation on a supplied molecule CSV. It runs on CPU and
uses no target labels, LLM calls, generated Python code, or random-forest fitting.

```bash
python scripts/run_raw_closure.py \
  --candidates saved_candidates.jsonl \
  --molecules data/splits/bace/bace_train.csv \
  --smiles-column mol \
  --output-dir reports/raw_closure_bace
```

## Candidate Input

CSV, JSONL and JSON lists accept these fields:

```json
{"raw_candidate_id":"claim-1","natural_language_rule_text":"Molecular weight increases","feature_hint":"molecular weight","direction_hint":"positive","rationale_short":"","extraction_success":true}
```

`raw_candidate_id` is retained. `candidate_id`, `source_candidate_id` and `draft_id`
are also supported as source identifiers. `source_record_index` always preserves
the one-based input row position, including when source IDs repeat. Missing
identifiers receive a positional ID. Group summaries use `model_id`, `dataset`
and `seed`; no label column is selected.

`extraction_success` defaults to true for supplied candidate records. Existing
false records remain in the input denominator and are reported as extraction
failures. JSON accepts booleans; CSV uses lowercase `true`/`false`. Invalid text
or extraction-status fields remain recorded as extraction failures. Malformed
JSON causes an input error instead of silently removing rows.

## Saved Stage-1 Records

Use `--input-mode stage1` for records with `draft_text` (or `text`). The command
also accepts an existing synthesis/inference artifact containing `stage1_drafts`.

```bash
python scripts/run_raw_closure.py \
  --input-mode stage1 \
  --candidates saved_synthesis_artifact.json \
  --molecules data/splits/bace/bace_train.csv \
  --smiles-column mol \
  --output-dir reports/stage1_closure_bace
```

Each supplied record is one candidate unit. A whole Stage-1 response is not split
into sentences or determinants. If it mentions several supported features, the
first matching entry in the frozen table wins. Whole-response rates therefore
have a different denominator from rates based on previously extracted individual
claims. Export individual claims as candidate records when that is the intended
unit of analysis.

## Frozen Semantics

The 17 ordered regular expressions and the projection acceptance/rejection logic
come from `p7c3_runner/scripts/p7c3_common.py` and
`p7c3_run_schema_normalization.py`. Compilation/execution accounting comes from
`p7c3_run_compile_and_execution.py`. Original source hashes are embedded in the
module and exported in provenance. The version is
`p7c3_operational_projection_v1`; its mapping and historical alias choices have
not been expanded or chemically reinterpreted.

Projection checks the claim text, feature hint and short rationale. It keeps the
original first-match priority, task-only rejection, directional-conflict rejection
and direction extraction. Nonconstant means at least two finite values differing
by more than `1e-12`. All supplied molecule rows are retained. `valid_value_count`
counts finite feature values, which is distinct from the number of valid SMILES;
invalid-SMILES behavior remains that of the existing compiler.

This entry point is an independent operational-closure diagnostic. The main
MoRACE pipeline continues to use its model-specific LLM projection mechanisms.
Historical P7c3 execution pooled its recorded train/validation/test molecules;
this command evaluates the explicit molecule CSV supplied by the caller.

## Outputs and Interpretation

The command creates a new directory and refuses to overwrite an existing one:

| File | Content |
| --- | --- |
| `candidates.jsonl` | Source fields, projection match/rejection, normalized rule, compilation and execution status, finite counts and nonconstant status |
| `candidates.csv` | The same candidate-level records for inspection |
| `summary.json` | Input and extracted-candidate denominators, stage counts, rejection counts, model/dataset/seed groups, and input/mapping/compiler hashes |

Repeated candidates remain repeated observations. The summary separately reports
the number of distinct compiled features. Acceptance, compilation, execution and
nonconstant rates use the explicitly named extracted-candidate denominator;
extraction failures are also retained in the input-record count. Zero denominators
produce null rates.

A nonconstant executable feature establishes operational closure on the supplied
molecules. Its predictive value requires a separate prediction experiment.
