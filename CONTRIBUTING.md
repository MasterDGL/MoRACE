# Contributing

Keep changes focused on MoRACE's rule-to-feature workflow and preserve the recorded experimental protocol. See [versioning](docs/PAPER_VERSION.md) for source and protocol identification.

## Checks

Use Python 3.10 or 3.11. For source changes, run the relevant regression tests and the standard checks from the checkout:

```bash
python -m pip install -r requirements-dev.txt
python -m pip install -e .
python scripts/verify_repository.py
python -m pytest
python -m compileall -q morace scripts tests examples
bash -n scripts/run_pipeline.sh scripts/setup_environment.sh
```

For documentation-only changes, check links, command syntax and any changed examples. Do not rerun model experiments merely for prose edits. GitHub CI separately checks both supported Python versions.

## Scope and Compatibility

Reusable code belongs in `morace/`, experiment entry points in `scripts/`, and focused regression tests in `tests/`. Reuse the existing terminology and module boundaries.

A cleanup must not silently change rule parsing, projection, canonicalization, compilation, train-derived sanitation, validation-based selection, seeds or RF settings. Method changes use a separate configuration and run identity, with the earlier validated behavior retained. Use appropriate equivalence checks and the [software verification workflow](docs/ENGINEERING_VALIDATION.md).

Keep generated responses, final research artifacts, metrics, logs, checkpoints and rerun workspaces out of Git. The approved manuscript method figure and documented manuscript-result summary are documentation exceptions; the full manuscript and generated result figures are not bundled.

Included split files are fixed inputs. New data or endpoint changes require an explicit protocol and a matching manifest with paths, counts, columns and hashes. Do not regenerate or replace paper splits as part of a cleanup.

Validate changes on a separate branch before merging into the official repository.
