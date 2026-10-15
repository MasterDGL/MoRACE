# Versioning

[MasterDGL/MoRACE](https://github.com/MasterDGL/MoRACE) is the official source repository.

Record the exact source commit, configuration, checkpoint identity and input provenance for each experiment. The Python package version is defined in `pyproject.toml`; a package version alone does not identify an experimental protocol.

## Development Workflow

Use a separate branch for changes and run the relevant checks before merging. See [contributing](../CONTRIBUTING.md) for the verification workflow.

Behavior-preserving refactors must retain rule semantics, prompts, selection policies, metrics and defaults, with appropriate equivalence checks. Method changes require an independent configuration and run namespace.

A source checkout contains tracked code, configurations and dataset inputs, not ignored model weights or experiment outputs. See [contributing](../CONTRIBUTING.md), [paper reproduction](PAPER_REPRODUCTION.md) and [repository contents](RELEASE_SCOPE.md).
