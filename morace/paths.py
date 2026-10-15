import os
from pathlib import Path

from morace.run_configuration import validate_run_tag


def sanitize_run_tag(run_tag: str) -> str:
    """Validate an output identity without mapping distinct names together."""
    if not run_tag:
        return ""
    return validate_run_tag(run_tag)


def resolve_run_tag(explicit_run_tag: str = "") -> str:
    if explicit_run_tag:
        return sanitize_run_tag(explicit_run_tag)
    env_run_tag = os.environ.get("MORACE_RUN_TAG") or os.environ.get("LLM4SD_RUN_TAG", "")
    return sanitize_run_tag(env_run_tag)


def with_run_tag(base_dir: str, run_tag: str, *, inherit_environment: bool = True) -> str:
    tag = resolve_run_tag(run_tag) if inherit_environment else sanitize_run_tag(run_tag)
    if not tag:
        return base_dir
    return os.path.join(base_dir, tag)


def resolve_artifact_path(
    primary_path: str, legacy_path: str, *, run_tag: str = "", inherit_environment: bool = True
) -> str:
    """A named run must never borrow an unscoped artifact from another run.

    Unscoped legacy use remains available by explicitly running without a tag.
    Missing named inputs fail at the boundary, before feature selection/training.
    """
    tag = resolve_run_tag(run_tag) if inherit_environment else sanitize_run_tag(run_tag)
    if Path(primary_path).is_file():
        return primary_path
    if tag:
        raise FileNotFoundError(
            f"Missing input for run {tag!r}: {primary_path}. "
            "Generate this run's preceding stage, or explicitly import existing artifacts."
        )
    if primary_path != legacy_path and Path(legacy_path).is_file():
        return legacy_path
    return primary_path
