from pathlib import Path
from types import SimpleNamespace

import pytest

from morace import evaluation_pipeline, rule_summary
from morace.paths import resolve_artifact_path, with_run_tag


@pytest.mark.parametrize("kind", ["evaluation", "summary"])
def test_named_run_cannot_read_legacy_input(kind, tmp_path, monkeypatch):
    legacy = tmp_path / "old.json"
    legacy.write_text("{}", encoding="utf-8")
    primary = tmp_path / "new-run" / "input.json"
    if kind == "evaluation":
        resolve = evaluation_pipeline.EvaluationPipeline(
            evaluation_pipeline.EvaluationConfig(run_tag="new-run")
        ).resolve_existing_path
    else:
        config = rule_summary.RuleSummaryConfig(run_tag="new-run")
        resolve = lambda primary, legacy: rule_summary.resolve_existing_path(primary, legacy, config)
    with pytest.raises(FileNotFoundError, match="new-run"):
        resolve(str(primary), str(legacy))
    primary.parent.mkdir()
    primary.write_text('{"run_tag":"new-run"}', encoding="utf-8")
    assert resolve(str(primary), str(legacy)) == str(primary)


def test_environment_named_run_is_also_strict(tmp_path, monkeypatch):
    legacy = tmp_path / "old.json"
    legacy.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MORACE_RUN_TAG", "from-env")
    with pytest.raises(FileNotFoundError, match="from-env"):
        resolve_artifact_path(str(tmp_path / "missing.json"), str(legacy))


def test_explicit_unscoped_legacy_use(tmp_path, monkeypatch):
    monkeypatch.delenv("MORACE_RUN_TAG", raising=False)
    monkeypatch.delenv("LLM4SD_RUN_TAG", raising=False)
    legacy = tmp_path / "legacy.json"
    legacy.write_text("{}", encoding="utf-8")
    assert resolve_artifact_path(str(tmp_path / "missing.json"), str(legacy)) == str(legacy)


@pytest.mark.parametrize("tag", ["trial 1", "../run", "a/b", "a\\b", " trial", "."])
def test_run_names_do_not_collide_by_sanitization(tag):
    with pytest.raises(ValueError):
        with_run_tag("outputs", tag)
