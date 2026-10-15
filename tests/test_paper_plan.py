import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_plan_test", ROOT / "scripts/run_paper_experiments.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def configure(monkeypatch, tmp_path, *extra):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "datasets": ["bbbp", "bace"], "seeds": [42, 43], "common_environment": {},
        "routes": [{"id": "fixture_route", "model": "qwen3.5-9b",
                    "model_path_env": "MORACE_MODEL_PATH_FIXTURE", "python_path_env": "MORACE_PYTHON_FIXTURE",
                    "environment": {"STAGE2_PROTOCOL_PROFILE": "qwen_legacy_d2"}}],
    }
    config_path = tmp_path / "paper.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    plan_path = tmp_path / "plans" / "jobs.json"
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    monkeypatch.setattr(paper.sys, "argv", ["paper", "--config", str(config_path),
        "--model-path", f"fixture_route={checkpoint}", "--python-path", f"fixture_route={sys.executable}",
        "--plan-output", str(plan_path), *extra])
    return plan_path


def read_plan(path):
    return json.loads(path.read_text(encoding="utf-8"))["jobs"]


@pytest.mark.parametrize("failure", ["preflight", "first_job"])
def test_full_plan_is_saved_before_first_failure(monkeypatch, tmp_path, failure):
    plan_path = configure(monkeypatch, tmp_path, "--execute")
    calls = []
    def preflight(*args):
        assert len(read_plan(plan_path)) == 4
        calls.append("preflight")
        if failure == "preflight":
            raise RuntimeError("preflight fixture failed")
    def execute(command, **kwargs):
        jobs = read_plan(plan_path)
        assert len(jobs) == 4
        assert kwargs["env"]["RUN_TAG"] == jobs[0]["run_tag"]
        assert kwargs["env"]["MORACE_EXPECTED_IDENTITY"] == jobs[0]["identity"]["fingerprint"]
        calls.append("execute")
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(paper, "run_route_preflight", preflight)
    monkeypatch.setattr(paper.subprocess, "run", execute)
    if failure == "preflight":
        with pytest.raises(RuntimeError, match="preflight fixture failed"):
            paper.main()
        assert calls == ["preflight"]
    else:
        assert paper.main() == 7
        assert calls == ["preflight", "execute"]
    assert [(job["dataset"], job["seed"]) for job in read_plan(plan_path)] == [
        ("bbbp", 42), ("bbbp", 43), ("bace", 42), ("bace", 43)]


@pytest.mark.parametrize("prefix", [None, "replication2"])
def test_run_prefix_changes_identity_paths_but_keeps_default_protocol(monkeypatch, tmp_path, prefix):
    plan_path = configure(monkeypatch, tmp_path, *(("--run-prefix", prefix) if prefix else ()))
    assert paper.main() == 0
    jobs = read_plan(plan_path)
    expected_prefix = prefix or "paper"
    assert jobs[0]["run_tag"] == f"{expected_prefix}_fixture_route_bbbp_seed42"
    assert jobs[-1]["run_tag"] == f"{expected_prefix}_fixture_route_bace_seed43"
    assert jobs[0]["identity"]["payload"]["environment"]["STAGE2_PROTOCOL_PROFILE"] == "qwen_legacy_d2"
    assert jobs[0]["identity"]["payload"]["environment"]["GLOBAL_SEED"] == "42"


@pytest.mark.parametrize("prefix", ["../escape", "trial one", "p" * 115])
def test_invalid_or_overlong_final_run_tag_fails_before_plan_or_execution(monkeypatch, tmp_path, prefix):
    plan_path = configure(monkeypatch, tmp_path, "--run-prefix", prefix, "--execute")
    monkeypatch.setattr(paper, "run_route_preflight", lambda *args: pytest.fail("must validate before preflight"))
    with pytest.raises(ValueError, match="run_tag"):
        paper.main()
    assert not plan_path.exists()
    assert not (tmp_path / "artifacts").exists()


def test_preflight_only_writes_full_plan_without_executing_jobs(monkeypatch, tmp_path):
    plan_path = configure(monkeypatch, tmp_path, "--preflight-only")
    calls = []
    def preflight(*args):
        assert len(read_plan(plan_path)) == 4
        calls.append(args[0]["id"])
    monkeypatch.setattr(paper, "run_route_preflight", preflight)
    monkeypatch.setattr(paper.subprocess, "run", lambda *args, **kwargs: pytest.fail("must not launch jobs"))
    assert paper.main() == 0
    assert calls == ["fixture_route"]
