"""Runtime metadata must bind a completed experiment even at an unchanged path."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from morace import run_configuration as config


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("runtime_identity_paper", ROOT / "scripts/run_paper_experiments.py")
paper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(paper)


def test_current_runtime_matches_standalone_probe_without_model_imports():
    expected = config.current_runtime_snapshot()
    command = config.RUNTIME_PROBE_SOURCE + "\nassert not {'torch', 'transformers', 'numpy', 'rdkit'} & set(sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", command], capture_output=True,
                            text=True, encoding="utf-8", check=True)
    assert json.loads(result.stdout) == expected
    assert expected["python_version"] == sys.version


def test_selected_route_interpreter_is_probed_instead_of_planner(monkeypatch, tmp_path):
    selected = str(tmp_path / "another_env" / "python")
    expected = config.current_runtime_snapshot()
    expected.update(python_executable=selected, python_version="3.99 target interpreter")
    expected["packages"]["transformers"] = "target-only-version"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps(expected))

    monkeypatch.setattr(config.subprocess, "run", run)
    environment = {"PYTHONPATH": "route-specific-path"}
    actual = config.probe_runtime(selected, environment=environment, cwd=tmp_path)
    assert actual == expected
    assert len(calls) == 1
    assert calls[0][0][:2] == [selected, "-c"]
    assert calls[0][1]["env"] == environment
    assert calls[0][1]["cwd"] == tmp_path


@pytest.mark.parametrize("component", ["torch", "transformers", "scikit-learn", "rdkit", "python", "platform"])
def test_same_interpreter_path_environment_change_invalidates_resume(monkeypatch, tmp_path, component):
    environment = {"RUN_TAG": "fixture", "PYTHON_BIN": sys.executable}
    resolved = config.prepare_run(tmp_path, environment, "bbbp")
    artifact = tmp_path / "artifacts/fixture"
    previous = json.loads((artifact / "run_config.json").read_text(encoding="utf-8"))["identity"]
    # A completion marker is enough here: the identity gate must reject the run
    # before artifact auditing, which is covered by the complete-run tests.
    config.write_json(artifact / "completion.json", {})
    changed = config.current_runtime_snapshot()
    if component == "python":
        changed["python_version"] += " upgraded"
    elif component == "platform":
        changed["platform"] += " changed"
    else:
        changed["packages"][component] = "999.0-upgraded-in-place"
    monkeypatch.setattr(config, "current_runtime_snapshot", lambda: changed)
    requested = config.build_identity(tmp_path, resolved, "bbbp")
    assert previous["payload"]["python_executable"] == requested["payload"]["python_executable"]
    assert previous["fingerprint"] != requested["fingerprint"]
    assert config.reusable_run(artifact, requested, root=tmp_path, run_tag="fixture") == (
        False, ["run identity differs from the requested experiment"])


def test_prepare_reprobes_actual_runtime_and_rejects_changed_plan_before_reservation(monkeypatch, tmp_path):
    environment = {"RUN_TAG": "fixture", "PYTHON_BIN": sys.executable}
    planned = config.build_identity(tmp_path, environment, "bbbp")
    actual = config.current_runtime_snapshot()
    actual["packages"]["numpy"] = "new-version-after-plan"
    monkeypatch.setattr(config, "current_runtime_snapshot", lambda: actual)
    with pytest.raises(ValueError, match="runtime changed"):
        config.prepare_run(tmp_path, {**environment, "MORACE_EXPECTED_IDENTITY": planned["fingerprint"]}, "bbbp")
    assert not (tmp_path / "artifacts").exists()


def test_prepare_records_exact_runtime_used_in_identity(tmp_path):
    config.prepare_run(tmp_path, {"RUN_TAG": "fixture"}, "bbbp")
    saved = json.loads((tmp_path / "artifacts/fixture/run_config.json").read_text(encoding="utf-8"))
    assert saved["runtime"] == saved["identity"]["payload"]["runtime"]
    assert saved["runtime"]["status"] == "available"


def setup_plan(monkeypatch, tmp_path, *, execute=False, missing=False):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    routes = [
        {"id": route, "model": "qwen3.5-9b", "model_path_env": f"MORACE_MODEL_PATH_{route.upper()}",
         "python_path_env": f"MORACE_PYTHON_{route.upper()}",
         "environment": {"STAGE2_PROTOCOL_PROFILE": "qwen_legacy_d2"}}
        for route in ("one", "two")
    ]
    configuration = {"datasets": ["bbbp", "bace"], "seeds": [42, 43],
                     "common_environment": {}, "routes": routes}
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(configuration), encoding="utf-8")
    output = tmp_path / "plan.json"
    argv = ["paper", "--config", str(path), "--plan-output", str(output)]
    for route in routes:
        selected = str(tmp_path / "missing" / "python") if missing else sys.executable
        argv += ["--python-path", f"{route['id']}={selected}",
                 "--model-path", f"{route['id']}={checkpoint}"]
    if execute:
        argv.append("--execute")
    monkeypatch.setattr(paper, "ROOT", tmp_path)
    monkeypatch.setattr(paper.sys, "argv", argv)
    return output


def test_planner_probes_once_per_route_not_per_job(monkeypatch, tmp_path):
    output = setup_plan(monkeypatch, tmp_path)
    calls = []
    original = paper.probe_runtime

    def probe(selected, **kwargs):
        calls.append((selected, kwargs["environment"]["RUN_TAG"]))
        return original(selected, **kwargs)

    monkeypatch.setattr(paper, "probe_runtime", probe)
    assert paper.main() == 0
    jobs = json.loads(output.read_text(encoding="utf-8"))["jobs"]
    assert len(jobs) == 8
    assert len(calls) == 2
    assert all(job["identity"]["payload"]["runtime"]["status"] == "available" for job in jobs)
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("execute", [False, True])
def test_unavailable_runtime_is_visible_in_dry_plan_and_rejected_for_execution(monkeypatch, tmp_path, execute):
    output = setup_plan(monkeypatch, tmp_path, execute=execute, missing=True)
    monkeypatch.setattr(paper, "run_route_preflight", lambda *args: pytest.fail("must reject unavailable runtime first"))
    if execute:
        with pytest.raises(RuntimeError, match="selected Python runtime is unavailable"):
            paper.main()
    else:
        assert paper.main() == 0
    jobs = json.loads(output.read_text(encoding="utf-8"))["jobs"]
    assert len(jobs) == 8
    assert all(job["identity"]["payload"]["runtime"]["status"] == "unavailable" for job in jobs)
    assert not (tmp_path / "artifacts").exists()


def test_strict_probe_reports_unavailable_interpreter(tmp_path):
    with pytest.raises(RuntimeError, match="cannot probe selected Python"):
        config.probe_runtime(str(tmp_path / "missing" / "python"), allow_unavailable=False)
